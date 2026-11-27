//! Exact photon-number-conserving (fixed-N) Fock-space simulator.
//!
//! Represents an `m`-mode, `N`-photon state in the `C(m + N - 1, N)`-dimensional
//! Fock subspace with zero truncation error. Supports all photon-number-conserving
//! operations: `Rotation`, `BeamSplitter`, `Passive`, `Kerr`, `CrossKerr`, plus
//! destructive photon-counting and threshold measurements with feed-forward.
//!
//! Basis layout: recursive colexicographic partitioning by the occupancy
//! `p = n_{m-1}` of the highest mode (`p = 0, 1, ..., N`), where block `p`
//! holds all `(m - 1)`-mode states of `N - p` photons contiguously.
//!
//! Key algorithms:
//! - Fused passive prefix -> recursive cache-optimal SLOS pass (Heurtel et al.
//!   2023) with zero index lookup tables, contiguous SIMD SAXPY blocks, and
//!   lock-free rayon parallelism across disjoint output blocks.
//! - Diagonal `Rotation`, `Kerr`, and `CrossKerr` in `O(D)` flops.
//! - Block-sparse `BeamSplitter` decomposed into `(s + 1) × (s + 1)` exact
//!   unitary blocks computed by recurrence; mid-circuit `Passive` blocks via
//!   exact Clements/Reck Givens decomposition into beam splitters + rotations.

use super::circuit::{Measurement, ModeCircuit, ModeOp};
use super::gate::{ModeGate, TwoModeGate};
use super::state::ModeState;
use crate::error::Error;
use num_complex::Complex64 as C64;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

/// Minimum output dimension to fork rayon tasks inside the recursive SLOS pass.
const SLOS_PAR_MIN_DIM: usize = 4096;

/// Minimum basis dimension for parallelising diagonal / two-mode updates.
const GATE_PAR_MIN_DIM: usize = 8192;

/// Minimum shot count for parallelising multi-shot sampling.
const SAMPLE_PAR_MIN_SHOTS: usize = 256;

/// Precomputed table of `D(m, k) = C(m + k - 1, k)` (number of `m`-mode states
/// with `k` photons).
#[derive(Clone, Debug)]
struct DimTable {
    max_m: usize,
    max_n: usize,
    /// `table[m * (max_n + 1) + k] = D(m, k)`.
    table: Vec<usize>,
}

impl DimTable {
    fn new(max_m: usize, max_n: usize) -> Self {
        let stride = max_n + 1;
        let mut table = vec![0usize; (max_m + 1) * stride];
        for m in 0..=max_m {
            table[m * stride] = 1; // D(m, 0) = 1
        }
        for m in 1..=max_m {
            for k in 1..=max_n {
                table[m * stride + k] = table[(m - 1) * stride + k] + table[m * stride + (k - 1)];
            }
        }
        DimTable {
            max_m,
            max_n,
            table,
        }
    }

    #[inline(always)]
    fn dim(&self, m: usize, k: usize) -> usize {
        debug_assert!(m <= self.max_m && k <= self.max_n);
        self.table[m * (self.max_n + 1) + k]
    }
}

/// Exact `m`-mode, `N`-photon Fock statevector in the `C(m + N - 1, N)` basis.
#[derive(Clone, Debug)]
pub struct FixedNFockState {
    n_modes: usize,
    n_photons: usize,
    dims: DimTable,
    amps: Vec<C64>,
}

impl FixedNFockState {
    /// Dimension `C(m + N - 1, N)` of the `m`-mode `N`-photon Fock space.
    pub fn basis_dim(n_modes: usize, n_photons: usize) -> usize {
        if n_photons == 0 {
            return 1;
        }
        if n_modes == 0 {
            return 0;
        }
        let n = n_modes + n_photons - 1;
        let k = n_photons.min(n_modes - 1);
        let mut acc: u128 = 1;
        for i in 0..k {
            acc = acc * ((n - i) as u128) / ((i + 1) as u128);
        }
        acc as usize
    }

    /// Construct the single Fock basis state `|pattern[0], …, pattern[m−1]⟩`.
    pub fn from_fock(pattern: &[usize]) -> Result<Self, Error> {
        let n_modes = pattern.len();
        if n_modes == 0 {
            return Err(Error::InvalidArgument(
                "FixedNFockState requires at least one mode".into(),
            ));
        }
        let n_photons: usize = pattern.iter().sum();
        let dims = DimTable::new(n_modes, n_photons);
        let total = dims.dim(n_modes, n_photons);
        let mut amps = vec![C64::new(0.0, 0.0); total];
        let idx = pattern_index(&dims, pattern);
        amps[idx] = C64::new(1.0, 0.0);
        Ok(FixedNFockState {
            n_modes,
            n_photons,
            dims,
            amps,
        })
    }

    /// Run a bound, measurement-free, photon-number-conserving circuit whose
    /// inputs are Fock states.
    ///
    /// Automatically fuses the maximal initial passive prefix (rotations, beam
    /// splitters, and passive blocks) into one `m × m` unitary and evaluates it
    /// via a single cache-optimal recursive SLOS pass.
    pub fn from_circuit(circuit: &ModeCircuit) -> Result<Self, Error> {
        if circuit.n_parameters() > 0 {
            return Err(Error::InvalidArgument(
                "circuit has unbound parameters; bind it first".into(),
            ));
        }
        if !circuit.is_unitary() {
            return Err(Error::InvalidArgument(
                "circuit has measurements; use FixedNFockState::run or FixedNFockState::sample"
                    .into(),
            ));
        }
        let input = extract_fock_inputs(circuit.inputs())?;
        Self::run_unitary_ops(&input, circuit.ops())
    }

    /// Run a single shot of a bound photon-number-conserving circuit with
    /// photon-counting / threshold measurements and feed-forward.
    pub fn run<R: Rng + ?Sized>(
        circuit: &ModeCircuit,
        rng: &mut R,
    ) -> Result<(Self, Vec<f64>), Error> {
        if circuit.n_parameters() > 0 {
            return Err(Error::InvalidArgument(
                "circuit has unbound parameters; bind it first".into(),
            ));
        }
        let input = extract_fock_inputs(circuit.inputs())?;
        let ops = circuit.ops();
        let first_meas = ops
            .iter()
            .position(|op| matches!(op, ModeOp::Measure { .. }))
            .unwrap_or(ops.len());
        let mut state = Self::run_unitary_ops(&input, &ops[..first_meas])?;
        let mut results = Vec::with_capacity(circuit.n_results());
        state.run_ops(&ops[first_meas..], &mut results, rng)?;
        Ok((state, results))
    }

    /// Sample `shots` independent measurement outcomes from a bound
    /// photon-number-conserving circuit.
    ///
    /// Runs the unitary prefix before the first measurement once (using fused
    /// SLOS for passive prefixes). When all measurements are terminal PNR /
    /// threshold measurements on distinct modes, draws all shots directly from
    /// the precomputed probability CDF in `O(log D)` per shot.
    pub fn sample<R: Rng + ?Sized>(
        circuit: &ModeCircuit,
        shots: usize,
        rng: &mut R,
    ) -> Result<Vec<Vec<f64>>, Error> {
        if circuit.n_parameters() > 0 {
            return Err(Error::InvalidArgument(
                "circuit has unbound parameters; bind it first".into(),
            ));
        }
        let input = extract_fock_inputs(circuit.inputs())?;
        let ops = circuit.ops();
        // Validate all ops up front.
        for op in ops {
            match op {
                ModeOp::Measure { measurement, .. } => {
                    if !matches!(
                        measurement,
                        Measurement::PhotonCount | Measurement::Threshold
                    ) {
                        return Err(Error::InvalidArgument(
                            "FixedNFockState supports photon-counting and threshold measurements; \
                             use FockState for homodyne/heterodyne"
                                .into(),
                        ));
                    }
                }
                op if !op.conserves_photons() => {
                    return Err(Error::InvalidArgument(format!(
                        "operation {op:?} does not conserve photon number"
                    )));
                }
                _ => {}
            }
        }
        if shots == 0 {
            return Ok(Vec::new());
        }

        let first_meas = ops
            .iter()
            .position(|op| matches!(op, ModeOp::Measure { .. }))
            .unwrap_or(ops.len());
        let prefix = Self::run_unitary_ops(&input, &ops[..first_meas])?;
        let suffix = &ops[first_meas..];
        if suffix.is_empty() {
            return Ok(vec![Vec::new(); shots]);
        }

        // Fast path: all suffix ops are terminal PNR / threshold measurements on
        // distinct modes -> sample basis states directly from one CDF!
        if let Some(meas_spec) = terminal_pnr_plan(suffix, prefix.n_modes) {
            return Ok(prefix.sample_terminal(&meas_spec, shots, rng));
        }

        // General mid-circuit feed-forward path.
        let n_res = circuit.n_results();
        if shots >= SAMPLE_PAR_MIN_SHOTS {
            let seeds: Vec<u64> = (0..shots).map(|_| rng.random()).collect();
            seeds
                .into_par_iter()
                .map(|seed| {
                    let mut local_rng = StdRng::seed_from_u64(seed);
                    let mut st = prefix.clone();
                    let mut res = Vec::with_capacity(n_res);
                    st.run_ops(suffix, &mut res, &mut local_rng)?;
                    Ok(res)
                })
                .collect()
        } else {
            let mut out = Vec::with_capacity(shots);
            for _ in 0..shots {
                let mut st = prefix.clone();
                let mut res = Vec::with_capacity(n_res);
                st.run_ops(suffix, &mut res, rng)?;
                out.push(res);
            }
            Ok(out)
        }
    }

    /// Execute a sequence of unitary number-conserving ops starting from Fock
    /// input `input`, fusing any leading passive ops into a single SLOS pass.
    fn run_unitary_ops(input: &[usize], ops: &[ModeOp]) -> Result<Self, Error> {
        let m = input.len();
        if m == 0 {
            return Err(Error::InvalidArgument(
                "FixedNFockState requires at least one mode".into(),
            ));
        }

        // Find maximal leading passive prefix.
        let passive_end = ops
            .iter()
            .position(|op| !op.is_passive())
            .unwrap_or(ops.len());

        let mut state = if passive_end > 0 {
            let mut u = vec![C64::new(0.0, 0.0); m * m];
            for i in 0..m {
                u[i * m + i] = C64::new(1.0, 0.0);
            }
            for op in &ops[..passive_end] {
                fuse_passive_op(&mut u, m, op)?;
            }
            Self::from_passive_unitary(input, &u)?
        } else {
            Self::from_fock(input)?
        };

        for op in &ops[passive_end..] {
            state.apply_unitary_op(op, &[])?;
        }
        Ok(state)
    }

    /// Compute `U |input⟩` for an `m × m` row-major unitary `U` and Fock input
    /// `input` via the recursive cache-optimal SLOS pass.
    pub fn from_passive_unitary(input: &[usize], unitary: &[C64]) -> Result<Self, Error> {
        let m = input.len();
        if m == 0 {
            return Err(Error::InvalidArgument(
                "FixedNFockState requires at least one mode".into(),
            ));
        }
        if unitary.len() != m * m {
            return Err(Error::InvalidArgument(format!(
                "unitary on {m} modes needs {} entries, got {}",
                m * m,
                unitary.len()
            )));
        }
        let n_photons: usize = input.iter().sum();
        let dims = DimTable::new(m, n_photons);
        if n_photons == 0 {
            return Ok(FixedNFockState {
                n_modes: m,
                n_photons: 0,
                dims,
                amps: vec![C64::new(1.0, 0.0)],
            });
        }

        let sqrt_table: Vec<f64> = (0..=n_photons).map(|k| (k as f64).sqrt()).collect();

        // Ping-pong buffers of sizes up to D(m, N).
        let mut cur = vec![C64::new(1.0, 0.0)];
        let mut step = 0usize;
        for (in_mode, &count) in input.iter().enumerate() {
            if count == 0 {
                continue;
            }
            // Extract column `in_mode` of `unitary`: col[i] = U[i, in_mode].
            let col: Vec<C64> = (0..m).map(|i| unitary[i * m + in_mode]).collect();
            for rep in 1..=count {
                step += 1;
                let inv_rep_sqrt = 1.0 / sqrt_table[rep];
                let scaled_col: Vec<C64> = col.iter().map(|&z| z * inv_rep_sqrt).collect();
                let next_dim = dims.dim(m, step);
                let mut next = vec![C64::new(0.0, 0.0); next_dim];
                slos_step(
                    m,
                    step,
                    &scaled_col,
                    &cur,
                    &mut next,
                    &dims,
                    &sqrt_table,
                    false,
                );
                cur = next;
            }
        }

        Ok(FixedNFockState {
            n_modes: m,
            n_photons,
            dims,
            amps: cur,
        })
    }

    fn run_ops<R: Rng + ?Sized>(
        &mut self,
        ops: &[ModeOp],
        results: &mut Vec<f64>,
        rng: &mut R,
    ) -> Result<(), Error> {
        for op in ops {
            match op {
                ModeOp::Measure {
                    mode, measurement, ..
                } => match *measurement {
                    Measurement::PhotonCount => {
                        let k = self.measure_photons(*mode, rng)?;
                        results.push(k as f64);
                    }
                    Measurement::Threshold => {
                        let clicked = self.measure_threshold(*mode, rng)?;
                        results.push(if clicked { 1.0 } else { 0.0 });
                    }
                    _ => {
                        return Err(Error::InvalidArgument(
                            "FixedNFockState supports photon-counting and threshold measurements"
                                .into(),
                        ));
                    }
                },
                other => self.apply_unitary_op(other, results)?,
            }
        }
        Ok(())
    }

    fn apply_unitary_op(&mut self, op: &ModeOp, results: &[f64]) -> Result<(), Error> {
        match op {
            ModeOp::Gate { gate, mode } => match *gate {
                ModeGate::Rotation { phi } => self.rotation(phi.evaluate(results)?, *mode),
                ModeGate::Kerr { kappa } => self.kerr(kappa.evaluate(results)?, *mode),
                _ => Err(Error::InvalidArgument(format!(
                    "gate {gate:?} does not conserve photon number"
                ))),
            },
            ModeOp::TwoModeGate { gate, modes } => {
                let [a, b] = *modes;
                match *gate {
                    TwoModeGate::BeamSplitter { theta, phi } => {
                        self.beam_splitter(theta.evaluate(results)?, phi.evaluate(results)?, a, b)
                    }
                    TwoModeGate::CrossKerr { kappa } => {
                        self.cross_kerr(kappa.evaluate(results)?, a, b)
                    }
                    _ => Err(Error::InvalidArgument(format!(
                        "gate {gate:?} does not conserve photon number"
                    ))),
                }
            }
            ModeOp::Passive { modes, matrix } => self.apply_passive(modes, matrix),
            ModeOp::Channel { .. } => Err(Error::InvalidArgument(
                "FixedNFockState does not support non-unitary noise channels".into(),
            )),
            ModeOp::Measure { .. } => Err(Error::InvalidArgument(
                "unexpected measurement in unitary execution".into(),
            )),
        }
    }

    // -----------------------------------------------------------------------
    // Accessors, basis indexing, & observables
    // -----------------------------------------------------------------------

    /// Number of modes.
    pub fn n_modes(&self) -> usize {
        self.n_modes
    }

    /// Total photon number `N`.
    pub fn n_photons(&self) -> usize {
        self.n_photons
    }

    /// Basis dimension `C(m + N - 1, N)`.
    pub fn dim(&self) -> usize {
        self.amps.len()
    }

    /// Raw amplitudes in the recursive colexicographic basis.
    pub fn amplitudes(&self) -> &[C64] {
        &self.amps
    }

    /// Basis occupation patterns corresponding to `amplitudes()`, in the same order.
    pub fn basis(&self) -> Vec<Vec<usize>> {
        let mut out = Vec::with_capacity(self.amps.len());
        let mut pat = vec![0usize; self.n_modes];
        enumerate_basis(self.n_modes, self.n_photons, &mut pat, &mut out);
        out
    }

    /// Index of `pattern` in `amplitudes()`, or `None` if `pattern.len() != n_modes`
    /// or `sum(pattern) != n_photons`.
    pub fn index_of(&self, pattern: &[usize]) -> Option<usize> {
        if pattern.len() != self.n_modes || pattern.iter().sum::<usize>() != self.n_photons {
            return None;
        }
        Some(pattern_index(&self.dims, pattern))
    }

    /// Amplitude `⟨pattern|ψ⟩` (returns `0` if `sum(pattern) != n_photons`).
    pub fn amplitude(&self, pattern: &[usize]) -> Result<C64, Error> {
        if pattern.len() != self.n_modes {
            return Err(Error::InvalidArgument(format!(
                "pattern has {} modes, expected {}",
                pattern.len(),
                self.n_modes
            )));
        }
        if pattern.iter().sum::<usize>() != self.n_photons {
            return Ok(C64::new(0.0, 0.0));
        }
        Ok(self.amps[pattern_index(&self.dims, pattern)])
    }

    /// Probability `|⟨pattern|ψ⟩|²`.
    pub fn probability(&self, pattern: &[usize]) -> Result<f64, Error> {
        Ok(self.amplitude(pattern)?.norm_sqr())
    }

    /// Exact probabilities `|c_i|²` for all basis states in `basis()` order.
    pub fn probabilities(&self) -> Vec<f64> {
        if self.amps.len() >= GATE_PAR_MIN_DIM {
            self.amps.par_iter().map(|z| z.norm_sqr()).collect()
        } else {
            self.amps.iter().map(|z| z.norm_sqr()).collect()
        }
    }

    /// Mean photon number `⟨n̂⟩` on `mode`.
    pub fn mean_photons(&self, mode: usize) -> Result<f64, Error> {
        self.check_mode(mode)?;
        let mut total = 0.0;
        visit_mode_occ(
            self.n_modes,
            self.n_photons,
            mode,
            &self.dims,
            &self.amps,
            &mut |occ, slice| {
                if occ > 0 {
                    let w: f64 = slice.iter().map(|z| z.norm_sqr()).sum();
                    total += (occ as f64) * w;
                }
            },
        );
        Ok(total)
    }

    // -----------------------------------------------------------------------
    // Number-conserving gates
    // -----------------------------------------------------------------------

    fn check_mode(&self, mode: usize) -> Result<(), Error> {
        if mode < self.n_modes {
            Ok(())
        } else {
            Err(Error::InvalidArgument(format!(
                "mode {mode} out of range for {} modes",
                self.n_modes
            )))
        }
    }

    fn check_pair(&self, a: usize, b: usize) -> Result<(), Error> {
        self.check_mode(a)?;
        self.check_mode(b)?;
        if a == b {
            return Err(Error::InvalidArgument(format!(
                "mode {a} used more than once"
            )));
        }
        Ok(())
    }

    /// Rotation `R(φ) = exp(iφ n̂)` on `mode`.
    pub fn rotation(&mut self, phi: f64, mode: usize) -> Result<(), Error> {
        self.check_mode(mode)?;
        if phi == 0.0 || self.n_photons == 0 {
            return Ok(());
        }
        let phases: Vec<C64> = (0..=self.n_photons)
            .map(|n| C64::from_polar(1.0, phi * (n as f64)))
            .collect();
        apply_diagonal_one_mode(
            self.n_modes,
            self.n_photons,
            mode,
            &self.dims,
            &phases,
            &mut self.amps,
        );
        Ok(())
    }

    /// Kerr `K(κ) = exp(iκ n̂²)` on `mode`.
    pub fn kerr(&mut self, kappa: f64, mode: usize) -> Result<(), Error> {
        self.check_mode(mode)?;
        if kappa == 0.0 || self.n_photons == 0 {
            return Ok(());
        }
        let phases: Vec<C64> = (0..=self.n_photons)
            .map(|n| C64::from_polar(1.0, kappa * ((n * n) as f64)))
            .collect();
        apply_diagonal_one_mode(
            self.n_modes,
            self.n_photons,
            mode,
            &self.dims,
            &phases,
            &mut self.amps,
        );
        Ok(())
    }

    /// Cross-Kerr `CK(κ) = exp(iκ n̂_a n̂_b)` on `(a, b)`.
    pub fn cross_kerr(&mut self, kappa: f64, a: usize, b: usize) -> Result<(), Error> {
        self.check_pair(a, b)?;
        if kappa == 0.0 || self.n_photons < 2 {
            return Ok(());
        }
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        apply_cross_kerr_rec(
            self.n_modes,
            self.n_photons,
            lo,
            hi,
            0,
            kappa,
            &self.dims,
            &mut self.amps,
        );
        Ok(())
    }

    /// Beam splitter `BS(θ, φ)` on `(a, b)` using exact `(s + 1) × (s + 1)`
    /// photon-number blocks.
    pub fn beam_splitter(&mut self, theta: f64, phi: f64, a: usize, b: usize) -> Result<(), Error> {
        self.check_pair(a, b)?;
        if self.n_photons == 0 {
            return Ok(());
        }
        let t = C64::new(theta.cos(), 0.0);
        let r = C64::from_polar(theta.sin(), phi);
        // 2×2 unitary [[t, -r*], [r, t]]
        let u2 = [t, -r.conj(), r, t];
        self.apply_two_mode_unitary(&u2, a, b);
        Ok(())
    }

    /// Apply an arbitrary `k × k` passive unitary `U` on `modes` via exact
    /// Givens (beam-splitter + phase) decomposition.
    pub fn apply_passive(&mut self, modes: &[usize], matrix: &[C64]) -> Result<(), Error> {
        let k = modes.len();
        if matrix.len() != k * k {
            return Err(Error::InvalidArgument(format!(
                "passive block on {k} modes needs {} entries, got {}",
                k * k,
                matrix.len()
            )));
        }
        for (i, &m) in modes.iter().enumerate() {
            self.check_mode(m)?;
            if modes[..i].contains(&m) {
                return Err(Error::InvalidArgument(format!(
                    "mode {m} used more than once"
                )));
            }
        }
        if k == 0 || self.n_photons == 0 {
            return Ok(());
        }
        if k == 1 {
            self.rotation(matrix[0].arg(), modes[0])?;
            return Ok(());
        }
        if k == 2 {
            self.apply_two_mode_unitary(
                &[matrix[0], matrix[1], matrix[2], matrix[3]],
                modes[0],
                modes[1],
            );
            return Ok(());
        }

        // Factor U = D · G_L · ... · G_1 (right-to-left Givens elimination of
        // U^† -> I so U = G_1^† ... G_L^† D).
        let (ops, diag_phases) = givens_decompose(matrix, k);
        for (i, &phi) in diag_phases.iter().enumerate() {
            if phi.abs() > 1e-15 {
                self.rotation(phi, modes[i])?;
            }
        }
        for (i, j, u2) in ops.into_iter().rev() {
            self.apply_two_mode_unitary(&u2, modes[i], modes[j]);
        }
        Ok(())
    }

    /// Apply an arbitrary `2 × 2` passive unitary `u2 = [u00, u01, u10, u11]`
    /// on `(a, b)` in place.
    fn apply_two_mode_unitary(&mut self, u2: &[C64; 4], a: usize, b: usize) {
        let blocks = build_two_mode_blocks(u2, self.n_photons);
        let m = self.n_modes;
        let n = self.n_photons;

        // Fast path when m == 2: one single matrix-vector product!
        if m == 2 {
            let block = &blocks[n];
            let s1 = n + 1;
            let mut out = vec![C64::new(0.0, 0.0); s1];
            // In our colex basis on 2 modes: index p is (n_0 = n - p, n_1 = p).
            // In `blocks[n]`, row/col k is (n_a = k, n_b = n - k).
            for out_k in 0..s1 {
                let dst_idx = if a == 0 { n - out_k } else { out_k };
                let mut acc = C64::new(0.0, 0.0);
                let row = out_k * s1;
                for in_k in 0..s1 {
                    let src_idx = if a == 0 { n - in_k } else { in_k };
                    acc += block[row + in_k] * self.amps[src_idx];
                }
                out[dst_idx] = acc;
            }
            self.amps = out;
            return;
        }

        // General m >= 3: iterate over all fixed occupations of the other m - 2
        // modes (sum r = 0..=N -> s = N - r photons in (a, b)).
        let other_modes: Vec<usize> = (0..m).filter(|&i| i != a && i != b).collect();
        let mut pat = vec![0usize; m];
        let mut buf_in = vec![C64::new(0.0, 0.0); n + 1];
        let mut idxs = vec![0usize; n + 1];

        #[allow(clippy::too_many_arguments)]
        fn walk_other(
            pos: usize,
            other_modes: &[usize],
            rem_budget: usize,
            a: usize,
            b: usize,
            pat: &mut [usize],
            dims: &DimTable,
            blocks: &[Vec<C64>],
            amps: &mut [C64],
            buf_in: &mut [C64],
            idxs: &mut [usize],
        ) {
            if pos == other_modes.len() {
                let s = rem_budget;
                if s == 0 {
                    return;
                }
                let s1 = s + 1;
                for k in 0..s1 {
                    pat[a] = k;
                    pat[b] = s - k;
                    let idx = pattern_index(dims, pat);
                    idxs[k] = idx;
                    buf_in[k] = amps[idx];
                }
                let block = &blocks[s];
                for out_k in 0..s1 {
                    let row = out_k * s1;
                    let mut acc = C64::new(0.0, 0.0);
                    for in_k in 0..s1 {
                        acc += block[row + in_k] * buf_in[in_k];
                    }
                    amps[idxs[out_k]] = acc;
                }
                return;
            }
            let mode = other_modes[pos];
            for occ in 0..=rem_budget {
                pat[mode] = occ;
                walk_other(
                    pos + 1,
                    other_modes,
                    rem_budget - occ,
                    a,
                    b,
                    pat,
                    dims,
                    blocks,
                    amps,
                    buf_in,
                    idxs,
                );
            }
            pat[mode] = 0;
        }

        walk_other(
            0,
            &other_modes,
            n,
            a,
            b,
            &mut pat,
            &self.dims,
            &blocks,
            &mut self.amps,
            &mut buf_in,
            &mut idxs,
        );
    }

    // -----------------------------------------------------------------------
    // Destructive PNR & threshold measurements (reset mode to vacuum)
    // -----------------------------------------------------------------------

    /// Destructive photon-number measurement on `mode`: samples `k`, projects
    /// onto `n_mode = k`, resets `mode` to vacuum (reducing total photon number
    /// to `N - k`), and returns `k`.
    pub fn measure_photons<R: Rng + ?Sized>(
        &mut self,
        mode: usize,
        rng: &mut R,
    ) -> Result<usize, Error> {
        self.check_mode(mode)?;
        let n = self.n_photons;
        let mut probs = vec![0.0f64; n + 1];
        visit_mode_occ(
            self.n_modes,
            n,
            mode,
            &self.dims,
            &self.amps,
            &mut |occ, slice| {
                probs[occ] += slice.iter().map(|z| z.norm_sqr()).sum::<f64>();
            },
        );
        let total: f64 = probs.iter().sum();
        let u = rng.random::<f64>() * total;
        let mut cum = 0.0;
        let mut outcome = n;
        for (k, &p) in probs.iter().enumerate() {
            cum += p;
            if u < cum {
                outcome = k;
                break;
            }
        }
        let prob = probs[outcome];
        let inv_norm = if prob > 0.0 { 1.0 / prob.sqrt() } else { 0.0 };

        // Project onto `n_mode == outcome` and shift to `n_mode == 0` in the
        // `(N - outcome)`-photon subspace!
        let new_n = n - outcome;
        let new_dims = DimTable::new(self.n_modes, new_n);
        let mut new_amps = vec![C64::new(0.0, 0.0); new_dims.dim(self.n_modes, new_n)];
        let mut pat = vec![0usize; self.n_modes];
        project_and_reset_mode(
            self.n_modes,
            n,
            mode,
            outcome,
            inv_norm,
            &self.dims,
            &new_dims,
            &mut pat,
            &self.amps,
            &mut new_amps,
        );
        self.n_photons = new_n;
        self.dims = new_dims;
        self.amps = new_amps;
        Ok(outcome)
    }

    /// Destructive threshold measurement on `mode` (exact trajectory unraveling:
    /// samples underlying PNR outcome `k`, projects & resets `mode` to vacuum,
    /// returns `k > 0`).
    pub fn measure_threshold<R: Rng + ?Sized>(
        &mut self,
        mode: usize,
        rng: &mut R,
    ) -> Result<bool, Error> {
        Ok(self.measure_photons(mode, rng)? > 0)
    }

    fn sample_terminal<R: Rng + ?Sized>(
        &self,
        meas_spec: &[(usize, bool)],
        shots: usize,
        rng: &mut R,
    ) -> Vec<Vec<f64>> {
        // Build cumulative distribution over the D(m, N) basis states.
        let d = self.amps.len();
        let mut cdf = Vec::with_capacity(d);
        let mut acc = 0.0;
        for z in &self.amps {
            acc += z.norm_sqr();
            cdf.push(acc);
        }
        if let Some(last) = cdf.last_mut() {
            *last = 1.0;
        }
        let draw_one = |r: f64| -> Vec<f64> {
            let idx = cdf.partition_point(|&c| c <= r).min(d - 1);
            let pat = index_to_pattern(&self.dims, self.n_modes, self.n_photons, idx);
            meas_spec
                .iter()
                .map(|&(mode, is_thresh)| {
                    if is_thresh {
                        if pat[mode] > 0 { 1.0 } else { 0.0 }
                    } else {
                        pat[mode] as f64
                    }
                })
                .collect()
        };
        if shots >= SAMPLE_PAR_MIN_SHOTS {
            let us: Vec<f64> = (0..shots).map(|_| rng.random::<f64>()).collect();
            us.into_par_iter().map(draw_one).collect()
        } else {
            (0..shots).map(|_| draw_one(rng.random::<f64>())).collect()
        }
    }
}

// ---------------------------------------------------------------------------
// Recursive cache-optimal SLOS kernel
// ---------------------------------------------------------------------------

/// Compute `dst += (Σ_{i=0}^{m-1} col[i] a_i†) src` (or `dst = ...` when
/// `!accumulate`) in the recursive colexicographic basis.
#[allow(clippy::too_many_arguments)]
fn slos_step(
    m: usize,
    k: usize,
    col: &[C64],
    src: &[C64],
    dst: &mut [C64],
    dims: &DimTable,
    sqrt_table: &[f64],
    accumulate: bool,
) {
    debug_assert_eq!(src.len(), dims.dim(m, k - 1));
    debug_assert_eq!(dst.len(), dims.dim(m, k));

    if m == 1 {
        let val = col[0] * sqrt_table[k] * src[0];
        if accumulate {
            dst[0] += val;
        } else {
            dst[0] = val;
        }
        return;
    }

    if k == 1 {
        let s0 = src[0];
        if accumulate {
            for (d, &u) in dst.iter_mut().zip(col) {
                *d += u * s0;
            }
        } else {
            for (d, &u) in dst.iter_mut().zip(col) {
                *d = u * s0;
            }
        }
        return;
    }

    if m == 2 {
        let u0 = col[0];
        let u1 = col[1];
        if accumulate {
            dst[0] += (u0 * sqrt_table[k]) * src[0];
            for p in 1..k {
                dst[p] += (u0 * sqrt_table[k - p]) * src[p] + (u1 * sqrt_table[p]) * src[p - 1];
            }
            dst[k] += (u1 * sqrt_table[k]) * src[k - 1];
        } else {
            dst[0] = (u0 * sqrt_table[k]) * src[0];
            for p in 1..k {
                dst[p] = (u0 * sqrt_table[k - p]) * src[p] + (u1 * sqrt_table[p]) * src[p - 1];
            }
            dst[k] = (u1 * sqrt_table[k]) * src[k - 1];
        }
        return;
    }

    // Partition `src` into blocks p = 0..k of lengths D(m - 1, k - 1 - p)
    // and `dst` into blocks p = 0..=k of lengths D(m - 1, k - p).
    let u_last = col[m - 1];
    let sub_col = &col[..m - 1];

    if dst.len() >= SLOS_PAR_MIN_DIM {
        // Precompute src block slices and split dst into disjoint mutable chunks
        // so every block `p` executes in parallel with zero contention!
        let mut src_slices: Vec<&[C64]> = Vec::with_capacity(k);
        let mut off = 0usize;
        for p in 0..k {
            let len = dims.dim(m - 1, k - 1 - p);
            src_slices.push(&src[off..off + len]);
            off += len;
        }
        let mut dst_slices: Vec<(usize, &mut [C64])> = Vec::with_capacity(k + 1);
        let mut rest = dst;
        for p in 0..=k {
            let len = dims.dim(m - 1, k - p);
            let (head, tail) = rest.split_at_mut(len);
            dst_slices.push((p, head));
            rest = tail;
        }
        dst_slices.into_par_iter().for_each(|(p, d_blk)| {
            if p > 0 {
                let scale = u_last * sqrt_table[p];
                let s_prev = src_slices[p - 1];
                if accumulate {
                    for (d, &s) in d_blk.iter_mut().zip(s_prev) {
                        *d += scale * s;
                    }
                } else {
                    for (d, &s) in d_blk.iter_mut().zip(s_prev) {
                        *d = scale * s;
                    }
                }
                if p < k {
                    slos_step(
                        m - 1,
                        k - p,
                        sub_col,
                        src_slices[p],
                        d_blk,
                        dims,
                        sqrt_table,
                        true,
                    );
                }
            } else {
                // p == 0: only contribution from modes 0..m-1.
                slos_step(
                    m - 1,
                    k,
                    sub_col,
                    src_slices[0],
                    d_blk,
                    dims,
                    sqrt_table,
                    accumulate,
                );
            }
        });
    } else {
        let mut dst_off = 0usize;
        let mut src_off = 0usize;
        let mut prev_src_off = 0usize;
        for p in 0..=k {
            let d_len = dims.dim(m - 1, k - p);
            let d_blk = &mut dst[dst_off..dst_off + d_len];
            if p > 0 {
                let scale = u_last * sqrt_table[p];
                let s_prev = &src[prev_src_off..prev_src_off + d_len];
                if accumulate {
                    for (d, &s) in d_blk.iter_mut().zip(s_prev) {
                        *d += scale * s;
                    }
                } else {
                    for (d, &s) in d_blk.iter_mut().zip(s_prev) {
                        *d = scale * s;
                    }
                }
                if p < k {
                    let s_len = dims.dim(m - 1, k - 1 - p);
                    let s_cur = &src[src_off..src_off + s_len];
                    slos_step(m - 1, k - p, sub_col, s_cur, d_blk, dims, sqrt_table, true);
                    prev_src_off = src_off;
                    src_off += s_len;
                }
            } else {
                let s_len = dims.dim(m - 1, k - 1);
                let s_cur = &src[0..s_len];
                slos_step(
                    m - 1,
                    k,
                    sub_col,
                    s_cur,
                    d_blk,
                    dims,
                    sqrt_table,
                    accumulate,
                );
                prev_src_off = 0;
                src_off = s_len;
            }
            dst_off += d_len;
        }
    }
}

// ---------------------------------------------------------------------------
// Basis indexing, diagonal gates, 2-mode blocks, and Givens decomposition
// ---------------------------------------------------------------------------

#[inline]
fn pattern_index(dims: &DimTable, pattern: &[usize]) -> usize {
    let m = pattern.len();
    let mut rem: usize = pattern.iter().sum();
    let mut idx = 0usize;
    for mode in (1..m).rev() {
        let p = pattern[mode];
        if p > 0 {
            idx += dims.dim(mode + 1, rem) - dims.dim(mode + 1, rem - p);
            rem -= p;
        }
    }
    idx
}

fn index_to_pattern(dims: &DimTable, m: usize, n: usize, mut idx: usize) -> Vec<usize> {
    let mut pat = vec![0usize; m];
    let mut rem = n;
    for mode in (1..m).rev() {
        let mut p = 0usize;
        while p < rem {
            let blk = dims.dim(mode, rem - p);
            if idx < blk {
                break;
            }
            idx -= blk;
            p += 1;
        }
        pat[mode] = p;
        rem -= p;
    }
    pat[0] = rem;
    pat
}

fn enumerate_basis(m: usize, n: usize, pat: &mut [usize], out: &mut Vec<Vec<usize>>) {
    if m == 1 {
        pat[0] = n;
        out.push(pat.to_vec());
        return;
    }
    for p in 0..=n {
        pat[m - 1] = p;
        enumerate_basis(m - 1, n - p, pat, out);
    }
}

fn extract_fock_inputs(inputs: &[ModeState]) -> Result<Vec<usize>, Error> {
    inputs
        .iter()
        .enumerate()
        .map(|(m, s)| match *s {
            ModeState::Vacuum => Ok(0),
            ModeState::Fock(n) => Ok(n),
            ModeState::Thermal { mean_photons: 0.0 } => Ok(0),
            _ => Err(Error::InvalidArgument(format!(
                "FixedNFockState requires Fock input states, mode {m} is {s:?}"
            ))),
        })
        .collect()
}

fn fuse_passive_op(u: &mut [C64], m: usize, op: &ModeOp) -> Result<(), Error> {
    match op {
        ModeOp::Gate {
            gate: ModeGate::Rotation { phi },
            mode,
        } => {
            let phase = C64::from_polar(1.0, phi.evaluate(&[])?);
            let row = *mode * m;
            for c in 0..m {
                u[row + c] *= phase;
            }
            Ok(())
        }
        ModeOp::TwoModeGate {
            gate: TwoModeGate::BeamSplitter { theta, phi },
            modes: [a, b],
        } => {
            let th = theta.evaluate(&[])?;
            let ph = phi.evaluate(&[])?;
            let t = C64::new(th.cos(), 0.0);
            let r = C64::from_polar(th.sin(), ph);
            let ra = *a * m;
            let rb = *b * m;
            for c in 0..m {
                let va = u[ra + c];
                let vb = u[rb + c];
                u[ra + c] = t * va - r.conj() * vb;
                u[rb + c] = r * va + t * vb;
            }
            Ok(())
        }
        ModeOp::Passive { modes, matrix } => {
            let k = modes.len();
            let old: Vec<Vec<C64>> = modes
                .iter()
                .map(|&row| u[row * m..(row + 1) * m].to_vec())
                .collect();
            for (r, &dst_mode) in modes.iter().enumerate() {
                let dst = dst_mode * m;
                for c in 0..m {
                    let mut acc = C64::new(0.0, 0.0);
                    for j in 0..k {
                        acc += matrix[r * k + j] * old[j][c];
                    }
                    u[dst + c] = acc;
                }
            }
            Ok(())
        }
        _ => Err(Error::InvalidArgument(format!(
            "operation {op:?} is not passive"
        ))),
    }
}

/// Build the `(s + 1) × (s + 1)` unitary matrices for a 2-mode unitary
/// `u2 = [u00, u01, u10, u11]` on total photon numbers `s = 0..=max_n`.
/// Row/col `k` corresponds to `|n_a = k, n_b = s - k⟩`.
pub(crate) fn build_two_mode_blocks(u2: &[C64; 4], max_n: usize) -> Vec<Vec<C64>> {
    let sqrt_t: Vec<f64> = (0..=max_n).map(|k| (k as f64).sqrt()).collect();
    let mut blocks = Vec::with_capacity(max_n + 1);
    blocks.push(vec![C64::new(1.0, 0.0)]); // s = 0

    let col0 = [u2[0], u2[2]]; // a_0† -> u00 a_0† + u10 a_1†
    let col1 = [u2[1], u2[3]]; // a_1† -> u01 a_0† + u11 a_1†

    for s in 1..=max_n {
        let s1 = s + 1;
        let mut mat = vec![C64::new(0.0, 0.0); s1 * s1];
        // Column `in_k` = U |in_k, s - in_k⟩ computed by s steps of 2-mode SLOS.
        // Even better: compute column `in_k` by applying `(col0 a_0† + col1 a_1†)`
        // once to column of `blocks[s - 1]`!!
        // - For `in_k > 0`: `|in_k, s - in_k⟩ = (a_0† / √in_k) |in_k - 1, s - in_k⟩`
        //   -> apply `(u00 a_0† + u10 a_1†) / √in_k` to column `in_k - 1` of `blocks[s - 1]`!
        // - For `in_k == 0`: `|0, s⟩ = (a_1† / √s) |0, s - 1⟩`
        //   -> apply `(u01 a_0† + u11 a_1†) / √s` to column `0` of `blocks[s - 1]`!
        let prev = &blocks[s - 1];
        // in_k = 0: from prev col 0 with col1 / sqrt(s).
        let inv_s = 1.0 / sqrt_t[s];
        for r in 0..s {
            let v = prev[r * s] * inv_s;
            // a_0† on |r, s - 1 - r⟩ -> sqrt(r + 1) |r + 1, s - 1 - r⟩
            mat[(r + 1) * s1] += col1[0] * sqrt_t[r + 1] * v;
            // a_1† on |r, s - 1 - r⟩ -> sqrt(s - r) |r, s - r⟩
            mat[r * s1] += col1[1] * sqrt_t[s - r] * v;
        }
        // in_k = 1..=s: from prev col (in_k - 1) with col0 / sqrt(in_k).
        for in_k in 1..s1 {
            let inv_k = 1.0 / sqrt_t[in_k];
            let prev_col = in_k - 1;
            for r in 0..s {
                let v = prev[r * s + prev_col] * inv_k;
                mat[(r + 1) * s1 + in_k] += col0[0] * sqrt_t[r + 1] * v;
                mat[r * s1 + in_k] += col0[1] * sqrt_t[s - r] * v;
            }
        }
        blocks.push(mat);
    }
    blocks
}

/// A Givens two-mode rotation `(mode_a, mode_b, u2_row_major)`.
pub(crate) type GivensOp = (usize, usize, [C64; 4]);

/// Exact Givens decomposition of a `k × k` row-major unitary `U`:
/// returns `(gates, diag_phases)` such that applying `diag_phases` first and
/// then `gates` in reverse order (`u2` on `(i, j)`) reproduces `U`.
pub(crate) fn givens_decompose(u: &[C64], k: usize) -> (Vec<GivensOp>, Vec<f64>) {
    let mut w = u.to_vec();
    let mut gates = Vec::new();
    // Eliminate lower triangle column by column using left-multiplied 2×2
    // unitaries G such that G_L ... G_1 U = D (diagonal unitary).
    // Then U = G_1† ... G_L† D.
    for col in 0..k {
        for row in (col + 1..k).rev() {
            let b = w[row * k + col];
            if b.norm() < 1e-14 {
                continue;
            }
            let a = w[(row - 1) * k + col];
            let r = (a.norm_sqr() + b.norm_sqr()).sqrt();
            // G = [[a*/r, b*/r], [-b/r, a/r]] zeros row `row` in column `col`.
            let g00 = a.conj() / r;
            let g01 = b.conj() / r;
            let g10 = -b / r;
            let g11 = a / r;
            for c in col..k {
                let x = w[(row - 1) * k + c];
                let y = w[row * k + c];
                w[(row - 1) * k + c] = g00 * x + g01 * y;
                w[row * k + c] = g10 * x + g11 * y;
            }
            // Store G† = [[a/r, -b*/r], [b/r, a*/r]].
            let g_dag = [a / r, -b.conj() / r, b / r, a.conj() / r];
            gates.push((row - 1, row, g_dag));
        }
    }
    let diag_phases: Vec<f64> = (0..k).map(|i| w[i * k + i].arg()).collect();
    (gates, diag_phases)
}

fn visit_mode_occ(
    m: usize,
    n: usize,
    target: usize,
    dims: &DimTable,
    amps: &[C64],
    f: &mut impl FnMut(usize, &[C64]),
) {
    if m == 1 {
        f(n, amps);
        return;
    }
    let cur_mode = m - 1;
    let mut off = 0usize;
    for p in 0..=n {
        let len = dims.dim(m - 1, n - p);
        let blk = &amps[off..off + len];
        if cur_mode == target {
            f(p, blk);
        } else {
            visit_mode_occ(m - 1, n - p, target, dims, blk, f);
        }
        off += len;
    }
}

fn apply_diagonal_one_mode(
    m: usize,
    n: usize,
    target: usize,
    dims: &DimTable,
    phases: &[C64],
    amps: &mut [C64],
) {
    if m == 1 {
        amps[0] *= phases[n];
        return;
    }
    let cur_mode = m - 1;
    if cur_mode == target {
        let mut off = 0usize;
        for (p, &ph) in phases[..=n].iter().enumerate() {
            let len = dims.dim(m - 1, n - p);
            if ph != C64::new(1.0, 0.0) {
                for z in &mut amps[off..off + len] {
                    *z *= ph;
                }
            }
            off += len;
        }
    } else {
        let mut off = 0usize;
        for p in 0..=n {
            let len = dims.dim(m - 1, n - p);
            apply_diagonal_one_mode(
                m - 1,
                n - p,
                target,
                dims,
                phases,
                &mut amps[off..off + len],
            );
            off += len;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_cross_kerr_rec(
    m: usize,
    n: usize,
    lo: usize,
    hi: usize,
    hi_occ: usize,
    kappa: f64,
    dims: &DimTable,
    amps: &mut [C64],
) {
    if m == 1 {
        // Must be lo == 0.
        if hi_occ > 0 && n > 0 {
            amps[0] *= C64::from_polar(1.0, kappa * ((hi_occ * n) as f64));
        }
        return;
    }
    let cur_mode = m - 1;
    let mut off = 0usize;
    for p in 0..=n {
        let len = dims.dim(m - 1, n - p);
        let blk = &mut amps[off..off + len];
        if cur_mode == hi {
            if p > 0 {
                apply_cross_kerr_rec(m - 1, n - p, lo, hi, p, kappa, dims, blk);
            }
        } else if cur_mode == lo {
            if p > 0 && hi_occ > 0 {
                let ph = C64::from_polar(1.0, kappa * ((hi_occ * p) as f64));
                for z in blk.iter_mut() {
                    *z *= ph;
                }
            }
        } else {
            apply_cross_kerr_rec(m - 1, n - p, lo, hi, hi_occ, kappa, dims, blk);
        }
        off += len;
    }
}

#[allow(clippy::too_many_arguments)]
fn project_and_reset_mode(
    m: usize,
    n: usize,
    target: usize,
    outcome: usize,
    inv_norm: f64,
    old_dims: &DimTable,
    new_dims: &DimTable,
    pat: &mut [usize],
    old_amps: &[C64],
    new_amps: &mut [C64],
) {
    // Enumerate old basis states with pat[target] == outcome and write into
    // new_amps with pat[target] = 0.
    fn walk(
        mode_count: usize,
        rem: usize,
        target: usize,
        outcome: usize,
        inv_norm: f64,
        old_dims: &DimTable,
        new_dims: &DimTable,
        pat: &mut [usize],
        old_amps: &[C64],
        new_amps: &mut [C64],
        old_off: usize,
    ) {
        if mode_count == 1 {
            if target == 0 {
                if rem != outcome {
                    return;
                }
                pat[0] = 0;
            } else {
                pat[0] = rem;
            }
            let new_idx = pattern_index(new_dims, pat);
            new_amps[new_idx] = old_amps[old_off] * inv_norm;
            return;
        }
        let cur = mode_count - 1;
        let mut off = old_off;
        for p in 0..=rem {
            let len = old_dims.dim(mode_count - 1, rem - p);
            if cur == target {
                if p == outcome {
                    pat[cur] = 0;
                    walk(
                        mode_count - 1,
                        rem - p,
                        target,
                        outcome,
                        inv_norm,
                        old_dims,
                        new_dims,
                        pat,
                        old_amps,
                        new_amps,
                        off,
                    );
                }
            } else {
                pat[cur] = p;
                walk(
                    mode_count - 1,
                    rem - p,
                    target,
                    outcome,
                    inv_norm,
                    old_dims,
                    new_dims,
                    pat,
                    old_amps,
                    new_amps,
                    off,
                );
            }
            off += len;
        }
    }
    walk(
        m, n, target, outcome, inv_norm, old_dims, new_dims, pat, old_amps, new_amps, 0,
    );
}

fn terminal_pnr_plan(suffix: &[ModeOp], n_modes: usize) -> Option<Vec<(usize, bool)>> {
    let mut seen = vec![false; n_modes];
    let mut spec = Vec::with_capacity(suffix.len());
    for op in suffix {
        let ModeOp::Measure {
            mode, measurement, ..
        } = op
        else {
            return None;
        };
        if seen[*mode] {
            return None;
        }
        seen[*mode] = true;
        match measurement {
            Measurement::PhotonCount => spec.push((*mode, false)),
            Measurement::Threshold => spec.push((*mode, true)),
            _ => return None,
        }
    }
    Some(spec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cv::reference::FockReference;
    use std::f64::consts::{FRAC_PI_4, PI};

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() < tol, "{a} vs {b}");
    }

    fn close_c(a: C64, b: C64, tol: f64) {
        assert!((a - b).norm() < tol, "{a} vs {b}");
    }

    #[test]
    fn hong_ou_mandel_and_passive_slos_match_reference() {
        let mut c = ModeCircuit::new(3);
        c.set_input(0, ModeState::Fock(2)).unwrap();
        c.set_input(1, ModeState::Fock(1)).unwrap();
        c.set_input(2, ModeState::Fock(1)).unwrap();
        c.beam_splitter(0.4, 0.3, 0, 1)
            .rotation(0.7, 2)
            .beam_splitter(FRAC_PI_4, -0.5, 1, 2)
            .beam_splitter(0.9, 1.1, 0, 2);

        let st = FixedNFockState::from_circuit(&c).unwrap();
        let rf = FockReference::run(&c, 5).unwrap();
        assert_eq!(st.n_photons(), 4);
        assert_eq!(st.dim(), FixedNFockState::basis_dim(3, 4));
        close(st.probabilities().iter().sum(), 1.0, 1e-12);

        for pat in st.basis() {
            close_c(st.amplitude(&pat).unwrap(), rf.amplitude(&pat), 1e-11);
        }
    }

    #[test]
    fn kerr_cross_kerr_and_mid_circuit_passive_match_reference() {
        let mut c = ModeCircuit::new(3);
        c.set_input(0, ModeState::Fock(1)).unwrap();
        c.set_input(1, ModeState::Fock(2)).unwrap();
        c.set_input(2, ModeState::Fock(1)).unwrap();
        c.beam_splitter(0.5, 0.2, 0, 1)
            .kerr(0.3, 1)
            .cross_kerr(-0.4, 0, 2)
            .beam_splitter(0.8, -0.3, 1, 2)
            .rotation(0.6, 0)
            .beam_splitter(0.4, 0.9, 0, 2);

        let st = FixedNFockState::from_circuit(&c).unwrap();
        let rf = FockReference::run(&c, 5).unwrap();
        for pat in st.basis() {
            close_c(st.amplitude(&pat).unwrap(), rf.amplitude(&pat), 1e-11);
        }
        for m in 0..3 {
            close(st.mean_photons(m).unwrap(), rf.mean_photons(m), 1e-11);
        }
    }

    #[test]
    fn givens_passive_after_kerr_matches_reference() {
        // 3×3 DFT unitary applied AFTER a Kerr gate (hits Givens path!).
        let w = C64::from_polar(1.0, 2.0 * PI / 3.0);
        let inv_s = 1.0 / 3.0f64.sqrt();
        let dft: Vec<C64> = (0..3)
            .flat_map(|r| (0..3).map(move |c| w.powu((r * c) as u32) * inv_s))
            .collect();

        let mut c = ModeCircuit::new(3);
        c.set_input(0, ModeState::Fock(1)).unwrap();
        c.set_input(1, ModeState::Fock(1)).unwrap();
        c.set_input(2, ModeState::Fock(1)).unwrap();
        c.beam_splitter(0.3, 0.1, 0, 1).kerr(0.5, 0);
        c.passive(&[0, 1, 2], dft).unwrap();

        let st = FixedNFockState::from_circuit(&c).unwrap();
        let rf = FockReference::run(&c, 4).unwrap();
        for pat in st.basis() {
            close_c(st.amplitude(&pat).unwrap(), rf.amplitude(&pat), 1e-11);
        }
    }

    #[test]
    fn pnr_measurement_and_feedforward_sampling() {
        // HOM: |1, 1⟩ on 50:50 BS -> always outputs (2, 0) or (0, 2), never (1, 1).
        // Feed-forward: measure mode 0; if 2 photons, mode 1 has 0; rotate mode 1 by m.
        let mut c = ModeCircuit::new(2);
        c.set_input(0, ModeState::Fock(1)).unwrap();
        c.set_input(1, ModeState::Fock(1)).unwrap();
        c.beam_splitter(FRAC_PI_4, 0.0, 0, 1);
        let m0 = c.measure_photons(0);
        c.rotation(0.5 * m0, 1);
        c.measure_photons(1);

        let mut rng = StdRng::seed_from_u64(42);
        let shots = 2000;
        let samples = FixedNFockState::sample(&c, shots, &mut rng).unwrap();
        let mut c20 = 0usize;
        let mut c02 = 0usize;
        for row in &samples {
            assert_eq!(row.len(), 2);
            if row[0] == 2.0 && row[1] == 0.0 {
                c20 += 1;
            } else if row[0] == 0.0 && row[1] == 2.0 {
                c02 += 1;
            } else {
                panic!("impossible HOM outcome {row:?}");
            }
        }
        assert_eq!(c20 + c02, shots);
        assert!(((c20 as f64 / shots as f64) - 0.5).abs() < 0.05);
    }
}
