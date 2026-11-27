//! Universal cutoff Fock-space simulators:
//! - [`FockState`]: pure statevector (`cutoff^m` amplitudes) supporting all
//!   Gaussian and non-Gaussian gates, all 4 measurement types (homodyne,
//!   heterodyne, photon counting, threshold) with mid-circuit feed-forward,
//!   Monte Carlo noise trajectories, and exact truncation-error reporting.
//! - [`FockDensityMatrix`]: exact mixed-state density matrix (`cutoff^m × cutoff^m`)
//!   supporting thermal inputs, all gates, and exact CPTP CV noise channels
//!   (`Loss`, `ThermalLoss`, `Dephasing`, `AdditiveNoise`).

use super::circuit::{Measurement, ModeCircuit, ModeOp};
use super::fixed_n::givens_decompose;
use super::fock_ops::{
    beam_splitter_blocks, cached_quad_eigen, cubic_phase_matrix, displacement_matrix,
    hermite_fock_wavefunctions, quadratic_phase_matrix, squeezing_matrix,
    two_mode_squeezing_blocks,
};
use super::gate::{ModeGate, TwoModeGate};
use super::noise::{
    CvChannel, amp_kraus_diags, apply_amp_kraus_vec, apply_loss_kraus_vec, loss_kraus_diags,
};
use super::state::ModeState;
use crate::error::Error;
use faer::Mat;
use num_complex::Complex64 as C64;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;

/// Minimum statevector length for parallelising gate loops with rayon.
const PAR_MIN_LEN: usize = 4096;

/// Minimum shot count for parallelising multi-shot sampling.
const SAMPLE_PAR_MIN_SHOTS: usize = 64;

/// Universal cutoff Fock-space pure statevector (`cutoff^m` amplitudes).
///
/// Mode `0` is the least significant base-`cutoff` digit of an index, matching
/// the qubit side's ordering.
#[derive(Clone, Debug)]
pub struct FockState {
    n_modes: usize,
    cutoff: usize,
    amps: Vec<C64>,
}

impl FockState {
    /// Checked `cutoff^n_modes`.
    fn checked_dim(n_modes: usize, cutoff: usize) -> Result<usize, Error> {
        if cutoff == 0 {
            return Err(Error::InvalidArgument(
                "Fock cutoff must be at least 1".into(),
            ));
        }
        let mut dim = 1usize;
        for _ in 0..n_modes {
            dim = dim.checked_mul(cutoff).ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "Fock Hilbert space dimension {cutoff}^{n_modes} overflows usize"
                ))
            })?;
        }
        Ok(dim)
    }

    /// Construct directly from raw amplitudes (used internally by `WindowSampler`).
    pub(crate) fn from_raw(n_modes: usize, cutoff: usize, amps: Vec<C64>) -> Self {
        debug_assert_eq!(amps.len(), cutoff.pow(n_modes as u32));
        FockState {
            n_modes,
            cutoff,
            amps,
        }
    }

    /// Apply a unitary `ModeOp` with evaluated feed-forward results (used by `WindowSampler`).
    pub(crate) fn apply_op_with_results(
        &mut self,
        op: &ModeOp,
        results: &[f64],
    ) -> Result<(), Error> {
        self.apply_unitary_op(op, results)
    }

    /// The `n_modes`-mode vacuum `|0…0⟩` with Fock cutoff `cutoff`.
    pub fn vacuum(n_modes: usize, cutoff: usize) -> Result<Self, Error> {
        let dim = Self::checked_dim(n_modes, cutoff)?;
        let mut amps = vec![C64::new(0.0, 0.0); dim];
        amps[0] = C64::new(1.0, 0.0);
        Ok(FockState {
            n_modes,
            cutoff,
            amps,
        })
    }

    /// Build initial product state from pure `ModeState` inputs.
    fn from_inputs(inputs: &[ModeState], cutoff: usize) -> Result<Self, Error> {
        let n_modes = inputs.len();
        let dim = Self::checked_dim(n_modes, cutoff)?;
        let per_mode: Vec<Vec<C64>> = inputs
            .iter()
            .enumerate()
            .map(|(m, s)| {
                s.fock_amplitudes(cutoff).ok_or_else(|| {
                    Error::InvalidArgument(format!(
                        "input on mode {m} is mixed ({s:?}); use FockDensityMatrix"
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut amps = vec![C64::new(0.0, 0.0); dim];
        // Build tensor product iteratively: O(dim) instead of O(m * dim).
        amps[0] = C64::new(1.0, 0.0);
        let mut cur_len = 1usize;
        for pm in &per_mode {
            // Expand [0..cur_len] by outer product with `pm[0..cutoff]`.
            for n in (0..cutoff).rev() {
                let c_n = pm[n];
                let dst_off = n * cur_len;
                for i in 0..cur_len {
                    amps[dst_off + i] = amps[i] * c_n;
                }
            }
            cur_len *= cutoff;
        }
        Ok(FockState {
            n_modes,
            cutoff,
            amps,
        })
    }

    /// Run a bound, measurement-free circuit with Fock cutoff `cutoff`.
    pub fn from_circuit(circuit: &ModeCircuit, cutoff: usize) -> Result<Self, Error> {
        if circuit.n_parameters() > 0 {
            return Err(Error::InvalidArgument(
                "circuit has unbound parameters; bind it first".into(),
            ));
        }
        if !circuit.is_unitary() {
            return Err(Error::InvalidArgument(
                "circuit has measurements; use FockState::run or FockState::sample".into(),
            ));
        }
        let mut state = Self::from_inputs(circuit.inputs(), cutoff)?;
        for op in circuit.ops() {
            state.apply_unitary_op(op, &[])?;
        }
        Ok(state)
    }

    /// Run one shot of a bound circuit (with arbitrary measurements and
    /// feed-forward), returning the final state and measurement results.
    pub fn run<R: Rng + ?Sized>(
        circuit: &ModeCircuit,
        cutoff: usize,
        rng: &mut R,
    ) -> Result<(Self, Vec<f64>), Error> {
        if circuit.n_parameters() > 0 {
            return Err(Error::InvalidArgument(
                "circuit has unbound parameters; bind it first".into(),
            ));
        }
        let mut state = Self::from_inputs(circuit.inputs(), cutoff)?;
        let mut results = Vec::with_capacity(circuit.n_results());
        state.run_ops(circuit.ops(), &mut results, rng)?;
        Ok((state, results))
    }

    /// Sample `shots` independent measurement trajectories from a bound circuit.
    ///
    /// Runs the unitary prefix before the first measurement once; uses a
    /// single-CDF fast path when all measurements are terminal PNR / threshold
    /// on distinct modes.
    pub fn sample<R: Rng + ?Sized>(
        circuit: &ModeCircuit,
        cutoff: usize,
        shots: usize,
        rng: &mut R,
    ) -> Result<Vec<Vec<f64>>, Error> {
        if circuit.n_parameters() > 0 {
            return Err(Error::InvalidArgument(
                "circuit has unbound parameters; bind it first".into(),
            ));
        }
        if shots == 0 {
            return Ok(Vec::new());
        }
        let ops = circuit.ops();
        let first_stochastic = ops
            .iter()
            .position(|op| matches!(op, ModeOp::Measure { .. } | ModeOp::Channel { .. }))
            .unwrap_or(ops.len());

        let mut prefix = Self::from_inputs(circuit.inputs(), cutoff)?;
        for op in &ops[..first_stochastic] {
            prefix.apply_unitary_op(op, &[])?;
        }
        let suffix = &ops[first_stochastic..];
        if suffix.is_empty() {
            return Ok(vec![Vec::new(); shots]);
        }

        // Fast path: terminal PNR / threshold on distinct modes.
        if let Some(spec) = terminal_pnr_plan(suffix, prefix.n_modes) {
            return Ok(prefix.sample_terminal_pnr(&spec, shots, rng));
        }

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
                        let c = self.measure_threshold(*mode, rng)?;
                        results.push(if c { 1.0 } else { 0.0 });
                    }
                    Measurement::Homodyne { phi } => {
                        let angle = phi.evaluate(results)?;
                        let q = self.measure_homodyne(*mode, angle, rng)?;
                        results.push(q);
                    }
                    Measurement::Heterodyne => {
                        let (re, im) = self.measure_heterodyne(*mode, rng)?;
                        results.push(re);
                        results.push(im);
                    }
                },
                ModeOp::Channel { channel, mode } => {
                    self.apply_channel(channel, *mode, rng)?;
                }
                other => self.apply_unitary_op(other, results)?,
            }
        }
        Ok(())
    }

    fn apply_unitary_op(&mut self, op: &ModeOp, results: &[f64]) -> Result<(), Error> {
        match op {
            ModeOp::Gate { gate, mode } => match *gate {
                ModeGate::Rotation { phi } => self.rotation(phi.evaluate(results)?, *mode),
                ModeGate::Displacement { r, phi } => {
                    self.displace(r.evaluate(results)?, phi.evaluate(results)?, *mode)
                }
                ModeGate::Squeezing { r, phi } => {
                    self.squeeze(r.evaluate(results)?, phi.evaluate(results)?, *mode)
                }
                ModeGate::QuadraticPhase { s } => self.quadratic_phase(s.evaluate(results)?, *mode),
                ModeGate::CubicPhase { gamma } => self.cubic_phase(gamma.evaluate(results)?, *mode),
                ModeGate::Kerr { kappa } => self.kerr(kappa.evaluate(results)?, *mode),
            },
            ModeOp::TwoModeGate { gate, modes } => {
                let [a, b] = *modes;
                match *gate {
                    TwoModeGate::BeamSplitter { theta, phi } => {
                        self.beam_splitter(theta.evaluate(results)?, phi.evaluate(results)?, a, b)
                    }
                    TwoModeGate::TwoModeSqueezing { r, phi } => {
                        self.two_mode_squeeze(r.evaluate(results)?, phi.evaluate(results)?, a, b)
                    }
                    TwoModeGate::ControlledX { s } => self.cx(s.evaluate(results)?, a, b),
                    TwoModeGate::ControlledZ { s } => self.cz(s.evaluate(results)?, a, b),
                    TwoModeGate::CrossKerr { kappa } => {
                        self.cross_kerr(kappa.evaluate(results)?, a, b)
                    }
                }
            }
            ModeOp::Passive { modes, matrix } => self.apply_passive(modes, matrix),
            ModeOp::Channel { .. } => Err(Error::InvalidArgument(
                "circuit has noise channels; use FockState::run, FockState::sample, or FockDensityMatrix::from_circuit".into(),
            )),
            ModeOp::Measure { .. } => Err(Error::InvalidArgument(
                "unexpected measurement in unitary execution".into(),
            )),
        }
    }

    // -----------------------------------------------------------------------
    // Accessors, error metrics, & observables
    // -----------------------------------------------------------------------

    /// Number of modes.
    pub fn n_modes(&self) -> usize {
        self.n_modes
    }

    /// Fock cutoff per mode.
    pub fn cutoff(&self) -> usize {
        self.cutoff
    }

    /// Raw amplitudes of length `cutoff^n_modes` (mode 0 least significant).
    pub fn amplitudes(&self) -> &[C64] {
        &self.amps
    }

    #[inline]
    fn stride(&self, mode: usize) -> usize {
        self.cutoff.pow(mode as u32)
    }

    #[inline]
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

    #[inline]
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

    /// Index of occupation pattern `pattern` (mode 0 least significant).
    pub fn index(&self, pattern: &[usize]) -> Result<usize, Error> {
        if pattern.len() != self.n_modes {
            return Err(Error::InvalidArgument(format!(
                "pattern has {} modes, expected {}",
                pattern.len(),
                self.n_modes
            )));
        }
        let mut idx = 0usize;
        let mut stride = 1usize;
        for &n in pattern {
            if n >= self.cutoff {
                return Err(Error::InvalidArgument(format!(
                    "photon number {n} exceeds cutoff {}",
                    self.cutoff
                )));
            }
            idx += n * stride;
            stride *= self.cutoff;
        }
        Ok(idx)
    }

    /// Amplitude `⟨pattern|ψ⟩`.
    pub fn amplitude(&self, pattern: &[usize]) -> Result<C64, Error> {
        Ok(self.amps[self.index(pattern)?])
    }

    /// Probability `|⟨pattern|ψ⟩|²`.
    pub fn probability(&self, pattern: &[usize]) -> Result<f64, Error> {
        Ok(self.amplitude(pattern)?.norm_sqr())
    }

    /// All basis probabilities `|c_i|²`.
    pub fn probabilities(&self) -> Vec<f64> {
        if self.amps.len() >= PAR_MIN_LEN {
            self.amps.par_iter().map(|z| z.norm_sqr()).collect()
        } else {
            self.amps.iter().map(|z| z.norm_sqr()).collect()
        }
    }

    /// Squared norm `‖ψ‖²`. Deviations below `1.0` measure probability leaked
    /// above the Fock cutoff.
    pub fn norm_sqr(&self) -> f64 {
        self.amps.iter().map(|z| z.norm_sqr()).sum()
    }

    /// Probability leaked outside the Fock cutoff: `(1 − ‖ψ‖²).max(0)`.
    pub fn truncation_error(&self) -> f64 {
        (1.0 - self.norm_sqr()).max(0.0)
    }

    /// Total probability mass sitting on the highest Fock level `cutoff - 1`
    /// of any mode, indicating proximity to truncation.
    pub fn boundary_weight(&self) -> f64 {
        let d = self.cutoff;
        if d <= 1 {
            return self.norm_sqr();
        }
        let top = d - 1;
        self.amps
            .iter()
            .enumerate()
            .filter(|&(mut idx, _)| {
                for _ in 0..self.n_modes {
                    if idx % d == top {
                        return true;
                    }
                    idx /= d;
                }
                false
            })
            .map(|(_, z)| z.norm_sqr())
            .sum()
    }

    /// `⟨n̂⟩` on `mode`.
    pub fn mean_photons(&self, mode: usize) -> Result<f64, Error> {
        self.check_mode(mode)?;
        let d = self.cutoff;
        let stride = self.stride(mode);
        let block = stride * d;
        let n_outer = self.amps.len() / block;
        let mut total = 0.0;
        for outer in 0..n_outer {
            let base = outer * block;
            for n in 1..d {
                let nf = n as f64;
                let off = base + n * stride;
                let w: f64 = self.amps[off..off + stride]
                    .iter()
                    .map(|z| z.norm_sqr())
                    .sum();
                total += nf * w;
            }
        }
        Ok(total)
    }

    /// `⟨a⟩` on `mode`.
    fn mean_a(&self, mode: usize) -> Result<C64, Error> {
        self.check_mode(mode)?;
        let d = self.cutoff;
        let stride = self.stride(mode);
        let block = stride * d;
        let n_outer = self.amps.len() / block;
        let mut total = C64::new(0.0, 0.0);
        for outer in 0..n_outer {
            let base = outer * block;
            for n in 1..d {
                let s = (n as f64).sqrt();
                let hi = base + n * stride;
                let lo = hi - stride;
                let mut dot = C64::new(0.0, 0.0);
                for inner in 0..stride {
                    dot += self.amps[lo + inner].conj() * self.amps[hi + inner];
                }
                total += dot * s;
            }
        }
        Ok(total)
    }

    /// `⟨x̂⟩ = 2 Re⟨a⟩` on `mode`.
    pub fn mean_x(&self, mode: usize) -> Result<f64, Error> {
        Ok(2.0 * self.mean_a(mode)?.re)
    }

    /// `⟨p̂⟩ = 2 Im⟨a⟩` on `mode`.
    pub fn mean_p(&self, mode: usize) -> Result<f64, Error> {
        Ok(2.0 * self.mean_a(mode)?.im)
    }

    // -----------------------------------------------------------------------
    // Single-mode, two-mode, and multi-mode gates
    // -----------------------------------------------------------------------

    /// Rotation `R(φ) = exp(iφ n̂)` on `mode` (diagonal `O(d^m)`).
    pub fn rotation(&mut self, phi: f64, mode: usize) -> Result<(), Error> {
        self.check_mode(mode)?;
        if phi == 0.0 {
            return Ok(());
        }
        let phases: Vec<C64> = (0..self.cutoff)
            .map(|n| C64::from_polar(1.0, phi * (n as f64)))
            .collect();
        self.apply_diagonal_1(mode, &phases);
        Ok(())
    }

    /// Kerr `K(κ) = exp(iκ n̂²)` on `mode` (diagonal `O(d^m)`).
    pub fn kerr(&mut self, kappa: f64, mode: usize) -> Result<(), Error> {
        self.check_mode(mode)?;
        if kappa == 0.0 {
            return Ok(());
        }
        let phases: Vec<C64> = (0..self.cutoff)
            .map(|n| C64::from_polar(1.0, kappa * ((n * n) as f64)))
            .collect();
        self.apply_diagonal_1(mode, &phases);
        Ok(())
    }

    /// Displacement `D(r e^{iφ})` on `mode`.
    pub fn displace(&mut self, r: f64, phi: f64, mode: usize) -> Result<(), Error> {
        self.check_mode(mode)?;
        if r == 0.0 {
            return Ok(());
        }
        let u = displacement_matrix(C64::from_polar(r, phi), self.cutoff);
        self.apply_dense_1(mode, &u);
        Ok(())
    }

    /// Squeezing `S(r e^{iφ})` on `mode`.
    pub fn squeeze(&mut self, r: f64, phi: f64, mode: usize) -> Result<(), Error> {
        self.check_mode(mode)?;
        if r == 0.0 {
            return Ok(());
        }
        let u = squeezing_matrix(r, phi, self.cutoff);
        self.apply_dense_1(mode, &u);
        Ok(())
    }

    /// Quadratic phase `P(s)` on `mode`.
    pub fn quadratic_phase(&mut self, s: f64, mode: usize) -> Result<(), Error> {
        self.check_mode(mode)?;
        if s == 0.0 {
            return Ok(());
        }
        let u = quadratic_phase_matrix(s, self.cutoff);
        self.apply_dense_1(mode, &u);
        Ok(())
    }

    /// Cubic phase `V(γ)` on `mode`.
    pub fn cubic_phase(&mut self, gamma: f64, mode: usize) -> Result<(), Error> {
        self.check_mode(mode)?;
        if gamma == 0.0 {
            return Ok(());
        }
        let u = cubic_phase_matrix(gamma, self.cutoff);
        self.apply_dense_1(mode, &u);
        Ok(())
    }

    /// Cross-Kerr `CK(κ) = exp(iκ n̂_a n̂_b)` on `(a, b)` (diagonal `O(d^m)`).
    pub fn cross_kerr(&mut self, kappa: f64, a: usize, b: usize) -> Result<(), Error> {
        self.check_pair(a, b)?;
        if kappa == 0.0 {
            return Ok(());
        }
        let d = self.cutoff;
        let sa = self.stride(a);
        let sb = self.stride(b);
        let phases: Vec<C64> = (0..d * d)
            .map(|ij| {
                let (na, nb) = (ij / d, ij % d);
                C64::from_polar(1.0, kappa * ((na * nb) as f64))
            })
            .collect();
        if self.amps.len() >= PAR_MIN_LEN {
            self.amps.par_iter_mut().enumerate().for_each(|(idx, z)| {
                let na = (idx / sa) % d;
                let nb = (idx / sb) % d;
                if na > 0 && nb > 0 {
                    *z *= phases[na * d + nb];
                }
            });
        } else {
            for (idx, z) in self.amps.iter_mut().enumerate() {
                let na = (idx / sa) % d;
                let nb = (idx / sb) % d;
                if na > 0 && nb > 0 {
                    *z *= phases[na * d + nb];
                }
            }
        }
        Ok(())
    }

    /// Beam splitter `BS(θ, φ)` on `(a, b)` using exact photon-number-conserving
    /// `(s + 1) × (s + 1)` blocks (`O(d^{m+1})` instead of `O(d^{m+2})`).
    pub fn beam_splitter(&mut self, theta: f64, phi: f64, a: usize, b: usize) -> Result<(), Error> {
        self.check_pair(a, b)?;
        if theta == 0.0 {
            return Ok(());
        }
        let blocks = beam_splitter_blocks(theta, phi, self.cutoff);
        self.apply_sum_conserving_2(a, b, &blocks);
        Ok(())
    }

    /// Two-mode squeezing `S₂(r e^{iφ})` on `(a, b)` using exact
    /// photon-difference-conserving blocks (`O(d^{m+1})` instead of `O(d^{m+2})`).
    pub fn two_mode_squeeze(&mut self, r: f64, phi: f64, a: usize, b: usize) -> Result<(), Error> {
        self.check_pair(a, b)?;
        if r == 0.0 {
            return Ok(());
        }
        let blocks = two_mode_squeezing_blocks(r, phi, self.cutoff);
        self.apply_diff_conserving_2(a, b, &blocks);
        Ok(())
    }

    /// `CX(s) = exp(−i s x̂_a p̂_b / 2)` on `(a, b)` via factored padded
    /// quadrature eigenbases (`O(d^{m-2} · d · big²)`).
    pub fn cx(&mut self, s: f64, a: usize, b: usize) -> Result<(), Error> {
        self.check_pair(a, b)?;
        if s == 0.0 {
            return Ok(());
        }
        self.apply_quad_product_2(a, b, s, true);
        Ok(())
    }

    /// `CZ(s) = exp(i s x̂_a x̂_b / 2)` on `(a, b)` via factored padded
    /// quadrature eigenbases.
    pub fn cz(&mut self, s: f64, a: usize, b: usize) -> Result<(), Error> {
        self.check_pair(a, b)?;
        if s == 0.0 {
            return Ok(());
        }
        self.apply_quad_product_2(a, b, s, false);
        Ok(())
    }

    /// Passive linear-optics unitary `U` on `modes`, decomposed via exact
    /// Givens rotations into `O(k²)` sparse 2-mode blocks + `k` rotations.
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
        if k == 0 {
            return Ok(());
        }
        if k == 1 {
            return self.rotation(matrix[0].arg(), modes[0]);
        }
        let (ops, diag_phases) = givens_decompose(matrix, k);
        for (i, &phi) in diag_phases.iter().enumerate() {
            if phi.abs() > 1e-15 {
                self.rotation(phi, modes[i])?;
            }
        }
        for (i, j, u2) in ops.into_iter().rev() {
            let blocks = super::fixed_n::build_two_mode_blocks(&u2, 2 * (self.cutoff - 1));
            self.apply_sum_conserving_2(modes[i], modes[j], &blocks);
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Fast Fock gate application kernels
    // -----------------------------------------------------------------------

    fn apply_diagonal_1(&mut self, mode: usize, phases: &[C64]) {
        let d = self.cutoff;
        let stride = self.stride(mode);
        let block = stride * d;
        if self.amps.len() >= PAR_MIN_LEN {
            self.amps.par_chunks_exact_mut(block).for_each(|chunk| {
                for (n, &ph) in phases.iter().enumerate() {
                    if ph != C64::new(1.0, 0.0) {
                        for z in &mut chunk[n * stride..(n + 1) * stride] {
                            *z *= ph;
                        }
                    }
                }
            });
        } else {
            for chunk in self.amps.chunks_exact_mut(block) {
                for (n, &ph) in phases.iter().enumerate() {
                    if ph != C64::new(1.0, 0.0) {
                        for z in &mut chunk[n * stride..(n + 1) * stride] {
                            *z *= ph;
                        }
                    }
                }
            }
        }
    }

    fn apply_dense_1(&mut self, mode: usize, u: &[C64]) {
        let d = self.cutoff;
        let stride = self.stride(mode);
        let block = stride * d;
        if stride == 1 {
            // Mode 0: each chunk of length d is contiguous -> matrix-vector (or GEMM)
            let n_blocks = self.amps.len() / d;
            if n_blocks >= 4 && self.amps.len() >= 1024 {
                let u_mat = Mat::from_fn(d, d, |r, c| u[r * d + c]);
                let a_mat = Mat::from_fn(d, n_blocks, |r, c| self.amps[c * d + r]);
                let res = &u_mat * &a_mat;
                for c in 0..n_blocks {
                    for r in 0..d {
                        self.amps[c * d + r] = res[(r, c)];
                    }
                }
            } else {
                let mut tmp = vec![C64::new(0.0, 0.0); d];
                for chunk in self.amps.chunks_exact_mut(d) {
                    tmp.copy_from_slice(chunk);
                    for r in 0..d {
                        let row = &u[r * d..(r + 1) * d];
                        let mut acc = C64::new(0.0, 0.0);
                        for c in 0..d {
                            acc += row[c] * tmp[c];
                        }
                        chunk[r] = acc;
                    }
                }
            }
        } else if self.amps.len() >= PAR_MIN_LEN {
            self.amps.par_chunks_exact_mut(block).for_each(|chunk| {
                apply_strided_chunk_1(chunk, u, d, stride);
            });
        } else {
            for chunk in self.amps.chunks_exact_mut(block) {
                apply_strided_chunk_1(chunk, u, d, stride);
            }
        }
    }

    /// Apply a photon-number-conserving 2-mode gate on `(a, b)` from its
    /// `(s + 1) × (s + 1)` blocks (`s = 0..=2(d - 1)`).
    fn apply_sum_conserving_2(&mut self, a: usize, b: usize, blocks: &[Vec<C64>]) {
        let d = self.cutoff;
        let sa = self.stride(a);
        let sb = self.stride(b);
        let bases = two_mode_bases(self.amps.len(), d, sa, sb);
        let mut buf_in = vec![C64::new(0.0, 0.0); d];
        for base in bases {
            for (s, blk) in blocks.iter().enumerate() {
                if s == 0 {
                    continue;
                }
                let s1 = s + 1;
                let lo = s.saturating_sub(d - 1);
                let hi = s.min(d - 1);
                // Active range of n_a is lo..=hi (length hi - lo + 1 <= d).
                let count = hi - lo + 1;
                for (i, na) in (lo..=hi).enumerate() {
                    let nb = s - na;
                    buf_in[i] = self.amps[base + na * sa + nb * sb];
                }
                for out_na in lo..=hi {
                    let out_nb = s - out_na;
                    let row = out_na * s1;
                    let mut acc = C64::new(0.0, 0.0);
                    for (i, &val) in buf_in[..count].iter().enumerate() {
                        let in_na = lo + i;
                        acc += blk[row + in_na] * val;
                    }
                    self.amps[base + out_na * sa + out_nb * sb] = acc;
                }
            }
        }
    }

    /// Apply a photon-difference-conserving 2-mode gate on `(a, b)` from its
    /// `2d - 1` blocks (`diff = n_a - n_b ∈ [-(d - 1), d - 1]`).
    fn apply_diff_conserving_2(&mut self, a: usize, b: usize, blocks: &[Vec<C64>]) {
        let d = self.cutoff;
        let sa = self.stride(a);
        let sb = self.stride(b);
        let bases = two_mode_bases(self.amps.len(), d, sa, sb);
        let mut buf_in = vec![C64::new(0.0, 0.0); d];
        for base in bases {
            for (idx, blk) in blocks.iter().enumerate() {
                let diff = idx as isize - (d as isize - 1);
                let oa = diff.max(0) as usize;
                let ob = (-diff).max(0) as usize;
                let len = d - diff.unsigned_abs();
                for k in 0..len {
                    buf_in[k] = self.amps[base + (k + oa) * sa + (k + ob) * sb];
                }
                for out in 0..len {
                    let row = out * len;
                    let mut acc = C64::new(0.0, 0.0);
                    for inp in 0..len {
                        acc += blk[row + inp] * buf_in[inp];
                    }
                    self.amps[base + (out + oa) * sa + (out + ob) * sb] = acc;
                }
            }
        }
    }

    /// Apply `CX(s)` (`is_cx = true`) or `CZ(s)` (`is_cx = false`) on `(a, b)`
    /// using the cached padded quadrature eigenbases:
    /// `M' = W_a (Φ ⊙ (W_a† M W_b*)) W_b^T`.
    fn apply_quad_product_2(&mut self, a: usize, b: usize, s: f64, is_cx: bool) {
        let d = self.cutoff;
        let sa = self.stride(a);
        let sb = self.stride(b);
        let qe = cached_quad_eigen(d);
        let big = qe.big;

        let wa = Mat::from_fn(d, big, |r, c| C64::new(qe.vx[r * big + c], 0.0));
        let wb = if is_cx {
            Mat::from_fn(d, big, |r, c| qe.vp[r * big + c])
        } else {
            wa.clone()
        };
        let wa_dag = wa.adjoint().to_owned();
        let wb_conj = Mat::from_fn(d, big, |r, c| wb[(r, c)].conj());
        let wb_t = wb.transpose().to_owned();

        let sign = if is_cx { -0.5 * s } else { 0.5 * s };
        let phi_mat = Mat::from_fn(big, big, |k, l| {
            C64::from_polar(1.0, sign * qe.evals[k] * qe.evals[l])
        });

        let bases = two_mode_bases(self.amps.len(), d, sa, sb);
        for base in bases {
            let m_in = Mat::from_fn(d, d, |na, nb| self.amps[base + na * sa + nb * sb]);
            // tmp = W_a† M W_b*  (big × big)
            let t1 = &wa_dag * &m_in; // big × d
            let mut t2 = &t1 * &wb_conj; // big × big
            for k in 0..big {
                for l in 0..big {
                    t2[(k, l)] *= phi_mat[(k, l)];
                }
            }
            // out = W_a t2 W_b^T  (d × d)
            let t3 = &wa * &t2; // d × big
            let out = &t3 * &wb_t; // d × d
            for na in 0..d {
                for nb in 0..d {
                    self.amps[base + na * sa + nb * sb] = out[(na, nb)];
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Measurements (destructive, reset measured mode to vacuum)
    // -----------------------------------------------------------------------

    /// Destructive photon-number measurement on `mode`: samples `k`, projects
    /// onto `|k⟩`, resets `mode` to vacuum `|0⟩`, and returns `k`.
    pub fn measure_photons<R: Rng + ?Sized>(
        &mut self,
        mode: usize,
        rng: &mut R,
    ) -> Result<usize, Error> {
        self.check_mode(mode)?;
        let d = self.cutoff;
        let stride = self.stride(mode);
        let block = stride * d;
        let n_outer = self.amps.len() / block;

        let mut probs = vec![0.0f64; d];
        for outer in 0..n_outer {
            let base = outer * block;
            for (n, p_slot) in probs.iter_mut().enumerate() {
                let off = base + n * stride;
                *p_slot += self.amps[off..off + stride]
                    .iter()
                    .map(|z| z.norm_sqr())
                    .sum::<f64>();
            }
        }
        let total: f64 = probs.iter().sum();
        let u = rng.random::<f64>() * total;
        let mut cum = 0.0;
        let last_nonzero = probs.iter().rposition(|&p| p > 0.0).unwrap_or(0);
        let mut outcome = last_nonzero;
        for (k, &p) in probs.iter().enumerate() {
            cum += p;
            if u < cum {
                outcome = k;
                break;
            }
        }
        let inv_norm = if probs[outcome] > 0.0 {
            1.0 / probs[outcome].sqrt()
        } else {
            0.0
        };

        // Move slice `n = outcome` to `n = 0` and zero all other levels.
        for outer in 0..n_outer {
            let base = outer * block;
            let src = base + outcome * stride;
            for inner in 0..stride {
                let v = self.amps[src + inner] * inv_norm;
                self.amps[base + inner] = v;
            }
            for n in 1..d {
                let off = base + n * stride;
                self.amps[off..off + stride].fill(C64::new(0.0, 0.0));
            }
        }
        Ok(outcome)
    }

    /// Destructive threshold measurement on `mode` (exact trajectory
    /// unraveling).
    pub fn measure_threshold<R: Rng + ?Sized>(
        &mut self,
        mode: usize,
        rng: &mut R,
    ) -> Result<bool, Error> {
        Ok(self.measure_photons(mode, rng)? > 0)
    }

    /// Destructive homodyne measurement of `x̂ cos φ + p̂ sin φ` on `mode`.
    /// Samples `q` from the exact marginal quadrature PDF on a fine grid,
    /// conditions the remaining modes on `⟨q|`, and resets `mode` to vacuum.
    pub fn measure_homodyne<R: Rng + ?Sized>(
        &mut self,
        mode: usize,
        phi: f64,
        rng: &mut R,
    ) -> Result<f64, Error> {
        self.check_mode(mode)?;
        if phi != 0.0 {
            self.rotation(-phi, mode)?;
        }
        let d = self.cutoff;
        let stride = self.stride(mode);
        let block = stride * d;
        let n_outer = self.amps.len() / block;

        // Compute the reduced 1-mode density matrix ρ_m (d × d) of `mode`
        // so evaluating p(q) = ψ(q)^T ρ_m ψ(q) costs only O(d²) per grid point!
        let mut rho_m = vec![C64::new(0.0, 0.0); d * d];
        for outer in 0..n_outer {
            let base = outer * block;
            for n1 in 0..d {
                let s1 = &self.amps[base + n1 * stride..base + (n1 + 1) * stride];
                for n2 in n1..d {
                    let s2 = &self.amps[base + n2 * stride..base + (n2 + 1) * stride];
                    let mut acc = C64::new(0.0, 0.0);
                    for i in 0..stride {
                        acc += s1[i] * s2[i].conj();
                    }
                    rho_m[n1 * d + n2] += acc;
                }
            }
        }
        for n1 in 0..d {
            for n2 in (n1 + 1)..d {
                rho_m[n2 * d + n1] = rho_m[n1 * d + n2].conj();
            }
        }

        // Classical turning point of level d is √(4d) (for ħ = 2); add margin 5.
        let q_max = (4.0 * (d as f64)).sqrt() + 5.0;
        let n_grid = (200 + 24 * d).min(2048);
        let dq = 2.0 * q_max / (n_grid as f64);

        let mut psi = vec![0.0f64; d];
        let mut cdf = Vec::with_capacity(n_grid);
        let mut cum = 0.0;
        for g in 0..n_grid {
            let q = -q_max + (g as f64 + 0.5) * dq;
            hermite_fock_wavefunctions(q, d, &mut psi);
            let mut pdf = 0.0;
            for n1 in 0..d {
                let p1 = psi[n1];
                pdf += p1 * p1 * rho_m[n1 * d + n1].re;
                for n2 in (n1 + 1)..d {
                    pdf += 2.0 * p1 * psi[n2] * rho_m[n1 * d + n2].re;
                }
            }
            cum += pdf.max(0.0) * dq;
            cdf.push(cum);
        }
        let u = rng.random::<f64>() * cum;
        let idx = cdf.partition_point(|&c| c < u).min(n_grid - 1);
        let frac = rng.random::<f64>();
        let outcome = -q_max + (idx as f64 + frac) * dq;

        // Project onto ⟨outcome| on `mode` and reset `mode` to vacuum.
        hermite_fock_wavefunctions(outcome, d, &mut psi);
        let mut norm_sqr = 0.0;
        for outer in 0..n_outer {
            let base = outer * block;
            for inner in 0..stride {
                let mut acc = C64::new(0.0, 0.0);
                for (n, &w) in psi.iter().enumerate() {
                    acc += self.amps[base + n * stride + inner] * w;
                }
                self.amps[base + inner] = acc;
                norm_sqr += acc.norm_sqr();
            }
            for n in 1..d {
                let off = base + n * stride;
                self.amps[off..off + stride].fill(C64::new(0.0, 0.0));
            }
        }
        if norm_sqr > 0.0 {
            let inv = 1.0 / norm_sqr.sqrt();
            for outer in 0..n_outer {
                let base = outer * block;
                for z in &mut self.amps[base..base + stride] {
                    *z *= inv;
                }
            }
        }
        Ok(outcome)
    }

    /// Destructive heterodyne measurement on `mode`, returning `(Re α, Im α)`.
    /// Samples `α` via exact Husimi Q-function radial/2D rejection sampling,
    /// conditions on `⟨α|`, and resets `mode` to vacuum.
    pub fn measure_heterodyne<R: Rng + ?Sized>(
        &mut self,
        mode: usize,
        rng: &mut R,
    ) -> Result<(f64, f64), Error> {
        self.check_mode(mode)?;
        let d = self.cutoff;
        let stride = self.stride(mode);
        let block = stride * d;
        let n_outer = self.amps.len() / block;

        // Reduced density matrix ρ_m of `mode`.
        let mut rho_m = vec![C64::new(0.0, 0.0); d * d];
        for outer in 0..n_outer {
            let base = outer * block;
            for n1 in 0..d {
                let s1 = &self.amps[base + n1 * stride..base + (n1 + 1) * stride];
                for n2 in n1..d {
                    let s2 = &self.amps[base + n2 * stride..base + (n2 + 1) * stride];
                    let mut acc = C64::new(0.0, 0.0);
                    for i in 0..stride {
                        acc += s1[i] * s2[i].conj();
                    }
                    rho_m[n1 * d + n2] += acc;
                }
            }
        }
        for n1 in 0..d {
            for n2 in (n1 + 1)..d {
                rho_m[n2 * d + n1] = rho_m[n1 * d + n2].conj();
            }
        }

        // Proposal distribution: 2D Gaussian centred at ⟨α⟩ with variance
        // σ² = max(1, 0.5 * (⟨n⟩ + 1)) -> covers the support of Q(α) tightly.
        // For exact sampling over the bounded support of cutoff `d`, compute
        // Q(α) on a polar / Cartesian grid around ⟨α⟩ with radius √d + 4.
        let mean_n: f64 = (0..d).map(|n| (n as f64) * rho_m[n * d + n].re).sum();
        let mean_a: C64 = (1..d)
            .map(|n| rho_m[n * d + (n - 1)] * (n as f64).sqrt())
            .sum();
        let rad = (mean_n + (d as f64)).sqrt() + 4.0;
        let n_side = 80usize;
        let step = 2.0 * rad / (n_side as f64);

        let mut coh = vec![C64::new(0.0, 0.0); d];
        let mut cdf = Vec::with_capacity(n_side * n_side);
        let mut cum = 0.0;
        for ix in 0..n_side {
            let re = mean_a.re - rad + (ix as f64 + 0.5) * step;
            for iy in 0..n_side {
                let im = mean_a.im - rad + (iy as f64 + 0.5) * step;
                let alpha = C64::new(re, im);
                coherent_vec(alpha, &mut coh);
                // ⟨α|ρ_m|α⟩ = Σ_{n1, n2} coh[n1]* rho_m[n1, n2] coh[n2]
                let mut val = 0.0;
                for n1 in 0..d {
                    let c1 = coh[n1].conj();
                    val += (c1 * coh[n1]).re * rho_m[n1 * d + n1].re;
                    for n2 in (n1 + 1)..d {
                        val += 2.0 * (c1 * rho_m[n1 * d + n2] * coh[n2]).re;
                    }
                }
                cum += val.max(0.0);
                cdf.push(cum);
            }
        }
        let u = rng.random::<f64>() * cum;
        let cell = cdf.partition_point(|&c| c < u).min(n_side * n_side - 1);
        let ix = cell / n_side;
        let iy = cell % n_side;
        let re = mean_a.re - rad + (ix as f64 + rng.random::<f64>()) * step;
        let im = mean_a.im - rad + (iy as f64 + rng.random::<f64>()) * step;
        let alpha = C64::new(re, im);

        // Project onto ⟨α| and reset `mode` to vacuum.
        coherent_vec(alpha, &mut coh);
        let mut norm_sqr = 0.0;
        for outer in 0..n_outer {
            let base = outer * block;
            for inner in 0..stride {
                let mut acc = C64::new(0.0, 0.0);
                for (n, &cn) in coh.iter().enumerate() {
                    acc += cn.conj() * self.amps[base + n * stride + inner];
                }
                self.amps[base + inner] = acc;
                norm_sqr += acc.norm_sqr();
            }
            for n in 1..d {
                let off = base + n * stride;
                self.amps[off..off + stride].fill(C64::new(0.0, 0.0));
            }
        }
        if norm_sqr > 0.0 {
            let inv = 1.0 / norm_sqr.sqrt();
            for outer in 0..n_outer {
                let base = outer * block;
                for z in &mut self.amps[base..base + stride] {
                    *z *= inv;
                }
            }
        }
        Ok((re, im))
    }

    // -----------------------------------------------------------------------
    // Monte Carlo quantum trajectories for opt-in CV noise
    // -----------------------------------------------------------------------

    /// Apply one stochastic Kraus / unraveling jump of `channel` on `mode`
    /// (for Monte Carlo quantum trajectories on pure statevectors).
    pub fn apply_channel<R: Rng + ?Sized>(
        &mut self,
        channel: &CvChannel,
        mode: usize,
        rng: &mut R,
    ) -> Result<(), Error> {
        self.check_mode(mode)?;
        channel.validate()?;
        match *channel {
            CvChannel::Loss { eta } => self.apply_loss_trajectory(mode, eta, rng),
            CvChannel::ThermalLoss { eta, mean_photons } => {
                // Exact decomposition: ThermalLoss(η, n̄) = Amp(g) ∘ Loss(η / g)
                // with g = 1 + (1 - η) n̄.
                let g = 1.0 + (1.0 - eta) * mean_photons;
                self.apply_loss_trajectory(mode, eta / g, rng);
                if g > 1.0 + 1e-15 {
                    self.apply_amp_trajectory(mode, g, rng);
                }
            }
            CvChannel::Dephasing { sigma } => {
                if sigma > 0.0 {
                    let z: f64 = StandardNormal.sample(rng);
                    self.rotation(sigma * z, mode)?;
                }
            }
            CvChannel::AdditiveNoise { variance } => {
                if variance > 0.0 {
                    // Add N(0, variance) to both x and p => Δα = ½(dx + i dp).
                    let sd = variance.sqrt();
                    let zx: f64 = StandardNormal.sample(rng);
                    let zp: f64 = StandardNormal.sample(rng);
                    let alpha = C64::new(0.5 * sd * zx, 0.5 * sd * zp);
                    self.displace(alpha.norm(), alpha.arg(), mode)?;
                }
            }
        }
        Ok(())
    }

    fn apply_loss_trajectory<R: Rng + ?Sized>(&mut self, mode: usize, eta: f64, rng: &mut R) {
        let d = self.cutoff;
        let stride = self.stride(mode);
        let ops = loss_kraus_diags(d, eta);
        let mut scratch = vec![C64::new(0.0, 0.0); self.amps.len()];
        let mut probs = vec![0.0f64; d];
        for (l, diag) in ops.iter().enumerate() {
            apply_loss_kraus_vec(&self.amps, &mut scratch, d, stride, l, diag);
            probs[l] = scratch.iter().map(|z| z.norm_sqr()).sum();
        }
        let total: f64 = probs.iter().sum();
        let u = rng.random::<f64>() * total;
        let mut cum = 0.0;
        let mut chosen = 0usize;
        for (l, &p) in probs.iter().enumerate() {
            cum += p;
            if u < cum {
                chosen = l;
                break;
            }
        }
        apply_loss_kraus_vec(&self.amps, &mut scratch, d, stride, chosen, &ops[chosen]);
        let inv = if probs[chosen] > 0.0 {
            1.0 / probs[chosen].sqrt()
        } else {
            0.0
        };
        for z in &mut scratch {
            *z *= inv;
        }
        self.amps = scratch;
    }

    fn apply_amp_trajectory<R: Rng + ?Sized>(&mut self, mode: usize, g: f64, rng: &mut R) {
        let d = self.cutoff;
        let stride = self.stride(mode);
        let ops = amp_kraus_diags(d, g);
        let mut scratch = vec![C64::new(0.0, 0.0); self.amps.len()];
        let mut probs = vec![0.0f64; d];
        for (l, diag) in ops.iter().enumerate() {
            apply_amp_kraus_vec(&self.amps, &mut scratch, d, stride, l, diag);
            probs[l] = scratch.iter().map(|z| z.norm_sqr()).sum();
        }
        let total: f64 = probs.iter().sum();
        let u = rng.random::<f64>() * total;
        let mut cum = 0.0;
        let mut chosen = 0usize;
        for (l, &p) in probs.iter().enumerate() {
            cum += p;
            if u < cum {
                chosen = l;
                break;
            }
        }
        apply_amp_kraus_vec(&self.amps, &mut scratch, d, stride, chosen, &ops[chosen]);
        let inv = if probs[chosen] > 0.0 {
            1.0 / probs[chosen].sqrt()
        } else {
            0.0
        };
        for z in &mut scratch {
            *z *= inv;
        }
        self.amps = scratch;
    }

    fn sample_terminal_pnr<R: Rng + ?Sized>(
        &self,
        spec: &[(usize, bool)],
        shots: usize,
        rng: &mut R,
    ) -> Vec<Vec<f64>> {
        let d = self.cutoff;
        let dim = self.amps.len();
        let mut cdf = Vec::with_capacity(dim);
        let mut acc = 0.0;
        for z in &self.amps {
            acc += z.norm_sqr();
            cdf.push(acc);
        }
        let total = acc;
        let strides: Vec<usize> = spec.iter().map(|&(m, _)| self.stride(m)).collect();
        let draw_one = |r: f64| -> Vec<f64> {
            let idx = cdf.partition_point(|&c| c <= r * total).min(dim - 1);
            spec.iter()
                .zip(&strides)
                .map(|(&(_, is_thresh), &s)| {
                    let occ = (idx / s) % d;
                    if is_thresh {
                        if occ > 0 { 1.0 } else { 0.0 }
                    } else {
                        occ as f64
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
// FockDensityMatrix: exact mixed-state cutoff Fock density matrix
// ---------------------------------------------------------------------------

/// Mixed-state cutoff Fock density matrix (`dim × dim`, row-major, where
/// `dim = cutoff^n_modes`).
///
/// Supports thermal inputs, all Gaussian and non-Gaussian gates, and exact
/// CPTP noise channels (`Loss`, `ThermalLoss`, `Dephasing`, `AdditiveNoise`).
#[derive(Clone, Debug)]
pub struct FockDensityMatrix {
    n_modes: usize,
    cutoff: usize,
    dim: usize,
    rho: Vec<C64>,
}

impl FockDensityMatrix {
    /// Vacuum density matrix `|0…0⟩⟨0…0|` on `n_modes` modes with Fock cutoff
    /// `cutoff`.
    pub fn vacuum(n_modes: usize, cutoff: usize) -> Result<Self, Error> {
        let dim = FockState::checked_dim(n_modes, cutoff)?;
        let entries = dim.checked_mul(dim).ok_or_else(|| {
            Error::InvalidArgument(format!(
                "FockDensityMatrix dimension {dim}×{dim} overflows usize"
            ))
        })?;
        let mut rho = vec![C64::new(0.0, 0.0); entries];
        rho[0] = C64::new(1.0, 0.0);
        Ok(FockDensityMatrix {
            n_modes,
            cutoff,
            dim,
            rho,
        })
    }

    /// Construct `|ψ⟩⟨ψ|` from a pure [`FockState`].
    pub fn from_state(state: &FockState) -> Self {
        let dim = state.amps.len();
        let mut rho = vec![C64::new(0.0, 0.0); dim * dim];
        for r in 0..dim {
            let ar = state.amps[r];
            for c in 0..dim {
                rho[r * dim + c] = ar * state.amps[c].conj();
            }
        }
        FockDensityMatrix {
            n_modes: state.n_modes,
            cutoff: state.cutoff,
            dim,
            rho,
        }
    }

    /// Run a bound, measurement-free circuit (including thermal inputs) on a
    /// cutoff Fock density matrix.
    pub fn from_circuit(circuit: &ModeCircuit, cutoff: usize) -> Result<Self, Error> {
        if circuit.n_parameters() > 0 {
            return Err(Error::InvalidArgument(
                "circuit has unbound parameters; bind it first".into(),
            ));
        }
        if circuit.has_measurements() {
            return Err(Error::InvalidArgument(
                "FockDensityMatrix::from_circuit requires a measurement-free circuit".into(),
            ));
        }
        let n_modes = circuit.n_modes();
        let dim = FockState::checked_dim(n_modes, cutoff)?;
        let entries = dim.checked_mul(dim).ok_or_else(|| {
            Error::InvalidArgument(format!(
                "FockDensityMatrix dimension {dim}×{dim} overflows usize"
            ))
        })?;

        // Build tensor product of 1-mode input density matrices.
        let mode_rhos: Vec<Vec<C64>> = circuit
            .inputs()
            .iter()
            .map(|s| single_mode_rho(s, cutoff))
            .collect();

        let mut rho = vec![C64::new(0.0, 0.0); entries];
        let mut strides = vec![1usize; n_modes];
        for m in 1..n_modes {
            strides[m] = strides[m - 1] * cutoff;
        }
        for r in 0..dim {
            for c in 0..dim {
                let mut prod = C64::new(1.0, 0.0);
                for m in 0..n_modes {
                    let rm = (r / strides[m]) % cutoff;
                    let cm = (c / strides[m]) % cutoff;
                    prod *= mode_rhos[m][rm * cutoff + cm];
                    if prod.re == 0.0 && prod.im == 0.0 {
                        break;
                    }
                }
                rho[r * dim + c] = prod;
            }
        }

        let mut dm = FockDensityMatrix {
            n_modes,
            cutoff,
            dim,
            rho,
        };
        for op in circuit.ops() {
            dm.apply_op(op)?;
        }
        Ok(dm)
    }

    fn apply_op(&mut self, op: &ModeOp) -> Result<(), Error> {
        // Apply any unitary gate U via ρ → U ρ U† by applying U to every
        // column of ρ, then U* to every row of ρ!
        // Because FockState already implements every single- and two-mode gate
        // with fast block-sparse / diagonal / recurrence kernels, we reuse
        // FockState's kernels column-by-column!
        match op {
            ModeOp::Gate { .. } | ModeOp::TwoModeGate { .. } | ModeOp::Passive { .. } => {
                let dim = self.dim;
                let mut tmp = vec![C64::new(0.0, 0.0); dim * dim];
                // Step 1: apply U to each column c of ρ -> store transposed as row c of tmp.
                for c in 0..dim {
                    let mut st = FockState {
                        n_modes: self.n_modes,
                        cutoff: self.cutoff,
                        amps: (0..dim).map(|r| self.rho[r * dim + c]).collect(),
                    };
                    st.apply_unitary_op(op, &[])?;
                    tmp[c * dim..(c + 1) * dim].copy_from_slice(&st.amps);
                }
                // Now tmp[c, r] = (U ρ)_{r, c}, so row r of (U ρ)† is tmp[:, r]*.
                // Step 2: apply U to each column r of (U ρ)† -> gives column r of U (U ρ)† = (U ρ U†)† = U ρ U†!
                for r in 0..dim {
                    let mut st = FockState {
                        n_modes: self.n_modes,
                        cutoff: self.cutoff,
                        amps: (0..dim).map(|c| tmp[c * dim + r].conj()).collect(),
                    };
                    st.apply_unitary_op(op, &[])?;
                    for c in 0..dim {
                        self.rho[r * dim + c] = st.amps[c].conj();
                    }
                }
                Ok(())
            }
            ModeOp::Channel { channel, mode } => self.apply_channel(channel, *mode),
            ModeOp::Measure { .. } => Err(Error::InvalidArgument(
                "measurements are not supported in FockDensityMatrix::from_circuit".into(),
            )),
        }
    }

    /// Apply an exact CPTP CV noise channel on `mode`.
    pub fn apply_channel(&mut self, channel: &CvChannel, mode: usize) -> Result<(), Error> {
        if mode >= self.n_modes {
            return Err(Error::InvalidArgument(format!(
                "mode {mode} out of range for {} modes",
                self.n_modes
            )));
        }
        channel.validate()?;
        match *channel {
            CvChannel::Loss { eta } => self.apply_loss_exact(mode, eta),
            CvChannel::ThermalLoss { eta, mean_photons } => {
                let g = 1.0 + (1.0 - eta) * mean_photons;
                self.apply_loss_exact(mode, eta / g);
                if g > 1.0 + 1e-15 {
                    self.apply_amp_exact(mode, g);
                }
            }
            CvChannel::Dephasing { sigma } => {
                if sigma == 0.0 {
                    return Ok(());
                }
                let d = self.cutoff;
                let stride = d.pow(mode as u32);
                let factor = -0.5 * sigma * sigma;
                let dim = self.dim;
                for r in 0..dim {
                    let nr = (r / stride) % d;
                    for c in 0..dim {
                        let nc = (c / stride) % d;
                        if nr != nc {
                            let diff = (nr as f64) - (nc as f64);
                            self.rho[r * dim + c] *= (factor * diff * diff).exp();
                        }
                    }
                }
            }
            CvChannel::AdditiveNoise { variance } => {
                if variance > 0.0 {
                    let g = 1.0 + 0.5 * variance;
                    self.apply_loss_exact(mode, 1.0 / g);
                    self.apply_amp_exact(mode, g);
                }
            }
        }
        Ok(())
    }

    fn apply_loss_exact(&mut self, mode: usize, eta: f64) {
        let d = self.cutoff;
        let stride = d.pow(mode as u32);
        let dim = self.dim;
        let ops = loss_kraus_diags(d, eta);
        let mut out = vec![C64::new(0.0, 0.0); dim * dim];
        for (l, diag) in ops.iter().enumerate() {
            for r in 0..dim {
                let nr = (r / stride) % d;
                if nr + l >= d {
                    continue;
                }
                let wr = diag[nr];
                if wr == 0.0 {
                    continue;
                }
                let src_r = r + l * stride;
                for c in 0..dim {
                    let nc = (c / stride) % d;
                    if nc + l >= d {
                        continue;
                    }
                    let wc = diag[nc];
                    if wc == 0.0 {
                        continue;
                    }
                    let src_c = c + l * stride;
                    out[r * dim + c] += self.rho[src_r * dim + src_c] * (wr * wc);
                }
            }
        }
        self.rho = out;
    }

    fn apply_amp_exact(&mut self, mode: usize, g: f64) {
        let d = self.cutoff;
        let stride = d.pow(mode as u32);
        let dim = self.dim;
        let ops = amp_kraus_diags(d, g);
        let mut out = vec![C64::new(0.0, 0.0); dim * dim];
        for (l, diag) in ops.iter().enumerate() {
            for r in 0..dim {
                let nr = (r / stride) % d;
                if nr < l {
                    continue;
                }
                let src_nr = nr - l;
                let wr = diag[src_nr];
                if wr == 0.0 {
                    continue;
                }
                let src_r = r - l * stride;
                for c in 0..dim {
                    let nc = (c / stride) % d;
                    if nc < l {
                        continue;
                    }
                    let src_nc = nc - l;
                    let wc = diag[src_nc];
                    if wc == 0.0 {
                        continue;
                    }
                    let src_c = c - l * stride;
                    out[r * dim + c] += self.rho[src_r * dim + src_c] * (wr * wc);
                }
            }
        }
        self.rho = out;
    }

    /// Number of modes.
    pub fn n_modes(&self) -> usize {
        self.n_modes
    }

    /// Fock cutoff per mode.
    pub fn cutoff(&self) -> usize {
        self.cutoff
    }

    /// Hilbert space dimension `cutoff^n_modes`.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Density matrix entries (`dim × dim`, row-major).
    pub fn matrix(&self) -> &[C64] {
        &self.rho
    }

    /// Trace `Tr(ρ)`.
    pub fn trace(&self) -> f64 {
        (0..self.dim).map(|i| self.rho[i * self.dim + i].re).sum()
    }

    /// Probability leaked outside the Fock cutoff: `(1 − Tr(ρ)).max(0)`.
    pub fn truncation_error(&self) -> f64 {
        (1.0 - self.trace()).max(0.0)
    }

    /// Purity `Tr(ρ²)`.
    pub fn purity(&self) -> f64 {
        // Since ρ is Hermitian, Tr(ρ²) = Σ_{r, c} |ρ_{r, c}|².
        self.rho.iter().map(|z| z.norm_sqr()).sum()
    }

    /// Diagonal Fock-basis probabilities `ρ_{ii}`.
    pub fn probabilities(&self) -> Vec<f64> {
        (0..self.dim)
            .map(|i| self.rho[i * self.dim + i].re.max(0.0))
            .collect()
    }

    /// Probability `⟨pattern|ρ|pattern⟩`.
    pub fn probability(&self, pattern: &[usize]) -> Result<f64, Error> {
        if pattern.len() != self.n_modes {
            return Err(Error::InvalidArgument(format!(
                "pattern has {} modes, expected {}",
                pattern.len(),
                self.n_modes
            )));
        }
        let mut idx = 0usize;
        let mut stride = 1usize;
        for &n in pattern {
            if n >= self.cutoff {
                return Err(Error::InvalidArgument(format!(
                    "photon number {n} exceeds cutoff {}",
                    self.cutoff
                )));
            }
            idx += n * stride;
            stride *= self.cutoff;
        }
        Ok(self.rho[idx * self.dim + idx].re.max(0.0))
    }

    /// Mean photon number `Tr(n̂ ρ)` on `mode`.
    pub fn mean_photons(&self, mode: usize) -> Result<f64, Error> {
        if mode >= self.n_modes {
            return Err(Error::InvalidArgument("mode out of range".into()));
        }
        let d = self.cutoff;
        let stride = d.pow(mode as u32);
        let mut total = 0.0;
        for i in 0..self.dim {
            let n = (i / stride) % d;
            if n > 0 {
                total += (n as f64) * self.rho[i * self.dim + i].re;
            }
        }
        Ok(total)
    }

    fn mean_a(&self, mode: usize) -> Result<C64, Error> {
        if mode >= self.n_modes {
            return Err(Error::InvalidArgument("mode out of range".into()));
        }
        let d = self.cutoff;
        let stride = d.pow(mode as u32);
        let mut total = C64::new(0.0, 0.0);
        for i in 0..self.dim {
            let n = (i / stride) % d;
            if n > 0 {
                // Tr(a ρ) = Σ ⟨n-1|a|n⟩ ⟨n|ρ|n-1⟩ = Σ √n ρ[i, i - stride]
                total += self.rho[i * self.dim + (i - stride)] * (n as f64).sqrt();
            }
        }
        Ok(total)
    }

    /// `⟨x̂⟩` on `mode`.
    pub fn mean_x(&self, mode: usize) -> Result<f64, Error> {
        Ok(2.0 * self.mean_a(mode)?.re)
    }

    /// `⟨p̂⟩` on `mode`.
    pub fn mean_p(&self, mode: usize) -> Result<f64, Error> {
        Ok(2.0 * self.mean_a(mode)?.im)
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn apply_strided_chunk_1(chunk: &mut [C64], u: &[C64], d: usize, stride: usize) {
    let mut out = vec![C64::new(0.0, 0.0); d * stride];
    for r in 0..d {
        let dst = &mut out[r * stride..(r + 1) * stride];
        let u_row = &u[r * d..(r + 1) * d];
        for (c, &uc) in u_row.iter().enumerate() {
            if uc.re == 0.0 && uc.im == 0.0 {
                continue;
            }
            let src = &chunk[c * stride..(c + 1) * stride];
            for i in 0..stride {
                dst[i] += uc * src[i];
            }
        }
    }
    chunk.copy_from_slice(&out);
}

/// All base indices in `0..len` where digits of modes `a` and `b` are both 0.
fn two_mode_bases(len: usize, d: usize, sa: usize, sb: usize) -> Vec<usize> {
    let mut bases = Vec::with_capacity(len / (d * d));
    for idx in 0..len {
        if (idx / sa).is_multiple_of(d) && (idx / sb).is_multiple_of(d) {
            bases.push(idx);
        }
    }
    bases
}

fn coherent_vec(alpha: C64, out: &mut [C64]) {
    if out.is_empty() {
        return;
    }
    out[0] = C64::new((-0.5 * alpha.norm_sqr()).exp(), 0.0);
    for n in 1..out.len() {
        out[n] = out[n - 1] * alpha / (n as f64).sqrt();
    }
}

fn single_mode_rho(state: &ModeState, cutoff: usize) -> Vec<C64> {
    let mut rho = vec![C64::new(0.0, 0.0); cutoff * cutoff];
    match *state {
        ModeState::Thermal { mean_photons } if mean_photons > 0.0 => {
            let p0 = 1.0 / (mean_photons + 1.0);
            let ratio = mean_photons / (mean_photons + 1.0);
            let mut p = p0;
            for n in 0..cutoff {
                rho[n * cutoff + n] = C64::new(p, 0.0);
                p *= ratio;
            }
        }
        _ => {
            let c = state.fock_amplitudes(cutoff).unwrap();
            for r in 0..cutoff {
                for col in 0..cutoff {
                    rho[r * cutoff + col] = c[r] * c[col].conj();
                }
            }
        }
    }
    rho
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
    use crate::cv::GaussianState;
    use crate::cv::reference::FockReference;
    use std::f64::consts::FRAC_PI_4;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() < tol, "{a} vs {b} (diff {})", (a - b).abs());
    }

    fn close_c(a: C64, b: C64, tol: f64) {
        assert!((a - b).norm() < tol, "{a} vs {b}");
    }

    #[test]
    fn universal_fock_circuit_matches_reference() {
        // Exercises every gate family: Gaussian + non-Gaussian (Kerr, CrossKerr, CubicPhase).
        let mut c = ModeCircuit::new(2);
        c.set_input(
            0,
            ModeState::Cat {
                r: 0.7,
                phi: 0.2,
                parity: 0.0,
            },
        )
        .unwrap()
        .set_input(1, ModeState::Fock(1))
        .unwrap()
        .squeeze(0.25, 0.4, 0)
        .displace(0.3, -0.5, 1)
        .beam_splitter(0.6, 0.3, 0, 1)
        .two_mode_squeeze(0.2, -0.7, 0, 1)
        .kerr(0.35, 0)
        .cross_kerr(-0.25, 0, 1)
        .quadratic_phase(0.2, 0)
        .cubic_phase(0.05, 1)
        .cx(0.2, 0, 1)
        .cz(-0.15, 0, 1)
        .rotation(0.5, 0);

        let d = 16;
        let st = FockState::from_circuit(&c, d).unwrap();
        let rf = FockReference::run(&c, d).unwrap();
        assert!(
            st.truncation_error() < 1e-4,
            "trunc={}",
            st.truncation_error()
        );
        for (a, b) in st
            .amplitudes()
            .iter()
            .zip((0..d * d).map(|idx| rf.amplitude(&[idx % d, idx / d])))
        {
            close_c(*a, b, 1e-9);
        }
        for m in 0..2 {
            close(st.mean_x(m).unwrap(), rf.mean_x(m), 1e-9);
            close(st.mean_p(m).unwrap(), rf.mean_p(m), 1e-9);
            close(st.mean_photons(m).unwrap(), rf.mean_photons(m), 1e-9);
        }
    }

    #[test]
    fn fock_gaussian_cross_check_with_noise() {
        // Displaced squeezed state + thermal loss + additive noise:
        // cross-check FockDensityMatrix against exact GaussianState!
        let mut c = ModeCircuit::new(1);
        c.squeeze(0.35, 0.4, 0).displace(0.5, -0.3, 0);

        let mut g = GaussianState::from_circuit(&c).unwrap();
        let mut dm = FockDensityMatrix::from_circuit(&c, 28).unwrap();
        assert!(dm.truncation_error() < 1e-9);

        let th = CvChannel::ThermalLoss {
            eta: 0.7,
            mean_photons: 0.4,
        };
        let add = CvChannel::AdditiveNoise { variance: 0.3 };
        g.apply_channel(&th, 0).unwrap();
        g.apply_channel(&add, 0).unwrap();
        dm.apply_channel(&th, 0).unwrap();
        dm.apply_channel(&add, 0).unwrap();

        assert!(dm.truncation_error() < 1e-6);
        close(dm.mean_x(0).unwrap(), g.mean_x(0), 1e-5);
        close(dm.mean_p(0).unwrap(), g.mean_p(0), 1e-5);
        close(dm.mean_photons(0).unwrap(), g.mean_photons(0), 1e-5);
        close(dm.purity(), g.purity(), 1e-5);
        for n in 0..6 {
            close(
                dm.probability(&[n]).unwrap(),
                g.fock_probability(&[n]).unwrap(),
                1e-5,
            );
        }

        // Dephasing kills off-diagonals and preserves diagonal photon distribution.
        let before = dm.probabilities();
        dm.apply_channel(&CvChannel::Dephasing { sigma: 0.8 }, 0)
            .unwrap();
        let after = dm.probabilities();
        for (a, b) in before.iter().zip(&after) {
            close(*a, *b, 1e-12);
        }
        // Coherent mean decays by exp(-σ² / 2).
        close(
            dm.mean_x(0).unwrap(),
            g.mean_x(0) * (-0.5 * 0.8f64.powi(2)).exp(),
            1e-5,
        );
    }

    #[test]
    fn fock_trajectories_match_density_matrix() {
        let mut c = ModeCircuit::new(1);
        c.set_input(0, ModeState::Fock(2)).unwrap();
        let ch = CvChannel::ThermalLoss {
            eta: 0.6,
            mean_photons: 0.3,
        };
        let mut dm = FockDensityMatrix::from_circuit(&c, 16).unwrap();
        dm.apply_channel(&ch, 0).unwrap();
        let want_n = dm.mean_photons(0).unwrap();

        let mut rng = StdRng::seed_from_u64(101);
        let shots = 4000;
        let mut sum_n = 0.0;
        for _ in 0..shots {
            let mut st = FockState::from_circuit(&c, 16).unwrap();
            st.apply_channel(&ch, 0, &mut rng).unwrap();
            sum_n += st.mean_photons(0).unwrap();
        }
        close(sum_n / shots as f64, want_n, 0.04);
    }

    #[test]
    fn fock_homodyne_and_heterodyne_match_gaussian_moments() {
        let mut c = ModeCircuit::new(1);
        c.squeeze(0.4, 0.2, 0).displace(0.6, -0.5, 0);
        let g = GaussianState::from_circuit(&c).unwrap();

        let mut c_hom = c.clone();
        c_hom.measure_homodyne(0.3, 0);
        let mut rng = StdRng::seed_from_u64(77);
        let hom = FockState::sample(&c_hom, 24, 3000, &mut rng).unwrap();
        let mean_q: f64 = hom.iter().map(|r| r[0]).sum::<f64>() / hom.len() as f64;
        let want_q = g.mean_x(0) * 0.3f64.cos() + g.mean_p(0) * 0.3f64.sin();
        close(mean_q, want_q, 0.06);

        let mut c_het = c.clone();
        c_het.measure_heterodyne(0);
        let het = FockState::sample(&c_het, 20, 2000, &mut rng).unwrap();
        let mean_re: f64 = het.iter().map(|r| r[0]).sum::<f64>() / het.len() as f64;
        let mean_im: f64 = het.iter().map(|r| r[1]).sum::<f64>() / het.len() as f64;
        close(mean_re, 0.5 * g.mean_x(0), 0.06);
        close(mean_im, 0.5 * g.mean_p(0), 0.06);
    }

    #[test]
    fn fock_pnr_and_feedforward_match_fixed_n() {
        let mut c = ModeCircuit::new(2);
        c.set_input(0, ModeState::Fock(1)).unwrap();
        c.set_input(1, ModeState::Fock(1)).unwrap();
        c.beam_splitter(FRAC_PI_4, 0.0, 0, 1);
        let m0 = c.measure_photons(0);
        c.displace(0.25 * m0, 0.0, 1);
        let mut rng = StdRng::seed_from_u64(9);
        let (st, res) = FockState::run(&c, 12, &mut rng).unwrap();
        assert!(res[0] == 0.0 || res[0] == 2.0);
        if res[0] == 2.0 {
            // Mode 1 was |0⟩, displaced by 0.5 -> coherent |0.5⟩ with mean_x = 1.0!
            close(st.mean_x(1).unwrap(), 1.0, 1e-8);
        } else {
            // Mode 1 was |2⟩, displaced by 0.0 -> Fock |2⟩ with mean_photons = 2.0!
            close(st.mean_photons(1).unwrap(), 2.0, 1e-8);
        }
    }
}
