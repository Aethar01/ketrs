//! Bosonic Matrix Product State (MPS) simulator for large, weakly entangled
//! continuous-variable circuits with local Fock cutoff `d`.
//!
//! Uses mixed-canonical form with explicit orthogonality-center tracking,
//! `faer`-backed QR/SVD (via [`crate::mps::linalg`]), block-sparse
//! photon-number-conserving `BeamSplitter` / `TwoModeSqueezing` contractions,
//! factored quadrature-basis `CX` / `CZ` contractions, SWAP routing for
//! non-adjacent gates, and destructive PNR / threshold / homodyne / heterodyne
//! measurements with feed-forward.
//!
//! Reports cumulative truncation error (Fock input truncation + discarded SVD
//! weight `Σ s_discarded²`).

use super::circuit::{Measurement, ModeCircuit, ModeOp};
use super::fixed_n::{build_two_mode_blocks, givens_decompose};
use super::fock_ops::{
    beam_splitter_blocks, cached_quad_eigen, cubic_phase_matrix, displacement_matrix,
    hermite_fock_wavefunctions, quadratic_phase_matrix, squeezing_matrix,
    two_mode_squeezing_blocks,
};
use super::gate::{ModeGate, TwoModeGate};
use crate::error::Error;
use crate::mps::linalg::{matmul, thin_qr, thin_svd};
use faer::Mat;
use num_complex::Complex64 as C64;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

/// Minimum shot count to parallelise independent MPS trajectories.
const SAMPLE_PAR_MIN_SHOTS: usize = 16;

/// Truncation controls for [`BosonicMps`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BosonicMpsOptions {
    /// Maximum bond dimension `χ` retained on every bond (default `64`).
    pub max_bond: usize,
    /// Relative singular-value cutoff: singular values `< cutoff_tol * s_0`
    /// are discarded (default `1e-10`).
    pub cutoff_tol: f64,
}

impl Default for BosonicMpsOptions {
    fn default() -> Self {
        BosonicMpsOptions {
            max_bond: 64,
            cutoff_tol: 1e-10,
        }
    }
}

/// One site tensor `A[l, p, r]` with physical dimension `d = cutoff`,
/// stored in row-major `(l * d + p) * right + r` order.
#[derive(Clone, Debug)]
struct Site {
    left: usize,
    right: usize,
    data: Vec<C64>,
}

impl Site {
    fn vacuum(d: usize) -> Self {
        let mut data = vec![C64::new(0.0, 0.0); d];
        if d > 0 {
            data[0] = C64::new(1.0, 0.0);
        }
        Site {
            left: 1,
            right: 1,
            data,
        }
    }

    fn from_amplitudes(amps: Vec<C64>) -> Self {
        Site {
            left: 1,
            right: 1,
            data: amps,
        }
    }
}

/// Matrix Product State simulator for `m`-mode CV circuits at local Fock
/// cutoff `d`.
#[derive(Clone, Debug)]
pub struct BosonicMps {
    n_modes: usize,
    cutoff: usize,
    sites: Vec<Site>,
    center: usize,
    options: BosonicMpsOptions,
    input_truncation_error: f64,
    svd_truncation_error: f64,
}

impl BosonicMps {
    /// Vacuum state `|0…0⟩` on `n_modes` modes with Fock cutoff `cutoff`.
    pub fn vacuum(
        n_modes: usize,
        cutoff: usize,
        options: BosonicMpsOptions,
    ) -> Result<Self, Error> {
        if cutoff == 0 {
            return Err(Error::InvalidArgument("Fock cutoff must be >= 1".into()));
        }
        if options.max_bond == 0 {
            return Err(Error::InvalidArgument("max_bond must be >= 1".into()));
        }
        let sites = (0..n_modes).map(|_| Site::vacuum(cutoff)).collect();
        Ok(BosonicMps {
            n_modes,
            cutoff,
            sites,
            center: 0,
            options,
            input_truncation_error: 0.0,
            svd_truncation_error: 0.0,
        })
    }

    /// Build a rank-1 product MPS from `circuit.inputs()`.
    fn from_inputs(
        circuit: &ModeCircuit,
        cutoff: usize,
        options: BosonicMpsOptions,
    ) -> Result<Self, Error> {
        if cutoff == 0 {
            return Err(Error::InvalidArgument("Fock cutoff must be >= 1".into()));
        }
        if options.max_bond == 0 {
            return Err(Error::InvalidArgument("max_bond must be >= 1".into()));
        }
        let n_modes = circuit.n_modes();
        let mut sites = Vec::with_capacity(n_modes);
        let mut norm_prod = 1.0f64;
        for (mode, st) in circuit.inputs().iter().enumerate() {
            let amps = st.fock_amplitudes(cutoff).ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "mode {mode} input {st:?} is mixed; BosonicMps supports pure inputs"
                ))
            })?;
            let n2: f64 = amps.iter().map(|z| z.norm_sqr()).sum();
            norm_prod *= n2.min(1.0);
            sites.push(Site::from_amplitudes(amps));
        }
        Ok(BosonicMps {
            n_modes,
            cutoff,
            sites,
            center: 0,
            options,
            input_truncation_error: (1.0 - norm_prod).max(0.0),
            svd_truncation_error: 0.0,
        })
    }

    /// Run a bound, measurement-free `ModeCircuit` in the MPS representation.
    pub fn from_circuit(
        circuit: &ModeCircuit,
        cutoff: usize,
        options: BosonicMpsOptions,
    ) -> Result<Self, Error> {
        if circuit.n_parameters() > 0 {
            return Err(Error::InvalidArgument(
                "circuit has unbound parameters; bind it first".into(),
            ));
        }
        if !circuit.is_unitary() {
            return Err(Error::InvalidArgument(
                "circuit has measurements; use BosonicMps::run or BosonicMps::sample".into(),
            ));
        }
        let mut st = Self::from_inputs(circuit, cutoff, options)?;
        for op in circuit.ops() {
            st.apply_op(op, &[])?;
        }
        Ok(st)
    }

    /// Run one shot of a bound `ModeCircuit` (with measurements & feed-forward),
    /// returning `(final_state, results)`.
    pub fn run<R: Rng + ?Sized>(
        circuit: &ModeCircuit,
        cutoff: usize,
        options: BosonicMpsOptions,
        rng: &mut R,
    ) -> Result<(Self, Vec<f64>), Error> {
        if circuit.n_parameters() > 0 {
            return Err(Error::InvalidArgument(
                "circuit has unbound parameters; bind it first".into(),
            ));
        }
        let mut st = Self::from_inputs(circuit, cutoff, options)?;
        let mut results = Vec::with_capacity(circuit.n_results());
        st.run_ops(circuit.ops(), &mut results, rng)?;
        Ok((st, results))
    }

    /// Sample `shots` independent measurement trajectories from a bound
    /// `ModeCircuit`, sharing the unitary prefix before the first measurement.
    pub fn sample<R: Rng + ?Sized>(
        circuit: &ModeCircuit,
        cutoff: usize,
        options: BosonicMpsOptions,
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
        let first_meas = ops
            .iter()
            .position(|op| matches!(op, ModeOp::Measure { .. }))
            .unwrap_or(ops.len());

        let mut prefix = Self::from_inputs(circuit, cutoff, options)?;
        for op in &ops[..first_meas] {
            prefix.apply_op(op, &[])?;
        }
        let suffix = &ops[first_meas..];
        if suffix.is_empty() {
            return Ok(vec![Vec::new(); shots]);
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
                        let n = self.measure_photons(*mode, rng)?;
                        results.push(n as f64);
                    }
                    Measurement::Threshold => {
                        let click = self.measure_threshold(*mode, rng)?;
                        results.push(if click { 1.0 } else { 0.0 });
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
                other => self.apply_op(other, results)?,
            }
        }
        Ok(())
    }

    fn apply_op(&mut self, op: &ModeOp, results: &[f64]) -> Result<(), Error> {
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
                "BosonicMps does not support non-unitary noise channels".into(),
            )),
            ModeOp::Measure { .. } => Err(Error::InvalidArgument(
                "unexpected measurement in unitary execution".into(),
            )),
        }
    }

    // -----------------------------------------------------------------------
    // Accessors & observables
    // -----------------------------------------------------------------------

    /// Number of modes.
    pub fn n_modes(&self) -> usize {
        self.n_modes
    }

    /// Local Fock cutoff `d` per mode.
    pub fn cutoff(&self) -> usize {
        self.cutoff
    }

    /// Bond dimensions between adjacent modes (length `n_modes.saturating_sub(1)`).
    pub fn bond_dims(&self) -> Vec<usize> {
        if self.n_modes <= 1 {
            return Vec::new();
        }
        (0..self.n_modes - 1).map(|i| self.sites[i].right).collect()
    }

    /// Largest current bond dimension across all bonds.
    pub fn max_bond_dim(&self) -> usize {
        self.bond_dims().into_iter().max().unwrap_or(1)
    }

    /// Squared norm `⟨ψ|ψ⟩` computed in `O(χ² d)` from the orthogonality center.
    pub fn norm_sqr(&self) -> f64 {
        if self.n_modes == 0 {
            return 1.0;
        }
        self.sites[self.center]
            .data
            .iter()
            .map(|z| z.norm_sqr())
            .sum()
    }

    /// Cumulative truncation error bound: Fock leakage `(1 − ‖ψ‖²)` plus
    /// discarded SVD weight `Σ s_discarded²`.
    pub fn truncation_error(&self) -> f64 {
        let norm_loss = (1.0 - self.norm_sqr()).max(0.0);
        norm_loss.max(self.input_truncation_error) + self.svd_truncation_error
    }

    /// Exact Fock amplitude `⟨n_0, …, n_{m-1} | ψ⟩` in `O(m χ²)`.
    pub fn amplitude(&self, pattern: &[usize]) -> Result<C64, Error> {
        if pattern.len() != self.n_modes {
            return Err(Error::InvalidArgument(format!(
                "pattern length {} != n_modes {}",
                pattern.len(),
                self.n_modes
            )));
        }
        let d = self.cutoff;
        for &n in pattern {
            if n >= d {
                return Ok(C64::new(0.0, 0.0));
            }
        }
        if self.n_modes == 0 {
            return Ok(C64::new(1.0, 0.0));
        }
        let mut vec = vec![C64::new(1.0, 0.0)];
        for (site, &p) in self.sites.iter().zip(pattern) {
            let mut next = vec![C64::new(0.0, 0.0); site.right];
            for (l, &vl) in vec.iter().enumerate().take(site.left) {
                if vl == C64::new(0.0, 0.0) {
                    continue;
                }
                let base = (l * d + p) * site.right;
                for (r, dst) in next.iter_mut().enumerate().take(site.right) {
                    *dst += vl * site.data[base + r];
                }
            }
            vec = next;
        }
        Ok(vec[0])
    }

    /// Fock probability `|⟨n_0, …, n_{m-1} | ψ⟩|²`.
    pub fn probability(&self, pattern: &[usize]) -> Result<f64, Error> {
        Ok(self.amplitude(pattern)?.norm_sqr())
    }

    /// Mean photon number `⟨n̂⟩` on `mode`.
    pub fn mean_photons(&mut self, mode: usize) -> Result<f64, Error> {
        self.check_mode(mode)?;
        self.move_center(mode);
        let d = self.cutoff;
        let s = &self.sites[mode];
        let mut total = 0.0;
        for l in 0..s.left {
            for p in 1..d {
                let base = (l * d + p) * s.right;
                let w: f64 = s.data[base..base + s.right]
                    .iter()
                    .map(|z| z.norm_sqr())
                    .sum();
                total += (p as f64) * w;
            }
        }
        Ok(total)
    }

    /// `⟨x̂⟩ = ⟨a + a†⟩` on `mode`.
    pub fn mean_x(&mut self, mode: usize) -> Result<f64, Error> {
        self.check_mode(mode)?;
        Ok(2.0 * self.mean_a(mode).re)
    }

    /// `⟨p̂⟩ = ⟨−i(a − a†)⟩` on `mode`.
    pub fn mean_p(&mut self, mode: usize) -> Result<f64, Error> {
        self.check_mode(mode)?;
        Ok(2.0 * self.mean_a(mode).im)
    }

    fn mean_a(&mut self, mode: usize) -> C64 {
        self.move_center(mode);
        let d = self.cutoff;
        let s = &self.sites[mode];
        let mut acc = C64::new(0.0, 0.0);
        for l in 0..s.left {
            for p in 0..d.saturating_sub(1) {
                let base = (l * d + p) * s.right;
                let base_next = (l * d + (p + 1)) * s.right;
                let sq = ((p + 1) as f64).sqrt();
                for r in 0..s.right {
                    acc += s.data[base + r].conj() * s.data[base_next + r] * sq;
                }
            }
        }
        acc
    }

    // -----------------------------------------------------------------------
    // Canonicalisation
    // -----------------------------------------------------------------------

    fn move_center(&mut self, target: usize) {
        while self.center < target {
            self.shift_center_right();
        }
        while self.center > target {
            self.shift_center_left();
        }
    }

    fn shift_center_right(&mut self) {
        let i = self.center;
        let d = self.cutoff;
        let s = &self.sites[i];
        let rows = s.left * d;
        let cols = s.right;
        let a_mat = Mat::from_fn(rows, cols, |r, c| s.data[r * cols + c]);
        let (q, r_mat) = thin_qr(a_mat.as_ref());
        let new_bond = q.ncols();

        let mut q_data = vec![C64::new(0.0, 0.0); rows * new_bond];
        for r in 0..rows {
            for c in 0..new_bond {
                q_data[r * new_bond + c] = q[(r, c)];
            }
        }
        self.sites[i] = Site {
            left: s.left,
            right: new_bond,
            data: q_data,
        };

        let next = &self.sites[i + 1];
        let next_cols = d * next.right;
        let b_mat = Mat::from_fn(next.left, next_cols, |r, c| next.data[r * next_cols + c]);
        let prod = matmul(r_mat.as_ref(), b_mat.as_ref());
        let mut next_data = vec![C64::new(0.0, 0.0); new_bond * next_cols];
        for r in 0..new_bond {
            for c in 0..next_cols {
                next_data[r * next_cols + c] = prod[(r, c)];
            }
        }
        self.sites[i + 1] = Site {
            left: new_bond,
            right: next.right,
            data: next_data,
        };
        self.center = i + 1;
    }

    fn shift_center_left(&mut self) {
        let i = self.center;
        let d = self.cutoff;
        let s = &self.sites[i];
        let rows = s.left;
        let cols = d * s.right;
        // LQ decomposition of M (rows × cols) via QR of M^T (cols × rows):
        // M^T = Q R => M = R^T Q^T, where Q^T has orthonormal rows (right-isometric)!
        // Note: use conjugate-transpose M† = Q R => M = R† Q† so rows of Q† are
        // orthonormal in the Hermitian inner product: Q† (Q†)† = Q† Q = I!
        let m_adj = Mat::from_fn(cols, rows, |r, c| s.data[c * cols + r].conj());
        let (q, r_mat) = thin_qr(m_adj.as_ref());
        let new_bond = q.ncols();

        let mut qt_data = vec![C64::new(0.0, 0.0); new_bond * cols];
        for r in 0..new_bond {
            for c in 0..cols {
                qt_data[r * cols + c] = q[(c, r)].conj();
            }
        }
        self.sites[i] = Site {
            left: new_bond,
            right: s.right,
            data: qt_data,
        };

        let prev = &self.sites[i - 1];
        let prev_rows = prev.left * d;
        let a_mat = Mat::from_fn(prev_rows, prev.right, |r, c| prev.data[r * prev.right + c]);
        let r_adj = Mat::from_fn(prev.right, new_bond, |r, c| r_mat[(c, r)].conj());
        let prod = matmul(a_mat.as_ref(), r_adj.as_ref());
        let mut prev_data = vec![C64::new(0.0, 0.0); prev_rows * new_bond];
        for r in 0..prev_rows {
            for c in 0..new_bond {
                prev_data[r * new_bond + c] = prod[(r, c)];
            }
        }
        self.sites[i - 1] = Site {
            left: prev.left,
            right: new_bond,
            data: prev_data,
        };
        self.center = i - 1;
    }

    // -----------------------------------------------------------------------
    // Single-mode gates
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

    fn apply_diag_1(&mut self, mode: usize, phases: &[C64]) {
        let d = self.cutoff;
        let s = &mut self.sites[mode];
        for l in 0..s.left {
            for (p, &ph) in phases.iter().enumerate() {
                if ph != C64::new(1.0, 0.0) {
                    let base = (l * d + p) * s.right;
                    for z in &mut s.data[base..base + s.right] {
                        *z *= ph;
                    }
                }
            }
        }
    }

    fn apply_dense_1(&mut self, mode: usize, u: &[C64]) {
        // Move center to `mode` so cutoff truncation never breaks isometry on other sites.
        self.move_center(mode);
        let d = self.cutoff;
        let s = &mut self.sites[mode];
        let mut out = vec![C64::new(0.0, 0.0); s.data.len()];
        for l in 0..s.left {
            for out_p in 0..d {
                let u_row = &u[out_p * d..(out_p + 1) * d];
                let dst = &mut out[(l * d + out_p) * s.right..(l * d + out_p + 1) * s.right];
                for (in_p, &u_val) in u_row.iter().enumerate() {
                    if u_val == C64::new(0.0, 0.0) {
                        continue;
                    }
                    let src = &s.data[(l * d + in_p) * s.right..(l * d + in_p + 1) * s.right];
                    for (dst_z, &src_z) in dst.iter_mut().zip(src) {
                        *dst_z += u_val * src_z;
                    }
                }
            }
        }
        s.data = out;
    }

    /// Rotation `R(φ) = exp(i φ n̂)` on `mode`.
    pub fn rotation(&mut self, phi: f64, mode: usize) -> Result<(), Error> {
        self.check_mode(mode)?;
        if phi == 0.0 {
            return Ok(());
        }
        let phases: Vec<C64> = (0..self.cutoff)
            .map(|n| C64::from_polar(1.0, phi * (n as f64)))
            .collect();
        self.apply_diag_1(mode, &phases);
        Ok(())
    }

    /// Kerr `K(κ) = exp(i κ n̂²)` on `mode`.
    pub fn kerr(&mut self, kappa: f64, mode: usize) -> Result<(), Error> {
        self.check_mode(mode)?;
        if kappa == 0.0 {
            return Ok(());
        }
        let phases: Vec<C64> = (0..self.cutoff)
            .map(|n| {
                let nf = n as f64;
                C64::from_polar(1.0, kappa * nf * nf)
            })
            .collect();
        self.apply_diag_1(mode, &phases);
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

    // -----------------------------------------------------------------------
    // Two-mode gates (with automatic SWAP routing for non-adjacent pairs)
    // -----------------------------------------------------------------------

    /// Beam splitter `BS(θ, φ)` on `(a, b)`.
    pub fn beam_splitter(&mut self, theta: f64, phi: f64, a: usize, b: usize) -> Result<(), Error> {
        self.check_pair(a, b)?;
        let blocks = beam_splitter_blocks(theta, phi, self.cutoff);
        self.apply_two_mode_routed(a, b, |theta_mat, chi_l, chi_r, d, swapped| {
            apply_sum_blocks_theta(theta_mat, chi_l, chi_r, d, &blocks, swapped)
        });
        Ok(())
    }

    /// Two-mode squeezing `S₂(r e^{iφ})` on `(a, b)`.
    pub fn two_mode_squeeze(&mut self, r: f64, phi: f64, a: usize, b: usize) -> Result<(), Error> {
        self.check_pair(a, b)?;
        if r == 0.0 {
            return Ok(());
        }
        let blocks = two_mode_squeezing_blocks(r, phi, self.cutoff);
        self.apply_two_mode_routed(a, b, |theta_mat, chi_l, chi_r, d, swapped| {
            apply_diff_blocks_theta(theta_mat, chi_l, chi_r, d, &blocks, swapped)
        });
        Ok(())
    }

    /// `CX(s) = exp(−i s x̂_a p̂_b / 2)` on `(a, b)`.
    pub fn cx(&mut self, s: f64, a: usize, b: usize) -> Result<(), Error> {
        self.check_pair(a, b)?;
        if s == 0.0 {
            return Ok(());
        }
        self.apply_two_mode_routed(a, b, |theta_mat, chi_l, chi_r, d, swapped| {
            apply_quad_product_theta(theta_mat, chi_l, chi_r, d, s, true, swapped)
        });
        Ok(())
    }

    /// `CZ(s) = exp(i s x̂_a x̂_b / 2)` on `(a, b)`.
    pub fn cz(&mut self, s: f64, a: usize, b: usize) -> Result<(), Error> {
        self.check_pair(a, b)?;
        if s == 0.0 {
            return Ok(());
        }
        self.apply_two_mode_routed(a, b, |theta_mat, chi_l, chi_r, d, swapped| {
            apply_quad_product_theta(theta_mat, chi_l, chi_r, d, s, false, swapped)
        });
        Ok(())
    }

    /// Cross-Kerr `CK(κ) = exp(i κ n̂_a n̂_b)` on `(a, b)`.
    pub fn cross_kerr(&mut self, kappa: f64, a: usize, b: usize) -> Result<(), Error> {
        self.check_pair(a, b)?;
        if kappa == 0.0 {
            return Ok(());
        }
        self.apply_two_mode_routed(a, b, |theta_mat, chi_l, chi_r, d, _swapped| {
            for l in 0..chi_l {
                for pa in 0..d {
                    let row = l * d + pa;
                    for pb in 0..d {
                        let ph = C64::from_polar(1.0, kappa * ((pa * pb) as f64));
                        let col0 = pb * chi_r;
                        for r in 0..chi_r {
                            theta_mat[(row, col0 + r)] *= ph;
                        }
                    }
                }
            }
        });
        Ok(())
    }

    /// Passive linear-optics unitary `U` on `modes` via exact Givens
    /// decomposition into 2-mode blocks + 1-mode rotations.
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
            let blocks = build_two_mode_blocks(&u2, 2 * (self.cutoff - 1));
            self.apply_two_mode_routed(
                modes[i],
                modes[j],
                |theta_mat, chi_l, chi_r, d, swapped| {
                    apply_sum_blocks_theta(theta_mat, chi_l, chi_r, d, &blocks, swapped)
                },
            );
        }
        Ok(())
    }

    fn apply_two_mode_routed(
        &mut self,
        a: usize,
        b: usize,
        gate_fn: impl FnOnce(&mut Mat<C64>, usize, usize, usize, bool),
    ) {
        let (lo, hi, swapped) = if a < b { (a, b, false) } else { (b, a, true) };
        // Route `hi` down to `lo + 1` via adjacent SWAPs.
        for s in (lo + 1..hi).rev() {
            self.swap_adjacent(s);
        }
        self.apply_adjacent_gate(lo, |theta, chi_l, chi_r, d| {
            gate_fn(theta, chi_l, chi_r, d, swapped)
        });
        // Route back.
        for s in lo + 1..hi {
            self.swap_adjacent(s);
        }
    }

    fn swap_adjacent(&mut self, i: usize) {
        self.apply_adjacent_gate(i, |theta, chi_l, chi_r, d| {
            for l in 0..chi_l {
                for pa in 0..d {
                    for pb in (pa + 1)..d {
                        let r1 = l * d + pa;
                        let c1 = pb * chi_r;
                        let r2 = l * d + pb;
                        let c2 = pa * chi_r;
                        for r in 0..chi_r {
                            let tmp = theta[(r1, c1 + r)];
                            theta[(r1, c1 + r)] = theta[(r2, c2 + r)];
                            theta[(r2, c2 + r)] = tmp;
                        }
                    }
                }
            }
        });
    }

    fn apply_adjacent_gate(
        &mut self,
        i: usize,
        gate_fn: impl FnOnce(&mut Mat<C64>, usize, usize, usize),
    ) {
        if self.center < i {
            self.move_center(i);
        } else if self.center > i + 1 {
            self.move_center(i + 1);
        }
        let d = self.cutoff;
        let chi_l = self.sites[i].left;
        let chi_m = self.sites[i].right;
        let chi_r = self.sites[i + 1].right;

        let a_mat = Mat::from_fn(chi_l * d, chi_m, |r, c| self.sites[i].data[r * chi_m + c]);
        let b_mat = Mat::from_fn(chi_m, d * chi_r, |r, c| {
            self.sites[i + 1].data[r * (d * chi_r) + c]
        });
        let mut theta = matmul(a_mat.as_ref(), b_mat.as_ref());

        gate_fn(&mut theta, chi_l, chi_r, d);

        let (u, s_vals, v) = thin_svd(theta.as_ref());
        let s0 = s_vals.first().copied().unwrap_or(0.0);
        let thresh = self.options.cutoff_tol * s0;
        let mut keep = s_vals
            .iter()
            .take(self.options.max_bond)
            .filter(|&&sv| sv > thresh)
            .count();
        if keep == 0 {
            keep = 1.min(s_vals.len());
        }
        let discarded: f64 = s_vals[keep..].iter().map(|sv| sv * sv).sum();
        self.svd_truncation_error += discarded;

        let rows = chi_l * d;
        let cols = d * chi_r;
        let mut u_data = vec![C64::new(0.0, 0.0); rows * keep];
        for r in 0..rows {
            for c in 0..keep {
                u_data[r * keep + c] = u[(r, c)];
            }
        }
        // Absorb singular values into right site `i + 1` (center moves to `i + 1`).
        let mut vt_data = vec![C64::new(0.0, 0.0); keep * cols];
        for r in 0..keep {
            let sv = s_vals[r];
            for c in 0..cols {
                vt_data[r * cols + c] = v[(c, r)].conj() * sv;
            }
        }
        self.sites[i] = Site {
            left: chi_l,
            right: keep,
            data: u_data,
        };
        self.sites[i + 1] = Site {
            left: keep,
            right: chi_r,
            data: vt_data,
        };
        self.center = i + 1;
    }

    // -----------------------------------------------------------------------
    // Destructive measurements (project, reset measured mode to vacuum)
    // -----------------------------------------------------------------------

    /// Destructive photon-number (PNR) measurement on `mode`.
    pub fn measure_photons<R: Rng + ?Sized>(
        &mut self,
        mode: usize,
        rng: &mut R,
    ) -> Result<usize, Error> {
        self.check_mode(mode)?;
        self.move_center(mode);
        let d = self.cutoff;
        let s = &mut self.sites[mode];

        let mut probs = vec![0.0; d];
        for l in 0..s.left {
            for (p, pr) in probs.iter_mut().enumerate() {
                let base = (l * d + p) * s.right;
                *pr += s.data[base..base + s.right]
                    .iter()
                    .map(|z| z.norm_sqr())
                    .sum::<f64>();
            }
        }
        let total: f64 = probs.iter().sum();
        let u = rng.random::<f64>() * total;
        let mut acc = 0.0;
        let mut outcome = d - 1;
        for (p, &pr) in probs.iter().enumerate() {
            acc += pr;
            if u < acc {
                outcome = p;
                break;
            }
        }
        let inv_norm = if probs[outcome] > 0.0 {
            1.0 / probs[outcome].sqrt()
        } else {
            1.0
        };

        // Project onto `p == outcome` and reset mode to `|0⟩`.
        let mut new_data = vec![C64::new(0.0, 0.0); s.data.len()];
        for l in 0..s.left {
            let src = (l * d + outcome) * s.right;
            let dst = (l * d) * s.right;
            for r in 0..s.right {
                new_data[dst + r] = s.data[src + r] * inv_norm;
            }
        }
        s.data = new_data;
        Ok(outcome)
    }

    /// Destructive threshold measurement on `mode` (`true` for ≥ 1 photon).
    pub fn measure_threshold<R: Rng + ?Sized>(
        &mut self,
        mode: usize,
        rng: &mut R,
    ) -> Result<bool, Error> {
        let n = self.measure_photons(mode, rng)?;
        Ok(n > 0)
    }

    /// Destructive homodyne measurement of `x̂ cos φ + p̂ sin φ` on `mode`.
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
        self.move_center(mode);
        let d = self.cutoff;
        let rho = self.local_reduced_density(mode);

        // Grid covering the Fock support ±(√(2d) + 5).
        let q_max = ((2 * d) as f64).sqrt() + 5.0;
        let n_grid = (d * 24).clamp(256, 2048);
        let dq = 2.0 * q_max / (n_grid as f64);
        let mut psi = vec![0.0; d];
        let mut cdf = Vec::with_capacity(n_grid);
        let mut cum = 0.0;
        for i in 0..n_grid {
            let q = -q_max + (i as f64 + 0.5) * dq;
            hermite_fock_wavefunctions(q, d, &mut psi);
            let mut dens = 0.0;
            for m in 0..d {
                let pm = psi[m];
                let row = m * d;
                dens += rho[row + m].re * pm * pm;
                for n in (m + 1)..d {
                    dens += 2.0 * rho[row + n].re * pm * psi[n];
                }
            }
            cum += dens.max(0.0) * dq;
            cdf.push(cum);
        }
        let u = rng.random::<f64>() * cum;
        let idx = cdf.partition_point(|&c| c < u).min(n_grid - 1);
        let outcome = -q_max + (idx as f64 + rng.random::<f64>()) * dq;

        // Project onto ⟨q| and reset `mode` to `|0⟩`.
        hermite_fock_wavefunctions(outcome, d, &mut psi);
        let ket: Vec<C64> = psi.iter().map(|&w| C64::new(w, 0.0)).collect();
        self.project_onto_bra_and_reset(mode, &ket);
        Ok(outcome)
    }

    /// Destructive heterodyne measurement on `mode`, returning `(Re α, Im α)`.
    pub fn measure_heterodyne<R: Rng + ?Sized>(
        &mut self,
        mode: usize,
        rng: &mut R,
    ) -> Result<(f64, f64), Error> {
        self.check_mode(mode)?;
        self.move_center(mode);
        let d = self.cutoff;
        let rho = self.local_reduced_density(mode);

        let a_max = (d as f64).sqrt() + 3.5;
        let n_grid = 64usize;
        let da = 2.0 * a_max / (n_grid as f64);
        let mut coh = vec![C64::new(0.0, 0.0); d];
        let mut cdf = Vec::with_capacity(n_grid * n_grid);
        let mut cum = 0.0;
        for ix in 0..n_grid {
            let re = -a_max + (ix as f64 + 0.5) * da;
            for ip in 0..n_grid {
                let im = -a_max + (ip as f64 + 0.5) * da;
                coherent_ket(C64::new(re, im), &mut coh);
                let mut q_val = 0.0;
                for m in 0..d {
                    let cm = coh[m].conj();
                    let row = m * d;
                    q_val += (cm * coh[m] * rho[row + m]).re;
                    for n in (m + 1)..d {
                        q_val += 2.0 * (cm * coh[n] * rho[row + n]).re;
                    }
                }
                cum += q_val.max(0.0);
                cdf.push(cum);
            }
        }
        let u = rng.random::<f64>() * cum;
        let cell = cdf.partition_point(|&c| c < u).min(n_grid * n_grid - 1);
        let re = -a_max + ((cell / n_grid) as f64 + rng.random::<f64>()) * da;
        let im = -a_max + ((cell % n_grid) as f64 + rng.random::<f64>()) * da;

        // Project onto ⟨α| (bra coefficients are `coh[p].conj()`) and reset to `|0⟩`.
        coherent_ket(C64::new(re, im), &mut coh);
        let bra: Vec<C64> = coh.into_iter().map(|z| z.conj()).collect();
        self.project_onto_bra_and_reset(mode, &bra);
        Ok((re, im))
    }

    fn local_reduced_density(&self, mode: usize) -> Vec<C64> {
        let d = self.cutoff;
        let s = &self.sites[mode];
        let mut rho = vec![C64::new(0.0, 0.0); d * d];
        for l in 0..s.left {
            for m in 0..d {
                let bm = (l * d + m) * s.right;
                for n in m..d {
                    let bn = (l * d + n) * s.right;
                    let mut acc = C64::new(0.0, 0.0);
                    for r in 0..s.right {
                        acc += s.data[bm + r] * s.data[bn + r].conj();
                    }
                    rho[m * d + n] += acc;
                    if m != n {
                        rho[n * d + m] += acc.conj();
                    }
                }
            }
        }
        rho
    }

    fn project_onto_bra_and_reset(&mut self, mode: usize, bra: &[C64]) {
        let d = self.cutoff;
        let s = &mut self.sites[mode];
        let mut new_data = vec![C64::new(0.0, 0.0); s.data.len()];
        let mut norm_sqr = 0.0;
        for l in 0..s.left {
            let dst = (l * d) * s.right;
            for r in 0..s.right {
                let mut acc = C64::new(0.0, 0.0);
                for (p, &bp) in bra.iter().enumerate() {
                    acc += bp * s.data[(l * d + p) * s.right + r];
                }
                new_data[dst + r] = acc;
                norm_sqr += acc.norm_sqr();
            }
        }
        if norm_sqr > 0.0 {
            let inv = 1.0 / norm_sqr.sqrt();
            for l in 0..s.left {
                let dst = (l * d) * s.right;
                for z in &mut new_data[dst..dst + s.right] {
                    *z *= inv;
                }
            }
        }
        s.data = new_data;
    }
}

fn coherent_ket(alpha: C64, out: &mut [C64]) {
    if out.is_empty() {
        return;
    }
    out[0] = C64::new((-0.5 * alpha.norm_sqr()).exp(), 0.0);
    for n in 1..out.len() {
        out[n] = out[n - 1] * alpha / (n as f64).sqrt();
    }
}

/// Apply total-photon-conserving `(s + 1) × (s + 1)` blocks (such as `BS` or
/// Givens 2-mode unitaries) to the two-site MPS tensor `theta` (`(chi_l*d) × (d*chi_r)`).
fn apply_sum_blocks_theta(
    theta: &mut Mat<C64>,
    chi_l: usize,
    chi_r: usize,
    d: usize,
    blocks: &[Vec<C64>],
    swapped: bool,
) {
    let max_s = 2 * (d - 1);
    let mut in_buf = vec![C64::new(0.0, 0.0); d];
    let mut out_buf = vec![C64::new(0.0, 0.0); d];
    for l in 0..chi_l {
        for r in 0..chi_r {
            for (s, blk) in blocks.iter().enumerate().take(max_s + 1) {
                let na_lo = s.saturating_sub(d - 1);
                let na_hi = s.min(d - 1);
                let dim_s = s + 1;
                let span = na_hi - na_lo + 1;
                for (k, dst) in in_buf.iter_mut().enumerate().take(span) {
                    let na = na_lo + k;
                    let nb = s - na;
                    let (p0, p1) = if swapped { (nb, na) } else { (na, nb) };
                    *dst = theta[(l * d + p0, p1 * chi_r + r)];
                }
                for (out_k, dst) in out_buf.iter_mut().enumerate().take(span) {
                    let out_a = na_lo + out_k;
                    let row = out_a * dim_s;
                    let mut acc = C64::new(0.0, 0.0);
                    for (in_k, &v) in in_buf.iter().enumerate().take(span) {
                        acc += blk[row + (na_lo + in_k)] * v;
                    }
                    *dst = acc;
                }
                for (k, &v) in out_buf.iter().enumerate().take(span) {
                    let na = na_lo + k;
                    let nb = s - na;
                    let (p0, p1) = if swapped { (nb, na) } else { (na, nb) };
                    theta[(l * d + p0, p1 * chi_r + r)] = v;
                }
            }
        }
    }
}

/// Apply photon-difference-conserving blocks (`TwoModeSqueezing`) to `theta`.
fn apply_diff_blocks_theta(
    theta: &mut Mat<C64>,
    chi_l: usize,
    chi_r: usize,
    d: usize,
    blocks: &[Vec<C64>],
    swapped: bool,
) {
    let mut in_buf = vec![C64::new(0.0, 0.0); d];
    let mut out_buf = vec![C64::new(0.0, 0.0); d];
    for l in 0..chi_l {
        for r in 0..chi_r {
            for (b_idx, blk) in blocks.iter().enumerate() {
                let diff = b_idx as isize - (d as isize - 1);
                let len = d - diff.unsigned_abs();
                let oa = diff.max(0) as usize;
                let ob = (-diff).max(0) as usize;
                for (k, dst) in in_buf.iter_mut().enumerate().take(len) {
                    let (p0, p1) = if swapped {
                        (k + ob, k + oa)
                    } else {
                        (k + oa, k + ob)
                    };
                    *dst = theta[(l * d + p0, p1 * chi_r + r)];
                }
                for (out_k, dst) in out_buf.iter_mut().enumerate().take(len) {
                    let row = out_k * len;
                    let mut acc = C64::new(0.0, 0.0);
                    for (in_k, &v) in in_buf.iter().enumerate().take(len) {
                        acc += blk[row + in_k] * v;
                    }
                    *dst = acc;
                }
                for (k, &v) in out_buf.iter().enumerate().take(len) {
                    let (p0, p1) = if swapped {
                        (k + ob, k + oa)
                    } else {
                        (k + oa, k + ob)
                    };
                    theta[(l * d + p0, p1 * chi_r + r)] = v;
                }
            }
        }
    }
}

/// Apply `CX(s) = exp(−i s x̂_a p̂_b / 2)` (`is_cx = true`) or
/// `CZ(s) = exp(i s x̂_a x̂_b / 2)` (`is_cx = false`) directly to the two-site
/// tensor `theta` in `O(χ_L χ_R d big²)` using the cached padded quadrature
/// eigenbases without forming a `d² × d²` matrix.
fn apply_quad_product_theta(
    theta: &mut Mat<C64>,
    chi_l: usize,
    chi_r: usize,
    d: usize,
    s: f64,
    is_cx: bool,
    swapped: bool,
) {
    let qe = cached_quad_eigen(d);
    let big = qe.big;
    let sign = if is_cx { -0.5 * s } else { 0.5 * s };

    // Precompute phase table `exp(i * sign * λ_ka * λ_kb)` (big × big).
    let mut phase = vec![C64::new(0.0, 0.0); big * big];
    for ka in 0..big {
        let la = qe.evals[ka] * sign;
        let row = ka * big;
        for kb in 0..big {
            phase[row + kb] = C64::from_polar(1.0, la * qe.evals[kb]);
        }
    }

    let mut mid = vec![C64::new(0.0, 0.0); big * d];
    let mut quad = vec![C64::new(0.0, 0.0); big * big];
    let mut back = vec![C64::new(0.0, 0.0); big * d];

    for l in 0..chi_l {
        for r in 0..chi_r {
            // Step 1: transform `a` from Fock (d) to x̂-eigenbasis (big):
            // mid[ka, nb] = Σ_na vx[na, ka] * T[na, nb].
            for ka in 0..big {
                for nb in 0..d {
                    let mut acc = C64::new(0.0, 0.0);
                    for na in 0..d {
                        let (p0, p1) = if swapped { (nb, na) } else { (na, nb) };
                        acc += theta[(l * d + p0, p1 * chi_r + r)] * qe.vx[na * big + ka];
                    }
                    mid[ka * d + nb] = acc;
                }
            }
            // Step 2: transform `b` to p̂ (CX) or x̂ (CZ) eigenbasis and multiply phase.
            if is_cx {
                for ka in 0..big {
                    let mrow = &mid[ka * d..(ka + 1) * d];
                    let prow = &phase[ka * big..(ka + 1) * big];
                    let qrow = &mut quad[ka * big..(ka + 1) * big];
                    for kb in 0..big {
                        let mut acc = C64::new(0.0, 0.0);
                        for nb in 0..d {
                            acc += mrow[nb] * qe.vp[nb * big + kb].conj();
                        }
                        qrow[kb] = acc * prow[kb];
                    }
                }
                // Step 3: transform `b` back to Fock: back[ka, nb] = Σ_kb vp[nb, kb] * quad[ka, kb].
                for ka in 0..big {
                    let qrow = &quad[ka * big..(ka + 1) * big];
                    let brow = &mut back[ka * d..(ka + 1) * d];
                    for nb in 0..d {
                        let vrow = &qe.vp[nb * big..(nb + 1) * big];
                        let mut acc = C64::new(0.0, 0.0);
                        for kb in 0..big {
                            acc += qrow[kb] * vrow[kb];
                        }
                        brow[nb] = acc;
                    }
                }
            } else {
                for ka in 0..big {
                    let mrow = &mid[ka * d..(ka + 1) * d];
                    let prow = &phase[ka * big..(ka + 1) * big];
                    let qrow = &mut quad[ka * big..(ka + 1) * big];
                    for kb in 0..big {
                        let mut acc = C64::new(0.0, 0.0);
                        for nb in 0..d {
                            acc += mrow[nb] * qe.vx[nb * big + kb];
                        }
                        qrow[kb] = acc * prow[kb];
                    }
                }
                for ka in 0..big {
                    let qrow = &quad[ka * big..(ka + 1) * big];
                    let brow = &mut back[ka * d..(ka + 1) * d];
                    for nb in 0..d {
                        let vrow = &qe.vx[nb * big..(nb + 1) * big];
                        let mut acc = C64::new(0.0, 0.0);
                        for kb in 0..big {
                            acc += qrow[kb] * vrow[kb];
                        }
                        brow[nb] = acc;
                    }
                }
            }
            // Step 4: transform `a` back to Fock.
            for na in 0..d {
                let vrow = &qe.vx[na * big..(na + 1) * big];
                for nb in 0..d {
                    let mut acc = C64::new(0.0, 0.0);
                    for ka in 0..big {
                        acc += back[ka * d + nb] * vrow[ka];
                    }
                    let (p0, p1) = if swapped { (nb, na) } else { (na, nb) };
                    theta[(l * d + p0, p1 * chi_r + r)] = acc;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cv::{FockState, ModeState};

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() < tol, "{a} vs {b} (diff {})", (a - b).abs());
    }

    #[test]
    fn bosonic_mps_matches_fock_state_on_universal_circuit() {
        let d = 10;
        let mut c = ModeCircuit::new(3);
        c.set_input(0, ModeState::Fock(1)).unwrap();
        c.set_input(1, ModeState::Coherent { r: 0.4, phi: 0.2 })
            .unwrap();
        c.squeeze(0.25, 0.3, 0)
            .displace(0.2, -0.4, 2)
            .kerr(0.35, 0)
            .beam_splitter(0.6, 0.2, 0, 1)
            .two_mode_squeeze(0.2, -0.3, 1, 2)
            .cross_kerr(-0.25, 0, 2) // non-adjacent!
            .cx(0.3, 2, 0) // non-adjacent & reversed!
            .cz(-0.2, 0, 1)
            .cubic_phase(0.08, 1);

        let fock = FockState::from_circuit(&c, d).unwrap();
        let mut mps = BosonicMps::from_circuit(
            &c,
            d,
            BosonicMpsOptions {
                max_bond: 64,
                cutoff_tol: 1e-12,
            },
        )
        .unwrap();

        for m in 0..3 {
            close(
                mps.mean_photons(m).unwrap(),
                fock.mean_photons(m).unwrap(),
                1e-7,
            );
            close(mps.mean_x(m).unwrap(), fock.mean_x(m).unwrap(), 1e-7);
            close(mps.mean_p(m).unwrap(), fock.mean_p(m).unwrap(), 1e-7);
        }
        for pat in [[0, 0, 0], [1, 0, 0], [0, 1, 1], [2, 1, 0], [1, 2, 1]] {
            let a_m = mps.amplitude(&pat).unwrap();
            let a_f = fock.amplitude(&pat).unwrap();
            assert!((a_m - a_f).norm() < 1e-7, "pat={pat:?}: {a_m} vs {a_f}");
        }
        assert!(mps.truncation_error() >= 0.0);
    }

    #[test]
    fn bosonic_mps_scales_to_twenty_modes_non_gaussian_and_samples() {
        let m = 20;
        let d = 8;
        let mut c = ModeCircuit::new(m);
        for i in 0..m {
            c.squeeze(0.2, 0.1 * (i as f64), i).kerr(0.3, i);
        }
        for i in 0..(m - 1) {
            c.beam_splitter(0.35, 0.2, i, i + 1);
        }
        let mut mps = BosonicMps::from_circuit(
            &c,
            d,
            BosonicMpsOptions {
                max_bond: 24,
                cutoff_tol: 1e-8,
            },
        )
        .unwrap();
        assert!(mps.max_bond_dim() <= 24);
        assert!(
            mps.truncation_error() < 1e-3,
            "trunc={}",
            mps.truncation_error()
        );
        let total_n: f64 = (0..m).map(|i| mps.mean_photons(i).unwrap()).sum();
        assert!(total_n > 0.5 && total_n < 5.0, "total_n={total_n}");

        // Mid-circuit PNR + homodyne + feed-forward sampling
        let mut cs = ModeCircuit::new(4);
        cs.set_input(0, ModeState::Fock(1)).unwrap();
        cs.set_input(1, ModeState::Fock(1)).unwrap();
        cs.beam_splitter(std::f64::consts::FRAC_PI_4, 0.0, 0, 1);
        let m0 = cs.measure_photons(0);
        cs.displace(m0 * 0.5, 0.0, 2);
        cs.measure_x(2);

        let mut rng = StdRng::seed_from_u64(42);
        let rows = BosonicMps::sample(&cs, 8, BosonicMpsOptions::default(), 200, &mut rng).unwrap();
        assert_eq!(rows.len(), 200);
        // HOM: m0 is always 0 or 2 (never 1)
        assert!(rows.iter().all(|r| r[0] == 0.0 || r[0] == 2.0));
        // When m0 == 2, mode 2 is displaced by α = 1 => ⟨x_2⟩ = 2; when m0 == 0 => ⟨x_2⟩ = 0.
        let mean_x2_m2: f64 = rows
            .iter()
            .filter(|r| r[0] == 2.0)
            .map(|r| r[1])
            .sum::<f64>()
            / rows.iter().filter(|r| r[0] == 2.0).count() as f64;
        assert!((mean_x2_m2 - 2.0).abs() < 0.35, "mean_x2_m2={mean_x2_m2}");
    }
}
