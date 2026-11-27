//! Exact Gaussian state simulator in the quadrature covariance representation.
//!
//! Conventions match [`crate::cv`] (Strawberry Fields, ħ = 2):
//! - Quadrature vector `r̂ = (x̂_0, …, x̂_{m−1}, p̂_0, …, p̂_{m−1})` (`xxpp`
//!   ordering), with `x̂ = a + a†`, `p̂ = −i(a − a†)`.
//! - Mean vector `μ_i = ⟨r̂_i⟩` (length `2m`) and symmetric covariance matrix
//!   `V_{ij} = ½ ⟨{r̂_i − μ_i, r̂_j − μ_j}⟩` (`2m × 2m`, row-major).
//! - The vacuum has `μ = 0` and `V = I_{2m}`.
//!
//! Single- and two-mode Gaussian gates update `V` in place in `O(m)` flops
//! using exact symmetry preservation; multi-mode passive blocks use `faer`
//! matrix multiplication when large. Homodyne and heterodyne measurements
//! condition the state via Schur complements and reset the measured mode to
//! the vacuum, supporting arbitrary mid-circuit feed-forward.

use super::circuit::{Measurement, ModeCircuit, ModeOp};
use super::gate::{ModeGate, TwoModeGate};
use super::hafnian::{hafnian_repeated, loop_hafnian_repeated, threshold_probability_real};
use super::state::ModeState;
use crate::error::Error;
use faer::Mat;
use num_complex::Complex64 as C64;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;

/// Minimum shot count for parallelising multi-shot Gaussian sampling.
const SAMPLE_PAR_MIN_SHOTS: usize = 64;

/// Tolerance for detecting a pure Gaussian state from the off-diagonal block
/// of its `A` matrix.
const PURE_BLOCK_TOL: f64 = 1e-12;

/// An exact `m`-mode Gaussian state, represented by its quadrature means `μ`
/// and `2m × 2m` covariance matrix `V` in `xxpp` ordering (`ħ = 2`).
#[derive(Clone, Debug, PartialEq)]
pub struct GaussianState {
    n_modes: usize,
    means: Vec<f64>,
    cov: Vec<f64>,
}

impl GaussianState {
    /// The `n_modes`-mode vacuum `|0…0⟩`: `μ = 0`, `V = I_{2m}`.
    pub fn vacuum(n_modes: usize) -> Self {
        let dim = 2 * n_modes;
        let mut cov = vec![0.0; dim * dim];
        for i in 0..dim {
            cov[i * dim + i] = 1.0;
        }
        GaussianState {
            n_modes,
            means: vec![0.0; dim],
            cov,
        }
    }

    /// Construct a Gaussian state from its `xxpp` quadrature mean vector `means`
    /// (length `2m`) and symmetric covariance matrix `cov` (`2m × 2m`,
    /// row-major, `ħ = 2`).
    pub fn from_moments(means: Vec<f64>, cov: Vec<f64>) -> Result<Self, Error> {
        if !means.len().is_multiple_of(2) {
            return Err(Error::InvalidArgument(format!(
                "means length must be even (2 * n_modes), got {}",
                means.len()
            )));
        }
        let dim = means.len();
        let n_modes = dim / 2;
        if cov.len() != dim * dim {
            return Err(Error::InvalidArgument(format!(
                "covariance matrix for {n_modes} modes needs {} entries, got {}",
                dim * dim,
                cov.len()
            )));
        }
        if !means.iter().all(|x| x.is_finite()) || !cov.iter().all(|x| x.is_finite()) {
            return Err(Error::InvalidArgument(
                "means and covariance entries must be finite".into(),
            ));
        }
        for i in 0..dim {
            for j in (i + 1)..dim {
                if (cov[i * dim + j] - cov[j * dim + i]).abs() > 1e-9 {
                    return Err(Error::InvalidArgument(
                        "covariance matrix must be symmetric".into(),
                    ));
                }
            }
        }
        // Symmetrise tiny numerical asymmetry.
        let mut cov = cov;
        for i in 0..dim {
            for j in (i + 1)..dim {
                let avg = 0.5 * (cov[i * dim + j] + cov[j * dim + i]);
                cov[i * dim + j] = avg;
                cov[j * dim + i] = avg;
            }
        }
        Ok(GaussianState {
            n_modes,
            means,
            cov,
        })
    }

    /// Build the initial product state from `circuit.inputs()`.
    fn from_inputs(inputs: &[ModeState]) -> Result<Self, Error> {
        let mut state = GaussianState::vacuum(inputs.len());
        for (m, input) in inputs.iter().enumerate() {
            match *input {
                ModeState::Vacuum | ModeState::Fock(0) => {}
                ModeState::Coherent { r, phi } => {
                    state.displace(r, phi, m)?;
                }
                ModeState::Squeezed { r, phi } => {
                    state.squeeze(r, phi, m)?;
                }
                ModeState::Thermal { mean_photons } => {
                    let v = 2.0 * mean_photons + 1.0;
                    let dim = 2 * state.n_modes;
                    let px = m;
                    let pp = state.n_modes + m;
                    state.cov[px * dim + px] = v;
                    state.cov[pp * dim + pp] = v;
                }
                _ => {
                    return Err(Error::InvalidArgument(format!(
                        "input state on mode {m} is not Gaussian"
                    )));
                }
            }
        }
        Ok(state)
    }

    /// Run a bound, measurement-free Gaussian circuit.
    pub fn from_circuit(circuit: &ModeCircuit) -> Result<Self, Error> {
        if circuit.n_parameters() > 0 {
            return Err(Error::InvalidArgument(
                "circuit has unbound parameters; bind it first".into(),
            ));
        }
        if circuit.has_measurements() {
            return Err(Error::InvalidArgument(
                "circuit has measurements; use GaussianState::run or GaussianState::sample".into(),
            ));
        }
        let mut state = Self::from_inputs(circuit.inputs())?;
        for op in circuit.ops() {
            state.apply_op(op, &[])?;
        }
        Ok(state)
    }

    /// Run one shot of a bound Gaussian circuit with homodyne/heterodyne
    /// measurements and feed-forward, returning the final state and the
    /// measurement results.
    pub fn run<R: Rng + ?Sized>(
        circuit: &ModeCircuit,
        rng: &mut R,
    ) -> Result<(Self, Vec<f64>), Error> {
        if circuit.n_parameters() > 0 {
            return Err(Error::InvalidArgument(
                "circuit has unbound parameters; bind it first".into(),
            ));
        }
        let mut state = Self::from_inputs(circuit.inputs())?;
        let mut results = Vec::with_capacity(circuit.n_results());
        state.run_ops(circuit.ops(), &mut results, rng)?;
        Ok((state, results))
    }

    /// Sample `shots` independent measurement trajectories from a bound
    /// Gaussian circuit.
    ///
    /// Runs the deterministic gate prefix before the first measurement once.
    /// When all measurements are terminal homodyne/heterodyne on distinct
    /// modes without result-dependent angles, computes one Cholesky factor of
    /// the marginal measurement covariance and draws all shots in `O(k²)` each.
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
        // Validate every op is Gaussian-runnable before executing.
        for op in circuit.ops() {
            match op {
                ModeOp::Measure { measurement, .. } => {
                    if !matches!(
                        measurement,
                        Measurement::Homodyne { .. } | Measurement::Heterodyne
                    ) {
                        return Err(Error::InvalidArgument(
                            "GaussianState::sample supports homodyne and heterodyne measurements; \
                             photon-counting and threshold sampling are not yet available"
                                .into(),
                        ));
                    }
                }
                op if !op.is_gaussian() => {
                    return Err(Error::InvalidArgument(format!(
                        "operation {op:?} is not Gaussian"
                    )));
                }
                _ => {}
            }
        }

        if shots == 0 {
            return Ok(Vec::new());
        }

        // Run the deterministic prefix before the first measurement once.
        let ops = circuit.ops();
        let first_meas = ops
            .iter()
            .position(|op| matches!(op, ModeOp::Measure { .. }))
            .unwrap_or(ops.len());

        let mut prefix = Self::from_inputs(circuit.inputs())?;
        for op in &ops[..first_meas] {
            prefix.apply_op(op, &[])?;
        }
        let suffix = &ops[first_meas..];
        let n_results = circuit.n_results();
        if suffix.is_empty() {
            return Ok(vec![Vec::new(); shots]);
        }

        // Fast path: all suffix ops are independent terminal homodyne/heterodyne
        // measurements on distinct modes with fixed angles.
        if let Some(plan) = TerminalGaussianPlan::try_build(&prefix, suffix) {
            return Ok(plan.sample(shots, rng));
        }

        // General path (mid-circuit feed-forward or re-measured modes):
        // clone the post-prefix state per shot.
        if shots >= SAMPLE_PAR_MIN_SHOTS {
            let seeds: Vec<u64> = (0..shots).map(|_| rng.random()).collect();
            seeds
                .into_par_iter()
                .map(|seed| {
                    let mut local_rng = StdRng::seed_from_u64(seed);
                    let mut st = prefix.clone();
                    let mut res = Vec::with_capacity(n_results);
                    st.run_ops(suffix, &mut res, &mut local_rng)?;
                    Ok(res)
                })
                .collect()
        } else {
            let mut out = Vec::with_capacity(shots);
            for _ in 0..shots {
                let mut st = prefix.clone();
                let mut res = Vec::with_capacity(n_results);
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
                    Measurement::Homodyne { phi } => {
                        let angle = phi.evaluate(results)?;
                        let val = self.measure_homodyne(*mode, angle, rng)?;
                        results.push(val);
                    }
                    Measurement::Heterodyne => {
                        let (re, im) = self.measure_heterodyne(*mode, rng)?;
                        results.push(re);
                        results.push(im);
                    }
                    Measurement::PhotonCount | Measurement::Threshold => {
                        return Err(Error::InvalidArgument(
                            "GaussianState cannot sample photon-counting or threshold \
                             measurements directly; use fock_probability or threshold_probability"
                                .into(),
                        ));
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
                ModeGate::CubicPhase { .. } | ModeGate::Kerr { .. } => Err(Error::InvalidArgument(
                    format!("gate {gate:?} is not Gaussian"),
                )),
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
                    TwoModeGate::CrossKerr { .. } => Err(Error::InvalidArgument(
                        "CrossKerr gate is not Gaussian".into(),
                    )),
                }
            }
            ModeOp::Passive { modes, matrix } => self.apply_passive(modes, matrix),
            ModeOp::Channel { channel, mode } => self.apply_channel(channel, *mode),
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

    /// Quadrature mean vector `(x_0, …, x_{m−1}, p_0, …, p_{m−1})`.
    pub fn means(&self) -> &[f64] {
        &self.means
    }

    /// `2m × 2m` row-major symmetric covariance matrix in `xxpp` ordering.
    pub fn cov(&self) -> &[f64] {
        &self.cov
    }

    /// `⟨x̂⟩` on `mode`.
    pub fn mean_x(&self, mode: usize) -> f64 {
        self.means[mode]
    }

    /// `⟨p̂⟩` on `mode`.
    pub fn mean_p(&self, mode: usize) -> f64 {
        self.means[self.n_modes + mode]
    }

    /// Mean photon number `⟨n̂⟩ = ¼ (V_{xx} + V_{pp} + μ_x² + μ_p² − 2)` on
    /// `mode`.
    pub fn mean_photons(&self, mode: usize) -> f64 {
        let dim = 2 * self.n_modes;
        let ix = mode;
        let ip = self.n_modes + mode;
        let vxx = self.cov[ix * dim + ix];
        let vpp = self.cov[ip * dim + ip];
        let mx = self.means[ix];
        let mp = self.means[ip];
        0.25 * (vxx + vpp + mx * mx + mp * mp - 2.0)
    }

    /// State purity `Tr(ρ²) = 1 / √det(V)`.
    pub fn purity(&self) -> f64 {
        let dim = 2 * self.n_modes;
        if dim == 0 {
            return 1.0;
        }
        let Some(l) = cholesky_real(&self.cov, dim) else {
            return 0.0;
        };
        let log_sqrt_det: f64 = (0..dim).map(|i| l[i * dim + i].ln()).sum();
        (-log_sqrt_det).exp()
    }

    /// Reduced (marginal) Gaussian state on `modes`.
    pub fn reduced(&self, modes: &[usize]) -> Result<GaussianState, Error> {
        let k = modes.len();
        for (i, &m) in modes.iter().enumerate() {
            self.check_mode(m)?;
            if modes[..i].contains(&m) {
                return Err(Error::InvalidArgument(format!(
                    "mode {m} repeated in reduced state"
                )));
            }
        }
        let old_dim = 2 * self.n_modes;
        let new_dim = 2 * k;
        let mut idx = Vec::with_capacity(new_dim);
        for &m in modes {
            idx.push(m);
        }
        for &m in modes {
            idx.push(self.n_modes + m);
        }
        let means: Vec<f64> = idx.iter().map(|&i| self.means[i]).collect();
        let mut cov = vec![0.0; new_dim * new_dim];
        for r in 0..new_dim {
            for c in 0..new_dim {
                cov[r * new_dim + c] = self.cov[idx[r] * old_dim + idx[c]];
            }
        }
        Ok(GaussianState {
            n_modes: k,
            means,
            cov,
        })
    }

    /// Wigner function `W(x, p)` evaluated at phase-space coordinates
    /// `x = (x_0, …, x_{m−1})` and `p = (p_0, …, p_{m−1})`.
    pub fn wigner(&self, x: &[f64], p: &[f64]) -> Result<f64, Error> {
        if x.len() != self.n_modes || p.len() != self.n_modes {
            return Err(Error::InvalidArgument(format!(
                "wigner expects {} x and p coordinates, got {} and {}",
                self.n_modes,
                x.len(),
                p.len()
            )));
        }
        let dim = 2 * self.n_modes;
        let Some(l) = cholesky_real(&self.cov, dim) else {
            return Err(Error::InvalidArgument(
                "covariance matrix is not positive definite".into(),
            ));
        };
        let mut diff = Vec::with_capacity(dim);
        for i in 0..self.n_modes {
            diff.push(x[i] - self.means[i]);
        }
        for i in 0..self.n_modes {
            diff.push(p[i] - self.means[self.n_modes + i]);
        }
        let y = forward_solve_real(&l, dim, &diff);
        let quad: f64 = y.iter().map(|v| v * v).sum();
        let log_sqrt_det: f64 = (0..dim).map(|i| l[i * dim + i].ln()).sum();
        let log_norm = (self.n_modes as f64) * std::f64::consts::TAU.ln() + log_sqrt_det;
        Ok((-0.5 * quad - log_norm).exp())
    }

    /// Probability of observing the all-vacuum outcome `|0…0⟩`.
    pub fn vacuum_probability(&self) -> f64 {
        let dim = 2 * self.n_modes;
        if dim == 0 {
            return 1.0;
        }
        let mut m_mat = vec![0.0; dim * dim];
        for i in 0..dim {
            for j in 0..dim {
                let mut v = 0.5 * self.cov[i * dim + j];
                if i == j {
                    v += 0.5;
                }
                m_mat[i * dim + j] = v;
            }
        }
        let Some(l) = cholesky_real(&m_mat, dim) else {
            return 0.0;
        };
        let log_sqrt_det: f64 = (0..dim).map(|i| l[i * dim + i].ln()).sum();
        let quad = if self.means.iter().all(|&x| x == 0.0) {
            0.0
        } else {
            let y = forward_solve_real(&l, dim, &self.means);
            y.iter().map(|v| v * v).sum::<f64>()
        };
        (-0.25 * quad - log_sqrt_det).exp()
    }

    /// Exact photon-number probability `P(n_0, …, n_{m−1})` via loop hafnians.
    ///
    /// Automatically uses the `N × N` pure-state fast path (`|lhaf(B_n)|²`)
    /// when the state is pure, and the `2N × 2N` mixed-state path otherwise.
    pub fn fock_probability(&self, pattern: &[usize]) -> Result<f64, Error> {
        if pattern.len() != self.n_modes {
            return Err(Error::InvalidArgument(format!(
                "pattern has {} modes, expected {}",
                pattern.len(),
                self.n_modes
            )));
        }
        if self.n_modes == 0 {
            return Ok(1.0);
        }
        if pattern.iter().all(|&n| n == 0) {
            return Ok(self.vacuum_probability());
        }

        let m = self.n_modes;
        let dim = 2 * m;

        // Build σ_Q = T V T† + ½ I_{2m} in (α, α*) ordering.
        // T = ½ [[I, iI], [I, -iI]]:
        //   (T V T†)_{00} = ¼ (V_xx + V_pp + i(V_px - V_xp)) = ¼(V_xx + V_pp) + ¼ i(V_xp^T - V_xp)
        //   (T V T†)_{01} = ¼ (V_xx - V_pp + i(V_xp + V_px))
        //   (T V T†)_{10} = (T V T†)_{01}^*
        //   (T V T†)_{11} = (T V T†)_{00}^*
        let mut sigma_q = vec![C64::new(0.0, 0.0); dim * dim];
        for i in 0..m {
            for j in 0..m {
                let vxx = self.cov[i * dim + j];
                let vxp = self.cov[i * dim + (m + j)];
                let vpx = self.cov[(m + i) * dim + j];
                let vpp = self.cov[(m + i) * dim + (m + j)];
                let mut q00 = C64::new(0.25 * (vxx + vpp), 0.25 * (vpx - vxp));
                if i == j {
                    q00.re += 0.5;
                }
                let q01 = C64::new(0.25 * (vxx - vpp), 0.25 * (vxp + vpx));
                sigma_q[i * dim + j] = q00;
                sigma_q[i * dim + (m + j)] = q01;
                sigma_q[(m + i) * dim + j] = q01.conj();
                sigma_q[(m + i) * dim + (m + j)] = q00.conj();
            }
        }

        // Complex Hermitian Cholesky σ_Q = L L†.
        let l = cholesky_herm(&sigma_q, dim).ok_or_else(|| {
            Error::InvalidArgument("Husimi Q covariance matrix is not positive definite".into())
        })?;
        let log_sqrt_det: f64 = (0..dim).map(|i| l[i * dim + i].re.ln()).sum();

        // Compute σ_Q^{-1} by solving L L† x = e_j for each column j.
        let q_inv = invert_from_cholesky_herm(&l, dim);

        // Displacement vector α = (ᾱ, ᾱ*) with ᾱ = ½ (x + i p).
        let is_displaced = self.means.iter().any(|&x| x != 0.0);
        let (log_p0, gamma) = if is_displaced {
            let mut alpha = vec![C64::new(0.0, 0.0); dim];
            for i in 0..m {
                let a = C64::new(0.5 * self.means[i], 0.5 * self.means[m + i]);
                alpha[i] = a;
                alpha[m + i] = a.conj();
            }
            // y = σ_Q^{-1} α, γ = X y, exponent = -½ Re(α† y).
            let mut y = vec![C64::new(0.0, 0.0); dim];
            let mut quad = 0.0;
            for i in 0..dim {
                let mut acc = C64::new(0.0, 0.0);
                for j in 0..dim {
                    acc += q_inv[i * dim + j] * alpha[j];
                }
                y[i] = acc;
                quad += (alpha[i].conj() * acc).re;
            }
            let mut g = vec![C64::new(0.0, 0.0); dim];
            g[..m].copy_from_slice(&y[m..dim]);
            g[m..dim].copy_from_slice(&y[..m]);
            (-0.5 * quad - log_sqrt_det, g)
        } else {
            (-log_sqrt_det, Vec::new())
        };

        // A = X (I - σ_Q^{-1}): top m rows come from bottom m rows of (I - σ_Q^{-1}),
        // bottom m rows come from top m rows of (I - σ_Q^{-1}).
        let mut a_mat = vec![C64::new(0.0, 0.0); dim * dim];
        for i in 0..dim {
            let src_row = if i < m { i + m } else { i - m };
            for j in 0..dim {
                let mut v = -q_inv[src_row * dim + j];
                if src_row == j {
                    v.re += 1.0;
                }
                a_mat[i * dim + j] = v;
            }
        }

        // Check if A is block-diagonal (pure state): off-diagonal block A[0..m, m..2m] ≈ 0.
        let mut max_off = 0.0f64;
        for i in 0..m {
            for j in m..dim {
                max_off = max_off.max(a_mat[i * dim + j].norm());
            }
        }

        let log_fact: f64 = pattern
            .iter()
            .map(|&n| (1..=n).map(|k| (k as f64).ln()).sum::<f64>())
            .sum();

        let p0 = log_p0.exp();
        let inv_fact = (-log_fact).exp();

        if max_off < PURE_BLOCK_TOL {
            // Pure-state fast path: A = B* ⊕ B, so P(n) = P(0) |lhaf(B_n, γ1_n)|² / ∏ n_i!.
            let total_n: usize = pattern.iter().sum();
            if !is_displaced && total_n % 2 == 1 {
                return Ok(0.0);
            }
            let mut b_mat = vec![C64::new(0.0, 0.0); m * m];
            for i in 0..m {
                for j in 0..m {
                    b_mat[i * m + j] = a_mat[(m + i) * dim + (m + j)];
                }
            }
            let h = if is_displaced {
                loop_hafnian_repeated(&b_mat, &gamma[m..dim], pattern)?
            } else {
                hafnian_repeated(&b_mat, pattern)?
            };
            Ok((p0 * h.norm_sqr() * inv_fact).max(0.0))
        } else {
            // Mixed-state general path: 2N × 2N hafnian / loop hafnian.
            let mut rep2 = Vec::with_capacity(dim);
            rep2.extend_from_slice(pattern);
            rep2.extend_from_slice(pattern);
            let h = if is_displaced {
                loop_hafnian_repeated(&a_mat, &gamma, &rep2)?
            } else {
                hafnian_repeated(&a_mat, &rep2)?
            };
            Ok((p0 * h.re * inv_fact).max(0.0))
        }
    }

    /// Exact threshold-detector probability `P(clicks)` where `clicks[i]` is
    /// `true` if mode `i` registers at least one photon and `false` for vacuum.
    ///
    /// Uses real recursive `2 × 2` block Cholesky updates in `O(k² 2ᵏ)` where
    /// `k` is the number of clicking modes, supporting displaced and mixed
    /// states alike.
    pub fn threshold_probability(&self, clicks: &[bool]) -> Result<f64, Error> {
        if clicks.len() != self.n_modes {
            return Err(Error::InvalidArgument(format!(
                "click pattern has {} modes, expected {}",
                clicks.len(),
                self.n_modes
            )));
        }
        if self.n_modes == 0 {
            return Ok(1.0);
        }

        // Order non-clicking modes first, clicking modes last.
        let mut order = Vec::with_capacity(self.n_modes);
        for (i, &c) in clicks.iter().enumerate() {
            if !c {
                order.push(i);
            }
        }
        let n_zero = order.len();
        for (i, &c) in clicks.iter().enumerate() {
            if c {
                order.push(i);
            }
        }

        // Build interleaved M = (V + I) / 2 and μ in the reordered mode order.
        let m = self.n_modes;
        let dim = 2 * m;
        let mut idx = Vec::with_capacity(dim);
        for &mode in &order {
            idx.push(mode);
            idx.push(m + mode);
        }
        let mut m_mat = vec![0.0; dim * dim];
        let mut mu_inter = vec![0.0; dim];
        for r in 0..dim {
            mu_inter[r] = self.means[idx[r]];
            for c in 0..dim {
                let mut v = 0.5 * self.cov[idx[r] * dim + idx[c]];
                if r == c {
                    v += 0.5;
                }
                m_mat[r * dim + c] = v;
            }
        }

        Ok(threshold_probability_real(&m_mat, &mu_inter, m, n_zero).max(0.0))
    }

    // -----------------------------------------------------------------------
    // Single-mode, two-mode, and multi-mode Gaussian gates
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

    /// Rotation `R(φ)`: `a → e^{iφ} a`.
    pub fn rotation(&mut self, phi: f64, mode: usize) -> Result<(), Error> {
        self.check_mode(mode)?;
        let (s, c) = phi.sin_cos();
        self.apply_one_mode_symplectic(mode, [[c, -s], [s, c]]);
        Ok(())
    }

    /// Displacement `D(r e^{iφ})`: `x̂ → x̂ + 2 r cos φ`, `p̂ → p̂ + 2 r sin φ`.
    pub fn displace(&mut self, r: f64, phi: f64, mode: usize) -> Result<(), Error> {
        self.check_mode(mode)?;
        let (s, c) = phi.sin_cos();
        self.means[mode] += 2.0 * r * c;
        self.means[self.n_modes + mode] += 2.0 * r * s;
        Ok(())
    }

    /// Squeezing `S(r e^{iφ})`.
    pub fn squeeze(&mut self, r: f64, phi: f64, mode: usize) -> Result<(), Error> {
        self.check_mode(mode)?;
        let ch = r.cosh();
        let sh = r.sinh();
        let (s, c) = phi.sin_cos();
        let off = -s * sh;
        self.apply_one_mode_symplectic(mode, [[ch - c * sh, off], [off, ch + c * sh]]);
        Ok(())
    }

    /// Quadratic phase `P(s)`: `p̂ → p̂ + s x̂`.
    pub fn quadratic_phase(&mut self, s: f64, mode: usize) -> Result<(), Error> {
        self.check_mode(mode)?;
        if s == 0.0 {
            return Ok(());
        }
        let dim = 2 * self.n_modes;
        let ix = mode;
        let ip = self.n_modes + mode;
        self.means[ip] += s * self.means[ix];

        let vxx = self.cov[ix * dim + ix];
        let vxp = self.cov[ix * dim + ip];
        let vpp = self.cov[ip * dim + ip];

        // Row ip += s * row ix
        let rx = ix * dim;
        let rp = ip * dim;
        for j in 0..dim {
            self.cov[rp + j] += s * self.cov[rx + j];
        }
        // Symmetrise col ip from row ip
        for j in 0..dim {
            self.cov[j * dim + ip] = self.cov[rp + j];
        }
        // Exact 2×2 diagonal block
        self.cov[ix * dim + ix] = vxx;
        self.cov[ix * dim + ip] = vxp + s * vxx;
        self.cov[ip * dim + ix] = vxp + s * vxx;
        self.cov[ip * dim + ip] = vpp + 2.0 * s * vxp + s * s * vxx;
        Ok(())
    }

    /// Beam splitter `BS(θ, φ)` on `(a, b)`.
    pub fn beam_splitter(&mut self, theta: f64, phi: f64, a: usize, b: usize) -> Result<(), Error> {
        self.check_pair(a, b)?;
        let (st, t) = theta.sin_cos();
        let (sp, cp) = phi.sin_cos();
        let c = cp * st;
        let s = sp * st;
        let mat = [
            [t, -c, 0.0, -s],
            [c, t, -s, 0.0],
            [0.0, s, t, -c],
            [s, 0.0, c, t],
        ];
        self.apply_two_mode_symplectic(a, b, mat);
        Ok(())
    }

    /// Two-mode squeezing `S₂(r e^{iφ})` on `(a, b)`.
    pub fn two_mode_squeeze(&mut self, r: f64, phi: f64, a: usize, b: usize) -> Result<(), Error> {
        self.check_pair(a, b)?;
        let ch = r.cosh();
        let sh = r.sinh();
        let (sp, cp) = phi.sin_cos();
        let sc = cp * sh;
        let ss = sp * sh;
        let mat = [
            [ch, -sc, 0.0, -ss],
            [-sc, ch, -ss, 0.0],
            [0.0, -ss, ch, sc],
            [-ss, 0.0, sc, ch],
        ];
        self.apply_two_mode_symplectic(a, b, mat);
        Ok(())
    }

    /// `CX(s)` with control `a` and target `b`: `x̂_b → x̂_b + s x̂_a`,
    /// `p̂_a → p̂_a − s p̂_b`.
    pub fn cx(&mut self, s: f64, a: usize, b: usize) -> Result<(), Error> {
        self.check_pair(a, b)?;
        if s == 0.0 {
            return Ok(());
        }
        let m = self.n_modes;
        let dim = 2 * m;
        let (xa, xb, pa, pb) = (a, b, m + a, m + b);

        // Save 4×4 block before modifying.
        let idx = [xa, xb, pa, pb];
        let mut old = [[0.0; 4]; 4];
        for r in 0..4 {
            for c in 0..4 {
                old[r][c] = self.cov[idx[r] * dim + idx[c]];
            }
        }

        // Only 2 rows change: xb += s * xa, pa -= s * pb.
        self.means[xb] += s * self.means[xa];
        self.means[pa] -= s * self.means[pb];

        let (r_xa, r_xb, r_pa, r_pb) = (xa * dim, xb * dim, pa * dim, pb * dim);
        for j in 0..dim {
            self.cov[r_xb + j] += s * self.cov[r_xa + j];
            self.cov[r_pa + j] -= s * self.cov[r_pb + j];
        }
        for j in 0..dim {
            self.cov[j * dim + xb] = self.cov[r_xb + j];
            self.cov[j * dim + pa] = self.cov[r_pa + j];
        }

        let mat = [
            [1.0, 0.0, 0.0, 0.0],
            [s, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -s],
            [0.0, 0.0, 0.0, 1.0],
        ];
        self.write_4x4_transformed(idx, old, mat);
        Ok(())
    }

    /// `CZ(s)` on `(a, b)`: `p̂_a → p̂_a + s x̂_b`, `p̂_b → p̂_b + s x̂_a`.
    pub fn cz(&mut self, s: f64, a: usize, b: usize) -> Result<(), Error> {
        self.check_pair(a, b)?;
        if s == 0.0 {
            return Ok(());
        }
        let m = self.n_modes;
        let dim = 2 * m;
        let (xa, xb, pa, pb) = (a, b, m + a, m + b);
        let idx = [xa, xb, pa, pb];
        let mut old = [[0.0; 4]; 4];
        for r in 0..4 {
            for c in 0..4 {
                old[r][c] = self.cov[idx[r] * dim + idx[c]];
            }
        }

        self.means[pa] += s * self.means[xb];
        self.means[pb] += s * self.means[xa];

        let (r_xa, r_xb, r_pa, r_pb) = (xa * dim, xb * dim, pa * dim, pb * dim);
        for j in 0..dim {
            self.cov[r_pa + j] += s * self.cov[r_xb + j];
            self.cov[r_pb + j] += s * self.cov[r_xa + j];
        }
        for j in 0..dim {
            self.cov[j * dim + pa] = self.cov[r_pa + j];
            self.cov[j * dim + pb] = self.cov[r_pb + j];
        }

        let mat = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, s, 1.0, 0.0],
            [s, 0.0, 0.0, 1.0],
        ];
        self.write_4x4_transformed(idx, old, mat);
        Ok(())
    }

    /// Apply a linear-optics (passive unitary) block `U` on `modes`:
    /// `α → U α`, row-major `k × k`.
    pub fn apply_passive(&mut self, modes: &[usize], matrix: &[C64]) -> Result<(), Error> {
        let k = modes.len();
        if matrix.len() != k * k {
            return Err(Error::InvalidArgument(format!(
                "passive block on {k} modes needs {} matrix entries, got {}",
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
        // Build 2k × 2k symplectic matrix [[Re U, -Im U], [Im U, Re U]].
        let s_dim = 2 * k;
        let mut s = vec![0.0; s_dim * s_dim];
        for r in 0..k {
            for c in 0..k {
                let z = matrix[r * k + c];
                s[r * s_dim + c] = z.re;
                s[r * s_dim + (k + c)] = -z.im;
                s[(k + r) * s_dim + c] = z.im;
                s[(k + r) * s_dim + (k + c)] = z.re;
            }
        }
        self.apply_symplectic(modes, &s)
    }

    /// Apply a `2k × 2k` real symplectic matrix `s` (row-major, `xxpp` ordering)
    /// to `modes`.
    pub fn apply_symplectic(&mut self, modes: &[usize], s: &[f64]) -> Result<(), Error> {
        let k = modes.len();
        let s_dim = 2 * k;
        if s.len() != s_dim * s_dim {
            return Err(Error::InvalidArgument(format!(
                "symplectic block on {k} modes needs {} entries, got {}",
                s_dim * s_dim,
                s.len()
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

        let dim = 2 * self.n_modes;
        let mut idx = Vec::with_capacity(s_dim);
        for &m in modes {
            idx.push(m);
        }
        for &m in modes {
            idx.push(self.n_modes + m);
        }

        // Update means: μ_I ← S μ_I.
        let old_mu: Vec<f64> = idx.iter().map(|&i| self.means[i]).collect();
        for r in 0..s_dim {
            let mut acc = 0.0;
            for c in 0..s_dim {
                acc += s[r * s_dim + c] * old_mu[c];
            }
            self.means[idx[r]] = acc;
        }

        // Use faer GEMM for larger blocks (k >= 6) or full-register updates.
        if k >= 6 {
            let s_mat = Mat::from_fn(s_dim, s_dim, |r, c| s[r * s_dim + c]);
            // V_I = rows `idx` of V (s_dim × dim).
            let v_rows = Mat::from_fn(s_dim, dim, |r, c| self.cov[idx[r] * dim + c]);
            let sv = &s_mat * &v_rows; // s_dim × dim

            // Diagonal block: S V_II S^T.
            let sv_ii = Mat::from_fn(s_dim, s_dim, |r, c| sv[(r, idx[c])]);
            let new_ii = &sv_ii * s_mat.transpose();

            // Write updated rows & symmetric columns.
            for r in 0..s_dim {
                let row_i = idx[r];
                for c in 0..dim {
                    self.cov[row_i * dim + c] = sv[(r, c)];
                }
            }
            for c in 0..dim {
                for r in 0..s_dim {
                    let row_i = idx[r];
                    self.cov[c * dim + row_i] = sv[(r, c)];
                }
            }
            for r in 0..s_dim {
                for c in r..s_dim {
                    let avg = 0.5 * (new_ii[(r, c)] + new_ii[(c, r)]);
                    self.cov[idx[r] * dim + idx[c]] = avg;
                    self.cov[idx[c] * dim + idx[r]] = avg;
                }
            }
        } else {
            // Small k: direct cache-friendly loops without faer overhead.
            let mut old_ii = vec![0.0; s_dim * s_dim];
            for r in 0..s_dim {
                for c in 0..s_dim {
                    old_ii[r * s_dim + c] = self.cov[idx[r] * dim + idx[c]];
                }
            }
            let mut sv = vec![0.0; s_dim * dim];
            for r in 0..s_dim {
                for c in 0..s_dim {
                    let sc = s[r * s_dim + c];
                    if sc != 0.0 {
                        let src = idx[c] * dim;
                        let dst = r * dim;
                        for j in 0..dim {
                            sv[dst + j] += sc * self.cov[src + j];
                        }
                    }
                }
            }
            for (r, &row_idx) in idx.iter().enumerate() {
                let dst = row_idx * dim;
                let src = r * dim;
                self.cov[dst..dst + dim].copy_from_slice(&sv[src..src + dim]);
            }
            for j in 0..dim {
                for r in 0..s_dim {
                    self.cov[j * dim + idx[r]] = sv[r * dim + j];
                }
            }
            let mut tmp = vec![0.0; s_dim * s_dim];
            for r in 0..s_dim {
                for c in 0..s_dim {
                    let mut acc = 0.0;
                    for t in 0..s_dim {
                        acc += s[r * s_dim + t] * old_ii[t * s_dim + c];
                    }
                    tmp[r * s_dim + c] = acc;
                }
            }
            for r in 0..s_dim {
                for c in r..s_dim {
                    let mut acc = 0.0;
                    for t in 0..s_dim {
                        acc += tmp[r * s_dim + t] * s[c * s_dim + t];
                    }
                    self.cov[idx[r] * dim + idx[c]] = acc;
                    self.cov[idx[c] * dim + idx[r]] = acc;
                }
            }
        }
        Ok(())
    }

    /// Pure-loss channel on `mode` with energy transmissivity `eta ∈ [0, 1]`.
    pub fn apply_loss(&mut self, mode: usize, eta: f64) -> Result<(), Error> {
        self.apply_thermal_loss(mode, eta, 0.0)
    }

    /// Thermal-loss channel on `mode`: transmissivity `eta ∈ [0, 1]` with a
    /// thermal environment of `mean_photons ≥ 0`.
    pub fn apply_thermal_loss(
        &mut self,
        mode: usize,
        eta: f64,
        mean_photons: f64,
    ) -> Result<(), Error> {
        self.check_mode(mode)?;
        if !(eta.is_finite() && (0.0..=1.0).contains(&eta)) {
            return Err(Error::InvalidArgument(format!(
                "loss transmissivity must lie in [0, 1], got {eta}"
            )));
        }
        if !(mean_photons.is_finite() && mean_photons >= 0.0) {
            return Err(Error::InvalidArgument(format!(
                "thermal bath mean photons must be finite and ≥ 0, got {mean_photons}"
            )));
        }
        let dim = 2 * self.n_modes;
        let ix = mode;
        let ip = self.n_modes + mode;
        let s = eta.sqrt();
        self.means[ix] *= s;
        self.means[ip] *= s;

        let vxx = self.cov[ix * dim + ix];
        let vxp = self.cov[ix * dim + ip];
        let vpp = self.cov[ip * dim + ip];

        let rx = ix * dim;
        let rp = ip * dim;
        for j in 0..dim {
            self.cov[rx + j] *= s;
            self.cov[rp + j] *= s;
        }
        for j in 0..dim {
            self.cov[j * dim + ix] = self.cov[rx + j];
            self.cov[j * dim + ip] = self.cov[rp + j];
        }
        let added = (1.0 - eta) * (2.0 * mean_photons + 1.0);
        self.cov[ix * dim + ix] = eta * vxx + added;
        self.cov[ix * dim + ip] = eta * vxp;
        self.cov[ip * dim + ix] = eta * vxp;
        self.cov[ip * dim + ip] = eta * vpp + added;
        Ok(())
    }

    /// Isotropic Gaussian additive noise on `mode`: `V → V + variance · I₂`.
    pub fn apply_additive_noise(&mut self, mode: usize, variance: f64) -> Result<(), Error> {
        self.check_mode(mode)?;
        if !(variance.is_finite() && variance >= 0.0) {
            return Err(Error::InvalidArgument(format!(
                "additive noise variance must be finite and ≥ 0, got {variance}"
            )));
        }
        let dim = 2 * self.n_modes;
        let ix = mode;
        let ip = self.n_modes + mode;
        self.cov[ix * dim + ix] += variance;
        self.cov[ip * dim + ip] += variance;
        Ok(())
    }

    /// Apply a Gaussian [`CvChannel`](super::CvChannel) on `mode`
    /// (`Loss`, `ThermalLoss`, or `AdditiveNoise`). Rejects non-Gaussian
    /// `Dephasing`.
    pub fn apply_channel(
        &mut self,
        channel: &super::noise::CvChannel,
        mode: usize,
    ) -> Result<(), Error> {
        channel.validate()?;
        match *channel {
            super::noise::CvChannel::Loss { eta } => self.apply_loss(mode, eta),
            super::noise::CvChannel::ThermalLoss { eta, mean_photons } => {
                self.apply_thermal_loss(mode, eta, mean_photons)
            }
            super::noise::CvChannel::AdditiveNoise { variance } => {
                self.apply_additive_noise(mode, variance)
            }
            super::noise::CvChannel::Dephasing { .. } => Err(Error::InvalidArgument(
                "dephasing is non-Gaussian; use FockDensityMatrix or FockState".into(),
            )),
        }
    }

    #[inline]
    fn apply_one_mode_symplectic(&mut self, mode: usize, s: [[f64; 2]; 2]) {
        let dim = 2 * self.n_modes;
        let ix = mode;
        let ip = self.n_modes + mode;

        let mx = self.means[ix];
        let mp = self.means[ip];
        self.means[ix] = s[0][0] * mx + s[0][1] * mp;
        self.means[ip] = s[1][0] * mx + s[1][1] * mp;

        let vxx = self.cov[ix * dim + ix];
        let vxp = self.cov[ix * dim + ip];
        let vpp = self.cov[ip * dim + ip];

        let rx = ix * dim;
        let rp = ip * dim;
        let (s00, s01, s10, s11) = (s[0][0], s[0][1], s[1][0], s[1][1]);
        for j in 0..dim {
            let a = self.cov[rx + j];
            let b = self.cov[rp + j];
            self.cov[rx + j] = s00 * a + s01 * b;
            self.cov[rp + j] = s10 * a + s11 * b;
        }
        for j in 0..dim {
            self.cov[j * dim + ix] = self.cov[rx + j];
            self.cov[j * dim + ip] = self.cov[rp + j];
        }

        // Exact 2×2 diagonal block S V_ii S^T.
        let t00 = s00 * vxx + s01 * vxp;
        let t01 = s00 * vxp + s01 * vpp;
        let t10 = s10 * vxx + s11 * vxp;
        let t11 = s10 * vxp + s11 * vpp;
        let nxx = t00 * s00 + t01 * s01;
        let nxp = t00 * s10 + t01 * s11;
        let npp = t10 * s10 + t11 * s11;
        self.cov[ix * dim + ix] = nxx;
        self.cov[ix * dim + ip] = nxp;
        self.cov[ip * dim + ix] = nxp;
        self.cov[ip * dim + ip] = npp;
    }

    #[inline]
    fn apply_two_mode_symplectic(&mut self, a: usize, b: usize, s: [[f64; 4]; 4]) {
        let m = self.n_modes;
        let dim = 2 * m;
        let idx = [a, b, m + a, m + b];

        let old_m = [
            self.means[idx[0]],
            self.means[idx[1]],
            self.means[idx[2]],
            self.means[idx[3]],
        ];
        for r in 0..4 {
            self.means[idx[r]] =
                s[r][0] * old_m[0] + s[r][1] * old_m[1] + s[r][2] * old_m[2] + s[r][3] * old_m[3];
        }

        let mut old_ii = [[0.0; 4]; 4];
        for r in 0..4 {
            for c in 0..4 {
                old_ii[r][c] = self.cov[idx[r] * dim + idx[c]];
            }
        }

        let (r0, r1, r2, r3) = (idx[0] * dim, idx[1] * dim, idx[2] * dim, idx[3] * dim);
        for j in 0..dim {
            let v0 = self.cov[r0 + j];
            let v1 = self.cov[r1 + j];
            let v2 = self.cov[r2 + j];
            let v3 = self.cov[r3 + j];
            self.cov[r0 + j] = s[0][0] * v0 + s[0][1] * v1 + s[0][2] * v2 + s[0][3] * v3;
            self.cov[r1 + j] = s[1][0] * v0 + s[1][1] * v1 + s[1][2] * v2 + s[1][3] * v3;
            self.cov[r2 + j] = s[2][0] * v0 + s[2][1] * v1 + s[2][2] * v2 + s[2][3] * v3;
            self.cov[r3 + j] = s[3][0] * v0 + s[3][1] * v1 + s[3][2] * v2 + s[3][3] * v3;
        }
        let (c0, c1, c2, c3) = (idx[0], idx[1], idx[2], idx[3]);
        for j in 0..dim {
            let row = j * dim;
            self.cov[row + c0] = self.cov[r0 + j];
            self.cov[row + c1] = self.cov[r1 + j];
            self.cov[row + c2] = self.cov[r2 + j];
            self.cov[row + c3] = self.cov[r3 + j];
        }

        self.write_4x4_transformed(idx, old_ii, s);
    }

    #[inline]
    fn write_4x4_transformed(&mut self, idx: [usize; 4], old: [[f64; 4]; 4], s: [[f64; 4]; 4]) {
        let dim = 2 * self.n_modes;
        let mut tmp = [[0.0; 4]; 4];
        for r in 0..4 {
            for c in 0..4 {
                tmp[r][c] = s[r][0] * old[0][c]
                    + s[r][1] * old[1][c]
                    + s[r][2] * old[2][c]
                    + s[r][3] * old[3][c];
            }
        }
        for r in 0..4 {
            for c in r..4 {
                let val = tmp[r][0] * s[c][0]
                    + tmp[r][1] * s[c][1]
                    + tmp[r][2] * s[c][2]
                    + tmp[r][3] * s[c][3];
                self.cov[idx[r] * dim + idx[c]] = val;
                self.cov[idx[c] * dim + idx[r]] = val;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Homodyne and heterodyne measurements (destructive, reset to vacuum)
    // -----------------------------------------------------------------------

    /// Destructive homodyne measurement of `x̂ cos φ + p̂ sin φ` on `mode`.
    /// Conditions the remaining modes and resets `mode` to the vacuum.
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
        let dim = 2 * self.n_modes;
        let mu = self.means[mode];
        let var = self.cov[mode * dim + mode].max(0.0);
        let z: f64 = StandardNormal.sample(rng);
        let outcome = mu + var.sqrt() * z;
        self.condition_x_and_reset(mode, outcome);
        Ok(outcome)
    }

    /// Postselect a homodyne measurement of `x̂ cos φ + p̂ sin φ` on `mode` onto
    /// `outcome`, conditioning the remaining modes and resetting `mode` to the
    /// vacuum. Returns the marginal probability density `p(outcome)`.
    pub fn postselect_homodyne(
        &mut self,
        mode: usize,
        phi: f64,
        outcome: f64,
    ) -> Result<f64, Error> {
        self.check_mode(mode)?;
        if phi != 0.0 {
            self.rotation(-phi, mode)?;
        }
        let dim = 2 * self.n_modes;
        let mu = self.means[mode];
        let var = self.cov[mode * dim + mode].max(0.0);
        let density = if var > 0.0 {
            let diff = outcome - mu;
            (-0.5 * diff * diff / var).exp() / (std::f64::consts::TAU * var).sqrt()
        } else {
            0.0
        };
        self.condition_x_and_reset(mode, outcome);
        Ok(density)
    }

    fn condition_x_and_reset(&mut self, mode: usize, outcome: f64) {
        let dim = 2 * self.n_modes;
        let ix = mode;
        let ip = self.n_modes + mode;
        let mu = self.means[ix];
        let var = self.cov[ix * dim + ix];

        if var > 1e-30 {
            let inv_var = 1.0 / var;
            let shift = (outcome - mu) * inv_var;
            let mut v = vec![0.0; dim];
            v.copy_from_slice(&self.cov[ix * dim..(ix + 1) * dim]);
            v[ix] = 0.0;
            v[ip] = 0.0;

            for i in 0..dim {
                let vi = v[i];
                if vi != 0.0 {
                    self.means[i] += shift * vi;
                    let scale = vi * inv_var;
                    let row = i * dim;
                    for j in 0..dim {
                        self.cov[row + j] -= scale * v[j];
                    }
                }
            }
        }

        self.reset_mode_to_vacuum(mode);
    }

    /// Destructive heterodyne measurement on `mode`, returning `(Re α, Im α)`.
    /// Conditions the remaining modes and resets `mode` to the vacuum.
    pub fn measure_heterodyne<R: Rng + ?Sized>(
        &mut self,
        mode: usize,
        rng: &mut R,
    ) -> Result<(f64, f64), Error> {
        self.check_mode(mode)?;
        let dim = 2 * self.n_modes;
        let ix = mode;
        let ip = self.n_modes + mode;

        let mx = self.means[ix];
        let mp = self.means[ip];
        let a = (self.cov[ix * dim + ix] + 1.0).max(0.0);
        let b = self.cov[ix * dim + ip];
        let c = (self.cov[ip * dim + ip] + 1.0).max(0.0);

        // 2×2 Cholesky of V_B + I_2.
        let l00 = a.sqrt();
        let l10 = if l00 > 0.0 { b / l00 } else { 0.0 };
        let l11 = (c - l10 * l10).max(0.0).sqrt();

        let z0: f64 = StandardNormal.sample(rng);
        let z1: f64 = StandardNormal.sample(rng);
        let rx = mx + l00 * z0;
        let rp = mp + l10 * z0 + l11 * z1;

        self.condition_heterodyne_and_reset(mode, rx, rp);
        Ok((0.5 * rx, 0.5 * rp))
    }

    /// Postselect a heterodyne measurement on `mode` onto `α = re + i·im`,
    /// conditioning the remaining modes and resetting `mode` to the vacuum.
    /// Returns the probability density `p(Re α, Im α)` (the Husimi Q function
    /// times `π`—wait: normalised over `d(Re α) d(Im α)` so `∫ p d²α = 1`,
    /// i.e. `Q(α) = ⟨α|ρ|α⟩ / π`).
    pub fn postselect_heterodyne(&mut self, mode: usize, re: f64, im: f64) -> Result<f64, Error> {
        self.check_mode(mode)?;
        let dim = 2 * self.n_modes;
        let ix = mode;
        let ip = self.n_modes + mode;

        let rx = 2.0 * re;
        let rp = 2.0 * im;
        let dx = rx - self.means[ix];
        let dp = rp - self.means[ip];

        let a = self.cov[ix * dim + ix] + 1.0;
        let b = self.cov[ix * dim + ip];
        let c = self.cov[ip * dim + ip] + 1.0;
        let det = a * c - b * b;
        // Jacobian |∂(rx, rp)/∂(Re α, Im α)| = 4:
        // p(Re α, Im α) = 4 / (2π √det) exp(-½ Δ^T (V_B + I)^{-1} Δ).
        let density = if det > 0.0 {
            let quad = (c * dx * dx - 2.0 * b * dx * dp + a * dp * dp) / det;
            (4.0 / (std::f64::consts::TAU * det.sqrt())) * (-0.5 * quad).exp()
        } else {
            0.0
        };

        self.condition_heterodyne_and_reset(mode, rx, rp);
        Ok(density)
    }

    fn condition_heterodyne_and_reset(&mut self, mode: usize, rx: f64, rp: f64) {
        let dim = 2 * self.n_modes;
        let ix = mode;
        let ip = self.n_modes + mode;
        let dx = rx - self.means[ix];
        let dp = rp - self.means[ip];

        let a = self.cov[ix * dim + ix] + 1.0;
        let b = self.cov[ix * dim + ip];
        let c = self.cov[ip * dim + ip] + 1.0;
        let det = a * c - b * b;

        if det > 1e-30 {
            let inv_det = 1.0 / det;
            let w00 = c * inv_det;
            let w01 = -b * inv_det;
            let w11 = a * inv_det;

            let mut cx = vec![0.0; dim];
            let mut cp = vec![0.0; dim];
            cx.copy_from_slice(&self.cov[ix * dim..(ix + 1) * dim]);
            cp.copy_from_slice(&self.cov[ip * dim..(ip + 1) * dim]);
            cx[ix] = 0.0;
            cx[ip] = 0.0;
            cp[ix] = 0.0;
            cp[ip] = 0.0;

            let u0 = w00 * dx + w01 * dp;
            let u1 = w01 * dx + w11 * dp;

            for i in 0..dim {
                let cxi = cx[i];
                let cpi = cp[i];
                if cxi == 0.0 && cpi == 0.0 {
                    continue;
                }
                self.means[i] += cxi * u0 + cpi * u1;
                let t0 = cxi * w00 + cpi * w01;
                let t1 = cxi * w01 + cpi * w11;
                let row = i * dim;
                for j in 0..dim {
                    self.cov[row + j] -= t0 * cx[j] + t1 * cp[j];
                }
            }
        }

        self.reset_mode_to_vacuum(mode);
    }

    fn reset_mode_to_vacuum(&mut self, mode: usize) {
        let dim = 2 * self.n_modes;
        let ix = mode;
        let ip = self.n_modes + mode;
        self.means[ix] = 0.0;
        self.means[ip] = 0.0;
        for j in 0..dim {
            self.cov[ix * dim + j] = 0.0;
            self.cov[ip * dim + j] = 0.0;
            self.cov[j * dim + ix] = 0.0;
            self.cov[j * dim + ip] = 0.0;
        }
        self.cov[ix * dim + ix] = 1.0;
        self.cov[ip * dim + ip] = 1.0;
    }
}

// ---------------------------------------------------------------------------
// Fast-path terminal Gaussian measurement sampler (single Cholesky)
// ---------------------------------------------------------------------------

struct TerminalGaussianPlan {
    n_results: usize,
    means: Vec<f64>,
    chol: Vec<f64>,
    scales: Vec<f64>,
}

impl TerminalGaussianPlan {
    fn try_build(prefix: &GaussianState, suffix: &[ModeOp]) -> Option<Self> {
        let m = prefix.n_modes;
        let mut seen = vec![false; m];
        let mut ops_info = Vec::with_capacity(suffix.len());

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
            match *measurement {
                Measurement::Homodyne { phi } => {
                    let angle = phi.value()?;
                    ops_info.push((*mode, Some(angle)));
                }
                Measurement::Heterodyne => {
                    ops_info.push((*mode, None));
                }
                _ => return None,
            }
        }

        // Rotate homodyne modes on a scratch copy so all measured quadratures
        // are plain x_i (or (x_i, p_i) for heterodyne).
        let mut st = prefix.clone();
        let mut indices = Vec::new();
        let mut scales = Vec::new();
        let mut add_het = Vec::new();

        for &(mode, homodyne_phi) in &ops_info {
            match homodyne_phi {
                Some(phi) => {
                    if phi != 0.0 {
                        st.rotation(-phi, mode).ok()?;
                    }
                    indices.push(mode);
                    scales.push(1.0);
                    add_het.push(false);
                }
                None => {
                    indices.push(mode);
                    indices.push(m + mode);
                    scales.push(0.5);
                    scales.push(0.5);
                    add_het.push(true);
                    add_het.push(true);
                }
            }
        }

        let k = indices.len();
        let dim = 2 * m;
        let means: Vec<f64> = indices.iter().map(|&i| st.means[i]).collect();
        let mut sub_cov = vec![0.0; k * k];
        for r in 0..k {
            for c in 0..k {
                let mut v = st.cov[indices[r] * dim + indices[c]];
                if r == c && add_het[r] {
                    v += 1.0;
                }
                sub_cov[r * k + c] = v;
            }
        }
        let chol = cholesky_real_psd(&sub_cov, k);

        Some(TerminalGaussianPlan {
            n_results: k,
            means,
            chol,
            scales,
        })
    }

    fn draw_one<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec<f64> {
        let k = self.n_results;
        let mut z = vec![0.0; k];
        for zi in &mut z {
            *zi = StandardNormal.sample(rng);
        }
        let mut out = vec![0.0; k];
        for r in 0..k {
            let mut acc = self.means[r];
            let row = r * k;
            for c in 0..=r {
                acc += self.chol[row + c] * z[c];
            }
            out[r] = acc * self.scales[r];
        }
        out
    }

    fn sample<R: Rng + ?Sized>(&self, shots: usize, rng: &mut R) -> Vec<Vec<f64>> {
        if shots >= SAMPLE_PAR_MIN_SHOTS {
            let seeds: Vec<u64> = (0..shots).map(|_| rng.random()).collect();
            seeds
                .into_par_iter()
                .map(|seed| {
                    let mut local_rng = StdRng::seed_from_u64(seed);
                    self.draw_one(&mut local_rng)
                })
                .collect()
        } else {
            (0..shots).map(|_| self.draw_one(rng)).collect()
        }
    }
}

// ---------------------------------------------------------------------------
// Dense linear-algebra helpers (real & Hermitian Cholesky)
// ---------------------------------------------------------------------------

/// Lower-triangular Cholesky factor `L` (`L L^T = A`) of a `dim × dim` real
/// symmetric positive-definite matrix.
fn cholesky_real(a: &[f64], dim: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0; dim * dim];
    for i in 0..dim {
        for j in 0..i {
            let mut s = a[i * dim + j];
            for k in 0..j {
                s -= l[i * dim + k] * l[j * dim + k];
            }
            l[i * dim + j] = s / l[j * dim + j];
        }
        let mut diag = a[i * dim + i];
        for k in 0..i {
            let v = l[i * dim + k];
            diag -= v * v;
        }
        if diag <= 0.0 || !diag.is_finite() {
            return None;
        }
        l[i * dim + i] = diag.sqrt();
    }
    Some(l)
}

/// Lower-triangular Cholesky factor tolerating zero-variance (semi-definite)
/// rows, used when sampling terminal measurements.
fn cholesky_real_psd(a: &[f64], dim: usize) -> Vec<f64> {
    let mut l = vec![0.0; dim * dim];
    for i in 0..dim {
        for j in 0..i {
            let dj = l[j * dim + j];
            if dj <= 1e-15 {
                l[i * dim + j] = 0.0;
                continue;
            }
            let mut s = a[i * dim + j];
            for k in 0..j {
                s -= l[i * dim + k] * l[j * dim + k];
            }
            l[i * dim + j] = s / dj;
        }
        let mut diag = a[i * dim + i];
        for k in 0..i {
            let v = l[i * dim + k];
            diag -= v * v;
        }
        l[i * dim + i] = diag.max(0.0).sqrt();
    }
    l
}

/// Forward substitution `L y = b` for lower-triangular `L`.
fn forward_solve_real(l: &[f64], dim: usize, b: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; dim];
    for i in 0..dim {
        let mut s = b[i];
        let row = i * dim;
        for j in 0..i {
            s -= l[row + j] * y[j];
        }
        y[i] = s / l[row + i];
    }
    y
}

/// Lower-triangular Cholesky factor `L` (`L L† = A`) of a `dim × dim` complex
/// Hermitian positive-definite matrix.
fn cholesky_herm(a: &[C64], dim: usize) -> Option<Vec<C64>> {
    let mut l = vec![C64::new(0.0, 0.0); dim * dim];
    for i in 0..dim {
        for j in 0..i {
            let mut s = a[i * dim + j];
            for k in 0..j {
                s -= l[i * dim + k] * l[j * dim + k].conj();
            }
            l[i * dim + j] = s / l[j * dim + j].re;
        }
        let mut diag = a[i * dim + i].re;
        for k in 0..i {
            diag -= l[i * dim + k].norm_sqr();
        }
        if diag <= 0.0 || !diag.is_finite() {
            return None;
        }
        l[i * dim + i] = C64::new(diag.sqrt(), 0.0);
    }
    Some(l)
}

/// Invert a Hermitian PD matrix given its lower-triangular Cholesky factor `L`.
fn invert_from_cholesky_herm(l: &[C64], dim: usize) -> Vec<C64> {
    // First compute L^{-1} (lower triangular).
    let mut linv = vec![C64::new(0.0, 0.0); dim * dim];
    for j in 0..dim {
        linv[j * dim + j] = C64::new(1.0 / l[j * dim + j].re, 0.0);
        for i in (j + 1)..dim {
            let mut s = C64::new(0.0, 0.0);
            for k in j..i {
                s -= l[i * dim + k] * linv[k * dim + j];
            }
            linv[i * dim + j] = s / l[i * dim + i].re;
        }
    }
    // A^{-1} = (L^{-1})† L^{-1}.
    let mut inv = vec![C64::new(0.0, 0.0); dim * dim];
    for i in 0..dim {
        for j in i..dim {
            let mut s = C64::new(0.0, 0.0);
            for k in j..dim {
                s += linv[k * dim + i].conj() * linv[k * dim + j];
            }
            inv[i * dim + j] = s;
            inv[j * dim + i] = s.conj();
        }
    }
    inv
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cv::reference::FockReference;
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() < tol, "{a} vs {b} (diff {})", (a - b).abs());
    }

    #[test]
    fn single_mode_gates_match_fock_reference() {
        let mut c = ModeCircuit::new(1);
        c.squeeze(0.45, 0.7, 0)
            .displace(0.5, -0.3, 0)
            .quadratic_phase(0.35, 0)
            .rotation(0.6, 0);

        let g = GaussianState::from_circuit(&c).unwrap();
        let f = FockReference::run(&c, 45).unwrap();
        close(f.norm_sqr(), 1.0, 1e-9);

        close(g.mean_x(0), f.mean_x(0), 1e-9);
        close(g.mean_p(0), f.mean_p(0), 1e-9);
        close(g.mean_photons(0), f.mean_photons(0), 1e-9);
        close(g.purity(), 1.0, 1e-11);

        // Variances: V_xx = ⟨x²⟩ - ⟨x⟩², V_pp = ⟨p²⟩ - ⟨p⟩².
        close(g.cov()[0], f.mean_x2(0) - f.mean_x(0).powi(2), 1e-8);
        close(g.cov()[3], f.mean_p2(0) - f.mean_p(0).powi(2), 1e-8);

        // Photon-number probabilities P(0..8) via loop hafnian.
        for n in 0..8 {
            close(g.fock_probability(&[n]).unwrap(), f.probability(&[n]), 1e-9);
        }
        // Threshold probabilities P(0) and P(click) = 1 - P(0).
        close(
            g.threshold_probability(&[false]).unwrap(),
            f.probability(&[0]),
            1e-10,
        );
        close(
            g.threshold_probability(&[true]).unwrap(),
            1.0 - f.probability(&[0]),
            1e-10,
        );
    }

    #[test]
    fn two_mode_gaussian_circuit_matches_fock_reference() {
        let mut c = ModeCircuit::new(2);
        c.set_input(0, ModeState::Squeezed { r: 0.35, phi: 0.4 })
            .unwrap()
            .set_input(1, ModeState::Coherent { r: 0.4, phi: -0.5 })
            .unwrap()
            .two_mode_squeeze(0.3, 0.9, 0, 1)
            .beam_splitter(0.6, -0.4, 0, 1)
            .cx(0.25, 0, 1)
            .cz(-0.2, 1, 0)
            .displace(0.25, 1.1, 0);

        let g = GaussianState::from_circuit(&c).unwrap();
        let f = FockReference::run(&c, 22).unwrap();
        close(f.norm_sqr(), 1.0, 1e-7);

        for m in 0..2 {
            close(g.mean_x(m), f.mean_x(m), 1e-6);
            close(g.mean_p(m), f.mean_p(m), 1e-6);
            close(g.mean_photons(m), f.mean_photons(m), 1e-6);
        }
        close(g.purity(), 1.0, 1e-11);

        // Check Fock probabilities on all patterns up to 3 photons per mode.
        for n0 in 0..4 {
            for n1 in 0..4 {
                close(
                    g.fock_probability(&[n0, n1]).unwrap(),
                    f.probability(&[n0, n1]),
                    1e-6,
                );
            }
        }

        // Check all 4 threshold click patterns against summed Fock probabilities.
        for c0 in [false, true] {
            for c1 in [false, true] {
                let mut want = 0.0;
                for n0 in 0..22 {
                    if (n0 > 0) != c0 {
                        continue;
                    }
                    for n1 in 0..22 {
                        if (n1 > 0) != c1 {
                            continue;
                        }
                        want += f.probability(&[n0, n1]);
                    }
                }
                close(g.threshold_probability(&[c0, c1]).unwrap(), want, 1e-6);
            }
        }
    }

    #[test]
    fn undisplaced_gbs_3mode_matches_fock_reference() {
        let mut c = ModeCircuit::new(3);
        c.squeeze(0.3, 0.2, 0)
            .squeeze(0.25, -0.5, 1)
            .squeeze(0.2, 1.1, 2)
            .beam_splitter(0.7, 0.3, 0, 1)
            .beam_splitter(0.5, -0.6, 1, 2)
            .beam_splitter(0.4, 0.8, 0, 2);

        let g = GaussianState::from_circuit(&c).unwrap();
        let f = FockReference::run(&c, 10).unwrap();
        close(f.norm_sqr(), 1.0, 5e-6);

        for pattern in [
            [0, 0, 0],
            [1, 1, 0],
            [2, 0, 0],
            [1, 0, 1],
            [2, 2, 0],
            [1, 1, 2],
            [1, 0, 0], // odd -> 0
            [1, 1, 1], // odd -> 0
        ] {
            close(
                g.fock_probability(&pattern).unwrap(),
                f.probability(&pattern),
                1e-6,
            );
        }

        // Verify threshold probabilities sum to 1 across all 8 patterns.
        let mut total = 0.0;
        for mask in 0..8usize {
            let clicks = [mask & 1 != 0, (mask >> 1) & 1 != 0, (mask >> 2) & 1 != 0];
            let p = g.threshold_probability(&clicks).unwrap();
            assert!(p >= -1e-12);
            total += p;
        }
        close(total, 1.0, 1e-11);
    }

    #[test]
    fn mixed_thermal_and_loss_probabilities() {
        // Single thermal mode: P(n) = n̄^n / (n̄ + 1)^{n+1}.
        let nbar = 0.8;
        let mut c = ModeCircuit::new(1);
        c.set_input(0, ModeState::Thermal { mean_photons: nbar })
            .unwrap();
        let g = GaussianState::from_circuit(&c).unwrap();
        close(g.purity(), 1.0 / (2.0 * nbar + 1.0), 1e-12);
        for n in 0..6 {
            let want = nbar.powi(n as i32) / (nbar + 1.0).powi(n as i32 + 1);
            close(g.fock_probability(&[n]).unwrap(), want, 1e-10);
        }

        // Two-mode thermal + squeezing + displacement (mixed displaced state):
        // verify threshold probs sum to 1 and match 1 - P(0) on marginals.
        let mut c2 = ModeCircuit::new(2);
        c2.set_input(0, ModeState::Thermal { mean_photons: 0.4 })
            .unwrap()
            .set_input(1, ModeState::Thermal { mean_photons: 0.3 })
            .unwrap()
            .two_mode_squeeze(0.3, 0.5, 0, 1)
            .beam_splitter(0.4, 0.2, 0, 1)
            .displace(0.3, -0.4, 0)
            .displace(0.2, 0.7, 1);
        let g2 = GaussianState::from_circuit(&c2).unwrap();
        assert!(g2.purity() < 0.9);
        let mut sum_th = 0.0;
        for c0 in [false, true] {
            for c1 in [false, true] {
                sum_th += g2.threshold_probability(&[c0, c1]).unwrap();
            }
        }
        close(sum_th, 1.0, 1e-11);
        // P(click on 0, 0 on 1) + P(0 on 0, 0 on 1) == P_reduced(0 on mode 1).
        let p00 = g2.threshold_probability(&[false, false]).unwrap();
        let p10 = g2.threshold_probability(&[true, false]).unwrap();
        let p_vac_1 = g2.reduced(&[1]).unwrap().vacuum_probability();
        close(p00 + p10, p_vac_1, 1e-11);
        // Check P(0, 0) matches fock_probability(&[0, 0]).
        close(p00, g2.fock_probability(&[0, 0]).unwrap(), 1e-12);
        // Check mixed Fock probabilities sum close to 1.
        let mut sum_fock = 0.0;
        for n0 in 0..5 {
            for n1 in 0..5 {
                let p = g2.fock_probability(&[n0, n1]).unwrap();
                assert!(p >= -1e-12);
                sum_fock += p;
            }
        }
        assert!(
            sum_fock > 0.96 && sum_fock <= 1.0 + 1e-9,
            "sum_fock={sum_fock}"
        );
    }

    #[test]
    fn passive_block_large_and_small_match_beam_splitters() {
        // Test both small (k < 6) and large (k >= 6, faer path) passive blocks.
        let k = 7;
        let mut gates = ModeCircuit::new(k);
        let mut block = ModeCircuit::new(k);
        for m in 0..k {
            let s = ModeState::Squeezed {
                r: 0.2 + 0.05 * m as f64,
                phi: 0.3 * m as f64,
            };
            gates.set_input(m, s.clone()).unwrap();
            block.set_input(m, s).unwrap();
        }
        gates.displace(0.5, 0.2, 0);
        block.displace(0.5, 0.2, 0);

        let mut u = vec![C64::new(0.0, 0.0); k * k];
        for i in 0..k {
            u[i * k + i] = C64::new(1.0, 0.0);
        }
        for i in 0..(k - 1) {
            let theta = 0.3 + 0.1 * i as f64;
            let phi = -0.2 + 0.15 * i as f64;
            gates.beam_splitter(theta, phi, i, i + 1);
            let t = C64::new(theta.cos(), 0.0);
            let r = C64::from_polar(theta.sin(), phi);
            // Left-multiply rows i, i+1 of u by BS matrix.
            for col in 0..k {
                let a = u[i * k + col];
                let b = u[(i + 1) * k + col];
                u[i * k + col] = t * a - r.conj() * b;
                u[(i + 1) * k + col] = r * a + t * b;
            }
        }
        let modes: Vec<usize> = (0..k).collect();
        block.passive(&modes, u).unwrap();

        let g1 = GaussianState::from_circuit(&gates).unwrap();
        let g2 = GaussianState::from_circuit(&block).unwrap();
        for (a, b) in g1.means().iter().zip(g2.means()) {
            close(*a, *b, 1e-11);
        }
        for (a, b) in g1.cov().iter().zip(g2.cov()) {
            close(*a, *b, 1e-11);
        }
    }

    #[test]
    fn cv_teleportation_with_homodyne_feedforward() {
        // Braunstein–Kimble CV quantum teleportation:
        // Mode 0: input coherent state |α⟩ (x_0 = 2 Re α, p_0 = 2 Im α).
        // Modes 1, 2: two-mode squeezed vacuum S_2(r, 0)|0, 0⟩.
        // 50:50 BS on (0, 1), measure x on mode 0 and p on mode 1.
        // Feed-forward displace mode 2 by √2·(m_x + i m_p) / 2 ->
        // in ħ=2: D(r, 0) shifts x by 2r, D(r, π/2) shifts p by 2r,
        // so D(m_x / √2, 0) shifts x by √2 m_x, and D(m_p / √2, π/2) shifts p by √2 m_p!
        // Wait: check sign of BS(π/4, 0):
        //   x_0' = (x_0 - x_1)/√2, x_1' = (x_0 + x_1)/√2
        //   p_0' = (p_0 - p_1)/√2, p_1' = (p_0 + p_1)/√2
        // And S_2(r, 0) correlates x_1 ≈ -x_2 (since x_1' = ch x_1 - sh x_2) and p_1 ≈ +p_2!
        // Check: for S_2(r, 0): x_1 + x_2 = e^{-r}(x_1^0 + x_2^0) → 0, so x_2 ≈ -x_1!
        // And p_1 - p_2 = e^{-r}(p_1^0 - p_2^0) → 0, so p_2 ≈ +p_1!
        // Then:
        //   m_x = x_0' = (x_0 - x_1)/√2 ≈ (x_0 + x_2)/√2 => x_2 ≈ √2 m_x - x_0 (so wait: flip BS or use S_2(r, π)!)
        // With S_2(r, π): x_1 - x_2 → 0 (x_2 ≈ x_1) and p_1 + p_2 → 0 (p_2 ≈ -p_1)!
        // Then:
        //   m_x = (x_0 - x_1)/√2 ≈ (x_0 - x_2)/√2 => x_2 + √2 m_x ≈ x_0!
        //   m_p = (p_0 + p_1)/√2 ≈ (p_0 - p_2)/√2 => p_2 + √2 m_p ≈ p_0!
        // Exact finite-squeezing teleportation identity: with feed-forward gain
        // g = tanh(r) / √2, the conditional state on mode 2 has mean
        // (tanh(r) x_0, tanh(r) p_0) and vacuum covariance I_2 identically,
        // independent of the random homodyne outcomes (mx, mp)!
        let r_sq: f64 = 1.8;
        let alpha_r: f64 = 1.3;
        let alpha_phi: f64 = 0.6;
        let want_x = r_sq.tanh() * 2.0 * alpha_r * alpha_phi.cos();
        let want_p = r_sq.tanh() * 2.0 * alpha_r * alpha_phi.sin();

        let mut c = ModeCircuit::new(3);
        c.set_input(
            0,
            ModeState::Coherent {
                r: alpha_r,
                phi: alpha_phi,
            },
        )
        .unwrap();
        c.two_mode_squeeze(r_sq, PI, 1, 2);
        c.beam_splitter(FRAC_PI_4, 0.0, 0, 1);
        let mx = c.measure_x(0);
        let mp = c.measure_p(1);
        let scale = r_sq.tanh() / 2.0f64.sqrt();
        c.displace(scale * mx, 0.0, 2);
        c.displace(scale * mp, FRAC_PI_2, 2);

        let mut rng = StdRng::seed_from_u64(2026);
        for _ in 0..10 {
            let (st, res) = GaussianState::run(&c, &mut rng).unwrap();
            assert_eq!(res.len(), 2);
            // Measured modes 0 and 1 are reset to vacuum.
            for m in [0, 1] {
                close(st.mean_x(m), 0.0, 1e-12);
                close(st.mean_p(m), 0.0, 1e-12);
                close(st.mean_photons(m), 0.0, 1e-12);
            }
            // Mode 2 matches the exact teleported coherent state to machine precision.
            close(st.mean_x(2), want_x, 1e-11);
            close(st.mean_p(2), want_p, 1e-11);
            let dim = 6;
            close(st.cov()[2 * dim + 2], 1.0, 1e-11);
            close(st.cov()[5 * dim + 5], 1.0, 1e-11);
        }
    }

    #[test]
    fn terminal_and_general_sampling_match_moments() {
        // Two-mode squeezed + displaced state measured with homodyne & heterodyne.
        let mut c = ModeCircuit::new(2);
        c.squeeze(0.5, 0.3, 0)
            .two_mode_squeeze(0.4, -0.2, 0, 1)
            .displace(0.8, 0.5, 0)
            .displace(0.6, -0.7, 1);
        let exact = GaussianState::from_circuit(&c).unwrap();
        c.measure_homodyne(0.4, 0);
        c.measure_heterodyne(1);

        let mut rng = StdRng::seed_from_u64(55);
        let shots = 20_000;
        let samples = GaussianState::sample(&c, shots, &mut rng).unwrap();
        assert_eq!(samples.len(), shots);

        let mean0: f64 = samples.iter().map(|s| s[0]).sum::<f64>() / shots as f64;
        let mean1: f64 = samples.iter().map(|s| s[1]).sum::<f64>() / shots as f64;
        let mean2: f64 = samples.iter().map(|s| s[2]).sum::<f64>() / shots as f64;

        let want0 = exact.mean_x(0) * 0.4f64.cos() + exact.mean_p(0) * 0.4f64.sin();
        let want1 = 0.5 * exact.mean_x(1);
        let want2 = 0.5 * exact.mean_p(1);
        close(mean0, want0, 0.04);
        close(mean1, want1, 0.04);
        close(mean2, want2, 0.04);

        // Heterodyne variance on Re α is (V_xx + 1) / 4.
        let var1: f64 = samples.iter().map(|s| (s[1] - mean1).powi(2)).sum::<f64>() / shots as f64;
        let want_var1 = (exact.cov()[4 + 1] + 1.0) / 4.0;
        close(var1, want_var1, 0.04);
    }

    #[test]
    fn postselect_homodyne_and_heterodyne_match_densities() {
        // Coherent state |α⟩: heterodyne density is Q(β) = exp(-|β - α|²) / π.
        let mut c = ModeCircuit::new(1);
        c.displace(0.9, 0.4, 0);
        let mut g = GaussianState::from_circuit(&c).unwrap();
        let alpha = C64::from_polar(0.9, 0.4);
        let beta = C64::new(0.5, -0.2);
        let dens = g.postselect_heterodyne(0, beta.re, beta.im).unwrap();
        let want = (-(beta - alpha).norm_sqr()).exp() / PI;
        close(dens, want, 1e-12);
        // Mode 0 is now vacuum.
        close(g.mean_photons(0), 0.0, 1e-12);

        // Wigner function of vacuum at (0, 0) is 1 / (2π).
        close(g.wigner(&[0.0], &[0.0]).unwrap(), 1.0 / (2.0 * PI), 1e-12);
    }
}
