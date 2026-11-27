//! Linear-combination-of-Gaussians (LCG) simulator for bosonic qubits
//! (Bourassa et al. 2021, *Fast simulation of bosonic qubits via Gaussian
//! functions in phase space*, Quantum 5, 492), using Strawberry Fields
//! conventions (`ħ = 2`, `xxpp` ordering).
//!
//! Represents an `m`-mode state's Wigner function as a sum of `M` phase-space
//! Gaussians with complex means `μ_k = a_k + i b_k` and real symmetric
//! positive-definite covariances `V_k`:
//!
//! `W(r) = Σ_k c̃_k · G_real(r; a_k, V_k) · exp(i b_k^T V_k^{-1} (r − a_k))`
//!
//! where `c̃_k = c_k · exp(½ b_k^T V_k^{-1} b_k)` is the overflow-free
//! phase-space normalized weight. Under all symplectic Gaussian gates, `c̃_k`
//! is an exact invariant and every common-covariance state (cat, GKP,
//! Gaussian) updates a single shared `2m × 2m` covariance matrix in `O(m)`
//! plus `O(1)` per term.

use super::circuit::{Measurement, ModeCircuit, ModeOp};
use super::gate::{ModeGate, TwoModeGate};
use super::noise::CvChannel;
use super::state::{ModeState, gkp_peaks};
use crate::error::Error;
use faer::Mat;
use num_complex::Complex64 as C64;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::collections::HashMap;
use std::f64::consts::{PI, TAU};

/// Minimum number of terms to parallelise gate/measurement loops with rayon.
const MIXTURE_PAR_MIN_TERMS: usize = 512;

/// Minimum shot count for parallelising independent mixture trajectories.
const SAMPLE_PAR_MIN_SHOTS: usize = 32;

/// Maximum Fock photon number `n` supported by the shared-covariance
/// coherent-state ring decomposition in [`GaussianMixture`].
pub(crate) const MAX_LCG_FOCK_N: usize = 64;

/// Accuracy and pruning settings for [`GaussianMixture`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MixtureOptions {
    /// Relative weight threshold below which GKP ket peaks and post-measurement
    /// terms are pruned (default `1e-6`).
    pub prune_tol: f64,
    /// Distance tolerance for merging duplicate `(a, b)` peaks after
    /// measurements (default `1e-9`).
    pub merge_tol: f64,
    /// Target accuracy scale for Fock input decompositions (default `1e-3`,
    /// targeting `≤ fock_step²` infidelity).
    pub fock_step: f64,
}

impl Default for MixtureOptions {
    fn default() -> Self {
        MixtureOptions {
            prune_tol: 1e-6,
            merge_tol: 1e-9,
            fock_step: 1e-3,
        }
    }
}

/// One Gaussian component in the overflow-free `(c̃, a, b)` representation:
/// `μ = a + i b`, `c̃ = c · exp(½ b^T V^{-1} b)`.
#[derive(Clone, Debug)]
struct Term {
    w_tilde: C64,
    a: Vec<f64>,
    b: Vec<f64>,
    /// Index into `GaussianMixture::covs` (usually `0` for shared-covariance
    /// states such as cat, GKP, coherent, squeezed, and thermal states).
    cov_id: usize,
}

/// Exact / controlled-accuracy Linear-Combination-of-Gaussians simulator
/// (Bourassa et al. 2021) for Gaussian circuits on cat, GKP, Fock, and
/// Gaussian inputs.
#[derive(Clone, Debug)]
pub struct GaussianMixture {
    n_modes: usize,
    /// Distinct real symmetric PD `2m × 2m` covariance matrices (`xxpp` ordering).
    /// Most bosonic qubit states (cat, GKP, Gaussian) have `covs.len() == 1`.
    covs: Vec<Vec<f64>>,
    terms: Vec<Term>,
    options: MixtureOptions,
    /// Cumulative upper bound on L1 weight pruned during preparation and
    /// post-measurement compression.
    error_bound: f64,
}

impl GaussianMixture {
    /// Vacuum state `|0…0⟩` on `n_modes` modes.
    pub fn vacuum(n_modes: usize, options: MixtureOptions) -> Self {
        let dim = 2 * n_modes;
        let mut cov = vec![0.0; dim * dim];
        for i in 0..dim {
            cov[i * dim + i] = 1.0;
        }
        GaussianMixture {
            n_modes,
            covs: vec![cov],
            terms: vec![Term {
                w_tilde: C64::new(1.0, 0.0),
                a: vec![0.0; dim],
                b: vec![0.0; dim],
                cov_id: 0,
            }],
            options,
            error_bound: 0.0,
        }
    }

    /// Build a product `GaussianMixture` from `circuit.inputs()`.
    fn from_inputs(inputs: &[ModeState], options: MixtureOptions) -> Result<Self, Error> {
        let m = inputs.len();
        if m == 0 {
            return Ok(Self::vacuum(0, options));
        }

        // Convert each mode's input state into 1-mode LCG components:
        // (covs_1mode: Vec<[f64; 4]>, terms_1mode: Vec<(w_tilde, [ax, ap], [bx, bp], cov_id)>).
        let mut per_mode = Vec::with_capacity(m);
        let mut total_err = 0.0;
        for (mode, st) in inputs.iter().enumerate() {
            let (m_covs, m_terms, err) = mode_state_to_lcg(st, &options).map_err(|e| {
                Error::InvalidArgument(format!(
                    "mode {mode} input unsupported in GaussianMixture: {e}"
                ))
            })?;
            per_mode.push((m_covs, m_terms));
            total_err += err;
        }

        // Build tensor-product terms, pruning negligible cross-mode products
        // immediately so multi-mode GKP/cat/Fock inputs stay compact with
        // bounded error, then lazily intern only the covariance combinations
        // referenced by surviving terms.
        let dim = 2 * m;
        #[allow(clippy::type_complexity)]
        let mut terms: Vec<(C64, Vec<f64>, Vec<f64>, Vec<usize>)> = vec![(
            C64::new(1.0, 0.0),
            vec![0.0; dim],
            vec![0.0; dim],
            Vec::with_capacity(m),
        )];
        let mut max_prefix = 1.0f64;
        for (mode, (_, m_terms)) in per_mode.iter().enumerate() {
            let max_mode = m_terms
                .iter()
                .map(|t| t.0.norm())
                .fold(0.0f64, f64::max)
                .max(1e-30);
            let next_max = max_prefix * max_mode;
            let prod_cutoff = options.prune_tol * 0.25 * next_max;
            let mut next = Vec::new();
            for (w0, a0, b0, c0) in &terms {
                let w0_norm = w0.norm();
                for &(wt, at, bt, cid) in m_terms {
                    let w = *w0 * wt;
                    let w_norm = w0_norm * wt.norm();
                    if w_norm < prod_cutoff {
                        total_err += w_norm;
                        continue;
                    }
                    let mut a = a0.clone();
                    let mut b = b0.clone();
                    let mut c = c0.clone();
                    a[mode] = at[0];
                    a[m + mode] = at[1];
                    b[mode] = bt[0];
                    b[m + mode] = bt[1];
                    c.push(cid);
                    next.push((w, a, b, c));
                }
            }
            max_prefix = next_max;
            terms = next;
        }

        let mut cov_map: HashMap<Vec<usize>, usize> = HashMap::new();
        let mut covs: Vec<Vec<f64>> = Vec::new();
        let mut final_terms = Vec::with_capacity(terms.len());
        for (w_tilde, a, b, cid_vec) in terms {
            let cov_id = if let Some(&id) = cov_map.get(&cid_vec) {
                id
            } else {
                let id = covs.len();
                let mut full = vec![0.0; dim * dim];
                for (mode, &cid) in cid_vec.iter().enumerate() {
                    let c2 = per_mode[mode].0[cid];
                    let ix = mode;
                    let ip = m + mode;
                    full[ix * dim + ix] = c2[0];
                    full[ix * dim + ip] = c2[1];
                    full[ip * dim + ix] = c2[2];
                    full[ip * dim + ip] = c2[3];
                }
                covs.push(full);
                cov_map.insert(cid_vec, id);
                id
            };
            final_terms.push(Term {
                w_tilde,
                a,
                b,
                cov_id,
            });
        }

        Ok(GaussianMixture {
            n_modes: m,
            covs,
            terms: final_terms,
            options,
            error_bound: total_err,
        })
    }

    /// Run a bound, measurement-free Gaussian circuit on arbitrary supported
    /// inputs (Gaussian, cat, GKP, Fock).
    pub fn from_circuit(circuit: &ModeCircuit, options: MixtureOptions) -> Result<Self, Error> {
        if circuit.n_parameters() > 0 {
            return Err(Error::InvalidArgument(
                "circuit has unbound parameters; bind it first".into(),
            ));
        }
        if circuit.has_measurements() {
            return Err(Error::InvalidArgument(
                "circuit has measurements; use GaussianMixture::run or GaussianMixture::sample"
                    .into(),
            ));
        }
        let mut st = Self::from_inputs(circuit.inputs(), options)?;
        for op in circuit.ops() {
            st.apply_op(op, &[])?;
        }
        Ok(st)
    }

    /// Run one shot of a bound Gaussian circuit with homodyne / heterodyne
    /// measurements and feed-forward.
    pub fn run<R: Rng + ?Sized>(
        circuit: &ModeCircuit,
        options: MixtureOptions,
        rng: &mut R,
    ) -> Result<(Self, Vec<f64>), Error> {
        if circuit.n_parameters() > 0 {
            return Err(Error::InvalidArgument(
                "circuit has unbound parameters; bind it first".into(),
            ));
        }
        let mut st = Self::from_inputs(circuit.inputs(), options)?;
        let mut results = Vec::with_capacity(circuit.n_results());
        st.run_ops(circuit.ops(), &mut results, rng)?;
        Ok((st, results))
    }

    /// Sample `shots` independent measurement trajectories from a bound
    /// Gaussian circuit on cat / GKP / Fock / Gaussian inputs.
    pub fn sample<R: Rng + ?Sized>(
        circuit: &ModeCircuit,
        options: MixtureOptions,
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

        let mut prefix = Self::from_inputs(circuit.inputs(), options)?;
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
                    _ => {
                        return Err(Error::InvalidArgument(
                            "GaussianMixture supports homodyne and heterodyne measurements".into(),
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
                _ => Err(Error::InvalidArgument(format!(
                    "gate {gate:?} is not Gaussian; GaussianMixture runs Gaussian circuits"
                ))),
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
                    _ => Err(Error::InvalidArgument(format!(
                        "gate {gate:?} is not Gaussian; GaussianMixture runs Gaussian circuits"
                    ))),
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

    /// Number of Gaussian components currently in the mixture.
    pub fn n_terms(&self) -> usize {
        self.terms.len()
    }

    /// Cumulative error bound from peak truncation and post-measurement pruning.
    pub fn error_bound(&self) -> f64 {
        self.error_bound
    }

    /// Evaluate the Wigner function `W(x, p)` at phase-space coordinates
    /// `(x, p)`.
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
        let mut r_vec = Vec::with_capacity(dim);
        r_vec.extend_from_slice(x);
        r_vec.extend_from_slice(p);

        // Precompute Cholesky factor and log-normalization for each distinct covariance.
        let chols: Vec<(Vec<f64>, f64)> = self
            .covs
            .iter()
            .map(|v| {
                let l = cholesky_real(v, dim).ok_or_else(|| {
                    Error::InvalidArgument("mixture covariance is not positive definite".into())
                })?;
                let log_sqrt_det: f64 = (0..dim).map(|i| l[i * dim + i].ln()).sum();
                let log_norm = (self.n_modes as f64) * TAU.ln() + log_sqrt_det;
                Ok((l, log_norm))
            })
            .collect::<Result<Vec<_>, Error>>()?;

        let eval_term = |t: &Term| -> f64 {
            let (l, log_norm) = &chols[t.cov_id];
            let diff: Vec<f64> = r_vec.iter().zip(&t.a).map(|(&ri, &ai)| ri - ai).collect();
            let y_diff = forward_solve_real(l, dim, &diff);
            let quad: f64 = y_diff.iter().map(|v| v * v).sum();
            let phase = if t.b.iter().any(|&bi| bi != 0.0) {
                let y_b = forward_solve_real(l, dim, &t.b);
                y_b.iter().zip(&y_diff).map(|(&u, &v)| u * v).sum::<f64>()
            } else {
                0.0
            };
            let mag = (-0.5 * quad - log_norm).exp();
            (t.w_tilde * C64::from_polar(mag, phase)).re
        };

        let val: f64 = if self.terms.len() >= MIXTURE_PAR_MIN_TERMS {
            self.terms.par_iter().map(eval_term).sum()
        } else {
            self.terms.iter().map(eval_term).sum()
        };
        Ok(val)
    }

    /// Marginal homodyne probability density `p(q)` of `x̂_mode cos φ + p̂_mode sin φ`
    /// evaluated at quadrature value `q`.
    pub fn homodyne_pdf(&self, mode: usize, phi: f64, q: f64) -> Result<f64, Error> {
        self.check_mode(mode)?;
        if phi != 0.0 {
            let mut copy = self.clone();
            copy.rotation(-phi, mode)?;
            return copy.homodyne_x_pdf_batch(mode, &[q]).map(|v| v[0]);
        }
        self.homodyne_x_pdf_batch(mode, &[q]).map(|v| v[0])
    }

    /// Exact marginal `x̂` PDF evaluated at a slice of quadrature points `qs`.
    fn homodyne_x_pdf_batch(&self, mode: usize, qs: &[f64]) -> Result<Vec<f64>, Error> {
        let dim = 2 * self.n_modes;
        let ix = mode;
        // Precompute per-covariance Schur complement of all indices EXCEPT `ix`
        // so we evaluate the exact damping factor `exp(-½ b'_rest^T V'_{rest}^{-1} b'_rest)`
        // once per term (O(M * m²) total, independent of `qs.len()`!).
        let rest_idx: Vec<usize> = (0..dim).filter(|&i| i != ix).collect();
        let rest_dim = rest_idx.len();

        struct CovMarginal {
            inv_sigma2: f64,
            inv_sqrt_2pi_s2: f64,
            v_col: Vec<f64>,
            chol_rest: Vec<f64>,
        }

        let cov_info: Vec<CovMarginal> = self
            .covs
            .iter()
            .map(|v| {
                let sigma2 = v[ix * dim + ix].max(1e-30);
                let inv_sigma2 = 1.0 / sigma2;
                let v_col: Vec<f64> = rest_idx.iter().map(|&i| v[i * dim + ix]).collect();
                let mut v_rest = vec![0.0; rest_dim * rest_dim];
                for r in 0..rest_dim {
                    for c in 0..rest_dim {
                        v_rest[r * rest_dim + c] =
                            v[rest_idx[r] * dim + rest_idx[c]] - v_col[r] * v_col[c] * inv_sigma2;
                    }
                }
                let chol_rest = if rest_dim > 0 {
                    cholesky_real(&v_rest, rest_dim).unwrap_or_else(|| {
                        // Tiny numerical floor if nearly singular
                        let mut vr = v_rest;
                        for i in 0..rest_dim {
                            vr[i * rest_dim + i] += 1e-13;
                        }
                        cholesky_real(&vr, rest_dim).expect("Schur complement must be PD")
                    })
                } else {
                    Vec::new()
                };
                CovMarginal {
                    inv_sigma2,
                    inv_sqrt_2pi_s2: 1.0 / (TAU * sigma2).sqrt(),
                    v_col,
                    chol_rest,
                }
            })
            .collect();

        // Precompute damped effective weight `eff_w[k]`, `a_m`, `b_m`, `inv_s2`, `norm` per term.
        let prep_term = |t: &Term| -> Option<(C64, f64, f64, f64, f64)> {
            let ci = &cov_info[t.cov_id];
            let bm = t.b[ix];
            let damp = if rest_dim > 0 && t.b.iter().any(|&bi| bi != 0.0) {
                let q_rest = if rest_dim <= 16 {
                    let mut b_rest = [0.0f64; 16];
                    for (r, &i) in rest_idx.iter().enumerate() {
                        b_rest[r] = t.b[i] - bm * ci.inv_sigma2 * ci.v_col[r];
                    }
                    forward_solve_norm_sqr(&ci.chol_rest, rest_dim, &b_rest[..rest_dim])
                } else {
                    let b_rest: Vec<f64> = rest_idx
                        .iter()
                        .enumerate()
                        .map(|(r, &i)| t.b[i] - bm * ci.inv_sigma2 * ci.v_col[r])
                        .collect();
                    forward_solve_norm_sqr(&ci.chol_rest, rest_dim, &b_rest)
                };
                (-0.5 * q_rest).exp()
            } else {
                1.0
            };
            let eff_w = t.w_tilde * damp;
            if eff_w.norm() < 1e-18 {
                None
            } else {
                Some((eff_w, t.a[ix], bm, ci.inv_sigma2, ci.inv_sqrt_2pi_s2))
            }
        };
        let term_1d: Vec<(C64, f64, f64, f64, f64)> = if self.terms.len() >= MIXTURE_PAR_MIN_TERMS {
            self.terms.par_iter().filter_map(prep_term).collect()
        } else {
            self.terms.iter().filter_map(prep_term).collect()
        };

        let eval_q = |&q: &f64| -> f64 {
            let mut sum = 0.0;
            for &(ew, am, bm, inv_s2, norm) in &term_1d {
                let d = q - am;
                let g = norm * (-0.5 * d * d * inv_s2).exp();
                let ph = bm * d * inv_s2;
                sum += (ew * C64::from_polar(g, ph)).re;
            }
            sum
        };

        let out = if qs.len() >= 128 && term_1d.len() >= 64 {
            qs.par_iter().map(eval_q).collect()
        } else {
            qs.iter().map(eval_q).collect()
        };
        Ok(out)
    }

    /// `⟨x̂⟩` on `mode`.
    pub fn mean_x(&self, mode: usize) -> Result<f64, Error> {
        self.check_mode(mode)?;
        Ok(self.quadrature_mean(mode))
    }

    /// `⟨p̂⟩` on `mode`.
    pub fn mean_p(&self, mode: usize) -> Result<f64, Error> {
        self.check_mode(mode)?;
        Ok(self.quadrature_mean(self.n_modes + mode))
    }

    /// Mean photon number `⟨n̂⟩ = ¼(⟨x̂²⟩ + ⟨p̂²⟩ − 2)` on `mode`.
    pub fn mean_photons(&self, mode: usize) -> Result<f64, Error> {
        self.check_mode(mode)?;
        let ix = mode;
        let ip = self.n_modes + mode;
        let x2 = self.quadrature_second_moment(ix);
        let p2 = self.quadrature_second_moment(ip);
        Ok(0.25 * (x2 + p2 - 2.0))
    }

    fn true_weights(&self) -> Vec<C64> {
        let dim = 2 * self.n_modes;
        let chols: Vec<Vec<f64>> = self
            .covs
            .iter()
            .map(|v| cholesky_real(v, dim).expect("covariance is PD"))
            .collect();
        let compute = |t: &Term| {
            if t.b.iter().all(|&bi| bi == 0.0) {
                t.w_tilde
            } else {
                let q = forward_solve_norm_sqr(&chols[t.cov_id], dim, &t.b);
                t.w_tilde * (-0.5 * q).exp()
            }
        };
        if self.terms.len() >= MIXTURE_PAR_MIN_TERMS {
            self.terms.par_iter().map(compute).collect()
        } else {
            self.terms.iter().map(compute).collect()
        }
    }

    fn quadrature_mean(&self, coord: usize) -> f64 {
        let ws = self.true_weights();
        let mut total = C64::new(0.0, 0.0);
        for (t, &c) in self.terms.iter().zip(&ws) {
            total += c * C64::new(t.a[coord], t.b[coord]);
        }
        total.re
    }

    fn quadrature_second_moment(&self, coord: usize) -> f64 {
        let dim = 2 * self.n_modes;
        let ws = self.true_weights();
        let mut total = C64::new(0.0, 0.0);
        for (t, &c) in self.terms.iter().zip(&ws) {
            let mu = C64::new(t.a[coord], t.b[coord]);
            let v_cc = self.covs[t.cov_id][coord * dim + coord];
            total += c * (mu * mu + v_cc);
        }
        total.re
    }

    // -----------------------------------------------------------------------
    // Gaussian gates (O(m) per distinct covariance + O(M) per term)
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

    /// Rotation `R(φ)` on `mode`.
    pub fn rotation(&mut self, phi: f64, mode: usize) -> Result<(), Error> {
        self.check_mode(mode)?;
        let (s, c) = phi.sin_cos();
        self.apply_one_mode_symplectic(mode, [[c, -s], [s, c]]);
        Ok(())
    }

    /// Displacement `D(r e^{iφ})` on `mode`.
    pub fn displace(&mut self, r: f64, phi: f64, mode: usize) -> Result<(), Error> {
        self.check_mode(mode)?;
        let (s, c) = phi.sin_cos();
        let dx = 2.0 * r * c;
        let dp = 2.0 * r * s;
        let ip = self.n_modes + mode;
        for t in &mut self.terms {
            t.a[mode] += dx;
            t.a[ip] += dp;
        }
        Ok(())
    }

    /// Squeezing `S(r e^{iφ})` on `mode`.
    pub fn squeeze(&mut self, r: f64, phi: f64, mode: usize) -> Result<(), Error> {
        self.check_mode(mode)?;
        let ch = r.cosh();
        let sh = r.sinh();
        let (s, c) = phi.sin_cos();
        let off = -s * sh;
        self.apply_one_mode_symplectic(mode, [[ch - c * sh, off], [off, ch + c * sh]]);
        Ok(())
    }

    /// Quadratic phase `P(s)` on `mode`.
    pub fn quadratic_phase(&mut self, s: f64, mode: usize) -> Result<(), Error> {
        self.check_mode(mode)?;
        if s == 0.0 {
            return Ok(());
        }
        self.apply_one_mode_symplectic(mode, [[1.0, 0.0], [s, 1.0]]);
        Ok(())
    }

    /// Beam splitter `BS(θ, φ)` on `(a, b)`.
    pub fn beam_splitter(&mut self, theta: f64, phi: f64, a: usize, b: usize) -> Result<(), Error> {
        self.check_pair(a, b)?;
        let (st, t) = theta.sin_cos();
        let (sp, cp) = phi.sin_cos();
        let c = cp * st;
        let s = sp * st;
        self.apply_two_mode_symplectic(
            a,
            b,
            [
                [t, -c, 0.0, -s],
                [c, t, -s, 0.0],
                [0.0, s, t, -c],
                [s, 0.0, c, t],
            ],
        );
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
        self.apply_two_mode_symplectic(
            a,
            b,
            [
                [ch, -sc, 0.0, -ss],
                [-sc, ch, -ss, 0.0],
                [0.0, -ss, ch, sc],
                [-ss, 0.0, sc, ch],
            ],
        );
        Ok(())
    }

    /// `CX(s)` on `(a, b)`.
    pub fn cx(&mut self, s: f64, a: usize, b: usize) -> Result<(), Error> {
        self.check_pair(a, b)?;
        if s == 0.0 {
            return Ok(());
        }
        self.apply_two_mode_symplectic(
            a,
            b,
            [
                [1.0, 0.0, 0.0, 0.0],
                [s, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, -s],
                [0.0, 0.0, 0.0, 1.0],
            ],
        );
        Ok(())
    }

    /// `CZ(s)` on `(a, b)`.
    pub fn cz(&mut self, s: f64, a: usize, b: usize) -> Result<(), Error> {
        self.check_pair(a, b)?;
        if s == 0.0 {
            return Ok(());
        }
        self.apply_two_mode_symplectic(
            a,
            b,
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, s, 1.0, 0.0],
                [s, 0.0, 0.0, 1.0],
            ],
        );
        Ok(())
    }

    /// Passive linear-optics unitary `U` on `modes`.
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
        let m = self.n_modes;
        let dim = 2 * m;
        let mut idx = Vec::with_capacity(s_dim);
        for &md in modes {
            idx.push(md);
        }
        for &md in modes {
            idx.push(m + md);
        }
        // Update each distinct covariance matrix.
        let s_mat = Mat::from_fn(s_dim, s_dim, |r, c| s[r * s_dim + c]);
        for cov in &mut self.covs {
            let v_rows = Mat::from_fn(s_dim, dim, |r, c| cov[idx[r] * dim + c]);
            let sv = &s_mat * &v_rows;
            let sv_ii = Mat::from_fn(s_dim, s_dim, |r, c| sv[(r, idx[c])]);
            let new_ii = &sv_ii * s_mat.transpose();
            for r in 0..s_dim {
                for c in 0..dim {
                    cov[idx[r] * dim + c] = sv[(r, c)];
                    cov[c * dim + idx[r]] = sv[(r, c)];
                }
            }
            for r in 0..s_dim {
                for c in r..s_dim {
                    let avg = 0.5 * (new_ii[(r, c)] + new_ii[(c, r)]);
                    cov[idx[r] * dim + idx[c]] = avg;
                    cov[idx[c] * dim + idx[r]] = avg;
                }
            }
        }
        // Update each term's (a, b).
        let update_term = |t: &mut Term| {
            let old_a: Vec<f64> = idx.iter().map(|&i| t.a[i]).collect();
            let old_b: Vec<f64> = idx.iter().map(|&i| t.b[i]).collect();
            for r in 0..s_dim {
                let mut aa = 0.0;
                let mut bb = 0.0;
                for c in 0..s_dim {
                    let sc = s[r * s_dim + c];
                    aa += sc * old_a[c];
                    bb += sc * old_b[c];
                }
                t.a[idx[r]] = aa;
                t.b[idx[r]] = bb;
            }
        };
        if self.terms.len() >= MIXTURE_PAR_MIN_TERMS {
            self.terms.par_iter_mut().for_each(update_term);
        } else {
            for t in &mut self.terms {
                update_term(t);
            }
        }
        Ok(())
    }

    /// Apply a Gaussian noise channel (`Loss`, `ThermalLoss`, `AdditiveNoise`)
    /// on `mode`, automatically damping and pruning extinguished interference
    /// fringes.
    pub fn apply_channel(&mut self, channel: &CvChannel, mode: usize) -> Result<(), Error> {
        self.check_mode(mode)?;
        channel.validate()?;
        let (eta, added) = match *channel {
            CvChannel::Loss { eta } => (eta, 1.0 - eta),
            CvChannel::ThermalLoss { eta, mean_photons } => {
                (eta, (1.0 - eta) * (2.0 * mean_photons + 1.0))
            }
            CvChannel::AdditiveNoise { variance } => (1.0, variance),
            CvChannel::Dephasing { .. } => {
                return Err(Error::InvalidArgument(
                    "dephasing is non-Gaussian; GaussianMixture supports Loss, ThermalLoss, and AdditiveNoise".into(),
                ));
            }
        };
        let dim = 2 * self.n_modes;
        let ix = mode;
        let ip = self.n_modes + mode;
        let s = eta.sqrt();

        // Precompute old Cholesky factors to compute exact damping `exp(½(b'^T V'^{-1} b' - b^T V^{-1} b))`.
        let old_chols: Vec<Vec<f64>> = self
            .covs
            .iter()
            .map(|v| cholesky_real(v, dim).expect("cov is PD"))
            .collect();

        for cov in &mut self.covs {
            let vxx = cov[ix * dim + ix];
            let vxp = cov[ix * dim + ip];
            let vpp = cov[ip * dim + ip];
            let rx = ix * dim;
            let rp = ip * dim;
            for j in 0..dim {
                cov[rx + j] *= s;
                cov[rp + j] *= s;
            }
            for j in 0..dim {
                cov[j * dim + ix] = cov[rx + j];
                cov[j * dim + ip] = cov[rp + j];
            }
            cov[ix * dim + ix] = eta * vxx + added;
            cov[ix * dim + ip] = eta * vxp;
            cov[ip * dim + ix] = eta * vxp;
            cov[ip * dim + ip] = eta * vpp + added;
        }

        let new_chols: Vec<Vec<f64>> = self
            .covs
            .iter()
            .map(|v| cholesky_real(v, dim).expect("cov is PD"))
            .collect();

        let update_term = |t: &mut Term| {
            let has_b = t.b.iter().any(|&bi| bi != 0.0);
            let old_q = if has_b {
                forward_solve_norm_sqr(&old_chols[t.cov_id], dim, &t.b)
            } else {
                0.0
            };
            t.a[ix] *= s;
            t.a[ip] *= s;
            t.b[ix] *= s;
            t.b[ip] *= s;
            if has_b {
                let new_q = forward_solve_norm_sqr(&new_chols[t.cov_id], dim, &t.b);
                // Guaranteed new_q <= old_q (damping <= 1).
                t.w_tilde *= (0.5 * (new_q - old_q)).exp();
            }
        };
        if self.terms.len() >= MIXTURE_PAR_MIN_TERMS {
            self.terms.par_iter_mut().for_each(update_term);
        } else {
            for t in &mut self.terms {
                update_term(t);
            }
        }
        self.prune_and_merge();
        Ok(())
    }

    fn apply_one_mode_symplectic(&mut self, mode: usize, s: [[f64; 2]; 2]) {
        let dim = 2 * self.n_modes;
        let ix = mode;
        let ip = self.n_modes + mode;
        let (s00, s01, s10, s11) = (s[0][0], s[0][1], s[1][0], s[1][1]);

        for cov in &mut self.covs {
            let vxx = cov[ix * dim + ix];
            let vxp = cov[ix * dim + ip];
            let vpp = cov[ip * dim + ip];
            let rx = ix * dim;
            let rp = ip * dim;
            for j in 0..dim {
                let a = cov[rx + j];
                let b = cov[rp + j];
                cov[rx + j] = s00 * a + s01 * b;
                cov[rp + j] = s10 * a + s11 * b;
            }
            for j in 0..dim {
                cov[j * dim + ix] = cov[rx + j];
                cov[j * dim + ip] = cov[rp + j];
            }
            let t00 = s00 * vxx + s01 * vxp;
            let t01 = s00 * vxp + s01 * vpp;
            let t10 = s10 * vxx + s11 * vxp;
            let t11 = s10 * vxp + s11 * vpp;
            cov[ix * dim + ix] = t00 * s00 + t01 * s01;
            let nxp = t00 * s10 + t01 * s11;
            cov[ix * dim + ip] = nxp;
            cov[ip * dim + ix] = nxp;
            cov[ip * dim + ip] = t10 * s10 + t11 * s11;
        }

        let update = |t: &mut Term| {
            let (ax, ap) = (t.a[ix], t.a[ip]);
            t.a[ix] = s00 * ax + s01 * ap;
            t.a[ip] = s10 * ax + s11 * ap;
            let (bx, bp) = (t.b[ix], t.b[ip]);
            t.b[ix] = s00 * bx + s01 * bp;
            t.b[ip] = s10 * bx + s11 * bp;
        };
        if self.terms.len() >= MIXTURE_PAR_MIN_TERMS {
            self.terms.par_iter_mut().for_each(update);
        } else {
            for t in &mut self.terms {
                update(t);
            }
        }
    }

    fn apply_two_mode_symplectic(&mut self, a: usize, b: usize, s: [[f64; 4]; 4]) {
        let m = self.n_modes;
        let dim = 2 * m;
        let idx = [a, b, m + a, m + b];

        for cov in &mut self.covs {
            let mut old = [[0.0; 4]; 4];
            for r in 0..4 {
                for c in 0..4 {
                    old[r][c] = cov[idx[r] * dim + idx[c]];
                }
            }
            let (r0, r1, r2, r3) = (idx[0] * dim, idx[1] * dim, idx[2] * dim, idx[3] * dim);
            for j in 0..dim {
                let (v0, v1, v2, v3) = (cov[r0 + j], cov[r1 + j], cov[r2 + j], cov[r3 + j]);
                cov[r0 + j] = s[0][0] * v0 + s[0][1] * v1 + s[0][2] * v2 + s[0][3] * v3;
                cov[r1 + j] = s[1][0] * v0 + s[1][1] * v1 + s[1][2] * v2 + s[1][3] * v3;
                cov[r2 + j] = s[2][0] * v0 + s[2][1] * v1 + s[2][2] * v2 + s[2][3] * v3;
                cov[r3 + j] = s[3][0] * v0 + s[3][1] * v1 + s[3][2] * v2 + s[3][3] * v3;
            }
            for j in 0..dim {
                let row = j * dim;
                cov[row + idx[0]] = cov[r0 + j];
                cov[row + idx[1]] = cov[r1 + j];
                cov[row + idx[2]] = cov[r2 + j];
                cov[row + idx[3]] = cov[r3 + j];
            }
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
                    cov[idx[r] * dim + idx[c]] = val;
                    cov[idx[c] * dim + idx[r]] = val;
                }
            }
        }

        let update = |t: &mut Term| {
            let oa = [t.a[idx[0]], t.a[idx[1]], t.a[idx[2]], t.a[idx[3]]];
            let ob = [t.b[idx[0]], t.b[idx[1]], t.b[idx[2]], t.b[idx[3]]];
            for r in 0..4 {
                t.a[idx[r]] = s[r][0] * oa[0] + s[r][1] * oa[1] + s[r][2] * oa[2] + s[r][3] * oa[3];
                t.b[idx[r]] = s[r][0] * ob[0] + s[r][1] * ob[1] + s[r][2] * ob[2] + s[r][3] * ob[3];
            }
        };
        if self.terms.len() >= MIXTURE_PAR_MIN_TERMS {
            self.terms.par_iter_mut().for_each(update);
        } else {
            for t in &mut self.terms {
                update(t);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Homodyne and heterodyne measurements (conditioning + pruning + merging)
    // -----------------------------------------------------------------------

    /// Destructive homodyne measurement of `x̂ cos φ + p̂ sin φ` on `mode`.
    /// Samples from the exact marginal PDF, conditions all remaining terms
    /// (pruning extinguished terms and merging collapsed peaks), and resets
    /// `mode` to vacuum.
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
        let ix = mode;

        // Determine support [q_lo, q_hi] and finest peak width σ_min across all terms.
        let mut q_lo = f64::INFINITY;
        let mut q_hi = f64::NEG_INFINITY;
        let mut min_sigma = f64::INFINITY;
        let mut max_sigma = 0.0f64;
        for cov in &self.covs {
            let s = cov[ix * dim + ix].max(1e-12).sqrt();
            min_sigma = min_sigma.min(s);
            max_sigma = max_sigma.max(s);
        }
        for t in &self.terms {
            let am = t.a[ix];
            q_lo = q_lo.min(am);
            q_hi = q_hi.max(am);
        }
        q_lo -= 6.0 * max_sigma;
        q_hi += 6.0 * max_sigma;

        // Grid resolution resolves both narrow GKP peaks (min_sigma) and
        // interference fringes (π / max |b_m / σ²|).
        let max_freq: f64 = self
            .terms
            .iter()
            .map(|t| t.b[ix].abs() / self.covs[t.cov_id][ix * dim + ix].max(1e-12))
            .fold(0.0, f64::max);
        let dq_target = (0.15 * min_sigma).min(if max_freq > 1e-6 {
            0.3 / max_freq
        } else {
            0.15 * min_sigma
        });
        let n_grid = (((q_hi - q_lo) / dq_target).ceil() as usize).clamp(256, 8192);
        let dq = (q_hi - q_lo) / (n_grid as f64);
        let qs: Vec<f64> = (0..n_grid).map(|i| q_lo + (i as f64 + 0.5) * dq).collect();
        let pdfs = self.homodyne_x_pdf_batch(mode, &qs)?;

        let mut cdf = Vec::with_capacity(n_grid);
        let mut cum = 0.0;
        for &p in &pdfs {
            cum += p.max(0.0) * dq;
            cdf.push(cum);
        }
        let u = rng.random::<f64>() * cum;
        let idx = cdf.partition_point(|&c| c < u).min(n_grid - 1);
        let outcome = q_lo + (idx as f64 + rng.random::<f64>()) * dq;

        self.condition_x_and_reset(mode, outcome)?;
        Ok(outcome)
    }

    /// Postselect homodyne measurement of `x̂ cos φ + p̂ sin φ` on `mode` onto
    /// `outcome`, returning the marginal probability density `p(outcome)`.
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
        let dens = self.homodyne_x_pdf_batch(mode, &[outcome])?[0];
        self.condition_x_and_reset(mode, outcome)?;
        Ok(dens)
    }

    fn condition_x_and_reset(&mut self, mode: usize, outcome: f64) -> Result<(), Error> {
        let dim = 2 * self.n_modes;
        let ix = mode;
        let ip = self.n_modes + mode;

        // Precompute old Cholesky factors so we include the exact marginalization
        // damping `exp(½(b_new^T V_new^{-1} b_new + b_m²/σ² - b_old^T V_old^{-1} b_old)) <= 1`
        // from integrating out the conjugate quadrature `p_mode`.
        let old_chols: Vec<Vec<f64>> = self
            .covs
            .iter()
            .map(|v| cholesky_real(v, dim).expect("cov is PD"))
            .collect();

        // Precompute per-covariance conditioning vectors and update `covs` in place.
        let mut cov_updates = Vec::with_capacity(self.covs.len());
        for cov in &mut self.covs {
            let sigma2 = cov[ix * dim + ix].max(1e-30);
            let inv_s2 = 1.0 / sigma2;
            let norm = 1.0 / (TAU * sigma2).sqrt();
            let mut v = vec![0.0; dim];
            v.copy_from_slice(&cov[ix * dim..(ix + 1) * dim]);
            v[ix] = 0.0;
            v[ip] = 0.0;

            for i in 0..dim {
                let vi = v[i];
                if vi != 0.0 {
                    let scale = vi * inv_s2;
                    let row = i * dim;
                    for j in 0..dim {
                        cov[row + j] -= scale * v[j];
                    }
                }
            }
            // Reset measured mode to vacuum in covariance.
            for j in 0..dim {
                cov[ix * dim + j] = 0.0;
                cov[ip * dim + j] = 0.0;
                cov[j * dim + ix] = 0.0;
                cov[j * dim + ip] = 0.0;
            }
            cov[ix * dim + ix] = 1.0;
            cov[ip * dim + ip] = 1.0;
            cov_updates.push((inv_s2, norm, v));
        }

        let new_chols: Vec<Vec<f64>> = self
            .covs
            .iter()
            .map(|v| cholesky_real(v, dim).expect("cov is PD"))
            .collect();

        // Exact overflow-free update of each term's `(c̃, a, b)`!
        let update_term = |t: &mut Term| {
            let (inv_s2, norm, v) = &cov_updates[t.cov_id];
            let has_b = t.b.iter().any(|&bi| bi != 0.0);
            let old_q = if has_b {
                forward_solve_norm_sqr(&old_chols[t.cov_id], dim, &t.b)
            } else {
                0.0
            };
            let am = t.a[ix];
            let bm = t.b[ix];
            let d = outcome - am;
            let ph = bm * d * inv_s2;

            let shift_a = d * inv_s2;
            let shift_b = -bm * inv_s2;
            for i in 0..dim {
                let vi = v[i];
                if vi != 0.0 {
                    t.a[i] += shift_a * vi;
                    t.b[i] += shift_b * vi;
                }
            }
            t.a[ix] = 0.0;
            t.a[ip] = 0.0;
            t.b[ix] = 0.0;
            t.b[ip] = 0.0;

            let new_q = if has_b && t.b.iter().any(|&bi| bi != 0.0) {
                forward_solve_norm_sqr(&new_chols[t.cov_id], dim, &t.b)
            } else {
                0.0
            };
            let log_mag = -0.5 * d * d * inv_s2 + 0.5 * (new_q + bm * bm * inv_s2 - old_q);
            t.w_tilde *= C64::from_polar(norm * log_mag.exp(), ph);
        };
        if self.terms.len() >= MIXTURE_PAR_MIN_TERMS {
            self.terms.par_iter_mut().for_each(update_term);
        } else {
            for t in &mut self.terms {
                update_term(t);
            }
        }

        // Renormalise so Σ c_k = 1, then prune and merge collapsed peaks!
        self.renormalize_and_compress()?;
        Ok(())
    }

    /// Destructive heterodyne measurement on `mode`, returning `(Re α, Im α)`.
    pub fn measure_heterodyne<R: Rng + ?Sized>(
        &mut self,
        mode: usize,
        rng: &mut R,
    ) -> Result<(f64, f64), Error> {
        self.check_mode(mode)?;
        // Sample (Re α, Im α) by applying a 50:50 beam splitter with a vacuum
        // ancilla? Even simpler: evaluate the 2D Husimi Q-function on a bounding
        // grid (or sample from the positive envelope mixture + rejection, or
        // direct 2D CDF grid), then condition via `postselect_heterodyne`!
        let dim = 2 * self.n_modes;
        let ix = mode;
        let ip = self.n_modes + mode;
        let mut x_lo = f64::INFINITY;
        let mut x_hi = f64::NEG_INFINITY;
        let mut p_lo = f64::INFINITY;
        let mut p_hi = f64::NEG_INFINITY;
        let mut max_s = 1.0f64;
        for cov in &self.covs {
            max_s = max_s
                .max((cov[ix * dim + ix] + 1.0).sqrt())
                .max((cov[ip * dim + ip] + 1.0).sqrt());
        }
        for t in &self.terms {
            x_lo = x_lo.min(t.a[ix]);
            x_hi = x_hi.max(t.a[ix]);
            p_lo = p_lo.min(t.a[ip]);
            p_hi = p_hi.max(t.a[ip]);
        }
        x_lo = 0.5 * (x_lo - 5.0 * max_s);
        x_hi = 0.5 * (x_hi + 5.0 * max_s);
        p_lo = 0.5 * (p_lo - 5.0 * max_s);
        p_hi = 0.5 * (p_hi + 5.0 * max_s);

        let nx = 64usize;
        let np = 64usize;
        let dx = (x_hi - x_lo) / (nx as f64);
        let dp = (p_hi - p_lo) / (np as f64);

        // Precompute marginal 2D heterodyne parameters per term.
        let rest_idx: Vec<usize> = (0..dim).filter(|&i| i != ix && i != ip).collect();
        let rest_dim = rest_idx.len();
        let het_terms = self.precompute_het_marginal(mode, &rest_idx, rest_dim)?;

        let mut cdf = Vec::with_capacity(nx * np);
        let mut cum = 0.0;
        for i in 0..nx {
            let re = x_lo + (i as f64 + 0.5) * dx;
            let rx = 2.0 * re;
            for j in 0..np {
                let im = p_lo + (j as f64 + 0.5) * dp;
                let rp = 2.0 * im;
                let mut sum = 0.0;
                for &(ew, ax, ap, bx, bp, w00, w01, w11, norm) in &het_terms {
                    let ux = rx - ax;
                    let up = rp - ap;
                    let quad = w00 * ux * ux + 2.0 * w01 * ux * up + w11 * up * up;
                    let ph = bx * (w00 * ux + w01 * up) + bp * (w01 * ux + w11 * up);
                    let g = norm * (-0.5 * quad).exp();
                    sum += (ew * C64::from_polar(g, ph)).re;
                }
                cum += sum.max(0.0);
                cdf.push(cum);
            }
        }
        let u = rng.random::<f64>() * cum;
        let cell = cdf.partition_point(|&c| c < u).min(nx * np - 1);
        let re = x_lo + ((cell / np) as f64 + rng.random::<f64>()) * dx;
        let im = p_lo + ((cell % np) as f64 + rng.random::<f64>()) * dp;

        self.postselect_heterodyne(mode, re, im)?;
        Ok((re, im))
    }

    /// Postselect heterodyne measurement on `mode` onto `(re, im)`, returning
    /// the probability density `p(Re α, Im α)`.
    pub fn postselect_heterodyne(&mut self, mode: usize, re: f64, im: f64) -> Result<f64, Error> {
        self.check_mode(mode)?;
        let dim = 2 * self.n_modes;
        let ix = mode;
        let ip = self.n_modes + mode;
        let rx = 2.0 * re;
        let rp = 2.0 * im;

        let rest_idx: Vec<usize> = (0..dim).filter(|&i| i != ix && i != ip).collect();
        let rest_dim = rest_idx.len();
        let het_terms = self.precompute_het_marginal(mode, &rest_idx, rest_dim)?;
        let mut dens = 0.0;
        for &(ew, ax, ap, bx, bp, w00, w01, w11, norm) in &het_terms {
            let ux = rx - ax;
            let up = rp - ap;
            let quad = w00 * ux * ux + 2.0 * w01 * ux * up + w11 * up * up;
            let ph = bx * (w00 * ux + w01 * up) + bp * (w01 * ux + w11 * up);
            dens += (ew * C64::from_polar(norm * (-0.5 * quad).exp(), ph)).re;
        }

        // Exact conditioning with old/new Cholesky damping for `w_tilde`.
        let old_chols: Vec<Vec<f64>> = self
            .covs
            .iter()
            .map(|v| cholesky_real(v, dim).expect("cov is PD"))
            .collect();

        struct HetCovUpd {
            w00: f64,
            w01: f64,
            w11: f64,
            norm: f64,
            cx: Vec<f64>,
            cp: Vec<f64>,
        }
        let mut upds = Vec::with_capacity(self.covs.len());
        for cov in &mut self.covs {
            let a = cov[ix * dim + ix] + 1.0;
            let b = cov[ix * dim + ip];
            let c = cov[ip * dim + ip] + 1.0;
            let det = (a * c - b * b).max(1e-30);
            let inv_det = 1.0 / det;
            let (w00, w01, w11) = (c * inv_det, -b * inv_det, a * inv_det);
            let norm = 4.0 / (TAU * det.sqrt());

            let mut cx = vec![0.0; dim];
            let mut cp = vec![0.0; dim];
            cx.copy_from_slice(&cov[ix * dim..(ix + 1) * dim]);
            cp.copy_from_slice(&cov[ip * dim..(ip + 1) * dim]);
            cx[ix] = 0.0;
            cx[ip] = 0.0;
            cp[ix] = 0.0;
            cp[ip] = 0.0;

            for i in 0..dim {
                let (cxi, cpi) = (cx[i], cp[i]);
                if cxi == 0.0 && cpi == 0.0 {
                    continue;
                }
                let t0 = cxi * w00 + cpi * w01;
                let t1 = cxi * w01 + cpi * w11;
                let row = i * dim;
                for j in 0..dim {
                    cov[row + j] -= t0 * cx[j] + t1 * cp[j];
                }
            }
            for j in 0..dim {
                cov[ix * dim + j] = 0.0;
                cov[ip * dim + j] = 0.0;
                cov[j * dim + ix] = 0.0;
                cov[j * dim + ip] = 0.0;
            }
            cov[ix * dim + ix] = 1.0;
            cov[ip * dim + ip] = 1.0;
            upds.push(HetCovUpd {
                w00,
                w01,
                w11,
                norm,
                cx,
                cp,
            });
        }

        let new_chols: Vec<Vec<f64>> = self
            .covs
            .iter()
            .map(|v| cholesky_real(v, dim).expect("cov is PD"))
            .collect();

        for t in &mut self.terms {
            let u = &upds[t.cov_id];
            let old_q = if t.b.iter().any(|&bi| bi != 0.0) {
                let y = forward_solve_real(&old_chols[t.cov_id], dim, &t.b);
                y.iter().map(|v| v * v).sum::<f64>()
            } else {
                0.0
            };
            let dx = rx - t.a[ix];
            let dp = rp - t.a[ip];
            let bx = t.b[ix];
            let bp = t.b[ip];

            let quad_a = u.w00 * dx * dx + 2.0 * u.w01 * dx * dp + u.w11 * dp * dp;
            let quad_b = u.w00 * bx * bx + 2.0 * u.w01 * bx * bp + u.w11 * bp * bp;
            let ph = bx * (u.w00 * dx + u.w01 * dp) + bp * (u.w01 * dx + u.w11 * dp);

            let ua0 = u.w00 * dx + u.w01 * dp;
            let ua1 = u.w01 * dx + u.w11 * dp;
            let ub0 = -(u.w00 * bx + u.w01 * bp);
            let ub1 = -(u.w01 * bx + u.w11 * bp);
            for i in 0..dim {
                t.a[i] += u.cx[i] * ua0 + u.cp[i] * ua1;
                t.b[i] += u.cx[i] * ub0 + u.cp[i] * ub1;
            }
            t.a[ix] = 0.0;
            t.a[ip] = 0.0;
            t.b[ix] = 0.0;
            t.b[ip] = 0.0;

            let new_q = if t.b.iter().any(|&bi| bi != 0.0) {
                let y = forward_solve_real(&new_chols[t.cov_id], dim, &t.b);
                y.iter().map(|v| v * v).sum::<f64>()
            } else {
                0.0
            };
            // Exact conversion from c to c̃: exp(½(new_q + quad_b - old_q)) <= 1!
            let log_mag = -0.5 * quad_a + 0.5 * (new_q + quad_b - old_q);
            t.w_tilde *= C64::from_polar(u.norm * log_mag.exp(), ph);
        }

        self.renormalize_and_compress()?;
        Ok(dens)
    }

    #[allow(clippy::type_complexity)]
    fn precompute_het_marginal(
        &self,
        mode: usize,
        rest_idx: &[usize],
        rest_dim: usize,
    ) -> Result<Vec<(C64, f64, f64, f64, f64, f64, f64, f64, f64)>, Error> {
        let dim = 2 * self.n_modes;
        let ix = mode;
        let ip = self.n_modes + mode;

        let mut cov_info = Vec::with_capacity(self.covs.len());
        for cov in &self.covs {
            let old_chol = cholesky_real(cov, dim).ok_or_else(|| {
                Error::InvalidArgument("mixture covariance is not positive definite".into())
            })?;
            let a = cov[ix * dim + ix] + 1.0;
            let b = cov[ix * dim + ip];
            let c = cov[ip * dim + ip] + 1.0;
            let det = (a * c - b * b).max(1e-30);
            let inv_det = 1.0 / det;
            let (w00, w01, w11) = (c * inv_det, -b * inv_det, a * inv_det);
            let norm = 4.0 / (TAU * det.sqrt());
            let _ = (rest_idx, rest_dim);
            cov_info.push((old_chol, w00, w01, w11, norm));
        }

        let mut out = Vec::with_capacity(self.terms.len());
        for t in &self.terms {
            let (old_chol, w00, w01, w11, norm) = &cov_info[t.cov_id];
            let bx = t.b[ix];
            let bp = t.b[ip];
            let quad_b = w00 * bx * bx + 2.0 * w01 * bx * bp + w11 * bp * bp;
            let old_q = if t.b.iter().any(|&bi| bi != 0.0) {
                let y = forward_solve_real(old_chol, dim, &t.b);
                y.iter().map(|v| v * v).sum::<f64>()
            } else {
                0.0
            };
            let eff_w = t.w_tilde * (0.5 * (quad_b - old_q)).exp();
            if eff_w.norm() >= 1e-18 {
                out.push((eff_w, t.a[ix], t.a[ip], bx, bp, *w00, *w01, *w11, *norm));
            }
        }
        Ok(out)
    }

    fn renormalize_and_compress(&mut self) -> Result<(), Error> {
        let ws = self.true_weights();
        let norm: f64 = ws.iter().map(|z| z.re).sum();
        if norm <= 1e-300 || !norm.is_finite() {
            return Err(Error::InvalidArgument(
                "postselected measurement outcome has zero probability".into(),
            ));
        }
        let inv = 1.0 / norm;
        for t in &mut self.terms {
            t.w_tilde *= inv;
        }
        self.prune_and_merge();
        Ok(())
    }

    /// Merge terms with identical `(cov_id, a, b)` (within `merge_tol`) and
    /// drop terms whose weight falls below `prune_tol * max_weight`, recording
    /// pruned weight in `error_bound`.
    fn prune_and_merge(&mut self) {
        if self.terms.is_empty() {
            return;
        }
        let merge_tol = self.options.merge_tol;
        let inv_q = if merge_tol > 0.0 {
            1.0 / merge_tol
        } else {
            1e12
        };

        // Quantised key for O(M) hash-merging of collapsed peaks.
        let mut buckets: HashMap<(usize, Vec<i64>, Vec<i64>), usize> =
            HashMap::with_capacity(self.terms.len());
        let mut merged: Vec<Term> = Vec::with_capacity(self.terms.len());

        for t in self.terms.drain(..) {
            let ka: Vec<i64> = t.a.iter().map(|&x| (x * inv_q).round() as i64).collect();
            let kb: Vec<i64> = t.b.iter().map(|&x| (x * inv_q).round() as i64).collect();
            let key = (t.cov_id, ka, kb);
            if let Some(&idx) = buckets.get(&key) {
                merged[idx].w_tilde += t.w_tilde;
            } else {
                buckets.insert(key, merged.len());
                merged.push(t);
            }
        }

        let max_w = merged
            .iter()
            .map(|t| t.w_tilde.norm())
            .fold(0.0f64, f64::max);
        let cutoff = self.options.prune_tol * max_w;
        let mut kept = Vec::with_capacity(merged.len());
        let mut pruned = 0.0;
        for t in merged {
            let w = t.w_tilde.norm();
            if w >= cutoff && w > 0.0 {
                kept.push(t);
            } else {
                pruned += w;
            }
        }
        if kept.is_empty() {
            // Never empty the mixture completely.
            return;
        }
        self.error_bound += pruned;
        self.terms = kept;
    }
}

// ---------------------------------------------------------------------------
// Single-mode ModeState -> LCG decomposition
// ---------------------------------------------------------------------------

type OneModeTerm = (C64, [f64; 2], [f64; 2], usize);
type OneModeLcg = (Vec<[f64; 4]>, Vec<OneModeTerm>, f64);

fn mode_state_to_lcg(state: &ModeState, options: &MixtureOptions) -> Result<OneModeLcg, Error> {
    let eye = [1.0, 0.0, 0.0, 1.0];
    match *state {
        ModeState::Vacuum | ModeState::Fock(0) => Ok((
            vec![eye],
            vec![(C64::new(1.0, 0.0), [0.0, 0.0], [0.0, 0.0], 0)],
            0.0,
        )),
        ModeState::Coherent { r, phi } => {
            let (s, c) = phi.sin_cos();
            Ok((
                vec![eye],
                vec![(
                    C64::new(1.0, 0.0),
                    [2.0 * r * c, 2.0 * r * s],
                    [0.0, 0.0],
                    0,
                )],
                0.0,
            ))
        }
        ModeState::Squeezed { r, phi } => {
            let ch = (2.0 * r).cosh();
            let sh = (2.0 * r).sinh();
            let (s, c) = phi.sin_cos();
            let cov = [ch - c * sh, -s * sh, -s * sh, ch + c * sh];
            Ok((
                vec![cov],
                vec![(C64::new(1.0, 0.0), [0.0, 0.0], [0.0, 0.0], 0)],
                0.0,
            ))
        }
        ModeState::Thermal { mean_photons } => {
            let v = 2.0 * mean_photons + 1.0;
            Ok((
                vec![[v, 0.0, 0.0, v]],
                vec![(C64::new(1.0, 0.0), [0.0, 0.0], [0.0, 0.0], 0)],
                0.0,
            ))
        }
        ModeState::Cat { r, phi, parity } => {
            let alpha = C64::from_polar(r, phi);
            let x0 = 2.0 * alpha.re;
            let p0 = 2.0 * alpha.im;
            let rel = C64::from_polar(1.0, PI * parity);
            let norm = 2.0 * (1.0 + (PI * parity).cos() * (-2.0 * r * r).exp());
            let inv_n = 1.0 / norm;
            // |α⟩⟨α| + |−α⟩⟨−α| + e^{-iπp} |α⟩⟨−α| + e^{iπp} |−α⟩⟨α|.
            // In c̃ representation, the cross terms have a = (0, 0),
            // b = (p0, -x0) and (-p0, x0), and c̃ = rel* / norm, rel / norm!
            let terms = vec![
                (C64::new(inv_n, 0.0), [x0, p0], [0.0, 0.0], 0),
                (C64::new(inv_n, 0.0), [-x0, -p0], [0.0, 0.0], 0),
                (rel.conj() * inv_n, [0.0, 0.0], [p0, -x0], 0),
                (rel * inv_n, [0.0, 0.0], [-p0, x0], 0),
            ];
            Ok((vec![eye], terms, 0.0))
        }
        ModeState::Gkp {
            epsilon,
            theta,
            phi,
        } => {
            // Exact finite-energy square GKP state as linear combination of
            // shared-covariance Gaussians with V = diag(tanh ε, coth ε).
            let ket_tol = (options.prune_tol * 0.1).min(1e-7);
            let (_r_sq, peaks) = gkp_peaks(epsilon, theta, phi, ket_tol);
            let v_x = epsilon.tanh();
            let v_p = 1.0 / v_x;
            let cov = [v_x, 0.0, 0.0, v_p];

            // Compute exact normalisation ⟨ψ|ψ⟩ = Σ_{j,k} d_j d_k* exp(-⅛ v_p (x_j - x_k)²).
            let mut norm_sqr = 0.0;
            for &(xj, wj) in &peaks {
                for &(xk, wk) in &peaks {
                    let dx = xj - xk;
                    norm_sqr += (wj * wk.conj()).re * (-0.125 * v_p * dx * dx).exp();
                }
            }
            let inv_norm = 1.0 / norm_sqr;

            // Build pairwise cross-Wigner terms:
            // c̃_{j,k} = (d_j d_k*) / norm_sqr,
            // a_{j,k} = ((x_j + x_k)/2, 0),
            // b_{j,k} = (0, -½ v_p (x_j - x_k)).
            // Merge pairs with identical (a, b) immediately!
            let mut terms = Vec::with_capacity(peaks.len() * peaks.len());
            let mut pruned_err = 0.0;
            for &(xj, wj) in &peaks {
                for &(xk, wk) in &peaks {
                    let wt = wj * wk.conj() * inv_norm;
                    if wt.norm() < options.prune_tol * 0.1 {
                        pruned_err += wt.norm();
                        continue;
                    }
                    let ax = 0.5 * (xj + xk);
                    let bp = -0.5 * v_p * (xj - xk);
                    terms.push((wt, [ax, 0.0], [0.0, bp], 0));
                }
            }
            Ok((vec![cov], terms, pruned_err))
        }
        ModeState::Fock(n) => {
            // Well-conditioned shared-covariance coherent-state ring expansion:
            // |ψ_n⟩ ∝ (1/K) Σ_{k=0}^{K-1} e^{-i 2π n k / K} |R e^{i 2π k / K}⟩.
            // Every cross-Wigner term |α_j⟩⟨α_k| shares V = I_2, avoiding both
            // finite-difference cancellation and multi-covariance blow-up.
            let tol = options
                .prune_tol
                .min(options.fock_step * options.fock_step)
                .clamp(1e-12, 1e-2);
            let (k_pts, radius, norm2, err_bound) = fock_ring_params(n, tol).ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "Fock({n}) exceeds GaussianMixture limit (max n = {MAX_LCG_FOCK_N}); use FockState"
                ))
            })?;
            let inv_k = 1.0 / (k_pts as f64);
            let inv_norm2 = 1.0 / norm2;
            let mut xs = Vec::with_capacity(k_pts);
            let mut ps = Vec::with_capacity(k_pts);
            let mut cs = Vec::with_capacity(k_pts);
            for k in 0..k_pts {
                let ang = TAU * (k as f64) * inv_k;
                let (s, c) = ang.sin_cos();
                xs.push(2.0 * radius * c);
                ps.push(2.0 * radius * s);
                cs.push(C64::from_polar(inv_k, -TAU * ((n * k) as f64) * inv_k));
            }
            let mut terms = Vec::with_capacity(k_pts * k_pts);
            for j in 0..k_pts {
                let (x1, p1, cj) = (xs[j], ps[j], cs[j]);
                for k in 0..k_pts {
                    let (x2, p2, ck) = (xs[k], ps[k], cs[k]);
                    let ax = 0.5 * (x1 + x2);
                    let ap = 0.5 * (p1 + p2);
                    let bx = 0.5 * (p1 - p2);
                    let bp = -0.5 * (x1 - x2);
                    let phase = C64::from_polar(1.0, 0.25 * (p1 * x2 - p2 * x1));
                    let wt = (cj * ck.conj() * inv_norm2) * phase;
                    terms.push((wt, [ax, ap], [bx, bp], 0));
                }
            }
            Ok((vec![eye], terms, err_bound))
        }
        ModeState::FockVector(_) => Err(Error::InvalidArgument(
            "arbitrary FockVector inputs are not yet decomposed in GaussianMixture; use FockState"
                .into(),
        )),
    }
}

fn ln_factorial(k: usize) -> f64 {
    (1..=k).map(|i| (i as f64).ln()).sum()
}

/// Choose `(k_pts, radius, norm2, err_bound)` for the coherent-state ring
/// representation of `ModeState::Fock(n)`.
pub(crate) fn fock_ring_params(n: usize, tol: f64) -> Option<(usize, f64, f64, f64)> {
    if n == 0 || n > MAX_LCG_FOCK_N {
        return None;
    }
    let target = tol.clamp(1e-12, 1e-2);
    let log_alias = -(target * 0.1).min(1e-7).ln();
    let ln_n_fact = ln_factorial(n);
    let mut k_pts = (n + 3).max(4);
    let max_k = n + 16;
    let mut radius;
    loop {
        let log_ratio = 0.5 * (ln_factorial(n + k_pts) - ln_n_fact);
        radius = ((log_ratio - log_alias) / (k_pts as f64))
            .exp()
            .min((n as f64).sqrt());
        let log_pn = -radius * radius + 2.0 * (n as f64) * radius.max(1e-300).ln() - ln_n_fact;
        if log_pn > -10.0 || k_pts >= max_k {
            break;
        }
        k_pts += 1;
    }
    let r2 = radius * radius;
    let ln_r2 = r2.max(1e-300).ln();
    let mut p_n = 0.0;
    let mut norm2 = 0.0;
    for m in 0..8usize {
        let l = n + m * k_pts;
        let log_p = -r2 + (l as f64) * ln_r2 - ln_factorial(l);
        if log_p < -740.0 {
            break;
        }
        let p = log_p.exp();
        if m == 0 {
            p_n = p;
        }
        norm2 += p;
    }
    let trace_err = (1.0 - p_n / norm2).max(0.0).sqrt();
    let cond_err = 1e-15 / norm2;
    Some((k_pts, radius, norm2, trace_err + cond_err))
}

/// Term count used by [`GaussianMixture`] for `ModeState::Fock(n)`.
pub(crate) fn fock_lcg_term_count(n: usize, tol: f64) -> Option<usize> {
    if n == 0 {
        return Some(1);
    }
    let (k_pts, _, _, _) = fock_ring_params(n, tol)?;
    Some(k_pts * k_pts)
}

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

/// Compute `‖L^{-1} b‖₂² = b^T (L L^T)^{-1} b` without heap allocation for `dim <= 16`.
fn forward_solve_norm_sqr(l: &[f64], dim: usize, b: &[f64]) -> f64 {
    if dim <= 16 {
        let mut y = [0.0f64; 16];
        let mut sum = 0.0;
        for i in 0..dim {
            let mut s = b[i];
            let row = i * dim;
            for j in 0..i {
                s -= l[row + j] * y[j];
            }
            let yi = s / l[row + i];
            y[i] = yi;
            sum += yi * yi;
        }
        sum
    } else {
        let y = forward_solve_real(l, dim, b);
        y.iter().map(|v| v * v).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cv::FockState;
    use crate::cv::fock_ops::hermite_fock_wavefunctions;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() < tol, "{a} vs {b} (diff {})", (a - b).abs());
    }

    #[test]
    fn cat_state_and_gaussian_gates_match_fock_state() {
        let mut c = ModeCircuit::new(2);
        c.set_input(
            0,
            ModeState::Cat {
                r: 1.0,
                phi: 0.3,
                parity: 0.0,
            },
        )
        .unwrap()
        .set_input(
            1,
            ModeState::Cat {
                r: 0.8,
                phi: -0.4,
                parity: 1.0,
            },
        )
        .unwrap()
        .squeeze(0.25, 0.5, 0)
        .beam_splitter(0.6, 0.2, 0, 1)
        .cx(0.2, 0, 1)
        .displace(0.3, -0.4, 1);

        let mix = GaussianMixture::from_circuit(&c, MixtureOptions::default()).unwrap();
        let fock = FockState::from_circuit(&c, 22).unwrap();
        assert!(fock.truncation_error() < 1e-6);

        for m in 0..2 {
            close(mix.mean_x(m).unwrap(), fock.mean_x(m).unwrap(), 1e-5);
            close(mix.mean_p(m).unwrap(), fock.mean_p(m).unwrap(), 1e-5);
            close(
                mix.mean_photons(m).unwrap(),
                fock.mean_photons(m).unwrap(),
                1e-5,
            );
        }

        // Verify homodyne marginal PDF on mode 0 against exact Fock wavefunction PDF!
        let d = fock.cutoff();
        let amps = fock.amplitudes();
        let mut psi = vec![0.0; d];
        for q in [-2.0, -0.5, 0.0, 0.7, 1.8] {
            hermite_fock_wavefunctions(q, d, &mut psi);
            let mut want = 0.0;
            for n1 in 0..d {
                let mut acc = C64::new(0.0, 0.0);
                for (n0, &w) in psi.iter().enumerate() {
                    acc += amps[n0 + d * n1] * w;
                }
                want += acc.norm_sqr();
            }
            let got = mix.homodyne_pdf(0, 0.0, q).unwrap();
            close(got, want, 1e-5);
        }
    }

    #[test]
    fn gkp_state_and_clifford_gates_match_fock_state() {
        // Finite-energy square GKP |0_Δ⟩ and |+_Δ⟩ with ε = 0.15 (~8.2 dB)
        // so both FockState (cutoff=36) and GaussianMixture are well converged!
        let eps = 0.15;
        let mut c = ModeCircuit::new(1);
        c.set_input(
            0,
            ModeState::Gkp {
                epsilon: eps,
                theta: 0.0,
                phi: 0.0,
            },
        )
        .unwrap();

        let mix0 = GaussianMixture::from_circuit(&c, MixtureOptions::default()).unwrap();
        let fock0 = FockState::from_circuit(&c, 64).unwrap();
        assert!(
            fock0.truncation_error() < 1e-5,
            "trunc={}",
            fock0.truncation_error()
        );
        close(
            mix0.mean_photons(0).unwrap(),
            fock0.mean_photons(0).unwrap(),
            1e-4,
        );

        // Verify homodyne x-PDF peaks at even multiples of √(2π) and vanishes at odd multiples!
        let d_grid = (2.0 * PI).sqrt();
        let p_even = mix0.homodyne_pdf(0, 0.0, 0.0).unwrap();
        let p_odd = mix0.homodyne_pdf(0, 0.0, d_grid).unwrap();
        assert!(
            p_even > 0.5 && p_odd < 1e-3,
            "p_even={p_even}, p_odd={p_odd}"
        );

        // Compare homodyne PDF across multiple x points against FockState!
        let d = fock0.cutoff();
        let amps = fock0.amplitudes();
        let mut psi = vec![0.0; d];
        for q in [0.0, 0.5, d_grid, 2.0 * d_grid, -1.2] {
            hermite_fock_wavefunctions(q, d, &mut psi);
            let amp: C64 = amps.iter().zip(&psi).map(|(&a, &w)| a * w).sum();
            let want = amp.norm_sqr();
            let got = mix0.homodyne_pdf(0, 0.0, q).unwrap();
            close(got, want, 1e-4);
        }

        // Apply logical X_L = D(√(π/2), 0): shifts |0_GKP⟩ -> |1_GKP⟩ (peaks at odd multiples!).
        let mut mix1 = mix0.clone();
        mix1.displace((0.5 * PI).sqrt(), 0.0, 0).unwrap();
        assert!(mix1.homodyne_pdf(0, 0.0, d_grid).unwrap() > 0.4);
        assert!(mix1.homodyne_pdf(0, 0.0, 0.0).unwrap() < 1e-3);

        // Apply logical Hadamard H_L = R(π/2) to |0_GKP⟩ -> |+_GKP⟩ (peaks at BOTH 0 and √(2π)!).
        let mut mix_plus = mix0.clone();
        mix_plus.rotation(std::f64::consts::FRAC_PI_2, 0).unwrap();
        let pp0 = mix_plus.homodyne_pdf(0, 0.0, 0.0).unwrap();
        let pp1 = mix_plus.homodyne_pdf(0, 0.0, d_grid).unwrap();
        assert!(pp0 > 0.25 && pp1 > 0.15, "pp0={pp0}, pp1={pp1}");
    }

    #[test]
    fn gkp_steane_error_correction_reduces_logical_error_rate() {
        // Steane-style GKP x-error correction:
        // Data mode 0 starts in |0_GKP⟩ (ε = 0.06, ~12.2 dB), gets additive noise
        // of variance σ² = 0.20 (σ ≈ 0.45, compared to half-grid √(2π)/2 ≈ 1.253).
        // Ancilla mode 1 starts in |+_GKP⟩, apply CX(-1, 0, 1) -> x_1 -> x_1 - x_0,
        // wait: CX(1, 0, 1) gives x_1 -> x_1 + x_0!
        // Since x_0 = 2s√(2π) + e_x and x_1 = k√(2π), measuring x_1 yields
        // m = (2s + k)√(2π) + e_x => m mod √(2π) estimates e_x!
        // Correct data by shifting x_0 by -wrap(m, √(2π))!
        let eps = 0.06;
        let noise_var = 0.20;
        let d_grid = (2.0 * PI).sqrt();
        let wrap = |x: f64, period: f64| -> f64 {
            let mut r = x.rem_euclid(period);
            if r > 0.5 * period {
                r -= period;
            }
            r
        };
        let is_logical_zero = |x: f64| -> bool { wrap(x, 2.0 * d_grid).abs() < 0.5 * d_grid };

        let mut rng = StdRng::seed_from_u64(2026);
        let trials = 60;
        let mut raw_errors = 0usize;
        let mut qec_errors = 0usize;

        for _ in 0..trials {
            // Uncorrected: data + noise + second noise round -> measure x
            let mut c_raw = ModeCircuit::new(1);
            c_raw
                .set_input(
                    0,
                    ModeState::Gkp {
                        epsilon: eps,
                        theta: 0.0,
                        phi: 0.0,
                    },
                )
                .unwrap();
            let mut st_raw =
                GaussianMixture::from_circuit(&c_raw, MixtureOptions::default()).unwrap();
            st_raw
                .apply_channel(
                    &CvChannel::AdditiveNoise {
                        variance: 2.0 * noise_var,
                    },
                    0,
                )
                .unwrap();
            let q_raw = st_raw.measure_homodyne(0, 0.0, &mut rng).unwrap();
            if !is_logical_zero(q_raw) {
                raw_errors += 1;
            }

            // Corrected: data + noise -> Steane x-syndrome with |+_GKP⟩ ancilla -> noise -> measure x
            let mut c_qec = ModeCircuit::new(2);
            c_qec
                .set_input(
                    0,
                    ModeState::Gkp {
                        epsilon: eps,
                        theta: 0.0,
                        phi: 0.0,
                    },
                )
                .unwrap()
                .set_input(
                    1,
                    ModeState::Gkp {
                        epsilon: eps,
                        theta: std::f64::consts::FRAC_PI_2,
                        phi: 0.0,
                    },
                )
                .unwrap();
            let mut st = GaussianMixture::from_circuit(&c_qec, MixtureOptions::default()).unwrap();
            st.apply_channel(
                &CvChannel::AdditiveNoise {
                    variance: noise_var,
                },
                0,
            )
            .unwrap();
            st.cx(1.0, 0, 1).unwrap();
            let m_x = st.measure_homodyne(1, 0.0, &mut rng).unwrap();
            let corr = -wrap(m_x, d_grid);
            st.displace(0.5 * corr, 0.0, 0).unwrap();
            // Apply second round of noise and read out logical Z (homodyne x)
            st.apply_channel(
                &CvChannel::AdditiveNoise {
                    variance: noise_var,
                },
                0,
            )
            .unwrap();
            let q_qec = st.measure_homodyne(0, 0.0, &mut rng).unwrap();
            if !is_logical_zero(q_qec) {
                qec_errors += 1;
            }
        }
        assert!(
            qec_errors < raw_errors,
            "QEC should reduce logical errors: qec={qec_errors} vs raw={raw_errors} out of {trials}"
        );
    }

    #[test]
    fn fock_input_in_gaussian_mixture_matches_exact_wigner() {
        // Check Fock |n⟩ for n = 0..=10: exact 2π W(0) = (-1)^n, exact ⟨n̂⟩ = n,
        // single shared covariance, and honest error bound.
        for n in [0usize, 1, 2, 4, 5, 6, 8, 10] {
            let mut c = ModeCircuit::new(1);
            c.set_input(0, ModeState::Fock(n)).unwrap();
            let mix = GaussianMixture::from_circuit(&c, MixtureOptions::default()).unwrap();
            let expected_w0 = if n % 2 == 0 { 1.0 } else { -1.0 } / TAU;
            close(mix.wigner(&[0.0], &[0.0]).unwrap(), expected_w0, 1e-6);
            close(mix.mean_photons(0).unwrap(), n as f64, 1e-6);
            assert_eq!(mix.covs.len(), 1);
            assert!(mix.error_bound() < 1e-5);
        }

        // High-n Fock input beyond coherent-ring limit returns a clean error (no panic).
        let mut big = ModeCircuit::new(1);
        big.set_input(0, ModeState::Fock(100)).unwrap();
        assert!(GaussianMixture::from_circuit(&big, MixtureOptions::default()).is_err());

        // Multi-mode Fock inputs share a single covariance matrix (no Cartesian cov blow-up).
        let mut multi = ModeCircuit::new(3);
        for m in 0..3 {
            multi.set_input(m, ModeState::Fock(1)).unwrap();
        }
        let mix_m = GaussianMixture::from_circuit(&multi, MixtureOptions::default()).unwrap();
        assert_eq!(mix_m.covs.len(), 1);
        for m in 0..3 {
            close(mix_m.mean_photons(m).unwrap(), 1.0, 1e-6);
        }
    }
}
