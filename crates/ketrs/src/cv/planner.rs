//! Automatic backend planner for continuous-variable ([`ModeCircuit`]) circuits.
//!
//! Inspects a circuit's input states, gates, measurements, mode count, and
//! liveness window width to pick the fastest exact or controlled-accuracy
//! engine, while always allowing callers to override the choice explicitly.

use super::circuit::{Measurement, ModeCircuit, ModeOp};
use super::state::ModeState;
use super::window::WindowSampler;
use crate::error::Error;

/// Maximum dense cutoff-Fock Hilbert dimension (`d^m`) before the planner
/// switches from [`CvBackend::Fock`] to [`CvBackend::BosonicMps`].
const MAX_DENSE_FOCK_DIM: usize = 1 << 18; // 262,144 amplitudes (~4 MiB)

/// Maximum fixed-N Hilbert dimension `C(m + N - 1, N)` before the planner
/// prefers [`CvBackend::BosonSampler`] over [`CvBackend::FixedN`] for
/// sampling workloads.
const MAX_FIXED_N_SAMPLING_DIM: usize = 200_000;

/// Maximum sliding-window width `W` for which [`CvBackend::WindowSampler`]
/// is automatically selected for streaming circuits with mid-circuit
/// measurements.
const MAX_AUTO_WINDOW_WIDTH: usize = 4;

/// Maximum initial Gaussian-mixture term count for automatic selection of
/// [`CvBackend::GaussianMixture`].
const MAX_AUTO_MIXTURE_TERMS: usize = 65_536;

/// Continuous-variable simulation backend selectable by the planner.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CvBackend {
    /// Exact covariance-matrix Gaussian simulator (`GaussianState`).
    Gaussian,
    /// Linear-combination-of-Gaussians simulator (`GaussianMixture`, Bourassa et al.).
    GaussianMixture,
    /// Exact photon-number-conserving SLOS / Fock simulator (`FixedNFockState`).
    FixedN,
    /// Exact Clifford & Clifford boson sampler (`boson_sample`).
    BosonSampler,
    /// Exact Quesada–Arrazola GBS chain-rule sampler (`sample_gbs_pnr` / `sample_gbs_threshold`).
    GbsSampler,
    /// Sliding-window Fock sampler (`WindowSampler`) for local / time-bin circuits.
    WindowSampler,
    /// Universal cutoff Fock statevector (`FockState`).
    Fock,
    /// Bosonic Matrix Product State (`BosonicMps`) for large 1D / weakly entangled circuits.
    BosonicMps,
}

impl CvBackend {
    /// Short lowercase identifier matching the Python API.
    pub fn as_str(self) -> &'static str {
        match self {
            CvBackend::Gaussian => "gaussian",
            CvBackend::GaussianMixture => "gaussian_mixture",
            CvBackend::FixedN => "fixed_n",
            CvBackend::BosonSampler => "boson_sampler",
            CvBackend::GbsSampler => "gbs_sampler",
            CvBackend::WindowSampler => "window_sampler",
            CvBackend::Fock => "fock",
            CvBackend::BosonicMps => "bosonic_mps",
        }
    }

    /// Parse a backend name (case-insensitive).
    pub fn parse(name: &str) -> Result<Self, Error> {
        match name.to_ascii_lowercase().as_str() {
            "gaussian" => Ok(CvBackend::Gaussian),
            "gaussian_mixture" | "mixture" | "lcg" => Ok(CvBackend::GaussianMixture),
            "fixed_n" | "slos" => Ok(CvBackend::FixedN),
            "boson_sampler" | "boson" => Ok(CvBackend::BosonSampler),
            "gbs_sampler" | "gbs" => Ok(CvBackend::GbsSampler),
            "window_sampler" | "window" => Ok(CvBackend::WindowSampler),
            "fock" => Ok(CvBackend::Fock),
            "bosonic_mps" | "mps" => Ok(CvBackend::BosonicMps),
            _ => Err(Error::InvalidArgument(format!(
                "unknown CV backend {name:?}; expected one of gaussian, gaussian_mixture, \
                 fixed_n, boson_sampler, gbs_sampler, window_sampler, fock, bosonic_mps"
            ))),
        }
    }
}

/// Options controlling automatic CV backend planning.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlanOptions {
    /// Fock cutoff `d` for Fock / MPS / Window backends (`None` picks a default
    /// from the circuit's inputs).
    pub cutoff: Option<usize>,
    /// Maximum bond dimension for [`CvBackend::BosonicMps`] (default `64`).
    pub max_bond: usize,
    /// Target accuracy tolerance for approximate backends (default `1e-6`).
    pub tolerance: f64,
    /// Whether the caller intends to draw samples rather than inspect full state
    /// amplitudes or moments (enables `BosonSampler`, `GbsSampler`, `WindowSampler`).
    pub sampling: bool,
    /// Optional explicit override: when `Some`, always choose this backend.
    pub prefer_backend: Option<CvBackend>,
}

impl Default for PlanOptions {
    fn default() -> Self {
        PlanOptions {
            cutoff: None,
            max_bond: 64,
            tolerance: 1e-6,
            sampling: false,
            prefer_backend: None,
        }
    }
}

/// Output of [`plan_circuit`]: the chosen backend, rationale, and recommended
/// truncation parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct CvPlan {
    /// Selected simulation backend.
    pub backend: CvBackend,
    /// Human-readable explanation of why this backend was chosen.
    pub reason: String,
    /// Recommended Fock cutoff (if the chosen backend uses one).
    pub cutoff: Option<usize>,
    /// Recommended MPS bond dimension (if `BosonicMps` was chosen).
    pub max_bond: Option<usize>,
    /// Active liveness window width (if `WindowSampler` was chosen).
    pub window_width: Option<usize>,
}

/// Choose the fastest appropriate CV engine for `circuit`.
///
/// The automatic choice follows `PLAN.md` and can always be overridden via
/// [`PlanOptions::prefer_backend`]:
/// 1. Pure Gaussian circuits → [`CvBackend::Gaussian`] (or [`CvBackend::GbsSampler`]
///    when sampling terminal PNR/threshold measurements).
/// 2. Gaussian gates + non-Gaussian inputs (cat, GKP, Fock) without PNR/threshold
///    measurements → [`CvBackend::GaussianMixture`].
/// 3. Passive linear optics with Fock inputs → [`CvBackend::BosonSampler`] for
///    large sampling problems, else [`CvBackend::FixedN`].
/// 4. Number-conserving circuits (passive + Kerr/CrossKerr) with Fock inputs →
///    [`CvBackend::FixedN`].
/// 5. Streaming / local circuits with small liveness window width →
///    [`CvBackend::WindowSampler`].
/// 6. Universal non-Gaussian circuits → [`CvBackend::Fock`] when `d^m` fits in
///    memory, else [`CvBackend::BosonicMps`].
pub fn plan_circuit(circuit: &ModeCircuit, options: &PlanOptions) -> CvPlan {
    let m = circuit.n_modes();
    let cutoff = options.cutoff.unwrap_or_else(|| estimate_cutoff(circuit));

    if let Some(backend) = options.prefer_backend {
        return CvPlan {
            backend,
            reason: format!("explicitly overridden to {}", backend.as_str()),
            cutoff: matches!(
                backend,
                CvBackend::Fock | CvBackend::BosonicMps | CvBackend::WindowSampler
            )
            .then_some(cutoff),
            max_bond: (backend == CvBackend::BosonicMps).then_some(options.max_bond),
            window_width: (backend == CvBackend::WindowSampler)
                .then(|| WindowSampler::max_window_width(circuit)),
        };
    }

    let ops = circuit.ops();
    let inputs = circuit.inputs();
    let all_gaussian_inputs = inputs.iter().all(ModeState::is_gaussian);
    let all_fock_inputs = inputs.iter().all(ModeState::is_fock);
    let all_lcg_inputs = inputs
        .iter()
        .all(|s| !matches!(s, ModeState::FockVector(_)));

    let all_gaussian_gates = ops.iter().all(|op| match op {
        ModeOp::Gate { gate, .. } => gate.is_gaussian(),
        ModeOp::TwoModeGate { gate, .. } => gate.is_gaussian(),
        ModeOp::Channel { .. } => op.is_gaussian(),
        ModeOp::Passive { .. } | ModeOp::Measure { .. } => true,
    });

    let has_pnr_or_thresh = ops.iter().any(|op| {
        matches!(
            op,
            ModeOp::Measure {
                measurement: Measurement::PhotonCount | Measurement::Threshold,
                ..
            }
        )
    });
    let has_hom_or_het = ops.iter().any(|op| {
        matches!(
            op,
            ModeOp::Measure {
                measurement: Measurement::Homodyne { .. } | Measurement::Heterodyne,
                ..
            }
        )
    });
    let has_measurements = has_pnr_or_thresh || has_hom_or_het;
    let terminal_measurements_only = has_measurements && {
        let first_m = ops
            .iter()
            .position(|op| matches!(op, ModeOp::Measure { .. }))
            .unwrap_or(ops.len());
        ops[first_m..]
            .iter()
            .all(|op| matches!(op, ModeOp::Measure { .. }))
    };

    // 1. Pure Gaussian circuits (homodyne/heterodyne or unitary) -> GaussianState;
    //    Gaussian + terminal PNR/threshold sampling -> GbsSampler.
    if all_gaussian_inputs && all_gaussian_gates {
        if has_pnr_or_thresh && !has_hom_or_het && terminal_measurements_only && options.sampling {
            return CvPlan {
                backend: CvBackend::GbsSampler,
                reason: "Gaussian circuit with terminal PNR/threshold sampling".into(),
                cutoff: None,
                max_bond: None,
                window_width: None,
            };
        }
        if !has_pnr_or_thresh {
            return CvPlan {
                backend: CvBackend::Gaussian,
                reason: "all inputs, gates, and measurements are Gaussian".into(),
                cutoff: None,
                max_bond: None,
                window_width: None,
            };
        }
    }

    // 2. Photon-number-conserving circuits with Fock inputs -> FixedN or BosonSampler.
    if all_fock_inputs && circuit.conserves_photons() && !has_hom_or_het {
        let total_n: usize = inputs
            .iter()
            .map(|s| match *s {
                ModeState::Fock(n) => n,
                _ => 0,
            })
            .sum();
        let dim = super::fixed_n::FixedNFockState::basis_dim(m, total_n);
        if options.sampling
            && circuit.is_passive()
            && terminal_measurements_only
            && dim > MAX_FIXED_N_SAMPLING_DIM
        {
            return CvPlan {
                backend: CvBackend::BosonSampler,
                reason: format!(
                    "passive Fock sampling with large subspace C({m}+{total_n}-1, {total_n}) = {dim}"
                ),
                cutoff: None,
                max_bond: None,
                window_width: None,
            };
        }
        if dim <= 2_000_000 {
            return CvPlan {
                backend: CvBackend::FixedN,
                reason: format!("photon-number-conserving Fock circuit (N={total_n}, dim={dim})"),
                cutoff: None,
                max_bond: None,
                window_width: None,
            };
        }
        if options.sampling && circuit.is_passive() && terminal_measurements_only {
            return CvPlan {
                backend: CvBackend::BosonSampler,
                reason: format!("passive Fock sampling (m={m}, N={total_n})"),
                cutoff: None,
                max_bond: None,
                window_width: None,
            };
        }
    }

    // 3. Gaussian gates + non-Gaussian inputs (cat / GKP / Fock) without PNR/threshold.
    if all_gaussian_gates && !all_gaussian_inputs && all_lcg_inputs && !has_pnr_or_thresh {
        let est_terms = estimate_lcg_terms(inputs);
        if est_terms <= MAX_AUTO_MIXTURE_TERMS {
            return CvPlan {
                backend: CvBackend::GaussianMixture,
                reason: format!("Gaussian circuit on non-Gaussian inputs (~{est_terms} LCG terms)"),
                cutoff: None,
                max_bond: None,
                window_width: None,
            };
        }
    }

    // 4. Local / streaming circuits with mid-circuit measurements -> WindowSampler.
    if options.sampling && has_measurements && m > 1 {
        let width = WindowSampler::max_window_width(circuit);
        if width < m && width <= MAX_AUTO_WINDOW_WIDTH {
            return CvPlan {
                backend: CvBackend::WindowSampler,
                reason: format!(
                    "streaming circuit on {m} modes has narrow active window width {width}"
                ),
                cutoff: Some(cutoff),
                max_bond: None,
                window_width: Some(width),
            };
        }
    }

    // 5. Universal non-Gaussian: dense FockState if d^m fits, else BosonicMps.
    let fock_dim = (cutoff as u128).checked_pow(m as u32);
    if fock_dim.is_some_and(|dim| dim <= MAX_DENSE_FOCK_DIM as u128) {
        CvPlan {
            backend: CvBackend::Fock,
            reason: format!(
                "universal non-Gaussian circuit fits in dense Fock state ({cutoff}^{m} amplitudes)"
            ),
            cutoff: Some(cutoff),
            max_bond: None,
            window_width: None,
        }
    } else {
        CvPlan {
            backend: CvBackend::BosonicMps,
            reason: format!(
                "non-Gaussian circuit ({cutoff}^{m} > {MAX_DENSE_FOCK_DIM}) exceeds dense Fock budget; using BosonicMps"
            ),
            cutoff: Some(cutoff),
            max_bond: Some(options.max_bond),
            window_width: None,
        }
    }
}

fn estimate_cutoff(circuit: &ModeCircuit) -> usize {
    let mut d = 12usize;
    for st in circuit.inputs() {
        let need = match st {
            ModeState::Vacuum => 8,
            ModeState::Fock(n) => (*n + 6).max(8),
            ModeState::Coherent { r, .. } | ModeState::Cat { r, .. } => {
                ((r * r + 5.0 * r.abs() + 8.0).ceil() as usize).max(12)
            }
            ModeState::Squeezed { r, .. } => {
                let nbar = r.sinh().powi(2);
                ((6.0 * nbar + 12.0).ceil() as usize).max(12)
            }
            ModeState::Thermal { mean_photons } => {
                ((6.0 * mean_photons + 12.0).ceil() as usize).max(12)
            }
            ModeState::Gkp { epsilon, .. } => {
                ((5.0 / epsilon.max(0.05)).ceil() as usize).clamp(24, 128)
            }
            ModeState::FockVector(v) => v.len().max(8),
        };
        d = d.max(need);
    }
    d
}

fn estimate_lcg_terms(inputs: &[ModeState]) -> usize {
    let mut prod = 1usize;
    for st in inputs {
        let k = match *st {
            ModeState::Vacuum
            | ModeState::Coherent { .. }
            | ModeState::Squeezed { .. }
            | ModeState::Thermal { .. }
            | ModeState::Fock(0) => 1,
            ModeState::Cat { .. } => 4,
            ModeState::Fock(n) => n + 1,
            ModeState::Gkp { epsilon, .. } => {
                let peaks = (((14.0 / (0.5 * std::f64::consts::PI * epsilon.tanh()))
                    .sqrt()
                    .ceil()) as usize)
                    .max(3);
                // Conservative upper bound on pruned pairwise terms per mode.
                (peaks * peaks).max(9)
            }
            ModeState::FockVector(_) => return usize::MAX,
        };
        prod = prod.saturating_mul(k);
    }
    prod
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_PI_4;

    #[test]
    fn planner_routes_each_circuit_class() {
        // 1. Gaussian circuit -> Gaussian
        let mut g = ModeCircuit::new(6);
        g.squeeze(0.5, 0.0, 0).beam_splitter(FRAC_PI_4, 0.0, 0, 1);
        assert_eq!(
            plan_circuit(&g, &PlanOptions::default()).backend,
            CvBackend::Gaussian
        );

        // 2. Gaussian + terminal PNR sampling -> GbsSampler
        let mut gbs = g.clone();
        for m in 0..6 {
            gbs.measure_photons(m);
        }
        assert_eq!(
            plan_circuit(
                &gbs,
                &PlanOptions {
                    sampling: true,
                    ..Default::default()
                }
            )
            .backend,
            CvBackend::GbsSampler
        );

        // 3. Gaussian gates + cat/GKP inputs -> GaussianMixture
        let mut cat_c = ModeCircuit::new(2);
        cat_c
            .set_input(
                0,
                ModeState::Cat {
                    r: 2.0,
                    phi: 0.0,
                    parity: 0.0,
                },
            )
            .unwrap()
            .beam_splitter(FRAC_PI_4, 0.0, 0, 1);
        assert_eq!(
            plan_circuit(&cat_c, &PlanOptions::default()).backend,
            CvBackend::GaussianMixture
        );

        // 4. Passive Fock -> FixedN (small) and BosonSampler (large sampling)
        let mut hom = ModeCircuit::new(4);
        hom.set_input(0, ModeState::Fock(1)).unwrap();
        hom.set_input(1, ModeState::Fock(1)).unwrap();
        hom.beam_splitter(FRAC_PI_4, 0.0, 0, 1);
        assert_eq!(
            plan_circuit(&hom, &PlanOptions::default()).backend,
            CvBackend::FixedN
        );

        let mut big_bs = ModeCircuit::new(20);
        for i in 0..8 {
            big_bs.set_input(i, ModeState::Fock(1)).unwrap();
        }
        for i in 0..19 {
            big_bs.beam_splitter(FRAC_PI_4, 0.0, i, i + 1);
        }
        for i in 0..20 {
            big_bs.measure_photons(i);
        }
        assert_eq!(
            plan_circuit(
                &big_bs,
                &PlanOptions {
                    sampling: true,
                    ..Default::default()
                }
            )
            .backend,
            CvBackend::BosonSampler
        );

        // 5. Streaming time-bin loop with non-Gaussian Kerr -> WindowSampler
        let mut loop_c = ModeCircuit::new(20);
        for i in 0..20 {
            loop_c
                .set_input(i, ModeState::Coherent { r: 0.5, phi: 0.0 })
                .unwrap();
        }
        for i in 0..19 {
            loop_c.kerr(0.2, i).beam_splitter(FRAC_PI_4, 0.0, i, i + 1);
            loop_c.measure_photons(i);
        }
        loop_c.measure_photons(19);
        assert_eq!(
            plan_circuit(
                &loop_c,
                &PlanOptions {
                    sampling: true,
                    ..Default::default()
                }
            )
            .backend,
            CvBackend::WindowSampler
        );

        // 6. Universal non-Gaussian: small -> Fock, large -> BosonicMps
        let mut small_ng = ModeCircuit::new(3);
        small_ng
            .squeeze(0.3, 0.0, 0)
            .kerr(0.4, 0)
            .beam_splitter(0.5, 0.0, 0, 1);
        assert_eq!(
            plan_circuit(
                &small_ng,
                &PlanOptions {
                    cutoff: Some(12),
                    ..Default::default()
                }
            )
            .backend,
            CvBackend::Fock
        );

        let mut large_ng = ModeCircuit::new(12);
        for i in 0..12 {
            large_ng.squeeze(0.3, 0.0, i).kerr(0.4, i);
        }
        for i in 0..11 {
            large_ng.beam_splitter(0.5, 0.0, i, i + 1);
        }
        assert_eq!(
            plan_circuit(
                &large_ng,
                &PlanOptions {
                    cutoff: Some(10),
                    ..Default::default()
                }
            )
            .backend,
            CvBackend::BosonicMps
        );

        // 7. Explicit override always wins
        assert_eq!(
            plan_circuit(
                &g,
                &PlanOptions {
                    prefer_backend: Some(CvBackend::BosonicMps),
                    ..Default::default()
                }
            )
            .backend,
            CvBackend::BosonicMps
        );
    }
}
