//! Opt-in continuous-variable noise channels (loss, thermal loss, dephasing,
//! and Gaussian additive noise) and Kraus operator builders for Fock-space
//! simulation.

use crate::error::Error;
use num_complex::Complex64 as C64;

/// A single-mode continuous-variable noise channel (`ħ = 2`).
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum CvChannel {
    /// Pure photon loss with energy transmissivity `eta ∈ [0, 1]`:
    /// `V → η V + (1 − η) I₂`, `μ → √η μ`.
    Loss {
        /// Energy transmissivity `η ∈ [0, 1]`.
        eta: f64,
    },
    /// Thermal-loss channel (beam splitter of transmissivity `eta ∈ [0, 1]`
    /// with a thermal environment of mean photon number `mean_photons ≥ 0`):
    /// `V → η V + (1 − η)(2 n̄ + 1) I₂`, `μ → √η μ`.
    ThermalLoss {
        /// Energy transmissivity `η ∈ [0, 1]`.
        eta: f64,
        /// Mean photon number `n̄ ≥ 0` of the thermal bath.
        mean_photons: f64,
    },
    /// Phase damping (dephasing): random rotation `R(φ)` with
    /// `φ ~ N(0, sigma²)`, damping `ρ_{m,n} → exp(−½ σ² (m − n)²) ρ_{m,n}`.
    Dephasing {
        /// Standard deviation `σ ≥ 0` of the phase angle in radians.
        sigma: f64,
    },
    /// Isotropic Gaussian additive noise: random displacement adding
    /// `variance ≥ 0` to both `V_{xx}` and `V_{pp}` (`ħ = 2`).
    AdditiveNoise {
        /// Added quadrature variance `v ≥ 0` (`V → V + v I₂`).
        variance: f64,
    },
}

impl CvChannel {
    /// Validate channel parameters.
    pub fn validate(&self) -> Result<(), Error> {
        match *self {
            CvChannel::Loss { eta } => {
                if eta.is_finite() && (0.0..=1.0).contains(&eta) {
                    Ok(())
                } else {
                    Err(Error::InvalidArgument(format!(
                        "loss transmissivity must lie in [0, 1], got {eta}"
                    )))
                }
            }
            CvChannel::ThermalLoss { eta, mean_photons } => {
                if !(eta.is_finite() && (0.0..=1.0).contains(&eta)) {
                    return Err(Error::InvalidArgument(format!(
                        "thermal loss transmissivity must lie in [0, 1], got {eta}"
                    )));
                }
                if !(mean_photons.is_finite() && mean_photons >= 0.0) {
                    return Err(Error::InvalidArgument(format!(
                        "thermal bath mean photons must be finite and ≥ 0, got {mean_photons}"
                    )));
                }
                Ok(())
            }
            CvChannel::Dephasing { sigma } => {
                if sigma.is_finite() && sigma >= 0.0 {
                    Ok(())
                } else {
                    Err(Error::InvalidArgument(format!(
                        "dephasing sigma must be finite and ≥ 0, got {sigma}"
                    )))
                }
            }
            CvChannel::AdditiveNoise { variance } => {
                if variance.is_finite() && variance >= 0.0 {
                    Ok(())
                } else {
                    Err(Error::InvalidArgument(format!(
                        "additive noise variance must be finite and ≥ 0, got {variance}"
                    )))
                }
            }
        }
    }
}

/// Sparse representation of a pure-loss Kraus operator `K_l`:
/// `K_l |n + l⟩ = coeff[n] |n⟩` for `n = 0..cutoff - l`.
pub(crate) fn loss_kraus_diags(cutoff: usize, eta: f64) -> Vec<Vec<f64>> {
    let mut log_fact = vec![0.0f64; cutoff + 1];
    for i in 1..=cutoff {
        log_fact[i] = log_fact[i - 1] + (i as f64).ln();
    }
    let log_eta = if eta > 0.0 {
        eta.ln()
    } else {
        f64::NEG_INFINITY
    };
    let log_loss = if eta < 1.0 {
        (1.0 - eta).ln()
    } else {
        f64::NEG_INFINITY
    };

    let mut ops = Vec::with_capacity(cutoff);
    for l in 0..cutoff {
        let len = cutoff - l;
        let mut diag = vec![0.0f64; len];
        for (n, slot) in diag.iter_mut().enumerate() {
            let val = if (l == 0 && eta == 1.0) || (n == 0 && eta == 0.0) {
                1.0
            } else if eta == 0.0 || eta == 1.0 {
                0.0
            } else {
                let log_binom = log_fact[n + l] - log_fact[n] - log_fact[l];
                (0.5 * (log_binom + (n as f64) * log_eta + (l as f64) * log_loss)).exp()
            };
            *slot = val;
        }
        ops.push(diag);
    }
    ops
}

/// Sparse representation of a quantum-limited amplifier Kraus operator `A_l`
/// with gain `g ≥ 1`: `A_l |n⟩ = coeff[n] |n + l⟩` for `n = 0..cutoff - l`.
pub(crate) fn amp_kraus_diags(cutoff: usize, g: f64) -> Vec<Vec<f64>> {
    let mut log_fact = vec![0.0f64; cutoff + 1];
    for i in 1..=cutoff {
        log_fact[i] = log_fact[i - 1] + (i as f64).ln();
    }
    if (g - 1.0).abs() < 1e-15 {
        let mut ops = vec![Vec::new(); cutoff];
        ops[0] = vec![1.0; cutoff];
        for (l, op) in ops.iter_mut().enumerate().skip(1) {
            *op = vec![0.0; cutoff - l];
        }
        return ops;
    }
    let log_g = g.ln();
    let log_1m = (1.0 - 1.0 / g).ln();
    let mut ops = Vec::with_capacity(cutoff);
    for l in 0..cutoff {
        let len = cutoff - l;
        let mut diag = vec![0.0f64; len];
        for (n, slot) in diag.iter_mut().enumerate() {
            let log_binom = log_fact[n + l] - log_fact[n] - log_fact[l];
            *slot = (0.5 * (log_binom - ((n + 1) as f64) * log_g + (l as f64) * log_1m)).exp();
        }
        ops.push(diag);
    }
    ops
}

/// Apply a pure-loss Kraus operator `K_l` (shift-down by `l` on `mode`) to a
/// Fock statevector slice of length `cutoff^m`.
pub(crate) fn apply_loss_kraus_vec(
    src: &[C64],
    dst: &mut [C64],
    cutoff: usize,
    stride: usize,
    l: usize,
    diag: &[f64],
) {
    let block = stride * cutoff;
    let n_outer = src.len() / block;
    dst.fill(C64::new(0.0, 0.0));
    for outer in 0..n_outer {
        let base = outer * block;
        for (n, &w) in diag.iter().enumerate() {
            if w == 0.0 {
                continue;
            }
            let src_off = base + (n + l) * stride;
            let dst_off = base + n * stride;
            for inner in 0..stride {
                dst[dst_off + inner] = src[src_off + inner] * w;
            }
        }
    }
}

/// Apply an amplifier Kraus operator `A_l` (shift-up by `l` on `mode`) to a
/// Fock statevector slice of length `cutoff^m`.
pub(crate) fn apply_amp_kraus_vec(
    src: &[C64],
    dst: &mut [C64],
    cutoff: usize,
    stride: usize,
    l: usize,
    diag: &[f64],
) {
    let block = stride * cutoff;
    let n_outer = src.len() / block;
    dst.fill(C64::new(0.0, 0.0));
    for outer in 0..n_outer {
        let base = outer * block;
        for (n, &w) in diag.iter().enumerate() {
            if w == 0.0 {
                continue;
            }
            let src_off = base + n * stride;
            let dst_off = base + (n + l) * stride;
            for inner in 0..stride {
                dst[dst_off + inner] = src[src_off + inner] * w;
            }
        }
    }
}
