//! Single-mode input states.

use crate::error::Error;
use num_complex::Complex64 as C64;
use std::f64::consts::PI;

/// The state a mode starts in.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum ModeState {
    /// The vacuum `|0⟩`.
    Vacuum,
    /// The Fock state `|n⟩`.
    Fock(usize),
    /// The coherent state `|α⟩ = D(α)|0⟩` with `α = r e^{iφ}`.
    Coherent {
        /// Magnitude r.
        r: f64,
        /// Phase φ.
        phi: f64,
    },
    /// Squeezed vacuum `S(z)|0⟩` with `z = r e^{iφ}`.
    Squeezed {
        /// Squeezing magnitude r.
        r: f64,
        /// Squeezing angle φ.
        phi: f64,
    },
    /// The thermal state with mean photon number `mean_photons`.
    Thermal {
        /// Mean photon number n̄ ≥ 0.
        mean_photons: f64,
    },
    /// The cat state `∝ |α⟩ + e^{iπp}|−α⟩` with `α = r e^{iφ}`: `p = 0`
    /// is the even cat, `p = 1` the odd cat.
    Cat {
        /// Magnitude r.
        r: f64,
        /// Phase φ.
        phi: f64,
        /// Relative phase p, in units of π.
        parity: f64,
    },
    /// A normalised superposition of Fock states, `Σ cₙ |n⟩`.
    FockVector(Vec<C64>),
    /// Finite-energy square GKP qubit (`ħ = 2`, grid spacing `√(2π)`):
    /// `∝ exp(−ε n̂) (cos(θ/2)|0_GKP⟩ + e^{iφ} sin(θ/2)|1_GKP⟩)` where
    /// `ε = Δ² > 0` sets the squeezing (`r_dB = −10 log₁₀ ε`).
    Gkp {
        /// Envelope energy parameter `ε = Δ² > 0`.
        epsilon: f64,
        /// Bloch polar angle `θ` (`0` for `|0_GKP⟩`, `π` for `|1_GKP⟩`, `π/2` for `|+_GKP⟩`).
        theta: f64,
        /// Bloch azimuthal angle `φ`.
        phi: f64,
    },
}

impl ModeState {
    /// Check the state's numbers.
    pub(crate) fn validate(&self) -> Result<(), Error> {
        let bad = |msg: String| Err(Error::InvalidArgument(msg));
        match self {
            ModeState::Vacuum | ModeState::Fock(_) => Ok(()),
            ModeState::Coherent { r, phi } | ModeState::Squeezed { r, phi } => {
                if r.is_finite() && phi.is_finite() {
                    Ok(())
                } else {
                    bad(format!(
                        "state parameters must be finite: r = {r}, φ = {phi}"
                    ))
                }
            }
            ModeState::Thermal { mean_photons } => {
                if mean_photons.is_finite() && *mean_photons >= 0.0 {
                    Ok(())
                } else {
                    bad(format!(
                        "mean photon number must be finite and ≥ 0, got {mean_photons}"
                    ))
                }
            }
            ModeState::Cat { r, phi, parity } => {
                if !(r.is_finite() && phi.is_finite() && parity.is_finite()) {
                    return bad("cat state parameters must be finite".into());
                }
                if cat_norm_sqr(*r, *parity) < 1e-24 {
                    return bad("this cat state is the zero vector".into());
                }
                Ok(())
            }
            ModeState::FockVector(c) => {
                let norm: f64 = c.iter().map(|a| a.norm_sqr()).sum();
                if (norm - 1.0).abs() > 1e-8 {
                    return bad(format!("Fock vector has norm² {norm}, not 1"));
                }
                Ok(())
            }
            ModeState::Gkp {
                epsilon,
                theta,
                phi,
            } => {
                if !(epsilon.is_finite() && *epsilon > 0.0) {
                    return bad(format!("GKP epsilon must be finite and > 0, got {epsilon}"));
                }
                if !(theta.is_finite() && phi.is_finite()) {
                    return bad("GKP Bloch angles must be finite".into());
                }
                Ok(())
            }
        }
    }

    /// Whether the state is Gaussian.
    pub fn is_gaussian(&self) -> bool {
        match self {
            ModeState::Vacuum
            | ModeState::Coherent { .. }
            | ModeState::Squeezed { .. }
            | ModeState::Thermal { .. } => true,
            ModeState::Fock(n) => *n == 0,
            ModeState::Cat { .. } | ModeState::FockVector(_) | ModeState::Gkp { .. } => false,
        }
    }

    /// Whether the state is pure.
    pub fn is_pure(&self) -> bool {
        match self {
            ModeState::Thermal { mean_photons } => *mean_photons == 0.0,
            _ => true,
        }
    }

    /// Whether the state has a definite photon number.
    pub fn is_fock(&self) -> bool {
        match self {
            ModeState::Vacuum | ModeState::Fock(_) => true,
            ModeState::Thermal { mean_photons } => *mean_photons == 0.0,
            _ => false,
        }
    }

    /// The first `cutoff` Fock amplitudes of a pure state, or `None` for a
    /// mixed one. The vector is truncated, not renormalised.
    // Used by the test reference now and by the Fock engines (M2).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn fock_amplitudes(&self, cutoff: usize) -> Option<Vec<C64>> {
        let mut c = vec![C64::new(0.0, 0.0); cutoff];
        if cutoff == 0 {
            return Some(c);
        }
        match self {
            ModeState::Vacuum => c[0] = C64::new(1.0, 0.0),
            ModeState::Thermal { mean_photons } => {
                if *mean_photons != 0.0 {
                    return None;
                }
                c[0] = C64::new(1.0, 0.0);
            }
            ModeState::Fock(n) => {
                if *n < cutoff {
                    c[*n] = C64::new(1.0, 0.0);
                }
            }
            ModeState::Coherent { r, phi } => coherent(C64::from_polar(*r, *phi), &mut c),
            ModeState::Squeezed { r, phi } => {
                let ratio = -C64::from_polar(r.tanh(), *phi);
                c[0] = C64::new(1.0 / r.cosh().sqrt(), 0.0);
                for n in (2..cutoff).step_by(2) {
                    let k = n as f64;
                    c[n] = c[n - 2] * ratio * ((k - 1.0) / k).sqrt();
                }
            }
            ModeState::Cat { r, phi, parity } => {
                coherent(C64::from_polar(*r, *phi), &mut c);
                let rel = C64::from_polar(1.0, PI * parity);
                let norm = 1.0 / cat_norm_sqr(*r, *parity).sqrt();
                for (n, a) in c.iter_mut().enumerate() {
                    let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
                    *a *= (C64::new(1.0, 0.0) + rel * sign) * norm;
                }
            }
            ModeState::FockVector(v) => {
                for (a, b) in c.iter_mut().zip(v) {
                    *a = *b;
                }
            }
            ModeState::Gkp {
                epsilon,
                theta,
                phi,
            } => {
                gkp_fock_amplitudes(*epsilon, *theta, *phi, &mut c);
            }
        }
        Some(c)
    }
}

/// Finite-energy square GKP ket amplitudes (`ħ = 2`, grid spacing `√(2π)`):
/// `|ψ_ε⟩ ∝ Σ_s d_s D(x̄_s / 2) S(r_ε) |0⟩` with `e^{-2 r_ε} = tanh ε`,
/// `x̄_s = s √(2π) / cosh ε`, and `d_s = c_{s mod 2} exp(−½ π tanh(ε) s²)`.
pub(crate) fn gkp_peaks(epsilon: f64, theta: f64, phi: f64, tol: f64) -> (f64, Vec<(f64, C64)>) {
    let th = epsilon.tanh();
    let ch = epsilon.cosh();
    let r_sq = -0.5 * th.ln(); // e^{-2 r_sq} = tanh(ε)
    let spacing = (2.0 * PI).sqrt();
    let c0 = C64::new((0.5 * theta).cos(), 0.0);
    let c1 = C64::from_polar((0.5 * theta).sin(), phi);

    // Envelope decay: exp(-0.5 * π * tanh(ε) * s²). Keep all peaks above `tol`.
    let rate = 0.5 * PI * th;
    let s_max = (((-tol.max(1e-16).ln()) / rate).sqrt().ceil() as isize).max(2);

    let mut peaks = Vec::new();
    for s in -s_max..=s_max {
        let coeff = if s.rem_euclid(2) == 0 { c0 } else { c1 };
        if coeff.norm() < 1e-15 {
            continue;
        }
        let env = (-rate * ((s * s) as f64)).exp();
        if env * coeff.norm() < tol {
            continue;
        }
        let x_bar = (s as f64) * spacing / ch;
        peaks.push((x_bar, coeff * env));
    }
    (r_sq, peaks)
}

#[cfg_attr(not(test), allow(dead_code))]
fn gkp_fock_amplitudes(epsilon: f64, theta: f64, phi: f64, c: &mut [C64]) {
    let d = c.len();
    if d == 0 {
        return;
    }
    let (r_sq, peaks) = gkp_peaks(epsilon, theta, phi, 1e-12);
    let t = r_sq.tanh();
    let inv_sqrt_ch = 1.0 / r_sq.cosh().sqrt();
    let inv_sqrt: Vec<f64> = (0..=d)
        .map(|k| if k == 0 { 0.0 } else { 1.0 / (k as f64).sqrt() })
        .collect();
    let sqrt_t: Vec<f64> = (0..=d).map(|k| (k as f64).sqrt()).collect();

    // Exact rescaled O(d) three-term Hermite recurrence for each displaced
    // squeezed peak D(x_bar / 2) S(r_sq) |0⟩:
    //   a_0 = (cosh r)^{-1/2} exp(-½ α² (1 + tanh r))
    //   a_{n+1} = (β / √(n+1)) a_n - tanh(r) √(n/(n+1)) a_{n-1},  β = α (1 + tanh r).
    // Rescaling prevents underflow at large |α| while preserving full precision.
    let mut peak_amps = vec![0.0f64; d];
    for &(x_bar, weight) in &peaks {
        let alpha = 0.5 * x_bar;
        let beta = alpha * (1.0 + t);
        let mut log_scale = -0.5 * alpha * alpha * (1.0 + t) + inv_sqrt_ch.ln();
        let mut prev = 0.0f64;
        let mut cur = 1.0f64;
        peak_amps[0] = if log_scale > -700.0 {
            cur * log_scale.exp()
        } else {
            0.0
        };
        for n in 0..d - 1 {
            let next = (beta * cur - t * sqrt_t[n] * prev) * inv_sqrt[n + 1];
            prev = cur;
            cur = next;
            let mag = cur.abs().max(prev.abs());
            if mag > 1e100 || (mag > 0.0 && mag < 1e-100) {
                let s = 1.0 / mag;
                prev *= s;
                cur *= s;
                log_scale += mag.ln();
            }
            peak_amps[n + 1] = if log_scale > -700.0 {
                cur * log_scale.exp()
            } else {
                0.0
            };
        }
        for (dst, &a) in c.iter_mut().zip(&peak_amps) {
            *dst += weight * a;
        }
    }
    // Normalise using the exact analytic peak-overlap norm so truncation at `d`
    // reflects true Fock leakage (not artificial renormalisation).
    let v_p = 1.0 / epsilon.tanh(); // e^{2 r_sq}
    let mut norm_sqr = 0.0;
    for &(xj, wj) in &peaks {
        for &(xk, wk) in &peaks {
            let dx = xj - xk;
            let ov = (-0.125 * v_p * dx * dx).exp();
            norm_sqr += (wj.conj() * wk).re * ov;
        }
    }
    let inv_norm = 1.0 / norm_sqr.sqrt();
    for a in c.iter_mut() {
        *a *= inv_norm;
    }
}

/// Coherent-state amplitudes `e^{−|α|²/2} αⁿ/√n!` into `c`.
#[cfg_attr(not(test), allow(dead_code))]
fn coherent(alpha: C64, c: &mut [C64]) {
    c[0] = C64::new((-alpha.norm_sqr() / 2.0).exp(), 0.0);
    for n in 1..c.len() {
        c[n] = c[n - 1] * alpha / (n as f64).sqrt();
    }
}

/// `‖ |α⟩ + e^{iπp}|−α⟩ ‖²`, which depends only on `|α|`.
fn cat_norm_sqr(r: f64, parity: f64) -> f64 {
    2.0 * (1.0 + (PI * parity).cos() * (-2.0 * r * r).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn norm(c: &[C64]) -> f64 {
        c.iter().map(|a| a.norm_sqr()).sum()
    }

    #[test]
    fn pure_states_are_normalised() {
        for state in [
            ModeState::Vacuum,
            ModeState::Fock(3),
            ModeState::Coherent { r: 1.2, phi: 0.4 },
            ModeState::Squeezed { r: 0.6, phi: 1.0 },
            ModeState::Cat {
                r: 1.5,
                phi: 0.3,
                parity: 0.0,
            },
            ModeState::Cat {
                r: 0.2,
                phi: 0.0,
                parity: 1.0,
            },
        ] {
            state.validate().unwrap();
            let c = state.fock_amplitudes(80).unwrap();
            assert!((norm(&c) - 1.0).abs() < 1e-12, "{state:?}");
        }
    }

    #[test]
    fn cats_have_definite_parity() {
        let even = ModeState::Cat {
            r: 1.0,
            phi: 0.0,
            parity: 0.0,
        };
        let odd = ModeState::Cat {
            r: 1.0,
            phi: 0.0,
            parity: 1.0,
        };
        let e = even.fock_amplitudes(30).unwrap();
        let o = odd.fock_amplitudes(30).unwrap();
        for n in 0..30 {
            if n % 2 == 1 {
                assert!(e[n].norm() < 1e-15);
            } else {
                assert!(o[n].norm() < 1e-15);
            }
        }
    }

    #[test]
    fn invalid_states_are_rejected() {
        assert!(
            ModeState::Thermal { mean_photons: -1.0 }
                .validate()
                .is_err()
        );
        assert!(
            ModeState::FockVector(vec![C64::new(1.0, 0.0), C64::new(1.0, 0.0)])
                .validate()
                .is_err()
        );
        let zero_cat = ModeState::Cat {
            r: 0.0,
            phi: 0.0,
            parity: 1.0,
        };
        assert!(zero_cat.validate().is_err());
        assert!(
            ModeState::Thermal { mean_photons: 0.5 }
                .fock_amplitudes(4)
                .is_none()
        );
    }
}
