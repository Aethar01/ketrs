//! Single-qubit unitary algebra and minimal canonical synthesis.

use crate::circuit::Operation;
use crate::gate::{Gate, Matrix2};
use crate::observable::Pauli;
use num_complex::Complex64 as C64;
use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI, TAU};

const ZERO: C64 = C64::new(0.0, 0.0);
const ONE: C64 = C64::new(1.0, 0.0);
const I: C64 = C64::new(0.0, 1.0);

/// Identity 2×2 matrix.
pub(crate) const I2: Matrix2 = [[ONE, ZERO], [ZERO, ONE]];

/// Multiply two 2×2 matrices: `a * b`.
#[inline]
pub(crate) fn mat2_mul(a: &Matrix2, b: &Matrix2) -> Matrix2 {
    [
        [
            a[0][0] * b[0][0] + a[0][1] * b[1][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1],
        ],
        [
            a[1][0] * b[0][0] + a[1][1] * b[1][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1],
        ],
    ]
}

/// Determinant of a 2×2 matrix.
#[inline]
pub(crate) fn mat2_det(m: &Matrix2) -> C64 {
    m[0][0] * m[1][1] - m[0][1] * m[1][0]
}

/// Wrap an angle into `(-π, π]`.
#[inline]
pub(crate) fn wrap_pi(mut angle: f64) -> f64 {
    angle %= TAU;
    if angle <= -PI {
        angle += TAU;
    } else if angle > PI {
        angle -= TAU;
    }
    if angle.abs() < 1e-15 { 0.0 } else { angle }
}

/// Wrap an angle into `(-2π, 2π]` (the period of `Rx`, `Ry`, `Rz`, `PauliRotation`).
#[inline]
pub(crate) fn wrap_2pi(mut angle: f64) -> f64 {
    let period = 2.0 * TAU;
    angle %= period;
    if angle <= -TAU {
        angle += period;
    } else if angle > TAU {
        angle -= period;
    }
    if angle.abs() < 1e-15 { 0.0 } else { angle }
}

/// Whether every entry of `m` is real within `tol`.
#[inline]
pub(crate) fn mat2_is_real(m: &Matrix2, tol: f64) -> bool {
    m.iter().flatten().all(|c| c.im.abs() <= tol)
}

/// Whether `m` is diagonal within `tol`.
#[inline]
pub(crate) fn mat2_is_diagonal(m: &Matrix2, tol: f64) -> bool {
    m[0][1].norm() <= tol && m[1][0].norm() <= tol
}

/// Whether `m = e^{iγ} I` within `tol`; returns `Some(γ)` if so.
pub(crate) fn mat2_identity_phase(m: &Matrix2, tol: f64) -> Option<f64> {
    if m[0][1].norm() <= tol && m[1][0].norm() <= tol && (m[0][0] - m[1][1]).norm() <= tol {
        let phase = wrap_pi(m[0][0].arg());
        Some(if phase.abs() <= tol { 0.0 } else { phase })
    } else {
        None
    }
}

/// Whether `m = e^{iγ} target` within `tol`; returns `Some(γ)` if so.
pub(crate) fn mat2_matches_up_to_phase(m: &Matrix2, target: &Matrix2, tol: f64) -> Option<f64> {
    let inner = (m[0][0] * target[0][0].conj()
        + m[0][1] * target[0][1].conj()
        + m[1][0] * target[1][0].conj()
        + m[1][1] * target[1][1].conj())
        * 0.5;
    if inner.norm() < 1.0 - tol.max(1e-12) {
        return None;
    }
    let phase = wrap_pi(inner.arg());
    let f = C64::cis(phase);
    for r in 0..2 {
        for c in 0..2 {
            if (m[r][c] - f * target[r][c]).norm() > tol {
                return None;
            }
        }
    }
    Some(if phase.abs() <= tol { 0.0 } else { phase })
}

/// Whether two 2×2 matrices commute: `‖ab − ba‖_∞ ≤ tol`.
pub(crate) fn mat2_commutes(a: &Matrix2, b: &Matrix2, tol: f64) -> bool {
    let ab = mat2_mul(a, b);
    let ba = mat2_mul(b, a);
    for r in 0..2 {
        for c in 0..2 {
            if (ab[r][c] - ba[r][c]).norm() > tol {
                return false;
            }
        }
    }
    true
}

/// 2×2 matrix of a single-qubit operation (`Gate` or weight-1 `PauliRotation`).
pub(crate) fn op_matrix2(op: &Operation) -> Option<(usize, Matrix2)> {
    match op {
        Operation::Gate { gate, target } => Some((*target, gate.matrix())),
        Operation::PauliRotation { pauli, theta } => match pauli.ops() {
            &[(target, p)] => {
                let gate = match p {
                    Pauli::I => return None,
                    Pauli::X => Gate::Rx(*theta),
                    Pauli::Y => Gate::Ry(*theta),
                    Pauli::Z => Gate::Rz(*theta),
                };
                Some((target, gate.matrix()))
            }
            _ => None,
        },
        _ => None,
    }
}

/// Check if a global phase `e^{iγ}` is real (`±1`) within `tol`.
#[inline]
pub(crate) fn phase_is_real(phase: f64, tol: f64) -> bool {
    C64::cis(phase).im.abs() <= tol
}

/// Canonical synthesis of a 2×2 unitary `m`: returns `(gate, gamma)` such
/// that `e^{iγ} · gate.matrix() == m` (or `e^{iγ} I == m` when `gate` is
/// `None`).
///
/// Prefers named gates (`X`, `Y`, `Z`, `H`, `S`, `Sdg`, `T`, `Tdg`), then
/// single-axis rotations (`Phase`, `Ry`, `Rx`, `Rz`), then `U(θ, φ, λ)`.
/// If `m` is real (all imaginary parts ≤ `tol`), the returned gate and phase
/// are guaranteed to be real so `RealStatevector` can run the result.
pub(crate) fn synthesize_1q(m: &Matrix2, tol: f64) -> (Option<Gate>, f64) {
    let real = mat2_is_real(m, tol);

    // 0 gates: identity up to global phase.
    if let Some(phase) = mat2_identity_phase(m, tol) {
        return (None, snap_real_phase(phase, real, tol));
    }

    // Named Pauli / Clifford / T gates (exact up to global phase).
    // When `m` is real, skip complex named gates so real circuits stay real.
    let named: &[Gate] = if real {
        &[Gate::X, Gate::Z, Gate::H]
    } else {
        &[
            Gate::X,
            Gate::Y,
            Gate::Z,
            Gate::H,
            Gate::S,
            Gate::Sdg,
            Gate::T,
            Gate::Tdg,
        ]
    };
    for &g in named {
        if let Some(phase) = mat2_matches_up_to_phase(m, &g.matrix(), tol)
            && (!real || phase_is_real(phase, tol))
        {
            return (Some(g), snap_real_phase(phase, real, tol));
        }
    }

    // Diagonal gate: e^{i p0} diag(1, e^{i lam}) = e^{i p0} Phase(lam).
    if mat2_is_diagonal(m, tol) {
        let p0 = m[0][0].arg();
        let p1 = m[1][1].arg();
        let lam = wrap_pi(p1 - p0);
        // Prefer Rz(lam) when p0 == -lam/2 (no global phase needed).
        let rz_phase = wrap_pi(p0 + lam / 2.0);
        if rz_phase.abs() <= tol && !real {
            return (Some(Gate::Rz(lam)), 0.0);
        }
        return (
            Some(Gate::Phase(lam)),
            snap_real_phase(wrap_pi(p0), real, tol),
        );
    }

    // Real orthogonal matrix: either rotation Ry(θ) (det = +1) or reflection
    // U(θ, 0, π) (det = -1), both with zero global phase!
    if real {
        let det = (m[0][0] * m[1][1] - m[0][1] * m[1][0]).re;
        let theta = 2.0 * m[1][0].re.atan2(m[0][0].re);
        if det > 0.0 {
            let g = Gate::Ry(wrap_2pi(theta));
            if let Some(phase) = mat2_matches_up_to_phase(m, &g.matrix(), tol)
                && phase_is_real(phase, tol)
            {
                return (Some(g), snap_real_phase(phase, true, tol));
            }
        } else {
            let g = Gate::U {
                theta: wrap_2pi(theta),
                phi: 0.0,
                lambda: PI,
            };
            if let Some(phase) = mat2_matches_up_to_phase(m, &g.matrix(), tol)
                && phase_is_real(phase, tol)
            {
                return (Some(g), snap_real_phase(phase, true, tol));
            }
        }
    }

    // Factor out det^{1/2} to inspect the SU(2) representative for Rx / Ry.
    let half_det = mat2_det(m).arg() / 2.0;
    let inv_phase = C64::cis(-half_det);
    let s = [
        [m[0][0] * inv_phase, m[0][1] * inv_phase],
        [m[1][0] * inv_phase, m[1][1] * inv_phase],
    ];

    // Pure Ry(θ) up to phase.
    if s[0][0].im.abs() <= tol && s[1][0].im.abs() <= tol {
        let theta = wrap_2pi(2.0 * s[1][0].re.atan2(s[0][0].re));
        let g = Gate::Ry(theta);
        if let Some(phase) = mat2_matches_up_to_phase(m, &g.matrix(), tol) {
            return (Some(g), phase);
        }
    }

    // Pure Rx(θ) up to phase.
    if s[0][0].im.abs() <= tol && s[1][0].re.abs() <= tol {
        let theta = wrap_2pi(2.0 * (-s[1][0].im).atan2(s[0][0].re));
        let g = Gate::Rx(theta);
        if let Some(phase) = mat2_matches_up_to_phase(m, &g.matrix(), tol) {
            return (Some(g), phase);
        }
    }

    // General ZYZ: m = e^{iγ} U(θ, φ, λ).
    let (theta, phi, lambda, gamma) = zyz_decompose(m, tol);
    (Some(Gate::U { theta, phi, lambda }), gamma)
}

fn snap_real_phase(phase: f64, real: bool, tol: f64) -> f64 {
    if !real {
        return if phase.abs() <= tol { 0.0 } else { phase };
    }
    let w = wrap_pi(phase);
    if w.abs() <= tol {
        0.0
    } else if (w.abs() - PI).abs() <= tol {
        PI
    } else {
        w
    }
}

/// Exact ZYZ decomposition: `m = e^{iγ} U(θ, φ, λ)`.
pub(crate) fn zyz_decompose(m: &Matrix2, tol: f64) -> (f64, f64, f64, f64) {
    let c = m[0][0].norm();
    let s = m[1][0].norm();
    let theta = 2.0 * s.atan2(c);
    let eps = tol.max(1e-13);
    if c <= eps {
        // U(π, φ, λ) = [[0, -e^{iλ}], [e^{iφ}, 0]]; choose λ = 0.
        let gamma = wrap_pi((-m[0][1]).arg());
        let phi = wrap_pi(m[1][0].arg() - gamma);
        return (PI, phi, 0.0, if gamma.abs() <= tol { 0.0 } else { gamma });
    }
    if s <= eps {
        let gamma = wrap_pi(m[0][0].arg());
        let lambda = wrap_pi(m[1][1].arg() - gamma);
        return (
            0.0,
            0.0,
            lambda,
            if gamma.abs() <= tol { 0.0 } else { gamma },
        );
    }
    let gamma = wrap_pi(m[0][0].arg());
    let phi = wrap_pi(m[1][0].arg() - gamma);
    let lambda = wrap_pi((-m[0][1]).arg() - gamma);
    (
        theta,
        if phi.abs() <= tol { 0.0 } else { phi },
        if lambda.abs() <= tol { 0.0 } else { lambda },
        if gamma.abs() <= tol { 0.0 } else { gamma },
    )
}

/// Check if a gate is strictly real (usable in `RealStatevector`).
pub(crate) fn gate_is_real(gate: &Gate, tol: f64) -> bool {
    mat2_is_real(&gate.matrix(), tol)
}

/// Check if a 1-qubit gate is diagonal in the computational basis.
pub(crate) fn gate_is_diagonal(gate: &Gate, tol: f64) -> bool {
    match gate {
        Gate::Z | Gate::S | Gate::Sdg | Gate::T | Gate::Tdg | Gate::Rz(_) | Gate::Phase(_) => true,
        Gate::H | Gate::X | Gate::Y => false,
        Gate::Rx(t) | Gate::Ry(t) => (t / 2.0).sin().abs() <= tol,
        Gate::U { theta, .. } => (theta / 2.0).sin().abs() <= tol,
        Gate::Unitary(m) => mat2_is_diagonal(m, tol),
    }
}

/// Check if a single-qubit gate commutes with `X` (i.e. `I`, `X`, `Rx(θ)`).
pub(crate) fn gate_commutes_with_x(gate: &Gate, tol: f64) -> bool {
    match gate {
        Gate::X | Gate::Rx(_) => true,
        Gate::H | Gate::Y | Gate::Z | Gate::S | Gate::Sdg | Gate::T | Gate::Tdg => false,
        _ => mat2_commutes(&gate.matrix(), &Gate::X.matrix(), tol),
    }
}

/// Check if a single-qubit gate commutes with `Y` (i.e. `I`, `Y`, `Ry(θ)`).
pub(crate) fn gate_commutes_with_y(gate: &Gate, tol: f64) -> bool {
    match gate {
        Gate::Y | Gate::Ry(_) => true,
        Gate::H | Gate::X | Gate::Z | Gate::S | Gate::Sdg | Gate::T | Gate::Tdg => false,
        _ => mat2_commutes(&gate.matrix(), &Gate::Y.matrix(), tol),
    }
}

/// Check if a single-qubit gate is a pure Z rotation / phase (and return its
/// `Rz` angle plus residual global phase `e^{iγ}` if so).
/// `gate.matrix() == e^{iγ} Rz(angle)`.
pub(crate) fn as_rz_and_phase(gate: &Gate, tol: f64) -> Option<(f64, f64)> {
    match *gate {
        Gate::Z => Some((PI, FRAC_PI_2)),
        Gate::S => Some((FRAC_PI_2, FRAC_PI_4)),
        Gate::Sdg => Some((-FRAC_PI_2, -FRAC_PI_4)),
        Gate::T => Some((FRAC_PI_4, FRAC_PI_4 / 2.0)),
        Gate::Tdg => Some((-FRAC_PI_4, -FRAC_PI_4 / 2.0)),
        Gate::Rz(t) => Some((t, 0.0)),
        Gate::Phase(lam) => Some((lam, lam / 2.0)),
        _ => {
            let m = gate.matrix();
            if !mat2_is_diagonal(&m, tol) {
                return None;
            }
            let p0 = m[0][0].arg();
            let p1 = m[1][1].arg();
            let angle = wrap_pi(p1 - p0);
            let gamma = wrap_pi(p0 + angle / 2.0);
            Some((angle, gamma))
        }
    }
}

/// Convert a fixed `Rx`/`Ry`/`Rz`/`Phase` with a constant angle into its
/// simplified `(Option<Gate>, global_phase)` form if it is trivial, otherwise `None`.
pub(crate) fn simplify_trivial_1q(gate: &Gate, tol: f64) -> Option<(Option<Gate>, f64)> {
    match *gate {
        Gate::Rx(t) | Gate::Ry(t) | Gate::Rz(t) => {
            let w = wrap_2pi(t);
            if w.abs() <= tol {
                Some((None, 0.0))
            } else if (w.abs() - TAU).abs() <= tol {
                Some((None, PI))
            } else {
                None
            }
        }
        Gate::Phase(lam) => {
            let w = wrap_pi(lam);
            if w.abs() <= tol {
                Some((None, 0.0))
            } else {
                None
            }
        }
        Gate::U { theta, phi, lambda } => {
            let m = Gate::U { theta, phi, lambda }.matrix();
            mat2_identity_phase(&m, tol).map(|p| (None, p))
        }
        Gate::Unitary(m) => Some(synthesize_1q(&m, tol)),
        _ => None,
    }
}

/// Standard `I`, `X`, `Y`, `Z` matrices for convenience.
pub(crate) fn pauli_mat2(p: Pauli) -> Matrix2 {
    match p {
        Pauli::I => I2,
        Pauli::X => [[ZERO, ONE], [ONE, ZERO]],
        Pauli::Y => [[ZERO, -I], [I, ZERO]],
        Pauli::Z => [[ONE, ZERO], [ZERO, -ONE]],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_mat_eq(a: &Matrix2, b: &Matrix2, phase: f64) {
        let f = C64::cis(phase);
        for r in 0..2 {
            for c in 0..2 {
                assert!(
                    (a[r][c] - f * b[r][c]).norm() < 1e-11,
                    "mismatch at ({r},{c}): {:?} vs {:?}",
                    a[r][c],
                    f * b[r][c]
                );
            }
        }
    }

    #[test]
    fn synthesize_roundtrips_named_and_general_gates() {
        let gates = [
            Gate::H,
            Gate::X,
            Gate::Y,
            Gate::Z,
            Gate::S,
            Gate::Sdg,
            Gate::T,
            Gate::Tdg,
            Gate::Rx(0.73),
            Gate::Ry(-1.42),
            Gate::Rz(2.11),
            Gate::Phase(0.95),
            Gate::U {
                theta: 1.2,
                phi: -0.7,
                lambda: 2.3,
            },
            Gate::U {
                theta: PI,
                phi: 0.4,
                lambda: -1.1,
            },
            Gate::U {
                theta: 0.0,
                phi: 0.5,
                lambda: 0.8,
            },
        ];
        for g in gates {
            let m = g.matrix();
            let (syn, phase) = synthesize_1q(&m, 1e-12);
            let got = syn.map_or(I2, |g| g.matrix());
            assert_mat_eq(&m, &got, phase);
        }
    }

    #[test]
    fn synthesize_preserves_real_matrices() {
        let m1 = mat2_mul(&Gate::Ry(0.4).matrix(), &Gate::H.matrix());
        let (syn, phase) = synthesize_1q(&m1, 1e-12);
        let got = syn.map_or(I2, |g| g.matrix());
        assert!(mat2_is_real(&got, 1e-12));
        assert!(phase_is_real(phase, 1e-12));
        assert_mat_eq(&m1, &got, phase);
    }
}
