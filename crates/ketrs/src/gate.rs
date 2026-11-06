//! Single-qubit gates and their matrices.
//!
//! Matrix conventions match Qiskit, e.g. `Rz(θ) = diag(e^{-iθ/2}, e^{iθ/2})`.

use num_complex::Complex64 as C64;
use std::f64::consts::FRAC_1_SQRT_2;

/// A 2×2 complex matrix, indexed `[row][column]`.
pub type Matrix2 = [[C64; 2]; 2];

/// A 4×4 complex matrix, indexed `[row][column]`.
pub type Matrix4 = [[C64; 4]; 4];

/// A single-qubit gate.
///
/// Multi-qubit operations are built from these via
/// [`Operation`](crate::Operation), e.g. CX is a controlled [`Gate::X`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Gate {
    /// Hadamard.
    H,
    /// Pauli X.
    X,
    /// Pauli Y.
    Y,
    /// Pauli Z.
    Z,
    /// √Z.
    S,
    /// S†.
    Sdg,
    /// ⁴√Z.
    T,
    /// T†.
    Tdg,
    /// Rotation about X by the given angle.
    Rx(f64),
    /// Rotation about Y by the given angle.
    Ry(f64),
    /// Rotation about Z by the given angle.
    Rz(f64),
    /// `diag(1, e^{iλ})`.
    Phase(f64),
    /// Generic rotation `U(θ, φ, λ)`, as in Qiskit's `UGate`.
    U {
        /// θ
        theta: f64,
        /// φ
        phi: f64,
        /// λ
        lambda: f64,
    },
    /// An arbitrary matrix. It is the caller's job to keep it unitary.
    Unitary(Matrix2),
}

const ZERO: C64 = C64::new(0.0, 0.0);
const ONE: C64 = C64::new(1.0, 0.0);
const I: C64 = C64::new(0.0, 1.0);

impl Gate {
    /// The gate's matrix in the basis `(|0⟩, |1⟩)`.
    pub fn matrix(&self) -> Matrix2 {
        let h = C64::new(FRAC_1_SQRT_2, 0.0);
        match *self {
            Gate::H => [[h, h], [h, -h]],
            Gate::X => [[ZERO, ONE], [ONE, ZERO]],
            Gate::Y => [[ZERO, -I], [I, ZERO]],
            Gate::Z => [[ONE, ZERO], [ZERO, -ONE]],
            Gate::S => [[ONE, ZERO], [ZERO, I]],
            Gate::Sdg => [[ONE, ZERO], [ZERO, -I]],
            Gate::T => [[ONE, ZERO], [ZERO, C64::cis(std::f64::consts::FRAC_PI_4)]],
            Gate::Tdg => [[ONE, ZERO], [ZERO, C64::cis(-std::f64::consts::FRAC_PI_4)]],
            Gate::Rx(theta) => {
                let (c, s) = cos_sin_half(theta);
                [[c, -I * s], [-I * s, c]]
            }
            Gate::Ry(theta) => {
                let (c, s) = cos_sin_half(theta);
                [[c, -s], [s, c]]
            }
            Gate::Rz(theta) => [
                [C64::cis(-theta / 2.0), ZERO],
                [ZERO, C64::cis(theta / 2.0)],
            ],
            Gate::Phase(lambda) => [[ONE, ZERO], [ZERO, C64::cis(lambda)]],
            Gate::U { theta, phi, lambda } => {
                let (c, s) = cos_sin_half(theta);
                [
                    [c, -C64::cis(lambda) * s],
                    [C64::cis(phi) * s, C64::cis(phi + lambda) * c],
                ]
            }
            Gate::Unitary(m) => m,
        }
    }

    /// The inverse gate, so that `g.inverse().matrix()` is `g.matrix()†`.
    pub fn inverse(&self) -> Gate {
        match *self {
            Gate::H | Gate::X | Gate::Y | Gate::Z => *self,
            Gate::S => Gate::Sdg,
            Gate::Sdg => Gate::S,
            Gate::T => Gate::Tdg,
            Gate::Tdg => Gate::T,
            Gate::Rx(theta) => Gate::Rx(-theta),
            Gate::Ry(theta) => Gate::Ry(-theta),
            Gate::Rz(theta) => Gate::Rz(-theta),
            Gate::Phase(lambda) => Gate::Phase(-lambda),
            Gate::U { theta, phi, lambda } => Gate::U {
                theta: -theta,
                phi: -lambda,
                lambda: -phi,
            },
            Gate::Unitary(m) => Gate::Unitary(adjoint(&m)),
        }
    }
}

fn cos_sin_half(theta: f64) -> (C64, C64) {
    let (s, c) = (theta / 2.0).sin_cos();
    (C64::new(c, 0.0), C64::new(s, 0.0))
}

/// Conjugate transpose of a square matrix.
pub(crate) fn adjoint<const N: usize>(m: &[[C64; N]; N]) -> [[C64; N]; N] {
    std::array::from_fn(|r| std::array::from_fn(|c| m[c][r].conj()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_gates() -> Vec<Gate> {
        vec![
            Gate::H,
            Gate::X,
            Gate::Y,
            Gate::Z,
            Gate::S,
            Gate::Sdg,
            Gate::T,
            Gate::Tdg,
            Gate::Rx(0.3),
            Gate::Ry(-1.2),
            Gate::Rz(2.5),
            Gate::Phase(0.7),
            Gate::U {
                theta: 0.4,
                phi: 1.1,
                lambda: -0.6,
            },
            Gate::Unitary(Gate::Ry(0.9).matrix()),
        ]
    }

    fn mul(a: &Matrix2, b: &Matrix2) -> Matrix2 {
        std::array::from_fn(|r| std::array::from_fn(|c| a[r][0] * b[0][c] + a[r][1] * b[1][c]))
    }

    fn assert_identity(m: &Matrix2, gate: Gate) {
        for (r, row) in m.iter().enumerate() {
            for (c, &v) in row.iter().enumerate() {
                let want = if r == c { ONE } else { ZERO };
                assert!((v - want).norm() < 1e-12, "{gate:?}: got {m:?}");
            }
        }
    }

    #[test]
    fn gates_are_unitary() {
        for g in all_gates() {
            let m = g.matrix();
            assert_identity(&mul(&m, &adjoint(&m)), g);
        }
    }

    #[test]
    fn inverse_undoes_gate() {
        for g in all_gates() {
            assert_identity(&mul(&g.inverse().matrix(), &g.matrix()), g);
        }
    }

    #[test]
    fn named_gates_match_rotations_up_to_phase() {
        // Rx(π) = -iX, Rz(π/2) = e^{-iπ/4} S, U(π/2, 0, π) = H.
        let pi = std::f64::consts::PI;
        let cases = [
            (Gate::Rx(pi), Gate::X, -I),
            (Gate::Rz(pi / 2.0), Gate::S, C64::cis(-pi / 4.0)),
            (
                Gate::U {
                    theta: pi / 2.0,
                    phi: 0.0,
                    lambda: pi,
                },
                Gate::H,
                ONE,
            ),
        ];
        for (rot, named, phase) in cases {
            let (a, b) = (rot.matrix(), named.matrix());
            for r in 0..2 {
                for c in 0..2 {
                    assert!((a[r][c] - phase * b[r][c]).norm() < 1e-12, "{rot:?}");
                }
            }
        }
    }
}
