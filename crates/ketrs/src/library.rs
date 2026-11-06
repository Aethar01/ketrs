//! Standard circuits.

use crate::circuit::Circuit;
use std::f64::consts::PI;

/// The quantum Fourier transform, built as in Qiskit's `QFT`.
///
/// Maps `|x⟩` to `2^{-n/2} Σₖ e^{2πi·xk/2ⁿ} |k⟩`.
pub fn qft(n_qubits: usize) -> Circuit {
    let mut circ = Circuit::new(n_qubits);
    for j in (0..n_qubits).rev() {
        circ.h(j);
        for k in (0..j).rev() {
            circ.cp(PI / f64::powi(2.0, (j - k) as i32), j, k);
        }
    }
    for i in 0..n_qubits / 2 {
        circ.swap(i, n_qubits - 1 - i);
    }
    circ
}

/// Prepares the GHZ state `(|0…0⟩ + |1…1⟩)/√2`.
pub fn ghz(n_qubits: usize) -> Circuit {
    let mut circ = Circuit::new(n_qubits);
    if n_qubits > 0 {
        circ.h(0);
    }
    for q in 1..n_qubits {
        circ.cx(q - 1, q);
    }
    circ
}
