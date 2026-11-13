//! Statevector methods for [`SparseOperator`]s.

use super::DenseState;
use super::evolve::{KRYLOV_DIM, krylov_evolve};
use super::krylov::LinearOperator;
use crate::error::Error;
use crate::scalar::{ComplexScalar, Scalar};
use crate::sparse::SparseOperator;

/// Keep the Krylov basis when it fits in this many bytes; otherwise replay.
const BASIS_MEMORY: usize = 2 << 30;

impl<T: Scalar> DenseState<T> {
    fn check_width(&self, op: &SparseOperator) -> Result<(), Error> {
        if op.n_qubits() != self.n_qubits {
            return Err(Error::QubitCountMismatch {
                expected: self.n_qubits,
                found: op.n_qubits(),
            });
        }
        Ok(())
    }

    /// `⟨ψ|A|ψ⟩` for a sparse Hermitian operator.
    pub fn expectation_sparse(&self, op: &SparseOperator) -> Result<f64, Error> {
        self.check_width(op)?;
        Ok(op.prepare::<T>().expectation(&self.amps))
    }

    /// `A|ψ⟩` for a sparse Hermitian operator, as a new, generally
    /// unnormalised, state. A real state needs a real operator.
    pub fn apply_sparse(&self, op: &SparseOperator) -> Result<Self, Error> {
        self.check_width(op)?;
        if T::from_complex(num_complex::Complex64::new(0.0, 1.0)).is_none() && !op.is_real() {
            return Err(Error::NotReal(
                "a complex operator applied to a real statevector".into(),
            ));
        }
        let mut out = vec![T::ZERO; self.amps.len()];
        op.prepare::<T>().apply(&self.amps, &mut out);
        Ok(DenseState {
            n_qubits: self.n_qubits,
            amps: out,
        })
    }
}

impl<T: ComplexScalar> DenseState<T> {
    /// Evolve by `exp(-iAt)` for a sparse Hermitian operator `A`, by the
    /// same Lanczos time-stepping as [`evolve`](Self::evolve).
    pub fn evolve_sparse(&mut self, op: &SparseOperator, time: f64) -> Result<(), Error> {
        self.check_width(op)?;
        let store = (KRYLOV_DIM + 3) * self.amps.len() * size_of::<T>() <= BASIS_MEMORY;
        krylov_evolve(&op.prepare::<T>(), &mut self.amps, time, store)
    }
}

#[cfg(test)]
mod tests {
    use crate::observable::PauliSum;
    use crate::sparse::SparseOperator;
    use crate::statevector::Statevector;
    use num_complex::Complex64 as C64;

    /// The matrix of `0.7 XY + 0.3 ZZ + 0.5 IX` on two qubits, as CSR.
    fn sparse_and_pauli() -> (SparseOperator, PauliSum) {
        let pauli = PauliSum::from_list([("XY", 0.7), ("ZZ", 0.3), ("IX", 0.5)]).unwrap();
        let x = [
            [C64::new(0., 0.), C64::new(1., 0.)],
            [C64::new(1., 0.), C64::new(0., 0.)],
        ];
        let y = [
            [C64::new(0., 0.), C64::new(0., -1.)],
            [C64::new(0., 1.), C64::new(0., 0.)],
        ];
        let z = [
            [C64::new(1., 0.), C64::new(0., 0.)],
            [C64::new(0., 0.), C64::new(-1., 0.)],
        ];
        let i = [
            [C64::new(1., 0.), C64::new(0., 0.)],
            [C64::new(0., 0.), C64::new(1., 0.)],
        ];
        // Label "AB": A on qubit 1 (high bit), B on qubit 0.
        let kron = |a: [[C64; 2]; 2], b: [[C64; 2]; 2], r: usize, c: usize| {
            a[r >> 1][c >> 1] * b[r & 1][c & 1]
        };
        let mut dense = [[C64::new(0., 0.); 4]; 4];
        for (r, row) in dense.iter_mut().enumerate() {
            for (c, v) in row.iter_mut().enumerate() {
                *v = 0.7 * kron(x, y, r, c) + 0.3 * kron(z, z, r, c) + 0.5 * kron(i, x, r, c);
            }
        }
        let (mut indptr, mut indices, mut data) = (vec![0], vec![], vec![]);
        for row in &dense {
            for (c, &v) in row.iter().enumerate() {
                if v.norm() > 0.0 {
                    indices.push(c);
                    data.push(v);
                }
            }
            indptr.push(indices.len());
        }
        (
            SparseOperator::from_csr(2, indptr, indices, data).unwrap(),
            pauli,
        )
    }

    #[test]
    fn matches_pauli_sum() {
        let (sparse, pauli) = sparse_and_pauli();
        let amps: Vec<C64> = (0..4)
            .map(|k| C64::new(0.3 + k as f64, 0.2 * k as f64))
            .collect();
        let norm = amps.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
        let state = Statevector::from_amplitudes(amps.iter().map(|a| a / norm).collect()).unwrap();

        let e = state.expectation(&pauli).unwrap();
        assert!((state.expectation_sparse(&sparse).unwrap() - e).abs() < 1e-14);

        let a = state.apply_pauli_sum(&pauli).unwrap();
        let b = state.apply_sparse(&sparse).unwrap();
        for (x, y) in a.amplitudes().iter().zip(b.amplitudes()) {
            assert!((x - y).norm() < 1e-14);
        }

        let (mut a, mut b) = (state.clone(), state);
        a.evolve(&pauli, 1.3).unwrap();
        b.evolve_sparse(&sparse, 1.3).unwrap();
        for (x, y) in a.amplitudes().iter().zip(b.amplitudes()) {
            assert!((x - y).norm() < 1e-12);
        }
    }

    #[test]
    fn rejects_wrong_width() {
        let (sparse, _) = sparse_and_pauli();
        let state = Statevector::zero(3);
        assert!(state.expectation_sparse(&sparse).is_err());
    }
}
