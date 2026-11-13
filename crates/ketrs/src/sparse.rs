//! Hermitian operators stored as sparse matrices in the computational basis.
//!
//! Some operators are cheap as matrices but expensive as Pauli sums: a hop
//! whose amplitude is an arbitrary function of the configuration, for
//! example, has one nonzero per hop but needs nearly every Z-string to be
//! written out in Paulis. [`SparseOperator`] stores such an operator in
//! compressed sparse row (CSR) form and works wherever a statevector takes
//! a Hamiltonian for `⟨ψ|A|ψ⟩`, `A|ψ⟩` or `exp(-iAt)|ψ⟩`.

use crate::error::Error;
use crate::scalar::Scalar;
use crate::statevector::krylov::LinearOperator;
use num_complex::Complex64 as C64;
use rayon::prelude::*;

/// Rows per state above which products run in parallel.
const PARALLEL_ROWS: usize = 1 << 12;

/// A Hermitian operator on `n` qubits, stored as a `2ⁿ × 2ⁿ` sparse matrix
/// in CSR form. Row and column `i` are the basis state with index `i`
/// (qubit 0 is the least significant bit, as everywhere in ketrs).
#[derive(Clone, Debug, PartialEq)]
pub struct SparseOperator {
    n_qubits: usize,
    indptr: Vec<usize>,
    indices: Vec<usize>,
    data: Vec<C64>,
    real: bool,
}

impl SparseOperator {
    /// Build from CSR arrays: the entries of row `i` are
    /// `data[indptr[i]..indptr[i+1]]` in columns
    /// `indices[indptr[i]..indptr[i+1]]`. Repeated entries are summed.
    ///
    /// Fails unless the arrays describe a Hermitian `2ⁿ × 2ⁿ` matrix (to a
    /// tolerance of `1e-12` relative to its largest entry).
    pub fn from_csr(
        n_qubits: usize,
        indptr: Vec<usize>,
        indices: Vec<usize>,
        data: Vec<C64>,
    ) -> Result<Self, Error> {
        if n_qubits >= usize::BITS as usize - 1 {
            return Err(Error::InvalidArgument(format!(
                "{n_qubits} qubits is too many"
            )));
        }
        let dim = 1usize << n_qubits;
        let invalid = |message: String| Err(Error::InvalidArgument(message));
        if indptr.len() != dim + 1 {
            return invalid(format!(
                "indptr has {} entries, a {dim}x{dim} matrix needs {}",
                indptr.len(),
                dim + 1
            ));
        }
        if indptr[0] != 0 || indptr.windows(2).any(|w| w[0] > w[1]) {
            return invalid("indptr must start at 0 and never decrease".into());
        }
        if indptr[dim] != indices.len() || indices.len() != data.len() {
            return invalid(format!(
                "indptr ends at {} but there are {} indices and {} values",
                indptr[dim],
                indices.len(),
                data.len()
            ));
        }
        if let Some(&j) = indices.iter().find(|&&j| j >= dim) {
            return invalid(format!("column {j} is out of range for dimension {dim}"));
        }
        if data.iter().any(|c| !c.re.is_finite() || !c.im.is_finite()) {
            return invalid("matrix entries must be finite".into());
        }

        // Sort each row by column and merge repeats, so lookups can bisect.
        let mut new_indptr = Vec::with_capacity(dim + 1);
        let mut new_indices = Vec::with_capacity(indices.len());
        let mut new_data = Vec::with_capacity(data.len());
        new_indptr.push(0);
        let mut row: Vec<(usize, C64)> = Vec::new();
        for i in 0..dim {
            row.clear();
            row.extend((indptr[i]..indptr[i + 1]).map(|k| (indices[k], data[k])));
            row.sort_unstable_by_key(|&(j, _)| j);
            for &(j, c) in &row {
                if new_indices.len() > new_indptr[i] && *new_indices.last().unwrap() == j {
                    *new_data.last_mut().unwrap() += c;
                } else {
                    new_indices.push(j);
                    new_data.push(c);
                }
            }
            new_indptr.push(new_indices.len());
        }

        let op = SparseOperator {
            n_qubits,
            real: new_data.iter().all(|c| c.im == 0.0),
            indptr: new_indptr,
            indices: new_indices,
            data: new_data,
        };
        op.check_hermitian()?;
        Ok(op)
    }

    fn check_hermitian(&self) -> Result<(), Error> {
        let scale = self.data.iter().map(|c| c.norm()).fold(0.0, f64::max);
        let tolerance = 1e-12 * scale.max(f64::MIN_POSITIVE);
        for i in 0..self.dim() {
            for k in self.indptr[i]..self.indptr[i + 1] {
                let j = self.indices[k];
                let mirrored = self.entry(j, i);
                if (self.data[k] - mirrored.conj()).norm() > tolerance {
                    return Err(Error::InvalidArgument(format!(
                        "the matrix is not Hermitian: entry ({i}, {j}) is {} but ({j}, {i}) is {}",
                        self.data[k], mirrored
                    )));
                }
            }
        }
        Ok(())
    }

    /// The entry in row `i`, column `j`.
    pub fn entry(&self, i: usize, j: usize) -> C64 {
        let row = self.indptr[i]..self.indptr[i + 1];
        match self.indices[row.clone()].binary_search(&j) {
            Ok(k) => self.data[row.start + k],
            Err(_) => C64::new(0.0, 0.0),
        }
    }

    /// Number of qubits the operator acts on.
    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    /// Matrix dimension, `2ⁿ`.
    pub fn dim(&self) -> usize {
        1 << self.n_qubits
    }

    /// Number of stored entries.
    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    /// Whether every entry is real (a real symmetric matrix).
    pub fn is_real(&self) -> bool {
        self.real
    }

    /// The matrix with entries converted to `T`, ready to multiply states
    /// of that type. Real `T` keeps only the real parts of the entries,
    /// which is exact for expectation values of real states (the imaginary
    /// part of a Hermitian matrix is antisymmetric) and is otherwise
    /// rejected by the callers.
    pub(crate) fn prepare<T: Scalar>(&self) -> PreparedSparse<'_, T> {
        let data = self
            .data
            .iter()
            .map(|&c| T::from_complex(c).unwrap_or_else(|| T::from_real(c.re)))
            .collect();
        PreparedSparse { op: self, data }
    }
}

/// A [`SparseOperator`] with its entries converted to the amplitude type.
pub(crate) struct PreparedSparse<'a, T> {
    op: &'a SparseOperator,
    data: Vec<T>,
}

impl<T: Scalar> PreparedSparse<'_, T> {
    fn row(&self, i: usize, x: &[T]) -> T {
        let (indptr, indices) = (&self.op.indptr, &self.op.indices);
        let mut acc = T::ZERO;
        for k in indptr[i]..indptr[i + 1] {
            acc += self.data[k] * x[indices[k]];
        }
        acc
    }

    /// `⟨x|A|x⟩`, real for Hermitian `A`.
    pub(crate) fn expectation(&self, x: &[T]) -> f64 {
        let term = |i: usize| (x[i].conj() * self.row(i, x)).to_complex().re;
        if x.len() >= PARALLEL_ROWS {
            (0..x.len()).into_par_iter().map(term).sum()
        } else {
            (0..x.len()).map(term).sum()
        }
    }
}

impl<T: Scalar> LinearOperator<T> for PreparedSparse<'_, T> {
    fn apply(&self, x: &[T], out: &mut [T]) {
        if x.len() >= PARALLEL_ROWS {
            out.par_iter_mut()
                .enumerate()
                .for_each(|(i, o)| *o = self.row(i, x));
        } else {
            for (i, o) in out.iter_mut().enumerate() {
                *o = self.row(i, x);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(re: f64, im: f64) -> C64 {
        C64::new(re, im)
    }

    #[test]
    fn rejects_non_hermitian() {
        // [[0, 1], [2, 0]]
        let err =
            SparseOperator::from_csr(1, vec![0, 1, 2], vec![1, 0], vec![c(1., 0.), c(2., 0.)]);
        assert!(err.is_err());
        // [[0, i], [i, 0]] is not Hermitian; [[0, i], [-i, 0]] is.
        assert!(
            SparseOperator::from_csr(1, vec![0, 1, 2], vec![1, 0], vec![c(0., 1.), c(0., 1.)])
                .is_err()
        );
        let y = SparseOperator::from_csr(1, vec![0, 1, 2], vec![1, 0], vec![c(0., -1.), c(0., 1.)])
            .unwrap();
        assert!(!y.is_real());
    }

    #[test]
    fn merges_repeated_entries() {
        let op = SparseOperator::from_csr(
            1,
            vec![0, 2, 3],
            vec![1, 1, 0],
            vec![c(0.5, 0.), c(0.5, 0.), c(1., 0.)],
        )
        .unwrap();
        assert_eq!(op.nnz(), 2);
        assert_eq!(op.entry(0, 1), c(1., 0.));
    }

    #[test]
    fn rejects_bad_shapes() {
        assert!(SparseOperator::from_csr(2, vec![0, 0], vec![], vec![]).is_err());
        assert!(SparseOperator::from_csr(1, vec![0, 1, 1], vec![2], vec![c(1., 0.)]).is_err());
    }
}
