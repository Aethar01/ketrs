//! Matrix-free Pauli sums on statevectors: `H|ψ⟩` and expectation values.

use super::DenseState;
use super::kernels::{BLOCK, PARALLEL_THRESHOLD};
use super::krylov::LinearOperator;
use crate::error::Error;
use crate::observable::PauliSum;
use crate::scalar::Scalar;
use rayon::prelude::*;
use std::collections::HashMap;
use std::ops::Range;

/// A [`PauliSum`] arranged for fast application to `2ⁿ`-amplitude vectors.
///
/// Every string with X-mask `x` sends `|i⟩` to `|i ⊕ x⟩`, so terms sharing
/// `x` form one group and `(Hψ)_j = Σₓ cₓ(j ⊕ x) ψ_{j⊕x}`, where
/// `cₓ(i) = Σₜ cₜ iⁿʸ (-1)^|i ∧ zₜ|`. All the Z-type terms (`x = 0`) form a
/// single diagonal, which can be tabulated once for repeated use.
pub(super) struct CompiledPauliSum<T> {
    diagonal: Diagonal,
    groups: Vec<Group<T>>,
}

enum Diagonal {
    /// `(z, c)` pairs, evaluated per amplitude.
    Terms(Vec<(usize, f64)>),
    /// The diagonal entry for every basis state.
    Table(Vec<f64>),
}

/// The terms sharing one X-mask.
pub(crate) struct Group<T> {
    pub x: usize,
    /// `(z, c·iⁿʸ)` for each term.
    pub terms: Vec<(usize, T)>,
}

impl<T: Scalar> Group<T> {
    /// `cₓ(i) = Σₜ cₜ iⁿʸ (-1)^|i ∧ zₜ|`, so `(Hψ)_{i⊕x} ∋ cₓ(i) ψᵢ`.
    #[inline]
    pub(crate) fn coefficient(&self, i: usize) -> T {
        self.terms
            .iter()
            .map(|&(z, c)| if (i & z).count_ones() % 2 == 1 { -c } else { c })
            .sum()
    }
}

/// A Pauli sum split into its diagonal (Z-type) terms and groups of terms
/// sharing an X-mask.
pub(crate) struct GroupedPauliSum<T> {
    /// `(z, c)` of each diagonal term.
    pub diagonal: Vec<(usize, f64)>,
    pub groups: Vec<Group<T>>,
}

impl<T: Scalar> GroupedPauliSum<T> {
    pub(crate) fn new(
        hamiltonian: &PauliSum,
        n_qubits: usize,
        imaginary: Imaginary,
    ) -> Result<Self, Error> {
        if hamiltonian.min_qubits() > n_qubits {
            return Err(Error::QubitOutOfRange {
                qubit: hamiltonian.min_qubits() - 1,
                n_qubits,
            });
        }
        let mut diagonal = Vec::new();
        let mut groups: Vec<Group<T>> = Vec::new();
        let mut group_of: HashMap<usize, usize> = HashMap::new();
        for (coeff, string) in hamiltonian.simplify(0.0).terms() {
            let m = string.masks();
            if m.x == 0 {
                diagonal.push((m.z, *coeff));
                continue;
            }
            let Some(c) = T::from_complex(*coeff * m.y_phase()) else {
                match imaginary {
                    Imaginary::Skip => continue,
                    Imaginary::Reject => {
                        return Err(Error::NotReal(format!(
                            "{string:?} is imaginary; use a complex statevector"
                        )));
                    }
                }
            };
            let g = *group_of.entry(m.x).or_insert_with(|| {
                groups.push(Group {
                    x: m.x,
                    terms: Vec::new(),
                });
                groups.len() - 1
            });
            groups[g].terms.push((m.z, c));
        }
        Ok(GroupedPauliSum { diagonal, groups })
    }
}

/// What to do with terms whose coefficient `T` can't hold: for a real `T`,
/// strings with an odd number of Y factors.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Imaginary {
    /// Fail with [`Error::NotReal`].
    Reject,
    /// Drop them. Correct for expectation values on real states, where such
    /// terms contribute exactly zero.
    Skip,
}

impl<T: Scalar> CompiledPauliSum<T> {
    pub(super) fn new(
        hamiltonian: &PauliSum,
        n_qubits: usize,
        imaginary: Imaginary,
        tabulate_diagonal: bool,
    ) -> Result<Self, Error> {
        let GroupedPauliSum { diagonal, groups } =
            GroupedPauliSum::new(hamiltonian, n_qubits, imaginary)?;
        let diagonal = if tabulate_diagonal && diagonal.len() > 1 {
            let mut table = vec![0.0; 1 << n_qubits];
            table
                .par_iter_mut()
                .enumerate()
                .for_each(|(j, d)| *d = diagonal_entry(&diagonal, j));
            Diagonal::Table(table)
        } else {
            Diagonal::Terms(diagonal)
        };
        Ok(CompiledPauliSum { diagonal, groups })
    }

    /// `(Hψ)_j`.
    #[inline]
    fn row(&self, psi: &[T], j: usize) -> T {
        let d = match &self.diagonal {
            Diagonal::Terms(terms) => diagonal_entry(terms, j),
            Diagonal::Table(table) => table[j],
        };
        let mut acc = T::from_real(d) * psi[j];
        for group in &self.groups {
            let i = j ^ group.x;
            acc += group.coefficient(i) * psi[i];
        }
        acc
    }

    /// `out = H psi`.
    pub(super) fn apply(&self, psi: &[T], out: &mut [T]) {
        LinearOperator::apply(self, psi, out)
    }
}

impl<T: Scalar> LinearOperator<T> for CompiledPauliSum<T> {
    fn apply(&self, psi: &[T], out: &mut [T]) {
        let fill = |offset: usize, chunk: &mut [T]| {
            for (k, o) in chunk.iter_mut().enumerate() {
                *o = self.row(psi, offset + k);
            }
        };
        if out.len() >= PARALLEL_THRESHOLD {
            out.par_chunks_mut(BLOCK)
                .enumerate()
                .for_each(|(i, chunk)| fill(i * BLOCK, chunk));
        } else {
            fill(0, out);
        }
    }
}

impl<T: Scalar> CompiledPauliSum<T> {
    /// `⟨ψ|H|ψ⟩` without storing `H|ψ⟩`.
    pub(super) fn expectation(&self, psi: &[T]) -> f64 {
        chunked_sum(psi.len(), |range| {
            range
                .map(|j| (psi[j].conj() * self.row(psi, j)).to_complex().re)
                .sum::<f64>()
        })
    }
}

pub(crate) fn diagonal_entry(terms: &[(usize, f64)], j: usize) -> f64 {
    terms
        .iter()
        .map(|&(z, c)| if (j & z).count_ones() % 2 == 1 { -c } else { c })
        .sum()
}

/// `Σ f(range)` over `0..len` in fixed chunks, summed in order, so the
/// result is identical from run to run despite running in parallel.
pub(crate) fn chunked_sum<S>(len: usize, f: impl Fn(Range<usize>) -> S + Sync) -> S
where
    S: Send + std::iter::Sum,
{
    if len < PARALLEL_THRESHOLD {
        return f(0..len);
    }
    let parts: Vec<S> = (0..len.div_ceil(BLOCK))
        .into_par_iter()
        .map(|c| f(c * BLOCK..((c + 1) * BLOCK).min(len)))
        .collect();
    parts.into_iter().sum()
}

impl<T: Scalar> DenseState<T> {
    /// `⟨ψ|H|ψ⟩` for a Hermitian `observable`, assuming `ψ` is normalised.
    ///
    /// Terms sharing an X-mask are evaluated in one pass, so this costs a
    /// pass per distinct X-mask rather than per term.
    pub fn expectation(&self, observable: &PauliSum) -> Result<f64, Error> {
        let op = CompiledPauliSum::<T>::new(observable, self.n_qubits, Imaginary::Skip, false)?;
        Ok(op.expectation(&self.amps))
    }

    /// `H|ψ⟩`, computed without forming a matrix. The result is generally
    /// not normalised.
    pub fn apply_pauli_sum(&self, hamiltonian: &PauliSum) -> Result<Self, Error> {
        let op = CompiledPauliSum::<T>::new(hamiltonian, self.n_qubits, Imaginary::Reject, false)?;
        let mut out = vec![T::ZERO; self.amps.len()];
        op.apply(&self.amps, &mut out);
        Ok(DenseState {
            n_qubits: self.n_qubits,
            amps: out,
        })
    }
}
