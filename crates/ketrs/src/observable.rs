//! Pauli-string observables.
//!
//! Labels follow Qiskit: the rightmost character acts on qubit 0, so
//! `"XIZ"` is Z on qubit 0 and X on qubit 2.

use crate::error::Error;

/// A single-qubit Pauli operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Pauli {
    /// Identity.
    I,
    /// Pauli X.
    X,
    /// Pauli Y.
    Y,
    /// Pauli Z.
    Z,
}

impl TryFrom<char> for Pauli {
    type Error = Error;

    fn try_from(ch: char) -> Result<Self, Error> {
        match ch {
            'I' => Ok(Pauli::I),
            'X' => Ok(Pauli::X),
            'Y' => Ok(Pauli::Y),
            'Z' => Ok(Pauli::Z),
            _ => Err(Error::InvalidPauli(ch)),
        }
    }
}

/// A tensor product of Paulis, stored sparsely as `(qubit, Pauli)` pairs
/// sorted by qubit, with identities left out.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct PauliString {
    ops: Vec<(usize, Pauli)>,
}

impl PauliString {
    /// The identity.
    pub fn identity() -> Self {
        Self::default()
    }

    /// Build from `(qubit, Pauli)` pairs in any order.
    pub fn new(ops: impl IntoIterator<Item = (usize, Pauli)>) -> Result<Self, Error> {
        let mut ops: Vec<_> = ops.into_iter().filter(|&(_, p)| p != Pauli::I).collect();
        ops.sort_by_key(|&(q, _)| q);
        if let Some(w) = ops.windows(2).find(|w| w[0].0 == w[1].0) {
            return Err(Error::DuplicateQubit(w[0].0));
        }
        Ok(PauliString { ops })
    }

    /// Parse a Qiskit-style label such as `"XIZ"`.
    pub fn from_label(label: &str) -> Result<Self, Error> {
        let ops = label
            .chars()
            .rev()
            .enumerate()
            .map(|(q, ch)| Pauli::try_from(ch).map(|p| (q, p)))
            .collect::<Result<Vec<_>, _>>()?;
        Self::new(ops)
    }

    /// Whether the two strings commute: they anticommute exactly when
    /// they differ (both non-identity) on an odd number of qubits.
    pub fn commutes_with(&self, other: &PauliString) -> bool {
        let (mut a, mut b) = (self.ops.iter().peekable(), other.ops.iter().peekable());
        let mut differ = 0;
        while let (Some(&&(qa, pa)), Some(&&(qb, pb))) = (a.peek(), b.peek()) {
            match qa.cmp(&qb) {
                std::cmp::Ordering::Less => {
                    a.next();
                }
                std::cmp::Ordering::Greater => {
                    b.next();
                }
                std::cmp::Ordering::Equal => {
                    differ += (pa != pb) as usize;
                    a.next();
                    b.next();
                }
            }
        }
        differ % 2 == 0
    }

    /// The non-identity factors, sorted by qubit.
    pub fn ops(&self) -> &[(usize, Pauli)] {
        &self.ops
    }

    /// Number of non-identity factors.
    pub fn weight(&self) -> usize {
        self.ops.len()
    }

    /// One more than the highest qubit acted on, or 0 for the identity.
    pub fn min_qubits(&self) -> usize {
        self.ops.last().map_or(0, |&(q, _)| q + 1)
    }

    /// The qubits acted on, in increasing order.
    pub fn qubits(&self) -> impl Iterator<Item = usize> + '_ {
        self.ops.iter().map(|&(q, _)| q)
    }

    /// The same string with each qubit `q` moved to `map[q]`.
    pub fn remap(&self, map: &[usize]) -> Result<Self, Error> {
        let ops = self.ops.iter().map(|&(q, p)| {
            map.get(q).map(|&m| (m, p)).ok_or(Error::QubitOutOfRange {
                qubit: q,
                n_qubits: map.len(),
            })
        });
        Self::new(ops.collect::<Result<Vec<_>, _>>()?)
    }

    /// Bitmasks describing the string's action on basis states, for strings
    /// on fewer than `usize::BITS` qubits: `P|i⟩ = iⁿʸ (-1)^|i ∧ z| |i ⊕ x⟩`.
    pub(crate) fn masks(&self) -> PauliMasks {
        let mut m = PauliMasks::default();
        for &(q, p) in &self.ops {
            let bit = 1 << q;
            match p {
                Pauli::I => {}
                Pauli::X => m.x |= bit,
                Pauli::Y => {
                    m.x |= bit;
                    m.z |= bit;
                    m.n_y += 1;
                }
                Pauli::Z => m.z |= bit,
            }
        }
        m
    }
}

/// See [`PauliString::masks`]. `Y = iXZ`, so a Y sets both masks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct PauliMasks {
    pub x: usize,
    pub z: usize,
    pub n_y: usize,
}

impl PauliMasks {
    /// `iⁿʸ`.
    pub fn y_phase(&self) -> num_complex::Complex64 {
        use num_complex::Complex64 as C64;
        [
            C64::new(1.0, 0.0),
            C64::new(0.0, 1.0),
            C64::new(-1.0, 0.0),
            C64::new(0.0, -1.0),
        ][self.n_y % 4]
    }

    /// `(-1)^|i ∧ z|`, as a boolean: true when negative.
    #[inline]
    pub fn z_sign(&self, i: usize) -> bool {
        (i & self.z).count_ones() % 2 == 1
    }
}

/// A real linear combination of Pauli strings, e.g. a Hamiltonian.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PauliSum {
    terms: Vec<(f64, PauliString)>,
}

impl PauliSum {
    /// The zero operator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build from `(label, coefficient)` pairs, like Qiskit's
    /// `SparsePauliOp.from_list`.
    pub fn from_list<'a>(terms: impl IntoIterator<Item = (&'a str, f64)>) -> Result<Self, Error> {
        let mut sum = Self::new();
        for (label, coeff) in terms {
            sum.add(coeff, PauliString::from_label(label)?);
        }
        Ok(sum)
    }

    /// Add `coeff · string`. Terms are not merged.
    pub fn add(&mut self, coeff: f64, string: PauliString) -> &mut Self {
        self.terms.push((coeff, string));
        self
    }

    /// The `(coefficient, string)` terms.
    pub fn terms(&self) -> &[(f64, PauliString)] {
        &self.terms
    }

    /// `Σ |cₖ|`, an upper bound on the operator's spectral norm.
    pub fn coefficient_norm(&self) -> f64 {
        self.terms.iter().map(|(c, _)| c.abs()).sum()
    }

    /// Merge terms with the same Pauli string, keeping first-seen order, and
    /// drop terms whose coefficient magnitude is at most `tolerance`.
    pub fn simplify(&self, tolerance: f64) -> PauliSum {
        let mut index: std::collections::HashMap<&PauliString, usize> =
            std::collections::HashMap::new();
        let mut terms: Vec<(f64, PauliString)> = Vec::new();
        for (coeff, string) in &self.terms {
            match index.get(string) {
                Some(&i) => terms[i].0 += coeff,
                None => {
                    index.insert(string, terms.len());
                    terms.push((*coeff, string.clone()));
                }
            }
        }
        terms.retain(|(c, _)| c.abs() > tolerance);
        PauliSum { terms }
    }

    /// One more than the highest qubit any term acts on.
    pub fn min_qubits(&self) -> usize {
        self.terms
            .iter()
            .map(|(_, s)| s.min_qubits())
            .max()
            .unwrap_or(0)
    }
}

impl From<PauliString> for PauliSum {
    fn from(string: PauliString) -> Self {
        PauliSum {
            terms: vec![(1.0, string)],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn label_rightmost_is_qubit_zero() {
        let s = PauliString::from_label("XIZ").unwrap();
        assert_eq!(s.ops(), &[(0, Pauli::Z), (2, Pauli::X)]);
        assert_eq!(s.min_qubits(), 3);
    }

    #[test]
    fn simplify_merges_and_drops() {
        let h =
            PauliSum::from_list([("ZZ", 1.0), ("XI", 0.5), ("ZZ", -1.0), ("XI", 0.25)]).unwrap();
        let s = h.simplify(1e-12);
        assert_eq!(s.terms(), &[(0.75, PauliString::from_label("XI").unwrap())]);
    }

    #[test]
    fn masks_encode_action() {
        let m = PauliString::from_label("YXZ").unwrap().masks();
        assert_eq!((m.x, m.z, m.n_y), (0b110, 0b101, 1));
    }

    #[test]
    fn rejects_bad_input() {
        assert_eq!(PauliString::from_label("XA"), Err(Error::InvalidPauli('A')));
        assert_eq!(
            PauliString::new([(1, Pauli::X), (1, Pauli::Z)]),
            Err(Error::DuplicateQubit(1))
        );
    }

    #[test]
    fn identity_factors_are_dropped() {
        let s = PauliString::new([(3, Pauli::I), (1, Pauli::Y)]).unwrap();
        assert_eq!(s.ops(), &[(1, Pauli::Y)]);
        assert_eq!(
            PauliString::from_label("III").unwrap(),
            PauliString::identity()
        );
    }
}
