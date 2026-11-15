//! Abelian charges for symmetric MPS, and the sector tables of their bonds.
//!
//! A charge gives each qubit state a value, `c_q(0)` and `c_q(1)`, and a
//! basis state the sum over its qubits: a U(1) charge in the integers, a
//! Z_n charge modulo `n`. Several charges combine into a label, a tuple
//! with one entry per charge. Each bond of a symmetric MPS has a basis
//! sorted by the label of the qubits to its left, so its basis splits into
//! contiguous sectors, and a site tensor `A[s, p, t]` is zero unless
//! `t = s + c(p)`.

use crate::error::Error;
use std::ops::Range;

/// A conserved additive charge: U(1) or Z_n, with a value for each qubit
/// state. A basis state's charge is the sum of its qubits' values (modulo
/// `n` for Z_n).
///
/// Particle number is `Charge::weights(vec![1; n])`, the parity of the
/// same is `Charge::modulo(2, vec![[0, 1]; n])`, and `2 S^z` is
/// `Charge::u1(vec![[1, -1]; n])`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Charge {
    modulus: Option<u32>,
    values: Vec<[i64; 2]>,
}

impl Charge {
    /// A U(1) charge with `values[q] = [c(|0⟩), c(|1⟩)]` for qubit `q`.
    pub fn u1(values: Vec<[i64; 2]>) -> Charge {
        Charge {
            modulus: None,
            values,
        }
    }

    /// A Z_n charge (`n ≥ 1`), values as for [`u1`](Self::u1), added
    /// modulo `n`.
    pub fn modulo(n: u32, values: Vec<[i64; 2]>) -> Charge {
        Charge {
            modulus: Some(n),
            values,
        }
    }

    /// A U(1) charge that counts `weights[q]` for each qubit `q` in `|1⟩`:
    /// values `[0, weights[q]]`.
    pub fn weights(weights: Vec<i64>) -> Charge {
        Charge::u1(weights.into_iter().map(|w| [0, w]).collect())
    }

    /// `Some(n)` for a Z_n charge, `None` for U(1).
    pub fn modulus(&self) -> Option<u32> {
        self.modulus
    }

    /// `[c(|0⟩), c(|1⟩)]` for each qubit.
    pub fn values(&self) -> &[[i64; 2]] {
        &self.values
    }

    /// The charge of a basis state, one bit per qubit.
    pub fn of(&self, bits: &[bool]) -> i64 {
        let sum = self
            .values
            .iter()
            .zip(bits)
            .map(|(v, &b)| v[b as usize])
            .sum();
        self.reduce(sum)
    }

    fn reduce(&self, v: i64) -> i64 {
        match self.modulus {
            Some(n) => v.rem_euclid(i64::from(n)),
            None => v,
        }
    }
}

/// A tuple of charge values, one per charge.
pub(super) type Label = Box<[i64]>;

/// Several charges, checked against a chain and held as labels per qubit
/// state.
#[derive(Clone, Debug)]
pub(super) struct Charges {
    charges: Vec<Charge>,
    /// The label of each qubit state, `site[q][p]`.
    site: Vec<[Label; 2]>,
}

impl Charges {
    pub(super) fn new(charges: &[Charge], n_qubits: usize) -> Result<Charges, Error> {
        for (k, c) in charges.iter().enumerate() {
            if c.values.len() != n_qubits {
                return Err(Error::InvalidArgument(format!(
                    "charge {k} has values for {} qubits, but the state has {n_qubits}",
                    c.values.len()
                )));
            }
            if c.modulus == Some(0) {
                return Err(Error::InvalidArgument(format!(
                    "charge {k} is modulo 0; Z_n needs n ≥ 1"
                )));
            }
        }
        let site = (0..n_qubits)
            .map(|q| {
                let label = |p: usize| charges.iter().map(|c| c.reduce(c.values[q][p])).collect();
                [label(0), label(1)]
            })
            .collect();
        Ok(Charges {
            charges: charges.to_vec(),
            site,
        })
    }

    pub(super) fn charges(&self) -> &[Charge] {
        &self.charges
    }

    /// The label of qubit `q` in state `p`.
    pub(super) fn site(&self, q: usize, p: usize) -> &[i64] {
        &self.site[q][p]
    }

    /// Whether some charge tells qubit `q`'s two states apart.
    pub(super) fn distinguishes(&self, q: usize) -> bool {
        self.site[q][0] != self.site[q][1]
    }

    pub(super) fn zero(&self) -> Label {
        vec![0; self.charges.len()].into()
    }

    pub(super) fn add(&self, a: &[i64], b: &[i64]) -> Label {
        self.combine(a, b, |x, y| x + y)
    }

    pub(super) fn sub(&self, a: &[i64], b: &[i64]) -> Label {
        self.combine(a, b, |x, y| x - y)
    }

    fn combine(&self, a: &[i64], b: &[i64], f: impl Fn(i64, i64) -> i64) -> Label {
        self.charges
            .iter()
            .zip(a.iter().zip(b))
            .map(|(c, (&x, &y))| c.reduce(f(x, y)))
            .collect()
    }
}

/// The sectors of one bond: labels in increasing order, each with a
/// contiguous range of the bond's basis.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Sectors {
    labels: Vec<Label>,
    /// `offsets[k] .. offsets[k + 1]` is sector `k`; one more than labels.
    offsets: Vec<usize>,
}

impl Sectors {
    /// Sectors from `(label, dimension)` pairs with distinct labels, in any
    /// order. Empty sectors are left out.
    pub(super) fn new(mut entries: Vec<(Label, usize)>) -> Sectors {
        entries.retain(|(_, d)| *d > 0);
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        debug_assert!(entries.windows(2).all(|w| w[0].0 != w[1].0));
        let mut offsets = Vec::with_capacity(entries.len() + 1);
        offsets.push(0);
        let mut labels = Vec::with_capacity(entries.len());
        for (label, d) in entries {
            offsets.push(offsets.last().unwrap() + d);
            labels.push(label);
        }
        Sectors { labels, offsets }
    }

    /// A one-dimensional bond with a single label.
    pub(super) fn single(label: Label) -> Sectors {
        Sectors::new(vec![(label, 1)])
    }

    pub(super) fn len(&self) -> usize {
        self.labels.len()
    }

    /// The bond dimension.
    pub(super) fn dim(&self) -> usize {
        *self.offsets.last().unwrap()
    }

    pub(super) fn label(&self, k: usize) -> &[i64] {
        &self.labels[k]
    }

    pub(super) fn range(&self, k: usize) -> Range<usize> {
        self.offsets[k]..self.offsets[k + 1]
    }

    pub(super) fn size(&self, k: usize) -> usize {
        self.offsets[k + 1] - self.offsets[k]
    }

    /// The sector with `label`, if the bond has one.
    pub(super) fn find(&self, label: &[i64]) -> Option<usize> {
        self.labels.binary_search_by(|l| (**l).cmp(label)).ok()
    }
}

/// Row or column indices of a matrix grouped by label: `(label, indices)`
/// with labels increasing and indices increasing within each group.
pub(super) type Groups = Vec<(Label, Vec<usize>)>;

/// Indices `l · 2 + p` (rows of a site's left matrix, or of a two-site
/// tensor) grouped by the label `s(l) + c(p)` of the bond to their right.
pub(super) fn group_rows(left: &Sectors, charges: &Charges, q: usize) -> Groups {
    let mut groups: std::collections::BTreeMap<Label, Vec<usize>> = Default::default();
    for k in 0..left.len() {
        for p in 0..2 {
            let label = charges.add(left.label(k), charges.site(q, p));
            let indices = groups.entry(label).or_default();
            indices.extend(left.range(k).map(|l| l * 2 + p));
        }
    }
    sorted(groups)
}

/// Indices `p · dim + r` (columns of a site's right matrix, or of a
/// two-site tensor) grouped by the label `t(r) − c(p)` of the bond to their
/// left.
pub(super) fn group_cols(right: &Sectors, charges: &Charges, q: usize) -> Groups {
    let mut groups: std::collections::BTreeMap<Label, Vec<usize>> = Default::default();
    let dim = right.dim();
    for k in 0..right.len() {
        for p in 0..2 {
            let label = charges.sub(right.label(k), charges.site(q, p));
            let indices = groups.entry(label).or_default();
            indices.extend(right.range(k).map(|r| p * dim + r));
        }
    }
    sorted(groups)
}

/// The indices of each sector of `bonds`, as groups.
pub(super) fn group_bond(bond: &Sectors) -> Groups {
    (0..bond.len())
        .map(|k| (bond.labels[k].clone(), bond.range(k).collect()))
        .collect()
}

fn sorted(groups: std::collections::BTreeMap<Label, Vec<usize>>) -> Groups {
    groups
        .into_iter()
        .map(|(label, mut indices)| {
            indices.sort_unstable();
            (label, indices)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_add_in_their_groups() {
        let charges = Charges::new(
            &[
                Charge::weights(vec![1, 2]),
                Charge::modulo(3, vec![[0, 2], [1, 2]]),
            ],
            2,
        )
        .unwrap();
        assert_eq!(charges.site(1, 1), &[2, 2]);
        assert_eq!(&*charges.add(&[1, 2], &[2, 2]), &[3, 1]);
        assert_eq!(&*charges.sub(&[1, 0], &[2, 2]), &[-1, 1]);
        assert_eq!(Charge::modulo(3, vec![[0, 2], [1, 2]]).of(&[true, true]), 1);
        assert!(Charges::new(&[Charge::weights(vec![1])], 2).is_err());
        assert!(Charges::new(&[Charge::modulo(0, vec![[0, 1]])], 1).is_err());
    }

    #[test]
    fn sectors_sort_and_find_their_labels() {
        let s = Sectors::new(vec![
            (vec![2].into(), 3),
            (vec![-1].into(), 2),
            (vec![5].into(), 0),
        ]);
        assert_eq!(s.len(), 2);
        assert_eq!(s.dim(), 5);
        assert_eq!(s.label(0), &[-1]);
        assert_eq!(s.range(1), 2..5);
        assert_eq!(s.find(&[2]), Some(1));
        assert_eq!(s.find(&[5]), None);
    }
}
