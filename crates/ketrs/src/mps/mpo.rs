//! Matrix product operators (MPOs) for Pauli sums.
//!
//! An MPO writes `H = Σₜ cₜ Pₜ` as a chain of small operator-valued
//! matrices, built here as a finite automaton read left to right. At each
//! cut between sites the automaton is either before every term (`start`),
//! after a finished term (`done`), or partway through some term, in a state
//! named by the operators that term still has to place to the right of the
//! cut. Terms with the same remaining operators share a state, which keeps
//! the bond small: nearest-neighbour Hamiltonians need a handful of states,
//! and all-to-all two-body terms `Σ cᵢⱼ ZᵢZⱼ` need O(n) rather than O(n²).

use crate::error::Error;
use crate::observable::{Pauli, PauliSum};
use num_complex::Complex64 as C64;
use std::collections::HashMap;

/// One site of an MPO: `W[a][b] = Σ coeff · P` over its entries.
#[derive(Clone, Debug)]
pub(crate) struct MpoSite {
    pub left: usize,
    pub right: usize,
    /// `(a, b, P, coeff)`; several entries may share `(a, b)`.
    pub entries: Vec<(usize, usize, Pauli, C64)>,
}

/// A Pauli sum as an MPO, with its identity terms kept as a constant.
#[derive(Clone, Debug)]
pub(crate) struct Mpo {
    pub sites: Vec<MpoSite>,
    /// The coefficient of the identity, added to energies separately.
    pub shift: f64,
}

const START: usize = 0;
const DONE: usize = 1;

/// The automaton state of a term at a cut: the operators it still has to
/// place to the right.
type Suffix = Vec<(usize, Pauli)>;

impl Mpo {
    pub(crate) fn new(hamiltonian: &PauliSum, n_qubits: usize) -> Result<Mpo, Error> {
        if hamiltonian.min_qubits() > n_qubits {
            return Err(Error::QubitOutOfRange {
                qubit: hamiltonian.min_qubits() - 1,
                n_qubits,
            });
        }
        let h = hamiltonian.simplify(0.0);
        let mut shift = 0.0;
        let mut terms = Vec::new();
        for (c, s) in h.terms() {
            if s.weight() == 0 {
                shift += c;
            } else {
                terms.push((*c, s.ops().to_vec()));
            }
        }

        // States at each cut c (between sites c and c+1): START, DONE, then
        // one per distinct suffix of a term crossing the cut.
        let mut states: Vec<HashMap<Suffix, usize>> =
            vec![HashMap::new(); n_qubits.saturating_sub(1)];
        let suffix = |ops: &[(usize, Pauli)], cut: usize| -> Suffix {
            ops.iter().filter(|&&(q, _)| q > cut).copied().collect()
        };
        for (_, ops) in &terms {
            let (first, last) = (ops[0].0, ops[ops.len() - 1].0);
            for (cut, map) in states.iter_mut().enumerate().take(last).skip(first) {
                let next = map.len() + 2;
                map.entry(suffix(ops, cut)).or_insert(next);
            }
        }
        let width = |cut: usize| states[cut].len() + 2;

        let mut sites = Vec::with_capacity(n_qubits);
        for i in 0..n_qubits {
            // Boundaries keep only the start (left end) and done (right end)
            // states, both renumbered to 0.
            let left = if i == 0 { 1 } else { width(i - 1) };
            let right = if i + 1 == n_qubits { 1 } else { width(i) };
            let left_index = |state: usize| {
                if i == 0 {
                    (state == START).then_some(0)
                } else {
                    Some(state)
                }
            };
            let right_index = |state: usize| {
                if i + 1 == n_qubits {
                    (state == DONE).then_some(0)
                } else {
                    Some(state)
                }
            };
            let mut entries = Vec::new();
            let mut push = |a: usize, b: usize, p: Pauli, c: C64| {
                if let (Some(a), Some(b)) = (left_index(a), right_index(b)) {
                    entries.push((a, b, p, c));
                }
            };
            push(START, START, Pauli::I, C64::new(1.0, 0.0));
            push(DONE, DONE, Pauli::I, C64::new(1.0, 0.0));

            // Transitions partway through a term are shared by every term
            // with the same suffix, so add each once.
            let mut shared = std::collections::HashSet::new();
            for (c, ops) in &terms {
                let (first, last) = (ops[0].0, ops[ops.len() - 1].0);
                if i < first || i > last {
                    continue;
                }
                let op = ops
                    .iter()
                    .find(|&&(q, _)| q == i)
                    .map_or(Pauli::I, |&(_, p)| p);
                let from = if i == first {
                    START
                } else {
                    states[i - 1][&suffix(ops, i - 1)]
                };
                let to = if i == last {
                    DONE
                } else {
                    states[i][&suffix(ops, i)]
                };
                if i == first {
                    push(from, to, op, C64::new(*c, 0.0));
                } else if shared.insert(from) {
                    push(from, to, op, C64::new(1.0, 0.0));
                }
            }
            sites.push(MpoSite {
                left,
                right,
                entries,
            });
        }
        Ok(Mpo { sites, shift })
    }

    /// Bond dimension between each pair of neighbouring sites.
    #[cfg(test)]
    pub(crate) fn bond_dimensions(&self) -> Vec<usize> {
        self.sites[..self.sites.len() - 1]
            .iter()
            .map(|s| s.right)
            .collect()
    }
}

/// The single non-zero entry in row `p_out` of a Pauli matrix:
/// `(column, value)`.
#[inline]
pub(crate) fn pauli_entry(p: Pauli, p_out: usize) -> (usize, C64) {
    match p {
        Pauli::I => (p_out, C64::new(1.0, 0.0)),
        Pauli::X => (1 - p_out, C64::new(1.0, 0.0)),
        Pauli::Y => {
            if p_out == 0 {
                (1, C64::new(0.0, -1.0))
            } else {
                (0, C64::new(0.0, 1.0))
            }
        }
        Pauli::Z => (p_out, C64::new(if p_out == 0 { 1.0 } else { -1.0 }, 0.0)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observable::PauliString;

    /// The MPO contracted into a dense matrix, `⟨i|H|j⟩`.
    fn dense(mpo: &Mpo) -> Vec<Vec<C64>> {
        let n = mpo.sites.len();
        let dim = 1 << n;
        let mut m = vec![vec![C64::new(0.0, 0.0); dim]; dim];
        for (r, row) in m.iter_mut().enumerate() {
            for (c, entry) in row.iter_mut().enumerate() {
                // Row vector over bond states, multiplied site by site.
                let mut v = vec![C64::new(1.0, 0.0)];
                for (q, site) in mpo.sites.iter().enumerate() {
                    let mut next = vec![C64::new(0.0, 0.0); site.right];
                    let (p_out, p_in) = ((r >> q) & 1, (c >> q) & 1);
                    for &(a, b, p, coeff) in &site.entries {
                        let (col, val) = pauli_entry(p, p_out);
                        if col == p_in {
                            next[b] += v[a] * coeff * val;
                        }
                    }
                    v = next;
                }
                let shift = if r == c { mpo.shift } else { 0.0 };
                *entry = v[0] + shift;
            }
        }
        m
    }

    #[test]
    fn mpo_reproduces_the_pauli_sum() {
        let n = 5;
        let mut h = PauliSum::from_list([
            ("ZZIII", 1.0),
            ("IXIXI", -0.5),
            ("YIIIY", 0.3),
            ("IIZII", 0.7),
            ("XYZXY", 0.2),
            ("IIIII", 1.5),
            ("IIIXX", -0.4),
        ])
        .unwrap();
        h.add(0.1, PauliString::from_label("IXIXI").unwrap());
        let mpo = Mpo::new(&h, n).unwrap();
        let got = dense(&mpo);
        // Reference: the sum of the terms' matrices.
        for (r, got_row) in got.iter().enumerate() {
            for (c, &got_entry) in got_row.iter().enumerate() {
                let mut want = C64::new(0.0, 0.0);
                for (coeff, s) in h.terms() {
                    let mut v = C64::new(*coeff, 0.0);
                    for q in 0..n {
                        let p = s
                            .ops()
                            .iter()
                            .find(|&&(k, _)| k == q)
                            .map_or(Pauli::I, |&(_, p)| p);
                        let (col, val) = pauli_entry(p, (r >> q) & 1);
                        v *= if col == (c >> q) & 1 {
                            val
                        } else {
                            C64::new(0.0, 0.0)
                        };
                    }
                    want += v;
                }
                assert!(
                    (got_entry - want).norm() < 1e-12,
                    "({r}, {c}): {} vs {want}",
                    got_entry
                );
            }
        }
    }

    #[test]
    fn long_range_two_body_terms_need_linear_bond() {
        let n = 12;
        let mut h = PauliSum::new();
        for i in 0..n {
            for j in i + 1..n {
                h.add(
                    1.0 / (j - i) as f64,
                    PauliString::new([(i, Pauli::Z), (j, Pauli::Z)]).unwrap(),
                );
            }
        }
        let bonds = Mpo::new(&h, n).unwrap().bond_dimensions();
        // start, done, and one state per site still to the right.
        assert_eq!(bonds.iter().max(), Some(&(2 + n - 1)));
    }
}
