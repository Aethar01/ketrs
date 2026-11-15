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

use super::charges::{Charges, Label};
use crate::error::Error;
use crate::observable::{Pauli, PauliSum};
use num_complex::Complex64 as C64;
use std::collections::{HashMap, HashSet};

/// A single-qubit operator in an MPO: a Pauli, or one of the ladder
/// operators that a charge-graded MPO writes `X` and `Y` in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum Local {
    I,
    X,
    Y,
    Z,
    /// `|0⟩⟨1|`.
    Lower,
    /// `|1⟩⟨0|`.
    Raise,
}

impl From<Pauli> for Local {
    fn from(p: Pauli) -> Local {
        match p {
            Pauli::I => Local::I,
            Pauli::X => Local::X,
            Pauli::Y => Local::Y,
            Pauli::Z => Local::Z,
        }
    }
}

/// One site of an MPO: `W[a][b] = Σ coeff · O` over its entries.
#[derive(Clone, Debug)]
pub(crate) struct MpoSite {
    pub left: usize,
    pub right: usize,
    /// `(a, b, O, coeff)`; several entries may share `(a, b)`.
    pub entries: Vec<(usize, usize, Local, C64)>,
}

/// A Pauli sum as an MPO, with its identity terms kept as a constant.
#[derive(Clone, Debug)]
pub(crate) struct Mpo {
    pub sites: Vec<MpoSite>,
    /// The coefficient of the identity, added to energies separately.
    pub shift: f64,
    /// For an MPO built for charges, the charge each channel still has to
    /// change by to its right, per cut: `channels[i]` for the bond left of
    /// site `i`, so `channels[0]` and `channels[n]` are the ends.
    pub channels: Option<Vec<Vec<Label>>>,
}

const START: usize = 0;
const DONE: usize = 1;

/// The automaton state of a term at a cut: the operators it still has to
/// place to the right.
type Suffix = Vec<(usize, Local)>;

/// A product of single-qubit operators on increasing qubits, with its
/// coefficient.
type Term = (C64, Vec<(usize, Local)>);

/// Most `X` and `Y` factors a term may have on qubits a charge sees: each
/// splits into two ladder operators, so a term becomes `2^w` strings.
const MAX_SPLIT: usize = 16;

impl Mpo {
    /// The MPO of `hamiltonian` on `n_qubits` qubits, over Paulis.
    pub(crate) fn new(hamiltonian: &PauliSum, n_qubits: usize) -> Result<Mpo, Error> {
        let (shift, terms) = pauli_terms(hamiltonian, n_qubits)?;
        let terms: Vec<Term> = terms
            .into_iter()
            .map(|(c, ops)| {
                let ops = ops.into_iter().map(|(q, p)| (q, Local::from(p))).collect();
                (C64::new(c, 0.0), ops)
            })
            .collect();
        Ok(Mpo::from_terms(&terms, n_qubits, shift, None))
    }

    /// The MPO of `hamiltonian` graded by `charges`: on qubits a charge
    /// sees, `X` and `Y` become ladder operators, so each channel changes
    /// the charges by a definite amount. Errors, naming terms, if the
    /// Hamiltonian doesn't conserve the charges.
    pub(super) fn with_charges(
        hamiltonian: &PauliSum,
        n_qubits: usize,
        charges: &Charges,
    ) -> Result<Mpo, Error> {
        let (shift, terms) = pauli_terms(hamiltonian, n_qubits)?;
        let scale = terms.iter().map(|t| t.0.abs()).fold(0.0, f64::max);
        // Elementary strings, with the input terms that gave each.
        let mut strings: HashMap<Suffix, (C64, Vec<usize>)> = HashMap::new();
        for (index, (c, ops)) in terms.iter().enumerate() {
            let split = ops
                .iter()
                .filter(|&&(q, p)| matches!(p, Pauli::X | Pauli::Y) && charges.distinguishes(q))
                .count();
            if split > MAX_SPLIT {
                return Err(Error::InvalidArgument(format!(
                    "term {} has {split} X or Y factors on charged qubits, more than {MAX_SPLIT}",
                    describe(ops)
                )));
            }
            let mut expanded: Vec<Term> = vec![(C64::new(*c, 0.0), Vec::with_capacity(ops.len()))];
            for &(q, p) in ops {
                let ladder = match p {
                    Pauli::X if charges.distinguishes(q) => {
                        [C64::new(1.0, 0.0), C64::new(1.0, 0.0)]
                    }
                    Pauli::Y if charges.distinguishes(q) => {
                        [C64::new(0.0, -1.0), C64::new(0.0, 1.0)]
                    }
                    _ => {
                        expanded.iter_mut().for_each(|t| t.1.push((q, p.into())));
                        continue;
                    }
                };
                expanded = expanded
                    .into_iter()
                    .flat_map(|(c, ops)| {
                        [(Local::Lower, ladder[0]), (Local::Raise, ladder[1])].map(|(op, f)| {
                            let mut ops = ops.clone();
                            ops.push((q, op));
                            (c * f, ops)
                        })
                    })
                    .collect();
            }
            for (c, ops) in expanded {
                let entry = strings
                    .entry(ops)
                    .or_insert((C64::new(0.0, 0.0), Vec::new()));
                entry.0 += c;
                entry.1.push(index);
            }
        }
        let tolerance = 1e-12 * scale;
        let mut kept: Vec<Term> = Vec::new();
        let mut broken = Vec::new();
        for (ops, (c, sources)) in strings {
            if c.norm() <= tolerance {
                continue;
            }
            let change = charge_change(&ops, charges);
            if change.iter().any(|&x| x != 0) {
                broken.push((change, sources));
            } else {
                kept.push((c, ops));
            }
        }
        if !broken.is_empty() {
            broken.sort();
            let examples: Vec<String> = broken
                .iter()
                .take(3)
                .map(|(change, sources)| {
                    let names: Vec<String> = sources
                        .iter()
                        .take(4)
                        .map(|&k| describe(&terms[k].1))
                        .collect();
                    format!("{} change the charges by {change:?}", names.join(" + "))
                })
                .collect();
            return Err(Error::BreaksSymmetry(format!(
                "the Hamiltonian doesn't conserve the charges: {} part(s) don't, e.g. {}",
                broken.len(),
                examples.join("; ")
            )));
        }
        // A deterministic order, so the MPO doesn't depend on hashing.
        kept.sort_by(|a, b| a.1.cmp(&b.1));
        Ok(Mpo::from_terms(&kept, n_qubits, shift, Some(charges)))
    }

    /// The automaton MPO of `terms`, each a product on increasing qubits,
    /// with the charge of each channel if `charges` are given.
    fn from_terms(terms: &[Term], n_qubits: usize, shift: f64, charges: Option<&Charges>) -> Mpo {
        // States at each cut c (between sites c and c+1): START, DONE, then
        // one per distinct suffix of a term crossing the cut.
        let mut states: Vec<HashMap<Suffix, usize>> =
            vec![HashMap::new(); n_qubits.saturating_sub(1)];
        for (_, ops) in terms {
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
            let mut push = |a: usize, b: usize, op: Local, c: C64| {
                if let (Some(a), Some(b)) = (left_index(a), right_index(b)) {
                    entries.push((a, b, op, c));
                }
            };
            push(START, START, Local::I, C64::new(1.0, 0.0));
            push(DONE, DONE, Local::I, C64::new(1.0, 0.0));

            // Transitions partway through a term are shared by every term
            // with the same suffix, so add each once.
            let mut shared = HashSet::new();
            for (c, ops) in terms {
                let (first, last) = (ops[0].0, ops[ops.len() - 1].0);
                if i < first || i > last {
                    continue;
                }
                let op = ops
                    .iter()
                    .find(|&&(q, _)| q == i)
                    .map_or(Local::I, |&(_, p)| p);
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
                    push(from, to, op, *c);
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
        // START and DONE change nothing more; a state partway through a
        // term changes what its suffix does.
        let channels = charges.map(|charges| {
            let mut cuts = vec![vec![charges.zero()]];
            for i in 0..n_qubits {
                let mut labels = vec![charges.zero(); sites[i].right];
                if i + 1 < n_qubits {
                    for (s, &k) in &states[i] {
                        labels[k] = charge_change(s, charges);
                    }
                }
                cuts.push(labels);
            }
            cuts
        });
        Mpo {
            sites,
            shift,
            channels,
        }
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

/// The identity coefficient and the other terms of `hamiltonian`, as
/// `(coefficient, ops)` with ops on increasing qubits.
#[allow(clippy::type_complexity)]
fn pauli_terms(
    hamiltonian: &PauliSum,
    n_qubits: usize,
) -> Result<(f64, Vec<(f64, Vec<(usize, Pauli)>)>), Error> {
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
    Ok((shift, terms))
}

/// The operators of `ops` on qubits after `cut`.
fn suffix(ops: &[(usize, Local)], cut: usize) -> Suffix {
    ops.iter().filter(|&&(q, _)| q > cut).copied().collect()
}

/// A Pauli product as text, such as `X0 Z1 X2`.
fn describe(ops: &[(usize, Pauli)]) -> String {
    let parts: Vec<String> = ops.iter().map(|(q, p)| format!("{p:?}{q}")).collect();
    parts.join(" ")
}

/// How much `ops` changes the charges of a state it acts on.
fn charge_change(ops: &[(usize, Local)], charges: &Charges) -> Label {
    let mut change = charges.zero();
    for &(q, op) in ops {
        let (from, to) = match op {
            Local::Lower => (1, 0),
            Local::Raise => (0, 1),
            _ => continue,
        };
        let step = charges.sub(charges.site(q, to), charges.site(q, from));
        change = charges.add(&change, &step);
    }
    change
}

/// A sparse mixing table for the effective Hamiltonians: for each output
/// key `block · phys + o`, the `(input block, input physical index,
/// factor)` terms.
pub(super) type Rows = Vec<Vec<(usize, usize, C64)>>;

/// For a single MPO site, keyed by its right channel and output physical
/// index: `out[b][p_out] = Σ W[a][b] … in[a][p_in]`.
pub(super) fn right_keyed(w: &MpoSite) -> Rows {
    let mut rows = vec![Vec::new(); w.right * 2];
    for &(a, b, op, coeff) in &w.entries {
        for p_out in 0..2 {
            if let Some((p_in, v)) = op_entry(op, p_out) {
                rows[b * 2 + p_out].push((a, p_in, coeff * v));
            }
        }
    }
    rows
}

/// The mirror of [`right_keyed`], keyed by the left channel.
pub(super) fn left_keyed(w: &MpoSite) -> Rows {
    let mut rows = vec![Vec::new(); w.left * 2];
    for &(a, b, op, coeff) in &w.entries {
        for p_out in 0..2 {
            if let Some((p_in, v)) = op_entry(op, p_out) {
                rows[a * 2 + p_out].push((b, p_in, coeff * v));
            }
        }
    }
    rows
}

/// Two neighbouring MPO sites contracted over their shared bond, keyed by
/// the right channel of the second and the output pair `2 p_out + q_out`.
pub(super) fn pair_keyed(w1: &MpoSite, w2: &MpoSite) -> Rows {
    let mut by_left = vec![Vec::new(); w2.left];
    for &(b, c, op, coeff) in &w2.entries {
        by_left[b].push((c, op, coeff));
    }
    let mut rows = vec![Vec::new(); w2.right * 4];
    for &(a, b, op1, c1) in &w1.entries {
        for &(c, op2, c2) in &by_left[b] {
            for p_out in 0..2 {
                let Some((p_in, v1)) = op_entry(op1, p_out) else {
                    continue;
                };
                for q_out in 0..2 {
                    if let Some((q_in, v2)) = op_entry(op2, q_out) {
                        let key = c * 4 + 2 * p_out + q_out;
                        rows[key].push((a, 2 * p_in + q_in, c1 * c2 * v1 * v2));
                    }
                }
            }
        }
    }
    rows
}

/// The non-zero entry in row `p_out` of a single-qubit operator, as
/// `(column, value)`, if the row has one.
#[inline]
pub(crate) fn op_entry(op: Local, p_out: usize) -> Option<(usize, C64)> {
    let one = C64::new(1.0, 0.0);
    match op {
        Local::I => Some((p_out, one)),
        Local::X => Some((1 - p_out, one)),
        Local::Y => Some(if p_out == 0 {
            (1, C64::new(0.0, -1.0))
        } else {
            (0, C64::new(0.0, 1.0))
        }),
        Local::Z => Some((p_out, C64::new(if p_out == 0 { 1.0 } else { -1.0 }, 0.0))),
        Local::Lower => (p_out == 0).then_some((1, one)),
        Local::Raise => (p_out == 1).then_some((0, one)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mps::charges::Charge;
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
                    for &(a, b, op, coeff) in &site.entries {
                        if let Some((col, val)) = op_entry(op, p_out)
                            && col == p_in
                        {
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

    /// The sum of the terms' matrices, `⟨i|H|j⟩`.
    fn reference(h: &PauliSum, n: usize) -> Vec<Vec<C64>> {
        let dim = 1 << n;
        let mut m = vec![vec![C64::new(0.0, 0.0); dim]; dim];
        for (r, row) in m.iter_mut().enumerate() {
            for (c, entry) in row.iter_mut().enumerate() {
                for (coeff, s) in h.terms() {
                    let mut v = C64::new(*coeff, 0.0);
                    for q in 0..n {
                        let p = s
                            .ops()
                            .iter()
                            .find(|&&(k, _)| k == q)
                            .map_or(Pauli::I, |&(_, p)| p);
                        let (col, val) = op_entry(p.into(), (r >> q) & 1).unwrap();
                        v *= if col == (c >> q) & 1 {
                            val
                        } else {
                            C64::new(0.0, 0.0)
                        };
                    }
                    *entry += v;
                }
            }
        }
        m
    }

    fn assert_same(got: &[Vec<C64>], want: &[Vec<C64>]) {
        for (r, (g, w)) in got.iter().zip(want).enumerate() {
            for (c, (g, w)) in g.iter().zip(w).enumerate() {
                assert!((g - w).norm() < 1e-12, "({r}, {c}): {g} vs {w}");
            }
        }
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
        assert_same(&dense(&mpo), &reference(&h, n));
    }

    /// Number-conserving hops (`XX + YY`, and the current `XY − YX`, also
    /// across a string of `Z`s), interactions and fields.
    fn conserving(n: usize) -> PauliSum {
        let mut h = PauliSum::new();
        let pair = |i: usize, j: usize, a: char, b: char, mid: char| -> PauliString {
            let label: String = (0..n)
                .rev()
                .map(|k| match k {
                    k if k == i => a,
                    k if k == j => b,
                    k if k > i && k < j => mid,
                    _ => 'I',
                })
                .collect();
            PauliString::from_label(&label).unwrap()
        };
        for i in 0..n - 1 {
            h.add(0.5 + 0.1 * i as f64, pair(i, i + 1, 'X', 'X', 'I'));
            h.add(0.5 + 0.1 * i as f64, pair(i, i + 1, 'Y', 'Y', 'I'));
            h.add(0.3, pair(i, i + 1, 'Z', 'Z', 'I'));
            h.add(0.2, pair(i, i + 1, 'X', 'Y', 'I'));
            h.add(-0.2, pair(i, i + 1, 'Y', 'X', 'I'));
        }
        h.add(0.7, pair(0, 3, 'X', 'X', 'Z'));
        h.add(0.7, pair(0, 3, 'Y', 'Y', 'Z'));
        for q in 0..n {
            h.add(0.1 * q as f64, PauliString::new([(q, Pauli::Z)]).unwrap());
        }
        h.add(1.5, PauliString::new([]).unwrap());
        h
    }

    #[test]
    fn charged_mpo_reproduces_the_pauli_sum() {
        let n = 5;
        let h = conserving(n);
        let charges = Charges::new(
            &[
                Charge::weights(vec![1; n]),
                Charge::modulo(2, vec![[0, 1]; n]),
            ],
            n,
        )
        .unwrap();
        let mpo = Mpo::with_charges(&h, n, &charges).unwrap();
        assert_same(&dense(&mpo), &reference(&h, n));
        // Every entry moves charge consistently: a channel's change is its
        // operator's plus the next channel's.
        let channels = mpo.channels.as_ref().unwrap();
        for (i, site) in mpo.sites.iter().enumerate() {
            for &(a, b, op, _) in &site.entries {
                let here = charge_change(&[(i, op)], &charges);
                let want = charges.add(&here, &channels[i + 1][b]);
                assert_eq!(channels[i][a], want, "site {i}, {a} → {b}, {op:?}");
            }
        }
    }

    #[test]
    fn non_conserving_terms_are_named() {
        let n = 3;
        let charges = Charges::new(&[Charge::weights(vec![1, 1, 0])], n).unwrap();
        let h = PauliSum::from_list([("IXX", 1.0), ("IYY", 1.0), ("XII", 0.5)]).unwrap();
        // X on qubit 2 is fine: no charge sees it.
        let ok = Mpo::with_charges(&h, n, &charges).unwrap();
        assert_same(&dense(&ok), &reference(&h, n));
        let h = PauliSum::from_list([("IXX", 1.0), ("IYY", -1.0)]).unwrap();
        let err = Mpo::with_charges(&h, n, &charges).unwrap_err();
        let Error::BreaksSymmetry(message) = err else {
            panic!("{err:?}")
        };
        assert!(message.contains("X0 X1"), "{message}");
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
