//! Symmetric MPS: states with definite Abelian charges (see
//! [`Charge`]), whose bonds are split into charge sectors.
//!
//! Site tensors keep their dense layout, with each bond's basis sorted by
//! sector, so a symmetric MPS is also a valid dense one: amplitudes,
//! sampling, expectations and projections work unchanged. What changes is
//! every factorisation. A tensor is block-diagonal in the charge of the
//! bond being factorised, so SVDs run block by block, the kept directions
//! of all blocks are ranked together under the usual `cutoff` and
//! `max_bond`, and every new basis vector has a definite charge. Sweeps
//! (DMRG, TDVP) keep their environments block-sparse as well.

use super::charges::{Charge, Charges, Groups, Label, Sectors, group_bond, group_cols, group_rows};
use super::linalg::{matmul, thin_svd};
use super::mpo::Mpo;
use super::{Absorb, Mps, MpsOptions, Tensor, ZERO, keep_count};
use crate::error::Error;
use crate::observable::PauliSum;
use faer::{Mat, MatRef};
use num_complex::Complex64 as C64;
use rayon::prelude::*;
use std::sync::Arc;

/// The largest weight [`Mps::with_charges`] drops from a state outside its
/// main charge sector, as a fraction of the norm squared.
pub const MAX_CHARGE_LEAK: f64 = 1e-8;

/// Singular values below this times the norm are rounding error when a
/// dense state is converted, and left out.
const ZERO_SINGULAR: f64 = 1e-13;

/// The charges of a symmetric MPS and the sectors of its bonds.
#[derive(Clone, Debug)]
pub(super) struct Symmetry {
    pub(super) charges: Arc<Charges>,
    /// `bonds[i]` is the bond left of site `i`: `bonds[0]` has the zero
    /// label and `bonds[n]` the total charge, both of dimension 1.
    pub(super) bonds: Vec<Sectors>,
}

/// Which directions a block-wise factorisation keeps.
#[derive(Clone, Copy)]
pub(super) enum Keep {
    /// Every one: an exact change of basis.
    All,
    /// The largest, ranked across blocks, under the state's truncation
    /// settings, never counting any below `floor` (an absolute singular
    /// value).
    Truncate { floor: f64 },
}

/// The SVD of a matrix that is block-diagonal in some charge, block by
/// block: block `b` has rows `rows[b]` and columns `cols[b]` of the
/// matrix, and is `u[b] diag(s[b]) v[b]†`.
pub(super) struct BlockSvd {
    pub(super) labels: Vec<Label>,
    pub(super) rows: Vec<Vec<usize>>,
    pub(super) cols: Vec<Vec<usize>>,
    pub(super) u: Vec<Mat<C64>>,
    pub(super) s: Vec<Vec<f64>>,
    pub(super) v: Vec<Mat<C64>>,
    nrows: usize,
    ncols: usize,
}

impl BlockSvd {
    /// The blocks of `m` whose labels appear among both the row and the
    /// column groups. Entries outside them are taken to be zero.
    pub(super) fn new(m: MatRef<'_, C64>, rows: Groups, cols: Groups) -> BlockSvd {
        let mut matched = Vec::new();
        let mut col_iter = cols.into_iter().peekable();
        for (label, r) in rows {
            while col_iter.next_if(|(l, _)| *l < label).is_some() {}
            if let Some((_, c)) = col_iter.next_if(|(l, _)| *l == label) {
                matched.push((label, r, c));
            }
        }
        let factors: Vec<_> = matched
            .par_iter()
            .map(|(_, r, c)| {
                let block = Mat::<C64>::from_fn(r.len(), c.len(), |a, b| m[(r[a], c[b])]);
                thin_svd(block.as_ref())
            })
            .collect();
        let (mut u, mut s, mut v) = (Vec::new(), Vec::new(), Vec::new());
        for (ub, sb, vb) in factors {
            u.push(ub);
            s.push(sb);
            v.push(vb);
        }
        let (mut labels, mut row_groups, mut col_groups) = (Vec::new(), Vec::new(), Vec::new());
        for (label, r, c) in matched {
            labels.push(label);
            row_groups.push(r);
            col_groups.push(c);
        }
        BlockSvd {
            labels,
            rows: row_groups,
            cols: col_groups,
            u,
            s,
            v,
            nrows: m.nrows(),
            ncols: m.ncols(),
        }
    }

    /// `Σ s²` over every block.
    pub(super) fn weight(&self) -> f64 {
        self.s.iter().flatten().map(|x| x * x).sum()
    }

    /// How many directions each block keeps.
    pub(super) fn counts(&self, keep: Keep, options: &MpsOptions) -> Vec<usize> {
        match keep {
            Keep::All => self.s.iter().map(Vec::len).collect(),
            Keep::Truncate { floor } => {
                // One ranking of all singular values, ties in block order,
                // so each block keeps a prefix of its own.
                let mut ranked: Vec<(f64, usize, usize)> = self
                    .s
                    .iter()
                    .enumerate()
                    .flat_map(|(b, s)| s.iter().enumerate().map(move |(j, &x)| (x, b, j)))
                    .collect();
                ranked.sort_by(|x, y| y.0.total_cmp(&x.0).then((x.1, x.2).cmp(&(y.1, y.2))));
                let above = ranked.iter().take_while(|r| r.0 > floor).count().max(1);
                ranked.truncate(above);
                let values: Vec<f64> = ranked.iter().map(|r| r.0).collect();
                let total = self.weight();
                let keep = keep_count(&values, total, options).min(values.len());
                let mut counts = vec![0; self.s.len()];
                for &(_, b, _) in &ranked[..keep] {
                    counts[b] += 1;
                }
                counts
            }
        }
    }

    /// `Σ s²` of the kept directions.
    pub(super) fn kept_weight(&self, counts: &[usize]) -> f64 {
        self.s
            .iter()
            .zip(counts)
            .flat_map(|(s, &k)| &s[..k])
            .map(|x| x * x)
            .sum()
    }

    /// The new bond: each block's label with its kept directions.
    pub(super) fn sectors(&self, counts: &[usize]) -> Sectors {
        Sectors::new(
            self.labels
                .iter()
                .cloned()
                .zip(counts.iter().copied())
                .collect(),
        )
    }

    /// The kept left singular vectors as columns, `nrows × Σ counts`, each
    /// scaled by `scale · s` if `scale` is given.
    fn left(&self, counts: &[usize], scale: Option<f64>) -> Mat<C64> {
        let width = counts.iter().sum();
        let mut out = Mat::<C64>::zeros(self.nrows, width);
        let mut at = 0;
        for (b, &k) in counts.iter().enumerate() {
            for j in 0..k {
                let w = scale.map_or(1.0, |r| r * self.s[b][j]);
                for (a, &row) in self.rows[b].iter().enumerate() {
                    out[(row, at + j)] = self.u[b][(a, j)] * w;
                }
            }
            at += k;
        }
        out
    }

    /// The kept right singular vectors as rows (`V†`), `Σ counts × ncols`,
    /// each scaled by `scale · s` if `scale` is given.
    fn right(&self, counts: &[usize], scale: Option<f64>) -> Mat<C64> {
        let height = counts.iter().sum();
        let mut out = Mat::<C64>::zeros(height, self.ncols);
        let mut at = 0;
        for (b, &k) in counts.iter().enumerate() {
            for j in 0..k {
                let w = scale.map_or(1.0, |r| r * self.s[b][j]);
                for (c, &col) in self.cols[b].iter().enumerate() {
                    out[(at + j, col)] = self.v[b][(c, j)].conj() * w;
                }
            }
            at += k;
        }
        out
    }
}

impl Mps {
    /// This state as a symmetric MPS with definite `charges` (each with a
    /// value per qubit; see [`Charge`]).
    ///
    /// Each bond's basis is rotated into charge sectors, by an SVD within
    /// each sector, and sorted by them. Every factorisation from then on
    /// keeps the sectors exact: DMRG and TDVP run within the state's
    /// sector, on block-sparse tensors, which saves time and memory roughly
    /// in proportion to the number of sectors per bond, and the state can
    /// no longer leak out of its sector through rounding.
    ///
    /// Product states such as [`Mps::basis`] convert exactly. A general
    /// state is projected onto its largest total-charge sector, and is
    /// truncated as its [`options`](Self::options) say. Errors if more
    /// than [`MAX_CHARGE_LEAK`] of the weight lies outside that sector, if
    /// a charge has the wrong number of values, or if the state is zero.
    /// Gates can't be run on a symmetric MPS; see
    /// [`to_dense`](Self::to_dense).
    pub fn with_charges(&self, charges: &[Charge]) -> Result<Mps, Error> {
        let n = self.n_qubits();
        let charges = Arc::new(Charges::new(charges, n)?);
        let mut mps = self.to_dense();
        mps.move_center(0);
        let total = mps.norm_sqr();
        if total == 0.0 || !total.is_finite() {
            return Err(Error::InvalidArgument(
                "can't give charges to a zero state".into(),
            ));
        }
        let floor = ZERO_SINGULAR * total.sqrt();

        // Left to right: split each bond into sectors of the charge to its
        // left. The rest of the chain is right-orthonormal, so the singular
        // values are Schmidt values and only numerical zeros are dropped.
        let mut bonds = vec![Sectors::single(charges.zero())];
        for i in 0..n - 1 {
            let a = mps.tensors[i].as_left_matrix();
            let rows = group_rows(&bonds[i], &charges, i);
            let all: Vec<usize> = (0..a.ncols()).collect();
            let cols = rows.iter().map(|(l, _)| (l.clone(), all.clone())).collect();
            let svd = BlockSvd::new(a, rows, cols);
            let exact = MpsOptions {
                max_bond: usize::MAX,
                cutoff: 0.0,
            };
            let counts = svd.counts(Keep::Truncate { floor }, &exact);
            let left = mps.tensors[i].left;
            let basis = svd.left(&counts, None);
            let carry = svd.right(&counts, Some(1.0));
            let width = basis.ncols();
            mps.tensors[i] = Tensor::from_matrix(basis.as_ref(), left, width);
            let next = &mps.tensors[i + 1];
            let merged = matmul(carry.as_ref(), next.as_right_matrix());
            mps.tensors[i + 1] = Tensor::from_matrix(merged.as_ref(), width, next.right);
            bonds.push(svd.sectors(&counts));
        }

        // The last site sets the total charge: keep its largest sector.
        let last = n - 1;
        let groups = group_rows(&bonds[last], &charges, last);
        let t = &mut mps.tensors[last];
        let weight = |rows: &[usize]| -> f64 { rows.iter().map(|&r| t.data[r].norm_sqr()).sum() };
        let (main, _) = groups
            .iter()
            .enumerate()
            .map(|(k, (_, rows))| (k, weight(rows)))
            .max_by(|x, y| x.1.total_cmp(&y.1))
            .expect("a site has at least one state");
        let kept = weight(&groups[main].1);
        let leak = ((total - kept) / total).max(0.0);
        if leak > MAX_CHARGE_LEAK {
            return Err(Error::BreaksSymmetry(format!(
                "{leak:.1e} of the state's weight lies outside its main charge sector {:?} \
                 (at most {MAX_CHARGE_LEAK:.0e} can be dropped)",
                groups[main].0
            )));
        }
        let mut keep_row = vec![false; t.data.len()];
        groups[main].1.iter().for_each(|&r| keep_row[r] = true);
        t.data
            .iter_mut()
            .zip(&keep_row)
            .filter(|(_, k)| !**k)
            .for_each(|(a, _)| *a = ZERO);
        bonds.push(Sectors::single(groups[main].0.clone()));
        mps.truncation_error += leak;
        mps.center = last;
        mps.symmetry = Some(Symmetry { charges, bonds });

        // Right to left: drop the sectors the total charge doesn't reach,
        // and truncate as the options say.
        while mps.center > 0 {
            mps.shift_left(Keep::Truncate { floor });
        }
        let scale = (total / mps.norm_sqr()).sqrt();
        mps.tensors[0].data.iter_mut().for_each(|a| *a *= scale);
        Ok(mps)
    }

    /// The same state without charges: a plain MPS that gates can run on.
    pub fn to_dense(&self) -> Mps {
        let mut mps = self.clone();
        mps.symmetry = None;
        mps
    }

    /// The charges of a symmetric MPS (see
    /// [`with_charges`](Self::with_charges)), or `None`.
    pub fn charges(&self) -> Option<&[Charge]> {
        self.symmetry.as_ref().map(|s| s.charges.charges())
    }

    /// The value of each charge on the state, or `None` without charges.
    pub fn total_charge(&self) -> Option<Vec<i64>> {
        self.symmetry
            .as_ref()
            .map(|s| s.bonds[self.n_qubits()].label(0).to_vec())
    }

    /// The sectors of the bond between qubits `bond` and `bond + 1`, as
    /// `(charges of the qubits to its left, dimension)` in the order of
    /// the bond's basis, or `None` without charges.
    pub fn bond_sectors(&self, bond: usize) -> Option<Vec<(Vec<i64>, usize)>> {
        assert!(bond + 1 < self.n_qubits(), "no bond {bond}");
        self.symmetry.as_ref().map(|s| {
            let sectors = &s.bonds[bond + 1];
            (0..sectors.len())
                .map(|k| (sectors.label(k).to_vec(), sectors.size(k)))
                .collect()
        })
    }

    /// The MPO of `hamiltonian` for this state's sweeps: graded by its
    /// charges if it has them (which `hamiltonian` must then conserve).
    pub(super) fn mpo(&self, hamiltonian: &PauliSum) -> Result<Mpo, Error> {
        match &self.symmetry {
            Some(sym) => Mpo::with_charges(hamiltonian, self.n_qubits(), &sym.charges),
            None => Mpo::new(hamiltonian, self.n_qubits()),
        }
    }

    /// Fails for a symmetric MPS, which `what` doesn't support.
    pub(super) fn require_dense(&self, what: &str) -> Result<(), Error> {
        if self.symmetry.is_some() {
            return Err(Error::InvalidArgument(format!(
                "{what} on an MPS with charges isn't supported; convert it with to_dense first"
            )));
        }
        Ok(())
    }

    /// [`split`](Mps::split) for a symmetric MPS: `θ` is block-diagonal in
    /// the charge of the middle bond, so it is factorised block by block,
    /// with one truncation over all blocks.
    pub(super) fn split_symmetric(&mut self, i: usize, theta: MatRef<'_, C64>, absorb: Absorb) {
        let sym = self.symmetry.as_ref().expect("a symmetric MPS");
        let (left, right) = (theta.nrows() / 2, theta.ncols() / 2);
        let rows = group_rows(&sym.bonds[i], &sym.charges, i);
        let cols = group_cols(&sym.bonds[i + 2], &sym.charges, i + 1);
        let svd = BlockSvd::new(theta, rows, cols);
        let counts = svd.counts(Keep::Truncate { floor: 0.0 }, &self.options);
        let total = svd.weight();
        let kept = svd.kept_weight(&counts);
        if total > 0.0 {
            self.truncation_error += (total - kept) / total;
        }
        let rescale = if kept > 0.0 {
            (total / kept).sqrt()
        } else {
            1.0
        };
        let (left_scale, right_scale) = match absorb {
            Absorb::Left => (Some(rescale), None),
            Absorb::Right => (None, Some(rescale)),
        };
        let a = svd.left(&counts, left_scale);
        let b = svd.right(&counts, right_scale);
        let bond = a.ncols();
        self.tensors[i] = Tensor::from_matrix(a.as_ref(), left, bond);
        self.tensors[i + 1] = Tensor::from_matrix(b.as_ref(), bond, right);
        self.symmetry.as_mut().unwrap().bonds[i + 1] = svd.sectors(&counts);
        self.center = match absorb {
            Absorb::Left => i,
            Absorb::Right => i + 1,
        };
    }

    /// Move the centre of a symmetric MPS one site right, keeping every
    /// direction: an exact change of basis within each sector.
    pub(super) fn shift_right(&mut self) {
        let i = self.center;
        let sym = self.symmetry.as_ref().expect("a symmetric MPS");
        let a = self.tensors[i].as_left_matrix();
        let rows = group_rows(&sym.bonds[i], &sym.charges, i);
        let cols = group_bond(&sym.bonds[i + 1]);
        let svd = BlockSvd::new(a, rows, cols);
        let counts = svd.counts(Keep::All, &self.options);
        let basis = svd.left(&counts, None);
        let carry = svd.right(&counts, Some(1.0));
        let (left, width) = (self.tensors[i].left, basis.ncols());
        self.tensors[i] = Tensor::from_matrix(basis.as_ref(), left, width);
        let next = &self.tensors[i + 1];
        let merged = matmul(carry.as_ref(), next.as_right_matrix());
        self.tensors[i + 1] = Tensor::from_matrix(merged.as_ref(), width, next.right);
        self.symmetry.as_mut().unwrap().bonds[i + 1] = svd.sectors(&counts);
        self.center += 1;
    }

    /// Move the centre of a symmetric MPS one site left, keeping the
    /// directions `keep` says (rescaled to keep the norm).
    fn shift_left(&mut self, keep: Keep) {
        let i = self.center;
        let sym = self.symmetry.as_ref().expect("a symmetric MPS");
        let a = self.tensors[i].as_right_matrix();
        let rows = group_bond(&sym.bonds[i]);
        let cols = group_cols(&sym.bonds[i + 1], &sym.charges, i);
        let svd = BlockSvd::new(a, rows, cols);
        let counts = svd.counts(keep, &self.options);
        let (total, kept) = (svd.weight(), svd.kept_weight(&counts));
        if total > 0.0 && kept < total {
            self.truncation_error += (total - kept) / total;
        }
        let rescale = if kept > 0.0 {
            (total / kept).sqrt()
        } else {
            1.0
        };
        let basis = svd.right(&counts, None);
        let carry = svd.left(&counts, Some(rescale));
        let (right, width) = (self.tensors[i].right, basis.nrows());
        self.tensors[i] = Tensor::from_matrix(basis.as_ref(), width, right);
        let prev = &self.tensors[i - 1];
        let merged = matmul(prev.as_left_matrix(), carry.as_ref());
        self.tensors[i - 1] = Tensor::from_matrix(merged.as_ref(), prev.left, width);
        self.symmetry.as_mut().unwrap().bonds[i] = svd.sectors(&counts);
        self.center -= 1;
    }

    /// [`shift_left`](Self::shift_left) keeping every direction.
    pub(super) fn shift_left_exact(&mut self) {
        self.shift_left(Keep::All);
    }

    /// The largest entry of any site tensor outside its allowed blocks:
    /// zero for a consistent symmetric MPS.
    #[cfg(test)]
    pub(super) fn charge_violation(&self) -> f64 {
        let sym = self.symmetry.as_ref().expect("a symmetric MPS");
        let mut worst: f64 = 0.0;
        for (i, t) in self.tensors.iter().enumerate() {
            let (left, right) = (&sym.bonds[i], &sym.bonds[i + 1]);
            assert_eq!(left.dim(), t.left);
            assert_eq!(right.dim(), t.right);
            let label = |b: &Sectors, x: usize| -> Label {
                let k = (0..b.len()).find(|&k| b.range(k).contains(&x)).unwrap();
                b.label(k).into()
            };
            for l in 0..t.left {
                for p in 0..2 {
                    let want = sym.charges.add(&label(left, l), sym.charges.site(i, p));
                    for r in 0..t.right {
                        if label(right, r) != want {
                            worst = worst.max(t.data[(l * 2 + p) * t.right + r].norm());
                        }
                    }
                }
            }
        }
        worst
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::Circuit;
    use crate::mps::MpsOptions;

    fn close(a: &[C64], b: &[C64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).norm())
            .fold(0.0, f64::max)
    }

    fn neel(n: usize) -> Vec<bool> {
        (0..n).map(|q| q % 2 == 1).collect()
    }

    /// Néel state under a few number-conserving hops, as a dense MPS.
    fn hopped(n: usize) -> Mps {
        let mut c = Circuit::new(n);
        for q in 0..n {
            if q % 2 == 1 {
                c.x(q);
            }
        }
        let mut h = PauliSum::new();
        for q in 0..n - 1 {
            let w = 0.3 + 0.1 * q as f64;
            h.add(w, PauliString::from_label(&pair(n, q, 'X')).unwrap());
            h.add(w, PauliString::from_label(&pair(n, q, 'Y')).unwrap());
            h.add(0.2, PauliString::from_label(&pair(n, q, 'Z')).unwrap());
        }
        c.pauli_evolution(&h, 0.7, 3, 2).unwrap();
        Mps::from_circuit(&c, MpsOptions::default()).unwrap()
    }

    fn pair(n: usize, q: usize, p: char) -> String {
        (0..n)
            .map(|k| if k == q || k == q + 1 { p } else { 'I' })
            .collect()
    }

    use crate::observable::PauliString;

    #[test]
    fn product_states_convert_exactly() {
        let n = 6;
        let mps = Mps::basis(n, &neel(n), MpsOptions::default());
        let sym = mps
            .with_charges(&[
                Charge::weights(vec![1; n]),
                Charge::modulo(2, vec![[0, 1]; n]),
            ])
            .unwrap();
        assert_eq!(sym.total_charge(), Some(vec![3, 1]));
        assert_eq!(sym.bond_sectors(2), Some(vec![(vec![1, 1], 1)]));
        assert!(close(&sym.to_amplitudes(), &mps.to_amplitudes()) < 1e-14);
        assert_eq!(sym.charge_violation(), 0.0);
        assert!(sym.to_dense().charges().is_none());
    }

    #[test]
    fn conserving_states_convert_into_sectors() {
        let n = 8;
        let mps = hopped(n);
        let sym = mps.with_charges(&[Charge::weights(vec![1; n])]).unwrap();
        assert_eq!(sym.total_charge(), Some(vec![4]));
        assert!(close(&sym.to_amplitudes(), &mps.to_amplitudes()) < 1e-12);
        assert!(sym.charge_violation() < 1e-14);
        // The middle bond has a sector for each particle count on the left
        // half that the hops reach.
        let sectors = sym.bond_sectors(3).unwrap();
        assert!(sectors.len() > 2, "{sectors:?}");
        assert_eq!(
            sectors.iter().map(|s| s.1).sum::<usize>(),
            sym.bond_dimensions()[3]
        );
    }

    #[test]
    fn states_outside_one_sector_are_refused() {
        let n = 4;
        let mut c = Circuit::new(n);
        c.h(1);
        let mps = Mps::from_circuit(&c, MpsOptions::default()).unwrap();
        let err = mps.with_charges(&[Charge::weights(vec![1; n])]);
        assert!(matches!(err, Err(Error::BreaksSymmetry(_))), "{err:?}");
        // A charge that doesn't see qubit 1 is fine.
        let sym = mps
            .with_charges(&[Charge::weights(vec![1, 0, 1, 1])])
            .unwrap();
        assert!(close(&sym.to_amplitudes(), &mps.to_amplitudes()) < 1e-14);
    }

    #[test]
    fn centre_moves_and_projections_keep_the_sectors() {
        let n = 8;
        let dense = hopped(n);
        let mut sym = dense.with_charges(&[Charge::weights(vec![1; n])]).unwrap();
        let mut dense = dense;
        sym.move_center(n - 1);
        sym.move_center(2);
        assert!(sym.charge_violation() < 1e-14);
        assert!(close(&sym.to_amplitudes(), &dense.to_amplitudes()) < 1e-12);
        let p = sym.project(5, true).unwrap();
        let q = dense.project(5, true).unwrap();
        assert!((p - q).abs() < 1e-12);
        assert!(sym.charge_violation() < 1e-14);
        assert!(close(&sym.to_amplitudes(), &dense.to_amplitudes()) < 1e-12);
        let z = PauliSum::from_list([("ZZIIIIII", 1.0), ("IIIXXIII", 0.5)]).unwrap();
        assert!((sym.expectation(&z).unwrap() - dense.expectation(&z).unwrap()).abs() < 1e-12);
    }

    #[test]
    fn gates_need_a_dense_state() {
        let n = 3;
        let mut sym = Mps::zero(n, MpsOptions::default())
            .with_charges(&[Charge::weights(vec![1; n])])
            .unwrap();
        let mut c = Circuit::new(n);
        c.x(0);
        assert!(sym.run(&c).is_err());
        assert!(sym.to_dense().run(&c).is_ok());
    }

    /// `Σ J (XX + YY + Δ ZZ)` on neighbours, a field, and a current.
    fn xxz(n: usize, delta: f64) -> PauliSum {
        let mut h = PauliSum::new();
        for q in 0..n - 1 {
            let j = 1.0 + 0.1 * q as f64;
            h.add(j, PauliString::from_label(&pair(n, q, 'X')).unwrap());
            h.add(j, PauliString::from_label(&pair(n, q, 'Y')).unwrap());
            h.add(
                j * delta,
                PauliString::from_label(&pair(n, q, 'Z')).unwrap(),
            );
            let mut xy: Vec<char> = pair(n, q, 'X').chars().collect();
            let k = xy.iter().rposition(|&c| c == 'X').unwrap();
            xy[k] = 'Y';
            let label: String = xy.iter().collect();
            h.add(0.2, PauliString::from_label(&label).unwrap());
            h.add(
                -0.2,
                PauliString::from_label(
                    &label.replace('X', "x").replace('Y', "X").replace('x', "Y"),
                )
                .unwrap(),
            );
        }
        for q in 0..n {
            h.add(
                0.05 * q as f64,
                PauliString::new([(q, crate::observable::Pauli::Z)]).unwrap(),
            );
        }
        h
    }

    #[test]
    fn dmrg_with_charges_matches_dense() {
        let n = 10;
        let h = xxz(n, 0.7);
        let options = crate::mps::DmrgOptions {
            max_bond: 32,
            noise: 0.0,
            ..Default::default()
        };
        let start = Mps::basis(n, &neel(n), MpsOptions::default());
        let mut dense = start.clone();
        let want = dense
            .dmrg(
                &h,
                &crate::mps::DmrgOptions {
                    max_bond: 32,
                    ..Default::default()
                },
            )
            .unwrap();
        let mut sym = start.with_charges(&[Charge::weights(vec![1; n])]).unwrap();
        let got = sym.dmrg(&h, &options).unwrap();
        assert!(got.converged);
        assert!(
            (got.energy - want.energy).abs() < 1e-8,
            "{} vs {}",
            got.energy,
            want.energy
        );
        assert!(sym.charge_violation() < 1e-12);
        assert_eq!(sym.total_charge(), Some(vec![5]));
        assert!((sym.expectation(&h).unwrap() - got.energy).abs() < 1e-8);
    }

    #[test]
    fn dmrg_with_a_parity_matches_dense() {
        // A transverse-field Ising chain written so its field is Z: XX
        // flips two qubits at once, so the parity of the ones is kept.
        let n = 9;
        let mut h = PauliSum::new();
        for q in 0..n - 1 {
            h.add(-1.0, PauliString::from_label(&pair(n, q, 'X')).unwrap());
        }
        for q in 0..n {
            h.add(
                -0.8,
                PauliString::new([(q, crate::observable::Pauli::Z)]).unwrap(),
            );
        }
        let options = crate::mps::DmrgOptions {
            max_bond: 32,
            noise: 0.0,
            ..Default::default()
        };
        let mut bits = vec![false; n];
        bits[4] = true;
        let start = Mps::basis(n, &bits, MpsOptions::default());
        let mut dense = start.clone();
        let want = dense
            .dmrg(
                &h,
                &crate::mps::DmrgOptions {
                    max_bond: 32,
                    ..Default::default()
                },
            )
            .unwrap();
        let mut sym = start
            .with_charges(&[Charge::modulo(2, vec![[0, 1]; n])])
            .unwrap();
        let got = sym.dmrg(&h, &options).unwrap();
        assert!(
            (got.energy - want.energy).abs() < 1e-8,
            "{} vs {}",
            got.energy,
            want.energy
        );
        assert_eq!(sym.total_charge(), Some(vec![1]));
        // Two sectors per bond (the parity to the left), never more.
        assert!((0..n - 1).all(|b| sym.bond_sectors(b).unwrap().len() <= 2));
    }

    #[test]
    fn tdvp_with_charges_matches_dense() {
        let n = 6;
        let h = xxz(n, 0.5);
        let dense = hopped(n);
        let mut sym = dense.with_charges(&[Charge::weights(vec![1; n])]).unwrap();
        let mut dense = dense;
        // Bonds are already full within the sector, so both are exact.
        dense.tdvp(&h, 0.6, 6).unwrap();
        sym.tdvp(&h, 0.6, 6).unwrap();
        let diff = close(&sym.to_amplitudes(), &dense.to_amplitudes());
        assert!(diff < 1e-9, "{diff}");
        assert!(sym.charge_violation() < 1e-12);
        assert!((sym.norm_sqr() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn dmrg_with_twenty_u1_charges_and_noise_does_not_diverge() {
        // Regression test for large-N / >= 18 U(1) charge symmetric DMRG
        // stability (both noise = 0.0 and noise = 1e-4).
        use crate::Pauli;
        let n_blocks = 20;
        let n = 2 * n_blocks; // 40 qubits, 20 local U(1) charges + 1 global U(1)
        let mut h = PauliSum::new();
        for b in 0..n_blocks {
            let (i, j) = (2 * b, 2 * b + 1);
            for p in [Pauli::X, Pauli::Y, Pauli::Z] {
                h.add(1.0, PauliString::new([(i, p), (j, p)]).unwrap());
            }
            if b + 1 < n_blocks {
                h.add(
                    0.4,
                    PauliString::new([(j, Pauli::Z), (j + 1, Pauli::Z)]).unwrap(),
                );
            }
        }
        let mut charges = Vec::with_capacity(n_blocks + 1);
        charges.push(Charge::weights(vec![1; n]));
        for b in 0..n_blocks {
            let mut w = vec![0i64; n];
            w[2 * b] = 1;
            w[2 * b + 1] = 1;
            charges.push(Charge::weights(w));
        }
        let neel: Vec<bool> = (0..n).map(|q| q % 2 == 1).collect();
        for &noise in &[0.0, 1e-4] {
            let mut mps = Mps::basis(n, &neel, MpsOptions::default())
                .with_charges(&charges)
                .unwrap();
            let res = mps
                .dmrg(
                    &h,
                    &crate::mps::DmrgOptions {
                        max_bond: 16,
                        cutoff: 1e-10,
                        max_sweeps: 4,
                        tolerance: 1e-8,
                        noise,
                        ..Default::default()
                    },
                )
                .unwrap();
            assert!(
                res.energy.is_finite(),
                "noise={noise}: energy={}",
                res.energy
            );
            assert!(mps.charge_violation() < 1e-12);
            assert!((mps.norm_sqr() - 1.0).abs() < 1e-10);
        }
    }
}
