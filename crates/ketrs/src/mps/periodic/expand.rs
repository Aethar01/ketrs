//! Global subspace expansion (Yang & White, Phys. Rev. B 102, 094315, 2020)
//! on a cyclic window `W = [w_0, ..., w_{L-1}]` of a [`PeriodicMps`].
//!
//! Two-site periodic TDVP projects onto states reachable within the current
//! bond bases, so from a weakly entangled or product start any multi-site
//! (`> 2`-qubit) or non-nearest-neighbour coupling has zero matrix element in
//! every local two-site effective Hamiltonian. Before each half-sweep on a
//! window `W`, enlarging the internal right (for a forward sweep) or left (for
//! a backward sweep) bases with the directions `H_W |ψ⟩` reaches — at zero
//! weight — leaves `|ψ⟩` unchanged while preserving strict canonical
//! orthonormality inside `W` (and exact charge sectors when `with_charges` is
//! active), so multi-qubit and long-range couplings act from the very first
//! step.

use super::PeriodicMps;
use super::symmetric::PeriodicSiteBlocks;
use crate::mps::dmrg::{Env, left_operator, right_operator};
use crate::mps::expand::complement;
use crate::mps::linalg::matmul;
use crate::mps::mpo::{Mpo, left_keyed, right_keyed};
use crate::mps::symmetric_expand::{Chain, Piece, Placement, chain_step};
use crate::mps::{Tensor, ZERO};
use faer::Mat;
use num_complex::Complex64 as C64;
use rayon::prelude::*;

impl PeriodicMps {
    /// Enlarge the internal right bases of cyclic window `win_sites`
    /// (`pos = 1..L`) with the directions `mpo |ψ⟩` reaches, leaving `|ψ⟩`
    /// unchanged and `win_sites[1..L]` strictly right-orthonormal.
    ///
    /// Requires `win_sites[1..L]` to already be right-orthonormal (as produced
    /// by `canonicalise_window_left`). Boundary bonds outside `win_sites` are
    /// left untouched.
    pub(super) fn expand_window_right(&mut self, win_sites: &[usize], mpo: &Mpo) {
        let l_win = win_sites.len();
        if l_win < 3 {
            return;
        }
        if self.symmetry.is_some() {
            self.expand_window_right_symmetric(win_sites, mpo);
            return;
        }
        let max_bond = self.options.max_bond;
        let r_end = self.tensors[win_sites[l_win - 1]].right;
        let mut env = Env {
            blocks: vec![Mat::<C64>::identity(r_end, r_end)],
        };
        let mut added = 0usize;
        for pos in (1..l_win).rev() {
            let q = win_sites[pos];
            let ket = &self.tensors[q];
            let (l, r) = (ket.left, ket.right + added);
            let padded = pad_right(ket, r);
            let room = max_bond.min(2 * r).saturating_sub(l);
            let channels = right_operator(&env, ket, &mpo.sites[pos]);
            let cols = 2 * r;
            let mut rows = Mat::<C64>::zeros(channels.len() * l, cols);
            for (a, ya) in channels.iter().enumerate() {
                rows.as_mut().submatrix_mut(a * l, 0, l, cols).copy_from(ya);
            }
            let new_rows = complement(rows.as_ref(), padded.as_right_matrix(), room);
            let k = new_rows.nrows();
            let mut data = padded.data;
            data.reserve(k * cols);
            for row in 0..k {
                for col in 0..cols {
                    data.push(new_rows[(row, col)]);
                }
            }
            let expanded = Tensor {
                left: l + k,
                right: r,
                data,
            };
            env = Env {
                blocks: channels
                    .par_iter()
                    .map(|ya| matmul(expanded.as_right_matrix().conjugate(), ya.transpose()))
                    .collect(),
            };
            self.tensors[q] = expanded;
            added = k;
        }
        if added > 0 {
            let q0 = win_sites[0];
            self.tensors[q0] = pad_right(&self.tensors[q0], self.tensors[q0].right + added);
        }
    }

    /// Mirror of [`expand_window_right`](Self::expand_window_right): enlarge
    /// the internal left bases of cyclic window `win_sites` (`pos = 0..L-1`)
    /// with the directions `mpo |ψ⟩` reaches, leaving `|ψ⟩` unchanged and
    /// `win_sites[0..L-1]` strictly left-orthonormal.
    ///
    /// Requires `win_sites[0..L-1]` to already be left-orthonormal (as produced
    /// by `canonicalise_window_right`).
    #[allow(clippy::needless_range_loop)]
    pub(super) fn expand_window_left(&mut self, win_sites: &[usize], mpo: &Mpo) {
        let l_win = win_sites.len();
        if l_win < 3 {
            return;
        }
        if self.symmetry.is_some() {
            self.expand_window_left_symmetric(win_sites, mpo);
            return;
        }
        let max_bond = self.options.max_bond;
        let l_start = self.tensors[win_sites[0]].left;
        let mut env = Env {
            blocks: vec![Mat::<C64>::identity(l_start, l_start)],
        };
        let mut added = 0usize;
        for pos in 0..l_win - 1 {
            let q = win_sites[pos];
            let ket = &self.tensors[q];
            let (l, r) = (ket.left + added, ket.right);
            let padded = pad_left(ket, l);
            let basis = padded.as_left_matrix().transpose();
            let room = max_bond.min(2 * l).saturating_sub(r);
            let channels = left_operator(&env, ket, &mpo.sites[pos]);
            let cols = 2 * l;
            let mut rows = Mat::<C64>::zeros(channels.len() * r, cols);
            for (b, yb) in channels.iter().enumerate() {
                rows.as_mut()
                    .submatrix_mut(b * r, 0, r, cols)
                    .copy_from(yb.transpose());
            }
            let new_cols = complement(rows.as_ref(), basis, room);
            let k = new_cols.nrows();
            let expanded = Mat::<C64>::from_fn(2 * l, r + k, |row, col| {
                if col < r {
                    padded.data[row * r + col]
                } else {
                    new_cols[(col - r, row)]
                }
            });
            let expanded = Tensor::from_matrix(expanded.as_ref(), l, r + k);
            env = Env {
                blocks: channels
                    .par_iter()
                    .map(|yb| matmul(expanded.as_left_matrix().adjoint(), yb.as_ref()))
                    .collect(),
            };
            self.tensors[q] = expanded;
            added = k;
        }
        if added > 0 {
            let q_last = win_sites[l_win - 1];
            self.tensors[q_last] =
                pad_left(&self.tensors[q_last], self.tensors[q_last].left + added);
        }
    }

    fn expand_window_right_symmetric(&mut self, win_sites: &[usize], mpo: &Mpo) {
        let l_win = win_sites.len();
        let max_bond = self.options.max_bond;
        let deltas = mpo.channels.as_ref().expect("charge-graded window MPO");
        let charges = self.symmetry.as_ref().unwrap().charges.clone();
        let mut ket_right = self
            .symmetry
            .as_ref()
            .unwrap()
            .right_bond(win_sites[l_win - 1])
            .clone();
        let mut chain: Chain = vec![
            (0..ket_right.len())
                .map(|s| {
                    let d = ket_right.size(s);
                    Some((s, Mat::identity(d, d)))
                })
                .collect(),
        ];
        let mut placed: Option<Placement> = None;

        for pos in (1..l_win).rev() {
            let q = win_sites[pos];
            let ket = self.tensors[q].clone();
            let padded = placed
                .as_ref()
                .map_or_else(|| ket.clone(), |p| p.right(&ket));
            let sym = self.symmetry.as_ref().unwrap();
            let bond = sym.left_bond(q).clone();
            let bra_right = sym.right_bond(q);
            let blocks = PeriodicSiteBlocks::new(&ket, &bond, &ket_right, sym, q);
            let rows = left_keyed(&mpo.sites[pos]);
            let dim = bra_right.dim();
            let shift = [sym.site_shift(q, 0), sym.site_shift(q, 1)];
            let pieces = |a: usize, s: usize| -> Vec<Piece> {
                let label = charges.sub(bond.label(s), &deltas[pos][a]);
                (0..2)
                    .filter_map(|p_out| {
                        let t_bra = bra_right.find(&charges.add(&label, &shift[p_out]))?;
                        let mut acc = Mat::<C64>::zeros(bond.size(s), bra_right.size(t_bra));
                        let mut any = false;
                        for &(c, p_in, f) in &rows[a * 2 + p_out] {
                            let Some((t, blk)) = blocks.block(s, p_in) else {
                                continue;
                            };
                            let Some((tb, r)) = &chain[c][t] else {
                                continue;
                            };
                            debug_assert_eq!(*tb, t_bra);
                            acc += faer::Scale(f) * matmul(blk, r.transpose());
                            any = true;
                        }
                        any.then_some(Piece {
                            p: p_out,
                            b: t_bra,
                            m: acc,
                        })
                    })
                    .collect()
            };
            let index = |p: usize, b: usize, j: usize| p * dim + bra_right.range(b).start + j;
            let cols = sym.group_cols(bra_right, q);
            let room = max_bond.min(2 * dim).saturating_sub(ket.left);
            let (sectors, placement, grown, next) = chain_step(
                &bond,
                &deltas[pos],
                &charges,
                cols,
                padded.as_right_matrix(),
                room,
                index,
                pieces,
            );
            self.tensors[q] = Tensor::from_matrix(grown.as_ref(), placement.dim, padded.right);
            self.symmetry.as_mut().unwrap().bonds[q] = sectors;
            chain = next;
            ket_right = bond;
            placed = Some(placement);
        }
        if let Some(p) = placed {
            let q0 = win_sites[0];
            self.tensors[q0] = p.right(&self.tensors[q0]);
        }
    }

    fn expand_window_left_symmetric(&mut self, win_sites: &[usize], mpo: &Mpo) {
        let l_win = win_sites.len();
        let max_bond = self.options.max_bond;
        let deltas = mpo.channels.as_ref().expect("charge-graded window MPO");
        let charges = self.symmetry.as_ref().unwrap().charges.clone();
        let mut ket_left = self
            .symmetry
            .as_ref()
            .unwrap()
            .left_bond(win_sites[0])
            .clone();
        let mut chain: Chain = vec![
            (0..ket_left.len())
                .map(|s| {
                    let d = ket_left.size(s);
                    Some((s, Mat::identity(d, d)))
                })
                .collect(),
        ];
        let mut placed: Option<Placement> = None;

        for pos in 0..l_win - 1 {
            let q = win_sites[pos];
            let q_next = win_sites[pos + 1];
            let ket = self.tensors[q].clone();
            let padded = placed
                .as_ref()
                .map_or_else(|| ket.clone(), |p| p.left(&ket));
            let sym = self.symmetry.as_ref().unwrap();
            let bond = sym.right_bond(q).clone();
            let bra_left = sym.left_bond(q);
            let blocks = PeriodicSiteBlocks::new(&ket, &ket_left, &bond, sym, q);
            let rows = right_keyed(&mpo.sites[pos]);
            let shift = [sym.site_shift(q, 0), sym.site_shift(q, 1)];
            let pieces = |b: usize, t: usize| -> Vec<Piece> {
                let label = charges.sub(bond.label(t), &deltas[pos + 1][b]);
                (0..2)
                    .filter_map(|p_out| {
                        let s_bra = bra_left.find(&charges.sub(&label, &shift[p_out]))?;
                        let mut acc = Mat::<C64>::zeros(bond.size(t), bra_left.size(s_bra));
                        let mut any = false;
                        for &(a, p_in, f) in &rows[b * 2 + p_out] {
                            let Some(s_ket) = blocks.left_of[t * 2 + p_in] else {
                                continue;
                            };
                            let (_, blk) = blocks.block(s_ket, p_in).expect("block reaches t");
                            let Some((sb, l)) = &chain[a][s_ket] else {
                                continue;
                            };
                            debug_assert_eq!(*sb, s_bra);
                            acc += faer::Scale(f) * matmul(blk.transpose(), l.transpose());
                            any = true;
                        }
                        any.then_some(Piece {
                            p: p_out,
                            b: s_bra,
                            m: acc,
                        })
                    })
                    .collect()
            };
            let index = |p: usize, b: usize, j: usize| (bra_left.range(b).start + j) * 2 + p;
            let cols = sym.group_rows(bra_left, q);
            let width = 2 * padded.left;
            let room = max_bond.min(width).saturating_sub(ket.right);
            let (sectors, placement, grown, next) = chain_step(
                &bond,
                &deltas[pos + 1],
                &charges,
                cols,
                padded.as_left_matrix().transpose(),
                room,
                index,
                pieces,
            );
            self.tensors[q] = Tensor::from_matrix(grown.transpose(), padded.left, placement.dim);
            self.symmetry.as_mut().unwrap().bonds[q_next] = sectors;
            chain = next;
            ket_left = bond;
            placed = Some(placement);
        }
        if let Some(p) = placed {
            let q_last = win_sites[l_win - 1];
            self.tensors[q_last] = p.left(&self.tensors[q_last]);
        }
    }
}

fn pad_left(t: &Tensor, left: usize) -> Tensor {
    let mut data = t.data.clone();
    data.resize(left * 2 * t.right, ZERO);
    Tensor {
        left,
        right: t.right,
        data,
    }
}

fn pad_right(t: &Tensor, right: usize) -> Tensor {
    let mut data = vec![ZERO; t.left * 2 * right];
    for lp in 0..t.left * 2 {
        data[lp * right..lp * right + t.right]
            .copy_from_slice(&t.data[lp * t.right..(lp + 1) * t.right]);
    }
    Tensor {
        left: t.left,
        right,
        data,
    }
}
