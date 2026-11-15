//! Block-sparse environments and effective Hamiltonians for symmetric MPS,
//! on the CPU.
//!
//! With a charge-graded MPO (see [`Mpo::with_charges`]) each channel `a` at
//! a cut changes the charge by a definite `δ_a` to its right. An
//! environment block `E_a[bra, ket]` is then non-zero only between one ket
//! sector `j` and one bra sector `j − δ_a`, so an environment is a list of
//! dense blocks, one per (channel, sector). They are stored stacked: for
//! each sector of the bond, the blocks of every channel that touch it, one
//! under another, so that one product covers all channels at once.
//!
//! Two-site (and one-site) vectors are stored compactly, as the allowed
//! blocks of the dense tensor only: for each left sector `s`, a row-major
//! matrix of its `d_s` rows by the right sectors `t = s + c(P)` that each
//! physical state `P` reaches, side by side. The effective Hamiltonian
//! takes three steps, as the dense one does (see
//! [`effective`](super::effective)), with every product a block product:
//!
//! 1. `Y_s = L_s Θ_s`: for each left sector `s`, the stacked left blocks
//!    for ket sector `s` times `Θ`'s rows in `s`;
//! 2. the MPO step, a list of scaled block copies from `Y` into `Z`,
//!    precomputed per output block;
//! 3. `out_(s, P) = Z_(s, P) R_t`: for each output block, `Z` times the
//!    right blocks for its bra sector `t`, stacked over channels.
//!
//! [`Mpo::with_charges`]: super::mpo::Mpo::with_charges

use super::Tensor;
use super::charges::{Charges, Label, Sectors};
use super::linalg::matmul;
use super::mpo::{MpoSite, Rows, left_keyed, pair_keyed, right_keyed};
use crate::statevector::krylov::LinearOperator;
use faer::linalg::matmul::matmul as faer_matmul;
use faer::{Accum, Mat, MatMut, MatRef, Par};
use num_complex::Complex64 as C64;
use rayon::prelude::*;

const ZERO: C64 = C64::new(0.0, 0.0);
const ONE: C64 = C64::new(1.0, 0.0);

/// Which side of the sites an environment covers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Side {
    /// Sites to the left: stack `j` is for ket sector `j`, and holds each
    /// channel's `L_a` block (bra sector `j − δ_a`) as rows.
    Left,
    /// Sites to the right: stack `j` is for bra sector `j`, and holds each
    /// channel's `R_cᵀ` block (ket sector `j + δ_c`) as rows.
    Right,
}

/// The blocks of one sector, stacked.
#[derive(Clone, Debug)]
pub(super) struct Stack {
    /// Where the stack starts in [`BlockEnv::data`].
    pub(super) offset: usize,
    pub(super) rows: usize,
    pub(super) cols: usize,
    /// For each channel, its block's first row in the stack and the sector
    /// of its rows, if it has a block here.
    pub(super) channels: Vec<Option<(usize, usize)>>,
}

/// Where the blocks of a block-sparse environment of one bond sit.
#[derive(Clone, Debug)]
pub(super) struct EnvLayout {
    pub(super) side: Side,
    pub(super) bond: Sectors,
    pub(super) stacks: Vec<Stack>,
    /// Entries in all stacks.
    pub(super) len: usize,
}

impl EnvLayout {
    /// The layout on `bond` of an environment whose channels change the
    /// charge by `deltas`.
    pub(super) fn new(side: Side, bond: &Sectors, deltas: &[Label], charges: &Charges) -> Self {
        let mut stacks = Vec::with_capacity(bond.len());
        let mut offset = 0;
        for j in 0..bond.len() {
            let mut rows = 0;
            let channels = deltas
                .iter()
                .map(|d| {
                    let label = match side {
                        Side::Left => charges.sub(bond.label(j), d),
                        Side::Right => charges.add(bond.label(j), d),
                    };
                    bond.find(&label).map(|k| {
                        let at = rows;
                        rows += bond.size(k);
                        (at, k)
                    })
                })
                .collect();
            let cols = bond.size(j);
            stacks.push(Stack {
                offset,
                rows,
                cols,
                channels,
            });
            offset += rows * cols;
        }
        EnvLayout {
            side,
            bond: bond.clone(),
            stacks,
            len: offset,
        }
    }

    /// The layout beyond an end of the chain: one channel on a
    /// one-dimensional bond.
    pub(super) fn boundary(side: Side, bond: &Sectors, charges: &Charges) -> Self {
        EnvLayout::new(side, bond, &[charges.zero()], charges)
    }

    pub(super) fn channels(&self) -> usize {
        self.stacks.first().map_or(0, |s| s.channels.len())
    }

    /// For each channel and row sector, the stack that has its block and
    /// the block's first row: the inverse of the stacks' channel lists.
    pub(super) fn by_row_sector(&self) -> Vec<Vec<Option<(usize, usize)>>> {
        let mut table = vec![vec![None; self.bond.len()]; self.channels()];
        for (j, s) in self.stacks.iter().enumerate() {
            for (ch, entry) in s.channels.iter().enumerate() {
                if let Some((row, k)) = *entry {
                    table[ch][k] = Some((j, row));
                }
            }
        }
        table
    }
}

/// A block-sparse environment of one bond on the CPU.
#[derive(Clone, Debug)]
pub(super) struct BlockEnv {
    pub(super) layout: EnvLayout,
    /// Every stack, row-major, one after another.
    pub(super) data: Vec<C64>,
}

impl std::ops::Deref for BlockEnv {
    type Target = EnvLayout;
    fn deref(&self) -> &EnvLayout {
        &self.layout
    }
}

impl BlockEnv {
    /// The zero environment of `bond` whose channels change the charge by
    /// `deltas`.
    pub(super) fn zeros(side: Side, bond: &Sectors, deltas: &[Label], charges: &Charges) -> Self {
        let layout = EnvLayout::new(side, bond, deltas, charges);
        let data = vec![ZERO; layout.len];
        BlockEnv { layout, data }
    }

    /// The environment beyond an end of the chain: one channel, the
    /// identity on a one-dimensional bond.
    pub(super) fn boundary(side: Side, bond: &Sectors, charges: &Charges) -> Self {
        let mut env = BlockEnv::zeros(side, bond, &[charges.zero()], charges);
        env.data[0] = ONE;
        env
    }

    /// Stack `j` as a matrix.
    pub(super) fn stack(&self, j: usize) -> MatRef<'_, C64> {
        let s = &self.stacks[j];
        MatRef::from_row_major_slice(&self.data[s.offset..][..s.rows * s.cols], s.rows, s.cols)
    }

    /// The same environment, dense: one `bra × ket` matrix per channel.
    pub(super) fn to_env(&self) -> super::dmrg::Env {
        let dim = self.bond.dim();
        let n_channels = self.channels();
        let mut blocks = vec![Mat::<C64>::zeros(dim, dim); n_channels];
        for (j, s) in self.stacks.iter().enumerate() {
            let stack = self.stack(j);
            for (ch, entry) in s.channels.iter().enumerate() {
                let Some((row, k)) = *entry else { continue };
                for (a, r) in self.bond.range(k).enumerate() {
                    for (b, c) in self.bond.range(j).enumerate() {
                        let v = stack[(row + a, b)];
                        match self.side {
                            Side::Left => blocks[ch][(r, c)] = v,
                            Side::Right => blocks[ch][(c, r)] = v,
                        }
                    }
                }
            }
        }
        super::dmrg::Env { blocks }
    }

    /// Every stack's slice of `data`, to fill in parallel.
    fn stacks_mut(&mut self) -> Vec<(&Stack, &mut [C64])> {
        let mut rest = self.data.as_mut_slice();
        let mut out = Vec::with_capacity(self.layout.stacks.len());
        for s in &self.layout.stacks {
            let (head, tail) = rest.split_at_mut(s.rows * s.cols);
            out.push((s, head));
            rest = tail;
        }
        out
    }
}

/// The allowed blocks of site `q`'s tensor `a`, between bonds `left` and
/// `right`: for left sector `s` and physical state `p`, the right sector
/// `t = s + c_q(p)` (if the right bond has it) and the `d_s × d_t` block.
pub(super) struct SiteBlocks<'a> {
    a: &'a Tensor,
    left: &'a Sectors,
    right: &'a Sectors,
    /// `right_of[s · 2 + p]`.
    pub(super) right_of: Vec<Option<usize>>,
    /// `left_of[t · 2 + p]`: the left sector `s` with `s + c(p) = t`.
    pub(super) left_of: Vec<Option<usize>>,
}

impl<'a> SiteBlocks<'a> {
    pub(super) fn new(
        a: &'a Tensor,
        left: &'a Sectors,
        right: &'a Sectors,
        charges: &Charges,
        q: usize,
    ) -> Self {
        debug_assert_eq!((left.dim(), right.dim()), (a.left, a.right));
        let right_of = (0..left.len() * 2)
            .map(|k| right.find(&charges.add(left.label(k / 2), charges.site(q, k % 2))))
            .collect();
        let left_of = (0..right.len() * 2)
            .map(|k| left.find(&charges.sub(right.label(k / 2), charges.site(q, k % 2))))
            .collect();
        SiteBlocks {
            a,
            left,
            right,
            right_of,
            left_of,
        }
    }

    /// The block from left sector `s` with physical state `p`, as its right
    /// sector and its first element in the tensor (rows `2 · right` apart).
    pub(super) fn offset(&self, s: usize, p: usize) -> Option<(usize, usize)> {
        let t = self.right_of[s * 2 + p]?;
        let start = (self.left.range(s).start * 2 + p) * self.a.right + self.right.range(t).start;
        Some((t, start))
    }

    /// The block from left sector `s` with physical state `p`.
    pub(super) fn block(&self, s: usize, p: usize) -> Option<(usize, MatRef<'a, C64>)> {
        let t = self.right_of[s * 2 + p]?;
        let (rs, rt) = (self.left.range(s), self.right.range(t));
        let stride = 2 * self.a.right;
        let start = (rs.start * 2 + p) * self.a.right + rt.start;
        let len = (rs.len() - 1) * stride + rt.len();
        let m = MatRef::from_row_major_slice_with_stride(
            &self.a.data[start..][..len],
            rs.len(),
            rt.len(),
            stride,
        );
        Some((t, m))
    }
}

/// `L'` at the bond right of site `q` from `L` at the bond left of it,
/// site `q`'s tensor `a` (between `env.bond` and `right`) and MPO site
/// `w`, whose right channels change the charge by `deltas`.
pub(super) fn update_left(
    env: &BlockEnv,
    a: &Tensor,
    w: &MpoSite,
    charges: &Charges,
    q: usize,
    right: &Sectors,
    deltas: &[Label],
) -> BlockEnv {
    debug_assert_eq!(env.side, Side::Left);
    let blocks = SiteBlocks::new(a, &env.bond, right, charges, q);
    // x[s · 2 + p] = L_s A_(s, p): every channel's rows at once.
    let x: Vec<Option<Mat<C64>>> = (0..env.bond.len() * 2)
        .into_par_iter()
        .map(|k| {
            let (_, b) = blocks.block(k / 2, k % 2)?;
            Some(matmul(env.stack(k / 2), b))
        })
        .collect();
    let rows = right_keyed(w);
    let mut out = BlockEnv::zeros(Side::Left, right, deltas, charges);
    out.stacks_mut()
        .into_par_iter()
        .enumerate()
        .for_each(|(t_in, (stack, data))| {
            let cols = stack.cols;
            for (b, entry) in stack.channels.iter().enumerate() {
                let Some((row0, t_out)) = *entry else {
                    continue;
                };
                let mut acc = Mat::<C64>::zeros(right.size(t_out), cols);
                for p_out in 0..2 {
                    let Some(s_out) = blocks.left_of[t_out * 2 + p_out] else {
                        continue;
                    };
                    let (_, bra) = blocks.block(s_out, p_out).expect("the block reaches t_out");
                    // y = Σ f · (rows of channel a in x[s_in, p_in]).
                    let mut y = Mat::<C64>::zeros(env.bond.size(s_out), cols);
                    let mut any = false;
                    for &(ch, p_in, f) in &rows[b * 2 + p_out] {
                        let Some(s_in) = blocks.left_of[t_in * 2 + p_in] else {
                            continue;
                        };
                        let (Some(xb), Some((r, k))) =
                            (&x[s_in * 2 + p_in], env.stacks[s_in].channels[ch])
                        else {
                            continue;
                        };
                        debug_assert_eq!(k, s_out);
                        let part = xb.as_ref().subrows(r, y.nrows());
                        y += faer::Scale(f) * part;
                        any = true;
                    }
                    if any {
                        acc += matmul(bra.adjoint(), y.as_ref());
                    }
                }
                write_rows(data, cols, row0, acc.as_ref());
            }
        });
    out
}

/// `R'` at the bond left of site `q` from `R` at the bond right of it,
/// site `q`'s tensor `a` (between `left` and `env.bond`) and MPO site `w`,
/// whose left channels change the charge by `deltas`.
pub(super) fn update_right(
    env: &BlockEnv,
    a: &Tensor,
    w: &MpoSite,
    charges: &Charges,
    q: usize,
    left: &Sectors,
    deltas: &[Label],
) -> BlockEnv {
    debug_assert_eq!(env.side, Side::Right);
    let blocks = SiteBlocks::new(a, left, &env.bond, charges, q);
    // x[s · 2 + p] = R_t A_(s, p)†, t = s + c(p): every channel's rows.
    let x: Vec<Option<Mat<C64>>> = (0..left.len() * 2)
        .into_par_iter()
        .map(|k| {
            let (t, b) = blocks.block(k / 2, k % 2)?;
            Some(matmul(env.stack(t), b.adjoint()))
        })
        .collect();
    let rows = left_keyed(w);
    let mut out = BlockEnv::zeros(Side::Right, left, deltas, charges);
    out.stacks_mut()
        .into_par_iter()
        .enumerate()
        .for_each(|(s_out, (stack, data))| {
            let cols = stack.cols;
            for (a_ch, entry) in stack.channels.iter().enumerate() {
                let Some((row0, s_in)) = *entry else {
                    continue;
                };
                let mut acc = Mat::<C64>::zeros(left.size(s_in), cols);
                for p_in in 0..2 {
                    let Some((t_in, ket)) = blocks.block(s_in, p_in) else {
                        continue;
                    };
                    // y = Σ f · (rows of channel c in x[s_out, p_out]).
                    let mut y = Mat::<C64>::zeros(env.bond.size(t_in), cols);
                    let mut any = false;
                    for p_out in 0..2 {
                        let Some(xb) = &x[s_out * 2 + p_out] else {
                            continue;
                        };
                        let t_out = blocks.right_of[s_out * 2 + p_out].unwrap();
                        for &(c, p, f) in &rows[a_ch * 2 + p_out] {
                            if p != p_in {
                                continue;
                            }
                            let Some((r, k)) = env.stacks[t_out].channels[c] else {
                                continue;
                            };
                            debug_assert_eq!(k, t_in);
                            y += faer::Scale(f) * xb.as_ref().subrows(r, y.nrows());
                            any = true;
                        }
                    }
                    if any {
                        acc += matmul(ket, y.as_ref());
                    }
                }
                write_rows(data, cols, row0, acc.as_ref());
            }
        });
    out
}

/// Copy `m` into rows `row0 ..` of a row-major matrix with `cols` columns.
fn write_rows(data: &mut [C64], cols: usize, row0: usize, m: MatRef<'_, C64>) {
    for r in 0..m.nrows() {
        let dst = &mut data[(row0 + r) * cols..][..cols];
        for (c, d) in dst.iter_mut().enumerate() {
            *d = m[(r, c)];
        }
    }
}

/// Where the allowed blocks of a one- or two-site tensor sit in its
/// compact vector.
pub(super) struct Layout {
    /// One per left sector with any block: `(s, d_s, offset, width)`,
    /// the `d_s × width` row-major matrix of its blocks at `offset`.
    pub(super) rows: Vec<(usize, usize, usize, usize)>,
    /// `cols[s · phys + P] = (row block, column, right sector)` for each
    /// allowed block.
    cols: Vec<Option<(usize, usize, usize)>>,
    phys: usize,
    left: Sectors,
    right: Sectors,
    pub(super) len: usize,
}

impl Layout {
    /// The blocks between `left` and `right`, where physical state `P`
    /// has charge `charge[P]`.
    pub(super) fn new(
        left: &Sectors,
        right: &Sectors,
        charge: &[Label],
        charges: &Charges,
    ) -> Self {
        let phys = charge.len();
        let mut rows = Vec::new();
        let mut cols = vec![None; left.len() * phys];
        let mut offset = 0;
        for s in 0..left.len() {
            let mut width = 0;
            for (p, c) in charge.iter().enumerate() {
                if let Some(t) = right.find(&charges.add(left.label(s), c)) {
                    cols[s * phys + p] = Some((rows.len(), width, t));
                    width += right.size(t);
                }
            }
            if width > 0 {
                rows.push((s, left.size(s), offset, width));
                offset += left.size(s) * width;
            }
        }
        Layout {
            rows,
            cols,
            phys,
            left: left.clone(),
            right: right.clone(),
            len: offset,
        }
    }

    /// The two-site tensor of sites `q` and `q + 1`, `a` and `b` (with the
    /// middle bond `mid`), straight into this layout: each block is one
    /// product of a block of `a` with one of `b`.
    pub(super) fn two_site(
        &self,
        a: &Tensor,
        b: &Tensor,
        mid: &Sectors,
        charges: &Charges,
        q: usize,
    ) -> Vec<C64> {
        debug_assert_eq!(self.phys, 4);
        let first = SiteBlocks::new(a, &self.left, mid, charges, q);
        let second = SiteBlocks::new(b, mid, &self.right, charges, q + 1);
        let mut out = vec![ZERO; self.len];
        let mut parts = Vec::with_capacity(self.rows.len());
        let mut rest = out.as_mut_slice();
        for &(_, d, _, width) in &self.rows {
            let (head, tail) = rest.split_at_mut(d * width);
            parts.push(head);
            rest = tail;
        }
        parts
            .into_par_iter()
            .zip(self.rows.par_iter())
            .for_each(|(part, &(s, d, _, width))| {
                for big_p in 0..4 {
                    let Some((_, col, t)) = self.cols[s * 4 + big_p] else {
                        continue;
                    };
                    let (p, qq) = (big_p / 2, big_p % 2);
                    let Some((m, x)) = first.block(s, p) else {
                        continue;
                    };
                    let Some((t2, y)) = second.block(m, qq) else {
                        continue;
                    };
                    debug_assert_eq!(t, t2);
                    let n = self.right.size(t);
                    let len = (d - 1) * width + n;
                    let dst = MatMut::from_column_major_slice_with_stride_mut(
                        &mut part[col..][..len],
                        n,
                        d,
                        width,
                    )
                    .transpose_mut();
                    faer_matmul(dst, Accum::Replace, x, y, ONE, Par::Seq);
                }
            });
        out
    }

    /// The allowed entries of a dense tensor, row-major `(l, P, r)`.
    pub(super) fn gather(&self, dense: &[C64]) -> Vec<C64> {
        let mut out = vec![ZERO; self.len];
        self.for_each_block(|dense_at, compact_at, n| {
            out[compact_at..][..n].copy_from_slice(&dense[dense_at..][..n]);
        });
        out
    }

    /// The dense tensor, row-major `(l, P, r)`, with zeros outside the
    /// blocks.
    pub(super) fn scatter(&self, compact: &[C64]) -> Vec<C64> {
        let mut out = vec![ZERO; self.left.dim() * self.phys * self.right.dim()];
        self.for_each_block(|dense_at, compact_at, n| {
            out[dense_at..][..n].copy_from_slice(&compact[compact_at..][..n]);
        });
        out
    }

    /// `f(dense offset, compact offset, length)` for each row of each block.
    fn for_each_block(&self, mut f: impl FnMut(usize, usize, usize)) {
        let r_dim = self.right.dim();
        for (s, entry) in self.cols.iter().enumerate() {
            let Some((rb, col, t)) = *entry else { continue };
            let (s, p) = (s / self.phys, s % self.phys);
            let (_, _, offset, width) = self.rows[rb];
            let (rs, rt) = (self.left.range(s), self.right.range(t));
            for (i, l) in rs.enumerate() {
                f(
                    (l * self.phys + p) * r_dim + rt.start,
                    offset + i * width + col,
                    rt.len(),
                );
            }
        }
    }
}

/// One scaled block copy of the MPO step: `z[:, z_col ..][.. cols] +=
/// f · y[y + r · stride ..][.. cols]` for each row `r`.
pub(super) struct Copy {
    pub(super) y: usize,
    pub(super) stride: usize,
    pub(super) z_col: usize,
    pub(super) cols: usize,
    pub(super) f: C64,
}

/// One output block: its place in its row block of the output, the right
/// stack it is multiplied by, and the copies that build its `Z`.
pub(super) struct Out {
    pub(super) column: usize,
    pub(super) t: usize,
    pub(super) copies: Vec<Copy>,
}

/// The block products of an effective Hamiltonian between two environment
/// layouts, the same on the CPU and the GPU.
pub(super) struct Plan {
    pub(super) layout: Layout,
    /// Where each row block's `Y` starts, and its height (the left
    /// stack's rows).
    pub(super) y: Vec<(usize, usize)>,
    pub(super) y_len: usize,
    /// For each row block, its output blocks.
    pub(super) out: Vec<Vec<Out>>,
}

impl Plan {
    /// Sites `q` and `q + 1` with MPO sites `w1` and `w2`.
    pub(super) fn two_site(
        left: &EnvLayout,
        right: &EnvLayout,
        w1: &MpoSite,
        w2: &MpoSite,
        charges: &Charges,
        q: usize,
    ) -> Self {
        let charge: Vec<Label> = (0..4)
            .map(|k| charges.add(charges.site(q, k / 2), charges.site(q + 1, k % 2)))
            .collect();
        Plan::new(left, right, pair_keyed(w1, w2), &charge, charges)
    }

    /// Site `q` with MPO site `w`.
    pub(super) fn one_site(
        left: &EnvLayout,
        right: &EnvLayout,
        w: &MpoSite,
        charges: &Charges,
        q: usize,
    ) -> Self {
        let charge: Vec<Label> = (0..2).map(|p| charges.site(q, p).into()).collect();
        Plan::new(left, right, right_keyed(w), &charge, charges)
    }

    fn new(
        left: &EnvLayout,
        right: &EnvLayout,
        rows: Rows,
        charge: &[Label],
        charges: &Charges,
    ) -> Self {
        let layout = Layout::new(&left.bond, &right.bond, charge, charges);
        let phys = layout.phys;
        let mut y = Vec::with_capacity(layout.rows.len());
        let mut y_len = 0;
        for &(s, _, _, width) in &layout.rows {
            let height = left.stacks[s].rows;
            y.push((y_len, height));
            y_len += height * width;
        }
        let ket_of = left.by_row_sector();
        let out = layout
            .rows
            .iter()
            .map(|&(s_out, _, _, _)| {
                (0..phys)
                    .filter_map(|p_out| {
                        let (_, column, t_out) = layout.cols[s_out * phys + p_out]?;
                        let stack = &right.stacks[t_out];
                        let mut copies = Vec::new();
                        for (c, entry) in stack.channels.iter().enumerate() {
                            let Some((z_col, t_in)) = *entry else {
                                continue;
                            };
                            for &(a, p_in, f) in &rows[c * phys + p_out] {
                                let Some((s_in, row)) = ket_of[a][s_out] else {
                                    continue;
                                };
                                let Some((rb, col, t)) = layout.cols[s_in * phys + p_in] else {
                                    continue;
                                };
                                debug_assert_eq!(t, t_in);
                                let (y_at, _) = y[rb];
                                let stride = layout.rows[rb].3;
                                copies.push(Copy {
                                    y: y_at + row * stride + col,
                                    stride,
                                    z_col,
                                    cols: right.bond.size(t),
                                    f,
                                });
                            }
                        }
                        Some(Out {
                            column,
                            t: t_out,
                            copies,
                        })
                    })
                    .collect()
            })
            .collect();
        Plan {
            layout,
            y,
            y_len,
            out,
        }
    }
}

/// The block-sparse effective Hamiltonian of one or two sites, on the CPU.
pub(super) struct BlockOp<'a> {
    left: &'a BlockEnv,
    right: &'a BlockEnv,
    plan: Plan,
}

impl<'a> BlockOp<'a> {
    /// Sites `q` and `q + 1` with MPO sites `w1` and `w2`.
    pub(super) fn two_site(
        left: &'a BlockEnv,
        right: &'a BlockEnv,
        w1: &MpoSite,
        w2: &MpoSite,
        charges: &Charges,
        q: usize,
    ) -> Self {
        let plan = Plan::two_site(left, right, w1, w2, charges, q);
        BlockOp { left, right, plan }
    }

    /// Site `q` with MPO site `w`.
    pub(super) fn one_site(
        left: &'a BlockEnv,
        right: &'a BlockEnv,
        w: &MpoSite,
        charges: &Charges,
        q: usize,
    ) -> Self {
        let plan = Plan::one_site(left, right, w, charges, q);
        BlockOp { left, right, plan }
    }

    /// The compact layout this operator acts on.
    pub(super) fn layout(&self) -> &Layout {
        &self.plan.layout
    }
}

impl LinearOperator<C64> for BlockOp<'_> {
    fn apply(&self, x: &[C64], out: &mut [C64]) {
        let plan = &self.plan;
        let rows = &plan.layout.rows;
        // Step 1: Y_s = L_s Θ_s, one product per left sector.
        let mut y = vec![ZERO; plan.y_len];
        let mut y_parts = Vec::with_capacity(rows.len());
        let mut rest = y.as_mut_slice();
        for (k, &(_, _, _, width)) in rows.iter().enumerate() {
            let (head, tail) = rest.split_at_mut(plan.y[k].1 * width);
            y_parts.push(head);
            rest = tail;
        }
        y_parts
            .into_par_iter()
            .zip(rows.par_iter())
            .for_each(|(part, &(s, d, offset, width))| {
                let theta = MatRef::from_row_major_slice(&x[offset..][..d * width], d, width);
                let stack = self.left.stack(s);
                faer_matmul(
                    MatMut::from_row_major_slice_mut(part, stack.nrows(), width),
                    Accum::Replace,
                    stack,
                    theta,
                    ONE,
                    Par::Seq,
                );
            });

        // Steps 2 and 3, one row block of the output per task.
        let mut out_parts = Vec::with_capacity(rows.len());
        let mut rest = out;
        for &(_, d, _, width) in rows {
            let (head, tail) = rest.split_at_mut(d * width);
            out_parts.push(head);
            rest = tail;
        }
        out_parts
            .into_par_iter()
            .zip(rows.par_iter().zip(&plan.out))
            .for_each(|(part, (&(_, d, _, width), blocks))| {
                part.fill(ZERO);
                for block in blocks {
                    let stack = self.right.stack(block.t);
                    if block.copies.is_empty() || stack.nrows() == 0 {
                        continue;
                    }
                    let k = stack.nrows();
                    let mut z = vec![ZERO; d * k];
                    for c in &block.copies {
                        for r in 0..d {
                            let src = &y[c.y + r * c.stride..][..c.cols];
                            let dst = &mut z[r * k + c.z_col..][..c.cols];
                            for (o, v) in dst.iter_mut().zip(src) {
                                *o += c.f * v;
                            }
                        }
                    }
                    let n = stack.ncols();
                    let len = (d - 1) * width + n;
                    // A strided row-major view, built as the transpose of a
                    // column-major one: faer 0.24's
                    // `from_row_major_slice_with_stride_mut` swaps the
                    // strides.
                    let dst = MatMut::from_column_major_slice_with_stride_mut(
                        &mut part[block.column..][..len],
                        n,
                        d,
                        width,
                    )
                    .transpose_mut();
                    faer_matmul(
                        dst,
                        Accum::Replace,
                        MatRef::from_row_major_slice(&z, d, k),
                        stack,
                        ONE,
                        Par::Seq,
                    );
                }
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mps::dmrg::{self, Env};
    use crate::mps::effective::Effective;
    use crate::mps::mpo::Mpo;
    use crate::mps::{Charge, Mps, MpsOptions};
    use crate::observable::{PauliString, PauliSum};

    /// A Heisenberg-like chain with a current and a longer hop, in a state
    /// with several sectors per bond.
    fn setup(n: usize) -> (Mps, PauliSum) {
        let mut h = PauliSum::new();
        let label = |ops: &[(usize, char)]| -> PauliString {
            let s: String = (0..n)
                .rev()
                .map(|k| ops.iter().find(|o| o.0 == k).map_or('I', |o| o.1))
                .collect();
            PauliString::from_label(&s).unwrap()
        };
        for i in 0..n - 1 {
            for p in ['X', 'Y', 'Z'] {
                h.add(0.4 + 0.1 * i as f64, label(&[(i, p), (i + 1, p)]));
            }
            h.add(0.3, label(&[(i, 'X'), (i + 1, 'Y')]));
            h.add(-0.3, label(&[(i, 'Y'), (i + 1, 'X')]));
        }
        h.add(0.5, label(&[(0, 'X'), (1, 'Z'), (2, 'X')]));
        h.add(0.5, label(&[(0, 'Y'), (1, 'Z'), (2, 'Y')]));
        let mut c = crate::circuit::Circuit::new(n);
        for q in (1..n).step_by(2) {
            c.x(q);
        }
        c.pauli_evolution(&h, 0.4, 2, 2).unwrap();
        let mps = Mps::from_circuit(&c, MpsOptions::default()).unwrap();
        (mps, h)
    }

    #[test]
    fn block_environments_and_operators_match_dense_ones() {
        let n = 7;
        let (dense, h) = setup(n);
        let charges = [Charge::weights(vec![1; n])];
        let mut sym = dense.with_charges(&charges).unwrap();
        sym.move_center(3);
        let symmetry = sym.symmetry.clone().unwrap();
        let ch = &symmetry.charges;
        let bonds = &symmetry.bonds;
        let mpo = Mpo::with_charges(&h, n, ch).unwrap();
        let deltas = mpo.channels.as_ref().unwrap();

        let mut left = vec![BlockEnv::boundary(Side::Left, &bonds[0], ch)];
        let mut dense_left = vec![Env::boundary()];
        for i in 0..3 {
            let next = update_left(
                &left[i],
                &sym.tensors[i],
                &mpo.sites[i],
                ch,
                i,
                &bonds[i + 1],
                &deltas[i + 1],
            );
            dense_left.push(dmrg::update_left(
                &dense_left[i],
                &sym.tensors[i],
                &mpo.sites[i],
            ));
            left.push(next);
        }
        let mut right = vec![BlockEnv::boundary(Side::Right, &bonds[n], ch)];
        let mut dense_right = vec![Env::boundary()];
        for i in (5..n).rev() {
            let k = n - 1 - i;
            let next = update_right(
                &right[k],
                &sym.tensors[i],
                &mpo.sites[i],
                ch,
                i,
                &bonds[i],
                &deltas[i],
            );
            dense_right.push(dmrg::update_right(
                &dense_right[k],
                &sym.tensors[i],
                &mpo.sites[i],
            ));
            right.push(next);
        }
        for (b, d) in left
            .iter()
            .zip(&dense_left)
            .chain(right.iter().zip(&dense_right))
        {
            let got = b.to_env();
            for (g, w) in got.blocks.iter().zip(&d.blocks) {
                assert!((g - w).norm_max() < 1e-12, "{:?}", b.side);
            }
            // Block storage is smaller than dense.
            assert!(b.data.len() <= d.blocks.len() * b.bond.dim().pow(2));
        }

        // Two sites 3 and 4, between left[3] (bond 3) and right[2] (bond 5).
        let (l, r) = (sym.tensors[3].left, sym.tensors[4].right);
        let op = BlockOp::two_site(&left[3], &right[2], &mpo.sites[3], &mpo.sites[4], ch, 3);
        let dense_op = Effective::two_site(
            &dense_left[3],
            &dense_right[2],
            &mpo.sites[3],
            &mpo.sites[4],
            l,
            r,
        );
        let theta = dmrg::two_site_vector(&sym.tensors[3], &sym.tensors[4]);
        let x = op.layout().gather(&theta);
        let direct = op
            .layout()
            .two_site(&sym.tensors[3], &sym.tensors[4], &bonds[4], ch, 3);
        assert!(x.iter().zip(&direct).all(|(a, b)| (a - b).norm() < 1e-14));
        assert!(
            (op.layout()
                .scatter(&x)
                .iter()
                .zip(&theta)
                .map(|(a, b)| (a - b).norm())
                .fold(0.0, f64::max))
                < 1e-15
        );
        let mut got = vec![ZERO; x.len()];
        op.apply(&x, &mut got);
        let mut want = vec![ZERO; theta.len()];
        dense_op.apply(&theta, &mut want);
        let got = op.layout().scatter(&got);
        let diff = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).norm())
            .fold(0.0, f64::max);
        assert!(diff < 1e-12, "two-site: {diff}");

        // One site 3, between left[3] and the right environment of bond 4.
        let r4 = update_right(
            &right[2],
            &sym.tensors[4],
            &mpo.sites[4],
            ch,
            4,
            &bonds[4],
            &deltas[4],
        );
        let d4 = dmrg::update_right(&dense_right[2], &sym.tensors[4], &mpo.sites[4]);
        sym.move_center(3);
        let t = &sym.tensors[3];
        let op = BlockOp::one_site(&left[3], &r4, &mpo.sites[3], ch, 3);
        let dense_op = Effective::one_site(&dense_left[3], &d4, &mpo.sites[3], t.left, t.right);
        let x = op.layout().gather(&t.data);
        let mut got = vec![ZERO; x.len()];
        op.apply(&x, &mut got);
        let mut want = vec![ZERO; t.data.len()];
        dense_op.apply(&t.data, &mut want);
        let got = op.layout().scatter(&got);
        let diff = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).norm())
            .fold(0.0, f64::max);
        assert!(diff < 1e-12, "one-site: {diff}");
    }
}
