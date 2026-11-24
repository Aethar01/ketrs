//! Subspace expansion and DMRG noise for symmetric MPS.
//!
//! Both add directions to a bond at zero weight: TDVP's expansion those
//! `H|ψ⟩` reaches (see [`expand`](super::expand)), DMRG's noise those of
//! White's perturbation (see [`dmrg`](super::dmrg)). In a symmetric MPS
//! every new direction must have a definite charge, and each bond must stay
//! sorted by sector. The matrices the dense code factorises once are
//! block-diagonal in the bond's charge, so here they are factorised one
//! sector at a time, in parallel, and the candidates of all sectors are
//! ranked together under the dense code's threshold and room: the same
//! directions a dense factorisation of the block-diagonal matrix would
//! pick. Each sector's new directions go after its old ones.

use super::blocks::SiteBlocks;
use super::charges::{Charges, Label, Sectors, group_cols, group_rows};
use super::dmrg::noise_room;
use super::expand::EXPAND_TOLERANCE;
use super::linalg::{complement_rows, matmul, qr_r, right_svd, thin_svd};
use super::mpo::Mpo;
use super::mpo::{left_keyed, right_keyed};
use super::symmetric::{BlockSvd, Keep};
use super::{Absorb, Mps, Tensor};
use faer::{Mat, MatRef};
use num_complex::Complex64 as C64;
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap};

const ZERO: C64 = C64::new(0.0, 0.0);

/// Sector-limited bond sizes are only worked out while the labels the
/// qubits on one side of a cut can reach stay this few.
const MAX_CAP_LABELS: usize = 1 << 16;

/// The positions of a bond's old basis in its expanded basis.
struct Placement {
    dim: usize,
    pos: Vec<usize>,
}

impl Placement {
    /// `t` with its right bond moved to the expanded basis, zero elsewhere.
    fn right(&self, t: &Tensor) -> Tensor {
        let mut data = vec![ZERO; t.left * 2 * self.dim];
        for lp in 0..t.left * 2 {
            for (c, &p) in self.pos.iter().enumerate() {
                data[lp * self.dim + p] = t.data[lp * t.right + c];
            }
        }
        Tensor {
            left: t.left,
            right: self.dim,
            data,
        }
    }

    /// `t` with its left bond moved to the expanded basis, zero elsewhere.
    fn left(&self, t: &Tensor) -> Tensor {
        let width = 2 * t.right;
        let mut data = vec![ZERO; self.dim * width];
        for (l, &p) in self.pos.iter().enumerate() {
            data[p * width..][..width].copy_from_slice(&t.data[l * width..][..width]);
        }
        Tensor {
            left: self.dim,
            right: t.right,
            data,
        }
    }
}

/// One sector of an expansion: the channel rows with its label and the
/// state's own basis rows in it, both on the sector's columns `cols`.
struct Part {
    label: Label,
    rows: Mat<C64>,
    basis: Mat<C64>,
    cols: Vec<usize>,
}

/// Up to `room` new orthonormal rows over all parts, each within one part:
/// the leading directions of each part's rows outside its basis, ranked
/// across parts, above [`EXPAND_TOLERANCE`] of `scale`. As
/// [`complement`](super::expand::complement), sector by sector.
fn complement_sectors(parts: &[Part], room: usize, scale: f64) -> Vec<Mat<C64>> {
    let factored: Vec<(Vec<f64>, Mat<C64>, Mat<C64>)> = parts
        .par_iter()
        .map(|p| {
            let n = p.cols.len();
            let others = if p.basis.nrows() == 0 {
                Mat::<C64>::identity(n, n)
            } else {
                complement_rows(p.basis.as_ref())
            };
            if others.nrows() == 0 || p.rows.nrows() == 0 || room == 0 {
                return (Vec::new(), Mat::zeros(0, 0), others);
            }
            let outside = matmul(p.rows.as_ref(), others.adjoint());
            let compact = if outside.nrows() > outside.ncols() {
                qr_r(outside.as_ref())
            } else {
                outside
            };
            let (s, v) = right_svd(compact.as_ref());
            (s, v, others)
        })
        .collect();
    let mut ranked: Vec<(f64, usize, usize)> = factored
        .iter()
        .enumerate()
        .flat_map(|(k, (s, _, _))| s.iter().enumerate().map(move |(j, &x)| (x, k, j)))
        .filter(|r| r.0 > EXPAND_TOLERANCE * scale)
        .collect();
    ranked.sort_by(|x, y| y.0.total_cmp(&x.0).then((x.1, x.2).cmp(&(y.1, y.2))));
    ranked.truncate(room);
    let mut counts = vec![0; parts.len()];
    for &(_, k, _) in &ranked {
        counts[k] += 1;
    }
    factored
        .iter()
        .zip(&counts)
        .zip(parts)
        .map(|(((_, v, others), &k), p)| {
            if k == 0 {
                return Mat::zeros(0, p.cols.len());
            }
            let lead = Mat::<C64>::from_fn(k, v.nrows(), |r, c| v[(c, r)].conj());
            matmul(lead.as_ref(), others.as_ref())
        })
        .collect()
}

/// `old` with the directions `added` (one matrix of rows per label, over
/// the full columns) put after each sector's own: the new sectors, the
/// old basis's positions, and the new basis as rows (old rows from `own`,
/// a matrix of rows on the old bond).
fn grow(
    old: &Sectors,
    own: MatRef<'_, C64>,
    added: &[(Label, Mat<C64>)],
) -> (Sectors, Placement, Mat<C64>) {
    let mut sizes: BTreeMap<Label, (Option<usize>, usize)> = BTreeMap::new();
    for k in 0..old.len() {
        sizes.insert(old.label(k).into(), (Some(k), 0));
    }
    let mut by_label: HashMap<&[i64], &Mat<C64>> = HashMap::new();
    for (label, rows) in added {
        if rows.nrows() > 0 {
            sizes.entry(label.clone()).or_insert((None, 0)).1 = rows.nrows();
            by_label.insert(label, rows);
        }
    }
    let dim = old.dim() + added.iter().map(|(_, m)| m.nrows()).sum::<usize>();
    let cols = own.ncols();
    let mut out = Mat::<C64>::zeros(dim, cols);
    let mut pos = vec![0; old.dim()];
    let mut entries = Vec::with_capacity(sizes.len());
    let mut at = 0;
    for (label, (sector, extra)) in &sizes {
        let start = at;
        if let Some(k) = *sector {
            for r in old.range(k) {
                pos[r] = at;
                out.as_mut().row_mut(at).copy_from(own.row(r));
                at += 1;
            }
        }
        if let Some(rows) = by_label.get(&**label) {
            out.as_mut().subrows_mut(at, rows.nrows()).copy_from(*rows);
            at += rows.nrows();
        }
        debug_assert_eq!(at - start, old_size(old, *sector) + extra);
        entries.push((label.clone(), at - start));
    }
    (Sectors::new(entries), Placement { dim, pos }, out)
}

fn old_size(old: &Sectors, sector: Option<usize>) -> usize {
    sector.map_or(0, |k| old.size(k))
}

/// The mixed environment of an expansion sweep, block by block:
/// `chain[channel][ket sector] = (bra sector, block)`, each block with rows
/// on the expanded basis and columns on ψ's.
type Chain = Vec<Vec<Option<(usize, Mat<C64>)>>>;

/// The chain beyond an end of the state.
fn boundary_chain() -> Chain {
    vec![vec![Some((0, Mat::identity(1, 1)))]]
}

/// A part of one row block of `H|ψ⟩`'s channels: its columns of physical
/// state `p` and bra sector `b`.
struct Piece {
    p: usize,
    b: usize,
    m: Mat<C64>,
}

/// One bond of an expansion sweep, either way. The channel rows are blocks
/// `(channel, sector k of bond)` labelled `k − δ_channel`, given by
/// `pieces`; `cols` groups the `width` columns of the expanded side by
/// label, `index(p, b, j)` is column `j` of bra sector `b` with physical
/// state `p`, and `basis` holds ψ's own rows (in `bond`'s order). Adds up
/// to `room` directions and returns the new sectors, where the old basis
/// went, the new basis as rows, and the next chain.
#[allow(clippy::too_many_arguments)]
fn chain_step(
    bond: &Sectors,
    deltas: &[Label],
    charges: &Charges,
    cols: Vec<(Label, Vec<usize>)>,
    basis: MatRef<'_, C64>,
    room: usize,
    index: impl Fn(usize, usize, usize) -> usize + Sync,
    pieces: impl Fn(usize, usize) -> Vec<Piece> + Sync,
) -> (Sectors, Placement, Mat<C64>, Chain) {
    let width = basis.ncols();
    let mut rows: BTreeMap<Label, Vec<(usize, usize)>> = BTreeMap::new();
    for (ch, d) in deltas.iter().enumerate() {
        for k in 0..bond.len() {
            rows.entry(charges.sub(bond.label(k), d))
                .or_default()
                .push((ch, k));
        }
    }
    // Each label's channel rows on its columns, and ψ's rows there.
    let built: Vec<(Part, Vec<(usize, usize)>)> = cols
        .into_par_iter()
        .filter_map(|(label, cols)| {
            let blocks = rows.get(&label)?.clone();
            let height: usize = blocks.iter().map(|&(_, k)| bond.size(k)).sum();
            let mut m = Mat::<C64>::zeros(height, cols.len());
            let mut at = 0;
            for &(ch, k) in &blocks {
                for piece in pieces(ch, k) {
                    for j in 0..piece.m.ncols() {
                        let col = index(piece.p, piece.b, j);
                        let y = cols.binary_search(&col).expect("a column of this label");
                        for r in 0..piece.m.nrows() {
                            m[(at + r, y)] = piece.m[(r, j)];
                        }
                    }
                }
                at += bond.size(k);
            }
            let own = bond.find(&label).map_or(0..0, |k| bond.range(k));
            let b = Mat::<C64>::from_fn(own.len(), cols.len(), |x, y| {
                basis[(own.start + x, cols[y])]
            });
            Some((
                Part {
                    label,
                    rows: m,
                    basis: b,
                    cols,
                },
                blocks,
            ))
        })
        .collect();
    let (parts, blocks): (Vec<Part>, Vec<Vec<(usize, usize)>>) = built.into_iter().unzip();
    let scale = parts
        .iter()
        .map(|p| p.rows.norm_l2().powi(2))
        .sum::<f64>()
        .sqrt();
    let new = if scale > 0.0 {
        complement_sectors(&parts, room, scale)
    } else {
        parts.iter().map(|p| Mat::zeros(0, p.cols.len())).collect()
    };
    let added = scatter(&parts, new, width);
    let (sectors, placement, grown) = grow(bond, basis, &added);
    // The next chain: per channel block, its rows against the new basis.
    let mut next: Chain = vec![vec![None; bond.len()]; deltas.len()];
    let products: Vec<_> = parts
        .par_iter()
        .zip(&blocks)
        .filter_map(|(part, blocks)| {
            let k_new = sectors.find(&part.label)?;
            let range = sectors.range(k_new);
            let e = Mat::<C64>::from_fn(range.len(), part.cols.len(), |x, y| {
                grown[(range.start + x, part.cols[y])]
            });
            let g = matmul(part.rows.as_ref(), e.adjoint());
            let mut at = 0;
            let out: Vec<_> = blocks
                .iter()
                .map(|&(ch, k)| {
                    let block = g.as_ref().subrows(at, bond.size(k)).transpose().to_owned();
                    at += bond.size(k);
                    (ch, k, k_new, block)
                })
                .collect();
            Some(out)
        })
        .collect();
    for (ch, k, k_new, block) in products.into_iter().flatten() {
        next[ch][k] = Some((k_new, block));
    }
    (sectors, placement, grown, next)
}

/// The new rows of each part, scattered onto all `width` columns.
fn scatter(parts: &[Part], new: Vec<Mat<C64>>, width: usize) -> Vec<(Label, Mat<C64>)> {
    parts
        .iter()
        .zip(new)
        .map(|(p, m)| {
            let mut full = Mat::<C64>::zeros(m.nrows(), width);
            for r in 0..m.nrows() {
                for (y, &c) in p.cols.iter().enumerate() {
                    full[(r, c)] = m[(r, y)];
                }
            }
            (p.label.clone(), full)
        })
        .collect()
}

impl Mps {
    /// The largest dimension each bond (left of each site, and the ends)
    /// can have within the state's sector, capped at `max_bond`: for each
    /// label, the fewer of the basis states left of the cut with it and
    /// right of it that complete the total charge.
    pub(super) fn sector_caps(&self) -> Vec<usize> {
        let sym = self.symmetry.as_ref().expect("a symmetric MPS");
        let (n, cap) = (self.n_qubits(), self.options.max_bond);
        let ch = &sym.charges;
        let total = sym.bonds[n].label(0);
        // The range each U(1) charge can still change by from qubit q on:
        // labels that can't reach the total from there are pruned.
        let k = total.len();
        let mut reach = vec![(vec![0i64; k], vec![0i64; k]); n + 1];
        for q in (0..n).rev() {
            let (mut lo, mut hi) = reach[q + 1].clone();
            for c in 0..k {
                let (a, b) = (ch.site(q, 0)[c], ch.site(q, 1)[c]);
                lo[c] += a.min(b);
                hi[c] += a.max(b);
            }
            reach[q] = (lo, hi);
        }
        let moduli: Vec<bool> = ch.charges().iter().map(|c| c.modulus().is_some()).collect();
        let fits = |label: &[i64], q: usize| {
            (0..k).all(|c| {
                let need = total[c] - label[c];
                moduli[c] || (reach[q].0[c] <= need && need <= reach[q].1[c])
            })
        };
        let mut forward: Vec<HashMap<Label, usize>> = vec![HashMap::from([(ch.zero(), 1)])];
        for q in 0..n {
            let mut next: HashMap<Label, usize> = HashMap::new();
            for (label, &count) in &forward[q] {
                for p in 0..2 {
                    let l = ch.add(label, ch.site(q, p));
                    if fits(&l, q + 1) {
                        let e = next.entry(l).or_insert(0);
                        *e = (*e + count).min(cap);
                    }
                }
            }
            if next.len() > MAX_CAP_LABELS {
                return vec![cap; n + 1];
            }
            forward.push(next);
        }
        let mut backward: Vec<HashMap<Label, usize>> = vec![HashMap::new(); n + 1];
        backward[n].insert(total.into(), 1);
        for q in (0..n).rev() {
            let mut next: HashMap<Label, usize> = HashMap::new();
            for (label, &count) in &backward[q + 1] {
                for p in 0..2 {
                    let l = ch.sub(label, ch.site(q, p));
                    if forward[q].contains_key(&l) {
                        let e = next.entry(l).or_insert(0);
                        *e = (*e + count).min(cap);
                    }
                }
            }
            backward[q] = next;
        }
        (0..=n)
            .map(|i| {
                let both: usize = forward[i]
                    .iter()
                    .map(|(l, &f)| backward[i].get(l).map_or(0, |&b| f.min(b)))
                    .sum();
                both.min(cap)
            })
            .collect()
    }

    /// [`expand_right_bases`](Mps::expand_right_bases) for a symmetric MPS,
    /// with its mixed environment block-sparse.
    pub(super) fn expand_right_symmetric(&mut self, mpo: &Mpo) {
        let n = self.n_qubits();
        self.move_center(0);
        let caps = self.sector_caps();
        let deltas = mpo.channels.as_ref().expect("a charge-graded MPO");
        let charges = self.symmetry.as_ref().unwrap().charges.clone();
        let mut chain = boundary_chain();
        let mut placed: Option<Placement> = None;
        // ψ's own sectors of the bond right of the site.
        let mut ket_right = self.symmetry.as_ref().unwrap().bonds[n].clone();
        for i in (1..n).rev() {
            let ket = self.tensors[i].clone();
            let padded = placed
                .as_ref()
                .map_or_else(|| ket.clone(), |p| p.right(&ket));
            let sym = self.symmetry.as_ref().unwrap();
            let bond = sym.bonds[i].clone();
            let bra_right = &sym.bonds[i + 1];
            let blocks = SiteBlocks::new(&ket, &bond, &ket_right, &charges, i);
            let rows = left_keyed(&mpo.sites[i]);
            let dim = bra_right.dim();
            let pieces = |a: usize, s: usize| -> Vec<Piece> {
                let label = charges.sub(bond.label(s), &deltas[i][a]);
                (0..2)
                    .filter_map(|p_out| {
                        let t_bra = bra_right.find(&charges.add(&label, charges.site(i, p_out)))?;
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
            let cols = group_cols(bra_right, &charges, i);
            let room = caps[i].min(2 * dim).saturating_sub(ket.left);
            let (sectors, placement, grown, next) = chain_step(
                &bond,
                &deltas[i],
                &charges,
                cols,
                padded.as_right_matrix(),
                room,
                index,
                pieces,
            );
            self.tensors[i] = Tensor::from_matrix(grown.as_ref(), placement.dim, padded.right);
            self.symmetry.as_mut().unwrap().bonds[i] = sectors;
            chain = next;
            ket_right = bond;
            placed = Some(placement);
        }
        if let Some(p) = placed {
            self.tensors[0] = p.right(&self.tensors[0]);
        }
        self.center = 0;
    }

    /// [`expand_left_bases`](Mps::expand_left_bases) for a symmetric MPS,
    /// with its mixed environment block-sparse.
    pub(super) fn expand_left_symmetric(&mut self, mpo: &Mpo) {
        let n = self.n_qubits();
        self.move_center(n - 1);
        let caps = self.sector_caps();
        let deltas = mpo.channels.as_ref().expect("a charge-graded MPO");
        let charges = self.symmetry.as_ref().unwrap().charges.clone();
        let mut chain = boundary_chain();
        let mut placed: Option<Placement> = None;
        // ψ's own sectors of the bond left of the site.
        let mut ket_left = self.symmetry.as_ref().unwrap().bonds[0].clone();
        for i in 0..n - 1 {
            let ket = self.tensors[i].clone();
            let padded = placed
                .as_ref()
                .map_or_else(|| ket.clone(), |p| p.left(&ket));
            let sym = self.symmetry.as_ref().unwrap();
            let bond = sym.bonds[i + 1].clone();
            let bra_left = &sym.bonds[i];
            let blocks = SiteBlocks::new(&ket, &ket_left, &bond, &charges, i);
            let rows = right_keyed(&mpo.sites[i]);
            let pieces = |b: usize, t: usize| -> Vec<Piece> {
                let label = charges.sub(bond.label(t), &deltas[i + 1][b]);
                (0..2)
                    .filter_map(|p_out| {
                        let s_bra = bra_left.find(&charges.sub(&label, charges.site(i, p_out)))?;
                        let mut acc = Mat::<C64>::zeros(bond.size(t), bra_left.size(s_bra));
                        let mut any = false;
                        for &(a, p_in, f) in &rows[b * 2 + p_out] {
                            let Some(s_ket) = blocks.left_of[t * 2 + p_in] else {
                                continue;
                            };
                            let (_, blk) = blocks.block(s_ket, p_in).expect("the block reaches t");
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
            let cols = group_rows(bra_left, &charges, i);
            let width = 2 * padded.left;
            let room = caps[i + 1].min(width).saturating_sub(ket.right);
            // Transposed, ψ's orthonormal columns are orthonormal rows.
            let (sectors, placement, grown, next) = chain_step(
                &bond,
                &deltas[i + 1],
                &charges,
                cols,
                padded.as_left_matrix().transpose(),
                room,
                index,
                pieces,
            );
            self.tensors[i] = Tensor::from_matrix(grown.transpose(), padded.left, placement.dim);
            self.symmetry.as_mut().unwrap().bonds[i + 1] = sectors;
            chain = next;
            ket_left = bond;
            placed = Some(placement);
        }
        if let Some(p) = placed {
            self.tensors[n - 1] = p.left(&self.tensors[n - 1]);
        }
        self.center = n - 1;
    }

    /// [`split_perturbed`](Mps::split_perturbed) for a symmetric MPS: θ's
    /// kept basis by blocks, and the noise directions per sector of the
    /// side being traced out, ranked across sectors.
    pub(super) fn split_perturbed_symmetric(
        &mut self,
        i: usize,
        theta: MatRef<'_, C64>,
        perturbation: &[Mat<C64>],
        noise: f64,
        absorb: Absorb,
    ) {
        let sym = self.symmetry.as_ref().expect("a symmetric MPS");
        let ch = sym.charges.clone();
        let (rows, cols) = (theta.nrows(), theta.ncols());
        let row_groups = group_rows(&sym.bonds[i], &ch, i);
        let col_groups = group_cols(&sym.bonds[i + 2], &ch, i + 1);
        let side_groups = match absorb {
            Absorb::Right => row_groups.clone(),
            Absorb::Left => col_groups.clone(),
        };
        let svd = BlockSvd::new(theta, row_groups, col_groups);
        let counts = svd.counts(Keep::Truncate { floor: 0.0 }, &self.options);
        let total = svd.weight();
        let keep: usize = counts.iter().sum();
        let side = match absorb {
            Absorb::Right => rows,
            Absorb::Left => cols,
        };
        // θ's kept basis in each side group, as columns over its indices.
        let own: Vec<Mat<C64>> = side_groups
            .iter()
            .map(|(label, idx)| match svd.labels.binary_search(label) {
                Ok(b) => {
                    let m = match absorb {
                        Absorb::Right => &svd.u[b],
                        Absorb::Left => &svd.v[b],
                    };
                    m.as_ref().subcols(0, counts[b]).to_owned()
                }
                Err(_) => Mat::zeros(idx.len(), 0),
            })
            .collect();
        let room = noise_room(keep)
            .min(self.options.max_bond.saturating_sub(keep))
            .min(rows.min(cols).saturating_sub(keep));
        let weight = |m: MatRef<'_, C64>| m.norm_l2().powi(2);
        let extra: Vec<Mat<C64>> = if room > 0 && total > 0.0 {
            // As the dense split: each channel scaled to weight `noise`,
            // ρ = Σ_c P_c P_c† on the side traced out, block by block.
            let blocks: Vec<Mat<C64>> = perturbation
                .iter()
                .filter_map(|p| {
                    let norm_p = p.as_ref().norm_l2();
                    if !norm_p.is_finite() || norm_p <= 1e-100 * total.sqrt() {
                        return None;
                    }
                    let scale = (noise * total).sqrt() / norm_p;
                    if !scale.is_finite() {
                        return None;
                    }
                    Some(match absorb {
                        Absorb::Right => {
                            Mat::<C64>::from_fn(side, p.ncols(), |r, c| p[(r, c)] * scale)
                        }
                        Absorb::Left => {
                            Mat::<C64>::from_fn(side, p.nrows(), |r, c| p[(c, r)].conj() * scale)
                        }
                    })
                })
                .collect();
            let width: usize = blocks.iter().map(|b| b.ncols()).sum();
            let found: Vec<(Mat<C64>, Vec<f64>)> = side_groups
                .par_iter()
                .zip(&own)
                .map(|((_, idx), own)| {
                    let mut stacked = Mat::<C64>::zeros(idx.len(), width);
                    let mut at = 0;
                    for b in &blocks {
                        for (x, &r) in idx.iter().enumerate() {
                            for c in 0..b.ncols() {
                                stacked[(x, at + c)] = b[(r, c)];
                            }
                        }
                        at += b.ncols();
                    }
                    let rho = matmul(stacked.as_ref(), stacked.adjoint());
                    let project = |m: MatRef<'_, C64>| -> Mat<C64> {
                        let mut out = m.to_owned();
                        out -= matmul(own.as_ref(), matmul(own.adjoint(), m).as_ref());
                        out
                    };
                    let rho = project(project(rho.as_ref()).adjoint().to_owned().as_ref());
                    let (vectors, weights, _) = thin_svd(rho.as_ref());
                    (vectors, weights)
                })
                .collect();
            let mut ranked: Vec<(f64, usize, usize)> = found
                .iter()
                .enumerate()
                .flat_map(|(g, (_, w))| w.iter().enumerate().map(move |(j, &x)| (x, g, j)))
                .filter(|r| r.0 > self.options.cutoff * total)
                .collect();
            ranked.sort_by(|x, y| y.0.total_cmp(&x.0).then((x.1, x.2).cmp(&(y.1, y.2))));
            ranked.truncate(room);
            let mut take = vec![0; side_groups.len()];
            for &(_, g, _) in &ranked {
                take[g] += 1;
            }
            found
                .into_par_iter()
                .zip(own.par_iter().zip(take.into_par_iter()))
                .map(|((vectors, _), (own, k))| {
                    if k == 0 {
                        return Mat::zeros(own.nrows(), 0);
                    }
                    // Orthonormalise again against θ's basis, for rounding.
                    let mut c = vectors.as_ref().subcols(0, k).to_owned();
                    c -= matmul(own.as_ref(), matmul(own.adjoint(), c.as_ref()).as_ref());
                    let (q, sv, _) = thin_svd(c.as_ref());
                    let count = sv.iter().take_while(|&&x| x > 0.5).count();
                    q.as_ref().subcols(0, count).to_owned()
                })
                .collect()
        } else {
            own.iter().map(|o| Mat::zeros(o.nrows(), 0)).collect()
        };

        // The new bond: each group's kept basis, then its new directions.
        let bond: usize = own
            .iter()
            .zip(&extra)
            .map(|(o, e)| o.ncols() + e.ncols())
            .sum();
        let mut basis = Mat::<C64>::zeros(side, bond);
        let mut entries = Vec::new();
        let mut at = 0;
        for (((label, idx), o), e) in side_groups.iter().zip(&own).zip(&extra) {
            let start = at;
            for m in [o, e] {
                for c in 0..m.ncols() {
                    for (x, &r) in idx.iter().enumerate() {
                        basis[(r, at)] = m[(x, c)];
                    }
                    at += 1;
                }
            }
            entries.push((label.clone(), at - start));
        }
        let (left, right) = (rows / 2, cols / 2);
        let centre = match absorb {
            Absorb::Right => matmul(basis.adjoint(), theta),
            Absorb::Left => matmul(theta, basis.as_ref()),
        };
        let kept = weight(centre.as_ref());
        if total > 0.0 {
            self.truncation_error += ((total - kept) / total).max(0.0);
        }
        let rescale = if kept > 0.0 {
            C64::new((total / kept).sqrt(), 0.0)
        } else {
            C64::new(1.0, 0.0)
        };
        let centre = Mat::<C64>::from_fn(centre.nrows(), centre.ncols(), |r, c| {
            rescale * centre[(r, c)]
        });
        match absorb {
            Absorb::Right => {
                self.tensors[i] = Tensor::from_matrix(basis.as_ref(), left, bond);
                self.tensors[i + 1] = Tensor::from_matrix(centre.as_ref(), bond, right);
                self.center = i + 1;
            }
            Absorb::Left => {
                let basis = basis.adjoint().to_owned();
                self.tensors[i] = Tensor::from_matrix(centre.as_ref(), left, bond);
                self.tensors[i + 1] = Tensor::from_matrix(basis.as_ref(), bond, right);
                self.center = i;
            }
        }
        self.symmetry.as_mut().unwrap().bonds[i + 1] = Sectors::new(entries);
    }
}

#[cfg(test)]
mod tests {
    use crate::mps::{Charge, DmrgOptions, Mps, MpsOptions};
    use crate::observable::{PauliString, PauliSum};
    use crate::statevector::Statevector;
    use num_complex::Complex64 as C64;

    fn label(n: usize, ops: &[(usize, char)]) -> PauliString {
        let s: String = (0..n)
            .rev()
            .map(|k| ops.iter().find(|o| o.0 == k).map_or('I', |o| o.1))
            .collect();
        PauliString::from_label(&s).unwrap()
    }

    /// The Fermi–Hubbard chain of `sites` sites through a Jordan–Wigner
    /// string, qubit `2j` for spin up at site `j` and `2j + 1` for down:
    /// hops span three qubits.
    fn hubbard(sites: usize, u: f64) -> (PauliSum, Vec<Charge>) {
        let n = 2 * sites;
        let mut h = PauliSum::new();
        for j in 0..sites - 1 {
            for s in 0..2 {
                let (a, b) = (2 * j + s, 2 * j + s + 2);
                let t = -0.5 * (1.0 + 0.1 * j as f64);
                for p in ['X', 'Y'] {
                    h.add(t, label(n, &[(a, p), (a + 1, 'Z'), (b, p)]));
                }
            }
        }
        for j in 0..sites {
            let (a, b) = (2 * j, 2 * j + 1);
            h.add(-u / 4.0, label(n, &[(a, 'Z')]));
            h.add(-u / 4.0, label(n, &[(b, 'Z')]));
            h.add(u / 4.0, label(n, &[(a, 'Z'), (b, 'Z')]));
            h.add(0.05 * j as f64, label(n, &[(a, 'Z')]));
        }
        let up = Charge::weights((0..n).map(|q| i64::from(q % 2 == 0)).collect());
        let down = Charge::weights((0..n).map(|q| i64::from(q % 2 == 1)).collect());
        (h, vec![up, down])
    }

    fn filled(n: usize) -> Vec<bool> {
        let mut bits = vec![false; n];
        for q in [0, 3, 4, 7] {
            bits[q] = true;
        }
        bits
    }

    fn close(a: &[C64], b: &[C64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).norm())
            .fold(0.0, f64::max)
    }

    #[test]
    fn sector_caps_count_states_on_both_sides() {
        let n = 6;
        let neel: Vec<bool> = (0..n).map(|q| q % 2 == 1).collect();
        let sym = Mps::basis(n, &neel, MpsOptions::default())
            .with_charges(&[Charge::weights(vec![1; n])])
            .unwrap();
        // Bond 3 holds every split of three particles: 1 + 3 + 3 + 1.
        assert_eq!(sym.sector_caps(), vec![1, 2, 4, 8, 4, 2, 1]);
        let mut capped = sym.clone();
        capped.set_options(MpsOptions {
            max_bond: 3,
            cutoff: 0.0,
        });
        assert_eq!(capped.sector_caps(), vec![1, 2, 3, 3, 3, 2, 1]);
    }

    #[test]
    fn tdvp_from_a_product_state_matches_dense() {
        let (h, charges) = hubbard(4, 2.0);
        let n = 8;
        let start = Mps::basis(n, &filled(n), MpsOptions::default());
        let mut dense = start.clone();
        let mut sym = start.with_charges(&charges).unwrap();
        dense.tdvp(&h, 0.8, 8).unwrap();
        sym.tdvp(&h, 0.8, 8).unwrap();
        assert!(sym.charge_violation() < 1e-12);
        assert_eq!(sym.total_charge(), Some(vec![2, 2]));
        let diff = close(&sym.to_amplitudes(), &dense.to_amplitudes());
        assert!(diff < 1e-8, "{diff}");
        let mut exact = Statevector::from_amplitudes(start_amplitudes(n)).unwrap();
        exact.evolve(&h, 0.8).unwrap();
        let err = close(&sym.to_amplitudes(), exact.amplitudes());
        assert!(err < 1e-2, "{err}");
    }

    fn start_amplitudes(n: usize) -> Vec<C64> {
        Mps::basis(n, &filled(n), MpsOptions::default()).to_amplitudes()
    }

    #[test]
    fn noisy_dmrg_finds_the_sector_ground_state() {
        let (h, charges) = hubbard(4, 2.0);
        let n = 8;
        let start = Mps::basis(n, &filled(n), MpsOptions::default());
        let options = DmrgOptions {
            max_bond: 32,
            ..Default::default()
        };
        let mut dense = start.clone();
        let want = dense.dmrg(&h, &options).unwrap();
        let mut sym = start.with_charges(&charges).unwrap();
        let got = sym.dmrg(&h, &options).unwrap();
        assert!(got.converged);
        assert!(
            (got.energy - want.energy).abs() < 1e-8,
            "{} vs {}",
            got.energy,
            want.energy
        );
        assert!(sym.charge_violation() < 1e-12);
        // Without noise, the three-qubit hops can't act on the product state.
        let mut stuck = start.with_charges(&charges).unwrap();
        let plain = stuck
            .dmrg(
                &h,
                &DmrgOptions {
                    noise: 0.0,
                    ..options
                },
            )
            .unwrap();
        assert!(plain.energy > got.energy + 0.1);
    }

    #[test]
    fn expansion_keeps_the_state_and_its_sectors() {
        let (h, charges) = hubbard(4, 2.0);
        let n = 8;
        let start = Mps::basis(n, &filled(n), MpsOptions::default());
        let mut sym = start.with_charges(&charges).unwrap();
        let mpo = sym.mpo(&h).unwrap();
        sym.expand_right_symmetric(&mpo);
        assert!(sym.bond_dimensions().iter().any(|&d| d > 1));
        assert!(sym.charge_violation() < 1e-14);
        assert!(close(&sym.to_amplitudes(), &start.to_amplitudes()) < 1e-14);
        sym.expand_left_symmetric(&mpo);
        assert!(sym.charge_violation() < 1e-14);
        assert!(close(&sym.to_amplitudes(), &start.to_amplitudes()) < 1e-14);
        // Every bond is orthonormal on the side away from the centre.
        sym.move_center(0);
        sym.move_center(n - 1);
        assert!(close(&sym.to_amplitudes(), &start.to_amplitudes()) < 1e-12);
    }

    #[test]
    fn subnormal_perturbation_channel_does_not_produce_nan() {
        // Regression test: an MPO perturbation block with entries around 1e-160
        // has squared norm ~1e-320 (non-zero subnormal), so dividing noise / w
        // overflowed to +inf and produced 0.0 * inf = NaN in rho before thin_svd.
        let n = 4;
        let bits = [true, false, true, false];
        let mut dense = Mps::basis(n, &bits, MpsOptions::default());
        let mut sym = dense
            .with_charges(&[Charge::weights(vec![1; n])])
            .unwrap();
        let theta = faer::Mat::<C64>::from_fn(2, 2, |r, c| {
            if r == 1 && c == 0 {
                C64::new(1.0, 0.0)
            } else {
                C64::new(0.0, 0.0)
            }
        });
        let mut tiny = faer::Mat::<C64>::zeros(2, 2);
        tiny[(0, 0)] = C64::new(1e-160, 0.0);
        for absorb in [super::Absorb::Right, super::Absorb::Left] {
            let mut d = dense.clone();
            d.split_perturbed(0, theta.as_ref(), &[tiny.clone()], 1e-4, absorb);
            assert!(d.norm_sqr().is_finite());

            let mut s = sym.clone();
            s.split_perturbed_symmetric(0, theta.as_ref(), &[tiny.clone()], 1e-4, absorb);
            assert!(s.norm_sqr().is_finite());
            assert!(s.charge_violation() < 1e-14);
        }
        // Also verify zero-norm perturbation blocks and zero-norm theta stay finite.
        let zero = faer::Mat::<C64>::zeros(2, 2);
        dense.split_perturbed(0, theta.as_ref(), std::slice::from_ref(&zero), 1e-4, super::Absorb::Right);
        sym.split_perturbed_symmetric(0, theta.as_ref(), &[zero], 1e-4, super::Absorb::Right);
        assert!((dense.norm_sqr() - 1.0).abs() < 1e-14);
        assert!((sym.norm_sqr() - 1.0).abs() < 1e-14);
    }
}
