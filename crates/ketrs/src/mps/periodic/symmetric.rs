//! Abelian charge symmetry (`U(1)` and `Z_n`) for periodic-trace MPS.
//!
//! On a closed ring of `n` qubits in a definite total-charge sector `Q_tot`,
//! every bond `i` (left of site `i`, with `bonds[0]` the seam bond between
//! `n - 1` and `0`) is split into contiguous charge sectors. Site `q`'s tensor
//! obeys the exact selection rule `q_{(q+1) % n} = q_q + Δ_q(p)`, where
//! `Δ_q(p) = c_q(p)` for `q < n - 1` and `Δ_{n-1}(p) = c_{n-1}(p) - Q_tot`.
//! Around any closed loop `Σ_q Δ_q(b_q) = Σ_q c_q(b_q) - Q_tot = 0`, so every
//! non-zero trace term lies strictly in the `Q_tot` sector, while `bonds[0]`
//! carries multiple sectors for particles or gauge fluxes crossing the seam.

use super::PeriodicMps;
use crate::error::Error;
use crate::mps::charges::{Charge, Charges, Groups, Label, Sectors};
use crate::mps::linalg::{hermitian_evd, matmul};
use crate::mps::symmetric::{BlockSvd, Keep, MAX_CHARGE_LEAK};
use crate::mps::{Absorb, MpsOptions, Tensor, ZERO};
use faer::{Mat, MatRef};
use num_complex::Complex64 as C64;
use std::collections::BTreeMap;
use std::sync::Arc;

/// Singular values below this times the norm are rounding error during
/// `with_charges` conversion.
const ZERO_SINGULAR: f64 = 1e-13;

/// Conserved charges and bond sector tables of a symmetric [`PeriodicMps`].
#[derive(Clone, Debug)]
pub(super) struct PeriodicSymmetry {
    pub(super) charges: Arc<Charges>,
    /// `bonds[i]` is the bond left of site `i` (`bonds[0]` is the seam bond
    /// between site `n - 1` and site `0`).
    pub(super) bonds: Vec<Sectors>,
    /// The state's fixed total charge `Q_tot`, subtracted at site `n - 1`.
    pub(super) total: Label,
}

impl PeriodicSymmetry {
    pub(super) fn n_qubits(&self) -> usize {
        self.bonds.len()
    }

    /// The bond left of site `q`.
    pub(super) fn left_bond(&self, q: usize) -> &Sectors {
        &self.bonds[q]
    }

    /// The bond right of site `q` (wrapping to `bonds[0]` at `q == n - 1`).
    pub(super) fn right_bond(&self, q: usize) -> &Sectors {
        &self.bonds[(q + 1) % self.n_qubits()]
    }

    /// The bond charge shift `Δ_q(p)` across site `q` in physical state `p`:
    /// `c_q(p)` for `q < n - 1`, and `c_{n-1}(p) - Q_tot` at the seam site
    /// `q == n - 1`.
    pub(super) fn site_shift(&self, q: usize, p: usize) -> Label {
        let base = self.charges.site(q, p);
        if q + 1 == self.n_qubits() {
            self.charges.sub(base, &self.total)
        } else {
            base.into()
        }
    }

    /// Rows `l * 2 + p` of site `q`'s left matrix grouped by the right bond
    /// label `s(l) + Δ_q(p)`.
    pub(super) fn group_rows(&self, left: &Sectors, q: usize) -> Groups {
        let mut groups: BTreeMap<Label, Vec<usize>> = BTreeMap::new();
        let shift = [self.site_shift(q, 0), self.site_shift(q, 1)];
        for k in 0..left.len() {
            for (p, sh) in shift.iter().enumerate() {
                let label = self.charges.add(left.label(k), sh);
                let indices = groups.entry(label).or_default();
                indices.extend(left.range(k).map(|l| l * 2 + p));
            }
        }
        sorted_groups(groups)
    }

    /// Columns `p * dim + r` of site `q`'s right matrix grouped by the left
    /// bond label `t(r) - Δ_q(p)`.
    pub(super) fn group_cols(&self, right: &Sectors, q: usize) -> Groups {
        let mut groups: BTreeMap<Label, Vec<usize>> = BTreeMap::new();
        let dim = right.dim();
        let shift = [self.site_shift(q, 0), self.site_shift(q, 1)];
        for k in 0..right.len() {
            for (p, sh) in shift.iter().enumerate() {
                let label = self.charges.sub(right.label(k), sh);
                let indices = groups.entry(label).or_default();
                indices.extend(right.range(k).map(|r| p * dim + r));
            }
        }
        sorted_groups(groups)
    }
}

fn sorted_groups(groups: BTreeMap<Label, Vec<usize>>) -> Groups {
    groups
        .into_iter()
        .map(|(label, mut indices)| {
            indices.sort_unstable();
            (label, indices)
        })
        .collect()
}

/// Allowed blocks of a periodic site tensor `a` at site `q`, using
/// `sym.site_shift(q, p)` (so `q == n - 1` handles the `-Q_tot` wrap).
pub(super) struct PeriodicSiteBlocks<'a> {
    a: &'a Tensor,
    pub(super) left: &'a Sectors,
    pub(super) right: &'a Sectors,
    /// `right_of[s * 2 + p]` = right sector `t` with `t = s + Δ_q(p)`.
    pub(super) right_of: Vec<Option<usize>>,
}

impl<'a> PeriodicSiteBlocks<'a> {
    pub(super) fn new(
        a: &'a Tensor,
        left: &'a Sectors,
        right: &'a Sectors,
        sym: &PeriodicSymmetry,
        q: usize,
    ) -> Self {
        debug_assert_eq!((left.dim(), right.dim()), (a.left, a.right));
        let shift = [sym.site_shift(q, 0), sym.site_shift(q, 1)];
        let right_of = (0..left.len() * 2)
            .map(|k| right.find(&sym.charges.add(left.label(k / 2), &shift[k % 2])))
            .collect();
        PeriodicSiteBlocks {
            a,
            left,
            right,
            right_of,
        }
    }

    /// The `d_s × d_t` block from left sector `s` with physical state `p`.
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

/// Block-sparse QR orthonormalization of a left matrix `A` of shape
/// `(left_bond * 2) × mid_bond` with sectors `left` and `mid`, returning
/// `(Q, R, new_mid_sectors)` where `Q` is strictly left-orthonormal and
/// `Q R = A` within every charge sector.
pub(super) fn qr_left_symmetric(
    a: MatRef<'_, C64>,
    left: &Sectors,
    mid: &Sectors,
    sym: &PeriodicSymmetry,
    q: usize,
) -> (Mat<C64>, Mat<C64>, Sectors) {
    let rows = sym.group_rows(left, q);
    let cols = crate::mps::charges::group_bond(mid);
    let svd = BlockSvd::new(a, rows, cols);
    let counts = svd.counts(Keep::All, &MpsOptions::default());
    let width: usize = counts.iter().sum();
    let mut q_mat = Mat::<C64>::zeros(a.nrows(), width);
    let mut r_mat = Mat::<C64>::zeros(width, a.ncols());
    let mut at = 0;
    let mut entries = Vec::new();
    for (b, &k) in counts.iter().enumerate() {
        if k == 0 {
            continue;
        }
        for j in 0..k {
            for (r_idx, &row) in svd.rows[b].iter().enumerate() {
                q_mat[(row, at + j)] = svd.u[b][(r_idx, j)];
            }
            let s_val = svd.s[b][j];
            for (c_idx, &col) in svd.cols[b].iter().enumerate() {
                r_mat[(at + j, col)] = svd.v[b][(c_idx, j)].conj() * s_val;
            }
        }
        entries.push((svd.labels[b].clone(), k));
        at += k;
    }
    (q_mat, r_mat, Sectors::new(entries))
}

/// Block-sparse right orthonormalization of a right matrix `B` of shape
/// `mid_bond × (2 * right_bond)` with sectors `mid` and `right`, returning
/// `(L, Q_right, new_mid_sectors)` where `Q_right` is strictly
/// right-orthonormal (`Q_right Q_right† = I`) and `L Q_right = B`.
pub(super) fn qr_right_symmetric(
    b: MatRef<'_, C64>,
    mid: &Sectors,
    right: &Sectors,
    sym: &PeriodicSymmetry,
    q: usize,
) -> (Mat<C64>, Mat<C64>, Sectors) {
    let rows = crate::mps::charges::group_bond(mid);
    let cols = sym.group_cols(right, q);
    let svd = BlockSvd::new(b, rows, cols);
    let counts = svd.counts(Keep::All, &MpsOptions::default());
    let height: usize = counts.iter().sum();
    let mut l_mat = Mat::<C64>::zeros(b.nrows(), height);
    let mut q_mat = Mat::<C64>::zeros(height, b.ncols());
    let mut at = 0;
    let mut entries = Vec::new();
    for (blk, &k) in counts.iter().enumerate() {
        if k == 0 {
            continue;
        }
        for j in 0..k {
            let s_val = svd.s[blk][j];
            for (r_idx, &row) in svd.rows[blk].iter().enumerate() {
                l_mat[(row, at + j)] = svd.u[blk][(r_idx, j)] * s_val;
            }
            for (c_idx, &col) in svd.cols[blk].iter().enumerate() {
                q_mat[(at + j, col)] = svd.v[blk][(c_idx, j)].conj();
            }
        }
        entries.push((svd.labels[blk].clone(), k));
        at += k;
    }
    (l_mat, q_mat, Sectors::new(entries))
}

/// Compact layout of a 1-site or 2-site tensor in a symmetric [`PeriodicMps`].
/// Maps between dense `(l, phys, r)` tensors and compact allowed-block vectors.
#[derive(Clone)]
pub(super) struct PeriodicLayout {
    /// `(left_sector, d_s, offset, width)`.
    pub(super) rows: Vec<(usize, usize, usize, usize)>,
    /// `cols[s * phys + p] = Some((row_block_idx, col_offset, right_sector))`.
    pub(super) cols: Vec<Option<(usize, usize, usize)>>,
    pub(super) phys: usize,
    pub(super) left: Sectors,
    pub(super) right: Sectors,
    pub(super) len: usize,
}

impl PeriodicLayout {
    pub(super) fn one_site(
        left: &Sectors,
        right: &Sectors,
        sym: &PeriodicSymmetry,
        q: usize,
    ) -> Self {
        let shifts = vec![sym.site_shift(q, 0), sym.site_shift(q, 1)];
        Self::from_shifts(left, right, &shifts, &sym.charges)
    }

    pub(super) fn two_site(
        left: &Sectors,
        right: &Sectors,
        sym: &PeriodicSymmetry,
        q1: usize,
        q2: usize,
    ) -> Self {
        let mut shifts = Vec::with_capacity(4);
        for p1 in 0..2 {
            let s1 = sym.site_shift(q1, p1);
            for p2 in 0..2 {
                let s2 = sym.site_shift(q2, p2);
                shifts.push(sym.charges.add(&s1, &s2));
            }
        }
        Self::from_shifts(left, right, &shifts, &sym.charges)
    }

    fn from_shifts(
        left: &Sectors,
        right: &Sectors,
        shifts: &[Label],
        charges: &Charges,
    ) -> Self {
        let phys = shifts.len();
        let mut rows = Vec::new();
        let mut cols = vec![None; left.len() * phys];
        let mut offset = 0;
        for s in 0..left.len() {
            let mut width = 0;
            for (p, sh) in shifts.iter().enumerate() {
                if let Some(t) = right.find(&charges.add(left.label(s), sh)) {
                    cols[s * phys + p] = Some((rows.len(), width, t));
                    width += right.size(t);
                }
            }
            if width > 0 {
                let ds = left.size(s);
                rows.push((s, ds, offset, width));
                offset += ds * width;
            }
        }
        PeriodicLayout {
            rows,
            cols,
            phys,
            left: left.clone(),
            right: right.clone(),
            len: offset,
        }
    }

    /// Gather allowed blocks from a dense `(l, phys, r)` slice into a compact vector.
    pub(super) fn gather(&self, dense: &[C64]) -> Vec<C64> {
        let r_dim = self.right.dim();
        let mut out = vec![ZERO; self.len];
        for &(s, _ds, offset, width) in &self.rows {
            let rs = self.left.range(s);
            for p in 0..self.phys {
                let Some((_, col0, t)) = self.cols[s * self.phys + p] else {
                    continue;
                };
                let rt = self.right.range(t);
                for (a, l) in rs.clone().enumerate() {
                    let src = &dense[(l * self.phys + p) * r_dim + rt.start..][..rt.len()];
                    let dst = &mut out[offset + a * width + col0..][..rt.len()];
                    dst.copy_from_slice(src);
                }
            }
        }
        out
    }

    /// Scatter a compact vector back into a dense `(l, phys, r)` tensor.
    pub(super) fn scatter(&self, compact: &[C64]) -> Vec<C64> {
        let (l_dim, r_dim) = (self.left.dim(), self.right.dim());
        let mut out = vec![ZERO; l_dim * self.phys * r_dim];
        for &(s, _ds, offset, width) in &self.rows {
            let rs = self.left.range(s);
            for p in 0..self.phys {
                let Some((_, col0, t)) = self.cols[s * self.phys + p] else {
                    continue;
                };
                let rt = self.right.range(t);
                for (a, l) in rs.clone().enumerate() {
                    let src = &compact[offset + a * width + col0..][..rt.len()];
                    let dst = &mut out[(l * self.phys + p) * r_dim + rt.start..][..rt.len()];
                    dst.copy_from_slice(src);
                }
            }
        }
        out
    }
}

/// Split a gauge-weighted two-site tensor `theta_tilde` (between `left_eff`
/// and `right_eff`, after metric preconditioning `G_L`, `G_R` which are
/// block-diagonal in charge sectors!) block by block, returning the new
/// `(A_i, A_j, new_mid_sectors, discarded_weight)` with strict orthonormality
/// on the side moved away from.
#[allow(clippy::too_many_arguments)]
pub(super) fn split_metric_symmetric(
    theta_tilde: MatRef<'_, C64>,
    g_left: MatRef<'_, C64>,
    g_right: MatRef<'_, C64>,
    left_full: &Sectors,
    left_eff: &Sectors,
    right_eff: &Sectors,
    right_full: &Sectors,
    sym: &PeriodicSymmetry,
    q1: usize,
    q2: usize,
    options: &MpsOptions,
    absorb: Absorb,
) -> (Tensor, Tensor, Sectors, f64) {
    let rows = sym.group_rows(left_eff, q1);
    let cols = sym.group_cols(right_eff, q2);
    let svd = BlockSvd::new(theta_tilde, rows, cols);
    let counts = svd.counts(Keep::Truncate { floor: 0.0 }, options);
    let total = svd.weight();
    let kept = svd.kept_weight(&counts);
    let discarded = if total > 0.0 {
        ((total - kept) / total).max(0.0)
    } else {
        0.0
    };
    let mid_sectors = svd.sectors(&counts);
    let mid_dim = mid_sectors.dim();

    // Build U_k (left_eff*2 × mid_dim), S_k, and V_k† (mid_dim × 2*right_eff).
    let mut u_k = Mat::<C64>::zeros(theta_tilde.nrows(), mid_dim);
    let mut vh_k = Mat::<C64>::zeros(mid_dim, theta_tilde.ncols());
    let mut s_k = Vec::with_capacity(mid_dim);
    let mut at = 0;
    for (b, &k) in counts.iter().enumerate() {
        for j in 0..k {
            s_k.push(svd.s[b][j]);
            for (r_idx, &row) in svd.rows[b].iter().enumerate() {
                u_k[(row, at + j)] = svd.u[b][(r_idx, j)];
            }
            for (c_idx, &col) in svd.cols[b].iter().enumerate() {
                vh_k[(at + j, col)] = svd.v[b][(c_idx, j)].conj();
            }
        }
        at += k;
    }

    // Map back to original bond coordinates:
    // A_raw = (G_L ⊗ I_2) U_k  [shape (left_full * 2) × mid_dim]
    // B_raw = V_k† (I_2 ⊗ G_R^T) [shape mid_dim × (2 * right_full)]
    let l_full = left_full.dim();
    let r_full = right_full.dim();
    let l_eff = left_eff.dim();
    let r_eff = right_eff.dim();

    let mut a_raw = Mat::<C64>::zeros(l_full * 2, mid_dim);
    for p in 0..2 {
        let u_p = Mat::<C64>::from_fn(l_eff, mid_dim, |l, c| u_k[(l * 2 + p, c)]);
        let prod = matmul(g_left, u_p.as_ref());
        for l in 0..l_full {
            for c in 0..mid_dim {
                a_raw[(l * 2 + p, c)] = prod[(l, c)];
            }
        }
    }
    let mut b_raw = Mat::<C64>::zeros(mid_dim, 2 * r_full);
    for p in 0..2 {
        let v_p = Mat::<C64>::from_fn(mid_dim, r_eff, |r, c| vh_k[(r, p * r_eff + c)]);
        let prod = matmul(v_p.as_ref(), g_right.transpose());
        for r in 0..mid_dim {
            for c in 0..r_full {
                b_raw[(r, p * r_full + c)] = prod[(r, c)];
            }
        }
    }

    // Scale S_k onto the side that absorbs the centre, then block-QR the other side.
    match absorb {
        Absorb::Right => {
            let (q_left, r_carry, final_mid) =
                qr_left_symmetric(a_raw.as_ref(), left_full, &mid_sectors, sym, q1);
            let rs_b = Mat::<C64>::from_fn(r_carry.nrows(), mid_dim, |r, c| {
                r_carry[(r, c)] * s_k[c]
            });
            let b_final = matmul(rs_b.as_ref(), b_raw.as_ref());
            let k = final_mid.dim();
            (
                Tensor::from_matrix(q_left.as_ref(), l_full, k),
                Tensor::from_matrix(b_final.as_ref(), k, r_full),
                final_mid,
                discarded,
            )
        }
        Absorb::Left => {
            let (l_carry, q_right, final_mid) =
                qr_right_symmetric(b_raw.as_ref(), &mid_sectors, right_full, sym, q2);
            let a_s = Mat::<C64>::from_fn(a_raw.nrows(), mid_dim, |r, c| a_raw[(r, c)] * s_k[c]);
            let a_final = matmul(a_s.as_ref(), l_carry.as_ref());
            let k = final_mid.dim();
            (
                Tensor::from_matrix(a_final.as_ref(), l_full, k),
                Tensor::from_matrix(q_right.as_ref(), k, r_full),
                final_mid,
                discarded,
            )
        }
    }
}

impl PeriodicMps {
    /// Convert this periodic state into a symmetric periodic MPS with definite
    /// Abelian `charges` (`U(1)` or `Z_n`, see [`Charge`]).
    ///
    /// Product states (such as [`PeriodicMps::basis`]) and states with seam
    /// bond dimension 1 convert directly in `O(n χ³)`. States with seam bond
    /// dimension `χ₀ > 1` are unfolded across the seam into an exact sector
    /// decomposition and re-compressed.
    ///
    /// Errors if more than [`MAX_CHARGE_LEAK`] of the norm squared lies outside
    /// the dominant total-charge sector, if a charge has the wrong qubit count,
    /// or if the state is zero.
    pub fn with_charges(&self, charges: &[Charge]) -> Result<PeriodicMps, Error> {
        let n = self.n_qubits();
        let charges_arc = Arc::new(Charges::new(charges, n)?);
        let dense = self.to_dense();
        let total_norm = dense.norm_sqr();
        if total_norm == 0.0 || !total_norm.is_finite() {
            return Err(Error::InvalidArgument(
                "can't give charges to a zero state".into(),
            ));
        }
        let chi0 = dense.tensors[0].left;
        if chi0 == 1 {
            // Fast exact O(n χ³) path: treat as an open chain with bond 0 = 1,
            // assign sectors left-to-right, determine Q_tot at site n - 1, then
            // sweep right-to-left.
            let open = crate::mps::Mps::from_raw_tensors(
                dense.tensors.clone(),
                dense.options,
            );
            let sym_open = open.with_charges(charges)?;
            let total_charge: Label = sym_open.total_charge().unwrap().into();
            let mut bonds = Vec::with_capacity(n);
            bonds.push(Sectors::single(charges_arc.zero()));
            for b in 0..n - 1 {
                let raw = sym_open.bond_sectors(b).unwrap();
                bonds.push(Sectors::new(
                    raw.into_iter().map(|(l, d)| (l.into(), d)).collect(),
                ));
            }
            let tensors = sym_open.raw_tensors().to_vec();
            let mut out = PeriodicMps {
                tensors,
                options: dense.options,
                truncation_error: dense.truncation_error + sym_open.truncation_error(),
                transfer_rank: dense.transfer_rank,
                transfer_cutoff: dense.transfer_cutoff,
                symmetry: Some(PeriodicSymmetry {
                    charges: charges_arc,
                    bonds,
                    total: total_charge,
                }),
                #[cfg(feature = "gpu")]
                gpu: self.gpu.clone(),
            };
            // Re-normalise to match the original norm.
            let new_norm = out.norm_sqr();
            if new_norm > 0.0 {
                let scale = (total_norm / new_norm).sqrt();
                out.tensors[0].data.iter_mut().for_each(|a| *a *= scale);
            }
            return Ok(out);
        }

        // General χ₀ > 1 path: unfold the seam trace Σ_{α=0}^{χ₀-1} into an
        // equivalent open-boundary MPS (bond dimension χ₀ · χᵢ), convert via
        // `Mps::with_charges` to verify total charge and isolate sectors, then
        // keep as a periodic MPS with seam bond 1 (further DMRG/TDVP sweeps
        // immediately re-grow the seam bond sectors).
        let unfolded = dense.unfold_to_open_mps();
        let sym_open = unfolded.with_charges(charges)?;
        let total_charge: Label = sym_open.total_charge().unwrap().into();
        let mut bonds = Vec::with_capacity(n);
        bonds.push(Sectors::single(charges_arc.zero()));
        for b in 0..n - 1 {
            let raw = sym_open.bond_sectors(b).unwrap();
            bonds.push(Sectors::new(
                raw.into_iter().map(|(l, d)| (l.into(), d)).collect(),
            ));
        }
        let mut out = PeriodicMps {
            tensors: sym_open.raw_tensors().to_vec(),
            options: dense.options,
            truncation_error: dense.truncation_error + sym_open.truncation_error(),
            transfer_rank: dense.transfer_rank,
            transfer_cutoff: dense.transfer_cutoff,
            symmetry: Some(PeriodicSymmetry {
                charges: charges_arc,
                bonds,
                total: total_charge,
            }),
            #[cfg(feature = "gpu")]
            gpu: self.gpu.clone(),
        };
        let new_norm = out.norm_sqr();
        if new_norm > 0.0 {
            let scale = (total_norm / new_norm).sqrt();
            out.tensors[0].data.iter_mut().for_each(|a| *a *= scale);
        }
        Ok(out)
    }

    /// Unfold the periodic trace `Tr(A[0] ... A[n-1])` into an open-boundary
    /// MPS with left/right boundary bond 1 by carrying the seam index `α ∈ 0..χ₀`
    /// as a spectator index across sites `0..n-1`.
    fn unfold_to_open_mps(&self) -> crate::mps::Mps {
        let n = self.n_qubits();
        let chi0 = self.tensors[0].left;
        let mut tensors = Vec::with_capacity(n);
        for (q, t) in self.tensors.iter().enumerate() {
            if q == 0 {
                // left = 1, right = chi0 * t.right; index (0, p, α * r + β) = A[0][α, p, β].
                let right = chi0 * t.right;
                let mut data = vec![ZERO; 2 * right];
                for alpha in 0..chi0 {
                    for p in 0..2 {
                        for beta in 0..t.right {
                            data[p * right + alpha * t.right + beta] =
                                t.data[(alpha * 2 + p) * t.right + beta];
                        }
                    }
                }
                tensors.push(Tensor {
                    left: 1,
                    right,
                    data,
                });
            } else if q + 1 == n {
                // left = chi0 * t.left, right = 1; index (α * l + γ, p, 0) = A[n-1][γ, p, α].
                let left = chi0 * t.left;
                let mut data = vec![ZERO; left * 2];
                for alpha in 0..chi0 {
                    for gamma in 0..t.left {
                        for p in 0..2 {
                            data[(alpha * t.left + gamma) * 2 + p] =
                                t.data[(gamma * 2 + p) * t.right + alpha];
                        }
                    }
                }
                tensors.push(Tensor {
                    left,
                    right: 1,
                    data,
                });
            } else {
                // Block-diagonal copy I_{chi0} ⊗ A[q].
                let left = chi0 * t.left;
                let right = chi0 * t.right;
                let mut data = vec![ZERO; left * 2 * right];
                for alpha in 0..chi0 {
                    for l in 0..t.left {
                        for p in 0..2 {
                            for r in 0..t.right {
                                let row = (alpha * t.left + l) * 2 + p;
                                let col = alpha * t.right + r;
                                data[row * right + col] = t.data[(l * 2 + p) * t.right + r];
                            }
                        }
                    }
                }
                tensors.push(Tensor { left, right, data });
            }
        }
        // Keep exact bond dim and zero cutoff during unfolding so `with_charges`
        // converts losslessly without spurious truncation error.
        let max_bond = self
            .options
            .max_bond
            .max(tensors.iter().map(|t| t.right).max().unwrap_or(1));
        let mut mps = crate::mps::Mps::from_raw_tensors(
            tensors,
            MpsOptions {
                max_bond,
                cutoff: 0.0,
            },
        );
        // Move center to 0 so `with_charges` sees canonical form.
        mps.normalise_center_zero();
        mps
    }

    /// The same state without charges: a plain dense [`PeriodicMps`] that gates
    /// can run on.
    pub fn to_dense(&self) -> PeriodicMps {
        let mut mps = self.clone();
        mps.symmetry = None;
        mps
    }

    /// The charges of a symmetric [`PeriodicMps`], or `None`.
    pub fn charges(&self) -> Option<&[Charge]> {
        self.symmetry.as_ref().map(|s| s.charges.charges())
    }

    /// The value of each charge on the state, or `None` without charges.
    pub fn total_charge(&self) -> Option<Vec<i64>> {
        self.symmetry.as_ref().map(|s| s.total.to_vec())
    }

    /// The sectors of the bond between qubits `bond` and `(bond + 1) % n`
    /// (`bond = n - 1` is the periodic seam bond between `n - 1` and `0`),
    /// as `(charge label, dimension)` pairs, or `None` without charges.
    pub fn bond_sectors(&self, bond: usize) -> Option<Vec<(Vec<i64>, usize)>> {
        assert!(bond < self.n_qubits(), "no bond {bond}");
        let idx = (bond + 1) % self.n_qubits();
        self.symmetry.as_ref().map(|s| {
            let sectors = &s.bonds[idx];
            (0..sectors.len())
                .map(|k| (sectors.label(k).to_vec(), sectors.size(k)))
                .collect()
        })
    }

    /// Fails for a symmetric [`PeriodicMps`], which `what` doesn't support.
    pub(super) fn require_dense(&self, what: &str) -> Result<(), Error> {
        if self.symmetry.is_some() {
            return Err(Error::InvalidArgument(format!(
                "{what} on a PeriodicMps with charges isn't supported; convert it with to_dense first"
            )));
        }
        Ok(())
    }

    /// Largest entry of any site tensor outside its charge-allowed blocks
    /// (zero for a consistent symmetric [`PeriodicMps`]).
    #[cfg(test)]
    pub(crate) fn charge_violation(&self) -> f64 {
        let sym = self.symmetry.as_ref().expect("a symmetric PeriodicMps");
        let mut worst: f64 = 0.0;
        let label_of = |b: &Sectors, x: usize| -> Label {
            let k = (0..b.len()).find(|&k| b.range(k).contains(&x)).unwrap();
            b.label(k).into()
        };
        for (i, t) in self.tensors.iter().enumerate() {
            let left = sym.left_bond(i);
            let right = sym.right_bond(i);
            assert_eq!(left.dim(), t.left);
            assert_eq!(right.dim(), t.right);
            for l in 0..t.left {
                let ql = label_of(left, l);
                for p in 0..2 {
                    let want = sym.charges.add(&ql, &sym.site_shift(i, p));
                    for r in 0..t.right {
                        if label_of(right, r) != want {
                            worst = worst.max(t.data[(l * 2 + p) * t.right + r].norm());
                        }
                    }
                }
            }
        }
        worst
    }
}

/// Suppress unused-import warning when `MAX_CHARGE_LEAK` or `ZERO_SINGULAR`
/// are referenced for doc/consistency.
const _: (f64, f64) = (MAX_CHARGE_LEAK, ZERO_SINGULAR);

/// Compute the strictly invertible block-diagonal eigendecomposition
/// preconditioner `G = U Λ_c^{-1/2}` and `G^{-1} = Λ_c^{1/2} U†` of a Hermitian
/// positive semidefinite matrix `M` that is block-diagonal in `sectors`,
/// clamping eigenvalues below `max(tol, 1e-9) * λ_max` so small physical
/// entanglement modes (and newly populated noise sectors!) are never dropped
/// while keeping `cond(G) ≤ 3.2e4`.
pub(super) fn block_metric_gauge(
    m: &Mat<C64>,
    sectors: &Sectors,
    tol: f64,
) -> (Mat<C64>, Sectors, Mat<C64>) {
    let eigs: Vec<(usize, Vec<f64>, Mat<C64>)> = (0..sectors.len())
        .map(|k| {
            let r = sectors.range(k);
            let d = r.len();
            let blk = Mat::<C64>::from_fn(d, d, |i, j| {
                (m[(r.start + i, r.start + j)] + m[(r.start + j, r.start + i)].conj()) * 0.5
            });
            let (raw_vals, u) = hermitian_evd(blk.as_ref());
            let vals: Vec<f64> = raw_vals.into_iter().map(|v| v.max(0.0)).collect();
            (k, vals, u)
        })
        .collect();
    let max_val = eigs
        .iter()
        .flat_map(|(_, vals, _)| vals.iter().copied())
        .fold(0.0f64, f64::max);
    let cutoff = tol * max_val;
    let mut entries = Vec::new();
    let mut kept_total = 0;
    let mut kept_indices: Vec<(usize, Vec<usize>)> = Vec::new();
    for (k, vals, _) in &eigs {
        let mut idx: Vec<usize> = vals
            .iter()
            .enumerate()
            .filter(|&(_, &v)| v > cutoff)
            .map(|(j, _)| j)
            .collect();
        if idx.is_empty() && max_val == 0.0 && !vals.is_empty() {
            idx.push(vals.len() - 1);
        }
        if !idx.is_empty() {
            entries.push((sectors.label(*k).into(), idx.len()));
            kept_total += idx.len();
            kept_indices.push((*k, idx));
        }
    }
    if kept_total == 0 {
        let best = eigs
            .iter()
            .enumerate()
            .flat_map(|(pos, (k, vals, _))| {
                vals.iter()
                    .enumerate()
                    .map(move |(j, &v)| (v, pos, *k, j))
            })
            .max_by(|a, b| a.0.total_cmp(&b.0))
            .expect("non-empty bond");
        entries.push((sectors.label(best.2).into(), 1));
        kept_total = 1;
        kept_indices.push((best.2, vec![best.3]));
    }
    let dim = sectors.dim();
    let mut g = Mat::<C64>::zeros(dim, kept_total);
    let mut g_inv = Mat::<C64>::zeros(kept_total, dim);
    let mut col = 0;
    for (k, idxs) in &kept_indices {
        let (_, vals, u) = &eigs[*k];
        let r = sectors.range(*k);
        for &j in idxs {
            let lam = vals[j].max(1e-300);
            let inv_sqrt = lam.sqrt().recip();
            let sqrt = lam.sqrt();
            for i in 0..r.len() {
                g[(r.start + i, col)] = u[(i, j)] * inv_sqrt;
                g_inv[(col, r.start + i)] = u[(i, j)].conj() * sqrt;
            }
            col += 1;
        }
    }
    (g, Sectors::new(entries), g_inv)
}
