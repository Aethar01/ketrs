//! Periodic transfer matrices, exact ring contractions, and Pippan–White–Evertz
//! (2010) low-rank transfer-matrix compression.
//!
//! For a site tensor `A` of shape `(l, 2, r)` (ket) and `B` of shape `(l', 2, r')`
//! (bra) with local operator `O`, the transfer matrix acts on a batch of `k`
//! matrices `X` of shape `(l' × l)` by:
//!
//!   `X'[j, β', β] = Σ_{α', α, p, p'} O[p, p'] B*[α', p, β'] X[j, α', α] A[α, p', β]`
//!
//! Stacking the `k` matrices turns the entire batch step into **two** BLAS-3
//! GEMMs costing `O(k χ³)` flops with minimal allocation.
//!
//! Exact ring traces start at the smallest bond `χ_min`, costing `O(n χ_min² χ³)`.
//! For periodic DMRG and TDVP sweeps, the opposite-half transfer matrix across
//! `L_opp ≈ n / 2` sites is factored into rank `p ≪ χ²` modes `Σ_μ R_μ ⊗ L_μ`:
//!
//! - When `min(χ_L², χ_R²) ≤ EXACT_TRANSFER_DIM` (or `transfer_rank == Some(0)`),
//!   the exact matrix is built and SVD'd (`O(L_opp χ⁵)` setup, `O(p χ³)` sweep).
//! - Otherwise, Halko–Martinsson–Tropp randomized SVD with power iteration finds
//!   the leading `p` singular triplets directly via matrix-free batched matvecs
//!   in pure `O(p L_opp χ³)`.

use super::symmetric::{PeriodicSiteBlocks, PeriodicSymmetry};
use crate::mps::charges::{Charges, Label, Sectors};
use crate::mps::linalg::{matmul, thin_qr, thin_svd};
use crate::mps::mpo::{Local, Mpo, MpoSite, left_keyed, op_entry, right_keyed};
use crate::mps::{ONE, Tensor, ZERO, pauli_matrix};
use crate::observable::Pauli;
use faer::linalg::matmul::matmul as faer_matmul;
use faer::{Accum, Mat, MatMut, MatRef, Par};
use num_complex::Complex64 as C64;
use rayon::prelude::*;
use std::collections::BTreeSet;

type SymTransferMode = (f64, Label, Label, Mat<C64>, Mat<C64>);
type ExpectationTerm = (usize, f64, Vec<(usize, Pauli)>);

/// Up to this squared bond dimension (`χ ≤ 8` on general rings), building the
/// opposite-half transfer matrix explicitly in one batched pass (`min_dim`
/// vectors) is both faster than multi-pass randomized SVD and exact to machine
/// precision. Above it, progressive Halko–Martinsson–Tropp randomized SVD runs
/// in pure `O(p L χ³)` unless exact mode (`transfer_rank == Some(0)`) is requested.
pub(super) const EXACT_TRANSFER_DIM: usize = 64;

/// On small rings (`n <= SHORT_RING_QUBITS`, e.g. `n <= 12`), the complement
/// transfer matrix has not yet decayed exponentially and `N_bond` can be
/// heavily rank-deficient (`χ² > 2^{L_opp}`), so exact SVD is kept up to
/// `SHORT_EXACT_TRANSFER_DIM` (`χ ≤ 16`) at negligible cost.
const SHORT_RING_QUBITS: usize = 12;
const SHORT_EXACT_TRANSFER_DIM: usize = 256;

/// Default maximum rank `p` for adaptive PWE compression when `transfer_rank`
/// is `None` and `min_dim > EXACT_TRANSFER_DIM`.
pub(super) const DEFAULT_MAX_PWE_RANK: usize = 24;

/// Progressive target ranks for adaptive randomized SVD: starts small (`p = 4`)
/// so gapped systems stop immediately once `σ_p / σ_1 ≤ transfer_cutoff`,
/// while multi-flux gauge rings can still extend past 24 if `σ_24 / σ_1 > 1e-3`.
const ADAPTIVE_RANK_STAGES: [usize; 6] = [4, 8, 16, DEFAULT_MAX_PWE_RANK, ADAPTIVE_HIGH_RANK, 48];

/// Maximum adaptive rank when singular values on a short critical complement
/// have not yet decayed below `1e-6`.
const ADAPTIVE_HIGH_RANK: usize = 32;

/// Oversampling count for randomized SVD.
const RSVD_OVERSAMPLE: usize = 8;

/// Apply a single-site transfer operator `T(B, A, op)` left-to-right to a batch
/// of `batch` matrices `X[j]` of shape `(l_bra × l_ket)`, stored consecutively
/// as `batch` row-major blocks (`data` length `batch * l_bra * l_ket`).
/// Returns `batch` row-major matrices of shape `(r_bra × r_ket)`.
pub(super) fn batch_transfer_right(
    x: &[C64],
    batch: usize,
    bra: &Tensor,
    ket: &Tensor,
    op: Local,
) -> Vec<C64> {
    let (lb, rb) = (bra.left, bra.right);
    let (lk, rk) = (ket.left, ket.right);
    debug_assert_eq!(x.len(), batch * lb * lk);
    if batch == 0 || lb == 0 || rb == 0 || lk == 0 || rk == 0 {
        return vec![ZERO; batch * rb * rk];
    }

    // Step 1: Y[(j, α'), (p', β)] = Σ_α X[(j, α'), α] * ket[α, (p', β)]
    // Shape: (batch * lb) × (2 * rk)
    let x_mat = MatRef::from_row_major_slice(x, batch * lb, lk);
    let mut y = vec![ZERO; batch * lb * 2 * rk];
    faer_matmul(
        MatMut::from_row_major_slice_mut(&mut y, batch * lb, 2 * rk),
        Accum::Replace,
        x_mat,
        ket.as_right_matrix(),
        ONE,
        par_for(batch * lb * lk * rk),
    );

    // Step 2: Apply local operator O[p, p'] and permute to Z[(α', p), (j, β)]
    // Shape: (lb * 2) × (batch * rk)
    let cols = batch * rk;
    let mut z = vec![ZERO; lb * 2 * cols];
    let op_rows = [op_entry(op, 0), op_entry(op, 1)];
    for (p_out, entry) in op_rows.iter().enumerate() {
        let Some((p_in, f)) = *entry else { continue };
        for j in 0..batch {
            let col_off = j * rk;
            for al in 0..lb {
                let src = &y[((j * lb + al) * 2 + p_in) * rk..][..rk];
                let dst = &mut z[(al * 2 + p_out) * cols + col_off..][..rk];
                if f == ONE {
                    dst.copy_from_slice(src);
                } else {
                    for (d, &s) in dst.iter_mut().zip(src) {
                        *d = f * s;
                    }
                }
            }
        }
    }

    // Step 3: W[β', (j, β)] = Σ_{α', p} bra*[α', p, β'] Z[(α', p), (j, β)]
    // Shape: rb × (batch * rk)
    let mut w = vec![ZERO; rb * cols];
    faer_matmul(
        MatMut::from_row_major_slice_mut(&mut w, rb, cols),
        Accum::Replace,
        bra.as_left_matrix().adjoint(),
        MatRef::from_row_major_slice(&z, lb * 2, cols),
        ONE,
        par_for(rb * lb * cols),
    );

    // Transpose (β', j, β) -> (j, β', β).
    let mut out = vec![ZERO; batch * rb * rk];
    for bp in 0..rb {
        for j in 0..batch {
            let src = &w[bp * cols + j * rk..][..rk];
            let dst = &mut out[(j * rb + bp) * rk..][..rk];
            dst.copy_from_slice(src);
        }
    }
    out
}

/// Apply a single-site transfer operator `T(B, A, op)` right-to-left to a batch
/// of `batch` matrices `X[j]` of shape `(r_bra × r_ket)`.
/// Computes `X'[j, α', α] = Σ_{β', β, p, p'} O[p, p'] B*[α', p, β'] X[j, β', β] A[α, p', β]`.
pub(super) fn batch_transfer_left(
    x: &[C64],
    batch: usize,
    bra: &Tensor,
    ket: &Tensor,
    op: Local,
) -> Vec<C64> {
    let (lb, rb) = (bra.left, bra.right);
    let (lk, rk) = (ket.left, ket.right);
    debug_assert_eq!(x.len(), batch * rb * rk);
    if batch == 0 || lb == 0 || rb == 0 || lk == 0 || rk == 0 {
        return vec![ZERO; batch * lb * lk];
    }

    // Step 1: Y[(j, β'), (α, p')] = Σ_β X[(j, β'), β] * A^T[β, (α, p')]
    // Wait: ket as_left_matrix is (lk * 2) × rk, so ket.as_left_matrix().transpose() is rk × (lk * 2).
    let x_mat = MatRef::from_row_major_slice(x, batch * rb, rk);
    let mut y = vec![ZERO; batch * rb * lk * 2];
    faer_matmul(
        MatMut::from_row_major_slice_mut(&mut y, batch * rb, lk * 2),
        Accum::Replace,
        x_mat,
        ket.as_left_matrix().transpose(),
        ONE,
        par_for(batch * rb * rk * lk),
    );

    // Step 2: Apply O[p, p'] and permute to Z[(p, β'), (j, α)]
    // Shape: (2 * rb) × (batch * lk)
    let cols = batch * lk;
    let mut z = vec![ZERO; 2 * rb * cols];
    let op_rows = [op_entry(op, 0), op_entry(op, 1)];
    for (p_out, entry) in op_rows.iter().enumerate() {
        let Some((p_in, f)) = *entry else { continue };
        for j in 0..batch {
            for bp in 0..rb {
                let y_row = &y[(j * rb + bp) * lk * 2..][..lk * 2];
                let dst = &mut z[(p_out * rb + bp) * cols + j * lk..][..lk];
                if f == ONE {
                    for al in 0..lk {
                        dst[al] = y_row[al * 2 + p_in];
                    }
                } else {
                    for al in 0..lk {
                        dst[al] = f * y_row[al * 2 + p_in];
                    }
                }
            }
        }
    }

    // Step 3: W[α', (j, α)] = Σ_{p, β'} bra*[α', (p, β')] * Z[(p, β'), (j, α)]
    // bra.as_right_matrix().conjugate() has shape lb × (2 * rb).
    let mut w = vec![ZERO; lb * cols];
    faer_matmul(
        MatMut::from_row_major_slice_mut(&mut w, lb, cols),
        Accum::Replace,
        bra.as_right_matrix().conjugate(),
        MatRef::from_row_major_slice(&z, 2 * rb, cols),
        ONE,
        par_for(lb * rb * cols),
    );

    // Transpose (α', j, α) -> (j, α', α).
    let mut out = vec![ZERO; batch * lb * lk];
    for ap in 0..lb {
        for j in 0..batch {
            let src = &w[ap * cols + j * lk..][..lk];
            let dst = &mut out[(j * lb + ap) * lk..][..lk];
            dst.copy_from_slice(src);
        }
    }
    out
}

/// Apply the **adjoint** of the right-to-left transfer map `T_left` (needed for
/// randomized SVD of `E_opp: cut_end -> cut_start`):
/// Given `Y[j, α', α]` at the left cut, compute
/// `X[j, β', β] = Σ_{α', α, p, p'} (O[p, p'])* B[α', p, β'] Y[j, α', α] A*[α, p', β]`.
/// Notice this equals `conj(batch_transfer_right(conj(Y), bra, ket, op_dagger))`?
/// Let's verify directly:
/// `X*[j, β', β] = Σ O[p, p'] B*[α', p, β'] Y*[j, α', α] A[α, p', β]` — YES!
/// That's LITERALLY `conj(batch_transfer_right(conj(Y), bra, ket, op))`!
pub(super) fn batch_transfer_left_adjoint(
    y: &[C64],
    batch: usize,
    bra: &Tensor,
    ket: &Tensor,
    op: Local,
) -> Vec<C64> {
    let y_conj: Vec<C64> = y.iter().map(|z| z.conj()).collect();
    let mut res = batch_transfer_right(&y_conj, batch, bra, ket, op);
    res.iter_mut().for_each(|z| *z = z.conj());
    res
}

fn par_for(work: usize) -> Par {
    if work >= 1 << 23 {
        faer::get_global_parallelism()
    } else {
        Par::Seq
    }
}

/// Apply an MPO site `w` right-to-left to a batch of `batch` multi-channel
/// matrices `X[ch][j]` (each `r × r`, stored as `Vec<Vec<C64>>` indexed by
/// MPO right channel `c`, each vec having `batch * r * r` elements).
pub(super) fn batch_mpo_transfer_left(
    x_by_chan: &[Vec<C64>],
    batch: usize,
    a: &Tensor,
    w: &MpoSite,
) -> Vec<Vec<C64>> {
    let (l, r) = (a.left, a.right);
    // Step 1: for each right channel c, multiply X[c] by A^T -> Y[c] of shape (batch * r) × (l * 2).
    let y_by_chan: Vec<Vec<C64>> = x_by_chan
        .par_iter()
        .map(|xc| {
            let mut y = vec![ZERO; batch * r * l * 2];
            if !xc.is_empty() && r > 0 && l > 0 {
                faer_matmul(
                    MatMut::from_row_major_slice_mut(&mut y, batch * r, l * 2),
                    Accum::Replace,
                    MatRef::from_row_major_slice(xc, batch * r, r),
                    a.as_left_matrix().transpose(),
                    ONE,
                    Par::Seq,
                );
            }
            y
        })
        .collect();

    let rows = left_keyed(w);
    let cols = batch * l;
    let bra_conj = a.as_right_matrix().conjugate();
    (0..w.left)
        .into_par_iter()
        .map(|a_ch| {
            let mut z = vec![ZERO; 2 * r * cols];
            let mut any = false;
            for p_out in 0..2 {
                let terms = &rows[a_ch * 2 + p_out];
                if terms.is_empty() {
                    continue;
                }
                any = true;
                for &(c, p_in, f) in terms {
                    let yc = &y_by_chan[c];
                    for j in 0..batch {
                        for bp in 0..r {
                            let y_row = &yc[(j * r + bp) * l * 2..][..l * 2];
                            let dst = &mut z[(p_out * r + bp) * cols + j * l..][..l];
                            for al in 0..l {
                                dst[al] += f * y_row[al * 2 + p_in];
                            }
                        }
                    }
                }
            }
            let mut out = vec![ZERO; batch * l * l];
            if !any || l == 0 || r == 0 || batch == 0 {
                return out;
            }
            let mut w_mat = vec![ZERO; l * cols];
            faer_matmul(
                MatMut::from_row_major_slice_mut(&mut w_mat, l, cols),
                Accum::Replace,
                bra_conj,
                MatRef::from_row_major_slice(&z, 2 * r, cols),
                ONE,
                Par::Seq,
            );
            for ap in 0..l {
                for j in 0..batch {
                    let src = &w_mat[ap * cols + j * l..][..l];
                    let dst = &mut out[(j * l + ap) * l..][..l];
                    dst.copy_from_slice(src);
                }
            }
            out
        })
        .collect()
}

/// Apply the adjoint of `batch_mpo_transfer_left` (left-to-right through MPO
/// site `w`) to `y_by_chan` (indexed by MPO left channel `a_ch`), returning
/// `x_by_chan` (indexed by MPO right channel `c`).
/// By exact conjugation symmetry:
/// `(T_mpo_left)† (Y) = conj( T_mpo_right(conj(Y)) )` where `T_mpo_right` uses
/// the same MPO entries `(a, b, op, coeff)` (not conjugated!).
pub(super) fn batch_mpo_transfer_left_adjoint(
    y_by_chan: &[Vec<C64>],
    batch: usize,
    a: &Tensor,
    w: &MpoSite,
) -> Vec<Vec<C64>> {
    let (l, r) = (a.left, a.right);
    // Conjugate inputs.
    let x_by_chan: Vec<Vec<C64>> = y_by_chan
        .par_iter()
        .map(|ya| {
            let conj_ya: Vec<C64> = ya.iter().map(|z| z.conj()).collect();
            let mut y = vec![ZERO; batch * l * 2 * r];
            if !conj_ya.is_empty() && l > 0 && r > 0 {
                faer_matmul(
                    MatMut::from_row_major_slice_mut(&mut y, batch * l, 2 * r),
                    Accum::Replace,
                    MatRef::from_row_major_slice(&conj_ya, batch * l, l),
                    a.as_right_matrix(),
                    ONE,
                    Par::Seq,
                );
            }
            y
        })
        .collect();

    let rows = right_keyed(w);
    let cols = batch * r;
    let bra_adj = a.as_left_matrix().adjoint();
    (0..w.right)
        .into_par_iter()
        .map(|b_ch| {
            let mut z = vec![ZERO; l * 2 * cols];
            let mut any = false;
            for p_out in 0..2 {
                let terms = &rows[b_ch * 2 + p_out];
                if terms.is_empty() {
                    continue;
                }
                any = true;
                for &(a_ch, p_in, f) in terms {
                    let xa = &x_by_chan[a_ch];
                    for j in 0..batch {
                        let col_off = j * r;
                        for al in 0..l {
                            let src = &xa[((j * l + al) * 2 + p_in) * r..][..r];
                            let dst = &mut z[(al * 2 + p_out) * cols + col_off..][..r];
                            for (d, &s) in dst.iter_mut().zip(src) {
                                *d += f * s;
                            }
                        }
                    }
                }
            }
            let mut out = vec![ZERO; batch * r * r];
            if !any || l == 0 || r == 0 || batch == 0 {
                return out;
            }
            let mut w_mat = vec![ZERO; r * cols];
            faer_matmul(
                MatMut::from_row_major_slice_mut(&mut w_mat, r, cols),
                Accum::Replace,
                bra_adj,
                MatRef::from_row_major_slice(&z, l * 2, cols),
                ONE,
                Par::Seq,
            );
            for bp in 0..r {
                for j in 0..batch {
                    let src = &w_mat[bp * cols + j * r..][..r];
                    let dst = &mut out[(j * r + bp) * r..][..r];
                    for (d, &s) in dst.iter_mut().zip(src) {
                        *d = s.conj();
                    }
                }
            }
            out
        })
        .collect()
}

/// Exact periodic inner product `⟨bra|ket⟩ = Tr(T[0] T[1] ... T[n-1])`.
/// Starts at the bond with smallest `D_bra * D_ket`, costing `O(n χ_min² χ³)`.
pub(super) fn exact_ring_inner(bra: &[Tensor], ket: &[Tensor]) -> C64 {
    let n = bra.len();
    debug_assert_eq!(n, ket.len());
    let start = (0..n)
        .min_by_key(|&i| bra[i].left * ket[i].left)
        .unwrap_or(0);
    let (d_bra, d_ket) = (bra[start].left, ket[start].left);
    let batch = d_bra * d_ket;
    // Basis of all (a, b) matrices E_{ab}.
    let mut state = vec![ZERO; batch * batch];
    for i in 0..batch {
        state[i * batch + i] = ONE;
    }
    for step in 0..n {
        let q = (start + step) % n;
        state = batch_transfer_right(&state, batch, &bra[q], &ket[q], Local::I);
    }
    let mut trace = ZERO;
    for i in 0..batch {
        trace += state[i * batch + i];
    }
    trace
}

/// Exact block-sparse squared norm `⟨ψ|ψ⟩` for a symmetric [`PeriodicMps`].
/// Decomposes the ring trace by conserved charge difference `Δq = s - s'`,
/// pruning any `Δq` not present on all `n` bonds.
pub(super) fn exact_ring_norm_symmetric(tensors: &[Tensor], sym: &PeriodicSymmetry) -> f64 {
    let n = tensors.len();
    // Collect Δq present on bond 0, then intersect with every other bond.
    let deltas_on = |bond: &Sectors| -> BTreeSet<Label> {
        let mut set = BTreeSet::new();
        for sp in 0..bond.len() {
            for s in 0..bond.len() {
                set.insert(sym.charges.sub(bond.label(s), bond.label(sp)));
            }
        }
        set
    };
    let mut common = deltas_on(&sym.bonds[0]);
    for bond in &sym.bonds[1..n] {
        let here = deltas_on(bond);
        common.retain(|d| here.contains(d));
        if common.is_empty() {
            return 0.0;
        }
    }

    let total: C64 = common
        .into_iter()
        .collect::<Vec<_>>()
        .par_iter()
        .map(|delta| trace_delta_block(tensors, sym, delta))
        .sum();
    total.re
}

/// Trace of `T[0](I) ... T[n-1](I)` restricted to the charge-difference sector
/// `Δq = s - s'`.
fn trace_delta_block(tensors: &[Tensor], sym: &PeriodicSymmetry, delta: &Label) -> C64 {
    let n = tensors.len();
    // Sector pairs `(sp, s)` with `s - sp == delta` on each bond.
    let pairs_on = |bond: &Sectors| -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        for s in 0..bond.len() {
            let bra_l = sym.charges.sub(bond.label(s), delta);
            if let Some(sp) = bond.find(&bra_l) {
                out.push((sp, s));
            }
        }
        out
    };
    let bond_pairs: Vec<Vec<(usize, usize)>> = (0..n).map(|i| pairs_on(&sym.bonds[i])).collect();
    let bond_dims: Vec<usize> = (0..n)
        .map(|i| {
            bond_pairs[i]
                .iter()
                .map(|&(sp, s)| sym.bonds[i].size(sp) * sym.bonds[i].size(s))
                .sum()
        })
        .collect();
    let start = (0..n).min_by_key(|&i| bond_dims[i]).unwrap_or(0);
    let r = bond_dims[start];
    if r == 0 {
        return ZERO;
    }

    // Current batch of `r` ShiftedMatrix-like blocks, indexed by pair index `p_idx`
    // in `bond_pairs[cur]`: `mats[p_idx]` has shape `(r * d_sp) × d_s` row-major.
    let mut cur = start;
    let mut mats: Vec<Vec<C64>> = {
        let mut out = Vec::with_capacity(bond_pairs[start].len());
        let mut basis_idx = 0;
        for &(sp, s) in &bond_pairs[start] {
            let (dp, ds) = (sym.bonds[start].size(sp), sym.bonds[start].size(s));
            let mut m = vec![ZERO; r * dp * ds];
            for a in 0..dp {
                for b in 0..ds {
                    m[(basis_idx * dp + a) * ds + b] = ONE;
                    basis_idx += 1;
                }
            }
            out.push(m);
        }
        out
    };

    for _ in 0..n {
        let next = (cur + 1) % n;
        let sb =
            PeriodicSiteBlocks::new(&tensors[cur], &sym.bonds[cur], &sym.bonds[next], sym, cur);
        let next_pairs = &bond_pairs[next];
        let mut next_mats: Vec<Vec<C64>> = next_pairs
            .iter()
            .map(|&(tp, t)| vec![ZERO; r * sym.bonds[next].size(tp) * sym.bonds[next].size(t)])
            .collect();

        for (p_idx, &(sp, s)) in bond_pairs[cur].iter().enumerate() {
            let (dp, ds) = (sym.bonds[cur].size(sp), sym.bonds[cur].size(s));
            let x_mat = MatRef::from_row_major_slice(&mats[p_idx], r * dp, ds);
            for p in 0..2 {
                let (Some((tp, a_bra)), Some((t, a_ket))) = (sb.block(sp, p), sb.block(s, p))
                else {
                    continue;
                };
                let Ok(next_idx) = next_pairs.binary_search_by_key(&t, |pair| pair.1) else {
                    continue;
                };
                debug_assert_eq!(next_pairs[next_idx].0, tp);
                let (dtp, dt) = (sym.bonds[next].size(tp), sym.bonds[next].size(t));
                // tmp[(j, a), b] = Σ_c X[(j, a), c] A_ket[c, b]
                let tmp = matmul(x_mat, a_ket); // (r * dp) × dt
                // For each j, acc_j = A_bra† * tmp_j.
                // Permute tmp to dp × (r * dt) and do one matmul with A_bra† (dtp × dp).
                let cols = r * dt;
                let mut z = vec![ZERO; dp * cols];
                for j in 0..r {
                    for a in 0..dp {
                        for b in 0..dt {
                            z[a * cols + j * dt + b] = tmp[(j * dp + a, b)];
                        }
                    }
                }
                let w = matmul(a_bra.adjoint(), MatRef::from_row_major_slice(&z, dp, cols));
                let dst = &mut next_mats[next_idx];
                for bp in 0..dtp {
                    for j in 0..r {
                        let out_row = &mut dst[(j * dtp + bp) * dt..][..dt];
                        for b in 0..dt {
                            out_row[b] += w[(bp, j * dt + b)];
                        }
                    }
                }
            }
        }
        mats = next_mats;
        cur = next;
    }

    let mut trace = ZERO;
    let mut basis_idx = 0;
    for (p_idx, &(sp, s)) in bond_pairs[start].iter().enumerate() {
        let (dp, ds) = (sym.bonds[start].size(sp), sym.bonds[start].size(s));
        for a in 0..dp {
            for b in 0..ds {
                trace += mats[p_idx][(basis_idx * dp + a) * ds + b];
                basis_idx += 1;
            }
        }
    }
    trace
}

/// Low-rank Pippan–White–Evertz factorization of a transfer matrix `E_opp`
/// across the complement window `opp = [u_0, ..., u_{L-1}]`:
///
///   `E_opp[(γ', γ), (α', α)] ≈ Σ_{μ=0}^{rank-1} R_μ[γ', γ] * L_μ[α', α]`
///
/// where `γ', γ` live on the left bond of `u_0` (which is the right boundary of
/// the active window `W`), and `α', α` live on the right bond of `u_{L-1}`
/// (which is the left boundary of the active window `W`).
///
/// Notice the contraction convention for active window `W = [w_0, ..., w_{K-1}]`:
///
///   `⟨ψ| O_W ⊗ O_opp |ψ⟩ = Σ_μ Σ_{α', α, γ', γ} L_μ[α', α] E_W[(α', α), (γ', γ)] R_μ[γ', γ]`
///
/// so `L_μ` is the initial LEFT environment at the left boundary of `W` (`w_0`),
/// and `R_μ` is the initial RIGHT environment at the right boundary of `W` (`w_{K-1}`)!
#[derive(Clone, Debug)]
pub(super) struct CompressedTransfer {
    /// Left boundary matrices at cut `w_0` (shape `χ_{w_0} × χ_{w_0}` each).
    pub(super) left: Vec<Mat<C64>>,
    /// Right boundary matrices at cut `w_L` (shape `χ_{w_L} × χ_{w_L}` each).
    pub(super) right: Vec<Mat<C64>>,
    /// Relative first dropped singular value `σ_{rank} / σ_0` (0.0 if exact).
    pub(super) trunc_err: f64,
    /// For symmetric pMPS: `(delta_left, delta_right)` for each mode `μ`,
    /// where `L_μ` shifts ket→bra by `delta_left` (`s_bra = s_ket - delta_left`)
    /// and `R_μ` shifts bra→ket by `delta_right` (`t_ket = t_bra + delta_right`).
    pub(super) _deltas: Option<Vec<(Label, Label)>>,
}

impl CompressedTransfer {
    pub(super) fn rank(&self) -> usize {
        self.left.len()
    }

    /// The exact identity transfer matrix across an empty complement (e.g. `n == 2`),
    /// where `w_L == w_0` has bond dimension `d`: `E[(γ', γ), (α', α)] = δ_{γ', α'} δ_{γ, α}`.
    pub(super) fn identity(d: usize, sym: Option<(&Sectors, &Charges)>) -> Self {
        if let Some((sectors, charges)) = sym {
            let mut left = Vec::new();
            let mut right = Vec::new();
            let mut deltas = Vec::new();
            for sp in 0..sectors.len() {
                for s in 0..sectors.len() {
                    let delta = charges.sub(sectors.label(s), sectors.label(sp));
                    for a in sectors.range(sp) {
                        for b in sectors.range(s) {
                            let mut e = Mat::<C64>::zeros(d, d);
                            e[(a, b)] = ONE;
                            left.push(e.clone());
                            right.push(e);
                            deltas.push((delta.clone(), delta.clone()));
                        }
                    }
                }
            }
            return CompressedTransfer {
                left,
                right,
                trunc_err: 0.0,
                _deltas: Some(deltas),
            };
        }
        let mut left = Vec::with_capacity(d * d);
        let mut right = Vec::with_capacity(d * d);
        for a in 0..d {
            for b in 0..d {
                let mut e = Mat::<C64>::zeros(d, d);
                e[(a, b)] = ONE;
                left.push(e.clone());
                right.push(e);
            }
        }
        CompressedTransfer {
            left,
            right,
            trunc_err: 0.0,
            _deltas: None,
        }
    }
}

/// Compression settings for PWE transfer-matrix factorization.
#[derive(Clone, Copy, Debug)]
pub(super) struct TransferCompression {
    /// `Some(0)` forces exact full-rank SVD (no randomized approximation, no
    /// rank cap). `Some(r)` caps rank at `r`. `None` uses adaptive rank.
    pub(super) max_rank: Option<usize>,
    /// Drop singular values `< cutoff * s[0]`.
    pub(super) cutoff: f64,
}

/// Compress the transfer matrix of a single operator string `ops` (one `Local`
/// per site of `opp`) across the complement sites `opp = [u_0, ..., u_{L-1}]`
/// (with `u_0 = w_L` and `u_{L-1} = w_0 - 1 mod n`).
///
/// The matrix `M` has rows `(γ', γ)` (dim `d_r²` at cut `u_0`) and columns
/// `(α', α)` (dim `d_l²` at cut `w_0`).
/// Applying `M` to a batch of column vectors `v` (`d_l²`) runs right-to-left
/// from `u_{L-1}` down to `u_0` using `batch_transfer_left`!
pub(super) fn compress_string_transfer(
    tensors: &[Tensor],
    opp_sites: &[usize],
    seam_site: usize,
    ops: &[Local],
    scale: C64,
    comp: TransferCompression,
    sym: Option<&PeriodicSymmetry>,
) -> CompressedTransfer {
    debug_assert_eq!(opp_sites.len(), ops.len());
    if opp_sites.is_empty() {
        let d = tensors[seam_site].left;
        let mut ct =
            CompressedTransfer::identity(d, sym.map(|s| (&s.bonds[seam_site], s.charges.as_ref())));
        if scale != ONE {
            for m in &mut ct.left {
                *m *= faer::Scale(scale);
            }
        }
        return ct;
    }
    if let Some(sym) = sym {
        return compress_string_transfer_symmetric(tensors, opp_sites, ops, scale, comp, sym);
    }

    let d_r = tensors[opp_sites[0]].left;
    let d_l = tensors[*opp_sites.last().unwrap()].right;
    let m_rows = d_r * d_r;
    let n_cols = d_l * d_l;

    let apply = |v: &[C64], batch: usize| -> Vec<C64> {
        let mut cur = v.to_vec();
        for (&site, &op) in opp_sites.iter().zip(ops).rev() {
            cur = batch_transfer_left(&cur, batch, &tensors[site], &tensors[site], op);
        }
        if scale != ONE {
            cur.iter_mut().for_each(|z| *z *= scale);
        }
        cur
    };
    let apply_adj = |y: &[C64], batch: usize| -> Vec<C64> {
        let mut cur = y.to_vec();
        if scale != ONE {
            let sc = scale.conj();
            cur.iter_mut().for_each(|z| *z *= sc);
        }
        for (&site, &op) in opp_sites.iter().zip(ops) {
            cur = batch_transfer_left_adjoint(&cur, batch, &tensors[site], &tensors[site], op);
        }
        cur
    };

    let short_opp = tensors.len() <= SHORT_RING_QUBITS;
    compress_linear_map(m_rows, n_cols, d_r, d_l, apply, apply_adj, comp, short_opp)
}

/// Compress the transfer matrix of an open-boundary MPO `mpo` defined on the
/// complement sites `opp_sites` (length `mpo.sites.len()`).
pub(super) fn compress_mpo_transfer(
    tensors: &[Tensor],
    opp_sites: &[usize],
    mpo: &Mpo,
    comp: TransferCompression,
    sym: Option<&PeriodicSymmetry>,
) -> CompressedTransfer {
    debug_assert_eq!(opp_sites.len(), mpo.sites.len());
    if opp_sites.is_empty() {
        return CompressedTransfer {
            left: Vec::new(),
            right: Vec::new(),
            trunc_err: 0.0,
            _deltas: sym.map(|_| Vec::new()),
        };
    }
    if let Some(sym) = sym {
        return compress_mpo_transfer_symmetric(tensors, opp_sites, mpo, comp, sym);
    }

    let d_r = tensors[opp_sites[0]].left;
    let d_l = tensors[*opp_sites.last().unwrap()].right;
    let m_rows = d_r * d_r;
    let n_cols = d_l * d_l;

    let apply = |v: &[C64], batch: usize| -> Vec<C64> {
        let mut cur = vec![v.to_vec()]; // right boundary of MPO has 1 channel (DONE)
        for (&site, w_site) in opp_sites.iter().zip(&mpo.sites).rev() {
            cur = batch_mpo_transfer_left(&cur, batch, &tensors[site], w_site);
        }
        debug_assert_eq!(cur.len(), 1);
        cur.pop().unwrap()
    };
    let apply_adj = |y: &[C64], batch: usize| -> Vec<C64> {
        let mut cur = vec![y.to_vec()]; // left boundary of MPO has 1 channel (START)
        for (&site, w_site) in opp_sites.iter().zip(&mpo.sites) {
            cur = batch_mpo_transfer_left_adjoint(&cur, batch, &tensors[site], w_site);
        }
        debug_assert_eq!(cur.len(), 1);
        cur.pop().unwrap()
    };

    let short_opp = tensors.len() <= SHORT_RING_QUBITS;
    compress_linear_map(m_rows, n_cols, d_r, d_l, apply, apply_adj, comp, short_opp)
}

/// Core exact / progressive randomized SVD compressor for a matrix `M` of shape
/// `m_rows × n_cols` (`d_r² × d_l²`), returning `Σ_μ R_μ ⊗ L_μ` where
/// `M[row, col] = Σ_μ R_μ[row] * L_μ[col]`.
#[allow(clippy::too_many_arguments)]
fn compress_linear_map(
    m_rows: usize,
    n_cols: usize,
    d_r: usize,
    d_l: usize,
    apply: impl Fn(&[C64], usize) -> Vec<C64>,
    apply_adj: impl Fn(&[C64], usize) -> Vec<C64>,
    comp: TransferCompression,
    short_opp: bool,
) -> CompressedTransfer {
    let min_dim = m_rows.min(n_cols);
    if min_dim == 0 {
        return CompressedTransfer {
            left: Vec::new(),
            right: Vec::new(),
            trunc_err: 0.0,
            _deltas: None,
        };
    }

    let exact_dim = if short_opp {
        SHORT_EXACT_TRANSFER_DIM
    } else {
        EXACT_TRANSFER_DIM
    };
    let exact_mode = match comp.max_rank {
        Some(0) => true,
        Some(r) => {
            let oversample = r.clamp(4, RSVD_OVERSAMPLE);
            min_dim <= exact_dim || 4 * (r + oversample) >= min_dim
        }
        None => min_dim <= exact_dim,
    };
    let (u, s, v) = if exact_mode {
        if n_cols <= m_rows {
            // Apply to all n_cols standard basis vectors.
            let mut eye = vec![ZERO; n_cols * n_cols];
            for j in 0..n_cols {
                eye[j * n_cols + j] = ONE;
            }
            let cols = apply(&eye, n_cols);
            // cols[j * m_rows + i] = M[i, j].
            let mat = Mat::<C64>::from_fn(m_rows, n_cols, |i, j| cols[j * m_rows + i]);
            thin_svd(mat.as_ref())
        } else {
            // Apply adjoint to all m_rows standard basis vectors: M† has cols M†[:, i].
            let mut eye = vec![ZERO; m_rows * m_rows];
            for i in 0..m_rows {
                eye[i * m_rows + i] = ONE;
            }
            let rows_conj = apply_adj(&eye, m_rows);
            // rows_conj[i * n_cols + j] = (M†)[j, i] = conj(M[i, j]).
            let mat = Mat::<C64>::from_fn(m_rows, n_cols, |i, j| rows_conj[i * n_cols + j].conj());
            thin_svd(mat.as_ref())
        }
    } else if let Some(r) = comp.max_rank {
        let target = r.min(min_dim);
        let oversample = r.clamp(4, RSVD_OVERSAMPLE);
        let probe = (target + oversample).min(min_dim);
        randomized_svd(m_rows, n_cols, probe, &apply, &apply_adj)
    } else {
        // Progressive O(p L χ³) adaptive randomized SVD: grows 4 → 8 → 16 → 24
        // incrementally, stopping as soon as σ_p / σ_1 ≤ transfer_cutoff!
        randomized_svd_adaptive(m_rows, n_cols, &apply, &apply_adj, comp.cutoff, short_opp)
    };

    if s.is_empty() || s[0] == 0.0 {
        return CompressedTransfer {
            left: Vec::new(),
            right: Vec::new(),
            trunc_err: 0.0,
            _deltas: None,
        };
    }

    let thresh = comp.cutoff * s[0];
    let keep = match comp.max_rank {
        Some(0) => s.iter().take_while(|&&sv| sv > thresh).count().max(1),
        Some(r) => s
            .iter()
            .take(r.min(s.len()))
            .take_while(|&&sv| sv > thresh)
            .count()
            .max(1),
        None => {
            let cap = if short_opp {
                ADAPTIVE_HIGH_RANK
            } else {
                DEFAULT_MAX_PWE_RANK
            };
            let mut k = s
                .iter()
                .take(cap.min(s.len()))
                .take_while(|&&sv| sv > thresh)
                .count()
                .max(1);
            if exact_mode && short_opp {
                let safe_floor = (1e-10 * s[0]).max(thresh);
                while k < s.len() && s[k] > safe_floor {
                    k += 1;
                }
            } else {
                // On multi-flux-sector gauge rings where transfer eigenvalues
                // haven't decayed by rank 24, keep modes above `1e-3 * s[0]`
                // (up to 48) so the periodic norm metric `Ñ` stays accurate.
                let flux_floor = (1e-3 * s[0]).max(thresh);
                while k < s.len().min(48) && s[k] > flux_floor {
                    k += 1;
                }
            }
            k
        }
    };
    let trunc_err = if keep < s.len() {
        s[keep] / s[0]
    } else if !exact_mode && s.len() < min_dim {
        *s.last().unwrap() / s[0]
    } else {
        0.0
    };

    // M[row, col] = Σ_μ u[row, μ] * s[μ] * conj(v[col, μ])
    // Put sqrt(s[μ]) on each side: R_μ[row] = sqrt(s[μ]) u[row, μ], L_μ[col] = sqrt(s[μ]) conj(v[col, μ]).
    let mut left = Vec::with_capacity(keep);
    let mut right = Vec::with_capacity(keep);
    for mu in 0..keep {
        let rt = s[mu].sqrt();
        let r_mat = Mat::<C64>::from_fn(d_r, d_r, |a, b| u[(a * d_r + b, mu)] * rt);
        let l_mat = Mat::<C64>::from_fn(d_l, d_l, |a, b| v[(a * d_l + b, mu)].conj() * rt);
        right.push(r_mat);
        left.push(l_mat);
    }
    CompressedTransfer {
        left,
        right,
        trunc_err,
        _deltas: None,
    }
}

/// Progressive Halko–Martinsson–Tropp adaptive randomized SVD: starts at a
/// small probe (`target = 4`) and incrementally extends the orthonormal range
/// basis `Q` and projected matrix `B = Q† M` through `4 → 8 → 16 → 24`
/// (up to `ADAPTIVE_HIGH_RANK` on short complements), stopping as soon as
/// `σ_target / σ_0 ≤ cutoff` without recomputing earlier matvecs.
fn randomized_svd_adaptive(
    m_rows: usize,
    n_cols: usize,
    apply: &impl Fn(&[C64], usize) -> Vec<C64>,
    apply_adj: &impl Fn(&[C64], usize) -> Vec<C64>,
    cutoff: f64,
    short_opp: bool,
) -> (Mat<C64>, Vec<f64>, Mat<C64>) {
    let min_dim = m_rows.min(n_cols);
    let stop_tol = cutoff.max(1e-12);
    let mut seed: u64 = 0x517C_C1B7_2722_0A95 ^ (m_rows as u64).wrapping_mul(31) ^ (n_cols as u64);
    let mut next_f64 = || {
        seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    };

    let stages: &[usize] = if short_opp {
        &[4, 8, 16, DEFAULT_MAX_PWE_RANK, ADAPTIVE_HIGH_RANK]
    } else {
        &ADAPTIVE_RANK_STAGES
    };

    let mut q_acc: Option<Mat<C64>> = None;
    let mut b_acc: Option<Mat<C64>> = None;
    let mut last_svd: Option<(Mat<C64>, Vec<f64>, Mat<C64>)> = None;

    for (stage_idx, &raw_target) in stages.iter().enumerate() {
        // Past DEFAULT_MAX_PWE_RANK on general rings, only grow further if the
        // transfer spectrum still has large O(10^-3) multi-flux modes.
        if !short_opp && raw_target > DEFAULT_MAX_PWE_RANK {
            let need_more = last_svd.as_ref().is_some_and(|(_, s, _)| {
                let prev = stages[stage_idx - 1];
                let floor = if raw_target >= 48 { 5e-3 } else { 1e-3 };
                s.len() >= prev && s[0] > 0.0 && s[prev - 1] > floor * s[0]
            });
            if !need_more {
                break;
            }
        }
        let target = raw_target.min(min_dim);
        let oversample = if target <= 4 {
            4
        } else {
            RSVD_OVERSAMPLE.min(target)
        };
        let desired_probe = (target + oversample).min(min_dim);
        let cur_k = q_acc.as_ref().map_or(0, |q| q.ncols());
        let delta = desired_probe.saturating_sub(cur_k);

        if delta > 0 {
            let mut omega = vec![ZERO; delta * n_cols];
            for x in &mut omega {
                *x = C64::new(next_f64(), next_f64());
            }
            // Include the identity matrix as the first probe vector in stage 0
            // when square, guaranteeing the dominant fixed point is in span!
            if stage_idx == 0 {
                let d_l = (n_cols as f64).sqrt().round() as usize;
                if d_l * d_l == n_cols {
                    for x in &mut omega[..n_cols] {
                        *x = ZERO;
                    }
                    let inv = 1.0 / (d_l as f64).sqrt();
                    for a in 0..d_l {
                        omega[a * d_l + a] = C64::new(inv, 0.0);
                    }
                }
            }

            // Deflate `q_prev` out of a matrix `m_rows × cols` in two-pass MGS
            // so the power iteration targets the *deflated* operator
            // `M_defl = (I - Q_prev Q_prev†) M` and amplifies `σ_{k+1}, ...`
            // instead of re-collapsing onto the dominant modes `Q_prev`.
            let deflate_left = |mat: &mut Mat<C64>, q_prev: &Option<Mat<C64>>| {
                if let Some(qp) = q_prev {
                    for _ in 0..2 {
                        let proj = matmul(qp.adjoint(), mat.as_ref());
                        *mat -= matmul(qp.as_ref(), proj.as_ref());
                    }
                }
            };

            // 1 deflated power iteration on the new block of `delta` vectors.
            let y0_raw = apply(&omega, delta);
            let mut y0_mat = Mat::<C64>::from_fn(m_rows, delta, |r, c| y0_raw[c * m_rows + r]);
            deflate_left(&mut y0_mat, &q_acc);
            let s0_ref = last_svd
                .as_ref()
                .and_then(|(_, s, _)| s.first().copied())
                .unwrap_or(0.0);
            let (u0, s0_vec, _) = thin_svd(y0_mat.as_ref());
            let s0_max = s0_vec.first().copied().unwrap_or(0.0);
            let keep0 = if s0_max > 1e-14 * s0_ref && s0_max > 0.0 {
                let floor = (1e-12 * s0_max).max(1e-14 * s0_ref);
                s0_vec.iter().take_while(|&&sv| sv > floor).count()
            } else {
                0
            };

            let (u_new, keep_new) = if keep0 == 0 {
                (Mat::zeros(m_rows, 0), 0)
            } else {
                let q0 = u0.as_ref().subcols(0, keep0).to_owned();
                let q0_batch = cols_to_batch(q0.as_ref());
                let z = apply_adj(&q0_batch, keep0);
                let qz = orthonormal_cols(&z, n_cols, keep0);
                let qz_batch = cols_to_batch(qz.as_ref());
                let y1 = apply(&qz_batch, qz.ncols());
                let mut q_new = Mat::<C64>::from_fn(m_rows, qz.ncols(), |r, c| y1[c * m_rows + r]);
                deflate_left(&mut q_new, &q_acc);

                // Orthonormalise `q_new`, dropping numerically exhausted directions.
                let (u_new, s_new, _) = thin_svd(q_new.as_ref());
                let s_max = s_new.first().copied().unwrap_or(0.0);
                let ref_scale = if s0_ref > 0.0 { s0_ref } else { s_max };
                let keep_new = if s_max > 1e-14 * ref_scale && s_max > 0.0 {
                    let floor = (1e-12 * s_max).max(1e-14 * ref_scale);
                    s_new.iter().take_while(|&&sv| sv > floor).count()
                } else {
                    0
                };
                (u_new, keep_new)
            };

            if keep_new > 0 {
                let q_ext = u_new.as_ref().subcols(0, keep_new).to_owned();
                let q_ext_batch = cols_to_batch(q_ext.as_ref());
                let bh_ext = apply_adj(&q_ext_batch, keep_new);
                let b_ext =
                    Mat::<C64>::from_fn(keep_new, n_cols, |i, j| bh_ext[i * n_cols + j].conj());

                match (q_acc.take(), b_acc.take()) {
                    (None, None) => {
                        q_acc = Some(q_ext);
                        b_acc = Some(b_ext);
                    }
                    (Some(q_prev), Some(b_prev)) => {
                        let k0 = q_prev.ncols();
                        let k_tot = k0 + keep_new;
                        let mut q_cat = Mat::<C64>::zeros(m_rows, k_tot);
                        q_cat.as_mut().subcols_mut(0, k0).copy_from(q_prev.as_ref());
                        q_cat
                            .as_mut()
                            .subcols_mut(k0, keep_new)
                            .copy_from(q_ext.as_ref());
                        let mut b_cat = Mat::<C64>::zeros(k_tot, n_cols);
                        b_cat.as_mut().subrows_mut(0, k0).copy_from(b_prev.as_ref());
                        b_cat
                            .as_mut()
                            .subrows_mut(k0, keep_new)
                            .copy_from(b_ext.as_ref());
                        q_acc = Some(q_cat);
                        b_acc = Some(b_cat);
                    }
                    _ => unreachable!(),
                }
            }
        }

        let Some(ref b_mat) = b_acc else {
            return (Mat::zeros(m_rows, 0), Vec::new(), Mat::zeros(n_cols, 0));
        };
        let (ub, s, v) = thin_svd(b_mat.as_ref());
        let decayed = s.is_empty()
            || s[0] == 0.0
            || (s.len() >= target && s[target - 1] <= stop_tol * s[0])
            || b_mat.nrows() >= min_dim;
        let exhausted = delta > 0 && q_acc.as_ref().map_or(0, |q| q.ncols()) == cur_k;
        last_svd = Some((ub, s, v));
        if decayed || exhausted || target >= min_dim {
            break;
        }
    }

    let (ub, s, v) = last_svd.unwrap();
    let q = q_acc.unwrap();
    let u = matmul(q.as_ref(), ub.as_ref());
    (u, s, v)
}

/// Matrix-free Halko–Martinsson–Tropp randomized SVD with 2 subspace power
/// iterations, costing `O(probe * matvec_cost)`.
fn randomized_svd(
    m_rows: usize,
    n_cols: usize,
    probe: usize,
    apply: &impl Fn(&[C64], usize) -> Vec<C64>,
    apply_adj: &impl Fn(&[C64], usize) -> Vec<C64>,
) -> (Mat<C64>, Vec<f64>, Mat<C64>) {
    // Deterministic pseudo-random test matrix Ω of shape probe × n_cols (batch of probe vectors).
    let mut seed: u64 = 0x517C_C1B7_2722_0A95 ^ (m_rows as u64).wrapping_mul(31) ^ (n_cols as u64);
    let mut next_f64 = || {
        seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    };
    let mut omega = vec![ZERO; probe * n_cols];
    for x in &mut omega {
        *x = C64::new(next_f64(), next_f64());
    }
    // Include the identity matrix as the first probe vector when square,
    // guaranteeing the dominant fixed point is always in the span!
    let d_l = (n_cols as f64).sqrt().round() as usize;
    if d_l * d_l == n_cols && probe > 0 {
        for x in &mut omega[..n_cols] {
            *x = ZERO;
        }
        let inv = 1.0 / (d_l as f64).sqrt();
        for a in 0..d_l {
            omega[a * d_l + a] = C64::new(inv, 0.0);
        }
    }

    // Y = M Ω (probe vectors of length m_rows).
    let mut y = apply(&omega, probe);
    let mut q = orthonormal_cols(&y, m_rows, probe);

    // 1 power iteration (4 batched passes total): Z = M† Q, orth(Z), Y = M Z, orth(Y).
    {
        let q_batch = cols_to_batch(q.as_ref());
        let z = apply_adj(&q_batch, q.ncols());
        let qz = orthonormal_cols(&z, n_cols, q.ncols());
        let qz_batch = cols_to_batch(qz.as_ref());
        y = apply(&qz_batch, qz.ncols());
        q = orthonormal_cols(&y, m_rows, qz.ncols());
    }

    // B = Q† M  <=>  B† = M† Q (shape n_cols × k).
    let k = q.ncols();
    let q_batch = cols_to_batch(q.as_ref());
    let bh_batch = apply_adj(&q_batch, k);
    // B[i, j] = conj((M† Q)[j, i]) = conj(bh_batch[i * n_cols + j]).
    let b = Mat::<C64>::from_fn(k, n_cols, |i, j| bh_batch[i * n_cols + j].conj());
    let (ub, s, v) = thin_svd(b.as_ref());
    let u = matmul(q.as_ref(), ub.as_ref());
    (u, s, v)
}

fn orthonormal_cols(batch_vecs: &[C64], rows: usize, ncols: usize) -> Mat<C64> {
    let m = Mat::<C64>::from_fn(rows, ncols, |r, c| batch_vecs[c * rows + r]);
    thin_qr(m.as_ref()).0
}

fn cols_to_batch(q: MatRef<'_, C64>) -> Vec<C64> {
    let (rows, cols) = (q.nrows(), q.ncols());
    let mut out = vec![ZERO; cols * rows];
    for c in 0..cols {
        for r in 0..rows {
            out[c * rows + r] = q[(r, c)];
        }
    }
    out
}

/// Block-sparse symmetric compression of a single operator string across `opp_sites`.
/// Decomposes into independent `(Δq_R, Δq_L)` blocks where `Δq_L = Δq_R - δ_op`,
/// pruning any `Δq` that vanishes on an intermediate bond of `opp_sites` and
/// contracting each surviving block strictly within its charge sectors in parallel.
fn compress_string_transfer_symmetric(
    tensors: &[Tensor],
    opp_sites: &[usize],
    ops: &[Local],
    scale: C64,
    comp: TransferCompression,
    sym: &PeriodicSymmetry,
) -> CompressedTransfer {
    let first_site = opp_sites[0];
    let last_site = *opp_sites.last().unwrap();
    let right_bond = sym.left_bond(first_site); // cut w_L
    let left_bond = sym.right_bond(last_site); // cut w_0
    let (d_r, d_l) = (right_bond.dim(), left_bond.dim());
    let l_opp = opp_sites.len();
    let short_opp = tensors.len() <= SHORT_RING_QUBITS;

    // On tiny rings (`n <= 12`), bond dimensions are small and dense-block
    // transfer has negligible cost while keeping exact canonical FP ordering.
    if short_opp {
        let mut delta_op = sym.charges.zero();
        for (&site, &op) in opp_sites.iter().zip(ops) {
            let step = local_charge_change(site, op, &sym.charges);
            delta_op = sym.charges.add(&delta_op, &step);
        }
        let mut delta_pairs: Vec<(Label, Label)> = Vec::new();
        let mut seen_dl: BTreeSet<Label> = BTreeSet::new();
        for sp in 0..left_bond.len() {
            for s in 0..left_bond.len() {
                let dl = sym.charges.sub(left_bond.label(s), left_bond.label(sp));
                if seen_dl.insert(dl.clone()) {
                    let dr = sym.charges.add(&dl, &delta_op);
                    if bond_has_delta(right_bond, &dr, &sym.charges) {
                        delta_pairs.push((dl, dr));
                    }
                }
            }
        }
        let mut modes: Vec<SymTransferMode> = Vec::new();
        for (dl, dr) in delta_pairs {
            let l_idx = delta_basis_indices(left_bond, &dl, &sym.charges);
            let r_idx = delta_basis_indices(right_bond, &dr, &sym.charges);
            let (nr, nl) = (r_idx.len(), l_idx.len());
            if nr == 0 || nl == 0 {
                continue;
            }
            let apply = |v: &[C64], batch: usize| -> Vec<C64> {
                let mut full = vec![ZERO; batch * d_l * d_l];
                for j in 0..batch {
                    for (pos, &(a, b)) in l_idx.iter().enumerate() {
                        full[(j * d_l + a) * d_l + b] = v[j * nl + pos];
                    }
                }
                for (&site, &op) in opp_sites.iter().zip(ops).rev() {
                    full = batch_transfer_left(&full, batch, &tensors[site], &tensors[site], op);
                }
                let mut out = vec![ZERO; batch * nr];
                for j in 0..batch {
                    for (pos, &(a, b)) in r_idx.iter().enumerate() {
                        out[j * nr + pos] = full[(j * d_r + a) * d_r + b] * scale;
                    }
                }
                out
            };
            let apply_adj = |y: &[C64], batch: usize| -> Vec<C64> {
                let sc = scale.conj();
                let mut full = vec![ZERO; batch * d_r * d_r];
                for j in 0..batch {
                    for (pos, &(a, b)) in r_idx.iter().enumerate() {
                        full[(j * d_r + a) * d_r + b] = y[j * nr + pos] * sc;
                    }
                }
                for (&site, &op) in opp_sites.iter().zip(ops) {
                    full = batch_transfer_left_adjoint(
                        &full,
                        batch,
                        &tensors[site],
                        &tensors[site],
                        op,
                    );
                }
                let mut out = vec![ZERO; batch * nl];
                for j in 0..batch {
                    for (pos, &(a, b)) in l_idx.iter().enumerate() {
                        out[j * nl + pos] = full[(j * d_l + a) * d_l + b];
                    }
                }
                out
            };
            let (u, s, v) = factor_subblock(nr, nl, &apply, &apply_adj, comp, true);
            for mu in 0..s.len() {
                if s[mu] == 0.0 {
                    continue;
                }
                let rt = s[mu].sqrt();
                let mut r_mat = Mat::<C64>::zeros(d_r, d_r);
                for (pos, &(a, b)) in r_idx.iter().enumerate() {
                    r_mat[(a, b)] = u[(pos, mu)] * rt;
                }
                let mut l_mat = Mat::<C64>::zeros(d_l, d_l);
                for (pos, &(a, b)) in l_idx.iter().enumerate() {
                    l_mat[(a, b)] = v[(pos, mu)].conj() * rt;
                }
                modes.push((s[mu], dl.clone(), dr.clone(), l_mat, r_mat));
            }
        }
        return select_symmetric_modes(modes, comp, true);
    }

    // Precompute per-site operator charge changes δ_q(op).
    let op_deltas: Vec<Label> = opp_sites
        .iter()
        .zip(ops)
        .map(|(&site, &op)| local_charge_change(site, op, &sym.charges))
        .collect();

    // Bonds along `opp_sites`: `opp_bonds[k]` is left of `opp_sites[k]` (`k < l_opp`),
    // and `opp_bonds[l_opp]` is right of `opp_sites[l_opp - 1]` (`left_bond`).
    let opp_bonds: Vec<&Sectors> = (0..l_opp)
        .map(|k| sym.left_bond(opp_sites[k]))
        .chain(std::iter::once(left_bond))
        .collect();

    // Collect candidate Δq_L on `left_bond` and prune any that vanish at any
    // intermediate bond along `opp_sites`.
    let pairs_for_delta = |bond: &Sectors, delta: &Label| -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        for s in 0..bond.len() {
            let bra_l = sym.charges.sub(bond.label(s), delta);
            if let Some(sp) = bond.find(&bra_l) {
                out.push((sp, s));
            }
        }
        out
    };

    type SurvivingDelta = (Label, Label, Vec<Vec<(usize, usize)>>);
    let mut surviving: Vec<SurvivingDelta> = Vec::new();
    let mut seen_dl: BTreeSet<Label> = BTreeSet::new();
    for sp in 0..left_bond.len() {
        for s in 0..left_bond.len() {
            let dl = sym.charges.sub(left_bond.label(s), left_bond.label(sp));
            if !seen_dl.insert(dl.clone()) {
                continue;
            }
            // Walk right-to-left from `opp_bonds[l_opp]` to `opp_bonds[0]`.
            let mut cur_delta = dl.clone();
            let mut chain_pairs: Vec<Vec<(usize, usize)>> = vec![Vec::new(); l_opp + 1];
            chain_pairs[l_opp] = pairs_for_delta(opp_bonds[l_opp], &cur_delta);
            if chain_pairs[l_opp].is_empty() {
                continue;
            }
            let mut alive = true;
            for k in (0..l_opp).rev() {
                cur_delta = sym.charges.add(&cur_delta, &op_deltas[k]);
                let ps = pairs_for_delta(opp_bonds[k], &cur_delta);
                if ps.is_empty() {
                    alive = false;
                    break;
                }
                chain_pairs[k] = ps;
            }
            if alive {
                surviving.push((dl, cur_delta, chain_pairs));
            }
        }
    }

    if surviving.is_empty() {
        return CompressedTransfer {
            left: Vec::new(),
            right: Vec::new(),
            trunc_err: 0.0,
            _deltas: Some(Vec::new()),
        };
    }

    // Precompute PeriodicSiteBlocks once for all opp_sites (shared across Δq blocks!).
    let site_blocks: Vec<PeriodicSiteBlocks<'_>> = (0..l_opp)
        .map(|k| {
            let site = opp_sites[k];
            PeriodicSiteBlocks::new(&tensors[site], opp_bonds[k], opp_bonds[k + 1], sym, site)
        })
        .collect();

    let modes: Vec<SymTransferMode> = surviving
        .par_iter()
        .flat_map(|(dl, dr, chain_pairs)| {
            let l_pairs = &chain_pairs[l_opp];
            let r_pairs = &chain_pairs[0];
            let mut l_idx = Vec::new();
            for &(sp, s) in l_pairs {
                for a in left_bond.range(sp) {
                    for b in left_bond.range(s) {
                        l_idx.push((a, b));
                    }
                }
            }
            let mut r_idx = Vec::new();
            for &(sp, s) in r_pairs {
                for a in right_bond.range(sp) {
                    for b in right_bond.range(s) {
                        r_idx.push((a, b));
                    }
                }
            }
            let (nr, nl) = (r_idx.len(), l_idx.len());
            if nr == 0 || nl == 0 {
                return Vec::new();
            }

            // Block-sparse right-to-left transfer: `opp_bonds[l_opp]` -> `opp_bonds[0]`.
            let apply = |v: &[C64], batch: usize| -> Vec<C64> {
                let mut cur_mats: Vec<Vec<C64>> = Vec::with_capacity(l_pairs.len());
                let mut off = 0usize;
                for &(sp, s) in l_pairs {
                    let (dp, ds) = (left_bond.size(sp), left_bond.size(s));
                    let sz = dp * ds;
                    let mut m = vec![ZERO; batch * sz];
                    for j in 0..batch {
                        m[j * sz..(j + 1) * sz].copy_from_slice(&v[j * nl + off..][..sz]);
                    }
                    cur_mats.push(m);
                    off += sz;
                }
                for k in (0..l_opp).rev() {
                    let b_left = opp_bonds[k];
                    let b_right = opp_bonds[k + 1];
                    let sb = &site_blocks[k];
                    let right_p = &chain_pairs[k + 1];
                    let left_p = &chain_pairs[k];
                    let op = ops[k];
                    let op_rows = [op_entry(op, 0), op_entry(op, 1)];
                    let mut next_mats: Vec<Vec<C64>> = left_p
                        .iter()
                        .map(|&(sp, s)| vec![ZERO; batch * b_left.size(sp) * b_left.size(s)])
                        .collect();
                    for (p_idx, &(tp, t)) in right_p.iter().enumerate() {
                        let (dtp, dt) = (b_right.size(tp), b_right.size(t));
                        let x_mat = MatRef::from_row_major_slice(&cur_mats[p_idx], batch * dtp, dt);
                        for (p_out, entry) in op_rows.iter().enumerate() {
                            let Some((p_in, f)) = *entry else { continue };
                            let (Some(s), Some(sp)) =
                                (sb.left_of[t * 2 + p_in], sb.left_of[tp * 2 + p_out])
                            else {
                                continue;
                            };
                            let Ok(out_idx) = left_p.binary_search_by_key(&s, |pair| pair.1) else {
                                continue;
                            };
                            debug_assert_eq!(left_p[out_idx].0, sp);
                            let (_, a_ket) = sb.block(s, p_in).unwrap();
                            let (_, a_bra) = sb.block(sp, p_out).unwrap();
                            let (dsp, ds) = (b_left.size(sp), b_left.size(s));
                            // tmp[(j, bp), al] = Σ_b X[(j, bp), b] A_ket[al, b] -> (batch * dtp) × ds
                            let tmp = matmul(x_mat, a_ket.transpose());
                            let cols = batch * ds;
                            let mut z = vec![ZERO; dtp * cols];
                            for j in 0..batch {
                                for bp in 0..dtp {
                                    for al in 0..ds {
                                        z[bp * cols + j * ds + al] = tmp[(j * dtp + bp, al)];
                                    }
                                }
                            }
                            let w = matmul(
                                a_bra.conjugate(),
                                MatRef::from_row_major_slice(&z, dtp, cols),
                            );
                            let dst = &mut next_mats[out_idx];
                            if f == ONE {
                                for ap in 0..dsp {
                                    for j in 0..batch {
                                        let out_row = &mut dst[(j * dsp + ap) * ds..][..ds];
                                        for al in 0..ds {
                                            out_row[al] += w[(ap, j * ds + al)];
                                        }
                                    }
                                }
                            } else {
                                for ap in 0..dsp {
                                    for j in 0..batch {
                                        let out_row = &mut dst[(j * dsp + ap) * ds..][..ds];
                                        for al in 0..ds {
                                            out_row[al] += f * w[(ap, j * ds + al)];
                                        }
                                    }
                                }
                            }
                        }
                    }
                    cur_mats = next_mats;
                }
                let mut out = vec![ZERO; batch * nr];
                let mut out_off = 0usize;
                for (p_idx, &(sp, s)) in r_pairs.iter().enumerate() {
                    let sz = right_bond.size(sp) * right_bond.size(s);
                    let src = &cur_mats[p_idx];
                    for j in 0..batch {
                        let dst_slice = &mut out[j * nr + out_off..][..sz];
                        let src_slice = &src[j * sz..(j + 1) * sz];
                        if scale == ONE {
                            dst_slice.copy_from_slice(src_slice);
                        } else {
                            for (d, &val) in dst_slice.iter_mut().zip(src_slice) {
                                *d = val * scale;
                            }
                        }
                    }
                    out_off += sz;
                }
                out
            };

            // Adjoint: conj(T_right(conj(Y))) walking left-to-right `opp_bonds[0]` -> `opp_bonds[l_opp]`.
            let apply_adj = |y: &[C64], batch: usize| -> Vec<C64> {
                let mut cur_mats: Vec<Vec<C64>> = Vec::with_capacity(r_pairs.len());
                let mut off = 0usize;
                for &(sp, s) in r_pairs {
                    let (dp, ds) = (right_bond.size(sp), right_bond.size(s));
                    let sz = dp * ds;
                    let mut m = vec![ZERO; batch * sz];
                    for j in 0..batch {
                        let dst = &mut m[j * sz..(j + 1) * sz];
                        let src = &y[j * nr + off..][..sz];
                        if scale == ONE {
                            for (d, &val) in dst.iter_mut().zip(src) {
                                *d = val.conj();
                            }
                        } else {
                            for (d, &val) in dst.iter_mut().zip(src) {
                                *d = val.conj() * scale;
                            }
                        }
                    }
                    cur_mats.push(m);
                    off += sz;
                }
                for k in 0..l_opp {
                    let b_left = opp_bonds[k];
                    let b_right = opp_bonds[k + 1];
                    let sb = &site_blocks[k];
                    let left_p = &chain_pairs[k];
                    let right_p = &chain_pairs[k + 1];
                    let op = ops[k];
                    let op_rows = [op_entry(op, 0), op_entry(op, 1)];
                    let mut next_mats: Vec<Vec<C64>> = right_p
                        .iter()
                        .map(|&(tp, t)| vec![ZERO; batch * b_right.size(tp) * b_right.size(t)])
                        .collect();
                    for (p_idx, &(sp, s)) in left_p.iter().enumerate() {
                        let (dsp, ds) = (b_left.size(sp), b_left.size(s));
                        let x_mat = MatRef::from_row_major_slice(&cur_mats[p_idx], batch * dsp, ds);
                        for (p_out, entry) in op_rows.iter().enumerate() {
                            let Some((p_in, f)) = *entry else { continue };
                            let (Some((t, a_ket)), Some((tp, a_bra))) =
                                (sb.block(s, p_in), sb.block(sp, p_out))
                            else {
                                continue;
                            };
                            let Ok(out_idx) = right_p.binary_search_by_key(&t, |pair| pair.1)
                            else {
                                continue;
                            };
                            debug_assert_eq!(right_p[out_idx].0, tp);
                            let (dtp, dt) = (b_right.size(tp), b_right.size(t));
                            let tmp = matmul(x_mat, a_ket); // (batch * dsp) × dt
                            let cols = batch * dt;
                            let mut z = vec![ZERO; dsp * cols];
                            for j in 0..batch {
                                for ap in 0..dsp {
                                    for b in 0..dt {
                                        z[ap * cols + j * dt + b] = tmp[(j * dsp + ap, b)];
                                    }
                                }
                            }
                            let w = matmul(
                                a_bra.adjoint(),
                                MatRef::from_row_major_slice(&z, dsp, cols),
                            );
                            let dst = &mut next_mats[out_idx];
                            if f == ONE {
                                for bp in 0..dtp {
                                    for j in 0..batch {
                                        let out_row = &mut dst[(j * dtp + bp) * dt..][..dt];
                                        for b in 0..dt {
                                            out_row[b] += w[(bp, j * dt + b)];
                                        }
                                    }
                                }
                            } else {
                                for bp in 0..dtp {
                                    for j in 0..batch {
                                        let out_row = &mut dst[(j * dtp + bp) * dt..][..dt];
                                        for b in 0..dt {
                                            out_row[b] += f * w[(bp, j * dt + b)];
                                        }
                                    }
                                }
                            }
                        }
                    }
                    cur_mats = next_mats;
                }
                let mut out = vec![ZERO; batch * nl];
                let mut out_off = 0usize;
                for (p_idx, &(sp, s)) in l_pairs.iter().enumerate() {
                    let sz = left_bond.size(sp) * left_bond.size(s);
                    let src = &cur_mats[p_idx];
                    for j in 0..batch {
                        let dst_slice = &mut out[j * nl + out_off..][..sz];
                        let src_slice = &src[j * sz..(j + 1) * sz];
                        for (d, &val) in dst_slice.iter_mut().zip(src_slice) {
                            *d = val.conj();
                        }
                    }
                    out_off += sz;
                }
                out
            };

            let (u, s, v) = factor_subblock(nr, nl, &apply, &apply_adj, comp, short_opp);
            let mut local_modes = Vec::with_capacity(s.len());
            for mu in 0..s.len() {
                if s[mu] == 0.0 {
                    continue;
                }
                let rt = s[mu].sqrt();
                let mut r_mat = Mat::<C64>::zeros(d_r, d_r);
                for (pos, &(a, b)) in r_idx.iter().enumerate() {
                    r_mat[(a, b)] = u[(pos, mu)] * rt;
                }
                let mut l_mat = Mat::<C64>::zeros(d_l, d_l);
                for (pos, &(a, b)) in l_idx.iter().enumerate() {
                    l_mat[(a, b)] = v[(pos, mu)].conj() * rt;
                }
                local_modes.push((s[mu], dl.clone(), dr.clone(), l_mat, r_mat));
            }
            local_modes
        })
        .collect();
    select_symmetric_modes(modes, comp, short_opp)
}

/// Block-sparse symmetric compression of a charge-graded MPO across `opp_sites`.
/// The MPO's total charge change is `δ_mpo = channels[0][0]` (0 for `H_opp`, or
/// the crossing operator's definite charge change).
fn compress_mpo_transfer_symmetric(
    tensors: &[Tensor],
    opp_sites: &[usize],
    mpo: &Mpo,
    comp: TransferCompression,
    sym: &PeriodicSymmetry,
) -> CompressedTransfer {
    let first_site = opp_sites[0];
    let last_site = *opp_sites.last().unwrap();
    let right_bond = sym.left_bond(first_site);
    let left_bond = sym.right_bond(last_site);
    let (d_r, d_l) = (right_bond.dim(), left_bond.dim());
    let short_opp = tensors.len() <= SHORT_RING_QUBITS;

    let delta_op = mpo
        .channels
        .as_ref()
        .map(|ch| ch[0][0].clone())
        .unwrap_or_else(|| sym.charges.zero());

    let mut delta_pairs: Vec<(Label, Label)> = Vec::new();
    let mut seen_dl: BTreeSet<Label> = BTreeSet::new();
    for sp in 0..left_bond.len() {
        for s in 0..left_bond.len() {
            let dl = sym.charges.sub(left_bond.label(s), left_bond.label(sp));
            if seen_dl.insert(dl.clone()) {
                let dr = sym.charges.add(&dl, &delta_op);
                if bond_has_delta(right_bond, &dr, &sym.charges) {
                    delta_pairs.push((dl, dr));
                }
            }
        }
    }

    let mut modes: Vec<SymTransferMode> = Vec::new();
    for (dl, dr) in delta_pairs {
        let l_idx = delta_basis_indices(left_bond, &dl, &sym.charges);
        let r_idx = delta_basis_indices(right_bond, &dr, &sym.charges);
        let (nr, nl) = (r_idx.len(), l_idx.len());
        if nr == 0 || nl == 0 {
            continue;
        }
        let apply = |v: &[C64], batch: usize| -> Vec<C64> {
            let mut full = vec![ZERO; batch * d_l * d_l];
            for j in 0..batch {
                for (pos, &(a, b)) in l_idx.iter().enumerate() {
                    full[(j * d_l + a) * d_l + b] = v[j * nl + pos];
                }
            }
            let mut cur = vec![full];
            for (&site, w_site) in opp_sites.iter().zip(&mpo.sites).rev() {
                cur = batch_mpo_transfer_left(&cur, batch, &tensors[site], w_site);
            }
            let res = &cur[0];
            let mut out = vec![ZERO; batch * nr];
            for j in 0..batch {
                for (pos, &(a, b)) in r_idx.iter().enumerate() {
                    out[j * nr + pos] = res[(j * d_r + a) * d_r + b];
                }
            }
            out
        };
        let apply_adj = |y: &[C64], batch: usize| -> Vec<C64> {
            let mut full = vec![ZERO; batch * d_r * d_r];
            for j in 0..batch {
                for (pos, &(a, b)) in r_idx.iter().enumerate() {
                    full[(j * d_r + a) * d_r + b] = y[j * nr + pos];
                }
            }
            let mut cur = vec![full];
            for (&site, w_site) in opp_sites.iter().zip(&mpo.sites) {
                cur = batch_mpo_transfer_left_adjoint(&cur, batch, &tensors[site], w_site);
            }
            let res = &cur[0];
            let mut out = vec![ZERO; batch * nl];
            for j in 0..batch {
                for (pos, &(a, b)) in l_idx.iter().enumerate() {
                    out[j * nl + pos] = res[(j * d_l + a) * d_l + b];
                }
            }
            out
        };
        let (u, s, v) = factor_subblock(nr, nl, &apply, &apply_adj, comp, short_opp);
        for mu in 0..s.len() {
            if s[mu] == 0.0 {
                continue;
            }
            let rt = s[mu].sqrt();
            let mut r_mat = Mat::<C64>::zeros(d_r, d_r);
            for (pos, &(a, b)) in r_idx.iter().enumerate() {
                r_mat[(a, b)] = u[(pos, mu)] * rt;
            }
            let mut l_mat = Mat::<C64>::zeros(d_l, d_l);
            for (pos, &(a, b)) in l_idx.iter().enumerate() {
                l_mat[(a, b)] = v[(pos, mu)].conj() * rt;
            }
            modes.push((s[mu], dl.clone(), dr.clone(), l_mat, r_mat));
        }
    }
    select_symmetric_modes(modes, comp, short_opp)
}

fn factor_subblock(
    nr: usize,
    nl: usize,
    apply: &impl Fn(&[C64], usize) -> Vec<C64>,
    apply_adj: &impl Fn(&[C64], usize) -> Vec<C64>,
    comp: TransferCompression,
    short_opp: bool,
) -> (Mat<C64>, Vec<f64>, Mat<C64>) {
    let min_dim = nr.min(nl);
    let exact_dim = if short_opp {
        SHORT_EXACT_TRANSFER_DIM
    } else {
        EXACT_TRANSFER_DIM
    };
    let exact_mode = match comp.max_rank {
        Some(0) => true,
        Some(r) => {
            let oversample = r.clamp(4, RSVD_OVERSAMPLE);
            min_dim <= exact_dim || 4 * (r + oversample) >= min_dim
        }
        None => min_dim <= exact_dim,
    };
    if exact_mode {
        if nl <= nr {
            let mut eye = vec![ZERO; nl * nl];
            for j in 0..nl {
                eye[j * nl + j] = ONE;
            }
            let cols = apply(&eye, nl);
            let mat = Mat::<C64>::from_fn(nr, nl, |i, j| cols[j * nr + i]);
            thin_svd(mat.as_ref())
        } else {
            let mut eye = vec![ZERO; nr * nr];
            for i in 0..nr {
                eye[i * nr + i] = ONE;
            }
            let rows_conj = apply_adj(&eye, nr);
            let mat = Mat::<C64>::from_fn(nr, nl, |i, j| rows_conj[i * nl + j].conj());
            thin_svd(mat.as_ref())
        }
    } else if let Some(r) = comp.max_rank {
        let target = r.min(min_dim);
        let oversample = r.clamp(4, RSVD_OVERSAMPLE);
        let probe = (target + oversample).min(min_dim);
        randomized_svd(nr, nl, probe, apply, apply_adj)
    } else {
        randomized_svd_adaptive(nr, nl, apply, apply_adj, comp.cutoff, short_opp)
    }
}

fn select_symmetric_modes(
    mut modes: Vec<SymTransferMode>,
    comp: TransferCompression,
    short_opp: bool,
) -> CompressedTransfer {
    modes.sort_by(|a, b| b.0.total_cmp(&a.0));
    if modes.is_empty() || modes[0].0 == 0.0 {
        return CompressedTransfer {
            left: Vec::new(),
            right: Vec::new(),
            trunc_err: 0.0,
            _deltas: Some(Vec::new()),
        };
    }
    let s0 = modes[0].0;
    let thresh = comp.cutoff * s0;
    let keep = match comp.max_rank {
        Some(0) => modes.iter().take_while(|m| m.0 > thresh).count().max(1),
        Some(r) => modes
            .iter()
            .take(r.min(modes.len()))
            .take_while(|m| m.0 > thresh)
            .count()
            .max(1),
        None => {
            let cap = if short_opp {
                ADAPTIVE_HIGH_RANK
            } else {
                DEFAULT_MAX_PWE_RANK
            };
            let mut k = modes
                .iter()
                .take(cap.min(modes.len()))
                .take_while(|m| m.0 > thresh)
                .count()
                .max(1);
            if short_opp {
                let safe_floor = (1e-10 * s0).max(thresh);
                while k < modes.len() && modes[k].0 > safe_floor {
                    k += 1;
                }
            } else {
                // On multi-flux-sector rings (e.g. lattice gauge theories) with
                // near-degenerate transfer eigenvalues across sectors, avoid
                // dropping large O(10^-3) modes already computed across Δq blocks.
                let flux_floor = (1e-3 * s0).max(thresh);
                while k < modes.len().min(64) && modes[k].0 > flux_floor {
                    k += 1;
                }
            }
            k
        }
    };
    let trunc_err = if keep < modes.len() {
        modes[keep].0 / s0
    } else {
        0.0
    };
    modes.truncate(keep);
    let mut left = Vec::with_capacity(keep);
    let mut right = Vec::with_capacity(keep);
    let mut deltas = Vec::with_capacity(keep);
    for (_, dl, dr, lm, rm) in modes {
        left.push(lm);
        right.push(rm);
        deltas.push((dl, dr));
    }
    CompressedTransfer {
        left,
        right,
        trunc_err,
        _deltas: Some(deltas),
    }
}

fn bond_has_delta(bond: &Sectors, delta: &[i64], charges: &Charges) -> bool {
    for s in 0..bond.len() {
        let bra_l = charges.sub(bond.label(s), delta);
        if bond.find(&bra_l).is_some() {
            return true;
        }
    }
    false
}

fn delta_basis_indices(bond: &Sectors, delta: &[i64], charges: &Charges) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for s in 0..bond.len() {
        let bra_l = charges.sub(bond.label(s), delta);
        if let Some(sp) = bond.find(&bra_l) {
            for a in bond.range(sp) {
                for b in bond.range(s) {
                    out.push((a, b));
                }
            }
        }
    }
    out
}

pub(super) fn local_charge_change(q: usize, op: Local, charges: &Charges) -> Label {
    match op {
        Local::Lower => charges.sub(charges.site(q, 0), charges.site(q, 1)),
        Local::Raise => charges.sub(charges.site(q, 1), charges.site(q, 0)),
        _ => charges.zero(),
    }
}

/// Exact expectation values of many Pauli strings at once using prefix/suffix
/// transfer caches at cut `0` and (lazily) at the antipodal cut `n / 2`, so
/// local strings — including across the `(n - 1) → 0` seam — cost only their
/// cyclic span `O(w χ_cut² χ³)`.
pub(super) fn exact_expectations(
    tensors: &[Tensor],
    norm_sqr: f64,
    observables: &[crate::observable::PauliSum],
) -> Result<Vec<f64>, crate::error::Error> {
    let n = tensors.len();
    for o in observables {
        if o.min_qubits() > n {
            return Err(crate::error::Error::QubitOutOfRange {
                qubit: o.min_qubits() - 1,
                n_qubits: n,
            });
        }
    }
    let mut out = vec![0.0; observables.len()];
    let mut terms: Vec<ExpectationTerm> = Vec::new();
    for (k, o) in observables.iter().enumerate() {
        for (c, s) in o.terms() {
            if s.weight() == 0 {
                out[k] += c;
            } else {
                terms.push((k, *c, s.ops().to_vec()));
            }
        }
    }
    if terms.is_empty() {
        return Ok(out);
    }

    let mid = n / 2;
    // Choose cut 0 or cut `mid` for each term to minimise cyclic span.
    let mut cut0_terms = Vec::new();
    let mut cutm_terms = Vec::new();
    for (idx, (_, _, ops)) in terms.iter().enumerate() {
        let span0 = ops.last().unwrap().0 - ops.first().unwrap().0 + 1;
        let spanm = cyclic_span(ops, mid, n);
        if spanm < span0 {
            cutm_terms.push(idx);
        } else {
            cut0_terms.push(idx);
        }
    }

    let mut values = vec![ZERO; terms.len()];
    if !cut0_terms.is_empty() {
        let (prefix, suffix) = build_ring_caches(tensors, 0);
        let results: Vec<(usize, C64)> = cut0_terms
            .par_iter()
            .map(|&idx| {
                let ops = &terms[idx].2;
                let (f, l) = (ops[0].0, ops.last().unwrap().0);
                let val = eval_arc(tensors, 0, f, l - f + 1, ops, &prefix, &suffix);
                (idx, val)
            })
            .collect();
        for (idx, val) in results {
            values[idx] = val;
        }
    }
    if !cutm_terms.is_empty() {
        let (prefix, suffix) = build_ring_caches(tensors, mid);
        let results: Vec<(usize, C64)> = cutm_terms
            .par_iter()
            .map(|&idx| {
                let ops = &terms[idx].2;
                let (start_off, span) = cyclic_arc(ops, mid, n);
                let val = eval_arc(tensors, mid, start_off, span, ops, &prefix, &suffix);
                (idx, val)
            })
            .collect();
        for (idx, val) in results {
            values[idx] = val;
        }
    }

    for (idx, &(k, coeff, _)) in terms.iter().enumerate() {
        out[k] += coeff * values[idx].re / norm_sqr;
    }
    Ok(out)
}

/// Cyclic offset `(q + n - cut) % n` range `[start_off, start_off + span - 1]`.
fn cyclic_arc(ops: &[(usize, Pauli)], cut: usize, n: usize) -> (usize, usize) {
    let mut lo = n;
    let mut hi = 0;
    for &(q, _) in ops {
        let off = (q + n - cut) % n;
        lo = lo.min(off);
        hi = hi.max(off);
    }
    (lo, hi - lo + 1)
}

fn cyclic_span(ops: &[(usize, Pauli)], cut: usize, n: usize) -> usize {
    cyclic_arc(ops, cut, n).1
}

/// Build prefix `P[k] = T[cut] ... T[cut+k-1]` (`k = 0..=n`) and suffix
/// `S[k] = T[cut+k] ... T[cut+n-1]` (`k = 0..=n`) for cut `cut`.
/// `P[k]` is stored as `batch = d_cut²` matrices of shape `(d_k × d_k)`.
/// `S[k]` is stored as `batch = d_cut²` matrices of shape `(d_k × d_k)` (where
/// row `j = (a, b)` of `S[k]` is column `j` of `T[cut+k] ... T[cut+n-1]`,
/// computed right-to-left from `I_{d_cut²}`).
fn build_ring_caches(tensors: &[Tensor], cut: usize) -> (Vec<Vec<C64>>, Vec<Vec<C64>>) {
    let n = tensors.len();
    let d0 = tensors[cut].left;
    let batch = d0 * d0;
    let mut eye = vec![ZERO; batch * batch];
    for i in 0..batch {
        eye[i * batch + i] = ONE;
    }
    let mut prefix = Vec::with_capacity(n + 1);
    prefix.push(eye.clone());
    for step in 0..n {
        let q = (cut + step) % n;
        let next = batch_transfer_right(
            prefix.last().unwrap(),
            batch,
            &tensors[q],
            &tensors[q],
            Local::I,
        );
        prefix.push(next);
    }
    let mut suffix = vec![Vec::new(); n + 1];
    suffix[n] = eye;
    for step in (0..n).rev() {
        let q = (cut + step) % n;
        suffix[step] =
            batch_transfer_left(&suffix[step + 1], batch, &tensors[q], &tensors[q], Local::I);
    }
    (prefix, suffix)
}

/// Evaluate `Tr(P[start_off] * T[start_off..start_off+span](ops) * S[start_off+span])`.
fn eval_arc(
    tensors: &[Tensor],
    cut: usize,
    start_off: usize,
    span: usize,
    ops: &[(usize, Pauli)],
    prefix: &[Vec<C64>],
    suffix: &[Vec<C64>],
) -> C64 {
    let n = tensors.len();
    let d0 = tensors[cut].left;
    let batch = d0 * d0;
    let mut cur = prefix[start_off].clone();
    for step in 0..span {
        let q = (cut + start_off + step) % n;
        let p = ops
            .iter()
            .find(|&&(site, _)| site == q)
            .map_or(Local::I, |&(_, p)| Local::from(p));
        cur = batch_transfer_right(&cur, batch, &tensors[q], &tensors[q], p);
    }
    let end_s = &suffix[start_off + span];
    debug_assert_eq!(cur.len(), end_s.len());
    // Tr(X * S) = Σ_{j, γ', γ} X[j, γ', γ] * S[j, γ', γ]
    // Wait: check index pairing!
    // P[k] has P[k][j, (γ', γ)] = (T[cut..cut+k])_{j, (γ', γ)}.
    // S[k] applied `batch_transfer_left` to E_j at the RIGHT end, so
    // S[k][j, (γ', γ)] = (T[cut+k..cut+n])_{(γ', γ), j}.
    // Therefore Tr(X * S) = Σ_{j, (γ', γ)} X[j, (γ', γ)] * S[j, (γ', γ)]!
    cur.iter().zip(end_s).map(|(&a, &b)| a * b).sum()
}

/// Silence unused helpers if not referenced yet.
const _: fn(Pauli) -> [[C64; 2]; 2] = pauli_matrix;
