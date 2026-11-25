//! Cyclic window splitting of arbitrary `PauliSum` Hamiltonians, Pippan–White–Evertz
//! compressed window environments, and gauge-conditioned local effective operators.
//!
//! For an active cyclic window `W = [w_0, ..., w_{L-1}]` and its complement
//! `opp = [u_0, ..., u_{L_opp-1}]`, any Hamiltonian `H` decomposes as:
//!
//!   `H = shift * I + H_W ⊗ I_opp + I_W ⊗ H_opp + Σ_{κ=1}^K A_κ^(W) ⊗ B_κ^(opp)`
//!
//! Crossing terms are deduplicated by grouping identical `W`-side (or `opp`-side)
//! operators so local Hamiltonians (Heisenberg, TFIM, explicit-link lattice gauge
//! theories) have tiny `K = O(1)` independent of system size.
//!
//! Each complement operator (`I_opp`, `H_opp`, and `B_κ^(opp)`) is compressed
//! once per half-sweep by PWE SVD into low-rank boundary pairs `(L_μ, R_μ)`.
//! Inside `W`, environments are propagated in `O(p D χ³)` per site, and local
//! 1-site or 2-site solves are gauge-conditioned by the exact positive
//! semidefinite partial-trace metrics `L̄ = Σ_μ Tr(R_μ) L_μ` and
//! `R̄ = Σ_μ Tr(L_μ) R_μ`.

use super::symmetric::{PeriodicLayout, PeriodicSymmetry, block_metric_gauge};
use super::transfer::{
    CompressedTransfer, TransferCompression, batch_transfer_left, batch_transfer_right,
    compress_mpo_transfer, compress_string_transfer,
};
use crate::error::Error;
use crate::mps::charges::Charges;
use crate::mps::dmrg::{Env, update_left, update_right};
use crate::mps::linalg::{hermitian_evd, matmul, thin_qr};
use crate::mps::mpo::{Local, Mpo, Rows, op_entry, pair_keyed, right_keyed};
use crate::mps::{ONE, Tensor, ZERO};
use crate::observable::PauliSum;
use crate::statevector::krylov::LinearOperator;
use faer::linalg::matmul::matmul as faer_matmul;
use faer::{Accum, Mat, MatMut, MatRef, Par};
use num_complex::Complex64 as C64;
use rayon::prelude::*;
use std::cell::RefCell;
use std::collections::BTreeMap;

/// Reusable scratch buffers for local effective operators (`StackedPairOp`,
/// `BatchedMpoOp`, `PackedOp`) so Krylov/Davidson matvecs do zero heap
/// allocations in the hot loop.
#[derive(Default)]
struct LocalScratch {
    y: Vec<C64>,
    z: Vec<C64>,
}

#[derive(Default)]
struct PackedScratch {
    dense_in: Vec<C64>,
    dense_out: Vec<C64>,
}

thread_local! {
    static LOCAL_SCRATCH: RefCell<LocalScratch> = RefCell::new(LocalScratch::default());
    static PACKED_SCRATCH: RefCell<PackedScratch> = RefCell::new(PackedScratch::default());
}

fn ensure_len(buf: &mut Vec<C64>, len: usize) {
    if buf.len() < len {
        buf.resize(len, ZERO);
    }
}

/// A crossing term `A_κ^(W) ⊗ B_κ^(opp)` between active window `W` and
/// complement `opp`.
pub(super) enum WindowFactor {
    /// A single operator string on `W` (MPO bond dimension 1 — zero MPO overhead!).
    String(Vec<Local>),
    /// A general mini-MPO on `W`.
    Mpo(Mpo),
}

pub(super) enum OppFactor {
    /// A single operator string on `opp` with scalar factor.
    String(C64, Vec<Local>),
    /// A general mini-MPO on `opp`.
    Mpo(Mpo),
}

pub(super) struct CrossChannel {
    pub(super) win: WindowFactor,
    pub(super) opp: OppFactor,
}

/// A `PauliSum` Hamiltonian split across cyclic window `win_sites` and
/// complement `opp_sites`.
pub(super) struct WindowHamiltonian {
    pub(super) win_sites: Vec<usize>,
    pub(super) opp_sites: Vec<usize>,
    pub(super) shift: f64,
    /// Terms supported purely inside `W` (None if empty).
    pub(super) mpo_w: Option<Mpo>,
    /// Terms supported purely inside `opp` (None if empty).
    pub(super) mpo_opp: Option<Mpo>,
    /// Grouped crossing channels `A_κ^(W) ⊗ B_κ^(opp)`.
    pub(super) cross: Vec<CrossChannel>,
    /// Internal window bond positions `pos ∈ 1..win_sites.len()` strictly
    /// spanned by at least one term in `mpo_w` of span `>= 3` (the only bonds
    /// where 2-site TDVP can ever need global subspace expansion).
    pub(super) gse_bonds: Vec<usize>,
}

type LocalOps = Vec<(usize, Local)>;

/// Split `hamiltonian` onto `win_sites` (`W`, length `L`) and `opp_sites`
/// (`opp`, length `N - L`), validating charge conservation if `charges` is
/// given.
pub(super) fn split_hamiltonian_on_window(
    hamiltonian: &PauliSum,
    win_sites: Vec<usize>,
    opp_sites: Vec<usize>,
    n_qubits: usize,
    charges: Option<&Charges>,
) -> Result<WindowHamiltonian, Error> {
    // Expand to elementary local terms (ladder-split if `charges` is Some,
    // which also validates exact symmetry conservation!).
    let (shift, terms) = Mpo::elementary_terms(hamiltonian, n_qubits, charges)?;

    // Map global qubit index -> (is_in_win, local_index).
    let mut loc = vec![(false, 0usize); n_qubits];
    for (idx, &q) in win_sites.iter().enumerate() {
        loc[q] = (true, idx);
    }
    for (idx, &q) in opp_sites.iter().enumerate() {
        loc[q] = (false, idx);
    }

    let l_win = win_sites.len();
    let l_opp = opp_sites.len();
    let mut w_only: Vec<(C64, LocalOps)> = Vec::new();
    let mut opp_only: Vec<(C64, LocalOps)> = Vec::new();
    let mut raw_cross: Vec<(C64, LocalOps, LocalOps)> = Vec::new();

    for (coeff, ops) in terms {
        let mut w_ops: LocalOps = Vec::new();
        let mut o_ops: LocalOps = Vec::new();
        for (q, op) in ops {
            let (in_w, local_idx) = loc[q];
            if in_w {
                w_ops.push((local_idx, op));
            } else {
                o_ops.push((local_idx, op));
            }
        }
        w_ops.sort_by_key(|&(idx, _)| idx);
        o_ops.sort_by_key(|&(idx, _)| idx);
        match (w_ops.is_empty(), o_ops.is_empty()) {
            (false, true) => w_only.push((coeff, w_ops)),
            (true, false) => opp_only.push((coeff, o_ops)),
            (false, false) => raw_cross.push((coeff, w_ops, o_ops)),
            (true, true) => {}
        }
    }

    // Collect internal bonds strictly spanned by terms of span >= 3 inside W.
    let mut gse_bond_set = std::collections::BTreeSet::new();
    for (_, w_ops) in &w_only {
        if w_ops.len() >= 2 {
            let first = w_ops[0].0;
            let last = w_ops.last().unwrap().0;
            if last - first >= 2 {
                for pos in (first + 1)..=last {
                    gse_bond_set.insert(pos);
                }
            }
        }
    }
    let gse_bonds: Vec<usize> = gse_bond_set.into_iter().collect();

    // Build remapped Charges for `W` and `opp` if needed for MPO channels.
    let win_charges = charges.map(|ch| remap_charges(ch, &win_sites));
    let opp_charges = charges.map(|ch| remap_charges(ch, &opp_sites));

    let mpo_w = if w_only.is_empty() {
        None
    } else {
        Some(Mpo::from_local_terms(
            &w_only,
            l_win,
            0.0,
            win_charges.as_ref(),
        ))
    };
    let mpo_opp = if opp_only.is_empty() {
        None
    } else {
        Some(Mpo::from_local_terms(
            &opp_only,
            l_opp,
            0.0,
            opp_charges.as_ref(),
        ))
    };

    // Group crossing terms to minimise K:
    // Compare grouping by W-string vs grouping by opp-string and pick fewer groups.
    let mut by_w: BTreeMap<LocalOps, Vec<(C64, LocalOps)>> = BTreeMap::new();
    let mut by_o: BTreeMap<LocalOps, Vec<(C64, LocalOps)>> = BTreeMap::new();
    for (c, w_ops, o_ops) in raw_cross {
        by_w.entry(w_ops.clone())
            .or_default()
            .push((c, o_ops.clone()));
        by_o.entry(o_ops).or_default().push((c, w_ops));
    }

    let mut cross = Vec::new();
    if by_w.len() <= by_o.len() {
        for (w_ops, opp_list) in by_w {
            let mut w_str = vec![Local::I; l_win];
            for (idx, op) in w_ops {
                w_str[idx] = op;
            }
            let opp = if opp_list.len() == 1 {
                let (c, o_ops) = &opp_list[0];
                let mut o_str = vec![Local::I; l_opp];
                for &(idx, op) in o_ops {
                    o_str[idx] = op;
                }
                OppFactor::String(*c, o_str)
            } else {
                let mpo = Mpo::from_local_terms(&opp_list, l_opp, 0.0, opp_charges.as_ref());
                OppFactor::Mpo(mpo)
            };
            cross.push(CrossChannel {
                win: WindowFactor::String(w_str),
                opp,
            });
        }
    } else {
        for (o_ops, win_list) in by_o {
            let mut o_str = vec![Local::I; l_opp];
            for (idx, op) in o_ops {
                o_str[idx] = op;
            }
            let (win, scale) = if win_list.len() == 1 {
                let (c, w_ops) = &win_list[0];
                let mut w_str = vec![Local::I; l_win];
                for &(idx, op) in w_ops {
                    w_str[idx] = op;
                }
                (WindowFactor::String(w_str), *c)
            } else {
                let mpo = Mpo::from_local_terms(&win_list, l_win, 0.0, None);
                (WindowFactor::Mpo(mpo), ONE)
            };
            cross.push(CrossChannel {
                win,
                opp: OppFactor::String(scale, o_str),
            });
        }
    }

    Ok(WindowHamiltonian {
        win_sites,
        opp_sites,
        shift,
        mpo_w,
        mpo_opp,
        cross,
        gse_bonds,
    })
}

fn remap_charges(charges: &Charges, sites: &[usize]) -> Charges {
    let remapped: Vec<crate::mps::Charge> = charges
        .charges()
        .iter()
        .map(|c| {
            let vals: Vec<[i64; 2]> = sites.iter().map(|&q| c.values()[q]).collect();
            match c.modulus() {
                Some(m) => crate::mps::Charge::modulo(m, vals),
                None => crate::mps::Charge::u1(vals),
            }
        })
        .collect();
    Charges::new(&remapped, sites.len()).expect("valid remapped charges")
}

pub(super) struct WindowEnvs<'a> {
    pub(super) win_sites: &'a [usize],
    /// True for small rings (`n <= 12`) where `N_bond` is structurally
    /// rank-deficient and direct bond-metric projection up to dim 256 is cheap.
    pub(super) short_ring: bool,
    /// Relative truncation error `σ_{p} / σ_0` of the opposite-half norm
    /// transfer matrix (used to set the safe null-space cutoff for `Ñ`).
    pub(super) norm_trunc_err: f64,
    /// Pure-norm environments: `norm_left[pos][mu]`, `norm_right[pos][mu]`.
    pub(super) norm_left: Vec<Option<Vec<Mat<C64>>>>,
    pub(super) norm_right: Vec<Option<Vec<Mat<C64>>>>,
    /// `H_W` MPO environments (using the same `p_0` boundary modes as norm).
    mpo_w: Option<&'a Mpo>,
    hw_left: Vec<Option<Vec<Env>>>,
    hw_right: Vec<Option<Vec<Env>>>,
    /// `H_opp` scalar-on-`W` environments (`p_H` modes).
    hopp_left: Vec<Option<Vec<Mat<C64>>>>,
    hopp_right: Vec<Option<Vec<Mat<C64>>>>,
    /// Crossing channel environments.
    cross: Vec<CrossEnvKind<'a>>,
    #[cfg(feature = "gpu")]
    gpu: Option<(
        std::sync::Arc<crate::gpu::GpuContext>,
        std::sync::Arc<std::sync::Mutex<super::gpu::GpuPeriodicScratch>>,
    )>,
}

enum CrossEnvKind<'a> {
    /// Single string on `W`: `left[pos][mode]` and `right[pos][mode]`.
    String {
        ops: &'a [Local],
        left: Vec<Option<Vec<Mat<C64>>>>,
        right: Vec<Option<Vec<Mat<C64>>>>,
    },
    /// Mini-MPO on `W`: `left[pos][mode]` and `right[pos][mode]` as `Env`.
    Mpo {
        mpo: &'a Mpo,
        left: Vec<Option<Vec<Env>>>,
        right: Vec<Option<Vec<Env>>>,
    },
}

impl<'a> WindowEnvs<'a> {
    /// Compress the complement `opp` once and initialize boundary environments
    /// at `left[0]` and `right[L - 1]`.
    pub(super) fn new(
        tensors: &[Tensor],
        wh: &'a WindowHamiltonian,
        comp: TransferCompression,
        sym: Option<&PeriodicSymmetry>,
    ) -> Self {
        let l = wh.win_sites.len();
        let l_opp = wh.opp_sites.len();

        // 1. Compress identity transfer matrix across `opp` (for norm and H_W).
        let id_ops = vec![Local::I; l_opp];
        let seam_site = wh.win_sites[0];
        let norm_ct =
            compress_string_transfer(tensors, &wh.opp_sites, seam_site, &id_ops, ONE, comp, sym);
        let norm_trunc_err = norm_ct.trunc_err;

        let mut norm_left = vec![None; l];
        let mut norm_right = vec![None; l];
        norm_left[0] = Some(norm_ct.left.clone());
        norm_right[l - 1] = Some(norm_ct.right.clone());

        // 2. H_W MPO boundary environments.
        let mut hw_left = vec![None; l];
        let mut hw_right = vec![None; l];
        if wh.mpo_w.is_some() {
            hw_left[0] = Some(
                norm_ct
                    .left
                    .iter()
                    .map(|m| Env {
                        blocks: vec![m.clone()],
                    })
                    .collect(),
            );
            hw_right[l - 1] = Some(
                norm_ct
                    .right
                    .iter()
                    .map(|m| Env {
                        blocks: vec![m.clone()],
                    })
                    .collect(),
            );
        }

        // 3. Compress H_opp across `opp`.
        let mut hopp_left = vec![None; l];
        let mut hopp_right = vec![None; l];
        if let Some(mpo_opp) = &wh.mpo_opp {
            let ct = compress_mpo_transfer(tensors, &wh.opp_sites, mpo_opp, comp, sym);
            hopp_left[0] = Some(ct.left);
            hopp_right[l - 1] = Some(ct.right);
        } else {
            hopp_left[0] = Some(Vec::new());
            hopp_right[l - 1] = Some(Vec::new());
        }

        // 4. Compress crossing channels in parallel!
        let cross_cts: Vec<CompressedTransfer> = wh
            .cross
            .par_iter()
            .map(|ch| match &ch.opp {
                OppFactor::String(sc, ops) => {
                    compress_string_transfer(tensors, &wh.opp_sites, seam_site, ops, *sc, comp, sym)
                }
                OppFactor::Mpo(mpo) => {
                    compress_mpo_transfer(tensors, &wh.opp_sites, mpo, comp, sym)
                }
            })
            .collect();

        let mut cross = Vec::with_capacity(wh.cross.len());
        for (ch, ct) in wh.cross.iter().zip(cross_cts) {
            if ct.rank() == 0 {
                continue;
            }
            match &ch.win {
                WindowFactor::String(ops) => {
                    let mut left = vec![None; l];
                    let mut right = vec![None; l];
                    left[0] = Some(ct.left);
                    right[l - 1] = Some(ct.right);
                    cross.push(CrossEnvKind::String { ops, left, right });
                }
                WindowFactor::Mpo(mpo) => {
                    let mut left = vec![None; l];
                    let mut right = vec![None; l];
                    left[0] = Some(
                        ct.left
                            .into_iter()
                            .map(|m| Env { blocks: vec![m] })
                            .collect(),
                    );
                    right[l - 1] = Some(
                        ct.right
                            .into_iter()
                            .map(|m| Env { blocks: vec![m] })
                            .collect(),
                    );
                    cross.push(CrossEnvKind::Mpo { mpo, left, right });
                }
            }
        }

        WindowEnvs {
            win_sites: &wh.win_sites,
            short_ring: tensors.len() <= 12,
            norm_trunc_err,
            norm_left,
            norm_right,
            mpo_w: wh.mpo_w.as_ref(),
            hw_left,
            hw_right,
            hopp_left,
            hopp_right,
            cross,
            #[cfg(feature = "gpu")]
            gpu: None,
        }
    }

    /// Attach a Vulkan GPU context and shared sweep scratch buffers so local
    /// effective operators (`H̃`, `Ñ`) execute on the GPU.
    #[cfg(feature = "gpu")]
    pub(super) fn with_gpu(
        mut self,
        ctx: &std::sync::Arc<crate::gpu::GpuContext>,
        scratch: &std::sync::Arc<std::sync::Mutex<super::gpu::GpuPeriodicScratch>>,
    ) -> Self {
        self.gpu = Some((ctx.clone(), scratch.clone()));
        self
    }

    /// Safe null-space cutoff accounting for PWE compression error of `Ñ`,
    /// capped at `0.5` so a random sweep-0 state under an explicit small
    /// `transfer_rank` always retains its leading physical modes while
    /// filtering out PWE-corrupted near-zero ghost modes.
    pub(super) fn safe_null_tol(&self, base_tol: f64) -> f64 {
        (20.0 * self.norm_trunc_err).min(0.5).max(base_tol)
    }

    pub(super) fn len(&self) -> usize {
        self.win_sites.len()
    }

    /// Update `left[pos + 1]` from `left[pos]` and `tensors[win_sites[pos]]`.
    pub(super) fn update_left(&mut self, pos: usize, tensors: &[Tensor]) {
        let a = &tensors[self.win_sites[pos]];
        let prev = self.norm_left[pos].as_ref().expect("norm_left built");
        self.norm_left[pos + 1] = Some(batch_mats_right(prev, a, Local::I));
        if let Some(mpo) = &self.mpo_w {
            let prev = self.hw_left[pos].as_ref().expect("hw_left built");
            let w_site = &mpo.sites[pos];
            self.hw_left[pos + 1] =
                Some(prev.par_iter().map(|e| update_left(e, a, w_site)).collect());
        }
        let prev_h = self.hopp_left[pos].as_ref().expect("hopp_left built");
        self.hopp_left[pos + 1] = Some(batch_mats_right(prev_h, a, Local::I));
        for ch in &mut self.cross {
            match ch {
                CrossEnvKind::String { ops, left, .. } => {
                    let op = ops[pos];
                    let prev = left[pos].as_ref().expect("cross left built");
                    left[pos + 1] = Some(batch_mats_right(prev, a, op));
                }
                CrossEnvKind::Mpo { mpo, left, .. } => {
                    let w_site = &mpo.sites[pos];
                    let prev = left[pos].as_ref().expect("cross mpo left built");
                    left[pos + 1] =
                        Some(prev.par_iter().map(|e| update_left(e, a, w_site)).collect());
                }
            }
        }
    }

    /// Update `right[pos]` from `right[pos + 1]` and `tensors[win_sites[pos + 1]]`.
    pub(super) fn update_right(&mut self, pos: usize, tensors: &[Tensor]) {
        let a = &tensors[self.win_sites[pos + 1]];
        let prev = self.norm_right[pos + 1].as_ref().expect("norm_right built");
        self.norm_right[pos] = Some(batch_mats_left(prev, a, Local::I));
        if let Some(mpo) = &self.mpo_w {
            let prev = self.hw_right[pos + 1].as_ref().expect("hw_right built");
            let w_site = &mpo.sites[pos + 1];
            self.hw_right[pos] = Some(
                prev.par_iter()
                    .map(|e| update_right(e, a, w_site))
                    .collect(),
            );
        }
        let prev_h = self.hopp_right[pos + 1].as_ref().expect("hopp_right built");
        self.hopp_right[pos] = Some(batch_mats_left(prev_h, a, Local::I));
        for ch in &mut self.cross {
            match ch {
                CrossEnvKind::String { ops, right, .. } => {
                    let op = ops[pos + 1];
                    let prev = right[pos + 1].as_ref().expect("cross right built");
                    right[pos] = Some(batch_mats_left(prev, a, op));
                }
                CrossEnvKind::Mpo { mpo, right, .. } => {
                    let w_site = &mpo.sites[pos + 1];
                    let prev = right[pos + 1].as_ref().expect("cross mpo right built");
                    right[pos] = Some(
                        prev.par_iter()
                            .map(|e| update_right(e, a, w_site))
                            .collect(),
                    );
                }
            }
        }
    }

    /// Build all right environments `right[L - 2] ..= right[0]`.
    pub(super) fn build_all_right(&mut self, tensors: &[Tensor]) {
        let l = self.len();
        for pos in (0..l - 1).rev() {
            self.update_right(pos, tensors);
        }
    }

    /// Build all left environments `left[1] ..= left[L - 1]`.
    pub(super) fn build_all_left(&mut self, tensors: &[Tensor]) {
        let l = self.len();
        for pos in 0..l - 1 {
            self.update_left(pos, tensors);
        }
    }

    /// Partial-trace positive semidefinite metrics `(L̄, R̄)` for a local block
    /// spanning `left_pos ..= right_pos` inside `W`.
    pub(super) fn metric_pair(&self, left_pos: usize, right_pos: usize) -> (Mat<C64>, Mat<C64>) {
        let ls = self.norm_left[left_pos].as_ref().expect("norm_left built");
        let rs = self.norm_right[right_pos]
            .as_ref()
            .expect("norm_right built");
        let d_l = ls.first().map_or(1, |m| m.nrows());
        let d_r = rs.first().map_or(1, |m| m.nrows());
        let mut l_bar = Mat::<C64>::zeros(d_l, d_l);
        let mut r_bar = Mat::<C64>::zeros(d_r, d_r);
        for (lm, rm) in ls.iter().zip(rs) {
            let tr_r: C64 = (0..d_r).map(|i| rm[(i, i)]).sum();
            let tr_l: C64 = (0..d_l).map(|i| lm[(i, i)]).sum();
            if tr_r != ZERO {
                l_bar += faer::Scale(tr_r) * lm;
            }
            if tr_l != ZERO {
                r_bar += faer::Scale(tr_l) * rm;
            }
        }
        // Check sign: if the overall trace is negative (overall sign flip of
        // singular vector pair), negate both.
        let tr_lbar: f64 = (0..d_l).map(|i| l_bar[(i, i)].re).sum();
        if tr_lbar < 0.0 {
            l_bar *= faer::Scale(C64::new(-1.0, 0.0));
        }
        let tr_rbar: f64 = (0..d_r).map(|i| r_bar[(i, i)].re).sum();
        if tr_rbar < 0.0 {
            r_bar *= faer::Scale(C64::new(-1.0, 0.0));
        }
        (l_bar, r_bar)
    }

    /// Build the gauge-conditioned local effective operators `(H̃, Ñ)` for two
    /// adjacent sites `(pos, pos + 1)` inside `W`.
    pub(super) fn two_site_problem(
        &self,
        pos: usize,
        tensors: &[Tensor],
        sym: Option<&PeriodicSymmetry>,
        gauge_tol: f64,
    ) -> LocalProblem {
        let q1 = self.win_sites[pos];
        let q2 = self.win_sites[pos + 1];
        let (l_full, r_full) = (tensors[q1].left, tensors[q2].right);
        self.build_local_problem(pos, pos + 1, &[q1, q2], l_full, r_full, sym, gauge_tol)
    }

    /// Build the gauge-conditioned local effective operators `(H̃, Ñ)` for a
    /// single site `pos` inside `W`.
    pub(super) fn one_site_problem(
        &self,
        pos: usize,
        tensors: &[Tensor],
        sym: Option<&PeriodicSymmetry>,
        gauge_tol: f64,
    ) -> LocalProblem {
        let q = self.win_sites[pos];
        let (l_full, r_full) = (tensors[q].left, tensors[q].right);
        self.build_local_problem(pos, pos, &[q], l_full, r_full, sym, gauge_tol)
    }

    #[allow(clippy::too_many_arguments)]
    fn build_local_problem(
        &self,
        left_pos: usize,
        right_pos: usize,
        sites: &[usize],
        l_full: usize,
        r_full: usize,
        sym: Option<&PeriodicSymmetry>,
        gauge_tol: f64,
    ) -> LocalProblem {
        let (l_bar, r_bar) = self.metric_pair(left_pos, right_pos);
        let (mut g_l, mut g_l_inv, mut g_r, mut g_r_inv, layout, left_eff_sec, right_eff_sec) =
            match sym {
                Some(sym) => {
                    let left_sec = sym.left_bond(sites[0]);
                    let right_sec = sym.right_bond(*sites.last().unwrap());
                    let (gl, l_eff_sec, gl_inv) = block_metric_gauge(&l_bar, left_sec, gauge_tol);
                    let (gr, r_eff_sec, gr_inv) = block_metric_gauge(&r_bar, right_sec, gauge_tol);
                    let layout = if sites.len() == 2 {
                        PeriodicLayout::two_site(&l_eff_sec, &r_eff_sec, sym, sites[0], sites[1])
                    } else {
                        PeriodicLayout::one_site(&l_eff_sec, &r_eff_sec, sym, sites[0])
                    };
                    (
                        gl,
                        gl_inv,
                        gr,
                        gr_inv,
                        Some(layout),
                        Some(l_eff_sec),
                        Some(r_eff_sec),
                    )
                }
                None => {
                    let (gl, gl_inv) = dense_metric_gauge(&l_bar, gauge_tol);
                    let (gr, gr_inv) = dense_metric_gauge(&r_bar, gauge_tol);
                    (gl, gl_inv, gr, gr_inv, None, None, None)
                }
            };

        let l_eff = g_l.ncols();
        let r_eff = g_r.ncols();
        let phys = 1usize << sites.len();
        debug_assert_eq!(g_l.nrows(), l_full);
        debug_assert_eq!(g_r.nrows(), r_full);

        // 1. Transformed norm terms. When `Ñ` is proportional to identity
        // (`p == 1` or negligible subleading tail `< 1e-11`), rescale `(G_L, G_R)`
        // by `prod^{-1/4}` so `Ñ == I` and the identity fast path triggers
        // without altering `p > 1` gauge conditioning.
        let nl_raw = self.norm_left[left_pos].as_ref().unwrap();
        let nr_raw = self.norm_right[right_pos].as_ref().unwrap();
        let mut norm_terms: Vec<(Mat<C64>, Mat<C64>)> = nl_raw
            .iter()
            .zip(nr_raw)
            .map(|(lm, rm)| {
                (
                    matmul(g_l.adjoint(), matmul(lm.as_ref(), g_l.as_ref()).as_ref()),
                    matmul(g_r.adjoint(), matmul(rm.as_ref(), g_r.as_ref()).as_ref()),
                )
            })
            .collect();

        let n_is_identity = if l_eff > 0 && r_eff > 0 && !norm_terms.is_empty() {
            let (lm0, rm0) = &norm_terms[0];
            let scale_l = lm0[(0, 0)];
            let scale_r = rm0[(0, 0)];
            let prod = scale_l * scale_r;
            let lead_is_id = prod.re > 0.0
                && prod.im.abs() <= 1e-11 * prod.re
                && is_scaled_identity(lm0, scale_l, 1e-11)
                && is_scaled_identity(rm0, scale_r, 1e-11);
            let tail_negligible = if !lead_is_id {
                false
            } else if norm_terms.len() == 1 {
                true
            } else {
                let tail_norm: f64 = norm_terms[1..]
                    .iter()
                    .map(|(lm, rm)| lm.as_ref().norm_l2() * rm.as_ref().norm_l2())
                    .sum();
                tail_norm < 1e-11 * prod.re
            };
            if !self.short_ring && tail_negligible && (prod.re - 1.0).abs() > 1e-12 {
                let s = prod.re.powf(-0.25);
                let inv_s = s.recip();
                let s2 = s * s;
                g_l *= faer::Scale(C64::new(s, 0.0));
                g_r *= faer::Scale(C64::new(s, 0.0));
                g_l_inv *= faer::Scale(C64::new(inv_s, 0.0));
                g_r_inv *= faer::Scale(C64::new(inv_s, 0.0));
                for (lm, rm) in &mut norm_terms {
                    *lm *= faer::Scale(C64::new(s2, 0.0));
                    *rm *= faer::Scale(C64::new(s2, 0.0));
                }
                true
            } else {
                norm_terms.len() == 1 && lead_is_id && (prod - ONE).norm() < 1e-11
            }
        } else {
            false
        };

        // Helper to transform a left matrix M -> G_L† M G_L and right matrix M -> G_R† M G_R.
        let xform_l = |m: &Mat<C64>| -> Mat<C64> {
            if !mat_is_nonzero(m) {
                return Mat::zeros(l_eff, l_eff);
            }
            matmul(g_l.adjoint(), matmul(m.as_ref(), g_l.as_ref()).as_ref())
        };
        let xform_r = |m: &Mat<C64>| -> Mat<C64> {
            if !mat_is_nonzero(m) {
                return Mat::zeros(r_eff, r_eff);
            }
            matmul(g_r.adjoint(), matmul(m.as_ref(), g_r.as_ref()).as_ref())
        };
        let xform_env_l = |e: &Env| -> Env {
            Env {
                blocks: e.blocks.iter().map(&xform_l).collect(),
            }
        };
        let xform_env_r = |e: &Env| -> Env {
            Env {
                blocks: e.blocks.iter().map(&xform_r).collect(),
            }
        };

        // Exact diagonal of Ñ for Jacobi-preconditioned CG (O(p * l_eff * r_eff)).
        let n_diag: Vec<f64> = if n_is_identity || l_eff == 0 || r_eff == 0 {
            Vec::new()
        } else {
            let mut diag_lr = vec![0.0f64; l_eff * r_eff];
            for (lm, rm) in &norm_terms {
                for i in 0..l_eff {
                    let lii = lm[(i, i)];
                    let row = &mut diag_lr[i * r_eff..(i + 1) * r_eff];
                    for j in 0..r_eff {
                        row[j] += (lii * rm[(j, j)]).re;
                    }
                }
            }
            match &layout {
                None => {
                    let mut out = vec![0.0f64; l_eff * phys * r_eff];
                    for i in 0..l_eff {
                        let src = &diag_lr[i * r_eff..(i + 1) * r_eff];
                        for p in 0..phys {
                            out[(i * phys + p) * r_eff..][..r_eff].copy_from_slice(src);
                        }
                    }
                    out
                }
                Some(lay) => {
                    let mut out = vec![0.0f64; lay.len];
                    for &(s, _ds, offset, width) in &lay.rows {
                        let rs = lay.left.range(s);
                        for p in 0..lay.phys {
                            let Some((_, col0, t)) = lay.cols[s * lay.phys + p] else {
                                continue;
                            };
                            let rt = lay.right.range(t);
                            for (a, li) in rs.clone().enumerate() {
                                let dst = &mut out[offset + a * width + col0..][..rt.len()];
                                dst.copy_from_slice(&diag_lr[li * r_eff + rt.start..][..rt.len()]);
                            }
                        }
                    }
                    out
                }
            }
        };

        // 2. Transformed H_W batched MPO operator (all p_0 modes stacked into ONE operator!).
        let mut mpo_ops: Vec<BatchedMpoOp> = Vec::new();
        if let Some(mpo) = &self.mpo_w {
            let hl = self.hw_left[left_pos].as_ref().unwrap();
            let hr = self.hw_right[right_pos].as_ref().unwrap();
            let tl: Vec<Env> = hl.iter().map(&xform_env_l).collect();
            let tr: Vec<Env> = hr.iter().map(&xform_env_r).collect();
            let rows = if sites.len() == 2 {
                pair_keyed(&mpo.sites[left_pos], &mpo.sites[right_pos])
            } else {
                right_keyed(&mpo.sites[left_pos])
            };
            if !tl.is_empty() {
                mpo_ops.push(BatchedMpoOp::new(&tl, &tr, rows, l_eff, r_eff, phys));
            }
        }

        // 3. Transformed H_opp scalar terms (act as identity on physical indices).
        let mut scalar_terms: Vec<(Mat<C64>, Mat<C64>)> = Vec::new();
        let hol = self.hopp_left[left_pos].as_ref().unwrap();
        let hor = self.hopp_right[right_pos].as_ref().unwrap();
        for (lm, rm) in hol.iter().zip(hor) {
            scalar_terms.push((xform_l(lm), xform_r(rm)));
        }

        // 4. Transformed crossing terms.
        let mut string_terms: Vec<(Mat<C64>, Mat<C64>, Vec<Local>)> = Vec::new();
        for ch in &self.cross {
            match ch {
                CrossEnvKind::String { ops, left, right } => {
                    let local_ops: Vec<Local> = if sites.len() == 2 {
                        vec![ops[left_pos], ops[right_pos]]
                    } else {
                        vec![ops[left_pos]]
                    };
                    let all_id = local_ops.iter().all(|&o| o == Local::I);
                    let ls = left[left_pos].as_ref().unwrap();
                    let rs = right[right_pos].as_ref().unwrap();
                    for (lm, rm) in ls.iter().zip(rs) {
                        let xl = xform_l(lm);
                        let xr = xform_r(rm);
                        if all_id {
                            scalar_terms.push((xl, xr));
                        } else {
                            string_terms.push((xl, xr, local_ops.clone()));
                        }
                    }
                }
                CrossEnvKind::Mpo { mpo, left, right } => {
                    let ls = left[left_pos].as_ref().unwrap();
                    let rs = right[right_pos].as_ref().unwrap();
                    let tl: Vec<Env> = ls.iter().map(&xform_env_l).collect();
                    let tr: Vec<Env> = rs.iter().map(&xform_env_r).collect();
                    let rows = if sites.len() == 2 {
                        pair_keyed(&mpo.sites[left_pos], &mpo.sites[right_pos])
                    } else {
                        right_keyed(&mpo.sites[left_pos])
                    };
                    if !tl.is_empty() {
                        mpo_ops.push(BatchedMpoOp::new(&tl, &tr, rows, l_eff, r_eff, phys));
                    }
                }
            }
        }

        // Compress redundant terms in `norm_terms` / `scalar_terms` only when
        // genuinely over-complete and inexpensive (`k <= 256` and matrix size small).
        let max_lr = (l_eff * l_eff).min(r_eff * r_eff);
        let norm_terms = if norm_terms.len() > max_lr && norm_terms.len() <= 256 && max_lr <= 256 {
            compress_lr_pairs(l_eff, r_eff, norm_terms, 1e-13)
        } else {
            norm_terms
        };
        let scalar_terms = if scalar_terms.len() > 8 && scalar_terms.len() <= 256 && max_lr <= 256 {
            compress_lr_pairs(l_eff, r_eff, scalar_terms, 1e-13)
        } else {
            scalar_terms
        };

        // Group identical non-trivial string_terms by their local ops, compress,
        // and pack each group into ONE batched two-GEMM `StackedStringOp`!
        type PairList = Vec<(Mat<C64>, Mat<C64>)>;
        let mut grouped_strings: BTreeMap<Vec<Local>, PairList> = BTreeMap::new();
        for (lm, rm, ops) in string_terms {
            grouped_strings.entry(ops).or_default().push((lm, rm));
        }
        let mut string_ops = Vec::with_capacity(grouped_strings.len());
        for (ops, pairs) in grouped_strings {
            let compressed = if pairs.len() > 8 && pairs.len() <= 256 && max_lr <= 256 {
                compress_lr_pairs(l_eff, r_eff, pairs, 1e-13)
            } else {
                pairs
            };
            if !compressed.is_empty() {
                string_ops.push(StackedStringOp::new(l_eff, r_eff, phys, ops, &compressed));
            }
        }

        let scalar_op = StackedPairOp::new(l_eff, r_eff, phys, &scalar_terms);
        let norm_stacked = StackedPairOp::new(l_eff, r_eff, phys, &norm_terms);

        #[cfg(feature = "gpu")]
        let (gpu_norm, gpu_ham) = match &self.gpu {
            Some((ctx, scratch)) => {
                let norm_spec = norm_stacked.gpu_spec();
                let gn = super::gpu::GpuLocalOp::new(
                    ctx,
                    scratch,
                    l_eff,
                    r_eff,
                    phys,
                    std::slice::from_ref(&norm_spec),
                )
                .ok()
                .flatten();

                let mut ham_specs: Vec<super::gpu::GpuTermSpec<'_>> = Vec::new();
                for op in &mpo_ops {
                    for ch in &op.chunks {
                        ham_specs.push(op.chunk_gpu_spec(ch));
                    }
                }
                if scalar_op.k > 0 {
                    ham_specs.push(scalar_op.gpu_spec());
                }
                for op in &string_ops {
                    if op.k > 0 {
                        ham_specs.push(op.gpu_spec());
                    }
                }
                let gh = super::gpu::GpuLocalOp::new(ctx, scratch, l_eff, r_eff, phys, &ham_specs)
                    .ok()
                    .flatten();
                (gn, gh)
            }
            None => (None, None),
        };

        LocalProblem {
            l_full,
            r_full,
            l_eff,
            r_eff,
            phys,
            short_ring: self.short_ring,
            g_l,
            g_l_inv,
            g_r,
            g_r_inv,
            layout,
            left_eff_sec,
            right_eff_sec,
            n_is_identity,
            n_diag,
            norm_op: LocalNormOp {
                stacked: norm_stacked,
                #[cfg(feature = "gpu")]
                gpu: gpu_norm,
            },
            norm_terms,
            ham_op: LocalHamOp {
                l_eff,
                r_eff,
                mpo_ops,
                scalar_op,
                string_ops,
                #[cfg(feature = "gpu")]
                gpu: gpu_ham,
            },
        }
    }

    /// White's perturbation blocks for DMRG noise on bond `(pos, pos + 1)`
    /// inside `W`, built from the leading norm/H_W mode and crossing channels
    /// that continue across the split.
    pub(super) fn perturbation_blocks(
        &self,
        pos: usize,
        theta_full: &[C64],
        l: usize,
        r: usize,
        absorb_right: bool,
    ) -> Vec<Mat<C64>> {
        let mut out = Vec::new();
        if let Some(mpo) = &self.mpo_w {
            if absorb_right {
                let envs = self.hw_left[pos].as_ref().unwrap();
                let w = &mpo.sites[pos];
                for env in envs.iter().take(4) {
                    out.extend(left_perturbation_local(env, w, theta_full, l, r));
                }
            } else {
                let envs = self.hw_right[pos + 1].as_ref().unwrap();
                let w = &mpo.sites[pos + 1];
                for env in envs.iter().take(4) {
                    out.extend(right_perturbation_local(env, w, theta_full, l, r));
                }
            }
        }
        // Also include crossing channels that act on `pos` (moving right) or `pos + 1` (moving left).
        for ch in &self.cross {
            match ch {
                CrossEnvKind::String { ops, left, right } => {
                    if absorb_right {
                        let op = ops[pos];
                        if op != Local::I {
                            for env in left[pos].as_ref().unwrap().iter().take(2) {
                                out.push(single_left_perturbation(env, op, theta_full, l, r));
                            }
                        }
                    } else {
                        let op = ops[pos + 1];
                        if op != Local::I {
                            for env in right[pos + 1].as_ref().unwrap().iter().take(2) {
                                out.push(single_right_perturbation(env, op, theta_full, l, r));
                            }
                        }
                    }
                }
                CrossEnvKind::Mpo { mpo, left, right } => {
                    if absorb_right {
                        let w = &mpo.sites[pos];
                        for env in left[pos].as_ref().unwrap().iter().take(2) {
                            out.extend(left_perturbation_local(env, w, theta_full, l, r));
                        }
                    } else {
                        let w = &mpo.sites[pos + 1];
                        for env in right[pos + 1].as_ref().unwrap().iter().take(2) {
                            out.extend(right_perturbation_local(env, w, theta_full, l, r));
                        }
                    }
                }
            }
        }
        out
    }
}

/// Gauge-conditioned local problem `(H̃, Ñ)` on 1 or 2 sites.
pub(super) struct LocalProblem {
    pub(super) l_full: usize,
    pub(super) r_full: usize,
    pub(super) l_eff: usize,
    pub(super) r_eff: usize,
    pub(super) phys: usize,
    pub(super) short_ring: bool,
    /// `G_L` (`l_full × l_eff`), maps gauge coords -> bond coords.
    pub(super) g_l: Mat<C64>,
    /// `G_L^{-1}` (`l_eff × l_full`), maps bond coords -> gauge coords.
    pub(super) g_l_inv: Mat<C64>,
    /// `G_R` (`r_full × r_eff`), maps gauge coords -> bond coords.
    pub(super) g_r: Mat<C64>,
    /// `G_R^{-1}` (`r_eff × r_full`), maps bond coords -> gauge coords.
    pub(super) g_r_inv: Mat<C64>,
    pub(super) layout: Option<PeriodicLayout>,
    pub(super) left_eff_sec: Option<crate::mps::charges::Sectors>,
    pub(super) right_eff_sec: Option<crate::mps::charges::Sectors>,
    pub(super) n_is_identity: bool,
    pub(super) n_diag: Vec<f64>,
    pub(super) norm_op: LocalNormOp,
    pub(super) norm_terms: Vec<(Mat<C64>, Mat<C64>)>,
    pub(super) ham_op: LocalHamOp,
}

impl LocalProblem {
    /// Convert a full bond-coordinate tensor `(l_full, phys, r_full)` into
    /// gauge coordinates `(l_eff, phys, r_eff)`: `Θ̃ = G_L^{-1} Θ (G_R^{-1})^T`,
    /// then pack to compact vector if symmetric.
    pub(super) fn to_solver_vec(&self, theta_full: &[C64]) -> Vec<C64> {
        let mut mid = vec![ZERO; self.l_eff * self.phys * self.r_full];
        // Left multiply by g_l_inv: (l_eff × l_full) * (l_full × (phys * r_full)).
        let t_mat = MatRef::from_row_major_slice(theta_full, self.l_full, self.phys * self.r_full);
        let prod_l = matmul(self.g_l_inv.as_ref(), t_mat);
        for i in 0..self.l_eff {
            for j in 0..self.phys * self.r_full {
                mid[i * self.phys * self.r_full + j] = prod_l[(i, j)];
            }
        }
        // Right multiply by g_r_inv^T: ((l_eff * phys) × r_full) * (r_full × r_eff).
        let mid_mat = MatRef::from_row_major_slice(&mid, self.l_eff * self.phys, self.r_full);
        let prod_r = matmul(mid_mat, self.g_r_inv.transpose());
        let mut dense = Vec::with_capacity(self.l_eff * self.phys * self.r_eff);
        for i in 0..self.l_eff * self.phys {
            for j in 0..self.r_eff {
                dense.push(prod_r[(i, j)]);
            }
        }
        match &self.layout {
            Some(layout) => layout.gather(&dense),
            None => dense,
        }
    }

    /// Unpack a solver vector back to dense gauge coordinates `(l_eff, phys, r_eff)`.
    pub(super) fn unpack_gauge(&self, x: &[C64]) -> Vec<C64> {
        match &self.layout {
            Some(layout) => layout.scatter(x),
            None => x.to_vec(),
        }
    }

    /// Map a dense gauge-coordinate tensor `(l_eff, phys, r_eff)` back to full
    /// bond coordinates `(l_full, phys, r_full)`: `Θ = G_L Θ̃ G_R^T`.
    pub(super) fn gauge_to_full(&self, theta_tilde: &[C64]) -> Vec<C64> {
        let t_mat = MatRef::from_row_major_slice(theta_tilde, self.l_eff, self.phys * self.r_eff);
        let prod_l = matmul(self.g_l.as_ref(), t_mat);
        let mut mid = Vec::with_capacity(self.l_full * self.phys * self.r_eff);
        for i in 0..self.l_full {
            for j in 0..self.phys * self.r_eff {
                mid.push(prod_l[(i, j)]);
            }
        }
        let mid_mat = MatRef::from_row_major_slice(&mid, self.l_full * self.phys, self.r_eff);
        let prod_r = matmul(mid_mat, self.g_r.transpose());
        let mut full = Vec::with_capacity(self.l_full * self.phys * self.r_full);
        for i in 0..self.l_full * self.phys {
            for j in 0..self.r_full {
                full.push(prod_r[(i, j)]);
            }
        }
        full
    }

    /// Wrap `ham_op` so it acts on compact symmetric vectors when `layout` is Some.
    pub(super) fn ham_linear_op(&self) -> PackedOp<'_, LocalHamOp> {
        PackedOp {
            inner: &self.ham_op,
            layout: self.layout.as_ref(),
        }
    }

    /// Wrap `norm_op` so it acts on compact symmetric vectors when `layout` is Some.
    pub(super) fn norm_linear_op(&self) -> PackedOp<'_, LocalNormOp> {
        PackedOp {
            inner: &self.norm_op,
            layout: self.layout.as_ref(),
        }
    }

    /// Solve the local generalized eigenproblem `H̃ x = E Ñ x` for the lowest
    /// eigenpair in the positive range of `Ñ`.
    ///
    /// When the per-physical-state bond metric dimension is `≤ 400` (all rings
    /// with `χ ≤ 20` dense, or much larger `χ` with symmetries), diagonalises
    /// the bond metric `N_bond` directly once, strictly projects out `ker(Ñ)`
    /// (zero ghost eigenvalues, zero Krylov stagnation!), and runs standard
    /// Lanczos (or exact direct EVD when projected dim `≤ 96`) in `Ñ = I`
    /// coordinates! Above that threshold, uses matrix-free Generalized Davidson.
    pub(super) fn solve_lowest(
        &self,
        v0: &[C64],
        options: &crate::statevector::eigen::EigenOptions,
        null_tol: f64,
        cap_norm: bool,
    ) -> Result<crate::statevector::eigen::Ritz<C64>, Error> {
        if self.n_is_identity {
            let dim = v0.len();
            if self.short_ring || dim <= 16 {
                return super::solver::solve_generalized_lowest(
                    &self.ham_linear_op(),
                    &self.norm_linear_op(),
                    v0,
                    true,
                    None,
                    options,
                    null_tol,
                );
            }
            let nrm = crate::statevector::krylov::norm(v0);
            let u0: Vec<C64> = if nrm > 0.0 {
                v0.iter().map(|&a| a / nrm).collect()
            } else {
                let mut u = vec![ZERO; dim];
                u[0] = ONE;
                u
            };
            let (value, vector, converged) = lanczos_full_reorth(
                &self.ham_linear_op(),
                &u0,
                options.tolerance,
                dim.min(options.max_iterations).min(64),
            );
            return Ok(crate::statevector::eigen::Ritz {
                value,
                vector,
                converged,
            });
        }
        if let Some(proj) = BondMetricProjector::try_build(self, null_tol.max(1e-10)) {
            let v0_norm = crate::statevector::krylov::norm(v0).max(1.0);
            let mut ritz = proj.solve_lowest(self, v0, options)?;
            // On 2-site updates (which undergo Frobenius SVD truncation), back
            // off if inverting a near-null mode of `N_bond` inflated `‖θ̃‖_F`.
            // On 1-site updates there is zero truncation, so keep full accuracy!
            if cap_norm {
                let mut reg = null_tol.max(1e-10);
                for _ in 0..3 {
                    let sol_norm = crate::statevector::krylov::norm(&ritz.vector);
                    if sol_norm <= 50.0 * v0_norm || reg >= 1e-4 {
                        break;
                    }
                    reg = (reg * 100.0).min(1e-4);
                    let Some(proj_reg) = BondMetricProjector::try_build(self, reg) else {
                        break;
                    };
                    ritz = proj_reg.solve_lowest(self, v0, options)?;
                }
            }
            return Ok(ritz);
        }
        super::solver::solve_generalized_lowest(
            &self.ham_linear_op(),
            &self.norm_linear_op(),
            v0,
            false,
            Some(&self.n_diag),
            options,
            null_tol,
        )
    }

    /// Evolve `i Ñ dx/dt = H̃ x` in place for time `dt`. Uses exact bond-metric
    /// projection + standard `krylov_evolve` whenever `bond_dim ≤ MAX_BOND_METRIC_DIM`,
    /// falling back to Jacobi-preconditioned matrix-free `N`-inner-product Lanczos
    /// in `O(p D χ³)` for larger bonds.
    ///
    /// Returns the built [`BondMetricProjector`] (if any) so two-site TDVP can
    /// reuse it for exact `N`-norm pair polishing without recomputing `N_bond` EVD.
    pub(super) fn evolve_local(
        &self,
        x: &mut [C64],
        dt: f64,
        null_tol: f64,
    ) -> Result<Option<BondMetricProjector>, Error> {
        if dt == 0.0 || x.is_empty() {
            return Ok(None);
        }
        if self.n_is_identity {
            super::solver::evolve_generalized(
                &self.ham_linear_op(),
                &self.norm_linear_op(),
                x,
                dt,
                true,
                None,
                null_tol,
            )?;
            return Ok(None);
        }
        if let Some(proj) = BondMetricProjector::try_build(self, null_tol) {
            let x_norm = crate::statevector::krylov::norm(x).max(1.0);
            let mut trial = x.to_vec();
            proj.evolve(self, &mut trial, dt)?;
            let mut reg = null_tol.max(1e-12);
            for _ in 0..3 {
                let t_norm = crate::statevector::krylov::norm(&trial);
                if t_norm <= 1e4 * x_norm || reg >= 1e-4 {
                    break;
                }
                reg = (reg * 100.0).min(1e-4);
                let Some(proj_reg) = BondMetricProjector::try_build(self, reg) else {
                    break;
                };
                trial.copy_from_slice(x);
                proj_reg.evolve(self, &mut trial, dt)?;
            }
            x.copy_from_slice(&trial);
            return Ok(Some(proj));
        }
        super::solver::evolve_generalized(
            &self.ham_linear_op(),
            &self.norm_linear_op(),
            x,
            dt,
            false,
            Some(&self.n_diag),
            null_tol,
        )?;
        Ok(None)
    }

    /// Polish a truncated two-site split `(a_full, b_full)` in the exact
    /// periodic `N`-norm (`min ‖Y (Ã B̃) - Y Θ̃‖_2^2`) via up to 2 sweeps of
    /// alternating minimum-norm least squares.
    ///
    /// Unweighted Frobenius SVD of `Θ̃` can waste rank on `ker(N_bond)` or
    /// `1/√λ`-amplified near-null modes, dropping physical `N`-norm weight.
    /// Polishing directly in the `Y`-metric (`N = Y† Y`) removes `ker(N)`
    /// contamination completely and restores exact norm conservation and
    /// optimal Hilbert-space truncation error for both dense and symmetric MPS.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn n_norm_polish_pair(
        &self,
        x_target: &[C64],
        a_full: &mut Tensor,
        b_full: &mut Tensor,
        sym: Option<&PeriodicSymmetry>,
        q1: usize,
        q2: usize,
        absorb: crate::mps::Absorb,
        cached_proj: Option<&BondMetricProjector>,
        null_tol: f64,
    ) -> bool {
        if self.n_is_identity || self.phys != 4 {
            return false;
        }
        let Some(proj) = cached_proj else {
            return false;
        };
        if proj.proj_dim() > 4 * MAX_BOND_METRIC_DIM {
            return false;
        }
        let _ = null_tol;
        let eval_err = |a: &Tensor, b: &Tensor| -> f64 {
            let theta = super::dmrg::two_site_dense(a, b);
            let va = self.to_solver_vec(&theta);
            let diff: Vec<C64> = va.iter().zip(x_target).map(|(&u, &v)| u - v).collect();
            let mut nd = vec![ZERO; diff.len()];
            self.norm_linear_op().apply(&diff, &mut nd);
            crate::statevector::krylov::dot(&diff, &nd).re.max(0.0)
        };
        let err0 = eval_err(a_full, b_full);
        if err0 <= 1e-16 {
            return false;
        }
        let z_target = proj.project_in(x_target);
        let k = a_full.right;
        if k == 0 {
            return false;
        }

        // Transform initial `(a_full, b_full)` into gauge coordinates `(a_tilde, b_tilde)`.
        let (l_eff, r_eff, l_full, r_full) = (self.l_eff, self.r_eff, self.l_full, self.r_full);
        let mut a_tilde = Mat::<C64>::zeros(l_eff * 2, k);
        for p1 in 0..2 {
            let ap = Mat::<C64>::from_fn(l_full, k, |l, c| a_full.data[(l * 2 + p1) * k + c]);
            let prod = matmul(self.g_l_inv.as_ref(), ap.as_ref());
            for i in 0..l_eff {
                for c in 0..k {
                    a_tilde[(i * 2 + p1, c)] = prod[(i, c)];
                }
            }
        }
        let mut b_tilde = Mat::<C64>::zeros(k, 2 * r_eff);
        for p2 in 0..2 {
            let bp = Mat::<C64>::from_fn(k, r_full, |c, r| b_full.data[(c * 2 + p2) * r_full + r]);
            let prod = matmul(bp.as_ref(), self.g_r_inv.transpose());
            for c in 0..k {
                for j in 0..r_eff {
                    b_tilde[(c, p2 * r_eff + j)] = prod[(c, j)];
                }
            }
        }

        let (a_cand, b_cand) = match (&proj, sym) {
            (BondMetricProjector::Dense { y_mat, .. }, None) => {
                let rk = y_mat.nrows();
                let mut cur_k = k;
                // Pre-reshape Y for B-step (`(rk * r_eff) × l_eff`) and A-step (`(rk * l_eff) × r_eff`).
                let y_for_b = Mat::<C64>::from_fn(rk * r_eff, l_eff, |row, i| {
                    let c = row / r_eff;
                    let j = row % r_eff;
                    y_mat[(c, i * r_eff + j)]
                });
                let y_for_a = Mat::<C64>::from_fn(rk * l_eff, r_eff, |row, j| {
                    let c = row / l_eff;
                    let i = row % l_eff;
                    y_mat[(c, i * r_eff + j)]
                });
                let rhs_b = Mat::<C64>::from_fn(2 * rk, 2, |row, p2| {
                    let p1 = row / rk;
                    let c = row % rk;
                    z_target[(p1 * 2 + p2) * rk + c]
                });
                let rhs_a = Mat::<C64>::from_fn(2 * rk, 2, |row, p1| {
                    let p2 = row / rk;
                    let c = row % rk;
                    z_target[(p1 * 2 + p2) * rk + c]
                });

                for _sweep in 0..2 {
                    // Orthonormalize A before solving for B.
                    if a_tilde.nrows() >= cur_k {
                        let (q, _r_c) = crate::mps::linalg::thin_qr(a_tilde.as_ref());
                        cur_k = q.ncols();
                        a_tilde = q;
                    }
                    // Solve min_B ‖M_A B - rhs_b‖_2^2.
                    let mut m_a = Mat::<C64>::zeros(2 * rk, cur_k * r_eff);
                    for p1 in 0..2 {
                        let ap =
                            Mat::<C64>::from_fn(l_eff, cur_k, |i, al| a_tilde[(i * 2 + p1, al)]);
                        let prod = matmul(y_for_b.as_ref(), ap.as_ref()); // (rk * r_eff) × cur_k
                        for c in 0..rk {
                            for j in 0..r_eff {
                                for al in 0..cur_k {
                                    m_a[(p1 * rk + c, al * r_eff + j)] = prod[(c * r_eff + j, al)];
                                }
                            }
                        }
                    }
                    let b_sol = min_norm_lstsq(m_a.as_ref(), rhs_b.as_ref(), 1e-12);
                    let res_b = matmul(m_a.as_ref(), b_sol.as_ref()) - &rhs_b;
                    b_tilde = Mat::<C64>::from_fn(cur_k, 2 * r_eff, |al, col| {
                        let p2 = col / r_eff;
                        let j = col % r_eff;
                        b_sol[(al * r_eff + j, p2)]
                    });
                    if res_b.norm_l2() <= 1e-12 {
                        break;
                    }

                    // Orthonormalize B before solving for A.
                    if b_tilde.ncols() >= cur_k {
                        let b_adj = b_tilde.adjoint().to_owned();
                        let (q_adj, _r_adj) = crate::mps::linalg::thin_qr(b_adj.as_ref());
                        cur_k = q_adj.ncols();
                        b_tilde = q_adj.adjoint().to_owned();
                    }
                    // Solve min_A ‖M_B A - rhs_a‖_2^2.
                    let mut m_b = Mat::<C64>::zeros(2 * rk, l_eff * cur_k);
                    for p2 in 0..2 {
                        let bp_t = Mat::<C64>::from_fn(r_eff, cur_k, |j, al| {
                            b_tilde[(al, p2 * r_eff + j)]
                        });
                        let prod = matmul(y_for_a.as_ref(), bp_t.as_ref()); // (rk * l_eff) × cur_k
                        for c in 0..rk {
                            for i in 0..l_eff {
                                for al in 0..cur_k {
                                    m_b[(p2 * rk + c, i * cur_k + al)] = prod[(c * l_eff + i, al)];
                                }
                            }
                        }
                    }
                    let a_sol = min_norm_lstsq(m_b.as_ref(), rhs_a.as_ref(), 1e-12);
                    let res_a = matmul(m_b.as_ref(), a_sol.as_ref()) - &rhs_a;
                    a_tilde = Mat::<C64>::from_fn(l_eff * 2, cur_k, |row, al| {
                        let i = row / 2;
                        let p1 = row % 2;
                        a_sol[(i * cur_k + al, p1)]
                    });
                    if res_a.norm_l2() <= 1e-12 {
                        break;
                    }
                }

                // Map back to full coordinates and restore strict canonical gauge.
                let mut a_raw = Mat::<C64>::zeros(l_full * 2, cur_k);
                for p1 in 0..2 {
                    let ap = Mat::<C64>::from_fn(l_eff, cur_k, |i, al| a_tilde[(i * 2 + p1, al)]);
                    let prod = matmul(self.g_l.as_ref(), ap.as_ref());
                    for l in 0..l_full {
                        for al in 0..cur_k {
                            a_raw[(l * 2 + p1, al)] = prod[(l, al)];
                        }
                    }
                }
                let mut b_raw = Mat::<C64>::zeros(cur_k, 2 * r_full);
                for p2 in 0..2 {
                    let bp =
                        Mat::<C64>::from_fn(cur_k, r_eff, |al, j| b_tilde[(al, p2 * r_eff + j)]);
                    let prod = matmul(bp.as_ref(), self.g_r.transpose());
                    for al in 0..cur_k {
                        for r in 0..r_full {
                            b_raw[(al, p2 * r_full + r)] = prod[(al, r)];
                        }
                    }
                }
                match absorb {
                    crate::mps::Absorb::Right => {
                        let (q_l, r_c) = crate::mps::linalg::thin_qr(a_raw.as_ref());
                        let fk = q_l.ncols();
                        let bf = matmul(r_c.as_ref(), b_raw.as_ref());
                        (
                            Tensor::from_matrix(q_l.as_ref(), l_full, fk),
                            Tensor::from_matrix(bf.as_ref(), fk, r_full),
                        )
                    }
                    crate::mps::Absorb::Left => {
                        let b_adj = b_raw.adjoint().to_owned();
                        let (q_adj, r_adj) = crate::mps::linalg::thin_qr(b_adj.as_ref());
                        let q_r = q_adj.adjoint().to_owned();
                        let l_c = r_adj.adjoint().to_owned();
                        let fk = q_r.nrows();
                        let af = matmul(a_raw.as_ref(), l_c.as_ref());
                        (
                            Tensor::from_matrix(af.as_ref(), l_full, fk),
                            Tensor::from_matrix(q_r.as_ref(), fk, r_full),
                        )
                    }
                }
            }
            (BondMetricProjector::Symmetric { blocks, .. }, Some(sym)) => {
                let left_sec = self.left_eff_sec.as_ref().unwrap();
                let right_sec = self.right_eff_sec.as_ref().unwrap();
                let mid_sec = &sym.bonds[q2];
                let n_mid = mid_sec.len();
                let mut a_sec_map = vec![0usize; l_eff];
                for s in 0..left_sec.len() {
                    for i in left_sec.range(s) {
                        a_sec_map[i] = s;
                    }
                }
                let mut r_sec_map = vec![0usize; r_eff];
                for s in 0..right_sec.len() {
                    for j in right_sec.range(s) {
                        r_sec_map[j] = s;
                    }
                }
                let shift1 = [sym.site_shift(q1, 0), sym.site_shift(q1, 1)];
                let shift2 = [sym.site_shift(q2, 0), sym.site_shift(q2, 1)];
                // Precompute sector-allowed mid ranges for `(left_sec, p1)` and `(right_sec, p2)`.
                let mid_for_lp1 = |ls: usize, p1: usize| -> Option<std::ops::Range<usize>> {
                    let q = sym.charges.add(left_sec.label(ls), &shift1[p1]);
                    mid_sec.find(&q).map(|s| mid_sec.range(s))
                };
                let mid_for_rp2 = |rs: usize, p2: usize| -> Option<std::ops::Range<usize>> {
                    let q = sym.charges.sub(right_sec.label(rs), &shift2[p2]);
                    mid_sec.find(&q).map(|s| mid_sec.range(s))
                };

                // Precompute allowed `(al, j)` columns for each `p2` and `(i, al)` columns for each `p1`.
                let mut b_cols: [Vec<(usize, usize)>; 2] = [Vec::new(), Vec::new()];
                let mut b_col_idx: [Vec<Option<usize>>; 2] =
                    [vec![None; k * r_eff], vec![None; k * r_eff]];
                for p2 in 0..2 {
                    for sm in 0..n_mid {
                        let q_r = sym.charges.add(mid_sec.label(sm), &shift2[p2]);
                        let Some(sr) = right_sec.find(&q_r) else {
                            continue;
                        };
                        for al in mid_sec.range(sm) {
                            for j in right_sec.range(sr) {
                                let idx = b_cols[p2].len();
                                b_cols[p2].push((al, j));
                                b_col_idx[p2][al * r_eff + j] = Some(idx);
                            }
                        }
                    }
                }
                let mut a_cols: [Vec<(usize, usize)>; 2] = [Vec::new(), Vec::new()];
                let mut a_col_idx: [Vec<Option<usize>>; 2] =
                    [vec![None; l_eff * k], vec![None; l_eff * k]];
                for p1 in 0..2 {
                    for sl in 0..left_sec.len() {
                        let q_m = sym.charges.add(left_sec.label(sl), &shift1[p1]);
                        let Some(sm) = mid_sec.find(&q_m) else {
                            continue;
                        };
                        for i in left_sec.range(sl) {
                            for al in mid_sec.range(sm) {
                                let idx = a_cols[p1].len();
                                a_cols[p1].push((i, al));
                                a_col_idx[p1][i * k + al] = Some(idx);
                            }
                        }
                    }
                }

                // Helper to QR-orthonormalize `a_tilde` per mid sector (preserving `mid_sec`).
                let qr_a_sym = |a_t: &mut Mat<C64>, b_t: &mut Mat<C64>| {
                    for sm in 0..n_mid {
                        let rm = mid_sec.range(sm);
                        let dm = rm.len();
                        let mut rows = Vec::new();
                        for (p1, sh) in shift1.iter().enumerate() {
                            let q_l = sym.charges.sub(mid_sec.label(sm), sh);
                            if let Some(sl) = left_sec.find(&q_l) {
                                for i in left_sec.range(sl) {
                                    rows.push(i * 2 + p1);
                                }
                            }
                        }
                        if rows.len() < dm || dm == 0 {
                            continue;
                        }
                        let sub = Mat::<C64>::from_fn(rows.len(), dm, |r, c| {
                            a_t[(rows[r], rm.start + c)]
                        });
                        let (q, rc) = crate::mps::linalg::thin_qr(sub.as_ref());
                        for (r, &row) in rows.iter().enumerate() {
                            for c in 0..dm {
                                a_t[(row, rm.start + c)] = q[(r, c)];
                            }
                        }
                        let b_sub =
                            Mat::<C64>::from_fn(dm, 2 * r_eff, |r, c| b_t[(rm.start + r, c)]);
                        let b_new = matmul(rc.as_ref(), b_sub.as_ref());
                        for r in 0..dm {
                            for c in 0..2 * r_eff {
                                b_t[(rm.start + r, c)] = b_new[(r, c)];
                            }
                        }
                    }
                };
                let lq_b_sym = |a_t: &mut Mat<C64>, b_t: &mut Mat<C64>| {
                    for sm in 0..n_mid {
                        let rm = mid_sec.range(sm);
                        let dm = rm.len();
                        let mut cols = Vec::new();
                        for (p2, sh) in shift2.iter().enumerate() {
                            let q_r = sym.charges.add(mid_sec.label(sm), sh);
                            if let Some(sr) = right_sec.find(&q_r) {
                                for j in right_sec.range(sr) {
                                    cols.push(p2 * r_eff + j);
                                }
                            }
                        }
                        if cols.len() < dm || dm == 0 {
                            continue;
                        }
                        let sub_adj = Mat::<C64>::from_fn(cols.len(), dm, |r, c| {
                            b_t[(rm.start + c, cols[r])].conj()
                        });
                        let (q_adj, r_adj) = crate::mps::linalg::thin_qr(sub_adj.as_ref());
                        let l_c = r_adj.adjoint().to_owned();
                        for (c_idx, &col) in cols.iter().enumerate() {
                            for r in 0..dm {
                                b_t[(rm.start + r, col)] = q_adj[(c_idx, r)].conj();
                            }
                        }
                        let a_sub =
                            Mat::<C64>::from_fn(l_eff * 2, dm, |r, c| a_t[(r, rm.start + c)]);
                        let a_new = matmul(a_sub.as_ref(), l_c.as_ref());
                        for r in 0..l_eff * 2 {
                            for c in 0..dm {
                                a_t[(r, rm.start + c)] = a_new[(r, c)];
                            }
                        }
                    }
                };

                for _sweep in 0..2 {
                    qr_a_sym(&mut a_tilde, &mut b_tilde);
                    // B-step: decoupled per p2.
                    let mut res_b_sq = 0.0f64;
                    b_tilde.fill(ZERO);
                    for p2 in 0..2 {
                        let nb = b_cols[p2].len();
                        let rk0 = blocks[p2].y_mat.nrows();
                        let rk1 = blocks[2 + p2].y_mat.nrows();
                        let nrows = rk0 + rk1;
                        if nb == 0 || nrows == 0 {
                            continue;
                        }
                        let mut m_a = Mat::<C64>::zeros(nrows, nb);
                        let mut rhs = Mat::<C64>::zeros(nrows, 1);
                        for p1 in 0..2 {
                            let blk = &blocks[p1 * 2 + p2];
                            let rkp = blk.y_mat.nrows();
                            if rkp == 0 {
                                continue;
                            }
                            let row_off = if p1 == 0 { 0 } else { rk0 };
                            for c in 0..rkp {
                                rhs[(row_off + c, 0)] = z_target[blk.offset + c];
                            }
                            for a_idx in 0..blk.compact.len() {
                                let li = blk.l_idx[a_idx];
                                let ri = blk.r_idx[a_idx];
                                let Some(rm) = mid_for_lp1(a_sec_map[li], p1) else {
                                    continue;
                                };
                                for al in rm {
                                    let Some(col) = b_col_idx[p2][al * r_eff + ri] else {
                                        continue;
                                    };
                                    let av = a_tilde[(li * 2 + p1, al)];
                                    if av != ZERO {
                                        for c in 0..rkp {
                                            m_a[(row_off + c, col)] += blk.y_mat[(c, a_idx)] * av;
                                        }
                                    }
                                }
                            }
                        }
                        let sol = min_norm_lstsq(m_a.as_ref(), rhs.as_ref(), 1e-12);
                        let res = matmul(m_a.as_ref(), sol.as_ref()) - &rhs;
                        res_b_sq += res.norm_l2().powi(2);
                        for (col, &(al, j)) in b_cols[p2].iter().enumerate() {
                            b_tilde[(al, p2 * r_eff + j)] = sol[(col, 0)];
                        }
                    }
                    if res_b_sq.sqrt() <= 1e-12 {
                        break;
                    }

                    lq_b_sym(&mut a_tilde, &mut b_tilde);
                    // A-step: decoupled per p1.
                    let mut res_a_sq = 0.0f64;
                    a_tilde.fill(ZERO);
                    for p1 in 0..2 {
                        let na = a_cols[p1].len();
                        let rk0 = blocks[p1 * 2].y_mat.nrows();
                        let rk1 = blocks[p1 * 2 + 1].y_mat.nrows();
                        let nrows = rk0 + rk1;
                        if na == 0 || nrows == 0 {
                            continue;
                        }
                        let mut m_b = Mat::<C64>::zeros(nrows, na);
                        let mut rhs = Mat::<C64>::zeros(nrows, 1);
                        for p2 in 0..2 {
                            let blk = &blocks[p1 * 2 + p2];
                            let rkp = blk.y_mat.nrows();
                            if rkp == 0 {
                                continue;
                            }
                            let row_off = if p2 == 0 { 0 } else { rk0 };
                            for c in 0..rkp {
                                rhs[(row_off + c, 0)] = z_target[blk.offset + c];
                            }
                            for a_idx in 0..blk.compact.len() {
                                let li = blk.l_idx[a_idx];
                                let ri = blk.r_idx[a_idx];
                                let Some(rm) = mid_for_rp2(r_sec_map[ri], p2) else {
                                    continue;
                                };
                                for al in rm {
                                    let Some(col) = a_col_idx[p1][li * k + al] else {
                                        continue;
                                    };
                                    let bv = b_tilde[(al, p2 * r_eff + ri)];
                                    if bv != ZERO {
                                        for c in 0..rkp {
                                            m_b[(row_off + c, col)] += blk.y_mat[(c, a_idx)] * bv;
                                        }
                                    }
                                }
                            }
                        }
                        let sol = min_norm_lstsq(m_b.as_ref(), rhs.as_ref(), 1e-12);
                        let res = matmul(m_b.as_ref(), sol.as_ref()) - &rhs;
                        res_a_sq += res.norm_l2().powi(2);
                        for (col, &(i, al)) in a_cols[p1].iter().enumerate() {
                            a_tilde[(i * 2 + p1, al)] = sol[(col, 0)];
                        }
                    }
                    if res_a_sq.sqrt() <= 1e-12 {
                        break;
                    }
                }

                // Map back to full coordinates and restore per-sector canonical gauge.
                let mut a_raw = Mat::<C64>::zeros(l_full * 2, k);
                for p1 in 0..2 {
                    let ap = Mat::<C64>::from_fn(l_eff, k, |i, al| a_tilde[(i * 2 + p1, al)]);
                    let prod = matmul(self.g_l.as_ref(), ap.as_ref());
                    for l in 0..l_full {
                        for al in 0..k {
                            a_raw[(l * 2 + p1, al)] = prod[(l, al)];
                        }
                    }
                }
                let mut b_raw = Mat::<C64>::zeros(k, 2 * r_full);
                for p2 in 0..2 {
                    let bp = Mat::<C64>::from_fn(k, r_eff, |al, j| b_tilde[(al, p2 * r_eff + j)]);
                    let prod = matmul(bp.as_ref(), self.g_r.transpose());
                    for al in 0..k {
                        for r in 0..r_full {
                            b_raw[(al, p2 * r_full + r)] = prod[(al, r)];
                        }
                    }
                }
                let left_full_sec = sym.left_bond(q1);
                let right_full_sec = sym.right_bond(q2);
                match absorb {
                    crate::mps::Absorb::Right => {
                        for sm in 0..n_mid {
                            let rm = mid_sec.range(sm);
                            let dm = rm.len();
                            let mut rows = Vec::new();
                            for (p1, sh) in shift1.iter().enumerate() {
                                let q_l = sym.charges.sub(mid_sec.label(sm), sh);
                                if let Some(sl) = left_full_sec.find(&q_l) {
                                    for l in left_full_sec.range(sl) {
                                        rows.push(l * 2 + p1);
                                    }
                                }
                            }
                            if rows.len() < dm || dm == 0 {
                                continue;
                            }
                            let sub = Mat::<C64>::from_fn(rows.len(), dm, |r, c| {
                                a_raw[(rows[r], rm.start + c)]
                            });
                            let (q, rc) = crate::mps::linalg::thin_qr(sub.as_ref());
                            for (r, &row) in rows.iter().enumerate() {
                                for c in 0..dm {
                                    a_raw[(row, rm.start + c)] = q[(r, c)];
                                }
                            }
                            let b_sub = Mat::<C64>::from_fn(dm, 2 * r_full, |r, c| {
                                b_raw[(rm.start + r, c)]
                            });
                            let b_new = matmul(rc.as_ref(), b_sub.as_ref());
                            for r in 0..dm {
                                for c in 0..2 * r_full {
                                    b_raw[(rm.start + r, c)] = b_new[(r, c)];
                                }
                            }
                        }
                    }
                    crate::mps::Absorb::Left => {
                        for sm in 0..n_mid {
                            let rm = mid_sec.range(sm);
                            let dm = rm.len();
                            let mut cols = Vec::new();
                            for (p2, sh) in shift2.iter().enumerate() {
                                let q_r = sym.charges.add(mid_sec.label(sm), sh);
                                if let Some(sr) = right_full_sec.find(&q_r) {
                                    for r in right_full_sec.range(sr) {
                                        cols.push(p2 * r_full + r);
                                    }
                                }
                            }
                            if cols.len() < dm || dm == 0 {
                                continue;
                            }
                            let sub_adj = Mat::<C64>::from_fn(cols.len(), dm, |r, c| {
                                b_raw[(rm.start + c, cols[r])].conj()
                            });
                            let (q_adj, r_adj) = crate::mps::linalg::thin_qr(sub_adj.as_ref());
                            let l_c = r_adj.adjoint().to_owned();
                            for (c_idx, &col) in cols.iter().enumerate() {
                                for r in 0..dm {
                                    b_raw[(rm.start + r, col)] = q_adj[(c_idx, r)].conj();
                                }
                            }
                            let a_sub = Mat::<C64>::from_fn(l_full * 2, dm, |r, c| {
                                a_raw[(r, rm.start + c)]
                            });
                            let a_new = matmul(a_sub.as_ref(), l_c.as_ref());
                            for r in 0..l_full * 2 {
                                for c in 0..dm {
                                    a_raw[(r, rm.start + c)] = a_new[(r, c)];
                                }
                            }
                        }
                    }
                }
                (
                    Tensor::from_matrix(a_raw.as_ref(), l_full, k),
                    Tensor::from_matrix(b_raw.as_ref(), k, r_full),
                )
            }
            _ => return false,
        };

        let err1 = eval_err(&a_cand, &b_cand);
        if err1.is_finite() && err1 < err0 {
            *a_full = a_cand;
            *b_full = b_cand;
            true
        } else {
            false
        }
    }
}

/// Minimum-norm least-squares solution `min ‖A X - B‖_F` via thin SVD pseudoinverse.
fn min_norm_lstsq(a: MatRef<'_, C64>, b: MatRef<'_, C64>, rcond: f64) -> Mat<C64> {
    let (u, s, v) = crate::mps::linalg::thin_svd(a);
    let s_max = s.iter().copied().fold(0.0f64, f64::max);
    if s_max <= 0.0 {
        return Mat::zeros(a.ncols(), b.ncols());
    }
    let thresh = rcond * s_max;
    let mut ub = matmul(u.adjoint(), b);
    for i in 0..s.len() {
        let inv = if s[i] > thresh { 1.0 / s[i] } else { 0.0 };
        for j in 0..ub.ncols() {
            ub[(i, j)] *= inv;
        }
    }
    matmul(v.as_ref(), ub.as_ref())
}

/// Maximum single-block bond metric dimension for direct `N_bond`
/// canonical orthogonalization on general rings (`χ ≤ 8` dense, or small
/// symmetric blocks; above this, `O(p D χ³)` Jacobi-preconditioned matrix-free
/// Davidson / subspace Rayleigh–Ritz runs so we never pay `O(χ⁶)` EVD cost).
const MAX_BOND_METRIC_DIM: usize = 64;

/// On small rings (`n <= 12`) where `N_bond` is structurally rank-deficient
/// (`χ² > 2^{L_opp}`), allow direct bond-metric projection up to dim 256.
const SHORT_RING_BOND_METRIC_DIM: usize = 256;

/// Exact canonical orthogonaliser `X = U_kept D_kept^{-1/2}`, `Y = D_kept^{1/2} U_kept†`
/// of the bond metric `N_bond` (dense: shared across all `phys` states;
/// symmetric: block-diagonal per physical state `p`).
pub(super) enum BondMetricProjector {
    Dense {
        l: usize,
        r: usize,
        phys: usize,
        /// `X` of shape `(l * r) × r_kept`.
        x_mat: Mat<C64>,
        /// `Y` of shape `r_kept × (l * r)`.
        y_mat: Mat<C64>,
    },
    Symmetric {
        r: usize,
        phys: usize,
        /// Per physical state `p`: `(compact_indices, l_indices, r_indices, X_p, Y_p, proj_offset)`.
        blocks: Vec<SymMetricBlock>,
        total_proj: usize,
    },
}

pub(super) struct SymMetricBlock {
    compact: Vec<usize>,
    l_idx: Vec<usize>,
    r_idx: Vec<usize>,
    x_mat: Mat<C64>,
    y_mat: Mat<C64>,
    offset: usize,
}

impl BondMetricProjector {
    fn try_build(prob: &LocalProblem, null_tol: f64) -> Option<Self> {
        let (l, r, phys) = (prob.l_eff, prob.r_eff, prob.phys);
        let max_dim = if prob.short_ring {
            SHORT_RING_BOND_METRIC_DIM
        } else {
            MAX_BOND_METRIC_DIM
        };
        let thresh_rel = null_tol.max(1e-12);
        match &prob.layout {
            None => {
                let m = l * r;
                if m == 0 || m > max_dim {
                    return None;
                }
                // Build N_bond[(i, j), (i', j')] = Σ_μ L_μ[i, i'] R_μ[j, j'] in ONE GEMM!
                let k = prob.norm_terms.len();
                let mut a_mat = Mat::<C64>::zeros(l * l, k);
                let mut b_mat = Mat::<C64>::zeros(k, r * r);
                for (mu, (lm, rm)) in prob.norm_terms.iter().enumerate() {
                    for i in 0..l {
                        for ip in 0..l {
                            a_mat[(i * l + ip, mu)] = lm[(i, ip)];
                        }
                    }
                    for j in 0..r {
                        for jp in 0..r {
                            b_mat[(mu, j * r + jp)] = rm[(j, jp)];
                        }
                    }
                }
                let prod = matmul(a_mat.as_ref(), b_mat.as_ref()); // (l²) × (r²)
                let mut n_bond = Mat::<C64>::zeros(m, m);
                for i in 0..l {
                    for j in 0..r {
                        let row = i * r + j;
                        for ip in 0..l {
                            for jp in 0..r {
                                let col = ip * r + jp;
                                if row >= col {
                                    let v1 = prod[(i * l + ip, j * r + jp)];
                                    let v2 = prod[(ip * l + i, jp * r + j)].conj();
                                    let val = (v1 + v2) * 0.5;
                                    n_bond[(row, col)] = val;
                                    n_bond[(col, row)] = val.conj();
                                }
                            }
                        }
                    }
                }
                let (vals, u) = hermitian_evd(n_bond.as_ref());
                let max_n = vals.iter().copied().fold(0.0f64, f64::max);
                if max_n <= 0.0 {
                    return None;
                }
                let thresh = thresh_rel * max_n;
                let kept: Vec<usize> = vals
                    .iter()
                    .enumerate()
                    .filter(|&(_, &v)| v > thresh)
                    .map(|(idx, _)| idx)
                    .collect();
                if kept.is_empty() {
                    return None;
                }
                let rk = kept.len();
                let x_mat = Mat::<C64>::from_fn(m, rk, |row, c| {
                    let idx = kept[c];
                    u[(row, idx)] / vals[idx].sqrt()
                });
                let y_mat = Mat::<C64>::from_fn(rk, m, |c, col| {
                    let idx = kept[c];
                    u[(col, idx)].conj() * vals[idx].sqrt()
                });
                Some(BondMetricProjector::Dense {
                    l,
                    r,
                    phys,
                    x_mat,
                    y_mat,
                })
            }
            Some(layout) => {
                // Group compact indices by physical state `p ∈ 0..phys`.
                let mut by_p: Vec<(Vec<usize>, Vec<usize>, Vec<usize>)> =
                    vec![(Vec::new(), Vec::new(), Vec::new()); phys];
                for &(s, _ds, offset, width) in &layout.rows {
                    let rs = layout.left.range(s);
                    for p in 0..phys {
                        let Some((_, col0, t)) = layout.cols[s * phys + p] else {
                            continue;
                        };
                        let rt = layout.right.range(t);
                        for (a, li) in rs.clone().enumerate() {
                            for (b, ri) in rt.clone().enumerate() {
                                by_p[p].0.push(offset + a * width + col0 + b);
                                by_p[p].1.push(li);
                                by_p[p].2.push(ri);
                            }
                        }
                    }
                }
                if by_p.iter().any(|b| b.0.len() > max_dim) {
                    return None;
                }
                // EVD each physical block `N_p`.
                let mut evds: Vec<(Vec<f64>, Mat<C64>)> = Vec::with_capacity(phys);
                let mut max_n = 0.0f64;
                for (_, l_idx, r_idx) in &by_p {
                    let mp = l_idx.len();
                    if mp == 0 {
                        evds.push((Vec::new(), Mat::zeros(0, 0)));
                        continue;
                    }
                    let mut n_p = Mat::<C64>::zeros(mp, mp);
                    for (lm, rm) in &prob.norm_terms {
                        for a in 0..mp {
                            let (la, ra) = (l_idx[a], r_idx[a]);
                            for b in 0..=a {
                                let (lb, rb) = (l_idx[b], r_idx[b]);
                                let v1 = lm[(la, lb)] * rm[(ra, rb)];
                                let v2 = (lm[(lb, la)] * rm[(rb, ra)]).conj();
                                n_p[(a, b)] += (v1 + v2) * 0.5;
                            }
                        }
                    }
                    for a in 0..mp {
                        n_p[(a, a)] = C64::new(n_p[(a, a)].re, 0.0);
                        for b in 0..a {
                            let v = n_p[(a, b)];
                            n_p[(b, a)] = v.conj();
                        }
                    }
                    let (vals, u) = hermitian_evd(n_p.as_ref());
                    for &v in &vals {
                        max_n = max_n.max(v);
                    }
                    evds.push((vals, u));
                }
                if max_n <= 0.0 {
                    return None;
                }
                let thresh = thresh_rel * max_n;
                let mut blocks = Vec::with_capacity(phys);
                let mut total_proj = 0;
                for ((compact, l_idx, r_idx), (vals, u)) in by_p.into_iter().zip(evds) {
                    let mp = compact.len();
                    let kept: Vec<usize> = vals
                        .iter()
                        .enumerate()
                        .filter(|&(_, &v)| v > thresh)
                        .map(|(idx, _)| idx)
                        .collect();
                    let rk = kept.len();
                    let x_mat = Mat::<C64>::from_fn(mp, rk, |row, c| {
                        let idx = kept[c];
                        u[(row, idx)] / vals[idx].sqrt()
                    });
                    let y_mat = Mat::<C64>::from_fn(rk, mp, |c, col| {
                        let idx = kept[c];
                        u[(col, idx)].conj() * vals[idx].sqrt()
                    });
                    blocks.push(SymMetricBlock {
                        compact,
                        l_idx,
                        r_idx,
                        x_mat,
                        y_mat,
                        offset: total_proj,
                    });
                    total_proj += rk;
                }
                if total_proj == 0 {
                    return None;
                }
                Some(BondMetricProjector::Symmetric {
                    r,
                    phys,
                    blocks,
                    total_proj,
                })
            }
        }
    }

    fn proj_dim(&self) -> usize {
        match self {
            BondMetricProjector::Dense { phys, x_mat, .. } => phys * x_mat.ncols(),
            BondMetricProjector::Symmetric { total_proj, .. } => *total_proj,
        }
    }

    /// Map full/compact vector `v` into projected orthonormal coordinates `z = Y v`.
    fn project_in(&self, v: &[C64]) -> Vec<C64> {
        match self {
            BondMetricProjector::Dense {
                l, r, phys, y_mat, ..
            } => {
                let m = l * r;
                let rk = y_mat.nrows();
                let mut v_mp = Mat::<C64>::zeros(m, *phys);
                for i in 0..*l {
                    for p in 0..*phys {
                        for j in 0..*r {
                            v_mp[(i * r + j, p)] = v[(i * phys + p) * r + j];
                        }
                    }
                }
                let z_mat = matmul(y_mat.as_ref(), v_mp.as_ref()); // rk × phys
                let mut out = vec![ZERO; phys * rk];
                for p in 0..*phys {
                    for c in 0..rk {
                        out[p * rk + c] = z_mat[(c, p)];
                    }
                }
                out
            }
            BondMetricProjector::Symmetric {
                blocks, total_proj, ..
            } => {
                let mut out = vec![ZERO; *total_proj];
                for blk in blocks {
                    let rk = blk.y_mat.nrows();
                    if rk == 0 {
                        continue;
                    }
                    let mp = blk.compact.len();
                    for c in 0..rk {
                        let mut acc = ZERO;
                        for a in 0..mp {
                            acc += blk.y_mat[(c, a)] * v[blk.compact[a]];
                        }
                        out[blk.offset + c] = acc;
                    }
                }
                out
            }
        }
    }

    /// Map projected coordinates `z` to a dense `(l, phys, r)` tensor `x = X z`.
    fn expand_to_dense(&self, z: &[C64], out: &mut [C64]) {
        match self {
            BondMetricProjector::Dense {
                l, r, phys, x_mat, ..
            } => {
                let rk = x_mat.ncols();
                let z_mat = Mat::<C64>::from_fn(rk, *phys, |c, p| z[p * rk + c]);
                let x_mp = matmul(x_mat.as_ref(), z_mat.as_ref()); // (l*r) × phys
                for i in 0..*l {
                    for p in 0..*phys {
                        for j in 0..*r {
                            out[(i * phys + p) * r + j] = x_mp[(i * r + j, p)];
                        }
                    }
                }
            }
            BondMetricProjector::Symmetric {
                r, phys, blocks, ..
            } => {
                out.fill(ZERO);
                for (p, blk) in blocks.iter().enumerate() {
                    let rk = blk.x_mat.ncols();
                    if rk == 0 {
                        continue;
                    }
                    let mp = blk.compact.len();
                    let zp = &z[blk.offset..][..rk];
                    for a in 0..mp {
                        let mut acc = ZERO;
                        for c in 0..rk {
                            acc += blk.x_mat[(a, c)] * zp[c];
                        }
                        out[(blk.l_idx[a] * phys + p) * r + blk.r_idx[a]] = acc;
                    }
                }
            }
        }
    }

    /// Map projected coordinates `z` back to solver (dense or compact) vector `x = X z`.
    fn expand_to_solver(&self, z: &[C64], solver_dim: usize) -> Vec<C64> {
        match self {
            BondMetricProjector::Dense { .. } => {
                let mut out = vec![ZERO; solver_dim];
                self.expand_to_dense(z, &mut out);
                out
            }
            BondMetricProjector::Symmetric { blocks, .. } => {
                let mut out = vec![ZERO; solver_dim];
                for blk in blocks {
                    let rk = blk.x_mat.ncols();
                    if rk == 0 {
                        continue;
                    }
                    let mp = blk.compact.len();
                    let zp = &z[blk.offset..][..rk];
                    for a in 0..mp {
                        let mut acc = ZERO;
                        for c in 0..rk {
                            acc += blk.x_mat[(a, c)] * zp[c];
                        }
                        out[blk.compact[a]] = acc;
                    }
                }
                out
            }
        }
    }

    /// Project a dense `(l, phys, r)` result `hx` by `X†`: `z_out = X† hx`.
    fn contract_from_dense(&self, hx: &[C64], out: &mut [C64]) {
        match self {
            BondMetricProjector::Dense {
                l, r, phys, x_mat, ..
            } => {
                let m = l * r;
                let rk = x_mat.ncols();
                let mut h_mp = Mat::<C64>::zeros(m, *phys);
                for i in 0..*l {
                    for p in 0..*phys {
                        for j in 0..*r {
                            h_mp[(i * r + j, p)] = hx[(i * phys + p) * r + j];
                        }
                    }
                }
                let z_mat = matmul(x_mat.adjoint(), h_mp.as_ref()); // rk × phys
                for p in 0..*phys {
                    for c in 0..rk {
                        out[p * rk + c] = z_mat[(c, p)];
                    }
                }
            }
            BondMetricProjector::Symmetric {
                r, phys, blocks, ..
            } => {
                out.fill(ZERO);
                for (p, blk) in blocks.iter().enumerate() {
                    let rk = blk.x_mat.ncols();
                    if rk == 0 {
                        continue;
                    }
                    let mp = blk.compact.len();
                    let out_p = &mut out[blk.offset..][..rk];
                    for a in 0..mp {
                        let hv = hx[(blk.l_idx[a] * phys + p) * r + blk.r_idx[a]];
                        if hv != ZERO {
                            for c in 0..rk {
                                out_p[c] += blk.x_mat[(a, c)].conj() * hv;
                            }
                        }
                    }
                }
            }
        }
    }

    fn solve_lowest(
        &self,
        prob: &LocalProblem,
        v0: &[C64],
        options: &crate::statevector::eigen::EigenOptions,
    ) -> Result<crate::statevector::eigen::Ritz<C64>, Error> {
        let pdim = self.proj_dim();
        let dense_len = prob.l_eff * prob.phys * prob.r_eff;
        let op = ProjectedHamOp {
            proj: self,
            ham: &prob.ham_op,
            dense_len,
        };
        let z0_raw = self.project_in(v0);
        let z_norm = crate::statevector::krylov::norm(&z0_raw);
        let mut z0 = if z_norm > 0.0 {
            z0_raw.iter().map(|&a| a / z_norm).collect()
        } else {
            let mut u = vec![ZERO; pdim];
            u[0] = ONE;
            u
        };
        // For very small projected spaces (≤ 12), direct EVD is exact to 1e-15
        // and takes ≤ 12 matvecs. Above that, 1-pass full-reorthogonalization
        // Lanczos from `z0` (storing only ≤ 48 tiny projected vectors!) avoids
        // the 2-pass replay and ghost-copy slowdown of un-reorthogonalised
        // Lanczos, converging in ~5–18 matvecs!
        let (e0, z_sol, converged) = if pdim <= 12 {
            let mut h_mat = Mat::<C64>::zeros(pdim, pdim);
            let mut e = vec![ZERO; pdim];
            let mut col = vec![ZERO; pdim];
            for j in 0..pdim {
                e[j] = ONE;
                op.apply(&e, &mut col);
                for i in 0..pdim {
                    h_mat[(i, j)] = col[i];
                }
                e[j] = ZERO;
            }
            for i in 0..pdim {
                h_mat[(i, i)] = C64::new(h_mat[(i, i)].re, 0.0);
                for j in 0..i {
                    let hij = (h_mat[(i, j)] + h_mat[(j, i)].conj()) * 0.5;
                    h_mat[(i, j)] = hij;
                    h_mat[(j, i)] = hij.conj();
                }
            }
            let (vals, u) = hermitian_evd(h_mat.as_ref());
            let vec: Vec<C64> = (0..pdim).map(|i| u[(i, 0)]).collect();
            (vals[0], vec, true)
        } else {
            lanczos_full_reorth(
                &op,
                &z0,
                options.tolerance,
                pdim.min(options.max_iterations).min(64),
            )
        };
        // Preserve `v0`'s null-space component in `ker(Ñ)` ONLY when `v0` has
        // negligible weight in dropped positive modes of `N_bond` (otherwise
        // adding `alpha * v0` back would contaminate the solution with dropped
        // modes when `reg` was raised!).
        let vector = if z_norm > 1e-12 {
            let overlap = crate::statevector::krylov::dot(&z0, &z_sol);
            let alpha = overlap / z_norm;
            let delta_z: Vec<C64> = z_sol
                .iter()
                .zip(&z0)
                .map(|(&s, &u)| s - overlap * u)
                .collect();
            let mut lifted = self.expand_to_solver(&delta_z, v0.len());
            for (dst, &v) in lifted.iter_mut().zip(v0) {
                *dst += alpha * v;
            }
            let mut nl = vec![ZERO; lifted.len()];
            prob.norm_linear_op().apply(&lifted, &mut nl);
            let nl_quad = crate::statevector::krylov::dot(&lifted, &nl).re;
            if (nl_quad - 1.0).abs() <= 1e-6 {
                lifted
            } else {
                self.expand_to_solver(&z_sol, v0.len())
            }
        } else {
            self.expand_to_solver(&z_sol, v0.len())
        };
        let _ = &mut z0;
        Ok(crate::statevector::eigen::Ritz {
            value: e0,
            vector,
            converged,
        })
    }

    fn evolve(&self, prob: &LocalProblem, x: &mut [C64], dt: f64) -> Result<(), Error> {
        let pdim = self.proj_dim();
        let dense_len = prob.l_eff * prob.phys * prob.r_eff;
        let op = ProjectedHamOp {
            proj: self,
            ham: &prob.ham_op,
            dense_len,
        };
        let z0 = self.project_in(x);
        let mut z = z0.clone();
        if pdim <= 12 {
            let mut h_mat = Mat::<C64>::zeros(pdim, pdim);
            let mut e = vec![ZERO; pdim];
            let mut col = vec![ZERO; pdim];
            for j in 0..pdim {
                e[j] = ONE;
                op.apply(&e, &mut col);
                for i in 0..pdim {
                    h_mat[(i, j)] = col[i];
                }
                e[j] = ZERO;
            }
            for i in 0..pdim {
                h_mat[(i, i)] = C64::new(h_mat[(i, i)].re, 0.0);
                for j in 0..i {
                    let hij = (h_mat[(i, j)] + h_mat[(j, i)].conj()) * 0.5;
                    h_mat[(i, j)] = hij;
                    h_mat[(j, i)] = hij.conj();
                }
            }
            let (vals, u) = hermitian_evd(h_mat.as_ref());
            let mut eig = vec![ZERO; pdim];
            for j in 0..pdim {
                let mut acc = ZERO;
                for i in 0..pdim {
                    acc += u[(i, j)].conj() * z[i];
                }
                eig[j] = acc * C64::cis(-vals[j] * dt);
            }
            for i in 0..pdim {
                let mut acc = ZERO;
                for j in 0..pdim {
                    acc += u[(i, j)] * eig[j];
                }
                z[i] = acc;
            }
        } else {
            lanczos_expm_full_reorth(&op, &mut z, dt, 1e-12, pdim.min(32));
        }
        // Incremental lift `x += X (z(dt) - z(0))`: preserves `ker(Ñ)` gauge
        // component so `x` changes smoothly by `O(dt)` with zero rank jump!
        let delta_z: Vec<C64> = z.iter().zip(&z0).map(|(&a, &b)| a - b).collect();
        let dx = self.expand_to_solver(&delta_z, x.len());
        for (dst, &d) in x.iter_mut().zip(&dx) {
            *dst += d;
        }
        Ok(())
    }
}

/// 1-pass full-reorthogonalization Lanczos matrix exponential
/// `z <- exp(-i H dt) z` for small-to-medium projected Hermitian operators.
/// Converges in ~4–8 matvecs for typical TDVP substeps (zero replay pass!).
fn lanczos_expm_full_reorth(
    op: &impl LinearOperator<C64>,
    z: &mut [C64],
    time: f64,
    tol: f64,
    max_krylov: usize,
) {
    use crate::mps::linalg::symmetric_evd_f64;
    use crate::statevector::krylov::{dot, norm};

    let dim = z.len();
    let max_k = max_krylov.min(dim).max(1);
    let mut remaining = time;
    let mut substeps = 0;

    while remaining.abs() > 1e-15 && substeps < 32 {
        substeps += 1;
        let beta0 = norm(z);
        if beta0 <= 1e-30 {
            return;
        }
        let inv0 = 1.0 / beta0;
        let mut basis: Vec<Vec<C64>> = Vec::with_capacity(max_k);
        basis.push(z.iter().map(|&a| a * inv0).collect());
        let mut alphas: Vec<f64> = Vec::with_capacity(max_k);
        let mut betas: Vec<f64> = Vec::with_capacity(max_k);
        let mut w = vec![ZERO; dim];
        let mut covered: Option<(f64, Vec<C64>)> = None;

        let expm_e1 = |a: &[f64], b: &[f64], dt: f64| -> Vec<C64> {
            let m = a.len();
            let mut t = Mat::<f64>::zeros(m, m);
            for i in 0..m {
                t[(i, i)] = a[i];
                if i + 1 < m {
                    t[(i, i + 1)] = b[i];
                    t[(i + 1, i)] = b[i];
                }
            }
            let (vals, u) = symmetric_evd_f64(t.as_ref());
            let mut out = vec![ZERO; m];
            for j in 0..m {
                let phase = C64::cis(-vals[j] * dt);
                let u0j = u[(0, j)];
                for i in 0..m {
                    out[i] += phase * (u[(i, j)] * u0j);
                }
            }
            out
        };

        for k in 0..max_k {
            op.apply(&basis[k], &mut w);
            let alpha = dot(&basis[k], &w).re;
            alphas.push(alpha);
            for _ in 0..2 {
                for b in &basis {
                    let c = dot(b, &w);
                    for (wj, &bj) in w.iter_mut().zip(b) {
                        *wj -= c * bj;
                    }
                }
            }
            let beta = norm(&w);
            let m = alphas.len();
            let scale = alphas.iter().fold(1.0f64, |acc, a| acc.max(a.abs()));
            let closed = beta <= 1e-13 * scale || k + 1 == dim;
            if closed || m >= 3 {
                let y = expm_e1(&alphas, &betas, remaining);
                let err = if closed {
                    0.0
                } else {
                    beta * remaining.abs() * y[m - 1].norm()
                };
                if err <= tol {
                    covered = Some((remaining, y));
                    break;
                }
            }
            if k + 1 == max_k {
                betas.push(beta);
                break;
            }
            betas.push(beta);
            let inv_b = 1.0 / beta;
            basis.push(w.iter().map(|&a| a * inv_b).collect());
        }

        let m = alphas.len();
        let (dt, y) = match covered {
            Some(pair) => pair,
            None => {
                let last_b = *betas.last().unwrap_or(&0.0);
                let mut frac = 0.5;
                let mut chosen = None;
                for _ in 0..16 {
                    let dt_try = remaining * frac;
                    let y_try = expm_e1(&alphas, &betas[..m - 1], dt_try);
                    if last_b * dt_try.abs() * y_try[m - 1].norm() <= tol {
                        chosen = Some((dt_try, y_try));
                        break;
                    }
                    frac *= 0.5;
                }
                chosen.unwrap_or_else(|| (remaining, expm_e1(&alphas, &betas[..m - 1], remaining)))
            }
        };

        z.fill(ZERO);
        for k in 0..m {
            let ck = y[k] * beta0;
            for (dst, &bk) in z.iter_mut().zip(&basis[k]) {
                *dst += ck * bk;
            }
        }
        remaining -= dt;
    }
}

struct ProjectedHamOp<'a> {
    proj: &'a BondMetricProjector,
    ham: &'a LocalHamOp,
    dense_len: usize,
}

impl LinearOperator<C64> for ProjectedHamOp<'_> {
    fn apply(&self, z: &[C64], out: &mut [C64]) {
        let mut dense_in = vec![ZERO; self.dense_len];
        let mut dense_out = vec![ZERO; self.dense_len];
        self.proj.expand_to_dense(z, &mut dense_in);
        self.ham.apply(&dense_in, &mut dense_out);
        self.proj.contract_from_dense(&dense_out, out);
    }
}

/// 1-pass full-reorthogonalization Lanczos for small-to-medium projected
/// Hermitian operators (`pdim ≤ 1600`): zero replay pass, guaranteed exact
/// orthogonality, converges in ~5–20 matvecs!
fn lanczos_full_reorth(
    op: &impl LinearOperator<C64>,
    v0: &[C64],
    tol: f64,
    max_iters: usize,
) -> (f64, Vec<C64>, bool) {
    use crate::mps::linalg::symmetric_evd_f64;
    use crate::statevector::krylov::{dot, norm};

    let dim = v0.len();
    let max_k = max_iters.min(dim).max(1);
    let mut basis: Vec<Vec<C64>> = Vec::with_capacity(max_k);
    basis.push(v0.to_vec());
    let mut alphas: Vec<f64> = Vec::with_capacity(max_k);
    let mut betas: Vec<f64> = Vec::with_capacity(max_k);
    let mut w = vec![ZERO; dim];

    let mut best_e = f64::INFINITY;
    let mut best_s: Vec<f64> = vec![1.0];
    let mut best_res = f64::INFINITY;
    let mut converged = false;

    for k in 0..max_k {
        op.apply(&basis[k], &mut w);
        let alpha = dot(&basis[k], &w).re;
        alphas.push(alpha);

        // Two-pass Modified Gram–Schmidt against all previous basis vectors.
        for _ in 0..2 {
            for b in &basis {
                let c = dot(b, &w);
                for (wj, &bj) in w.iter_mut().zip(b) {
                    *wj -= c * bj;
                }
            }
        }
        let beta = norm(&w);
        let scale = alphas.iter().fold(1.0f64, |acc, a| acc.max(a.abs()));
        let floor = 1e-13 * scale;

        // Check Ritz convergence at every step (tridiagonal EVD of size ≤ 64 is < 20 µs).
        let m = alphas.len();
        let mut t = Mat::<f64>::zeros(m, m);
        for i in 0..m {
            t[(i, i)] = alphas[i];
            if i + 1 < m {
                t[(i, i + 1)] = betas[i];
                t[(i + 1, i)] = betas[i];
            }
        }
        let (vals, u) = symmetric_evd_f64(t.as_ref());
        let e_ritz = vals[0];
        let s: Vec<f64> = (0..m).map(|i| u[(i, 0)]).collect();
        let res_est = beta * s[m - 1].abs();
        if res_est < best_res {
            best_res = res_est;
            best_e = e_ritz;
            best_s = s;
        }
        if res_est <= tol.max(floor) || beta <= floor || k + 1 == dim {
            converged = true;
            break;
        }
        if k + 1 == max_k {
            break;
        }
        betas.push(beta);
        let inv_b = 1.0 / beta;
        basis.push(w.iter().map(|&a| a * inv_b).collect());
    }

    let mut sol = vec![ZERO; dim];
    for (k, &sk) in best_s.iter().enumerate() {
        let c = C64::new(sk, 0.0);
        for (dst, &bk) in sol.iter_mut().zip(&basis[k]) {
            *dst += c * bk;
        }
    }
    let sn = norm(&sol);
    if sn > 0.0 {
        sol.iter_mut().for_each(|a| *a /= sn);
    }
    (best_e, sol, converged)
}

/// Wrapper applying a dense gauge-coordinate operator on compact symmetric
/// vectors (scattering in, applying, gathering out — guaranteeing zero charge
/// leakage in Krylov/Davidson recurrences!).
pub(super) struct PackedOp<'a, O> {
    inner: &'a O,
    layout: Option<&'a PeriodicLayout>,
}

impl<O: LinearOperator<C64>> LinearOperator<C64> for PackedOp<'_, O> {
    fn apply(&self, x: &[C64], out: &mut [C64]) {
        match self.layout {
            None => self.inner.apply(x, out),
            Some(layout) => {
                // Note: LOCAL_SCRATCH is borrowed inside `self.inner.apply`, so
                // PACKED_SCRATCH is kept in a separate thread-local cell.
                let dense_len = layout.left.dim() * layout.phys * layout.right.dim();
                PACKED_SCRATCH.with(|cell| {
                    let mut sc = cell.borrow_mut();
                    ensure_len(&mut sc.dense_in, dense_len);
                    ensure_len(&mut sc.dense_out, dense_len);
                    let PackedScratch {
                        dense_in,
                        dense_out,
                    } = &mut *sc;
                    layout.scatter_into(x, &mut dense_in[..dense_len]);
                    self.inner
                        .apply(&dense_in[..dense_len], &mut dense_out[..dense_len]);
                    layout.gather_into(&dense_out[..dense_len], out);
                });
            }
        }
    }
}

/// Stacked two-GEMM operator `out += Σ_{m=0}^{k-1} L_m x R_m^T` (identity on
/// physical indices), replacing `k` individual matmuls with **two** BLAS-3 GEMMs!
struct StackedPairOp {
    l: usize,
    r: usize,
    phys: usize,
    k: usize,
    /// Row-major `(k * l) × l`: blocks `L_0, ..., L_{k-1}` stacked vertically.
    l_stack: Vec<C64>,
    /// Row-major `(k * r) × r`: blocks `R_0^T, ..., R_{k-1}^T` stacked vertically.
    r_stack: Vec<C64>,
}

impl StackedPairOp {
    fn new(l: usize, r: usize, phys: usize, terms: &[(Mat<C64>, Mat<C64>)]) -> Self {
        let k = terms.len();
        let mut l_stack = Vec::with_capacity(k * l * l);
        let mut r_stack = Vec::with_capacity(k * r * r);
        for (lm, rm) in terms {
            for i in 0..l {
                for j in 0..l {
                    l_stack.push(lm[(i, j)]);
                }
            }
            for i in 0..r {
                for j in 0..r {
                    r_stack.push(rm[(j, i)]); // R_m^T[i, j] = R_m[j, i]
                }
            }
        }
        StackedPairOp {
            l,
            r,
            phys,
            k,
            l_stack,
            r_stack,
        }
    }

    fn apply_add(&self, x: &[C64], out: &mut [C64]) {
        let (l, r, phys, k) = (self.l, self.r, self.phys, self.k);
        if k == 0 || l == 0 || r == 0 {
            return;
        }
        let y_len = k * l * phys * r;
        let kr = k * r;
        let pr = phys * r;
        let par = if k * l * l * pr >= 1 << 26 {
            faer::get_global_parallelism()
        } else {
            Par::Seq
        };
        LOCAL_SCRATCH.with(|cell| {
            let mut sc = cell.borrow_mut();
            ensure_len(&mut sc.y, y_len);
            ensure_len(&mut sc.z, y_len);
            let LocalScratch { y, z, .. } = &mut *sc;
            // Step 1: Y[(m, i), (p, j)] = Σ_a L_m[i, a] X[a, (p, j)] -> (k * l) × (phys * r)
            faer_matmul(
                MatMut::from_row_major_slice_mut(&mut y[..y_len], k * l, pr),
                Accum::Replace,
                MatRef::from_row_major_slice(&self.l_stack, k * l, l),
                MatRef::from_row_major_slice(x, l, pr),
                ONE,
                par,
            );
            // Permute Y(m, i, p, j) -> Z(i, p, m, j) of shape (l * phys) × (k * r).
            for m in 0..k {
                let m_off = m * l * pr;
                let dst_m = m * r;
                for i in 0..l {
                    for p in 0..phys {
                        let src = &y[m_off + i * pr + p * r..][..r];
                        let dst = &mut z[(i * phys + p) * kr + dst_m..][..r];
                        dst.copy_from_slice(src);
                    }
                }
            }
            // Step 2: out[(i, p), j'] += Σ_{m, j} Z[(i, p), (m, j)] R_m^T[j, j']
            faer_matmul(
                MatMut::from_row_major_slice_mut(out, l * phys, r),
                Accum::Add,
                MatRef::from_row_major_slice(&z[..y_len], l * phys, kr),
                MatRef::from_row_major_slice(&self.r_stack, kr, r),
                ONE,
                par,
            );
        });
    }

    #[cfg(feature = "gpu")]
    fn gpu_spec(&self) -> super::gpu::GpuTermSpec<'_> {
        let mut rows = Vec::with_capacity(self.k * self.phys);
        for m in 0..self.k {
            for p in 0..self.phys {
                rows.push(vec![(m, p, ONE)]);
            }
        }
        super::gpu::GpuTermSpec {
            n_l: self.k,
            l_stack: &self.l_stack,
            n_r: self.k,
            r_stack: &self.r_stack,
            rows,
        }
    }
}

/// One cache-sized tile of `ck ≤ MPO_MODE_TILE` PWE boundary modes inside
/// [`BatchedMpoOp`], storing only non-zero left/right channel blocks and a
/// pre-flattened instruction list so block-sparse symmetric MPOs skip empty
/// charge channels with zero branching in the hot matvec loop.
struct MpoModeChunk {
    /// Compact active `(m_rel, a)` left blocks stacked row-major: `(n_l * l) × l`.
    l_stack: Vec<C64>,
    n_l: usize,
    /// Compact active `(m_rel, c)` right blocks stacked row-major: `(n_r * r) × r`.
    r_stack: Vec<C64>,
    n_r: usize,
    /// Pre-flattened non-zero mix instructions `(sl, p_in, sr, p_out, coeff)`.
    mix: Vec<(usize, usize, usize, usize, C64)>,
    #[cfg(feature = "gpu")]
    ck: usize,
    #[cfg(feature = "gpu")]
    l_slot: Vec<Option<usize>>,
    #[cfg(feature = "gpu")]
    r_slot: Vec<Option<usize>>,
}

const MPO_MODE_TILE: usize = 8;

/// Batched MPO effective operator over all `k` PWE boundary modes, tiled into
/// cache-resident chunks of `≤ 8` modes that run in parallel across cores with
/// automatic skipping of all-zero `(mode, channel)` blocks.
struct BatchedMpoOp {
    l: usize,
    r: usize,
    phys: usize,
    k: usize,
    lb: usize,
    #[cfg(feature = "gpu")]
    rb: usize,
    chunks: Vec<MpoModeChunk>,
    #[cfg(feature = "gpu")]
    rows: Rows,
}

fn mat_is_nonzero(m: &Mat<C64>) -> bool {
    for i in 0..m.nrows() {
        for j in 0..m.ncols() {
            if m[(i, j)] != ZERO {
                return true;
            }
        }
    }
    false
}

impl BatchedMpoOp {
    fn new(lefts: &[Env], rights: &[Env], rows: Rows, l: usize, r: usize, phys: usize) -> Self {
        let k = lefts.len();
        let lb = lefts[0].blocks.len();
        let rb = rights[0].blocks.len();
        let mut chunks = Vec::with_capacity(k.div_ceil(MPO_MODE_TILE));
        for start in (0..k).step_by(MPO_MODE_TILE) {
            let end = (start + MPO_MODE_TILE).min(k);
            let ck = end - start;
            let mut l_slot = vec![None; ck * lb];
            let mut l_stack = Vec::new();
            let mut n_l = 0;
            for (mr, env) in lefts[start..end].iter().enumerate() {
                for (a, blk) in env.blocks.iter().enumerate() {
                    if mat_is_nonzero(blk) {
                        l_slot[mr * lb + a] = Some(n_l);
                        n_l += 1;
                        for i in 0..l {
                            for j in 0..l {
                                l_stack.push(blk[(i, j)]);
                            }
                        }
                    }
                }
            }
            let mut r_slot = vec![None; ck * rb];
            let mut r_stack = Vec::new();
            let mut n_r = 0;
            for (mr, env) in rights[start..end].iter().enumerate() {
                for (c, blk) in env.blocks.iter().enumerate() {
                    if mat_is_nonzero(blk) {
                        r_slot[mr * rb + c] = Some(n_r);
                        n_r += 1;
                        for i in 0..r {
                            for j in 0..r {
                                r_stack.push(blk[(j, i)]); // R^T
                            }
                        }
                    }
                }
            }
            if n_l > 0 && n_r > 0 {
                let mut mix = Vec::new();
                for mr in 0..ck {
                    let l_base = mr * lb;
                    let r_base = mr * rb;
                    for (key, terms) in rows.iter().enumerate() {
                        if terms.is_empty() {
                            continue;
                        }
                        let (c, p_out) = (key / phys, key % phys);
                        let Some(sr) = r_slot[r_base + c] else {
                            continue;
                        };
                        for &(a, p_in, f) in terms {
                            let Some(sl) = l_slot[l_base + a] else {
                                continue;
                            };
                            mix.push((sl, p_in, sr, p_out, f));
                        }
                    }
                }
                if !mix.is_empty() {
                    chunks.push(MpoModeChunk {
                        l_stack,
                        n_l,
                        r_stack,
                        n_r,
                        mix,
                        #[cfg(feature = "gpu")]
                        ck,
                        #[cfg(feature = "gpu")]
                        l_slot,
                        #[cfg(feature = "gpu")]
                        r_slot,
                    });
                }
            }
        }
        BatchedMpoOp {
            l,
            r,
            phys,
            k,
            lb,
            #[cfg(feature = "gpu")]
            rb,
            chunks,
            #[cfg(feature = "gpu")]
            rows,
        }
    }

    fn apply_chunk(&self, ch: &MpoModeChunk, x: &[C64], out: &mut [C64]) {
        let (l, r, phys) = (self.l, self.r, self.phys);
        let pr = phys * r;
        let y_len = ch.n_l * l * pr;
        let nr_r = ch.n_r * r;
        let z_len = l * phys * nr_r;
        let neg_one = C64::new(-1.0, 0.0);
        LOCAL_SCRATCH.with(|cell| {
            let mut sc = cell.borrow_mut();
            ensure_len(&mut sc.y, y_len);
            ensure_len(&mut sc.z, z_len);
            let LocalScratch { y, z } = &mut *sc;
            // Step 1: Y[(slot_l, i), (p', j)] = Σ_u L_slot[i, u] X[u, (p', j)]
            faer_matmul(
                MatMut::from_row_major_slice_mut(&mut y[..y_len], ch.n_l * l, pr),
                Accum::Replace,
                MatRef::from_row_major_slice(&ch.l_stack, ch.n_l * l, l),
                MatRef::from_row_major_slice(x, l, pr),
                ONE,
                Par::Seq,
            );
            // Step 2: Z[(i, p_out), (slot_r, j)] = Σ W[a, c](p_out, p_in) Y[(slot_l, i), (p_in, j)]
            let z_slice = &mut z[..z_len];
            z_slice.fill(ZERO);
            let y_slice = &y[..y_len];
            for &(sl, p_in, sr, p_out, f) in &ch.mix {
                let y_a_off = sl * l * pr + p_in * r;
                let z_col = sr * r;
                if f == ONE {
                    for i in 0..l {
                        let y_src = &y_slice[y_a_off + i * pr..][..r];
                        let z_dst = &mut z_slice[(i * phys + p_out) * nr_r + z_col..][..r];
                        for (zv, &yv) in z_dst.iter_mut().zip(y_src) {
                            *zv += yv;
                        }
                    }
                } else if f == neg_one {
                    for i in 0..l {
                        let y_src = &y_slice[y_a_off + i * pr..][..r];
                        let z_dst = &mut z_slice[(i * phys + p_out) * nr_r + z_col..][..r];
                        for (zv, &yv) in z_dst.iter_mut().zip(y_src) {
                            *zv -= yv;
                        }
                    }
                } else {
                    for i in 0..l {
                        let y_src = &y_slice[y_a_off + i * pr..][..r];
                        let z_dst = &mut z_slice[(i * phys + p_out) * nr_r + z_col..][..r];
                        for (zv, &yv) in z_dst.iter_mut().zip(y_src) {
                            *zv += f * yv;
                        }
                    }
                }
            }
            // Step 3: out[(i, p_out), j'] += Σ_{slot_r, j} Z[(i, p_out), (slot_r, j)] R_slot^T[j, j']
            faer_matmul(
                MatMut::from_row_major_slice_mut(out, l * phys, r),
                Accum::Add,
                MatRef::from_row_major_slice(z_slice, l * phys, nr_r),
                MatRef::from_row_major_slice(&ch.r_stack, nr_r, r),
                ONE,
                Par::Seq,
            );
        });
    }

    fn apply_add(&self, x: &[C64], out: &mut [C64]) {
        let (l, r, phys) = (self.l, self.r, self.phys);
        if self.chunks.is_empty() || l == 0 || r == 0 {
            return;
        }
        let work = self.k * self.lb * l * l * phys * r;
        if self.chunks.len() >= 2 && work >= 1 << 22 {
            let partials: Vec<Vec<C64>> = self
                .chunks
                .par_iter()
                .map(|ch| {
                    let mut local = vec![ZERO; l * phys * r];
                    self.apply_chunk(ch, x, &mut local);
                    local
                })
                .collect();
            for p in partials {
                for (o, &v) in out.iter_mut().zip(&p) {
                    *o += v;
                }
            }
        } else {
            for ch in &self.chunks {
                self.apply_chunk(ch, x, out);
            }
        }
    }

    #[cfg(feature = "gpu")]
    fn chunk_gpu_spec<'a>(&self, ch: &'a MpoModeChunk) -> super::gpu::GpuTermSpec<'a> {
        let mut rows: Vec<Vec<(usize, usize, C64)>> = vec![Vec::new(); ch.n_r * self.phys];
        for mr in 0..ch.ck {
            let l_base = mr * self.lb;
            let r_base = mr * self.rb;
            for (key, terms) in self.rows.iter().enumerate() {
                if terms.is_empty() {
                    continue;
                }
                let (c, p_out) = (key / self.phys, key % self.phys);
                let Some(sr) = ch.r_slot[r_base + c] else {
                    continue;
                };
                let dst = &mut rows[sr * self.phys + p_out];
                for &(a, p_in, f) in terms {
                    let Some(sl) = ch.l_slot[l_base + a] else {
                        continue;
                    };
                    dst.push((sl, p_in, f));
                }
            }
        }
        super::gpu::GpuTermSpec {
            n_l: ch.n_l,
            l_stack: &ch.l_stack,
            n_r: ch.n_r,
            r_stack: &ch.r_stack,
            rows,
        }
    }
}

/// Local effective metric operator `Ñ x = Σ_μ L̃_μ x R̃_μ^T`.
pub(super) struct LocalNormOp {
    stacked: StackedPairOp,
    #[cfg(feature = "gpu")]
    gpu: Option<super::gpu::GpuLocalOp>,
}

impl LinearOperator<C64> for LocalNormOp {
    fn apply(&self, x: &[C64], out: &mut [C64]) {
        #[cfg(feature = "gpu")]
        if let Some(gpu) = &self.gpu
            && gpu.apply(x, out)
        {
            return;
        }
        out.fill(ZERO);
        self.stacked.apply_add(x, out);
    }
}

/// Batched two-GEMM operator `out += Σ_{m=0}^{k-1} (L_m ⊗ O_ops ⊗ R_m^T) x`
/// for a group of crossing terms sharing the same 1-site or 2-site `Local` ops.
struct StackedStringOp {
    l: usize,
    r: usize,
    phys: usize,
    k: usize,
    /// Map `p_out -> Some((p_in, coeff))` for the 1-site or 2-site local string.
    phys_map: Vec<Option<(usize, C64)>>,
    l_stack: Vec<C64>,
    r_stack: Vec<C64>,
}

impl StackedStringOp {
    fn new(
        l: usize,
        r: usize,
        phys: usize,
        ops: Vec<Local>,
        terms: &[(Mat<C64>, Mat<C64>)],
    ) -> Self {
        let mut phys_map = vec![None; phys];
        if ops.len() == 1 {
            for (p_out, slot) in phys_map.iter_mut().enumerate().take(2) {
                *slot = op_entry(ops[0], p_out);
            }
        } else {
            debug_assert_eq!(ops.len(), 2);
            for p1_out in 0..2 {
                let Some((p1_in, f1)) = op_entry(ops[0], p1_out) else {
                    continue;
                };
                for p2_out in 0..2 {
                    let Some((p2_in, f2)) = op_entry(ops[1], p2_out) else {
                        continue;
                    };
                    phys_map[p1_out * 2 + p2_out] = Some((p1_in * 2 + p2_in, f1 * f2));
                }
            }
        }
        let k = terms.len();
        let mut l_stack = Vec::with_capacity(k * l * l);
        let mut r_stack = Vec::with_capacity(k * r * r);
        for (lm, rm) in terms {
            for i in 0..l {
                for j in 0..l {
                    l_stack.push(lm[(i, j)]);
                }
            }
            for i in 0..r {
                for j in 0..r {
                    r_stack.push(rm[(j, i)]);
                }
            }
        }
        StackedStringOp {
            l,
            r,
            phys,
            k,
            phys_map,
            l_stack,
            r_stack,
        }
    }

    fn apply_add(&self, x: &[C64], out: &mut [C64]) {
        let (l, r, phys, k) = (self.l, self.r, self.phys, self.k);
        if k == 0 || l == 0 || r == 0 {
            return;
        }
        let y_len = k * l * phys * r;
        let kr = k * r;
        let pr = phys * r;
        let par = if k * l * l * pr >= 1 << 26 {
            faer::get_global_parallelism()
        } else {
            Par::Seq
        };
        LOCAL_SCRATCH.with(|cell| {
            let mut sc = cell.borrow_mut();
            ensure_len(&mut sc.y, y_len);
            ensure_len(&mut sc.z, y_len);
            let LocalScratch { y, z } = &mut *sc;
            faer_matmul(
                MatMut::from_row_major_slice_mut(&mut y[..y_len], k * l, pr),
                Accum::Replace,
                MatRef::from_row_major_slice(&self.l_stack, k * l, l),
                MatRef::from_row_major_slice(x, l, pr),
                ONE,
                par,
            );
            let z_slice = &mut z[..y_len];
            z_slice.fill(ZERO);
            for (p_out, entry) in self.phys_map.iter().enumerate() {
                let Some((p_in, f)) = *entry else { continue };
                for m in 0..k {
                    let m_off = m * l * pr + p_in * r;
                    let dst_m = m * r;
                    for i in 0..l {
                        let src = &y[m_off + i * pr..][..r];
                        let dst = &mut z_slice[(i * phys + p_out) * kr + dst_m..][..r];
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
            faer_matmul(
                MatMut::from_row_major_slice_mut(out, l * phys, r),
                Accum::Add,
                MatRef::from_row_major_slice(z_slice, l * phys, kr),
                MatRef::from_row_major_slice(&self.r_stack, kr, r),
                ONE,
                par,
            );
        });
    }

    #[cfg(feature = "gpu")]
    fn gpu_spec(&self) -> super::gpu::GpuTermSpec<'_> {
        let mut rows = Vec::with_capacity(self.k * self.phys);
        for m in 0..self.k {
            for &entry in self.phys_map.iter().take(self.phys) {
                if let Some((p_in, f)) = entry {
                    rows.push(vec![(m, p_in, f)]);
                } else {
                    rows.push(Vec::new());
                }
            }
        }
        super::gpu::GpuTermSpec {
            n_l: self.k,
            l_stack: &self.l_stack,
            n_r: self.k,
            r_stack: &self.r_stack,
            rows,
        }
    }
}

/// Local effective Hamiltonian operator `H̃ = H̃_W + H̃_opp + H̃_cross`.
pub(super) struct LocalHamOp {
    l_eff: usize,
    r_eff: usize,
    mpo_ops: Vec<BatchedMpoOp>,
    scalar_op: StackedPairOp,
    string_ops: Vec<StackedStringOp>,
    #[cfg(feature = "gpu")]
    gpu: Option<super::gpu::GpuLocalOp>,
}

impl LinearOperator<C64> for LocalHamOp {
    fn apply(&self, x: &[C64], out: &mut [C64]) {
        if self.l_eff == 0 || self.r_eff == 0 {
            out.fill(ZERO);
            return;
        }
        #[cfg(feature = "gpu")]
        if let Some(gpu) = &self.gpu
            && gpu.apply(x, out)
        {
            return;
        }
        out.fill(ZERO);
        for op in &self.mpo_ops {
            op.apply_add(x, out);
        }
        self.scalar_op.apply_add(x, out);
        for op in &self.string_ops {
            op.apply_add(x, out);
        }
    }
}

/// Batched left-to-right transfer of a list of `Mat<C64>` through one site.
fn batch_mats_right(mats: &[Mat<C64>], a: &Tensor, op: Local) -> Vec<Mat<C64>> {
    let k = mats.len();
    if k == 0 {
        return Vec::new();
    }
    let l = a.left;
    let r = a.right;
    let mut flat = Vec::with_capacity(k * l * l);
    for m in mats {
        for i in 0..l {
            for j in 0..l {
                flat.push(m[(i, j)]);
            }
        }
    }
    let next = batch_transfer_right(&flat, k, a, a, op);
    (0..k)
        .map(|idx| Mat::from_fn(r, r, |i, j| next[(idx * r + i) * r + j]))
        .collect()
}

/// Batched right-to-left transfer of a list of `Mat<C64>` through one site.
fn batch_mats_left(mats: &[Mat<C64>], a: &Tensor, op: Local) -> Vec<Mat<C64>> {
    let k = mats.len();
    if k == 0 {
        return Vec::new();
    }
    let l = a.left;
    let r = a.right;
    let mut flat = Vec::with_capacity(k * r * r);
    for m in mats {
        for i in 0..r {
            for j in 0..r {
                flat.push(m[(i, j)]);
            }
        }
    }
    let next = batch_transfer_left(&flat, k, a, a, op);
    (0..k)
        .map(|idx| Mat::from_fn(l, l, |i, j| next[(idx * l + i) * l + j]))
        .collect()
}

/// Exact factorized QR + thin-SVD re-compression of `Σ_{m=0}^{k-1} L_m ⊗ R_m`.
/// Collapses redundant boundary/crossing pairs down to minimal rank `≤ min(l², r², k)`
/// in `O(k² (l² + r²))` without forming an `l² × r²` matrix.
fn compress_lr_pairs(
    l: usize,
    r: usize,
    terms: Vec<(Mat<C64>, Mat<C64>)>,
    tol: f64,
) -> Vec<(Mat<C64>, Mat<C64>)> {
    let k = terms.len();
    if k <= 4 || l == 0 || r == 0 {
        return terms;
    }
    let l2 = l * l;
    let r2 = r * r;
    let mut a_mat = Mat::<C64>::zeros(l2, k);
    let mut b_mat = Mat::<C64>::zeros(r2, k);
    for (m, (lm, rm)) in terms.iter().enumerate() {
        for i in 0..l {
            for j in 0..l {
                a_mat[(i * l + j, m)] = lm[(i, j)];
            }
        }
        for i in 0..r {
            for j in 0..r {
                b_mat[(i * r + j, m)] = rm[(i, j)];
            }
        }
    }
    let (q_a, r_a) = thin_qr(a_mat.as_ref());
    let (q_b, r_b) = thin_qr(b_mat.as_ref());

    // C = R_A R_B^T  (transpose, not adjoint: Σ_m A[:,m] B[:,m]^T = Q_A C Q_B^T).
    let core = matmul(r_a.as_ref(), r_b.transpose());
    let (u, s, v) = crate::mps::linalg::thin_svd(core.as_ref());
    if s.is_empty() || s[0] == 0.0 {
        return Vec::new();
    }
    let thresh = tol * s[0];
    let keep = s.iter().take_while(|&&sv| sv > thresh).count();
    if keep == 0 {
        return Vec::new();
    }
    if keep >= k {
        return terms;
    }
    let l_new = matmul(q_a.as_ref(), u.as_ref().subcols(0, keep));
    let v_conj = Mat::<C64>::from_fn(v.nrows(), keep, |row, col| v[(row, col)].conj());
    let r_new = matmul(q_b.as_ref(), v_conj.as_ref());
    let mut out = Vec::with_capacity(keep);
    for mu in 0..keep {
        let rt = s[mu].sqrt();
        let lm = Mat::<C64>::from_fn(l, l, |i, j| l_new[(i * l + j, mu)] * rt);
        let rm = Mat::<C64>::from_fn(r, r, |i, j| r_new[(i * r + j, mu)] * rt);
        out.push((lm, rm));
    }
    out
}

/// Dense Hermitian PSD gauge factorisation `G = U Λ^{-1/2}` and `G^{-1} = Λ^{1/2} U†`,
/// dropping eigenvalues `<= tol * λ_max`.
pub(super) fn dense_metric_gauge(m: &Mat<C64>, tol: f64) -> (Mat<C64>, Mat<C64>) {
    let d = m.nrows();
    if d == 0 {
        return (Mat::zeros(0, 0), Mat::zeros(0, 0));
    }
    let sym = Mat::<C64>::from_fn(d, d, |i, j| (m[(i, j)] + m[(j, i)].conj()) * 0.5);
    let (raw_vals, u) = hermitian_evd(sym.as_ref());
    let vals: Vec<f64> = raw_vals.into_iter().map(|v| v.max(0.0)).collect();
    let max_v = vals.iter().copied().fold(0.0f64, f64::max);
    let cutoff = tol * max_v;
    let mut kept: Vec<usize> = vals
        .iter()
        .enumerate()
        .filter(|&(_, &v)| v > cutoff)
        .map(|(k, _)| k)
        .collect();
    if kept.is_empty() {
        kept.push(d - 1);
    }
    let k = kept.len();
    let mut g = Mat::<C64>::zeros(d, k);
    let mut g_inv = Mat::<C64>::zeros(k, d);
    for (col, &idx) in kept.iter().enumerate() {
        let lam = vals[idx].max(1e-300);
        let inv_rt = lam.sqrt().recip();
        let rt = lam.sqrt();
        for i in 0..d {
            g[(i, col)] = u[(i, idx)] * inv_rt;
            g_inv[(col, i)] = u[(i, idx)].conj() * rt;
        }
    }
    (g, g_inv)
}

fn is_scaled_identity(m: &Mat<C64>, scale: C64, tol: f64) -> bool {
    let ref_norm = scale.norm();
    if ref_norm <= 0.0 || !ref_norm.is_finite() {
        return false;
    }
    let thresh = tol * ref_norm;
    let d = m.nrows();
    for i in 0..d {
        for j in 0..d {
            let want = if i == j { scale } else { ZERO };
            if (m[(i, j)] - want).norm() > thresh {
                return false;
            }
        }
    }
    true
}

/// Perturbation blocks moving right from an MPO site `w` and left env `env`.
fn left_perturbation_local(
    env: &Env,
    w: &crate::mps::mpo::MpoSite,
    theta: &[C64],
    l: usize,
    r: usize,
) -> Vec<Mat<C64>> {
    let theta_m = MatRef::from_row_major_slice(theta, l, 4 * r);
    let y: Vec<Mat<C64>> = env
        .blocks
        .iter()
        .map(|e| matmul(e.as_ref(), theta_m))
        .collect();
    let rows = crate::mps::mpo::right_keyed(w);
    (0..w.right)
        .map(|b| {
            let mut out = Mat::<C64>::zeros(2 * l, 2 * r);
            for p_out in 0..2 {
                for &(a, p_in, f) in &rows[b * 2 + p_out] {
                    for q in 0..2 {
                        for col in 0..r {
                            for row in 0..l {
                                out[(2 * row + p_out, q * r + col)] +=
                                    f * y[a][(row, (2 * p_in + q) * r + col)];
                            }
                        }
                    }
                }
            }
            out
        })
        .collect()
}

fn right_perturbation_local(
    env: &Env,
    w: &crate::mps::mpo::MpoSite,
    theta: &[C64],
    l: usize,
    r: usize,
) -> Vec<Mat<C64>> {
    let theta_m = MatRef::from_row_major_slice(theta, 4 * l, r);
    let z: Vec<Mat<C64>> = env
        .blocks
        .iter()
        .map(|e| matmul(theta_m, e.transpose()))
        .collect();
    let rows = crate::mps::mpo::left_keyed(w);
    (0..w.left)
        .map(|a| {
            let mut out = Mat::<C64>::zeros(2 * l, 2 * r);
            for q_out in 0..2 {
                for &(c, q_in, f) in &rows[a * 2 + q_out] {
                    for col in 0..r {
                        for row in 0..2 * l {
                            out[(row, q_out * r + col)] += f * z[c][(2 * row + q_in, col)];
                        }
                    }
                }
            }
            out
        })
        .collect()
}

fn single_left_perturbation(
    env: &Mat<C64>,
    op: Local,
    theta: &[C64],
    l: usize,
    r: usize,
) -> Mat<C64> {
    let theta_m = MatRef::from_row_major_slice(theta, l, 4 * r);
    let y = matmul(env.as_ref(), theta_m);
    let mut out = Mat::<C64>::zeros(2 * l, 2 * r);
    for p_out in 0..2 {
        let Some((p_in, f)) = op_entry(op, p_out) else {
            continue;
        };
        for q in 0..2 {
            for col in 0..r {
                for row in 0..l {
                    out[(2 * row + p_out, q * r + col)] = f * y[(row, (2 * p_in + q) * r + col)];
                }
            }
        }
    }
    out
}

fn single_right_perturbation(
    env: &Mat<C64>,
    op: Local,
    theta: &[C64],
    l: usize,
    r: usize,
) -> Mat<C64> {
    let theta_m = MatRef::from_row_major_slice(theta, 4 * l, r);
    let z = matmul(theta_m, env.transpose());
    let mut out = Mat::<C64>::zeros(2 * l, 2 * r);
    for q_out in 0..2 {
        let Some((q_in, f)) = op_entry(op, q_out) else {
            continue;
        };
        for col in 0..r {
            for row in 0..2 * l {
                out[(row, q_out * r + col)] = f * z[(2 * row + q_in, col)];
            }
        }
    }
    out
}
