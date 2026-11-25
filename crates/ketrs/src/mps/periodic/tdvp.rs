//! Periodic two-site TDVP (time-dependent variational principle) with
//! Pippan–White–Evertz (2010) `O(p D χ³)` transfer-matrix compression.
//!
//! Each step of time `dt` is a **time-reversible second-order** Strang-split
//! antipodal window projector-splitting integrator:
//!
//! 1. Partition `H = shift·I + H₁ + H₂ (+ H₃ if terms straddle both cuts)`
//!    across overlapping half-ring windows `W₁ = [0 ..= m]` and
//!    `W₂ = [m .. n-1, 0]` (with `(n - 1, 0)` as an interior seam pair of `W₂`).
//!    For any Hamiltonian of range `≤ n/2`, every `H_k` acts strictly inside
//!    its own open window `W_k` (`H_opp = 0`, zero crossing channels!), which
//!    completely eliminates closed-loop tangent-space double counting at any
//!    bond dimension `χ` (including starting from a product state!).
//! 2. Apply the symmetric time-reversible sequence
//!    `Fwd(W₁, H₁, dt/2) · Fwd(W₂, H₂, dt/2) · Bwd(W₂, H₂, dt/2) · Bwd(W₁, H₁, dt/2)`.
//! 3. Every local evolution `i Ñ dx/dt = H̃ x` runs in `O(p D χ³)` under the
//!    exact periodic trace metric `Ñ` (with exact `Y`-metric ALS polishing on
//!    truncated 2-site splits to preserve periodic Hilbert norm and optimal
//!    truncation error).

use super::PeriodicMps;
use super::dmrg::two_site_dense;
use super::window::{WindowEnvs, WindowHamiltonian, split_hamiltonian_on_window};
use crate::error::Error;
use crate::mps::{Absorb, Tensor};
use crate::observable::PauliSum;
use faer::MatRef;
use num_complex::Complex64 as C64;

const GAUGE_TOL: f64 = 1e-13;

impl PeriodicMps {
    /// Evolve by `exp(-iHt)` with `steps` second-order steps of periodic
    /// two-site TDVP.
    ///
    /// Works for periodic Hamiltonians of any range, supports both dense and
    /// symmetric (`with_charges`) periodic MPS, and grows all `n` bonds
    /// (including the `(n - 1, 0)` seam bond) dynamically up to `max_bond`.
    pub fn tdvp(&mut self, hamiltonian: &PauliSum, time: f64, steps: usize) -> Result<(), Error> {
        let n = self.n_qubits();
        if n < 2 {
            return Err(Error::InvalidArgument(
                "periodic TDVP needs at least two qubits".into(),
            ));
        }
        if steps == 0 {
            return Err(Error::InvalidArgument(
                "periodic TDVP needs at least one step".into(),
            ));
        }
        if hamiltonian.min_qubits() > n {
            return Err(Error::QubitOutOfRange {
                qubit: hamiltonian.min_qubits() - 1,
                n_qubits: n,
            });
        }

        // Validate symmetry up front and extract global identity shift.
        let shift = {
            let (s, _) = crate::mps::mpo::Mpo::elementary_terms(
                hamiltonian,
                n,
                self.symmetry.as_ref().map(|sym| sym.charges.as_ref()),
            )?;
            s
        };

        let dt = time / steps as f64;
        let null_tol = (10.0 * self.transfer_cutoff).max(1e-12);

        if n == 2 {
            // Single 2-site periodic ring: exact single-window evolution.
            let wh = split_hamiltonian_on_window(
                hamiltonian,
                vec![0, 1],
                vec![],
                n,
                self.symmetry.as_ref().map(|s| s.charges.as_ref()),
            )?;
            #[cfg(feature = "gpu")]
            let gpu_scratch = self.gpu().map(|ctx| {
                (
                    ctx.clone(),
                    std::sync::Arc::new(std::sync::Mutex::new(
                        super::gpu::GpuPeriodicScratch::default(),
                    )),
                )
            });
            for _ in 0..steps {
                #[allow(unused_mut)]
                let mut envs = WindowEnvs::new(
                    &self.tensors,
                    &wh,
                    self.transfer_compression(),
                    self.symmetry.as_ref(),
                );
                #[cfg(feature = "gpu")]
                if let Some((ctx, scratch)) = &gpu_scratch {
                    envs = envs.with_gpu(ctx, scratch);
                }
                let safe_tol = envs.safe_null_tol(null_tol);
                self.tdvp_evolve_pair(0, &envs, dt, Absorb::Right, safe_tol)?;
            }
        } else {
            let windows = partition_hamiltonian_windows(
                hamiltonian,
                n,
                self.symmetry.as_ref().map(|s| s.charges.as_ref()),
            )?;
            #[cfg(feature = "gpu")]
            let gpu_scratch = self.gpu().map(|ctx| {
                (
                    ctx.clone(),
                    std::sync::Arc::new(std::sync::Mutex::new(
                        super::gpu::GpuPeriodicScratch::default(),
                    )),
                )
            });
            let half = dt / 2.0;
            // Track per-(window, direction) bond dimensions at `wh.gse_bonds`
            // when the last GSE call added 0 directions so converged / sector-capped
            // windows don't re-run GSE every step.
            let mut gse_done: Vec<[Option<Vec<usize>>; 2]> = vec![[None, None]; windows.len()];
            for _step in 0..steps {
                for (w_idx, wh) in windows.iter().enumerate() {
                    #[cfg(feature = "gpu")]
                    let g = gpu_scratch.as_ref();
                    #[cfg(not(feature = "gpu"))]
                    let g = None::<&()>;
                    self.tdvp_window_forward(wh, half, null_tol, &mut gse_done[w_idx][0], g)?;
                }
                for (w_idx, wh) in windows.iter().enumerate().rev() {
                    #[cfg(feature = "gpu")]
                    let g = gpu_scratch.as_ref();
                    #[cfg(not(feature = "gpu"))]
                    let g = None::<&()>;
                    self.tdvp_window_backward(wh, half, null_tol, &mut gse_done[w_idx][1], g)?;
                }
            }
        }

        if shift != 0.0 {
            let phase = C64::cis(-shift * time);
            self.tensors[0].data.iter_mut().for_each(|a| *a *= phase);
        }
        Ok(())
    }

    /// Forward projector-splitting pass of time `dt` on a single window `wh`.
    fn tdvp_window_forward(
        &mut self,
        wh: &WindowHamiltonian,
        dt: f64,
        null_tol: f64,
        gse_done_bonds: &mut Option<Vec<usize>>,
        #[cfg(feature = "gpu")] gpu: Option<&(
            std::sync::Arc<crate::gpu::GpuContext>,
            std::sync::Arc<std::sync::Mutex<super::gpu::GpuPeriodicScratch>>,
        )>,
        #[cfg(not(feature = "gpu"))] _gpu: Option<&()>,
    ) -> Result<(), Error> {
        let l = wh.win_sites.len();
        if l < 2 {
            return Ok(());
        }
        let comp = self.transfer_compression();
        self.canonicalise_opp_right(&wh.opp_sites, wh.win_sites[0]);
        self.canonicalise_window_left(&wh.win_sites);
        if l >= 3
            && !wh.gse_bonds.is_empty()
            && let Some(mpo) = &wh.mpo_w
            && wh
                .gse_bonds
                .iter()
                .any(|&pos| self.tensors[wh.win_sites[pos]].left < self.options.max_bond)
        {
            let cur_bonds: Vec<usize> = wh
                .gse_bonds
                .iter()
                .map(|&pos| self.tensors[wh.win_sites[pos]].left)
                .collect();
            if gse_done_bonds.as_ref() != Some(&cur_bonds) {
                let added = self.expand_window_right(&wh.win_sites, mpo);
                if added == 0 {
                    *gse_done_bonds = Some(cur_bonds);
                } else {
                    *gse_done_bonds = None;
                }
            }
        }
        let mut envs = WindowEnvs::new(&self.tensors, wh, comp, self.symmetry.as_ref());
        #[cfg(feature = "gpu")]
        if let Some((ctx, scratch)) = gpu {
            envs = envs.with_gpu(ctx, scratch);
        }
        let safe = envs.safe_null_tol(null_tol);
        envs.build_all_right(&self.tensors);

        for pos in 0..l - 1 {
            self.tdvp_evolve_pair(pos, &envs, dt, Absorb::Right, safe)?;
            envs.update_left(pos, &self.tensors);
            if pos + 1 < l - 1 {
                self.tdvp_evolve_site(pos + 1, &envs, -dt, safe)?;
            }
        }
        Ok(())
    }

    /// Exact time-reversed backward projector-splitting pass of time `dt` on `wh`.
    fn tdvp_window_backward(
        &mut self,
        wh: &WindowHamiltonian,
        dt: f64,
        null_tol: f64,
        gse_done_bonds: &mut Option<Vec<usize>>,
        #[cfg(feature = "gpu")] gpu: Option<&(
            std::sync::Arc<crate::gpu::GpuContext>,
            std::sync::Arc<std::sync::Mutex<super::gpu::GpuPeriodicScratch>>,
        )>,
        #[cfg(not(feature = "gpu"))] _gpu: Option<&()>,
    ) -> Result<(), Error> {
        let l = wh.win_sites.len();
        if l < 2 {
            return Ok(());
        }
        let comp = self.transfer_compression();
        self.canonicalise_opp_left(&wh.opp_sites, *wh.win_sites.last().unwrap());
        self.canonicalise_window_right(&wh.win_sites);
        if l >= 3
            && !wh.gse_bonds.is_empty()
            && let Some(mpo) = &wh.mpo_w
            && wh
                .gse_bonds
                .iter()
                .any(|&pos| self.tensors[wh.win_sites[pos - 1]].right < self.options.max_bond)
        {
            let cur_bonds: Vec<usize> = wh
                .gse_bonds
                .iter()
                .map(|&pos| self.tensors[wh.win_sites[pos - 1]].right)
                .collect();
            if gse_done_bonds.as_ref() != Some(&cur_bonds) {
                let added = self.expand_window_left(&wh.win_sites, mpo);
                if added == 0 {
                    *gse_done_bonds = Some(cur_bonds);
                } else {
                    *gse_done_bonds = None;
                }
            }
        }
        let mut envs = WindowEnvs::new(&self.tensors, wh, comp, self.symmetry.as_ref());
        #[cfg(feature = "gpu")]
        if let Some((ctx, scratch)) = gpu {
            envs = envs.with_gpu(ctx, scratch);
        }
        let safe = envs.safe_null_tol(null_tol);
        envs.build_all_left(&self.tensors);

        for pos in (0..l - 1).rev() {
            self.tdvp_evolve_pair(pos, &envs, dt, Absorb::Left, safe)?;
            envs.update_right(pos, &self.tensors);
            if pos > 0 {
                self.tdvp_evolve_site(pos, &envs, -dt, safe)?;
            }
        }
        Ok(())
    }

    fn tdvp_evolve_pair(
        &mut self,
        pos: usize,
        envs: &WindowEnvs<'_>,
        dt: f64,
        absorb: Absorb,
        null_tol: f64,
    ) -> Result<(), Error> {
        let q1 = envs.win_sites[pos];
        let q2 = envs.win_sites[pos + 1];
        let prob = envs.two_site_problem(pos, &self.tensors, self.symmetry.as_ref(), GAUGE_TOL);
        let theta_cur = two_site_dense(&self.tensors[q1], &self.tensors[q2]);
        let mut x = prob.to_solver_vec(&theta_cur);
        let proj = prob.evolve_local(&mut x, dt, null_tol)?;
        let theta_tilde_vec = prob.unpack_gauge(&x);
        let theta_tilde =
            MatRef::from_row_major_slice(&theta_tilde_vec, prob.l_eff * 2, 2 * prob.r_eff);
        let discarded = self.split_window_pair(q1, q2, &prob, theta_tilde, absorb);
        if discarded > 0.0 {
            if proj.is_some() {
                let (mut a_pol, mut b_pol) = (self.tensors[q1].clone(), self.tensors[q2].clone());
                if prob.n_norm_polish_pair(
                    &x,
                    &mut a_pol,
                    &mut b_pol,
                    self.symmetry.as_ref(),
                    q1,
                    q2,
                    absorb,
                    proj.as_ref(),
                    null_tol,
                ) {
                    self.tensors[q1] = a_pol;
                    self.tensors[q2] = b_pol;
                }
            }
            // Preserve exact periodic Hilbert norm `⟨ψ|ψ⟩` across truncated splits
            // by rescaling the absorbed site tensor (leaving the other site strictly
            // left-/right-orthonormal).
            let scale = if prob.n_is_identity || envs.norm_trunc_err > 1e-2 {
                let kept = (1.0 - discarded).max(0.0);
                if kept > 1e-30 {
                    kept.sqrt().recip()
                } else {
                    1.0
                }
            } else {
                use crate::statevector::krylov::{LinearOperator, dot};
                let n_op = prob.norm_linear_op();
                let mut nx = vec![crate::mps::ZERO; x.len()];
                n_op.apply(&x, &mut nx);
                let n_before = dot(&x, &nx).re;
                let ta = two_site_dense(&self.tensors[q1], &self.tensors[q2]);
                let va = prob.to_solver_vec(&ta);
                let mut nva = vec![crate::mps::ZERO; va.len()];
                n_op.apply(&va, &mut nva);
                let n_after = dot(&va, &nva).re;
                if n_before > 1e-30 && n_after > 1e-30 {
                    (n_before / n_after).sqrt().clamp(0.5, 2.0)
                } else {
                    1.0
                }
            };
            if scale.is_finite() && (scale - 1.0).abs() > 1e-13 {
                let target_q = match absorb {
                    Absorb::Right => q2,
                    Absorb::Left => q1,
                };
                self.tensors[target_q]
                    .data
                    .iter_mut()
                    .for_each(|a| *a *= scale);
            }
        }
        Ok(())
    }

    fn tdvp_evolve_site(
        &mut self,
        pos: usize,
        envs: &WindowEnvs<'_>,
        dt: f64,
        null_tol: f64,
    ) -> Result<(), Error> {
        let q = envs.win_sites[pos];
        let prob = envs.one_site_problem(pos, &self.tensors, self.symmetry.as_ref(), GAUGE_TOL);
        let mut x = prob.to_solver_vec(&self.tensors[q].data);
        prob.evolve_local(&mut x, dt, null_tol)?;
        let gauge_vec = prob.unpack_gauge(&x);
        let full_vec = prob.gauge_to_full(&gauge_vec);
        self.tensors[q] = Tensor {
            left: prob.l_full,
            right: prob.r_full,
            data: full_vec,
        };
        Ok(())
    }
}

/// Shortest cyclic arc `(start, span)` on a ring of `n` qubits containing all
/// sorted distinct sites in `qubits` (`2 <= span <= n`).
pub(super) fn shortest_cyclic_arc(qubits: &[usize], n: usize) -> (usize, usize) {
    debug_assert!(!qubits.is_empty());
    let k = qubits.len();
    if k == 1 {
        return (qubits[0], 2.min(n));
    }
    let mut best_gap = n - qubits[k - 1] + qubits[0];
    let mut best_start = qubits[0];
    for i in 0..k - 1 {
        let gap = qubits[i + 1] - qubits[i];
        if gap > best_gap {
            best_gap = gap;
            best_start = qubits[i + 1];
        }
    }
    let span = (n - best_gap + 1).max(2);
    (best_start, span)
}

/// Cyclic window `win` of length `span` starting at `start`, and its
/// complement `opp` of length `n - span`.
pub(super) fn cyclic_window(start: usize, span: usize, n: usize) -> (Vec<usize>, Vec<usize>) {
    let win = (0..span).map(|k| (start + k) % n).collect();
    let opp = (span..n).map(|k| (start + k) % n).collect();
    (win, opp)
}

/// Partition `hamiltonian` (excluding global identity shift) across overlapping
/// cyclic windows so **every** term acts strictly inside its assigned window
/// (`H_opp = 0`, zero crossing channels for any range, including multi-qubit
/// wrap-around and full-ring terms).
fn partition_hamiltonian_windows(
    hamiltonian: &PauliSum,
    n: usize,
    charges: Option<&crate::mps::charges::Charges>,
) -> Result<Vec<WindowHamiltonian>, Error> {
    let in_set = |sites: &[usize], ps: &crate::observable::PauliString| -> bool {
        ps.ops().iter().all(|&(q, _)| sites.contains(&q))
    };

    if n == 3 {
        let base = [
            (vec![0, 1], vec![2]),
            (vec![1, 2], vec![0]),
            (vec![2, 0], vec![1]),
        ];
        let mut h_base = [PauliSum::new(), PauliSum::new(), PauliSum::new()];
        let mut h_full = PauliSum::new();
        for &(coeff, ref ps) in hamiltonian.terms() {
            if ps.ops().is_empty() || coeff == 0.0 {
                continue;
            }
            let matching: Vec<usize> = (0..3).filter(|&k| in_set(&base[k].0, ps)).collect();
            if matching.is_empty() {
                h_full.add(coeff, ps.clone());
            } else {
                let share = coeff / matching.len() as f64;
                for k in matching {
                    h_base[k].add(share, ps.clone());
                }
            }
        }
        let mut out = Vec::with_capacity(4);
        for ((w, o), h) in base.into_iter().zip(h_base) {
            if !h.terms().is_empty() {
                out.push(split_hamiltonian_on_window(&h, w, o, n, charges)?);
            }
        }
        if !h_full.terms().is_empty() {
            out.push(split_hamiltonian_on_window(
                &h_full,
                vec![0, 1, 2],
                vec![],
                n,
                charges,
            )?);
        }
        return Ok(out);
    }

    let m = n / 2;
    // Window 1: [0, 1, ..., m]
    let w1: Vec<usize> = (0..=m).collect();
    // Window 2: [m, m+1, ..., n-1, 0]
    let mut w2: Vec<usize> = (m..n).collect();
    w2.push(0);
    // Window 3: shifted half-ring centred near `m`.
    let q0 = m / 2;
    let (w3, _) = cyclic_window(q0, m + 1, n);
    // Window 4: shifted half-ring centred near seam `0` (covers wrap-around terms).
    let (w4, _) = cyclic_window((q0 + m) % n, m + 1, n);

    let mut h1 = PauliSum::new();
    let mut h2 = PauliSum::new();
    let mut h3 = PauliSum::new();
    let mut h4 = PauliSum::new();
    // Leftover terms (span > m + 1 or straddling all 4 base windows), grouped by
    // exact qubit support so symmetry-partner terms stay together.
    let mut leftover: std::collections::BTreeMap<
        Vec<usize>,
        Vec<(f64, crate::observable::PauliString)>,
    > = std::collections::BTreeMap::new();

    for &(coeff, ref ps) in hamiltonian.terms() {
        if ps.ops().is_empty() || coeff == 0.0 {
            continue;
        }
        let in1 = in_set(&w1, ps);
        let in2 = in_set(&w2, ps);
        if in1 && in2 {
            // Single-site or `{0, m}` term shared by both windows: split evenly for symmetry.
            h1.add(0.5 * coeff, ps.clone());
            h2.add(0.5 * coeff, ps.clone());
        } else if in1 {
            h1.add(coeff, ps.clone());
        } else if in2 {
            h2.add(coeff, ps.clone());
        } else if in_set(&w3, ps) {
            h3.add(coeff, ps.clone());
        } else if in_set(&w4, ps) {
            h4.add(coeff, ps.clone());
        } else {
            let support: Vec<usize> = ps.ops().iter().map(|&(q, _)| q).collect();
            leftover
                .entry(support)
                .or_default()
                .push((coeff, ps.clone()));
        }
    }

    let mut out = Vec::with_capacity(4);
    for h in [h1, h2, h3, h4] {
        if !h.terms().is_empty() {
            // Shrink each window to the minimal cyclic arc covering the qubits
            // actually touched by `h` (at least 2 sites), eliminating redundant
            // sweeps over untouched sites (especially for seam-straddling W3/W4).
            let mut touched: Vec<usize> = h
                .terms()
                .iter()
                .flat_map(|(_, ps)| ps.ops().iter().map(|&(q, _)| q))
                .collect();
            touched.sort_unstable();
            touched.dedup();
            let (start, span) = shortest_cyclic_arc(&touched, n);
            let (w, o) = cyclic_window(start, span, n);
            let wh = split_hamiltonian_on_window(&h, w, o, n, charges)?;
            debug_assert!(wh.mpo_opp.is_none() && wh.cross.is_empty());
            out.push(wh);
        }
    }

    if !leftover.is_empty() {
        // Pack leftover supports into minimal cyclic windows (largest span first
        // so smaller supports reuse compatible windows).
        type TermGroup = (
            usize,
            usize,
            Vec<usize>,
            Vec<(f64, crate::observable::PauliString)>,
        );
        let mut groups: Vec<TermGroup> = leftover
            .into_iter()
            .map(|(supp, terms)| {
                let (start, span) = shortest_cyclic_arc(&supp, n);
                (span, start, supp, terms)
            })
            .collect();
        groups.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));

        let mut extra: Vec<(Vec<usize>, Vec<usize>, PauliSum)> = Vec::new();
        for (span, start, supp, terms) in groups {
            let target = extra
                .iter_mut()
                .find(|(w, _, _)| supp.iter().all(|q| w.contains(q)));
            if let Some((_, _, h_ext)) = target {
                for (c, ps) in terms {
                    h_ext.add(c, ps);
                }
            } else {
                let (w, o) = cyclic_window(start, span, n);
                let mut h_ext = PauliSum::new();
                for (c, ps) in terms {
                    h_ext.add(c, ps);
                }
                extra.push((w, o, h_ext));
            }
        }
        for (w, o, h_ext) in extra {
            let wh = split_hamiltonian_on_window(&h_ext, w, o, n, charges)?;
            debug_assert!(wh.mpo_opp.is_none() && wh.cross.is_empty());
            out.push(wh);
        }
    }
    Ok(out)
}
