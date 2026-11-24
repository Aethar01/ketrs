//! Periodic two-site DMRG with Pippan–White–Evertz (2010) `O(χ³)` transfer-matrix
//! compression (and automatic exact `O(χ⁵)` SVD for small `χ` or high accuracy).
//!
//! Each sweep covers the entire ring — including the periodic seam bond
//! `(n - 1, 0)` — using overlapping half-ring windows:
//!
//! - For `n ≥ 4` with `m = n / 2`:
//!   - Window 1: `[0, 1, ..., m]` (optimises bonds `0 .. m - 1`, complement `[m+1 .. n-1]`).
//!   - Window 2: `[m, ..., n - 1, 0]` (optimises bonds `m .. n - 1` including `(n-1, 0)`, complement `[1 .. m-1]`).
//! - For `n == 3`: three overlapping 2-site windows `[0, 1]`, `[1, 2]`, `[2, 0]`.
//! - For `n == 2`: single 2-site window `[0, 1]` with empty complement.
//!
//! Inside each window, sites away from the active pair are kept strictly
//! left-/right-orthonormal by QR, local generalized eigenproblems
//! `H̃ θ̃ = E Ñ θ̃` are solved in gauge-conditioned coordinates, and two-site
//! splits truncate in the metric-weighted norm `‖θ̃‖_F ≈ ‖ψ‖_2` (with optional
//! White subspace-expansion noise so multi-qubit lattice gauge terms never
//! stall from a product state).

use super::PeriodicMps;
use super::symmetric::{qr_left_symmetric, qr_right_symmetric, split_metric_symmetric};
use super::window::{WindowEnvs, split_hamiltonian_on_window};
use crate::error::Error;
use crate::mps::dmrg::{DmrgOptions, DmrgResult, noise_room};
use crate::mps::linalg::{matmul, thin_qr, thin_svd};
use crate::mps::{Absorb, MpsOptions, Tensor, keep_count};
use crate::observable::PauliSum;
use crate::statevector::eigen::EigenOptions;
use crate::statevector::krylov::LinearOperator;
use faer::{Mat, MatRef};
use num_complex::Complex64 as C64;

/// Relative eigenvalue cutoff for single-bond gauge preconditioning `G_L, G_R`.
const GAUGE_TOL: f64 = 1e-11;

impl PeriodicMps {
    /// Find the ground state of `hamiltonian` on a periodic ring by two-site
    /// periodic DMRG, replacing this state in place.
    ///
    /// Uses Pippan–White–Evertz (2010) low-rank compression of the opposite-half
    /// transfer matrices (`O(p D χ³)` per site) with automatic exact SVD for
    /// small `χ` (or when `transfer_rank` is `Some(0)`). Supports both dense
    /// and symmetric (`with_charges`) periodic MPS.
    pub fn dmrg(
        &mut self,
        hamiltonian: &PauliSum,
        options: &DmrgOptions,
    ) -> Result<DmrgResult, Error> {
        let n = self.n_qubits();
        if n < 2 {
            return Err(Error::InvalidArgument(
                "periodic DMRG needs at least two qubits".into(),
            ));
        }
        if hamiltonian.min_qubits() > n {
            return Err(Error::QubitOutOfRange {
                qubit: hamiltonian.min_qubits() - 1,
                n_qubits: n,
            });
        }
        self.set_options(MpsOptions {
            max_bond: options.max_bond,
            cutoff: options.cutoff,
        });
        let lanczos = EigenOptions {
            tolerance: options.lanczos_tolerance,
            max_iterations: 200,
            seed: 0,
        };

        // Split Hamiltonian onto each window ONCE before sweeping (also validates
        // symmetry conservation up front).
        let windows = ring_windows(n);
        let charges = self.symmetry.as_ref().map(|s| s.charges.clone());
        let win_hams: Vec<_> = windows
            .into_iter()
            .map(|(w, o)| split_hamiltonian_on_window(hamiltonian, w, o, n, charges.as_deref()))
            .collect::<Result<Vec<_>, _>>()?;

        self.normalise();

        let mut sweep_energies = Vec::new();
        let mut prev_bonds: Vec<usize> = Vec::new();
        let mut energy = f64::INFINITY;
        let mut converged = false;
        let mut unconverged_solves = 0usize;
        let mut noise = options.noise.max(0.0);
        let comp = self.transfer_compression();
        let base_null_tol = (10.0 * self.transfer_cutoff).max(1e-11);
        #[cfg(feature = "gpu")]
        let gpu_scratch = self.gpu().map(|ctx| {
            (
                ctx.clone(),
                std::sync::Arc::new(std::sync::Mutex::new(
                    super::gpu::GpuPeriodicScratch::default(),
                )),
            )
        });

        for sweep in 0..options.max_sweeps {
            if noise < options.cutoff.max(1e-6) || sweep >= 5 {
                noise = 0.0;
            }
            // Hybrid periodic DMRG (Pippan–White–Evertz 2010):
            // - Early / noisy sweeps use 2-site updates to grow bond dimensions
            //   and Abelian charge sectors dynamically up to `max_bond`.
            // - Once `noise == 0` and bonds have grown (`sweep >= 1`), finish
            //   every window with a 1-site exact variational sweep (zero SVD
            //   truncation, half the local dimension, 100% monotonic to 1e-13!).
            // - Only switch both directions to pure 1-site polishing once bond
            //   dimensions have stabilized and 2-site energy change < 1e-8.
            let prev_change = sweep_energies
                .windows(2)
                .last()
                .map_or(f64::INFINITY, |w: &[f64]| (w[0] - w[1]).abs());
            let cur_bonds = self.bond_dimensions();
            let bonds_stable = !prev_bonds.is_empty() && cur_bonds == prev_bonds;
            let all_at_max = cur_bonds.iter().all(|&b| b >= options.max_bond);
            let pure_one_site = noise == 0.0
                && sweep >= 2
                && (all_at_max || (bonds_stable && prev_change < 1e-6));
            let polish_back = noise == 0.0 && sweep >= 1;
            prev_bonds = cur_bonds;

            for wh in &win_hams {
                // Left-canonicalise `opp_sites` into `win_sites[0]` and
                // right-canonicalise `win_sites` so all window bonds stay
                // well-conditioned without shrinking boundary bonds.
                self.canonicalise_opp_right(&wh.opp_sites, wh.win_sites[0]);
                self.canonicalise_window_left(&wh.win_sites);

                let shift = wh.shift;
                let mut envs = WindowEnvs::new(&self.tensors, wh, comp, self.symmetry.as_ref());
                #[cfg(feature = "gpu")]
                if let Some((ctx, scratch)) = &gpu_scratch {
                    envs = envs.with_gpu(ctx, scratch);
                }
                let null_tol = envs.safe_null_tol(base_null_tol);
                envs.build_all_right(&self.tensors);

                let l = wh.win_sites.len();
                if pure_one_site {
                    // Forward 1-site sweep: pos = 0 .. l - 1.
                    for pos in 0..l - 1 {
                        let (e, ok) = self.optimise_window_site(
                            pos,
                            &envs,
                            &lanczos,
                            Some(Absorb::Right),
                            null_tol,
                        )?;
                        energy = e + shift;
                        unconverged_solves += usize::from(!ok);
                        envs.update_left(pos, &self.tensors);
                    }
                    // Backward 1-site sweep: pos = (l - 1) ..= 0.
                    for pos in (0..l).rev() {
                        let absorb = (pos > 0).then_some(Absorb::Left);
                        let (e, ok) =
                            self.optimise_window_site(pos, &envs, &lanczos, absorb, null_tol)?;
                        energy = e + shift;
                        unconverged_solves += usize::from(!ok);
                        if pos > 0 {
                            envs.update_right(pos - 1, &self.tensors);
                        }
                    }
                } else {
                    // Forward 2-site sweep across window: pos = 0 .. l - 1.
                    for pos in 0..l - 1 {
                        let (e, ok) = self.optimise_window_pair(
                            pos,
                            &envs,
                            &lanczos,
                            noise,
                            Absorb::Right,
                            null_tol,
                        )?;
                        energy = e + shift;
                        unconverged_solves += usize::from(!ok);
                        envs.update_left(pos, &self.tensors);
                    }
                    if polish_back {
                        // Backward 1-site exact variational polish: pos = (l - 1) ..= 0.
                        for pos in (0..l).rev() {
                            let absorb = (pos > 0).then_some(Absorb::Left);
                            let (e, ok) = self.optimise_window_site(
                                pos,
                                &envs,
                                &lanczos,
                                absorb,
                                null_tol,
                            )?;
                            energy = e + shift;
                            unconverged_solves += usize::from(!ok);
                            if pos > 0 {
                                envs.update_right(pos - 1, &self.tensors);
                            }
                        }
                    } else {
                        // Backward 2-site sweep across window: pos = (l - 2) ..= 0.
                        for pos in (0..l - 1).rev() {
                            let (e, ok) = self.optimise_window_pair(
                                pos,
                                &envs,
                                &lanczos,
                                noise,
                                Absorb::Left,
                                null_tol,
                            )?;
                            energy = e + shift;
                            unconverged_solves += usize::from(!ok);
                            envs.update_right(pos, &self.tensors);
                        }
                    }
                }
            }

            // Re-normalise once per full ring sweep to prevent drift.
            self.normalise();

            let change = sweep_energies
                .last()
                .map_or(f64::INFINITY, |prev: &f64| (prev - energy).abs());
            sweep_energies.push(energy);
            if change < options.tolerance {
                if noise == 0.0 {
                    converged = true;
                    break;
                }
                noise = 0.0;
            } else if noise > 0.0 && sweep >= 2 && change < 1e-4 {
                // Once noisy sweeps have populated multi-qubit sectors and
                // stabilised, switch noise off so 1-site polishing converges
                // cleanly to machine precision.
                noise = 0.0;
            }
            noise *= options.noise_decay;
        }

        // Report exact expectation energy on the final state for small χ,
        // or the last converged variational sweep energy.
        let final_energy = if energy.is_finite() {
            energy
        } else {
            self.expectation(hamiltonian)?
        };
        Ok(DmrgResult {
            energy: final_energy,
            sweep_energies,
            converged,
            unconverged_solves,
        })
    }

    /// Put `opp_sites` into strictly left-orthonormal form, absorbing the carry
    /// `R` into `win_first` (`(opp_sites.last() + 1) % n == win_first`).
    pub(super) fn canonicalise_opp_right(&mut self, opp_sites: &[usize], win_first: usize) {
        let k = opp_sites.len();
        for i in 0..k {
            let next = if i + 1 < k {
                opp_sites[i + 1]
            } else {
                win_first
            };
            self.qr_move_right_bond(opp_sites[i], next);
        }
    }

    /// Put `opp_sites` into strictly right-orthonormal form, absorbing the carry
    /// `L` into `win_last` (`(win_last + 1) % n == opp_sites[0]`).
    pub(super) fn canonicalise_opp_left(&mut self, opp_sites: &[usize], win_last: usize) {
        let k = opp_sites.len();
        for i in (0..k).rev() {
            let prev = if i > 0 { opp_sites[i - 1] } else { win_last };
            self.qr_move_left_bond(prev, opp_sites[i]);
        }
    }

    /// Put `win_sites` into right-canonical form from `L - 1` down to `0` so
    /// sites `1..L` are strictly right-orthonormal before a rightward sweep.
    pub(super) fn canonicalise_window_left(&mut self, win_sites: &[usize]) {
        let l = win_sites.len();
        for pos in (1..l).rev() {
            let q_cur = win_sites[pos];
            let q_prev = win_sites[pos - 1];
            self.qr_move_left_bond(q_prev, q_cur);
        }
    }

    /// Put `win_sites` into left-canonical form from `0` up to `L - 1` so sites
    /// `0..L-1` are strictly left-orthonormal before a leftward sweep.
    pub(super) fn canonicalise_window_right(&mut self, win_sites: &[usize]) {
        let l = win_sites.len();
        for pos in 0..l - 1 {
            let q_cur = win_sites[pos];
            let q_next = win_sites[pos + 1];
            self.qr_move_right_bond(q_cur, q_next);
        }
    }

    /// Move gauge weight from `q_cur` to `q_next = (q_cur + 1) % n`, leaving
    /// `tensors[q_cur]` strictly left-orthonormal.
    pub(super) fn qr_move_right_bond(&mut self, q_cur: usize, q_next: usize) {
        if let Some(sym) = &self.symmetry {
            let (q_mat, r_mat, new_mid) = qr_left_symmetric(
                self.tensors[q_cur].as_left_matrix(),
                sym.left_bond(q_cur),
                sym.right_bond(q_cur),
                sym,
                q_cur,
            );
            let k = new_mid.dim();
            let left = self.tensors[q_cur].left;
            self.tensors[q_cur] = Tensor::from_matrix(q_mat.as_ref(), left, k);
            let next_r = self.tensors[q_next].right;
            let merged = matmul(r_mat.as_ref(), self.tensors[q_next].as_right_matrix());
            self.tensors[q_next] = Tensor::from_matrix(merged.as_ref(), k, next_r);
            self.symmetry.as_mut().unwrap().bonds[q_next] = new_mid;
            return;
        }
        let left = self.tensors[q_cur].left;
        let (q_mat, r_mat) = thin_qr(self.tensors[q_cur].as_left_matrix());
        let k = q_mat.ncols();
        self.tensors[q_cur] = Tensor::from_matrix(q_mat.as_ref(), left, k);
        let next_r = self.tensors[q_next].right;
        let merged = matmul(r_mat.as_ref(), self.tensors[q_next].as_right_matrix());
        self.tensors[q_next] = Tensor::from_matrix(merged.as_ref(), k, next_r);
    }

    /// Move gauge weight from `q_cur` to `q_prev = (q_cur + n - 1) % n`,
    /// leaving `tensors[q_cur]` strictly right-orthonormal.
    pub(super) fn qr_move_left_bond(&mut self, q_prev: usize, q_cur: usize) {
        if let Some(sym) = &self.symmetry {
            let (l_mat, q_mat, new_mid) = qr_right_symmetric(
                self.tensors[q_cur].as_right_matrix(),
                sym.left_bond(q_cur),
                sym.right_bond(q_cur),
                sym,
                q_cur,
            );
            let k = new_mid.dim();
            let right = self.tensors[q_cur].right;
            self.tensors[q_cur] = Tensor::from_matrix(q_mat.as_ref(), k, right);
            let prev_l = self.tensors[q_prev].left;
            let merged = matmul(self.tensors[q_prev].as_left_matrix(), l_mat.as_ref());
            self.tensors[q_prev] = Tensor::from_matrix(merged.as_ref(), prev_l, k);
            self.symmetry.as_mut().unwrap().bonds[q_cur] = new_mid;
            return;
        }
        let right = self.tensors[q_cur].right;
        let adj = self.tensors[q_cur].as_right_matrix().adjoint().to_owned();
        let (q_adj, r_adj) = thin_qr(adj.as_ref());
        let q_right = q_adj.adjoint().to_owned();
        let l_carry = r_adj.adjoint().to_owned();
        let k = q_right.nrows();
        self.tensors[q_cur] = Tensor::from_matrix(q_right.as_ref(), k, right);
        let prev_l = self.tensors[q_prev].left;
        let merged = matmul(self.tensors[q_prev].as_left_matrix(), l_carry.as_ref());
        self.tensors[q_prev] = Tensor::from_matrix(merged.as_ref(), prev_l, k);
    }

    /// Exact 1-site variational update at `win_sites[pos]` (zero SVD truncation,
    /// strictly monotonically lowering the exact periodic energy), followed by
    /// an exact QR gauge shift if `absorb` is `Some`.
    fn optimise_window_site(
        &mut self,
        pos: usize,
        envs: &WindowEnvs<'_>,
        lanczos: &EigenOptions,
        absorb: Option<Absorb>,
        null_tol: f64,
    ) -> Result<(f64, bool), Error> {
        let q = envs.win_sites[pos];
        let prob = envs.one_site_problem(pos, &self.tensors, self.symmetry.as_ref(), GAUGE_TOL);
        let v0 = prob.to_solver_vec(&self.tensors[q].data);
        let h_op = prob.ham_linear_op();
        let n_op = prob.norm_linear_op();
        let mut hv = vec![crate::mps::ZERO; v0.len()];
        let mut nv = vec![crate::mps::ZERO; v0.len()];
        h_op.apply(&v0, &mut hv);
        n_op.apply(&v0, &mut nv);
        let n0 = crate::statevector::krylov::dot(&v0, &nv).re;
        let e0 = if n0 > 1e-14 {
            crate::statevector::krylov::dot(&v0, &hv).re / n0
        } else {
            f64::INFINITY
        };

        let ritz = prob.solve_lowest(&v0, lanczos, null_tol, false)?;
        h_op.apply(&ritz.vector, &mut hv);
        n_op.apply(&ritz.vector, &mut nv);
        let na = crate::statevector::krylov::dot(&ritz.vector, &nv).re;
        let ea = if na > 1e-14 {
            crate::statevector::krylov::dot(&ritz.vector, &hv).re / na
        } else {
            f64::INFINITY
        };
        let out_e = if ea.is_finite() && (!e0.is_finite() || ea <= e0 + 1e-9 * (e0.abs() + 1.0)) {
            let gauge_vec = prob.unpack_gauge(&ritz.vector);
            let full_vec = prob.gauge_to_full(&gauge_vec);
            self.tensors[q] = Tensor {
                left: prob.l_full,
                right: prob.r_full,
                data: full_vec,
            };
            ea
        } else {
            e0
        };
        if let Some(dir) = absorb {
            match dir {
                Absorb::Right => self.qr_move_right_bond(q, envs.win_sites[pos + 1]),
                Absorb::Left => self.qr_move_left_bond(envs.win_sites[pos - 1], q),
            }
        }
        Ok((out_e, ritz.converged))
    }

    /// Optimise pair `(win_sites[pos], win_sites[pos + 1])` under `envs`.
    fn optimise_window_pair(
        &mut self,
        pos: usize,
        envs: &WindowEnvs<'_>,
        lanczos: &EigenOptions,
        noise: f64,
        absorb: Absorb,
        null_tol: f64,
    ) -> Result<(f64, bool), Error> {
        let q1 = envs.win_sites[pos];
        let q2 = envs.win_sites[pos + 1];
        let prob = envs.two_site_problem(pos, &self.tensors, self.symmetry.as_ref(), GAUGE_TOL);
        let theta_cur = two_site_dense(&self.tensors[q1], &self.tensors[q2]);
        let v0 = prob.to_solver_vec(&theta_cur);

        let ritz = prob.solve_lowest(&v0, lanczos, null_tol, true)?;
        let theta_tilde_vec = prob.unpack_gauge(&ritz.vector);
        let theta_tilde =
            MatRef::from_row_major_slice(&theta_tilde_vec, prob.l_eff * 2, 2 * prob.r_eff);

        // Save pre-split state when noise == 0 so a badly conditioned Frobenius
        // truncation can never degrade the variational energy.
        let saved = if noise == 0.0 {
            let h_op = prob.ham_linear_op();
            let n_op = prob.norm_linear_op();
            let mut hv = vec![crate::mps::ZERO; v0.len()];
            let mut nv = vec![crate::mps::ZERO; v0.len()];
            h_op.apply(&v0, &mut hv);
            n_op.apply(&v0, &mut nv);
            let n0 = crate::statevector::krylov::dot(&v0, &nv).re;
            let e0 = if n0 > 1e-14 {
                crate::statevector::krylov::dot(&v0, &hv).re / n0
            } else {
                f64::INFINITY
            };
            Some((
                e0,
                n0,
                self.tensors[q1].clone(),
                self.tensors[q2].clone(),
                self.symmetry.as_ref().map(|s| s.bonds[q2].clone()),
            ))
        } else {
            None
        };

        if noise > 0.0 {
            let theta_full = prob.gauge_to_full(&theta_tilde_vec);
            let pert = envs.perturbation_blocks(
                pos,
                &theta_full,
                prob.l_full,
                prob.r_full,
                matches!(absorb, Absorb::Right),
            );
            self.split_window_pair_perturbed(q1, q2, &prob, theta_tilde, &theta_full, &pert, noise, absorb);
        } else {
            self.split_window_pair(q1, q2, &prob, theta_tilde, absorb);
        }

        let mut out_energy = ritz.value;
        if let Some((e0, n0, old_a, old_b, old_mid)) = saved {
            let h_op = prob.ham_linear_op();
            let n_op = prob.norm_linear_op();
            let mut hv = vec![crate::mps::ZERO; v0.len()];
            let mut nv = vec![crate::mps::ZERO; v0.len()];
            let eval_current = |tensors: &[Tensor],
                                hv: &mut [C64],
                                nv: &mut [C64]|
             -> f64 {
                let ta = two_site_dense(&tensors[q1], &tensors[q2]);
                let va = prob.to_solver_vec(&ta);
                h_op.apply(&va, hv);
                n_op.apply(&va, nv);
                let na = crate::statevector::krylov::dot(&va, nv).re;
                if na > 1e-14 {
                    crate::statevector::krylov::dot(&va, hv).re / na
                } else {
                    f64::INFINITY
                }
            };

            let ea = eval_current(&self.tensors, &mut hv, &mut nv);
            let slack = 1e-9 * (e0.abs() + 1.0);
            if !ea.is_finite() || (e0.is_finite() && ea > e0 + slack) {
                // Frobenius-norm SVD truncation degraded the true periodic
                // variational energy. Perform a damped line search along
                // `v(t) = (1 - t) v0_n + t · phase · ritz.vector`: as `t → 0`,
                // `v(t)` approaches the rank-`old_r ≤ max_bond` starting tensor
                // (zero truncation error) while growing bond dimension smoothly
                // with strictly decreasing energy!
                let mut best_e = if ea.is_finite() { ea } else { f64::INFINITY };
                let mut best_state = if ea.is_finite() {
                    Some((
                        self.tensors[q1].clone(),
                        self.tensors[q2].clone(),
                        self.symmetry.as_ref().map(|s| s.bonds[q2].clone()),
                    ))
                } else {
                    None
                };

                if n0 > 1e-14 {
                    let s0 = 1.0 / n0.sqrt();
                    n_op.apply(&ritz.vector, &mut nv);
                    let nr = crate::statevector::krylov::dot(&ritz.vector, &nv).re;
                    let sr = if nr > 1e-14 { 1.0 / nr.sqrt() } else { 1.0 };
                    let ov = crate::statevector::krylov::dot(&v0, &nv) * (s0 * sr);
                    let phase = if ov.norm() > 1e-14 {
                        ov.conj() / ov.norm()
                    } else {
                        crate::mps::ONE
                    };
                    let mut v_try = vec![crate::mps::ZERO; v0.len()];
                    for &t in &[0.5f64, 0.25, 0.125, 0.0625] {
                        let a_coef = C64::new((1.0 - t) * s0, 0.0);
                        let b_coef = phase * (t * sr);
                        for idx in 0..v0.len() {
                            v_try[idx] = a_coef * v0[idx] + b_coef * ritz.vector[idx];
                        }
                        let tt_vec = prob.unpack_gauge(&v_try);
                        let tt = MatRef::from_row_major_slice(
                            &tt_vec,
                            prob.l_eff * 2,
                            2 * prob.r_eff,
                        );
                        self.split_window_pair(q1, q2, &prob, tt, absorb);
                        let e_try = eval_current(&self.tensors, &mut hv, &mut nv);
                        if e_try.is_finite() && e_try < best_e {
                            best_e = e_try;
                            best_state = Some((
                                self.tensors[q1].clone(),
                                self.tensors[q2].clone(),
                                self.symmetry.as_ref().map(|s| s.bonds[q2].clone()),
                            ));
                            if e_try <= e0 - 1e-12 * (e0.abs() + 1.0) {
                                break;
                            }
                        }
                    }
                }

                if best_e <= e0 + slack {
                    if let Some((a_b, b_b, mid_b)) = best_state {
                        self.tensors[q1] = a_b;
                        self.tensors[q2] = b_b;
                        if let Some(mid) = mid_b {
                            self.symmetry.as_mut().unwrap().bonds[q2] = mid;
                        }
                    }
                    out_energy = best_e;
                } else {
                    self.tensors[q1] = old_a;
                    self.tensors[q2] = old_b;
                    if let Some(mid) = old_mid {
                        self.symmetry.as_mut().unwrap().bonds[q2] = mid;
                    }
                    match absorb {
                        Absorb::Right => self.qr_move_right_bond(q1, q2),
                        Absorb::Left => self.qr_move_left_bond(q1, q2),
                    }
                    out_energy = e0;
                }
            } else {
                out_energy = ea;
            }
        }
        Ok((out_energy, ritz.converged))
    }

    /// Split gauge-conditioned two-site tensor `theta_tilde` into `tensors[q1]`
    /// and `tensors[q2]` with optimal metric-weighted truncation and strict QR
    /// orthonormality on the side moved away from.
    pub(super) fn split_window_pair(
        &mut self,
        q1: usize,
        q2: usize,
        prob: &super::window::LocalProblem,
        theta_tilde: MatRef<'_, C64>,
        absorb: Absorb,
    ) {
        if let Some(sym) = &self.symmetry {
            let (a_new, b_new, mid_sec, disc) = split_metric_symmetric(
                theta_tilde,
                prob.g_l.as_ref(),
                prob.g_r.as_ref(),
                sym.left_bond(q1),
                prob.left_eff_sec.as_ref().unwrap(),
                prob.right_eff_sec.as_ref().unwrap(),
                sym.right_bond(q2),
                sym,
                q1,
                q2,
                &self.options,
                absorb,
            );
            self.tensors[q1] = a_new;
            self.tensors[q2] = b_new;
            self.symmetry.as_mut().unwrap().bonds[q2] = mid_sec;
            self.truncation_error += disc;
            return;
        }

        let (u, s, v) = thin_svd(theta_tilde);
        let total: f64 = s.iter().map(|x| x * x).sum();
        let keep = keep_count(&s, total, &self.options);
        let kept: f64 = s[..keep].iter().map(|x| x * x).sum();
        if total > 0.0 {
            self.truncation_error += ((total - kept) / total).max(0.0);
        }

        let u_k = u.as_ref().subcols(0, keep);
        let vh_k = Mat::<C64>::from_fn(keep, v.nrows(), |r, c| v[(c, r)].conj());

        // Map back from gauge coordinates: A_raw = (G_L ⊗ I_2) U_k, B_raw = V_k† (I_2 ⊗ G_R^T).
        let (l_full, r_full) = (prob.l_full, prob.r_full);
        let (l_eff, r_eff) = (prob.l_eff, prob.r_eff);

        let mut a_raw = Mat::<C64>::zeros(l_full * 2, keep);
        for p in 0..2 {
            let u_p = Mat::<C64>::from_fn(l_eff, keep, |l, c| u_k[(l * 2 + p, c)]);
            let prod = matmul(prob.g_l.as_ref(), u_p.as_ref());
            for l in 0..l_full {
                for c in 0..keep {
                    a_raw[(l * 2 + p, c)] = prod[(l, c)];
                }
            }
        }
        let mut b_raw = Mat::<C64>::zeros(keep, 2 * r_full);
        for p in 0..2 {
            let v_p = Mat::<C64>::from_fn(keep, r_eff, |r, c| vh_k[(r, p * r_eff + c)]);
            let prod = matmul(v_p.as_ref(), prob.g_r.transpose());
            for r in 0..keep {
                for c in 0..r_full {
                    b_raw[(r, p * r_full + c)] = prod[(r, c)];
                }
            }
        }

        match absorb {
            Absorb::Right => {
                let (q_left, r_carry) = thin_qr(a_raw.as_ref());
                let k = q_left.ncols();
                let rs = Mat::<C64>::from_fn(k, keep, |r, c| r_carry[(r, c)] * s[c]);
                let b_final = matmul(rs.as_ref(), b_raw.as_ref());
                self.tensors[q1] = Tensor::from_matrix(q_left.as_ref(), l_full, k);
                self.tensors[q2] = Tensor::from_matrix(b_final.as_ref(), k, r_full);
            }
            Absorb::Left => {
                let b_adj = b_raw.adjoint().to_owned();
                let (q_adj, r_adj) = thin_qr(b_adj.as_ref());
                let q_right = q_adj.adjoint().to_owned();
                let l_carry = r_adj.adjoint().to_owned();
                let k = q_right.nrows();
                let a_s = Mat::<C64>::from_fn(l_full * 2, keep, |r, c| a_raw[(r, c)] * s[c]);
                let a_final = matmul(a_s.as_ref(), l_carry.as_ref());
                self.tensors[q1] = Tensor::from_matrix(a_final.as_ref(), l_full, k);
                self.tensors[q2] = Tensor::from_matrix(q_right.as_ref(), k, r_full);
            }
        }
    }

    /// Metric-weighted split with White's subspace-expansion noise: keeps the
    /// exact truncated basis of `theta_full` and widens the bond (at zero
    /// weight) with leading perturbation directions so 3/4/6-qubit lattice
    /// gauge terms act even from a product state.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn split_window_pair_perturbed(
        &mut self,
        q1: usize,
        q2: usize,
        prob: &super::window::LocalProblem,
        theta_tilde: MatRef<'_, C64>,
        theta_full: &[C64],
        perturbation: &[Mat<C64>],
        noise: f64,
        absorb: Absorb,
    ) {
        // First perform the exact metric-weighted split to get the state's own basis.
        self.split_window_pair(q1, q2, prob, theta_tilde, absorb);
        let keep = self.tensors[q1].right;
        let (rows, cols) = (prob.l_full * 2, 2 * prob.r_full);
        let room = noise_room(keep)
            .min(self.options.max_bond.saturating_sub(keep))
            .min(rows.min(cols).saturating_sub(keep));
        if room == 0 || perturbation.is_empty() {
            return;
        }

        let theta_mat = MatRef::from_row_major_slice(theta_full, rows, cols);
        let total = theta_mat.norm_l2().powi(2);
        if total <= 0.0 || !total.is_finite() {
            return;
        }
        let side = match absorb {
            Absorb::Right => rows,
            Absorb::Left => cols,
        };
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
        if blocks.is_empty() {
            return;
        }
        let width: usize = blocks.iter().map(|b| b.ncols()).sum();
        if width == 0 {
            return;
        }

        if let Some(sym) = &self.symmetry {
            let side_groups = match absorb {
                Absorb::Right => sym.group_rows(sym.left_bond(q1), q1),
                Absorb::Left => sym.group_cols(sym.right_bond(q2), q2),
            };
            let cur_mid = sym.left_bond(q2).clone();
            let own_mat = match absorb {
                Absorb::Right => self.tensors[q1].as_left_matrix().to_owned(),
                Absorb::Left => self.tensors[q2].as_right_matrix().adjoint().to_owned(),
            };
            let mut own_blocks: Vec<Mat<C64>> = Vec::with_capacity(side_groups.len());
            let mut extra_blocks: Vec<Mat<C64>> = Vec::with_capacity(side_groups.len());
            let mut ranked: Vec<(f64, usize, usize)> = Vec::new();
            let mut cand_vecs: Vec<Mat<C64>> = Vec::with_capacity(side_groups.len());

            for (g, (label, idx)) in side_groups.iter().enumerate() {
                let own_g = match cur_mid.find(label) {
                    Some(k) => {
                        let rng = cur_mid.range(k);
                        Mat::<C64>::from_fn(idx.len(), rng.len(), |x, c| {
                            own_mat[(idx[x], rng.start + c)]
                        })
                    }
                    None => Mat::zeros(idx.len(), 0),
                };
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
                    let mut o = m.to_owned();
                    if own_g.ncols() > 0 {
                        o -= matmul(own_g.as_ref(), matmul(own_g.adjoint(), m).as_ref());
                    }
                    o
                };
                let rho = project(project(rho.as_ref()).adjoint().to_owned().as_ref());
                let (vecs, wts, _) = thin_svd(rho.as_ref());
                for (j, &w) in wts.iter().enumerate() {
                    if w > self.options.cutoff * total {
                        ranked.push((w, g, j));
                    }
                }
                own_blocks.push(own_g);
                cand_vecs.push(vecs);
            }
            ranked.sort_by(|a, b| b.0.total_cmp(&a.0).then((a.1, a.2).cmp(&(b.1, b.2))));
            ranked.truncate(room);
            let mut take = vec![0usize; side_groups.len()];
            for &(_, g, _) in &ranked {
                take[g] += 1;
            }
            for g in 0..side_groups.len() {
                let k = take[g];
                if k == 0 {
                    extra_blocks.push(Mat::zeros(own_blocks[g].nrows(), 0));
                } else {
                    let mut c = cand_vecs[g].as_ref().subcols(0, k).to_owned();
                    if own_blocks[g].ncols() > 0 {
                        c -= matmul(
                            own_blocks[g].as_ref(),
                            matmul(own_blocks[g].adjoint(), c.as_ref()).as_ref(),
                        );
                    }
                    let (q, sv, _) = thin_svd(c.as_ref());
                    let cnt = sv.iter().take_while(|&&x| x > 0.5).count();
                    extra_blocks.push(q.as_ref().subcols(0, cnt).to_owned());
                }
            }
            let added_total: usize = extra_blocks.iter().map(|e| e.ncols()).sum();
            if added_total == 0 {
                return;
            }
            let new_bond = keep + added_total;
            let mut basis = Mat::<C64>::zeros(side, new_bond);
            let mut entries = Vec::new();
            let mut old_pos = vec![0usize; keep];
            let mut at = 0;
            for (((label, idx), own_g), ext_g) in side_groups
                .iter()
                .zip(&own_blocks)
                .zip(&extra_blocks)
            {
                let start = at;
                if let Some(k_old) = cur_mid.find(label) {
                    let rng = cur_mid.range(k_old);
                    for (c, old_idx) in rng.enumerate() {
                        old_pos[old_idx] = at;
                        for (x, &r) in idx.iter().enumerate() {
                            basis[(r, at)] = own_g[(x, c)];
                        }
                        at += 1;
                    }
                }
                for c in 0..ext_g.ncols() {
                    for (x, &r) in idx.iter().enumerate() {
                        basis[(r, at)] = ext_g[(x, c)];
                    }
                    at += 1;
                }
                if at > start {
                    entries.push((label.clone(), at - start));
                }
            }
            match absorb {
                Absorb::Right => {
                    let old_b = &self.tensors[q2];
                    let mut new_b = Mat::<C64>::zeros(new_bond, cols);
                    for (old_r, &new_r) in old_pos.iter().enumerate() {
                        for c in 0..cols {
                            new_b[(new_r, c)] = old_b.data[old_r * cols + c];
                        }
                    }
                    self.tensors[q1] = Tensor::from_matrix(basis.as_ref(), prob.l_full, new_bond);
                    self.tensors[q2] = Tensor::from_matrix(new_b.as_ref(), new_bond, prob.r_full);
                }
                Absorb::Left => {
                    let old_a = &self.tensors[q1];
                    let mut new_a = Mat::<C64>::zeros(rows, new_bond);
                    for r in 0..rows {
                        for (old_c, &new_c) in old_pos.iter().enumerate() {
                            new_a[(r, new_c)] = old_a.data[r * keep + old_c];
                        }
                    }
                    let bh = basis.adjoint().to_owned();
                    self.tensors[q1] = Tensor::from_matrix(new_a.as_ref(), prob.l_full, new_bond);
                    self.tensors[q2] = Tensor::from_matrix(bh.as_ref(), new_bond, prob.r_full);
                }
            }
            self.symmetry.as_mut().unwrap().bonds[q2] = crate::mps::charges::Sectors::new(entries);
            return;
        }

        // Dense noise padding (appends orthonormal extra columns/rows at zero weight).
        let own = match absorb {
            Absorb::Right => self.tensors[q1].as_left_matrix().to_owned(),
            Absorb::Left => self.tensors[q2].as_right_matrix().adjoint().to_owned(),
        };
        let mut stacked = Mat::<C64>::zeros(side, width);
        let mut at = 0;
        for b in &blocks {
            stacked
                .as_mut()
                .subcols_mut(at, b.ncols())
                .copy_from(b);
            at += b.ncols();
        }
        let rho = matmul(stacked.as_ref(), stacked.adjoint());
        let project = |m: MatRef<'_, C64>| -> Mat<C64> {
            let mut o = m.to_owned();
            o -= matmul(own.as_ref(), matmul(own.adjoint(), m).as_ref());
            o
        };
        let rho = project(project(rho.as_ref()).adjoint().to_owned().as_ref());
        let (vecs, wts, _) = thin_svd(rho.as_ref());
        let cnt = wts
            .iter()
            .take_while(|&&w| w > self.options.cutoff * total)
            .count()
            .min(room);
        if cnt == 0 {
            return;
        }
        let cands = project(vecs.as_ref().subcols(0, cnt));
        let (q, sv, _) = thin_svd(cands.as_ref());
        let added = sv.iter().take_while(|&&x| x > 0.5).count();
        if added == 0 {
            return;
        }
        let new_bond = keep + added;
        match absorb {
            Absorb::Right => {
                let mut new_a = Mat::<C64>::zeros(rows, new_bond);
                new_a.as_mut().subcols_mut(0, keep).copy_from(own.as_ref());
                new_a
                    .as_mut()
                    .subcols_mut(keep, added)
                    .copy_from(q.as_ref().subcols(0, added));
                let old_b = self.tensors[q2].as_right_matrix();
                let mut new_b = Mat::<C64>::zeros(new_bond, cols);
                new_b.as_mut().subrows_mut(0, keep).copy_from(old_b);
                self.tensors[q1] = Tensor::from_matrix(new_a.as_ref(), prob.l_full, new_bond);
                self.tensors[q2] = Tensor::from_matrix(new_b.as_ref(), new_bond, prob.r_full);
            }
            Absorb::Left => {
                let mut new_b_adj = Mat::<C64>::zeros(cols, new_bond);
                new_b_adj
                    .as_mut()
                    .subcols_mut(0, keep)
                    .copy_from(own.as_ref());
                new_b_adj
                    .as_mut()
                    .subcols_mut(keep, added)
                    .copy_from(q.as_ref().subcols(0, added));
                let new_b = new_b_adj.adjoint().to_owned();
                let old_a = self.tensors[q1].as_left_matrix();
                let mut new_a = Mat::<C64>::zeros(rows, new_bond);
                new_a.as_mut().subcols_mut(0, keep).copy_from(old_a);
                self.tensors[q1] = Tensor::from_matrix(new_a.as_ref(), prob.l_full, new_bond);
                self.tensors[q2] = Tensor::from_matrix(new_b.as_ref(), new_bond, prob.r_full);
            }
        }
    }
}

/// Overlapping half-ring `(win_sites, opp_sites)` windows covering all `n`
/// periodic bonds (including `(n - 1, 0)`).
pub(super) fn ring_windows(n: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
    if n == 2 {
        return vec![(vec![0, 1], vec![])];
    }
    if n == 3 {
        return vec![
            (vec![0, 1], vec![2]),
            (vec![1, 2], vec![0]),
            (vec![2, 0], vec![1]),
        ];
    }
    let m = n / 2;
    let w1: Vec<usize> = (0..=m).collect();
    let o1: Vec<usize> = (m + 1..n).collect();
    let mut w2: Vec<usize> = (m..n).collect();
    w2.push(0);
    let o2: Vec<usize> = (1..m).collect();
    vec![(w1, o1), (w2, o2)]
}

/// Dense two-site tensor `theta[(l, p, q), r]` from `a` and `b`.
pub(super) fn two_site_dense(a: &Tensor, b: &Tensor) -> Vec<C64> {
    let theta = matmul(a.as_left_matrix(), b.as_right_matrix());
    let mut out = Vec::with_capacity(theta.nrows() * theta.ncols());
    for r in 0..theta.nrows() {
        for c in 0..theta.ncols() {
            out.push(theta[(r, c)]);
        }
    }
    out
}
