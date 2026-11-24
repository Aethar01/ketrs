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
    pub fn tdvp(
        &mut self,
        hamiltonian: &PauliSum,
        time: f64,
        steps: usize,
    ) -> Result<(), Error> {
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
            for _step in 0..steps {
                for wh in &windows {
                    #[cfg(feature = "gpu")]
                    let g = gpu_scratch.as_ref();
                    #[cfg(not(feature = "gpu"))]
                    let g = None::<&()>;
                    self.tdvp_window_forward(wh, half, null_tol, g)?;
                }
                for wh in windows.iter().rev() {
                    #[cfg(feature = "gpu")]
                    let g = gpu_scratch.as_ref();
                    #[cfg(not(feature = "gpu"))]
                    let g = None::<&()>;
                    self.tdvp_window_backward(wh, half, null_tol, g)?;
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
        self.split_window_pair(q1, q2, &prob, theta_tilde, absorb);
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

/// Partition `hamiltonian` (excluding global identity shift) across overlapping
/// half-ring windows so each window's sub-Hamiltonian acts purely inside its
/// own open interval (zero `H_opp`, zero crossing channels for any range `≤ n/2`).
fn partition_hamiltonian_windows(
    hamiltonian: &PauliSum,
    n: usize,
    charges: Option<&crate::mps::charges::Charges>,
) -> Result<Vec<WindowHamiltonian>, Error> {
    let m = n / 2;
    // Window 1: [0, 1, ..., m], opp: [m+1, ..., n-1]
    let w1: Vec<usize> = (0..=m).collect();
    let o1: Vec<usize> = (m + 1..n).collect();
    // Window 2: [m, m+1, ..., n-1, 0], opp: [1, ..., m-1]
    let mut w2: Vec<usize> = (m..n).collect();
    w2.push(0);
    let o2: Vec<usize> = (1..m).collect();
    // Optional shifted window 3 for mid-range terms touching both `1..m-1` and `m+1..n-1`:
    let q0 = m / 2;
    let w3: Vec<usize> = (0..=m).map(|k| (q0 + k) % n).collect();
    let o3: Vec<usize> = (m + 1..n).map(|k| (q0 + k) % n).collect();

    let in_set = |sites: &[usize], ps: &crate::observable::PauliString| -> bool {
        ps.ops().iter().all(|&(q, _)| sites.contains(&q))
    };

    let mut h1 = PauliSum::new();
    let mut h2 = PauliSum::new();
    let mut h3 = PauliSum::new();

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
        } else {
            // Very long-range term (span > n/2): keep in window 1 as a crossing channel.
            h1.add(coeff, ps.clone());
        }
    }

    let mut out = Vec::with_capacity(3);
    if !h1.terms().is_empty() {
        out.push(split_hamiltonian_on_window(&h1, w1, o1, n, charges)?);
    }
    if !h2.terms().is_empty() {
        out.push(split_hamiltonian_on_window(&h2, w2, o2, n, charges)?);
    }
    if !h3.terms().is_empty() {
        out.push(split_hamiltonian_on_window(&h3, w3, o3, n, charges)?);
    }
    Ok(out)
}
