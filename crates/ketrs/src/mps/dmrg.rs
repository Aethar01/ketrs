//! Ground states by two-site DMRG.
//!
//! The Hamiltonian is turned into an MPO, and left and right environments
//! (the MPO sandwiched between the MPS and its conjugate, contracted up to a
//! pair of sites) are kept for every bond. Each step solves for the lowest
//! eigenvector of the effective Hamiltonian on two neighbouring sites by
//! Lanczos, starting from the current two-site tensor, then splits it by
//! truncated SVD and moves on. Sweeps go right and then left until the
//! energy stops changing.
//!
//! Two-site updates only see the parts of terms on more than two sites
//! that the bonds already carry, so plain two-site DMRG stalls when no
//! two-site move lowers the energy. A product state under a four-qubit hop
//! is one such case. Early sweeps therefore add noise. White's
//! density-matrix perturbation (Phys. Rev. B 72, 180403, 2005) is used in
//! the form of a subspace expansion (as in DMRG3S, Hubig et al., Phys.
//! Rev. B 91, 155115, 2015). Each split keeps the two-site tensor's own
//! truncated basis exactly and widens it with the strongest directions
//! that the partial terms crossing the bond reach. Those directions are
//! weighted by a noise strength that decays, and noise is switched off
//! once it stops helping.

use super::envs::Envs;
use super::linalg::{matmul, thin_svd};
use super::mpo::{Mpo, MpoSite, left_keyed, right_keyed};
use super::{Absorb, Mps, MpsOptions, Tensor};
use crate::error::Error;
use crate::observable::PauliSum;
use crate::statevector::eigen::EigenOptions;
use crate::statevector::krylov::norm;
use faer::{Mat, MatRef};
use num_complex::Complex64 as C64;
use rayon::prelude::*;

/// Settings for [`Mps::dmrg`].
#[derive(Clone, Debug)]
pub struct DmrgOptions {
    /// Most singular values kept on any bond.
    pub max_bond: usize,
    /// Discarded-weight cutoff for each split (see [`MpsOptions`]).
    pub cutoff: f64,
    /// Most sweeps (each one right and back).
    pub max_sweeps: usize,
    /// Stop once a sweep changes the energy by less than this.
    pub tolerance: f64,
    /// Residual tolerance of each two-site Lanczos solve.
    pub lanczos_tolerance: f64,
    /// Strength α of the noise in the first sweep (0 turns it off). Each
    /// split keeps the two-site tensor θ's own truncated basis and adds,
    /// at zero weight, the leading directions of `α Σ_c P_c P_c†` outside
    /// it. Here `P_c` is θ acted on by the part of `H` behind the split
    /// along one MPO channel `c` (a partial term continuing past the two
    /// sites), scaled to θ's norm. Directions whose weight is below
    /// `cutoff`, or beyond twice θ's own bond, are not added. The extra
    /// directions let terms on more than two sites act. Without them, DMRG
    /// can stall in a state no two-site move improves, such as a product
    /// state under a four-qubit hop. Since θ itself is never perturbed,
    /// noise costs time (wider bonds) but never accuracy.
    pub noise: f64,
    /// Factor applied to `noise` after each sweep. Noise stops once it is
    /// below `cutoff` or once a sweep with it changes the energy by less
    /// than `tolerance`. Convergence is only declared after a sweep without
    /// it.
    pub noise_decay: f64,
}

impl Default for DmrgOptions {
    fn default() -> Self {
        DmrgOptions {
            max_bond: 64,
            cutoff: 1e-12,
            max_sweeps: 20,
            tolerance: 1e-10,
            lanczos_tolerance: 1e-10,
            noise: 1e-4,
            noise_decay: 0.1,
        }
    }
}

/// The outcome of [`Mps::dmrg`].
#[derive(Clone, Debug)]
pub struct DmrgResult {
    /// The final ground-state energy estimate.
    pub energy: f64,
    /// The energy after each sweep.
    pub sweep_energies: Vec<f64>,
    /// Whether the energy settled within `tolerance` in a sweep without
    /// noise before `max_sweeps`.
    pub converged: bool,
    /// Two-site Lanczos solves that didn't reach `lanczos_tolerance` within
    /// 1000 iterations and went on with their best Ritz pair instead.
    pub unconverged_solves: usize,
}

/// An environment: for each MPO bond index, a matrix with rows on the bra
/// bond and columns on the ket bond.
#[derive(Clone)]
pub(super) struct Env {
    pub(super) blocks: Vec<Mat<C64>>,
}

impl Env {
    pub(super) fn boundary() -> Env {
        Env {
            blocks: vec![Mat::identity(1, 1)],
        }
    }
}

impl Mps {
    /// Find the ground state of `hamiltonian` by two-site DMRG, starting
    /// from this state and replacing it.
    ///
    /// DMRG keeps whatever symmetries the start shares with `H`: from
    /// `|0…0⟩` a number-conserving Hamiltonian never leaves the empty
    /// sector. Start from [`Mps::random`] for the true ground state, or from
    /// a state in the sector you want (e.g. a Néel state for `Sz = 0`).
    /// The perturbation set by [`DmrgOptions::noise`] keeps these
    /// symmetries too, since it is built from `H`.
    pub fn dmrg(
        &mut self,
        hamiltonian: &PauliSum,
        options: &DmrgOptions,
    ) -> Result<DmrgResult, Error> {
        let n = self.n_qubits();
        if n < 2 {
            return Err(Error::InvalidArgument(
                "DMRG needs at least two qubits".into(),
            ));
        }
        let mpo = self.mpo(hamiltonian)?;
        self.set_options(MpsOptions {
            max_bond: options.max_bond,
            cutoff: options.cutoff,
        });
        let lanczos = EigenOptions {
            tolerance: options.lanczos_tolerance,
            max_iterations: 1000,
            seed: 0,
        };

        self.move_center(0);
        let norm = self.norm_sqr().sqrt();
        self.tensors[0].data.iter_mut().for_each(|a| *a /= norm);

        let mut envs = Envs::new(self, &mpo)?;
        for i in (0..n - 1).rev() {
            envs.update_right(i, self, &mpo)?;
        }

        let mut sweep_energies = Vec::new();
        let mut energy = f64::INFINITY;
        let mut converged = false;
        let mut unconverged_solves = 0;
        let mut noise = options.noise.max(0.0);
        for _ in 0..options.max_sweeps {
            // Below this the perturbation is lost to the cutoff anyway.
            if noise < options.cutoff.max(1e-14) {
                noise = 0.0;
            }
            for i in 0..n - 1 {
                let solved;
                (energy, solved) =
                    self.optimise_pair(i, &mpo, &envs, &lanczos, noise, Absorb::Right)?;
                unconverged_solves += usize::from(!solved);
                envs.update_left(i, self, &mpo)?;
                // Rebuilt from the right end before it is needed again.
                envs.release_right(i + 1);
            }
            for i in (0..n - 1).rev() {
                let solved;
                (energy, solved) =
                    self.optimise_pair(i, &mpo, &envs, &lanczos, noise, Absorb::Left)?;
                unconverged_solves += usize::from(!solved);
                envs.update_right(i, self, &mpo)?;
                envs.release_left(i);
            }
            let total = energy + mpo.shift;
            let change = sweep_energies
                .last()
                .map_or(f64::INFINITY, |e: &f64| (e - total).abs());
            sweep_energies.push(total);
            if change < options.tolerance {
                if noise == 0.0 {
                    converged = true;
                    break;
                }
                // The noise has found what it can; settle without it.
                noise = 0.0;
            }
            noise *= options.noise_decay;
        }
        Ok(DmrgResult {
            energy: energy + mpo.shift,
            sweep_energies,
            converged,
            unconverged_solves,
        })
    }

    /// Replace sites `i` and `i + 1` with the lowest eigenvector of the
    /// effective Hamiltonian there, split with a perturbation of weight
    /// `noise`. Returns its energy (without the shift) and whether Lanczos
    /// converged (else this is its best Ritz pair).
    fn optimise_pair(
        &mut self,
        i: usize,
        mpo: &Mpo,
        envs: &Envs,
        lanczos: &EigenOptions,
        noise: f64,
        absorb: Absorb,
    ) -> Result<(f64, bool), Error> {
        let (l, r) = (self.tensors[i].left, self.tensors[i + 1].right);
        let op = envs.two_site(i, self, mpo)?;
        let mut v0 = op.two_site_vector(self, i);
        let scale = 1.0 / norm(&v0);
        v0.iter_mut().for_each(|a| *a *= scale);

        // The Ritz value is the energy of `x` to within the Lanczos
        // tolerance squared, without another product.
        let found = op.lowest(&v0, lanczos);
        envs.check()?;
        let found = found?;
        let (energy, x, solved) = (found.value, op.unpack(found.vector), found.converged);
        let theta = MatRef::from_row_major_slice(&x, l * 2, 2 * r);
        if noise > 0.0 {
            let perturbation = match absorb {
                Absorb::Right => left_perturbation(&*envs.left(i)?, &mpo.sites[i], &x, l, r),
                Absorb::Left => {
                    right_perturbation(&*envs.right(i + 1)?, &mpo.sites[i + 1], &x, l, r)
                }
            };
            self.split_perturbed(i, theta, &perturbation, noise, absorb);
        } else {
            self.split(i, theta, absorb);
        }
        Ok((energy, solved))
    }

    /// [`split`](Mps::split), but the side that does not take the centre
    /// also gets up to [`noise_room`] directions outside `θ`'s kept basis:
    /// the leading eigenvectors of `noise · Σ_c P_c P_c†` (`P_c† P_c` for a
    /// left move) projected off that basis, with each block `P_c` scaled to
    /// `θ`'s norm, while their weight is above the cutoff. `θ` keeps exactly
    /// the basis and weights a plain split would give it, so the energy is
    /// unchanged and the new directions only widen the next updates.
    pub(super) fn split_perturbed(
        &mut self,
        i: usize,
        theta: MatRef<'_, C64>,
        perturbation: &[Mat<C64>],
        noise: f64,
        absorb: Absorb,
    ) {
        if self.symmetry.is_some() {
            return self.split_perturbed_symmetric(i, theta, perturbation, noise, absorb);
        }
        let (rows, cols) = (theta.nrows(), theta.ncols());
        let weight = |m: MatRef<'_, C64>| -> f64 {
            (0..m.nrows())
                .flat_map(|r| (0..m.ncols()).map(move |c| m[(r, c)].norm_sqr()))
                .sum()
        };
        let total = weight(theta);
        let (u, s, v) = thin_svd(theta);
        let keep = self.bond_to_keep(&s, total);
        // θ's kept basis on the side being traced out, as columns.
        let (own, side) = match absorb {
            Absorb::Right => (u.as_ref().subcols(0, keep), rows),
            Absorb::Left => (v.as_ref().subcols(0, keep), cols),
        };
        let room = noise_room(keep)
            .min(self.options.max_bond.saturating_sub(keep))
            .min(rows.min(cols).saturating_sub(keep));
        let extra = if room > 0 && total > 0.0 {
            // Each channel (partial term) gets weight `noise`, however small
            // its coefficients: normalised over all of P together, hops
            // would be drowned by large penalty terms.
            // ρ = Σ_c s_c P_c P_c† (P_c† P_c moving left) is one product of
            // the channels side by side, each scaled by √s_c.
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
            let mut stacked = Mat::<C64>::zeros(side, width);
            let mut at = 0;
            for block in &blocks {
                stacked
                    .as_mut()
                    .subcols_mut(at, block.ncols())
                    .copy_from(block);
                at += block.ncols();
            }
            let rho = matmul(stacked.as_ref(), stacked.adjoint());
            // Off θ's basis: Q ρ Q with Q = 1 − own·own†.
            let project = |m: MatRef<'_, C64>| -> Mat<C64> {
                let mut out = m.to_owned();
                out -= matmul(own, matmul(own.adjoint(), m).as_ref());
                out
            };
            let rho = project(project(rho.as_ref()).adjoint().to_owned().as_ref());
            let (vectors, weights, _) = thin_svd(rho.as_ref());
            let count = weights
                .iter()
                .take_while(|&&w| w > self.options.cutoff * total)
                .count()
                .min(room);
            // Orthonormalise again against θ's basis, for rounding.
            let candidates = project(vectors.as_ref().subcols(0, count));
            let (q, sv, _) = thin_svd(candidates.as_ref());
            let count = sv.iter().take_while(|&&x| x > 0.5).count();
            q.as_ref().subcols(0, count).to_owned()
        } else {
            Mat::<C64>::zeros(side, 0)
        };
        let bond = keep + extra.ncols();
        let basis = Mat::<C64>::from_fn(side, bond, |r, c| {
            if c < keep {
                own[(r, c)]
            } else {
                extra[(r, c - keep)]
            }
        });
        let (left, right) = (rows / 2, cols / 2);
        match absorb {
            Absorb::Right => {
                let centre = matmul(basis.adjoint(), theta);
                let kept = weight(centre.as_ref());
                if total > 0.0 {
                    self.truncation_error += ((total - kept) / total).max(0.0);
                }
                let rescale = if kept > 0.0 {
                    C64::new((total / kept).sqrt(), 0.0)
                } else {
                    C64::new(1.0, 0.0)
                };
                let centre = Mat::<C64>::from_fn(bond, cols, |r, c| rescale * centre[(r, c)]);
                self.tensors[i] = Tensor::from_matrix(basis.as_ref(), left, bond);
                self.tensors[i + 1] = Tensor::from_matrix(centre.as_ref(), bond, right);
                self.center = i + 1;
            }
            Absorb::Left => {
                let centre = matmul(theta, basis.as_ref());
                let kept = weight(centre.as_ref());
                if total > 0.0 {
                    self.truncation_error += ((total - kept) / total).max(0.0);
                }
                let rescale = if kept > 0.0 {
                    C64::new((total / kept).sqrt(), 0.0)
                } else {
                    C64::new(1.0, 0.0)
                };
                let centre = Mat::<C64>::from_fn(rows, bond, |r, c| rescale * centre[(r, c)]);
                let basis = basis.adjoint().to_owned();
                self.tensors[i] = Tensor::from_matrix(centre.as_ref(), left, bond);
                self.tensors[i + 1] = Tensor::from_matrix(basis.as_ref(), bond, right);
                self.center = i;
            }
        }
    }
}

/// How many noise directions a split may add to a bond that keeps `keep`
/// of its own. Directions are ranked by noise weight, which only roughly
/// tracks usefulness, so fewer than twice `keep` can miss the one a trapped
/// state needs; more makes every noisy sweep slower.
pub(super) fn noise_room(keep: usize) -> usize {
    (2 * keep).max(4)
}

/// White's perturbation for a split moving right: for each MPO bond index
/// `b` between the two sites, `Σ_a (L_a ⊗ W_i[a][b]) θ` as a `(l, p) × (q, r)`
/// matrix. Its columns are the left halves of the partial terms of `H` that
/// continue past site `i`.
fn left_perturbation(env: &Env, w: &MpoSite, theta: &[C64], l: usize, r: usize) -> Vec<Mat<C64>> {
    let theta = MatRef::from_row_major_slice(theta, l, 4 * r);
    // y_a[l, (p, q, r)] = Σ_l' L_a[l, l'] θ[l', p, q, r]
    let y: Vec<Mat<C64>> = env
        .blocks
        .par_iter()
        .map(|e| matmul(e.as_ref(), theta))
        .collect();
    let rows = right_keyed(w);
    (0..w.right)
        .into_par_iter()
        .map(|b| {
            let mut out = Mat::<C64>::zeros(2 * l, 2 * r);
            for p_out in 0..2 {
                for &(a, p_in, f) in &rows[b * 2 + p_out] {
                    for q in 0..2 {
                        for col in 0..r {
                            let y_col = y[a].col_as_slice((2 * p_in + q) * r + col);
                            let out_col = out.col_as_slice_mut(q * r + col);
                            for (row, &v) in y_col.iter().enumerate() {
                                out_col[2 * row + p_out] += f * v;
                            }
                        }
                    }
                }
            }
            out
        })
        .collect()
}

/// The mirror of [`left_perturbation`] for a split moving left: for each
/// MPO bond index `a` between the two sites, `Σ_c (W_{i+1}[a][c] ⊗ R_c) θ`.
fn right_perturbation(env: &Env, w: &MpoSite, theta: &[C64], l: usize, r: usize) -> Vec<Mat<C64>> {
    let theta = MatRef::from_row_major_slice(theta, 4 * l, r);
    // z_c[(l, p, q), r] = Σ_r' θ[l, p, q, r'] R_c[r, r']
    let z: Vec<Mat<C64>> = env
        .blocks
        .par_iter()
        .map(|e| matmul(theta, e.transpose()))
        .collect();
    let rows = left_keyed(w);
    (0..w.left)
        .into_par_iter()
        .map(|a| {
            let mut out = Mat::<C64>::zeros(2 * l, 2 * r);
            for q_out in 0..2 {
                for &(c, q_in, f) in &rows[a * 2 + q_out] {
                    for col in 0..r {
                        let z_col = z[c].col_as_slice(col);
                        let out_col = out.col_as_slice_mut(q_out * r + col);
                        for (row, v) in out_col.iter_mut().enumerate() {
                            *v += f * z_col[2 * row + q_in];
                        }
                    }
                }
            }
            out
        })
        .collect()
}

/// Two neighbouring sites contracted, row-major `(l, p, q, r)` to match
/// [`Effective`](super::effective::Effective).
pub(super) fn two_site_vector(a: &Tensor, b: &Tensor) -> Vec<C64> {
    let theta = matmul(a.as_left_matrix(), b.as_right_matrix());
    (0..theta.nrows())
        .flat_map(|row| (0..theta.ncols()).map(move |col| (row, col)))
        .map(|(row, col)| theta[(row, col)])
        .collect()
}

/// The environment of sites `< i + 1` from that of sites `< i`, site `i`'s
/// tensor and its MPO site.
pub(super) fn update_left(env: &Env, a: &Tensor, w: &MpoSite) -> Env {
    let y = left_operator(env, a, w);
    let bra = a.as_left_matrix().adjoint();
    Env {
        blocks: y.par_iter().map(|yb| matmul(bra, yb.as_ref())).collect(),
    }
}

/// The mirror of [`right_operator`]: the MPO site `w` applied to `ket` and
/// contracted with a left environment whose rows are on a bra basis. For
/// each right MPO index `b`, a matrix with rows `(l_bra, p)` and columns on
/// `ket`'s right bond.
pub(super) fn left_operator(env: &Env, ket: &Tensor, w: &MpoSite) -> Vec<Mat<C64>> {
    let r = ket.right;
    let l = env.blocks[0].nrows();
    // x_a[l, (p, r')] = Σ_l' env_a[l, l'] A[l', p, r']
    let x: Vec<Mat<C64>> = env
        .blocks
        .par_iter()
        .map(|e| matmul(e.as_ref(), ket.as_right_matrix()))
        .collect();
    let rows = right_keyed(w);
    (0..w.right)
        .into_par_iter()
        .map(|b| {
            let mut y = Mat::<C64>::zeros(l * 2, r);
            for p_out in 0..2 {
                for &(a, p_in, f) in &rows[b * 2 + p_out] {
                    for col in 0..r {
                        let x_col = x[a].col_as_slice(p_in * r + col);
                        let y_col = y.col_as_slice_mut(col);
                        for (row, &v) in x_col.iter().enumerate() {
                            y_col[2 * row + p_out] += f * v;
                        }
                    }
                }
            }
            y
        })
        .collect()
}

/// The environment of sites `> i − 1` from that of sites `> i`, site `i`'s
/// tensor and its MPO site.
pub(super) fn update_right(env: &Env, a: &Tensor, w: &MpoSite) -> Env {
    let y = right_operator(env, a, w);
    let bra = a.as_right_matrix().conjugate();
    Env {
        blocks: y.par_iter().map(|ya| matmul(bra, ya.transpose())).collect(),
    }
}

/// The MPO site `w` applied to `ket` and contracted with a right
/// environment whose rows are on another (bra) basis: for each left MPO
/// index `a`, a matrix with rows on `ket`'s left bond and columns
/// `(p, r_bra)`. Contracting its columns with a bra tensor gives the next
/// environment; with a ket tensor, [`update_right`].
pub(super) fn right_operator(env: &Env, ket: &Tensor, w: &MpoSite) -> Vec<Mat<C64>> {
    let l = ket.left;
    let r = env.blocks[0].nrows();
    // x_c[(l', p), r] = Σ_r' A[l', p, r'] env_c[r, r']
    let x: Vec<Mat<C64>> = env
        .blocks
        .par_iter()
        .map(|e| matmul(ket.as_left_matrix(), e.transpose()))
        .collect();
    let rows = left_keyed(w);
    (0..w.left)
        .into_par_iter()
        .map(|a| {
            let mut y = Mat::<C64>::zeros(l, 2 * r);
            for p_out in 0..2 {
                for &(c, p_in, f) in &rows[a * 2 + p_out] {
                    for col in 0..r {
                        let x_col = x[c].col_as_slice(col);
                        let y_col = y.col_as_slice_mut(p_out * r + col);
                        for (row, v) in y_col.iter_mut().enumerate() {
                            *v += f * x_col[2 * row + p_in];
                        }
                    }
                }
            }
            y
        })
        .collect()
}
