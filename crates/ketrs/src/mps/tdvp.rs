//! Real-time evolution by the time-dependent variational principle (TDVP).
//!
//! TEBD needs the Hamiltonian as nearest-neighbour gates, routing anything
//! longer-ranged with swaps. TDVP instead evolves under the Hamiltonian's
//! MPO directly, so couplings of any range cost only MPO bond dimension.
//! This is the two-site, second-order projector-splitting integrator
//! (Haegeman et al., Phys. Rev. B 94, 2016): a sweep right with half steps
//! evolves each two-site tensor forward under its effective Hamiltonian,
//! splits it by truncated SVD and evolves the new centre site backward;
//! a sweep left does the same in reverse. Each local exponential is taken
//! by Krylov iteration, as for statevectors. The bond dimension grows as
//! the state needs, up to `max_bond`.
//!
//! TDVP projects the evolution onto states reachable within the current
//! bonds, so from a weakly entangled start couplings the bonds can't yet
//! carry are lost. While bonds are still growing, each step therefore
//! begins with a subspace expansion (after Yang and White, Phys. Rev. B
//! 102, 2020): the right bases are enlarged by those of `H|ψ⟩`, through
//! the direct sum `ψ ⊕ 0·Hψ`, which is still exactly `ψ`, so the step can
//! move into the directions `H` actually reaches.

use super::dmrg::{Env, TwoSite, two_site_vector, update_left, update_right};
use super::linalg::{matmul, matmul_add, thin_svd};
use super::mpo::{Mpo, MpoSite, pauli_entry};
use super::{Absorb, Mps, Tensor};
use crate::error::Error;
use crate::observable::PauliSum;
use crate::statevector::evolve::krylov_evolve;
use crate::statevector::krylov::LinearOperator;
use faer::{Mat, MatRef};
use num_complex::Complex64 as C64;

impl Mps {
    /// Evolve by `exp(-iHt)` with `steps` steps of two-site TDVP.
    ///
    /// Works for Hamiltonians of any range, with the error of each step
    /// third order in `t / steps` plus the truncation set by this state's
    /// [`options`](Self::options) (counted in
    /// [`truncation_error`](Self::truncation_error)). Norm and, without
    /// truncation, energy are conserved.
    ///
    /// While bonds are still growing (from a product state, say), each step
    /// first enlarges them with the directions of `H|ψ⟩` (see the
    /// [module docs](self)), so long-range couplings act from the first
    /// step. With bonds already large enough, steps are exact up to the
    /// Krylov tolerance.
    pub fn tdvp(&mut self, hamiltonian: &PauliSum, time: f64, steps: usize) -> Result<(), Error> {
        let n = self.n_qubits();
        if n < 2 {
            return Err(Error::InvalidArgument(
                "TDVP needs at least two qubits".into(),
            ));
        }
        if steps == 0 {
            return Err(Error::InvalidArgument(
                "TDVP needs at least one step".into(),
            ));
        }
        let mpo = Mpo::new(hamiltonian, n)?;
        let half = time / steps as f64 / 2.0;

        self.move_center(0);
        // left[i]: sites < i; right[i]: sites > i.
        let mut left: Vec<Option<Env>> = (0..n).map(|_| None).collect();
        let mut right: Vec<Option<Env>> = (0..n).map(|_| None).collect();
        left[0] = Some(Env::boundary());
        right[n - 1] = Some(Env::boundary());
        let mut stale = true;

        for _ in 0..steps {
            if self.bonds_growing(&mpo) {
                self.expand(&mpo);
                stale = true;
            }
            if stale {
                for i in (0..n - 1).rev() {
                    right[i] = Some(update_right(
                        right[i + 1].as_ref().unwrap(),
                        &self.tensors[i + 1],
                        &mpo.sites[i + 1],
                    ));
                }
                stale = false;
            }
            // Sweep right: the centre starts at site 0.
            for i in 0..n - 1 {
                self.evolve_pair(i, &mpo, &left, &right, half, Absorb::Right)?;
                left[i + 1] = Some(update_left(
                    left[i].as_ref().unwrap(),
                    &self.tensors[i],
                    &mpo.sites[i],
                ));
                if i + 1 < n - 1 {
                    self.evolve_site(i + 1, &mpo, &left, &right, -half)?;
                }
            }
            // Sweep left: the centre starts at site n − 1.
            for i in (0..n - 1).rev() {
                self.evolve_pair(i, &mpo, &left, &right, half, Absorb::Left)?;
                right[i] = Some(update_right(
                    right[i + 1].as_ref().unwrap(),
                    &self.tensors[i + 1],
                    &mpo.sites[i + 1],
                ));
                if i > 0 {
                    self.evolve_site(i, &mpo, &left, &right, -half)?;
                }
            }
        }
        if mpo.shift != 0.0 {
            let phase = C64::cis(-mpo.shift * time);
            self.tensors[self.center]
                .data
                .iter_mut()
                .for_each(|a| *a *= phase);
        }
        Ok(())
    }

    /// Whether some bond is below both its largest useful size and
    /// `max_bond`, and expanding it stays affordable.
    fn bonds_growing(&self, mpo: &Mpo) -> bool {
        let n = self.n_qubits();
        let bonds = self.bond_dimensions();
        let largest = bonds.iter().copied().max().unwrap_or(1);
        let width = mpo.sites.iter().map(|s| s.right).max().unwrap_or(1);
        largest * width <= 4096
            && bonds.iter().enumerate().any(|(i, &d)| {
                let full = 1usize << (i + 1).min(n - 1 - i).min(30);
                d < full.min(self.options.max_bond)
            })
    }

    /// Enlarge the right bases with those of `H|ψ⟩`, leaving the state the
    /// same and right-canonical with the centre at site 0.
    fn expand(&mut self, mpo: &Mpo) {
        let n = self.n_qubits();
        // Hψ site by site: φ[(l, a), p', (r, b)] = Σ W[a][b]_{p' p} A[l, p, r].
        let phi: Vec<Tensor> = self
            .tensors
            .iter()
            .zip(&mpo.sites)
            .map(|(t, w)| {
                let (left, right) = (t.left * w.left, t.right * w.right);
                let mut data = vec![C64::new(0.0, 0.0); left * 2 * right];
                for &(a, b, pauli, coeff) in &w.entries {
                    for p_out in 0..2 {
                        let (p_in, v) = pauli_entry(pauli, p_out);
                        let f = coeff * v;
                        for l in 0..t.left {
                            for r in 0..t.right {
                                let x = t.data[(l * 2 + p_in) * t.right + r];
                                let (ll, rr) = (l * w.left + a, r * w.right + b);
                                data[(ll * 2 + p_out) * right + rr] += f * x;
                            }
                        }
                    }
                }
                // Scaled to the size of ψ's tensor, so ψ dominates.
                let norm = |d: &[C64]| d.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
                let (np, nt) = (norm(&data), norm(&t.data));
                if np > 0.0 {
                    data.iter_mut().for_each(|x| *x *= nt / np);
                }
                Tensor { left, right, data }
            })
            .collect();
        // The direct sum ψ ⊕ ε·φ, with φ's block at site 0 zero so the
        // state is still ψ; ε keeps φ's directions below ψ's.
        const EPS: f64 = 1e-6;
        for (i, f) in phi.iter().enumerate() {
            let t = &self.tensors[i];
            let left = if i == 0 { 1 } else { t.left + f.left };
            let right = if i == n - 1 { 1 } else { t.right + f.right };
            let mut data = vec![C64::new(0.0, 0.0); left * 2 * right];
            for l in 0..t.left {
                for p in 0..2 {
                    for r in 0..t.right {
                        data[(l * 2 + p) * right + r] = t.data[(l * 2 + p) * t.right + r];
                    }
                }
            }
            if i > 0 {
                let (l0, r0) = (t.left, if i == n - 1 { 0 } else { t.right });
                let scale = if i == n - 1 { EPS } else { 1.0 };
                for l in 0..f.left {
                    for p in 0..2 {
                        for r in 0..f.right {
                            data[((l0 + l) * 2 + p) * right + r0 + r] =
                                scale * f.data[(l * 2 + p) * f.right + r];
                        }
                    }
                }
            }
            self.tensors[i] = Tensor { left, right, data };
        }
        // Right-canonicalise from the end, keeping every direction above
        // a tiny threshold, at most `max_bond` (and never fewer than ψ had).
        for i in (1..n).rev() {
            let right = self.tensors[i].right;
            let (u, s, v) = thin_svd(self.tensors[i].as_right_matrix());
            let top = s.first().copied().unwrap_or(0.0);
            let keep = s
                .iter()
                .take_while(|&&x| x > 1e-13 * top)
                .count()
                .clamp(1, self.options.max_bond.max(1));
            let vh = Mat::<C64>::from_fn(keep, 2 * right, |r, c| v[(c, r)].conj());
            self.tensors[i] = Tensor::from_matrix(vh.as_ref(), keep, right);
            let carry =
                Mat::<C64>::from_fn(u.nrows(), keep, |r, c| u[(r, c)] * C64::new(s[c], 0.0));
            let prev = &self.tensors[i - 1];
            let prev_left = prev.left;
            let merged = matmul(prev.as_left_matrix(), carry.as_ref());
            self.tensors[i - 1] = Tensor::from_matrix(merged.as_ref(), prev_left, keep);
        }
        self.center = 0;
    }

    /// Evolve sites `i` and `i + 1` by `exp(-iH t)` under their effective
    /// Hamiltonian and split them again, the centre going to `absorb`.
    fn evolve_pair(
        &mut self,
        i: usize,
        mpo: &Mpo,
        left: &[Option<Env>],
        right: &[Option<Env>],
        t: f64,
        absorb: Absorb,
    ) -> Result<(), Error> {
        let (l, r) = (self.tensors[i].left, self.tensors[i + 1].right);
        let mut x = two_site_vector(&self.tensors[i], &self.tensors[i + 1]);
        let op = TwoSite::new(
            left[i].as_ref().unwrap(),
            right[i + 1].as_ref().unwrap(),
            &mpo.sites[i],
            &mpo.sites[i + 1],
            l,
            r,
        );
        krylov_evolve(&op, &mut x, t, true)?;
        self.split(i, MatRef::from_row_major_slice(&x, l * 2, 2 * r), absorb);
        Ok(())
    }

    /// Evolve site `i`, the centre, by `exp(-iH t)` under its effective
    /// Hamiltonian.
    fn evolve_site(
        &mut self,
        i: usize,
        mpo: &Mpo,
        left: &[Option<Env>],
        right: &[Option<Env>],
        t: f64,
    ) -> Result<(), Error> {
        let op = OneSite {
            left: left[i].as_ref().unwrap(),
            right: right[i].as_ref().unwrap(),
            site: &mpo.sites[i],
            l: self.tensors[i].left,
            r: self.tensors[i].right,
        };
        krylov_evolve(&op, &mut self.tensors[i].data, t, true)
    }
}

/// The effective Hamiltonian on one site, between the environments of the
/// sites either side.
struct OneSite<'a> {
    left: &'a Env,
    right: &'a Env,
    site: &'a MpoSite,
    l: usize,
    r: usize,
}

impl LinearOperator<C64> for OneSite<'_> {
    fn apply(&self, x: &[C64], out: &mut [C64]) {
        let (l, r) = (self.l, self.r);
        let a = MatRef::from_row_major_slice(x, l, 2 * r);
        // y_a[l, (p, r')] = Σ_l' L_a[l, l'] A[l', p, r']
        let y: Vec<Mat<C64>> = self
            .left
            .blocks
            .iter()
            .map(|e| matmul(e.as_ref(), a))
            .collect();
        let mut z = vec![Mat::<C64>::zeros(l * 2, r); self.right.blocks.len()];
        for &(ai, bi, p, coeff) in &self.site.entries {
            for p_out in 0..2 {
                let (p_in, v) = pauli_entry(p, p_out);
                let f = coeff * v;
                for row in 0..l {
                    for col in 0..r {
                        z[bi][(2 * row + p_out, col)] += f * y[ai][(row, p_in * r + col)];
                    }
                }
            }
        }
        // out[(l, p), r] = Σ_b Σ_r' z_b[(l, p), r'] R_b[r, r']
        let mut result = Mat::<C64>::zeros(l * 2, r);
        for (zb, rb) in z.iter().zip(&self.right.blocks) {
            matmul_add(&mut result, zb.as_ref(), rb.transpose());
        }
        for row in 0..l * 2 {
            for col in 0..r {
                out[row * r + col] = result[(row, col)];
            }
        }
    }
}
