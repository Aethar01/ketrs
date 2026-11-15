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
//! runs a global subspace expansion (see [`expand`](super::expand)) before
//! each half-sweep: the bases that half-sweep projects onto (right bases
//! for the sweep right, left bases for the sweep left) are enlarged with
//! the directions `H|ψ⟩` reaches there, at zero weight, so the step can
//! move into them. The expansion holds `H|ψ⟩` but not `H²|ψ⟩`, so a step
//! that starts from a product state still makes an `O(dt²)` error;
//! overall the integrator stays second order.

use super::dmrg::two_site_vector;
use super::envs::Envs;
use super::mpo::Mpo;
use super::{Absorb, Mps};
use crate::error::Error;
use crate::observable::PauliSum;
use faer::MatRef;
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
    /// first enlarges them with the directions of `H|ψ⟩` (a subspace
    /// expansion), so long-range and many-qubit couplings act from the
    /// first step. With bonds already large enough, steps are exact up to
    /// the Krylov tolerance.
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
        let mpo = self.mpo(hamiltonian)?;
        let half = time / steps as f64 / 2.0;

        self.move_center(0);
        let mut envs = Envs::new(self, &mpo)?;
        let mut stale = true;
        // The largest each bond can be, within the sector for a symmetric MPS.
        let caps = match self.symmetry {
            Some(_) => self.sector_caps()[1..n].to_vec(),
            None => (0..n - 1)
                .map(|i| (1usize << (i + 1).min(n - 1 - i).min(30)).min(self.options.max_bond))
                .collect(),
        };

        for _ in 0..steps {
            let growing = self.bond_dimensions().iter().zip(&caps).any(|(d, c)| d < c);
            if growing {
                self.expand_right_bases(&mpo, &envs)?;
                stale = true;
            }
            if stale {
                for i in (0..n - 1).rev() {
                    envs.update_right(i, self, &mpo)?;
                }
                stale = false;
            }
            // Sweep right: the centre starts at site 0.
            for i in 0..n - 1 {
                self.evolve_pair(i, &mpo, &envs, half, Absorb::Right)?;
                envs.update_left(i, self, &mpo)?;
                if i + 1 < n - 1 {
                    self.evolve_site(i + 1, &mpo, &envs, -half)?;
                }
                // Rebuilt from the right end before it is needed again.
                envs.release_right(i + 1);
            }
            // Sweep left: the centre starts at site n − 1.
            if growing {
                self.expand_left_bases(&mpo, &envs)?;
                for i in 0..n - 1 {
                    envs.update_left(i, self, &mpo)?;
                }
            }
            for i in (0..n - 1).rev() {
                self.evolve_pair(i, &mpo, &envs, half, Absorb::Left)?;
                envs.update_right(i, self, &mpo)?;
                if i > 0 {
                    self.evolve_site(i, &mpo, &envs, -half)?;
                }
                envs.release_left(i);
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

    /// Evolve sites `i` and `i + 1` by `exp(-iH t)` under their effective
    /// Hamiltonian and split them again, the centre going to `absorb`.
    fn evolve_pair(
        &mut self,
        i: usize,
        mpo: &Mpo,
        envs: &Envs,
        t: f64,
        absorb: Absorb,
    ) -> Result<(), Error> {
        let (l, r) = (self.tensors[i].left, self.tensors[i + 1].right);
        let op = envs.two_site(i, self, mpo);
        let mut x = op.pack(two_site_vector(&self.tensors[i], &self.tensors[i + 1]));
        let evolved = op.evolve(&mut x, t);
        envs.check()?;
        evolved?;
        let x = op.unpack(x);
        self.split(i, MatRef::from_row_major_slice(&x, l * 2, 2 * r), absorb);
        Ok(())
    }

    /// Evolve site `i`, the centre, by `exp(-iH t)` under its effective
    /// Hamiltonian.
    fn evolve_site(&mut self, i: usize, mpo: &Mpo, envs: &Envs, t: f64) -> Result<(), Error> {
        let op = envs.one_site(i, self, mpo);
        let mut x = op.pack(std::mem::take(&mut self.tensors[i].data));
        let evolved = op.evolve(&mut x, t);
        self.tensors[i].data = op.unpack(x);
        envs.check()?;
        evolved
    }
}
