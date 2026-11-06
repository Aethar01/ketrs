//! Ground states by two-site DMRG.
//!
//! The Hamiltonian is turned into an MPO, and left and right environments
//! (the MPO sandwiched between the MPS and its conjugate, contracted up to a
//! pair of sites) are kept for every bond. Each step solves for the lowest
//! eigenvector of the effective Hamiltonian on two neighbouring sites by
//! Lanczos, starting from the current two-site tensor, then splits it by
//! truncated SVD and moves on. Sweeps go right and then left until the
//! energy stops changing.

use super::linalg::{matmul, matmul_add};
use super::mpo::{Mpo, MpoSite, pauli_entry};
use super::{Absorb, Mps, MpsOptions, Tensor};
use crate::error::Error;
use crate::observable::{Pauli, PauliSum};
use crate::statevector::eigen::{EigenOptions, lanczos_lowest_from};
use crate::statevector::krylov::{LinearOperator, dot, norm};
use faer::{Mat, MatRef};
use num_complex::Complex64 as C64;

const ZERO: C64 = C64::new(0.0, 0.0);

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
}

impl Default for DmrgOptions {
    fn default() -> Self {
        DmrgOptions {
            max_bond: 64,
            cutoff: 1e-12,
            max_sweeps: 20,
            tolerance: 1e-10,
            lanczos_tolerance: 1e-10,
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
    /// Whether the energy settled within `tolerance` before `max_sweeps`.
    pub converged: bool,
}

/// An environment: for each MPO bond index, a matrix with rows on the bra
/// bond and columns on the ket bond.
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
        let mpo = Mpo::new(hamiltonian, n)?;
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

        // left[i]: sites < i; right[i]: sites > i.
        let mut left: Vec<Option<Env>> = (0..n).map(|_| None).collect();
        let mut right: Vec<Option<Env>> = (0..n).map(|_| None).collect();
        left[0] = Some(Env::boundary());
        right[n - 1] = Some(Env::boundary());
        for i in (0..n - 1).rev() {
            right[i] = Some(update_right(
                right[i + 1].as_ref().unwrap(),
                &self.tensors[i + 1],
                &mpo.sites[i + 1],
            ));
        }

        let mut sweep_energies = Vec::new();
        let mut energy = f64::INFINITY;
        let mut converged = false;
        for _ in 0..options.max_sweeps {
            for i in 0..n - 1 {
                energy = self.optimise_pair(i, &mpo, &left, &right, &lanczos, Absorb::Right)?;
                left[i + 1] = Some(update_left(
                    left[i].as_ref().unwrap(),
                    &self.tensors[i],
                    &mpo.sites[i],
                ));
            }
            for i in (0..n - 1).rev() {
                energy = self.optimise_pair(i, &mpo, &left, &right, &lanczos, Absorb::Left)?;
                right[i] = Some(update_right(
                    right[i + 1].as_ref().unwrap(),
                    &self.tensors[i + 1],
                    &mpo.sites[i + 1],
                ));
            }
            let total = energy + mpo.shift;
            let change = sweep_energies
                .last()
                .map_or(f64::INFINITY, |e: &f64| (e - total).abs());
            sweep_energies.push(total);
            if change < options.tolerance {
                converged = true;
                break;
            }
        }
        Ok(DmrgResult {
            energy: energy + mpo.shift,
            sweep_energies,
            converged,
        })
    }

    /// Replace sites `i` and `i + 1` with the lowest eigenvector of the
    /// effective Hamiltonian there. Returns its energy (without the shift).
    fn optimise_pair(
        &mut self,
        i: usize,
        mpo: &Mpo,
        left: &[Option<Env>],
        right: &[Option<Env>],
        lanczos: &EigenOptions,
        absorb: Absorb,
    ) -> Result<f64, Error> {
        let (l, r) = (self.tensors[i].left, self.tensors[i + 1].right);
        let mut v0 = two_site_vector(&self.tensors[i], &self.tensors[i + 1]);
        let scale = 1.0 / norm(&v0);
        v0.iter_mut().for_each(|a| *a *= scale);

        let op = TwoSite::new(
            left[i].as_ref().unwrap(),
            right[i + 1].as_ref().unwrap(),
            &mpo.sites[i],
            &mpo.sites[i + 1],
            l,
            r,
        );
        let x = lanczos_lowest_from(&op, &v0, &[], lanczos)?;
        let mut hx = vec![ZERO; x.len()];
        op.apply(&x, &mut hx);
        let energy = dot(&x, &hx).re;
        self.split(i, MatRef::from_row_major_slice(&x, l * 2, 2 * r), absorb);
        Ok(energy)
    }
}

/// Two neighbouring sites contracted, row-major `(l, p, q, r)` to match
/// [`TwoSite`].
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
    let (l, r) = (a.left, a.right);
    // x_a[l, (p, r')] = Σ_l' env_a[l, l'] A[l', p, r']
    let x: Vec<Mat<C64>> = env
        .blocks
        .iter()
        .map(|e| matmul(e.as_ref(), a.as_right_matrix()))
        .collect();
    let mut y = vec![Mat::<C64>::zeros(l * 2, r); w.right];
    for &(ai, bi, p, coeff) in &w.entries {
        for p_out in 0..2 {
            let (p_in, val) = pauli_entry(p, p_out);
            let f = coeff * val;
            for row in 0..l {
                for col in 0..r {
                    y[bi][(2 * row + p_out, col)] += f * x[ai][(row, p_in * r + col)];
                }
            }
        }
    }
    let bra = a.as_left_matrix().adjoint();
    Env {
        blocks: y.iter().map(|yb| matmul(bra, yb.as_ref())).collect(),
    }
}

/// The environment of sites `> i − 1` from that of sites `> i`, site `i`'s
/// tensor and its MPO site.
pub(super) fn update_right(env: &Env, a: &Tensor, w: &MpoSite) -> Env {
    let (l, r) = (a.left, a.right);
    // x_c[(l', p), r] = Σ_r' A[l', p, r'] env_c[r, r']
    let x: Vec<Mat<C64>> = env
        .blocks
        .iter()
        .map(|e| matmul(a.as_left_matrix(), e.transpose()))
        .collect();
    let mut y = vec![Mat::<C64>::zeros(l, 2 * r); w.left];
    for &(ai, ci, p, coeff) in &w.entries {
        for p_out in 0..2 {
            let (p_in, val) = pauli_entry(p, p_out);
            let f = coeff * val;
            for row in 0..l {
                for col in 0..r {
                    y[ai][(row, p_out * r + col)] += f * x[ci][(2 * row + p_in, col)];
                }
            }
        }
    }
    let bra = a.as_right_matrix().conjugate();
    Env {
        blocks: y.iter().map(|ya| matmul(bra, ya.transpose())).collect(),
    }
}

/// The effective Hamiltonian on two neighbouring sites.
pub(super) struct TwoSite<'a> {
    left: &'a Env,
    right: &'a Env,
    /// `(a, c, P₁, P₂, coeff)`: the two MPO sites contracted over their
    /// shared bond.
    entries: Vec<(usize, usize, Pauli, Pauli, C64)>,
    l: usize,
    r: usize,
}

impl<'a> TwoSite<'a> {
    pub(super) fn new(
        left: &'a Env,
        right: &'a Env,
        w1: &MpoSite,
        w2: &MpoSite,
        l: usize,
        r: usize,
    ) -> Self {
        let mut by_left = vec![Vec::new(); w2.left];
        for &(b, c, p, coeff) in &w2.entries {
            by_left[b].push((c, p, coeff));
        }
        let mut entries = Vec::new();
        for &(a, b, p1, c1) in &w1.entries {
            for &(c, p2, c2) in &by_left[b] {
                entries.push((a, c, p1, p2, c1 * c2));
            }
        }
        TwoSite {
            left,
            right,
            entries,
            l,
            r,
        }
    }
}

impl LinearOperator<C64> for TwoSite<'_> {
    fn apply(&self, x: &[C64], out: &mut [C64]) {
        let (l, r) = (self.l, self.r);
        let theta = MatRef::from_row_major_slice(x, l, 4 * r);
        // y_a[l, (p, q, r')] = Σ_l' L_a[l, l'] θ[l', p, q, r']
        let y: Vec<Mat<C64>> = self
            .left
            .blocks
            .iter()
            .map(|e| matmul(e.as_ref(), theta))
            .collect();
        let mut z = vec![Mat::<C64>::zeros(l * 4, r); self.right.blocks.len()];
        for &(a, c, p1, p2, coeff) in &self.entries {
            for p_out in 0..2 {
                let (p_in, v1) = pauli_entry(p1, p_out);
                for q_out in 0..2 {
                    let (q_in, v2) = pauli_entry(p2, q_out);
                    let f = coeff * v1 * v2;
                    let (out_row, in_col) = (2 * p_out + q_out, (2 * p_in + q_in) * r);
                    for row in 0..l {
                        for col in 0..r {
                            z[c][(4 * row + out_row, col)] += f * y[a][(row, in_col + col)];
                        }
                    }
                }
            }
        }
        // out[(l, p, q), r] = Σ_c Σ_r' z_c[(l, p, q), r'] R_c[r, r']
        let mut result = Mat::<C64>::zeros(l * 4, r);
        for (zc, rc) in z.iter().zip(&self.right.blocks) {
            matmul_add(&mut result, zc.as_ref(), rc.transpose());
        }
        for row in 0..l * 4 {
            for col in 0..r {
                out[row * r + col] = result[(row, col)];
            }
        }
    }
}
