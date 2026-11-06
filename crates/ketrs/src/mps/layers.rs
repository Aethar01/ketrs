//! Circuits as layers of gates on disjoint bonds, applied in parallel.
//!
//! Mixed canonical form needs the orthogonality centre at each gate, so
//! gates run one at a time. [`Mps::run`] instead uses Hastings' form of
//! TEBD (Phys. Rev. B 79, 2009): every tensor `B` right-orthonormal, plus
//! the singular values `Λᵢ` of every bond. A gate `U` on bond `(i, i+1)`
//! then needs only its own two tensors and `Λᵢ₋₁`: with
//! `G = U · (Bᵢ Bᵢ₊₁)`, the SVD `Λᵢ₋₁ G = X Y Z†` gives the new
//! `Bᵢ₊₁ = Z†`, `Λᵢ = Y` and `Bᵢ = G Z`, with no division by singular
//! values. Gates on disjoint bonds touch disjoint data, so each layer of
//! them runs in parallel. Right-orthonormal tensors are already mixed
//! canonical with the centre at site 0, so leaving the form is free.

use super::linalg::{matmul, thin_svd};
use super::{Mps, Tensor, ZERO, decompose_pauli_rotation, swap_matrix};
use crate::circuit::Operation;
use crate::fusion::local_matrix;
use faer::Mat;
use num_complex::Complex64 as C64;
use rayon::prelude::*;

/// One step of a circuit on an MPS: a gate on one site, or on a bond.
pub(super) enum Step {
    /// A 2×2 matrix on a site.
    Site(usize, [[C64; 2]; 2]),
    /// A 4×4 matrix on sites `(i, i + 1)`, with bit 0 for site `i`.
    Bond(usize, [[C64; 4]; 4]),
}

impl Step {
    fn sites(&self) -> (usize, usize) {
        match *self {
            Step::Site(q, _) => (q, q),
            Step::Bond(i, _) => (i, i + 1),
        }
    }
}

/// `op` as steps on sites and neighbouring bonds: gates on distant qubits
/// are routed with swaps, and Pauli rotations on more than two qubits and
/// multi-controlled operations are decomposed exactly.
pub(super) fn steps(op: &Operation, out: &mut Vec<Step>) {
    match local_matrix(op) {
        Some((qubits, m)) if qubits.len() == 1 => {
            out.push(Step::Site(
                qubits[0],
                [[m[0][0], m[0][1]], [m[1][0], m[1][1]]],
            ));
        }
        Some((qubits, m)) => {
            let m: [[C64; 4]; 4] = std::array::from_fn(|r| std::array::from_fn(|c| m[r][c]));
            let (a, b) = (qubits[0], qubits[1]);
            // Put the lower qubit on bit 0.
            let (lo, hi, m) = if a < b {
                (a, b, m)
            } else {
                const P: [usize; 4] = [0, 2, 1, 3];
                (
                    b,
                    a,
                    std::array::from_fn(|r| std::array::from_fn(|c| m[P[r]][P[c]])),
                )
            };
            // Bring qubit `hi` next to `lo`, apply, and move it back.
            for s in (lo + 1..hi).rev() {
                out.push(Step::Bond(s, swap_matrix()));
            }
            out.push(Step::Bond(lo, m));
            for s in lo + 1..hi {
                out.push(Step::Bond(s, swap_matrix()));
            }
        }
        None => {
            let parts = match op {
                Operation::PauliRotation { pauli, theta } => {
                    decompose_pauli_rotation(pauli, *theta)
                }
                _ => {
                    let mut parts = Vec::new();
                    op.decompose(&mut parts);
                    parts
                }
            };
            for part in parts {
                steps(&part, out);
            }
        }
    }
}

/// `G[(l, p'), (q', r)] = Σ m[p' + 2q'][p + 2q] θ[(l, p), (q, r)]`.
pub(super) fn apply_bond_matrix(theta: &Mat<C64>, m: &[[C64; 4]; 4], right: usize) -> Mat<C64> {
    Mat::<C64>::from_fn(theta.nrows(), theta.ncols(), |row, col| {
        let (l, p_out) = (row / 2, row % 2);
        let (q_out, r) = (col / right, col % right);
        let out = p_out + 2 * q_out;
        let mut acc = ZERO;
        for p in 0..2 {
            for q in 0..2 {
                let g = m[out][p + 2 * q];
                if g != ZERO {
                    acc += g * theta[(2 * l + p, q * right + r)];
                }
            }
        }
        acc
    })
}

/// A bond update's result: the bond's new tensors, singular values and
/// discarded weight.
struct Updated {
    bond: usize,
    left: Tensor,
    right: Tensor,
    lambda: Vec<f64>,
    discarded: f64,
}

impl Mps {
    /// Apply `steps` in layers of gates on disjoint sites, each layer in
    /// parallel. Leaves the state right-orthonormal with the centre at 0.
    pub(super) fn run_layers(&mut self, steps: &[Step]) {
        // Each step goes in the layer after the last one touching its
        // sites, as early as its dependencies allow; steps in one layer act
        // on disjoint sites, so commute.
        let mut layers: Vec<Vec<&Step>> = Vec::new();
        let mut next = vec![0; self.n_qubits()];
        for step in steps {
            let (a, b) = step.sites();
            let layer = next[a].max(next[b]);
            if layer == layers.len() {
                layers.push(Vec::new());
            }
            layers[layer].push(step);
            next[a] = layer + 1;
            next[b] = layer + 1;
        }
        let mut lambda = self.right_canonical_bonds();
        for layer in &layers {
            self.apply_layer(layer, &mut lambda);
        }
        self.center = 0;
    }

    /// Apply steps on disjoint sites, with `lambda[i]` the singular values
    /// of bond `i`.
    fn apply_layer(&mut self, layer: &[&Step], lambda: &mut [Vec<f64>]) {
        let bonds: Vec<(usize, &[[C64; 4]; 4])> = layer
            .iter()
            .filter_map(|step| match *step {
                Step::Bond(i, m) => Some((*i, m)),
                Step::Site(..) => None,
            })
            .collect();
        let one = [1.0];
        let update = |&(i, m): &(usize, &[[C64; 4]; 4])| {
            let before: &[f64] = if i == 0 { &one } else { &lambda[i - 1] };
            self.update_bond(i, m, before)
        };
        // Bond updates are small dense SVDs; only spread them over threads
        // when there's enough work.
        let work: usize = bonds
            .iter()
            .map(|&(i, _)| {
                let (l, r) = (self.tensors[i].left, self.tensors[i + 1].right);
                (l * r).max(1).pow(2)
            })
            .sum();
        let updated: Vec<Updated> = if bonds.len() > 1 && work >= PARALLEL_WORK {
            bonds.par_iter().map(update).collect()
        } else {
            bonds.iter().map(update).collect()
        };
        for u in updated {
            self.tensors[u.bond] = u.left;
            self.tensors[u.bond + 1] = u.right;
            lambda[u.bond] = u.lambda;
            self.truncation_error += u.discarded;
        }
        for &step in layer {
            if let Step::Site(q, m) = step {
                self.apply_1q(*q, *m);
            }
        }
    }

    /// Hastings' update of bond `i` by `m`, given the singular values
    /// `before` of bond `i - 1` (`[1]` for the first bond).
    fn update_bond(&self, i: usize, m: &[[C64; 4]; 4], before: &[f64]) -> Updated {
        let (a, b) = (&self.tensors[i], &self.tensors[i + 1]);
        let (left, right) = (a.left, b.right);
        let theta = matmul(a.as_left_matrix(), b.as_right_matrix());
        let g = apply_bond_matrix(&theta, m, right);
        let weighted = Mat::<C64>::from_fn(g.nrows(), g.ncols(), |r, c| g[(r, c)] * before[r / 2]);
        let (_, s, z) = thin_svd(weighted.as_ref());
        let total: f64 = s.iter().map(|x| x * x).sum();
        let keep = self.bond_to_keep(&s, total);
        let kept: f64 = s[..keep].iter().map(|x| x * x).sum();
        let rescale = if kept > 0.0 {
            (total / kept).sqrt()
        } else {
            1.0
        };
        let z = z.subcols(0, keep);
        let new_left = matmul(g.as_ref(), z);
        let new_left =
            Mat::<C64>::from_fn(new_left.nrows(), keep, |r, c| new_left[(r, c)] * rescale);
        let new_right = z.adjoint().to_owned();
        Updated {
            bond: i,
            left: Tensor::from_matrix(new_left.as_ref(), left, keep),
            right: Tensor::from_matrix(new_right.as_ref(), keep, right),
            lambda: s[..keep].iter().map(|x| x * rescale).collect(),
            discarded: if total > 0.0 {
                (total - kept) / total
            } else {
                0.0
            },
        }
    }

    /// Make every tensor right-orthonormal and return each bond's singular
    /// values, rotating each bond to its Schmidt basis.
    fn right_canonical_bonds(&mut self) -> Vec<Vec<f64>> {
        self.move_center(0);
        let n = self.n_qubits();
        let mut lambda: Vec<Vec<f64>> = Vec::with_capacity(n.saturating_sub(1));
        for i in 0..n.saturating_sub(1) {
            let t = &self.tensors[i];
            let right = t.right;
            let before: &[f64] = if i == 0 { &[1.0] } else { &lambda[i - 1] };
            // Λᵢ₋₁ Bᵢ = U S V†: rotating bond i by V makes it diagonal.
            let weighted = Mat::<C64>::from_fn(t.left * 2, right, |r, c| {
                t.data[r * right + c] * before[r / 2]
            });
            let (_, s, v) = thin_svd(weighted.as_ref());
            // Directions with no weight carry nothing; drop them.
            let keep = s.iter().take_while(|&&x| x > s[0] * 1e-15).count().max(1);
            let v = v.subcols(0, keep);
            let rotated = matmul(t.as_left_matrix(), v);
            let next = &self.tensors[i + 1];
            let next_right = next.right;
            let next_rotated = matmul(v.adjoint(), next.as_right_matrix());
            let left = t.left;
            self.tensors[i] = Tensor::from_matrix(rotated.as_ref(), left, keep);
            self.tensors[i + 1] = Tensor::from_matrix(next_rotated.as_ref(), keep, next_right);
            lambda.push(s[..keep].to_vec());
        }
        lambda
    }
}

/// Least total work (Σ (χ_left χ_right)² over a layer's bonds) worth
/// spreading over threads.
const PARALLEL_WORK: usize = 1 << 14;
