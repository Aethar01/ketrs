//! Matrix product state (MPS) simulation.
//!
//! The state is a chain of tensors `A[q]` of shape `(χ_left, 2, χ_right)`,
//! one per qubit, with amplitudes `ψ(b₀ … bₙ₋₁) = A[0][b₀] ⋯ A[n−1][bₙ₋₁]`.
//! Memory and time grow with the bond dimension χ rather than `2ⁿ`, so
//! weakly entangled states of hundreds of qubits are cheap, and capping χ
//! trades accuracy for speed in a controlled way.
//!
//! The chain is kept in mixed canonical form around an orthogonality centre:
//! tensors to its left are left-orthonormal and to its right
//! right-orthonormal. A two-qubit gate on neighbours moves the centre there
//! by QR steps, contracts both tensors with the gate, and splits them again
//! by SVD, keeping at most `max_bond` singular values and dropping any whose
//! discarded weight is below `cutoff`. Gates on distant qubits are routed
//! with swaps, and Pauli rotations on more than two qubits are decomposed
//! exactly. Running a Trotter circuit is therefore TEBD; circuits are fused
//! first, so each neighbouring pair costs one SVD per layer, and gates on
//! disjoint bonds run in parallel, in Hastings' form of TEBD.

mod dmrg;
mod layers;
mod linalg;
mod mpo;
mod tdvp;

use linalg::{matmul, thin_svd};

pub use dmrg::{DmrgOptions, DmrgResult};

use crate::circuit::{Circuit, Operation};
use crate::error::Error;
use crate::gate::Gate;
use crate::observable::{Pauli, PauliString, PauliSum};
use crate::statevector::shots::pack;
use crate::trajectory::{self, Trajectory};
use faer::{Mat, MatRef};
use num_complex::Complex64 as C64;
use rand::Rng;
use std::collections::BTreeMap;

const ZERO: C64 = C64::new(0.0, 0.0);
const ONE: C64 = C64::new(1.0, 0.0);

/// Which side of a split takes the singular values (and the centre).
#[derive(Clone, Copy)]
enum Absorb {
    Left,
    Right,
}

/// Truncation settings for [`Mps`].
#[derive(Clone, Copy, Debug)]
pub struct MpsOptions {
    /// Most singular values kept on any bond.
    pub max_bond: usize,
    /// Singular values are dropped while the discarded weight
    /// `Σ s²_dropped / Σ s²` stays at most this.
    pub cutoff: f64,
}

impl Default for MpsOptions {
    fn default() -> Self {
        MpsOptions {
            max_bond: 64,
            cutoff: 1e-14,
        }
    }
}

/// One site tensor, stored row-major as `data[(l·2 + p)·right + r]`.
#[derive(Clone, Debug)]
struct Tensor {
    left: usize,
    right: usize,
    data: Vec<C64>,
}

impl Tensor {
    /// The tensor as a `(left·2) × right` matrix.
    fn as_left_matrix(&self) -> MatRef<'_, C64> {
        MatRef::from_row_major_slice(&self.data, self.left * 2, self.right)
    }

    /// The tensor as a `left × (2·right)` matrix.
    fn as_right_matrix(&self) -> MatRef<'_, C64> {
        MatRef::from_row_major_slice(&self.data, self.left, 2 * self.right)
    }

    fn from_matrix(m: MatRef<'_, C64>, left: usize, right: usize) -> Tensor {
        debug_assert_eq!(m.nrows() * m.ncols(), left * 2 * right);
        let mut data = Vec::with_capacity(m.nrows() * m.ncols());
        for i in 0..m.nrows() {
            for j in 0..m.ncols() {
                data.push(m[(i, j)]);
            }
        }
        Tensor { left, right, data }
    }
}

/// A pure state of `n` qubits as a matrix product state.
#[derive(Clone, Debug)]
pub struct Mps {
    tensors: Vec<Tensor>,
    /// The orthogonality centre.
    center: usize,
    options: MpsOptions,
    /// Total discarded weight from truncations so far: roughly the lost
    /// fidelity, `1 − |⟨ψ_exact|ψ⟩|²`, while it is small.
    truncation_error: f64,
}

impl Mps {
    /// The all-zero state on `n_qubits` qubits.
    pub fn zero(n_qubits: usize, options: MpsOptions) -> Self {
        Self::basis(n_qubits, &vec![false; n_qubits], options)
    }

    /// The product state with qubit `q` in `|bits[q]⟩`.
    pub fn basis(n_qubits: usize, bits: &[bool], options: MpsOptions) -> Self {
        assert_eq!(bits.len(), n_qubits, "one bit per qubit");
        assert!(n_qubits > 0, "an MPS needs at least one qubit");
        assert!(options.max_bond > 0, "max_bond must be at least 1");
        let tensors = bits
            .iter()
            .map(|&b| Tensor {
                left: 1,
                right: 1,
                data: if b { vec![ZERO, ONE] } else { vec![ONE, ZERO] },
            })
            .collect();
        Mps {
            tensors,
            center: 0,
            options,
            truncation_error: 0.0,
        }
    }

    /// A random normalised MPS with bonds of up to `bond` (capped by what
    /// the qubits can support), in canonical form. A generic starting point
    /// for [`dmrg`](Self::dmrg).
    pub fn random(n_qubits: usize, bond: usize, seed: u64, options: MpsOptions) -> Self {
        assert!(n_qubits > 0 && bond > 0, "need at least one qubit and bond");
        let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
        let mut uniform = move || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 52) as f64 - 1.0
        };
        // Bond q (between q and q+1) can't exceed 2^min(q+1, n-q-1).
        let cap = |q: usize| -> usize {
            let side = (q + 1).min(n_qubits - q - 1).min(30);
            bond.min(1 << side)
        };
        let tensors = (0..n_qubits)
            .map(|q| {
                let left = if q == 0 { 1 } else { cap(q - 1) };
                let right = if q + 1 == n_qubits { 1 } else { cap(q) };
                let data = (0..left * 2 * right)
                    .map(|_| C64::new(uniform(), uniform()))
                    .collect();
                Tensor { left, right, data }
            })
            .collect();
        let mut mps = Mps {
            tensors,
            center: n_qubits - 1,
            options,
            truncation_error: 0.0,
        };
        mps.move_center(0);
        let norm = mps.norm_sqr().sqrt();
        mps.tensors[0].data.iter_mut().for_each(|a| *a /= norm);
        mps
    }

    /// Run `circuit` on `|0…0⟩`.
    pub fn from_circuit(circuit: &Circuit, options: MpsOptions) -> Result<Self, Error> {
        let mut mps = Self::zero(circuit.n_qubits(), options);
        mps.run(circuit)?;
        Ok(mps)
    }

    /// Number of qubits.
    pub fn n_qubits(&self) -> usize {
        self.tensors.len()
    }

    /// The truncation settings.
    pub fn options(&self) -> MpsOptions {
        self.options
    }

    /// Change the truncation settings for later gates.
    pub fn set_options(&mut self, options: MpsOptions) {
        assert!(options.max_bond > 0, "max_bond must be at least 1");
        self.options = options;
    }

    /// Bond dimension between each pair of neighbouring qubits.
    pub fn bond_dimensions(&self) -> Vec<usize> {
        self.tensors[..self.tensors.len() - 1]
            .iter()
            .map(|t| t.right)
            .collect()
    }

    /// Total discarded weight from truncation so far.
    pub fn truncation_error(&self) -> f64 {
        self.truncation_error
    }

    /// `⟨ψ|ψ⟩`, the squared norm of the centre tensor.
    pub fn norm_sqr(&self) -> f64 {
        self.tensors[self.center]
            .data
            .iter()
            .map(|a| a.norm_sqr())
            .sum()
    }

    /// Apply every operation in `circuit`, which must have the same width
    /// and no unbound parameters.
    pub fn run(&mut self, circuit: &Circuit) -> Result<(), Error> {
        circuit.check_unitary()?;
        circuit.check_no_multi_controlled("the MPS engine")?;
        if circuit.n_qubits() != self.n_qubits() {
            return Err(Error::QubitCountMismatch {
                expected: self.n_qubits(),
                found: circuit.n_qubits(),
            });
        }
        if circuit.n_parameters() > 0 {
            return Err(Error::InvalidArgument(format!(
                "circuit has {} unbound parameters; call bind first",
                circuit.n_parameters()
            )));
        }
        let mut steps = Vec::new();
        for op in circuit.fused().ops() {
            layers::steps(op, &mut steps);
        }
        self.run_layers(&steps);
        if circuit.global_phase() != 0.0 {
            let phase = C64::cis(circuit.global_phase());
            self.tensors[self.center]
                .data
                .iter_mut()
                .for_each(|a| *a *= phase);
        }
        Ok(())
    }

    /// Apply one operation.
    pub fn apply(&mut self, op: &Operation) -> Result<(), Error> {
        op.validate(self.n_qubits())?;
        if op.is_multi_controlled() {
            return Err(Error::InvalidArgument(
                "the MPS engine doesn't support multi-controlled operations yet".into(),
            ));
        }
        self.apply_unchecked(op);
        Ok(())
    }

    fn apply_unchecked(&mut self, op: &Operation) {
        let mut steps = Vec::new();
        layers::steps(op, &mut steps);
        for step in steps {
            match step {
                layers::Step::Site(q, m) => self.apply_1q(q, m),
                layers::Step::Bond(i, m) => self.apply_adjacent(i, &m),
            }
        }
    }

    /// Apply a single-qubit matrix. Unitary on the physical index, so the
    /// canonical form is unchanged.
    fn apply_1q(&mut self, q: usize, m: [[C64; 2]; 2]) {
        let t = &mut self.tensors[q];
        let r = t.right;
        for l in 0..t.left {
            for k in 0..r {
                let (i0, i1) = (2 * l * r + k, (2 * l + 1) * r + k);
                let (a0, a1) = (t.data[i0], t.data[i1]);
                t.data[i0] = m[0][0] * a0 + m[0][1] * a1;
                t.data[i1] = m[1][0] * a0 + m[1][1] * a1;
            }
        }
    }

    /// Apply `m` (bit 0 = site `i`, bit 1 = site `i+1`) to neighbouring sites
    /// by contraction and truncated SVD. Leaves the centre at `i + 1`.
    fn apply_adjacent(&mut self, i: usize, m: &[[C64; 4]; 4]) {
        self.move_center(i);
        let right = self.tensors[i + 1].right;
        let theta = matmul(
            self.tensors[i].as_left_matrix(),
            self.tensors[i + 1].as_right_matrix(),
        );
        let gated = layers::apply_bond_matrix(&theta, m, right);
        self.split(i, gated.as_ref(), Absorb::Right);
    }

    /// Split a two-site tensor `theta` (rows `(l, p)`, columns `(q, r)`) back
    /// into sites `i` and `i + 1` by truncated SVD. The singular values go
    /// to the side named by `absorb`, which becomes the centre.
    fn split(&mut self, i: usize, theta: MatRef<'_, C64>, absorb: Absorb) {
        let (left, right) = (theta.nrows() / 2, theta.ncols() / 2);
        let (u, s, v) = thin_svd(theta);
        let total: f64 = s.iter().map(|x| x * x).sum();
        let keep = self.bond_to_keep(&s, total);
        let kept: f64 = s[..keep].iter().map(|x| x * x).sum();
        if total > 0.0 {
            self.truncation_error += (total - kept) / total;
        }
        // Keep the norm: rescale the kept singular values.
        let rescale = if kept > 0.0 {
            (total / kept).sqrt()
        } else {
            1.0
        };
        let weight = |k: usize| C64::new(s[k] * rescale, 0.0);

        let (s_left, s_right) = match absorb {
            Absorb::Left => (true, false),
            Absorb::Right => (false, true),
        };
        let left_tensor = Mat::<C64>::from_fn(left * 2, keep, |r, c| {
            if s_left {
                u[(r, c)] * weight(c)
            } else {
                u[(r, c)]
            }
        });
        let right_tensor = Mat::<C64>::from_fn(keep, 2 * right, |r, c| {
            let vh = v[(c, r)].conj();
            if s_right { weight(r) * vh } else { vh }
        });
        self.tensors[i] = Tensor::from_matrix(left_tensor.as_ref(), left, keep);
        self.tensors[i + 1] = Tensor::from_matrix(right_tensor.as_ref(), keep, right);
        self.center = if s_left { i } else { i + 1 };
    }

    /// How many singular values to keep: as few as the cutoff allows, at
    /// most `max_bond`, at least one.
    fn bond_to_keep(&self, s: &[f64], total: f64) -> usize {
        let mut keep = s.len();
        let mut discarded = 0.0;
        while keep > 1 {
            let next = discarded + s[keep - 1] * s[keep - 1];
            if next > self.options.cutoff * total {
                break;
            }
            discarded = next;
            keep -= 1;
        }
        keep.min(self.options.max_bond).max(1)
    }

    /// Move the orthogonality centre to `target`, one site at a time, by
    /// SVD: the site keeps the orthonormal factor and the rest moves on.
    fn move_center(&mut self, target: usize) {
        while self.center < target {
            let i = self.center;
            let left = self.tensors[i].left;
            // A = U (S V†): keep U, carry S V† right.
            let (u, s, v) = thin_svd(self.tensors[i].as_left_matrix());
            let k = s.len();
            let carry =
                Mat::<C64>::from_fn(k, v.nrows(), |r, c| C64::new(s[r], 0.0) * v[(c, r)].conj());
            self.tensors[i] = Tensor::from_matrix(u.as_ref(), left, k);
            let next = &self.tensors[i + 1];
            let next_right = next.right;
            let merged = matmul(carry.as_ref(), next.as_right_matrix());
            self.tensors[i + 1] = Tensor::from_matrix(merged.as_ref(), k, next_right);
            self.center += 1;
        }
        while self.center > target {
            let i = self.center;
            let right = self.tensors[i].right;
            // A = (U S) V†: keep V†, carry U S left.
            let (u, s, v) = thin_svd(self.tensors[i].as_right_matrix());
            let k = s.len();
            let carry = Mat::<C64>::from_fn(u.nrows(), k, |r, c| u[(r, c)] * C64::new(s[c], 0.0));
            self.tensors[i] = Tensor::from_matrix(v.adjoint().to_owned().as_ref(), k, right);
            let prev = &self.tensors[i - 1];
            let prev_left = prev.left;
            let merged = matmul(prev.as_left_matrix(), carry.as_ref());
            self.tensors[i - 1] = Tensor::from_matrix(merged.as_ref(), prev_left, k);
            self.center -= 1;
        }
    }

    /// The amplitude of the basis state `bits` (one per qubit).
    pub fn amplitude(&self, bits: &[bool]) -> C64 {
        assert_eq!(bits.len(), self.n_qubits(), "one bit per qubit");
        let mut row = vec![ONE];
        for (t, &b) in self.tensors.iter().zip(bits) {
            let p = b as usize;
            row = (0..t.right)
                .map(|r| {
                    (0..t.left)
                        .map(|l| row[l] * t.data[(2 * l + p) * t.right + r])
                        .sum()
                })
                .collect();
        }
        row[0]
    }

    /// The full `2ⁿ` statevector, with qubit 0 as the least significant bit.
    /// Only for small `n`.
    pub fn to_amplitudes(&self) -> Vec<C64> {
        assert!(self.n_qubits() <= 30, "too many qubits to expand");
        // rows[index of qubits so far][bond]
        let mut rows: Vec<Vec<C64>> = vec![vec![ONE]];
        for (q, t) in self.tensors.iter().enumerate() {
            let mut next = vec![Vec::new(); rows.len() * 2];
            for (idx, row) in rows.iter().enumerate() {
                for p in 0..2 {
                    next[idx | p << q] = (0..t.right)
                        .map(|r| {
                            (0..t.left)
                                .map(|l| row[l] * t.data[(2 * l + p) * t.right + r])
                                .sum()
                        })
                        .collect();
                }
            }
            rows = next;
        }
        rows.into_iter().map(|r| r[0]).collect()
    }

    /// `⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩`. Each term is contracted only over the sites
    /// between it and the orthogonality centre.
    pub fn expectation(&self, observable: &PauliSum) -> Result<f64, Error> {
        if observable.min_qubits() > self.n_qubits() {
            return Err(Error::QubitOutOfRange {
                qubit: observable.min_qubits() - 1,
                n_qubits: self.n_qubits(),
            });
        }
        let norm = self.norm_sqr();
        Ok(observable
            .terms()
            .iter()
            .map(|(c, s)| c * self.expectation_string(s).re)
            .sum::<f64>()
            / norm)
    }

    /// `⟨ψ|P|ψ⟩` (unnormalised), using the canonical form: sites outside
    /// the window from the string to the centre contract to the identity.
    fn expectation_string(&self, string: &PauliString) -> C64 {
        let ops = string.ops();
        let (first, last) = match (ops.first(), ops.last()) {
            (Some(&(f, _)), Some(&(l, _))) => (f, l),
            _ => return C64::new(self.norm_sqr(), 0.0),
        };
        let (start, end) = (first.min(self.center), last.max(self.center));
        let dim = self.tensors[start].left;
        let mut env = Mat::<C64>::identity(dim, dim);
        let mut next_op = ops.iter().peekable();
        for site in start..=end {
            let pauli = next_op
                .next_if(|&&(q, _)| q == site)
                .map_or(Pauli::I, |&(_, p)| p);
            env = transfer(&env, &self.tensors[site], pauli);
        }
        (0..env.nrows()).map(|i| env[(i, i)]).sum()
    }

    /// Draw `shots` basis states (qubit 0 as the least significant bit) by
    /// sequential conditional sampling, `O(n χ²)` per shot. Moves the
    /// centre, which doesn't change the state.
    pub fn sample(&mut self, shots: usize, rng: &mut impl Rng) -> Vec<usize> {
        assert!(
            self.n_qubits() <= usize::BITS as usize,
            "too many qubits to index"
        );
        self.sample_bits(shots, rng)
            .into_iter()
            .map(|bits| {
                bits.iter()
                    .enumerate()
                    .fold(0, |acc, (q, &b)| acc | (b as usize) << q)
            })
            .collect()
    }

    /// Draw `shots` samples of every qubit, as one bit per qubit: works for
    /// chains of any length.
    pub fn sample_bits(&mut self, shots: usize, rng: &mut impl Rng) -> Vec<Vec<bool>> {
        self.move_center(0);
        let norm = self.norm_sqr();
        (0..shots)
            .map(|_| {
                // With everything right of the centre right-orthonormal, the
                // marginal of each next qubit is the norm of the projected row.
                let mut row = vec![C64::new(1.0 / norm.sqrt(), 0.0)];
                let mut bits = Vec::with_capacity(self.n_qubits());
                for t in &self.tensors {
                    let project = |p: usize| -> Vec<C64> {
                        (0..t.right)
                            .map(|r| {
                                (0..t.left)
                                    .map(|l| row[l] * t.data[(2 * l + p) * t.right + r])
                                    .sum()
                            })
                            .collect()
                    };
                    let zero = project(0);
                    let p0: f64 = zero.iter().map(|a| a.norm_sqr()).sum();
                    let p1 = (1.0 - p0).max(0.0);
                    let one_out = rng.random::<f64>() * (p0 + p1) >= p0;
                    let (chosen, prob) = if one_out {
                        (project(1), p1)
                    } else {
                        (zero, p0)
                    };
                    bits.push(one_out);
                    let scale = 1.0 / prob.sqrt();
                    row = chosen.into_iter().map(|a| a * scale).collect();
                }
                bits
            })
            .collect()
    }

    /// Probability that measuring `qubit` gives 1.
    pub fn probability_one(&mut self, qubit: usize) -> Result<f64, Error> {
        let (one, total) = self.qubit_weights(qubit)?;
        Ok(one / total)
    }

    /// Measure `qubit`, collapsing the state. Returns the outcome.
    pub fn measure(&mut self, qubit: usize, rng: &mut impl Rng) -> Result<bool, Error> {
        let (one, total) = self.qubit_weights(qubit)?;
        let outcome = rng.random::<f64>() * total < one;
        // The centre is at `qubit`: project its physical index and
        // renormalise; the rest stays orthonormal.
        let weight = if outcome { one } else { total - one };
        let scale = (total / weight).sqrt();
        let t = &mut self.tensors[qubit];
        let r = t.right;
        for (i, a) in t.data.iter_mut().enumerate() {
            let bit = (i / r) % 2 == 1;
            *a = if bit == outcome { *a * scale } else { ZERO };
        }
        Ok(outcome)
    }

    /// `(weight with qubit = 1, total weight)`, moving the centre there.
    fn qubit_weights(&mut self, qubit: usize) -> Result<(f64, f64), Error> {
        if qubit >= self.n_qubits() {
            return Err(Error::QubitOutOfRange {
                qubit,
                n_qubits: self.n_qubits(),
            });
        }
        self.move_center(qubit);
        let t = &self.tensors[qubit];
        let r = t.right;
        let (mut one, mut total) = (0.0, 0.0);
        for (i, a) in t.data.iter().enumerate() {
            let w = a.norm_sqr();
            total += w;
            if (i / r) % 2 == 1 {
                one += w;
            }
        }
        Ok((one, total))
    }

    /// Run `circuit`, including measurements, resets and classical
    /// control, following one random history. Returns the classical bits.
    pub fn run_trajectory(
        &mut self,
        circuit: &Circuit,
        rng: &mut impl Rng,
    ) -> Result<Vec<bool>, Error> {
        trajectory::run_trajectory(self, circuit, rng)
    }

    /// Run `circuit` from this state `shots` times and count the final
    /// classical bits (bit 0 least significant); see
    /// [`DenseState::sample_circuit`](crate::DenseState::sample_circuit).
    pub fn sample_circuit(
        &self,
        circuit: &Circuit,
        shots: usize,
        rng: &mut impl Rng,
    ) -> Result<BTreeMap<u64, usize>, Error> {
        trajectory::sample_circuit(self, circuit, shots, rng)
    }

    /// As [`sample`](Self::sample), tallied by basis state.
    pub fn sample_counts(&mut self, shots: usize, rng: &mut impl Rng) -> BTreeMap<usize, usize> {
        let mut counts = BTreeMap::new();
        for j in self.sample(shots, rng) {
            *counts.entry(j).or_insert(0) += 1;
        }
        counts
    }

    /// TEBD: evolve by `steps` Trotter–Suzuki steps of the given order (see
    /// [`Circuit::pauli_evolution`]) for `exp(-iHt)`.
    pub fn tebd(
        &mut self,
        hamiltonian: &PauliSum,
        time: f64,
        steps: usize,
        order: usize,
    ) -> Result<(), Error> {
        let mut circuit = Circuit::new(self.n_qubits());
        circuit.pauli_evolution(hamiltonian, time, steps, order)?;
        self.run(&circuit)
    }
}

impl Trajectory for Mps {
    fn n_qubits(&self) -> usize {
        self.tensors.len()
    }

    fn run_gates(&mut self, gates: &Circuit) -> Result<(), Error> {
        self.run(gates)
    }

    fn measure_qubit<R: Rng>(&mut self, qubit: usize, rng: &mut R) -> Result<bool, Error> {
        self.measure(qubit, rng)
    }

    fn duplicate(&self) -> Result<Self, Error> {
        Ok(self.clone())
    }

    fn sample_qubits<R: Rng>(
        &mut self,
        shots: usize,
        qubits: &[usize],
        rng: &mut R,
    ) -> Result<Vec<u64>, Error> {
        Ok(self
            .sample_bits(shots, rng)
            .into_iter()
            .map(|bits| pack(qubits, |q| bits[q]))
            .collect())
    }
}

/// `env' = Σ A†[p] O[p][p'] env A[p']`, the transfer step of
/// `⟨ψ|O|ψ⟩` through one site.
fn transfer(env: &Mat<C64>, t: &Tensor, pauli: Pauli) -> Mat<C64> {
    // x[(l, p'), r'] = Σ_l' env[l, l'] A[l', p', r']
    let a = t.as_right_matrix();
    let x = matmul(env.as_ref(), a); // left × (2·right), columns (p', r')
    let r = t.right;
    let o = pauli_matrix(pauli);
    // y[(l, p), r'] = Σ_p' O[p][p'] x[l, (p', r')]
    let y = Mat::<C64>::from_fn(t.left * 2, r, |row, col| {
        let (l, p) = (row / 2, row % 2);
        o[p][0] * x[(l, col)] + o[p][1] * x[(l, r + col)]
    });
    matmul(t.as_left_matrix().adjoint(), y.as_ref())
}

fn pauli_matrix(p: Pauli) -> [[C64; 2]; 2] {
    match p {
        Pauli::I => Gate::Phase(0.0).matrix(),
        Pauli::X => Gate::X.matrix(),
        Pauli::Y => Gate::Y.matrix(),
        Pauli::Z => Gate::Z.matrix(),
    }
}

fn swap_matrix() -> [[C64; 4]; 4] {
    let mut m = [[ZERO; 4]; 4];
    for (r, c) in [(0, 0), (1, 2), (2, 1), (3, 3)] {
        m[r][c] = ONE;
    }
    m
}

/// `exp(-iθ/2 · P)` as basis changes, a CX parity ladder and one `Rz`:
/// `P = V Z…Z V†` with `V = H` for X and `V = S H` for Y.
fn decompose_pauli_rotation(pauli: &PauliString, theta: f64) -> Vec<Operation> {
    let ops = pauli.ops();
    let mut before = Vec::new();
    let mut after = Vec::new();
    for &(q, p) in ops {
        let gate = |gate| Operation::Gate { gate, target: q };
        match p {
            Pauli::X => {
                before.push(gate(Gate::H));
                after.push(gate(Gate::H));
            }
            Pauli::Y => {
                before.extend([gate(Gate::Sdg), gate(Gate::H)]);
                after.extend([gate(Gate::H), gate(Gate::S)]);
            }
            Pauli::Z | Pauli::I => {}
        }
    }
    let ladder: Vec<Operation> = ops
        .windows(2)
        .map(|w| Operation::Controlled {
            gate: Gate::X,
            control: w[0].0,
            target: w[1].0,
        })
        .collect();
    let last = ops.last().expect("weight above two").0;
    let mut out = before;
    out.extend(ladder.iter().cloned());
    out.push(Operation::Gate {
        gate: Gate::Rz(theta),
        target: last,
    });
    out.extend(ladder.into_iter().rev());
    out.extend(after);
    out
}
