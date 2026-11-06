//! Mixed states as density matrices, and noise as quantum channels.
//!
//! An `n`-qubit density matrix ρ is stored as a `2n`-qubit statevector
//! `vec(ρ)` with `vec(ρ)[i + 2ⁿ j] = ρᵢⱼ`: row index on qubits `0..n`,
//! column index on `n..2n`. A gate `U` acts as `ρ → UρU†`, which on
//! `vec(ρ)` is `U` on the low qubits and `U*` on the high ones, so circuits
//! run on the statevector engine with all its fusion, tiling and
//! parallelism. A channel with Kraus operators `Kₖ`, `ρ → Σ Kₖ ρ Kₖ†`, is
//! the superoperator `Σ Kₖ ⊗ Kₖ*` on the row and column copies of its
//! qubits. Memory is that of a statevector on twice as many qubits.

use crate::circuit::{Circuit, Operation};
use crate::error::Error;
use crate::fusion::local_matrix;
use crate::gate::Gate;
use crate::observable::{Pauli, PauliString, PauliSum};
use crate::statevector::hamiltonian::{GroupedPauliSum, Imaginary};
use crate::statevector::{Statevector, Step};
use num_complex::Complex64 as C64;
use rand::Rng;

const ZERO: C64 = C64::new(0.0, 0.0);
const ONE: C64 = C64::new(1.0, 0.0);

/// A quantum channel on `k` qubits, given by Kraus operators: `ρ → Σ Kᵢ ρ
/// Kᵢ†`, with `Σ Kᵢ† Kᵢ = 1`. Matrices are `2^k × 2^k`, row-major, with
/// basis index bit `b` for the channel's `b`-th qubit (Qiskit's order).
#[derive(Clone, Debug, PartialEq)]
pub struct Channel {
    n_qubits: usize,
    kraus: Vec<Vec<C64>>,
}

impl Channel {
    /// A channel from its Kraus operators, each a list of rows. Fails
    /// unless they are square, of one power-of-two size, and trace
    /// preserving to `1e-10`.
    pub fn from_kraus(kraus: Vec<Vec<Vec<C64>>>) -> Result<Self, Error> {
        let dim = kraus.first().map_or(0, Vec::len);
        if dim == 0 || !dim.is_power_of_two() {
            return Err(Error::InvalidArgument(
                "Kraus operators must be square with a power-of-two size".into(),
            ));
        }
        let mut flat = Vec::with_capacity(kraus.len());
        for k in &kraus {
            if k.len() != dim || k.iter().any(|row| row.len() != dim) {
                return Err(Error::InvalidArgument(
                    "Kraus operators must all have the same square shape".into(),
                ));
            }
            flat.push(k.iter().flatten().copied().collect::<Vec<_>>());
        }
        // Σ K†K = 1.
        for r in 0..dim {
            for c in 0..dim {
                let sum: C64 = flat
                    .iter()
                    .map(|k| {
                        (0..dim)
                            .map(|m| k[m * dim + r].conj() * k[m * dim + c])
                            .sum::<C64>()
                    })
                    .sum();
                let want = if r == c { ONE } else { ZERO };
                if (sum - want).norm() > 1e-10 {
                    return Err(Error::InvalidArgument(
                        "Kraus operators are not trace preserving (Σ K†K ≠ 1)".into(),
                    ));
                }
            }
        }
        Ok(Channel {
            n_qubits: dim.trailing_zeros() as usize,
            kraus: flat,
        })
    }

    /// A mixture of Pauli strings on `n_qubits` qubits: `ρ → Σ pₚ P ρ P`,
    /// with the identity taking the remaining probability.
    pub fn pauli(n_qubits: usize, terms: &[(PauliString, f64)]) -> Result<Self, Error> {
        let total: f64 = terms.iter().map(|(_, p)| p).sum();
        if terms.iter().any(|&(_, p)| p < 0.0) || total > 1.0 + 1e-12 {
            return Err(Error::InvalidArgument(
                "Pauli probabilities must be non-negative and sum to at most 1".into(),
            ));
        }
        let mut kraus = vec![scaled(
            &pauli_matrix(n_qubits, &PauliString::identity()),
            (1.0 - total).max(0.0),
        )];
        for (string, p) in terms {
            if string.min_qubits() > n_qubits {
                return Err(Error::InvalidArgument(format!(
                    "{string:?} is wider than {n_qubits} qubits"
                )));
            }
            kraus.push(scaled(&pauli_matrix(n_qubits, string), *p));
        }
        Ok(Channel { n_qubits, kraus })
    }

    /// Depolarizing noise on `n_qubits` qubits, as in Qiskit Aer's
    /// `depolarizing_error`: `ρ → (1 − p) ρ + p · 1/2ⁿ`, for `p` in `[0, 4ⁿ/(4ⁿ − 1)]`.
    pub fn depolarizing(p: f64, n_qubits: usize) -> Result<Self, Error> {
        let d2 = 1usize << (2 * n_qubits);
        // Every non-identity Pauli with probability p/4ⁿ.
        let each = p / d2 as f64;
        let terms: Vec<(PauliString, f64)> = (1..d2)
            .map(|k| {
                let ops = (0..n_qubits).map(|q| {
                    (
                        q,
                        [Pauli::I, Pauli::X, Pauli::Y, Pauli::Z][k >> (2 * q) & 3],
                    )
                });
                (PauliString::new(ops).expect("distinct qubits"), each)
            })
            .collect();
        Self::pauli(n_qubits, &terms)
    }

    /// Bit flip: X with probability `p`.
    pub fn bit_flip(p: f64) -> Result<Self, Error> {
        Self::pauli(1, &[(PauliString::from_label("X")?, p)])
    }

    /// Phase flip: Z with probability `p`.
    pub fn phase_flip(p: f64) -> Result<Self, Error> {
        Self::pauli(1, &[(PauliString::from_label("Z")?, p)])
    }

    /// Amplitude damping (energy relaxation): `|1⟩` decays to `|0⟩` with
    /// probability `γ`.
    pub fn amplitude_damping(gamma: f64) -> Result<Self, Error> {
        let c = |x: f64| C64::new(x, 0.0);
        Self::from_kraus(vec![
            vec![vec![ONE, ZERO], vec![ZERO, c((1.0 - gamma).sqrt())]],
            vec![vec![ZERO, c(gamma.sqrt())], vec![ZERO, ZERO]],
        ])
    }

    /// Phase damping (pure dephasing): off-diagonal elements shrink by
    /// `√(1 − λ)`.
    pub fn phase_damping(lambda: f64) -> Result<Self, Error> {
        let c = |x: f64| C64::new(x, 0.0);
        Self::from_kraus(vec![
            vec![vec![ONE, ZERO], vec![ZERO, c((1.0 - lambda).sqrt())]],
            vec![vec![ZERO, ZERO], vec![ZERO, c(lambda.sqrt())]],
        ])
    }

    /// Number of qubits the channel acts on.
    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    /// The Kraus operators, each `2^k × 2^k` row-major.
    pub fn kraus(&self) -> &[Vec<C64>] {
        &self.kraus
    }

    /// `Σ K ⊗ K*` as a `4^k × 4^k` matrix on (row qubits, column qubits),
    /// basis index `r + 2^k c`.
    fn superoperator(&self) -> Vec<C64> {
        let dim = 1 << self.n_qubits;
        let big = dim * dim;
        let mut s = vec![ZERO; big * big];
        for k in &self.kraus {
            for (r_out, c_out, r_in, c_in) in (0..dim).flat_map(|a| {
                (0..dim).flat_map(move |b| {
                    (0..dim).flat_map(move |c| (0..dim).map(move |d| (a, b, c, d)))
                })
            }) {
                s[(r_out + dim * c_out) * big + r_in + dim * c_in] +=
                    k[r_out * dim + r_in] * k[c_out * dim + c_in].conj();
            }
        }
        s
    }
}

fn scaled(m: &[C64], p: f64) -> Vec<C64> {
    m.iter().map(|&x| x * p.sqrt()).collect()
}

/// The matrix of `string` on `n_qubits` qubits.
fn pauli_matrix(n_qubits: usize, string: &PauliString) -> Vec<C64> {
    let dim = 1 << n_qubits;
    let m = string.masks();
    let mut out = vec![ZERO; dim * dim];
    for col in 0..dim {
        // P|col⟩ = iⁿʸ (−1)^|col ∧ z| |col ⊕ x⟩.
        let sign = if m.z_sign(col) { -1.0 } else { 1.0 };
        out[(col ^ m.x) * dim + col] = m.y_phase() * sign;
    }
    out
}

/// Noise applied after every gate: `one_qubit` on each qubit of a gate
/// that isn't a two-qubit gate with `two_qubit` set, and `two_qubit` on
/// the pair of a two-qubit gate.
#[derive(Clone, Debug, Default)]
pub struct NoiseModel {
    /// A one-qubit channel after every gate, on each of its qubits.
    pub one_qubit: Option<Channel>,
    /// A two-qubit channel after every two-qubit gate.
    pub two_qubit: Option<Channel>,
}

/// A mixed state of `n` qubits as a density matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct DensityMatrix {
    n_qubits: usize,
    /// `vec(ρ)` on `2n` qubits.
    vec: Statevector,
}

impl DensityMatrix {
    /// `|0…0⟩⟨0…0|`.
    pub fn zero(n_qubits: usize) -> Self {
        DensityMatrix {
            n_qubits,
            vec: Statevector::zero(2 * n_qubits),
        }
    }

    /// The pure state `|ψ⟩⟨ψ|`.
    pub fn from_statevector(psi: &Statevector) -> Self {
        let a = psi.amplitudes();
        let dim = a.len();
        let amps = (0..dim * dim)
            .map(|k| a[k % dim] * a[k / dim].conj())
            .collect();
        DensityMatrix {
            n_qubits: psi.n_qubits(),
            vec: Statevector::from_amplitudes(amps).expect("power of two"),
        }
    }

    /// A density matrix from its elements, row-major (`rho[i·2ⁿ + j] =
    /// ρᵢⱼ`). Not checked for being Hermitian, positive or of unit trace.
    pub fn from_matrix(n_qubits: usize, rho: &[C64]) -> Result<Self, Error> {
        let dim = 1usize << n_qubits;
        if rho.len() != dim * dim {
            return Err(Error::InvalidLength(rho.len()));
        }
        let amps = (0..dim * dim)
            .map(|k| rho[(k % dim) * dim + k / dim])
            .collect();
        Ok(DensityMatrix {
            n_qubits,
            vec: Statevector::from_amplitudes(amps)?,
        })
    }

    /// Number of qubits.
    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    fn dim(&self) -> usize {
        1 << self.n_qubits
    }

    /// `ρᵢⱼ`.
    pub fn element(&self, i: usize, j: usize) -> C64 {
        self.vec.amplitudes()[i + self.dim() * j]
    }

    /// The matrix, row-major: `out[i·2ⁿ + j] = ρᵢⱼ`.
    pub fn to_matrix(&self) -> Vec<C64> {
        let dim = self.dim();
        (0..dim * dim)
            .map(|k| self.element(k / dim, k % dim))
            .collect()
    }

    /// `Tr ρ`.
    pub fn trace(&self) -> f64 {
        (0..self.dim()).map(|i| self.element(i, i).re).sum()
    }

    /// `Tr ρ²`: 1 for pure states, `1/2ⁿ` for the maximally mixed one.
    pub fn purity(&self) -> f64 {
        self.vec.norm_sqr()
    }

    /// Measurement probability of each basis state: the diagonal.
    pub fn probabilities(&self) -> Vec<f64> {
        (0..self.dim()).map(|i| self.element(i, i).re).collect()
    }

    /// `⟨ψ|ρ|ψ⟩`, the fidelity with a pure state.
    pub fn fidelity(&self, psi: &Statevector) -> Result<f64, Error> {
        if psi.n_qubits() != self.n_qubits {
            return Err(Error::QubitCountMismatch {
                expected: self.n_qubits,
                found: psi.n_qubits(),
            });
        }
        let a = psi.amplitudes();
        let dim = self.dim();
        let mut sum = ZERO;
        for i in 0..dim {
            for j in 0..dim {
                sum += a[i].conj() * self.element(i, j) * a[j];
            }
        }
        Ok(sum.re)
    }

    /// `Tr(ρH)` for a Hermitian `observable`.
    pub fn expectation(&self, observable: &PauliSum) -> Result<f64, Error> {
        let grouped = GroupedPauliSum::<C64>::new(observable, self.n_qubits, Imaginary::Reject)?;
        let amps = self.vec.amplitudes();
        let dim = self.dim();
        // Tr(ρP) = Σₖ c(k) ρ_{k, k⊕x}, with P|k⟩ = c(k)|k ⊕ x⟩.
        let mut sum = ZERO;
        for k in 0..dim {
            let d: f64 = grouped
                .diagonal
                .iter()
                .map(|&(z, c)| if (k & z).count_ones() % 2 == 1 { -c } else { c })
                .sum();
            sum += amps[k + dim * k] * d;
            for g in &grouped.groups {
                sum += g.coefficient(k) * amps[k + dim * (k ^ g.x)];
            }
        }
        Ok(sum.re)
    }

    /// Draw `shots` basis-state indices from the diagonal.
    pub fn sample(&self, shots: usize, rng: &mut impl Rng) -> Vec<usize> {
        let cumulative: Vec<f64> = self
            .probabilities()
            .iter()
            .scan(0.0, |acc, p| {
                *acc += p.max(0.0);
                Some(*acc)
            })
            .collect();
        let total = cumulative[cumulative.len() - 1];
        (0..shots)
            .map(|_| {
                let r = rng.random::<f64>() * total;
                cumulative
                    .partition_point(|&c| c <= r)
                    .min(cumulative.len() - 1)
            })
            .collect()
    }

    /// Apply every gate of `circuit` (unitary, bound, of the same width):
    /// `ρ → UρU†`.
    pub fn run(&mut self, circuit: &Circuit) -> Result<(), Error> {
        self.check(circuit)?;
        let mut doubled = Circuit::new(2 * self.n_qubits);
        for op in circuit.ops() {
            doubled.push(op.clone())?;
            doubled.push(conjugate(op, self.n_qubits))?;
        }
        self.vec.run(&doubled)
    }

    /// Run `circuit` with `noise` after every gate. A gate on one or two
    /// qubits and its noise are applied together, as one superoperator.
    pub fn run_noisy(&mut self, circuit: &Circuit, noise: &NoiseModel) -> Result<(), Error> {
        self.check(circuit)?;
        let n = self.n_qubits;
        let mut steps = Vec::new();
        for op in circuit.ops() {
            let qubits = op.qubits();
            let channels: Vec<(&Channel, Vec<usize>)> = match (&noise.two_qubit, qubits.len()) {
                (Some(channel), 2) => vec![(channel, qubits.clone())],
                _ => match &noise.one_qubit {
                    Some(channel) => qubits.iter().map(|&q| (channel, vec![q])).collect(),
                    None => Vec::new(),
                },
            };
            match local_matrix(op) {
                Some((local, u)) if !channels.is_empty() => {
                    steps.push(superoperator_step(
                        &noisy_gate(&u, &local, &channels),
                        &local,
                        n,
                    ));
                }
                _ => {
                    steps.push(Step::Gate(Box::new(op.clone())));
                    steps.push(Step::Gate(Box::new(conjugate(op, n))));
                    for (channel, on) in channels {
                        steps.push(superoperator_step(channel, &on, n));
                    }
                }
            }
        }
        self.vec.apply_steps(&steps)
    }

    /// Apply `channel` to `qubits` (as many as it acts on, distinct).
    pub fn apply_channel(&mut self, channel: &Channel, qubits: &[usize]) -> Result<(), Error> {
        if qubits.len() != channel.n_qubits {
            return Err(Error::InvalidArgument(format!(
                "a {}-qubit channel applied to {} qubits",
                channel.n_qubits,
                qubits.len()
            )));
        }
        for (k, &q) in qubits.iter().enumerate() {
            if q >= self.n_qubits {
                return Err(Error::QubitOutOfRange {
                    qubit: q,
                    n_qubits: self.n_qubits,
                });
            }
            if qubits[..k].contains(&q) {
                return Err(Error::DuplicateQubit(q));
            }
        }
        let step = superoperator_step(channel, qubits, self.n_qubits);
        self.vec.apply_steps(&[step])
    }

    fn check(&self, circuit: &Circuit) -> Result<(), Error> {
        circuit.check_unitary()?;
        if circuit.n_qubits() != self.n_qubits {
            return Err(Error::QubitCountMismatch {
                expected: self.n_qubits,
                found: circuit.n_qubits(),
            });
        }
        if circuit.n_parameters() > 0 {
            return Err(Error::InvalidArgument(format!(
                "circuit has {} unbound parameters; call bind first",
                circuit.n_parameters()
            )));
        }
        Ok(())
    }
}

/// `channel` on `qubits` of an `n`-qubit density matrix: its superoperator
/// on the row copies of the qubits, then the column copies.
fn superoperator_step(channel: &Channel, qubits: &[usize], n: usize) -> Step {
    Step::Matrix {
        qubits: qubits
            .iter()
            .copied()
            .chain(qubits.iter().map(|q| q + n))
            .collect(),
        matrix: channel.superoperator(),
    }
}

/// The gate `u` on `qubits` followed by `channels` (each on some of those
/// qubits), as one channel on `qubits`: Kraus operators `Kₙ ⋯ K₁ u` for
/// every choice of one Kraus operator per channel.
fn noisy_gate(u: &[Vec<C64>], qubits: &[usize], channels: &[(&Channel, Vec<usize>)]) -> Channel {
    let dim = 1 << qubits.len();
    let mut kraus: Vec<Vec<C64>> = vec![u.iter().flatten().copied().collect()];
    for (channel, on) in channels {
        // Positions of the channel's qubits among the gate's.
        let bits: Vec<usize> = on
            .iter()
            .map(|q| {
                qubits
                    .iter()
                    .position(|g| g == q)
                    .expect("noise on the gate's qubits")
            })
            .collect();
        let local = |i: usize| {
            bits.iter()
                .enumerate()
                .fold(0, |acc, (b, &p)| acc | (i >> p & 1) << b)
        };
        let rest = |i: usize| bits.iter().fold(i, |acc, &p| acc & !(1 << p));
        let mut next = Vec::with_capacity(kraus.len() * channel.kraus.len());
        for k in channel.kraus() {
            let small = 1 << on.len();
            for before in &kraus {
                // (K on `on`, identity elsewhere) · before.
                let mut product = vec![ZERO; dim * dim];
                for r in 0..dim {
                    for m in 0..dim {
                        if rest(r) != rest(m) {
                            continue;
                        }
                        let entry = k[local(r) * small + local(m)];
                        if entry == ZERO {
                            continue;
                        }
                        for c in 0..dim {
                            product[r * dim + c] += entry * before[m * dim + c];
                        }
                    }
                }
                next.push(product);
            }
        }
        kraus = next;
    }
    Channel {
        n_qubits: qubits.len(),
        kraus,
    }
}

/// `op*` on the qubits `shift` higher: `U*` for the column side of `UρU†`.
fn conjugate(op: &Operation, shift: usize) -> Operation {
    let conj2 = |m: [[C64; 2]; 2]| m.map(|row| row.map(|x| x.conj()));
    match op {
        Operation::Gate { gate, target } => Operation::Gate {
            gate: Gate::Unitary(conj2(gate.matrix())),
            target: target + shift,
        },
        Operation::Controlled {
            gate,
            control,
            target,
        } => Operation::Controlled {
            gate: Gate::Unitary(conj2(gate.matrix())),
            control: control + shift,
            target: target + shift,
        },
        Operation::Swap(a, b) => Operation::Swap(a + shift, b + shift),
        Operation::MultiControlled {
            gate,
            controls,
            target,
        } => Operation::MultiControlled {
            gate: Gate::Unitary(conj2(gate.matrix())),
            controls: controls.iter().map(|c| c + shift).collect(),
            target: target + shift,
        },
        Operation::ControlledSwap { controls, a, b } => Operation::ControlledSwap {
            controls: controls.iter().map(|c| c + shift).collect(),
            a: a + shift,
            b: b + shift,
        },
        Operation::Unitary2 { .. } => {
            let (qubits, m) = local_matrix(op).expect("two-qubit operation");
            Operation::Unitary2 {
                matrix: std::array::from_fn(|r| std::array::from_fn(|c| m[r][c].conj())),
                qubits: [qubits[0] + shift, qubits[1] + shift],
            }
        }
        Operation::PauliRotation { pauli, theta } => {
            // (e^{-iθP/2})* = e^{iθP*/2}, and P* = (−1)^{#Y} P.
            let n_y = pauli.ops().iter().filter(|&&(_, p)| p == Pauli::Y).count();
            let shifted = PauliString::new(pauli.ops().iter().map(|&(q, p)| (q + shift, p)))
                .expect("distinct qubits");
            Operation::PauliRotation {
                pauli: shifted,
                theta: if n_y % 2 == 1 { *theta } else { -*theta },
            }
        }
    }
}
