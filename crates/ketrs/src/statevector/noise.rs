//! Noise on statevectors by quantum trajectories.
//!
//! Each shot follows one pure state: after every gate, each channel of the
//! [`NoiseModel`] applies one of its Kraus operators `Kᵢ`, chosen with
//! probability `‖Kᵢψ‖²`, and renormalises. Averaged over shots this is the
//! channel, so counts converge to the density matrix's at the memory cost
//! of a statevector (twice as many qubits as a density matrix allows).
//!
//! Mixtures of unitaries (Pauli channels such as depolarizing and bit or
//! phase flips) have probabilities independent of the state, so their
//! choices are drawn up front and run with the gates as one batch through
//! the tiled kernels. Other channels (amplitude and phase damping) read
//! their probabilities from the reduced density matrix of their qubits.

use super::{DenseState, Step};
use crate::circuit::Circuit;
use crate::density::{Channel, NoiseModel};
use crate::error::Error;
use crate::scalar::Scalar;
use crate::trajectory::{self, Trajectory};
use num_complex::Complex64 as C64;
use rand::Rng;
use std::collections::BTreeMap;

impl<T: Scalar> DenseState<T> {
    /// Run `circuit` (which may measure, reset and branch) following one
    /// random history with `noise` after every gate and on every
    /// measurement's readout. Returns the classical bits.
    ///
    /// Fails for channels with complex Kraus operators on a real state.
    pub fn run_noisy_trajectory(
        &mut self,
        circuit: &Circuit,
        noise: &NoiseModel,
        rng: &mut impl Rng,
    ) -> Result<Vec<bool>, Error> {
        noise.check()?;
        let mut noisy = Noisy {
            state: std::mem::replace(self, DenseState::zero(0)),
            noise: PreparedNoise::new(noise),
        };
        let result = trajectory::run_trajectory(&mut noisy, circuit, rng);
        *self = noisy.state;
        result
    }

    /// Counts of the classical bits over `shots` noisy trajectories from
    /// this state (see [`run_noisy_trajectory`](Self::run_noisy_trajectory)),
    /// with classical bit 0 least significant.
    pub fn sample_noisy(
        &self,
        circuit: &Circuit,
        noise: &NoiseModel,
        shots: usize,
        rng: &mut impl Rng,
    ) -> Result<BTreeMap<u64, usize>, Error> {
        noise.check()?;
        let noisy = Noisy {
            state: self.clone(),
            noise: PreparedNoise::new(noise),
        };
        trajectory::sample_circuit(&noisy, circuit, shots, rng)
    }
}

/// A channel ready for sampling.
#[derive(Clone)]
enum Prepared {
    /// Unitaries with fixed probabilities; `None` for the identity.
    Mixture(Vec<(f64, Option<Vec<C64>>)>),
    /// Kraus operators, chosen by the state.
    Kraus(Vec<Vec<C64>>),
}

impl Prepared {
    fn new(channel: &Channel) -> Prepared {
        match channel.as_mixture() {
            Some(terms) => Prepared::Mixture(
                terms
                    .into_iter()
                    .map(|(p, u)| {
                        let dim = 1 << channel.n_qubits();
                        let identity = (0..dim).all(|r| {
                            (0..dim).all(|c| {
                                let want = if r == c { u[0] } else { C64::new(0.0, 0.0) };
                                (u[r * dim + c] - want).norm() < 1e-14
                            })
                        });
                        // A phase times the identity changes nothing.
                        (p, (!identity).then_some(u))
                    })
                    .collect(),
            ),
            None => Prepared::Kraus(channel.kraus().to_vec()),
        }
    }
}

#[derive(Clone)]
struct PreparedNoise<'a> {
    model: &'a NoiseModel,
    one_qubit: Option<Prepared>,
    two_qubit: Option<Prepared>,
}

impl<'a> PreparedNoise<'a> {
    fn new(model: &'a NoiseModel) -> Self {
        PreparedNoise {
            model,
            one_qubit: model.one_qubit.as_ref().map(Prepared::new),
            two_qubit: model.two_qubit.as_ref().map(Prepared::new),
        }
    }

    fn prepared(&self, channel: &Channel) -> &Prepared {
        let one = self.model.one_qubit.as_ref();
        if one.is_some_and(|c| std::ptr::eq(c, channel)) {
            self.one_qubit.as_ref().expect("prepared with the model")
        } else {
            self.two_qubit.as_ref().expect("prepared with the model")
        }
    }
}

/// A statevector following noisy trajectories.
#[derive(Clone)]
struct Noisy<'a, T: Scalar> {
    state: DenseState<T>,
    noise: PreparedNoise<'a>,
}

impl<T: Scalar> Trajectory for Noisy<'_, T> {
    const DETERMINISTIC: bool = false;

    fn n_qubits(&self) -> usize {
        self.state.n_qubits
    }

    fn run_gates(&mut self, gates: &Circuit) -> Result<(), Error> {
        self.state.run(gates)
    }

    fn run_gates_in<R: Rng>(&mut self, gates: &Circuit, rng: &mut R) -> Result<(), Error> {
        let mut steps = Vec::new();
        for op in gates.ops() {
            steps.push(Step::Gate(Box::new(op.clone())));
            for (channel, qubits) in self.noise.model.channels_after(op) {
                match self.noise.prepared(channel) {
                    Prepared::Mixture(terms) => {
                        if let Some(u) = choose(terms, |(p, _)| *p, rng).1.clone() {
                            steps.push(Step::Matrix { qubits, matrix: u });
                        }
                    }
                    Prepared::Kraus(kraus) => {
                        self.state.apply_steps(&std::mem::take(&mut steps))?;
                        let rho = self.state.reduced_density_matrix(&qubits);
                        let probs: Vec<f64> = kraus.iter().map(|k| expectation(k, &rho)).collect();
                        let i = choose_index(&probs, rng);
                        let scale = 1.0 / probs[i].sqrt();
                        let matrix = kraus[i].iter().map(|&x| x * scale).collect();
                        steps.push(Step::Matrix { qubits, matrix });
                    }
                }
            }
        }
        self.state.apply_steps(&steps)?;
        if gates.global_phase() != 0.0 {
            let phase = T::from_complex(C64::cis(gates.global_phase()))
                .ok_or_else(|| Error::NotReal("global phase is not real".into()))?;
            self.state.amps.iter_mut().for_each(|a| *a *= phase);
        }
        Ok(())
    }

    fn record<R: Rng>(&self, outcome: bool, rng: &mut R) -> bool {
        match self.noise.model.readout {
            Some(p) if rng.random::<f64>() < p[outcome as usize] => !outcome,
            _ => outcome,
        }
    }

    fn measure_qubit<R: Rng>(&mut self, qubit: usize, rng: &mut R) -> Result<bool, Error> {
        self.state.measure(qubit, rng)
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
        Trajectory::sample_qubits(&mut self.state, shots, qubits, rng)
    }
}

impl<T: Scalar> DenseState<T> {
    /// `ρ[a][b] = Σ ψ(a, rest) ψ*(b, rest)` over the other qubits, with
    /// basis bit `k` of `a` and `b` for `qubits[k]`; row-major.
    fn reduced_density_matrix(&self, qubits: &[usize]) -> Vec<C64> {
        let dim = 1 << qubits.len();
        let mask = qubits.iter().fold(0usize, |m, &q| m | 1 << q);
        let spread = |a: usize| {
            qubits
                .iter()
                .enumerate()
                .fold(0, |acc, (k, &q)| acc | (a >> k & 1) << q)
        };
        let offsets: Vec<usize> = (0..dim).map(spread).collect();
        let mut rho = vec![C64::new(0.0, 0.0); dim * dim];
        let mut column = vec![C64::new(0.0, 0.0); dim];
        for base in (0..self.amps.len()).filter(|i| i & mask == 0) {
            for (c, &o) in column.iter_mut().zip(&offsets) {
                *c = self.amps[base | o].to_complex();
            }
            for a in 0..dim {
                for b in 0..dim {
                    rho[a * dim + b] += column[a] * column[b].conj();
                }
            }
        }
        rho
    }
}

/// `Tr(K ρ K†)`, the probability of Kraus operator `K`.
fn expectation(k: &[C64], rho: &[C64]) -> f64 {
    let dim = (rho.len() as f64).sqrt() as usize;
    let mut total = 0.0;
    for r in 0..dim {
        for a in 0..dim {
            for b in 0..dim {
                total += (k[r * dim + a] * rho[a * dim + b] * k[r * dim + b].conj()).re;
            }
        }
    }
    total.max(0.0)
}

fn choose_index(weights: &[f64], rng: &mut impl Rng) -> usize {
    let total: f64 = weights.iter().sum();
    let mut x = rng.random::<f64>() * total;
    for (i, &w) in weights.iter().enumerate() {
        if x < w {
            return i;
        }
        x -= w;
    }
    // Rounding: the last term with any weight.
    weights.iter().rposition(|&w| w > 0.0).unwrap_or(0)
}

fn choose<'t, I>(items: &'t [I], weight: impl Fn(&I) -> f64, rng: &mut impl Rng) -> &'t I {
    let weights: Vec<f64> = items.iter().map(weight).collect();
    &items[choose_index(&weights, rng)]
}
