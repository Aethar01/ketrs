//! Circuits with measurement, reset and classical control on statevectors
//! (see [`crate::trajectory`]).

use super::DenseState;
use crate::circuit::Circuit;
use crate::error::Error;
use crate::scalar::Scalar;
use crate::trajectory::{self, Trajectory};
use rand::Rng;
use std::collections::BTreeMap;

impl<T: Scalar> DenseState<T> {
    /// Run `circuit`, including measurements, resets and classical
    /// control, following one random history. Returns the classical bits
    /// (all starting at 0) as the circuit left them.
    pub fn run_trajectory(
        &mut self,
        circuit: &Circuit,
        rng: &mut impl Rng,
    ) -> Result<Vec<bool>, Error> {
        trajectory::run_trajectory(self, circuit, rng)
    }

    /// Run `circuit` from this state `shots` times, returning how often each
    /// final value of the classical bits occurred, as an integer with
    /// classical bit 0 least significant.
    ///
    /// When every measurement comes at the end, the state is simulated once
    /// and sampled. Otherwise the gates before the first measurement, reset
    /// or classical control are simulated once, and each shot continues
    /// from there.
    pub fn sample_circuit(
        &self,
        circuit: &Circuit,
        shots: usize,
        rng: &mut impl Rng,
    ) -> Result<BTreeMap<u64, usize>, Error> {
        trajectory::sample_circuit(self, circuit, shots, rng)
    }
}

impl<T: Scalar> Trajectory for DenseState<T> {
    fn n_qubits(&self) -> usize {
        self.n_qubits
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
            .sample(shots, rng)
            .into_iter()
            .map(|index| pack(qubits, |q| index >> q & 1 == 1))
            .collect())
    }
}

/// The values `bit(q)` of `qubits`, with `qubits[k]` at bit `k`.
pub(crate) fn pack(qubits: &[usize], bit: impl Fn(usize) -> bool) -> u64 {
    qubits
        .iter()
        .enumerate()
        .fold(0, |acc, (k, &q)| acc | (bit(q) as u64) << k)
}
