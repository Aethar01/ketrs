//! Expectation values, sampling and measurement.

use super::DenseState;
use super::kernels::Prepared;
use crate::circuit::Operation;
use crate::error::Error;
use crate::gate::Gate;
use crate::observable::PauliString;
use crate::scalar::Scalar;
use rand::Rng;
use std::collections::BTreeMap;

impl<T: Scalar> DenseState<T> {
    /// `⟨ψ|P|ψ⟩` for a single Pauli string, assuming `ψ` is normalised.
    pub fn expectation_string(&self, string: &PauliString) -> Result<f64, Error> {
        for q in string.qubits() {
            self.check_qubit(q)?;
        }
        let masks = string.masks();
        // ⟨ψ|P|ψ⟩ = iⁿʸ Σᵢ (-1)^|i ∧ z| ψ*_{i⊕x} ψᵢ.
        let sum: T = self
            .amps
            .iter()
            .enumerate()
            .map(|(i, &a)| {
                let term = self.amps[i ^ masks.x].conj() * a;
                if masks.z_sign(i) { -term } else { term }
            })
            .sum();
        Ok((masks.y_phase() * sum.to_complex()).re)
    }

    /// Probability that measuring `qubit` gives 1.
    pub fn probability_one(&self, qubit: usize) -> Result<f64, Error> {
        self.check_qubit(qubit)?;
        Ok(self
            .amps
            .iter()
            .enumerate()
            .filter(|(i, _)| i & (1 << qubit) != 0)
            .map(|(_, a)| a.norm_sqr())
            .sum::<f64>()
            / self.norm_sqr())
    }

    /// Draw `shots` basis-state indices from the Born distribution, without
    /// changing the state.
    pub fn sample(&self, shots: usize, rng: &mut impl Rng) -> Vec<usize> {
        let cumulative: Vec<f64> = self
            .amps
            .iter()
            .scan(0.0, |acc, a| {
                *acc += a.norm_sqr();
                Some(*acc)
            })
            .collect();
        let total = cumulative[cumulative.len() - 1];
        let last = cumulative.len() - 1;
        (0..shots)
            .map(|_| {
                let r = rng.random::<f64>() * total;
                cumulative.partition_point(|&c| c <= r).min(last)
            })
            .collect()
    }

    /// As [`sample`](Self::sample), tallied into counts per basis state.
    pub fn sample_counts(&self, shots: usize, rng: &mut impl Rng) -> BTreeMap<usize, usize> {
        let mut counts = BTreeMap::new();
        for index in self.sample(shots, rng) {
            *counts.entry(index).or_insert(0) += 1;
        }
        counts
    }

    /// Measure `qubit` in the computational basis, collapsing the state.
    /// Returns `true` for outcome 1.
    pub fn measure(&mut self, qubit: usize, rng: &mut impl Rng) -> Result<bool, Error> {
        let p1 = self.probability_one(qubit)?;
        let outcome = rng.random::<f64>() < p1;
        self.collapse(qubit, outcome);
        Ok(outcome)
    }

    /// Measure every qubit, collapsing to a basis state. Returns its index.
    pub fn measure_all(&mut self, rng: &mut impl Rng) -> usize {
        let index = self.sample(1, rng)[0];
        *self = Self::basis(self.n_qubits, index);
        index
    }

    /// Measure `qubit` and flip it to `|0⟩` if the outcome was 1.
    pub fn reset(&mut self, qubit: usize, rng: &mut impl Rng) -> Result<(), Error> {
        if self.measure(qubit, rng)? {
            let flip = Operation::Gate {
                gate: Gate::X,
                target: qubit,
            };
            Prepared::new(&flip)
                .expect("X is real")
                .apply(&mut self.amps);
        }
        Ok(())
    }

    /// Project `qubit` onto `outcome` and renormalise.
    pub(crate) fn collapse(&mut self, qubit: usize, outcome: bool) {
        let mask = 1 << qubit;
        for (i, a) in self.amps.iter_mut().enumerate() {
            if (i & mask != 0) != outcome {
                *a = T::ZERO;
            }
        }
        let scale = T::from_real(1.0 / self.norm_sqr().sqrt());
        for a in &mut self.amps {
            *a *= scale;
        }
    }
}
