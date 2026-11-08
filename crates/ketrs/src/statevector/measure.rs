//! Expectation values, sampling and measurement.

use super::DenseState;
use super::hamiltonian::chunked_sum;
use super::kernels::{BLOCK, PARALLEL_THRESHOLD, Prepared};
use crate::circuit::Operation;
use crate::error::Error;
use crate::gate::Gate;
use crate::observable::PauliString;
use crate::scalar::Scalar;
use rand::Rng;
use rayon::prelude::*;
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
        let (one, total) = self.qubit_weights(qubit)?;
        Ok(one / total)
    }

    /// `(weight with qubit = 1, total weight)`, in one parallel pass.
    pub(crate) fn qubit_weights(&self, qubit: usize) -> Result<(f64, f64), Error> {
        self.check_qubit(qubit)?;
        let mask = 1 << qubit;
        let Weights(one, total) = chunked_sum(self.amps.len(), |range| {
            let mut w = Weights(0.0, 0.0);
            for (i, a) in range.clone().zip(&self.amps[range]) {
                let p = a.norm_sqr();
                w.0 += if i & mask != 0 { p } else { 0.0 };
                w.1 += p;
            }
            w
        });
        Ok((one, total))
    }

    /// Draw `shots` basis-state indices from the Born distribution, without
    /// changing the state.
    ///
    /// Each shot's uniform draw picks the first index whose cumulative
    /// weight exceeds it. Rather than form the cumulative weights, the
    /// draws are sorted and each block of amplitudes is swept once, in
    /// parallel, for the draws that land in it.
    pub fn sample(&self, shots: usize, rng: &mut impl Rng) -> Vec<usize> {
        let len = self.amps.len();
        let block = BLOCK.min(len);
        let block_weight = |b: usize| -> f64 {
            self.amps[b * block..((b + 1) * block).min(len)]
                .iter()
                .map(|a| a.norm_sqr())
                .sum()
        };
        let n_blocks = len.div_ceil(block);
        let weights: Vec<f64> = if len >= PARALLEL_THRESHOLD {
            (0..n_blocks).into_par_iter().map(block_weight).collect()
        } else {
            (0..n_blocks).map(block_weight).collect()
        };
        // ends[b]: the cumulative weight through block b.
        let ends: Vec<f64> = weights
            .iter()
            .scan(0.0, |acc, w| {
                *acc += w;
                Some(*acc)
            })
            .collect();
        let total = ends[n_blocks - 1];
        let mut draws: Vec<(f64, usize)> = (0..shots)
            .map(|k| (rng.random::<f64>() * total, k))
            .collect();
        draws.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        // The draws landing in each block, in order.
        let mut groups = Vec::new();
        let mut rest = &draws[..];
        while let Some(&(r, _)) = rest.first() {
            let b = ends.partition_point(|&e| e <= r).min(n_blocks - 1);
            let n = if b == n_blocks - 1 {
                rest.len()
            } else {
                rest.partition_point(|&(r, _)| r < ends[b])
            };
            let (here, after) = rest.split_at(n);
            groups.push((b, here));
            rest = after;
        }
        let sweep = |&(b, here): &(usize, &[(f64, usize)])| -> Vec<(usize, usize)> {
            let (start, end) = (b * block, ((b + 1) * block).min(len));
            let mut acc = if b == 0 { 0.0 } else { ends[b - 1] };
            let mut i = start;
            // Where rounding leaves a draw past the block's last amplitude,
            // the last one with any weight.
            let mut last = start;
            let mut found = Vec::with_capacity(here.len());
            for &(r, k) in here {
                while i < end {
                    let w = self.amps[i].norm_sqr();
                    if w > 0.0 {
                        last = i;
                    }
                    if acc + w > r {
                        break;
                    }
                    acc += w;
                    i += 1;
                }
                found.push((k, if i < end { i } else { last }));
            }
            found
        };
        let found: Vec<Vec<(usize, usize)>> = if len >= PARALLEL_THRESHOLD {
            groups.par_iter().map(sweep).collect()
        } else {
            groups.iter().map(sweep).collect()
        };
        let mut samples = vec![0; shots];
        for (k, index) in found.into_iter().flatten() {
            samples[k] = index;
        }
        samples
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
        let (one, total) = self.qubit_weights(qubit)?;
        let outcome = rng.random::<f64>() < one / total;
        self.project(qubit, outcome, if outcome { one } else { total - one });
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

    /// Project `qubit` onto `outcome`, whose weight is `weight`, and
    /// renormalise, in one parallel pass.
    pub(crate) fn project(&mut self, qubit: usize, outcome: bool, weight: f64) {
        let mask = 1 << qubit;
        let keep = if outcome { mask } else { 0 };
        let scale = T::from_real(1.0 / weight.sqrt());
        let run = |offset: usize, chunk: &mut [T]| {
            for (i, a) in (offset..).zip(chunk) {
                *a = if i & mask == keep {
                    *a * scale
                } else {
                    T::ZERO
                };
            }
        };
        if self.amps.len() >= PARALLEL_THRESHOLD {
            self.amps
                .par_chunks_mut(BLOCK)
                .enumerate()
                .for_each(|(c, chunk)| run(c * BLOCK, chunk));
        } else {
            run(0, &mut self.amps);
        }
    }
}

/// Weights with a qubit 1 and in total, summed over chunks.
struct Weights(f64, f64);

impl std::iter::Sum for Weights {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Weights(0.0, 0.0), |a, b| Weights(a.0 + b.0, a.1 + b.1))
    }
}
