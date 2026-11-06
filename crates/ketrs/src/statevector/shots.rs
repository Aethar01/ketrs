//! Circuits with measurement, reset and classical control.

use super::DenseState;
use crate::circuit::Circuit;
use crate::classical::{Instruction, Piece};
use crate::error::Error;
use crate::scalar::Scalar;
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
        self.check_runnable(circuit)?;
        let mut clbits = vec![false; circuit.n_clbits()];
        self.follow(circuit, &mut clbits, rng)?;
        Ok(clbits)
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
        self.check_runnable(circuit)?;
        if circuit.n_clbits() > 64 {
            return Err(Error::InvalidArgument(
                "counting outcomes needs at most 64 classical bits".into(),
            ));
        }
        let mut counts = BTreeMap::new();
        if let Some(measured) = terminal_measurements(circuit) {
            // Simulate once, then read each sample's measured qubits into
            // their classical bits (a later measurement into the same bit
            // wins).
            let mut state = self.clone();
            state.run(&circuit.slice_with_phase(0, circuit.len()))?;
            for index in state.sample(shots, rng) {
                let mut bits = 0u64;
                for &(qubit, clbit) in &measured {
                    let one = index >> qubit & 1 == 1;
                    bits = (bits & !(1 << clbit)) | (one as u64) << clbit;
                }
                *counts.entry(bits).or_insert(0) += 1;
            }
            return Ok(counts);
        }
        let pieces = circuit.pieces();
        let (prefix, rest) = match pieces.split_first() {
            Some((Piece::Gates(gates), rest)) => (Some(gates), rest),
            _ => (None, &pieces[..]),
        };
        let mut start = self.clone();
        if let Some(gates) = prefix {
            start.run(gates)?;
        }
        for _ in 0..shots {
            let mut state = start.clone();
            let mut clbits = vec![false; circuit.n_clbits()];
            state.follow_pieces(rest, &mut clbits, rng)?;
            let bits = clbits
                .iter()
                .enumerate()
                .fold(0u64, |acc, (c, &b)| acc | (b as u64) << c);
            *counts.entry(bits).or_insert(0) += 1;
        }
        Ok(counts)
    }

    fn check_runnable(&self, circuit: &Circuit) -> Result<(), Error> {
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

    fn follow(
        &mut self,
        circuit: &Circuit,
        clbits: &mut [bool],
        rng: &mut impl Rng,
    ) -> Result<(), Error> {
        self.follow_pieces(&circuit.pieces(), clbits, rng)
    }

    fn follow_pieces(
        &mut self,
        pieces: &[Piece<'_>],
        clbits: &mut [bool],
        rng: &mut impl Rng,
    ) -> Result<(), Error> {
        for piece in pieces {
            match piece {
                Piece::Gates(gates) => self.run(gates)?,
                Piece::Instruction(Instruction::Measure { qubit, clbit }) => {
                    clbits[*clbit] = self.measure(*qubit, rng)?;
                }
                Piece::Instruction(Instruction::Reset { qubit }) => self.reset(*qubit, rng)?,
                Piece::Instruction(Instruction::IfElse {
                    condition,
                    then,
                    otherwise,
                }) => {
                    let branch = if condition.holds(clbits) {
                        then
                    } else {
                        otherwise
                    };
                    self.follow(branch, clbits, rng)?;
                }
            }
        }
        Ok(())
    }
}

/// `(qubit, clbit)` for each measurement, if every instruction is a
/// measurement after the last gate.
fn terminal_measurements(circuit: &Circuit) -> Option<Vec<(usize, usize)>> {
    circuit
        .instructions()
        .iter()
        .map(|(position, instruction)| match instruction {
            Instruction::Measure { qubit, clbit } if *position == circuit.len() => {
                Some((*qubit, *clbit))
            }
            _ => None,
        })
        .collect()
}
