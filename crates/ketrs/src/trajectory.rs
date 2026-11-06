//! Following circuits with measurement, reset and classical control on any
//! engine.
//!
//! Each engine implements [`Trajectory`]: running a unitary piece,
//! measuring a qubit, and sampling. [`run_trajectory`] follows one random
//! history; [`sample_circuit`] counts outcomes over many, simulating the
//! deterministic prefix once, or the whole circuit once when every
//! measurement comes at the end.

use crate::circuit::Circuit;
use crate::classical::{Instruction, Piece};
use crate::error::Error;
use crate::gate::Gate;
use rand::Rng;
use std::collections::BTreeMap;

/// What an engine needs to follow a circuit's random history.
pub(crate) trait Trajectory: Sized {
    fn n_qubits(&self) -> usize;
    /// Run a circuit of gates only.
    fn run_gates(&mut self, gates: &Circuit) -> Result<(), Error>;
    /// Measure `qubit`, collapsing the state, and return the outcome.
    fn measure_qubit<R: Rng>(&mut self, qubit: usize, rng: &mut R) -> Result<bool, Error>;
    /// An independent copy, for another shot.
    fn duplicate(&self) -> Result<Self, Error>;
    /// For each of `shots` samples of every qubit, the values of `qubits`
    /// packed with `qubits[k]` at bit `k`. The state is unchanged.
    fn sample_qubits<R: Rng>(
        &mut self,
        shots: usize,
        qubits: &[usize],
        rng: &mut R,
    ) -> Result<Vec<u64>, Error>;
}

/// Run `circuit` on `engine` following one random history; returns the
/// classical bits (all starting at 0) as the circuit left them.
pub(crate) fn run_trajectory<E: Trajectory, R: Rng>(
    engine: &mut E,
    circuit: &Circuit,
    rng: &mut R,
) -> Result<Vec<bool>, Error> {
    check_runnable(engine, circuit)?;
    let mut clbits = vec![false; circuit.n_clbits()];
    follow(engine, &circuit.pieces(), &mut clbits, rng)?;
    Ok(clbits)
}

/// Run `circuit` from `engine`'s state `shots` times, counting each final
/// value of the classical bits (classical bit 0 least significant).
pub(crate) fn sample_circuit<E: Trajectory, R: Rng>(
    engine: &E,
    circuit: &Circuit,
    shots: usize,
    rng: &mut R,
) -> Result<BTreeMap<u64, usize>, Error> {
    check_runnable(engine, circuit)?;
    if circuit.n_clbits() > 64 {
        return Err(Error::InvalidArgument(
            "counting outcomes needs at most 64 classical bits".into(),
        ));
    }
    let mut counts = BTreeMap::new();
    if let Some(measured) = terminal_measurements(circuit).filter(|m| m.len() <= 64) {
        // Simulate once, then read each sample's measured qubits into their
        // classical bits (a later measurement into the same bit wins).
        let mut state = engine.duplicate()?;
        state.run_gates(&circuit.slice_with_phase(0, circuit.len()))?;
        let qubits: Vec<usize> = measured.iter().map(|&(q, _)| q).collect();
        for sample in state.sample_qubits(shots, &qubits, rng)? {
            let mut bits = 0u64;
            for (k, &(_, clbit)) in measured.iter().enumerate() {
                let one = sample >> k & 1 == 1;
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
    let mut start = engine.duplicate()?;
    if let Some(gates) = prefix {
        start.run_gates(gates)?;
    }
    for _ in 0..shots {
        let mut state = start.duplicate()?;
        let mut clbits = vec![false; circuit.n_clbits()];
        follow(&mut state, rest, &mut clbits, rng)?;
        let bits = clbits
            .iter()
            .enumerate()
            .fold(0u64, |acc, (c, &b)| acc | (b as u64) << c);
        *counts.entry(bits).or_insert(0) += 1;
    }
    Ok(counts)
}

fn check_runnable<E: Trajectory>(engine: &E, circuit: &Circuit) -> Result<(), Error> {
    if circuit.n_qubits() != engine.n_qubits() {
        return Err(Error::QubitCountMismatch {
            expected: engine.n_qubits(),
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

fn follow<E: Trajectory, R: Rng>(
    engine: &mut E,
    pieces: &[Piece<'_>],
    clbits: &mut [bool],
    rng: &mut R,
) -> Result<(), Error> {
    for piece in pieces {
        match piece {
            Piece::Gates(gates) => engine.run_gates(gates)?,
            Piece::Instruction(Instruction::Measure { qubit, clbit }) => {
                clbits[*clbit] = engine.measure_qubit(*qubit, rng)?;
            }
            Piece::Instruction(Instruction::Reset { qubit }) => {
                if engine.measure_qubit(*qubit, rng)? {
                    let mut flip = Circuit::new(engine.n_qubits());
                    flip.gate(Gate::X, *qubit);
                    engine.run_gates(&flip)?;
                }
            }
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
                follow(engine, &branch.pieces(), clbits, rng)?;
            }
        }
    }
    Ok(())
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
