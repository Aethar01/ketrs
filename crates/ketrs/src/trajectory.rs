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

/// Passes of one `while` loop after which a trajectory gives up, rather
/// than hang on a condition that never fails.
const MAX_LOOP_PASSES: usize = 1_000_000;

/// What an engine needs to follow a circuit's random history.
pub(crate) trait Trajectory: Sized {
    /// Whether running gates is deterministic, so that a shared prefix or a
    /// final state can serve every shot. Not so for noisy trajectories.
    const DETERMINISTIC: bool = true;

    fn n_qubits(&self) -> usize;
    /// Run a circuit of gates only.
    fn run_gates(&mut self, gates: &Circuit) -> Result<(), Error>;
    /// Run the circuit's gates as part of a trajectory, with any noise.
    fn run_gates_in<R: Rng>(&mut self, gates: &Circuit, _rng: &mut R) -> Result<(), Error> {
        self.run_gates(gates)
    }
    /// The bit recorded for a measurement with `outcome`: the outcome
    /// itself unless reading it out can fail.
    fn record<R: Rng>(&self, outcome: bool, _rng: &mut R) -> bool {
        outcome
    }
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
    top_level(follow(engine, &circuit.pieces(), &mut clbits, rng)?)?;
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
    if !E::DETERMINISTIC {
        for _ in 0..shots {
            let mut state = engine.duplicate()?;
            let mut clbits = vec![false; circuit.n_clbits()];
            top_level(follow(&mut state, &circuit.pieces(), &mut clbits, rng)?)?;
            *counts.entry(pack_bits(&clbits)).or_insert(0) += 1;
        }
        return Ok(counts);
    }
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
    // The state up to the first measurement or control flow is the same for
    // every shot, and so are the bits that assignments there set.
    let pieces = circuit.pieces();
    let mut start = engine.duplicate()?;
    let mut initial = vec![false; circuit.n_clbits()];
    let mut shared = 0;
    for piece in &pieces {
        match piece {
            Piece::Gates(gates) => start.run_gates(gates)?,
            Piece::Instruction(Instruction::Assign { clbit, value }) => {
                initial[*clbit] = value.holds(&initial);
            }
            _ => break,
        }
        shared += 1;
    }
    for _ in 0..shots {
        let mut state = start.duplicate()?;
        let mut clbits = initial.clone();
        top_level(follow(&mut state, &pieces[shared..], &mut clbits, rng)?)?;
        *counts.entry(pack_bits(&clbits)).or_insert(0) += 1;
    }
    Ok(counts)
}

/// Classical bits as an integer, bit 0 least significant.
fn pack_bits(clbits: &[bool]) -> u64 {
    clbits
        .iter()
        .enumerate()
        .fold(0u64, |acc, (c, &b)| acc | (b as u64) << c)
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

/// How a block ended: normally, or by `break` or `continue`, which pass up
/// through `if` branches to the innermost loop.
#[derive(Clone, Copy, PartialEq)]
enum Flow {
    Normal,
    Break,
    Continue,
}

/// A whole circuit must not end its trajectory with a stray `break`.
fn top_level(flow: Flow) -> Result<(), Error> {
    match flow {
        Flow::Normal => Ok(()),
        _ => Err(Error::InvalidArgument(
            "`break` or `continue` outside a loop".into(),
        )),
    }
}

fn follow<E: Trajectory, R: Rng>(
    engine: &mut E,
    pieces: &[Piece<'_>],
    clbits: &mut [bool],
    rng: &mut R,
) -> Result<Flow, Error> {
    for piece in pieces {
        match piece {
            Piece::Gates(gates) => engine.run_gates_in(gates, rng)?,
            Piece::Instruction(Instruction::Measure { qubit, clbit }) => {
                let outcome = engine.measure_qubit(*qubit, rng)?;
                clbits[*clbit] = engine.record(outcome, rng);
            }
            Piece::Instruction(Instruction::Assign { clbit, value }) => {
                clbits[*clbit] = value.holds(clbits);
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
                let flow = follow(engine, &branch.pieces(), clbits, rng)?;
                if flow != Flow::Normal {
                    return Ok(flow);
                }
            }
            Piece::Instruction(Instruction::While { condition, body }) => {
                let pieces = body.pieces();
                let mut passes = 0;
                while condition.holds(clbits) {
                    if passes == MAX_LOOP_PASSES {
                        return Err(Error::NotConverged(format!(
                            "a while loop ran {MAX_LOOP_PASSES} times without its condition failing"
                        )));
                    }
                    passes += 1;
                    if follow(engine, &pieces, clbits, rng)? == Flow::Break {
                        break;
                    }
                }
            }
            Piece::Instruction(Instruction::For { passes }) => {
                for body in passes {
                    if follow(engine, &body.pieces(), clbits, rng)? == Flow::Break {
                        break;
                    }
                }
            }
            Piece::Instruction(Instruction::Break) => return Ok(Flow::Break),
            Piece::Instruction(Instruction::Continue) => return Ok(Flow::Continue),
        }
    }
    Ok(Flow::Normal)
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
