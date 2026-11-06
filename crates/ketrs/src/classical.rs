//! Measurement, reset and classical control inside circuits.
//!
//! A circuit's gates are its [`ops`](crate::Circuit::ops); these
//! non-unitary [`Instruction`]s sit between them, each at a position in the
//! gate list (see [`Circuit::instructions`](crate::Circuit::instructions)).
//! Engines' `run` methods take only unitary circuits; circuits with
//! instructions run one random history at a time with `run_trajectory`, or
//! many with `sample_circuit`, on statevectors
//! ([`DenseState::run_trajectory`](crate::DenseState::run_trajectory)),
//! MPS ([`Mps::run_trajectory`](crate::Mps::run_trajectory)) and GPU
//! states alike.

use crate::circuit::Circuit;

/// A test on classical bits: the bits `clbits`, read as an integer with
/// `clbits[0]` least significant, equal `value`. As in Qiskit, a single bit
/// is `{clbits: [c], value: 0 or 1}` and a register lists all its bits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Condition {
    /// The bits read, least significant first.
    pub clbits: Vec<usize>,
    /// The value they must hold.
    pub value: u64,
}

impl Condition {
    /// Whether the condition holds for the classical bits `bits`.
    pub fn holds(&self, bits: &[bool]) -> bool {
        self.clbits
            .iter()
            .enumerate()
            .all(|(k, &c)| bits[c] == (self.value >> k & 1 == 1))
    }
}

/// A non-unitary step of a circuit.
#[derive(Clone, Debug, PartialEq)]
pub enum Instruction {
    /// Measure `qubit` in the computational basis into `clbit`.
    Measure {
        /// The qubit measured.
        qubit: usize,
        /// The classical bit written.
        clbit: usize,
    },
    /// Measure `qubit` and set it to `|0⟩`.
    Reset {
        /// The qubit reset.
        qubit: usize,
    },
    /// Run `then` if `condition` holds, otherwise `otherwise`. Both are
    /// circuits on the same qubits and classical bits as the parent, and may
    /// themselves contain instructions.
    IfElse {
        /// The test.
        condition: Condition,
        /// Run when the test holds.
        then: Box<Circuit>,
        /// Run when it doesn't.
        otherwise: Box<Circuit>,
    },
}

/// A circuit cut at its instructions: runs of gates and the instructions
/// between them, in order.
pub(crate) enum Piece<'a> {
    Gates(Circuit),
    Instruction(&'a Instruction),
}

impl Circuit {
    /// The circuit cut into runs of gates and the instructions between
    /// them. The global phase goes with the first run of gates.
    pub(crate) fn pieces(&self) -> Vec<Piece<'_>> {
        let mut pieces = Vec::new();
        let mut start = 0;
        let mut phase = self.global_phase();
        let mut gates = |from: usize, to: usize, pieces: &mut Vec<Piece<'_>>| {
            if to > from || phase != 0.0 {
                let mut run = self.slice(from, to);
                run.set_global_phase(phase);
                phase = 0.0;
                pieces.push(Piece::Gates(run));
            }
        };
        for (position, instruction) in self.instructions() {
            gates(start, *position, &mut pieces);
            pieces.push(Piece::Instruction(instruction));
            start = *position;
        }
        gates(start, self.len(), &mut pieces);
        pieces
    }
}
