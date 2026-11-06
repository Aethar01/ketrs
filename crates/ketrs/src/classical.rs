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

/// A classical expression over a circuit's bits, evaluated as an unsigned
/// 64-bit integer; as a condition, any nonzero value holds. Booleans are 0
/// or 1. Covers Qiskit's classical expressions on bits and registers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Expr {
    /// A classical bit: 0 or 1.
    Bit(usize),
    /// Bits read as an integer, the first least significant.
    Register(Vec<usize>),
    /// A constant.
    Value(u64),
    /// Logical not: 1 if the operand is 0, else 0.
    Not(Box<Expr>),
    /// Bitwise not of the operand's low `width` bits.
    BitNot {
        /// The operand.
        value: Box<Expr>,
        /// Its width in bits.
        width: u32,
    },
    /// A binary operation.
    Binary {
        /// The operation.
        op: BinaryOp,
        /// Left operand.
        left: Box<Expr>,
        /// Right operand.
        right: Box<Expr>,
    },
}

/// Operations for [`Expr::Binary`]. Logical operations and comparisons
/// give 0 or 1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryOp {
    /// Logical and: both nonzero.
    And,
    /// Logical or: either nonzero.
    Or,
    /// Bitwise and.
    BitAnd,
    /// Bitwise or.
    BitOr,
    /// Bitwise exclusive or.
    BitXor,
    /// `==`.
    Equal,
    /// `!=`.
    NotEqual,
    /// `<`.
    Less,
    /// `<=`.
    LessEqual,
    /// `>`.
    Greater,
    /// `>=`.
    GreaterEqual,
}

impl Expr {
    /// `op(left, right)`.
    pub fn binary(op: BinaryOp, left: Expr, right: Expr) -> Expr {
        Expr::Binary {
            op,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    /// The expression's value for the classical bits `bits`.
    pub fn eval(&self, bits: &[bool]) -> u64 {
        match self {
            Expr::Bit(c) => bits[*c] as u64,
            Expr::Register(cs) => cs
                .iter()
                .enumerate()
                .fold(0, |acc, (k, &c)| acc | (bits[c] as u64) << k),
            Expr::Value(v) => *v,
            Expr::Not(e) => (e.eval(bits) == 0) as u64,
            Expr::BitNot { value, width } => {
                let mask = if *width >= 64 {
                    u64::MAX
                } else {
                    (1 << width) - 1
                };
                !value.eval(bits) & mask
            }
            Expr::Binary { op, left, right } => {
                let (a, b) = (left.eval(bits), right.eval(bits));
                match op {
                    BinaryOp::And => (a != 0 && b != 0) as u64,
                    BinaryOp::Or => (a != 0 || b != 0) as u64,
                    BinaryOp::BitAnd => a & b,
                    BinaryOp::BitOr => a | b,
                    BinaryOp::BitXor => a ^ b,
                    BinaryOp::Equal => (a == b) as u64,
                    BinaryOp::NotEqual => (a != b) as u64,
                    BinaryOp::Less => (a < b) as u64,
                    BinaryOp::LessEqual => (a <= b) as u64,
                    BinaryOp::Greater => (a > b) as u64,
                    BinaryOp::GreaterEqual => (a >= b) as u64,
                }
            }
        }
    }

    /// Whether the expression is nonzero for `bits`.
    pub fn holds(&self, bits: &[bool]) -> bool {
        self.eval(bits) != 0
    }

    /// Every classical bit the expression reads, and the widest register.
    fn bits_read(&self, out: &mut Vec<usize>) -> usize {
        match self {
            Expr::Bit(c) => {
                out.push(*c);
                1
            }
            Expr::Register(cs) => {
                out.extend(cs);
                cs.len()
            }
            Expr::Value(_) => 0,
            Expr::Not(e) | Expr::BitNot { value: e, .. } => e.bits_read(out),
            Expr::Binary { left, right, .. } => left.bits_read(out).max(right.bits_read(out)),
        }
    }

    /// Check the expression against a circuit with `n_clbits` bits.
    pub(crate) fn validate(&self, n_clbits: usize) -> Result<(), crate::error::Error> {
        let mut bits = Vec::new();
        if self.bits_read(&mut bits) > 64 {
            return Err(crate::error::Error::InvalidArgument(
                "registers in conditions hold at most 64 bits".into(),
            ));
        }
        match bits.into_iter().find(|&c| c >= n_clbits) {
            Some(c) => Err(crate::error::Error::InvalidArgument(format!(
                "classical bit {c} out of range for {n_clbits} bits"
            ))),
            None => Ok(()),
        }
    }
}

impl From<Condition> for Expr {
    fn from(c: Condition) -> Expr {
        Expr::binary(
            BinaryOp::Equal,
            Expr::Register(c.clbits),
            Expr::Value(c.value),
        )
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
    /// Set `clbit` to 1 if `value` is nonzero, else 0. Purely classical:
    /// the state is untouched.
    Assign {
        /// The classical bit written.
        clbit: usize,
        /// The value stored, evaluated when the instruction runs.
        value: Expr,
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
        condition: Expr,
        /// Run when the test holds.
        then: Box<Circuit>,
        /// Run when it doesn't.
        otherwise: Box<Circuit>,
    },
    /// Run `body` while `condition` holds, testing before each pass. The
    /// body is a circuit on the same qubits and classical bits.
    While {
        /// The test.
        condition: Expr,
        /// Run on each pass.
        body: Box<Circuit>,
    },
    /// Run each circuit of `passes` in turn: a `for` loop, one circuit per
    /// value of its variable. Loops are normally unrolled into plain gates;
    /// this form is for bodies that [`Break`](Self::Break) or
    /// [`Continue`](Self::Continue).
    For {
        /// The body for each pass, in order.
        passes: Vec<Circuit>,
    },
    /// Leave the innermost enclosing `while` or `for` loop.
    Break,
    /// Go on to the innermost enclosing loop's next pass.
    Continue,
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
