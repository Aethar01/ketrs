use std::fmt;

/// Errors from building circuits and constructing or evolving states.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// A qubit index is not less than the register size.
    QubitOutOfRange {
        /// The offending qubit index.
        qubit: usize,
        /// Number of qubits in the register.
        n_qubits: usize,
    },
    /// An operation names the same qubit more than once.
    DuplicateQubit(usize),
    /// A circuit and a state have different numbers of qubits.
    QubitCountMismatch {
        /// Qubits in the state.
        expected: usize,
        /// Qubits in the circuit.
        found: usize,
    },
    /// An amplitude vector whose length is not a power of two.
    InvalidLength(usize),
    /// A Pauli label containing a character other than `I`, `X`, `Y`, `Z`.
    InvalidPauli(char),
    /// An argument outside its allowed range, with an explanation.
    InvalidArgument(String),
    /// An iterative method stopped before reaching its tolerance.
    NotConverged(String),
    /// A complex operation or value given to a real statevector.
    NotReal(String),
    /// An operation or Hamiltonian that would take a state out of its
    /// symmetry sector.
    BreaksSymmetry(String),
    /// No usable GPU, or a GPU operation failed.
    Gpu(String),
    /// Source text (OpenQASM) that couldn't be read, at a 1-based line.
    Parse {
        /// The line of the problem.
        line: usize,
        /// What went wrong.
        message: String,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::QubitOutOfRange { qubit, n_qubits } => {
                write!(f, "qubit {qubit} out of range for {n_qubits} qubits")
            }
            Error::DuplicateQubit(q) => write!(f, "qubit {q} used more than once"),
            Error::QubitCountMismatch { expected, found } => {
                write!(f, "expected {expected} qubits, found {found}")
            }
            Error::InvalidLength(len) => {
                write!(f, "amplitude vector length {len} is not a power of two")
            }
            Error::InvalidPauli(ch) => write!(f, "invalid Pauli label character {ch:?}"),
            Error::InvalidArgument(msg)
            | Error::NotConverged(msg)
            | Error::NotReal(msg)
            | Error::BreaksSymmetry(msg)
            | Error::Gpu(msg) => f.write_str(msg),
            Error::Parse { line, message } => write!(f, "line {line}: {message}"),
        }
    }
}

impl std::error::Error for Error {}
