//! Quantum computer simulation backend.
//!
//! Qubit ordering follows Qiskit: qubit 0 is the least significant bit of a
//! basis-state index, so on three qubits `|q2 q1 q0⟩ = |001⟩` is index 1.
//!
//! ```
//! use ketrs::{Circuit, Statevector};
//!
//! let mut bell = Circuit::new(2);
//! bell.h(0).cx(0, 1);
//!
//! let state = Statevector::from_circuit(&bell);
//! let probs = state.probabilities();
//! assert!((probs[0] - 0.5).abs() < 1e-12);
//! assert!((probs[3] - 0.5).abs() < 1e-12);
//! ```

pub mod circuit;
pub mod classical;
pub mod density;
mod error;
mod fusion;
pub mod gate;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod library;
pub mod mps;
pub mod observable;
pub mod parameter;
pub mod qasm;
pub mod scalar;
pub mod sector;
pub mod statevector;
mod trajectory;

pub use circuit::{Circuit, Operation};
pub use classical::{BinaryOp, Condition, Expr, Instruction};
pub use density::{Channel, DensityMatrix, NoiseModel};
pub use error::Error;
pub use gate::Gate;
#[cfg(feature = "gpu")]
pub use gpu::{GpuContext, GpuPrecision, GpuStatevector};
pub use mps::{DmrgOptions, DmrgResult, Mps, MpsOptions};
pub use num_complex::{Complex32, Complex64};
pub use observable::{Pauli, PauliString, PauliSum};
pub use parameter::{Angle, Parameter};
pub use scalar::{ComplexScalar, Scalar};
pub use sector::{
    RealSectorStatevector, RealSectorStatevector32, SectorBasis, SectorState, SectorStatevector,
    SectorStatevector32,
};
pub use statevector::{DenseState, RealStatevector, RealStatevector32, Statevector, Statevector32};
