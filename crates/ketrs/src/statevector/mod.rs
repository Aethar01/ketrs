//! Dense statevector simulation.
//!
//! Stores all `2ⁿ` amplitudes. [`Statevector`] uses complex amplitudes
//! (16 bytes each, so 28 qubits needs 4 GiB); [`RealStatevector`] uses `f64`
//! at half the memory and time for circuits whose gates are all real.

pub(crate) mod eigen;
pub(crate) mod evolve;
pub(crate) mod gradient;
pub(crate) mod hamiltonian;
pub(crate) mod kernels;
pub(crate) mod krylov;
mod measure;
mod noise;
pub(crate) mod shots;

pub use eigen::{EigenOptions, Eigenstate};

use crate::circuit::{Circuit, Operation};
use crate::error::Error;
use crate::scalar::Scalar;
use kernels::Prepared;
use num_complex::Complex64 as C64;
use rayon::prelude::*;

/// Circuits are partly fused (see [`Circuit::fused_where`]) before running
/// on states at least this large. Below it the state fits in cache and the
/// specialised kernels beat fused generic matrices.
const FUSION_MIN_QUBITS: usize = 16;

/// Whether to fuse circuits on `n_qubits`. From `GATHER_MIN_QUBITS`, tiles
/// take gates on any few qubits, so fusing saves no passes over memory,
/// and it hides diagonal gates, which tiles take on any qubits.
fn fuses(n_qubits: usize) -> bool {
    (FUSION_MIN_QUBITS..kernels::GATHER_MIN_QUBITS).contains(&n_qubits)
}

/// A pure state on `n` qubits, stored as `2ⁿ` amplitudes of type `T`.
///
/// Usually named through [`Statevector`] or [`RealStatevector`].
#[derive(Clone, Debug, PartialEq)]
pub struct DenseState<T: Scalar> {
    n_qubits: usize,
    amps: Vec<T>,
}

/// A step for [`DenseState::apply_steps`].
pub(crate) enum Step {
    /// A gate.
    Gate(Box<Operation>),
    /// Any `2^m × 2^m` row-major matrix on `qubits`, basis bit `k` for
    /// `qubits[k]`.
    Matrix {
        qubits: Vec<usize>,
        matrix: Vec<C64>,
    },
}

/// A statevector with complex amplitudes; runs any circuit.
pub type Statevector = DenseState<C64>;

/// A statevector with real amplitudes. Runs circuits whose gates are all
/// real (e.g. H, X, Z, Ry, CX, CZ, SWAP, Pauli rotations with an odd number
/// of Y factors) at half the memory of [`Statevector`].
pub type RealStatevector = DenseState<f64>;

/// [`Statevector`] in single precision: half the memory, and errors of
/// about `1e-6` instead of `1e-15`.
pub type Statevector32 = DenseState<num_complex::Complex32>;

/// [`RealStatevector`] in single precision: a quarter of the memory of
/// [`Statevector`].
pub type RealStatevector32 = DenseState<f32>;

impl<T: Scalar> DenseState<T> {
    /// The all-zero state `|0…0⟩`.
    pub fn zero(n_qubits: usize) -> Self {
        Self::basis(n_qubits, 0)
    }

    /// The computational basis state with the given index.
    ///
    /// # Panics
    /// If `index >= 2^n_qubits`.
    pub fn basis(n_qubits: usize, index: usize) -> Self {
        let dim = 1usize << n_qubits;
        assert!(
            index < dim,
            "basis index {index} out of range for {n_qubits} qubits"
        );
        let mut amps = vec![T::ZERO; dim];
        amps[index] = T::ONE;
        DenseState { n_qubits, amps }
    }

    /// Wrap existing amplitudes. The length must be a power of two; the
    /// vector is not normalised.
    pub fn from_amplitudes(amps: Vec<T>) -> Result<Self, Error> {
        if !amps.len().is_power_of_two() {
            return Err(Error::InvalidLength(amps.len()));
        }
        Ok(DenseState {
            n_qubits: amps.len().trailing_zeros() as usize,
            amps,
        })
    }

    /// Run `circuit` on `|0…0⟩`. Fails with [`Error::NotReal`] if `T` can't
    /// represent one of its operations.
    pub fn try_from_circuit(circuit: &Circuit) -> Result<Self, Error> {
        let mut state = Self::zero(circuit.n_qubits());
        state.run(circuit)?;
        Ok(state)
    }

    /// Number of qubits.
    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    /// The amplitudes, indexed by basis state.
    pub fn amplitudes(&self) -> &[T] {
        &self.amps
    }

    /// Take ownership of the amplitudes.
    pub fn into_amplitudes(self) -> Vec<T> {
        self.amps
    }

    /// Measurement probabilities `|ψᵢ|²` for each basis state.
    pub fn probabilities(&self) -> Vec<f64> {
        self.amps.iter().map(|a| a.norm_sqr()).collect()
    }

    /// The squared norm `⟨ψ|ψ⟩`.
    pub fn norm_sqr(&self) -> f64 {
        self.amps.iter().map(|a| a.norm_sqr()).sum()
    }

    /// The inner product `⟨self|other⟩`.
    pub fn inner(&self, other: &Self) -> Result<T, Error> {
        self.check_same_width(other)?;
        Ok(hamiltonian::chunked_sum(self.amps.len(), |range| {
            self.amps[range.clone()]
                .iter()
                .zip(&other.amps[range])
                .map(|(a, b)| a.conj() * *b)
                .sum::<T>()
        }))
    }

    /// Add a qubit in `|0⟩` as the new most significant qubit: the state
    /// becomes `|0⟩ ⊗ |ψ⟩`, keeping every existing qubit's index.
    pub fn add_qubit(&mut self) {
        self.amps.resize(2 * self.amps.len(), T::ZERO);
        self.n_qubits += 1;
    }

    /// Apply one operation.
    pub fn apply(&mut self, op: &Operation) -> Result<(), Error> {
        op.validate(self.n_qubits)?;
        Prepared::new(op)?.apply(&mut self.amps);
        Ok(())
    }

    /// Apply every operation in `circuit`, which must have the same width.
    /// Nothing is applied if an operation can't be represented in `T`.
    pub fn run(&mut self, circuit: &Circuit) -> Result<(), Error> {
        circuit.check_unitary()?;
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
        let phase = T::from_complex(C64::cis(circuit.global_phase())).ok_or_else(|| {
            Error::NotReal(format!(
                "global phase {} is not real",
                circuit.global_phase()
            ))
        })?;
        // Circuit operations are validated on insertion, so only the width
        // needed checking.
        let prepared = if fuses(self.n_qubits) {
            // Gates on low qubits are already cheap through cache blocking,
            // so only fuse gates that need a full pass over memory.
            let fused = circuit.fused_where(|op| op.max_qubit() >= kernels::CACHE_BLOCK_QUBITS);
            prepare(fused.ops())?
        } else {
            prepare(circuit.ops())?
        };
        kernels::run(&mut self.amps, &prepared);
        if phase != T::ONE {
            self.amps.par_iter_mut().for_each(|a| *a *= phase);
        }
        Ok(())
    }

    /// Apply `steps` in order, gates and arbitrary matrices alike, through
    /// the tiled kernels (so runs of them on few qubits share passes over
    /// memory).
    pub(crate) fn apply_steps(&mut self, steps: &[Step]) -> Result<(), Error> {
        let prepared = steps
            .iter()
            .map(|step| match step {
                Step::Gate(op) => Prepared::new(op),
                Step::Matrix { qubits, matrix } => {
                    let matrix = matrix
                        .iter()
                        .map(|&c| T::from_complex(c))
                        .collect::<Option<Vec<T>>>()
                        .ok_or_else(|| Error::NotReal("matrix has complex entries".into()))?;
                    Ok(Prepared::dense(qubits, &matrix))
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        kernels::run(&mut self.amps, &prepared);
        Ok(())
    }

    fn check_qubit(&self, qubit: usize) -> Result<(), Error> {
        if qubit >= self.n_qubits {
            return Err(Error::QubitOutOfRange {
                qubit,
                n_qubits: self.n_qubits,
            });
        }
        Ok(())
    }

    fn check_same_width(&self, other: &Self) -> Result<(), Error> {
        if self.n_qubits != other.n_qubits {
            return Err(Error::QubitCountMismatch {
                expected: self.n_qubits,
                found: other.n_qubits,
            });
        }
        Ok(())
    }
}

fn prepare<T: Scalar>(ops: &[Operation]) -> Result<Vec<Prepared<T>>, Error> {
    ops.iter().map(Prepared::new).collect()
}

/// Infallible and fallible `from_circuit` for the complex and real types.
macro_rules! from_circuit {
    (complex: $($t:ty),*; real: $($r:ty),*) => {
        $(impl DenseState<$t> {
            /// Run `circuit` on `|0…0⟩`.
            ///
            /// # Panics
            /// If `circuit` has unbound parameters; use
            /// [`try_from_circuit`](Self::try_from_circuit) to get an error
            /// instead. Complex amplitudes represent every operation, so
            /// nothing else fails.
            pub fn from_circuit(circuit: &Circuit) -> Self {
                Self::try_from_circuit(circuit).unwrap_or_else(|e| panic!("{e}"))
            }
        })*
        $(impl DenseState<$r> {
            /// Run `circuit` on `|0…0⟩`, or fail with [`Error::NotReal`] if
            /// it contains a complex gate.
            pub fn from_circuit(circuit: &Circuit) -> Result<Self, Error> {
                Self::try_from_circuit(circuit)
            }
        })*
    };
}

from_circuit!(complex: C64, num_complex::Complex32; real: f64, f32);
