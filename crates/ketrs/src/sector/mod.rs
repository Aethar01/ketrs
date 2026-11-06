//! Simulation restricted to a symmetry sector.
//!
//! When every operation conserves the Hamming weight of some sets of qubits
//! (particle number, the number of each spin, a charge, …) the state stays in
//! the subspace with fixed weights. [`SectorState`] stores only that
//! subspace: at half filling of `n` qubits that is `C(n, n/2)` amplitudes
//! rather than `2ⁿ`, about 5.7× fewer at 20 qubits and 6.9× at 30.
//!
//! Circuits are fused first, so a hopping term written as `XX` and `YY`
//! rotations on the same pair becomes one conserving two-qubit block. Every
//! operation is then checked, and one that would leave the sector is
//! rejected with [`Error::BreaksSymmetry`] before anything is applied.
//! Hamiltonians are checked the same way, and their eigenstates and time
//! evolution are computed within the sector.

mod basis;

pub use basis::SectorBasis;

use crate::circuit::{Circuit, Operation};
use crate::error::Error;
use crate::fusion::local_matrix;
use crate::observable::{PauliMasks, PauliSum};
use crate::scalar::Scalar;
use crate::statevector::DenseState;
use crate::statevector::eigen::{EigenOptions, Eigenstate, lowest_eigenvectors};
use crate::statevector::evolve::krylov_evolve;
use crate::statevector::hamiltonian::{GroupedPauliSum, Imaginary, diagonal_entry};
use crate::statevector::krylov::LinearOperator;
use basis::SegmentStates;
use num_complex::Complex64 as C64;
use rand::Rng;
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::sync::Arc;

/// Matrix entries this small that would leave the sector are taken as
/// rounding error from fusion and dropped.
const LEAK_TOLERANCE: f64 = 1e-12;

/// A pure state within a symmetry sector, stored as one amplitude per
/// sector basis state (in [`SectorBasis`] order).
#[derive(Clone, Debug)]
pub struct SectorState<T: Scalar> {
    basis: Arc<SectorBasis>,
    amps: Vec<T>,
}

/// A complex statevector within a sector.
pub type SectorStatevector = SectorState<C64>;

/// A real statevector within a sector.
pub type RealSectorStatevector = SectorState<f64>;

impl<T: Scalar> SectorState<T> {
    /// The computational basis state `state` (a full `n`-qubit index), which
    /// must be in the sector.
    pub fn basis_state(basis: Arc<SectorBasis>, state: usize) -> Result<Self, Error> {
        let position = basis.index(state).ok_or_else(|| {
            Error::InvalidArgument(format!("basis state {state:b} is not in the sector"))
        })?;
        let mut amps = vec![T::ZERO; basis.dim()];
        amps[position] = T::ONE;
        Ok(SectorState { basis, amps })
    }

    /// The part of `state` inside the sector (not renormalised).
    pub fn from_dense(basis: Arc<SectorBasis>, state: &DenseState<T>) -> Result<Self, Error> {
        if state.n_qubits() != basis.n_qubits() {
            return Err(Error::QubitCountMismatch {
                expected: basis.n_qubits(),
                found: state.n_qubits(),
            });
        }
        let full = state.amplitudes();
        let mut amps = vec![T::ZERO; basis.dim()];
        for_each_segment_mut(&basis, &mut amps, |_, states, out| {
            for (o, j) in out.iter_mut().zip(states.iter()) {
                *o = full[j];
            }
        });
        Ok(SectorState { basis, amps })
    }

    /// The same state as a full statevector.
    pub fn to_dense(&self) -> DenseState<T> {
        let mut full = vec![T::ZERO; 1 << self.basis.n_qubits()];
        for (start, states) in self.basis.segments() {
            for (k, j) in states.iter().enumerate() {
                full[j] = self.amps[start + k];
            }
        }
        DenseState::from_amplitudes(full).expect("length is a power of two")
    }

    /// The sector this state lives in.
    pub fn basis(&self) -> &Arc<SectorBasis> {
        &self.basis
    }

    /// Number of qubits.
    pub fn n_qubits(&self) -> usize {
        self.basis.n_qubits()
    }

    /// The amplitudes, in sector order; [`SectorBasis::state`] gives the
    /// basis state of each.
    pub fn amplitudes(&self) -> &[T] {
        &self.amps
    }

    /// The squared norm.
    pub fn norm_sqr(&self) -> f64 {
        self.amps.iter().map(|a| a.norm_sqr()).sum()
    }

    /// Measurement probability of each sector basis state, in sector order.
    pub fn probabilities(&self) -> Vec<f64> {
        self.amps.iter().map(|a| a.norm_sqr()).collect()
    }

    /// `⟨self|other⟩`; both must be in the same sector.
    pub fn inner(&self, other: &Self) -> Result<T, Error> {
        self.check_same_sector(other)?;
        Ok(self
            .amps
            .iter()
            .zip(&other.amps)
            .map(|(a, b)| a.conj() * *b)
            .sum())
    }

    /// Draw `shots` full basis-state indices from the Born distribution.
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
        (0..shots)
            .map(|_| {
                let r = rng.random::<f64>() * total;
                let position = cumulative
                    .partition_point(|&c| c <= r)
                    .min(cumulative.len() - 1);
                self.basis.state(position)
            })
            .collect()
    }

    /// As [`sample`](Self::sample), tallied by basis state.
    pub fn sample_counts(&self, shots: usize, rng: &mut impl Rng) -> BTreeMap<usize, usize> {
        let mut counts = BTreeMap::new();
        for j in self.sample(shots, rng) {
            *counts.entry(j).or_insert(0) += 1;
        }
        counts
    }

    /// Apply `circuit`, which must conserve the sector. It is fused first,
    /// then checked; nothing is applied if any operation would leave the
    /// sector ([`Error::BreaksSymmetry`]) or isn't representable in `T`.
    pub fn run(&mut self, circuit: &Circuit) -> Result<(), Error> {
        circuit.check_unitary()?;
        if circuit.n_qubits() != self.n_qubits() {
            return Err(Error::QubitCountMismatch {
                expected: self.n_qubits(),
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
        let ops = circuit
            .fused()
            .ops()
            .iter()
            .map(|op| SectorOp::new(op, &self.basis))
            .collect::<Result<Vec<_>, _>>()?;
        let mut scratch = vec![T::ZERO; self.amps.len()];
        for op in &ops {
            op.apply(&self.basis, &mut self.amps, &mut scratch);
        }
        if phase != T::ONE {
            self.amps.par_iter_mut().for_each(|a| *a *= phase);
        }
        Ok(())
    }

    /// `⟨ψ|H|ψ⟩`. Terms that leave the sector contribute nothing and are
    /// skipped, so `H` need not conserve the symmetry here.
    pub fn expectation(&self, observable: &PauliSum) -> Result<f64, Error> {
        let op = SectorPauliSum::<T>::new(&self.basis, observable, Imaginary::Skip, false, false)?;
        Ok(op.expectation(&self.amps))
    }

    /// `H|ψ⟩`; `H` must conserve the sector.
    pub fn apply_pauli_sum(&self, hamiltonian: &PauliSum) -> Result<Self, Error> {
        let op =
            SectorPauliSum::<T>::new(&self.basis, hamiltonian, Imaginary::Reject, false, true)?;
        let mut out = vec![T::ZERO; self.amps.len()];
        op.apply(&self.amps, &mut out);
        Ok(SectorState {
            basis: self.basis.clone(),
            amps: out,
        })
    }

    /// The `k` lowest eigenstates of `hamiltonian` within `basis`, which it
    /// must conserve, by Lanczos iteration with deflation.
    pub fn lowest_eigenstates(
        basis: Arc<SectorBasis>,
        hamiltonian: &PauliSum,
        k: usize,
        options: &EigenOptions,
    ) -> Result<Vec<Eigenstate<Self>>, Error> {
        let op = SectorPauliSum::<T>::new(&basis, hamiltonian, Imaginary::Reject, true, true)?;
        Ok(lowest_eigenvectors(&op, basis.dim(), k, options)?
            .into_iter()
            .map(|(energy, amps, residual)| Eigenstate {
                energy,
                state: SectorState {
                    basis: basis.clone(),
                    amps,
                },
                residual,
            })
            .collect())
    }

    fn check_same_sector(&self, other: &Self) -> Result<(), Error> {
        let same = Arc::ptr_eq(&self.basis, &other.basis)
            || (self.basis.n_qubits() == other.basis.n_qubits()
                && self.basis.constraints() == other.basis.constraints());
        if same {
            Ok(())
        } else {
            Err(Error::InvalidArgument(
                "states are in different sectors".into(),
            ))
        }
    }
}

impl SectorStatevector {
    /// Evolve by `exp(-iHt)` within the sector, which `H` must conserve.
    /// See [`Statevector::evolve`](crate::Statevector::evolve).
    pub fn evolve(&mut self, hamiltonian: &PauliSum, time: f64) -> Result<(), Error> {
        let mut shift = 0.0;
        let mut rest = PauliSum::new();
        for (c, s) in hamiltonian.simplify(0.0).terms() {
            if s.weight() == 0 {
                shift += c;
            } else {
                rest.add(*c, s.clone());
            }
        }
        if !rest.terms().is_empty() {
            let op = SectorPauliSum::<C64>::new(&self.basis, &rest, Imaginary::Reject, true, true)?;
            krylov_evolve(&op, &mut self.amps, time, true)?;
        }
        if shift != 0.0 {
            let phase = C64::cis(-shift * time);
            self.amps.par_iter_mut().for_each(|a| *a *= phase);
        }
        Ok(())
    }
}

/// Call `f(first position, states, amplitudes)` for each segment of the
/// sector in parallel, with that segment's slice of `amps`.
fn for_each_segment_mut<T: Send>(
    basis: &SectorBasis,
    amps: &mut [T],
    f: impl Fn(usize, SegmentStates<'_>, &mut [T]) + Sync,
) {
    let mut parts = Vec::new();
    let mut rest = amps;
    for (start, states) in basis.segments() {
        let (head, tail) = rest.split_at_mut(states.len());
        parts.push((start, states, head));
        rest = tail;
    }
    parts
        .into_par_iter()
        .for_each(|(start, states, slice)| f(start, states, slice));
}

/// The value of `qubits` in basis state `j`: bit `t` is `qubits[t]`.
#[inline]
fn local(j: usize, qubits: &[usize]) -> usize {
    qubits
        .iter()
        .enumerate()
        .fold(0, |acc, (t, &q)| acc | ((j >> q) & 1) << t)
}

/// Basis state `j` with `qubits` set to the local configuration `c`.
#[inline]
fn with_local(j: usize, qubits: &[usize], c: usize) -> usize {
    qubits
        .iter()
        .enumerate()
        .fold(j, |acc, (t, &q)| (acc & !(1 << q)) | ((c >> t) & 1) << q)
}

/// An operation in the form its sector kernel needs.
enum SectorOp<T> {
    /// Multiply by `d[local configuration]`, in place.
    Diagonal { qubits: Vec<usize>, d: Vec<T> },
    /// `out_j = Σ_c rows[r(j)][c] ψ_{j with qubits = c}`, over the allowed,
    /// non-zero entries only.
    Local {
        qubits: Vec<usize>,
        rows: Vec<Vec<(usize, T)>>,
    },
    /// A diagonal Pauli rotation: `plus` or `minus` by the parity of `j ∧ z`.
    PauliPhase { z: usize, plus: T, minus: T },
    /// A Pauli rotation that flips only unconstrained qubits.
    PauliPair {
        masks: PauliMasks,
        cos: T,
        factor: T,
    },
}

impl<T: Scalar> SectorOp<T> {
    fn new(op: &Operation, basis: &SectorBasis) -> Result<Self, Error> {
        let breaks = || Error::BreaksSymmetry(format!("{op:?} leaves the sector"));
        let not_real =
            || Error::NotReal(format!("{op:?} has complex entries; use a complex state"));
        if let Some((qubits, m)) = local_matrix(op) {
            let mut rows = vec![Vec::new(); m.len()];
            for (r, row) in m.iter().enumerate() {
                for (c, &v) in row.iter().enumerate() {
                    if !basis.allows(&qubits, r, c) {
                        if v.norm() > LEAK_TOLERANCE {
                            return Err(breaks());
                        }
                    } else if v != C64::new(0.0, 0.0) {
                        rows[r].push((c, T::from_complex(v).ok_or_else(not_real)?));
                    }
                }
            }
            let diagonal = rows
                .iter()
                .enumerate()
                .all(|(r, row)| row.iter().all(|&(c, _)| c == r));
            return Ok(if diagonal {
                let d = rows
                    .iter()
                    .map(|row| row.first().map_or(T::ZERO, |&(_, v)| v))
                    .collect();
                SectorOp::Diagonal { qubits, d }
            } else {
                SectorOp::Local { qubits, rows }
            });
        }
        let Operation::PauliRotation { pauli, theta } = op else {
            unreachable!("operations on more than two qubits are Pauli rotations")
        };
        let masks = pauli.masks();
        let (sin, cos) = (theta / 2.0).sin_cos();
        if masks.x == 0 {
            return Ok(SectorOp::PauliPhase {
                z: masks.z,
                plus: T::from_complex(C64::cis(-theta / 2.0)).ok_or_else(not_real)?,
                minus: T::from_complex(C64::cis(theta / 2.0)).ok_or_else(not_real)?,
            });
        }
        if basis.constraints().iter().any(|&(m, _)| m & masks.x != 0) {
            return Err(breaks());
        }
        Ok(SectorOp::PauliPair {
            masks,
            cos: T::from_real(cos),
            factor: T::from_complex(C64::new(0.0, -sin) * masks.y_phase()).ok_or_else(not_real)?,
        })
    }

    /// Apply to `amps`, using `scratch` (same length) for gathers.
    fn apply(&self, basis: &SectorBasis, amps: &mut Vec<T>, scratch: &mut Vec<T>) {
        match self {
            SectorOp::Diagonal { qubits, d } => {
                for_each_segment_mut(basis, amps, |_, states, out| {
                    for (a, j) in out.iter_mut().zip(states.iter()) {
                        *a *= d[local(j, qubits)];
                    }
                })
            }
            SectorOp::PauliPhase { z, plus, minus } => {
                for_each_segment_mut(basis, amps, |_, states, out| {
                    for (a, j) in out.iter_mut().zip(states.iter()) {
                        *a *= if (j & z).count_ones() % 2 == 1 {
                            *minus
                        } else {
                            *plus
                        };
                    }
                })
            }
            SectorOp::Local { qubits, rows } => {
                let psi = &amps[..];
                for_each_segment_mut(basis, scratch, |_, states, out| {
                    for (o, j) in out.iter_mut().zip(states.iter()) {
                        *o = rows[local(j, qubits)]
                            .iter()
                            .map(|&(c, v)| v * psi[basis.index_unchecked(with_local(j, qubits, c))])
                            .sum();
                    }
                });
                std::mem::swap(amps, scratch);
            }
            SectorOp::PauliPair { masks, cos, factor } => {
                let psi = &amps[..];
                for_each_segment_mut(basis, scratch, |start, states, out| {
                    for (k, (o, j)) in out.iter_mut().zip(states.iter()).enumerate() {
                        let i = j ^ masks.x;
                        let f = if masks.z_sign(i) { -*factor } else { *factor };
                        *o = *cos * psi[start + k] + f * psi[basis.index_unchecked(i)];
                    }
                });
                std::mem::swap(amps, scratch);
            }
        }
    }
}

/// A Pauli sum acting on sector vectors.
struct SectorPauliSum<'a, T> {
    basis: &'a SectorBasis,
    grouped: GroupedPauliSum<T>,
    /// The diagonal per sector position, when tabulated.
    table: Option<Vec<f64>>,
}

impl<'a, T: Scalar> SectorPauliSum<'a, T> {
    /// With `check`, fail with [`Error::BreaksSymmetry`] unless `H` maps the
    /// sector into itself.
    fn new(
        basis: &'a SectorBasis,
        hamiltonian: &PauliSum,
        imaginary: Imaginary,
        tabulate_diagonal: bool,
        check: bool,
    ) -> Result<Self, Error> {
        let grouped = GroupedPauliSum::<T>::new(hamiltonian, basis.n_qubits(), imaginary)?;
        if check {
            let leaks = basis.segments().par_bridge().any(|(_, states)| {
                states.iter().any(|i| {
                    grouped.groups.iter().any(|g| {
                        !basis.contains(i ^ g.x)
                            && g.coefficient(i).to_complex().norm() > LEAK_TOLERANCE
                    })
                })
            });
            if leaks {
                return Err(Error::BreaksSymmetry(
                    "the Hamiltonian does not conserve the sector".into(),
                ));
            }
        }
        let table = tabulate_diagonal.then(|| {
            let mut table = vec![0.0; basis.dim()];
            for_each_segment_mut(basis, &mut table, |_, states, out| {
                for (d, j) in out.iter_mut().zip(states.iter()) {
                    *d = diagonal_entry(&grouped.diagonal, j);
                }
            });
            table
        });
        Ok(SectorPauliSum {
            basis,
            grouped,
            table,
        })
    }

    /// `(Hψ)` at sector position `p`, basis state `j`; terms leaving the
    /// sector are dropped.
    #[inline]
    fn row(&self, psi: &[T], p: usize, j: usize) -> T {
        let d = match &self.table {
            Some(table) => table[p],
            None => diagonal_entry(&self.grouped.diagonal, j),
        };
        let mut acc = T::from_real(d) * psi[p];
        for g in &self.grouped.groups {
            let i = j ^ g.x;
            if let Some(q) = self.basis.index(i) {
                acc += g.coefficient(i) * psi[q];
            }
        }
        acc
    }

    fn expectation(&self, psi: &[T]) -> f64 {
        // Per-segment sums, added in order so the result is reproducible.
        let parts: Vec<f64> = self
            .basis
            .segments()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(start, states)| {
                states
                    .iter()
                    .enumerate()
                    .map(|(k, j)| {
                        (psi[start + k].conj() * self.row(psi, start + k, j))
                            .to_complex()
                            .re
                    })
                    .sum::<f64>()
            })
            .collect();
        parts.into_iter().sum()
    }
}

impl<T: Scalar> LinearOperator<T> for SectorPauliSum<'_, T> {
    fn apply(&self, psi: &[T], out: &mut [T]) {
        for_each_segment_mut(self.basis, out, |start, states, out| {
            for (k, (o, j)) in out.iter_mut().zip(states.iter()).enumerate() {
                *o = self.row(psi, start + k, j);
            }
        });
    }
}
