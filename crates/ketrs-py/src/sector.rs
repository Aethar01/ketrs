//! Python bindings for symmetry-sector simulation.

use super::{DType, Observable, PyCircuit, PyStatevector, State, rng_from_seed, to_py_err};
use ketrs::statevector::Eigenstate;
use ketrs::{
    Complex64 as C64, RealSectorStatevector, RealSectorStatevector32, Scalar, SectorBasis,
    SectorState, SectorStatevector, SectorStatevector32,
};
use numpy::PyArray1;
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::rngs::StdRng;
use std::sync::Arc;

/// The basis states of `n_qubits` qubits with fixed Hamming weights: for
/// each `(qubits, weight)` constraint, exactly `weight` of `qubits` are
/// |1⟩. Other qubits are unrestricted. For example
/// `SectorBasis(8, [(range(8), 4)])` is half filling of 8 sites.
#[pyclass(name = "SectorBasis", module = "ketrs", frozen)]
pub(crate) struct PySectorBasis {
    pub(crate) inner: Arc<SectorBasis>,
}

#[pymethods]
impl PySectorBasis {
    #[new]
    fn new(n_qubits: usize, constraints: Vec<(Vec<usize>, usize)>) -> PyResult<Self> {
        Ok(PySectorBasis {
            inner: Arc::new(SectorBasis::new(n_qubits, &constraints).map_err(to_py_err)?),
        })
    }

    #[getter]
    fn n_qubits(&self) -> usize {
        self.inner.n_qubits()
    }

    /// Number of basis states in the sector.
    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn __len__(&self) -> usize {
        self.inner.dim()
    }

    fn __contains__(&self, state: usize) -> bool {
        self.inner.contains(state)
    }

    fn __repr__(&self) -> String {
        format!(
            "SectorBasis(n_qubits={}, dim={})",
            self.inner.n_qubits(),
            self.inner.dim()
        )
    }

    /// Position of basis state `state` in the sector, or None.
    fn index(&self, state: usize) -> Option<usize> {
        self.inner.index(state)
    }

    /// The basis state at `position`.
    fn state(&self, position: usize) -> PyResult<usize> {
        if position >= self.inner.dim() {
            return Err(PyIndexError::new_err("position outside the sector"));
        }
        Ok(self.inner.state(position))
    }

    /// Every basis state in the sector, in order.
    fn states<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<usize>> {
        PyArray1::from_vec(
            py,
            (0..self.inner.dim()).map(|p| self.inner.state(p)).collect(),
        )
    }
}

/// A sector state of any dtype; with eigenpairs, one per eigenstate.
#[derive(Clone)]
pub(crate) enum Kind<
    C = SectorStatevector,
    R = RealSectorStatevector,
    C32 = SectorStatevector32,
    R32 = RealSectorStatevector32,
> {
    Complex(C),
    Real(R),
    Complex32(C32),
    Real32(R32),
}

impl Kind {
    fn dtype(&self) -> DType {
        match self {
            Kind::Complex(_) => DType::Complex,
            Kind::Real(_) => DType::Real,
            Kind::Complex32(_) => DType::Complex32,
            Kind::Real32(_) => DType::Real32,
        }
    }
}

/// Eigenpairs found for one dtype.
pub(crate) type FoundEigenstates = Kind<
    Vec<Eigenstate<SectorStatevector>>,
    Vec<Eigenstate<RealSectorStatevector>>,
    Vec<Eigenstate<SectorStatevector32>>,
    Vec<Eigenstate<RealSectorStatevector32>>,
>;

/// The eigenpairs found for some dtype, as Python sector states.
pub(crate) fn eigenstates(found: FoundEigenstates) -> Vec<(f64, f64, PySectorStatevector)> {
    fn each<S>(
        found: Vec<Eigenstate<S>>,
        wrap: fn(S) -> Kind,
    ) -> Vec<(f64, f64, PySectorStatevector)> {
        found
            .into_iter()
            .map(|e| {
                (
                    e.energy,
                    e.residual,
                    PySectorStatevector::wrap(wrap(e.state)),
                )
            })
            .collect()
    }
    match found {
        Kind::Complex(f) => each(f, Kind::Complex),
        Kind::Real(f) => each(f, Kind::Real),
        Kind::Complex32(f) => each(f, Kind::Complex32),
        Kind::Real32(f) => each(f, Kind::Real32),
    }
}

/// A state within a symmetry sector, storing one amplitude per sector basis
/// state. Circuits and Hamiltonians must conserve the sector; anything that
/// would leave it raises `ValueError` before the state changes.
#[pyclass(name = "SectorStatevector", module = "ketrs")]
pub(crate) struct PySectorStatevector {
    state: Kind,
    rng: StdRng,
}

impl PySectorStatevector {
    fn wrap(state: Kind) -> Self {
        PySectorStatevector {
            state,
            rng: rng_from_seed(None),
        }
    }
}

#[pymethods]
impl PySectorStatevector {
    /// The computational basis state `state`, which must be in `basis`.
    #[new]
    #[pyo3(signature = (basis, state, seed=None, dtype="complex"))]
    fn new(
        basis: PyRef<'_, PySectorBasis>,
        state: usize,
        seed: Option<u64>,
        dtype: &str,
    ) -> PyResult<Self> {
        let b = basis.inner.clone();
        let state = by_dtype!(Kind, DType::parse(dtype)?, T => {
            SectorState::<T>::basis_state(b, state).map_err(to_py_err)?
        });
        Ok(PySectorStatevector {
            state,
            rng: rng_from_seed(seed),
        })
    }

    /// The part of a full `Statevector` inside `basis` (not renormalised).
    #[staticmethod]
    #[pyo3(signature = (basis, state, seed=None))]
    fn from_dense(
        basis: PyRef<'_, PySectorBasis>,
        state: PyRef<'_, PyStatevector>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let b = basis.inner.clone();
        let state = convert!(State => Kind, &state.state, s => {
            SectorState::from_dense(b, s).map_err(to_py_err)?
        });
        Ok(PySectorStatevector {
            state,
            rng: rng_from_seed(seed),
        })
    }

    #[getter]
    fn n_qubits(&self) -> usize {
        dispatch!(Kind, &self.state, s => s.n_qubits())
    }

    /// Number of stored amplitudes.
    #[getter]
    fn dim(&self) -> usize {
        dispatch!(Kind, &self.state, s => s.basis().dim())
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        self.state.dtype().name()
    }

    #[getter]
    fn basis(&self) -> PySectorBasis {
        PySectorBasis {
            inner: dispatch!(Kind, &self.state, s => s.basis().clone()),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "SectorStatevector(n_qubits={}, dim={}, dtype={})",
            self.n_qubits(),
            self.dim(),
            self.dtype()
        )
    }

    /// Apply `circuit`, which must conserve the sector.
    fn run(&mut self, py: Python<'_>, circuit: PyRef<'_, PyCircuit>) -> PyResult<()> {
        let circ = &circuit.inner;
        dispatch!(Kind, &mut self.state, s => py.detach(|| s.run(circ))).map_err(to_py_err)
    }

    /// The amplitudes in sector order (see `SectorBasis.states`).
    fn amplitudes<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        dispatch!(Kind, &self.state, s => PyArray1::from_slice(py, s.amplitudes()).into_any())
    }

    /// Probability of each sector basis state, in sector order.
    fn probabilities<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, dispatch!(Kind, &self.state, s => s.probabilities()))
    }

    /// The same state as a full `Statevector`.
    fn to_dense(&self) -> PyStatevector {
        let state = convert!(Kind => State, &self.state, s => s.to_dense());
        PyStatevector::wrap(state, self.rng.clone())
    }

    /// `⟨self|other⟩`; both must share the sector and dtype.
    fn inner(&self, other: PyRef<'_, PySectorStatevector>) -> PyResult<C64> {
        same_dtype!(Kind, &self.state, &other.state, (a, b) => {
            a.inner(b).map(Scalar::to_complex).map_err(to_py_err)
        })
    }

    /// `⟨ψ|H|ψ⟩`; terms leaving the sector contribute nothing.
    fn expectation(&self, py: Python<'_>, observable: Observable) -> PyResult<f64> {
        let sum = observable.into_sum()?;
        dispatch!(Kind, &self.state, s => py.detach(|| s.expectation(&sum))).map_err(to_py_err)
    }

    /// `H|ψ⟩`; `H` must conserve the sector.
    fn apply_pauli_sum(&self, py: Python<'_>, hamiltonian: Observable) -> PyResult<Self> {
        let sum = hamiltonian.into_sum()?;
        let state = convert!(Kind => Kind, &self.state, s => {
            py.detach(|| s.apply_pauli_sum(&sum)).map_err(to_py_err)?
        });
        Ok(PySectorStatevector {
            state,
            rng: self.rng.clone(),
        })
    }

    /// Evolve by `exp(-iHt)` within the sector; needs a complex dtype.
    fn evolve(&mut self, py: Python<'_>, hamiltonian: Observable, time: f64) -> PyResult<()> {
        let sum = hamiltonian.into_sum()?;
        match &mut self.state {
            Kind::Complex(s) => py.detach(|| s.evolve(&sum, time)).map_err(to_py_err),
            Kind::Complex32(s) => py.detach(|| s.evolve(&sum, time)).map_err(to_py_err),
            Kind::Real(_) | Kind::Real32(_) => Err(PyValueError::new_err(
                "exp(-iHt) is complex; evolve needs dtype='complex' or 'complex64'",
            )),
        }
    }

    /// Draw `shots` full basis-state indices.
    fn sample<'py>(&mut self, py: Python<'py>, shots: usize) -> Bound<'py, PyArray1<usize>> {
        let rng = &mut self.rng;
        PyArray1::from_vec(py, dispatch!(Kind, &self.state, s => s.sample(shots, rng)))
    }

    /// Counts keyed by bitstring, most significant qubit first.
    fn sample_counts<'py>(
        &mut self,
        py: Python<'py>,
        shots: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let n = self.n_qubits();
        let rng = &mut self.rng;
        let counts = dispatch!(Kind, &self.state, s => s.sample_counts(shots, rng));
        let dict = PyDict::new(py);
        for (index, count) in counts {
            dict.set_item(format!("{index:0n$b}"), count)?;
        }
        Ok(dict)
    }
}
