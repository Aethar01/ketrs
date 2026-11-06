//! Python bindings for symmetry-sector simulation.

use super::{DType, Observable, PyCircuit, PyStatevector, State, rng_from_seed, to_py_err};
use ketrs::{Complex64 as C64, RealSectorStatevector, SectorBasis, SectorStatevector};
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

#[derive(Clone)]
enum Kind {
    Complex(SectorStatevector),
    Real(RealSectorStatevector),
}

macro_rules! dispatch {
    ($state:expr, $s:ident => $body:expr) => {
        match $state {
            Kind::Complex($s) => $body,
            Kind::Real($s) => $body,
        }
    };
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
    pub(crate) fn complex(state: SectorStatevector) -> Self {
        PySectorStatevector {
            state: Kind::Complex(state),
            rng: rng_from_seed(None),
        }
    }

    pub(crate) fn real(state: RealSectorStatevector) -> Self {
        PySectorStatevector {
            state: Kind::Real(state),
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
        let state = match DType::parse(dtype)? {
            DType::Complex => {
                Kind::Complex(SectorStatevector::basis_state(b, state).map_err(to_py_err)?)
            }
            DType::Real => {
                Kind::Real(RealSectorStatevector::basis_state(b, state).map_err(to_py_err)?)
            }
        };
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
        let state = match &state.state {
            State::Complex(s) => {
                Kind::Complex(SectorStatevector::from_dense(b, s).map_err(to_py_err)?)
            }
            State::Real(s) => {
                Kind::Real(RealSectorStatevector::from_dense(b, s).map_err(to_py_err)?)
            }
        };
        Ok(PySectorStatevector {
            state,
            rng: rng_from_seed(seed),
        })
    }

    #[getter]
    fn n_qubits(&self) -> usize {
        dispatch!(&self.state, s => s.n_qubits())
    }

    /// Number of stored amplitudes.
    #[getter]
    fn dim(&self) -> usize {
        dispatch!(&self.state, s => s.basis().dim())
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        match self.state {
            Kind::Complex(_) => "complex128",
            Kind::Real(_) => "float64",
        }
    }

    #[getter]
    fn basis(&self) -> PySectorBasis {
        PySectorBasis {
            inner: dispatch!(&self.state, s => s.basis().clone()),
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
        dispatch!(&mut self.state, s => py.detach(|| s.run(circ))).map_err(to_py_err)
    }

    /// The amplitudes in sector order (see `SectorBasis.states`).
    fn amplitudes<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        dispatch!(&self.state, s => PyArray1::from_slice(py, s.amplitudes()).into_any())
    }

    /// Probability of each sector basis state, in sector order.
    fn probabilities<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, dispatch!(&self.state, s => s.probabilities()))
    }

    /// The same state as a full `Statevector`.
    fn to_dense(&self) -> PyStatevector {
        let state = match &self.state {
            Kind::Complex(s) => State::Complex(s.to_dense()),
            Kind::Real(s) => State::Real(s.to_dense()),
        };
        PyStatevector::wrap(state, self.rng.clone())
    }

    /// `⟨self|other⟩`; both must share the sector and dtype.
    fn inner(&self, other: PyRef<'_, PySectorStatevector>) -> PyResult<C64> {
        match (&self.state, &other.state) {
            (Kind::Complex(a), Kind::Complex(b)) => a.inner(b).map_err(to_py_err),
            (Kind::Real(a), Kind::Real(b)) => {
                a.inner(b).map(|x| C64::new(x, 0.0)).map_err(to_py_err)
            }
            _ => Err(PyValueError::new_err("states have different dtypes")),
        }
    }

    /// `⟨ψ|H|ψ⟩`; terms leaving the sector contribute nothing.
    fn expectation(&self, py: Python<'_>, observable: Observable) -> PyResult<f64> {
        let sum = observable.into_sum()?;
        dispatch!(&self.state, s => py.detach(|| s.expectation(&sum))).map_err(to_py_err)
    }

    /// `H|ψ⟩`; `H` must conserve the sector.
    fn apply_pauli_sum(&self, py: Python<'_>, hamiltonian: Observable) -> PyResult<Self> {
        let sum = hamiltonian.into_sum()?;
        let state = match &self.state {
            Kind::Complex(s) => {
                Kind::Complex(py.detach(|| s.apply_pauli_sum(&sum)).map_err(to_py_err)?)
            }
            Kind::Real(s) => Kind::Real(py.detach(|| s.apply_pauli_sum(&sum)).map_err(to_py_err)?),
        };
        Ok(PySectorStatevector {
            state,
            rng: self.rng.clone(),
        })
    }

    /// Evolve by `exp(-iHt)` within the sector; needs dtype='complex'.
    fn evolve(&mut self, py: Python<'_>, hamiltonian: Observable, time: f64) -> PyResult<()> {
        let sum = hamiltonian.into_sum()?;
        match &mut self.state {
            Kind::Complex(s) => py.detach(|| s.evolve(&sum, time)).map_err(to_py_err),
            Kind::Real(_) => Err(PyValueError::new_err(
                "exp(-iHt) is complex; evolve needs dtype='complex'",
            )),
        }
    }

    /// Draw `shots` full basis-state indices.
    fn sample<'py>(&mut self, py: Python<'py>, shots: usize) -> Bound<'py, PyArray1<usize>> {
        let rng = &mut self.rng;
        PyArray1::from_vec(py, dispatch!(&self.state, s => s.sample(shots, rng)))
    }

    /// Counts keyed by bitstring, most significant qubit first.
    fn sample_counts<'py>(
        &mut self,
        py: Python<'py>,
        shots: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let n = self.n_qubits();
        let rng = &mut self.rng;
        let counts = dispatch!(&self.state, s => s.sample_counts(shots, rng));
        let dict = PyDict::new(py);
        for (index, count) in counts {
            dict.set_item(format!("{index:0n$b}"), count)?;
        }
        Ok(dict)
    }
}
