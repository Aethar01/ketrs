//! Python bindings for density matrices and noise.

use super::{Observable, PyCircuit, PyStatevector, State, rng_from_seed, to_py_err};
use ketrs::{Channel, Complex64 as C64, DensityMatrix, NoiseModel, PauliString};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::rngs::StdRng;

/// A quantum channel given by Kraus operators, applied to chosen qubits of
/// a `DensityMatrix` (or after gates, through a `NoiseModel`).
#[pyclass(name = "Channel", module = "ketrs", frozen)]
pub(crate) struct PyChannel {
    inner: Channel,
}

#[pymethods]
impl PyChannel {
    /// A channel from Kraus operators (square complex arrays of one
    /// power-of-two size, with Σ K†K = 1). Basis index bit b is the
    /// channel's b-th qubit, as in Qiskit.
    #[staticmethod]
    fn from_kraus(kraus: Vec<PyReadonlyArray2<'_, C64>>) -> PyResult<Self> {
        let ops = kraus
            .iter()
            .map(|k| {
                let a = k.as_array();
                a.rows().into_iter().map(|row| row.to_vec()).collect()
            })
            .collect();
        Ok(PyChannel {
            inner: Channel::from_kraus(ops).map_err(to_py_err)?,
        })
    }

    /// Depolarizing noise, as Qiskit Aer's `depolarizing_error(p, n)`:
    /// ρ → (1 − p) ρ + p · I/2ⁿ.
    #[staticmethod]
    #[pyo3(signature = (p, n_qubits=1))]
    fn depolarizing(p: f64, n_qubits: usize) -> PyResult<Self> {
        Self::wrap(Channel::depolarizing(p, n_qubits))
    }

    /// A mixture of Pauli strings, given as `[(label, probability), …]`;
    /// the identity takes the remaining probability.
    #[staticmethod]
    fn pauli(n_qubits: usize, terms: Vec<(String, f64)>) -> PyResult<Self> {
        let terms = terms
            .iter()
            .map(|(label, p)| Ok((PauliString::from_label(label).map_err(to_py_err)?, *p)))
            .collect::<PyResult<Vec<_>>>()?;
        Self::wrap(Channel::pauli(n_qubits, &terms))
    }

    /// X with probability `p`.
    #[staticmethod]
    fn bit_flip(p: f64) -> PyResult<Self> {
        Self::wrap(Channel::bit_flip(p))
    }

    /// Z with probability `p`.
    #[staticmethod]
    fn phase_flip(p: f64) -> PyResult<Self> {
        Self::wrap(Channel::phase_flip(p))
    }

    /// |1⟩ decays to |0⟩ with probability `gamma`.
    #[staticmethod]
    fn amplitude_damping(gamma: f64) -> PyResult<Self> {
        Self::wrap(Channel::amplitude_damping(gamma))
    }

    /// Off-diagonal elements shrink by √(1 − λ).
    #[staticmethod]
    fn phase_damping(lam: f64) -> PyResult<Self> {
        Self::wrap(Channel::phase_damping(lam))
    }

    #[getter]
    fn n_qubits(&self) -> usize {
        self.inner.n_qubits()
    }

    /// The Kraus operators as 2-D arrays.
    fn kraus<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyArray2<C64>>>> {
        let dim = 1 << self.inner.n_qubits();
        self.inner
            .kraus()
            .iter()
            .map(|k| PyArray1::from_slice(py, k).reshape([dim, dim]))
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "Channel(n_qubits={}, kraus={})",
            self.inner.n_qubits(),
            self.inner.kraus().len()
        )
    }
}

impl PyChannel {
    fn wrap(channel: Result<Channel, ketrs::Error>) -> PyResult<Self> {
        Ok(PyChannel {
            inner: channel.map_err(to_py_err)?,
        })
    }
}

/// Noise after every gate: `one_qubit` (a 1-qubit `Channel`) on each qubit
/// of a gate, except that two-qubit gates get `two_qubit` on the pair when
/// it is given. `readout = (p10, p01)` gives the chances that a measurement
/// records 1 for outcome 0 and 0 for outcome 1; statevector trajectories
/// (`Statevector.run`/`sample_circuit` with `noise=`) apply it.
#[pyclass(name = "NoiseModel", module = "ketrs", frozen)]
pub(crate) struct PyNoiseModel {
    pub(crate) inner: NoiseModel,
}

#[pymethods]
impl PyNoiseModel {
    #[new]
    #[pyo3(signature = (one_qubit=None, two_qubit=None, readout=None))]
    fn new(
        one_qubit: Option<PyRef<'_, PyChannel>>,
        two_qubit: Option<PyRef<'_, PyChannel>>,
        readout: Option<(f64, f64)>,
    ) -> PyResult<Self> {
        let check = |c: &Option<PyRef<'_, PyChannel>>, k: usize| match c {
            Some(c) if c.inner.n_qubits() != k => Err(PyValueError::new_err(format!(
                "expected a {k}-qubit channel, got {} qubits",
                c.inner.n_qubits()
            ))),
            _ => Ok(()),
        };
        check(&one_qubit, 1)?;
        check(&two_qubit, 2)?;
        if let Some((a, b)) = readout
            && !((0.0..=1.0).contains(&a) && (0.0..=1.0).contains(&b))
        {
            return Err(PyValueError::new_err(
                "readout error probabilities must lie in [0, 1]",
            ));
        }
        Ok(PyNoiseModel {
            inner: NoiseModel {
                one_qubit: one_qubit.map(|c| c.inner.clone()),
                two_qubit: two_qubit.map(|c| c.inner.clone()),
                readout: readout.map(|(a, b)| [a, b]),
            },
        })
    }
}

/// A mixed state of `n` qubits as a density matrix: twice the qubits of a
/// statevector in memory. Runs unitary circuits, channels, and circuits
/// with a noise model after every gate.
#[pyclass(name = "DensityMatrix", module = "ketrs")]
pub(crate) struct PyDensityMatrix {
    inner: DensityMatrix,
    rng: StdRng,
}

#[pymethods]
impl PyDensityMatrix {
    /// |0…0⟩⟨0…0| on `n_qubits` qubits.
    #[new]
    #[pyo3(signature = (n_qubits, seed=None))]
    fn new(n_qubits: usize, seed: Option<u64>) -> Self {
        PyDensityMatrix {
            inner: DensityMatrix::zero(n_qubits),
            rng: rng_from_seed(seed),
        }
    }

    /// |ψ⟩⟨ψ| for a complex statevector.
    #[staticmethod]
    #[pyo3(signature = (state, seed=None))]
    fn from_statevector(state: PyRef<'_, PyStatevector>, seed: Option<u64>) -> PyResult<Self> {
        let State::Complex(psi) = &state.state else {
            return Err(PyValueError::new_err(
                "density matrices are built from complex128 statevectors",
            ));
        };
        Ok(PyDensityMatrix {
            inner: DensityMatrix::from_statevector(psi),
            rng: rng_from_seed(seed),
        })
    }

    /// A density matrix from a square complex array (not checked for being
    /// a valid state).
    #[staticmethod]
    #[pyo3(signature = (matrix, seed=None))]
    fn from_matrix(matrix: PyReadonlyArray2<'_, C64>, seed: Option<u64>) -> PyResult<Self> {
        let dim = matrix.shape()[0];
        if matrix.shape()[1] != dim || !dim.is_power_of_two() {
            return Err(PyValueError::new_err(
                "expected a square array with a power-of-two size",
            ));
        }
        let rho: Vec<C64> = matrix.as_array().iter().copied().collect();
        Ok(PyDensityMatrix {
            inner: DensityMatrix::from_matrix(dim.trailing_zeros() as usize, &rho)
                .map_err(to_py_err)?,
            rng: rng_from_seed(seed),
        })
    }

    #[getter]
    fn n_qubits(&self) -> usize {
        self.inner.n_qubits()
    }

    fn __repr__(&self) -> String {
        format!("DensityMatrix(n_qubits={})", self.inner.n_qubits())
    }

    /// Apply a unitary circuit: ρ → UρU†.
    fn run(&mut self, py: Python<'_>, circuit: PyRef<'_, PyCircuit>) -> PyResult<()> {
        let (rho, circ) = (&mut self.inner, &circuit.inner);
        py.detach(|| rho.run(circ)).map_err(to_py_err)
    }

    /// Run `circuit` with `noise` after every gate.
    fn run_noisy(
        &mut self,
        py: Python<'_>,
        circuit: PyRef<'_, PyCircuit>,
        noise: PyRef<'_, PyNoiseModel>,
    ) -> PyResult<()> {
        let (rho, circ, noise) = (&mut self.inner, &circuit.inner, &noise.inner);
        py.detach(|| rho.run_noisy(circ, noise)).map_err(to_py_err)
    }

    /// Apply `channel` to `qubits`.
    fn apply_channel(&mut self, channel: PyRef<'_, PyChannel>, qubits: Vec<usize>) -> PyResult<()> {
        self.inner
            .apply_channel(&channel.inner, &qubits)
            .map_err(to_py_err)
    }

    /// The matrix as a 2-D complex array.
    fn to_matrix<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<C64>>> {
        let dim = 1 << self.inner.n_qubits();
        PyArray1::from_vec(py, self.inner.to_matrix()).reshape([dim, dim])
    }

    /// Measurement probability of each basis state.
    fn probabilities<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.probabilities())
    }

    /// Tr(ρH). Accepts a `PauliSum`, a label, or `(label, coefficient)`
    /// pairs.
    fn expectation(&self, observable: Observable) -> PyResult<f64> {
        let sum = observable.into_sum()?;
        self.inner.expectation(&sum).map_err(to_py_err)
    }

    /// Tr ρ.
    fn trace(&self) -> f64 {
        self.inner.trace()
    }

    /// Tr ρ².
    fn purity(&self) -> f64 {
        self.inner.purity()
    }

    /// ⟨ψ|ρ|ψ⟩ for a complex statevector.
    fn fidelity(&self, state: PyRef<'_, PyStatevector>) -> PyResult<f64> {
        let State::Complex(psi) = &state.state else {
            return Err(PyValueError::new_err("expected a complex128 statevector"));
        };
        self.inner.fidelity(psi).map_err(to_py_err)
    }

    /// Counts of `shots` measurements of every qubit, as bitstrings with
    /// qubit 0 rightmost.
    fn sample_counts<'py>(
        &mut self,
        py: Python<'py>,
        shots: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let n = self.inner.n_qubits();
        let mut counts = std::collections::BTreeMap::new();
        for index in self.inner.sample(shots, &mut self.rng) {
            *counts.entry(index).or_insert(0usize) += 1;
        }
        let dict = PyDict::new(py);
        for (index, count) in counts {
            dict.set_item(format!("{index:0n$b}"), count)?;
        }
        Ok(dict)
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyChannel>()?;
    m.add_class::<PyNoiseModel>()?;
    m.add_class::<PyDensityMatrix>()
}
