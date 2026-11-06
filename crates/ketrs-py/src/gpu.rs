//! Python bindings for GPU statevectors.

use super::{Observable, PyCircuit, PyStatevector, State, rng_from_seed, to_py_err};
use ketrs::{Complex64 as C64, GpuContext, GpuStatevector};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::rngs::StdRng;
use std::sync::{Arc, OnceLock};

/// One GPU context per process, created on first use.
fn context() -> PyResult<Arc<GpuContext>> {
    static CONTEXT: OnceLock<Result<Arc<GpuContext>, String>> = OnceLock::new();
    CONTEXT
        .get_or_init(|| GpuContext::new().map_err(|e| e.to_string()))
        .clone()
        .map_err(PyRuntimeError::new_err)
}

/// The name of the GPU ketrs would use, or None if there is no Vulkan GPU
/// with 64-bit float support.
#[pyfunction]
fn gpu_device() -> Option<String> {
    context().ok().map(|ctx| ctx.device_name().to_owned())
}

/// A statevector in GPU memory (complex128), simulated with Vulkan compute
/// on any NVIDIA, AMD or other GPU with 64-bit floats. Needs only the
/// system's Vulkan driver.
#[pyclass(name = "GpuStatevector", module = "ketrs", unsendable)]
struct PyGpuStatevector {
    inner: GpuStatevector,
    rng: StdRng,
}

#[pymethods]
impl PyGpuStatevector {
    /// `|0…0⟩` on `n_qubits` qubits. States over one GPU buffer (usually
    /// 27 qubits) are split across several, up to 30 qubits.
    #[new]
    #[pyo3(signature = (n_qubits, seed=None))]
    fn new(n_qubits: usize, seed: Option<u64>) -> PyResult<Self> {
        Ok(PyGpuStatevector {
            inner: GpuStatevector::zero(&context()?, n_qubits).map_err(to_py_err)?,
            rng: rng_from_seed(seed),
        })
    }

    /// Run `circuit` on `|0…0⟩`.
    #[staticmethod]
    #[pyo3(signature = (circuit, seed=None))]
    fn from_circuit(
        py: Python<'_>,
        circuit: PyRef<'_, PyCircuit>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let ctx = context()?;
        let circ = &circuit.inner;
        let inner = py
            .detach(|| GpuStatevector::from_circuit(&ctx, circ))
            .map_err(to_py_err)?;
        Ok(PyGpuStatevector {
            inner,
            rng: rng_from_seed(seed),
        })
    }

    /// Upload a complex128 array whose length is a power of two.
    #[staticmethod]
    #[pyo3(signature = (amplitudes, seed=None))]
    fn from_amplitudes(amplitudes: PyReadonlyArray1<'_, C64>, seed: Option<u64>) -> PyResult<Self> {
        let amps = amplitudes.as_slice()?;
        Ok(PyGpuStatevector {
            inner: GpuStatevector::from_amplitudes(&context()?, amps).map_err(to_py_err)?,
            rng: rng_from_seed(seed),
        })
    }

    #[getter]
    fn n_qubits(&self) -> usize {
        self.inner.n_qubits()
    }

    /// The GPU this state lives on.
    #[getter]
    fn device(&self) -> String {
        self.inner.context().device_name().to_owned()
    }

    fn __repr__(&self) -> String {
        format!(
            "GpuStatevector(n_qubits={}, device={:?})",
            self.inner.n_qubits(),
            self.inner.context().device_name()
        )
    }

    /// Apply `circuit`, which must have the same width.
    fn run(&mut self, py: Python<'_>, circuit: PyRef<'_, PyCircuit>) -> PyResult<()> {
        let (state, circ) = (&mut self.inner, &circuit.inner);
        py.detach(|| state.run(circ)).map_err(to_py_err)
    }

    /// Download the amplitudes as a complex128 array.
    fn amplitudes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<C64>>> {
        let amps = py.detach(|| self.inner.amplitudes()).map_err(to_py_err)?;
        Ok(PyArray1::from_vec(py, amps))
    }

    /// Download into a CPU `Statevector`.
    #[pyo3(signature = (seed=None))]
    fn to_statevector(&self, py: Python<'_>, seed: Option<u64>) -> PyResult<PyStatevector> {
        let state = py
            .detach(|| self.inner.to_statevector())
            .map_err(to_py_err)?;
        Ok(PyStatevector::wrap(
            State::Complex(state),
            rng_from_seed(seed),
        ))
    }

    /// `⟨ψ|H|ψ⟩`, computed on the GPU (only partial sums are downloaded).
    fn expectation(&mut self, py: Python<'_>, observable: Observable) -> PyResult<f64> {
        let sum = observable.into_sum()?;
        let state = &mut self.inner;
        py.detach(|| state.expectation(&sum)).map_err(to_py_err)
    }

    /// `⟨ψ|ψ⟩`, computed on the GPU.
    fn norm_sqr(&mut self, py: Python<'_>) -> PyResult<f64> {
        let state = &mut self.inner;
        py.detach(|| state.norm_sqr()).map_err(to_py_err)
    }

    /// Draw `shots` basis-state indices on the GPU, without changing the
    /// state; only the sampled blocks of amplitudes are downloaded.
    fn sample<'py>(
        &mut self,
        py: Python<'py>,
        shots: usize,
    ) -> PyResult<Bound<'py, PyArray1<usize>>> {
        let (state, rng) = (&mut self.inner, &mut self.rng);
        let samples = py.detach(|| state.sample(shots, rng)).map_err(to_py_err)?;
        Ok(PyArray1::from_vec(py, samples))
    }

    /// Counts keyed by bitstring, most significant qubit first.
    fn sample_counts<'py>(
        &mut self,
        py: Python<'py>,
        shots: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let n = self.inner.n_qubits();
        let (state, rng) = (&mut self.inner, &mut self.rng);
        let counts = py
            .detach(|| state.sample_counts(shots, rng))
            .map_err(to_py_err)?;
        let dict = PyDict::new(py);
        for (index, count) in counts {
            dict.set_item(format!("{index:0n$b}"), count)?;
        }
        Ok(dict)
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGpuStatevector>()?;
    m.add_function(wrap_pyfunction!(gpu_device, m)?)
}
