//! Python bindings for GPU statevectors.

use super::{Observable, PyCircuit, PyStatevector, State, clbit_counts, rng_from_seed, to_py_err};
use ketrs::{Circuit, Complex64 as C64, GpuContext, GpuPrecision, GpuStatevector};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
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

/// The precision for `dtype`: "complex128", "complex64", or None for the
/// GPU's best.
fn precision(ctx: &GpuContext, dtype: Option<&str>) -> PyResult<GpuPrecision> {
    match dtype {
        None if ctx.supports_f64() => Ok(GpuPrecision::Double),
        None => Ok(GpuPrecision::Single),
        Some("complex" | "complex128") => Ok(GpuPrecision::Double),
        Some("complex64") => Ok(GpuPrecision::Single),
        Some(other) => Err(PyValueError::new_err(format!(
            "GPU dtype must be 'complex128' or 'complex64', got {other:?}"
        ))),
    }
}

/// The name of the GPU ketrs would use, or None if there is no Vulkan GPU.
#[pyfunction]
fn gpu_device() -> Option<String> {
    context().ok().map(|ctx| ctx.device_name().to_owned())
}

/// A statevector in GPU memory, simulated with Vulkan compute on any NVIDIA,
/// AMD or other GPU. Needs only the system's Vulkan driver.
///
/// `dtype="complex128"` needs a GPU with 64-bit floats and is the default
/// where available; `"complex64"` halves the memory, works on any GPU and is
/// accurate to about 1e-6.
#[pyclass(name = "GpuStatevector", module = "ketrs", unsendable)]
struct PyGpuStatevector {
    inner: GpuStatevector,
    rng: StdRng,
}

#[pymethods]
impl PyGpuStatevector {
    /// `|0…0⟩` on `n_qubits` qubits. States over one GPU buffer (usually
    /// 27 qubits) are split across several, up to 31 qubits.
    #[new]
    #[pyo3(signature = (n_qubits, seed=None, dtype=None))]
    fn new(n_qubits: usize, seed: Option<u64>, dtype: Option<&str>) -> PyResult<Self> {
        let ctx = context()?;
        let precision = precision(&ctx, dtype)?;
        Ok(PyGpuStatevector {
            inner: GpuStatevector::zero_with_precision(&ctx, n_qubits, precision)
                .map_err(to_py_err)?,
            rng: rng_from_seed(seed),
        })
    }

    /// Run `circuit` on `|0…0⟩`.
    #[staticmethod]
    #[pyo3(signature = (circuit, seed=None, dtype=None))]
    fn from_circuit(
        py: Python<'_>,
        circuit: PyRef<'_, PyCircuit>,
        seed: Option<u64>,
        dtype: Option<&str>,
    ) -> PyResult<Self> {
        let ctx = context()?;
        let precision = precision(&ctx, dtype)?;
        let circ: &Circuit = &circuit.inner;
        let inner = py
            .detach(|| {
                let mut state =
                    GpuStatevector::zero_with_precision(&ctx, circ.n_qubits(), precision)?;
                state.run(circ)?;
                Ok::<_, ketrs::Error>(state)
            })
            .map_err(to_py_err)?;
        Ok(PyGpuStatevector {
            inner,
            rng: rng_from_seed(seed),
        })
    }

    /// Upload a complex128 array whose length is a power of two.
    #[staticmethod]
    #[pyo3(signature = (amplitudes, seed=None, dtype=None))]
    fn from_amplitudes(
        amplitudes: PyReadonlyArray1<'_, C64>,
        seed: Option<u64>,
        dtype: Option<&str>,
    ) -> PyResult<Self> {
        let amps = amplitudes.as_slice()?;
        let ctx = context()?;
        let precision = precision(&ctx, dtype)?;
        Ok(PyGpuStatevector {
            inner: GpuStatevector::from_amplitudes_with_precision(&ctx, amps, precision)
                .map_err(to_py_err)?,
            rng: rng_from_seed(seed),
        })
    }

    /// `"complex128"` or `"complex64"`.
    #[getter]
    fn dtype(&self) -> &'static str {
        match self.inner.precision() {
            GpuPrecision::Double => "complex128",
            GpuPrecision::Single => "complex64",
        }
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
            "GpuStatevector(n_qubits={}, dtype={}, device={:?})",
            self.inner.n_qubits(),
            self.dtype(),
            self.inner.context().device_name()
        )
    }

    /// Apply `circuit`, which must have the same width. A circuit with
    /// measurements, resets or classical control follows one random history
    /// (using this state's generator) and returns its classical bits;
    /// otherwise this returns None.
    fn run(
        &mut self,
        py: Python<'_>,
        circuit: PyRef<'_, PyCircuit>,
    ) -> PyResult<Option<Vec<usize>>> {
        let (state, circ) = (&mut self.inner, &circuit.inner);
        if circ.is_unitary() {
            py.detach(|| state.run(circ)).map_err(to_py_err)?;
            return Ok(None);
        }
        let rng = &mut self.rng;
        let bits = py
            .detach(|| state.run_trajectory(circ, rng))
            .map_err(to_py_err)?;
        Ok(Some(bits.into_iter().map(usize::from).collect()))
    }

    /// Run `circuit` from this state `shots` times and count the final
    /// classical bits, as bitstrings with bit 0 rightmost (as in Qiskit's
    /// counts). The state itself is unchanged.
    fn sample_circuit<'py>(
        &mut self,
        py: Python<'py>,
        circuit: PyRef<'_, PyCircuit>,
        shots: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let (state, circ) = (&self.inner, &circuit.inner);
        let rng = &mut self.rng;
        let counts = py
            .detach(|| state.sample_circuit(circ, shots, rng))
            .map_err(to_py_err)?;
        clbit_counts(py, counts, circ.n_clbits())
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

    /// Evolve by exp(-iHt) exactly (Krylov time-stepping) on the GPU,
    /// needing room for three more statevectors.
    fn evolve(&mut self, py: Python<'_>, hamiltonian: Observable, time: f64) -> PyResult<()> {
        let sum = hamiltonian.into_sum()?;
        let state = &mut self.inner;
        py.detach(|| state.evolve(&sum, time)).map_err(to_py_err)
    }

    /// `H|ψ⟩` as a new GPU state, computed on the GPU.
    fn apply_pauli_sum(&mut self, py: Python<'_>, hamiltonian: Observable) -> PyResult<Self> {
        let sum = hamiltonian.into_sum()?;
        let state = &mut self.inner;
        let applied = py
            .detach(|| state.apply_pauli_sum(&sum))
            .map_err(to_py_err)?;
        Ok(PyGpuStatevector {
            inner: applied,
            rng: self.rng.clone(),
        })
    }

    /// `⟨ψ(θ)|H|ψ(θ)⟩` and its gradient by adjoint differentiation on the
    /// GPU, where `ψ(θ)` is `circuit` bound to `parameters` applied to this
    /// state (which is unchanged). Returns `(value, gradient)`.
    fn expectation_gradient<'py>(
        &self,
        py: Python<'py>,
        circuit: PyRef<'_, PyCircuit>,
        parameters: Vec<f64>,
        observable: Observable,
    ) -> PyResult<(f64, Bound<'py, PyArray1<f64>>)> {
        let sum = observable.into_sum()?;
        let (state, circ) = (&self.inner, &circuit.inner);
        let (value, gradient) = py
            .detach(|| state.expectation_gradient(circ, &parameters, &sum))
            .map_err(to_py_err)?;
        Ok((value, PyArray1::from_vec(py, gradient)))
    }

    /// Gradient of a real loss of the output of `circuit` on the GPU; see
    /// `Statevector.adjoint_gradient`.
    #[staticmethod]
    fn adjoint_gradient<'py>(
        py: Python<'py>,
        circuit: PyRef<'_, PyCircuit>,
        parameters: Vec<f64>,
        final_state: PyRef<'_, PyGpuStatevector>,
        cotangent: PyRef<'_, PyGpuStatevector>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let (circ, f, c) = (&circuit.inner, &final_state.inner, &cotangent.inner);
        let gradient = py
            .detach(|| GpuStatevector::adjoint_gradient(circ, &parameters, f, c))
            .map_err(to_py_err)?;
        Ok(PyArray1::from_vec(py, gradient))
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
