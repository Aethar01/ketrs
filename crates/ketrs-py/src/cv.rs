//! Python bindings for continuous-variable (qumode) circuits, Gaussian states,
//! and hafnian / torontonian kernels.

use super::{PyParameter, rng_from_seed, to_py_err};
use ketrs::Complex64 as C64;
use ketrs::cv::{
    Arg, BosonicMps, BosonicMpsOptions, CvBackend, CvChannel, CvPlan, FixedNFockState,
    FockDensityMatrix, FockState, GaussianMixture, GaussianState, HybridCircuit, HybridState,
    Measured, Measurement, MixtureOptions, ModeCircuit, ModeGate, ModeOp, ModeState, PlanOptions,
    TwoModeGate, WindowSampler, boson_sample as rust_boson_sample, hafnian as rust_hafnian,
    hafnian_repeated as rust_hafnian_repeated, loop_hafnian as rust_loop_hafnian,
    loop_hafnian_repeated as rust_loop_hafnian_repeated, permanent as rust_permanent,
    plan_circuit as rust_plan_circuit, sample_gbs_pnr as rust_sample_gbs_pnr,
    sample_gbs_threshold as rust_sample_gbs_threshold, torontonian as rust_torontonian,
};
#[cfg(feature = "gpu")]
use ketrs::cv::{GpuFockState, permanent_gpu as rust_permanent_gpu};
use numpy::{
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::rngs::StdRng;

/// A measurement result slot, or a linear form `scale·m + offset` of one,
/// returned by `ModeCircuit.measure_*` for feed-forward gate arguments.
#[pyclass(name = "Measured", module = "ketrs", frozen, from_py_object)]
#[derive(Clone, Copy)]
pub(crate) struct PyMeasured {
    result: Measured,
    scale: f64,
    offset: f64,
}

impl PyMeasured {
    fn from_slot(result: Measured) -> Self {
        PyMeasured {
            result,
            scale: 1.0,
            offset: 0.0,
        }
    }

    fn arg(&self) -> Arg {
        Arg::Result {
            result: self.result,
            scale: self.scale,
            offset: self.offset,
        }
    }

    fn with(&self, scale: f64, offset: f64) -> Self {
        PyMeasured {
            result: self.result,
            scale,
            offset,
        }
    }
}

#[pymethods]
impl PyMeasured {
    /// Position of this measurement in a shot's result vector.
    #[getter]
    fn index(&self) -> usize {
        self.result.index()
    }
    #[getter]
    fn scale(&self) -> f64 {
        self.scale
    }
    #[getter]
    fn offset(&self) -> f64 {
        self.offset
    }
    fn __mul__(&self, k: f64) -> Self {
        self.with(self.scale * k, self.offset * k)
    }
    fn __rmul__(&self, k: f64) -> Self {
        self.__mul__(k)
    }
    fn __truediv__(&self, k: f64) -> Self {
        self.__mul__(1.0 / k)
    }
    fn __add__(&self, k: f64) -> Self {
        self.with(self.scale, self.offset + k)
    }
    fn __radd__(&self, k: f64) -> Self {
        self.__add__(k)
    }
    fn __sub__(&self, k: f64) -> Self {
        self.__add__(-k)
    }
    fn __rsub__(&self, k: f64) -> Self {
        self.with(-self.scale, k - self.offset)
    }
    fn __neg__(&self) -> Self {
        self.__mul__(-1.0)
    }
    fn __repr__(&self) -> String {
        format!("{}·m{} + {}", self.scale, self.result.index(), self.offset)
    }
}

/// A real CV gate argument: a float, a `Parameter`, or a `Measured` slot.
#[derive(FromPyObject)]
enum CvArg {
    Value(f64),
    Parameter(PyParameter),
    Measured(PyMeasured),
}

impl CvArg {
    fn to_arg(&self) -> Arg {
        match self {
            CvArg::Value(v) => Arg::Value(*v),
            CvArg::Parameter(p) => Arg::from(p.angle()),
            CvArg::Measured(m) => m.arg(),
        }
    }
}

/// A 2D real or complex array converted to a row-major `Vec<C64>`.
#[derive(FromPyObject)]
enum Matrix2C<'py> {
    Complex(PyReadonlyArray2<'py, C64>),
    Real(PyReadonlyArray2<'py, f64>),
    Nested(Vec<Vec<C64>>),
}

impl Matrix2C<'_> {
    fn to_square_vec(&self) -> PyResult<(Vec<C64>, usize)> {
        match self {
            Matrix2C::Complex(arr) => {
                let shape = arr.shape();
                if shape[0] != shape[1] {
                    return Err(PyValueError::new_err(format!(
                        "expected a square matrix, got {}×{}",
                        shape[0], shape[1]
                    )));
                }
                let a = arr.as_array();
                let mut out = Vec::with_capacity(shape[0] * shape[1]);
                for row in a.rows() {
                    out.extend(row.iter().copied());
                }
                Ok((out, shape[0]))
            }
            Matrix2C::Real(arr) => {
                let shape = arr.shape();
                if shape[0] != shape[1] {
                    return Err(PyValueError::new_err(format!(
                        "expected a square matrix, got {}×{}",
                        shape[0], shape[1]
                    )));
                }
                let a = arr.as_array();
                let mut out = Vec::with_capacity(shape[0] * shape[1]);
                for row in a.rows() {
                    out.extend(row.iter().map(|&x| C64::new(x, 0.0)));
                }
                Ok((out, shape[0]))
            }
            Matrix2C::Nested(rows) => {
                let n = rows.len();
                let mut out = Vec::with_capacity(n * n);
                for row in rows {
                    if row.len() != n {
                        return Err(PyValueError::new_err("matrix rows must have equal length"));
                    }
                    out.extend_from_slice(row);
                }
                Ok((out, n))
            }
        }
    }
}

/// A 1D real or complex vector converted to `Vec<C64>`.
#[derive(FromPyObject)]
enum Vec1C<'py> {
    Complex(PyReadonlyArray1<'py, C64>),
    Real(PyReadonlyArray1<'py, f64>),
    List(Vec<C64>),
}

impl Vec1C<'_> {
    fn to_vec(&self) -> Vec<C64> {
        match self {
            Vec1C::Complex(a) => a.as_array().to_vec(),
            Vec1C::Real(a) => a.as_array().iter().map(|&x| C64::new(x, 0.0)).collect(),
            Vec1C::List(v) => v.clone(),
        }
    }
}

/// A 2D float64 array converted to a row-major `Vec<f64>`.
#[derive(FromPyObject)]
enum Matrix2F<'py> {
    Array(PyReadonlyArray2<'py, f64>),
    Nested(Vec<Vec<f64>>),
}

impl Matrix2F<'_> {
    fn to_square_vec(&self) -> PyResult<(Vec<f64>, usize)> {
        match self {
            Matrix2F::Array(arr) => {
                let shape = arr.shape();
                if shape[0] != shape[1] {
                    return Err(PyValueError::new_err(format!(
                        "expected a square matrix, got {}×{}",
                        shape[0], shape[1]
                    )));
                }
                let a = arr.as_array();
                let mut out = Vec::with_capacity(shape[0] * shape[1]);
                for row in a.rows() {
                    out.extend(row.iter().copied());
                }
                Ok((out, shape[0]))
            }
            Matrix2F::Nested(rows) => {
                let n = rows.len();
                let mut out = Vec::with_capacity(n * n);
                for row in rows {
                    if row.len() != n {
                        return Err(PyValueError::new_err("matrix rows must have equal length"));
                    }
                    out.extend_from_slice(row);
                }
                Ok((out, n))
            }
        }
    }
}

/// A single-mode input state for [`ModeCircuit.set_input`].
#[pyclass(name = "ModeState", module = "ketrs", frozen, from_py_object)]
#[derive(Clone)]
pub(crate) struct PyModeState {
    inner: ModeState,
}

#[pymethods]
impl PyModeState {
    /// Vacuum state `|0⟩`.
    #[staticmethod]
    fn vacuum() -> Self {
        PyModeState {
            inner: ModeState::Vacuum,
        }
    }

    /// Fock state `|n⟩`.
    #[staticmethod]
    fn fock(n: usize) -> Self {
        PyModeState {
            inner: ModeState::Fock(n),
        }
    }

    /// Coherent state `|α⟩ = D(α)|0⟩` with `α = r e^{iφ}`.
    #[staticmethod]
    #[pyo3(signature = (r, phi=0.0))]
    fn coherent(r: f64, phi: f64) -> Self {
        PyModeState {
            inner: ModeState::Coherent { r, phi },
        }
    }

    /// Squeezed vacuum `S(r e^{iφ})|0⟩`.
    #[staticmethod]
    #[pyo3(signature = (r, phi=0.0))]
    fn squeezed(r: f64, phi: f64) -> Self {
        PyModeState {
            inner: ModeState::Squeezed { r, phi },
        }
    }

    /// Thermal state with mean photon number `mean_photons ≥ 0`.
    #[staticmethod]
    fn thermal(mean_photons: f64) -> Self {
        PyModeState {
            inner: ModeState::Thermal { mean_photons },
        }
    }

    /// Cat state `∝ |α⟩ + e^{iπp}|−α⟩` with `α = r e^{iφ}` and `parity = p`.
    #[staticmethod]
    #[pyo3(signature = (r, phi=0.0, parity=0.0))]
    fn cat(r: f64, phi: f64, parity: f64) -> Self {
        PyModeState {
            inner: ModeState::Cat { r, phi, parity },
        }
    }

    /// Finite-energy square GKP qubit `cos(θ/2)|0_ε⟩ + e^{iφ} sin(θ/2)|1_ε⟩`
    /// (`ħ = 2`, grid spacing `√(2π)`, squeezing `-10 log10(tanh ε)` dB).
    #[staticmethod]
    #[pyo3(signature = (epsilon, theta=0.0, phi=0.0))]
    fn gkp(epsilon: f64, theta: f64, phi: f64) -> Self {
        PyModeState {
            inner: ModeState::Gkp {
                epsilon,
                theta,
                phi,
            },
        }
    }

    /// Arbitrary normalised superposition of Fock states `Σ c_n |n⟩`.
    #[staticmethod]
    fn fock_vector(amplitudes: Vec1C<'_>) -> Self {
        PyModeState {
            inner: ModeState::FockVector(amplitudes.to_vec()),
        }
    }

    #[getter]
    fn is_gaussian(&self) -> bool {
        self.inner.is_gaussian()
    }

    #[getter]
    fn is_pure(&self) -> bool {
        self.inner.is_pure()
    }

    #[getter]
    fn is_fock(&self) -> bool {
        self.inner.is_fock()
    }

    fn __repr__(&self) -> String {
        format!("ModeState({:?})", self.inner)
    }
}

/// A continuous-variable (qumode) circuit with Strawberry Fields conventions
/// (`ħ = 2`, `x̂ = a + a†`, `p̂ = −i(a − a†)`).
#[pyclass(name = "ModeCircuit", module = "ketrs", from_py_object)]
#[derive(Clone)]
pub(crate) struct PyModeCircuit {
    inner: ModeCircuit,
}

impl PyModeCircuit {
    fn push_op(mut slf: PyRefMut<'_, Self>, op: ModeOp) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.push(op).map_err(to_py_err)?;
        Ok(slf)
    }
}

#[pymethods]
impl PyModeCircuit {
    /// A circuit on `n_modes` modes, all starting in the vacuum.
    #[new]
    fn new(n_modes: usize) -> Self {
        PyModeCircuit {
            inner: ModeCircuit::new(n_modes),
        }
    }

    #[getter]
    fn n_modes(&self) -> usize {
        self.inner.n_modes()
    }

    #[getter]
    fn n_parameters(&self) -> usize {
        self.inner.n_parameters()
    }

    #[getter]
    fn n_results(&self) -> usize {
        self.inner.n_results()
    }

    #[getter]
    fn is_gaussian(&self) -> bool {
        self.inner.is_gaussian()
    }

    #[getter]
    fn is_passive(&self) -> bool {
        self.inner.is_passive()
    }

    #[getter]
    fn conserves_photons(&self) -> bool {
        self.inner.conserves_photons()
    }

    #[getter]
    fn is_unitary(&self) -> bool {
        self.inner.is_unitary()
    }

    /// Create a new circuit parameter.
    fn parameter(&mut self) -> PyParameter {
        PyParameter {
            parameter: self.inner.parameter(),
            scale: 1.0,
            offset: 0.0,
        }
    }

    /// Create `n` new circuit parameters.
    fn parameters(&mut self, n: usize) -> Vec<PyParameter> {
        self.inner
            .parameters(n)
            .into_iter()
            .map(|parameter| PyParameter {
                parameter,
                scale: 1.0,
                offset: 0.0,
            })
            .collect()
    }

    /// Return a new circuit with all parameters replaced by `values`.
    fn bind(&self, values: Vec<f64>) -> PyResult<Self> {
        Ok(PyModeCircuit {
            inner: self.inner.bind(&values).map_err(to_py_err)?,
        })
    }

    /// Set `mode`'s initial state.
    fn set_input<'py>(
        mut slf: PyRefMut<'py, Self>,
        mode: usize,
        state: PyModeState,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.inner.set_input(mode, state.inner).map_err(to_py_err)?;
        Ok(slf)
    }

    /// Rotation `R(φ)`.
    fn rotation<'py>(
        slf: PyRefMut<'py, Self>,
        phi: CvArg,
        mode: usize,
    ) -> PyResult<PyRefMut<'py, Self>> {
        Self::push_op(
            slf,
            ModeOp::Gate {
                gate: ModeGate::Rotation { phi: phi.to_arg() },
                mode,
            },
        )
    }

    /// Displacement `D(r e^{iφ})`.
    fn displace<'py>(
        slf: PyRefMut<'py, Self>,
        r: CvArg,
        phi: CvArg,
        mode: usize,
    ) -> PyResult<PyRefMut<'py, Self>> {
        Self::push_op(
            slf,
            ModeOp::Gate {
                gate: ModeGate::Displacement {
                    r: r.to_arg(),
                    phi: phi.to_arg(),
                },
                mode,
            },
        )
    }

    /// Squeezing `S(r e^{iφ})`.
    fn squeeze<'py>(
        slf: PyRefMut<'py, Self>,
        r: CvArg,
        phi: CvArg,
        mode: usize,
    ) -> PyResult<PyRefMut<'py, Self>> {
        Self::push_op(
            slf,
            ModeOp::Gate {
                gate: ModeGate::Squeezing {
                    r: r.to_arg(),
                    phi: phi.to_arg(),
                },
                mode,
            },
        )
    }

    /// Quadratic phase `P(s)`.
    fn quadratic_phase<'py>(
        slf: PyRefMut<'py, Self>,
        s: CvArg,
        mode: usize,
    ) -> PyResult<PyRefMut<'py, Self>> {
        Self::push_op(
            slf,
            ModeOp::Gate {
                gate: ModeGate::QuadraticPhase { s: s.to_arg() },
                mode,
            },
        )
    }

    /// Cubic phase `V(γ)` (non-Gaussian).
    fn cubic_phase<'py>(
        slf: PyRefMut<'py, Self>,
        gamma: CvArg,
        mode: usize,
    ) -> PyResult<PyRefMut<'py, Self>> {
        Self::push_op(
            slf,
            ModeOp::Gate {
                gate: ModeGate::CubicPhase {
                    gamma: gamma.to_arg(),
                },
                mode,
            },
        )
    }

    /// Kerr `K(κ)` (non-Gaussian).
    fn kerr<'py>(
        slf: PyRefMut<'py, Self>,
        kappa: CvArg,
        mode: usize,
    ) -> PyResult<PyRefMut<'py, Self>> {
        Self::push_op(
            slf,
            ModeOp::Gate {
                gate: ModeGate::Kerr {
                    kappa: kappa.to_arg(),
                },
                mode,
            },
        )
    }

    /// Beam splitter `BS(θ, φ)` on `(a, b)`.
    fn beam_splitter<'py>(
        slf: PyRefMut<'py, Self>,
        theta: CvArg,
        phi: CvArg,
        a: usize,
        b: usize,
    ) -> PyResult<PyRefMut<'py, Self>> {
        Self::push_op(
            slf,
            ModeOp::TwoModeGate {
                gate: TwoModeGate::BeamSplitter {
                    theta: theta.to_arg(),
                    phi: phi.to_arg(),
                },
                modes: [a, b],
            },
        )
    }

    /// Two-mode squeezing `S₂(r e^{iφ})` on `(a, b)`.
    fn two_mode_squeeze<'py>(
        slf: PyRefMut<'py, Self>,
        r: CvArg,
        phi: CvArg,
        a: usize,
        b: usize,
    ) -> PyResult<PyRefMut<'py, Self>> {
        Self::push_op(
            slf,
            ModeOp::TwoModeGate {
                gate: TwoModeGate::TwoModeSqueezing {
                    r: r.to_arg(),
                    phi: phi.to_arg(),
                },
                modes: [a, b],
            },
        )
    }

    /// `CX(s)` with control `a` and target `b`.
    fn cx<'py>(
        slf: PyRefMut<'py, Self>,
        s: CvArg,
        a: usize,
        b: usize,
    ) -> PyResult<PyRefMut<'py, Self>> {
        Self::push_op(
            slf,
            ModeOp::TwoModeGate {
                gate: TwoModeGate::ControlledX { s: s.to_arg() },
                modes: [a, b],
            },
        )
    }

    /// `CZ(s)` on `(a, b)`.
    fn cz<'py>(
        slf: PyRefMut<'py, Self>,
        s: CvArg,
        a: usize,
        b: usize,
    ) -> PyResult<PyRefMut<'py, Self>> {
        Self::push_op(
            slf,
            ModeOp::TwoModeGate {
                gate: TwoModeGate::ControlledZ { s: s.to_arg() },
                modes: [a, b],
            },
        )
    }

    /// Cross-Kerr `CK(κ)` on `(a, b)` (non-Gaussian).
    fn cross_kerr<'py>(
        slf: PyRefMut<'py, Self>,
        kappa: CvArg,
        a: usize,
        b: usize,
    ) -> PyResult<PyRefMut<'py, Self>> {
        Self::push_op(
            slf,
            ModeOp::TwoModeGate {
                gate: TwoModeGate::CrossKerr {
                    kappa: kappa.to_arg(),
                },
                modes: [a, b],
            },
        )
    }

    /// Linear-optics unitary block `U` on `modes` (`α → U α`).
    fn passive<'py>(
        mut slf: PyRefMut<'py, Self>,
        modes: Vec<usize>,
        matrix: Matrix2C<'_>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let (mat, k) = matrix.to_square_vec()?;
        if k != modes.len() {
            return Err(PyValueError::new_err(format!(
                "passive block on {} modes got a {k}×{k} matrix",
                modes.len()
            )));
        }
        slf.inner.passive(&modes, mat).map_err(to_py_err)?;
        Ok(slf)
    }

    /// Apply a single-mode CPTP noise channel (`CvChannel`) on `mode`.
    fn apply_channel<'py>(
        mut slf: PyRefMut<'py, Self>,
        channel: PyCvChannel,
        mode: usize,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.inner
            .apply_channel(channel.inner, mode)
            .map_err(to_py_err)?;
        Ok(slf)
    }

    /// Destructive homodyne measurement of `x̂ cos φ + p̂ sin φ` on `mode`.
    fn measure_homodyne(&mut self, phi: CvArg, mode: usize) -> PyResult<PyMeasured> {
        let m = self
            .inner
            .try_measure(Measurement::Homodyne { phi: phi.to_arg() }, mode)
            .map_err(to_py_err)?;
        Ok(PyMeasured::from_slot(m))
    }

    /// Destructive measurement of `x̂` on `mode`.
    fn measure_x(&mut self, mode: usize) -> PyResult<PyMeasured> {
        self.measure_homodyne(CvArg::Value(0.0), mode)
    }

    /// Destructive measurement of `p̂` on `mode`.
    fn measure_p(&mut self, mode: usize) -> PyResult<PyMeasured> {
        self.measure_homodyne(CvArg::Value(std::f64::consts::FRAC_PI_2), mode)
    }

    /// Destructive heterodyne measurement on `mode`, returning `(re, im)` slots.
    fn measure_heterodyne(&mut self, mode: usize) -> PyResult<(PyMeasured, PyMeasured)> {
        if mode >= self.inner.n_modes() {
            return Err(PyValueError::new_err(format!(
                "mode {mode} out of range for {} modes",
                self.inner.n_modes()
            )));
        }
        let (re, im) = self.inner.measure_heterodyne(mode);
        Ok((PyMeasured::from_slot(re), PyMeasured::from_slot(im)))
    }

    /// Destructive photon-number measurement on `mode`.
    fn measure_photons(&mut self, mode: usize) -> PyResult<PyMeasured> {
        let m = self
            .inner
            .try_measure(Measurement::PhotonCount, mode)
            .map_err(to_py_err)?;
        Ok(PyMeasured::from_slot(m))
    }

    /// Destructive threshold measurement on `mode` (0 or 1).
    fn measure_threshold(&mut self, mode: usize) -> PyResult<PyMeasured> {
        let m = self
            .inner
            .try_measure(Measurement::Threshold, mode)
            .map_err(to_py_err)?;
        Ok(PyMeasured::from_slot(m))
    }

    fn __repr__(&self) -> String {
        format!(
            "ModeCircuit(n_modes={}, ops={}, n_results={})",
            self.inner.n_modes(),
            self.inner.ops().len(),
            self.inner.n_results()
        )
    }
}

/// Exact Gaussian state simulator in the quadrature covariance representation
/// (`xxpp` ordering, `ħ = 2`).
#[pyclass(name = "GaussianState", module = "ketrs")]
pub(crate) struct PyGaussianState {
    inner: GaussianState,
    rng: StdRng,
}

#[pymethods]
impl PyGaussianState {
    /// Vacuum state `|0…0⟩` on `n_modes` modes.
    #[new]
    #[pyo3(signature = (n_modes, seed=None))]
    fn new(n_modes: usize, seed: Option<u64>) -> Self {
        PyGaussianState {
            inner: GaussianState::vacuum(n_modes),
            rng: rng_from_seed(seed),
        }
    }

    /// Construct from `xxpp` quadrature `means` (length `2m`) and symmetric
    /// covariance matrix `cov` (`2m × 2m`, `ħ = 2`).
    #[staticmethod]
    #[pyo3(signature = (means, cov, seed=None))]
    fn from_moments(means: Vec<f64>, cov: Matrix2F<'_>, seed: Option<u64>) -> PyResult<Self> {
        let (cov_vec, _) = cov.to_square_vec()?;
        Ok(PyGaussianState {
            inner: GaussianState::from_moments(means, cov_vec).map_err(to_py_err)?,
            rng: rng_from_seed(seed),
        })
    }

    /// Run a bound, measurement-free Gaussian `ModeCircuit`.
    #[staticmethod]
    #[pyo3(signature = (circuit, seed=None))]
    fn from_circuit(
        py: Python<'_>,
        circuit: PyRef<'_, PyModeCircuit>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let circ = &circuit.inner;
        let inner = py
            .detach(|| GaussianState::from_circuit(circ))
            .map_err(to_py_err)?;
        Ok(PyGaussianState {
            inner,
            rng: rng_from_seed(seed),
        })
    }

    /// Run one shot of a bound Gaussian `ModeCircuit` (with homodyne/heterodyne
    /// measurements and feed-forward), returning `(state, results)`.
    #[staticmethod]
    #[pyo3(signature = (circuit, seed=None))]
    fn run<'py>(
        py: Python<'py>,
        circuit: PyRef<'_, PyModeCircuit>,
        seed: Option<u64>,
    ) -> PyResult<(Self, Bound<'py, PyArray1<f64>>)> {
        let circ = &circuit.inner;
        let mut rng = rng_from_seed(seed);
        let (inner, results) = py
            .detach(|| GaussianState::run(circ, &mut rng))
            .map_err(to_py_err)?;
        Ok((
            PyGaussianState { inner, rng },
            PyArray1::from_vec(py, results),
        ))
    }

    /// Sample `shots` independent measurement trajectories from a bound
    /// Gaussian `ModeCircuit`, returning an array of shape `(shots, n_results)`.
    #[staticmethod]
    #[pyo3(signature = (circuit, shots, seed=None))]
    fn sample_circuit<'py>(
        py: Python<'py>,
        circuit: PyRef<'_, PyModeCircuit>,
        shots: usize,
        seed: Option<u64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let circ = &circuit.inner;
        let n_res = circ.n_results();
        let mut rng = rng_from_seed(seed);
        let rows = py
            .detach(|| GaussianState::sample(circ, shots, &mut rng))
            .map_err(to_py_err)?;
        let mut flat = Vec::with_capacity(shots * n_res);
        for row in rows {
            flat.extend_from_slice(&row);
        }
        PyArray1::from_vec(py, flat).reshape([shots, n_res])
    }

    #[getter]
    fn n_modes(&self) -> usize {
        self.inner.n_modes()
    }

    /// Quadrature mean vector `(x_0, …, x_{m−1}, p_0, …, p_{m−1})`.
    fn means<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, self.inner.means())
    }

    /// `2m × 2m` symmetric covariance matrix in `xxpp` ordering.
    fn cov<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let dim = 2 * self.inner.n_modes();
        PyArray1::from_slice(py, self.inner.cov()).reshape([dim, dim])
    }

    /// `⟨x̂⟩` on `mode`.
    fn mean_x(&self, mode: usize) -> PyResult<f64> {
        if mode >= self.inner.n_modes() {
            return Err(PyValueError::new_err("mode out of range"));
        }
        Ok(self.inner.mean_x(mode))
    }

    /// `⟨p̂⟩` on `mode`.
    fn mean_p(&self, mode: usize) -> PyResult<f64> {
        if mode >= self.inner.n_modes() {
            return Err(PyValueError::new_err("mode out of range"));
        }
        Ok(self.inner.mean_p(mode))
    }

    /// Mean photon number `⟨n̂⟩` on `mode`.
    fn mean_photons(&self, mode: usize) -> PyResult<f64> {
        if mode >= self.inner.n_modes() {
            return Err(PyValueError::new_err("mode out of range"));
        }
        Ok(self.inner.mean_photons(mode))
    }

    /// State purity `Tr(ρ²) = 1 / √det(V)`.
    fn purity(&self) -> f64 {
        self.inner.purity()
    }

    /// Marginal Gaussian state on `modes`.
    fn reduced(&self, modes: Vec<usize>) -> PyResult<Self> {
        Ok(PyGaussianState {
            inner: self.inner.reduced(&modes).map_err(to_py_err)?,
            rng: rng_from_seed(None),
        })
    }

    /// Wigner function `W(x, p)` at phase-space coordinates `(x, p)`.
    fn wigner(&self, x: Vec<f64>, p: Vec<f64>) -> PyResult<f64> {
        self.inner.wigner(&x, &p).map_err(to_py_err)
    }

    /// Probability of observing vacuum on all modes.
    fn vacuum_probability(&self) -> f64 {
        self.inner.vacuum_probability()
    }

    /// Exact photon-number probability `P(n_0, …, n_{m−1})` via loop hafnians.
    fn fock_probability(&self, py: Python<'_>, pattern: Vec<usize>) -> PyResult<f64> {
        let inner = &self.inner;
        py.detach(|| inner.fock_probability(&pattern))
            .map_err(to_py_err)
    }

    /// Exact threshold-detector probability `P(clicks)` via recursive Cholesky.
    fn threshold_probability(&self, py: Python<'_>, clicks: Vec<bool>) -> PyResult<f64> {
        let inner = &self.inner;
        py.detach(|| inner.threshold_probability(&clicks))
            .map_err(to_py_err)
    }

    /// Apply rotation `R(φ)` in place.
    fn rotation(
        mut slf: PyRefMut<'_, Self>,
        phi: f64,
        mode: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.rotation(phi, mode).map_err(to_py_err)?;
        Ok(slf)
    }

    /// Apply displacement `D(r e^{iφ})` in place.
    fn displace(
        mut slf: PyRefMut<'_, Self>,
        r: f64,
        phi: f64,
        mode: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.displace(r, phi, mode).map_err(to_py_err)?;
        Ok(slf)
    }

    /// Apply squeezing `S(r e^{iφ})` in place.
    fn squeeze(
        mut slf: PyRefMut<'_, Self>,
        r: f64,
        phi: f64,
        mode: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.squeeze(r, phi, mode).map_err(to_py_err)?;
        Ok(slf)
    }

    /// Apply quadratic phase `P(s)` in place.
    fn quadratic_phase(
        mut slf: PyRefMut<'_, Self>,
        s: f64,
        mode: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.quadratic_phase(s, mode).map_err(to_py_err)?;
        Ok(slf)
    }

    /// Apply beam splitter `BS(θ, φ)` on `(a, b)` in place.
    fn beam_splitter(
        mut slf: PyRefMut<'_, Self>,
        theta: f64,
        phi: f64,
        a: usize,
        b: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner
            .beam_splitter(theta, phi, a, b)
            .map_err(to_py_err)?;
        Ok(slf)
    }

    /// Apply two-mode squeezing `S₂(r e^{iφ})` on `(a, b)` in place.
    fn two_mode_squeeze(
        mut slf: PyRefMut<'_, Self>,
        r: f64,
        phi: f64,
        a: usize,
        b: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner
            .two_mode_squeeze(r, phi, a, b)
            .map_err(to_py_err)?;
        Ok(slf)
    }

    /// Apply `CX(s)` on `(a, b)` in place.
    fn cx(mut slf: PyRefMut<'_, Self>, s: f64, a: usize, b: usize) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.cx(s, a, b).map_err(to_py_err)?;
        Ok(slf)
    }

    /// Apply `CZ(s)` on `(a, b)` in place.
    fn cz(mut slf: PyRefMut<'_, Self>, s: f64, a: usize, b: usize) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.cz(s, a, b).map_err(to_py_err)?;
        Ok(slf)
    }

    /// Apply passive linear-optics unitary `U` on `modes` in place.
    fn passive<'py>(
        mut slf: PyRefMut<'py, Self>,
        modes: Vec<usize>,
        matrix: Matrix2C<'_>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let (mat, k) = matrix.to_square_vec()?;
        if k != modes.len() {
            return Err(PyValueError::new_err(format!(
                "passive block on {} modes got a {k}×{k} matrix",
                modes.len()
            )));
        }
        slf.inner.apply_passive(&modes, &mat).map_err(to_py_err)?;
        Ok(slf)
    }

    /// Apply a `2k × 2k` real symplectic matrix on `modes` (`xxpp` ordering) in place.
    fn symplectic<'py>(
        mut slf: PyRefMut<'py, Self>,
        modes: Vec<usize>,
        matrix: Matrix2F<'_>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let (mat, d) = matrix.to_square_vec()?;
        if d != 2 * modes.len() {
            return Err(PyValueError::new_err(format!(
                "symplectic block on {} modes expects a {0}×{0} matrix, got {d}×{d}",
                2 * modes.len()
            )));
        }
        slf.inner
            .apply_symplectic(&modes, &mat)
            .map_err(to_py_err)?;
        Ok(slf)
    }

    /// Apply a pure-loss channel with energy transmissivity `eta ∈ [0, 1]` on `mode`.
    fn loss(mut slf: PyRefMut<'_, Self>, mode: usize, eta: f64) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.apply_loss(mode, eta).map_err(to_py_err)?;
        Ok(slf)
    }

    /// Apply a thermal-loss channel on `mode`.
    fn thermal_loss(
        mut slf: PyRefMut<'_, Self>,
        mode: usize,
        eta: f64,
        mean_photons: f64,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner
            .apply_thermal_loss(mode, eta, mean_photons)
            .map_err(to_py_err)?;
        Ok(slf)
    }

    /// Apply isotropic Gaussian additive noise (`V → V + variance · I₂`) on `mode`.
    fn additive_noise(
        mut slf: PyRefMut<'_, Self>,
        mode: usize,
        variance: f64,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner
            .apply_additive_noise(mode, variance)
            .map_err(to_py_err)?;
        Ok(slf)
    }

    /// Apply a Gaussian `CvChannel` on `mode`.
    fn apply_channel(
        mut slf: PyRefMut<'_, Self>,
        channel: PyCvChannel,
        mode: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner
            .apply_channel(&channel.inner, mode)
            .map_err(to_py_err)?;
        Ok(slf)
    }

    /// Destructive homodyne measurement of `x̂ cos φ + p̂ sin φ` on `mode`.
    fn measure_homodyne(&mut self, phi: f64, mode: usize) -> PyResult<f64> {
        self.inner
            .measure_homodyne(mode, phi, &mut self.rng)
            .map_err(to_py_err)
    }

    /// Postselect homodyne measurement on `mode` onto `outcome`, returning its density.
    fn postselect_homodyne(&mut self, phi: f64, outcome: f64, mode: usize) -> PyResult<f64> {
        self.inner
            .postselect_homodyne(mode, phi, outcome)
            .map_err(to_py_err)
    }

    /// Destructive heterodyne measurement on `mode`, returning `(Re α, Im α)`.
    fn measure_heterodyne(&mut self, mode: usize) -> PyResult<(f64, f64)> {
        self.inner
            .measure_heterodyne(mode, &mut self.rng)
            .map_err(to_py_err)
    }

    /// Postselect heterodyne measurement on `mode` onto `(re, im)`, returning its density.
    fn postselect_heterodyne(&mut self, re: f64, im: f64, mode: usize) -> PyResult<f64> {
        self.inner
            .postselect_heterodyne(mode, re, im)
            .map_err(to_py_err)
    }

    /// Draw `shots` exact photon-number-resolving (PNR) GBS samples, shape `(shots, n_modes)`.
    #[pyo3(signature = (shots, max_photons=None))]
    fn sample_pnr<'py>(
        &mut self,
        py: Python<'py>,
        shots: usize,
        max_photons: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray2<usize>>> {
        let inner = &self.inner;
        let m = inner.n_modes();
        let mut rng = self.rng.clone();
        let rows = py
            .detach(|| rust_sample_gbs_pnr(inner, max_photons, shots, &mut rng))
            .map_err(to_py_err)?;
        self.rng = rng;
        let mut flat = Vec::with_capacity(shots * m);
        for row in rows {
            flat.extend_from_slice(&row);
        }
        PyArray1::from_vec(py, flat).reshape([shots, m])
    }

    /// Draw `shots` exact threshold-detector GBS click patterns, shape `(shots, n_modes)`.
    fn sample_threshold<'py>(
        &mut self,
        py: Python<'py>,
        shots: usize,
    ) -> PyResult<Bound<'py, PyArray2<bool>>> {
        let inner = &self.inner;
        let m = inner.n_modes();
        let mut rng = self.rng.clone();
        let rows = py
            .detach(|| rust_sample_gbs_threshold(inner, shots, &mut rng))
            .map_err(to_py_err)?;
        self.rng = rng;
        let mut flat = Vec::with_capacity(shots * m);
        for row in rows {
            flat.extend_from_slice(&row);
        }
        PyArray1::from_vec(py, flat).reshape([shots, m])
    }

    fn __repr__(&self) -> String {
        format!("GaussianState(n_modes={})", self.inner.n_modes())
    }
}

/// Hafnian (or loop hafnian when `loop=True`) of a square symmetric matrix.
#[pyfunction]
#[pyo3(signature = (matrix, r#loop=false))]
fn hafnian(py: Python<'_>, matrix: Matrix2C<'_>, r#loop: bool) -> PyResult<C64> {
    let (a, n) = matrix.to_square_vec()?;
    if r#loop {
        let diag: Vec<C64> = (0..n).map(|i| a[i * n + i]).collect();
        py.detach(|| rust_loop_hafnian(&a, &diag, n))
            .map_err(to_py_err)
    } else {
        py.detach(|| rust_hafnian(&a, n)).map_err(to_py_err)
    }
}

/// Loop hafnian of a square symmetric matrix with optional explicit loop weights `diag`.
#[pyfunction]
#[pyo3(signature = (matrix, diag=None))]
fn loop_hafnian(py: Python<'_>, matrix: Matrix2C<'_>, diag: Option<Vec1C<'_>>) -> PyResult<C64> {
    let (a, n) = matrix.to_square_vec()?;
    let d = match diag {
        Some(v) => v.to_vec(),
        None => (0..n).map(|i| a[i * n + i]).collect(),
    };
    py.detach(|| rust_loop_hafnian(&a, &d, n))
        .map_err(to_py_err)
}

/// Hafnian (or loop hafnian) of a symmetric matrix with repeated rows/columns.
#[pyfunction]
#[pyo3(signature = (matrix, repeats, r#loop=false, diag=None))]
fn hafnian_repeated(
    py: Python<'_>,
    matrix: Matrix2C<'_>,
    repeats: Vec<usize>,
    r#loop: bool,
    diag: Option<Vec1C<'_>>,
) -> PyResult<C64> {
    let (a, m) = matrix.to_square_vec()?;
    if repeats.len() != m {
        return Err(PyValueError::new_err(format!(
            "matrix is {m}×{m}, got {} repeat counts",
            repeats.len()
        )));
    }
    if r#loop || diag.is_some() {
        let d = match diag {
            Some(v) => v.to_vec(),
            None => (0..m).map(|i| a[i * m + i]).collect(),
        };
        py.detach(|| rust_loop_hafnian_repeated(&a, &d, &repeats))
            .map_err(to_py_err)
    } else {
        py.detach(|| rust_hafnian_repeated(&a, &repeats))
            .map_err(to_py_err)
    }
}

/// Torontonian of a `2m × 2m` matrix `O` in `xxpp` ordering.
#[pyfunction]
fn torontonian(py: Python<'_>, matrix: Matrix2C<'_>) -> PyResult<C64> {
    let (a, dim) = matrix.to_square_vec()?;
    if dim % 2 != 0 {
        return Err(PyValueError::new_err(format!(
            "torontonian expects an even-sized 2m×2m matrix, got {dim}×{dim}"
        )));
    }
    py.detach(|| rust_torontonian(&a, dim / 2))
        .map_err(to_py_err)
}

/// A single-mode continuous-variable noise channel (`ħ = 2`).
#[pyclass(name = "CvChannel", module = "ketrs", frozen, from_py_object)]
#[derive(Clone, Copy)]
pub(crate) struct PyCvChannel {
    inner: CvChannel,
}

#[pymethods]
impl PyCvChannel {
    /// Pure photon loss with energy transmissivity `eta ∈ [0, 1]`.
    #[staticmethod]
    fn loss(eta: f64) -> PyResult<Self> {
        let ch = CvChannel::Loss { eta };
        ch.validate().map_err(to_py_err)?;
        Ok(PyCvChannel { inner: ch })
    }

    /// Thermal loss with transmissivity `eta ∈ [0, 1]` and bath `mean_photons ≥ 0`.
    #[staticmethod]
    fn thermal_loss(eta: f64, mean_photons: f64) -> PyResult<Self> {
        let ch = CvChannel::ThermalLoss { eta, mean_photons };
        ch.validate().map_err(to_py_err)?;
        Ok(PyCvChannel { inner: ch })
    }

    /// Phase damping (dephasing) with angle standard deviation `sigma ≥ 0`.
    #[staticmethod]
    fn dephasing(sigma: f64) -> PyResult<Self> {
        let ch = CvChannel::Dephasing { sigma };
        ch.validate().map_err(to_py_err)?;
        Ok(PyCvChannel { inner: ch })
    }

    /// Isotropic Gaussian additive noise adding `variance ≥ 0` to `V_xx` and `V_pp`.
    #[staticmethod]
    fn additive_noise(variance: f64) -> PyResult<Self> {
        let ch = CvChannel::AdditiveNoise { variance };
        ch.validate().map_err(to_py_err)?;
        Ok(PyCvChannel { inner: ch })
    }

    fn __repr__(&self) -> String {
        format!("CvChannel({:?})", self.inner)
    }
}

/// Exact photon-number-conserving (fixed-N) Fock-space simulator in the
/// `C(m + N - 1, N)` basis.
#[pyclass(name = "FixedNFockState", module = "ketrs")]
pub(crate) struct PyFixedNFockState {
    inner: FixedNFockState,
    rng: StdRng,
}

#[pymethods]
impl PyFixedNFockState {
    /// Dimension `C(m + N - 1, N)` of the `m`-mode `N`-photon subspace.
    #[staticmethod]
    fn basis_dim(n_modes: usize, n_photons: usize) -> usize {
        FixedNFockState::basis_dim(n_modes, n_photons)
    }

    /// Construct the single Fock basis state `|pattern[0], …, pattern[m−1]⟩`.
    #[staticmethod]
    #[pyo3(signature = (pattern, seed=None))]
    fn from_fock(pattern: Vec<usize>, seed: Option<u64>) -> PyResult<Self> {
        Ok(PyFixedNFockState {
            inner: FixedNFockState::from_fock(&pattern).map_err(to_py_err)?,
            rng: rng_from_seed(seed),
        })
    }

    /// Compute `U |input⟩` for an `m × m` passive unitary `U` via the recursive
    /// cache-optimal SLOS pass.
    #[staticmethod]
    #[pyo3(signature = (input, unitary, seed=None))]
    fn from_passive_unitary(
        py: Python<'_>,
        input: Vec<usize>,
        unitary: Matrix2C<'_>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let (u, m) = unitary.to_square_vec()?;
        if m != input.len() {
            return Err(PyValueError::new_err(format!(
                "input has {} modes, got a {m}×{m} unitary",
                input.len()
            )));
        }
        let inner = py
            .detach(|| FixedNFockState::from_passive_unitary(&input, &u))
            .map_err(to_py_err)?;
        Ok(PyFixedNFockState {
            inner,
            rng: rng_from_seed(seed),
        })
    }

    /// Run a bound, measurement-free photon-number-conserving `ModeCircuit`.
    #[staticmethod]
    #[pyo3(signature = (circuit, seed=None))]
    fn from_circuit(
        py: Python<'_>,
        circuit: PyRef<'_, PyModeCircuit>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let circ = &circuit.inner;
        let inner = py
            .detach(|| FixedNFockState::from_circuit(circ))
            .map_err(to_py_err)?;
        Ok(PyFixedNFockState {
            inner,
            rng: rng_from_seed(seed),
        })
    }

    /// Run one shot of a bound photon-number-conserving circuit with PNR /
    /// threshold measurements and feed-forward.
    #[staticmethod]
    #[pyo3(signature = (circuit, seed=None))]
    fn run<'py>(
        py: Python<'py>,
        circuit: PyRef<'_, PyModeCircuit>,
        seed: Option<u64>,
    ) -> PyResult<(Self, Bound<'py, PyArray1<f64>>)> {
        let circ = &circuit.inner;
        let mut rng = rng_from_seed(seed);
        let (inner, results) = py
            .detach(|| FixedNFockState::run(circ, &mut rng))
            .map_err(to_py_err)?;
        Ok((
            PyFixedNFockState { inner, rng },
            PyArray1::from_vec(py, results),
        ))
    }

    /// Sample `shots` independent measurement outcomes, shape `(shots, n_results)`.
    #[staticmethod]
    #[pyo3(signature = (circuit, shots, seed=None))]
    fn sample_circuit<'py>(
        py: Python<'py>,
        circuit: PyRef<'_, PyModeCircuit>,
        shots: usize,
        seed: Option<u64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let circ = &circuit.inner;
        let n_res = circ.n_results();
        let mut rng = rng_from_seed(seed);
        let rows = py
            .detach(|| FixedNFockState::sample(circ, shots, &mut rng))
            .map_err(to_py_err)?;
        let mut flat = Vec::with_capacity(shots * n_res);
        for row in rows {
            flat.extend_from_slice(&row);
        }
        PyArray1::from_vec(py, flat).reshape([shots, n_res])
    }

    #[getter]
    fn n_modes(&self) -> usize {
        self.inner.n_modes()
    }

    #[getter]
    fn n_photons(&self) -> usize {
        self.inner.n_photons()
    }

    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn amplitudes<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<C64>> {
        PyArray1::from_slice(py, self.inner.amplitudes())
    }

    fn basis(&self) -> Vec<Vec<usize>> {
        self.inner.basis()
    }

    fn probabilities<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.probabilities())
    }

    fn amplitude(&self, pattern: Vec<usize>) -> PyResult<C64> {
        self.inner.amplitude(&pattern).map_err(to_py_err)
    }

    fn probability(&self, pattern: Vec<usize>) -> PyResult<f64> {
        self.inner.probability(&pattern).map_err(to_py_err)
    }

    fn mean_photons(&self, mode: usize) -> PyResult<f64> {
        self.inner.mean_photons(mode).map_err(to_py_err)
    }

    fn rotation(
        mut slf: PyRefMut<'_, Self>,
        phi: f64,
        mode: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.rotation(phi, mode).map_err(to_py_err)?;
        Ok(slf)
    }

    fn kerr(mut slf: PyRefMut<'_, Self>, kappa: f64, mode: usize) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.kerr(kappa, mode).map_err(to_py_err)?;
        Ok(slf)
    }

    fn cross_kerr(
        mut slf: PyRefMut<'_, Self>,
        kappa: f64,
        a: usize,
        b: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.cross_kerr(kappa, a, b).map_err(to_py_err)?;
        Ok(slf)
    }

    fn beam_splitter(
        mut slf: PyRefMut<'_, Self>,
        theta: f64,
        phi: f64,
        a: usize,
        b: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner
            .beam_splitter(theta, phi, a, b)
            .map_err(to_py_err)?;
        Ok(slf)
    }

    fn passive<'py>(
        mut slf: PyRefMut<'py, Self>,
        modes: Vec<usize>,
        matrix: Matrix2C<'_>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let (mat, k) = matrix.to_square_vec()?;
        if k != modes.len() {
            return Err(PyValueError::new_err(format!(
                "passive block on {} modes got a {k}×{k} matrix",
                modes.len()
            )));
        }
        slf.inner.apply_passive(&modes, &mat).map_err(to_py_err)?;
        Ok(slf)
    }

    fn measure_photons(&mut self, mode: usize) -> PyResult<usize> {
        self.inner
            .measure_photons(mode, &mut self.rng)
            .map_err(to_py_err)
    }

    fn measure_threshold(&mut self, mode: usize) -> PyResult<bool> {
        self.inner
            .measure_threshold(mode, &mut self.rng)
            .map_err(to_py_err)
    }

    fn __repr__(&self) -> String {
        format!(
            "FixedNFockState(n_modes={}, n_photons={}, dim={})",
            self.inner.n_modes(),
            self.inner.n_photons(),
            self.inner.dim()
        )
    }
}

/// Universal cutoff Fock-space statevector (`cutoff^n_modes` amplitudes).
#[pyclass(name = "FockState", module = "ketrs")]
pub(crate) struct PyFockState {
    inner: FockState,
    rng: StdRng,
}

#[pymethods]
impl PyFockState {
    #[new]
    #[pyo3(signature = (n_modes, cutoff, seed=None))]
    fn new(n_modes: usize, cutoff: usize, seed: Option<u64>) -> PyResult<Self> {
        Ok(PyFockState {
            inner: FockState::vacuum(n_modes, cutoff).map_err(to_py_err)?,
            rng: rng_from_seed(seed),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (circuit, cutoff, seed=None))]
    fn from_circuit(
        py: Python<'_>,
        circuit: PyRef<'_, PyModeCircuit>,
        cutoff: usize,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let circ = &circuit.inner;
        let inner = py
            .detach(|| FockState::from_circuit(circ, cutoff))
            .map_err(to_py_err)?;
        Ok(PyFockState {
            inner,
            rng: rng_from_seed(seed),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (circuit, cutoff, seed=None))]
    fn run<'py>(
        py: Python<'py>,
        circuit: PyRef<'_, PyModeCircuit>,
        cutoff: usize,
        seed: Option<u64>,
    ) -> PyResult<(Self, Bound<'py, PyArray1<f64>>)> {
        let circ = &circuit.inner;
        let mut rng = rng_from_seed(seed);
        let (inner, results) = py
            .detach(|| FockState::run(circ, cutoff, &mut rng))
            .map_err(to_py_err)?;
        Ok((PyFockState { inner, rng }, PyArray1::from_vec(py, results)))
    }

    #[staticmethod]
    #[pyo3(signature = (circuit, cutoff, shots, seed=None))]
    fn sample_circuit<'py>(
        py: Python<'py>,
        circuit: PyRef<'_, PyModeCircuit>,
        cutoff: usize,
        shots: usize,
        seed: Option<u64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let circ = &circuit.inner;
        let n_res = circ.n_results();
        let mut rng = rng_from_seed(seed);
        let rows = py
            .detach(|| FockState::sample(circ, cutoff, shots, &mut rng))
            .map_err(to_py_err)?;
        let mut flat = Vec::with_capacity(shots * n_res);
        for row in rows {
            flat.extend_from_slice(&row);
        }
        PyArray1::from_vec(py, flat).reshape([shots, n_res])
    }

    #[getter]
    fn n_modes(&self) -> usize {
        self.inner.n_modes()
    }

    #[getter]
    fn cutoff(&self) -> usize {
        self.inner.cutoff()
    }

    fn amplitudes<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<C64>> {
        PyArray1::from_slice(py, self.inner.amplitudes())
    }

    fn probabilities<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.probabilities())
    }

    fn amplitude(&self, pattern: Vec<usize>) -> PyResult<C64> {
        self.inner.amplitude(&pattern).map_err(to_py_err)
    }

    fn probability(&self, pattern: Vec<usize>) -> PyResult<f64> {
        self.inner.probability(&pattern).map_err(to_py_err)
    }

    fn norm_sqr(&self) -> f64 {
        self.inner.norm_sqr()
    }

    fn truncation_error(&self) -> f64 {
        self.inner.truncation_error()
    }

    fn boundary_weight(&self) -> f64 {
        self.inner.boundary_weight()
    }

    fn mean_photons(&self, mode: usize) -> PyResult<f64> {
        self.inner.mean_photons(mode).map_err(to_py_err)
    }

    fn mean_x(&self, mode: usize) -> PyResult<f64> {
        self.inner.mean_x(mode).map_err(to_py_err)
    }

    fn mean_p(&self, mode: usize) -> PyResult<f64> {
        self.inner.mean_p(mode).map_err(to_py_err)
    }

    fn rotation(
        mut slf: PyRefMut<'_, Self>,
        phi: f64,
        mode: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.rotation(phi, mode).map_err(to_py_err)?;
        Ok(slf)
    }

    fn displace(
        mut slf: PyRefMut<'_, Self>,
        r: f64,
        phi: f64,
        mode: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.displace(r, phi, mode).map_err(to_py_err)?;
        Ok(slf)
    }

    fn squeeze(
        mut slf: PyRefMut<'_, Self>,
        r: f64,
        phi: f64,
        mode: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.squeeze(r, phi, mode).map_err(to_py_err)?;
        Ok(slf)
    }

    fn quadratic_phase(
        mut slf: PyRefMut<'_, Self>,
        s: f64,
        mode: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.quadratic_phase(s, mode).map_err(to_py_err)?;
        Ok(slf)
    }

    fn cubic_phase(
        mut slf: PyRefMut<'_, Self>,
        gamma: f64,
        mode: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.cubic_phase(gamma, mode).map_err(to_py_err)?;
        Ok(slf)
    }

    fn kerr(mut slf: PyRefMut<'_, Self>, kappa: f64, mode: usize) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.kerr(kappa, mode).map_err(to_py_err)?;
        Ok(slf)
    }

    fn beam_splitter(
        mut slf: PyRefMut<'_, Self>,
        theta: f64,
        phi: f64,
        a: usize,
        b: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner
            .beam_splitter(theta, phi, a, b)
            .map_err(to_py_err)?;
        Ok(slf)
    }

    fn two_mode_squeeze(
        mut slf: PyRefMut<'_, Self>,
        r: f64,
        phi: f64,
        a: usize,
        b: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner
            .two_mode_squeeze(r, phi, a, b)
            .map_err(to_py_err)?;
        Ok(slf)
    }

    fn cx(mut slf: PyRefMut<'_, Self>, s: f64, a: usize, b: usize) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.cx(s, a, b).map_err(to_py_err)?;
        Ok(slf)
    }

    fn cz(mut slf: PyRefMut<'_, Self>, s: f64, a: usize, b: usize) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.cz(s, a, b).map_err(to_py_err)?;
        Ok(slf)
    }

    fn cross_kerr(
        mut slf: PyRefMut<'_, Self>,
        kappa: f64,
        a: usize,
        b: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.cross_kerr(kappa, a, b).map_err(to_py_err)?;
        Ok(slf)
    }

    fn passive<'py>(
        mut slf: PyRefMut<'py, Self>,
        modes: Vec<usize>,
        matrix: Matrix2C<'_>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let (mat, k) = matrix.to_square_vec()?;
        if k != modes.len() {
            return Err(PyValueError::new_err(format!(
                "passive block on {} modes got a {k}×{k} matrix",
                modes.len()
            )));
        }
        slf.inner.apply_passive(&modes, &mat).map_err(to_py_err)?;
        Ok(slf)
    }

    /// Apply one stochastic quantum-trajectory jump of `channel` on `mode`.
    fn apply_channel(
        mut slf: PyRefMut<'_, Self>,
        channel: PyCvChannel,
        mode: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        let mut rng = slf.rng.clone();
        slf.inner
            .apply_channel(&channel.inner, mode, &mut rng)
            .map_err(to_py_err)?;
        slf.rng = rng;
        Ok(slf)
    }

    fn measure_photons(&mut self, mode: usize) -> PyResult<usize> {
        self.inner
            .measure_photons(mode, &mut self.rng)
            .map_err(to_py_err)
    }

    fn measure_threshold(&mut self, mode: usize) -> PyResult<bool> {
        self.inner
            .measure_threshold(mode, &mut self.rng)
            .map_err(to_py_err)
    }

    fn measure_homodyne(&mut self, phi: f64, mode: usize) -> PyResult<f64> {
        self.inner
            .measure_homodyne(mode, phi, &mut self.rng)
            .map_err(to_py_err)
    }

    fn measure_heterodyne(&mut self, mode: usize) -> PyResult<(f64, f64)> {
        self.inner
            .measure_heterodyne(mode, &mut self.rng)
            .map_err(to_py_err)
    }

    fn __repr__(&self) -> String {
        format!(
            "FockState(n_modes={}, cutoff={})",
            self.inner.n_modes(),
            self.inner.cutoff()
        )
    }
}

/// Mixed-state cutoff Fock density matrix (`cutoff^m × cutoff^m`).
#[pyclass(name = "FockDensityMatrix", module = "ketrs")]
pub(crate) struct PyFockDensityMatrix {
    inner: FockDensityMatrix,
}

#[pymethods]
impl PyFockDensityMatrix {
    #[new]
    fn new(n_modes: usize, cutoff: usize) -> PyResult<Self> {
        Ok(PyFockDensityMatrix {
            inner: FockDensityMatrix::vacuum(n_modes, cutoff).map_err(to_py_err)?,
        })
    }

    #[staticmethod]
    fn from_state(state: PyRef<'_, PyFockState>) -> Self {
        PyFockDensityMatrix {
            inner: FockDensityMatrix::from_state(&state.inner),
        }
    }

    #[staticmethod]
    fn from_circuit(
        py: Python<'_>,
        circuit: PyRef<'_, PyModeCircuit>,
        cutoff: usize,
    ) -> PyResult<Self> {
        let circ = &circuit.inner;
        let inner = py
            .detach(|| FockDensityMatrix::from_circuit(circ, cutoff))
            .map_err(to_py_err)?;
        Ok(PyFockDensityMatrix { inner })
    }

    #[getter]
    fn n_modes(&self) -> usize {
        self.inner.n_modes()
    }

    #[getter]
    fn cutoff(&self) -> usize {
        self.inner.cutoff()
    }

    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn matrix<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<C64>>> {
        let dim = self.inner.dim();
        PyArray1::from_slice(py, self.inner.matrix()).reshape([dim, dim])
    }

    fn trace(&self) -> f64 {
        self.inner.trace()
    }

    fn truncation_error(&self) -> f64 {
        self.inner.truncation_error()
    }

    fn purity(&self) -> f64 {
        self.inner.purity()
    }

    fn probabilities<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.probabilities())
    }

    fn probability(&self, pattern: Vec<usize>) -> PyResult<f64> {
        self.inner.probability(&pattern).map_err(to_py_err)
    }

    fn mean_photons(&self, mode: usize) -> PyResult<f64> {
        self.inner.mean_photons(mode).map_err(to_py_err)
    }

    fn mean_x(&self, mode: usize) -> PyResult<f64> {
        self.inner.mean_x(mode).map_err(to_py_err)
    }

    fn mean_p(&self, mode: usize) -> PyResult<f64> {
        self.inner.mean_p(mode).map_err(to_py_err)
    }

    fn apply_channel(
        mut slf: PyRefMut<'_, Self>,
        channel: PyCvChannel,
        mode: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner
            .apply_channel(&channel.inner, mode)
            .map_err(to_py_err)?;
        Ok(slf)
    }

    fn __repr__(&self) -> String {
        format!(
            "FockDensityMatrix(n_modes={}, cutoff={}, dim={})",
            self.inner.n_modes(),
            self.inner.cutoff(),
            self.inner.dim()
        )
    }
}

/// Sliding-window sampler for local / streaming CV circuits (e.g. time-bin loops).
#[pyclass(name = "WindowSampler", module = "ketrs", frozen)]
pub(crate) struct PyWindowSampler;

#[pymethods]
impl PyWindowSampler {
    /// Peak simultaneous active mode count `W` required by `circuit`.
    #[staticmethod]
    fn max_window_width(circuit: PyRef<'_, PyModeCircuit>) -> usize {
        WindowSampler::max_window_width(&circuit.inner)
    }

    /// Sample `shots` independent measurement trajectories using the sliding
    /// window cutoff Fock engine, returning shape `(shots, n_results)`.
    #[staticmethod]
    #[pyo3(signature = (circuit, cutoff, shots, seed=None))]
    fn sample<'py>(
        py: Python<'py>,
        circuit: PyRef<'_, PyModeCircuit>,
        cutoff: usize,
        shots: usize,
        seed: Option<u64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let circ = &circuit.inner;
        let n_res = circ.n_results();
        let mut rng = rng_from_seed(seed);
        let rows = py
            .detach(|| WindowSampler::sample(circ, cutoff, shots, &mut rng))
            .map_err(to_py_err)?;
        let mut flat = Vec::with_capacity(shots * n_res);
        for row in rows {
            flat.extend_from_slice(&row);
        }
        PyArray1::from_vec(py, flat).reshape([shots, n_res])
    }
}

/// Exact permanent of a square matrix via Glynn's Gray-code formula.
#[pyfunction]
fn permanent(py: Python<'_>, matrix: Matrix2C<'_>) -> PyResult<C64> {
    let (a, n) = matrix.to_square_vec()?;
    py.detach(|| rust_permanent(&a, n)).map_err(to_py_err)
}

/// Exact boson sampling via Clifford & Clifford (2018 / 2020) with simultaneous
/// Glynn Gray-code Laplace permanents. Returns shape `(shots, m)`.
#[pyfunction]
#[pyo3(signature = (unitary, input, shots, seed=None))]
fn boson_sample<'py>(
    py: Python<'py>,
    unitary: Matrix2C<'_>,
    input: Vec<usize>,
    shots: usize,
    seed: Option<u64>,
) -> PyResult<Bound<'py, PyArray2<usize>>> {
    let (u, m) = unitary.to_square_vec()?;
    if m != input.len() {
        return Err(PyValueError::new_err(format!(
            "input has {} modes, got a {m}×{m} unitary",
            input.len()
        )));
    }
    let mut rng = rng_from_seed(seed);
    let rows = py
        .detach(|| rust_boson_sample(&u, &input, shots, &mut rng))
        .map_err(to_py_err)?;
    let mut flat = Vec::with_capacity(shots * m);
    for row in rows {
        flat.extend_from_slice(&row);
    }
    PyArray1::from_vec(py, flat).reshape([shots, m])
}

/// Exact PNR Gaussian Boson Sampling from a `GaussianState`, shape `(shots, n_modes)`.
#[pyfunction]
#[pyo3(signature = (state, shots, max_photons=None, seed=None))]
fn sample_gbs_pnr<'py>(
    py: Python<'py>,
    state: PyRef<'_, PyGaussianState>,
    shots: usize,
    max_photons: Option<usize>,
    seed: Option<u64>,
) -> PyResult<Bound<'py, PyArray2<usize>>> {
    let inner = &state.inner;
    let m = inner.n_modes();
    let mut rng = rng_from_seed(seed);
    let rows = py
        .detach(|| rust_sample_gbs_pnr(inner, max_photons, shots, &mut rng))
        .map_err(to_py_err)?;
    let mut flat = Vec::with_capacity(shots * m);
    for row in rows {
        flat.extend_from_slice(&row);
    }
    PyArray1::from_vec(py, flat).reshape([shots, m])
}

/// Exact threshold-detector Gaussian Boson Sampling from a `GaussianState`, shape `(shots, n_modes)`.
#[pyfunction]
#[pyo3(signature = (state, shots, seed=None))]
fn sample_gbs_threshold<'py>(
    py: Python<'py>,
    state: PyRef<'_, PyGaussianState>,
    shots: usize,
    seed: Option<u64>,
) -> PyResult<Bound<'py, PyArray2<bool>>> {
    let inner = &state.inner;
    let m = inner.n_modes();
    let mut rng = rng_from_seed(seed);
    let rows = py
        .detach(|| rust_sample_gbs_threshold(inner, shots, &mut rng))
        .map_err(to_py_err)?;
    let mut flat = Vec::with_capacity(shots * m);
    for row in rows {
        flat.extend_from_slice(&row);
    }
    PyArray1::from_vec(py, flat).reshape([shots, m])
}

/// Linear-combination-of-Gaussians (Bourassa et al. 2021) simulator for
/// Gaussian circuits on cat, GKP, Fock, and Gaussian inputs.
#[pyclass(name = "GaussianMixture", module = "ketrs")]
pub(crate) struct PyGaussianMixture {
    inner: GaussianMixture,
    rng: StdRng,
}

fn make_mixture_options(prune_tol: f64, merge_tol: f64, fock_step: f64) -> MixtureOptions {
    MixtureOptions {
        prune_tol,
        merge_tol,
        fock_step,
    }
}

#[pymethods]
impl PyGaussianMixture {
    /// Vacuum state `|0…0⟩` on `n_modes` modes.
    #[new]
    #[pyo3(signature = (n_modes, prune_tol=1e-6, merge_tol=1e-9, fock_step=1e-3, seed=None))]
    fn new(
        n_modes: usize,
        prune_tol: f64,
        merge_tol: f64,
        fock_step: f64,
        seed: Option<u64>,
    ) -> Self {
        PyGaussianMixture {
            inner: GaussianMixture::vacuum(
                n_modes,
                make_mixture_options(prune_tol, merge_tol, fock_step),
            ),
            rng: rng_from_seed(seed),
        }
    }

    /// Run a bound, measurement-free Gaussian `ModeCircuit` on cat, GKP, Fock,
    /// or Gaussian inputs.
    #[staticmethod]
    #[pyo3(signature = (circuit, prune_tol=1e-6, merge_tol=1e-9, fock_step=1e-3, seed=None))]
    fn from_circuit(
        py: Python<'_>,
        circuit: PyRef<'_, PyModeCircuit>,
        prune_tol: f64,
        merge_tol: f64,
        fock_step: f64,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let circ = &circuit.inner;
        let opts = make_mixture_options(prune_tol, merge_tol, fock_step);
        let inner = py
            .detach(|| GaussianMixture::from_circuit(circ, opts))
            .map_err(to_py_err)?;
        Ok(PyGaussianMixture {
            inner,
            rng: rng_from_seed(seed),
        })
    }

    /// Run one shot of a bound Gaussian `ModeCircuit` (with homodyne/heterodyne
    /// measurements and feed-forward), returning `(state, results)`.
    #[staticmethod]
    #[pyo3(signature = (circuit, prune_tol=1e-6, merge_tol=1e-9, fock_step=1e-3, seed=None))]
    fn run<'py>(
        py: Python<'py>,
        circuit: PyRef<'_, PyModeCircuit>,
        prune_tol: f64,
        merge_tol: f64,
        fock_step: f64,
        seed: Option<u64>,
    ) -> PyResult<(Self, Bound<'py, PyArray1<f64>>)> {
        let circ = &circuit.inner;
        let opts = make_mixture_options(prune_tol, merge_tol, fock_step);
        let mut rng = rng_from_seed(seed);
        let (inner, results) = py
            .detach(|| GaussianMixture::run(circ, opts, &mut rng))
            .map_err(to_py_err)?;
        Ok((
            PyGaussianMixture { inner, rng },
            PyArray1::from_vec(py, results),
        ))
    }

    /// Sample `shots` independent measurement trajectories from a bound
    /// Gaussian `ModeCircuit`, returning shape `(shots, n_results)`.
    #[staticmethod]
    #[pyo3(signature = (circuit, shots, prune_tol=1e-6, merge_tol=1e-9, fock_step=1e-3, seed=None))]
    fn sample_circuit<'py>(
        py: Python<'py>,
        circuit: PyRef<'_, PyModeCircuit>,
        shots: usize,
        prune_tol: f64,
        merge_tol: f64,
        fock_step: f64,
        seed: Option<u64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let circ = &circuit.inner;
        let n_res = circ.n_results();
        let opts = make_mixture_options(prune_tol, merge_tol, fock_step);
        let mut rng = rng_from_seed(seed);
        let rows = py
            .detach(|| GaussianMixture::sample(circ, opts, shots, &mut rng))
            .map_err(to_py_err)?;
        let mut flat = Vec::with_capacity(shots * n_res);
        for row in rows {
            flat.extend_from_slice(&row);
        }
        PyArray1::from_vec(py, flat).reshape([shots, n_res])
    }

    #[getter]
    fn n_modes(&self) -> usize {
        self.inner.n_modes()
    }

    #[getter]
    fn n_terms(&self) -> usize {
        self.inner.n_terms()
    }

    #[getter]
    fn error_bound(&self) -> f64 {
        self.inner.error_bound()
    }

    /// Evaluate the Wigner function `W(x, p)`.
    fn wigner(&self, x: Vec<f64>, p: Vec<f64>) -> PyResult<f64> {
        self.inner.wigner(&x, &p).map_err(to_py_err)
    }

    /// Marginal homodyne probability density `p(q)` of `x̂_mode cos φ + p̂_mode sin φ`.
    fn homodyne_pdf(&self, phi: f64, q: f64, mode: usize) -> PyResult<f64> {
        self.inner.homodyne_pdf(mode, phi, q).map_err(to_py_err)
    }

    fn mean_x(&self, mode: usize) -> PyResult<f64> {
        self.inner.mean_x(mode).map_err(to_py_err)
    }

    fn mean_p(&self, mode: usize) -> PyResult<f64> {
        self.inner.mean_p(mode).map_err(to_py_err)
    }

    fn mean_photons(&self, mode: usize) -> PyResult<f64> {
        self.inner.mean_photons(mode).map_err(to_py_err)
    }

    fn rotation(
        mut slf: PyRefMut<'_, Self>,
        phi: f64,
        mode: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.rotation(phi, mode).map_err(to_py_err)?;
        Ok(slf)
    }

    fn displace(
        mut slf: PyRefMut<'_, Self>,
        r: f64,
        phi: f64,
        mode: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.displace(r, phi, mode).map_err(to_py_err)?;
        Ok(slf)
    }

    fn squeeze(
        mut slf: PyRefMut<'_, Self>,
        r: f64,
        phi: f64,
        mode: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.squeeze(r, phi, mode).map_err(to_py_err)?;
        Ok(slf)
    }

    fn quadratic_phase(
        mut slf: PyRefMut<'_, Self>,
        s: f64,
        mode: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.quadratic_phase(s, mode).map_err(to_py_err)?;
        Ok(slf)
    }

    fn beam_splitter(
        mut slf: PyRefMut<'_, Self>,
        theta: f64,
        phi: f64,
        a: usize,
        b: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner
            .beam_splitter(theta, phi, a, b)
            .map_err(to_py_err)?;
        Ok(slf)
    }

    fn two_mode_squeeze(
        mut slf: PyRefMut<'_, Self>,
        r: f64,
        phi: f64,
        a: usize,
        b: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner
            .two_mode_squeeze(r, phi, a, b)
            .map_err(to_py_err)?;
        Ok(slf)
    }

    fn cx(mut slf: PyRefMut<'_, Self>, s: f64, a: usize, b: usize) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.cx(s, a, b).map_err(to_py_err)?;
        Ok(slf)
    }

    fn cz(mut slf: PyRefMut<'_, Self>, s: f64, a: usize, b: usize) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.cz(s, a, b).map_err(to_py_err)?;
        Ok(slf)
    }

    fn passive<'py>(
        mut slf: PyRefMut<'py, Self>,
        modes: Vec<usize>,
        matrix: Matrix2C<'_>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let (mat, k) = matrix.to_square_vec()?;
        if k != modes.len() {
            return Err(PyValueError::new_err(format!(
                "passive block on {} modes got a {k}×{k} matrix",
                modes.len()
            )));
        }
        slf.inner.apply_passive(&modes, &mat).map_err(to_py_err)?;
        Ok(slf)
    }

    fn apply_channel(
        mut slf: PyRefMut<'_, Self>,
        channel: PyCvChannel,
        mode: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner
            .apply_channel(&channel.inner, mode)
            .map_err(to_py_err)?;
        Ok(slf)
    }

    fn measure_homodyne(&mut self, phi: f64, mode: usize) -> PyResult<f64> {
        let Self { inner, rng } = self;
        inner.measure_homodyne(mode, phi, rng).map_err(to_py_err)
    }

    fn postselect_homodyne(&mut self, phi: f64, outcome: f64, mode: usize) -> PyResult<f64> {
        self.inner
            .postselect_homodyne(mode, phi, outcome)
            .map_err(to_py_err)
    }

    fn measure_heterodyne(&mut self, mode: usize) -> PyResult<(f64, f64)> {
        let Self { inner, rng } = self;
        inner.measure_heterodyne(mode, rng).map_err(to_py_err)
    }

    fn postselect_heterodyne(&mut self, re: f64, im: f64, mode: usize) -> PyResult<f64> {
        self.inner
            .postselect_heterodyne(mode, re, im)
            .map_err(to_py_err)
    }

    fn __repr__(&self) -> String {
        format!(
            "GaussianMixture(n_modes={}, n_terms={}, error_bound={:.2e})",
            self.inner.n_modes(),
            self.inner.n_terms(),
            self.inner.error_bound()
        )
    }
}

/// Bosonic Matrix Product State simulator for large, weakly entangled CV circuits.
#[pyclass(name = "BosonicMps", module = "ketrs")]
pub(crate) struct PyBosonicMps {
    inner: BosonicMps,
    rng: StdRng,
}

#[pymethods]
impl PyBosonicMps {
    /// Vacuum state `|0…0⟩` on `n_modes` modes with Fock cutoff `cutoff`.
    #[new]
    #[pyo3(signature = (n_modes, cutoff, max_bond=64, cutoff_tol=1e-10, seed=None))]
    fn new(
        n_modes: usize,
        cutoff: usize,
        max_bond: usize,
        cutoff_tol: f64,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let opts = BosonicMpsOptions {
            max_bond,
            cutoff_tol,
        };
        Ok(PyBosonicMps {
            inner: BosonicMps::vacuum(n_modes, cutoff, opts).map_err(to_py_err)?,
            rng: rng_from_seed(seed),
        })
    }

    /// Run a bound, measurement-free `ModeCircuit` in the Bosonic MPS representation.
    #[staticmethod]
    #[pyo3(signature = (circuit, cutoff, max_bond=64, cutoff_tol=1e-10, seed=None))]
    fn from_circuit(
        py: Python<'_>,
        circuit: PyRef<'_, PyModeCircuit>,
        cutoff: usize,
        max_bond: usize,
        cutoff_tol: f64,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let circ = &circuit.inner;
        let opts = BosonicMpsOptions {
            max_bond,
            cutoff_tol,
        };
        let inner = py
            .detach(|| BosonicMps::from_circuit(circ, cutoff, opts))
            .map_err(to_py_err)?;
        Ok(PyBosonicMps {
            inner,
            rng: rng_from_seed(seed),
        })
    }

    /// Run one shot of a bound `ModeCircuit` with measurements & feed-forward.
    #[staticmethod]
    #[pyo3(signature = (circuit, cutoff, max_bond=64, cutoff_tol=1e-10, seed=None))]
    fn run<'py>(
        py: Python<'py>,
        circuit: PyRef<'_, PyModeCircuit>,
        cutoff: usize,
        max_bond: usize,
        cutoff_tol: f64,
        seed: Option<u64>,
    ) -> PyResult<(Self, Bound<'py, PyArray1<f64>>)> {
        let circ = &circuit.inner;
        let opts = BosonicMpsOptions {
            max_bond,
            cutoff_tol,
        };
        let mut rng = rng_from_seed(seed);
        let (inner, results) = py
            .detach(|| BosonicMps::run(circ, cutoff, opts, &mut rng))
            .map_err(to_py_err)?;
        Ok((PyBosonicMps { inner, rng }, PyArray1::from_vec(py, results)))
    }

    /// Sample `shots` independent measurement trajectories, returning shape `(shots, n_results)`.
    #[staticmethod]
    #[pyo3(signature = (circuit, cutoff, shots, max_bond=64, cutoff_tol=1e-10, seed=None))]
    fn sample_circuit<'py>(
        py: Python<'py>,
        circuit: PyRef<'_, PyModeCircuit>,
        cutoff: usize,
        shots: usize,
        max_bond: usize,
        cutoff_tol: f64,
        seed: Option<u64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let circ = &circuit.inner;
        let n_res = circ.n_results();
        let opts = BosonicMpsOptions {
            max_bond,
            cutoff_tol,
        };
        let mut rng = rng_from_seed(seed);
        let rows = py
            .detach(|| BosonicMps::sample(circ, cutoff, opts, shots, &mut rng))
            .map_err(to_py_err)?;
        let mut flat = Vec::with_capacity(shots * n_res);
        for row in rows {
            flat.extend_from_slice(&row);
        }
        PyArray1::from_vec(py, flat).reshape([shots, n_res])
    }

    #[getter]
    fn n_modes(&self) -> usize {
        self.inner.n_modes()
    }

    #[getter]
    fn cutoff(&self) -> usize {
        self.inner.cutoff()
    }

    #[getter]
    fn bond_dims(&self) -> Vec<usize> {
        self.inner.bond_dims()
    }

    #[getter]
    fn max_bond_dim(&self) -> usize {
        self.inner.max_bond_dim()
    }

    fn norm_sqr(&self) -> f64 {
        self.inner.norm_sqr()
    }

    fn truncation_error(&self) -> f64 {
        self.inner.truncation_error()
    }

    fn amplitude(&self, pattern: Vec<usize>) -> PyResult<C64> {
        self.inner.amplitude(&pattern).map_err(to_py_err)
    }

    fn probability(&self, pattern: Vec<usize>) -> PyResult<f64> {
        self.inner.probability(&pattern).map_err(to_py_err)
    }

    fn mean_photons(&mut self, mode: usize) -> PyResult<f64> {
        self.inner.mean_photons(mode).map_err(to_py_err)
    }

    fn mean_x(&mut self, mode: usize) -> PyResult<f64> {
        self.inner.mean_x(mode).map_err(to_py_err)
    }

    fn mean_p(&mut self, mode: usize) -> PyResult<f64> {
        self.inner.mean_p(mode).map_err(to_py_err)
    }

    fn rotation(
        mut slf: PyRefMut<'_, Self>,
        phi: f64,
        mode: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.rotation(phi, mode).map_err(to_py_err)?;
        Ok(slf)
    }

    fn displace(
        mut slf: PyRefMut<'_, Self>,
        r: f64,
        phi: f64,
        mode: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.displace(r, phi, mode).map_err(to_py_err)?;
        Ok(slf)
    }

    fn squeeze(
        mut slf: PyRefMut<'_, Self>,
        r: f64,
        phi: f64,
        mode: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.squeeze(r, phi, mode).map_err(to_py_err)?;
        Ok(slf)
    }

    fn quadratic_phase(
        mut slf: PyRefMut<'_, Self>,
        s: f64,
        mode: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.quadratic_phase(s, mode).map_err(to_py_err)?;
        Ok(slf)
    }

    fn cubic_phase(
        mut slf: PyRefMut<'_, Self>,
        gamma: f64,
        mode: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.cubic_phase(gamma, mode).map_err(to_py_err)?;
        Ok(slf)
    }

    fn kerr(mut slf: PyRefMut<'_, Self>, kappa: f64, mode: usize) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.kerr(kappa, mode).map_err(to_py_err)?;
        Ok(slf)
    }

    fn beam_splitter(
        mut slf: PyRefMut<'_, Self>,
        theta: f64,
        phi: f64,
        a: usize,
        b: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner
            .beam_splitter(theta, phi, a, b)
            .map_err(to_py_err)?;
        Ok(slf)
    }

    fn two_mode_squeeze(
        mut slf: PyRefMut<'_, Self>,
        r: f64,
        phi: f64,
        a: usize,
        b: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner
            .two_mode_squeeze(r, phi, a, b)
            .map_err(to_py_err)?;
        Ok(slf)
    }

    fn cx(mut slf: PyRefMut<'_, Self>, s: f64, a: usize, b: usize) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.cx(s, a, b).map_err(to_py_err)?;
        Ok(slf)
    }

    fn cz(mut slf: PyRefMut<'_, Self>, s: f64, a: usize, b: usize) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.cz(s, a, b).map_err(to_py_err)?;
        Ok(slf)
    }

    fn cross_kerr(
        mut slf: PyRefMut<'_, Self>,
        kappa: f64,
        a: usize,
        b: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.cross_kerr(kappa, a, b).map_err(to_py_err)?;
        Ok(slf)
    }

    fn passive<'py>(
        mut slf: PyRefMut<'py, Self>,
        modes: Vec<usize>,
        matrix: Matrix2C<'_>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let (mat, k) = matrix.to_square_vec()?;
        if k != modes.len() {
            return Err(PyValueError::new_err(format!(
                "passive block on {} modes got a {k}×{k} matrix",
                modes.len()
            )));
        }
        slf.inner.apply_passive(&modes, &mat).map_err(to_py_err)?;
        Ok(slf)
    }

    fn measure_photons(&mut self, mode: usize) -> PyResult<usize> {
        let Self { inner, rng } = self;
        inner.measure_photons(mode, rng).map_err(to_py_err)
    }

    fn measure_threshold(&mut self, mode: usize) -> PyResult<bool> {
        let Self { inner, rng } = self;
        inner.measure_threshold(mode, rng).map_err(to_py_err)
    }

    fn measure_homodyne(&mut self, phi: f64, mode: usize) -> PyResult<f64> {
        let Self { inner, rng } = self;
        inner.measure_homodyne(mode, phi, rng).map_err(to_py_err)
    }

    fn measure_heterodyne(&mut self, mode: usize) -> PyResult<(f64, f64)> {
        let Self { inner, rng } = self;
        inner.measure_heterodyne(mode, rng).map_err(to_py_err)
    }

    fn __repr__(&self) -> String {
        format!(
            "BosonicMps(n_modes={}, cutoff={}, max_bond_dim={}, trunc={:.2e})",
            self.inner.n_modes(),
            self.inner.cutoff(),
            self.inner.max_bond_dim(),
            self.inner.truncation_error()
        )
    }
}

/// Output of `plan_circuit`: selected CV backend, rationale, and parameters.
#[pyclass(name = "CvPlan", module = "ketrs", frozen)]
pub(crate) struct PyCvPlan {
    inner: CvPlan,
}

#[pymethods]
impl PyCvPlan {
    #[getter]
    fn backend(&self) -> &'static str {
        self.inner.backend.as_str()
    }

    #[getter]
    fn reason(&self) -> &str {
        &self.inner.reason
    }

    #[getter]
    fn cutoff(&self) -> Option<usize> {
        self.inner.cutoff
    }

    #[getter]
    fn max_bond(&self) -> Option<usize> {
        self.inner.max_bond
    }

    #[getter]
    fn window_width(&self) -> Option<usize> {
        self.inner.window_width
    }

    fn __repr__(&self) -> String {
        format!(
            "CvPlan(backend={:?}, cutoff={:?}, max_bond={:?}, window_width={:?}, reason={:?})",
            self.inner.backend.as_str(),
            self.inner.cutoff,
            self.inner.max_bond,
            self.inner.window_width,
            self.inner.reason
        )
    }
}

/// Choose the fastest appropriate CV backend for `circuit` (always overridable
/// via `prefer_backend`).
#[pyfunction]
#[pyo3(signature = (circuit, cutoff=None, max_bond=64, tolerance=1e-6, sampling=false, prefer_backend=None))]
fn plan_circuit(
    circuit: PyRef<'_, PyModeCircuit>,
    cutoff: Option<usize>,
    max_bond: usize,
    tolerance: f64,
    sampling: bool,
    prefer_backend: Option<&str>,
) -> PyResult<PyCvPlan> {
    let prefer = match prefer_backend {
        Some(name) => Some(CvBackend::parse(name).map_err(to_py_err)?),
        None => None,
    };
    let opts = PlanOptions {
        cutoff,
        max_bond,
        tolerance,
        sampling,
        prefer_backend: prefer,
    };
    Ok(PyCvPlan {
        inner: rust_plan_circuit(&circuit.inner, &opts),
    })
}

/// Cutoff Fock statevector resident in Vulkan GPU VRAM (double precision).
#[cfg(feature = "gpu")]
#[pyclass(name = "GpuFockState", module = "ketrs", unsendable)]
pub(crate) struct PyGpuFockState {
    inner: GpuFockState,
}

#[cfg(feature = "gpu")]
#[pymethods]
impl PyGpuFockState {
    /// Vacuum state `|0…0⟩` on `n_modes` modes with Fock cutoff `cutoff` in GPU VRAM.
    #[new]
    fn new(n_modes: usize, cutoff: usize) -> PyResult<Self> {
        let ctx = super::gpu::context()?;
        Ok(PyGpuFockState {
            inner: GpuFockState::vacuum(&ctx, n_modes, cutoff).map_err(to_py_err)?,
        })
    }

    /// Run a bound, measurement-free `ModeCircuit` completely on the GPU in one batch.
    #[staticmethod]
    fn from_circuit(
        py: Python<'_>,
        circuit: PyRef<'_, PyModeCircuit>,
        cutoff: usize,
    ) -> PyResult<Self> {
        let ctx = super::gpu::context()?;
        let circ = &circuit.inner;
        let inner = py
            .detach(|| GpuFockState::from_circuit(&ctx, circ, cutoff))
            .map_err(to_py_err)?;
        Ok(PyGpuFockState { inner })
    }

    #[getter]
    fn n_modes(&self) -> usize {
        self.inner.n_modes()
    }

    #[getter]
    fn cutoff(&self) -> usize {
        self.inner.cutoff()
    }

    fn amplitudes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<C64>>> {
        let v = self.inner.amplitudes().map_err(to_py_err)?;
        Ok(PyArray1::from_vec(py, v))
    }

    fn probabilities<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let v = self.inner.probabilities().map_err(to_py_err)?;
        Ok(PyArray1::from_vec(py, v))
    }

    fn norm_sqr(&self) -> PyResult<f64> {
        self.inner.norm_sqr().map_err(to_py_err)
    }

    fn truncation_error(&self) -> PyResult<f64> {
        self.inner.truncation_error().map_err(to_py_err)
    }

    fn mean_photons(&self, mode: usize) -> PyResult<f64> {
        self.inner.mean_photons(mode).map_err(to_py_err)
    }

    fn rotation(
        mut slf: PyRefMut<'_, Self>,
        phi: f64,
        mode: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.rotation(phi, mode).map_err(to_py_err)?;
        Ok(slf)
    }

    fn displace(
        mut slf: PyRefMut<'_, Self>,
        r: f64,
        phi: f64,
        mode: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.displace(r, phi, mode).map_err(to_py_err)?;
        Ok(slf)
    }

    fn squeeze(
        mut slf: PyRefMut<'_, Self>,
        r: f64,
        phi: f64,
        mode: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.squeeze(r, phi, mode).map_err(to_py_err)?;
        Ok(slf)
    }

    fn quadratic_phase(
        mut slf: PyRefMut<'_, Self>,
        s: f64,
        mode: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.quadratic_phase(s, mode).map_err(to_py_err)?;
        Ok(slf)
    }

    fn cubic_phase(
        mut slf: PyRefMut<'_, Self>,
        gamma: f64,
        mode: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.cubic_phase(gamma, mode).map_err(to_py_err)?;
        Ok(slf)
    }

    fn kerr(mut slf: PyRefMut<'_, Self>, kappa: f64, mode: usize) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.kerr(kappa, mode).map_err(to_py_err)?;
        Ok(slf)
    }

    fn beam_splitter(
        mut slf: PyRefMut<'_, Self>,
        theta: f64,
        phi: f64,
        a: usize,
        b: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner
            .beam_splitter(theta, phi, a, b)
            .map_err(to_py_err)?;
        Ok(slf)
    }

    fn two_mode_squeeze(
        mut slf: PyRefMut<'_, Self>,
        r: f64,
        phi: f64,
        a: usize,
        b: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner
            .two_mode_squeeze(r, phi, a, b)
            .map_err(to_py_err)?;
        Ok(slf)
    }

    fn cx(mut slf: PyRefMut<'_, Self>, s: f64, a: usize, b: usize) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.cx(s, a, b).map_err(to_py_err)?;
        Ok(slf)
    }

    fn cz(mut slf: PyRefMut<'_, Self>, s: f64, a: usize, b: usize) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.cz(s, a, b).map_err(to_py_err)?;
        Ok(slf)
    }

    fn cross_kerr(
        mut slf: PyRefMut<'_, Self>,
        kappa: f64,
        a: usize,
        b: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.cross_kerr(kappa, a, b).map_err(to_py_err)?;
        Ok(slf)
    }

    fn passive<'py>(
        mut slf: PyRefMut<'py, Self>,
        modes: Vec<usize>,
        matrix: Matrix2C<'_>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let (mat, k) = matrix.to_square_vec()?;
        if k != modes.len() {
            return Err(PyValueError::new_err(format!(
                "passive block on {} modes got a {k}×{k} matrix",
                modes.len()
            )));
        }
        slf.inner.apply_passive(&modes, &mat).map_err(to_py_err)?;
        Ok(slf)
    }

    fn __repr__(&self) -> String {
        format!(
            "GpuFockState(n_modes={}, cutoff={})",
            self.inner.n_modes(),
            self.inner.cutoff()
        )
    }
}

/// Exact double-precision permanent of a square matrix on the Vulkan GPU.
#[cfg(feature = "gpu")]
#[pyfunction]
fn permanent_gpu(py: Python<'_>, matrix: Matrix2C<'_>) -> PyResult<C64> {
    let ctx = super::gpu::context()?;
    let (a, n) = matrix.to_square_vec()?;
    py.detach(|| rust_permanent_gpu(&ctx, &a, n))
        .map_err(to_py_err)
}

/// Hybrid circuit acting on `n_qubits` qubits and `n_modes` qumodes.
#[pyclass(name = "HybridCircuit", module = "ketrs", from_py_object)]
#[derive(Clone)]
pub(crate) struct PyHybridCircuit {
    inner: HybridCircuit,
}

#[pymethods]
impl PyHybridCircuit {
    #[new]
    fn new(n_qubits: usize, n_modes: usize) -> Self {
        PyHybridCircuit {
            inner: HybridCircuit::new(n_qubits, n_modes),
        }
    }

    #[getter]
    fn n_qubits(&self) -> usize {
        self.inner.n_qubits()
    }

    #[getter]
    fn n_modes(&self) -> usize {
        self.inner.n_modes()
    }

    #[getter]
    fn n_results(&self) -> usize {
        self.inner.n_results()
    }

    #[getter]
    fn is_unitary(&self) -> bool {
        self.inner.is_unitary()
    }

    fn set_mode_input<'py>(
        mut slf: PyRefMut<'py, Self>,
        mode: usize,
        state: PyModeState,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.inner
            .set_mode_input(mode, state.inner)
            .map_err(to_py_err)?;
        Ok(slf)
    }

    fn h(mut slf: PyRefMut<'_, Self>, qubit: usize) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.h(qubit).map_err(to_py_err)?;
        Ok(slf)
    }

    fn x(mut slf: PyRefMut<'_, Self>, qubit: usize) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.x(qubit).map_err(to_py_err)?;
        Ok(slf)
    }

    fn y(mut slf: PyRefMut<'_, Self>, qubit: usize) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.y(qubit).map_err(to_py_err)?;
        Ok(slf)
    }

    fn z(mut slf: PyRefMut<'_, Self>, qubit: usize) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.z(qubit).map_err(to_py_err)?;
        Ok(slf)
    }

    fn s(mut slf: PyRefMut<'_, Self>, qubit: usize) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.s(qubit).map_err(to_py_err)?;
        Ok(slf)
    }

    fn sdg(mut slf: PyRefMut<'_, Self>, qubit: usize) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.sdg(qubit).map_err(to_py_err)?;
        Ok(slf)
    }

    fn t(mut slf: PyRefMut<'_, Self>, qubit: usize) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.t(qubit).map_err(to_py_err)?;
        Ok(slf)
    }

    fn tdg(mut slf: PyRefMut<'_, Self>, qubit: usize) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.tdg(qubit).map_err(to_py_err)?;
        Ok(slf)
    }

    fn p(mut slf: PyRefMut<'_, Self>, lam: f64, qubit: usize) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.p(lam, qubit).map_err(to_py_err)?;
        Ok(slf)
    }

    fn rx(mut slf: PyRefMut<'_, Self>, theta: f64, qubit: usize) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.rx(theta, qubit).map_err(to_py_err)?;
        Ok(slf)
    }

    fn ry(mut slf: PyRefMut<'_, Self>, theta: f64, qubit: usize) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.ry(theta, qubit).map_err(to_py_err)?;
        Ok(slf)
    }

    fn rz(mut slf: PyRefMut<'_, Self>, theta: f64, qubit: usize) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.rz(theta, qubit).map_err(to_py_err)?;
        Ok(slf)
    }

    fn cx_qubit(
        mut slf: PyRefMut<'_, Self>,
        control: usize,
        target: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.cx_qubit(control, target).map_err(to_py_err)?;
        Ok(slf)
    }

    fn cz_qubit(mut slf: PyRefMut<'_, Self>, a: usize, b: usize) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.cz_qubit(a, b).map_err(to_py_err)?;
        Ok(slf)
    }

    fn swap_qubit(mut slf: PyRefMut<'_, Self>, a: usize, b: usize) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.swap_qubit(a, b).map_err(to_py_err)?;
        Ok(slf)
    }

    fn rotation(
        mut slf: PyRefMut<'_, Self>,
        phi: f64,
        mode: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.rotation(phi, mode).map_err(to_py_err)?;
        Ok(slf)
    }

    fn displace(
        mut slf: PyRefMut<'_, Self>,
        r: f64,
        phi: f64,
        mode: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.displace(r, phi, mode).map_err(to_py_err)?;
        Ok(slf)
    }

    fn squeeze(
        mut slf: PyRefMut<'_, Self>,
        r: f64,
        phi: f64,
        mode: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.squeeze(r, phi, mode).map_err(to_py_err)?;
        Ok(slf)
    }

    fn quadratic_phase(
        mut slf: PyRefMut<'_, Self>,
        s: f64,
        mode: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.quadratic_phase(s, mode).map_err(to_py_err)?;
        Ok(slf)
    }

    fn cubic_phase(
        mut slf: PyRefMut<'_, Self>,
        gamma: f64,
        mode: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.cubic_phase(gamma, mode).map_err(to_py_err)?;
        Ok(slf)
    }

    fn kerr(mut slf: PyRefMut<'_, Self>, kappa: f64, mode: usize) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.kerr(kappa, mode).map_err(to_py_err)?;
        Ok(slf)
    }

    fn snap(
        mut slf: PyRefMut<'_, Self>,
        phases: Vec<f64>,
        mode: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.snap(phases, mode).map_err(to_py_err)?;
        Ok(slf)
    }

    fn beam_splitter(
        mut slf: PyRefMut<'_, Self>,
        theta: f64,
        phi: f64,
        a: usize,
        b: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner
            .beam_splitter(theta, phi, a, b)
            .map_err(to_py_err)?;
        Ok(slf)
    }

    fn two_mode_squeeze(
        mut slf: PyRefMut<'_, Self>,
        r: f64,
        phi: f64,
        a: usize,
        b: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner
            .two_mode_squeeze(r, phi, a, b)
            .map_err(to_py_err)?;
        Ok(slf)
    }

    fn cx_mode(
        mut slf: PyRefMut<'_, Self>,
        s: f64,
        a: usize,
        b: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.cx_mode(s, a, b).map_err(to_py_err)?;
        Ok(slf)
    }

    fn cz_mode(
        mut slf: PyRefMut<'_, Self>,
        s: f64,
        a: usize,
        b: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.cz_mode(s, a, b).map_err(to_py_err)?;
        Ok(slf)
    }

    fn cross_kerr(
        mut slf: PyRefMut<'_, Self>,
        kappa: f64,
        a: usize,
        b: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.cross_kerr(kappa, a, b).map_err(to_py_err)?;
        Ok(slf)
    }

    fn passive<'py>(
        mut slf: PyRefMut<'py, Self>,
        modes: Vec<usize>,
        matrix: Matrix2C<'_>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let (mat, k) = matrix.to_square_vec()?;
        if k != modes.len() {
            return Err(PyValueError::new_err(format!(
                "passive block on {} modes got a {k}×{k} matrix",
                modes.len()
            )));
        }
        slf.inner.passive(modes, mat).map_err(to_py_err)?;
        Ok(slf)
    }

    fn conditional_displace(
        mut slf: PyRefMut<'_, Self>,
        r: f64,
        phi: f64,
        qubit: usize,
        mode: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner
            .conditional_displace(r, phi, qubit, mode)
            .map_err(to_py_err)?;
        Ok(slf)
    }

    fn conditional_rotation(
        mut slf: PyRefMut<'_, Self>,
        theta: f64,
        qubit: usize,
        mode: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner
            .conditional_rotation(theta, qubit, mode)
            .map_err(to_py_err)?;
        Ok(slf)
    }

    fn conditional_squeeze(
        mut slf: PyRefMut<'_, Self>,
        r: f64,
        phi: f64,
        qubit: usize,
        mode: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner
            .conditional_squeeze(r, phi, qubit, mode)
            .map_err(to_py_err)?;
        Ok(slf)
    }

    fn conditional_snap(
        mut slf: PyRefMut<'_, Self>,
        phases: Vec<f64>,
        qubit: usize,
        mode: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner
            .conditional_snap(phases, qubit, mode)
            .map_err(to_py_err)?;
        Ok(slf)
    }

    fn conditional_beam_splitter(
        mut slf: PyRefMut<'_, Self>,
        theta: f64,
        phi: f64,
        qubit: usize,
        a: usize,
        b: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner
            .conditional_beam_splitter(theta, phi, qubit, a, b)
            .map_err(to_py_err)?;
        Ok(slf)
    }

    #[pyo3(signature = (theta, phi, qubit, mode))]
    fn jaynes_cummings(
        mut slf: PyRefMut<'_, Self>,
        theta: f64,
        phi: f64,
        qubit: usize,
        mode: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner
            .jaynes_cummings(theta, phi, qubit, mode)
            .map_err(to_py_err)?;
        Ok(slf)
    }

    #[pyo3(signature = (theta, phi, qubit, mode))]
    fn anti_jaynes_cummings(
        mut slf: PyRefMut<'_, Self>,
        theta: f64,
        phi: f64,
        qubit: usize,
        mode: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner
            .anti_jaynes_cummings(theta, phi, qubit, mode)
            .map_err(to_py_err)?;
        Ok(slf)
    }

    fn measure_qubit(mut slf: PyRefMut<'_, Self>, qubit: usize) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.measure_qubit(qubit).map_err(to_py_err)?;
        Ok(slf)
    }

    fn measure_photons(mut slf: PyRefMut<'_, Self>, mode: usize) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.measure_photons(mode).map_err(to_py_err)?;
        Ok(slf)
    }

    fn measure_threshold(mut slf: PyRefMut<'_, Self>, mode: usize) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.measure_threshold(mode).map_err(to_py_err)?;
        Ok(slf)
    }

    fn measure_homodyne(
        mut slf: PyRefMut<'_, Self>,
        phi: f64,
        mode: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.measure_homodyne(phi, mode).map_err(to_py_err)?;
        Ok(slf)
    }

    fn __repr__(&self) -> String {
        format!(
            "HybridCircuit(n_qubits={}, n_modes={}, ops={})",
            self.inner.n_qubits(),
            self.inner.n_modes(),
            self.inner.ops().len()
        )
    }
}

/// Mixed-radix statevector simulator on `n_qubits` qubits and `n_modes` qumodes.
#[pyclass(name = "HybridState", module = "ketrs")]
pub(crate) struct PyHybridState {
    inner: HybridState,
    rng: StdRng,
}

#[pymethods]
impl PyHybridState {
    #[new]
    #[pyo3(signature = (n_qubits, n_modes, cutoff, seed=None))]
    fn new(n_qubits: usize, n_modes: usize, cutoff: usize, seed: Option<u64>) -> PyResult<Self> {
        Ok(PyHybridState {
            inner: HybridState::new(n_qubits, n_modes, cutoff).map_err(to_py_err)?,
            rng: rng_from_seed(seed),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (circuit, cutoff, seed=None))]
    fn from_circuit(
        py: Python<'_>,
        circuit: PyRef<'_, PyHybridCircuit>,
        cutoff: usize,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let circ = &circuit.inner;
        let inner = py
            .detach(|| HybridState::from_circuit(circ, cutoff))
            .map_err(to_py_err)?;
        Ok(PyHybridState {
            inner,
            rng: rng_from_seed(seed),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (circuit, cutoff, seed=None))]
    fn run<'py>(
        py: Python<'py>,
        circuit: PyRef<'_, PyHybridCircuit>,
        cutoff: usize,
        seed: Option<u64>,
    ) -> PyResult<(Self, Bound<'py, PyArray1<f64>>)> {
        let circ = &circuit.inner;
        let mut rng = rng_from_seed(seed);
        let (inner, results) = py
            .detach(|| HybridState::run(circ, cutoff, &mut rng))
            .map_err(to_py_err)?;
        Ok((
            PyHybridState { inner, rng },
            PyArray1::from_vec(py, results),
        ))
    }

    #[staticmethod]
    #[pyo3(signature = (circuit, cutoff, shots, seed=None))]
    fn sample_circuit<'py>(
        py: Python<'py>,
        circuit: PyRef<'_, PyHybridCircuit>,
        cutoff: usize,
        shots: usize,
        seed: Option<u64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let circ = &circuit.inner;
        let n_res = circ.n_results();
        let mut rng = rng_from_seed(seed);
        let rows = py
            .detach(|| HybridState::sample(circ, cutoff, shots, &mut rng))
            .map_err(to_py_err)?;
        let mut flat = Vec::with_capacity(shots * n_res);
        for row in rows {
            flat.extend_from_slice(&row);
        }
        PyArray1::from_vec(py, flat).reshape([shots, n_res])
    }

    #[getter]
    fn n_qubits(&self) -> usize {
        self.inner.n_qubits()
    }

    #[getter]
    fn n_modes(&self) -> usize {
        self.inner.n_modes()
    }

    #[getter]
    fn cutoff(&self) -> usize {
        self.inner.cutoff()
    }

    fn amplitudes<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<C64>> {
        PyArray1::from_slice(py, self.inner.amplitudes())
    }

    fn probabilities<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.probabilities())
    }

    fn amplitude(&self, qubit_bits: usize, pattern: Vec<usize>) -> PyResult<C64> {
        self.inner
            .amplitude(qubit_bits, &pattern)
            .map_err(to_py_err)
    }

    fn probability(&self, qubit_bits: usize, pattern: Vec<usize>) -> PyResult<f64> {
        self.inner
            .probability(qubit_bits, &pattern)
            .map_err(to_py_err)
    }

    fn norm_sqr(&self) -> f64 {
        self.inner.norm_sqr()
    }

    fn truncation_error(&self) -> f64 {
        self.inner.truncation_error()
    }

    fn qubit_prob_one(&self, qubit: usize) -> PyResult<f64> {
        self.inner.qubit_prob_one(qubit).map_err(to_py_err)
    }

    fn qubit_z(&self, qubit: usize) -> PyResult<f64> {
        self.inner.qubit_z(qubit).map_err(to_py_err)
    }

    fn qubit_x(&self, qubit: usize) -> PyResult<f64> {
        self.inner.qubit_x(qubit).map_err(to_py_err)
    }

    fn qubit_y(&self, qubit: usize) -> PyResult<f64> {
        self.inner.qubit_y(qubit).map_err(to_py_err)
    }

    fn mean_photons(&self, mode: usize) -> PyResult<f64> {
        self.inner.mean_photons(mode).map_err(to_py_err)
    }

    fn mean_x(&self, mode: usize) -> PyResult<f64> {
        self.inner.mean_x(mode).map_err(to_py_err)
    }

    fn mean_p(&self, mode: usize) -> PyResult<f64> {
        self.inner.mean_p(mode).map_err(to_py_err)
    }

    fn h(mut slf: PyRefMut<'_, Self>, qubit: usize) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.h(qubit).map_err(to_py_err)?;
        Ok(slf)
    }

    fn x(mut slf: PyRefMut<'_, Self>, qubit: usize) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.x(qubit).map_err(to_py_err)?;
        Ok(slf)
    }

    fn y(mut slf: PyRefMut<'_, Self>, qubit: usize) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.y(qubit).map_err(to_py_err)?;
        Ok(slf)
    }

    fn z(mut slf: PyRefMut<'_, Self>, qubit: usize) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.z(qubit).map_err(to_py_err)?;
        Ok(slf)
    }

    fn s(mut slf: PyRefMut<'_, Self>, qubit: usize) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.s(qubit).map_err(to_py_err)?;
        Ok(slf)
    }

    fn sdg(mut slf: PyRefMut<'_, Self>, qubit: usize) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.sdg(qubit).map_err(to_py_err)?;
        Ok(slf)
    }

    fn t(mut slf: PyRefMut<'_, Self>, qubit: usize) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.t(qubit).map_err(to_py_err)?;
        Ok(slf)
    }

    fn tdg(mut slf: PyRefMut<'_, Self>, qubit: usize) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.tdg(qubit).map_err(to_py_err)?;
        Ok(slf)
    }

    fn p(mut slf: PyRefMut<'_, Self>, lam: f64, qubit: usize) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.p(lam, qubit).map_err(to_py_err)?;
        Ok(slf)
    }

    fn rx(mut slf: PyRefMut<'_, Self>, theta: f64, qubit: usize) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.rx(theta, qubit).map_err(to_py_err)?;
        Ok(slf)
    }

    fn ry(mut slf: PyRefMut<'_, Self>, theta: f64, qubit: usize) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.ry(theta, qubit).map_err(to_py_err)?;
        Ok(slf)
    }

    fn rz(mut slf: PyRefMut<'_, Self>, theta: f64, qubit: usize) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.rz(theta, qubit).map_err(to_py_err)?;
        Ok(slf)
    }

    fn cx_qubit(
        mut slf: PyRefMut<'_, Self>,
        control: usize,
        target: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.cx_qubit(control, target).map_err(to_py_err)?;
        Ok(slf)
    }

    fn cz_qubit(mut slf: PyRefMut<'_, Self>, a: usize, b: usize) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.cz_qubit(a, b).map_err(to_py_err)?;
        Ok(slf)
    }

    fn swap_qubit(mut slf: PyRefMut<'_, Self>, a: usize, b: usize) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.swap_qubit(a, b).map_err(to_py_err)?;
        Ok(slf)
    }

    fn rotation(
        mut slf: PyRefMut<'_, Self>,
        phi: f64,
        mode: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.rotation(phi, mode).map_err(to_py_err)?;
        Ok(slf)
    }

    fn displace(
        mut slf: PyRefMut<'_, Self>,
        r: f64,
        phi: f64,
        mode: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.displace(r, phi, mode).map_err(to_py_err)?;
        Ok(slf)
    }

    fn squeeze(
        mut slf: PyRefMut<'_, Self>,
        r: f64,
        phi: f64,
        mode: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.squeeze(r, phi, mode).map_err(to_py_err)?;
        Ok(slf)
    }

    fn quadratic_phase(
        mut slf: PyRefMut<'_, Self>,
        s: f64,
        mode: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.quadratic_phase(s, mode).map_err(to_py_err)?;
        Ok(slf)
    }

    fn cubic_phase(
        mut slf: PyRefMut<'_, Self>,
        gamma: f64,
        mode: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.cubic_phase(gamma, mode).map_err(to_py_err)?;
        Ok(slf)
    }

    fn kerr(mut slf: PyRefMut<'_, Self>, kappa: f64, mode: usize) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.kerr(kappa, mode).map_err(to_py_err)?;
        Ok(slf)
    }

    fn snap(
        mut slf: PyRefMut<'_, Self>,
        phases: Vec<f64>,
        mode: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.snap(&phases, mode).map_err(to_py_err)?;
        Ok(slf)
    }

    fn beam_splitter(
        mut slf: PyRefMut<'_, Self>,
        theta: f64,
        phi: f64,
        a: usize,
        b: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner
            .beam_splitter(theta, phi, a, b)
            .map_err(to_py_err)?;
        Ok(slf)
    }

    fn two_mode_squeeze(
        mut slf: PyRefMut<'_, Self>,
        r: f64,
        phi: f64,
        a: usize,
        b: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner
            .two_mode_squeeze(r, phi, a, b)
            .map_err(to_py_err)?;
        Ok(slf)
    }

    fn cx_mode(
        mut slf: PyRefMut<'_, Self>,
        s: f64,
        a: usize,
        b: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.cx_mode(s, a, b).map_err(to_py_err)?;
        Ok(slf)
    }

    fn cz_mode(
        mut slf: PyRefMut<'_, Self>,
        s: f64,
        a: usize,
        b: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.cz_mode(s, a, b).map_err(to_py_err)?;
        Ok(slf)
    }

    fn cross_kerr(
        mut slf: PyRefMut<'_, Self>,
        kappa: f64,
        a: usize,
        b: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.cross_kerr(kappa, a, b).map_err(to_py_err)?;
        Ok(slf)
    }

    fn passive<'py>(
        mut slf: PyRefMut<'py, Self>,
        modes: Vec<usize>,
        matrix: Matrix2C<'_>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let (mat, k) = matrix.to_square_vec()?;
        if k != modes.len() {
            return Err(PyValueError::new_err(format!(
                "passive block on {} modes got a {k}×{k} matrix",
                modes.len()
            )));
        }
        slf.inner.apply_passive(&modes, &mat).map_err(to_py_err)?;
        Ok(slf)
    }

    fn conditional_displace(
        mut slf: PyRefMut<'_, Self>,
        r: f64,
        phi: f64,
        qubit: usize,
        mode: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner
            .conditional_displace(r, phi, qubit, mode)
            .map_err(to_py_err)?;
        Ok(slf)
    }

    fn conditional_rotation(
        mut slf: PyRefMut<'_, Self>,
        theta: f64,
        qubit: usize,
        mode: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner
            .conditional_rotation(theta, qubit, mode)
            .map_err(to_py_err)?;
        Ok(slf)
    }

    fn conditional_squeeze(
        mut slf: PyRefMut<'_, Self>,
        r: f64,
        phi: f64,
        qubit: usize,
        mode: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner
            .conditional_squeeze(r, phi, qubit, mode)
            .map_err(to_py_err)?;
        Ok(slf)
    }

    fn conditional_snap(
        mut slf: PyRefMut<'_, Self>,
        phases: Vec<f64>,
        qubit: usize,
        mode: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner
            .conditional_snap(&phases, qubit, mode)
            .map_err(to_py_err)?;
        Ok(slf)
    }

    fn conditional_beam_splitter(
        mut slf: PyRefMut<'_, Self>,
        theta: f64,
        phi: f64,
        qubit: usize,
        a: usize,
        b: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner
            .conditional_beam_splitter(theta, phi, qubit, a, b)
            .map_err(to_py_err)?;
        Ok(slf)
    }

    fn jaynes_cummings(
        mut slf: PyRefMut<'_, Self>,
        theta: f64,
        phi: f64,
        qubit: usize,
        mode: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner
            .jaynes_cummings(theta, phi, qubit, mode)
            .map_err(to_py_err)?;
        Ok(slf)
    }

    fn anti_jaynes_cummings(
        mut slf: PyRefMut<'_, Self>,
        theta: f64,
        phi: f64,
        qubit: usize,
        mode: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner
            .anti_jaynes_cummings(theta, phi, qubit, mode)
            .map_err(to_py_err)?;
        Ok(slf)
    }

    fn measure_qubit(&mut self, qubit: usize) -> PyResult<bool> {
        let Self { inner, rng } = self;
        inner.measure_qubit(qubit, rng).map_err(to_py_err)
    }

    fn postselect_qubit(&mut self, qubit: usize, outcome: bool) -> PyResult<f64> {
        self.inner
            .postselect_qubit(qubit, outcome)
            .map_err(to_py_err)
    }

    fn measure_photons(&mut self, mode: usize) -> PyResult<usize> {
        let Self { inner, rng } = self;
        inner.measure_photons(mode, rng).map_err(to_py_err)
    }

    fn measure_threshold(&mut self, mode: usize) -> PyResult<bool> {
        let Self { inner, rng } = self;
        inner.measure_threshold(mode, rng).map_err(to_py_err)
    }

    fn measure_homodyne(&mut self, phi: f64, mode: usize) -> PyResult<f64> {
        let Self { inner, rng } = self;
        inner.measure_homodyne(phi, mode, rng).map_err(to_py_err)
    }

    fn __repr__(&self) -> String {
        format!(
            "HybridState(n_qubits={}, n_modes={}, cutoff={})",
            self.inner.n_qubits(),
            self.inner.n_modes(),
            self.inner.cutoff()
        )
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMeasured>()?;
    m.add_class::<PyModeState>()?;
    m.add_class::<PyModeCircuit>()?;
    m.add_class::<PyCvChannel>()?;
    m.add_class::<PyGaussianState>()?;
    m.add_class::<PyGaussianMixture>()?;
    m.add_class::<PyFixedNFockState>()?;
    m.add_class::<PyFockState>()?;
    m.add_class::<PyFockDensityMatrix>()?;
    m.add_class::<PyBosonicMps>()?;
    m.add_class::<PyCvPlan>()?;
    m.add_class::<PyHybridCircuit>()?;
    m.add_class::<PyHybridState>()?;
    m.add_class::<PyWindowSampler>()?;
    #[cfg(feature = "gpu")]
    {
        m.add_class::<PyGpuFockState>()?;
        m.add_function(wrap_pyfunction!(permanent_gpu, m)?)?;
    }
    m.add_function(wrap_pyfunction!(plan_circuit, m)?)?;
    m.add_function(wrap_pyfunction!(hafnian, m)?)?;
    m.add_function(wrap_pyfunction!(loop_hafnian, m)?)?;
    m.add_function(wrap_pyfunction!(hafnian_repeated, m)?)?;
    m.add_function(wrap_pyfunction!(torontonian, m)?)?;
    m.add_function(wrap_pyfunction!(permanent, m)?)?;
    m.add_function(wrap_pyfunction!(boson_sample, m)?)?;
    m.add_function(wrap_pyfunction!(sample_gbs_pnr, m)?)?;
    m.add_function(wrap_pyfunction!(sample_gbs_threshold, m)?)?;
    Ok(())
}
