//! Python bindings for matrix product states.

use super::{Observable, PyCircuit, PyStatevector, State, clbit_counts, rng_from_seed, to_py_err};
use ketrs::{Complex64 as C64, DmrgOptions, Mps, MpsOptions, Statevector};
use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::rngs::StdRng;

/// A matrix product state: memory and time grow with the bond dimension
/// rather than 2ⁿ, so weakly entangled states of hundreds of qubits are
/// cheap. Each bond keeps at most `max_bond` singular values, dropping any
/// whose discarded weight stays below `cutoff`; `truncation_error` reports
/// the total discarded weight.
#[pyclass(name = "MPS", module = "ketrs")]
struct PyMps {
    inner: Mps,
    rng: StdRng,
}

fn options(max_bond: usize, cutoff: f64) -> PyResult<MpsOptions> {
    if max_bond == 0 {
        return Err(PyValueError::new_err("max_bond must be at least 1"));
    }
    Ok(MpsOptions { max_bond, cutoff })
}

/// Bits per qubit from a Qiskit-style bitstring (qubit 0 rightmost) or an
/// integer index.
#[derive(FromPyObject)]
enum BasisState {
    Bitstring(String),
    Index(usize),
}

impl BasisState {
    fn bits(&self, n: usize) -> PyResult<Vec<bool>> {
        match self {
            BasisState::Bitstring(s) => {
                if s.len() != n || !s.chars().all(|c| c == '0' || c == '1') {
                    return Err(PyValueError::new_err(format!("expected {n} binary digits")));
                }
                Ok(s.chars().rev().map(|c| c == '1').collect())
            }
            BasisState::Index(i) => {
                if n < usize::BITS as usize && *i >> n != 0 {
                    return Err(PyValueError::new_err("index out of range"));
                }
                Ok((0..n)
                    .map(|q| q < usize::BITS as usize && i >> q & 1 == 1)
                    .collect())
            }
        }
    }
}

#[pymethods]
impl PyMps {
    /// `|0…0⟩` on `n_qubits` qubits.
    #[new]
    #[pyo3(signature = (n_qubits, max_bond=64, cutoff=1e-14, seed=None))]
    fn new(n_qubits: usize, max_bond: usize, cutoff: f64, seed: Option<u64>) -> PyResult<Self> {
        if n_qubits == 0 {
            return Err(PyValueError::new_err("an MPS needs at least one qubit"));
        }
        Ok(PyMps {
            inner: Mps::zero(n_qubits, options(max_bond, cutoff)?),
            rng: rng_from_seed(seed),
        })
    }

    /// Run `circuit` on `|0…0⟩`.
    #[staticmethod]
    #[pyo3(signature = (circuit, max_bond=64, cutoff=1e-14, seed=None))]
    fn from_circuit(
        py: Python<'_>,
        circuit: PyRef<'_, PyCircuit>,
        max_bond: usize,
        cutoff: f64,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let opts = options(max_bond, cutoff)?;
        let circ = &circuit.inner;
        Ok(PyMps {
            inner: py
                .detach(|| Mps::from_circuit(circ, opts))
                .map_err(to_py_err)?,
            rng: rng_from_seed(seed),
        })
    }

    /// A random normalised MPS with bonds up to `bond`: a generic starting
    /// point for `dmrg`.
    #[staticmethod]
    #[pyo3(signature = (n_qubits, bond=8, seed=0, max_bond=64, cutoff=1e-14))]
    fn random(
        n_qubits: usize,
        bond: usize,
        seed: u64,
        max_bond: usize,
        cutoff: f64,
    ) -> PyResult<Self> {
        if n_qubits == 0 || bond == 0 {
            return Err(PyValueError::new_err("need at least one qubit and bond"));
        }
        Ok(PyMps {
            inner: Mps::random(n_qubits, bond, seed, options(max_bond, cutoff)?),
            rng: rng_from_seed(Some(seed)),
        })
    }

    /// Find the ground state of `hamiltonian` by two-site DMRG, starting
    /// from this state and replacing it. Returns
    /// `(energy, sweep_energies, converged)`.
    ///
    /// DMRG keeps the symmetries the start shares with H; start from
    /// `MPS.random` for the true ground state, or from a state in the sector
    /// you want.
    #[pyo3(signature = (hamiltonian, max_bond=64, cutoff=1e-12, max_sweeps=20, tolerance=1e-10))]
    fn dmrg(
        &mut self,
        py: Python<'_>,
        hamiltonian: Observable,
        max_bond: usize,
        cutoff: f64,
        max_sweeps: usize,
        tolerance: f64,
    ) -> PyResult<(f64, Vec<f64>, bool)> {
        options(max_bond, cutoff)?;
        let sum = hamiltonian.into_sum()?;
        let opts = DmrgOptions {
            max_bond,
            cutoff,
            max_sweeps,
            tolerance,
            ..DmrgOptions::default()
        };
        let mps = &mut self.inner;
        let result = py.detach(|| mps.dmrg(&sum, &opts)).map_err(to_py_err)?;
        Ok((result.energy, result.sweep_energies, result.converged))
    }

    #[getter]
    fn n_qubits(&self) -> usize {
        self.inner.n_qubits()
    }

    /// Bond dimension between each pair of neighbouring qubits.
    #[getter]
    fn bond_dimensions(&self) -> Vec<usize> {
        self.inner.bond_dimensions()
    }

    /// Total discarded weight from truncation: roughly the lost fidelity.
    #[getter]
    fn truncation_error(&self) -> f64 {
        self.inner.truncation_error()
    }

    #[getter]
    fn max_bond(&self) -> usize {
        self.inner.options().max_bond
    }

    #[setter]
    fn set_max_bond(&mut self, max_bond: usize) -> PyResult<()> {
        let cutoff = self.inner.options().cutoff;
        self.inner.set_options(options(max_bond, cutoff)?);
        Ok(())
    }

    #[getter]
    fn cutoff(&self) -> f64 {
        self.inner.options().cutoff
    }

    #[setter]
    fn set_cutoff(&mut self, cutoff: f64) -> PyResult<()> {
        let max_bond = self.inner.options().max_bond;
        self.inner.set_options(options(max_bond, cutoff)?);
        Ok(())
    }

    fn norm_sqr(&self) -> f64 {
        self.inner.norm_sqr()
    }

    fn __repr__(&self) -> String {
        let max = self.inner.bond_dimensions().into_iter().max().unwrap_or(1);
        format!(
            "MPS(n_qubits={}, max bond in use={max}, truncation_error={:.1e})",
            self.inner.n_qubits(),
            self.inner.truncation_error()
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
        let (mps, circ) = (&mut self.inner, &circuit.inner);
        if circ.is_unitary() {
            py.detach(|| mps.run(circ)).map_err(to_py_err)?;
            return Ok(None);
        }
        let rng = &mut self.rng;
        let bits = py
            .detach(|| mps.run_trajectory(circ, rng))
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
        let (mps, circ) = (&self.inner, &circuit.inner);
        let rng = &mut self.rng;
        let counts = py
            .detach(|| mps.sample_circuit(circ, shots, rng))
            .map_err(to_py_err)?;
        clbit_counts(py, counts, circ.n_clbits())
    }

    /// TEBD: `steps` Trotter–Suzuki steps of the given order for exp(-iHt).
    #[pyo3(signature = (hamiltonian, time, steps=1, order=2))]
    fn tebd(
        &mut self,
        py: Python<'_>,
        hamiltonian: Observable,
        time: f64,
        steps: usize,
        order: usize,
    ) -> PyResult<()> {
        let sum = hamiltonian.into_sum()?;
        let mps = &mut self.inner;
        py.detach(|| mps.tebd(&sum, time, steps, order))
            .map_err(to_py_err)
    }

    /// Evolve by exp(-iHt) with `steps` steps of two-site TDVP, for
    /// Hamiltonians of any range (TEBD routes long-range terms with swaps).
    /// From a weakly entangled start such as a product state, accuracy is
    /// lost while bonds grow; take more steps, or start with more bond
    /// dimension.
    #[pyo3(signature = (hamiltonian, time, steps=1))]
    fn tdvp(
        &mut self,
        py: Python<'_>,
        hamiltonian: Observable,
        time: f64,
        steps: usize,
    ) -> PyResult<()> {
        let sum = hamiltonian.into_sum()?;
        let mps = &mut self.inner;
        py.detach(|| mps.tdvp(&sum, time, steps)).map_err(to_py_err)
    }

    /// `⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩`.
    fn expectation(&self, py: Python<'_>, observable: Observable) -> PyResult<f64> {
        let sum = observable.into_sum()?;
        let mps = &self.inner;
        py.detach(|| mps.expectation(&sum)).map_err(to_py_err)
    }

    /// The amplitude of a basis state, given as a Qiskit-style bitstring
    /// (qubit 0 rightmost) or an integer index.
    fn amplitude(&self, state: BasisState) -> PyResult<C64> {
        Ok(self.inner.amplitude(&state.bits(self.inner.n_qubits())?))
    }

    /// The full statevector; at most 30 qubits.
    fn to_statevector(&self) -> PyResult<PyStatevector> {
        if self.inner.n_qubits() > 30 {
            return Err(PyValueError::new_err("too many qubits to expand"));
        }
        let state = Statevector::from_amplitudes(self.inner.to_amplitudes()).map_err(to_py_err)?;
        Ok(PyStatevector::wrap(State::Complex(state), self.rng.clone()))
    }

    /// Draw `shots` basis-state indices; at most 64 qubits.
    fn sample<'py>(
        &mut self,
        py: Python<'py>,
        shots: usize,
    ) -> PyResult<Bound<'py, PyArray1<usize>>> {
        self.check_indexable()?;
        let samples = self.inner.sample(shots, &mut self.rng);
        Ok(PyArray1::from_vec(py, samples))
    }

    /// Counts keyed by bitstring, most significant qubit first.
    fn sample_counts<'py>(
        &mut self,
        py: Python<'py>,
        shots: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        self.check_indexable()?;
        let n = self.inner.n_qubits();
        let dict = PyDict::new(py);
        for (index, count) in self.inner.sample_counts(shots, &mut self.rng) {
            dict.set_item(format!("{index:0n$b}"), count)?;
        }
        Ok(dict)
    }
}

impl PyMps {
    fn check_indexable(&self) -> PyResult<()> {
        if self.inner.n_qubits() > usize::BITS as usize {
            return Err(PyValueError::new_err("sampling is limited to 64 qubits"));
        }
        Ok(())
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMps>()
}
