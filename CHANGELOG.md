# Changelog

## 0.1.0 (unreleased)

First release.

### Engines

- `Statevector` and `RealStatevector`, in double and single precision:
  cache-blocked, parallel kernels with AVX2/FMA paths chosen at run time,
  selective gate fusion, and tiles over any set of qubits.
- `SectorStatevector`: fixed Hamming weights on any qubit subsets.
- `DensityMatrix`: Kraus channels and noise models, run through the
  statevector kernels as superoperators.
- `Mps`: parallel TEBD in Hastings' form, two-site TDVP on an MPO with
  subspace expansion while bonds grow, DMRG, sampling.
- `GpuStatevector` (feature `gpu`): Vulkan compute in double or single
  precision, shared-memory tiles, gradients and Krylov evolution on the GPU.

### Circuits

- Qiskit's standard gates, arbitrary one- and two-qubit unitaries, Pauli
  rotations of any weight, and gates and swaps with any number of
  controls, on every engine. `Circuit::control` controls a whole circuit.
- Linear parameters, `bind`, and adjoint gradients for any loss.
- Measurement, reset, `if`/`else` and `while` on classical expressions,
  through one trajectory driver on every engine.
- Trotter–Suzuki circuits of any order, with commuting terms packed into
  layers.

### Hamiltonians

- Pauli sums applied without forming a matrix: expectation values, `H|ψ⟩`,
  exact `exp(-iHt)` by Krylov steps, lowest eigenstates by Lanczos.

### Noise

- Noisy statevector trajectories (`run_noisy_trajectory`, `sample_noisy`),
  with Pauli mixtures batched through the tiled kernels, and readout error.

### Interoperability

- OpenQASM 3 and 2 import (`qasm::parse`): gate definitions, `def`
  subroutines, `inv`/`pow`/`ctrl(n)`/`negctrl(n)` modifiers, control flow.
- OpenQASM 3 export (`Circuit::to_qasm`) with standard gates only.
- Python bindings (`pip install ketrs`, module `ketrs`), with converters
  from Qiskit circuits, `SparsePauliOp`s and Aer noise models.

### Known limitations

- OpenQASM `break`, `continue`, `switch` and subroutines with return
  values are not supported.
- Aer noise models must apply the same errors on every qubit.
- The GPU engine needs a Vulkan 1.2 driver; double precision needs
  `shaderFloat64`.
