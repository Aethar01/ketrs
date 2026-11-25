# Python API Reference

This reference documents the public Python API exported by `ketrs` and `ketrs.qiskit`.
For the Rust crate API, see [docs.rs/ketrs](https://docs.rs/ketrs).

## Modules & Sections

- **[Circuits & Observables](circuits.md):** `Circuit`, `Parameter`, `PauliSum`, `SparseOperator`, `from_qasm`, `qft`, `ghz`.
- **[Statevectors & GPU](statevectors.md):** `Statevector`, `GpuStatevector`, `SectorBasis`, `SectorStatevector`, `Eigenstate`, `lowest_eigenstates`, `gpu_device`.
- **[Noise & Density Matrices](noise.md):** `DensityMatrix`, `NoiseModel`, `Channel`.
- **[Matrix Product States](mps.md):** `MPS`, `PeriodicMPS`, `Charge`.
- **[Qiskit Interop](qiskit.md):** `from_qiskit`, `pauli_sum_from_qiskit`, `noise_model_from_aer`.
