# Choosing an Engine

All six simulation engines in `ketrs` share the same `Circuit` and `PauliSum` representations, making it easy to prototype on a small CPU statevector and scale up later.

## Quick Decision Table

| Workload | Recommended Engine | Typical Scale | Key Strengths |
|---|---|---|---|
| **General exact circuits (CPU)** | `Statevector` | Up to ~28 qubits | Parallel, cache-blocked, gate-fused; supports `complex128`, `float64`, `complex64`, `float32`. |
| **Large exact circuits (GPU)** | `GpuStatevector` | 18–31 qubits | Any Vulkan 1.2 GPU (NVIDIA, AMD, Intel, Apple Silicon via MoltenVK); gates, expectations, sampling, gradients, and `exp(-iHt)` on GPU. |
| **Conserved particle / spin number** | `SectorStatevector` | Sector-dependent | Stores only states matching fixed Hamming weights; rejects sector-breaking ops. |
| **Exact mixed states & noise** | `DensityMatrix` | Up to ~14 qubits | Exact Kraus channels and per-gate `NoiseModel`s (1.3–4.2× faster than Aer). |
| **Noisy circuits (>14 qubits)** | `Statevector` (`noise=...`) | Up to ~28 qubits | Quantum trajectories with per-gate Kraus channels and readout error. |
| **Weakly entangled open chains** | `MPS` | Tens to hundreds of qubits | TEBD, TDVP (arbitrary range), DMRG, conserved `U(1)`/`Zₙ` charges, optional Vulkan FP64 GPU. |
| **Weakly entangled closed rings** | `PeriodicMPS` | Tens to hundreds of qubits | Periodic-trace MPS with `(n-1, 0)` seam gates, PWE compressed DMRG, TDVP, TEBD, conserved charges, GPU. |

---

## Precision & Memory Trade-offs (`Statevector`)

`Statevector` (and `SectorStatevector`) accepts a `dtype` parameter that lets you trade unneeded complex/FP64 storage for large speedups and memory savings:

| `dtype` | Rust Type | Bytes / Amplitude | Memory (26q) | When to use |
|---|---|---:|---:|---|
| `"complex"` / `"complex128"` | `Statevector` | 16 B | 1 GiB | Default; any circuit, ~10⁻¹⁵ accuracy. |
| `"real"` / `"float64"` | `RealStatevector` | 8 B | 512 MiB | Circuits with only real gates (`H`, `X`, `Z`, `Ry`, `CX`, `CZ`, `SWAP`, `CCX`). Halves memory. |
| `"complex64"` | `Statevector32` | 8 B | 512 MiB | General circuits where ~10⁻⁶ accuracy suffices; ~1.65× faster at 22–26q. |
| `"float32"` | `RealStatevector32` | 4 B | 256 MiB | Real circuits at ~10⁻⁶ accuracy; ~4× faster at 22–26q. |

```python
import ketrs

circ = ketrs.Circuit(24)
for q in range(24):
    circ.ry(0.3, q)
for q in range(23):
    circ.cx(q, q + 1)

# Real single precision: 64 MiB instead of 256 MiB, ~4x faster
state = ketrs.Statevector.from_circuit(circ, dtype="float32")
```

---

## When to Switch from Statevectors to MPS

Full statevectors double in memory with every qubit:

- 20 qubits (`complex128`): 16 MiB
- 26 qubits (`complex128`): 1 GiB
- 30 qubits (`complex128`): 16 GiB

When your state has limited entanglement (ground states of 1D / quasi-1D Hamiltonians, shallow circuits, short-to-moderate time evolution), [`MPS`](../guides/mps.md) and [`PeriodicMPS`](../guides/mps.md#periodic-rings-periodicmps) scale linearly in the number of qubits as `O(n · χ²)` in memory instead of `O(2ⁿ)`, routinely handling hundreds of qubits. Adding conserved symmetries (`Charge` + `with_charges`) reduces memory and runtime by another order of magnitude or more on structured models like lattice gauge theories.
