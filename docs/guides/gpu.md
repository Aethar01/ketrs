# GPU Acceleration (Vulkan)

`ketrs` accelerates both exact statevectors (`GpuStatevector`) and MPS DMRG/TDVP sweeps (`mps.gpu = True`) using **Vulkan 1.2 compute shaders**. Because it uses Vulkan, it works across NVIDIA, AMD, Intel, and Apple Silicon (via MoltenVK) GPUs without installing CUDA or ROCm.

## Checking Your GPU

```python
import ketrs

print("GPU:", ketrs.gpu_device())
```

If this prints `None`, check that you have a Vulkan 1.2 driver installed (on macOS, prebuilt wheels bundle MoltenVK automatically; Rust/source builds use `brew install molten-vk` or the Vulkan SDK). Note that Apple Silicon GPUs lack native 64-bit floats, so `GpuStatevector` runs in single precision (`complex64`) and `mps.gpu = True` (which requires FP64) is unavailable on Apple Silicon.

---

## `GpuStatevector`

`GpuStatevector` keeps the amplitudes in GPU memory (split across multiple buffers above ~27 qubits, up to 31 qubits memory permitting). Gates are scheduled into shared-memory tiles over 10 qubits (11 on NVIDIA), reordering commuting gates so most of a circuit requires one memory pass per tile rather than per gate.

```python
import ketrs

circ = ketrs.qft(24)

# Automatically picks FP64 if supported, otherwise FP32
gpu_state = ketrs.GpuStatevector.from_circuit(circ)
print(gpu_state)  # GpuStatevector(n_qubits=24, dtype=complex128, device=...)

# Keep work on the GPU: expectations, sampling, evolution, and gradients
z0 = gpu_state.expectation("I" * 23 + "Z")
counts = gpu_state.sample_counts(1000)

# Download to CPU only when you actually need the full amplitude array
cpu_state = gpu_state.to_statevector()
```

### Single Precision (`complex64`) vs Double Precision (`complex128`)

Consumer GPUs typically run FP32 at 16–64× the rate of FP64. If ~10⁻⁶ accuracy is sufficient (for example in circuit sampling, VQE sweeps, or exploratory simulations), explicitly select `"complex64"`:

```python
gpu_f32 = ketrs.GpuStatevector.from_circuit(circ, dtype="complex64")
```

- Halves VRAM usage (29 qubits in `complex64` fits in 4 GiB; 30 qubits in 8 GiB).
- Runs 2.8–5.8× faster on consumer GPUs.

### GPU Hamiltonians & Adjoint Gradients

Exact Krylov time evolution (`evolve`), Hamiltonian application (`apply_pauli_sum`), and adjoint differentiation (`expectation_gradient`, `adjoint_gradient`) all run natively on the GPU:

```python
import ketrs

n = 20
H = ketrs.PauliSum([
    ("I" * (n - 2 - i) + "ZZ" + "I" * i, -1.0) for i in range(n - 1)
] + [
    ("I" * (n - 1 - i) + "X" + "I" * i, -0.5) for i in range(n)
])

state = ketrs.GpuStatevector(n, dtype="complex64")
state.evolve(H, time=1.0)
print("⟨H⟩:", state.expectation(H))
```

!!! tip "Avoid unnecessary host downloads"
    Computing `gpu_state.expectation(...)` or `gpu_state.sample_counts(...)` on the GPU is 14–15× faster than downloading the full statevector over PCIe first.

---

## GPU-Accelerated MPS & PeriodicMPS Sweeps

For large bond dimensions (`χ >= 64`), you can offload the effective-Hamiltonian contractions, Krylov/Lanczos recurrences, and TDVP subspace-expansion QR factorizations to a Vulkan GPU with FP64 support:

```python
import ketrs

mps = ketrs.MPS.random(64, bond=8, seed=0, max_bond=64)
mps.gpu = True
# mps.dmrg(H, max_bond=64)
# mps.tdvp(H, time=0.5, steps=5)
```

`mps.gpu = True` also works with `PeriodicMPS` and with symmetric (`with_charges`) MPS. On symmetric MPS with small charge blocks, the CPU is often faster at moderate bonds (`χ <= 128`), while the GPU pulls ahead at larger bond dimensions (`χ >= 256–512`).
