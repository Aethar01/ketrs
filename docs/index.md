# ketrs

**A fast quantum computer simulator in Rust, with Python bindings, Vulkan GPU acceleration, and Qiskit / OpenQASM 3 support.**

`ketrs` gives you one clean API across qubit, continuous-variable (qumode), and qubit–qumode hybrid simulation engines — from exact CPU and Vulkan GPU statevectors to density matrices, symmetry sectors, Matrix Product States on hundreds of qubits, Gaussian/bosonic/Fock CV backends, and circuit-QED hybrids.

---

## Why ketrs?

- **Fast out of the box:** 2.2–9.1× faster than Qiskit Aer on CPU statevectors, 18–460× faster on mid-circuit measurement circuits, 6.3–32.5× faster on MPS TEBD, and faster than The Walrus, Piquasso, Perceval, QuTiP, and Bosonic Qiskit across CV and hybrid workloads.
- **Vendor-neutral GPU support:** runs on NVIDIA, AMD, Intel, and Apple Silicon GPUs through Vulkan 1.2 (and MoltenVK on macOS, up to 31 qubits, plus GPU-accelerated MPS sweeps, CV Fock statevectors, and permanents). No CUDA or ROCm install needed.
- **Scales from a few qubits to hundreds:** exact `Statevector` (`complex128`, `float64`, `complex64`, `float32`), `GpuStatevector`, `SectorStatevector`, `DensityMatrix`, `MPS`, and `PeriodicMPS` with DMRG, TDVP, TEBD, and conserved $\mathrm{U}(1)$ / $\mathbb{Z}_n$ charges.
- **Complete qumode & hybrid suite:** `ModeCircuit` with `GaussianState`, `GaussianMixture` (cat/GKP), `FixedNFockState` (SLOS), `FockState` / `GpuFockState`, `FockDensityMatrix`, `BosonicMps`, `WindowSampler`, exact GBS & boson samplers, automatic backend `plan_circuit`, and mixed-radix `HybridCircuit` / `HybridState`.
- **Drop-in interop:** convert Qiskit `QuantumCircuit`, `SparsePauliOp`, and Aer `NoiseModel` objects directly, or import/export OpenQASM 3.

---

## 30-Second Preview

=== "Python"

    ```python
    import ketrs

    bell = ketrs.Circuit(2).h(0).cx(0, 1)
    state = ketrs.Statevector.from_circuit(bell, seed=42)

    print(state.probabilities())      # [0.5 0.  0.  0.5]
    print(state.sample_counts(1000))  # {'00': ~500, '11': ~500}
    print(state.expectation("ZZ"))    # 1.0
    ```

=== "Rust"

    ```rust
    use ketrs::{Circuit, PauliSum, Statevector};

    let mut bell = Circuit::new(2);
    bell.h(0).cx(0, 1);

    let state = Statevector::from_circuit(&bell);
    let probs = state.probabilities(); // [0.5, 0.0, 0.0, 0.5]
    let zz = PauliSum::from_list([("ZZ", 1.0)]).unwrap();
    let exp = state.expectation(&zz).unwrap(); // 1.0
    ```

---

## Where to Go Next

- **[Installation](getting-started/install.md):** `pip install ketrs`, `cargo add ketrs`, GPU driver notes, and building from source.
- **[Quick Start](getting-started/quickstart.md):** your first circuits, measurements, expectations, and qubit ordering conventions.
- **[Choosing an Engine](getting-started/engines.md):** pick the right backend for your qubit count, precision, noise, or symmetry.
- **[Guides](guides/circuits.md):** in-depth walkthroughs for optimization, Vulkan GPU, Hamiltonians/VQE, noise, MPS/DMRG/TDVP, [qumodes & hybrids](guides/qumodes.md), and Qiskit/OpenQASM 3.
- **[Python API Reference](api/index.md):** full class and method reference (or [docs.rs/ketrs](https://docs.rs/ketrs) for Rust).
