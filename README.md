# ketrs

A quantum computer simulation backend in Rust, with Python bindings and a
Qiskit converter.

```python
import ketrs

bell = ketrs.Circuit(2).h(0).cx(0, 1)
state = ketrs.Statevector.from_circuit(bell)
state.probabilities()          # array([0.5, 0. , 0. , 0.5])
state.expectation("ZZ")        # 1.0
```

```rust
use ketrs::{Circuit, Statevector};

let mut bell = Circuit::new(2);
bell.h(0).cx(0, 1);
let state = Statevector::from_circuit(&bell);
```

Qubit ordering follows Qiskit: qubit 0 is the least significant bit of a
basis-state index, and Pauli labels have qubit 0 rightmost.

## Engines

| Engine | Use it for |
|---|---|
| `Statevector` / `RealStatevector` | Any circuit, exactly. Parallel, cache-blocked, with gate fusion. The real (f64) variant halves memory for real circuits; single-precision `Statevector32` / `RealStatevector32` (`dtype="complex64"` / `"float32"`) halve it again and run 1.65× / 4× faster than complex128 at 22–26 qubits, to about `1e-6`. |
| `SectorStatevector` | States confined to fixed Hamming weights (particle number, spin, charge): stores only that subspace, rejecting anything that leaves it. |
| `MPS` | Weakly entangled states of many qubits: TEBD, TDVP for Hamiltonians of any range, DMRG ground states, sampling, with controlled truncation. |
| `GpuStatevector` | Large circuits on any Vulkan 1.2 GPU with 64-bit floats (NVIDIA, AMD, …), up to 30 qubits memory permitting; expectation values, sampling, `H|ψ⟩`, adjoint gradients and exact `exp(-iHt)` (2.8–6.5× faster than the CPU at 18–24 qubits) run on the GPU. Needs only the system Vulkan driver at run time: no CUDA or ROCm. |

## Features

- Gates from Qiskit's standard set, arbitrary 1- and 2-qubit unitaries, and
  Pauli rotations `exp(-iθ/2·P)` of any weight applied in one pass.
- Trotter–Suzuki circuits of any order; Qiskit `PauliEvolutionGate`s map
  directly.
- Hamiltonians as Pauli sums, applied without forming a matrix: expectation
  values, `H|ψ⟩`, exact `exp(-iHt)` (Krylov), and lowest eigenstates (Lanczos).
- Parameters (`2·θ + 0.5` and so on) and adjoint gradients for any loss;
  an energy and its full gradient cost 2.3–3.5 energy evaluations (4-layer
  ansatz, 14–22 qubits), however many parameters there are.
- Sampling and mid-circuit measurement on states.
- Measurement, reset and classical control (`if_test`, with else branches
  and register conditions) inside circuits: `run_trajectory` samples one
  history, `sample_circuit` returns counts, simulating the deterministic
  prefix once and sampling the final state directly when all measurements
  are terminal.
- `ketrs.qiskit.from_qiskit` converts Qiskit circuits, including parametric
  ones, measurements, resets and `if_test` blocks; unsupported gates are
  expanded through their Qiskit definitions.

## Performance

On a Ryzen 5 5600X, against Qiskit Aer 0.18 in its fastest configuration
(see `bench/`):

- Statevector circuits: 3.1–3.8× faster than Aer at 18–26 qubits on layered
  ansätze, 1.9–3.0× on QFT, and up to 127× faster than Qiskit's NumPy
  `quantum_info.Statevector`.
- On the GPU (AMD RX 9060 XT, through Vulkan), end to end including
  downloading the state: 6–9× faster than Aer on layered ansätze at 18–26
  qubits, 4–7× on QFT. Gates are batched into shared-memory tiles over any
  set of 10 qubits, so most of a circuit takes one pass over GPU memory per
  tile rather than per gate. Expectation values on the GPU are 14–15×
  faster than downloading the state; 29 qubits (8 GiB) runs on a 16 GB
  card.
- MPS (TEBD of an Ising chain, 50–100 qubits): 9–16× faster than Aer's
  MPS method at bond dimension ≤ 64 and 34× at 128 (2.2 s against 76 s),
  with errors against the exact result at the truncation level for both.
  Commuting Trotter terms are packed into layers and disjoint bonds are
  updated in parallel.
- Exact evolution and ground states: faster than scipy's `expm_multiply`
  and `eigsh` (with the sparse matrix built in advance) at every size from
  12 qubits: 1.7× at 12, 2.5–2.7× at 14, 3.5–12× at 18–20.

## Example

[`examples/schwinger`](examples/schwinger) simulates the lattice Schwinger
model: exact particle production in the charge-neutral sector, and DMRG
vacuum properties up to 100 sites, each checked against an independent
method.

## Development

```sh
uv venv && uv pip install maturin pytest qiskit qiskit-aer numpy scipy
maturin develop --release     # build the Python module (add --features gpu)
cargo test --workspace        # Rust tests (add --all-features for the GPU)
pytest                        # Python tests
```

GPU support is opt-in: build with `--features gpu` (Rust) or
`maturin develop --features gpu` (Python). It compiles its GLSL shaders with
`glslc` during the build: install it from the Vulkan SDK or your
distribution's `shaderc`/`glslc` package, or set `GLSLC` to its path. Running
needs only a Vulkan driver.

Status: early development. See [PLAN.md](PLAN.md).

## Licence

Licensed under either of [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT),
at your option.
