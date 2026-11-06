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
| `Statevector` / `RealStatevector` | Any circuit, exactly. Parallel, cache-blocked, with gate fusion. The real (f64) variant halves memory for real circuits; single-precision `Statevector32` / `RealStatevector32` (`dtype="complex64"` / `"float32"`) halve it again and run 1.65× / 4× faster than complex128 at 22–26 qubits, to about $`10^{-6}`$. |
| `SectorStatevector` | States confined to fixed Hamming weights (particle number, spin, charge): stores only that subspace, rejecting anything that leaves it. |
| `DensityMatrix` | Mixed states and noise: Kraus channels (depolarizing, amplitude and phase damping, Pauli, any Kraus set) and noise models after every gate, matching Qiskit Aer's density-matrix method and 1.3–4.2× faster. For more qubits, statevectors run the same noise models (plus readout error) as quantum trajectories. |
| `MPS` | Weakly entangled states of many qubits: TEBD, TDVP for Hamiltonians of any range (with subspace expansion while bonds grow), DMRG ground states, sampling, with controlled truncation. |
| `GpuStatevector` | Large circuits on any Vulkan 1.2 GPU (NVIDIA, AMD, …), up to 31 qubits memory permitting, in double precision where the GPU has 64-bit floats or single (`complex64`, 2.8–5.8× faster on consumer cards) anywhere; expectation values, sampling, $`H\vert\psi\rangle`$, adjoint gradients and exact $`e^{-iHt}`$ (2.8–6.5× faster than the CPU at 18–24 qubits) run on the GPU. Needs only the system Vulkan driver at run time: no CUDA or ROCm. |

## Features

- Gates from Qiskit's standard set, arbitrary 1- and 2-qubit unitaries,
  Pauli rotations $`e^{-i\theta P/2}`$ of any weight applied in one pass, and
  gates and swaps with any number of controls (`ccx`, `mcx`, `cswap`, …) on
  every engine. `Circuit.control` controls a whole circuit.
- Trotter–Suzuki circuits of any order; Qiskit `PauliEvolutionGate`s map
  directly.
- Hamiltonians as Pauli sums, applied without forming a matrix: expectation
  values, $`H\vert\psi\rangle`$, exact $`e^{-iHt}`$ (Krylov), and lowest eigenstates (Lanczos).
- Parameters ($`2\theta + 0.5`$ and so on) and adjoint gradients for any loss;
  an energy and its full gradient cost 2.3–3.5 energy evaluations (4-layer
  ansatz, 14–22 qubits), however many parameters there are.
- Sampling and mid-circuit measurement on states.
- Noise on statevectors by quantum trajectories (`noise=` on `run` and
  `sample_circuit`), with readout error; `ketrs.qiskit.noise_model_from_aer`
  converts Aer noise models that treat every qubit alike.
- Measurement, reset and classical control (`if_test` with else branches,
  `while_loop`, `for_loop`, `break_loop`/`continue_loop`, conditions on
  bits, registers or classical expressions, and `assign` to set a bit to
  an expression's value) inside circuits, on every
  engine: `run_trajectory` samples one history, `sample_circuit` returns
  counts, simulating the deterministic prefix once and sampling the final
  state directly when all measurements are terminal.
- `ketrs.from_qasm` (Rust: `ketrs::qasm::parse`) reads OpenQASM 3 and 2:
  registers, inputs as parameters, the standard gates and `qelib1.inc`,
  `gate` definitions and `def` subroutines, `inv`/`ctrl(n)`/`negctrl(n)`/`pow`
  modifiers, measurement, reset, classical assignments (`c[k] = …;`,
  `bit b = …;`), `if`, `while` and `for` with `break` and `continue`.
  `Circuit.to_qasm()` writes OpenQASM 3 with standard gates only, which
  Qiskit's `qasm3.loads` reads.
- `ketrs.qiskit.from_qiskit` converts Qiskit circuits, including parametric
  ones, measurements, resets, `if_test`, `while_loop` and `for_loop` blocks
  with `break_loop`/`continue_loop`, classical expressions, and `store`
  into classical bits; controlled gates become native multi-controlled
  ones, and other unsupported gates are
  expanded through their Qiskit definitions.

## Performance

Measured on a Ryzen 5 5600X and an AMD RX 9060 XT (through Vulkan), against
Qiskit Aer 0.18 in its fastest configuration (passive OpenMP waiting, and
fusion on or off, whichever is faster at each size). Reproduce with the
scripts in [`bench/`](bench). In the bar charts each block is 1× unless
noted.

**Statevector circuits**, end to end from $`\lvert 0\dots 0\rangle`$ to the
final amplitudes. The GPU column is compared with ketrs on the CPU, not with
Aer: Aer's GPU mode needs CUDA, which this card can't run. GPU times include
downloading the state.

*Layered ansatz: 10 layers of $`R_y`$ rotations and a CNOT chain*

| Qubits | Aer | ketrs | vs Aer | ketrs GPU | GPU vs CPU |
|---:|---:|---:|:---|---:|:---|
| 18 | 44.8 ms | 10.9 ms | `████▏` 4.1× | 6.3 ms | `█▊` 1.7× |
| 20 | 157.5 ms | 46.0 ms | `███▍` 3.4× | 21.4 ms | `██▏` 2.2× |
| 22 | 906.3 ms | 178.8 ms | `█████▏` 5.1× | 104.8 ms | `█▋` 1.7× |
| 24 | 3.85 s | 818.6 ms | `████▊` 4.7× | 440.3 ms | `█▉` 1.9× |
| 26 | 17.42 s | 3.97 s | `████▍` 4.4× | 2.00 s | `██` 2.0× |

*Quantum Fourier transform*

| Qubits | Aer | ketrs | vs Aer | ketrs GPU | GPU vs CPU |
|---:|---:|---:|:---|---:|:---|
| 18 | 19.0 ms | 7.2 ms | `██▋` 2.6× | 3.0 ms | `██▍` 2.4× |
| 20 | 48.4 ms | 25.8 ms | `█▉` 1.9× | 12.8 ms | `██` 2.0× |
| 22 | 292.9 ms | 115.6 ms | `██▌` 2.5× | 71.8 ms | `█▋` 1.6× |
| 24 | 1.60 s | 521.7 ms | `███▏` 3.1× | 313.5 ms | `█▋` 1.7× |
| 26 | 7.42 s | 3.02 s | `██▌` 2.5× | 1.56 s | `█▉` 1.9× |

Against Qiskit's NumPy `quantum_info.Statevector`, ketrs is 67–88× faster at
18–20 qubits. Below 18 qubits the comparison with Aer is dominated by Aer's
per-job overhead of about 2 ms, so it is left out.

On the GPU, gates are batched into shared-memory tiles over any set of 10
qubits, so most of a circuit takes one pass over GPU memory per tile rather
than per gate. Expectation values computed on the GPU are 14–15× faster than
downloading the state; 29 qubits (8 GiB) runs on a 16 GB card.

**MPS**: TEBD of a transverse-field Ising chain (second-order Trotter,
$`\Delta t = 0.1`$) against Aer's MPS method with the same bond-dimension cap
$`\chi`$. The two agree on $`\langle Z\rangle`$ to the truncation level.
One block is 2×.

| Qubits | Steps | $`\chi`$ cap | ketrs | Aer MPS | Speedup |
|---:|---:|---:|---:|---:|:---|
| 50 | 10 | 32 | 0.01 s | 0.04 s | `███▏` 6.3× |
| 100 | 10 | 32 | 0.01 s | 0.07 s | `█████▊` 11.5× |
| 50 | 20 | 64 | 0.02 s | 0.18 s | `████▌` 9.0× |
| 100 | 20 | 64 | 0.03 s | 0.38 s | `███████▎` 14.6× |
| 100 | 40 | 128 | 2.33 s | 75.73 s | `████████████████▎` 32.5× |

Commuting Trotter terms are packed into layers and disjoint bonds are updated
in parallel.

**Exact evolution and ground states** of a Heisenberg chain, against scipy's
`expm_multiply` and `eigsh` on a sparse matrix built in advance (construction
not timed). Results agree to about $`10^{-14}`$.

| Workload | Qubits | ketrs | scipy | Speedup |
|---|---:|---:|---:|:---|
| $`e^{-iHt}\lvert\psi\rangle`$ | 14 | 10.4 ms | 28.7 ms | `██▊` 2.8× |
| $`e^{-iHt}\lvert\psi\rangle`$ | 18 | 196.5 ms | 728.7 ms | `███▊` 3.7× |
| $`e^{-iHt}\lvert\psi\rangle`$ | 20 | 732.7 ms | 3.80 s | `█████▏` 5.2× |
| Ground state | 14 | 16.4 ms | 67.8 ms | `████▏` 4.1× |
| Ground state | 18 | 165.7 ms | 2.81 s | `█████████████████` 17.0× |
| Ground state | 20 | 2.74 s | 11.06 s | `████` 4.0× |

At 20 qubits the Pauli-sum table exceeds its 64 MiB budget and ketrs falls
back to a slower path, hence the drop.

## Example

[`examples/schwinger`](examples/schwinger) simulates the lattice Schwinger
model: exact particle production in the charge-neutral sector, and DMRG
vacuum properties up to 100 sites, each checked against an independent
method.

## Development

```sh
uv venv && uv pip install maturin pytest qiskit qiskit-aer qiskit-qasm3-import numpy scipy
maturin develop --release     # build the Python module (add --features gpu)
cargo test --workspace        # Rust tests (add --all-features for the GPU)
pytest                        # Python tests
```

GPU support is opt-in: build with `--features gpu` (Rust) or
`maturin develop --features gpu` (Python). It compiles its GLSL shaders with
`glslc` during the build: install it from the Vulkan SDK or your
distribution's `shaderc`/`glslc` package, or set `GLSLC` to its path. Running
needs only a Vulkan driver.

Status: early releases (0.1); APIs may still change. See
[CHANGELOG.md](CHANGELOG.md).

## Licence

Licensed under either of [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT),
at your option.
