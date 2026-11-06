# ketrs plan

A quantum computer simulation backend in Rust, with Python bindings.

## Goal

A public, tested and benchmarked simulator. Make the core solid before adding more features.

## Decisions

- **Name:** `ketrs` for the Rust crate and the Python module (`import ketrs`).
- **Qubit ordering:** same as Qiskit. Qubit 0 is the least significant bit of the basis-state index, so results can be compared with Qiskit directly.
- **Licence:** MIT OR Apache-2.0.
- **Linear algebra:** `faer` (pure Rust, fast SVD, no system BLAS, so wheels are easy to build).
- **MSRV:** Rust 1.88 (let-chains).
- **Release point:** not decided yet.

## Layout

```
ketrs/
├── Cargo.toml              # workspace
├── crates/
│   ├── ketrs/              # core library
│   │   ├── src/circuit.rs      # Gate enum, Circuit, parameters
│   │   ├── src/statevector/    # dense engine
│   │   ├── src/mps/            # tensor-network engine
│   │   ├── src/observable.rs   # PauliString / PauliSum, expectation values
│   │   └── benches/            # criterion
│   └── ketrs-py/           # PyO3 + maturin bindings
├── python/                 # Python package, Qiskit converter, pytest
└── examples/schwinger/     # lattice Schwinger model + result plot
```

## Core design

- **Circuit IR**, used by every backend:
  - `Gate`: H, X, Y, Z, S, T, RX/RY/RZ(θ), CX, CZ, SWAP, arbitrary 1q/2q unitaries, controlled gates.
  - `Circuit { n_qubits, ops }`, plus measurement.
  - A `Backend` trait with `run`, `amplitudes`, `expectation(&PauliSum)` and `sample(shots)`.
- **Statevector engine:**
  - State stored as `Vec<Complex64>`.
  - Stride-loop gate kernels, with special cases for diagonal and permutation gates (Z, S, T, CZ, CX, SWAP).
  - `rayon` parallelism above about 14 qubits.
  - Practical limit about 28 qubits; 28 qubits needs 4 GiB.
- **MPS engine:**
  - A chain of rank-3 tensors, truncated by SVD to a maximum bond dimension χ and a cutoff.
  - Two-qubit gates act on neighbouring qubits; non-neighbouring qubits are moved together with swaps.
  - TEBD for time evolution.
- **Time evolution helper:** turns a `PauliSum` Hamiltonian into a Trotterised circuit.

## Testing and benchmarks

- Unit tests against known gate results.
- `proptest` checks that gates are unitary and the norm is preserved.
- Engine cross-check: at large χ, the MPS engine must match the statevector engine on random circuits.
- Python tests comparing results with Qiskit Aer.
- Benchmarks with `criterion` and a Python harness:
  - QFT and random circuits, from 10 to 28 qubits.
  - Compared with NumPy and Qiskit Aer.
  - Only quote numbers that hold up.

## Milestones

| #     | Deliverable                                                      | Status |
| ----- | ---------------------------------------------------------------- | ------ |
| M0    | Workspace, CI (fmt, clippy, test), licences                      | Done |
| M1    | Circuit IR, statevector engine, tests                            | Done |
| M2    | Observables, sampling, measurement                               | Done |
| M3    | Parallel and specialised kernels, benchmarks                     | Done: kernels, cache blocking, selective gate fusion, Aer/NumPy comparison |
| M4    | PyO3 bindings, Qiskit converter                                  | Done |
| M5    | MPS engine, TEBD and DMRG                                        | Done |
| M6    | Schwinger model example with a result plot                       | Done: `examples/schwinger` |
| M7    | GPU statevector (Vulkan compute, f64 via `shaderFloat64`)        | Done: `GpuStatevector`, NVIDIA and AMD through Vulkan only |

Added along the way (from the Schwinger project's optimisations):

- **Pauli rotations** `exp(-iθ/2·P)` of any weight, applied in one pass;
  `rxx`/`ryy`/`rzz`/`rzx`; Trotter–Suzuki circuits of any order; Qiskit
  `PauliEvolutionGate` mapped directly.
- **Matrix-free Pauli sums**: terms grouped by X-mask for `H|ψ⟩` and
  expectation values; exact `exp(-iHt)` by Krylov time-stepping; lowest
  eigenstates by Lanczos with deflation (two-pass, memory-light).
- **Generic engine**: `DenseState<T: Scalar>`, with `Statevector` (complex)
  and `RealStatevector` (f64, half the memory) sharing every kernel, and
  single-precision `Statevector32`/`RealStatevector32` (and sector
  equivalents): everything but MPS and the GPU, with tolerances scaled to
  the precision. Python takes `dtype="complex64"` or `"float32"`.
- **Parameters and adjoint gradients**: linear parameter forms, `bind`,
  `adjoint_gradient` for any loss given its cotangent, `expectation_gradient`.
  The backward sweep fuses the gates between parameters and runs gates and
  overlaps tile by tile, so the energy and full gradient of a 4-layer
  ansatz cost 2.3–3.5 energy evaluations (was 3.4–7.5).
- **Tiles over any qubits**: from 20 qubits, runs of gates on any 14 qubits
  (plus the lowest few, for contiguous reads) are gathered into a cached
  tile and applied together, as on the GPU: one pass over memory per run
  instead of per gate (1.3× faster at 22 qubits). States of 13–15 qubits,
  too small for per-gate threading, are cut into 16 tiles that run in
  parallel (2.1× faster at 14 qubits, 1.5× for gradients).
- **Tabulated Pauli sums**: for evolution and eigensolvers, each X-mask
  group's coefficients are tabulated per row (up to 64 MiB), an implicit
  sparse matrix, and rows run in parallel from 2^11 amplitudes. Ground
  states at 14 qubits went from 4.5× slower than scipy's `eigsh` to 2.7×
  faster.
- **Parallel TEBD**: `Mps::run` schedules gates as early as their sites
  allow and updates each layer's disjoint bonds in parallel, in Hastings'
  form (right-orthonormal tensors plus bond singular values, no inverses).
  Trotter circuits pack runs of commuting terms into layers on disjoint
  qubits, exactly. MPS TEBD went from 1.5–2.5× to 9–34× faster than Aer.
- **TDVP**: two-site, second-order projector-splitting TDVP on the MPO, for
  real-time dynamics with couplings of any range. Exact to 1e-14 when bonds
  are full; from product states, accuracy is lost while bonds grow (a
  subspace-expansion start would fix that).
- **GPU gradients**: `H|ψ⟩` in one pass, and adjoint gradients with ψ and
  λ bound as each other's partner state, so the whole backward sweep (tiled
  inverse segments and overlaps) is a few submissions and one download.
  2.2–2.6× faster than the CPU at 16–24 qubits, as for forward circuits.
- **GPU evolution**: Krylov `exp(-iHt)` with four statevectors on the GPU:
  one pass of the recurrence downloading only α and β, then a replay from
  the stored coefficients with no downloads. 2.8–6.5× faster than the CPU
  at 18–24 qubits.
- **GPU precision and size**: every kernel also compiles in single
  precision, for any GPU (f64 parameters are decoded in the shader with
  integer arithmetic, so 31-bit masks stay exact); `complex64` is 2.8–5.8×
  faster than `complex128` on a consumer card. States span up to sixteen
  buffers, so 31 qubits on any device.
- **Density matrices and noise**: ρ as a 2n-qubit vector, so gates run on
  the statevector kernels (U on rows, U* on columns); channels as sparse
  superoperators, and each noisy gate fused with its noise into one, all
  through the tiled kernels. Matches Aer's density-matrix method to 1e-10;
  2.9–4.2× faster noiseless and 1.3–1.6× with noise at 8–13 qubits.
- **SIMD kernels**: hand-written AVX2/FMA for complex doubles in the general
  one- and two-qubit kernels and diagonal scaling (one `fmaddsub` per
  complex multiply), chosen at run time with portable fallbacks
  (`KETRS_NO_SIMD=1` forces them). Layered circuits 17–23% faster at 18–22
  qubits; `target-cpu=native` auto-vectorisation gained only 6–12%.
- **OpenQASM 3 import**: a hand-written lexer and recursive-descent parser to
  a syntax tree, lowered onto `Circuit`: registers, inputs as parameters,
  `stdgates.inc` and `qelib1.inc`, gate definitions, modifiers,
  broadcasting, measurement, reset, `if`/`while` on classical expressions,
  unrolled `for`. Round-trips Qiskit's `qasm3.dumps` and `qasm2.dumps`.
- **Symmetry sectors**: fixed Hamming weights on any qubit subsets (Lin
  two-table indexing); circuits, observables, evolution and eigenstates in
  the sector, with symmetry-breaking operations rejected.
- **Classical control**: `measure`, `reset`, `if_test` (else branches,
  nesting) and `while_loop` inside circuits, conditioned on bits, registers
  or classical expressions (logic, bitwise, comparisons); trajectories and
  counts on statevectors, MPS of any length and the GPU through one shared
  driver. Qiskit import of all of these, with `for_loop`s unrolled. Fusion
  never crosses an instruction.
- **Multi-controlled operations**: gates and swaps with any number of
  controls, applied directly on statevectors and density matrices; the GPU
  pair kernel takes the control mask, and MPS decomposes them exactly
  (Barenco's `V² = U` recursion) into one- and two-qubit gates.

## Later

- Noise on statevectors by quantum trajectories (for more qubits than a
  density matrix allows); readout error; Aer noise-model import.
- OpenQASM: export (`to_qasm`), `def` subroutines, `ctrl(n) @` and `c3x`
  onto the native multi-controlled operations.

## Benchmarks

`bench/compare.py` runs the same circuits on ketrs, Qiskit Aer and
`qiskit.quantum_info.Statevector` (NumPy), all from Python. Aer runs with
`OMP_WAIT_POLICY=PASSIVE`, its fastest setting on the Ryzen 5600X here.
