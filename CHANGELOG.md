# Changelog

## Unreleased

### MPS

- `Mps::tdvp` and `Mps::dmrg` can run on a GPU (the `gpu` feature):
  `Mps::set_gpu(Some(ctx))` in Rust, `mps.gpu = True` in Python. The
  sweeps' environments then live in GPU memory and the one- and two-site
  effective Hamiltonians are applied there, as complex matrix products
  around a sparse step for the MPO, which is nearly all of a sweep's
  work. TDVP's Krylov time steps and DMRG's Lanczos eigensolves run their
  recurrence in GPU memory as well, bringing back only its scalars; DMRG
  keeps the Lanczos basis there instead of replaying the recurrence, so it
  takes half the products. The tensors, SVDs, subspace expansion and DMRG
  noise stay on the CPU, so results match the CPU path up to rounding. It needs 64-bit floats and buffer device addresses
  (`GpuContext::supports_mps`). The CPU path stays the default.
- TDVP and DMRG sweeps on the CPU use all cores. The one- and two-site
  effective Hamiltonians split by rows of the left bond, one chunk per
  thread with nothing to combine, and environment updates and the subspace
  expansion split by MPO channel. Before, nearly every matrix product in a
  sweep was too small for faer to split, so sweeps ran on one core. A
  192-qubit TDVP step at bond dimension 64 went from 416 s to 132 s on
  three threads; one thread is also about 25% faster on two-site
  products, as the sparse MPO step now runs along contiguous rows.
- TDVP's subspace expansion is about 2.6× faster. It first reduces each
  bond's `Hψ` channels, many more rows than columns, to the triangular
  factor of a QR decomposition, which has the same row space, so the
  projections and SVDs that follow work on a square matrix.
- `Mps::dmrg` adds noise in early sweeps, so Hamiltonians whose terms span
  more than two sites no longer trap it in metastable states. One example
  is a lattice gauge theory whose hops act on four qubits: from its
  strong-coupling vacuum, plain two-site DMRG never moved, and from weak
  random starts it settled above the ground state and reported
  convergence. The noise is White's density-matrix perturbation in
  subspace-expansion form. Each split keeps the two-site tensor's own
  truncated basis exactly and widens it, at zero weight, with the
  strongest directions that the partial terms crossing the bond reach
  (each MPO channel weighted equally). It therefore costs time but never
  accuracy. The strength is
  `DmrgOptions::noise` (default `1e-4`). It falls by `noise_decay`
  (default `0.1`) each sweep and is switched off once below the cutoff or
  once a sweep settles. Convergence is only reported after a sweep without
  it. Python: `MPS.dmrg(..., noise=1e-4, noise_decay=0.1)`; `noise=0`
  gives the previous behaviour.
- `Mps::tdvp`'s subspace expansion now reaches every bond. It enlarged the
  bonds with `ψ ⊕ ε·Hψ` and then dropped every direction below `1e-13` of
  ψ's largest singular value, but `Hψ`'s weight shrank by about 10× per
  site from the right end. Hops more than a few sites from the right end
  therefore never acted, and from a product state the left bonds stayed
  at 1. The error stayed at 0.35 however many steps were taken. The
  expansion now projects each MPO channel of `Hψ` onto the already
  expanded basis and appends the orthonormalised complement, enlarging the
  right bases before the right half-sweep and the left bases before the
  left half-sweep. From a product state, the error now falls as for a
  second-order integrator.
- MPS SVDs no longer panic or return NaN on finite, rank-deficient
  tensors (for example two-site tensors whose bonds an expansion padded with
  zero-weight directions). faer's divide-and-conquer SVD can stop with
  `NoConvergence` or report success with NaN singular vectors there; the
  output is now checked, and the SVD is retried with QR iteration, then on
  the triangular factor of a QR decomposition, then through the Gram
  matrix's eigendecomposition.

### Time evolution

- Krylov time evolution (`Statevector.evolve`, `GpuStatevector.evolve`,
  sparse-operator evolution and MPS TDVP on the CPU or the GPU) grows each
  Lanczos basis only until it covers the remaining time, checked every 8
  vectors, up to 64 vectors. It used to build 40 every time, and a basis
  that fell short took half or a quarter of the time; now it takes about
  the longest step it covers.
  The tolerance is unchanged (1e-13 per step). Each check is a small
  Taylor-series exponential, and an eigendecomposition screens out the
  checks that clearly fail first. A 192-qubit TDVP step at bond dimension
  64 took 27% less time on the CPU and 17% less on the GPU;
  `GpuStatevector.evolve` of a Heisenberg chain took 20–32% less at 18–24
  qubits.

## 0.1.6 (2026-11-13)

### Features

- `SparseOperator`: a Hermitian operator stored as a sparse matrix in the
  computational basis, for operators that are sparse as matrices but long as
  Pauli sums (for example hops whose amplitude depends on the whole
  configuration). `Statevector.expectation`, `apply_pauli_sum` and `evolve`
  accept one in place of a `PauliSum`, and `SparseOperator.from_scipy`
  builds one from any SciPy sparse matrix. In Rust:
  `SparseOperator::from_csr` and `DenseState::{expectation_sparse,
  apply_sparse, evolve_sparse}`.

## 0.1.5 (2026-11-09)

### Python

- First release on PyPI: `pip install ketrs`. Wheels for CPython 3.10–3.15
  and free-threaded 3.14t and 3.15t, on Linux (x86-64 and arm64, glibc
  2.28+), Windows (x86-64) and macOS (Apple silicon). The Linux and Windows
  wheels include `GpuStatevector`.
- Free-threaded Python is supported: importing ketrs leaves the GIL
  disabled, and ketrs can be used from several threads at once.

### Releases

- Releases are published to crates.io and PyPI from CI, by trusted
  publishing, when a version tag is pushed.

## 0.1.4 (2026-11-09)

### GPU

- Tiles apply each gate in the cheapest form its matrix allows: diagonal
  gates multiply only the amplitudes they change, X-like gates and swaps
  move amplitudes without arithmetic, real matrices skip the imaginary
  parts, and controlled gates visit only the amplitudes with every control
  set, so no thread idles. In double precision, which consumer GPUs run at
  a fraction of their float speed, only runs of one-qubit gates are fused,
  keeping two-qubit gates in these forms. On an RX 9060 XT, double-precision
  QFT runs 2.2–2.6× faster at 26–28 qubits and the layered ansatz 1.6–1.9×,
  so the GPU is ahead of the CPU again: 1.0–1.8× on QFT and 1.3–2.6× on the
  layered ansatz at 18–26 qubits, end to end.
- On an RTX 4090 at 22–26 qubits, double-precision circuits run 2.2–3.3×
  faster.
- Every tile includes the two lowest qubits, so neighbouring threads load
  neighbouring amplitudes. Tiles of only high qubits scattered each warp's
  loads across the state, which at 28 qubits in double precision (4 GiB)
  made passes about 15× slower on an RTX 4090: 28-qubit circuits there run
  6–10× faster (QFT 4.89 s to 502 ms). Single precision at 28 qubits runs
  1.15–1.7× faster on an RTX 4090 and an RX 9060 XT.
- Against Aer 0.18's CUDA backend on an RTX 4090 (GPU work only), ketrs is
  now 1.7–2.3× faster in double precision at 22–28 qubits, against Aer or
  cuStateVec, whichever is faster (0.1.3: 0.29–0.96×), and 2.3–4.2× faster
  in single precision.

## 0.1.3 (2026-11-08)

### GPU

- Tiles take gates out of order: a later gate joins a tile ahead of gates
  it shares no qubit with, so layered circuits need far fewer passes over
  GPU memory. Single-precision circuits run 1.15–1.5× faster at 24–26
  qubits on an RX 9060 XT; on an RTX 4090, 28-qubit circuits in double
  precision run 4.3–4.9× faster (with the larger tiles below).
- NVIDIA GPUs use 11-qubit tiles, which beat 10 there; other GPUs keep 10.
- States split across several buffers reach them through buffer device
  addresses instead of arrays of descriptors, so the GPU needs 6 storage
  buffers per shader instead of 34, and splitting needs Vulkan 1.2's
  `bufferDeviceAddress` instead of non-uniform descriptor indexing.
- Downloading a state copies on every core, straight into uninitialised
  memory: 1 GiB takes 208 ms instead of 496 ms.

### Benchmarks

- `bench/compare.py` compares with Aer on the GPU (double and single
  precision, and cuStateVec if installed) when Aer has a CUDA device;
  `--gpu-max-qubits` runs larger sizes on the GPU only.

## 0.1.2 (2026-11-08)

### Circuits

- `sample_circuit` follows each distinct measurement history once instead
  of every shot. At a measurement or reset whose outcome is random, the
  shots still on that path split between its outcomes by a binomial draw,
  and each outcome carries its share; measurements and assignments after
  the last control flow are sampled from each path's final state. Counts
  have the same distribution as before, but the random stream differs, so
  seeded counts of circuits with mid-circuit measurement change.
  Statevectors, MPS and GPU states all branch; noisy trajectories still run
  shot by shot.
- On 1000 shots against 0.1.1: teleportation runs in 40 µs instead of
  1 ms, a 16-qubit circuit with feed-forward in 9.3 ms instead of 15.4 s,
  and a 20-qubit repeat-until-success loop in 0.70 s instead of 404 s.

### Statevectors

- Faster gate kernels: gates on the lowest qubits run through block-level
  kernels; tiles cover only the qubits gates mix, with commuting gates
  batched together; runs of diagonal gates in a tile apply in one sweep;
  swaps relabel qubits instead of moving amplitudes; and Rx-like gates
  skip full complex multiplies.
- Measurement, collapse and sampling each take one parallel pass over the
  state: measuring a qubit of a 20-qubit state takes 1.2 ms instead of
  12 ms, and drawing 1000 samples 0.7 ms instead of 2.9 ms. `sample` no
  longer builds every cumulative weight.

## 0.1.1 (2026-11-06)

### Circuits

- `Instruction::Assign { clbit, value }` and `Circuit::assign` (Python:
  `Circuit.assign`) set a classical bit to whether a classical expression
  is nonzero, anywhere in a circuit, including inside branches and loops.
  `sample_circuit` still simulates the gates before the first measurement
  once, applying leading assignments along the way.
- OpenQASM 3 import and export of classical assignments: `c[k] = <expr>;`
  and `bit b = <expr>;` (and `bit b = measure q;`).
- `from_qiskit` converts `store` into a classical bit (`qc.store(clbit,
  expr)`) to an assignment.

## 0.1.0 (2026-11-06)

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
- Measurement, reset, `if`/`else`, `while` and `for` on classical
  expressions, with `break` and `continue`, through one trajectory driver
  on every engine.
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
  subroutines, `inv`/`pow`/`ctrl(n)`/`negctrl(n)` modifiers, control flow
  including `break` and `continue`.
- OpenQASM 3 export (`Circuit::to_qasm`) with standard gates only.
- Python bindings (`pip install ketrs`, module `ketrs`), with converters
  from Qiskit circuits, `SparsePauliOp`s and Aer noise models.

### Known limitations

- OpenQASM `switch`, classical variables other than bits, and subroutines
  with return values are not supported.
- Aer noise models must apply the same errors on every qubit.
- The GPU engine needs a Vulkan 1.2 driver; double precision needs
  `shaderFloat64`.
