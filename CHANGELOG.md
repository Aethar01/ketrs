# Changelog

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
