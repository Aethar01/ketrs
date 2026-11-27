# Noise & Dynamic Circuits

`ketrs` supports exact density-matrix noise simulation, quantum trajectories on statevectors (with readout error), and dynamic mid-circuit measurements with full classical control flow across every engine.

## Quantum Channels & Noise Models

Build channels from standard noise primitives (`depolarizing`, `amplitude_damping`, `phase_damping`, `bit_flip`, `phase_flip`, `pauli`) or arbitrary Kraus operators (`Channel.from_kraus`):

```python
import ketrs

noise = ketrs.NoiseModel(
    one_qubit=ketrs.Channel.depolarizing(0.005, n_qubits=1),
    two_qubit=ketrs.Channel.depolarizing(0.02, n_qubits=2),
    readout=(0.01, 0.02),  # P(record 1 | true 0), P(record 0 | true 1)
)
```

### Exact Density Matrix (`DensityMatrix`)

For up to ~14 qubits, `DensityMatrix` tracks the exact mixed state $\rho$:

```python
import ketrs

bell = ketrs.Circuit(2).h(0).cx(0, 1)

rho = ketrs.DensityMatrix(2, seed=0)
rho.run_noisy(bell, noise)

# Apply an extra channel directly to a specific qubit
rho.apply_channel(ketrs.Channel.amplitude_damping(0.05), [1])

ideal = ketrs.Statevector.from_circuit(bell)
print(f"Trace:    {rho.trace():.6f}")
print(f"Purity:   {rho.purity():.6f}")
print(f"Fidelity: {rho.fidelity(ideal):.6f}")
print("⟨ZZ⟩:    ", rho.expectation("ZZ"))
```

### Quantum Trajectories on `Statevector`

For larger circuits where storing a $2^n \times 2^n$ density matrix is infeasible, pass `noise=` to `Statevector.run` or `Statevector.sample_circuit`. Each shot randomly samples Kraus operators after each gate and applies readout error on measurements:

```python
import ketrs

bell_meas = ketrs.Circuit(2, 2).h(0).cx(0, 1).measure_all()
counts = ketrs.Statevector(2, seed=0).sample_circuit(bell_meas, shots=2000, noise=noise)
print(counts)
```

If you already have a Qiskit Aer `NoiseModel` with uniform gate errors, convert it directly with `ketrs.qiskit.noise_model_from_aer(aer_noise)`.

---

## Dynamic Circuits & Classical Control Flow

Circuits can interleave gates with `measure`, `reset`, `if_test`, `if_expr`, `assign`, `while_loop`, `for_loop`, `break_loop`, and `continue_loop` on every engine (`Statevector`, `GpuStatevector`, `MPS`).

### Feed-Forward Corrections (`if_test`)

```python
import ketrs

# Teleport-style feed-forward: measure q0, flip q1 if outcome == 1
c = ketrs.Circuit(2, 2)
c.h(0).measure(0, 0)
c.if_test(0, 1, ketrs.Circuit(2, 2).x(1))
c.measure(1, 1)

# Single trajectory (mutates state, returns classical bits)
sv = ketrs.Statevector(2, seed=1)
bits = sv.run(c)
print("Single run bits:", bits)

# Fast batched shot sampling (leaves initial state unchanged)
counts = ketrs.Statevector(2, seed=1).sample_circuit(c, shots=1000)
print("Counts:", counts)  # {'00': ~500, '11': ~500}
```

### Classical Expressions & `while_loop` (Repeat-Until-Success)

Conditions for `if_expr`, `while_loop`, and `assign` use nested tuples:

- `("bit", c)` or `("register", [c0, c1, ...])` (least significant bit first)
- `("value", v)`
- `("not", e)`, `("bit_not", e, width)`
- `(op, a, b)` for `op` in `"and"`, `"or"`, `"bit_and"`, `"bit_or"`, `"bit_xor"`, `"=="`, `"!="`, `"<"`, `"<="`, `">"`, `">="`

```python
import ketrs

# Repeat preparing & measuring q0 until it reads 1
body = ketrs.Circuit(2, 2).reset(0).ry(0.8, 0).measure(0, 0)

rus = ketrs.Circuit(2, 2)
rus.while_loop(("==", ("bit", 0), ("value", 0)), body)
rus.cx(0, 1).measure(1, 1)

counts = ketrs.Statevector(2, seed=0).sample_circuit(rus, shots=500)
print(counts)  # {'11': 500}
```

!!! info "Why `sample_circuit` is so fast on dynamic circuits"
    `sample_circuit` executes each distinct mid-circuit measurement history **once**, splitting shots across outcomes with binomial draws, and samples any terminal measurements directly from each branch's final state (up to 460× faster than Qiskit Aer on 20-qubit feed-forward circuits).
