# Qiskit & OpenQASM 3

You don't need to rewrite existing circuits to use `ketrs`. Convert Qiskit circuits, observables, and noise models with `ketrs.qiskit`, or import/export OpenQASM 3 directly.

## Qiskit Conversion (`ketrs.qiskit`)

Install the Qiskit extra first:

```sh
pip install "ketrs[qiskit]"
```

### Circuits (`from_qiskit`)

```python
from qiskit import QuantumCircuit
import ketrs
from ketrs.qiskit import from_qiskit

qc = QuantumCircuit(3)
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)

circ = from_qiskit(qc)
state = ketrs.Statevector.from_circuit(circ)
print(state.probabilities())  # [0.5 0. 0. 0. 0. 0. 0. 0.5]
```

What `from_qiskit` converts natively:

- All standard 1- and 2-qubit gates, plus arbitrary 1- and 2-qubit `UnitaryGate`s.
- Parametric circuits (matching `qc.parameters` order so `circ.bind(values)` matches `qc.assign_parameters(values)`), including affine angle expressions like `2 * theta + 1`.
- `PauliEvolutionGate`s with standard Trotter–Suzuki synthesis (mapped directly to native `ketrs` Pauli rotations without CNOT ladders).
- Multi-controlled gates (`ccx`, `mcx`, `cswap`, and bound `ControlledGate` / `AnnotatedOperation` gates) mapped to native multi-controlled operations.
- `measure`, `reset`, `if_test`, `while_loop`, `for_loop`, `break_loop`, `continue_loop`, classical expressions, and `store` into classical bits.
- Any other composite gate is expanded automatically through its Qiskit `.definition` until only supported gates remain.

### Observables (`pauli_sum_from_qiskit`)

Convert any Hermitian `SparsePauliOp` to a `ketrs.PauliSum`:

```python
from qiskit.quantum_info import SparsePauliOp
from ketrs.qiskit import pauli_sum_from_qiskit

op = SparsePauliOp.from_list([("ZZ", -1.0), ("IX", -0.5), ("XI", -0.5)])
H = pauli_sum_from_qiskit(op)
```

### Aer Noise Models (`noise_model_from_aer`)

Convert a Qiskit Aer `NoiseModel` whose 1-qubit gate error, 2-qubit gate error, and readout error are uniform across qubits:

```python
from qiskit_aer.noise import NoiseModel as AerNoiseModel, depolarizing_error
from ketrs.qiskit import noise_model_from_aer

aer_noise = AerNoiseModel()
aer_noise.add_all_qubit_quantum_error(depolarizing_error(0.01, 1), ["h", "rx", "ry", "rz"])
aer_noise.add_all_qubit_quantum_error(depolarizing_error(0.03, 2), ["cx"])

noise = noise_model_from_aer(aer_noise)
```

---

## OpenQASM 3 & 2 (`from_qasm` / `to_qasm`)

`ketrs.from_qasm` (Rust: `ketrs::qasm::parse`) parses OpenQASM 3 and OpenQASM 2 directly in Rust:

```python
import ketrs

qasm = """
OPENQASM 3.0;
include "stdgates.inc";
input float[64] theta;
qubit[2] q;
bit[2] c;

ry(theta) q[0];
cx q[0], q[1];
c = measure q;
"""

circ = ketrs.from_qasm(qasm)
bound = circ.bind([1.57079632679])
print(ketrs.Statevector(2, seed=0).sample_circuit(bound, shots=1000))
```

Supported OpenQASM features include registers, `input` parameters, standard gates and `qelib1.inc`, custom `gate` definitions and `def` subroutines, `inv`/`ctrl(n)`/`negctrl(n)`/`pow` modifiers, `measure`, `reset`, classical assignments (`c[k] = ...;`, `bit b = ...;`), `if`, `while`, and `for` with `break` and `continue`.

Export any circuit back to standard OpenQASM 3 (readable by `from_qasm` and Qiskit's `qiskit.qasm3.loads`):

```python
print(ketrs.ghz(3).to_qasm())
```
