# Circuits, Parameters & Optimization

`ketrs.Circuit` provides a chainable builder API covering standard 1-qubit and 2-qubit gates, arbitrary 1- and 2-qubit unitaries, native multi-controlled gates, multi-qubit Pauli rotations, Trotter–Suzuki evolution, parameters, and lossless circuit optimization.

## Building Circuits

```python
import math
import ketrs

c = ketrs.Circuit(4)
c.h(0).s(0).t(1)
c.rx(math.pi / 4, 2).ry(0.5, 3).rz(-0.3, 1)
c.cx(0, 1).cz(1, 2).swap(2, 3)

# Multi-controlled gates and swaps run natively on every engine
c.ccx(0, 1, 2)
c.mcx([0, 1, 2], 3)
c.cswap(0, 1, 2)
```

### Pauli Rotations & Trotter–Suzuki Evolution

Pauli rotations $\exp(-i \frac{\theta}{2} P)$ of any weight are applied in a single pass rather than decomposed into CNOT ladders:

```python
import ketrs

c = ketrs.Circuit(4)
c.pauli_rotation(0.4, "XZYI")                  # acts on all 4 qubits (rightmost is q0)
c.pauli_rotation(0.4, "XZ", qubits=[0, 3])     # Z on q0, X on q3
c.rxx(0.2, 0, 1).ryy(0.2, 1, 2).rzz(0.2, 2, 3)

# Order-2 (Strang) or any higher even-order Suzuki–Trotter formula
H = ketrs.PauliSum([("IIZZ", -1.0), ("IZZI", -1.0), ("IIIX", -0.5)])
trotter = ketrs.Circuit(4).pauli_evolution(H, time=1.0, steps=4, order=2)
```

### Circuit Transforms

- `circ.inverse()`: returns the adjoint circuit.
- `circ.control([c0, c1])`: controls the entire circuit on `c0, c1` (preserving parameters).
- `circ.extend(other)`: appends another circuit of the same width in place.

---

## Parameters & Binding

Create parameters with `circ.parameter()` or `circ.parameters(n)`. Any rotation angle accepts a parameter or an affine expression (`a * theta + b`):

```python
import ketrs

c = ketrs.Circuit(2)
a, b = c.parameters(2)
c.ry(a, 0).rz(2.0 * a + 0.5, 0).crx(-b / 2, 0, 1)

bound = c.bind([0.3, 1.2])
state = ketrs.Statevector.from_circuit(bound)
```

For computing exact gradients with respect to parameters, see [VQE & Adjoint Gradients](hamiltonians.md#variational-circuits-adjoint-gradients).

---

## Lossless Circuit Optimization

`Circuit.optimize` simplifies circuits without changing their unitary (up to `tolerance`), supporting levels `0` through `3` and two target bases (`"native"` or `"cx"`):

| Level | Passes included |
|---|---|
| `0` | No-op copy. |
| `1` | Identity removal, adjacent gate cancellation, 1-qubit run consolidation. |
| `2` *(default)* | Commutation cancellation, local Clifford/SWAP rewrites, exact 2-qubit KAK synthesis (preserving real $\mathrm{SO}(4)$ blocks when `preserve_real=True`), smart Pauli-gadget lowering under `basis="cx"`. |
| `3` | Multi-qubit Clifford/Pauli propagation, Pauli-rotation conjugation, and $\mathrm{GF}(2)$ CNOT subcircuit optimization. |

```python
import ketrs

H = ketrs.PauliSum([
    ("XXII", 0.5), ("YYII", 0.5),
    ("IIXX", 0.5), ("IIYY", 0.5),
    ("ZZZZ", 0.25),
])
circ = ketrs.Circuit(4).pauli_evolution(H, time=0.5, steps=2, order=2)

opt = circ.optimize(level=3, basis="cx")
print("Total depth:", circ.depth(), "->", opt.depth())
print("2Q depth:   ", circ.two_qubit_depth(), "->", opt.two_qubit_depth())
print("Gate counts:", opt.gate_counts())
```

!!! tip "Real statevector compatibility"
    By default (`preserve_real=None`), `basis="native"` preserves realness so optimized real circuits still run on `dtype="real"` / `"float32"`. Pass `preserve_real=True` explicitly if you also want `basis="cx"` to stay real.
