# Hamiltonians, Evolution, Ground States & VQE

`ketrs` operates directly on symbolic Pauli sums (`PauliSum`) without building an explicit `2ⁿ × 2ⁿ` matrix, enabling fast expectation values, exact Krylov time evolution, matrix-free Lanczos diagonalization, and adjoint differentiation.

## Building Hamiltonians (`PauliSum`)

Construct a Hermitian operator from `(label, coefficient)` pairs (rightmost character is qubit `0`):

```python
import ketrs

# 6-qubit transverse-field Ising model: H = -∑ Z_i Z_{i+1} - 0.5 ∑ X_i
n = 6
terms = []
for i in range(n - 1):
    terms.append(("I" * (n - 2 - i) + "ZZ" + "I" * i, -1.0))
for i in range(n):
    terms.append(("I" * (n - 1 - i) + "X" + "I" * i, -0.5))

H = ketrs.PauliSum(terms).simplify()
print(H, "norm bound:", H.coefficient_norm())
```

Anywhere an observable is expected (`expectation`, `evolve`, `dmrg`, `tdvp`, `lowest_eigenstates`), you may also pass a single string like `"ZZ"` or a plain list of `(label, coeff)` pairs.

### Sparse Matrix Operators (`SparseOperator`)

If your Hermitian operator is sparse in the computational basis but expensive as a Pauli sum, wrap a SciPy CSR matrix (or dense NumPy array) directly:

```python
import scipy.sparse as sp
import ketrs

mat = sp.diags([1.0, -1.0, -1.0, 1.0], format="csr")
op = ketrs.SparseOperator.from_scipy(mat)

state = ketrs.Statevector.from_circuit(ketrs.Circuit(2).h(0).cx(0, 1))
print(state.expectation(op))  # 1.0
```

`SparseOperator` is supported by `Statevector.expectation`, `apply_pauli_sum`, and `evolve`.

---

## Exact Time Evolution (`evolve`)

`Statevector.evolve`, `SectorStatevector.evolve`, and `GpuStatevector.evolve` compute `exp(-iHt)|ψ⟩` to machine precision via matrix-free Krylov time-stepping:

```python
import ketrs

psi = ketrs.Statevector(6)
psi.evolve(H, time=1.5)
print("Energy after quench:", psi.expectation(H))
```

For Trotterized circuit evolution instead of exact Krylov evolution, use `Circuit.pauli_evolution(H, time, steps=..., order=...)`.

---

## Exact Ground & Excited States (`lowest_eigenstates`)

Find the `k` lowest eigenstates by matrix-free Lanczos iteration with deflation:

```python
import ketrs

states = ketrs.lowest_eigenstates(H, n_qubits=6, k=2)
for idx, eig in enumerate(states):
    print(f"E_{idx} = {eig.energy:.10f}  (residual {eig.residual:.1e})")
```

### Restricting to a Symmetry Sector (`SectorBasis`)

When `H` conserves particle number, spin projection, or charge, restrict both time evolution and Lanczos diagonalization to the fixed Hamming-weight subspace:

```python
import ketrs

# Half filling on 12 qubits: C(12, 6) = 924 states instead of 4096
basis = ketrs.SectorBasis(12, [(range(12), 6)])
print("Sector dimension:", basis.dim)

# Start in a basis state inside the sector (e.g. Néel state 0b010101010101)
neel = sum(1 << q for q in range(0, 12, 2))
psi = ketrs.SectorStatevector(basis, neel)
# psi.evolve(H_conserving, time=1.0)

# Or diagonalize directly inside the sector:
# ground = ketrs.lowest_eigenstates(H_conserving, n_qubits=12, k=1, sector=basis)[0]
```

You can pass multiple `(qubits, weight)` constraints to `SectorBasis`, such as separate spin-up and spin-down particle numbers for a Fermi–Hubbard model.

---

## Variational Circuits & Adjoint Gradients

`expectation_gradient` computes both `⟨ψ(θ)|H|ψ(θ)⟩` and its exact gradient `∇⟨H⟩` in one adjoint pass (costing ~2.3–3.5 forward evaluations regardless of the number of parameters):

```python
import numpy as np
from scipy.optimize import minimize
import ketrs

n = 4
ansatz = ketrs.Circuit(n)
params = ansatz.parameters(2 * n)
for q in range(n):
    ansatz.ry(params[q], q)
for q in range(n - 1):
    ansatz.cx(q, q + 1)
for q in range(n):
    ansatz.ry(params[n + q], q)

H4 = ketrs.PauliSum([
    ("IIZZ", -1.0), ("IZZI", -1.0), ("ZZII", -1.0),
    ("IIIX", -0.5), ("IIXI", -0.5), ("IXII", -0.5), ("XIII", -0.5),
])

zero = ketrs.Statevector(n)

def loss_and_grad(x):
    val, grad = zero.expectation_gradient(ansatz, x.tolist(), H4)
    return val, grad

res = minimize(loss_and_grad, np.zeros(2 * n), jac=True, method="BFGS")
print(f"VQE energy: {res.fun:.8f}")
```

For custom losses (such as state overlap `|⟨φ|ψ(θ)⟩|²`), use `Statevector.adjoint_gradient(circuit, parameters, final_state, cotangent)`. Both methods also exist on `GpuStatevector`.
