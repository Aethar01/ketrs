# Matrix Product States (MPS, PeriodicMPS & Symmetries)

For weakly entangled states on tens or hundreds of qubits, `MPS` (open chains) and `PeriodicMPS` (closed rings) replace exponential `2ⁿ` storage with tensors of bond dimension `χ`.

## Ground States with DMRG

Two-site DMRG (`MPS.dmrg`) works for Hamiltonians of arbitrary range (automatically compressing all-to-all terms into a compact MPO) and includes subspace-expansion noise by default so multi-qubit (`>2`-site) interactions cannot trap a product state:

```python
import ketrs

n = 40
terms = []
for i in range(n - 1):
    terms.append(("I" * (n - 2 - i) + "ZZ" + "I" * i, -1.0))
for i in range(n):
    terms.append(("I" * (n - 1 - i) + "X" + "I" * i, -0.5))
H = ketrs.PauliSum(terms)

mps = ketrs.MPS.random(n, bond=8, seed=0, max_bond=32)
energy, sweep_energies, converged = mps.dmrg(H, max_bond=32, cutoff=1e-12)
print(f"E0 = {energy:.8f}, converged={converged}, max bond={max(mps.bond_dimensions)}")
```

---

## Time Evolution: TDVP & TEBD

- **`mps.tdvp(H, time, steps=...)`**: two-site TDVP for Hamiltonians of **any range**. While bonds are below `max_bond`, each step automatically applies global subspace expansion (`H|ψ⟩` directions) so long-range and multi-qubit terms evolve accurately even from a product state.
- **`mps.tebd(H, time, steps=..., order=2)`**: Trotter–Suzuki TEBD with parallel updates on disjoint bonds (ideal for nearest-neighbor chains).

```python
# Evolve the state with TDVP
mps.tdvp(H, time=0.5, steps=5)
print("Truncation error:", mps.truncation_error)
```

---

## Fast Batched Expectations, Post-Selection & Sampling

When measuring many local observables on a long chain, use `mps.expectations(list)` instead of calling `mps.expectation(...)` in a loop. It builds environments in a single sweep and contracts each Pauli string only over its own support (sharing prefixes and running in parallel):

```python
z_ops = ["I" * (n - 1 - i) + "Z" + "I" * i for i in range(n)]
z_profile = mps.expectations(z_ops)

# Post-select qubit 0 onto |0⟩ and renormalise (returns outcome probability)
state_copy = mps.copy()
p0 = state_copy.project(0, 0)

# Exact sampling (up to 64 qubits for sample_counts)
counts = mps.sample_counts(1000)
```

---

## Conserved Symmetries (`Charge` & `with_charges`)

If your Hamiltonian conserves one or more Abelian `U(1)` or `Zₙ` charges (such as particle number, `S^z`, parity, or local Gauss laws in a lattice gauge theory), convert the state with `with_charges`.

Every bond is decomposed into charge sectors and DMRG / TDVP run on block-sparse tensors with a charge-graded MPO:

```python
import ketrs

n = 20
# Build an XXZ Heisenberg chain conserving total particle number (∑ Z_i)
terms = []
for i in range(n - 1):
    for p, c in [("XX", 0.5), ("YY", 0.5), ("ZZ", 1.0)]:
        terms.append(("I" * (n - 2 - i) + p + "I" * i, c))
H_xxz = ketrs.PauliSum(terms)

# Start from a Néel state (half filling: 10 qubits in |1⟩)
neel = ketrs.Circuit(n)
for q in range(0, n, 2):
    neel.x(q)

charge = ketrs.Charge.weights([1] * n)
sym = ketrs.MPS.from_circuit(neel, max_bond=64).with_charges([charge])

energy, _, converged = sym.dmrg(H_xxz, max_bond=64)
print(f"Symmetric DMRG E0 = {energy:.8f}, total charge = {sym.total_charge}")
print("Middle bond sectors:", sym.bond_sectors(n // 2))
```

Available charge constructors:

- `Charge.weights(weights)`: `U(1)` charge counting `weights[q]` when qubit `q` is `|1⟩`.
- `Charge.u1(values)`: `U(1)` charge with explicit `(val_for_0, val_for_1)` per qubit (e.g. `[(1, -1)] * n` for `2 S^z`).
- `Charge.modulo(n, values)`: `Zₙ` charge added modulo `n` (e.g. parity with `n=2`).

!!! note "Applying gates to a symmetric MPS"
    DMRG, TDVP, `expectation`, `expectations`, `project`, `sample`, and `copy` all work directly on symmetric MPS. To apply arbitrary circuit gates, convert back first via `sym.to_dense()`.

---

## Periodic Rings (`PeriodicMPS`)

For systems with periodic boundary conditions, `PeriodicMPS` represents a periodic-trace MPS `ψ(b₀…bₙ₋₁) = Tr(A[0][b₀] ⋯ A[n-1][bₙ₋₁])` with a direct `(n-1, 0)` seam bond and Pippan–White–Evertz adaptive transfer-matrix compression:

```python
import ketrs

n = 16
# Periodic transverse-field Ising ring (including the (n-1, 0) wrap-around bond)
terms = []
for i in range(n):
    j = (i + 1) % n
    chars = ["I"] * n
    chars[n - 1 - i] = "Z"
    chars[n - 1 - j] = "Z"
    terms.append(("".join(chars), -1.0))
    chars = ["I"] * n
    chars[n - 1 - i] = "X"
    terms.append(("".join(chars), -0.5))
H_ring = ketrs.PauliSum(terms)

pmps = ketrs.PeriodicMPS.random(n, bond=4, seed=0, max_bond=32)
energy, _, converged = pmps.dmrg(H_ring, max_bond=32)
print(f"Periodic ring E0 = {energy:.8f}, seam bond = {pmps.bond_dimensions[-1]}")

pmps.tdvp(H_ring, time=0.2, steps=2)
```

`PeriodicMPS` supports `dmrg`, `tdvp`, `tebd`, `run`, `with_charges`, `expectations`, `inner`, `project`, `sample_counts`, and Vulkan FP64 GPU acceleration (`pmps.gpu = True`).
