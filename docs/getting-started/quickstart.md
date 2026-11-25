# Quick Start

This page walks through building a circuit, simulating it, sampling measurement shots, computing expectation values, and understanding `ketrs`'s qubit ordering.

## Your First Circuit

Circuit builder methods mutate the circuit in place and return it, so calls chain cleanly:

=== "Python"

    ```python
    import ketrs

    # Create a 2-qubit Bell state: (|00⟩ + |11⟩) / √2
    bell = ketrs.Circuit(2).h(0).cx(0, 1)

    # Run from |00⟩
    state = ketrs.Statevector.from_circuit(bell, seed=42)

    print("Amplitudes:   ", state.amplitudes())
    print("Probabilities:", state.probabilities())
    print("1000 shots:   ", state.sample_counts(1000))
    print("⟨ZZ⟩:         ", state.expectation("ZZ"))
    print("⟨XX⟩:         ", state.expectation("XX"))
    ```

    Expected output:

    ```text
    Amplitudes:    [0.70710678+0.j 0.        +0.j 0.        +0.j 0.70710678+0.j]
    Probabilities: [0.5 0.  0.  0.5]
    1000 shots:    {'00': 498, '11': 502}
    ⟨ZZ⟩:          1.0
    ⟨XX⟩:          1.0
    ```

=== "Rust"

    ```rust
    use ketrs::{Circuit, PauliSum, Statevector};

    fn main() {
        let mut bell = Circuit::new(2);
        bell.h(0).cx(0, 1);

        let state = Statevector::from_circuit(&bell);
        println!("Probabilities: {:?}", state.probabilities());

        let zz = PauliSum::from_list([("ZZ", 1.0)]).unwrap();
        println!("⟨ZZ⟩ = {}", state.expectation(&zz).unwrap());
    }
    ```

---

## Qubit Ordering Convention

!!! info "Qubit ordering matches Qiskit"
    **Qubit `0` is always the least significant bit** of basis-state indices and the **rightmost character** of bitstrings and Pauli labels.

| Notation | Meaning |
|---|---|
| Bitstring `"10"` | `q1 = 1, q0 = 0` (basis index `2`, i.e. `0b10`) |
| Bitstring `"01"` | `q1 = 0, q0 = 1` (basis index `1`, i.e. `0b01`) |
| Pauli label `"XZ"` | `X` on `q1`, `Z` on `q0` |
| `Circuit(3).x(0)` | Prepares `∣001⟩` (index `1`) |
| `Circuit(3).x(2)` | Prepares `∣100⟩` (index `4`) |

You can verify this directly:

```python
import ketrs

state = ketrs.Statevector.from_circuit(ketrs.Circuit(2).x(1))
print(state.sample_counts(10))  # {'10': 10}
print(state.expectation("IZ"))  # +1.0 (q0 is |0⟩)
print(state.expectation("ZI"))  # -1.0 (q1 is |1⟩)
```

---

## Reusing and Mutating States

You can also initialize a state in `|0…0⟩` (or from custom amplitudes) and apply multiple circuits sequentially:

```python
import ketrs

state = ketrs.Statevector(3, seed=0)
state.run(ketrs.Circuit(3).h(0).cx(0, 1))
state.run(ketrs.Circuit(3).cx(1, 2))

print(state.sample_counts(100))  # {'000': ~50, '111': ~50}
```

Built-in circuit generators `ketrs.ghz(n)` and `ketrs.qft(n)` are also provided for convenience.

---

## Next Steps

- Not sure whether to use `Statevector`, `GpuStatevector`, `SectorStatevector`, `DensityMatrix`, or `MPS`? Read [Choosing an Engine](engines.md).
- Already have Qiskit circuits or OpenQASM 3 files? Jump to [Qiskit & OpenQASM 3](../guides/qiskit-and-qasm.md).
