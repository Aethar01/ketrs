"""Benchmark the MPS engine against Aer's matrix_product_state method.

Both run the same second-order Trotter circuit for a transverse-field Ising
chain from |0…0⟩ with the same bond-dimension cap, and report ⟨Z⟩ on the
middle qubit so the results can be compared as well as timed.

    python bench/mps.py
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")  # see compare.py

import time

from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import SuzukiTrotter
from qiskit_aer import AerSimulator

import ketrs
from ketrs.qiskit import from_qiskit


def ising(n: int) -> SparsePauliOp:
    return SparsePauliOp.from_sparse_list(
        [("ZZ", [q, q + 1], -1.0) for q in range(n - 1)] + [("X", [q], -1.2) for q in range(n)],
        num_qubits=n,
    )


def run(n: int, steps: int, max_bond: int) -> None:
    qc = QuantumCircuit(n)
    qc.append(PauliEvolutionGate(ising(n), time=0.1 * steps, synthesis=SuzukiTrotter(order=2, reps=steps)), range(n))
    middle = SparsePauliOp.from_sparse_list([("Z", [n // 2], 1.0)], num_qubits=n)

    circ = from_qiskit(qc)
    start = time.perf_counter()
    mps = ketrs.MPS.from_circuit(circ, max_bond=max_bond, cutoff=1e-12)
    z_ketrs = mps.expectation([(_label(n, n // 2), 1.0)])
    t_ketrs = time.perf_counter() - start

    sim = AerSimulator(
        method="matrix_product_state",
        matrix_product_state_max_bond_dimension=max_bond,
        matrix_product_state_truncation_threshold=1e-12,
    )
    aer_circ = _transpile(qc)
    aer_circ.save_expectation_value(middle, range(n))
    start = time.perf_counter()
    z_aer = sim.run(aer_circ).result().data()["expectation_value"]
    t_aer = time.perf_counter() - start

    print(
        f"| {n} | {steps} | {max_bond} | {max(mps.bond_dimensions)} | {mps.truncation_error:.1e} "
        f"| {t_ketrs:.2f} s | {t_aer:.2f} s | {t_aer / t_ketrs:.1f}× | {abs(z_ketrs - z_aer):.1e} |",
        flush=True,
    )


def _label(n: int, q: int) -> str:
    chars = ["I"] * n
    chars[n - 1 - q] = "Z"
    return "".join(chars)


def _transpile(qc):
    """Synthesise the evolution into gates Aer's MPS method supports, without
    Aer's target (which caps width at 63 qubits)."""
    from qiskit import transpile

    return transpile(qc, basis_gates=["rzz", "rx", "rz", "h", "cx"], optimization_level=0)


if __name__ == "__main__":
    print("| qubits | steps | χ cap | χ used | ketrs trunc. error | ketrs | Aer MPS | speedup | ⟨Z⟩ diff |")
    print("|---|---|---|---|---|---|---|---|---|")
    for n, steps, bond in [(50, 10, 32), (100, 10, 32), (50, 20, 64), (100, 20, 64), (100, 40, 128)]:
        run(n, steps, bond)
