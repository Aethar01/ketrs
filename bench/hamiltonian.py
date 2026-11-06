"""Benchmark Hamiltonian workloads: Trotter circuits, exact evolution and
ground states.

- Trotter: a Heisenberg-chain PauliEvolutionGate run by ketrs as one-pass
  Pauli rotations, by ketrs on the CNOT-ladder decomposition, and by Aer.
- Evolve: exact exp(-iHt)|ψ⟩, ketrs ``evolve`` vs scipy ``expm_multiply``.
- Ground state: ketrs ``lowest_eigenstates`` vs scipy ``eigsh`` (ARPACK) on
  the sparse matrix (matrix construction not timed).

    python bench/hamiltonian.py
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")  # see compare.py

import time

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import LieTrotter
from qiskit_aer import AerSimulator
from scipy.sparse.linalg import eigsh, expm_multiply

import ketrs
from ketrs.qiskit import from_qiskit, pauli_sum_from_qiskit


def heisenberg(n: int, field: float = 0.5) -> SparsePauliOp:
    terms = []
    for q in range(n - 1):
        for p in "XYZ":
            terms.append((p + p, [q, q + 1], 1.0))
    terms += [("Z", [q], field) for q in range(n)]
    return SparsePauliOp.from_sparse_list(terms, num_qubits=n)


def best(fn, repeats=3) -> float:
    fn()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times)


def fmt(s: float) -> str:
    return f"{s * 1e3:.1f} ms" if s < 1 else f"{s:.2f} s"


def trotter(n: int, steps: int = 10) -> None:
    qc = QuantumCircuit(n)
    qc.h(range(n))
    gate = PauliEvolutionGate(heisenberg(n), time=1.0, synthesis=LieTrotter(reps=steps))
    qc.append(gate, range(n))
    ladders = transpile(qc, basis_gates=["cx", "rz", "h", "sx", "x", "rx", "ry"], optimization_level=0)
    direct, decomposed = from_qiskit(qc), from_qiskit(ladders)
    t_direct = best(lambda: ketrs.Statevector.from_circuit(direct).amplitudes())
    t_ladder = best(lambda: ketrs.Statevector.from_circuit(decomposed).amplitudes())
    sim = AerSimulator(method="statevector")
    saved = qc.copy()
    saved.save_statevector()
    # Aer needs the evolution synthesised; it has native rxx/ryy/rzz gates.
    saved = transpile(saved, sim)
    t_aer = best(lambda: sim.run(saved).result().get_statevector())
    print(
        f"| trotter | {n} | {len(direct)} rotations vs {len(decomposed)} gates | "
        f"{fmt(t_direct)} | {fmt(t_ladder)} ({t_ladder / t_direct:.1f}×) | {fmt(t_aer)} ({t_aer / t_direct:.1f}×) |",
        flush=True,
    )


def evolve(n: int) -> None:
    op = heisenberg(n)
    h, matrix = pauli_sum_from_qiskit(op), op.to_matrix(sparse=True)
    psi0 = np.zeros(2**n, dtype=complex)
    psi0[0b0101 % 2**n] = 1.0

    def run_ketrs():
        state = ketrs.Statevector.from_amplitudes(psi0)
        state.evolve(h, 1.0)
        return state

    t_ketrs = best(run_ketrs)
    t_scipy = best(lambda: expm_multiply(-1j * matrix, psi0), repeats=1)
    err = np.abs(run_ketrs().amplitudes() - expm_multiply(-1j * matrix, psi0)).max()
    print(f"| evolve | {n} | max diff {err:.0e} | {fmt(t_ketrs)} | scipy {fmt(t_scipy)} ({t_scipy / t_ketrs:.1f}×) | |", flush=True)


def ground(n: int) -> None:
    op = heisenberg(n)
    h, matrix = pauli_sum_from_qiskit(op), op.to_matrix(sparse=True).real
    t_ketrs = best(lambda: ketrs.lowest_eigenstates(h, n, dtype="real"), repeats=1)
    t_scipy = best(lambda: eigsh(matrix, k=1, which="SA"), repeats=1)
    e_ketrs = ketrs.lowest_eigenstates(h, n, dtype="real")[0].energy
    e_scipy = eigsh(matrix, k=1, which="SA")[0][0]
    print(
        f"| ground state | {n} | ΔE {abs(e_ketrs - e_scipy):.0e} | {fmt(t_ketrs)} | "
        f"scipy {fmt(t_scipy)} ({t_scipy / t_ketrs:.1f}×) | |",
        flush=True,
    )


if __name__ == "__main__":
    print("| workload | qubits | detail | ketrs | baseline | Aer |")
    print("|---|---|---|---|---|---|")
    for n in (16, 20, 22):
        trotter(n)
    for n in (14, 18, 20):
        evolve(n)
    for n in (14, 18, 20):
        ground(n)
