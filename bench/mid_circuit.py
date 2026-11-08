"""Compare ketrs with Qiskit Aer on circuits with mid-circuit measurement.

Each simulator samples the same circuit, with measurements, resets and
classical control in the middle, for a number of shots from |0…0⟩ and
returns the counts. Aer runs in its faster configuration at each size: with
or without its shot branching (off by default). Reports the best of several
runs; ketrs timings exclude the one-off Qiskit-to-ketrs conversion.

    python bench/mid_circuit.py [--max-qubits 20] [--shots 1000]
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")  # see compare.py

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator

import ketrs
import time

from compare import best_time, fmt
from ketrs.qiskit import from_qiskit


def layers(qc: QuantumCircuit, depth: int) -> None:
    """`depth` layers of Ry on every qubit and a CNOT chain, as in the Rust
    `mid_circuit` benchmarks."""
    n = qc.num_qubits
    for layer in range(depth):
        for q in range(n):
            qc.ry(0.3 + 0.1 * (layer * n + q), q)
        for q in range(n - 1):
            qc.cx(q, q + 1)


def new_circuit(n: int) -> QuantumCircuit:
    return QuantumCircuit(QuantumRegister(n), ClassicalRegister(n))


def feed_forward(n: int) -> QuantumCircuit:
    """Layers, two measurements with corrections conditioned on them, more
    layers, then every qubit measured."""
    qc = new_circuit(n)
    layers(qc, 4)
    qc.measure(0, 0)
    qc.measure(1, 1)
    with qc.if_test((qc.clbits[0], 1)):
        qc.z(2)
    with qc.if_test((qc.clbits[1], 1)):
        qc.x(2)
    layers(qc, 4)
    qc.measure(range(n), range(n))
    return qc


def repeat_until_success(n: int) -> QuantumCircuit:
    """Re-prepare and measure qubit 0, entangled with qubit 1, until it reads
    1; then more layers and every qubit measured."""
    qc = new_circuit(n)
    layers(qc, 2)
    with qc.while_loop((qc.clbits[0], 0)):
        qc.reset(0)
        qc.ry(1.0, 0)
        qc.cx(0, 1)
        qc.measure(0, 0)
    layers(qc, 4)
    qc.measure(range(n), range(n))
    return qc


def run_ketrs(qc: QuantumCircuit, shots: int):
    circ = from_qiskit(qc)
    state = ketrs.Statevector(qc.num_qubits, seed=1)
    return lambda: state.sample_circuit(circ, shots)


def run_aer(qc: QuantumCircuit, shots: int, branching: bool):
    sim = AerSimulator(method="statevector", shot_branching_enable=branching, seed_simulator=1)
    return lambda: sim.run(qc, shots=shots).result().get_counts()


def once(fn, _repeats: int) -> float:
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-qubits", type=int, default=20)
    parser.add_argument("--shots", type=int, default=1000)
    args = parser.parse_args()

    sizes = [n for n in (12, 16, 20) if n <= args.max_qubits]
    print(f"{args.shots} shots")
    print("| circuit | qubits | ketrs | Aer | Aer (shot branching) | vs best Aer |")
    print("|---|---|---|---|---|---|")
    for name, build in [("feed_forward", feed_forward), ("repeat_until_success", repeat_until_success)]:
        for n in sizes:
            qc = build(n)
            # At 20 qubits one Aer run takes minutes: time it once, cold.
            timed = best_time if n <= 16 else once
            ketrs_time = best_time(run_ketrs(qc, args.shots), 5)
            aer = timed(run_aer(qc, args.shots, branching=False), 3)
            aer_branching = timed(run_aer(qc, args.shots, branching=True), 3)
            best_aer = min(aer, aer_branching)
            print(
                f"| {name} | {n} | {fmt(ketrs_time)} | {fmt(aer)} | {fmt(aer_branching)} "
                f"| {best_aer / ketrs_time:.2f}× |",
                flush=True,
            )


if __name__ == "__main__":
    main()
