"""Compare ketrs with Qiskit Aer and Qiskit's NumPy statevector.

Each simulator runs the same gate sequence from |0…0⟩ and returns the final
statevector. Reports the best of several runs; ketrs timings exclude the
one-off Qiskit-to-ketrs conversion. Both ketrs and Aer use every core.

    python bench/compare.py [--max-qubits 24] [--numpy-max-qubits 20] [--gpu-max-qubits 30]

When Aer has a GPU (CUDA), a first table compares ketrs on the GPU with Aer on
the GPU at matching precision, and with cuStateVec if it is installed.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

# Aer's OpenMP threads spin-wait by default, which on this machine (Ryzen
# 5600X) slows Aer by up to 13x at 18 qubits. Passive waiting is Aer's best
# configuration here; it must be set before Aer is imported.
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector as NumpyStatevector
from qiskit_aer import AerSimulator

import ketrs
from ketrs.qiskit import from_qiskit


def qft(n: int) -> QuantumCircuit:
    """Same construction as ketrs.library.qft."""
    qc = QuantumCircuit(n)
    for j in reversed(range(n)):
        qc.h(j)
        for k in reversed(range(j)):
            qc.cp(np.pi / 2 ** (j - k), j, k)
    for i in range(n // 2):
        qc.swap(i, n - 1 - i)
    return qc


def layered(n: int, depth: int = 10) -> QuantumCircuit:
    """Same circuit as the Rust `layered_d10` benchmark."""
    qc = QuantumCircuit(n)
    for layer in range(depth):
        for q in range(n):
            qc.ry(0.1 * (layer * n + q), q)
        for q in range(n - 1):
            qc.cx(q, q + 1)
    return qc


def best_time(fn, repeats: int) -> float:
    fn()  # warm up
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times)


def run_ketrs(qc: QuantumCircuit):
    circ = from_qiskit(qc)
    return lambda: ketrs.Statevector.from_circuit(circ).amplitudes()


def run_ketrs_gpu(qc: QuantumCircuit):
    """End to end on the GPU: allocate, run, and download the statevector."""
    circ = from_qiskit(qc)
    return lambda: ketrs.GpuStatevector.from_circuit(circ).amplitudes()


def run_ketrs_gpu32(qc: QuantumCircuit):
    circ = from_qiskit(qc)
    return lambda: ketrs.GpuStatevector.from_circuit(circ, dtype="complex64").amplitudes()


def run_aer(qc: QuantumCircuit, fusion: bool, **options):
    sim = AerSimulator(method="statevector", fusion_enable=fusion, **options)
    saved = qc.copy()
    saved.save_statevector()
    return lambda: sim.run(saved).result().get_statevector()


def best_aer_gpu(qc: QuantumCircuit, repeats: int, precision: str, custatevec: bool = False):
    """Aer on the GPU, fusion on or off, whichever is faster."""
    options = {"device": "GPU", "precision": precision, "cuStateVec_enable": custatevec}
    return min(best_time(run_aer(qc, fusion, **options), repeats) for fusion in (True, False))


def has_custatevec() -> bool:
    try:
        run_aer(QuantumCircuit(2), True, device="GPU", cuStateVec_enable=True)()
        return True
    except Exception:
        return False


def run_numpy(qc: QuantumCircuit):
    return lambda: NumpyStatevector(qc).data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-qubits", type=int, default=24)
    parser.add_argument("--numpy-max-qubits", type=int, default=20)
    # Sizes above --max-qubits and up to this one run the GPU columns only.
    parser.add_argument("--gpu-max-qubits", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("results.json"))
    args = parser.parse_args()

    sizes = [n for n in (10, 14, 18, 20, 22, 24, 26) if n <= args.max_qubits]
    results = []
    gpu = ketrs.gpu_device()
    print(f"GPU: {gpu or 'none'}")
    aer_gpu = "GPU" in AerSimulator().available_devices()
    if gpu and aer_gpu:
        custatevec = has_custatevec()
        print(f"Aer GPU: yes, cuStateVec: {'yes' if custatevec else 'no'}")
        results.extend(gpu_table(sorted(set(sizes) | set(range(18, args.gpu_max_qubits + 1, 2))), custatevec))
    print("| circuit | qubits | ketrs | ketrs GPU | Aer | Aer (no fusion) | NumPy | vs best Aer | GPU vs ketrs CPU | vs NumPy |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name, build in [("qft", qft), ("layered_d10", layered)]:
        for n in sizes:
            qc = build(n)
            repeats = 5 if n <= 20 else 3
            row = {"circuit": name, "qubits": n, "gates": qc.size()}
            row["ketrs"] = best_time(run_ketrs(qc), repeats)
            row["ketrs_gpu"] = best_time(run_ketrs_gpu(qc), repeats) if gpu else None
            row["aer"] = best_time(run_aer(qc, fusion=True), repeats)
            row["aer_no_fusion"] = best_time(run_aer(qc, fusion=False), repeats)
            row["numpy"] = (
                best_time(run_numpy(qc), 1 if n > 16 else repeats)
                if n <= args.numpy_max_qubits
                else None
            )
            results.append(row)
            best_aer = min(row["aer"], row["aer_no_fusion"])
            numpy_cell = fmt(row["numpy"]) if row["numpy"] else "–"
            vs_numpy = f"{row['numpy'] / row['ketrs']:.1f}×" if row["numpy"] else "–"
            gpu_cell = fmt(row["ketrs_gpu"]) if row["ketrs_gpu"] else "–"
            gpu_vs = f"{row['ketrs'] / row['ketrs_gpu']:.2f}×" if row["ketrs_gpu"] else "–"
            print(
                f"| {name} | {n} | {fmt(row['ketrs'])} | {gpu_cell} | {fmt(row['aer'])} "
                f"| {fmt(row['aer_no_fusion'])} | {numpy_cell} "
                f"| {best_aer / row['ketrs']:.2f}× | {gpu_vs} | {vs_numpy} |",
                flush=True,
            )

    meta = {
        "cpu": platform.processor() or platform.machine(),
        "gpu": gpu,
        "python": platform.python_version(),
    }
    args.out.write_text(json.dumps({"meta": meta, "results": results}, indent=2))


def gpu_table(sizes: list[int], custatevec: bool) -> list[dict]:
    """ketrs GPU against Aer GPU at matching precision, end to end."""
    cols = ["ketrs GPU", "Aer GPU", "vs Aer GPU", "ketrs GPU f32", "Aer GPU f32", "f32 vs Aer GPU"]
    if custatevec:
        cols += ["cuStateVec", "vs cuStateVec"]
    print("| circuit | qubits | " + " | ".join(cols) + " |")
    print("|---|---|" + "---|" * len(cols))
    rows = []
    for name, build in [("qft", qft), ("layered_d10", layered)]:
        for n in sizes:
            qc = build(n)
            repeats = 5 if n <= 20 else 3
            row = {"circuit": name, "qubits": n, "gates": qc.size(), "table": "gpu"}
            row["ketrs_gpu"] = best_time(run_ketrs_gpu(qc), repeats)
            row["aer_gpu"] = best_aer_gpu(qc, repeats, "double")
            row["ketrs_gpu32"] = best_time(run_ketrs_gpu32(qc), repeats)
            row["aer_gpu32"] = best_aer_gpu(qc, repeats, "single")
            cells = [
                fmt(row["ketrs_gpu"]),
                fmt(row["aer_gpu"]),
                f"{row['aer_gpu'] / row['ketrs_gpu']:.2f}×",
                fmt(row["ketrs_gpu32"]),
                fmt(row["aer_gpu32"]),
                f"{row['aer_gpu32'] / row['ketrs_gpu32']:.2f}×",
            ]
            if custatevec:
                row["custatevec"] = best_aer_gpu(qc, repeats, "double", custatevec=True)
                cells += [fmt(row["custatevec"]), f"{row['custatevec'] / row['ketrs_gpu']:.2f}×"]
            rows.append(row)
            print(f"| {name} | {n} | " + " | ".join(cells) + " |", flush=True)
    print()
    return rows


def fmt(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:.0f} µs"
    if seconds < 1:
        return f"{seconds * 1e3:.1f} ms"
    return f"{seconds:.2f} s"


if __name__ == "__main__":
    main()
