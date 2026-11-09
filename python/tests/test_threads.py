"""ketrs from several threads at once. On free-threaded Python (3.14t and
later) these run in parallel without the GIL; elsewhere they exercise the
calls that release it. Needs only numpy, so it runs where Qiskit doesn't."""

import sys
import sysconfig
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import ketrs

THREADS = 8


@pytest.mark.skipif(
    not sysconfig.get_config_var("Py_GIL_DISABLED"), reason="needs free-threaded Python"
)
def test_import_keeps_gil_disabled():
    assert not sys._is_gil_enabled()


def circuit(n: int, seed: int) -> ketrs.Circuit:
    rng = np.random.default_rng(seed)
    circ = ketrs.Circuit(n)
    for _ in range(4):
        for q in range(n):
            circ.ry(float(rng.uniform(0, np.pi)), q)
        for q in range(n - 1):
            circ.cx(q, q + 1)
    return circ


def run_together(work):
    """Call `work(i)` on THREADS threads released at the same moment."""
    start = threading.Barrier(THREADS)

    def task(i):
        start.wait()
        return work(i)

    with ThreadPoolExecutor(THREADS) as pool:
        return list(pool.map(task, range(THREADS)))


def test_parallel_simulations_match_serial():
    def simulate(i):
        return [
            ketrs.Statevector.from_circuit(circuit(12, 10 * i + k)).amplitudes()
            for k in range(10)
        ]

    want = [simulate(i) for i in range(THREADS)]
    got = run_together(simulate)
    for g, w in zip(got, want):
        for a, b in zip(g, w):
            np.testing.assert_array_equal(a, b)


def test_shared_state_reads():
    state = ketrs.Statevector.from_circuit(circuit(14, 1))
    observable = [("Z" * 14, 1.0), ("X" + "I" * 13, 0.5)]
    want = (state.expectation(observable), state.probabilities())

    def read(_):
        return [(state.expectation(observable), state.probabilities()) for _ in range(20)]

    for results in run_together(read):
        for value, probs in results:
            assert value == want[0]
            np.testing.assert_array_equal(probs, want[1])
