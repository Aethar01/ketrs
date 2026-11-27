"""Steane-style GKP bosonic quantum error correction with ketrs.GaussianMixture.

Simulates finite-energy square GKP qubits (ħ = 2, grid spacing d = √(2π))
undergoing Gaussian random-displacement (additive quadrature) noise, comparing
uncorrected decay against repeated Steane syndrome-extraction rounds using
|+_GKP⟩ (for x-errors) and |0_GKP⟩ (for p-errors) ancillas.

Run:
    python examples/gkp_qec.py
"""

from __future__ import annotations

import math
import time

import ketrs

D_GRID = math.sqrt(2.0 * math.pi)


def wrap(x: float, period: float) -> float:
    """Wrap `x` into `(-period / 2, period / 2]`."""
    r = x % period
    if r > 0.5 * period:
        r -= period
    return r


def is_logical_zero(q: float) -> bool:
    """Nearest-peak logical Z decoder on homodyne x outcome `q`."""
    return abs(wrap(q, 2.0 * D_GRID)) < 0.5 * D_GRID


def run_trial_uncorrected(
    epsilon: float, noise_var: float, rounds: int, seed: int
) -> bool:
    """Return True if a logical X error occurred after `rounds` noise steps."""
    c = ketrs.ModeCircuit(1).set_input(0, ketrs.ModeState.gkp(epsilon, 0.0, 0.0))
    st = ketrs.GaussianMixture.from_circuit(c, seed=seed)
    st.apply_channel(ketrs.CvChannel.additive_noise(rounds * noise_var), 0)
    q = st.measure_homodyne(0.0, 0)
    return not is_logical_zero(q)


def run_trial_steane_qec(
    epsilon: float, noise_var: float, rounds: int, seed: int
) -> bool:
    """Return True if a logical X error occurred with Steane x-QEC between rounds."""
    # Mode 0: data |0_GKP⟩; Mode 1..rounds: fresh |+_GKP⟩ ancillas (1 ancilla per
    # inter-round QEC step, reusing mode 1 sequentially by resetting fresh circuits
    # via 2-mode rounds). Here we use 1 QEC round between 2 noise rounds (2 modes).
    if rounds != 2:
        raise ValueError("this example runs 2 noise rounds with 1 Steane QEC step")
    c = (
        ketrs.ModeCircuit(2)
        .set_input(0, ketrs.ModeState.gkp(epsilon, 0.0, 0.0))
        .set_input(1, ketrs.ModeState.gkp(epsilon, 0.5 * math.pi, 0.0))
    )
    st = ketrs.GaussianMixture.from_circuit(c, seed=seed)

    # Round 1 noise on data mode 0
    st.apply_channel(ketrs.CvChannel.additive_noise(noise_var), 0)

    # Steane x-syndrome extraction: CX(1, data=0, ancilla=1) maps x_1 -> x_1 + x_0.
    # Since ancilla |+_GKP⟩ has x_1 ≡ 0 (mod √(2π)), measuring x_1 gives e_x mod √(2π).
    st.cx(1.0, 0, 1)
    m_x = st.measure_homodyne(0.0, 1)
    corr_x = -wrap(m_x, D_GRID)
    st.displace(0.5 * corr_x, 0.0, 0)

    # Round 2 noise + logical Z readout
    st.apply_channel(ketrs.CvChannel.additive_noise(noise_var), 0)
    q_final = st.measure_homodyne(0.0, 0)
    return not is_logical_zero(q_final)


def main() -> None:
    epsilon = 0.06  # ~12.2 dB GKP squeezing
    shots = 200
    rounds = 2
    sq_db = -10.0 * math.log10(math.tanh(epsilon))

    print(f"GKP Steane x-QEC (ε = {epsilon}, {sq_db:.1f} dB, {shots} shots/point)")
    print(f"{'σ²/round':>9} | {'raw err':>9} | {'QEC err':>9} | {'reduction':>9} | {'time':>8}")
    print("-" * 57)

    for idx, noise_var in enumerate((0.10, 0.16, 0.22, 0.30)):
        t0 = time.perf_counter()
        raw_err = sum(
            run_trial_uncorrected(epsilon, noise_var, rounds, seed=10_000 * idx + s)
            for s in range(shots)
        )
        qec_err = sum(
            run_trial_steane_qec(epsilon, noise_var, rounds, seed=20_000 * idx + s)
            for s in range(shots)
        )
        dt_ms = (time.perf_counter() - t0) * 1e3
        raw_rate = raw_err / shots
        qec_rate = qec_err / shots
        red = f"{raw_rate / qec_rate:6.2f}x" if qec_rate > 0 else "   inf"
        print(
            f"{noise_var:9.2f} | {raw_rate:8.1%} | {qec_rate:8.1%} | {red:>9} | {dt_ms:6.1f} ms"
        )


if __name__ == "__main__":
    main()
