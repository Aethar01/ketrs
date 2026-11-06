"""The lattice Schwinger model with ketrs.

Kogut–Susskind staggered fermions on N sites, Jordan–Wigner mapped to
qubits, with the gauge field eliminated through Gauss's law (open
boundaries):

    H = w Σₙ (XₙXₙ₊₁ + YₙYₙ₊₁)/2 + (m/2) Σₙ (−1)ⁿ Zₙ + J Σₙ Lₙ²,
    Lₙ = ε₀ + ½ Σ_{k≤n} (Z_k + (−1)^k).

The electric energy makes every pair of sites interact (all-to-all ZZ). H
conserves the charge Σ Zₙ, so the physical, charge-neutral states form a
symmetry sector of half filling.

Two calculations, each validated against an independent method:

1. Particle production after a quench from the bare vacuum, computed
   exactly within the neutral sector, and checked against a Trotter circuit.
2. The vacuum's particle density as the lattice grows, by DMRG (the
   all-to-all terms cost an O(N) MPO bond), checked against exact sector
   diagonalisation where that is feasible.

    python examples/schwinger/schwinger.py
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import ketrs

W, MASS, J, THETA = 1.0, 0.5, 1.0, 0.0  # hopping, mass, coupling, background field


def label(n_sites: int, ops: dict[int, str]) -> str:
    """A Pauli label, with qubit 0 rightmost."""
    chars = ["I"] * n_sites
    for q, p in ops.items():
        chars[n_sites - 1 - q] = p
    return "".join(chars)


def hamiltonian(n_sites: int) -> ketrs.PauliSum:
    terms: dict[str, float] = defaultdict(float)
    for n in range(n_sites - 1):
        terms[label(n_sites, {n: "X", n + 1: "X"})] += W / 2
        terms[label(n_sites, {n: "Y", n + 1: "Y"})] += W / 2
    for n in range(n_sites):
        terms[label(n_sites, {n: "Z"})] += MASS / 2 * (-1) ** n
    # Lₙ = cₙ + ½ Σ_{k≤n} Z_k, with cₙ = ε₀ + ½ Σ_{k≤n} (−1)^k, so
    # Lₙ² = cₙ² + (n+1)/4 + cₙ Σ_k Z_k + ½ Σ_{k<l} Z_k Z_l.
    for n in range(n_sites - 1):
        c = THETA + 0.5 * sum((-1) ** k for k in range(n + 1))
        terms[label(n_sites, {})] += J * (c * c + (n + 1) / 4)
        for k in range(n + 1):
            terms[label(n_sites, {k: "Z"})] += J * c
            for l in range(k + 1, n + 1):
                terms[label(n_sites, {k: "Z", l: "Z"})] += J / 2
    return ketrs.PauliSum([(lab, c) for lab, c in terms.items() if c != 0.0])


def particle_density(n_sites: int) -> ketrs.PauliSum:
    """ν = (1/2N) Σₙ ((−1)ⁿ Zₙ + 1): zero in the bare vacuum."""
    terms = [(label(n_sites, {n: "Z"}), (-1) ** n / (2 * n_sites)) for n in range(n_sites)]
    return ketrs.PauliSum(terms + [(label(n_sites, {}), 0.5)])


def bare_vacuum(n_sites: int) -> int:
    """Even sites |1⟩ (Z = −1), odd sites |0⟩: no particles, Lₙ = ε₀."""
    return sum(1 << n for n in range(0, n_sites, 2))


def neutral_sector(n_sites: int) -> ketrs.SectorBasis:
    return ketrs.SectorBasis(n_sites, [(range(n_sites), n_sites // 2)])


def quench(n_sites: int, times: np.ndarray) -> np.ndarray:
    """ν(t) after starting in the bare vacuum, exactly, in the neutral sector."""
    h, nu = hamiltonian(n_sites), particle_density(n_sites)
    state = ketrs.SectorStatevector(neutral_sector(n_sites), bare_vacuum(n_sites))
    out, now = [], 0.0
    for t in times:
        state.evolve(h, t - now)
        now = t
        out.append(state.expectation(nu))
    return np.array(out)


def trotter_quench(n_sites: int, times: np.ndarray, dt: float) -> np.ndarray:
    """The same quench as a second-order Trotter circuit on a statevector."""
    h, nu = hamiltonian(n_sites), particle_density(n_sites)
    state = ketrs.Statevector(n_sites)
    state.run(_prepare(n_sites))
    out, now = [], 0.0
    for t in times:
        steps = round((t - now) / dt)
        if steps:
            state.run(ketrs.Circuit(n_sites).pauli_evolution(h, t - now, steps=steps, order=2))
        now = t
        out.append(state.expectation(nu))
    return np.array(out)


def _prepare(n_sites: int) -> ketrs.Circuit:
    circ = ketrs.Circuit(n_sites)
    for n in range(0, n_sites, 2):
        circ.x(n)
    return circ


def vacuum_density_dmrg(n_sites: int, max_bond: int) -> tuple[float, float]:
    """Ground-state energy and particle density by DMRG from the bare vacuum
    (which keeps DMRG in the neutral sector)."""
    mps = ketrs.MPS(n_sites, max_bond=max_bond)
    mps.run(_prepare(n_sites))
    energy, _, converged = mps.dmrg(hamiltonian(n_sites), max_bond=max_bond, tolerance=1e-9)
    if not converged:
        print(f"  warning: DMRG at N={n_sites} did not converge")
    return energy, mps.expectation(particle_density(n_sites))


def vacuum_density_exact(n_sites: int) -> tuple[float, float]:
    (ground,) = ketrs.lowest_eigenstates(
        hamiltonian(n_sites), n_sites, k=1, dtype="real", sector=neutral_sector(n_sites)
    )
    return ground.energy, ground.state.expectation(particle_density(n_sites))


def main() -> None:
    out_dir = Path(__file__).parent
    fig, (ax_quench, ax_vacuum) = plt.subplots(1, 2, figsize=(11, 4.2))

    # 1. Particle production after a quench.
    times = np.linspace(0, 4, 81)
    for n_sites in (8, 12, 16, 20):
        start = time.perf_counter()
        nu = quench(n_sites, times)
        dim = neutral_sector(n_sites).dim
        print(f"quench N={n_sites}: sector dim {dim} (vs {2**n_sites}), {time.perf_counter() - start:.2f} s")
        ax_quench.plot(times, nu, label=f"N = {n_sites} (exact)")
    trotter_n = 12
    trotter = trotter_quench(trotter_n, times[::4], dt=0.05)
    exact = quench(trotter_n, times[::4])
    print(f"Trotter (dt = 0.05, 2nd order) vs exact at N={trotter_n}: max |Δν| = {np.abs(trotter - exact).max():.1e}")
    ax_quench.plot(times[::4], trotter, "k.", label=f"N = {trotter_n}, Trotter circuit")
    ax_quench.set(xlabel="time t", ylabel="particle density ν(t)", title="Particle production after a quench")
    ax_quench.legend(fontsize=8)

    # 2. Vacuum particle density against lattice size.
    sizes = [8, 12, 16, 20, 24, 32, 40, 60, 80, 100]
    dmrg = []
    for n_sites in sizes:
        start = time.perf_counter()
        dmrg.append(vacuum_density_dmrg(n_sites, max_bond=32))
        print(f"DMRG N={n_sites}: E0 = {dmrg[-1][0]:.8f}, ν0 = {dmrg[-1][1]:.6f}, {time.perf_counter() - start:.2f} s")
    exact_sizes = [n for n in sizes if n <= 20]
    exact = [vacuum_density_exact(n) for n in exact_sizes]
    for n_sites, (e_exact, _) in zip(exact_sizes, exact):
        e_dmrg, _ = dmrg[sizes.index(n_sites)]
        print(f"  N={n_sites}: |E0(DMRG) − E0(exact sector)| = {abs(e_dmrg - e_exact):.1e}")
    ax_vacuum.plot(sizes, [d[1] for d in dmrg], "o-", label="DMRG (χ = 32)")
    ax_vacuum.plot(exact_sizes, [nu for _, nu in exact], "kx", ms=9, label="exact (neutral sector)")
    ax_vacuum.set(xlabel="lattice sites N", ylabel="vacuum particle density ν₀", title="Interacting vacuum")
    ax_vacuum.legend(fontsize=8)

    fig.suptitle(f"Lattice Schwinger model (w = {W}, m = {MASS}, J = {J}, ε₀ = {THETA})")
    fig.tight_layout()
    path = out_dir / "schwinger.png"
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
