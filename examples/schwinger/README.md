# Lattice Schwinger model

The Schwinger model is quantum electrodynamics in one space dimension, a
standard testbed for quantum simulation of gauge theories. This example
simulates its lattice form with three of ketrs's engines and checks each
result against an independent method.

![Particle production and vacuum density](schwinger.png)

## Model

Kogut–Susskind staggered fermions on $`N`$ sites, mapped to qubits by
Jordan–Wigner, with the gauge field removed using Gauss's law (open
boundaries):

```math
H = \frac{w}{2} \sum_n \left( X_n X_{n+1} + Y_n Y_{n+1} \right)
  + \frac{m}{2} \sum_n (-1)^n Z_n
  + J \sum_n L_n^2,
\qquad
L_n = \varepsilon_0 + \frac{1}{2} \sum_{k \le n} \left( Z_k + (-1)^k \right)
```

Squaring the electric field $`L_n`$ couples every pair of sites (all-to-all
$`ZZ`$ terms). $`H`$ conserves the charge $`\sum_n Z_n`$; physical states are
neutral, which is the half-filling symmetry sector. Here $`w = J = 1`$,
$`m = 0.5`$, $`\varepsilon_0 = 0`$.

## What it computes

**Left: particle production after a quench.** Starting from the bare vacuum
(no particles), the particle density

```math
\nu(t) = \frac{1}{2N} \sum_n \left( (-1)^n \langle Z_n \rangle_t + 1 \right)
```

oscillates as pairs are created and annihilated. Curves are exact time evolution
(`SectorStatevector.evolve`, Krylov) within the neutral sector, which at
$`N = 20`$ holds $`\binom{20}{10} = 184{,}756`$ amplitudes instead of $`2^{20} = 1{,}048{,}576`$. The dots are a
second-order Trotter circuit (`Circuit.pauli_evolution`) run on a full
statevector; they agree with the exact result to $`10^{-3}`$ at $`\Delta t = 0.05`$.

**Right: the interacting vacuum.** The ground state's particle density as the
lattice grows, by DMRG (`MPS.dmrg`, $`\chi = 32`$) up to 100 sites. The all-to-all
electric terms would naively need an $`O(N^2)`$ MPO bond; ketrs's MPO shares
states between terms with the same remaining operators, which needs only
$`O(N)`$. Crosses are exact diagonalisation within the neutral sector
(`lowest_eigenstates(..., sector=...)`); DMRG energies match them to about
$`10^{-11}`$.

## Running it

```sh
python examples/schwinger/schwinger.py
```

Takes about 40 seconds on a Ryzen 5 5600X (most of it the exact $`N = 20`$
quench); DMRG at $`N = 100`$ takes about 6 seconds. It prints the validation
numbers above and writes `schwinger.png`.
