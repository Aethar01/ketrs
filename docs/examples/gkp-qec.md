# Example: GKP Bosonic Error Correction

Gottesman–Kitaev–Preskill (GKP) codes encode a logical qubit in the quadrature grid of a single bosonic mode. Simulating finite-energy GKP error correction at realistic squeezing (10–15 dB) is notoriously expensive in cutoff Fock space because multi-mode entangling gates such as $\mathrm{CX}(1)$ spread photon support past $d > 100$ per mode.

[`examples/gkp_qec.py`](https://github.com/Aethar01/ketrs/blob/main/examples/gkp_qec.py) simulates Steane-style GKP error correction with `ketrs.GaussianMixture` (Bourassa et al. 2021) at $\varepsilon = 0.06$ ($12.2\text{ dB}$ squeezing), where Gaussian gates update a single shared covariance matrix in $O(m)$ and noise channels automatically damp and prune extinguished interference fringes.

## The Encoding ($\hbar = 2$)

For $\hbar = 2$, the square GKP grid spacing is $d = \sqrt{\pi \hbar} = \sqrt{2\pi}$:

- Computational basis states $|0_\varepsilon\rangle$ and $|1_\varepsilon\rangle$ have $\hat{x}$ peaks at even and odd integer multiples of $\sqrt{2\pi}$ respectively, with each peak squeezed to variance $\tanh\varepsilon$ and damped by the finite-energy envelope $\exp\bigl(-\frac{1}{2}\pi\tanh(\varepsilon) s^2\bigr)$.
- Logical Clifford gates are native Gaussian operations:

$$
X_L = D\!\left(\sqrt{\tfrac{\pi}{2}}, 0\right), \qquad
H_L = R\!\left(\tfrac{\pi}{2}\right), \qquad
\mathrm{CNOT}_L = \mathrm{CX}(1).
$$

## Steane $x$-Syndrome Extraction

1. Prepare a data mode in $|0_\varepsilon\rangle$ and an ancilla mode in $|+_\varepsilon\rangle$ (peaks at *all* integer multiples of $\sqrt{2\pi}$).
2. Apply Gaussian random-displacement noise (`CvChannel.additive_noise(sigma2)`) to the data mode.
3. Entangle data $\to$ ancilla with $\mathrm{CX}(1)$, which maps $\hat{x}_{\mathrm{anc}} \mapsto \hat{x}_{\mathrm{anc}} + \hat{x}_{\mathrm{data}}$.
4. Measure $\hat{x}_{\mathrm{anc}}$ with homodyne detection (`measure_homodyne(0.0, 1)`). Wrapping the outcome modulo $\sqrt{2\pi}$ into $(-\sqrt{2\pi}/2, \sqrt{2\pi}/2]$ estimates the continuous displacement error $e_x$.
5. Apply the feedback displacement $D(-\mathrm{wrap}(m_x, \sqrt{2\pi})/2, 0)$ to the data mode before the next noise round and logical $Z$ readout.

## Running It

```sh
python examples/gkp_qec.py
```

Measured output at $\varepsilon = 0.06$ ($12.2\text{ dB}$, 200 shots per noise level, 2 noise rounds):

| $\sigma^2$/round | Uncorrected error | Steane QEC error | Error reduction | Runtime (200 shots) |
|---:|---:|---:|---:|---:|
| 0.10 | 1.5% | 1.0% | 1.50× | 2.9 s |
| 0.16 | 4.5% | 0.5% | **9.00×** | 2.7 s |
| 0.22 | 9.0% | 3.0% | **3.00×** | 2.5 s |
| 0.30 | 16.5% | 8.0% | **2.06×** | 2.5 s |
