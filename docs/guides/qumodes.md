# Qumodes & Hybrids (CV)

`ketrs` includes a full continuous-variable (CV) gate-model simulator and a mixed-radix qubit–qumode hybrid simulator, built around the same principles as the qubit side: ideal modes, exact gates, opt-in noise, and specialised engines for every circuit class.

## Conventions

All CV engines follow Strawberry Fields conventions with $\hbar = 2$:

$$
\hat{x} = a + a^\dagger, \qquad \hat{p} = -i(a - a^\dagger), \qquad [\hat{x}, \hat{p}] = 2i.
$$

- The vacuum $|0\rangle$ has quadrature variance $\langle \hat{x}^2 \rangle = \langle \hat{p}^2 \rangle = 1$ (covariance matrix $V = I_{2m}$ in `xxpp` ordering).
- Mode `0` is the least-significant digit in Fock-basis indices, matching qubit `0` on the discrete-variable side.
- Mid-circuit measurements are destructive and reset the measured mode to vacuum $|0\rangle$.
- Noise is opt-in: attach `CvChannel` operations directly to a `ModeCircuit` via `apply_channel(channel, mode)` (supported by `GaussianState`, `GaussianMixture`, `FockDensityMatrix`, `FockState.sample`, and `WindowSampler` without widening the active window).

---

## Building Circuits (`ModeCircuit`)

```python
import math
import ketrs

# Hong–Ou–Mandel interference with a variational beam splitter
c = ketrs.ModeCircuit(2)
theta = c.parameter()
c.set_input(0, ketrs.ModeState.fock(1))
c.set_input(1, ketrs.ModeState.fock(1))
c.beam_splitter(theta, 0.0, 0, 1)

bound = c.bind([math.pi / 4.0])
state = ketrs.FixedNFockState.from_circuit(bound)
print(state.probability([1, 1]))  # 0.0 (destructive interference)
print(state.probability([2, 0]))  # 0.5
```

### Feed-Forward from Measurements

Measurement calls on `ModeCircuit` return `Measured` handles that support linear arithmetic (`scale * m + offset`) and can be passed directly as gate arguments:

```python
import math
import ketrs

# Continuous-variable quantum teleportation
c = ketrs.ModeCircuit(3)
c.set_input(0, ketrs.ModeState.coherent(1.2, 0.4))
c.two_mode_squeeze(1.5, 0.0, 1, 2)
c.beam_splitter(math.pi / 4.0, 0.0, 0, 1)

m_x = c.measure_x(0)
m_p = c.measure_p(1)
c.displace(m_x / math.sqrt(2.0), 0.0, 2)
c.displace(m_p / math.sqrt(2.0), math.pi / 2.0, 2)

final_state, results = ketrs.GaussianState.run(c, seed=42)
```

---

## Automatic Backend Planner

Use `plan_circuit` to inspect the recommended backend for any `ModeCircuit` (always overridable with `prefer_backend`):

```python
plan = ketrs.plan_circuit(bound)
print(plan.backend, "-", plan.reason)
# fixed_n - photon-number-conserving Fock circuit (N=2, dim=3)

# Override explicitly when desired
custom = ketrs.plan_circuit(bound, prefer_backend="bosonic_mps", max_bond=32)
```

| Circuit Class | Backend | Approximation | Highlights |
|---|---|---|---|
| **Gaussian gates & states** | `GaussianState` | Exact | $O(m)$ 1-/2-mode covariance updates; exact PNR (`loop_hafnian`) & threshold (`torontonian`) probabilities |
| **Gaussian Boson Sampling** | `sample_gbs_pnr`, `sample_gbs_threshold` | Exact | Quesada–Arrazola chain-rule samplers with Williamson pure-state decomposition & incremental Cholesky |
| **Passive / number-conserving Fock** | `FixedNFockState`, `boson_sample` | Exact | $\binom{m+N-1}{N}$ SLOS interferometer pass; Clifford & Clifford boson sampling with Glynn permanents |
| **Streaming / time-bin loops** | `WindowSampler` | Cutoff $d$ | Static liveness analysis with $O(d^W)$ memory ($W = 2$ for single-loop delay lines) |
| **Gaussian circuits on cat / GKP / Fock** | `GaussianMixture` | Term pruning | Bourassa et al. (2021) overflow-free phase-space LCG with exact homodyne/heterodyne conditioning |
| **Universal non-Gaussian** | `FockState`, `FockDensityMatrix`, `GpuFockState` | Cutoff $d$ | Miatto–Quesada recurrences, block-sparse BS/TMS, CPTP noise channels, Vulkan FP64 GPU |
| **Large 1D / shallow non-Gaussian** | `BosonicMps` | Cutoff $d$, bond $\chi$ | Mixed-canonical MPS with block-sparse contractions, SWAP routing, and all 4 measurements |
| **Qubit–qumode hybrids** | `HybridCircuit`, `HybridState` | Cutoff $d$ | Mixed-radix $2^{n_q} d^{n_m}$ statevector with analytic JC, AJC, CD, CR, CS, SNAP, and CBS gates |

---

## Gaussian Circuits, GBS & Hafnians

`GaussianState` tracks the $2m$-element quadrature mean vector and $2m \times 2m$ covariance matrix, supporting exact Gaussian evolution, homodyne/heterodyne conditioning, noise channels (`CvChannel.loss`, `thermal_loss`, `additive_noise`), Wigner evaluation, and exact Fock/threshold probabilities:

```python
import ketrs

c = ketrs.ModeCircuit(4)
for m in range(4):
    c.squeeze(0.6, 0.0, m)
c.beam_splitter(0.7, 0.2, 0, 1).beam_splitter(0.5, -0.3, 2, 3)
c.beam_splitter(0.6, 0.1, 1, 2)

st = ketrs.GaussianState.from_circuit(c, seed=7)
print("P(1,1,0,0) =", st.fock_probability([1, 1, 0, 0]))
print("P(clicks)  =", st.threshold_probability([True, True, False, False]))

pnr_shots = ketrs.sample_gbs_pnr(st, shots=1000, seed=7)
thresh_shots = ketrs.sample_gbs_threshold(st, shots=1000, seed=7)
```

Direct combinatorial matrix functions are also exported: `hafnian`, `loop_hafnian`, `hafnian_repeated`, `torontonian`, `permanent`, and (with GPU support) `permanent_gpu`.

---

## Non-Gaussian Bosonic Qubits (`GaussianMixture`)

For Gaussian circuits acting on cat states (`ModeState.cat`), finite-energy square GKP states (`ModeState.gkp`), or Fock states (`ModeState.fock`), `GaussianMixture` implements the linear-combination-of-Gaussians representation (Bourassa et al. 2021) in overflow-free phase-space coordinates:

```python
import math
import ketrs

c = (
    ketrs.ModeCircuit(2)
    .set_input(0, ketrs.ModeState.gkp(epsilon=0.08, theta=0.0, phi=0.0))
    .set_input(1, ketrs.ModeState.gkp(epsilon=0.08, theta=math.pi / 2, phi=0.0))
    .cx(1.0, 0, 1)
)
mix = ketrs.GaussianMixture.from_circuit(c, seed=1)
mix.apply_channel(ketrs.CvChannel.additive_noise(0.15), 0)
syndrome = mix.measure_homodyne(0.0, 1)
```

See the full **[GKP Error Correction example](../examples/gkp-qec.md)** for a Steane-style syndrome-extraction workflow.

---

## Universal Fock, Bosonic MPS & Vulkan GPU

For circuits with non-Gaussian gates (`kerr`, `cross_kerr`, `cubic_phase`), choose between dense `FockState`, mixed-state `FockDensityMatrix`, `BosonicMps` for large 1D chains, or `GpuFockState` on Vulkan GPUs:

```python
import ketrs

c = ketrs.ModeCircuit(16)
for i in range(16):
    c.squeeze(0.2, 0.1 * i, i).kerr(0.3, i)
for i in range(15):
    c.beam_splitter(0.35, 0.2, i, i + 1)

# 16 modes at cutoff 8 (8^16 ≈ 2.8e14 dense amplitudes) runs in < 1 ms as an MPS
mps = ketrs.BosonicMps.from_circuit(c, cutoff=8, max_bond=24)
print("max bond:", mps.max_bond_dim, "truncation error:", mps.truncation_error())
```

When built with `--features gpu` on a Vulkan 1.2 FP64 device, `GpuFockState.from_circuit(circuit, cutoff)` batches the entire circuit into a single GPU submission, and `permanent_gpu(matrix)` accelerates exact double-precision permanents.

---

## Qubit–Qumode Hybrids (`HybridCircuit` & `HybridState`)

`HybridCircuit` and `HybridState` simulate $n_q$ qubits coupled to $n_m$ qumodes in a mixed-radix statevector of dimension $2^{n_q} d^{n_m}$, with exact closed-form kernels for circuit-QED gates:

- **Jaynes–Cummings**: $U_{\mathrm{JC}}(\theta, \phi) = \exp\bigl(-i\theta(e^{i\phi}\sigma_+ a + e^{-i\phi}\sigma_- a^\dagger)\bigr)$
- **Anti-Jaynes–Cummings**: $U_{\mathrm{AJC}}(\theta, \phi) = \exp\bigl(-i\theta(e^{i\phi}\sigma_+ a^\dagger + e^{-i\phi}\sigma_- a)\bigr)$
- **Conditional displacement**: $|0\rangle\langle 0|\otimes D(\alpha) + |1\rangle\langle 1|\otimes D(-\alpha)$
- **Conditional rotation / squeezing / beam splitter / SNAP** and **SNAP** $\sum_n e^{i\theta_n} |n\rangle\langle n|$

```python
import math
import ketrs

# Synthesise an even cat state via conditional displacement + qubit postselection
c = (
    ketrs.HybridCircuit(1, 1)
    .h(0)
    .conditional_displace(1.6, 0.0, qubit=0, mode=0)
    .h(0)
)
st = ketrs.HybridState.from_circuit(c, cutoff=32)
p_even = st.postselect_qubit(0, False)

# Verify even photon parity with a dispersive conditional rotation CR(π/2)
st.h(0).conditional_rotation(math.pi / 2, qubit=0, mode=0).h(0)
print("P(odd parity) =", st.qubit_prob_one(0))  # ~0.0
```
