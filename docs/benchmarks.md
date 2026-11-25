# Benchmarks

Measured on a Ryzen 5 5600X and an AMD RX 9060 XT (through Vulkan), plus an NVIDIA RTX 4090 where noted, against Qiskit Aer 0.18 in its fastest configuration (passive OpenMP waiting, and fusion on or off, whichever is faster at each size) and SciPy. Reproduce with the scripts in [`bench/`](https://github.com/Aethar01/ketrs/tree/main/bench).

## Summary

| Benchmark | Hardware | Baseline | ketrs speedup |
|---|---|---|---:|
| **CPU statevector** (18–26q ansatz & QFT) | Ryzen 5 5600X | Qiskit Aer 0.18 CPU | **2.2–9.1×** |
| **GPU statevector, f32** (22–28q ansatz & QFT) | NVIDIA RTX 4090 | Aer 0.18 CUDA | **2.3–4.2×** |
| **GPU statevector, f64** (22–28q ansatz & QFT) | NVIDIA RTX 4090 | Best of Aer CUDA / cuStateVec | **1.7–2.3×** |
| **Mid-circuit measurement** (12–20q, 1000 shots) | Ryzen 5 5600X | Best of Aer / Aer shot-branching | **18–460×** |
| **MPS TEBD** (50–100q, χ = 32–128) | Ryzen 5 5600X | Aer MPS | **6.3–32.5×** |
| **MPS GPU TDVP & DMRG** (64–192q, χ = 64) | RX 9060 XT (Vulkan FP64) | ketrs CPU (3 threads) | **2.7–3.4×** |
| **Symmetric MPS TDVP** (194q gauge theory, χ = 64–256) | Ryzen 5 5600X (6 threads) | ketrs dense GPU (RX 9060 XT) | **28–137×** |
| **Exact `exp(-iHt)` & Lanczos** (14–20q Heisenberg) | Ryzen 5 5600X | SciPy `expm_multiply` / `eigsh` | **2.8–17.0×** |

---

## Statevector Circuits (CPU & AMD Vulkan GPU)

End to end from `|0…0⟩` to the final amplitudes. The GPU column is compared with `ketrs` on the CPU (Aer's GPU mode needs CUDA, which this AMD card cannot run; see below for RTX 4090). GPU times are in double precision and include downloading the state. In the bar charts each block is 1× unless noted.

### Layered Ansatz (10 layers of `Ry` + CNOT chain)

| Qubits | Aer | ketrs | vs Aer | ketrs GPU | GPU vs CPU |
|---:|---:|---:|:---|---:|:---|
| 18 | 37.1 ms | 7.1 ms | `█████▎` 5.2× | 5.5 ms | `█▎` 1.3× |
| 20 | 115.7 ms | 52.3 ms | `██▎` 2.2× | 20.0 ms | `██▋` 2.6× |
| 22 | 736.8 ms | 125.3 ms | `█████▉` 5.9× | 78.3 ms | `█▋` 1.6× |
| 24 | 3.12 s | 555.4 ms | `█████▋` 5.6× | 238.2 ms | `██▍` 2.3× |
| 26 | 13.82 s | 2.42 s | `█████▊` 5.7× | 935.2 ms | `██▋` 2.6× |

### Quantum Fourier Transform

| Qubits | Aer | ketrs | vs Aer | ketrs GPU | GPU vs CPU |
|---:|---:|---:|:---|---:|:---|
| 18 | 18.8 ms | 3.6 ms | `█████▎` 5.2× | 2.6 ms | `█▍` 1.4× |
| 20 | 71.9 ms | 16.1 ms | `████▌` 4.5× | 11.8 ms | `█▍` 1.4× |
| 22 | 236.0 ms | 70.7 ms | `███▍` 3.3× | 40.3 ms | `█▊` 1.8× |
| 24 | 1.33 s | 169.9 ms | `███████▉` 7.8× | 171.6 ms | `█` 1.0× |
| 26 | 6.21 s | 678.7 ms | `█████████▏` 9.1× | 613.0 ms | `█▏` 1.1× |

Against Qiskit's NumPy `quantum_info.Statevector`, `ketrs` is 29–87× faster at 18–20 qubits. Below 18 qubits the comparison with Aer is dominated by Aer's ~2 ms per-job overhead.

---

## NVIDIA GPU (RTX 4090 vs Aer CUDA / cuStateVec)

Measured on a rented RTX 4090 against Aer 0.18's CUDA backend at matching precision, and in FP64 taking the faster of Aer CUDA and cuStateVec. Times measure GPU work only, reading back one `⟨Z⟩`.

| Circuit | Qubits | Aer f32 | ketrs f32 | vs Aer | best Aer f64 | ketrs f64 | vs Aer |
|---|---:|---:|---:|:---|---:|---:|:---|
| Layered | 22 | 37.4 ms | 14.1 ms | `██▋` 2.7× | 32.5 ms | 18.1 ms | `█▊` 1.8× |
| Layered | 24 | 158.5 ms | 45.5 ms | `███▌` 3.5× | 129.5 ms | 56.4 ms | `██▎` 2.3× |
| Layered | 26 | 671.7 ms | 176.7 ms | `███▊` 3.8× | 498.5 ms | 247.3 ms | `██` 2.0× |
| Layered | 28 | 2.86 s | 687.5 ms | `████▏` 4.2× | 2.09 s | 1.02 s | `██` 2.0× |
| QFT | 22 | 25.0 ms | 9.9 ms | `██▌` 2.5× | 21.9 ms | 13.0 ms | `█▋` 1.7× |
| QFT | 24 | 68.0 ms | 27.5 ms | `██▌` 2.5× | 73.2 ms | 43.7 ms | `█▋` 1.7× |
| QFT | 26 | 255.5 ms | 110.0 ms | `██▍` 2.3× | 304.5 ms | 182.3 ms | `█▋` 1.7× |
| QFT | 28 | 1.08 s | 380.7 ms | `██▉` 2.8× | 1.33 s | 720.2 ms | `█▉` 1.8× |

---

## Mid-Circuit Measurement (1000 shots)

*Feed-forward*: 4 ansatz layers, two measurements with conditioned corrections, 4 more layers, terminal measurements on all qubits. *Repeat until success*: re-prepares and measures an entangled qubit until it reads 1, then runs 4 more layers. One block is 20×.

| Circuit | Qubits | Aer | Aer, shot branching | ketrs | vs best Aer |
|---|---:|---:|---:|---:|:---|
| Feed-forward | 12 | 300 ms | 64.2 ms | 896 µs | `███▋` 72× |
| Feed-forward | 16 | 7.63 s | 1.36 s | 7.2 ms | `█████████▌` 189× |
| Feed-forward | 20 | 160.0 s | 31.7 s | 69.0 ms | `███████████████████████` 460× |
| Repeat until success | 12 | 347 ms | 127 ms | 7.0 ms | `▉` 18× |
| Repeat until success | 16 | 7.26 s | 2.24 s | 49.4 ms | `██▎` 45× |
| Repeat until success | 20 | 143.1 s | 45.2 s | 428 ms | `█████▎` 106× |

---

## MPS TEBD

Transverse-field Ising chain (second-order Trotter, `Δt = 0.1`) against Aer's MPS method at identical bond-dimension cap `χ`. One block is 2×.

| Qubits | Steps | χ cap | ketrs | Aer MPS | Speedup |
|---:|---:|---:|---:|---:|:---|
| 50 | 10 | 32 | 0.01 s | 0.04 s | `███▏` 6.3× |
| 100 | 10 | 32 | 0.01 s | 0.07 s | `█████▊` 11.5× |
| 50 | 20 | 64 | 0.02 s | 0.18 s | `████▌` 9.0× |
| 100 | 20 | 64 | 0.03 s | 0.38 s | `███████▎` 14.6× |
| 100 | 40 | 128 | 2.33 s | 75.73 s | `████████████████▎` 32.5× |

---

## MPS on the GPU (RX 9060 XT, FP64)

At bond dimension cap `χ = 64`: one two-site TDVP step (`Δt = 0.6`) of a 192-qubit Schwinger model with link registers (1537 Pauli terms, MPO bond ~25), and six DMRG sweeps from a Néel state for a 64-qubit `J₁–J₂` Heisenberg chain (`J₂ = 0.2`). CPU had 3 threads. One block is 1×.

| Workload | Mean bond | GPU | CPU, 3 threads | Speedup |
|---|---:|---:|---:|:---|
| TDVP step | 56 | 38.3 s | 130.5 s | `███▍` 3.4× |
| DMRG, 6 sweeps | 50 | 12.2 s | 33.6 s | `██▊` 2.7× |

---

## Symmetric MPS (`with_charges`)

Preparing a wave packet in a 194-qubit Schwinger model with link registers (48 sites, 3 qubits per link, 2 packet ancillas): two TDVP steps of a 4032-term generator on the DMRG vacuum, conserving total charge and all 48 Gauss laws, compared against dense TDVP on the RX 9060 XT. Block-sparse environments use 28–33× less memory (0.14 GB vs 4.1 GB at `χ = 256`). One block is 8×.

| χ cap | Dense, GPU | Symmetric, 6 threads | Symmetric, GPU | Speedup |
|---:|---:|---:|---:|:---|
| 64 | 33.7 s | 1.2 s | 2.7 s | `███▌` 28× |
| 128 | 199.5 s | 2.6 s | 4.5 s | `█████████▋` 77× |
| 256 | 811 s | 5.9 s | 6.5 s | `█████████████████▏` 137× |

At `χ = 512`, symmetric GPU runs 1.6× faster than 6 CPU threads (10.2 s vs 16.4 s). With 402 qubits (100 sites, 101 charges) at `χ = 256`, an evolution step took 19.2 s on six CPU threads and 14.5 s on the GPU.

---

## Exact Evolution & Ground States vs SciPy

Heisenberg chain against SciPy's `expm_multiply` and `eigsh` on a pre-built sparse matrix (construction not timed). Results agree to ~`10⁻¹⁴`.

| Workload | Qubits | ketrs | SciPy | Speedup |
|---|---:|---:|---:|:---|
| `exp(-iHt)∣ψ⟩` | 14 | 10.4 ms | 28.7 ms | `██▊` 2.8× |
| `exp(-iHt)∣ψ⟩` | 18 | 196.5 ms | 728.7 ms | `███▊` 3.7× |
| `exp(-iHt)∣ψ⟩` | 20 | 732.7 ms | 3.80 s | `█████▏` 5.2× |
| Ground state | 14 | 16.4 ms | 67.8 ms | `████▏` 4.1× |
| Ground state | 18 | 165.7 ms | 2.81 s | `█████████████████` 17.0× |
| Ground state | 20 | 2.74 s | 11.06 s | `████` 4.0× |

At 20 qubits the Pauli-sum table exceeds its 64 MiB budget and `ketrs` falls back to a slower path.
