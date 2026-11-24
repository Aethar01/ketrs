//! Periodic-trace matrix product states (`PeriodicMps`).
//!
//! A periodic-trace MPS represents a pure state of `n` qubits on a closed ring as
//!
//!   `ψ(b₀ … bₙ₋₁) = Tr(A[0][b₀] A[1][b₁] ⋯ A[n−1][bₙ₋₁])`,
//!
//! where `A[q]` has shape `(χ_q, 2, χ_{(q+1) % n})` and `χ_0` is the periodic
//! seam bond connecting qubit `n - 1` back to qubit `0`.
//!
//! Unlike folding a periodic ring into an open-boundary MPS (which severs two
//! lattice bonds per cut and squares the required bond dimension), a periodic
//! MPS keeps cutwidth 1 and translation symmetry across the `(n - 1, 0)` seam.
//!
//! - Exact periodic trace norm (`norm_sqr`), inner product (`inner`),
//!   amplitudes (`amplitude`, `to_amplitudes`), and local/span expectation
//!   values (`expectation`, `expectations`, using antipodal cut caching so
//!   seam-crossing terms cost only their cyclic span).
//! - Fast `O(p D χ³)` periodic two-site DMRG (`dmrg`) and periodic two-site
//!   TDVP (`tdvp`) using Pippan–White–Evertz (2010) low-rank transfer-matrix
//!   compression, with automatic exact `O(χ⁵)` SVD for small `χ` (or when
//!   `transfer_rank` is set to `Some(0)`).
//! - Periodic circuits and TEBD (`run`, `apply`, `tebd`) with direct
//!   `(n - 1, 0)` seam gates and shortest-arc swap routing.
//! - Conserved Abelian `U(1)` and `Z_n` charges (`with_charges`), including
//!   multi-sector seam bonds for periodic lattice gauge theories with local
//!   Gauss's laws.

mod dmrg;
#[cfg(feature = "gpu")]
mod gpu;
mod solver;
mod symmetric;
mod tdvp;
mod tebd;
mod transfer;
mod window;

use crate::circuit::Circuit;
use crate::error::Error;
use crate::mps::linalg::matmul;
use crate::mps::{MIN_PROJECTION, Mps, MpsOptions, ONE, Tensor, ZERO};
use crate::observable::PauliSum;
use num_complex::Complex64 as C64;
use rand::Rng;
use std::collections::BTreeMap;
use symmetric::PeriodicSymmetry;
use transfer::{TransferCompression, exact_expectations, exact_ring_inner, exact_ring_norm_symmetric};

/// Default singular value cutoff for Pippan–White–Evertz transfer-matrix
/// compression.
pub const DEFAULT_TRANSFER_CUTOFF: f64 = 1e-12;

/// A pure state of `n` qubits on a periodic ring as a periodic-trace matrix
/// product state.
#[derive(Clone, Debug)]
pub struct PeriodicMps {
    tensors: Vec<Tensor>,
    options: MpsOptions,
    truncation_error: f64,
    /// Optional rank cap for PWE opposite-half transfer-matrix compression:
    /// - `None`: adaptive (exact SVD for small `χ`, randomized SVD capped at
    ///   default rank for larger `χ`),
    /// - `Some(0)`: exact full-rank SVD (no low-rank approximation),
    /// - `Some(r)`: cap at rank `r`.
    transfer_rank: Option<usize>,
    /// Relative singular value cutoff for PWE transfer-matrix compression.
    transfer_cutoff: f64,
    symmetry: Option<PeriodicSymmetry>,
    #[cfg(feature = "gpu")]
    gpu: Option<crate::mps::GpuHandle>,
}

impl PeriodicMps {
    /// The all-zero state `|0…0⟩` on a ring of `n_qubits` qubits.
    pub fn zero(n_qubits: usize, options: MpsOptions) -> Self {
        Self::basis(n_qubits, &vec![false; n_qubits], options)
    }

    /// The product state with qubit `q` in `|bits[q]⟩` on a periodic ring.
    pub fn basis(n_qubits: usize, bits: &[bool], options: MpsOptions) -> Self {
        assert_eq!(bits.len(), n_qubits, "one bit per qubit");
        assert!(n_qubits > 0, "a PeriodicMps needs at least one qubit");
        assert!(options.max_bond > 0, "max_bond must be at least 1");
        let tensors = bits
            .iter()
            .map(|&b| Tensor {
                left: 1,
                right: 1,
                data: if b { vec![ZERO, ONE] } else { vec![ONE, ZERO] },
            })
            .collect();
        PeriodicMps {
            tensors,
            options,
            truncation_error: 0.0,
            transfer_rank: None,
            transfer_cutoff: DEFAULT_TRANSFER_CUTOFF,
            symmetry: None,
            #[cfg(feature = "gpu")]
            gpu: None,
        }
    }

    /// A random normalised periodic MPS with bond dimension `bond` on every
    /// ring bond (including the `(n - 1, 0)` seam bond). A generic starting
    /// point for [`dmrg`](Self::dmrg).
    pub fn random(n_qubits: usize, bond: usize, seed: u64, options: MpsOptions) -> Self {
        assert!(n_qubits > 0 && bond > 0, "need at least one qubit and bond");
        assert!(options.max_bond > 0, "max_bond must be at least 1");
        let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
        let mut uniform = move || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 52) as f64 - 1.0
        };
        let b = if n_qubits == 1 { 1 } else { bond };
        let tensors = (0..n_qubits)
            .map(|_| {
                let data = (0..b * 2 * b)
                    .map(|_| C64::new(uniform(), uniform()))
                    .collect();
                Tensor {
                    left: b,
                    right: b,
                    data,
                }
            })
            .collect();
        let mut pmps = PeriodicMps {
            tensors,
            options,
            truncation_error: 0.0,
            transfer_rank: None,
            transfer_cutoff: DEFAULT_TRANSFER_CUTOFF,
            symmetry: None,
            #[cfg(feature = "gpu")]
            gpu: None,
        };
        // Pre-scale tensors so exact norm doesn't overflow/underflow for large n.
        for t in &mut pmps.tensors {
            let nrm: f64 = t.data.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
            if nrm > 0.0 {
                let target = (b as f64).sqrt();
                t.data.iter_mut().for_each(|a| *a *= target / nrm);
            }
        }
        pmps.normalise();
        pmps
    }

    /// Wrap an open-boundary [`Mps`] as a [`PeriodicMps`] with seam bond
    /// dimension 1. Preserves conserved charges (and GPU handle) if present.
    pub fn from_mps(mps: &Mps) -> Self {
        let mut pmps = PeriodicMps {
            tensors: mps.raw_tensors().to_vec(),
            options: mps.options(),
            truncation_error: mps.truncation_error(),
            transfer_rank: None,
            transfer_cutoff: DEFAULT_TRANSFER_CUTOFF,
            symmetry: None,
            #[cfg(feature = "gpu")]
            gpu: mps.gpu().cloned().map(crate::mps::GpuHandle),
        };
        if let Some(charges) = mps.charges() {
            pmps = pmps
                .with_charges(charges)
                .expect("symmetric open MPS converts cleanly to symmetric PeriodicMps");
        }
        pmps
    }

    /// Run `circuit` on `|0…0⟩` on the periodic ring.
    pub fn from_circuit(circuit: &Circuit, options: MpsOptions) -> Result<Self, Error> {
        let mut pmps = Self::zero(circuit.n_qubits(), options);
        pmps.run(circuit)?;
        Ok(pmps)
    }

    /// Number of qubits on the ring.
    pub fn n_qubits(&self) -> usize {
        self.tensors.len()
    }

    /// Truncation settings (`max_bond` and `cutoff`).
    pub fn options(&self) -> MpsOptions {
        self.options
    }

    /// Change truncation settings for later operations.
    pub fn set_options(&mut self, options: MpsOptions) {
        assert!(options.max_bond > 0, "max_bond must be at least 1");
        self.options = options;
    }

    /// Run [`dmrg`](Self::dmrg) and [`tdvp`](Self::tdvp) local effective
    /// operators on `gpu`, or on the CPU with `None` (the default).
    ///
    /// Executes the batched Pippan–White–Evertz MPO, scalar, crossing-string,
    /// and norm-metric contractions on Vulkan FP64 compute shaders
    /// (`mps_gemm.comp`, `mps_mix.comp`, `mps_sum.comp`), keeping SVDs and
    /// tensor gauge splits on the CPU so results agree with the CPU path up to
    /// rounding. Fails if the GPU lacks 64-bit floats or buffer device
    /// addresses ([`GpuContext::supports_mps`](crate::gpu::GpuContext::supports_mps)).
    /// Clones share the GPU.
    #[cfg(feature = "gpu")]
    pub fn set_gpu(
        &mut self,
        gpu: Option<std::sync::Arc<crate::gpu::GpuContext>>,
    ) -> Result<(), Error> {
        if let Some(ctx) = &gpu
            && !ctx.supports_mps()
        {
            return Err(Error::Gpu(format!(
                "{} can't run MPS sweeps: they need 64-bit floats and buffer device addresses",
                ctx.device_name()
            )));
        }
        self.gpu = gpu.map(crate::mps::GpuHandle);
        Ok(())
    }

    /// The GPU set by [`set_gpu`](Self::set_gpu), if any.
    #[cfg(feature = "gpu")]
    pub fn gpu(&self) -> Option<&std::sync::Arc<crate::gpu::GpuContext>> {
        self.gpu.as_ref().map(|g| &g.0)
    }

    /// Pippan–White–Evertz transfer-matrix rank cap:
    /// - `None`: adaptive (exact SVD for small `χ`, randomized SVD for larger `χ`),
    /// - `Some(0)`: force exact full-rank SVD,
    /// - `Some(r)`: cap PWE transfer rank at `r`.
    pub fn transfer_rank(&self) -> Option<usize> {
        self.transfer_rank
    }

    /// Set the Pippan–White–Evertz transfer-matrix rank cap.
    pub fn set_transfer_rank(&mut self, rank: Option<usize>) {
        self.transfer_rank = rank;
    }

    /// Relative singular-value cutoff for PWE transfer-matrix compression.
    pub fn transfer_cutoff(&self) -> f64 {
        self.transfer_cutoff
    }

    /// Set the relative singular-value cutoff for PWE transfer-matrix compression.
    pub fn set_transfer_cutoff(&mut self, cutoff: f64) {
        assert!(
            cutoff >= 0.0 && cutoff.is_finite(),
            "transfer_cutoff must be finite and non-negative"
        );
        self.transfer_cutoff = cutoff;
    }

    fn transfer_compression(&self) -> TransferCompression {
        TransferCompression {
            max_rank: self.transfer_rank,
            cutoff: self.transfer_cutoff,
        }
    }

    /// Bond dimension of every ring bond: index `i` is the bond between
    /// qubits `i` and `(i + 1) % n` (`bond_dimensions()[n - 1]` is the seam
    /// bond between `n - 1` and `0`).
    pub fn bond_dimensions(&self) -> Vec<usize> {
        self.tensors.iter().map(|t| t.right).collect()
    }

    /// Total discarded weight from truncations so far.
    pub fn truncation_error(&self) -> f64 {
        self.truncation_error
    }

    /// Exact squared norm `⟨ψ|ψ⟩ = Tr(T[0] T[1] ⋯ T[n−1])`.
    pub fn norm_sqr(&self) -> f64 {
        if let Some(sym) = &self.symmetry {
            return exact_ring_norm_symmetric(&self.tensors, sym).max(0.0);
        }
        exact_ring_inner(&self.tensors, &self.tensors).re.max(0.0)
    }

    /// Exact periodic trace inner product `⟨self|other⟩`.
    pub fn inner(&self, other: &PeriodicMps) -> Result<C64, Error> {
        if self.n_qubits() != other.n_qubits() {
            return Err(Error::QubitCountMismatch {
                expected: self.n_qubits(),
                found: other.n_qubits(),
            });
        }
        Ok(exact_ring_inner(&self.tensors, &other.tensors))
    }

    /// Rescale site tensors evenly so `norm_sqr() == 1` without disturbing
    /// relative bond scales.
    pub fn normalise(&mut self) {
        let n = self.n_qubits();
        // First balance individual tensor Frobenius norms so the product doesn't
        // overflow/underflow before computing the exact trace norm.
        for t in &mut self.tensors {
            let nrm: f64 = t.data.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
            if nrm > 0.0 && nrm.is_finite() {
                let target = (t.left.max(t.right) as f64).sqrt();
                let s = target / nrm;
                t.data.iter_mut().for_each(|a| *a *= s);
            }
        }
        let ns = self.norm_sqr();
        if ns > 0.0 && ns.is_finite() {
            let factor = ns.powf(-0.5 / n as f64);
            for t in &mut self.tensors {
                t.data.iter_mut().for_each(|a| *a *= factor);
            }
        }
    }

    /// Exact amplitude `Tr(A[0][b₀] ⋯ A[n−1][bₙ₋₁])` of basis state `bits`.
    pub fn amplitude(&self, bits: &[bool]) -> C64 {
        assert_eq!(bits.len(), self.n_qubits(), "one bit per qubit");
        let n = self.n_qubits();
        let slice_of = |q: usize, b: bool| -> faer::Mat<C64> {
            let t = &self.tensors[q];
            let p = usize::from(b);
            faer::Mat::from_fn(t.left, t.right, |l, r| t.data[(2 * l + p) * t.right + r])
        };
        let mut prod = slice_of(0, bits[0]);
        for (q, &bit) in bits.iter().enumerate().take(n).skip(1) {
            let next = slice_of(q, bit);
            prod = matmul(prod.as_ref(), next.as_ref());
        }
        debug_assert_eq!(prod.nrows(), prod.ncols());
        (0..prod.nrows()).map(|i| prod[(i, i)]).sum()
    }

    /// Expand the full `2ⁿ` statevector (qubit 0 as LSB). Only for small `n`.
    pub fn to_amplitudes(&self) -> Vec<C64> {
        let n = self.n_qubits();
        assert!(n <= 24, "too many qubits to expand");
        let chi0 = self.tensors[0].left;
        // Start with identity matrix `chi0 × chi0` for index 0.
        let mut mats: Vec<faer::Mat<C64>> = vec![faer::Mat::identity(chi0, chi0)];
        for (q, t) in self.tensors.iter().enumerate() {
            let a0 = faer::Mat::from_fn(t.left, t.right, |l, r| t.data[(2 * l) * t.right + r]);
            let a1 = faer::Mat::from_fn(t.left, t.right, |l, r| t.data[(2 * l + 1) * t.right + r]);
            if q + 1 == n {
                // Final step: only need diagonal trace Tr(M * A_p).
                let mut amps = vec![ZERO; 1usize << n];
                for (idx, m) in mats.iter().enumerate() {
                    let mut tr0 = ZERO;
                    let mut tr1 = ZERO;
                    for a in 0..chi0 {
                        for b in 0..t.left {
                            let mab = m[(a, b)];
                            tr0 += mab * a0[(b, a)];
                            tr1 += mab * a1[(b, a)];
                        }
                    }
                    amps[idx] = tr0;
                    amps[idx | (1 << q)] = tr1;
                }
                return amps;
            }
            let mut next = Vec::with_capacity(mats.len() * 2);
            next.resize_with(mats.len() * 2, || faer::Mat::zeros(0, 0));
            for (idx, m) in mats.into_iter().enumerate() {
                next[idx] = matmul(m.as_ref(), a0.as_ref());
                next[idx | (1 << q)] = matmul(m.as_ref(), a1.as_ref());
            }
            mats = next;
        }
        unreachable!()
    }

    /// Exact expectation value `⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩`, supporting terms spanning
    /// across the periodic `(n - 1) → 0` seam.
    pub fn expectation(&self, observable: &PauliSum) -> Result<f64, Error> {
        let vals = self.expectations(std::slice::from_ref(observable))?;
        Ok(vals[0])
    }

    /// Exact batched expectation values `⟨ψ|O|ψ⟩ / ⟨ψ|ψ⟩` for every observable,
    /// sharing prefix/suffix ring transfer caches across all terms and using
    /// antipodal cuts so seam-crossing terms cost only their cyclic span.
    pub fn expectations(&self, observables: &[PauliSum]) -> Result<Vec<f64>, Error> {
        if observables.is_empty() {
            return Ok(Vec::new());
        }
        let norm = self.norm_sqr();
        if norm == 0.0 {
            return Err(Error::InvalidArgument(
                "expectation value of a zero state".into(),
            ));
        }
        exact_expectations(&self.tensors, norm, observables)
    }

    /// Probability that measuring `qubit` gives `1`.
    pub fn probability_one(&self, qubit: usize) -> Result<f64, Error> {
        if qubit >= self.n_qubits() {
            return Err(Error::QubitOutOfRange {
                qubit,
                n_qubits: self.n_qubits(),
            });
        }
        let z = PauliSum::from(
            crate::observable::PauliString::new([(qubit, crate::observable::Pauli::Z)]).unwrap(),
        );
        let ez = self.expectation(&z)?;
        Ok(((1.0 - ez) * 0.5).clamp(0.0, 1.0))
    }

    /// Project `qubit` onto `outcome` (`|b⟩⟨b|`) and renormalise, returning its
    /// probability. Errors, leaving the state unchanged, if the probability is
    /// below [`MIN_PROJECTION`].
    pub fn project(&mut self, qubit: usize, outcome: bool) -> Result<f64, Error> {
        let p1 = self.probability_one(qubit)?;
        let prob = if outcome { p1 } else { 1.0 - p1 };
        if prob.is_nan() || prob < MIN_PROJECTION {
            return Err(Error::InvalidArgument(format!(
                "qubit {qubit} is {} with probability {prob:.1e}, too small to project onto",
                u8::from(outcome)
            )));
        }
        let scale = prob.sqrt().recip();
        let t = &mut self.tensors[qubit];
        let r = t.right;
        for (i, a) in t.data.iter_mut().enumerate() {
            let bit = (i / r) % 2 == 1;
            *a = if bit == outcome { *a * scale } else { ZERO };
        }
        Ok(prob)
    }

    /// Project onto each `(qubit, outcome)` in turn, returning their joint
    /// probability.
    pub fn project_all(&mut self, outcomes: &[(usize, bool)]) -> Result<f64, Error> {
        let mut projected = self.clone();
        let mut prob = 1.0;
        for &(qubit, outcome) in outcomes {
            prob *= projected.project(qubit, outcome)?;
        }
        *self = projected;
        Ok(prob)
    }

    /// Measure `qubit`, collapsing the state and returning the outcome.
    pub fn measure(&mut self, qubit: usize, rng: &mut impl Rng) -> Result<bool, Error> {
        let p1 = self.probability_one(qubit)?;
        let outcome = rng.random::<f64>() < p1;
        let _ = self.project(qubit, outcome)?;
        Ok(outcome)
    }

    /// Draw `shots` samples of every qubit (one bit per qubit) by sequential
    /// exact conditional sampling.
    pub fn sample_bits(&self, shots: usize, rng: &mut impl Rng) -> Vec<Vec<bool>> {
        let n = self.n_qubits();
        (0..shots)
            .map(|_| {
                let mut state = self.clone();
                let mut bits = Vec::with_capacity(n);
                for q in 0..n {
                    let b = state.measure(q, rng).unwrap_or(false);
                    bits.push(b);
                }
                bits
            })
            .collect()
    }

    /// Draw `shots` basis-state indices (qubit 0 as LSB).
    pub fn sample(&self, shots: usize, rng: &mut impl Rng) -> Vec<usize> {
        assert!(
            self.n_qubits() <= usize::BITS as usize,
            "too many qubits to index"
        );
        self.sample_bits(shots, rng)
            .into_iter()
            .map(|bits| {
                bits.iter()
                    .enumerate()
                    .fold(0, |acc, (q, &b)| acc | (usize::from(b) << q))
            })
            .collect()
    }

    /// Tally `shots` samples by basis-state index.
    pub fn sample_counts(&self, shots: usize, rng: &mut impl Rng) -> BTreeMap<usize, usize> {
        let mut counts = BTreeMap::new();
        for j in self.sample(shots, rng) {
            *counts.entry(j).or_insert(0) += 1;
        }
        counts
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mps::periodic::dmrg::{ring_windows, two_site_dense};
    use crate::mps::periodic::window::{WindowEnvs, split_hamiltonian_on_window};
    use crate::observable::{Pauli, PauliString};
    use crate::statevector::krylov::{LinearOperator, dot};

    #[test]
    fn window_envs_match_exact_norm_and_energy_on_all_positions() {
        let n = 6;
        let mut h = PauliSum::new();
        for i in 0..n {
            let j = (i + 1) % n;
            h.add(1.0, PauliString::new([(i, Pauli::X), (j, Pauli::X)]).unwrap());
            h.add(1.0, PauliString::new([(i, Pauli::Y), (j, Pauli::Y)]).unwrap());
            h.add(0.7, PauliString::new([(i, Pauli::Z), (j, Pauli::Z)]).unwrap());
            h.add(0.3 * i as f64, PauliString::new([(i, Pauli::Z)]).unwrap());
        }
        let mut pmps = PeriodicMps::random(n, 3, 17, MpsOptions::default());
        pmps.set_transfer_rank(Some(0)); // exact
        let want_norm = pmps.norm_sqr();
        let want_e = pmps.expectation(&h).unwrap();

        for (win, opp) in ring_windows(n) {
            let mut st = pmps.clone();
            st.canonicalise_window_left(&win);
            let wh = split_hamiltonian_on_window(&h, win.clone(), opp.clone(), n, None).unwrap();
            let shift = wh.shift;
            let mut envs = WindowEnvs::new(&st.tensors, &wh, st.transfer_compression(), None);
            envs.build_all_right(&st.tensors);
            for pos in 0..win.len() - 1 {
                let q1 = win[pos];
                let q2 = win[pos + 1];
                let prob = envs.two_site_problem(pos, &st.tensors, None, 1e-12);
                let theta = two_site_dense(&st.tensors[q1], &st.tensors[q2]);
                let v = prob.to_solver_vec(&theta);
                let mut nv = vec![ZERO; v.len()];
                let mut hv = vec![ZERO; v.len()];
                prob.norm_linear_op().apply(&v, &mut nv);
                prob.ham_linear_op().apply(&v, &mut hv);
                let got_norm = dot(&v, &nv).re;
                let got_e = dot(&v, &hv).re / got_norm + shift;
                assert!(
                    (got_norm - want_norm).abs() < 1e-10,
                    "win {win:?} pos {pos}: norm {got_norm} vs {want_norm}"
                );
                assert!(
                    (got_e - want_e).abs() < 1e-10,
                    "win {win:?} pos {pos}: energy {got_e} vs {want_e}"
                );
                // Also test identity split preserves exact amplitudes!
                let before = st.to_amplitudes();
                let tilde = prob.unpack_gauge(&v);
                st.split_window_pair(
                    q1,
                    q2,
                    &prob,
                    faer::MatRef::from_row_major_slice(&tilde, prob.l_eff * 2, 2 * prob.r_eff),
                    crate::mps::Absorb::Right,
                );
                let after = st.to_amplitudes();
                let diff = before
                    .iter()
                    .zip(&after)
                    .map(|(a, b)| (a - b).norm())
                    .fold(0.0f64, f64::max);
                assert!(diff < 1e-10, "win {win:?} pos {pos} split diff {diff}");
                envs.update_left(pos, &st.tensors);
            }
        }
    }

    #[test]
    fn symmetric_periodic_charge_violation_stays_zero() {
        let n = 6;
        let mut h = PauliSum::new();
        for i in 0..n {
            let j = (i + 1) % n;
            h.add(1.0, PauliString::new([(i, Pauli::X), (j, Pauli::X)]).unwrap());
            h.add(1.0, PauliString::new([(i, Pauli::Y), (j, Pauli::Y)]).unwrap());
            h.add(1.0, PauliString::new([(i, Pauli::Z), (j, Pauli::Z)]).unwrap());
        }
        let neel: Vec<bool> = (0..n).map(|q| q % 2 == 1).collect();
        let mut sym = PeriodicMps::basis(n, &neel, MpsOptions { max_bond: 8, cutoff: 1e-13 })
            .with_charges(&[crate::mps::Charge::weights(vec![1; n])])
            .unwrap();
        sym.dmrg(
            &h,
            &crate::mps::DmrgOptions {
                max_bond: 8,
                max_sweeps: 4,
                noise: 1e-4,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(sym.charge_violation() < 1e-12);
        sym.tdvp(&h, 0.2, 2).unwrap();
        assert!(sym.charge_violation() < 1e-12);
    }
}
