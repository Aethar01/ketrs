//! Qubit–qumode hybrid (mixed-radix) statevector simulator.
//!
//! Combines `n_qubits` two-level qubits (radix 2, least-significant bits
//! `0..n_qubits`, matching Qiskit / `ketrs::Statevector`) with `n_modes`
//! bosonic qumodes at Fock cutoff `d` (digits `0..n_modes` above the qubit
//! register):
//!
//! `index = qubit_bits + (1 << n_qubits) * Σ_{m} n_m · d^m`.
//!
//! Because qubit indices occupy the contiguous low bits and qumode strides
//! sit above them, every pure-qubit gate runs across contiguous `2^{n_q}`
//! slices, every pure-qumode gate vectorises across `2^{n_q}`, and all
//! native cQED hybrid gates — Jaynes–Cummings, anti-Jaynes–Cummings,
//! conditional displacement, conditional rotation/parity, conditional
//! squeezing, SNAP, conditional SNAP, and conditional beam splitter — run
//! analytically without matrix exponentiation.

use super::fixed_n::{build_two_mode_blocks, givens_decompose};
use super::fock_ops::{
    beam_splitter_blocks, cached_quad_eigen, cubic_phase_matrix, displacement_matrix,
    hermite_fock_wavefunctions, quadratic_phase_matrix, squeezing_matrix,
    two_mode_squeezing_blocks,
};
use super::state::ModeState;
use crate::error::Error;
use num_complex::Complex64 as C64;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::f64::consts::FRAC_1_SQRT_2;

/// Minimum statevector length to parallelise gate loops across rayon threads.
const PAR_MIN_LEN: usize = 4096;

/// Minimum shot count to parallelise independent hybrid trajectories.
const SAMPLE_PAR_MIN_SHOTS: usize = 16;

/// One operation in a [`HybridCircuit`].
#[derive(Clone, Debug)]
pub enum HybridOp {
    /// Arbitrary single-qubit unitary `[[u00, u01], [u10, u11]]` on `qubit`.
    Qubit1 {
        /// Qubit index.
        qubit: usize,
        /// `2 × 2` row-major unitary matrix.
        u: [[C64; 2]; 2],
    },
    /// Qubit CNOT (`CX`) with `control` and `target`.
    QubitCx {
        /// Control qubit.
        control: usize,
        /// Target qubit.
        target: usize,
    },
    /// Qubit `CZ` on `(a, b)`.
    QubitCz {
        /// First qubit.
        a: usize,
        /// Second qubit.
        b: usize,
    },
    /// Qubit `SWAP` on `(a, b)`.
    QubitSwap {
        /// First qubit.
        a: usize,
        /// Second qubit.
        b: usize,
    },
    /// Qumode phase rotation `R(φ) = exp(i φ n̂)`.
    Rotation {
        /// Angle `φ`.
        phi: f64,
        /// Mode index.
        mode: usize,
    },
    /// Qumode displacement `D(r e^{iφ})`.
    Displacement {
        /// Magnitude `r`.
        r: f64,
        /// Phase `φ`.
        phi: f64,
        /// Mode index.
        mode: usize,
    },
    /// Qumode squeezing `S(r e^{iφ})`.
    Squeezing {
        /// Magnitude `r`.
        r: f64,
        /// Phase `φ`.
        phi: f64,
        /// Mode index.
        mode: usize,
    },
    /// Qumode quadratic phase `P(s)`.
    QuadraticPhase {
        /// Shear `s`.
        s: f64,
        /// Mode index.
        mode: usize,
    },
    /// Qumode cubic phase `V(γ)`.
    CubicPhase {
        /// Coupling `γ`.
        gamma: f64,
        /// Mode index.
        mode: usize,
    },
    /// Qumode Kerr `K(κ) = exp(i κ n̂²)`.
    Kerr {
        /// Kerr strength `κ`.
        kappa: f64,
        /// Mode index.
        mode: usize,
    },
    /// Selective Number-dependent Arbitrary Phase `Σ_n e^{i θ_n} |n⟩⟨n|`.
    Snap {
        /// Per-Fock-level phases `θ_n` (levels past `phases.len()` get phase `0`).
        phases: Vec<f64>,
        /// Mode index.
        mode: usize,
    },
    /// Two-mode beam splitter `BS(θ, φ)` on `(a, b)`.
    BeamSplitter {
        /// Angle `θ`.
        theta: f64,
        /// Phase `φ`.
        phi: f64,
        /// First mode.
        a: usize,
        /// Second mode.
        b: usize,
    },
    /// Two-mode squeezing `S₂(r e^{iφ})` on `(a, b)`.
    TwoModeSqueezing {
        /// Magnitude `r`.
        r: f64,
        /// Phase `φ`.
        phi: f64,
        /// First mode.
        a: usize,
        /// Second mode.
        b: usize,
    },
    /// Two-mode `CX(s)` on `(a, b)`.
    ModeCx {
        /// Gain `s`.
        s: f64,
        /// Control mode.
        a: usize,
        /// Target mode.
        b: usize,
    },
    /// Two-mode `CZ(s)` on `(a, b)`.
    ModeCz {
        /// Gain `s`.
        s: f64,
        /// First mode.
        a: usize,
        /// Second mode.
        b: usize,
    },
    /// Two-mode cross-Kerr `CK(κ) = exp(i κ n̂_a n̂_b)` on `(a, b)`.
    CrossKerr {
        /// Coupling `κ`.
        kappa: f64,
        /// First mode.
        a: usize,
        /// Second mode.
        b: usize,
    },
    /// Passive linear-optics unitary `U` on `modes`.
    Passive {
        /// Target modes.
        modes: Vec<usize>,
        /// `k × k` row-major unitary matrix.
        matrix: Vec<C64>,
    },
    /// Conditional displacement `|0⟩⟨0|_q ⊗ D(+α)_m + |1⟩⟨1|_q ⊗ D(−α)_m`,
    /// where `α = r e^{iφ}`.
    ConditionalDisplacement {
        /// Magnitude `r`.
        r: f64,
        /// Phase `φ`.
        phi: f64,
        /// Control qubit.
        qubit: usize,
        /// Target mode.
        mode: usize,
    },
    /// Conditional rotation (dispersive shift) `exp(i θ Z_q n̂_m) = |0⟩⟨0| ⊗ R(+θ) + |1⟩⟨1| ⊗ R(−θ)`.
    ConditionalRotation {
        /// Rotation angle `θ` (`π/2` gives conditional parity up to a local mode rotation `R(π/2)`).
        theta: f64,
        /// Control qubit.
        qubit: usize,
        /// Target mode.
        mode: usize,
    },
    /// Conditional squeezing `|0⟩⟨0|_q ⊗ S(+z)_m + |1⟩⟨1|_q ⊗ S(−z)_m`,
    /// where `z = r e^{iφ}`.
    ConditionalSqueezing {
        /// Magnitude `r`.
        r: f64,
        /// Phase `φ`.
        phi: f64,
        /// Control qubit.
        qubit: usize,
        /// Target mode.
        mode: usize,
    },
    /// Conditional SNAP `exp(i Z_q Σ_n θ_n |n⟩⟨n|_m)`.
    ConditionalSnap {
        /// Per-Fock-level phases `θ_n`.
        phases: Vec<f64>,
        /// Control qubit.
        qubit: usize,
        /// Target mode.
        mode: usize,
    },
    /// Conditional beam splitter `|0⟩⟨0|_q ⊗ BS(+θ, φ)_{a,b} + |1⟩⟨1|_q ⊗ BS(−θ, φ)_{a,b}`.
    ConditionalBeamSplitter {
        /// Angle `θ`.
        theta: f64,
        /// Phase `φ`.
        phi: f64,
        /// Control qubit.
        qubit: usize,
        /// First mode.
        a: usize,
        /// Second mode.
        b: usize,
    },
    /// Jaynes–Cummings interaction `exp(−i θ (e^{iφ} σ_+ a + e^{-iφ} σ_- a†))`,
    /// with `|0⟩ = |g⟩`, `|1⟩ = |e⟩`, `σ_+ = |1⟩⟨0|`.
    JaynesCummings {
        /// Coupling angle `θ = g t`.
        theta: f64,
        /// Coupling phase `φ`.
        phi: f64,
        /// Qubit index.
        qubit: usize,
        /// Mode index.
        mode: usize,
    },
    /// Anti-Jaynes–Cummings interaction `exp(−i θ (e^{iφ} σ_+ a† + e^{-iφ} σ_- a))`.
    AntiJaynesCummings {
        /// Coupling angle `θ = g t`.
        theta: f64,
        /// Coupling phase `φ`.
        phi: f64,
        /// Qubit index.
        qubit: usize,
        /// Mode index.
        mode: usize,
    },
    /// Projective computational-basis (Z) measurement on `qubit`.
    MeasureQubit {
        /// Qubit index.
        qubit: usize,
    },
    /// Destructive photon-number measurement on `mode` (resets mode to vacuum).
    MeasurePhotons {
        /// Mode index.
        mode: usize,
    },
    /// Destructive threshold measurement on `mode` (`0` or `1`, resets mode to vacuum).
    MeasureThreshold {
        /// Mode index.
        mode: usize,
    },
    /// Destructive homodyne measurement of `x̂ cos φ + p̂ sin φ` on `mode`.
    MeasureHomodyne {
        /// Quadrature angle `φ`.
        phi: f64,
        /// Mode index.
        mode: usize,
    },
}

/// A hybrid circuit acting on `n_qubits` qubits and `n_modes` qumodes.
#[derive(Clone, Debug)]
pub struct HybridCircuit {
    n_qubits: usize,
    n_modes: usize,
    mode_inputs: Vec<ModeState>,
    ops: Vec<HybridOp>,
    n_results: usize,
}

impl HybridCircuit {
    /// Create a hybrid circuit with `n_qubits` qubits starting in `|0…0⟩` and
    /// `n_modes` qumodes starting in vacuum `|0…0⟩`.
    pub fn new(n_qubits: usize, n_modes: usize) -> Self {
        HybridCircuit {
            n_qubits,
            n_modes,
            mode_inputs: vec![ModeState::Vacuum; n_modes],
            ops: Vec::new(),
            n_results: 0,
        }
    }

    /// Number of qubits.
    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    /// Number of qumodes.
    pub fn n_modes(&self) -> usize {
        self.n_modes
    }

    /// Number of measurement results produced per shot.
    pub fn n_results(&self) -> usize {
        self.n_results
    }

    /// Initial state of each qumode.
    pub fn mode_inputs(&self) -> &[ModeState] {
        &self.mode_inputs
    }

    /// Operations in program order.
    pub fn ops(&self) -> &[HybridOp] {
        &self.ops
    }

    /// Whether the circuit has no measurements.
    pub fn is_unitary(&self) -> bool {
        self.n_results == 0
    }

    /// Set the initial pure state of qumode `mode`.
    pub fn set_mode_input(&mut self, mode: usize, state: ModeState) -> Result<&mut Self, Error> {
        if mode >= self.n_modes {
            return Err(Error::InvalidArgument(format!(
                "mode {mode} out of range for {} modes",
                self.n_modes
            )));
        }
        state.validate()?;
        if !state.is_pure() {
            return Err(Error::InvalidArgument(
                "HybridCircuit supports pure qumode inputs".into(),
            ));
        }
        self.mode_inputs[mode] = state;
        Ok(self)
    }

    fn check_q(&self, q: usize) -> Result<(), Error> {
        if q < self.n_qubits {
            Ok(())
        } else {
            Err(Error::InvalidArgument(format!(
                "qubit {q} out of range for {} qubits",
                self.n_qubits
            )))
        }
    }

    fn check_q_pair(&self, a: usize, b: usize) -> Result<(), Error> {
        self.check_q(a)?;
        self.check_q(b)?;
        if a == b {
            return Err(Error::InvalidArgument(format!(
                "qubit {a} used more than once"
            )));
        }
        Ok(())
    }

    fn check_m(&self, m: usize) -> Result<(), Error> {
        if m < self.n_modes {
            Ok(())
        } else {
            Err(Error::InvalidArgument(format!(
                "mode {m} out of range for {} modes",
                self.n_modes
            )))
        }
    }

    fn check_m_pair(&self, a: usize, b: usize) -> Result<(), Error> {
        self.check_m(a)?;
        self.check_m(b)?;
        if a == b {
            return Err(Error::InvalidArgument(format!(
                "mode {a} used more than once"
            )));
        }
        Ok(())
    }

    // --- Qubit gates ---

    /// Hadamard `H` on `qubit`.
    pub fn h(&mut self, qubit: usize) -> Result<&mut Self, Error> {
        self.check_q(qubit)?;
        let s = C64::new(FRAC_1_SQRT_2, 0.0);
        self.ops.push(HybridOp::Qubit1 {
            qubit,
            u: [[s, s], [s, -s]],
        });
        Ok(self)
    }

    /// Pauli `X` on `qubit`.
    pub fn x(&mut self, qubit: usize) -> Result<&mut Self, Error> {
        self.check_q(qubit)?;
        let z = C64::new(0.0, 0.0);
        let o = C64::new(1.0, 0.0);
        self.ops.push(HybridOp::Qubit1 {
            qubit,
            u: [[z, o], [o, z]],
        });
        Ok(self)
    }

    /// Pauli `Y` on `qubit`.
    pub fn y(&mut self, qubit: usize) -> Result<&mut Self, Error> {
        self.check_q(qubit)?;
        let z = C64::new(0.0, 0.0);
        let i = C64::new(0.0, 1.0);
        self.ops.push(HybridOp::Qubit1 {
            qubit,
            u: [[z, -i], [i, z]],
        });
        Ok(self)
    }

    /// Pauli `Z` on `qubit`.
    pub fn z(&mut self, qubit: usize) -> Result<&mut Self, Error> {
        self.check_q(qubit)?;
        let z = C64::new(0.0, 0.0);
        let o = C64::new(1.0, 0.0);
        self.ops.push(HybridOp::Qubit1 {
            qubit,
            u: [[o, z], [z, -o]],
        });
        Ok(self)
    }

    /// Phase gate `S = diag(1, i)` on `qubit`.
    pub fn s(&mut self, qubit: usize) -> Result<&mut Self, Error> {
        self.p(std::f64::consts::FRAC_PI_2, qubit)
    }

    /// `S† = diag(1, −i)` on `qubit`.
    pub fn sdg(&mut self, qubit: usize) -> Result<&mut Self, Error> {
        self.p(-std::f64::consts::FRAC_PI_2, qubit)
    }

    /// `T = diag(1, e^{iπ/4})` on `qubit`.
    pub fn t(&mut self, qubit: usize) -> Result<&mut Self, Error> {
        self.p(std::f64::consts::FRAC_PI_4, qubit)
    }

    /// `T† = diag(1, e^{−iπ/4})` on `qubit`.
    pub fn tdg(&mut self, qubit: usize) -> Result<&mut Self, Error> {
        self.p(-std::f64::consts::FRAC_PI_4, qubit)
    }

    /// Phase gate `P(λ) = diag(1, e^{iλ})` on `qubit`.
    pub fn p(&mut self, lam: f64, qubit: usize) -> Result<&mut Self, Error> {
        self.check_q(qubit)?;
        let z = C64::new(0.0, 0.0);
        self.ops.push(HybridOp::Qubit1 {
            qubit,
            u: [[C64::new(1.0, 0.0), z], [z, C64::from_polar(1.0, lam)]],
        });
        Ok(self)
    }

    /// `Rx(θ) = exp(−i θ X / 2)` on `qubit`.
    pub fn rx(&mut self, theta: f64, qubit: usize) -> Result<&mut Self, Error> {
        self.check_q(qubit)?;
        let (s, c) = (0.5 * theta).sin_cos();
        let cc = C64::new(c, 0.0);
        let is = C64::new(0.0, -s);
        self.ops.push(HybridOp::Qubit1 {
            qubit,
            u: [[cc, is], [is, cc]],
        });
        Ok(self)
    }

    /// `Ry(θ) = exp(−i θ Y / 2)` on `qubit`.
    pub fn ry(&mut self, theta: f64, qubit: usize) -> Result<&mut Self, Error> {
        self.check_q(qubit)?;
        let (s, c) = (0.5 * theta).sin_cos();
        self.ops.push(HybridOp::Qubit1 {
            qubit,
            u: [
                [C64::new(c, 0.0), C64::new(-s, 0.0)],
                [C64::new(s, 0.0), C64::new(c, 0.0)],
            ],
        });
        Ok(self)
    }

    /// `Rz(θ) = exp(−i θ Z / 2)` on `qubit`.
    pub fn rz(&mut self, theta: f64, qubit: usize) -> Result<&mut Self, Error> {
        self.check_q(qubit)?;
        let z = C64::new(0.0, 0.0);
        self.ops.push(HybridOp::Qubit1 {
            qubit,
            u: [
                [C64::from_polar(1.0, -0.5 * theta), z],
                [z, C64::from_polar(1.0, 0.5 * theta)],
            ],
        });
        Ok(self)
    }

    /// Arbitrary `2 × 2` unitary on `qubit`.
    pub fn qubit_unitary(&mut self, u: [[C64; 2]; 2], qubit: usize) -> Result<&mut Self, Error> {
        self.check_q(qubit)?;
        self.ops.push(HybridOp::Qubit1 { qubit, u });
        Ok(self)
    }

    /// Qubit CNOT (`CX`) from `control` to `target`.
    pub fn cx_qubit(&mut self, control: usize, target: usize) -> Result<&mut Self, Error> {
        self.check_q_pair(control, target)?;
        self.ops.push(HybridOp::QubitCx { control, target });
        Ok(self)
    }

    /// Qubit `CZ` on `(a, b)`.
    pub fn cz_qubit(&mut self, a: usize, b: usize) -> Result<&mut Self, Error> {
        self.check_q_pair(a, b)?;
        self.ops.push(HybridOp::QubitCz { a, b });
        Ok(self)
    }

    /// Qubit `SWAP` on `(a, b)`.
    pub fn swap_qubit(&mut self, a: usize, b: usize) -> Result<&mut Self, Error> {
        self.check_q_pair(a, b)?;
        self.ops.push(HybridOp::QubitSwap { a, b });
        Ok(self)
    }

    // --- Qumode gates ---

    /// Qumode rotation `R(φ)` on `mode`.
    pub fn rotation(&mut self, phi: f64, mode: usize) -> Result<&mut Self, Error> {
        self.check_m(mode)?;
        self.ops.push(HybridOp::Rotation { phi, mode });
        Ok(self)
    }

    /// Qumode displacement `D(r e^{iφ})` on `mode`.
    pub fn displace(&mut self, r: f64, phi: f64, mode: usize) -> Result<&mut Self, Error> {
        self.check_m(mode)?;
        self.ops.push(HybridOp::Displacement { r, phi, mode });
        Ok(self)
    }

    /// Qumode squeezing `S(r e^{iφ})` on `mode`.
    pub fn squeeze(&mut self, r: f64, phi: f64, mode: usize) -> Result<&mut Self, Error> {
        self.check_m(mode)?;
        self.ops.push(HybridOp::Squeezing { r, phi, mode });
        Ok(self)
    }

    /// Qumode quadratic phase `P(s)` on `mode`.
    pub fn quadratic_phase(&mut self, s: f64, mode: usize) -> Result<&mut Self, Error> {
        self.check_m(mode)?;
        self.ops.push(HybridOp::QuadraticPhase { s, mode });
        Ok(self)
    }

    /// Qumode cubic phase `V(γ)` on `mode`.
    pub fn cubic_phase(&mut self, gamma: f64, mode: usize) -> Result<&mut Self, Error> {
        self.check_m(mode)?;
        self.ops.push(HybridOp::CubicPhase { gamma, mode });
        Ok(self)
    }

    /// Qumode Kerr `K(κ)` on `mode`.
    pub fn kerr(&mut self, kappa: f64, mode: usize) -> Result<&mut Self, Error> {
        self.check_m(mode)?;
        self.ops.push(HybridOp::Kerr { kappa, mode });
        Ok(self)
    }

    /// Selective Number-dependent Arbitrary Phase `SNAP(phases)` on `mode`.
    pub fn snap(&mut self, phases: Vec<f64>, mode: usize) -> Result<&mut Self, Error> {
        self.check_m(mode)?;
        self.ops.push(HybridOp::Snap { phases, mode });
        Ok(self)
    }

    /// Beam splitter `BS(θ, φ)` on modes `(a, b)`.
    pub fn beam_splitter(
        &mut self,
        theta: f64,
        phi: f64,
        a: usize,
        b: usize,
    ) -> Result<&mut Self, Error> {
        self.check_m_pair(a, b)?;
        self.ops.push(HybridOp::BeamSplitter { theta, phi, a, b });
        Ok(self)
    }

    /// Two-mode squeezing `S₂(r e^{iφ})` on modes `(a, b)`.
    pub fn two_mode_squeeze(
        &mut self,
        r: f64,
        phi: f64,
        a: usize,
        b: usize,
    ) -> Result<&mut Self, Error> {
        self.check_m_pair(a, b)?;
        self.ops.push(HybridOp::TwoModeSqueezing { r, phi, a, b });
        Ok(self)
    }

    /// Two-mode `CX(s)` on modes `(a, b)`.
    pub fn cx_mode(&mut self, s: f64, a: usize, b: usize) -> Result<&mut Self, Error> {
        self.check_m_pair(a, b)?;
        self.ops.push(HybridOp::ModeCx { s, a, b });
        Ok(self)
    }

    /// Two-mode `CZ(s)` on modes `(a, b)`.
    pub fn cz_mode(&mut self, s: f64, a: usize, b: usize) -> Result<&mut Self, Error> {
        self.check_m_pair(a, b)?;
        self.ops.push(HybridOp::ModeCz { s, a, b });
        Ok(self)
    }

    /// Cross-Kerr `CK(κ)` on modes `(a, b)`.
    pub fn cross_kerr(&mut self, kappa: f64, a: usize, b: usize) -> Result<&mut Self, Error> {
        self.check_m_pair(a, b)?;
        self.ops.push(HybridOp::CrossKerr { kappa, a, b });
        Ok(self)
    }

    /// Passive linear-optics unitary `U` on `modes`.
    pub fn passive(&mut self, modes: Vec<usize>, matrix: Vec<C64>) -> Result<&mut Self, Error> {
        let k = modes.len();
        if matrix.len() != k * k {
            return Err(Error::InvalidArgument(format!(
                "passive block on {k} modes needs {} entries, got {}",
                k * k,
                matrix.len()
            )));
        }
        for (i, &m) in modes.iter().enumerate() {
            self.check_m(m)?;
            if modes[..i].contains(&m) {
                return Err(Error::InvalidArgument(format!(
                    "mode {m} used more than once"
                )));
            }
        }
        self.ops.push(HybridOp::Passive { modes, matrix });
        Ok(self)
    }

    // --- Hybrid qubit–qumode gates ---

    /// Conditional displacement `|0⟩⟨0|_q ⊗ D(+r e^{iφ})_m + |1⟩⟨1|_q ⊗ D(−r e^{iφ})_m`.
    pub fn conditional_displace(
        &mut self,
        r: f64,
        phi: f64,
        qubit: usize,
        mode: usize,
    ) -> Result<&mut Self, Error> {
        self.check_q(qubit)?;
        self.check_m(mode)?;
        self.ops.push(HybridOp::ConditionalDisplacement {
            r,
            phi,
            qubit,
            mode,
        });
        Ok(self)
    }

    /// Conditional rotation `exp(i θ Z_q n̂_m)`.
    pub fn conditional_rotation(
        &mut self,
        theta: f64,
        qubit: usize,
        mode: usize,
    ) -> Result<&mut Self, Error> {
        self.check_q(qubit)?;
        self.check_m(mode)?;
        self.ops
            .push(HybridOp::ConditionalRotation { theta, qubit, mode });
        Ok(self)
    }

    /// Conditional squeezing `|0⟩⟨0|_q ⊗ S(+r e^{iφ})_m + |1⟩⟨1|_q ⊗ S(−r e^{iφ})_m`.
    pub fn conditional_squeeze(
        &mut self,
        r: f64,
        phi: f64,
        qubit: usize,
        mode: usize,
    ) -> Result<&mut Self, Error> {
        self.check_q(qubit)?;
        self.check_m(mode)?;
        self.ops.push(HybridOp::ConditionalSqueezing {
            r,
            phi,
            qubit,
            mode,
        });
        Ok(self)
    }

    /// Conditional SNAP `exp(i Z_q Σ_n θ_n |n⟩⟨n|_m)`.
    pub fn conditional_snap(
        &mut self,
        phases: Vec<f64>,
        qubit: usize,
        mode: usize,
    ) -> Result<&mut Self, Error> {
        self.check_q(qubit)?;
        self.check_m(mode)?;
        self.ops.push(HybridOp::ConditionalSnap {
            phases,
            qubit,
            mode,
        });
        Ok(self)
    }

    /// Conditional beam splitter `|0⟩⟨0|_q ⊗ BS(+θ, φ)_{a,b} + |1⟩⟨1|_q ⊗ BS(−θ, φ)_{a,b}`.
    pub fn conditional_beam_splitter(
        &mut self,
        theta: f64,
        phi: f64,
        qubit: usize,
        a: usize,
        b: usize,
    ) -> Result<&mut Self, Error> {
        self.check_q(qubit)?;
        self.check_m_pair(a, b)?;
        self.ops.push(HybridOp::ConditionalBeamSplitter {
            theta,
            phi,
            qubit,
            a,
            b,
        });
        Ok(self)
    }

    /// Jaynes–Cummings gate `exp(−i θ (e^{iφ} σ_+ a + e^{-iφ} σ_- a†))`.
    pub fn jaynes_cummings(
        &mut self,
        theta: f64,
        phi: f64,
        qubit: usize,
        mode: usize,
    ) -> Result<&mut Self, Error> {
        self.check_q(qubit)?;
        self.check_m(mode)?;
        self.ops.push(HybridOp::JaynesCummings {
            theta,
            phi,
            qubit,
            mode,
        });
        Ok(self)
    }

    /// Anti-Jaynes–Cummings gate `exp(−i θ (e^{iφ} σ_+ a† + e^{-iφ} σ_- a))`.
    pub fn anti_jaynes_cummings(
        &mut self,
        theta: f64,
        phi: f64,
        qubit: usize,
        mode: usize,
    ) -> Result<&mut Self, Error> {
        self.check_q(qubit)?;
        self.check_m(mode)?;
        self.ops.push(HybridOp::AntiJaynesCummings {
            theta,
            phi,
            qubit,
            mode,
        });
        Ok(self)
    }

    // --- Measurements ---

    /// Projective computational-basis measurement on `qubit`.
    pub fn measure_qubit(&mut self, qubit: usize) -> Result<&mut Self, Error> {
        self.check_q(qubit)?;
        self.ops.push(HybridOp::MeasureQubit { qubit });
        self.n_results += 1;
        Ok(self)
    }

    /// Destructive photon-number measurement on `mode`.
    pub fn measure_photons(&mut self, mode: usize) -> Result<&mut Self, Error> {
        self.check_m(mode)?;
        self.ops.push(HybridOp::MeasurePhotons { mode });
        self.n_results += 1;
        Ok(self)
    }

    /// Destructive threshold measurement on `mode`.
    pub fn measure_threshold(&mut self, mode: usize) -> Result<&mut Self, Error> {
        self.check_m(mode)?;
        self.ops.push(HybridOp::MeasureThreshold { mode });
        self.n_results += 1;
        Ok(self)
    }

    /// Destructive homodyne measurement of `x̂ cos φ + p̂ sin φ` on `mode`.
    pub fn measure_homodyne(&mut self, phi: f64, mode: usize) -> Result<&mut Self, Error> {
        self.check_m(mode)?;
        self.ops.push(HybridOp::MeasureHomodyne { phi, mode });
        self.n_results += 1;
        Ok(self)
    }
}

/// Mixed-radix statevector on `n_qubits` qubits and `n_modes` qumodes at Fock
/// cutoff `cutoff` (`2^{n_qubits} · cutoff^{n_modes}` amplitudes).
#[derive(Clone, Debug)]
pub struct HybridState {
    n_qubits: usize,
    n_modes: usize,
    cutoff: usize,
    q_dim: usize,
    amps: Vec<C64>,
}

impl HybridState {
    /// Vacuum / `|0…0⟩ ⊗ |0…0⟩` hybrid state.
    pub fn new(n_qubits: usize, n_modes: usize, cutoff: usize) -> Result<Self, Error> {
        let (q_dim, total) = checked_hybrid_dim(n_qubits, n_modes, cutoff)?;
        let mut amps = vec![C64::new(0.0, 0.0); total];
        amps[0] = C64::new(1.0, 0.0);
        Ok(HybridState {
            n_qubits,
            n_modes,
            cutoff,
            q_dim,
            amps,
        })
    }

    fn from_inputs(circuit: &HybridCircuit, cutoff: usize) -> Result<Self, Error> {
        let (q_dim, total) = checked_hybrid_dim(circuit.n_qubits, circuit.n_modes, cutoff)?;
        let mut amps = vec![C64::new(0.0, 0.0); total];
        amps[0] = C64::new(1.0, 0.0);
        let mut fock_len = 1usize;
        for (mode, st) in circuit.mode_inputs.iter().enumerate() {
            let local = st.fock_amplitudes(cutoff).ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "mode {mode} input {st:?} is mixed; HybridState supports pure inputs"
                ))
            })?;
            for n in (0..cutoff).rev() {
                let factor = local[n];
                let dst_off = n * fock_len * q_dim;
                for i in 0..fock_len {
                    // Only qubit state 0 is initially populated.
                    amps[dst_off + i * q_dim] = amps[i * q_dim] * factor;
                }
            }
            fock_len *= cutoff;
        }
        Ok(HybridState {
            n_qubits: circuit.n_qubits,
            n_modes: circuit.n_modes,
            cutoff,
            q_dim,
            amps,
        })
    }

    /// Execute a measurement-free [`HybridCircuit`].
    pub fn from_circuit(circuit: &HybridCircuit, cutoff: usize) -> Result<Self, Error> {
        if !circuit.is_unitary() {
            return Err(Error::InvalidArgument(
                "circuit has measurements; use HybridState::run or HybridState::sample".into(),
            ));
        }
        let mut st = Self::from_inputs(circuit, cutoff)?;
        for op in &circuit.ops {
            st.apply_unitary_op(op)?;
        }
        Ok(st)
    }

    /// Run one shot of a [`HybridCircuit`] (with measurements), returning
    /// `(final_state, results)`.
    pub fn run<R: Rng + ?Sized>(
        circuit: &HybridCircuit,
        cutoff: usize,
        rng: &mut R,
    ) -> Result<(Self, Vec<f64>), Error> {
        let mut st = Self::from_inputs(circuit, cutoff)?;
        let mut results = Vec::with_capacity(circuit.n_results);
        st.run_ops(&circuit.ops, &mut results, rng)?;
        Ok((st, results))
    }

    /// Sample `shots` independent measurement trajectories from `circuit`,
    /// reusing the unitary prefix before the first measurement.
    pub fn sample<R: Rng + ?Sized>(
        circuit: &HybridCircuit,
        cutoff: usize,
        shots: usize,
        rng: &mut R,
    ) -> Result<Vec<Vec<f64>>, Error> {
        if shots == 0 {
            return Ok(Vec::new());
        }
        let first_meas = circuit
            .ops
            .iter()
            .position(is_measure_op)
            .unwrap_or(circuit.ops.len());
        let mut prefix = Self::from_inputs(circuit, cutoff)?;
        for op in &circuit.ops[..first_meas] {
            prefix.apply_unitary_op(op)?;
        }
        let suffix = &circuit.ops[first_meas..];
        if suffix.is_empty() {
            return Ok(vec![Vec::new(); shots]);
        }
        let n_res = circuit.n_results;
        if shots >= SAMPLE_PAR_MIN_SHOTS {
            let seeds: Vec<u64> = (0..shots).map(|_| rng.random()).collect();
            seeds
                .into_par_iter()
                .map(|seed| {
                    let mut local_rng = StdRng::seed_from_u64(seed);
                    let mut st = prefix.clone();
                    let mut res = Vec::with_capacity(n_res);
                    st.run_ops(suffix, &mut res, &mut local_rng)?;
                    Ok(res)
                })
                .collect()
        } else {
            let mut out = Vec::with_capacity(shots);
            for _ in 0..shots {
                let mut st = prefix.clone();
                let mut res = Vec::with_capacity(n_res);
                st.run_ops(suffix, &mut res, rng)?;
                out.push(res);
            }
            Ok(out)
        }
    }

    fn run_ops<R: Rng + ?Sized>(
        &mut self,
        ops: &[HybridOp],
        results: &mut Vec<f64>,
        rng: &mut R,
    ) -> Result<(), Error> {
        for op in ops {
            match *op {
                HybridOp::MeasureQubit { qubit } => {
                    let b = self.measure_qubit(qubit, rng)?;
                    results.push(if b { 1.0 } else { 0.0 });
                }
                HybridOp::MeasurePhotons { mode } => {
                    let n = self.measure_photons(mode, rng)?;
                    results.push(n as f64);
                }
                HybridOp::MeasureThreshold { mode } => {
                    let c = self.measure_threshold(mode, rng)?;
                    results.push(if c { 1.0 } else { 0.0 });
                }
                HybridOp::MeasureHomodyne { phi, mode } => {
                    let q = self.measure_homodyne(phi, mode, rng)?;
                    results.push(q);
                }
                ref other => self.apply_unitary_op(other)?,
            }
        }
        Ok(())
    }

    fn apply_unitary_op(&mut self, op: &HybridOp) -> Result<(), Error> {
        match op {
            HybridOp::Qubit1 { qubit, u } => self.apply_qubit_1(*qubit, *u),
            HybridOp::QubitCx { control, target } => self.cx_qubit(*control, *target),
            HybridOp::QubitCz { a, b } => self.cz_qubit(*a, *b),
            HybridOp::QubitSwap { a, b } => self.swap_qubit(*a, *b),
            HybridOp::Rotation { phi, mode } => self.rotation(*phi, *mode),
            HybridOp::Displacement { r, phi, mode } => self.displace(*r, *phi, *mode),
            HybridOp::Squeezing { r, phi, mode } => self.squeeze(*r, *phi, *mode),
            HybridOp::QuadraticPhase { s, mode } => self.quadratic_phase(*s, *mode),
            HybridOp::CubicPhase { gamma, mode } => self.cubic_phase(*gamma, *mode),
            HybridOp::Kerr { kappa, mode } => self.kerr(*kappa, *mode),
            HybridOp::Snap { phases, mode } => self.snap(phases, *mode),
            HybridOp::BeamSplitter { theta, phi, a, b } => self.beam_splitter(*theta, *phi, *a, *b),
            HybridOp::TwoModeSqueezing { r, phi, a, b } => self.two_mode_squeeze(*r, *phi, *a, *b),
            HybridOp::ModeCx { s, a, b } => self.cx_mode(*s, *a, *b),
            HybridOp::ModeCz { s, a, b } => self.cz_mode(*s, *a, *b),
            HybridOp::CrossKerr { kappa, a, b } => self.cross_kerr(*kappa, *a, *b),
            HybridOp::Passive { modes, matrix } => self.apply_passive(modes, matrix),
            HybridOp::ConditionalDisplacement {
                r,
                phi,
                qubit,
                mode,
            } => self.conditional_displace(*r, *phi, *qubit, *mode),
            HybridOp::ConditionalRotation { theta, qubit, mode } => {
                self.conditional_rotation(*theta, *qubit, *mode)
            }
            HybridOp::ConditionalSqueezing {
                r,
                phi,
                qubit,
                mode,
            } => self.conditional_squeeze(*r, *phi, *qubit, *mode),
            HybridOp::ConditionalSnap {
                phases,
                qubit,
                mode,
            } => self.conditional_snap(phases, *qubit, *mode),
            HybridOp::ConditionalBeamSplitter {
                theta,
                phi,
                qubit,
                a,
                b,
            } => self.conditional_beam_splitter(*theta, *phi, *qubit, *a, *b),
            HybridOp::JaynesCummings {
                theta,
                phi,
                qubit,
                mode,
            } => self.jaynes_cummings(*theta, *phi, *qubit, *mode),
            HybridOp::AntiJaynesCummings {
                theta,
                phi,
                qubit,
                mode,
            } => self.anti_jaynes_cummings(*theta, *phi, *qubit, *mode),
            HybridOp::MeasureQubit { .. }
            | HybridOp::MeasurePhotons { .. }
            | HybridOp::MeasureThreshold { .. }
            | HybridOp::MeasureHomodyne { .. } => Err(Error::InvalidArgument(
                "unexpected measurement in unitary execution".into(),
            )),
        }
    }

    // -----------------------------------------------------------------------
    // Accessors & observables
    // -----------------------------------------------------------------------

    /// Number of qubits.
    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    /// Number of qumodes.
    pub fn n_modes(&self) -> usize {
        self.n_modes
    }

    /// Local Fock cutoff `d` per qumode.
    pub fn cutoff(&self) -> usize {
        self.cutoff
    }

    /// Full mixed-radix statevector slice (`2^{n_qubits} · cutoff^{n_modes}`).
    pub fn amplitudes(&self) -> &[C64] {
        &self.amps
    }

    /// Probabilities `|ψ_i|²` for every mixed-radix basis state.
    pub fn probabilities(&self) -> Vec<f64> {
        self.amps.iter().map(|z| z.norm_sqr()).collect()
    }

    /// Amplitude `⟨qubit_bits; pattern | ψ⟩`.
    pub fn amplitude(&self, qubit_bits: usize, pattern: &[usize]) -> Result<C64, Error> {
        if qubit_bits >= self.q_dim {
            return Err(Error::InvalidArgument(format!(
                "qubit_bits {qubit_bits} out of range for {} qubits",
                self.n_qubits
            )));
        }
        if pattern.len() != self.n_modes {
            return Err(Error::InvalidArgument(format!(
                "pattern length {} != n_modes {}",
                pattern.len(),
                self.n_modes
            )));
        }
        let mut idx = qubit_bits;
        let mut stride = self.q_dim;
        for &n in pattern {
            if n >= self.cutoff {
                return Ok(C64::new(0.0, 0.0));
            }
            idx += n * stride;
            stride *= self.cutoff;
        }
        Ok(self.amps[idx])
    }

    /// Probability `|⟨qubit_bits; pattern | ψ⟩|²`.
    pub fn probability(&self, qubit_bits: usize, pattern: &[usize]) -> Result<f64, Error> {
        Ok(self.amplitude(qubit_bits, pattern)?.norm_sqr())
    }

    /// Squared norm `⟨ψ|ψ⟩`.
    pub fn norm_sqr(&self) -> f64 {
        self.amps.iter().map(|z| z.norm_sqr()).sum()
    }

    /// Cutoff truncation error `1 − ⟨ψ|ψ⟩`.
    pub fn truncation_error(&self) -> f64 {
        (1.0 - self.norm_sqr()).max(0.0)
    }

    /// Probability `P(q = 1)` of measuring `1` on `qubit`.
    pub fn qubit_prob_one(&self, qubit: usize) -> Result<f64, Error> {
        self.check_q(qubit)?;
        let bit = 1usize << qubit;
        let mut p1 = 0.0;
        for (i, z) in self.amps.iter().enumerate() {
            if (i & bit) != 0 {
                p1 += z.norm_sqr();
            }
        }
        Ok(p1)
    }

    /// Pauli expectation `⟨Z_q⟩ = P(0) − P(1)` on `qubit`.
    pub fn qubit_z(&self, qubit: usize) -> Result<f64, Error> {
        self.check_q(qubit)?;
        let bit = 1usize << qubit;
        let mut acc = 0.0;
        for (i, z) in self.amps.iter().enumerate() {
            if (i & bit) == 0 {
                acc += z.norm_sqr();
            } else {
                acc -= z.norm_sqr();
            }
        }
        Ok(acc)
    }

    /// Pauli expectation `⟨X_q⟩` on `qubit`.
    pub fn qubit_x(&self, qubit: usize) -> Result<f64, Error> {
        self.check_q(qubit)?;
        let bit = 1usize << qubit;
        let block = bit << 1;
        let mut acc = 0.0;
        for chunk in self.amps.chunks_exact(block) {
            for j in 0..bit {
                acc += 2.0 * (chunk[j].conj() * chunk[j + bit]).re;
            }
        }
        Ok(acc)
    }

    /// Pauli expectation `⟨Y_q⟩` on `qubit`.
    pub fn qubit_y(&self, qubit: usize) -> Result<f64, Error> {
        self.check_q(qubit)?;
        let bit = 1usize << qubit;
        let block = bit << 1;
        let mut acc = 0.0;
        for chunk in self.amps.chunks_exact(block) {
            for j in 0..bit {
                acc += 2.0 * (chunk[j].conj() * chunk[j + bit]).im;
            }
        }
        Ok(acc)
    }

    /// Mean photon number `⟨n̂⟩` on qumode `mode`.
    pub fn mean_photons(&self, mode: usize) -> Result<f64, Error> {
        self.check_m(mode)?;
        let d = self.cutoff;
        let stride = self.mode_stride(mode);
        let block = stride * d;
        let mut total = 0.0;
        for chunk in self.amps.chunks_exact(block) {
            for n in 1..d {
                let w: f64 = chunk[n * stride..(n + 1) * stride]
                    .iter()
                    .map(|z| z.norm_sqr())
                    .sum();
                total += (n as f64) * w;
            }
        }
        Ok(total)
    }

    /// `⟨x̂⟩ = ⟨a + a†⟩` on qumode `mode`.
    pub fn mean_x(&self, mode: usize) -> Result<f64, Error> {
        Ok(2.0 * self.mean_a(mode)?.re)
    }

    /// `⟨p̂⟩ = ⟨−i(a − a†)⟩` on qumode `mode`.
    pub fn mean_p(&self, mode: usize) -> Result<f64, Error> {
        Ok(2.0 * self.mean_a(mode)?.im)
    }

    fn mean_a(&self, mode: usize) -> Result<C64, Error> {
        self.check_m(mode)?;
        let d = self.cutoff;
        let stride = self.mode_stride(mode);
        let block = stride * d;
        let mut acc = C64::new(0.0, 0.0);
        for chunk in self.amps.chunks_exact(block) {
            for n in 0..d.saturating_sub(1) {
                let sq = ((n + 1) as f64).sqrt();
                let s0 = &chunk[n * stride..(n + 1) * stride];
                let s1 = &chunk[(n + 1) * stride..(n + 2) * stride];
                for (&a0, &a1) in s0.iter().zip(s1) {
                    acc += a0.conj() * a1 * sq;
                }
            }
        }
        Ok(acc)
    }

    // -----------------------------------------------------------------------
    // Stride & validation helpers
    // -----------------------------------------------------------------------

    #[inline]
    fn mode_stride(&self, mode: usize) -> usize {
        self.q_dim * self.cutoff.pow(mode as u32)
    }

    fn check_q(&self, q: usize) -> Result<(), Error> {
        if q < self.n_qubits {
            Ok(())
        } else {
            Err(Error::InvalidArgument(format!(
                "qubit {q} out of range for {} qubits",
                self.n_qubits
            )))
        }
    }

    fn check_q_pair(&self, a: usize, b: usize) -> Result<(), Error> {
        self.check_q(a)?;
        self.check_q(b)?;
        if a == b {
            return Err(Error::InvalidArgument(format!(
                "qubit {a} used more than once"
            )));
        }
        Ok(())
    }

    fn check_m(&self, m: usize) -> Result<(), Error> {
        if m < self.n_modes {
            Ok(())
        } else {
            Err(Error::InvalidArgument(format!(
                "mode {m} out of range for {} modes",
                self.n_modes
            )))
        }
    }

    fn check_m_pair(&self, a: usize, b: usize) -> Result<(), Error> {
        self.check_m(a)?;
        self.check_m(b)?;
        if a == b {
            return Err(Error::InvalidArgument(format!(
                "mode {a} used more than once"
            )));
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Qubit gates (contiguous inside each 2^{n_q} slice)
    // -----------------------------------------------------------------------

    /// Apply an arbitrary `2 × 2` unitary on `qubit`.
    pub fn apply_qubit_1(&mut self, qubit: usize, u: [[C64; 2]; 2]) -> Result<(), Error> {
        self.check_q(qubit)?;
        let bit = 1usize << qubit;
        let block = bit << 1;
        let (u00, u01, u10, u11) = (u[0][0], u[0][1], u[1][0], u[1][1]);
        let kernel = |chunk: &mut [C64]| {
            for j in 0..bit {
                let a = chunk[j];
                let b = chunk[j + bit];
                chunk[j] = u00 * a + u01 * b;
                chunk[j + bit] = u10 * a + u11 * b;
            }
        };
        if self.amps.len() >= PAR_MIN_LEN {
            self.amps.par_chunks_exact_mut(block).for_each(kernel);
        } else {
            self.amps.chunks_exact_mut(block).for_each(kernel);
        }
        Ok(())
    }

    /// Hadamard `H` on `qubit`.
    pub fn h(&mut self, qubit: usize) -> Result<(), Error> {
        let s = C64::new(FRAC_1_SQRT_2, 0.0);
        self.apply_qubit_1(qubit, [[s, s], [s, -s]])
    }

    /// Pauli `X` on `qubit`.
    pub fn x(&mut self, qubit: usize) -> Result<(), Error> {
        let z = C64::new(0.0, 0.0);
        let o = C64::new(1.0, 0.0);
        self.apply_qubit_1(qubit, [[z, o], [o, z]])
    }

    /// Pauli `Y` on `qubit`.
    pub fn y(&mut self, qubit: usize) -> Result<(), Error> {
        let z = C64::new(0.0, 0.0);
        let i = C64::new(0.0, 1.0);
        self.apply_qubit_1(qubit, [[z, -i], [i, z]])
    }

    /// Pauli `Z` on `qubit`.
    pub fn z(&mut self, qubit: usize) -> Result<(), Error> {
        let z = C64::new(0.0, 0.0);
        let o = C64::new(1.0, 0.0);
        self.apply_qubit_1(qubit, [[o, z], [z, -o]])
    }

    /// Phase gate `S = diag(1, i)` on `qubit`.
    pub fn s(&mut self, qubit: usize) -> Result<(), Error> {
        self.p(std::f64::consts::FRAC_PI_2, qubit)
    }

    /// `S† = diag(1, −i)` on `qubit`.
    pub fn sdg(&mut self, qubit: usize) -> Result<(), Error> {
        self.p(-std::f64::consts::FRAC_PI_2, qubit)
    }

    /// `T = diag(1, e^{iπ/4})` on `qubit`.
    pub fn t(&mut self, qubit: usize) -> Result<(), Error> {
        self.p(std::f64::consts::FRAC_PI_4, qubit)
    }

    /// `T† = diag(1, e^{−iπ/4})` on `qubit`.
    pub fn tdg(&mut self, qubit: usize) -> Result<(), Error> {
        self.p(-std::f64::consts::FRAC_PI_4, qubit)
    }

    /// Phase gate `P(λ) = diag(1, e^{iλ})` on `qubit`.
    pub fn p(&mut self, lam: f64, qubit: usize) -> Result<(), Error> {
        let z = C64::new(0.0, 0.0);
        self.apply_qubit_1(
            qubit,
            [[C64::new(1.0, 0.0), z], [z, C64::from_polar(1.0, lam)]],
        )
    }

    /// `Rx(θ)` on `qubit`.
    pub fn rx(&mut self, theta: f64, qubit: usize) -> Result<(), Error> {
        let (s, c) = (0.5 * theta).sin_cos();
        let cc = C64::new(c, 0.0);
        let is = C64::new(0.0, -s);
        self.apply_qubit_1(qubit, [[cc, is], [is, cc]])
    }

    /// `Ry(θ)` on `qubit`.
    pub fn ry(&mut self, theta: f64, qubit: usize) -> Result<(), Error> {
        let (s, c) = (0.5 * theta).sin_cos();
        self.apply_qubit_1(
            qubit,
            [
                [C64::new(c, 0.0), C64::new(-s, 0.0)],
                [C64::new(s, 0.0), C64::new(c, 0.0)],
            ],
        )
    }

    /// `Rz(θ)` on `qubit`.
    pub fn rz(&mut self, theta: f64, qubit: usize) -> Result<(), Error> {
        let z = C64::new(0.0, 0.0);
        self.apply_qubit_1(
            qubit,
            [
                [C64::from_polar(1.0, -0.5 * theta), z],
                [z, C64::from_polar(1.0, 0.5 * theta)],
            ],
        )
    }

    /// Qubit `CX` (`control`, `target`).
    pub fn cx_qubit(&mut self, control: usize, target: usize) -> Result<(), Error> {
        self.check_q_pair(control, target)?;
        let c_bit = 1usize << control;
        let t_bit = 1usize << target;
        let q_dim = self.q_dim;
        let kernel = |slice: &mut [C64]| {
            for q in 0..q_dim {
                if (q & c_bit) != 0 && (q & t_bit) == 0 {
                    slice.swap(q, q | t_bit);
                }
            }
        };
        if self.amps.len() >= PAR_MIN_LEN {
            self.amps.par_chunks_exact_mut(q_dim).for_each(kernel);
        } else {
            self.amps.chunks_exact_mut(q_dim).for_each(kernel);
        }
        Ok(())
    }

    /// Qubit `CZ` on `(a, b)`.
    pub fn cz_qubit(&mut self, a: usize, b: usize) -> Result<(), Error> {
        self.check_q_pair(a, b)?;
        let mask = (1usize << a) | (1usize << b);
        let q_dim = self.q_dim;
        let kernel = |slice: &mut [C64]| {
            for (q, z) in slice.iter_mut().enumerate() {
                if (q & mask) == mask {
                    *z = -*z;
                }
            }
        };
        if self.amps.len() >= PAR_MIN_LEN {
            self.amps.par_chunks_exact_mut(q_dim).for_each(kernel);
        } else {
            self.amps.chunks_exact_mut(q_dim).for_each(kernel);
        }
        Ok(())
    }

    /// Qubit `SWAP` on `(a, b)`.
    pub fn swap_qubit(&mut self, a: usize, b: usize) -> Result<(), Error> {
        self.check_q_pair(a, b)?;
        let ba = 1usize << a;
        let bb = 1usize << b;
        let q_dim = self.q_dim;
        let kernel = |slice: &mut [C64]| {
            for q in 0..q_dim {
                if (q & ba) != 0 && (q & bb) == 0 {
                    slice.swap(q, (q ^ ba) | bb);
                }
            }
        };
        if self.amps.len() >= PAR_MIN_LEN {
            self.amps.par_chunks_exact_mut(q_dim).for_each(kernel);
        } else {
            self.amps.chunks_exact_mut(q_dim).for_each(kernel);
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Qumode gates
    // -----------------------------------------------------------------------

    /// Rotation `R(φ)` on `mode`.
    pub fn rotation(&mut self, phi: f64, mode: usize) -> Result<(), Error> {
        self.check_m(mode)?;
        if phi == 0.0 {
            return Ok(());
        }
        let phases: Vec<C64> = (0..self.cutoff)
            .map(|n| C64::from_polar(1.0, phi * (n as f64)))
            .collect();
        self.apply_mode_diag(mode, &phases);
        Ok(())
    }

    /// Kerr `K(κ)` on `mode`.
    pub fn kerr(&mut self, kappa: f64, mode: usize) -> Result<(), Error> {
        self.check_m(mode)?;
        if kappa == 0.0 {
            return Ok(());
        }
        let phases: Vec<C64> = (0..self.cutoff)
            .map(|n| {
                let nf = n as f64;
                C64::from_polar(1.0, kappa * nf * nf)
            })
            .collect();
        self.apply_mode_diag(mode, &phases);
        Ok(())
    }

    /// SNAP gate `Σ_n e^{i θ_n} |n⟩⟨n|` on `mode`.
    pub fn snap(&mut self, phases: &[f64], mode: usize) -> Result<(), Error> {
        self.check_m(mode)?;
        let table: Vec<C64> = (0..self.cutoff)
            .map(|n| {
                let th = phases.get(n).copied().unwrap_or(0.0);
                C64::from_polar(1.0, th)
            })
            .collect();
        self.apply_mode_diag(mode, &table);
        Ok(())
    }

    /// Displacement `D(r e^{iφ})` on `mode`.
    pub fn displace(&mut self, r: f64, phi: f64, mode: usize) -> Result<(), Error> {
        self.check_m(mode)?;
        if r == 0.0 {
            return Ok(());
        }
        let u = displacement_matrix(C64::from_polar(r, phi), self.cutoff);
        self.apply_mode_dense_1(mode, &u);
        Ok(())
    }

    /// Squeezing `S(r e^{iφ})` on `mode`.
    pub fn squeeze(&mut self, r: f64, phi: f64, mode: usize) -> Result<(), Error> {
        self.check_m(mode)?;
        if r == 0.0 {
            return Ok(());
        }
        let u = squeezing_matrix(r, phi, self.cutoff);
        self.apply_mode_dense_1(mode, &u);
        Ok(())
    }

    /// Quadratic phase `P(s)` on `mode`.
    pub fn quadratic_phase(&mut self, s: f64, mode: usize) -> Result<(), Error> {
        self.check_m(mode)?;
        if s == 0.0 {
            return Ok(());
        }
        let u = quadratic_phase_matrix(s, self.cutoff);
        self.apply_mode_dense_1(mode, &u);
        Ok(())
    }

    /// Cubic phase `V(γ)` on `mode`.
    pub fn cubic_phase(&mut self, gamma: f64, mode: usize) -> Result<(), Error> {
        self.check_m(mode)?;
        if gamma == 0.0 {
            return Ok(());
        }
        let u = cubic_phase_matrix(gamma, self.cutoff);
        self.apply_mode_dense_1(mode, &u);
        Ok(())
    }

    /// Beam splitter `BS(θ, φ)` on `(a, b)`.
    pub fn beam_splitter(&mut self, theta: f64, phi: f64, a: usize, b: usize) -> Result<(), Error> {
        self.check_m_pair(a, b)?;
        let blocks = beam_splitter_blocks(theta, phi, self.cutoff);
        self.apply_mode_sum_blocks(a, b, &blocks, None);
        Ok(())
    }

    /// Two-mode squeezing `S₂(r e^{iφ})` on `(a, b)`.
    pub fn two_mode_squeeze(&mut self, r: f64, phi: f64, a: usize, b: usize) -> Result<(), Error> {
        self.check_m_pair(a, b)?;
        if r == 0.0 {
            return Ok(());
        }
        let blocks = two_mode_squeezing_blocks(r, phi, self.cutoff);
        self.apply_mode_diff_blocks(a, b, &blocks);
        Ok(())
    }

    /// `CX(s)` on modes `(a, b)`.
    pub fn cx_mode(&mut self, s: f64, a: usize, b: usize) -> Result<(), Error> {
        self.check_m_pair(a, b)?;
        if s == 0.0 {
            return Ok(());
        }
        self.apply_mode_quad_2(a, b, s, true);
        Ok(())
    }

    /// `CZ(s)` on modes `(a, b)`.
    pub fn cz_mode(&mut self, s: f64, a: usize, b: usize) -> Result<(), Error> {
        self.check_m_pair(a, b)?;
        if s == 0.0 {
            return Ok(());
        }
        self.apply_mode_quad_2(a, b, s, false);
        Ok(())
    }

    /// Cross-Kerr `CK(κ)` on modes `(a, b)`.
    pub fn cross_kerr(&mut self, kappa: f64, a: usize, b: usize) -> Result<(), Error> {
        self.check_m_pair(a, b)?;
        if kappa == 0.0 {
            return Ok(());
        }
        let d = self.cutoff;
        let sa = self.mode_stride(a);
        let sb = self.mode_stride(b);
        let mut phases = Vec::with_capacity(d * d);
        for na in 0..d {
            for nb in 0..d {
                phases.push(C64::from_polar(1.0, kappa * ((na * nb) as f64)));
            }
        }
        if self.amps.len() >= PAR_MIN_LEN {
            self.amps.par_iter_mut().enumerate().for_each(|(i, z)| {
                let na = (i / sa) % d;
                let nb = (i / sb) % d;
                *z *= phases[na * d + nb];
            });
        } else {
            for (i, z) in self.amps.iter_mut().enumerate() {
                let na = (i / sa) % d;
                let nb = (i / sb) % d;
                *z *= phases[na * d + nb];
            }
        }
        Ok(())
    }

    /// Passive linear-optics unitary `U` on `modes`.
    pub fn apply_passive(&mut self, modes: &[usize], matrix: &[C64]) -> Result<(), Error> {
        let k = modes.len();
        if matrix.len() != k * k {
            return Err(Error::InvalidArgument(format!(
                "passive block on {k} modes needs {} entries, got {}",
                k * k,
                matrix.len()
            )));
        }
        for (i, &m) in modes.iter().enumerate() {
            self.check_m(m)?;
            if modes[..i].contains(&m) {
                return Err(Error::InvalidArgument(format!(
                    "mode {m} used more than once"
                )));
            }
        }
        if k == 0 {
            return Ok(());
        }
        if k == 1 {
            return self.rotation(matrix[0].arg(), modes[0]);
        }
        let (ops, diag_phases) = givens_decompose(matrix, k);
        for (i, &phi) in diag_phases.iter().enumerate() {
            if phi.abs() > 1e-15 {
                self.rotation(phi, modes[i])?;
            }
        }
        for (i, j, u2) in ops.into_iter().rev() {
            let blocks = build_two_mode_blocks(&u2, 2 * (self.cutoff - 1));
            self.apply_mode_sum_blocks(modes[i], modes[j], &blocks, None);
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Native hybrid qubit–qumode gates (exact analytic kernels)
    // -----------------------------------------------------------------------

    /// Conditional displacement `|0⟩⟨0|_q ⊗ D(+r e^{iφ})_m + |1⟩⟨1|_q ⊗ D(−r e^{iφ})_m`.
    pub fn conditional_displace(
        &mut self,
        r: f64,
        phi: f64,
        qubit: usize,
        mode: usize,
    ) -> Result<(), Error> {
        self.check_q(qubit)?;
        self.check_m(mode)?;
        if r == 0.0 {
            return Ok(());
        }
        let alpha = C64::from_polar(r, phi);
        let u_pos = displacement_matrix(alpha, self.cutoff);
        let u_neg = displacement_matrix(-alpha, self.cutoff);
        self.apply_conditional_dense_1(qubit, mode, &u_pos, &u_neg);
        Ok(())
    }

    /// Conditional rotation `exp(i θ Z_q n̂_m) = |0⟩⟨0|_q ⊗ R(+θ)_m + |1⟩⟨1|_q ⊗ R(−θ)_m`.
    pub fn conditional_rotation(
        &mut self,
        theta: f64,
        qubit: usize,
        mode: usize,
    ) -> Result<(), Error> {
        self.check_q(qubit)?;
        self.check_m(mode)?;
        if theta == 0.0 {
            return Ok(());
        }
        let d = self.cutoff;
        let p0: Vec<C64> = (0..d)
            .map(|n| C64::from_polar(1.0, theta * (n as f64)))
            .collect();
        let p1: Vec<C64> = p0.iter().map(|z| z.conj()).collect();
        self.apply_conditional_diag(qubit, mode, &p0, &p1);
        Ok(())
    }

    /// Conditional squeezing `|0⟩⟨0|_q ⊗ S(+r e^{iφ})_m + |1⟩⟨1|_q ⊗ S(−r e^{iφ})_m`.
    pub fn conditional_squeeze(
        &mut self,
        r: f64,
        phi: f64,
        qubit: usize,
        mode: usize,
    ) -> Result<(), Error> {
        self.check_q(qubit)?;
        self.check_m(mode)?;
        if r == 0.0 {
            return Ok(());
        }
        let u_pos = squeezing_matrix(r, phi, self.cutoff);
        let u_neg = squeezing_matrix(-r, phi, self.cutoff);
        self.apply_conditional_dense_1(qubit, mode, &u_pos, &u_neg);
        Ok(())
    }

    /// Conditional SNAP `exp(i Z_q Σ_n θ_n |n⟩⟨n|_m)`.
    pub fn conditional_snap(
        &mut self,
        phases: &[f64],
        qubit: usize,
        mode: usize,
    ) -> Result<(), Error> {
        self.check_q(qubit)?;
        self.check_m(mode)?;
        let d = self.cutoff;
        let p0: Vec<C64> = (0..d)
            .map(|n| C64::from_polar(1.0, phases.get(n).copied().unwrap_or(0.0)))
            .collect();
        let p1: Vec<C64> = p0.iter().map(|z| z.conj()).collect();
        self.apply_conditional_diag(qubit, mode, &p0, &p1);
        Ok(())
    }

    /// Conditional beam splitter `|0⟩⟨0|_q ⊗ BS(+θ, φ)_{a,b} + |1⟩⟨1|_q ⊗ BS(−θ, φ)_{a,b}`.
    pub fn conditional_beam_splitter(
        &mut self,
        theta: f64,
        phi: f64,
        qubit: usize,
        a: usize,
        b: usize,
    ) -> Result<(), Error> {
        self.check_q(qubit)?;
        self.check_m_pair(a, b)?;
        let b_pos = beam_splitter_blocks(theta, phi, self.cutoff);
        let b_neg = beam_splitter_blocks(-theta, phi, self.cutoff);
        self.apply_mode_sum_blocks(a, b, &b_pos, Some((qubit, &b_neg)));
        Ok(())
    }

    /// Exact closed-form Jaynes–Cummings gate
    /// `exp(−i θ (e^{iφ} σ_+ a + e^{-iφ} σ_- a†))` on `(qubit, mode)` in
    /// `O(2^{n_q} d^m)` in-place.
    pub fn jaynes_cummings(
        &mut self,
        theta: f64,
        phi: f64,
        qubit: usize,
        mode: usize,
    ) -> Result<(), Error> {
        self.check_q(qubit)?;
        self.check_m(mode)?;
        if theta == 0.0 || self.cutoff <= 1 {
            return Ok(());
        }
        let d = self.cutoff;
        let stride = self.mode_stride(mode);
        let block = stride * d;
        let q_bit = 1usize << qubit;
        let e_pos = C64::from_polar(1.0, phi);
        let e_neg = e_pos.conj();
        let neg_i = C64::new(0.0, -1.0);

        let cs: Vec<(f64, C64, C64)> = (0..(d - 1))
            .map(|n| {
                let omega = theta * ((n + 1) as f64).sqrt();
                let (s, c) = omega.sin_cos();
                (c, neg_i * e_pos * s, neg_i * e_neg * s)
            })
            .collect();

        // Pairs (|1, n⟩, |0, n+1⟩) for n = 0..d-2 are mutually disjoint inside each block!
        let kernel = |chunk: &mut [C64]| {
            for (n, &(c, u01, u10)) in cs.iter().enumerate() {
                let off_e = n * stride;
                let off_g = (n + 1) * stride;
                let mut inner = 0usize;
                while inner < stride {
                    if (inner & q_bit) == 0 {
                        let i_e = off_e + (inner | q_bit); // |1, n⟩
                        let i_g = off_g + inner; // |0, n+1⟩
                        let ae = chunk[i_e];
                        let ag = chunk[i_g];
                        chunk[i_e] = ae * c + u01 * ag;
                        chunk[i_g] = u10 * ae + ag * c;
                    }
                    inner += 1;
                }
            }
        };
        if self.amps.len() >= PAR_MIN_LEN {
            self.amps.par_chunks_exact_mut(block).for_each(kernel);
        } else {
            self.amps.chunks_exact_mut(block).for_each(kernel);
        }
        Ok(())
    }

    /// Exact closed-form Anti-Jaynes–Cummings gate
    /// `exp(−i θ (e^{iφ} σ_+ a† + e^{-iφ} σ_- a))` on `(qubit, mode)` in
    /// `O(2^{n_q} d^m)` in-place.
    pub fn anti_jaynes_cummings(
        &mut self,
        theta: f64,
        phi: f64,
        qubit: usize,
        mode: usize,
    ) -> Result<(), Error> {
        self.check_q(qubit)?;
        self.check_m(mode)?;
        if theta == 0.0 || self.cutoff <= 1 {
            return Ok(());
        }
        let d = self.cutoff;
        let stride = self.mode_stride(mode);
        let block = stride * d;
        let q_bit = 1usize << qubit;
        let e_pos = C64::from_polar(1.0, phi);
        let e_neg = e_pos.conj();
        let neg_i = C64::new(0.0, -1.0);

        let cs: Vec<(f64, C64, C64)> = (0..(d - 1))
            .map(|n| {
                let omega = theta * ((n + 1) as f64).sqrt();
                let (s, c) = omega.sin_cos();
                // Subspace (|1, n+1⟩, |0, n⟩): σ_+ a† |0, n⟩ = √(n+1) |1, n+1⟩.
                (c, neg_i * e_pos * s, neg_i * e_neg * s)
            })
            .collect();

        let kernel = |chunk: &mut [C64]| {
            for (n, &(c, u01, u10)) in cs.iter().enumerate() {
                let off_e = (n + 1) * stride;
                let off_g = n * stride;
                let mut inner = 0usize;
                while inner < stride {
                    if (inner & q_bit) == 0 {
                        let i_e = off_e + (inner | q_bit); // |1, n+1⟩
                        let i_g = off_g + inner; // |0, n⟩
                        let ae = chunk[i_e];
                        let ag = chunk[i_g];
                        chunk[i_e] = ae * c + u01 * ag;
                        chunk[i_g] = u10 * ae + ag * c;
                    }
                    inner += 1;
                }
            }
        };
        if self.amps.len() >= PAR_MIN_LEN {
            self.amps.par_chunks_exact_mut(block).for_each(kernel);
        } else {
            self.amps.chunks_exact_mut(block).for_each(kernel);
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Measurements
    // -----------------------------------------------------------------------

    /// Projective computational-basis (Z) measurement on `qubit`.
    pub fn measure_qubit<R: Rng + ?Sized>(
        &mut self,
        qubit: usize,
        rng: &mut R,
    ) -> Result<bool, Error> {
        self.check_q(qubit)?;
        let bit = 1usize << qubit;
        let mut p0 = 0.0;
        let mut p1 = 0.0;
        for (i, z) in self.amps.iter().enumerate() {
            if (i & bit) == 0 {
                p0 += z.norm_sqr();
            } else {
                p1 += z.norm_sqr();
            }
        }
        let u = rng.random::<f64>() * (p0 + p1);
        let outcome = u >= p0;
        let prob = if outcome { p1 } else { p0 };
        let inv = if prob > 0.0 { 1.0 / prob.sqrt() } else { 1.0 };
        let zero = C64::new(0.0, 0.0);
        for (i, z) in self.amps.iter_mut().enumerate() {
            if ((i & bit) != 0) == outcome {
                *z *= inv;
            } else {
                *z = zero;
            }
        }
        Ok(outcome)
    }

    /// Postselect `qubit` onto `outcome` (`false` = `|0⟩`, `true` = `|1⟩`),
    /// returning the probability of that outcome.
    pub fn postselect_qubit(&mut self, qubit: usize, outcome: bool) -> Result<f64, Error> {
        self.check_q(qubit)?;
        let bit = 1usize << qubit;
        let mut prob = 0.0;
        for (i, z) in self.amps.iter().enumerate() {
            if ((i & bit) != 0) == outcome {
                prob += z.norm_sqr();
            }
        }
        if prob <= 1e-300 {
            return Err(Error::InvalidArgument(
                "postselected qubit outcome has zero probability".into(),
            ));
        }
        let inv = 1.0 / prob.sqrt();
        let zero = C64::new(0.0, 0.0);
        for (i, z) in self.amps.iter_mut().enumerate() {
            if ((i & bit) != 0) == outcome {
                *z *= inv;
            } else {
                *z = zero;
            }
        }
        Ok(prob)
    }

    /// Destructive photon-number (PNR) measurement on `mode` (resets mode to `|0⟩`).
    pub fn measure_photons<R: Rng + ?Sized>(
        &mut self,
        mode: usize,
        rng: &mut R,
    ) -> Result<usize, Error> {
        self.check_m(mode)?;
        let d = self.cutoff;
        let stride = self.mode_stride(mode);
        let block = stride * d;

        let mut probs = vec![0.0; d];
        for chunk in self.amps.chunks_exact(block) {
            for (n, p) in probs.iter_mut().enumerate() {
                *p += chunk[n * stride..(n + 1) * stride]
                    .iter()
                    .map(|z| z.norm_sqr())
                    .sum::<f64>();
            }
        }
        let total: f64 = probs.iter().sum();
        let u = rng.random::<f64>() * total;
        let mut acc = 0.0;
        let last_nonzero = probs.iter().rposition(|&p| p > 0.0).unwrap_or(0);
        let mut outcome = last_nonzero;
        for (n, &p) in probs.iter().enumerate() {
            acc += p;
            if u < acc {
                outcome = n;
                break;
            }
        }
        let inv = if probs[outcome] > 0.0 {
            1.0 / probs[outcome].sqrt()
        } else {
            1.0
        };
        let zero = C64::new(0.0, 0.0);
        for chunk in self.amps.chunks_exact_mut(block) {
            if outcome != 0 {
                let src = outcome * stride;
                for j in 0..stride {
                    chunk[j] = chunk[src + j] * inv;
                }
            } else {
                for z in &mut chunk[..stride] {
                    *z *= inv;
                }
            }
            for z in &mut chunk[stride..] {
                *z = zero;
            }
        }
        Ok(outcome)
    }

    /// Destructive threshold measurement on `mode` (`true` for ≥ 1 photon).
    pub fn measure_threshold<R: Rng + ?Sized>(
        &mut self,
        mode: usize,
        rng: &mut R,
    ) -> Result<bool, Error> {
        Ok(self.measure_photons(mode, rng)? > 0)
    }

    /// Destructive homodyne measurement of `x̂ cos φ + p̂ sin φ` on `mode`.
    pub fn measure_homodyne<R: Rng + ?Sized>(
        &mut self,
        phi: f64,
        mode: usize,
        rng: &mut R,
    ) -> Result<f64, Error> {
        self.check_m(mode)?;
        if phi != 0.0 {
            self.rotation(-phi, mode)?;
        }
        let d = self.cutoff;
        let stride = self.mode_stride(mode);
        let block = stride * d;

        // Reduced `d × d` density matrix of `mode`.
        let mut rho = vec![C64::new(0.0, 0.0); d * d];
        for chunk in self.amps.chunks_exact(block) {
            for m in 0..d {
                let sm = &chunk[m * stride..(m + 1) * stride];
                for n in m..d {
                    let sn = &chunk[n * stride..(n + 1) * stride];
                    let mut acc = C64::new(0.0, 0.0);
                    for (&am, &an) in sm.iter().zip(sn) {
                        acc += am * an.conj();
                    }
                    rho[m * d + n] += acc;
                    if m != n {
                        rho[n * d + m] += acc.conj();
                    }
                }
            }
        }

        // Classical turning point of level d is √(4d) (for ħ = 2); add margin 5.
        let q_max = (4.0 * (d as f64)).sqrt() + 5.0;
        let n_grid = (d * 24).clamp(256, 2048);
        let dq = 2.0 * q_max / (n_grid as f64);
        let mut psi = vec![0.0; d];
        let mut cdf = Vec::with_capacity(n_grid);
        let mut cum = 0.0;
        for i in 0..n_grid {
            let q = -q_max + (i as f64 + 0.5) * dq;
            hermite_fock_wavefunctions(q, d, &mut psi);
            let mut dens = 0.0;
            for m in 0..d {
                let pm = psi[m];
                let row = m * d;
                dens += rho[row + m].re * pm * pm;
                for n in (m + 1)..d {
                    dens += 2.0 * rho[row + n].re * pm * psi[n];
                }
            }
            cum += dens.max(0.0) * dq;
            cdf.push(cum);
        }
        let u = rng.random::<f64>() * cum;
        let idx = cdf.partition_point(|&c| c < u).min(n_grid - 1);
        let outcome = -q_max + (idx as f64 + rng.random::<f64>()) * dq;

        hermite_fock_wavefunctions(outcome, d, &mut psi);
        let zero = C64::new(0.0, 0.0);
        let mut norm_sqr = 0.0;
        for chunk in self.amps.chunks_exact_mut(block) {
            for j in 0..stride {
                let mut acc = zero;
                for (n, &w) in psi.iter().enumerate() {
                    acc += chunk[n * stride + j] * w;
                }
                chunk[j] = acc;
                norm_sqr += acc.norm_sqr();
            }
            for z in &mut chunk[stride..] {
                *z = zero;
            }
        }
        if norm_sqr > 0.0 {
            let inv = 1.0 / norm_sqr.sqrt();
            for chunk in self.amps.chunks_exact_mut(block) {
                for z in &mut chunk[..stride] {
                    *z *= inv;
                }
            }
        }
        Ok(outcome)
    }

    // -----------------------------------------------------------------------
    // Low-level mixed-radix qumode / hybrid kernels
    // -----------------------------------------------------------------------

    fn apply_mode_diag(&mut self, mode: usize, phases: &[C64]) {
        let d = self.cutoff;
        let stride = self.mode_stride(mode);
        let block = stride * d;
        let kernel = |chunk: &mut [C64]| {
            for (n, &ph) in phases.iter().enumerate() {
                if ph != C64::new(1.0, 0.0) {
                    for z in &mut chunk[n * stride..(n + 1) * stride] {
                        *z *= ph;
                    }
                }
            }
        };
        if self.amps.len() >= PAR_MIN_LEN {
            self.amps.par_chunks_exact_mut(block).for_each(kernel);
        } else {
            self.amps.chunks_exact_mut(block).for_each(kernel);
        }
    }

    fn apply_conditional_diag(&mut self, qubit: usize, mode: usize, p0: &[C64], p1: &[C64]) {
        let d = self.cutoff;
        let stride = self.mode_stride(mode);
        let block = stride * d;
        let q_bit = 1usize << qubit;
        let kernel = |chunk: &mut [C64]| {
            for n in 0..d {
                let ph0 = p0[n];
                let ph1 = p1[n];
                let slice = &mut chunk[n * stride..(n + 1) * stride];
                for (j, z) in slice.iter_mut().enumerate() {
                    *z *= if (j & q_bit) == 0 { ph0 } else { ph1 };
                }
            }
        };
        if self.amps.len() >= PAR_MIN_LEN {
            self.amps.par_chunks_exact_mut(block).for_each(kernel);
        } else {
            self.amps.chunks_exact_mut(block).for_each(kernel);
        }
    }

    fn apply_mode_dense_1(&mut self, mode: usize, u: &[C64]) {
        let d = self.cutoff;
        let stride = self.mode_stride(mode);
        let block = stride * d;
        let kernel = |chunk: &mut [C64]| {
            let mut tmp = vec![C64::new(0.0, 0.0); block];
            for out_n in 0..d {
                let dst = &mut tmp[out_n * stride..(out_n + 1) * stride];
                let row = &u[out_n * d..(out_n + 1) * d];
                for (in_n, &u_val) in row.iter().enumerate() {
                    if u_val == C64::new(0.0, 0.0) {
                        continue;
                    }
                    let src = &chunk[in_n * stride..(in_n + 1) * stride];
                    for (dz, &sz) in dst.iter_mut().zip(src) {
                        *dz += u_val * sz;
                    }
                }
            }
            chunk.copy_from_slice(&tmp);
        };
        if self.amps.len() >= PAR_MIN_LEN {
            self.amps.par_chunks_exact_mut(block).for_each(kernel);
        } else {
            self.amps.chunks_exact_mut(block).for_each(kernel);
        }
    }

    fn apply_conditional_dense_1(&mut self, qubit: usize, mode: usize, u0: &[C64], u1: &[C64]) {
        let d = self.cutoff;
        let stride = self.mode_stride(mode);
        let block = stride * d;
        let q_bit = 1usize << qubit;
        let kernel = |chunk: &mut [C64]| {
            let mut tmp = vec![C64::new(0.0, 0.0); block];
            for out_n in 0..d {
                let dst = &mut tmp[out_n * stride..(out_n + 1) * stride];
                let r0 = &u0[out_n * d..(out_n + 1) * d];
                let r1 = &u1[out_n * d..(out_n + 1) * d];
                for in_n in 0..d {
                    let v0 = r0[in_n];
                    let v1 = r1[in_n];
                    let src = &chunk[in_n * stride..(in_n + 1) * stride];
                    for j in 0..stride {
                        let u_val = if (j & q_bit) == 0 { v0 } else { v1 };
                        dst[j] += u_val * src[j];
                    }
                }
            }
            chunk.copy_from_slice(&tmp);
        };
        if self.amps.len() >= PAR_MIN_LEN {
            self.amps.par_chunks_exact_mut(block).for_each(kernel);
        } else {
            self.amps.chunks_exact_mut(block).for_each(kernel);
        }
    }

    /// Apply photon-sum-conserving `(s+1)×(s+1)` blocks on modes `(a, b)`,
    /// optionally conditioned on `qubit` (`blocks_q0` when `q=0`, `blocks_q1` when `q=1`).
    fn apply_mode_sum_blocks(
        &mut self,
        a: usize,
        b: usize,
        blocks_q0: &[Vec<C64>],
        cond_q1: Option<(usize, &[Vec<C64>])>,
    ) {
        let d = self.cutoff;
        let sa = self.mode_stride(a);
        let sb = self.mode_stride(b);
        let (s_lo, s_hi) = if sa < sb { (sa, sb) } else { (sb, sa) };
        let hi_block = s_hi * d;
        let n_mid = s_hi / (s_lo * d);
        let max_s = 2 * (d - 1);

        let kernel = |hi_chunk: &mut [C64]| {
            let mut in_buf = vec![C64::new(0.0, 0.0); d];
            let mut out_buf = vec![C64::new(0.0, 0.0); d];
            for mid in 0..n_mid {
                let mid_off = mid * (s_lo * d);
                for inner in 0..s_lo {
                    let base = mid_off + inner;
                    let blocks = match cond_q1 {
                        Some((q, b1)) if (inner & (1usize << q)) != 0 => b1,
                        _ => blocks_q0,
                    };
                    for (s, blk) in blocks.iter().enumerate().take(max_s + 1) {
                        let na_lo = s.saturating_sub(d - 1);
                        let na_hi = s.min(d - 1);
                        let dim_s = s + 1;
                        let span = na_hi - na_lo + 1;
                        for (k, dst) in in_buf.iter_mut().enumerate().take(span) {
                            let na = na_lo + k;
                            let nb = s - na;
                            *dst = hi_chunk[base + na * sa + nb * sb];
                        }
                        for (out_k, dst) in out_buf.iter_mut().enumerate().take(span) {
                            let out_a = na_lo + out_k;
                            let row = out_a * dim_s;
                            let mut acc = C64::new(0.0, 0.0);
                            for (in_k, &v) in in_buf.iter().enumerate().take(span) {
                                acc += blk[row + (na_lo + in_k)] * v;
                            }
                            *dst = acc;
                        }
                        for (k, &v) in out_buf.iter().enumerate().take(span) {
                            let na = na_lo + k;
                            let nb = s - na;
                            hi_chunk[base + na * sa + nb * sb] = v;
                        }
                    }
                }
            }
        };
        if self.amps.len() >= PAR_MIN_LEN {
            self.amps.par_chunks_exact_mut(hi_block).for_each(kernel);
        } else {
            self.amps.chunks_exact_mut(hi_block).for_each(kernel);
        }
    }

    fn apply_mode_diff_blocks(&mut self, a: usize, b: usize, blocks: &[Vec<C64>]) {
        let d = self.cutoff;
        let sa = self.mode_stride(a);
        let sb = self.mode_stride(b);
        let (s_lo, s_hi) = if sa < sb { (sa, sb) } else { (sb, sa) };
        let hi_block = s_hi * d;
        let n_mid = s_hi / (s_lo * d);

        let kernel = |hi_chunk: &mut [C64]| {
            let mut in_buf = vec![C64::new(0.0, 0.0); d];
            let mut out_buf = vec![C64::new(0.0, 0.0); d];
            for mid in 0..n_mid {
                let mid_off = mid * (s_lo * d);
                for inner in 0..s_lo {
                    let base = mid_off + inner;
                    for (b_idx, blk) in blocks.iter().enumerate() {
                        let diff = b_idx as isize - (d as isize - 1);
                        let len = d - diff.unsigned_abs();
                        let oa = diff.max(0) as usize;
                        let ob = (-diff).max(0) as usize;
                        for (k, dst) in in_buf.iter_mut().enumerate().take(len) {
                            *dst = hi_chunk[base + (k + oa) * sa + (k + ob) * sb];
                        }
                        for (out_k, dst) in out_buf.iter_mut().enumerate().take(len) {
                            let row = out_k * len;
                            let mut acc = C64::new(0.0, 0.0);
                            for (in_k, &v) in in_buf.iter().enumerate().take(len) {
                                acc += blk[row + in_k] * v;
                            }
                            *dst = acc;
                        }
                        for (k, &v) in out_buf.iter().enumerate().take(len) {
                            hi_chunk[base + (k + oa) * sa + (k + ob) * sb] = v;
                        }
                    }
                }
            }
        };
        if self.amps.len() >= PAR_MIN_LEN {
            self.amps.par_chunks_exact_mut(hi_block).for_each(kernel);
        } else {
            self.amps.chunks_exact_mut(hi_block).for_each(kernel);
        }
    }

    fn apply_mode_quad_2(&mut self, a: usize, b: usize, s: f64, is_cx: bool) {
        let d = self.cutoff;
        let qe = cached_quad_eigen(d);
        let big = qe.big;
        let sign = if is_cx { -0.5 * s } else { 0.5 * s };
        let mut phase = vec![C64::new(0.0, 0.0); big * big];
        for ka in 0..big {
            let la = qe.evals[ka] * sign;
            let row = ka * big;
            for kb in 0..big {
                phase[row + kb] = C64::from_polar(1.0, la * qe.evals[kb]);
            }
        }
        let sa = self.mode_stride(a);
        let sb = self.mode_stride(b);
        let (s_lo, s_hi) = if sa < sb { (sa, sb) } else { (sb, sa) };
        let hi_block = s_hi * d;
        let n_mid = s_hi / (s_lo * d);

        let kernel = |hi_chunk: &mut [C64]| {
            let mut mid = vec![C64::new(0.0, 0.0); big * d];
            let mut quad = vec![C64::new(0.0, 0.0); big * big];
            let mut back = vec![C64::new(0.0, 0.0); big * d];
            for mid_i in 0..n_mid {
                let mid_off = mid_i * (s_lo * d);
                for inner in 0..s_lo {
                    let base = mid_off + inner;
                    for ka in 0..big {
                        for nb in 0..d {
                            let mut acc = C64::new(0.0, 0.0);
                            for na in 0..d {
                                acc += hi_chunk[base + na * sa + nb * sb] * qe.vx[na * big + ka];
                            }
                            mid[ka * d + nb] = acc;
                        }
                    }
                    if is_cx {
                        for ka in 0..big {
                            let mrow = &mid[ka * d..(ka + 1) * d];
                            let prow = &phase[ka * big..(ka + 1) * big];
                            let qrow = &mut quad[ka * big..(ka + 1) * big];
                            for kb in 0..big {
                                let mut acc = C64::new(0.0, 0.0);
                                for (nb, &mv) in mrow.iter().enumerate() {
                                    acc += mv * qe.vp[nb * big + kb].conj();
                                }
                                qrow[kb] = acc * prow[kb];
                            }
                        }
                        for ka in 0..big {
                            let qrow = &quad[ka * big..(ka + 1) * big];
                            let brow = &mut back[ka * d..(ka + 1) * d];
                            for (nb, dst) in brow.iter_mut().enumerate() {
                                let vrow = &qe.vp[nb * big..(nb + 1) * big];
                                let mut acc = C64::new(0.0, 0.0);
                                for (&qv, &vv) in qrow.iter().zip(vrow) {
                                    acc += qv * vv;
                                }
                                *dst = acc;
                            }
                        }
                    } else {
                        for ka in 0..big {
                            let mrow = &mid[ka * d..(ka + 1) * d];
                            let prow = &phase[ka * big..(ka + 1) * big];
                            let qrow = &mut quad[ka * big..(ka + 1) * big];
                            for kb in 0..big {
                                let mut acc = C64::new(0.0, 0.0);
                                for (nb, &mv) in mrow.iter().enumerate() {
                                    acc += mv * qe.vx[nb * big + kb];
                                }
                                qrow[kb] = acc * prow[kb];
                            }
                        }
                        for ka in 0..big {
                            let qrow = &quad[ka * big..(ka + 1) * big];
                            let brow = &mut back[ka * d..(ka + 1) * d];
                            for (nb, dst) in brow.iter_mut().enumerate() {
                                let vrow = &qe.vx[nb * big..(nb + 1) * big];
                                let mut acc = C64::new(0.0, 0.0);
                                for (&qv, &vv) in qrow.iter().zip(vrow) {
                                    acc += qv * vv;
                                }
                                *dst = acc;
                            }
                        }
                    }
                    for na in 0..d {
                        let vrow = &qe.vx[na * big..(na + 1) * big];
                        for nb in 0..d {
                            let mut acc = C64::new(0.0, 0.0);
                            for (ka, &vv) in vrow.iter().enumerate() {
                                acc += back[ka * d + nb] * vv;
                            }
                            hi_chunk[base + na * sa + nb * sb] = acc;
                        }
                    }
                }
            }
        };
        if self.amps.len() >= PAR_MIN_LEN {
            self.amps.par_chunks_exact_mut(hi_block).for_each(kernel);
        } else {
            self.amps.chunks_exact_mut(hi_block).for_each(kernel);
        }
    }
}

fn is_measure_op(op: &HybridOp) -> bool {
    matches!(
        op,
        HybridOp::MeasureQubit { .. }
            | HybridOp::MeasurePhotons { .. }
            | HybridOp::MeasureThreshold { .. }
            | HybridOp::MeasureHomodyne { .. }
    )
}

fn checked_hybrid_dim(
    n_qubits: usize,
    n_modes: usize,
    cutoff: usize,
) -> Result<(usize, usize), Error> {
    if cutoff == 0 && n_modes > 0 {
        return Err(Error::InvalidArgument("Fock cutoff must be >= 1".into()));
    }
    let q_dim = 1usize
        .checked_shl(n_qubits as u32)
        .ok_or_else(|| Error::InvalidArgument(format!("2^{n_qubits} overflows usize")))?;
    let m_dim = if n_modes == 0 {
        1usize
    } else {
        cutoff
            .checked_pow(n_modes as u32)
            .ok_or_else(|| Error::InvalidArgument(format!("{cutoff}^{n_modes} overflows usize")))?
    };
    let total = q_dim
        .checked_mul(m_dim)
        .ok_or_else(|| Error::InvalidArgument("hybrid dimension overflows usize".into()))?;
    Ok((q_dim, total))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, PI};

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() < tol, "{a} vs {b} (diff {})", (a - b).abs());
    }

    #[test]
    fn jaynes_cummings_and_anti_jc_rabi_oscillations() {
        let d = 12;
        // Vacuum Rabi oscillation: |1, 0⟩ --JC(θ, φ)--> cos θ |1, 0⟩ - i e^{-iφ} sin θ |0, 1⟩
        let theta = 0.7;
        let phi = 0.4;
        let mut c = HybridCircuit::new(1, 1);
        c.x(0).unwrap().jaynes_cummings(theta, phi, 0, 0).unwrap();
        let st = HybridState::from_circuit(&c, d).unwrap();
        let a10 = st.amplitude(1, &[0]).unwrap();
        let a01 = st.amplitude(0, &[1]).unwrap();
        assert!((a10 - C64::new(theta.cos(), 0.0)).norm() < 1e-12);
        let want01 = C64::new(0.0, -1.0) * C64::from_polar(theta.sin(), -phi);
        assert!((a01 - want01).norm() < 1e-12);
        close(st.mean_photons(0).unwrap(), theta.sin().powi(2), 1e-12);
        close(
            st.qubit_z(0).unwrap(),
            -theta.cos().powi(2) + theta.sin().powi(2),
            1e-12,
        );

        // n-photon Rabi frequency scales as √(n+1): |1, 3⟩ oscillates at 2θ
        let mut c2 = HybridCircuit::new(1, 1);
        c2.set_mode_input(0, ModeState::Fock(3)).unwrap();
        c2.x(0)
            .unwrap()
            .jaynes_cummings(FRAC_PI_2 / 2.0, 0.0, 0, 0)
            .unwrap();
        let st2 = HybridState::from_circuit(&c2, d).unwrap();
        // At θ = π/4 and n = 3, Ω = 2 * π/4 = π/2 => complete swap |1, 3⟩ -> -i |0, 4⟩!
        close(st2.probability(0, &[4]).unwrap(), 1.0, 1e-12);
        close(st2.probability(1, &[3]).unwrap(), 0.0, 1e-12);

        // Anti-Jaynes-Cummings: |0, 0⟩ --AJC(π/2, 0)--> -i |1, 1⟩
        let mut c3 = HybridCircuit::new(1, 1);
        c3.anti_jaynes_cummings(FRAC_PI_2, 0.0, 0, 0).unwrap();
        let st3 = HybridState::from_circuit(&c3, d).unwrap();
        close(st3.probability(1, &[1]).unwrap(), 1.0, 1e-12);
    }

    #[test]
    fn conditional_displacement_cat_preparation_and_snap() {
        // Prepare even & odd cat states via H -> CD(α) -> H -> postselect qubit!
        let d = 32;
        let r = 1.6;
        let phi = 0.25;
        let mut c = HybridCircuit::new(1, 1);
        c.h(0)
            .unwrap()
            .conditional_displace(r, phi, 0, 0)
            .unwrap()
            .h(0)
            .unwrap();
        let st = HybridState::from_circuit(&c, d).unwrap();

        let mut even = st.clone();
        even.postselect_qubit(0, false).unwrap();
        let want_even = ModeState::Cat {
            r,
            phi,
            parity: 0.0,
        }
        .fock_amplitudes(d)
        .unwrap();
        for (n, &w) in want_even.iter().enumerate() {
            assert!(
                (even.amplitude(0, &[n]).unwrap() - w).norm() < 1e-10,
                "even cat n={n}"
            );
        }

        let mut odd = st;
        odd.postselect_qubit(0, true).unwrap();
        let want_odd = ModeState::Cat {
            r,
            phi,
            parity: 1.0,
        }
        .fock_amplitudes(d)
        .unwrap();
        for (n, &w) in want_odd.iter().enumerate() {
            assert!(
                (odd.amplitude(1, &[n]).unwrap() - w).norm() < 1e-10,
                "odd cat n={n}"
            );
        }

        // Dispersive parity check via CR(π/2): even cat stays |+⟩ -> H -> |0⟩, odd -> |1⟩!
        even.h(0).unwrap();
        even.conditional_rotation(FRAC_PI_2, 0, 0).unwrap();
        even.h(0).unwrap();
        close(even.qubit_prob_one(0).unwrap(), 0.0, 1e-10);

        // Conditional SNAP + SNAP phase verification
        let mut cs = HybridCircuit::new(1, 1);
        cs.set_mode_input(0, ModeState::Coherent { r: 0.8, phi: 0.0 })
            .unwrap();
        cs.h(0)
            .unwrap()
            .snap(vec![0.0, PI, 0.5], 0)
            .unwrap()
            .conditional_snap(vec![0.0, 0.3, -0.4], 0, 0)
            .unwrap();
        let sts = HybridState::from_circuit(&cs, 16).unwrap();
        let coh = ModeState::Coherent { r: 0.8, phi: 0.0 }
            .fock_amplitudes(16)
            .unwrap();
        let want_q0_n1 = coh[1] * FRAC_1_SQRT_2 * C64::from_polar(1.0, PI + 0.3);
        let want_q1_n1 = coh[1] * FRAC_1_SQRT_2 * C64::from_polar(1.0, PI - 0.3);
        assert!((sts.amplitude(0, &[1]).unwrap() - want_q0_n1).norm() < 1e-12);
        assert!((sts.amplitude(1, &[1]).unwrap() - want_q1_n1).norm() < 1e-12);
    }

    #[test]
    fn multi_qubit_multi_mode_cbs_and_sampling() {
        // Conditional beam splitter creates entanglement between qubit and two modes:
        // |+⟩_q |1, 0⟩_m --CBS(π/4, 0)--> (|0⟩(|1,0⟩+|0,1⟩) + |1⟩(|1,0⟩-|0,1⟩)) / 2
        let mut c = HybridCircuit::new(2, 2);
        c.set_mode_input(0, ModeState::Fock(1)).unwrap();
        c.h(0)
            .unwrap()
            .cx_qubit(0, 1)
            .unwrap()
            .conditional_beam_splitter(std::f64::consts::FRAC_PI_4, 0.0, 0, 0, 1)
            .unwrap();
        let st = HybridState::from_circuit(&c, 6).unwrap();
        close(st.norm_sqr(), 1.0, 1e-12);
        // BS(+π/4)|1,0⟩ = (|1,0⟩ + |0,1⟩)/√2; BS(-π/4)|1,0⟩ = (|1,0⟩ - |0,1⟩)/√2
        close(st.probability(0b00, &[1, 0]).unwrap(), 0.25, 1e-12);
        close(st.probability(0b00, &[0, 1]).unwrap(), 0.25, 1e-12);
        close(st.probability(0b11, &[1, 0]).unwrap(), 0.25, 1e-12);
        close(st.probability(0b11, &[0, 1]).unwrap(), 0.25, 1e-12);

        // Sampling test
        c.measure_qubit(0).unwrap().measure_photons(0).unwrap();
        let mut rng = StdRng::seed_from_u64(99);
        let rows = HybridState::sample(&c, 6, 400, &mut rng).unwrap();
        assert_eq!(rows.len(), 400);
        let q1_rate: f64 = rows.iter().map(|r| r[0]).sum::<f64>() / 400.0;
        let n0_rate: f64 = rows.iter().map(|r| r[1]).sum::<f64>() / 400.0;
        assert!((q1_rate - 0.5).abs() < 0.08);
        assert!((n0_rate - 0.5).abs() < 0.08);
    }
}
