//! CV gates, defined as in Strawberry Fields with ħ = 2, `x̂ = a + a†` and
//! `p̂ = −i(a − a†)`.

use super::arg::Arg;

/// A single-mode gate.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum ModeGate {
    /// `R(φ) = exp(iφ n̂)`: `a → e^{iφ} a`.
    Rotation {
        /// The angle φ.
        phi: Arg,
    },
    /// `D(α) = exp(α a† − α* a)` with `α = r e^{iφ}`: `x̂ → x̂ + 2 Re α`,
    /// `p̂ → p̂ + 2 Im α`.
    Displacement {
        /// Magnitude r.
        r: Arg,
        /// Phase φ.
        phi: Arg,
    },
    /// `S(z) = exp((z* a² − z a†²)/2)` with `z = r e^{iφ}`; for φ = 0,
    /// `x̂ → e^{−r} x̂` and `p̂ → e^{r} p̂`.
    Squeezing {
        /// Squeezing magnitude r.
        r: Arg,
        /// Squeezing angle φ.
        phi: Arg,
    },
    /// `P(s) = exp(i s x̂²/(2ħ))`: `p̂ → p̂ + s x̂`.
    QuadraticPhase {
        /// Shear s.
        s: Arg,
    },
    /// `V(γ) = exp(i γ x̂³/(3ħ))`: `p̂ → p̂ + γ x̂²`. Non-Gaussian.
    CubicPhase {
        /// Strength γ.
        gamma: Arg,
    },
    /// `K(κ) = exp(iκ n̂²)`. Non-Gaussian.
    Kerr {
        /// Strength κ.
        kappa: Arg,
    },
}

/// A two-mode gate on modes `(a, b)`, in that order.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum TwoModeGate {
    /// `BS(θ, φ) = exp(θ(e^{iφ} a b† − e^{−iφ} a† b))`: coherent amplitudes
    /// `(α, β)` become `(t α − r* β, r α + t β)` with `t = cos θ`,
    /// `r = e^{iφ} sin θ`. θ = π/4 is 50:50.
    BeamSplitter {
        /// Transmissivity angle θ.
        theta: Arg,
        /// Phase φ.
        phi: Arg,
    },
    /// `S₂(z) = exp(z* a b − z a† b†)` with `z = r e^{iφ}`.
    TwoModeSqueezing {
        /// Squeezing magnitude r.
        r: Arg,
        /// Squeezing angle φ.
        phi: Arg,
    },
    /// `CX(s) = exp(−i s x̂_a p̂_b/ħ)`: `x̂_b → x̂_b + s x̂_a`,
    /// `p̂_a → p̂_a − s p̂_b`. The CV counterpart of CNOT.
    ControlledX {
        /// Strength s.
        s: Arg,
    },
    /// `CZ(s) = exp(i s x̂_a x̂_b/ħ)`: `p̂_b → p̂_b + s x̂_a`,
    /// `p̂_a → p̂_a + s x̂_b`.
    ControlledZ {
        /// Strength s.
        s: Arg,
    },
    /// `CK(κ) = exp(iκ n̂_a n̂_b)`. Non-Gaussian.
    CrossKerr {
        /// Strength κ.
        kappa: Arg,
    },
}

impl ModeGate {
    /// The gate's arguments.
    pub fn args(&self) -> Vec<Arg> {
        match *self {
            ModeGate::Rotation { phi } => vec![phi],
            ModeGate::Displacement { r, phi } | ModeGate::Squeezing { r, phi } => vec![r, phi],
            ModeGate::QuadraticPhase { s } => vec![s],
            ModeGate::CubicPhase { gamma } => vec![gamma],
            ModeGate::Kerr { kappa } => vec![kappa],
        }
    }

    /// The gate with each argument replaced by `f` of it.
    pub(crate) fn map_args(self, f: impl Fn(Arg) -> Arg) -> ModeGate {
        match self {
            ModeGate::Rotation { phi } => ModeGate::Rotation { phi: f(phi) },
            ModeGate::Displacement { r, phi } => ModeGate::Displacement {
                r: f(r),
                phi: f(phi),
            },
            ModeGate::Squeezing { r, phi } => ModeGate::Squeezing {
                r: f(r),
                phi: f(phi),
            },
            ModeGate::QuadraticPhase { s } => ModeGate::QuadraticPhase { s: f(s) },
            ModeGate::CubicPhase { gamma } => ModeGate::CubicPhase { gamma: f(gamma) },
            ModeGate::Kerr { kappa } => ModeGate::Kerr { kappa: f(kappa) },
        }
    }

    /// Whether the gate maps Gaussian states to Gaussian states.
    pub fn is_gaussian(&self) -> bool {
        !matches!(self, ModeGate::CubicPhase { .. } | ModeGate::Kerr { .. })
    }

    /// Whether the gate is a linear-optics (passive Gaussian) element.
    pub fn is_passive(&self) -> bool {
        matches!(self, ModeGate::Rotation { .. })
    }

    /// Whether the gate commutes with photon number.
    pub fn conserves_photons(&self) -> bool {
        matches!(self, ModeGate::Rotation { .. } | ModeGate::Kerr { .. })
    }
}

impl TwoModeGate {
    /// The gate's arguments.
    pub fn args(&self) -> Vec<Arg> {
        match *self {
            TwoModeGate::BeamSplitter { theta, phi } => vec![theta, phi],
            TwoModeGate::TwoModeSqueezing { r, phi } => vec![r, phi],
            TwoModeGate::ControlledX { s } | TwoModeGate::ControlledZ { s } => vec![s],
            TwoModeGate::CrossKerr { kappa } => vec![kappa],
        }
    }

    /// The gate with each argument replaced by `f` of it.
    pub(crate) fn map_args(self, f: impl Fn(Arg) -> Arg) -> TwoModeGate {
        match self {
            TwoModeGate::BeamSplitter { theta, phi } => TwoModeGate::BeamSplitter {
                theta: f(theta),
                phi: f(phi),
            },
            TwoModeGate::TwoModeSqueezing { r, phi } => TwoModeGate::TwoModeSqueezing {
                r: f(r),
                phi: f(phi),
            },
            TwoModeGate::ControlledX { s } => TwoModeGate::ControlledX { s: f(s) },
            TwoModeGate::ControlledZ { s } => TwoModeGate::ControlledZ { s: f(s) },
            TwoModeGate::CrossKerr { kappa } => TwoModeGate::CrossKerr { kappa: f(kappa) },
        }
    }

    /// Whether the gate maps Gaussian states to Gaussian states.
    pub fn is_gaussian(&self) -> bool {
        !matches!(self, TwoModeGate::CrossKerr { .. })
    }

    /// Whether the gate is a linear-optics (passive Gaussian) element.
    pub fn is_passive(&self) -> bool {
        matches!(self, TwoModeGate::BeamSplitter { .. })
    }

    /// Whether the gate commutes with total photon number.
    pub fn conserves_photons(&self) -> bool {
        matches!(
            self,
            TwoModeGate::BeamSplitter { .. } | TwoModeGate::CrossKerr { .. }
        )
    }
}
