//! Circuits on qumodes.

use super::arg::{Arg, Measured};
use super::gate::{ModeGate, TwoModeGate};
use super::noise::CvChannel;
use super::state::ModeState;
use crate::error::Error;
use crate::parameter::Parameter;
use num_complex::Complex64 as C64;

/// Largest deviation of `U†U` from the identity accepted for a passive
/// block.
const UNITARY_TOLERANCE: f64 = 1e-9;

/// How a mode is measured. Measurement is destructive: afterwards the mode
/// is in the vacuum, so it can be reused as a fresh ancilla.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum Measurement {
    /// The quadrature `x̂ cos φ + p̂ sin φ`; one result.
    Homodyne {
        /// Quadrature angle φ (0 for x̂, π/2 for p̂).
        phi: Arg,
    },
    /// Heterodyne: the estimate of α; two results, Re α then Im α.
    Heterodyne,
    /// The photon number; one result.
    PhotonCount,
    /// Whether any photon arrived; one result, 0 or 1.
    Threshold,
}

impl Measurement {
    /// Number of result slots the measurement fills.
    pub fn n_results(&self) -> usize {
        match self {
            Measurement::Heterodyne => 2,
            _ => 1,
        }
    }

    fn map_args(self, f: impl Fn(Arg) -> Arg) -> Measurement {
        match self {
            Measurement::Homodyne { phi } => Measurement::Homodyne { phi: f(phi) },
            other => other,
        }
    }

    fn args(&self) -> Vec<Arg> {
        match *self {
            Measurement::Homodyne { phi } => vec![phi],
            _ => Vec::new(),
        }
    }
}

/// One step of a [`ModeCircuit`].
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum ModeOp {
    /// A single-mode gate.
    Gate {
        /// The gate.
        gate: ModeGate,
        /// The mode it acts on.
        mode: usize,
    },
    /// A two-mode gate.
    TwoModeGate {
        /// The gate.
        gate: TwoModeGate,
        /// The modes `(a, b)` in the gate's definition.
        modes: [usize; 2],
    },
    /// A linear-optics block: coherent amplitudes on `modes` become
    /// `U α`, i.e. `a†ᵢ → Σⱼ U_{ji} a†ⱼ`.
    Passive {
        /// The modes, in the matrix's order.
        modes: Vec<usize>,
        /// The `k × k` unitary, row-major.
        matrix: Vec<C64>,
    },
    /// A single-mode continuous-variable noise channel applied on `mode`.
    Channel {
        /// The CPTP noise channel.
        channel: CvChannel,
        /// The mode it acts on.
        mode: usize,
    },
    /// A measurement writing `measurement.n_results()` results from slot
    /// `result` on.
    Measure {
        /// The measured mode.
        mode: usize,
        /// What is measured.
        measurement: Measurement,
        /// The first result slot.
        result: usize,
    },
}

impl ModeOp {
    /// The modes the operation acts on.
    pub fn modes(&self) -> Vec<usize> {
        match self {
            ModeOp::Gate { mode, .. }
            | ModeOp::Channel { mode, .. }
            | ModeOp::Measure { mode, .. } => vec![*mode],
            ModeOp::TwoModeGate { modes, .. } => modes.to_vec(),
            ModeOp::Passive { modes, .. } => modes.clone(),
        }
    }

    /// The operation's arguments.
    pub fn args(&self) -> Vec<Arg> {
        match self {
            ModeOp::Gate { gate, .. } => gate.args(),
            ModeOp::TwoModeGate { gate, .. } => gate.args(),
            ModeOp::Passive { .. } | ModeOp::Channel { .. } => Vec::new(),
            ModeOp::Measure { measurement, .. } => measurement.args(),
        }
    }

    fn map_args(&self, f: impl Fn(Arg) -> Arg) -> ModeOp {
        match self {
            ModeOp::Gate { gate, mode } => ModeOp::Gate {
                gate: gate.map_args(f),
                mode: *mode,
            },
            ModeOp::TwoModeGate { gate, modes } => ModeOp::TwoModeGate {
                gate: gate.map_args(f),
                modes: *modes,
            },
            ModeOp::Measure {
                mode,
                measurement,
                result,
            } => ModeOp::Measure {
                mode: *mode,
                measurement: measurement.map_args(f),
                result: *result,
            },
            other => other.clone(),
        }
    }

    /// Whether the operation maps Gaussian states to Gaussian states.
    pub fn is_gaussian(&self) -> bool {
        match self {
            ModeOp::Gate { gate, .. } => gate.is_gaussian(),
            ModeOp::TwoModeGate { gate, .. } => gate.is_gaussian(),
            ModeOp::Passive { .. } => true,
            ModeOp::Channel { channel, .. } => matches!(
                channel,
                CvChannel::Loss { .. }
                    | CvChannel::ThermalLoss { .. }
                    | CvChannel::AdditiveNoise { .. }
            ),
            ModeOp::Measure { .. } => false,
        }
    }

    /// Whether the operation is a linear-optics gate.
    pub fn is_passive(&self) -> bool {
        match self {
            ModeOp::Gate { gate, .. } => gate.is_passive(),
            ModeOp::TwoModeGate { gate, .. } => gate.is_passive(),
            ModeOp::Passive { .. } => true,
            ModeOp::Channel { .. } | ModeOp::Measure { .. } => false,
        }
    }

    /// Whether the operation is a gate that commutes with total photon
    /// number.
    pub fn conserves_photons(&self) -> bool {
        match self {
            ModeOp::Gate { gate, .. } => gate.conserves_photons(),
            ModeOp::TwoModeGate { gate, .. } => gate.conserves_photons(),
            ModeOp::Passive { .. } => true,
            ModeOp::Channel { .. } | ModeOp::Measure { .. } => false,
        }
    }
}

/// A circuit on qumodes: each mode's input state, then gates and
/// measurements in order.
///
/// Gate arguments may be fixed values, linear forms in circuit
/// [`Parameter`]s (bound with [`bind`](Self::bind)), or linear forms in
/// earlier measurement results, which is how feed-forward is written:
///
/// ```
/// use ketrs::cv::ModeCircuit;
///
/// // Measure x on mode 0 and displace mode 1 by the outcome.
/// let mut c = ModeCircuit::new(2);
/// c.squeeze(1.0, 0.0, 0).two_mode_squeeze(0.5, 0.0, 0, 1);
/// let m = c.measure_x(0);
/// c.displace(0.5 * m, 0.0, 1);
/// assert_eq!(c.n_results(), 1);
/// ```
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ModeCircuit {
    inputs: Vec<ModeState>,
    ops: Vec<ModeOp>,
    n_parameters: usize,
    n_results: usize,
}

impl ModeCircuit {
    /// A circuit on `n_modes` modes, all starting in the vacuum.
    pub fn new(n_modes: usize) -> Self {
        ModeCircuit {
            inputs: vec![ModeState::Vacuum; n_modes],
            ..Default::default()
        }
    }

    /// Number of modes.
    pub fn n_modes(&self) -> usize {
        self.inputs.len()
    }

    /// Each mode's input state.
    pub fn inputs(&self) -> &[ModeState] {
        &self.inputs
    }

    /// The operations, in order.
    pub fn ops(&self) -> &[ModeOp] {
        &self.ops
    }

    /// Number of parameters; [`bind`](Self::bind) takes this many values.
    pub fn n_parameters(&self) -> usize {
        self.n_parameters
    }

    /// Number of measurement results a shot produces.
    pub fn n_results(&self) -> usize {
        self.n_results
    }

    /// Create a new parameter.
    pub fn parameter(&mut self) -> Parameter {
        self.n_parameters += 1;
        Parameter(self.n_parameters - 1)
    }

    /// Create `n` new parameters.
    pub fn parameters(&mut self, n: usize) -> Vec<Parameter> {
        (0..n).map(|_| self.parameter()).collect()
    }

    /// The circuit with parameters replaced by `values`.
    pub fn bind(&self, values: &[f64]) -> Result<ModeCircuit, Error> {
        if values.len() != self.n_parameters {
            return Err(Error::InvalidArgument(format!(
                "circuit has {} parameters, got {} values",
                self.n_parameters,
                values.len()
            )));
        }
        for (i, &v) in values.iter().enumerate() {
            if !v.is_finite() {
                return Err(Error::InvalidArgument(format!(
                    "parameter {i} value {v} is not finite"
                )));
            }
        }
        Ok(ModeCircuit {
            inputs: self.inputs.clone(),
            ops: self
                .ops
                .iter()
                .map(|op| op.map_args(|a| a.bind(values)))
                .collect(),
            n_parameters: 0,
            n_results: self.n_results,
        })
    }

    /// Set `mode`'s input state.
    pub fn set_input(&mut self, mode: usize, state: ModeState) -> Result<&mut Self, Error> {
        self.check_mode(mode)?;
        state.validate()?;
        self.inputs[mode] = state;
        Ok(self)
    }

    /// Whether every gate and input state is Gaussian. Homodyne and
    /// heterodyne measurements keep a circuit Gaussian; photon counting
    /// doesn't.
    pub fn is_gaussian(&self) -> bool {
        self.inputs.iter().all(ModeState::is_gaussian)
            && self.ops.iter().all(|op| match op {
                ModeOp::Measure { measurement, .. } => matches!(
                    measurement,
                    Measurement::Homodyne { .. } | Measurement::Heterodyne
                ),
                op => op.is_gaussian(),
            })
    }

    /// Whether every gate is a linear-optics element (measurements
    /// allowed).
    pub fn is_passive(&self) -> bool {
        self.ops
            .iter()
            .all(|op| matches!(op, ModeOp::Measure { .. }) || op.is_passive())
    }

    /// Whether every gate commutes with total photon number (measurements
    /// allowed).
    pub fn conserves_photons(&self) -> bool {
        self.ops
            .iter()
            .all(|op| matches!(op, ModeOp::Measure { .. }) || op.conserves_photons())
    }

    /// Whether the circuit has no measurements or noise channels.
    pub fn is_unitary(&self) -> bool {
        !self
            .ops
            .iter()
            .any(|op| matches!(op, ModeOp::Measure { .. } | ModeOp::Channel { .. }))
    }

    /// Whether the circuit contains any measurements.
    pub fn has_measurements(&self) -> bool {
        self.n_results > 0
    }

    /// Append a single-mode CPTP noise channel on `mode`.
    pub fn apply_channel(&mut self, channel: CvChannel, mode: usize) -> Result<&mut Self, Error> {
        self.push(ModeOp::Channel { channel, mode })
    }

    /// Append an operation after checking it.
    pub fn push(&mut self, op: ModeOp) -> Result<&mut Self, Error> {
        self.validate(&op)?;
        if let ModeOp::Measure { measurement, .. } = &op {
            self.n_results += measurement.n_results();
        }
        self.ops.push(op);
        Ok(self)
    }

    fn check_mode(&self, mode: usize) -> Result<(), Error> {
        if mode < self.n_modes() {
            Ok(())
        } else {
            Err(Error::InvalidArgument(format!(
                "mode {mode} out of range for {} modes",
                self.n_modes()
            )))
        }
    }

    fn check_arg(&self, arg: Arg) -> Result<(), Error> {
        match arg {
            Arg::Value(v) if !v.is_finite() => Err(Error::InvalidArgument(format!(
                "gate argument {v} is not finite"
            ))),
            Arg::Parameter { parameter, .. } if parameter.0 >= self.n_parameters => {
                Err(Error::InvalidArgument(format!(
                    "parameter {} is not one of this circuit's {}",
                    parameter.0, self.n_parameters
                )))
            }
            Arg::Result { result, .. } if result.0 >= self.n_results => {
                Err(Error::InvalidArgument(format!(
                    "result {} is not measured before this operation",
                    result.0
                )))
            }
            _ => Ok(()),
        }
    }

    fn validate(&self, op: &ModeOp) -> Result<(), Error> {
        let modes = op.modes();
        for (i, &m) in modes.iter().enumerate() {
            self.check_mode(m)?;
            if modes[..i].contains(&m) {
                return Err(Error::InvalidArgument(format!(
                    "mode {m} used more than once"
                )));
            }
        }
        for arg in op.args() {
            self.check_arg(arg)?;
        }
        match op {
            ModeOp::Passive { modes, matrix } => check_unitary(modes.len(), matrix),
            ModeOp::Channel { channel, .. } => channel.validate(),
            ModeOp::Measure { result, .. } if *result != self.n_results => {
                Err(Error::InvalidArgument(format!(
                    "measurement must write the next result slot, {}",
                    self.n_results
                )))
            }
            _ => Ok(()),
        }
    }

    fn add(&mut self, op: ModeOp) -> &mut Self {
        if let Err(e) = self.push(op.clone()) {
            panic!("invalid operation {op:?}: {e}");
        }
        self
    }

    fn gate(&mut self, gate: ModeGate, mode: usize) -> &mut Self {
        self.add(ModeOp::Gate { gate, mode })
    }

    fn gate2(&mut self, gate: TwoModeGate, a: usize, b: usize) -> &mut Self {
        self.add(ModeOp::TwoModeGate {
            gate,
            modes: [a, b],
        })
    }

    /// Append a measurement on `mode`, returning its first result slot without
    /// panicking on invalid modes or arguments.
    pub fn try_measure(
        &mut self,
        measurement: Measurement,
        mode: usize,
    ) -> Result<Measured, Error> {
        let result = self.n_results;
        self.push(ModeOp::Measure {
            mode,
            measurement,
            result,
        })?;
        Ok(Measured(result))
    }

    fn measure(&mut self, measurement: Measurement, mode: usize) -> Measured {
        match self.try_measure(measurement, mode) {
            Ok(m) => m,
            Err(e) => panic!("invalid measurement {measurement:?} on mode {mode}: {e}"),
        }
    }

    /// Rotation `R(φ)`.
    pub fn rotation(&mut self, phi: impl Into<Arg>, mode: usize) -> &mut Self {
        self.gate(ModeGate::Rotation { phi: phi.into() }, mode)
    }

    /// Displacement `D(r e^{iφ})`.
    pub fn displace(&mut self, r: impl Into<Arg>, phi: impl Into<Arg>, mode: usize) -> &mut Self {
        let (r, phi) = (r.into(), phi.into());
        self.gate(ModeGate::Displacement { r, phi }, mode)
    }

    /// Squeezing `S(r e^{iφ})`.
    pub fn squeeze(&mut self, r: impl Into<Arg>, phi: impl Into<Arg>, mode: usize) -> &mut Self {
        let (r, phi) = (r.into(), phi.into());
        self.gate(ModeGate::Squeezing { r, phi }, mode)
    }

    /// Quadratic phase `P(s)`.
    pub fn quadratic_phase(&mut self, s: impl Into<Arg>, mode: usize) -> &mut Self {
        self.gate(ModeGate::QuadraticPhase { s: s.into() }, mode)
    }

    /// Cubic phase `V(γ)`.
    pub fn cubic_phase(&mut self, gamma: impl Into<Arg>, mode: usize) -> &mut Self {
        self.gate(
            ModeGate::CubicPhase {
                gamma: gamma.into(),
            },
            mode,
        )
    }

    /// Kerr `K(κ)`.
    pub fn kerr(&mut self, kappa: impl Into<Arg>, mode: usize) -> &mut Self {
        self.gate(
            ModeGate::Kerr {
                kappa: kappa.into(),
            },
            mode,
        )
    }

    /// Beam splitter `BS(θ, φ)` on modes `(a, b)`.
    pub fn beam_splitter(
        &mut self,
        theta: impl Into<Arg>,
        phi: impl Into<Arg>,
        a: usize,
        b: usize,
    ) -> &mut Self {
        let (theta, phi) = (theta.into(), phi.into());
        self.gate2(TwoModeGate::BeamSplitter { theta, phi }, a, b)
    }

    /// Two-mode squeezing `S₂(r e^{iφ})` on modes `(a, b)`.
    pub fn two_mode_squeeze(
        &mut self,
        r: impl Into<Arg>,
        phi: impl Into<Arg>,
        a: usize,
        b: usize,
    ) -> &mut Self {
        let (r, phi) = (r.into(), phi.into());
        self.gate2(TwoModeGate::TwoModeSqueezing { r, phi }, a, b)
    }

    /// `CX(s)` with control `a` and target `b`.
    pub fn cx(&mut self, s: impl Into<Arg>, a: usize, b: usize) -> &mut Self {
        self.gate2(TwoModeGate::ControlledX { s: s.into() }, a, b)
    }

    /// `CZ(s)` on modes `a` and `b`.
    pub fn cz(&mut self, s: impl Into<Arg>, a: usize, b: usize) -> &mut Self {
        self.gate2(TwoModeGate::ControlledZ { s: s.into() }, a, b)
    }

    /// Cross-Kerr `CK(κ)` on modes `a` and `b`.
    pub fn cross_kerr(&mut self, kappa: impl Into<Arg>, a: usize, b: usize) -> &mut Self {
        let kappa = kappa.into();
        self.gate2(TwoModeGate::CrossKerr { kappa }, a, b)
    }

    /// A linear-optics block: coherent amplitudes on `modes` become `U α`
    /// for the row-major unitary `matrix`.
    pub fn passive(&mut self, modes: &[usize], matrix: Vec<C64>) -> Result<&mut Self, Error> {
        self.push(ModeOp::Passive {
            modes: modes.to_vec(),
            matrix,
        })
    }

    /// Measure the quadrature `x̂ cos φ + p̂ sin φ`.
    pub fn measure_homodyne(&mut self, phi: impl Into<Arg>, mode: usize) -> Measured {
        self.measure(Measurement::Homodyne { phi: phi.into() }, mode)
    }

    /// Measure `x̂`.
    pub fn measure_x(&mut self, mode: usize) -> Measured {
        self.measure_homodyne(0.0, mode)
    }

    /// Measure `p̂`.
    pub fn measure_p(&mut self, mode: usize) -> Measured {
        self.measure_homodyne(std::f64::consts::FRAC_PI_2, mode)
    }

    /// Heterodyne measurement; returns the slots of Re α and Im α.
    pub fn measure_heterodyne(&mut self, mode: usize) -> (Measured, Measured) {
        let re = self.measure(Measurement::Heterodyne, mode);
        (re, Measured(re.0 + 1))
    }

    /// Count photons.
    pub fn measure_photons(&mut self, mode: usize) -> Measured {
        self.measure(Measurement::PhotonCount, mode)
    }

    /// Detect whether any photon arrived (0 or 1).
    pub fn measure_threshold(&mut self, mode: usize) -> Measured {
        self.measure(Measurement::Threshold, mode)
    }
}

fn check_unitary(k: usize, u: &[C64]) -> Result<(), Error> {
    if u.len() != k * k {
        return Err(Error::InvalidArgument(format!(
            "passive block on {k} modes needs {} matrix entries, got {}",
            k * k,
            u.len()
        )));
    }
    if !u.iter().all(|z| z.re.is_finite() && z.im.is_finite()) {
        return Err(Error::InvalidArgument(
            "passive block matrix entries must be finite".into(),
        ));
    }
    for i in 0..k {
        for j in 0..k {
            let dot: C64 = (0..k).map(|r| u[r * k + i].conj() * u[r * k + j]).sum();
            let want = if i == j { 1.0 } else { 0.0 };
            let err = (dot - want).norm();
            if !err.is_finite() || err > UNITARY_TOLERANCE {
                return Err(Error::InvalidArgument(
                    "passive block matrix is not unitary".into(),
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builders_validate() {
        let mut c = ModeCircuit::new(2);
        assert!(c.set_input(2, ModeState::Fock(1)).is_err());
        let bad = ModeOp::TwoModeGate {
            gate: TwoModeGate::CrossKerr {
                kappa: Arg::Value(1.0),
            },
            modes: [1, 1],
        };
        assert!(c.push(bad).is_err());
        let nan = ModeOp::Gate {
            gate: ModeGate::Kerr {
                kappa: Arg::Value(f64::NAN),
            },
            mode: 0,
        };
        assert!(c.push(nan).is_err());
        let not_unitary = vec![C64::new(1.0, 0.0); 4];
        assert!(c.passive(&[0, 1], not_unitary).is_err());
        let early = ModeOp::Gate {
            gate: ModeGate::Rotation {
                phi: Measured(0).into(),
            },
            mode: 0,
        };
        assert!(c.push(early.clone()).is_err());
        c.measure_x(1);
        assert!(c.push(early).is_ok());
        let foreign = ModeOp::Gate {
            gate: ModeGate::Rotation {
                phi: Parameter(0).into(),
            },
            mode: 0,
        };
        assert!(c.push(foreign).is_err());
    }

    #[test]
    fn bind_replaces_parameters_only() {
        let mut c = ModeCircuit::new(2);
        let theta = c.parameter();
        c.beam_splitter(2.0 * theta, 0.0, 0, 1);
        let m = c.measure_photons(0);
        c.displace(m * 0.5, theta, 1);
        assert!(c.bind(&[]).is_err());
        assert!(c.bind(&[f64::NAN]).is_err());
        assert!(c.bind(&[f64::INFINITY]).is_err());
        let nan_mat = vec![C64::new(f64::NAN, 0.0); 4];
        assert!(c.clone().passive(&[0, 1], nan_mat).is_err());
        let bound = c.bind(&[0.25]).unwrap();
        assert_eq!(bound.n_parameters(), 0);
        assert_eq!(
            bound.ops()[0].args(),
            vec![Arg::Value(0.5), Arg::Value(0.0)]
        );
        assert_eq!(bound.ops()[2].args(), vec![m * 0.5, Arg::Value(0.25)]);
    }

    #[test]
    fn classification() {
        let mut c = ModeCircuit::new(3);
        c.beam_splitter(0.3, 0.1, 0, 1).rotation(0.2, 2);
        assert!(c.is_passive() && c.is_gaussian() && c.conserves_photons());
        c.set_input(0, ModeState::Fock(2)).unwrap();
        assert!(!c.is_gaussian() && c.is_passive());
        c.cross_kerr(0.1, 1, 2);
        assert!(!c.is_passive() && c.conserves_photons());
        let mut g = ModeCircuit::new(2);
        g.squeeze(0.5, 0.0, 0).cx(1.0, 0, 1);
        g.measure_x(1);
        assert!(g.is_gaussian() && !g.conserves_photons() && !g.is_unitary());
        g.measure_photons(0);
        assert!(!g.is_gaussian());
        let (re, im) = g.measure_heterodyne(1);
        assert_eq!((re.index(), im.index(), g.n_results()), (2, 3, 4));
    }
}
