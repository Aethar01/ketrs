//! Continuous-variable (qumode) circuits.
//!
//! The ideal CV gate model: modes, a universal gate set applied exactly,
//! perfect measurements, and noise only when asked for. Conventions follow
//! Strawberry Fields: ħ = 2, `x̂ = a + a†`, `p̂ = −i(a − a†)`, so the
//! vacuum has `⟨x̂²⟩ = ⟨p̂²⟩ = 1`. Gates are defined in [`ModeGate`] and
//! [`TwoModeGate`].
//!
//! Mode `0` is the least significant digit of a Fock-basis index, matching
//! the qubit side's ordering.
//!
//! ```
//! use ketrs::cv::{ModeCircuit, ModeState};
//!
//! // Hong–Ou–Mandel: two photons on a 50:50 beam splitter.
//! let mut c = ModeCircuit::new(2);
//! c.set_input(0, ModeState::Fock(1)).unwrap();
//! c.set_input(1, ModeState::Fock(1)).unwrap();
//! c.beam_splitter(std::f64::consts::FRAC_PI_4, 0.0, 0, 1);
//! assert!(c.is_passive() && c.conserves_photons());
//! ```

mod arg;
mod boson_sampling;
mod circuit;
mod fixed_n;
mod fock;
mod fock_ops;
mod gate;
mod gaussian;
mod gbs_sampler;
#[cfg(feature = "gpu")]
mod gpu;
mod hafnian;
mod hybrid;
mod mixture;
mod mps;
mod noise;
mod planner;
#[cfg(test)]
mod reference;
mod state;
mod window;

pub use arg::{Arg, Measured};
pub use boson_sampling::{boson_sample, permanent};
pub use circuit::{Measurement, ModeCircuit, ModeOp};
pub use fixed_n::FixedNFockState;
pub use fock::{FockDensityMatrix, FockState};
pub use gate::{ModeGate, TwoModeGate};
pub use gaussian::GaussianState;
pub use gbs_sampler::{sample_gbs_pnr, sample_gbs_threshold};
#[cfg(feature = "gpu")]
pub use gpu::{GpuFockState, permanent_gpu};
pub use hafnian::{hafnian, hafnian_repeated, loop_hafnian, loop_hafnian_repeated, torontonian};
pub use hybrid::{HybridCircuit, HybridOp, HybridState};
pub use mixture::{GaussianMixture, MixtureOptions};
pub use mps::{BosonicMps, BosonicMpsOptions};
pub use noise::CvChannel;
pub use planner::{CvBackend, CvPlan, PlanOptions, plan_circuit};
pub use state::ModeState;
pub use window::WindowSampler;
