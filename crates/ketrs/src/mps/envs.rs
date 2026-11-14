//! Where a sweep keeps its environments and applies effective Hamiltonians:
//! on the CPU, or on a GPU when the state has one (see
//! [`Mps::set_gpu`](super::Mps::set_gpu)).

use super::dmrg::{Env, update_left, update_right};
use super::effective::Effective;
#[cfg(feature = "gpu")]
use super::gpu::{GpuOperator, GpuSweep};
use super::mpo::Mpo;
use super::{Mps, Tensor};
use crate::error::Error;
use crate::statevector::evolve::krylov_evolve;
use crate::statevector::krylov::LinearOperator;
use num_complex::Complex64 as C64;
use std::borrow::Cow;
use std::marker::PhantomData;

/// The environments of a sweep: `left[i]` covers sites `< i` and
/// `right[i]` sites `> i`.
pub(super) enum Envs {
    Cpu {
        left: Vec<Option<Env>>,
        right: Vec<Option<Env>>,
    },
    #[cfg(feature = "gpu")]
    Gpu(Box<GpuSweep>),
}

impl Envs {
    /// Only the boundary environments, on `mps`'s GPU if it has one.
    pub(super) fn new(mps: &Mps, mpo: &Mpo) -> Result<Envs, Error> {
        #[cfg(feature = "gpu")]
        if let Some(gpu) = &mps.gpu {
            return Ok(Envs::Gpu(Box::new(GpuSweep::new(&gpu.0, mpo)?)));
        }
        let n = mps.n_qubits();
        debug_assert_eq!(mpo.sites.len(), n);
        let mut left: Vec<Option<Env>> = (0..n).map(|_| None).collect();
        let mut right: Vec<Option<Env>> = (0..n).map(|_| None).collect();
        left[0] = Some(Env::boundary());
        right[n - 1] = Some(Env::boundary());
        Ok(Envs::Cpu { left, right })
    }

    /// `left[i + 1]` from `left[i]` and site `i`'s tensor `a`.
    pub(super) fn update_left(&mut self, i: usize, a: &Tensor, mpo: &Mpo) -> Result<(), Error> {
        match self {
            Envs::Cpu { left, .. } => {
                left[i + 1] = Some(update_left(
                    left[i].as_ref().expect("left environment built"),
                    a,
                    &mpo.sites[i],
                ));
                Ok(())
            }
            #[cfg(feature = "gpu")]
            Envs::Gpu(g) => g.update_left(i, a, mpo),
        }
    }

    /// `right[i]` from `right[i + 1]` and site `i + 1`'s tensor `a`.
    pub(super) fn update_right(&mut self, i: usize, a: &Tensor, mpo: &Mpo) -> Result<(), Error> {
        match self {
            Envs::Cpu { right, .. } => {
                right[i] = Some(update_right(
                    right[i + 1].as_ref().expect("right environment built"),
                    a,
                    &mpo.sites[i + 1],
                ));
                Ok(())
            }
            #[cfg(feature = "gpu")]
            Envs::Gpu(g) => g.update_right(i, a, mpo),
        }
    }

    /// The effective Hamiltonian of sites `i` and `i + 1`, between bonds of
    /// dimension `l` and `r`.
    pub(super) fn two_site<'a>(&'a self, i: usize, mpo: &Mpo, l: usize, r: usize) -> Op<'a> {
        match self {
            Envs::Cpu { left, right } => Op::Cpu(
                Effective::two_site(
                    left[i].as_ref().expect("left environment built"),
                    right[i + 1].as_ref().expect("right environment built"),
                    &mpo.sites[i],
                    &mpo.sites[i + 1],
                    l,
                    r,
                ),
                PhantomData,
            ),
            #[cfg(feature = "gpu")]
            Envs::Gpu(g) => Op::Gpu(g.two_site(i, l, r)),
        }
    }

    /// The effective Hamiltonian of site `i`.
    pub(super) fn one_site<'a>(&'a self, i: usize, mpo: &'a Mpo, l: usize, r: usize) -> Op<'a> {
        match self {
            Envs::Cpu { left, right } => Op::Cpu(
                Effective::one_site(
                    left[i].as_ref().expect("left environment built"),
                    right[i].as_ref().expect("right environment built"),
                    &mpo.sites[i],
                    l,
                    r,
                ),
                PhantomData,
            ),
            #[cfg(feature = "gpu")]
            Envs::Gpu(g) => Op::Gpu(g.one_site(i, l, r)),
        }
    }

    /// `left[i]`, on the CPU.
    pub(super) fn left(&self, i: usize) -> Result<Cow<'_, Env>, Error> {
        match self {
            Envs::Cpu { left, .. } => Ok(Cow::Borrowed(
                left[i].as_ref().expect("left environment built"),
            )),
            #[cfg(feature = "gpu")]
            Envs::Gpu(g) => Ok(Cow::Owned(g.download_left(i)?)),
        }
    }

    /// `right[i]`, on the CPU.
    pub(super) fn right(&self, i: usize) -> Result<Cow<'_, Env>, Error> {
        match self {
            Envs::Cpu { right, .. } => Ok(Cow::Borrowed(
                right[i].as_ref().expect("right environment built"),
            )),
            #[cfg(feature = "gpu")]
            Envs::Gpu(g) => Ok(Cow::Owned(g.download_right(i)?)),
        }
    }

    /// Any error raised while applying an operator (only GPU applications
    /// can fail).
    pub(super) fn check(&self) -> Result<(), Error> {
        match self {
            Envs::Cpu { .. } => Ok(()),
            #[cfg(feature = "gpu")]
            Envs::Gpu(g) => g.take_error(),
        }
    }
}

/// An effective Hamiltonian from [`Envs`].
pub(super) enum Op<'a> {
    /// The marker keeps `'a` used in builds without the `gpu` feature.
    Cpu(Effective, PhantomData<&'a ()>),
    #[cfg(feature = "gpu")]
    Gpu(GpuOperator<'a>),
}

impl Op<'_> {
    /// `x ← exp(-iHt) x` by Krylov time-stepping, with the recurrence in
    /// GPU memory for a GPU operator.
    pub(super) fn evolve(&self, x: &mut [C64], t: f64) -> Result<(), Error> {
        match self {
            Op::Cpu(..) => krylov_evolve(self, x, t, true),
            #[cfg(feature = "gpu")]
            Op::Gpu(op) => op.evolve(x, t),
        }
    }
}

impl LinearOperator<C64> for Op<'_> {
    fn apply(&self, x: &[C64], out: &mut [C64]) {
        match self {
            Op::Cpu(op, _) => op.apply(x, out),
            #[cfg(feature = "gpu")]
            Op::Gpu(op) => op.apply(x, out),
        }
    }
}
