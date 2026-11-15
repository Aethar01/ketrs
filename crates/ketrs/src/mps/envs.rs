//! Where a sweep keeps its environments and applies effective Hamiltonians:
//! on the CPU, or on a GPU when the state has one (see
//! [`Mps::set_gpu`](super::Mps::set_gpu)).

use super::Mps;
use super::blocks::{self, BlockEnv, BlockOp, Side};
use super::dmrg::{Env, update_left, update_right};
use super::effective::Effective;
#[cfg(feature = "gpu")]
use super::gpu::{GpuOperator, GpuSweep};
use super::mpo::Mpo;
use crate::error::Error;
use crate::statevector::eigen::{EigenOptions, Ritz, lanczos_lowest_from};
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
    /// Block-sparse environments of a symmetric MPS.
    Blocks {
        left: Vec<Option<BlockEnv>>,
        right: Vec<Option<BlockEnv>>,
    },
    #[cfg(feature = "gpu")]
    Gpu(Box<GpuSweep>),
}

impl Envs {
    /// Only the boundary environments, on `mps`'s GPU if it has one.
    pub(super) fn new(mps: &Mps, mpo: &Mpo) -> Result<Envs, Error> {
        let n = mps.n_qubits();
        if let Some(sym) = &mps.symmetry {
            debug_assert!(mpo.channels.is_some(), "a charge-graded MPO");
            let mut left: Vec<Option<BlockEnv>> = (0..n).map(|_| None).collect();
            let mut right: Vec<Option<BlockEnv>> = (0..n).map(|_| None).collect();
            left[0] = Some(BlockEnv::boundary(Side::Left, &sym.bonds[0], &sym.charges));
            right[n - 1] = Some(BlockEnv::boundary(Side::Right, &sym.bonds[n], &sym.charges));
            return Ok(Envs::Blocks { left, right });
        }
        #[cfg(feature = "gpu")]
        if let Some(gpu) = &mps.gpu {
            return Ok(Envs::Gpu(Box::new(GpuSweep::new(&gpu.0, mpo)?)));
        }
        debug_assert_eq!(mpo.sites.len(), n);
        let mut left: Vec<Option<Env>> = (0..n).map(|_| None).collect();
        let mut right: Vec<Option<Env>> = (0..n).map(|_| None).collect();
        left[0] = Some(Env::boundary());
        right[n - 1] = Some(Env::boundary());
        Ok(Envs::Cpu { left, right })
    }

    /// `left[i + 1]` from `left[i]` and site `i`'s tensor.
    pub(super) fn update_left(&mut self, i: usize, mps: &Mps, mpo: &Mpo) -> Result<(), Error> {
        let a = &mps.tensors[i];
        match self {
            Envs::Blocks { left, .. } => {
                let sym = mps.symmetry.as_ref().expect("a symmetric MPS");
                let channels = mpo.channels.as_ref().expect("a charge-graded MPO");
                left[i + 1] = Some(blocks::update_left(
                    left[i].as_ref().expect("left environment built"),
                    a,
                    &mpo.sites[i],
                    &sym.charges,
                    i,
                    &sym.bonds[i + 1],
                    &channels[i + 1],
                ));
                Ok(())
            }
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

    /// `right[i]` from `right[i + 1]` and site `i + 1`'s tensor.
    pub(super) fn update_right(&mut self, i: usize, mps: &Mps, mpo: &Mpo) -> Result<(), Error> {
        let a = &mps.tensors[i + 1];
        match self {
            Envs::Blocks { right, .. } => {
                let sym = mps.symmetry.as_ref().expect("a symmetric MPS");
                let channels = mpo.channels.as_ref().expect("a charge-graded MPO");
                right[i] = Some(blocks::update_right(
                    right[i + 1].as_ref().expect("right environment built"),
                    a,
                    &mpo.sites[i + 1],
                    &sym.charges,
                    i + 1,
                    &sym.bonds[i + 1],
                    &channels[i + 1],
                ));
                Ok(())
            }
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

    /// The effective Hamiltonian of sites `i` and `i + 1`.
    pub(super) fn two_site<'a>(&'a self, i: usize, mps: &Mps, mpo: &Mpo) -> Op<'a> {
        let (l, r) = (mps.tensors[i].left, mps.tensors[i + 1].right);
        match self {
            Envs::Blocks { left, right } => Op::Blocks(BlockOp::two_site(
                left[i].as_ref().expect("left environment built"),
                right[i + 1].as_ref().expect("right environment built"),
                &mpo.sites[i],
                &mpo.sites[i + 1],
                &mps.symmetry.as_ref().expect("a symmetric MPS").charges,
                i,
            )),
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
    pub(super) fn one_site<'a>(&'a self, i: usize, mps: &Mps, mpo: &'a Mpo) -> Op<'a> {
        let (l, r) = (mps.tensors[i].left, mps.tensors[i].right);
        match self {
            Envs::Blocks { left, right } => Op::Blocks(BlockOp::one_site(
                left[i].as_ref().expect("left environment built"),
                right[i].as_ref().expect("right environment built"),
                &mpo.sites[i],
                &mps.symmetry.as_ref().expect("a symmetric MPS").charges,
                i,
            )),
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

    /// Free `left[i]`, which the sweep has moved past, on a GPU (where
    /// environments of every bond would otherwise fill its memory). The
    /// boundary is kept; the CPU keeps everything.
    pub(super) fn release_left(&mut self, i: usize) {
        #[cfg(feature = "gpu")]
        if let Envs::Gpu(g) = self {
            g.release_left(i);
        }
        let _ = i;
    }

    /// Free `right[i]` (see [`release_left`](Self::release_left)).
    pub(super) fn release_right(&mut self, i: usize) {
        #[cfg(feature = "gpu")]
        if let Envs::Gpu(g) = self {
            g.release_right(i);
        }
        let _ = i;
    }

    /// `left[i]`, on the CPU.
    pub(super) fn left(&self, i: usize) -> Result<Cow<'_, Env>, Error> {
        match self {
            Envs::Blocks { .. } => unreachable!("dense environments of a symmetric MPS"),
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
            Envs::Blocks { .. } => unreachable!("dense environments of a symmetric MPS"),
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
            Envs::Cpu { .. } | Envs::Blocks { .. } => Ok(()),
            #[cfg(feature = "gpu")]
            Envs::Gpu(g) => g.take_error(),
        }
    }
}

/// An effective Hamiltonian from [`Envs`].
pub(super) enum Op<'a> {
    /// The marker keeps `'a` used in builds without the `gpu` feature.
    Cpu(Effective, PhantomData<&'a ()>),
    /// Acts on compact vectors: see [`pack`](Op::pack).
    Blocks(BlockOp<'a>),
    #[cfg(feature = "gpu")]
    Gpu(GpuOperator<'a>),
}

impl Op<'_> {
    /// The vector this operator acts on for a dense tensor, row-major
    /// `(l, P, r)`: the tensor itself, or for a block operator only its
    /// allowed blocks.
    pub(super) fn pack(&self, dense: Vec<C64>) -> Vec<C64> {
        match self {
            Op::Blocks(op) => op.layout().gather(&dense),
            _ => dense,
        }
    }

    /// The dense tensor of a vector from [`pack`](Self::pack).
    pub(super) fn unpack(&self, x: Vec<C64>) -> Vec<C64> {
        match self {
            Op::Blocks(op) => op.layout().scatter(&x),
            _ => x,
        }
    }

    /// `x ← exp(-iHt) x` by Krylov time-stepping, with the recurrence in
    /// GPU memory for a GPU operator.
    pub(super) fn evolve(&self, x: &mut [C64], t: f64) -> Result<(), Error> {
        match self {
            Op::Cpu(..) | Op::Blocks(_) => krylov_evolve(self, x, t, true),
            #[cfg(feature = "gpu")]
            Op::Gpu(op) => op.evolve(x, t),
        }
    }

    /// The lowest Ritz pair, by Lanczos iteration from the unit vector
    /// `v0`, with the recurrence in GPU memory for a GPU operator.
    pub(super) fn lowest(&self, v0: &[C64], options: &EigenOptions) -> Result<Ritz<C64>, Error> {
        match self {
            Op::Cpu(..) | Op::Blocks(_) => lanczos_lowest_from(self, v0, &[], options),
            #[cfg(feature = "gpu")]
            Op::Gpu(op) => op.lowest(v0, options),
        }
    }
}

impl LinearOperator<C64> for Op<'_> {
    fn apply(&self, x: &[C64], out: &mut [C64]) {
        match self {
            Op::Cpu(op, _) => op.apply(x, out),
            Op::Blocks(op) => op.apply(x, out),
            #[cfg(feature = "gpu")]
            Op::Gpu(op) => op.apply(x, out),
        }
    }
}
