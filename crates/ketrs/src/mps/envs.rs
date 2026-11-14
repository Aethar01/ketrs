//! Where a sweep keeps its environments and applies effective Hamiltonians.

use super::dmrg::{Env, TwoSite, update_left, update_right};
use super::mpo::Mpo;
use super::tdvp::OneSite;
use super::{Mps, Tensor};
use crate::error::Error;
use crate::statevector::krylov::LinearOperator;
use num_complex::Complex64 as C64;
use std::borrow::Cow;

/// The environments of a sweep: `left[i]` covers sites `< i` and
/// `right[i]` sites `> i`.
pub(super) enum Envs {
    Cpu {
        left: Vec<Option<Env>>,
        right: Vec<Option<Env>>,
    },
}

impl Envs {
    /// Only the boundary environments.
    pub(super) fn new(mps: &Mps, mpo: &Mpo) -> Result<Envs, Error> {
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
        }
    }

    /// The effective Hamiltonian of sites `i` and `i + 1`, between bonds of
    /// dimension `l` and `r`.
    pub(super) fn two_site<'a>(&'a self, i: usize, mpo: &Mpo, l: usize, r: usize) -> Op<'a> {
        match self {
            Envs::Cpu { left, right } => Op::Two(TwoSite::new(
                left[i].as_ref().expect("left environment built"),
                right[i + 1].as_ref().expect("right environment built"),
                &mpo.sites[i],
                &mpo.sites[i + 1],
                l,
                r,
            )),
        }
    }

    /// The effective Hamiltonian of site `i`.
    pub(super) fn one_site<'a>(&'a self, i: usize, mpo: &'a Mpo, l: usize, r: usize) -> Op<'a> {
        match self {
            Envs::Cpu { left, right } => Op::One(OneSite {
                left: left[i].as_ref().expect("left environment built"),
                right: right[i].as_ref().expect("right environment built"),
                site: &mpo.sites[i],
                l,
                r,
            }),
        }
    }

    /// `left[i]`, on the CPU.
    pub(super) fn left(&self, i: usize) -> Result<Cow<'_, Env>, Error> {
        match self {
            Envs::Cpu { left, .. } => Ok(Cow::Borrowed(
                left[i].as_ref().expect("left environment built"),
            )),
        }
    }

    /// `right[i]`, on the CPU.
    pub(super) fn right(&self, i: usize) -> Result<Cow<'_, Env>, Error> {
        match self {
            Envs::Cpu { right, .. } => Ok(Cow::Borrowed(
                right[i].as_ref().expect("right environment built"),
            )),
        }
    }

    /// Any error raised while applying an operator.
    pub(super) fn check(&self) -> Result<(), Error> {
        match self {
            Envs::Cpu { .. } => Ok(()),
        }
    }
}

/// An effective Hamiltonian from [`Envs`].
pub(super) enum Op<'a> {
    Two(TwoSite<'a>),
    One(OneSite<'a>),
}

impl LinearOperator<C64> for Op<'_> {
    fn apply(&self, x: &[C64], out: &mut [C64]) {
        match self {
            Op::Two(op) => op.apply(x, out),
            Op::One(op) => op.apply(x, out),
        }
    }
}
