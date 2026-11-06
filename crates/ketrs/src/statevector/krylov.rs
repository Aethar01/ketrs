//! Lanczos recurrence and vector helpers shared by the eigensolver and
//! Krylov time evolution.
//!
//! Reductions are summed in a fixed order (see [`chunked_sum`]), so rerunning
//! the recurrence from the same start vector reproduces it exactly. Callers
//! use that to rebuild a Krylov basis instead of storing it.

use super::hamiltonian::chunked_sum;
use super::kernels::PARALLEL_THRESHOLD;
use crate::scalar::Scalar;
use rayon::prelude::*;
use std::ops::ControlFlow;

/// A Hermitian operator on vectors of amplitudes, such as a Pauli sum on the
/// full space or restricted to a symmetry sector.
pub(crate) trait LinearOperator<T>: Sync {
    /// `out = A x`.
    fn apply(&self, x: &[T], out: &mut [T]);
}

/// Run the Lanczos recurrence from `v0`, calling `visit(k, vₖ, αₖ, βₖ)` at
/// each step. Stops when `visit` breaks, after `max_steps`, or when the
/// Krylov space closes.
pub(crate) fn recurrence<T: Scalar>(
    op: &impl LinearOperator<T>,
    v0: &[T],
    locked: &[&[T]],
    max_steps: usize,
    mut visit: impl FnMut(usize, &[T], f64, f64) -> ControlFlow<()>,
) {
    let dim = v0.len();
    let mut v_prev = vec![T::ZERO; dim];
    let mut v = v0.to_vec();
    let mut w = vec![T::ZERO; dim];
    let mut beta_prev = 0.0;
    for k in 0..max_steps {
        op.apply(&v, &mut w);
        orthogonalise(&mut w, locked);
        let alpha = dot(&v, &w).to_complex().re;
        let (a, b) = (T::from_real(alpha), T::from_real(beta_prev));
        let step = |((w, &v), &vp): ((&mut T, &T), &T)| *w = *w - a * v - b * vp;
        if dim >= PARALLEL_THRESHOLD {
            w.par_iter_mut().zip(&v).zip(&v_prev).for_each(step);
        } else {
            w.iter_mut().zip(&v).zip(&v_prev).for_each(step);
        }
        orthogonalise(&mut w, locked);
        let beta = norm(&w);
        if visit(k, &v, alpha, beta).is_break() || beta == 0.0 {
            return;
        }
        // Shift: v_prev ← v, v ← w/β.
        std::mem::swap(&mut v_prev, &mut v);
        std::mem::swap(&mut v, &mut w);
        scale(&mut v, T::from_real(1.0 / beta));
        beta_prev = beta;
    }
}

pub(crate) fn dot<T: Scalar>(a: &[T], b: &[T]) -> T {
    chunked_sum(a.len(), |r| r.map(|j| a[j].conj() * b[j]).sum::<T>())
}

pub(crate) fn norm<T: Scalar>(a: &[T]) -> f64 {
    chunked_sum(a.len(), |r| r.map(|j| a[j].norm_sqr()).sum::<f64>()).sqrt()
}

pub(crate) fn normalise<T: Scalar>(a: &mut [T]) {
    let inv = T::from_real(1.0 / norm(a));
    scale(a, inv);
}

/// Remove the components along each (orthonormal) `locked` vector.
pub(crate) fn orthogonalise<T: Scalar>(w: &mut [T], locked: &[&[T]]) {
    for l in locked {
        let c = dot(l, w);
        axpy(w, -c, l);
    }
}

/// A β this small relative to the matrix scale means the Krylov space has
/// closed: the Ritz pair is exact.
pub(crate) fn breakdown(alphas: &[f64]) -> f64 {
    let scale = alphas.iter().fold(1.0f64, |m, a| m.max(a.abs()));
    1e-13 * scale
}

/// `a ← s·a`; parallel only for long vectors, where it pays.
pub(crate) fn scale<T: Scalar>(a: &mut [T], s: T) {
    if a.len() >= PARALLEL_THRESHOLD {
        a.par_iter_mut().for_each(|x| *x *= s);
    } else {
        a.iter_mut().for_each(|x| *x *= s);
    }
}

/// `y ← y + c·x`; parallel only for long vectors.
pub(crate) fn axpy<T: Scalar>(y: &mut [T], c: T, x: &[T]) {
    if y.len() >= PARALLEL_THRESHOLD {
        y.par_iter_mut().zip(x).for_each(|(y, &x)| *y += c * x);
    } else {
        y.iter_mut().zip(x).for_each(|(y, &x)| *y += c * x);
    }
}
