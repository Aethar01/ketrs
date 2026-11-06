//! Exact time evolution `exp(-iHt)|ψ⟩` by Krylov (Lanczos) time-stepping.
//!
//! Each step builds a Lanczos basis `V` of up to [`KRYLOV_DIM`] vectors from
//! ψ, so that `H ≈ V T V†` with `T` tridiagonal, and takes
//! `exp(-iHΔt)ψ ≈ ‖ψ‖ V exp(-iTΔt) e₁`. The basis doesn't depend on Δt, so
//! the step is chosen afterwards, at no extra cost, as the largest that
//! meets the a-posteriori error estimate `‖ψ‖ β_m |[exp(-iTΔt)e₁]_m|`
//! (Saad 1992). Small states keep the basis; large ones rebuild it by
//! replaying the deterministic recurrence, trading extra `H|ψ⟩` products
//! for memory.

use super::DenseState;
use super::hamiltonian::{CompiledPauliSum, Imaginary};
use super::krylov::{LinearOperator, axpy, breakdown, norm, recurrence, scale};
use crate::error::Error;
use crate::observable::PauliSum;
use crate::scalar::ComplexScalar;
use num_complex::Complex64 as C64;
use rayon::prelude::*;
use std::ops::ControlFlow;

/// Lanczos vectors per step.
const KRYLOV_DIM: usize = 40;
/// Allowed error per step, relative to `‖ψ‖`, in double precision; single
/// precision allows a few times its epsilon instead.
const STEP_TOLERANCE: f64 = 1e-13;
/// Keep the Krylov basis when it fits in this many bytes; otherwise replay.
const BASIS_MEMORY: usize = 2 << 30;

impl<T: ComplexScalar> DenseState<T> {
    /// Evolve by `exp(-iHt)`, to about 1e-13 per step (1e-6 in single
    /// precision), without forming the
    /// matrix exponential.
    ///
    /// Uses Lanczos time-stepping with adaptive steps (see the module
    /// documentation), so the cost grows roughly with `t` times the spectral
    /// width of `H`. Identity terms only contribute a phase and are applied
    /// separately.
    pub fn evolve(&mut self, hamiltonian: &PauliSum, time: f64) -> Result<(), Error> {
        let store = (KRYLOV_DIM + 3) * self.amps.len() * size_of::<T>() <= BASIS_MEMORY;
        self.evolve_krylov(hamiltonian, time, store)
    }

    /// [`evolve`](Self::evolve), either keeping the Krylov basis or
    /// rebuilding it by replay.
    fn evolve_krylov(
        &mut self,
        hamiltonian: &PauliSum,
        time: f64,
        store: bool,
    ) -> Result<(), Error> {
        let mut shift = 0.0;
        let mut rest = PauliSum::new();
        for (c, s) in hamiltonian.simplify(0.0).terms() {
            if s.weight() == 0 {
                shift += c;
            } else {
                rest.add(*c, s.clone());
            }
        }
        let op = CompiledPauliSum::<T>::new(&rest, self.n_qubits, Imaginary::Reject, true)?;
        if !rest.terms().is_empty() {
            krylov_evolve(&op, &mut self.amps, time, store)?;
        }
        apply_phase(&mut self.amps, shift * time);
        Ok(())
    }
}

/// `amps ← exp(-iAt) amps` for a Hermitian operator `A`, by Lanczos
/// time-stepping; `store` keeps each Krylov basis instead of replaying it.
pub(crate) fn krylov_evolve<T: ComplexScalar>(
    op: &impl LinearOperator<T>,
    amps: &mut [T],
    time: f64,
    store: bool,
) -> Result<(), Error> {
    let tolerance = STEP_TOLERANCE.max(10.0 * T::EPSILON);
    let complex = |c: C64| T::from_complex(c).expect("complex scalar");
    let mut remaining = time;
    while remaining != 0.0 {
        let beta0 = norm(amps);
        if beta0 == 0.0 {
            break;
        }
        let inv = T::from_real(1.0 / beta0);
        let mut v0 = amps.to_vec();
        scale(&mut v0, inv);

        let (mut alphas, mut betas, mut basis) = (Vec::new(), Vec::new(), Vec::new());
        let mut closed = false;
        recurrence(op, &v0, &[], KRYLOV_DIM, |_, v, alpha, beta| {
            if store {
                basis.push(v.to_vec());
            }
            alphas.push(alpha);
            betas.push(beta);
            if beta <= breakdown(&alphas) {
                closed = true;
                return ControlFlow::Break(());
            }
            ControlFlow::Continue(())
        });
        let m = alphas.len();
        // β coupling the basis to the next Lanczos vector; zero when the
        // Krylov space is invariant and the step is exact.
        let residual = if closed { 0.0 } else { betas[m - 1] };

        let mut dt = remaining;
        let y = loop {
            let y = tridiagonal_exp(&alphas, &betas[..m - 1], dt);
            if residual * y[m - 1].norm() <= tolerance {
                break y;
            }
            dt /= 2.0;
            if dt.abs() <= f64::EPSILON * time.abs() {
                return Err(Error::NotConverged(
                    "Krylov evolution step shrank to zero".into(),
                ));
            }
        };

        // ψ ← ‖ψ‖ Σₖ yₖ vₖ.
        scale(amps, T::ZERO);
        let accumulate = |amps: &mut [T], k: usize, v: &[T]| {
            let c = complex(y[k] * beta0);
            axpy(amps, c, v);
        };
        if store {
            for (k, v) in basis.iter().enumerate() {
                accumulate(amps, k, v);
            }
        } else {
            recurrence(op, &v0, &[], m, |k, v, _, _| {
                accumulate(amps, k, v);
                if k + 1 == m {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            });
        }
        remaining = if dt == remaining { 0.0 } else { remaining - dt };
    }
    Ok(())
}

/// `amps ← e^{-iφ} amps`.
pub(crate) fn apply_phase<T: ComplexScalar>(amps: &mut [T], phi: f64) {
    if phi != 0.0 {
        let phase = T::from_complex(C64::cis(-phi)).expect("complex scalar");
        amps.par_iter_mut().for_each(|a| *a *= phase);
    }
}

/// `exp(-iTt) e₁` for the symmetric tridiagonal `T` with diagonal `a` and
/// off-diagonal `b`, by a Taylor series in substeps of norm at most 1, so
/// the terms shrink from the start and nothing cancels.
fn tridiagonal_exp(a: &[f64], b: &[f64], t: f64) -> Vec<C64> {
    let m = a.len();
    let off = |i: usize| if i < b.len() { b[i] } else { 0.0 };
    let bound = (0..m)
        .map(|i| a[i].abs() + off(i).abs() + if i > 0 { off(i - 1).abs() } else { 0.0 })
        .fold(0.0, f64::max);
    let substeps = ((bound * t.abs()).ceil() as usize).max(1);
    let h = t / substeps as f64;

    let mut y = vec![C64::new(0.0, 0.0); m];
    y[0] = C64::new(1.0, 0.0);
    let mut term = vec![C64::new(0.0, 0.0); m];
    let mut next = term.clone();
    for _ in 0..substeps {
        term.copy_from_slice(&y);
        for k in 1..60 {
            // next = (−ih/k) T term.
            let f = C64::new(0.0, -h / k as f64);
            for i in 0..m {
                let mut v = a[i] * term[i];
                if i > 0 {
                    v += off(i - 1) * term[i - 1];
                }
                if i + 1 < m {
                    v += off(i) * term[i + 1];
                }
                next[i] = f * v;
            }
            std::mem::swap(&mut term, &mut next);
            let size: f64 = term.iter().map(|x| x.norm_sqr()).sum();
            y.iter_mut().zip(&term).for_each(|(y, t)| *y += t);
            if size.sqrt() <= 1e-18 {
                break;
            }
        }
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Circuit, Statevector};

    #[test]
    fn replaying_the_basis_matches_storing_it() {
        let n = 6;
        let mut h =
            PauliSum::from_list([("ZZIIII", 1.0), ("IIXXII", 0.7), ("YIIIYI", -0.4)]).unwrap();
        h.add(0.3, crate::PauliString::from_label("IIIIIX").unwrap());
        let mut circ = Circuit::new(n);
        for q in 0..n {
            circ.ry(0.3 + q as f64, q);
        }
        let start = Statevector::from_circuit(&circ);
        let (mut stored, mut replayed) = (start.clone(), start);
        // Long enough to need several Krylov steps.
        stored.evolve_krylov(&h, 25.0, true).unwrap();
        replayed.evolve_krylov(&h, 25.0, false).unwrap();
        let diff = stored
            .amplitudes()
            .iter()
            .zip(replayed.amplitudes())
            .map(|(a, b)| (a - b).norm())
            .fold(0.0, f64::max);
        assert!(diff < 1e-12, "{diff}");
        assert!((stored.norm_sqr() - 1.0).abs() < 1e-10);
    }
}
