//! Exact time evolution `exp(-iHt)|ψ⟩` by Krylov (Lanczos) time-stepping.
//!
//! Each step builds a Lanczos basis `V` of up to [`KRYLOV_MAX`] vectors from
//! ψ, so that `H ≈ V T V†` with `T` tridiagonal, and takes
//! `exp(-iHΔt)ψ ≈ ‖ψ‖ V exp(-iTΔt) e₁`, within the a-posteriori error
//! estimate `‖ψ‖ β_m |[exp(-iTΔt)e₁]_m|` (Saad 1992). The basis stops
//! growing as soon as it covers all the remaining time (checked every
//! [`KRYLOV_CHECK`] vectors); a full basis that doesn't takes the longest
//! step it covers, and the next basis starts from there. Small states keep the basis; large ones rebuild it by
//! replaying the deterministic recurrence, trading extra `H|ψ⟩` products
//! for memory.

use super::DenseState;
use super::hamiltonian::{CompiledPauliSum, Imaginary};
use super::krylov::{LinearOperator, axpy, breakdown, norm, recurrence, scale};
use crate::error::Error;
use crate::observable::PauliSum;
use crate::scalar::ComplexScalar;
use faer::diag::Diag;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::evd::{ComputeEigenvectors, self_adjoint_evd, self_adjoint_evd_scratch};
use faer::{Mat, Par};
use num_complex::Complex64 as C64;
use rayon::prelude::*;
use std::ops::ControlFlow;

/// Most Lanczos vectors per basis of [`krylov_evolve`].
pub(crate) const KRYLOV_MAX: usize = 64;
/// [`krylov_evolve`] checks whether its basis covers the remaining time
/// every this many vectors, as each check is a small matrix exponential
/// (GPU recurrences run this many steps per submission).
pub(crate) const KRYLOV_CHECK: usize = 8;
/// Allowed error per step, relative to `‖ψ‖`, in double precision; single
/// precision allows a few times its epsilon instead.
pub(crate) const STEP_TOLERANCE: f64 = 1e-13;
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
        let store = (KRYLOV_MAX + 3) * self.amps.len() * size_of::<T>() <= BASIS_MEMORY;
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
        let mut covered = None;
        recurrence(op, &v0, &[], KRYLOV_MAX, |_, v, alpha, beta| {
            if store {
                basis.push(v.to_vec());
            }
            alphas.push(alpha);
            betas.push(beta);
            let m = alphas.len();
            // A closed Krylov space is invariant: the step is exact.
            let closed = beta <= breakdown(&alphas);
            if closed || m.is_multiple_of(KRYLOV_CHECK) {
                let residual = if closed { 0.0 } else { beta };
                covered = krylov_covers(&alphas, &betas[..m - 1], residual, remaining, tolerance);
                if covered.is_some() {
                    return ControlFlow::Break(());
                }
            }
            ControlFlow::Continue(())
        });
        let m = alphas.len();
        let (dt, y) = match covered {
            Some(y) => (remaining, y),
            None => krylov_partial_step(
                &alphas,
                &betas[..m - 1],
                betas[m - 1],
                remaining,
                time,
                tolerance,
            )?,
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

/// `exp(-iTt) e₁` if a Lanczos basis with tridiagonal `T` (diagonal `a`,
/// off-diagonal `b`) evolves by `t` within `tolerance`. `residual` is the β
/// coupling the basis to the next vector, zero for a closed space.
pub(crate) fn krylov_covers(
    a: &[f64],
    b: &[f64],
    residual: f64,
    t: f64,
    tolerance: f64,
) -> Option<Vec<C64>> {
    // The Taylor series below is accurate but costs about a millisecond for
    // a wide spectrum, while most checks fail by orders of magnitude. The
    // eigendecomposition gives the last component to about 1e-14
    // absolutely (too coarse to accept with), enough to reject those.
    if residual != 0.0
        && let Some(last) = last_component(a, b, t)
        && residual * (last.norm() - 1e-12) > tolerance
    {
        return None;
    }
    let y = tridiagonal_exp(a, b, t);
    (residual * y[a.len() - 1].norm() <= tolerance).then_some(y)
}

/// The last component of `exp(-iTt) e₁` from the eigendecomposition
/// `T = U Λ Uᵀ`: `Σ_j U[m−1, j] e^{−iλ_j t} U[0, j]`.
fn last_component(a: &[f64], b: &[f64], t: f64) -> Option<C64> {
    let m = a.len();
    let tri = Mat::<f64>::from_fn(m, m, |i, j| match i.abs_diff(j) {
        0 => a[i],
        1 => b[i.min(j)],
        _ => 0.0,
    });
    let mut values = Diag::<f64>::zeros(m);
    let mut vectors = Mat::<f64>::zeros(m, m);
    let params = Default::default();
    let req = self_adjoint_evd_scratch::<f64>(m, ComputeEigenvectors::Yes, Par::Seq, params);
    let mut buffer = MemBuffer::new(req);
    self_adjoint_evd(
        tri.as_ref(),
        values.as_mut(),
        Some(vectors.as_mut()),
        Par::Seq,
        MemStack::new(&mut buffer),
        params,
    )
    .ok()?;
    let values = values.column_vector();
    Some(
        (0..m)
            .map(|j| C64::cis(-values[j] * t) * (vectors[(m - 1, j)] * vectors[(0, j)]))
            .sum(),
    )
}

/// For a full basis that doesn't cover `remaining` (see [`krylov_covers`]):
/// about the longest step it does cover, by halving and then three
/// bisections, with its `exp(-iTΔt) e₁`.
pub(crate) fn krylov_partial_step(
    a: &[f64],
    b: &[f64],
    residual: f64,
    remaining: f64,
    time: f64,
    tolerance: f64,
) -> Result<(f64, Vec<C64>), Error> {
    let mut long = remaining;
    let (mut dt, mut y) = loop {
        let dt = long / 2.0;
        if dt.abs() <= f64::EPSILON * time.abs() {
            return Err(Error::NotConverged(
                "Krylov evolution step shrank to zero".into(),
            ));
        }
        if let Some(y) = krylov_covers(a, b, residual, dt, tolerance) {
            break (dt, y);
        }
        long = dt;
    };
    for _ in 0..3 {
        let mid = 0.5 * (dt + long);
        match krylov_covers(a, b, residual, mid, tolerance) {
            Some(ym) => (dt, y) = (mid, ym),
            None => long = mid,
        }
    }
    Ok((dt, y))
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
pub(crate) fn tridiagonal_exp(a: &[f64], b: &[f64], t: f64) -> Vec<C64> {
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
    fn eigendecomposition_screen_agrees_with_the_taylor_series() {
        // Lanczos-like tridiagonals with a spectrum of width about 2000,
        // like a lattice gauge theory's penalty term, up to 64 vectors.
        let mut x = 7u64;
        let mut next = move || {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (x >> 11) as f64 / (1u64 << 53) as f64
        };
        for m in [2, 8, 24, 40, 64] {
            let a: Vec<f64> = (0..m).map(|_| 2000.0 * (next() - 0.5)).collect();
            let b: Vec<f64> = (0..m - 1).map(|_| 300.0 * next() + 1.0).collect();
            for t in [1e-3, 0.05, 0.3] {
                let taylor = tridiagonal_exp(&a, &b, t)[m - 1];
                let eig = last_component(&a, &b, t).unwrap();
                assert!((taylor - eig).norm() < 1e-13, "m {m} t {t}: {taylor} {eig}");
            }
        }
    }

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
