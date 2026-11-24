//! Lowest eigenstates of Pauli sums by Lanczos iteration.
//!
//! Matrix-free and memory-light: each eigenstate needs a handful of
//! statevectors, however many iterations it takes. The first pass builds the
//! Lanczos tridiagonal matrix; a second pass replays the identical recurrence
//! to assemble the eigenvector, so the Krylov basis is never stored. Every
//! reduction is summed in a fixed order, which keeps the replay exact.
//! Further eigenstates are found by deflation: the recurrence is kept
//! orthogonal to the states already found.

use super::DenseState;
use super::hamiltonian::{CompiledPauliSum, Imaginary, chunked_sum};
use super::krylov::{
    LinearOperator, axpy, breakdown, dot, norm, normalise, orthogonalise, recurrence,
};
use crate::error::Error;
use crate::observable::PauliSum;
use crate::scalar::Scalar;
use num_complex::Complex64 as C64;
use std::ops::ControlFlow;

/// Settings for [`DenseState::lowest_eigenstates`].
#[derive(Clone, Debug)]
pub struct EigenOptions {
    /// Stop when the estimated residual `‖Hx − Ex‖` is at most this.
    pub tolerance: f64,
    /// Most Lanczos iterations per eigenstate.
    pub max_iterations: usize,
    /// Seed for the random starting vectors.
    pub seed: u64,
}

impl Default for EigenOptions {
    fn default() -> Self {
        EigenOptions {
            tolerance: 1e-9,
            max_iterations: 1000,
            seed: 0,
        }
    }
}

/// An eigenpair found by `lowest_eigenstates`.
#[derive(Clone, Debug)]
pub struct Eigenstate<S> {
    /// The eigenvalue, as the Rayleigh quotient `⟨x|H|x⟩`.
    pub energy: f64,
    /// The normalised eigenvector.
    pub state: S,
    /// `‖Hx − Ex‖`, measured directly.
    pub residual: f64,
}

impl<T: Scalar> DenseState<T> {
    /// The `k` lowest eigenstates of `hamiltonian` on `n_qubits` qubits, in
    /// increasing energy, by Lanczos iteration with deflation. Degenerate
    /// eigenvalues are returned once per independent eigenvector.
    ///
    /// For a real statevector, `hamiltonian` must be real: no string may
    /// have an odd number of Y factors.
    pub fn lowest_eigenstates(
        hamiltonian: &PauliSum,
        n_qubits: usize,
        k: usize,
        options: &EigenOptions,
    ) -> Result<Vec<Eigenstate<Self>>, Error> {
        let dim = 1usize << n_qubits;
        let op = CompiledPauliSum::<T>::new(hamiltonian, n_qubits, Imaginary::Reject, true)?;
        Ok(lowest_eigenvectors(&op, dim, k, options)?
            .into_iter()
            .map(|(energy, amps, residual)| Eigenstate {
                energy,
                state: DenseState { n_qubits, amps },
                residual,
            })
            .collect())
    }
}

/// The `k` lowest eigenpairs `(energy, vector, residual)` of a Hermitian
/// operator on `dim`-dimensional vectors, in increasing energy.
pub(crate) fn lowest_eigenvectors<T: Scalar>(
    op: &impl LinearOperator<T>,
    dim: usize,
    k: usize,
    options: &EigenOptions,
) -> Result<Vec<(f64, Vec<T>, f64)>, Error> {
    if k > dim {
        return Err(Error::InvalidArgument(format!(
            "asked for {k} eigenstates of a {dim}-dimensional space"
        )));
    }
    let mut found: Vec<(f64, Vec<T>, f64)> = Vec::with_capacity(k);
    for index in 0..k {
        let locked: Vec<&[T]> = found.iter().map(|(_, x, _)| x.as_slice()).collect();
        let seed = options.seed.wrapping_add(index as u64);
        let x = lanczos_lowest(op, dim, &locked, options, seed)?;
        let mut hx = vec![T::ZERO; dim];
        op.apply(&x, &mut hx);
        let energy = dot(&x, &hx).to_complex().re;
        let e = T::from_real(energy);
        let residual = chunked_sum(dim, |r| {
            r.map(|j| (hx[j] - e * x[j]).norm_sqr()).sum::<f64>()
        })
        .sqrt();
        found.push((energy, x, residual));
    }
    found.sort_by(|a, b| a.0.total_cmp(&b.0));
    Ok(found)
}

/// The lowest eigenvector of `H` restricted to the complement of `locked`.
fn lanczos_lowest<T: Scalar>(
    op: &impl LinearOperator<T>,
    dim: usize,
    locked: &[&[T]],
    options: &EigenOptions,
    seed: u64,
) -> Result<Vec<T>, Error> {
    let v0 = start_vector(dim, locked, seed)?;
    let ritz = lanczos_lowest_from(op, &v0, locked, options)?;
    if !ritz.converged {
        return Err(Error::NotConverged(format!(
            "Lanczos did not converge in {} iterations",
            options.max_iterations
        )));
    }
    Ok(ritz.vector)
}

/// The lowest Ritz pair of a Lanczos solve.
pub(crate) struct Ritz<T> {
    /// The eigenvalue estimate, within about `residual² / gap` of the
    /// vector's energy.
    pub(crate) value: f64,
    /// The normalised Ritz vector.
    pub(crate) vector: Vec<T>,
    /// Whether the residual reached the tolerance. If not, this is the pair
    /// with the smallest residual estimate seen.
    pub(crate) converged: bool,
}

/// The Lanczos convergence test, one step at a time, keeping the best Ritz
/// pair seen in case none converges.
///
/// The residual of the lowest Ritz pair is estimated as `β |s_m|`, `s` its
/// eigenvector of the tridiagonal matrix. Rounding in `H x` keeps it above
/// about 1e-13 of the spectrum's scale (as in [`breakdown`]), so it counts as
/// converged there too, whatever the tolerance: with a spectrum near
/// −4000, as for lattice gauge theories with a strong Gauss penalty, a
/// tolerance of 1e-10 is within about 100 rounding errors and may never be
/// reached. Without reorthogonalisation the estimate only gets worse after
/// that, as copies of converged Ritz values appear.
pub(crate) struct RitzTracker {
    tolerance: f64,
    alphas: Vec<f64>,
    betas: Vec<f64>,
    /// `(estimate, value, s)` with the smallest estimate so far.
    best: Option<(f64, f64, Vec<f64>)>,
}

impl RitzTracker {
    pub(crate) fn new(tolerance: f64) -> Self {
        RitzTracker {
            tolerance,
            alphas: Vec::new(),
            betas: Vec::new(),
            best: None,
        }
    }

    /// Record a step's `α` and `β`; returns the lowest Ritz value and its
    /// coefficients `s` once converged.
    pub(crate) fn step(&mut self, alpha: f64, beta: f64) -> Option<(f64, Vec<f64>)> {
        self.alphas.push(alpha);
        let (value, s) = tridiagonal_lowest(&self.alphas, &self.betas);
        let estimate = beta * s[s.len() - 1].abs();
        let floor = breakdown(&self.alphas);
        if estimate <= self.tolerance.max(floor) || beta <= floor {
            return Some((value, s));
        }
        if self.best.as_ref().is_none_or(|b| estimate < b.0) {
            self.best = Some((estimate, value, s));
        }
        self.betas.push(beta);
        None
    }

    /// The pair with the smallest residual estimate, for a solve that
    /// didn't converge.
    pub(crate) fn best(self) -> Option<(f64, Vec<f64>)> {
        self.best.map(|(_, value, s)| (value, s))
    }
}

/// As [`lanczos_lowest`], from a given unit start vector orthogonal to
/// `locked`, returning the lowest Ritz pair (see [`Ritz`]): the best one
/// seen if none converges within `max_iterations`. A good guess (such as
/// the previous solution in DMRG) converges in far fewer iterations.
pub(crate) fn lanczos_lowest_from<T: Scalar>(
    op: &impl LinearOperator<T>,
    v0: &[T],
    locked: &[&[T]],
    options: &EigenOptions,
) -> Result<Ritz<T>, Error> {
    let dim = v0.len();

    // Pass 1: build the tridiagonal matrix until its lowest Ritz pair
    // converges.
    let mut tracker = RitzTracker::new(options.tolerance);
    let mut ritz: Option<(f64, Vec<f64>)> = None;
    recurrence(
        op,
        v0,
        locked,
        options.max_iterations,
        |_, _, alpha, beta| {
            ritz = tracker.step(alpha, beta);
            if ritz.is_some() {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        },
    );
    let converged = ritz.is_some();
    let (theta, s) = ritz
        .or_else(|| tracker.best())
        .ok_or_else(|| Error::InvalidArgument("Lanczos needs at least one iteration".into()))?;

    // Pass 2: replay the recurrence, accumulating x = Σₖ sₖ vₖ.
    let mut x = vec![T::ZERO; dim];
    recurrence(op, v0, locked, s.len(), |k, v, _, _| {
        let c = T::from_real(s[k]);
        axpy(&mut x, c, v);
        if k + 1 == s.len() {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    });
    orthogonalise(&mut x, locked);
    normalise(&mut x);
    Ok(Ritz {
        value: theta,
        vector: x,
        converged,
    })
}

/// A random unit vector orthogonal to `locked`.
fn start_vector<T: Scalar>(dim: usize, locked: &[&[T]], seed: u64) -> Result<Vec<T>, Error> {
    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut uniform = move || {
        // splitmix64, mapped to [-1, 1).
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    };
    let mut v: Vec<T> = (0..dim)
        .map(|_| {
            let (re, im) = (uniform(), uniform());
            T::from_complex(C64::new(re, im)).unwrap_or(T::from_real(re))
        })
        .collect();
    orthogonalise(&mut v, locked);
    if norm(&v) < 1e-8 {
        return Err(Error::InvalidArgument(
            "no space left for another eigenstate".into(),
        ));
    }
    normalise(&mut v);
    Ok(v)
}

/// The lowest eigenvalue and unit eigenvector of the symmetric tridiagonal
/// matrix with diagonal `a` and off-diagonal `b`: bisection on the Sturm
/// sequence, then inverse iteration.
pub(crate) fn tridiagonal_lowest(a: &[f64], b: &[f64]) -> (f64, Vec<f64>) {
    let m = a.len();
    let off = |i: usize| if i < b.len() { b[i].abs() } else { 0.0 };
    let radius = |i: usize| off(i) + if i > 0 { off(i - 1) } else { 0.0 };
    let mut lo = (0..m)
        .map(|i| a[i] - radius(i))
        .fold(f64::INFINITY, f64::min);
    let mut hi = (0..m)
        .map(|i| a[i] + radius(i))
        .fold(f64::NEG_INFINITY, f64::max);
    let scale = lo.abs().max(hi.abs()).max(f64::MIN_POSITIVE);
    let tiny = f64::EPSILON * scale;

    // Number of eigenvalues below x, from the signs of the LDLᵀ pivots.
    let count_below = |x: f64| {
        let mut count = 0;
        let mut q = 1.0;
        for i in 0..m {
            q = a[i] - x - if i > 0 { b[i - 1] * b[i - 1] / q } else { 0.0 };
            if q.abs() < tiny {
                q = -tiny;
            }
            if q < 0.0 {
                count += 1;
            }
        }
        count
    };
    while hi - lo > 2.0 * tiny {
        let mid = 0.5 * (lo + hi);
        if mid <= lo || mid >= hi {
            break;
        }
        if count_below(mid) >= 1 {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let lambda = 0.5 * (lo + hi);

    // Inverse iteration: solve (T − λ)y = y_prev with partial-pivoted
    // tridiagonal Gaussian elimination (preventing pivot blow-up and overflow
    // in unpivoted Thomas elimination on large/near-degenerate Krylov spaces)
    // and overflow-safe Euclidean normalisation.
    let mut y = vec![1.0 / (m as f64).sqrt(); m];
    for _ in 0..3 {
        y = solve_tridiag_pivoted(a, b, lambda, &y, tiny);
        let max_abs = y.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        if max_abs == 0.0 || !max_abs.is_finite() {
            return tridiagonal_lowest_evd(a, b);
        }
        let norm = max_abs
            * y.iter()
                .map(|v| {
                    let s = v / max_abs;
                    s * s
                })
                .sum::<f64>()
                .sqrt();
        if norm == 0.0 || !norm.is_finite() {
            return tridiagonal_lowest_evd(a, b);
        }
        y.iter_mut().for_each(|v| *v /= norm);
    }
    (lambda, y)
}

/// Solve `(T - lambda I) x = rhs` for symmetric tridiagonal `T(a, b)` using
/// Gaussian elimination with row partial pivoting (LAPACK `dgttrf`/`dgttrs`).
fn solve_tridiag_pivoted(a: &[f64], b: &[f64], lambda: f64, rhs: &[f64], tiny: f64) -> Vec<f64> {
    let m = a.len();
    let mut diag: Vec<f64> = a.iter().map(|&ai| ai - lambda).collect();
    let mut upper: Vec<f64> = b.to_vec();
    let mut upper2 = vec![0.0; m.saturating_sub(2)];
    let mut x = rhs.to_vec();

    for i in 0..m - 1 {
        let sub = b[i];
        if diag[i].abs() >= sub.abs() {
            let piv = if diag[i].abs() < tiny {
                if diag[i] < 0.0 { -tiny } else { tiny }
            } else {
                diag[i]
            };
            diag[i] = piv;
            let mult = sub / piv;
            diag[i + 1] -= mult * upper[i];
            x[i + 1] -= mult * x[i];
        } else {
            // Swap rows i and i + 1.
            let piv = if sub.abs() < tiny {
                if sub < 0.0 { -tiny } else { tiny }
            } else {
                sub
            };
            let mult = diag[i] / piv;
            let u_i = upper[i];
            let d_next = diag[i + 1];
            let u_next = if i + 1 < b.len() { b[i + 1] } else { 0.0 };
            diag[i] = piv;
            upper[i] = d_next;
            if i + 2 < m {
                upper2[i] = u_next;
            }
            diag[i + 1] = u_i - mult * d_next;
            if i + 1 < upper.len() {
                upper[i + 1] = -mult * u_next;
            }
            x.swap(i, i + 1);
            x[i + 1] -= mult * x[i];
        }
    }
    if diag[m - 1].abs() < tiny {
        diag[m - 1] = if diag[m - 1] < 0.0 { -tiny } else { tiny };
    }
    // Back-substitution with overflow protection.
    for i in (0..m).rev() {
        let mut s = x[i];
        if i + 1 < m {
            s -= upper[i] * x[i + 1];
        }
        if i + 2 < m {
            s -= upper2[i] * x[i + 2];
        }
        x[i] = s / diag[i];
        if x[i].abs() > 1e100 {
            let scale = x[i].abs();
            for v in &mut x[i..] {
                *v /= scale;
            }
        }
    }
    x
}

/// Dense symmetric eigendecomposition fallback for `tridiagonal_lowest`.
fn tridiagonal_lowest_evd(a: &[f64], b: &[f64]) -> (f64, Vec<f64>) {
    let m = a.len();
    let mut t = faer::Mat::<f64>::zeros(m, m);
    for i in 0..m {
        t[(i, i)] = a[i];
        if i < b.len() {
            t[(i, i + 1)] = b[i];
            t[(i + 1, i)] = b[i];
        }
    }
    let evd = t
        .self_adjoint_eigen(faer::Side::Lower)
        .expect("tridiagonal eigendecomposition converges");
    let lambda = evd.S().column_vector()[0];
    let y = (0..m).map(|i| evd.U()[(i, 0)]).collect();
    (lambda, y)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A diagonal operator.
    struct Diagonal(Vec<f64>);

    impl LinearOperator<C64> for Diagonal {
        fn apply(&self, x: &[C64], out: &mut [C64]) {
            for ((o, &xi), &d) in out.iter_mut().zip(x).zip(&self.0) {
                *o = xi * d;
            }
        }
    }

    #[test]
    fn lanczos_out_of_iterations_returns_its_best_ritz_pair() {
        let dim = 200;
        let op = Diagonal((0..dim).map(|i| (i as f64).sqrt()).collect());
        let v0: Vec<C64> = (0..dim)
            .map(|i| C64::new(1.0 / (dim as f64).sqrt(), 0.0) * C64::cis(i as f64))
            .collect();
        let options = EigenOptions {
            tolerance: 1e-12,
            max_iterations: 6,
            seed: 0,
        };
        let ritz = lanczos_lowest_from(&op, &v0, &[], &options).unwrap();
        assert!(!ritz.converged);
        assert!((norm(&ritz.vector) - 1.0).abs() < 1e-12);
        // The Ritz value is the vector's energy, above the true lowest (0).
        let mut hx = vec![C64::new(0.0, 0.0); dim];
        op.apply(&ritz.vector, &mut hx);
        let energy = dot(&ritz.vector, &hx).re;
        assert!((energy - ritz.value).abs() < 1e-10 && ritz.value > 0.0);
        // Given enough iterations it converges.
        let options = EigenOptions {
            max_iterations: 1000,
            ..options
        };
        assert!(
            lanczos_lowest_from(&op, &v0, &[], &options)
                .unwrap()
                .converged
        );
    }

    #[test]
    fn ritz_tracker_accepts_residuals_at_rounding_of_a_large_spectrum() {
        // Near −4000, 1e-10 is within rounding: a residual estimate of 3e-10
        // counts as converged, but not at a scale near 1.
        let mut far = RitzTracker::new(1e-10);
        assert!(far.step(-4000.0, 3e-10).is_some());
        let mut near = RitzTracker::new(1e-10);
        assert!(near.step(-1.0, 3e-10).is_none());
    }

    #[test]
    fn tridiagonal_lowest_matches_known_spectrum() {
        // The path graph Laplacian-like matrix: diag 2, off-diag -1, size 6,
        // has eigenvalues 2 - 2cos(kπ/7).
        let (a, b) = (vec![2.0; 6], vec![-1.0; 5]);
        let (lambda, y) = tridiagonal_lowest(&a, &b);
        let want = 2.0 - 2.0 * (std::f64::consts::PI / 7.0).cos();
        assert!((lambda - want).abs() < 1e-12, "{lambda} vs {want}");
        // Ty = λy.
        for i in 0..6 {
            let left = if i > 0 { -y[i - 1] } else { 0.0 };
            let right = if i < 5 { -y[i + 1] } else { 0.0 };
            assert!((2.0 * y[i] + left + right - lambda * y[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn tridiagonal_lowest_does_not_overflow_on_large_near_degenerate_matrices() {
        // Regression test: unpivoted Thomas elimination overflowed past 1e154
        // on large Krylov spaces with small off-diagonals / large spectral shift
        // (such as L=20 lattice gauge DMRG), yielding inf/inf = NaN.
        let m = 250;
        let a: Vec<f64> = (0..m)
            .map(|i| -4000.0 + if i % 2 == 0 { 0.0 } else { 100.0 } + 1e-6 * i as f64)
            .collect();
        let b: Vec<f64> = (0..m - 1)
            .map(|i| if i % 10 == 0 { 1e-8 } else { 5.0 })
            .collect();
        let (lambda, y) = tridiagonal_lowest(&a, &b);
        assert!(lambda.is_finite(), "non-finite lambda: {lambda}");
        assert!(y.iter().all(|v| v.is_finite()), "non-finite eigenvector");
        let nrm: f64 = y.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!((nrm - 1.0).abs() < 1e-10, "norm {nrm}");
        let mut max_res = 0.0f64;
        for i in 0..m {
            let mut ty = a[i] * y[i];
            if i > 0 {
                ty += b[i - 1] * y[i - 1];
            }
            if i + 1 < m {
                ty += b[i] * y[i + 1];
            }
            max_res = max_res.max((ty - lambda * y[i]).abs());
        }
        assert!(max_res < 1e-8, "residual {max_res}");
    }
}
