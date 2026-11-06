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
    lanczos_lowest_from(op, &v0, locked, options)
}

/// As [`lanczos_lowest`], from a given unit start vector orthogonal to
/// `locked`. A good guess (such as the previous solution in DMRG) converges
/// in far fewer iterations.
pub(crate) fn lanczos_lowest_from<T: Scalar>(
    op: &impl LinearOperator<T>,
    v0: &[T],
    locked: &[&[T]],
    options: &EigenOptions,
) -> Result<Vec<T>, Error> {
    let dim = v0.len();

    // Pass 1: build the tridiagonal matrix until its lowest Ritz pair
    // converges.
    let (mut alphas, mut betas) = (Vec::new(), Vec::new());
    let mut ritz: Option<Vec<f64>> = None;
    recurrence(
        op,
        v0,
        locked,
        options.max_iterations,
        |_, _, alpha, beta| {
            alphas.push(alpha);
            let (_, s) = tridiagonal_lowest(&alphas, &betas);
            // Residual of the Ritz pair: β times the last eigenvector component.
            if beta * s[s.len() - 1].abs() <= options.tolerance || beta <= breakdown(&alphas) {
                ritz = Some(s);
                return ControlFlow::Break(());
            }
            betas.push(beta);
            ControlFlow::Continue(())
        },
    );
    let s = ritz.ok_or_else(|| {
        Error::NotConverged(format!(
            "Lanczos did not converge in {} iterations",
            options.max_iterations
        ))
    })?;

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
    Ok(x)
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
fn tridiagonal_lowest(a: &[f64], b: &[f64]) -> (f64, Vec<f64>) {
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

    // Inverse iteration: solve (T − λ)y = y_prev by the Thomas algorithm.
    let mut y = vec![1.0; m];
    for _ in 0..3 {
        let mut c = vec![0.0; m];
        let mut d = vec![0.0; m];
        for i in 0..m {
            let sub = if i > 0 { b[i - 1] } else { 0.0 };
            let mut pivot = a[i] - lambda - if i > 0 { sub * c[i - 1] } else { 0.0 };
            if pivot.abs() < tiny {
                pivot = tiny;
            }
            c[i] = if i < b.len() { b[i] / pivot } else { 0.0 };
            d[i] = (y[i] - if i > 0 { sub * d[i - 1] } else { 0.0 }) / pivot;
        }
        for i in (0..m).rev() {
            y[i] = d[i] - if i + 1 < m { c[i] * y[i + 1] } else { 0.0 };
        }
        let n = y.iter().map(|v| v * v).sum::<f64>().sqrt();
        y.iter_mut().for_each(|v| *v /= n);
    }
    (lambda, y)
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
