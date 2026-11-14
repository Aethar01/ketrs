//! Dense linear algebra for MPS tensors, with explicit parallelism.
//!
//! MPS matrices are mostly small (a few hundred rows at most), where
//! spawning threads costs more than it saves: DMRG ran 4× slower with faer's
//! default parallel settings. faer only offers that setting globally, so
//! these wrappers call its lower-level routines with a per-call choice:
//! sequential below [`PARALLEL_SIZE`], faer's global setting above.

use faer::diag::Diag;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::matmul::matmul as faer_matmul;
use faer::linalg::qr::no_pivoting::factor::{
    qr_in_place, qr_in_place_scratch, recommended_block_size,
};
use faer::linalg::svd::{ComputeSvdVectors, SvdError, SvdParams, svd as faer_svd, svd_scratch};
use faer::traits::Conjugate;
use faer::{Accum, Mat, MatRef, Par, Side, Spec};
use num_complex::Complex64 as C64;

/// Matrices with at least this many entries (rows × columns of the larger
/// operand) use faer's global parallelism.
const PARALLEL_SIZE: usize = 256 * 256;

fn par(size: usize) -> Par {
    if size >= PARALLEL_SIZE {
        faer::get_global_parallelism()
    } else {
        Par::Seq
    }
}

/// `a · b`. Either side may be a conjugated or transposed view.
pub(crate) fn matmul<L, R>(a: MatRef<'_, L>, b: MatRef<'_, R>) -> Mat<C64>
where
    L: Conjugate<Canonical = C64>,
    R: Conjugate<Canonical = C64>,
{
    let mut out = Mat::<C64>::zeros(a.nrows(), b.ncols());
    let size = (a.nrows() * a.ncols()).max(b.nrows() * b.ncols());
    faer_matmul(
        out.as_mut(),
        Accum::Replace,
        a,
        b,
        C64::new(1.0, 0.0),
        par(size),
    );
    out
}

/// The triangular factor `R` (`min(m, n) × n`) of `a = Q R`.
///
/// Always sequential: on a 1600 × 128 matrix faer's parallel QR took 3.8 ms
/// on three threads against 2.3 ms on one, and several times that when
/// other processes shared the cores.
pub(crate) fn qr_r(a: MatRef<'_, C64>) -> Mat<C64> {
    let (m, n) = (a.nrows(), a.ncols());
    let k = m.min(n);
    let mut factors = a.to_owned();
    let block = recommended_block_size::<C64>(m, n);
    let mut coeffs = Mat::<C64>::zeros(block, k);
    let req = qr_in_place_scratch::<C64>(m, n, block, Par::Seq, Default::default());
    let mut buffer = MemBuffer::new(req);
    qr_in_place(
        factors.as_mut(),
        coeffs.as_mut(),
        Par::Seq,
        MemStack::new(&mut buffer),
        Default::default(),
    );
    Mat::from_fn(k, n, |i, j| {
        if i <= j {
            factors[(i, j)]
        } else {
            C64::new(0.0, 0.0)
        }
    })
}

/// `(U, s, V)` of a thin SVD.
type Svd = (Mat<C64>, Vec<f64>, Mat<C64>);

/// The thin SVD `a = U diag(s) V†`, with singular values in decreasing
/// order.
///
/// faer's divide-and-conquer SVD can fail on finite, rank-deficient
/// matrices, such as two-site tensors after a subspace expansion pads a bond
/// with directions of zero weight: it either stops with `NoConvergence` or
/// reports success with NaN singular vectors. Either is caught here, and the
/// SVD is retried with plain QR iteration, then on the triangular factor of
/// a QR decomposition, and last through the eigendecomposition of the
/// smaller Gram matrix, which loses the relative accuracy of singular values
/// below √ε of the largest (those are truncated anyway). Only non-finite
/// input panics.
pub(crate) fn thin_svd(a: MatRef<'_, C64>) -> Svd {
    if let Ok(out) = try_svd(a, Default::default()) {
        return out;
    }
    assert!(
        all_finite(a),
        "MPS tensor has non-finite entries: the state has diverged"
    );
    let mut params = <SvdParams as faer::Auto<C64>>::auto();
    params.recursion_threshold = usize::MAX;
    if let Ok(out) = try_svd(a, params.into()) {
        return out;
    }
    if let Some(out) = svd_via_qr(a) {
        return out;
    }
    svd_via_gram(a)
}

fn try_svd(a: MatRef<'_, C64>, params: Spec<SvdParams, C64>) -> Result<Svd, SvdError> {
    let (m, n) = (a.nrows(), a.ncols());
    let k = m.min(n);
    let par = par(m * n);
    let mut u = Mat::<C64>::zeros(m, k);
    let mut v = Mat::<C64>::zeros(n, k);
    let mut s = Diag::<C64>::zeros(k);
    let req = svd_scratch::<C64>(
        m,
        n,
        ComputeSvdVectors::Thin,
        ComputeSvdVectors::Thin,
        par,
        params,
    );
    let mut buffer = MemBuffer::new(req);
    faer_svd(
        a,
        s.as_mut(),
        Some(u.as_mut()),
        Some(v.as_mut()),
        par,
        MemStack::new(&mut buffer),
        params,
    )?;
    let s: Vec<f64> = (0..k).map(|i| s.column_vector()[i].re).collect();
    if !(s.iter().all(|x| x.is_finite()) && all_finite(u.as_ref()) && all_finite(v.as_ref())) {
        return Err(SvdError::NoConvergence);
    }
    Ok((u, s, v))
}

/// SVD of the square triangular factor: `a = Q R` (tall) or `a† = Q R`
/// (wide), with the factor's singular vectors rotated back by `Q`.
fn svd_via_qr(a: MatRef<'_, C64>) -> Option<Svd> {
    if a.nrows() >= a.ncols() {
        let qr = a.qr();
        let (ur, s, v) = try_svd(qr.thin_R(), Default::default()).ok()?;
        Some((matmul(qr.compute_thin_Q().as_ref(), ur.as_ref()), s, v))
    } else {
        let qr = a.adjoint().qr();
        let r = qr.thin_R().adjoint().to_owned();
        let (u, s, w) = try_svd(r.as_ref(), Default::default()).ok()?;
        Some((u, s, matmul(qr.compute_thin_Q().as_ref(), w.as_ref())))
    }
}

fn svd_via_gram(a: MatRef<'_, C64>) -> Svd {
    let wide = a.nrows() < a.ncols();
    // b is the tall one of a and a†; b = U s V†.
    let b = if wide {
        a.adjoint().to_owned()
    } else {
        a.to_owned()
    };
    let gram = matmul(b.adjoint(), b.as_ref());
    let eig = gram
        .self_adjoint_eigen(Side::Lower)
        .expect("Hermitian eigendecomposition converges");
    let k = b.ncols();
    let order: Vec<usize> = (0..k).rev().collect(); // eigenvalues ascend
    let s: Vec<f64> = order
        .iter()
        .map(|&i| eig.S().column_vector()[i].re.max(0.0).sqrt())
        .collect();
    let v = Mat::<C64>::from_fn(k, k, |i, j| eig.U()[(i, order[j])]);
    // b V = U diag(s): orthonormalise its columns by QR rather than dividing
    // by s, which would blow up the directions with s ≈ 0. R is then
    // diagonal up to rounding, and its phases go into U.
    let bv = matmul(b.as_ref(), v.as_ref());
    let qr = bv.qr();
    let mut u = qr.compute_thin_Q();
    let r = qr.thin_R();
    for j in 0..k {
        let d = r[(j, j)];
        if d.norm() > 0.0 {
            let phase = d / d.norm();
            for i in 0..u.nrows() {
                u[(i, j)] *= phase;
            }
        }
    }
    if wide { (v, s, u) } else { (u, s, v) }
}

fn all_finite(a: MatRef<'_, C64>) -> bool {
    (0..a.ncols()).all(|j| (0..a.nrows()).all(|i| a[(i, j)].is_finite()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random(m: usize, n: usize, seed: u64) -> Mat<C64> {
        let mut x = seed;
        let mut next = move || {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (x >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        Mat::from_fn(m, n, |_, _| C64::new(next(), next()))
    }

    fn check(a: &Mat<C64>, (u, s, v): (Mat<C64>, Vec<f64>, Mat<C64>), tol: f64) {
        let k = s.len();
        assert_eq!((u.ncols(), v.ncols()), (k, k));
        assert!(s.windows(2).all(|w| w[0] >= w[1]));
        let us = Mat::from_fn(u.nrows(), k, |i, j| u[(i, j)] * s[j]);
        let back = matmul(us.as_ref(), v.adjoint());
        let err = (0..a.nrows())
            .flat_map(|i| (0..a.ncols()).map(move |j| (i, j)))
            .map(|(i, j)| (back[(i, j)] - a[(i, j)]).norm())
            .fold(0.0, f64::max);
        assert!(err < tol, "reconstruction error {err}");
        let uu = matmul(u.adjoint(), u.as_ref());
        let vv = matmul(v.adjoint(), v.as_ref());
        for i in 0..k {
            for j in 0..k {
                let id = if i == j { 1.0 } else { 0.0 };
                assert!((uu[(i, j)].re - id).abs() < tol && (vv[(i, j)].re - id).abs() < tol);
            }
        }
    }

    #[test]
    fn qr_factor_is_triangular_with_the_same_gram_matrix() {
        // Tall (blocked), wide, and smaller than one block.
        for (m, n) in [(300, 40), (12, 30), (9, 5)] {
            let a = random(m, n, (m * 7 + n) as u64);
            let r = qr_r(a.as_ref());
            assert_eq!((r.nrows(), r.ncols()), (m.min(n), n));
            for i in 0..r.nrows() {
                for j in 0..i {
                    assert_eq!(r[(i, j)], C64::new(0.0, 0.0));
                }
            }
            let (want, got) = (
                matmul(a.adjoint(), a.as_ref()),
                matmul(r.adjoint(), r.as_ref()),
            );
            let diff = (want - got).norm_max();
            assert!(diff < 1e-11, "{m} × {n}: {diff}");
        }
    }

    #[test]
    fn svd_fallbacks_reconstruct_tall_and_wide_matrices() {
        for (m, n) in [(40, 12), (12, 40), (20, 20)] {
            let a = random(m, n, (m * 100 + n) as u64);
            check(&a, thin_svd(a.as_ref()), 1e-12);
            check(&a, svd_via_qr(a.as_ref()).unwrap(), 1e-12);
            check(&a, svd_via_gram(a.as_ref()), 1e-10);
        }
    }

    #[test]
    fn svd_fallbacks_handle_rank_deficient_matrices() {
        // Rank 5 with zero-padded rows and columns, as after an expansion.
        let (m, n, rank) = (48, 36, 5);
        let x = random(m - 8, rank, 7);
        let y = random(rank, n - 6, 8);
        let core = matmul(x.as_ref(), y.as_ref());
        let a = Mat::from_fn(m, n, |i, j| {
            if i < m - 8 && j < n - 6 {
                core[(i, j)]
            } else {
                C64::new(0.0, 0.0)
            }
        });
        check(&a, thin_svd(a.as_ref()), 1e-12);
        check(&a, svd_via_qr(a.as_ref()).unwrap(), 1e-12);
        check(&a, svd_via_gram(a.as_ref()), 1e-7);
    }
}
