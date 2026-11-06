//! Dense linear algebra for MPS tensors, with explicit parallelism.
//!
//! MPS matrices are mostly small (a few hundred rows at most), where
//! spawning threads costs more than it saves: DMRG ran 4× slower with faer's
//! default parallel settings. faer only offers that setting globally, so
//! these wrappers call its lower-level routines with a per-call choice:
//! sequential below [`PARALLEL_SIZE`], faer's global setting above.

use faer::diag::Diag;
use faer::linalg::matmul::matmul as faer_matmul;
use faer::linalg::svd::{ComputeSvdVectors, svd as faer_svd, svd_scratch};
use faer::traits::Conjugate;
use faer::{Accum, Mat, MatRef, Par};
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

/// `out += a · b`.
pub(crate) fn matmul_add<L, R>(out: &mut Mat<C64>, a: MatRef<'_, L>, b: MatRef<'_, R>)
where
    L: Conjugate<Canonical = C64>,
    R: Conjugate<Canonical = C64>,
{
    let size = (a.nrows() * a.ncols()).max(b.nrows() * b.ncols());
    faer_matmul(
        out.as_mut(),
        Accum::Add,
        a,
        b,
        C64::new(1.0, 0.0),
        par(size),
    );
}

/// The thin SVD `a = U diag(s) V†`, with singular values in decreasing
/// order.
pub(crate) fn thin_svd(a: MatRef<'_, C64>) -> (Mat<C64>, Vec<f64>, Mat<C64>) {
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
        Default::default(),
    );
    let mut buffer = faer::dyn_stack::MemBuffer::new(req);
    faer_svd(
        a,
        s.as_mut(),
        Some(u.as_mut()),
        Some(v.as_mut()),
        par,
        faer::dyn_stack::MemStack::new(&mut buffer),
        Default::default(),
    )
    .expect("SVD converges");
    let s = (0..k).map(|i| s.column_vector()[i].re).collect();
    (u, s, v)
}
