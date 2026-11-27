//! Hafnians, loop hafnians, and torontonians for Gaussian boson sampling
//! and photon-counting / threshold probabilities.
//!
//! Algorithms:
//! - General hafnian & loop hafnian: symmetric `2 × 2` polynomial Schur
//!   complement of `K(λ) = X − λ A`, running in `O(n² 2ⁿ)` for a `2n × 2n`
//!   matrix with zero divisions by matrix entries and exact real (`f64`) /
//!   complex (`Complex64`) specializations, parallelised with rayon.
//! - Repeated-entry hafnian & loop hafnian: automatically selects between the
//!   compressed Glynn finite-difference trace algorithm (on the smaller
//!   `m × m` compressed matrix) and the expanded Schur-complement algorithm.
//! - Torontonian and Gaussian threshold probabilities: recursive `2 × 2`
//!   block Cholesky update (Bulmer et al. 2022) in `O(m² 2ᵐ)`, parallelised
//!   with rayon, with a general complex fallback for non-Hermitian inputs.

use crate::error::Error;
use faer::Mat;
use num_complex::Complex64 as C64;
use rayon::prelude::*;
use std::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

/// Threshold in pair count `n` above which the Schur-complement recursion
/// splits top levels across rayon tasks.
const SCHUR_PAR_MIN_PAIRS: usize = 7;

/// Threshold in mode count `m` above which the recursive Cholesky torontonian
/// splits top levels across rayon tasks.
const TOR_PAR_MIN_MODES: usize = 10;

/// Scalar trait unifying real (`f64`) and complex (`C64`) hafnian kernels.
trait HafScalar:
    Copy
    + Default
    + Send
    + Sync
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + Mul<Output = Self>
    + Neg<Output = Self>
{
    fn zero() -> Self;
    fn one() -> Self;
    fn scale_f64(self, k: f64) -> Self;
    fn into_c64(self) -> C64;
}

impl HafScalar for f64 {
    #[inline(always)]
    fn zero() -> Self {
        0.0
    }
    #[inline(always)]
    fn one() -> Self {
        1.0
    }
    #[inline(always)]
    fn scale_f64(self, k: f64) -> Self {
        self * k
    }
    #[inline(always)]
    fn into_c64(self) -> C64 {
        C64::new(self, 0.0)
    }
}

impl HafScalar for C64 {
    #[inline(always)]
    fn zero() -> Self {
        C64::new(0.0, 0.0)
    }
    #[inline(always)]
    fn one() -> Self {
        C64::new(1.0, 0.0)
    }
    #[inline(always)]
    fn scale_f64(self, k: f64) -> Self {
        C64::new(self.re * k, self.im * k)
    }
    #[inline(always)]
    fn into_c64(self) -> C64 {
        self
    }
}

fn is_all_real(slice: &[C64]) -> bool {
    slice.iter().all(|z| z.im == 0.0)
}

fn check_symmetric(matrix: &[C64], n: usize) -> Result<(), Error> {
    for i in 0..n {
        for j in (i + 1)..n {
            let a = matrix[i * n + j];
            let b = matrix[j * n + i];
            let scale = a.norm().max(b.norm()).max(1.0);
            let diff = (a - b).norm();
            if !diff.is_finite() || diff > 1e-9 * scale {
                return Err(Error::InvalidArgument(format!(
                    "hafnian requires a symmetric matrix, but A[{i},{j}] = {a} != A[{j},{i}] = {b}"
                )));
            }
        }
    }
    Ok(())
}

/// Hafnian of an `n × n` row-major symmetric matrix (diagonal entries are
/// ignored).
///
/// Returns `1` for `n = 0` and `0` for odd `n`.
pub fn hafnian(matrix: &[C64], n: usize) -> Result<C64, Error> {
    if matrix.len() != n * n {
        return Err(Error::InvalidArgument(format!(
            "hafnian expects {n}×{n} ({}) entries, got {}",
            n * n,
            matrix.len()
        )));
    }
    check_symmetric(matrix, n)?;
    if n == 0 {
        return Ok(C64::new(1.0, 0.0));
    }
    if n % 2 == 1 {
        return Ok(C64::new(0.0, 0.0));
    }
    if is_all_real(matrix) {
        let a: Vec<f64> = matrix.iter().map(|z| z.re).collect();
        Ok(hafnian_dispatch::<f64, false>(&a, &[], n).into_c64())
    } else {
        Ok(hafnian_dispatch::<C64, false>(matrix, &[], n))
    }
}

/// Loop hafnian of an `n × n` row-major symmetric matrix with self-loop
/// weights `diag` (length `n`).
///
/// Off-diagonal entries of `matrix` give edge weights (its diagonal is
/// ignored); `diag[i]` is the loop weight on vertex `i`. Works for both even
/// and odd `n`.
pub fn loop_hafnian(matrix: &[C64], diag: &[C64], n: usize) -> Result<C64, Error> {
    if matrix.len() != n * n {
        return Err(Error::InvalidArgument(format!(
            "loop_hafnian expects {n}×{n} ({}) entries, got {}",
            n * n,
            matrix.len()
        )));
    }
    if diag.len() != n {
        return Err(Error::InvalidArgument(format!(
            "loop_hafnian expects {n} diagonal entries, got {}",
            diag.len()
        )));
    }
    check_symmetric(matrix, n)?;
    if n == 0 {
        return Ok(C64::new(1.0, 0.0));
    }
    if n == 1 {
        return Ok(diag[0]);
    }
    if diag.iter().all(|z| z.re == 0.0 && z.im == 0.0) {
        return hafnian(matrix, n);
    }
    if is_all_real(matrix) && is_all_real(diag) {
        let a: Vec<f64> = matrix.iter().map(|z| z.re).collect();
        let d: Vec<f64> = diag.iter().map(|z| z.re).collect();
        Ok(loop_hafnian_even_or_odd::<f64>(&a, &d, n).into_c64())
    } else {
        Ok(loop_hafnian_even_or_odd::<C64>(matrix, diag, n))
    }
}

/// Hafnian of the expanded matrix formed by repeating row/column `i` of the
/// `m × m` symmetric `matrix` `repeats[i]` times.
///
/// Diagonal entries `matrix[i * m + i]` give the edge weight between two
/// distinct copies of index `i`.
pub fn hafnian_repeated(matrix: &[C64], repeats: &[usize]) -> Result<C64, Error> {
    let m = repeats.len();
    if matrix.len() != m * m {
        return Err(Error::InvalidArgument(format!(
            "hafnian_repeated expects {m}×{m} ({}) entries, got {}",
            m * m,
            matrix.len()
        )));
    }
    check_symmetric(matrix, m)?;
    let total: usize = repeats.iter().sum();
    if total == 0 {
        return Ok(C64::new(1.0, 0.0));
    }
    if total % 2 == 1 {
        return Ok(C64::new(0.0, 0.0));
    }
    if is_all_real(matrix) {
        let a: Vec<f64> = matrix.iter().map(|z| z.re).collect();
        Ok(repeated_dispatch::<f64, false>(&a, &[], repeats).into_c64())
    } else {
        Ok(repeated_dispatch::<C64, false>(matrix, &[], repeats))
    }
}

/// Loop hafnian of the expanded matrix formed by repeating row/column `i` of
/// `matrix` (`m × m`) and `diag` (`m`) `repeats[i]` times.
///
/// `matrix[i * m + j]` is the edge weight between distinct copies of `i` and
/// `j` (including `i == j` when `repeats[i] >= 2`); `diag[i]` is the loop
/// weight on each copy of `i`.
pub fn loop_hafnian_repeated(
    matrix: &[C64],
    diag: &[C64],
    repeats: &[usize],
) -> Result<C64, Error> {
    let m = repeats.len();
    if matrix.len() != m * m {
        return Err(Error::InvalidArgument(format!(
            "loop_hafnian_repeated expects {m}×{m} ({}) entries, got {}",
            m * m,
            matrix.len()
        )));
    }
    if diag.len() != m {
        return Err(Error::InvalidArgument(format!(
            "loop_hafnian_repeated expects {m} diagonal entries, got {}",
            diag.len()
        )));
    }
    check_symmetric(matrix, m)?;
    let total: usize = repeats.iter().sum();
    if total == 0 {
        return Ok(C64::new(1.0, 0.0));
    }
    if diag.iter().all(|z| z.re == 0.0 && z.im == 0.0) {
        return hafnian_repeated(matrix, repeats);
    }
    if total == 1 {
        let i = repeats.iter().position(|&r| r == 1).unwrap();
        return Ok(diag[i]);
    }
    if is_all_real(matrix) && is_all_real(diag) {
        let a: Vec<f64> = matrix.iter().map(|z| z.re).collect();
        let d: Vec<f64> = diag.iter().map(|z| z.re).collect();
        Ok(repeated_dispatch::<f64, true>(&a, &d, repeats).into_c64())
    } else {
        Ok(repeated_dispatch::<C64, true>(matrix, diag, repeats))
    }
}

/// Pad odd `n` with one isolated vertex of loop weight 1, then run the even
/// loop-hafnian kernel.
fn loop_hafnian_even_or_odd<T: HafScalar>(a: &[T], diag: &[T], n: usize) -> T {
    if n.is_multiple_of(2) {
        hafnian_dispatch::<T, true>(a, diag, n)
    } else {
        let n2 = n + 1;
        let mut a2 = vec![T::zero(); n2 * n2];
        for i in 0..n {
            for j in 0..n {
                a2[i * n2 + j] = a[i * n + j];
            }
        }
        let mut d2 = Vec::with_capacity(n2);
        d2.extend_from_slice(diag);
        d2.push(T::one());
        hafnian_dispatch::<T, true>(&a2, &d2, n2)
    }
}

/// Small-size fast paths + symmetric polynomial Schur complement for even `n`.
fn hafnian_dispatch<T: HafScalar, const HAS_LOOPS: bool>(a: &[T], diag: &[T], n: usize) -> T {
    debug_assert_eq!(n % 2, 0);
    let pairs = n / 2;
    match pairs {
        0 => T::one(),
        1 => {
            let e = a[1];
            if HAS_LOOPS { e + diag[0] * diag[1] } else { e }
        }
        2 => {
            // 4×4 exact closed form.
            let a01 = a[1];
            let a02 = a[2];
            let a03 = a[3];
            let a12 = a[6];
            let a13 = a[7];
            let a23 = a[11];
            let mut sum = a01 * a23 + a02 * a13 + a03 * a12;
            if HAS_LOOPS {
                let (v0, v1, v2, v3) = (diag[0], diag[1], diag[2], diag[3]);
                sum += a01 * v2 * v3
                    + a02 * v1 * v3
                    + a03 * v1 * v2
                    + a12 * v0 * v3
                    + a13 * v0 * v2
                    + a23 * v0 * v1
                    + v0 * v1 * v2 * v3;
            }
            sum
        }
        _ => schur_hafnian::<T, HAS_LOOPS>(a, diag, pairs),
    }
}

/// Choose between compressed Glynn and expanded Schur complement for repeated
/// matrices.
fn repeated_dispatch<T: HafScalar, const HAS_LOOPS: bool>(
    matrix: &[T],
    diag: &[T],
    repeats: &[usize],
) -> T {
    let m = repeats.len();
    let active: Vec<usize> = (0..m).filter(|&i| repeats[i] > 0).collect();
    let k = active.len();
    let total: usize = active.iter().map(|&i| repeats[i]).sum();
    if total == 0 {
        return T::one();
    }
    if !HAS_LOOPS && total % 2 == 1 {
        return T::zero();
    }
    let has_repeats = active.iter().any(|&i| repeats[i] >= 2);

    // Estimate Glynn grid size vs Schur tree size.
    let pairs = total.div_ceil(2);
    let mut glynn_grid: u128 = 1;
    let mut n_odd = 0usize;
    for &i in &active {
        let r = repeats[i];
        glynn_grid = glynn_grid.saturating_mul((r / 2 + 1) as u128);
        if r % 2 == 1 {
            n_odd += 1;
        }
    }
    // Odd total adds one dummy odd mode of count 1.
    let n_odd_padded = n_odd + (total % 2);
    let n_cross = n_odd_padded / 2;
    glynn_grid = glynn_grid.saturating_mul(1u128.checked_shl(n_cross as u32).unwrap_or(u128::MAX));

    let k_eff = k + (total % 2);
    let glynn_cost = (glynn_grid / 2).saturating_mul(
        ((pairs / 2).max(1) as u128)
            .saturating_mul((k_eff * k_eff * k_eff) as u128)
            .max(1),
    );
    let schur_cost = if pairs >= 60 {
        u128::MAX
    } else {
        (1u128 << pairs).saturating_mul((6 * pairs * pairs) as u128)
    };

    if has_repeats && glynn_cost < schur_cost {
        compressed_glynn::<T, HAS_LOOPS>(matrix, diag, m, &active, repeats, total)
    } else {
        // Expand into a total × total matrix (padding if odd + loops).
        let n_exp = if total % 2 == 1 { total + 1 } else { total };
        let mut idx_map = Vec::with_capacity(total);
        for &i in &active {
            for _ in 0..repeats[i] {
                idx_map.push(i);
            }
        }
        let mut a_exp = vec![T::zero(); n_exp * n_exp];
        for r in 0..total {
            let ri = idx_map[r];
            for c in 0..total {
                if r != c {
                    a_exp[r * n_exp + c] = matrix[ri * m + idx_map[c]];
                }
            }
        }
        if HAS_LOOPS {
            let mut d_exp = Vec::with_capacity(n_exp);
            for &ri in &idx_map {
                d_exp.push(diag[ri]);
            }
            if total % 2 == 1 {
                d_exp.push(T::one());
            }
            hafnian_dispatch::<T, true>(&a_exp, &d_exp, n_exp)
        } else {
            hafnian_dispatch::<T, false>(&a_exp, &[], n_exp)
        }
    }
}

// ---------------------------------------------------------------------------
// Symmetric 2×2 polynomial Schur-complement loop hafnian
// ---------------------------------------------------------------------------

/// Upper-triangle entry count for a `dim × dim` symmetric matrix.
#[inline(always)]
fn ut_len(dim: usize) -> usize {
    dim * (dim + 1) / 2
}

/// Index `(row, col)` (`row <= col`) in row-major upper triangle of `dim × dim`.
#[inline(always)]
fn ut_idx(dim: usize, row: usize, col: usize) -> usize {
    debug_assert!(row <= col && col < dim);
    row * (2 * dim - row - 1) / 2 + col
}

/// Pre-allocated per-depth buffers for one thread of the Schur recursion.
struct SchurWorkspace<T> {
    /// For each remaining pair count `rem = 1..pairs`, storage for `K`
    /// (`ut_len(2*rem) * (n+1)`).
    k_stack: Vec<Vec<T>>,
    /// For each remaining pair count `rem = 1..pairs`, storage for `v`
    /// (`2*rem * n`).
    v_stack: Vec<Vec<T>>,
    /// For each remaining pair count `rem = 1..pairs`, storage for `h`
    /// (`n + 1`).
    h_stack: Vec<Vec<T>>,
    /// Scratch polynomials (length `n + 1`).
    d: Vec<T>,
    inv: Vec<T>,
    s00: Vec<T>,
    s01: Vec<T>,
    s11: Vec<T>,
    u0: Vec<T>,
    u1: Vec<T>,
    w0: Vec<T>,
    w1: Vec<T>,
    h_inc: Vec<T>,
    exp_p: Vec<T>,
    inv_k: Vec<f64>,
}

impl<T: HafScalar> SchurWorkspace<T> {
    fn new(pairs: usize, has_loops: bool) -> Self {
        let poly_len = pairs + 1;
        let max_r = 2 * pairs;
        let k_stack = (0..pairs)
            .map(|rem| vec![T::zero(); ut_len(2 * rem) * poly_len])
            .collect();
        let v_stack = if has_loops {
            (0..pairs)
                .map(|rem| vec![T::zero(); 2 * rem * pairs])
                .collect()
        } else {
            Vec::new()
        };
        let h_stack = (0..pairs).map(|_| vec![T::zero(); poly_len]).collect();
        let inv_k: Vec<f64> = (0..=pairs)
            .map(|k| if k == 0 { 0.0 } else { 1.0 / (k as f64) })
            .collect();
        SchurWorkspace {
            k_stack,
            v_stack,
            h_stack,
            d: vec![T::zero(); poly_len],
            inv: vec![T::zero(); pairs],
            s00: vec![T::zero(); pairs],
            s01: vec![T::zero(); pairs],
            s11: vec![T::zero(); pairs],
            u0: vec![T::zero(); pairs],
            u1: vec![T::zero(); pairs],
            w0: vec![T::zero(); max_r * pairs],
            w1: vec![T::zero(); max_r * pairs],
            h_inc: vec![T::zero(); poly_len],
            exp_p: vec![T::zero(); poly_len],
            inv_k,
        }
    }
}

fn schur_hafnian<T: HafScalar, const HAS_LOOPS: bool>(a: &[T], diag: &[T], pairs: usize) -> T {
    let dim = 2 * pairs;
    let poly_len = pairs + 1;
    let mut k_root = vec![T::zero(); ut_len(dim) * poly_len];

    // Build K(λ) = X − λ A in row-major upper triangle.
    for i in 0..dim {
        for j in i..dim {
            let off = ut_idx(dim, i, j) * poly_len;
            if i == j {
                continue;
            }
            if i % 2 == 0 && j == i + 1 {
                k_root[off] = T::one();
            }
            k_root[off + 1] = -a[i * dim + j];
        }
    }

    let v_root = if HAS_LOOPS {
        let mut v = vec![T::zero(); dim * pairs];
        for i in 0..dim {
            v[i * pairs] = diag[i];
        }
        v
    } else {
        Vec::new()
    };

    let h_root = vec![T::zero(); poly_len];
    let sign = if pairs.is_multiple_of(2) { 1.0 } else { -1.0 };

    if pairs >= SCHUR_PAR_MIN_PAIRS {
        // Split the top levels across rayon so every worker gets a large
        // independent subtree with its own L1-resident scratch buffers.
        let par_depth = match pairs {
            6 => 3,
            7 => 4,
            _ => 5.min(pairs - 2),
        };
        schur_par::<T, HAS_LOOPS>(&k_root, &v_root, &h_root, pairs, pairs, sign, par_depth)
    } else {
        let mut ws = SchurWorkspace::<T>::new(pairs, HAS_LOOPS);
        schur_seq::<T, HAS_LOOPS>(&k_root, &v_root, &h_root, pairs, pairs, sign, &mut ws)
    }
}

/// Top-level parallel split: clones only the small remaining subproblem for
/// the include branch, and slices zero-copy for the exclude branch.
fn schur_par<T: HafScalar, const HAS_LOOPS: bool>(
    k_mat: &[T],
    v_vec: &[T],
    h: &[T],
    n: usize,
    rem_pairs: usize,
    sign: f64,
    par_depth: usize,
) -> T {
    if par_depth == 0 || rem_pairs <= 2 {
        let mut ws = SchurWorkspace::<T>::new(n, HAS_LOOPS);
        return schur_seq::<T, HAS_LOOPS>(k_mat, v_vec, h, n, rem_pairs, sign, &mut ws);
    }

    let poly_len = n + 1;
    let dim = 2 * rem_pairs;
    let next_rem = rem_pairs - 1;
    let r = 2 * next_rem;

    // Exclude branch: zero-copy suffix slice of K and v.
    let drop_entries = 2 * dim - 1;
    let k_exc = &k_mat[drop_entries * poly_len..];
    let v_exc = if HAS_LOOPS { &v_vec[2 * n..] } else { &[] };

    // Include branch: allocate child buffers for the parallel split.
    let mut ws = SchurWorkspace::<T>::new(n, HAS_LOOPS);
    let mut h_inc = h.to_vec();
    let mut k_inc = vec![T::zero(); ut_len(r) * poly_len];
    let mut v_inc = if HAS_LOOPS {
        vec![T::zero(); r * n]
    } else {
        Vec::new()
    };

    eliminate_pair::<T, HAS_LOOPS>(
        k_mat, v_vec, &mut h_inc, &mut k_inc, &mut v_inc, n, rem_pairs, &mut ws,
    );

    let (a, b) = rayon::join(
        || schur_par::<T, HAS_LOOPS>(k_exc, v_exc, h, n, next_rem, sign, par_depth - 1),
        || schur_par::<T, HAS_LOOPS>(&k_inc, &v_inc, &h_inc, n, next_rem, -sign, par_depth - 1),
    );
    a + b
}

/// Sequential zero-allocation recursion using per-depth scratch buffers.
fn schur_seq<T: HafScalar, const HAS_LOOPS: bool>(
    k_mat: &[T],
    v_vec: &[T],
    h: &[T],
    n: usize,
    rem_pairs: usize,
    sign: f64,
    ws: &mut SchurWorkspace<T>,
) -> T {
    let poly_len = n + 1;
    if rem_pairs == 1 {
        // Leaf: evaluate exclude (h unchanged) and include (eliminate final 2×2 block).
        let exc_val = exp_coeff_n(h, &mut ws.exp_p, &ws.inv_k, n).scale_f64(sign);

        ws.h_inc.copy_from_slice(h);
        eliminate_last_pair::<T, HAS_LOOPS>(k_mat, v_vec, n, ws);
        let inc_val = exp_coeff_n(&ws.h_inc, &mut ws.exp_p, &ws.inv_k, n).scale_f64(-sign);
        return exc_val + inc_val;
    }

    let dim = 2 * rem_pairs;
    let next_rem = rem_pairs - 1;
    let drop_entries = 2 * dim - 1;

    // Exclude branch: zero-copy suffix slice.
    let k_exc = &k_mat[drop_entries * poly_len..];
    let v_exc = if HAS_LOOPS { &v_vec[2 * n..] } else { &[] };
    let sum_exc = schur_seq::<T, HAS_LOOPS>(k_exc, v_exc, h, n, next_rem, sign, ws);

    // Include branch: use stack slot `next_rem` (zero heap allocations).
    let mut k_child = std::mem::take(&mut ws.k_stack[next_rem]);
    let mut v_child = if HAS_LOOPS {
        std::mem::take(&mut ws.v_stack[next_rem])
    } else {
        Vec::new()
    };
    let mut h_child = std::mem::take(&mut ws.h_stack[next_rem]);
    h_child.copy_from_slice(h);

    eliminate_pair::<T, HAS_LOOPS>(
        k_mat,
        v_vec,
        &mut h_child,
        &mut k_child,
        &mut v_child,
        n,
        rem_pairs,
        ws,
    );

    let sum_inc = schur_seq::<T, HAS_LOOPS>(&k_child, &v_child, &h_child, n, next_rem, -sign, ws);

    ws.k_stack[next_rem] = k_child;
    if HAS_LOOPS {
        ws.v_stack[next_rem] = v_child;
    }
    ws.h_stack[next_rem] = h_child;

    sum_exc + sum_inc
}

/// Eliminate the pivot `2 × 2` pair at the front of `k_mat` when `rem_pairs == 1`,
/// accumulating into `ws.h_inc`.
fn eliminate_last_pair<T: HafScalar, const HAS_LOOPS: bool>(
    k_mat: &[T],
    v_vec: &[T],
    n: usize,
    ws: &mut SchurWorkspace<T>,
) {
    let poly_len = n + 1;
    // Upper triangle of 2×2 has indices: 0 -> (0,0), 1 -> (0,1), 2 -> (1,1).
    let a = &k_mat[0..poly_len];
    let b = &k_mat[poly_len..2 * poly_len];
    let c = &k_mat[2 * poly_len..3 * poly_len];

    compute_det_and_logdet(a, b, c, n, &mut ws.d, &mut ws.inv, &mut ws.h_inc);

    if HAS_LOOPS {
        let v0 = &v_vec[0..n];
        let v1 = &v_vec[n..2 * n];
        compute_loop_pivot(
            a,
            b,
            c,
            &ws.inv,
            v0,
            v1,
            n,
            &mut ws.u0,
            &mut ws.u1,
            &mut ws.h_inc,
        );
    }
}

/// Eliminate the pivot `2 × 2` pair at the front of `k_mat` and write the
/// Schur complement into `k_out` (and `v_out` when `HAS_LOOPS`).
#[allow(clippy::too_many_arguments)]
fn eliminate_pair<T: HafScalar, const HAS_LOOPS: bool>(
    k_mat: &[T],
    v_vec: &[T],
    h_out: &mut [T],
    k_out: &mut [T],
    v_out: &mut [T],
    n: usize,
    rem_pairs: usize,
    ws: &mut SchurWorkspace<T>,
) {
    let poly_len = n + 1;
    let dim = 2 * rem_pairs;
    let r = dim - 2;

    // Pivot block: (0,0) at 0, (0,1) at 1, (1,1) at dim.
    let a = &k_mat[0..poly_len];
    let b = &k_mat[poly_len..2 * poly_len];
    let c = &k_mat[dim * poly_len..(dim + 1) * poly_len];

    compute_det_and_logdet(a, b, c, n, &mut ws.d, &mut ws.inv, h_out);

    if HAS_LOOPS {
        let v0 = &v_vec[0..n];
        let v1 = &v_vec[n..2 * n];
        compute_loop_pivot(a, b, c, &ws.inv, v0, v1, n, &mut ws.u0, &mut ws.u1, h_out);
    }

    // S = K_00^{-1} = inv * [[-c, b], [b, -a]], needed up to degree n - 2.
    // Since a[0] = c[0] = 0 and b[0] = inv[0] = 1: s00[0] = s11[0] = 0, s01[0] = 1.
    let s_len = n.saturating_sub(1);
    if s_len > 0 {
        ws.s00[0] = T::zero();
        ws.s01[0] = T::one();
        ws.s11[0] = T::zero();
    }
    for k in 1..s_len {
        let mut s00 = T::zero();
        let mut s01 = b[k];
        let mut s11 = T::zero();
        for (j, &iv) in ws.inv[..k].iter().enumerate() {
            let kmj = k - j;
            s00 -= iv * c[kmj];
            s11 -= iv * a[kmj];
        }
        for (jm1, &iv) in ws.inv[1..=k].iter().enumerate() {
            s01 += iv * b[k - (jm1 + 1)];
        }
        ws.s00[k] = s00;
        ws.s01[k] = s01;
        ws.s11[k] = s11;
    }

    // E_0[i] = K[0, 2 + i] at index (2 + i), E_1[i] = K[1, 2 + i] at index (dim + 1 + i).
    let e0_base = 2 * poly_len;
    let e1_base = (dim + 1) * poly_len;

    // Precompute W_0[i] = s00 * E_0[i] + s01 * E_1[i], W_1[i] = s01 * E_0[i] + s11 * E_1[i]
    // for degrees 1..n (since E[0] == 0, s00[0] == s11[0] == 0, s01[0] == 1).
    for i in 0..r {
        let e0 = &k_mat[e0_base + i * poly_len..e0_base + (i + 1) * poly_len];
        let e1 = &k_mat[e1_base + i * poly_len..e1_base + (i + 1) * poly_len];
        let w0 = &mut ws.w0[i * n..(i + 1) * n];
        let w1 = &mut ws.w1[i * n..(i + 1) * n];
        w0[0] = T::zero();
        w1[0] = T::zero();
        for k in 1..n {
            // t = k term uses s01[0] = 1, s00[0] = s11[0] = 0.
            let mut acc0 = e1[k];
            let mut acc1 = e0[k];
            for t in 1..k {
                let s_idx = k - t;
                let x0 = e0[t];
                let x1 = e1[t];
                acc0 += ws.s00[s_idx] * x0 + ws.s01[s_idx] * x1;
                acc1 += ws.s01[s_idx] * x0 + ws.s11[s_idx] * x1;
            }
            w0[k] = acc0;
            w1[k] = acc1;
        }
    }

    // Update upper triangle of K_RR (contiguous suffix starting at drop_entries = 2*dim - 1).
    let drop_entries = 2 * dim - 1;
    let k_rr = &k_mat[drop_entries * poly_len..];
    k_out.copy_from_slice(k_rr);

    let mut out_entry = 0usize;
    for i in 0..r {
        let e0 = &k_mat[e0_base + i * poly_len..e0_base + (i + 1) * poly_len];
        let e1 = &k_mat[e1_base + i * poly_len..e1_base + (i + 1) * poly_len];
        for j in i..r {
            let w0 = &ws.w0[j * n..(j + 1) * n];
            let w1 = &ws.w1[j * n..(j + 1) * n];
            let dst = &mut k_out[out_entry * poly_len..(out_entry + 1) * poly_len];
            // E[0] == 0 and W[0] == 0, so the product starts at λ².
            for k in 2..=n {
                let mut sub = T::zero();
                for t in 1..k {
                    sub += e0[t] * w0[k - t] + e1[t] * w1[k - t];
                }
                dst[k] -= sub;
            }
            out_entry += 1;
        }
    }

    if HAS_LOOPS {
        // v'_R[i] = v_R[i] - (E_0[i] * u0 + E_1[i] * u1) for degrees 1..n.
        v_out.copy_from_slice(&v_vec[2 * n..]);
        for i in 0..r {
            let e0 = &k_mat[e0_base + i * poly_len..e0_base + (i + 1) * poly_len];
            let e1 = &k_mat[e1_base + i * poly_len..e1_base + (i + 1) * poly_len];
            let dst = &mut v_out[i * n..(i + 1) * n];
            for k in 1..n {
                let mut sub = T::zero();
                for t in 1..=k {
                    sub += e0[t] * ws.u0[k - t] + e1[t] * ws.u1[k - t];
                }
                dst[k] -= sub;
            }
        }
    }
}

/// Compute `d = b² − a·c`, `inv = d⁻¹` (up to degree `n - 1`), and add the
/// log-det contribution `h[k] += -0.5 * [λ^{k-1}](d' * inv)` for `k = 1..=n`.
#[inline]
fn compute_det_and_logdet<T: HafScalar>(
    a: &[T],
    b: &[T],
    c: &[T],
    n: usize,
    d: &mut [T],
    inv: &mut [T],
    h_out: &mut [T],
) {
    // d = b^2 - a*c. Note b[0] = 1, a[0] = c[0] = 0.
    // Exploit symmetry in b[j] * b[k - j] to halve squaring multiplications.
    d[0] = T::one();
    for k in 1..=n {
        let half = k / 2;
        let mut sq = b[k]; // j = 0 term: b[0] * b[k] = b[k]
        for j in 1..half {
            sq += b[j] * b[k - j];
        }
        let mut acc = sq + sq;
        if k.is_multiple_of(2) {
            acc += b[half] * b[half];
        } else if half > 0 {
            // j = half and k - half = half + 1 wasn't included in 1..half
            let mid = b[half] * b[half + 1];
            acc += mid + mid;
        }
        // a[0] = c[0] = 0 so j runs 1..k.
        for j in 1..k {
            acc -= a[j] * c[k - j];
        }
        d[k] = acc;
    }

    // inv = d^{-1} up to degree n - 1.
    inv[0] = T::one();
    for k in 1..n {
        let mut acc = T::zero();
        for j in 1..=k {
            acc += d[j] * inv[k - j];
        }
        inv[k] = -acc;
    }

    // h[k] += -0.5 * sum_{j=1..=k} (j * d[j]) * inv[k - j].
    for k in 1..=n {
        let mut acc = T::zero();
        for j in 1..=k {
            acc += d[j].scale_f64(j as f64) * inv[k - j];
        }
        h_out[k] -= acc.scale_f64(0.5);
    }
}

/// Compute `u = K_00⁻¹ v_0` (length `n`) and add the loop contribution
/// `h[k] += (k / 2) * [λ^{k-1}](v_0 · u_0 + v_1 · u_1)` for `k = 1..=n`.
#[allow(clippy::too_many_arguments)]
#[inline]
fn compute_loop_pivot<T: HafScalar>(
    a: &[T],
    b: &[T],
    c: &[T],
    inv: &[T],
    v0: &[T],
    v1: &[T],
    n: usize,
    u0: &mut [T],
    u1: &mut [T],
    h_out: &mut [T],
) {
    // First t0 = b*v1 - c*v0, t1 = b*v0 - a*v1, then u = inv * t.
    // Reuse u0, u1 via small stack arrays? For typical n <= 32, tiny Vec/stack:
    let mut t0 = [T::zero(); 64];
    let mut t1 = [T::zero(); 64];
    let (t0_sl, t1_sl, mut t0_vec, mut t1_vec);
    if n <= 64 {
        t0_sl = &mut t0[..n];
        t1_sl = &mut t1[..n];
    } else {
        t0_vec = vec![T::zero(); n];
        t1_vec = vec![T::zero(); n];
        t0_sl = &mut t0_vec[..];
        t1_sl = &mut t1_vec[..];
    }

    for k in 0..n {
        let mut x0 = T::zero();
        let mut x1 = T::zero();
        for j in 0..=k {
            let bj = b[j];
            x0 += bj * v1[k - j];
            x1 += bj * v0[k - j];
        }
        for j in 1..=k {
            x0 -= c[j] * v0[k - j];
            x1 -= a[j] * v1[k - j];
        }
        t0_sl[k] = x0;
        t1_sl[k] = x1;
    }

    for k in 0..n {
        let mut y0 = T::zero();
        let mut y1 = T::zero();
        for j in 0..=k {
            let iv = inv[j];
            y0 += iv * t0_sl[k - j];
            y1 += iv * t1_sl[k - j];
        }
        u0[k] = y0;
        u1[k] = y1;
    }

    // s[k - 1] = [λ^{k-1}](v0 * u0 + v1 * u1), h[k] += 0.5 * k * s[k - 1].
    for (km1, h_slot) in h_out[1..=n].iter_mut().enumerate() {
        let k = km1 + 1;
        let mut s = T::zero();
        for j in 0..=km1 {
            s += v0[j] * u0[km1 - j] + v1[j] * u1[km1 - j];
        }
        *h_slot += s.scale_f64(0.5 * (k as f64));
    }
}

/// `[λⁿ] exp(Q(λ))` given `h[k] = k · Q[k]` for `k = 1..=n`.
#[inline]
fn exp_coeff_n<T: HafScalar>(h: &[T], p: &mut [T], inv_k: &[f64], n: usize) -> T {
    p[0] = T::one();
    for j in 1..=n {
        let mut acc = T::zero();
        for k in 1..=j {
            acc += h[k] * p[j - k];
        }
        p[j] = acc.scale_f64(inv_k[j]);
    }
    p[n]
}

// ---------------------------------------------------------------------------
// Compressed Glynn finite-difference loop hafnian for repeated entries
// ---------------------------------------------------------------------------

fn binom_f64(n: usize, k: usize) -> f64 {
    let k = k.min(n - k);
    let mut num = 1.0f64;
    for i in 0..k {
        num = num * ((n - i) as f64) / ((i + 1) as f64);
    }
    num
}

/// Compressed Glynn algorithm on the `k × k` active submatrix (with optional
/// dummy mode for odd total photon count when `HAS_LOOPS`).
fn compressed_glynn<T: HafScalar, const HAS_LOOPS: bool>(
    matrix: &[T],
    diag: &[T],
    orig_m: usize,
    active: &[usize],
    repeats: &[usize],
    total: usize,
) -> T {
    let odd_pad = HAS_LOOPS && total % 2 == 1;
    let k = active.len() + usize::from(odd_pad);
    let pairs = total.div_ceil(2);

    // Build compact k × k matrix B and length-k vector v.
    let mut b = vec![T::zero(); k * k];
    for (ri, &orig_r) in active.iter().enumerate() {
        for (ci, &orig_c) in active.iter().enumerate() {
            b[ri * k + ci] = matrix[orig_r * orig_m + orig_c];
        }
    }
    let v: Vec<T> = if HAS_LOOPS {
        let mut vv: Vec<T> = active.iter().map(|&i| diag[i]).collect();
        if odd_pad {
            vv.push(T::one());
        }
        vv
    } else {
        Vec::new()
    };

    // Self-pair axes and cross-pairs.
    let mut self_axes: Vec<(usize, usize)> = Vec::new(); // (compact_mode, q_i)
    let mut odd_modes: Vec<usize> = Vec::new();
    for (ci, &orig_i) in active.iter().enumerate() {
        let r = repeats[orig_i];
        if r >= 2 {
            self_axes.push((ci, r / 2));
        }
        if r % 2 == 1 {
            odd_modes.push(ci);
        }
    }
    if odd_pad {
        odd_modes.push(active.len());
    }
    debug_assert_eq!(odd_modes.len() % 2, 0);
    let cross_pairs: Vec<(usize, usize)> = odd_modes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&[a, b]| (a, b))
        .collect();

    // Radices of the mixed-radix Glynn grid.
    let mut radices: Vec<usize> = self_axes.iter().map(|&(_, q)| q + 1).collect();
    radices.extend(std::iter::repeat_n(2usize, cross_pairs.len()));

    let total_grid: usize = radices.iter().product();
    let half_grid = total_grid / 2;
    let inv_k: Vec<f64> = (0..=pairs)
        .map(|j| if j == 0 { 0.0 } else { 1.0 / (j as f64) })
        .collect();

    // Precompute binomial tables for self-pair axes.
    let binoms: Vec<Vec<f64>> = self_axes
        .iter()
        .map(|&(_, q)| {
            (0..=q)
                .map(|c| {
                    let s = if c % 2 == 0 { 1.0 } else { -1.0 };
                    s * binom_f64(q, c)
                })
                .collect()
        })
        .collect();

    let eval_point = |idx: usize| -> T {
        let mut rem = idx;
        let mut weight = 2.0f64; // ×2 from δ ↔ -δ symmetry
        let mut diag_w = vec![0.0f64; k];
        for (ax, &(mode, q)) in self_axes.iter().enumerate() {
            let rad = q + 1;
            let c = rem % rad;
            rem /= rad;
            weight *= binoms[ax][c];
            diag_w[mode] = (2 * q as isize - 4 * c as isize) as f64;
        }
        let mut cross_signs = Vec::with_capacity(cross_pairs.len());
        for _ in 0..cross_pairs.len() {
            let bit = rem & 1;
            rem >>= 1;
            let s = if bit == 0 { 1.0 } else { -1.0 };
            weight *= s;
            cross_signs.push(s);
        }

        // Form C = W * B (O(k²)).
        let mut c_mat = vec![T::zero(); k * k];
        for r in 0..k {
            let dw = diag_w[r];
            if dw != 0.0 {
                for col in 0..k {
                    c_mat[r * k + col] = b[r * k + col].scale_f64(dw);
                }
            }
        }
        for (&(ma, mb), &s) in cross_pairs.iter().zip(&cross_signs) {
            for col in 0..k {
                c_mat[ma * k + col] += b[mb * k + col].scale_f64(s);
                c_mat[mb * k + col] += b[ma * k + col].scale_f64(s);
            }
        }

        // Compute traces tr(C^1..=pairs) using half-powers C^1..=C^{ceil(pairs/2)}.
        let max_pow = pairs.div_ceil(2).max(1);
        let mut pows: Vec<Vec<T>> = Vec::with_capacity(max_pow);
        pows.push(c_mat);
        for p in 1..max_pow {
            let prev = &pows[p - 1];
            let c0 = &pows[0];
            let mut next = vec![T::zero(); k * k];
            for i in 0..k {
                for l in 0..k {
                    let a_il = prev[i * k + l];
                    for j in 0..k {
                        next[i * k + j] += a_il * c0[l * k + j];
                    }
                }
            }
            pows.push(next);
        }

        let mut h = vec![T::zero(); pairs + 1];
        // tr(C^1)
        let mut tr1 = T::zero();
        for i in 0..k {
            tr1 += pows[0][i * k + i];
        }
        h[1] = tr1.scale_f64(0.5);
        for (offset, h_slot) in h[2..=pairs].iter_mut().enumerate() {
            let m_pow = offset + 2;
            let p1 = m_pow / 2;
            let p2 = m_pow - p1;
            let m1 = &pows[p1 - 1];
            let m2 = &pows[p2 - 1];
            let mut tr = T::zero();
            for i in 0..k {
                for j in 0..k {
                    tr += m1[i * k + j] * m2[j * k + i];
                }
            }
            *h_slot = tr.scale_f64(0.5);
        }

        if HAS_LOOPS {
            // u_0 = W * v, u_t = C * u_{t-1}.
            let mut u = vec![T::zero(); k];
            for i in 0..k {
                if diag_w[i] != 0.0 {
                    u[i] = v[i].scale_f64(diag_w[i]);
                }
            }
            for (&(ma, mb), &s) in cross_pairs.iter().zip(&cross_signs) {
                u[ma] += v[mb].scale_f64(s);
                u[mb] += v[ma].scale_f64(s);
            }
            let c0 = &pows[0];
            let mut u_next = vec![T::zero(); k];
            for (tm1, h_slot) in h[1..=pairs].iter_mut().enumerate() {
                let t = tm1 + 1;
                let mut dot = T::zero();
                for i in 0..k {
                    dot += v[i] * u[i];
                }
                *h_slot += dot.scale_f64(0.5 * (t as f64));
                if t < pairs {
                    for i in 0..k {
                        let mut acc = T::zero();
                        for j in 0..k {
                            acc += c0[i * k + j] * u[j];
                        }
                        u_next[i] = acc;
                    }
                    std::mem::swap(&mut u, &mut u_next);
                }
            }
        }

        let mut p = vec![T::zero(); pairs + 1];
        exp_coeff_n(&h, &mut p, &inv_k, pairs).scale_f64(weight)
    };

    let total_sum: T = if half_grid >= 256 {
        (0..half_grid)
            .into_par_iter()
            .map(eval_point)
            .reduce(T::zero, |a, b| a + b)
    } else {
        let mut acc = T::zero();
        for idx in 0..half_grid {
            acc += eval_point(idx);
        }
        acc
    };

    // Scale by 2^{-pairs}.
    total_sum.scale_f64(0.5f64.powi(pairs as i32))
}

// ---------------------------------------------------------------------------
// Torontonian & Gaussian threshold probability (recursive 2×2 Cholesky)
// ---------------------------------------------------------------------------

/// Torontonian of a `2m × 2m` row-major matrix `O` in `xxpp` ordering
/// (mode `i` owns indices `i` and `m + i`), matching The Walrus's `tor(O)`:
///
/// `tor(O) = Σ_{Z ⊆ [m]} (−1)^{m − |Z|} / √det(I − O_{(Z)})`.
pub fn torontonian(matrix: &[C64], n_modes: usize) -> Result<C64, Error> {
    let dim = 2 * n_modes;
    if matrix.len() != dim * dim {
        return Err(Error::InvalidArgument(format!(
            "torontonian expects {dim}×{dim} ({}) entries, got {}",
            dim * dim,
            matrix.len()
        )));
    }
    if n_modes == 0 {
        return Ok(C64::new(1.0, 0.0));
    }

    // Build M = I − O in interleaved mode ordering (0, m, 1, m+1, ...) so each
    // mode is a contiguous 2×2 block.
    let perm = |p: usize| -> usize {
        let mode = p / 2;
        if p.is_multiple_of(2) {
            mode
        } else {
            n_modes + mode
        }
    };
    let mut m_mat = vec![C64::new(0.0, 0.0); dim * dim];
    for i in 0..dim {
        let pi = perm(i);
        for j in 0..dim {
            let pj = perm(j);
            let mut val = -matrix[pi * dim + pj];
            if i == j {
                val += C64::new(1.0, 0.0);
            }
            m_mat[i * dim + j] = val;
        }
    }

    // Fast path 1: real symmetric positive-definite.
    if is_real_symmetric(&m_mat, dim) {
        let real_m: Vec<f64> = m_mat.iter().map(|z| z.re).collect();
        if let Some(val) = tor_real_cholesky(&real_m, n_modes) {
            return Ok(C64::new(val, 0.0));
        }
    }

    // Fast path 2: complex Hermitian positive-definite (all physical Gaussian
    // states have M = σ_Q⁻¹ Hermitian PD).
    if is_hermitian(&m_mat, dim)
        && let Some(val) = tor_hermitian_cholesky(&m_mat, n_modes)
    {
        return Ok(C64::new(val, 0.0));
    }

    // General complex fallback: direct determinant per subset.
    Ok(tor_general(&m_mat, n_modes))
}

fn is_real_symmetric(m: &[C64], dim: usize) -> bool {
    for i in 0..dim {
        for j in i..dim {
            let a = m[i * dim + j];
            let b = m[j * dim + i];
            if a.im.abs() > 1e-12 || b.im.abs() > 1e-12 || (a.re - b.re).abs() > 1e-11 {
                return false;
            }
        }
    }
    true
}

fn is_hermitian(m: &[C64], dim: usize) -> bool {
    for i in 0..dim {
        if m[i * dim + i].im.abs() > 1e-12 {
            return false;
        }
        for j in (i + 1)..dim {
            let a = m[i * dim + j];
            let b = m[j * dim + i];
            if (a - b.conj()).norm() > 1e-11 {
                return false;
            }
        }
    }
    true
}

/// Real symmetric PD recursive `2 × 2` block Cholesky torontonian.
/// Returns `None` if `M` is not positive definite.
fn tor_real_cholesky(m_mat: &[f64], n_modes: usize) -> Option<f64> {
    let dim = 2 * n_modes;
    // Quick PD check on the full matrix (if M is PD, every principal submatrix is PD).
    let mut check_l = vec![0.0f64; dim * dim];
    for r in 0..dim {
        for c in 0..r {
            let mut s = m_mat[r * dim + c];
            for k in 0..c {
                s -= check_l[r * dim + k] * check_l[c * dim + k];
            }
            check_l[r * dim + c] = s / check_l[c * dim + c];
        }
        let mut diag = m_mat[r * dim + r];
        for k in 0..r {
            let v = check_l[r * dim + k];
            diag -= v * v;
        }
        if diag <= 0.0 || !diag.is_finite() {
            return None;
        }
        check_l[r * dim + r] = diag.sqrt();
    }

    let sign0 = if n_modes.is_multiple_of(2) { 1.0 } else { -1.0 };
    let val = if n_modes >= TOR_PAR_MIN_MODES {
        let par_depth = 5.min(n_modes - 2);
        let l_init = vec![0.0f64; dim * dim];
        let active = Vec::with_capacity(n_modes);
        tor_real_par(
            m_mat, dim, n_modes, 0, &l_init, &active, 1.0, sign0, par_depth,
        )
    } else {
        let mut l = vec![0.0f64; dim * dim];
        let mut active = Vec::with_capacity(n_modes);
        tor_real_seq(m_mat, dim, n_modes, 0, &mut l, &mut active, 1.0, sign0)
    };
    Some(val)
}

#[allow(clippy::too_many_arguments)]
fn tor_real_par(
    m_mat: &[f64],
    dim: usize,
    n_modes: usize,
    mode: usize,
    l: &[f64],
    active: &[usize],
    inv_det: f64,
    sign: f64,
    par_depth: usize,
) -> f64 {
    if par_depth == 0 || mode == n_modes {
        let mut l_loc = l.to_vec();
        let mut act_loc = active.to_vec();
        return tor_real_seq(
            m_mat,
            dim,
            n_modes,
            mode,
            &mut l_loc,
            &mut act_loc,
            inv_det,
            sign,
        );
    }

    let mut l_inc = l.to_vec();
    let mut act_inc = active.to_vec();
    let inv_det_inc = extend_cholesky_real(m_mat, dim, &mut l_inc, &act_inc, mode, inv_det);
    act_inc.push(mode);

    let (a, b) = rayon::join(
        || {
            tor_real_par(
                m_mat,
                dim,
                n_modes,
                mode + 1,
                l,
                active,
                inv_det,
                sign,
                par_depth - 1,
            )
        },
        || {
            tor_real_par(
                m_mat,
                dim,
                n_modes,
                mode + 1,
                &l_inc,
                &act_inc,
                inv_det_inc,
                -sign,
                par_depth - 1,
            )
        },
    );
    a + b
}

#[allow(clippy::too_many_arguments)]
fn tor_real_seq(
    m_mat: &[f64],
    dim: usize,
    n_modes: usize,
    mode: usize,
    l: &mut [f64],
    active: &mut Vec<usize>,
    inv_det: f64,
    sign: f64,
) -> f64 {
    if mode == n_modes {
        return sign * inv_det;
    }
    let exc = tor_real_seq(m_mat, dim, n_modes, mode + 1, l, active, inv_det, sign);
    let inv_det_inc = extend_cholesky_real(m_mat, dim, l, active, mode, inv_det);
    active.push(mode);
    let inc = tor_real_seq(m_mat, dim, n_modes, mode + 1, l, active, inv_det_inc, -sign);
    active.pop();
    exc + inc
}

/// Extend real lower-triangular Cholesky factor `l` (stride `dim`) by the two
/// rows of `mode`, returning updated `inv_det = 1 / ∏ L_ii`.
#[inline]
fn extend_cholesky_real(
    m_mat: &[f64],
    dim: usize,
    l: &mut [f64],
    active: &[usize],
    mode: usize,
    inv_det: f64,
) -> f64 {
    let p = 2 * active.len();
    let m0 = 2 * mode;
    let m1 = m0 + 1;
    let row0 = p * dim;
    let row1 = (p + 1) * dim;

    // Row p (first coordinate of `mode`).
    let mut norm0 = 0.0;
    for (ai, &am) in active.iter().enumerate() {
        for off in 0..2 {
            let c = 2 * ai + off;
            let mut s = m_mat[m0 * dim + 2 * am + off];
            let crow = c * dim;
            for k in 0..c {
                s -= l[row0 + k] * l[crow + k];
            }
            let val = s / l[crow + c];
            l[row0 + c] = val;
            norm0 += val * val;
        }
    }
    let d0 = (m_mat[m0 * dim + m0] - norm0).sqrt();
    l[row0 + p] = d0;

    // Row p + 1 (second coordinate of `mode`).
    let mut norm1 = 0.0;
    for (ai, &am) in active.iter().enumerate() {
        for off in 0..2 {
            let c = 2 * ai + off;
            let mut s = m_mat[m1 * dim + 2 * am + off];
            let crow = c * dim;
            for k in 0..c {
                s -= l[row1 + k] * l[crow + k];
            }
            let val = s / l[crow + c];
            l[row1 + c] = val;
            norm1 += val * val;
        }
    }
    // Entry (p + 1, p).
    let mut s = m_mat[m1 * dim + m0];
    for k in 0..p {
        s -= l[row1 + k] * l[row0 + k];
    }
    let l10 = s / d0;
    l[row1 + p] = l10;
    norm1 += l10 * l10;
    let d1 = (m_mat[m1 * dim + m1] - norm1).sqrt();
    l[row1 + p + 1] = d1;

    inv_det / (d0 * d1)
}

/// Complex Hermitian PD recursive `2 × 2` block Cholesky torontonian.
fn tor_hermitian_cholesky(m_mat: &[C64], n_modes: usize) -> Option<f64> {
    let dim = 2 * n_modes;
    // Quick Hermitian PD check on the full matrix.
    let mut check_l = vec![C64::new(0.0, 0.0); dim * dim];
    for r in 0..dim {
        for c in 0..r {
            let mut s = m_mat[r * dim + c];
            for k in 0..c {
                s -= check_l[r * dim + k] * check_l[c * dim + k].conj();
            }
            check_l[r * dim + c] = s / check_l[c * dim + c].re;
        }
        let mut diag = m_mat[r * dim + r].re;
        for k in 0..r {
            diag -= check_l[r * dim + k].norm_sqr();
        }
        if diag <= 0.0 || !diag.is_finite() {
            return None;
        }
        check_l[r * dim + r] = C64::new(diag.sqrt(), 0.0);
    }

    let sign0 = if n_modes.is_multiple_of(2) { 1.0 } else { -1.0 };
    let val = if n_modes >= TOR_PAR_MIN_MODES {
        let par_depth = 5.min(n_modes - 2);
        let l_init = vec![C64::new(0.0, 0.0); dim * dim];
        let active = Vec::with_capacity(n_modes);
        tor_herm_par(
            m_mat, dim, n_modes, 0, &l_init, &active, 1.0, sign0, par_depth,
        )
    } else {
        let mut l = vec![C64::new(0.0, 0.0); dim * dim];
        let mut active = Vec::with_capacity(n_modes);
        tor_herm_seq(m_mat, dim, n_modes, 0, &mut l, &mut active, 1.0, sign0)
    };
    Some(val)
}

#[allow(clippy::too_many_arguments)]
fn tor_herm_par(
    m_mat: &[C64],
    dim: usize,
    n_modes: usize,
    mode: usize,
    l: &[C64],
    active: &[usize],
    inv_det: f64,
    sign: f64,
    par_depth: usize,
) -> f64 {
    if par_depth == 0 || mode == n_modes {
        let mut l_loc = l.to_vec();
        let mut act_loc = active.to_vec();
        return tor_herm_seq(
            m_mat,
            dim,
            n_modes,
            mode,
            &mut l_loc,
            &mut act_loc,
            inv_det,
            sign,
        );
    }
    let mut l_inc = l.to_vec();
    let mut act_inc = active.to_vec();
    let inv_det_inc = extend_cholesky_herm(m_mat, dim, &mut l_inc, &act_inc, mode, inv_det);
    act_inc.push(mode);
    let (a, b) = rayon::join(
        || {
            tor_herm_par(
                m_mat,
                dim,
                n_modes,
                mode + 1,
                l,
                active,
                inv_det,
                sign,
                par_depth - 1,
            )
        },
        || {
            tor_herm_par(
                m_mat,
                dim,
                n_modes,
                mode + 1,
                &l_inc,
                &act_inc,
                inv_det_inc,
                -sign,
                par_depth - 1,
            )
        },
    );
    a + b
}

#[allow(clippy::too_many_arguments)]
fn tor_herm_seq(
    m_mat: &[C64],
    dim: usize,
    n_modes: usize,
    mode: usize,
    l: &mut [C64],
    active: &mut Vec<usize>,
    inv_det: f64,
    sign: f64,
) -> f64 {
    if mode == n_modes {
        return sign * inv_det;
    }
    let exc = tor_herm_seq(m_mat, dim, n_modes, mode + 1, l, active, inv_det, sign);
    let inv_det_inc = extend_cholesky_herm(m_mat, dim, l, active, mode, inv_det);
    active.push(mode);
    let inc = tor_herm_seq(m_mat, dim, n_modes, mode + 1, l, active, inv_det_inc, -sign);
    active.pop();
    exc + inc
}

#[inline]
fn extend_cholesky_herm(
    m_mat: &[C64],
    dim: usize,
    l: &mut [C64],
    active: &[usize],
    mode: usize,
    inv_det: f64,
) -> f64 {
    let p = 2 * active.len();
    let m0 = 2 * mode;
    let m1 = m0 + 1;
    let row0 = p * dim;
    let row1 = (p + 1) * dim;

    let mut norm0 = 0.0;
    for (ai, &am) in active.iter().enumerate() {
        for off in 0..2 {
            let c = 2 * ai + off;
            let mut s = m_mat[m0 * dim + 2 * am + off];
            let crow = c * dim;
            for k in 0..c {
                s -= l[row0 + k] * l[crow + k].conj();
            }
            let val = s / l[crow + c].re;
            l[row0 + c] = val;
            norm0 += val.norm_sqr();
        }
    }
    let d0 = (m_mat[m0 * dim + m0].re - norm0).sqrt();
    l[row0 + p] = C64::new(d0, 0.0);

    let mut norm1 = 0.0;
    for (ai, &am) in active.iter().enumerate() {
        for off in 0..2 {
            let c = 2 * ai + off;
            let mut s = m_mat[m1 * dim + 2 * am + off];
            let crow = c * dim;
            for k in 0..c {
                s -= l[row1 + k] * l[crow + k].conj();
            }
            let val = s / l[crow + c].re;
            l[row1 + c] = val;
            norm1 += val.norm_sqr();
        }
    }
    let mut s = m_mat[m1 * dim + m0];
    for k in 0..p {
        s -= l[row1 + k] * l[row0 + k].conj();
    }
    let l10 = s / d0;
    l[row1 + p] = l10;
    norm1 += l10.norm_sqr();
    let d1 = (m_mat[m1 * dim + m1].re - norm1).sqrt();
    l[row1 + p + 1] = C64::new(d1, 0.0);

    inv_det / (d0 * d1)
}

/// Fallback torontonian for arbitrary complex non-Hermitian / non-PD matrices.
fn tor_general(m_mat: &[C64], n_modes: usize) -> C64 {
    let dim = 2 * n_modes;
    let total = 1usize << n_modes;
    let eval = |mask: usize| -> C64 {
        let k = mask.count_ones() as usize;
        let sign = if (n_modes - k).is_multiple_of(2) {
            1.0
        } else {
            -1.0
        };
        if k == 0 {
            return C64::new(sign, 0.0);
        }
        let mut idxs = Vec::with_capacity(2 * k);
        for mode in 0..n_modes {
            if (mask >> mode) & 1 == 1 {
                idxs.push(2 * mode);
                idxs.push(2 * mode + 1);
            }
        }
        let sub = Mat::from_fn(2 * k, 2 * k, |r, c| m_mat[idxs[r] * dim + idxs[c]]);
        let det = sub.determinant();
        C64::new(sign, 0.0) / det.sqrt()
    };
    if total >= 256 {
        (0..total)
            .into_par_iter()
            .map(eval)
            .reduce(|| C64::new(0.0, 0.0), |a, b| a + b)
    } else {
        (0..total).map(eval).sum()
    }
}

/// Threshold-detector probability `P(clicks)` for a Gaussian state with
/// `m_interleaved = (V + I) / 2` and `mu_interleaved` in interleaved ordering
/// `(x_0, p_0, x_1, p_1, ...)`, where modes have already been ordered so that
/// non-clicking modes occupy `0..n_zero` and clicking modes occupy
/// `n_zero..n_modes`.
///
/// Computes `Σ_{Z ⊆ S} (−1)^{|Z|} P_vac(S̄ ∪ Z)` using a single real recursive
/// `2 × 2` block Cholesky factorization that also tracks `‖L⁻¹ μ‖²`.
pub(crate) fn threshold_probability_real(
    m_mat: &[f64],
    mu: &[f64],
    n_modes: usize,
    n_zero: usize,
) -> f64 {
    let dim = 2 * n_modes;
    let mut l = vec![0.0f64; dim * dim];
    let mut w = vec![0.0f64; dim];
    let mut active = Vec::with_capacity(n_modes);

    let mut inv_det = 1.0;
    let mut quad = 0.0;
    // Factor the non-clicking modes once.
    for mode in 0..n_zero {
        let (id, q) = extend_cholesky_real_displaced(
            m_mat, mu, dim, &mut l, &mut w, &active, mode, inv_det, quad,
        );
        inv_det = id;
        quad = q;
        active.push(mode);
    }

    let n_clicks = n_modes - n_zero;
    if n_clicks == 0 {
        return inv_det * (-0.25 * quad).exp();
    }

    if n_clicks >= TOR_PAR_MIN_MODES {
        let par_depth = 5.min(n_clicks - 2);
        thresh_par(
            m_mat, mu, dim, n_modes, n_zero, &l, &w, &active, inv_det, quad, 1.0, par_depth,
        )
    } else {
        thresh_seq(
            m_mat,
            mu,
            dim,
            n_modes,
            n_zero,
            &mut l,
            &mut w,
            &mut active,
            inv_det,
            quad,
            1.0,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn thresh_par(
    m_mat: &[f64],
    mu: &[f64],
    dim: usize,
    n_modes: usize,
    mode: usize,
    l: &[f64],
    w: &[f64],
    active: &[usize],
    inv_det: f64,
    quad: f64,
    sign: f64,
    par_depth: usize,
) -> f64 {
    if par_depth == 0 || mode == n_modes {
        let mut l_loc = l.to_vec();
        let mut w_loc = w.to_vec();
        let mut act_loc = active.to_vec();
        return thresh_seq(
            m_mat,
            mu,
            dim,
            n_modes,
            mode,
            &mut l_loc,
            &mut w_loc,
            &mut act_loc,
            inv_det,
            quad,
            sign,
        );
    }

    let mut l_inc = l.to_vec();
    let mut w_inc = w.to_vec();
    let mut act_inc = active.to_vec();
    let (id_inc, q_inc) = extend_cholesky_real_displaced(
        m_mat, mu, dim, &mut l_inc, &mut w_inc, &act_inc, mode, inv_det, quad,
    );
    act_inc.push(mode);

    let (a, b) = rayon::join(
        || {
            thresh_par(
                m_mat,
                mu,
                dim,
                n_modes,
                mode + 1,
                l,
                w,
                active,
                inv_det,
                quad,
                sign,
                par_depth - 1,
            )
        },
        || {
            thresh_par(
                m_mat,
                mu,
                dim,
                n_modes,
                mode + 1,
                &l_inc,
                &w_inc,
                &act_inc,
                id_inc,
                q_inc,
                -sign,
                par_depth - 1,
            )
        },
    );
    a + b
}

#[allow(clippy::too_many_arguments)]
fn thresh_seq(
    m_mat: &[f64],
    mu: &[f64],
    dim: usize,
    n_modes: usize,
    mode: usize,
    l: &mut [f64],
    w: &mut [f64],
    active: &mut Vec<usize>,
    inv_det: f64,
    quad: f64,
    sign: f64,
) -> f64 {
    if mode == n_modes {
        return sign * inv_det * (-0.25 * quad).exp();
    }
    let exc = thresh_seq(
        m_mat,
        mu,
        dim,
        n_modes,
        mode + 1,
        l,
        w,
        active,
        inv_det,
        quad,
        sign,
    );
    let (id_inc, q_inc) =
        extend_cholesky_real_displaced(m_mat, mu, dim, l, w, active, mode, inv_det, quad);
    active.push(mode);
    let inc = thresh_seq(
        m_mat,
        mu,
        dim,
        n_modes,
        mode + 1,
        l,
        w,
        active,
        id_inc,
        q_inc,
        -sign,
    );
    active.pop();
    exc + inc
}

/// Evaluate `Σ_{Z ⊆ clicks} (−1)^{|Z|} P_vac(zeros ∪ Z)` starting from a
/// pre-factored Cholesky factor `l` / `w` of `zeros` (used by the incremental
/// threshold GBS chain-rule sampler).
#[allow(clippy::too_many_arguments)]
pub(crate) fn threshold_from_partial_cholesky(
    m_mat: &[f64],
    mu: &[f64],
    dim: usize,
    l: &mut [f64],
    w: &mut [f64],
    active: &mut Vec<usize>,
    clicks: &[usize],
    inv_det: f64,
    quad: f64,
) -> f64 {
    fn rec(
        m_mat: &[f64],
        mu: &[f64],
        dim: usize,
        l: &mut [f64],
        w: &mut [f64],
        active: &mut Vec<usize>,
        clicks: &[usize],
        idx: usize,
        inv_det: f64,
        quad: f64,
        sign: f64,
    ) -> f64 {
        if idx == clicks.len() {
            return sign * inv_det * (-0.25 * quad).exp();
        }
        let exc = rec(
            m_mat,
            mu,
            dim,
            l,
            w,
            active,
            clicks,
            idx + 1,
            inv_det,
            quad,
            sign,
        );
        let mode = clicks[idx];
        let (id_inc, q_inc) =
            extend_cholesky_real_displaced(m_mat, mu, dim, l, w, active, mode, inv_det, quad);
        active.push(mode);
        let inc = rec(
            m_mat,
            mu,
            dim,
            l,
            w,
            active,
            clicks,
            idx + 1,
            id_inc,
            q_inc,
            -sign,
        );
        active.pop();
        exc + inc
    }
    rec(m_mat, mu, dim, l, w, active, clicks, 0, inv_det, quad, 1.0)
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn extend_cholesky_real_displaced(
    m_mat: &[f64],
    mu: &[f64],
    dim: usize,
    l: &mut [f64],
    w: &mut [f64],
    active: &[usize],
    mode: usize,
    inv_det: f64,
    quad: f64,
) -> (f64, f64) {
    let p = 2 * active.len();
    let m0 = 2 * mode;
    let m1 = m0 + 1;
    let row0 = p * dim;
    let row1 = (p + 1) * dim;

    let mut norm0 = 0.0;
    let mut dot_w0 = 0.0;
    for (ai, &am) in active.iter().enumerate() {
        for off in 0..2 {
            let c = 2 * ai + off;
            let mut s = m_mat[m0 * dim + 2 * am + off];
            let crow = c * dim;
            for k in 0..c {
                s -= l[row0 + k] * l[crow + k];
            }
            let val = s / l[crow + c];
            l[row0 + c] = val;
            norm0 += val * val;
            dot_w0 += val * w[c];
        }
    }
    let d0 = (m_mat[m0 * dim + m0] - norm0).max(0.0).sqrt();
    l[row0 + p] = d0;
    let wp0 = (mu[m0] - dot_w0) / d0;
    w[p] = wp0;

    let mut norm1 = 0.0;
    let mut dot_w1 = 0.0;
    for (ai, &am) in active.iter().enumerate() {
        for off in 0..2 {
            let c = 2 * ai + off;
            let mut s = m_mat[m1 * dim + 2 * am + off];
            let crow = c * dim;
            for k in 0..c {
                s -= l[row1 + k] * l[crow + k];
            }
            let val = s / l[crow + c];
            l[row1 + c] = val;
            norm1 += val * val;
            dot_w1 += val * w[c];
        }
    }
    let mut s = m_mat[m1 * dim + m0];
    for k in 0..p {
        s -= l[row1 + k] * l[row0 + k];
    }
    let l10 = s / d0;
    l[row1 + p] = l10;
    norm1 += l10 * l10;
    dot_w1 += l10 * wp0;
    let d1 = (m_mat[m1 * dim + m1] - norm1).max(0.0).sqrt();
    l[row1 + p + 1] = d1;
    let wp1 = (mu[m1] - dot_w1) / d1;
    w[p + 1] = wp1;

    (inv_det / (d0 * d1), quad + wp0 * wp0 + wp1 * wp1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn c(re: f64, im: f64) -> C64 {
        C64::new(re, im)
    }

    /// Brute-force definition of the loop hafnian (sums over all single-pair
    /// matchings + loops). Only for small `n <= 10`.
    fn brute_loop_hafnian(a: &[C64], diag: &[C64], n: usize) -> C64 {
        fn rec(a: &[C64], diag: &[C64], n: usize, used: u32) -> C64 {
            let first = (0..n).find(|&i| (used >> i) & 1 == 0);
            let Some(i) = first else {
                return C64::new(1.0, 0.0);
            };
            let mut sum = diag[i] * rec(a, diag, n, used | (1 << i));
            for j in (i + 1)..n {
                if (used >> j) & 1 == 0 {
                    sum += a[i * n + j] * rec(a, diag, n, used | (1 << i) | (1 << j));
                }
            }
            sum
        }
        rec(a, diag, n, 0)
    }

    fn random_sym(rng: &mut StdRng, n: usize, real: bool) -> (Vec<C64>, Vec<C64>) {
        let mut a = vec![c(0.0, 0.0); n * n];
        for i in 0..n {
            for j in i..n {
                let z = if real {
                    c(rng.random_range(-1.0..1.0), 0.0)
                } else {
                    c(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0))
                };
                a[i * n + j] = z;
                a[j * n + i] = z;
            }
        }
        let d: Vec<C64> = (0..n)
            .map(|_| {
                if real {
                    c(rng.random_range(-1.0..1.0), 0.0)
                } else {
                    c(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0))
                }
            })
            .collect();
        (a, d)
    }

    #[test]
    fn hafnian_matches_brute_force() {
        let mut rng = StdRng::seed_from_u64(42);
        for real in [true, false] {
            for n in [0, 1, 2, 3, 4, 6, 8] {
                let (a, d) = random_sym(&mut rng, n, real);
                let zero_d = vec![c(0.0, 0.0); n];
                let got_h = hafnian(&a, n).unwrap();
                let want_h = brute_loop_hafnian(&a, &zero_d, n);
                assert!(
                    (got_h - want_h).norm() < 1e-10,
                    "hafnian n={n} real={real}: got {got_h}, want {want_h}"
                );

                let got_lh = loop_hafnian(&a, &d, n).unwrap();
                let want_lh = brute_loop_hafnian(&a, &d, n);
                assert!(
                    (got_lh - want_lh).norm() < 1e-10,
                    "loop_hafnian n={n} real={real}: got {got_lh}, want {want_lh}"
                );
            }
        }
    }

    #[test]
    fn hafnian_all_ones_double_factorial() {
        // haf(J_{2n} - I) = (2n - 1)!!
        for (n, want) in [(2, 1.0), (4, 3.0), (6, 15.0), (8, 105.0), (16, 2_027_025.0)] {
            let mut a = vec![c(1.0, 0.0); n * n];
            for i in 0..n {
                a[i * n + i] = c(0.0, 0.0);
            }
            let got = hafnian(&a, n).unwrap();
            assert!(
                (got - c(want, 0.0)).norm() < 1e-6 * want,
                "n={n}: {got} vs {want}"
            );
        }
    }

    #[test]
    fn hafnian_parallel_matches_sequential_large() {
        // n = 16 -> pairs = 8 hits SCHUR_PAR_MIN_PAIRS. Check block-diagonal
        // matrix whose hafnian is the product of two 8×8 hafnians.
        let mut rng = StdRng::seed_from_u64(99);
        let (a1, d1) = random_sym(&mut rng, 8, false);
        let (a2, d2) = random_sym(&mut rng, 8, false);
        let mut big = vec![c(0.0, 0.0); 16 * 16];
        for i in 0..8 {
            for j in 0..8 {
                big[i * 16 + j] = a1[i * 8 + j];
                big[(8 + i) * 16 + (8 + j)] = a2[i * 8 + j];
            }
        }
        let mut big_d = d1.clone();
        big_d.extend_from_slice(&d2);

        let want_h = hafnian(&a1, 8).unwrap() * hafnian(&a2, 8).unwrap();
        let got_h = hafnian(&big, 16).unwrap();
        assert!((got_h - want_h).norm() < 1e-8 * want_h.norm().max(1.0));

        let want_lh = loop_hafnian(&a1, &d1, 8).unwrap() * loop_hafnian(&a2, &d2, 8).unwrap();
        let got_lh = loop_hafnian(&big, &big_d, 16).unwrap();
        assert!((got_lh - want_lh).norm() < 1e-8 * want_lh.norm().max(1.0));
    }

    #[test]
    fn repeated_matches_expanded_and_glynn() {
        let mut rng = StdRng::seed_from_u64(7);
        for real in [true, false] {
            let (b, d) = random_sym(&mut rng, 4, real);
            for repeats in [
                vec![2, 2, 0, 0],
                vec![3, 1, 0, 0],
                vec![4, 0, 0, 0],
                vec![2, 1, 1, 2],
                vec![3, 2, 1, 0],
                vec![3, 2, 0, 0], // odd total (5)
                vec![1, 1, 1, 0], // odd total (3)
                vec![4, 2, 2, 0],
            ] {
                let total: usize = repeats.iter().sum();
                let mut idx_map = Vec::new();
                for (i, &r) in repeats.iter().enumerate() {
                    for _ in 0..r {
                        idx_map.push(i);
                    }
                }
                let mut exp = vec![c(0.0, 0.0); total * total];
                for r in 0..total {
                    for col in 0..total {
                        if r != col {
                            exp[r * total + col] = b[idx_map[r] * 4 + idx_map[col]];
                        }
                    }
                }
                let exp_d: Vec<C64> = idx_map.iter().map(|&i| d[i]).collect();

                // Force compressed Glynn directly AND compare with expanded + dispatch.
                let active: Vec<usize> = (0..4).filter(|&i| repeats[i] > 0).collect();
                if total.is_multiple_of(2) {
                    let want = hafnian(&exp, total).unwrap();
                    let got_disp = hafnian_repeated(&b, &repeats).unwrap();
                    let got_glynn =
                        compressed_glynn::<C64, false>(&b, &[], 4, &active, &repeats, total);
                    assert!(
                        (got_disp - want).norm() < 1e-9,
                        "disp haf repeats={repeats:?}: {got_disp} vs {want}"
                    );
                    assert!(
                        (got_glynn - want).norm() < 1e-9,
                        "glynn haf repeats={repeats:?}: {got_glynn} vs {want}"
                    );
                }

                let want_lh = loop_hafnian(&exp, &exp_d, total).unwrap();
                let got_disp_lh = loop_hafnian_repeated(&b, &d, &repeats).unwrap();
                let got_glynn_lh =
                    compressed_glynn::<C64, true>(&b, &d, 4, &active, &repeats, total);
                assert!(
                    (got_disp_lh - want_lh).norm() < 1e-9,
                    "disp lhaf repeats={repeats:?}: {got_disp_lh} vs {want_lh}"
                );
                assert!(
                    (got_glynn_lh - want_lh).norm() < 1e-9,
                    "glynn lhaf repeats={repeats:?}: {got_glynn_lh} vs {want_lh}"
                );
            }
        }
    }

    #[test]
    fn torontonian_matches_general_formula() {
        let mut rng = StdRng::seed_from_u64(123);
        for n_modes in [1, 2, 3, 4] {
            let dim = 2 * n_modes;
            // Make a random Hermitian PD matrix M = A^H A + 0.5 I, then O = I - M.
            let mut g = Mat::<C64>::zeros(dim, dim);
            for i in 0..dim {
                for j in 0..dim {
                    g[(i, j)] = c(rng.random_range(-0.4..0.4), rng.random_range(-0.4..0.4));
                }
            }
            let m_herm = g.adjoint() * &g;
            let mut o_complex = vec![c(0.0, 0.0); dim * dim];
            let mut o_real = vec![c(0.0, 0.0); dim * dim];
            for i in 0..dim {
                for j in 0..dim {
                    let mij = m_herm[(i, j)] + if i == j { c(0.5, 0.0) } else { c(0.0, 0.0) };
                    let oij = if i == j { c(1.0, 0.0) - mij } else { -mij };
                    o_complex[i * dim + j] = oij;
                    o_real[i * dim + j] = c(oij.re, 0.0);
                }
            }
            // Compare fast Hermitian / real Cholesky path against general LU fallback.
            for o in [&o_complex, &o_real] {
                let got = torontonian(o, n_modes).unwrap();
                // Permute M = I - O to interleaved for tor_general.
                let perm = |p: usize| {
                    if p.is_multiple_of(2) {
                        p / 2
                    } else {
                        n_modes + p / 2
                    }
                };
                let mut m_inter = vec![c(0.0, 0.0); dim * dim];
                for i in 0..dim {
                    for j in 0..dim {
                        let v = -o[perm(i) * dim + perm(j)];
                        m_inter[i * dim + j] = if i == j { v + c(1.0, 0.0) } else { v };
                    }
                }
                let want = tor_general(&m_inter, n_modes);
                assert!(
                    (got - want).norm() < 1e-10,
                    "tor n={n_modes}: got {got}, want {want}"
                );
            }
        }
    }

    #[test]
    fn hafnian_rejects_non_symmetric_matrices() {
        let asym = vec![c(0.0, 0.0), c(1.0, 0.0), c(5.0, 0.0), c(0.0, 0.0)];
        let d = vec![c(1.0, 0.0), c(1.0, 0.0)];
        assert!(hafnian(&asym, 2).is_err());
        assert!(loop_hafnian(&asym, &d, 2).is_err());
        assert!(hafnian_repeated(&asym, &[1, 1]).is_err());
        assert!(loop_hafnian_repeated(&asym, &d, &[1, 1]).is_err());
    }
}
