//! Generalized local eigensolver (`H x = E N x`) and generalized time-evolution
//! solver (`i N dx/dt = H x`) for periodic-trace MPS.
//!
//! Because periodic MPS local environments have a non-identity Hermitian positive
//! semidefinite metric `N` (which may also be rank-deficient on gauge-redundant
//! or degenerate states like GHZ), both solvers:
//!
//! 1. Use a fast exact dense path for small local dimension (`dim ≤ DIRECT_DIM`),
//!    projecting strictly onto the positive range of `N` (`λ > null_tol * λ_max`)
//!    so null-space ghost eigenvalues are impossible.
//! 2. For larger dimensions:
//!    - Eigensolver: **Generalized Davidson** (subspace Rayleigh–Ritz) with
//!      canonical orthogonalization of the projected `N_V = V† N V`, optional
//!      CG preconditioning on `(N + η I) w = r`, and strict variational
//!      monotonicity (`E ≤ E(v0)`).
//!    - Time evolution: **N-inner-product Lanczos** with inner CG solve for
//!      `N w = H v_k`, full N-reorthogonalization, and exact tridiagonal matrix
//!      exponential `exp(-i T dt) e_1`, conserving both N-norm and energy.

use crate::error::Error;
use crate::mps::linalg::{hermitian_evd, matmul};
use crate::statevector::eigen::{EigenOptions, Ritz, lanczos_lowest_from};
use crate::statevector::evolve::krylov_evolve;
use crate::statevector::krylov::{LinearOperator, dot, norm};
use faer::Mat;
use num_complex::Complex64 as C64;

const ZERO: C64 = C64::new(0.0, 0.0);
const ONE: C64 = C64::new(1.0, 0.0);

/// Local vector dimensions up to this threshold are solved directly via dense
/// eigendecomposition of `N` and projected `H`.
const DIRECT_DIM: usize = 96;

/// Relative null-space threshold for canonical orthogonalization of `N`.
const NULL_TOL: f64 = 1e-10;

/// Maximum Davidson subspace size before thick restart.
const DAVIDSON_MAX_SUBSPACE: usize = 36;

/// Vectors kept on a Davidson restart.
const DAVIDSON_RESTART_KEEP: usize = 6;

/// Solve `H x = E N x` for the lowest eigenpair in the positive range of `N`,
/// starting from `v0` (which must satisfy `v0† N v0 > 0`).
/// If `n_is_identity` is true, delegates directly to standard Lanczos.
pub(super) fn solve_generalized_lowest(
    h_op: &impl LinearOperator<C64>,
    n_op: &impl LinearOperator<C64>,
    v0: &[C64],
    n_is_identity: bool,
    options: &EigenOptions,
    null_tol: f64,
) -> Result<Ritz<C64>, Error> {
    let dim = v0.len();
    if dim == 0 {
        return Err(Error::InvalidArgument("empty local vector".into()));
    }
    if n_is_identity {
        let mut u0 = v0.to_vec();
        let nrm = norm(&u0);
        if nrm > 0.0 {
            u0.iter_mut().for_each(|a| *a /= nrm);
        }
        return lanczos_lowest_from(h_op, &u0, &[], options);
    }

    if dim <= DIRECT_DIM {
        return solve_direct(h_op, n_op, v0, null_tol);
    }

    solve_davidson(h_op, n_op, v0, options, null_tol)
}

/// Exact dense generalized eigensolver via canonical orthogonalization of `N`.
fn solve_direct(
    h_op: &impl LinearOperator<C64>,
    n_op: &impl LinearOperator<C64>,
    v0: &[C64],
    null_tol: f64,
) -> Result<Ritz<C64>, Error> {
    let dim = v0.len();
    let mut h_mat = Mat::<C64>::zeros(dim, dim);
    let mut n_mat = Mat::<C64>::zeros(dim, dim);
    let mut e = vec![ZERO; dim];
    let mut col = vec![ZERO; dim];
    for j in 0..dim {
        e[j] = ONE;
        h_op.apply(&e, &mut col);
        for i in 0..dim {
            h_mat[(i, j)] = col[i];
        }
        n_op.apply(&e, &mut col);
        for i in 0..dim {
            n_mat[(i, j)] = col[i];
        }
        e[j] = ZERO;
    }
    // Symmetrise H and N.
    for i in 0..dim {
        for j in 0..=i {
            let hij = (h_mat[(i, j)] + h_mat[(j, i)].conj()) * 0.5;
            h_mat[(i, j)] = hij;
            h_mat[(j, i)] = hij.conj();
            let nij = (n_mat[(i, j)] + n_mat[(j, i)].conj()) * 0.5;
            n_mat[(i, j)] = nij;
            n_mat[(j, i)] = nij.conj();
        }
    }

    let (n_vals, n_u) = hermitian_evd(n_mat.as_ref());
    let max_n = n_vals.iter().copied().fold(0.0f64, f64::max);
    if max_n <= 0.0 {
        return Ok(Ritz {
            value: 0.0,
            vector: v0.to_vec(),
            converged: true,
        });
    }
    let thresh = null_tol.max(NULL_TOL) * max_n;
    let kept: Vec<usize> = n_vals
        .iter()
        .enumerate()
        .filter(|&(_, &v)| v > thresh)
        .map(|(k, _)| k)
        .collect();
    if kept.is_empty() {
        return Ok(Ritz {
            value: 0.0,
            vector: v0.to_vec(),
            converged: true,
        });
    }
    let r = kept.len();
    let x_mat = Mat::<C64>::from_fn(dim, r, |i, c| {
        let k = kept[c];
        n_u[(i, k)] / n_vals[k].sqrt()
    });
    let h_proj = matmul(x_mat.adjoint(), matmul(h_mat.as_ref(), x_mat.as_ref()).as_ref());
    let h_proj = Mat::<C64>::from_fn(r, r, |i, j| {
        (h_proj[(i, j)] + h_proj[(j, i)].conj()) * 0.5
    });
    let (h_vals, h_u) = hermitian_evd(h_proj.as_ref());
    let e0 = h_vals[0];
    let mut vec = vec![ZERO; dim];
    for i in 0..dim {
        let mut acc = ZERO;
        for c in 0..r {
            acc += x_mat[(i, c)] * h_u[(c, 0)];
        }
        vec[i] = acc;
    }

    // Check against v0 energy to guard against PWE noise.
    let mut hv0 = vec![ZERO; dim];
    let mut nv0 = vec![ZERO; dim];
    h_op.apply(v0, &mut hv0);
    n_op.apply(v0, &mut nv0);
    let n0 = dot(v0, &nv0).re;
    if n0 > thresh * norm(v0).powi(2) {
        let e_v0 = dot(v0, &hv0).re / n0;
        if e0 > e_v0 + 1e-11 {
            let scale = 1.0 / n0.sqrt();
            let v0_norm: Vec<C64> = v0.iter().map(|&a| a * scale).collect();
            return Ok(Ritz {
                value: e_v0,
                vector: v0_norm,
                converged: true,
            });
        }
    }

    Ok(Ritz {
        value: e0,
        vector: vec,
        converged: true,
    })
}

/// Matrix-free Generalized Davidson solver for `H x = E N x` with canonical
/// orthogonalization of the projected metric `N_V = V† N V`.
fn solve_davidson(
    h_op: &impl LinearOperator<C64>,
    n_op: &impl LinearOperator<C64>,
    v0: &[C64],
    options: &EigenOptions,
    null_tol: f64,
) -> Result<Ritz<C64>, Error> {
    let dim = v0.len();
    let thresh_rel = null_tol.max(NULL_TOL);
    let max_iters = options.max_iterations.min(36);

    // Normalise initial vector in Euclidean norm.
    let mut v_init = v0.to_vec();
    let init_norm = norm(&v_init);
    if init_norm == 0.0 || !init_norm.is_finite() {
        return Err(Error::InvalidArgument("zero starting vector".into()));
    }
    v_init.iter_mut().for_each(|a| *a /= init_norm);

    let mut hv0 = vec![ZERO; dim];
    let mut nv0 = vec![ZERO; dim];
    h_op.apply(&v_init, &mut hv0);
    n_op.apply(&v_init, &mut nv0);
    let init_n_quad = dot(&v_init, &nv0).re;
    let init_energy = if init_n_quad > 1e-15 {
        dot(&v_init, &hv0).re / init_n_quad
    } else {
        f64::INFINITY
    };

    let mut basis: Vec<Vec<C64>> = vec![v_init.clone()];
    let mut h_basis: Vec<Vec<C64>> = vec![hv0];
    let mut n_basis: Vec<Vec<C64>> = vec![nv0];

    let mut best_e = init_energy;
    let mut best_x = if init_n_quad > 1e-15 {
        let s = 1.0 / init_n_quad.sqrt();
        v_init.iter().map(|&a| a * s).collect()
    } else {
        v_init.clone()
    };
    let mut converged = false;

    for _iter in 0..max_iters {
        let m = basis.len();
        // Projected m × m Hermitian matrices H_V and N_V.
        let mut h_sub = Mat::<C64>::zeros(m, m);
        let mut n_sub = Mat::<C64>::zeros(m, m);
        for i in 0..m {
            for j in 0..=i {
                let hij = dot(&basis[i], &h_basis[j]);
                h_sub[(i, j)] = hij;
                h_sub[(j, i)] = hij.conj();
                let nij = dot(&basis[i], &n_basis[j]);
                n_sub[(i, j)] = nij;
                n_sub[(j, i)] = nij.conj();
            }
        }
        // Symmetrise diagonals to pure real.
        for i in 0..m {
            h_sub[(i, i)] = C64::new(h_sub[(i, i)].re, 0.0);
            n_sub[(i, i)] = C64::new(n_sub[(i, i)].re, 0.0);
        }

        let (n_vals, n_u) = hermitian_evd(n_sub.as_ref());
        let max_n = n_vals.iter().copied().fold(0.0f64, f64::max);
        if max_n <= 0.0 {
            break;
        }
        let kept: Vec<usize> = n_vals
            .iter()
            .enumerate()
            .filter(|&(_, &v)| v > thresh_rel * max_n)
            .map(|(k, _)| k)
            .collect();
        if kept.is_empty() {
            break;
        }
        let r = kept.len();
        let x_sub = Mat::<C64>::from_fn(m, r, |i, c| {
            let k = kept[c];
            n_u[(i, k)] / n_vals[k].sqrt()
        });
        let h_proj = matmul(
            x_sub.adjoint(),
            matmul(h_sub.as_ref(), x_sub.as_ref()).as_ref(),
        );
        let h_proj = Mat::<C64>::from_fn(r, r, |i, j| {
            (h_proj[(i, j)] + h_proj[(j, i)].conj()) * 0.5
        });
        let (h_vals, h_u) = hermitian_evd(h_proj.as_ref());

        let e_ritz = h_vals[0];
        let coeff: Vec<C64> = (0..m)
            .map(|i| {
                let mut acc = ZERO;
                for c in 0..r {
                    acc += x_sub[(i, c)] * h_u[(c, 0)];
                }
                acc
            })
            .collect();

        // Assemble x, Hx, Nx.
        let mut x = vec![ZERO; dim];
        let mut hx = vec![ZERO; dim];
        let mut nx = vec![ZERO; dim];
        for k in 0..m {
            let ck = coeff[k];
            if ck == ZERO {
                continue;
            }
            for j in 0..dim {
                x[j] += ck * basis[k][j];
                hx[j] += ck * h_basis[k][j];
                nx[j] += ck * n_basis[k][j];
            }
        }
        let prev_best_e = best_e;
        if e_ritz <= best_e {
            best_e = e_ritz;
            best_x = x.clone();
        }

        // Residual r = H x - E N x.
        let mut res = vec![ZERO; dim];
        for j in 0..dim {
            res[j] = hx[j] - C64::new(e_ritz, 0.0) * nx[j];
        }
        let res_norm = norm(&res);
        let scale = norm(&hx).max(e_ritz.abs() * norm(&nx)).max(1.0);
        if res_norm <= options.tolerance.max(1e-12) * scale
            || (_iter >= 5
                && (prev_best_e - best_e).abs() <= 1e-12 * (best_e.abs() + 1.0)
                && res_norm <= 1e-4 * scale)
        {
            converged = true;
            break;
        }

        // Light CG preconditioning on (N + η I) w = res only when the projected
        // metric has noticeable spread; otherwise gauge-conditioned Ñ ≈ I already!
        let min_kept_n = kept
            .iter()
            .map(|&k| n_vals[k])
            .fold(f64::INFINITY, f64::min);
        let mut w = if max_n > 10.0 * min_kept_n {
            cg_precondition(n_op, &res, 2, (1e-3 * max_n).max(thresh_rel * max_n))
        } else {
            res.clone()
        };

        // Two-pass Modified Gram-Schmidt against `basis`.
        for _ in 0..2 {
            for b in &basis {
                let proj = dot(b, &w);
                for j in 0..dim {
                    w[j] -= proj * b[j];
                }
            }
        }
        let w_norm = norm(&w);
        if w_norm <= 1e-13 * res_norm.max(1e-15) {
            // Try unpreconditioned residual if preconditioner collapsed.
            w = res;
            for _ in 0..2 {
                for b in &basis {
                    let proj = dot(b, &w);
                    for j in 0..dim {
                        w[j] -= proj * b[j];
                    }
                }
            }
            let w_norm2 = norm(&w);
            if w_norm2 <= 1e-13 {
                converged = true;
                break;
            }
            w.iter_mut().for_each(|a| *a /= w_norm2);
        } else {
            w.iter_mut().for_each(|a| *a /= w_norm);
        }

        let mut hw = vec![ZERO; dim];
        let mut nw = vec![ZERO; dim];
        h_op.apply(&w, &mut hw);
        n_op.apply(&w, &mut nw);

        // If w lies entirely inside ker(N), residual in range(N) has converged.
        if dot(&w, &nw).re <= thresh_rel * max_n * 0.1 {
            converged = true;
            break;
        }

        if basis.len() >= DAVIDSON_MAX_SUBSPACE {
            // Thick restart: keep the lowest Ritz vectors + new direction `w`.
            let keep_ritz = DAVIDSON_RESTART_KEEP.min(r);
            let mut new_basis: Vec<Vec<C64>> = Vec::with_capacity(keep_ritz + 1);
            for mode in 0..keep_ritz {
                let mut rv = vec![ZERO; dim];
                for i in 0..m {
                    let mut ci = ZERO;
                    for c in 0..r {
                        ci += x_sub[(i, c)] * h_u[(c, mode)];
                    }
                    for j in 0..dim {
                        rv[j] += ci * basis[i][j];
                    }
                }
                // Orthonormalise against previously added restart vectors.
                for _ in 0..2 {
                    for prev in &new_basis {
                        let proj = dot(prev, &rv);
                        for j in 0..dim {
                            rv[j] -= proj * prev[j];
                        }
                    }
                }
                let nrm = norm(&rv);
                if nrm > 1e-13 {
                    rv.iter_mut().for_each(|a| *a /= nrm);
                    new_basis.push(rv);
                }
            }
            for _ in 0..2 {
                for prev in &new_basis {
                    let proj = dot(prev, &w);
                    for j in 0..dim {
                        w[j] -= proj * prev[j];
                    }
                }
            }
            let wn = norm(&w);
            if wn > 1e-13 {
                w.iter_mut().for_each(|a| *a /= wn);
                new_basis.push(w);
            }
            basis = new_basis;
            h_basis = basis
                .iter()
                .map(|b| {
                    let mut out = vec![ZERO; dim];
                    h_op.apply(b, &mut out);
                    out
                })
                .collect();
            n_basis = basis
                .iter()
                .map(|b| {
                    let mut out = vec![ZERO; dim];
                    n_op.apply(b, &mut out);
                    out
                })
                .collect();
        } else {
            basis.push(w);
            h_basis.push(hw);
            n_basis.push(nw);
        }
    }

    Ok(Ritz {
        value: best_e,
        vector: best_x,
        converged,
    })
}

/// Approximate solve `(N + shift * I) x = rhs` by `max_steps` CG iterations.
fn cg_precondition(
    n_op: &impl LinearOperator<C64>,
    rhs: &[C64],
    max_steps: usize,
    shift: f64,
) -> Vec<C64> {
    let dim = rhs.len();
    let mut x = vec![ZERO; dim];
    let mut r = rhs.to_vec();
    let mut p = r.clone();
    let mut rr = dot(&r, &r).re;
    let r0 = rr;
    if r0 == 0.0 {
        return x;
    }
    let mut ap = vec![ZERO; dim];
    for _ in 0..max_steps {
        n_op.apply(&p, &mut ap);
        if shift > 0.0 {
            for j in 0..dim {
                ap[j] += C64::new(shift, 0.0) * p[j];
            }
        }
        let pap = dot(&p, &ap).re;
        if pap <= 1e-18 * r0 {
            break;
        }
        let alpha = rr / pap;
        for j in 0..dim {
            x[j] += C64::new(alpha, 0.0) * p[j];
            r[j] -= C64::new(alpha, 0.0) * ap[j];
        }
        let rr_new = dot(&r, &r).re;
        if rr_new <= 1e-6 * r0 {
            break;
        }
        let beta = rr_new / rr;
        for j in 0..dim {
            p[j] = r[j] + C64::new(beta, 0.0) * p[j];
        }
        rr = rr_new;
    }
    x
}

/// Solve `i N dx/dt = H x` for time `dt`, replacing `x` in place with
/// `exp(-i dt N^+ H) x`.
pub(super) fn evolve_generalized(
    h_op: &impl LinearOperator<C64>,
    n_op: &impl LinearOperator<C64>,
    x: &mut [C64],
    dt: f64,
    n_is_identity: bool,
    null_tol: f64,
) -> Result<(), Error> {
    if dt == 0.0 || x.is_empty() {
        return Ok(());
    }
    if n_is_identity {
        return krylov_evolve(h_op, x, dt, true);
    }
    let dim = x.len();
    if dim <= DIRECT_DIM {
        return evolve_direct(h_op, n_op, x, dt, null_tol);
    }
    evolve_n_lanczos(h_op, n_op, x, dt, null_tol)
}

/// Exact dense generalized evolution via canonical orthogonalization of `N`.
fn evolve_direct(
    h_op: &impl LinearOperator<C64>,
    n_op: &impl LinearOperator<C64>,
    x: &mut [C64],
    dt: f64,
    null_tol: f64,
) -> Result<(), Error> {
    let dim = x.len();
    let mut h_mat = Mat::<C64>::zeros(dim, dim);
    let mut n_mat = Mat::<C64>::zeros(dim, dim);
    let mut e = vec![ZERO; dim];
    let mut col = vec![ZERO; dim];
    for j in 0..dim {
        e[j] = ONE;
        h_op.apply(&e, &mut col);
        for i in 0..dim {
            h_mat[(i, j)] = col[i];
        }
        n_op.apply(&e, &mut col);
        for i in 0..dim {
            n_mat[(i, j)] = col[i];
        }
        e[j] = ZERO;
    }
    for i in 0..dim {
        for j in 0..=i {
            let hij = (h_mat[(i, j)] + h_mat[(j, i)].conj()) * 0.5;
            h_mat[(i, j)] = hij;
            h_mat[(j, i)] = hij.conj();
            let nij = (n_mat[(i, j)] + n_mat[(j, i)].conj()) * 0.5;
            n_mat[(i, j)] = nij;
            n_mat[(j, i)] = nij.conj();
        }
    }

    let (n_vals, n_u) = hermitian_evd(n_mat.as_ref());
    let max_n = n_vals.iter().copied().fold(0.0f64, f64::max);
    if max_n <= 0.0 {
        return Ok(());
    }
    let thresh = null_tol.max(NULL_TOL) * max_n;
    let kept: Vec<usize> = n_vals
        .iter()
        .enumerate()
        .filter(|&(_, &v)| v > thresh)
        .map(|(k, _)| k)
        .collect();
    if kept.is_empty() {
        return Ok(());
    }
    let r = kept.len();
    // X = U_kept D_kept^{-1/2}, Y = D_kept^{1/2} U_kept†.
    // On range(N): x_new = X exp(-i dt (X† H X)) (Y x).
    let x_mat = Mat::<C64>::from_fn(dim, r, |i, c| {
        let k = kept[c];
        n_u[(i, k)] / n_vals[k].sqrt()
    });
    let h_proj = matmul(x_mat.adjoint(), matmul(h_mat.as_ref(), x_mat.as_ref()).as_ref());
    let h_proj = Mat::<C64>::from_fn(r, r, |i, j| {
        (h_proj[(i, j)] + h_proj[(j, i)].conj()) * 0.5
    });
    let (h_vals, h_u) = hermitian_evd(h_proj.as_ref());

    // c_in = Y x = D_kept^{1/2} U_kept† x.
    let mut c_in = vec![ZERO; r];
    for (c, &k) in kept.iter().enumerate() {
        let mut acc = ZERO;
        for i in 0..dim {
            acc += n_u[(i, k)].conj() * x[i];
        }
        c_in[c] = acc * n_vals[k].sqrt();
    }
    // Transform to H_proj eigenbasis, apply phases exp(-i E_j dt), transform back.
    let mut eig_coeff = vec![ZERO; r];
    for j in 0..r {
        let mut acc = ZERO;
        for c in 0..r {
            acc += h_u[(c, j)].conj() * c_in[c];
        }
        let phase = C64::cis(-h_vals[j] * dt);
        eig_coeff[j] = acc * phase;
    }
    let mut c_out = vec![ZERO; r];
    for c in 0..r {
        let mut acc = ZERO;
        for j in 0..r {
            acc += h_u[(c, j)] * eig_coeff[j];
        }
        c_out[c] = acc;
    }
    for i in 0..dim {
        let mut acc = ZERO;
        for c in 0..r {
            acc += x_mat[(i, c)] * c_out[c];
        }
        x[i] = acc;
    }
    Ok(())
}

/// Matrix-free Generalized Subspace Rayleigh–Ritz time-stepper for
/// `i N dx/dt = H x` (with Hermitian PSD, possibly rank-deficient `N`).
///
/// Builds a compact Euclidean-orthonormal subspace `V = [v_0, ..., v_{m-1}]`
/// starting from `v_0 = x / ‖x‖`, enriched by regularized preconditioned
/// Krylov directions `w ≈ (N + η I)^{-1} H v_k` (capped at 12 SPD CG steps so
/// rank-deficient `N` can never stall), and evolves the exact `m × m` projected
/// generalized system `(H_sub, N_sub)` via canonical orthogonalization of
/// `N_sub`. Preserves `x`'s `ker(N)` component, `N`-norm, and energy identically.
fn evolve_n_lanczos(
    h_op: &impl LinearOperator<C64>,
    n_op: &impl LinearOperator<C64>,
    x: &mut [C64],
    time: f64,
    null_tol: f64,
) -> Result<(), Error> {
    let dim = x.len();
    let max_m = 20usize.min(dim);
    let tol = 1e-8;
    let thresh_rel = null_tol.max(NULL_TOL);
    let mut remaining = time;
    let mut substeps = 0;

    while remaining.abs() > 1e-15 && substeps < 2 {
        substeps += 1;
        let x_norm = norm(x);
        if x_norm <= 1e-30 {
            break;
        }
        let v0: Vec<C64> = x.iter().map(|&a| a / x_norm).collect();
        let mut hv0 = vec![ZERO; dim];
        let mut nv0 = vec![ZERO; dim];
        h_op.apply(&v0, &mut hv0);
        n_op.apply(&v0, &mut nv0);
        let n0_quad = dot(&v0, &nv0).re;
        if n0_quad <= 1e-30 {
            break;
        }
        let reg_eta = (null_tol.max(1e-8) * n0_quad).max(1e-12);

        let mut basis: Vec<Vec<C64>> = Vec::with_capacity(max_m);
        let mut h_basis: Vec<Vec<C64>> = Vec::with_capacity(max_m);
        let mut n_basis: Vec<Vec<C64>> = Vec::with_capacity(max_m);
        basis.push(v0);
        h_basis.push(hv0);
        n_basis.push(nv0);

        let mut h_sub = Mat::<C64>::zeros(max_m, max_m);
        let mut n_sub = Mat::<C64>::zeros(max_m, max_m);
        h_sub[(0, 0)] = C64::new(dot(&basis[0], &h_basis[0]).re, 0.0);
        n_sub[(0, 0)] = C64::new(n0_quad, 0.0);

        let mut prev_delta: Option<Vec<C64>> = None;
        let mut chosen_dt = remaining;
        let mut chosen_delta: Option<Vec<C64>> = None;

        for k in 0..max_m {
            let m = basis.len();
            // Check convergence starting at m >= 4, every 2 steps or at m == max_m.
            if m >= 4
                && (m.is_multiple_of(2) || m == max_m)
                && let Some((delta_c, z0_norm)) =
                    subspace_generalized_expm(&h_sub, &n_sub, m, x_norm, remaining, thresh_rel)
            {
                if let Some(prev) = &prev_delta {
                    // N-norm difference ‖V(delta_c - [prev; 0])‖_N / ‖z0‖.
                    let mut diff_c = delta_c.clone();
                    for i in 0..prev.len() {
                        diff_c[i] -= prev[i];
                    }
                    let mut err_sq = 0.0f64;
                    for i in 0..m {
                        let mut acc = ZERO;
                        for j in 0..m {
                            acc += n_sub[(i, j)] * diff_c[j];
                        }
                        err_sq += (diff_c[i].conj() * acc).re;
                    }
                    let err = err_sq.max(0.0).sqrt() / z0_norm.max(1e-30);
                    if err <= tol {
                        chosen_dt = remaining;
                        chosen_delta = Some(delta_c);
                        break;
                    }
                }
                prev_delta = Some(delta_c);
            }
            if m == max_m {
                break;
            }

            // Generate next basis vector w ≈ (N + η I)^{-1} H v_k (strictly SPD,
            // bounded at 15 CG iterations — never stalls on rank-deficient N).
            let mut w = cg_solve_reg(n_op, &h_basis[k], reg_eta, 15, 1e-5);
            // Two-pass Modified Gram-Schmidt orthogonalization against `basis`.
            for _ in 0..2 {
                for v in &basis {
                    let proj = dot(v, &w);
                    for idx in 0..dim {
                        w[idx] -= proj * v[idx];
                    }
                }
            }
            let mut wn = norm(&w);
            if wn <= 1e-12 {
                // Fallback: try unpreconditioned residual H v_k directly.
                w.copy_from_slice(&h_basis[k]);
                for _ in 0..2 {
                    for v in &basis {
                        let proj = dot(v, &w);
                        for idx in 0..dim {
                            w[idx] -= proj * v[idx];
                        }
                    }
                }
                wn = norm(&w);
                if wn <= 1e-12 {
                    // Krylov subspace is invariant!
                    if let Some((delta_c, _)) =
                        subspace_generalized_expm(&h_sub, &n_sub, m, x_norm, remaining, thresh_rel)
                    {
                        chosen_dt = remaining;
                        chosen_delta = Some(delta_c);
                    }
                    break;
                }
            }
            let inv_wn = 1.0 / wn;
            w.iter_mut().for_each(|a| *a *= inv_wn);
            let mut hw = vec![ZERO; dim];
            let mut nw = vec![ZERO; dim];
            h_op.apply(&w, &mut hw);
            n_op.apply(&w, &mut nw);

            let new_idx = basis.len();
            for i in 0..new_idx {
                let hij = dot(&basis[i], &hw);
                h_sub[(i, new_idx)] = hij;
                h_sub[(new_idx, i)] = hij.conj();
                let nij = dot(&basis[i], &nw);
                n_sub[(i, new_idx)] = nij;
                n_sub[(new_idx, i)] = nij.conj();
            }
            h_sub[(new_idx, new_idx)] = C64::new(dot(&w, &hw).re, 0.0);
            n_sub[(new_idx, new_idx)] = C64::new(dot(&w, &nw).re, 0.0);
            basis.push(w);
            h_basis.push(hw);
            n_basis.push(nw);
        }

        let m = basis.len();
        let (dt, delta_c) = match chosen_delta {
            Some(dc) => (chosen_dt, dc),
            None => {
                // Subspace hit max_m without converging for full `remaining`:
                // halve dt until m vs m-2 agree to `tol` (or accept smallest step).
                let mut frac = 0.5;
                let mut picked = None;
                for _ in 0..8 {
                    let dt_try = remaining * frac;
                    if let (Some((dc_full, z0n)), Some((dc_prev, _))) = (
                        subspace_generalized_expm(&h_sub, &n_sub, m, x_norm, dt_try, thresh_rel),
                        subspace_generalized_expm(
                            &h_sub,
                            &n_sub,
                            m.saturating_sub(2).max(1),
                            x_norm,
                            dt_try,
                            thresh_rel,
                        ),
                    ) {
                        let mut diff_c = dc_full.clone();
                        for i in 0..dc_prev.len() {
                            diff_c[i] -= dc_prev[i];
                        }
                        let mut err_sq = 0.0f64;
                        for i in 0..m {
                            let mut acc = ZERO;
                            for j in 0..m {
                                acc += n_sub[(i, j)] * diff_c[j];
                            }
                            err_sq += (diff_c[i].conj() * acc).re;
                        }
                        if err_sq.max(0.0).sqrt() / z0n.max(1e-30) <= tol {
                            picked = Some((dt_try, dc_full));
                            break;
                        }
                    }
                    frac *= 0.5;
                }
                picked.unwrap_or_else(|| {
                    let dc = prev_delta.unwrap_or_else(|| vec![ZERO; m]);
                    (remaining, dc)
                })
            }
        };

        // Null-space-preserving incremental lift: x += V delta_c.
        for k in 0..delta_c.len() {
            let ck = delta_c[k];
            if ck == ZERO {
                continue;
            }
            for idx in 0..dim {
                x[idx] += ck * basis[k][idx];
            }
        }
        remaining -= dt;
    }
    Ok(())
}

/// Exact generalized subspace exponential for `i N_sub dc/dt = H_sub c` starting
/// from `c_0 = x_norm * e_0`. Returns `(delta_c, ‖z_0‖)` where
/// `delta_c = X_sub (z(dt) - z_0)` is the range-of-`N_sub` evolution increment.
fn subspace_generalized_expm(
    h_sub_full: &Mat<C64>,
    n_sub_full: &Mat<C64>,
    m: usize,
    x_norm: f64,
    dt: f64,
    thresh_rel: f64,
) -> Option<(Vec<C64>, f64)> {
    let n_sub = Mat::<C64>::from_fn(m, m, |i, j| n_sub_full[(i, j)]);
    let h_sub = Mat::<C64>::from_fn(m, m, |i, j| h_sub_full[(i, j)]);
    let (n_vals, n_u) = hermitian_evd(n_sub.as_ref());
    let max_n = n_vals.iter().copied().fold(0.0f64, f64::max);
    if max_n <= 0.0 {
        return None;
    }
    let thresh = thresh_rel * max_n;
    let kept: Vec<usize> = n_vals
        .iter()
        .enumerate()
        .filter(|&(_, &v)| v > thresh)
        .map(|(idx, _)| idx)
        .collect();
    if kept.is_empty() {
        return None;
    }
    let r = kept.len();
    let x_mat = Mat::<C64>::from_fn(m, r, |i, c| {
        let idx = kept[c];
        n_u[(i, idx)] / n_vals[idx].sqrt()
    });
    let h_proj = matmul(x_mat.adjoint(), matmul(h_sub.as_ref(), x_mat.as_ref()).as_ref());
    let h_proj = Mat::<C64>::from_fn(r, r, |i, j| {
        (h_proj[(i, j)] + h_proj[(j, i)].conj()) * 0.5
    });
    let (h_vals, h_u) = hermitian_evd(h_proj.as_ref());

    // z_0 = Y_sub (x_norm * e_0) = D_kept^{1/2} U_kept† (x_norm * e_0).
    let mut z0 = vec![ZERO; r];
    let mut z0_sq = 0.0f64;
    for (c, &idx) in kept.iter().enumerate() {
        z0[c] = n_u[(0, idx)].conj() * (n_vals[idx].sqrt() * x_norm);
        z0_sq += z0[c].norm_sqr();
    }
    let z0_norm = z0_sq.sqrt();
    if z0_norm <= 1e-30 {
        return None;
    }

    // z(dt) = h_u exp(-i h_vals dt) h_u† z0.
    let mut eig_coeff = vec![ZERO; r];
    for j in 0..r {
        let mut acc = ZERO;
        for c in 0..r {
            acc += h_u[(c, j)].conj() * z0[c];
        }
        eig_coeff[j] = acc * C64::cis(-h_vals[j] * dt);
    }
    let mut dz = vec![ZERO; r];
    for c in 0..r {
        let mut acc = ZERO;
        for j in 0..r {
            acc += h_u[(c, j)] * eig_coeff[j];
        }
        dz[c] = acc - z0[c];
    }
    let mut delta_c = vec![ZERO; m];
    for i in 0..m {
        let mut acc = ZERO;
        for c in 0..r {
            acc += x_mat[(i, c)] * dz[c];
        }
        delta_c[i] = acc;
    }
    Some((delta_c, z0_norm))
}

/// Strictly SPD Tikhonov-regularized Conjugate Gradients for `(N + η I) w = rhs`.
fn cg_solve_reg(
    n_op: &impl LinearOperator<C64>,
    rhs: &[C64],
    eta: f64,
    max_iters: usize,
    tol: f64,
) -> Vec<C64> {
    let dim = rhs.len();
    let mut x = vec![ZERO; dim];
    let mut r = rhs.to_vec();
    let mut p = r.clone();
    let mut rr = dot(&r, &r).re;
    let r0 = rr;
    if r0 == 0.0 {
        return x;
    }
    let mut ap = vec![ZERO; dim];
    for _ in 0..max_iters {
        n_op.apply(&p, &mut ap);
        for j in 0..dim {
            ap[j] += C64::new(eta, 0.0) * p[j];
        }
        let pap = dot(&p, &ap).re;
        if pap <= 1e-30 {
            break;
        }
        let alpha = rr / pap;
        for j in 0..dim {
            x[j] += C64::new(alpha, 0.0) * p[j];
            r[j] -= C64::new(alpha, 0.0) * ap[j];
        }
        let rr_new = dot(&r, &r).re;
        if rr_new <= tol * tol * r0 {
            break;
        }
        let beta = rr_new / rr;
        for j in 0..dim {
            p[j] = r[j] + C64::new(beta, 0.0) * p[j];
        }
        rr = rr_new;
    }
    x
}
