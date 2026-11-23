//! Exact two-qubit Cartan (KAK / Weyl chamber) decomposition and optimal synthesis.
//!
//! Any `U ∈ U(4)` decomposes losslessly as:
//! `U = e^{iγ} (B₁ ⊗ A₁) · exp(i(cₓ XX + cᵧ YY + c_z ZZ)) · (B₀ ⊗ A₀)`
//! (with ketrs's qubit ordering: basis index `2·b(q₁) + b(q₀)`, so `A` acts on
//! `qubits[0]` and `B` acts on `qubits[1]`), with canonical Weyl coordinates
//! `π/4 ≥ cₓ ≥ cᵧ ≥ |c_z| ≥ 0` (and `c_z ≥ 0` when `cₓ = π/4`).
//!
//! From `(cₓ, cᵧ, c_z)` we synthesize the exact minimum-entangling-gate circuit:
//! - 0 two-qubit gates when `cₓ = 0` (product `B ⊗ A`),
//! - 1 `CZ` or `CP` for diagonal unitaries (in `Basis::Native`),
//! - 1 `CX` when `(cₓ, cᵧ, c_z) = (π/4, 0, 0)`,
//! - 1 `Swap` when `(cₓ, cᵧ, c_z) = (π/4, π/4, π/4)` (in `Basis::Native`),
//! - 1 `Rzz` when `cᵧ = c_z = 0` (in `Basis::Native`),
//! - 2 `CX`s when `c_z = 0`,
//! - 3 `CX`s in the general case (Vatan & Williams, Phys. Rev. A 69, 032315).

use crate::circuit::Operation;
use crate::gate::{Gate, Matrix2, Matrix4};
use crate::observable::{Pauli, PauliString};
use crate::optimize::Basis;
use crate::optimize::one_qubit::{
    I2, mat2_is_real, mat2_mul, phase_is_real, synthesize_1q, wrap_pi,
};
use num_complex::Complex64 as C64;
use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_2, FRAC_PI_4, PI};

const ZERO: C64 = C64::new(0.0, 0.0);
const ONE: C64 = C64::new(1.0, 0.0);

/// Magic basis matrix `Q` such that `Q† (SU(2) ⊗ SU(2)) Q = SO(4)`.
fn magic_q() -> Matrix4 {
    let s = C64::new(FRAC_1_SQRT_2, 0.0);
    let si = C64::new(0.0, FRAC_1_SQRT_2);
    [
        [s, ZERO, ZERO, si],
        [ZERO, si, s, ZERO],
        [ZERO, si, -s, ZERO],
        [s, ZERO, ZERO, -si],
    ]
}

fn mat4_mul(a: &Matrix4, b: &Matrix4) -> Matrix4 {
    std::array::from_fn(|r| std::array::from_fn(|c| (0..4).map(|k| a[r][k] * b[k][c]).sum()))
}

fn mat4_adj(m: &Matrix4) -> Matrix4 {
    std::array::from_fn(|r| std::array::from_fn(|c| m[c][r].conj()))
}

fn mat4_transpose(m: &Matrix4) -> Matrix4 {
    std::array::from_fn(|r| std::array::from_fn(|c| m[c][r]))
}

pub(crate) fn mat4_is_real(m: &Matrix4, tol: f64) -> bool {
    m.iter().flatten().all(|c| c.im.abs() <= tol)
}

/// Kronecker product `b ⊗ a`, matching ketrs's `2·b(q₁) + b(q₀)` indexing
/// (`a` on `qubits[0]`, `b` on `qubits[1]`).
pub(crate) fn kron2(b: &Matrix2, a: &Matrix2) -> Matrix4 {
    std::array::from_fn(|r| std::array::from_fn(|c| b[r >> 1][c >> 1] * a[r & 1][c & 1]))
}

/// Determinant of a 4×4 complex matrix via cofactor expansion.
fn mat4_det(m: &Matrix4) -> C64 {
    let det3 = |rows: [usize; 3], cols: [usize; 3]| -> C64 {
        let [r0, r1, r2] = rows;
        let [c0, c1, c2] = cols;
        m[r0][c0] * (m[r1][c1] * m[r2][c2] - m[r1][c2] * m[r2][c1])
            - m[r0][c1] * (m[r1][c0] * m[r2][c2] - m[r1][c2] * m[r2][c0])
            + m[r0][c2] * (m[r1][c0] * m[r2][c1] - m[r1][c1] * m[r2][c0])
    };
    m[0][0] * det3([1, 2, 3], [1, 2, 3]) - m[0][1] * det3([1, 2, 3], [0, 2, 3])
        + m[0][2] * det3([1, 2, 3], [0, 1, 3])
        - m[0][3] * det3([1, 2, 3], [0, 1, 2])
}

/// Determinant of a 4×4 real matrix.
fn mat4_det_real(m: &[[f64; 4]; 4]) -> f64 {
    let det3 = |rows: [usize; 3], cols: [usize; 3]| -> f64 {
        let [r0, r1, r2] = rows;
        let [c0, c1, c2] = cols;
        m[r0][c0] * (m[r1][c1] * m[r2][c2] - m[r1][c2] * m[r2][c1])
            - m[r0][c1] * (m[r1][c0] * m[r2][c2] - m[r1][c2] * m[r2][c0])
            + m[r0][c2] * (m[r1][c0] * m[r2][c1] - m[r1][c1] * m[r2][c0])
    };
    m[0][0] * det3([1, 2, 3], [1, 2, 3]) - m[0][1] * det3([1, 2, 3], [0, 2, 3])
        + m[0][2] * det3([1, 2, 3], [0, 1, 3])
        - m[0][3] * det3([1, 2, 3], [0, 1, 2])
}

/// Jacobi eigendecomposition of an `n × n` real symmetric matrix (`n <= 4`).
/// Returns `(eigenvalues, eigenvectors)` where column `k` of `eigenvectors`
/// is the unit eigenvector for `eigenvalues[k]`.
fn jacobi_sym(a: &mut [[f64; 4]; 4], n: usize) -> ([f64; 4], [[f64; 4]; 4]) {
    let mut v = [[0.0; 4]; 4];
    for (i, row) in v.iter_mut().enumerate().take(n) {
        row[i] = 1.0;
    }
    for _ in 0..80 {
        // Largest off-diagonal element.
        let mut p = 0;
        let mut q = 1;
        let mut max_off = 0.0;
        for (i, row) in a.iter().enumerate().take(n) {
            for (j, &val) in row.iter().enumerate().take(n).skip(i + 1) {
                if val.abs() > max_off {
                    max_off = val.abs();
                    p = i;
                    q = j;
                }
            }
        }
        if max_off < 1e-15 {
            break;
        }
        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];
        let tau = (aqq - app) / (2.0 * apq);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;
        let h = s / (1.0 + c);

        a[p][p] = app - t * apq;
        a[q][q] = aqq + t * apq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;

        for r in 0..n {
            if r != p && r != q {
                let arp = a[r][p];
                let arq = a[r][q];
                a[r][p] = arp - s * (arq + h * arp);
                a[p][r] = a[r][p];
                a[r][q] = arq + s * (arp - h * arq);
                a[q][r] = a[r][q];
            }
        }
        for row in v.iter_mut().take(n) {
            let vrp = row[p];
            let vrq = row[q];
            row[p] = vrp - s * (vrq + h * vrp);
            row[q] = vrq + s * (vrp - h * vrq);
        }
    }
    let mut vals = [0.0; 4];
    for (i, val) in vals.iter_mut().enumerate().take(n) {
        *val = a[i][i];
    }
    (vals, v)
}

/// Simultaneously diagonalize commuting real symmetric 4×4 matrices `re` and
/// `im` (from `M = U_B^T U_B = re + i·im`), returning an orthogonal matrix
/// `O ∈ SO(4)` whose columns are simultaneous eigenvectors.
fn simultaneous_real_evd(re: &[[f64; 4]; 4], im: &[[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let mut re_work = *re;
    let (vals, mut v) = jacobi_sym(&mut re_work, 4);

    // Sort eigenpairs by eigenvalue so degenerate subspaces are contiguous.
    let mut order = [0, 1, 2, 3];
    order.sort_by(|&i, &j| vals[i].total_cmp(&vals[j]));
    let sorted_vals = order.map(|i| vals[i]);
    let mut basis = [[0.0; 4]; 4];
    for r in 0..4 {
        for (c, &idx) in order.iter().enumerate() {
            basis[r][c] = v[r][idx];
        }
    }
    v = basis;

    // Refine each degenerate eigenspace of `re` by diagonalizing projected `im`.
    let deg_tol = 1e-8;
    let mut start = 0;
    while start < 4 {
        let mut end = start + 1;
        while end < 4 && (sorted_vals[end] - sorted_vals[start]).abs() < deg_tol {
            end += 1;
        }
        let dim = end - start;
        if dim > 1 {
            // Project `im` into the subspace spanned by columns `start..end`.
            let mut proj = [[0.0; 4]; 4];
            for i in 0..dim {
                for j in i..dim {
                    let mut sum = 0.0;
                    for (r, im_row) in im.iter().enumerate() {
                        let vi = v[r][start + i];
                        for (c, &im_rc) in im_row.iter().enumerate() {
                            sum += vi * im_rc * v[c][start + j];
                        }
                    }
                    proj[i][j] = sum;
                    proj[j][i] = sum;
                }
            }
            let (_, sub_v) = jacobi_sym(&mut proj, dim);
            let old = v;
            for r in 0..4 {
                for j in 0..dim {
                    let mut sum = 0.0;
                    for i in 0..dim {
                        sum += old[r][start + i] * sub_v[i][j];
                    }
                    v[r][start + j] = sum;
                }
            }
        }
        start = end;
    }

    // Ensure det(v) = +1 so v ∈ SO(4).
    if mat4_det_real(&v) < 0.0 {
        for row in &mut v {
            row[0] = -row[0];
        }
    }
    v
}

/// Factor `K = B ⊗ A` (where `K ∈ SU(2) ⊗ SU(2)`) into `(A, B)` with
/// `A, B ∈ SU(2)` (up to global sign ±1, returned as `sign`).
fn factor_kron(k: &Matrix4) -> (Matrix2, Matrix2, f64) {
    // Pick the entry with largest magnitude to avoid dividing by zero.
    let mut best_r = 0;
    let mut best_c = 0;
    let mut best_norm = -1.0;
    for (r, row) in k.iter().enumerate() {
        for (c, &v) in row.iter().enumerate() {
            let n = v.norm_sqr();
            if n > best_norm {
                best_norm = n;
                best_r = r;
                best_c = c;
            }
        }
    }
    let br = best_r >> 1;
    let ar = best_r & 1;
    let bc = best_c >> 1;
    let ac = best_c & 1;

    // Extract unnormalized B from slice (ar, ac) and A from slice (br, bc).
    let mut b: Matrix2 =
        std::array::from_fn(|r| std::array::from_fn(|c| k[(r << 1) | ar][(c << 1) | ac]));
    let mut a: Matrix2 =
        std::array::from_fn(|r| std::array::from_fn(|c| k[(br << 1) | r][(bc << 1) | c]));

    // Normalize both to SU(2) (det = 1).
    let norm_su2 = |m: &mut Matrix2| {
        let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
        let s = det.sqrt();
        for row in m.iter_mut() {
            for v in row.iter_mut() {
                *v /= s;
            }
        }
    };
    norm_su2(&mut a);
    norm_su2(&mut b);

    // Fix overall phase: kron2(&b, &a) = ±k (since both are in SU(2) ⊗ SU(2)).
    let prod = kron2(&b, &a);
    let ratio = k[best_r][best_c] / prod[best_r][best_c];
    let sign = if ratio.re < 0.0 { -1.0 } else { 1.0 };
    (a, b, sign)
}

/// Convert `O ∈ SO(4)` to `(A, B, sign)` with `Q O Q† = sign · (B ⊗ A)`.
fn so4_to_su2_pair(o: &[[f64; 4]; 4]) -> (Matrix2, Matrix2, f64) {
    let q = magic_q();
    let oc: Matrix4 = std::array::from_fn(|r| std::array::from_fn(|c| C64::new(o[r][c], 0.0)));
    let k = mat4_mul(&mat4_mul(&q, &oc), &mat4_adj(&q));
    factor_kron(&k)
}

/// Canonical KAK decomposition of `U ∈ U(4)`:
/// `U = e^{i·global_phase} (b1 ⊗ a1) · exp(i(cx XX + cy YY + cz ZZ)) · (b0 ⊗ a0)`
/// where `a0, a1` act on `qubits[0]` and `b0, b1` act on `qubits[1]`.
pub(crate) struct KakDecomposition {
    pub a0: Matrix2,
    pub b0: Matrix2,
    pub a1: Matrix2,
    pub b1: Matrix2,
    pub cx: f64,
    pub cy: f64,
    pub cz: f64,
    pub global_phase: f64,
}

/// Decompose any `U ∈ U(4)` into canonical Weyl chamber form.
pub(crate) fn kak_decompose(u: &Matrix4) -> KakDecomposition {
    let det = mat4_det(u);
    let mut global_phase = det.arg() / 4.0;
    let scale = C64::cis(-global_phase);
    let u_su: Matrix4 = std::array::from_fn(|r| std::array::from_fn(|c| u[r][c] * scale));

    let q = magic_q();
    let q_adj = mat4_adj(&q);
    let u_b = mat4_mul(&mat4_mul(&q_adj, &u_su), &q);

    // M = U_B^T U_B is complex symmetric & unitary; Re(M) and Im(M) commute.
    let m = mat4_mul(&mat4_transpose(&u_b), &u_b);
    let re: [[f64; 4]; 4] = std::array::from_fn(|r| std::array::from_fn(|c| m[r][c].re));
    let im: [[f64; 4]; 4] = std::array::from_fn(|r| std::array::from_fn(|c| m[r][c].im));
    let o_r = simultaneous_real_evd(&re, &im);

    // Eigenvalues λ_k = (O_R^T M O_R)_{kk} = e^{2i φ_k}.
    // Critically: degenerate eigenvalues of M MUST receive the SAME square root
    // so that O_L = U_B O_R D† is guaranteed purely real!
    let mut lambdas = [ONE; 4];
    for k in 0..4 {
        let mut acc = ZERO;
        for r in 0..4 {
            for c in 0..4 {
                acc += o_r[r][k] * m[r][c] * o_r[c][k];
            }
        }
        lambdas[k] = acc;
    }

    let mut phis = [0.0; 4];
    let deg_tol = 1e-7;
    for k in 0..4 {
        if let Some(prev) = (0..k).find(|&j| (lambdas[k] - lambdas[j]).norm() < deg_tol) {
            phis[k] = phis[prev];
        } else {
            phis[k] = lambdas[k].arg() / 2.0;
        }
    }

    // Compute O_L = U_B O_R D† (guaranteed real orthogonal!).
    let mut o_l = [[0.0; 4]; 4];
    for r in 0..4 {
        for k in 0..4 {
            let mut acc = ZERO;
            for c in 0..4 {
                acc += u_b[r][c] * o_r[c][k];
            }
            let entry = acc * C64::cis(-phis[k]);
            o_l[r][k] = entry.re;
        }
    }

    // Ensure det(O_L) = +1 (if -1, negate col 0 of O_L and shift φ_0 by π).
    if mat4_det_real(&o_l) < 0.0 {
        for row in &mut o_l {
            row[0] = -row[0];
        }
        phis[0] += PI;
    }

    // Now U_B = O_L · diag(e^{i φ_k}) · O_R^T with O_L, O_R ∈ SO(4) and
    // det(D) = e^{i Σ φ_k} = +1, so Σ φ_k = 2π·n. Shift φ_0 so Σ φ_k = 0.
    let mut sum_phi: f64 = phis.iter().sum();
    // Round sum_phi / (2π) to nearest integer.
    let wraps = (sum_phi / (2.0 * PI)).round();
    phis[0] -= wraps * 2.0 * PI;
    sum_phi = phis.iter().sum();
    let _ = sum_phi;

    // Extract raw interaction coordinates (cx, cy, cz):
    // φ_0 =  cx - cy + cz
    // φ_1 =  cx + cy - cz
    // φ_2 = -cx - cy - cz
    // φ_3 = -cx + cy + cz
    let mut cx = (phis[0] + phis[1]) * 0.5;
    let mut cy = (phis[1] + phis[3]) * 0.5;
    let mut cz = (phis[0] + phis[3]) * 0.5;

    let (mut a1, mut b1, s1) = so4_to_su2_pair(&o_l);
    let o_rt = std::array::from_fn(|r| std::array::from_fn(|c| o_r[c][r]));
    let (mut a0, mut b0, s0) = so4_to_su2_pair(&o_rt);
    if s0 * s1 < 0.0 {
        global_phase += PI;
    }

    // Canonicalize (cx, cy, cz) into the Weyl chamber:
    //   π/4 ≥ cx ≥ cy ≥ |cz| ≥ 0, and if cx == π/4 then cz ≥ 0.
    // All local Clifford adjustments preserve
    //   e^{iγ} (b1 ⊗ a1) exp(i(cx XX + cy YY + cz ZZ)) (b0 ⊗ a0) == U!
    canonicalize_weyl(
        &mut cx,
        &mut cy,
        &mut cz,
        &mut a0,
        &mut b0,
        &mut a1,
        &mut b1,
        &mut global_phase,
    );

    KakDecomposition {
        a0,
        b0,
        a1,
        b1,
        cx,
        cy,
        cz,
        global_phase: wrap_pi(global_phase),
    }
}

/// Shift `(cx, cy, cz)` into the canonical Weyl chamber by absorbing local
/// Clifford gates into `(a0, b0, a1, b1)` and `global_phase`.
#[allow(clippy::too_many_arguments)]
fn canonicalize_weyl(
    cx: &mut f64,
    cy: &mut f64,
    cz: &mut f64,
    a0: &mut Matrix2,
    b0: &mut Matrix2,
    a1: &mut Matrix2,
    b1: &mut Matrix2,
    phase: &mut f64,
) {
    // Helper: shift coordinate by k * π/2 using exp(i π/2 PP) = i (P ⊗ P).
    // exp(i(c + k π/2) PP) = i^k (P^k ⊗ P^k) exp(i c PP).
    let shift_axis =
        |c: &mut f64, p: &Matrix2, a1: &mut Matrix2, b1: &mut Matrix2, phase: &mut f64| {
            // Reduce c into (-π/4, π/4] by subtracting multiples of π/2:
            // exp(i c PP) = exp(i (c - k π/2) PP) · exp(i k π/2 PP) = i^k (P^k ⊗ P^k) exp(i c' PP).
            let k = (*c / FRAC_PI_2).round() as i64;
            if k != 0 {
                *c -= (k as f64) * FRAC_PI_2;
                let k_mod4 = k.rem_euclid(4);
                *phase += (k_mod4 as f64) * FRAC_PI_2;
                if k_mod4 % 2 == 1 {
                    *a1 = mat2_mul(a1, p);
                    *b1 = mat2_mul(b1, p);
                }
            }
        };

    let mx = Gate::X.matrix();
    let my = Gate::Y.matrix();
    let mz = Gate::Z.matrix();
    let mh = Gate::H.matrix(); // H X H = Z, H Z H = X, H Y H = -Y
    let ms = Gate::S.matrix(); // S X S† = Y, S Y S† = -X, S Z S† = Z
    let msdg = Gate::Sdg.matrix();

    // Step 1: reduce each coordinate into [-π/4, π/4].
    shift_axis(cx, &mx, a1, b1, phase);
    shift_axis(cy, &my, a1, b1, phase);
    shift_axis(cz, &mz, a1, b1, phase);

    // Helper: conjugate by local (u ⊗ u) on left and (u† ⊗ u†) on right:
    // (u† ⊗ u†) · [u⊗u · E · u†⊗u†] · (u ⊗ u) = E.
    // So replacing E with E' = (u⊗u) E (u†⊗u†) updates:
    //   a1 <- a1 u†,  b1 <- b1 u†,  a0 <- u a0,  b0 <- u b0.
    let conj_both = |u: &Matrix2,
                     u_adj: &Matrix2,
                     a0: &mut Matrix2,
                     b0: &mut Matrix2,
                     a1: &mut Matrix2,
                     b1: &mut Matrix2| {
        *a1 = mat2_mul(a1, u_adj);
        *b1 = mat2_mul(b1, u_adj);
        *a0 = mat2_mul(u, a0);
        *b0 = mat2_mul(u, b0);
    };

    // Helper: negate two coordinates using (P ⊗ I):
    // (X ⊗ I) (cx XX + cy YY + cz ZZ) (X ⊗ I) = cx XX - cy YY - cz ZZ!
    let flip_yz = |cy: &mut f64, cz: &mut f64, a0: &mut Matrix2, a1: &mut Matrix2| {
        *cy = -*cy;
        *cz = -*cz;
        *a1 = mat2_mul(a1, &mx);
        *a0 = mat2_mul(&mx, a0);
    };
    let flip_xz = |cx: &mut f64, cz: &mut f64, a0: &mut Matrix2, a1: &mut Matrix2| {
        *cx = -*cx;
        *cz = -*cz;
        *a1 = mat2_mul(a1, &my);
        *a0 = mat2_mul(&my, a0);
    };
    let flip_xy = |cx: &mut f64, cy: &mut f64, a0: &mut Matrix2, a1: &mut Matrix2| {
        *cx = -*cx;
        *cy = -*cy;
        *a1 = mat2_mul(a1, &mz);
        *a0 = mat2_mul(&mz, a0);
    };

    // Step 2: sort |cx| >= |cy| >= |cz| via local Cliffords.
    // (S ⊗ S) swaps X <-> Y (with (-X)(-X)=+XX, YY=+YY), preserves Z!
    // (H ⊗ H) swaps X <-> Z, preserves Y (with (-Y)(-Y)=+YY)!
    if cx.abs() < cy.abs() {
        std::mem::swap(cx, cy);
        conj_both(&ms, &msdg, a0, b0, a1, b1);
    }
    if cx.abs() < cz.abs() {
        std::mem::swap(cx, cz);
        conj_both(&mh, &mh, a0, b0, a1, b1);
    }
    if cy.abs() < cz.abs() {
        // Swap Y <-> Z via Rx(π/2): Rx(π/2) Y Rx(-π/2) = Z, Rx(π/2) Z Rx(-π/2) = -Y.
        std::mem::swap(cy, cz);
        let rx_p = Gate::Rx(FRAC_PI_2).matrix();
        let rx_m = Gate::Rx(-FRAC_PI_2).matrix();
        conj_both(&rx_p, &rx_m, a0, b0, a1, b1);
    }

    // Step 3: make cx >= 0 and cy >= 0 by flipping pairs of signs.
    if *cx < 0.0 && *cy < 0.0 {
        flip_xy(cx, cy, a0, a1);
    } else if *cx < 0.0 {
        flip_xz(cx, cz, a0, a1);
    } else if *cy < 0.0 {
        flip_yz(cy, cz, a0, a1);
    }

    // Step 4: boundary of Weyl chamber: if cx == π/4 and cz < 0,
    // shift cx by -π/2 -> -π/4, then flip_xz gives cx = +π/4, cz = -cz > 0!
    if (*cx - FRAC_PI_4).abs() < 1e-12 && *cz < -1e-12 {
        // exp(i(π/4 XX + cy YY + cz ZZ)) = i (X⊗X) exp(i(-π/4 XX + cy YY + cz ZZ))
        *cx -= FRAC_PI_2;
        *phase += FRAC_PI_2;
        *a1 = mat2_mul(a1, &mx);
        *b1 = mat2_mul(b1, &mx);
        flip_xz(cx, cz, a0, a1);
    }

    // Clean tiny zeros.
    if cx.abs() < 1e-13 {
        *cx = 0.0;
    }
    if cy.abs() < 1e-13 {
        *cy = 0.0;
    }
    if cz.abs() < 1e-13 {
        *cz = 0.0;
    }
}

/// Emit a 1-qubit gate (if non-identity) and accumulate its global phase.
fn push_1q(ops: &mut Vec<Operation>, phase: &mut f64, m: &Matrix2, target: usize, tol: f64) {
    let (gate, g) = synthesize_1q(m, tol);
    *phase += g;
    if let Some(gate) = gate {
        ops.push(Operation::Gate { gate, target });
    }
}

/// Check if `u` is diagonal within `tol`; if so, synthesize with at most one
/// `CZ` or `CP` (in `Basis::Native`) or `0` 2q gates if separable.
fn try_synthesize_diagonal(
    u: &Matrix4,
    qubits: [usize; 2],
    basis: Basis,
    keep_real: bool,
    tol: f64,
) -> Option<(Vec<Operation>, f64)> {
    for (r, row) in u.iter().enumerate() {
        for (c, &v) in row.iter().enumerate() {
            if r != c && v.norm() > tol {
                return None;
            }
        }
    }
    let p0 = u[0][0].arg();
    let p1 = u[1][1].arg();
    let p2 = u[2][2].arg();
    let p3 = u[3][3].arg();
    let lam0 = wrap_pi(p1 - p0);
    let lam1 = wrap_pi(p2 - p0);
    let lam_cp = wrap_pi(p3 + p0 - p1 - p2);

    let mut ops = Vec::new();
    let mut phase = wrap_pi(p0);

    // Local phases on q0 and q1.
    if lam0.abs() > tol {
        let m0 = Gate::Phase(lam0).matrix();
        push_1q(&mut ops, &mut phase, &m0, qubits[0], tol);
    }
    if lam1.abs() > tol {
        let m1 = Gate::Phase(lam1).matrix();
        push_1q(&mut ops, &mut phase, &m1, qubits[1], tol);
    }

    if lam_cp.abs() <= tol {
        if keep_real && !phase_is_real(phase, tol) {
            return None;
        }
        return Some((ops, wrap_pi(phase)));
    }

    if (lam_cp.abs() - PI).abs() <= tol {
        // Entangling part is CZ!
        if basis == Basis::Native {
            let (c, t) = if qubits[0] < qubits[1] {
                (qubits[0], qubits[1])
            } else {
                (qubits[1], qubits[0])
            };
            ops.push(Operation::Controlled {
                gate: Gate::Z,
                control: c,
                target: t,
            });
            if keep_real && !phase_is_real(phase, tol) {
                return None;
            }
            return Some((ops, wrap_pi(phase)));
        }
        // In CX basis, let KAK synthesize 1 CX (absorbing H into local gates).
        return None;
    }

    if basis == Basis::Native && !keep_real {
        let (c, t) = if qubits[0] < qubits[1] {
            (qubits[0], qubits[1])
        } else {
            (qubits[1], qubits[0])
        };
        ops.push(Operation::Controlled {
            gate: Gate::Phase(lam_cp),
            control: c,
            target: t,
        });
        return Some((ops, wrap_pi(phase)));
    }
    None
}

/// Synthesize a 4×4 unitary `u` on `qubits = [q0, q1]` using the minimum
/// number of two-qubit gates for the requested `basis`.
///
/// Returns `Some((ops, global_phase))` if synthesis succeeds (and stays real
/// when `keep_real` is true), or `None` if `keep_real` prevented complex
/// intermediates.
pub(crate) fn synthesize_2q(
    u: &Matrix4,
    qubits: [usize; 2],
    basis: Basis,
    keep_real: bool,
    tol: f64,
) -> Option<(Vec<Operation>, f64)> {
    // Fast path for diagonal 2q unitaries (captures 0-gate, CZ, CP cleanly).
    if let Some(res) = try_synthesize_diagonal(u, qubits, basis, keep_real, tol) {
        return Some(res);
    }

    let kak = kak_decompose(u);
    let [q0, q1] = qubits;
    let mut ops = Vec::new();
    let mut phase = kak.global_phase;

    let synth_tol = tol.max(1e-11);

    if kak.cx <= synth_tol {
        // 0 two-qubit gates: pure product (b1*b0) ⊗ (a1*a0)!
        let a = mat2_mul(&kak.a1, &kak.a0);
        let b = mat2_mul(&kak.b1, &kak.b0);
        // When the full matrix is real, factor_kron may have introduced a
        // common scalar phase e^{iα} on A and e^{-iα} on B; align phases so
        // both A and B are real if possible.
        let (a, b) = align_product_phases(&a, &b, keep_real, tol);
        push_1q(&mut ops, &mut phase, &a, q0, tol);
        push_1q(&mut ops, &mut phase, &b, q1, tol);
    } else if basis == Basis::Native
        && (kak.cx - FRAC_PI_4).abs() <= synth_tol
        && (kak.cy - FRAC_PI_4).abs() <= synth_tol
        && (kak.cz - FRAC_PI_4).abs() <= synth_tol
    {
        // Locally equivalent to SWAP!
        // exp(i π/4 (XX + YY + ZZ)) = e^{i π/4} SWAP,
        // and SWAP · (B0 ⊗ A0) = (A0 ⊗ B0) · SWAP, so only 2 1q gates total!
        // U = e^{i(γ + π/4)} (b1 ⊗ a1) · SWAP · (b0 ⊗ a0)
        //   = e^{i(γ + π/4)} ((b1 · a0) ⊗ (a1 · b0)) · SWAP.
        phase += FRAC_PI_4;
        let on_q0 = mat2_mul(&kak.a1, &kak.b0);
        let on_q1 = mat2_mul(&kak.b1, &kak.a0);
        let (on_q0, on_q1) = align_product_phases(&on_q0, &on_q1, keep_real, tol);
        let (lo, hi) = if q0 < q1 { (q0, q1) } else { (q1, q0) };
        ops.push(Operation::Swap(lo, hi));
        push_1q(&mut ops, &mut phase, &on_q0, q0, tol);
        push_1q(&mut ops, &mut phase, &on_q1, q1, tol);
    } else if (kak.cx - FRAC_PI_4).abs() <= synth_tol
        && kak.cy <= synth_tol
        && kak.cz.abs() <= synth_tol
    {
        // Exactly 1 CX:
        // exp(i π/4 X_0 X_1) = e^{-i π/4} (Rx(-π/2)_1 ⊗ (H Rz(-π/2))_0) · CX(q0, q1) · (I_1 ⊗ H_0).
        phase -= FRAC_PI_4;
        let h = Gate::H.matrix();
        let a0 = mat2_mul(&h, &kak.a0);
        let b0 = kak.b0;
        let a1 = mat2_mul(&kak.a1, &mat2_mul(&h, &Gate::Rz(-FRAC_PI_2).matrix()));
        let b1 = mat2_mul(&kak.b1, &Gate::Rx(-FRAC_PI_2).matrix());
        let (a0, b0) = align_product_phases(&a0, &b0, keep_real, tol);
        let (a1, b1) = align_product_phases(&a1, &b1, keep_real, tol);
        push_1q(&mut ops, &mut phase, &a0, q0, tol);
        push_1q(&mut ops, &mut phase, &b0, q1, tol);
        ops.push(Operation::Controlled {
            gate: Gate::X,
            control: q0,
            target: q1,
        });
        push_1q(&mut ops, &mut phase, &a1, q0, tol);
        push_1q(&mut ops, &mut phase, &b1, q1, tol);
    } else if basis == Basis::Native
        && !keep_real
        && kak.cy <= synth_tol
        && kak.cz.abs() <= synth_tol
    {
        // Single-parameter entangler: exp(i cx XX) = (H ⊗ H) · Rzz(-2 cx) · (H ⊗ H).
        // Rzz is diagonal, so it commutes with neighbouring Z rotations!
        let h = Gate::H.matrix();
        let a0 = mat2_mul(&h, &kak.a0);
        let b0 = mat2_mul(&h, &kak.b0);
        let a1 = mat2_mul(&kak.a1, &h);
        let b1 = mat2_mul(&kak.b1, &h);
        push_1q(&mut ops, &mut phase, &a0, q0, tol);
        push_1q(&mut ops, &mut phase, &b0, q1, tol);
        let pauli = PauliString::new([(q0, Pauli::Z), (q1, Pauli::Z)]).expect("distinct qubits");
        ops.push(Operation::PauliRotation {
            pauli,
            theta: -2.0 * kak.cx,
        });
        push_1q(&mut ops, &mut phase, &a1, q0, tol);
        push_1q(&mut ops, &mut phase, &b1, q1, tol);
    } else if kak.cz.abs() <= synth_tol {
        // Exactly 2 CXs with a purely-real core!
        // exp(i(cx XX + cy YY))
        //   = (I_1 ⊗ (S·H)_0) · CX(q0, q1) · (Ry(-2 cy)_1 ⊗ Ry(-2 cx)_0)
        //     · CX(q0, q1) · (I_1 ⊗ (H·S†)_0).
        // When `u` is real (e.g. SO(4) with cz = 0), absorbing (S·H) and (H·S†)
        // into `a1`/`a0` yields 100% real outer single-qubit gates!
        let sh = mat2_mul(&Gate::S.matrix(), &Gate::H.matrix());
        let hsdg = mat2_mul(&Gate::H.matrix(), &Gate::Sdg.matrix());
        let a0 = mat2_mul(&hsdg, &kak.a0);
        let b0 = kak.b0;
        let a1 = mat2_mul(&kak.a1, &sh);
        let b1 = kak.b1;
        let (a0, b0) = align_product_phases(&a0, &b0, keep_real, tol);
        let (a1, b1) = align_product_phases(&a1, &b1, keep_real, tol);
        push_1q(&mut ops, &mut phase, &a0, q0, tol);
        push_1q(&mut ops, &mut phase, &b0, q1, tol);
        ops.push(Operation::Controlled {
            gate: Gate::X,
            control: q0,
            target: q1,
        });
        push_1q(
            &mut ops,
            &mut phase,
            &Gate::Ry(-2.0 * kak.cx).matrix(),
            q0,
            tol,
        );
        push_1q(
            &mut ops,
            &mut phase,
            &Gate::Ry(-2.0 * kak.cy).matrix(),
            q1,
            tol,
        );
        ops.push(Operation::Controlled {
            gate: Gate::X,
            control: q0,
            target: q1,
        });
        push_1q(&mut ops, &mut phase, &a1, q0, tol);
        push_1q(&mut ops, &mut phase, &b1, q1, tol);
    } else {
        // General SU(4): exactly 3 CXs (Vatan & Williams):
        // exp(i(cx XX + cy YY + cz ZZ))
        //   = e^{i π/4} · (Rz(π/2)_1 ⊗ I_0)
        //     · CX(q1, q0)
        //     · (Ry(2 cx - π/2)_1 ⊗ Rz(π/2 - 2 cz)_0)
        //     · CX(q0, q1)
        //     · (Ry(π/2 - 2 cy)_1 ⊗ I_0)
        //     · CX(q1, q0)
        //     · (I_1 ⊗ Rz(-π/2)_0).
        phase += FRAC_PI_4;
        let a0 = mat2_mul(&Gate::Rz(-FRAC_PI_2).matrix(), &kak.a0);
        let b0 = kak.b0;
        let a1 = kak.a1;
        let b1 = mat2_mul(&kak.b1, &Gate::Rz(FRAC_PI_2).matrix());
        push_1q(&mut ops, &mut phase, &a0, q0, tol);
        push_1q(&mut ops, &mut phase, &b0, q1, tol);
        ops.push(Operation::Controlled {
            gate: Gate::X,
            control: q1,
            target: q0,
        });
        push_1q(
            &mut ops,
            &mut phase,
            &Gate::Ry(FRAC_PI_2 - 2.0 * kak.cy).matrix(),
            q1,
            tol,
        );
        ops.push(Operation::Controlled {
            gate: Gate::X,
            control: q0,
            target: q1,
        });
        push_1q(
            &mut ops,
            &mut phase,
            &Gate::Rz(FRAC_PI_2 - 2.0 * kak.cz).matrix(),
            q0,
            tol,
        );
        push_1q(
            &mut ops,
            &mut phase,
            &Gate::Ry(2.0 * kak.cx - FRAC_PI_2).matrix(),
            q1,
            tol,
        );
        ops.push(Operation::Controlled {
            gate: Gate::X,
            control: q1,
            target: q0,
        });
        push_1q(&mut ops, &mut phase, &a1, q0, tol);
        push_1q(&mut ops, &mut phase, &b1, q1, tol);
    }

    phase = wrap_pi(phase);
    if keep_real {
        if !phase_is_real(phase, tol) {
            return None;
        }
        for op in &ops {
            if let Operation::Gate { gate, .. } = op
                && !mat2_is_real(&gate.matrix(), tol)
            {
                return None;
            }
        }
    }

    Some((ops, phase))
}

/// Try to shift a common phase between `a` and `b` (`a -> e^{iα} a`,
/// `b -> e^{-iα} b`, leaving `b ⊗ a` unchanged) so that both matrices are
/// real whenever `b ⊗ a` is real.
fn align_product_phases(a: &Matrix2, b: &Matrix2, keep_real: bool, tol: f64) -> (Matrix2, Matrix2) {
    if !keep_real {
        return (*a, *b);
    }
    let find_phase = |m: &Matrix2| -> f64 {
        let mut best = ZERO;
        for row in m {
            for &v in row {
                if v.norm_sqr() > best.norm_sqr() {
                    best = v;
                }
            }
        }
        best.arg()
    };
    let pa = find_phase(a);
    let fa = C64::cis(-pa);
    let fb = C64::cis(pa);
    let a2 = std::array::from_fn(|r| std::array::from_fn(|c| a[r][c] * fa));
    let b2 = std::array::from_fn(|r| std::array::from_fn(|c| b[r][c] * fb));
    if mat2_is_real(&a2, tol) && mat2_is_real(&b2, tol) {
        (a2, b2)
    } else {
        (*a, *b)
    }
}

/// Evaluate the 4×4 matrix of a sequence of operations on `qubits` (with
/// `global_phase`), for testing and verification.
#[cfg(test)]
pub(crate) fn eval_2q_ops(ops: &[Operation], qubits: [usize; 2], global_phase: f64) -> Matrix4 {
    let mut acc: Matrix4 = std::array::from_fn(|r| {
        std::array::from_fn(|c| if r == c { C64::cis(global_phase) } else { ZERO })
    });
    for op in ops {
        let m = op_matrix4_on(op, qubits);
        acc = mat4_mul(&m, &acc);
    }
    acc
}

/// Build the 4×4 matrix of any 1q or 2q operation whose qubits are a subset of
/// `qubits = [q0, q1]`.
pub(crate) fn op_matrix4_on(op: &Operation, qubits: [usize; 2]) -> Matrix4 {
    match *op {
        Operation::Gate { gate, target } => {
            if target == qubits[0] {
                kron2(&I2, &gate.matrix())
            } else {
                debug_assert_eq!(target, qubits[1]);
                kron2(&gate.matrix(), &I2)
            }
        }
        Operation::Controlled {
            gate,
            control,
            target,
        } => {
            let u = gate.matrix();
            let mut m = [[ZERO; 4]; 4];
            if control == qubits[0] && target == qubits[1] {
                m[0][0] = ONE;
                m[2][2] = ONE;
                for r in 0..2 {
                    for c in 0..2 {
                        m[2 * r + 1][2 * c + 1] = u[r][c];
                    }
                }
            } else {
                debug_assert!(control == qubits[1] && target == qubits[0]);
                m[0][0] = ONE;
                m[1][1] = ONE;
                for r in 0..2 {
                    for c in 0..2 {
                        m[2 + r][2 + c] = u[r][c];
                    }
                }
            }
            m
        }
        Operation::Swap(a, b) => {
            debug_assert!((a == qubits[0] && b == qubits[1]) || (a == qubits[1] && b == qubits[0]));
            let mut m = [[ZERO; 4]; 4];
            for (r, c) in [(0, 0), (1, 2), (2, 1), (3, 3)] {
                m[r][c] = ONE;
            }
            m
        }
        Operation::Unitary2 { matrix, qubits: qs } => {
            if qs == qubits {
                matrix
            } else {
                debug_assert_eq!(qs, [qubits[1], qubits[0]]);
                const P: [usize; 4] = [0, 2, 1, 3];
                std::array::from_fn(|r| std::array::from_fn(|c| matrix[P[r]][P[c]]))
            }
        }
        Operation::PauliRotation { ref pauli, theta } => {
            let p0 = pauli
                .ops()
                .iter()
                .find(|&&(q, _)| q == qubits[0])
                .map_or(Pauli::I, |&(_, p)| p);
            let p1 = pauli
                .ops()
                .iter()
                .find(|&&(q, _)| q == qubits[1])
                .map_or(Pauli::I, |&(_, p)| p);
            let pm = kron2(
                &crate::optimize::one_qubit::pauli_mat2(p1),
                &crate::optimize::one_qubit::pauli_mat2(p0),
            );
            let (s, c) = (theta / 2.0).sin_cos();
            std::array::from_fn(|row| {
                std::array::from_fn(|col| {
                    let id = if row == col { C64::new(c, 0.0) } else { ZERO };
                    id + C64::new(0.0, -s) * pm[row][col]
                })
            })
        }
        Operation::MultiControlled { .. } | Operation::ControlledSwap { .. } => {
            unreachable!("multi-controlled operations are not 2q blocks")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn assert_mat4_close(a: &Matrix4, b: &Matrix4, msg: &str) {
        for r in 0..4 {
            for c in 0..4 {
                assert!(
                    (a[r][c] - b[r][c]).norm() < 1e-10,
                    "{msg}: mismatch at ({r},{c}): {:?} vs {:?}",
                    a[r][c],
                    b[r][c]
                );
            }
        }
    }

    fn count_2q(ops: &[Operation]) -> usize {
        ops.iter().filter(|op| op.qubits().len() >= 2).count()
    }

    #[test]
    fn kak_exact_on_degenerate_and_special_gates() {
        let qubits = [0, 1];
        let cases: Vec<(&str, Matrix4, usize, usize)> = vec![
            ("identity", kron2(&I2, &I2), 0, 0),
            (
                "product",
                kron2(
                    &Gate::Ry(0.7).matrix(),
                    &Gate::U {
                        theta: 1.1,
                        phi: 0.3,
                        lambda: -0.5,
                    }
                    .matrix(),
                ),
                0,
                0,
            ),
            (
                "cx",
                op_matrix4_on(
                    &Operation::Controlled {
                        gate: Gate::X,
                        control: 0,
                        target: 1,
                    },
                    qubits,
                ),
                1,
                1,
            ),
            (
                "cz",
                op_matrix4_on(
                    &Operation::Controlled {
                        gate: Gate::Z,
                        control: 0,
                        target: 1,
                    },
                    qubits,
                ),
                1,
                1,
            ),
            (
                "cp",
                op_matrix4_on(
                    &Operation::Controlled {
                        gate: Gate::Phase(0.7),
                        control: 0,
                        target: 1,
                    },
                    qubits,
                ),
                1,
                2,
            ),
            ("swap", op_matrix4_on(&Operation::Swap(0, 1), qubits), 1, 3),
            (
                "rxx",
                op_matrix4_on(
                    &Operation::PauliRotation {
                        pauli: PauliString::from_label("XX").unwrap(),
                        theta: 0.9,
                    },
                    qubits,
                ),
                1,
                2,
            ),
        ];

        for (name, u, max_native_2q, max_cx_2q) in cases {
            for (basis, max_2q) in [(Basis::Native, max_native_2q), (Basis::Cx, max_cx_2q)] {
                let (ops, phase) = synthesize_2q(&u, qubits, basis, false, 1e-12).unwrap();
                assert!(
                    count_2q(&ops) <= max_2q,
                    "{name} ({basis:?}): got {} 2q gates, expected <= {max_2q}: {ops:?}",
                    count_2q(&ops)
                );
                let got = eval_2q_ops(&ops, qubits, phase);
                assert_mat4_close(&u, &got, &format!("{name} ({basis:?})"));
            }
        }
    }

    #[test]
    fn kak_exact_on_random_2q_unitaries() {
        let mut rng = StdRng::seed_from_u64(12345);
        let qubits = [0, 1];
        for trial in 0..100 {
            let mut u = kron2(&I2, &I2);
            for _ in 0..8 {
                let g0 = Gate::U {
                    theta: rng.random_range(-3.0..3.0),
                    phi: rng.random_range(-3.0..3.0),
                    lambda: rng.random_range(-3.0..3.0),
                };
                let g1 = Gate::U {
                    theta: rng.random_range(-3.0..3.0),
                    phi: rng.random_range(-3.0..3.0),
                    lambda: rng.random_range(-3.0..3.0),
                };
                u = mat4_mul(&kron2(&g1.matrix(), &g0.matrix()), &u);
                let (c, t) = if rng.random() { (0, 1) } else { (1, 0) };
                let cx = op_matrix4_on(
                    &Operation::Controlled {
                        gate: Gate::X,
                        control: c,
                        target: t,
                    },
                    qubits,
                );
                u = mat4_mul(&cx, &u);
            }
            for basis in [Basis::Native, Basis::Cx] {
                let (ops, phase) = synthesize_2q(&u, qubits, basis, false, 1e-12).unwrap();
                assert!(count_2q(&ops) <= 3);
                let got = eval_2q_ops(&ops, qubits, phase);
                assert_mat4_close(&u, &got, &format!("random trial {trial} ({basis:?})"));
            }
        }
    }
}
