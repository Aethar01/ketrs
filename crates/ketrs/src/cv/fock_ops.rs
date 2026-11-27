//! Fast Fock-basis gate matrix builders using Miatto–Quesada recurrences and
//! cached padded quadrature eigendecompositions.
//!
//! Conventions match [`crate::cv`] (Strawberry Fields, ħ = 2):
//! `x̂ = a + a†`, `p̂ = −i(a − a†)`.

use super::fixed_n::build_two_mode_blocks;
use faer::{Mat, Side};
use num_complex::Complex64 as C64;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// Extra Fock levels used when exponentiating quadrature polynomials
/// (`P(s)`, `V(γ)`, `CX(s)`, `CZ(s)`) so truncation only distorts levels far
/// above `cutoff`.
const QUAD_PAD: usize = 60;

/// Single-mode displacement matrix `D(α)_{n, k} = ⟨n|D(α)|k⟩` (`d × d`,
/// row-major) by the exact `O(d²)` Miatto–Quesada recurrence:
///
/// `D_{n, k} = √(k/n) D_{n-1, k-1} + (α / √n) D_{n-1, k}`.
pub(crate) fn displacement_matrix(alpha: C64, d: usize) -> Vec<C64> {
    let mut mat = vec![C64::new(0.0, 0.0); d * d];
    if d == 0 {
        return mat;
    }
    let sqrt_t: Vec<f64> = (0..=d).map(|k| (k as f64).sqrt()).collect();
    let inv_sqrt: Vec<f64> = (0..=d)
        .map(|k| if k == 0 { 0.0 } else { 1.0 / sqrt_t[k] })
        .collect();

    mat[0] = C64::new((-0.5 * alpha.norm_sqr()).exp(), 0.0);
    // Top row: D_{0, k} = (-α* / √k) D_{0, k-1}.
    let neg_conj = -alpha.conj();
    for k in 1..d {
        mat[k] = mat[k - 1] * neg_conj * inv_sqrt[k];
    }
    // Remaining rows: D_{n, 0} = (α / √n) D_{n-1, 0},
    // D_{n, k} = √(k/n) D_{n-1, k-1} + (α / √n) D_{n-1, k}.
    for (n, &inv_n) in inv_sqrt.iter().enumerate().take(d).skip(1) {
        let a_n = alpha * inv_n;
        let prev = (n - 1) * d;
        let row = n * d;
        mat[row] = mat[prev] * a_n;
        for k in 1..d {
            mat[row + k] = mat[prev + k - 1] * (sqrt_t[k] * inv_n) + mat[prev + k] * a_n;
        }
    }
    mat
}

/// Single-mode squeezing matrix `S(r e^{iφ})_{n, k} = ⟨n|S(z)|k⟩` (`d × d`,
/// row-major) by the exact `O(d²)` checkerboard-sparse recurrence:
///
/// `S_{n, k} = (√k / (√n cosh r)) S_{n-1, k-1} − e^{iφ} tanh r √((n−1)/n) S_{n-2, k}`.
pub(crate) fn squeezing_matrix(r: f64, phi: f64, d: usize) -> Vec<C64> {
    let mut mat = vec![C64::new(0.0, 0.0); d * d];
    if d == 0 {
        return mat;
    }
    let sqrt_t: Vec<f64> = (0..=d).map(|k| (k as f64).sqrt()).collect();
    let inv_sqrt: Vec<f64> = (0..=d)
        .map(|k| if k == 0 { 0.0 } else { 1.0 / sqrt_t[k] })
        .collect();

    let inv_ch = 1.0 / r.cosh();
    let th = r.tanh();
    let ratio_down = -C64::from_polar(th, phi); // -e^{iφ} tanh r
    let ratio_right = C64::from_polar(th, -phi); // +e^{-iφ} tanh r

    mat[0] = C64::new(inv_ch.sqrt(), 0.0);
    // Top row: only even k nonzero.
    for k in (2..d).step_by(2) {
        mat[k] = mat[k - 2] * ratio_right * (sqrt_t[k - 1] * inv_sqrt[k]);
    }
    // Row 1: S_{1, k} = (√k / cosh r) S_{0, k-1} (only odd k nonzero).
    if d > 1 {
        for k in (1..d).step_by(2) {
            mat[d + k] = mat[k - 1] * (sqrt_t[k] * inv_ch);
        }
    }
    // Rows n >= 2: only k with (n + k) even nonzero.
    for n in 2..d {
        let c1 = inv_sqrt[n] * inv_ch;
        let c2 = ratio_down * (sqrt_t[n - 1] * inv_sqrt[n]);
        let row = n * d;
        let prev1 = (n - 1) * d;
        let prev2 = (n - 2) * d;
        let k_start = n % 2;
        if k_start == 0 {
            mat[row] = mat[prev2] * c2;
        }
        let mut k = if k_start == 0 { 2 } else { 1 };
        while k < d {
            mat[row + k] = mat[prev1 + k - 1] * (sqrt_t[k] * c1) + mat[prev2 + k] * c2;
            k += 2;
        }
    }
    mat
}

/// Photon-number-conserving `(s + 1) × (s + 1)` blocks for `BS(θ, φ)` up to
/// `s = 2 * (d - 1)`. Entry `[out_a * (s + 1) + in_a]` is
/// `⟨out_a, s - out_a | BS(θ, φ) | in_a, s - in_a⟩`.
pub(crate) fn beam_splitter_blocks(theta: f64, phi: f64, d: usize) -> Vec<Vec<C64>> {
    let t = C64::new(theta.cos(), 0.0);
    let r = C64::from_polar(theta.sin(), phi);
    let u2 = [t, -r.conj(), r, t];
    build_two_mode_blocks(&u2, 2 * d.saturating_sub(1))
}

/// Photon-difference-conserving blocks for two-mode squeezing `S₂(r e^{iφ})`
/// on cutoff `d`.
///
/// Returns `2d - 1` blocks indexed by `idx = (diff + d - 1) as usize` for
/// `diff = n_a - n_b ∈ [-(d - 1), d - 1]`. Block `diff` has size `len × len`
/// where `len = d - |diff|`, and entry `[out * len + inp]` is
/// `⟨out + oa, out + ob | S₂(z) | inp + oa, inp + ob⟩` with
/// `oa = max(diff, 0)`, `ob = max(-diff, 0)`.
pub(crate) fn two_mode_squeezing_blocks(r: f64, phi: f64, d: usize) -> Vec<Vec<C64>> {
    if d == 0 {
        return Vec::new();
    }
    let sqrt_t: Vec<f64> = (0..=d).map(|k| (k as f64).sqrt()).collect();
    let inv_sqrt: Vec<f64> = (0..=d)
        .map(|k| if k == 0 { 0.0 } else { 1.0 / sqrt_t[k] })
        .collect();

    let inv_ch = 1.0 / r.cosh();
    let th = r.tanh();
    let neg_z_th = -C64::from_polar(th, phi); // -e^{iφ} tanh r
    let pos_zc_th = C64::from_polar(th, -phi); // +e^{-iφ} tanh r

    // Store full tensor S[ma, mb, na] (since nb = na - (ma - mb) is fixed by Δ).
    // Even cleaner: compute blocks by increasing Δ = -(d-1) ..= (d-1)!
    // Wait: the recurrence for ma > 0 is:
    //   S[ma, mb; na, nb] = (√na / (√ma cosh r)) S[ma-1, mb; na-1, nb]
    //                       - e^{iφ} tanh r √(mb / ma) S[ma-1, mb-1; na, nb]
    // Notice S[ma-1, mb; na-1, nb] has Δ' = Δ - 1, and S[ma-1, mb-1; na, nb] has Δ' = Δ!
    // So computing in strictly increasing order of Δ = -(d-1), ..., +(d-1) has ALL
    // dependencies ALREADY computed!!
    let n_blocks = 2 * d - 1;
    let mut blocks: Vec<Vec<C64>> = Vec::with_capacity(n_blocks);

    for diff in -(d as isize - 1)..=(d as isize - 1) {
        let abs_d = diff.unsigned_abs();
        let len = d - abs_d;
        let oa = diff.max(0) as usize;
        let ob = (-diff).max(0) as usize;
        let mut blk = vec![C64::new(0.0, 0.0); len * len];

        // Top row: out == 0 -> (ma, mb) = (oa, ob).
        // For diff <= 0: oa == 0, ob == |diff| -> ma == 0.
        //   S[0, ob; inp, inp + ob] = (cosh r)^{-(1 + ob)} * (+e^{-iφ} tanh r)^inp * √(C(inp + ob, inp))
        //   Recurrence: blk[0, inp] = blk[0, inp-1] * (+e^{-iφ} tanh r) * √((inp + ob) / inp).
        // For diff > 0: oa == diff > 0, ob == 0 -> mb == 0!
        //   Same formula by mode symmetry a ↔ b:
        //   blk[0, 0] = (cosh r)^{-(1 + oa)},
        //   blk[0, inp] = blk[0, inp-1] * (+e^{-iφ} tanh r) * √((inp + oa) / inp).
        blk[0] = C64::new(inv_ch.powi((1 + abs_d) as i32), 0.0);
        for inp in 1..len {
            blk[inp] = blk[inp - 1] * pos_zc_th * (sqrt_t[inp + abs_d] * inv_sqrt[inp]);
        }

        // Remaining rows: out = 1..len.
        // Here ma = out + oa >= 1 and mb = out + ob >= 1 (since out >= 1!).
        // Check: we need S[ma-1, mb; na-1, nb] (which has Δ - 1) and S[ma-1, mb-1; na, nb] (Δ).
        // Wait: if diff == -(d - 1), len == 1, so out loop 1..1 is empty!
        // For diff > -(d - 1), block (diff - 1) ALREADY exists at `blocks[blocks.len() - 1]`!
        if len > 1 {
            let prev_diff = diff - 1;
            let prev_abs = prev_diff.unsigned_abs();
            let prev_len = d - prev_abs;
            let prev_oa = prev_diff.max(0) as usize;
            let prev_blk = &blocks[blocks.len() - 1];

            for out in 1..len {
                let ma = out + oa;
                let mb = out + ob;
                let c_prev = inv_sqrt[ma] * inv_ch;
                let c_same = neg_z_th * (sqrt_t[mb] * inv_sqrt[ma]);
                let row = out * len;
                let same_prev_row = (out - 1) * len;
                // In block (diff - 1), state (ma - 1, mb) has row index `(ma - 1) - prev_oa`.
                let prev_row = (ma - 1 - prev_oa) * prev_len;

                for inp in 0..len {
                    let na = inp + oa;
                    let mut val = blk[same_prev_row + inp] * c_same;
                    if na > 0 {
                        // In block (diff - 1), state (na - 1, nb) has col index `(na - 1) - prev_oa`.
                        let prev_col = na - 1 - prev_oa;
                        val += prev_blk[prev_row + prev_col] * (sqrt_t[na] * c_prev);
                    }
                    blk[row + inp] = val;
                }
            }
        }
        blocks.push(blk);
    }
    blocks
}

// ---------------------------------------------------------------------------
// Cached padded quadrature eigenbases for P(s), V(γ), CX(s), CZ(s)
// ---------------------------------------------------------------------------

pub(crate) struct QuadEigen {
    pub(crate) big: usize,
    /// Eigenvalues of `x̂ = a + a†` on `big` levels (real).
    pub(crate) evals: Vec<f64>,
    /// Top `d` rows of `x̂` eigenvectors (`d × big`, row-major, real).
    pub(crate) vx: Vec<f64>,
    /// Top `d` rows of `p̂` eigenvectors (`d × big`, row-major, complex:
    /// `p̂ = R(−π/2)† x̂ R(−π/2)` => `vp[n, k] = i^n * vx[n, k]`!).
    pub(crate) vp: Vec<C64>,
}

fn quad_eigen_cache() -> &'static Mutex<HashMap<usize, Arc<QuadEigen>>> {
    static CACHE: OnceLock<Mutex<HashMap<usize, Arc<QuadEigen>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

pub(crate) fn cached_quad_eigen(d: usize) -> Arc<QuadEigen> {
    let mut map = quad_eigen_cache().lock().unwrap();
    if let Some(entry) = map.get(&d) {
        return Arc::clone(entry);
    }
    let big = d + QUAD_PAD;
    // Build real symmetric tridiagonal matrix for x̂ = a + a†.
    let x = Mat::from_fn(big, big, |i, j| {
        if j == i + 1 {
            (j as f64).sqrt()
        } else if i == j + 1 {
            (i as f64).sqrt()
        } else {
            0.0
        }
    });
    let eig = x
        .self_adjoint_eigen(Side::Lower)
        .expect("tridiagonal x̂ eigensolver converges");
    let u = eig.U();
    let s = eig.S().column_vector();
    let evals: Vec<f64> = (0..big).map(|k| s[k]).collect();
    let mut vx = vec![0.0; d * big];
    let mut vp = vec![C64::new(0.0, 0.0); d * big];
    // Since R(π/2) x̂ R(-π/2) = p̂ (check: R(φ)† x̂ R(φ) = x̂ cos φ - p̂ sin φ,
    // wait: R(φ)|n⟩ = e^{iφn}|n⟩ => R(π/2) x̂ R(-π/2) has matrix element
    // ⟨n|R(π/2) x̂ R(-π/2)|k⟩ = i^{n - k} ⟨n|x̂|k⟩:
    // for k = n + 1: i^{-1} √(n+1) = -i √(n+1) = ⟨n|-i(a - a†)|n+1⟩ = ⟨n|p̂|n+1⟩!
    // EXACT! So if x̂ v = λ v, then p̂ (R(π/2) v) = λ (R(π/2) v),
    // i.e., vp[n, k] = i^n * vx[n, k]!).
    let i_pows = [
        C64::new(1.0, 0.0),
        C64::new(0.0, 1.0),
        C64::new(-1.0, 0.0),
        C64::new(0.0, -1.0),
    ];
    for n in 0..d {
        let ph = i_pows[n % 4];
        for k in 0..big {
            let v = u[(n, k)];
            vx[n * big + k] = v;
            vp[n * big + k] = ph * v;
        }
    }
    let arc = Arc::new(QuadEigen { big, evals, vx, vp });
    map.insert(d, Arc::clone(&arc));
    arc
}

/// `P(s) = exp(i s x̂² / 4)` on `d` levels (`d × d`, row-major).
pub(crate) fn quadratic_phase_matrix(s: f64, d: usize) -> Vec<C64> {
    let qe = cached_quad_eigen(d);
    let phases: Vec<C64> = qe
        .evals
        .iter()
        .map(|&l| C64::from_polar(1.0, 0.25 * s * l * l))
        .collect();
    x_diagonal_gate(&qe, d, &phases)
}

/// `V(γ) = exp(i γ x̂³ / 6)` on `d` levels (`d × d`, row-major).
pub(crate) fn cubic_phase_matrix(gamma: f64, d: usize) -> Vec<C64> {
    let qe = cached_quad_eigen(d);
    let phases: Vec<C64> = qe
        .evals
        .iter()
        .map(|&l| C64::from_polar(1.0, gamma * l * l * l / 6.0))
        .collect();
    x_diagonal_gate(&qe, d, &phases)
}

fn x_diagonal_gate(qe: &QuadEigen, d: usize, phases: &[C64]) -> Vec<C64> {
    let big = qe.big;
    let mut out = vec![C64::new(0.0, 0.0); d * d];
    for i in 0..d {
        let ri = &qe.vx[i * big..(i + 1) * big];
        for j in i..d {
            let rj = &qe.vx[j * big..(j + 1) * big];
            let mut acc = C64::new(0.0, 0.0);
            for k in 0..big {
                acc += phases[k] * (ri[k] * rj[k]);
            }
            out[i * d + j] = acc;
            out[j * d + i] = acc;
        }
    }
    out
}

/// Hermite-Gauss wavefunction values `ψ_n(q) = ⟨q|n⟩` for `n = 0..d` with
/// `ħ = 2` (so `x̂ = a + a†`, `ψ_0(q) = (2π)^{-1/4} exp(-q² / 4)`, and
/// `ψ_{n+1}(q) = (q ψ_n(q) − √n ψ_{n-1}(q)) / √(n + 1)`).
///
/// Verified: `∫ |ψ_0(q)|² dq = 1` and `⟨0|x̂|1⟩ = ∫ q ψ_0(q) ψ_1(q) dq = 1`.
pub(crate) fn hermite_fock_wavefunctions(q: f64, d: usize, out: &mut [f64]) {
    if d == 0 {
        return;
    }
    let norm0 = std::f64::consts::TAU.powf(-0.25);
    out[0] = norm0 * (-0.25 * q * q).exp();
    if d == 1 {
        return;
    }
    out[1] = q * out[0];
    for n in 1..(d - 1) {
        let nf = n as f64;
        let np1 = (n + 1) as f64;
        out[n + 1] = (q * out[n] - nf.sqrt() * out[n - 1]) / np1.sqrt();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cv::ModeCircuit;
    use crate::cv::reference::FockReference;

    #[test]
    fn recurrence_gates_match_reference() {
        let d = 14;
        // Displacement
        let mut c = ModeCircuit::new(1);
        c.displace(0.6, 0.9, 0);
        let rf = FockReference::run(&c, d).unwrap();
        let dm = displacement_matrix(C64::from_polar(0.6, 0.9), d);
        for n in 0..d {
            assert!((dm[n * d] - rf.amplitude(&[n])).norm() < 1e-11);
        }

        // Squeezing
        let mut c = ModeCircuit::new(1);
        c.squeeze(0.45, -0.7, 0);
        let rf = FockReference::run(&c, d).unwrap();
        let sm = squeezing_matrix(0.45, -0.7, d);
        for n in 0..d {
            assert!((sm[n * d] - rf.amplitude(&[n])).norm() < 1e-11);
        }

        // Two-mode squeezing from |1, 0⟩ and |1, 1⟩
        let d2 = 10;
        let blocks = two_mode_squeezing_blocks(0.4, 0.8, d2);
        for (ia, ib) in [(0, 0), (1, 0), (0, 2), (1, 1), (2, 1)] {
            let mut c = ModeCircuit::new(2);
            c.set_input(0, crate::cv::ModeState::Fock(ia)).unwrap();
            c.set_input(1, crate::cv::ModeState::Fock(ib)).unwrap();
            c.two_mode_squeeze(0.4, 0.8, 0, 1);
            let rf = FockReference::run(&c, d2).unwrap();
            let diff = ia as isize - ib as isize;
            let idx = (diff + d2 as isize - 1) as usize;
            let oa = diff.max(0) as usize;
            let ob = (-diff).max(0) as usize;
            let len = d2 - diff.unsigned_abs();
            let inp = ia - oa;
            let blk = &blocks[idx];
            for out in 0..len {
                let got = blk[out * len + inp];
                let want = rf.amplitude(&[out + oa, out + ob]);
                assert!(
                    (got - want).norm() < 1e-10,
                    "S2 ({ia},{ib})->({},{}) diff={diff}: {got} vs {want}",
                    out + oa,
                    out + ob
                );
            }
        }
    }
}
