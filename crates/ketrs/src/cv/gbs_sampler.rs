//! Exact Gaussian Boson Sampling (GBS) chain-rule samplers for both
//! photon-number-resolving (PNR) and threshold detectors.
//!
//! Algorithms:
//! - Threshold GBS (`sample_gbs_threshold`): Quesada–Arrazola (2020) chain rule
//!   in increasing single-mode click-probability order, using incremental real
//!   `2 × 2` block Cholesky updates (`O(|S|² 2^{|S|})` per step, zero extra
//!   branching for non-clicking modes).
//! - PNR GBS (`sample_gbs_pnr`): Quesada–Arrazola (2020) / Bulmer et al. (2022)
//!   heterodyne-conditioned pure-state chain rule. Mixed states are decomposed
//!   via Williamson decomposition `V = T + W` (`T` pure, `W ≽ 0`) once up
//!   front, and all `m` conditional covariance matrices, gain matrices `K^{(k)}`,
//!   and `B^{(k)}` matrices are precomputed once across all shots.

use super::gaussian::GaussianState;
use super::hafnian::{
    extend_cholesky_real_displaced, loop_hafnian_repeated, threshold_from_partial_cholesky,
};
use crate::error::Error;
use faer::{Mat, Side};
use num_complex::Complex64 as C64;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;

/// Minimum shot count for parallelising independent GBS samples across threads.
const GBS_PAR_MIN_SHOTS: usize = 4;

/// Draw `shots` exact threshold-detector click patterns (`Vec<bool>` of length
/// `state.n_modes()`) from any Gaussian state (pure, mixed, lossy, or
/// displaced) via the chain rule with incremental real recursive Cholesky.
pub fn sample_gbs_threshold<R: Rng + ?Sized>(
    state: &GaussianState,
    shots: usize,
    rng: &mut R,
) -> Result<Vec<Vec<bool>>, Error> {
    let m = state.n_modes();
    if shots == 0 {
        return Ok(Vec::new());
    }
    if m == 0 {
        return Ok(vec![Vec::new(); shots]);
    }

    // Order modes by increasing single-mode click probability so high-click
    // modes come last (keeps |S_k| small on early steps).
    let mut order: Vec<usize> = (0..m).collect();
    let click_probs: Vec<f64> = (0..m)
        .map(|i| {
            let p0 = state.reduced(&[i]).unwrap().vacuum_probability();
            1.0 - p0
        })
        .collect();
    order.sort_by(|&a, &b| click_probs[a].total_cmp(&click_probs[b]));

    // Build interleaved M = (V + I) / 2 and μ in `order` once across all shots.
    let dim = 2 * m;
    let cov = state.cov();
    let means = state.means();
    let mut idx = Vec::with_capacity(dim);
    for &mode in &order {
        idx.push(mode);
        idx.push(m + mode);
    }
    let mut m_mat = vec![0.0; dim * dim];
    let mut mu_inter = vec![0.0; dim];
    for r in 0..dim {
        mu_inter[r] = means[idx[r]];
        for c in 0..dim {
            let mut v = 0.5 * cov[idx[r] * dim + idx[c]];
            if r == c {
                v += 0.5;
            }
            m_mat[r * dim + c] = v;
        }
    }

    let draw_one = |local_rng: &mut StdRng| -> Vec<bool> {
        // Maintain incremental Cholesky factor of all non-clicking modes so far.
        let mut l = vec![0.0; dim * dim];
        let mut w = vec![0.0; dim];
        let mut zeros: Vec<usize> = Vec::with_capacity(m);
        let mut clicks: Vec<usize> = Vec::with_capacity(m);
        let mut inv_det_z = 1.0f64;
        let mut quad_z = 0.0f64;
        let mut p_prev = 1.0f64;
        let mut res_sorted = vec![false; m];

        for (k, slot) in res_sorted.iter_mut().enumerate() {
            // Tentatively extend the non-clicking Cholesky factor with mode `k`.
            let (id_try, q_try) = extend_cholesky_real_displaced(
                &m_mat, &mu_inter, dim, &mut l, &mut w, &zeros, k, inv_det_z, quad_z,
            );
            zeros.push(k);
            let p0 = threshold_from_partial_cholesky(
                &m_mat, &mu_inter, dim, &mut l, &mut w, &mut zeros, &clicks, id_try, q_try,
            )
            .clamp(0.0, p_prev);

            let u = local_rng.random::<f64>() * p_prev;
            if u < p0 {
                // Mode k did NOT click: keep the extended Cholesky factor!
                inv_det_z = id_try;
                quad_z = q_try;
                p_prev = p0.max(1e-300);
                *slot = false;
            } else {
                // Mode k clicked: revert `zeros` and add k to `clicks`.
                zeros.pop();
                clicks.push(k);
                p_prev = (p_prev - p0).max(1e-300);
                *slot = true;
            }
        }

        let mut out = vec![false; m];
        for (pos, &orig_mode) in order.iter().enumerate() {
            out[orig_mode] = res_sorted[pos];
        }
        out
    };

    if shots >= GBS_PAR_MIN_SHOTS {
        let seeds: Vec<u64> = (0..shots).map(|_| rng.random()).collect();
        Ok(seeds
            .into_par_iter()
            .map(|seed| {
                let mut local_rng = StdRng::seed_from_u64(seed);
                draw_one(&mut local_rng)
            })
            .collect())
    } else {
        let mut out = Vec::with_capacity(shots);
        for _ in 0..shots {
            let mut local_rng = StdRng::seed_from_u64(rng.random());
            out.push(draw_one(&mut local_rng));
        }
        Ok(out)
    }
}

/// Precomputed step-`k` pure Gaussian data for the Quesada–Arrazola PNR chain rule.
struct PnrStepData {
    /// Gain matrix `K^{(k)} = T_{AC} (T_{CC} + I)^{-1}` of shape `2(k+1) × 2(m-1-k)`
    /// (row-major, xxpp ordering on A = 0..=k and C = k+1..m).
    gain: Vec<f64>,
    n_c: usize,
    /// `σ_Q^{-1}` of the pure `(k+1)`-mode conditional state (`2(k+1) × 2(k+1)`).
    q_inv: Vec<C64>,
    /// Pure symmetric `(k+1) × (k+1)` matrix `B^{(k)}` (bottom-right block of `A^{(k)}`).
    b_mat: Vec<C64>,
    /// Mean photon variance proxy of mode `k` in `V^{(k)}` for adaptive stopping.
    var_k: f64,
}

/// Draw `shots` exact photon-number-resolving (PNR) GBS samples (`Vec<usize>`
/// of length `state.n_modes()`) from any Gaussian state (pure, mixed, lossy, or
/// displaced) via the Quesada–Arrazola (2020) heterodyne-conditioned pure-state
/// chain rule.
///
/// `max_photons` caps the per-mode photon number sampled at each step
/// (if `None`, chosen automatically from the state's maximum mode mean photon
/// number).
pub fn sample_gbs_pnr<R: Rng + ?Sized>(
    state: &GaussianState,
    max_photons: Option<usize>,
    shots: usize,
    rng: &mut R,
) -> Result<Vec<Vec<usize>>, Error> {
    let m = state.n_modes();
    if shots == 0 {
        return Ok(Vec::new());
    }
    if m == 0 {
        return Ok(vec![Vec::new(); shots]);
    }

    // Order modes by increasing mean photon number so early modes stay mostly
    // at 0 photons (hitting the O(n) 1-mode Hermite fast path!).
    let mut order: Vec<usize> = (0..m).collect();
    let mode_means: Vec<f64> = (0..m).map(|i| state.mean_photons(i)).collect();
    order.sort_by(|&a, &b| mode_means[a].total_cmp(&mode_means[b]));

    let ordered_state = state.reduced(&order)?;
    let max_nbar = mode_means.iter().copied().fold(0.0f64, f64::max);
    let cutoff = max_photons.unwrap_or_else(|| {
        ((max_nbar + 6.0 * (max_nbar + 1.0).sqrt() + 8.0).ceil() as usize).max(16)
    });

    // Decompose V = T + W where T is pure (det T = 1) and W ≽ 0.
    let dim = 2 * m;
    let (t_cov, w_chol) = if ordered_state.purity() > 1.0 - 1e-10 {
        (ordered_state.cov().to_vec(), None)
    } else {
        let (t, w) = williamson_pure_decomposition(ordered_state.cov(), m)?;
        let lw = cholesky_real_psd(&w, dim);
        (t, Some(lw))
    };

    let base_mu = ordered_state.means().to_vec();

    // Precompute Cholesky factor for heterodyne on modes 1..m (xxpp ordering of C0).
    let n_het = m - 1;
    let het_chol = if n_het > 0 {
        let h_dim = 2 * n_het;
        let c_idx = xxpp_subset_indices(m, 1..m);
        let mut h_cov = vec![0.0; h_dim * h_dim];
        for r in 0..h_dim {
            for c in 0..h_dim {
                let mut v = t_cov[c_idx[r] * dim + c_idx[c]];
                if r == c {
                    v += 1.0;
                }
                h_cov[r * h_dim + c] = v;
            }
        }
        cholesky_real_psd(&h_cov, h_dim)
    } else {
        Vec::new()
    };

    // Precompute conditional pure Gaussian states for each step k = 0..m.
    let mut steps = Vec::with_capacity(m);
    for k in 0..m {
        steps.push(build_pnr_step(&t_cov, m, k)?);
    }

    // Precompute 1 / n! up to cutoff.
    let mut inv_fact = vec![1.0f64; cutoff + 1];
    for n in 1..=cutoff {
        inv_fact[n] = inv_fact[n - 1] / (n as f64);
    }

    let draw_one = |local_rng: &mut StdRng| -> Result<Vec<usize>, Error> {
        // Step 0: draw pure displacement μ_shot = μ + L_W z_w (if mixed).
        let mut mu_shot = base_mu.clone();
        if let Some(lw) = &w_chol {
            let mut z = vec![0.0; dim];
            for zi in &mut z {
                *zi = StandardNormal.sample(local_rng);
            }
            for (r, slot) in mu_shot.iter_mut().enumerate() {
                let mut add = 0.0;
                let row = r * dim;
                for (c, &zc) in z[..=r].iter().enumerate() {
                    add += lw[row + c] * zc;
                }
                *slot += add;
            }
        }

        // Step 1: draw initial heterodyne vector on modes 1..m (xxpp ordering).
        let h_dim = 2 * n_het;
        let mut r_het = vec![0.0; h_dim];
        if n_het > 0 {
            let mut z = vec![0.0; h_dim];
            for zi in &mut z {
                *zi = StandardNormal.sample(local_rng);
            }
            // Mean of modes 1..m in xxpp: x_1..x_{m-1}, p_1..p_{m-1}.
            r_het[..n_het].copy_from_slice(&mu_shot[1..1 + n_het]);
            r_het[n_het..].copy_from_slice(&mu_shot[m + 1..m + 1 + n_het]);
            for (r, slot) in r_het.iter_mut().enumerate() {
                let mut add = 0.0;
                let row = r * h_dim;
                for (c, &zc) in z[..=r].iter().enumerate() {
                    add += het_chol[row + c] * zc;
                }
                *slot += add;
            }
        }

        // Step 2: sample n_0, ..., n_{m-1} sequentially!
        let mut sampled = vec![0usize; m];
        let mut n_prev = 0usize;
        let mut probs = Vec::with_capacity(cutoff + 1);

        for k in 0..m {
            let sd = &steps[k];
            let a_modes = k + 1;
            let a_dim = 2 * a_modes;

            // Compute conditional mean μ^{(k)} on modes 0..=k.
            let mut mu_k = vec![0.0; a_dim];
            mu_k[..a_modes].copy_from_slice(&mu_shot[..a_modes]);
            mu_k[a_modes..].copy_from_slice(&mu_shot[m..m + a_modes]);
            if sd.n_c > 0 {
                let c_dim = 2 * sd.n_c;
                // Δ_C = r_het[k+1..m] - μ_shot[k+1..m] in xxpp ordering.
                let mut delta_c = vec![0.0; c_dim];
                for i in 0..sd.n_c {
                    let mode_i = k + 1 + i; // in 1..m -> index in r_het is (mode_i - 1)
                    let h_idx = mode_i - 1;
                    delta_c[i] = r_het[h_idx] - mu_shot[mode_i];
                    delta_c[sd.n_c + i] = r_het[n_het + h_idx] - mu_shot[m + mode_i];
                }
                for (r, slot) in mu_k.iter_mut().enumerate() {
                    let mut acc = 0.0;
                    let row = r * c_dim;
                    for (c, &dc) in delta_c.iter().enumerate() {
                        acc += sd.gain[row + c] * dc;
                    }
                    *slot += acc;
                }
            }

            // Compute γ1 = (σ_Q^{-1} α)[0..a_modes] matching B^{(k)}.
            let mut alpha = vec![C64::new(0.0, 0.0); a_dim];
            for i in 0..a_modes {
                let a = C64::new(0.5 * mu_k[i], 0.5 * mu_k[a_modes + i]);
                alpha[i] = a;
                alpha[a_modes + i] = a.conj();
            }
            let mut gamma1 = vec![C64::new(0.0, 0.0); a_modes];
            for (i, slot) in gamma1.iter_mut().enumerate() {
                let mut acc = C64::new(0.0, 0.0);
                let row = i * a_dim;
                for (j, &aj) in alpha.iter().enumerate() {
                    acc += sd.q_inv[row + j] * aj;
                }
                *slot = acc;
            }

            probs.clear();
            let mut sum_w = 0.0f64;

            if n_prev == 0 {
                // Fast path: 1-mode Hermite recurrence in O(cutoff) (~50 ns)!
                let b_kk = sd.b_mat[k * a_modes + k];
                let g_k = gamma1[k];
                let mut h_prev2 = C64::new(0.0, 0.0);
                let mut h_prev1 = C64::new(1.0, 0.0);
                probs.push(1.0);
                sum_w += 1.0;
                for (nk, &inv_f) in inv_fact.iter().enumerate().skip(1) {
                    let h = if nk == 1 {
                        g_k
                    } else {
                        g_k * h_prev1 + b_kk * (((nk - 1) as f64) * h_prev2)
                    };
                    h_prev2 = h_prev1;
                    h_prev1 = h;
                    let w = h.norm_sqr() * inv_f;
                    probs.push(w);
                    sum_w += w;
                }
            } else {
                // General active-mode loop hafnian over modes {j < k : sampled[j] > 0} ∪ {k}.
                let mut active: Vec<usize> = (0..k).filter(|&j| sampled[j] > 0).collect();
                active.push(k);
                let ka = active.len();
                let mut b_sub = vec![C64::new(0.0, 0.0); ka * ka];
                let mut g_sub = vec![C64::new(0.0, 0.0); ka];
                for (ri, &mr) in active.iter().enumerate() {
                    g_sub[ri] = gamma1[mr];
                    for (ci, &mc) in active.iter().enumerate() {
                        b_sub[ri * ka + ci] = sd.b_mat[mr * a_modes + mc];
                    }
                }
                let mut rep: Vec<usize> = active.iter().map(|&j| sampled[j]).collect();
                // Estimate mode k conditional mean photons for safe adaptive stopping.
                let mu_sq = mu_k[k] * mu_k[k] + mu_k[a_modes + k] * mu_k[a_modes + k];
                let est_nk = (0.25 * (sd.var_k + mu_sq - 2.0)).max(0.0);
                let min_stop = ((2.0 * est_nk + 6.0).ceil() as usize).min(cutoff);

                for nk in 0..=cutoff {
                    rep[ka - 1] = nk;
                    let h = loop_hafnian_repeated(&b_sub, &g_sub, &rep)?;
                    let w = h.norm_sqr() * inv_fact[nk];
                    probs.push(w);
                    sum_w += w;
                    if nk >= min_stop && nk >= 2 && sum_w > 0.0 {
                        let tail = probs[nk].max(probs[nk - 1]);
                        if tail < 1e-10 * sum_w {
                            break;
                        }
                    }
                }
            }

            let u = local_rng.random::<f64>() * sum_w;
            let mut cum = 0.0;
            let mut chosen = probs.len() - 1;
            for (nk, &w) in probs.iter().enumerate() {
                cum += w;
                if u < cum {
                    chosen = nk;
                    break;
                }
            }
            sampled[k] = chosen;
            n_prev += chosen;
        }

        // Un-permute back to original mode ordering.
        let mut out = vec![0usize; m];
        for (pos, &orig_mode) in order.iter().enumerate() {
            out[orig_mode] = sampled[pos];
        }
        Ok(out)
    };

    if shots >= GBS_PAR_MIN_SHOTS {
        let seeds: Vec<u64> = (0..shots).map(|_| rng.random()).collect();
        seeds
            .into_par_iter()
            .map(|seed| {
                let mut local_rng = StdRng::seed_from_u64(seed);
                draw_one(&mut local_rng)
            })
            .collect()
    } else {
        let mut out = Vec::with_capacity(shots);
        for _ in 0..shots {
            let mut local_rng = StdRng::seed_from_u64(rng.random());
            out.push(draw_one(&mut local_rng)?);
        }
        Ok(out)
    }
}

fn xxpp_subset_indices(m: usize, range: std::ops::Range<usize>) -> Vec<usize> {
    let mut idx = Vec::with_capacity(2 * range.len());
    for i in range.clone() {
        idx.push(i);
    }
    for i in range {
        idx.push(m + i);
    }
    idx
}

fn build_pnr_step(t_cov: &[f64], m: usize, k: usize) -> Result<PnrStepData, Error> {
    let dim = 2 * m;
    let a_modes = k + 1;
    let a_dim = 2 * a_modes;
    let n_c = m - a_modes;
    let a_idx = xxpp_subset_indices(m, 0..a_modes);

    let (v_cond, gain) = if n_c == 0 {
        let mut v = vec![0.0; a_dim * a_dim];
        for r in 0..a_dim {
            for c in 0..a_dim {
                v[r * a_dim + c] = t_cov[a_idx[r] * dim + a_idx[c]];
            }
        }
        (v, Vec::new())
    } else {
        let c_dim = 2 * n_c;
        let c_idx = xxpp_subset_indices(m, a_modes..m);
        let t_aa = Mat::from_fn(a_dim, a_dim, |r, c| t_cov[a_idx[r] * dim + a_idx[c]]);
        let t_ac = Mat::from_fn(a_dim, c_dim, |r, c| t_cov[a_idx[r] * dim + c_idx[c]]);
        let t_cc_p1 = Mat::from_fn(c_dim, c_dim, |r, c| {
            t_cov[c_idx[r] * dim + c_idx[c]] + if r == c { 1.0 } else { 0.0 }
        });
        let inv_cc = invert_spd_real(&t_cc_p1);
        let k_mat = &t_ac * &inv_cc;
        let v_mat = &t_aa - &k_mat * t_ac.transpose();
        let mut v = vec![0.0; a_dim * a_dim];
        for r in 0..a_dim {
            for c in 0..a_dim {
                v[r * a_dim + c] = 0.5 * (v_mat[(r, c)] + v_mat[(c, r)]);
            }
        }
        let mut gain = vec![0.0; a_dim * c_dim];
        for r in 0..a_dim {
            for c in 0..c_dim {
                gain[r * c_dim + c] = k_mat[(r, c)];
            }
        }
        (v, gain)
    };

    let var_k = v_cond[k * a_dim + k] + v_cond[(a_modes + k) * a_dim + (a_modes + k)];

    // Build σ_Q = T V_cond T† + 0.5 I, invert via Hermitian Cholesky, extract B^{(k)}.
    let mut sigma_q = vec![C64::new(0.0, 0.0); a_dim * a_dim];
    for i in 0..a_modes {
        for j in 0..a_modes {
            let vxx = v_cond[i * a_dim + j];
            let vxp = v_cond[i * a_dim + (a_modes + j)];
            let vpx = v_cond[(a_modes + i) * a_dim + j];
            let vpp = v_cond[(a_modes + i) * a_dim + (a_modes + j)];
            let mut q00 = C64::new(0.25 * (vxx + vpp), 0.25 * (vpx - vxp));
            if i == j {
                q00.re += 0.5;
            }
            let q01 = C64::new(0.25 * (vxx - vpp), 0.25 * (vxp + vpx));
            sigma_q[i * a_dim + j] = q00;
            sigma_q[i * a_dim + (a_modes + j)] = q01;
            sigma_q[(a_modes + i) * a_dim + j] = q01.conj();
            sigma_q[(a_modes + i) * a_dim + (a_modes + j)] = q00.conj();
        }
    }
    let q_inv = invert_herm_pd(&sigma_q, a_dim).ok_or_else(|| {
        Error::InvalidArgument("conditional Husimi Q covariance is not positive definite".into())
    })?;

    // B^{(k)} is the bottom-right a_modes × a_modes block of A = X(I - σ_Q^{-1}):
    // row (a_modes + i) of A comes from row i of (I - σ_Q^{-1}), cols (a_modes + j).
    // Since i != a_modes + j, B[i, j] = -q_inv[i, a_modes + j]!
    let mut b_mat = vec![C64::new(0.0, 0.0); a_modes * a_modes];
    for i in 0..a_modes {
        for j in 0..a_modes {
            let v = -q_inv[i * a_dim + (a_modes + j)];
            b_mat[i * a_modes + j] = v;
        }
    }
    // Enforce exact symmetry B^T = B.
    for i in 0..a_modes {
        for j in (i + 1)..a_modes {
            let avg = 0.5 * (b_mat[i * a_modes + j] + b_mat[j * a_modes + i]);
            b_mat[i * a_modes + j] = avg;
            b_mat[j * a_modes + i] = avg;
        }
    }

    Ok(PnrStepData {
        gain,
        n_c,
        q_inv,
        b_mat,
        var_k,
    })
}

/// Williamson pure-state decomposition `V = T + W` where `T = S S^T` is a pure
/// Gaussian covariance matrix (`det T = 1`) and `W = V - T ≽ 0`.
fn williamson_pure_decomposition(cov: &[f64], m: usize) -> Result<(Vec<f64>, Vec<f64>), Error> {
    let dim = 2 * m;
    let v_mat = Mat::from_fn(dim, dim, |r, c| cov[r * dim + c]);
    let eig = v_mat
        .self_adjoint_eigen(Side::Lower)
        .map_err(|_| Error::NotConverged("Williamson eigendecomposition failed".into()))?;
    let u = eig.U();
    let s = eig.S().column_vector();

    let mut v_sqrt = Mat::<f64>::zeros(dim, dim);
    let mut v_inv_sqrt = Mat::<f64>::zeros(dim, dim);
    for i in 0..dim {
        for j in 0..dim {
            let mut a = 0.0;
            let mut b = 0.0;
            for k in 0..dim {
                let lam = s[k].max(1e-14);
                let prod = u[(i, k)] * u[(j, k)];
                a += prod * lam.sqrt();
                b += prod / lam.sqrt();
            }
            v_sqrt[(i, j)] = a;
            v_inv_sqrt[(i, j)] = b;
        }
    }

    // K = V^{-1/2} Ω V^{-1/2} (real skew-symmetric); H = i K is Hermitian with
    // eigenvalue pairs ±(1 / d_j).
    let mut omega = Mat::<f64>::zeros(dim, dim);
    for i in 0..m {
        omega[(i, m + i)] = 1.0;
        omega[(m + i, i)] = -1.0;
    }
    let k_real = &v_inv_sqrt * &omega * &v_inv_sqrt;
    let h_herm = Mat::from_fn(dim, dim, |r, c| C64::new(0.0, k_real[(r, c)]));
    let heig = h_herm.self_adjoint_eigen(Side::Lower).map_err(|_| {
        Error::NotConverged("Williamson symplectic eigendecomposition failed".into())
    })?;
    let hu = heig.U();
    let hs = heig.S().column_vector();

    // Eigenvalues of H = i K are sorted ascending: the top m eigenvalues are positive (+1 / d_j).
    // If H w_j = +λ_j w_j with w_j = (u_j + i v_j) / √2, then K u_j = λ_j v_j, K v_j = -λ_j u_j.
    // Note: T = V^{1/2} O D^{-1} O^T V^{1/2} where D^{-1} = diag(λ_0..λ_{m-1}, λ_0..λ_{m-1})
    // and the column space of [u_j, v_j] has projector 2 Re(w_j w_j†)!
    // So O D^{-1} O^T = Σ_{j: λ_j > 0} 2 λ_j Re(w_j w_j†) = |H| (the Hermitian absolute value of H)!!
    // WAIT: verify! For real skew-symmetric K, H = i K has eigenvalues ±λ_j with conjugate eigenvectors
    // w_j and w_j*, so |H| = Σ_j λ_j (w_j w_j† + w_j* w_j^T) = Σ_j 2 λ_j Re(w_j w_j†), which is REAL SYMMETRIC!!
    // And T = V^{1/2} |i V^{-1/2} Ω V^{-1/2}| V^{1/2}!!
    // Check 1-mode thermal V = d I_2: |i d^{-1} Ω| = d^{-1} I_2 => T = d^{1/2} d^{-1} d^{1/2} I_2 = I_2!
    // Check 1-mode pure V = diag(e^{-2r}, e^{2r}): V^{-1/2} Ω V^{-1/2} = Ω => |i Ω| = I_2 => T = V!
    let mut abs_h = Mat::<f64>::zeros(dim, dim);
    for k in m..dim {
        let lam = hs[k].re.abs().min(1.0); // physical state has d_j >= 1 => λ_j <= 1
        for i in 0..dim {
            for j in 0..dim {
                abs_h[(i, j)] += 2.0 * lam * (hu[(i, k)] * hu[(j, k)].conj()).re;
            }
        }
    }

    let t_mat = &v_sqrt * &abs_h * &v_sqrt;
    let mut t_vec = vec![0.0; dim * dim];
    let mut w_vec = vec![0.0; dim * dim];
    for r in 0..dim {
        for c in 0..dim {
            let trc = 0.5 * (t_mat[(r, c)] + t_mat[(c, r)]);
            t_vec[r * dim + c] = trc;
            w_vec[r * dim + c] = cov[r * dim + c] - trc;
        }
    }
    Ok((t_vec, w_vec))
}

fn invert_spd_real(a: &Mat<f64>) -> Mat<f64> {
    let n = a.nrows();
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..i {
            let mut s = a[(i, j)];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = s / l[j * n + j];
        }
        let mut diag = a[(i, i)];
        for k in 0..i {
            let v = l[i * n + k];
            diag -= v * v;
        }
        l[i * n + i] = diag.max(1e-15).sqrt();
    }
    let mut linv = vec![0.0; n * n];
    for j in 0..n {
        linv[j * n + j] = 1.0 / l[j * n + j];
        for i in (j + 1)..n {
            let mut s = 0.0;
            for k in j..i {
                s -= l[i * n + k] * linv[k * n + j];
            }
            linv[i * n + j] = s / l[i * n + i];
        }
    }
    Mat::from_fn(n, n, |i, j| {
        let mut s = 0.0;
        let start = i.max(j);
        for k in start..n {
            s += linv[k * n + i] * linv[k * n + j];
        }
        s
    })
}

fn cholesky_real_psd(a: &[f64], dim: usize) -> Vec<f64> {
    let mut l = vec![0.0; dim * dim];
    for i in 0..dim {
        for j in 0..i {
            let dj = l[j * dim + j];
            if dj <= 1e-14 {
                l[i * dim + j] = 0.0;
                continue;
            }
            let mut s = a[i * dim + j];
            for k in 0..j {
                s -= l[i * dim + k] * l[j * dim + k];
            }
            l[i * dim + j] = s / dj;
        }
        let mut diag = a[i * dim + i];
        for k in 0..i {
            let v = l[i * dim + k];
            diag -= v * v;
        }
        l[i * dim + i] = diag.max(0.0).sqrt();
    }
    l
}

fn invert_herm_pd(a: &[C64], dim: usize) -> Option<Vec<C64>> {
    let mut l = vec![C64::new(0.0, 0.0); dim * dim];
    for i in 0..dim {
        for j in 0..i {
            let mut s = a[i * dim + j];
            for k in 0..j {
                s -= l[i * dim + k] * l[j * dim + k].conj();
            }
            l[i * dim + j] = s / l[j * dim + j].re;
        }
        let mut diag = a[i * dim + i].re;
        for k in 0..i {
            diag -= l[i * dim + k].norm_sqr();
        }
        if diag <= 0.0 || !diag.is_finite() {
            return None;
        }
        l[i * dim + i] = C64::new(diag.sqrt(), 0.0);
    }
    let mut linv = vec![C64::new(0.0, 0.0); dim * dim];
    for j in 0..dim {
        linv[j * dim + j] = C64::new(1.0 / l[j * dim + j].re, 0.0);
        for i in (j + 1)..dim {
            let mut s = C64::new(0.0, 0.0);
            for k in j..i {
                s -= l[i * dim + k] * linv[k * dim + j];
            }
            linv[i * dim + j] = s / l[i * dim + i].re;
        }
    }
    let mut inv = vec![C64::new(0.0, 0.0); dim * dim];
    for i in 0..dim {
        for j in i..dim {
            let mut s = C64::new(0.0, 0.0);
            for k in j..dim {
                s += linv[k * dim + i].conj() * linv[k * dim + j];
            }
            inv[i * dim + j] = s;
            inv[j * dim + i] = s.conj();
        }
    }
    Some(inv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cv::{CvChannel, ModeCircuit};
    use std::collections::HashMap;

    #[test]
    fn gbs_threshold_sampler_matches_exact_tvd_pure_and_mixed() {
        let mut c = ModeCircuit::new(3);
        c.squeeze(0.5, 0.2, 0)
            .squeeze(0.4, -0.5, 1)
            .squeeze(0.35, 0.8, 2)
            .beam_splitter(0.7, 0.3, 0, 1)
            .beam_splitter(0.6, -0.4, 1, 2)
            .displace(0.3, 0.5, 0);

        let pure = GaussianState::from_circuit(&c).unwrap();
        let mut mixed = pure.clone();
        mixed
            .apply_channel(
                &CvChannel::ThermalLoss {
                    eta: 0.75,
                    mean_photons: 0.2,
                },
                1,
            )
            .unwrap();

        let mut rng = StdRng::seed_from_u64(42);
        let shots = 20_000;
        for (name, st) in [("pure", &pure), ("mixed", &mixed)] {
            let samples = sample_gbs_threshold(st, shots, &mut rng).unwrap();
            let mut counts: HashMap<Vec<bool>, usize> = HashMap::new();
            for s in samples {
                *counts.entry(s).or_insert(0) += 1;
            }
            let mut tvd = 0.0;
            for mask in 0..8usize {
                let pat = vec![mask & 1 != 0, (mask >> 1) & 1 != 0, (mask >> 2) & 1 != 0];
                let p_exact = st.threshold_probability(&pat).unwrap();
                let p_emp = (*counts.get(&pat).unwrap_or(&0) as f64) / (shots as f64);
                tvd += (p_emp - p_exact).abs();
            }
            tvd *= 0.5;
            assert!(tvd < 0.02, "GBS threshold ({name}) TVD={tvd}");
        }
    }

    #[test]
    fn gbs_pnr_sampler_matches_exact_tvd_pure_displaced_and_mixed() {
        // 2-mode squeezed + displaced + BS (pure) AND with thermal loss (mixed).
        let mut c = ModeCircuit::new(2);
        c.squeeze(0.45, 0.3, 0)
            .squeeze(0.35, -0.6, 1)
            .beam_splitter(0.65, 0.4, 0, 1)
            .displace(0.35, -0.2, 0);

        let pure = GaussianState::from_circuit(&c).unwrap();
        let mut mixed = pure.clone();
        mixed
            .apply_channel(
                &CvChannel::ThermalLoss {
                    eta: 0.7,
                    mean_photons: 0.25,
                },
                0,
            )
            .unwrap();

        let mut rng = StdRng::seed_from_u64(99);
        let shots = 12_000;
        for (name, st) in [("pure", &pure), ("mixed", &mixed)] {
            let samples = sample_gbs_pnr(st, Some(10), shots, &mut rng).unwrap();
            let mut counts: HashMap<Vec<usize>, usize> = HashMap::new();
            for s in samples {
                *counts.entry(s).or_insert(0) += 1;
            }
            let mut tvd = 0.0;
            for n0 in 0..5 {
                for n1 in 0..5 {
                    let pat = vec![n0, n1];
                    let p_exact = st.fock_probability(&pat).unwrap();
                    let p_emp = (*counts.get(&pat).unwrap_or(&0) as f64) / (shots as f64);
                    tvd += (p_emp - p_exact).abs();
                }
            }
            tvd *= 0.5;
            assert!(tvd < 0.025, "GBS PNR ({name}) TVD={tvd}");
        }
    }
}
