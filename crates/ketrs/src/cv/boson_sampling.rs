//! Exact boson sampling via Clifford & Clifford (2018 / 2020) with simultaneous
//! Glynn Gray-code Laplace permanents, plus a standalone Glynn `permanent`.
//!
//! Complexity per sample for `N` photons in `m` modes: `O(N 2^N + m N²)`,
//! roughly `2×` the cost of a single `N × N` permanent, with zero divisions
//! and linear multi-shot scaling across rayon threads.

use crate::error::Error;
use num_complex::Complex64 as C64;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

/// Minimum matrix size `n` for parallelising a single permanent over Gray-code chunks.
const PERM_PAR_MIN_N: usize = 16;

/// Minimum shot count for parallelising independent boson samples across threads.
const SAMPLE_PAR_MIN_SHOTS: usize = 4;

/// Exact permanent of an `n × n` row-major complex matrix via Glynn's formula
/// with Gray-code column-sum updates (`O(n 2^{n-1})`).
pub fn permanent(matrix: &[C64], n: usize) -> Result<C64, Error> {
    if matrix.len() != n * n {
        return Err(Error::InvalidArgument(format!(
            "permanent expects {n}×{n} ({}) entries, got {}",
            n * n,
            matrix.len()
        )));
    }
    match n {
        0 => Ok(C64::new(1.0, 0.0)),
        1 => Ok(matrix[0]),
        2 => Ok(matrix[0] * matrix[3] + matrix[1] * matrix[2]),
        _ => Ok(glynn_permanent(matrix, n)),
    }
}

fn glynn_permanent(a: &[C64], n: usize) -> C64 {
    // Glynn's formula: perm(A) = 2^{-(n-1)} Σ_{δ_0=+1, δ_1..δ_{n-1} ∈ {±1}} (∏ δ_r) ∏_c (Σ_r δ_r A_{r, c}).
    let n_bits = n - 1;
    let total: usize = 1usize << n_bits;

    // Precompute 2 * A rows for Gray-code updates.
    let two_a: Vec<C64> = a.iter().map(|&z| z * 2.0).collect();

    let eval_range = |start: usize, end: usize| -> C64 {
        let mut s = vec![C64::new(0.0, 0.0); n];
        let mut gray = start ^ (start >> 1);
        let mut sign = if (gray.count_ones() as usize).is_multiple_of(2) {
            1.0f64
        } else {
            -1.0f64
        };
        // Initialize s from row 0 (+1) and rows 1..n according to `gray`.
        s.copy_from_slice(&a[0..n]);
        for r in 1..n {
            let bit = (gray >> (r - 1)) & 1;
            let row = &a[r * n..(r + 1) * n];
            if bit == 0 {
                for (sj, &val) in s.iter_mut().zip(row) {
                    *sj += val;
                }
            } else {
                for (sj, &val) in s.iter_mut().zip(row) {
                    *sj -= val;
                }
            }
        }

        let mut acc = C64::new(0.0, 0.0);
        for step in start..end {
            let mut prod = C64::new(sign, 0.0);
            for &sj in &s {
                prod *= sj;
            }
            acc += prod;

            let next = step + 1;
            if next < end {
                let next_gray = next ^ (next >> 1);
                let diff = gray ^ next_gray;
                let bit_idx = diff.trailing_zeros() as usize;
                let row = &two_a[(bit_idx + 1) * n..(bit_idx + 2) * n];
                if (next_gray >> bit_idx) & 1 == 1 {
                    // Flipped +1 -> -1
                    for (sj, &v2) in s.iter_mut().zip(row) {
                        *sj -= v2;
                    }
                } else {
                    // Flipped -1 -> +1
                    for (sj, &v2) in s.iter_mut().zip(row) {
                        *sj += v2;
                    }
                }
                gray = next_gray;
                sign = -sign;
            }
        }
        acc
    };

    let sum = if n >= PERM_PAR_MIN_N {
        let n_chunks = (rayon::current_num_threads() * 4).max(8).min(total);
        let chunk_sz = total.div_ceil(n_chunks);
        (0..n_chunks)
            .into_par_iter()
            .map(|c| {
                let start = c * chunk_sz;
                let end = (start + chunk_sz).min(total);
                if start < end {
                    eval_range(start, end)
                } else {
                    C64::new(0.0, 0.0)
                }
            })
            .reduce(|| C64::new(0.0, 0.0), |x, y| x + y)
    } else {
        eval_range(0, total)
    };

    sum * 0.5f64.powi(n_bits as i32)
}

/// Simultaneous computation of all `k` complementary `(k-1) × (k-1)` permanents
/// of the `(k - 1) × k` row-major matrix `b` (for `k >= 2`):
///
/// `out[j] = perm(b[:, [k] \ {j}])` (scaled uniformly by `2^{k - 2}`, which
/// cancels on normalisation, or exact when multiplied by `2^{-(k - 2)}`).
///
/// Uses a single `2^{k-2}` Glynn Gray-code pass with division-free prefix/suffix
/// products (`O(k 2^{k-1})` total flops for all `k` permanents combined).
fn simultaneous_laplace_permanents(b: &[C64], k: usize, out: &mut [C64], par_single: bool) {
    debug_assert!(k >= 2);
    debug_assert_eq!(b.len(), (k - 1) * k);
    debug_assert_eq!(out.len(), k);

    if k == 2 {
        out[0] = b[1];
        out[1] = b[0];
        return;
    }

    let n_rows = k - 1;
    let n_bits = n_rows - 1; // = k - 2
    let total: usize = 1usize << n_bits;
    let two_b: Vec<C64> = b.iter().map(|&z| z * 2.0).collect();

    let eval_range = |start: usize, end: usize| -> Vec<C64> {
        let mut s = vec![C64::new(0.0, 0.0); k];
        let mut prefix = vec![C64::new(0.0, 0.0); k];
        let mut local_out = vec![C64::new(0.0, 0.0); k];

        let mut gray = start ^ (start >> 1);
        let mut sign = if (gray.count_ones() as usize).is_multiple_of(2) {
            1.0f64
        } else {
            -1.0f64
        };
        s.copy_from_slice(&b[0..k]);
        for r in 1..n_rows {
            let bit = (gray >> (r - 1)) & 1;
            let row = &b[r * k..(r + 1) * k];
            if bit == 0 {
                for (sj, &v) in s.iter_mut().zip(row) {
                    *sj += v;
                }
            } else {
                for (sj, &v) in s.iter_mut().zip(row) {
                    *sj -= v;
                }
            }
        }

        for step in start..end {
            // Prefix products: prefix[j] = sign * ∏_{t < j} s[t].
            let mut p = C64::new(sign, 0.0);
            for j in 0..k {
                prefix[j] = p;
                p *= s[j];
            }
            // Suffix sweep right-to-left: accumulate prefix[j] * suffix[j] into local_out[j].
            let mut suf = C64::new(1.0, 0.0);
            for j in (0..k).rev() {
                local_out[j] += prefix[j] * suf;
                suf *= s[j];
            }

            let next = step + 1;
            if next < end {
                let next_gray = next ^ (next >> 1);
                let diff = gray ^ next_gray;
                let bit_idx = diff.trailing_zeros() as usize;
                let row = &two_b[(bit_idx + 1) * k..(bit_idx + 2) * k];
                if (next_gray >> bit_idx) & 1 == 1 {
                    for (sj, &v2) in s.iter_mut().zip(row) {
                        *sj -= v2;
                    }
                } else {
                    for (sj, &v2) in s.iter_mut().zip(row) {
                        *sj += v2;
                    }
                }
                gray = next_gray;
                sign = -sign;
            }
        }
        local_out
    };

    let scale = 0.5f64.powi(n_bits as i32);
    if par_single && k >= PERM_PAR_MIN_N {
        let n_chunks = (rayon::current_num_threads() * 4).max(8).min(total);
        let chunk_sz = total.div_ceil(n_chunks);
        let summed = (0..n_chunks)
            .into_par_iter()
            .map(|c| {
                let start = c * chunk_sz;
                let end = (start + chunk_sz).min(total);
                if start < end {
                    eval_range(start, end)
                } else {
                    vec![C64::new(0.0, 0.0); k]
                }
            })
            .reduce(
                || vec![C64::new(0.0, 0.0); k],
                |mut a, b| {
                    for (x, y) in a.iter_mut().zip(b) {
                        *x += y;
                    }
                    a
                },
            );
        for (dst, val) in out.iter_mut().zip(summed) {
            *dst = val * scale;
        }
    } else {
        let summed = eval_range(0, total);
        for (dst, val) in out.iter_mut().zip(summed) {
            *dst = val * scale;
        }
    }
}

/// Draw `shots` exact boson-sampling output occupation vectors from an `m × m`
/// row-major unitary `unitary` and Fock input `input` using Clifford & Clifford
/// (2018 / 2020) with simultaneous Glynn Gray-code Laplace permanents.
///
/// Works for both collision-free (`input[j] ∈ {0, 1}`) and multi-photon inputs.
pub fn boson_sample<R: Rng + ?Sized>(
    unitary: &[C64],
    input: &[usize],
    shots: usize,
    rng: &mut R,
) -> Result<Vec<Vec<usize>>, Error> {
    let m = input.len();
    if m == 0 {
        return Err(Error::InvalidArgument(
            "boson_sample requires at least one mode".into(),
        ));
    }
    if unitary.len() != m * m {
        return Err(Error::InvalidArgument(format!(
            "unitary on {m} modes needs {} entries, got {}",
            m * m,
            unitary.len()
        )));
    }
    let n: usize = input.iter().sum();
    if shots == 0 {
        return Ok(Vec::new());
    }
    if n == 0 {
        return Ok(vec![vec![0; m]; shots]);
    }

    // Build the m × N input-column matrix `a_cols` stored column-major for
    // cache-friendly column permutation, plus row-major view per shot.
    let mut in_modes = Vec::with_capacity(n);
    for (mode, &count) in input.iter().enumerate() {
        for _ in 0..count {
            in_modes.push(mode);
        }
    }

    let draw_one = |local_rng: &mut StdRng, par_single: bool| -> Vec<usize> {
        // Random permutation of the N input photons' columns.
        let mut perm_modes = in_modes.clone();
        perm_modes.shuffle(local_rng);

        // Build row-major m × N submatrix `a` of the permuted columns.
        let mut a = vec![C64::new(0.0, 0.0); m * n];
        for r in 0..m {
            let u_row = r * m;
            let a_row = r * n;
            for (c, &in_m) in perm_modes.iter().enumerate() {
                a[a_row + c] = unitary[u_row + in_m];
            }
        }

        let mut sampled_modes = Vec::with_capacity(n);
        let mut weights = vec![0.0f64; m];
        let mut cofactors = vec![C64::new(0.0, 0.0); n];

        // Step k = 1: p_i = |a[i, 0]|².
        let mut total_w = 0.0;
        for i in 0..m {
            let w = a[i * n].norm_sqr();
            weights[i] = w;
            total_w += w;
        }
        let r0 = pick_discrete(&weights, total_w, local_rng);
        sampled_modes.push(r0);

        // Steps k = 2..=N.
        let mut b = Vec::with_capacity((n - 1) * n);
        for k in 2..=n {
            // Build (k - 1) × k matrix B from rows `sampled_modes[0..k-1]` and cols `0..k`.
            b.clear();
            for &row_mode in &sampled_modes {
                let off = row_mode * n;
                b.extend_from_slice(&a[off..off + k]);
            }

            // Compute all k Laplace cofactors simultaneously in one 2^{k-2} Gray-code pass.
            simultaneous_laplace_permanents(&b, k, &mut cofactors[..k], par_single);

            // p_i ∝ |Σ_{j=0}^{k-1} a[i, j] * cofactors[j]|².
            let mut sum_w = 0.0;
            for i in 0..m {
                let row = &a[i * n..i * n + k];
                let mut amp = C64::new(0.0, 0.0);
                for (j, &cj) in cofactors[..k].iter().enumerate() {
                    amp += row[j] * cj;
                }
                let w = amp.norm_sqr();
                weights[i] = w;
                sum_w += w;
            }
            let rk = pick_discrete(&weights, sum_w, local_rng);
            sampled_modes.push(rk);
        }

        let mut occ = vec![0usize; m];
        for mode in sampled_modes {
            occ[mode] += 1;
        }
        occ
    };

    if shots >= SAMPLE_PAR_MIN_SHOTS {
        let seeds: Vec<u64> = (0..shots).map(|_| rng.random()).collect();
        Ok(seeds
            .into_par_iter()
            .map(|seed| {
                let mut local_rng = StdRng::seed_from_u64(seed);
                draw_one(&mut local_rng, false)
            })
            .collect())
    } else {
        let mut out = Vec::with_capacity(shots);
        for _ in 0..shots {
            let mut local_rng = StdRng::seed_from_u64(rng.random());
            out.push(draw_one(&mut local_rng, true));
        }
        Ok(out)
    }
}

#[inline]
fn pick_discrete<R: Rng + ?Sized>(weights: &[f64], total: f64, rng: &mut R) -> usize {
    let u = rng.random::<f64>() * total;
    let mut cum = 0.0;
    for (i, &w) in weights.iter().enumerate() {
        cum += w;
        if u < cum {
            return i;
        }
    }
    weights.len() - 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cv::FixedNFockState;
    use std::collections::HashMap;
    use std::f64::consts::FRAC_PI_4;

    #[test]
    fn permanent_matches_small_analytic_and_ryser() {
        // 3×3 all-ones permanent = 3! = 6; 4×4 = 24; 5×5 = 120.
        for (n, want) in [
            (0, 1.0),
            (1, 1.0),
            (2, 2.0),
            (3, 6.0),
            (4, 24.0),
            (5, 120.0),
        ] {
            let a = vec![C64::new(1.0, 0.0); n * n];
            let got = permanent(&a, n).unwrap();
            assert!((got - C64::new(want, 0.0)).norm() < 1e-10);
        }

        // Complex 5×5 matrix: compare against Ryser.
        let mut rng = StdRng::seed_from_u64(17);
        let n = 5;
        let a: Vec<C64> = (0..n * n)
            .map(|_| C64::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)))
            .collect();
        let got = permanent(&a, n).unwrap();
        let mut ryser = C64::new(0.0, 0.0);
        for subset in 1..(1usize << n) {
            let prod: C64 = (0..n)
                .map(|r| {
                    (0..n)
                        .filter(|&c| (subset >> c) & 1 == 1)
                        .map(|c| a[r * n + c])
                        .sum::<C64>()
                })
                .product();
            let sign = if (n - subset.count_ones() as usize).is_multiple_of(2) {
                1.0
            } else {
                -1.0
            };
            ryser += prod * sign;
        }
        assert!((got - ryser).norm() < 1e-10);
    }

    #[test]
    fn boson_sampling_matches_exact_slos_distribution_tvd() {
        // 4 modes, 3 photons (both collision-free [1,1,1,0] and bunched [2,1,0,0])
        // on a fixed 4×4 unitary built from beam splitters + rotations.
        let m = 4;
        let mut u = vec![C64::new(0.0, 0.0); m * m];
        for i in 0..m {
            u[i * m + i] = C64::new(1.0, 0.0);
        }
        let apply_bs = |u: &mut [C64], a: usize, b: usize, theta: f64, phi: f64| {
            let t = C64::new(theta.cos(), 0.0);
            let r = C64::from_polar(theta.sin(), phi);
            for col in 0..m {
                let x = u[a * m + col];
                let y = u[b * m + col];
                u[a * m + col] = t * x - r.conj() * y;
                u[b * m + col] = r * x + t * y;
            }
        };
        apply_bs(&mut u, 0, 1, FRAC_PI_4, 0.3);
        apply_bs(&mut u, 2, 3, 0.6, -0.5);
        apply_bs(&mut u, 1, 2, 0.9, 0.7);
        apply_bs(&mut u, 0, 3, 0.4, -0.8);

        let mut rng = StdRng::seed_from_u64(2026);
        for input in [vec![1, 1, 1, 0], vec![2, 1, 0, 0]] {
            let exact = FixedNFockState::from_passive_unitary(&input, &u).unwrap();
            let shots = 20_000;
            let samples = boson_sample(&u, &input, shots, &mut rng).unwrap();
            let mut counts: HashMap<Vec<usize>, usize> = HashMap::new();
            for s in samples {
                *counts.entry(s).or_insert(0) += 1;
            }
            // Total variation distance: ½ Σ |p_emp - p_exact|.
            let mut tvd = 0.0;
            for pat in exact.basis() {
                let p_exact = exact.probability(&pat).unwrap();
                let p_emp = (*counts.get(&pat).unwrap_or(&0) as f64) / (shots as f64);
                tvd += (p_emp - p_exact).abs();
            }
            tvd *= 0.5;
            assert!(tvd < 0.02, "input={input:?} TVD={tvd}");
        }
    }
}
