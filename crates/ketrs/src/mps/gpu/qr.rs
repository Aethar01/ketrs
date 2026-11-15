//! The triangular factor of a tall matrix's QR decomposition on the GPU,
//! by blocked Householder QR as in LAPACK's `zgeqrf`.
//!
//! Each panel of [`PANEL`] columns is factorised a column at a time
//! (`mps_qr.comp`: the column's norm, its reflector, and the reflector
//! applied to the rest of the panel). The columns after the panel are then
//! updated by the panel's block reflector `I − V T V†` in three matrix
//! products (`mps_gemm.comp`). `T`, upper triangular and `PANEL` wide, is
//! formed on the CPU from `V† V` and the reflectors' `τ`, between the
//! panels' submissions. The result is the `R` of the CPU's Householder QR
//! up to rounding (and a unit phase per row).

use super::*;

/// Columns per panel.
pub(super) const PANEL: usize = 32;

/// Device scratch for [`qr_r`], grown as needed.
#[derive(Default)]
pub(super) struct QrScratch {
    /// Per-workgroup partial sums, up to 256 per panel column.
    partials: Option<Buffer>,
    /// `τ` and `σ` of the current reflector, then every column's `τ`.
    scalars: Option<Buffer>,
    /// `v† A` for the rest of the panel.
    w: Option<Buffer>,
    /// The panel's reflectors, column-major.
    v: Option<Buffer>,
    /// Partial products of split matrix products.
    parts: Option<Buffer>,
    /// `V† A` and `−T† V† A` for the columns after the panel.
    vta: Option<Buffer>,
    tvta: Option<Buffer>,
    /// `V† V`, on the device and copied back with the panel's `τ`.
    gram: Option<Buffer>,
    gram_down: Option<Buffer>,
    /// `−T†`, written by the CPU.
    t_up: Option<Buffer>,
    /// `R`, copied back.
    r_down: Option<Buffer>,
}

/// Workgroups for a reduction over `len` rows: at most 256, so one
/// workgroup can sum their results.
fn reduction_groups(len: usize) -> usize {
    len.div_ceil(256).clamp(1, 256)
}

/// `T` of the block reflector `H₁ ⋯ H_b = I − V T V†` (`zlarft`, forward,
/// columnwise) from `gram = V† V` and the reflectors' `τ`, all `b × b`
/// row-major.
fn block_t(gram: &[C64], tau: &[C64], b: usize) -> Vec<C64> {
    let zero = C64::new(0.0, 0.0);
    let mut t = vec![zero; b * b];
    for i in 0..b {
        t[i * b + i] = tau[i];
        let z: Vec<C64> = (0..i).map(|r| -tau[i] * gram[r * b + i]).collect();
        for r in 0..i {
            t[r * b + i] = (r..i).map(|q| t[r * b + q] * z[q]).sum();
        }
    }
    t
}

/// `R` (`min(m, n) × n`) of `A = Q R` for the `m × n` column-major matrix
/// at device address `a` in `a_buffer`, which is overwritten.
pub(super) fn qr_r(
    ctx: &Arc<GpuContext>,
    a: u64,
    a_buffer: vk::Buffer,
    m: usize,
    n: usize,
    s: &mut QrScratch,
) -> Result<Mat<C64>, Error> {
    let k_total = m.min(n);
    let max_splits = (m / MIN_SPLIT_K as usize).clamp(1, TARGET_GROUPS as usize);
    let width = n.max(PANEL) as u64;
    let device = |slot: &mut Option<Buffer>, len: u64| -> Result<u64, Error> {
        Ok(ensure(ctx, slot, len * COMPLEX, Memory::Device)?.device_address())
    };
    let partials = device(&mut s.partials, 256 * PANEL as u64)?;
    let scalars = device(&mut s.scalars, 2 + n as u64)?;
    let w = device(&mut s.w, PANEL as u64)?;
    let v = device(&mut s.v, (m * PANEL) as u64)?;
    let parts = device(&mut s.parts, (max_splits * PANEL) as u64 * width)?;
    let vta = device(&mut s.vta, PANEL as u64 * width)?;
    let tvta = device(&mut s.tvta, PANEL as u64 * width)?;
    let gram = device(&mut s.gram, (PANEL * PANEL) as u64)?;
    let gram_bytes = (PANEL * PANEL + PANEL) as u64 * COMPLEX;
    ensure(ctx, &mut s.gram_down, gram_bytes, Memory::Download)?;
    let t_up = ensure(
        ctx,
        &mut s.t_up,
        (PANEL * PANEL) as u64 * COMPLEX,
        Memory::Upload,
    )?
    .device_address();
    let r_bytes = (k_total * n) as u64 * COMPLEX;
    ensure(ctx, &mut s.r_down, r_bytes, Memory::Download)?;
    let buffer = |b: &Option<Buffer>| b.as_ref().expect("allocated").buffer;
    let (scalars_buffer, gram_buffer, gram_down, r_down) = (
        buffer(&s.scalars),
        buffer(&s.gram),
        buffer(&s.gram_down),
        buffer(&s.r_down),
    );

    let step = |cmd: vk::CommandBuffer,
                mode: usize,
                k: usize,
                end: usize,
                grid: [u32; 3],
                sums: usize,
                p: u64| {
        let push = PushBytes::default()
            .addr(a)
            .addr(p)
            .addr(scalars)
            .addr(w)
            .u32s(&[m, m, k, end, mode, sums]);
        ctx.mps_dispatch(cmd, MpsKernel::Qr, &push.0, grid);
        compute_barrier(ctx, cmd);
    };
    // Factorise columns j … j+b−1, then extract V and form V† V if columns
    // follow, copying it back with the panel's τ.
    let panel = |cmd: vk::CommandBuffer, j: usize, b: usize| {
        let end = j + b;
        for k in j..end {
            let below = reduction_groups(m - k - 1);
            step(cmd, 0, k, end, [below as u32, 1, 1], below, partials);
            step(cmd, 1, k, end, [1, 1, 1], below, partials);
            step(cmd, 2, k, end, [groups(m - k - 1), 1, 1], 1, partials);
            if k + 1 < end {
                let count = end - k - 1;
                let from = reduction_groups(m - k);
                step(
                    cmd,
                    3,
                    k,
                    end,
                    [from as u32, count as u32, 1],
                    from,
                    partials,
                );
                step(cmd, 4, k, end, [1, 1, 1], from, partials);
                step(cmd, 5, k, end, [groups((m - k) * count), 1, 1], 1, partials);
            }
        }
        if end < n {
            let rows = m - j;
            step(cmd, 6, j, end, [groups(rows * b), 1, 1], 1, v);
            gemm(
                ctx,
                cmd,
                Gemm {
                    a: Operand::new(v, 0, rows, 1).conj(),
                    b: Operand::new(v, 0, 1, rows),
                    c: gram,
                    c_batch: 0,
                    c_row: b,
                    m: b,
                    n: b,
                    k: rows,
                    batch: 1,
                    add: false,
                },
                Some((parts, max_splits)),
            );
            compute_barrier(ctx, cmd);
            copy(ctx, cmd, gram_buffer, gram_down, (b * b) as u64 * COMPLEX);
            let region = [vk::BufferCopy::default()
                .src_offset((2 + j) as u64 * COMPLEX)
                .dst_offset((PANEL * PANEL) as u64 * COMPLEX)
                .size(b as u64 * COMPLEX)];
            // SAFETY: recording; both buffers hold the copied ranges.
            unsafe {
                ctx.device
                    .cmd_copy_buffer(cmd, scalars_buffer, gram_down, &region)
            };
        }
    };
    // Apply the block reflector of the panel at column j (b wide) to the
    // columns after it: A ← A − V (T† (V† A)), with −T† at `t_up`.
    let trailing = |cmd: vk::CommandBuffer, j: usize, b: usize| {
        let (rows, after) = (m - j, n - j - b);
        let rest = a + ((j + b) * m + j) as u64 * COMPLEX;
        gemm(
            ctx,
            cmd,
            Gemm {
                a: Operand::new(v, 0, rows, 1).conj(),
                b: Operand::new(rest, 0, 1, m),
                c: vta,
                c_batch: 0,
                c_row: after,
                m: b,
                n: after,
                k: rows,
                batch: 1,
                add: false,
            },
            Some((parts, max_splits)),
        );
        compute_barrier(ctx, cmd);
        gemm(
            ctx,
            cmd,
            Gemm {
                a: Operand::new(t_up, 0, b, 1),
                b: Operand::new(vta, 0, after, 1),
                c: tvta,
                c_batch: 0,
                c_row: after,
                m: b,
                n: after,
                k: b,
                batch: 1,
                add: false,
            },
            None,
        );
        compute_barrier(ctx, cmd);
        // Transposed, as the products write row-major: Aᵀ += (−T† V† A)ᵀ Vᵀ.
        gemm(
            ctx,
            cmd,
            Gemm {
                a: Operand::new(tvta, 0, 1, after),
                b: Operand::new(v, 0, rows, 1),
                c: rest,
                c_batch: 0,
                c_row: m,
                m: after,
                n: rows,
                k: b,
                batch: 1,
                add: true,
            },
            None,
        );
        compute_barrier(ctx, cmd);
    };

    // One submission per panel: the previous panel's update, then this
    // panel. Its `V† V` and `τ` come back to form `T` for the next.
    let mut previous: Option<(usize, usize)> = None;
    for j in (0..k_total).step_by(PANEL) {
        let b = PANEL.min(k_total - j);
        ctx.submit(|cmd| {
            if let Some((pj, pb)) = previous {
                trailing(cmd, pj, pb);
            }
            panel(cmd, j, b);
        })?;
        previous = None;
        if j + b < n {
            let raw = s
                .gram_down
                .as_ref()
                .expect("allocated")
                .mapped(gram_bytes as usize)?;
            let values = read_complex(raw, PANEL * PANEL + PANEL);
            let t = block_t(
                &values[..b * b],
                &values[PANEL * PANEL..PANEL * PANEL + b],
                b,
            );
            let minus_t_adjoint: Vec<C64> =
                (0..b * b).map(|e| -t[(e % b) * b + e / b].conj()).collect();
            s.t_up
                .as_mut()
                .expect("allocated")
                .write(bytes_of(&minus_t_adjoint));
            previous = Some((j, b));
        }
    }
    ctx.submit(|cmd| {
        if let Some((pj, pb)) = previous {
            trailing(cmd, pj, pb);
        }
        // R's rows: the top k_total entries of each column.
        let regions: Vec<vk::BufferCopy> = (0..n)
            .map(|c| {
                vk::BufferCopy::default()
                    .src_offset((c * m) as u64 * COMPLEX)
                    .dst_offset((c * k_total) as u64 * COMPLEX)
                    .size(k_total as u64 * COMPLEX)
            })
            .collect();
        // SAFETY: recording; `a_buffer` holds the m × n matrix and the
        // download buffer `k_total × n` entries.
        unsafe { ctx.device.cmd_copy_buffer(cmd, a_buffer, r_down, &regions) };
    })?;
    let raw = s
        .r_down
        .as_ref()
        .expect("allocated")
        .mapped(r_bytes as usize)?;
    let values = read_complex(raw, k_total * n);
    Ok(Mat::from_fn(k_total, n, |i, c| {
        if i <= c {
            values[c * k_total + i]
        } else {
            C64::new(0.0, 0.0)
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mps::linalg::{matmul, qr_r as cpu_qr_r, thin_svd};

    fn random(m: usize, n: usize, seed: u64) -> Mat<C64> {
        let mut x = seed;
        Mat::from_fn(m, n, |_, _| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            C64::new(
                (x >> 11) as f64 / (1u64 << 53) as f64 - 0.5,
                (x >> 23) as f64 / (1u64 << 41) as f64 - 0.5,
            )
        })
    }

    #[test]
    fn gpu_qr_matches_the_cpu() {
        let Ok(ctx) = GpuContext::new() else { return };
        if !ctx.supports_mps() {
            return;
        }
        let mut scratch = QrScratch::default();
        // Tall, square, wide, several panels with a partial last one, and a
        // rank-deficient tall matrix.
        let rank_deficient = matmul(random(300, 7, 5).as_ref(), random(7, 70, 6).as_ref());
        for (case, a) in [
            random(200, 40, 1),
            random(37, 37, 2),
            random(20, 50, 3),
            random(700, 100, 4),
            rank_deficient,
        ]
        .into_iter()
        .enumerate()
        {
            let (m, n) = (a.nrows(), a.ncols());
            let column_major: Vec<C64> = (0..n)
                .flat_map(|c| (0..m).map(move |r| (r, c)))
                .map(|(r, c)| a[(r, c)])
                .collect();
            let buffer = upload(&ctx, bytes_of(&column_major)).unwrap();
            let r = qr_r(
                &ctx,
                buffer.device_address(),
                buffer.buffer,
                m,
                n,
                &mut scratch,
            )
            .unwrap();
            let want = cpu_qr_r(a.as_ref());
            assert_eq!((r.nrows(), r.ncols()), (want.nrows(), want.ncols()));
            // R†R = A†A, and the singular values agree with the CPU's R.
            let scale = matmul(a.adjoint(), a.as_ref()).norm_max();
            let gram = matmul(r.adjoint(), r.as_ref()) - matmul(want.adjoint(), want.as_ref());
            assert!(
                gram.norm_max() < 1e-13 * scale,
                "case {case}: {}",
                gram.norm_max()
            );
            let (s_gpu, s_cpu) = (thin_svd(r.as_ref()).1, thin_svd(want.as_ref()).1);
            for (x, y) in s_gpu.iter().zip(&s_cpu) {
                assert!((x - y).abs() < 1e-12 * s_cpu[0], "case {case}: {x} vs {y}");
            }
        }
    }
}
