//! Vulkan GPU acceleration for [`PeriodicMps`](super::PeriodicMps) sweeps
//! (`--features gpu`; see [`PeriodicMps::set_gpu`](super::PeriodicMps::set_gpu)).
//!
//! Reuses `ketrs`'s existing FP64 Vulkan compute shaders (`mps_gemm.comp`,
//! `mps_mix.comp`, `mps_sum.comp`) to execute every Pippan–White–Evertz local
//! effective Hamiltonian (`BatchedMpoOp`, `StackedPairOp`, `StackedStringOp`)
//! and metric operator (`LocalNormOp`) in a single Vulkan command-buffer
//! submission per Krylov/Davidson matvec.
//!
//! All three periodic operator types share the exact same 3-stage pipeline:
//! 1. `Y = L_stack · X` (`mps_gemm.comp`: `(n_l·l) × l` by `l × (phys·r)`),
//! 2. `Z = Mix(table, Y)` (`mps_mix.comp`: sparse multi-mode MPO/string/identity mix),
//! 3. `out (+)= Z · R_stack` (`mps_gemm.comp` with split-K on the first term).

use crate::error::Error;
use crate::gpu::{Buffer, GpuContext, Memory};
use crate::mps::gpu::{
    COMPLEX, Gemm, MIN_SPLIT_K, Mix, Operand, TARGET_GROUPS, TILE, Table, bytes_of,
    compute_barrier, copy, ensure, gemm, mix, read_complex, upload,
};
use num_complex::Complex64 as C64;
use std::sync::{Arc, Mutex};

/// Persistent Vulkan scratch buffers shared across all local problems in a
/// periodic sweep so Krylov/Davidson iterations do zero per-matvec allocations.
#[derive(Default)]
pub(super) struct GpuPeriodicScratch {
    x: Option<Buffer>,
    y: Option<Buffer>,
    z: Option<Buffer>,
    parts: Option<Buffer>,
    out: Option<Buffer>,
    up: Option<Buffer>,
    down: Option<Buffer>,
}

/// CPU-side specification of one 3-stage stacked operator term to pack onto GPU.
pub(super) struct GpuTermSpec<'a> {
    pub(super) n_l: usize,
    pub(super) l_stack: &'a [C64],
    pub(super) n_r: usize,
    pub(super) r_stack: &'a [C64],
    /// Length `n_r * phys`: `rows[sr * phys + p_out]` lists `(sl, p_in, coeff)`.
    pub(super) rows: Vec<Vec<(usize, usize, C64)>>,
}

struct PackedTerm {
    l_addr: u64,
    r_addr: u64,
    table: Table,
    n_l: usize,
    n_r: usize,
}

/// GPU-resident local periodic effective operator (`H̃` or `Ñ`), executing all
/// stacked MPO, scalar, and crossing-string terms in one Vulkan submission per
/// matvec.
pub(super) struct GpuLocalOp {
    ctx: Arc<GpuContext>,
    scratch: Arc<Mutex<GpuPeriodicScratch>>,
    l: usize,
    r: usize,
    phys: usize,
    _l_buf: Buffer,
    _r_buf: Buffer,
    _tables_buf: Buffer,
    terms: Vec<PackedTerm>,
    max_n_l: usize,
    max_n_r: usize,
    error: Mutex<Option<Error>>,
}

impl GpuLocalOp {
    /// Upload all non-empty stacked operator terms into three contiguous GPU
    /// buffers (`L`, `R^T`, and `Mix` tables). Returns `Ok(None)` if there are
    /// no active terms.
    pub(super) fn new(
        ctx: &Arc<GpuContext>,
        scratch: &Arc<Mutex<GpuPeriodicScratch>>,
        l: usize,
        r: usize,
        phys: usize,
        specs: &[GpuTermSpec<'_>],
    ) -> Result<Option<Self>, Error> {
        if l == 0 || r == 0 || phys == 0 {
            return Ok(None);
        }
        let active: Vec<&GpuTermSpec<'_>> = specs
            .iter()
            .filter(|s| s.n_l > 0 && s.n_r > 0 && !s.l_stack.is_empty() && !s.r_stack.is_empty())
            .collect();
        if active.is_empty() {
            return Ok(None);
        }

        let mut all_l: Vec<C64> = Vec::new();
        let mut all_r: Vec<C64> = Vec::new();
        let mut starts: Vec<u8> = Vec::new();
        let mut entries: Vec<u8> = Vec::new();
        let mut meta: Vec<(u64, u64, u64, u64, usize, usize)> = Vec::with_capacity(active.len());

        let mut max_n_l = 0usize;
        let mut max_n_r = 0usize;
        for spec in &active {
            let l_byte_off = all_l.len() as u64 * COMPLEX;
            let r_byte_off = all_r.len() as u64 * COMPLEX;
            all_l.extend_from_slice(spec.l_stack);
            all_r.extend_from_slice(spec.r_stack);

            let s_off = starts.len() as u64;
            let e_off = entries.len() as u64;
            let mut at = 0u32;
            starts.extend_from_slice(&at.to_le_bytes());
            debug_assert_eq!(spec.rows.len(), spec.n_r * phys);
            for row in &spec.rows {
                for &(sl, p_in, f) in row {
                    entries.extend_from_slice(&f.re.to_le_bytes());
                    entries.extend_from_slice(&f.im.to_le_bytes());
                    entries.extend_from_slice(&(sl as u32).to_le_bytes());
                    entries.extend_from_slice(&(p_in as u32).to_le_bytes());
                    entries.extend_from_slice(&[0u8; 8]);
                    at += 1;
                }
                starts.extend_from_slice(&at.to_le_bytes());
            }
            while !starts.len().is_multiple_of(16) {
                starts.push(0);
            }
            meta.push((l_byte_off, r_byte_off, s_off, e_off, spec.n_l, spec.n_r));
            max_n_l = max_n_l.max(spec.n_l);
            max_n_r = max_n_r.max(spec.n_r);
        }

        // Ensure entries buffer is non-empty (Vulkan forbids 0-byte buffers).
        if entries.is_empty() {
            entries.resize(32, 0);
        }
        let entries_at = (starts.len() as u64).next_multiple_of(16);
        let mut table_bytes = starts;
        table_bytes.resize(entries_at as usize, 0);
        table_bytes.extend_from_slice(&entries);

        let l_buf = upload(ctx, bytes_of(&all_l))?;
        let r_buf = upload(ctx, bytes_of(&all_r))?;
        let tables_buf = upload(ctx, &table_bytes)?;

        let l_base = l_buf.device_address();
        let r_base = r_buf.device_address();
        let t_base = tables_buf.device_address();
        let terms = meta
            .into_iter()
            .map(|(lo, ro, so, eo, n_l, n_r)| PackedTerm {
                l_addr: l_base + lo,
                r_addr: r_base + ro,
                table: Table {
                    starts: t_base + so,
                    entries: t_base + entries_at + eo,
                },
                n_l,
                n_r,
            })
            .collect();

        Ok(Some(GpuLocalOp {
            ctx: ctx.clone(),
            scratch: scratch.clone(),
            l,
            r,
            phys,
            _l_buf: l_buf,
            _r_buf: r_buf,
            _tables_buf: tables_buf,
            terms,
            max_n_l,
            max_n_r,
            error: Mutex::new(None),
        }))
    }

    /// Execute `out = Op x` on the GPU in one command-buffer submission.
    /// Returns `true` on success, `false` if a Vulkan error occurred (leaving
    /// the error recorded for inspection and allowing graceful fallback).
    pub(super) fn apply(&self, x: &[C64], out: &mut [C64]) -> bool {
        let len = self.l * self.phys * self.r;
        debug_assert_eq!(x.len(), len);
        debug_assert_eq!(out.len(), len);
        if self.terms.is_empty() {
            out.fill(C64::new(0.0, 0.0));
            return true;
        }
        let mut run = || -> Result<(), Error> {
            let (l, r, phys) = (self.l, self.r, self.phys);
            let pr = phys * r;
            let bytes = len as u64 * COMPLEX;
            let y_bytes = (self.max_n_l * l * pr) as u64 * COMPLEX;
            let z_bytes = (l * phys * self.max_n_r * r) as u64 * COMPLEX;

            // Split-K headroom for term 0's second contraction.
            let tiles = (l * phys).div_ceil(TILE as usize) * r.div_ceil(TILE as usize);
            let max_splits = (self.terms[0].n_r * r / MIN_SPLIT_K as usize)
                .min((TARGET_GROUPS as usize).div_ceil(tiles.max(1)))
                .max(1);
            let parts_bytes = bytes * max_splits as u64;

            let ctx = &self.ctx;
            let mut guard = self.scratch.lock().expect("GPU periodic scratch lock");
            let s = &mut *guard;
            let x_addr = ensure(ctx, &mut s.x, bytes, Memory::Device)?.device_address();
            let y_addr = ensure(ctx, &mut s.y, y_bytes, Memory::Device)?.device_address();
            let z_addr = ensure(ctx, &mut s.z, z_bytes, Memory::Device)?.device_address();
            let parts_addr =
                ensure(ctx, &mut s.parts, parts_bytes, Memory::Device)?.device_address();
            let out_addr = ensure(ctx, &mut s.out, bytes, Memory::Device)?.device_address();
            ensure(ctx, &mut s.down, bytes, Memory::Download)?;
            let up = ensure(ctx, &mut s.up, bytes, Memory::Upload)?;
            up.write(bytes_of(x));

            let buf = |b: &Option<Buffer>| b.as_ref().expect("allocated").buffer;
            let (up_b, x_b, out_b, down_b) = (buf(&s.up), buf(&s.x), buf(&s.out), buf(&s.down));

            ctx.submit(|cmd| {
                copy(ctx, cmd, up_b, x_b, bytes);
                compute_barrier(ctx, cmd);
                for (idx, term) in self.terms.iter().enumerate() {
                    let add = idx > 0;
                    // Stage 1: Y[(sl, i), (p, j)] = L_stack[(sl, i), a] · X[a, (p, j)].
                    gemm(
                        ctx,
                        cmd,
                        Gemm {
                            a: Operand::new(term.l_addr, 0, l, 1),
                            b: Operand::new(x_addr, 0, pr, 1),
                            c: y_addr,
                            c_batch: 0,
                            c_row: pr,
                            m: term.n_l * l,
                            n: pr,
                            k: l,
                            batch: 1,
                            add: false,
                        },
                        None,
                    );
                    compute_barrier(ctx, cmd);
                    // Stage 2: Z[(i, p_out), (sr, j)] = Σ f · Y[(sl, i), (p_in, j)].
                    let nr_r = term.n_r * r;
                    mix(
                        ctx,
                        cmd,
                        Mix {
                            input: y_addr,
                            output: z_addr,
                            table: term.table,
                            blocks: term.n_r,
                            phys,
                            rows: l,
                            cols: r,
                            in_strides: [l * pr, pr, r, 1],
                            out_strides: [r, phys * nr_r, nr_r, 1],
                        },
                    );
                    compute_barrier(ctx, cmd);
                    // Stage 3: out[(i, p_out), j'] (+)= Z[(i, p_out), (sr, j)] · R_stack[(sr, j), j'].
                    gemm(
                        ctx,
                        cmd,
                        Gemm {
                            a: Operand::new(z_addr, 0, nr_r, 1),
                            b: Operand::new(term.r_addr, 0, r, 1),
                            c: out_addr,
                            c_batch: 0,
                            c_row: r,
                            m: l * phys,
                            n: r,
                            k: nr_r,
                            batch: 1,
                            add,
                        },
                        if !add {
                            Some((parts_addr, max_splits))
                        } else {
                            None
                        },
                    );
                    compute_barrier(ctx, cmd);
                }
                copy(ctx, cmd, out_b, down_b, bytes);
            })?;

            let mapped = s.down.as_ref().expect("allocated").mapped(bytes as usize)?;
            out.copy_from_slice(&read_complex(mapped, len));
            Ok(())
        };
        match run() {
            Ok(()) => true,
            Err(e) => {
                self.error
                    .lock()
                    .expect("GPU periodic error lock")
                    .get_or_insert(e);
                false
            }
        }
    }
}
