//! TDVP and DMRG with their environments and effective Hamiltonians on a
//! GPU (the opt-in `gpu` feature; see [`Mps::set_gpu`](super::Mps::set_gpu)).
//!
//! Sweeps spend nearly all their time applying the effective Hamiltonian of
//! one or two sites inside Krylov and Lanczos iterations. Each application
//! is two dense contractions with the environments, `Σ_a L_a θ` and
//! `Σ_c Z_c R_cᵀ`, around a sparse step for the MPO sites. Here the
//! environments live in GPU memory, built there as the sweep moves, and an
//! application runs the two contractions as complex matrix products
//! (`mps_gemm.comp`) around the sparse step (`mps_mix.comp`). TDVP's Krylov
//! time steps and DMRG's Lanczos eigensolves keep their whole recurrence
//! there too (`mps_krylov.comp`, [`GpuOperator::evolve`] and
//! [`GpuOperator::lowest`]), so only the recurrence's scalars cross to the
//! CPU. SVDs, the subspace expansion and the tensors themselves stay on the
//! CPU. The algorithms are exactly those of the CPU path, with
//! results equal up to rounding.
//!
//! Layouts, all row-major complex doubles: a left environment with `D`
//! channels on a bond of dimension `l` is the `D·l × l` stack of its
//! blocks `L_a` (rows on the bra). A right environment is stored as the
//! stack of transposed blocks `R_cᵀ`, `D·r × r`, so that `Σ_c Z_c R_cᵀ` is a
//! single product of `Z = [Z_0 … Z_{D−1}]` with the stack.

use super::Tensor;
use super::dmrg::Env;
use super::mpo::{Mpo, Rows, left_keyed, pair_keyed, right_keyed};
use crate::error::Error;
use crate::gpu::{Buffer, GpuContext, MPS_PUSH_BYTES, Memory, MpsKernel};
use crate::statevector::eigen::{EigenOptions, lanczos_lowest_from, tridiagonal_lowest};
use crate::statevector::evolve::{KRYLOV_DIM, STEP_TOLERANCE, tridiagonal_exp};
use crate::statevector::krylov::{LinearOperator, breakdown};
use ash::vk;
use faer::Mat;
use num_complex::Complex64 as C64;
use std::ops::Range;
use std::sync::{Arc, Mutex};

const COMPLEX: u64 = 16;

/// Tile edge of `mps_gemm.comp`.
const TILE: u32 = 16;

/// Output tiles below which a product's k range is split. At bond 32 a
/// split made the second contraction 2× faster; at 64, with 64 tiles, it
/// no longer helped.
const TARGET_GROUPS: u32 = 128;

/// Shortest k range of a split product.
const MIN_SPLIT_K: u32 = 64;

/// Workgroups of a grid-stride kernel.
const MAX_GROUPS: u32 = 65_535;

fn usage() -> vk::BufferUsageFlags {
    vk::BufferUsageFlags::STORAGE_BUFFER
        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
        | vk::BufferUsageFlags::TRANSFER_SRC
        | vk::BufferUsageFlags::TRANSFER_DST
}

fn bytes_of(values: &[C64]) -> &[u8] {
    // SAFETY: Complex64 is two f64s with no padding, so any value is bytes.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

/// Little-endian push constants.
#[derive(Default)]
struct PushBytes(Vec<u8>);

impl PushBytes {
    fn addr(mut self, a: u64) -> Self {
        self.0.extend_from_slice(&a.to_le_bytes());
        self
    }
    fn u32s(mut self, values: &[usize]) -> Self {
        for &v in values {
            let v = u32::try_from(v).expect("MPS dimensions fit in 32 bits");
            self.0.extend_from_slice(&v.to_le_bytes());
        }
        self
    }
}

/// One operand of a product: a device address and element strides
/// `(batch, row, column)`.
#[derive(Clone, Copy)]
struct Operand {
    addr: u64,
    batch: usize,
    row: usize,
    col: usize,
    conj: bool,
}

impl Operand {
    fn new(addr: u64, batch: usize, row: usize, col: usize) -> Self {
        Operand {
            addr,
            batch,
            row,
            col,
            conj: false,
        }
    }
    fn conj(mut self) -> Self {
        self.conj = true;
        self
    }
}

/// `C_b = op(A_b) op(B_b)` for `batch` products of shape `m × k` by
/// `k × n`; C has strides `(c_batch, c_row, 1)`.
#[derive(Clone, Copy)]
struct Gemm {
    a: Operand,
    b: Operand,
    c: u64,
    c_batch: usize,
    c_row: usize,
    m: usize,
    n: usize,
    k: usize,
    batch: usize,
}

/// A sparse mixing table, `mps_mix.comp`'s `starts` and `entries`, at
/// these byte offsets in [`GpuSweep::tables`].
#[derive(Clone, Copy)]
struct Table {
    starts: u64,
    entries: u64,
}

/// An environment in GPU memory: `blocks` channels on a bond of dimension
/// `dim`, in the layout described in the module documentation.
struct DeviceEnv {
    buffer: Buffer,
    blocks: usize,
    dim: usize,
}

impl DeviceEnv {
    fn addr(&self) -> u64 {
        self.buffer.device_address()
    }
}

/// Device scratch, grown as needed: for one contraction, and for the
/// Lanczos recurrence of [`GpuOperator::evolve`].
#[derive(Default)]
struct Scratch {
    x: Option<Buffer>,
    y: Option<Buffer>,
    z: Option<Buffer>,
    parts: Option<Buffer>,
    out: Option<Buffer>,
    up: Option<Buffer>,
    down: Option<Buffer>,
    /// The Lanczos vectors `v_0 … v_KRYLOV_DIM`.
    basis: Option<Buffer>,
    /// Per-workgroup partial dot products.
    partials: Option<Buffer>,
    /// The recurrence's scalars (see [`Slots`]), on the device and copied
    /// back.
    scalars: Option<Buffer>,
    scalars_down: Option<Buffer>,
    /// The coefficients of a new state in the basis.
    coeffs: Option<Buffer>,
}

/// A buffer of at least `bytes`, reusing `slot`'s if it is large enough.
fn ensure<'a>(
    ctx: &Arc<GpuContext>,
    slot: &'a mut Option<Buffer>,
    bytes: u64,
    memory: Memory,
) -> Result<&'a mut Buffer, Error> {
    if slot.as_ref().is_none_or(|b| b.size < bytes) {
        // Some headroom, as bonds grow during a sweep.
        let size = (bytes + bytes / 4).max(4096);
        *slot = None;
        *slot = Some(Buffer::new(ctx, size, usage(), memory)?);
    }
    Ok(slot.as_mut().expect("just allocated"))
}

/// The GPU side of a sweep: every environment, the MPO's mixing tables and
/// scratch space.
pub(super) struct GpuSweep {
    ctx: Arc<GpuContext>,
    /// `left[i]`: sites `< i`; `right[i]`: sites `> i`.
    left: Vec<Option<DeviceEnv>>,
    right: Vec<Option<DeviceEnv>>,
    /// Every mixing table; the [`Table`]s hold addresses into it.
    _tables: Buffer,
    /// Per site: keyed by the right channel (one-site products and left
    /// environment updates) and by the left channel (right updates).
    by_right: Vec<Table>,
    by_left: Vec<Table>,
    /// Per bond `i` (sites `i`, `i + 1`): the two-site table.
    pairs: Vec<Table>,
    scratch: Mutex<Scratch>,
    /// The first error raised inside an operator application, which can't
    /// return one.
    error: Mutex<Option<Error>>,
}

impl GpuSweep {
    /// Upload `mpo`'s tables and set the boundary environments.
    pub(super) fn new(ctx: &Arc<GpuContext>, mpo: &Mpo) -> Result<Self, Error> {
        if !ctx.supports_mps() {
            return Err(Error::Gpu(
                "this GPU can't run MPS kernels (they need 64-bit floats and buffer device \
                 addresses)"
                    .into(),
            ));
        }
        let n = mpo.sites.len();
        let mut starts: Vec<u8> = Vec::new();
        let mut entries: Vec<u8> = Vec::new();
        let mut layout = Vec::new();
        let mut pack = |rows: Rows| {
            let table = (starts.len() as u64, entries.len() as u64);
            let mut at = 0u32;
            starts.extend_from_slice(&at.to_le_bytes());
            for row in &rows {
                for &(block, phys, f) in row {
                    entries.extend_from_slice(&f.re.to_le_bytes());
                    entries.extend_from_slice(&f.im.to_le_bytes());
                    entries.extend_from_slice(&(block as u32).to_le_bytes());
                    entries.extend_from_slice(&(phys as u32).to_le_bytes());
                    entries.extend_from_slice(&[0; 8]);
                    at += 1;
                }
                starts.extend_from_slice(&at.to_le_bytes());
            }
            // Keep the next table's entries 16-byte aligned.
            while !starts.len().is_multiple_of(16) {
                starts.push(0);
            }
            layout.push(table);
        };
        for w in &mpo.sites {
            pack(right_keyed(w));
            pack(left_keyed(w));
        }
        for i in 0..n.saturating_sub(1) {
            pack(pair_keyed(&mpo.sites[i], &mpo.sites[i + 1]));
        }
        let entries_at = (starts.len() as u64).next_multiple_of(16);
        let mut all = starts;
        all.resize(entries_at as usize, 0);
        all.extend_from_slice(&entries);
        let tables = upload(ctx, &all)?;
        let base = tables.device_address();
        let table = |(s, e): (u64, u64)| Table {
            starts: base + s,
            entries: base + entries_at + e,
        };
        let by_right = (0..n).map(|i| table(layout[2 * i])).collect();
        let by_left = (0..n).map(|i| table(layout[2 * i + 1])).collect();
        let pairs = (0..n.saturating_sub(1))
            .map(|i| table(layout[2 * n + i]))
            .collect();

        let mut sweep = GpuSweep {
            ctx: ctx.clone(),
            left: (0..n).map(|_| None).collect(),
            right: (0..n).map(|_| None).collect(),
            _tables: tables,
            by_right,
            by_left,
            pairs,
            scratch: Mutex::default(),
            error: Mutex::new(None),
        };
        let one = [C64::new(1.0, 0.0)];
        sweep.left[0] = Some(DeviceEnv {
            buffer: upload(ctx, bytes_of(&one))?,
            blocks: 1,
            dim: 1,
        });
        sweep.right[n - 1] = Some(DeviceEnv {
            buffer: upload(ctx, bytes_of(&one))?,
            blocks: 1,
            dim: 1,
        });
        Ok(sweep)
    }

    /// The first error raised inside an operator application, if any.
    pub(super) fn take_error(&self) -> Result<(), Error> {
        match self.error.lock().expect("GPU error lock").take() {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    fn left_env(&self, i: usize) -> &DeviceEnv {
        self.left[i].as_ref().expect("left environment built")
    }

    fn right_env(&self, i: usize) -> &DeviceEnv {
        self.right[i].as_ref().expect("right environment built")
    }

    /// The environment of sites `< i + 1` from that of sites `< i`, with
    /// site `i`'s tensor.
    pub(super) fn update_left(&mut self, i: usize, a: &Tensor, mpo: &Mpo) -> Result<(), Error> {
        let (l, r) = (a.left, a.right);
        let env = self.left_env(i);
        let (dl, dr) = (mpo.sites[i].left, mpo.sites[i].right);
        assert_eq!((env.blocks, env.dim), (dl, l));
        let target = self.left[i + 1].take();
        let target = reuse(&self.ctx, target, dr * r * r)?;
        let ctx = self.ctx.clone();
        let mut guard = self.scratch.lock().expect("GPU scratch lock");
        let s = &mut *guard;
        let x =
            ensure(&ctx, &mut s.x, (l * 2 * r) as u64 * COMPLEX, Memory::Device)?.device_address();
        let y = ensure(
            &ctx,
            &mut s.y,
            (dl * l * 2 * r) as u64 * COMPLEX,
            Memory::Device,
        )?
        .device_address();
        let z = ensure(
            &ctx,
            &mut s.z,
            (2 * l * dr * r) as u64 * COMPLEX,
            Memory::Device,
        )?
        .device_address();
        let up = ensure(
            &ctx,
            &mut s.up,
            (l * 2 * r) as u64 * COMPLEX,
            Memory::Upload,
        )?;
        up.write(bytes_of(&a.data));
        let (up, xb) = (up.buffer, s.x.as_ref().expect("x").buffer);
        let env_addr = self.left_env(i).addr();
        let out = target.device_address();
        let table = self.by_right[i];
        ctx.submit(|cmd| {
            copy(&ctx, cmd, up, xb, (l * 2 * r) as u64 * COMPLEX);
            compute_barrier(&ctx, cmd);
            // y_a = L_a · A, A as l × 2r.
            gemm(
                &ctx,
                cmd,
                Gemm {
                    a: Operand::new(env_addr, 0, l, 1),
                    b: Operand::new(x, 0, 2 * r, 1),
                    c: y,
                    c_batch: 0,
                    c_row: 2 * r,
                    m: dl * l,
                    n: 2 * r,
                    k: l,
                    batch: 1,
                },
                None,
            );
            compute_barrier(&ctx, cmd);
            // z[(row, p_out), (b, col)] = Σ W … y_a[row, (p_in, col)].
            mix(
                &ctx,
                cmd,
                Mix {
                    input: y,
                    output: z,
                    table,
                    blocks: dr,
                    phys: 2,
                    rows: l,
                    cols: r,
                    in_strides: [l * 2 * r, 2 * r, r, 1],
                    out_strides: [r, 2 * dr * r, dr * r, 1],
                },
            );
            compute_barrier(&ctx, cmd);
            // L'_b = A† z_b, A as 2l × r.
            gemm(
                &ctx,
                cmd,
                Gemm {
                    a: Operand::new(x, 0, 1, r).conj(),
                    b: Operand::new(z, r, dr * r, 1),
                    c: out,
                    c_batch: r * r,
                    c_row: r,
                    m: r,
                    n: r,
                    k: 2 * l,
                    batch: dr,
                },
                None,
            );
        })?;
        drop(guard);
        self.left[i + 1] = Some(DeviceEnv {
            buffer: target,
            blocks: dr,
            dim: r,
        });
        Ok(())
    }

    /// The environment of sites `> i` from that of sites `> i + 1`, with
    /// site `i + 1`'s tensor `a`.
    pub(super) fn update_right(&mut self, i: usize, a: &Tensor, mpo: &Mpo) -> Result<(), Error> {
        let (l, r) = (a.left, a.right);
        let w = &mpo.sites[i + 1];
        let (dl, dr) = (w.left, w.right);
        let env = self.right_env(i + 1);
        assert_eq!((env.blocks, env.dim), (dr, r));
        let target = self.right[i].take();
        let target = reuse(&self.ctx, target, dl * l * l)?;
        let ctx = self.ctx.clone();
        let mut guard = self.scratch.lock().expect("GPU scratch lock");
        let s = &mut *guard;
        let x =
            ensure(&ctx, &mut s.x, (l * 2 * r) as u64 * COMPLEX, Memory::Device)?.device_address();
        let y = ensure(
            &ctx,
            &mut s.y,
            (dr * 2 * l * r) as u64 * COMPLEX,
            Memory::Device,
        )?
        .device_address();
        let z = ensure(
            &ctx,
            &mut s.z,
            (dl * l * 2 * r) as u64 * COMPLEX,
            Memory::Device,
        )?
        .device_address();
        let up = ensure(
            &ctx,
            &mut s.up,
            (l * 2 * r) as u64 * COMPLEX,
            Memory::Upload,
        )?;
        up.write(bytes_of(&a.data));
        let (up, xb) = (up.buffer, s.x.as_ref().expect("x").buffer);
        let env_addr = self.right_env(i + 1).addr();
        let out = target.device_address();
        let table = self.by_left[i + 1];
        ctx.submit(|cmd| {
            copy(&ctx, cmd, up, xb, (l * 2 * r) as u64 * COMPLEX);
            compute_barrier(&ctx, cmd);
            // y_c = A R_cᵀ, A as 2l × r.
            gemm(
                &ctx,
                cmd,
                Gemm {
                    a: Operand::new(x, 0, r, 1),
                    b: Operand::new(env_addr, r * r, r, 1),
                    c: y,
                    c_batch: 2 * l * r,
                    c_row: r,
                    m: 2 * l,
                    n: r,
                    k: r,
                    batch: dr,
                },
                None,
            );
            compute_barrier(&ctx, cmd);
            // z_a[row, (p_out, col)] = Σ W … y_c[(row, p_in), col].
            mix(
                &ctx,
                cmd,
                Mix {
                    input: y,
                    output: z,
                    table,
                    blocks: dl,
                    phys: 2,
                    rows: l,
                    cols: r,
                    in_strides: [2 * l * r, 2 * r, r, 1],
                    out_strides: [l * 2 * r, 2 * r, r, 1],
                },
            );
            compute_barrier(&ctx, cmd);
            // R'_aᵀ = z_a A†, A as l × 2r.
            gemm(
                &ctx,
                cmd,
                Gemm {
                    a: Operand::new(z, l * 2 * r, 2 * r, 1),
                    b: Operand::new(x, 0, 1, 2 * r).conj(),
                    c: out,
                    c_batch: l * l,
                    c_row: l,
                    m: l,
                    n: l,
                    k: 2 * r,
                    batch: dl,
                },
                None,
            );
        })?;
        drop(guard);
        self.right[i] = Some(DeviceEnv {
            buffer: target,
            blocks: dl,
            dim: l,
        });
        Ok(())
    }

    /// The effective Hamiltonian of sites `i` and `i + 1`, on vectors
    /// `(l, p, q, r)`.
    pub(super) fn two_site(&self, i: usize, l: usize, r: usize) -> GpuOperator<'_> {
        let (left, right) = (self.left_env(i), self.right_env(i + 1));
        assert_eq!((left.dim, right.dim), (l, r));
        GpuOperator {
            sweep: self,
            left,
            right,
            table: self.pairs[i],
            phys: 4,
            l,
            r,
        }
    }

    /// The effective Hamiltonian of site `i`, on vectors `(l, p, r)`.
    pub(super) fn one_site(&self, i: usize, l: usize, r: usize) -> GpuOperator<'_> {
        let (left, right) = (self.left_env(i), self.right_env(i));
        assert_eq!((left.dim, right.dim), (l, r));
        GpuOperator {
            sweep: self,
            left,
            right,
            table: self.by_right[i],
            phys: 2,
            l,
            r,
        }
    }

    /// The left environment of sites `< i`, copied to the CPU.
    pub(super) fn download_left(&self, i: usize) -> Result<Env, Error> {
        let env = self.left_env(i);
        let (d, l) = (env.blocks, env.dim);
        let data = self.download(&env.buffer, d * l * l)?;
        Ok(Env {
            blocks: (0..d)
                .map(|a| Mat::from_fn(l, l, |row, col| data[a * l * l + row * l + col]))
                .collect(),
        })
    }

    /// The right environment of sites `> i`, copied to the CPU.
    pub(super) fn download_right(&self, i: usize) -> Result<Env, Error> {
        let env = self.right_env(i);
        let (d, r) = (env.blocks, env.dim);
        let data = self.download(&env.buffer, d * r * r)?;
        // Stored transposed.
        Ok(Env {
            blocks: (0..d)
                .map(|c| Mat::from_fn(r, r, |row, col| data[c * r * r + col * r + row]))
                .collect(),
        })
    }

    fn download(&self, buffer: &Buffer, len: usize) -> Result<Vec<C64>, Error> {
        let bytes = len as u64 * COMPLEX;
        let mut guard = self.scratch.lock().expect("GPU scratch lock");
        let down = ensure(&self.ctx, &mut guard.down, bytes, Memory::Download)?;
        let dst = down.buffer;
        self.ctx
            .submit(|cmd| copy(&self.ctx, cmd, buffer.buffer, dst, bytes))?;
        Ok(read_complex(down.mapped(bytes as usize)?, len))
    }
}

fn read_complex(bytes: &[u8], len: usize) -> Vec<C64> {
    bytes[..len * 16]
        .as_chunks::<16>()
        .0
        .iter()
        .map(|c| {
            let (re, im) = c.split_at(8);
            C64::new(
                f64::from_le_bytes(re.try_into().expect("8 bytes")),
                f64::from_le_bytes(im.try_into().expect("8 bytes")),
            )
        })
        .collect()
}

/// `old` if it holds `len` complex numbers, else a new device buffer.
fn reuse(ctx: &Arc<GpuContext>, old: Option<DeviceEnv>, len: usize) -> Result<Buffer, Error> {
    let bytes = len as u64 * COMPLEX;
    match old {
        Some(env) if env.buffer.size >= bytes => Ok(env.buffer),
        _ => Buffer::new(ctx, bytes, usage(), Memory::Device),
    }
}

/// A device buffer holding `bytes`.
fn upload(ctx: &Arc<GpuContext>, bytes: &[u8]) -> Result<Buffer, Error> {
    let size = bytes.len() as u64;
    let buffer = Buffer::new(ctx, size, usage(), Memory::Device)?;
    let mut staging = Buffer::new(ctx, size, usage(), Memory::Upload)?;
    staging.write(bytes);
    ctx.submit(|cmd| copy(ctx, cmd, staging.buffer, buffer.buffer, size))?;
    Ok(buffer)
}

fn copy(ctx: &GpuContext, cmd: vk::CommandBuffer, src: vk::Buffer, dst: vk::Buffer, bytes: u64) {
    let region = [vk::BufferCopy::default().size(bytes)];
    // SAFETY: recording into a command buffer in the recording state; both
    // buffers hold at least `bytes`.
    unsafe { ctx.device.cmd_copy_buffer(cmd, src, dst, &region) };
}

fn compute_barrier(ctx: &GpuContext, cmd: vk::CommandBuffer) {
    let stages = vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER;
    ctx.barrier(cmd, stages, stages);
}

/// Record a product. With `parts` (a scratch buffer of at least
/// `splits · batch · m · n` elements, and a batch of one), the k range may
/// be split to fill the GPU, the partial products going to `parts` and
/// their sum to `c`.
fn gemm(ctx: &GpuContext, cmd: vk::CommandBuffer, g: Gemm, parts: Option<(u64, usize)>) {
    let tiles = (g.m as u32).div_ceil(TILE) * (g.n as u32).div_ceil(TILE) * g.batch as u32;
    let splits = match parts {
        Some((_, max)) if g.batch == 1 => (TARGET_GROUPS.div_ceil(tiles))
            .min(g.k as u32 / MIN_SPLIT_K)
            .clamp(1, max as u32),
        _ => 1,
    } as usize;
    let (c, c_split) = match parts {
        Some((addr, _)) if splits > 1 => (addr, g.m * g.n),
        _ => (g.c, 0),
    };
    let flags = g.a.conj as usize | (g.b.conj as usize) << 1;
    let push = PushBytes::default()
        .addr(g.a.addr)
        .addr(g.b.addr)
        .addr(c)
        .u32s(&[g.m, g.n, g.k, splits])
        .u32s(&[g.a.batch, g.a.row, g.a.col])
        .u32s(&[g.b.batch, g.b.row, g.b.col])
        .u32s(&[g.c_batch, g.c_row, c_split, flags]);
    debug_assert!(push.0.len() as u32 <= MPS_PUSH_BYTES);
    ctx.mps_dispatch(
        cmd,
        MpsKernel::Gemm,
        &push.0,
        [
            (g.n as u32).div_ceil(TILE),
            (g.m as u32).div_ceil(TILE),
            (g.batch * splits) as u32,
        ],
    );
    if splits > 1 {
        compute_barrier(ctx, cmd);
        let count = g.m * g.n;
        let push = PushBytes::default()
            .addr(c)
            .addr(g.c)
            .u32s(&[count, splits, count]);
        ctx.mps_dispatch(cmd, MpsKernel::Sum, &push.0, [groups(count), 1, 1]);
    }
}

fn groups(threads: usize) -> u32 {
    (threads as u32).div_ceil(256).clamp(1, MAX_GROUPS)
}

/// One `mps_mix.comp` dispatch: `blocks` output channels of `phys`
/// physical indices, each `rows × cols`, with element strides
/// `(block, row, physical, column)` for input and output.
struct Mix {
    input: u64,
    output: u64,
    table: Table,
    blocks: usize,
    phys: usize,
    rows: usize,
    cols: usize,
    in_strides: [usize; 4],
    out_strides: [usize; 4],
}

fn mix(ctx: &GpuContext, cmd: vk::CommandBuffer, m: Mix) {
    let push = PushBytes::default()
        .addr(m.input)
        .addr(m.output)
        .addr(m.table.starts)
        .addr(m.table.entries)
        .u32s(&[m.blocks, m.phys, m.rows, m.cols])
        .u32s(&m.in_strides)
        .u32s(&m.out_strides);
    let total = m.blocks * m.phys * m.rows * m.cols;
    ctx.mps_dispatch(cmd, MpsKernel::Mix, &push.0, [groups(total), 1, 1]);
}

/// The effective Hamiltonian of one or two sites (`phys` = 2 or 4 physical
/// index combinations) on the GPU.
pub(super) struct GpuOperator<'a> {
    sweep: &'a GpuSweep,
    left: &'a DeviceEnv,
    right: &'a DeviceEnv,
    table: Table,
    phys: usize,
    l: usize,
    r: usize,
}

/// Device scratch for one application of an operator.
#[derive(Clone, Copy)]
struct ApplyScratch {
    y: u64,
    z: u64,
    parts: u64,
    max_splits: usize,
}

/// Indices into the Lanczos scalars buffer.
struct Slots;

impl Slots {
    const ZERO: usize = 2 * KRYLOV_DIM;
    const NORM: usize = 2 * KRYLOV_DIM + 1;
    const COUNT: usize = 2 * KRYLOV_DIM + 2;
    // Step `j`'s α and β, reused every `KRYLOV_DIM` steps: a submission
    // takes at most that many steps and needs only the previous β.
    fn alpha(j: usize) -> usize {
        j % KRYLOV_DIM
    }
    fn beta(j: usize) -> usize {
        KRYLOV_DIM + j % KRYLOV_DIM
    }
}

/// `mps_krylov.comp` on vectors of `n` elements.
#[derive(Clone, Copy)]
struct Vectors {
    n: usize,
    /// Workgroups of a dot product's first pass (at most 256).
    groups: usize,
    partials: u64,
    scalars: u64,
}

impl Vectors {
    #[allow(clippy::too_many_arguments)]
    fn dispatch(
        &self,
        ctx: &GpuContext,
        cmd: vk::CommandBuffer,
        (a, b, c): (u64, u64, u64),
        mode: usize,
        (slot0, slot1): (usize, usize),
        (m, stride, op): (usize, usize, usize),
        workgroups: u32,
    ) {
        let push = PushBytes::default()
            .addr(a)
            .addr(b)
            .addr(c)
            .addr(self.scalars)
            .u32s(&[self.n, mode, slot0, slot1, self.groups, m, stride, op]);
        ctx.mps_dispatch(cmd, MpsKernel::Krylov, &push.0, [workgroups, 1, 1]);
    }

    /// `s[slot] = re⟨a, b⟩`, or `‖a‖` with `sqrt` (and `a = b`), followed
    /// by a barrier.
    fn dot(
        &self,
        ctx: &GpuContext,
        cmd: vk::CommandBuffer,
        a: u64,
        b: u64,
        slot: usize,
        sqrt: bool,
    ) {
        let p = self.partials;
        self.dispatch(
            ctx,
            cmd,
            (a, b, p),
            0,
            (0, 0),
            (0, 0, 0),
            self.groups as u32,
        );
        compute_barrier(ctx, cmd);
        self.dispatch(ctx, cmd, (p, p, p), 1, (slot, 0), (0, 0, sqrt as usize), 1);
        compute_barrier(ctx, cmd);
    }

    /// `c −= s[x] a + s[y] b`.
    #[allow(clippy::too_many_arguments)]
    fn update(
        &self,
        ctx: &GpuContext,
        cmd: vk::CommandBuffer,
        c: u64,
        a: u64,
        b: u64,
        x: usize,
        y: usize,
    ) {
        self.dispatch(ctx, cmd, (a, b, c), 2, (x, y), (0, 0, 0), groups(self.n));
    }

    /// `c = a / s[slot]`, or zero when `s[slot]` is zero.
    fn scale(&self, ctx: &GpuContext, cmd: vk::CommandBuffer, a: u64, c: u64, slot: usize) {
        self.dispatch(ctx, cmd, (a, a, c), 3, (slot, 0), (0, 0, 0), groups(self.n));
    }

    /// `c = Σ_{k < m} coeffs[k] v_k` for vectors `v_k` stored back to back
    /// from `basis`.
    fn combine(
        &self,
        ctx: &GpuContext,
        cmd: vk::CommandBuffer,
        basis: u64,
        coeffs: u64,
        m: usize,
        c: u64,
    ) {
        self.dispatch(
            ctx,
            cmd,
            (basis, coeffs, c),
            4,
            (0, 0),
            (m, self.n, 0),
            groups(self.n),
        );
    }
}

impl GpuOperator<'_> {
    fn len(&self) -> usize {
        self.l * self.phys * self.r
    }

    /// Scratch for [`record_apply`](Self::record_apply), allocated in `s`.
    fn apply_scratch(&self, s: &mut Scratch) -> Result<ApplyScratch, Error> {
        let ctx = &self.sweep.ctx;
        let (l, r, phys) = (self.l, self.r, self.phys);
        let (dl, dr) = (self.left.blocks, self.right.blocks);
        let bytes = self.len() as u64 * COMPLEX;
        // At most this many partial products of the second contraction.
        let max_splits = (dr * r / MIN_SPLIT_K as usize).max(1);
        let y = (dl * l * phys * r) as u64 * COMPLEX;
        Ok(ApplyScratch {
            y: ensure(ctx, &mut s.y, y, Memory::Device)?.device_address(),
            z: ensure(ctx, &mut s.z, bytes * dr as u64, Memory::Device)?.device_address(),
            parts: ensure(ctx, &mut s.parts, bytes * max_splits as u64, Memory::Device)?
                .device_address(),
            max_splits,
        })
    }

    /// Record `out = H x` for vectors at device addresses `x` and `out`.
    fn record_apply(&self, cmd: vk::CommandBuffer, x: u64, out: u64, s: ApplyScratch) {
        let ctx = &self.sweep.ctx;
        let (l, r, phys) = (self.l, self.r, self.phys);
        let (dl, dr) = (self.left.blocks, self.right.blocks);
        let (left, right) = (self.left.addr(), self.right.addr());
        // y_a = L_a θ, θ as l × phys·r.
        gemm(
            ctx,
            cmd,
            Gemm {
                a: Operand::new(left, 0, l, 1),
                b: Operand::new(x, 0, phys * r, 1),
                c: s.y,
                c_batch: 0,
                c_row: phys * r,
                m: dl * l,
                n: phys * r,
                k: l,
                batch: 1,
            },
            None,
        );
        compute_barrier(ctx, cmd);
        // Z[(row, o), (c, col)] = Σ W … y_a[row, (i, col)].
        mix(
            ctx,
            cmd,
            Mix {
                input: s.y,
                output: s.z,
                table: self.table,
                blocks: dr,
                phys,
                rows: l,
                cols: r,
                in_strides: [l * phys * r, phys * r, r, 1],
                out_strides: [r, phys * dr * r, dr * r, 1],
            },
        );
        compute_barrier(ctx, cmd);
        // out = Z · [R_0ᵀ; …; R_{D−1}ᵀ].
        gemm(
            ctx,
            cmd,
            Gemm {
                a: Operand::new(s.z, 0, dr * r, 1),
                b: Operand::new(right, 0, r, 1),
                c: out,
                c_batch: 0,
                c_row: r,
                m: phys * l,
                n: r,
                k: dr * r,
                batch: 1,
            },
            Some((s.parts, s.max_splits)),
        );
    }

    fn try_apply(&self, x: &[C64], out: &mut [C64]) -> Result<(), Error> {
        let len = self.len();
        let bytes = len as u64 * COMPLEX;
        let ctx = &self.sweep.ctx;
        let mut guard = self.sweep.scratch.lock().expect("GPU scratch lock");
        let s = &mut *guard;
        let scratch = self.apply_scratch(s)?;
        let xa = ensure(ctx, &mut s.x, bytes, Memory::Device)?.device_address();
        let outa = ensure(ctx, &mut s.out, bytes, Memory::Device)?.device_address();
        ensure(ctx, &mut s.down, bytes, Memory::Download)?;
        let up = ensure(ctx, &mut s.up, bytes, Memory::Upload)?;
        up.write(bytes_of(x));
        let buffer = |b: &Option<Buffer>| b.as_ref().expect("allocated").buffer;
        let (up, xb, outb, down) = (buffer(&s.up), buffer(&s.x), buffer(&s.out), buffer(&s.down));
        ctx.submit(|cmd| {
            copy(ctx, cmd, up, xb, bytes);
            compute_barrier(ctx, cmd);
            self.record_apply(cmd, xa, outa, scratch);
            compute_barrier(ctx, cmd);
            copy(ctx, cmd, outb, down, bytes);
        })?;
        let result = s.down.as_ref().expect("allocated").mapped(bytes as usize)?;
        out.copy_from_slice(&read_complex(result, len));
        Ok(())
    }

    /// GPU buffers for a Lanczos recurrence holding up to `capacity` basis
    /// vectors, with up to `batch` steps per submission.
    fn recurrence<'s>(
        &'s self,
        s: &mut Scratch,
        capacity: usize,
        batch: usize,
    ) -> Result<Recurrence<'s>, Error> {
        let len = self.len();
        let bytes = len as u64 * COMPLEX;
        let ctx = &self.sweep.ctx;
        let apply = self.apply_scratch(s)?;
        let psi = ensure(ctx, &mut s.x, bytes, Memory::Device)?.device_address();
        let w = ensure(ctx, &mut s.out, bytes, Memory::Device)?.device_address();
        let basis =
            ensure(ctx, &mut s.basis, bytes * capacity as u64, Memory::Device)?.device_address();
        let partials =
            ensure(ctx, &mut s.partials, 256 * COMPLEX, Memory::Device)?.device_address();
        let scalars = ensure(ctx, &mut s.scalars, SCALAR_BYTES, Memory::Device)?;
        let (scalars, scalars_buffer) = (scalars.device_address(), scalars.buffer);
        ensure(ctx, &mut s.scalars_down, SCALAR_BYTES, Memory::Download)?;
        let coeffs = ensure(
            ctx,
            &mut s.coeffs,
            capacity as u64 * COMPLEX,
            Memory::Upload,
        )?
        .device_address();
        ensure(ctx, &mut s.down, bytes, Memory::Download)?;
        ensure(ctx, &mut s.up, bytes, Memory::Upload)?;
        // A submission's steps cost about 20 GFLOP at most, well inside
        // driver timeouts at any bond dimension.
        let (l, r, phys) = (self.l, self.r, self.phys);
        let (dl, dr) = (self.left.blocks, self.right.blocks);
        let flops = 8 * phys * l * r * (dl * l + dr * r);
        let batch = (20_000_000_000 / flops.max(1)).clamp(1, batch);
        Ok(Recurrence {
            op: self,
            apply,
            vectors: Vectors {
                n: len,
                groups: len.div_ceil(256).clamp(1, 256),
                partials,
                scalars,
            },
            len,
            psi,
            w,
            basis,
            capacity,
            coeffs,
            scalars: scalars_buffer,
            batch,
        })
    }

    /// `amps ← exp(-iHt) amps` by the same Lanczos time-stepping as
    /// [`krylov_evolve`](crate::statevector::evolve::krylov_evolve), with
    /// the recurrence in GPU memory: each batch of steps is one submission,
    /// and only its scalars α and β come back. The step size and the small
    /// exponential `exp(-iTΔt)` are worked out on the CPU, and the new state
    /// is combined from the basis on the GPU.
    pub(super) fn evolve(&self, amps: &mut [C64], time: f64) -> Result<(), Error> {
        let len = self.len();
        assert_eq!(amps.len(), len);
        let mut guard = self.sweep.scratch.lock().expect("GPU scratch lock");
        let s = &mut *guard;
        let rec = self.recurrence(s, KRYLOV_DIM + 1, KRYLOV_DIM)?;
        let mut beta0 = rec.start(s, amps, true)?;

        let tolerance = STEP_TOLERANCE.max(10.0 * f64::EPSILON);
        let mut remaining = time;
        while remaining != 0.0 {
            if beta0 == 0.0 {
                break;
            }
            let (mut alphas, mut betas) = (Vec::new(), Vec::new());
            let mut closed = false;
            // The Krylov space closes after at most `len` steps.
            let steps = KRYLOV_DIM.min(len);
            let mut k = 0;
            while k < steps && !closed {
                let end = (k + rec.batch).min(steps);
                for (alpha, beta) in rec.steps(s, k..end)? {
                    alphas.push(alpha);
                    betas.push(beta);
                    if beta <= breakdown(&alphas) {
                        closed = true;
                        break;
                    }
                }
                k = end;
            }
            let m = alphas.len();
            // β coupling the basis to the next Lanczos vector; zero when the
            // Krylov space is invariant and the step is exact.
            let residual = if closed { 0.0 } else { betas[m - 1] };

            let mut dt = remaining;
            let y = loop {
                let y = tridiagonal_exp(&alphas, &betas[..m - 1], dt);
                if residual * y[m - 1].norm() <= tolerance {
                    break y;
                }
                dt /= 2.0;
                if dt.abs() <= f64::EPSILON * time.abs() {
                    return Err(Error::NotConverged(
                        "Krylov evolution step shrank to zero".into(),
                    ));
                }
            };

            // ψ ← ‖ψ‖ Σₖ yₖ vₖ, then the next step's ‖ψ‖ and v_0.
            let c: Vec<C64> = y.iter().map(|&yk| yk * beta0).collect();
            beta0 = rec.combine(s, &c)?;
            remaining = if dt == remaining { 0.0 } else { remaining - dt };
        }
        rec.download(s, rec.psi, amps)
    }

    /// The lowest eigenvector of `H`, by the same Lanczos iteration as
    /// [`lanczos_lowest_from`] from the unit vector `v0`, with the
    /// recurrence in GPU memory. The basis is kept there rather than
    /// replayed, so this takes half the products; if it would outgrow
    /// [`LANCZOS_BYTES`] before converging, the solve is redone with the
    /// CPU's recurrence (still applying `H` on the GPU).
    pub(super) fn lowest(
        &self,
        v0: &[C64],
        options: &EigenOptions,
    ) -> Result<(f64, Vec<C64>), Error> {
        self.lowest_within(v0, options, LANCZOS_BYTES)
    }

    /// [`lowest`](Self::lowest) with at most `budget` bytes of basis.
    fn lowest_within(
        &self,
        v0: &[C64],
        options: &EigenOptions,
        budget: u64,
    ) -> Result<(f64, Vec<C64>), Error> {
        let len = self.len();
        assert_eq!(v0.len(), len);
        let fits = (budget / (len as u64 * COMPLEX)) as usize;
        let capacity = (options.max_iterations + 1).min(fits);
        if capacity < 2 {
            return lanczos_lowest_from(self, v0, &[], options);
        }
        let mut guard = self.sweep.scratch.lock().expect("GPU scratch lock");
        let s = &mut *guard;
        let rec = self.recurrence(s, capacity, LANCZOS_BATCH)?;
        rec.start(s, v0, false)?;

        // As `lanczos_lowest_from`: grow the tridiagonal matrix until its
        // lowest Ritz pair converges.
        let (mut alphas, mut betas) = (Vec::new(), Vec::new());
        let mut ritz = None;
        let limit = options.max_iterations.min(capacity - 1);
        let mut k = 0;
        'steps: while k < limit {
            let end = (k + rec.batch).min(limit);
            for (alpha, beta) in rec.steps(s, k..end)? {
                alphas.push(alpha);
                let (theta, v) = tridiagonal_lowest(&alphas, &betas);
                if beta * v[v.len() - 1].abs() <= options.tolerance || beta <= breakdown(&alphas) {
                    ritz = Some((theta, v));
                    break 'steps;
                }
                betas.push(beta);
            }
            k = end;
        }
        let Some((theta, v)) = ritz else {
            if limit < options.max_iterations {
                drop(guard);
                return lanczos_lowest_from(self, v0, &[], options);
            }
            return Err(Error::NotConverged(format!(
                "Lanczos did not converge in {} iterations",
                options.max_iterations
            )));
        };
        // x = Σₖ vₖ eₖ, normalised (into v_0).
        let c: Vec<C64> = v.iter().map(|&x| C64::new(x, 0.0)).collect();
        rec.combine(s, &c)?;
        let mut x = vec![C64::new(0.0, 0.0); len];
        rec.download(s, rec.v(0), &mut x)?;
        Ok((theta, x))
    }
}

/// Bytes of Lanczos basis [`GpuOperator::lowest`] keeps in GPU memory.
const LANCZOS_BYTES: u64 = 1 << 30;

/// Steps per submission in [`GpuOperator::lowest`], which checks for
/// convergence after every step: a batch runs at most this many steps past
/// convergence.
const LANCZOS_BATCH: usize = 8;

const SCALAR_BYTES: u64 = Slots::COUNT as u64 * 8;

/// A Lanczos recurrence in GPU memory: the state `ψ`, the work vector `w`,
/// the basis `v_0 …` (up to `capacity` of them) and the scalars in
/// [`Slots`].
struct Recurrence<'a> {
    op: &'a GpuOperator<'a>,
    apply: ApplyScratch,
    vectors: Vectors,
    len: usize,
    psi: u64,
    w: u64,
    basis: u64,
    capacity: usize,
    coeffs: u64,
    scalars: vk::Buffer,
    batch: usize,
}

impl Recurrence<'_> {
    fn ctx(&self) -> &GpuContext {
        &self.op.sweep.ctx
    }

    fn bytes(&self) -> u64 {
        self.len as u64 * COMPLEX
    }

    fn v(&self, j: usize) -> u64 {
        assert!(j < self.capacity, "Lanczos basis full");
        self.basis + j as u64 * self.bytes()
    }

    /// Record `v_0 = ψ / ‖ψ‖` (‖ψ‖ in its slot) and the scalars' download.
    fn record_normalise(&self, cmd: vk::CommandBuffer, s: &Scratch) {
        let ctx = self.ctx();
        let down = s.scalars_down.as_ref().expect("allocated").buffer;
        self.vectors
            .dot(ctx, cmd, self.psi, self.psi, Slots::NORM, true);
        self.vectors
            .scale(ctx, cmd, self.psi, self.v(0), Slots::NORM);
        compute_barrier(ctx, cmd);
        copy(ctx, cmd, self.scalars, down, SCALAR_BYTES);
    }

    fn read_scalars(s: &Scratch) -> Result<Vec<f64>, Error> {
        let raw = s
            .scalars_down
            .as_ref()
            .expect("allocated")
            .mapped(SCALAR_BYTES as usize)?;
        Ok(raw
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| f64::from_le_bytes(*b))
            .collect())
    }

    /// Upload `x` as ψ and set `v_0`: `ψ / ‖ψ‖` when `normalise`, else `x`
    /// itself (a unit vector). Returns ‖ψ‖ (zero without `normalise`).
    fn start(&self, s: &mut Scratch, x: &[C64], normalise: bool) -> Result<f64, Error> {
        let ctx = self.ctx();
        let bytes = self.bytes();
        s.up.as_mut().expect("allocated").write(bytes_of(x));
        let up = s.up.as_ref().expect("allocated").buffer;
        let psi = s.x.as_ref().expect("allocated").buffer;
        let basis = s.basis.as_ref().expect("allocated").buffer;
        ctx.submit(|cmd| {
            // SAFETY: recording; the buffer holds `SCALAR_BYTES`.
            unsafe {
                ctx.device
                    .cmd_fill_buffer(cmd, self.scalars, 0, SCALAR_BYTES, 0)
            };
            if normalise {
                copy(ctx, cmd, up, psi, bytes);
                compute_barrier(ctx, cmd);
                self.record_normalise(cmd, s);
            } else {
                copy(ctx, cmd, up, basis, bytes);
            }
        })?;
        Ok(if normalise {
            Self::read_scalars(s)?[Slots::NORM]
        } else {
            0.0
        })
    }

    /// Steps `j` in `steps` (one submission), each `w = H v_j`,
    /// `α_j = ⟨v_j, w⟩`, `w −= α_j v_j + β_{j−1} v_{j−1}`, `β_j = ‖w‖`,
    /// `v_{j+1} = w / β_j`. Returns each step's `(α_j, β_j)`.
    fn steps(&self, s: &Scratch, steps: Range<usize>) -> Result<Vec<(f64, f64)>, Error> {
        let ctx = self.ctx();
        let (vectors, w) = (self.vectors, self.w);
        let down = s.scalars_down.as_ref().expect("allocated").buffer;
        ctx.submit(|cmd| {
            for j in steps.clone() {
                self.op.record_apply(cmd, self.v(j), w, self.apply);
                compute_barrier(ctx, cmd);
                vectors.dot(ctx, cmd, self.v(j), w, Slots::alpha(j), false);
                let previous = if j == 0 {
                    Slots::ZERO
                } else {
                    Slots::beta(j - 1)
                };
                let v_prev = self.v(j.saturating_sub(1));
                vectors.update(ctx, cmd, w, self.v(j), v_prev, Slots::alpha(j), previous);
                compute_barrier(ctx, cmd);
                vectors.dot(ctx, cmd, w, w, Slots::beta(j), true);
                vectors.scale(ctx, cmd, w, self.v(j + 1), Slots::beta(j));
                compute_barrier(ctx, cmd);
            }
            copy(ctx, cmd, self.scalars, down, SCALAR_BYTES);
        })?;
        let values = Self::read_scalars(s)?;
        Ok(steps
            .map(|j| (values[Slots::alpha(j)], values[Slots::beta(j)]))
            .collect())
    }

    /// `ψ = Σₖ c[k] vₖ`, then `v_0 = ψ / ‖ψ‖`. Returns ‖ψ‖.
    fn combine(&self, s: &mut Scratch, c: &[C64]) -> Result<f64, Error> {
        assert!(c.len() <= self.capacity);
        s.coeffs.as_mut().expect("allocated").write(bytes_of(c));
        let ctx = self.ctx();
        ctx.submit(|cmd| {
            self.vectors
                .combine(ctx, cmd, self.basis, self.coeffs, c.len(), self.psi);
            compute_barrier(ctx, cmd);
            self.record_normalise(cmd, s);
        })?;
        Ok(Self::read_scalars(s)?[Slots::NORM])
    }

    /// Copy the vector at device address `from` into `out`.
    fn download(&self, s: &Scratch, from: u64, out: &mut [C64]) -> Result<(), Error> {
        let ctx = self.ctx();
        let bytes = self.bytes();
        let down = s.down.as_ref().expect("allocated");
        // `from` is ψ or a basis vector: find its buffer and offset.
        let (src, offset) = if from == self.psi {
            (s.x.as_ref().expect("allocated").buffer, 0)
        } else {
            (
                s.basis.as_ref().expect("allocated").buffer,
                from - self.basis,
            )
        };
        let region = [vk::BufferCopy::default().src_offset(offset).size(bytes)];
        ctx.submit(|cmd| {
            // SAFETY: recording; both buffers hold the copied range.
            unsafe { ctx.device.cmd_copy_buffer(cmd, src, down.buffer, &region) }
        })?;
        out.copy_from_slice(&read_complex(down.mapped(bytes as usize)?, self.len));
        Ok(())
    }
}

impl LinearOperator<C64> for GpuOperator<'_> {
    fn apply(&self, x: &[C64], out: &mut [C64]) {
        if let Err(e) = self.try_apply(x, out) {
            // Krylov iterations can't fail midway: record the error for the
            // caller (see `GpuSweep::take_error`) and return zeros, which
            // end the iteration.
            out.fill(C64::new(0.0, 0.0));
            let mut error = self.sweep.error.lock().expect("GPU error lock");
            error.get_or_insert(e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::envs::{Envs, Op};
    use super::super::{Mps, MpsOptions};
    use super::*;
    use crate::observable::{Pauli, PauliString, PauliSum};

    fn gpu() -> Option<Arc<GpuContext>> {
        match GpuContext::new() {
            Ok(ctx) if ctx.supports_mps() => Some(ctx),
            Ok(_) => {
                eprintln!("skipping GPU MPS test: no 64-bit floats");
                None
            }
            Err(e) => {
                eprintln!("skipping GPU MPS test: {e}");
                None
            }
        }
    }

    /// Long-range couplings, fields with all Paulis, and four-qubit strings,
    /// so MPO bonds are a few tens and every Pauli appears.
    fn hamiltonian(n: usize) -> PauliSum {
        let paulis = [Pauli::X, Pauli::Y, Pauli::Z];
        let mut h = PauliSum::new();
        for i in 0..n {
            h.add(
                0.3 + 0.1 * i as f64,
                PauliString::new([(i, paulis[i % 3])]).unwrap(),
            );
            for j in i + 1..n.min(i + 6) {
                let c = 1.0 / (j - i) as f64;
                h.add(c, PauliString::new([(i, Pauli::Z), (j, Pauli::Z)]).unwrap());
                h.add(
                    0.7 * c,
                    PauliString::new([(i, Pauli::X), (j, Pauli::Y)]).unwrap(),
                );
            }
            if i + 5 < n {
                let ops = [
                    (i, Pauli::X),
                    (i + 2, Pauli::Z),
                    (i + 3, Pauli::Y),
                    (i + 5, Pauli::X),
                ];
                h.add(0.4, PauliString::new(ops).unwrap());
            }
        }
        h
    }

    fn max_diff(a: &[C64], b: &[C64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).norm())
            .fold(0.0, f64::max)
    }

    fn norm(a: &[C64]) -> f64 {
        a.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt()
    }

    fn env_diff(a: &Env, b: &Env) -> f64 {
        assert_eq!(a.blocks.len(), b.blocks.len());
        let mut worst = 0.0f64;
        for (x, y) in a.blocks.iter().zip(&b.blocks) {
            assert_eq!((x.nrows(), x.ncols()), (y.nrows(), y.ncols()));
            for i in 0..x.nrows() {
                for j in 0..x.ncols() {
                    worst = worst.max((x[(i, j)] - y[(i, j)]).norm());
                }
            }
        }
        worst
    }

    fn env_max(a: &Env) -> f64 {
        a.blocks
            .iter()
            .flat_map(|b| {
                (0..b.nrows()).flat_map(move |i| (0..b.ncols()).map(move |j| b[(i, j)].norm()))
            })
            .fold(0.0, f64::max)
    }

    #[test]
    fn environments_and_operators_match_cpu() {
        let Some(ctx) = gpu() else { return };
        let n = 14;
        let options = MpsOptions {
            max_bond: 40,
            cutoff: 0.0,
        };
        // Bonds of 1, 2, 4, … up to 37: sizes off the 16-wide tiles.
        let mut mps = Mps::random(n, 37, 5, options);
        let mpo = Mpo::new(&hamiltonian(n), n).unwrap();
        let mut cpu = Envs::new(&mps, &mpo).unwrap();
        mps.set_gpu(Some(ctx)).unwrap();
        let mut gpu = Envs::new(&mps, &mpo).unwrap();
        for i in (0..n - 1).rev() {
            cpu.update_right(i, &mps.tensors[i + 1], &mpo).unwrap();
            gpu.update_right(i, &mps.tensors[i + 1], &mpo).unwrap();
        }
        for (i, t) in mps.tensors[..n - 1].iter().enumerate() {
            cpu.update_left(i, t, &mpo).unwrap();
            gpu.update_left(i, t, &mpo).unwrap();
        }
        let mut seed = 1u64;
        let mut random = |len: usize| -> Vec<C64> {
            (0..len)
                .map(|_| {
                    seed = seed
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let a = (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
                    C64::new(a, 1.0 - 2.0 * a * a)
                })
                .collect()
        };
        for i in 0..n {
            let (l, r) = (mps.tensors[i].left, mps.tensors[i].right);
            let (want_l, got_l) = (cpu.left(i).unwrap(), gpu.left(i).unwrap());
            let (want_r, got_r) = (cpu.right(i).unwrap(), gpu.right(i).unwrap());
            let tol_l = 1e-12 * env_max(&want_l).max(1.0);
            let tol_r = 1e-12 * env_max(&want_r).max(1.0);
            assert!(env_diff(&want_l, &got_l) < tol_l, "left {i}");
            assert!(env_diff(&want_r, &got_r) < tol_r, "right {i}");

            let x = random(l * 2 * r);
            let (mut want, mut got) =
                (vec![C64::default(); x.len()], vec![C64::default(); x.len()]);
            cpu.one_site(i, &mpo, l, r).apply(&x, &mut want);
            gpu.one_site(i, &mpo, l, r).apply(&x, &mut got);
            gpu.check().unwrap();
            assert!(
                max_diff(&want, &got) < 1e-11,
                "one site {i}: {}",
                max_diff(&want, &got)
            );
            // Krylov evolution with the recurrence on the GPU, against the
            // CPU's; long enough to take several Krylov steps.
            let (mut want, mut got) = (x.clone(), x.clone());
            cpu.one_site(i, &mpo, l, r).evolve(&mut want, -0.9).unwrap();
            gpu.one_site(i, &mpo, l, r).evolve(&mut got, -0.9).unwrap();
            let diff = max_diff(&want, &got);
            assert!(diff < 1e-10 * norm(&x), "evolve one site {i}: {diff}");

            if i + 1 < n {
                let r2 = mps.tensors[i + 1].right;
                let x = random(l * 4 * r2);
                let (mut want, mut got) =
                    (vec![C64::default(); x.len()], vec![C64::default(); x.len()]);
                cpu.two_site(i, &mpo, l, r2).apply(&x, &mut want);
                gpu.two_site(i, &mpo, l, r2).apply(&x, &mut got);
                gpu.check().unwrap();
                assert!(
                    max_diff(&want, &got) < 1e-11,
                    "two sites {i}: {}",
                    max_diff(&want, &got)
                );

                let (mut want, mut got) = (x.clone(), x.clone());
                cpu.two_site(i, &mpo, l, r2).evolve(&mut want, 0.9).unwrap();
                gpu.two_site(i, &mpo, l, r2).evolve(&mut got, 0.9).unwrap();
                let diff = max_diff(&want, &got);
                assert!(diff < 1e-10 * norm(&x), "evolve two sites {i}: {diff}");

                // The lowest eigenvector by Lanczos on the GPU, keeping the
                // basis there or (with room for 4 vectors) falling back to
                // the CPU's recurrence, against the CPU's.
                let v0: Vec<C64> = x.iter().map(|v| v / norm(&x)).collect();
                let options = EigenOptions::default();
                let (want_e, want) =
                    lanczos_lowest_from(&cpu.two_site(i, &mpo, l, r2), &v0, &[], &options).unwrap();
                let Op::Gpu(op) = gpu.two_site(i, &mpo, l, r2) else {
                    unreachable!()
                };
                let bytes = (x.len() * 16) as u64;
                for budget in [LANCZOS_BYTES, 4 * bytes] {
                    let (got_e, got) = op.lowest_within(&v0, &options, budget).unwrap();
                    assert!((want_e - got_e).abs() < 1e-9, "{want_e} {got_e}");
                    // Unique ground states agree up to a phase.
                    let overlap: C64 = want.iter().zip(&got).map(|(a, b)| a.conj() * b).sum();
                    assert!(
                        (overlap.norm() - 1.0).abs() < 1e-9,
                        "lowest two sites {i}, budget {budget}: {overlap}"
                    );
                    // The Ritz value is the vector's energy.
                    let mut hx = vec![C64::default(); got.len()];
                    cpu.two_site(i, &mpo, l, r2).apply(&got, &mut hx);
                    let energy: f64 = got.iter().zip(&hx).map(|(a, b)| (a.conj() * b).re).sum();
                    assert!((energy - got_e).abs() < 1e-9, "{energy} {got_e}");
                }
            }
        }
    }
}
