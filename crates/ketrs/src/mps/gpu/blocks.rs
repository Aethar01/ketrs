//! Sweeps of a symmetric MPS on the GPU: block-sparse environments in GPU
//! memory, and effective Hamiltonians applied there.
//!
//! Every contraction is the CPU's (see [`blocks`](crate::mps::blocks)), as
//! three steps: a list of independent block products, a sparse step that
//! sums scaled blocks into disjoint segments, and another list of
//! products. Each list is one dispatch of the grouped product kernel
//! (`mps_bgemm.comp`, in 8×8 and 16×16 output tiles, chosen per product to
//! waste the least of small blocks), and the sparse step one dispatch of
//! `mps_bmix.comp`. The lists are built on the CPU, uploaded once per
//! operator or environment update, and reused by every application in a
//! Krylov or Lanczos recurrence, which runs on the GPU as for dense MPS.

use super::{
    ApplyScratch, COMPLEX, DeviceOp, LANCZOS_BYTES, MAX_GROUPS, Scratch, bytes_of, compute_barrier,
    copy, ensure, evolve, groups, lowest_within, read_complex, upload, usage,
};
use crate::error::Error;
use crate::gpu::{Buffer, GpuContext, Memory, MpsKernel};
use crate::mps::Mps;
use crate::mps::blocks::{BlockEnv, EnvLayout, Layout, Plan, Side, SiteBlocks};
use crate::mps::charges::{Charges, Sectors};
use crate::mps::mpo::{Mpo, MpoSite, left_keyed, right_keyed};
use crate::statevector::eigen::{EigenOptions, Ritz};
use crate::statevector::krylov::LinearOperator;
use ash::vk;
use num_complex::Complex64 as C64;
use std::sync::{Arc, Mutex};

/// Flop-weighted mean block width (the smaller of a product's output
/// dimensions) from which a list of products uses 16×16 tiles.
const LARGE_BLOCK: usize = 64;

const CONJ_A: u32 = 1;
const CONJ_B: u32 = 2;
const ADD: u32 = 4;

/// One product of a grouped dispatch: element offsets into its A, B and C
/// buffers, its shape `m × k` by `k × n`, and element strides.
#[derive(Clone, Copy, Debug)]
struct Product {
    a: usize,
    b: usize,
    c: usize,
    m: usize,
    n: usize,
    k: usize,
    a_row: usize,
    a_col: usize,
    b_row: usize,
    b_col: usize,
    c_row: usize,
}

impl Product {
    fn flops(&self) -> usize {
        8 * self.m * self.n * self.k
    }
}

/// A segment of the sparse step: `z[z + row · z_row + col] = Σ f ·
/// y[y + row · y_row + col]` over its entries `(f, y, y_row)`.
struct Segment {
    z: usize,
    z_row: usize,
    rows: usize,
    cols: usize,
    entries: Vec<(C64, usize, usize)>,
}

/// Little-endian tables, 16-byte aligned, to upload.
#[derive(Default)]
struct Tables(Vec<u8>);

impl Tables {
    fn align(&mut self) -> u64 {
        while !self.0.len().is_multiple_of(16) {
            self.0.push(0);
        }
        self.0.len() as u64
    }

    fn u32(&mut self, v: usize) {
        let v = u32::try_from(v).expect("MPS dimensions fit in 32 bits");
        self.0.extend_from_slice(&v.to_le_bytes());
    }

    /// The products, with their tiles split by tile size.
    fn products(&mut self, products: &[Product]) -> Grouped {
        let tasks = self.align();
        for p in products {
            for v in [
                p.a, p.b, p.c, p.m, p.n, p.k, p.a_row, p.a_col, p.b_row, p.b_col, p.c_row, 0,
            ] {
                self.u32(v);
            }
        }
        // One tile size per list, so it is one dispatch: 8×8 unless the
        // blocks are large. Symmetric blocks are mostly tens wide, where
        // 16×16 tiles leave much of each tile idle (8×8 tiles made a
        // bond-256 sweep of a lattice gauge theory 12% faster on an
        // RX 9060 XT, and were as fast at bond 512).
        let flops: usize = products.iter().map(Product::flops).sum();
        let wide: usize = products.iter().map(|p| p.flops() * p.m.min(p.n)).sum();
        let edge = if wide >= LARGE_BLOCK * flops.max(1) {
            16
        } else {
            8
        };
        let at = self.align();
        let mut count = 0u32;
        for (index, p) in products.iter().enumerate() {
            for row in 0..p.m.div_ceil(edge) {
                for col in 0..p.n.div_ceil(edge) {
                    for v in [index, row, col, 0] {
                        self.u32(v);
                    }
                    count += 1;
                }
            }
        }
        let mut parts = [(0u64, 0u32); 2];
        parts[usize::from(edge == 16)] = (at, count);
        Grouped { tasks, parts }
    }

    /// The segments of a sparse step, empty ones left out.
    fn segments(&mut self, segments: &[Segment]) -> Mixing {
        let at = self.align();
        let (mut first, mut start, mut count) = (0, 0, 0);
        for s in segments.iter().filter(|s| s.rows * s.cols > 0) {
            let last = first + s.entries.len();
            for v in [s.z, s.z_row, s.rows, s.cols, first, last, start, 0] {
                self.u32(v);
            }
            first = last;
            start += s.rows * s.cols;
            count += 1;
        }
        let entries = self.align();
        for s in segments.iter().filter(|s| s.rows * s.cols > 0) {
            for &(f, y, y_row) in &s.entries {
                self.0.extend_from_slice(&f.re.to_le_bytes());
                self.0.extend_from_slice(&f.im.to_le_bytes());
                self.u32(y);
                self.u32(y_row);
                self.0.extend_from_slice(&[0; 8]);
            }
        }
        Mixing {
            segments: at,
            entries,
            count,
            total: start,
        }
    }
}

/// A grouped dispatch's tables: byte offsets of its tasks and, per tile
/// size (8, 16), of its tiles with their count.
#[derive(Clone, Copy)]
struct Grouped {
    tasks: u64,
    parts: [(u64, u32); 2],
}

/// A sparse step's tables: byte offsets and sizes.
#[derive(Clone, Copy)]
struct Mixing {
    segments: u64,
    entries: u64,
    count: usize,
    total: usize,
}

/// Record a grouped product dispatch over buffers `a`, `b` and `c`, with
/// tables at device address `base`.
fn record_products(
    ctx: &GpuContext,
    cmd: vk::CommandBuffer,
    base: u64,
    g: Grouped,
    (a, b, c): (u64, u64, u64),
    flags: u32,
) {
    for (kernel, (tiles, count)) in [MpsKernel::BlockGemm8, MpsKernel::BlockGemm16]
        .into_iter()
        .zip(g.parts)
    {
        if count == 0 {
            continue;
        }
        let mut push = Vec::with_capacity(48);
        for addr in [a, b, c, base + g.tasks, base + tiles] {
            push.extend_from_slice(&addr.to_le_bytes());
        }
        push.extend_from_slice(&count.to_le_bytes());
        push.extend_from_slice(&flags.to_le_bytes());
        let x = count.min(MAX_GROUPS);
        ctx.mps_dispatch(cmd, kernel, &push, [x, count.div_ceil(x), 1]);
    }
}

/// Record a sparse step from `y` into `z`, with tables at `base`.
fn record_mix(ctx: &GpuContext, cmd: vk::CommandBuffer, base: u64, m: Mixing, y: u64, z: u64) {
    if m.total == 0 {
        return;
    }
    let mut push = Vec::with_capacity(40);
    for addr in [y, z, base + m.segments, base + m.entries] {
        push.extend_from_slice(&addr.to_le_bytes());
    }
    push.extend_from_slice(&(m.count as u32).to_le_bytes());
    push.extend_from_slice(&(m.total as u32).to_le_bytes());
    ctx.mps_dispatch(cmd, MpsKernel::BlockMix, &push, [groups(m.total), 1, 1]);
}

/// A block-sparse environment in GPU memory.
struct DeviceBlockEnv {
    layout: EnvLayout,
    buffer: Buffer,
}

/// Scratch beyond [`Scratch`]: tables and their staging, the site tensor,
/// and the intermediate blocks of an environment update.
#[derive(Default)]
struct BlockScratch {
    staging: Option<Buffer>,
    update_tables: Option<Buffer>,
    op_tables: Option<Buffer>,
    first: Option<Buffer>,
    mixed: Option<Buffer>,
}

/// The GPU side of a symmetric sweep.
pub(in crate::mps) struct GpuBlockSweep {
    ctx: Arc<GpuContext>,
    left: Vec<Option<DeviceBlockEnv>>,
    right: Vec<Option<DeviceBlockEnv>>,
    scratch: Mutex<Scratch>,
    blocks: Mutex<BlockScratch>,
    error: Mutex<Option<Error>>,
}

impl GpuBlockSweep {
    /// The boundary environments of `mps`, a symmetric MPS.
    pub(in crate::mps) fn new(ctx: &Arc<GpuContext>, mps: &Mps) -> Result<Self, Error> {
        let sym = mps.symmetry.as_ref().expect("a symmetric MPS");
        let n = mps.n_qubits();
        let one = [C64::new(1.0, 0.0)];
        let boundary = |side: Side, bond: &Sectors| -> Result<DeviceBlockEnv, Error> {
            Ok(DeviceBlockEnv {
                layout: EnvLayout::boundary(side, bond, &sym.charges),
                buffer: upload(ctx, bytes_of(&one))?,
            })
        };
        let mut left: Vec<Option<DeviceBlockEnv>> = (0..n).map(|_| None).collect();
        let mut right: Vec<Option<DeviceBlockEnv>> = (0..n).map(|_| None).collect();
        left[0] = Some(boundary(Side::Left, &sym.bonds[0])?);
        right[n - 1] = Some(boundary(Side::Right, &sym.bonds[n])?);
        Ok(GpuBlockSweep {
            ctx: ctx.clone(),
            left,
            right,
            scratch: Mutex::default(),
            blocks: Mutex::default(),
            error: Mutex::new(None),
        })
    }

    pub(in crate::mps) fn take_error(&self) -> Result<(), Error> {
        match self.error.lock().expect("GPU error lock").take() {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    pub(in crate::mps) fn release_left(&mut self, i: usize) {
        if i > 0 {
            self.left[i] = None;
        }
    }

    pub(in crate::mps) fn release_right(&mut self, i: usize) {
        if i + 1 < self.right.len() {
            self.right[i] = None;
        }
    }

    fn left_env(&self, i: usize) -> &DeviceBlockEnv {
        self.left[i].as_ref().expect("left environment built")
    }

    fn right_env(&self, i: usize) -> &DeviceBlockEnv {
        self.right[i].as_ref().expect("right environment built")
    }

    /// `left[i + 1]` from `left[i]` and site `i`'s tensor.
    pub(in crate::mps) fn update_left(
        &mut self,
        i: usize,
        mps: &Mps,
        mpo: &Mpo,
    ) -> Result<(), Error> {
        let sym = mps.symmetry.as_ref().expect("a symmetric MPS");
        let deltas = mpo.channels.as_ref().expect("a charge-graded MPO");
        let a = &mps.tensors[i];
        let env = self.left_env(i);
        let out = EnvLayout::new(Side::Left, &sym.bonds[i + 1], &deltas[i + 1], &sym.charges);
        let plan = left_update(&env.layout, &out, a, &mpo.sites[i], &sym.charges, i);
        let target = self.left[i + 1].take().map(|e| e.buffer);
        let buffer = self.run_update(&plan, &a.data, i, Side::Left, out.len, target)?;
        self.left[i + 1] = Some(DeviceBlockEnv {
            layout: out,
            buffer,
        });
        Ok(())
    }

    /// `right[i]` from `right[i + 1]` and site `i + 1`'s tensor.
    pub(in crate::mps) fn update_right(
        &mut self,
        i: usize,
        mps: &Mps,
        mpo: &Mpo,
    ) -> Result<(), Error> {
        let sym = mps.symmetry.as_ref().expect("a symmetric MPS");
        let deltas = mpo.channels.as_ref().expect("a charge-graded MPO");
        let a = &mps.tensors[i + 1];
        let env = self.right_env(i + 1);
        let out = EnvLayout::new(Side::Right, &sym.bonds[i + 1], &deltas[i + 1], &sym.charges);
        let plan = right_update(&env.layout, &out, a, &mpo.sites[i + 1], &sym.charges, i + 1);
        let target = self.right[i].take().map(|e| e.buffer);
        let buffer = self.run_update(&plan, &a.data, i + 1, Side::Right, out.len, target)?;
        self.right[i] = Some(DeviceBlockEnv {
            layout: out,
            buffer,
        });
        Ok(())
    }

    /// Run an environment update: upload the site tensor and the plan's
    /// tables, then products from the old environment and the site, the
    /// sparse step, and two passes of products into the new environment
    /// (zeroed first), returned in `target` if it is large enough. The old
    /// environment is `left[i]` or `right[i]`, by `side`.
    fn run_update(
        &self,
        plan: &UpdatePlan,
        site: &[C64],
        i: usize,
        side: Side,
        len: usize,
        target: Option<Buffer>,
    ) -> Result<Buffer, Error> {
        let ctx = &self.ctx;
        let env = match side {
            Side::Left => self.left_env(i).buffer.device_address(),
            Side::Right => self.right_env(i).buffer.device_address(),
        };
        let out_bytes = (len.max(1) as u64) * COMPLEX;
        let target = match target {
            Some(b) if b.size >= out_bytes => b,
            _ => Buffer::new(ctx, out_bytes, usage(), Memory::Device)?,
        };
        let mut tables = Tables::default();
        let first = tables.products(&plan.first);
        let mixing = tables.segments(&plan.segments);
        let passes = [
            tables.products(&plan.last[0]),
            tables.products(&plan.last[1]),
        ];
        tables.align();
        let site_at = tables.0.len() as u64;
        tables.0.extend_from_slice(bytes_of(site));

        let mut guard = self.blocks.lock().expect("GPU scratch lock");
        let s = &mut *guard;
        let bytes = tables.0.len() as u64;
        let staging = ensure(ctx, &mut s.staging, bytes, Memory::Upload)?;
        staging.write(&tables.0);
        let staging = staging.buffer;
        let device = ensure(ctx, &mut s.update_tables, bytes, Memory::Device)?;
        let (device, base) = (device.buffer, device.device_address());
        let x = ensure(
            ctx,
            &mut s.first,
            (plan.first_len.max(1) * 16) as u64,
            Memory::Device,
        )?
        .device_address();
        let y = ensure(
            ctx,
            &mut s.mixed,
            (plan.mixed_len.max(1) * 16) as u64,
            Memory::Device,
        )?
        .device_address();
        let a = base + site_at;
        let out = target.device_address();
        let (first_ops, last_ops) = match side {
            // x = L_s A, then L' += A† y.
            Side::Left => ((env, a, x), (a, y, out)),
            // x = R_t A†, then R'ᵀ += A y.
            Side::Right => ((env, a, x), (a, y, out)),
        };
        let (first_flags, last_flags) = match side {
            Side::Left => (0, CONJ_A | ADD),
            Side::Right => (CONJ_B, ADD),
        };
        ctx.submit(|cmd| {
            copy(ctx, cmd, staging, device, bytes);
            // SAFETY: recording; the buffer holds `out_bytes`.
            unsafe {
                ctx.device
                    .cmd_fill_buffer(cmd, target.buffer, 0, out_bytes, 0)
            };
            compute_barrier(ctx, cmd);
            record_products(ctx, cmd, base, first, first_ops, first_flags);
            compute_barrier(ctx, cmd);
            record_mix(ctx, cmd, base, mixing, x, y);
            compute_barrier(ctx, cmd);
            record_products(ctx, cmd, base, passes[0], last_ops, last_flags);
            compute_barrier(ctx, cmd);
            record_products(ctx, cmd, base, passes[1], last_ops, last_flags);
        })?;
        Ok(target)
    }

    /// The effective Hamiltonian of sites `i` and `i + 1`.
    pub(in crate::mps) fn two_site(
        &self,
        i: usize,
        mps: &Mps,
        mpo: &Mpo,
    ) -> Result<GpuBlockOp<'_>, Error> {
        let sym = mps.symmetry.as_ref().expect("a symmetric MPS");
        let (left, right) = (self.left_env(i), self.right_env(i + 1));
        let plan = Plan::two_site(
            &left.layout,
            &right.layout,
            &mpo.sites[i],
            &mpo.sites[i + 1],
            &sym.charges,
            i,
        );
        GpuBlockOp::new(self, left, right, plan)
    }

    /// The effective Hamiltonian of site `i`.
    pub(in crate::mps) fn one_site(
        &self,
        i: usize,
        mps: &Mps,
        mpo: &Mpo,
    ) -> Result<GpuBlockOp<'_>, Error> {
        let sym = mps.symmetry.as_ref().expect("a symmetric MPS");
        let (left, right) = (self.left_env(i), self.right_env(i));
        let plan = Plan::one_site(&left.layout, &right.layout, &mpo.sites[i], &sym.charges, i);
        GpuBlockOp::new(self, left, right, plan)
    }

    /// `left[i]` copied to the CPU.
    pub(in crate::mps) fn download_left(&self, i: usize) -> Result<BlockEnv, Error> {
        self.download(self.left_env(i))
    }

    /// `right[i]` copied to the CPU.
    pub(in crate::mps) fn download_right(&self, i: usize) -> Result<BlockEnv, Error> {
        self.download(self.right_env(i))
    }

    fn download(&self, env: &DeviceBlockEnv) -> Result<BlockEnv, Error> {
        let len = env.layout.len;
        let bytes = (len.max(1) as u64) * COMPLEX;
        let mut guard = self.scratch.lock().expect("GPU scratch lock");
        let down = ensure(&self.ctx, &mut guard.down, bytes, Memory::Download)?;
        let dst = down.buffer;
        self.ctx
            .submit(|cmd| copy(&self.ctx, cmd, env.buffer.buffer, dst, bytes))?;
        let data = read_complex(down.mapped(bytes as usize)?, len);
        Ok(BlockEnv {
            layout: env.layout.clone(),
            data,
        })
    }
}

/// The products and sparse step of an environment update.
struct UpdatePlan {
    first: Vec<Product>,
    first_len: usize,
    segments: Vec<Segment>,
    mixed_len: usize,
    /// Products into the new environment, in two passes that each write
    /// a block at most once.
    last: [Vec<Product>; 2],
}

/// The plan of [`update_left`](crate::mps::blocks::update_left) for the
/// GPU: `x[s, p] = L_s A_(s, p)`, then for each new block
/// `L'_b[t_out, t_in] = Σ_{p_out} A_(s_out, p_out)† y`, with `y` the MPO's
/// sum of rows of the `x[s_in, p_in]`.
fn left_update(
    env: &EnvLayout,
    out: &EnvLayout,
    a: &crate::mps::Tensor,
    w: &MpoSite,
    charges: &Charges,
    q: usize,
) -> UpdatePlan {
    let blocks = SiteBlocks::new(a, &env.bond, &out.bond, charges, q);
    let stride = 2 * a.right;
    let mut first = Vec::new();
    let mut x_at = vec![None; env.bond.len() * 2];
    let mut first_len = 0;
    for s in 0..env.bond.len() {
        for p in 0..2 {
            let Some((t, at)) = blocks.offset(s, p) else {
                continue;
            };
            let stack = &env.stacks[s];
            let (d, dt) = (env.bond.size(s), out.bond.size(t));
            x_at[s * 2 + p] = Some(first_len);
            first.push(Product {
                a: stack.offset,
                b: at,
                c: first_len,
                m: stack.rows,
                n: dt,
                k: d,
                a_row: d,
                a_col: 1,
                b_row: stride,
                b_col: 1,
                c_row: dt,
            });
            first_len += stack.rows * dt;
        }
    }
    let rows = right_keyed(w);
    let mut segments = Vec::new();
    let mut mixed_len = 0;
    let mut last = [Vec::new(), Vec::new()];
    for (t_in, stack) in out.stacks.iter().enumerate() {
        let cols = stack.cols;
        for (b, entry) in stack.channels.iter().enumerate() {
            let Some((row0, t_out)) = *entry else {
                continue;
            };
            for p_out in 0..2 {
                let Some(s_out) = blocks.left_of[t_out * 2 + p_out] else {
                    continue;
                };
                let (_, bra) = blocks
                    .offset(s_out, p_out)
                    .expect("the block reaches t_out");
                let d_out = env.bond.size(s_out);
                let mut entries = Vec::new();
                for &(ch, p_in, f) in &rows[b * 2 + p_out] {
                    let Some(s_in) = blocks.left_of[t_in * 2 + p_in] else {
                        continue;
                    };
                    let (Some(x), Some((r, k))) =
                        (x_at[s_in * 2 + p_in], env.stacks[s_in].channels[ch])
                    else {
                        continue;
                    };
                    debug_assert_eq!(k, s_out);
                    entries.push((f, x + r * cols, cols));
                }
                if entries.is_empty() {
                    continue;
                }
                segments.push(Segment {
                    z: mixed_len,
                    z_row: cols,
                    rows: d_out,
                    cols,
                    entries,
                });
                last[p_out].push(Product {
                    a: bra,
                    b: mixed_len,
                    c: stack.offset + row0 * cols,
                    m: out.bond.size(t_out),
                    n: cols,
                    k: d_out,
                    a_row: 1,
                    a_col: stride,
                    b_row: cols,
                    b_col: 1,
                    c_row: cols,
                });
                mixed_len += d_out * cols;
            }
        }
    }
    UpdatePlan {
        first,
        first_len,
        segments,
        mixed_len,
        last,
    }
}

/// The plan of [`update_right`](crate::mps::blocks::update_right) for the
/// GPU: `x[s, p] = R_t A_(s, p)†`, then for each new block
/// `R'_aᵀ[s_in, s_out] = Σ_{p_in} A_(s_in, p_in) y`.
fn right_update(
    env: &EnvLayout,
    out: &EnvLayout,
    a: &crate::mps::Tensor,
    w: &MpoSite,
    charges: &Charges,
    q: usize,
) -> UpdatePlan {
    let blocks = SiteBlocks::new(a, &out.bond, &env.bond, charges, q);
    let stride = 2 * a.right;
    let mut first = Vec::new();
    let mut x_at = vec![None; out.bond.len() * 2];
    let mut first_len = 0;
    for s in 0..out.bond.len() {
        for p in 0..2 {
            let Some((t, at)) = blocks.offset(s, p) else {
                continue;
            };
            let stack = &env.stacks[t];
            let (d, dt) = (out.bond.size(s), env.bond.size(t));
            x_at[s * 2 + p] = Some(first_len);
            first.push(Product {
                a: stack.offset,
                b: at,
                c: first_len,
                m: stack.rows,
                n: d,
                k: dt,
                a_row: dt,
                a_col: 1,
                b_row: 1,
                b_col: stride,
                c_row: d,
            });
            first_len += stack.rows * d;
        }
    }
    let rows = left_keyed(w);
    let mut segments = Vec::new();
    let mut mixed_len = 0;
    let mut last = [Vec::new(), Vec::new()];
    for (s_out, stack) in out.stacks.iter().enumerate() {
        let cols = stack.cols;
        for (a_ch, entry) in stack.channels.iter().enumerate() {
            let Some((row0, s_in)) = *entry else {
                continue;
            };
            for (p_in, pass) in last.iter_mut().enumerate() {
                let Some((t_in, ket)) = blocks.offset(s_in, p_in) else {
                    continue;
                };
                let mut entries = Vec::new();
                for p_out in 0..2 {
                    let (Some(x), Some(t_out)) =
                        (x_at[s_out * 2 + p_out], blocks.right_of[s_out * 2 + p_out])
                    else {
                        continue;
                    };
                    for &(c, p, f) in &rows[a_ch * 2 + p_out] {
                        if p != p_in {
                            continue;
                        }
                        let Some((r, k)) = env.stacks[t_out].channels[c] else {
                            continue;
                        };
                        debug_assert_eq!(k, t_in);
                        entries.push((f, x + r * cols, cols));
                    }
                }
                if entries.is_empty() {
                    continue;
                }
                let d_t = env.bond.size(t_in);
                segments.push(Segment {
                    z: mixed_len,
                    z_row: cols,
                    rows: d_t,
                    cols,
                    entries,
                });
                pass.push(Product {
                    a: ket,
                    b: mixed_len,
                    c: stack.offset + row0 * cols,
                    m: out.bond.size(s_in),
                    n: cols,
                    k: d_t,
                    a_row: stride,
                    a_col: 1,
                    b_row: cols,
                    b_col: 1,
                    c_row: cols,
                });
                mixed_len += d_t * cols;
            }
        }
    }
    UpdatePlan {
        first,
        first_len,
        segments,
        mixed_len,
        last,
    }
}

/// The block-sparse effective Hamiltonian of one or two sites on the GPU,
/// on compact vectors (see [`Layout`]).
pub(in crate::mps) struct GpuBlockOp<'a> {
    sweep: &'a GpuBlockSweep,
    left: u64,
    right: u64,
    layout: Layout,
    /// The tables' device address, and each step's place in them.
    base: u64,
    first: Grouped,
    mixing: Mixing,
    last: Grouped,
    y_len: usize,
    z_len: usize,
    flops: usize,
}

impl<'a> GpuBlockOp<'a> {
    fn new(
        sweep: &'a GpuBlockSweep,
        left: &DeviceBlockEnv,
        right: &DeviceBlockEnv,
        plan: Plan,
    ) -> Result<Self, Error> {
        let rows = &plan.layout.rows;
        let mut first = Vec::with_capacity(rows.len());
        for (rb, &(s, d, offset, width)) in rows.iter().enumerate() {
            let (y, height) = plan.y[rb];
            first.push(Product {
                a: left.layout.stacks[s].offset,
                b: offset,
                c: y,
                m: height,
                n: width,
                k: d,
                a_row: d,
                a_col: 1,
                b_row: width,
                b_col: 1,
                c_row: width,
            });
        }
        let mut segments = Vec::new();
        let mut last = Vec::new();
        let mut z_len = 0;
        for (&(_, d, offset, width), blocks) in rows.iter().zip(&plan.out) {
            for block in blocks {
                let stack = &right.layout.stacks[block.t];
                let k = stack.rows;
                for entry in &stack.channels {
                    let Some((z_col, t_in)) = *entry else {
                        continue;
                    };
                    let entries = block
                        .copies
                        .iter()
                        .filter(|copy| copy.z_col == z_col)
                        .map(|copy| (copy.f, copy.y, copy.stride))
                        .collect();
                    segments.push(Segment {
                        z: z_len + z_col,
                        z_row: k,
                        rows: d,
                        cols: right.layout.bond.size(t_in),
                        entries,
                    });
                }
                last.push(Product {
                    a: z_len,
                    b: stack.offset,
                    c: offset + block.column,
                    m: d,
                    n: stack.cols,
                    k,
                    a_row: k,
                    a_col: 1,
                    b_row: stack.cols,
                    b_col: 1,
                    c_row: width,
                });
                z_len += d * k;
            }
        }
        let flops = first.iter().chain(&last).map(Product::flops).sum();
        let mut tables = Tables::default();
        let first_g = tables.products(&first);
        let mixing = tables.segments(&segments);
        let last_g = tables.products(&last);
        let ctx = &sweep.ctx;
        let bytes = tables.0.len() as u64;
        let mut guard = sweep.blocks.lock().expect("GPU scratch lock");
        let s = &mut *guard;
        let staging = ensure(ctx, &mut s.staging, bytes, Memory::Upload)?;
        staging.write(&tables.0);
        let staging = staging.buffer;
        let device = ensure(ctx, &mut s.op_tables, bytes, Memory::Device)?;
        let (device, base) = (device.buffer, device.device_address());
        ctx.submit(|cmd| copy(ctx, cmd, staging, device, bytes))?;
        Ok(GpuBlockOp {
            sweep,
            left: left.buffer.device_address(),
            right: right.buffer.device_address(),
            y_len: plan.y_len,
            layout: plan.layout,
            base,
            first: first_g,
            mixing,
            last: last_g,
            z_len,
            flops,
        })
    }

    /// The compact layout this operator acts on.
    pub(in crate::mps) fn layout(&self) -> &Layout {
        &self.layout
    }

    /// `amps ← exp(-iHt) amps`, with the recurrence in GPU memory.
    pub(in crate::mps) fn evolve(&self, amps: &mut [C64], time: f64) -> Result<(), Error> {
        evolve(self, amps, time)
    }

    /// The lowest Ritz pair, with the recurrence in GPU memory.
    pub(in crate::mps) fn lowest(
        &self,
        v0: &[C64],
        options: &EigenOptions,
    ) -> Result<Ritz<C64>, Error> {
        lowest_within(self, v0, options, LANCZOS_BYTES)
    }
}

impl DeviceOp for GpuBlockOp<'_> {
    fn ctx(&self) -> &Arc<GpuContext> {
        &self.sweep.ctx
    }

    fn scratch(&self) -> &Mutex<Scratch> {
        &self.sweep.scratch
    }

    fn len(&self) -> usize {
        self.layout.len
    }

    fn flops(&self) -> usize {
        self.flops
    }

    fn prepare(&self, s: &mut Scratch) -> Result<ApplyScratch, Error> {
        let ctx = &self.sweep.ctx;
        Ok(ApplyScratch {
            y: ensure(
                ctx,
                &mut s.y,
                (self.y_len.max(1) as u64) * COMPLEX,
                Memory::Device,
            )?
            .device_address(),
            z: ensure(
                ctx,
                &mut s.z,
                (self.z_len.max(1) as u64) * COMPLEX,
                Memory::Device,
            )?
            .device_address(),
            parts: 0,
            max_splits: 1,
        })
    }

    fn record_apply(&self, cmd: vk::CommandBuffer, x: u64, out: u64, s: ApplyScratch) {
        let ctx = &self.sweep.ctx;
        // Y_s = L_s Θ_s.
        record_products(ctx, cmd, self.base, self.first, (self.left, x, s.y), 0);
        compute_barrier(ctx, cmd);
        // Z: the MPO step.
        record_mix(ctx, cmd, self.base, self.mixing, s.y, s.z);
        compute_barrier(ctx, cmd);
        // out_(s, P) = Z_(s, P) R_t.
        record_products(ctx, cmd, self.base, self.last, (s.z, self.right, out), 0);
    }

    fn errors(&self) -> &Mutex<Option<Error>> {
        &self.sweep.error
    }
}

impl LinearOperator<C64> for GpuBlockOp<'_> {
    fn apply(&self, x: &[C64], out: &mut [C64]) {
        super::apply_on_device(self, x, out);
    }
}

#[cfg(test)]
mod tests {
    use crate::gpu::GpuContext;
    use crate::mps::envs::{Envs, Op};
    use crate::mps::{Charge, DmrgOptions, Mps, MpsOptions};
    use crate::observable::{PauliString, PauliSum};
    use crate::statevector::eigen::EigenOptions;
    use crate::statevector::krylov::LinearOperator;
    use num_complex::Complex64 as C64;
    use std::sync::Arc;

    fn gpu() -> Option<Arc<GpuContext>> {
        match GpuContext::new() {
            Ok(ctx) if ctx.supports_mps() => Some(ctx),
            _ => {
                eprintln!("skipping GPU symmetric MPS test: no usable GPU");
                None
            }
        }
    }

    fn label(n: usize, ops: &[(usize, char)]) -> PauliString {
        let s: String = (0..n)
            .rev()
            .map(|k| ops.iter().find(|o| o.0 == k).map_or('I', |o| o.1))
            .collect();
        PauliString::from_label(&s).unwrap()
    }

    /// A spinful Hubbard chain through a Jordan–Wigner string, with spin up
    /// and down particle numbers as charges.
    fn hubbard(sites: usize) -> (PauliSum, Vec<Charge>, Vec<bool>) {
        let n = 2 * sites;
        let mut h = PauliSum::new();
        for j in 0..sites - 1 {
            for s in 0..2 {
                let (a, b) = (2 * j + s, 2 * j + s + 2);
                for p in ['X', 'Y'] {
                    h.add(
                        -0.5 - 0.05 * j as f64,
                        label(n, &[(a, p), (a + 1, 'Z'), (b, p)]),
                    );
                }
            }
        }
        for j in 0..sites {
            h.add(0.5, label(n, &[(2 * j, 'Z'), (2 * j + 1, 'Z')]));
            h.add(0.1 * j as f64, label(n, &[(2 * j, 'Z')]));
        }
        let up = Charge::weights((0..n).map(|q| i64::from(q % 2 == 0)).collect());
        let down = Charge::weights((0..n).map(|q| i64::from(q % 2 == 1)).collect());
        let bits = (0..n).map(|q| [0, 3, 4, 7, 8, 11].contains(&q)).collect();
        (h, vec![up, down], bits)
    }

    fn max_diff(a: &[C64], b: &[C64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).norm())
            .fold(0.0, f64::max)
    }

    #[test]
    fn block_environments_and_operators_match_the_cpu() {
        let Some(ctx) = gpu() else { return };
        let (h, charges, bits) = hubbard(6);
        let n = bits.len();
        // A state with several sectors per bond: a few TDVP steps.
        let mut mps = Mps::basis(n, &bits, MpsOptions::default())
            .with_charges(&charges)
            .unwrap();
        mps.tdvp(&h, 0.6, 3).unwrap();
        mps.move_center(0);
        let mpo = mps.mpo(&h).unwrap();
        let mut cpu = Envs::new(&mps, &mpo).unwrap();
        let mut on_gpu = mps.clone();
        on_gpu.set_gpu(Some(ctx)).unwrap();
        let mut gpu = Envs::new(&on_gpu, &mpo).unwrap();
        assert!(matches!(gpu, Envs::GpuBlocks(_)));
        for i in (0..n - 1).rev() {
            cpu.update_right(i, &mps, &mpo).unwrap();
            gpu.update_right(i, &mps, &mpo).unwrap();
            let (Envs::Blocks { right, .. }, Envs::GpuBlocks(g)) = (&cpu, &gpu) else {
                unreachable!()
            };
            let want = right[i].as_ref().unwrap();
            let got = g.download_right(i).unwrap();
            assert!(max_diff(&got.data, &want.data) < 1e-12, "right {i}");
        }
        let mut seed = 7u64;
        let mut random = |len: usize| -> Vec<C64> {
            (0..len)
                .map(|_| {
                    seed = seed
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let a = (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
                    let b = (seed >> 20) as f64 / (1u64 << 44) as f64 - 0.5;
                    C64::new(a, b)
                })
                .collect()
        };
        for i in 0..n - 1 {
            // Two sites at i, one site at i.
            for two in [true, false] {
                let (c_op, g_op) = if two {
                    (
                        cpu.two_site(i, &mps, &mpo).unwrap(),
                        gpu.two_site(i, &mps, &mpo).unwrap(),
                    )
                } else {
                    (
                        cpu.one_site(i, &mps, &mpo).unwrap(),
                        gpu.one_site(i, &mps, &mpo).unwrap(),
                    )
                };
                let (Op::Blocks(c), Op::GpuBlocks(g)) = (&c_op, &g_op) else {
                    unreachable!()
                };
                let len = c.layout().len;
                let x = random(len);
                let (mut want, mut got) =
                    (vec![C64::new(0.0, 0.0); len], vec![C64::new(0.0, 0.0); len]);
                c.apply(&x, &mut want);
                g.apply(&x, &mut got);
                let scale = want.iter().map(|v| v.norm()).fold(1.0, f64::max);
                assert!(max_diff(&got, &want) < 1e-12 * scale, "apply {i} {two}");
                let (mut a, mut b) = (x.clone(), x.clone());
                c_op.evolve(&mut a, 0.7).unwrap();
                g_op.evolve(&mut b, 0.7).unwrap();
                assert!(max_diff(&a, &b) < 1e-10, "evolve {i} {two}");
                if two {
                    let norm = x.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
                    let v0: Vec<C64> = x.iter().map(|v| v / norm).collect();
                    let options = EigenOptions {
                        tolerance: 1e-10,
                        max_iterations: 300,
                        seed: 0,
                    };
                    let want = c_op.lowest(&v0, &options).unwrap();
                    let got = g_op.lowest(&v0, &options).unwrap();
                    assert!((want.value - got.value).abs() < 1e-9, "lowest {i}");
                }
            }
            cpu.update_left(i, &mps, &mpo).unwrap();
            gpu.update_left(i, &mps, &mpo).unwrap();
            let (Envs::Blocks { left, .. }, Envs::GpuBlocks(g)) = (&cpu, &gpu) else {
                unreachable!()
            };
            let want = left[i + 1].as_ref().unwrap();
            let got = g.download_left(i + 1).unwrap();
            assert!(max_diff(&got.data, &want.data) < 1e-12, "left {i}");
        }
    }

    #[test]
    fn symmetric_sweeps_on_the_gpu_match_the_cpu() {
        let Some(ctx) = gpu() else { return };
        let (h, charges, bits) = hubbard(5);
        let n = bits.len();
        let start = Mps::basis(n, &bits, MpsOptions::default())
            .with_charges(&charges)
            .unwrap();
        let mut cpu = start.clone();
        let mut gpu = start.clone();
        gpu.set_gpu(Some(ctx)).unwrap();
        cpu.tdvp(&h, 0.8, 6).unwrap();
        gpu.tdvp(&h, 0.8, 6).unwrap();
        let diff = max_diff(&cpu.to_amplitudes(), &gpu.to_amplitudes());
        assert!(diff < 1e-9, "TDVP: {diff}");
        let options = DmrgOptions {
            max_bond: 32,
            ..Default::default()
        };
        let want = cpu.dmrg(&h, &options).unwrap();
        let got = gpu.dmrg(&h, &options).unwrap();
        assert!(
            (want.energy - got.energy).abs() < 1e-9,
            "{} vs {}",
            want.energy,
            got.energy
        );
    }
}
