//! GPU statevectors through Vulkan compute.
//!
//! Works on any Vulkan 1.2 GPU with 64-bit floats (NVIDIA, AMD and others)
//! with nothing but the system Vulkan driver, loaded at run time. The GLSL
//! shaders in `shaders/` are compiled to SPIR-V by `build.rs` with glslc.
//!
//! A circuit is fused (see [`Circuit::fused`]) and planned into dispatches:
//! runs of gates touching at most ten qubits become one shared-memory tile
//! pass, everything else one dispatch per gate, with every gate matrix in a
//! single parameter buffer.
//!
//! Amplitudes are complex doubles, `16 · 2ⁿ` bytes in all. A state larger
//! than one storage buffer (4 GiB or just under, so 27 or 28 qubits) is
//! split over up to sixteen buffers, which kernels index per amplitude;
//! that needs Vulkan 1.2's non-uniform descriptor indexing and allows up to
//! 31 qubits, memory permitting. Expectation values and sampling run on the GPU, downloading
//! only partial sums and the sampled blocks.

mod context;
mod evolve;
mod gradient;

pub use context::GpuContext;

use crate::circuit::{Circuit, Operation};
use crate::error::Error;
use crate::fusion::local_matrix;
use crate::observable::PauliSum;
use crate::statevector::Statevector;
use crate::statevector::hamiltonian::{GroupedPauliSum, Imaginary};
use ash::vk;
use context::{Buffer, Kernel, MAX_CHUNKS, Memory, Push};
use num_complex::Complex64 as C64;
use rand::Rng;
use std::collections::BTreeMap;
use std::sync::Arc;

const AMPLITUDE_BYTES: u64 = size_of::<C64>() as u64;

/// Work per submission, in amplitudes touched. 2²⁸ is about four passes
/// over a 26-qubit state, tens of milliseconds, far below driver timeouts
/// (amdgpu resets a context whose submission runs for around a second).
const BATCH_UNITS: u64 = 1 << 28;

/// Amplitudes per block when sampling, as a power of two: the GPU totals
/// each block's probability, and only sampled blocks are downloaded.
const SAMPLE_BLOCK_BITS: u32 = 10;

/// A statevector held in GPU memory, with complex `f64` amplitudes.
pub struct GpuStatevector {
    ctx: Arc<GpuContext>,
    n_qubits: usize,
    /// The amplitudes, `2^chunk_bits` per buffer, in index order.
    chunks: Vec<Buffer>,
    chunk_bits: u32,
    /// Gate parameters for the current circuit, bound at binding 1.
    params: Buffer,
    /// Partial sums from reductions, bound at binding 2.
    output: Buffer,
    set: vk::DescriptorSet,
}

impl GpuStatevector {
    /// `|0…0⟩` on `n_qubits` qubits.
    pub fn zero(ctx: &Arc<GpuContext>, n_qubits: usize) -> Result<Self, Error> {
        let chunk_bits = (n_qubits as u32).min(ctx.max_chunk_bits());
        Self::zero_with_chunk_bits(ctx, n_qubits, chunk_bits)
    }

    /// `|0…0⟩` with the state split into buffers of `2^chunk_bits`
    /// amplitudes. For testing the multi-buffer kernels on small states.
    #[doc(hidden)]
    pub fn zero_with_chunk_bits(
        ctx: &Arc<GpuContext>,
        n_qubits: usize,
        chunk_bits: u32,
    ) -> Result<Self, Error> {
        let state = Self::allocate(ctx, n_qubits, chunk_bits)?;
        state.set_basis(0)?;
        Ok(state)
    }

    /// Upload a statevector's amplitudes (length a power of two).
    pub fn from_amplitudes(ctx: &Arc<GpuContext>, amplitudes: &[C64]) -> Result<Self, Error> {
        if !amplitudes.len().is_power_of_two() {
            return Err(Error::InvalidLength(amplitudes.len()));
        }
        let n = amplitudes.len().trailing_zeros();
        let state = Self::allocate(ctx, n as usize, n.min(ctx.max_chunk_bits()))?;
        state.upload(amplitudes)?;
        Ok(state)
    }

    /// Run `circuit` on `|0…0⟩`.
    pub fn from_circuit(ctx: &Arc<GpuContext>, circuit: &Circuit) -> Result<Self, Error> {
        let mut state = Self::zero(ctx, circuit.n_qubits())?;
        state.run(circuit)?;
        Ok(state)
    }

    fn allocate(ctx: &Arc<GpuContext>, n_qubits: usize, chunk_bits: u32) -> Result<Self, Error> {
        let chunk_bits = chunk_bits.min(n_qubits as u32);
        let n_chunks = 1usize << (n_qubits as u32 - chunk_bits).min(31);
        let too_big = |why: String| {
            Error::Gpu(format!(
                "{n_qubits} qubits needs {} MiB of GPU memory; {why}",
                (16u128 << n_qubits) >> 20
            ))
        };
        if n_qubits > 31 || n_chunks > MAX_CHUNKS {
            return Err(too_big(format!(
                "{} holds at most {} amplitudes per buffer and {MAX_CHUNKS} buffers",
                ctx.device_name(),
                1u64 << ctx.max_chunk_bits()
            )));
        }
        if n_chunks > 1 && !ctx.chunks_supported() {
            return Err(too_big(format!(
                "{} can't split a state across buffers (no non-uniform descriptor indexing)",
                ctx.device_name()
            )));
        }
        let chunks = (0..n_chunks)
            .map(|_| {
                Buffer::new(
                    ctx,
                    AMPLITUDE_BYTES << chunk_bits,
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_SRC
                        | vk::BufferUsageFlags::TRANSFER_DST,
                    Memory::Device,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let params = Buffer::new(
            ctx,
            256,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            Memory::Upload,
        )?;
        let output = Buffer::new(
            ctx,
            256,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            Memory::Download,
        )?;
        let set = {
            let _guard = ctx.lock.lock().expect("GPU lock poisoned");
            let layouts = [ctx.descriptor_layout];
            // SAFETY: the pool is used only under the lock.
            unsafe {
                ctx.device.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(ctx.descriptor_pool)
                        .set_layouts(&layouts),
                )
            }
            .map_err(|e| Error::Gpu(format!("allocating descriptor set: {e}")))?[0]
        };
        let gpu = GpuStatevector {
            ctx: ctx.clone(),
            n_qubits,
            chunks,
            chunk_bits,
            params,
            output,
            set,
        };
        gpu.bind_buffers();
        Ok(gpu)
    }

    /// Point the descriptor set at the state chunks, parameters and output.
    /// Unused chunk slots repeat the first chunk, so every slot is valid.
    fn bind_buffers(&self) {
        let info = |b: &Buffer| {
            [vk::DescriptorBufferInfo::default()
                .buffer(b.buffer)
                .range(vk::WHOLE_SIZE)]
        };
        let chunks: Vec<_> = (0..MAX_CHUNKS)
            .map(|i| info(&self.chunks[i % self.chunks.len()])[0])
            .collect();
        let (params, output) = (info(&self.params), info(&self.output));
        let writes = [
            write_set(self.set, 0, &chunks),
            write_set(self.set, 1, &params),
            write_set(self.set, 2, &output),
            write_set(self.set, 3, &chunks),
        ];
        // SAFETY: the set isn't in use: every submission is waited on.
        unsafe { self.ctx.device.update_descriptor_sets(&writes, &[]) };
    }

    /// Number of qubits.
    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    /// The device this state lives on.
    pub fn context(&self) -> &Arc<GpuContext> {
        &self.ctx
    }

    fn len(&self) -> usize {
        1 << self.n_qubits
    }

    fn chunked(&self) -> bool {
        self.chunks.len() > 1
    }

    fn chunk_len(&self) -> usize {
        1 << self.chunk_bits
    }

    /// Set the state to basis state `index`.
    fn set_basis(&self, index: usize) -> Result<(), Error> {
        let one: [u8; 16] = bytes_of(&[C64::new(1.0, 0.0)])
            .try_into()
            .expect("16 bytes");
        let (chunk, offset) = (index >> self.chunk_bits, index & (self.chunk_len() - 1));
        self.ctx.submit(|cmd| {
            // SAFETY: recording transfers into this state's buffers.
            unsafe {
                let d = &self.ctx.device;
                for c in &self.chunks {
                    d.cmd_fill_buffer(cmd, c.buffer, 0, vk::WHOLE_SIZE, 0);
                }
                self.ctx.barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TRANSFER,
                );
                d.cmd_update_buffer(
                    cmd,
                    self.chunks[chunk].buffer,
                    offset as u64 * AMPLITUDE_BYTES,
                    &one,
                );
            }
        })
    }

    fn upload(&self, amplitudes: &[C64]) -> Result<(), Error> {
        let mut staging = Buffer::new(
            &self.ctx,
            amplitudes.len() as u64 * AMPLITUDE_BYTES,
            vk::BufferUsageFlags::TRANSFER_SRC,
            Memory::Upload,
        )?;
        staging.write(bytes_of(amplitudes));
        let size = AMPLITUDE_BYTES << self.chunk_bits;
        self.ctx.submit(|cmd| {
            for (i, chunk) in self.chunks.iter().enumerate() {
                let region = [vk::BufferCopy::default()
                    .src_offset(i as u64 * size)
                    .size(size)];
                // SAFETY: the staging buffer holds every chunk in order.
                unsafe {
                    self.ctx
                        .device
                        .cmd_copy_buffer(cmd, staging.buffer, chunk.buffer, &region)
                };
            }
        })
    }

    /// Copy `(index, count)` ranges of amplitudes back to the CPU, each
    /// within one chunk, concatenated.
    fn download_ranges(&self, ranges: &[(usize, usize)]) -> Result<Vec<C64>, Error> {
        let total: usize = ranges.iter().map(|&(_, n)| n).sum();
        let staging = Buffer::new(
            &self.ctx,
            total as u64 * AMPLITUDE_BYTES,
            vk::BufferUsageFlags::TRANSFER_DST,
            Memory::Download,
        )?;
        self.ctx.submit(|cmd| {
            let mut dst = 0u64;
            for &(index, count) in ranges {
                let chunk = &self.chunks[index >> self.chunk_bits];
                let region = [vk::BufferCopy::default()
                    .src_offset((index & (self.chunk_len() - 1)) as u64 * AMPLITUDE_BYTES)
                    .dst_offset(dst)
                    .size(count as u64 * AMPLITUDE_BYTES)];
                // SAFETY: ranges lie within their chunk and the staging buffer.
                unsafe {
                    self.ctx
                        .device
                        .cmd_copy_buffer(cmd, chunk.buffer, staging.buffer, &region)
                };
                dst += count as u64 * AMPLITUDE_BYTES;
            }
        })?;
        let mut out = vec![C64::new(0.0, 0.0); total];
        staging.read(bytes_of_mut(&mut out))?;
        Ok(out)
    }

    /// Download the amplitudes.
    pub fn amplitudes(&self) -> Result<Vec<C64>, Error> {
        let ranges: Vec<_> = (0..self.chunks.len())
            .map(|c| (c << self.chunk_bits, self.chunk_len()))
            .collect();
        self.download_ranges(&ranges)
    }

    /// Download into a CPU [`Statevector`].
    pub fn to_statevector(&self) -> Result<Statevector, Error> {
        Statevector::from_amplitudes(self.amplitudes()?)
    }

    /// `⟨ψ|H|ψ⟩` for a Hermitian `observable`, computed on the GPU, assuming
    /// `ψ` is normalised. Terms sharing an X-mask take one pass between them;
    /// only per-workgroup partial sums are downloaded.
    pub fn expectation(&mut self, observable: &PauliSum) -> Result<f64, Error> {
        let grouped = GroupedPauliSum::<C64>::new(observable, self.n_qubits, Imaginary::Reject)?;
        // Each group as (x mask, [(z mask, coefficient)]); Z-type terms have
        // x = 0.
        let mut groups: Vec<(u32, Vec<(u32, C64)>)> = Vec::new();
        if !grouped.diagonal.is_empty() {
            let terms = grouped
                .diagonal
                .iter()
                .map(|&(z, c)| (z as u32, C64::new(c, 0.0)))
                .collect();
            groups.push((0, terms));
        }
        for g in &grouped.groups {
            groups.push((
                g.x as u32,
                g.terms.iter().map(|&(z, c)| (z as u32, c)).collect(),
            ));
        }
        let sum = self.reduce(&groups)?;
        Ok(sum.re)
    }

    /// `⟨ψ|ψ⟩`, computed on the GPU.
    pub fn norm_sqr(&mut self) -> Result<f64, Error> {
        Ok(self.reduce(&[(0, vec![(0, C64::new(1.0, 0.0))])])?.re)
    }

    /// `Σ_groups Σ_j conj(ψ_j) c(j ⊕ x) ψ_{j⊕x}` on the GPU.
    fn reduce(&mut self, groups: &[(u32, Vec<(u32, C64)>)]) -> Result<C64, Error> {
        let units = self.len() as u32;
        let per_group = Kernel::Expect.groups(units) as usize;
        let mut params = Vec::new();
        let mut dispatches = Vec::new();
        for (k, (x, terms)) in groups.iter().enumerate() {
            let offset = params.len() as u32;
            for &(z, c) in terms {
                params.extend([z as f64, c.re, c.im]);
            }
            dispatches.push((
                Kernel::Expect,
                Push {
                    a: *x,
                    b: terms.len() as u32,
                    c: (k * per_group) as u32,
                    units,
                    offset,
                    chunk_bits: 0,
                },
            ));
        }
        self.ensure_buffers(&params, groups.len() * per_group * AMPLITUDE_BYTES as usize)?;
        for batch in dispatches.chunks((BATCH_UNITS / units as u64).max(1) as usize) {
            self.submit_dispatches(batch)?;
        }
        let mut partials = vec![C64::new(0.0, 0.0); groups.len() * per_group];
        self.output.read(bytes_of_mut(&mut partials))?;
        Ok(partials.iter().sum())
    }

    /// Draw `shots` basis-state indices from the Born distribution on the
    /// GPU, without changing the state. The GPU totals the probability of
    /// each block of amplitudes; only the blocks that are sampled come back.
    pub fn sample(&mut self, shots: usize, rng: &mut impl Rng) -> Result<Vec<usize>, Error> {
        let block_bits = SAMPLE_BLOCK_BITS.min(self.n_qubits as u32);
        let blocks = self.len() >> block_bits;
        self.ensure_buffers(&[], blocks * AMPLITUDE_BYTES as usize)?;
        self.submit_dispatches(&[(
            Kernel::Probs,
            Push {
                c: block_bits,
                units: blocks as u32,
                ..Push::default()
            },
        )])?;
        let mut totals = vec![C64::new(0.0, 0.0); blocks];
        self.output.read(bytes_of_mut(&mut totals))?;
        let cumulative: Vec<f64> = totals
            .iter()
            .scan(0.0, |acc, t| {
                *acc += t.re;
                Some(*acc)
            })
            .collect();
        let total = cumulative[blocks - 1];

        // Pick a block and a target weight within it for each shot.
        let picks: Vec<(usize, f64)> = (0..shots)
            .map(|_| {
                let r = rng.random::<f64>() * total;
                let b = cumulative.partition_point(|&c| c <= r).min(blocks - 1);
                let before = if b == 0 { 0.0 } else { cumulative[b - 1] };
                (b, r - before)
            })
            .collect();
        let mut needed: Vec<usize> = picks.iter().map(|&(b, _)| b).collect();
        needed.sort_unstable();
        needed.dedup();
        let size = 1usize << block_bits;
        let ranges: Vec<_> = needed.iter().map(|&b| (b << block_bits, size)).collect();
        let amps = self.download_ranges(&ranges)?;
        Ok(picks
            .into_iter()
            .map(|(b, mut r)| {
                let slot = needed.binary_search(&b).expect("downloaded block");
                let block = &amps[slot * size..(slot + 1) * size];
                let mut i = 0;
                while i + 1 < size {
                    r -= block[i].norm_sqr();
                    if r < 0.0 {
                        break;
                    }
                    i += 1;
                }
                (b << block_bits) + i
            })
            .collect())
    }

    /// As [`sample`](Self::sample), tallied by basis state.
    pub fn sample_counts(
        &mut self,
        shots: usize,
        rng: &mut impl Rng,
    ) -> Result<BTreeMap<usize, usize>, Error> {
        let mut counts = BTreeMap::new();
        for j in self.sample(shots, rng)? {
            *counts.entry(j).or_insert(0) += 1;
        }
        Ok(counts)
    }

    /// Upload `params` and make sure the output buffer holds `output_bytes`,
    /// growing either buffer if needed.
    fn ensure_buffers(&mut self, params: &[f64], output_bytes: usize) -> Result<(), Error> {
        let bytes = bytes_of_f64(params);
        let mut rebind = false;
        if bytes.len() as u64 > self.params.size {
            self.params = Buffer::new(
                &self.ctx,
                (bytes.len() as u64).next_power_of_two(),
                vk::BufferUsageFlags::STORAGE_BUFFER,
                Memory::Upload,
            )?;
            rebind = true;
        }
        if output_bytes as u64 > self.output.size {
            self.output = Buffer::new(
                &self.ctx,
                (output_bytes as u64).next_power_of_two(),
                vk::BufferUsageFlags::STORAGE_BUFFER,
                Memory::Download,
            )?;
            rebind = true;
        }
        if rebind {
            self.bind_buffers();
        }
        self.params.write(bytes);
        Ok(())
    }

    /// Apply every operation in `circuit`, which must have the same width
    /// and no unbound parameters.
    pub fn run(&mut self, circuit: &Circuit) -> Result<(), Error> {
        circuit.check_unitary()?;
        if circuit.n_qubits() != self.n_qubits {
            return Err(Error::QubitCountMismatch {
                expected: self.n_qubits,
                found: circuit.n_qubits(),
            });
        }
        if circuit.n_parameters() > 0 {
            return Err(Error::InvalidArgument(format!(
                "circuit has {} unbound parameters; call bind first",
                circuit.n_parameters()
            )));
        }
        let mut ops: Vec<GpuOp> = circuit.fused().ops().iter().map(GpuOp::new).collect();
        if circuit.global_phase() != 0.0 {
            let phase = C64::cis(circuit.global_phase());
            ops.push(GpuOp::Phase {
                z: 0,
                plus: phase,
                minus: phase,
            });
        }
        let plan = Plan::new(&ops, self.n_qubits, self.ctx.tile_bits());
        if plan.dispatches.is_empty() {
            return Ok(());
        }
        self.ensure_buffers(&plan.params, 0)?;

        // Submit in batches of bounded work: drivers reset a GPU whose
        // submission runs for seconds (seen at 27 qubits on amdgpu).
        let mut start = 0;
        while start < plan.dispatches.len() {
            let mut work = 0u64;
            let mut end = start;
            while end < plan.dispatches.len() && (end == start || work < BATCH_UNITS) {
                work += plan.costs[end];
                end += 1;
            }
            self.submit_dispatches(&plan.dispatches[start..end])?;
            start = end;
        }
        Ok(())
    }

    /// Record and run `dispatches`, in the chunked variants if the state
    /// spans several buffers.
    fn submit_dispatches(&self, dispatches: &[(Kernel, Push)]) -> Result<(), Error> {
        let with_set: Vec<_> = dispatches
            .iter()
            .map(|&(kernel, push)| (self.set, kernel, push))
            .collect();
        self.submit_with_sets(&with_set)
    }

    /// As [`submit_dispatches`](Self::submit_dispatches), each dispatch with
    /// its own descriptor set (for states of this size and chunking).
    fn submit_with_sets(
        &self,
        dispatches: &[(vk::DescriptorSet, Kernel, Push)],
    ) -> Result<(), Error> {
        let ctx = &self.ctx;
        let chunked = self.chunked();
        ctx.submit(|cmd| {
            let d = &ctx.device;
            // SAFETY: recording dispatches against live descriptor sets.
            unsafe {
                let mut bound_set = None;
                let mut bound = None;
                for &(set, kernel, push) in dispatches {
                    if bound_set != Some(set) {
                        d.cmd_bind_descriptor_sets(
                            cmd,
                            vk::PipelineBindPoint::COMPUTE,
                            ctx.pipeline_layout,
                            0,
                            &[set],
                            &[],
                        );
                        bound_set = Some(set);
                    }
                    if bound != Some(kernel) {
                        d.cmd_bind_pipeline(
                            cmd,
                            vk::PipelineBindPoint::COMPUTE,
                            ctx.pipeline(kernel, chunked),
                        );
                        bound = Some(kernel);
                    }
                    let push = Push {
                        chunk_bits: self.chunk_bits,
                        ..push
                    };
                    ctx.push_constants(cmd, &push);
                    d.cmd_dispatch(cmd, kernel.groups(push.units), 1, 1);
                    ctx.barrier(
                        cmd,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                    );
                }
            }
        })
    }
}

impl Drop for GpuStatevector {
    fn drop(&mut self) {
        let _guard = self.ctx.lock.lock().expect("GPU lock poisoned");
        // SAFETY: the pool allows freeing; the set isn't in use.
        unsafe {
            let _ = self
                .ctx
                .device
                .free_descriptor_sets(self.ctx.descriptor_pool, &[self.set]);
        }
    }
}

/// One operation in the form the GPU kernels take.
#[derive(Clone, Debug)]
enum GpuOp {
    /// A 2×2 matrix on `target` where all of `control` is set.
    Pair {
        target: u32,
        control: u32,
        m: [C64; 4],
    },
    /// A 4×4 matrix; index bit 0 is `q0`, bit 1 is `q1`.
    Quad { q0: u32, q1: u32, m: [C64; 16] },
    /// A Pauli rotation with a non-empty X-mask.
    Pauli {
        x: u32,
        z: u32,
        cos: f64,
        factor: C64,
    },
    /// Multiply by `plus` or `minus` by the parity of the index ∧ `z`.
    Phase { z: u32, plus: C64, minus: C64 },
}

impl GpuOp {
    fn new(op: &Operation) -> GpuOp {
        match op {
            Operation::Gate { gate, target } => GpuOp::pair(&gate.matrix(), *target as u32, 0),
            Operation::Controlled {
                gate,
                control,
                target,
            } => GpuOp::pair(&gate.matrix(), *target as u32, 1 << control),
            Operation::PauliRotation { pauli, theta } => {
                let masks = pauli.masks();
                let (sin, cos) = (theta / 2.0).sin_cos();
                if masks.x == 0 {
                    GpuOp::Phase {
                        z: masks.z as u32,
                        plus: C64::cis(-theta / 2.0),
                        minus: C64::cis(theta / 2.0),
                    }
                } else {
                    GpuOp::Pauli {
                        x: masks.x as u32,
                        z: masks.z as u32,
                        cos,
                        factor: C64::new(0.0, -sin) * masks.y_phase(),
                    }
                }
            }
            Operation::Swap(..) | Operation::Unitary2 { .. } => {
                let (qubits, m) = local_matrix(op).expect("two-qubit operation");
                let flat: Vec<C64> = m.into_iter().flatten().collect();
                GpuOp::Quad {
                    q0: qubits[0] as u32,
                    q1: qubits[1] as u32,
                    m: flat.try_into().expect("4×4"),
                }
            }
        }
    }

    fn pair(m: &[[C64; 2]; 2], target: u32, control: u32) -> GpuOp {
        GpuOp::Pair {
            target,
            control,
            m: [m[0][0], m[0][1], m[1][0], m[1][1]],
        }
    }

    /// The same operation with every qubit `q` moved to `map(q)`.
    fn remap(&self, map: impl Fn(u32) -> u32) -> GpuOp {
        let mask = |m: u32| {
            (0..32)
                .filter(|q| m >> q & 1 == 1)
                .fold(0, |acc, q| acc | 1 << map(q))
        };
        match self.clone() {
            GpuOp::Pair { target, control, m } => GpuOp::Pair {
                target: map(target),
                control: mask(control),
                m,
            },
            GpuOp::Quad { q0, q1, m } => GpuOp::Quad {
                q0: map(q0),
                q1: map(q1),
                m,
            },
            GpuOp::Pauli { x, z, cos, factor } => GpuOp::Pauli {
                x: mask(x),
                z: mask(z),
                cos,
                factor,
            },
            GpuOp::Phase { z, plus, minus } => GpuOp::Phase {
                z: mask(z),
                plus,
                minus,
            },
        }
    }

    /// Every bit position the operation reads or depends on.
    fn bits(&self) -> u32 {
        match *self {
            GpuOp::Pair {
                target, control, ..
            } => control | 1 << target,
            GpuOp::Quad { q0, q1, .. } => 1 << q0 | 1 << q1,
            GpuOp::Pauli { x, z, .. } => x | z,
            GpuOp::Phase { z, .. } => z,
        }
    }

    /// Append to `params` as a standalone kernel; returns the dispatch.
    fn standalone(&self, params: &mut Vec<f64>, amps: u32) -> (Kernel, Push) {
        let offset = params.len() as u32;
        let push = |kernel, a, b, c, units| {
            (
                kernel,
                Push {
                    a,
                    b,
                    c,
                    units,
                    offset,
                    chunk_bits: 0,
                },
            )
        };
        match self {
            GpuOp::Pair { target, control, m } => {
                push_complex(params, m);
                push(Kernel::Pair, *target, *control, 0, amps / 2)
            }
            GpuOp::Quad { q0, q1, m } => {
                push_complex(params, m);
                push(Kernel::Quad, *q0, *q1, (*q0).min(*q1), amps / 4)
            }
            GpuOp::Pauli { x, z, cos, factor } => {
                params.extend([*cos, factor.re, factor.im]);
                push(Kernel::Pauli, *x, *z, highest_bit(*x), amps / 2)
            }
            GpuOp::Phase { z, plus, minus } => {
                push_complex(params, &[*plus, *minus]);
                push(Kernel::Phase, *z, 0, 0, amps)
            }
        }
    }

    /// Append to a tile kernel's gate list (see `shaders/tile.comp`).
    fn tile_entry(&self, params: &mut Vec<f64>) {
        match self {
            GpuOp::Pair { target, control, m } => {
                params.extend([0.0, *target as f64, *control as f64]);
                push_complex(params, m);
            }
            GpuOp::Quad { q0, q1, m } => {
                params.extend([1.0, *q0 as f64, *q1 as f64, (*q0).min(*q1) as f64]);
                push_complex(params, m);
            }
            GpuOp::Pauli { x, z, cos, factor } => {
                params.extend([2.0, *x as f64, *z as f64, highest_bit(*x) as f64]);
                params.extend([*cos, factor.re, factor.im]);
            }
            GpuOp::Phase { z, plus, minus } => {
                params.extend([3.0, *z as f64]);
                push_complex(params, &[*plus, *minus]);
            }
        }
    }
}

fn push_complex(params: &mut Vec<f64>, values: &[C64]) {
    for v in values {
        params.extend([v.re, v.im]);
    }
}

fn highest_bit(x: u32) -> u32 {
    u32::BITS - 1 - x.leading_zeros()
}

/// The dispatches and parameters for one circuit.
struct Plan {
    params: Vec<f64>,
    dispatches: Vec<(Kernel, Push)>,
    /// Rough work per dispatch, for splitting into submissions.
    costs: Vec<u64>,
}

impl Plan {
    /// Runs of operations that together touch at most `tile_bits` qubits
    /// become one tile dispatch, applied in shared memory; anything else is
    /// its own dispatch.
    fn new(ops: &[GpuOp], n_qubits: usize, tile_bits: u32) -> Self {
        let amps = 1u32 << n_qubits;
        let mut plan = Plan {
            params: Vec::new(),
            dispatches: Vec::new(),
            costs: Vec::new(),
        };
        let tiling = n_qubits as u32 >= tile_bits;
        let mut rest = ops;
        while let Some(first) = rest.first() {
            // Grow the run while its qubits still fit in one tile.
            let mut inside = 0u32;
            let mut run = 0;
            for op in rest {
                let union = inside | op.bits();
                if !tiling || union.count_ones() > tile_bits {
                    break;
                }
                inside = union;
                run += 1;
            }
            if run < 2 {
                let dispatch = first.standalone(&mut plan.params, amps);
                plan.costs.push(amps as u64);
                plan.dispatches.push(dispatch);
                rest = &rest[1..];
                continue;
            }
            // Fill the tile with the lowest free qubits, which keeps memory
            // accesses largely contiguous.
            for q in 0..n_qubits as u32 {
                if inside.count_ones() == tile_bits {
                    break;
                }
                inside |= 1 << q;
            }
            // A qubit's position in the tile is its rank among the tile's
            // qubits.
            let rank = |q: u32| (inside & ((1 << q) - 1)).count_ones();
            let offset = plan.params.len() as u32;
            for op in &rest[..run] {
                op.remap(rank).tile_entry(&mut plan.params);
            }
            plan.dispatches.push((
                Kernel::Tile,
                Push {
                    a: run as u32,
                    b: inside,
                    c: n_qubits as u32,
                    units: amps >> tile_bits,
                    offset,
                    chunk_bits: 0,
                },
            ));
            // One pass over memory plus the gates' shared-memory work.
            plan.costs.push(amps as u64 * (1 + run as u64 / 8));
            rest = &rest[run..];
        }
        plan
    }
}

/// A descriptor write of storage buffers to `binding`.
fn write_set(
    set: vk::DescriptorSet,
    binding: u32,
    infos: &[vk::DescriptorBufferInfo],
) -> vk::WriteDescriptorSet<'_> {
    vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(binding)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .buffer_info(infos)
}

fn bytes_of(values: &[C64]) -> &[u8] {
    // SAFETY: Complex<f64> is repr(C) as two f64s, with no padding.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

fn bytes_of_mut(values: &mut [C64]) -> &mut [u8] {
    // SAFETY: as for `bytes_of`; every bit pattern is a valid f64.
    unsafe {
        std::slice::from_raw_parts_mut(values.as_mut_ptr().cast(), std::mem::size_of_val(values))
    }
}

fn bytes_of_f64(values: &[f64]) -> &[u8] {
    // SAFETY: f64 has no padding.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}
