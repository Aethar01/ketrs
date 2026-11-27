//! Vulkan GPU compute kernels for continuous-variable Fock statevectors
//! ([`GpuFockState`]) and exact double-precision permanents ([`permanent_gpu`]).
//!
//! Uses `ketrs`'s existing [`GpuContext`] (Vulkan 1.2, `buffer_device_address`,
//! and `shader_float64`). Circuits run back-to-back in a single command-buffer
//! submission with ping-pong VRAM buffers so intermediate amplitudes never
//! leave the GPU.

use super::boson_sampling::permanent as cpu_permanent;
use super::circuit::{ModeCircuit, ModeOp};
use super::fixed_n::{build_two_mode_blocks, givens_decompose};
use super::fock_ops::{
    beam_splitter_blocks, cached_quad_eigen, cubic_phase_matrix, displacement_matrix,
    quadratic_phase_matrix, squeezing_matrix, two_mode_squeezing_blocks,
};
use super::gate::{ModeGate, TwoModeGate};
use crate::error::Error;
use crate::gpu::context::{Buffer, CvKernel, GpuContext, MAX_GROUPS, Memory, REDUCTION_GROUPS};
use ash::vk;
use num_complex::Complex64 as C64;
use std::sync::Arc;

/// Largest matrix dimension supported directly by `cv_perm.comp` (`2^27` steps).
const MAX_GPU_PERM_N: usize = 28;

/// Workgroup size declared in `cv_perm.comp`.
const PERM_WORKGROUP: u32 = 64;

/// Workgroup size declared in `cv_fock_*.comp`.
const FOCK_WORKGROUP: u32 = 256;

fn c64_as_bytes(slice: &[C64]) -> &[u8] {
    // SAFETY: `Complex<f64>` is `#[repr(C)]` two contiguous `f64`s (16 bytes).
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, std::mem::size_of_val(slice)) }
}

fn bytes_to_c64_vec(bytes: &[u8]) -> Vec<C64> {
    assert_eq!(bytes.len() % 16, 0);
    let n = bytes.len() / 16;
    let mut out = vec![C64::new(0.0, 0.0); n];
    // SAFETY: copying `16 * n` bytes into a freshly allocated `Vec<C64>`.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr() as *mut u8, bytes.len());
    }
    out
}

fn bytes_to_f64_vec(bytes: &[u8]) -> Vec<f64> {
    assert_eq!(bytes.len() % 8, 0);
    let n = bytes.len() / 8;
    let mut out = vec![0.0f64; n];
    // SAFETY: copying `8 * n` bytes into a freshly allocated `Vec<f64>`.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr() as *mut u8, bytes.len());
    }
    out
}

fn addr_uvec2(addr: u64) -> [u32; 2] {
    [addr as u32, (addr >> 32) as u32]
}

fn storage_bda_usage(extra: vk::BufferUsageFlags) -> vk::BufferUsageFlags {
    vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | extra
}

/// Exact double-precision permanent of an `n × n` row-major complex matrix on
/// the Vulkan GPU via parallel Glynn Gray-code chunking (`cv_perm.comp`).
///
/// Falls back to the CPU implementation when `n <= 1`, `n > 28`, or the GPU
/// lacks 64-bit float / buffer-device-address support.
pub fn permanent_gpu(ctx: &Arc<GpuContext>, matrix: &[C64], n: usize) -> Result<C64, Error> {
    if matrix.len() != n * n {
        return Err(Error::InvalidArgument(format!(
            "permanent_gpu needs {} entries for {n}×{n}, got {}",
            n * n,
            matrix.len()
        )));
    }
    if n <= 1 || n > MAX_GPU_PERM_N || !ctx.supports_cv() {
        return cpu_permanent(matrix, n);
    }

    // Transpose row-major input to column-major so column `j` is contiguous for
    // the Gray-code update loop in `cv_perm.comp`.
    let mut col_major = vec![C64::new(0.0, 0.0); n * n];
    for r in 0..n {
        for c in 0..n {
            col_major[c * n + r] = matrix[r * n + c];
        }
    }

    let total_steps: u32 = 1u32 << (n - 1);
    // Pick workgroup count to saturate all CUs while keeping enough Gray-code
    // steps per thread to amortize the initial O(n²) setup.
    let target_groups: u32 = if total_steps <= 4096 {
        total_steps.div_ceil(PERM_WORKGROUP).max(1)
    } else {
        (total_steps / (PERM_WORKGROUP * 64)).clamp(64, 4096)
    };
    let total_threads = target_groups * PERM_WORKGROUP;
    let steps_per_thread = total_steps.div_ceil(total_threads);
    let num_groups = total_steps.div_ceil(steps_per_thread * PERM_WORKGROUP);

    let mat_bytes = (n * n * 16) as u64;
    let out_bytes = (num_groups as u64) * 16;

    let mut mat_buf = Buffer::new(
        ctx,
        mat_bytes,
        storage_bda_usage(vk::BufferUsageFlags::empty()),
        Memory::Upload,
    )?;
    mat_buf.write(c64_as_bytes(&col_major));

    let out_buf = Buffer::new(
        ctx,
        out_bytes,
        storage_bda_usage(vk::BufferUsageFlags::empty()),
        Memory::Download,
    )?;

    #[repr(C)]
    struct PermPush {
        mat_addr: [u32; 2],
        out_addr: [u32; 2],
        n: u32,
        total_steps: u32,
        steps_per_thread: u32,
        _pad: u32,
    }
    let push = PermPush {
        mat_addr: addr_uvec2(mat_buf.device_address()),
        out_addr: addr_uvec2(out_buf.device_address()),
        n: n as u32,
        total_steps,
        steps_per_thread,
        _pad: 0,
    };
    let push_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(&push as *const PermPush as *const u8, size_of::<PermPush>())
    };

    ctx.submit(|cmd| {
        ctx.cv_dispatch(cmd, CvKernel::Permanent, push_bytes, [num_groups, 1, 1]);
    })?;

    let partials = bytes_to_c64_vec(out_buf.mapped(out_bytes as usize)?);
    let mut sum = C64::new(0.0, 0.0);
    for z in partials {
        sum += z;
    }
    let scale = 1.0 / (total_steps as f64);
    Ok(sum * scale)
}

/// Prepared GPU gate dispatch recorded into a single batched command buffer.
enum PreparedGate {
    Diag {
        offset_bytes: u64,
        stride_a: u32,
        stride_b: u32,
    },
    Dense1 {
        offset_bytes: u64,
        stride: u32,
    },
    TwoMode {
        offset_bytes: u64,
        stride_a: u32,
        stride_b: u32,
        kind: u32,
    },
}

/// Cutoff Fock statevector resident in Vulkan GPU VRAM (double precision).
pub struct GpuFockState {
    ctx: Arc<GpuContext>,
    n_modes: usize,
    cutoff: usize,
    total_len: usize,
    /// Primary statevector buffer in `Memory::Device`.
    state: Buffer,
    /// Scratch buffer of equal size in `Memory::Device` for ping-pong gates.
    scratch: Buffer,
}

impl GpuFockState {
    /// Vacuum state `|0…0⟩` on `n_modes` modes with Fock cutoff `cutoff` in GPU VRAM.
    pub fn vacuum(ctx: &Arc<GpuContext>, n_modes: usize, cutoff: usize) -> Result<Self, Error> {
        if !ctx.supports_cv() {
            return Err(Error::Gpu(
                "GPU lacks 64-bit float or buffer-device-address support for CV Fock kernels"
                    .into(),
            ));
        }
        let total_len = checked_dim(cutoff, n_modes)?;
        let mut init = vec![C64::new(0.0, 0.0); total_len];
        init[0] = C64::new(1.0, 0.0);
        Self::from_amplitudes(ctx, n_modes, cutoff, &init)
    }

    /// Upload a host slice of `cutoff^n_modes` amplitudes into GPU VRAM.
    pub fn from_amplitudes(
        ctx: &Arc<GpuContext>,
        n_modes: usize,
        cutoff: usize,
        amps: &[C64],
    ) -> Result<Self, Error> {
        if !ctx.supports_cv() {
            return Err(Error::Gpu(
                "GPU lacks 64-bit float or buffer-device-address support for CV Fock kernels"
                    .into(),
            ));
        }
        let total_len = checked_dim(cutoff, n_modes)?;
        if amps.len() != total_len {
            return Err(Error::InvalidArgument(format!(
                "expected {total_len} amplitudes for {n_modes} modes at cutoff {cutoff}, got {}",
                amps.len()
            )));
        }
        let bytes = (total_len * 16) as u64;
        let state = Buffer::new(
            ctx,
            bytes,
            storage_bda_usage(
                vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC,
            ),
            Memory::Device,
        )?;
        let scratch = Buffer::new(
            ctx,
            bytes,
            storage_bda_usage(
                vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC,
            ),
            Memory::Device,
        )?;

        let mut staging = Buffer::new(
            ctx,
            bytes,
            vk::BufferUsageFlags::TRANSFER_SRC,
            Memory::Upload,
        )?;
        staging.write(c64_as_bytes(amps));

        ctx.submit(|cmd| {
            let region = [vk::BufferCopy::default().size(bytes)];
            // SAFETY: valid buffers of size `bytes` in a recording command buffer.
            unsafe {
                ctx.device
                    .cmd_copy_buffer(cmd, staging.buffer, state.buffer, &region);
            }
        })?;

        Ok(GpuFockState {
            ctx: Arc::clone(ctx),
            n_modes,
            cutoff,
            total_len,
            state,
            scratch,
        })
    }

    /// Run a bound, measurement-free `ModeCircuit` completely on the GPU in a
    /// single batched Vulkan command-buffer submission.
    pub fn from_circuit(
        ctx: &Arc<GpuContext>,
        circuit: &ModeCircuit,
        cutoff: usize,
    ) -> Result<Self, Error> {
        if circuit.n_parameters() > 0 {
            return Err(Error::InvalidArgument(
                "circuit has unbound parameters; bind it first".into(),
            ));
        }
        if !circuit.is_unitary() {
            return Err(Error::InvalidArgument(
                "GpuFockState::from_circuit expects a measurement-free circuit".into(),
            ));
        }
        let n_modes = circuit.n_modes();
        let total_len = checked_dim(cutoff, n_modes)?;
        let init = build_product_input(circuit, cutoff, total_len)?;
        let mut st = Self::from_amplitudes(ctx, n_modes, cutoff, &init)?;

        // Prepare all gate tables into one contiguous host upload buffer so
        // every gate in the circuit executes in ONE Vulkan submission!
        let mut gate_pool: Vec<C64> = Vec::new();
        let mut prepared: Vec<PreparedGate> = Vec::new();
        for op in circuit.ops() {
            prepare_op(&mut gate_pool, &mut prepared, op, n_modes, cutoff)?;
        }
        st.run_prepared(&gate_pool, &prepared)?;
        Ok(st)
    }

    /// Number of modes.
    pub fn n_modes(&self) -> usize {
        self.n_modes
    }

    /// Local Fock cutoff `d`.
    pub fn cutoff(&self) -> usize {
        self.cutoff
    }

    /// Download all `cutoff^n_modes` amplitudes to host memory.
    pub fn amplitudes(&self) -> Result<Vec<C64>, Error> {
        let bytes = (self.total_len * 16) as u64;
        let dl = Buffer::new(
            &self.ctx,
            bytes,
            vk::BufferUsageFlags::TRANSFER_DST,
            Memory::Download,
        )?;
        self.ctx.submit(|cmd| {
            let region = [vk::BufferCopy::default().size(bytes)];
            // SAFETY: valid buffers of size `bytes`.
            unsafe {
                self.ctx
                    .device
                    .cmd_copy_buffer(cmd, self.state.buffer, dl.buffer, &region);
            }
        })?;
        Ok(bytes_to_c64_vec(dl.mapped(bytes as usize)?))
    }

    /// Download all Fock probabilities `|ψ_i|²`.
    pub fn probabilities(&self) -> Result<Vec<f64>, Error> {
        Ok(self.amplitudes()?.iter().map(|z| z.norm_sqr()).collect())
    }

    /// Squared norm `⟨ψ|ψ⟩` reduced directly on the GPU (no full state download).
    pub fn norm_sqr(&self) -> Result<f64, Error> {
        self.reduce_scalar(0)
    }

    /// Cutoff truncation error `1 − ⟨ψ|ψ⟩`.
    pub fn truncation_error(&self) -> Result<f64, Error> {
        Ok((1.0 - self.norm_sqr()?).max(0.0))
    }

    /// Mean photon number `⟨n̂⟩` on `mode`, reduced directly on the GPU.
    pub fn mean_photons(&self, mode: usize) -> Result<f64, Error> {
        if mode >= self.n_modes {
            return Err(Error::InvalidArgument(format!(
                "mode {mode} out of range for {} modes",
                self.n_modes
            )));
        }
        let stride = self.cutoff.pow(mode as u32) as u32;
        self.reduce_scalar(stride)
    }

    fn reduce_scalar(&self, stride: u32) -> Result<f64, Error> {
        let total_u32 = self.total_len as u32;
        let groups = total_u32
            .div_ceil(FOCK_WORKGROUP)
            .clamp(1, REDUCTION_GROUPS);
        let out_bytes = (groups as u64) * 8;
        let out_buf = Buffer::new(
            &self.ctx,
            out_bytes,
            storage_bda_usage(vk::BufferUsageFlags::empty()),
            Memory::Download,
        )?;
        #[repr(C)]
        struct ReducePush {
            state_addr: [u32; 2],
            out_addr: [u32; 2],
            total_len: u32,
            d: u32,
            stride: u32,
            _pad: u32,
        }
        let push = ReducePush {
            state_addr: addr_uvec2(self.state.device_address()),
            out_addr: addr_uvec2(out_buf.device_address()),
            total_len: total_u32,
            d: self.cutoff as u32,
            stride,
            _pad: 0,
        };
        let push_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                &push as *const ReducePush as *const u8,
                size_of::<ReducePush>(),
            )
        };
        self.ctx.submit(|cmd| {
            self.ctx
                .cv_dispatch(cmd, CvKernel::FockReduce, push_bytes, [groups, 1, 1]);
        })?;
        let partials = bytes_to_f64_vec(out_buf.mapped(out_bytes as usize)?);
        Ok(partials.into_iter().sum())
    }

    // -----------------------------------------------------------------------
    // Single-gate imperative API
    // -----------------------------------------------------------------------

    /// Rotation `R(φ)` on `mode`.
    pub fn rotation(&mut self, phi: f64, mode: usize) -> Result<(), Error> {
        self.apply_single_op(&ModeOp::Gate {
            gate: ModeGate::Rotation { phi: phi.into() },
            mode,
        })
    }

    /// Kerr `K(κ)` on `mode`.
    pub fn kerr(&mut self, kappa: f64, mode: usize) -> Result<(), Error> {
        self.apply_single_op(&ModeOp::Gate {
            gate: ModeGate::Kerr {
                kappa: kappa.into(),
            },
            mode,
        })
    }

    /// Displacement `D(r e^{iφ})` on `mode`.
    pub fn displace(&mut self, r: f64, phi: f64, mode: usize) -> Result<(), Error> {
        self.apply_single_op(&ModeOp::Gate {
            gate: ModeGate::Displacement {
                r: r.into(),
                phi: phi.into(),
            },
            mode,
        })
    }

    /// Squeezing `S(r e^{iφ})` on `mode`.
    pub fn squeeze(&mut self, r: f64, phi: f64, mode: usize) -> Result<(), Error> {
        self.apply_single_op(&ModeOp::Gate {
            gate: ModeGate::Squeezing {
                r: r.into(),
                phi: phi.into(),
            },
            mode,
        })
    }

    /// Quadratic phase `P(s)` on `mode`.
    pub fn quadratic_phase(&mut self, s: f64, mode: usize) -> Result<(), Error> {
        self.apply_single_op(&ModeOp::Gate {
            gate: ModeGate::QuadraticPhase { s: s.into() },
            mode,
        })
    }

    /// Cubic phase `V(γ)` on `mode`.
    pub fn cubic_phase(&mut self, gamma: f64, mode: usize) -> Result<(), Error> {
        self.apply_single_op(&ModeOp::Gate {
            gate: ModeGate::CubicPhase {
                gamma: gamma.into(),
            },
            mode,
        })
    }

    /// Beam splitter `BS(θ, φ)` on `(a, b)`.
    pub fn beam_splitter(&mut self, theta: f64, phi: f64, a: usize, b: usize) -> Result<(), Error> {
        self.apply_single_op(&ModeOp::TwoModeGate {
            gate: TwoModeGate::BeamSplitter {
                theta: theta.into(),
                phi: phi.into(),
            },
            modes: [a, b],
        })
    }

    /// Two-mode squeezing `S₂(r e^{iφ})` on `(a, b)`.
    pub fn two_mode_squeeze(&mut self, r: f64, phi: f64, a: usize, b: usize) -> Result<(), Error> {
        self.apply_single_op(&ModeOp::TwoModeGate {
            gate: TwoModeGate::TwoModeSqueezing {
                r: r.into(),
                phi: phi.into(),
            },
            modes: [a, b],
        })
    }

    /// `CX(s)` on `(a, b)`.
    pub fn cx(&mut self, s: f64, a: usize, b: usize) -> Result<(), Error> {
        self.apply_single_op(&ModeOp::TwoModeGate {
            gate: TwoModeGate::ControlledX { s: s.into() },
            modes: [a, b],
        })
    }

    /// `CZ(s)` on `(a, b)`.
    pub fn cz(&mut self, s: f64, a: usize, b: usize) -> Result<(), Error> {
        self.apply_single_op(&ModeOp::TwoModeGate {
            gate: TwoModeGate::ControlledZ { s: s.into() },
            modes: [a, b],
        })
    }

    /// Cross-Kerr `CK(κ)` on `(a, b)`.
    pub fn cross_kerr(&mut self, kappa: f64, a: usize, b: usize) -> Result<(), Error> {
        self.apply_single_op(&ModeOp::TwoModeGate {
            gate: TwoModeGate::CrossKerr {
                kappa: kappa.into(),
            },
            modes: [a, b],
        })
    }

    /// Passive linear-optics unitary `U` on `modes`.
    pub fn apply_passive(&mut self, modes: &[usize], matrix: &[C64]) -> Result<(), Error> {
        self.apply_single_op(&ModeOp::Passive {
            modes: modes.to_vec(),
            matrix: matrix.to_vec(),
        })
    }

    fn apply_single_op(&mut self, op: &ModeOp) -> Result<(), Error> {
        let mut pool = Vec::new();
        let mut prepared = Vec::new();
        prepare_op(&mut pool, &mut prepared, op, self.n_modes, self.cutoff)?;
        self.run_prepared(&pool, &prepared)
    }

    fn run_prepared(&mut self, pool: &[C64], ops: &[PreparedGate]) -> Result<(), Error> {
        if ops.is_empty() {
            return Ok(());
        }
        let pool_bytes = (pool.len() * 16).max(16) as u64;
        let mut gate_buf = Buffer::new(
            &self.ctx,
            pool_bytes,
            storage_bda_usage(vk::BufferUsageFlags::empty()),
            Memory::Upload,
        )?;
        if !pool.is_empty() {
            gate_buf.write(c64_as_bytes(pool));
        }
        let gate_base = gate_buf.device_address();
        let mut cur_addr = self.state.device_address();
        let mut alt_addr = self.scratch.device_address();
        let mut swapped = false;

        let total_u32 = self.total_len as u32;
        let groups = total_u32.div_ceil(FOCK_WORKGROUP).clamp(1, MAX_GROUPS);
        let d_u32 = self.cutoff as u32;

        self.ctx.submit(|cmd| {
            for (idx, g) in ops.iter().enumerate() {
                if idx > 0 {
                    self.ctx.barrier(
                        cmd,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                    );
                }
                match *g {
                    PreparedGate::Diag {
                        offset_bytes,
                        stride_a,
                        stride_b,
                    } => {
                        #[repr(C)]
                        struct DiagPush {
                            state_addr: [u32; 2],
                            phase_addr: [u32; 2],
                            total_len: u32,
                            d: u32,
                            stride_a: u32,
                            stride_b: u32,
                        }
                        let push = DiagPush {
                            state_addr: addr_uvec2(cur_addr),
                            phase_addr: addr_uvec2(gate_base + offset_bytes),
                            total_len: total_u32,
                            d: d_u32,
                            stride_a,
                            stride_b,
                        };
                        let bytes: &[u8] = unsafe {
                            std::slice::from_raw_parts(
                                &push as *const DiagPush as *const u8,
                                size_of::<DiagPush>(),
                            )
                        };
                        self.ctx
                            .cv_dispatch(cmd, CvKernel::FockDiag, bytes, [groups, 1, 1]);
                    }
                    PreparedGate::Dense1 {
                        offset_bytes,
                        stride,
                    } => {
                        #[repr(C)]
                        struct Push1 {
                            in_addr: [u32; 2],
                            out_addr: [u32; 2],
                            gate_addr: [u32; 2],
                            total_len: u32,
                            d: u32,
                            stride: u32,
                            _pad: u32,
                        }
                        let push = Push1 {
                            in_addr: addr_uvec2(cur_addr),
                            out_addr: addr_uvec2(alt_addr),
                            gate_addr: addr_uvec2(gate_base + offset_bytes),
                            total_len: total_u32,
                            d: d_u32,
                            stride,
                            _pad: 0,
                        };
                        let bytes: &[u8] = unsafe {
                            std::slice::from_raw_parts(
                                &push as *const Push1 as *const u8,
                                size_of::<Push1>(),
                            )
                        };
                        self.ctx
                            .cv_dispatch(cmd, CvKernel::Fock1Mode, bytes, [groups, 1, 1]);
                        std::mem::swap(&mut cur_addr, &mut alt_addr);
                        swapped = !swapped;
                    }
                    PreparedGate::TwoMode {
                        offset_bytes,
                        stride_a,
                        stride_b,
                        kind,
                    } => {
                        #[repr(C)]
                        struct Push2 {
                            in_addr: [u32; 2],
                            out_addr: [u32; 2],
                            gate_addr: [u32; 2],
                            total_len: u32,
                            d: u32,
                            stride_a: u32,
                            stride_b: u32,
                            kind: u32,
                            _pad: u32,
                        }
                        let push = Push2 {
                            in_addr: addr_uvec2(cur_addr),
                            out_addr: addr_uvec2(alt_addr),
                            gate_addr: addr_uvec2(gate_base + offset_bytes),
                            total_len: total_u32,
                            d: d_u32,
                            stride_a,
                            stride_b,
                            kind,
                            _pad: 0,
                        };
                        let bytes: &[u8] = unsafe {
                            std::slice::from_raw_parts(
                                &push as *const Push2 as *const u8,
                                size_of::<Push2>(),
                            )
                        };
                        self.ctx
                            .cv_dispatch(cmd, CvKernel::Fock2Mode, bytes, [groups, 1, 1]);
                        std::mem::swap(&mut cur_addr, &mut alt_addr);
                        swapped = !swapped;
                    }
                }
            }
        })?;

        if swapped {
            std::mem::swap(&mut self.state, &mut self.scratch);
        }
        Ok(())
    }
}

fn checked_dim(cutoff: usize, n_modes: usize) -> Result<usize, Error> {
    if cutoff == 0 {
        return Err(Error::InvalidArgument("Fock cutoff must be >= 1".into()));
    }
    let dim = cutoff
        .checked_pow(n_modes as u32)
        .ok_or_else(|| Error::InvalidArgument(format!("cutoff^{n_modes} overflows usize")))?;
    if dim > u32::MAX as usize {
        return Err(Error::InvalidArgument(format!(
            "GpuFockState dimension {dim} exceeds 32-bit shader index range"
        )));
    }
    Ok(dim)
}

fn build_product_input(
    circuit: &ModeCircuit,
    cutoff: usize,
    total_len: usize,
) -> Result<Vec<C64>, Error> {
    let mut amps = vec![C64::new(0.0, 0.0); total_len];
    let mut cur_len = 1usize;
    amps[0] = C64::new(1.0, 0.0);
    for (mode, st) in circuit.inputs().iter().enumerate() {
        let local = st.fock_amplitudes(cutoff).ok_or_else(|| {
            Error::InvalidArgument(format!(
                "mode {mode} input {st:?} is mixed; GpuFockState supports pure inputs"
            ))
        })?;
        // Expand tensor product with mode `mode` as the next-higher digit (LSB = mode 0).
        for n in (0..cutoff).rev() {
            let factor = local[n];
            let dst_off = n * cur_len;
            for i in 0..cur_len {
                amps[dst_off + i] = amps[i] * factor;
            }
        }
        cur_len *= cutoff;
    }
    Ok(amps)
}

fn push_pool(pool: &mut Vec<C64>, data: &[C64]) -> u64 {
    let off = (pool.len() * 16) as u64;
    pool.extend_from_slice(data);
    off
}

fn prepare_op(
    pool: &mut Vec<C64>,
    out: &mut Vec<PreparedGate>,
    op: &ModeOp,
    n_modes: usize,
    cutoff: usize,
) -> Result<(), Error> {
    let stride = |m: usize| -> Result<u32, Error> {
        if m >= n_modes {
            return Err(Error::InvalidArgument(format!(
                "mode {m} out of range for {n_modes} modes"
            )));
        }
        Ok(cutoff.pow(m as u32) as u32)
    };

    match op {
        ModeOp::Gate { gate, mode } => {
            let s = stride(*mode)?;
            match *gate {
                ModeGate::Rotation { phi } => {
                    let angle = phi.evaluate(&[])?;
                    if angle != 0.0 {
                        let phases: Vec<C64> = (0..cutoff)
                            .map(|n| C64::from_polar(1.0, angle * (n as f64)))
                            .collect();
                        let off = push_pool(pool, &phases);
                        out.push(PreparedGate::Diag {
                            offset_bytes: off,
                            stride_a: s,
                            stride_b: 0,
                        });
                    }
                }
                ModeGate::Kerr { kappa } => {
                    let k = kappa.evaluate(&[])?;
                    if k != 0.0 {
                        let phases: Vec<C64> = (0..cutoff)
                            .map(|n| {
                                let nf = n as f64;
                                C64::from_polar(1.0, k * nf * nf)
                            })
                            .collect();
                        let off = push_pool(pool, &phases);
                        out.push(PreparedGate::Diag {
                            offset_bytes: off,
                            stride_a: s,
                            stride_b: 0,
                        });
                    }
                }
                ModeGate::Displacement { r, phi } => {
                    let rv = r.evaluate(&[])?;
                    let pv = phi.evaluate(&[])?;
                    if rv != 0.0 {
                        let mat = displacement_matrix(C64::from_polar(rv, pv), cutoff);
                        let off = push_pool(pool, &mat);
                        out.push(PreparedGate::Dense1 {
                            offset_bytes: off,
                            stride: s,
                        });
                    }
                }
                ModeGate::Squeezing { r, phi } => {
                    let rv = r.evaluate(&[])?;
                    let pv = phi.evaluate(&[])?;
                    if rv != 0.0 {
                        let mat = squeezing_matrix(rv, pv, cutoff);
                        let off = push_pool(pool, &mat);
                        out.push(PreparedGate::Dense1 {
                            offset_bytes: off,
                            stride: s,
                        });
                    }
                }
                ModeGate::QuadraticPhase { s: sq } => {
                    let sv = sq.evaluate(&[])?;
                    if sv != 0.0 {
                        let mat = quadratic_phase_matrix(sv, cutoff);
                        let off = push_pool(pool, &mat);
                        out.push(PreparedGate::Dense1 {
                            offset_bytes: off,
                            stride: s,
                        });
                    }
                }
                ModeGate::CubicPhase { gamma } => {
                    let gv = gamma.evaluate(&[])?;
                    if gv != 0.0 {
                        let mat = cubic_phase_matrix(gv, cutoff);
                        let off = push_pool(pool, &mat);
                        out.push(PreparedGate::Dense1 {
                            offset_bytes: off,
                            stride: s,
                        });
                    }
                }
            }
        }
        ModeOp::TwoModeGate { gate, modes } => {
            let [a, b] = *modes;
            if a == b {
                return Err(Error::InvalidArgument(format!(
                    "mode {a} used more than once"
                )));
            }
            let sa = stride(a)?;
            let sb = stride(b)?;
            match *gate {
                TwoModeGate::BeamSplitter { theta, phi } => {
                    let blocks =
                        beam_splitter_blocks(theta.evaluate(&[])?, phi.evaluate(&[])?, cutoff);
                    let off = (pool.len() * 16) as u64;
                    for blk in blocks {
                        pool.extend_from_slice(&blk);
                    }
                    out.push(PreparedGate::TwoMode {
                        offset_bytes: off,
                        stride_a: sa,
                        stride_b: sb,
                        kind: 0,
                    });
                }
                TwoModeGate::TwoModeSqueezing { r, phi } => {
                    let rv = r.evaluate(&[])?;
                    let pv = phi.evaluate(&[])?;
                    if rv != 0.0 {
                        let blocks = two_mode_squeezing_blocks(rv, pv, cutoff);
                        let off = (pool.len() * 16) as u64;
                        for blk in blocks {
                            pool.extend_from_slice(&blk);
                        }
                        out.push(PreparedGate::TwoMode {
                            offset_bytes: off,
                            stride_a: sa,
                            stride_b: sb,
                            kind: 1,
                        });
                    }
                }
                TwoModeGate::CrossKerr { kappa } => {
                    let k = kappa.evaluate(&[])?;
                    if k != 0.0 {
                        let mut phases = Vec::with_capacity(cutoff * cutoff);
                        for na in 0..cutoff {
                            for nb in 0..cutoff {
                                phases.push(C64::from_polar(1.0, k * ((na * nb) as f64)));
                            }
                        }
                        let off = push_pool(pool, &phases);
                        out.push(PreparedGate::Diag {
                            offset_bytes: off,
                            stride_a: sa,
                            stride_b: sb,
                        });
                    }
                }
                TwoModeGate::ControlledX { s } => {
                    let sv = s.evaluate(&[])?;
                    if sv != 0.0 {
                        let mat = build_quad_product_matrix(cutoff, sv, true);
                        let off = push_pool(pool, &mat);
                        out.push(PreparedGate::TwoMode {
                            offset_bytes: off,
                            stride_a: sa,
                            stride_b: sb,
                            kind: 2,
                        });
                    }
                }
                TwoModeGate::ControlledZ { s } => {
                    let sv = s.evaluate(&[])?;
                    if sv != 0.0 {
                        let mat = build_quad_product_matrix(cutoff, sv, false);
                        let off = push_pool(pool, &mat);
                        out.push(PreparedGate::TwoMode {
                            offset_bytes: off,
                            stride_a: sa,
                            stride_b: sb,
                            kind: 2,
                        });
                    }
                }
            }
        }
        ModeOp::Passive { modes, matrix } => {
            let k = modes.len();
            if matrix.len() != k * k {
                return Err(Error::InvalidArgument(format!(
                    "passive block on {k} modes needs {} entries, got {}",
                    k * k,
                    matrix.len()
                )));
            }
            for (i, &m) in modes.iter().enumerate() {
                let _ = stride(m)?;
                if modes[..i].contains(&m) {
                    return Err(Error::InvalidArgument(format!(
                        "mode {m} used more than once"
                    )));
                }
            }
            if k == 1 {
                prepare_op(
                    pool,
                    out,
                    &ModeOp::Gate {
                        gate: ModeGate::Rotation {
                            phi: matrix[0].arg().into(),
                        },
                        mode: modes[0],
                    },
                    n_modes,
                    cutoff,
                )?;
            } else if k >= 2 {
                let (ops, diag_phases) = givens_decompose(matrix, k);
                for (i, &phi) in diag_phases.iter().enumerate() {
                    if phi.abs() > 1e-15 {
                        prepare_op(
                            pool,
                            out,
                            &ModeOp::Gate {
                                gate: ModeGate::Rotation { phi: phi.into() },
                                mode: modes[i],
                            },
                            n_modes,
                            cutoff,
                        )?;
                    }
                }
                for (i, j, u2) in ops.into_iter().rev() {
                    let blocks = build_two_mode_blocks(&u2, 2 * (cutoff - 1));
                    let off = (pool.len() * 16) as u64;
                    for blk in blocks {
                        pool.extend_from_slice(&blk);
                    }
                    out.push(PreparedGate::TwoMode {
                        offset_bytes: off,
                        stride_a: stride(modes[i])?,
                        stride_b: stride(modes[j])?,
                        kind: 0,
                    });
                }
            }
        }
        ModeOp::Channel { .. } => {
            return Err(Error::InvalidArgument(
                "GpuFockState does not support noise channels".into(),
            ));
        }
        ModeOp::Measure { .. } => {
            return Err(Error::InvalidArgument(
                "GpuFockState does not support mid-circuit measurements".into(),
            ));
        }
    }
    Ok(())
}

/// Dense `d² × d²` projected matrix for `CX(s)` (`is_cx = true`) or `CZ(s)`
/// (`is_cx = false`) computed via the cached padded quadrature eigenbasis.
fn build_quad_product_matrix(d: usize, s: f64, is_cx: bool) -> Vec<C64> {
    let qe = cached_quad_eigen(d);
    let big = qe.big;
    let sign = if is_cx { -0.5 * s } else { 0.5 * s };
    let dd = d * d;
    let mut out = vec![C64::new(0.0, 0.0); dd * dd];

    // Factor: U[(ma, mb), (na, nb)] = Σ_ka vx[ma, ka] vx[na, ka] * B_ka[mb, nb]
    // where B_ka[mb, nb] = Σ_kb V_b[mb, kb] exp(i * sign * λ_ka * λ_kb) V_b[nb, kb]*.
    let mut b_mat = vec![C64::new(0.0, 0.0); d * d];
    for ka in 0..big {
        let la = qe.evals[ka] * sign;
        if is_cx {
            for mb in 0..d {
                for nb in 0..d {
                    let mut acc = C64::new(0.0, 0.0);
                    for kb in 0..big {
                        let ph = C64::from_polar(1.0, la * qe.evals[kb]);
                        acc += qe.vp[mb * big + kb] * ph * qe.vp[nb * big + kb].conj();
                    }
                    b_mat[mb * d + nb] = acc;
                }
            }
        } else {
            for mb in 0..d {
                for nb in mb..d {
                    let mut acc = C64::new(0.0, 0.0);
                    for kb in 0..big {
                        let ph = C64::from_polar(1.0, la * qe.evals[kb]);
                        acc += ph * (qe.vx[mb * big + kb] * qe.vx[nb * big + kb]);
                    }
                    b_mat[mb * d + nb] = acc;
                    b_mat[nb * d + mb] = acc;
                }
            }
        }
        for ma in 0..d {
            let vma = qe.vx[ma * big + ka];
            for na in 0..d {
                let weight = vma * qe.vx[na * big + ka];
                for mb in 0..d {
                    let row = (ma * d + mb) * dd + na * d;
                    let brow = mb * d;
                    for nb in 0..d {
                        out[row + nb] += b_mat[brow + nb] * weight;
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cv::{FockState, ModeState};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    #[test]
    fn permanent_gpu_matches_cpu_exact() {
        let Ok(ctx) = GpuContext::new() else { return };
        if !ctx.supports_cv() {
            return;
        }
        let mut rng = StdRng::seed_from_u64(12345);
        for n in [4, 8, 12, 16] {
            let mat: Vec<C64> = (0..n * n)
                .map(|_| C64::new(rng.random_range(-0.6..0.6), rng.random_range(-0.6..0.6)))
                .collect();
            let cpu = cpu_permanent(&mat, n).unwrap();
            let gpu = permanent_gpu(&ctx, &mat, n).unwrap();
            let diff = (cpu - gpu).norm();
            let scale = cpu.norm().max(1.0);
            assert!(
                diff / scale < 1e-11,
                "n={n}: cpu={cpu} vs gpu={gpu} (rel diff {})",
                diff / scale
            );
        }
    }

    #[test]
    fn gpu_fock_state_matches_cpu_fock_state() {
        let Ok(ctx) = GpuContext::new() else { return };
        if !ctx.supports_cv() {
            return;
        }
        let d = 10;
        let mut c = ModeCircuit::new(3);
        c.set_input(0, ModeState::Fock(1)).unwrap();
        c.set_input(1, ModeState::Coherent { r: 0.4, phi: 0.2 })
            .unwrap();
        c.squeeze(0.25, 0.3, 0)
            .displace(0.2, -0.4, 2)
            .rotation(0.5, 1)
            .kerr(0.35, 0)
            .beam_splitter(0.6, 0.2, 0, 1)
            .two_mode_squeeze(0.2, -0.3, 1, 2)
            .cross_kerr(-0.25, 0, 2)
            .cx(0.3, 2, 0)
            .cz(-0.2, 0, 1)
            .cubic_phase(0.08, 1);

        let cpu = FockState::from_circuit(&c, d).unwrap();
        let gpu = GpuFockState::from_circuit(&ctx, &c, d).unwrap();

        assert!((gpu.norm_sqr().unwrap() - cpu.norm_sqr()).abs() < 1e-11);
        for m in 0..3 {
            assert!((gpu.mean_photons(m).unwrap() - cpu.mean_photons(m).unwrap()).abs() < 1e-11);
        }
        let g_amps = gpu.amplitudes().unwrap();
        for (i, (&ga, &ca)) in g_amps.iter().zip(cpu.amplitudes()).enumerate() {
            assert!(
                (ga - ca).norm() < 1e-11,
                "amp[{i}] mismatch: gpu={ga} vs cpu={ca}"
            );
        }
    }
}
