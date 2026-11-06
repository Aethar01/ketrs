//! `H|ψ⟩` and adjoint gradients on the GPU.
//!
//! As on the CPU (see [`crate::statevector`]'s gradients), one backward
//! sweep uncomputes the state ψ and the cotangent λ together, taking
//! `Im⟨λ|G|ψ⟩` at each parametric gate. Here both states stay on the GPU:
//! a pair of descriptor sets binds each as the other's partner, so the
//! inverse gates (planned and tiled as for [`GpuStatevector::run`]) and the
//! overlaps all go into a few submissions, and every overlap's partial sums
//! come back in a single download at the end.

use super::context::{GpuContext, Kernel, MAX_CHUNKS, Push};
use super::{BATCH_UNITS, GpuOp, GpuStatevector, Plan, write_set};
use crate::circuit::Circuit;
use crate::error::Error;
use crate::observable::PauliSum;
use crate::statevector::gradient::Generator;
use crate::statevector::hamiltonian::{GroupedPauliSum, Imaginary};
use ash::vk;
use num_complex::Complex64 as C64;
use std::sync::Arc;

impl GpuStatevector {
    /// A copy of this state on the same GPU.
    pub fn try_clone(&self) -> Result<Self, Error> {
        let copy = Self::allocate(&self.ctx, self.n_qubits, self.chunk_bits, self.precision)?;
        let size = self.amplitude_bytes() << self.chunk_bits;
        self.ctx.submit(|cmd| {
            for (from, to) in self.chunks.iter().zip(&copy.chunks) {
                let region = [vk::BufferCopy::default().size(size)];
                // SAFETY: both chunks hold `size` bytes.
                unsafe {
                    self.ctx
                        .device
                        .cmd_copy_buffer(cmd, from.buffer, to.buffer, &region)
                };
            }
        })?;
        Ok(copy)
    }

    /// `H|ψ⟩` as a new (generally unnormalised) state, computed on the GPU
    /// in one pass.
    pub fn apply_pauli_sum(&mut self, hamiltonian: &PauliSum) -> Result<Self, Error> {
        let out = Self::allocate(&self.ctx, self.n_qubits, self.chunk_bits, self.precision)?;
        self.apply_pauli_sum_into(hamiltonian, &out)?;
        Ok(out)
    }

    /// `out = H|ψ⟩`.
    fn apply_pauli_sum_into(&mut self, hamiltonian: &PauliSum, out: &Self) -> Result<(), Error> {
        let (params, n_groups) = hamil_params(hamiltonian, self.n_qubits)?;
        self.ensure_buffers(&params, 0)?;
        let mut sets = Sets::new(&self.ctx);
        let set = sets.pair(self, out, self)?;
        self.submit_with_sets(&[(set, Kernel::Hamil, hamil_push(n_groups, self.len()))])
    }

    /// `⟨ψ(θ)|H|ψ(θ)⟩` and its gradient by adjoint differentiation, where
    /// `ψ(θ)` is `circuit` bound to `parameters` applied to this state, all
    /// on the GPU. Costs about three circuit runs and holds two more states.
    pub fn expectation_gradient(
        &self,
        circuit: &Circuit,
        parameters: &[f64],
        observable: &PauliSum,
    ) -> Result<(f64, Vec<f64>), Error> {
        let mut psi = self.try_clone()?;
        psi.run(&circuit.bind(parameters)?)?;
        let lambda = psi.apply_pauli_sum(observable)?;
        sweep(circuit, parameters, psi, lambda)
    }

    /// Gradient of a real loss `L` of the output of `circuit`, by adjoint
    /// differentiation on the GPU: `final_state` is `circuit` bound to
    /// `parameters` applied to some initial state, and `cotangent` is
    /// `λ = ∂L/∂ψ*` there. See
    /// [`DenseState::adjoint_gradient`](crate::DenseState::adjoint_gradient).
    pub fn adjoint_gradient(
        circuit: &Circuit,
        parameters: &[f64],
        final_state: &Self,
        cotangent: &Self,
    ) -> Result<Vec<f64>, Error> {
        for state in [final_state, cotangent] {
            if state.n_qubits != circuit.n_qubits() {
                return Err(Error::QubitCountMismatch {
                    expected: circuit.n_qubits(),
                    found: state.n_qubits,
                });
            }
        }
        if !Arc::ptr_eq(&final_state.ctx, &cotangent.ctx)
            || final_state.chunk_bits != cotangent.chunk_bits
            || final_state.precision != cotangent.precision
        {
            return Err(Error::InvalidArgument(
                "states must share a GPU, buffer layout and precision".into(),
            ));
        }
        let (psi, lambda) = (final_state.try_clone()?, cotangent.try_clone()?);
        Ok(sweep(circuit, parameters, psi, lambda)?.1)
    }
}

/// `hamil.comp`'s parameters for `H` (see the shader) and its number of
/// groups: the diagonal terms, then each X-mask group.
pub(super) fn hamil_params(
    hamiltonian: &PauliSum,
    n_qubits: usize,
) -> Result<(Vec<f64>, u32), Error> {
    let grouped = GroupedPauliSum::<C64>::new(hamiltonian, n_qubits, Imaginary::Reject)?;
    let mut params = Vec::new();
    let mut n_groups = 0;
    if !grouped.diagonal.is_empty() {
        params.extend([0.0, grouped.diagonal.len() as f64]);
        for &(z, c) in &grouped.diagonal {
            params.extend([z as f64, c, 0.0]);
        }
        n_groups += 1;
    }
    for g in &grouped.groups {
        params.extend([g.x as f64, g.terms.len() as f64]);
        for &(z, c) in &g.terms {
            params.extend([z as f64, c.re, c.im]);
        }
        n_groups += 1;
    }
    if n_groups == 0 {
        // H = 0: one empty group writes zeros.
        params.extend([0.0, 0.0]);
        n_groups = 1;
    }
    Ok((params, n_groups))
}

/// The push constants for `hamil.comp` with parameters at offset 0.
pub(super) fn hamil_push(n_groups: u32, amps: usize) -> Push {
    Push {
        b: n_groups,
        units: amps as u32,
        ..Push::default()
    }
}

/// Which state a sweep dispatch runs on: ψ or λ at binding 0. Overlaps
/// run on ψ's set, with λ as its partner.
#[derive(Clone, Copy)]
enum On {
    Psi,
    Lambda,
}

/// A sweep dispatch and its rough cost.
type SweepDispatch = (On, Kernel, Push, u64);

/// Append an overlap `⟨λ|G|ψ⟩` writing its partial sums at `output`, for
/// a generator with one term `c · Π_controls ⊗ P(x, z)`.
fn push_overlap(
    params: &mut Vec<f64>,
    out: &mut Vec<SweepDispatch>,
    amps: u32,
    output: u32,
    (x, controls, z): (u32, u32, u32),
    c: C64,
) {
    let offset = params.len() as u32;
    params.extend([controls as f64, z as f64, c.re, c.im]);
    let push = Push {
        a: x,
        b: 1,
        c: output,
        units: amps,
        offset,
        chunk_bits: 0,
    };
    out.push((On::Psi, Kernel::Overlap, push, amps as u64));
}

/// Append `ops` (fused and planned as for `run`) on both states.
fn push_segment(
    params: &mut Vec<f64>,
    out: &mut Vec<SweepDispatch>,
    ops: &Circuit,
    tile_bits: u32,
) {
    let gpu_ops: Vec<GpuOp> = ops.fused().ops().iter().map(GpuOp::new).collect();
    let plan = Plan::new(&gpu_ops, ops.n_qubits(), tile_bits);
    let base = params.len() as u32;
    params.extend(&plan.params);
    for on in [On::Psi, On::Lambda] {
        for (&(kernel, push), &cost) in plan.dispatches.iter().zip(&plan.costs) {
            let push = Push {
                offset: push.offset + base,
                ..push
            };
            out.push((on, kernel, push, cost));
        }
    }
}

/// The backward sweep, consuming `ψ` and `λ`. Returns `Re⟨λ|ψ⟩` (the
/// expectation value when `λ = Hψ`) and the gradient.
fn sweep(
    circuit: &Circuit,
    parameters: &[f64],
    mut psi: GpuStatevector,
    lambda: GpuStatevector,
) -> Result<(f64, Vec<f64>), Error> {
    circuit.check_no_multi_controlled("the GPU engine")?;
    let bound = circuit.bind(parameters)?;
    let n = bound.n_qubits();
    let amps = psi.len() as u32;
    let per_overlap = Kernel::Overlap.groups(amps);
    let tile_bits = psi.ctx.tile_bits();

    let mut params: Vec<f64> = Vec::new();
    let mut dispatches: Vec<SweepDispatch> = Vec::new();
    // Each overlap's (parameter, scale) terms, in output order; the first
    // is the value ⟨λ|ψ⟩ (G = 1). Global phases cancel between ψ and λ.
    let mut overlaps: Vec<Vec<(usize, f64)>> = vec![Vec::new()];
    let one = C64::new(1.0, 0.0);
    push_overlap(&mut params, &mut dispatches, amps, 0, (0, 0, 0), one);

    let mut pending = Circuit::new(n);
    let mut bindings = circuit.bindings().iter().rev().peekable();
    for (k, op) in bound.ops().iter().enumerate().rev() {
        let mut terms = Vec::new();
        while let Some(b) = bindings.next_if(|b| b.op == k) {
            terms.push((b.parameter, b.scale));
        }
        if !terms.is_empty() {
            push_segment(&mut params, &mut dispatches, &pending, tile_bits);
            pending = Circuit::new(n);
            let g = Generator::of(op);
            let masks = (g.masks.x as u32, g.controls as u32, g.masks.z as u32);
            let output = overlaps.len() as u32 * per_overlap;
            let c = g.coeff * g.masks.y_phase();
            push_overlap(&mut params, &mut dispatches, amps, output, masks, c);
            overlaps.push(terms);
        }
        // Gates before the first parametric one are never needed.
        pending.push(op.inverse())?;
    }

    psi.ensure_buffers(
        &params,
        overlaps.len() * per_overlap as usize * psi.amplitude_bytes() as usize,
    )?;
    let mut sets = Sets::new(&psi.ctx);
    let (on_psi, on_lambda) = (
        sets.pair(&psi, &lambda, &psi)?,
        sets.pair(&lambda, &psi, &psi)?,
    );
    let mut start = 0;
    while start < dispatches.len() {
        let mut work = 0u64;
        let mut end = start;
        while end < dispatches.len() && (end == start || work < BATCH_UNITS) {
            work += dispatches[end].3;
            end += 1;
        }
        let batch: Vec<_> = dispatches[start..end]
            .iter()
            .map(|&(on, kernel, push, _)| {
                let set = match on {
                    On::Psi => on_psi,
                    On::Lambda => on_lambda,
                };
                (set, kernel, push)
            })
            .collect();
        psi.submit_with_sets(&batch)?;
        start = end;
    }
    drop(sets);

    let partials = psi.read_output(overlaps.len() * per_overlap as usize)?;
    let sums: Vec<C64> = partials
        .chunks(per_overlap as usize)
        .map(|c| c.iter().sum())
        .collect();
    let mut gradient = vec![0.0; parameters.len()];
    for (terms, sum) in overlaps.iter().zip(&sums).skip(1) {
        for &(parameter, scale) in terms {
            gradient[parameter] += scale * sum.im;
        }
    }
    Ok((sums[0].re, gradient))
}

/// Descriptor sets binding one state and a partner, freed together on
/// drop. Every set reads one owner's parameter and output buffers, which
/// must not be reallocated while the sets live.
pub(super) struct Sets {
    ctx: Arc<GpuContext>,
    sets: Vec<vk::DescriptorSet>,
}

impl Sets {
    pub(super) fn new(ctx: &Arc<GpuContext>) -> Self {
        Sets {
            ctx: ctx.clone(),
            sets: Vec::new(),
        }
    }

    /// A set with `state` at binding 0 and `partner` at binding 3, both of
    /// the same layout, reading `owner`'s parameters and output.
    pub(super) fn pair(
        &mut self,
        state: &GpuStatevector,
        partner: &GpuStatevector,
        owner: &GpuStatevector,
    ) -> Result<vk::DescriptorSet, Error> {
        let ctx = &self.ctx;
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
        self.sets.push(set);
        let chunks = |s: &GpuStatevector| -> Vec<vk::DescriptorBufferInfo> {
            (0..MAX_CHUNKS)
                .map(|i| {
                    vk::DescriptorBufferInfo::default()
                        .buffer(s.chunks[i % s.chunks.len()].buffer)
                        .range(vk::WHOLE_SIZE)
                })
                .collect()
        };
        let one = |buffer: vk::Buffer| {
            [vk::DescriptorBufferInfo::default()
                .buffer(buffer)
                .range(vk::WHOLE_SIZE)]
        };
        let (a, b) = (chunks(state), chunks(partner));
        let (params, output) = (one(owner.params.buffer), one(owner.output.buffer));
        let writes = [
            write_set(set, 0, &a),
            write_set(set, 1, &params),
            write_set(set, 2, &output),
            write_set(set, 3, &b),
        ];
        // SAFETY: the new set isn't in use yet.
        unsafe { ctx.device.update_descriptor_sets(&writes, &[]) };
        Ok(set)
    }
}

impl Drop for Sets {
    fn drop(&mut self) {
        if self.sets.is_empty() {
            return;
        }
        let _guard = self.ctx.lock.lock().expect("GPU lock poisoned");
        // SAFETY: every submission using the sets has been waited on.
        unsafe {
            let _ = self
                .ctx
                .device
                .free_descriptor_sets(self.ctx.descriptor_pool, &self.sets);
        }
    }
}
