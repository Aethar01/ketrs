//! Exact time evolution `exp(-iHt)|ψ⟩` on the GPU.
//!
//! The same Krylov time-stepping as on the CPU (see
//! [`crate::statevector`]'s `evolve`), with every vector on the GPU and
//! only scalars coming back. Each step runs the Lanczos recurrence once to
//! build the tridiagonal matrix, downloading `α` and `β` as it goes, until
//! the basis covers the remaining time (checked every [`KRYLOV_CHECK`]
//! vectors, up to [`KRYLOV_MAX`]), then replays it from the stored
//! coefficients to accumulate `Σₖ yₖ vₖ`, which needs no downloads at all.
//! Memory is four statevectors (the state and three Lanczos vectors)
//! however long the basis.

use super::context::{Kernel, Push};
use super::gradient::{Sets, hamil_params, hamil_push};
use super::{BATCH_UNITS, GpuStatevector};
use crate::error::Error;
use crate::observable::PauliSum;
use crate::statevector::evolve::{
    KRYLOV_CHECK, KRYLOV_MAX, STEP_TOLERANCE, krylov_covers, krylov_partial_step,
};
use crate::statevector::krylov::breakdown;
use ash::vk;
use num_complex::Complex64 as C64;

/// Room for the dispatches' own parameters after the Hamiltonian's: a
/// replay takes at most ten doubles per Lanczos vector.
const EXTRA_PARAMS: usize = 16 * (KRYLOV_MAX + 2);

impl GpuStatevector {
    /// Evolve by `exp(-iHt)` exactly (to about 1e-13 per step, or 1e-6 in
    /// single precision), on the GPU.
    /// See [`Statevector::evolve`](crate::Statevector::evolve); needs room
    /// for three more statevectors.
    pub fn evolve(&mut self, hamiltonian: &PauliSum, time: f64) -> Result<(), Error> {
        let mut shift = 0.0;
        let mut rest = PauliSum::new();
        for (c, s) in hamiltonian.simplify(0.0).terms() {
            if s.weight() == 0 {
                shift += c;
            } else {
                rest.add(*c, s.clone());
            }
        }
        if !rest.terms().is_empty() {
            let (hamil, n_groups) = hamil_params(&rest, self.n_qubits)?;
            let mut krylov = Krylov::new(self, hamil, n_groups)?;
            krylov.evolve(time)?;
        }
        if shift != 0.0 {
            let mut sets = Sets::new(&self.ctx);
            let set = sets.pair(self, self, self)?;
            let mut params = Vec::new();
            let push = scale(&mut params, C64::cis(-shift * time), self.len());
            self.ensure_buffers(&params, 0)?;
            self.submit_with_sets(&[(set, Kernel::Phase, push)])?;
        }
        Ok(())
    }
}

/// The working state of a Krylov evolution: the state, three Lanczos
/// vectors whose roles rotate, and descriptor sets for every pair.
struct Krylov<'a> {
    psi: &'a mut GpuStatevector,
    vectors: [GpuStatevector; 3],
    /// `sets[i][j]`: vector `i` at binding 0, `j` as partner, where index 3
    /// is ψ.
    sets: [[vk::DescriptorSet; 4]; 4],
    _owner: Sets,
    /// `hamil.comp` parameters, always at offset 0.
    hamil: Vec<f64>,
    n_groups: u32,
}

impl<'a> Krylov<'a> {
    fn new(psi: &'a mut GpuStatevector, hamil: Vec<f64>, n_groups: u32) -> Result<Self, Error> {
        let vectors = [(); 3].map(|()| {
            GpuStatevector::allocate(&psi.ctx, psi.n_qubits, psi.chunk_bits, psi.precision)
        });
        let [a, b, c] = vectors;
        let vectors = [a?, b?, c?];
        // Size ψ's buffers for the largest submission now: the sets below
        // point at them, so they must not move.
        let per = Kernel::Overlap.groups(psi.len() as u32) as usize;
        psi.ensure_buffers(
            &vec![0.0; hamil.len() + EXTRA_PARAMS],
            (RECORDS + per) * psi.amplitude_bytes() as usize,
        )?;
        let mut owner = Sets::new(&psi.ctx);
        let mut sets = [[vk::DescriptorSet::null(); 4]; 4];
        let state = |k: usize| if k == 3 { &*psi } else { &vectors[k] };
        for (i, row) in sets.iter_mut().enumerate() {
            for (j, set) in row.iter_mut().enumerate() {
                *set = owner.pair(state(i), state(j), psi)?;
            }
        }
        Ok(Krylov {
            psi,
            vectors,
            sets,
            _owner: owner,
            hamil,
            n_groups,
        })
    }

    fn amps(&self) -> usize {
        self.psi.len()
    }

    /// Upload `extra` after the Hamiltonian's parameters, run `dispatches`
    /// (with offsets relative to `extra`) in bounded batches, and return
    /// the sum of the first overlap's partial sums.
    fn submit(
        &mut self,
        extra: &[f64],
        dispatches: &[(vk::DescriptorSet, Kernel, Push)],
    ) -> Result<C64, Error> {
        let base = self.hamil.len() as u32;
        let mut params = self.hamil.clone();
        params.extend(extra);
        assert!(
            params.len() * 8 <= self.psi.params.size as usize,
            "parameters fit"
        );
        self.psi.params.write(super::bytes_of_f64(&params));
        let amps = self.amps() as u64;
        let mut start = 0;
        while start < dispatches.len() {
            let mut work = 0;
            let mut end = start;
            while end < dispatches.len() && (end == start || work < BATCH_UNITS) {
                let (_, kernel, _) = dispatches[end];
                work += if kernel == Kernel::Hamil {
                    amps * self.n_groups as u64 / 4 + amps
                } else {
                    amps
                };
                end += 1;
            }
            let batch: Vec<_> = dispatches[start..end]
                .iter()
                .map(|&(set, kernel, push)| {
                    let offset = if kernel == Kernel::Hamil {
                        0
                    } else {
                        push.offset + base
                    };
                    (set, kernel, Push { offset, ..push })
                })
                .collect();
            self.psi.submit_with_sets(&batch)?;
            start = end;
        }
        let per = Kernel::Overlap.groups(self.amps() as u32) as usize;
        Ok(self.psi.read_output(per)?.iter().sum())
    }

    /// Copy ψ into vector `v`.
    fn copy_psi_to(&self, v: usize) -> Result<(), Error> {
        let size = self.psi.amplitude_bytes() << self.psi.chunk_bits;
        let ctx = &self.psi.ctx;
        ctx.submit(|cmd| {
            for (from, to) in self.psi.chunks.iter().zip(&self.vectors[v].chunks) {
                let region = [vk::BufferCopy::default().size(size)];
                // SAFETY: both chunks hold `size` bytes.
                unsafe {
                    ctx.device
                        .cmd_copy_buffer(cmd, from.buffer, to.buffer, &region)
                };
            }
        })
    }

    /// `‖ψ‖`.
    fn psi_norm(&mut self) -> Result<f64, Error> {
        let mut extra = Vec::new();
        let set = self.sets[3][3];
        let push = overlap(&mut extra, 0, self.amps());
        Ok(self
            .submit(&extra, &[(set, Kernel::Overlap, push)])?
            .re
            .sqrt())
    }

    /// `ψ ← exp(-iHt) ψ` in adaptive Krylov steps.
    fn evolve(&mut self, time: f64) -> Result<(), Error> {
        let amps = self.amps();
        let tolerance = STEP_TOLERANCE.max(10.0 * self.psi.precision.epsilon());
        let mut remaining = time;
        while remaining != 0.0 {
            let beta0 = self.psi_norm()?;
            if beta0 == 0.0 {
                break;
            }

            // Pass 1: the recurrence, `KRYLOV_CHECK` steps per submission.
            // `lanczos.comp` sums each α and β on the GPU, writes the
            // coefficients the following dispatches need into the
            // parameters, and records α and β, read back after the batch.
            let (mut alphas, mut betas): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
            let mut covered = None;
            let (mut prev, mut v, mut w) = (0usize, 1usize, 2usize);
            self.copy_psi_to(v)?;
            let per = Kernel::Overlap.groups(amps as u32) as usize;
            let units = amps as u32;
            let mut k = 0;
            'basis: while k < KRYLOV_MAX {
                let end = (k + KRYLOV_CHECK).min(KRYLOV_MAX);
                let mut extra = Vec::new();
                let o = overlap(&mut extra, RECORDS as u32, amps);
                // A block of parameters per step, plus one for the step
                // after the batch: the factor scaling v (as a phase, four
                // doubles), −β of the previous step and −α. The first step's
                // come from the host; the rest from `lanczos.comp`.
                let first = extra.len() as u32;
                let block = |j: usize| first + 8 * (j - k) as u32;
                let (scale_v, beta_prev) = match k {
                    0 => (1.0 / beta0, 0.0),
                    _ => (1.0 / betas[k - 1], betas[k - 1]),
                };
                extra.extend([scale_v, 0.0, scale_v, 0.0, -beta_prev, 0.0, 0.0, 0.0]);
                extra.resize(extra.len() + 8 * (end - k), 0.0);
                let mut dispatches = Vec::new();
                for j in k..end {
                    let at = block(j);
                    let with = |offset: u32| Push {
                        units,
                        offset,
                        ..Push::default()
                    };
                    // v ← v · scale; w = H v; α = ⟨v|w⟩.
                    dispatches.push((self.sets[v][v], Kernel::Phase, with(at)));
                    dispatches.push((
                        self.sets[v][w],
                        Kernel::Hamil,
                        hamil_push(self.n_groups, amps),
                    ));
                    dispatches.push((self.sets[w][v], Kernel::Overlap, o));
                    dispatches.push((
                        self.sets[w][v],
                        Kernel::Lanczos,
                        lanczos(0, j - k, per, at + 6),
                    ));
                    // w ← w − α v − β v_prev; β = ‖w‖.
                    dispatches.push((self.sets[v][w], Kernel::Axpy, with(at + 6)));
                    if j > 0 {
                        dispatches.push((self.sets[prev][w], Kernel::Axpy, with(at + 4)));
                    }
                    dispatches.push((self.sets[w][w], Kernel::Overlap, o));
                    dispatches.push((
                        self.sets[w][w],
                        Kernel::Lanczos,
                        lanczos(1, j - k, per, block(j + 1)),
                    ));
                    (prev, v, w) = (v, w, prev);
                }
                self.submit(&extra, &dispatches)?;
                for (j, record) in (k..end).zip(self.psi.read_output(end - k)?) {
                    let (alpha, beta) = (record.re, record.im);
                    alphas.push(alpha);
                    betas.push(beta);
                    let m = j + 1;
                    // A closed Krylov space is invariant: the step is exact.
                    let closed = beta <= breakdown(&alphas);
                    if closed || m.is_multiple_of(KRYLOV_CHECK) || m == KRYLOV_MAX {
                        let residual = if closed { 0.0 } else { beta };
                        covered =
                            krylov_covers(&alphas, &betas[..m - 1], residual, remaining, tolerance);
                        if covered.is_some() || closed {
                            break 'basis;
                        }
                    }
                }
                k = end;
            }
            let m = alphas.len();
            let (dt, y) = match covered {
                Some(y) => (remaining, y),
                None => krylov_partial_step(
                    &alphas,
                    &betas[..m - 1],
                    betas[m - 1],
                    remaining,
                    time,
                    tolerance,
                )?,
            };

            // Pass 2: replay from the stored coefficients, ψ ← β₀ Σ yₖ vₖ.
            let (mut prev, mut v, mut w) = (0usize, 1usize, 2usize);
            self.copy_psi_to(v)?;
            let mut extra = Vec::new();
            let mut dispatches = vec![
                (
                    self.sets[v][v],
                    Kernel::Phase,
                    scale(&mut extra, C64::new(1.0 / beta0, 0.0), amps),
                ),
                (
                    self.sets[3][3],
                    Kernel::Phase,
                    scale(&mut extra, C64::new(0.0, 0.0), amps),
                ),
            ];
            for k in 0..m {
                dispatches.push((
                    self.sets[v][3],
                    Kernel::Axpy,
                    axpy(&mut extra, y[k] * beta0, amps),
                ));
                if k + 1 == m {
                    break;
                }
                dispatches.push((
                    self.sets[v][w],
                    Kernel::Hamil,
                    hamil_push(self.n_groups, amps),
                ));
                dispatches.push((
                    self.sets[v][w],
                    Kernel::Axpy,
                    axpy(&mut extra, C64::new(-alphas[k], 0.0), amps),
                ));
                if k > 0 {
                    dispatches.push((
                        self.sets[prev][w],
                        Kernel::Axpy,
                        axpy(&mut extra, C64::new(-betas[k - 1], 0.0), amps),
                    ));
                }
                (prev, v, w) = (v, w, prev);
                dispatches.push((
                    self.sets[v][v],
                    Kernel::Phase,
                    scale(&mut extra, C64::new(1.0 / betas[k], 0.0), amps),
                ));
            }
            self.submit(&extra, &dispatches)?;
            remaining = if dt == remaining { 0.0 } else { remaining - dt };
        }
        Ok(())
    }
}

/// Output slots before the overlaps' partial sums, where `lanczos.comp`
/// records each step of a batch's `(α, β)`.
const RECORDS: usize = KRYLOV_CHECK;

/// `lanczos.comp` for one step's α (`mode` 0) or β (1) from the `per`
/// partial sums after [`RECORDS`], recorded in slot `record`, writing
/// coefficients at `offset`.
fn lanczos(mode: u32, record: usize, per: usize, offset: u32) -> Push {
    Push {
        a: mode,
        b: record as u32,
        c: RECORDS as u32,
        units: per as u32,
        offset,
        chunk_bits: 0,
    }
}

/// `phase.comp` multiplying every amplitude by `a`.
fn scale(extra: &mut Vec<f64>, a: C64, amps: usize) -> Push {
    let offset = extra.len() as u32;
    extra.extend([a.re, a.im, a.re, a.im]);
    Push {
        units: amps as u32,
        offset,
        ..Push::default()
    }
}

/// `axpy.comp`: partner += a · state.
fn axpy(extra: &mut Vec<f64>, a: C64, amps: usize) -> Push {
    let offset = extra.len() as u32;
    extra.extend([a.re, a.im]);
    Push {
        units: amps as u32,
        offset,
        ..Push::default()
    }
}

/// `overlap.comp` with `G = 1`: `⟨partner|state⟩` into partial sums at
/// `output`.
fn overlap(extra: &mut Vec<f64>, output: u32, amps: usize) -> Push {
    let offset = extra.len() as u32;
    extra.extend([0.0, 0.0, 1.0, 0.0]);
    Push {
        a: 0,
        b: 1,
        c: output,
        units: amps as u32,
        offset,
        chunk_bits: 0,
    }
}
