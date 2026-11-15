//! The subspace expansion's mixed environment on the GPU (a [`Chain`]).
//!
//! At each bond the channels of `H|ψ⟩` come out of the first two stages of
//! an environment update (a matrix product, then the MPO's sparse step), as
//! in [`GpuSweep::update_right`], but against the mixed environment, whose
//! rows are on the expanded basis. Their part outside ψ's basis,
//! `K = C·others†`, is formed and factorised (`R` of `K = Q R`, see
//! [`qr`](super::qr)) on the GPU, so only `R` comes back. The CPU takes its
//! SVD and builds the expanded tensor, which then closes the environment
//! update on the GPU. Small bonds are factorised on the CPU, from the
//! channels themselves.

use super::qr::{QrScratch, qr_r};
use super::*;
use crate::mps::expand::{Chain, complement, directions};
use crate::mps::linalg::complement_rows;
use faer::MatRef;

/// Below this many entries in the channels' part outside the basis, the
/// CPU factorises it: the GPU's column steps would cost more than they save.
/// Unit tests use the GPU for nearly every bond.
const GPU_FACTOR_SIZE: usize = if cfg!(test) { 64 } else { 1 << 16 };

/// Device scratch for a chain, grown as needed.
#[derive(Default)]
struct ChainScratch {
    /// The ket, `others` and the expanded tensor, on the device and staged.
    x: Option<Buffer>,
    x_up: Option<Buffer>,
    others: Option<Buffer>,
    others_up: Option<Buffer>,
    bra: Option<Buffer>,
    bra_up: Option<Buffer>,
    /// The ket against each environment block.
    y: Option<Buffer>,
    /// The channels.
    z: Option<Buffer>,
    z_down: Option<Buffer>,
    /// The channels' part outside the basis, column-major.
    outside: Option<Buffer>,
    partials: Option<Buffer>,
    partials_down: Option<Buffer>,
    /// The environment, alternating between the two.
    env: [Option<Buffer>; 2],
    qr: QrScratch,
}

/// A [`Chain`] on the GPU.
pub(in crate::mps) struct GpuChain<'a> {
    sweep: &'a GpuSweep,
    s: ChainScratch,
    /// The environment in `s.env[current]`: `blocks` blocks between a bra
    /// bond of `bra` and a ket bond of `ket`, each stored `ket × bra` while
    /// expanding right bases (as right environments are) and `bra × ket`
    /// while expanding left bases.
    current: usize,
    blocks: usize,
    bra: usize,
    ket: usize,
    /// The channels' layout at the last site: rows per MPO index, MPO
    /// indices, and the ket bond it leaves for the next environment.
    last: (usize, usize),
}

impl GpuSweep {
    /// A chain starting from the boundary.
    pub(in crate::mps) fn chain(&self) -> Result<GpuChain<'_>, Error> {
        let mut s = ChainScratch::default();
        s.env[0] = Some(upload(&self.ctx, bytes_of(&[C64::new(1.0, 0.0)]))?);
        Ok(GpuChain {
            sweep: self,
            s,
            current: 0,
            blocks: 1,
            bra: 1,
            ket: 1,
            last: (0, 0),
        })
    }
}

/// Stage `data` in `up` and return the device address of `device`, which
/// the staged data is copied into by the next submission (see
/// [`GpuChain::copies`]).
fn stage(
    ctx: &Arc<GpuContext>,
    device: &mut Option<Buffer>,
    up: &mut Option<Buffer>,
    data: &[C64],
) -> Result<u64, Error> {
    let bytes = (data.len().max(1) as u64) * COMPLEX;
    ensure(ctx, up, bytes, Memory::Upload)?.write(bytes_of(data));
    Ok(ensure(ctx, device, bytes, Memory::Device)?.device_address())
}

/// The ingredients of a bond's factorisation: the basis's complement and
/// whether the GPU forms and factorises `K`.
struct Outside {
    others: Mat<C64>,
    on_gpu: bool,
}

impl GpuChain<'_> {
    fn ctx(&self) -> &Arc<GpuContext> {
        &self.sweep.ctx
    }

    fn device(
        &mut self,
        which: fn(&mut ChainScratch) -> &mut Option<Buffer>,
        len: usize,
    ) -> Result<u64, Error> {
        let ctx = self.sweep.ctx.clone();
        Ok(ensure(
            &ctx,
            which(&mut self.s),
            (len.max(1) as u64) * COMPLEX,
            Memory::Device,
        )?
        .device_address())
    }

    fn buffer(slot: &Option<Buffer>) -> vk::Buffer {
        slot.as_ref().expect("allocated").buffer
    }

    /// The complement of `basis`, if there is room, and where to factorise
    /// the `m × cols` channels' part outside it.
    fn outside(&self, basis: MatRef<'_, C64>, room: usize, m: usize) -> Option<Outside> {
        if room == 0 {
            return None;
        }
        let others = complement_rows(basis);
        let n = others.nrows();
        (n > 0).then(|| Outside {
            on_gpu: m > n && m * n >= GPU_FACTOR_SIZE,
            others,
        })
    }

    /// Record the norm of the `len` channels and its download.
    fn record_norm(&self, cmd: vk::CommandBuffer, z: u64, len: usize, partials: u64) {
        let ctx = self.ctx();
        let sums = len.div_ceil(256).clamp(1, 256);
        let push = PushBytes::default()
            .addr(z)
            .addr(partials)
            .addr(partials)
            .addr(partials)
            .u32s(&[len, 0, 0, 0, 7, sums]);
        ctx.mps_dispatch(cmd, MpsKernel::Qr, &push.0, [sums as u32, 1, 1]);
        compute_barrier(ctx, cmd);
        let (from, to) = (
            Self::buffer(&self.s.partials),
            Self::buffer(&self.s.partials_down),
        );
        copy(ctx, cmd, from, to, sums as u64 * COMPLEX);
    }

    fn read_norm(&self, len: usize) -> Result<f64, Error> {
        let sums = len.div_ceil(256).clamp(1, 256);
        let raw = self
            .s
            .partials_down
            .as_ref()
            .expect("allocated")
            .mapped(sums * COMPLEX as usize)?;
        Ok(read_complex(raw, sums)
            .iter()
            .map(|v| v.re)
            .sum::<f64>()
            .sqrt())
    }

    /// The new directions, given the channels at `z` (`len` entries) and
    /// their factorisation set up by `outside`. `rows` turns the downloaded
    /// channels into the `m × cols` matrix for the CPU's factorisation.
    #[allow(clippy::too_many_arguments)]
    fn finish(
        &mut self,
        basis: MatRef<'_, C64>,
        room: usize,
        outside: Option<Outside>,
        m: usize,
        len: usize,
        rows: impl Fn(&[C64]) -> Mat<C64>,
    ) -> Result<Mat<C64>, Error> {
        let cols = basis.ncols();
        let scale = self.read_norm(len)?;
        let Some(outside) = outside else {
            return Ok(Mat::zeros(0, cols));
        };
        if scale == 0.0 {
            return Ok(Mat::zeros(0, cols));
        }
        if !outside.on_gpu {
            let ctx = self.sweep.ctx.clone();
            let bytes = len as u64 * COMPLEX;
            ensure(&ctx, &mut self.s.z_down, bytes, Memory::Download)?;
            let (z, down) = (Self::buffer(&self.s.z), Self::buffer(&self.s.z_down));
            ctx.submit(|cmd| copy(&ctx, cmd, z, down, bytes))?;
            let raw = self
                .s
                .z_down
                .as_ref()
                .expect("allocated")
                .mapped(bytes as usize)?;
            let channels = rows(&read_complex(raw, len));
            return Ok(complement(channels.as_ref(), basis, room));
        }
        let n = outside.others.nrows();
        let ctx = self.sweep.ctx.clone();
        let (k, k_buffer) = (
            self.s.outside.as_ref().expect("allocated").device_address(),
            Self::buffer(&self.s.outside),
        );
        let r = qr_r(&ctx, k, k_buffer, m, n, &mut self.s.qr)?;
        Ok(directions(r.as_ref(), scale, outside.others.as_ref(), room))
    }

    /// Stage `others` and size `K` for the GPU's factorisation.
    fn stage_outside(
        &mut self,
        outside: &Option<Outside>,
        m: usize,
    ) -> Result<Option<(u64, usize)>, Error> {
        match outside {
            Some(o) if o.on_gpu => {
                let ctx = self.sweep.ctx.clone();
                let data: Vec<C64> = (0..o.others.nrows())
                    .flat_map(|r| (0..o.others.ncols()).map(move |c| (r, c)))
                    .map(|(r, c)| o.others[(r, c)])
                    .collect();
                let addr = stage(&ctx, &mut self.s.others, &mut self.s.others_up, &data)?;
                self.device(|s| &mut s.outside, m * o.others.nrows())?;
                Ok(Some((addr, o.others.nrows())))
            }
            _ => Ok(None),
        }
    }

    /// Record the copies of every staged upload.
    fn copies(&self, cmd: vk::CommandBuffer, staged: &[(&Option<Buffer>, &Option<Buffer>, usize)]) {
        let ctx = self.ctx();
        for &(up, device, len) in staged {
            copy(
                ctx,
                cmd,
                Self::buffer(up),
                Self::buffer(device),
                (len.max(1) as u64) * COMPLEX,
            );
        }
        compute_barrier(ctx, cmd);
    }
}

impl Chain for GpuChain<'_> {
    fn right(
        &mut self,
        i: usize,
        ket: &Tensor,
        mpo: &Mpo,
        basis: MatRef<'_, C64>,
        room: usize,
    ) -> Result<Mat<C64>, Error> {
        let w = &mpo.sites[i];
        let (dl, dr) = (w.left, w.right);
        let (l, rk, rb) = (ket.left, ket.right, self.bra);
        assert_eq!(
            (self.blocks, self.ket),
            (dr, rk),
            "environment matches the site"
        );
        let cols = 2 * rb;
        let (m, len) = (dl * l, dl * l * cols);
        let ctx = self.sweep.ctx.clone();
        let x = stage(&ctx, &mut self.s.x, &mut self.s.x_up, &ket.data)?;
        let y = self.device(|s| &mut s.y, dr * 2 * l * rb)?;
        let z = self.device(|s| &mut s.z, len)?;
        let partials = self.device(|s| &mut s.partials, 256)?;
        ensure(
            &ctx,
            &mut self.s.partials_down,
            256 * COMPLEX,
            Memory::Download,
        )?;
        let outside = self.outside(basis, room, m);
        let staged_outside = self.stage_outside(&outside, m)?;
        let env = self.s.env[self.current]
            .as_ref()
            .expect("environment")
            .device_address();
        let table = self.sweep.by_left[i];
        let k = self.s.outside.as_ref().map(|b| b.device_address());
        let s = &self.s;
        let mut staged = vec![(&s.x_up, &s.x, ket.data.len())];
        if let Some((_, n)) = staged_outside {
            staged.push((&s.others_up, &s.others, n * cols));
        }
        ctx.submit(|cmd| {
            self.copies(cmd, &staged);
            // y_c = A R_cᵀ, A as 2l × r_ket and R_cᵀ as r_ket × r_bra.
            gemm(
                &ctx,
                cmd,
                Gemm {
                    a: Operand::new(x, 0, rk, 1),
                    b: Operand::new(env, rk * rb, rb, 1),
                    c: y,
                    c_batch: 2 * l * rb,
                    c_row: rb,
                    m: 2 * l,
                    n: rb,
                    k: rk,
                    batch: dr,
                    add: false,
                },
                None,
            );
            compute_barrier(&ctx, cmd);
            // The channels z_a[row, (p_out, col)] = Σ W … y_c[(row, p_in), col].
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
                    cols: rb,
                    in_strides: [2 * l * rb, 2 * rb, rb, 1],
                    out_strides: [l * cols, cols, rb, 1],
                },
            );
            compute_barrier(&ctx, cmd);
            self.record_norm(cmd, z, len, partials);
            if let Some((others, n)) = staged_outside {
                // Kᵀ = conj(others) Cᵀ: K column-major.
                gemm(
                    &ctx,
                    cmd,
                    Gemm {
                        a: Operand::new(others, 0, cols, 1).conj(),
                        b: Operand::new(z, 0, 1, cols),
                        c: k.expect("allocated"),
                        c_batch: 0,
                        c_row: m,
                        m: n,
                        n: m,
                        k: cols,
                        batch: 1,
                        add: false,
                    },
                    None,
                );
                compute_barrier(&ctx, cmd);
            }
        })?;
        self.last = (l, dl);
        self.finish(basis, room, outside, m, len, |values| {
            Mat::from_fn(m, cols, |r, c| values[r * cols + c])
        })
    }

    fn advance_right(&mut self, expanded: &Tensor) -> Result<(), Error> {
        let (l, dl) = self.last;
        let (lb, rb) = (expanded.left, expanded.right);
        assert_eq!(rb, self.bra);
        let cols = 2 * rb;
        let ctx = self.sweep.ctx.clone();
        let bra = stage(&ctx, &mut self.s.bra, &mut self.s.bra_up, &expanded.data)?;
        let next = 1 - self.current;
        let out = ensure(
            &ctx,
            &mut self.s.env[next],
            (dl * l * lb).max(1) as u64 * COMPLEX,
            Memory::Device,
        )?
        .device_address();
        let z = self.s.z.as_ref().expect("channels").device_address();
        let s = &self.s;
        let staged = [(&s.bra_up, &s.bra, expanded.data.len())];
        ctx.submit(|cmd| {
            self.copies(cmd, &staged);
            // The next environment, stored transposed: z_a bra†.
            gemm(
                &ctx,
                cmd,
                Gemm {
                    a: Operand::new(z, l * cols, cols, 1),
                    b: Operand::new(bra, 0, 1, cols).conj(),
                    c: out,
                    c_batch: l * lb,
                    c_row: lb,
                    m: l,
                    n: lb,
                    k: cols,
                    batch: dl,
                    add: false,
                },
                None,
            );
        })?;
        (self.current, self.blocks, self.bra, self.ket) = (next, dl, lb, l);
        Ok(())
    }

    fn left(
        &mut self,
        i: usize,
        ket: &Tensor,
        mpo: &Mpo,
        basis: MatRef<'_, C64>,
        room: usize,
    ) -> Result<Mat<C64>, Error> {
        let w = &mpo.sites[i];
        let (dl, dr) = (w.left, w.right);
        let (lk, r, lb) = (ket.left, ket.right, self.bra);
        assert_eq!(
            (self.blocks, self.ket),
            (dl, lk),
            "environment matches the site"
        );
        let cols = 2 * lb;
        let (m, len) = (dr * r, dr * r * cols);
        let ctx = self.sweep.ctx.clone();
        let x = stage(&ctx, &mut self.s.x, &mut self.s.x_up, &ket.data)?;
        let y = self.device(|s| &mut s.y, dl * lb * 2 * r)?;
        let z = self.device(|s| &mut s.z, len)?;
        let partials = self.device(|s| &mut s.partials, 256)?;
        ensure(
            &ctx,
            &mut self.s.partials_down,
            256 * COMPLEX,
            Memory::Download,
        )?;
        let outside = self.outside(basis, room, m);
        let staged_outside = self.stage_outside(&outside, m)?;
        let env = self.s.env[self.current]
            .as_ref()
            .expect("environment")
            .device_address();
        let table = self.sweep.by_right[i];
        let k = self.s.outside.as_ref().map(|b| b.device_address());
        let s = &self.s;
        let mut staged = vec![(&s.x_up, &s.x, ket.data.len())];
        if let Some((_, n)) = staged_outside {
            staged.push((&s.others_up, &s.others, n * cols));
        }
        ctx.submit(|cmd| {
            self.copies(cmd, &staged);
            // y_a = L_a A, the blocks stacked: (D·l_bra × l_ket)(l_ket × 2r).
            gemm(
                &ctx,
                cmd,
                Gemm {
                    a: Operand::new(env, 0, lk, 1),
                    b: Operand::new(x, 0, 2 * r, 1),
                    c: y,
                    c_batch: 0,
                    c_row: 2 * r,
                    m: dl * lb,
                    n: 2 * r,
                    k: lk,
                    batch: 1,
                    add: false,
                },
                None,
            );
            compute_barrier(&ctx, cmd);
            // z[(row, p_out), (b, col)] = Σ W … y_a[row, (p_in, col)]: the
            // channels, transposed.
            mix(
                &ctx,
                cmd,
                Mix {
                    input: y,
                    output: z,
                    table,
                    blocks: dr,
                    phys: 2,
                    rows: lb,
                    cols: r,
                    in_strides: [lb * 2 * r, 2 * r, r, 1],
                    out_strides: [r, 2 * m, m, 1],
                },
            );
            compute_barrier(&ctx, cmd);
            self.record_norm(cmd, z, len, partials);
            if let Some((others, n)) = staged_outside {
                // Kᵀ = conj(others) Cᵀ, with Cᵀ = z.
                gemm(
                    &ctx,
                    cmd,
                    Gemm {
                        a: Operand::new(others, 0, cols, 1).conj(),
                        b: Operand::new(z, 0, m, 1),
                        c: k.expect("allocated"),
                        c_batch: 0,
                        c_row: m,
                        m: n,
                        n: m,
                        k: cols,
                        batch: 1,
                        add: false,
                    },
                    None,
                );
                compute_barrier(&ctx, cmd);
            }
        })?;
        self.last = (r, dr);
        self.finish(basis, room, outside, m, len, |values| {
            Mat::from_fn(m, cols, |row, c| values[c * m + row])
        })
    }

    fn advance_left(&mut self, expanded: &Tensor) -> Result<(), Error> {
        let (r, dr) = self.last;
        let (lb, rb) = (expanded.left, expanded.right);
        assert_eq!(lb, self.bra);
        let cols = 2 * lb;
        let ctx = self.sweep.ctx.clone();
        let bra = stage(&ctx, &mut self.s.bra, &mut self.s.bra_up, &expanded.data)?;
        let next = 1 - self.current;
        let out = ensure(
            &ctx,
            &mut self.s.env[next],
            (dr * rb * r).max(1) as u64 * COMPLEX,
            Memory::Device,
        )?
        .device_address();
        let z = self.s.z.as_ref().expect("channels").device_address();
        let s = &self.s;
        let staged = [(&s.bra_up, &s.bra, expanded.data.len())];
        ctx.submit(|cmd| {
            self.copies(cmd, &staged);
            // L'_b = bra† z_b, bra as 2l × r'.
            gemm(
                &ctx,
                cmd,
                Gemm {
                    a: Operand::new(bra, 0, 1, rb).conj(),
                    b: Operand::new(z, r, dr * r, 1),
                    c: out,
                    c_batch: rb * r,
                    c_row: r,
                    m: rb,
                    n: r,
                    k: cols,
                    batch: dr,
                    add: false,
                },
                None,
            );
        })?;
        (self.current, self.blocks, self.bra, self.ket) = (next, dr, rb, r);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::gpu::GpuContext;
    use crate::mps::envs::Envs;
    use crate::mps::linalg::matmul;
    use crate::mps::mpo::Mpo;
    use crate::mps::{Mps, MpsOptions};
    use crate::observable::{Pauli, PauliString, PauliSum};
    use faer::Mat;
    use num_complex::Complex64 as C64;

    /// Couplings up to five sites apart and four-qubit strings, so the MPO's
    /// bonds are a few tens.
    fn hamiltonian(n: usize) -> PauliSum {
        let mut h = PauliSum::new();
        for i in 0..n {
            h.add(
                0.3 + 0.1 * i as f64,
                PauliString::new([(i, Pauli::X)]).unwrap(),
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

    fn gpu() -> Option<std::sync::Arc<GpuContext>> {
        GpuContext::new().ok().filter(|c| c.supports_mps())
    }

    /// The right basis of bond `i − 1` as states of sites `i …`: row `b` is
    /// the state of right basis vector `b` (orthonormal rows).
    fn right_states(mps: &Mps, i: usize) -> Mat<C64> {
        let n = mps.tensors.len();
        let mut states = mps.tensors[n - 1].as_right_matrix().to_owned();
        for t in mps.tensors[i..n - 1].iter().rev() {
            let width = states.ncols();
            let prod = matmul(t.as_left_matrix(), states.as_ref());
            states = Mat::from_fn(t.left, 2 * width, |l, c| {
                prod[(2 * l + c / width, c % width)]
            });
        }
        states
    }

    /// The left basis of bond `i` as states of sites `… i`: column `b` is
    /// the state of left basis vector `b` (orthonormal columns).
    fn left_states(mps: &Mps, i: usize) -> Mat<C64> {
        let mut states = mps.tensors[0].as_left_matrix().to_owned();
        for t in &mps.tensors[1..=i] {
            let prod = matmul(states.as_ref(), t.as_right_matrix());
            states = Mat::from_fn(2 * states.nrows(), t.right, |row, c| {
                prod[(row / 2, (row % 2) * t.right + c)]
            });
        }
        states
    }

    /// Whether `overlap` (between two orthonormal bases) is unitary: the two
    /// bases span the same space.
    fn assert_unitary(overlap: Mat<C64>, what: &str) {
        let product = matmul(overlap.adjoint(), overlap.as_ref());
        let identity = Mat::<C64>::identity(product.nrows(), product.ncols());
        let diff = (product - identity).norm_max();
        assert!(diff < 1e-9, "{what}: {diff}");
    }

    #[test]
    fn gpu_expansion_spans_the_cpu_expansion() {
        let Some(ctx) = gpu() else { return };
        let n = 12;
        let mpo = Mpo::new(&hamiltonian(n), n).unwrap();
        let options = MpsOptions {
            max_bond: 40,
            cutoff: 1e-12,
        };
        // Random states with small bonds. (A product state's channels have
        // exactly degenerate singular values, so where `room` cuts through
        // them either expansion is as good, and rounding picks.)
        for start in [Mps::random(n, 3, 2, options), Mps::random(n, 2, 9, options)] {
            let before = start.to_amplitudes();
            let mut cpu = start.clone();
            let mut gpu = start;
            gpu.set_gpu(Some(ctx.clone())).unwrap();
            for mps in [&mut cpu, &mut gpu] {
                let envs = Envs::new(mps, &mpo).unwrap();
                mps.expand_right_bases(&mpo, &envs).unwrap();
            }
            assert_eq!(cpu.bond_dimensions(), gpu.bond_dimensions());
            // Each bond's right basis spans the same states. (The bases
            // themselves differ, by a phase per new direction.)
            for i in 1..n {
                let (c, g) = (right_states(&cpu, i), right_states(&gpu, i));
                assert_unitary(matmul(c.as_ref(), g.adjoint()), "right bases");
            }
            for mps in [&mut cpu, &mut gpu] {
                let envs = Envs::new(mps, &mpo).unwrap();
                mps.expand_left_bases(&mpo, &envs).unwrap();
            }
            assert_eq!(cpu.bond_dimensions(), gpu.bond_dimensions());
            for i in 0..n - 1 {
                let (c, g) = (left_states(&cpu, i), left_states(&gpu, i));
                assert_unitary(matmul(c.adjoint(), g.as_ref()), "left bases");
            }
            // The state itself is unchanged.
            for (a, b) in gpu.to_amplitudes().iter().zip(&before) {
                assert!((a - b).norm() < 1e-12);
            }
        }
    }

    #[test]
    fn gpu_expansion_of_a_product_state_keeps_it() {
        let Some(ctx) = gpu() else { return };
        let n = 12;
        let mpo = Mpo::new(&hamiltonian(n), n).unwrap();
        let neel: Vec<bool> = (0..n).map(|q| q % 2 == 1).collect();
        let options = MpsOptions {
            max_bond: 40,
            cutoff: 1e-12,
        };
        let mut cpu = Mps::basis(n, &neel, options);
        let mut gpu = cpu.clone();
        gpu.set_gpu(Some(ctx)).unwrap();
        for mps in [&mut cpu, &mut gpu] {
            let envs = Envs::new(mps, &mpo).unwrap();
            mps.expand_right_bases(&mpo, &envs).unwrap();
            mps.expand_left_bases(&mpo, &envs).unwrap();
        }
        assert_eq!(cpu.bond_dimensions(), gpu.bond_dimensions());
        for (a, b) in gpu.to_amplitudes().iter().zip(cpu.to_amplitudes()) {
            assert!((a - b).norm() < 1e-12);
        }
    }

    #[test]
    fn tdvp_with_the_gpu_expansion_matches_the_cpu() {
        let Some(ctx) = gpu() else { return };
        let n = 12;
        let h = hamiltonian(n);
        let neel: Vec<bool> = (0..n).map(|q| q % 2 == 1).collect();
        let options = MpsOptions {
            max_bond: 24,
            cutoff: 1e-12,
        };
        let mut cpu = Mps::basis(n, &neel, options);
        let mut gpu = cpu.clone();
        gpu.set_gpu(Some(ctx)).unwrap();
        cpu.tdvp(&h, 0.6, 3).unwrap();
        gpu.tdvp(&h, 0.6, 3).unwrap();
        assert_eq!(cpu.bond_dimensions(), gpu.bond_dimensions());
        let diff = cpu
            .to_amplitudes()
            .iter()
            .zip(gpu.to_amplitudes())
            .map(|(a, b)| (a - b).norm())
            .fold(0.0, f64::max);
        assert!(diff < 1e-9, "{diff}");
    }
}
