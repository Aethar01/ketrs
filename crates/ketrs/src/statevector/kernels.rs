//! Gate application kernels, generic over the amplitude type.
//!
//! Operations are first converted to [`Prepared`] form: matrices in the
//! amplitude type, with the cheapest kernel chosen. This happens once per
//! circuit, and is where a real statevector rejects complex gates.
//!
//! Every kernel is built on [`for_each_pair`], which hands out matching
//! slices `lo` and `hi` where `hi[j]` is `lo[j]` with one bit set. Large
//! states are split across threads there, so the kernels stay serial.
//! `tests/common` keeps a plain index-loop version these are checked against.
//!
//! Large states don't fit in cache, so each gate streams the whole state
//! through memory. [`run`] reduces that by applying runs of gates on low
//! qubits block by block, keeping each block in cache.

use crate::circuit::Operation;
use crate::error::Error;
use crate::observable::PauliMasks;
use crate::scalar::{Scalar, convert_matrix};
use num_complex::Complex64 as C64;
use rayon::prelude::*;

/// States with at least this many amplitudes are processed in parallel.
pub(super) const PARALLEL_THRESHOLD: usize = 1 << 16;

/// Gates acting only on qubits below this are run block by block, on
/// blocks of `2^CACHE_BLOCK_QUBITS` amplitudes (256 KiB of complex values,
/// inside L2). Must stay below `PARALLEL_THRESHOLD` so per-block kernels run
/// serially.
pub(super) const CACHE_BLOCK_QUBITS: usize = 14;

/// Amplitudes handed to each parallel task.
pub(super) const BLOCK: usize = 1 << 12;

/// An operation ready to apply to amplitudes of type `T`.
#[derive(Clone)]
pub(super) enum Prepared<T> {
    Pair {
        op: PairOp<T>,
        target: usize,
    },
    Controlled {
        op: PairOp<T>,
        control: usize,
        target: usize,
    },
    Swap(usize, usize),
    /// A 4×4 matrix with basis index `2·b(q1) + b(q0)`.
    Quad {
        op: QuadOp<T>,
        q0: usize,
        q1: usize,
    },
    Pauli {
        masks: PauliMasks,
        op: RotationOp<T>,
    },
}

impl<T: Scalar> Prepared<T> {
    /// Fails with [`Error::NotReal`] if `T` can't represent the operation.
    pub(super) fn new(op: &Operation) -> Result<Self, Error> {
        let not_real = || {
            Error::NotReal(format!(
                "{op:?} has complex entries; use a complex statevector"
            ))
        };
        Ok(match op {
            Operation::Gate { gate, target } => Prepared::Pair {
                op: PairOp::new(&gate.matrix()).ok_or_else(not_real)?,
                target: *target,
            },
            Operation::Controlled {
                gate,
                control,
                target,
            } => Prepared::Controlled {
                op: PairOp::new(&gate.matrix()).ok_or_else(not_real)?,
                control: *control,
                target: *target,
            },
            Operation::Swap(a, b) => Prepared::Swap(*a, *b),
            Operation::Unitary2 { matrix, qubits } => Prepared::Quad {
                op: QuadOp::new(matrix).ok_or_else(not_real)?,
                q0: qubits[0],
                q1: qubits[1],
            },
            Operation::PauliRotation { pauli, theta } => {
                let masks = pauli.masks();
                Prepared::Pauli {
                    masks,
                    op: RotationOp::new(masks, *theta).ok_or_else(not_real)?,
                }
            }
        })
    }

    /// The qubits acted on, as a bit mask.
    pub(super) fn qubit_mask(&self) -> usize {
        match *self {
            Prepared::Pair { target, .. } => 1 << target,
            Prepared::Controlled {
                control, target, ..
            } => 1 << control | 1 << target,
            Prepared::Swap(a, b) => 1 << a | 1 << b,
            Prepared::Quad { q0, q1, .. } => 1 << q0 | 1 << q1,
            Prepared::Pauli { masks, .. } => masks.x | masks.z,
        }
    }

    /// The same operation on `tile`'s amplitudes, whose qubits must
    /// include all of this one's.
    pub(super) fn localize(&self, tile: &Tile) -> Self {
        let q = |qubit: usize| tile.local(1 << qubit).trailing_zeros() as usize;
        let mut local = self.clone();
        match &mut local {
            Prepared::Pair { target, .. } => *target = q(*target),
            Prepared::Controlled {
                control, target, ..
            } => (*control, *target) = (q(*control), q(*target)),
            Prepared::Swap(a, b) => (*a, *b) = (q(*a), q(*b)),
            Prepared::Quad { q0, q1, .. } => (*q0, *q1) = (q(*q0), q(*q1)),
            Prepared::Pauli { masks, .. } => {
                masks.x = tile.local(masks.x);
                masks.z = tile.local(masks.z);
            }
        }
        local
    }

    /// Apply to `amps`; qubits must be valid for its length.
    pub(super) fn apply(&self, amps: &mut [T]) {
        match self {
            Prepared::Pair { op, target } => {
                for_each_pair(amps, *target, 1, |_, lo, hi| op.apply(lo, hi))
            }
            Prepared::Controlled {
                op,
                control,
                target,
            } => apply_controlled(amps, op, *control, *target),
            Prepared::Swap(a, b) => apply_swap(amps, *a, *b),
            Prepared::Quad { op, q0, q1 } => apply_quad(amps, op, *q0, *q1),
            Prepared::Pauli { masks, op, .. } => apply_pauli_rotation(amps, *masks, op),
        }
    }
}

/// The cheapest way to apply a 2×2 matrix to an amplitude pair.
#[derive(Clone)]
pub(super) enum PairOp<T> {
    General([[T; 2]; 2]),
    /// `diag(d0, d1)`; a factor of exactly 1 is skipped, so phase gates
    /// only touch half the state.
    Diagonal(T, T),
    /// Pauli X: swap the pair.
    Flip,
}

impl<T: Scalar> PairOp<T> {
    fn new(m: &[[C64; 2]; 2]) -> Option<Self> {
        let m = convert_matrix::<T, 2>(m)?;
        Some(if m[0][1] == T::ZERO && m[1][0] == T::ZERO {
            PairOp::Diagonal(m[0][0], m[1][1])
        } else if m[0][0] == T::ZERO && m[1][1] == T::ZERO && m[0][1] == T::ONE && m[1][0] == T::ONE
        {
            PairOp::Flip
        } else {
            PairOp::General(m)
        })
    }

    #[inline]
    fn apply(&self, lo: &mut [T], hi: &mut [T]) {
        match *self {
            PairOp::General(m) => {
                for (a, b) in lo.iter_mut().zip(hi) {
                    let (x, y) = (*a, *b);
                    *a = m[0][0] * x + m[0][1] * y;
                    *b = m[1][0] * x + m[1][1] * y;
                }
            }
            PairOp::Diagonal(d0, d1) => {
                scale(lo, d0);
                scale(hi, d1);
            }
            PairOp::Flip => lo.swap_with_slice(hi),
        }
    }
}

/// A 4×4 matrix, with diagonal ones (fused phase gates) kept cheap.
#[derive(Clone)]
pub(super) enum QuadOp<T> {
    General([[T; 4]; 4]),
    Diagonal([T; 4]),
}

impl<T: Scalar> QuadOp<T> {
    fn new(m: &[[C64; 4]; 4]) -> Option<Self> {
        let m = convert_matrix::<T, 4>(m)?;
        let diagonal = (0..4).all(|r| (0..4).all(|c| r == c || m[r][c] == T::ZERO));
        Some(if diagonal {
            QuadOp::Diagonal(std::array::from_fn(|i| m[i][i]))
        } else {
            QuadOp::General(m)
        })
    }
}

/// `exp(-iθ/2 · P)` in the form its kernel needs.
#[derive(Clone)]
pub(super) enum RotationOp<T> {
    /// `P` is diagonal: multiply by `plus` or `minus` by the parity of
    /// `j ∧ z`.
    Diagonal { plus: T, minus: T },
    /// `ψ_j ← cos ψ_j + factor · (±1) ψ_{j⊕x}`, with
    /// `factor = −i sin(θ/2) iⁿʸ`.
    Pairing { cos: T, factor: T },
}

impl<T: Scalar> RotationOp<T> {
    fn new(masks: PauliMasks, theta: f64) -> Option<Self> {
        let (sin, cos) = (theta / 2.0).sin_cos();
        Some(if masks.x == 0 {
            // The y phase is 1 here: a diagonal string has no Y.
            RotationOp::Diagonal {
                plus: T::from_complex(C64::cis(-theta / 2.0))?,
                minus: T::from_complex(C64::cis(theta / 2.0))?,
            }
        } else {
            RotationOp::Pairing {
                cos: T::from_real(cos),
                factor: T::from_complex(C64::new(0.0, -sin) * masks.y_phase())?,
            }
        })
    }
}

/// Apply every operation in order.
pub(super) fn run<T: Scalar>(amps: &mut [T], ops: &[Prepared<T>]) {
    let n_qubits = amps.len().trailing_zeros() as usize;
    let mut rest = ops;
    while let Some(first) = rest.first() {
        let (tile, count) = Tile::cover(
            n_qubits,
            CACHE_BLOCK_QUBITS,
            rest.iter().map(Prepared::qubit_mask),
        );
        if count >= 2 {
            let (batch, tail) = rest.split_at(count);
            let local: Vec<_> = batch.iter().map(|op| op.localize(&tile)).collect();
            tile.for_each([amps], |_, [block]| {
                for op in &local {
                    op.apply(block);
                }
            });
            rest = tail;
        } else {
            first.apply(amps);
            rest = &rest[1..];
        }
    }
}

/// A set of `k` qubits, splitting a state into `2^(n-k)` tiles of `2^k`
/// amplitudes: the amplitudes that agree on every other qubit. Runs of
/// operations on a tile's qubits can be applied one tile at a time, keeping
/// it in cache, so the whole run costs one pass over memory.
///
/// Tiles always include the lowest qubits, so they are gathered in
/// contiguous runs of at least `2^MIN_LOW_QUBITS` amplitudes; when they
/// are exactly the lowest `k`, tiles are contiguous and used in place.
pub(super) struct Tile {
    /// The tile's qubits, as a bit mask.
    qubits: usize,
    n_qubits: usize,
    /// Number of low qubits in every tile, so of amplitudes per run, log 2.
    low: usize,
    /// Offsets of the runs within a tile.
    runs: Vec<usize>,
}

/// The fewest lowest qubits a tile includes.
const MIN_LOW_QUBITS: usize = 4;

/// Below this many qubits the state mostly fits in the last-level cache,
/// where gathering tiles costs more than it saves, so tiles are only ever
/// the lowest qubits.
const GATHER_MIN_QUBITS: usize = 20;

/// States too small for per-gate parallelism are cut into `2^SPLIT_QUBITS`
/// tiles, enough to keep every core busy...
const SPLIT_QUBITS: usize = 4;

/// ...from this many qubits; smaller states are cheaper on one thread.
const SPLIT_MIN_QUBITS: usize = 13;

impl Tile {
    /// The tile for the longest prefix of operations, given by their qubit
    /// masks, that fit in one, and the length of that prefix. Unused places
    /// go to the lowest remaining qubits.
    ///
    /// Large states use tiles of `cache_qubits`, sized so a tile of every
    /// vector involved fits in cache; below `GATHER_MIN_QUBITS` these are
    /// only ever the lowest qubits. States below `PARALLEL_THRESHOLD`
    /// amplitudes fit in cache already, but would run every gate on one
    /// thread, so they are cut into `2^SPLIT_QUBITS` tiles that run in
    /// parallel. States of fewer than `SPLIT_MIN_QUBITS` give count 0.
    pub(super) fn cover(
        n_qubits: usize,
        cache_qubits: usize,
        masks: impl IntoIterator<Item = usize>,
    ) -> (Tile, usize) {
        let large = 1 << n_qubits >= PARALLEL_THRESHOLD;
        let k = cache_qubits.min(n_qubits.saturating_sub(SPLIT_QUBITS));
        if n_qubits <= k || (!large && n_qubits < SPLIT_MIN_QUBITS) {
            return (Tile::new((1 << n_qubits) - 1, n_qubits), 0);
        }
        let mut qubits = if large && n_qubits < GATHER_MIN_QUBITS {
            (1 << k) - 1
        } else {
            (1 << MIN_LOW_QUBITS) - 1
        };
        let mut count = 0;
        for mask in masks {
            if (qubits | mask).count_ones() as usize > k {
                break;
            }
            qubits |= mask;
            count += 1;
        }
        let mut fill = 0;
        while (qubits.count_ones() as usize) < k {
            qubits |= 1 << fill;
            fill += 1;
        }
        (Tile::new(qubits, n_qubits), count)
    }

    fn new(qubits: usize, n_qubits: usize) -> Tile {
        let low = qubits.trailing_ones() as usize;
        let k = qubits.count_ones() as usize;
        let runs = (0..1 << (k - low))
            .map(|r| deposit(r << low, qubits))
            .collect();
        Tile {
            qubits,
            n_qubits,
            low,
            runs,
        }
    }

    /// The tile's size, log 2.
    pub(super) fn k(&self) -> usize {
        self.qubits.count_ones() as usize
    }

    /// The tile's qubits, as a bit mask.
    pub(super) fn qubits(&self) -> usize {
        self.qubits
    }

    /// A mask of qubits in the tile, in tile-local numbering.
    pub(super) fn local(&self, mask: usize) -> usize {
        extract(mask, self.qubits)
    }

    /// Call `f(base, tiles)` on every tile of the `N` states `vectors`, in
    /// parallel, where `base` is the global index of each tile's first
    /// amplitude (its bits are all outside the tile) and `tiles` the
    /// tile's amplitudes in each state, ordered by local index.
    pub(super) fn for_each<T: Scalar, const N: usize>(
        &self,
        vectors: [&mut [T]; N],
        f: impl Fn(usize, [&mut [T]; N]) + Sync,
    ) {
        self.map_reduce(vectors, |base, tiles| f(base, tiles), || (), |(), ()| ());
    }

    /// [`for_each`](Self::for_each), combining the results of `f` with
    /// `reduce`.
    pub(super) fn map_reduce<T: Scalar, const N: usize, R: Send>(
        &self,
        vectors: [&mut [T]; N],
        f: impl Fn(usize, [&mut [T]; N]) -> R + Sync,
        identity: impl Fn() -> R + Sync + Send,
        reduce: impl Fn(R, R) -> R + Sync + Send,
    ) -> R {
        for v in &vectors {
            assert_eq!(v.len(), 1 << self.n_qubits);
        }
        let k = self.k();
        let outside = ((1 << self.n_qubits) - 1) & !self.qubits;
        let pointers = vectors.map(|v| SharedMut(v.as_mut_ptr()));
        let contiguous = self.low == k;
        let run = 1 << self.low;
        (0..1usize << (self.n_qubits - k))
            .into_par_iter()
            .map_init(
                || {
                    let size = if contiguous { 0 } else { 1 << k };
                    std::array::from_fn::<_, N, _>(|_| vec![T::ZERO; size])
                },
                |buffers, o| {
                    let base = deposit(o, outside);
                    // SAFETY: tiles are disjoint (their indices differ in
                    // the bits outside the tile, set only by `base`), each
                    // is handled by one task, and every index is below the
                    // length asserted above. The vectors are borrowed
                    // mutably for the whole call.
                    if contiguous {
                        let tiles = std::array::from_fn(|v| unsafe {
                            std::slice::from_raw_parts_mut(pointers[v].0.add(base), 1 << k)
                        });
                        return f(base, tiles);
                    }
                    for (buffer, pointer) in buffers.iter_mut().zip(&pointers) {
                        for (chunk, &offset) in buffer.chunks_exact_mut(run).zip(&self.runs) {
                            let src = unsafe {
                                std::slice::from_raw_parts(pointer.0.add(base + offset), run)
                            };
                            chunk.copy_from_slice(src);
                        }
                    }
                    let mut iter = buffers.iter_mut();
                    let tiles = std::array::from_fn(|_| iter.next().unwrap().as_mut_slice());
                    let result = f(base, tiles);
                    for (buffer, pointer) in buffers.iter().zip(&pointers) {
                        for (chunk, &offset) in buffer.chunks_exact(run).zip(&self.runs) {
                            let dst = unsafe {
                                std::slice::from_raw_parts_mut(pointer.0.add(base + offset), run)
                            };
                            dst.copy_from_slice(chunk);
                        }
                    }
                    result
                },
            )
            .reduce(identity, reduce)
    }
}

/// A pointer shared between tasks writing disjoint parts of its target.
struct SharedMut<T>(*mut T);

// SAFETY: only used by `Tile::map_reduce`, whose tasks touch disjoint
// elements.
unsafe impl<T: Send> Sync for SharedMut<T> {}
unsafe impl<T: Send> Send for SharedMut<T> {}

/// Scatter the low bits of `value` into the set bits of `mask`, in order
/// (`pdep`).
fn deposit(mut value: usize, mut mask: usize) -> usize {
    let mut out = 0;
    while mask != 0 {
        let bit = mask & mask.wrapping_neg();
        if value & 1 == 1 {
            out |= bit;
        }
        value >>= 1;
        mask &= mask - 1;
    }
    out
}

/// Gather the bits of `value` at the set bits of `mask` into the low bits
/// (`pext`).
fn extract(value: usize, mut mask: usize) -> usize {
    let mut out = 0;
    let mut k = 0;
    while mask != 0 {
        let bit = mask & mask.wrapping_neg();
        if value & bit != 0 {
            out |= 1 << k;
        }
        k += 1;
        mask &= mask - 1;
    }
    out
}

/// Multiply every amplitude by `f`, skipping `f = 1`.
#[inline]
fn scale<T: Scalar>(s: &mut [T], f: T) {
    if f != T::ONE {
        s.iter_mut().for_each(|a| *a *= f);
    }
}

/// Call `f(base, lo, hi)` on matching slices that together cover every
/// pair of amplitudes differing only in `bit`. `base` is the index of
/// `lo[0]`; `hi[k]` has index `base + k + 2^bit`.
///
/// Slice lengths are multiples of `min_len` and start at multiples of it in
/// index space, so callers can pair on a lower bit inside them. `min_len`
/// must be a power of two no larger than `2^bit`.
fn for_each_pair<T, F>(amps: &mut [T], bit: usize, min_len: usize, f: F)
where
    T: Scalar,
    F: Fn(usize, &mut [T], &mut [T]) + Sync,
{
    let stride = 1 << bit;
    let parallel = amps.len() >= PARALLEL_THRESHOLD;
    if 2 * stride <= BLOCK {
        // Many small pairs: hand whole blocks to each task.
        let run = |offset: usize, block: &mut [T]| {
            for (c, chunk) in block.chunks_exact_mut(2 * stride).enumerate() {
                let (lo, hi) = chunk.split_at_mut(stride);
                f(offset + c * 2 * stride, lo, hi);
            }
        };
        if parallel {
            amps.par_chunks_mut(BLOCK)
                .enumerate()
                .for_each(|(i, block)| run(i * BLOCK, block));
        } else {
            run(0, amps);
        }
    } else if parallel {
        // Large halves: split across chunks and within each half.
        let task = (BLOCK / 2).max(min_len);
        amps.par_chunks_mut(2 * stride)
            .enumerate()
            .for_each(|(c, chunk)| {
                let (lo, hi) = chunk.split_at_mut(stride);
                lo.par_chunks_mut(task)
                    .zip(hi.par_chunks_mut(task))
                    .enumerate()
                    .for_each(|(t, (l, h))| f(c * 2 * stride + t * task, l, h));
            });
    } else {
        for (c, chunk) in amps.chunks_exact_mut(2 * stride).enumerate() {
            let (lo, hi) = chunk.split_at_mut(stride);
            f(c * 2 * stride, lo, hi);
        }
    }
}

/// Call `f(s00, s01, s10, s11)` on matching slices covering every group of
/// four amplitudes that differ only in bits `low < high`. In `s_hl`, `h` is
/// the value of bit `high` and `l` the value of bit `low`.
fn for_each_quad<T, F>(amps: &mut [T], low: usize, high: usize, f: F)
where
    T: Scalar,
    F: Fn(&mut [T], &mut [T], &mut [T], &mut [T]) + Sync,
{
    let half = 1 << low;
    let parallel = amps.len() >= PARALLEL_THRESHOLD;
    for_each_pair(amps, high, 2 * half, |_, lo, hi| {
        for (l, h) in lo
            .chunks_exact_mut(2 * half)
            .zip(hi.chunks_exact_mut(2 * half))
        {
            let (s00, s01) = l.split_at_mut(half);
            let (s10, s11) = h.split_at_mut(half);
            if parallel && half > BLOCK / 2 {
                // `low` is high too, so each quarter is large: split it.
                let task = BLOCK / 2;
                s00.par_chunks_mut(task)
                    .zip(s01.par_chunks_mut(task))
                    .zip(s10.par_chunks_mut(task))
                    .zip(s11.par_chunks_mut(task))
                    .for_each(|(((a, b), c), d)| f(a, b, c, d));
            } else {
                f(s00, s01, s10, s11);
            }
        }
    });
}

/// Apply `op` to `target` where `control` is set.
fn apply_controlled<T: Scalar>(amps: &mut [T], op: &PairOp<T>, control: usize, target: usize) {
    let (low, high) = (control.min(target), control.max(target));
    let control_is_high = control > target;
    for_each_quad(amps, low, high, |_, s01, s10, s11| {
        if control_is_high {
            op.apply(s10, s11);
        } else {
            op.apply(s01, s11);
        }
    });
}

fn apply_swap<T: Scalar>(amps: &mut [T], a: usize, b: usize) {
    let (low, high) = (a.min(b), a.max(b));
    for_each_quad(amps, low, high, |_, s01, s10, _| s01.swap_with_slice(s10));
}

fn apply_quad<T: Scalar>(amps: &mut [T], op: &QuadOp<T>, q0: usize, q1: usize) {
    let (low, high) = (q0.min(q1), q0.max(q1));
    let q0_is_low = q0 < q1;
    match op {
        QuadOp::Diagonal(d) => {
            let mut d = *d;
            if !q0_is_low {
                d.swap(1, 2);
            }
            for_each_quad(amps, low, high, |s00, s01, s10, s11| {
                scale(s00, d[0]);
                scale(s01, d[1]);
                scale(s10, d[2]);
                scale(s11, d[3]);
            });
        }
        // Order the quarters by matrix basis index.
        QuadOp::General(m) => for_each_quad(amps, low, high, |s00, s01, s10, s11| {
            if q0_is_low {
                mat4_apply(m, s00, s01, s10, s11);
            } else {
                mat4_apply(m, s00, s10, s01, s11);
            }
        }),
    }
}

/// Apply `m` to each quadruple `(s0[j], s1[j], s2[j], s3[j])`. Written out
/// in full so the compiler keeps everything in registers.
#[inline]
fn mat4_apply<T: Scalar>(m: &[[T; 4]; 4], s0: &mut [T], s1: &mut [T], s2: &mut [T], s3: &mut [T]) {
    let quads = s0.iter_mut().zip(s1.iter_mut()).zip(s2.iter_mut()).zip(s3);
    for (((a0, a1), a2), a3) in quads {
        let (v0, v1, v2, v3) = (*a0, *a1, *a2, *a3);
        *a0 = m[0][0] * v0 + m[0][1] * v1 + m[0][2] * v2 + m[0][3] * v3;
        *a1 = m[1][0] * v0 + m[1][1] * v1 + m[1][2] * v2 + m[1][3] * v3;
        *a2 = m[2][0] * v0 + m[2][1] * v1 + m[2][2] * v2 + m[2][3] * v3;
        *a3 = m[3][0] * v0 + m[3][1] * v1 + m[3][2] * v2 + m[3][3] * v3;
    }
}

/// `exp(-iθ/2 · P)` in one pass. With `P|j⟩ = s(j)|j ⊕ x⟩`, each pair
/// `(j, j' = j ⊕ x)` maps to `ψ_j ← cos ψ_j − i sin s(j') ψ_j'` and
/// `ψ_j' ← cos ψ_j' − i sin s(j) ψ_j`.
fn apply_pauli_rotation<T: Scalar>(amps: &mut [T], masks: PauliMasks, op: &RotationOp<T>) {
    match *op {
        RotationOp::Diagonal { plus, minus } => {
            let apply = |offset: usize, chunk: &mut [T]| {
                for (k, a) in chunk.iter_mut().enumerate() {
                    *a *= if masks.z_sign(offset + k) {
                        minus
                    } else {
                        plus
                    };
                }
            };
            if amps.len() >= PARALLEL_THRESHOLD {
                amps.par_chunks_mut(BLOCK)
                    .enumerate()
                    .for_each(|(i, chunk)| apply(i * BLOCK, chunk));
            } else {
                apply(0, amps);
            }
        }
        RotationOp::Pairing { cos, factor } => {
            // Pair on the highest x bit; the partner of lo[k] is hi[k ⊕ x_low].
            let high = usize::BITS as usize - 1 - masks.x.leading_zeros() as usize;
            let x_low = masks.x & !(1 << high);
            let min_len = (x_low + 1).next_power_of_two();
            let signed = |negative: bool| if negative { -factor } else { factor };
            for_each_pair(amps, high, min_len, |base, lo, hi| {
                let hi_base = base + (1 << high);
                for (k, a) in lo.iter_mut().enumerate() {
                    let kp = k ^ x_low;
                    let (x, y) = (*a, hi[kp]);
                    *a = cos * x + signed(masks.z_sign(hi_base + kp)) * y;
                    hi[kp] = cos * y + signed(masks.z_sign(base + k)) * x;
                }
            });
        }
    }
}
