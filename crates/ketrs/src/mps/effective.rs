//! Effective Hamiltonians on the CPU: one or two sites between a left and a
//! right environment.
//!
//! For `x` with rows on the left bond `l` and columns `(p, r)`,
//! `out[(l, p), r] = Σ L_a[l, l'] W[a, c](p, p') x[l', p', r'] R_c[r, r']`.
//! Output rows `l` need only rows `l` of every left block, so the work
//! splits into chunks of left-bond rows, one thread each, with nothing to
//! combine afterwards. Each chunk takes three steps:
//!
//! 1. `y[(l, a), (p', r')] = Σ_l' L_a[l, l'] x[l', (p', r')]`, one product
//!    with the left blocks interleaved by row;
//! 2. `z[(l, p), (c, r')] = Σ W[a, c](p, p') y[(l, a), (p', r')]`, the
//!    sparse MPO step, along contiguous rows of length `r`;
//! 3. `out[(l, p), r] = Σ_(c, r') z[(l, p), (c, r')] R_c[r, r']`, one
//!    product with the transposed right blocks stacked.

use super::dmrg::Env;
use super::mpo::{MpoSite, Rows, pair_keyed, right_keyed};
use crate::statevector::krylov::LinearOperator;
use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, MatMut, MatRef, Par};
use num_complex::Complex64 as C64;
use rayon::prelude::*;
use std::cell::RefCell;

const ZERO: C64 = C64::new(0.0, 0.0);
const ONE: C64 = C64::new(1.0, 0.0);

/// Left-bond rows per chunk. Fixed, so results don't depend on the number
/// of threads.
const ROWS: usize = 4;

/// The effective Hamiltonian of one site (`phys = 2`) or two (`phys = 4`).
pub(super) struct Effective {
    /// Row-major `(l · left_blocks) × l`: row `l · left_blocks + a` is row
    /// `l` of left block `a`.
    left: Vec<C64>,
    /// `(right_blocks · r) × r`: rows `c · r .. (c + 1) · r` hold `R_cᵀ`.
    right: Mat<C64>,
    /// The MPO step, keyed by `c · phys + p`.
    rows: Rows,
    left_blocks: usize,
    right_blocks: usize,
    l: usize,
    r: usize,
    phys: usize,
}

impl Effective {
    /// Sites `i` and `i + 1` with MPO sites `w1` and `w2`, between `left`
    /// (bond `l`) and `right` (bond `r`).
    pub(super) fn two_site(
        left: &Env,
        right: &Env,
        w1: &MpoSite,
        w2: &MpoSite,
        l: usize,
        r: usize,
    ) -> Effective {
        Effective::new(left, right, pair_keyed(w1, w2), l, r, 4)
    }

    /// One site with MPO site `w`.
    pub(super) fn one_site(left: &Env, right: &Env, w: &MpoSite, l: usize, r: usize) -> Effective {
        Effective::new(left, right, right_keyed(w), l, r, 2)
    }

    fn new(left: &Env, right: &Env, rows: Rows, l: usize, r: usize, phys: usize) -> Effective {
        let (left_blocks, right_blocks) = (left.blocks.len(), right.blocks.len());
        debug_assert_eq!(rows.len(), right_blocks * phys);
        let mut stacked = Vec::with_capacity(l * left_blocks * l);
        for row in 0..l {
            for block in &left.blocks {
                stacked.extend((0..l).map(|col| block[(row, col)]));
            }
        }
        let right = Mat::<C64>::from_fn(right_blocks * r, r, |row, col| {
            right.blocks[row / r][(col, row % r)]
        });
        Effective {
            left: stacked,
            right,
            rows,
            left_blocks,
            right_blocks,
            l,
            r,
            phys,
        }
    }

    /// Output rows for left-bond rows `first ..` into `out`, with `y` and
    /// `z` as scratch.
    fn chunk(&self, first: usize, x: MatRef<'_, C64>, out: &mut [C64], scratch: &mut Scratch) {
        let (r, phys, lb, rb) = (self.r, self.phys, self.left_blocks, self.right_blocks);
        let m = out.len() / (phys * r);
        let (y, z) = (&mut scratch.y, &mut scratch.z);

        y.resize(m * lb * phys * r, ZERO);
        let left = &self.left[first * lb * self.l..(first + m) * lb * self.l];
        matmul(
            MatMut::from_row_major_slice_mut(y, m * lb, phys * r),
            Accum::Replace,
            MatRef::from_row_major_slice(left, m * lb, self.l),
            x,
            ONE,
            Par::Seq,
        );

        z.clear();
        z.resize(m * phys * rb * r, ZERO);
        for row in 0..m {
            for (key, terms) in self.rows.iter().enumerate() {
                let (c, p) = (key / phys, key % phys);
                let z_row = &mut z[((row * phys + p) * rb + c) * r..][..r];
                for &(a, p_in, f) in terms {
                    let y_row = &y[(row * lb + a) * phys * r + p_in * r..][..r];
                    for (zv, yv) in z_row.iter_mut().zip(y_row) {
                        *zv += f * yv;
                    }
                }
            }
        }

        matmul(
            MatMut::from_row_major_slice_mut(out, m * phys, r),
            Accum::Replace,
            MatRef::from_row_major_slice(z, m * phys, rb * r),
            self.right.as_ref(),
            ONE,
            Par::Seq,
        );
    }
}

/// A thread's `y` and `z`, kept between applications.
#[derive(Default)]
struct Scratch {
    y: Vec<C64>,
    z: Vec<C64>,
}

thread_local! {
    static SCRATCH: RefCell<Scratch> = RefCell::default();
}

impl LinearOperator<C64> for Effective {
    fn apply(&self, x: &[C64], out: &mut [C64]) {
        let x = MatRef::from_row_major_slice(x, self.l, self.phys * self.r);
        out.par_chunks_mut(ROWS * self.phys * self.r)
            .enumerate()
            .for_each(|(k, out)| {
                SCRATCH.with_borrow_mut(|scratch| self.chunk(k * ROWS, x, out, scratch))
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mps::mpo::pauli_entry;
    use crate::observable::Pauli;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn random_mat(rows: usize, cols: usize, rng: &mut StdRng) -> Mat<C64> {
        Mat::from_fn(rows, cols, |_, _| {
            C64::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0))
        })
    }

    fn random_env(blocks: usize, dim: usize, rng: &mut StdRng) -> Env {
        Env {
            blocks: (0..blocks).map(|_| random_mat(dim, dim, rng)).collect(),
        }
    }

    fn random_site(left: usize, right: usize, rng: &mut StdRng) -> MpoSite {
        let paulis = [Pauli::I, Pauli::X, Pauli::Y, Pauli::Z];
        let entries = (0..left * right + 3)
            .map(|_| {
                let coeff = C64::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0));
                (
                    rng.random_range(0..left),
                    rng.random_range(0..right),
                    paulis[rng.random_range(0..4)],
                    coeff,
                )
            })
            .collect();
        MpoSite {
            left,
            right,
            entries,
        }
    }

    /// The Hamiltonian entry by entry, summing over every index.
    fn reference(
        left: &Env,
        right: &Env,
        sites: &[&MpoSite],
        x: &[C64],
        l: usize,
        r: usize,
    ) -> Vec<C64> {
        let n = sites.len();
        let phys = 1 << n;
        // Operator strings through the sites: (a, c, P per site, coeff).
        let mut strings: Vec<(usize, usize, Vec<Pauli>, C64)> = sites[0]
            .entries
            .iter()
            .map(|&(a, b, p, c)| (a, b, vec![p], c))
            .collect();
        for site in &sites[1..] {
            strings = strings
                .iter()
                .flat_map(|(a, b, ps, c)| {
                    site.entries
                        .iter()
                        .filter(|e| e.0 == *b)
                        .map(move |&(_, d, p, c2)| {
                            let mut ps = ps.clone();
                            ps.push(p);
                            (*a, d, ps, c * c2)
                        })
                })
                .collect();
        }
        let mut out = vec![ZERO; l * phys * r];
        for (a, c, ps, coeff) in &strings {
            for p_out in 0..phys {
                let mut p_in = 0;
                let mut v = *coeff;
                for (k, &p) in ps.iter().enumerate() {
                    let bit = (p_out >> (n - 1 - k)) & 1;
                    let (b_in, val) = pauli_entry(p, bit);
                    p_in |= b_in << (n - 1 - k);
                    v *= val;
                }
                for lo in 0..l {
                    for ro in 0..r {
                        let mut sum = ZERO;
                        for li in 0..l {
                            for ri in 0..r {
                                sum += left.blocks[*a][(lo, li)]
                                    * x[(li * phys + p_in) * r + ri]
                                    * right.blocks[*c][(ro, ri)];
                            }
                        }
                        out[(lo * phys + p_out) * r + ro] += v * sum;
                    }
                }
            }
        }
        out
    }

    #[test]
    fn matches_the_hamiltonian_summed_term_by_term() {
        let mut rng = StdRng::seed_from_u64(5);
        // Bonds that don't divide into whole chunks, and a boundary.
        for (l, r) in [(1, 3), (7, 5), (9, 1), (6, 10)] {
            let (d0, d1, d2) = (3, 4, 2);
            let left = random_env(d0, l, &mut rng);
            let w1 = random_site(d0, d1, &mut rng);
            let w2 = random_site(d1, d2, &mut rng);
            let right_one = random_env(d1, r, &mut rng);
            let right_two = random_env(d2, r, &mut rng);

            let x: Vec<C64> = random_mat(1, l * 2 * r, &mut rng)
                .row(0)
                .iter()
                .copied()
                .collect();
            let mut out = vec![ZERO; x.len()];
            Effective::one_site(&left, &right_one, &w1, l, r).apply(&x, &mut out);
            let want = reference(&left, &right_one, &[&w1], &x, l, r);
            let diff = out
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).norm())
                .fold(0.0, f64::max);
            assert!(diff < 1e-12, "one site, l {l} r {r}: {diff}");

            let x: Vec<C64> = random_mat(1, l * 4 * r, &mut rng)
                .row(0)
                .iter()
                .copied()
                .collect();
            let mut out = vec![ZERO; x.len()];
            Effective::two_site(&left, &right_two, &w1, &w2, l, r).apply(&x, &mut out);
            let want = reference(&left, &right_two, &[&w1, &w2], &x, l, r);
            let diff = out
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).norm())
                .fold(0.0, f64::max);
            assert!(diff < 1e-12, "two sites, l {l} r {r}: {diff}");
        }
    }
}
