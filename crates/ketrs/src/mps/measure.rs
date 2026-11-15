//! Many expectation values at once ([`Mps::expectations`]).
//!
//! In mixed canonical form with the centre at site `c`, the sites left of
//! `c` contract to the identity from the left and those right of it from
//! the right. A Pauli string on sites `f ..= l` with `l ≥ c` therefore
//! needs only the window from `f` to `l`, started from the identity if
//! `f ≤ c`, or else from the norm environment `L_f` of sites `c .. f`
//! (built once, by one sweep, for every string). Strings with `l < c` are
//! contracted the other way, from the norm environment `R_l` of sites
//! `l + 1 ..= c`. So each string costs only its own span, rather than its
//! distance to the centre.
//!
//! Strings that start at the same place are walked as a tree of their
//! operators, so a shared prefix (several strings of one projector, say) is
//! contracted once; starts and branches run in parallel.
//!
//! For a symmetric MPS, a window over operators that keep the charges
//! (`I` and `Z`, or any Pauli on a qubit no charge sees) stays
//! block-diagonal in the bond's sectors, and each step is a set of block
//! products.

use super::blocks::SiteBlocks;
use super::charges::Sectors;
use super::linalg::matmul;
use super::{Mps, Tensor, pauli_matrix, transfer};
use crate::error::Error;
use crate::observable::{Pauli, PauliSum};
use faer::{Mat, MatRef};
use num_complex::Complex64 as C64;
use rayon::prelude::*;

const ZERO: C64 = C64::new(0.0, 0.0);

/// Start positions whose windows are contracted together: bounds the
/// windows held at once.
const BATCH: usize = 64;

/// A partial contraction `⟨ψ| … |ψ⟩` up to a bond, rows on the bra.
#[derive(Clone)]
enum Window {
    Dense(Mat<C64>),
    /// Block-diagonal in the bond's sectors, one block per sector.
    Blocks(Vec<Mat<C64>>),
}

impl Window {
    fn trace(&self) -> C64 {
        let trace = |m: &Mat<C64>| (0..m.nrows()).map(|i| m[(i, i)]).sum::<C64>();
        match self {
            Window::Dense(m) => trace(m),
            Window::Blocks(blocks) => blocks.iter().map(trace).sum(),
        }
    }

    fn identity(bond: usize, sectors: Option<&Sectors>) -> Window {
        match sectors {
            Some(s) => Window::Blocks(
                (0..s.len())
                    .map(|k| Mat::identity(s.size(k), s.size(k)))
                    .collect(),
            ),
            None => Window::Dense(Mat::identity(bond, bond)),
        }
    }

    fn dense(&self, sectors: Option<&Sectors>) -> Mat<C64> {
        match self {
            Window::Dense(m) => m.clone(),
            Window::Blocks(blocks) => {
                let s = sectors.expect("sectors of a block window");
                let mut m = Mat::<C64>::zeros(s.dim(), s.dim());
                for (k, b) in blocks.iter().enumerate() {
                    let at = s.range(k).start;
                    m.as_mut()
                        .submatrix_mut(at, at, b.nrows(), b.ncols())
                        .copy_from(b);
                }
                m
            }
        }
    }
}

/// Which way a window grows.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Direction {
    /// From the left: the next site is to the right.
    Right,
    /// From the right: the next site is to the left.
    Left,
}

/// A window's start and the strings it serves: `(term, operators in the
/// order the window takes its sites)`.
struct Job {
    direction: Direction,
    site: usize,
    window: Window,
    terms: Vec<(usize, Vec<Pauli>)>,
}

impl Mps {
    /// `⟨ψ|O|ψ⟩ / ⟨ψ|ψ⟩` for every observable, as
    /// [`expectation`](Self::expectation) gives it one at a time, but
    /// sharing the work: each Pauli string is contracted only over its own
    /// span, from environments built by one sweep each way from the
    /// orthogonality centre, rather than over its distance to the centre;
    /// strings with a common start and prefix share their contraction; and
    /// strings run in parallel. For many local observables on a long chain
    /// this is orders of magnitude faster than separate calls.
    ///
    /// For an MPS with charges, strings of operators that keep the charges
    /// (`I` and `Z`, and `X` and `Y` on qubits no charge sees) are
    /// contracted block by block.
    pub fn expectations(&self, observables: &[PauliSum]) -> Result<Vec<f64>, Error> {
        let n = self.n_qubits();
        for o in observables {
            if o.min_qubits() > n {
                return Err(Error::QubitOutOfRange {
                    qubit: o.min_qubits() - 1,
                    n_qubits: n,
                });
            }
        }
        let norm = self.norm_sqr();
        let mut constant = vec![0.0; observables.len()];
        // (observable, coefficient, ops on increasing qubits)
        let mut terms = Vec::new();
        for (k, o) in observables.iter().enumerate() {
            for (c, s) in o.terms() {
                if s.weight() == 0 {
                    constant[k] += c;
                } else {
                    terms.push((k, *c, s.ops().to_vec()));
                }
            }
        }

        let c = self.center;
        // Strings by where their windows start: at their first site going
        // right (when they reach the centre), else at their last going left.
        let mut right: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut left: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (t, (_, _, ops)) in terms.iter().enumerate() {
            let (first, last) = (ops[0].0, ops[ops.len() - 1].0);
            if last >= c {
                right[first].push(t);
            } else {
                left[last].push(t);
            }
        }
        let sequence = |t: usize, direction: Direction| -> Vec<Pauli> {
            let ops = &terms[t].2;
            let (first, last) = (ops[0].0, ops[ops.len() - 1].0);
            let mut seq = vec![Pauli::I; last - first + 1];
            for &(q, p) in ops {
                seq[q - first] = p;
            }
            if direction == Direction::Left {
                seq.reverse();
            }
            seq
        };

        let mut values = vec![ZERO; terms.len()];
        let mut jobs: Vec<Job> = Vec::new();
        let run = |jobs: &mut Vec<Job>, values: &mut [C64]| {
            let found: Vec<(usize, C64)> = jobs.par_iter().flat_map(|j| self.walk_job(j)).collect();
            for (t, v) in found {
                values[t] = v;
            }
            jobs.clear();
        };
        let job = |direction: Direction, site: usize, window: Window, ts: &[usize]| Job {
            direction,
            site,
            window,
            terms: ts.iter().map(|&t| (t, sequence(t, direction))).collect(),
        };

        // Rightwards: from the identity up to the centre, then from the
        // norm environment of sites c .. f.
        for (f, ts) in right.iter().enumerate().take(c + 1) {
            if !ts.is_empty() {
                let window = Window::identity(self.tensors[f].left, self.sectors(f));
                jobs.push(job(Direction::Right, f, window, ts));
            }
        }
        let last_right = right.iter().rposition(|ts| !ts.is_empty()).unwrap_or(0);
        let mut env = Window::identity(self.tensors[c].left, self.sectors(c));
        for (f, ts) in right.iter().enumerate().take(last_right + 1).skip(c + 1) {
            env = self.step(&env, f - 1, Pauli::I, Direction::Right);
            if !ts.is_empty() {
                jobs.push(job(Direction::Right, f, env.clone(), ts));
                if jobs.len() >= BATCH {
                    run(&mut jobs, &mut values);
                }
            }
        }
        // Leftwards from the norm environment of sites l + 1 ..= c.
        let first_left = left.iter().position(|ts| !ts.is_empty()).unwrap_or(n);
        if first_left < c {
            let mut env = Window::identity(self.tensors[c].right, self.sectors(c + 1));
            for l in (first_left..c).rev() {
                env = self.step(&env, l + 1, Pauli::I, Direction::Left);
                if !left[l].is_empty() {
                    jobs.push(job(Direction::Left, l, env.clone(), &left[l]));
                    if jobs.len() >= BATCH {
                        run(&mut jobs, &mut values);
                    }
                }
            }
        }
        run(&mut jobs, &mut values);

        let mut out = constant;
        for (&(k, coeff, _), v) in terms.iter().zip(&values) {
            out[k] += coeff * v.re / norm;
        }
        Ok(out)
    }

    /// The sectors of the bond left of site `i` (`i = n`: the right end),
    /// for a symmetric MPS.
    fn sectors(&self, i: usize) -> Option<&Sectors> {
        self.symmetry.as_ref().map(|s| &s.bonds[i])
    }

    /// Every string of a job: walk the tree of their operator sequences.
    fn walk_job(&self, job: &Job) -> Vec<(usize, C64)> {
        let mut order: Vec<usize> = (0..job.terms.len()).collect();
        order.sort_by(|&a, &b| key(&job.terms[a].1).cmp(&key(&job.terms[b].1)));
        let members: Vec<(usize, &[Pauli])> = order
            .iter()
            .map(|&k| (job.terms[k].0, job.terms[k].1.as_slice()))
            .collect();
        self.walk(job.direction, job.site, 0, &job.window, &members)
    }

    /// Strings `members`, sorted, agreeing on their first `depth`
    /// operators, which `window` has taken; `site` is the next site.
    fn walk(
        &self,
        direction: Direction,
        site: usize,
        depth: usize,
        window: &Window,
        members: &[(usize, &[Pauli])],
    ) -> Vec<(usize, C64)> {
        let mut out = Vec::new();
        let mut groups: Vec<&[(usize, &[Pauli])]> = Vec::new();
        let mut start = 0;
        while start < members.len() {
            let (t, seq) = members[start];
            if seq.len() == depth {
                out.push((t, window.trace()));
                start += 1;
                continue;
            }
            let op = seq[depth];
            let end = start
                + members[start..]
                    .iter()
                    .take_while(|(_, s)| s.len() > depth && s[depth] == op)
                    .count();
            groups.push(&members[start..end]);
            start = end;
        }
        let next = |s: usize| match direction {
            Direction::Right => s + 1,
            Direction::Left => s.wrapping_sub(1),
        };
        let deeper: Vec<(usize, C64)> = groups
            .par_iter()
            .flat_map(|group| {
                let op = group[0].1[depth];
                let w = self.step(window, site, op, direction);
                self.walk(direction, next(site), depth + 1, &w, group)
            })
            .collect();
        out.extend(deeper);
        out
    }

    /// `window` extended over site `i` with `op`.
    fn step(&self, window: &Window, i: usize, op: Pauli, direction: Direction) -> Window {
        let t = &self.tensors[i];
        // The bond the window is on now.
        let from = match direction {
            Direction::Right => i,
            Direction::Left => i + 1,
        };
        if let (Window::Blocks(blocks), Some(sym)) = (window, &self.symmetry)
            && (matches!(op, Pauli::I | Pauli::Z) || !sym.charges.distinguishes(i))
        {
            return Window::Blocks(block_step(blocks, t, sym, i, op, direction));
        }
        let m = window.dense(self.sectors(from));
        Window::Dense(match direction {
            Direction::Right => transfer(&m, t, op),
            Direction::Left => transfer_left(&m, t, op),
        })
    }
}

/// An operator sequence as sortable bytes.
fn key(seq: &[Pauli]) -> Vec<u8> {
    seq.iter()
        .map(|p| match p {
            Pauli::I => 0,
            Pauli::X => 1,
            Pauli::Y => 2,
            Pauli::Z => 3,
        })
        .collect()
}

/// `Σ O[p][p'] conj(A_p) R A_p'ᵀ`: the right-to-left mirror of
/// [`transfer`], `A_p` the `l × r` slice of `t` at physical index `p`.
// `p` and `p2` index both the Pauli matrix and the tensor's slices.
#[allow(clippy::needless_range_loop)]
fn transfer_left(env: &Mat<C64>, t: &Tensor, op: Pauli) -> Mat<C64> {
    let o = pauli_matrix(op);
    let (l, r) = (t.left, t.right);
    let slice = |p: usize| MatRef::from_row_major_slice_with_stride(&t.data[p * r..], l, r, 2 * r);
    let mut out = Mat::<C64>::zeros(l, l);
    for p2 in 0..2 {
        let x = matmul(env.as_ref(), slice(p2).transpose()); // r × l
        for p in 0..2 {
            let f = o[p][p2];
            if f != ZERO {
                out += faer::Scale(f) * matmul(slice(p).conjugate(), x.as_ref());
            }
        }
    }
    out
}

/// [`Mps::step`] for a block-diagonal window and an operator that keeps
/// the charges at site `i`: every block maps a sector to one sector.
// `p` and `p2` index both the Pauli matrix and the site's blocks.
#[allow(clippy::needless_range_loop)]
fn block_step(
    blocks: &[Mat<C64>],
    t: &Tensor,
    sym: &super::symmetric::Symmetry,
    i: usize,
    op: Pauli,
    direction: Direction,
) -> Vec<Mat<C64>> {
    let o = pauli_matrix(op);
    let (left, right) = (&sym.bonds[i], &sym.bonds[i + 1]);
    let site = SiteBlocks::new(t, left, right, &sym.charges, i);
    let mut out: Vec<Mat<C64>> = match direction {
        Direction::Right => (0..right.len())
            .map(|k| Mat::zeros(right.size(k), right.size(k)))
            .collect(),
        Direction::Left => (0..left.len())
            .map(|k| Mat::zeros(left.size(k), left.size(k)))
            .collect(),
    };
    for s in 0..left.len() {
        for p2 in 0..2 {
            let Some((tt, b)) = site.block(s, p2) else {
                continue;
            };
            // The product with the ket block, shared by both bra blocks.
            let x = match direction {
                Direction::Right => matmul(blocks[s].as_ref(), b), // d_s × d_t
                Direction::Left => matmul(blocks[tt].as_ref(), b.transpose()), // d_t × d_s
            };
            for p in 0..2 {
                let f = o[p][p2];
                if f == ZERO {
                    continue;
                }
                let (t1, a) = site.block(s, p).expect("a charge-keeping operator");
                debug_assert_eq!(t1, tt);
                match direction {
                    Direction::Right => out[tt] += faer::Scale(f) * matmul(a.adjoint(), x.as_ref()),
                    Direction::Left => out[s] += faer::Scale(f) * matmul(a.conjugate(), x.as_ref()),
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use crate::circuit::Circuit;
    use crate::mps::{Charge, Mps, MpsOptions};
    use crate::observable::{Pauli, PauliString, PauliSum};

    /// Observables of every shape: single Zs, Z strings with gaps, a
    /// projector-like sum of Z strings, hops (XX + YY), a lone X, and
    /// constants.
    fn observables(n: usize) -> Vec<PauliSum> {
        let z = |qs: &[usize]| PauliString::new(qs.iter().map(|&q| (q, Pauli::Z))).unwrap();
        let mut out = Vec::new();
        for q in 0..n {
            out.push(PauliSum::from(z(&[q])));
        }
        for q in 0..n - 3 {
            let mut p = PauliSum::new();
            p.add(0.25, PauliString::new([]).unwrap());
            p.add(0.5, z(&[q]));
            p.add(-0.25, z(&[q, q + 3]));
            p.add(0.125, z(&[q, q + 1, q + 3]));
            out.push(p);
        }
        for q in 0..n - 1 {
            let mut h = PauliSum::new();
            h.add(
                0.5,
                PauliString::new([(q, Pauli::X), (q + 1, Pauli::X)]).unwrap(),
            );
            h.add(
                0.5,
                PauliString::new([(q, Pauli::Y), (q + 1, Pauli::Y)]).unwrap(),
            );
            out.push(h);
        }
        out.push(PauliSum::from(
            PauliString::new([(2, Pauli::X), (n - 1, Pauli::Z)]).unwrap(),
        ));
        out.push(PauliSum::from(
            PauliString::new([(0, Pauli::Z), (n - 1, Pauli::Y)]).unwrap(),
        ));
        out
    }

    fn state(n: usize) -> Mps {
        let mut c = Circuit::new(n);
        for q in (1..n).step_by(2) {
            c.x(q);
        }
        let mut h = PauliSum::new();
        for q in 0..n - 1 {
            for p in [Pauli::X, Pauli::Y] {
                h.add(
                    0.4 + 0.05 * q as f64,
                    PauliString::new([(q, p), (q + 1, p)]).unwrap(),
                );
            }
            h.add(
                0.3,
                PauliString::new([(q, Pauli::Z), (q + 1, Pauli::Z)]).unwrap(),
            );
        }
        c.pauli_evolution(&h, 0.8, 4, 2).unwrap();
        Mps::from_circuit(&c, MpsOptions::default()).unwrap()
    }

    fn check(mps: &Mps, obs: &[PauliSum]) {
        let got = mps.expectations(obs).unwrap();
        for (o, g) in obs.iter().zip(&got) {
            let want = mps.expectation(o).unwrap();
            assert!((g - want).abs() < 1e-12, "{g} vs {want}");
        }
    }

    #[test]
    fn batched_expectations_match_one_at_a_time() {
        let n = 10;
        let obs = observables(n);
        let mut dense = state(n);
        // The centre at either end and in the middle, and unnormalised.
        for centre in [0, n - 1, 4] {
            dense.move_center(centre);
            check(&dense, &obs);
        }
        dense.tensors[4].data.iter_mut().for_each(|a| *a *= 1.7);
        check(&dense, &obs);

        let mut sym = state(n)
            .with_charges(&[
                Charge::weights(vec![1; n]),
                Charge::modulo(2, vec![[0, 1]; n]),
            ])
            .unwrap();
        for centre in [0, n - 1, 6] {
            sym.move_center(centre);
            check(&sym, &obs);
        }
        // A qubit no charge sees (3, in |+⟩) takes X and Y block by block.
        let n = 6;
        let mut c = Circuit::new(n);
        c.x(1).x(4).h(3);
        let mut h = PauliSum::new();
        for (a, b) in [(0, 1), (1, 2), (4, 5)] {
            for p in [Pauli::X, Pauli::Y] {
                h.add(0.6, PauliString::new([(a, p), (b, p)]).unwrap());
            }
        }
        c.pauli_evolution(&h, 0.7, 3, 2).unwrap();
        let mut sym = Mps::from_circuit(&c, MpsOptions::default())
            .unwrap()
            .with_charges(&[Charge::weights(vec![1, 1, 1, 0, 1, 1])])
            .unwrap();
        let x3 =
            |p: Pauli, q: usize| PauliSum::from(PauliString::new([(3, p), (q, Pauli::Z)]).unwrap());
        for centre in [0, 5, 2] {
            sym.move_center(centre);
            check(&sym, &[x3(Pauli::X, 5), x3(Pauli::Y, 1), x3(Pauli::X, 4)]);
        }
        assert!(dense.expectations(&[]).unwrap().is_empty());
        assert!(
            dense
                .expectations(&[PauliSum::from(
                    PauliString::new([(dense.n_qubits(), Pauli::Z)]).unwrap()
                )])
                .is_err()
        );
    }
}
