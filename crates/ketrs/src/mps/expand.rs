//! Global subspace expansion (after Yang and White, Phys. Rev. B 102,
//! 094315, 2020).
//!
//! Projected methods (TDVP, DMRG) can only move within the bases the bonds
//! already carry, so from a weakly entangled start a coupling that needs a
//! basis state the bonds lack never acts. One example is a four-qubit hop
//! acting on a product state. The expansion adds to each bond's right basis
//! the directions `H|ψ⟩` reaches there, with zero weight, so the state is
//! unchanged and the next projected step can move into them.
//!
//! Sweeping from the right, each MPO channel of `H|ψ⟩` (a partial term still
//! to be completed on the left) is projected onto the already-expanded right
//! basis through a mixed environment. The part orthogonal to ψ's own basis
//! is orthonormalised and appended. The channels are normalised per bond, so
//! their directions survive however far they are from the right end. A
//! global direct sum `ψ ⊕ ε·Hψ` thresholded against ψ's singular values
//! loses them geometrically.

use super::dmrg::{Env, left_operator, right_operator};
use super::envs::Envs;
use super::linalg::{complement_rows, matmul, qr_r, right_svd};
use super::mpo::Mpo;
use super::{Mps, Tensor};
use crate::error::Error;
use faer::{Mat, MatRef};
use num_complex::Complex64 as C64;
use rayon::prelude::*;

/// Directions of `H|ψ⟩` whose singular value is below this, relative to
/// the norm of all its channels at the bond, are not added.
const EXPAND_TOLERANCE: f64 = 1e-10;

impl Mps {
    /// Enlarge every bond's right basis with the directions of `H|ψ⟩`,
    /// leaving the state the same and right-canonical with the centre at
    /// site 0. No bond grows past `max_bond`. With `envs` on a GPU, the
    /// channels and their factorisation are computed there.
    pub(super) fn expand_right_bases(&mut self, mpo: &Mpo, envs: &Envs) -> Result<(), Error> {
        let n = self.n_qubits();
        self.move_center(0);
        let max_bond = self.options.max_bond;
        let mut chain = chain(envs)?;
        // Columns added to the right bond of the site to the left.
        let mut added = 0;
        for i in (1..n).rev() {
            let ket = &self.tensors[i];
            let (l, r) = (ket.left, ket.right + added);
            // ψ's site in the expanded right basis: the new directions come
            // after ψ's own and carry no weight.
            let padded = pad_right(ket, r);
            let room = useful(i, n, max_bond).min(2 * r).saturating_sub(l);
            let new_rows = chain.right(i, ket, mpo, padded.as_right_matrix(), room)?;
            let k = new_rows.nrows();
            let mut data = padded.data;
            for row in 0..k {
                for col in 0..2 * r {
                    data.push(new_rows[(row, col)]);
                }
            }
            let expanded = Tensor {
                left: l + k,
                right: r,
                data,
            };
            chain.advance_right(&expanded)?;
            self.tensors[i] = expanded;
            added = k;
        }
        let first = &self.tensors[0];
        self.tensors[0] = pad_right(first, first.right + added);
        self.center = 0;
        Ok(())
    }

    /// The mirror of [`expand_right_bases`](Mps::expand_right_bases):
    /// enlarge every bond's left basis with the directions of `H|ψ⟩`,
    /// leaving the state the same and left-canonical with the centre at the
    /// last site.
    // The chain needs each site's index as well as its tensor, which is
    // replaced as the loop goes.
    #[allow(clippy::needless_range_loop)]
    pub(super) fn expand_left_bases(&mut self, mpo: &Mpo, envs: &Envs) -> Result<(), Error> {
        let n = self.n_qubits();
        self.move_center(n - 1);
        let max_bond = self.options.max_bond;
        let mut chain = chain(envs)?;
        let mut added = 0;
        for i in 0..n - 1 {
            let ket = &self.tensors[i];
            let (l, r) = (ket.left + added, ket.right);
            let padded = pad_left(ket, l);
            // Transposed, ψ's orthonormal columns are orthonormal rows.
            let basis = padded.as_left_matrix().transpose();
            let room = useful(i + 1, n, max_bond).min(2 * l).saturating_sub(r);
            let new_cols = chain.left(i, ket, mpo, basis, room)?;
            let k = new_cols.nrows();
            let expanded = Mat::<C64>::from_fn(2 * l, r + k, |row, col| {
                if col < r {
                    padded.data[row * r + col]
                } else {
                    new_cols[(col - r, row)]
                }
            });
            let expanded = Tensor::from_matrix(expanded.as_ref(), l, r + k);
            chain.advance_left(&expanded)?;
            self.tensors[i] = expanded;
            added = k;
        }
        let last = &self.tensors[n - 1];
        self.tensors[n - 1] = pad_left(last, last.left + added);
        self.center = n - 1;
        Ok(())
    }
}

/// The mixed environment of an expansion sweep (rows on the expanded basis,
/// columns on ψ's), giving each bond's new directions and moving on to the
/// next site.
pub(super) trait Chain {
    /// Up to `room` new right-basis directions for site `i` (rows over
    /// `(p, r)`, `r` the expanded right bond) from `H|ψ⟩`'s channels there,
    /// outside the orthonormal rows of `basis`.
    fn right(
        &mut self,
        i: usize,
        ket: &Tensor,
        mpo: &Mpo,
        basis: MatRef<'_, C64>,
        room: usize,
    ) -> Result<Mat<C64>, Error>;

    /// Extend the environment over the site just expanded.
    fn advance_right(&mut self, expanded: &Tensor) -> Result<(), Error>;

    /// The mirror of [`right`](Chain::right): new left-basis directions, as
    /// rows over `(l, p)`.
    fn left(
        &mut self,
        i: usize,
        ket: &Tensor,
        mpo: &Mpo,
        basis: MatRef<'_, C64>,
        room: usize,
    ) -> Result<Mat<C64>, Error>;

    fn advance_left(&mut self, expanded: &Tensor) -> Result<(), Error>;
}

fn chain<'a>(envs: &'a Envs) -> Result<Box<dyn Chain + 'a>, Error> {
    match envs {
        Envs::Cpu { .. } | Envs::Blocks { .. } => Ok(Box::new(CpuChain {
            env: Env::boundary(),
            channels: Vec::new(),
        })),
        #[cfg(feature = "gpu")]
        Envs::Gpu(g) => Ok(Box::new(g.chain()?)),
    }
}

/// The chain on the CPU.
struct CpuChain {
    env: Env,
    /// The last site's channels, one matrix per MPO bond index.
    channels: Vec<Mat<C64>>,
}

impl Chain for CpuChain {
    fn right(
        &mut self,
        i: usize,
        ket: &Tensor,
        mpo: &Mpo,
        basis: MatRef<'_, C64>,
        room: usize,
    ) -> Result<Mat<C64>, Error> {
        // Hψ's channels at this bond, each a set of rows over `(p, r)`.
        self.channels = right_operator(&self.env, ket, &mpo.sites[i]);
        let (l, cols) = (ket.left, basis.ncols());
        let mut rows = Mat::<C64>::zeros(self.channels.len() * l, cols);
        for (a, ya) in self.channels.iter().enumerate() {
            rows.as_mut().submatrix_mut(a * l, 0, l, cols).copy_from(ya);
        }
        Ok(complement(rows.as_ref(), basis, room))
    }

    fn advance_right(&mut self, expanded: &Tensor) -> Result<(), Error> {
        self.env = Env {
            blocks: self
                .channels
                .par_iter()
                .map(|ya| matmul(expanded.as_right_matrix().conjugate(), ya.transpose()))
                .collect(),
        };
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
        // Hψ's channels at this bond, each a set of columns over `(l, p)`.
        self.channels = left_operator(&self.env, ket, &mpo.sites[i]);
        let (r, cols) = (ket.right, basis.ncols());
        let mut rows = Mat::<C64>::zeros(self.channels.len() * r, cols);
        for (b, yb) in self.channels.iter().enumerate() {
            rows.as_mut()
                .submatrix_mut(b * r, 0, r, cols)
                .copy_from(yb.transpose());
        }
        Ok(complement(rows.as_ref(), basis, room))
    }

    fn advance_left(&mut self, expanded: &Tensor) -> Result<(), Error> {
        self.env = Env {
            blocks: self
                .channels
                .par_iter()
                .map(|yb| matmul(expanded.as_left_matrix().adjoint(), yb.as_ref()))
                .collect(),
        };
        Ok(())
    }
}

/// The largest useful dimension of the bond left of site `i`, capped at
/// `max_bond`.
fn useful(i: usize, n: usize, max_bond: usize) -> usize {
    (1usize << i.min(n - i).min(30)).min(max_bond)
}

/// `t` with zero rows appended on its left bond, up to `left`.
fn pad_left(t: &Tensor, left: usize) -> Tensor {
    let mut data = t.data.clone();
    data.resize(left * 2 * t.right, C64::new(0.0, 0.0));
    Tensor {
        left,
        right: t.right,
        data,
    }
}

/// `t` with zero columns appended on its right bond, up to `right`.
fn pad_right(t: &Tensor, right: usize) -> Tensor {
    let mut data = vec![C64::new(0.0, 0.0); t.left * 2 * right];
    for lp in 0..t.left * 2 {
        data[lp * right..lp * right + t.right]
            .copy_from_slice(&t.data[lp * t.right..(lp + 1) * t.right]);
    }
    Tensor {
        left: t.left,
        right,
        data,
    }
}

/// Up to `room` orthonormal rows spanning the most of `rows` outside the
/// span of the orthonormal rows of `basis`.
pub(super) fn complement(rows: MatRef<'_, C64>, basis: MatRef<'_, C64>, room: usize) -> Mat<C64> {
    let cols = rows.ncols();
    if room == 0 || rows.nrows() == 0 {
        return Mat::zeros(0, cols);
    }
    let scale = (0..rows.nrows())
        .flat_map(|r| (0..cols).map(move |c| rows[(r, c)].norm_sqr()))
        .sum::<f64>()
        .sqrt();
    if scale == 0.0 {
        return Mat::zeros(0, cols);
    }
    // In an orthonormal basis of the directions outside `basis`, the part of
    // `rows` outside it is `rows · others†`: the same singular values as
    // `rows` projected off `basis`, and right singular vectors that `others`
    // maps back, in `cols − b` columns rather than `cols`. Directions built
    // from `others` are orthonormal and outside `basis` to rounding.
    let others = complement_rows(basis);
    if others.nrows() == 0 {
        return Mat::zeros(0, cols);
    }
    let outside = matmul(rows, others.adjoint());
    // There are usually many more rows (one set per MPO channel) than
    // columns. `outside = Q R` with `Q`'s columns orthonormal, so `R` has the
    // same singular values and right singular vectors in far fewer rows.
    let compact = if outside.nrows() > outside.ncols() {
        qr_r(outside.as_ref())
    } else {
        outside
    };
    directions(compact.as_ref(), scale, others.as_ref(), room)
}

/// The new directions from `compact` (with the singular values and right
/// singular vectors of the channels' part outside the basis, in the
/// coordinates of `others`): its leading right singular vectors above
/// [`EXPAND_TOLERANCE`] of `scale`, at most `room`, mapped back through
/// `others`.
pub(super) fn directions(
    compact: MatRef<'_, C64>,
    scale: f64,
    others: MatRef<'_, C64>,
    room: usize,
) -> Mat<C64> {
    let (s, v) = right_svd(compact);
    let keep = s
        .iter()
        .take_while(|&&x| x > EXPAND_TOLERANCE * scale)
        .count()
        .min(room);
    let lead = Mat::<C64>::from_fn(keep, v.nrows(), |r, c| v[(c, r)].conj());
    matmul(lead.as_ref(), others)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mps::MpsOptions;
    use crate::mps::linalg::thin_svd;
    use crate::observable::{Pauli, PauliString, PauliSum};

    fn assert_orthonormal(m: MatRef<'_, C64>) {
        let g = matmul(m, m.adjoint());
        for r in 0..g.nrows() {
            for c in 0..g.ncols() {
                let want = if r == c { 1.0 } else { 0.0 };
                assert!(
                    (g[(r, c)] - want).norm() < 1e-12,
                    "({r}, {c}): {}",
                    g[(r, c)]
                );
            }
        }
    }

    /// The complement as first written: project `rows` (compressed by QR)
    /// off `basis` twice, take its leading right singular vectors, and
    /// re-orthogonalise them.
    fn reference_complement(
        rows: MatRef<'_, C64>,
        basis: MatRef<'_, C64>,
        room: usize,
    ) -> Mat<C64> {
        let cols = rows.ncols();
        let project_out = |m: MatRef<'_, C64>| -> Mat<C64> {
            let overlap = matmul(m, basis.adjoint());
            let mut out = m.to_owned();
            out -= matmul(overlap.as_ref(), basis);
            out
        };
        let scale = rows.norm_l2();
        let compact = qr_r(rows);
        let rest = project_out(project_out(compact.as_ref()).as_ref());
        let (_, s, v) = thin_svd(rest.as_ref());
        let keep = s
            .iter()
            .take_while(|&&x| x > EXPAND_TOLERANCE * scale)
            .count()
            .min(room);
        let new = Mat::<C64>::from_fn(keep, cols, |r, c| v[(c, r)].conj());
        let new = project_out(new.as_ref());
        let (_, s, v) = thin_svd(new.as_ref());
        let keep = s.iter().take_while(|&&x| x > 0.5).count();
        Mat::<C64>::from_fn(keep, cols, |r, c| v[(c, r)].conj())
    }

    #[test]
    fn complement_matches_projecting_off_the_basis() {
        let mut seed = 3u64;
        let mut random = |m: usize, n: usize| {
            Mat::<C64>::from_fn(m, n, |_, _| {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                C64::new(
                    (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5,
                    (seed >> 23) as f64 / (1u64 << 41) as f64 - 0.5,
                )
            })
        };
        // (channel rows, columns, basis rows, rank of the channels, room)
        for (m, cols, b, rank, room) in [
            (60, 16, 6, 9, 20),
            (200, 24, 10, 24, 5),
            (12, 20, 4, 12, 30),
            (90, 30, 30, 10, 4),
        ] {
            // Rank-deficient channels with spread singular values.
            let left = random(m, rank);
            let right = random(rank, cols);
            let weights = Mat::<C64>::from_fn(rank, rank, |i, j| {
                if i == j {
                    C64::new(10f64.powi(-(i as i32) / 2), 0.0)
                } else {
                    C64::new(0.0, 0.0)
                }
            });
            let rows = matmul(
                matmul(left.as_ref(), weights.as_ref()).as_ref(),
                right.as_ref(),
            );
            let basis = random(cols, b).qr().compute_thin_Q().adjoint().to_owned();
            let got = complement(rows.as_ref(), basis.as_ref(), room);
            let want = reference_complement(rows.as_ref(), basis.as_ref(), room);
            assert_eq!(got.nrows(), want.nrows(), "{m} {cols} {b}");
            if got.nrows() > 0 {
                assert_orthonormal(got.as_ref());
                assert!(matmul(got.as_ref(), basis.adjoint()).norm_max() < 1e-12);
                // The same subspace: equal projectors.
                let (pg, pw) = (
                    matmul(got.adjoint(), got.as_ref()),
                    matmul(want.adjoint(), want.as_ref()),
                );
                assert!((pg - pw).norm_max() < 1e-9, "{m} {cols} {b}");
            }
        }
    }

    #[test]
    fn expansion_keeps_the_state_and_canonical_form() {
        // Four-qubit hops with gaps, and a strong diagonal penalty whose
        // channels dwarf them.
        let n = 12;
        let mut h = PauliSum::new();
        for k in 0..n - 7 {
            for p in [Pauli::X, Pauli::Y] {
                let hop = [(k, p), (k + 3, p), (k + 4, p), (k + 7, p)];
                h.add(0.3, PauliString::new(hop).unwrap());
            }
            h.add(
                50.0,
                PauliString::new([(k, Pauli::Z), (k + 5, Pauli::Z)]).unwrap(),
            );
        }
        let mpo = Mpo::new(&h, n).unwrap();
        let bits: Vec<bool> = (0..n).map(|q| q % 3 == 1).collect();
        let options = MpsOptions {
            max_bond: 1024,
            cutoff: 0.0,
        };
        let mut mps = Mps::basis(n, &bits, options);
        let before = mps.to_amplitudes();
        let same = |mps: &Mps| {
            let after = mps.to_amplitudes();
            before
                .iter()
                .zip(&after)
                .all(|(a, b)| (a - b).norm() < 1e-14)
        };

        let envs = Envs::new(&mps, &mpo).unwrap();
        mps.expand_right_bases(&mpo, &envs).unwrap();
        assert!(same(&mps));
        let bonds = mps.bond_dimensions();
        // Every bond grows, up to its largest useful size.
        for (i, &d) in bonds.iter().enumerate() {
            assert!(d > 1 && d <= useful(i + 1, n, 1024), "{bonds:?}");
        }
        for t in &mps.tensors[1..] {
            assert_orthonormal(t.as_right_matrix());
        }

        mps.expand_left_bases(&mpo, &envs).unwrap();
        assert!(same(&mps));
        for t in &mps.tensors[..n - 1] {
            assert_orthonormal(t.as_left_matrix().transpose());
        }
    }

    #[test]
    fn expansion_respects_max_bond() {
        let n = 10;
        let mut h = PauliSum::new();
        for i in 0..n {
            for j in i + 1..n {
                h.add(
                    1.0,
                    PauliString::new([(i, Pauli::X), (j, Pauli::X)]).unwrap(),
                );
            }
        }
        let mpo = Mpo::new(&h, n).unwrap();
        let options = MpsOptions {
            max_bond: 3,
            cutoff: 0.0,
        };
        let mut mps = Mps::zero(n, options);
        let envs = Envs::new(&mps, &mpo).unwrap();
        mps.expand_right_bases(&mpo, &envs).unwrap();
        mps.expand_left_bases(&mpo, &envs).unwrap();
        assert!(mps.bond_dimensions().iter().all(|&d| d <= 3));
        assert!((mps.norm_sqr() - 1.0).abs() < 1e-14);
    }
}
