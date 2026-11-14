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
use super::linalg::{matmul, qr_r, thin_svd};
use super::mpo::Mpo;
use super::{Mps, Tensor};
use faer::{Mat, MatRef};
use num_complex::Complex64 as C64;
use rayon::prelude::*;

/// Directions of `H|ψ⟩` whose singular value is below this, relative to
/// the norm of all its channels at the bond, are not added.
const EXPAND_TOLERANCE: f64 = 1e-10;

impl Mps {
    /// Enlarge every bond's right basis with the directions of `H|ψ⟩`,
    /// leaving the state the same and right-canonical with the centre at
    /// site 0. No bond grows past `max_bond`.
    pub(super) fn expand_right_bases(&mut self, mpo: &Mpo) {
        let n = self.n_qubits();
        self.move_center(0);
        let max_bond = self.options.max_bond;
        // Mixed environment of the sites to the right: rows on the expanded
        // basis, columns on ψ's.
        let mut env = Env::boundary();
        // Columns added to the right bond of the site to the left.
        let mut added = 0;
        for i in (1..n).rev() {
            let ket = &self.tensors[i];
            let (l, r_old) = (ket.left, ket.right);
            let r = r_old + added;
            // ψ's site in the expanded right basis: the new directions come
            // after ψ's own and carry no weight.
            let padded = pad_right(ket, r);
            // Hψ's channels at this bond, each a set of rows over `(p, r)`.
            let y = right_operator(&env, ket, &mpo.sites[i]);
            let mut channels = Mat::<C64>::zeros(y.len() * l, 2 * r);
            for (a, ya) in y.iter().enumerate() {
                channels
                    .as_mut()
                    .submatrix_mut(a * l, 0, l, 2 * r)
                    .copy_from(ya);
            }
            let basis = padded.as_right_matrix();
            let room = useful(i, n, max_bond).min(2 * r).saturating_sub(l);
            let new_rows = complement(channels.as_ref(), basis, room);
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
            env = Env {
                blocks: y
                    .par_iter()
                    .map(|ya| matmul(expanded.as_right_matrix().conjugate(), ya.transpose()))
                    .collect(),
            };
            self.tensors[i] = expanded;
            added = k;
        }
        let first = &self.tensors[0];
        self.tensors[0] = pad_right(first, first.right + added);
        self.center = 0;
    }

    /// The mirror of [`expand_right_bases`](Mps::expand_right_bases):
    /// enlarge every bond's left basis with the directions of `H|ψ⟩`,
    /// leaving the state the same and left-canonical with the centre at the
    /// last site.
    pub(super) fn expand_left_bases(&mut self, mpo: &Mpo) {
        let n = self.n_qubits();
        self.move_center(n - 1);
        let max_bond = self.options.max_bond;
        let mut env = Env::boundary();
        let mut added = 0;
        for i in 0..n - 1 {
            let ket = &self.tensors[i];
            let (l_old, r) = (ket.left, ket.right);
            let l = l_old + added;
            let padded = pad_left(ket, l);
            // Hψ's channels at this bond, each a set of columns over `(l, p)`.
            let y = left_operator(&env, ket, &mpo.sites[i]);
            let mut channels = Mat::<C64>::zeros(y.len() * r, 2 * l);
            for (b, yb) in y.iter().enumerate() {
                channels
                    .as_mut()
                    .submatrix_mut(b * r, 0, r, 2 * l)
                    .copy_from(yb.transpose());
            }
            // Transposed, ψ's orthonormal columns are orthonormal rows.
            let basis = padded.as_left_matrix().transpose();
            let room = useful(i + 1, n, max_bond).min(2 * l).saturating_sub(r);
            let new_cols = complement(channels.as_ref(), basis, room);
            let k = new_cols.nrows();
            let expanded = Mat::<C64>::from_fn(2 * l, r + k, |row, col| {
                if col < r {
                    padded.data[row * r + col]
                } else {
                    new_cols[(col - r, row)]
                }
            });
            let expanded = Tensor::from_matrix(expanded.as_ref(), l, r + k);
            env = Env {
                blocks: y
                    .par_iter()
                    .map(|yb| matmul(expanded.as_left_matrix().adjoint(), yb.as_ref()))
                    .collect(),
            };
            self.tensors[i] = expanded;
            added = k;
        }
        let last = &self.tensors[n - 1];
        self.tensors[n - 1] = pad_left(last, last.left + added);
        self.center = n - 1;
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
fn complement(rows: MatRef<'_, C64>, basis: MatRef<'_, C64>, room: usize) -> Mat<C64> {
    let cols = rows.ncols();
    if room == 0 || rows.nrows() == 0 {
        return Mat::zeros(0, cols);
    }
    let project_out = |m: MatRef<'_, C64>| -> Mat<C64> {
        let overlap = matmul(m, basis.adjoint());
        let mut out = m.to_owned();
        out -= matmul(overlap.as_ref(), basis);
        out
    };
    let scale = (0..rows.nrows())
        .flat_map(|r| (0..cols).map(move |c| rows[(r, c)].norm_sqr()))
        .sum::<f64>()
        .sqrt();
    if scale == 0.0 {
        return Mat::zeros(0, cols);
    }
    // There are usually many more rows (one set per MPO channel) than
    // columns. `rows = Q R` with `Q`'s columns orthonormal, so `R` has the
    // same row space and, after any projection on the right, the same
    // singular values and right singular vectors, in far fewer rows.
    let compact = if rows.nrows() > cols {
        qr_r(rows)
    } else {
        rows.to_owned()
    };
    // Twice, as in classical Gram–Schmidt, for orthogonality to rounding.
    let rest = project_out(project_out(compact.as_ref()).as_ref());
    let (_, s, v) = thin_svd(rest.as_ref());
    let keep = s
        .iter()
        .take_while(|&&x| x > EXPAND_TOLERANCE * scale)
        .count()
        .min(room);
    let new = Mat::<C64>::from_fn(keep, cols, |r, c| v[(c, r)].conj());
    // Re-orthogonalise the weakest directions, which carry the most
    // rounding from the projection.
    let new = project_out(new.as_ref());
    let (_, s, v) = thin_svd(new.as_ref());
    let keep = s.iter().take_while(|&&x| x > 0.5).count();
    Mat::<C64>::from_fn(keep, cols, |r, c| v[(c, r)].conj())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mps::MpsOptions;
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

        mps.expand_right_bases(&mpo);
        assert!(same(&mps));
        let bonds = mps.bond_dimensions();
        // Every bond grows, up to its largest useful size.
        for (i, &d) in bonds.iter().enumerate() {
            assert!(d > 1 && d <= useful(i + 1, n, 1024), "{bonds:?}");
        }
        for t in &mps.tensors[1..] {
            assert_orthonormal(t.as_right_matrix());
        }

        mps.expand_left_bases(&mpo);
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
        mps.expand_right_bases(&mpo);
        mps.expand_left_bases(&mpo);
        assert!(mps.bond_dimensions().iter().all(|&d| d <= 3));
        assert!((mps.norm_sqr() - 1.0).abs() < 1e-14);
    }
}
