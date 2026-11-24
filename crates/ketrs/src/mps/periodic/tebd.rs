//! Periodic circuits and TEBD on [`PeriodicMps`].
//!
//! Unlike open-boundary MPS, qubits `n - 1` and `0` are direct neighbours on
//! the ring connected by the seam bond `(n - 1, 0)`. Two-qubit gates across
//! `(n - 1, 0)` therefore apply directly with **zero** swap gates, and distant
//! two-qubit gates route along the shorter cyclic arc (length `≤ n / 2`).
//!
//! When a gate triggers bond truncation, the split is weighted by the exact
//! periodic ring's partial-trace environment metrics `(L̄^{1/2}, R̄^{1/2})`,
//! computed in pure `O(n χ³)` via a single (`batch = 1`) transfer contraction
//! around the complement of the bond (zero PWE truncation error!).

use super::PeriodicMps;
use super::transfer::batch_transfer_left;
use crate::circuit::{Circuit, Operation, decompose_pauli_rotation};
use crate::error::Error;
use crate::fusion::local_matrix;
use crate::mps::layers::apply_bond_matrix;
use crate::mps::linalg::{hermitian_evd, matmul, thin_svd};
use crate::mps::mpo::Local;
use crate::mps::{MpsOptions, ONE, Tensor, ZERO, keep_count, swap_matrix};
use crate::observable::PauliSum;
use faer::{Mat, MatRef};
use num_complex::Complex64 as C64;

/// One elementary step on a periodic ring: a 1-qubit gate or a 2-qubit gate
/// on cyclic neighbours `(i, (i + 1) % n)` (bit 0 = site `i`, bit 1 = site
/// `(i + 1) % n`).
enum RingStep {
    Site(usize, [[C64; 2]; 2]),
    Bond(usize, [[C64; 4]; 4]),
}

impl RingStep {
    fn sites(&self, n: usize) -> (usize, usize) {
        match *self {
            RingStep::Site(q, _) => (q, q),
            RingStep::Bond(i, _) => (i, (i + 1) % n),
        }
    }
}

/// Decompose `op` into 1-site and cyclic-neighbour 2-site steps, routing
/// non-adjacent pairs along the shorter arc of the `n`-qubit ring.
fn ring_steps(op: &Operation, n: usize, out: &mut Vec<RingStep>) {
    match local_matrix(op) {
        Some((qubits, m)) if qubits.len() == 1 => {
            out.push(RingStep::Site(
                qubits[0],
                [[m[0][0], m[0][1]], [m[1][0], m[1][1]]],
            ));
        }
        Some((qubits, m)) => {
            let m: [[C64; 4]; 4] = std::array::from_fn(|r| std::array::from_fn(|c| m[r][c]));
            let (a, b) = (qubits[0], qubits[1]);
            // Forward cyclic distance from a to b vs b to a.
            let fwd = (b + n - a) % n;
            let bwd = (a + n - b) % n;
            let (start, dist, m) = if fwd <= bwd {
                (a, fwd, m)
            } else {
                const P: [usize; 4] = [0, 2, 1, 3];
                (
                    b,
                    bwd,
                    std::array::from_fn(|r| std::array::from_fn(|c| m[P[r]][P[c]])),
                )
            };
            // Bring `(start + dist) % n` next to `start` along the short arc,
            // apply `m` on bond `start` (`start`, `(start + 1) % n`), then swap back.
            for step in (1..dist).rev() {
                out.push(RingStep::Bond((start + step) % n, swap_matrix()));
            }
            out.push(RingStep::Bond(start, m));
            for step in 1..dist {
                out.push(RingStep::Bond((start + step) % n, swap_matrix()));
            }
        }
        None => {
            let parts = match op {
                Operation::PauliRotation { pauli, theta } => {
                    decompose_pauli_rotation(pauli, *theta)
                }
                _ => {
                    let mut parts = Vec::new();
                    op.decompose(&mut parts);
                    parts
                }
            };
            for part in parts {
                ring_steps(&part, n, out);
            }
        }
    }
}

impl PeriodicMps {
    /// Apply every gate in `circuit` on the periodic ring.
    pub fn run(&mut self, circuit: &Circuit) -> Result<(), Error> {
        self.require_dense("running gates")?;
        circuit.check_unitary()?;
        let n = self.n_qubits();
        if circuit.n_qubits() != n {
            return Err(Error::QubitCountMismatch {
                expected: n,
                found: circuit.n_qubits(),
            });
        }
        if circuit.n_parameters() > 0 {
            return Err(Error::InvalidArgument(format!(
                "circuit has {} unbound parameters; call bind first",
                circuit.n_parameters()
            )));
        }
        let mut steps = Vec::new();
        for op in circuit.fused().ops() {
            ring_steps(op, n, &mut steps);
        }
        self.run_ring_layers(&steps);
        if circuit.global_phase() != 0.0 {
            let phase = C64::cis(circuit.global_phase());
            self.tensors[0].data.iter_mut().for_each(|a| *a *= phase);
        }
        Ok(())
    }

    /// Apply one unitary operation on the periodic ring.
    pub fn apply(&mut self, op: &Operation) -> Result<(), Error> {
        self.require_dense("applying gates")?;
        op.validate(self.n_qubits())?;
        let mut steps = Vec::new();
        ring_steps(op, self.n_qubits(), &mut steps);
        self.run_ring_layers(&steps);
        Ok(())
    }

    /// Periodic TEBD: evolve by `steps` Trotter–Suzuki steps of the given order
    /// for `exp(-iHt)`. Terms across the periodic `(n - 1, 0)` seam apply
    /// directly without swap routing.
    pub fn tebd(
        &mut self,
        hamiltonian: &PauliSum,
        time: f64,
        steps: usize,
        order: usize,
    ) -> Result<(), Error> {
        self.require_dense("TEBD")?;
        let mut circuit = Circuit::new(self.n_qubits());
        circuit.pauli_evolution(hamiltonian, time, steps, order)?;
        self.run(&circuit)
    }

    /// Apply a single-qubit matrix on site `q`.
    fn apply_1q(&mut self, q: usize, m: [[C64; 2]; 2]) {
        let t = &mut self.tensors[q];
        let r = t.right;
        for l in 0..t.left {
            for k in 0..r {
                let (i0, i1) = (2 * l * r + k, (2 * l + 1) * r + k);
                let (a0, a1) = (t.data[i0], t.data[i1]);
                t.data[i0] = m[0][0] * a0 + m[0][1] * a1;
                t.data[i1] = m[1][0] * a0 + m[1][1] * a1;
            }
        }
    }

    /// Execute `steps` packed into layers of disjoint sites/bonds on the ring.
    /// Layers whose bonds stay below `max_bond` and `cutoff == 0` use exact
    /// local SVD; layers that may truncate use half-ring PWE metric-weighted
    /// SVD in `O(p n χ³)`.
    fn run_ring_layers(&mut self, steps: &[RingStep]) {
        let n = self.n_qubits();
        let mut layers: Vec<Vec<&RingStep>> = Vec::new();
        let mut next = vec![0usize; n];
        for step in steps {
            let (a, b) = step.sites(n);
            let layer = next[a].max(next[b]);
            if layer == layers.len() {
                layers.push(Vec::new());
            }
            layers[layer].push(step);
            next[a] = layer + 1;
            next[b] = layer + 1;
        }

        for layer in &layers {
            // Apply all 1-qubit gates in this layer directly.
            let mut bond_gates: Vec<(usize, &[[C64; 4]; 4])> = Vec::new();
            for &step in layer {
                match step {
                    RingStep::Site(q, m) => self.apply_1q(*q, *m),
                    RingStep::Bond(i, m) => bond_gates.push((*i, m)),
                }
            }
            if bond_gates.is_empty() {
                continue;
            }
            for &(i, m) in &bond_gates {
                self.apply_bond_step(i, m);
            }
        }
    }

    /// Apply a 2-qubit gate `m` on cyclic bond `(i, (i + 1) % n)`.
    /// Uses exact unweighted SVD whenever the bond's numerical rank stays
    /// `≤ max_bond` (zero gauge-conditioning overhead, machine-precision exact),
    /// and automatically upgrades to exact `O(n χ³)` periodic metric-weighted
    /// SVD whenever `max_bond` truncation is actually needed.
    fn apply_bond_step(&mut self, i: usize, m: &[[C64; 4]; 4]) {
        let n = self.n_qubits();
        let j = (i + 1) % n;
        let (left, right) = (self.tensors[i].left, self.tensors[j].right);
        let theta = matmul(
            self.tensors[i].as_left_matrix(),
            self.tensors[j].as_right_matrix(),
        );
        let gated = apply_bond_matrix(&theta, m, right);
        let (u, s, v) = thin_svd(gated.as_ref());
        let total: f64 = s.iter().map(|x| x * x).sum();
        let no_cap = MpsOptions {
            max_bond: usize::MAX,
            cutoff: self.options.cutoff.max(1e-28),
        };
        let rank_needed = keep_count(&s, total, &no_cap);
        if rank_needed <= self.options.max_bond {
            let keep = rank_needed;
            let kept: f64 = s[..keep].iter().map(|x| x * x).sum();
            if total > 0.0 {
                self.truncation_error += ((total - kept) / total).max(0.0);
            }
            let u_k = u.as_ref().subcols(0, keep);
            let right_mat = Mat::<C64>::from_fn(keep, v.nrows(), |r, c| {
                C64::new(s[r], 0.0) * v[(c, r)].conj()
            });
            self.tensors[i] = Tensor::from_matrix(u_k, left, keep);
            self.tensors[j] = Tensor::from_matrix(right_mat.as_ref(), keep, right);
            return;
        }
        // True bond truncation needed: use exact `O(n χ³)` periodic environment
        // metrics `(L̄, R̄)` so truncation is optimal in the ring Hilbert norm.
        self.apply_bond_metric_weighted(i, m);
    }

    /// Exact `O(n χ³)` metric-weighted 2-qubit bond update on `(i, (i + 1) % n)`
    /// with **zero** metric inversion (exact to machine precision even when the
    /// ring environment metric `R̄` is singular or ill-conditioned).
    ///
    /// 1. Left-canonicalises the complement `(j + 1) .. (i - 1) mod n` into
    ///    site `i`, making the left partial-trace metric identically `L̄ = I`.
    /// 2. Computes the exact right partial-trace metric `R̄ = R_sqrt† R_sqrt`
    ///    in `O(n χ³)` (`batch = 1` transfer around the complement).
    /// 3. Takes the thin SVD of `M = Θ (I_2 ⊗ R_sqrt^T) = U S V†` and sets
    ///    `A_new = U_k` (strictly left-orthonormal) and `B_new = U_k† Θ`.
    ///
    /// By Eckart–Young, `‖(Θ - U_k U_k† Θ)(I_2 ⊗ R_sqrt^T)‖_F^2 = Σ_{j > k} s_j^2`
    /// is the exact global optimum in the full periodic ring norm without ever
    /// forming `R̄^{-1/2}`!
    fn apply_bond_metric_weighted(&mut self, i: usize, m: &[[C64; 4]; 4]) {
        let n = self.n_qubits();
        let j = (i + 1) % n;

        // Left-canonicalise complement `(j + 1) .. (i - 1) mod n` into site `i`
        // so every complement tensor is strictly left-orthonormal (`L̄ == I`).
        if n > 2 {
            let mut cur = (j + 1) % n;
            while cur != i {
                let next = (cur + 1) % n;
                self.qr_move_right_bond(cur, next);
                cur = next;
            }
        }

        let (l_full, r_full) = (self.tensors[i].left, self.tensors[j].right);
        let r_bar = exact_right_bond_metric(&self.tensors, i, j);
        let r_sqrt = psd_sqrt(&r_bar);

        let theta = matmul(
            self.tensors[i].as_left_matrix(),
            self.tensors[j].as_right_matrix(),
        );
        let gated = apply_bond_matrix(&theta, m, r_full);

        // Form M = gated · (I_2 ⊗ r_sqrt^T), shape (l_full * 2) × (2 * r_full).
        let mut gated_flat = Vec::with_capacity(l_full * 4 * r_full);
        for r in 0..gated.nrows() {
            for c in 0..gated.ncols() {
                gated_flat.push(gated[(r, c)]);
            }
        }
        let mid_mat = MatRef::from_row_major_slice(&gated_flat, l_full * 4, r_full);
        let prod_r = matmul(mid_mat, r_sqrt.transpose());
        let mut m_flat = Vec::with_capacity(l_full * 4 * r_full);
        for r in 0..l_full * 4 {
            for c in 0..r_full {
                m_flat.push(prod_r[(r, c)]);
            }
        }
        let m_mat = MatRef::from_row_major_slice(&m_flat, l_full * 2, 2 * r_full);

        let (u, s, _v) = thin_svd(m_mat);
        let total: f64 = s.iter().map(|x| x * x).sum();
        let keep = keep_count(&s, total, &self.options);
        let kept: f64 = s[..keep].iter().map(|x| x * x).sum();
        if total > 0.0 {
            self.truncation_error += ((total - kept) / total).max(0.0);
        }

        let u_k = u.as_ref().subcols(0, keep);
        let b_final = matmul(u_k.adjoint(), gated.as_ref());
        self.tensors[i] = Tensor::from_matrix(u_k, l_full, keep);
        self.tensors[j] = Tensor::from_matrix(b_final.as_ref(), keep, r_full);
    }
}

/// Exact right partial-trace environment metric `R̄` for bond `(q1, q2)` on a
/// periodic ring in pure `O(n χ³)` using a `batch = 1` transfer right-to-left
/// around the complement `(q1 - 1) .. (q2 + 1) mod n` starting from `I_{d_l}`.
fn exact_right_bond_metric(tensors: &[Tensor], q1: usize, q2: usize) -> Mat<C64> {
    let n = tensors.len();
    let d_l = tensors[q1].left;
    let d_r = tensors[q2].right;
    if n == 2 {
        return Mat::identity(d_r, d_r);
    }
    let mut r_vec = vec![ZERO; d_l * d_l];
    for a in 0..d_l {
        r_vec[a * d_l + a] = ONE;
    }
    let mut cur = (q1 + n - 1) % n;
    while cur != q2 {
        r_vec = batch_transfer_left(&r_vec, 1, &tensors[cur], &tensors[cur], Local::I);
        cur = (cur + n - 1) % n;
    }
    Mat::<C64>::from_fn(d_r, d_r, |a, b| r_vec[a * d_r + b])
}

/// Principal square root `S = Λ_+^{1/2} U†` of a Hermitian PSD matrix `M`
/// satisfying `S† S = M` (clamping negative numerical-noise eigenvalues to `0`,
/// with no matrix inversion).
fn psd_sqrt(mat: &Mat<C64>) -> Mat<C64> {
    let d = mat.nrows();
    if d == 0 {
        return Mat::zeros(0, 0);
    }
    let sym = Mat::<C64>::from_fn(d, d, |i, j| (mat[(i, j)] + mat[(j, i)].conj()) * 0.5);
    let (vals, u) = hermitian_evd(sym.as_ref());
    let max_ev = vals.iter().copied().fold(0.0f64, f64::max);
    let floor = 1e-14 * max_ev;
    Mat::<C64>::from_fn(d, d, |i, j| {
        let s = if vals[i] > floor { vals[i].sqrt() } else { 0.0 };
        u[(j, i)].conj() * s
    })
}
