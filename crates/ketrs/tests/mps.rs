//! Matrix product states, checked against the statevector engine.

mod common;

use common::TOL;
use ketrs::{Circuit, Mps, MpsOptions, Pauli, PauliString, PauliSum, Statevector, library};
use rand::SeedableRng;
use rand::rngs::StdRng;

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() * n as f64) as usize % n
    }
}

fn random_circuit(n: usize, depth: usize, seed: u64) -> Circuit {
    let mut rng = Lcg(seed);
    let mut circ = Circuit::new(n);
    let paulis = [Pauli::I, Pauli::X, Pauli::Y, Pauli::Z];
    for _ in 0..depth {
        let (a, b) = (rng.below(n), rng.below(n));
        let b = if a == b {
            (a + 1 + rng.below(n - 1)) % n
        } else {
            b
        };
        let angle = rng.next() * 6.0;
        match rng.below(7) {
            0 => circ.u(angle, rng.next(), rng.next(), a),
            1 => circ.cx(a, b),
            2 => circ.crz(angle, a, b),
            3 => circ.swap(a, b),
            4 => circ.rxx(angle, a, b),
            5 => {
                let string = PauliString::new((0..n).map(|q| (q, paulis[rng.below(4)]))).unwrap();
                circ.pauli_rotation(angle, string)
            }
            _ => circ.h(a).ry(angle, b),
        };
    }
    circ.set_global_phase(0.4);
    circ
}

fn exact() -> MpsOptions {
    MpsOptions {
        max_bond: 1 << 10,
        cutoff: 0.0,
    }
}

fn assert_amplitudes_close(mps: &Mps, state: &Statevector) {
    for (i, (a, b)) in mps
        .to_amplitudes()
        .iter()
        .zip(state.amplitudes())
        .enumerate()
    {
        assert!((a - b).norm() < 1e-10, "amplitude {i}: {a} vs {b}");
    }
}

#[test]
fn untruncated_mps_matches_statevector() {
    for seed in 0..10 {
        let n = 6;
        let circ = random_circuit(n, 40, seed);
        let mps = Mps::from_circuit(&circ, exact()).unwrap();
        assert_amplitudes_close(&mps, &Statevector::from_circuit(&circ));
        assert!(mps.truncation_error() < 1e-20);
        assert!((mps.norm_sqr() - 1.0).abs() < TOL);
    }
}

#[test]
fn expectation_values_match_statevector() {
    let n = 7;
    let circ = random_circuit(n, 50, 99);
    let mps = Mps::from_circuit(&circ, exact()).unwrap();
    let state = Statevector::from_circuit(&circ);
    let h = PauliSum::from_list([
        ("ZIIIIIZ", 0.7),
        ("IXYIIII", -1.1),
        ("YIIIXIZ", 0.4),
        ("IIIZIII", 0.9),
        ("IIIIIII", 0.3),
    ])
    .unwrap();
    assert!((mps.expectation(&h).unwrap() - state.expectation(&h).unwrap()).abs() < 1e-10);
    let bits: Vec<bool> = (0..n).map(|q| q % 3 == 0).collect();
    let index: usize = bits
        .iter()
        .enumerate()
        .map(|(q, &b)| (b as usize) << q)
        .sum();
    assert!((mps.amplitude(&bits) - state.amplitudes()[index]).norm() < 1e-10);
}

#[test]
fn large_ghz_state_has_bond_dimension_two() {
    let n = 60;
    let mut mps = Mps::from_circuit(&library::ghz(n), MpsOptions::default()).unwrap();
    assert!(mps.bond_dimensions().iter().all(|&d| d == 2));
    assert!((mps.amplitude(&vec![true; n]).norm_sqr() - 0.5).abs() < TOL);
    let zz = PauliSum::from(PauliString::new([(0, Pauli::Z), (n - 1, Pauli::Z)]).unwrap());
    assert!((mps.expectation(&zz).unwrap() - 1.0).abs() < TOL);
    let mut rng = StdRng::seed_from_u64(1);
    let counts = mps.sample_counts(200, &mut rng);
    assert_eq!(
        counts.keys().copied().collect::<Vec<_>>(),
        [0, (1usize << n) - 1]
    );
}

#[test]
fn truncation_caps_bonds_and_reports_error() {
    let n = 10;
    let circ = random_circuit(n, 120, 5);
    let options = MpsOptions {
        max_bond: 4,
        cutoff: 0.0,
    };
    let mps = Mps::from_circuit(&circ, options).unwrap();
    assert!(mps.bond_dimensions().iter().all(|&d| d <= 4));
    let error = mps.truncation_error();
    assert!(error > 1e-6, "a random circuit needs more than χ = 4");
    let exact = Statevector::from_circuit(&circ);
    let approx = Statevector::from_amplitudes(mps.to_amplitudes()).unwrap();
    let infidelity = 1.0 - approx.inner(&exact).unwrap().norm_sqr() / approx.norm_sqr();
    assert!(
        infidelity <= 2.0 * error + 1e-9,
        "infidelity {infidelity}, reported {error}"
    );
}

fn ising(n: usize) -> PauliSum {
    let mut h = PauliSum::new();
    for q in 0..n {
        if q + 1 < n {
            h.add(
                -1.0,
                PauliString::new([(q, Pauli::Z), (q + 1, Pauli::Z)]).unwrap(),
            );
        }
        h.add(-0.8, PauliString::new([(q, Pauli::X)]).unwrap());
    }
    h
}

#[test]
fn tebd_matches_statevector() {
    let n = 10;
    let h = ising(n);
    let mut mps = Mps::zero(n, exact());
    mps.tebd(&h, 1.0, 20, 2).unwrap();
    // The same Trotter circuit on the statevector agrees exactly...
    let mut circ = Circuit::new(n);
    circ.pauli_evolution(&h, 1.0, 20, 2).unwrap();
    assert_amplitudes_close(&mps, &Statevector::from_circuit(&circ));
    // ...and second-order Trotter is close to exact evolution.
    let mut exact_state = Statevector::zero(n);
    exact_state.evolve(&h, 1.0).unwrap();
    let observable = PauliSum::from_list([("IIIIZZIIII", 1.0), ("IIIIIXIIII", 1.0)]).unwrap();
    let diff =
        mps.expectation(&observable).unwrap() - exact_state.expectation(&observable).unwrap();
    assert!(diff.abs() < 1e-3, "{diff}");
}

#[test]
fn unbound_circuits_are_rejected() {
    let mut circ = Circuit::new(2);
    let theta = circ.parameter();
    circ.rx(theta, 0);
    assert!(Mps::from_circuit(&circ, MpsOptions::default()).is_err());
}

// --- DMRG ---------------------------------------------------------------------

use ketrs::DmrgOptions;
use ketrs::statevector::EigenOptions;

fn exact_ground(h: &PauliSum, n: usize) -> f64 {
    Statevector::lowest_eigenstates(h, n, 1, &EigenOptions::default()).unwrap()[0].energy
}

fn heisenberg(n: usize) -> PauliSum {
    let mut h = PauliSum::new();
    for q in 0..n - 1 {
        for p in [Pauli::X, Pauli::Y, Pauli::Z] {
            h.add(1.0, PauliString::new([(q, p), (q + 1, p)]).unwrap());
        }
    }
    h
}

#[test]
fn dmrg_finds_ising_ground_state() {
    let n = 12;
    let mut h = ising(n);
    h.add(0.5, PauliString::new([]).unwrap());
    let mut mps = Mps::random(n, 8, 3, MpsOptions::default());
    let result = mps.dmrg(&h, &DmrgOptions::default()).unwrap();
    assert!(result.converged);
    assert!(
        (result.energy - exact_ground(&h, n)).abs() < 1e-8,
        "{result:?}"
    );
    assert!((mps.expectation(&h).unwrap() - result.energy).abs() < 1e-8);
}

#[test]
fn dmrg_from_neel_state_finds_heisenberg_ground_state() {
    let n = 12;
    let h = heisenberg(n);
    let neel: Vec<bool> = (0..n).map(|q| q % 2 == 1).collect();
    let mut mps = Mps::basis(n, &neel, MpsOptions::default());
    let result = mps.dmrg(&h, &DmrgOptions::default()).unwrap();
    assert!(
        (result.energy - exact_ground(&h, n)).abs() < 1e-8,
        "{result:?}"
    );
}

/// Hops and a field under a strong Gauss-law-like penalty
/// `big Σ_{i<j} Zᵢ Zⱼ` (`big/2 (Σ Zᵢ)²` without its constant), so the
/// spectrum sits near `−big n/2`.
fn penalised(n: usize, big: f64) -> PauliSum {
    let mut h = PauliSum::new();
    for q in 0..n - 1 {
        for p in [Pauli::X, Pauli::Y] {
            h.add(1.0, PauliString::new([(q, p), (q + 1, p)]).unwrap());
        }
    }
    for q in 0..n {
        h.add(
            0.3 * (q % 3) as f64,
            PauliString::new([(q, Pauli::Z)]).unwrap(),
        );
        for j in q + 1..n {
            h.add(
                big,
                PauliString::new([(q, Pauli::Z), (j, Pauli::Z)]).unwrap(),
            );
        }
    }
    h
}

#[test]
fn dmrg_converges_with_a_spectrum_far_from_zero() {
    // Near −6e6, rounding in H x keeps Lanczos's residual above the default
    // tolerance of 1e-10, which used to abort DMRG ("Lanczos did not
    // converge in 1000 iterations").
    let n = 12;
    let h = penalised(n, 1e6);
    let neel: Vec<bool> = (0..n).map(|q| q % 2 == 1).collect();
    let mut mps = Mps::basis(n, &neel, MpsOptions::default());
    let options = DmrgOptions {
        max_bond: 32,
        ..Default::default()
    };
    let result = mps.dmrg(&h, &options).unwrap();
    // Converged within rounding, not merely given up on.
    assert_eq!(result.unconverged_solves, 0);
    let eigen = EigenOptions {
        tolerance: 1e-6,
        ..Default::default()
    };
    let exact = Statevector::lowest_eigenstates(&h, n, 1, &eigen).unwrap()[0].energy;
    assert!(
        (result.energy - exact).abs() < 1e-6,
        "{} vs {exact}",
        result.energy
    );
}

#[test]
fn dmrg_handles_all_to_all_couplings() {
    let n = 10;
    let mut h = PauliSum::new();
    for i in 0..n {
        for j in i + 1..n {
            h.add(
                0.5 / (j - i) as f64,
                PauliString::new([(i, Pauli::Z), (j, Pauli::Z)]).unwrap(),
            );
        }
        h.add(-1.0, PauliString::new([(i, Pauli::X)]).unwrap());
    }
    let mut mps = Mps::random(n, 8, 7, MpsOptions::default());
    let result = mps.dmrg(&h, &DmrgOptions::default()).unwrap();
    assert!(
        (result.energy - exact_ground(&h, n)).abs() < 1e-8,
        "{result:?}"
    );
}

#[test]
fn dmrg_on_a_long_critical_chain() {
    // H = −Σ ZZ − Σ X at the critical point: −4/π per site in the bulk.
    let n = 80;
    let mut h = PauliSum::new();
    for q in 0..n {
        if q + 1 < n {
            h.add(
                -1.0,
                PauliString::new([(q, Pauli::Z), (q + 1, Pauli::Z)]).unwrap(),
            );
        }
        h.add(-1.0, PauliString::new([(q, Pauli::X)]).unwrap());
    }
    let mut mps = Mps::random(n, 8, 1, MpsOptions::default());
    let options = DmrgOptions {
        max_bond: 32,
        ..DmrgOptions::default()
    };
    let result = mps.dmrg(&h, &options).unwrap();
    let per_site = result.energy / n as f64;
    assert!(
        (per_site + 4.0 / std::f64::consts::PI).abs() < 0.01,
        "{per_site}"
    );
    // Sweeps only ever lower the energy (up to rounding).
    for w in result.sweep_energies.windows(2) {
        assert!(w[1] <= w[0] + 1e-9, "{:?}", result.sweep_energies);
    }
}

/// All-to-all ZZ and XX couplings decaying as 1/r^1.5, and a field.
fn long_range(n: usize) -> PauliSum {
    let mut h = PauliSum::new();
    for i in 0..n {
        for j in i + 1..n {
            let c = 1.0 / ((j - i) as f64).powf(1.5);
            h.add(c, PauliString::new([(i, Pauli::Z), (j, Pauli::Z)]).unwrap());
            h.add(
                0.5 * c,
                PauliString::new([(i, Pauli::X), (j, Pauli::X)]).unwrap(),
            );
        }
        h.add(0.8, PauliString::new([(i, Pauli::X)]).unwrap());
    }
    h.add(0.3, PauliString::identity());
    h
}

fn tdvp_error(n: usize, time: f64, steps: usize) -> f64 {
    let h = long_range(n);
    let neel: Vec<bool> = (0..n).map(|q| q % 2 == 0).collect();
    let mut mps = Mps::basis(n, &neel, exact());
    mps.tdvp(&h, time, steps).unwrap();
    let start = neel
        .iter()
        .enumerate()
        .map(|(q, &b)| (b as usize) << q)
        .sum();
    let mut want = Statevector::basis(n, start);
    want.evolve(&h, time).unwrap();
    let got = mps.to_amplitudes();
    got.iter()
        .zip(want.amplitudes())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0, f64::max)
}

fn tdvp_error_full(n: usize, time: f64, steps: usize) -> f64 {
    let h = long_range(n);
    let mut mps = Mps::random(n, 1 << (n / 2), 3, exact());
    let mut want = Statevector::from_amplitudes(mps.to_amplitudes()).unwrap();
    mps.tdvp(&h, time, steps).unwrap();
    want.evolve(&h, time).unwrap();
    mps.to_amplitudes()
        .iter()
        .zip(want.amplitudes())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0, f64::max)
}

#[test]
fn tdvp_is_exact_with_full_bonds() {
    // With every bond at full dimension the projection is exact, so only
    // the local Krylov exponentials contribute error.
    for steps in [1, 5] {
        let error = tdvp_error_full(8, 1.0, steps);
        assert!(error < 1e-11, "{steps} steps: {error}");
    }
}

#[test]
fn tdvp_from_a_product_state_converges() {
    // Bonds must grow from 1. Subspace expansion lets long-range couplings
    // act from the first step, so the error falls as for a second-order
    // integrator (it was 1e-3, falling linearly, without it).
    let coarse = tdvp_error(8, 1.0, 10);
    let fine = tdvp_error(8, 1.0, 40);
    assert!(coarse < 1e-5, "{coarse}");
    assert!(fine < coarse / 16.0, "{coarse} -> {fine}");
}

#[test]
fn tdvp_conserves_norm_and_energy_under_truncation() {
    let n = 16;
    let h = long_range(n);
    let neel: Vec<bool> = (0..n).map(|q| q % 2 == 0).collect();
    let options = MpsOptions {
        max_bond: 8,
        cutoff: 1e-12,
    };
    let mut mps = Mps::basis(n, &neel, options);
    let e0 = mps.expectation(&h).unwrap();
    mps.tdvp(&h, 1.0, 10).unwrap();
    assert!(mps.bond_dimensions().iter().all(|&d| d <= 8));
    assert!(mps.truncation_error() > 1e-6, "χ = 8 should truncate");
    assert!((mps.norm_sqr() - 1.0).abs() < 1e-10);
    let e1 = mps.expectation(&h).unwrap();
    assert!(((e1 - e0) / e0).abs() < 2e-3, "{e0} -> {e1}");
}

#[test]
fn tdvp_rejects_bad_arguments() {
    let h = long_range(3);
    assert!(Mps::zero(3, exact()).tdvp(&h, 1.0, 0).is_err());
    assert!(
        Mps::zero(1, exact())
            .tdvp(&PauliSum::from_list([("X", 1.0)]).unwrap(), 1.0, 1)
            .is_err()
    );
    // Terms beyond the chain are rejected.
    assert!(Mps::zero(2, exact()).tdvp(&h, 1.0, 1).is_err());
}

// --- Terms on more than two sites, from product states -----------------------

use ketrs::Complex64 as C64;

/// Adds the Hermitian `op` acting on `qubits` (local bit `j` is
/// `qubits[j]`) to `sum` as Pauli strings.
fn add_dense(sum: &mut PauliSum, qubits: &[usize], op: &[Vec<C64>]) {
    let k = qubits.len();
    let dim = 1usize << k;
    let paulis = [Pauli::I, Pauli::X, Pauli::Y, Pauli::Z];
    for code in 0..1usize << (2 * k) {
        let ps: Vec<Pauli> = (0..k).map(|j| paulis[(code >> (2 * j)) & 3]).collect();
        // Tr(P O) / dim, with P a signed permutation: row → col.
        let mut trace = C64::new(0.0, 0.0);
        for (row, op_row) in op.iter().enumerate() {
            let mut col = 0;
            let mut v = C64::new(1.0, 0.0);
            for (j, p) in ps.iter().enumerate() {
                let b = (row >> j) & 1;
                let (cb, val) = match p {
                    Pauli::I => (b, C64::new(1.0, 0.0)),
                    Pauli::X => (1 - b, C64::new(1.0, 0.0)),
                    Pauli::Y => (1 - b, C64::new(0.0, if b == 0 { -1.0 } else { 1.0 })),
                    Pauli::Z => (b, C64::new(if b == 0 { 1.0 } else { -1.0 }, 0.0)),
                };
                col |= cb << j;
                v *= val;
            }
            trace += v * op_row[col].conj();
        }
        let c = trace.re / dim as f64;
        if c.abs() > 1e-12 {
            let ops = qubits
                .iter()
                .zip(&ps)
                .filter(|(_, p)| **p != Pauli::I)
                .map(|(&q, &p)| (q, p));
            sum.add(c, PauliString::new(ops).unwrap());
        }
    }
}

/// Adds `scale · (c0 + Σ c_q Z_q)²`.
fn add_square(sum: &mut PauliSum, scale: f64, c0: f64, linear: &[(usize, f64)]) {
    let mut constant = c0 * c0;
    for (i, &(q, c)) in linear.iter().enumerate() {
        constant += c * c;
        sum.add(
            scale * 2.0 * c0 * c,
            PauliString::new([(q, Pauli::Z)]).unwrap(),
        );
        for &(r, d) in &linear[i + 1..] {
            sum.add(
                scale * 2.0 * c * d,
                PauliString::new([(q, Pauli::Z), (r, Pauli::Z)]).unwrap(),
            );
        }
    }
    sum.add(scale * constant, PauliString::identity());
}

/// A qubit-encoded U(1) lattice gauge theory, like the Schwinger model:
/// each cell holds a staggered fermion and a two-qubit link register
/// (electric field E ∈ {−1, 0, 1, 2}), with
///
/// `H = x Σ (ψ†ₙ Uₙ ψₙ₊₁ + h.c.) + μ Σ (−1)ⁿ nₙ + Σ Eₙ² + λ Σ Gₙ²`,
///
/// where U raises E and `Gₙ = Eₙ − Eₙ₋₁ − qₙ` is Gauss's law. Each hop acts
/// on four qubits. An open chain has cells in order, so each hop's qubits
/// are consecutive. A ring has cells in the folded order 0, N−1, 1, N−2, …,
/// so hops span up to eight qubits with gaps. Returns H and the
/// strong-coupling vacuum, from which no two-site move creates a hop.
fn gauge_model(cells: usize, ring: bool) -> (PauliSum, Vec<bool>) {
    let (x, mu, lambda) = (0.5, 0.5, 5.0);
    let links = if ring { cells } else { cells - 1 };
    let mut position = vec![0; cells];
    if ring {
        let mut order = Vec::new();
        for j in 0..cells.div_ceil(2) {
            order.push(j);
            if cells - 1 - j != j {
                order.push(cells - 1 - j);
            }
        }
        for (p, &cell) in order.iter().enumerate() {
            position[cell] = p;
        }
    } else {
        position = (0..cells).collect();
    }
    let matter = |n: usize| 3 * position[n % cells];
    let link = |n: usize| [matter(n) + 1, matter(n) + 2];
    let n_qubits = if ring { 3 * cells } else { 3 * cells - 2 };

    let mut h = PauliSum::new();
    let zero = C64::new(0.0, 0.0);
    for n in 0..links {
        // Local bits: 0 = ψₙ, 1-2 = link n (E + 1 in binary), 3 = ψₙ₊₁.
        let mut op = vec![vec![zero; 16]; 16];
        for e in 0..3 {
            let (from, to) = ((e << 1) | 8, 1 | ((e + 1) << 1));
            op[to][from] = C64::new(x, 0.0);
            op[from][to] = C64::new(x, 0.0);
        }
        let [a, b] = link(n);
        add_dense(&mut h, &[matter(n), a, b, matter(n + 1)], &op);
    }
    // E = 1/2 − Z_a/2 − Z_b.
    let field = |n: usize| {
        let [a, b] = link(n);
        [(a, -0.5), (b, -1.0)]
    };
    for n in 0..cells {
        let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
        // μ (−1)ⁿ (1 − Z)/2
        h.add(
            -mu * sign / 2.0,
            PauliString::new([(matter(n), Pauli::Z)]).unwrap(),
        );
        h.add(mu * sign / 2.0, PauliString::identity());
        // Gₙ, with qₙ = (1 − Z)/2 − [n odd].
        let mut c0 = -0.5 + (n % 2) as f64;
        let mut linear = vec![(matter(n), 0.5)];
        if n < links {
            add_square(&mut h, 1.0, 0.5, &field(n));
            c0 += 0.5;
            linear.extend(field(n));
        }
        if n > 0 || ring {
            c0 -= 0.5;
            linear.extend(field((n + cells - 1) % cells).map(|(q, c)| (q, -c)));
        }
        add_square(&mut h, lambda, c0, &linear);
    }
    let mut vacuum = vec![false; n_qubits];
    for n in 0..cells {
        vacuum[matter(n)] = n % 2 == 1;
        if n < links {
            vacuum[link(n)[0]] = true; // E = 0
        }
    }
    (h.simplify(1e-14), vacuum)
}

fn plain() -> DmrgOptions {
    DmrgOptions {
        noise: 0.0,
        ..DmrgOptions::default()
    }
}

#[test]
fn dmrg_noise_escapes_product_states_under_four_qubit_hops() {
    for (cells, ring) in [(4, false), (4, true)] {
        let (h, vacuum) = gauge_model(cells, ring);
        let n = vacuum.len();
        let exact = exact_ground(&h, n);
        let start = Mps::basis(n, &vacuum, MpsOptions::default());
        let e_vacuum = start.expectation(&h).unwrap();

        // Plain two-site DMRG never leaves the vacuum.
        let stuck = start.clone().dmrg(&h, &plain()).unwrap();
        assert!(stuck.converged);
        assert!((stuck.energy - e_vacuum).abs() < 1e-10, "{stuck:?}");
        assert!(stuck.energy > exact + 0.3, "{exact} {stuck:?}");

        let mut mps = start;
        let result = mps.dmrg(&h, &DmrgOptions::default()).unwrap();
        assert!(result.converged);
        assert!(
            (result.energy - exact).abs() < 1e-8,
            "ring {ring}: {exact} {result:?}"
        );
        assert!((mps.expectation(&h).unwrap() - result.energy).abs() < 1e-8);
    }
}

#[test]
fn dmrg_noise_escapes_weak_random_starts() {
    // From bond-2 random starts, plain DMRG settles in metastable states
    // for some seeds; with noise every seed reaches the ground state.
    let (h, vacuum) = gauge_model(4, true);
    let n = vacuum.len();
    let exact = exact_ground(&h, n);
    let mut worst_plain: f64 = 0.0;
    for seed in 0..6 {
        let start = Mps::random(n, 2, seed, MpsOptions::default());
        let plain = start.clone().dmrg(&h, &plain()).unwrap();
        worst_plain = worst_plain.max(plain.energy - exact);
        let result = start.clone().dmrg(&h, &DmrgOptions::default()).unwrap();
        assert!(
            (result.energy - exact).abs() < 1e-8,
            "seed {seed}: {exact} {result:?}"
        );
    }
    assert!(worst_plain > 1e-6, "{worst_plain}");
}

fn gauge_tdvp_error(ring: bool, steps: usize) -> (f64, Vec<usize>) {
    let (h, vacuum) = gauge_model(4, ring);
    let n = vacuum.len();
    let mut mps = Mps::basis(n, &vacuum, exact());
    mps.tdvp(&h, 1.0, steps).unwrap();
    let start = vacuum
        .iter()
        .enumerate()
        .map(|(q, &b)| (b as usize) << q)
        .sum();
    let mut want = Statevector::basis(n, start);
    want.evolve(&h, 1.0).unwrap();
    let error = mps
        .to_amplitudes()
        .iter()
        .zip(want.amplitudes())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0, f64::max);
    (error, mps.bond_dimensions())
}

#[test]
fn tdvp_from_a_product_state_with_four_qubit_hops() {
    // The subspace expansion must reach every bond, not only those near
    // the right end, or the hops on the left never act: it used to leave
    // the left bonds at 1 and an error of 0.35 at any step count.
    let (coarse, bonds) = gauge_tdvp_error(false, 10);
    assert!(bonds.iter().all(|&d| d > 1), "{bonds:?}");
    assert!(coarse < 2e-5, "{coarse}");
    // The error falls as for a second-order integrator.
    let (fine, _) = gauge_tdvp_error(false, 40);
    assert!(fine < coarse / 12.0, "{coarse} -> {fine}");
    // Folded onto a chain, hops span up to eight qubits.
    let (error, bonds) = gauge_tdvp_error(true, 10);
    assert!(bonds.iter().all(|&d| d > 1), "{bonds:?}");
    assert!(error < 1e-3, "{error}");
}
