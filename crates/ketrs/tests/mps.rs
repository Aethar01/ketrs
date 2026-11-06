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
    // Bonds must grow from 1, which costs accuracy in the first steps;
    // smaller steps shrink that error.
    let coarse = tdvp_error(8, 1.0, 10);
    let fine = tdvp_error(8, 1.0, 40);
    assert!(fine < 5e-4, "{fine}");
    assert!(fine < 0.5 * coarse, "{coarse} -> {fine}");
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
