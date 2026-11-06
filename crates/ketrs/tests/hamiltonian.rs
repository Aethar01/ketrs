//! Pauli sums on statevectors (apply, expectation, evolve, eigenstates),
//! Pauli rotations and Trotter circuits, and real statevectors, checked
//! against dense matrices built entry by entry.

mod common;

use common::{TOL, assert_state_eq, c};
use ketrs::statevector::EigenOptions;
use ketrs::{
    Circuit, Complex64 as C64, Error, Pauli, PauliString, PauliSum, RealStatevector, Statevector,
};
use proptest::prelude::*;

type Dense = Vec<Vec<C64>>;

/// `⟨i|H|j⟩ = Σₜ cₜ Πq ⟨i_q|Pq|j_q⟩`, with i_q the value of bit q.
fn dense(h: &PauliSum, n: usize) -> Dense {
    let dim = 1 << n;
    let single = |p: Pauli, r: usize, col: usize| -> C64 {
        let (zero, one, i) = (c(0.0, 0.0), c(1.0, 0.0), c(0.0, 1.0));
        match (p, r, col) {
            (Pauli::I, r, col) => {
                if r == col {
                    one
                } else {
                    zero
                }
            }
            (Pauli::X, r, col) => {
                if r != col {
                    one
                } else {
                    zero
                }
            }
            (Pauli::Y, 0, 1) => -i,
            (Pauli::Y, 1, 0) => i,
            (Pauli::Y, _, _) => zero,
            (Pauli::Z, r, col) if r == col => {
                if r == 0 {
                    one
                } else {
                    -one
                }
            }
            (Pauli::Z, _, _) => zero,
        }
    };
    let mut m = vec![vec![c(0.0, 0.0); dim]; dim];
    for (coeff, string) in h.terms() {
        for (r, row) in m.iter_mut().enumerate() {
            for (col, entry) in row.iter_mut().enumerate() {
                let mut v = c(*coeff, 0.0);
                for q in 0..n {
                    let p = string
                        .ops()
                        .iter()
                        .find(|&&(k, _)| k == q)
                        .map_or(Pauli::I, |&(_, p)| p);
                    v *= single(p, (r >> q) & 1, (col >> q) & 1);
                }
                *entry += v;
            }
        }
    }
    m
}

fn matvec(m: &Dense, v: &[C64]) -> Vec<C64> {
    m.iter()
        .map(|row| row.iter().zip(v).map(|(a, b)| a * b).sum())
        .collect()
}

fn matmul(a: &Dense, b: &Dense) -> Dense {
    let n = a.len();
    (0..n)
        .map(|r| {
            (0..n)
                .map(|col| (0..n).map(|k| a[r][k] * b[k][col]).sum())
                .collect()
        })
        .collect()
}

/// exp(-iHt) by scaling and squaring a Taylor series.
fn dense_expm(h: &Dense, t: f64) -> Dense {
    let n = h.len();
    let squarings = 10;
    let a: Dense = h
        .iter()
        .map(|row| {
            row.iter()
                .map(|x| x * c(0.0, -t / f64::powi(2.0, squarings)))
                .collect()
        })
        .collect();
    let mut result: Dense = (0..n)
        .map(|r| (0..n).map(|col| c((r == col) as u8 as f64, 0.0)).collect())
        .collect();
    let mut term = result.clone();
    for k in 1..30 {
        term = matmul(&term, &a);
        for row in term.iter_mut() {
            for x in row.iter_mut() {
                *x /= k as f64;
            }
        }
        for (rr, tr) in result.iter_mut().zip(&term) {
            for (x, y) in rr.iter_mut().zip(tr) {
                *x += y;
            }
        }
    }
    for _ in 0..squarings {
        result = matmul(&result, &result);
    }
    result
}

fn random_state(n: usize, seed: u64) -> Statevector {
    let mut circ = Circuit::new(n);
    let mut s = seed;
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 11) as f64 / (1u64 << 53) as f64 * 6.0
    };
    for q in 0..n {
        circ.u(next(), next(), next(), q);
    }
    for q in 1..n {
        circ.cx(q - 1, q).ry(next(), q).rz(next(), q - 1);
    }
    Statevector::from_circuit(&circ)
}

fn pauli() -> impl Strategy<Value = Pauli> {
    prop_oneof![
        Just(Pauli::I),
        Just(Pauli::X),
        Just(Pauli::Y),
        Just(Pauli::Z)
    ]
}

fn hamiltonian(n: usize) -> impl Strategy<Value = PauliSum> {
    prop::collection::vec((-2.0..2.0f64, prop::collection::vec(pauli(), n)), 1..8).prop_map(
        |terms| {
            let mut h = PauliSum::new();
            for (coeff, ps) in terms {
                h.add(coeff, PauliString::new(ps.into_iter().enumerate()).unwrap());
            }
            h
        },
    )
}

proptest! {
    #[test]
    fn apply_and_expectation_match_dense((n, h, seed) in (1usize..=5).prop_flat_map(|n| (Just(n), hamiltonian(n), any::<u64>()))) {
        let psi = random_state(n, seed);
        let want = matvec(&dense(&h, n), psi.amplitudes());
        let got = psi.apply_pauli_sum(&h).unwrap();
        for (g, w) in got.amplitudes().iter().zip(&want) {
            prop_assert!((g - w).norm() < TOL);
        }
        let want_e: C64 = psi.amplitudes().iter().zip(&want).map(|(a, b)| a.conj() * b).sum();
        prop_assert!((psi.expectation(&h).unwrap() - want_e.re).abs() < TOL);
    }

    #[test]
    fn evolve_matches_dense_exponential((n, h, seed, t) in (1usize..=4).prop_flat_map(|n| (Just(n), hamiltonian(n), any::<u64>(), -3.0..3.0f64))) {
        let mut psi = random_state(n, seed);
        let want = matvec(&dense_expm(&dense(&h, n), t), psi.amplitudes());
        psi.evolve(&h, t).unwrap();
        for (g, w) in psi.amplitudes().iter().zip(&want) {
            prop_assert!((g - w).norm() < 1e-9, "{g} vs {w}");
        }
    }

    #[test]
    fn pauli_rotation_is_exp_of_pauli_string((n, ps, theta, seed) in (1usize..=5).prop_flat_map(|n| (Just(n), prop::collection::vec(pauli(), n), -6.0..6.0f64, any::<u64>()))) {
        // exp(-iθ/2·P) is exp(-iHt) with H = P and t = θ/2.
        let string = PauliString::new(ps.into_iter().enumerate()).unwrap();
        let psi = random_state(n, seed);
        let h = PauliSum::from(string.clone());
        let want = matvec(&dense_expm(&dense(&h, n), theta / 2.0), psi.amplitudes());
        let mut circ = Circuit::new(n);
        circ.pauli_rotation(theta, string);
        let mut got = psi.clone();
        got.run(&circ).unwrap();
        for (g, w) in got.amplitudes().iter().zip(&want) {
            prop_assert!((g - w).norm() < 1e-9);
        }
    }
}

#[test]
fn single_qubit_rabi_oscillation() {
    // H = ωX: |0⟩ → cos(ωt)|0⟩ − i sin(ωt)|1⟩.
    let (omega, t) = (0.7, 1.9);
    let mut psi = Statevector::zero(1);
    psi.evolve(&PauliSum::from_list([("X", omega)]).unwrap(), t)
        .unwrap();
    assert_state_eq(
        &psi,
        &[c((omega * t).cos(), 0.0), c(0.0, -(omega * t).sin())],
    );
}

#[test]
fn identity_terms_only_add_phase() {
    let h = PauliSum::from_list([("II", 1.5), ("ZZ", 0.3)]).unwrap();
    let mut psi = Statevector::zero(2);
    psi.evolve(&h, 2.0).unwrap();
    assert_state_eq(
        &psi,
        &Statevector::basis(2, 0)
            .into_amplitudes()
            .iter()
            .map(|a| a * C64::cis(-1.8 * 2.0))
            .collect::<Vec<_>>(),
    );
}

#[test]
fn qiskit_rotation_conventions() {
    // RZZ(θ)|00⟩ = e^{-iθ/2}|00⟩; RXX(π)|00⟩ = -i|11⟩.
    let mut circ = Circuit::new(2);
    circ.rzz(0.8, 0, 1);
    assert_state_eq(
        &Statevector::from_circuit(&circ),
        &[C64::cis(-0.4), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)],
    );
    let mut circ = Circuit::new(2);
    circ.rxx(std::f64::consts::PI, 0, 1);
    assert_state_eq(
        &Statevector::from_circuit(&circ),
        &[c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, -1.0)],
    );
}

fn ising(n: usize, field: f64) -> PauliSum {
    let mut h = PauliSum::new();
    for q in 0..n {
        if q + 1 < n {
            h.add(
                -1.0,
                PauliString::new([(q, Pauli::Z), (q + 1, Pauli::Z)]).unwrap(),
            );
        }
        h.add(-field, PauliString::new([(q, Pauli::X)]).unwrap());
    }
    h
}

#[test]
fn trotter_error_scales_with_order() {
    let (n, t) = (4, 1.0);
    let h = ising(n, 0.9);
    let psi = random_state(n, 3);
    let mut exact = psi.clone();
    exact.evolve(&h, t).unwrap();
    for order in [1, 2, 4] {
        let error = |steps| {
            let mut circ = Circuit::new(n);
            circ.pauli_evolution(&h, t, steps, order).unwrap();
            let mut s = psi.clone();
            s.run(&circ).unwrap();
            1.0 - s.inner(&exact).unwrap().norm()
        };
        // Infidelity scales as steps^(-2·order).
        let ratio = error(4) / error(8);
        let want = f64::powi(2.0, 2 * order as i32);
        assert!(
            ratio > 0.6 * want && ratio < 1.6 * want,
            "order {order}: ratio {ratio}, want ~{want}"
        );
    }
}

#[test]
fn trotter_rejects_bad_order() {
    let h = ising(2, 1.0);
    assert!(matches!(
        Circuit::new(2).pauli_evolution(&h, 1.0, 1, 3),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        Circuit::new(2).pauli_evolution(&h, 1.0, 0, 1),
        Err(Error::InvalidArgument(_))
    ));
}

#[test]
fn eigenstates_with_degeneracy() {
    // H = Σ Zᵢ: lowest -n, then -(n-2) with multiplicity n.
    let n = 4;
    let labels: Vec<String> = (0..n)
        .map(|q| {
            let mut l = vec!['I'; n];
            l[n - 1 - q] = 'Z';
            l.into_iter().collect()
        })
        .collect();
    let h = PauliSum::from_list(labels.iter().map(|l| (l.as_str(), 1.0))).unwrap();
    let states = Statevector::lowest_eigenstates(&h, n, 1 + n, &EigenOptions::default()).unwrap();
    let energies: Vec<f64> = states.iter().map(|e| e.energy).collect();
    assert!((energies[0] + 4.0).abs() < 1e-8, "{energies:?}");
    for e in &energies[1..] {
        assert!((e + 2.0).abs() < 1e-8, "{energies:?}");
    }
    // Orthonormal, with small residuals.
    for (i, a) in states.iter().enumerate() {
        assert!(a.residual < 1e-7);
        for b in &states[i + 1..] {
            assert!(a.state.inner(&b.state).unwrap().norm() < 1e-8);
        }
    }
}

#[test]
fn heisenberg_dimer_singlet_and_triplet() {
    let h = PauliSum::from_list([("XX", 1.0), ("YY", 1.0), ("ZZ", 1.0)]).unwrap();
    let states = Statevector::lowest_eigenstates(&h, 2, 4, &EigenOptions::default()).unwrap();
    let energies: Vec<f64> = states.iter().map(|e| e.energy).collect();
    for (got, want) in energies.iter().zip([-3.0, 1.0, 1.0, 1.0]) {
        assert!((got - want).abs() < 1e-8, "{energies:?}");
    }
}

#[test]
fn ground_state_matches_dense_minimum_and_real_engine() {
    let n = 8;
    let h = ising(n, 0.7);
    let opts = EigenOptions::default();
    let complex = &Statevector::lowest_eigenstates(&h, n, 2, &opts).unwrap();
    let real = &RealStatevector::lowest_eigenstates(&h, n, 2, &opts).unwrap();
    for (a, b) in complex.iter().zip(real) {
        assert!((a.energy - b.energy).abs() < 1e-8);
        assert!(b.residual < 1e-7);
    }
    // The ground energy is below ⟨ψ|H|ψ⟩ for any state, and is an
    // eigenvalue: check against the dense Rayleigh quotient of the result.
    let m = dense(&h, n);
    let x = complex[0].state.amplitudes();
    let hx = matvec(&m, x);
    let e: C64 = x.iter().zip(&hx).map(|(a, b)| a.conj() * b).sum();
    assert!((e.re - complex[0].energy).abs() < 1e-8);
    for seed in 0..5 {
        assert!(random_state(n, seed).expectation(&h).unwrap() >= complex[0].energy - 1e-9);
    }
}

#[test]
fn real_statevector_runs_real_circuits() {
    let mut circ = Circuit::new(5);
    circ.h(0).ry(0.4, 1).cx(0, 2).cz(1, 3).swap(2, 4).x(3).z(0);
    circ.pauli_rotation(0.9, PauliString::from_label("YXZII").unwrap());
    let real = RealStatevector::from_circuit(&circ).unwrap();
    let complex = Statevector::from_circuit(&circ);
    for (r, z) in real.amplitudes().iter().zip(complex.amplitudes()) {
        assert!((c(*r, 0.0) - z).norm() < TOL);
    }
    let h = ising(5, 0.5);
    assert!((real.expectation(&h).unwrap() - complex.expectation(&h).unwrap()).abs() < TOL);
}

#[test]
fn real_statevector_rejects_complex_gates_without_changing_state() {
    let mut circ = Circuit::new(2);
    circ.h(0).s(0);
    let mut state = RealStatevector::zero(2);
    assert!(matches!(state.run(&circ), Err(Error::NotReal(_))));
    assert_eq!(state, RealStatevector::zero(2));

    // exp(-iθ/2·P) = cos − i·sin·P is real only when P is imaginary, i.e.
    // has an odd number of Y factors. YY is a real matrix, so RYY is complex.
    let mut circ = Circuit::new(2);
    circ.ryy(0.3, 0, 1);
    assert!(matches!(
        RealStatevector::from_circuit(&circ),
        Err(Error::NotReal(_))
    ));
}
