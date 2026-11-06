//! Expectation values, sampling and measurement on the statevector engine.

mod common;

use common::TOL;
use ketrs::{Circuit, Error, PauliString, PauliSum, Statevector};
use rand::SeedableRng;
use rand::rngs::StdRng;

fn bell() -> Statevector {
    let mut circ = Circuit::new(2);
    circ.h(0).cx(0, 1);
    Statevector::from_circuit(&circ)
}

fn expect(state: &Statevector, label: &str) -> f64 {
    state
        .expectation_string(&PauliString::from_label(label).unwrap())
        .unwrap()
}

#[test]
fn single_qubit_expectations() {
    let theta = 0.73;
    let mut circ = Circuit::new(1);
    circ.ry(theta, 0);
    let state = Statevector::from_circuit(&circ);
    assert!((expect(&state, "Z") - theta.cos()).abs() < TOL);
    assert!((expect(&state, "X") - theta.sin()).abs() < TOL);
    assert!(expect(&state, "Y").abs() < TOL);

    let mut circ = Circuit::new(1);
    circ.rx(theta, 0);
    let state = Statevector::from_circuit(&circ);
    assert!((expect(&state, "Y") + theta.sin()).abs() < TOL);
}

#[test]
fn bell_correlations() {
    let state = bell();
    for (label, want) in [
        ("ZZ", 1.0),
        ("XX", 1.0),
        ("YY", -1.0),
        ("ZI", 0.0),
        ("IX", 0.0),
    ] {
        assert!((expect(&state, label) - want).abs() < TOL, "{label}");
    }
}

#[test]
fn pauli_sum_is_linear() {
    let h = PauliSum::from_list([("ZZ", -1.0), ("XX", 0.5), ("YY", 2.0), ("II", 0.25)]).unwrap();
    let got = bell().expectation(&h).unwrap();
    assert!((got - (-1.0 + 0.5 - 2.0 + 0.25)).abs() < TOL);
}

#[test]
fn observable_wider_than_state_is_error() {
    let err = Statevector::zero(2).expectation_string(&PauliString::from_label("ZII").unwrap());
    assert_eq!(
        err,
        Err(Error::QubitOutOfRange {
            qubit: 2,
            n_qubits: 2
        })
    );
}

#[test]
fn sampling_bell_state() {
    let mut rng = StdRng::seed_from_u64(1);
    let shots = 10_000;
    let counts = bell().sample_counts(shots, &mut rng);
    assert_eq!(counts.keys().copied().collect::<Vec<_>>(), [0b00, 0b11]);
    // Binomial σ = 50; allow 5σ.
    let zeros = counts[&0] as f64;
    assert!((zeros - 5000.0).abs() < 250.0, "{counts:?}");
}

#[test]
fn sampling_basis_state_is_deterministic() {
    let mut rng = StdRng::seed_from_u64(2);
    let state = Statevector::basis(3, 0b101);
    assert!(state.sample(100, &mut rng).iter().all(|&i| i == 0b101));
}

#[test]
fn measuring_one_half_of_bell_pair_fixes_the_other() {
    let mut rng = StdRng::seed_from_u64(3);
    let mut seen = [false; 2];
    for _ in 0..50 {
        let mut state = bell();
        let first = state.measure(0, &mut rng).unwrap();
        assert!((state.norm_sqr() - 1.0).abs() < TOL);
        let p1 = state.probability_one(1).unwrap();
        assert!((p1 - if first { 1.0 } else { 0.0 }).abs() < TOL);
        assert_eq!(state.measure(1, &mut rng).unwrap(), first);
        seen[first as usize] = true;
    }
    assert_eq!(seen, [true, true], "both outcomes should occur");
}

#[test]
fn reset_returns_qubit_to_zero() {
    let mut rng = StdRng::seed_from_u64(4);
    for _ in 0..20 {
        let mut circ = Circuit::new(2);
        circ.h(0).cx(0, 1);
        let mut state = Statevector::from_circuit(&circ);
        state.reset(1, &mut rng).unwrap();
        assert!(state.probability_one(1).unwrap().abs() < TOL);
        assert!((state.norm_sqr() - 1.0).abs() < TOL);
    }
}

#[test]
fn measure_all_collapses_to_basis_state() {
    let mut rng = StdRng::seed_from_u64(5);
    let mut state = bell();
    let index = state.measure_all(&mut rng);
    assert!(index == 0b00 || index == 0b11);
    assert_eq!(state, Statevector::basis(2, index));
}
