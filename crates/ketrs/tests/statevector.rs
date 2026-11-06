//! Known-state checks for the statevector engine.

mod common;

use common::{TOL, assert_state_eq, c, kron};
use ketrs::{Circuit, Complex64 as C64, Error, Gate, Statevector, library};
use std::f64::consts::{FRAC_1_SQRT_2, PI};

fn basis(n: usize, index: usize) -> Vec<C64> {
    Statevector::basis(n, index).into_amplitudes()
}

#[test]
fn qubit_zero_is_least_significant_bit() {
    let mut circ = Circuit::new(3);
    circ.x(0);
    assert_state_eq(&Statevector::from_circuit(&circ), &basis(3, 0b001));

    let mut circ = Circuit::new(3);
    circ.x(2);
    assert_state_eq(&Statevector::from_circuit(&circ), &basis(3, 0b100));
}

#[test]
fn bell_state() {
    let mut circ = Circuit::new(2);
    circ.h(0).cx(0, 1);
    let r = FRAC_1_SQRT_2;
    assert_state_eq(
        &Statevector::from_circuit(&circ),
        &[c(r, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(r, 0.0)],
    );
}

#[test]
fn cx_direction() {
    // Control on qubit 1, target qubit 0: |10⟩ → |11⟩.
    let mut circ = Circuit::new(2);
    circ.x(1).cx(1, 0);
    assert_state_eq(&Statevector::from_circuit(&circ), &basis(2, 0b11));

    // Control clear: nothing happens.
    let mut circ = Circuit::new(2);
    circ.x(0).cx(1, 0);
    assert_state_eq(&Statevector::from_circuit(&circ), &basis(2, 0b01));
}

#[test]
fn swap_moves_excitation() {
    let mut circ = Circuit::new(3);
    circ.x(0).swap(0, 2);
    assert_state_eq(&Statevector::from_circuit(&circ), &basis(3, 0b100));
}

#[test]
fn ghz_state() {
    let n = 5;
    let circ = library::ghz(n);
    let mut want = vec![c(0.0, 0.0); 1 << n];
    want[0] = c(FRAC_1_SQRT_2, 0.0);
    want[(1 << n) - 1] = c(FRAC_1_SQRT_2, 0.0);
    assert_state_eq(&Statevector::from_circuit(&circ), &want);
}

#[test]
fn qft_matches_discrete_fourier_transform() {
    let n = 4;
    let dim = 1 << n;
    for x in [0, 1, 5, 11] {
        let mut state = Statevector::basis(n, x);
        state.run(&library::qft(n)).unwrap();
        let want: Vec<C64> = (0..dim)
            .map(|k| C64::cis(2.0 * PI * (x * k) as f64 / dim as f64) / (dim as f64).sqrt())
            .collect();
        assert_state_eq(&state, &want);
    }
}

#[test]
fn unitary2_uses_qubits0_as_low_bit() {
    // X on qubits[0] and identity on qubits[1], applied to qubits [2, 0]:
    // only qubit 2 should flip.
    let identity = Gate::Phase(0.0).matrix();
    let m = kron(&identity, &Gate::X.matrix());
    let mut circ = Circuit::new(3);
    circ.unitary2(m, 2, 0);
    assert_state_eq(&Statevector::from_circuit(&circ), &basis(3, 0b100));
}

#[test]
fn inner_product_and_norm() {
    let mut circ = Circuit::new(3);
    circ.ry(0.4, 0).rx(1.3, 1).cx(0, 2).rz(0.8, 2);
    let state = Statevector::from_circuit(&circ);
    assert!((state.norm_sqr() - 1.0).abs() < TOL);
    assert!((state.inner(&state).unwrap() - c(1.0, 0.0)).norm() < TOL);
    let probs: f64 = state.probabilities().iter().sum();
    assert!((probs - 1.0).abs() < TOL);
}

#[test]
fn width_mismatches_are_errors() {
    let mut state = Statevector::zero(2);
    assert_eq!(
        state.run(&Circuit::new(3)),
        Err(Error::QubitCountMismatch {
            expected: 2,
            found: 3
        })
    );
    assert_eq!(
        Statevector::from_amplitudes(vec![c(1.0, 0.0); 3]),
        Err(Error::InvalidLength(3))
    );
}

#[test]
fn global_phase_is_applied_and_inverted() {
    // P(λ) = e^{iλ/2} Rz(λ).
    let lambda = 0.9;
    let mut rz = Circuit::new(1);
    rz.h(0).rz(lambda, 0).set_global_phase(lambda / 2.0);
    let mut p = Circuit::new(1);
    p.h(0).p(lambda, 0);
    assert_state_eq(
        &Statevector::from_circuit(&rz),
        Statevector::from_circuit(&p).amplitudes(),
    );

    let mut state = Statevector::from_circuit(&rz);
    state.run(&rz.inverse()).unwrap();
    assert_state_eq(&state, &basis(1, 0));
}

#[test]
fn diagonal_unitary2_matches_phase_gates() {
    // diag(1, e^{ia}, e^{ib}, e^{ic}) in 2·b(q1) + b(q0) order, applied with
    // q0 both below and above q1.
    let (a, b, cc) = (0.3, 1.1, -0.7);
    let zero = c(0.0, 0.0);
    let mut m = [[zero; 4]; 4];
    for (i, phase) in [0.0, a, b, cc].into_iter().enumerate() {
        m[i][i] = C64::cis(phase);
    }
    for (q0, q1) in [(0, 2), (2, 0)] {
        let mut prep = Circuit::new(3);
        prep.h(0).h(1).h(2).ry(0.4, 1);

        let mut fused = prep.clone();
        fused.unitary2(m, q0, q1);

        // |01⟩ gets a, |10⟩ gets b, |11⟩ gets c (bits as b(q1) b(q0)).
        let mut gates = prep;
        gates.p(a, q0).p(b, q1).cp(cc - a - b, q0, q1);

        assert_state_eq(
            &Statevector::from_circuit(&fused),
            Statevector::from_circuit(&gates).amplitudes(),
        );
    }
}
