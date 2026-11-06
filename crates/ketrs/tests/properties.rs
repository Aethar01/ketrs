//! Property tests: norm, inversion, and cross-checks against the reference
//! simulator and against direct Pauli application.

mod common;

use common::{TOL, controlled_matrix, kron, reference_run, swap_matrix};
use ketrs::{Circuit, Gate, Operation, Pauli, PauliString, Statevector};
use proptest::prelude::*;

fn angle() -> impl Strategy<Value = f64> {
    -10.0..10.0f64
}

fn gate() -> impl Strategy<Value = Gate> {
    prop_oneof![
        Just(Gate::H),
        Just(Gate::X),
        Just(Gate::Y),
        Just(Gate::Z),
        Just(Gate::S),
        Just(Gate::Sdg),
        Just(Gate::T),
        Just(Gate::Tdg),
        angle().prop_map(Gate::Rx),
        angle().prop_map(Gate::Ry),
        angle().prop_map(Gate::Rz),
        angle().prop_map(Gate::Phase),
        (angle(), angle(), angle()).prop_map(|(theta, phi, lambda)| Gate::U { theta, phi, lambda }),
    ]
}

fn pair(n: usize) -> impl Strategy<Value = (usize, usize)> {
    (0..n, 0..n).prop_filter("distinct qubits", |(a, b)| a != b)
}

fn operation(n: usize) -> impl Strategy<Value = Operation> {
    prop_oneof![
        (gate(), 0..n).prop_map(|(gate, target)| Operation::Gate { gate, target }),
        (gate(), pair(n)).prop_map(|(gate, (control, target))| Operation::Controlled {
            gate,
            control,
            target
        }),
        pair(n).prop_map(|(a, b)| Operation::Swap(a, b)),
        // Any set of controls, including none.
        (gate(), 0..n, prop::collection::vec(any::<bool>(), n)).prop_map(
            |(gate, target, chosen)| Operation::MultiControlled {
                gate,
                controls: (0..chosen.len())
                    .filter(|&q| chosen[q] && q != target)
                    .collect(),
                target,
            }
        ),
        (pair(n), prop::collection::vec(any::<bool>(), n)).prop_map(|((a, b), chosen)| {
            Operation::ControlledSwap {
                controls: (0..chosen.len())
                    .filter(|&q| chosen[q] && q != a && q != b)
                    .collect(),
                a,
                b,
            }
        }),
        (gate(), gate(), pair(n)).prop_map(|(g0, g1, (q0, q1))| Operation::Unitary2 {
            matrix: kron(&g1.matrix(), &g0.matrix()),
            qubits: [q0, q1],
        }),
        (prop::collection::vec(pauli(), n), angle()).prop_map(|(ps, theta)| {
            Operation::PauliRotation {
                pauli: PauliString::new(ps.into_iter().enumerate()).unwrap(),
                theta,
            }
        }),
    ]
}

fn circuit() -> impl Strategy<Value = Circuit> {
    circuit_on(2..=6)
}

fn circuit_on(sizes: std::ops::RangeInclusive<usize>) -> impl Strategy<Value = Circuit> {
    sizes.prop_flat_map(|n| {
        prop::collection::vec(operation(n), 0..40).prop_map(move |ops| {
            let mut circ = Circuit::new(n);
            for op in ops {
                circ.push(op).unwrap();
            }
            circ
        })
    })
}

/// A circuit together with a single extra operation on the same qubits.
fn circuit_and_op() -> impl Strategy<Value = (Circuit, Operation)> {
    circuit().prop_flat_map(|circ| {
        let n = circ.n_qubits();
        (Just(circ), operation(n))
    })
}

fn assert_close(a: &Statevector, b: &Statevector) -> Result<(), TestCaseError> {
    for (x, y) in a.amplitudes().iter().zip(b.amplitudes()) {
        prop_assert!((x - y).norm() < TOL, "{x} != {y}");
    }
    Ok(())
}

/// The same operation expressed as a generic `Unitary2`, so the specialised
/// kernels are checked against the generic one.
fn as_unitary2(op: &Operation) -> Option<Operation> {
    match *op {
        Operation::Controlled {
            gate,
            control,
            target,
        } => Some(Operation::Unitary2 {
            matrix: controlled_matrix(&gate.matrix()),
            qubits: [control, target],
        }),
        Operation::Swap(a, b) => Some(Operation::Unitary2 {
            matrix: swap_matrix(),
            qubits: [a, b],
        }),
        Operation::MultiControlled { .. } | Operation::ControlledSwap { .. } => None,
        Operation::Gate { gate, target } => {
            // Pair with any other qubit, acting as identity on it.
            let other = if target == 0 { 1 } else { 0 };
            Some(Operation::Unitary2 {
                matrix: kron(&Gate::Phase(0.0).matrix(), &gate.matrix()),
                qubits: [target, other],
            })
        }
        Operation::Unitary2 { .. } | Operation::PauliRotation { .. } => None,
    }
}

fn pauli() -> impl Strategy<Value = Pauli> {
    prop_oneof![
        Just(Pauli::I),
        Just(Pauli::X),
        Just(Pauli::Y),
        Just(Pauli::Z)
    ]
}

/// A circuit together with a Pauli string on the same qubits.
fn circuit_and_pauli() -> impl Strategy<Value = (Circuit, PauliString)> {
    circuit().prop_flat_map(|circ| {
        let n = circ.n_qubits();
        let string = prop::collection::vec(pauli(), n)
            .prop_map(|ps| PauliString::new(ps.into_iter().enumerate()).unwrap());
        (Just(circ), string)
    })
}

proptest! {
    #[test]
    fn evolution_preserves_norm(circ in circuit()) {
        let state = Statevector::from_circuit(&circ);
        prop_assert!((state.norm_sqr() - 1.0).abs() < TOL);
    }

    #[test]
    fn inverse_returns_to_zero_state(circ in circuit()) {
        let mut state = Statevector::from_circuit(&circ);
        state.run(&circ.inverse()).unwrap();
        assert_close(&state, &Statevector::zero(circ.n_qubits()))?;
    }

    #[test]
    fn kernels_agree_with_generic_two_qubit_kernel((circ, op) in circuit_and_op()) {
        if let Some(generic) = as_unitary2(&op) {
            let mut a = Statevector::from_circuit(&circ);
            let mut b = a.clone();
            a.apply(&op).unwrap();
            b.apply(&generic).unwrap();
            assert_close(&a, &b)?;
        }
    }

    #[test]
    fn expectation_matches_applying_paulis((circ, string) in circuit_and_pauli()) {
        let state = Statevector::from_circuit(&circ);
        let mut applied = state.clone();
        for &(q, p) in string.ops() {
            let gate = match p {
                Pauli::I => continue,
                Pauli::X => Gate::X,
                Pauli::Y => Gate::Y,
                Pauli::Z => Gate::Z,
            };
            applied.apply(&Operation::Gate { gate, target: q }).unwrap();
        }
        let want = state.inner(&applied).unwrap();
        prop_assert!(want.im.abs() < TOL, "Pauli expectation must be real");
        let got = state.expectation_string(&string).unwrap();
        prop_assert!((got - want.re).abs() < TOL, "{got} != {}", want.re);
    }
}

fn assert_matches_reference(circ: &Circuit) -> Result<(), TestCaseError> {
    let got = Statevector::from_circuit(circ);
    let want = Statevector::from_amplitudes(reference_run(circ)).unwrap();
    assert_close(&got, &want)
}

proptest! {
    #[test]
    fn small_circuits_match_reference(circ in circuit()) {
        assert_matches_reference(&circ)?;
    }
}

proptest! {
    // 15 qubits takes the cache-blocked path, 16 the parallel kernels, and
    // qubits 12+ the large-stride branch.
    #![proptest_config(ProptestConfig::with_cases(12))]

    #[test]
    fn large_circuits_match_reference(circ in circuit_on(15..=16)) {
        assert_matches_reference(&circ)?;
    }
}

proptest! {
    // From 20 qubits, runs of gates on any few qubits are applied tile by
    // tile, gathering amplitudes from scattered qubits.
    #![proptest_config(ProptestConfig::with_cases(4))]

    #[test]
    fn gathered_tiles_match_reference(circ in circuit_on(20..=21)) {
        assert_matches_reference(&circ)?;
    }
}

proptest! {
    #[test]
    fn fused_circuit_is_equivalent(circ in circuit()) {
        let fused = circ.fused();
        prop_assert!(fused.len() <= circ.len());
        assert_close(&Statevector::from_circuit(&fused), &Statevector::from_circuit(&circ))?;
    }

    #[test]
    fn partially_fused_circuit_is_equivalent(circ in circuit(), cutoff in 0usize..6) {
        let fused = circ.fused_where(|op| op.max_qubit() >= cutoff);
        prop_assert!(fused.len() <= circ.len());
        assert_close(&Statevector::from_circuit(&fused), &Statevector::from_circuit(&circ))?;
    }
}
