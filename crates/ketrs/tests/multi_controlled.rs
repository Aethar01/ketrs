//! Gates and swaps with any number of controls.

mod common;

use common::{assert_state_eq, reference_run};
use ketrs::{Circuit, Error, Gate, Operation, RealStatevector, Statevector};

/// A random-looking state on `n` qubits, from a fixed circuit.
fn scrambled(n: usize, seed: u64) -> Circuit {
    let mut circ = Circuit::new(n);
    for layer in 0..3 {
        for q in 0..n {
            let a = 0.37 * (seed as f64 + 1.0) + 0.11 * (q + 7 * layer) as f64;
            circ.ry(a, q).rz(1.3 * a, q);
        }
        for q in 0..n - 1 {
            circ.cx(q, q + 1);
        }
    }
    circ
}

fn basis_prep(n: usize, index: usize) -> Circuit {
    let mut circ = Circuit::new(n);
    for q in 0..n {
        if index >> q & 1 == 1 {
            circ.x(q);
        }
    }
    circ
}

/// The basis state a circuit sends `|index⟩` to, asserting it is one.
fn image(circ: &Circuit, index: usize) -> usize {
    let mut full = basis_prep(circ.n_qubits(), index);
    full.extend(circ).unwrap();
    let state = Statevector::from_circuit(&full);
    let (out, a) = state
        .amplitudes()
        .iter()
        .enumerate()
        .max_by(|x, y| x.1.norm().total_cmp(&y.1.norm()))
        .unwrap();
    assert!((a.norm() - 1.0).abs() < 1e-12, "not a basis state");
    out
}

#[test]
fn toffoli_truth_table() {
    // Controls above and below the target.
    for (a, b, t) in [(0, 1, 2), (2, 0, 1), (1, 2, 0)] {
        let mut circ = Circuit::new(3);
        circ.ccx(a, b, t);
        for i in 0..8 {
            let flips = i >> a & 1 == 1 && i >> b & 1 == 1;
            let want = if flips { i ^ 1 << t } else { i };
            assert_eq!(image(&circ, i), want, "ccx({a}, {b}, {t}) on {i:03b}");
        }
    }
}

#[test]
fn fredkin_truth_table() {
    for (c, a, b) in [(0, 1, 2), (2, 0, 1), (1, 2, 0)] {
        let mut circ = Circuit::new(3);
        circ.cswap(c, a, b);
        for i in 0..8 {
            let differ = (i >> a & 1) != (i >> b & 1);
            let want = if i >> c & 1 == 1 && differ {
                i ^ (1 << a | 1 << b)
            } else {
                i
            };
            assert_eq!(image(&circ, i), want, "cswap({c}, {a}, {b}) on {i:03b}");
        }
    }
}

/// Toffoli in Clifford+T.
fn toffoli_decomposed(circ: &mut Circuit, a: usize, b: usize, c: usize) {
    circ.h(c)
        .cx(b, c)
        .tdg(c)
        .cx(a, c)
        .t(c)
        .cx(b, c)
        .tdg(c)
        .cx(a, c);
    circ.t(b).t(c).h(c).cx(a, b).t(a).tdg(b).cx(a, b);
}

#[test]
fn toffoli_matches_its_decomposition_on_random_states() {
    // 16 qubits runs the parallel kernel; high controls and target too.
    for (n, a, b, t) in [(4, 0, 3, 1), (7, 6, 2, 4), (16, 15, 3, 9), (16, 0, 1, 15)] {
        let mut native = scrambled(n, 1);
        native.ccx(a, b, t);
        let mut decomposed = scrambled(n, 1);
        toffoli_decomposed(&mut decomposed, a, b, t);
        assert_state_eq(
            &Statevector::from_circuit(&native),
            Statevector::from_circuit(&decomposed).amplitudes(),
        );
    }
}

#[test]
fn many_controls_match_a_plain_loop() {
    for n in [5, 17] {
        let mut circ = scrambled(n, 2);
        circ.mcx(&[0, 2, n - 1], 1)
            .multi_controlled(
                Gate::U {
                    theta: 0.7,
                    phi: 0.2,
                    lambda: -0.4,
                },
                &[1, 2],
                n - 2,
            )
            .controlled_swap(&[3, 0], 2, n - 1)
            .multi_controlled(Gate::Ry(0.9), &[0, 1, 2, 3], 4);
        assert_state_eq(&Statevector::from_circuit(&circ), &reference_run(&circ));
    }
}

#[test]
fn zero_controls_are_the_plain_gate() {
    let mut gate = scrambled(4, 3);
    gate.multi_controlled(Gate::H, &[], 2)
        .controlled_swap(&[], 0, 3);
    let mut plain = scrambled(4, 3);
    plain.h(2).swap(0, 3);
    assert_state_eq(
        &Statevector::from_circuit(&gate),
        Statevector::from_circuit(&plain).amplitudes(),
    );
}

#[test]
fn invalid_control_sets_are_rejected() {
    let bad = [
        Operation::MultiControlled {
            gate: Gate::X,
            controls: vec![0, 2],
            target: 2,
        },
        Operation::MultiControlled {
            gate: Gate::X,
            controls: vec![1, 1],
            target: 0,
        },
        Operation::MultiControlled {
            gate: Gate::X,
            controls: vec![5],
            target: 0,
        },
        Operation::ControlledSwap {
            controls: vec![1],
            a: 1,
            b: 2,
        },
        Operation::ControlledSwap {
            controls: vec![0],
            a: 2,
            b: 2,
        },
    ];
    for op in bad {
        assert!(Circuit::new(3).push(op.clone()).is_err(), "{op:?}");
        assert!(Statevector::zero(3).apply(&op).is_err(), "{op:?}");
    }
}

#[test]
fn inverses_undo_and_real_states_work() {
    let mut circ = scrambled(5, 4);
    circ.multi_controlled(Gate::T, &[0, 1], 3)
        .controlled_swap(&[2, 3], 0, 4)
        .mcx(&[4], 1);
    let mut there_and_back = circ.clone();
    there_and_back.extend(&circ.inverse()).unwrap();
    assert!(
        (Statevector::from_circuit(&there_and_back).amplitudes()[0].norm() - 1.0).abs() < 1e-12
    );

    // Real gates run on real states; complex ones are refused.
    let mut real = Circuit::new(3);
    real.h(0).h(1).ccx(0, 1, 2).cswap(2, 0, 1);
    let r = RealStatevector::from_circuit(&real).unwrap();
    let c = Statevector::from_circuit(&real);
    for (x, y) in r.amplitudes().iter().zip(c.amplitudes()) {
        assert!((x - y.re).abs() < 1e-12);
    }
    let mut complex = Circuit::new(3);
    complex.multi_controlled(Gate::S, &[0, 1], 2);
    assert!(matches!(
        RealStatevector::from_circuit(&complex),
        Err(Error::NotReal(_))
    ));
}

#[test]
fn density_matrices_and_unsupported_engines() {
    let mut circ = scrambled(3, 5);
    circ.ccx(0, 1, 2).cswap(2, 1, 0);
    let mut rho = ketrs::DensityMatrix::zero(3);
    rho.run(&circ).unwrap();
    let psi = Statevector::from_circuit(&circ);
    assert!((rho.fidelity(&psi).unwrap() - 1.0).abs() < 1e-10);
    // Engines without the operations say so instead of misbehaving.
    assert!(matches!(
        ketrs::Mps::from_circuit(&circ, ketrs::MpsOptions::default()),
        Err(Error::InvalidArgument(_))
    ));
}

#[test]
fn add_qubit_appends_a_zero_qubit() {
    let circ = scrambled(3, 6);
    let mut state = Statevector::from_circuit(&circ);
    state.add_qubit();
    assert_eq!(state.n_qubits(), 4);
    let mut wider = Circuit::new(4);
    wider
        .extend(&{
            let mut c = Circuit::new(4);
            for op in circ.ops() {
                c.push(op.clone()).unwrap();
            }
            c
        })
        .unwrap();
    assert_state_eq(&state, Statevector::from_circuit(&wider).amplitudes());
    // The new qubit is usable straight away.
    let mut more = Circuit::new(4);
    more.ccx(0, 1, 3);
    state.run(&more).unwrap();
    assert!((state.norm_sqr() - 1.0).abs() < 1e-12);
}
