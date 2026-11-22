//! Integration & property tests for lossless circuit optimization.

mod common;

use common::{TOL, kron};
use ketrs::library::{ghz, qft};
use ketrs::{
    Basis, Circuit, Condition, Gate, Operation, OptimizationLevel, OptimizeOptions, Pauli,
    PauliString, PauliSum, RealStatevector, Statevector,
};
use proptest::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;

fn assert_unitary_equal(a: &Circuit, b: &Circuit) {
    assert_eq!(a.n_qubits(), b.n_qubits());
    let n = a.n_qubits();
    let dim = 1usize << n;
    // Check on every computational basis state (full unitary + global phase!).
    for idx in 0..dim {
        let mut sa = Statevector::basis(n, idx);
        let mut sb = Statevector::basis(n, idx);
        sa.run(a).unwrap();
        sb.run(b).unwrap();
        for (k, (x, y)) in sa.amplitudes().iter().zip(sb.amplitudes()).enumerate() {
            assert!(
                (x - y).norm() < 1e-10,
                "unitary mismatch on basis |{idx:0n$b}⟩ amp {k}: {x} vs {y}\norig: {a:?}\nopt:  {b:?}"
            );
        }
    }
}

fn count_2q(circ: &Circuit) -> usize {
    circ.ops().iter().filter(|op| op.qubits().len() >= 2).count()
}

#[test]
fn qft_times_inverse_cancels_to_empty() {
    let q = qft(5);
    let mut circ = q.clone();
    circ.extend(&q.inverse()).unwrap();

    for level in [
        OptimizationLevel::Level1,
        OptimizationLevel::Level2,
        OptimizationLevel::Level3,
    ] {
        let opt = circ.optimize_level(level);
        assert_eq!(opt.len(), 0, "level {level:?} left ops: {:?}", opt.ops());
        assert_eq!(opt.depth(), 0);
        assert_unitary_equal(&circ, &opt);
    }
}

#[test]
fn commuting_cancellations_match_or_beat_qiskit() {
    // H(0) CX(0,1) Rz(0.4, 0) X(1) CX(0,2) Rz(-0.4, 0) CX(0,2) X(1) CX(0,1) H(0) -> empty!
    let mut circ = Circuit::new(3);
    circ.h(0)
        .cx(0, 1)
        .rz(0.4, 0)
        .x(1)
        .cx(0, 2)
        .rz(-0.4, 0)
        .cx(0, 2)
        .x(1)
        .cx(0, 1)
        .h(0);

    for basis in [Basis::Native, Basis::Cx] {
        for level in [OptimizationLevel::Level2, OptimizationLevel::Level3] {
            let opt = circ.optimize_with(&OptimizeOptions {
                level,
                basis,
                ..Default::default()
            });
            assert_eq!(opt.len(), 0, "{level:?} {basis:?}: {:?}", opt.ops());
            assert_unitary_equal(&circ, &opt);
        }
    }
}

#[test]
fn two_qubit_block_collapses_via_kak() {
    // 6 CXs + arbitrary 1q gates on 2 qubits -> at most 3 CXs!
    let mut circ = Circuit::new(2);
    for k in 0..6 {
        let f = (k + 1) as f64;
        circ.rx(0.31 * f, 0).ry(-0.47 * f, 1).rz(0.19 * f, 0).cx(0, 1);
    }
    for basis in [Basis::Native, Basis::Cx] {
        let opt = circ.optimize_with(&OptimizeOptions {
            level: OptimizationLevel::Level2,
            basis,
            ..Default::default()
        });
        assert!(count_2q(&opt) <= 3, "got {} 2q gates", count_2q(&opt));
        assert_unitary_equal(&circ, &opt);
    }
}

#[test]
fn disentangled_two_qubit_block_splits_to_zero_2q_gates() {
    // CX(0, 1) · (Z(0) ⊗ X(1)) · CX(0, 1) = Z(0) ⊗ X(1) (0 two-qubit gates!).
    let mut circ = Circuit::new(2);
    circ.cx(0, 1).z(0).x(1).cx(0, 1);
    let opt = circ.optimize();
    assert_eq!(count_2q(&opt), 0);
    assert_unitary_equal(&circ, &opt);

    // 3 CXs = SWAP, followed by SWAP = Identity!
    let mut c2 = Circuit::new(2);
    c2.cx(0, 1).cx(1, 0).cx(0, 1).swap(0, 1);
    let opt2 = c2.optimize();
    assert_eq!(opt2.len(), 0);
    assert_unitary_equal(&c2, &opt2);
}

#[test]
fn three_qubit_cnot_bridge_reduces_at_level3() {
    // CX(0, 1) CX(1, 2) CX(0, 1) == CX(1, 2) CX(0, 2): 3 CXs -> 2 CXs!
    let mut circ = Circuit::new(3);
    circ.cx(0, 1).cx(1, 2).cx(0, 1);
    let opt = circ.optimize_level(OptimizationLevel::Level3);
    assert_eq!(count_2q(&opt), 2, "got {:?}", opt.ops());
    assert_unitary_equal(&circ, &opt);

    // And a 6-CX identity network on 4 qubits -> 0 CXs!
    let mut id_net = Circuit::new(4);
    id_net
        .cx(0, 1)
        .cx(1, 2)
        .cx(2, 3)
        .cx(0, 1)
        .cx(0, 2)
        .cx(1, 2)
        .cx(0, 3)
        .cx(1, 3)
        .cx(2, 3);
    let opt_net = id_net.optimize_level(OptimizationLevel::Level3);
    assert!(opt_net.len() < id_net.len());
    assert_unitary_equal(&id_net, &opt_net);
}

#[test]
fn clifford_pauli_conjugation_and_phase_gadgets() {
    // CX(0,1) Rz(0.3, 1) CX(0,1) CX(0,1) Rz(0.5, 1) CX(0,1) -> single Rzz(0.8, 0, 1)!
    let mut circ = Circuit::new(2);
    circ.cx(0, 1)
        .rz(0.3, 1)
        .cx(0, 1)
        .cx(0, 1)
        .rz(0.5, 1)
        .cx(0, 1);
    let opt_native = circ.optimize();
    assert_eq!(opt_native.len(), 1, "expected single Rzz, got {:?}", opt_native.ops());
    assert_unitary_equal(&circ, &opt_native);

    // In Basis::Cx, merges to 2 CXs + 1 Rz (3 gates total instead of 6).
    let opt_cx = circ.optimize_with(&OptimizeOptions {
        level: OptimizationLevel::Level2,
        basis: Basis::Cx,
        ..Default::default()
    });
    assert_eq!(count_2q(&opt_cx), 2);
    assert_unitary_equal(&circ, &opt_cx);

    // H(0) H(1) Rzz(0.6, 0, 1) H(0) H(1) -> Rxx(0.6, 0, 1) (5 gates -> 1 gate at L3!).
    let mut conj = Circuit::new(2);
    conj.h(0).h(1).rzz(0.6, 0, 1).h(0).h(1);
    let opt_conj = conj.optimize_level(OptimizationLevel::Level3);
    assert_eq!(opt_conj.len(), 1, "got {:?}", opt_conj.ops());
    assert_unitary_equal(&conj, &opt_conj);

    // H(0) H(1) CX(0, 1) H(0) H(1) -> CX(1, 0) (5 gates -> 1 gate!).
    let mut hcx = Circuit::new(2);
    hcx.h(0).h(1).cx(0, 1).h(0).h(1);
    let opt_hcx = hcx.optimize();
    assert_eq!(opt_hcx.len(), 1, "got {:?}", opt_hcx.ops());
    assert_unitary_equal(&hcx, &opt_hcx);
}

#[test]
fn parametric_circuits_optimize_and_preserve_bindings() {
    let mut circ = Circuit::new(3);
    let p = circ.parameters(2);
    // Rz(p0) and Rz(2*p0 + 0.3) separated by commuting CX(0, 1) merge!
    // CX(1, 2) Rz(p1, 2) CX(1, 2) -> Rzz(p1, 1, 2)!
    // Rx(p1) Rx(-p1) cancels completely!
    circ.rz(p[0], 0)
        .cx(0, 1)
        .rz(2.0 * p[0] + 0.3, 0)
        .cx(1, 2)
        .rz(p[1], 2)
        .cx(1, 2)
        .rx(p[1], 0)
        .rx(-p[1], 0)
        .h(2)
        .h(2);

    let opt = circ.optimize_level(OptimizationLevel::Level3);
    assert_eq!(opt.n_parameters(), 2);
    assert!(opt.len() < circ.len(), "orig={} opt={}", circ.len(), opt.len());

    // Verify exact unitary equality across multiple parameter bindings.
    for vals in [[0.0, 0.0], [0.7, -1.2], [2.3, 0.41], [-0.9, 3.1]] {
        let bound_orig = circ.bind(&vals).unwrap();
        let bound_opt = opt.bind(&vals).unwrap();
        assert_unitary_equal(&bound_orig, &bound_opt);
    }
}

#[test]
fn real_circuits_stay_runnable_on_real_statevector() {
    let mut circ = ghz(4);
    circ.ry(0.3, 0)
        .ry(0.5, 0)
        .h(1)
        .cx(1, 2)
        .h(1)
        .cx(2, 3)
        .cx(3, 2)
        .cx(2, 3)
        .z(0)
        .z(0);
    for level in [
        OptimizationLevel::Level1,
        OptimizationLevel::Level2,
        OptimizationLevel::Level3,
    ] {
        let opt = circ.optimize_level(level);
        let a = RealStatevector::from_circuit(&circ).unwrap();
        let b = RealStatevector::from_circuit(&opt).unwrap();
        for (x, y) in a.amplitudes().iter().zip(b.amplitudes()) {
            assert!((x - y).abs() < TOL, "level {level:?}: {x} != {y}");
        }
    }
}

#[test]
fn multi_controlled_and_trotter_circuits_optimize() {
    // Symmetric CCX(0, 1, 2) and CCX(1, 0, 2) cancel across commuting Z(0).
    let mut circ = Circuit::new(3);
    circ.ccx(0, 1, 2).z(0).ccx(1, 0, 2);
    let opt = circ.optimize();
    assert_eq!(opt.len(), 1);
    assert_unitary_equal(&circ, &opt);

    // CSWAP(0, 1, 2) and CSWAP(0, 2, 1) cancel.
    let mut cs = Circuit::new(3);
    cs.cswap(0, 1, 2).rz(0.5, 0).cswap(0, 2, 1);
    let opt_cs = cs.optimize();
    assert_eq!(opt_cs.len(), 1);
    assert_unitary_equal(&cs, &opt_cs);

    // Commuting Trotter evolution merges duplicate Pauli terms!
    let h = PauliSum::from_list([("ZZI", 0.5), ("IZZ", 0.5), ("ZZI", 0.5)]).unwrap();
    let mut trot = Circuit::new(3);
    trot.pauli_evolution(&h, 1.0, 2, 1).unwrap();
    let opt_trot = trot.optimize();
    assert_eq!(opt_trot.len(), 2, "got {:?}", opt_trot.ops());
    assert_unitary_equal(&trot, &opt_trot);
}

#[test]
fn pauli_evolution_cx_basis_and_inspection_metrics() {
    // Y1 = {IY, YI, ZY, YZ} (all real, 4 raw CXs -> <= 2 CXs under basis=Cx!).
    let h_y1 = PauliSum::from_list([
        ("IY", 0.3),
        ("YI", 0.3),
        ("ZY", 0.3),
        ("YZ", 0.3),
    ])
    .unwrap();
    let mut y1 = Circuit::new(2);
    y1.pauli_evolution(&h_y1, 0.25, 1, 1).unwrap();
    assert_eq!(y1.two_qubit_depth(), 4);

    let y1_cx = y1.optimize_with(&OptimizeOptions {
        level: OptimizationLevel::Level3,
        basis: Basis::Cx,
        ..Default::default()
    });
    assert!(
        *y1_cx.gate_counts().get("cx").unwrap_or(&0) <= 2,
        "Y1 CX count = {:?}",
        y1_cx.gate_counts()
    );
    assert!(y1_cx.two_qubit_depth() <= 2);
    assert_unitary_equal(&y1, &y1_cx);

    // Real-preserving mode on Y1 ALSO achieves <= 2 CXs thanks to real-core 2-CX KAK!
    let y1_real = y1.optimize_with(&OptimizeOptions {
        level: OptimizationLevel::Level3,
        basis: Basis::Cx,
        preserve_real: Some(true),
        ..Default::default()
    });
    assert!(*y1_real.gate_counts().get("cx").unwrap_or(&0) <= 2);
    let _ = RealStatevector::from_circuit(&y1_real).unwrap();
    assert_unitary_equal(&y1, &y1_real);

    // 4-qubit 16-Pauli n_Y1_2p block (48 raw CX -> <= 16 CXs, beating Qiskit O3!).
    let h_ny1 = PauliSum::from_list([
        ("IIIY", 1.0 / 12.0),
        ("IIYI", 1.0 / 12.0),
        ("IIYZ", 1.0 / 12.0),
        ("IIZY", 1.0 / 12.0),
        ("IYII", 1.0 / 12.0),
        ("IYZZ", -1.0 / 12.0),
        ("YIII", 1.0 / 12.0),
        ("YIZZ", -1.0 / 12.0),
        ("YZII", 1.0 / 12.0),
        ("YZZZ", -1.0 / 12.0),
        ("ZYII", 1.0 / 12.0),
        ("ZYZZ", -1.0 / 12.0),
        ("ZZIY", -1.0 / 12.0),
        ("ZZYI", -1.0 / 12.0),
        ("ZZYZ", -1.0 / 12.0),
        ("ZZZY", -1.0 / 12.0),
    ])
    .unwrap();
    let mut ny1 = Circuit::new(4);
    ny1.pauli_evolution(&h_ny1, 0.25, 1, 1).unwrap();
    let ny1_cx = ny1.optimize_with(&OptimizeOptions {
        level: OptimizationLevel::Level3,
        basis: Basis::Cx,
        ..Default::default()
    });
    let cx = *ny1_cx.gate_counts().get("cx").unwrap_or(&0);
    assert!(cx <= 16, "expected <= 16 CXs on n_Y1_2p, got {cx}");
    assert_eq!(*ny1_cx.gate_counts().get("pauli_rotation").unwrap_or(&0), 0);
    assert_unitary_equal(&ny1, &ny1_cx);

    // Parametric multi-qubit PauliRotation run under Basis::Cx preserves parameters.
    let mut pc = Circuit::new(4);
    let t = pc.parameter();
    pc.pauli_rotation(t, PauliString::from_label("XXYY").unwrap())
        .pauli_rotation(2.0 * t, PauliString::from_label("XYXY").unwrap())
        .pauli_rotation(-t + 0.3, PauliString::from_label("ZXYI").unwrap());
    let pc_opt = pc.optimize_with(&OptimizeOptions {
        level: OptimizationLevel::Level3,
        basis: Basis::Cx,
        ..Default::default()
    });
    assert_eq!(pc_opt.n_parameters(), 1);
    assert_eq!(*pc_opt.gate_counts().get("pauli_rotation").unwrap_or(&0), 0);
    for val in [0.0, 0.4, -1.3] {
        assert_unitary_equal(&pc.bind(&[val]).unwrap(), &pc_opt.bind(&[val]).unwrap());
    }
}

#[test]
fn non_unitary_and_control_flow_optimization() {
    // Reset followed by diagonal gate & controlled gate from reset qubit, plus
    // diagonal gates before terminal measure, plus redundant gates inside if_test.
    let mut circ = Circuit::with_clbits(2, 2);
    circ.h(0)
        .h(0)
        .x(0)
        .measure(0, 0)
        .reset(0)
        .reset(0)
        .z(0)
        .cx(0, 1);
    let mut branch = Circuit::with_clbits(2, 2);
    branch.x(1).z(1).z(1);
    circ.if_test(
        Condition {
            clbits: vec![0],
            value: 1,
        },
        branch,
        None,
    );
    circ.rz(0.7, 1).measure(1, 1);

    let opt = circ.optimize();
    assert!(opt.len() < circ.len());
    // Original sets q0=1 -> c0=1, resets q0 -> 0, branch flips q1 -> 1 -> c1=1 => "11".
    let c_orig = Statevector::zero(2)
        .sample_circuit(&circ, 200, &mut StdRng::seed_from_u64(1))
        .unwrap();
    let c_opt = Statevector::zero(2)
        .sample_circuit(&opt, 200, &mut StdRng::seed_from_u64(1))
        .unwrap();
    assert_eq!(c_orig, c_opt);
}

// --- Property tests across random circuits ---

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

fn pauli() -> impl Strategy<Value = Pauli> {
    prop_oneof![
        Just(Pauli::I),
        Just(Pauli::X),
        Just(Pauli::Y),
        Just(Pauli::Z)
    ]
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

fn small_circuit() -> impl Strategy<Value = Circuit> {
    (2usize..=4).prop_flat_map(|n| {
        prop::collection::vec(operation(n), 0..35).prop_map(move |ops| {
            let mut circ = Circuit::new(n);
            for op in ops {
                circ.push(op).unwrap();
            }
            circ
        })
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn optimize_is_strictly_unitary_exact_all_levels(circ in small_circuit()) {
        for basis in [Basis::Native, Basis::Cx] {
            for level in [
                OptimizationLevel::Level1,
                OptimizationLevel::Level2,
                OptimizationLevel::Level3,
            ] {
                let opt = circ.optimize_with(&OptimizeOptions {
                    level,
                    basis,
                    ..Default::default()
                });
                assert_unitary_equal(&circ, &opt);
            }
        }
    }
}
