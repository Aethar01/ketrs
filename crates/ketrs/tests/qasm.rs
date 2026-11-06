//! OpenQASM import, checked against circuits built directly.

mod common;

use common::assert_state_eq;
use ketrs::{Circuit, Complex64 as C64, Error, Statevector, qasm};
use rand::SeedableRng;
use rand::rngs::StdRng;

fn state(source: &str) -> Statevector {
    Statevector::from_circuit(&qasm::parse(source).unwrap())
}

#[test]
fn standard_gates_match_circuit_methods() {
    let source = r#"
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[3] q;
        h q[0]; x q[1]; y q[2]; z q[0]; s q[1]; sdg q[2]; t q[0]; tdg q[1];
        sx q[2]; sxdg q[0];
        rx(0.1) q[0]; ry(pi/3) q[1]; rz(-2*π/5) q[2]; p(0.4) q[0]; u1(0.2) q[1];
        u2(0.3, 0.5) q[2]; u3(0.1, 0.2, 0.3) q[0]; U(0.4, 0.5, 0.6) q[1];
        cx q[0], q[1]; CX q[1], q[2]; cy q[2], q[0]; cz q[0], q[2]; ch q[1], q[0];
        cp(0.7) q[0], q[1]; crx(0.8) q[1], q[2]; cry(0.9) q[2], q[0]; crz(1.0) q[0], q[2];
        cu(0.1, 0.2, 0.3, 0.4) q[1], q[0];
        swap q[0], q[2]; rxx(0.3) q[0], q[1]; ryy(0.4) q[1], q[2]; rzz(0.5) q[0], q[2];
        id q[1]; barrier q;
        gphase(0.25);
    "#;
    let (pi, sq) = (std::f64::consts::PI, C64::new(0.5, 0.5));
    let mut circ = Circuit::new(3);
    circ.h(0).x(1).y(2).z(0).s(1).sdg(2).t(0).tdg(1);
    circ.unitary([[sq, sq.conj()], [sq.conj(), sq]], 2);
    circ.unitary([[sq.conj(), sq], [sq, sq.conj()]], 0);
    circ.rx(0.1, 0)
        .ry(pi / 3.0, 1)
        .rz(-2.0 * pi / 5.0, 2)
        .p(0.4, 0)
        .p(0.2, 1);
    circ.u(pi / 2.0, 0.3, 0.5, 2)
        .u(0.1, 0.2, 0.3, 0)
        .u(0.4, 0.5, 0.6, 1);
    circ.cx(0, 1)
        .cx(1, 2)
        .cy(2, 0)
        .cz(0, 2)
        .controlled(ketrs::Gate::H, 1, 0);
    circ.cp(0.7, 0, 1)
        .crx(0.8, 1, 2)
        .cry(0.9, 2, 0)
        .crz(1.0, 0, 2);
    let (s, c) = (0.05f64.sin(), 0.05f64.cos());
    let g = C64::cis(0.4);
    circ.controlled(
        ketrs::Gate::Unitary([
            [g * c, -g * C64::cis(0.3) * s],
            [g * C64::cis(0.2) * s, g * C64::cis(0.5) * c],
        ]),
        1,
        0,
    );
    circ.swap(0, 2).rxx(0.3, 0, 1).ryy(0.4, 1, 2).rzz(0.5, 0, 2);
    circ.set_global_phase(0.25);
    assert_state_eq(
        &state(source),
        Statevector::from_circuit(&circ).amplitudes(),
    );
}

#[test]
fn toffoli_and_fredkin_are_exact() {
    // Every input basis state of three qubits.
    for input in 0..8usize {
        let prep: String = (0..3)
            .filter(|q| input >> q & 1 == 1)
            .map(|q| format!("x q[{q}];"))
            .collect();
        let toffoli = state(&format!("qubit[3] q; {prep} ccx q[0], q[1], q[2];"));
        let want = if input & 3 == 3 { input ^ 4 } else { input };
        assert!(
            (toffoli.amplitudes()[want].norm() - 1.0).abs() < 1e-12,
            "ccx {input}"
        );
        let fredkin = state(&format!("qubit[3] q; {prep} cswap q[0], q[1], q[2];"));
        let want = if input & 1 == 1 {
            (input & 1) | (input >> 2 & 1) << 1 | (input >> 1 & 1) << 2
        } else {
            input
        };
        assert!(
            (fredkin.amplitudes()[want].norm() - 1.0).abs() < 1e-12,
            "cswap {input}"
        );
    }
}

#[test]
fn custom_gates_modifiers_and_broadcasting() {
    let source = r#"
        OPENQASM 3;
        include "stdgates.inc";
        gate rot(a, b) x, y { ry(a) x; cx x, y; rz(2 * b + 0.1) y; }
        gate twice q { pow(2) @ s q; }
        qubit[2] q;
        qubit r;
        h q;
        rot(0.3, -0.4) q[0], q[1];
        inv @ rot(0.2, 0.5) q[1], r;
        ctrl @ x r, q[0];
        ctrl @ ry(0.6) q[1], r;
        ctrl @ h q[0], r;
        twice r;
    "#;
    let mut circ = Circuit::new(3);
    circ.h(0).h(1);
    circ.ry(0.3, 0).cx(0, 1).rz(2.0 * -0.4 + 0.1, 1);
    let mut rot = Circuit::new(3);
    rot.ry(0.2, 1).cx(1, 2).rz(2.0 * 0.5 + 0.1, 2);
    circ.extend(&rot.inverse()).unwrap();
    circ.cx(2, 0)
        .cry(0.6, 1, 2)
        .controlled(ketrs::Gate::H, 0, 2);
    circ.s(2).s(2);
    assert_state_eq(
        &state(source),
        Statevector::from_circuit(&circ).amplitudes(),
    );
}

#[test]
fn control_modifiers_on_any_gate() {
    let source = r#"
        OPENQASM 3.0;
        include "stdgates.inc";
        gate rot(a) x, y { ry(a) x; cx x, y; rzz(a / 2) x, y; gphase(0.3); }
        qubit[5] q;
        h q;
        ctrl(2) @ x q[0], q[3], q[1];
        negctrl @ ctrl @ h q[4], q[0], q[2];
        ctrl(2) @ rot(0.7) q[1], q[2], q[3], q[4];
        ctrl @ inv @ rot(-0.4) q[0], q[1], q[2];
        ctrl @ swap q[4], q[0], q[1];
        c3x q[0], q[1], q[2], q[3];
        c4x q[4], q[3], q[2], q[1], q[0];
        ctrl @ ctrl @ rx(0.2) q[2], q[3], q[4];
    "#;
    let mut rot = Circuit::new(5);
    rot.ry(0.7, 3).cx(3, 4).rzz(0.35, 3, 4);
    rot.set_global_phase(0.3);
    let mut back = Circuit::new(5);
    back.ry(-0.4, 1).cx(1, 2).rzz(-0.2, 1, 2);
    back.set_global_phase(0.3);
    let mut fixed = Circuit::new(5);
    for q in 0..5 {
        fixed.h(q);
    }
    fixed.ccx(0, 3, 1);
    fixed.x(4).multi_controlled(ketrs::Gate::H, &[4, 0], 2).x(4);
    fixed.extend(&rot.control(&[1, 2]).unwrap()).unwrap();
    fixed
        .extend(&back.inverse().control(&[0]).unwrap())
        .unwrap();
    fixed
        .cswap(4, 0, 1)
        .mcx(&[0, 1, 2], 3)
        .mcx(&[4, 3, 2, 1], 0);
    fixed.multi_controlled(ketrs::Gate::Rx(0.2), &[2, 3], 4);
    assert_state_eq(
        &state(source),
        Statevector::from_circuit(&fixed).amplitudes(),
    );
}

#[test]
fn control_matches_controlled_gates() {
    // Circuit::control against the dense controlled unitary: |0⟩ on the
    // control leaves the state alone, |1⟩ applies the circuit.
    let p = std::f64::consts::PI;
    let mut body = Circuit::new(4);
    let theta = body.parameter();
    body.h(1)
        .rx(theta, 2)
        .cx(1, 3)
        .rzz(0.4, 2, 3)
        .pauli_rotation(0.3, ketrs::PauliString::from_label("XYZI").unwrap());
    body.swap(1, 2).ccx(1, 2, 3).p(p / 3.0, 1);
    body.pauli_rotation(theta, ketrs::PauliString::from_label("I").unwrap());
    body.set_global_phase(0.25);
    let controlled = body.control(&[0]).unwrap();
    for value in [0.3, -1.1] {
        for c in [false, true] {
            let mut prep = Circuit::new(4);
            prep.parameter();
            prep.ry(0.4, 1).ry(1.2, 2).ry(-0.7, 3);
            if c {
                prep.x(0);
            }
            let mut got = prep.clone();
            got.extend(&controlled).unwrap();
            let mut want = prep.clone();
            if c {
                want.extend(&body).unwrap();
            }
            let bind = |circ: &Circuit| Statevector::from_circuit(&circ.bind(&[value]).unwrap());
            assert_state_eq(&bind(&got), bind(&want).amplitudes());
        }
    }
    assert!(body.control(&[1]).is_err());
    assert!(body.control(&[9]).is_err());
    assert_eq!(body.control(&[]).unwrap(), body);
}

#[test]
fn relative_phase_toffolis() {
    // rccx and rc3x act as Toffoli on basis states, up to phases.
    for (name, n) in [("rccx", 3), ("rc3x", 4)] {
        for input in 0..1usize << n {
            let prep: String = (0..n)
                .filter(|q| input >> q & 1 == 1)
                .map(|q| format!("x q[{q}];"))
                .collect();
            let operands: Vec<String> = (0..n).map(|q| format!("q[{q}]")).collect();
            let s = state(&format!(
                "qubit[{n}] q; {prep} {name} {};",
                operands.join(", ")
            ));
            let all = (1 << (n - 1)) - 1;
            let want = if input & all == all {
                input ^ 1 << (n - 1)
            } else {
                input
            };
            assert!(
                (s.amplitudes()[want].norm() - 1.0).abs() < 1e-12,
                "{name} {input}"
            );
        }
    }
}

#[test]
fn subroutines_are_inlined() {
    let source = r#"
        OPENQASM 3.0;
        include "stdgates.inc";
        input float theta;
        qubit[3] r;
        qubit s;
        def entangle(qubit[3] reg, float[64] a) {
            h reg[0];
            for uint i in [0:1] { cx reg[i], reg[i + 1]; }
            rz(a) reg[2];
        }
        def pair(qubit a, qubit b) { cx a, b; }
        entangle(r, theta / 2);
        pair(r[2], s);
        entangle(r, 0.3);
    "#;
    let circ = qasm::parse(source).unwrap();
    let mut want = Circuit::new(4);
    let p = want.parameter();
    want.h(0).cx(0, 1).cx(1, 2).rz(0.5 * p, 2);
    want.cx(2, 3);
    want.h(0).cx(0, 1).cx(1, 2).rz(0.3, 2);
    for value in [0.4, -1.0] {
        assert_state_eq(
            &Statevector::from_circuit(&circ.bind(&[value]).unwrap()),
            Statevector::from_circuit(&want.bind(&[value]).unwrap()).amplitudes(),
        );
    }

    // Measurements and conditions inside a subroutine.
    let source = r#"
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        bit c;
        def copy(qubit from, qubit to) {
            c = measure from;
            if (c) { x to; }
        }
        x q[0];
        copy(q[0], q[1]);
    "#;
    let circ = qasm::parse(source).unwrap();
    let mut s = Statevector::zero(2);
    s.run_trajectory(&circ, &mut StdRng::seed_from_u64(1))
        .unwrap();
    assert!((s.amplitudes()[3].norm() - 1.0).abs() < 1e-12);
}

#[test]
fn inputs_become_parameters() {
    let source = r#"
        OPENQASM 3.0;
        include "stdgates.inc";
        input float[64] theta;
        input angle phi;
        qubit[2] q;
        rx(theta) q[0];
        ry(2 * phi - pi / 4) q[1];
        crz(theta / 2) q[0], q[1];
    "#;
    let circ = qasm::parse(source).unwrap();
    assert_eq!(circ.n_parameters(), 2);
    let values = [0.7, -0.3];
    let mut want = Circuit::new(2);
    want.rx(0.7, 0)
        .ry(2.0 * -0.3 - std::f64::consts::FRAC_PI_4, 1)
        .crz(0.35, 0, 1);
    assert_state_eq(
        &Statevector::from_circuit(&circ.bind(&values).unwrap()),
        Statevector::from_circuit(&want).amplitudes(),
    );
}

#[test]
fn openqasm2_programs() {
    let source = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q -> c;
        if (c == 3) x q[0];
    "#;
    let circ = qasm::parse(source).unwrap();
    assert_eq!((circ.n_qubits(), circ.n_clbits()), (2, 2));
    let counts = Statevector::zero(2)
        .sample_circuit(&circ, 500, &mut StdRng::seed_from_u64(1))
        .unwrap();
    assert_eq!(counts.keys().copied().collect::<Vec<_>>(), [0b00, 0b11]);
}

#[test]
fn control_flow_and_expressions() {
    // Repeat-until-success, a loop over qubits, and expression conditions.
    let source = r#"
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[4] q;
        bit[2] c;
        bit flag;
        h q[0];
        flag = measure q[0];
        while (!flag) {
            reset q[0];
            h q[0];
            flag = measure q[0];
        }
        for uint i in [1:2] {
            x q[i];
        }
        c[0] = measure q[1];
        c[1] = measure q[2];
        if (c == 3 && flag) {
            x q[3];
        } else {
            z q[3];
        }
        for int k in {0, 3} { }
    "#;
    let circ = qasm::parse(source).unwrap();
    let mut s = Statevector::zero(4);
    let bits = s
        .run_trajectory(&circ, &mut StdRng::seed_from_u64(3))
        .unwrap();
    assert_eq!(bits, [true, true, true]);
    assert!((s.probability_one(3).unwrap() - 1.0).abs() < 1e-12);
}

#[test]
fn errors_name_the_line() {
    let cases = [
        ("qubit[2] q;\nfoo q[0];", 2, "unknown gate"),
        ("qubit[2] q;\nh q[5];", 2, "out of range"),
        ("qubit q;\nlet x = q;", 2, "not supported"),
        ("qubit q;\ndef f(qubit a) -> bit { }", 2, "return values"),
        (
            "qubit[2] q;\ndef f(qubit a, qubit b) { cx a, b; }\nf(q[0], q[0]);",
            3,
            "twice",
        ),
        ("qubit q;\ndef f(qubit a) { f(a); }\nf(q);", 2, "recurses"),
        ("input float t;\nqubit q;\nrx(t * t) q;", 3, "linear"),
        ("qubit[2] q;\ncx q[0] q[1];", 2, "expected `;`"),
        ("qubit q;\nif (true) { qubit r; }", 2, "top level"),
    ];
    for (source, want_line, text) in cases {
        match qasm::parse(source) {
            Err(Error::Parse { line, message }) => {
                assert_eq!(line, want_line, "{source}: {message}");
                assert!(message.contains(text), "{source}: {message}");
            }
            other => panic!("{source}: {other:?}"),
        }
    }
}

/// The 4×4 matrix of a two-qubit circuit, column `j` the image of `|j⟩`.
fn matrix_of(circ: &Circuit) -> [[C64; 4]; 4] {
    let columns: Vec<Statevector> = (0..4)
        .map(|j| {
            let mut s = Statevector::basis(2, j);
            s.run(circ).unwrap();
            s
        })
        .collect();
    std::array::from_fn(|i| std::array::from_fn(|j| columns[j].amplitudes()[i]))
}

fn round_trip(circ: &Circuit) -> Circuit {
    let source = circ.to_qasm().unwrap();
    qasm::parse(&source).unwrap_or_else(|e| panic!("{e}\n{source}"))
}

#[test]
fn export_round_trips_every_operation() {
    let mut body = Circuit::new(2);
    body.h(0).ry(0.7, 1).cx(0, 1).t(1).rzx(0.4, 0, 1).s(0);
    body.set_global_phase(0.9);
    let u2 = matrix_of(&body);
    let mut circ = Circuit::new(5);
    let p = circ.parameters(2);
    circ.set_global_phase(-0.3);
    circ.h(0).x(1).y(2).z(3).s(4).sdg(0).t(1).tdg(2);
    circ.rx(p[0], 0)
        .ry(2.0 * p[1] - 0.5, 1)
        .rz(0.3, 2)
        .p(-0.25 * p[0], 3);
    circ.u(0.1, 0.2, 0.3, 4);
    circ.unitary(
        [
            [C64::new(0.0, 0.6), C64::new(0.8, 0.0)],
            [C64::new(0.8, 0.0), C64::new(0.0, 0.6)],
        ],
        0,
    );
    circ.cx(0, 1)
        .cy(1, 2)
        .cz(2, 3)
        .controlled(ketrs::Gate::H, 3, 4);
    circ.crx(p[1], 4, 0)
        .cry(0.2, 0, 2)
        .crz(0.4, 1, 3)
        .cp(p[0], 2, 4);
    circ.controlled(ketrs::Gate::T, 0, 4).controlled(
        ketrs::Gate::U {
            theta: 0.5,
            phi: -0.1,
            lambda: 0.7,
        },
        3,
        1,
    );
    circ.ccx(0, 1, 2)
        .mcx(&[4, 3, 1], 0)
        .cswap(0, 3, 4)
        .controlled_swap(&[1, 2], 0, 4);
    circ.multi_controlled(ketrs::Gate::Ry(0.6), &[0, 1], 3);
    circ.multi_controlled(
        ketrs::Gate::Unitary([
            [C64::new(0.0, 0.6), C64::new(0.8, 0.0)],
            [C64::new(0.8, 0.0), C64::new(0.0, 0.6)],
        ]),
        &[2, 4, 0],
        1,
    );
    circ.swap(1, 3).rzz(p[1], 0, 2).rxx(0.3, 1, 4);
    circ.pauli_rotation(0.8, ketrs::PauliString::from_label("YZXI").unwrap());
    circ.pauli_rotation(p[0], ketrs::PauliString::identity());
    circ.push(ketrs::Operation::Unitary2 {
        matrix: u2,
        qubits: [3, 1],
    })
    .unwrap();
    let back = round_trip(&circ);
    assert_eq!(back.n_parameters(), 2);
    for values in [[0.3, -0.7], [1.9, 0.4]] {
        assert_state_eq(
            &Statevector::from_circuit(&back.bind(&values).unwrap()),
            Statevector::from_circuit(&circ.bind(&values).unwrap()).amplitudes(),
        );
    }
}

#[test]
fn two_qubit_unitaries_decompose_exactly() {
    // Generic, diagonal, permutation and single-column cases, on both
    // qubit orders and under a control.
    let mut generic = Circuit::new(2);
    generic
        .h(0)
        .ry(1.1, 1)
        .cx(1, 0)
        .rz(0.4, 0)
        .cy(0, 1)
        .u(0.3, 0.9, -0.2, 1);
    let mut diagonal = Circuit::new(2);
    diagonal.rz(0.3, 0).cp(0.8, 0, 1).p(-0.4, 1);
    let mut permutation = Circuit::new(2);
    permutation.swap(0, 1).cx(0, 1).x(1);
    for body in [generic, diagonal, permutation] {
        let m = matrix_of(&body);
        for qubits in [[0, 2], [2, 0]] {
            let mut circ = Circuit::new(3);
            circ.h(0).ry(0.4, 1).ry(1.3, 2);
            circ.push(ketrs::Operation::Unitary2 { matrix: m, qubits })
                .unwrap();
            assert_state_eq(
                &Statevector::from_circuit(&round_trip(&circ)),
                Statevector::from_circuit(&circ).amplitudes(),
            );
            // Controlled on qubit 1, against applying it only on |1⟩.
            let mut inner = Circuit::new(3);
            inner
                .push(ketrs::Operation::Unitary2 { matrix: m, qubits })
                .unwrap();
            for c in [false, true] {
                let mut prep = Circuit::new(3);
                prep.h(0).ry(1.3, 2);
                if c {
                    prep.x(1);
                }
                let mut got = prep.clone();
                got.extend(&inner.control(&[1]).unwrap()).unwrap();
                let mut want = prep.clone();
                if c {
                    want.extend(&inner).unwrap();
                }
                assert_state_eq(
                    &Statevector::from_circuit(&got),
                    Statevector::from_circuit(&want).amplitudes(),
                );
            }
        }
    }
}

#[test]
fn export_round_trips_classical_control() {
    use ketrs::{BinaryOp, Condition, Expr};
    let mut circ = Circuit::with_clbits(3, 3);
    circ.h(0).measure(0, 0);
    let mut fix = Circuit::with_clbits(3, 3);
    fix.reset(0).h(0).measure(0, 0);
    circ.while_loop(Expr::Not(Box::new(Expr::Bit(0))), fix);
    circ.h(1).measure(1, 1);
    let mut then = Circuit::with_clbits(3, 3);
    then.x(2);
    then.set_global_phase(0.2);
    let mut otherwise = Circuit::with_clbits(3, 3);
    otherwise.h(2);
    // A partial register, written bit by bit, and a whole one.
    circ.if_test(
        Condition {
            clbits: vec![0, 1],
            value: 3,
        },
        then,
        Some(otherwise),
    );
    let mut flip = Circuit::with_clbits(3, 3);
    flip.z(2);
    circ.if_test(
        Expr::binary(
            BinaryOp::Or,
            Expr::binary(
                BinaryOp::Less,
                Expr::Register(vec![0, 1, 2]),
                Expr::Value(2),
            ),
            Expr::binary(
                BinaryOp::NotEqual,
                Expr::Register(vec![1, 2]),
                Expr::Value(1),
            ),
        ),
        flip,
        None,
    );
    circ.measure(2, 2);
    let back = round_trip(&circ);
    let counts = |c: &Circuit| {
        Statevector::zero(3)
            .sample_circuit(c, 400, &mut StdRng::seed_from_u64(9))
            .unwrap()
    };
    assert_eq!(counts(&back), counts(&circ));

    // Other uses of partial registers can't be written.
    let mut bad = Circuit::with_clbits(1, 2);
    bad.if_test(
        Expr::binary(BinaryOp::Less, Expr::Register(vec![1]), Expr::Value(1)),
        Circuit::with_clbits(1, 2),
        None,
    );
    assert!(bad.to_qasm().is_err());
}
