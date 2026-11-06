//! Measurement, reset and classical control inside circuits.

use ketrs::{Circuit, Condition, Error, Instruction, RealStatevector, Statevector};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::collections::BTreeMap;

fn rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

/// `count` is within 5σ of `shots · p`.
fn assert_binomial(count: usize, shots: usize, p: f64) {
    let sigma = (shots as f64 * p * (1.0 - p)).sqrt().max(1.0);
    let expected = shots as f64 * p;
    assert!(
        (count as f64 - expected).abs() < 5.0 * sigma,
        "{count} vs {expected:.0} ± {sigma:.0}"
    );
}

#[test]
fn terminal_measurement_of_a_bell_pair() {
    let mut circ = Circuit::new(2);
    circ.h(0).cx(0, 1).measure_all();
    assert_eq!(circ.n_clbits(), 2);
    let shots = 10_000;
    let counts = Statevector::zero(2)
        .sample_circuit(&circ, shots, &mut rng(1))
        .unwrap();
    assert_eq!(counts.keys().copied().collect::<Vec<_>>(), [0b00, 0b11]);
    assert_binomial(counts[&0b11], shots, 0.5);
}

#[test]
fn measurements_can_permute_bits() {
    // Qubit 0 into bit 2, qubit 2 into bit 0.
    let mut circ = Circuit::with_clbits(3, 3);
    circ.x(0).measure(0, 2).measure(2, 0).measure(1, 1);
    let counts = Statevector::zero(3)
        .sample_circuit(&circ, 50, &mut rng(2))
        .unwrap();
    assert_eq!(counts, BTreeMap::from([(0b100, 50)]));
}

/// Teleport Ry(θ)|0⟩ from qubit 0 to qubit 2 with mid-circuit measurement
/// and classically controlled corrections.
fn teleportation(theta: f64) -> Circuit {
    let mut circ = Circuit::with_clbits(3, 3);
    circ.ry(theta, 0);
    circ.h(1).cx(1, 2);
    circ.cx(0, 1).h(0);
    circ.measure(0, 0).measure(1, 1);
    let mut x = Circuit::with_clbits(3, 3);
    x.x(2);
    let mut z = Circuit::with_clbits(3, 3);
    z.z(2);
    circ.if_test(
        Condition {
            clbits: vec![1],
            value: 1,
        },
        x,
        None,
    );
    circ.if_test(
        Condition {
            clbits: vec![0],
            value: 1,
        },
        z,
        None,
    );
    circ.measure(2, 2);
    circ
}

#[test]
fn teleportation_reproduces_the_input_state() {
    let theta = 1.1;
    let circ = teleportation(theta);
    assert!(!circ.is_unitary());
    let shots = 20_000;
    let counts = Statevector::zero(3)
        .sample_circuit(&circ, shots, &mut rng(3))
        .unwrap();
    let ones: usize = counts
        .iter()
        .filter(|(bits, _)| *bits >> 2 & 1 == 1)
        .map(|(_, c)| c)
        .sum();
    assert_binomial(ones, shots, (theta / 2.0).sin().powi(2));
    // The two measured bits are uniformly random.
    for low in 0..4u64 {
        let n: usize = counts
            .iter()
            .filter(|(bits, _)| *bits & 0b11 == low)
            .map(|(_, c)| c)
            .sum();
        assert_binomial(n, shots, 0.25);
    }
}

#[test]
fn trajectories_collapse_the_state() {
    let mut circ = Circuit::with_clbits(2, 2);
    circ.h(0).cx(0, 1).measure(0, 0);
    for seed in 0..20 {
        let mut state = Statevector::zero(2);
        let bits = state.run_trajectory(&circ, &mut rng(seed)).unwrap();
        // After measuring qubit 0 the pair is a basis state matching it.
        let p1 = state.probability_one(1).unwrap();
        assert!((p1 - if bits[0] { 1.0 } else { 0.0 }).abs() < 1e-12);
    }
}

#[test]
fn reset_and_repeated_measurement() {
    let mut circ = Circuit::with_clbits(1, 3);
    circ.h(0).measure(0, 0).measure(0, 1).reset(0).measure(0, 2);
    let counts = Statevector::zero(1)
        .sample_circuit(&circ, 2_000, &mut rng(4))
        .unwrap();
    // Repeated measurement agrees; after reset the qubit reads 0.
    assert_eq!(counts.keys().copied().collect::<Vec<_>>(), [0b000, 0b011]);
}

#[test]
fn register_conditions_and_else_branches() {
    // Write 0b10 into bits (0, 1), then branch on the pair as a register.
    let mut circ = Circuit::with_clbits(3, 3);
    circ.x(1).measure(0, 0).measure(1, 1);
    let mut then = Circuit::with_clbits(3, 3);
    then.x(2);
    let mut otherwise = Circuit::with_clbits(3, 3);
    otherwise.h(2);
    let register = |value| Condition {
        clbits: vec![0, 1],
        value,
    };
    let mut a = circ.clone();
    a.if_test(register(2), then.clone(), Some(otherwise.clone()))
        .measure(2, 2);
    let mut b = circ.clone();
    b.if_test(register(3), then, Some(otherwise)).measure(2, 2);
    let shots = 4_000;
    // Condition holds: X, so bit 2 is always 1.
    let counts = Statevector::zero(3)
        .sample_circuit(&a, shots, &mut rng(5))
        .unwrap();
    assert_eq!(counts, BTreeMap::from([(0b110, shots)]));
    // Condition fails: H, so bit 2 is a coin flip.
    let counts = Statevector::zero(3)
        .sample_circuit(&b, shots, &mut rng(6))
        .unwrap();
    assert_binomial(counts[&0b110], shots, 0.5);
    assert_eq!(counts[&0b110] + counts[&0b010], shots);
}

#[test]
fn nested_branches() {
    let mut inner = Circuit::with_clbits(2, 2);
    inner.x(1);
    let mut outer_then = Circuit::with_clbits(2, 2);
    outer_then.measure(0, 1);
    outer_then.if_test(
        Condition {
            clbits: vec![1],
            value: 1,
        },
        inner,
        None,
    );
    let mut circ = Circuit::with_clbits(2, 2);
    circ.x(0).measure(0, 0);
    circ.if_test(
        Condition {
            clbits: vec![0],
            value: 1,
        },
        outer_then,
        None,
    );
    circ.measure(1, 1);
    let counts = Statevector::zero(2)
        .sample_circuit(&circ, 10, &mut rng(7))
        .unwrap();
    assert_eq!(counts, BTreeMap::from([(0b11, 10)]));
}

#[test]
fn fusion_never_crosses_instructions() {
    let circ = teleportation(0.7);
    let fused = circ.fused();
    assert_eq!(fused.instructions().len(), circ.instructions().len());
    // Same outcome distribution as the unfused circuit.
    let shots = 20_000;
    let a = Statevector::zero(3)
        .sample_circuit(&circ, shots, &mut rng(8))
        .unwrap();
    let b = Statevector::zero(3)
        .sample_circuit(&fused, shots, &mut rng(8))
        .unwrap();
    for bits in 0..8u64 {
        let p = *a.get(&bits).unwrap_or(&0) as f64 / shots as f64;
        assert_binomial(*b.get(&bits).unwrap_or(&0), shots, p);
    }
}

#[test]
fn engines_reject_non_unitary_circuits() {
    let mut circ = Circuit::new(1);
    circ.h(0).measure_all();
    assert!(matches!(
        Statevector::zero(1).run(&circ),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        ketrs::Mps::from_circuit(&circ, ketrs::MpsOptions::default()),
        Err(Error::InvalidArgument(_))
    ));
    // Real statevectors run trajectories too.
    let bits = RealStatevector::zero(1)
        .run_trajectory(&circ, &mut rng(9))
        .unwrap();
    assert_eq!(bits.len(), 1);
}

#[test]
fn instructions_are_validated() {
    let mut circ = Circuit::with_clbits(2, 1);
    assert!(
        circ.push_instruction(Instruction::Measure { qubit: 0, clbit: 1 })
            .is_err()
    );
    assert!(
        circ.push_instruction(Instruction::Reset { qubit: 2 })
            .is_err()
    );
    let wrong_width = Circuit::with_clbits(3, 1);
    assert!(
        circ.push_instruction(Instruction::IfElse {
            condition: Condition {
                clbits: vec![0],
                value: 1
            },
            then: Box::new(wrong_width.clone()),
            otherwise: Box::new(wrong_width),
        })
        .is_err()
    );
}
