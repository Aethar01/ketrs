//! Noisy statevector trajectories, checked against density matrices.

use ketrs::{Channel, Circuit, DensityMatrix, NoiseModel, Statevector};
use rand::SeedableRng;
use rand::rngs::StdRng;

fn layered(n: usize) -> Circuit {
    let mut circ = Circuit::new(n);
    for layer in 0..2 {
        for q in 0..n {
            circ.ry(0.4 + 0.3 * (q + layer) as f64, q)
                .rz(0.2 * q as f64, q);
        }
        for q in 0..n - 1 {
            circ.cx(q, q + 1);
        }
    }
    circ.ccx(0, 1, n - 1).ryy(0.5, 0, 2);
    circ.set_global_phase(0.3);
    circ
}

/// Average the final probabilities of `trajectories` noisy runs.
fn average_probabilities(circ: &Circuit, noise: &NoiseModel, trajectories: usize) -> Vec<f64> {
    let n = circ.n_qubits();
    let mut rng = StdRng::seed_from_u64(5);
    let mut total = vec![0.0; 1 << n];
    for _ in 0..trajectories {
        let mut psi = Statevector::zero(n);
        psi.run_noisy_trajectory(circ, noise, &mut rng).unwrap();
        assert!((psi.norm_sqr() - 1.0).abs() < 1e-10);
        for (t, a) in total.iter_mut().zip(psi.amplitudes()) {
            *t += a.norm_sqr() / trajectories as f64;
        }
    }
    total
}

#[test]
fn trajectories_average_to_the_density_matrix() {
    // A Pauli mixture (sampled up front), a damping channel (sampled from
    // the state), and both.
    let models = [
        NoiseModel {
            one_qubit: Some(Channel::depolarizing(0.05, 1).unwrap()),
            two_qubit: Some(Channel::depolarizing(0.1, 2).unwrap()),
            ..Default::default()
        },
        NoiseModel {
            one_qubit: Some(Channel::amplitude_damping(0.1).unwrap()),
            ..Default::default()
        },
        NoiseModel {
            one_qubit: Some(Channel::phase_damping(0.2).unwrap()),
            two_qubit: Some(Channel::depolarizing(0.08, 2).unwrap()),
            ..Default::default()
        },
    ];
    let circ = layered(4);
    for noise in &models {
        let mut rho = DensityMatrix::zero(4);
        rho.run_noisy(&circ, noise).unwrap();
        let got = average_probabilities(&circ, noise, 4000);
        for (i, (g, w)) in got.iter().zip(rho.probabilities()).enumerate() {
            // Each average has standard error below 0.008.
            assert!((g - w).abs() < 0.025, "{noise:?}: P({i}) {g} vs {w}");
        }
    }
}

#[test]
fn noiseless_trajectories_are_exact() {
    let circ = layered(5);
    let mut psi = Statevector::zero(5);
    psi.run_noisy_trajectory(&circ, &NoiseModel::default(), &mut StdRng::seed_from_u64(1))
        .unwrap();
    let want = Statevector::from_circuit(&circ);
    for (a, b) in psi.amplitudes().iter().zip(want.amplitudes()) {
        assert!((a - b).norm() < 1e-12);
    }
}

#[test]
fn readout_error_flips_recorded_bits() {
    let mut circ = Circuit::with_clbits(2, 2);
    circ.x(0).measure(0, 0).measure(1, 1);
    let noise = NoiseModel {
        readout: Some([0.1, 0.25]),
        ..Default::default()
    };
    let shots = 20_000;
    let counts = Statevector::zero(2)
        .sample_noisy(&circ, &noise, shots, &mut StdRng::seed_from_u64(2))
        .unwrap();
    let freq = |bits: u64| *counts.get(&bits).unwrap_or(&0) as f64 / shots as f64;
    // Qubit 0 is 1 (read 0 with 0.25), qubit 1 is 0 (read 1 with 0.1).
    for (bits, want) in [
        (0b01, 0.75 * 0.9),
        (0b00, 0.25 * 0.9),
        (0b11, 0.75 * 0.1),
        (0b10, 0.025),
    ] {
        assert!(
            (freq(bits) - want).abs() < 0.015,
            "{bits:02b}: {} vs {want}",
            freq(bits)
        );
    }
    // Probabilities outside [0, 1] are refused.
    let bad = NoiseModel {
        readout: Some([0.0, 1.5]),
        ..Default::default()
    };
    assert!(
        Statevector::zero(2)
            .sample_noisy(&circ, &bad, 1, &mut StdRng::seed_from_u64(0))
            .is_err()
    );
}

#[test]
fn noise_follows_classical_control() {
    // Bit-flip noise on the X gate inside a branch: the branch's outcome
    // reads 1 with probability 1 − p.
    let p = 0.2;
    let mut circ = Circuit::with_clbits(2, 2);
    circ.x(0).measure(0, 0);
    let mut then = Circuit::with_clbits(2, 2);
    then.x(1);
    circ.if_test(ketrs::Expr::Bit(0), then, None);
    circ.measure(1, 1);
    let noise = NoiseModel {
        one_qubit: Some(Channel::bit_flip(p).unwrap()),
        ..Default::default()
    };
    let shots = 20_000;
    let counts = Statevector::zero(2)
        .sample_noisy(&circ, &noise, shots, &mut StdRng::seed_from_u64(3))
        .unwrap();
    let freq = |bits: u64| *counts.get(&bits).unwrap_or(&0) as f64 / shots as f64;
    // First flip lands with 1 − p; if so, the second with 1 − p.
    let want = [
        (0b11, (1.0 - p) * (1.0 - p)),
        (0b01, (1.0 - p) * p),
        (0b00, p),
    ];
    for (bits, w) in want {
        assert!(
            (freq(bits) - w).abs() < 0.015,
            "{bits:02b}: {} vs {w}",
            freq(bits)
        );
    }
}
