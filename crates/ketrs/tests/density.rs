//! Density matrices and channels, checked against plain matrix arithmetic.

use ketrs::{
    Channel, Circuit, Complex64 as C64, DensityMatrix, NoiseModel, Pauli, PauliString, PauliSum,
    Statevector,
};

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

/// Every operation kind, with complex and Y-containing rotations.
fn random_circuit(n: usize, depth: usize, seed: u64) -> Circuit {
    let mut rng = Lcg(seed);
    let mut circ = Circuit::new(n);
    let paulis = [Pauli::I, Pauli::X, Pauli::Y, Pauli::Z];
    for _ in 0..depth {
        let (a, b) = (rng.below(n), rng.below(n));
        let b = if a == b { (a + 1) % n } else { b };
        let angle = rng.next() * 6.0;
        match rng.below(7) {
            0 => circ.u(angle, rng.next(), rng.next(), a),
            1 => circ.cx(a, b),
            2 => circ.cp(angle, a, b),
            3 => circ.swap(a, b),
            4 => circ.ryy(angle, a, b),
            5 => {
                let string = PauliString::new((0..n).map(|q| (q, paulis[rng.below(4)]))).unwrap();
                circ.pauli_rotation(angle, string)
            }
            _ => circ.s(a).t(b).crx(angle, a, b),
        };
    }
    circ.set_global_phase(0.7);
    circ
}

fn assert_matrix_close(got: &[C64], want: &[C64], tol: f64) {
    for (k, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).norm() < tol, "element {k}: {g} vs {w}");
    }
}

/// `ρ = Σ pᵢ |ψᵢ⟩⟨ψᵢ|` for a few random pure states.
fn mixed_state(n: usize, seed: u64) -> Vec<C64> {
    let dim = 1 << n;
    let mut rho = vec![C64::new(0.0, 0.0); dim * dim];
    for (k, p) in [0.5, 0.3, 0.2].into_iter().enumerate() {
        let psi = Statevector::from_circuit(&random_circuit(n, 12, seed + k as u64));
        let a = psi.amplitudes();
        for i in 0..dim {
            for j in 0..dim {
                rho[i * dim + j] += p * a[i] * a[j].conj();
            }
        }
    }
    rho
}

/// `Σ K ρ K†` with each `K` acting on `qubits` of an `n`-qubit `ρ`.
fn apply_kraus(rho: &[C64], n: usize, kraus: &[Vec<C64>], qubits: &[usize]) -> Vec<C64> {
    let dim = 1 << n;
    let k_dim = 1 << qubits.len();
    // The full operator of K on `qubits`.
    let full = |k: &[C64]| -> Vec<C64> {
        let local = |i: usize| {
            qubits
                .iter()
                .enumerate()
                .fold(0, |acc, (b, &q)| acc | (i >> q & 1) << b)
        };
        let mask: usize = qubits.iter().map(|q| 1 << q).sum();
        let mut out = vec![C64::new(0.0, 0.0); dim * dim];
        for r in 0..dim {
            for c in 0..dim {
                if r & !mask == c & !mask {
                    out[r * dim + c] = k[local(r) * k_dim + local(c)];
                }
            }
        }
        out
    };
    let mul = |a: &[C64], b: &[C64]| -> Vec<C64> {
        let mut out = vec![C64::new(0.0, 0.0); dim * dim];
        for i in 0..dim {
            for k in 0..dim {
                let x = a[i * dim + k];
                if x != C64::new(0.0, 0.0) {
                    for j in 0..dim {
                        out[i * dim + j] += x * b[k * dim + j];
                    }
                }
            }
        }
        out
    };
    let mut out = vec![C64::new(0.0, 0.0); dim * dim];
    for k in kraus {
        let kf = full(k);
        let dagger: Vec<C64> = (0..dim * dim)
            .map(|idx| kf[(idx % dim) * dim + idx / dim].conj())
            .collect();
        for (o, x) in out.iter_mut().zip(mul(&mul(&kf, rho), &dagger)) {
            *o += x;
        }
    }
    out
}

#[test]
fn unitary_circuits_give_pure_states() {
    // 8 qubits is a 16-qubit vector: fused and tiled.
    for (n, seed) in [(4, 1), (8, 2)] {
        let circ = random_circuit(n, 40, seed);
        let mut rho = DensityMatrix::zero(n);
        rho.run(&circ).unwrap();
        let want = DensityMatrix::from_statevector(&Statevector::from_circuit(&circ));
        assert_matrix_close(&rho.to_matrix(), &want.to_matrix(), 1e-10);
        assert!((rho.purity() - 1.0).abs() < 1e-10);
        assert!((rho.trace() - 1.0).abs() < 1e-10);
    }
}

#[test]
fn channels_match_kraus_sums() {
    let n = 3;
    let rho0 = mixed_state(n, 5);
    let channels = [
        (Channel::depolarizing(0.3, 1).unwrap(), vec![1]),
        (Channel::amplitude_damping(0.4).unwrap(), vec![2]),
        (Channel::phase_damping(0.6).unwrap(), vec![0]),
        (Channel::bit_flip(0.2).unwrap(), vec![1]),
        (Channel::depolarizing(0.5, 2).unwrap(), vec![2, 0]),
        (
            Channel::pauli(2, &[(PauliString::from_label("XY").unwrap(), 0.25)]).unwrap(),
            vec![0, 2],
        ),
    ];
    for (k, (channel, qubits)) in channels.iter().enumerate() {
        let mut rho = DensityMatrix::from_matrix(n, &rho0).unwrap();
        rho.apply_channel(channel, qubits).unwrap();
        let want = apply_kraus(&rho0, n, channel.kraus(), qubits);
        assert_matrix_close(&rho.to_matrix(), &want, 1e-12);
        assert!((rho.trace() - 1.0).abs() < 1e-12, "channel {k}");
    }
}

#[test]
fn standard_channels_do_what_they_say() {
    // Depolarizing with p = 1 leaves the maximally mixed state; at the
    // largest p, 4/3, the identity drops out and |0⟩ goes to diag(1/3, 2/3).
    for (p, want) in [(1.0, [0.5, 0.5]), (4.0 / 3.0, [1.0 / 3.0, 2.0 / 3.0])] {
        let mut rho = DensityMatrix::zero(1);
        rho.apply_channel(&Channel::depolarizing(p, 1).unwrap(), &[0])
            .unwrap();
        let probs = rho.probabilities();
        assert!((probs[0] - want[0]).abs() < 1e-12 && (probs[1] - want[1]).abs() < 1e-12);
    }
    // Amplitude damping moves |1⟩ to |0⟩ with probability γ.
    let mut one = Circuit::new(1);
    one.x(0);
    let mut rho = DensityMatrix::zero(1);
    rho.run(&one).unwrap();
    rho.apply_channel(&Channel::amplitude_damping(0.3).unwrap(), &[0])
        .unwrap();
    assert!((rho.probabilities()[0] - 0.3).abs() < 1e-12);
    // Phase damping shrinks |+⟩'s coherence by √(1 − λ).
    let mut plus = Circuit::new(1);
    plus.h(0);
    let mut rho = DensityMatrix::zero(1);
    rho.run(&plus).unwrap();
    rho.apply_channel(&Channel::phase_damping(0.36).unwrap(), &[0])
        .unwrap();
    assert!((rho.element(0, 1).re - 0.5 * 0.8).abs() < 1e-12);
    // Not trace preserving.
    let half = C64::new(0.5, 0.0);
    let zero = C64::new(0.0, 0.0);
    assert!(Channel::from_kraus(vec![vec![vec![half, zero], vec![zero, half]]]).is_err());
}

#[test]
fn expectation_values_of_mixed_states() {
    let n = 3;
    let rho0 = mixed_state(n, 9);
    let rho = DensityMatrix::from_matrix(n, &rho0).unwrap();
    let h = PauliSum::from_list([
        ("ZZI", 0.7),
        ("XIY", -0.4),
        ("IYY", 0.3),
        ("IIX", 1.1),
        ("III", 0.2),
    ])
    .unwrap();
    // Tr(ρH) directly, from H's matrix.
    let dim = 1 << n;
    let mut want = C64::new(0.0, 0.0);
    for (c, string) in h.terms() {
        for col in 0..dim {
            // P|col⟩ = phase |row⟩, one factor at a time.
            let (mut row, mut phase) = (col, C64::new(*c, 0.0));
            for &(q, p) in string.ops() {
                let bit = col >> q & 1 == 1;
                match p {
                    Pauli::X => row ^= 1 << q,
                    Pauli::Y => {
                        row ^= 1 << q;
                        phase *= if bit {
                            C64::new(0.0, -1.0)
                        } else {
                            C64::new(0.0, 1.0)
                        };
                    }
                    Pauli::Z => {
                        if bit {
                            phase = -phase;
                        }
                    }
                    Pauli::I => {}
                }
            }
            // H[row][col] ρ[col][row]
            want += phase * rho0[col * dim + row];
        }
    }
    assert!((rho.expectation(&h).unwrap() - want.re).abs() < 1e-12);
    // And for a pure state, the statevector's value.
    let circ = random_circuit(n, 20, 3);
    let mut pure = DensityMatrix::zero(n);
    pure.run(&circ).unwrap();
    let psi = Statevector::from_circuit(&circ);
    assert!((pure.expectation(&h).unwrap() - psi.expectation(&h).unwrap()).abs() < 1e-10);
    assert!((pure.fidelity(&psi).unwrap() - 1.0).abs() < 1e-10);
}

#[test]
fn noise_models_apply_after_every_gate() {
    let n = 3;
    let mut circ = Circuit::new(n);
    circ.h(0).cx(0, 1).cx(1, 2);
    let noise = NoiseModel {
        one_qubit: Some(Channel::depolarizing(0.01, 1).unwrap()),
        two_qubit: Some(Channel::depolarizing(0.05, 2).unwrap()),
        ..Default::default()
    };
    let mut noisy = DensityMatrix::zero(n);
    noisy.run_noisy(&circ, &noise).unwrap();
    // The same by hand.
    let mut by_hand = DensityMatrix::zero(n);
    for op in circ.ops() {
        let mut one = Circuit::new(n);
        one.push(op.clone()).unwrap();
        by_hand.run(&one).unwrap();
        let qubits = op.qubits();
        if qubits.len() == 2 {
            by_hand
                .apply_channel(noise.two_qubit.as_ref().unwrap(), &qubits)
                .unwrap();
        } else {
            by_hand
                .apply_channel(noise.one_qubit.as_ref().unwrap(), &qubits)
                .unwrap();
        }
    }
    assert_matrix_close(&noisy.to_matrix(), &by_hand.to_matrix(), 1e-14);
    // Noise lowers the GHZ fidelity below 1 but not by much.
    let ghz = Statevector::from_circuit(&circ);
    let f = noisy.fidelity(&ghz).unwrap();
    assert!(f < 0.99 && f > 0.85, "{f}");
}

/// Noisy runs go through the tiled kernels as one sequence of
/// superoperators; applying each step on its own must agree. 7 qubits is a
/// 14-qubit vector (split tiles), 10 a 20-qubit one (gathered tiles).
#[test]
fn noisy_runs_on_larger_states_match_step_by_step() {
    for n in [7, 10] {
        let mut circ = Circuit::new(n);
        for layer in 0..2 {
            for q in 0..n {
                circ.ry(0.3 + 0.1 * (q + layer) as f64, q);
            }
            for q in 0..n - 1 {
                circ.cx(q, q + 1);
            }
        }
        circ.rzz(0.4, 0, n - 1);
        let noise = NoiseModel {
            one_qubit: Some(Channel::amplitude_damping(0.05).unwrap()),
            two_qubit: Some(Channel::depolarizing(0.03, 2).unwrap()),
            ..Default::default()
        };
        let mut fast = DensityMatrix::zero(n);
        fast.run_noisy(&circ, &noise).unwrap();
        let mut slow = DensityMatrix::zero(n);
        for op in circ.ops() {
            let mut one = Circuit::new(n);
            one.push(op.clone()).unwrap();
            slow.run(&one).unwrap();
            let qubits = op.qubits();
            if qubits.len() == 2 {
                slow.apply_channel(noise.two_qubit.as_ref().unwrap(), &qubits)
                    .unwrap();
            } else {
                slow.apply_channel(noise.one_qubit.as_ref().unwrap(), &qubits)
                    .unwrap();
            }
        }
        assert_matrix_close(&fast.to_matrix(), &slow.to_matrix(), 1e-12);
        assert!((fast.trace() - 1.0).abs() < 1e-10);
    }
}
