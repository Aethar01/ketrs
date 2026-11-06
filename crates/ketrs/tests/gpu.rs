//! GPU statevectors, checked against the CPU engine. Tests skip (passing)
//! when no Vulkan GPU with 64-bit floats is available.
#![cfg(feature = "gpu")]

mod common;

use ketrs::{
    Circuit, Complex64 as C64, Error, GpuContext, GpuPrecision, GpuStatevector, Pauli, PauliString,
    PauliSum, Statevector,
};
use std::sync::Arc;

fn gpu() -> Option<Arc<GpuContext>> {
    match GpuContext::new() {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            eprintln!("skipping GPU test: {e}");
            None
        }
    }
}

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

/// Every operation kind the GPU plans: single, controlled, swap, two-qubit
/// unitaries (from fusion), Pauli rotations with and without X, and phase.
fn random_circuit(n: usize, depth: usize, seed: u64) -> Circuit {
    let mut rng = Lcg(seed);
    let mut circ = Circuit::new(n);
    let paulis = [Pauli::I, Pauli::X, Pauli::Y, Pauli::Z];
    for _ in 0..depth {
        let (a, b) = (rng.below(n), rng.below(n));
        let b = if a == b { (a + 1) % n } else { b };
        let angle = rng.next() * 6.0;
        match rng.below(8) {
            0 => circ.u(angle, rng.next(), rng.next(), a),
            1 => circ.cx(a, b),
            2 => circ.cp(angle, a, b),
            3 => circ.swap(a, b),
            4 => circ.rxx(angle, a, b),
            5 => {
                let string = PauliString::new((0..n).map(|q| (q, paulis[rng.below(4)]))).unwrap();
                circ.pauli_rotation(angle, string)
            }
            6 => {
                let string = PauliString::new((0..n).map(|q| {
                    (
                        q,
                        if rng.below(2) == 0 {
                            Pauli::Z
                        } else {
                            Pauli::I
                        },
                    )
                }))
                .unwrap();
                circ.pauli_rotation(angle, string)
            }
            _ => circ.h(a).ry(angle, b).t(a),
        };
    }
    circ.set_global_phase(0.7);
    circ
}

fn assert_close(got: &[C64], want: &[C64]) {
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).norm() < 1e-10, "amplitude {i}: GPU {g}, CPU {w}");
    }
}

#[test]
fn random_circuits_match_cpu() {
    let Some(ctx) = gpu() else { return };
    for (n, seed) in [(2, 1), (5, 2), (9, 3), (12, 4), (20, 5)] {
        let circ = random_circuit(n, 60, seed);
        let gpu_state = GpuStatevector::from_circuit(&ctx, &circ).unwrap();
        assert_close(
            &gpu_state.amplitudes().unwrap(),
            Statevector::from_circuit(&circ).amplitudes(),
        );
    }
}

#[test]
fn multi_controlled_operations_match_cpu() {
    let Some(ctx) = gpu() else { return };
    // 5 qubits runs gate by gate, 13 in tiles.
    for (n, seed) in [(5, 6), (13, 7)] {
        let mut circ = random_circuit(n, 30, seed);
        circ.ccx(0, n - 1, 2)
            .cswap(1, n - 1, 3)
            .controlled_swap(&[0, 2, 4], 1, n - 2)
            .multi_controlled(ketrs::Gate::Ry(0.3), &[1, 3], 0)
            .multi_controlled(ketrs::Gate::H, &[], 4);
        circ.extend(&random_circuit(n, 30, seed + 1)).unwrap();
        let gpu_state = GpuStatevector::from_circuit(&ctx, &circ).unwrap();
        assert_close(
            &gpu_state.amplitudes().unwrap(),
            Statevector::from_circuit(&circ).amplitudes(),
        );
    }
}

#[test]
fn upload_run_download_round_trip() {
    let Some(ctx) = gpu() else { return };
    let n = 10;
    let start = Statevector::from_circuit(&random_circuit(n, 30, 9));
    let mut gpu_state = GpuStatevector::from_amplitudes(&ctx, start.amplitudes()).unwrap();
    let circ = random_circuit(n, 30, 10);
    gpu_state.run(&circ).unwrap();
    let mut cpu = start.clone();
    cpu.run(&circ).unwrap();
    assert_close(&gpu_state.amplitudes().unwrap(), cpu.amplitudes());
    let h = PauliSum::from_list([("ZZIIIIIIII", 1.0), ("IIIIXYIIII", 0.4)]).unwrap();
    assert!((gpu_state.expectation(&h).unwrap() - cpu.expectation(&h).unwrap()).abs() < 1e-10);
}

#[test]
fn oversized_states_are_rejected() {
    let Some(ctx) = gpu() else { return };
    assert!(matches!(GpuStatevector::zero(&ctx, 40), Err(Error::Gpu(_))));
}

/// Long runs of low-qubit gates go through the shared-memory tile kernel;
/// repeat to catch synchronisation races, which show up as run-to-run
/// differences.
#[test]
fn tiled_runs_are_correct_every_time() {
    let Some(ctx) = gpu() else { return };
    let n = 16;
    let mut circ = random_circuit(9, 200, 21);
    // Widen to 16 qubits: the random gates stay on qubits 0–8 (inside a
    // tile), with a few gates on high qubits splitting the runs.
    let mut wide = Circuit::new(n);
    for (i, op) in circ.ops().iter().enumerate() {
        wide.push(op.clone()).unwrap();
        if i % 50 == 49 {
            wide.h(15).cx(14, 3);
        }
    }
    wide.set_global_phase(circ.global_phase());
    circ = wide;
    let want = Statevector::from_circuit(&circ);
    for _ in 0..5 {
        let got = GpuStatevector::from_circuit(&ctx, &circ).unwrap();
        assert_close(&got.amplitudes().unwrap(), want.amplitudes());
    }
}

fn hamiltonian(n: usize, seed: u64) -> PauliSum {
    let mut rng = Lcg(seed);
    let paulis = [Pauli::I, Pauli::X, Pauli::Y, Pauli::Z];
    let mut h = PauliSum::new();
    for _ in 0..12 {
        let string = PauliString::new((0..n).map(|q| (q, paulis[rng.below(4)]))).unwrap();
        h.add(rng.next() * 2.0 - 1.0, string);
    }
    h.add(0.3, PauliString::identity());
    h
}

/// States split over several buffers, forced on small states so the
/// multi-buffer kernels are checked against the CPU.
#[test]
fn chunked_states_match_cpu() {
    let Some(ctx) = gpu() else { return };
    let n = 12;
    // chunk_bits 8 uses all sixteen buffers.
    for chunk_bits in [8, 9, 10, 11] {
        let circ = random_circuit(n, 80, chunk_bits as u64);
        let mut gpu_state = GpuStatevector::zero_with_chunk_bits(&ctx, n, chunk_bits).unwrap();
        gpu_state.run(&circ).unwrap();
        let cpu = Statevector::from_circuit(&circ);
        assert_close(&gpu_state.amplitudes().unwrap(), cpu.amplitudes());
        let h = hamiltonian(n, 7);
        let got = gpu_state.expectation(&h).unwrap();
        assert!(
            (got - cpu.expectation(&h).unwrap()).abs() < 1e-10,
            "chunk_bits {chunk_bits}"
        );
        assert!((gpu_state.norm_sqr().unwrap() - 1.0).abs() < 1e-10);
    }
}

#[test]
fn expectation_on_gpu_matches_cpu() {
    let Some(ctx) = gpu() else { return };
    for (n, seed) in [(3, 1), (11, 2), (16, 3)] {
        let circ = random_circuit(n, 60, seed);
        let mut gpu_state = GpuStatevector::from_circuit(&ctx, &circ).unwrap();
        let cpu = Statevector::from_circuit(&circ);
        let h = hamiltonian(n, seed + 10);
        let (got, want) = (
            gpu_state.expectation(&h).unwrap(),
            cpu.expectation(&h).unwrap(),
        );
        assert!((got - want).abs() < 1e-10, "n={n}: GPU {got}, CPU {want}");
    }
}

#[test]
fn sampling_on_gpu_follows_born_rule() {
    use rand::SeedableRng;
    let Some(ctx) = gpu() else { return };
    let n = 12;
    let circ = random_circuit(n, 40, 31);
    let probs = Statevector::from_circuit(&circ).probabilities();
    let mut rng = rand::rngs::StdRng::seed_from_u64(5);
    let shots = 200_000;
    for chunk_bits in [12, 10] {
        let mut gpu_state = GpuStatevector::zero_with_chunk_bits(&ctx, n, chunk_bits).unwrap();
        gpu_state.run(&circ).unwrap();
        let counts = gpu_state.sample_counts(shots, &mut rng).unwrap();
        assert_eq!(counts.values().sum::<usize>(), shots);
        // Every observed count within 5σ of its binomial expectation.
        for (&index, &count) in &counts {
            let p = probs[index];
            let sigma = (shots as f64 * p * (1.0 - p)).sqrt().max(1.0);
            assert!(
                (count as f64 - shots as f64 * p).abs() < 5.0 * sigma + 1.0,
                "index {index}"
            );
        }
        // States of negligible probability are (almost) never drawn.
        let impossible: usize = counts
            .iter()
            .filter(|(i, _)| probs[**i] < 1e-12)
            .map(|(_, c)| c)
            .sum();
        assert_eq!(impossible, 0);
    }
}

#[test]
fn pauli_sums_apply_on_gpu() {
    let Some(ctx) = gpu() else { return };
    for (n, chunk_bits, seed) in [(5, 5, 1), (12, 12, 2), (12, 10, 3)] {
        let circ = random_circuit(n, 40, seed);
        let mut gpu_state = GpuStatevector::zero_with_chunk_bits(&ctx, n, chunk_bits).unwrap();
        gpu_state.run(&circ).unwrap();
        let h = hamiltonian(n, seed + 20);
        let got = gpu_state.apply_pauli_sum(&h).unwrap();
        let want = Statevector::from_circuit(&circ)
            .apply_pauli_sum(&h)
            .unwrap();
        assert_close(&got.amplitudes().unwrap(), want.amplitudes());
    }
    // H = 0 gives the zero vector.
    let mut gpu_state = GpuStatevector::zero(&ctx, 3).unwrap();
    let zero = gpu_state.apply_pauli_sum(&PauliSum::new()).unwrap();
    assert!(zero.amplitudes().unwrap().iter().all(|a| a.norm() == 0.0));
}

/// A parametric circuit using every parametrisable gate, parameters shared
/// through linear forms.
fn parametric(n: usize, seed: u64) -> (Circuit, Vec<f64>) {
    let mut rng = Lcg(seed);
    let mut circ = Circuit::new(n);
    let params = circ.parameters(5);
    circ.set_global_phase(0.4);
    for layer in 0..3 {
        for q in 0..n {
            let p = params[rng.below(params.len())];
            let angle = (rng.next() * 2.0 - 1.0) * p + rng.next();
            let b = (q + 1 + layer) % n;
            match rng.below(9) {
                0 => circ.rx(angle, q),
                1 => circ.ry(angle, q),
                2 => circ.rz(angle, q),
                3 => circ.p(angle, q),
                4 => circ.crx(angle, q, b),
                5 => circ.cry(angle, b, q),
                6 => circ.crz(angle, q, b),
                7 => circ.rzz(angle, q, b),
                _ => circ.ryy(angle, q, b),
            };
            circ.h(q).cx(q, b);
        }
    }
    let values = (0..params.len()).map(|_| rng.next() * 4.0 - 2.0).collect();
    (circ, values)
}

#[test]
fn gradients_on_gpu_match_cpu() {
    let Some(ctx) = gpu() else { return };
    // 7 qubits runs gate by gate, 13 in tiles, and chunk_bits 11 splits the
    // state over buffers.
    for (n, chunk_bits, seed) in [(7, 7, 1), (13, 13, 2), (13, 11, 3)] {
        let (circ, values) = parametric(n, seed);
        let h = hamiltonian(n, seed + 30);
        let start = GpuStatevector::zero_with_chunk_bits(&ctx, n, chunk_bits).unwrap();
        let (value, gradient) = start.expectation_gradient(&circ, &values, &h).unwrap();
        let (want_value, want) = Statevector::zero(n)
            .expectation_gradient(&circ, &values, &h)
            .unwrap();
        assert!(
            (value - want_value).abs() < 1e-10,
            "{value} vs {want_value}"
        );
        for (g, w) in gradient.iter().zip(&want) {
            assert!((g - w).abs() < 1e-10, "n={n}: {gradient:?} vs {want:?}");
        }
    }
}

#[test]
fn gradients_through_multi_controlled_operations() {
    let Some(ctx) = gpu() else { return };
    let (mut circ, values) = parametric(7, 4);
    let p = circ.parameters(2);
    circ.ccx(0, 3, 6).cswap(1, 2, 5).ry(p[0], 6).crx(p[1], 5, 1);
    circ.ccx(6, 5, 0).h(0);
    let values = [values, vec![0.3, -0.8]].concat();
    let h = hamiltonian(7, 40);
    let start = GpuStatevector::zero(&ctx, 7).unwrap();
    let (_, gradient) = start.expectation_gradient(&circ, &values, &h).unwrap();
    let (_, want) = Statevector::zero(7)
        .expectation_gradient(&circ, &values, &h)
        .unwrap();
    for (g, w) in gradient.iter().zip(&want) {
        assert!((g - w).abs() < 1e-10, "{gradient:?} vs {want:?}");
    }
}

#[test]
fn adjoint_gradient_on_gpu_for_an_overlap_loss() {
    let Some(ctx) = gpu() else { return };
    let n = 6;
    let (circ, values) = parametric(n, 9);
    let target = Statevector::from_circuit(&random_circuit(n, 30, 4));
    let psi = Statevector::from_circuit(&circ.bind(&values).unwrap());
    // L = |⟨φ|ψ⟩|², λ = |φ⟩⟨φ|ψ⟩.
    let overlap = target.inner(&psi).unwrap();
    let cotangent: Vec<C64> = target.amplitudes().iter().map(|a| a * overlap).collect();
    let want = Statevector::adjoint_gradient(
        &circ,
        &values,
        &psi,
        &Statevector::from_amplitudes(cotangent.clone()).unwrap(),
    )
    .unwrap();
    let got = GpuStatevector::adjoint_gradient(
        &circ,
        &values,
        &GpuStatevector::from_amplitudes(&ctx, psi.amplitudes()).unwrap(),
        &GpuStatevector::from_amplitudes(&ctx, &cotangent).unwrap(),
    )
    .unwrap();
    for (g, w) in got.iter().zip(&want) {
        assert!((g - w).abs() < 1e-10, "{got:?} vs {want:?}");
    }
}

#[test]
fn evolution_on_gpu_matches_cpu() {
    let Some(ctx) = gpu() else { return };
    // Long enough for several Krylov steps; 12 qubits with chunk_bits 10
    // splits the state over buffers.
    for (n, chunk_bits, seed, time) in [(6, 6, 1, 0.3), (12, 12, 2, 4.0), (12, 10, 3, 2.5)] {
        let circ = random_circuit(n, 40, seed);
        let mut gpu_state = GpuStatevector::zero_with_chunk_bits(&ctx, n, chunk_bits).unwrap();
        gpu_state.run(&circ).unwrap();
        let h = hamiltonian(n, seed + 40);
        gpu_state.evolve(&h, time).unwrap();
        let mut cpu = Statevector::from_circuit(&circ);
        cpu.evolve(&h, time).unwrap();
        assert_close(&gpu_state.amplitudes().unwrap(), cpu.amplitudes());
        assert!((gpu_state.norm_sqr().unwrap() - 1.0).abs() < 1e-10);
    }
}

#[test]
fn too_many_buffers_are_rejected() {
    let Some(ctx) = gpu() else { return };
    // 12 qubits in buffers of 2⁷ would need 32 buffers.
    assert!(matches!(
        GpuStatevector::zero_with_chunk_bits(&ctx, 12, 7),
        Err(Error::Gpu(_))
    ));
}

fn assert_close_single(got: &[C64], want: &[C64]) {
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).norm() < 2e-5, "amplitude {i}: GPU {g}, CPU {w}");
    }
}

#[test]
fn single_precision_matches_cpu() {
    let Some(ctx) = gpu() else { return };
    // 7 qubits gate by gate, 13 in tiles, and 13 over eight buffers.
    for (n, chunk_bits, seed) in [(7, 7, 1), (13, 13, 2), (13, 10, 3)] {
        let circ = random_circuit(n, 60, seed);
        let mut gpu_state =
            GpuStatevector::zero_with_layout(&ctx, n, chunk_bits, GpuPrecision::Single).unwrap();
        assert_eq!(gpu_state.precision(), GpuPrecision::Single);
        gpu_state.run(&circ).unwrap();
        let cpu = Statevector::from_circuit(&circ);
        assert_close_single(&gpu_state.amplitudes().unwrap(), cpu.amplitudes());
        let h = hamiltonian(n, seed + 50);
        let got = gpu_state.expectation(&h).unwrap();
        assert!((got - cpu.expectation(&h).unwrap()).abs() < 1e-4);
        let applied = gpu_state.apply_pauli_sum(&h).unwrap();
        assert_close_single(
            &applied.amplitudes().unwrap(),
            cpu.apply_pauli_sum(&h).unwrap().amplitudes(),
        );

        let (pcirc, values) = parametric(n, seed);
        let start =
            GpuStatevector::zero_with_layout(&ctx, n, chunk_bits, GpuPrecision::Single).unwrap();
        let (value, gradient) = start.expectation_gradient(&pcirc, &values, &h).unwrap();
        let (want_value, want) = Statevector::zero(n)
            .expectation_gradient(&pcirc, &values, &h)
            .unwrap();
        assert!((value - want_value).abs() < 1e-4);
        for (g, w) in gradient.iter().zip(&want) {
            assert!((g - w).abs() < 1e-4, "{gradient:?} vs {want:?}");
        }

        gpu_state.evolve(&h, 1.5).unwrap();
        let mut evolved = cpu.clone();
        evolved.evolve(&h, 1.5).unwrap();
        assert!(
            gpu_state
                .amplitudes()
                .unwrap()
                .iter()
                .zip(evolved.amplitudes())
                .all(|(g, w)| (g - w).norm() < 1e-4)
        );
    }
}

/// Masks above 2^24 can't be held exactly as f32; single-precision kernels
/// decode them from the f64 parameters with integer arithmetic.
#[test]
fn single_precision_handles_high_qubits() {
    let Some(ctx) = gpu() else { return };
    let n = 26;
    let theta = 0.8;
    let mut circ = Circuit::new(n);
    circ.pauli_rotation(
        theta,
        PauliString::new([(25, Pauli::X), (24, Pauli::X)]).unwrap(),
    );
    circ.h(0).cx(0, 25).rz(0.3, 25).cp(0.7, 25, 3);
    let mut state = GpuStatevector::zero_with_precision(&ctx, n, GpuPrecision::Single).unwrap();
    state.run(&circ).unwrap();
    // After the rotation ⟨Z₂₄⟩ = cos θ; CX copies qubit 0's superposition,
    // so ⟨Z₀Z₂₅⟩ tracks qubit 25's state.
    let z24 = PauliSum::from_list([(&*label(n, &[(24, 'Z')]), 1.0)]).unwrap();
    assert!((state.expectation(&z24).unwrap() - theta.cos()).abs() < 1e-5);
    let p_flip = (theta / 2.0).sin().powi(2);
    let zz = PauliSum::from_list([(&*label(n, &[(25, 'Z'), (0, 'Z')]), 1.0)]).unwrap();
    // Qubit 25 flipped with probability p, independent of the CX parity.
    let want = (1.0 - p_flip) - p_flip;
    assert!((state.expectation(&zz).unwrap() - want).abs() < 1e-5);
}

fn label(n: usize, ops: &[(usize, char)]) -> String {
    let mut l = vec!['I'; n];
    for &(q, p) in ops {
        l[n - 1 - q] = p;
    }
    l.into_iter().collect()
}

#[test]
fn classical_control_on_gpu() {
    use ketrs::Condition;
    use rand::SeedableRng;
    let Some(ctx) = gpu() else { return };
    let mut rng = rand::rngs::StdRng::seed_from_u64(3);
    // Teleport Ry(θ)|0⟩ from qubit 0 to qubit 2.
    let theta = 1.1;
    let mut circ = Circuit::with_clbits(3, 3);
    circ.ry(theta, 0).h(1).cx(1, 2).cx(0, 1).h(0);
    circ.measure(0, 0).measure(1, 1);
    let mut x = Circuit::with_clbits(3, 3);
    x.x(2);
    let mut z = Circuit::with_clbits(3, 3);
    z.z(2);
    let condition = |c| Condition {
        clbits: vec![c],
        value: 1,
    };
    circ.if_test(condition(1), x, None);
    circ.if_test(condition(0), z, None);
    circ.reset(0).measure(0, 0).measure(2, 2);
    for precision in [GpuPrecision::Double, GpuPrecision::Single] {
        let start = GpuStatevector::zero_with_precision(&ctx, 3, precision).unwrap();
        let shots = 2_000;
        let counts = start.sample_circuit(&circ, shots, &mut rng).unwrap();
        let ones: usize = counts
            .iter()
            .filter(|(bits, _)| *bits >> 2 & 1 == 1)
            .map(|(_, c)| c)
            .sum();
        let p = (theta / 2.0).sin().powi(2);
        let sigma = (shots as f64 * p * (1.0 - p)).sqrt();
        assert!((ones as f64 - shots as f64 * p).abs() < 5.0 * sigma);
        // Reset leaves qubit 0 at 0.
        assert!(counts.keys().all(|bits| bits & 1 == 0));
    }
    // A trajectory collapses the state.
    let mut bell = Circuit::with_clbits(2, 1);
    bell.h(0).cx(0, 1).measure(0, 0);
    let mut state = GpuStatevector::zero(&ctx, 2).unwrap();
    let bits = state.run_trajectory(&bell, &mut rng).unwrap();
    let p1 = state.probability_one(1).unwrap();
    assert!((p1 - if bits[0] { 1.0 } else { 0.0 }).abs() < 1e-12);
    assert!((state.norm_sqr().unwrap() - 1.0).abs() < 1e-12);
}

/// A 28-qubit state (4 GiB, two buffers on most GPUs) without downloading
/// it. Ignored by default for its memory; run with `--ignored`.
#[test]
#[ignore]
fn twenty_eight_qubits() {
    use rand::SeedableRng;
    let Some(ctx) = gpu() else { return };
    let n = 28;
    let mut ghz = GpuStatevector::from_circuit(&ctx, &ketrs::library::ghz(n)).unwrap();
    assert!((ghz.norm_sqr().unwrap() - 1.0).abs() < 1e-10);
    let ends = PauliSum::from(PauliString::new([(0, Pauli::Z), (n - 1, Pauli::Z)]).unwrap());
    assert!((ghz.expectation(&ends).unwrap() - 1.0).abs() < 1e-10);
    let parity = PauliSum::from(PauliString::new((0..n).map(|q| (q, Pauli::X))).unwrap());
    assert!((ghz.expectation(&parity).unwrap() - 1.0).abs() < 1e-10);
    let mut rng = rand::rngs::StdRng::seed_from_u64(1);
    let counts = ghz.sample_counts(1000, &mut rng).unwrap();
    assert_eq!(
        counts.keys().copied().collect::<Vec<_>>(),
        [0, (1 << n) - 1]
    );
}
