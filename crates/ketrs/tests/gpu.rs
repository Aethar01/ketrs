//! GPU statevectors, checked against the CPU engine. Tests skip (passing)
//! when no Vulkan GPU with 64-bit floats is available.
#![cfg(feature = "gpu")]

mod common;

use ketrs::{
    Circuit, Complex64 as C64, Error, GpuContext, GpuStatevector, Pauli, PauliString, PauliSum,
    Statevector,
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
    for chunk_bits in [9, 10, 11] {
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
