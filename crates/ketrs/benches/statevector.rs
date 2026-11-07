//! Statevector engine benchmarks.

// `criterion_group!` generates undocumented items.
#![allow(missing_docs)]

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use ketrs::{Circuit, Expr, Pauli, PauliString, PauliSum, Statevector, library};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::hint::black_box;
use std::time::Duration;

type Builder = fn(usize) -> Circuit;

const SIZES: [usize; 6] = [10, 12, 14, 16, 18, 22];

/// `depth` layers of Ry on every qubit followed by a CX ladder, roughly the
/// shape of a hardware-efficient variational ansatz.
fn layered(n_qubits: usize, depth: usize) -> Circuit {
    let mut circ = Circuit::new(n_qubits);
    for layer in 0..depth {
        for q in 0..n_qubits {
            circ.ry(0.1 * (layer * n_qubits + q) as f64, q);
        }
        for q in 0..n_qubits - 1 {
            circ.cx(q, q + 1);
        }
    }
    circ
}

fn bench_circuits(c: &mut Criterion) {
    let circuits: [(&str, Builder); 2] =
        [("qft", library::qft), ("layered_d10", |n| layered(n, 10))];
    for (name, build) in circuits {
        let mut group = c.benchmark_group(name);
        group
            .sample_size(10)
            .measurement_time(Duration::from_secs(5));
        for n in SIZES {
            let circ = build(n);
            group.bench_with_input(BenchmarkId::from_parameter(n), &circ, |b, circ| {
                b.iter(|| Statevector::from_circuit(black_box(circ)))
            });
        }
        group.finish();
    }
}

/// The layered ansatz with a free parameter per Ry, and a transverse-field
/// Ising Hamiltonian to differentiate against.
fn variational(n_qubits: usize, depth: usize) -> (Circuit, Vec<f64>, PauliSum) {
    let mut circ = Circuit::new(n_qubits);
    for _ in 0..depth {
        for (q, p) in circ.parameters(n_qubits).into_iter().enumerate() {
            circ.ry(p, q);
        }
        for q in 0..n_qubits - 1 {
            circ.cx(q, q + 1);
        }
    }
    let values = (0..circ.n_parameters()).map(|k| 0.1 * k as f64).collect();
    let mut h = PauliSum::new();
    for q in 0..n_qubits {
        h.add(0.7, PauliString::new([(q, Pauli::X)]).unwrap());
        if q + 1 < n_qubits {
            h.add(
                1.0,
                PauliString::new([(q, Pauli::Z), (q + 1, Pauli::Z)]).unwrap(),
            );
        }
    }
    (circ, values, h)
}

/// One energy evaluation against the energy and its full gradient.
fn bench_gradient(c: &mut Criterion) {
    let mut group = c.benchmark_group("gradient_d4");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(5));
    for n in [14, 18, 22] {
        let (circ, values, h) = variational(n, 4);
        let zero = Statevector::zero(n);
        group.bench_function(BenchmarkId::new("energy", n), |b| {
            b.iter(|| {
                let mut state = zero.clone();
                state.run(&circ.bind(&values).unwrap()).unwrap();
                state.expectation(&h).unwrap()
            })
        });
        group.bench_function(BenchmarkId::new("gradient", n), |b| {
            b.iter(|| zero.expectation_gradient(&circ, &values, &h).unwrap())
        });
    }
    group.finish();
}

/// `depth` layers of the ansatz appended to `circ`.
fn append_layers(circ: &mut Circuit, depth: usize) {
    let n = circ.n_qubits();
    for layer in 0..depth {
        for q in 0..n {
            circ.ry(0.3 + 0.1 * (layer * n + q) as f64, q);
        }
        for q in 0..n - 1 {
            circ.cx(q, q + 1);
        }
    }
}

/// Teleportation-style feed-forward: layers, two mid-circuit measurements
/// with corrections conditioned on them, more layers, then every qubit
/// measured.
fn feed_forward(n: usize) -> Circuit {
    let mut circ = Circuit::with_clbits(n, n);
    append_layers(&mut circ, 4);
    circ.measure(0, 0).measure(1, 1);
    let mut z = Circuit::with_clbits(n, n);
    z.z(2);
    let mut x = Circuit::with_clbits(n, n);
    x.x(2);
    circ.if_test(Expr::Bit(0), z, None);
    circ.if_test(Expr::Bit(1), x, None);
    append_layers(&mut circ, 4);
    for q in 0..n {
        circ.measure(q, q);
    }
    circ
}

/// Repeat until success: re-prepare and measure qubit 0, entangled with the
/// rest, until it reads 1, then more layers and every qubit measured.
fn repeat_until_success(n: usize) -> Circuit {
    let mut circ = Circuit::with_clbits(n, n);
    append_layers(&mut circ, 2);
    let mut body = Circuit::with_clbits(n, n);
    body.reset(0).ry(1.0, 0).cx(0, 1).measure(0, 0);
    circ.while_loop(Expr::Not(Box::new(Expr::Bit(0))), body);
    append_layers(&mut circ, 4);
    for q in 0..n {
        circ.measure(q, q);
    }
    circ
}

/// `sample_circuit` on circuits with mid-circuit measurement, where it
/// follows each distinct measurement history once.
fn bench_mid_circuit(c: &mut Criterion) {
    let circuits: [(&str, Builder); 2] = [
        ("feed_forward", feed_forward),
        ("repeat_until_success", repeat_until_success),
    ];
    let shots = 1000;
    for (name, build) in circuits {
        let mut group = c.benchmark_group(format!("mid_circuit_{name}"));
        group
            .sample_size(10)
            .measurement_time(Duration::from_secs(5));
        for n in [12, 16] {
            let circ = build(n);
            let zero = Statevector::zero(n);
            group.bench_function(BenchmarkId::new("sample_circuit", n), |b| {
                let mut rng = StdRng::seed_from_u64(1);
                b.iter(|| {
                    zero.sample_circuit(black_box(&circ), shots, &mut rng)
                        .unwrap()
                })
            });
        }
        group.finish();
    }
}

criterion_group!(benches, bench_circuits, bench_gradient, bench_mid_circuit);
criterion_main!(benches);
