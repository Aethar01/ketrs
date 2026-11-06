//! Statevector engine benchmarks.

// `criterion_group!` generates undocumented items.
#![allow(missing_docs)]

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use ketrs::{Circuit, Pauli, PauliString, PauliSum, Statevector, library};
use std::hint::black_box;
use std::time::Duration;

type Builder = fn(usize) -> Circuit;

const SIZES: [usize; 4] = [10, 14, 18, 22];

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

criterion_group!(benches, bench_circuits, bench_gradient);
criterion_main!(benches);
