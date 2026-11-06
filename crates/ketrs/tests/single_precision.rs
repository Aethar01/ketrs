//! Single-precision engines, checked against double precision.

use ketrs::statevector::EigenOptions;
use ketrs::{
    Circuit, Complex64 as C64, DenseState, Pauli, PauliString, PauliSum, RealSectorStatevector32,
    RealStatevector, RealStatevector32, Scalar, SectorBasis, SectorStatevector,
    SectorStatevector32, Statevector, Statevector32,
};
use std::sync::Arc;

/// Single precision keeps about 7 digits; errors accumulate over a circuit.
const TOL: f64 = 2e-5;

fn max_diff<T: Scalar, U: Scalar>(a: &[T], b: &[U]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(a, b)| (a.to_complex() - b.to_complex()).norm())
        .fold(0.0, f64::max)
}

/// Layers of every kind of gate the kernels specialise.
fn mixed_circuit(n: usize) -> Circuit {
    let mut circ = Circuit::new(n);
    for layer in 0..4 {
        for q in 0..n {
            let a = 0.3 + 0.17 * (q + 3 * layer) as f64;
            circ.h(q).rz(a, q).ry(-a, q).t(q);
        }
        for q in 0..n - 1 {
            circ.cx(q, q + 1).cp(0.4, q + 1, q);
        }
        circ.rzz(0.3, 0, n - 1).swap(1, n - 2);
        circ.pauli_rotation(
            0.7,
            PauliString::new((0..n).map(|q| (q, [Pauli::X, Pauli::Y, Pauli::Z][q % 3]))).unwrap(),
        );
    }
    circ.set_global_phase(0.4);
    circ
}

/// Real gates only.
fn real_circuit(n: usize) -> Circuit {
    let mut circ = Circuit::new(n);
    for layer in 0..4 {
        for q in 0..n {
            circ.h(q).ry(0.2 + 0.31 * (q + layer) as f64, q);
        }
        for q in 0..n - 1 {
            circ.cx(q, q + 1).cz(q, (q + 2) % n);
        }
    }
    circ
}

fn ising(n: usize) -> PauliSum {
    let mut h = PauliSum::new();
    for q in 0..n {
        h.add(0.7, PauliString::new([(q, Pauli::X)]).unwrap());
        if q + 1 < n {
            h.add(
                1.0,
                PauliString::new([(q, Pauli::Z), (q + 1, Pauli::Z)]).unwrap(),
            );
        }
    }
    h
}

#[test]
fn circuits_match_double_precision() {
    // 8 qubits runs gate by gate; 15 and 21 take the tiled paths.
    for n in [8, 15, 21] {
        let circ = mixed_circuit(n);
        let single = Statevector32::try_from_circuit(&circ).unwrap();
        let double = Statevector::from_circuit(&circ);
        assert!(max_diff(single.amplitudes(), double.amplitudes()) < TOL);
        assert!((single.norm_sqr() - 1.0).abs() < 1e-5);

        let real = real_circuit(n);
        let single = RealStatevector32::from_circuit(&real).unwrap();
        let double = RealStatevector::from_circuit(&real).unwrap();
        assert!(max_diff(single.amplitudes(), double.amplitudes()) < TOL);
    }
}

#[test]
fn observables_and_gradients_match_double_precision() {
    let n = 10;
    let mut circ = Circuit::new(n);
    let params = circ.parameters(n);
    for (q, &p) in params.iter().enumerate() {
        circ.ry(p, q).rz(0.5 * p, q);
    }
    for q in 0..n - 1 {
        circ.cx(q, q + 1);
    }
    let values: Vec<f64> = (0..n).map(|k| 0.3 * k as f64 - 1.0).collect();
    let h = ising(n);
    let (e32, g32) = Statevector32::zero(n)
        .expectation_gradient(&circ, &values, &h)
        .unwrap();
    let (e64, g64) = Statevector::zero(n)
        .expectation_gradient(&circ, &values, &h)
        .unwrap();
    assert!((e32 - e64).abs() < 1e-4, "{e32} vs {e64}");
    for (a, b) in g32.iter().zip(&g64) {
        assert!((a - b).abs() < 1e-4, "{a} vs {b}");
    }
}

#[test]
fn evolution_matches_double_precision() {
    let n = 10;
    let h = ising(n);
    let mut single = Statevector32::basis(n, 0b1011001010);
    let mut double = Statevector::basis(n, 0b1011001010);
    single.evolve(&h, 3.0).unwrap();
    double.evolve(&h, 3.0).unwrap();
    assert!(max_diff(single.amplitudes(), double.amplitudes()) < 1e-4);
    assert!((single.norm_sqr() - 1.0).abs() < 1e-4);
}

#[test]
fn eigenstates_converge_in_single_precision() {
    let n = 10;
    let h = ising(n);
    let opts = EigenOptions::default();
    let double = Statevector::lowest_eigenstates(&h, n, 2, &opts).unwrap();
    let single = Statevector32::lowest_eigenstates(&h, n, 2, &opts).unwrap();
    let real = RealStatevector32::lowest_eigenstates(&h, n, 2, &opts).unwrap();
    for found in [
        single
            .iter()
            .map(|e| (e.energy, e.residual))
            .collect::<Vec<_>>(),
        real.iter().map(|e| (e.energy, e.residual)).collect(),
    ] {
        for ((energy, residual), want) in found.iter().zip(&double) {
            assert!(
                (energy - want.energy).abs() < 1e-4,
                "{energy} vs {}",
                want.energy
            );
            assert!(*residual < 1e-3, "residual {residual}");
        }
    }
}

#[test]
fn sectors_in_single_precision() {
    let n = 8;
    let basis = Arc::new(SectorBasis::new(n, &[((0..n).collect(), n / 2)]).unwrap());
    let mut h = PauliSum::new();
    for q in 0..n - 1 {
        for p in [Pauli::X, Pauli::Y, Pauli::Z] {
            h.add(1.0, PauliString::new([(q, p), (q + 1, p)]).unwrap());
        }
    }
    let start = 0b10110100;
    let mut single = SectorStatevector32::basis_state(basis.clone(), start).unwrap();
    let mut double = SectorStatevector::basis_state(basis.clone(), start).unwrap();
    single.evolve(&h, 1.5).unwrap();
    double.evolve(&h, 1.5).unwrap();
    assert!(max_diff(single.amplitudes(), double.amplitudes()) < 1e-4);

    let opts = EigenOptions::default();
    let want = SectorStatevector::lowest_eigenstates(basis.clone(), &h, 1, &opts).unwrap();
    let got = RealSectorStatevector32::lowest_eigenstates(basis, &h, 1, &opts).unwrap();
    assert!((got[0].energy - want[0].energy).abs() < 1e-4);
}

#[test]
fn single_precision_halves_memory() {
    assert_eq!(size_of::<<Statevector32 as Amps>::T>(), 8);
    assert_eq!(size_of::<<RealStatevector32 as Amps>::T>(), 4);
    let _: C64 = Statevector32::zero(1).amplitudes()[0].to_complex();
}

trait Amps {
    type T;
}

impl<T: Scalar> Amps for DenseState<T> {
    type T = T;
}
