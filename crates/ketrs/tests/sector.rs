//! Symmetry-sector simulation, checked against the full statevector engine.

mod common;

use common::{TOL, c};
use ketrs::statevector::EigenOptions;
use ketrs::{
    Circuit, Complex64 as C64, Error, Pauli, PauliString, PauliSum, RealSectorStatevector,
    SectorBasis, SectorStatevector, Statevector,
};
use std::sync::Arc;

fn assert_matches_dense(sector: &SectorStatevector, dense: &Statevector) {
    let expanded = sector.to_dense();
    for (i, (a, b)) in expanded
        .amplitudes()
        .iter()
        .zip(dense.amplitudes())
        .enumerate()
    {
        assert!((a - b).norm() < 1e-10, "amplitude {i}: {a} vs {b}");
    }
}

/// Number-conserving circuit on 6 qubits; qubit 5 is unconstrained.
fn conserving_circuit() -> Circuit {
    let mut circ = Circuit::new(6);
    for (a, b) in [(0, 1), (1, 2), (3, 4), (2, 3)] {
        // Hopping exp(-iθ(XX+YY)/2) as two rotations; fusion joins them.
        circ.rxx(0.7, a, b).ryy(0.7, a, b);
        circ.rz(0.3 * a as f64, a).cp(0.4, a, b).rzz(-0.2, a, b);
    }
    circ.swap(0, 4).cz(1, 3);
    // A Givens rotation on |01⟩, |10⟩.
    let (s, co) = 0.5f64.sin_cos();
    let zero = c(0.0, 0.0);
    let one = c(1.0, 0.0);
    circ.unitary2(
        [
            [one, zero, zero, zero],
            [zero, c(co, 0.0), c(-s, 0.0), zero],
            [zero, c(s, 0.0), c(co, 0.0), zero],
            [zero, zero, zero, one],
        ],
        2,
        4,
    );
    // Weight-4 diagonal rotation, and gates on the free qubit.
    circ.pauli_rotation(0.9, PauliString::from_label("IZZIZZ").unwrap());
    circ.h(5).rx(0.6, 5).cx(0, 5);
    circ.pauli_rotation(0.8, PauliString::from_label("XZIIZI").unwrap());
    circ.set_global_phase(0.2);
    circ
}

fn half_filling(n_constrained: usize, n: usize) -> Arc<SectorBasis> {
    Arc::new(SectorBasis::new(n, &[((0..n_constrained).collect(), n_constrained / 2)]).unwrap())
}

#[test]
fn conserving_circuit_matches_dense() {
    let basis = half_filling(5, 6);
    // Start from a superposition inside the sector: two particles.
    let start = 0b00011;
    let mut sector = SectorStatevector::basis_state(basis.clone(), start).unwrap();
    let mut dense = Statevector::basis(6, start);
    let circ = conserving_circuit();
    sector.run(&circ).unwrap();
    dense.run(&circ).unwrap();
    assert_matches_dense(&sector, &dense);
    assert!((sector.norm_sqr() - 1.0).abs() < TOL);
}

#[test]
fn symmetry_breaking_operations_are_rejected() {
    let basis = half_filling(4, 4);
    let start = SectorStatevector::basis_state(basis, 0b0011).unwrap();
    for circ in [
        {
            let mut c = Circuit::new(4);
            c.x(0);
            c
        },
        {
            // XX alone moves |00⟩ to |11⟩.
            let mut c = Circuit::new(4);
            c.rxx(0.5, 0, 1);
            c
        },
        {
            let mut c = Circuit::new(4);
            c.pauli_rotation(0.3, PauliString::from_label("XXXI").unwrap());
            c
        },
    ] {
        let mut state = start.clone();
        assert!(
            matches!(state.run(&circ), Err(Error::BreaksSymmetry(_))),
            "{circ:?}"
        );
        assert_eq!(state.amplitudes(), start.amplitudes());
    }
    assert!(SectorStatevector::basis_state(half_filling(4, 4), 0b0111).is_err());
}

fn heisenberg(n: usize, field: f64) -> PauliSum {
    let mut h = PauliSum::new();
    for q in 0..n - 1 {
        for p in [Pauli::X, Pauli::Y, Pauli::Z] {
            h.add(1.0, PauliString::new([(q, p), (q + 1, p)]).unwrap());
        }
    }
    for q in 0..n {
        h.add(
            field * (q as f64 - 2.0),
            PauliString::new([(q, Pauli::Z)]).unwrap(),
        );
    }
    h
}

#[test]
fn observables_and_evolution_match_dense() {
    let n = 8;
    let basis = half_filling(n, n);
    let h = heisenberg(n, 0.3);
    let start = 0b10110100;
    let mut sector = SectorStatevector::basis_state(basis, start).unwrap();
    let mut dense = Statevector::basis(n, start);
    sector.evolve(&h, 1.7).unwrap();
    dense.evolve(&h, 1.7).unwrap();
    assert_matches_dense(&sector, &dense);

    // Includes terms that leave the sector, which contribute nothing.
    let observable = PauliSum::from_list([
        ("ZZIIIIII", 1.0),
        ("IIXXIIII", 0.5),
        ("XIIIIIIX", -0.3),
        ("IYYIIIII", 0.7),
    ])
    .unwrap();
    assert!(
        (sector.expectation(&observable).unwrap() - dense.expectation(&observable).unwrap()).abs()
            < TOL
    );

    let applied = sector.apply_pauli_sum(&h).unwrap();
    assert_matches_dense(&applied, &dense.apply_pauli_sum(&h).unwrap());
}

#[test]
fn ground_state_in_sector() {
    // The antiferromagnetic Heisenberg ground state has Sz = 0.
    let n = 10;
    let h = heisenberg(n, 0.0);
    let basis = half_filling(n, n);
    assert_eq!(basis.dim(), 252);
    let opts = EigenOptions::default();
    let in_sector = SectorStatevector::lowest_eigenstates(basis.clone(), &h, 3, &opts).unwrap();
    let full = Statevector::lowest_eigenstates(&h, n, 1, &opts).unwrap();
    assert!((in_sector[0].energy - full[0].energy).abs() < 1e-8);
    for e in &in_sector {
        assert!(e.residual < 1e-7);
    }
    let real = RealSectorStatevector::lowest_eigenstates(basis.clone(), &h, 3, &opts).unwrap();
    for (a, b) in in_sector.iter().zip(&real) {
        assert!((a.energy - b.energy).abs() < 1e-8);
    }

    // A field along X breaks number conservation.
    let mut broken = h.clone();
    broken.add(0.1, PauliString::new([(0, Pauli::X)]).unwrap());
    assert!(matches!(
        SectorStatevector::lowest_eigenstates(basis, &broken, 1, &opts),
        Err(Error::BreaksSymmetry(_))
    ));
}

#[test]
fn two_species_with_a_free_qubit() {
    // Spin-up on even qubits, spin-down on odd, qubit 6 unconstrained.
    let basis = Arc::new(SectorBasis::new(7, &[(vec![0, 2, 4], 1), (vec![1, 3, 5], 2)]).unwrap());
    assert_eq!(basis.dim(), 3 * 3 * 2);
    let mut circ = Circuit::new(7);
    // Hopping within each species, an interaction, and a free-qubit gate.
    circ.rxx(0.4, 0, 2)
        .ryy(0.4, 0, 2)
        .rxx(0.9, 1, 5)
        .ryy(0.9, 1, 5);
    circ.cz(0, 1).rzz(0.3, 2, 3).ry(1.1, 6).cry(0.5, 0, 6);
    let start = 0b000_1011;
    let mut sector = SectorStatevector::basis_state(basis.clone(), start).unwrap();
    let mut dense = Statevector::basis(7, start);
    sector.run(&circ).unwrap();
    dense.run(&circ).unwrap();
    assert_matches_dense(&sector, &dense);

    // Hopping between species changes both weights.
    let mut mixing = Circuit::new(7);
    mixing.rxx(0.4, 0, 1).ryy(0.4, 0, 1);
    assert!(matches!(
        sector.clone().run(&mixing),
        Err(Error::BreaksSymmetry(_))
    ));
}

#[test]
fn sampling_returns_full_basis_states() {
    use rand::SeedableRng;
    let basis = half_filling(4, 4);
    let mut circ = Circuit::new(4);
    circ.rxx(1.0, 0, 1).ryy(1.0, 0, 1);
    let mut state = SectorStatevector::basis_state(basis, 0b0101).unwrap();
    state.run(&circ).unwrap();
    let mut rng = rand::rngs::StdRng::seed_from_u64(1);
    for (j, _) in state.sample_counts(500, &mut rng) {
        assert_eq!(j.count_ones(), 2, "{j:b}");
    }
    let dense = state.to_dense();
    let roundtrip = SectorStatevector::from_dense(state.basis().clone(), &dense).unwrap();
    assert_eq!(roundtrip.amplitudes(), state.amplitudes());
    let _: C64 = state.inner(&roundtrip).unwrap();
}
