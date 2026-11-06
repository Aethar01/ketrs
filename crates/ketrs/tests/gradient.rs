//! Parametric circuits and adjoint gradients, checked against central
//! finite differences.

mod common;

use common::assert_state_eq;
use ketrs::{
    Circuit, Complex64 as C64, DenseState, Error, Pauli, PauliString, PauliSum, RealStatevector,
    Scalar, Statevector,
};

/// A small deterministic generator, so failures are reproducible.
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
    fn angle(&mut self) -> f64 {
        (self.next() - 0.5) * 6.0
    }
}

/// A random circuit using every parametrisable gate, with parameters shared
/// between gates and used through linear forms.
fn random_parametric(n: usize, seed: u64) -> (Circuit, Vec<f64>) {
    let mut rng = Lcg(seed);
    let mut circ = Circuit::new(n);
    let params = circ.parameters(4);
    circ.set_global_phase(0.3);
    for _ in 0..30 {
        let p = params[rng.below(params.len())];
        let angle = rng.angle() * p + rng.angle();
        let (a, b) = (rng.below(n), rng.below(n));
        let b = if a == b { (a + 1) % n } else { b };
        match rng.below(12) {
            0 => circ.rx(angle, a),
            1 => circ.ry(angle, a),
            2 => circ.rz(angle, a),
            3 => circ.p(angle, a),
            4 => circ.crx(angle, a, b),
            5 => circ.cry(angle, a, b),
            6 => circ.crz(angle, a, b),
            7 => circ.cp(angle, a, b),
            8 => circ.rzz(angle, a, b),
            9 => circ.ryy(angle, a, b),
            10 => {
                let ps = [Pauli::X, Pauli::Y, Pauli::Z];
                let string = PauliString::new((0..n).map(|q| (q, ps[rng.below(3)]))).unwrap();
                circ.pauli_rotation(angle, string)
            }
            _ => circ.h(a).cx(a, b).s(b),
        };
    }
    let values = (0..params.len()).map(|_| rng.angle()).collect();
    (circ, values)
}

fn hamiltonian(n: usize, seed: u64) -> PauliSum {
    let mut rng = Lcg(seed);
    let ps = [Pauli::I, Pauli::X, Pauli::Y, Pauli::Z];
    let mut h = PauliSum::new();
    for _ in 0..6 {
        h.add(
            rng.angle(),
            PauliString::new((0..n).map(|q| (q, ps[rng.below(4)]))).unwrap(),
        );
    }
    h
}

/// Central finite differences of `f` at `values`.
fn finite_difference(values: &[f64], f: impl Fn(&[f64]) -> f64) -> Vec<f64> {
    let h = 1e-6;
    (0..values.len())
        .map(|i| {
            let (mut up, mut down) = (values.to_vec(), values.to_vec());
            up[i] += h;
            down[i] -= h;
            (f(&up) - f(&down)) / (2.0 * h)
        })
        .collect()
}

fn assert_gradients_close(got: &[f64], want: &[f64]) {
    for (g, w) in got.iter().zip(want) {
        assert!(
            (g - w).abs() < 1e-6,
            "adjoint {got:?}\nfinite difference {want:?}"
        );
    }
}

fn energy<T: Scalar>(circ: &Circuit, values: &[f64], h: &PauliSum) -> f64 {
    DenseState::<T>::try_from_circuit(&circ.bind(values).unwrap())
        .unwrap()
        .expectation(h)
        .unwrap()
}

#[test]
fn expectation_gradient_matches_finite_differences() {
    for seed in 0..8 {
        let n = 4;
        let (circ, values) = random_parametric(n, seed);
        let h = hamiltonian(n, seed + 100);
        let (value, gradient) = Statevector::zero(n)
            .expectation_gradient(&circ, &values, &h)
            .unwrap();
        assert!((value - energy::<C64>(&circ, &values, &h)).abs() < 1e-12);
        let want = finite_difference(&values, |v| energy::<C64>(&circ, v, &h));
        assert_gradients_close(&gradient, &want);
    }
}

/// Large enough that the backward sweep works tile by tile, streams steps
/// too wide for a tile, and fuses between parameters; 21 qubits also
/// gathers tiles from scattered qubits.
#[test]
fn gradient_on_large_states() {
    for n in [17, 21] {
        let random = (0..2).map(|seed| random_parametric(n, seed));
        for (seed, (circ, values)) in std::iter::once(layered(n)).chain(random).enumerate() {
            let h = hamiltonian(n, seed as u64 + 100);
            let (_, gradient) = Statevector::zero(n)
                .expectation_gradient(&circ, &values, &h)
                .unwrap();
            let want = finite_difference(&values, |v| energy::<C64>(&circ, v, &h));
            assert_gradients_close(&gradient, &want);
        }
    }
}

/// Layers of rotations on every qubit, including diagonal ones and
/// controls on high qubits whose signs depend on the tile.
fn layered(n: usize) -> (Circuit, Vec<f64>) {
    let mut circ = Circuit::new(n);
    let params = circ.parameters(5);
    for (layer, &p) in params.iter().enumerate() {
        for q in 0..n {
            let a = 0.3 + 0.1 * (q + layer) as f64;
            circ.ry(a * p, q).rz(0.5 * p + a, q);
            if n - 1 - q != q {
                circ.crx(a * p, n - 1 - q, q);
            }
        }
        for q in 0..n - 1 {
            circ.cx(q, q + 1);
        }
    }
    (circ, vec![0.4, -0.7, 1.1, 0.2, -1.3])
}

#[test]
fn real_engine_gradient_matches_complex() {
    // Real gates only: Ry, CRy, H, CX, CZ and Pauli rotations with an odd
    // number of Ys.
    let n = 5;
    let mut circ = Circuit::new(n);
    let theta = circ.parameters(3);
    for q in 0..n {
        circ.ry(theta[q % 3] * (1.0 + q as f64) + 0.1, q);
    }
    circ.cx(0, 1).cz(2, 3).h(4);
    circ.cry(-theta[1], 1, 2);
    circ.pauli_rotation(2.0 * theta[2], PauliString::from_label("XYZIX").unwrap());
    let values = [0.4, -1.1, 0.7];
    let h = PauliSum::from_list([
        ("ZZIII", 1.0),
        ("IXXII", -0.5),
        ("IIIYY", 0.3),
        ("YIIIX", 0.8),
    ])
    .unwrap();

    let (e_real, g_real) = RealStatevector::zero(n)
        .expectation_gradient(&circ, &values, &h)
        .unwrap();
    let (e_complex, g_complex) = Statevector::zero(n)
        .expectation_gradient(&circ, &values, &h)
        .unwrap();
    assert!((e_real - e_complex).abs() < 1e-12);
    assert_gradients_close(&g_real, &g_complex);
    assert_gradients_close(
        &g_real,
        &finite_difference(&values, |v| energy::<f64>(&circ, v, &h)),
    );
}

#[test]
fn overlap_loss_gradient() {
    // L = |⟨φ|ψ(θ)⟩|², the deflation penalty in VQD; λ = |φ⟩⟨φ|ψ⟩.
    let n = 3;
    let (circ, values) = random_parametric(n, 42);
    let (target_circ, target_values) = random_parametric(n, 7);
    let phi = Statevector::from_circuit(&target_circ.bind(&target_values).unwrap());
    let loss = |v: &[f64]| {
        let psi = Statevector::from_circuit(&circ.bind(v).unwrap());
        phi.inner(&psi).unwrap().norm_sqr()
    };
    let psi = Statevector::from_circuit(&circ.bind(&values).unwrap());
    let overlap = phi.inner(&psi).unwrap();
    let lambda =
        Statevector::from_amplitudes(phi.amplitudes().iter().map(|a| a * overlap).collect())
            .unwrap();
    let gradient = Statevector::adjoint_gradient(&circ, &values, &psi, &lambda).unwrap();
    assert_gradients_close(&gradient, &finite_difference(&values, loss));
}

#[test]
fn parameters_survive_fusion_inverse_and_extend() {
    let n = 4;
    let (circ, values) = random_parametric(n, 3);
    let reference = Statevector::from_circuit(&circ.bind(&values).unwrap());

    let fused = circ.fused();
    assert_eq!(fused.n_parameters(), circ.n_parameters());
    assert_state_eq(
        &Statevector::from_circuit(&fused.bind(&values).unwrap()),
        reference.amplitudes(),
    );

    let mut round_trip = circ.clone();
    round_trip.extend(&circ.inverse()).unwrap();
    assert_state_eq(
        &Statevector::from_circuit(&round_trip.bind(&values).unwrap()),
        Statevector::zero(n).amplitudes(),
    );
}

#[test]
fn unbound_and_misbound_circuits_are_errors() {
    let mut circ = Circuit::new(1);
    let theta = circ.parameter();
    circ.rx(theta, 0);
    assert!(matches!(
        Statevector::zero(1).run(&circ),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        circ.bind(&[0.1, 0.2]),
        Err(Error::InvalidArgument(_))
    ));
}
