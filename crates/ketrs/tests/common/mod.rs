#![allow(dead_code)]

use ketrs::gate::{Matrix2, Matrix4};
use ketrs::{Circuit, Complex64 as C64, Gate, Operation, Pauli, Statevector};

pub const TOL: f64 = 1e-10;

pub fn assert_state_eq(got: &Statevector, want: &[C64]) {
    assert_eq!(got.amplitudes().len(), want.len(), "dimension mismatch");
    for (i, (g, w)) in got.amplitudes().iter().zip(want).enumerate() {
        assert!(
            (g - w).norm() < TOL,
            "amplitude {i}: got {g}, want {w}\nfull state: {:?}",
            got.amplitudes()
        );
    }
}

pub fn c(re: f64, im: f64) -> C64 {
    C64::new(re, im)
}

/// `high ⊗ low`, in the `2·b₁ + b₀` ordering used by `Unitary2`, where
/// `low` acts on `qubits[0]`.
pub fn kron(high: &Matrix2, low: &Matrix2) -> Matrix4 {
    std::array::from_fn(|r| std::array::from_fn(|col| high[r >> 1][col >> 1] * low[r & 1][col & 1]))
}

/// `u` on `qubits[1]`, controlled by `qubits[0]`, as a `Unitary2` matrix.
pub fn controlled_matrix(u: &Matrix2) -> Matrix4 {
    std::array::from_fn(|r| {
        std::array::from_fn(|col| {
            let (r0, r1, c0, c1) = (r & 1, r >> 1, col & 1, col >> 1);
            match (r0 == c0, r0) {
                (false, _) => c(0.0, 0.0),
                (true, 0) => c(if r1 == c1 { 1.0 } else { 0.0 }, 0.0),
                (true, _) => u[r1][c1],
            }
        })
    })
}

pub fn swap_matrix() -> Matrix4 {
    let mut m = [[c(0.0, 0.0); 4]; 4];
    for (r, col) in [(0, 0), (1, 2), (2, 1), (3, 3)] {
        m[r][col] = c(1.0, 0.0);
    }
    m
}

/// Straightforward index-loop simulator, kept as a reference for the
/// optimised kernels.
pub fn reference_run(circ: &Circuit) -> Vec<C64> {
    let mut amps = Statevector::zero(circ.n_qubits()).into_amplitudes();
    for op in circ.ops() {
        if let Operation::PauliRotation { pauli, theta } = op {
            // cos(θ/2) ψ − i sin(θ/2) Pψ, applying P factor by factor.
            let mut p_psi = amps.clone();
            for &(q, p) in pauli.ops() {
                let gate = match p {
                    Pauli::I => continue,
                    Pauli::X => Gate::X,
                    Pauli::Y => Gate::Y,
                    Pauli::Z => Gate::Z,
                };
                ref_2x2(&mut p_psi, &gate.matrix(), None, q);
            }
            let (sin, cos) = (theta / 2.0).sin_cos();
            for (a, pa) in amps.iter_mut().zip(&p_psi) {
                *a = *a * cos + C64::new(0.0, -sin) * pa;
            }
            continue;
        }
        match *op {
            Operation::Gate { gate, target } => ref_2x2(&mut amps, &gate.matrix(), None, target),
            Operation::Controlled {
                gate,
                control,
                target,
            } => ref_2x2(&mut amps, &gate.matrix(), Some(control), target),
            Operation::Swap(a, b) => {
                for i in 0..amps.len() {
                    if i & (1 << a) != 0 && i & (1 << b) == 0 {
                        amps.swap(i, i ^ (1 << a) ^ (1 << b));
                    }
                }
            }
            Operation::MultiControlled {
                gate,
                ref controls,
                target,
            } => {
                let m = gate.matrix();
                let cmask: usize = controls.iter().map(|c| 1 << c).sum();
                for i in 0..amps.len() {
                    if i & cmask == cmask && i & (1 << target) == 0 {
                        let j = i | 1 << target;
                        let (x, y) = (amps[i], amps[j]);
                        amps[i] = m[0][0] * x + m[0][1] * y;
                        amps[j] = m[1][0] * x + m[1][1] * y;
                    }
                }
            }
            Operation::ControlledSwap { ref controls, a, b } => {
                let cmask: usize = controls.iter().map(|c| 1 << c).sum();
                for i in 0..amps.len() {
                    if i & cmask == cmask && i & (1 << a) != 0 && i & (1 << b) == 0 {
                        amps.swap(i, i ^ (1 << a) ^ (1 << b));
                    }
                }
            }
            Operation::Unitary2 { matrix, qubits } => {
                let [q0, q1] = qubits;
                for i in 0..amps.len() {
                    if i & (1 << q0) == 0 && i & (1 << q1) == 0 {
                        let idx = [i, i | 1 << q0, i | 1 << q1, i | 1 << q0 | 1 << q1];
                        let v = idx.map(|k| amps[k]);
                        for (row, &k) in matrix.iter().zip(&idx) {
                            amps[k] = row.iter().zip(&v).map(|(x, y)| x * y).sum();
                        }
                    }
                }
            }
            Operation::PauliRotation { .. } => unreachable!(),
        }
    }
    let phase = C64::cis(circ.global_phase());
    amps.iter_mut().for_each(|a| *a *= phase);
    amps
}

fn ref_2x2(amps: &mut [C64], m: &Matrix2, control: Option<usize>, target: usize) {
    for i in 0..amps.len() {
        let control_set = control.is_none_or(|c| i & (1 << c) != 0);
        if i & (1 << target) == 0 && control_set {
            let j = i | 1 << target;
            let (a, b) = (amps[i], amps[j]);
            amps[i] = m[0][0] * a + m[0][1] * b;
            amps[j] = m[1][0] * a + m[1][1] * b;
        }
    }
}
