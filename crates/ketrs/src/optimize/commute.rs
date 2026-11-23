//! Exact commutation checking for circuit operations.
//!
//! Combines fast algebraic rules (valid for both fixed and parametric gates)
//! with an exact 2×2 / 4×4 matrix commutator check for arbitrary fixed 1- and
//! 2-qubit operations.

use crate::circuit::Operation;
use crate::gate::{Gate, Matrix4};
use crate::observable::Pauli;
use crate::optimize::kak::op_matrix4_on;
use crate::optimize::one_qubit::{
    gate_commutes_with_x, gate_commutes_with_y, gate_is_diagonal, mat2_commutes,
};

/// Whether an operation is diagonal in the computational basis (for any
/// parameter value if `is_param` is true).
pub(crate) fn op_is_diagonal(op: &Operation, is_param: bool, tol: f64) -> bool {
    match op {
        Operation::Gate { gate, .. }
        | Operation::Controlled { gate, .. }
        | Operation::MultiControlled { gate, .. } => {
            if is_param {
                matches!(gate, Gate::Rz(_) | Gate::Phase(_))
            } else {
                gate_is_diagonal(gate, tol)
            }
        }
        Operation::PauliRotation { pauli, .. } => pauli.ops().iter().all(|&(_, p)| p == Pauli::Z),
        Operation::Unitary2 { matrix, .. } => mat4_is_diagonal(matrix, tol),
        Operation::Swap(..) | Operation::ControlledSwap { .. } => false,
    }
}

fn mat4_is_diagonal(m: &Matrix4, tol: f64) -> bool {
    for (r, row) in m.iter().enumerate() {
        for (c, &v) in row.iter().enumerate() {
            if r != c && v.norm() > tol {
                return false;
            }
        }
    }
    true
}

fn mat4_commutes(a: &Matrix4, b: &Matrix4, tol: f64) -> bool {
    for r in 0..4 {
        for c in 0..4 {
            let mut ab = num_complex::Complex64::new(0.0, 0.0);
            let mut ba = ab;
            for k in 0..4 {
                ab += a[r][k] * b[k][c];
                ba += b[r][k] * a[k][c];
            }
            if (ab - ba).norm() > tol {
                return false;
            }
        }
    }
    true
}

fn gates_commute(a: &Gate, b: &Gate, a_param: bool, b_param: bool, tol: f64) -> bool {
    if a_param || b_param {
        let diag = |g: &Gate, p: bool| {
            if p {
                matches!(g, Gate::Rz(_) | Gate::Phase(_))
            } else {
                gate_is_diagonal(g, tol)
            }
        };
        if diag(a, a_param) && diag(b, b_param) {
            return true;
        }
        let is_x = |g: &Gate, p: bool| {
            if p {
                matches!(g, Gate::Rx(_))
            } else {
                gate_commutes_with_x(g, tol)
            }
        };
        if is_x(a, a_param) && is_x(b, b_param) {
            return true;
        }
        let is_y = |g: &Gate, p: bool| {
            if p {
                matches!(g, Gate::Ry(_))
            } else {
                gate_commutes_with_y(g, tol)
            }
        };
        if is_y(a, a_param) && is_y(b, b_param) {
            return true;
        }
        return false;
    }
    mat2_commutes(&a.matrix(), &b.matrix(), tol)
}

/// Whether `a` and `b` commute (`a·b == b·a` for all parameter values when
/// parametric).
pub(crate) fn ops_commute(
    a: &Operation,
    b: &Operation,
    a_param: bool,
    b_param: bool,
    tol: f64,
) -> bool {
    let qa = a.qubits();
    let qb = b.qubits();

    // Disjoint qubits always commute.
    if !qa.iter().any(|q| qb.contains(q)) {
        return true;
    }

    // Two diagonal operations always commute.
    if op_is_diagonal(a, a_param, tol) && op_is_diagonal(b, b_param, tol) {
        return true;
    }

    // Two Pauli rotations commute iff their Pauli strings commute.
    if let (
        Operation::PauliRotation { pauli: pa, .. },
        Operation::PauliRotation { pauli: pb, .. },
    ) = (a, b)
    {
        return pa.commutes_with(pb);
    }

    // Single-qubit gate vs anything else.
    if let Operation::Gate { gate, target } = a {
        return gate_commutes_with_op(gate, *target, a_param, b, b_param, tol);
    }
    if let Operation::Gate { gate, target } = b {
        return gate_commutes_with_op(gate, *target, b_param, a, a_param, tol);
    }

    // Controlled / MultiControlled vs Controlled / MultiControlled:
    // Every control acts diagonally in Z; targets act by their gate.
    if let (Some((ca, ta, ga)), Some((cb, tb, gb))) = (as_controlled(a), as_controlled(b)) {
        let ga_diag = if a_param {
            matches!(ga, Gate::Rz(_) | Gate::Phase(_))
        } else {
            gate_is_diagonal(ga, tol)
        };
        let gb_diag = if b_param {
            matches!(gb, Gate::Rz(_) | Gate::Phase(_))
        } else {
            gate_is_diagonal(gb, tol)
        };
        // If ta is in cb, ga must be diagonal on ta (which commutes with cb).
        // If tb is in ca, gb must be diagonal on tb.
        // If ta == tb, ga and gb must commute.
        let ok_a_in_b = !cb.contains(&ta) || ga_diag;
        let ok_b_in_a = !ca.contains(&tb) || gb_diag;
        let ok_targets = ta != tb || gates_commute(ga, gb, a_param, b_param, tol);
        if ok_a_in_b && ok_b_in_a && ok_targets {
            return true;
        }
    }

    // ControlledSwap vs diagonal gate on its controls.
    if let (
        Operation::ControlledSwap {
            controls,
            a: sa,
            b: sb,
        },
        other,
    )
    | (
        other,
        Operation::ControlledSwap {
            controls,
            a: sa,
            b: sb,
        },
    ) = (a, b)
    {
        let other_param = if std::ptr::eq(other, a) {
            a_param
        } else {
            b_param
        };
        let oq = other.qubits();
        if !oq.contains(sa) && !oq.contains(sb) && op_is_diagonal(other, other_param, tol) {
            // Only touches control qubits diagonally.
            let _ = controls;
            return true;
        }
    }

    // Exact 4×4 matrix check when both ops are fixed and their union has ≤ 2 qubits.
    if !a_param && !b_param && !a.is_multi_controlled() && !b.is_multi_controlled() {
        let mut union = qa.clone();
        for &q in &qb {
            if !union.contains(&q) {
                union.push(q);
            }
        }
        if union.len() == 2 {
            let pair = [union[0], union[1]];
            let ma = op_matrix4_on(a, pair);
            let mb = op_matrix4_on(b, pair);
            return mat4_commutes(&ma, &mb, tol);
        }
    }

    false
}

fn as_controlled(op: &Operation) -> Option<(Vec<usize>, usize, &Gate)> {
    match op {
        Operation::Controlled {
            gate,
            control,
            target,
        } => Some((vec![*control], *target, gate)),
        Operation::MultiControlled {
            gate,
            controls,
            target,
        } => Some((controls.clone(), *target, gate)),
        _ => None,
    }
}

fn gate_commutes_with_op(
    gate: &Gate,
    q: usize,
    g_param: bool,
    op: &Operation,
    op_param: bool,
    tol: f64,
) -> bool {
    let g_diag = if g_param {
        matches!(gate, Gate::Rz(_) | Gate::Phase(_))
    } else {
        gate_is_diagonal(gate, tol)
    };
    let g_x = if g_param {
        matches!(gate, Gate::Rx(_))
    } else {
        gate_commutes_with_x(gate, tol)
    };
    let g_y = if g_param {
        matches!(gate, Gate::Ry(_))
    } else {
        gate_commutes_with_y(gate, tol)
    };

    match op {
        Operation::Gate {
            gate: other,
            target,
        } => *target != q || gates_commute(gate, other, g_param, op_param, tol),
        Operation::Controlled {
            gate: other,
            control,
            target,
        } => {
            if *control == q {
                g_diag
            } else if *target == q {
                gates_commute(gate, other, g_param, op_param, tol)
            } else {
                true
            }
        }
        Operation::MultiControlled {
            gate: other,
            controls,
            target,
        } => {
            if controls.contains(&q) {
                g_diag
            } else if *target == q {
                gates_commute(gate, other, g_param, op_param, tol)
            } else {
                true
            }
        }
        Operation::ControlledSwap { controls, a, b } => {
            if controls.contains(&q) {
                g_diag
            } else {
                *a != q && *b != q
            }
        }
        Operation::Swap(a, b) => *a != q && *b != q,
        Operation::PauliRotation { pauli, .. } => {
            match pauli.ops().iter().find(|&&(pq, _)| pq == q) {
                None => true,
                Some(&(_, Pauli::I)) => true,
                Some(&(_, Pauli::X)) => g_x,
                Some(&(_, Pauli::Y)) => g_y,
                Some(&(_, Pauli::Z)) => g_diag,
            }
        }
        Operation::Unitary2 { matrix, qubits } => {
            if !qubits.contains(&q) {
                return true;
            }
            if g_param {
                // Parametric 1q gate commutes with Unitary2 if Unitary2 is diagonal and g is diagonal.
                return g_diag && mat4_is_diagonal(matrix, tol);
            }
            let mg = op_matrix4_on(
                &Operation::Gate {
                    gate: *gate,
                    target: q,
                },
                *qubits,
            );
            mat4_commutes(&mg, matrix, tol)
        }
    }
}
