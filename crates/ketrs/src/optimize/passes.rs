//! Unitary circuit optimization passes.

use crate::circuit::{Binding, Circuit, Operation};
use crate::gate::{Gate, Matrix2, Matrix4};
use crate::observable::{Pauli, PauliString};
use crate::optimize::commute::{op_is_diagonal, ops_commute};
use crate::optimize::kak::{kron2, mat4_is_real, op_matrix4_on, synthesize_2q};
use crate::optimize::one_qubit::{
    I2, gate_is_real, mat2_identity_phase, mat2_mul, op_matrix2, phase_is_real,
    simplify_trivial_1q, synthesize_1q, wrap_2pi, wrap_pi,
};
use crate::optimize::{Basis, OptimizationLevel};
use num_complex::Complex64 as C64;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::f64::consts::{FRAC_PI_2, PI, TAU};

/// Parameter binding attached directly to a node (without an op index).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ParamBinding {
    pub parameter: usize,
    pub scale: f64,
    pub offset: f64,
}

/// One operation in a unitary segment, with optional parameter binding.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Node {
    pub op: Operation,
    pub binding: Option<ParamBinding>,
}

/// A unitary circuit represented as a flat list of nodes plus global phase.
#[derive(Clone, Debug)]
pub(crate) struct Segment {
    pub n_qubits: usize,
    pub n_parameters: usize,
    pub n_clbits: usize,
    pub nodes: Vec<Node>,
    pub global_phase: f64,
}

impl Segment {
    pub(crate) fn from_circuit(circuit: &Circuit) -> Self {
        let mut binding_of = vec![None; circuit.len()];
        for b in circuit.bindings() {
            binding_of[b.op] = Some(ParamBinding {
                parameter: b.parameter,
                scale: b.scale,
                offset: b.offset,
            });
        }
        let nodes = circuit
            .ops()
            .iter()
            .cloned()
            .zip(binding_of)
            .map(|(op, binding)| Node { op, binding })
            .collect();
        Segment {
            n_qubits: circuit.n_qubits(),
            n_parameters: circuit.n_parameters(),
            n_clbits: circuit.n_clbits(),
            nodes,
            global_phase: circuit.global_phase(),
        }
    }

    pub(crate) fn into_circuit(self) -> Circuit {
        let mut out = Circuit::with_clbits(self.n_qubits, self.n_clbits);
        out.set_global_phase(wrap_pi(self.global_phase));
        for _ in 0..self.n_parameters {
            out.parameter();
        }
        for node in self.nodes {
            let binding = node.binding.map(|b| Binding {
                op: 0,
                parameter: b.parameter,
                scale: b.scale,
                offset: b.offset,
            });
            out.push_with_binding(node.op, binding);
        }
        out
    }

    /// Whether every operation and the global phase are representable in
    /// `RealStatevector`.
    pub(crate) fn is_real(&self, tol: f64) -> bool {
        if !phase_is_real(self.global_phase, tol) {
            return false;
        }
        self.nodes
            .iter()
            .all(|n| op_is_real(&n.op, n.binding.is_some(), tol))
    }

    /// `(out_of_basis, weighted_multi_qubit_count, total_ops, depth)` cost tuple.
    ///
    /// Under [`Basis::Cx`], weight-`w` `PauliRotation`s (`w >= 2`), `Swap`s,
    /// `Unitary2`s, and non-`X` `Controlled` gates are counted as out-of-basis
    /// and weighted by their actual CNOT cost (`2*(w-1)` for `PauliRotation`).
    pub(crate) fn cost(&self, basis: Basis) -> (usize, usize, usize, usize) {
        let mut out_of_basis = 0;
        let mut multi = 0;
        let mut depths = vec![0usize; self.n_qubits];
        for n in &self.nodes {
            let qs = n.op.qubits();
            if matches!(n.op, Operation::Unitary2 { .. }) {
                out_of_basis += 1;
            } else if basis == Basis::Cx {
                match &n.op {
                    Operation::PauliRotation { pauli, .. } if pauli.weight() >= 2 => {
                        out_of_basis += 1;
                    }
                    Operation::Swap(..) => {
                        out_of_basis += 1;
                    }
                    Operation::Controlled { gate, .. }
                        if n.binding.is_none() && !matches!(gate, Gate::X) =>
                    {
                        out_of_basis += 1;
                    }
                    _ => {}
                }
            }
            match &n.op {
                Operation::PauliRotation { pauli, .. } if pauli.weight() >= 2 => {
                    multi += if basis == Basis::Cx {
                        2 * (pauli.weight() - 1)
                    } else {
                        1
                    };
                }
                Operation::Swap(..) if basis == Basis::Cx => {
                    multi += 3;
                }
                _ if qs.len() >= 2 => {
                    multi += 1;
                }
                _ => {}
            }
            let d = qs.iter().map(|&q| depths[q]).max().unwrap_or(0) + 1;
            for &q in &qs {
                depths[q] = d;
            }
        }
        let depth = depths.into_iter().max().unwrap_or(0);
        (out_of_basis, multi, self.nodes.len(), depth)
    }
}

fn op_is_real(op: &Operation, is_param: bool, tol: f64) -> bool {
    match op {
        Operation::Gate { gate, .. }
        | Operation::Controlled { gate, .. }
        | Operation::MultiControlled { gate, .. } => {
            if is_param {
                matches!(gate, Gate::Ry(_))
            } else {
                gate_is_real(gate, tol)
            }
        }
        Operation::Swap(..) | Operation::ControlledSwap { .. } => true,
        Operation::Unitary2 { matrix, .. } => mat4_is_real(matrix, tol),
        Operation::PauliRotation { pauli, theta } => {
            // exp(-i θ/2 P) = cos(θ/2) I - i sin(θ/2) P is real iff P has an
            // odd number of Ys (so -i P is real), or θ is a multiple of 2π.
            let n_y = pauli.ops().iter().filter(|&&(_, p)| p == Pauli::Y).count();
            if n_y % 2 == 1 {
                true
            } else if is_param {
                false
            } else {
                (theta / 2.0).sin().abs() <= tol
            }
        }
    }
}

/// Normalize a single operation, simplifying trivial identities, canonical
/// qubit orderings, weight-0/1 Pauli rotations, and trivial controlled gates.
pub(crate) fn canonicalize_and_remove_identities(seg: &mut Segment, tol: f64) -> bool {
    let mut changed = false;
    let mut out = Vec::with_capacity(seg.nodes.len());

    for mut node in std::mem::take(&mut seg.nodes) {
        // Zero-scale parameter binding is actually a constant angle.
        if let Some(b) = node.binding
            && b.scale.abs() <= tol
        {
            set_op_angle(&mut node.op, b.offset);
            node.binding = None;
            changed = true;
        }

        // Simplify MultiControlled with 0/1 controls and ControlledSwap with 0 controls.
        let before_simp = node.op.clone();
        node.op = node.op.simplified();
        if node.op != before_simp {
            changed = true;
        }

        // Canonicalize symmetric qubit orders (so inverses/merges match!).
        match &mut node.op {
            Operation::Swap(a, b) => {
                if *a > *b {
                    std::mem::swap(a, b);
                    changed = true;
                }
            }
            Operation::ControlledSwap { controls, a, b } => {
                let mut sorted = controls.clone();
                sorted.sort_unstable();
                if sorted != *controls {
                    *controls = sorted;
                    changed = true;
                }
                if *a > *b {
                    std::mem::swap(a, b);
                    changed = true;
                }
            }
            Operation::MultiControlled {
                gate,
                controls,
                target,
            } => {
                // For symmetric phase gates (Z, Phase, S, Sdg, T, Tdg), all
                // qubits are interchangeable: sort controls + target together.
                let symmetric = matches!(
                    gate,
                    Gate::Z | Gate::S | Gate::Sdg | Gate::T | Gate::Tdg | Gate::Phase(_)
                );
                if symmetric {
                    let mut all = controls.clone();
                    all.push(*target);
                    all.sort_unstable();
                    let new_target = all.pop().unwrap();
                    if all != *controls || new_target != *target {
                        *controls = all;
                        *target = new_target;
                        changed = true;
                    }
                } else {
                    let mut sorted = controls.clone();
                    sorted.sort_unstable();
                    if sorted != *controls {
                        *controls = sorted;
                        changed = true;
                    }
                }
            }
            Operation::Controlled {
                gate,
                control,
                target,
            } => {
                let symmetric = matches!(
                    gate,
                    Gate::Z | Gate::S | Gate::Sdg | Gate::T | Gate::Tdg | Gate::Phase(_)
                );
                if symmetric && *control > *target {
                    std::mem::swap(control, target);
                    changed = true;
                }
            }
            _ => {}
        }

        // Simplify Pauli rotations.
        if let Operation::PauliRotation { pauli, theta } = &node.op {
            if pauli.ops().is_empty() && node.binding.is_none() {
                seg.global_phase -= *theta / 2.0;
                changed = true;
                continue;
            }
            if let &[(target, p)] = pauli.ops() {
                let gate = match p {
                    Pauli::I => {
                        if node.binding.is_none() {
                            seg.global_phase -= *theta / 2.0;
                            changed = true;
                            continue;
                        }
                        out.push(node);
                        continue;
                    }
                    Pauli::X => Gate::Rx(*theta),
                    Pauli::Y => Gate::Ry(*theta),
                    Pauli::Z => Gate::Rz(*theta),
                };
                node.op = Operation::Gate { gate, target };
                changed = true;
            } else if node.binding.is_none() {
                let w = wrap_2pi(*theta);
                if w.abs() <= tol {
                    changed = true;
                    continue;
                }
                if (w.abs() - TAU).abs() <= tol {
                    seg.global_phase += PI;
                    changed = true;
                    continue;
                }
                // exp(-i (±π)/2 P) = ∓i P -> single-qubit Paulis!
                if (w - PI).abs() <= tol {
                    seg.global_phase -= FRAC_PI_2;
                    for &(q, p) in pauli.ops() {
                        if let Some(g) = pauli_to_gate(p) {
                            out.push(Node {
                                op: Operation::Gate { gate: g, target: q },
                                binding: None,
                            });
                        }
                    }
                    changed = true;
                    continue;
                }
                if (w + PI).abs() <= tol {
                    seg.global_phase += FRAC_PI_2;
                    for &(q, p) in pauli.ops() {
                        if let Some(g) = pauli_to_gate(p) {
                            out.push(Node {
                                op: Operation::Gate { gate: g, target: q },
                                binding: None,
                            });
                        }
                    }
                    changed = true;
                    continue;
                }
            }
        }

        // Simplify fixed 1-qubit gates.
        if node.binding.is_none()
            && let Operation::Gate { gate, target } = node.op
            && let Some((simplified, phase)) = simplify_trivial_1q(&gate, tol)
        {
            seg.global_phase += phase;
            changed = true;
            if let Some(g) = simplified {
                out.push(Node {
                    op: Operation::Gate { gate: g, target },
                    binding: None,
                });
            }
            continue;
        }

        // Simplify fixed Controlled / MultiControlled where target gate is e^{iφ} I.
        if node.binding.is_none() {
            match &node.op {
                Operation::Controlled {
                    gate,
                    control,
                    target: _,
                } => {
                    if let Some(phi) = mat2_identity_phase(&gate.matrix(), tol) {
                        changed = true;
                        if phi.abs() > tol {
                            let (g, gp) = synthesize_1q(&Gate::Phase(phi).matrix(), tol);
                            seg.global_phase += gp;
                            if let Some(g) = g {
                                out.push(Node {
                                    op: Operation::Gate {
                                        gate: g,
                                        target: *control,
                                    },
                                    binding: None,
                                });
                            }
                        }
                        continue;
                    }
                }
                Operation::MultiControlled {
                    gate,
                    controls,
                    target: _,
                } => {
                    if let Some(phi) = mat2_identity_phase(&gate.matrix(), tol) {
                        changed = true;
                        if phi.abs() > tol {
                            let mut cs = controls.clone();
                            cs.sort_unstable();
                            match cs.pop() {
                                None => seg.global_phase += phi,
                                Some(last) => {
                                    let phase_gate = if (phi.abs() - PI).abs() <= tol {
                                        Gate::Z
                                    } else {
                                        Gate::Phase(phi)
                                    };
                                    let op = Operation::MultiControlled {
                                        gate: phase_gate,
                                        controls: cs,
                                        target: last,
                                    }
                                    .simplified();
                                    out.push(Node { op, binding: None });
                                }
                            }
                        }
                        continue;
                    }
                }
                Operation::Unitary2 { matrix, .. } => {
                    if let Some(phase) = mat4_identity_phase(matrix, tol) {
                        seg.global_phase += phase;
                        changed = true;
                        continue;
                    }
                }
                _ => {}
            }
        }

        out.push(node);
    }

    seg.nodes = out;
    seg.global_phase = wrap_pi(seg.global_phase);
    changed
}

fn mat4_identity_phase(m: &Matrix4, tol: f64) -> Option<f64> {
    let p = m[0][0];
    for (r, row) in m.iter().enumerate() {
        for (c, &v) in row.iter().enumerate() {
            let want = if r == c { p } else { C64::new(0.0, 0.0) };
            if (v - want).norm() > tol {
                return None;
            }
        }
    }
    let phase = wrap_pi(p.arg());
    Some(if phase.abs() <= tol { 0.0 } else { phase })
}

fn pauli_to_gate(p: Pauli) -> Option<Gate> {
    match p {
        Pauli::I => None,
        Pauli::X => Some(Gate::X),
        Pauli::Y => Some(Gate::Y),
        Pauli::Z => Some(Gate::Z),
    }
}

fn set_op_angle(op: &mut Operation, angle: f64) {
    match op {
        Operation::Gate { gate, .. }
        | Operation::Controlled { gate, .. }
        | Operation::MultiControlled { gate, .. } => {
            *gate = match gate {
                Gate::Rx(_) => Gate::Rx(angle),
                Gate::Ry(_) => Gate::Ry(angle),
                Gate::Rz(_) => Gate::Rz(angle),
                Gate::Phase(_) => Gate::Phase(angle),
                _ => *gate,
            };
        }
        Operation::PauliRotation { theta, .. } => *theta = angle,
        _ => {}
    }
}

/// Outcome of trying to combine two adjacent (or commuted-adjacent) nodes `a`
/// followed by `b` (`b · a` in matrix order).
#[allow(clippy::large_enum_variant)]
enum CombineResult {
    /// Cannot combine.
    None,
    /// Cancel completely, adding `phase` to `global_phase`.
    Cancel(f64),
    /// Replace both with a single node, adding `phase` to `global_phase`.
    Merge(Node, f64),
}

/// Try to cancel or merge `a` followed by `b` acting on the same qubits.
fn try_combine(a: &Node, b: &Node, tol: f64) -> CombineResult {
    // Same-axis rotations (works for BOTH fixed and parametric on the same parameter!).
    if let Some(res) = try_combine_rotations(a, b, tol) {
        return res;
    }

    // Parametric nodes that didn't merge as rotations cannot be combined via matrices.
    if a.binding.is_some() || b.binding.is_some() {
        return CombineResult::None;
    }

    // Fixed single-qubit gates on the same qubit: check exact inverse or exact merge.
    if let (
        Operation::Gate {
            gate: ga,
            target: ta,
        },
        Operation::Gate {
            gate: gb,
            target: tb,
        },
    ) = (&a.op, &b.op)
        && ta == tb
    {
        let prod = mat2_mul(&gb.matrix(), &ga.matrix());
        if let Some(phase) = mat2_identity_phase(&prod, tol) {
            return CombineResult::Cancel(phase);
        }
    }

    // Identical self-inverse swaps or controlled swaps.
    match (&a.op, &b.op) {
        (Operation::Swap(a0, a1), Operation::Swap(b0, b1))
            if (a0 == b0 && a1 == b1) || (a0 == b1 && a1 == b0) =>
        {
            return CombineResult::Cancel(0.0);
        }
        (
            Operation::ControlledSwap {
                controls: ca,
                a: a0,
                b: a1,
            },
            Operation::ControlledSwap {
                controls: cb,
                a: b0,
                b: b1,
            },
        ) if same_qubit_set(ca, cb) && ((a0 == b0 && a1 == b1) || (a0 == b1 && a1 == b0)) => {
            return CombineResult::Cancel(0.0);
        }
        (
            Operation::Controlled {
                gate: ga,
                control: ca,
                target: ta,
            },
            Operation::Controlled {
                gate: gb,
                control: cb,
                target: tb,
            },
        ) => {
            // Symmetric CZ / CP / CS / CT with swapped control & target.
            if ca == tb
                && ta == cb
                && let (Some(la), Some(lb)) = (phase_angle(ga), phase_angle(gb))
            {
                let sum = wrap_pi(la + lb);
                if sum.abs() <= tol {
                    return CombineResult::Cancel(0.0);
                }
                let gate = phase_gate_from_angle(sum, tol);
                let (c, t) = if *ca < *ta { (*ca, *ta) } else { (*ta, *ca) };
                return CombineResult::Merge(
                    Node {
                        op: Operation::Controlled {
                            gate,
                            control: c,
                            target: t,
                        },
                        binding: None,
                    },
                    0.0,
                );
            }
            if ca == cb && ta == tb {
                let prod = mat2_mul(&gb.matrix(), &ga.matrix());
                if let Some(phi) = mat2_identity_phase(&prod, tol) {
                    if phi.abs() <= tol {
                        return CombineResult::Cancel(0.0);
                    }
                    // Controlled(e^{iφ} I, c, t) = Phase(φ, c)!
                    let (g, gp) = synthesize_1q(&Gate::Phase(phi).matrix(), tol);
                    return match g {
                        None => CombineResult::Cancel(gp),
                        Some(gate) => CombineResult::Merge(
                            Node {
                                op: Operation::Gate { gate, target: *ca },
                                binding: None,
                            },
                            gp,
                        ),
                    };
                }
                // Merge two Controlled gates on the same (control, target) into one!
                let (syn, gp) = synthesize_1q(&prod, tol);
                if gp.abs() <= tol
                    && let Some(g) = syn
                {
                    return CombineResult::Merge(
                        Node {
                            op: Operation::Controlled {
                                gate: g,
                                control: *ca,
                                target: *ta,
                            },
                            binding: None,
                        },
                        0.0,
                    );
                }
                // Non-zero phase on controlled gate: keep exact matrix via Gate::Unitary
                // or keep Controlled(Unitary(prod)) which is 1 2q op instead of 2!
                return CombineResult::Merge(
                    Node {
                        op: Operation::Controlled {
                            gate: Gate::Unitary(prod),
                            control: *ca,
                            target: *ta,
                        },
                        binding: None,
                    },
                    0.0,
                );
            }
        }
        (
            Operation::MultiControlled {
                gate: ga,
                controls: ca,
                target: ta,
            },
            Operation::MultiControlled {
                gate: gb,
                controls: cb,
                target: tb,
            },
        ) => {
            if ta == tb && same_qubit_set(ca, cb) {
                let prod = mat2_mul(&gb.matrix(), &ga.matrix());
                if let Some(phi) = mat2_identity_phase(&prod, tol) {
                    if phi.abs() <= tol {
                        return CombineResult::Cancel(0.0);
                    }
                    let mut cs = ca.clone();
                    cs.sort_unstable();
                    match cs.pop() {
                        None => return CombineResult::Cancel(phi),
                        Some(last) => {
                            let gate = phase_gate_from_angle(phi, tol);
                            let op = Operation::MultiControlled {
                                gate,
                                controls: cs,
                                target: last,
                            }
                            .simplified();
                            return CombineResult::Merge(Node { op, binding: None }, 0.0);
                        }
                    }
                }
                let (syn, gp) = synthesize_1q(&prod, tol);
                let gate = if gp.abs() <= tol {
                    syn.unwrap_or(Gate::Unitary(prod))
                } else {
                    Gate::Unitary(prod)
                };
                let mut cs = ca.clone();
                cs.sort_unstable();
                return CombineResult::Merge(
                    Node {
                        op: Operation::MultiControlled {
                            gate,
                            controls: cs,
                            target: *ta,
                        },
                        binding: None,
                    },
                    0.0,
                );
            }
            // Symmetric MultiControlled phase gates (e.g. CCZ with permuted qubits).
            if let (Some(la), Some(lb)) = (phase_angle(ga), phase_angle(gb)) {
                let mut qa = ca.clone();
                qa.push(*ta);
                let mut qb = cb.clone();
                qb.push(*tb);
                if same_qubit_set(&qa, &qb) {
                    let sum = wrap_pi(la + lb);
                    if sum.abs() <= tol {
                        return CombineResult::Cancel(0.0);
                    }
                    qa.sort_unstable();
                    let last = qa.pop().unwrap();
                    return CombineResult::Merge(
                        Node {
                            op: Operation::MultiControlled {
                                gate: phase_gate_from_angle(sum, tol),
                                controls: qa,
                                target: last,
                            }
                            .simplified(),
                            binding: None,
                        },
                        0.0,
                    );
                }
            }
        }
        (
            Operation::Unitary2 {
                matrix: ma,
                qubits: qa,
            },
            Operation::Unitary2 {
                matrix: mb,
                qubits: qb,
            },
        ) if (qa[0] == qb[0] && qa[1] == qb[1]) || (qa[0] == qb[1] && qa[1] == qb[0]) => {
            let mb_aligned = op_matrix4_on(&b.op, *qa);
            let mut prod = [[C64::new(0.0, 0.0); 4]; 4];
            for r in 0..4 {
                for c in 0..4 {
                    for k in 0..4 {
                        prod[r][c] += mb_aligned[r][k] * ma[k][c];
                    }
                }
            }
            if let Some(phase) = mat4_identity_phase(&prod, tol) {
                return CombineResult::Cancel(phase);
            }
        }
        _ => {}
    }

    CombineResult::None
}

fn phase_angle(g: &Gate) -> Option<f64> {
    match *g {
        Gate::Z => Some(PI),
        Gate::S => Some(FRAC_PI_2),
        Gate::Sdg => Some(-FRAC_PI_2),
        Gate::T => Some(PI / 4.0),
        Gate::Tdg => Some(-PI / 4.0),
        Gate::Phase(l) => Some(l),
        _ => None,
    }
}

fn phase_gate_from_angle(angle: f64, tol: f64) -> Gate {
    let w = wrap_pi(angle);
    if (w.abs() - PI).abs() <= tol {
        Gate::Z
    } else if (w - FRAC_PI_2).abs() <= tol {
        Gate::S
    } else if (w + FRAC_PI_2).abs() <= tol {
        Gate::Sdg
    } else if (w - PI / 4.0).abs() <= tol {
        Gate::T
    } else if (w + PI / 4.0).abs() <= tol {
        Gate::Tdg
    } else {
        Gate::Phase(w)
    }
}

fn same_qubit_set(a: &[usize], b: &[usize]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut sa = a.to_vec();
    let mut sb = b.to_vec();
    sa.sort_unstable();
    sb.sort_unstable();
    sa == sb
}

/// Rotation family for merging fixed & parametric same-axis rotations.
#[derive(Clone, PartialEq, Eq)]
enum RotKind {
    Rx(usize),
    Ry(usize),
    Rz(usize),
    Phase(usize),
    CRx(usize, usize),
    CRy(usize, usize),
    CRz(usize, usize),
    CP(usize, usize), // canonicalized (min, max)
    Pauli(PauliString),
}

impl RotKind {
    fn is_phase_period_2pi(&self) -> bool {
        matches!(self, RotKind::Phase(_) | RotKind::CP(_, _))
    }

    fn build(&self, angle: f64) -> Operation {
        match self {
            RotKind::Rx(q) => Operation::Gate {
                gate: Gate::Rx(angle),
                target: *q,
            },
            RotKind::Ry(q) => Operation::Gate {
                gate: Gate::Ry(angle),
                target: *q,
            },
            RotKind::Rz(q) => Operation::Gate {
                gate: Gate::Rz(angle),
                target: *q,
            },
            RotKind::Phase(q) => Operation::Gate {
                gate: Gate::Phase(angle),
                target: *q,
            },
            RotKind::CRx(c, t) => Operation::Controlled {
                gate: Gate::Rx(angle),
                control: *c,
                target: *t,
            },
            RotKind::CRy(c, t) => Operation::Controlled {
                gate: Gate::Ry(angle),
                control: *c,
                target: *t,
            },
            RotKind::CRz(c, t) => Operation::Controlled {
                gate: Gate::Rz(angle),
                control: *c,
                target: *t,
            },
            RotKind::CP(c, t) => Operation::Controlled {
                gate: Gate::Phase(angle),
                control: *c,
                target: *t,
            },
            RotKind::Pauli(p) => Operation::PauliRotation {
                pauli: p.clone(),
                theta: angle,
            },
        }
    }
}

/// Extract `(RotKind, fixed_angle, extra_global_phase)` from a node if it is
/// a rotation (fixed or parametric).
fn as_rotation(node: &Node) -> Option<(RotKind, f64, f64)> {
    match &node.op {
        Operation::Gate { gate, target } => match *gate {
            Gate::Rx(v) => Some((RotKind::Rx(*target), v, 0.0)),
            Gate::Ry(v) => Some((RotKind::Ry(*target), v, 0.0)),
            Gate::Rz(v) => Some((RotKind::Rz(*target), v, 0.0)),
            Gate::Phase(v) => Some((RotKind::Phase(*target), v, 0.0)),
            Gate::X if node.binding.is_none() => Some((RotKind::Rx(*target), PI, FRAC_PI_2)),
            Gate::Y if node.binding.is_none() => Some((RotKind::Ry(*target), PI, FRAC_PI_2)),
            Gate::Z if node.binding.is_none() => Some((RotKind::Phase(*target), PI, 0.0)),
            Gate::S if node.binding.is_none() => Some((RotKind::Phase(*target), FRAC_PI_2, 0.0)),
            Gate::Sdg if node.binding.is_none() => Some((RotKind::Phase(*target), -FRAC_PI_2, 0.0)),
            Gate::T if node.binding.is_none() => Some((RotKind::Phase(*target), PI / 4.0, 0.0)),
            Gate::Tdg if node.binding.is_none() => Some((RotKind::Phase(*target), -PI / 4.0, 0.0)),
            _ => None,
        },
        Operation::Controlled {
            gate,
            control,
            target,
        } => match *gate {
            Gate::Rx(v) => Some((RotKind::CRx(*control, *target), v, 0.0)),
            Gate::Ry(v) => Some((RotKind::CRy(*control, *target), v, 0.0)),
            Gate::Rz(v) => Some((RotKind::CRz(*control, *target), v, 0.0)),
            Gate::Phase(v) => {
                let (c, t) = if control < target {
                    (*control, *target)
                } else {
                    (*target, *control)
                };
                Some((RotKind::CP(c, t), v, 0.0))
            }
            Gate::Z | Gate::S | Gate::Sdg | Gate::T | Gate::Tdg if node.binding.is_none() => {
                let (c, t) = if control < target {
                    (*control, *target)
                } else {
                    (*target, *control)
                };
                Some((RotKind::CP(c, t), phase_angle(gate).unwrap(), 0.0))
            }
            _ => None,
        },
        Operation::PauliRotation { pauli, theta } => {
            Some((RotKind::Pauli(pauli.clone()), *theta, 0.0))
        }
        _ => None,
    }
}

fn try_combine_rotations(a: &Node, b: &Node, tol: f64) -> Option<CombineResult> {
    let (mut ka, va, mut pa) = as_rotation(a)?;
    let (mut kb, vb, mut pb) = as_rotation(b)?;

    // Allow Rz and Phase on the same qubit to merge (Phase(λ) = e^{iλ/2} Rz(λ))!
    match (&ka, &kb) {
        (RotKind::Rz(qa), RotKind::Phase(qb)) if qa == qb && b.binding.is_none() => {
            kb = RotKind::Rz(*qb);
            pb += vb / 2.0;
        }
        (RotKind::Phase(qa), RotKind::Rz(qb)) if qa == qb && a.binding.is_none() => {
            ka = RotKind::Rz(*qa);
            pa += va / 2.0;
        }
        _ => {}
    }

    if ka != kb {
        return None;
    }

    let extra_phase = pa + pb;
    let combined_binding = match (a.binding, b.binding) {
        (None, None) => None,
        (Some(ba), None) => Some(ParamBinding {
            parameter: ba.parameter,
            scale: ba.scale,
            offset: ba.offset + vb,
        }),
        (None, Some(bb)) => Some(ParamBinding {
            parameter: bb.parameter,
            scale: bb.scale,
            offset: bb.offset + va,
        }),
        (Some(ba), Some(bb)) => {
            if ba.parameter != bb.parameter {
                return None;
            }
            let scale = ba.scale + bb.scale;
            let offset = ba.offset + bb.offset;
            if scale.abs() <= tol {
                None // Parametric parts cancelled! Only constant `offset` remains.
            } else {
                Some(ParamBinding {
                    parameter: ba.parameter,
                    scale,
                    offset,
                })
            }
        }
    };

    if let Some(binding) = combined_binding {
        let mut op = ka.build(binding.offset);
        set_op_angle(&mut op, binding.offset);
        return Some(CombineResult::Merge(
            Node {
                op,
                binding: Some(binding),
            },
            extra_phase,
        ));
    }

    // Both fixed (or parametric parts cancelled out).
    let total = match (a.binding, b.binding) {
        (Some(ba), Some(bb)) => ba.offset + bb.offset,
        _ => va + vb,
    };
    if ka.is_phase_period_2pi() {
        let w = wrap_pi(total);
        if w.abs() <= tol {
            return Some(CombineResult::Cancel(extra_phase));
        }
        let op = match ka {
            RotKind::Phase(q) => Operation::Gate {
                gate: phase_gate_from_angle(w, tol),
                target: q,
            },
            RotKind::CP(c, t) => Operation::Controlled {
                gate: phase_gate_from_angle(w, tol),
                control: c,
                target: t,
            },
            _ => unreachable!(),
        };
        Some(CombineResult::Merge(
            Node { op, binding: None },
            extra_phase,
        ))
    } else {
        let w = wrap_2pi(total);
        if w.abs() <= tol {
            return Some(CombineResult::Cancel(extra_phase));
        }
        if (w.abs() - TAU).abs() <= tol {
            return match ka {
                // Controlled(-I) = Z on control!
                RotKind::CRx(c, _) | RotKind::CRy(c, _) | RotKind::CRz(c, _) => {
                    Some(CombineResult::Merge(
                        Node {
                            op: Operation::Gate {
                                gate: Gate::Z,
                                target: c,
                            },
                            binding: None,
                        },
                        extra_phase,
                    ))
                }
                _ => Some(CombineResult::Cancel(extra_phase + PI)),
            };
        }
        // Clean single-qubit synthesis if it became a named gate (e.g. Rz(π) -> Z).
        let op = ka.build(w);
        if let Operation::Gate { gate, target } = op {
            let (syn, gp) = synthesize_1q(&gate.matrix(), tol);
            return match syn {
                None => Some(CombineResult::Cancel(extra_phase + gp)),
                Some(g) => Some(CombineResult::Merge(
                    Node {
                        op: Operation::Gate { gate: g, target },
                        binding: None,
                    },
                    extra_phase + gp,
                )),
            };
        }
        Some(CombineResult::Merge(
            Node { op, binding: None },
            extra_phase,
        ))
    }
}

/// Pass: cancel adjacent inverses and merge adjacent same-axis rotations
/// (respecting qubit dependencies in O(N)).
pub(crate) fn merge_and_cancel_adjacent(seg: &mut Segment, tol: f64) -> bool {
    let mut changed = false;
    let mut nodes: Vec<Option<Node>> = Vec::with_capacity(seg.nodes.len());
    let mut last_on: Vec<Option<usize>> = vec![None; seg.n_qubits];

    for node in std::mem::take(&mut seg.nodes) {
        let qs = node.op.qubits();
        // Check if every qubit of `node` currently points to the same live predecessor.
        let same_pred = qs
            .first()
            .and_then(|&q0| last_on[q0])
            .filter(|&p| qs.iter().all(|&q| last_on[q] == Some(p)));

        if let Some(prev_idx) = same_pred
            && let Some(prev_node) = &nodes[prev_idx]
        {
            match try_combine(prev_node, &node, tol) {
                CombineResult::Cancel(phase) => {
                    seg.global_phase += phase;
                    nodes[prev_idx] = None;
                    // Recompute last_on for affected qubits by scanning backward.
                    recompute_last_on(&nodes, &qs, &mut last_on);
                    changed = true;
                    continue;
                }
                CombineResult::Merge(merged, phase) => {
                    seg.global_phase += phase;
                    let old_qs = prev_node.op.qubits();
                    let new_qs = merged.op.qubits();
                    nodes[prev_idx] = Some(merged);
                    if old_qs != new_qs {
                        recompute_last_on(&nodes, &old_qs, &mut last_on);
                    }
                    changed = true;
                    continue;
                }
                CombineResult::None => {}
            }
        }

        let idx = nodes.len();
        for &q in &qs {
            last_on[q] = Some(idx);
        }
        nodes.push(Some(node));
    }

    seg.nodes = nodes.into_iter().flatten().collect();
    seg.global_phase = wrap_pi(seg.global_phase);
    changed
}

fn recompute_last_on(nodes: &[Option<Node>], qubits: &[usize], last_on: &mut [Option<usize>]) {
    for &q in qubits {
        last_on[q] = None;
        for (idx, maybe) in nodes.iter().enumerate().rev() {
            if let Some(n) = maybe
                && n.op.qubits().contains(&q)
            {
                last_on[q] = Some(idx);
                break;
            }
        }
    }
}

/// Pass: consolidate maximal runs of fixed 1-qubit gates on each qubit into
/// 0 or 1 minimal canonical gate.
pub(crate) fn optimize_1q_runs(seg: &mut Segment, tol: f64) -> bool {
    let n = seg.n_qubits;
    let mut changed = false;
    let mut out: Vec<Option<Node>> = Vec::with_capacity(seg.nodes.len());

    // Per-qubit active run: (first_slot_in_out, accumulated_matrix, count).
    let mut run: Vec<Option<(usize, Matrix2, usize)>> = vec![None; n];

    let flush_qubit = |q: usize,
                       run: &mut [Option<(usize, Matrix2, usize)>],
                       out: &mut [Option<Node>],
                       global_phase: &mut f64,
                       changed: &mut bool| {
        if let Some((slot, mat, count)) = run[q].take() {
            let needs_synth = count > 1
                || matches!(
                    out[slot].as_ref().map(|n| &n.op),
                    Some(Operation::Gate {
                        gate: Gate::Unitary(_) | Gate::U { .. },
                        ..
                    })
                );
            if needs_synth {
                let (syn, phase) = synthesize_1q(&mat, tol);
                match syn {
                    None => {
                        *global_phase += phase;
                        out[slot] = None;
                        *changed = true;
                    }
                    Some(gate) => {
                        let existing_is_u = matches!(
                            out[slot].as_ref().map(|n| &n.op),
                            Some(Operation::Gate {
                                gate: Gate::U { .. },
                                ..
                            })
                        );
                        // Avoid floating-point oscillation replacing a single U gate with another U gate.
                        if count == 1 && existing_is_u && matches!(gate, Gate::U { .. }) {
                            return;
                        }
                        let new_op = Operation::Gate { gate, target: q };
                        if count > 1 || out[slot].as_ref().map(|n| &n.op) != Some(&new_op) {
                            *global_phase += phase;
                            out[slot] = Some(Node {
                                op: new_op,
                                binding: None,
                            });
                            *changed = true;
                        }
                    }
                }
            }
        }
    };

    for node in std::mem::take(&mut seg.nodes) {
        if node.binding.is_none()
            && let Some((q, m)) = op_matrix2(&node.op)
        {
            if let Some((_, acc, count)) = &mut run[q] {
                *acc = mat2_mul(&m, acc);
                *count += 1;
                changed = true;
            } else {
                let slot = out.len();
                out.push(Some(node));
                run[q] = Some((slot, m, 1));
            }
            continue;
        }

        // Flush any qubit touched by this non-1q / parametric op.
        for q in node.op.qubits() {
            flush_qubit(q, &mut run, &mut out, &mut seg.global_phase, &mut changed);
        }
        out.push(Some(node));
    }

    for q in 0..n {
        flush_qubit(q, &mut run, &mut out, &mut seg.global_phase, &mut changed);
    }

    seg.nodes = out.into_iter().flatten().collect();
    seg.global_phase = wrap_pi(seg.global_phase);
    changed
}

/// Pass: commutation-aware cancellation and rotation merging.
///
/// For each operation `i`, scans forward across commuting operations on its
/// qubits to find a matching inverse or mergeable rotation `j`. Moving `i`
/// forward to `j` is provably valid whenever `i` commutes with every live
/// operation between `i` and `j` that shares a qubit with `i` (since ops on
/// disjoint qubits trivially commute!).
pub(crate) fn commutative_cancellation(seg: &mut Segment, tol: f64) -> bool {
    let mut changed = false;
    let len = seg.nodes.len();
    if len < 2 {
        return false;
    }

    // Build per-qubit occurrence lists so we only inspect ops sharing a qubit.
    let mut nodes: Vec<Option<Node>> = seg.nodes.drain(..).map(Some).collect();
    let mut wire: Vec<Vec<usize>> = vec![Vec::new(); seg.n_qubits];
    for (idx, maybe) in nodes.iter().enumerate() {
        if let Some(n) = maybe {
            for q in n.op.qubits() {
                wire[q].push(idx);
            }
        }
    }

    // Track cursor position in each wire to avoid rescanning past `i`.
    let mut cursor = vec![0usize; seg.n_qubits];

    for i in 0..len {
        let Some(node_i) = nodes[i].clone() else {
            continue;
        };
        let qs_i = node_i.op.qubits();
        let i_param = node_i.binding.is_some();
        if qs_i.is_empty() {
            continue;
        }

        // Advance each qubit's cursor to `i`.
        for &q in &qs_i {
            while cursor[q] < wire[q].len() && wire[q][cursor[q]] <= i {
                cursor[q] += 1;
            }
        }

        // Walk forward in increasing index order over the union of wires in `qs_i`.
        let mut ptrs: Vec<usize> = qs_i.iter().map(|&q| cursor[q]).collect();
        let mut last_checked = i;

        loop {
            // Find the smallest next index across `qs_i`.
            let mut next_j = usize::MAX;
            for (k, &q) in qs_i.iter().enumerate() {
                while ptrs[k] < wire[q].len() && nodes[wire[q][ptrs[k]]].is_none() {
                    ptrs[k] += 1;
                }
                if ptrs[k] < wire[q].len() {
                    next_j = next_j.min(wire[q][ptrs[k]]);
                }
            }
            if next_j == usize::MAX {
                break;
            }
            // Advance all pointers sitting at `next_j`.
            for (k, &q) in qs_i.iter().enumerate() {
                if ptrs[k] < wire[q].len() && wire[q][ptrs[k]] == next_j {
                    ptrs[k] += 1;
                }
            }
            if next_j == last_checked {
                continue;
            }
            last_checked = next_j;

            let Some(node_j) = &nodes[next_j] else {
                continue;
            };
            let j_param = node_j.binding.is_some();

            // First check if `node_i` and `node_j` can combine (cancel or merge)!
            // Because `node_i` commuted with every live op in `(i, next_j)` sharing
            // a qubit with `node_i`, moving `node_i` directly in front of `node_j`
            // is 100% sound!
            match try_combine(&node_i, node_j, tol) {
                CombineResult::Cancel(phase) => {
                    seg.global_phase += phase;
                    nodes[i] = None;
                    nodes[next_j] = None;
                    changed = true;
                    break;
                }
                CombineResult::Merge(merged, phase) => {
                    // Only commit a merge if qubit support didn't grow unexpectedly
                    // (it's placed at `next_j`, which already holds `node_j`).
                    let merged_qs = merged.op.qubits();
                    let j_qs = node_j.op.qubits();
                    debug_assert!(merged_qs.iter().all(|q| j_qs.contains(q)));
                    seg.global_phase += phase;
                    nodes[i] = None;
                    nodes[next_j] = Some(merged);
                    changed = true;
                    break;
                }
                CombineResult::None => {}
            }

            // Can `node_i` commute past `node_j`?
            if !ops_commute(&node_i.op, &node_j.op, i_param, j_param, tol) {
                break;
            }
        }
    }

    seg.nodes = nodes.into_iter().flatten().collect();
    seg.global_phase = wrap_pi(seg.global_phase);
    changed
}

/// Pass: DAG-based two-qubit peephole consolidation & exact KAK synthesis.
///
/// Collects maximal convex blocks on pairs of qubits `{a, b}` (skipping over
/// gates on disjoint qubits via the DAG), computes each block's exact 4×4
/// unitary, synthesizes the optimal circuit via KAK, and replaces the block
/// whenever the synthesized form strictly improves `(n_2q, total_ops)`.
pub(crate) fn two_qubit_peephole_kak(
    seg: &mut Segment,
    basis: Basis,
    keep_real: bool,
    tol: f64,
) -> bool {
    let n_ops = seg.nodes.len();
    if n_ops == 0 {
        return false;
    }

    // Build per-qubit doubly-linked DAG adjacency.
    let mut prev: Vec<Vec<(usize, Option<usize>)>> = Vec::with_capacity(n_ops);
    let mut next: Vec<Vec<(usize, Option<usize>)>> = Vec::with_capacity(n_ops);
    let mut last_on: Vec<Option<usize>> = vec![None; seg.n_qubits];

    for (i, node) in seg.nodes.iter().enumerate() {
        let qs = node.op.qubits();
        let mut p = Vec::with_capacity(qs.len());
        let mut nx = Vec::with_capacity(qs.len());
        for &q in &qs {
            p.push((q, last_on[q]));
            nx.push((q, None));
            if let Some(prev_idx) = last_on[q] {
                for (pq, link) in &mut next[prev_idx] {
                    if *pq == q {
                        *link = Some(i);
                        break;
                    }
                }
            }
            last_on[q] = Some(i);
        }
        prev.push(p);
        next.push(nx);
    }

    let get_prev = |idx: usize, q: usize| -> Option<usize> {
        prev[idx]
            .iter()
            .find(|&&(qq, _)| qq == q)
            .and_then(|&(_, p)| p)
    };
    let get_next = |idx: usize, q: usize| -> Option<usize> {
        next[idx]
            .iter()
            .find(|&&(qq, _)| qq == q)
            .and_then(|&(_, n)| n)
    };

    let is_eligible_1q = |idx: usize, q: usize| -> bool {
        let node = &seg.nodes[idx];
        node.binding.is_none()
            && !node.op.is_multi_controlled()
            && node.op.qubits().as_slice() == [q]
    };
    let is_eligible_2q_on = |idx: usize, a: usize, b: usize| -> bool {
        let node = &seg.nodes[idx];
        if node.binding.is_some() || node.op.is_multi_controlled() {
            return false;
        }
        let qs = node.op.qubits();
        qs.len() == 2 && ((qs[0] == a && qs[1] == b) || (qs[0] == b && qs[1] == a))
    };

    // Map each original node to its super-node group id (initially none).
    let mut group_of: Vec<Option<usize>> = vec![None; n_ops];
    let mut groups: Vec<Vec<Node>> = Vec::new();
    let mut group_min_idx: Vec<usize> = Vec::new();
    let mut any_changed = false;

    for seed in 0..n_ops {
        if group_of[seed].is_some() {
            continue;
        }
        let node = &seg.nodes[seed];
        if node.binding.is_some() || node.op.is_multi_controlled() {
            continue;
        }
        let qs = node.op.qubits();
        if qs.len() != 2 {
            continue;
        }
        let (a, b) = if qs[0] < qs[1] {
            (qs[0], qs[1])
        } else {
            (qs[1], qs[0])
        };

        // Grow maximal convex block on `{a, b}` around `seed`.
        let mut members = vec![seed];
        let mut in_block = std::collections::HashSet::new();
        in_block.insert(seed);

        let (mut first_a, mut last_a) = (seed, seed);
        let (mut first_b, mut last_b) = (seed, seed);

        // Grow forward.
        loop {
            while let Some(nx) = get_next(last_a, a)
                && group_of[nx].is_none()
                && !in_block.contains(&nx)
                && is_eligible_1q(nx, a)
            {
                in_block.insert(nx);
                members.push(nx);
                last_a = nx;
            }
            while let Some(nx) = get_next(last_b, b)
                && group_of[nx].is_none()
                && !in_block.contains(&nx)
                && is_eligible_1q(nx, b)
            {
                in_block.insert(nx);
                members.push(nx);
                last_b = nx;
            }
            if let (Some(na), Some(nb)) = (get_next(last_a, a), get_next(last_b, b))
                && na == nb
                && group_of[na].is_none()
                && !in_block.contains(&na)
                && is_eligible_2q_on(na, a, b)
            {
                in_block.insert(na);
                members.push(na);
                last_a = na;
                last_b = na;
            } else {
                break;
            }
        }

        // Grow backward.
        loop {
            while let Some(pr) = get_prev(first_a, a)
                && group_of[pr].is_none()
                && !in_block.contains(&pr)
                && is_eligible_1q(pr, a)
            {
                in_block.insert(pr);
                members.push(pr);
                first_a = pr;
            }
            while let Some(pr) = get_prev(first_b, b)
                && group_of[pr].is_none()
                && !in_block.contains(&pr)
                && is_eligible_1q(pr, b)
            {
                in_block.insert(pr);
                members.push(pr);
                first_b = pr;
            }
            if let (Some(pa), Some(pb)) = (get_prev(first_a, a), get_prev(first_b, b))
                && pa == pb
                && group_of[pa].is_none()
                && !in_block.contains(&pa)
                && is_eligible_2q_on(pa, a, b)
            {
                in_block.insert(pa);
                members.push(pa);
                first_a = pa;
                first_b = pa;
            } else {
                break;
            }
        }

        members.sort_unstable();

        // Count original 2q cost.
        let orig_has_unitary2 = members
            .iter()
            .any(|&idx| matches!(seg.nodes[idx].op, Operation::Unitary2 { .. }));
        let orig_2q = members
            .iter()
            .filter(|&&idx| seg.nodes[idx].op.qubits().len() == 2)
            .count();
        let orig_total = members.len();

        // A single in-basis 2q gate surrounded by 1q gates cannot decrease its
        // 2q gate count via local KAK (local 1q gates preserve Weyl coordinates).
        let needs_cx_rebasis = basis == Basis::Cx
            && members.iter().any(|&idx| {
                let op = &seg.nodes[idx].op;
                op.qubits().len() == 2 && !matches!(op, Operation::Controlled { gate: Gate::X, .. })
            });
        if orig_2q <= 1 && !orig_has_unitary2 && !needs_cx_rebasis {
            continue;
        }

        // Multiply the exact 4×4 unitary of the block.
        let pair = [a, b];
        let mut u = kron2(&I2, &I2);
        for &idx in &members {
            let m = op_matrix4_on(&seg.nodes[idx].op, pair);
            let mut next_u = [[C64::new(0.0, 0.0); 4]; 4];
            for r in 0..4 {
                for c in 0..4 {
                    for k in 0..4 {
                        next_u[r][c] += m[r][k] * u[k][c];
                    }
                }
            }
            u = next_u;
        }

        let Some((synth_ops, synth_phase)) = synthesize_2q(&u, pair, basis, keep_real, tol) else {
            continue;
        };

        let synth_2q = synth_ops.iter().filter(|op| op.qubits().len() == 2).count();
        let synth_total = synth_ops.len();

        let better = orig_has_unitary2
            || (needs_cx_rebasis && synth_2q <= 3)
            || (synth_2q < orig_2q)
            || (synth_2q == orig_2q && synth_total < orig_total);

        if better {
            let gid = groups.len();
            group_min_idx.push(members[0]);
            for &idx in &members {
                group_of[idx] = Some(gid);
            }
            groups.push(
                synth_ops
                    .into_iter()
                    .map(|op| Node { op, binding: None })
                    .collect(),
            );
            seg.global_phase += synth_phase;
            any_changed = true;
        }
    }

    if !any_changed {
        return false;
    }

    // Assign singleton groups to every untouched node.
    for (idx, g) in group_of.iter_mut().enumerate() {
        if g.is_none() {
            let gid = groups.len();
            *g = Some(gid);
            group_min_idx.push(idx);
            groups.push(vec![seg.nodes[idx].clone()]);
        }
    }

    // Build contracted DAG edges and run Kahn's topological sort (ordered by
    // earliest original node index so untouched structure is preserved).
    let n_groups = groups.len();
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n_groups];
    let mut indeg = vec![0usize; n_groups];

    for (u_idx, links) in next.iter().enumerate() {
        let gu = group_of[u_idx].unwrap();
        for &(_, maybe_v) in links {
            if let Some(v_idx) = maybe_v {
                let gv = group_of[v_idx].unwrap();
                if gu != gv && !adj[gu].contains(&gv) {
                    adj[gu].push(gv);
                    indeg[gv] += 1;
                }
            }
        }
    }

    let mut heap: BinaryHeap<Reverse<(usize, usize)>> = BinaryHeap::new();
    for g in 0..n_groups {
        if indeg[g] == 0 {
            heap.push(Reverse((group_min_idx[g], g)));
        }
    }

    let mut out = Vec::with_capacity(n_ops);
    let mut visited = 0;
    while let Some(Reverse((_, g))) = heap.pop() {
        visited += 1;
        out.extend(std::mem::take(&mut groups[g]));
        for &v in &adj[g] {
            indeg[v] -= 1;
            if indeg[v] == 0 {
                heap.push(Reverse((group_min_idx[v], v)));
            }
        }
    }
    debug_assert_eq!(visited, n_groups, "contracted DAG must remain acyclic");

    seg.nodes = out;
    seg.global_phase = wrap_pi(seg.global_phase);
    true
}

/// Pass: Clifford, phase-gadget, swap, and multi-qubit template rewrites.
pub(crate) fn clifford_and_template_rewrites(
    seg: &mut Segment,
    level: OptimizationLevel,
    basis: Basis,
    keep_real: bool,
    tol: f64,
) -> bool {
    let mut changed = false;

    // 1. Push 1q gates past SWAPs when it exposes adjacent SWAP cancellation,
    //    and simplify contiguous SWAP networks into minimal transpositions.
    if simplify_swap_networks(seg) {
        changed = true;
    }

    // 2. Local Clifford / phase-gadget templates on per-qubit windows.
    if apply_local_templates(seg, basis, keep_real, tol) {
        changed = true;
    }

    // 3. Level 3 multi-qubit optimizations:
    if level == OptimizationLevel::Level3 {
        // 3a. Clifford conjugation of Pauli rotations (C P C† -> P', saves 2 gates!).
        if conjugate_pauli_rotations(seg, keep_real) {
            changed = true;
        }
        // 3b. Clifford Pauli pushing (X/Z through CX/CZ to cancel on the other side).
        if push_paulis_through_cliffords(seg) {
            changed = true;
        }
        // 3c. 3-qubit CNOT bridge/triangle reductions & GF(2) CNOT subcircuit optimization!
        if optimize_cnot_subcircuits(seg) {
            changed = true;
        }
    }

    changed
}

/// Simplify contiguous runs of SWAP gates into the minimal number of
/// transpositions (`n - cycles`), and push 1-qubit gates across a SWAP when
/// another SWAP on the same pair immediately follows (`SWAP G(a) SWAP -> G(b)`).
fn simplify_swap_networks(seg: &mut Segment) -> bool {
    let mut changed = false;

    // Pattern: SWAP(a, b) followed (on wires a, b) only by 1q gates on a/b, then SWAP(a, b).
    let n_ops = seg.nodes.len();
    if n_ops >= 2 {
        let mut nodes: Vec<Option<Node>> = seg.nodes.drain(..).map(Some).collect();
        for i in 0..n_ops {
            let Some(&Operation::Swap(a, b)) = nodes[i].as_ref().map(|n| &n.op) else {
                continue;
            };
            // Walk forward on wires {a, b}: collect 1q gates on a or b until hitting
            // either a matching Swap(a, b) or a blocker.
            let mut middle_1q = Vec::new();
            let mut matched = None;
            for (j, maybe) in nodes.iter().enumerate().skip(i + 1) {
                let Some(nj) = maybe else { continue };
                let qs = nj.op.qubits();
                if !qs.contains(&a) && !qs.contains(&b) {
                    continue;
                }
                if matches!(nj.op, Operation::Swap(x, y) if (x == a && y == b) || (x == b && y == a))
                {
                    matched = Some(j);
                    break;
                }
                if qs.len() == 1 && (qs[0] == a || qs[0] == b) {
                    middle_1q.push(j);
                } else {
                    break;
                }
            }
            if let Some(j) = matched {
                // Cancel both SWAPs and swap a <-> b on all intervening 1q ops!
                nodes[i] = None;
                nodes[j] = None;
                let mut map: Vec<usize> = (0..seg.n_qubits).collect();
                map.swap(a, b);
                for m in middle_1q {
                    if let Some(nm) = &mut nodes[m] {
                        nm.op = nm.op.remapped(&map);
                    }
                }
                changed = true;
            }
        }
        seg.nodes = nodes.into_iter().flatten().collect();
    }

    // Collapse any contiguous run of >= 2 pure SWAP gates into minimal transpositions.
    let mut out = Vec::with_capacity(seg.nodes.len());
    let mut i = 0;
    while i < seg.nodes.len() {
        if !matches!(seg.nodes[i].op, Operation::Swap(..)) {
            out.push(seg.nodes[i].clone());
            i += 1;
            continue;
        }
        let start = i;
        while i < seg.nodes.len() && matches!(seg.nodes[i].op, Operation::Swap(..)) {
            i += 1;
        }
        let run_len = i - start;
        if run_len < 2 {
            out.push(seg.nodes[start].clone());
            continue;
        }
        // Compute net permutation of the swap run.
        let mut perm: Vec<usize> = (0..seg.n_qubits).collect();
        for node in &seg.nodes[start..i] {
            if let Operation::Swap(a, b) = node.op {
                perm.swap(a, b);
            }
        }
        // Decompose `perm` into minimal swaps (`n - cycles`).
        let mut cur: Vec<usize> = (0..seg.n_qubits).collect();
        let mut pos: Vec<usize> = (0..seg.n_qubits).collect();
        let mut minimal = Vec::new();
        for q in 0..seg.n_qubits {
            let want = perm[q];
            if cur[q] != want {
                let p = pos[want];
                let (lo, hi) = if q < p { (q, p) } else { (p, q) };
                minimal.push(Node {
                    op: Operation::Swap(lo, hi),
                    binding: None,
                });
                let old_q = cur[q];
                cur.swap(q, p);
                pos[want] = q;
                pos[old_q] = p;
            }
        }
        if minimal.len() < run_len {
            out.extend(minimal);
            changed = true;
        } else {
            out.extend_from_slice(&seg.nodes[start..i]);
        }
    }
    seg.nodes = out;
    changed
}

/// Apply local Clifford & phase-gadget rewrites on the DAG:
/// - `H(t) CX(c, t) H(t)` -> `CZ(c, t)` (in `Basis::Native`)
/// - `H(c) H(t) CX(c, t) H(c) H(t)` -> `CX(t, c)`
/// - `CX(c, t) Rz(θ, t) CX(c, t)` -> `Rzz(θ, c, t)` (in `Basis::Native`, fixed OR parametric!)
/// - `CX(c, t) Rx(θ, c) CX(c, t)` -> `Rxx(θ, c, t)` (in `Basis::Native`, fixed OR parametric!)
/// - `CX(c, t) Rz(π, t) CX(c, t)` -> `Z(c) Z(t)` + phase `-π/2` (even in `Basis::Cx`!)
fn apply_local_templates(seg: &mut Segment, basis: Basis, keep_real: bool, tol: f64) -> bool {
    let n_ops = seg.nodes.len();
    if n_ops < 3 {
        return false;
    }

    let mut nodes: Vec<Option<Node>> = seg.nodes.drain(..).map(Some).collect();
    let mut changed = false;

    // Helper: find immediate previous / next live node touching `q` from `idx`.
    let prev_on = |nodes: &[Option<Node>], idx: usize, q: usize| -> Option<usize> {
        (0..idx).rev().find(|&k| {
            nodes[k]
                .as_ref()
                .is_some_and(|n| n.op.qubits().contains(&q))
        })
    };
    let next_on = |nodes: &[Option<Node>], idx: usize, q: usize| -> Option<usize> {
        (idx + 1..nodes.len()).find(|&k| {
            nodes[k]
                .as_ref()
                .is_some_and(|n| n.op.qubits().contains(&q))
        })
    };
    let is_h = |nodes: &[Option<Node>], idx: usize, q: usize| -> bool {
        matches!(
            nodes[idx].as_ref(),
            Some(Node {
                op: Operation::Gate {
                    gate: Gate::H,
                    target
                },
                binding: None,
            }) if *target == q
        )
    };

    for i in 0..n_ops {
        let Some(Node {
            op:
                Operation::Controlled {
                    gate: Gate::X,
                    control: c,
                    target: t,
                },
            binding: None,
        }) = nodes[i]
        else {
            continue;
        };

        // Template 1: H(c) H(t) CX(c, t) H(c) H(t) -> CX(t, c).
        if let (Some(pc), Some(pt), Some(nc), Some(nt)) = (
            prev_on(&nodes, i, c),
            prev_on(&nodes, i, t),
            next_on(&nodes, i, c),
            next_on(&nodes, i, t),
        ) && is_h(&nodes, pc, c)
            && is_h(&nodes, pt, t)
            && is_h(&nodes, nc, c)
            && is_h(&nodes, nt, t)
        {
            nodes[pc] = None;
            nodes[pt] = None;
            nodes[nc] = None;
            nodes[nt] = None;
            nodes[i] = Some(Node {
                op: Operation::Controlled {
                    gate: Gate::X,
                    control: t,
                    target: c,
                },
                binding: None,
            });
            changed = true;
            continue;
        }

        // Template 2: H(t) CX(c, t) H(t) -> CZ(c, t) (Native basis).
        if basis == Basis::Native
            && let (Some(pt), Some(nt)) = (prev_on(&nodes, i, t), next_on(&nodes, i, t))
            && is_h(&nodes, pt, t)
            && is_h(&nodes, nt, t)
        {
            nodes[pt] = None;
            nodes[nt] = None;
            let (lo, hi) = if c < t { (c, t) } else { (t, c) };
            nodes[i] = Some(Node {
                op: Operation::Controlled {
                    gate: Gate::Z,
                    control: lo,
                    target: hi,
                },
                binding: None,
            });
            changed = true;
            continue;
        }

        // Template 3: CX(c, t) · Rz(θ, t) · CX(c, t) -> Rzz(θ, c, t) (in Native, !keep_real).
        // Also works for fixed Phase(λ, t) (adds λ/2 to global phase) and parametric Rz!
        if !keep_real
            && let (Some(nc), Some(nt)) = (next_on(&nodes, i, c), next_on(&nodes, i, t))
            && let Some(mid) = nodes[nt].clone()
            && let Some(nt2) = next_on(&nodes, nt, t)
            && nc == nt2
            && matches!(
                nodes[nc],
                Some(Node {
                    op: Operation::Controlled {
                        gate: Gate::X,
                        control: c2,
                        target: t2,
                    },
                    binding: None,
                }) if c2 == c && t2 == t
            )
        {
            // Check if `mid` is Rz / Phase / Z / S / Sdg / T / Tdg on `t`.
            let rz_info = match (&mid.op, mid.binding) {
                (
                    Operation::Gate {
                        gate: Gate::Rz(v),
                        target,
                    },
                    b,
                ) if *target == t => Some((*v, 0.0, b)),
                (
                    Operation::Gate {
                        gate: Gate::Phase(v),
                        target,
                    },
                    None,
                ) if *target == t => Some((*v, *v / 2.0, None)),
                (Operation::Gate { gate, target }, None) if *target == t => {
                    crate::optimize::one_qubit::as_rz_and_phase(gate, tol)
                        .map(|(ang, ph)| (ang, ph, None))
                }
                _ => None,
            };
            if let Some((theta, extra_phase, binding)) = rz_info {
                // Even in Basis::Cx, if theta == ±π (fixed), replace with Z(c) Z(t)!
                if binding.is_none() && (wrap_pi(theta).abs() - PI).abs() <= tol {
                    let sign = if wrap_2pi(theta) > 0.0 { -1.0 } else { 1.0 };
                    seg.global_phase += extra_phase + sign * FRAC_PI_2;
                    nodes[i] = Some(Node {
                        op: Operation::Gate {
                            gate: Gate::Z,
                            target: c,
                        },
                        binding: None,
                    });
                    nodes[nt] = Some(Node {
                        op: Operation::Gate {
                            gate: Gate::Z,
                            target: t,
                        },
                        binding: None,
                    });
                    nodes[nc] = None;
                    changed = true;
                    continue;
                }
                if basis == Basis::Native {
                    seg.global_phase += extra_phase;
                    let pauli =
                        PauliString::new([(c, Pauli::Z), (t, Pauli::Z)]).expect("distinct qubits");
                    nodes[i] = None;
                    nodes[nt] = Some(Node {
                        op: Operation::PauliRotation { pauli, theta },
                        binding,
                    });
                    nodes[nc] = None;
                    changed = true;
                    continue;
                }
            }
        }

        // Template 4: CX(c, t) · Rx(θ, c) · CX(c, t) -> Rxx(θ, c, t) (in Native, !keep_real).
        if basis == Basis::Native
            && !keep_real
            && let (Some(nc), Some(nt)) = (next_on(&nodes, i, c), next_on(&nodes, i, t))
            && let Some(mid) = nodes[nc].clone()
            && let Some(nc2) = next_on(&nodes, nc, c)
            && nt == nc2
            && matches!(
                nodes[nt],
                Some(Node {
                    op: Operation::Controlled {
                        gate: Gate::X,
                        control: c2,
                        target: t2,
                    },
                    binding: None,
                }) if c2 == c && t2 == t
            )
            && let Operation::Gate {
                gate: Gate::Rx(theta),
                target,
            } = mid.op
            && target == c
        {
            let pauli = PauliString::new([(c, Pauli::X), (t, Pauli::X)]).expect("distinct qubits");
            nodes[i] = None;
            nodes[nc] = Some(Node {
                op: Operation::PauliRotation { pauli, theta },
                binding: mid.binding,
            });
            nodes[nt] = None;
            changed = true;
            continue;
        }
    }

    seg.nodes = nodes.into_iter().flatten().collect();
    seg.global_phase = wrap_pi(seg.global_phase);
    changed
}

/// Level 3 pass: conjugate a `PauliRotation` by matching single-qubit Clifford
/// gates `C(q)` on both sides: `C(q) · exp(-i θ/2 P) · C†(q) = exp(-i (±θ)/2 P')`!
/// Saves 2 gates per match (works for both fixed and parametric rotations).
fn conjugate_pauli_rotations(seg: &mut Segment, keep_real: bool) -> bool {
    let n_ops = seg.nodes.len();
    if n_ops < 3 {
        return false;
    }
    let mut nodes: Vec<Option<Node>> = seg.nodes.drain(..).map(Some).collect();
    let mut changed = false;

    for i in 0..n_ops {
        let Some(Node {
            op: Operation::PauliRotation { pauli, theta },
            binding,
        }) = nodes[i].clone()
        else {
            continue;
        };
        if pauli.weight() < 2 {
            continue;
        }

        let mut ops_vec: Vec<(usize, Pauli)> = pauli.ops().to_vec();
        let mut cur_theta = theta;
        let mut cur_binding = binding;
        let mut any_here = false;

        for entry in &mut ops_vec {
            let (q, p) = *entry;
            let prev = (0..i).rev().find(|&k| {
                nodes[k]
                    .as_ref()
                    .is_some_and(|n| n.op.qubits().contains(&q))
            });
            let next = (i + 1..n_ops).find(|&k| {
                nodes[k]
                    .as_ref()
                    .is_some_and(|n| n.op.qubits().contains(&q))
            });
            let (Some(prev_idx), Some(next_idx)) = (prev, next) else {
                continue;
            };
            let (Some(pn), Some(nn)) = (&nodes[prev_idx], &nodes[next_idx]) else {
                continue;
            };
            if pn.binding.is_some() || nn.binding.is_some() {
                continue;
            }
            let (
                Operation::Gate {
                    gate: gp,
                    target: tp,
                },
                Operation::Gate {
                    gate: gn,
                    target: tn,
                },
            ) = (&pn.op, &nn.op)
            else {
                continue;
            };
            if *tp != q || *tn != q {
                continue;
            }

            // Matrix order: gn · exp(-i θ/2 P) · gp = exp(-i θ/2 (gn P gp))
            // when gn == gp†!
            let Some((new_p, sign)) = clifford_conjugate_pauli(*gn, *gp, p) else {
                continue;
            };
            if keep_real {
                // Only accept if parity of Y count stays odd (keeps rotation real).
                let old_y = usize::from(p == Pauli::Y);
                let new_y = usize::from(new_p == Pauli::Y);
                if old_y != new_y {
                    continue;
                }
            }
            entry.1 = new_p;
            if sign < 0.0 {
                cur_theta = -cur_theta;
                if let Some(b) = &mut cur_binding {
                    b.scale = -b.scale;
                    b.offset = -b.offset;
                }
            }
            nodes[prev_idx] = None;
            nodes[next_idx] = None;
            any_here = true;
            changed = true;
        }

        if any_here {
            let new_pauli = PauliString::new(ops_vec).expect("same distinct qubits");
            nodes[i] = Some(Node {
                op: Operation::PauliRotation {
                    pauli: new_pauli,
                    theta: cur_theta,
                },
                binding: cur_binding,
            });
        }
    }

    seg.nodes = nodes.into_iter().flatten().collect();
    changed
}

/// For `after · P · before` where `after == before†` is a single-qubit Clifford,
/// return `(P', sign)` such that `after · P · before = sign · P'`.
fn clifford_conjugate_pauli(after: Gate, before: Gate, p: Pauli) -> Option<(Pauli, f64)> {
    match (after, before) {
        (Gate::H, Gate::H) => Some(match p {
            Pauli::X => (Pauli::Z, 1.0),
            Pauli::Y => (Pauli::Y, -1.0),
            Pauli::Z => (Pauli::X, 1.0),
            Pauli::I => (Pauli::I, 1.0),
        }),
        // S · P · S†: S X S† = Y, S Y S† = -X, S Z S† = Z.
        (Gate::S, Gate::Sdg) => Some(match p {
            Pauli::X => (Pauli::Y, 1.0),
            Pauli::Y => (Pauli::X, -1.0),
            Pauli::Z => (Pauli::Z, 1.0),
            Pauli::I => (Pauli::I, 1.0),
        }),
        // S† · P · S: S† X S = -Y, S† Y S = X, S† Z S = Z.
        (Gate::Sdg, Gate::S) => Some(match p {
            Pauli::X => (Pauli::Y, -1.0),
            Pauli::Y => (Pauli::X, 1.0),
            Pauli::Z => (Pauli::Z, 1.0),
            Pauli::I => (Pauli::I, 1.0),
        }),
        (Gate::X, Gate::X) => Some(match p {
            Pauli::X => (Pauli::X, 1.0),
            Pauli::Y => (Pauli::Y, -1.0),
            Pauli::Z => (Pauli::Z, -1.0),
            Pauli::I => (Pauli::I, 1.0),
        }),
        (Gate::Y, Gate::Y) => Some(match p {
            Pauli::X => (Pauli::X, -1.0),
            Pauli::Y => (Pauli::Y, 1.0),
            Pauli::Z => (Pauli::Z, -1.0),
            Pauli::I => (Pauli::I, 1.0),
        }),
        (Gate::Z, Gate::Z) => Some(match p {
            Pauli::X => (Pauli::X, -1.0),
            Pauli::Y => (Pauli::Y, -1.0),
            Pauli::Z => (Pauli::Z, 1.0),
            Pauli::I => (Pauli::I, 1.0),
        }),
        _ => None,
    }
}

/// Level 3 pass: push single-qubit Paulis across `CX` / `CZ` when doing so
/// cancels a matching Pauli on the other side:
/// - `X(c) CX(c, t) X(c)` -> `CX(c, t) X(t)`
/// - `Z(t) CX(c, t) Z(t)` -> `CX(c, t) Z(c)`
/// - `X(a) CZ(a, b) X(a)` -> `CZ(a, b) Z(b)`
fn push_paulis_through_cliffords(seg: &mut Segment) -> bool {
    let n_ops = seg.nodes.len();
    if n_ops < 3 {
        return false;
    }
    let mut nodes: Vec<Option<Node>> = seg.nodes.drain(..).map(Some).collect();
    let mut extras: Vec<(usize, Node)> = Vec::new();
    let mut changed = false;

    let prev_on = |nodes: &[Option<Node>], idx: usize, q: usize| -> Option<usize> {
        (0..idx).rev().find(|&k| {
            nodes[k]
                .as_ref()
                .is_some_and(|n| n.op.qubits().contains(&q))
        })
    };
    let next_on = |nodes: &[Option<Node>], idx: usize, q: usize| -> Option<usize> {
        (idx + 1..nodes.len()).find(|&k| {
            nodes[k]
                .as_ref()
                .is_some_and(|n| n.op.qubits().contains(&q))
        })
    };
    let is_gate = |nodes: &[Option<Node>], idx: usize, g: Gate, q: usize| -> bool {
        matches!(
            nodes[idx].as_ref(),
            Some(Node {
                op: Operation::Gate { gate, target },
                binding: None,
            }) if *gate == g && *target == q
        )
    };

    for i in 0..n_ops {
        let Some(Node { op, binding: None }) = nodes[i].clone() else {
            continue;
        };
        match op {
            Operation::Controlled {
                gate: Gate::X,
                control: c,
                target: t,
            } => {
                // X(c) CX(c, t) X(c) -> CX(c, t) X(t)
                if let (Some(pc), Some(nc)) = (prev_on(&nodes, i, c), next_on(&nodes, i, c))
                    && is_gate(&nodes, pc, Gate::X, c)
                    && is_gate(&nodes, nc, Gate::X, c)
                {
                    nodes[pc] = None;
                    nodes[nc] = None;
                    extras.push((
                        i,
                        Node {
                            op: Operation::Gate {
                                gate: Gate::X,
                                target: t,
                            },
                            binding: None,
                        },
                    ));
                    changed = true;
                }
                // Z(t) CX(c, t) Z(t) -> CX(c, t) Z(c)
                if let (Some(pt), Some(nt)) = (prev_on(&nodes, i, t), next_on(&nodes, i, t))
                    && is_gate(&nodes, pt, Gate::Z, t)
                    && is_gate(&nodes, nt, Gate::Z, t)
                {
                    nodes[pt] = None;
                    nodes[nt] = None;
                    extras.push((
                        i,
                        Node {
                            op: Operation::Gate {
                                gate: Gate::Z,
                                target: c,
                            },
                            binding: None,
                        },
                    ));
                    changed = true;
                }
            }
            Operation::Controlled {
                gate: Gate::Z,
                control: a,
                target: b,
            } => {
                // X(a) CZ(a, b) X(a) -> CZ(a, b) Z(b)
                if let (Some(pa), Some(na)) = (prev_on(&nodes, i, a), next_on(&nodes, i, a))
                    && is_gate(&nodes, pa, Gate::X, a)
                    && is_gate(&nodes, na, Gate::X, a)
                {
                    nodes[pa] = None;
                    nodes[na] = None;
                    extras.push((
                        i,
                        Node {
                            op: Operation::Gate {
                                gate: Gate::Z,
                                target: b,
                            },
                            binding: None,
                        },
                    ));
                    changed = true;
                }
                // X(b) CZ(a, b) X(b) -> CZ(a, b) Z(a)
                if let (Some(pb), Some(nb)) = (prev_on(&nodes, i, b), next_on(&nodes, i, b))
                    && is_gate(&nodes, pb, Gate::X, b)
                    && is_gate(&nodes, nb, Gate::X, b)
                {
                    nodes[pb] = None;
                    nodes[nb] = None;
                    extras.push((
                        i,
                        Node {
                            op: Operation::Gate {
                                gate: Gate::Z,
                                target: a,
                            },
                            binding: None,
                        },
                    ));
                    changed = true;
                }
            }
            _ => {}
        }
    }

    if !changed {
        seg.nodes = nodes.into_iter().flatten().collect();
        return false;
    }

    let mut by_after: HashMap<usize, Vec<Node>> = HashMap::new();
    for (idx, n) in extras {
        by_after.entry(idx).or_default().push(n);
    }
    let mut out = Vec::with_capacity(n_ops);
    for (i, maybe) in nodes.into_iter().enumerate() {
        if let Some(n) = maybe {
            out.push(n);
        }
        if let Some(more) = by_after.remove(&i) {
            out.extend(more);
        }
    }
    seg.nodes = out;
    true
}

/// Level 3 pass: optimize contiguous pure-CNOT subcircuits using GF(2) linear
/// algebra! Captures 3-qubit bridges (`CX(a,b) CX(b,c) CX(a,b) -> CX(b,c) CX(a,c)`),
/// parity cancellations, and multi-qubit CNOT network reductions that 2-qubit
/// peephole optimization cannot see.
fn optimize_cnot_subcircuits(seg: &mut Segment) -> bool {
    let n_ops = seg.nodes.len();
    if n_ops < 3 {
        return false;
    }

    let is_cx = |n: &Node| {
        matches!(
            n,
            Node {
                op: Operation::Controlled { gate: Gate::X, .. },
                binding: None,
            }
        )
    };

    // Build DAG adjacency so we can collect maximal convex pure-CX blocks!
    let mut next: Vec<Vec<(usize, Option<usize>)>> = Vec::with_capacity(n_ops);
    let mut prev: Vec<Vec<(usize, Option<usize>)>> = Vec::with_capacity(n_ops);
    let mut last_on: Vec<Option<usize>> = vec![None; seg.n_qubits];
    for (i, node) in seg.nodes.iter().enumerate() {
        let qs = node.op.qubits();
        let mut p = Vec::with_capacity(qs.len());
        let mut nx = Vec::with_capacity(qs.len());
        for &q in &qs {
            p.push((q, last_on[q]));
            nx.push((q, None));
            if let Some(prev_idx) = last_on[q] {
                for (pq, link) in &mut next[prev_idx] {
                    if *pq == q {
                        *link = Some(i);
                        break;
                    }
                }
            }
            last_on[q] = Some(i);
        }
        prev.push(p);
        next.push(nx);
    }

    // Collect convex pure-CX blocks greedily: a node `j` can join the current
    // CX block iff `is_cx(j)` and every predecessor of `j` that is reachable
    // from the block is ALREADY in the block (ensuring convexity!).
    // Even simpler & 100% convex: track per-qubit `blocked[q]` (whether a
    // non-CX gate has touched `q` since the block started); any CX whose qubits
    // are both unblocked joins the block, and any non-CX marks its qubits blocked
    // (propagating forward through all subsequent ops touching blocked qubits).
    let mut visited = vec![false; n_ops];
    let mut group_of: Vec<Option<usize>> = vec![None; n_ops];
    let mut groups: Vec<Vec<Node>> = Vec::new();
    let mut group_min_idx: Vec<usize> = Vec::new();
    let mut any_changed = false;

    for seed in 0..n_ops {
        if visited[seed] || !is_cx(&seg.nodes[seed]) {
            continue;
        }
        // Grow a convex pure-CX block forward from `seed` over at most 16 qubits.
        let mut members = Vec::new();
        let mut active_qubits: Vec<usize> = Vec::new();
        let mut blocked_qubit = vec![false; seg.n_qubits];

        for j in seed..n_ops {
            if visited[j] {
                // Already assigned to an earlier block -> blocks its qubits for this block.
                for q in seg.nodes[j].op.qubits() {
                    blocked_qubit[q] = true;
                }
                continue;
            }
            let qs = seg.nodes[j].op.qubits();
            if qs.iter().any(|&q| blocked_qubit[q]) {
                // Any op touching a blocked qubit propagates the block to all its qubits
                // (preserving convexity!).
                for &q in &qs {
                    blocked_qubit[q] = true;
                }
                continue;
            }
            if is_cx(&seg.nodes[j]) {
                let mut new_qs = 0;
                for &q in &qs {
                    if !active_qubits.contains(&q) {
                        new_qs += 1;
                    }
                }
                if active_qubits.len() + new_qs <= 16 {
                    // Only join if connected to existing active qubits (or seed).
                    if members.is_empty() || qs.iter().any(|q| active_qubits.contains(q)) {
                        for &q in &qs {
                            if !active_qubits.contains(&q) {
                                active_qubits.push(q);
                            }
                        }
                        members.push(j);
                        continue;
                    }
                } else {
                    for &q in &qs {
                        blocked_qubit[q] = true;
                    }
                }
            } else if qs.iter().any(|q| active_qubits.contains(q)) {
                // Non-CX touching an active qubit blocks all its qubits.
                for &q in &qs {
                    blocked_qubit[q] = true;
                }
            }
        }

        if members.len() < 3 {
            visited[seed] = true;
            continue;
        }

        active_qubits.sort_unstable();
        let k = active_qubits.len();
        let mut q_to_local = HashMap::with_capacity(k);
        for (idx, &q) in active_qubits.iter().enumerate() {
            q_to_local.insert(q, idx);
        }

        // Build GF(2) linear map M (`row[t] ^= row[c]` for each CX(c, t)).
        let mut mat: Vec<u16> = (0..k).map(|i| 1u16 << i).collect();
        for &m in &members {
            if let Operation::Controlled {
                control, target, ..
            } = seg.nodes[m].op
            {
                let lc = q_to_local[&control];
                let lt = q_to_local[&target];
                mat[lt] ^= mat[lc];
            }
        }

        if let Some(synth) = synthesize_gf2_cnot(&mat, &active_qubits, members.len()) {
            let gid = groups.len();
            group_min_idx.push(members[0]);
            for &m in &members {
                visited[m] = true;
                group_of[m] = Some(gid);
            }
            groups.push(
                synth
                    .into_iter()
                    .map(|(c, t)| Node {
                        op: Operation::Controlled {
                            gate: Gate::X,
                            control: c,
                            target: t,
                        },
                        binding: None,
                    })
                    .collect(),
            );
            any_changed = true;
        } else {
            for &m in &members {
                visited[m] = true;
            }
        }
    }

    if !any_changed {
        return false;
    }

    for (idx, g) in group_of.iter_mut().enumerate() {
        if g.is_none() {
            let gid = groups.len();
            *g = Some(gid);
            group_min_idx.push(idx);
            groups.push(vec![seg.nodes[idx].clone()]);
        }
    }

    let n_groups = groups.len();
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n_groups];
    let mut indeg = vec![0usize; n_groups];
    for (u_idx, links) in next.iter().enumerate() {
        let gu = group_of[u_idx].unwrap();
        for &(_, maybe_v) in links {
            if let Some(v_idx) = maybe_v {
                let gv = group_of[v_idx].unwrap();
                if gu != gv && !adj[gu].contains(&gv) {
                    adj[gu].push(gv);
                    indeg[gv] += 1;
                }
            }
        }
    }

    let mut heap: BinaryHeap<Reverse<(usize, usize)>> = BinaryHeap::new();
    for g in 0..n_groups {
        if indeg[g] == 0 {
            heap.push(Reverse((group_min_idx[g], g)));
        }
    }

    let mut out = Vec::with_capacity(n_ops);
    while let Some(Reverse((_, g))) = heap.pop() {
        out.extend(std::mem::take(&mut groups[g]));
        for &v in &adj[g] {
            indeg[v] -= 1;
            if indeg[v] == 0 {
                heap.push(Reverse((group_min_idx[v], v)));
            }
        }
    }

    seg.nodes = out;
    true
}

/// Synthesize a GF(2) invertible matrix `mat` into a sequence of `(control, target)`
/// CNOTs, returning `Some(seq)` only if `seq.len() < max_cnots`.
fn synthesize_gf2_cnot(
    mat: &[u16],
    qubits: &[usize],
    max_cnots: usize,
) -> Option<Vec<(usize, usize)>> {
    let k = mat.len();
    // Identity check -> 0 CNOTs!
    if mat.iter().enumerate().all(|(i, &r)| r == (1u16 << i)) {
        return Some(Vec::new());
    }

    // Try reducing M to I (and also M^{-1} to I) with greedy row operations.
    let mut best: Option<Vec<(usize, usize)>> = None;
    let mut best_len = max_cnots;

    if let Some(ops) = reduce_gf2_greedy(mat, best_len) {
        // Reversing row ops that reduce M to I yields M!
        let seq: Vec<(usize, usize)> = ops
            .into_iter()
            .rev()
            .map(|(c, t)| (qubits[c], qubits[t]))
            .collect();
        best_len = seq.len();
        best = Some(seq);
    }

    // Also try reducing M^{-1} to I: those row ops in FORWARD order yield M!
    if let Some(inv) = invert_gf2(mat)
        && let Some(ops) = reduce_gf2_greedy(&inv, best_len)
    {
        let seq: Vec<(usize, usize)> = ops
            .into_iter()
            .map(|(c, t)| (qubits[c], qubits[t]))
            .collect();
        if seq.len() < best_len {
            best = Some(seq);
        }
    }

    // Verify exactness on small sanity check (fast: O(k + len)).
    if let Some(ref seq) = best {
        let mut check: Vec<u16> = (0..k).map(|i| 1u16 << i).collect();
        for &(c, t) in seq {
            let lc = qubits.iter().position(|&q| q == c).unwrap();
            let lt = qubits.iter().position(|&q| q == t).unwrap();
            check[lt] ^= check[lc];
        }
        debug_assert_eq!(check, mat, "GF(2) CNOT synthesis mismatch");
    }

    best
}

/// Reduce `mat` to `I` using row operations `row[t] ^= row[c]`, returning the
/// sequence of `(c, t)` pairs if strictly fewer than `limit`.
pub(crate) fn reduce_gf2_greedy(mat: &[u16], limit: usize) -> Option<Vec<(usize, usize)>> {
    let k = mat.len();
    let mut m = mat.to_vec();
    let mut ops = Vec::new();
    let is_id = |m: &[u16]| m.iter().enumerate().all(|(i, &r)| r == (1u16 << i));

    // Phase 1: greedy weight-reducing row ops (often solves small/sparse circuits
    // like 3-qubit bridges and near-identity networks optimally in 1–2 steps!).
    while !is_id(&m) && ops.len() < limit {
        let mut best_op: Option<(usize, usize)> = None;
        let mut best_gain: i32 = 0;

        for t in 0..k {
            let old_dist = (m[t] ^ (1u16 << t)).count_ones() as i32;
            if old_dist == 0 {
                continue;
            }
            for c in 0..k {
                if c == t {
                    continue;
                }
                let new_row = m[t] ^ m[c];
                let new_dist = (new_row ^ (1u16 << t)).count_ones() as i32;
                let gain = old_dist - new_dist;
                if gain > best_gain {
                    best_gain = gain;
                    best_op = Some((c, t));
                }
            }
        }

        if let Some((c, t)) = best_op {
            m[t] ^= m[c];
            ops.push((c, t));
        } else {
            break;
        }
    }

    if is_id(&m) {
        return (ops.len() < limit).then_some(ops);
    }

    // Phase 2: fallback to Gaussian elimination (with diagonal pivot fixing) to finish.
    for col in 0..k {
        if ops.len() >= limit {
            return None;
        }
        let bit = 1u16 << col;
        if m[col] & bit == 0 {
            // Find row > col (or any row != col if needed) with `bit` set and XOR into `col`.
            let src = (col + 1..k)
                .find(|&r| m[r] & bit != 0)
                .or_else(|| (0..col).find(|&r| m[r] & bit != 0))?;
            m[col] ^= m[src];
            ops.push((src, col));
        }
        for r in 0..k {
            if r != col && m[r] & bit != 0 {
                if ops.len() >= limit {
                    return None;
                }
                m[r] ^= m[col];
                ops.push((col, r));
            }
        }
    }

    (is_id(&m) && ops.len() < limit).then_some(ops)
}

fn invert_gf2(mat: &[u16]) -> Option<Vec<u16>> {
    let k = mat.len();
    let mut a = mat.to_vec();
    let mut inv: Vec<u16> = (0..k).map(|i| 1u16 << i).collect();
    for col in 0..k {
        let bit = 1u16 << col;
        let pivot = (col..k).find(|&r| a[r] & bit != 0)?;
        a.swap(col, pivot);
        inv.swap(col, pivot);
        for r in 0..k {
            if r != col && a[r] & bit != 0 {
                a[r] ^= a[col];
                inv[r] ^= inv[col];
            }
        }
    }
    Some(inv)
}

/// Check if a 1q/2q op is a diagonal gate whose qubits are all in `qubits`.
pub(crate) fn is_diagonal_on_subset(node: &Node, subset: &[bool], tol: f64) -> bool {
    if !op_is_diagonal(&node.op, node.binding.is_some(), tol) {
        return false;
    }
    node.op.qubits().into_iter().all(|q| subset[q])
}
