//! Lossless quantum circuit optimization.
//!
//! Reduces gate count, two-qubit gate count, and circuit depth while
//! preserving the exact unitary action and global phase (to floating-point
//! tolerance).
//!
//! # Optimization levels
//!
//! - [`OptimizationLevel::Level0`] — no-op (returns a clone).
//! - [`OptimizationLevel::Level1`] — identity removal, adjacent inverse
//!   cancellation, same-axis rotation merging (including parametric gates),
//!   and single-qubit gate consolidation.
//! - [`OptimizationLevel::Level2`] (default) — adds commutation-aware
//!   cancellation and rotation merging, diagonal/phase-gadget simplifications,
//!   swap cancellation, diagonal removal before measurement (configurable),
//!   and DAG-based two-qubit block consolidation with exact Cartan (KAK / Weyl
//!   chamber) optimal resynthesis.
//! - [`OptimizationLevel::Level3`] — adds multi-qubit Clifford/CNOT
//!   optimizations: Pauli pushing through `CX`/`CZ`, Clifford conjugation of
//!   Pauli rotations, and GF(2) CNOT subcircuit reduction, keeping the lowest
//!   cost circuit found across iterations.
//!
//! # Losslessness & real circuits
//!
//! All optimization levels are **unitary-exact** (lossless) up to `tolerance`
//! (`1e-12` by default), preserving `global_phase` and parameter bindings.
//! If the input circuit uses only real gates (usable with
//! [`RealStatevector`](crate::RealStatevector)), the optimizer preserves
//! realness so the optimized circuit continues to run on `RealStatevector`.
//!
//! For circuits with measurements, resets, and classical control flow, every
//! unitary run and every branch/loop body is optimized independently;
//! consecutive redundant resets are deduplicated, trivial post-reset gates on
//! `|0⟩` are eliminated, and (when `remove_diagonal_before_measure` is enabled
//! at level ≥ 2) diagonal gates immediately before measurement are dropped
//! (preserving all measurement distributions and post-measurement reduced
//! states).

mod commute;
mod kak;
mod one_qubit;
mod passes;
mod pauli;

use crate::circuit::{Circuit, Operation};
use crate::classical::{Instruction, Piece};
use crate::error::Error;
use crate::gate::Gate;
use crate::observable::Pauli;
use crate::optimize::one_qubit::{gate_is_diagonal, wrap_pi};
use crate::optimize::passes::{
    Node, Segment, canonicalize_and_remove_identities, clifford_and_template_rewrites,
    commutative_cancellation, is_diagonal_on_subset, merge_and_cancel_adjacent, optimize_1q_runs,
    two_qubit_peephole_kak,
};
use crate::optimize::pauli::lower_pauli_rotations;
use std::collections::BTreeMap;

/// How aggressively to optimize a [`Circuit`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum OptimizationLevel {
    /// No optimization (returns a clone of the circuit).
    Level0 = 0,
    /// Light: identity removal, adjacent cancellation/merging, 1-qubit
    /// consolidation.
    Level1 = 1,
    /// Medium (default): adds commutation cancellation, local Clifford/swap
    /// rewrites, and exact two-qubit KAK peephole optimization.
    #[default]
    Level2 = 2,
    /// Heavy: adds multi-qubit Clifford/Pauli propagation, Pauli-rotation
    /// conjugation, and GF(2) CNOT subcircuit optimization with minimum-point
    /// tracking.
    Level3 = 3,
}

impl TryFrom<u8> for OptimizationLevel {
    type Error = Error;

    fn try_from(level: u8) -> Result<Self, Self::Error> {
        match level {
            0 => Ok(OptimizationLevel::Level0),
            1 => Ok(OptimizationLevel::Level1),
            2 => Ok(OptimizationLevel::Level2),
            3 => Ok(OptimizationLevel::Level3),
            _ => Err(Error::InvalidArgument(format!(
                "optimization level must be 0, 1, 2 or 3, got {level}"
            ))),
        }
    }
}

/// Target two-qubit gate basis for synthesis passes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Basis {
    /// Ketrs's full standard gate set (`CX`, `CZ`, `CP`, `SWAP`, `Rzz`, `Rxx`,
    /// named single-qubit gates, and `U`). Minimizes total and two-qubit
    /// operations using native multi-qubit gates.
    #[default]
    Native,
    /// Synthesize two-qubit blocks using only `CX` and single-qubit gates
    /// (matching Qiskit's `{u, cx}` target basis).
    Cx,
}

/// Options controlling [`Circuit::optimize_with`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OptimizeOptions {
    /// Optimization aggressiveness (default: [`OptimizationLevel::Level2`]).
    pub level: OptimizationLevel,
    /// Target two-qubit synthesis basis (default: [`Basis::Native`]).
    pub basis: Basis,
    /// Numerical tolerance for identity, commutation, and synthesis checks
    /// (default: `1e-12`).
    pub tolerance: f64,
    /// Remove diagonal gates acting solely on qubits that are immediately
    /// measured or reset (default: `true`; active at level ≥ 2). Preserves all
    /// measurement outcome distributions and post-measurement reduced states;
    /// disable if you inspect raw global phases of post-measurement
    /// trajectories.
    pub remove_diagonal_before_measure: bool,
    /// Whether to keep circuits whose operations are all real representable on
    /// [`RealStatevector`](crate::RealStatevector). `None` (default) preserves
    /// realness under [`Basis::Native`] and allows general complex `{U, CX}`
    /// synthesis under [`Basis::Cx`] so CX count is always optimal.
    pub preserve_real: Option<bool>,
}

impl Default for OptimizeOptions {
    fn default() -> Self {
        OptimizeOptions {
            level: OptimizationLevel::Level2,
            basis: Basis::Native,
            tolerance: 1e-12,
            remove_diagonal_before_measure: true,
            preserve_real: None,
        }
    }
}

impl OptimizeOptions {
    /// Default options at the given optimization `level`.
    pub fn with_level(level: OptimizationLevel) -> Self {
        OptimizeOptions {
            level,
            ..Default::default()
        }
    }

    fn should_preserve_real(&self, seg: &Segment, tol: f64) -> bool {
        match self.preserve_real {
            Some(v) => v && seg.is_real(tol),
            None => self.basis == Basis::Native && seg.is_real(tol),
        }
    }
}

impl Circuit {
    /// Return a losslessly optimized equivalent circuit using the default
    /// settings ([`OptimizationLevel::Level2`], [`Basis::Native`]).
    pub fn optimize(&self) -> Circuit {
        self.optimize_with(&OptimizeOptions::default())
    }

    /// Return a losslessly optimized equivalent circuit at the given
    /// [`OptimizationLevel`].
    pub fn optimize_level(&self, level: OptimizationLevel) -> Circuit {
        self.optimize_with(&OptimizeOptions::with_level(level))
    }

    /// Return a losslessly optimized equivalent circuit with custom
    /// [`OptimizeOptions`].
    pub fn optimize_with(&self, options: &OptimizeOptions) -> Circuit {
        if options.level == OptimizationLevel::Level0 {
            return self.clone();
        }
        let tol = options.tolerance.max(1e-15);
        if self.is_unitary() {
            let mut seg = Segment::from_circuit(self);
            let keep_real = options.should_preserve_real(&seg, tol);
            optimize_segment(&mut seg, options, keep_real, tol);
            return seg.into_circuit();
        }
        optimize_non_unitary(self, options, tol)
    }

    /// Gate and instruction counts by lowercase name (e.g. `"cx"`, `"rz"`, `"u"`,
    /// `"rzz"`, `"pauli_rotation"`, `"measure"`).
    pub fn gate_counts(&self) -> BTreeMap<String, usize> {
        let mut counts = BTreeMap::new();
        for op in self.ops() {
            *counts.entry(op_name(op).to_string()).or_insert(0) += 1;
        }
        for (_, inst) in self.instructions() {
            let name = match inst {
                Instruction::Measure { .. } => "measure",
                Instruction::Reset { .. } => "reset",
                Instruction::Assign { .. } => "assign",
                Instruction::IfElse { .. } => "if_else",
                Instruction::While { .. } => "while_loop",
                Instruction::For { .. } => "for_loop",
                Instruction::Break => "break",
                Instruction::Continue => "continue",
            };
            *counts.entry(name.to_string()).or_insert(0) += 1;
        }
        counts
    }

    /// Circuit depth considering only multi-qubit operations (two-qubit gates
    /// and multi-qubit gates/Pauli rotations). Single-qubit gates contribute 0.
    /// An undecomposed weight-`w` `PauliRotation` (`w >= 2`) contributes its
    /// ladder depth `2*(w - 1)` on its support qubits.
    pub fn two_qubit_depth(&self) -> usize {
        if self.is_unitary() {
            let mut depths = vec![0usize; self.n_qubits()];
            for op in self.ops() {
                let qs = op.qubits();
                if qs.len() < 2 {
                    continue;
                }
                let weight = match op {
                    Operation::PauliRotation { pauli, .. } => 2 * (pauli.weight() - 1),
                    _ => 1,
                };
                let d = qs.iter().map(|&q| depths[q]).max().unwrap_or(0) + weight;
                for &q in &qs {
                    depths[q] = d;
                }
            }
            return depths.into_iter().max().unwrap_or(0);
        }
        let mut depths = vec![0usize; self.n_qubits()];
        for piece in self.pieces() {
            match piece {
                Piece::Gates(run) => {
                    for op in run.ops() {
                        let qs = op.qubits();
                        if qs.len() < 2 {
                            continue;
                        }
                        let weight = match op {
                            Operation::PauliRotation { pauli, .. } => 2 * (pauli.weight() - 1),
                            _ => 1,
                        };
                        let d = qs.iter().map(|&q| depths[q]).max().unwrap_or(0) + weight;
                        for &q in &qs {
                            depths[q] = d;
                        }
                    }
                }
                Piece::Instruction(inst) => match inst {
                    Instruction::Measure { .. }
                    | Instruction::Reset { .. }
                    | Instruction::Assign { .. }
                    | Instruction::Break
                    | Instruction::Continue => {}
                    Instruction::IfElse {
                        then, otherwise, ..
                    } => {
                        let cur = depths.iter().copied().max().unwrap_or(0);
                        let sub = then.two_qubit_depth().max(otherwise.two_qubit_depth());
                        depths.fill(cur + sub);
                    }
                    Instruction::While { body, .. } => {
                        let cur = depths.iter().copied().max().unwrap_or(0);
                        depths.fill(cur + body.two_qubit_depth());
                    }
                    Instruction::For { passes } => {
                        let cur = depths.iter().copied().max().unwrap_or(0);
                        let sub: usize = passes.iter().map(Circuit::two_qubit_depth).sum();
                        depths.fill(cur + sub);
                    }
                },
            }
        }
        depths.into_iter().max().unwrap_or(0)
    }

    /// Circuit depth: the length of the longest path through operations
    /// (and measurements/resets/control-flow instructions) sharing qubits.
    pub fn depth(&self) -> usize {
        if self.is_unitary() {
            let mut depths = vec![0usize; self.n_qubits()];
            for op in self.ops() {
                let qs = op.qubits();
                if qs.is_empty() {
                    continue;
                }
                let d = qs.iter().map(|&q| depths[q]).max().unwrap_or(0) + 1;
                for &q in &qs {
                    depths[q] = d;
                }
            }
            return depths.into_iter().max().unwrap_or(0);
        }
        let mut depths = vec![0usize; self.n_qubits()];
        for piece in self.pieces() {
            match piece {
                Piece::Gates(run) => {
                    for op in run.ops() {
                        let qs = op.qubits();
                        if qs.is_empty() {
                            continue;
                        }
                        let d = qs.iter().map(|&q| depths[q]).max().unwrap_or(0) + 1;
                        for &q in &qs {
                            depths[q] = d;
                        }
                    }
                }
                Piece::Instruction(inst) => match inst {
                    Instruction::Measure { qubit, .. } | Instruction::Reset { qubit } => {
                        depths[*qubit] += 1;
                    }
                    Instruction::Assign { .. } | Instruction::Break | Instruction::Continue => {}
                    Instruction::IfElse {
                        then, otherwise, ..
                    } => {
                        let cur = depths.iter().copied().max().unwrap_or(0);
                        let sub = then.depth().max(otherwise.depth());
                        depths.fill(cur + sub);
                    }
                    Instruction::While { body, .. } => {
                        let cur = depths.iter().copied().max().unwrap_or(0);
                        depths.fill(cur + body.depth());
                    }
                    Instruction::For { passes } => {
                        let cur = depths.iter().copied().max().unwrap_or(0);
                        let sub: usize = passes.iter().map(Circuit::depth).sum();
                        depths.fill(cur + sub);
                    }
                },
            }
        }
        depths.into_iter().max().unwrap_or(0)
    }
}

/// Optimize a single unitary segment to convergence.
fn optimize_segment(seg: &mut Segment, options: &OptimizeOptions, keep_real: bool, tol: f64) {
    let max_iters = match options.level {
        OptimizationLevel::Level0 => return,
        OptimizationLevel::Level1 => 16,
        OptimizationLevel::Level2 => 24,
        OptimizationLevel::Level3 => 32,
    };

    canonicalize_and_remove_identities(seg, tol);

    // Before lowering PauliRotations under Basis::Cx, run commutation &
    // Clifford conjugation so duplicate/cancelling PauliRotations merge first!
    if options.basis == Basis::Cx {
        if options.level >= OptimizationLevel::Level2 {
            commutative_cancellation(seg, tol);
            clifford_and_template_rewrites(seg, options.level, Basis::Native, keep_real, tol);
            canonicalize_and_remove_identities(seg, tol);
        }
        lower_pauli_rotations(seg, options.level >= OptimizationLevel::Level2, tol);
        canonicalize_and_remove_identities(seg, tol);
    }

    let mut best = seg.clone();
    let mut best_cost = seg.cost(options.basis);

    for _ in 0..max_iters {
        let mut changed = false;

        changed |= merge_and_cancel_adjacent(seg, tol);
        changed |= optimize_1q_runs(seg, tol);

        if options.level >= OptimizationLevel::Level2 {
            changed |= commutative_cancellation(seg, tol);
            changed |=
                clifford_and_template_rewrites(seg, options.level, options.basis, keep_real, tol);
            changed |= canonicalize_and_remove_identities(seg, tol);
            changed |= optimize_1q_runs(seg, tol);
            changed |= two_qubit_peephole_kak(seg, options.basis, keep_real, tol);
            changed |= canonicalize_and_remove_identities(seg, tol);
            changed |= merge_and_cancel_adjacent(seg, tol);
            changed |= optimize_1q_runs(seg, tol);
        }

        let cost = seg.cost(options.basis);
        if cost < best_cost {
            best = seg.clone();
            best_cost = cost;
        }

        if !changed {
            break;
        }
    }

    if options.level == OptimizationLevel::Level3 && seg.cost(options.basis) > best_cost {
        *seg = best;
    }
    seg.global_phase = wrap_pi(seg.global_phase);
}

/// Optimize a circuit containing measurements, resets, or classical control.
fn optimize_non_unitary(circuit: &Circuit, options: &OptimizeOptions, tol: f64) -> Circuit {
    // First collect owned pieces (optimizing sub-circuits recursively).
    enum OwnedPiece {
        Gates(Segment),
        Inst(Instruction),
    }

    let mut pieces: Vec<OwnedPiece> = Vec::new();
    for piece in circuit.pieces() {
        match piece {
            Piece::Gates(run) => {
                let mut seg = Segment::from_circuit(&run);
                let keep_real = options.should_preserve_real(&seg, tol);
                optimize_segment(&mut seg, options, keep_real, tol);
                pieces.push(OwnedPiece::Gates(seg));
            }
            Piece::Instruction(inst) => {
                let opt_inst = match inst {
                    Instruction::IfElse {
                        condition,
                        then,
                        otherwise,
                    } => {
                        let then_opt = then.optimize_with(options);
                        let else_opt = otherwise.optimize_with(options);
                        // Drop empty no-op IfElse.
                        if then_opt.is_empty()
                            && then_opt.instructions().is_empty()
                            && then_opt.global_phase().abs() <= tol
                            && else_opt.is_empty()
                            && else_opt.instructions().is_empty()
                            && else_opt.global_phase().abs() <= tol
                        {
                            continue;
                        }
                        Instruction::IfElse {
                            condition: condition.clone(),
                            then: Box::new(then_opt),
                            otherwise: Box::new(else_opt),
                        }
                    }
                    Instruction::While { condition, body } => Instruction::While {
                        condition: condition.clone(),
                        body: Box::new(body.optimize_with(options)),
                    },
                    Instruction::For { passes } => Instruction::For {
                        passes: passes.iter().map(|p| p.optimize_with(options)).collect(),
                    },
                    other => other.clone(),
                };
                pieces.push(OwnedPiece::Inst(opt_inst));
            }
        }
    }

    // Merge adjacent Gate runs that became adjacent after dropping empty IfElse.
    let mut merged: Vec<OwnedPiece> = Vec::with_capacity(pieces.len());
    for piece in pieces {
        match (merged.last_mut(), piece) {
            (Some(OwnedPiece::Gates(prev)), OwnedPiece::Gates(next)) => {
                prev.nodes.extend(next.nodes);
                prev.global_phase += next.global_phase;
                let keep_real = options.should_preserve_real(prev, tol);
                optimize_segment(prev, options, keep_real, tol);
            }
            (_, other) => merged.push(other),
        }
    }

    // Forward pass: deduplicate consecutive resets and eliminate trivial gates
    // on qubits guaranteed to be in |0⟩ immediately after a Reset.
    let n = circuit.n_qubits();
    let mut known_zero = vec![false; n];
    let mut cleaned: Vec<OwnedPiece> = Vec::with_capacity(merged.len());

    for piece in merged {
        match piece {
            OwnedPiece::Inst(Instruction::Reset { qubit }) => {
                if known_zero[qubit] {
                    // Redundant reset on a qubit already in |0⟩.
                    continue;
                }
                known_zero[qubit] = true;
                cleaned.push(OwnedPiece::Inst(Instruction::Reset { qubit }));
            }
            OwnedPiece::Inst(inst) => {
                match &inst {
                    Instruction::Measure { qubit, .. } => {
                        // Measurement leaves |0⟩ untouched if known_zero, else unknown.
                        let _ = qubit;
                    }
                    Instruction::Assign { .. } => {}
                    Instruction::IfElse { .. }
                    | Instruction::While { .. }
                    | Instruction::For { .. }
                    | Instruction::Break
                    | Instruction::Continue => {
                        known_zero.fill(false);
                    }
                    Instruction::Reset { .. } => unreachable!(),
                }
                cleaned.push(OwnedPiece::Inst(inst));
            }
            OwnedPiece::Gates(mut seg) => {
                if known_zero.iter().any(|&z| z) {
                    let removed = eliminate_known_zero_gates(&mut seg, &mut known_zero, tol);
                    if removed {
                        let keep_real = options.should_preserve_real(&seg, tol);
                        optimize_segment(&mut seg, options, keep_real, tol);
                    }
                }
                cleaned.push(OwnedPiece::Gates(seg));
            }
        }
    }

    // Backward pass (Level >= 2): remove diagonal gates immediately before
    // Measure or Reset on all their qubits.
    if options.remove_diagonal_before_measure && options.level >= OptimizationLevel::Level2 {
        let mut measured = vec![false; n];
        for piece in cleaned.iter_mut().rev() {
            match piece {
                OwnedPiece::Inst(Instruction::Measure { qubit, .. })
                | OwnedPiece::Inst(Instruction::Reset { qubit }) => {
                    measured[*qubit] = true;
                }
                OwnedPiece::Inst(Instruction::Assign { .. }) => {}
                OwnedPiece::Inst(_) => {
                    measured.fill(false);
                }
                OwnedPiece::Gates(seg) => {
                    if measured.iter().any(|&m| m) {
                        let mut keep = vec![true; seg.nodes.len()];
                        let mut any_dropped = false;
                        for (idx, node) in seg.nodes.iter().enumerate().rev() {
                            if is_diagonal_on_subset(node, &measured, tol) {
                                keep[idx] = false;
                                any_dropped = true;
                            } else {
                                for q in node.op.qubits() {
                                    measured[q] = false;
                                }
                            }
                        }
                        if any_dropped {
                            let mut new_nodes = Vec::with_capacity(seg.nodes.len());
                            for (node, k) in seg.nodes.drain(..).zip(keep) {
                                if k {
                                    new_nodes.push(node);
                                }
                            }
                            seg.nodes = new_nodes;
                            let keep_real = options.should_preserve_real(seg, tol);
                            optimize_segment(seg, options, keep_real, tol);
                        }
                    }
                }
            }
        }
    }

    // Reassemble the optimized Circuit.
    let mut out = Circuit::with_clbits(circuit.n_qubits(), circuit.n_clbits());
    for _ in 0..circuit.n_parameters() {
        out.parameter();
    }
    for piece in cleaned {
        match piece {
            OwnedPiece::Gates(seg) => {
                if !seg.nodes.is_empty() || seg.global_phase.abs() > tol {
                    let sub = seg.into_circuit();
                    out.extend(&sub).expect("same width");
                }
            }
            OwnedPiece::Inst(inst) => {
                out.push_instruction(inst)
                    .expect("valid instruction in original circuit");
            }
        }
    }
    out
}

/// Remove leading gates that act trivially on qubits guaranteed to be `|0⟩`
/// (e.g. diagonal 1q gates and controlled gates with a `|0⟩` control),
/// updating `known_zero` as remaining gates touch qubits.
fn eliminate_known_zero_gates(seg: &mut Segment, known_zero: &mut [bool], tol: f64) -> bool {
    let mut removed = false;
    let mut out: Vec<Node> = Vec::with_capacity(seg.nodes.len());

    for node in std::mem::take(&mut seg.nodes) {
        if node.binding.is_none() {
            match &node.op {
                Operation::Gate { gate, target } if known_zero[*target] => {
                    if gate_is_diagonal(gate, tol) {
                        // Acting on |0⟩ multiplies by m[0][0] = e^{i φ}.
                        let p0 = gate.matrix()[0][0].arg();
                        seg.global_phase += p0;
                        removed = true;
                        continue;
                    }
                }
                Operation::Controlled { control, .. } if known_zero[*control] => {
                    removed = true;
                    continue;
                }
                Operation::MultiControlled { controls, .. }
                    if controls.iter().any(|&c| known_zero[c]) =>
                {
                    removed = true;
                    continue;
                }
                Operation::ControlledSwap { controls, .. }
                    if controls.iter().any(|&c| known_zero[c]) =>
                {
                    removed = true;
                    continue;
                }
                Operation::Swap(a, b) => {
                    if known_zero[*a] && known_zero[*b] {
                        removed = true;
                        continue;
                    }
                    known_zero.swap(*a, *b);
                    out.push(node);
                    continue;
                }
                _ => {}
            }
        }
        // Any remaining gate clears `known_zero` on all qubits it acts on.
        for q in node.op.qubits() {
            known_zero[q] = false;
        }
        out.push(node);
    }

    seg.nodes = out;
    seg.global_phase = wrap_pi(seg.global_phase);
    removed
}

fn op_name(op: &Operation) -> &'static str {
    match op {
        Operation::Gate { gate, .. } => match gate {
            Gate::H => "h",
            Gate::X => "x",
            Gate::Y => "y",
            Gate::Z => "z",
            Gate::S => "s",
            Gate::Sdg => "sdg",
            Gate::T => "t",
            Gate::Tdg => "tdg",
            Gate::Rx(_) => "rx",
            Gate::Ry(_) => "ry",
            Gate::Rz(_) => "rz",
            Gate::Phase(_) => "p",
            Gate::U { .. } => "u",
            Gate::Unitary(_) => "unitary",
        },
        Operation::Controlled { gate, .. } => match gate {
            Gate::X => "cx",
            Gate::Y => "cy",
            Gate::Z => "cz",
            Gate::H => "ch",
            Gate::Rx(_) => "crx",
            Gate::Ry(_) => "cry",
            Gate::Rz(_) => "crz",
            Gate::Phase(_) => "cp",
            _ => "cu",
        },
        Operation::MultiControlled { gate, controls, .. } => match (gate, controls.len()) {
            (Gate::X, 2) => "ccx",
            (Gate::X, _) => "mcx",
            (Gate::Phase(_), _) => "mcp",
            _ => "multi_controlled",
        },
        Operation::Swap(..) => "swap",
        Operation::ControlledSwap { controls, .. } => {
            if controls.len() == 1 {
                "cswap"
            } else {
                "controlled_swap"
            }
        }
        Operation::Unitary2 { .. } => "unitary2",
        Operation::PauliRotation { pauli, .. } => match pauli.ops() {
            [(_, Pauli::X), (_, Pauli::X)] => "rxx",
            [(_, Pauli::Y), (_, Pauli::Y)] => "ryy",
            [(_, Pauli::Z), (_, Pauli::Z)] => "rzz",
            [(_, Pauli::Z), (_, Pauli::X)] | [(_, Pauli::X), (_, Pauli::Z)] => "rzx",
            _ => "pauli_rotation",
        },
    }
}
