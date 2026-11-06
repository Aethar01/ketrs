//! Gate fusion: merge neighbouring gates into one- and two-qubit unitaries.
//!
//! Large statevectors are limited by memory bandwidth, so every gate costs a
//! pass over the state. Fusing gates cuts the number of passes.

use crate::circuit::{Binding, Circuit, Operation};
use crate::gate::{Gate, Matrix2, Matrix4};
use crate::observable::{Pauli, PauliString};
use num_complex::Complex64 as C64;

const ZERO: C64 = C64::new(0.0, 0.0);
const ONE: C64 = C64::new(1.0, 0.0);

/// A run of operations merged into one unitary.
struct Block {
    kind: BlockKind,
    /// The first operation, kept so a block of one stays as it was and
    /// keeps its specialised kernel.
    first: Operation,
    len: usize,
    /// Left as it is: nothing may join it.
    frozen: bool,
    /// The parameter binding of a frozen parametric operation.
    binding: Option<Binding>,
}

enum BlockKind {
    One {
        qubit: usize,
        matrix: Matrix2,
    },
    /// Matrix basis index is `2·b(qubits[1]) + b(qubits[0])`.
    Two {
        qubits: [usize; 2],
        matrix: Matrix4,
    },
    /// Kept as is; no matrix is formed.
    Frozen,
}

impl Block {
    fn new(op: &Operation, frozen: bool) -> Self {
        let kind = match op.qubits()[..] {
            _ if frozen => BlockKind::Frozen,
            [qubit] => BlockKind::One {
                qubit,
                matrix: matrix2(op),
            },
            [a, b] => BlockKind::Two {
                qubits: [a, b],
                matrix: matrix4(op, [a, b]),
            },
            _ => unreachable!("only one- and two-qubit operations are fused"),
        };
        Block {
            kind,
            first: op.clone(),
            len: 1,
            frozen,
            binding: None,
        }
    }

    /// Append `op`, which acts only on this block's qubits.
    fn push(&mut self, op: &Operation) {
        match &mut self.kind {
            BlockKind::One { matrix, .. } => *matrix = mul(&matrix2(op), matrix),
            BlockKind::Two { qubits, matrix } => *matrix = mul(&matrix4(op, *qubits), matrix),
            BlockKind::Frozen => unreachable!("nothing joins a frozen block"),
        }
        self.len += 1;
    }

    /// Prepend a one-qubit block acting on one of this block's qubits.
    fn absorb_before(&mut self, earlier: Block) {
        let (BlockKind::Two { qubits, matrix }, BlockKind::One { qubit, matrix: m }) =
            (&mut self.kind, &earlier.kind)
        else {
            unreachable!("only two-qubit blocks absorb one-qubit blocks")
        };
        *matrix = mul(matrix, &embed(m, *qubit == qubits[1]));
        self.len += earlier.len;
    }

    fn into_operation(self) -> Operation {
        if self.len == 1 {
            return self.first;
        }
        match self.kind {
            BlockKind::One { qubit, matrix } => Operation::Gate {
                gate: Gate::Unitary(matrix),
                target: qubit,
            },
            BlockKind::Two { qubits, matrix } => Operation::Unitary2 { matrix, qubits },
            BlockKind::Frozen => unreachable!("frozen blocks hold one operation"),
        }
    }
}

impl Circuit {
    /// An equivalent circuit with neighbouring gates merged into one- and
    /// two-qubit unitaries.
    ///
    /// A gate joins the latest block on its qubits when nothing after that
    /// block touches them, so it commutes with everything in between. Blocks
    /// never grow beyond two qubits.
    pub fn fused(&self) -> Circuit {
        self.fused_where(|_| true)
    }

    /// As [`fused`](Self::fused), but only operations for which `eligible`
    /// returns true are merged. Others are kept as they are and block
    /// fusion across their qubits. Parameter-dependent operations are never
    /// merged, so the result keeps the circuit's parameters.
    pub fn fused_where(&self, eligible: impl Fn(&Operation) -> bool) -> Circuit {
        self.fused_with(&eligible)
    }

    fn fused_with(&self, eligible: &dyn Fn(&Operation) -> bool) -> Circuit {
        if !self.is_unitary() {
            // Fuse each run of gates on its own: nothing may move across a
            // measurement, reset or classical control.
            let mut fused = self.empty_like();
            fused.set_global_phase(0.0);
            for piece in self.pieces() {
                match piece {
                    crate::classical::Piece::Gates(run) => {
                        fused.extend(&run.fused_with(eligible)).expect("same width");
                    }
                    crate::classical::Piece::Instruction(instruction) => {
                        fused
                            .push_instruction(instruction.clone())
                            .expect("valid in the original circuit");
                    }
                }
            }
            return fused;
        }
        let mut blocks: Vec<Option<Block>> = Vec::new();
        // The latest block touching each qubit.
        let mut last: Vec<Option<usize>> = vec![None; self.n_qubits()];
        let open = |blocks: &[Option<Block>], i: Option<usize>| {
            i.filter(|&i| blocks[i].as_ref().is_some_and(|b| !b.frozen))
        };

        let mut binding_of = vec![None; self.len()];
        for b in self.bindings() {
            binding_of[b.op] = Some(*b);
        }
        for (i, op) in self.ops().iter().enumerate() {
            let qubits = op.qubits();
            let binding = binding_of[i];
            // Larger operations, identity rotations acting on no qubit, and
            // parametric operations are kept as they are.
            let fuse = matches!(qubits.len(), 1 | 2) && binding.is_none() && eligible(op);
            let new_block = |frozen| {
                let mut block = Block::new(op, frozen);
                block.binding = binding;
                block
            };
            if qubits.len() > 2 || qubits.is_empty() {
                for &q in &qubits {
                    last[q] = Some(blocks.len());
                }
                blocks.push(Some(new_block(true)));
                continue;
            }
            if let [q] = qubits[..] {
                if fuse && let Some(i) = open(&blocks, last[q]) {
                    blocks[i].as_mut().expect("live block").push(op);
                } else {
                    last[q] = Some(blocks.len());
                    blocks.push(Some(new_block(!fuse)));
                }
                continue;
            }

            let (a, b) = (qubits[0], qubits[1]);
            if fuse
                && let Some(i) = open(&blocks, last[a])
                && last[b] == Some(i)
            {
                blocks[i].as_mut().expect("live block").push(op);
                continue;
            }
            let mut block = new_block(!fuse);
            for q in [a, b] {
                if fuse
                    && let Some(i) = open(&blocks, last[q])
                    && matches!(
                        blocks[i].as_ref().map(|b| &b.kind),
                        Some(BlockKind::One { .. })
                    )
                {
                    block.absorb_before(blocks[i].take().expect("live block"));
                }
            }
            last[a] = Some(blocks.len());
            last[b] = Some(blocks.len());
            blocks.push(Some(block));
        }

        let mut fused = self.empty_like();
        for block in blocks.into_iter().flatten() {
            let binding = block.binding;
            fused.push_with_binding(block.into_operation(), binding);
        }
        fused
    }
}

/// The matrix of an operation on one or two qubits, indexed so that bit `t`
/// of a row or column is the value of `qubits[t]`, with
/// `qubits = op.qubits()`. `None` for larger operations.
pub(crate) fn local_matrix(op: &Operation) -> Option<(Vec<usize>, Vec<Vec<C64>>)> {
    let qubits = op.qubits();
    let matrix = match qubits[..] {
        [_] => matrix2(op).iter().map(|row| row.to_vec()).collect(),
        [a, b] => matrix4(op, [a, b]).iter().map(|row| row.to_vec()).collect(),
        _ => return None,
    };
    Some((qubits, matrix))
}

/// A single-qubit `op` as a 2×2 matrix.
fn matrix2(op: &Operation) -> Matrix2 {
    match op {
        Operation::Gate { gate, .. } => gate.matrix(),
        Operation::PauliRotation { pauli, theta } => {
            let p = pauli_matrix(pauli.ops()[0].1);
            rotation(*theta, |r, c| p[r][c])
        }
        _ => unreachable!("not a single-qubit operation"),
    }
}

/// `op` as a 4×4 matrix on `qubits`, which must be the qubits it acts on.
fn matrix4(op: &Operation, qubits: [usize; 2]) -> Matrix4 {
    let (matrix, order) = match *op {
        Operation::Gate { gate, target } => return embed(&gate.matrix(), target == qubits[1]),
        Operation::PauliRotation { ref pauli, theta } => {
            return rotation(theta, |r, c| pauli_product(pauli, qubits, r, c));
        }
        Operation::Controlled {
            gate,
            control,
            target,
        } => (controlled(&gate.matrix()), [control, target]),
        Operation::Swap(a, b) => (swap(), [a, b]),
        Operation::Unitary2 { matrix, qubits } => (matrix, qubits),
    };
    if order == qubits {
        matrix
    } else {
        swap_qubit_order(&matrix)
    }
}

/// `exp(-iθ/2 · P) = cos(θ/2) I − i sin(θ/2) P`, for `P` given entrywise.
fn rotation<const N: usize>(theta: f64, p: impl Fn(usize, usize) -> C64) -> [[C64; N]; N] {
    let (sin, cos) = (theta / 2.0).sin_cos();
    std::array::from_fn(|r| {
        std::array::from_fn(|c| {
            let identity = if r == c { cos } else { 0.0 };
            C64::new(identity, 0.0) + C64::new(0.0, -sin) * p(r, c)
        })
    })
}

fn pauli_matrix(p: Pauli) -> Matrix2 {
    match p {
        Pauli::I => Gate::Phase(0.0).matrix(),
        Pauli::X => Gate::X.matrix(),
        Pauli::Y => Gate::Y.matrix(),
        Pauli::Z => Gate::Z.matrix(),
    }
}

/// Entry `(r, c)` of a Pauli string on `qubits` (low bit = `qubits[0]`).
fn pauli_product(pauli: &PauliString, qubits: [usize; 2], r: usize, c: usize) -> C64 {
    let factor = |bit: usize| {
        let p = pauli
            .ops()
            .iter()
            .find(|&&(q, _)| q == qubits[bit])
            .map_or(Pauli::I, |&(_, p)| p);
        pauli_matrix(p)[(r >> bit) & 1][(c >> bit) & 1]
    };
    factor(0) * factor(1)
}

/// A single-qubit matrix on the low (`high = false`) or high qubit of a pair.
fn embed(m: &Matrix2, high: bool) -> Matrix4 {
    std::array::from_fn(|r| {
        std::array::from_fn(|c| {
            let (own, other) = if high { (1, 0) } else { (0, 1) };
            let (r_own, r_other) = ((r >> own) & 1, (r >> other) & 1);
            let (c_own, c_other) = ((c >> own) & 1, (c >> other) & 1);
            if r_other == c_other {
                m[r_own][c_own]
            } else {
                ZERO
            }
        })
    })
}

/// `u` on the high qubit, controlled by the low one.
fn controlled(u: &Matrix2) -> Matrix4 {
    let mut m = [[ZERO; 4]; 4];
    m[0][0] = ONE;
    m[2][2] = ONE;
    for r in 0..2 {
        for c in 0..2 {
            m[2 * r + 1][2 * c + 1] = u[r][c];
        }
    }
    m
}

fn swap() -> Matrix4 {
    let mut m = [[ZERO; 4]; 4];
    for (r, c) in [(0, 0), (1, 2), (2, 1), (3, 3)] {
        m[r][c] = ONE;
    }
    m
}

/// The same operator with its two qubits listed the other way round.
fn swap_qubit_order(m: &Matrix4) -> Matrix4 {
    const P: [usize; 4] = [0, 2, 1, 3];
    std::array::from_fn(|r| std::array::from_fn(|c| m[P[r]][P[c]]))
}

fn mul<const N: usize>(a: &[[C64; N]; N], b: &[[C64; N]; N]) -> [[C64; N]; N] {
    std::array::from_fn(|r| std::array::from_fn(|c| (0..N).map(|k| a[r][k] * b[k][c]).sum()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layered_circuit_fuses_to_one_block_per_pair() {
        // Ry on both qubits then CX: everything fits one two-qubit block.
        let mut circ = Circuit::new(2);
        circ.ry(0.1, 0).ry(0.2, 1).cx(0, 1).rz(0.3, 1);
        let fused = circ.fused();
        assert_eq!(fused.len(), 1);
        assert!(matches!(fused.ops()[0], Operation::Unitary2 { .. }));
    }

    #[test]
    fn single_gates_are_left_alone() {
        let mut circ = Circuit::new(3);
        circ.h(0).cx(1, 2);
        assert_eq!(circ.fused().ops(), circ.ops());
    }

    #[test]
    fn ineligible_operations_block_fusion() {
        let mut circ = Circuit::new(2);
        circ.ry(0.1, 0).h(1).cx(0, 1).rz(0.3, 1);
        // H is kept, so CX can absorb Ry(0) but not H(1); Rz(1) joins CX.
        let fused = circ.fused_where(|op| {
            *op != Operation::Gate {
                gate: Gate::H,
                target: 1,
            }
        });
        assert_eq!(fused.len(), 2);
        assert_eq!(
            fused.ops()[0],
            Operation::Gate {
                gate: Gate::H,
                target: 1
            }
        );
    }

    #[test]
    fn blocks_do_not_grow_past_two_qubits() {
        let mut circ = Circuit::new(3);
        circ.cx(0, 1).cx(1, 2).cx(0, 1);
        assert_eq!(circ.fused().len(), 3);
    }
}
