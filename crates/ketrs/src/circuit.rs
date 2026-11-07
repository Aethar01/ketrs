//! Backend-independent circuit representation.

use crate::classical::{Expr, Instruction};
use crate::error::Error;
use crate::gate::{Gate, Matrix2, Matrix4, adjoint};
use crate::observable::{Pauli, PauliString, PauliSum};
use crate::parameter::{Angle, Parameter};
use num_complex::Complex64 as C64;

/// One step of a circuit.
#[derive(Clone, Debug, PartialEq)]
pub enum Operation {
    /// A single-qubit gate.
    Gate {
        /// The gate.
        gate: Gate,
        /// Qubit it acts on.
        target: usize,
    },
    /// A single-qubit gate applied only when `control` is `|1⟩`.
    Controlled {
        /// The gate applied to `target`.
        gate: Gate,
        /// Control qubit.
        control: usize,
        /// Target qubit.
        target: usize,
    },
    /// A single-qubit gate applied to `target` where every one of
    /// `controls` is `|1⟩` (any number, including none). Toffoli is `X` with
    /// two controls.
    MultiControlled {
        /// The gate applied to `target`.
        gate: Gate,
        /// Control qubits.
        controls: Vec<usize>,
        /// Target qubit.
        target: usize,
    },
    /// Exchange two qubits.
    Swap(usize, usize),
    /// Exchange `a` and `b` where every one of `controls` is `|1⟩` (any
    /// number); Fredkin is one control.
    ControlledSwap {
        /// Control qubits.
        controls: Vec<usize>,
        /// First qubit exchanged.
        a: usize,
        /// Second qubit exchanged.
        b: usize,
    },
    /// An arbitrary two-qubit unitary.
    ///
    /// The matrix basis index is `2·b₁ + b₀`, where `bᵢ` is the value of
    /// `qubits[i]`, matching Qiskit's `UnitaryGate` on `[q0, q1]`.
    Unitary2 {
        /// The 4×4 matrix.
        matrix: Matrix4,
        /// The qubits, least significant first.
        qubits: [usize; 2],
    },
    /// `exp(-iθ/2 · P)` for a Pauli string `P` on any number of qubits, as
    /// in Qiskit's `rzz` or `PauliEvolutionGate` (which uses `θ = 2t`).
    ///
    /// Simulated directly rather than as a CNOT ladder, since
    /// `exp(-iθ/2·P) = cos(θ/2) I − i sin(θ/2) P`.
    PauliRotation {
        /// The Pauli string `P`.
        pauli: PauliString,
        /// The rotation angle θ.
        theta: f64,
    },
}

impl Operation {
    /// The qubits this operation acts on.
    pub fn qubits(&self) -> Vec<usize> {
        match self {
            Operation::Gate { target, .. } => vec![*target],
            Operation::Controlled {
                control, target, ..
            } => vec![*control, *target],
            Operation::Swap(a, b) => vec![*a, *b],
            Operation::Unitary2 { qubits, .. } => qubits.to_vec(),
            Operation::PauliRotation { pauli, .. } => pauli.qubits().collect(),
            Operation::MultiControlled {
                controls, target, ..
            } => controls.iter().copied().chain([*target]).collect(),
            Operation::ControlledSwap { controls, a, b } => {
                controls.iter().copied().chain([*a, *b]).collect()
            }
        }
    }

    /// Whether the operation is a multi-controlled one, which engines
    /// apply directly rather than as a matrix.
    pub(crate) fn is_multi_controlled(&self) -> bool {
        matches!(
            self,
            Operation::MultiControlled { .. } | Operation::ControlledSwap { .. }
        )
    }

    /// The operation in its plainest form: multi-controlled ones with at
    /// most one control (or swaps with none) as the ordinary operation.
    pub(crate) fn simplified(self) -> Operation {
        match self {
            Operation::MultiControlled {
                gate,
                controls,
                target,
            } => match controls[..] {
                [] => Operation::Gate { gate, target },
                [control] => Operation::Controlled {
                    gate,
                    control,
                    target,
                },
                _ => Operation::MultiControlled {
                    gate,
                    controls,
                    target,
                },
            },
            Operation::ControlledSwap { controls, a, b } if controls.is_empty() => {
                Operation::Swap(a, b)
            }
            op => op,
        }
    }

    /// A controlled swap as `CX(b, a)`, `X` on `b` controlled by `controls`
    /// and `a`, then `CX(b, a)` again.
    pub(crate) fn controlled_swap_parts(controls: &[usize], a: usize, b: usize) -> [Operation; 3] {
        let cx = Operation::Controlled {
            gate: Gate::X,
            control: b,
            target: a,
        };
        let middle = Operation::MultiControlled {
            gate: Gate::X,
            controls: controls.iter().copied().chain([a]).collect(),
            target: b,
        };
        [cx.clone(), middle, cx]
    }

    /// The operation as gates on at most two qubits, for engines that only
    /// take those. Multi-controlled gates use Barenco et al.'s recursion
    /// (Phys. Rev. A 52, 1995, lemma 7.5) with `V² = U`: `C(V)` from the
    /// last control, `Cᵏ⁻¹(X)` onto it, `C(V†)`, `Cᵏ⁻¹(X)`, `Cᵏ⁻¹(V)`. Exact,
    /// with `(3ᵏ − 1)/2` two-qubit gates for `k ≥ 1` controls.
    pub(crate) fn decompose(&self, out: &mut Vec<Operation>) {
        match self {
            Operation::ControlledSwap { controls, a, b } => {
                for part in Operation::controlled_swap_parts(controls, *a, *b) {
                    part.decompose(out);
                }
            }
            Operation::MultiControlled {
                gate,
                controls,
                target,
            } => match controls.as_slice() {
                [] => out.push(Operation::Gate {
                    gate: *gate,
                    target: *target,
                }),
                [control] => out.push(Operation::Controlled {
                    gate: *gate,
                    control: *control,
                    target: *target,
                }),
                [rest @ .., last] => {
                    let v = sqrt_2x2(&gate.matrix());
                    let flip = Operation::MultiControlled {
                        gate: Gate::X,
                        controls: rest.to_vec(),
                        target: *last,
                    };
                    let parts = [
                        Operation::Controlled {
                            gate: Gate::Unitary(v),
                            control: *last,
                            target: *target,
                        },
                        flip.clone(),
                        Operation::Controlled {
                            gate: Gate::Unitary(adjoint(&v)),
                            control: *last,
                            target: *target,
                        },
                        flip,
                        Operation::MultiControlled {
                            gate: Gate::Unitary(v),
                            controls: rest.to_vec(),
                            target: *target,
                        },
                    ];
                    for part in parts {
                        part.decompose(out);
                    }
                }
            },
            _ => out.push(self.clone()),
        }
    }

    /// The same operation with each qubit `q` moved to `map[q]`, a
    /// permutation of at least as many qubits as it uses.
    pub(crate) fn remapped(&self, map: &[usize]) -> Operation {
        let m = |q: &usize| map[*q];
        match self {
            Operation::Gate { gate, target } => Operation::Gate {
                gate: *gate,
                target: m(target),
            },
            Operation::Controlled {
                gate,
                control,
                target,
            } => Operation::Controlled {
                gate: *gate,
                control: m(control),
                target: m(target),
            },
            Operation::MultiControlled {
                gate,
                controls,
                target,
            } => Operation::MultiControlled {
                gate: *gate,
                controls: controls.iter().map(m).collect(),
                target: m(target),
            },
            Operation::Swap(a, b) => Operation::Swap(m(a), m(b)),
            Operation::ControlledSwap { controls, a, b } => Operation::ControlledSwap {
                controls: controls.iter().map(m).collect(),
                a: m(a),
                b: m(b),
            },
            Operation::Unitary2 { matrix, qubits } => Operation::Unitary2 {
                matrix: *matrix,
                qubits: qubits.map(|q| map[q]),
            },
            Operation::PauliRotation { pauli, theta } => Operation::PauliRotation {
                pauli: pauli.remap(map).expect("map covers every qubit"),
                theta: *theta,
            },
        }
    }

    /// The highest qubit index this operation acts on, or 0 if it acts on
    /// none (an identity Pauli rotation, which is a global phase).
    pub fn max_qubit(&self) -> usize {
        match self {
            Operation::Gate { target, .. } => *target,
            Operation::Controlled {
                control, target, ..
            } => *control.max(target),
            Operation::Swap(a, b) => *a.max(b),
            Operation::Unitary2 { qubits, .. } => qubits[0].max(qubits[1]),
            Operation::PauliRotation { pauli, .. } => pauli.min_qubits().saturating_sub(1),
            Operation::MultiControlled { .. } | Operation::ControlledSwap { .. } => {
                self.qubits().into_iter().max().unwrap_or(0)
            }
        }
    }

    /// Check that every qubit is in range and none is repeated.
    pub fn validate(&self, n_qubits: usize) -> Result<(), Error> {
        let qubits = self.qubits();
        for (i, &q) in qubits.iter().enumerate() {
            if q >= n_qubits {
                return Err(Error::QubitOutOfRange { qubit: q, n_qubits });
            }
            if qubits[..i].contains(&q) {
                return Err(Error::DuplicateQubit(q));
            }
        }
        Ok(())
    }

    /// Replace the rotation angle of a parametrisable operation.
    fn set_angle(&mut self, value: f64) {
        match self {
            Operation::Gate { gate, .. }
            | Operation::Controlled { gate, .. }
            | Operation::MultiControlled { gate, .. } => {
                *gate = match gate {
                    Gate::Rx(_) => Gate::Rx(value),
                    Gate::Ry(_) => Gate::Ry(value),
                    Gate::Rz(_) => Gate::Rz(value),
                    Gate::Phase(_) => Gate::Phase(value),
                    _ => unreachable!("only rotations take parameters"),
                }
            }
            Operation::PauliRotation { theta, .. } => *theta = value,
            _ => unreachable!("only rotations take parameters"),
        }
    }

    /// The inverse operation.
    pub fn inverse(&self) -> Operation {
        match self {
            Operation::Gate { gate, target } => Operation::Gate {
                gate: gate.inverse(),
                target: *target,
            },
            Operation::Controlled {
                gate,
                control,
                target,
            } => Operation::Controlled {
                gate: gate.inverse(),
                control: *control,
                target: *target,
            },
            Operation::Swap(a, b) => Operation::Swap(*a, *b),
            Operation::MultiControlled {
                gate,
                controls,
                target,
            } => Operation::MultiControlled {
                gate: gate.inverse(),
                controls: controls.clone(),
                target: *target,
            },
            Operation::ControlledSwap { .. } => self.clone(),
            Operation::Unitary2 { matrix, qubits } => Operation::Unitary2 {
                matrix: adjoint(matrix),
                qubits: *qubits,
            },
            Operation::PauliRotation { pauli, theta } => Operation::PauliRotation {
                pauli: pauli.clone(),
                theta: -theta,
            },
        }
    }
}

/// A sequence of operations on a fixed number of qubits.
///
/// Builder methods take arguments in Qiskit's order (angle first, then
/// control, then target) and panic on invalid qubits. Use [`Circuit::push`]
/// for a fallible version.
///
/// A circuit also carries a global phase `e^{iφ}`, as in Qiskit, so that
/// circuits differing only by phase (such as `Rz` and `P`) simulate exactly.
///
/// Rotation angles may depend on [`Parameter`]s (see [`Circuit::parameter`]).
/// A circuit with parameters must be [`bind`](Circuit::bind)-ed to values
/// before it is simulated.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Circuit {
    n_qubits: usize,
    ops: Vec<Operation>,
    global_phase: f64,
    n_parameters: usize,
    /// Which operation angles depend on parameters, in operation order.
    bindings: Vec<Binding>,
    n_clbits: usize,
    /// Measurements, resets and classical control, each placed before the
    /// gate at its position (`len()` for the end), in order.
    instructions: Vec<(usize, Instruction)>,
}

/// Operation `op`'s angle is `scale·values[parameter] + offset`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Binding {
    pub op: usize,
    pub parameter: usize,
    pub scale: f64,
    pub offset: f64,
}

impl Circuit {
    /// An empty circuit on `n_qubits` qubits.
    pub fn new(n_qubits: usize) -> Self {
        Circuit {
            n_qubits,
            ..Default::default()
        }
    }

    /// An empty circuit on `n_qubits` qubits and `n_clbits` classical bits.
    pub fn with_clbits(n_qubits: usize, n_clbits: usize) -> Self {
        Circuit {
            n_qubits,
            n_clbits,
            ..Default::default()
        }
    }

    /// Number of classical bits.
    pub fn n_clbits(&self) -> usize {
        self.n_clbits
    }

    /// The measurements, resets and classical control, each with the number
    /// of gates before it.
    pub fn instructions(&self) -> &[(usize, Instruction)] {
        &self.instructions
    }

    /// Whether the circuit is only gates: no measurement, reset or
    /// classical control. Engines' `run` methods need this.
    pub fn is_unitary(&self) -> bool {
        self.instructions.is_empty()
    }

    /// An error for engines that run only unitary circuits.
    pub(crate) fn check_unitary(&self) -> Result<(), Error> {
        if self.is_unitary() {
            return Ok(());
        }
        Err(Error::InvalidArgument(
            "circuit has measurements, resets or classical control; run it with \
             run_trajectory or sample_circuit"
                .into(),
        ))
    }

    /// Gates `from..to` as their own circuit (same qubits, bits and
    /// parameters, no instructions or global phase).
    pub(crate) fn slice(&self, from: usize, to: usize) -> Circuit {
        Circuit {
            n_qubits: self.n_qubits,
            ops: self.ops[from..to].to_vec(),
            global_phase: 0.0,
            n_parameters: self.n_parameters,
            bindings: self
                .bindings
                .iter()
                .filter(|b| (from..to).contains(&b.op))
                .map(|b| Binding {
                    op: b.op - from,
                    ..*b
                })
                .collect(),
            n_clbits: self.n_clbits,
            instructions: Vec::new(),
        }
    }

    /// Append a measurement, reset or classical control after checking its
    /// qubits and bits.
    pub fn push_instruction(&mut self, instruction: Instruction) -> Result<&mut Self, Error> {
        let qubit_ok = |q: usize| {
            if q < self.n_qubits {
                Ok(())
            } else {
                Err(Error::QubitOutOfRange {
                    qubit: q,
                    n_qubits: self.n_qubits,
                })
            }
        };
        let clbit_ok = |c: usize| {
            if c < self.n_clbits {
                Ok(())
            } else {
                Err(Error::InvalidArgument(format!(
                    "classical bit {c} out of range for {} bits",
                    self.n_clbits
                )))
            }
        };
        match &instruction {
            Instruction::Measure { qubit, clbit } => {
                qubit_ok(*qubit)?;
                clbit_ok(*clbit)?;
            }
            Instruction::Reset { qubit } => qubit_ok(*qubit)?,
            Instruction::Assign { clbit, value } => {
                clbit_ok(*clbit)?;
                value.validate(self.n_clbits)?;
            }
            Instruction::IfElse {
                condition,
                then,
                otherwise,
            } => {
                condition.validate(self.n_clbits)?;
                for body in [then, otherwise] {
                    self.check_block(body)?;
                }
            }
            Instruction::While { condition, body } => {
                condition.validate(self.n_clbits)?;
                self.check_block(body)?;
            }
            Instruction::For { passes } => {
                for body in passes {
                    self.check_block(body)?;
                }
            }
            Instruction::Break | Instruction::Continue => {}
        }
        self.instructions.push((self.ops.len(), instruction));
        Ok(self)
    }

    /// A branch or loop body must have this circuit's qubits and at most its
    /// classical bits.
    fn check_block(&self, body: &Circuit) -> Result<(), Error> {
        if body.n_qubits != self.n_qubits || body.n_clbits > self.n_clbits {
            return Err(Error::InvalidArgument(
                "a block must have the circuit's qubits and at most its classical bits".into(),
            ));
        }
        Ok(())
    }

    fn add_instruction(&mut self, instruction: Instruction) -> &mut Self {
        if let Err(e) = self.push_instruction(instruction) {
            panic!("invalid instruction: {e}");
        }
        self
    }

    /// Measure `qubit` into classical bit `clbit`.
    pub fn measure(&mut self, qubit: usize, clbit: usize) -> &mut Self {
        self.add_instruction(Instruction::Measure { qubit, clbit })
    }

    /// Measure every qubit `q` into classical bit `q`, adding classical bits
    /// if there are fewer than qubits.
    pub fn measure_all(&mut self) -> &mut Self {
        self.n_clbits = self.n_clbits.max(self.n_qubits);
        for q in 0..self.n_qubits {
            self.measure(q, q);
        }
        self
    }

    /// Set classical bit `clbit` to whether `value` (a
    /// [`Condition`](crate::Condition) or any [`Expr`]) is nonzero, as
    /// `c[clbit] = value;` in OpenQASM 3.
    pub fn assign(&mut self, clbit: usize, value: impl Into<Expr>) -> &mut Self {
        self.add_instruction(Instruction::Assign {
            clbit,
            value: value.into(),
        })
    }

    /// Measure `qubit` and set it to `|0⟩`.
    pub fn reset(&mut self, qubit: usize) -> &mut Self {
        self.add_instruction(Instruction::Reset { qubit })
    }

    /// Run `then` if `condition` (a [`Condition`](crate::Condition) or
    /// any [`Expr`]) holds, else `otherwise` (or nothing). The branches use
    /// the same qubits and classical bits as this circuit.
    pub fn if_test(
        &mut self,
        condition: impl Into<Expr>,
        then: Circuit,
        otherwise: Option<Circuit>,
    ) -> &mut Self {
        let otherwise =
            otherwise.unwrap_or_else(|| Circuit::with_clbits(self.n_qubits, self.n_clbits));
        self.add_instruction(Instruction::IfElse {
            condition: condition.into(),
            then: Box::new(then),
            otherwise: Box::new(otherwise),
        })
    }

    /// Run `body` while `condition` holds. The body uses the same qubits and
    /// classical bits as this circuit, and normally measures into the bits
    /// the condition reads.
    pub fn while_loop(&mut self, condition: impl Into<Expr>, body: Circuit) -> &mut Self {
        self.add_instruction(Instruction::While {
            condition: condition.into(),
            body: Box::new(body),
        })
    }

    /// Run each circuit of `passes` in turn, as a `for` loop whose bodies
    /// may [`break_loop`](Self::break_loop) or
    /// [`continue_loop`](Self::continue_loop). Each pass uses the same
    /// qubits and classical bits as this circuit.
    pub fn for_loop(&mut self, passes: Vec<Circuit>) -> &mut Self {
        self.add_instruction(Instruction::For { passes })
    }

    /// Leave the innermost enclosing loop. Only valid inside a loop body
    /// (possibly within `if_test` branches there); running it anywhere
    /// else is an error.
    pub fn break_loop(&mut self) -> &mut Self {
        self.add_instruction(Instruction::Break)
    }

    /// Go on to the innermost enclosing loop's next pass (re-testing a
    /// `while` loop's condition). Only valid inside a loop body.
    pub fn continue_loop(&mut self) -> &mut Self {
        self.add_instruction(Instruction::Continue)
    }

    /// Create a new parameter. Its index is the position of its value in
    /// [`bind`](Self::bind).
    pub fn parameter(&mut self) -> Parameter {
        self.n_parameters += 1;
        Parameter(self.n_parameters - 1)
    }

    /// Create `n` new parameters.
    pub fn parameters(&mut self, n: usize) -> Vec<Parameter> {
        (0..n).map(|_| self.parameter()).collect()
    }

    /// Number of parameters; [`bind`](Self::bind) takes this many values.
    pub fn n_parameters(&self) -> usize {
        self.n_parameters
    }

    /// The parameter-dependent angles, in operation order.
    pub(crate) fn bindings(&self) -> &[Binding] {
        &self.bindings
    }

    /// A circuit with the same width, phase and parameters but no
    /// operations.
    pub(crate) fn empty_like(&self) -> Circuit {
        Circuit {
            n_qubits: self.n_qubits,
            global_phase: self.global_phase,
            n_parameters: self.n_parameters,
            n_clbits: self.n_clbits,
            ..Default::default()
        }
    }

    /// Append a valid operation, keeping its parameter binding if any.
    pub(crate) fn push_with_binding(&mut self, op: Operation, binding: Option<Binding>) {
        op.validate(self.n_qubits)
            .expect("operation uses the circuit's qubits");
        if let Some(b) = binding {
            self.bindings.push(Binding {
                op: self.ops.len(),
                ..b
            });
        }
        self.ops.push(op);
    }

    /// The same circuit with every parameter replaced by its value.
    pub fn bind(&self, values: &[f64]) -> Result<Circuit, Error> {
        if values.len() != self.n_parameters {
            return Err(Error::InvalidArgument(format!(
                "circuit has {} parameters, got {} values",
                self.n_parameters,
                values.len()
            )));
        }
        let mut ops = self.ops.clone();
        for b in &self.bindings {
            ops[b.op].set_angle(b.scale * values[b.parameter] + b.offset);
        }
        // Branches share the parameters, by index.
        let instructions = self
            .instructions
            .iter()
            .map(|(position, instruction)| {
                let instruction = match instruction {
                    Instruction::IfElse {
                        condition,
                        then,
                        otherwise,
                    } => Instruction::IfElse {
                        condition: condition.clone(),
                        then: Box::new(then.bind(&values[..then.n_parameters])?),
                        otherwise: Box::new(otherwise.bind(&values[..otherwise.n_parameters])?),
                    },
                    Instruction::While { condition, body } => Instruction::While {
                        condition: condition.clone(),
                        body: Box::new(body.bind(&values[..body.n_parameters])?),
                    },
                    Instruction::For { passes } => Instruction::For {
                        passes: passes
                            .iter()
                            .map(|body| body.bind(&values[..body.n_parameters]))
                            .collect::<Result<_, _>>()?,
                    },
                    other => other.clone(),
                };
                Ok((*position, instruction))
            })
            .collect::<Result<_, Error>>()?;
        Ok(Circuit {
            n_qubits: self.n_qubits,
            ops,
            global_phase: self.global_phase,
            n_parameters: 0,
            bindings: Vec::new(),
            n_clbits: self.n_clbits,
            instructions,
        })
    }

    /// Number of qubits.
    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    /// The operations, in order.
    pub fn ops(&self) -> &[Operation] {
        &self.ops
    }

    /// The global phase φ, in radians.
    pub fn global_phase(&self) -> f64 {
        self.global_phase
    }

    /// Set the global phase φ, in radians.
    pub fn set_global_phase(&mut self, phase: f64) -> &mut Self {
        self.global_phase = phase;
        self
    }

    /// Number of operations.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Whether the circuit has no operations.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Append an operation after checking its qubits.
    pub fn push(&mut self, op: Operation) -> Result<&mut Self, Error> {
        op.validate(self.n_qubits)?;
        self.ops.push(op);
        Ok(self)
    }

    /// Append every operation of `other`, which must have the same width.
    pub fn extend(&mut self, other: &Circuit) -> Result<&mut Self, Error> {
        if other.n_qubits != self.n_qubits {
            return Err(Error::QubitCountMismatch {
                expected: self.n_qubits,
                found: other.n_qubits,
            });
        }
        let start = self.ops.len();
        self.n_clbits = self.n_clbits.max(other.n_clbits);
        self.instructions.extend(
            other
                .instructions
                .iter()
                .map(|(position, instruction)| (position + start, instruction.clone())),
        );
        self.ops.extend_from_slice(&other.ops);
        self.global_phase += other.global_phase;
        // Parameters are shared by index.
        self.n_parameters = self.n_parameters.max(other.n_parameters);
        self.bindings.extend(other.bindings.iter().map(|b| Binding {
            op: b.op + start,
            ..*b
        }));
        Ok(self)
    }

    /// The circuit that undoes this one.
    ///
    /// # Panics
    /// If the circuit has measurements, resets or classical control, which
    /// can't be undone.
    pub fn inverse(&self) -> Circuit {
        assert!(self.is_unitary(), "only unitary circuits can be inverted");
        let last = self.ops.len().saturating_sub(1);
        Circuit {
            n_qubits: self.n_qubits,
            ops: self.ops.iter().rev().map(Operation::inverse).collect(),
            global_phase: -self.global_phase,
            n_parameters: self.n_parameters,
            n_clbits: self.n_clbits,
            instructions: Vec::new(),
            bindings: self
                .bindings
                .iter()
                .rev()
                .map(|b| Binding {
                    op: last - b.op,
                    parameter: b.parameter,
                    scale: -b.scale,
                    offset: -b.offset,
                })
                .collect(),
        }
    }

    /// The circuit as an OpenQASM 3 program; see [`qasm::dumps`](crate::qasm::dumps).
    pub fn to_qasm(&self) -> Result<String, Error> {
        crate::qasm::dumps(self)
    }

    /// This circuit applied only where every one of `controls` is `|1⟩`,
    /// as in Qiskit's `QuantumCircuit.control` but on qubits of the same
    /// register. Gates and swaps gain the controls; Pauli rotations keep
    /// their basis changes and CX ladders and control only the central
    /// `Rz`; the global phase becomes a controlled phase. Parameters carry
    /// over.
    ///
    /// Fails for circuits with measurements or classical control, and for
    /// controls the circuit acts on or out of range.
    pub fn control(&self, controls: &[usize]) -> Result<Circuit, Error> {
        self.check_unitary()?;
        for (i, &c) in controls.iter().enumerate() {
            if c >= self.n_qubits {
                return Err(Error::QubitOutOfRange {
                    qubit: c,
                    n_qubits: self.n_qubits,
                });
            }
            if controls[..i].contains(&c) {
                return Err(Error::DuplicateQubit(c));
            }
        }
        let Some((&last, rest)) = controls.split_last() else {
            return Ok(self.clone());
        };
        let mut out = Circuit::with_clbits(self.n_qubits, self.n_clbits);
        out.n_parameters = self.n_parameters;
        let with =
            |extra: &[usize]| -> Vec<usize> { controls.iter().chain(extra).copied().collect() };
        // A phase on |1…1⟩ of the controls.
        let phase = |angle: f64| Operation::MultiControlled {
            gate: Gate::Phase(angle),
            controls: rest.to_vec(),
            target: last,
        };
        // Where each operation's angle ends up, and its factor there.
        let mut angle_at = Vec::with_capacity(self.ops.len());
        for op in &self.ops {
            if let Some(&q) = op.qubits().iter().find(|q| controls.contains(q)) {
                return Err(Error::DuplicateQubit(q));
            }
            let controlled =
                |gate: &Gate, extra: &[usize], target: usize| Operation::MultiControlled {
                    gate: *gate,
                    controls: with(extra),
                    target,
                };
            angle_at.push((out.ops.len(), 1.0));
            match op {
                Operation::Gate { gate, target } => out.ops.push(controlled(gate, &[], *target)),
                Operation::Controlled {
                    gate,
                    control,
                    target,
                } => out.ops.push(controlled(gate, &[*control], *target)),
                Operation::MultiControlled {
                    gate,
                    controls: inner,
                    target,
                } => out.ops.push(controlled(gate, inner, *target)),
                Operation::Swap(a, b) => out.ops.push(Operation::ControlledSwap {
                    controls: with(&[]),
                    a: *a,
                    b: *b,
                }),
                Operation::ControlledSwap {
                    controls: inner,
                    a,
                    b,
                } => out.ops.push(Operation::ControlledSwap {
                    controls: with(inner),
                    a: *a,
                    b: *b,
                }),
                Operation::PauliRotation { pauli, theta } if pauli.ops().is_empty() => {
                    // exp(-iθ/2), a global phase, becomes P(-θ/2).
                    angle_at.last_mut().expect("pushed above").1 = -0.5;
                    out.ops.push(phase(-theta / 2.0));
                }
                Operation::PauliRotation { pauli, theta } => {
                    let parts = decompose_pauli_rotation(pauli, *theta);
                    let centre = parts
                        .iter()
                        .position(|p| {
                            matches!(
                                p,
                                Operation::Gate {
                                    gate: Gate::Rz(_),
                                    ..
                                }
                            )
                        })
                        .expect("decomposition has an Rz");
                    angle_at.last_mut().expect("pushed above").0 = out.ops.len() + centre;
                    for (k, part) in parts.into_iter().enumerate() {
                        out.ops.push(match part {
                            Operation::Gate { gate, target } if k == centre => {
                                controlled(&gate, &[], target)
                            }
                            other => other,
                        });
                    }
                }
                Operation::Unitary2 { matrix, qubits } => {
                    let mut parts = Circuit::new(self.n_qubits);
                    parts.ops = decompose_unitary2(matrix, *qubits);
                    out.ops.extend(parts.control(controls)?.ops);
                }
            }
        }
        out.bindings = self
            .bindings
            .iter()
            .map(|b| {
                let (op, factor) = angle_at[b.op];
                Binding {
                    op,
                    parameter: b.parameter,
                    scale: factor * b.scale,
                    offset: factor * b.offset,
                }
            })
            .collect();
        if self.global_phase != 0.0 {
            out.ops.push(phase(self.global_phase));
        }
        out.ops = out.ops.into_iter().map(Operation::simplified).collect();
        Ok(out)
    }

    fn add(&mut self, op: Operation) -> &mut Self {
        if let Err(e) = op.validate(self.n_qubits) {
            panic!("invalid operation {op:?}: {e}");
        }
        self.ops.push(op);
        self
    }

    /// Append a rotation whose angle is `angle`; `make` builds the
    /// operation for a numeric angle.
    fn add_rotation(
        &mut self,
        angle: impl Into<Angle>,
        make: impl FnOnce(f64) -> Operation,
    ) -> &mut Self {
        match angle.into() {
            Angle::Value(v) => self.add(make(v)),
            Angle::Linear {
                parameter,
                scale,
                offset,
            } => {
                assert!(
                    parameter.0 < self.n_parameters,
                    "parameter {} does not belong to this circuit",
                    parameter.0
                );
                self.add(make(offset));
                self.bindings.push(Binding {
                    op: self.ops.len() - 1,
                    parameter: parameter.0,
                    scale,
                    offset,
                });
                self
            }
        }
    }

    fn gate_rotation(
        &mut self,
        angle: impl Into<Angle>,
        q: usize,
        gate: fn(f64) -> Gate,
    ) -> &mut Self {
        self.add_rotation(angle, |v| Operation::Gate {
            gate: gate(v),
            target: q,
        })
    }

    fn controlled_rotation(
        &mut self,
        angle: impl Into<Angle>,
        control: usize,
        target: usize,
        gate: fn(f64) -> Gate,
    ) -> &mut Self {
        self.add_rotation(angle, |v| Operation::Controlled {
            gate: gate(v),
            control,
            target,
        })
    }

    /// Apply a single-qubit gate.
    pub fn gate(&mut self, gate: Gate, target: usize) -> &mut Self {
        self.add(Operation::Gate { gate, target })
    }

    /// `gate` on `target` where every qubit in `controls` is `|1⟩`.
    ///
    /// # Panics
    /// If a qubit is out of range or repeated (controls included); use
    /// [`push`](Self::push) with [`Operation::MultiControlled`] for an
    /// error instead.
    pub fn multi_controlled(&mut self, gate: Gate, controls: &[usize], target: usize) -> &mut Self {
        self.add(Operation::MultiControlled {
            gate,
            controls: controls.to_vec(),
            target,
        })
    }

    /// `X` on `target` where every qubit in `controls` is `|1⟩`.
    pub fn mcx(&mut self, controls: &[usize], target: usize) -> &mut Self {
        self.multi_controlled(Gate::X, controls, target)
    }

    /// Toffoli: `X` on `target` where `a` and `b` are both `|1⟩`.
    pub fn ccx(&mut self, a: usize, b: usize, target: usize) -> &mut Self {
        self.mcx(&[a, b], target)
    }

    /// Exchange `a` and `b` where every qubit in `controls` is `|1⟩`.
    pub fn controlled_swap(&mut self, controls: &[usize], a: usize, b: usize) -> &mut Self {
        self.add(Operation::ControlledSwap {
            controls: controls.to_vec(),
            a,
            b,
        })
    }

    /// Fredkin: exchange `a` and `b` where `control` is `|1⟩`.
    pub fn cswap(&mut self, control: usize, a: usize, b: usize) -> &mut Self {
        self.controlled_swap(&[control], a, b)
    }

    /// Apply a controlled single-qubit gate.
    pub fn controlled(&mut self, gate: Gate, control: usize, target: usize) -> &mut Self {
        self.add(Operation::Controlled {
            gate,
            control,
            target,
        })
    }

    /// Hadamard.
    pub fn h(&mut self, q: usize) -> &mut Self {
        self.gate(Gate::H, q)
    }
    /// Pauli X.
    pub fn x(&mut self, q: usize) -> &mut Self {
        self.gate(Gate::X, q)
    }
    /// Pauli Y.
    pub fn y(&mut self, q: usize) -> &mut Self {
        self.gate(Gate::Y, q)
    }
    /// Pauli Z.
    pub fn z(&mut self, q: usize) -> &mut Self {
        self.gate(Gate::Z, q)
    }
    /// S gate.
    pub fn s(&mut self, q: usize) -> &mut Self {
        self.gate(Gate::S, q)
    }
    /// S† gate.
    pub fn sdg(&mut self, q: usize) -> &mut Self {
        self.gate(Gate::Sdg, q)
    }
    /// T gate.
    pub fn t(&mut self, q: usize) -> &mut Self {
        self.gate(Gate::T, q)
    }
    /// T† gate.
    pub fn tdg(&mut self, q: usize) -> &mut Self {
        self.gate(Gate::Tdg, q)
    }
    /// X rotation.
    pub fn rx(&mut self, theta: impl Into<Angle>, q: usize) -> &mut Self {
        self.gate_rotation(theta, q, Gate::Rx)
    }
    /// Y rotation.
    pub fn ry(&mut self, theta: impl Into<Angle>, q: usize) -> &mut Self {
        self.gate_rotation(theta, q, Gate::Ry)
    }
    /// Z rotation.
    pub fn rz(&mut self, theta: impl Into<Angle>, q: usize) -> &mut Self {
        self.gate_rotation(theta, q, Gate::Rz)
    }
    /// Phase gate.
    pub fn p(&mut self, lambda: impl Into<Angle>, q: usize) -> &mut Self {
        self.gate_rotation(lambda, q, Gate::Phase)
    }
    /// Generic single-qubit rotation.
    pub fn u(&mut self, theta: f64, phi: f64, lambda: f64, q: usize) -> &mut Self {
        self.gate(Gate::U { theta, phi, lambda }, q)
    }
    /// Arbitrary single-qubit unitary.
    pub fn unitary(&mut self, matrix: Matrix2, q: usize) -> &mut Self {
        self.gate(Gate::Unitary(matrix), q)
    }

    /// Controlled X (CNOT).
    pub fn cx(&mut self, control: usize, target: usize) -> &mut Self {
        self.controlled(Gate::X, control, target)
    }
    /// Controlled Y.
    pub fn cy(&mut self, control: usize, target: usize) -> &mut Self {
        self.controlled(Gate::Y, control, target)
    }
    /// Controlled Z.
    pub fn cz(&mut self, control: usize, target: usize) -> &mut Self {
        self.controlled(Gate::Z, control, target)
    }
    /// Controlled phase.
    pub fn cp(&mut self, lambda: impl Into<Angle>, control: usize, target: usize) -> &mut Self {
        self.controlled_rotation(lambda, control, target, Gate::Phase)
    }
    /// Controlled X rotation.
    pub fn crx(&mut self, theta: impl Into<Angle>, control: usize, target: usize) -> &mut Self {
        self.controlled_rotation(theta, control, target, Gate::Rx)
    }
    /// Controlled Y rotation.
    pub fn cry(&mut self, theta: impl Into<Angle>, control: usize, target: usize) -> &mut Self {
        self.controlled_rotation(theta, control, target, Gate::Ry)
    }
    /// Controlled Z rotation.
    pub fn crz(&mut self, theta: impl Into<Angle>, control: usize, target: usize) -> &mut Self {
        self.controlled_rotation(theta, control, target, Gate::Rz)
    }
    /// Swap two qubits.
    pub fn swap(&mut self, a: usize, b: usize) -> &mut Self {
        self.add(Operation::Swap(a, b))
    }
    /// `exp(-iθ/2 · P)`; see [`Operation::PauliRotation`]. A fixed rotation
    /// by the identity string only changes the global phase, so it is added
    /// there.
    pub fn pauli_rotation(&mut self, theta: impl Into<Angle>, pauli: PauliString) -> &mut Self {
        let theta = theta.into();
        if let (Angle::Value(v), 0) = (theta, pauli.weight()) {
            self.global_phase -= v / 2.0;
            return self;
        }
        self.add_rotation(theta, |theta| Operation::PauliRotation { pauli, theta })
    }

    fn two_qubit_rotation(
        &mut self,
        theta: impl Into<Angle>,
        pa: Pauli,
        a: usize,
        pb: Pauli,
        b: usize,
    ) -> &mut Self {
        match PauliString::new([(a, pa), (b, pb)]) {
            Ok(pauli) => self.pauli_rotation(theta, pauli),
            Err(e) => panic!("invalid rotation on qubits {a}, {b}: {e}"),
        }
    }
    /// `exp(-iθ/2 · X⊗X)`.
    pub fn rxx(&mut self, theta: impl Into<Angle>, a: usize, b: usize) -> &mut Self {
        self.two_qubit_rotation(theta, Pauli::X, a, Pauli::X, b)
    }
    /// `exp(-iθ/2 · Y⊗Y)`.
    pub fn ryy(&mut self, theta: impl Into<Angle>, a: usize, b: usize) -> &mut Self {
        self.two_qubit_rotation(theta, Pauli::Y, a, Pauli::Y, b)
    }
    /// `exp(-iθ/2 · Z⊗Z)`.
    pub fn rzz(&mut self, theta: impl Into<Angle>, a: usize, b: usize) -> &mut Self {
        self.two_qubit_rotation(theta, Pauli::Z, a, Pauli::Z, b)
    }
    /// `exp(-iθ/2 · Z_a X_b)`, as Qiskit's `rzx` on `[a, b]`.
    pub fn rzx(&mut self, theta: impl Into<Angle>, a: usize, b: usize) -> &mut Self {
        self.two_qubit_rotation(theta, Pauli::Z, a, Pauli::X, b)
    }

    /// Append a product-formula approximation of `exp(-iHt)`: `steps`
    /// repetitions of a Trotter–Suzuki formula of the given `order`.
    ///
    /// Order 1 is Lie–Trotter, applying each term in turn; order 2 is the
    /// symmetric Strang splitting; higher even orders use Suzuki's recursion
    /// `S₂ₖ(t) = S₂ₖ₋₂(pt)² S₂ₖ₋₂((1−4p)t) S₂ₖ₋₂(pt)²`,
    /// `p = 1/(4 − 4^{1/(2k−1)})`.
    ///
    /// Terms are applied in the order given, except that each run of
    /// consecutive, mutually commuting terms is rearranged into layers on
    /// disjoint qubits: a chain `Z₀Z₁, Z₁Z₂, Z₂Z₃, …` becomes the even
    /// bonds, then the odd ones. That leaves every step's unitary exactly
    /// unchanged, but lets engines apply each layer at once (the MPS engine
    /// runs disjoint bonds in parallel) instead of as a staircase.
    pub fn pauli_evolution(
        &mut self,
        hamiltonian: &PauliSum,
        time: f64,
        steps: usize,
        order: usize,
    ) -> Result<&mut Self, Error> {
        if order == 0 || (order > 1 && order % 2 == 1) {
            return Err(Error::InvalidArgument(format!(
                "Trotter order must be 1 or even, got {order}"
            )));
        }
        if steps == 0 {
            return Err(Error::InvalidArgument(
                "Trotter steps must be at least 1".into(),
            ));
        }
        for (_, string) in hamiltonian.terms() {
            for q in string.qubits() {
                if q >= self.n_qubits {
                    return Err(Error::QubitOutOfRange {
                        qubit: q,
                        n_qubits: self.n_qubits,
                    });
                }
            }
        }
        let terms = commuting_layers(hamiltonian.terms(), self.n_qubits);
        let dt = time / steps as f64;
        for _ in 0..steps {
            self.product_formula(&terms, dt, order);
        }
        Ok(self)
    }

    fn product_formula(&mut self, terms: &[(f64, PauliString)], dt: f64, order: usize) {
        // exp(-i c t P) is a rotation by θ = 2ct.
        match order {
            1 => {
                for (c, p) in terms {
                    self.pauli_rotation(2.0 * c * dt, p.clone());
                }
            }
            2 => {
                let Some(((c_last, p_last), rest)) = terms.split_last() else {
                    return;
                };
                for (c, p) in rest {
                    self.pauli_rotation(c * dt, p.clone());
                }
                self.pauli_rotation(2.0 * c_last * dt, p_last.clone());
                for (c, p) in rest.iter().rev() {
                    self.pauli_rotation(c * dt, p.clone());
                }
            }
            _ => {
                let p = 1.0 / (4.0 - 4f64.powf(1.0 / (order as f64 - 1.0)));
                for scale in [p, p, 1.0 - 4.0 * p, p, p] {
                    self.product_formula(terms, scale * dt, order - 2);
                }
            }
        }
    }

    /// Arbitrary two-qubit unitary; see [`Operation::Unitary2`] for ordering.
    pub fn unitary2(&mut self, matrix: Matrix4, q0: usize, q1: usize) -> &mut Self {
        self.add(Operation::Unitary2 {
            matrix,
            qubits: [q0, q1],
        })
    }
}

/// `exp(-iθ/2 · P)` as basis changes, a CX parity ladder and one `Rz`:
/// `P = V Z…Z V†` with `V = H` for X and `V = S H` for Y.
pub(crate) fn decompose_pauli_rotation(pauli: &PauliString, theta: f64) -> Vec<Operation> {
    let ops = pauli.ops();
    let mut before = Vec::new();
    let mut after = Vec::new();
    for &(q, p) in ops {
        let gate = |gate| Operation::Gate { gate, target: q };
        match p {
            Pauli::X => {
                before.push(gate(Gate::H));
                after.push(gate(Gate::H));
            }
            Pauli::Y => {
                before.extend([gate(Gate::Sdg), gate(Gate::H)]);
                after.extend([gate(Gate::H), gate(Gate::S)]);
            }
            Pauli::Z | Pauli::I => {}
        }
    }
    let ladder: Vec<Operation> = ops
        .windows(2)
        .map(|w| Operation::Controlled {
            gate: Gate::X,
            control: w[0].0,
            target: w[1].0,
        })
        .collect();
    let last = ops.last().expect("weight above two").0;
    let mut out = before;
    out.extend(ladder.iter().cloned());
    out.push(Operation::Gate {
        gate: Gate::Rz(theta),
        target: last,
    });
    out.extend(ladder.into_iter().rev());
    out.extend(after);
    out
}

/// A two-qubit unitary (index `2·b₁ + b₀` for `qubits = [q₀, q₁]`) as
/// gates on at most two qubits: Givens rotations on pairs of basis states
/// adjacent in Gray order (`00, 01, 11, 10`, so each pair differs in one
/// bit) reduce it to a diagonal, column by column. Each rotation is then a
/// single-qubit gate controlled by the other qubit (negated by `X`s when
/// that bit is 0), and the diagonal is two phases, a controlled phase and
/// a global phase (an identity Pauli rotation).
pub(crate) fn decompose_unitary2(matrix: &Matrix4, qubits: [usize; 2]) -> Vec<Operation> {
    const GRAY: [usize; 4] = [0, 1, 3, 2];
    let mut m = *matrix;
    // Each rotation G on rows (r0, r1), so that G_K ⋯ G_1 · U = D.
    let mut rotations = Vec::new();
    for (j, &col) in GRAY.iter().enumerate().take(3) {
        for k in (j + 1..4).rev() {
            let (r0, r1) = (GRAY[k - 1], GRAY[k]);
            let (a, b) = (m[r0][col], m[r1][col]);
            let norm = (a.norm_sqr() + b.norm_sqr()).sqrt();
            if b.norm() < 1e-15 * norm.max(1.0) {
                continue;
            }
            let g = [[a.conj() / norm, b.conj() / norm], [-b / norm, a / norm]];
            let (x, y) = (m[r0], m[r1]);
            m[r0] = std::array::from_fn(|c| g[0][0] * x[c] + g[0][1] * y[c]);
            m[r1] = std::array::from_fn(|c| g[1][0] * x[c] + g[1][1] * y[c]);
            rotations.push((r0, r1, g));
        }
    }
    let d: [C64; 4] = std::array::from_fn(|i| m[i][i]);
    let phase = d[0].arg();
    let mut out = vec![
        // exp(-iθ/2) with θ = -2·phase.
        Operation::PauliRotation {
            pauli: PauliString::identity(),
            theta: -2.0 * phase,
        },
        Operation::Gate {
            gate: Gate::Phase((d[1] / d[0]).arg()),
            target: qubits[0],
        },
        Operation::Gate {
            gate: Gate::Phase((d[2] / d[0]).arg()),
            target: qubits[1],
        },
        Operation::Controlled {
            gate: Gate::Phase((d[3] * d[0] / (d[1] * d[2])).arg()),
            control: qubits[0],
            target: qubits[1],
        },
    ];
    // U = G_1† ⋯ G_K† · D: D first, then G_K†, …, G_1†.
    for (r0, r1, g) in rotations.into_iter().rev() {
        let bit = (r0 ^ r1).trailing_zeros() as usize;
        let (target, control) = (qubits[bit], qubits[1 - bit]);
        let negated = r0 >> (1 - bit) & 1 == 0;
        // In the target's own basis, |0⟩ first.
        let g = if r0 >> bit & 1 == 0 {
            g
        } else {
            [[g[1][1], g[1][0]], [g[0][1], g[0][0]]]
        };
        let flip = Operation::Gate {
            gate: Gate::X,
            target: control,
        };
        if negated {
            out.push(flip.clone());
        }
        out.push(Operation::Controlled {
            gate: Gate::Unitary(adjoint(&g)),
            control,
            target,
        });
        if negated {
            out.push(flip);
        }
    }
    out
}

/// A square root of a 2×2 unitary: by Cayley–Hamilton,
/// `(M + sI)² = (tr M + 2s) M` with `s² = det M`, choosing the sign of `s`
/// that keeps `tr M + 2s` away from zero.
fn sqrt_2x2(m: &Matrix2) -> Matrix2 {
    let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    let trace = m[0][0] + m[1][1];
    let mut s = det.sqrt();
    if (trace + 2.0 * s).norm() < (trace - 2.0 * s).norm() {
        s = -s;
    }
    let t = (trace + 2.0 * s).sqrt();
    [
        [(m[0][0] + s) / t, m[0][1] / t],
        [m[1][0] / t, (m[1][1] + s) / t],
    ]
}

/// A Hamiltonian term `(coefficient, string)`.
type Term = (f64, PauliString);

/// `terms` with each run of consecutive, mutually commuting terms
/// rearranged into layers of terms on disjoint qubits, each term in the
/// first layer it fits. Products over a run are unchanged exactly.
fn commuting_layers(terms: &[(f64, PauliString)], n_qubits: usize) -> Vec<(f64, PauliString)> {
    let mut out = Vec::with_capacity(terms.len());
    let mut rest = terms;
    while !rest.is_empty() {
        let mut run = 1;
        while run < rest.len()
            && rest[..run]
                .iter()
                .all(|(_, p)| p.commutes_with(&rest[run].1))
        {
            run += 1;
        }
        // Each layer's used qubits and terms.
        let mut layers: Vec<(Vec<bool>, Vec<&Term>)> = Vec::new();
        for term in &rest[..run] {
            let fits = |used: &[bool]| term.1.qubits().all(|q| !used[q]);
            let layer = match layers.iter().position(|(used, _)| fits(used)) {
                Some(layer) => layer,
                None => {
                    layers.push((vec![false; n_qubits], Vec::new()));
                    layers.len() - 1
                }
            };
            for q in term.1.qubits() {
                layers[layer].0[q] = true;
            }
            layers[layer].1.push(term);
        }
        out.extend(
            layers
                .into_iter()
                .flat_map(|(_, terms)| terms.into_iter().cloned()),
        );
        rest = &rest[run..];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_rejects_bad_qubits() {
        let mut c = Circuit::new(2);
        assert_eq!(
            c.push(Operation::Gate {
                gate: Gate::H,
                target: 2
            }),
            Err(Error::QubitOutOfRange {
                qubit: 2,
                n_qubits: 2
            })
        );
        assert_eq!(c.push(Operation::Swap(1, 1)), Err(Error::DuplicateQubit(1)));
        assert!(c.is_empty());
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn builder_panics_on_bad_qubit() {
        Circuit::new(1).cx(0, 1);
    }

    #[test]
    fn inverse_reverses_and_inverts() {
        let mut c = Circuit::new(2);
        c.s(0).cx(0, 1);
        let inv = c.inverse();
        assert_eq!(
            inv.ops(),
            &[
                Operation::Controlled {
                    gate: Gate::X,
                    control: 0,
                    target: 1
                },
                Operation::Gate {
                    gate: Gate::Sdg,
                    target: 0
                },
            ]
        );
    }
}
