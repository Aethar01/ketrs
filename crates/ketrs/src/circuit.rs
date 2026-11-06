//! Backend-independent circuit representation.

use crate::classical::{Condition, Instruction};
use crate::error::Error;
use crate::gate::{Gate, Matrix2, Matrix4, adjoint};
use crate::observable::{Pauli, PauliString, PauliSum};
use crate::parameter::{Angle, Parameter};

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
    /// Exchange two qubits.
    Swap(usize, usize),
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
            Operation::Gate { gate, .. } | Operation::Controlled { gate, .. } => {
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

    /// As [`slice`](Self::slice), keeping the global phase.
    pub(crate) fn slice_with_phase(&self, from: usize, to: usize) -> Circuit {
        let mut run = self.slice(from, to);
        run.set_global_phase(self.global_phase);
        run
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
            Instruction::IfElse {
                condition,
                then,
                otherwise,
            } => {
                if condition.clbits.len() > 64 {
                    return Err(Error::InvalidArgument(
                        "conditions read at most 64 bits".into(),
                    ));
                }
                for &c in &condition.clbits {
                    clbit_ok(c)?;
                }
                for body in [then, otherwise] {
                    if body.n_qubits != self.n_qubits || body.n_clbits > self.n_clbits {
                        return Err(Error::InvalidArgument(
                            "a branch must have the circuit's qubits and at most its classical bits"
                                .into(),
                        ));
                    }
                }
            }
        }
        self.instructions.push((self.ops.len(), instruction));
        Ok(self)
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

    /// Measure `qubit` and set it to `|0⟩`.
    pub fn reset(&mut self, qubit: usize) -> &mut Self {
        self.add_instruction(Instruction::Reset { qubit })
    }

    /// Run `then` if `condition` holds, else `otherwise` (or nothing). The
    /// branches use the same qubits and classical bits as this circuit.
    pub fn if_test(
        &mut self,
        condition: Condition,
        then: Circuit,
        otherwise: Option<Circuit>,
    ) -> &mut Self {
        let otherwise =
            otherwise.unwrap_or_else(|| Circuit::with_clbits(self.n_qubits, self.n_clbits));
        self.add_instruction(Instruction::IfElse {
            condition,
            then: Box::new(then),
            otherwise: Box::new(otherwise),
        })
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
