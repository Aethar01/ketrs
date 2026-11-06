//! OpenQASM 3 export.
//!
//! Gates use `stdgates.inc` names where one fits (`cu` for other
//! controlled gates, `U` plus `gphase` for arbitrary matrices), and the
//! `ctrl(n) @` modifier for multi-controlled gates and swaps. Pauli
//! rotations are written as basis changes, a CX ladder and `rz`, and
//! two-qubit unitaries as controlled gates (both exact), so the output
//! uses only standard gates. Parameters become `input float[64] p0;` and
//! so on, in order.

use crate::circuit::{Binding, Circuit, Operation, decompose_pauli_rotation, decompose_unitary2};
use crate::classical::{BinaryOp, Expr, Instruction};
use crate::error::Error;
use crate::gate::{Gate, Matrix2};
use std::collections::HashMap;
use std::fmt::Write;

/// The circuit as an OpenQASM 3 program, which [`parse`](super::parse)
/// reads back to the same circuit up to rounding. Qubits form one register
/// `q` and classical bits one register `c`.
///
/// Fails only for conditions on a register other than all of `c` in
/// anything but an equality test (equalities are written bit by bit).
pub fn dumps(circuit: &Circuit) -> Result<String, Error> {
    let mut out = String::from("OPENQASM 3.0;\ninclude \"stdgates.inc\";\n");
    for k in 0..circuit.n_parameters() {
        writeln!(out, "input float[64] p{k};").unwrap();
    }
    writeln!(out, "qubit[{}] q;", circuit.n_qubits()).unwrap();
    if circuit.n_clbits() > 0 {
        writeln!(out, "bit[{}] c;", circuit.n_clbits()).unwrap();
    }
    Writer {
        out,
        n_clbits: circuit.n_clbits(),
    }
    .circuit(circuit, 0)
}

struct Writer {
    out: String,
    n_clbits: usize,
}

impl Writer {
    fn circuit(mut self, circuit: &Circuit, depth: usize) -> Result<String, Error> {
        self.body(circuit, depth)?;
        Ok(self.out)
    }

    fn line(&mut self, depth: usize, text: &str) {
        writeln!(self.out, "{}{text}", "  ".repeat(depth)).unwrap();
    }

    fn body(&mut self, circuit: &Circuit, depth: usize) -> Result<(), Error> {
        if circuit.global_phase() != 0.0 {
            self.line(depth, &format!("gphase({});", num(circuit.global_phase())));
        }
        let bindings: HashMap<usize, Binding> =
            circuit.bindings().iter().map(|b| (b.op, *b)).collect();
        let mut instructions = circuit.instructions().iter().peekable();
        for (k, op) in circuit.ops().iter().enumerate() {
            while let Some((_, instruction)) = instructions.next_if(|(at, _)| *at <= k) {
                self.instruction(instruction, depth)?;
            }
            let angle = |value: f64| match bindings.get(&k) {
                Some(b) => linear(b),
                None => num(value),
            };
            self.operation(op, &angle, depth);
        }
        for (_, instruction) in instructions {
            self.instruction(instruction, depth)?;
        }
        Ok(())
    }

    fn instruction(&mut self, instruction: &Instruction, depth: usize) -> Result<(), Error> {
        match instruction {
            Instruction::Measure { qubit, clbit } => {
                self.line(depth, &format!("c[{clbit}] = measure q[{qubit}];"));
            }
            Instruction::Reset { qubit } => self.line(depth, &format!("reset q[{qubit}];")),
            Instruction::Assign { clbit, value } => {
                let value = match value {
                    Expr::Value(0) => "false".into(),
                    Expr::Value(1) => "true".into(),
                    Expr::Bit(_) | Expr::Not(_) => self.expr(value)?,
                    Expr::Binary { op, .. }
                        if !matches!(op, BinaryOp::BitAnd | BinaryOp::BitOr | BinaryOp::BitXor) =>
                    {
                        self.expr(value)?
                    }
                    _ => format!("{} != 0", self.expr(value)?),
                };
                self.line(depth, &format!("c[{clbit}] = {value};"));
            }
            Instruction::IfElse {
                condition,
                then,
                otherwise,
            } => {
                let condition = self.expr(condition)?;
                self.line(depth, &format!("if ({condition}) {{"));
                self.body(then, depth + 1)?;
                if !otherwise.ops().is_empty()
                    || !otherwise.instructions().is_empty()
                    || otherwise.global_phase() != 0.0
                {
                    self.line(depth, "} else {");
                    self.body(otherwise, depth + 1)?;
                }
                self.line(depth, "}");
            }
            Instruction::While { condition, body } => {
                let condition = self.expr(condition)?;
                self.line(depth, &format!("while ({condition}) {{"));
                self.body(body, depth + 1)?;
                self.line(depth, "}");
            }
            Instruction::For { passes } if passes.is_empty() => {}
            Instruction::For { passes } => {
                // Passes differ only by the loop variable they were made
                // from, which is gone, so each is picked by the pass index.
                let var = format!("i{depth}");
                self.line(
                    depth,
                    &format!("for uint {var} in [0:{}] {{", passes.len() - 1),
                );
                if passes.iter().all(|p| p == &passes[0]) {
                    self.body(&passes[0], depth + 1)?;
                } else {
                    for (k, body) in passes.iter().enumerate() {
                        let keyword = if k == 0 { "if" } else { "} else if" };
                        self.line(depth + 1, &format!("{keyword} ({var} == {k}) {{"));
                        self.body(body, depth + 2)?;
                    }
                    self.line(depth + 1, "}");
                }
                self.line(depth, "}");
            }
            Instruction::Break => self.line(depth, "break;"),
            Instruction::Continue => self.line(depth, "continue;"),
        }
        Ok(())
    }

    /// One operation; `angle` formats its angle, which may be a parameter.
    fn operation(&mut self, op: &Operation, angle: &dyn Fn(f64) -> String, depth: usize) {
        match op {
            Operation::Gate { gate, target } => {
                let (call, phase) = gate_call(gate, angle);
                self.line(depth, &format!("{call} q[{target}];"));
                if let Some(phase) = phase {
                    self.line(depth, &format!("gphase({});", num(phase)));
                }
            }
            Operation::Controlled {
                gate,
                control,
                target,
            } => {
                let named = match gate {
                    Gate::X => Some("cx".to_string()),
                    Gate::Y => Some("cy".to_string()),
                    Gate::Z => Some("cz".to_string()),
                    Gate::H => Some("ch".to_string()),
                    Gate::Rx(v) => Some(format!("crx({})", angle(*v))),
                    Gate::Ry(v) => Some(format!("cry({})", angle(*v))),
                    Gate::Rz(v) => Some(format!("crz({})", angle(*v))),
                    Gate::Phase(v) => Some(format!("cp({})", angle(*v))),
                    _ => None,
                };
                let call = named.unwrap_or_else(|| {
                    let (theta, phi, lambda, gamma) = zyz(&gate.matrix());
                    format!(
                        "cu({}, {}, {}, {})",
                        num(theta),
                        num(phi),
                        num(lambda),
                        num(gamma)
                    )
                });
                self.line(depth, &format!("{call} q[{control}], q[{target}];"));
            }
            Operation::MultiControlled {
                gate,
                controls,
                target,
            } => {
                let (call, phase) = gate_call(gate, angle);
                let k = controls.len();
                let qubits = list(controls.iter().chain([target]));
                if k == 0 {
                    self.line(depth, &format!("{call} {qubits};"));
                } else {
                    self.line(depth, &format!("ctrl({k}) @ {call} {qubits};"));
                }
                // The matrix's phase, on the controls' |1…1⟩.
                if let Some(phase) = phase {
                    let p = format!("p({})", num(phase));
                    match controls.split_last() {
                        None => self.line(depth, &format!("gphase({});", num(phase))),
                        Some((last, [])) => self.line(depth, &format!("{p} q[{last}];")),
                        Some((_, rest)) => self.line(
                            depth,
                            &format!("ctrl({}) @ {p} {};", rest.len(), list(controls.iter())),
                        ),
                    }
                }
            }
            Operation::Swap(a, b) => self.line(depth, &format!("swap q[{a}], q[{b}];")),
            Operation::ControlledSwap { controls, a, b } => {
                let qubits = list(controls.iter().chain([a, b]));
                match controls.len() {
                    0 => self.line(depth, &format!("swap {qubits};")),
                    1 => self.line(depth, &format!("cswap {qubits};")),
                    k => self.line(depth, &format!("ctrl({k}) @ swap {qubits};")),
                }
            }
            Operation::PauliRotation { pauli, theta } if pauli.ops().is_empty() => {
                // exp(-iθ/2).
                self.line(depth, &format!("gphase(-({}) / 2);", angle(*theta)));
            }
            Operation::PauliRotation { pauli, theta } => {
                // Only the central rz carries the angle.
                for part in decompose_pauli_rotation(pauli, *theta) {
                    match part {
                        Operation::Gate {
                            gate: Gate::Rz(_),
                            target,
                        } => self.line(depth, &format!("rz({}) q[{target}];", angle(*theta))),
                        part => self.operation(&part, &num, depth),
                    }
                }
            }
            Operation::Unitary2 { matrix, qubits } => {
                for part in decompose_unitary2(matrix, *qubits) {
                    self.operation(&part, &num, depth);
                }
            }
        }
    }

    fn expr(&self, e: &Expr) -> Result<String, Error> {
        Ok(match e {
            Expr::Bit(c) => format!("c[{c}]"),
            Expr::Register(bits) if self.is_all(bits) => "c".into(),
            Expr::Value(v) => v.to_string(),
            Expr::Not(e) => format!("!({})", self.expr(e)?),
            Expr::BitNot { value, width }
                if *width as usize == self.n_clbits
                    && matches!(&**value, Expr::Register(bits) if self.is_all(bits)) =>
            {
                "~c".into()
            }
            Expr::Binary { op, left, right } => {
                // A partial register compared with a constant, bit by bit.
                if let (BinaryOp::Equal | BinaryOp::NotEqual, Expr::Register(bits), Expr::Value(v))
                | (BinaryOp::Equal | BinaryOp::NotEqual, Expr::Value(v), Expr::Register(bits)) =
                    (*op, &**left, &**right)
                    && !self.is_all(bits)
                {
                    let fits = bits.len() >= 64 || v >> bits.len() == 0;
                    let tests: Vec<String> = bits
                        .iter()
                        .enumerate()
                        .map(|(k, c)| {
                            let set = k < 64 && v >> k & 1 == 1;
                            if set {
                                format!("c[{c}]")
                            } else {
                                format!("!c[{c}]")
                            }
                        })
                        .collect();
                    let equal = if !fits {
                        "false".to_string()
                    } else if tests.is_empty() {
                        "true".to_string()
                    } else {
                        format!("({})", tests.join(" && "))
                    };
                    return Ok(if *op == BinaryOp::Equal {
                        equal
                    } else {
                        format!("!{equal}")
                    });
                }
                let symbol = match op {
                    BinaryOp::And => "&&",
                    BinaryOp::Or => "||",
                    BinaryOp::BitAnd => "&",
                    BinaryOp::BitOr => "|",
                    BinaryOp::BitXor => "^",
                    BinaryOp::Equal => "==",
                    BinaryOp::NotEqual => "!=",
                    BinaryOp::Less => "<",
                    BinaryOp::LessEqual => "<=",
                    BinaryOp::Greater => ">",
                    BinaryOp::GreaterEqual => ">=",
                };
                format!("({} {symbol} {})", self.expr(left)?, self.expr(right)?)
            }
            _ => {
                return Err(Error::InvalidArgument(format!(
                    "can't write {e:?} in OpenQASM: registers other than all of `c` \
                     are only supported in equality tests"
                )));
            }
        })
    }

    fn is_all(&self, bits: &[usize]) -> bool {
        bits.len() == self.n_clbits && bits.iter().enumerate().all(|(k, &c)| k == c)
    }
}

/// A single-qubit gate as a call without operands, and any global phase it
/// needs besides (for arbitrary matrices written as `U`).
fn gate_call(gate: &Gate, angle: &dyn Fn(f64) -> String) -> (String, Option<f64>) {
    let name = match gate {
        Gate::H => "h",
        Gate::X => "x",
        Gate::Y => "y",
        Gate::Z => "z",
        Gate::S => "s",
        Gate::Sdg => "sdg",
        Gate::T => "t",
        Gate::Tdg => "tdg",
        Gate::Rx(v) => return (format!("rx({})", angle(*v)), None),
        Gate::Ry(v) => return (format!("ry({})", angle(*v)), None),
        Gate::Rz(v) => return (format!("rz({})", angle(*v)), None),
        Gate::Phase(v) => return (format!("p({})", angle(*v)), None),
        Gate::U { theta, phi, lambda } => {
            return (
                format!("U({}, {}, {})", num(*theta), num(*phi), num(*lambda)),
                None,
            );
        }
        Gate::Unitary(m) => {
            let (theta, phi, lambda, gamma) = zyz(m);
            let call = format!("U({}, {}, {})", num(theta), num(phi), num(lambda));
            return (call, (gamma != 0.0).then_some(gamma));
        }
    };
    (name.into(), None)
}

/// `(θ, φ, λ, γ)` with `m = e^{iγ} U(θ, φ, λ)`.
fn zyz(m: &Matrix2) -> (f64, f64, f64, f64) {
    let (c, s) = (m[0][0].norm(), m[1][0].norm());
    let theta = 2.0 * s.atan2(c);
    const EPS: f64 = 1e-12;
    // m₀₀ = e^{iγ} cos, m₁₀ = e^{i(γ+φ)} sin, m₀₁ = -e^{i(γ+λ)} sin,
    // m₁₁ = e^{i(γ+φ+λ)} cos; with cos = 0, take φ = 0.
    let gamma = if c > EPS {
        m[0][0].arg()
    } else {
        m[1][0].arg()
    };
    let (phi, lambda) = if s > EPS {
        let phi = m[1][0].arg() - gamma;
        (phi, (-m[0][1]).arg() - gamma)
    } else {
        (0.0, m[1][1].arg() - gamma)
    };
    (theta, phi, lambda, gamma)
}

/// A number that reads back exactly.
fn num(v: f64) -> String {
    format!("{v:?}")
}

/// `scale·p + offset`.
fn linear(b: &Binding) -> String {
    let p = format!("p{}", b.parameter);
    let term = if b.scale == 1.0 {
        p
    } else {
        format!("{} * {p}", num(b.scale))
    };
    if b.offset == 0.0 {
        term
    } else {
        format!("{term} + {}", num(b.offset))
    }
}

fn list<'a>(qubits: impl Iterator<Item = &'a usize>) -> String {
    qubits
        .map(|q| format!("q[{q}]"))
        .collect::<Vec<_>>()
        .join(", ")
}
