//! Python bindings for ketrs, exposed as the `ketrs._ketrs` extension module.
//!
//! The Python-facing API mirrors the Rust one; `python/ketrs` re-exports it.

use ketrs::gate::{Matrix2, Matrix4};
/// Counts of classical-bit values as a dict of bitstrings, bit 0 rightmost.
pub(crate) fn clbit_counts(
    py: Python<'_>,
    counts: std::collections::BTreeMap<u64, usize>,
    n_clbits: usize,
) -> PyResult<Bound<'_, PyDict>> {
    let dict = PyDict::new(py);
    for (bits, count) in counts {
        dict.set_item(format!("{bits:0n_clbits$b}"), count)?;
    }
    Ok(dict)
}

/// Run `$body` with `$s` bound to the state of whatever dtype `$state`
/// holds, for a dtype enum `$kind` (`State` or the sector `Kind`).
macro_rules! dispatch {
    ($kind:ident, $state:expr, $s:ident => $body:expr) => {
        match $state {
            $kind::Complex($s) => $body,
            $kind::Real($s) => $body,
            $kind::Complex32($s) => $body,
            $kind::Real32($s) => $body,
        }
    };
}

/// Like `dispatch!`, wrapping the result in the same dtype of enum `$to`.
macro_rules! convert {
    ($from:ident => $to:ident, $state:expr, $s:ident => $body:expr) => {
        match $state {
            $from::Complex($s) => $to::Complex($body),
            $from::Real($s) => $to::Real($body),
            $from::Complex32($s) => $to::Complex32($body),
            $from::Real32($s) => $to::Real32($body),
        }
    };
}

/// Run `$body` on two states of the same dtype, or fail.
macro_rules! same_dtype {
    ($kind:ident, $a:expr, $b:expr, ($x:ident, $y:ident) => $body:expr) => {
        match ($a, $b) {
            ($kind::Complex($x), $kind::Complex($y)) => $body,
            ($kind::Real($x), $kind::Real($y)) => $body,
            ($kind::Complex32($x), $kind::Complex32($y)) => $body,
            ($kind::Real32($x), $kind::Real32($y)) => $body,
            _ => return Err(PyValueError::new_err("states have different dtypes")),
        }
    };
}

/// Build a `$kind` of `dtype`, evaluating `$body` with `$t` naming the
/// amplitude type.
macro_rules! by_dtype {
    ($kind:ident, $dtype:expr, $t:ident => $body:expr) => {
        match $dtype {
            DType::Complex => {
                type $t = ketrs::Complex64;
                $kind::Complex($body)
            }
            DType::Real => {
                type $t = f64;
                $kind::Real($body)
            }
            DType::Complex32 => {
                type $t = ketrs::Complex32;
                $kind::Complex32($body)
            }
            DType::Real32 => {
                type $t = f32;
                $kind::Real32($body)
            }
        }
    };
}

mod density;
#[cfg(feature = "gpu")]
mod gpu;
mod mps;
mod sector;

use ketrs::SectorState;
use ketrs::statevector::EigenOptions;
use ketrs::{Angle, BinaryOp, Condition, Expr, Instruction, Parameter};
use ketrs::{
    Circuit, Complex32, Complex64 as C64, DenseState, Error, Gate, Operation, Pauli, PauliString,
    PauliSum, RealStatevector, RealStatevector32, Scalar, Statevector, Statevector32,
};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use rand::SeedableRng;
use rand::rngs::StdRng;
use sector::Kind as SectorKind;
use sector::{PySectorBasis, PySectorStatevector};

fn to_py_err(e: Error) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// A circuit parameter, or a linear form `scale·θ + offset` of one, as made
/// by `Circuit.parameter()` and arithmetic such as `2 * theta + 0.5`.
#[pyclass(name = "Parameter", module = "ketrs", frozen, from_py_object)]
#[derive(Clone, Copy)]
struct PyParameter {
    parameter: Parameter,
    scale: f64,
    offset: f64,
}

impl PyParameter {
    fn angle(&self) -> Angle {
        Angle::Linear {
            parameter: self.parameter,
            scale: self.scale,
            offset: self.offset,
        }
    }

    fn with(&self, scale: f64, offset: f64) -> Self {
        PyParameter {
            parameter: self.parameter,
            scale,
            offset,
        }
    }
}

#[pymethods]
impl PyParameter {
    /// Position of this parameter's value in `Circuit.bind`.
    #[getter]
    fn index(&self) -> usize {
        self.parameter.index()
    }
    #[getter]
    fn scale(&self) -> f64 {
        self.scale
    }
    #[getter]
    fn offset(&self) -> f64 {
        self.offset
    }
    fn __mul__(&self, k: f64) -> Self {
        self.with(self.scale * k, self.offset * k)
    }
    fn __rmul__(&self, k: f64) -> Self {
        self.__mul__(k)
    }
    fn __truediv__(&self, k: f64) -> Self {
        self.__mul__(1.0 / k)
    }
    fn __add__(&self, k: f64) -> Self {
        self.with(self.scale, self.offset + k)
    }
    fn __radd__(&self, k: f64) -> Self {
        self.__add__(k)
    }
    fn __sub__(&self, k: f64) -> Self {
        self.__add__(-k)
    }
    fn __rsub__(&self, k: f64) -> Self {
        self.with(-self.scale, k - self.offset)
    }
    fn __neg__(&self) -> Self {
        self.__mul__(-1.0)
    }
    fn __repr__(&self) -> String {
        format!(
            "{}·θ{} + {}",
            self.scale,
            self.parameter.index(),
            self.offset
        )
    }
}

/// Classical bits for a condition: one index or a list.
#[derive(FromPyObject)]
enum Clbits {
    One(usize),
    Many(Vec<usize>),
}

impl Clbits {
    fn into_vec(self) -> Vec<usize> {
        match self {
            Clbits::One(c) => vec![c],
            Clbits::Many(cs) => cs,
        }
    }
}

/// A classical expression from nested tuples (see `Circuit.if_expr`).
fn parse_expr(obj: &Bound<'_, PyAny>) -> PyResult<Expr> {
    let bad = || {
        PyValueError::new_err(format!(
            "expected an expression tuple such as ('bit', 0) or ('and', a, b), got {obj}"
        ))
    };
    let tuple = obj.cast::<PyTuple>().map_err(|_| bad())?;
    let tag: String = tuple.get_item(0).map_err(|_| bad())?.extract()?;
    let arg = |k: usize| tuple.get_item(k).map_err(|_| bad());
    let sub = |k: usize| arg(k).and_then(|e| parse_expr(&e)).map(Box::new);
    Ok(match (tag.as_str(), tuple.len()) {
        ("bit", 2) => Expr::Bit(arg(1)?.extract()?),
        ("register", 2) => Expr::Register(arg(1)?.extract()?),
        ("value", 2) => Expr::Value(arg(1)?.extract()?),
        ("not", 2) => Expr::Not(sub(1)?),
        ("bit_not", 3) => Expr::BitNot {
            value: sub(1)?,
            width: arg(2)?.extract()?,
        },
        (op, 3) => {
            let op = match op {
                "and" => BinaryOp::And,
                "or" => BinaryOp::Or,
                "bit_and" => BinaryOp::BitAnd,
                "bit_or" => BinaryOp::BitOr,
                "bit_xor" => BinaryOp::BitXor,
                "==" => BinaryOp::Equal,
                "!=" => BinaryOp::NotEqual,
                "<" => BinaryOp::Less,
                "<=" => BinaryOp::LessEqual,
                ">" => BinaryOp::Greater,
                ">=" => BinaryOp::GreaterEqual,
                _ => return Err(bad()),
            };
            Expr::Binary {
                op,
                left: sub(1)?,
                right: sub(2)?,
            }
        }
        _ => return Err(bad()),
    })
}

/// A rotation angle argument: a number or a `Parameter`.
#[derive(FromPyObject)]
enum AngleArg {
    Value(f64),
    Parameter(PyParameter),
}

impl AngleArg {
    fn angle(&self) -> Angle {
        match self {
            AngleArg::Value(v) => Angle::Value(*v),
            AngleArg::Parameter(p) => p.angle(),
        }
    }
}

/// A quantum circuit. Builder methods return the circuit, so calls chain.
#[pyclass(name = "Circuit", module = "ketrs", from_py_object)]
#[derive(Clone)]
struct PyCircuit {
    inner: Circuit,
}

impl PyCircuit {
    fn push(mut slf: PyRefMut<'_, Self>, op: Operation) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner.push(op).map_err(to_py_err)?;
        Ok(slf)
    }

    fn gate(slf: PyRefMut<'_, Self>, gate: Gate, target: usize) -> PyResult<PyRefMut<'_, Self>> {
        Self::push(slf, Operation::Gate { gate, target })
    }

    fn controlled(
        slf: PyRefMut<'_, Self>,
        gate: Gate,
        control: usize,
        target: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        Self::push(
            slf,
            Operation::Controlled {
                gate,
                control,
                target,
            },
        )
    }

    /// Check `qubits` and `angle` so the Rust builder `build` can't panic,
    /// then run it.
    fn rotation<'py>(
        mut slf: PyRefMut<'py, Self>,
        angle: &AngleArg,
        qubits: &[usize],
        build: impl FnOnce(&mut Circuit, Angle),
    ) -> PyResult<PyRefMut<'py, Self>> {
        let n_qubits = slf.inner.n_qubits();
        for (i, &q) in qubits.iter().enumerate() {
            if q >= n_qubits {
                return Err(to_py_err(Error::QubitOutOfRange { qubit: q, n_qubits }));
            }
            if qubits[..i].contains(&q) {
                return Err(to_py_err(Error::DuplicateQubit(q)));
            }
        }
        let angle = angle.angle();
        if let Angle::Linear { parameter, .. } = angle
            && parameter.index() >= slf.inner.n_parameters()
        {
            return Err(PyValueError::new_err(
                "parameter does not belong to this circuit",
            ));
        }
        build(&mut slf.inner, angle);
        Ok(slf)
    }
}

#[pymethods]
impl PyCircuit {
    /// A circuit on `n_qubits` qubits and `n_clbits` classical bits.
    #[new]
    #[pyo3(signature = (n_qubits, n_clbits=0))]
    fn new(n_qubits: usize, n_clbits: usize) -> Self {
        PyCircuit {
            inner: Circuit::with_clbits(n_qubits, n_clbits),
        }
    }

    #[getter]
    fn n_clbits(&self) -> usize {
        self.inner.n_clbits()
    }

    /// Whether the circuit is only gates (no measurement, reset or
    /// classical control).
    #[getter]
    fn is_unitary(&self) -> bool {
        self.inner.is_unitary()
    }

    /// Measure `qubit` into classical bit `clbit`.
    fn measure(
        mut slf: PyRefMut<'_, Self>,
        qubit: usize,
        clbit: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner
            .push_instruction(Instruction::Measure { qubit, clbit })
            .map_err(to_py_err)?;
        Ok(slf)
    }

    /// Measure every qubit `q` into classical bit `q`, adding classical bits
    /// if needed.
    fn measure_all(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.inner.measure_all();
        slf
    }

    /// Measure `qubit` and set it to |0⟩.
    fn reset(mut slf: PyRefMut<'_, Self>, qubit: usize) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner
            .push_instruction(Instruction::Reset { qubit })
            .map_err(to_py_err)?;
        Ok(slf)
    }

    /// Run `then` if the classical bits `clbits` (one index, or a list with
    /// the least significant first) hold `value`, else `otherwise`. Branches
    /// are circuits on the same qubits and classical bits.
    #[pyo3(signature = (clbits, value, then, otherwise=None))]
    fn if_test<'py>(
        mut slf: PyRefMut<'py, Self>,
        clbits: Clbits,
        value: u64,
        then: PyRef<'_, PyCircuit>,
        otherwise: Option<PyRef<'_, PyCircuit>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let otherwise = match otherwise {
            Some(c) => c.inner.clone(),
            None => Circuit::with_clbits(slf.inner.n_qubits(), slf.inner.n_clbits()),
        };
        let condition = Condition {
            clbits: clbits.into_vec(),
            value,
        };
        slf.inner
            .push_instruction(Instruction::IfElse {
                condition: condition.into(),
                then: Box::new(then.inner.clone()),
                otherwise: Box::new(otherwise),
            })
            .map_err(to_py_err)?;
        Ok(slf)
    }

    /// Run `then` if the classical expression `condition` is nonzero, else
    /// `otherwise`. Expressions are nested tuples: `("bit", c)`,
    /// `("register", [c0, c1, …])` (least significant first),
    /// `("value", v)`, `("not", e)`, `("bit_not", e, width)`, and
    /// `(op, a, b)` for op in `and or bit_and bit_or bit_xor == != < <= > >=`.
    #[pyo3(signature = (condition, then, otherwise=None))]
    fn if_expr<'py>(
        mut slf: PyRefMut<'py, Self>,
        condition: &Bound<'_, PyAny>,
        then: PyRef<'_, PyCircuit>,
        otherwise: Option<PyRef<'_, PyCircuit>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let otherwise = match otherwise {
            Some(c) => c.inner.clone(),
            None => Circuit::with_clbits(slf.inner.n_qubits(), slf.inner.n_clbits()),
        };
        slf.inner
            .push_instruction(Instruction::IfElse {
                condition: parse_expr(condition)?,
                then: Box::new(then.inner.clone()),
                otherwise: Box::new(otherwise),
            })
            .map_err(to_py_err)?;
        Ok(slf)
    }

    /// Set classical bit `clbit` to whether the classical expression `value`
    /// (see `if_expr`) is nonzero.
    fn assign<'py>(
        mut slf: PyRefMut<'py, Self>,
        clbit: usize,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.inner
            .push_instruction(Instruction::Assign {
                clbit,
                value: parse_expr(value)?,
            })
            .map_err(to_py_err)?;
        Ok(slf)
    }

    /// Run `body` while the classical expression `condition` (see
    /// `if_expr`) is nonzero, testing before each pass. Runs that loop a
    /// million times raise.
    fn while_loop<'py>(
        mut slf: PyRefMut<'py, Self>,
        condition: &Bound<'_, PyAny>,
        body: PyRef<'_, PyCircuit>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.inner
            .push_instruction(Instruction::While {
                condition: parse_expr(condition)?,
                body: Box::new(body.inner.clone()),
            })
            .map_err(to_py_err)?;
        Ok(slf)
    }

    /// Run each circuit of `passes` in turn: a `for` loop whose bodies may
    /// `break_loop` or `continue_loop` (loops without either can simply be
    /// appended pass by pass).
    fn for_loop<'py>(
        mut slf: PyRefMut<'py, Self>,
        passes: Vec<PyRef<'_, PyCircuit>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let passes = passes.iter().map(|p| p.inner.clone()).collect();
        slf.inner
            .push_instruction(Instruction::For { passes })
            .map_err(to_py_err)?;
        Ok(slf)
    }

    /// Leave the innermost enclosing loop (valid only inside a loop body).
    fn break_loop(mut slf: PyRefMut<'_, Self>) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner
            .push_instruction(Instruction::Break)
            .map_err(to_py_err)?;
        Ok(slf)
    }

    /// Go on to the innermost enclosing loop's next pass.
    fn continue_loop(mut slf: PyRefMut<'_, Self>) -> PyResult<PyRefMut<'_, Self>> {
        slf.inner
            .push_instruction(Instruction::Continue)
            .map_err(to_py_err)?;
        Ok(slf)
    }

    #[getter]
    fn n_qubits(&self) -> usize {
        self.inner.n_qubits()
    }

    #[getter]
    fn global_phase(&self) -> f64 {
        self.inner.global_phase()
    }

    #[setter]
    fn set_global_phase(&mut self, phase: f64) {
        self.inner.set_global_phase(phase);
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "Circuit(n_qubits={}, n_clbits={}, ops={}, instructions={})",
            self.inner.n_qubits(),
            self.inner.n_clbits(),
            self.inner.len(),
            self.inner.instructions().len()
        )
    }

    /// Create a new parameter, usable as any rotation angle (also as
    /// `2 * theta + 0.5` and similar).
    fn parameter(&mut self) -> PyParameter {
        PyParameter {
            parameter: self.inner.parameter(),
            scale: 1.0,
            offset: 0.0,
        }
    }

    /// Create `n` new parameters.
    fn parameters(&mut self, n: usize) -> Vec<PyParameter> {
        (0..n).map(|_| self.parameter()).collect()
    }

    /// Number of parameters; `bind` takes this many values.
    #[getter]
    fn n_parameters(&self) -> usize {
        self.inner.n_parameters()
    }

    /// The circuit with each parameter replaced by its value.
    fn bind(&self, values: Vec<f64>) -> PyResult<Self> {
        Ok(PyCircuit {
            inner: self.inner.bind(&values).map_err(to_py_err)?,
        })
    }

    /// The circuit that undoes this one.
    fn inverse(&self) -> Self {
        PyCircuit {
            inner: self.inner.inverse(),
        }
    }

    /// The circuit as an OpenQASM 3 program (qubits `q`, bits `c`,
    /// parameters `input float[64] p0;` …), readable by `from_qasm` and by
    /// Qiskit's `qasm3.loads`.
    fn to_qasm(&self) -> PyResult<String> {
        self.inner.to_qasm().map_err(to_py_err)
    }

    /// This circuit applied only where every one of `controls` (qubits it
    /// doesn't act on) is `|1⟩`. Parameters carry over.
    fn control(&self, controls: Vec<usize>) -> PyResult<Self> {
        Ok(PyCircuit {
            inner: self.inner.control(&controls).map_err(to_py_err)?,
        })
    }

    /// Append every operation of `other`, which must have the same width.
    fn extend<'py>(
        mut slf: PyRefMut<'py, Self>,
        other: PyRef<'_, PyCircuit>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.inner.extend(&other.inner).map_err(to_py_err)?;
        Ok(slf)
    }

    fn h(slf: PyRefMut<'_, Self>, qubit: usize) -> PyResult<PyRefMut<'_, Self>> {
        Self::gate(slf, Gate::H, qubit)
    }
    fn x(slf: PyRefMut<'_, Self>, qubit: usize) -> PyResult<PyRefMut<'_, Self>> {
        Self::gate(slf, Gate::X, qubit)
    }
    fn y(slf: PyRefMut<'_, Self>, qubit: usize) -> PyResult<PyRefMut<'_, Self>> {
        Self::gate(slf, Gate::Y, qubit)
    }
    fn z(slf: PyRefMut<'_, Self>, qubit: usize) -> PyResult<PyRefMut<'_, Self>> {
        Self::gate(slf, Gate::Z, qubit)
    }
    fn s(slf: PyRefMut<'_, Self>, qubit: usize) -> PyResult<PyRefMut<'_, Self>> {
        Self::gate(slf, Gate::S, qubit)
    }
    fn sdg(slf: PyRefMut<'_, Self>, qubit: usize) -> PyResult<PyRefMut<'_, Self>> {
        Self::gate(slf, Gate::Sdg, qubit)
    }
    fn t(slf: PyRefMut<'_, Self>, qubit: usize) -> PyResult<PyRefMut<'_, Self>> {
        Self::gate(slf, Gate::T, qubit)
    }
    fn tdg(slf: PyRefMut<'_, Self>, qubit: usize) -> PyResult<PyRefMut<'_, Self>> {
        Self::gate(slf, Gate::Tdg, qubit)
    }
    fn rx(slf: PyRefMut<'_, Self>, theta: AngleArg, qubit: usize) -> PyResult<PyRefMut<'_, Self>> {
        Self::rotation(slf, &theta, &[qubit], |c, a| {
            c.rx(a, qubit);
        })
    }
    fn ry(slf: PyRefMut<'_, Self>, theta: AngleArg, qubit: usize) -> PyResult<PyRefMut<'_, Self>> {
        Self::rotation(slf, &theta, &[qubit], |c, a| {
            c.ry(a, qubit);
        })
    }
    fn rz(slf: PyRefMut<'_, Self>, theta: AngleArg, qubit: usize) -> PyResult<PyRefMut<'_, Self>> {
        Self::rotation(slf, &theta, &[qubit], |c, a| {
            c.rz(a, qubit);
        })
    }
    fn p(slf: PyRefMut<'_, Self>, lam: AngleArg, qubit: usize) -> PyResult<PyRefMut<'_, Self>> {
        Self::rotation(slf, &lam, &[qubit], |c, a| {
            c.p(a, qubit);
        })
    }
    fn u(
        slf: PyRefMut<'_, Self>,
        theta: f64,
        phi: f64,
        lam: f64,
        qubit: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        let gate = Gate::U {
            theta,
            phi,
            lambda: lam,
        };
        Self::gate(slf, gate, qubit)
    }
    /// Arbitrary single-qubit unitary, given as a 2×2 nested sequence.
    fn unitary(
        slf: PyRefMut<'_, Self>,
        matrix: Matrix2,
        qubit: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        Self::gate(slf, Gate::Unitary(matrix), qubit)
    }

    fn cx(slf: PyRefMut<'_, Self>, control: usize, target: usize) -> PyResult<PyRefMut<'_, Self>> {
        Self::controlled(slf, Gate::X, control, target)
    }
    fn cy(slf: PyRefMut<'_, Self>, control: usize, target: usize) -> PyResult<PyRefMut<'_, Self>> {
        Self::controlled(slf, Gate::Y, control, target)
    }
    fn cz(slf: PyRefMut<'_, Self>, control: usize, target: usize) -> PyResult<PyRefMut<'_, Self>> {
        Self::controlled(slf, Gate::Z, control, target)
    }
    fn cp(
        slf: PyRefMut<'_, Self>,
        lam: AngleArg,
        control: usize,
        target: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        Self::rotation(slf, &lam, &[control, target], |c, a| {
            c.cp(a, control, target);
        })
    }
    fn crx(
        slf: PyRefMut<'_, Self>,
        theta: AngleArg,
        control: usize,
        target: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        Self::rotation(slf, &theta, &[control, target], |c, a| {
            c.crx(a, control, target);
        })
    }
    fn cry(
        slf: PyRefMut<'_, Self>,
        theta: AngleArg,
        control: usize,
        target: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        Self::rotation(slf, &theta, &[control, target], |c, a| {
            c.cry(a, control, target);
        })
    }
    fn crz(
        slf: PyRefMut<'_, Self>,
        theta: AngleArg,
        control: usize,
        target: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        Self::rotation(slf, &theta, &[control, target], |c, a| {
            c.crz(a, control, target);
        })
    }
    /// Controlled arbitrary single-qubit unitary.
    fn cunitary(
        slf: PyRefMut<'_, Self>,
        matrix: Matrix2,
        control: usize,
        target: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        Self::controlled(slf, Gate::Unitary(matrix), control, target)
    }
    fn swap(slf: PyRefMut<'_, Self>, a: usize, b: usize) -> PyResult<PyRefMut<'_, Self>> {
        Self::push(slf, Operation::Swap(a, b))
    }
    /// Toffoli: `X` on `target` when both controls are `|1⟩`.
    fn ccx(
        slf: PyRefMut<'_, Self>,
        control1: usize,
        control2: usize,
        target: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        Self::mcx(slf, vec![control1, control2], target)
    }
    /// `X` on `target` when every one of `controls` is `|1⟩`.
    fn mcx(
        slf: PyRefMut<'_, Self>,
        controls: Vec<usize>,
        target: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        Self::push(
            slf,
            Operation::MultiControlled {
                gate: Gate::X,
                controls,
                target,
            },
        )
    }
    /// An arbitrary single-qubit unitary on `target`, applied when every one
    /// of `controls` is `|1⟩`.
    fn multi_controlled(
        slf: PyRefMut<'_, Self>,
        matrix: Matrix2,
        controls: Vec<usize>,
        target: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        Self::push(
            slf,
            Operation::MultiControlled {
                gate: Gate::Unitary(matrix),
                controls,
                target,
            },
        )
    }
    /// Fredkin: swap `a` and `b` when `control` is `|1⟩`.
    fn cswap(
        slf: PyRefMut<'_, Self>,
        control: usize,
        a: usize,
        b: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        Self::controlled_swap(slf, vec![control], a, b)
    }
    /// Swap `a` and `b` when every one of `controls` is `|1⟩`.
    fn controlled_swap(
        slf: PyRefMut<'_, Self>,
        controls: Vec<usize>,
        a: usize,
        b: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        Self::push(slf, Operation::ControlledSwap { controls, a, b })
    }
    /// `exp(-iθ/2 · P)` for the Pauli string `label` (rightmost character on
    /// the first qubit). Without `qubits`, character `k` from the right acts
    /// on qubit `k`; otherwise it acts on `qubits[k]`.
    #[pyo3(signature = (theta, label, qubits=None))]
    fn pauli_rotation<'py>(
        slf: PyRefMut<'py, Self>,
        theta: AngleArg,
        label: &str,
        qubits: Option<Vec<usize>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let mut pauli = PauliString::from_label(label).map_err(to_py_err)?;
        if let Some(map) = qubits {
            if map.len() != label.chars().count() {
                return Err(PyValueError::new_err(format!(
                    "label {label:?} has {} characters but {} qubits were given",
                    label.chars().count(),
                    map.len()
                )));
            }
            pauli = pauli.remap(&map).map_err(to_py_err)?;
        }
        let qubits: Vec<usize> = pauli.qubits().collect();
        Self::rotation(slf, &theta, &qubits, |c, a| {
            c.pauli_rotation(a, pauli);
        })
    }
    /// `exp(-iθ/2 · X⊗X)`.
    fn rxx(
        slf: PyRefMut<'_, Self>,
        theta: AngleArg,
        a: usize,
        b: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        Self::pauli_rotation(slf, theta, "XX", Some(vec![a, b]))
    }
    /// `exp(-iθ/2 · Y⊗Y)`.
    fn ryy(
        slf: PyRefMut<'_, Self>,
        theta: AngleArg,
        a: usize,
        b: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        Self::pauli_rotation(slf, theta, "YY", Some(vec![a, b]))
    }
    /// `exp(-iθ/2 · Z⊗Z)`.
    fn rzz(
        slf: PyRefMut<'_, Self>,
        theta: AngleArg,
        a: usize,
        b: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        Self::pauli_rotation(slf, theta, "ZZ", Some(vec![a, b]))
    }
    /// `exp(-iθ/2 · Z_a X_b)`, as Qiskit's `rzx` on `[a, b]`.
    fn rzx(
        slf: PyRefMut<'_, Self>,
        theta: AngleArg,
        a: usize,
        b: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        // Label characters run from the last qubit to the first.
        Self::pauli_rotation(slf, theta, "XZ", Some(vec![a, b]))
    }

    /// Append `steps` repetitions of a Trotter–Suzuki formula for
    /// `exp(-iHt)`: order 1 (Lie–Trotter), 2 (symmetric), or any higher even
    /// order by Suzuki's recursion. Terms are applied in the order given.
    #[pyo3(signature = (hamiltonian, time, steps=1, order=1))]
    fn pauli_evolution<'py>(
        mut slf: PyRefMut<'py, Self>,
        hamiltonian: Observable,
        time: f64,
        steps: usize,
        order: usize,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let sum = hamiltonian.into_sum()?;
        slf.inner
            .pauli_evolution(&sum, time, steps, order)
            .map_err(to_py_err)?;
        Ok(slf)
    }

    /// Arbitrary two-qubit unitary. The matrix basis index is
    /// `2·b(q1) + b(q0)`, as for Qiskit's `UnitaryGate` on `[q0, q1]`.
    fn unitary2(
        slf: PyRefMut<'_, Self>,
        matrix: Matrix4,
        q0: usize,
        q1: usize,
    ) -> PyResult<PyRefMut<'_, Self>> {
        Self::push(
            slf,
            Operation::Unitary2 {
                matrix,
                qubits: [q0, q1],
            },
        )
    }
}

/// A real linear combination of Pauli strings, such as a Hamiltonian.
#[pyclass(name = "PauliSum", module = "ketrs", from_py_object)]
#[derive(Clone)]
struct PyPauliSum {
    inner: PauliSum,
}

#[pymethods]
impl PyPauliSum {
    /// Build from `(label, coefficient)` pairs, like Qiskit's
    /// `SparsePauliOp.from_list`. The rightmost label character is qubit 0.
    #[new]
    fn new(terms: Vec<(String, f64)>) -> PyResult<Self> {
        let inner =
            PauliSum::from_list(terms.iter().map(|(l, c)| (l.as_str(), *c))).map_err(to_py_err)?;
        Ok(PyPauliSum { inner })
    }

    fn __len__(&self) -> usize {
        self.inner.terms().len()
    }

    fn __repr__(&self) -> String {
        format!("PauliSum(terms={})", self.inner.terms().len())
    }

    /// One more than the highest qubit any term acts on.
    #[getter]
    fn n_qubits(&self) -> usize {
        self.inner.min_qubits()
    }

    /// `(label, coefficient)` pairs, with labels padded to `n_qubits`
    /// characters (default: the smallest width that fits).
    #[pyo3(signature = (n_qubits=None))]
    fn to_list(&self, n_qubits: Option<usize>) -> PyResult<Vec<(String, f64)>> {
        let n = n_qubits.unwrap_or(self.inner.min_qubits());
        if n < self.inner.min_qubits() {
            return Err(PyValueError::new_err(format!(
                "terms need {} qubits, got {n}",
                self.inner.min_qubits()
            )));
        }
        Ok(self
            .inner
            .terms()
            .iter()
            .map(|(c, string)| {
                let mut label = vec!['I'; n];
                for &(q, p) in string.ops() {
                    label[n - 1 - q] = match p {
                        Pauli::I => 'I',
                        Pauli::X => 'X',
                        Pauli::Y => 'Y',
                        Pauli::Z => 'Z',
                    };
                }
                (label.into_iter().collect(), *c)
            })
            .collect())
    }

    /// Merge repeated strings and drop terms with |coefficient| ≤ `tolerance`.
    #[pyo3(signature = (tolerance=0.0))]
    fn simplify(&self, tolerance: f64) -> Self {
        PyPauliSum {
            inner: self.inner.simplify(tolerance),
        }
    }

    /// Σ|cₖ|, an upper bound on the spectral norm.
    fn coefficient_norm(&self) -> f64 {
        self.inner.coefficient_norm()
    }
}

/// Anything `Statevector.expectation` accepts.
#[derive(FromPyObject)]
enum Observable {
    Sum(PyPauliSum),
    Label(String),
    List(Vec<(String, f64)>),
}

impl Observable {
    fn into_sum(self) -> PyResult<PauliSum> {
        match self {
            Observable::Sum(s) => Ok(s.inner),
            Observable::Label(label) => PauliString::from_label(&label)
                .map(PauliSum::from)
                .map_err(to_py_err),
            Observable::List(terms) => Ok(PyPauliSum::new(terms)?.inner),
        }
    }
}

fn rng_from_seed(seed: Option<u64>) -> StdRng {
    match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_os_rng(),
    }
}

/// The amplitude type behind a Python `Statevector`.
#[derive(Clone)]
enum State {
    Complex(Statevector),
    Real(RealStatevector),
    Complex32(Statevector32),
    Real32(RealStatevector32),
}

#[derive(Clone, Copy, PartialEq)]
enum DType {
    Complex,
    Real,
    Complex32,
    Real32,
}

impl DType {
    fn parse(name: &str) -> PyResult<Self> {
        match name {
            "complex" | "complex128" => Ok(DType::Complex),
            "real" | "float64" => Ok(DType::Real),
            "complex64" => Ok(DType::Complex32),
            "float32" => Ok(DType::Real32),
            _ => Err(PyValueError::new_err(format!(
                "dtype must be 'complex', 'real', 'complex64' or 'float32', got {name:?}"
            ))),
        }
    }

    /// The numpy name of each dtype.
    fn name(self) -> &'static str {
        match self {
            DType::Complex => "complex128",
            DType::Real => "float64",
            DType::Complex32 => "complex64",
            DType::Real32 => "float32",
        }
    }
}

impl State {
    fn dtype(&self) -> DType {
        match self {
            State::Complex(_) => DType::Complex,
            State::Real(_) => DType::Real,
            State::Complex32(_) => DType::Complex32,
            State::Real32(_) => DType::Real32,
        }
    }
}

/// Amplitude arrays accepted by `Statevector.from_amplitudes`.
#[derive(FromPyObject)]
enum Amplitudes<'py> {
    Complex(PyReadonlyArray1<'py, C64>),
    Real(PyReadonlyArray1<'py, f64>),
    Complex32(PyReadonlyArray1<'py, Complex32>),
    Real32(PyReadonlyArray1<'py, f32>),
}

/// A pure state on `n` qubits. Qubit 0 is the least significant bit of a
/// basis-state index, as in Qiskit.
///
/// `dtype="complex"` (the default) runs any circuit. `dtype="real"` stores
/// real amplitudes at half the memory and runs circuits whose gates are all
/// real (H, X, Z, Ry, CX, CZ, SWAP, …); complex gates raise `ValueError`.
/// `"complex64"` and `"float32"` are the same in single precision: half the
/// memory again, with errors of about 1e-6 instead of 1e-15.
///
/// Sampling and measurement draw from a random generator owned by the state;
/// pass `seed` for reproducible results.
#[pyclass(name = "Statevector", module = "ketrs")]
struct PyStatevector {
    state: State,
    rng: StdRng,
}

impl PyStatevector {
    fn wrap(state: State, rng: StdRng) -> Self {
        PyStatevector { state, rng }
    }
}

#[pymethods]
impl PyStatevector {
    /// The all-zero state on `n_qubits` qubits.
    #[new]
    #[pyo3(signature = (n_qubits, seed=None, dtype="complex"))]
    fn new(n_qubits: usize, seed: Option<u64>, dtype: &str) -> PyResult<Self> {
        let state = by_dtype!(State, DType::parse(dtype)?, T => DenseState::<T>::zero(n_qubits));
        Ok(Self::wrap(state, rng_from_seed(seed)))
    }

    /// Run `circuit` on `|0…0⟩`.
    #[staticmethod]
    #[pyo3(signature = (circuit, seed=None, dtype="complex"))]
    fn from_circuit(
        py: Python<'_>,
        circuit: PyRef<'_, PyCircuit>,
        seed: Option<u64>,
        dtype: &str,
    ) -> PyResult<Self> {
        let circ = &circuit.inner;
        let state = by_dtype!(State, DType::parse(dtype)?, T => py
            .detach(|| DenseState::<T>::try_from_circuit(circ))
            .map_err(to_py_err)?);
        Ok(Self::wrap(state, rng_from_seed(seed)))
    }

    /// Wrap a 1-D array whose length is a power of two. The array's dtype
    /// (complex128, float64, complex64 or float32) sets the state's.
    #[staticmethod]
    #[pyo3(signature = (amplitudes, seed=None))]
    fn from_amplitudes(amplitudes: Amplitudes<'_>, seed: Option<u64>) -> PyResult<Self> {
        let state = match amplitudes {
            Amplitudes::Complex(a) => State::Complex(
                Statevector::from_amplitudes(a.as_array().to_vec()).map_err(to_py_err)?,
            ),
            Amplitudes::Real(a) => State::Real(
                RealStatevector::from_amplitudes(a.as_array().to_vec()).map_err(to_py_err)?,
            ),
            Amplitudes::Complex32(a) => State::Complex32(
                Statevector32::from_amplitudes(a.as_array().to_vec()).map_err(to_py_err)?,
            ),
            Amplitudes::Real32(a) => State::Real32(
                RealStatevector32::from_amplitudes(a.as_array().to_vec()).map_err(to_py_err)?,
            ),
        };
        Ok(Self::wrap(state, rng_from_seed(seed)))
    }

    #[getter]
    fn n_qubits(&self) -> usize {
        dispatch!(State, &self.state, s => s.n_qubits())
    }

    /// `"complex128"`, `"float64"`, `"complex64"` or `"float32"`.
    #[getter]
    fn dtype(&self) -> &'static str {
        self.state.dtype().name()
    }

    fn __repr__(&self) -> String {
        format!(
            "Statevector(n_qubits={}, dtype={})",
            self.n_qubits(),
            self.dtype()
        )
    }

    /// Apply `circuit`, which must have the same width. A circuit with
    /// measurements, resets or classical control follows one random history
    /// (using this state's generator) and returns its classical bits;
    /// otherwise this returns None. With `noise` (a `NoiseModel`), the
    /// history is a noisy trajectory: channels after every gate pick one of
    /// their Kraus operators at random, and readout errors flip measured
    /// bits.
    #[pyo3(signature = (circuit, noise=None))]
    fn run(
        &mut self,
        py: Python<'_>,
        circuit: PyRef<'_, PyCircuit>,
        noise: Option<PyRef<'_, density::PyNoiseModel>>,
    ) -> PyResult<Option<Vec<usize>>> {
        let circ = &circuit.inner;
        if let Some(noise) = noise {
            let (noise, rng) = (&noise.inner, &mut self.rng);
            let bits = dispatch!(State, &mut self.state, s => py.detach(|| s.run_noisy_trajectory(circ, noise, rng)))
                .map_err(to_py_err)?;
            return Ok((!circ.is_unitary()).then(|| bits.into_iter().map(usize::from).collect()));
        }
        if circ.is_unitary() {
            dispatch!(State, &mut self.state, s => py.detach(|| s.run(circ))).map_err(to_py_err)?;
            return Ok(None);
        }
        let rng = &mut self.rng;
        let bits =
            dispatch!(State, &mut self.state, s => py.detach(|| s.run_trajectory(circ, rng)))
                .map_err(to_py_err)?;
        Ok(Some(bits.into_iter().map(usize::from).collect()))
    }

    /// Run `circuit` from this state `shots` times and count the final
    /// classical bits, as bitstrings with bit 0 rightmost (as in Qiskit's
    /// counts). The state itself is unchanged. With `noise`, each shot is a
    /// noisy trajectory (see `run`).
    #[pyo3(signature = (circuit, shots, noise=None))]
    fn sample_circuit<'py>(
        &mut self,
        py: Python<'py>,
        circuit: PyRef<'_, PyCircuit>,
        shots: usize,
        noise: Option<PyRef<'_, density::PyNoiseModel>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let circ = &circuit.inner;
        let n = circ.n_clbits();
        let rng = &mut self.rng;
        let counts = match noise {
            Some(noise) => {
                let noise = &noise.inner;
                dispatch!(State, &self.state, s => py.detach(|| s.sample_noisy(circ, noise, shots, rng)))
            }
            None => {
                dispatch!(State, &self.state, s => py.detach(|| s.sample_circuit(circ, shots, rng)))
            }
        }
        .map_err(to_py_err)?;
        clbit_counts(py, counts, n)
    }

    /// A copy of the amplitudes, as an array of the state's dtype.
    fn amplitudes<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        dispatch!(State, &self.state, s => PyArray1::from_slice(py, s.amplitudes()).into_any())
    }

    /// Measurement probability of each basis state.
    fn probabilities<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, dispatch!(State, &self.state, s => s.probabilities()))
    }

    /// `⟨self|other⟩`. Both states must have the same dtype.
    fn inner(&self, other: PyRef<'_, PyStatevector>) -> PyResult<C64> {
        same_dtype!(State, &self.state, &other.state, (a, b) => {
            a.inner(b).map(Scalar::to_complex).map_err(to_py_err)
        })
    }

    /// `⟨ψ|H|ψ⟩`. Accepts a `PauliSum`, a single label such as `"ZZ"`, or a
    /// list of `(label, coefficient)` pairs.
    fn expectation(&self, py: Python<'_>, observable: Observable) -> PyResult<f64> {
        let sum = observable.into_sum()?;
        dispatch!(State, &self.state, s => py.detach(|| s.expectation(&sum))).map_err(to_py_err)
    }

    /// `H|ψ⟩` as a new, generally unnormalised, statevector.
    fn apply_pauli_sum(&self, py: Python<'_>, hamiltonian: Observable) -> PyResult<Self> {
        let sum = hamiltonian.into_sum()?;
        let state = convert!(State => State, &self.state, s => {
            py.detach(|| s.apply_pauli_sum(&sum)).map_err(to_py_err)?
        });
        Ok(Self::wrap(state, self.rng.clone()))
    }

    /// Evolve by `exp(-iHt)` exactly (to about machine precision), without
    /// forming the matrix exponential. Needs a complex statevector.
    fn evolve(&mut self, py: Python<'_>, hamiltonian: Observable, time: f64) -> PyResult<()> {
        let sum = hamiltonian.into_sum()?;
        match &mut self.state {
            State::Complex(s) => py.detach(|| s.evolve(&sum, time)).map_err(to_py_err),
            State::Complex32(s) => py.detach(|| s.evolve(&sum, time)).map_err(to_py_err),
            State::Real(_) | State::Real32(_) => Err(PyValueError::new_err(
                "exp(-iHt) is complex; evolve needs dtype='complex' or 'complex64'",
            )),
        }
    }

    /// `⟨ψ(θ)|H|ψ(θ)⟩` and its gradient by adjoint differentiation, where
    /// `ψ(θ)` is `circuit` bound to `parameters` applied to this state.
    /// Returns `(value, gradient)`.
    fn expectation_gradient<'py>(
        &self,
        py: Python<'py>,
        circuit: PyRef<'_, PyCircuit>,
        parameters: Vec<f64>,
        observable: Observable,
    ) -> PyResult<(f64, Bound<'py, PyArray1<f64>>)> {
        let sum = observable.into_sum()?;
        let circ = &circuit.inner;
        let (value, gradient) = dispatch!(State, &self.state, s => py.detach(|| s.expectation_gradient(circ, &parameters, &sum)))
            .map_err(to_py_err)?;
        Ok((value, PyArray1::from_vec(py, gradient)))
    }

    /// Gradient of a real loss L of the output of `circuit`:
    /// `dL/dθ = 2 Re⟨λ|∂ψ/∂θ⟩`. `final_state` is the circuit's output at
    /// `parameters` and `cotangent` is `λ = ∂L/∂ψ*` there, e.g. `H|ψ⟩` for an
    /// expectation value or `|φ⟩⟨φ|ψ⟩` for an overlap `|⟨φ|ψ⟩|²`.
    #[staticmethod]
    fn adjoint_gradient<'py>(
        py: Python<'py>,
        circuit: PyRef<'_, PyCircuit>,
        parameters: Vec<f64>,
        final_state: PyRef<'_, PyStatevector>,
        cotangent: PyRef<'_, PyStatevector>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let circ = &circuit.inner;
        let gradient = same_dtype!(State, &final_state.state, &cotangent.state, (f, c) => {
            py.detach(|| DenseState::adjoint_gradient(circ, &parameters, f, c))
        })
        .map_err(to_py_err)?;
        Ok(PyArray1::from_vec(py, gradient))
    }

    /// Probability that measuring `qubit` gives 1.
    fn probability_one(&self, qubit: usize) -> PyResult<f64> {
        dispatch!(State, &self.state, s => s.probability_one(qubit)).map_err(to_py_err)
    }

    /// Draw `shots` basis-state indices without changing the state.
    fn sample<'py>(&mut self, py: Python<'py>, shots: usize) -> Bound<'py, PyArray1<usize>> {
        let rng = &mut self.rng;
        PyArray1::from_vec(py, dispatch!(State, &self.state, s => s.sample(shots, rng)))
    }

    /// Counts keyed by bitstring, most significant qubit first (Qiskit style).
    fn sample_counts<'py>(
        &mut self,
        py: Python<'py>,
        shots: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let n = self.n_qubits();
        let rng = &mut self.rng;
        let counts = dispatch!(State, &self.state, s => s.sample_counts(shots, rng));
        let dict = PyDict::new(py);
        for (index, count) in counts {
            dict.set_item(format!("{index:0n$b}"), count)?;
        }
        Ok(dict)
    }

    /// Measure `qubit`, collapsing the state. Returns 0 or 1.
    fn measure(&mut self, qubit: usize) -> PyResult<u8> {
        let rng = &mut self.rng;
        let one =
            dispatch!(State, &mut self.state, s => s.measure(qubit, rng)).map_err(to_py_err)?;
        Ok(one as u8)
    }

    /// Measure every qubit, collapsing to a basis state. Returns its index.
    fn measure_all(&mut self) -> usize {
        let rng = &mut self.rng;
        dispatch!(State, &mut self.state, s => s.measure_all(rng))
    }

    /// Measure `qubit` and flip it to `|0⟩` if the outcome was 1.
    fn reset(&mut self, qubit: usize) -> PyResult<()> {
        let rng = &mut self.rng;
        dispatch!(State, &mut self.state, s => s.reset(qubit, rng)).map_err(to_py_err)
    }
}

/// An eigenpair from `lowest_eigenstates`.
#[pyclass(name = "Eigenstate", module = "ketrs", get_all)]
struct PyEigenstate {
    /// The eigenvalue, as the Rayleigh quotient ⟨x|H|x⟩.
    energy: f64,
    /// The normalised eigenvector: a `Statevector`, or a
    /// `SectorStatevector` when solved within a sector.
    state: Py<PyAny>,
    /// ‖Hx − Ex‖, measured directly.
    residual: f64,
}

#[pymethods]
impl PyEigenstate {
    fn __repr__(&self) -> String {
        format!(
            "Eigenstate(energy={}, residual={:.1e})",
            self.energy, self.residual
        )
    }
}

/// The `k` lowest eigenstates of `hamiltonian` on `n_qubits` qubits, in
/// increasing energy, by matrix-free Lanczos iteration with deflation.
/// Degenerate eigenvalues appear once per independent eigenvector.
///
/// With `sector`, solve within that `SectorBasis`, which `hamiltonian` must
/// conserve: far less memory and time when a symmetry applies.
#[pyfunction]
#[pyo3(signature = (hamiltonian, n_qubits, k=1, dtype="complex", tolerance=1e-9, max_iterations=1000, seed=0, sector=None))]
#[allow(clippy::too_many_arguments)]
fn lowest_eigenstates(
    py: Python<'_>,
    hamiltonian: Observable,
    n_qubits: usize,
    k: usize,
    dtype: &str,
    tolerance: f64,
    max_iterations: usize,
    seed: u64,
    sector: Option<PyRef<'_, PySectorBasis>>,
) -> PyResult<Vec<PyEigenstate>> {
    let sum = hamiltonian.into_sum()?;
    let options = EigenOptions {
        tolerance,
        max_iterations,
        seed,
    };
    if let Some(sector) = sector {
        let basis = sector.inner.clone();
        if basis.n_qubits() != n_qubits {
            return Err(PyValueError::new_err(
                "sector has a different number of qubits",
            ));
        }
        let found = by_dtype!(SectorKind, DType::parse(dtype)?, T => py
            .detach(|| SectorState::<T>::lowest_eigenstates(basis, &sum, k, &options))
            .map_err(to_py_err)?);
        let found: Vec<(f64, f64, PySectorStatevector)> = sector::eigenstates(found);
        return found
            .into_iter()
            .map(|(energy, residual, state)| {
                Ok(PyEigenstate {
                    energy,
                    residual,
                    state: Py::new(py, state)?.into_any(),
                })
            })
            .collect();
    }
    let found = match DType::parse(dtype)? {
        DType::Complex => eigenstates(py, &sum, n_qubits, k, &options, State::Complex)?,
        DType::Real => eigenstates(py, &sum, n_qubits, k, &options, State::Real)?,
        DType::Complex32 => eigenstates(py, &sum, n_qubits, k, &options, State::Complex32)?,
        DType::Real32 => eigenstates(py, &sum, n_qubits, k, &options, State::Real32)?,
    };
    found
        .into_iter()
        .map(|(energy, residual, state)| {
            Ok(PyEigenstate {
                energy,
                residual,
                state: Py::new(py, PyStatevector::wrap(state, rng_from_seed(None)))?.into_any(),
            })
        })
        .collect()
}

/// `DenseState::<T>::lowest_eigenstates`, each state wrapped by `wrap`.
fn eigenstates<T: Scalar>(
    py: Python<'_>,
    sum: &PauliSum,
    n_qubits: usize,
    k: usize,
    options: &EigenOptions,
    wrap: fn(DenseState<T>) -> State,
) -> PyResult<Vec<(f64, f64, State)>> {
    Ok(py
        .detach(|| DenseState::<T>::lowest_eigenstates(sum, n_qubits, k, options))
        .map_err(to_py_err)?
        .into_iter()
        .map(|e| (e.energy, e.residual, wrap(e.state)))
        .collect())
}

/// Read an OpenQASM 3 (or 2) program into a `Circuit`: registers, inputs
/// (as parameters, in declaration order), the standard gates, `gate`
/// definitions and `def` subroutines, `inv`/`ctrl`/`negctrl`/`pow`
/// modifiers, measurement, reset, `if`, `while` and `for`. Errors name the
/// line.
#[pyfunction]
fn from_qasm(source: &str) -> PyResult<PyCircuit> {
    Ok(PyCircuit {
        inner: ketrs::qasm::parse(source).map_err(to_py_err)?,
    })
}

/// The name of the GPU ketrs would use, or None: this build has no GPU
/// support (build with the `gpu` feature to enable it).
#[cfg(not(feature = "gpu"))]
#[pyfunction]
fn gpu_device() -> Option<String> {
    None
}

/// The quantum Fourier transform, built as in Qiskit's `QFT`.
#[pyfunction]
fn qft(n_qubits: usize) -> PyCircuit {
    PyCircuit {
        inner: ketrs::library::qft(n_qubits),
    }
}

/// Prepares `(|0…0⟩ + |1…1⟩)/√2`.
#[pyfunction]
fn ghz(n_qubits: usize) -> PyCircuit {
    PyCircuit {
        inner: ketrs::library::ghz(n_qubits),
    }
}

/// Safe without the GIL: there's no shared mutable state outside objects
/// (whose borrows PyO3 checks), and `GpuStatevector` stays on its thread.
#[pymodule(gil_used = false)]
fn _ketrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<PyCircuit>()?;
    m.add_class::<PyParameter>()?;
    m.add_class::<PyPauliSum>()?;
    m.add_class::<PyStatevector>()?;
    m.add_class::<PyEigenstate>()?;
    m.add_class::<PySectorBasis>()?;
    m.add_class::<PySectorStatevector>()?;
    mps::register(m)?;
    density::register(m)?;
    #[cfg(feature = "gpu")]
    gpu::register(m)?;
    #[cfg(not(feature = "gpu"))]
    m.add_function(wrap_pyfunction!(gpu_device, m)?)?;
    m.add_function(wrap_pyfunction!(lowest_eigenstates, m)?)?;
    m.add_function(wrap_pyfunction!(from_qasm, m)?)?;
    m.add_function(wrap_pyfunction!(qft, m)?)?;
    m.add_function(wrap_pyfunction!(ghz, m)?)?;
    Ok(())
}
