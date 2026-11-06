//! OpenQASM 3 import (and the OpenQASM 2 forms still in common use).
//!
//! [`parse`] reads a program into a [`Circuit`]:
//!
//! - registers: `qubit[n] q;`, `qubit q;`, `bit[n] c;`, `bit c;`, and
//!   OpenQASM 2's `qreg`/`creg`, laid out in declaration order;
//! - `input float[64] θ;` or `input angle θ;`: circuit parameters, in
//!   declaration order, usable in gate angles linear in one of them;
//! - every gate of `stdgates.inc` and OpenQASM 2's `qelib1.inc` up to two
//!   controls (`ccx`, `cswap`; not `c3x` and the like), gates
//!   defined with `gate`, the modifiers `inv @`, `ctrl @` (one control on a
//!   one-qubit gate) and `pow(k) @` for integer `k`, and broadcasting over
//!   whole registers;
//! - `measure` in both syntaxes, `reset`, `barrier` (ignored), `gphase`;
//! - `if`/`else` and `while` on classical expressions of bits and
//!   registers, and `for` loops over ranges or sets, unrolled.
//!
//! Angles may use `pi`, `π`, `tau`, `euler`, arithmetic and the usual
//! functions. Other statements (`def`, `let`, `switch`, timing, …) are
//! errors naming the line.

use crate::circuit::Circuit;
use crate::classical::{BinaryOp, Expr};
use crate::error::Error;
use crate::gate::Gate;
use crate::parameter::{Angle, Parameter};
use num_complex::Complex64 as C64;
use std::collections::HashMap;
use std::f64::consts::{E, PI, TAU};

/// Read an OpenQASM 3 (or 2) program. See the [module docs](self) for what
/// is supported.
pub fn parse(source: &str) -> Result<Circuit, Error> {
    let tokens = lex(source)?;
    let program = Parser { tokens, pos: 0 }.program()?;
    Lowering::new(&program)?.run(&program)
}

fn error(line: usize, message: impl Into<String>) -> Error {
    Error::Parse {
        line,
        message: message.into(),
    }
}

// ---------------------------------------------------------------- lexing

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Ident(String),
    Int(u64),
    Float(f64),
    Str,
    Sym(&'static str),
    Eof,
}

#[derive(Clone, Debug)]
struct Token {
    tok: Tok,
    line: usize,
}

const SYMBOLS: [&str; 33] = [
    "->", "==", "!=", "<=", ">=", "&&", "||", "**", "<<", ">>", "++", ";", ",", "[", "]", "(", ")",
    "{", "}", ":", "=", "<", ">", "!", "~", "&", "|", "^", "+", "-", "*", "/", "@",
];

fn lex(source: &str) -> Result<Vec<Token>, Error> {
    let chars: Vec<char> = source.chars().collect();
    let mut tokens = Vec::new();
    let (mut i, mut line) = (0, 1);
    while i < chars.len() {
        let c = chars[i];
        if c == '\n' {
            line += 1;
            i += 1;
        } else if c.is_whitespace() {
            i += 1;
        } else if c == '/' && chars.get(i + 1) == Some(&'/') {
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
        } else if c == '/' && chars.get(i + 1) == Some(&'*') {
            i += 2;
            while i < chars.len() && !(chars[i] == '*' && chars.get(i + 1) == Some(&'/')) {
                line += (chars[i] == '\n') as usize;
                i += 1;
            }
            i += 2;
        } else if c == '"' {
            i += 1;
            while i < chars.len() && chars[i] != '"' {
                i += 1;
            }
            i += 1;
            tokens.push(Token {
                tok: Tok::Str,
                line,
            });
        } else if c.is_ascii_digit()
            || (c == '.' && chars.get(i + 1).is_some_and(|d| d.is_ascii_digit()))
        {
            let start = i;
            let mut float = false;
            while i < chars.len() {
                let d = chars[i];
                if d.is_ascii_digit() || d == '_' {
                    i += 1;
                } else if d == '.' {
                    float = true;
                    i += 1;
                } else if (d == 'e' || d == 'E')
                    && chars
                        .get(i + 1)
                        .is_some_and(|n| n.is_ascii_digit() || *n == '-' || *n == '+')
                {
                    float = true;
                    i += 2;
                } else {
                    break;
                }
            }
            let text: String = chars[start..i].iter().filter(|&&d| d != '_').collect();
            let tok = if float {
                Tok::Float(
                    text.parse()
                        .map_err(|_| error(line, format!("bad number {text}")))?,
                )
            } else {
                Tok::Int(
                    text.parse()
                        .map_err(|_| error(line, format!("bad number {text}")))?,
                )
            };
            tokens.push(Token { tok, line });
        } else if c.is_alphabetic() || c == '_' || c == '$' {
            let start = i;
            while i < chars.len()
                && (chars[i].is_alphanumeric() || chars[i] == '_' || chars[i] == '$')
            {
                i += 1;
            }
            tokens.push(Token {
                tok: Tok::Ident(chars[start..i].iter().collect()),
                line,
            });
        } else {
            let rest: String = chars[i..chars.len().min(i + 2)].iter().collect();
            let sym = SYMBOLS
                .iter()
                .find(|s| rest.starts_with(**s))
                .ok_or_else(|| error(line, format!("unexpected character {c:?}")))?;
            i += sym.chars().count();
            tokens.push(Token {
                tok: Tok::Sym(sym),
                line,
            });
        }
    }
    tokens.push(Token {
        tok: Tok::Eof,
        line,
    });
    Ok(tokens)
}

// ---------------------------------------------------------------- syntax

#[derive(Clone, Debug)]
enum Ex {
    Num(f64),
    Bool(bool),
    Ident(String),
    Index(String, Box<Ex>),
    Call(String, Vec<Ex>),
    Neg(Box<Ex>),
    Not(Box<Ex>),
    BitNot(Box<Ex>),
    Bin(Op, Box<Ex>, Box<Ex>),
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Op {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    And,
    Or,
    BitAnd,
    BitOr,
    BitXor,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Clone, Debug)]
struct Operand {
    name: String,
    index: Option<Ex>,
}

#[derive(Clone, Debug)]
enum Modifier {
    Inv,
    Ctrl(usize),
    Pow(Ex),
}

#[derive(Clone, Debug)]
enum Values {
    Range(Ex, Option<Ex>, Ex),
    Set(Vec<Ex>),
}

#[derive(Clone, Debug)]
enum Kind {
    Qubits {
        name: String,
        size: Option<Ex>,
    },
    Bits {
        name: String,
        size: Option<Ex>,
    },
    Input {
        name: String,
    },
    GateDef(GateDef),
    Call {
        name: String,
        modifiers: Vec<Modifier>,
        args: Vec<Ex>,
        operands: Vec<Operand>,
    },
    Measure {
        source: Operand,
        target: Option<Operand>,
    },
    Reset(Vec<Operand>),
    If {
        condition: Ex,
        then: Vec<Stmt>,
        otherwise: Vec<Stmt>,
    },
    While {
        condition: Ex,
        body: Vec<Stmt>,
    },
    For {
        var: String,
        values: Values,
        body: Vec<Stmt>,
    },
    GlobalPhase(Ex),
    Nothing,
}

#[derive(Clone, Debug)]
struct GateDef {
    name: String,
    params: Vec<String>,
    qubits: Vec<String>,
    body: Vec<Stmt>,
}

#[derive(Clone, Debug)]
struct Stmt {
    kind: Kind,
    line: usize,
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> &Tok {
        &self.tokens[self.pos].tok
    }

    fn peek_at(&self, k: usize) -> &Tok {
        &self.tokens[(self.pos + k).min(self.tokens.len() - 1)].tok
    }

    fn line(&self) -> usize {
        self.tokens[self.pos].line
    }

    fn next(&mut self) -> Tok {
        let tok = self.tokens[self.pos].tok.clone();
        if self.pos + 1 < self.tokens.len() {
            self.pos += 1;
        }
        tok
    }

    fn is_sym(&self, s: &str) -> bool {
        matches!(self.peek(), Tok::Sym(t) if *t == s)
    }

    fn is_word(&self, w: &str) -> bool {
        matches!(self.peek(), Tok::Ident(t) if t == w)
    }

    fn eat_sym(&mut self, s: &str) -> bool {
        let found = self.is_sym(s);
        if found {
            self.next();
        }
        found
    }

    fn expect_sym(&mut self, s: &str) -> Result<(), Error> {
        if self.eat_sym(s) {
            Ok(())
        } else {
            Err(error(
                self.line(),
                format!("expected `{s}`, found {}", describe(self.peek())),
            ))
        }
    }

    fn ident(&mut self) -> Result<String, Error> {
        match self.next() {
            Tok::Ident(name) => Ok(name),
            other => Err(error(
                self.line(),
                format!("expected a name, found {}", describe(&other)),
            )),
        }
    }

    fn program(mut self) -> Result<Vec<Stmt>, Error> {
        let mut stmts = Vec::new();
        while *self.peek() != Tok::Eof {
            stmts.push(self.statement()?);
        }
        Ok(stmts)
    }

    /// A braced block, or a single statement.
    fn block(&mut self) -> Result<Vec<Stmt>, Error> {
        if self.eat_sym("{") {
            let mut stmts = Vec::new();
            while !self.eat_sym("}") {
                if *self.peek() == Tok::Eof {
                    return Err(error(self.line(), "unclosed `{`"));
                }
                stmts.push(self.statement()?);
            }
            Ok(stmts)
        } else {
            Ok(vec![self.statement()?])
        }
    }

    fn skip_to_semicolon(&mut self) {
        while !self.eat_sym(";") && *self.peek() != Tok::Eof {
            self.next();
        }
    }

    fn statement(&mut self) -> Result<Stmt, Error> {
        let line = self.line();
        let kind = self.statement_kind()?;
        Ok(Stmt { kind, line })
    }

    fn statement_kind(&mut self) -> Result<Kind, Error> {
        let word = match self.peek() {
            Tok::Ident(w) => w.clone(),
            other => {
                return Err(error(
                    self.line(),
                    format!("unexpected {}", describe(other)),
                ));
            }
        };
        match word.as_str() {
            "OPENQASM" | "include" | "barrier" => {
                self.skip_to_semicolon();
                Ok(Kind::Nothing)
            }
            "qubit" | "bit" => {
                self.next();
                let size = if self.eat_sym("[") {
                    let size = self.expr()?;
                    self.expect_sym("]")?;
                    Some(size)
                } else {
                    None
                };
                let name = self.ident()?;
                if word == "bit" && self.eat_sym("=") {
                    return Err(error(self.line(), "initialised bits are not supported"));
                }
                self.expect_sym(";")?;
                Ok(if word == "qubit" {
                    Kind::Qubits { name, size }
                } else {
                    Kind::Bits { name, size }
                })
            }
            "qreg" | "creg" => {
                self.next();
                let name = self.ident()?;
                self.expect_sym("[")?;
                let size = Some(self.expr()?);
                self.expect_sym("]")?;
                self.expect_sym(";")?;
                Ok(if word == "qreg" {
                    Kind::Qubits { name, size }
                } else {
                    Kind::Bits { name, size }
                })
            }
            "input" => {
                self.next();
                let ty = self.ident()?;
                if ty != "float" && ty != "angle" {
                    return Err(error(
                        self.line(),
                        format!("inputs of type {ty} are not supported; use float or angle"),
                    ));
                }
                if self.eat_sym("[") {
                    self.expr()?;
                    self.expect_sym("]")?;
                }
                let name = self.ident()?;
                self.expect_sym(";")?;
                Ok(Kind::Input { name })
            }
            "gate" => {
                self.next();
                let name = self.ident()?;
                let mut params = Vec::new();
                if self.eat_sym("(") {
                    while !self.eat_sym(")") {
                        params.push(self.ident()?);
                        self.eat_sym(",");
                    }
                }
                let mut qubits = vec![self.ident()?];
                while self.eat_sym(",") {
                    qubits.push(self.ident()?);
                }
                if !self.is_sym("{") {
                    return Err(error(self.line(), "expected the gate body"));
                }
                let body = self.block()?;
                Ok(Kind::GateDef(GateDef {
                    name,
                    params,
                    qubits,
                    body,
                }))
            }
            "measure" => {
                self.next();
                let source = self.operand()?;
                let target = if self.eat_sym("->") {
                    Some(self.operand()?)
                } else {
                    None
                };
                self.expect_sym(";")?;
                Ok(Kind::Measure { source, target })
            }
            "reset" => {
                self.next();
                let operands = self.operands()?;
                self.expect_sym(";")?;
                Ok(Kind::Reset(operands))
            }
            "if" => {
                self.next();
                self.expect_sym("(")?;
                let condition = self.expr()?;
                self.expect_sym(")")?;
                let then = self.block()?;
                let otherwise = if self.is_word("else") {
                    self.next();
                    self.block()?
                } else {
                    Vec::new()
                };
                Ok(Kind::If {
                    condition,
                    then,
                    otherwise,
                })
            }
            "while" => {
                self.next();
                self.expect_sym("(")?;
                let condition = self.expr()?;
                self.expect_sym(")")?;
                let body = self.block()?;
                Ok(Kind::While { condition, body })
            }
            "for" => {
                self.next();
                // An optional type before the variable: `for uint i in …`.
                if matches!(self.peek_at(1), Tok::Ident(w) if w != "in")
                    || matches!(self.peek_at(1), Tok::Sym("["))
                {
                    self.ident()?;
                    if self.eat_sym("[") {
                        self.expr()?;
                        self.expect_sym("]")?;
                    }
                }
                let var = self.ident()?;
                if !self.is_word("in") {
                    return Err(error(self.line(), "expected `in`"));
                }
                self.next();
                let values = if self.eat_sym("[") {
                    let start = self.expr()?;
                    self.expect_sym(":")?;
                    let second = self.expr()?;
                    let values = if self.eat_sym(":") {
                        Values::Range(start, Some(second), self.expr()?)
                    } else {
                        Values::Range(start, None, second)
                    };
                    self.expect_sym("]")?;
                    values
                } else if self.eat_sym("{") {
                    let mut set = Vec::new();
                    while !self.eat_sym("}") {
                        set.push(self.expr()?);
                        self.eat_sym(",");
                    }
                    Values::Set(set)
                } else {
                    return Err(error(self.line(), "expected a range `[a:b]` or set `{…}`"));
                };
                let body = self.block()?;
                Ok(Kind::For { var, values, body })
            }
            "gphase" => {
                self.next();
                self.expect_sym("(")?;
                let phase = self.expr()?;
                self.expect_sym(")")?;
                self.expect_sym(";")?;
                Ok(Kind::GlobalPhase(phase))
            }
            "def" | "let" | "const" | "switch" | "box" | "delay" | "defcal" | "cal" | "extern"
            | "return" | "break" | "continue" | "output" | "int" | "uint" | "float" | "angle"
            | "bool" | "complex" | "stretch" | "duration" => Err(error(
                self.line(),
                format!("`{word}` statements are not supported"),
            )),
            _ => {
                // An assignment from a measurement, or a gate call.
                if matches!(self.peek_at(1), Tok::Sym("=" | "[")) && self.assignment_ahead() {
                    let target = self.operand()?;
                    self.expect_sym("=")?;
                    if !self.is_word("measure") {
                        return Err(error(self.line(), "only measurements can be assigned"));
                    }
                    self.next();
                    let source = self.operand()?;
                    self.expect_sym(";")?;
                    return Ok(Kind::Measure {
                        source,
                        target: Some(target),
                    });
                }
                self.call()
            }
        }
    }

    /// Whether the statement ahead is `operand = …`.
    fn assignment_ahead(&self) -> bool {
        let mut k = 1;
        if matches!(self.peek_at(k), Tok::Sym("[")) {
            let mut depth = 0;
            loop {
                match self.peek_at(k) {
                    Tok::Sym("[") => depth += 1,
                    Tok::Sym("]") => {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                    }
                    Tok::Eof | Tok::Sym(";") => return false,
                    _ => {}
                }
                k += 1;
            }
            k += 1;
        }
        matches!(self.peek_at(k), Tok::Sym("="))
    }

    fn call(&mut self) -> Result<Kind, Error> {
        let mut modifiers = Vec::new();
        loop {
            if self.is_word("inv") && matches!(self.peek_at(1), Tok::Sym("@")) {
                self.next();
                self.next();
                modifiers.push(Modifier::Inv);
            } else if self.is_word("ctrl") && matches!(self.peek_at(1), Tok::Sym("@" | "(")) {
                self.next();
                let n = if self.eat_sym("(") {
                    let n = self.expr()?;
                    self.expect_sym(")")?;
                    match n {
                        Ex::Num(v) if v >= 1.0 && v.fract() == 0.0 => v as usize,
                        _ => return Err(error(self.line(), "ctrl takes a positive integer")),
                    }
                } else {
                    1
                };
                self.expect_sym("@")?;
                modifiers.push(Modifier::Ctrl(n));
            } else if self.is_word("pow") && matches!(self.peek_at(1), Tok::Sym("(")) {
                self.next();
                self.next();
                let k = self.expr()?;
                self.expect_sym(")")?;
                self.expect_sym("@")?;
                modifiers.push(Modifier::Pow(k));
            } else if self.is_word("negctrl") {
                return Err(error(self.line(), "negctrl is not supported"));
            } else {
                break;
            }
        }
        let name = self.ident()?;
        let mut args = Vec::new();
        if self.eat_sym("(") {
            while !self.eat_sym(")") {
                args.push(self.expr()?);
                self.eat_sym(",");
            }
        }
        let operands = self.operands()?;
        self.expect_sym(";")?;
        Ok(Kind::Call {
            name,
            modifiers,
            args,
            operands,
        })
    }

    fn operand(&mut self) -> Result<Operand, Error> {
        let name = self.ident()?;
        let index = if self.eat_sym("[") {
            let index = self.expr()?;
            self.expect_sym("]")?;
            Some(index)
        } else {
            None
        };
        Ok(Operand { name, index })
    }

    fn operands(&mut self) -> Result<Vec<Operand>, Error> {
        let mut operands = vec![self.operand()?];
        while self.eat_sym(",") {
            operands.push(self.operand()?);
        }
        Ok(operands)
    }

    fn expr(&mut self) -> Result<Ex, Error> {
        self.binary(0)
    }

    /// Precedence climbing over the binary operators.
    fn binary(&mut self, level: usize) -> Result<Ex, Error> {
        const LEVELS: [&[(&str, Op)]; 8] = [
            &[("||", Op::Or)],
            &[("&&", Op::And)],
            &[("|", Op::BitOr)],
            &[("^", Op::BitXor)],
            &[("&", Op::BitAnd)],
            &[("==", Op::Eq), ("!=", Op::Ne)],
            &[("<", Op::Lt), ("<=", Op::Le), (">", Op::Gt), (">=", Op::Ge)],
            &[("+", Op::Add), ("-", Op::Sub)],
        ];
        if level == LEVELS.len() {
            return self.product();
        }
        let mut left = self.binary(level + 1)?;
        'outer: loop {
            for &(sym, op) in LEVELS[level] {
                if self.eat_sym(sym) {
                    let right = self.binary(level + 1)?;
                    left = Ex::Bin(op, Box::new(left), Box::new(right));
                    continue 'outer;
                }
            }
            return Ok(left);
        }
    }

    fn product(&mut self) -> Result<Ex, Error> {
        let mut left = self.unary()?;
        loop {
            let op = if self.eat_sym("*") {
                Op::Mul
            } else if self.eat_sym("/") {
                Op::Div
            } else {
                return Ok(left);
            };
            left = Ex::Bin(op, Box::new(left), Box::new(self.unary()?));
        }
    }

    fn unary(&mut self) -> Result<Ex, Error> {
        if self.eat_sym("-") {
            Ok(Ex::Neg(Box::new(self.unary()?)))
        } else if self.eat_sym("+") {
            self.unary()
        } else if self.eat_sym("!") {
            Ok(Ex::Not(Box::new(self.unary()?)))
        } else if self.eat_sym("~") {
            Ok(Ex::BitNot(Box::new(self.unary()?)))
        } else {
            let base = self.primary()?;
            if self.eat_sym("**") {
                Ok(Ex::Bin(Op::Pow, Box::new(base), Box::new(self.unary()?)))
            } else {
                Ok(base)
            }
        }
    }

    fn primary(&mut self) -> Result<Ex, Error> {
        let line = self.line();
        match self.next() {
            Tok::Int(v) => Ok(Ex::Num(v as f64)),
            Tok::Float(v) => Ok(Ex::Num(v)),
            Tok::Sym("(") => {
                let e = self.expr()?;
                self.expect_sym(")")?;
                Ok(e)
            }
            Tok::Ident(name) => match name.as_str() {
                "true" => Ok(Ex::Bool(true)),
                "false" => Ok(Ex::Bool(false)),
                _ if self.eat_sym("[") => {
                    let index = self.expr()?;
                    self.expect_sym("]")?;
                    Ok(Ex::Index(name, Box::new(index)))
                }
                _ if self.eat_sym("(") => {
                    let mut args = Vec::new();
                    while !self.eat_sym(")") {
                        args.push(self.expr()?);
                        self.eat_sym(",");
                    }
                    Ok(Ex::Call(name, args))
                }
                _ => Ok(Ex::Ident(name)),
            },
            other => Err(error(
                line,
                format!("expected an expression, found {}", describe(&other)),
            )),
        }
    }
}

fn describe(tok: &Tok) -> String {
    match tok {
        Tok::Ident(name) => format!("`{name}`"),
        Tok::Int(v) => format!("`{v}`"),
        Tok::Float(v) => format!("`{v}`"),
        Tok::Str => "a string".into(),
        Tok::Sym(s) => format!("`{s}`"),
        Tok::Eof => "the end of the program".into(),
    }
}

// ---------------------------------------------------------------- lowering

/// A register: its first index and size, and whether it was declared as
/// an array (so a bare name means every element).
#[derive(Clone, Copy)]
struct Register {
    start: usize,
    size: usize,
    array: bool,
}

/// What names mean inside a gate body or loop.
#[derive(Clone, Default)]
struct Scope {
    values: HashMap<String, Angle>,
    qubits: HashMap<String, usize>,
    /// Inside a block, loop or gate body, where declarations aren't allowed.
    nested: bool,
}

impl Scope {
    fn nested(&self) -> Scope {
        Scope {
            nested: true,
            ..self.clone()
        }
    }
}

struct Lowering {
    n_qubits: usize,
    n_clbits: usize,
    qregs: HashMap<String, Register>,
    cregs: HashMap<String, Register>,
    inputs: HashMap<String, Parameter>,
    n_parameters: usize,
    gates: HashMap<String, GateDef>,
}

impl Lowering {
    /// Lay out every declaration (all at the top level).
    fn new(program: &[Stmt]) -> Result<Self, Error> {
        let mut l = Lowering {
            n_qubits: 0,
            n_clbits: 0,
            qregs: HashMap::new(),
            cregs: HashMap::new(),
            inputs: HashMap::new(),
            n_parameters: 0,
            gates: HashMap::new(),
        };
        for stmt in program {
            match &stmt.kind {
                Kind::Qubits { name, size } | Kind::Bits { name, size } => {
                    let n = match size {
                        Some(size) => l.int(size, &Scope::default(), stmt.line)? as usize,
                        None => 1,
                    };
                    let quantum = matches!(stmt.kind, Kind::Qubits { .. });
                    let (count, registers) = if quantum {
                        (&mut l.n_qubits, &mut l.qregs)
                    } else {
                        (&mut l.n_clbits, &mut l.cregs)
                    };
                    let register = Register {
                        start: *count,
                        size: n,
                        array: size.is_some(),
                    };
                    if registers.insert(name.clone(), register).is_some() {
                        return Err(error(stmt.line, format!("`{name}` declared twice")));
                    }
                    *count += n;
                }
                Kind::Input { name } => {
                    l.inputs.insert(name.clone(), Parameter(l.n_parameters));
                    l.n_parameters += 1;
                }
                Kind::GateDef(def) => {
                    l.gates.insert(def.name.clone(), def.clone());
                }
                _ => {}
            }
        }
        Ok(l)
    }

    /// A fresh circuit on every qubit and bit, with every parameter, as
    /// control-flow blocks need.
    fn empty(&self) -> Circuit {
        let mut circ = Circuit::with_clbits(self.n_qubits, self.n_clbits);
        circ.parameters(self.n_parameters);
        circ
    }

    fn run(&self, program: &[Stmt]) -> Result<Circuit, Error> {
        let mut circ = self.empty();
        self.statements(&mut circ, program, &Scope::default())?;
        Ok(circ)
    }

    fn statements(&self, circ: &mut Circuit, stmts: &[Stmt], scope: &Scope) -> Result<(), Error> {
        for stmt in stmts {
            self.statement(circ, stmt, scope)?;
        }
        Ok(())
    }

    fn statement(&self, circ: &mut Circuit, stmt: &Stmt, scope: &Scope) -> Result<(), Error> {
        let line = stmt.line;
        let instruction =
            |r: Result<&mut Circuit, Error>| r.map(|_| ()).map_err(|e| error(line, e.to_string()));
        match &stmt.kind {
            Kind::Nothing | Kind::Input { .. } | Kind::GateDef(_) => Ok(()),
            Kind::Qubits { .. } | Kind::Bits { .. } if scope.nested => {
                Err(error(line, "declarations must be at the top level"))
            }
            Kind::Qubits { .. } | Kind::Bits { .. } => Ok(()),
            Kind::Call {
                name,
                modifiers,
                args,
                operands,
            } => {
                let args = args
                    .iter()
                    .map(|a| self.angle(a, scope, line))
                    .collect::<Result<Vec<_>, _>>()?;
                let groups = self.broadcast(operands, scope, line)?;
                for qubits in groups {
                    self.call(circ, name, modifiers, &args, &qubits, scope, line)?;
                }
                Ok(())
            }
            Kind::Measure { source, target } => {
                let qubits = self.qubits(source, scope, line)?;
                let clbits = match target {
                    Some(target) => self.clbits(target, scope, line)?,
                    None => return Err(error(line, "a measurement needs a classical target")),
                };
                if qubits.len() != clbits.len() {
                    return Err(error(line, "measured registers have different sizes"));
                }
                for (q, c) in qubits.into_iter().zip(clbits) {
                    instruction(
                        circ.push_instruction(crate::Instruction::Measure { qubit: q, clbit: c }),
                    )?;
                }
                Ok(())
            }
            Kind::Reset(operands) => {
                for operand in operands {
                    for q in self.qubits(operand, scope, line)? {
                        instruction(circ.push_instruction(crate::Instruction::Reset { qubit: q }))?;
                    }
                }
                Ok(())
            }
            Kind::If {
                condition,
                then,
                otherwise,
            } => {
                let condition = self.condition(condition, scope, line)?;
                let mut then_circ = self.empty();
                self.statements(&mut then_circ, then, &scope.nested())?;
                let mut else_circ = self.empty();
                self.statements(&mut else_circ, otherwise, &scope.nested())?;
                instruction(circ.push_instruction(crate::Instruction::IfElse {
                    condition,
                    then: Box::new(then_circ),
                    otherwise: Box::new(else_circ),
                }))
            }
            Kind::While { condition, body } => {
                let condition = self.condition(condition, scope, line)?;
                let mut body_circ = self.empty();
                self.statements(&mut body_circ, body, &scope.nested())?;
                instruction(circ.push_instruction(crate::Instruction::While {
                    condition,
                    body: Box::new(body_circ),
                }))
            }
            Kind::For { var, values, body } => {
                let values: Vec<f64> = match values {
                    Values::Set(set) => set
                        .iter()
                        .map(|e| self.int(e, scope, line).map(|v| v as f64))
                        .collect::<Result<_, _>>()?,
                    Values::Range(start, step, stop) => {
                        let (a, b) = (self.int(start, scope, line)?, self.int(stop, scope, line)?);
                        let step = match step {
                            Some(s) => self.int(s, scope, line)?,
                            None => 1,
                        };
                        if step == 0 {
                            return Err(error(line, "a range step can't be zero"));
                        }
                        // OpenQASM ranges include their end.
                        let mut values = Vec::new();
                        let mut v = a;
                        while (step > 0 && v <= b) || (step < 0 && v >= b) {
                            values.push(v as f64);
                            v += step;
                        }
                        values
                    }
                };
                for v in values {
                    let mut inner = scope.nested();
                    inner.values.insert(var.clone(), Angle::Value(v));
                    self.statements(circ, body, &inner)?;
                }
                Ok(())
            }
            Kind::GlobalPhase(phase) => match self.angle(phase, scope, line)? {
                Angle::Value(v) => {
                    let total = circ.global_phase() + v;
                    circ.set_global_phase(total);
                    Ok(())
                }
                _ => Err(error(line, "gphase must not depend on a parameter")),
            },
        }
    }

    /// The qubit lists of a gate call: whole registers broadcast together.
    fn broadcast(
        &self,
        operands: &[Operand],
        scope: &Scope,
        line: usize,
    ) -> Result<Vec<Vec<usize>>, Error> {
        let resolved = operands
            .iter()
            .map(|o| self.qubits(o, scope, line))
            .collect::<Result<Vec<_>, _>>()?;
        let n = resolved.iter().map(Vec::len).max().unwrap_or(0);
        if resolved.iter().any(|r| r.len() != 1 && r.len() != n) {
            return Err(error(line, "broadcast registers have different sizes"));
        }
        Ok((0..n)
            .map(|k| {
                resolved
                    .iter()
                    .map(|r| if r.len() == 1 { r[0] } else { r[k] })
                    .collect()
            })
            .collect())
    }

    fn register(
        &self,
        registers: &HashMap<String, Register>,
        operand: &Operand,
        scope: &Scope,
        line: usize,
        kind: &str,
    ) -> Result<Vec<usize>, Error> {
        let register = registers
            .get(&operand.name)
            .ok_or_else(|| error(line, format!("unknown {kind} `{}`", operand.name)))?;
        match &operand.index {
            Some(index) => {
                let k = self.int(index, scope, line)?;
                if k < 0 || k as usize >= register.size {
                    return Err(error(
                        line,
                        format!("index {k} out of range for `{}`", operand.name),
                    ));
                }
                Ok(vec![register.start + k as usize])
            }
            None => Ok((register.start..register.start + register.size).collect()),
        }
    }

    fn qubits(&self, operand: &Operand, scope: &Scope, line: usize) -> Result<Vec<usize>, Error> {
        if let Some(&q) = scope.qubits.get(&operand.name) {
            if operand.index.is_some() {
                return Err(error(line, "gate arguments can't be indexed"));
            }
            return Ok(vec![q]);
        }
        self.register(&self.qregs, operand, scope, line, "qubit register")
    }

    fn clbits(&self, operand: &Operand, scope: &Scope, line: usize) -> Result<Vec<usize>, Error> {
        self.register(&self.cregs, operand, scope, line, "bit register")
    }

    /// An angle: a number, or linear in one input parameter.
    fn angle(&self, e: &Ex, scope: &Scope, line: usize) -> Result<Angle, Error> {
        let constant = |a: Angle| match a {
            Angle::Value(v) => Ok(v),
            _ => Err(error(line, "angle is not linear in one parameter")),
        };
        Ok(match e {
            Ex::Num(v) => Angle::Value(*v),
            Ex::Bool(b) => Angle::Value(*b as u8 as f64),
            Ex::Ident(name) => match name.as_str() {
                _ if scope.values.contains_key(name) => scope.values[name],
                _ if self.inputs.contains_key(name) => Angle::from(self.inputs[name]),
                "pi" | "π" => Angle::Value(PI),
                "tau" | "τ" => Angle::Value(TAU),
                "euler" | "ℇ" => Angle::Value(E),
                _ => return Err(error(line, format!("unknown name `{name}`"))),
            },
            Ex::Neg(a) => -self.angle(a, scope, line)?,
            Ex::Bin(op, a, b) => {
                let (a, b) = (self.angle(a, scope, line)?, self.angle(b, scope, line)?);
                match op {
                    Op::Add | Op::Sub => {
                        let b = if *op == Op::Sub { -b } else { b };
                        match (a, b) {
                            (x, Angle::Value(v)) | (Angle::Value(v), x) => x + v,
                            (
                                Angle::Linear {
                                    parameter: p,
                                    scale: s,
                                    offset: o,
                                },
                                Angle::Linear {
                                    parameter: q,
                                    scale: t,
                                    offset: u,
                                },
                            ) if p == q => Angle::Linear {
                                parameter: p,
                                scale: s + t,
                                offset: o + u,
                            },
                            _ => return Err(error(line, "angle is not linear in one parameter")),
                        }
                    }
                    Op::Mul => match (a, b) {
                        (x, Angle::Value(v)) | (Angle::Value(v), x) => x * v,
                        _ => return Err(error(line, "angle is not linear in one parameter")),
                    },
                    Op::Div => a * (1.0 / constant(b)?),
                    Op::Pow => Angle::Value(constant(a)?.powf(constant(b)?)),
                    _ => return Err(error(line, "expected an arithmetic expression")),
                }
            }
            Ex::Call(f, args) => {
                let [x] = args.as_slice() else {
                    return Err(error(line, format!("`{f}` takes one argument")));
                };
                let x = constant(self.angle(x, scope, line)?)?;
                Angle::Value(match f.as_str() {
                    "sin" => x.sin(),
                    "cos" => x.cos(),
                    "tan" => x.tan(),
                    "arcsin" | "asin" => x.asin(),
                    "arccos" | "acos" => x.acos(),
                    "arctan" | "atan" => x.atan(),
                    "exp" => x.exp(),
                    "ln" | "log" => x.ln(),
                    "sqrt" => x.sqrt(),
                    _ => return Err(error(line, format!("unknown function `{f}`"))),
                })
            }
            Ex::Index(..) | Ex::Not(_) | Ex::BitNot(_) => {
                return Err(error(line, "expected a number"));
            }
        })
    }

    fn int(&self, e: &Ex, scope: &Scope, line: usize) -> Result<i64, Error> {
        match self.angle(e, scope, line)? {
            Angle::Value(v) if v.fract() == 0.0 => Ok(v as i64),
            _ => Err(error(line, "expected an integer")),
        }
    }

    /// A classical condition over the bit registers.
    fn condition(&self, e: &Ex, scope: &Scope, line: usize) -> Result<Expr, Error> {
        Ok(match e {
            Ex::Num(v) if v.fract() == 0.0 && *v >= 0.0 => Expr::Value(*v as u64),
            Ex::Bool(b) => Expr::Value(*b as u64),
            Ex::Ident(name) if self.cregs.contains_key(name) => {
                let r = self.cregs[name];
                if r.array {
                    Expr::Register((r.start..r.start + r.size).collect())
                } else {
                    Expr::Bit(r.start)
                }
            }
            Ex::Ident(_) => Expr::Value(self.int(e, scope, line)?.max(0) as u64),
            Ex::Index(name, index) => {
                let operand = Operand {
                    name: name.clone(),
                    index: Some((**index).clone()),
                };
                Expr::Bit(self.clbits(&operand, scope, line)?[0])
            }
            Ex::Not(a) => Expr::Not(Box::new(self.condition(a, scope, line)?)),
            Ex::BitNot(a) => {
                let width = match &**a {
                    Ex::Ident(name) if self.cregs.contains_key(name) => {
                        self.cregs[name].size as u32
                    }
                    _ => 64,
                };
                Expr::BitNot {
                    value: Box::new(self.condition(a, scope, line)?),
                    width,
                }
            }
            Ex::Bin(op, a, b) => {
                let op = match op {
                    Op::And => BinaryOp::And,
                    Op::Or => BinaryOp::Or,
                    Op::BitAnd => BinaryOp::BitAnd,
                    Op::BitOr => BinaryOp::BitOr,
                    Op::BitXor => BinaryOp::BitXor,
                    Op::Eq => BinaryOp::Equal,
                    Op::Ne => BinaryOp::NotEqual,
                    Op::Lt => BinaryOp::Less,
                    Op::Le => BinaryOp::LessEqual,
                    Op::Gt => BinaryOp::Greater,
                    Op::Ge => BinaryOp::GreaterEqual,
                    _ => return Err(error(line, "arithmetic is not supported in conditions")),
                };
                Expr::binary(
                    op,
                    self.condition(a, scope, line)?,
                    self.condition(b, scope, line)?,
                )
            }
            _ => return Err(error(line, "unsupported condition")),
        })
    }

    /// One gate on `qubits`, with modifiers.
    #[allow(clippy::too_many_arguments)]
    fn call(
        &self,
        circ: &mut Circuit,
        name: &str,
        modifiers: &[Modifier],
        args: &[Angle],
        qubits: &[usize],
        scope: &Scope,
        line: usize,
    ) -> Result<(), Error> {
        let Some((first, rest)) = modifiers.split_first() else {
            return self.base_gate(circ, name, args, qubits, line);
        };
        match first {
            Modifier::Inv => {
                let mut inner = self.empty();
                self.call(&mut inner, name, rest, args, qubits, scope, line)?;
                circ.extend(&inner.inverse())
                    .map_err(|e| error(line, e.to_string()))?;
            }
            Modifier::Pow(k) => {
                let k = self.int(k, scope, line)?;
                let mut inner = self.empty();
                self.call(&mut inner, name, rest, args, qubits, scope, line)?;
                let step = if k < 0 { inner.inverse() } else { inner };
                for _ in 0..k.unsigned_abs() {
                    circ.extend(&step).map_err(|e| error(line, e.to_string()))?;
                }
            }
            Modifier::Ctrl(n) => {
                if *n != 1 || !rest.is_empty() || qubits.len() != 2 {
                    return Err(error(
                        line,
                        "only `ctrl @` with one control on a one-qubit gate is supported",
                    ));
                }
                let (control, target) = (qubits[0], qubits[1]);
                let controlled = match (name, args) {
                    ("rx", [a]) => Some(("crx", *a)),
                    ("ry", [a]) => Some(("cry", *a)),
                    ("rz", [a]) => Some(("crz", *a)),
                    ("p" | "phase" | "u1", [a]) => Some(("cp", *a)),
                    _ => None,
                };
                if let Some((controlled, angle)) = controlled {
                    return self.base_gate(circ, controlled, &[angle], &[control, target], line);
                }
                let mut inner = Circuit::new(1);
                inner.parameters(self.n_parameters);
                self.base_gate(&mut inner, name, args, &[0], line)?;
                let [op] = inner.ops() else {
                    return Err(error(line, format!("can't control `{name}`")));
                };
                let crate::circuit::Operation::Gate { gate, .. } = op else {
                    return Err(error(line, format!("can't control `{name}`")));
                };
                // A one-qubit gate's global phase becomes a relative phase.
                let phase = C64::cis(inner.global_phase());
                let m = gate.matrix().map(|row| row.map(|x| x * phase));
                circ.controlled(Gate::Unitary(m), control, target);
            }
        }
        Ok(())
    }

    fn base_gate(
        &self,
        circ: &mut Circuit,
        name: &str,
        args: &[Angle],
        qubits: &[usize],
        line: usize,
    ) -> Result<(), Error> {
        let arity = |params: usize, n: usize| -> Result<(), Error> {
            if args.len() != params || qubits.len() != n {
                return Err(error(
                    line,
                    format!("`{name}` takes {params} parameters and {n} qubits"),
                ));
            }
            if (1..n).any(|k| qubits[..k].contains(&qubits[k])) {
                return Err(error(line, format!("`{name}` uses a qubit twice")));
            }
            Ok(())
        };
        let constant = |a: Angle| match a {
            Angle::Value(v) => Ok(v),
            _ => Err(error(line, format!("`{name}` can't take a parameter here"))),
        };
        let q = qubits;
        match name {
            "id" | "i" | "u0" => {
                if name == "u0" {
                    arity(1, 1)?;
                } else {
                    arity(0, 1)?;
                }
            }
            "x" => {
                arity(0, 1)?;
                circ.x(q[0]);
            }
            "y" => {
                arity(0, 1)?;
                circ.y(q[0]);
            }
            "z" => {
                arity(0, 1)?;
                circ.z(q[0]);
            }
            "h" => {
                arity(0, 1)?;
                circ.h(q[0]);
            }
            "s" => {
                arity(0, 1)?;
                circ.s(q[0]);
            }
            "sdg" => {
                arity(0, 1)?;
                circ.sdg(q[0]);
            }
            "t" => {
                arity(0, 1)?;
                circ.t(q[0]);
            }
            "tdg" => {
                arity(0, 1)?;
                circ.tdg(q[0]);
            }
            "sx" | "sxdg" => {
                arity(0, 1)?;
                let (a, b) = (C64::new(0.5, 0.5), C64::new(0.5, -0.5));
                let (a, b) = if name == "sx" { (a, b) } else { (b, a) };
                circ.unitary([[a, b], [b, a]], q[0]);
            }
            "rx" => {
                arity(1, 1)?;
                circ.rx(args[0], q[0]);
            }
            "ry" => {
                arity(1, 1)?;
                circ.ry(args[0], q[0]);
            }
            "rz" => {
                arity(1, 1)?;
                circ.rz(args[0], q[0]);
            }
            "p" | "phase" | "u1" => {
                arity(1, 1)?;
                circ.p(args[0], q[0]);
            }
            "u2" => {
                arity(2, 1)?;
                circ.u(PI / 2.0, constant(args[0])?, constant(args[1])?, q[0]);
            }
            "u" | "U" | "u3" => {
                arity(3, 1)?;
                circ.u(
                    constant(args[0])?,
                    constant(args[1])?,
                    constant(args[2])?,
                    q[0],
                );
            }
            "cx" | "CX" | "cnot" => {
                arity(0, 2)?;
                circ.cx(q[0], q[1]);
            }
            "cy" => {
                arity(0, 2)?;
                circ.cy(q[0], q[1]);
            }
            "cz" => {
                arity(0, 2)?;
                circ.cz(q[0], q[1]);
            }
            "ch" => {
                arity(0, 2)?;
                circ.controlled(Gate::H, q[0], q[1]);
            }
            "cp" | "cphase" | "cu1" => {
                arity(1, 2)?;
                circ.cp(args[0], q[0], q[1]);
            }
            "crx" => {
                arity(1, 2)?;
                circ.crx(args[0], q[0], q[1]);
            }
            "cry" => {
                arity(1, 2)?;
                circ.cry(args[0], q[0], q[1]);
            }
            "crz" => {
                arity(1, 2)?;
                circ.crz(args[0], q[0], q[1]);
            }
            "cu" => {
                arity(4, 2)?;
                let u = Gate::Unitary(
                    u_matrix(constant(args[0])?, constant(args[1])?, constant(args[2])?)
                        .map(|row| row.map(|x| x * C64::cis(constant(args[3]).unwrap_or(0.0)))),
                );
                circ.controlled(u, q[0], q[1]);
            }
            "csx" => {
                arity(0, 2)?;
                let (a, b) = (C64::new(0.5, 0.5), C64::new(0.5, -0.5));
                circ.controlled(Gate::Unitary([[a, b], [b, a]]), q[0], q[1]);
            }
            "cu3" => {
                arity(3, 2)?;
                let u = u_matrix(constant(args[0])?, constant(args[1])?, constant(args[2])?);
                circ.controlled(Gate::Unitary(u), q[0], q[1]);
            }
            "swap" => {
                arity(0, 2)?;
                circ.swap(q[0], q[1]);
            }
            "rxx" => {
                arity(1, 2)?;
                circ.rxx(args[0], q[0], q[1]);
            }
            "ryy" => {
                arity(1, 2)?;
                circ.ryy(args[0], q[0], q[1]);
            }
            "rzz" => {
                arity(1, 2)?;
                circ.rzz(args[0], q[0], q[1]);
            }
            "rzx" => {
                arity(1, 2)?;
                circ.rzx(args[0], q[0], q[1]);
            }
            "ccx" | "toffoli" => {
                arity(0, 3)?;
                toffoli(circ, q[0], q[1], q[2]);
            }
            "cswap" | "fredkin" => {
                arity(0, 3)?;
                circ.cx(q[2], q[1]);
                toffoli(circ, q[0], q[1], q[2]);
                circ.cx(q[2], q[1]);
            }
            _ => {
                let def = self
                    .gates
                    .get(name)
                    .ok_or_else(|| error(line, format!("unknown gate `{name}`")))?;
                if def.params.len() != args.len() || def.qubits.len() != qubits.len() {
                    return Err(error(
                        line,
                        format!(
                            "`{name}` takes {} parameters and {} qubits",
                            def.params.len(),
                            def.qubits.len()
                        ),
                    ));
                }
                let scope = Scope {
                    values: def
                        .params
                        .iter()
                        .cloned()
                        .zip(args.iter().copied())
                        .collect(),
                    qubits: def
                        .qubits
                        .iter()
                        .cloned()
                        .zip(qubits.iter().copied())
                        .collect(),
                    nested: true,
                };
                for stmt in &def.body {
                    match &stmt.kind {
                        Kind::Call { .. } | Kind::GlobalPhase(_) | Kind::Nothing => {
                            self.statement(circ, stmt, &scope)?
                        }
                        _ => return Err(error(stmt.line, "gate bodies may only call gates")),
                    }
                }
            }
        }
        Ok(())
    }
}

/// `U(θ, φ, λ)` as in OpenQASM.
fn u_matrix(theta: f64, phi: f64, lambda: f64) -> [[C64; 2]; 2] {
    let (s, c) = (theta / 2.0).sin_cos();
    [
        [C64::new(c, 0.0), -C64::cis(lambda) * s],
        [C64::cis(phi) * s, C64::cis(phi + lambda) * c],
    ]
}

/// The Toffoli gate in Clifford+T.
fn toffoli(circ: &mut Circuit, a: usize, b: usize, c: usize) {
    circ.h(c)
        .cx(b, c)
        .tdg(c)
        .cx(a, c)
        .t(c)
        .cx(b, c)
        .tdg(c)
        .cx(a, c);
    circ.t(b).t(c).h(c).cx(a, b).t(a).tdg(b).cx(a, b);
}
