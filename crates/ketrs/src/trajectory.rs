//! Following circuits with measurement, reset and classical control on any
//! engine.
//!
//! Each engine implements [`Trajectory`]: running a unitary piece,
//! measuring a qubit, and sampling. [`run_trajectory`] follows one random
//! history; [`sample_circuit`] counts outcomes over many. On deterministic
//! engines it follows each distinct history once with its share of the
//! shots (see [`branch`]), sampling measurements after the last control
//! flow from the final state.

use crate::circuit::Circuit;
use crate::classical::{Expr, Instruction, Piece};
use crate::error::Error;
use crate::gate::Gate;
use rand::Rng;
use rand_distr::{Binomial, Distribution};
use std::collections::BTreeMap;
use std::rc::Rc;

/// Passes of one `while` loop after which a trajectory gives up, rather
/// than hang on a condition that never fails.
const MAX_LOOP_PASSES: usize = 1_000_000;

/// What an engine needs to follow a circuit's random history.
pub(crate) trait Trajectory: Sized {
    /// Whether running gates is deterministic, so that shots can share
    /// their simulation until their outcomes differ. Not so for noisy
    /// trajectories.
    const DETERMINISTIC: bool = true;

    fn n_qubits(&self) -> usize;
    /// Run a circuit of gates only.
    fn run_gates(&mut self, gates: &Circuit) -> Result<(), Error>;
    /// Run the circuit's gates as part of a trajectory, with any noise.
    fn run_gates_in<R: Rng>(&mut self, gates: &Circuit, _rng: &mut R) -> Result<(), Error> {
        self.run_gates(gates)
    }
    /// The bit recorded for a measurement with `outcome`: the outcome
    /// itself unless reading it out can fail.
    fn record<R: Rng>(&self, outcome: bool, _rng: &mut R) -> bool {
        outcome
    }
    /// Measure `qubit`, collapsing the state, and return the outcome.
    fn measure_qubit<R: Rng>(&mut self, qubit: usize, rng: &mut R) -> Result<bool, Error>;
    /// Probability that measuring `qubit` gives 1.
    fn probability_one(&mut self, qubit: usize) -> Result<f64, Error>;
    /// Project `qubit` onto `outcome` and renormalise.
    fn collapse(&mut self, qubit: usize, outcome: bool) -> Result<(), Error>;
    /// An independent copy, for another shot.
    fn duplicate(&self) -> Result<Self, Error>;
    /// For each of `shots` samples of every qubit, the values of `qubits`
    /// packed with `qubits[k]` at bit `k`. The state is unchanged.
    fn sample_qubits<R: Rng>(
        &mut self,
        shots: usize,
        qubits: &[usize],
        rng: &mut R,
    ) -> Result<Vec<u64>, Error>;
}

/// Run `circuit` on `engine` following one random history; returns the
/// classical bits (all starting at 0) as the circuit left them.
pub(crate) fn run_trajectory<E: Trajectory, R: Rng>(
    engine: &mut E,
    circuit: &Circuit,
    rng: &mut R,
) -> Result<Vec<bool>, Error> {
    check_runnable(engine, circuit)?;
    let mut clbits = vec![false; circuit.n_clbits()];
    top_level(follow(engine, &circuit.pieces(), &mut clbits, rng)?)?;
    Ok(clbits)
}

/// Run `circuit` from `engine`'s state `shots` times, counting each final
/// value of the classical bits (classical bit 0 least significant).
pub(crate) fn sample_circuit<E: Trajectory, R: Rng>(
    engine: &E,
    circuit: &Circuit,
    shots: usize,
    rng: &mut R,
) -> Result<BTreeMap<u64, usize>, Error> {
    check_runnable(engine, circuit)?;
    if circuit.n_clbits() > 64 {
        return Err(Error::InvalidArgument(
            "counting outcomes needs at most 64 classical bits".into(),
        ));
    }
    let mut counts = BTreeMap::new();
    if !E::DETERMINISTIC {
        for _ in 0..shots {
            let mut state = engine.duplicate()?;
            let mut clbits = vec![false; circuit.n_clbits()];
            top_level(follow(&mut state, &circuit.pieces(), &mut clbits, rng)?)?;
            *counts.entry(pack_bits(&clbits)).or_insert(0) += 1;
        }
        return Ok(counts);
    }
    if shots == 0 {
        return Ok(counts);
    }
    let pieces = circuit.pieces();
    let tail = tail_start(&pieces);
    let mut frames = vec![Frame::block(pieces)];
    let mut clbits = vec![false; circuit.n_clbits()];
    let mut state = engine.duplicate()?;
    branch(
        &mut state,
        &mut clbits,
        &mut frames,
        tail,
        shots,
        &mut counts,
        rng,
    )?;
    Ok(counts)
}

/// Classical bits as an integer, bit 0 least significant.
fn pack_bits(clbits: &[bool]) -> u64 {
    clbits
        .iter()
        .enumerate()
        .fold(0u64, |acc, (c, &b)| acc | (b as u64) << c)
}

fn check_runnable<E: Trajectory>(engine: &E, circuit: &Circuit) -> Result<(), Error> {
    if circuit.n_qubits() != engine.n_qubits() {
        return Err(Error::QubitCountMismatch {
            expected: engine.n_qubits(),
            found: circuit.n_qubits(),
        });
    }
    if circuit.n_parameters() > 0 {
        return Err(Error::InvalidArgument(format!(
            "circuit has {} unbound parameters; call bind first",
            circuit.n_parameters()
        )));
    }
    Ok(())
}

/// How a block ended: normally, or by `break` or `continue`, which pass up
/// through `if` branches to the innermost loop.
#[derive(Clone, Copy, PartialEq)]
enum Flow {
    Normal,
    Break,
    Continue,
}

/// A whole circuit must not end its trajectory with a stray `break`.
fn top_level(flow: Flow) -> Result<(), Error> {
    match flow {
        Flow::Normal => Ok(()),
        _ => Err(Error::InvalidArgument(
            "`break` or `continue` outside a loop".into(),
        )),
    }
}

fn follow<E: Trajectory, R: Rng>(
    engine: &mut E,
    pieces: &[Piece<'_>],
    clbits: &mut [bool],
    rng: &mut R,
) -> Result<Flow, Error> {
    for piece in pieces {
        match piece {
            Piece::Gates(gates) => engine.run_gates_in(gates, rng)?,
            Piece::Instruction(Instruction::Measure { qubit, clbit }) => {
                let outcome = engine.measure_qubit(*qubit, rng)?;
                clbits[*clbit] = engine.record(outcome, rng);
            }
            Piece::Instruction(Instruction::Assign { clbit, value }) => {
                clbits[*clbit] = value.holds(clbits);
            }
            Piece::Instruction(Instruction::Reset { qubit }) => {
                if engine.measure_qubit(*qubit, rng)? {
                    let mut flip = Circuit::new(engine.n_qubits());
                    flip.gate(Gate::X, *qubit);
                    engine.run_gates(&flip)?;
                }
            }
            Piece::Instruction(Instruction::IfElse {
                condition,
                then,
                otherwise,
            }) => {
                let branch = if condition.holds(clbits) {
                    then
                } else {
                    otherwise
                };
                let flow = follow(engine, &branch.pieces(), clbits, rng)?;
                if flow != Flow::Normal {
                    return Ok(flow);
                }
            }
            Piece::Instruction(Instruction::While { condition, body }) => {
                let pieces = body.pieces();
                let mut passes = 0;
                while condition.holds(clbits) {
                    if passes == MAX_LOOP_PASSES {
                        return Err(runaway_loop());
                    }
                    passes += 1;
                    if follow(engine, &pieces, clbits, rng)? == Flow::Break {
                        break;
                    }
                }
            }
            Piece::Instruction(Instruction::For { passes }) => {
                for body in passes {
                    if follow(engine, &body.pieces(), clbits, rng)? == Flow::Break {
                        break;
                    }
                }
            }
            Piece::Instruction(Instruction::Break) => return Ok(Flow::Break),
            Piece::Instruction(Instruction::Continue) => return Ok(Flow::Continue),
        }
    }
    Ok(Flow::Normal)
}

/// A block of pieces a path is running, and where it is in it.
#[derive(Clone)]
struct Frame<'a> {
    pieces: Rc<[Piece<'a>]>,
    next: usize,
    kind: Kind<'a>,
}

#[derive(Clone)]
enum Kind<'a> {
    /// A whole circuit or an `if` branch: done once its pieces are.
    Block,
    /// A `while` body on its `passes`th pass, tested again after each.
    While { condition: &'a Expr, passes: usize },
    /// A `for` body, with the passes still to come.
    For { rest: &'a [Circuit] },
}

impl<'a> Frame<'a> {
    fn new(pieces: Vec<Piece<'a>>, kind: Kind<'a>) -> Self {
        Frame {
            pieces: pieces.into(),
            next: 0,
            kind,
        }
    }

    fn block(pieces: Vec<Piece<'a>>) -> Self {
        Self::new(pieces, Kind::Block)
    }

    fn is_loop(&self) -> bool {
        !matches!(self.kind, Kind::Block)
    }
}

/// Follow `shots` shots from `state`, `clbits` and the position `frames`
/// (innermost block last) to the end of the circuit, adding their final
/// bits to `counts`. From the top-level piece `tail` on, the circuit only
/// measures and assigns.
///
/// At a measurement or reset whose outcome is random, the shots split
/// between its outcomes by a binomial draw, the same distribution as
/// measuring each shot. The smaller share follows a copy of the state
/// first, by recursion, and the larger then continues here, so at most
/// `log₂ shots` copies live at once. Each outcome with shots is followed
/// once, so no more paths are simulated than there are shots.
fn branch<E: Trajectory, R: Rng>(
    state: &mut E,
    clbits: &mut [bool],
    frames: &mut Vec<Frame<'_>>,
    tail: usize,
    mut shots: usize,
    counts: &mut BTreeMap<u64, usize>,
    rng: &mut R,
) -> Result<(), Error> {
    loop {
        let top_level = frames.len() == 1;
        let Some(frame) = frames.last_mut() else {
            *counts.entry(pack_bits(clbits)).or_insert(0) += shots;
            return Ok(());
        };
        if frame.next == frame.pieces.len() {
            // The block ended normally: go round its loop again or leave.
            match &mut frame.kind {
                Kind::Block => {
                    frames.pop();
                }
                Kind::While { condition, passes } => {
                    if condition.holds(clbits) {
                        if *passes == MAX_LOOP_PASSES {
                            return Err(runaway_loop());
                        }
                        *passes += 1;
                        frame.next = 0;
                    } else {
                        frames.pop();
                    }
                }
                Kind::For { rest } => match rest.split_first() {
                    Some((body, rest)) => *frame = Frame::new(body.pieces(), Kind::For { rest }),
                    None => {
                        frames.pop();
                    }
                },
            }
            continue;
        }
        // Reached once per path, so the tail is scanned once per path.
        if top_level
            && frame.next == tail
            && sample_tail(
                state,
                clbits,
                &frame.pieces[frame.next..],
                shots,
                counts,
                rng,
            )?
        {
            return Ok(());
        }
        let pieces = Rc::clone(&frame.pieces);
        frame.next += 1;
        let instruction = match &pieces[frame.next - 1] {
            Piece::Gates(gates) => {
                state.run_gates(gates)?;
                continue;
            }
            Piece::Instruction(instruction) => *instruction,
        };
        // A measurement records its outcome in a bit; a reset doesn't.
        let measured = match instruction {
            Instruction::Measure { qubit, clbit } => Some((*qubit, Some(*clbit))),
            Instruction::Reset { qubit } => Some((*qubit, None)),
            _ => None,
        };
        if let Some((qubit, clbit)) = measured {
            let ones = binomial(shots, state.probability_one(qubit)?, rng);
            // Settle a path on `outcome`: collapse, record the bit or
            // flip a reset qubit back to |0⟩.
            let settle = |state: &mut E, clbits: &mut [bool], outcome: bool| {
                state.collapse(qubit, outcome)?;
                match clbit {
                    Some(c) => clbits[c] = outcome,
                    None if outcome => {
                        let mut flip = Circuit::new(state.n_qubits());
                        flip.gate(Gate::X, qubit);
                        state.run_gates(&flip)?;
                    }
                    None => {}
                }
                Ok::<_, Error>(())
            };
            let zeros = shots - ones;
            if ones > 0 && zeros > 0 {
                let (fork, n_fork) = if ones <= zeros {
                    (true, ones)
                } else {
                    (false, zeros)
                };
                let mut other = state.duplicate()?;
                let mut other_bits = clbits.to_vec();
                settle(&mut other, &mut other_bits, fork)?;
                branch(
                    &mut other,
                    &mut other_bits,
                    &mut frames.clone(),
                    tail,
                    n_fork,
                    counts,
                    rng,
                )?;
                drop(other);
                shots -= n_fork;
                settle(state, clbits, !fork)?;
            } else {
                settle(state, clbits, ones > 0)?;
            }
            continue;
        }
        match instruction {
            Instruction::Measure { .. } | Instruction::Reset { .. } => {
                unreachable!("handled above")
            }
            Instruction::Assign { clbit, value } => clbits[*clbit] = value.holds(clbits),
            Instruction::IfElse {
                condition,
                then,
                otherwise,
            } => {
                let taken = if condition.holds(clbits) {
                    then
                } else {
                    otherwise
                };
                frames.push(Frame::block(taken.pieces()));
            }
            Instruction::While { condition, body } => {
                if condition.holds(clbits) {
                    frames.push(Frame::new(
                        body.pieces(),
                        Kind::While {
                            condition,
                            passes: 1,
                        },
                    ));
                }
            }
            Instruction::For { passes } => {
                if let Some((body, rest)) = passes.split_first() {
                    frames.push(Frame::new(body.pieces(), Kind::For { rest }));
                }
            }
            Instruction::Break => {
                // Leave every block up to and including the innermost loop.
                let depth = innermost_loop(frames)?;
                frames.truncate(depth);
            }
            Instruction::Continue => {
                // End the innermost loop's pass, as though its body ran out.
                let depth = innermost_loop(frames)?;
                frames.truncate(depth + 1);
                let frame = &mut frames[depth];
                frame.next = frame.pieces.len();
            }
        }
    }
}

/// The index in `frames` of the innermost loop.
fn innermost_loop(frames: &[Frame<'_>]) -> Result<usize, Error> {
    frames
        .iter()
        .rposition(Frame::is_loop)
        .ok_or_else(|| Error::InvalidArgument("`break` or `continue` outside a loop".into()))
}

fn runaway_loop() -> Error {
    Error::NotConverged(format!(
        "a while loop ran {MAX_LOOP_PASSES} times without its condition failing"
    ))
}

/// How many of `n` shots measure 1 with probability `p`: a binomial draw,
/// in constant time however many shots there are.
fn binomial<R: Rng>(n: usize, p: f64, rng: &mut R) -> usize {
    if p <= 0.0 {
        0
    } else if p >= 1.0 {
        n
    } else {
        Binomial::new(n as u64, p).expect("0 < p < 1").sample(rng) as usize
    }
}

/// Where the top-level `pieces` end in nothing but measurements and
/// assignments.
fn tail_start(pieces: &[Piece<'_>]) -> usize {
    pieces
        .iter()
        .rposition(|piece| {
            !matches!(
                piece,
                Piece::Instruction(Instruction::Measure { .. } | Instruction::Assign { .. })
            )
        })
        .map_or(0, |last| last + 1)
}

/// If `rest`, the end of the whole circuit, only measures and assigns,
/// finish the path's `shots` by sampling the measured qubits from `state`
/// (unchanged by measuring them, since no gate follows) and return true.
fn sample_tail<E: Trajectory, R: Rng>(
    state: &mut E,
    clbits: &[bool],
    rest: &[Piece<'_>],
    shots: usize,
    counts: &mut BTreeMap<u64, usize>,
    rng: &mut R,
) -> Result<bool, Error> {
    let mut qubits = Vec::new();
    for piece in rest {
        match piece {
            Piece::Instruction(Instruction::Measure { qubit, .. }) => {
                if !qubits.contains(qubit) {
                    qubits.push(*qubit);
                }
            }
            Piece::Instruction(Instruction::Assign { .. }) => {}
            _ => return Ok(false),
        }
    }
    if qubits.is_empty() || qubits.len() > 64 {
        return Ok(false);
    }
    let mut samples = BTreeMap::new();
    for sample in state.sample_qubits(shots, &qubits, rng)? {
        *samples.entry(sample).or_insert(0) += 1;
    }
    for (sample, n) in samples {
        let mut bits = clbits.to_vec();
        for piece in rest {
            match piece {
                Piece::Instruction(Instruction::Measure { qubit, clbit }) => {
                    let k = qubits.iter().position(|q| q == qubit).expect("listed");
                    bits[*clbit] = sample >> k & 1 == 1;
                }
                Piece::Instruction(Instruction::Assign { clbit, value }) => {
                    bits[*clbit] = value.holds(&bits);
                }
                _ => unreachable!("checked above"),
            }
        }
        *counts.entry(pack_bits(&bits)).or_insert(0) += n;
    }
    Ok(true)
}
