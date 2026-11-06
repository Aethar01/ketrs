//! Parameter gradients by adjoint differentiation.
//!
//! For `ψ = U_N ⋯ U_1 ψ₀` and a real loss `L(ψ)` with cotangent
//! `λ = ∂L/∂ψ*`, `dL/dθ = 2 Re⟨λ|∂ψ/∂θ⟩`. Every parametric gate has the form
//! `U(a) = exp(-ia/2 · G)` for a Hermitian generator `G`, so
//! `∂U/∂a = -i/2 · G U` and gate `k` contributes `Im⟨λₖ|G ψₖ⟩`, where `ψₖ`
//! is the state just after it and `λₖ` the cotangent pulled back to the
//! same point. One backward sweep that uncomputes both vectors gives every
//! parameter's gradient for about two circuit runs and three statevectors of
//! memory, however many parameters there are.

use super::hamiltonian::{CompiledPauliSum, Imaginary, chunked_sum};
use super::kernels::{CACHE_BLOCK_QUBITS, Prepared, Tile};
use super::{DenseState, FUSION_MIN_QUBITS};
use crate::circuit::{Circuit, Operation};
use crate::error::Error;
use crate::gate::Gate;
use crate::observable::{Pauli, PauliMasks, PauliString, PauliSum};
use crate::scalar::Scalar;
use num_complex::Complex64 as C64;
use std::ops::Range;

/// `G = coeff · Π_c |1⟩⟨1|_c ⊗ P`: a Pauli string restricted to basis states
/// whose `controls` bits are all set.
pub(crate) struct Generator {
    pub(crate) masks: PauliMasks,
    pub(crate) controls: usize,
    pub(crate) coeff: f64,
}

impl Generator {
    /// The generator of a parametric operation, `U(a) = exp(-ia/2 · G)`.
    pub(crate) fn of(op: &Operation) -> Generator {
        let pauli = |p, q| PauliString::new([(q, p)]).expect("one qubit").masks();
        let single = |gate: &Gate, target: usize, controls: usize| match gate {
            Gate::Rx(_) => Generator {
                masks: pauli(Pauli::X, target),
                controls,
                coeff: 1.0,
            },
            Gate::Ry(_) => Generator {
                masks: pauli(Pauli::Y, target),
                controls,
                coeff: 1.0,
            },
            Gate::Rz(_) => Generator {
                masks: pauli(Pauli::Z, target),
                controls,
                coeff: 1.0,
            },
            // P(a) = exp(ia|1⟩⟨1|) = exp(-ia/2 · (−2|1⟩⟨1|)).
            Gate::Phase(_) => Generator {
                masks: PauliMasks::default(),
                controls: controls | 1 << target,
                coeff: -2.0,
            },
            _ => unreachable!("only rotations take parameters"),
        };
        match op {
            Operation::Gate { gate, target } => single(gate, *target, 0),
            Operation::Controlled {
                gate,
                control,
                target,
            } => single(gate, *target, 1 << control),
            Operation::MultiControlled {
                gate,
                controls,
                target,
            } => single(
                gate,
                *target,
                controls.iter().fold(0, |acc, c| acc | 1 << c),
            ),
            Operation::PauliRotation { pauli, .. } => Generator {
                masks: pauli.masks(),
                controls: 0,
                coeff: 1.0,
            },
            _ => unreachable!("only rotations take parameters"),
        }
    }

    /// `⟨λ|Gψ⟩ = coeff · iⁿʸ Σ_{j ⊇ controls} λ*_j (-1)^|(j⊕x) ∧ z| ψ_{j⊕x}`.
    fn overlap<T: Scalar>(&self, lambda: &[T], psi: &[T]) -> C64 {
        let sum = chunked_sum(psi.len(), |range| {
            self.partial_sum(lambda, psi, range).to_complex()
        });
        self.coeff * self.masks.y_phase() * sum
    }

    /// The sum in [`overlap`](Self::overlap) over `range`.
    #[inline]
    fn partial_sum<T: Scalar>(&self, lambda: &[T], psi: &[T], range: Range<usize>) -> T {
        let m = self.masks;
        range
            .filter(|j| j & self.controls == self.controls)
            .map(|j| {
                let i = j ^ m.x;
                let t = lambda[j].conj() * psi[i];
                if m.z_sign(i) { -t } else { t }
            })
            .sum()
    }

    /// The generator seen from inside `tile`, whose qubits must include
    /// those of `x`.
    fn localize(&self, tile: &Tile) -> TileGenerator {
        let outside = !tile.qubits();
        TileGenerator {
            local: Generator {
                masks: PauliMasks {
                    x: tile.local(self.masks.x),
                    z: tile.local(self.masks.z),
                    n_y: self.masks.n_y,
                },
                controls: tile.local(self.controls),
                coeff: self.coeff,
            },
            controls: self.controls & outside,
            z: self.masks.z & outside,
        }
    }
}

/// A [`Generator`] restricted to one tile at a time: its action within the
/// tile, and the controls and Z signs fixed by the tile's other qubits.
struct TileGenerator {
    local: Generator,
    controls: usize,
    z: usize,
}

impl TileGenerator {
    /// The tile's part of the overlap, for the tile starting at `base`.
    fn overlap<T: Scalar>(&self, base: usize, lambda: &[T], psi: &[T]) -> C64 {
        if base & self.controls != self.controls {
            return C64::ZERO;
        }
        let g = &self.local;
        let sum =
            g.coeff * g.masks.y_phase() * g.partial_sum(lambda, psi, 0..psi.len()).to_complex();
        if (base & self.z).count_ones() % 2 == 1 {
            -sum
        } else {
            sum
        }
    }
}

/// Backward sweeps work on tiles of this many qubits: two of them (ψ and
/// λ) fit in L2 together.
const TILE_QUBITS: usize = CACHE_BLOCK_QUBITS - 1;

/// One step of the backward sweep.
enum Step<T> {
    /// Add `scale · Im⟨λ|Gψ⟩` to `gradient[parameter]` for each
    /// `(parameter, scale)`.
    Overlap(Generator, Vec<(usize, f64)>),
    /// Uncompute a gate on both ψ and λ.
    Apply(Prepared<T>),
}

/// A [`Step`] localised to a tile.
enum TileStep<T> {
    Overlap(TileGenerator),
    Apply(Prepared<T>),
}

impl<T: Scalar> Step<T> {
    /// The qubits a tile must include to take the step.
    fn qubit_mask(&self) -> usize {
        match self {
            Step::Overlap(g, _) => g.masks.x,
            Step::Apply(op) => op.qubit_mask(),
        }
    }
}

/// The backward sweep for `circuit`, bound to `bound`: an overlap before
/// each parametric gate and the inverses of the gates between them, fused
/// as [`DenseState::run`] would on large states. Gates before the first
/// parametric one are never uncomputed, since nothing reads the result.
fn backward_steps<T: Scalar>(circuit: &Circuit, bound: &Circuit) -> Result<Vec<Step<T>>, Error> {
    let n = bound.n_qubits();
    let mut steps = Vec::new();
    let mut segment = Circuit::new(n);
    let mut bindings = circuit.bindings().iter().rev().peekable();
    for (k, op) in bound.ops().iter().enumerate().rev() {
        let mut terms = Vec::new();
        while let Some(b) = bindings.next_if(|b| b.op == k) {
            terms.push((b.parameter, b.scale));
        }
        if !terms.is_empty() {
            let ops = if n >= FUSION_MIN_QUBITS {
                segment.fused_where(|op| op.max_qubit() >= CACHE_BLOCK_QUBITS)
            } else {
                segment
            };
            for op in ops.ops() {
                steps.push(Step::Apply(Prepared::new(op)?));
            }
            steps.push(Step::Overlap(Generator::of(op), terms));
            segment = Circuit::new(n);
        }
        segment.push(op.inverse())?;
    }
    Ok(steps)
}

/// Run `steps` on ψ and λ, accumulating overlaps into `gradient`. Runs of
/// steps on few enough qubits go tile by tile (see [`Tile`]), so each tile
/// of both vectors stays in cache while every gate and overlap in the run
/// passes over it.
fn sweep<T: Scalar>(steps: &[Step<T>], psi: &mut [T], lambda: &mut [T], gradient: &mut [f64]) {
    let mut record = |terms: &[(usize, f64)], overlap: C64| {
        for &(parameter, scale) in terms {
            gradient[parameter] += scale * overlap.im;
        }
    };
    let n_qubits = psi.len().trailing_zeros() as usize;
    let mut rest = steps;
    while let Some(first) = rest.first() {
        let (tile, count) = Tile::cover(n_qubits, TILE_QUBITS, rest.iter().map(Step::qubit_mask));
        if count >= 2 {
            let (batch, tail) = rest.split_at(count);
            let local: Vec<_> = batch
                .iter()
                .map(|step| match step {
                    Step::Overlap(g, _) => TileStep::Overlap(g.localize(&tile)),
                    Step::Apply(op) => TileStep::Apply(op.localize(&tile)),
                })
                .collect();
            let n_overlaps = batch
                .iter()
                .filter(|step| matches!(step, Step::Overlap(..)))
                .count();
            let sums = tile.map_reduce(
                [&mut *psi, &mut *lambda],
                |base, [psi, lambda]| {
                    let mut sums = Vec::with_capacity(n_overlaps);
                    for step in &local {
                        match step {
                            TileStep::Overlap(g) => sums.push(g.overlap(base, lambda, psi)),
                            TileStep::Apply(op) => {
                                op.apply(psi);
                                op.apply(lambda);
                            }
                        }
                    }
                    sums
                },
                || vec![C64::ZERO; n_overlaps],
                |a, b| a.iter().zip(&b).map(|(a, b)| a + b).collect(),
            );
            let overlaps = batch.iter().filter_map(|step| match step {
                Step::Overlap(_, terms) => Some(terms),
                Step::Apply(_) => None,
            });
            for (terms, sum) in overlaps.zip(sums) {
                record(terms, sum);
            }
            rest = tail;
        } else {
            match first {
                Step::Overlap(g, terms) => record(terms, g.overlap(lambda, psi)),
                Step::Apply(op) => {
                    op.apply(psi);
                    op.apply(lambda);
                }
            }
            rest = &rest[1..];
        }
    }
}

impl<T: Scalar> DenseState<T> {
    /// Gradient of a real loss `L` of the output of `circuit`, by adjoint
    /// differentiation.
    ///
    /// `final_state` must be `circuit` bound to `parameters` applied to some
    /// initial state, and `cotangent` is `λ = ∂L/∂ψ*` at `final_state`; for
    /// example `H|ψ⟩` for `L = ⟨ψ|H|ψ⟩`, or `|φ⟩⟨φ|ψ⟩` for `L = |⟨φ|ψ⟩|²`.
    /// Returns `dL/dθₚ = 2 Re⟨λ|∂ψ/∂θₚ⟩` for each parameter.
    pub fn adjoint_gradient(
        circuit: &Circuit,
        parameters: &[f64],
        final_state: &Self,
        cotangent: &Self,
    ) -> Result<Vec<f64>, Error> {
        let bound = circuit.bind(parameters)?;
        for state in [final_state, cotangent] {
            if state.n_qubits != circuit.n_qubits() {
                return Err(Error::QubitCountMismatch {
                    expected: circuit.n_qubits(),
                    found: state.n_qubits,
                });
            }
        }
        // Global phases cancel between ψ and λ, so need no handling.
        let steps = backward_steps::<T>(circuit, &bound)?;
        let mut psi = final_state.amps.clone();
        let mut lambda = cotangent.amps.clone();
        let mut gradient = vec![0.0; parameters.len()];
        sweep(&steps, &mut psi, &mut lambda, &mut gradient);
        Ok(gradient)
    }

    /// `⟨ψ(θ)|H|ψ(θ)⟩` and its gradient, where `ψ(θ)` is `circuit` bound to
    /// `parameters` applied to this state.
    pub fn expectation_gradient(
        &self,
        circuit: &Circuit,
        parameters: &[f64],
        observable: &PauliSum,
    ) -> Result<(f64, Vec<f64>), Error> {
        let mut final_state = self.clone();
        final_state.run(&circuit.bind(parameters)?)?;
        // On a real state, terms with an odd number of Ys vanish identically,
        // so dropping them changes neither the value nor the gradient.
        let op = CompiledPauliSum::<T>::new(observable, self.n_qubits, Imaginary::Skip, false)?;
        let mut h_psi = vec![T::ZERO; self.amps.len()];
        op.apply(&final_state.amps, &mut h_psi);
        let cotangent = DenseState {
            n_qubits: self.n_qubits,
            amps: h_psi,
        };
        let value = final_state.inner(&cotangent)?.to_complex().re;
        let gradient = Self::adjoint_gradient(circuit, parameters, &final_state, &cotangent)?;
        Ok((value, gradient))
    }
}
