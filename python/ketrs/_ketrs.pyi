"""Type stubs for the compiled `ketrs._ketrs` Rust extension module."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Literal, overload

import numpy as np
from numpy.typing import NDArray

__version__: str

DType = Literal["complex", "complex128", "real", "float64", "complex64", "float32"]
ResolvedDType = Literal["complex128", "float64", "complex64", "float32"]
GpuDType = Literal["complex", "complex128", "complex64"]
ResolvedGpuDType = Literal["complex128", "complex64"]
Angle = float | Parameter
Observable = PauliSum | SparseOperator | str | Sequence[tuple[str, float]]
Matrix2 = Sequence[Sequence[complex]]
Matrix4 = Sequence[Sequence[complex]]
ClassicalExpr = tuple[Any, ...]
BasisState = str | int
BondSectors = list[tuple[list[int], int]]

class Parameter:
    """A circuit parameter, or a linear form `scale * theta + offset` of one.

    Created via `Circuit.parameter()` or `Circuit.parameters()` and usable as any
    rotation angle, including affine expressions such as `2 * theta + 0.5`.
    """

    @property
    def index(self) -> int:
        """Position of this parameter's value in `Circuit.bind`."""
        ...

    @property
    def scale(self) -> float:
        """Linear scale factor multiplying the underlying parameter."""
        ...

    @property
    def offset(self) -> float:
        """Constant angular offset added to the scaled parameter."""
        ...

    def __mul__(self, k: float) -> Parameter: ...
    def __rmul__(self, k: float) -> Parameter: ...
    def __truediv__(self, k: float) -> Parameter: ...
    def __add__(self, k: float) -> Parameter: ...
    def __radd__(self, k: float) -> Parameter: ...
    def __sub__(self, k: float) -> Parameter: ...
    def __rsub__(self, k: float) -> Parameter: ...
    def __neg__(self) -> Parameter: ...
    def __repr__(self) -> str: ...

class Circuit:
    """A quantum circuit on `n_qubits` qubits and `n_clbits` classical bits.

    Builder methods mutate the circuit in place and return `self`, so gate calls
    can be chained: `Circuit(2).h(0).cx(0, 1)`.
    """

    def __init__(self, n_qubits: int, n_clbits: int = 0) -> None: ...
    @property
    def n_qubits(self) -> int:
        """Number of qubits in the circuit."""
        ...

    @property
    def n_clbits(self) -> int:
        """Number of classical bits in the circuit."""
        ...

    @property
    def is_unitary(self) -> bool:
        """Whether the circuit contains only unitary gates (no measurement, reset, or control flow)."""
        ...

    @property
    def global_phase(self) -> float:
        """Global phase angle in radians."""
        ...

    @global_phase.setter
    def global_phase(self, phase: float) -> None: ...
    @property
    def n_parameters(self) -> int:
        """Number of circuit parameters; `bind` expects this many values."""
        ...

    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...

    # Parameters & circuit-level transforms
    def parameter(self) -> Parameter:
        """Create a new parameter usable as any rotation angle."""
        ...

    def parameters(self, n: int) -> list[Parameter]:
        """Create `n` new circuit parameters."""
        ...

    def bind(self, values: Sequence[float]) -> Circuit:
        """Return a new circuit with every parameter replaced by its numerical value."""
        ...

    def inverse(self) -> Circuit:
        """Return the adjoint (inverse) circuit."""
        ...

    def control(self, controls: Sequence[int]) -> Circuit:
        """Return this circuit controlled on all qubits in `controls` being `|1⟩`.

        Parameters carry over to the controlled circuit.
        """
        ...

    def extend(self, other: Circuit) -> Circuit:
        """Append every operation of `other` (which must have the same width)."""
        ...

    # Inspection, optimization & QASM export
    def depth(self, two_qubit_only: bool = False) -> int:
        """Longest path through operations sharing qubits.

        With `two_qubit_only=True`, counts only multi-qubit operations
        (equivalent to `two_qubit_depth()`).
        """
        ...

    def two_qubit_depth(self) -> int:
        """Circuit depth counting only multi-qubit operations.

        Single-qubit gates contribute 0; undecomposed weight-`w` Pauli rotations
        contribute their CNOT-ladder depth `2 * (w - 1)`.
        """
        ...

    def gate_counts(self) -> dict[str, int]:
        """Gate and instruction counts keyed by lowercase name (e.g. `"cx"`, `"rz"`)."""
        ...

    def optimize(
        self,
        level: int = 2,
        basis: Literal["native", "cx"] = "native",
        tolerance: float = 1e-12,
        remove_diagonal_before_measure: bool = True,
        preserve_real: bool | None = None,
    ) -> Circuit:
        """Return a losslessly optimized equivalent circuit.

        Args:
            level: Optimization level from `0` (no-op) to `3` (heavy).
                `1` removes identities and consolidates 1-qubit runs;
                `2` (default) adds commutation cancellation, Clifford/swap rewrites,
                exact 2-qubit KAK synthesis, and smart Pauli-gadget lowering under `basis="cx"`;
                `3` adds multi-qubit Clifford/Pauli propagation, Pauli-rotation conjugation,
                and GF(2) CNOT subcircuit optimization.
            basis: `"native"` (default, ketrs's full standard gate set) or `"cx"`
                (lowers two-qubit gates and Pauli rotations to `{CX, 1q}`).
            tolerance: Numerical tolerance for equivalence checks.
            remove_diagonal_before_measure: Remove diagonal gates immediately preceding
                measurement or reset at `level >= 2`.
            preserve_real: Keep real circuits compatible with `dtype="real"`.
                `None` (default) preserves realness under `basis="native"` and allows
                general complex `{U, CX}` synthesis under `basis="cx"`.
        """
        ...

    def to_qasm(self) -> str:
        """Export the circuit as an OpenQASM 3 program readable by `from_qasm` and Qiskit."""
        ...

    # Single-qubit gates
    def h(self, qubit: int) -> Circuit:
        """Hadamard gate on `qubit`."""
        ...

    def x(self, qubit: int) -> Circuit:
        """Pauli-X gate on `qubit`."""
        ...

    def y(self, qubit: int) -> Circuit:
        """Pauli-Y gate on `qubit`."""
        ...

    def z(self, qubit: int) -> Circuit:
        """Pauli-Z gate on `qubit`."""
        ...

    def s(self, qubit: int) -> Circuit:
        """Phase gate `S = diag(1, i)` on `qubit`."""
        ...

    def sdg(self, qubit: int) -> Circuit:
        """Inverse phase gate `S†` on `qubit`."""
        ...

    def t(self, qubit: int) -> Circuit:
        """`T = diag(1, exp(iπ/4))` gate on `qubit`."""
        ...

    def tdg(self, qubit: int) -> Circuit:
        """Inverse `T†` gate on `qubit`."""
        ...

    def rx(self, theta: Angle, qubit: int) -> Circuit:
        """Rotation `exp(-i θ/2 · X)` on `qubit`."""
        ...

    def ry(self, theta: Angle, qubit: int) -> Circuit:
        """Rotation `exp(-i θ/2 · Y)` on `qubit`."""
        ...

    def rz(self, theta: Angle, qubit: int) -> Circuit:
        """Rotation `exp(-i θ/2 · Z)` on `qubit`."""
        ...

    def p(self, lam: Angle, qubit: int) -> Circuit:
        """Phase gate `diag(1, exp(i λ))` on `qubit`."""
        ...

    def u(self, theta: float, phi: float, lam: float, qubit: int) -> Circuit:
        """General single-qubit `U(θ, φ, λ)` gate (Qiskit convention) on `qubit`."""
        ...

    def unitary(self, matrix: Matrix2, qubit: int) -> Circuit:
        """Arbitrary single-qubit unitary given as a 2×2 nested sequence."""
        ...

    # Two-qubit and multi-qubit gates
    def cx(self, control: int, target: int) -> Circuit:
        """Controlled-X (CNOT) gate."""
        ...

    def cy(self, control: int, target: int) -> Circuit:
        """Controlled-Y gate."""
        ...

    def cz(self, control: int, target: int) -> Circuit:
        """Controlled-Z gate."""
        ...

    def cp(self, lam: Angle, control: int, target: int) -> Circuit:
        """Controlled phase gate `diag(1, 1, 1, exp(i λ))`."""
        ...

    def crx(self, theta: Angle, control: int, target: int) -> Circuit:
        """Controlled `Rx(θ)` gate."""
        ...

    def cry(self, theta: Angle, control: int, target: int) -> Circuit:
        """Controlled `Ry(θ)` gate."""
        ...

    def crz(self, theta: Angle, control: int, target: int) -> Circuit:
        """Controlled `Rz(θ)` gate."""
        ...

    def cunitary(self, matrix: Matrix2, control: int, target: int) -> Circuit:
        """Controlled arbitrary single-qubit unitary."""
        ...

    def swap(self, a: int, b: int) -> Circuit:
        """SWAP gate between qubits `a` and `b`."""
        ...

    def ccx(self, control1: int, control2: int, target: int) -> Circuit:
        """Toffoli (CCX) gate: `X` on `target` when both controls are `|1⟩`."""
        ...

    def mcx(self, controls: Sequence[int], target: int) -> Circuit:
        """Multi-controlled `X` on `target` when every qubit in `controls` is `|1⟩`."""
        ...

    def multi_controlled(
        self, matrix: Matrix2, controls: Sequence[int], target: int
    ) -> Circuit:
        """Arbitrary single-qubit unitary on `target` controlled on all `controls` being `|1⟩`."""
        ...

    def cswap(self, control: int, a: int, b: int) -> Circuit:
        """Fredkin (controlled-SWAP) gate."""
        ...

    def controlled_swap(
        self, controls: Sequence[int], a: int, b: int
    ) -> Circuit:
        """Swap `a` and `b` when every qubit in `controls` is `|1⟩`."""
        ...

    def pauli_rotation(
        self, theta: Angle, label: str, qubits: Sequence[int] | None = None
    ) -> Circuit:
        """Pauli rotation `exp(-i θ/2 · P)` applied in one pass.

        Without `qubits`, character `k` from the right of `label` acts on qubit `k`;
        otherwise it acts on `qubits[k]`.
        """
        ...

    def rxx(self, theta: Angle, a: int, b: int) -> Circuit:
        """Two-qubit rotation `exp(-i θ/2 · X⊗X)`."""
        ...

    def ryy(self, theta: Angle, a: int, b: int) -> Circuit:
        """Two-qubit rotation `exp(-i θ/2 · Y⊗Y)`."""
        ...

    def rzz(self, theta: Angle, a: int, b: int) -> Circuit:
        """Two-qubit rotation `exp(-i θ/2 · Z⊗Z)`."""
        ...

    def rzx(self, theta: Angle, a: int, b: int) -> Circuit:
        """Two-qubit rotation `exp(-i θ/2 · Z_a X_b)`, matching Qiskit's `rzx(θ, a, b)`."""
        ...

    def pauli_evolution(
        self,
        hamiltonian: Observable,
        time: float,
        steps: int = 1,
        order: int = 1,
    ) -> Circuit:
        """Append `steps` repetitions of an order-`order` Trotter–Suzuki formula for `exp(-iHt)`.

        Supports order 1 (Lie–Trotter), order 2 (symmetric Strang), and any higher even order
        via Suzuki's recursion.
        """
        ...

    def unitary2(self, matrix: Matrix4, q0: int, q1: int) -> Circuit:
        """Arbitrary two-qubit unitary with basis index `2 * b(q1) + b(q0)` (Qiskit convention)."""
        ...

    # Measurement, reset & classical control flow
    def measure(self, qubit: int, clbit: int) -> Circuit:
        """Measure `qubit` in the Z basis into classical bit `clbit`."""
        ...

    def measure_all(self) -> Circuit:
        """Measure every qubit `q` into classical bit `q`, adding classical bits if needed."""
        ...

    def reset(self, qubit: int) -> Circuit:
        """Measure `qubit` and reset it to `|0⟩`."""
        ...

    def if_test(
        self,
        clbits: int | Sequence[int],
        value: int,
        then: Circuit,
        otherwise: Circuit | None = None,
    ) -> Circuit:
        """Run `then` if `clbits` (least significant bit first) equal `value`, else `otherwise`."""
        ...

    def if_expr(
        self,
        condition: ClassicalExpr,
        then: Circuit,
        otherwise: Circuit | None = None,
    ) -> Circuit:
        """Run `then` if the classical expression `condition` is nonzero, else `otherwise`.

        Expressions are nested tuples: `("bit", c)`, `("register", [c0, c1, ...])`,
        `("value", v)`, `("not", e)`, `("bit_not", e, width)`, and `(op, a, b)` for
        `op` in `and`, `or`, `bit_and`, `bit_or`, `bit_xor`, `==`, `!=`,
        `<`, `<=`, `>`, `>=`.
        """
        ...

    def assign(self, clbit: int, value: ClassicalExpr) -> Circuit:
        """Set classical bit `clbit` to whether `value` (see `if_expr`) is nonzero."""
        ...

    def while_loop(self, condition: ClassicalExpr, body: Circuit) -> Circuit:
        """Run `body` while `condition` (see `if_expr`) is nonzero, testing before each pass."""
        ...

    def for_loop(self, passes: Sequence[Circuit]) -> Circuit:
        """Run each circuit of `passes` in turn, supporting `break_loop` and `continue_loop`."""
        ...

    def break_loop(self) -> Circuit:
        """Exit the innermost enclosing loop."""
        ...

    def continue_loop(self) -> Circuit:
        """Skip to the next iteration of the innermost enclosing loop."""
        ...

class PauliSum:
    """A real linear combination of Pauli strings, such as a Hermitian Hamiltonian."""

    def __init__(self, terms: Sequence[tuple[str, float]]) -> None:
        """Build from `(label, coefficient)` pairs. The rightmost character is qubit 0."""
        ...

    @property
    def n_qubits(self) -> int:
        """One more than the highest qubit index acted on by any term."""
        ...

    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...
    def to_list(self, n_qubits: int | None = None) -> list[tuple[str, float]]:
        """Return `(label, coefficient)` pairs padded to `n_qubits` characters."""
        ...

    def simplify(self, tolerance: float = 0.0) -> PauliSum:
        """Merge repeated Pauli strings and drop terms with `|coefficient| <= tolerance`."""
        ...

    def coefficient_norm(self) -> float:
        """Return `sum(|c_k|)`, an upper bound on the spectral norm."""
        ...

class SparseOperator:
    """A Hermitian operator stored as a CSR sparse matrix in the computational basis.

    Accepted by `Statevector.expectation`, `Statevector.apply_pauli_sum`,
    and `Statevector.evolve` in place of a `PauliSum`.
    """

    def __init__(
        self,
        n_qubits: int,
        indptr: NDArray[np.int64],
        indices: NDArray[np.int64],
        data: NDArray[np.complex128],
    ) -> None:
        """Build from CSR arrays on `n_qubits` qubits. Raises `ValueError` if not Hermitian."""
        ...

    @staticmethod
    def from_scipy(matrix: Any) -> SparseOperator:
        """Build from any SciPy sparse matrix/array (anything with `tocsr()`) or 2-D NumPy array."""
        ...

    @property
    def n_qubits(self) -> int:
        """Number of qubits the operator acts on."""
        ...

    @property
    def nnz(self) -> int:
        """Number of stored nonzero entries."""
        ...

    def is_real(self) -> bool:
        """Whether every stored entry has zero imaginary part."""
        ...

    def __repr__(self) -> str: ...

class Statevector:
    """An exact pure state on `n_qubits` qubits on the CPU.

    Qubit 0 is the least significant bit of a basis-state index (matching Qiskit).

    Args:
        n_qubits: Number of qubits.
        seed: Optional RNG seed for reproducible measurements and sampling.
        dtype: `"complex"` / `"complex128"` (default), `"real"` / `"float64"`
            (half memory, real-only gates), `"complex64"`, or `"float32"`
            (single precision, ~1e-6 accuracy).
    """

    def __init__(
        self, n_qubits: int, seed: int | None = None, dtype: DType = "complex"
    ) -> None: ...
    @staticmethod
    def from_circuit(
        circuit: Circuit, seed: int | None = None, dtype: DType = "complex"
    ) -> Statevector:
        """Run `circuit` starting from `|0…0⟩`."""
        ...

    @staticmethod
    def from_amplitudes(
        amplitudes: NDArray[np.complex128]
        | NDArray[np.float64]
        | NDArray[np.complex64]
        | NDArray[np.float32],
        seed: int | None = None,
    ) -> Statevector:
        """Wrap a 1-D NumPy array whose length is a power of two.

        The array's dtype (`complex128`, `float64`, `complex64`, or `float32`)
        determines the statevector's dtype.
        """
        ...

    @property
    def n_qubits(self) -> int:
        """Number of qubits."""
        ...

    @property
    def dtype(self) -> ResolvedDType:
        """Amplitude storage dtype: `"complex128"`, `"float64"`, `"complex64"`, or `"float32"`."""
        ...

    def __repr__(self) -> str: ...
    def run(
        self, circuit: Circuit, noise: NoiseModel | None = None
    ) -> list[int] | None:
        """Apply `circuit` in place.

        For a unitary circuit without `noise`, returns `None`. For a circuit with
        measurements, resets, or classical control, follows one random trajectory and
        returns its classical bit values. With `noise`, runs a noisy quantum trajectory.
        """
        ...

    def sample_circuit(
        self, circuit: Circuit, shots: int, noise: NoiseModel | None = None
    ) -> dict[str, int]:
        """Run `circuit` `shots` times from this state (without modifying it) and count final classical bits.

        Keys are bitstrings with classical bit 0 rightmost. With `noise`, each shot
        follows a noisy quantum trajectory including readout error.
        """
        ...

    def amplitudes(self) -> NDArray[Any]:
        """Return a copy of the statevector amplitudes as a 1-D NumPy array."""
        ...

    def probabilities(self) -> NDArray[np.float64]:
        """Measurement probability of every computational basis state."""
        ...

    def inner(self, other: Statevector) -> complex:
        """Inner product `⟨self|other⟩`. Both states must share the same dtype."""
        ...

    def expectation(self, observable: Observable) -> float:
        """Expectation value `⟨ψ|H|ψ⟩`."""
        ...

    def apply_pauli_sum(self, hamiltonian: Observable) -> Statevector:
        """Return `H|ψ⟩` as a new (generally unnormalised) statevector."""
        ...

    def evolve(self, hamiltonian: Observable, time: float) -> None:
        """Evolve in place by exact `exp(-iHt)` via matrix-free Krylov time-stepping.

        Requires `dtype="complex"` or `"complex64"`.
        """
        ...

    def expectation_gradient(
        self,
        circuit: Circuit,
        parameters: Sequence[float],
        observable: Observable,
    ) -> tuple[float, NDArray[np.float64]]:
        """Compute `⟨ψ(θ)|H|ψ(θ)⟩` and its exact gradient by adjoint differentiation.

        `ψ(θ)` is `circuit` bound to `parameters` applied to this state (unchanged).
        Returns `(value, gradient)`.
        """
        ...

    @staticmethod
    def adjoint_gradient(
        circuit: Circuit,
        parameters: Sequence[float],
        final_state: Statevector,
        cotangent: Statevector,
    ) -> NDArray[np.float64]:
        """Compute `dL/dθ = 2 Re⟨λ|∂ψ/∂θ⟩` for any real loss `L` of `circuit`'s output.

        Args:
            circuit: Parametric circuit.
            parameters: Parameter values at which `final_state` was evaluated.
            final_state: Output state `|ψ(θ)⟩`.
            cotangent: `|λ⟩ = ∂L/∂ψ*` (e.g. `H|ψ⟩` for an expectation value).
        """
        ...

    def probability_one(self, qubit: int) -> float:
        """Probability that measuring `qubit` yields `1`."""
        ...

    def sample(self, shots: int) -> NDArray[np.uintp]:
        """Draw `shots` computational basis-state indices without modifying the state."""
        ...

    def sample_counts(self, shots: int) -> dict[str, int]:
        """Sample `shots` bitstrings (qubit 0 rightmost) without modifying the state."""
        ...

    def measure(self, qubit: int) -> int:
        """Projectively measure `qubit`, collapsing the state. Returns `0` or `1`."""
        ...

    def measure_all(self) -> int:
        """Measure every qubit, collapsing to a basis state. Returns its integer index."""
        ...

    def reset(self, qubit: int) -> None:
        """Measure `qubit` and flip it to `|0⟩` if the outcome was `1`."""
        ...

class Eigenstate:
    """An eigenpair returned by `lowest_eigenstates`."""

    @property
    def energy(self) -> float:
        """Eigenvalue computed as the Rayleigh quotient `⟨x|H|x⟩`."""
        ...

    @property
    def state(self) -> Statevector | SectorStatevector:
        """Normalised eigenvector (a `SectorStatevector` when solved in a sector)."""
        ...

    @property
    def residual(self) -> float:
        """Direct residual norm `‖Hx - Ex‖`."""
        ...

    def __repr__(self) -> str: ...

class SectorBasis:
    """Computational basis states constrained to fixed Hamming weights.

    For each `(qubits, weight)` pair in `constraints`, exactly `weight` qubits
    among `qubits` are in `|1⟩`; unconstrained qubits may be `0` or `1`.
    For example, `SectorBasis(8, [(range(8), 4)])` is half filling on 8 sites.
    """

    def __init__(
        self, n_qubits: int, constraints: Sequence[tuple[Iterable[int], int]]
    ) -> None: ...
    @property
    def n_qubits(self) -> int:
        """Total number of qubits."""
        ...

    @property
    def dim(self) -> int:
        """Number of basis states in the symmetry sector."""
        ...

    def __len__(self) -> int: ...
    def __contains__(self, state: int) -> bool: ...
    def __repr__(self) -> str: ...
    def index(self, state: int) -> int | None:
        """Position of computational basis state `state` within the sector, or `None`."""
        ...

    def state(self, position: int) -> int:
        """Computational basis state integer at `position` in sector order."""
        ...

    def states(self) -> NDArray[np.uintp]:
        """All computational basis state integers in the sector, in order."""
        ...

class SectorStatevector:
    """A statevector restricted to a `SectorBasis` symmetry subspace.

    Stores only one amplitude per sector basis state. Circuits and Hamiltonians must
    conserve the sector; anything that would leave it raises `ValueError` without
    modifying the state.
    """

    def __init__(
        self,
        basis: SectorBasis,
        state: int,
        seed: int | None = None,
        dtype: DType = "complex",
    ) -> None:
        """Initialise to computational basis state `state` (which must belong to `basis`)."""
        ...

    @staticmethod
    def from_dense(
        basis: SectorBasis, state: Statevector, seed: int | None = None
    ) -> SectorStatevector:
        """Extract the components of a full `Statevector` lying in `basis` (unrenormalised)."""
        ...

    @property
    def n_qubits(self) -> int:
        """Total number of qubits."""
        ...

    @property
    def dim(self) -> int:
        """Number of stored amplitudes (sector dimension)."""
        ...

    @property
    def dtype(self) -> ResolvedDType:
        """Amplitude storage dtype."""
        ...

    @property
    def basis(self) -> SectorBasis:
        """The `SectorBasis` defining this sector."""
        ...

    def __repr__(self) -> str: ...
    def run(self, circuit: Circuit) -> None:
        """Apply a sector-conserving `circuit` in place."""
        ...

    def amplitudes(self) -> NDArray[Any]:
        """Amplitudes in sector order (matching `SectorBasis.states`)."""
        ...

    def probabilities(self) -> NDArray[np.float64]:
        """Probabilities of each sector basis state, in sector order."""
        ...

    def to_dense(self) -> Statevector:
        """Expand to a full `Statevector`."""
        ...

    def inner(self, other: SectorStatevector) -> complex:
        """Inner product `⟨self|other⟩` (same sector and dtype required)."""
        ...

    def expectation(self, observable: Observable) -> float:
        """Expectation value `⟨ψ|H|ψ⟩`; terms leaving the sector contribute zero."""
        ...

    def apply_pauli_sum(self, hamiltonian: Observable) -> SectorStatevector:
        """Return `H|ψ⟩` for a sector-conserving Hamiltonian."""
        ...

    def evolve(self, hamiltonian: Observable, time: float) -> None:
        """Evolve in place by exact `exp(-iHt)` within the sector (requires a complex dtype)."""
        ...

    def sample(self, shots: int) -> NDArray[np.uintp]:
        """Draw `shots` full computational basis-state indices."""
        ...

    def sample_counts(self, shots: int) -> dict[str, int]:
        """Sample `shots` bitstrings (qubit 0 rightmost)."""
        ...

class Channel:
    """A completely positive trace-preserving quantum channel given by Kraus operators."""

    @staticmethod
    def from_kraus(kraus: Sequence[NDArray[np.complex128]]) -> Channel:
        """Build a channel from square complex Kraus arrays satisfying `sum(K† K) = I`."""
        ...

    @staticmethod
    def depolarizing(p: float, n_qubits: int = 1) -> Channel:
        """Depolarizing noise `ρ -> (1 - p) ρ + p · I / 2^n` (matches Qiskit Aer)."""
        ...

    @staticmethod
    def pauli(n_qubits: int, terms: Sequence[tuple[str, float]]) -> Channel:
        """Mixture of Pauli strings `[(label, prob), ...]`; identity gets the remaining weight."""
        ...

    @staticmethod
    def bit_flip(p: float) -> Channel:
        """Single-qubit bit-flip channel (`X` with probability `p`)."""
        ...

    @staticmethod
    def phase_flip(p: float) -> Channel:
        """Single-qubit phase-flip channel (`Z` with probability `p`)."""
        ...

    @staticmethod
    def amplitude_damping(gamma: float) -> Channel:
        """Amplitude damping: `|1⟩` decays to `|0⟩` with probability `gamma`."""
        ...

    @staticmethod
    def phase_damping(lam: float) -> Channel:
        """Phase damping: off-diagonal elements shrink by `sqrt(1 - lam)`."""
        ...

    @property
    def n_qubits(self) -> int:
        """Number of qubits this channel acts on."""
        ...

    def kraus(self) -> list[NDArray[np.complex128]]:
        """Return the Kraus operators as 2-D complex NumPy arrays."""
        ...

    def __repr__(self) -> str: ...

class NoiseModel:
    """Per-gate and readout noise model for `DensityMatrix` and noisy `Statevector` trajectories.

    Args:
        one_qubit: 1-qubit `Channel` applied after every single-qubit gate (and
            to each qubit of gates without a matching `two_qubit` channel).
        two_qubit: 2-qubit `Channel` applied after every two-qubit gate.
        readout: `(p10, p01)`: probabilities of recording `1` given true outcome `0`,
            and recording `0` given true outcome `1`. Applied by statevector trajectories.
    """

    def __init__(
        self,
        one_qubit: Channel | None = None,
        two_qubit: Channel | None = None,
        readout: tuple[float, float] | None = None,
    ) -> None: ...

class DensityMatrix:
    """Exact mixed-state density matrix on `n_qubits` qubits."""

    def __init__(self, n_qubits: int, seed: int | None = None) -> None:
        """Initialise to `|0…0⟩⟨0…0|`."""
        ...

    @staticmethod
    def from_statevector(
        state: Statevector, seed: int | None = None
    ) -> DensityMatrix:
        """Build `|ψ⟩⟨ψ|` from a `complex128` `Statevector`."""
        ...

    @staticmethod
    def from_matrix(
        matrix: NDArray[np.complex128], seed: int | None = None
    ) -> DensityMatrix:
        """Build from a square complex NumPy array of power-of-two dimension."""
        ...

    @property
    def n_qubits(self) -> int:
        """Number of qubits."""
        ...

    def __repr__(self) -> str: ...
    def run(self, circuit: Circuit) -> None:
        """Apply a unitary circuit: `ρ -> U ρ U†`."""
        ...

    def run_noisy(self, circuit: Circuit, noise: NoiseModel) -> None:
        """Run `circuit` applying `noise` after every gate."""
        ...

    def apply_channel(self, channel: Channel, qubits: Sequence[int]) -> None:
        """Apply a Kraus `Channel` to `qubits`."""
        ...

    def to_matrix(self) -> NDArray[np.complex128]:
        """Return the density matrix as a 2-D complex NumPy array."""
        ...

    def probabilities(self) -> NDArray[np.float64]:
        """Diagonal measurement probabilities of every basis state."""
        ...

    def expectation(self, observable: Observable) -> float:
        """Return `Tr(ρ H)`."""
        ...

    def trace(self) -> float:
        """Return `Tr(ρ)`."""
        ...

    def purity(self) -> float:
        """Return `Tr(ρ²)`."""
        ...

    def fidelity(self, state: Statevector) -> float:
        """Return `⟨ψ|ρ|ψ⟩` against a pure `complex128` `Statevector`."""
        ...

    def sample_counts(self, shots: int) -> dict[str, int]:
        """Sample `shots` bitstrings (qubit 0 rightmost) from the diagonal probabilities."""
        ...

class Charge:
    """A conserved additive Abelian charge (`U(1)` or `Z_n`) for symmetric MPS.

    Used with `MPS.with_charges` and `PeriodicMPS.with_charges`.
    A basis state's charge is the sum (modulo `n` for `Z_n`) of its qubits' values.

    Examples:
        - Particle number on `n` qubits: `Charge.weights([1] * n)`
        - Parity: `Charge.modulo(2, [(0, 1)] * n)`
        - `2 S^z`: `Charge.u1([(1, -1)] * n)`
    """

    @staticmethod
    def u1(values: Sequence[tuple[int, int]]) -> Charge:
        """Create a `U(1)` charge where `values[q] = (val_for_0, val_for_1)`."""
        ...

    @staticmethod
    def modulo(n: int, values: Sequence[tuple[int, int]]) -> Charge:
        """Create a `Z_n` charge (`n >= 1`) added modulo `n`."""
        ...

    @staticmethod
    def weights(weights: Sequence[int]) -> Charge:
        """Create a `U(1)` charge counting `weights[q]` when qubit `q` is `|1⟩` (0 for `|0⟩`)."""
        ...

    @property
    def modulus(self) -> int | None:
        """Modulus `n` for a `Z_n` charge, or `None` for `U(1)`."""
        ...

    @property
    def values(self) -> list[tuple[int, int]]:
        """`(value of |0⟩, value of |1⟩)` for each qubit."""
        ...

    def __repr__(self) -> str: ...

class MPS:
    """Open-boundary Matrix Product State for weakly entangled states of many qubits.

    Memory and runtime scale with the bond dimension `χ` rather than `2^n`.
    Each bond keeps at most `max_bond` singular values, dropping values whose
    discarded weight stays below `cutoff`.
    """

    def __init__(
        self,
        n_qubits: int,
        max_bond: int = 64,
        cutoff: float = 1e-14,
        seed: int | None = None,
    ) -> None:
        """Initialise `|0…0⟩` on `n_qubits` qubits."""
        ...

    @staticmethod
    def from_circuit(
        circuit: Circuit,
        max_bond: int = 64,
        cutoff: float = 1e-14,
        seed: int | None = None,
    ) -> MPS:
        """Run `circuit` on `|0…0⟩`."""
        ...

    @staticmethod
    def random(
        n_qubits: int,
        bond: int = 8,
        seed: int = 0,
        max_bond: int = 64,
        cutoff: float = 1e-14,
    ) -> MPS:
        """Create a random normalised MPS with bond dimension up to `bond` (useful for DMRG)."""
        ...

    @property
    def n_qubits(self) -> int:
        """Number of qubits."""
        ...

    @property
    def bond_dimensions(self) -> list[int]:
        """Bond dimension between each pair of adjacent qubits (length `n_qubits - 1`)."""
        ...

    @property
    def truncation_error(self) -> float:
        """Cumulative discarded weight from SVD truncations."""
        ...

    @property
    def gpu(self) -> bool:
        """Whether `tdvp` and `dmrg` run effective Hamiltonians on a Vulkan FP64 GPU."""
        ...

    @gpu.setter
    def gpu(self, enabled: bool) -> None: ...
    @property
    def max_bond(self) -> int:
        """Maximum allowed bond dimension `χ`."""
        ...

    @max_bond.setter
    def max_bond(self, max_bond: int) -> None: ...
    @property
    def cutoff(self) -> float:
        """Singular-value discarded-weight truncation cutoff."""
        ...

    @cutoff.setter
    def cutoff(self, cutoff: float) -> None: ...
    @property
    def charges(self) -> list[Charge] | None:
        """Conserved charges of a symmetric MPS (see `with_charges`), or `None`."""
        ...

    @property
    def total_charge(self) -> list[int] | None:
        """Total charge vector of a symmetric MPS, or `None`."""
        ...

    def __repr__(self) -> str: ...
    def __copy__(self) -> MPS: ...
    def __deepcopy__(self, memo: Any) -> MPS: ...
    def copy(self, seed: int | None = None) -> MPS:
        """Return an independent copy of the MPS, optionally with a new RNG `seed`."""
        ...

    def norm_sqr(self) -> float:
        """Squared norm `⟨ψ|ψ⟩`."""
        ...

    def dmrg(
        self,
        hamiltonian: Observable,
        max_bond: int = 64,
        cutoff: float = 1e-12,
        max_sweeps: int = 20,
        tolerance: float = 1e-10,
        noise: float = 1e-4,
        noise_decay: float = 0.1,
    ) -> tuple[float, list[float], bool]:
        """Find the ground state of `hamiltonian` in place by two-site DMRG.

        Returns `(energy, sweep_energies, converged)`. Start from `MPS.random`
        for the unconstrained ground state or from a definite-symmetry state to stay
        within a sector. Nonzero `noise` widens bonds using subspace expansion so
        multi-qubit (>2-site) terms cannot trap a product state.
        """
        ...

    def tebd(
        self,
        hamiltonian: Observable,
        time: float,
        steps: int = 1,
        order: int = 2,
    ) -> None:
        """Evolve in place by `exp(-iHt)` using `steps` Trotter–Suzuki TEBD steps."""
        ...

    def tdvp(
        self, hamiltonian: Observable, time: float, steps: int = 1
    ) -> None:
        """Evolve in place by `exp(-iHt)` using `steps` two-site TDVP steps.

        Supports Hamiltonians of arbitrary range and automatically performs global
        subspace expansion while bonds are below `max_bond`.
        """
        ...

    def run(self, circuit: Circuit) -> list[int] | None:
        """Apply `circuit` in place. Returns classical bits if non-unitary, else `None`."""
        ...

    def sample_circuit(self, circuit: Circuit, shots: int) -> dict[str, int]:
        """Run `circuit` `shots` times from this MPS (unchanged) and return bitstring counts."""
        ...

    def expectation(self, observable: Observable) -> float:
        """Return `⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩`."""
        ...

    def expectations(self, observables: Sequence[Observable]) -> list[float]:
        """Evaluate many expectation values at once, contracting each string only over its own span."""
        ...

    def amplitude(self, state: BasisState) -> complex:
        """Exact amplitude of a basis state (Qiskit-style bitstring or integer index)."""
        ...

    def to_statevector(self) -> Statevector:
        """Expand to a full `Statevector` (up to 30 qubits)."""
        ...

    @overload
    def project(self, qubits: int, outcome: int) -> float: ...
    @overload
    def project(
        self, qubits: Sequence[tuple[int, int]], outcome: None = None
    ) -> float: ...
    def project(
        self, qubits: int | Sequence[tuple[int, int]], outcome: int | None = None
    ) -> float:
        """Post-select onto `|outcome⟩⟨outcome|` on one or more qubits and renormalise.

        Returns the (joint) probability of the chosen outcome(s).
        """
        ...

    def with_charges(self, charges: Sequence[Charge]) -> MPS:
        """Convert to a block-sparse symmetric MPS conserving `charges`.

        DMRG and TDVP then run block-by-block within the state's charge sector.
        """
        ...

    def to_dense(self) -> MPS:
        """Convert a symmetric MPS back to a dense MPS (required before applying gates)."""
        ...

    def bond_sectors(self, bond: int) -> BondSectors | None:
        """Return `[(left_charges, dim), ...]` for `bond` (between `bond` and `bond + 1`), or `None`."""
        ...

    def sample(self, shots: int) -> NDArray[np.uintp]:
        """Draw `shots` basis-state indices (up to 64 qubits)."""
        ...

    def sample_counts(self, shots: int) -> dict[str, int]:
        """Sample `shots` bitstrings (qubit 0 rightmost; up to 64 qubits)."""
        ...

class PeriodicMPS:
    """Periodic-trace Matrix Product State on a closed ring of qubits.

    Represents `ψ(b_0 … b_{n-1}) = Tr(A[0][b_0] ⋯ A[n-1][b_{n-1}])`, preserving
    translation symmetry across the `(n - 1, 0)` seam with Pippan–White–Evertz
    adaptive transfer-matrix compression for periodic DMRG, TDVP, and TEBD.
    """

    def __init__(
        self,
        n_qubits: int,
        max_bond: int = 64,
        cutoff: float = 1e-14,
        transfer_rank: int | None = None,
        transfer_cutoff: float = 1e-12,
        seed: int | None = None,
    ) -> None:
        """Initialise `|0…0⟩` on a closed ring of `n_qubits` qubits."""
        ...

    @staticmethod
    def from_circuit(
        circuit: Circuit,
        max_bond: int = 64,
        cutoff: float = 1e-14,
        seed: int | None = None,
    ) -> PeriodicMPS:
        """Run `circuit` on `|0…0⟩` on the periodic ring."""
        ...

    @staticmethod
    def from_mps(mps: MPS) -> PeriodicMPS:
        """Wrap an open-boundary `MPS` as a `PeriodicMPS` with seam bond 1."""
        ...

    @staticmethod
    def random(
        n_qubits: int,
        bond: int = 4,
        seed: int = 0,
        max_bond: int = 64,
        cutoff: float = 1e-14,
    ) -> PeriodicMPS:
        """Create a random normalised periodic MPS with bond dimension `bond` on every ring bond."""
        ...

    @property
    def n_qubits(self) -> int:
        """Number of qubits on the ring."""
        ...

    @property
    def bond_dimensions(self) -> list[int]:
        """Bond dimensions of every ring bond (length `n_qubits`; last entry is the `(n - 1, 0)` seam)."""
        ...

    @property
    def truncation_error(self) -> float:
        """Cumulative discarded weight from truncations."""
        ...

    @property
    def gpu(self) -> bool:
        """Whether periodic DMRG and TDVP local operators run on a Vulkan FP64 GPU."""
        ...

    @gpu.setter
    def gpu(self, enabled: bool) -> None: ...
    @property
    def max_bond(self) -> int:
        """Maximum allowed bond dimension `χ`."""
        ...

    @max_bond.setter
    def max_bond(self, max_bond: int) -> None: ...
    @property
    def cutoff(self) -> float:
        """Singular-value truncation cutoff."""
        ...

    @cutoff.setter
    def cutoff(self, cutoff: float) -> None: ...
    @property
    def transfer_rank(self) -> int | None:
        """PWE opposite-half transfer-matrix rank cap (`None` = adaptive, `0` = exact)."""
        ...

    @transfer_rank.setter
    def transfer_rank(self, rank: int | None) -> None: ...
    @property
    def transfer_cutoff(self) -> float:
        """Relative singular-value cutoff for PWE transfer-matrix compression."""
        ...

    @transfer_cutoff.setter
    def transfer_cutoff(self, cutoff: float) -> None: ...
    @property
    def charges(self) -> list[Charge] | None:
        """Conserved charges of a symmetric periodic MPS, or `None`."""
        ...

    @property
    def total_charge(self) -> list[int] | None:
        """Total charge vector of a symmetric periodic MPS, or `None`."""
        ...

    def __repr__(self) -> str: ...
    def __copy__(self) -> PeriodicMPS: ...
    def __deepcopy__(self, memo: Any) -> PeriodicMPS: ...
    def copy(self, seed: int | None = None) -> PeriodicMPS:
        """Return an independent copy, optionally with a new RNG `seed`."""
        ...

    def norm_sqr(self) -> float:
        """Exact periodic trace squared norm `⟨ψ|ψ⟩`."""
        ...

    def normalise(self) -> None:
        """Normalise the periodic MPS in place."""
        ...

    def inner(self, other: PeriodicMPS) -> complex:
        """Exact periodic trace inner product `⟨self|other⟩`."""
        ...

    def dmrg(
        self,
        hamiltonian: Observable,
        max_bond: int = 64,
        cutoff: float = 1e-12,
        max_sweeps: int = 20,
        tolerance: float = 1e-10,
        noise: float = 1e-4,
        noise_decay: float = 0.1,
    ) -> tuple[float, list[float], bool]:
        """Find the ground state on the periodic ring by periodic DMRG.

        Returns `(energy, sweep_energies, converged)`.
        """
        ...

    def tdvp(
        self, hamiltonian: Observable, time: float, steps: int = 1
    ) -> None:
        """Evolve in place by `exp(-iHt)` with `steps` second-order periodic TDVP steps."""
        ...

    def tebd(
        self,
        hamiltonian: Observable,
        time: float,
        steps: int = 1,
        order: int = 2,
    ) -> None:
        """Evolve in place by `exp(-iHt)` with `steps` periodic TEBD steps."""
        ...

    def run(self, circuit: Circuit) -> None:
        """Apply a unitary `circuit` in place on the periodic ring."""
        ...

    def expectation(self, observable: Observable) -> float:
        """Exact expectation value `⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩`."""
        ...

    def expectations(self, observables: Sequence[Observable]) -> list[float]:
        """Exact batched expectation values for every observable."""
        ...

    def amplitude(self, state: BasisState) -> complex:
        """Exact periodic trace amplitude of a basis state."""
        ...

    def to_statevector(self) -> Statevector:
        """Expand to a full `Statevector` (up to 24 qubits)."""
        ...

    def with_charges(self, charges: Sequence[Charge]) -> PeriodicMPS:
        """Convert to a symmetric `PeriodicMPS` conserving `charges`."""
        ...

    def to_dense(self) -> PeriodicMPS:
        """Convert a symmetric periodic MPS back to a dense `PeriodicMPS`."""
        ...

    def bond_sectors(self, bond: int) -> BondSectors | None:
        """Sectors of ring bond `bond` (`n_qubits - 1` is the `(n - 1, 0)` seam bond)."""
        ...

    @overload
    def project(self, qubits: int, outcome: int) -> float: ...
    @overload
    def project(
        self, qubits: Sequence[tuple[int, int]], outcome: None = None
    ) -> float: ...
    def project(
        self, qubits: int | Sequence[tuple[int, int]], outcome: int | None = None
    ) -> float:
        """Post-select onto `|outcome⟩⟨outcome|` on one or more qubits and renormalise."""
        ...

    def sample(self, shots: int) -> NDArray[np.uintp]:
        """Draw `shots` basis-state indices (up to 64 qubits)."""
        ...

    def sample_counts(self, shots: int) -> dict[str, int]:
        """Sample `shots` bitstrings (qubit 0 rightmost; up to 64 qubits)."""
        ...

class GpuStatevector:
    """A statevector in GPU memory, simulated via Vulkan 1.2 compute shaders.

    Works on any NVIDIA, AMD, or Intel GPU with a Vulkan 1.2 driver (no CUDA or ROCm).
    Supports up to 31 qubits across split GPU buffers (memory permitting).
    Available in builds compiled with the `gpu` feature (included in Linux and Windows wheels).
    """

    def __init__(
        self,
        n_qubits: int,
        seed: int | None = None,
        dtype: GpuDType | None = None,
    ) -> None:
        """Initialise `|0…0⟩` on the GPU.

        Args:
            n_qubits: Number of qubits (up to 31, GPU memory permitting).
            seed: Optional RNG seed.
            dtype: `"complex128"`, `"complex64"`, or `None` (automatically chooses
                the GPU's highest supported precision).
        """
        ...

    @staticmethod
    def from_circuit(
        circuit: Circuit,
        seed: int | None = None,
        dtype: GpuDType | None = None,
    ) -> GpuStatevector:
        """Run `circuit` from `|0…0⟩` on the GPU."""
        ...

    @staticmethod
    def from_amplitudes(
        amplitudes: NDArray[np.complex128],
        seed: int | None = None,
        dtype: GpuDType | None = None,
    ) -> GpuStatevector:
        """Upload a 1-D `complex128` NumPy array whose length is a power of two."""
        ...

    @property
    def dtype(self) -> ResolvedGpuDType:
        """Active GPU precision: `"complex128"` or `"complex64"`."""
        ...

    @property
    def n_qubits(self) -> int:
        """Number of qubits."""
        ...

    @property
    def device(self) -> str:
        """Name of the Vulkan GPU hosting this statevector."""
        ...

    def __repr__(self) -> str: ...
    def run(self, circuit: Circuit) -> list[int] | None:
        """Apply `circuit` on the GPU. Returns classical bits if non-unitary, else `None`."""
        ...

    def sample_circuit(self, circuit: Circuit, shots: int) -> dict[str, int]:
        """Run `circuit` `shots` times from this GPU state (unchanged) and return bitstring counts."""
        ...

    def amplitudes(self) -> NDArray[np.complex128]:
        """Download all amplitudes from the GPU as a `complex128` NumPy array."""
        ...

    def to_statevector(self, seed: int | None = None) -> Statevector:
        """Download into a CPU `Statevector`."""
        ...

    def expectation(self, observable: Observable) -> float:
        """Compute `⟨ψ|H|ψ⟩` directly on the GPU (only partial sums are read back)."""
        ...

    def evolve(self, hamiltonian: Observable, time: float) -> None:
        """Evolve in place by exact `exp(-iHt)` on the GPU via Krylov time-stepping."""
        ...

    def apply_pauli_sum(self, hamiltonian: Observable) -> GpuStatevector:
        """Compute `H|ψ⟩` as a new `GpuStatevector` on the GPU."""
        ...

    def expectation_gradient(
        self,
        circuit: Circuit,
        parameters: Sequence[float],
        observable: Observable,
    ) -> tuple[float, NDArray[np.float64]]:
        """Compute `⟨ψ(θ)|H|ψ(θ)⟩` and its gradient by adjoint differentiation on the GPU."""
        ...

    @staticmethod
    def adjoint_gradient(
        circuit: Circuit,
        parameters: Sequence[float],
        final_state: GpuStatevector,
        cotangent: GpuStatevector,
    ) -> NDArray[np.float64]:
        """Compute the adjoint gradient of a real loss on the GPU (see `Statevector.adjoint_gradient`)."""
        ...

    def norm_sqr(self) -> float:
        """Compute `⟨ψ|ψ⟩` on the GPU."""
        ...

    def sample(self, shots: int) -> NDArray[np.uintp]:
        """Draw `shots` basis-state indices on the GPU without downloading the full state."""
        ...

    def sample_counts(self, shots: int) -> dict[str, int]:
        """Sample `shots` bitstrings (qubit 0 rightmost) on the GPU."""
        ...

def gpu_device() -> str | None:
    """Return the name of the Vulkan GPU ketrs would use, or `None` if unavailable."""
    ...

def lowest_eigenstates(
    hamiltonian: Observable,
    n_qubits: int,
    k: int = 1,
    dtype: DType = "complex",
    tolerance: float = 1e-9,
    max_iterations: int = 1000,
    seed: int = 0,
    sector: SectorBasis | None = None,
) -> list[Eigenstate]:
    """Compute the `k` lowest eigenstates of `hamiltonian` by matrix-free Lanczos with deflation.

    Pass `sector` to solve inside a conserved `SectorBasis` subspace.
    """
    ...

def from_qasm(source: str) -> Circuit:
    """Parse an OpenQASM 3 (or 2) program into a `Circuit`.

    Supports registers, `input` parameters, standard gates and `qelib1.inc`,
    `gate` definitions, `def` subroutines, `inv`/`ctrl`/`negctrl`/`pow`
    modifiers, measurement, reset, classical assignments, `if`, `while`, and `for`.
    """
    ...

def qft(n_qubits: int) -> Circuit:
    """Build an `n_qubits`-qubit Quantum Fourier Transform circuit (matching Qiskit's `QFT`)."""
    ...

def ghz(n_qubits: int) -> Circuit:
    """Build a circuit preparing the `n_qubits`-qubit GHZ state `(|0…0⟩ + |1…1⟩) / √2`."""
    ...
