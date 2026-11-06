"""Convert Qiskit circuits to ketrs circuits.

Gates ketrs supports directly are mapped one to one, including Pauli
rotations (``rxx``, ``rzz``, …) and Trotterised ``PauliEvolutionGate``s, which
ketrs applies in one pass each rather than as CNOT ladders. Controlled
gates with any number of controls on a fixed single-qubit gate (``ccx``,
``mcx``, ``ch``, ``mcphase`` with a bound angle, …) and controlled swaps
(``cswap``) map onto ketrs' multi-controlled operations. Anything else is
expanded through its Qiskit definition until only supported gates remain, so
other gates work without transpiling first.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ketrs import Channel, Circuit, NoiseModel, Parameter, PauliSum

if TYPE_CHECKING:
    from qiskit import QuantumCircuit
    from qiskit.circuit import Instruction
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_aer.noise import NoiseModel as AerNoiseModel

# Gate name -> number of parameters, for gates whose ketrs builder method has
# the same name and takes (params..., qubits...).
_DIRECT = {
    "h": 0, "x": 0, "y": 0, "z": 0, "s": 0, "sdg": 0, "t": 0, "tdg": 0,
    "rx": 1, "ry": 1, "rz": 1, "p": 1, "u": 3,
    "cx": 0, "cy": 0, "cz": 0, "swap": 0,
    "cp": 1, "crx": 1, "cry": 1, "crz": 1,
    "rxx": 1, "ryy": 1, "rzz": 1, "rzx": 1,
}  # fmt: skip

# Instructions that don't change the state.
_IGNORED = {"barrier", "id", "delay"}


def from_qiskit(qc: QuantumCircuit) -> Circuit:
    """Convert a Qiskit ``QuantumCircuit`` to a ketrs ``Circuit``.

    Qiskit parameters become ketrs parameters in the order of
    ``qc.parameters``, so ``circuit.bind(values)`` takes the same values as
    ``qc.assign_parameters(values)``. Gate angles may be any expression
    linear in one parameter, such as ``2*θ + 1``.

    Measurements, resets, ``if_test`` and ``while_loop`` blocks (conditioned
    on a bit, a register or a classical expression on them), ``break_loop``,
    ``continue_loop`` and ``store`` into a classical bit are converted; ``for_loop`` blocks are unrolled,
    or kept as one body per pass when they break or continue. ``switch``
    raises ``NotImplementedError``. Instructions whose
    name starts with ``save_`` (Aer's ``save_statevector`` and friends) are
    skipped.
    """
    circ = Circuit(qc.num_qubits, qc.num_clbits)
    parameters = dict(zip(qc.parameters, circ.parameters(len(qc.parameters))))
    phase = _append_all(circ, qc, list(range(qc.num_qubits)), list(range(qc.num_clbits)), parameters)
    # Add rather than assign: Pauli evolutions may already have added phase.
    circ.global_phase += phase
    return circ


def _append_all(
    circ: Circuit, qc: QuantumCircuit, qubit_map: list[int], clbit_map: list[int], parameters: dict
) -> float:
    """Append every instruction of ``qc``, whose qubit and classical bit k are
    ``qubit_map[k]`` and ``clbit_map[k]`` in ``circ``. Returns the total
    global phase."""
    phase = _float(qc.global_phase)
    for inst in qc.data:
        qubits = [qubit_map[qc.find_bit(q).index] for q in inst.qubits]
        clbits = [clbit_map[qc.find_bit(c).index] for c in inst.clbits]
        context = (qc, clbit_map, parameters)
        phase += _append(circ, inst.operation, qubits, clbits, context)
    return phase


def _append(circ: Circuit, op: Instruction, qubits: list[int], clbits: list[int], context) -> float:
    qc, clbit_map, parameters = context
    name = op.name
    if name in _IGNORED or name.startswith("save_"):
        return 0.0
    if name in _DIRECT and len(op.params) == _DIRECT[name]:
        angles = [_angle(p, parameters) for p in op.params]
        if name == "u" and any(isinstance(a, Parameter) for a in angles):
            raise NotImplementedError("parametric u gates are not supported; use rx/ry/rz")
        getattr(circ, name)(*angles, *qubits)
        return 0.0
    if name == "unitary" and len(qubits) in (1, 2):
        matrix = np.asarray(op.to_matrix(), dtype=complex).tolist()
        if len(qubits) == 1:
            circ.unitary(matrix, qubits[0])
        else:
            circ.unitary2(matrix, qubits[0], qubits[1])
        return 0.0
    if name == "PauliEvolution" and _append_pauli_evolution(circ, op, qubits):
        return 0.0
    if _append_controlled(circ, op, qubits):
        return 0.0
    if name == "measure":
        circ.measure(qubits[0], clbits[0])
        return 0.0
    if name == "reset":
        circ.reset(qubits[0])
        return 0.0
    if name == "store":
        _append_store(circ, op, context)
        return 0.0
    if name == "if_else":
        _append_if_else(circ, op, qubits, clbits, context)
        return 0.0
    if name == "while_loop":
        condition = _condition(op.condition, context)
        circ.while_loop(condition, _block(circ, op.params[0], qubits, clbits, parameters))
        return 0.0
    if name == "for_loop":
        # Unrolled, binding the loop parameter on each pass.
        indices, loop_parameter, body = op.params
        blocks = [
            body if loop_parameter is None else body.assign_parameters({loop_parameter: index}) for index in indices
        ]
        if _breaks(body):
            circ.for_loop([_block(circ, b, qubits, clbits, parameters) for b in blocks])
            return 0.0
        return sum(_append_all(circ, b, qubits, clbits, parameters) for b in blocks)
    if name == "break_loop":
        circ.break_loop()
        return 0.0
    if name == "continue_loop":
        circ.continue_loop()
        return 0.0
    if name == "switch_case":
        raise NotImplementedError(f"cannot convert {name!r}")
    definition = op.definition
    if definition is None:
        raise NotImplementedError(f"cannot convert {name!r}: no Qiskit definition")
    return _append_all(circ, definition, qubits, clbits, parameters)


def _breaks(body: QuantumCircuit) -> bool:
    """Whether a loop body breaks or continues its own loop."""
    for inst in body.data:
        name = inst.operation.name
        if name in ("break_loop", "continue_loop"):
            return True
        if name == "if_else" and any(b is not None and _breaks(b) for b in inst.operation.params):
            return True
    return False


def _append_if_else(circ: Circuit, op: Instruction, qubits: list[int], clbits: list[int], context) -> None:
    """An ``if_test`` block. Its branches use only the qubits and classical
    bits of the instruction; its condition names bits of the enclosing
    circuit."""
    _, _, parameters = context
    true_body, false_body = op.params
    then = _block(circ, true_body, qubits, clbits, parameters)
    otherwise = None if false_body is None else _block(circ, false_body, qubits, clbits, parameters)
    circ.if_expr(_condition(op.condition, context), then, otherwise)


def _append_store(circ: Circuit, op: Instruction, context) -> None:
    """A ``store`` into a classical bit: ``qc.store(clbit, expression)``."""
    from qiskit.circuit import Clbit
    from qiskit.circuit.classical import expr

    qc, clbit_map, _ = context
    target = op.lvalue.var if isinstance(op.lvalue, expr.Var) else None
    if not isinstance(target, Clbit):
        raise NotImplementedError(f"can only store into single classical bits, not {op.lvalue!r}")
    circ.assign(clbit_map[qc.find_bit(target).index], _condition(op.rvalue, context))


def _block(circ: Circuit, body: QuantumCircuit, qubits: list[int], clbits: list[int], parameters) -> Circuit:
    """A control-flow block as a circuit on all of ``circ``'s bits."""
    out = Circuit(circ.n_qubits, circ.n_clbits)
    out.global_phase += _append_all(out, body, qubits, clbits, parameters)
    return out


def _condition(condition, context) -> tuple:
    """A Qiskit condition (a ``(bit or register, value)`` pair or a
    classical expression) as a ketrs expression tuple."""
    from qiskit.circuit import Clbit, ClassicalRegister
    from qiskit.circuit.classical import expr

    qc, clbit_map, _ = context

    def bits(target):
        if isinstance(target, Clbit):
            return ("bit", clbit_map[qc.find_bit(target).index])
        if isinstance(target, ClassicalRegister):
            return ("register", [clbit_map[qc.find_bit(b).index] for b in target])
        raise NotImplementedError(f"unsupported classical variable {target!r}")

    unary = {expr.Unary.Op.LOGIC_NOT: "not"}
    binary = {
        expr.Binary.Op.LOGIC_AND: "and",
        expr.Binary.Op.LOGIC_OR: "or",
        expr.Binary.Op.BIT_AND: "bit_and",
        expr.Binary.Op.BIT_OR: "bit_or",
        expr.Binary.Op.BIT_XOR: "bit_xor",
        expr.Binary.Op.EQUAL: "==",
        expr.Binary.Op.NOT_EQUAL: "!=",
        expr.Binary.Op.LESS: "<",
        expr.Binary.Op.LESS_EQUAL: "<=",
        expr.Binary.Op.GREATER: ">",
        expr.Binary.Op.GREATER_EQUAL: ">=",
    }

    def convert(e):
        if isinstance(e, expr.Var):
            return bits(e.var)
        if isinstance(e, expr.Value):
            return ("value", int(e.value))
        if isinstance(e, expr.Cast):
            # Bool ↔ unsigned integer casts keep the value (nonzero is true).
            return convert(e.operand)
        if isinstance(e, expr.Unary) and e.op == expr.Unary.Op.BIT_NOT:
            return ("bit_not", convert(e.operand), e.type.width)
        if isinstance(e, expr.Unary) and e.op in unary:
            return (unary[e.op], convert(e.operand))
        if isinstance(e, expr.Binary) and e.op in binary:
            return (binary[e.op], convert(e.left), convert(e.right))
        raise NotImplementedError(f"unsupported classical expression {e!r}")

    if isinstance(condition, tuple):
        target, value = condition
        return ("==", bits(target), ("value", int(value)))
    return convert(condition)


def pauli_sum_from_qiskit(
    op: SparsePauliOp, qubits: list[int] | None = None, n_qubits: int | None = None
) -> PauliSum:
    """Convert a Hermitian ``SparsePauliOp`` to a ketrs ``PauliSum``.

    Qubit ``k`` of ``op`` maps to ``qubits[k]`` (default: ``k``) in a register
    of ``n_qubits`` (default: just wide enough).
    """
    qubits = list(range(op.num_qubits)) if qubits is None else list(qubits)
    width = n_qubits if n_qubits is not None else max(qubits, default=-1) + 1
    terms = []
    for label, coeff in op.to_list():
        if abs(complex(coeff).imag) > 1e-12:
            raise ValueError(f"coefficient {coeff} of {label} is not real; PauliSum is Hermitian")
        chars = ["I"] * width
        for k, ch in enumerate(reversed(label)):
            chars[width - 1 - qubits[k]] = ch
        terms.append(("".join(chars), float(complex(coeff).real)))
    return PauliSum(terms)


def noise_model_from_aer(noise_model: AerNoiseModel) -> NoiseModel:
    """Convert a Qiskit Aer ``NoiseModel`` with the same noise on every
    qubit to a ketrs ``NoiseModel``.

    All one-qubit gate errors must be the same channel, as must all
    two-qubit ones, and the readout error must be one for all qubits.
    ketrs applies the one-qubit channel after every gate (on each of its
    qubits) and the two-qubit one after every two-qubit gate, where Aer
    applies them only after the instructions they were added for, so the
    two agree on circuits made of those instructions (for example,
    transpiled to the model's basis gates). Raises ``NotImplementedError``
    for qubit-specific errors and for differing errors on different gates.
    """
    from qiskit.quantum_info import Kraus

    # Aer keeps these as attributes without public accessors.
    if noise_model._local_quantum_errors or noise_model._local_readout_errors:
        raise NotImplementedError("qubit-specific errors are not supported")
    channels = {}
    for name, error in noise_model._default_quantum_errors.items():
        kraus = Kraus(error)
        k = error.num_qubits
        if k not in (1, 2):
            raise NotImplementedError(f"the error on {name!r} acts on {k} qubits")
        if k in channels and not _same_channel(channels[k][1], kraus):
            raise NotImplementedError(
                f"{name!r} and {channels[k][0]!r} have different {k}-qubit errors; "
                "ketrs applies one per gate size"
            )
        channels.setdefault(k, (name, kraus))

    def channel(k):
        if k not in channels:
            return None
        return Channel.from_kraus([np.asarray(m, dtype=complex) for m in channels[k][1].data])

    readout = None
    if noise_model._default_readout_error is not None:
        p = np.asarray(noise_model._default_readout_error.probabilities)
        if p.shape != (2, 2):
            raise NotImplementedError("only one-qubit readout errors are supported")
        readout = (float(p[0, 1]), float(p[1, 0]))
    return NoiseModel(one_qubit=channel(1), two_qubit=channel(2), readout=readout)


def _same_channel(a, b) -> bool:
    from qiskit.quantum_info import SuperOp

    return np.allclose(SuperOp(a).data, SuperOp(b).data, atol=1e-12)


def _append_controlled(circ: Circuit, op: Instruction, qubits: list[int]) -> bool:
    """Append a bound gate controlled on all ones (a ``ControlledGate``, or
    an ``AnnotatedOperation`` with one control modifier) as a ketrs
    multi-controlled gate or swap. Returns False if it isn't one."""
    from qiskit.circuit import AnnotatedOperation, ControlledGate, ControlModifier

    if isinstance(op, ControlledGate):
        base, k, state = op.base_gate, op.num_ctrl_qubits, op.ctrl_state
    elif (
        isinstance(op, AnnotatedOperation)
        and len(op.modifiers) == 1
        and isinstance(op.modifiers[0], ControlModifier)
    ):
        base, k, state = op.base_op, op.modifiers[0].num_ctrl_qubits, op.modifiers[0].ctrl_state
    else:
        return False
    if state != (1 << k) - 1 or any(not _is_bound(p) for p in op.params):
        return False
    controls, rest = qubits[:k], qubits[k:]
    if base.name == "swap" and len(rest) == 2:
        circ.controlled_swap(controls, *rest)
        return True
    if len(rest) != 1:
        return False
    # The block of the full matrix where every control is set keeps phases
    # that some gates (such as ``cu``'s γ) carry outside their base gate.
    # Controls are the low bits of the index and the target the next one.
    try:
        if op.num_qubits > 8:
            raise ValueError("too large to build")
        ones = (1 << k) - 1
        idx = [ones, ones | 1 << k]
        matrix = op.to_matrix()[np.ix_(idx, idx)]
    except Exception:
        try:
            matrix = base.to_matrix()
        except Exception:
            return False
    circ.multi_controlled(np.asarray(matrix, dtype=complex).tolist(), controls, rest[0])
    return True


def _is_bound(value) -> bool:
    try:
        complex(value)
    except TypeError:
        return not hasattr(value, "parameters")
    return True


def _append_pauli_evolution(circ: Circuit, op: Instruction, qubits: list[int]) -> bool:
    """Append a PauliEvolutionGate as ketrs Pauli rotations, if its synthesis
    is a plain Trotter–Suzuki product formula that ketrs reproduces exactly.
    Returns False to fall back to Qiskit's own decomposition.

    Like Aer and hardware, this follows the gate's Trotter synthesis.
    (``qiskit.quantum_info.Statevector`` instead uses the exact exp(-iHt);
    for that, use ``ketrs.Statevector.evolve``.)
    """
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_aer.noise import NoiseModel as AerNoiseModel
    from qiskit.synthesis import SuzukiTrotter

    synthesis = op.synthesis
    if (
        not isinstance(synthesis, SuzukiTrotter)
        or not synthesis.preserve_order
        or synthesis.atomic_evolution is not None
        or not isinstance(op.operator, SparsePauliOp)
    ):
        return False
    try:
        hamiltonian = pauli_sum_from_qiskit(op.operator, qubits, circ.n_qubits)
        time = _float(op.time)
    except ValueError:
        # Complex coefficients or a parametric time: use Qiskit's synthesis.
        return False
    circ.pauli_evolution(hamiltonian, time, synthesis.reps, synthesis.order)
    return True


def _angle(value, parameters: dict) -> float | Parameter:
    """A number, or a ketrs parameter form for an expression linear in one
    Qiskit parameter."""
    try:
        return float(value)
    except TypeError:
        pass
    free = list(value.parameters)
    if len(free) != 1 or free[0] not in parameters:
        raise ValueError(f"angle {value} must depend on exactly one circuit parameter")
    (p,) = free
    try:
        scale = float(value.gradient(p))
        offset = float(value.assign(p, 0.0))
    except TypeError:
        raise ValueError(f"angle {value} is not linear in {p}") from None
    return scale * parameters[p] + offset


def _float(value) -> float:
    try:
        return float(value)
    except TypeError:
        raise ValueError(f"unbound parameter {value!r}; bind parameters first") from None
