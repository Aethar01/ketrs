"""Python bindings and Qiskit converter, checked against qiskit.quantum_info."""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info import Statevector as QiskitStatevector

import ketrs
from ketrs.qiskit import from_qiskit


def run(qc: QuantumCircuit) -> np.ndarray:
    return ketrs.Statevector.from_circuit(from_qiskit(qc)).amplitudes()


def test_bell_state():
    state = ketrs.Statevector.from_circuit(ketrs.Circuit(2).h(0).cx(0, 1))
    np.testing.assert_allclose(state.amplitudes(), [2**-0.5, 0, 0, 2**-0.5], atol=1e-12)


def test_bad_qubit_raises_value_error():
    with pytest.raises(ValueError, match="out of range"):
        ketrs.Circuit(2).cx(0, 2)


@pytest.mark.parametrize("seed", range(20))
def test_random_circuits_match_qiskit(seed):
    qc = random_circuit(5, depth=8, max_operands=3, seed=seed)
    np.testing.assert_allclose(run(qc), QiskitStatevector(qc).data, atol=1e-10)


def test_qft_matches_qiskit_library():
    qc = QuantumCircuit(5)
    qc.x([0, 2])
    qc.append(QFTGate(5), range(5))
    np.testing.assert_allclose(run(qc), QiskitStatevector(qc).data, atol=1e-10)

    state = ketrs.Statevector.from_amplitudes(QiskitStatevector.from_label("00101").data)
    state.run(ketrs.qft(5))
    np.testing.assert_allclose(state.amplitudes(), QiskitStatevector(qc).data, atol=1e-10)


def test_unitary_gates_use_qiskit_ordering():
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import random_unitary

    qc = QuantumCircuit(3)
    qc.h([0, 1, 2])
    qc.append(UnitaryGate(random_unitary(4, seed=1)), [2, 0])
    qc.append(UnitaryGate(random_unitary(2, seed=2)), [1])
    np.testing.assert_allclose(run(qc), QiskitStatevector(qc).data, atol=1e-10)


def test_expectation_matches_qiskit():
    qc = random_circuit(4, depth=6, max_operands=2, seed=7)
    terms = [("ZZII", 0.5), ("XIXI", -1.25), ("IYZY", 2.0), ("IIII", 0.1)]
    want = QiskitStatevector(qc).expectation_value(SparsePauliOp.from_list(terms)).real
    state = ketrs.Statevector.from_circuit(from_qiskit(qc))
    assert state.expectation(terms) == pytest.approx(want, abs=1e-10)
    assert state.expectation(ketrs.PauliSum(terms)) == pytest.approx(want, abs=1e-10)


def test_sample_counts_are_qiskit_bitstrings():
    qc = QuantumCircuit(3)
    qc.x(0)
    qc.h(2)
    counts = ketrs.Statevector.from_circuit(from_qiskit(qc), seed=1).sample_counts(1000)
    assert set(counts) == {"001", "101"}
    assert sum(counts.values()) == 1000


def test_seeded_sampling_is_reproducible():
    circ = ketrs.Circuit(4).h(0).h(1).h(2).h(3)
    a = ketrs.Statevector.from_circuit(circ, seed=42).sample(50)
    b = ketrs.Statevector.from_circuit(circ, seed=42).sample(50)
    np.testing.assert_array_equal(a, b)


def test_loops_are_rejected():
    qc = QuantumCircuit(1, 1)
    with qc.while_loop((qc.clbits[0], 0)):
        qc.x(0)
        qc.measure(0, 0)
    with pytest.raises(NotImplementedError, match="while_loop"):
        from_qiskit(qc)


def test_nonlinear_parameter_is_rejected():
    from qiskit.circuit import Parameter

    theta = Parameter("theta")
    qc = QuantumCircuit(1)
    qc.rx(theta * theta, 0)
    with pytest.raises(ValueError, match="not linear"):
        from_qiskit(qc)


# --- Pauli rotations, Hamiltonians, real statevectors --------------------------

from qiskit.circuit.library import PauliEvolutionGate  # noqa: E402
from qiskit.synthesis import LieTrotter, SuzukiTrotter  # noqa: E402
from scipy.linalg import expm  # noqa: E402

from ketrs.qiskit import pauli_sum_from_qiskit  # noqa: E402

ISING = SparsePauliOp.from_list(
    [("ZZII", -1.0), ("IZZI", -1.0), ("IIZZ", -1.0), ("XIII", -0.7), ("IXII", -0.7),
     ("IIXI", -0.7), ("IIIX", -0.7), ("YYII", 0.3), ("IIII", 0.25)]
)  # fmt: skip


@pytest.mark.parametrize("gate", ["rxx", "ryy", "rzz", "rzx"])
def test_two_qubit_rotations_match_qiskit(gate):
    qc = random_circuit(3, depth=3, seed=11)
    getattr(qc, gate)(0.83, 2, 0)
    np.testing.assert_allclose(run(qc), QiskitStatevector(qc).data, atol=1e-10)


@pytest.mark.parametrize(
    "synthesis",
    [LieTrotter(reps=3), SuzukiTrotter(order=2, reps=2), SuzukiTrotter(order=4, reps=1)],
    ids=["lie", "suzuki2", "suzuki4"],
)
def test_pauli_evolution_gate_matches_qiskit(synthesis):
    qc = QuantumCircuit(5)
    qc.h(range(5))
    qc.append(PauliEvolutionGate(ISING, time=0.9, synthesis=synthesis), [4, 0, 2, 1])
    # quantum_info would use the exact exp(-iHt); compare with the Trotter
    # circuit the gate synthesises to, which is what Aer and hardware run.
    trotterised = qc.decompose(gates_to_decompose=["PauliEvolution"])
    np.testing.assert_allclose(run(qc), QiskitStatevector(trotterised).data, atol=1e-10)
    # Mapped directly: one Pauli rotation per Trotter term, no CNOT ladders.
    assert len(from_qiskit(qc)) == len(trotterised.data)


def test_evolve_matches_scipy_expm():
    h = pauli_sum_from_qiskit(ISING)
    psi0 = QiskitStatevector(random_circuit(4, depth=4, seed=5)).data
    state = ketrs.Statevector.from_amplitudes(psi0)
    state.evolve(h, 1.3)
    want = expm(-1j * 1.3 * ISING.to_matrix()) @ psi0
    np.testing.assert_allclose(state.amplitudes(), want, atol=1e-10)


def test_apply_pauli_sum_matches_matrix():
    psi0 = QiskitStatevector(random_circuit(4, depth=4, seed=6)).data
    got = ketrs.Statevector.from_amplitudes(psi0).apply_pauli_sum(pauli_sum_from_qiskit(ISING))
    np.testing.assert_allclose(got.amplitudes(), ISING.to_matrix() @ psi0, atol=1e-12)


NUMPY_DTYPE = {"complex": "complex128", "real": "float64", "complex64": "complex64", "float32": "float32"}


@pytest.mark.parametrize("dtype", ["complex", "real", "complex64", "float32"])
def test_lowest_eigenstates_match_numpy(dtype):
    real_ising = SparsePauliOp.from_list([(l, c) for l, c in ISING.to_list() if l != "YYII"])
    h = pauli_sum_from_qiskit(real_ising)
    want = np.linalg.eigvalsh(real_ising.to_matrix().real)[:4]
    found = ketrs.lowest_eigenstates(h, 4, k=4, dtype=dtype)
    single = dtype in ("complex64", "float32")
    np.testing.assert_allclose([e.energy for e in found], want, atol=1e-4 if single else 1e-8)
    assert found[0].state.dtype == NUMPY_DTYPE[dtype]
    assert max(e.residual for e in found) < (1e-3 if single else 1e-7)


def test_single_precision():
    circ = from_qiskit(efficient_su2(6, reps=2))
    x = np.random.default_rng(5).uniform(-2, 2, circ.n_parameters)
    bound = circ.bind(x)
    double = ketrs.Statevector.from_circuit(bound)
    single = ketrs.Statevector.from_circuit(bound, dtype="complex64")
    assert single.dtype == "complex64"
    assert single.amplitudes().dtype == np.complex64
    np.testing.assert_allclose(single.amplitudes(), double.amplitudes(), atol=1e-5)

    h = pauli_sum_from_qiskit(ISING.tensor(SparsePauliOp("II")))
    value, grad = ketrs.Statevector(6, dtype="complex64").expectation_gradient(circ, x, h)
    want_value, want_grad = ketrs.Statevector(6).expectation_gradient(circ, x, h)
    assert value == pytest.approx(want_value, abs=1e-4)
    np.testing.assert_allclose(grad, want_grad, atol=1e-4)

    evolved = ketrs.Statevector.from_amplitudes(double.amplitudes().astype(np.complex64))
    evolved.evolve(h, 1.0)
    want = ketrs.Statevector.from_amplitudes(double.amplitudes())
    want.evolve(h, 1.0)
    np.testing.assert_allclose(evolved.amplitudes(), want.amplitudes(), atol=1e-4)

    real = ketrs.Statevector.from_circuit(ketrs.Circuit(2).h(0).cx(0, 1), dtype="float32")
    assert real.amplitudes().dtype == np.float32
    assert real.inner(real) == pytest.approx(1.0, abs=1e-6)
    with pytest.raises(ValueError, match="complex"):
        real.evolve("ZZ", 1.0)
    with pytest.raises(ValueError, match="different dtypes"):
        single.inner(double)


def test_real_statevector():
    circ = ketrs.Circuit(3).h(0).cx(0, 1).ry(0.4, 2).cz(1, 2)
    real = ketrs.Statevector.from_circuit(circ, dtype="real")
    assert real.amplitudes().dtype == np.float64
    np.testing.assert_allclose(real.amplitudes(), ketrs.Statevector.from_circuit(circ).amplitudes())
    with pytest.raises(ValueError, match="complex"):
        ketrs.Statevector.from_circuit(ketrs.Circuit(1).s(0), dtype="real")
    with pytest.raises(ValueError, match="complex"):
        real.evolve("ZZZ", 1.0)


def test_pauli_sum_round_trip():
    h = pauli_sum_from_qiskit(ISING)
    assert SparsePauliOp.from_list(h.to_list()).simplify() == ISING.simplify()
    assert h.coefficient_norm() == pytest.approx(sum(abs(c) for c in ISING.coeffs))


# --- Parameters and gradients -------------------------------------------------

from qiskit.circuit import Parameter as QiskitParameter  # noqa: E402
from qiskit.circuit.library import efficient_su2, real_amplitudes  # noqa: E402


def finite_difference(f, x, h=1e-6):
    x = np.asarray(x, dtype=float)
    return np.array([(f(x + h * e) - f(x - h * e)) / (2 * h) for e in np.eye(len(x))])


def test_parameter_arithmetic():
    circ = ketrs.Circuit(1)
    theta = circ.parameter()
    circ.rx(2 * theta + 0.5, 0).ry(0.3 - theta, 0).rz(-theta / 4, 0)
    bound = circ.bind([0.7])
    direct = ketrs.Circuit(1).rx(1.9, 0).ry(-0.4, 0).rz(-0.175, 0)
    np.testing.assert_allclose(
        ketrs.Statevector.from_circuit(bound).amplitudes(),
        ketrs.Statevector.from_circuit(direct).amplitudes(),
        atol=1e-14,
    )
    with pytest.raises(ValueError, match="unbound"):
        ketrs.Statevector.from_circuit(circ)


def test_parametric_qiskit_circuit_binds_like_qiskit():
    a, b = QiskitParameter("a"), QiskitParameter("b")
    qc = efficient_su2(4, reps=2)
    qc.rzz(2 * a + 1, 0, 3)
    qc.crx(b - 0.5, 1, 2)
    values = np.random.default_rng(1).uniform(-2, 2, qc.num_parameters)
    circ = from_qiskit(qc)
    assert circ.n_parameters == qc.num_parameters
    np.testing.assert_allclose(
        ketrs.Statevector.from_circuit(circ.bind(values)).amplitudes(),
        QiskitStatevector(qc.assign_parameters(values)).data,
        atol=1e-10,
    )


@pytest.mark.parametrize("dtype", ["complex", "real"])
def test_expectation_gradient_matches_finite_differences(dtype):
    qc = real_amplitudes(5, reps=3) if dtype == "real" else efficient_su2(5, reps=2)
    circ = from_qiskit(qc)
    h = pauli_sum_from_qiskit(
        SparsePauliOp.from_list([("ZZIII", 1.0), ("IXXII", -0.6), ("IIZIZ", 0.4), ("XIIIX", 0.3)])
    )
    x = np.random.default_rng(2).uniform(-2, 2, circ.n_parameters)
    start = ketrs.Statevector(5, dtype=dtype)
    value, grad = start.expectation_gradient(circ, x, h)

    def energy(v):
        return ketrs.Statevector.from_circuit(circ.bind(v), dtype=dtype).expectation(h)

    assert value == pytest.approx(energy(x), abs=1e-12)
    np.testing.assert_allclose(grad, finite_difference(energy, x), atol=1e-6)


def test_adjoint_gradient_for_overlap_loss():
    circ = from_qiskit(efficient_su2(3, reps=2))
    rng = np.random.default_rng(3)
    x = rng.uniform(-2, 2, circ.n_parameters)
    phi = ketrs.Statevector.from_circuit(circ.bind(rng.uniform(-2, 2, circ.n_parameters)))

    def loss(v):
        return abs(phi.inner(ketrs.Statevector.from_circuit(circ.bind(v)))) ** 2

    psi = ketrs.Statevector.from_circuit(circ.bind(x))
    cotangent = ketrs.Statevector.from_amplitudes(phi.amplitudes() * phi.inner(psi))
    grad = ketrs.Statevector.adjoint_gradient(circ, x, psi, cotangent)
    np.testing.assert_allclose(grad, finite_difference(loss, x), atol=1e-6)


# --- Symmetry sectors ------------------------------------------------------------


def heisenberg_op(n: int) -> SparsePauliOp:
    return SparsePauliOp.from_sparse_list(
        [(p + p, [q, q + 1], 1.0) for q in range(n - 1) for p in "XYZ"]
        + [("Z", [q], 0.1 * q) for q in range(n)],
        num_qubits=n,
    )


def test_sector_basis_indexing():
    basis = ketrs.SectorBasis(6, [(range(6), 3)])
    states = basis.states()
    assert len(basis) == 20 and all(bin(s).count("1") == 3 for s in states)
    assert [basis.index(int(s)) for s in states] == list(range(20))
    assert 0b000111 in basis and 0b001111 not in basis


def test_sector_evolution_and_eigenstates_match_full():
    n = 8
    op = heisenberg_op(n)
    h = pauli_sum_from_qiskit(op)
    basis = ketrs.SectorBasis(n, [(range(n), n // 2)])
    neel = 0b01010101

    # A Trotter circuit, run in the sector and on the full space.
    qc = QuantumCircuit(n)
    qc.append(PauliEvolutionGate(op, time=0.8, synthesis=SuzukiTrotter(order=2, reps=3)), range(n))
    circ = from_qiskit(qc)
    sector = ketrs.SectorStatevector(basis, neel)
    sector.run(circ)
    full = ketrs.Statevector(n)
    full.run(ketrs.Circuit(n).x(0).x(2).x(4).x(6))
    full.run(circ)
    np.testing.assert_allclose(sector.to_dense().amplitudes(), full.amplitudes(), atol=1e-10)

    # The lowest energies match the Hamiltonian restricted to the sector.
    states = basis.states()
    block = op.to_matrix()[np.ix_(states, states)]
    want = np.linalg.eigvalsh(block)[:2]
    found = ketrs.lowest_eigenstates(h, n, k=2, dtype="real", sector=basis)
    assert isinstance(found[0].state, ketrs.SectorStatevector)
    np.testing.assert_allclose([e.energy for e in found], want, atol=1e-8)

    # Single precision, within its accuracy.
    found = ketrs.lowest_eigenstates(h, n, k=2, dtype="float32", sector=basis)
    assert found[0].state.dtype == "float32"
    np.testing.assert_allclose([e.energy for e in found], want, atol=1e-4)
    single = ketrs.SectorStatevector(basis, neel, dtype="complex64")
    single.run(circ)
    single.evolve(h, 0.5)
    double = ketrs.SectorStatevector(basis, neel)
    double.run(circ)
    double.evolve(h, 0.5)
    np.testing.assert_allclose(single.amplitudes(), double.amplitudes(), atol=1e-4)
    dense = single.to_dense()
    assert dense.dtype == "complex64"
    assert ketrs.SectorStatevector.from_dense(basis, dense).dtype == "complex64"


def test_sector_rejects_symmetry_breaking():
    basis = ketrs.SectorBasis(4, [(range(4), 2)])
    state = ketrs.SectorStatevector(basis, 0b0011)
    before = state.amplitudes().copy()
    with pytest.raises(ValueError, match="sector"):
        state.run(ketrs.Circuit(4).rx(0.3, 0))
    np.testing.assert_array_equal(state.amplitudes(), before)
    with pytest.raises(ValueError, match="conserve"):
        state.evolve([("XIII", 1.0)], 1.0)


# --- Matrix product states --------------------------------------------------------


def test_mps_matches_statevector_on_qiskit_circuits():
    for seed in range(5):
        qc = random_circuit(6, depth=8, max_operands=3, seed=seed)
        mps = ketrs.MPS.from_circuit(from_qiskit(qc), max_bond=64, cutoff=0.0)
        np.testing.assert_allclose(
            mps.to_statevector().amplitudes(), QiskitStatevector(qc).data, atol=1e-10
        )


def test_mps_large_ghz_and_amplitudes():
    n = 80
    mps = ketrs.MPS.from_circuit(ketrs.ghz(n), seed=1)
    assert max(mps.bond_dimensions) == 2 and mps.truncation_error == 0.0
    assert abs(mps.amplitude("1" * n)) ** 2 == pytest.approx(0.5)
    assert mps.expectation([("Z" + "I" * (n - 2) + "Z", 1.0)]) == pytest.approx(1.0)


def test_mps_tdvp_matches_exact_evolution():
    # Long-range couplings, from a state with full bond dimension.
    n = 6
    terms = [("ZZ", [i, j], 1.0 / (j - i)) for i in range(n) for j in range(i + 1, n)]
    terms += [("X", [q], 0.7) for q in range(n)]
    op = SparsePauliOp.from_sparse_list(terms, num_qubits=n)
    h = pauli_sum_from_qiskit(op)
    mps = ketrs.MPS.from_circuit(from_qiskit(efficient_su2(n, reps=3).assign_parameters(
        np.random.default_rng(4).uniform(-2, 2, efficient_su2(n, reps=3).num_parameters)
    )), max_bond=64)
    want = ketrs.Statevector.from_amplitudes(mps.to_statevector().amplitudes())
    mps.tdvp(h, 0.7, steps=4)
    want.evolve(h, 0.7)
    np.testing.assert_allclose(mps.to_statevector().amplitudes(), want.amplitudes(), atol=1e-9)


def test_mps_tebd_matches_trotter_statevector():
    n = 8
    h = pauli_sum_from_qiskit(heisenberg_op(n))
    mps = ketrs.MPS(n, max_bond=32)
    mps.tebd(h, 0.5, steps=5, order=2)
    circ = ketrs.Circuit(n).pauli_evolution(h, 0.5, steps=5, order=2)
    np.testing.assert_allclose(
        mps.to_statevector().amplitudes(),
        ketrs.Statevector.from_circuit(circ).amplitudes(),
        atol=1e-9,
    )


def test_dmrg_ground_state_matches_numpy():
    n = 10
    op = heisenberg_op(n)
    want = np.linalg.eigvalsh(op.to_matrix())[0]
    mps = ketrs.MPS.random(n, bond=8, seed=4)
    energy, sweeps, converged = mps.dmrg(pauli_sum_from_qiskit(op), max_bond=48)
    assert converged
    assert energy == pytest.approx(want, abs=1e-8)
    assert mps.expectation(pauli_sum_from_qiskit(op)) == pytest.approx(energy, abs=1e-8)


# --- GPU -----------------------------------------------------------------------------

gpu = pytest.mark.skipif(ketrs.gpu_device() is None, reason="no Vulkan GPU with 64-bit floats")


@gpu
def test_gpu_matches_qiskit():
    for seed in range(5):
        qc = random_circuit(8, depth=10, max_operands=3, seed=seed)
        state = ketrs.GpuStatevector.from_circuit(from_qiskit(qc))
        np.testing.assert_allclose(state.amplitudes(), QiskitStatevector(qc).data, atol=1e-10)


@gpu
def test_gpu_run_on_uploaded_state_and_expectation():
    psi0 = QiskitStatevector(random_circuit(6, depth=5, seed=3)).data
    gpu_state = ketrs.GpuStatevector.from_amplitudes(psi0)
    ansatz = from_qiskit(efficient_su2(6, reps=2))
    circ = ansatz.bind(np.linspace(0, 1, ansatz.n_parameters))
    gpu_state.run(circ)
    cpu = ketrs.Statevector.from_amplitudes(psi0)
    cpu.run(circ)
    np.testing.assert_allclose(gpu_state.amplitudes(), cpu.amplitudes(), atol=1e-10)
    assert gpu_state.expectation("ZZIIXI") == pytest.approx(cpu.expectation("ZZIIXI"), abs=1e-10)


@gpu
def test_gpu_expectation_and_sampling():
    qc = random_circuit(8, depth=8, seed=12)
    gpu_state = ketrs.GpuStatevector.from_circuit(from_qiskit(qc), seed=3)
    exact = QiskitStatevector(qc)
    h = pauli_sum_from_qiskit(ISING.tensor(SparsePauliOp("IIII")))
    assert gpu_state.expectation(h) == pytest.approx(exact.expectation_value(ISING.tensor(SparsePauliOp("IIII"))).real, abs=1e-10)
    assert gpu_state.norm_sqr() == pytest.approx(1.0, abs=1e-12)
    counts = gpu_state.sample_counts(50_000)
    probs = exact.probabilities_dict()
    for bits, count in counts.items():
        p = probs[bits]
        assert abs(count - 50_000 * p) < 5 * np.sqrt(50_000 * p * (1 - p)) + 1


# --- Measurement and classical control ------------------------------------------------


def binomial_ok(count, shots, p):
    return abs(count - shots * p) < 5 * np.sqrt(shots * p * (1 - p)) + 1


def test_measure_all_counts_like_qiskit():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure_all()
    circ = from_qiskit(qc)
    assert circ.n_clbits == 3 and not circ.is_unitary
    counts = ketrs.Statevector(3, seed=1).sample_circuit(circ, 4000)
    assert set(counts) == {"000", "111"}
    assert binomial_ok(counts["111"], 4000, 0.5)


def test_qiskit_teleportation_with_if_test():
    from qiskit import ClassicalRegister, QuantumRegister

    theta = 1.3
    q, m, out = QuantumRegister(3), ClassicalRegister(2, "m"), ClassicalRegister(1, "out")
    qc = QuantumCircuit(q, m, out)
    qc.ry(theta, 0)
    qc.h(1)
    qc.cx(1, 2)
    qc.cx(0, 1)
    qc.h(0)
    qc.measure(0, m[0])
    qc.measure(1, m[1])
    with qc.if_test((m[1], 1)):
        qc.x(2)
    with qc.if_test((m[0], 1)):
        qc.z(2)
    qc.measure(2, out[0])
    shots = 20_000
    counts = ketrs.Statevector(3, seed=2).sample_circuit(from_qiskit(qc), shots)
    ones = sum(c for bits, c in counts.items() if bits[0] == "1")  # bit 2 is leftmost
    assert binomial_ok(ones, shots, np.sin(theta / 2) ** 2)


def test_register_condition_else_branch_and_reset():
    from qiskit import ClassicalRegister, QuantumRegister

    q, c = QuantumRegister(3), ClassicalRegister(3, "c")
    qc = QuantumCircuit(q, c)
    qc.x(1)
    qc.measure(0, 0)
    qc.measure(1, 1)
    reg = ClassicalRegister(name="pair", bits=[c[0], c[1]])
    qc.add_register(reg)
    with qc.if_test((reg, 2)) as else_:
        qc.x(2)
    with else_:
        qc.h(2)
    qc.measure(2, 2)
    qc.reset(1)
    counts = ketrs.Statevector(3, seed=3).sample_circuit(from_qiskit(qc), 200)
    assert counts == {"110": 200}

    state = ketrs.Statevector(3, seed=4)
    bits = state.run(from_qiskit(qc))
    assert bits == [0, 1, 1]
    assert state.probability_one(1) == pytest.approx(0.0)  # reset after measuring
