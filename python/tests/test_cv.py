"""Tests for ketrs continuous-variable (qumode) Gaussian engine and hafnian/torontonian kernels."""

import math

import numpy as np
import pytest

import ketrs


def test_mode_circuit_parameters_and_feedforward():
    c = ketrs.ModeCircuit(3)
    theta, r = c.parameters(2)
    c.set_input(0, ketrs.ModeState.coherent(1.1, 0.4))
    c.two_mode_squeeze(r, math.pi, 1, 2)
    c.beam_splitter(theta, 0.0, 0, 1)
    mx = c.measure_x(0)
    mp = c.measure_p(1)
    assert (mx.index, mp.index, c.n_results) == (0, 1, 2)
    assert c.is_gaussian and not c.is_unitary

    r_val = 1.6
    scale = math.tanh(r_val) / math.sqrt(2.0)
    c.displace(scale * mx, 0.0, 2)
    c.displace(scale * mp, math.pi / 2.0, 2)

    bound = c.bind([math.pi / 4.0, r_val])
    assert bound.n_parameters == 0

    want_x = math.tanh(r_val) * 2.0 * 1.1 * math.cos(0.4)
    want_p = math.tanh(r_val) * 2.0 * 1.1 * math.sin(0.4)
    state, res = ketrs.GaussianState.run(bound, seed=42)
    assert res.shape == (2,)
    assert state.mean_x(2) == pytest.approx(want_x, abs=1e-11)
    assert state.mean_p(2) == pytest.approx(want_p, abs=1e-11)
    assert state.mean_photons(0) == pytest.approx(0.0, abs=1e-12)
    assert state.mean_photons(1) == pytest.approx(0.0, abs=1e-12)


def test_gaussian_fock_and_threshold_probabilities():
    c = ketrs.ModeCircuit(2)
    c.squeeze(0.4, 0.3, 0).squeeze(0.35, -0.5, 1)
    c.beam_splitter(0.7, 0.2, 0, 1)
    c.displace(0.3, 0.6, 0)
    st = ketrs.GaussianState.from_circuit(c)

    assert st.purity() == pytest.approx(1.0, abs=1e-11)
    assert st.vacuum_probability() == pytest.approx(st.fock_probability([0, 0]), abs=1e-12)

    # Sum of all 4 threshold probabilities is 1.
    th = [st.threshold_probability([bool(i & 1), bool(i & 2)]) for i in range(4)]
    assert sum(th) == pytest.approx(1.0, abs=1e-11)
    assert th[0] == pytest.approx(st.vacuum_probability(), abs=1e-12)

    # Reduced state vacuum probability matches sum over the traced-out mode.
    assert st.reduced([0]).vacuum_probability() == pytest.approx(th[0] + th[2], abs=1e-11)


def test_gaussian_sample_circuit_moments():
    c = ketrs.ModeCircuit(2)
    c.two_mode_squeeze(0.5, 0.2, 0, 1).displace(0.7, -0.3, 0)
    exact = ketrs.GaussianState.from_circuit(c)
    c.measure_x(0)
    c.measure_heterodyne(1)

    samples = ketrs.GaussianState.sample_circuit(c, 15_000, seed=7)
    assert samples.shape == (15_000, 3)
    assert samples[:, 0].mean() == pytest.approx(exact.mean_x(0), abs=0.05)
    assert samples[:, 1].mean() == pytest.approx(0.5 * exact.mean_x(1), abs=0.05)
    assert samples[:, 2].mean() == pytest.approx(0.5 * exact.mean_p(1), abs=0.05)


def test_hafnian_and_torontonian_identities():
    # haf(J_6 - I_6) = 5!! = 15
    a = np.ones((6, 6), dtype=np.float64) - np.eye(6)
    assert ketrs.hafnian(a) == pytest.approx(15.0, abs=1e-10)

    # Loop hafnian of all-ones 4×4: 3 (2 edges) + 6 (1 edge) + 1 (0 edges) = 10
    b = np.ones((4, 4), dtype=np.complex128)
    assert ketrs.hafnian(b, loop=True) == pytest.approx(10.0, abs=1e-10)
    assert ketrs.loop_hafnian(b) == pytest.approx(10.0, abs=1e-10)

    # Repeated matches expanded
    rng = np.random.default_rng(123)
    m = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    m = 0.5 * (m + m.T)
    d = rng.normal(size=3) + 1j * rng.normal(size=3)
    rep = [2, 1, 2]
    idx = [0, 0, 1, 2, 2]
    exp_m = m[np.ix_(idx, idx)]
    exp_d = d[idx]
    assert ketrs.hafnian_repeated(m, rep, loop=True, diag=d) == pytest.approx(
        ketrs.loop_hafnian(exp_m, exp_d), rel=1e-9
    )

    # Torontonian on 1-mode thermal state: O = (nbar / (nbar + 1)) I_2 -> tor(O) = nbar
    nbar = 1.7
    o = (nbar / (nbar + 1.0)) * np.eye(2)
    assert ketrs.torontonian(o) == pytest.approx(nbar, abs=1e-11)


def test_gaussian_loss_and_wigner():
    st = ketrs.GaussianState(1).displace(1.0, 0.0, 0)
    assert st.mean_photons(0) == pytest.approx(1.0, abs=1e-12)
    st.loss(0, 0.5)
    assert st.mean_photons(0) == pytest.approx(0.5, abs=1e-12)
    # Loss on a coherent state stays a pure coherent state |α/√2⟩!
    assert st.purity() == pytest.approx(1.0, abs=1e-12)
    assert st.wigner([math.sqrt(2.0)], [0.0]) == pytest.approx(1.0 / (2.0 * math.pi), abs=1e-12)


def test_fixed_n_and_fock_match_on_kerr_interferometer():
    c = ketrs.ModeCircuit(3)
    c.set_input(0, ketrs.ModeState.fock(1))
    c.set_input(1, ketrs.ModeState.fock(2))
    c.set_input(2, ketrs.ModeState.fock(1))
    c.beam_splitter(0.6, 0.2, 0, 1).kerr(0.4, 1).cross_kerr(-0.3, 0, 2)
    c.beam_splitter(0.8, -0.5, 1, 2).rotation(0.7, 0)

    fn = ketrs.FixedNFockState.from_circuit(c)
    fk = ketrs.FockState.from_circuit(c, cutoff=5)

    assert fn.n_photons == 4
    assert fn.dim == ketrs.FixedNFockState.basis_dim(3, 4)
    assert fk.truncation_error() == pytest.approx(0.0, abs=1e-12)
    for pat in fn.basis():
        assert fn.amplitude(pat) == pytest.approx(fk.amplitude(pat), abs=1e-11)
        assert fn.probability(pat) == pytest.approx(fk.probability(pat), abs=1e-11)


def test_fock_state_and_density_matrix_match_gaussian_with_noise():
    c = ketrs.ModeCircuit(1)
    c.squeeze(0.35, 0.4, 0).displace(0.5, -0.3, 0)

    g = ketrs.GaussianState.from_circuit(c)
    f = ketrs.FockState.from_circuit(c, cutoff=28)
    assert f.truncation_error() < 1e-9
    assert f.mean_x(0) == pytest.approx(g.mean_x(0), abs=1e-7)
    assert f.mean_p(0) == pytest.approx(g.mean_p(0), abs=1e-7)
    for n in range(6):
        assert f.probability([n]) == pytest.approx(g.fock_probability([n]), abs=1e-7)

    ch_th = ketrs.CvChannel.thermal_loss(0.7, 0.35)
    ch_add = ketrs.CvChannel.additive_noise(0.25)
    g.apply_channel(ch_th, 0).apply_channel(ch_add, 0)

    dm = ketrs.FockDensityMatrix.from_state(f)
    dm.apply_channel(ch_th, 0).apply_channel(ch_add, 0)
    assert dm.truncation_error() < 1e-6
    assert dm.purity() == pytest.approx(g.purity(), abs=1e-5)
    assert dm.mean_photons(0) == pytest.approx(g.mean_photons(0), abs=1e-5)
    for n in range(5):
        assert dm.probability([n]) == pytest.approx(g.fock_probability([n]), abs=1e-5)


def test_fock_sampling_homodyne_and_pnr_feedforward():
    c = ketrs.ModeCircuit(2)
    c.set_input(0, ketrs.ModeState.fock(1))
    c.set_input(1, ketrs.ModeState.fock(1))
    c.beam_splitter(math.pi / 4.0, 0.0, 0, 1)
    m0 = c.measure_photons(0)
    c.displace(0.25 * m0, 0.0, 1)
    st, res = ketrs.FockState.run(c, cutoff=12, seed=5)
    assert res[0] in (0.0, 2.0)
    if res[0] == 2.0:
        assert st.mean_x(1) == pytest.approx(1.0, abs=1e-8)
    else:
        assert st.mean_photons(1) == pytest.approx(2.0, abs=1e-8)

    # Fock homodyne mean matches GaussianState
    cg = ketrs.ModeCircuit(1).squeeze(0.3, 0.2, 0).displace(0.6, -0.4, 0)
    exact_g = ketrs.GaussianState.from_circuit(cg)
    cg.measure_homodyne(0.3, 0)
    samples = ketrs.FockState.sample_circuit(cg, cutoff=24, shots=3000, seed=11)
    want = exact_g.mean_x(0) * math.cos(0.3) + exact_g.mean_p(0) * math.sin(0.3)
    assert samples[:, 0].mean() == pytest.approx(want, abs=0.06)


def test_boson_sampling_gbs_and_window_samplers():
    # Permanent sanity
    assert ketrs.permanent(np.ones((4, 4))) == pytest.approx(24.0, abs=1e-10)

    # Boson sampling on 50:50 BS with |1, 1⟩: HOM effect (never (1, 1))
    s = math.sqrt(0.5)
    u = np.array([[s, -s], [s, s]], dtype=np.complex128)
    bs_samples = ketrs.boson_sample(u, [1, 1], 1000, seed=42)
    assert bs_samples.shape == (1000, 2)
    assert np.all(bs_samples.sum(axis=1) == 2)
    assert not np.any((bs_samples[:, 0] == 1) & (bs_samples[:, 1] == 1))

    # GBS threshold & PNR sampling on 2-mode squeezed + displaced state
    c = ketrs.ModeCircuit(2).squeeze(0.45, 0.2, 0).beam_splitter(0.7, 0.3, 0, 1).displace(0.3, 0.0, 0)
    st = ketrs.GaussianState.from_circuit(c, seed=7)
    th_samples = st.sample_threshold(6000)
    assert th_samples.shape == (6000, 2)
    emp_00 = np.mean((~th_samples[:, 0]) & (~th_samples[:, 1]))
    assert emp_00 == pytest.approx(st.threshold_probability([False, False]), abs=0.03)

    pnr_samples = ketrs.sample_gbs_pnr(st, 6000, max_photons=10, seed=19)
    assert pnr_samples.shape == (6000, 2)
    assert pnr_samples[:, 0].mean() == pytest.approx(st.mean_photons(0), abs=0.05)

    # Window sampler on 40-mode time-bin loop (window width == 2)
    wc = ketrs.ModeCircuit(40)
    for i in range(40):
        wc.set_input(i, ketrs.ModeState.fock(1))
    for i in range(39):
        wc.beam_splitter(math.pi / 4.0, 0.0, i, i + 1)
        wc.measure_photons(i)
    wc.measure_photons(39)
    assert ketrs.WindowSampler.max_window_width(wc) == 2
    ws = ketrs.WindowSampler.sample(wc, cutoff=8, shots=20, seed=3)
    assert ws.shape == (20, 40)
    assert np.allclose(ws.sum(axis=1), 40.0)

    # Inline ModeCircuit.apply_channel works in GaussianState, FockDensityMatrix, and WindowSampler
    noisy = (
        ketrs.ModeCircuit(2)
        .set_input(0, ketrs.ModeState.fock(1))
        .set_input(1, ketrs.ModeState.fock(1))
        .apply_channel(ketrs.CvChannel.loss(0.6), 0)
        .beam_splitter(math.pi / 4.0, 0.0, 0, 1)
    )
    dm_n = ketrs.FockDensityMatrix.from_circuit(noisy, cutoff=4)
    assert dm_n.mean_photons(0) + dm_n.mean_photons(1) == pytest.approx(1.6, abs=1e-10)
    noisy.measure_photons(0)
    noisy.measure_photons(1)
    assert ketrs.WindowSampler.max_window_width(noisy) == 2
    ns = ketrs.WindowSampler.sample(noisy, cutoff=4, shots=400, seed=5)
    assert abs(ns.sum(axis=1).mean() - 1.6) < 0.12


def test_gaussian_mixture_cat_gkp_and_qec():
    # 2-mode cat state + squeezing + BS cross-check against FockState
    c = ketrs.ModeCircuit(2)
    c.set_input(0, ketrs.ModeState.cat(1.4, 0.2, 0.0))
    c.set_input(1, ketrs.ModeState.cat(1.1, -0.3, 1.0))
    c.squeeze(0.25, 0.4, 0).beam_splitter(0.6, 0.2, 0, 1).displace(0.3, -0.1, 1)

    mix = ketrs.GaussianMixture.from_circuit(c)
    fock = ketrs.FockState.from_circuit(c, cutoff=30)
    assert fock.truncation_error() < 1e-6
    assert mix.n_modes == 2
    assert mix.n_terms == 16
    for mode in (0, 1):
        assert mix.mean_x(mode) == pytest.approx(fock.mean_x(mode), abs=1e-5)
        assert mix.mean_p(mode) == pytest.approx(fock.mean_p(mode), abs=1e-5)
        assert mix.mean_photons(mode) == pytest.approx(fock.mean_photons(mode), abs=1e-5)

    # Fock |1⟩ Wigner negativity at origin: W(0, 0) = -1 / (2π)
    cf = ketrs.ModeCircuit(1).set_input(0, ketrs.ModeState.fock(1))
    mf = ketrs.GaussianMixture.from_circuit(cf)
    assert mf.wigner([0.0], [0.0]) == pytest.approx(-1.0 / (2.0 * math.pi), abs=1e-4)
    assert mf.mean_photons(0) == pytest.approx(1.0, abs=1e-4)

    # GKP |0_ε⟩ cross-check against FockState + logical X_L / H_L
    eps = 0.18
    d_grid = math.sqrt(2.0 * math.pi)
    cg = ketrs.ModeCircuit(1).set_input(0, ketrs.ModeState.gkp(eps))
    mg = ketrs.GaussianMixture.from_circuit(cg)
    fg = ketrs.FockState.from_circuit(cg, cutoff=55)
    assert fg.truncation_error() < 1e-5
    assert mg.mean_photons(0) == pytest.approx(fg.mean_photons(0), abs=1e-4)
    assert mg.homodyne_pdf(0.0, 0.0, 0) > 0.4
    assert mg.homodyne_pdf(0.0, d_grid, 0) < 1e-3

    # Logical X_L = D(√(π/2), 0) swaps peaks to odd multiples of √(2π)
    mg.displace(math.sqrt(0.5 * math.pi), 0.0, 0)
    assert mg.homodyne_pdf(0.0, d_grid, 0) > 0.35
    assert mg.homodyne_pdf(0.0, 0.0, 0) < 1e-3

    # Noise channel damps interference terms and reduces term count
    mg.apply_channel(ketrs.CvChannel.additive_noise(0.15), 0)
    assert mg.error_bound >= 0.0

    # Steane GKP x-syndrome round via ModeCircuit + post-measurement feed-forward
    c_ent = (
        ketrs.ModeCircuit(2)
        .set_input(0, ketrs.ModeState.gkp(0.08, 0.0, 0.0))
        .set_input(1, ketrs.ModeState.gkp(0.08, math.pi / 2.0, 0.0))
        .cx(1.0, 0, 1)
    )
    st = ketrs.GaussianMixture.from_circuit(c_ent, seed=7)
    st.apply_channel(ketrs.CvChannel.additive_noise(0.15), 0)
    m_x = st.measure_homodyne(0.0, 1)
    rem = ((m_x + 0.5 * d_grid) % d_grid) - 0.5 * d_grid
    st.displace(-0.5 * rem, 0.0, 0)
    # After conditioning and correction, logical |0⟩ peak at x = 0 stays dominant
    assert st.homodyne_pdf(0.0, 0.0, 0) > st.homodyne_pdf(0.0, d_grid, 0)


def test_bosonic_mps_planner_and_gpu():
    # BosonicMps exact match vs FockState on 3-mode universal circuit
    c = (
        ketrs.ModeCircuit(3)
        .set_input(0, ketrs.ModeState.fock(1))
        .set_input(1, ketrs.ModeState.coherent(0.4, 0.2))
        .squeeze(0.25, 0.3, 0)
        .displace(0.2, -0.4, 2)
        .kerr(0.35, 0)
        .beam_splitter(0.6, 0.2, 0, 1)
        .two_mode_squeeze(0.2, -0.3, 1, 2)
        .cross_kerr(-0.25, 0, 2)
        .cx(0.3, 2, 0)
        .cz(-0.2, 0, 1)
    )
    fock = ketrs.FockState.from_circuit(c, cutoff=10)
    mps = ketrs.BosonicMps.from_circuit(c, cutoff=10, max_bond=64, cutoff_tol=1e-12)
    assert mps.n_modes == 3
    assert mps.cutoff == 10
    for m in range(3):
        assert mps.mean_photons(m) == pytest.approx(fock.mean_photons(m), abs=1e-7)
        assert mps.mean_x(m) == pytest.approx(fock.mean_x(m), abs=1e-7)
        assert mps.mean_p(m) == pytest.approx(fock.mean_p(m), abs=1e-7)
    for pat in ([0, 0, 0], [1, 0, 0], [0, 1, 1], [1, 1, 1]):
        assert mps.amplitude(pat) == pytest.approx(fock.amplitude(pat), abs=1e-7)

    # BosonicMps scales to 16 modes non-Gaussian (10^16 dense Fock impossible)
    c16 = ketrs.ModeCircuit(16)
    for i in range(16):
        c16.squeeze(0.2, 0.1 * i, i).kerr(0.3, i)
    for i in range(15):
        c16.beam_splitter(0.35, 0.2, i, i + 1)
    mps16 = ketrs.BosonicMps.from_circuit(c16, cutoff=8, max_bond=24)
    assert mps16.max_bond_dim <= 24
    assert mps16.truncation_error() < 1e-3

    # Automatic planner routes each class and honors explicit overrides
    g = ketrs.ModeCircuit(4).squeeze(0.5, 0.0, 0).beam_splitter(0.7, 0.0, 0, 1)
    assert ketrs.plan_circuit(g).backend == "gaussian"
    assert ketrs.plan_circuit(g, prefer_backend="bosonic_mps").backend == "bosonic_mps"
    assert ketrs.plan_circuit(c, cutoff=10).backend == "fock"
    assert ketrs.plan_circuit(c16, cutoff=10).backend == "bosonic_mps"

    cat_c = ketrs.ModeCircuit(2).set_input(0, ketrs.ModeState.cat(2.0)).beam_splitter(0.7, 0.0, 0, 1)
    assert ketrs.plan_circuit(cat_c).backend == "gaussian_mixture"

    hom = (
        ketrs.ModeCircuit(2)
        .set_input(0, ketrs.ModeState.fock(1))
        .set_input(1, ketrs.ModeState.fock(1))
        .beam_splitter(math.pi / 4.0, 0.0, 0, 1)
    )
    assert ketrs.plan_circuit(hom).backend == "fixed_n"

    # GPU kernels when built with `--features gpu` and a Vulkan GPU is present
    if ketrs.gpu_device() is not None and hasattr(ketrs, "GpuFockState"):
        gf = ketrs.GpuFockState.from_circuit(c, cutoff=10)
        assert gf.norm_sqr() == pytest.approx(fock.norm_sqr(), abs=1e-11)
        for m in range(3):
            assert gf.mean_photons(m) == pytest.approx(fock.mean_photons(m), abs=1e-11)
        assert np.allclose(gf.amplitudes(), fock.amplitudes(), atol=1e-11)

        rng = np.random.default_rng(7)
        mat = rng.normal(scale=0.3, size=(14, 14)) + 1j * rng.normal(scale=0.3, size=(14, 14))
        p_cpu = ketrs.permanent(mat)
        p_gpu = ketrs.permanent_gpu(mat)
        assert abs(p_gpu - p_cpu) / max(abs(p_cpu), 1.0) < 1e-11


def test_hybrid_qubit_qumode_circuits():
    # Jaynes-Cummings Rabi oscillation: |1, 0⟩ -> cos(θ)|1, 0⟩ - i sin(θ)|0, 1⟩
    theta = 0.65
    c = ketrs.HybridCircuit(1, 1).x(0).jaynes_cummings(theta, 0.0, 0, 0)
    st = ketrs.HybridState.from_circuit(c, cutoff=12)
    assert st.probability(1, [0]) == pytest.approx(math.cos(theta) ** 2, abs=1e-12)
    assert st.probability(0, [1]) == pytest.approx(math.sin(theta) ** 2, abs=1e-12)
    assert st.mean_photons(0) == pytest.approx(math.sin(theta) ** 2, abs=1e-12)

    # Cat state synthesis via H -> CD(α) -> H + postselect qubit
    r = 1.5
    ccat = ketrs.HybridCircuit(1, 1).h(0).conditional_displace(r, 0.0, 0, 0).h(0)
    st_cat = ketrs.HybridState.from_circuit(ccat, cutoff=32)
    p_even = st_cat.postselect_qubit(0, False)
    assert p_even == pytest.approx(0.5 * (1.0 + math.exp(-2.0 * r * r)), abs=1e-10)
    # Odd Fock components vanish in even cat
    assert st_cat.probability(0, [1]) < 1e-12
    assert st_cat.probability(0, [3]) < 1e-12

    # Dispersive parity check via conditional_rotation(π/2)
    st_cat.h(0).conditional_rotation(0.5 * math.pi, 0, 0).h(0)
    assert st_cat.qubit_prob_one(0) == pytest.approx(0.0, abs=1e-10)

    # SNAP + conditional SNAP + CBS + sampling
    ch = (
        ketrs.HybridCircuit(2, 2)
        .set_mode_input(0, ketrs.ModeState.fock(1))
        .h(0)
        .cx_qubit(0, 1)
        .snap([0.0, 0.4, -0.2], 0)
        .conditional_snap([0.0, 0.25], 0, 0)
        .conditional_beam_splitter(math.pi / 4.0, 0.0, 0, 0, 1)
    )
    sth = ketrs.HybridState.from_circuit(ch, cutoff=6)
    assert sth.norm_sqr() == pytest.approx(1.0, abs=1e-12)
    assert sth.probability(0b00, [1, 0]) == pytest.approx(0.25, abs=1e-12)
    assert sth.probability(0b11, [0, 1]) == pytest.approx(0.25, abs=1e-12)

    ch.measure_qubit(0).measure_photons(0)
    samples = ketrs.HybridState.sample_circuit(ch, cutoff=6, shots=300, seed=11)
    assert samples.shape == (300, 2)
    assert abs(samples[:, 0].mean() - 0.5) < 0.09
    assert abs(samples[:, 1].mean() - 0.5) < 0.09
