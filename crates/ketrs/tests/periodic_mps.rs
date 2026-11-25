//! Rigorous verification of periodic-trace MPS (`PeriodicMps`) against exact
//! `Statevector` diagonalization and evolution:
//!
//! - Exact trace norm, inner products, amplitudes, and seam-crossing expectations.
//! - Periodic circuits and TEBD (direct `(n - 1, 0)` seam gates).
//! - Periodic DMRG (dense + symmetric `with_charges`, exact vs PWE low-rank) on
//!   periodic TFIM, periodic Heisenberg, and periodic explicit-link `U(1)`
//!   Schwinger rings with exact local Gauss's laws and `θ`-angle periodicity.
//! - Periodic TDVP real-time evolution (dense + symmetric `with_charges`).

use ketrs::statevector::EigenOptions;
use ketrs::{
    Charge, Circuit, Complex64 as C64, DmrgOptions, Mps, MpsOptions, Pauli, PauliString, PauliSum,
    PeriodicMps, Statevector,
};
use rand::SeedableRng;
use rand::rngs::StdRng;

fn pauli_term(ops: &[(usize, Pauli)]) -> PauliString {
    PauliString::new(ops.iter().copied()).unwrap()
}

/// Periodic transverse-field Ising ring:
/// `H = -J Σ_i Z_i Z_{(i+1)%n} - h Σ_i X_i`.
fn periodic_tfim(n: usize, j: f64, h_field: f64) -> PauliSum {
    let mut h = PauliSum::new();
    for i in 0..n {
        let next = (i + 1) % n;
        h.add(-j, pauli_term(&[(i, Pauli::Z), (next, Pauli::Z)]));
        h.add(-h_field, pauli_term(&[(i, Pauli::X)]));
    }
    h
}

/// Periodic Heisenberg XXZ ring:
/// `H = Σ_i (X_i X_{i+1} + Y_i Y_{i+1} + Δ Z_i Z_{i+1})`.
fn periodic_heisenberg(n: usize, delta: f64) -> PauliSum {
    let mut h = PauliSum::new();
    for i in 0..n {
        let next = (i + 1) % n;
        h.add(1.0, pauli_term(&[(i, Pauli::X), (next, Pauli::X)]));
        h.add(1.0, pauli_term(&[(i, Pauli::Y), (next, Pauli::Y)]));
        h.add(delta, pauli_term(&[(i, Pauli::Z), (next, Pauli::Z)]));
    }
    h
}

/// Periodic explicit-link `U(1)` Schwinger ring (quantum link model / Kogut–Susskind
/// with 1-qubit links `S = 1/2`, `L_n = (1/2) Z_{link_n} ∈ {-1/2, +1/2}`, OR
/// 2-qubit spin-1 links `L_n = (Z_{a} + Z_{b}) / 2 ∈ {-1, 0, +1}`).
///
/// Here we use a **2-qubit link** encoding (`L_n ∈ {-1, 0, +1}` with electric
/// raising operator `U_n = σ^+_a + σ^+_b`? Even cleaner, exact spin-1 / binary
/// 2-qubit link `L_n = n_0 + 2 n_1 - 1 ∈ {-1, 0, +1, +2}` OR 2-qubit symmetric
/// spin-1 `L_n = (Z_a + Z_b) / 2` where `|00⟩ -> +1`, `|01⟩,|10⟩ -> 0`, `|11⟩ -> -1`,
/// or a **1-qubit link** (`M` matter sites + `M` link qubits = `2M` qubits)
/// where `L_n = (1/2) Z_{2n+1} ∈ {-1/2, +1/2}` and `U_n = σ^+_{2n+1} = (X - iY)/2`!
///
/// Let's build the exact **periodic 2-qubit explicit-link U(1) Schwinger ring**
/// (`3M` qubits: site `3n` is staggered matter `q_n`, qubits `3n+1, 3n+2` encode
/// 4 electric flux states `E_n ∈ {-1, 0, +1, +2}` in binary `E_n = n_{a} + 2 n_{b} - 1`
/// with exact ladder operator `U_n = |0⟩⟨-1| + |1⟩⟨0| + |2⟩⟨1|`!):
///
/// Even better: let's provide BOTH:
/// 1. `periodic_qlm_1qubit(m_sites, w, mass, j_elec, theta)` (`2M` qubits, exact
///    spin-1/2 `U(1)` quantum link ring with `L_n = -Z_{2n+1}/2 ∈ {-1/2, +1/2}`).
/// 2. `periodic_schwinger_2qubit_links(m_sites, w, mass, j_elec, theta)` (`3M`
///    qubits, `E_n ∈ {-1, 0, +1, +2}` with exact binary `U_n` raising `E_n -> E_n + 1`,
///    so `θ` shifts `E_n -> E_n + θ/(2π)` and exhibits exact `2π`-periodicity in
///    the untruncated limit and exact local Gauss's law `G_n = E_n - E_{n-1} - Q_n = 0`!).
///
/// Qubit layout for 2-qubit links on `M` matter sites (`3M` qubits):
/// - Matter site `n`: qubit `3n`, staggered charge `Q_n = (1 - Z_{3n})/2 - (1 - (-1)^n)/2`
///   (`n` even: `|0⟩ -> 0, |1⟩ -> +1`; `n` odd: `|0⟩ -> -1, |1⟩ -> 0`).
/// - Link `n` (between site `n` and `(n+1)%M`): qubits `a = 3n + 1` (weight 1),
///   `b = 3n + 2` (weight 2), so `E_n = n_a + 2 n_b - 1 ∈ {-1, 0, +1, +2}` where
///   `n_q = (1 - Z_q)/2`.
/// - Local Gauss's law at EVERY site `n = 0..M` (including seam site `n = 0`!):
///   `G_n = E_n - E_{(n-1)%M} - Q_n = 0`.
///   Check `U(1)` charge weights for `G_n`:
///   - `E_n`: `n_a` (+1), `n_b` (+2) (constant -1 cancels with `-E_{n-1}`'s +1!)
///   - `-E_{n-1}`: `n_{a_prev}` (-1), `n_{b_prev}` (-2)
///   - `-Q_n`: matter `n_{3n}` (-1), plus constant `+(1 - (-1)^n)/2` (absorbed in total charge).
/// - Hopping term across link `n` (between matter `m0 = 3n`, link `(a, b)`,
///   matter `m1 = 3((n+1)%M)`): `H_hop,n = w (c†_{m0} U_n c_{m1} + h.c.)`,
///   where `U_n = σ^+_a + σ^-_a σ^+_b` raises `E_n` by `+1` and commutes with
///   every local Gauss's law `G_n` including across the periodic seam.
fn periodic_schwinger_ring(
    m_sites: usize,
    w: f64,
    mass: f64,
    j_elec: f64,
    theta: f64,
) -> (PauliSum, Vec<Charge>, Vec<bool>) {
    assert!(m_sites >= 2 && m_sites.is_multiple_of(2));
    let n = 3 * m_sites;
    let mut h = PauliSum::new();
    let e0 = theta / (2.0 * std::f64::consts::PI);

    // Helper: add `coeff * Re(term)` where `term` is a product of `Raise` (|1⟩⟨0| = (X-iY)/2)
    // and `Lower` (|0⟩⟨1| = (X+iY)/2) operators, plus its Hermitian conjugate!
    // For a product of k ladder ops Π_r (X_r + i s_r Y_r)/2 (s_r = -1 for Raise, +1 for Lower):
    //   term + term† = 2 Re(term) = (2 / 2^k) Σ_{even # of Y's} (-1)^{#Y/2} (Π_{Y} s_r) P
    let add_ladder_hc = |h: &mut PauliSum, coeff: f64, ops: &[(usize, bool)]| {
        // ops: (qubit, is_raise). Raise = (X - iY)/2 (s = -1), Lower = (X + iY)/2 (s = +1).
        let k = ops.len();
        let pre = coeff * 2.0 / (1usize << k) as f64;
        for mask in 0..(1usize << k) {
            let y_count = mask.count_ones() as usize;
            if !y_count.is_multiple_of(2) {
                continue;
            }
            let mut sign = if (y_count / 2).is_multiple_of(2) {
                1.0
            } else {
                -1.0
            };
            let mut paulis = Vec::with_capacity(k);
            for (bit, &(q, is_raise)) in ops.iter().enumerate() {
                if (mask >> bit) & 1 == 1 {
                    if is_raise {
                        sign = -sign;
                    }
                    paulis.push((q, Pauli::Y));
                } else {
                    paulis.push((q, Pauli::X));
                }
            }
            h.add(pre * sign, pauli_term(&paulis));
        }
    };

    for site in 0..m_sites {
        let m0 = 3 * site;
        let la = 3 * site + 1;
        let lb = 3 * site + 2;
        let m1 = 3 * ((site + 1) % m_sites);

        // Staggered mass term: (mass / 2) * (-1)^site * (1 - Z_{m0}) (or -(-1)^site Z_{m0} / 2)
        let stag = if site % 2 == 0 { 1.0 } else { -1.0 };
        h.add(-0.5 * mass * stag, pauli_term(&[(m0, Pauli::Z)]));

        // Electric field energy: J * (E_n + e0)^2
        // E_n = n_a + 2 n_b - 1 = (1 - Z_a)/2 + (1 - Z_b) - 1 = 0.5 - 0.5 Z_a - Z_b
        // Let c = 0.5 + e0: (c - 0.5 Z_a - Z_b)^2
        //   = c^2 + 0.25 + 1.0 - c Z_a - 2 c Z_b + Z_a Z_b
        let c = 0.5 + e0;
        h.add(j_elec * (c * c + 1.25), pauli_term(&[]));
        h.add(-j_elec * c, pauli_term(&[(la, Pauli::Z)]));
        h.add(-2.0 * j_elec * c, pauli_term(&[(lb, Pauli::Z)]));
        h.add(j_elec, pauli_term(&[(la, Pauli::Z), (lb, Pauli::Z)]));

        // Gauge-invariant hopping: w * (c†_{m0} U_n c_{m1} + h.c.)
        // U_n = σ^+_a + σ^-_a σ^+_b  (raises E_n by +1; note cyclic modular wrap for
        // exact 2π periodicity uses U_n_cyc = σ^+_a + σ^-_a σ^+_b + σ^-_a σ^-_b, which
        // wraps E_n = +2 -> -1 as a Z_4 clock! Without wrap it's truncated U(1).)
        // Here we use exact U(1) truncated U_n = σ^+_a + σ^-_a σ^+_b.
        // Note: on a periodic ring with even M, the seam hopping has Jordan-Wigner
        // parity sign (-1)^{N_f - 1}; in the fixed N_f = M/2 sector that is a constant
        // sign `jw_sign` on the seam link `site == m_sites - 1`!
        let jw = if site + 1 == m_sites && (m_sites / 2).is_multiple_of(2) {
            -w
        } else {
            w
        };
        // Term 1: c†_{m0} σ^+_a c_{m1} + h.c. (3-qubit term)
        add_ladder_hc(&mut h, jw, &[(m0, true), (la, true), (m1, false)]);
        // Term 2: c†_{m0} σ^-_a σ^+_b c_{m1} + h.c. (4-qubit term)
        add_ladder_hc(
            &mut h,
            jw,
            &[(m0, true), (la, false), (lb, true), (m1, false)],
        );
    }

    // Conserved charges:
    // 1. Global fermion number Σ_n n_{3n} (weights +1 on matter qubits).
    // 2. Local Gauss's laws G_n = E_n - E_{n-1} - Q_n for n = 0..M-1 (M-1 independent
    //    charges; the M-th is minus the sum of the others + global charge, and including
    //    all M is also completely valid!).
    let mut charges = Vec::with_capacity(1 + m_sites);
    let mut global_w = vec![0i64; n];
    for site in 0..m_sites {
        global_w[3 * site] = 1;
    }
    charges.push(Charge::weights(global_w));

    for site in 0..m_sites {
        let prev = (site + m_sites - 1) % m_sites;
        let mut g_w = vec![0i64; n];
        // +E_site: +1 on la, +2 on lb
        g_w[3 * site + 1] = 1;
        g_w[3 * site + 2] = 2;
        // -E_prev: -1 on prev_la, -2 on prev_lb
        g_w[3 * prev + 1] = -1;
        g_w[3 * prev + 2] = -2;
        // -Q_site: -1 on matter qubit 3*site
        g_w[3 * site] = -1;
        charges.push(Charge::weights(g_w));
    }

    // Bare vacuum product state satisfying ALL M local Gauss's laws G_n = 0:
    // - Even matter sites: |0⟩ (n=0 => Q_even = 0)
    // - Odd matter sites:  |1⟩ (n=1 => Q_odd = 1 - 1 = 0)
    // - All links: E_n = 0 => (la = 1, lb = 0) so n_a + 2 n_b - 1 = 1 + 0 - 1 = 0!
    let mut vac = vec![false; n];
    for site in 0..m_sites {
        if site % 2 == 1 {
            vac[3 * site] = true;
        }
        vac[3 * site + 1] = true; // E_n = 0
    }

    (h.simplify(0.0), charges, vac)
}

/// Exact local Gauss's law squared operator `G_site^2` for verification.
fn gauss_law_sqr(m_sites: usize, site: usize) -> PauliSum {
    let prev = (site + m_sites - 1) % m_sites;
    // G_site = E_site - E_prev - Q_site
    // E_site = 0.5 - 0.5 Z_{la} - Z_{lb}
    // E_prev = 0.5 - 0.5 Z_{pa} - Z_{pb}
    // Q_site = 0.5 (1 - Z_{m}) - (0 if even else 1) = (0.5 if even else -0.5) - 0.5 Z_m
    // So G_site = (-0.5 if even else +0.5) - 0.5 Z_{la} - Z_{lb} + 0.5 Z_{pa} + Z_{pb} + 0.5 Z_m.
    let c0 = if site.is_multiple_of(2) { -0.5 } else { 0.5 };
    let terms: Vec<(f64, Option<usize>)> = vec![
        (c0, None),
        (-0.5, Some(3 * site + 1)),
        (-1.0, Some(3 * site + 2)),
        (0.5, Some(3 * prev + 1)),
        (1.0, Some(3 * prev + 2)),
        (0.5, Some(3 * site)),
    ];
    let mut g2 = PauliSum::new();
    for &(c1, q1) in &terms {
        for &(c2, q2) in &terms {
            let c = c1 * c2;
            match (q1, q2) {
                (None, None) => g2.add(c, pauli_term(&[])),
                (Some(a), None) | (None, Some(a)) => g2.add(c, pauli_term(&[(a, Pauli::Z)])),
                (Some(a), Some(b)) if a == b => g2.add(c, pauli_term(&[])),
                (Some(a), Some(b)) => g2.add(c, pauli_term(&[(a, Pauli::Z), (b, Pauli::Z)])),
            };
        }
    }
    g2.simplify(0.0)
}

#[test]
fn periodic_trace_norm_inner_amplitudes_and_expectations_match_statevector() {
    let n = 8;
    let a = PeriodicMps::random(n, 4, 11, MpsOptions::default());
    let b = PeriodicMps::random(n, 3, 29, MpsOptions::default());

    let sv_a = Statevector::from_amplitudes(a.to_amplitudes()).unwrap();
    let sv_b = Statevector::from_amplitudes(b.to_amplitudes()).unwrap();

    // Norm squared.
    assert!((a.norm_sqr() - sv_a.norm_sqr()).abs() < 1e-12);
    assert!((a.norm_sqr() - 1.0).abs() < 1e-12);

    // Inner product.
    let inner_pmps = a.inner(&b).unwrap();
    let inner_sv = sv_a.inner(&sv_b).unwrap();
    assert!(
        (inner_pmps - inner_sv).norm() < 1e-12,
        "{inner_pmps} vs {inner_sv}"
    );

    // Individual amplitudes.
    for idx in [0usize, 1, 42, 173, (1 << n) - 1] {
        let bits: Vec<bool> = (0..n).map(|q| (idx >> q) & 1 == 1).collect();
        let amp = a.amplitude(&bits);
        assert!(
            (amp - sv_a.amplitudes()[idx]).norm() < 1e-12,
            "idx {idx}: {amp} vs {}",
            sv_a.amplitudes()[idx]
        );
    }

    // Local, long-range, and seam-crossing expectations (single and batched).
    let obs = vec![
        PauliSum::from(pauli_term(&[(0, Pauli::Z)])),
        PauliSum::from(pauli_term(&[(n - 1, Pauli::X)])),
        // Directly across the (n - 1, 0) seam!
        PauliSum::from(pauli_term(&[(n - 1, Pauli::Z), (0, Pauli::Z)])),
        PauliSum::from(pauli_term(&[(n - 1, Pauli::X), (0, Pauli::Y)])),
        // Multi-site across the seam: n-2, n-1, 0, 1.
        PauliSum::from(pauli_term(&[
            (n - 2, Pauli::X),
            (n - 1, Pauli::Z),
            (0, Pauli::Y),
            (1, Pauli::Z),
        ])),
        periodic_heisenberg(n, 0.7),
    ];
    let batch = a.expectations(&obs).unwrap();
    for (k, o) in obs.iter().enumerate() {
        let want = sv_a.expectation(o).unwrap();
        let single = a.expectation(o).unwrap();
        assert!(
            (single - want).abs() < 1e-12,
            "obs {k} single: {single} vs {want}"
        );
        assert!(
            (batch[k] - want).abs() < 1e-12,
            "obs {k} batch: {} vs {want}",
            batch[k]
        );
    }
}

#[test]
fn periodic_circuits_and_tebd_match_statevector_with_direct_seam_gates() {
    let n = 8;
    // Circuit with direct seam gates between qubit n-1 and qubit 0.
    let mut circ = Circuit::new(n);
    for q in 0..n {
        circ.ry(0.3 + 0.1 * q as f64, q);
    }
    for q in 0..n {
        circ.cx(q, (q + 1) % n);
        circ.rzz(0.25, q, (q + 1) % n);
    }
    let pmps = PeriodicMps::from_circuit(
        &circ,
        MpsOptions {
            max_bond: 32,
            cutoff: 0.0,
        },
    )
    .unwrap();
    let sv = Statevector::from_circuit(&circ);
    let diff = pmps
        .to_amplitudes()
        .iter()
        .zip(sv.amplitudes())
        .map(|(x, y)| (x - y).norm())
        .fold(0.0f64, f64::max);
    assert!(diff < 1e-12, "periodic circuit diff {diff}");
    // Seam bond is non-trivial (entangled across n-1 -> 0)!
    assert!(pmps.bond_dimensions()[n - 1] > 1);

    // Periodic TEBD on periodic Heisenberg ring (exact at cutoff = 0, max_bond = 64).
    let h = periodic_heisenberg(n, 1.0);
    let neel: Vec<bool> = (0..n).map(|q| q % 2 == 1).collect();
    let mut pmps = PeriodicMps::basis(
        n,
        &neel,
        MpsOptions {
            max_bond: 64,
            cutoff: 0.0,
        },
    );
    pmps.tebd(&h, 0.4, 8, 2).unwrap();

    let mut exact_circ = Circuit::new(n);
    for (q, &b) in neel.iter().enumerate() {
        if b {
            exact_circ.x(q);
        }
    }
    exact_circ.pauli_evolution(&h, 0.4, 8, 2).unwrap();
    let exact_sv = Statevector::from_circuit(&exact_circ);
    let diff = pmps
        .to_amplitudes()
        .iter()
        .zip(exact_sv.amplitudes())
        .map(|(x, y)| (x - y).norm())
        .fold(0.0f64, f64::max);
    assert!(diff < 1e-8, "periodic TEBD diff {diff}");

    // Truncated periodic TEBD exercises exact O(n χ³) metric-weighted bond splits.
    let mut pmps_trunc = PeriodicMps::basis(
        n,
        &neel,
        MpsOptions {
            max_bond: 8,
            cutoff: 1e-12,
        },
    );
    pmps_trunc.tebd(&h, 0.4, 8, 2).unwrap();
    assert!(pmps_trunc.bond_dimensions().into_iter().max().unwrap_or(0) <= 8);
    assert!((pmps_trunc.norm_sqr() - 1.0).abs() < 1e-3);
    let overlap = Statevector::from_amplitudes(pmps_trunc.to_amplitudes())
        .unwrap()
        .inner(&exact_sv)
        .unwrap()
        .norm();
    assert!(
        overlap > 0.999,
        "truncated periodic TEBD fidelity {overlap}"
    );
}

#[test]
fn periodic_dmrg_tfim_matches_exact_diagonalization_and_preserves_translation_invariance() {
    let n = 8;
    let h = periodic_tfim(n, 1.0, 0.8);
    let exact = Statevector::lowest_eigenstates(&h, n, 1, &EigenOptions::default()).unwrap();
    let e_exact = exact[0].energy;

    let mut pmps = PeriodicMps::random(n, 4, 7, MpsOptions::default());
    let res = pmps
        .dmrg(
            &h,
            &DmrgOptions {
                max_bond: 12,
                cutoff: 1e-13,
                max_sweeps: 10,
                tolerance: 1e-11,
                noise: 0.0,
                ..Default::default()
            },
        )
        .unwrap();
    assert!(
        (res.energy - e_exact).abs() < 1e-10,
        "TFIM periodic DMRG {} vs exact {e_exact}",
        res.energy
    );
    let e_check = pmps.expectation(&h).unwrap();
    assert!((e_check - e_exact).abs() < 1e-10);

    // Check translation invariance across EVERY bond including the (n-1, 0) seam!
    let zz_obs: Vec<PauliSum> = (0..n)
        .map(|i| PauliSum::from(pauli_term(&[(i, Pauli::Z), ((i + 1) % n, Pauli::Z)])))
        .collect();
    let zzs = pmps.expectations(&zz_obs).unwrap();
    for (i, &val) in zzs.iter().enumerate() {
        assert!(
            (val - zzs[0]).abs() < 1e-8,
            "translation symmetry broken on bond {i}: {val} vs {}",
            zzs[0]
        );
    }
}

#[test]
fn periodic_dmrg_heisenberg_dense_and_symmetric_match_exact() {
    let n = 8;
    let h = periodic_heisenberg(n, 1.0);
    let exact = Statevector::lowest_eigenstates(&h, n, 1, &EigenOptions::default()).unwrap();
    let e_exact = exact[0].energy;

    // 1. Dense periodic DMRG.
    let mut dense = PeriodicMps::random(n, 4, 42, MpsOptions::default());
    let res_dense = dense
        .dmrg(
            &h,
            &DmrgOptions {
                max_bond: 16,
                cutoff: 1e-13,
                max_sweeps: 12,
                tolerance: 1e-11,
                noise: 0.0,
                ..Default::default()
            },
        )
        .unwrap();
    assert!(
        (res_dense.energy - e_exact).abs() < 1e-10,
        "dense Heisenberg pDMRG {} vs exact {e_exact}",
        res_dense.energy
    );

    // 2. Symmetric U(1) + Z_2 periodic DMRG from Néel product state.
    let neel: Vec<bool> = (0..n).map(|q| q % 2 == 1).collect();
    let charges = vec![
        Charge::weights(vec![1; n]),
        Charge::modulo(2, vec![[0, 1]; n]),
    ];
    let mut sym = PeriodicMps::basis(n, &neel, MpsOptions::default())
        .with_charges(&charges)
        .unwrap();
    assert_eq!(
        sym.total_charge(),
        Some(vec![(n / 2) as i64, (n / 2 % 2) as i64])
    );

    let res_sym = sym
        .dmrg(
            &h,
            &DmrgOptions {
                max_bond: 16,
                cutoff: 1e-13,
                max_sweeps: 12,
                tolerance: 1e-11,
                noise: 1e-4,
                ..Default::default()
            },
        )
        .unwrap();
    assert!(
        (res_sym.energy - e_exact).abs() < 1e-10,
        "symmetric Heisenberg pDMRG {} vs exact {e_exact}",
        res_sym.energy
    );
    // Seam bond grew multiple U(1) charge sectors!
    let seam_sectors = sym.bond_sectors(n - 1).unwrap();
    assert!(
        seam_sectors.len() >= 2,
        "expected multi-sector seam bond, got {seam_sectors:?}"
    );
    assert!((sym.expectation(&h).unwrap() - e_exact).abs() < 1e-10);
}

#[test]
fn periodic_dmrg_pwe_low_rank_matches_exact_mode() {
    let n = 12;
    let h = periodic_tfim(n, 1.0, 2.5);
    let exact = Statevector::lowest_eigenstates(&h, n, 1, &EigenOptions::default()).unwrap();
    let e_exact = exact[0].energy;

    // Force exact mode (transfer_rank = Some(0)).
    let mut pmps_exact = PeriodicMps::random(n, 4, 3, MpsOptions::default());
    pmps_exact.set_transfer_rank(Some(0));
    let r_exact = pmps_exact
        .dmrg(
            &h,
            &DmrgOptions {
                max_bond: 8,
                cutoff: 1e-12,
                max_sweeps: 10,
                tolerance: 1e-11,
                noise: 0.0,
                ..Default::default()
            },
        )
        .unwrap();

    // Adaptive PWE mode (transfer_rank = None) + explicit capped rank (Some(32) < 64).
    let mut pmps_pwe = PeriodicMps::random(n, 4, 3, MpsOptions::default());
    pmps_pwe.set_transfer_rank(Some(32));
    let r_pwe = pmps_pwe
        .dmrg(
            &h,
            &DmrgOptions {
                max_bond: 8,
                cutoff: 1e-12,
                max_sweeps: 10,
                tolerance: 1e-11,
                noise: 0.0,
                ..Default::default()
            },
        )
        .unwrap();

    assert!(
        (r_exact.energy - e_exact).abs() < 1e-6,
        "{} vs {e_exact}",
        r_exact.energy
    );
    assert!(
        (r_pwe.energy - e_exact).abs() < 1e-4,
        "{} vs {e_exact}",
        r_pwe.energy
    );
    assert!(
        (r_pwe.energy - r_exact.energy).abs() < 1e-4,
        "PWE {} vs exact-mode {}",
        r_pwe.energy,
        r_exact.energy
    );
    assert!((pmps_pwe.expectation(&h).unwrap() - e_exact).abs() < 1e-4);
}

#[test]
fn periodic_explicit_link_u1_schwinger_ring_gauss_laws_and_theta_symmetry() {
    // M = 4 matter sites + 4 explicit 2-qubit links = 12 qubits on a closed ring!
    // Exact local Gauss's laws G_0, G_1, G_2, G_3 = 0 enforced by `with_charges`.
    let m_sites = 4;
    let n = 3 * m_sites; // 12 qubits
    let (w, mass, j_elec) = (0.8, 0.5, 1.0);

    // Check against exact diagonalization in the physical Gauss-law sector by
    // adding a large Gauss penalty to exact ED (which shifts non-physical states up).
    let exact_gs_energy = |theta: f64| -> f64 {
        let (h, _, _) = periodic_schwinger_ring(m_sites, w, mass, j_elec, theta);
        let mut h_pen = h.clone();
        for site in 0..m_sites {
            let g2 = gauss_law_sqr(m_sites, site);
            for (c, s) in g2.terms() {
                h_pen.add(50.0 * c, s.clone());
            }
        }
        // Also pin global charge N_f = M/2 with penalty.
        // N_f - M/2 = -0.5 Σ_site Z_{3*site}
        for s1 in 0..m_sites {
            for s2 in 0..m_sites {
                if s1 == s2 {
                    h_pen.add(50.0 * 0.25, pauli_term(&[]));
                } else {
                    h_pen.add(
                        50.0 * 0.25,
                        pauli_term(&[(3 * s1, Pauli::Z), (3 * s2, Pauli::Z)]),
                    );
                }
            }
        }
        let res =
            Statevector::lowest_eigenstates(&h_pen.simplify(0.0), n, 1, &EigenOptions::default())
                .unwrap();
        res[0].energy
    };

    // Test at θ = 0 and θ = 0.7π (non-trivial background electric field).
    for &theta in &[0.0, 0.7 * std::f64::consts::PI] {
        let e_exact = exact_gs_energy(theta);
        let (h, charges, vac) = periodic_schwinger_ring(m_sites, w, mass, j_elec, theta);
        let mut sym = PeriodicMps::basis(n, &vac, MpsOptions::default())
            .with_charges(&charges)
            .unwrap();
        let res = sym
            .dmrg(
                &h,
                &DmrgOptions {
                    max_bond: 24,
                    cutoff: 1e-13,
                    max_sweeps: 14,
                    tolerance: 1e-11,
                    noise: 1e-2,
                    noise_decay: 0.3,
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(
            (res.energy - e_exact).abs() < 1e-5,
            "Schwinger ring θ={theta:.2}: pDMRG {} vs exact {e_exact}",
            res.energy
        );
        // Verify exact local Gauss's law <G_n^2> == 0 on ALL sites including seam site 0!
        for site in 0..m_sites {
            let g2 = sym.expectation(&gauss_law_sqr(m_sites, site)).unwrap();
            assert!(
                g2.abs() < 1e-11,
                "Gauss's law violated at site {site}: <G^2> = {g2}"
            );
        }
        // Verify seam bond carries multiple electric flux sectors!
        let seam = sym.bond_sectors(n - 1).unwrap();
        assert!(
            seam.len() >= 2,
            "expected multi-sector electric flux seam bond, got {seam:?}"
        );
    }

    // Check charge-conjugation / θ symmetry:
    // For binary links E ∈ {-1, 0, +1, +2}, E + e0 has exact reflection symmetry
    // around e0 = -0.5 (θ = -π) via E_n -> 1 - E_n (which maps the W = 0 vacuum
    // E_n = 0 to the W = +1 vacuum E_n = +1, both satisfying all local G_n = 0).
    // Thus E(-π - δ) == E(-π + δ)!
    let delta = 0.4;
    let theta_a = -std::f64::consts::PI - delta;
    let theta_b = -std::f64::consts::PI + delta;
    let run_theta = |theta: f64, link_e1: bool| -> f64 {
        let (h, charges, mut vac) = periodic_schwinger_ring(m_sites, w, mass, j_elec, theta);
        if link_e1 {
            // Charge-conjugate vacuum: E_n = +1 on all links => (la = 0, lb = 1).
            for site in 0..m_sites {
                vac[3 * site + 1] = false;
                vac[3 * site + 2] = true;
            }
        }
        let mut sym = PeriodicMps::basis(n, &vac, MpsOptions::default())
            .with_charges(&charges)
            .unwrap();
        sym.dmrg(
            &h,
            &DmrgOptions {
                max_bond: 16,
                cutoff: 1e-13,
                max_sweeps: 14,
                tolerance: 1e-11,
                noise: 1e-2,
                noise_decay: 0.3,
                ..Default::default()
            },
        )
        .unwrap()
        .energy
    };
    let ea = run_theta(theta_a, true);
    let eb = run_theta(theta_b, false);
    assert!(
        (ea - eb).abs() < 1e-6,
        "θ-reflection symmetry broken: E(-π-δ)={ea} vs E(-π+δ)={eb}"
    );
}

#[test]
fn periodic_tdvp_dense_and_symmetric_match_exact_evolution() {
    let n = 6;
    let h = periodic_heisenberg(n, 0.6);
    let neel: Vec<bool> = (0..n).map(|q| q % 2 == 1).collect();

    // Start from a Néel product state and evolve under periodic Heisenberg.
    let mut exact = Statevector::zero(n);
    let mut prep = Circuit::new(n);
    for (q, &b) in neel.iter().enumerate() {
        if b {
            prep.x(q);
        }
    }
    exact.run(&prep).unwrap();
    let t = 0.4;
    exact.evolve(&h, t).unwrap();

    // 1. Dense periodic TDVP (antipodal window-split 2nd-order Strang integrator).
    let mut dense = PeriodicMps::basis(
        n,
        &neel,
        MpsOptions {
            max_bond: 12,
            cutoff: 1e-14,
        },
    );
    let e0 = dense.expectation(&h).unwrap();
    dense.tdvp(&h, t, 8).unwrap();
    let e1 = dense.expectation(&h).unwrap();
    assert!(
        (dense.norm_sqr() - 1.0).abs() < 1e-9,
        "norm_sqr = {}",
        dense.norm_sqr()
    );
    let de_coarse = (e1 - e0).abs();
    assert!(de_coarse < 5e-2, "energy drift: {e0} -> {e1}");
    let diff = dense
        .to_amplitudes()
        .iter()
        .zip(exact.amplitudes())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0f64, f64::max);
    assert!(diff < 5e-3, "dense periodic TDVP diff {diff}");

    // 2. Second-order convergence: halving dt cuts both wavefunction and energy error by ~4x!
    let mut fine = PeriodicMps::basis(
        n,
        &neel,
        MpsOptions {
            max_bond: 12,
            cutoff: 1e-14,
        },
    );
    fine.tdvp(&h, t, 16).unwrap();
    let de_fine = (fine.expectation(&h).unwrap() - e0).abs();
    let diff_fine = fine
        .to_amplitudes()
        .iter()
        .zip(exact.amplitudes())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0f64, f64::max);
    assert!(
        diff_fine < diff / 3.5,
        "expected 2nd-order state convergence: coarse={diff:.2e}, fine={diff_fine:.2e}"
    );
    assert!(
        de_fine < de_coarse / 3.5,
        "expected 2nd-order energy convergence: coarse={de_coarse:.2e}, fine={de_fine:.2e}"
    );

    // 3. Symmetric U(1) periodic TDVP matches dense periodic TDVP!
    let mut sym = PeriodicMps::basis(
        n,
        &neel,
        MpsOptions {
            max_bond: 12,
            cutoff: 1e-14,
        },
    )
    .with_charges(&[Charge::weights(vec![1; n])])
    .unwrap();
    sym.tdvp(&h, t, 8).unwrap();
    let sym_diff = sym
        .to_amplitudes()
        .iter()
        .zip(dense.to_amplitudes())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0f64, f64::max);
    assert!(sym_diff < 1e-8, "symmetric vs dense pTDVP diff {sym_diff}");
    // Seam bond grew multi-sector entanglement during TDVP!
    assert!(sym.bond_sectors(n - 1).unwrap().len() >= 2);
}

#[test]
fn periodic_projection_and_sampling_match_statevector() {
    let n = 6;
    let pmps = PeriodicMps::random(n, 3, 99, MpsOptions::default());
    let sv = Statevector::from_amplitudes(pmps.to_amplitudes()).unwrap();

    // Check marginal probabilities on every qubit.
    for q in 0..n {
        let p_pmps = pmps.probability_one(q).unwrap();
        let p_sv: f64 = sv
            .probabilities()
            .iter()
            .enumerate()
            .filter(|(idx, _)| (idx >> q) & 1 == 1)
            .map(|(_, &p)| p)
            .sum();
        assert!((p_pmps - p_sv).abs() < 1e-12);
    }

    // Project onto qubit 2 = 1 and compare renormalised amplitudes.
    let mut proj = pmps.clone();
    let p = proj.project(2, true).unwrap();
    assert!((proj.norm_sqr() - 1.0).abs() < 1e-11);
    let scale = p.sqrt().recip();
    for (idx, (&a, &b)) in proj.to_amplitudes().iter().zip(sv.amplitudes()).enumerate() {
        let want = if (idx >> 2) & 1 == 1 {
            b * scale
        } else {
            C64::new(0.0, 0.0)
        };
        assert!((a - want).norm() < 1e-11, "idx {idx}: {a} vs {want}");
    }

    // Sampling on a periodic GHZ ring.
    let mut ghz = Circuit::new(n);
    ghz.h(0);
    for q in 0..n - 1 {
        ghz.cx(q, q + 1);
    }
    let pmps_ghz = PeriodicMps::from_circuit(&ghz, MpsOptions::default()).unwrap();
    let mut rng = StdRng::seed_from_u64(5);
    let counts = pmps_ghz.sample_counts(100, &mut rng);
    assert_eq!(
        counts.keys().copied().collect::<Vec<_>>(),
        vec![0, (1usize << n) - 1]
    );
}

#[test]
fn periodic_from_mps_roundtrip_and_with_charges_seam_unfold() {
    let n = 6;
    let h = periodic_heisenberg(n, 1.0);
    let neel: Vec<bool> = (0..n).map(|q| q % 2 == 1).collect();
    let mut open = Mps::basis(n, &neel, MpsOptions::default());
    open.tebd(&h, 0.3, 4, 2).unwrap();
    let pmps = PeriodicMps::from_mps(&open);
    let diff = pmps
        .to_amplitudes()
        .iter()
        .zip(open.to_amplitudes())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0f64, f64::max);
    assert!(diff < 1e-12);

    // Now evolve PeriodicMps so seam bond > 1, then call with_charges!
    let mut pmps2 = PeriodicMps::basis(n, &neel, MpsOptions::default());
    pmps2.tebd(&h, 0.3, 4, 2).unwrap();
    assert!(pmps2.bond_dimensions()[n - 1] > 1);
    let sym = pmps2.with_charges(&[Charge::weights(vec![1; n])]).unwrap();
    assert_eq!(sym.total_charge(), Some(vec![3]));
    let diff2 = sym
        .to_amplitudes()
        .iter()
        .zip(pmps2.to_amplitudes())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0f64, f64::max);
    assert!(diff2 < 1e-11, "unfolded with_charges diff {diff2}");
}

#[test]
fn periodic_tdvp_multisite_long_range_and_wrap_around_match_exact_and_open_mps() {
    let n = 8;
    let mut bits = vec![false; n];
    bits[0] = true; // |10000000⟩ (qubit 0 excited)
    let mut prep = Circuit::new(n);
    prep.x(0);

    let z_ops: Vec<PauliSum> = (0..n)
        .map(|q| PauliSum::from(pauli_term(&[(q, Pauli::Z)])))
        .collect();

    let h_from = |list: &[(&str, f64)]| PauliSum::from_list(list.iter().copied()).unwrap();

    // Cases: 2q NN, 3q contiguous, 2q span-4, 3q wrap-around across seam,
    // antipodal span-5 (1,5), 6q wrap-around across seam (6,7,0,1,2,3), and
    // full-ring 8q term (exercises empty complement `opp = []`).
    let cases: Vec<(&str, PauliSum)> = vec![
        (
            "2q_NN(0,1)",
            h_from(&[("IIIIIIXX", 1.0), ("IIIIIIYY", 1.0)]),
        ),
        (
            "3q_contig(0,1,2)",
            h_from(&[("IIIIIXXX", 1.0), ("IIIIIXYY", 1.0)]),
        ),
        (
            "2q_span4(0,3)",
            h_from(&[("IIIIXIIX", 1.0), ("IIIIYIIY", 1.0)]),
        ),
        (
            "3q_wrap(7,0,1)",
            h_from(&[("XIIIIIXX", 1.0), ("XIIIIIYY", 1.0)]),
        ),
        (
            "2q_antipodal(1,5)",
            h_from(&[("IIXIIIXI", 1.0), ("IIYIIIYI", 1.0)]),
        ),
        (
            "6q_wrap(6,7,0,1,2,3)",
            h_from(&[("XXIIXXXX", 1.0), ("XXIIXXYY", 1.0)]),
        ),
        (
            "8q_full_ring",
            h_from(&[("XXXXXXXX", 1.0), ("XXXXXXYY", 1.0)]),
        ),
    ];

    let opts = MpsOptions {
        max_bond: 16,
        cutoff: 1e-12,
    };
    let t = 0.5;
    let steps = 4;

    for (name, h) in &cases {
        let mut sv = Statevector::from_circuit(&prep);
        sv.evolve(h, t).unwrap();
        let mut open = Mps::basis(n, &bits, opts);
        open.tdvp(h, t, steps).unwrap();
        let mut pbc = PeriodicMps::basis(n, &bits, opts);
        pbc.tdvp(h, t, steps).unwrap();

        assert!(
            (pbc.norm_sqr() - 1.0).abs() < 1e-10,
            "{name}: norm_sqr = {}",
            pbc.norm_sqr()
        );
        let z_exact: Vec<f64> = z_ops.iter().map(|o| sv.expectation(o).unwrap()).collect();
        let z_open = open.expectations(&z_ops).unwrap();
        let z_pbc = pbc.expectations(&z_ops).unwrap();
        for q in 0..n {
            assert!(
                (z_pbc[q] - z_exact[q]).abs() < 1e-6,
                "{name} q{q}: PeriodicMps <Z>={} vs exact {} (open {})",
                z_pbc[q],
                z_exact[q],
                z_open[q]
            );
            assert!(
                (z_pbc[q] - z_open[q]).abs() < 1e-6,
                "{name} q{q}: PeriodicMps <Z>={} vs MPS {}",
                z_pbc[q],
                z_open[q]
            );
        }
        let amp_diff = pbc
            .to_amplitudes()
            .iter()
            .zip(sv.amplitudes())
            .map(|(a, b)| (a - b).norm())
            .fold(0.0f64, f64::max);
        assert!(amp_diff < 1e-6, "{name}: amplitude diff {amp_diff}");
    }

    // Combined non-commuting Hamiltonian mixing NN, 3q, span-4, and wrap-around terms.
    let mut h_mix = PauliSum::new();
    for (_, h) in &cases[..4] {
        for (c, s) in h.terms() {
            h_mix.add(0.4 * c, s.clone());
        }
    }
    let mut sv = Statevector::from_circuit(&prep);
    sv.evolve(&h_mix, 0.4).unwrap();
    let mut pbc_coarse = PeriodicMps::basis(
        n,
        &bits,
        MpsOptions {
            max_bond: 32,
            cutoff: 1e-13,
        },
    );
    pbc_coarse.tdvp(&h_mix, 0.4, 4).unwrap();
    let mut pbc_fine = PeriodicMps::basis(
        n,
        &bits,
        MpsOptions {
            max_bond: 32,
            cutoff: 1e-13,
        },
    );
    pbc_fine.tdvp(&h_mix, 0.4, 8).unwrap();
    let err_coarse = pbc_coarse
        .to_amplitudes()
        .iter()
        .zip(sv.amplitudes())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0f64, f64::max);
    let err_fine = pbc_fine
        .to_amplitudes()
        .iter()
        .zip(sv.amplitudes())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0f64, f64::max);
    assert!(
        err_coarse < 5e-3,
        "combined multi-site pTDVP coarse err {err_coarse}"
    );
    assert!(
        err_fine < err_coarse / 3.0,
        "expected 2nd-order convergence: coarse={err_coarse:.2e}, fine={err_fine:.2e}"
    );
}

#[test]
fn periodic_tdvp_multisite_and_wrap_around_with_charges() {
    let n = 8;
    let mut bits = vec![false; n];
    bits[0] = true;
    bits[2] = true;
    bits[6] = true;
    let charge = Charge::weights(vec![1; n]);

    // Charge-conserving multi-site terms:
    // - 3q contiguous JW hop (0,1,2)
    // - 2q span-4 hop (0,3)
    // - 5q wrap-around JW hop across seam (6,7,0,1,2)
    // - 6q wrap-around dressed hop across seam (5,6,7,0,1,2) with Z dressers
    let mut h = PauliSum::new();
    for p in [Pauli::X, Pauli::Y] {
        h.add(0.7, pauli_term(&[(0, p), (1, Pauli::Z), (2, p)]));
        h.add(0.6, pauli_term(&[(0, p), (3, p)]));
        h.add(
            0.5,
            pauli_term(&[(6, p), (7, Pauli::Z), (0, Pauli::Z), (1, Pauli::Z), (2, p)]),
        );
        h.add(
            0.4,
            pauli_term(&[
                (5, p),
                (6, Pauli::Z),
                (7, Pauli::Z),
                (0, Pauli::Z),
                (1, Pauli::Z),
                (2, p),
            ]),
        );
    }

    let mut prep = Circuit::new(n);
    for (q, &b) in bits.iter().enumerate() {
        if b {
            prep.x(q);
        }
    }
    let t = 0.35;
    let mut sv = Statevector::from_circuit(&prep);
    sv.evolve(&h, t).unwrap();

    let opts = MpsOptions {
        max_bond: 32,
        cutoff: 1e-13,
    };
    let mut dense = PeriodicMps::basis(n, &bits, opts);
    dense.tdvp(&h, t, 8).unwrap();
    let mut sym = PeriodicMps::basis(n, &bits, opts)
        .with_charges(std::slice::from_ref(&charge))
        .unwrap();
    sym.tdvp(&h, t, 8).unwrap();

    let diff_dense_sv = dense
        .to_amplitudes()
        .iter()
        .zip(sv.amplitudes())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0f64, f64::max);
    let diff_sym_dense = sym
        .to_amplitudes()
        .iter()
        .zip(dense.to_amplitudes())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0f64, f64::max);
    assert!(
        diff_dense_sv < 5e-3,
        "dense charged multi-site pTDVP vs exact: {diff_dense_sv}"
    );
    assert!(
        diff_sym_dense < 1e-7,
        "symmetric vs dense charged multi-site pTDVP: {diff_sym_dense}"
    );
    assert!(sym.bond_sectors(n - 1).unwrap().len() >= 2);

    // Explicit-link U(1) Schwinger ring (3-qubit and 4-qubit gauge-covariant
    // hopping including across the periodic seam) evolved by symmetric TDVP
    // from the bare vacuum product state with ALL local Gauss's laws conserved!
    let m_sites = 4;
    let n_sch = 3 * m_sites; // 12 qubits
    let (h_sch, charges_sch, vac) = periodic_schwinger_ring(m_sites, 0.8, 0.5, 1.0, 0.3);
    let mut prep_sch = Circuit::new(n_sch);
    for (q, &b) in vac.iter().enumerate() {
        if b {
            prep_sch.x(q);
        }
    }
    let mut sv_sch = Statevector::from_circuit(&prep_sch);
    sv_sch.evolve(&h_sch, 0.3).unwrap();

    let mut sym_sch = PeriodicMps::basis(
        n_sch,
        &vac,
        MpsOptions {
            max_bond: 24,
            cutoff: 1e-13,
        },
    )
    .with_charges(&charges_sch)
    .unwrap();
    sym_sch.tdvp(&h_sch, 0.3, 6).unwrap();

    // Must evolve non-trivially and match exact Schwinger evolution!
    let diff_sch = sym_sch
        .to_amplitudes()
        .iter()
        .zip(sv_sch.amplitudes())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0f64, f64::max);
    assert!(
        diff_sch < 1e-2,
        "Schwinger ring symmetric pTDVP vs exact: {diff_sch}"
    );
    for site in 0..m_sites {
        let g2 = sym_sch.expectation(&gauss_law_sqr(m_sites, site)).unwrap();
        assert!(
            g2.abs() < 1e-10,
            "Gauss's law violated after pTDVP at site {site}: <G^2>={g2}"
        );
    }
    assert!(sym_sch.bond_sectors(n_sch - 1).unwrap().len() >= 2);
}

#[test]
fn periodic_tdvp_n3_and_dmrg_expectation_multisite_wrap_around() {
    // 1. n = 3 with 2-qubit seam terms and a 3-qubit full-ring term (dense + symmetric).
    let n3 = 3;
    let mut h3 = PauliSum::new();
    for p in [Pauli::X, Pauli::Y] {
        h3.add(0.8, pauli_term(&[(2, p), (0, p)]));
        h3.add(0.6, pauli_term(&[(0, p), (1, Pauli::Z), (2, p)]));
    }
    let bits3 = [true, false, false];
    let mut prep3 = Circuit::new(n3);
    prep3.x(0);
    let mut sv3 = Statevector::from_circuit(&prep3);
    sv3.evolve(&h3, 0.4).unwrap();

    let opts3 = MpsOptions {
        max_bond: 8,
        cutoff: 1e-14,
    };
    let mut p3 = PeriodicMps::basis(n3, &bits3, opts3);
    p3.tdvp(&h3, 0.4, 8).unwrap();
    let mut s3 = PeriodicMps::basis(n3, &bits3, opts3)
        .with_charges(&[Charge::weights(vec![1; n3])])
        .unwrap();
    s3.tdvp(&h3, 0.4, 8).unwrap();
    let d3 = p3
        .to_amplitudes()
        .iter()
        .zip(sv3.amplitudes())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0f64, f64::max);
    let ds3 = s3
        .to_amplitudes()
        .iter()
        .zip(p3.to_amplitudes())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0f64, f64::max);
    assert!(d3 < 5e-3, "n=3 pTDVP diff {d3}");
    assert!(ds3 < 1e-8, "n=3 sym vs dense pTDVP diff {ds3}");

    // 2. Periodic DMRG & expectation on seam-straddling 3q and 5q wrap-around
    //    Hamiltonian starting from a product state (dense + symmetric).
    let n = 8;
    let mut h_wrap = PauliSum::new();
    for p in [Pauli::X, Pauli::Y] {
        // Straddles seam 7 <-> 0 (not in W1=[0..=4] and not in W2=[4..7,0]!)
        h_wrap.add(1.0, pauli_term(&[(7, p), (0, Pauli::Z), (1, p)]));
        // 5-qubit wrap-around straddling seam
        h_wrap.add(
            0.7,
            pauli_term(&[(6, p), (7, Pauli::Z), (0, Pauli::Z), (1, Pauli::Z), (2, p)]),
        );
        // Span-4 non-NN term
        h_wrap.add(0.5, pauli_term(&[(0, p), (3, p)]));
        h_wrap.add(0.5, pauli_term(&[(4, p), (7, p)]));
    }
    for q in 0..n {
        h_wrap.add(0.15 * (q as f64 - 3.5), pauli_term(&[(q, Pauli::Z)]));
    }

    // Exact ground state in the 2-particle sector (match via large penalty).
    let mut h_pen = h_wrap.clone();
    // (Σ n_q - 2)^2 = (2 - 0.5 Σ Z_q)^2
    h_pen.add(50.0 * 4.0, pauli_term(&[]));
    for q in 0..n {
        h_pen.add(50.0 * (-2.0), pauli_term(&[(q, Pauli::Z)]));
        for r in 0..n {
            if q == r {
                h_pen.add(50.0 * 0.25, pauli_term(&[]));
            } else {
                h_pen.add(50.0 * 0.25, pauli_term(&[(q, Pauli::Z), (r, Pauli::Z)]));
            }
        }
    }
    let e_exact_sector =
        Statevector::lowest_eigenstates(&h_pen.simplify(0.0), n, 1, &EigenOptions::default())
            .unwrap()[0]
            .energy;

    // Symmetric DMRG starting from a 2-particle product state!
    let mut bits = vec![false; n];
    bits[0] = true;
    bits[7] = true;
    let mut sym = PeriodicMps::basis(n, &bits, MpsOptions::default())
        .with_charges(&[Charge::weights(vec![1; n])])
        .unwrap();
    let res = sym
        .dmrg(
            &h_wrap,
            &DmrgOptions {
                max_bond: 16,
                cutoff: 1e-13,
                max_sweeps: 10,
                tolerance: 1e-10,
                noise: 1e-2,
                noise_decay: 0.2,
                ..Default::default()
            },
        )
        .unwrap();
    assert!(
        (res.energy - e_exact_sector).abs() < 1e-7,
        "seam-straddling multi-site pDMRG {} vs exact {e_exact_sector}",
        res.energy
    );
    let sv_res = Statevector::from_amplitudes(sym.to_amplitudes()).unwrap();
    let e_pmps = sym.expectation(&h_wrap).unwrap();
    let e_sv = sv_res.expectation(&h_wrap).unwrap();
    assert!(
        (e_pmps - e_sv).abs() < 1e-11,
        "multi-site wrap expectation {e_pmps} vs {e_sv}"
    );
}

#[cfg(feature = "gpu")]
#[test]
fn periodic_gpu_dmrg_and_tdvp_match_cpu() {
    let ctx = match ketrs::gpu::GpuContext::new() {
        Ok(c) if c.supports_mps() => c,
        _ => return,
    };

    // 1. Dense periodic DMRG on GPU matches CPU to < 1e-10.
    let n = 8;
    let h = periodic_heisenberg(n, 0.5);
    let opts = DmrgOptions {
        max_bond: 12,
        cutoff: 1e-13,
        max_sweeps: 8,
        tolerance: 1e-11,
        noise: 0.0,
        ..Default::default()
    };
    let mut cpu = PeriodicMps::random(n, 4, 17, MpsOptions::default());
    let mut gpu = cpu.clone();
    gpu.set_gpu(Some(ctx.clone())).unwrap();
    assert!(gpu.gpu().is_some());

    let r_cpu = cpu.dmrg(&h, &opts).unwrap();
    let r_gpu = gpu.dmrg(&h, &opts).unwrap();
    assert!(
        (r_cpu.energy - r_gpu.energy).abs() < 1e-10,
        "dense GPU pDMRG {} vs CPU {}",
        r_gpu.energy,
        r_cpu.energy
    );

    // 2. Symmetric U(1) periodic DMRG on GPU matches CPU to < 1e-10.
    let neel: Vec<bool> = (0..n).map(|q| q % 2 == 1).collect();
    let mut cpu_sym = PeriodicMps::basis(n, &neel, MpsOptions::default())
        .with_charges(&[Charge::weights(vec![1; n])])
        .unwrap();
    let mut gpu_sym = cpu_sym.clone();
    gpu_sym.set_gpu(Some(ctx.clone())).unwrap();
    let sym_opts = DmrgOptions {
        max_bond: 12,
        cutoff: 1e-13,
        max_sweeps: 8,
        tolerance: 1e-11,
        noise: 1e-3,
        noise_decay: 0.2,
        ..Default::default()
    };
    let rs_cpu = cpu_sym.dmrg(&h, &sym_opts).unwrap();
    let rs_gpu = gpu_sym.dmrg(&h, &sym_opts).unwrap();
    assert!(
        (rs_cpu.energy - rs_gpu.energy).abs() < 1e-10,
        "symmetric GPU pDMRG {} vs CPU {}",
        rs_gpu.energy,
        rs_cpu.energy
    );
    assert!(gpu_sym.bond_sectors(n - 1).unwrap().len() >= 2);

    // 3. Dense and symmetric periodic TDVP on GPU match CPU amplitudes to < 1e-10.
    let n6 = 6;
    let h6 = periodic_heisenberg(n6, 0.6);
    let neel6: Vec<bool> = (0..n6).map(|q| q % 2 == 1).collect();
    let mps_opts = MpsOptions {
        max_bond: 12,
        cutoff: 1e-14,
    };
    let mut tdvp_cpu = PeriodicMps::basis(n6, &neel6, mps_opts);
    let mut tdvp_gpu = tdvp_cpu.clone();
    tdvp_gpu.set_gpu(Some(ctx.clone())).unwrap();
    tdvp_cpu.tdvp(&h6, 0.3, 4).unwrap();
    tdvp_gpu.tdvp(&h6, 0.3, 4).unwrap();
    let tdvp_diff = tdvp_cpu
        .to_amplitudes()
        .iter()
        .zip(tdvp_gpu.to_amplitudes())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0f64, f64::max);
    assert!(tdvp_diff < 1e-10, "dense GPU pTDVP diff {tdvp_diff}");

    let mut tdvp_sym_cpu = PeriodicMps::basis(n6, &neel6, mps_opts)
        .with_charges(&[Charge::weights(vec![1; n6])])
        .unwrap();
    let mut tdvp_sym_gpu = tdvp_sym_cpu.clone();
    tdvp_sym_gpu.set_gpu(Some(ctx)).unwrap();
    tdvp_sym_cpu.tdvp(&h6, 0.3, 4).unwrap();
    tdvp_sym_gpu.tdvp(&h6, 0.3, 4).unwrap();
    let tdvp_sym_diff = tdvp_sym_cpu
        .to_amplitudes()
        .iter()
        .zip(tdvp_sym_gpu.to_amplitudes())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0f64, f64::max);
    assert!(
        tdvp_sym_diff < 1e-10,
        "symmetric GPU pTDVP diff {tdvp_sym_diff}"
    );
}
