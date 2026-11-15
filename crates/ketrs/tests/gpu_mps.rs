//! MPS sweeps on a GPU (`Mps::set_gpu`), checked against the CPU path.
//! Tests skip (passing) when no Vulkan GPU with 64-bit floats is available.
#![cfg(feature = "gpu")]

use ketrs::{
    Complex64 as C64, DmrgOptions, GpuContext, Mps, MpsOptions, Pauli, PauliString, PauliSum,
};
use std::sync::Arc;

fn gpu() -> Option<Arc<GpuContext>> {
    match GpuContext::new() {
        Ok(ctx) if ctx.supports_mps() => Some(ctx),
        Ok(ctx) => {
            eprintln!(
                "skipping GPU MPS test: {} lacks 64-bit floats",
                ctx.device_name()
            );
            None
        }
        Err(e) => {
            eprintln!("skipping GPU MPS test: {e}");
            None
        }
    }
}

/// Couplings up to five sites apart, fields along every axis and
/// four-qubit strings spanning six sites, like hops through link registers.
fn hamiltonian(n: usize) -> PauliSum {
    let paulis = [Pauli::X, Pauli::Y, Pauli::Z];
    let mut h = PauliSum::new();
    for i in 0..n {
        h.add(
            0.3 + 0.05 * i as f64,
            PauliString::new([(i, paulis[i % 3])]).unwrap(),
        );
        for j in i + 1..n.min(i + 6) {
            let c = 1.0 / (j - i) as f64;
            h.add(c, PauliString::new([(i, Pauli::Z), (j, Pauli::Z)]).unwrap());
            h.add(
                0.5 * c,
                PauliString::new([(i, Pauli::X), (j, Pauli::X)]).unwrap(),
            );
            h.add(
                0.5 * c,
                PauliString::new([(i, Pauli::Y), (j, Pauli::Y)]).unwrap(),
            );
        }
        if i + 5 < n {
            for (a, b) in [(Pauli::X, Pauli::X), (Pauli::Y, Pauli::Y)] {
                let ops = [(i, a), (i + 2, Pauli::Z), (i + 3, Pauli::Z), (i + 5, b)];
                h.add(0.25, PauliString::new(ops).unwrap());
            }
        }
    }
    h.add(1.5, PauliString::identity());
    h
}

fn max_diff(a: &[C64], b: &[C64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).norm())
        .fold(0.0, f64::max)
}

#[test]
fn tdvp_on_gpu_matches_cpu() {
    let Some(ctx) = gpu() else { return };
    let n = 12;
    let h = hamiltonian(n);
    let options = MpsOptions {
        max_bond: 20,
        cutoff: 1e-12,
    };
    // A random start with full bonds, and a Néel state whose bonds grow
    // through the subspace expansion.
    let neel: Vec<bool> = (0..n).map(|q| q % 2 == 0).collect();
    for start in [
        Mps::random(n, 20, 7, options),
        Mps::basis(n, &neel, options),
    ] {
        let mut cpu = start.clone();
        let mut gpu = start;
        gpu.set_gpu(Some(ctx.clone())).unwrap();
        cpu.tdvp(&h, 0.6, 3).unwrap();
        gpu.tdvp(&h, 0.6, 3).unwrap();
        assert_eq!(cpu.bond_dimensions(), gpu.bond_dimensions());
        assert!(cpu.bond_dimensions().contains(&20), "truncates");
        let diff = max_diff(&cpu.to_amplitudes(), &gpu.to_amplitudes());
        assert!(diff < 1e-9, "{diff}");
        let (e_cpu, e_gpu) = (cpu.expectation(&h).unwrap(), gpu.expectation(&h).unwrap());
        assert!((e_cpu - e_gpu).abs() < 1e-10, "{e_cpu} {e_gpu}");
        assert!((cpu.truncation_error() - gpu.truncation_error()).abs() < 1e-10);
    }
}

#[test]
fn dmrg_on_gpu_matches_cpu() {
    let Some(ctx) = gpu() else { return };
    let n = 12;
    let h = hamiltonian(n);
    let options = DmrgOptions {
        max_bond: 24,
        cutoff: 1e-12,
        max_sweeps: 6,
        ..Default::default()
    };
    let start = Mps::random(n, 4, 11, MpsOptions::default());
    let mut cpu = start.clone();
    let mut gpu = start;
    gpu.set_gpu(Some(ctx)).unwrap();
    let want = cpu.dmrg(&h, &options).unwrap();
    let got = gpu.dmrg(&h, &options).unwrap();
    assert_eq!(want.sweep_energies.len(), got.sweep_energies.len());
    for (a, b) in want.sweep_energies.iter().zip(&got.sweep_energies) {
        assert!((a - b).abs() < 1e-9, "{want:?} {got:?}");
    }
    assert!((gpu.expectation(&h).unwrap() - got.energy).abs() < 1e-8);
    // The ground state is unique, so the states agree up to a phase.
    let (a, b) = (cpu.to_amplitudes(), gpu.to_amplitudes());
    let overlap: C64 = a.iter().zip(&b).map(|(x, y)| x.conj() * y).sum();
    assert!((overlap.norm() - 1.0).abs() < 1e-8, "{overlap}");
}

#[test]
fn dmrg_on_gpu_converges_with_a_spectrum_far_from_zero() {
    // Hops under a Gauss-law-like penalty `1e6 Σ_{i<j} Zᵢ Zⱼ`: the spectrum
    // sits near −6e6, where the default Lanczos tolerance of 1e-10 is below
    // rounding. GPU DMRG used to abort ("Lanczos did not converge in 1000
    // iterations") on a lattice gauge theory like this.
    let Some(ctx) = gpu() else { return };
    let n = 12;
    let mut h = PauliSum::new();
    for q in 0..n - 1 {
        for p in [Pauli::X, Pauli::Y] {
            h.add(1.0, PauliString::new([(q, p), (q + 1, p)]).unwrap());
        }
    }
    for q in 0..n {
        h.add(
            0.3 * (q % 3) as f64,
            PauliString::new([(q, Pauli::Z)]).unwrap(),
        );
        for j in q + 1..n {
            h.add(
                1e6,
                PauliString::new([(q, Pauli::Z), (j, Pauli::Z)]).unwrap(),
            );
        }
    }
    let neel: Vec<bool> = (0..n).map(|q| q % 2 == 1).collect();
    let options = DmrgOptions {
        max_bond: 32,
        ..Default::default()
    };
    let start = Mps::basis(n, &neel, MpsOptions::default());
    let (mut cpu, mut gpu) = (start.clone(), start);
    gpu.set_gpu(Some(ctx)).unwrap();
    let want = cpu.dmrg(&h, &options).unwrap();
    let got = gpu.dmrg(&h, &options).unwrap();
    assert_eq!(got.unconverged_solves, 0);
    assert!((want.energy - got.energy).abs() < 1e-6, "{want:?} {got:?}");
}

#[test]
fn gpu_setting_is_kept_by_clones_and_can_be_cleared() {
    let Some(ctx) = gpu() else { return };
    let mut mps = Mps::zero(4, MpsOptions::default());
    assert!(mps.gpu().is_none());
    mps.set_gpu(Some(ctx)).unwrap();
    assert!(mps.clone().gpu().is_some());
    mps.set_gpu(None).unwrap();
    assert!(mps.gpu().is_none());
}
