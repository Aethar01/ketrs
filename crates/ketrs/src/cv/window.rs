//! Sliding-window (active-mode) exact trajectory sampler for local and
//! streaming CV circuits (e.g. time-bin loops, delay-line interferometers, and
//! 1D photonic chains).
//!
//! Performs static liveness analysis over [`ModeCircuit`] so each mode is
//! allocated into the active register only when first touched and freed
//! immediately upon destructive measurement (which resets it to vacuum) or
//! after its last operation. Memory and per-shot cost scale with `cutoff^W`
//! where `W` is the peak simultaneous active window width, rather than
//! `cutoff^n_modes`.

use super::circuit::{Measurement, ModeCircuit, ModeOp};
use super::fock::FockState;
use super::state::ModeState;
use crate::error::Error;
use num_complex::Complex64 as C64;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

/// Minimum shot count for parallelising independent window trajectories.
const WINDOW_PAR_MIN_SHOTS: usize = 32;

/// Sliding-window sampler for local / streaming CV circuits.
pub struct WindowSampler;

impl WindowSampler {
    /// Peak number of simultaneously active modes `W` needed to simulate
    /// `circuit` with the sliding-window schedule.
    ///
    /// A mode becomes active at the first operation that touches it and is
    /// freed immediately after any destructive measurement (which leaves it in
    /// vacuum) or after the last operation that touches it.
    pub fn max_window_width(circuit: &ModeCircuit) -> usize {
        let m = circuit.n_modes();
        let ops = circuit.ops();
        let last_use = compute_last_use(m, ops);
        let mut active = vec![false; m];
        let mut cur = 0usize;
        let mut max_w = 0usize;

        for (step, op) in ops.iter().enumerate() {
            for mode in op.modes() {
                if !active[mode] {
                    active[mode] = true;
                    cur += 1;
                    max_w = max_w.max(cur);
                }
            }
            if let ModeOp::Measure { mode, .. } = *op {
                if active[mode] {
                    active[mode] = false;
                    cur -= 1;
                }
            } else {
                for mode in op.modes() {
                    if last_use[mode] == Some(step) && active[mode] {
                        active[mode] = false;
                        cur -= 1;
                    }
                }
            }
        }
        max_w
    }

    /// Sample `shots` independent measurement trajectories from a bound
    /// `ModeCircuit` using the sliding-window cutoff Fock engine.
    ///
    /// Supports all gates, all 4 measurement types (homodyne, heterodyne,
    /// photon counting, threshold), and arbitrary mid-circuit feed-forward.
    pub fn sample<R: Rng + ?Sized>(
        circuit: &ModeCircuit,
        cutoff: usize,
        shots: usize,
        rng: &mut R,
    ) -> Result<Vec<Vec<f64>>, Error> {
        if circuit.n_parameters() > 0 {
            return Err(Error::InvalidArgument(
                "circuit has unbound parameters; bind it first".into(),
            ));
        }
        if cutoff == 0 {
            return Err(Error::InvalidArgument(
                "Fock cutoff must be at least 1".into(),
            ));
        }
        // Validate all inputs are pure Fock-representable.
        let input_amps: Vec<Vec<C64>> = circuit
            .inputs()
            .iter()
            .enumerate()
            .map(|(m, s)| {
                s.fock_amplitudes(cutoff).ok_or_else(|| {
                    Error::InvalidArgument(format!(
                        "WindowSampler requires pure inputs, mode {m} is {s:?}"
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        if shots == 0 {
            return Ok(Vec::new());
        }

        let m = circuit.n_modes();
        let ops = circuit.ops();
        let n_results = circuit.n_results();
        let last_use = compute_last_use(m, ops);
        let vac_vec = ModeState::Vacuum.fock_amplitudes(cutoff).unwrap();

        let run_one = |local_rng: &mut StdRng| -> Result<Vec<f64>, Error> {
            let mut active_modes: Vec<usize> = Vec::new();
            let mut ever_initialized = vec![false; m];
            let mut st = FockState::vacuum(0, cutoff)?;
            let mut results = Vec::with_capacity(n_results);

            for (step, op) in ops.iter().enumerate() {
                // Ensure every mode touched by `op` is present in `active_modes`.
                for mode in op.modes() {
                    if !active_modes.contains(&mode) {
                        let init = if !ever_initialized[mode] {
                            ever_initialized[mode] = true;
                            &input_amps[mode]
                        } else {
                            // Mode was previously destructively measured -> fresh vacuum!
                            &vac_vec
                        };
                        st = append_mode(&st, init);
                        active_modes.push(mode);
                    }
                }

                // Map global mode indices to local slot indices in `active_modes`.
                match op {
                    ModeOp::Gate { gate, mode } => {
                        let slot = slot_of(&active_modes, *mode);
                        let local_op = ModeOp::Gate {
                            gate: *gate,
                            mode: slot,
                        };
                        apply_local_unitary(&mut st, &local_op, &results)?;
                    }
                    ModeOp::TwoModeGate { gate, modes } => {
                        let local_op = ModeOp::TwoModeGate {
                            gate: *gate,
                            modes: [
                                slot_of(&active_modes, modes[0]),
                                slot_of(&active_modes, modes[1]),
                            ],
                        };
                        apply_local_unitary(&mut st, &local_op, &results)?;
                    }
                    ModeOp::Passive { modes, matrix } => {
                        let local_modes: Vec<usize> =
                            modes.iter().map(|&gm| slot_of(&active_modes, gm)).collect();
                        st.apply_passive(&local_modes, matrix)?;
                    }
                    ModeOp::Channel { channel, mode } => {
                        let slot = slot_of(&active_modes, *mode);
                        st.apply_channel(channel, slot, local_rng)?;
                    }
                    ModeOp::Measure {
                        mode, measurement, ..
                    } => {
                        let slot = slot_of(&active_modes, *mode);
                        match *measurement {
                            Measurement::PhotonCount => {
                                let k = st.measure_photons(slot, local_rng)?;
                                results.push(k as f64);
                            }
                            Measurement::Threshold => {
                                let c = st.measure_threshold(slot, local_rng)?;
                                results.push(if c { 1.0 } else { 0.0 });
                            }
                            Measurement::Homodyne { phi } => {
                                let angle = phi.evaluate(&results)?;
                                let q = st.measure_homodyne(slot, angle, local_rng)?;
                                results.push(q);
                            }
                            Measurement::Heterodyne => {
                                let (re, im) = st.measure_heterodyne(slot, local_rng)?;
                                results.push(re);
                                results.push(im);
                            }
                        }
                        // Measured slot is now in exact vacuum |0⟩ -> drop it from active window!
                        st = drop_vacuum_mode(&st, slot);
                        active_modes.remove(slot);
                        continue;
                    }
                }

                // Retire any mode whose last_use == step (trace out via destructive PNR discard).
                let to_retire: Vec<usize> = op
                    .modes()
                    .into_iter()
                    .filter(|&gm| last_use[gm] == Some(step))
                    .collect();
                for gm in to_retire {
                    if let Some(slot) = active_modes.iter().position(|&x| x == gm) {
                        let _ = st.measure_photons(slot, local_rng)?;
                        st = drop_vacuum_mode(&st, slot);
                        active_modes.remove(slot);
                    }
                }
            }
            Ok(results)
        };

        if shots >= WINDOW_PAR_MIN_SHOTS {
            let seeds: Vec<u64> = (0..shots).map(|_| rng.random()).collect();
            seeds
                .into_par_iter()
                .map(|seed| {
                    let mut local_rng = StdRng::seed_from_u64(seed);
                    run_one(&mut local_rng)
                })
                .collect()
        } else {
            let mut out = Vec::with_capacity(shots);
            for _ in 0..shots {
                let mut local_rng = StdRng::seed_from_u64(rng.random());
                out.push(run_one(&mut local_rng)?);
            }
            Ok(out)
        }
    }
}

fn compute_last_use(n_modes: usize, ops: &[ModeOp]) -> Vec<Option<usize>> {
    let mut last = vec![None; n_modes];
    for (step, op) in ops.iter().enumerate() {
        for mode in op.modes() {
            last[mode] = Some(step);
        }
    }
    last
}

#[inline]
fn slot_of(active: &[usize], global_mode: usize) -> usize {
    active
        .iter()
        .position(|&m| m == global_mode)
        .expect("mode must be active")
}

/// Tensor-append a new mode (as the most significant digit `new_slot = st.n_modes()`)
/// with 1-mode amplitudes `mode_amps`.
fn append_mode(st: &FockState, mode_amps: &[C64]) -> FockState {
    let d = st.cutoff();
    let old = st.amplitudes();
    let old_len = old.len();
    let mut amps = vec![C64::new(0.0, 0.0); old_len * d];
    for (n, &cn) in mode_amps.iter().enumerate() {
        if cn.re == 0.0 && cn.im == 0.0 {
            continue;
        }
        let dst = &mut amps[n * old_len..(n + 1) * old_len];
        for (y, &x) in dst.iter_mut().zip(old) {
            *y = x * cn;
        }
    }
    FockState::from_raw(st.n_modes() + 1, d, amps)
}

/// Remove `slot` from `st` assuming `slot` is already projected onto vacuum `|0⟩`.
fn drop_vacuum_mode(st: &FockState, slot: usize) -> FockState {
    let d = st.cutoff();
    let old_m = st.n_modes();
    let stride = d.pow(slot as u32);
    let block = stride * d;
    let old = st.amplitudes();
    let n_outer = old.len() / block;
    let mut amps = Vec::with_capacity(old.len() / d);
    for outer in 0..n_outer {
        let base = outer * block;
        // Keep only n_slot == 0 slice: [base .. base + stride].
        amps.extend_from_slice(&old[base..base + stride]);
    }
    FockState::from_raw(old_m - 1, d, amps)
}

fn apply_local_unitary(st: &mut FockState, op: &ModeOp, results: &[f64]) -> Result<(), Error> {
    st.apply_op_with_results(op, results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cv::FixedNFockState;
    use std::collections::HashMap;
    use std::f64::consts::FRAC_PI_4;

    #[test]
    fn time_bin_loop_window_width_is_two() {
        // 100-mode time-bin loop (single delay line): only 2 modes ever live at once!
        let m = 100;
        let mut c = ModeCircuit::new(m);
        for i in 0..m {
            c.set_input(i, ModeState::Fock(1)).unwrap();
        }
        for i in 0..(m - 1) {
            c.beam_splitter(FRAC_PI_4, 0.0, i, i + 1);
            c.measure_photons(i);
        }
        c.measure_photons(m - 1);

        assert_eq!(WindowSampler::max_window_width(&c), 2);

        let mut rng = StdRng::seed_from_u64(42);
        let samples = WindowSampler::sample(&c, 8, 50, &mut rng).unwrap();
        assert_eq!(samples.len(), 50);

        // Inline loss channels inside ModeCircuit keep window width == 2 and
        // match exact expected photon survival η = 0.7!
        let mut noisy = ModeCircuit::new(10);
        for i in 0..10 {
            noisy.set_input(i, ModeState::Fock(1)).unwrap();
        }
        for i in 0..9 {
            noisy
                .apply_channel(crate::cv::CvChannel::Loss { eta: 0.7 }, i)
                .unwrap();
            noisy.beam_splitter(FRAC_PI_4, 0.0, i, i + 1);
            noisy.measure_photons(i);
        }
        noisy
            .apply_channel(crate::cv::CvChannel::Loss { eta: 0.7 }, 9)
            .unwrap();
        noisy.measure_photons(9);
        assert_eq!(WindowSampler::max_window_width(&noisy), 2);
        let n_samples = WindowSampler::sample(&noisy, 8, 400, &mut rng).unwrap();
        let mean_total: f64 = n_samples.iter().map(|r| r.iter().sum::<f64>()).sum::<f64>() / 400.0;
        // Analytic recurrence for loss η = 0.7 before each 50:50 BS:
        // m_0 = 1, d_i = m_{i+1} = 0.5 * (η * m_i + 1) for i < 9, d_9 = η * m_9.
        let mut m_cur = 1.0f64;
        let mut exact_total = 0.0f64;
        for _ in 0..9 {
            let next = 0.5 * (0.7 * m_cur + 1.0);
            exact_total += next;
            m_cur = next;
        }
        exact_total += 0.7 * m_cur;
        assert!(
            (mean_total - exact_total).abs() < 0.25,
            "expected ~{exact_total:.4} surviving photons, got {mean_total:.4}"
        );
        for row in &samples {
            assert_eq!(row.len(), m);
            // Total photons conserved across all 100 time bins (small cutoff leakage possible
            // only if >= 8 photons bunch into one bin, astronomically rare for 50:50 chain).
            let sum: f64 = row.iter().sum();
            assert!((sum - m as f64).abs() < 1e-9, "sum={sum}");
        }
    }

    #[test]
    fn window_sampler_matches_exact_fixed_n_tvd() {
        // 4-mode nearest-neighbour chain with 4 photons -> window width 2!
        let mut c = ModeCircuit::new(4);
        for i in 0..4 {
            c.set_input(i, ModeState::Fock(1)).unwrap();
        }
        c.beam_splitter(0.6, 0.2, 0, 1);
        c.measure_photons(0);
        c.beam_splitter(0.8, -0.4, 1, 2);
        c.measure_photons(1);
        c.beam_splitter(0.5, 0.7, 2, 3);
        c.measure_photons(2);
        c.measure_photons(3);

        assert_eq!(WindowSampler::max_window_width(&c), 2);

        // Build exact unitary version (measurements at the end commute since
        // no later gate touches a measured mode!).
        let mut c_unitary = ModeCircuit::new(4);
        for i in 0..4 {
            c_unitary.set_input(i, ModeState::Fock(1)).unwrap();
        }
        c_unitary
            .beam_splitter(0.6, 0.2, 0, 1)
            .beam_splitter(0.8, -0.4, 1, 2)
            .beam_splitter(0.5, 0.7, 2, 3);
        let exact = FixedNFockState::from_circuit(&c_unitary).unwrap();

        let mut rng = StdRng::seed_from_u64(7);
        let shots = 30_000;
        let samples = WindowSampler::sample(&c, 6, shots, &mut rng).unwrap();
        let mut counts: HashMap<Vec<usize>, usize> = HashMap::new();
        for row in samples {
            let pat: Vec<usize> = row.into_iter().map(|x| x as usize).collect();
            *counts.entry(pat).or_insert(0) += 1;
        }

        let mut tvd = 0.0;
        for pat in exact.basis() {
            let p_exact = exact.probability(&pat).unwrap();
            let p_emp = (*counts.get(&pat).unwrap_or(&0) as f64) / (shots as f64);
            tvd += (p_emp - p_exact).abs();
        }
        tvd *= 0.5;
        assert!(tvd < 0.02, "WindowSampler TVD={tvd}");
    }
}
