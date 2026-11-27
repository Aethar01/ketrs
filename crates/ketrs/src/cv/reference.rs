//! A brute-force Fock-space simulator for testing the CV engines: dense
//! amplitudes over `cutoff^m` occupation patterns, with every gate built
//! independently of the engines (matrix exponentials of the generators,
//! permanents for passive blocks). Slow and only as accurate as its cutoff;
//! the tests keep states well inside it and check the norm.

use super::circuit::{ModeCircuit, ModeOp};
use super::gate::{ModeGate, TwoModeGate};
use crate::error::Error;
use faer::{Mat, Side};
use num_complex::Complex64 as C64;

/// Extra Fock levels used when exponentiating generators that don't
/// conserve photon number, so truncation only distorts levels far above
/// the cutoff.
const PAD: usize = 60;

pub(crate) struct FockReference {
    n_modes: usize,
    cutoff: usize,
    amps: Vec<C64>,
}

fn c(re: f64) -> C64 {
    C64::new(re, 0.0)
}

impl FockReference {
    /// Run a bound, measurement-free circuit with `cutoff` levels per mode.
    pub(crate) fn run(circuit: &ModeCircuit, cutoff: usize) -> Result<Self, Error> {
        if !circuit.is_unitary() {
            return Err(Error::InvalidArgument(
                "the reference runs only circuits without measurements".into(),
            ));
        }
        let n_modes = circuit.n_modes();
        let per_mode = circuit
            .inputs()
            .iter()
            .map(|s| {
                s.fock_amplitudes(cutoff)
                    .ok_or_else(|| Error::InvalidArgument("the reference needs pure inputs".into()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let dim = cutoff.pow(n_modes as u32);
        let mut state = FockReference {
            n_modes,
            cutoff,
            amps: vec![C64::new(0.0, 0.0); dim],
        };
        for idx in 0..dim {
            state.amps[idx] = (0..n_modes)
                .map(|m| per_mode[m][state.digit(idx, m)])
                .product();
        }
        for op in circuit.ops() {
            let value = |a: super::Arg| a.evaluate(&[]);
            match op {
                ModeOp::Gate { gate, mode } => {
                    let u = state.single_mode(gate, &value)?;
                    state.apply(&[*mode], &u);
                }
                ModeOp::TwoModeGate { gate, modes } => {
                    let u = state.two_mode(gate, &value)?;
                    state.apply(modes, &u);
                }
                ModeOp::Passive { modes, matrix } => {
                    let u = state.passive(modes.len(), matrix);
                    state.apply(modes, &u);
                }
                ModeOp::Channel { .. } | ModeOp::Measure { .. } => unreachable!(),
            }
        }
        Ok(state)
    }

    fn stride(&self, mode: usize) -> usize {
        self.cutoff.pow(mode as u32)
    }

    fn digit(&self, idx: usize, mode: usize) -> usize {
        (idx / self.stride(mode)) % self.cutoff
    }

    fn index(&self, pattern: &[usize]) -> usize {
        pattern
            .iter()
            .enumerate()
            .map(|(m, &n)| n * self.stride(m))
            .sum()
    }

    pub(crate) fn amplitude(&self, pattern: &[usize]) -> C64 {
        self.amps[self.index(pattern)]
    }

    pub(crate) fn probability(&self, pattern: &[usize]) -> f64 {
        self.amplitude(pattern).norm_sqr()
    }

    pub(crate) fn norm_sqr(&self) -> f64 {
        self.amps.iter().map(|a| a.norm_sqr()).sum()
    }

    /// `⟨a^k⟩` on `mode`.
    pub(crate) fn moment(&self, mode: usize, k: usize) -> C64 {
        let s = self.stride(mode);
        (0..self.amps.len())
            .filter(|&i| self.digit(i, mode) >= k)
            .map(|i| {
                let n = self.digit(i, mode);
                let falling: f64 = (n + 1 - k..=n).map(|j| j as f64).product();
                self.amps[i - k * s].conj() * self.amps[i] * falling.sqrt()
            })
            .sum()
    }

    /// `⟨a_a a_b⟩`.
    pub(crate) fn correlation(&self, a: usize, b: usize) -> C64 {
        let (sa, sb) = (self.stride(a), self.stride(b));
        (0..self.amps.len())
            .filter(|&i| self.digit(i, a) > 0 && self.digit(i, b) > 0)
            .map(|i| {
                let w = (self.digit(i, a) * self.digit(i, b)) as f64;
                self.amps[i - sa - sb].conj() * self.amps[i] * w.sqrt()
            })
            .sum()
    }

    pub(crate) fn mean_photons(&self, mode: usize) -> f64 {
        (0..self.amps.len())
            .map(|i| self.digit(i, mode) as f64 * self.amps[i].norm_sqr())
            .sum()
    }

    pub(crate) fn mean_x(&self, mode: usize) -> f64 {
        2.0 * self.moment(mode, 1).re
    }

    pub(crate) fn mean_p(&self, mode: usize) -> f64 {
        2.0 * self.moment(mode, 1).im
    }

    /// `⟨x̂²⟩ = 2 Re⟨a²⟩ + 2⟨n̂⟩ + 1`.
    pub(crate) fn mean_x2(&self, mode: usize) -> f64 {
        2.0 * self.moment(mode, 2).re + 2.0 * self.mean_photons(mode) + 1.0
    }

    /// `⟨p̂²⟩ = −2 Re⟨a²⟩ + 2⟨n̂⟩ + 1`.
    pub(crate) fn mean_p2(&self, mode: usize) -> f64 {
        -2.0 * self.moment(mode, 2).re + 2.0 * self.mean_photons(mode) + 1.0
    }

    /// Apply `u`, a `d^k × d^k` row-major matrix on `modes` whose index is
    /// `Σ nᵢ dⁱ` over `modes[i]`.
    fn apply(&mut self, modes: &[usize], u: &[C64]) {
        let d = self.cutoff;
        let local = d.pow(modes.len() as u32);
        let offsets: Vec<usize> = (0..local)
            .map(|l| {
                modes
                    .iter()
                    .enumerate()
                    .map(|(i, &m)| (l / d.pow(i as u32)) % d * self.stride(m))
                    .sum()
            })
            .collect();
        let mut v = vec![C64::new(0.0, 0.0); local];
        for base in 0..self.amps.len() {
            if modes.iter().any(|&m| self.digit(base, m) != 0) {
                continue;
            }
            for (x, &o) in v.iter_mut().zip(&offsets) {
                *x = self.amps[base + o];
            }
            for (row, &o) in offsets.iter().enumerate() {
                self.amps[base + o] = (0..local).map(|col| u[row * local + col] * v[col]).sum();
            }
        }
        let _ = self.n_modes;
    }

    fn single_mode(
        &self,
        gate: &ModeGate,
        value: &impl Fn(super::Arg) -> Result<f64, Error>,
    ) -> Result<Vec<C64>, Error> {
        let d = self.cutoff;
        let big = d + PAD;
        let diagonal = |f: &dyn Fn(f64) -> C64| {
            let mut u = vec![C64::new(0.0, 0.0); d * d];
            for n in 0..d {
                u[n * d + n] = f(n as f64);
            }
            u
        };
        let a = lowering(big);
        let adag = a.adjoint().to_owned();
        let x = &a + &adag;
        let full = match *gate {
            ModeGate::Rotation { phi } => {
                let phi = value(phi)?;
                return Ok(diagonal(&|n| C64::from_polar(1.0, phi * n)));
            }
            ModeGate::Kerr { kappa } => {
                let kappa = value(kappa)?;
                return Ok(diagonal(&|n| C64::from_polar(1.0, kappa * n * n)));
            }
            ModeGate::Displacement { r, phi } => {
                let alpha = C64::from_polar(value(r)?, value(phi)?);
                let k = scale(&adag, alpha) - scale(&a, alpha.conj());
                hermitian_fn(&scale(&k, C64::i()), |l| C64::from_polar(1.0, -l))
            }
            ModeGate::Squeezing { r, phi } => {
                let z = C64::from_polar(value(r)?, value(phi)?);
                let k = scale(&(&a * &a), z.conj() / 2.0) - scale(&(&adag * &adag), z / 2.0);
                hermitian_fn(&scale(&k, C64::i()), |l| C64::from_polar(1.0, -l))
            }
            ModeGate::QuadraticPhase { s } => {
                let s = value(s)?;
                hermitian_fn(&x, |l| C64::from_polar(1.0, s * l * l / 4.0))
            }
            ModeGate::CubicPhase { gamma } => {
                let g = value(gamma)?;
                hermitian_fn(&x, |l| C64::from_polar(1.0, g * l * l * l / 6.0))
            }
        };
        Ok((0..d * d).map(|i| full[(i / d, i % d)]).collect())
    }

    fn two_mode(
        &self,
        gate: &TwoModeGate,
        value: &impl Fn(super::Arg) -> Result<f64, Error>,
    ) -> Result<Vec<C64>, Error> {
        let d = self.cutoff;
        let big = d + PAD;
        let mut u = vec![C64::new(0.0, 0.0); d.pow(4)];
        let at = |na: usize, nb: usize| na + d * nb;
        match *gate {
            TwoModeGate::BeamSplitter { theta, phi } => {
                // Conserves na + nb = N: exponentiate each block exactly.
                let (theta, phi) = (value(theta)?, value(phi)?);
                for total in 0..=2 * (d - 1) {
                    let mut k = Mat::<C64>::zeros(total + 1, total + 1);
                    for na in 0..=total {
                        let nb = total - na;
                        if na > 0 {
                            let w = ((na * (nb + 1)) as f64).sqrt();
                            k[(na - 1, na)] += C64::from_polar(theta * w, phi);
                        }
                        if nb > 0 {
                            let w = (((na + 1) * nb) as f64).sqrt();
                            k[(na + 1, na)] -= C64::from_polar(theta * w, -phi);
                        }
                    }
                    let block = hermitian_fn(&scale(&k, C64::i()), |l| C64::from_polar(1.0, -l));
                    for out in 0..=total {
                        for inp in 0..=total {
                            let (ob, ib) = (total - out, total - inp);
                            if out < d && ob < d && inp < d && ib < d {
                                u[at(out, ob) * d * d + at(inp, ib)] = block[(out, inp)];
                            }
                        }
                    }
                }
            }
            TwoModeGate::TwoModeSqueezing { r, phi } => {
                // Conserves na − nb: exponentiate each (padded) block.
                let z = C64::from_polar(value(r)?, value(phi)?);
                for diff in -(big as isize - 1)..=(big as isize - 1) {
                    let (oa, ob) = (diff.max(0) as usize, (-diff).max(0) as usize);
                    let len = big - diff.unsigned_abs();
                    let mut k = Mat::<C64>::zeros(len, len);
                    for n in 0..len {
                        let (na, nb) = (n + oa, n + ob);
                        if n > 0 {
                            k[(n - 1, n)] += z.conj() * ((na * nb) as f64).sqrt();
                        }
                        if n + 1 < len {
                            k[(n + 1, n)] -= z * (((na + 1) * (nb + 1)) as f64).sqrt();
                        }
                    }
                    let block = hermitian_fn(&scale(&k, C64::i()), |l| C64::from_polar(1.0, -l));
                    for out in 0..len {
                        for inp in 0..len {
                            let (a1, b1, a0, b0) = (out + oa, out + ob, inp + oa, inp + ob);
                            if a1 < d && b1 < d && a0 < d && b0 < d {
                                u[at(a1, b1) * d * d + at(a0, b0)] = block[(out, inp)];
                            }
                        }
                    }
                }
            }
            TwoModeGate::CrossKerr { kappa } => {
                let kappa = value(kappa)?;
                for na in 0..d {
                    for nb in 0..d {
                        let i = at(na, nb);
                        u[i * d * d + i] = C64::from_polar(1.0, kappa * (na * nb) as f64);
                    }
                }
            }
            TwoModeGate::ControlledX { s } => {
                // exp(−i s x̂_a p̂_b / 2) in the product of eigenbases.
                let s = value(s)?;
                let a = lowering(big);
                let x = &a + a.adjoint();
                let p = scale(&(&a - a.adjoint()), C64::new(0.0, -1.0));
                product_gate(
                    d,
                    &x,
                    &p,
                    |l, m| C64::from_polar(1.0, -s * l * m / 2.0),
                    &mut u,
                );
            }
            TwoModeGate::ControlledZ { s } => {
                let s = value(s)?;
                let a = lowering(big);
                let x = &a + a.adjoint();
                product_gate(
                    d,
                    &x,
                    &x,
                    |l, m| C64::from_polar(1.0, s * l * m / 2.0),
                    &mut u,
                );
            }
        }
        Ok(u)
    }

    /// A passive block from permanents: `⟨n'|U|n⟩ = perm(U[n', n]) /
    /// √(Π nᵢ! Π n'ⱼ!)`, rows of `U` repeated by `n'`, columns by `n`.
    fn passive(&self, k: usize, matrix: &[C64]) -> Vec<C64> {
        let d = self.cutoff;
        let local = d.pow(k as u32);
        let pattern =
            |l: usize| -> Vec<usize> { (0..k).map(|i| (l / d.pow(i as u32)) % d).collect() };
        let repeat = |p: &[usize]| -> Vec<usize> {
            p.iter()
                .enumerate()
                .flat_map(|(m, &n)| std::iter::repeat_n(m, n))
                .collect()
        };
        let fact = |p: &[usize]| -> f64 {
            p.iter()
                .map(|&n| (1..=n).map(|j| j as f64).product::<f64>())
                .product()
        };
        let mut u = vec![C64::new(0.0, 0.0); local * local];
        for out in 0..local {
            let po = pattern(out);
            for inp in 0..local {
                let pi = pattern(inp);
                if po.iter().sum::<usize>() != pi.iter().sum::<usize>() {
                    continue;
                }
                let (rows, cols) = (repeat(&po), repeat(&pi));
                let sub: Vec<C64> = rows
                    .iter()
                    .flat_map(|&r| cols.iter().map(move |&c| matrix[r * k + c]))
                    .collect();
                u[out * local + inp] = permanent(&sub, rows.len()) / (fact(&po) * fact(&pi)).sqrt();
            }
        }
        u
    }
}

/// `z·m`.
fn scale(m: &Mat<C64>, z: C64) -> Mat<C64> {
    Mat::from_fn(m.nrows(), m.ncols(), |i, j| m[(i, j)] * z)
}

/// The lowering operator on `n` levels.
fn lowering(n: usize) -> Mat<C64> {
    Mat::from_fn(n, n, |i, j| {
        if j == i + 1 {
            c((j as f64).sqrt())
        } else {
            c(0.0)
        }
    })
}

/// `f(H)` for Hermitian `H`, through its eigendecomposition.
fn hermitian_fn(h: &Mat<C64>, f: impl Fn(f64) -> C64) -> Mat<C64> {
    let eig = h
        .self_adjoint_eigen(Side::Lower)
        .expect("eigensolver converges");
    let (v, s) = (eig.U(), eig.S().column_vector());
    let n = h.nrows();
    let fs: Vec<C64> = (0..n).map(|k| f(s[k].re)).collect();
    Mat::from_fn(n, n, |i, j| {
        (0..n).map(|k| v[(i, k)] * fs[k] * v[(j, k)].conj()).sum()
    })
}

/// `exp(i f(A ⊗ B))` restricted to `d × d` levels per mode, for Hermitian
/// `A` on mode a and `B` on mode b given on more levels:
/// `Σ_{k,l} f(λ_k, μ_l) |v_k w_l⟩⟨v_k w_l|`.
fn product_gate(
    d: usize,
    a: &Mat<C64>,
    b: &Mat<C64>,
    phase: impl Fn(f64, f64) -> C64,
    u: &mut [C64],
) {
    let ea = a
        .self_adjoint_eigen(Side::Lower)
        .expect("eigensolver converges");
    let eb = b
        .self_adjoint_eigen(Side::Lower)
        .expect("eigensolver converges");
    let (va, la) = (ea.U(), ea.S().column_vector());
    let (vb, lb) = (eb.U(), eb.S().column_vector());
    let big = a.nrows();
    // t[(i, j, l)] = Σ_k va[i,k] conj(va[j,k]) phase(λ_k, μ_l).
    let mut t = vec![C64::new(0.0, 0.0); d * d * big];
    for i in 0..d {
        for j in 0..d {
            for l in 0..big {
                t[(i * d + j) * big + l] = (0..big)
                    .map(|k| va[(i, k)] * va[(j, k)].conj() * phase(la[k].re, lb[l].re))
                    .sum();
            }
        }
    }
    for i in 0..d {
        for j in 0..d {
            for p in 0..d {
                for q in 0..d {
                    let x: C64 = (0..big)
                        .map(|l| t[(i * d + j) * big + l] * vb[(p, l)] * vb[(q, l)].conj())
                        .sum();
                    u[(i + d * p) * d * d + (j + d * q)] = x;
                }
            }
        }
    }
}

/// Permanent of an `n × n` row-major matrix by Ryser's formula.
fn permanent(m: &[C64], n: usize) -> C64 {
    if n == 0 {
        return c(1.0);
    }
    let mut total = C64::new(0.0, 0.0);
    for subset in 1..(1usize << n) {
        let prod: C64 = (0..n)
            .map(|r| {
                (0..n)
                    .filter(|&col| subset >> col & 1 == 1)
                    .map(|col| m[r * n + col])
                    .sum::<C64>()
            })
            .product();
        let sign = if (n - subset.count_ones() as usize).is_multiple_of(2) {
            1.0
        } else {
            -1.0
        };
        total += prod * sign;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cv::ModeState;
    use std::f64::consts::{FRAC_PI_4, PI};

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() < tol, "{a} vs {b}");
    }

    fn close_c(a: C64, b: C64, tol: f64) {
        assert!((a - b).norm() < tol, "{a} vs {b}");
    }

    fn run(c: &ModeCircuit, cutoff: usize) -> FockReference {
        let s = FockReference::run(c, cutoff).unwrap();
        close(s.norm_sqr(), 1.0, 1e-9);
        s
    }

    #[test]
    fn displacement_makes_coherent_states() {
        let (r, phi) = (0.7, 0.9);
        let mut c = ModeCircuit::new(1);
        c.displace(r, phi, 0);
        let s = run(&c, 25);
        let want = ModeState::Coherent { r, phi }.fock_amplitudes(25).unwrap();
        for (n, &w) in want.iter().enumerate() {
            close_c(s.amplitude(&[n]), w, 1e-10);
        }
        close(s.mean_x(0), 2.0 * r * phi.cos(), 1e-10);
        close(s.mean_p(0), 2.0 * r * phi.sin(), 1e-10);
    }

    #[test]
    fn squeezed_vacuum_statistics() {
        let r = 0.6;
        let mut c = ModeCircuit::new(1);
        c.squeeze(r, 0.0, 0);
        let s = run(&c, 40);
        let want = ModeState::Squeezed { r, phi: 0.0 }
            .fock_amplitudes(40)
            .unwrap();
        for (n, &w) in want.iter().enumerate() {
            close_c(s.amplitude(&[n]), w, 1e-10);
        }
        close(s.mean_photons(0), r.sinh().powi(2), 1e-9);
        close(s.mean_x2(0), (-2.0 * r).exp(), 1e-9);
        close(s.mean_p2(0), (2.0 * r).exp(), 1e-9);

        // ⟨a²⟩ = −e^{iφ} sinh r cosh r fixes the sign of the angle.
        let phi = 1.1;
        let mut c = ModeCircuit::new(1);
        c.squeeze(r, phi, 0);
        let s = run(&c, 40);
        close_c(
            s.moment(0, 2),
            -C64::from_polar(r.sinh() * r.cosh(), phi),
            1e-9,
        );
    }

    #[test]
    fn rotation_and_kerr_phases() {
        let alpha = C64::from_polar(0.8, 0.3);
        let input = ModeState::Coherent { r: 0.8, phi: 0.3 };

        let mut c = ModeCircuit::new(1);
        c.set_input(0, input.clone()).unwrap().rotation(0.5, 0);
        close_c(
            run(&c, 30).moment(0, 1),
            alpha * C64::from_polar(1.0, 0.5),
            1e-10,
        );

        // K† a K = e^{iκ(2n̂+1)} a, so ⟨a⟩ = α e^{iκ} exp(|α|²(e^{2iκ} − 1)).
        let kappa = 0.3;
        let mut c = ModeCircuit::new(1);
        c.set_input(0, input).unwrap().kerr(kappa, 0);
        let want = alpha
            * C64::from_polar(1.0, kappa)
            * ((C64::from_polar(1.0, 2.0 * kappa) - 1.0) * alpha.norm_sqr()).exp();
        close_c(run(&c, 30).moment(0, 1), want, 1e-10);
    }

    #[test]
    fn hong_ou_mandel() {
        let mut c = ModeCircuit::new(2);
        c.set_input(0, ModeState::Fock(1)).unwrap();
        c.set_input(1, ModeState::Fock(1)).unwrap();
        c.beam_splitter(FRAC_PI_4, 0.0, 0, 1);
        let s = run(&c, 4);
        close(s.probability(&[1, 1]), 0.0, 1e-12);
        close(s.probability(&[2, 0]), 0.5, 1e-12);
        close(s.probability(&[0, 2]), 0.5, 1e-12);
    }

    #[test]
    fn beam_splitter_convention() {
        let (theta, phi) = (0.4, 0.7);
        let alpha = C64::from_polar(0.6, -0.2);
        let mut c = ModeCircuit::new(2);
        c.set_input(0, ModeState::Coherent { r: 0.6, phi: -0.2 })
            .unwrap()
            .beam_splitter(theta, phi, 0, 1);
        let s = run(&c, 15);
        close_c(s.moment(0, 1), alpha * theta.cos(), 1e-10);
        close_c(
            s.moment(1, 1),
            alpha * C64::from_polar(theta.sin(), phi),
            1e-10,
        );
    }

    fn bs_matrix(theta: f64, phi: f64) -> Vec<C64> {
        let (t, r) = (c(theta.cos()), C64::from_polar(theta.sin(), phi));
        vec![t, -r.conj(), r, t]
    }

    #[test]
    fn passive_blocks_match_beam_splitters() {
        let inputs = [ModeState::Fock(2), ModeState::Fock(1), ModeState::Fock(1)];
        let mut gates = ModeCircuit::new(3);
        let mut block = ModeCircuit::new(3);
        for (m, s) in inputs.iter().enumerate() {
            gates.set_input(m, s.clone()).unwrap();
            block.set_input(m, s.clone()).unwrap();
        }
        gates
            .beam_splitter(0.3, 0.5, 0, 1)
            .beam_splitter(1.1, -0.4, 1, 2);
        // The product U₂U₁ embedded in three modes.
        let embed = |u: &[C64], lo: usize| {
            let mut m = vec![c(0.0); 9];
            for i in 0..3 {
                m[i * 3 + i] = c(1.0);
            }
            for i in 0..2 {
                for j in 0..2 {
                    m[(lo + i) * 3 + lo + j] = u[i * 2 + j];
                }
            }
            m
        };
        let (u1, u2) = (
            embed(&bs_matrix(0.3, 0.5), 0),
            embed(&bs_matrix(1.1, -0.4), 1),
        );
        let u: Vec<C64> = (0..9)
            .map(|ij| {
                (0..3)
                    .map(|k| u2[ij / 3 * 3 + k] * u1[k * 3 + ij % 3])
                    .sum()
            })
            .collect();
        block.passive(&[0, 1, 2], u).unwrap();
        let (a, b) = (run(&gates, 5), run(&block, 5));
        for i in 0..a.amps.len() {
            close_c(a.amps[i], b.amps[i], 1e-12);
        }
    }

    #[test]
    fn two_mode_squeezed_vacuum() {
        let (r, phi) = (0.5, 0.8);
        let mut c = ModeCircuit::new(2);
        c.two_mode_squeeze(r, phi, 0, 1);
        let s = run(&c, 25);
        for na in 0..6 {
            for nb in 0..6 {
                let want = if na == nb {
                    r.tanh().powi(2 * na as i32) / r.cosh().powi(2)
                } else {
                    0.0
                };
                close(s.probability(&[na, nb]), want, 1e-10);
            }
        }
        close_c(
            s.correlation(0, 1),
            -C64::from_polar(r.sinh() * r.cosh(), phi),
            1e-9,
        );
    }

    fn two_coherent(c: &mut ModeCircuit) -> ([f64; 2], [f64; 2]) {
        let (a, b) = (C64::new(0.3, 0.2), C64::new(-0.1, 0.25));
        c.set_input(
            0,
            ModeState::Coherent {
                r: a.norm(),
                phi: a.arg(),
            },
        )
        .unwrap();
        c.set_input(
            1,
            ModeState::Coherent {
                r: b.norm(),
                phi: b.arg(),
            },
        )
        .unwrap();
        ([2.0 * a.re, 2.0 * b.re], [2.0 * a.im, 2.0 * b.im])
    }

    #[test]
    fn controlled_x_quadrature_map() {
        let s = 0.7;
        let mut c = ModeCircuit::new(2);
        let (x, p) = two_coherent(&mut c);
        c.cx(s, 0, 1);
        let st = run(&c, 22);
        close(st.mean_x(0), x[0], 1e-8);
        close(st.mean_x(1), x[1] + s * x[0], 1e-8);
        close(st.mean_p(0), p[0] - s * p[1], 1e-8);
        close(st.mean_p(1), p[1], 1e-8);
    }

    #[test]
    fn controlled_z_quadrature_map() {
        let s = -0.6;
        let mut c = ModeCircuit::new(2);
        let (x, p) = two_coherent(&mut c);
        c.cz(s, 0, 1);
        let st = run(&c, 22);
        close(st.mean_x(0), x[0], 1e-8);
        close(st.mean_x(1), x[1], 1e-8);
        close(st.mean_p(0), p[0] + s * x[1], 1e-8);
        close(st.mean_p(1), p[1] + s * x[0], 1e-8);
    }

    #[test]
    fn quadratic_and_cubic_phase_shear_p() {
        let input = ModeState::Coherent { r: 0.4, phi: 0.0 };
        let (x, x2) = (0.8, 0.8 * 0.8 + 1.0);

        let s = 0.5;
        let mut c = ModeCircuit::new(1);
        c.set_input(0, input.clone()).unwrap().quadratic_phase(s, 0);
        let st = run(&c, 30);
        close(st.mean_x(0), x, 1e-9);
        close(st.mean_p(0), s * x, 1e-9);

        let g = 0.1;
        let mut c = ModeCircuit::new(1);
        c.set_input(0, input).unwrap().cubic_phase(g, 0);
        let st = run(&c, 30);
        close(st.mean_x(0), x, 1e-9);
        close(st.mean_p(0), g * x2, 1e-9);
    }

    #[test]
    fn cross_kerr_phase() {
        let mut c = ModeCircuit::new(2);
        c.set_input(0, ModeState::Coherent { r: 0.5, phi: 0.0 })
            .unwrap()
            .set_input(1, ModeState::Fock(1))
            .unwrap()
            .cross_kerr(PI / 3.0, 0, 1);
        let st = run(&c, 15);
        close_c(st.moment(0, 1), C64::from_polar(0.5, PI / 3.0), 1e-10);
    }

    #[test]
    fn rejects_measurements_and_mixed_inputs() {
        let mut c = ModeCircuit::new(1);
        c.set_input(0, ModeState::Thermal { mean_photons: 0.3 })
            .unwrap();
        assert!(FockReference::run(&c, 5).is_err());
        let mut c = ModeCircuit::new(1);
        c.measure_x(0);
        assert!(FockReference::run(&c, 5).is_err());
    }
}
