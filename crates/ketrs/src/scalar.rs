//! Amplitude types. The statevector engine is generic over [`Scalar`], so the
//! same kernels run on complex amplitudes or, for real circuits, on real
//! ones at half the memory, each in double or single precision.
//!
//! Gate matrices, coefficients and reductions (norms, probabilities,
//! expectation values) are computed in `f64` and converted, so single
//! precision only rounds the stored amplitudes and the arithmetic on them:
//! errors grow to about `1e-6` instead of `1e-15`.

use num_complex::{Complex32 as C32, Complex64 as C64};
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub};

/// Imaginary parts at most this large count as zero when converting to a
/// real type, absorbing rounding in values such as `cis(π)`.
const REAL_TOLERANCE: f64 = 1e-14;

/// A field of amplitudes the statevector engine can run on.
pub trait Scalar:
    Copy
    + Send
    + Sync
    + Debug
    + PartialEq
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + MulAssign
    + Sum
    + 'static
{
    /// Additive identity.
    const ZERO: Self;
    /// Multiplicative identity.
    const ONE: Self;
    /// Machine epsilon of the underlying real type, for tolerances.
    const EPSILON: f64;

    /// Convert a complex number, or `None` if this type can't represent it.
    fn from_complex(c: C64) -> Option<Self>;
    /// Widen to a complex number.
    fn to_complex(self) -> C64;
    /// Embed a real number.
    fn from_real(x: f64) -> Self;
    /// Complex conjugate (the identity for real types).
    fn conj(self) -> Self;
    /// `|z|²`.
    fn norm_sqr(self) -> f64;

    /// `(lo[k], hi[k]) ← m · (lo[k], hi[k])` for every `k`: the innermost
    /// loop of a general one-qubit gate. Types may override it with SIMD.
    #[inline]
    fn apply_pair(m: &[[Self; 2]; 2], lo: &mut [Self], hi: &mut [Self]) {
        for (a, b) in lo.iter_mut().zip(hi) {
            let (x, y) = (*a, *b);
            *a = m[0][0] * x + m[0][1] * y;
            *b = m[1][0] * x + m[1][1] * y;
        }
    }

    /// The same for a two-qubit gate: `m` applied to `(s[0][k], …,
    /// s[3][k])` for every `k`.
    #[inline]
    fn apply_quad(m: &[[Self; 4]; 4], s: [&mut [Self]; 4]) {
        let [s0, s1, s2, s3] = s;
        let quads = s0.iter_mut().zip(s1.iter_mut()).zip(s2.iter_mut()).zip(s3);
        for (((a0, a1), a2), a3) in quads {
            let (v0, v1, v2, v3) = (*a0, *a1, *a2, *a3);
            *a0 = m[0][0] * v0 + m[0][1] * v1 + m[0][2] * v2 + m[0][3] * v3;
            *a1 = m[1][0] * v0 + m[1][1] * v1 + m[1][2] * v2 + m[1][3] * v3;
            *a2 = m[2][0] * v0 + m[2][1] * v1 + m[2][2] * v2 + m[2][3] * v3;
            *a3 = m[3][0] * v0 + m[3][1] * v1 + m[3][2] * v2 + m[3][3] * v3;
        }
    }

    /// Multiply every element of `s` by `f`.
    #[inline]
    fn scale_slice(s: &mut [Self], f: Self) {
        s.iter_mut().for_each(|a| *a *= f);
    }

    /// Multiply `s` elementwise by `factors`, of the same length.
    #[inline]
    fn mul_slices(s: &mut [Self], factors: &[Self]) {
        s.iter_mut().zip(factors).for_each(|(a, &f)| *a *= f);
    }

    /// Multiply `s` elementwise by `pattern` repeated, for diagonal gates
    /// on low qubits. `s.len()` is a multiple of `pattern.len()`.
    #[inline]
    fn scale_pattern(s: &mut [Self], pattern: &[Self]) {
        for chunk in s.chunks_exact_mut(pattern.len()) {
            chunk.iter_mut().zip(pattern).for_each(|(a, &f)| *a *= f);
        }
    }

    /// [`apply_pair`](Self::apply_pair) on every pair `(s[i], s[i + stride])`
    /// of a block whose length is a multiple of `2·stride`: gates on low
    /// qubits, whose pairs are too short to hand out one run at a time.
    #[inline]
    fn apply_pair_strided(m: &[[Self; 2]; 2], s: &mut [Self], stride: usize) {
        for chunk in s.chunks_exact_mut(2 * stride) {
            let (lo, hi) = chunk.split_at_mut(stride);
            Self::apply_pair(m, lo, hi);
        }
    }

    /// [`apply_quad`](Self::apply_quad) on matching slices `lo` and `hi`
    /// (the high qubit clear and set) whose low qubit has stride `half`:
    /// every `(lo[i], lo[i + half], hi[i], hi[i + half])`. Lengths are
    /// multiples of `2·half`.
    #[inline]
    fn apply_quad_strided(m: &[[Self; 4]; 4], lo: &mut [Self], hi: &mut [Self], half: usize) {
        let pairs = lo
            .chunks_exact_mut(2 * half)
            .zip(hi.chunks_exact_mut(2 * half));
        for (l, h) in pairs {
            let (s00, s01) = l.split_at_mut(half);
            let (s10, s11) = h.split_at_mut(half);
            Self::apply_quad(m, [s00, s01, s10, s11]);
        }
    }
}

/// A complex amplitude type, which can hold any state or operator.
pub trait ComplexScalar: Scalar {}

impl ComplexScalar for C64 {}
impl ComplexScalar for C32 {}

impl Scalar for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const EPSILON: f64 = f64::EPSILON;

    fn from_complex(c: C64) -> Option<Self> {
        (c.im.abs() <= REAL_TOLERANCE).then_some(c.re)
    }
    fn to_complex(self) -> C64 {
        C64::new(self, 0.0)
    }
    fn from_real(x: f64) -> Self {
        x
    }
    fn conj(self) -> Self {
        self
    }
    fn norm_sqr(self) -> f64 {
        self * self
    }
}

impl Scalar for C64 {
    const ZERO: Self = C64::new(0.0, 0.0);
    const ONE: Self = C64::new(1.0, 0.0);
    const EPSILON: f64 = f64::EPSILON;

    fn from_complex(c: C64) -> Option<Self> {
        Some(c)
    }
    fn to_complex(self) -> C64 {
        self
    }
    fn from_real(x: f64) -> Self {
        C64::new(x, 0.0)
    }
    fn conj(self) -> Self {
        C64::conj(&self)
    }
    fn norm_sqr(self) -> f64 {
        C64::norm_sqr(&self)
    }

    #[inline]
    fn apply_pair(m: &[[Self; 2]; 2], lo: &mut [Self], hi: &mut [Self]) {
        #[cfg(target_arch = "x86_64")]
        if simd::avx2_fma() {
            // SAFETY: the CPU supports AVX2 and FMA.
            unsafe { simd::apply_pair_avx2(m, lo, hi) };
            return;
        }
        for (a, b) in lo.iter_mut().zip(hi) {
            let (x, y) = (*a, *b);
            *a = m[0][0] * x + m[0][1] * y;
            *b = m[1][0] * x + m[1][1] * y;
        }
    }

    #[inline]
    fn apply_quad(m: &[[Self; 4]; 4], s: [&mut [Self]; 4]) {
        #[cfg(target_arch = "x86_64")]
        if simd::avx2_fma() {
            // SAFETY: the CPU supports AVX2 and FMA.
            unsafe { simd::apply_quad_avx2(m, s) };
            return;
        }
        let [s0, s1, s2, s3] = s;
        let quads = s0.iter_mut().zip(s1.iter_mut()).zip(s2.iter_mut()).zip(s3);
        for (((a0, a1), a2), a3) in quads {
            let (v0, v1, v2, v3) = (*a0, *a1, *a2, *a3);
            *a0 = m[0][0] * v0 + m[0][1] * v1 + m[0][2] * v2 + m[0][3] * v3;
            *a1 = m[1][0] * v0 + m[1][1] * v1 + m[1][2] * v2 + m[1][3] * v3;
            *a2 = m[2][0] * v0 + m[2][1] * v1 + m[2][2] * v2 + m[2][3] * v3;
            *a3 = m[3][0] * v0 + m[3][1] * v1 + m[3][2] * v2 + m[3][3] * v3;
        }
    }

    #[inline]
    fn scale_slice(s: &mut [Self], f: Self) {
        #[cfg(target_arch = "x86_64")]
        if simd::avx2_fma() {
            // SAFETY: the CPU supports AVX2 and FMA.
            unsafe { simd::scale_avx2(s, f) };
            return;
        }
        s.iter_mut().for_each(|a| *a *= f);
    }

    #[inline]
    fn mul_slices(s: &mut [Self], factors: &[Self]) {
        #[cfg(target_arch = "x86_64")]
        if simd::avx2_fma() {
            // SAFETY: the CPU supports AVX2 and FMA.
            unsafe { simd::mul_slices_avx2(s, factors) };
            return;
        }
        s.iter_mut().zip(factors).for_each(|(a, &f)| *a *= f);
    }

    #[inline]
    fn scale_pattern(s: &mut [Self], pattern: &[Self]) {
        #[cfg(target_arch = "x86_64")]
        if simd::avx2_fma() && (2..=simd::MAX_PATTERN).contains(&pattern.len()) {
            // SAFETY: the CPU supports AVX2 and FMA.
            unsafe { simd::scale_pattern_avx2(s, pattern) };
            return;
        }
        for chunk in s.chunks_exact_mut(pattern.len()) {
            chunk.iter_mut().zip(pattern).for_each(|(a, &f)| *a *= f);
        }
    }

    #[inline]
    fn apply_pair_strided(m: &[[Self; 2]; 2], s: &mut [Self], stride: usize) {
        #[cfg(target_arch = "x86_64")]
        if simd::avx2_fma() {
            // SAFETY: the CPU supports AVX2 and FMA.
            unsafe { simd::apply_pair_strided_avx2(m, s, stride) };
            return;
        }
        for chunk in s.chunks_exact_mut(2 * stride) {
            let (lo, hi) = chunk.split_at_mut(stride);
            for (a, b) in lo.iter_mut().zip(hi) {
                let (x, y) = (*a, *b);
                *a = m[0][0] * x + m[0][1] * y;
                *b = m[1][0] * x + m[1][1] * y;
            }
        }
    }

    #[inline]
    fn apply_quad_strided(m: &[[Self; 4]; 4], lo: &mut [Self], hi: &mut [Self], half: usize) {
        #[cfg(target_arch = "x86_64")]
        if simd::avx2_fma() {
            // SAFETY: the CPU supports AVX2 and FMA.
            unsafe { simd::apply_quad_strided_avx2(m, lo, hi, half) };
            return;
        }
        let pairs = lo
            .chunks_exact_mut(2 * half)
            .zip(hi.chunks_exact_mut(2 * half));
        for (l, h) in pairs {
            let (s00, s01) = l.split_at_mut(half);
            let (s10, s11) = h.split_at_mut(half);
            let quads = s00
                .iter_mut()
                .zip(s01.iter_mut())
                .zip(s10.iter_mut())
                .zip(s11);
            for (((a0, a1), a2), a3) in quads {
                let v = [*a0, *a1, *a2, *a3];
                let row = |r: usize| (0..4).map(|c| m[r][c] * v[c]).sum::<C64>();
                (*a0, *a1, *a2, *a3) = (row(0), row(1), row(2), row(3));
            }
        }
    }
}

impl Scalar for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const EPSILON: f64 = f32::EPSILON as f64;

    fn from_complex(c: C64) -> Option<Self> {
        (c.im.abs() <= REAL_TOLERANCE).then_some(c.re as f32)
    }
    fn to_complex(self) -> C64 {
        C64::new(self as f64, 0.0)
    }
    fn from_real(x: f64) -> Self {
        x as f32
    }
    fn conj(self) -> Self {
        self
    }
    fn norm_sqr(self) -> f64 {
        let x = self as f64;
        x * x
    }
}

impl Scalar for C32 {
    const ZERO: Self = C32::new(0.0, 0.0);
    const ONE: Self = C32::new(1.0, 0.0);
    const EPSILON: f64 = f32::EPSILON as f64;

    fn from_complex(c: C64) -> Option<Self> {
        Some(C32::new(c.re as f32, c.im as f32))
    }
    fn to_complex(self) -> C64 {
        C64::new(self.re as f64, self.im as f64)
    }
    fn from_real(x: f64) -> Self {
        C32::new(x as f32, 0.0)
    }
    fn conj(self) -> Self {
        C32::conj(&self)
    }
    fn norm_sqr(self) -> f64 {
        self.to_complex().norm_sqr()
    }
}

/// Hand-vectorised kernels for complex doubles, chosen at run time.
#[cfg(target_arch = "x86_64")]
mod simd {
    use num_complex::Complex64 as C64;
    use std::arch::x86_64::*;
    use std::sync::OnceLock;

    /// Whether the CPU has AVX2 and FMA (checked once).
    #[inline]
    pub(super) fn avx2_fma() -> bool {
        static DETECTED: OnceLock<bool> = OnceLock::new();
        *DETECTED.get_or_init(|| {
            std::env::var_os("KETRS_NO_SIMD").is_none()
                && is_x86_feature_detected!("avx2")
                && is_x86_feature_detected!("fma")
        })
    }

    /// `m · x` for two complex numbers `x` in one register and `m` given as
    /// `(re, re, re, re)` and `(im, im, im, im)`:
    /// `(re·x_re − im·x_im, re·x_im + im·x_re)` by one `fmaddsub`.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    fn cmul(re: __m256d, im: __m256d, x: __m256d) -> __m256d {
        let swapped = _mm256_permute_pd::<0b0101>(x);
        _mm256_fmaddsub_pd(re, x, _mm256_mul_pd(im, swapped))
    }

    /// [`Scalar::apply_pair`](super::Scalar::apply_pair) for complex
    /// doubles, two amplitudes per instruction.
    ///
    /// # Safety
    /// The CPU must support AVX2 and FMA.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn apply_pair_avx2(m: &[[C64; 2]; 2], lo: &mut [C64], hi: &mut [C64]) {
        let n = lo.len().min(hi.len());
        let split = |c: C64| (_mm256_set1_pd(c.re), _mm256_set1_pd(c.im));
        let ((r00, i00), (r01, i01)) = (split(m[0][0]), split(m[0][1]));
        let ((r10, i10), (r11, i11)) = (split(m[1][0]), split(m[1][1]));
        let (lo_ptr, hi_ptr) = (lo.as_mut_ptr().cast::<f64>(), hi.as_mut_ptr().cast::<f64>());
        let mut k = 0;
        if m.iter().flatten().all(|c| c.im == 0.0) {
            // Real matrices (Ry, H) need no cross terms: half the work.
            while k + 2 <= n {
                // SAFETY: as below.
                unsafe {
                    let x = _mm256_loadu_pd(lo_ptr.add(2 * k));
                    let y = _mm256_loadu_pd(hi_ptr.add(2 * k));
                    let a = _mm256_fmadd_pd(r01, y, _mm256_mul_pd(r00, x));
                    let b = _mm256_fmadd_pd(r11, y, _mm256_mul_pd(r10, x));
                    _mm256_storeu_pd(lo_ptr.add(2 * k), a);
                    _mm256_storeu_pd(hi_ptr.add(2 * k), b);
                }
                k += 2;
            }
        } else if m[0][0].im == 0.0 && m[1][1].im == 0.0 && m[0][1].re == 0.0 && m[1][0].re == 0.0 {
            // Real diagonal, imaginary off-diagonal (Rx, Y): `i t · y` is
            // `(−t y_im, t y_re)`, the swapped pair times `(−t, t)`.
            let imaginary = |t: f64| _mm256_setr_pd(-t, t, -t, t);
            let (t01, t10) = (imaginary(m[0][1].im), imaginary(m[1][0].im));
            while k + 2 <= n {
                // SAFETY: as below.
                unsafe {
                    let x = _mm256_loadu_pd(lo_ptr.add(2 * k));
                    let y = _mm256_loadu_pd(hi_ptr.add(2 * k));
                    let (xs, ys) = (
                        _mm256_permute_pd::<0b0101>(x),
                        _mm256_permute_pd::<0b0101>(y),
                    );
                    let a = _mm256_fmadd_pd(t01, ys, _mm256_mul_pd(r00, x));
                    let b = _mm256_fmadd_pd(t10, xs, _mm256_mul_pd(r11, y));
                    _mm256_storeu_pd(lo_ptr.add(2 * k), a);
                    _mm256_storeu_pd(hi_ptr.add(2 * k), b);
                }
                k += 2;
            }
        }
        while k + 2 <= n {
            // SAFETY: k + 2 ≤ n, so four f64s from 2k are in bounds;
            // Complex64 is two f64s with no padding.
            unsafe {
                let x = _mm256_loadu_pd(lo_ptr.add(2 * k));
                let y = _mm256_loadu_pd(hi_ptr.add(2 * k));
                let a = _mm256_add_pd(cmul(r00, i00, x), cmul(r01, i01, y));
                let b = _mm256_add_pd(cmul(r10, i10, x), cmul(r11, i11, y));
                _mm256_storeu_pd(lo_ptr.add(2 * k), a);
                _mm256_storeu_pd(hi_ptr.add(2 * k), b);
            }
            k += 2;
        }
        for k in k..n {
            let (x, y) = (lo[k], hi[k]);
            lo[k] = m[0][0] * x + m[0][1] * y;
            hi[k] = m[1][0] * x + m[1][1] * y;
        }
    }

    /// [`Scalar::apply_quad`](super::Scalar::apply_quad) for complex
    /// doubles.
    ///
    /// # Safety
    /// The CPU must support AVX2 and FMA.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn apply_quad_avx2(m: &[[C64; 4]; 4], s: [&mut [C64]; 4]) {
        let n = s.iter().map(|x| x.len()).min().unwrap_or(0);
        let parts: [[(__m256d, __m256d); 4]; 4] = std::array::from_fn(|r| {
            std::array::from_fn(|c| (_mm256_set1_pd(m[r][c].re), _mm256_set1_pd(m[r][c].im)))
        });
        let ptrs = s.map(|x| x.as_mut_ptr().cast::<f64>());
        let mut k = 0;
        while k + 2 <= n {
            // SAFETY: k + 2 ≤ n for every slice.
            unsafe {
                let v = ptrs.map(|p| _mm256_loadu_pd(p.add(2 * k)));
                for (r, row) in parts.iter().enumerate() {
                    let mut acc = cmul(row[0].0, row[0].1, v[0]);
                    for c in 1..4 {
                        acc = _mm256_add_pd(acc, cmul(row[c].0, row[c].1, v[c]));
                    }
                    _mm256_storeu_pd(ptrs[r].add(2 * k), acc);
                }
            }
            k += 2;
        }
        // SAFETY: the remaining index is below every slice's length.
        unsafe {
            for k in k..n {
                let v = ptrs.map(|p| C64::new(*p.add(2 * k), *p.add(2 * k + 1)));
                for (r, p) in ptrs.iter().enumerate() {
                    let out = (0..4).map(|c| m[r][c] * v[c]).sum::<C64>();
                    *p.add(2 * k) = out.re;
                    *p.add(2 * k + 1) = out.im;
                }
            }
        }
    }

    /// [`Scalar::scale_slice`](super::Scalar::scale_slice) for complex
    /// doubles.
    ///
    /// # Safety
    /// The CPU must support AVX2 and FMA.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn scale_avx2(s: &mut [C64], f: C64) {
        let (re, im) = (_mm256_set1_pd(f.re), _mm256_set1_pd(f.im));
        let n = s.len();
        let ptr = s.as_mut_ptr().cast::<f64>();
        let mut k = 0;
        while k + 2 <= n {
            // SAFETY: k + 2 ≤ n.
            unsafe {
                let x = _mm256_loadu_pd(ptr.add(2 * k));
                _mm256_storeu_pd(ptr.add(2 * k), cmul(re, im, x));
            }
            k += 2;
        }
        for a in &mut s[k..] {
            *a *= f;
        }
    }

    /// [`Scalar::mul_slices`](super::Scalar::mul_slices) for complex
    /// doubles.
    ///
    /// # Safety
    /// The CPU must support AVX2 and FMA.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn mul_slices_avx2(s: &mut [C64], factors: &[C64]) {
        let n = s.len().min(factors.len());
        let (ptr, f_ptr) = (s.as_mut_ptr().cast::<f64>(), factors.as_ptr().cast::<f64>());
        let mut k = 0;
        while k + 2 <= n {
            // SAFETY: k + 2 ≤ n for both slices.
            unsafe {
                let f = _mm256_loadu_pd(f_ptr.add(2 * k));
                let (re, im) = (_mm256_movedup_pd(f), _mm256_permute_pd::<0b1111>(f));
                let x = _mm256_loadu_pd(ptr.add(2 * k));
                _mm256_storeu_pd(ptr.add(2 * k), cmul(re, im, x));
            }
            k += 2;
        }
        for (a, &f) in s[k..n].iter_mut().zip(&factors[k..n]) {
            *a *= f;
        }
    }

    /// The longest pattern [`scale_pattern_avx2`] takes.
    pub(super) const MAX_PATTERN: usize = 16;

    /// [`Scalar::scale_pattern`](super::Scalar::scale_pattern) for complex
    /// doubles, with an even pattern of at most `MAX_PATTERN` factors held
    /// in registers.
    ///
    /// # Safety
    /// The CPU must support AVX2 and FMA.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn scale_pattern_avx2(s: &mut [C64], pattern: &[C64]) {
        let len = pattern.len();
        assert!(len.is_multiple_of(2) && len <= MAX_PATTERN && s.len().is_multiple_of(len));
        let factors: [_; MAX_PATTERN / 2] = std::array::from_fn(|j| {
            let (a, b) = (pattern.get(2 * j), pattern.get(2 * j + 1));
            column(
                a.copied().unwrap_or_default(),
                b.copied().unwrap_or_default(),
            )
        });
        let ptr = s.as_mut_ptr().cast::<f64>();
        for chunk in 0..s.len() / len {
            for (j, &(re, im)) in factors[..len / 2].iter().enumerate() {
                // SAFETY: amplitudes chunk·len + 2j and the next are in
                // bounds, as len divides s.len().
                unsafe {
                    let p = ptr.add(2 * (chunk * len + 2 * j));
                    _mm256_storeu_pd(p, cmul(re, im, _mm256_loadu_pd(p)));
                }
            }
        }
    }

    /// Broadcast the low (`HIGH = false`) or high complex number of `v`
    /// to both halves.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    fn broadcast<const HIGH: bool>(v: __m256d) -> __m256d {
        if HIGH {
            _mm256_permute2f128_pd::<0x11>(v, v)
        } else {
            _mm256_permute2f128_pd::<0x00>(v, v)
        }
    }

    /// `(a, b)` as a register's `(re, re, re, re)` and `(im, im, im, im)`
    /// parts, `a` in the low half: one column of a matrix applied to
    /// adjacent amplitudes.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    fn column(a: C64, b: C64) -> (__m256d, __m256d) {
        (
            _mm256_setr_pd(a.re, a.re, b.re, b.re),
            _mm256_setr_pd(a.im, a.im, b.im, b.im),
        )
    }

    /// [`Scalar::apply_pair_strided`](super::Scalar::apply_pair_strided)
    /// for complex doubles. At stride 1 each pair fills one register.
    ///
    /// # Safety
    /// The CPU must support AVX2 and FMA.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn apply_pair_strided_avx2(m: &[[C64; 2]; 2], s: &mut [C64], stride: usize) {
        if stride > 1 {
            for chunk in s.chunks_exact_mut(2 * stride) {
                let (lo, hi) = chunk.split_at_mut(stride);
                // SAFETY: the CPU supports AVX2 and FMA.
                unsafe { apply_pair_avx2(m, lo, hi) };
            }
            return;
        }
        let (c0, c1) = (column(m[0][0], m[1][0]), column(m[0][1], m[1][1]));
        let ptr = s.as_mut_ptr().cast::<f64>();
        for k in 0..s.len() / 2 {
            // SAFETY: amplitudes 2k and 2k + 1 are in bounds.
            unsafe {
                let v = _mm256_loadu_pd(ptr.add(4 * k));
                let (x, y) = (broadcast::<false>(v), broadcast::<true>(v));
                let out = _mm256_add_pd(cmul(c0.0, c0.1, x), cmul(c1.0, c1.1, y));
                _mm256_storeu_pd(ptr.add(4 * k), out);
            }
        }
    }

    /// [`Scalar::apply_quad_strided`](super::Scalar::apply_quad_strided)
    /// for complex doubles. At `half = 1` each half of a quad fills one
    /// register.
    ///
    /// # Safety
    /// The CPU must support AVX2 and FMA.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn apply_quad_strided_avx2(
        m: &[[C64; 4]; 4],
        lo: &mut [C64],
        hi: &mut [C64],
        half: usize,
    ) {
        if half > 1 {
            let pairs = lo
                .chunks_exact_mut(2 * half)
                .zip(hi.chunks_exact_mut(2 * half));
            for (l, h) in pairs {
                let (s00, s01) = l.split_at_mut(half);
                let (s10, s11) = h.split_at_mut(half);
                // SAFETY: the CPU supports AVX2 and FMA.
                unsafe { apply_quad_avx2(m, [s00, s01, s10, s11]) };
            }
            return;
        }
        // Columns of the top (rows 0, 1) and bottom (rows 2, 3) halves.
        let top: [_; 4] = std::array::from_fn(|c| column(m[0][c], m[1][c]));
        let bottom: [_; 4] = std::array::from_fn(|c| column(m[2][c], m[3][c]));
        let n = lo.len().min(hi.len()) / 2;
        let (lo_ptr, hi_ptr) = (lo.as_mut_ptr().cast::<f64>(), hi.as_mut_ptr().cast::<f64>());
        for k in 0..n {
            // SAFETY: amplitudes 2k and 2k + 1 of both slices are in bounds.
            unsafe {
                let (l, h) = (
                    _mm256_loadu_pd(lo_ptr.add(4 * k)),
                    _mm256_loadu_pd(hi_ptr.add(4 * k)),
                );
                let v = [
                    broadcast::<false>(l),
                    broadcast::<true>(l),
                    broadcast::<false>(h),
                    broadcast::<true>(h),
                ];
                let half_of = |cols: &[(__m256d, __m256d); 4]| {
                    let mut acc = cmul(cols[0].0, cols[0].1, v[0]);
                    for c in 1..4 {
                        acc = _mm256_add_pd(acc, cmul(cols[c].0, cols[c].1, v[c]));
                    }
                    acc
                };
                _mm256_storeu_pd(lo_ptr.add(4 * k), half_of(&top));
                _mm256_storeu_pd(hi_ptr.add(4 * k), half_of(&bottom));
            }
        }
    }
}

/// Convert a complex matrix entrywise, or `None` if any entry isn't
/// representable.
pub(crate) fn convert_matrix<T: Scalar, const N: usize>(m: &[[C64; N]; N]) -> Option<[[T; N]; N]> {
    let mut out = [[T::ZERO; N]; N];
    for (row_out, row) in out.iter_mut().zip(m) {
        for (o, &v) in row_out.iter_mut().zip(row) {
            *o = T::from_complex(v)?;
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_conversion_rejects_imaginary_parts() {
        assert_eq!(f64::from_complex(C64::new(2.0, 0.0)), Some(2.0));
        assert_eq!(
            f64::from_complex(C64::cis(std::f64::consts::PI)),
            Some(-1.0)
        );
        assert_eq!(f64::from_complex(C64::new(0.0, 1.0)), None);
    }
}
