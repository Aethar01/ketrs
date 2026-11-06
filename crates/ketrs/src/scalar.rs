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
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn apply_pair_avx2(m: &[[C64; 2]; 2], lo: &mut [C64], hi: &mut [C64]) {
        let n = lo.len().min(hi.len());
        let split = |c: C64| (_mm256_set1_pd(c.re), _mm256_set1_pd(c.im));
        let ((r00, i00), (r01, i01)) = (split(m[0][0]), split(m[0][1]));
        let ((r10, i10), (r11, i11)) = (split(m[1][0]), split(m[1][1]));
        let (lo_ptr, hi_ptr) = (lo.as_mut_ptr().cast::<f64>(), hi.as_mut_ptr().cast::<f64>());
        let mut k = 0;
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
