//! Amplitude types. The statevector engine is generic over [`Scalar`], so the
//! same kernels run on complex amplitudes or, for real circuits, on `f64` at
//! half the memory.

use num_complex::Complex64 as C64;
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
}

impl Scalar for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;

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
