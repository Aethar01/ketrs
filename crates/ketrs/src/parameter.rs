//! Circuit parameters and the angles built from them.
//!
//! A [`Parameter`] is a handle created by
//! [`Circuit::parameter`](crate::Circuit::parameter). Rotation builders take
//! any `impl Into<Angle>`: a plain `f64`, a parameter, or a linear form such
//! as `2.0 * theta + 0.5`, which is what Trotter steps and most ansätze need
//! and keeps gradients exact.

use std::ops::{Add, Mul, Neg, Sub};

/// A circuit parameter: an index into the values passed to
/// [`Circuit::bind`](crate::Circuit::bind).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Parameter(pub(crate) usize);

impl Parameter {
    /// The parameter's position in the value vector.
    pub fn index(self) -> usize {
        self.0
    }
}

/// A rotation angle: a fixed value, or `scale·θ + offset` for a parameter θ.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Angle {
    /// A fixed angle.
    Value(f64),
    /// `scale·θ + offset`.
    Linear {
        /// The parameter θ.
        parameter: Parameter,
        /// Multiplier on θ.
        scale: f64,
        /// Constant added.
        offset: f64,
    },
}

impl Angle {
    /// The angle for the given parameter values.
    pub fn evaluate(self, values: &[f64]) -> f64 {
        match self {
            Angle::Value(v) => v,
            Angle::Linear {
                parameter,
                scale,
                offset,
            } => scale * values[parameter.0] + offset,
        }
    }

    fn map(self, f: impl Fn(f64) -> f64, scale_too: bool) -> Angle {
        match self {
            Angle::Value(v) => Angle::Value(f(v)),
            Angle::Linear {
                parameter,
                scale,
                offset,
            } => Angle::Linear {
                parameter,
                scale: if scale_too { f(scale) } else { scale },
                offset: f(offset),
            },
        }
    }
}

impl From<f64> for Angle {
    fn from(v: f64) -> Self {
        Angle::Value(v)
    }
}

impl From<Parameter> for Angle {
    fn from(parameter: Parameter) -> Self {
        Angle::Linear {
            parameter,
            scale: 1.0,
            offset: 0.0,
        }
    }
}

impl Mul<f64> for Angle {
    type Output = Angle;
    fn mul(self, k: f64) -> Angle {
        self.map(|x| x * k, true)
    }
}

impl Mul<Angle> for f64 {
    type Output = Angle;
    fn mul(self, a: Angle) -> Angle {
        a * self
    }
}

impl Add<f64> for Angle {
    type Output = Angle;
    fn add(self, k: f64) -> Angle {
        match self {
            Angle::Value(v) => Angle::Value(v + k),
            Angle::Linear {
                parameter,
                scale,
                offset,
            } => Angle::Linear {
                parameter,
                scale,
                offset: offset + k,
            },
        }
    }
}

impl Sub<f64> for Angle {
    type Output = Angle;
    fn sub(self, k: f64) -> Angle {
        self + (-k)
    }
}

impl Neg for Angle {
    type Output = Angle;
    fn neg(self) -> Angle {
        self * -1.0
    }
}

impl Mul<f64> for Parameter {
    type Output = Angle;
    fn mul(self, k: f64) -> Angle {
        Angle::from(self) * k
    }
}

impl Mul<Parameter> for f64 {
    type Output = Angle;
    fn mul(self, p: Parameter) -> Angle {
        Angle::from(p) * self
    }
}

impl Add<f64> for Parameter {
    type Output = Angle;
    fn add(self, k: f64) -> Angle {
        Angle::from(self) + k
    }
}

impl Sub<f64> for Parameter {
    type Output = Angle;
    fn sub(self, k: f64) -> Angle {
        Angle::from(self) - k
    }
}

impl Neg for Parameter {
    type Output = Angle;
    fn neg(self) -> Angle {
        -Angle::from(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_forms_evaluate() {
        let theta = Parameter(1);
        let values = [0.0, 3.0];
        assert_eq!((2.0 * theta + 0.5).evaluate(&values), 6.5);
        assert_eq!((-theta).evaluate(&values), -3.0);
        assert_eq!((theta - 1.0).evaluate(&values), 2.0);
        assert_eq!(Angle::from(0.25).evaluate(&values), 0.25);
    }
}
