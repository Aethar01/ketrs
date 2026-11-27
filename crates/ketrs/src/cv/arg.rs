//! Gate arguments: fixed values, linear forms in circuit parameters, or
//! linear forms in earlier measurement results (feed-forward).

use crate::error::Error;
use crate::parameter::{Angle, Parameter};
use std::ops::{Add, Mul, Neg, Sub};

/// A measurement result slot, returned by the measurement builders of
/// [`ModeCircuit`](super::ModeCircuit). Using it in a later gate's argument
/// makes that gate depend on the outcome.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Measured(pub(crate) usize);

impl Measured {
    /// The slot's position in a shot's results.
    pub fn index(self) -> usize {
        self.0
    }
}

/// A real gate argument.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Arg {
    /// A fixed value.
    Value(f64),
    /// `scale·θ + offset` for a circuit parameter θ.
    Parameter {
        /// The parameter θ.
        parameter: Parameter,
        /// Multiplier on θ.
        scale: f64,
        /// Constant added.
        offset: f64,
    },
    /// `scale·m + offset` for an earlier measurement result `m`.
    Result {
        /// The result.
        result: Measured,
        /// Multiplier on the result.
        scale: f64,
        /// Constant added.
        offset: f64,
    },
}

impl Arg {
    /// The fixed value, if the argument doesn't depend on a parameter or a
    /// result.
    pub fn value(self) -> Option<f64> {
        match self {
            Arg::Value(v) => Some(v),
            _ => None,
        }
    }

    /// The argument with parameters replaced by `values`.
    pub(crate) fn bind(self, values: &[f64]) -> Arg {
        match self {
            Arg::Parameter {
                parameter,
                scale,
                offset,
            } => Arg::Value(scale * values[parameter.0] + offset),
            other => other,
        }
    }

    /// The argument's value given a shot's measurement `results` so far.
    /// Parameters must already be bound.
    pub fn evaluate(self, results: &[f64]) -> Result<f64, Error> {
        match self {
            Arg::Value(v) => Ok(v),
            Arg::Parameter { .. } => Err(Error::InvalidArgument(
                "circuit has unbound parameters; bind it first".into(),
            )),
            Arg::Result {
                result,
                scale,
                offset,
            } => results
                .get(result.0)
                .map(|m| scale * m + offset)
                .ok_or_else(|| {
                    Error::InvalidArgument(format!("result {} not measured yet", result.0))
                }),
        }
    }

    fn affine(self, k: f64, c: f64) -> Arg {
        match self {
            Arg::Value(v) => Arg::Value(k * v + c),
            Arg::Parameter {
                parameter,
                scale,
                offset,
            } => Arg::Parameter {
                parameter,
                scale: k * scale,
                offset: k * offset + c,
            },
            Arg::Result {
                result,
                scale,
                offset,
            } => Arg::Result {
                result,
                scale: k * scale,
                offset: k * offset + c,
            },
        }
    }
}

impl From<f64> for Arg {
    fn from(v: f64) -> Self {
        Arg::Value(v)
    }
}

impl From<Parameter> for Arg {
    fn from(parameter: Parameter) -> Self {
        Arg::Parameter {
            parameter,
            scale: 1.0,
            offset: 0.0,
        }
    }
}

impl From<Angle> for Arg {
    fn from(angle: Angle) -> Self {
        match angle {
            Angle::Value(v) => Arg::Value(v),
            Angle::Linear {
                parameter,
                scale,
                offset,
            } => Arg::Parameter {
                parameter,
                scale,
                offset,
            },
        }
    }
}

impl From<Measured> for Arg {
    fn from(result: Measured) -> Self {
        Arg::Result {
            result,
            scale: 1.0,
            offset: 0.0,
        }
    }
}

impl Mul<f64> for Arg {
    type Output = Arg;
    fn mul(self, k: f64) -> Arg {
        self.affine(k, 0.0)
    }
}

impl Mul<Arg> for f64 {
    type Output = Arg;
    fn mul(self, a: Arg) -> Arg {
        a * self
    }
}

impl Add<f64> for Arg {
    type Output = Arg;
    fn add(self, k: f64) -> Arg {
        self.affine(1.0, k)
    }
}

impl Sub<f64> for Arg {
    type Output = Arg;
    fn sub(self, k: f64) -> Arg {
        self.affine(1.0, -k)
    }
}

impl Neg for Arg {
    type Output = Arg;
    fn neg(self) -> Arg {
        self.affine(-1.0, 0.0)
    }
}

impl Mul<f64> for Measured {
    type Output = Arg;
    fn mul(self, k: f64) -> Arg {
        Arg::from(self) * k
    }
}

impl Mul<Measured> for f64 {
    type Output = Arg;
    fn mul(self, m: Measured) -> Arg {
        Arg::from(m) * self
    }
}

impl Add<f64> for Measured {
    type Output = Arg;
    fn add(self, k: f64) -> Arg {
        Arg::from(self) + k
    }
}

impl Sub<f64> for Measured {
    type Output = Arg;
    fn sub(self, k: f64) -> Arg {
        Arg::from(self) - k
    }
}

impl Neg for Measured {
    type Output = Arg;
    fn neg(self) -> Arg {
        -Arg::from(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_forms_evaluate() {
        let m = Measured(1);
        let results = [0.0, 3.0];
        assert_eq!((2.0 * m + 0.5).evaluate(&results), Ok(6.5));
        assert_eq!((-m).evaluate(&results), Ok(-3.0));
        assert_eq!(Arg::from(0.25).evaluate(&results), Ok(0.25));
        assert!(Arg::from(Measured(2)).evaluate(&results).is_err());

        let theta = Parameter(0);
        let a = Arg::from(2.0 * theta + 1.0) * 3.0;
        assert!(a.evaluate(&[]).is_err());
        assert_eq!(a.bind(&[0.5]), Arg::Value(6.0));
    }
}
