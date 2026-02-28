//! Ndarray-native ML domain crate.

use nabled_core::errors::{IntoNabledError, NabledError, ShapeError};
pub use nabled_core::prelude;
pub use nabled_linalg as linalg;

pub mod iterative;
pub mod jacobian;
pub mod optimization;
pub mod pca;
pub mod regression;
pub mod stats;

pub use iterative::{IterativeConfig, IterativeError};
pub use jacobian::{JacobianConfig, JacobianError};
pub use optimization::{AdamConfig, LineSearchConfig, OptimizationError, SGDConfig};
pub use pca::{NdarrayPCAResult, PCAError};
pub use regression::{NdarrayRegressionResult, RegressionError};
pub use stats::StatsError;

impl IntoNabledError for IterativeError {
    fn into_nabled_error(self) -> NabledError {
        match self {
            IterativeError::EmptyMatrix => NabledError::Shape(ShapeError::EmptyInput),
            IterativeError::DimensionMismatch => NabledError::Shape(ShapeError::DimensionMismatch),
            IterativeError::MaxIterationsExceeded | IterativeError::Breakdown => {
                NabledError::ConvergenceFailed
            }
            IterativeError::NotPositiveDefinite => NabledError::NotPositiveDefinite,
        }
    }
}

impl IntoNabledError for JacobianError {
    fn into_nabled_error(self) -> NabledError {
        match self {
            JacobianError::FunctionError(message) => {
                NabledError::Other(format!("function error: {message}"))
            }
            JacobianError::InvalidDimensions(message) => NabledError::InvalidInput(message),
            JacobianError::InvalidStepSize => {
                NabledError::InvalidInput("invalid step size".to_string())
            }
            JacobianError::ConvergenceFailed => NabledError::ConvergenceFailed,
            JacobianError::EmptyInput => NabledError::Shape(ShapeError::EmptyInput),
            JacobianError::DimensionMismatch => NabledError::Shape(ShapeError::DimensionMismatch),
        }
    }
}

impl IntoNabledError for OptimizationError {
    fn into_nabled_error(self) -> NabledError {
        match self {
            OptimizationError::EmptyInput => NabledError::Shape(ShapeError::EmptyInput),
            OptimizationError::DimensionMismatch => {
                NabledError::Shape(ShapeError::DimensionMismatch)
            }
            OptimizationError::NonFiniteInput => NabledError::NumericalInstability,
            OptimizationError::InvalidConfig => {
                NabledError::InvalidInput("invalid optimizer configuration".to_string())
            }
            OptimizationError::MaxIterationsExceeded => NabledError::ConvergenceFailed,
        }
    }
}

impl IntoNabledError for PCAError {
    fn into_nabled_error(self) -> NabledError {
        match self {
            PCAError::EmptyMatrix => NabledError::Shape(ShapeError::EmptyInput),
            PCAError::InvalidInput(message) => NabledError::InvalidInput(message),
            PCAError::DecompositionFailed => NabledError::ConvergenceFailed,
        }
    }
}

impl IntoNabledError for RegressionError {
    fn into_nabled_error(self) -> NabledError {
        match self {
            RegressionError::EmptyInput => NabledError::Shape(ShapeError::EmptyInput),
            RegressionError::DimensionMismatch => NabledError::Shape(ShapeError::DimensionMismatch),
            RegressionError::Singular => NabledError::SingularMatrix,
            RegressionError::InvalidInput(message) => NabledError::InvalidInput(message),
        }
    }
}

impl IntoNabledError for StatsError {
    fn into_nabled_error(self) -> NabledError {
        match self {
            StatsError::EmptyMatrix => NabledError::Shape(ShapeError::EmptyInput),
            StatsError::InsufficientSamples => {
                NabledError::InvalidInput("at least two observations are required".to_string())
            }
            StatsError::NumericalInstability => NabledError::NumericalInstability,
        }
    }
}

#[cfg(test)]
mod tests {
    use nabled_core::errors::{IntoNabledError, NabledError, ShapeError};

    use super::*;

    #[test]
    fn ml_errors_map_to_shared_taxonomy() {
        assert!(matches!(
            IterativeError::EmptyMatrix.into_nabled_error(),
            NabledError::Shape(ShapeError::EmptyInput)
        ));
        assert!(matches!(
            IterativeError::DimensionMismatch.into_nabled_error(),
            NabledError::Shape(ShapeError::DimensionMismatch)
        ));
        assert!(matches!(
            IterativeError::MaxIterationsExceeded.into_nabled_error(),
            NabledError::ConvergenceFailed
        ));
        assert!(matches!(
            IterativeError::Breakdown.into_nabled_error(),
            NabledError::ConvergenceFailed
        ));
        assert!(matches!(
            IterativeError::NotPositiveDefinite.into_nabled_error(),
            NabledError::NotPositiveDefinite
        ));

        assert!(matches!(
            JacobianError::FunctionError("x".to_string()).into_nabled_error(),
            NabledError::Other(_)
        ));
        assert!(matches!(
            JacobianError::InvalidDimensions("x".to_string()).into_nabled_error(),
            NabledError::InvalidInput(_)
        ));
        assert!(matches!(
            JacobianError::InvalidStepSize.into_nabled_error(),
            NabledError::InvalidInput(_)
        ));
        assert!(matches!(
            JacobianError::ConvergenceFailed.into_nabled_error(),
            NabledError::ConvergenceFailed
        ));
        assert!(matches!(
            JacobianError::EmptyInput.into_nabled_error(),
            NabledError::Shape(ShapeError::EmptyInput)
        ));
        assert!(matches!(
            JacobianError::DimensionMismatch.into_nabled_error(),
            NabledError::Shape(ShapeError::DimensionMismatch)
        ));

        assert!(matches!(
            OptimizationError::NonFiniteInput.into_nabled_error(),
            NabledError::NumericalInstability
        ));
        assert!(matches!(
            OptimizationError::InvalidConfig.into_nabled_error(),
            NabledError::InvalidInput(_)
        ));
        assert!(matches!(
            OptimizationError::MaxIterationsExceeded.into_nabled_error(),
            NabledError::ConvergenceFailed
        ));

        assert!(matches!(
            PCAError::DecompositionFailed.into_nabled_error(),
            NabledError::ConvergenceFailed
        ));
        assert!(matches!(
            PCAError::InvalidInput("x".to_string()).into_nabled_error(),
            NabledError::InvalidInput(_)
        ));

        assert!(matches!(
            RegressionError::Singular.into_nabled_error(),
            NabledError::SingularMatrix
        ));

        assert!(matches!(
            StatsError::EmptyMatrix.into_nabled_error(),
            NabledError::Shape(ShapeError::EmptyInput)
        ));
        assert!(matches!(
            StatsError::InsufficientSamples.into_nabled_error(),
            NabledError::InvalidInput(_)
        ));
        assert!(matches!(
            StatsError::NumericalInstability.into_nabled_error(),
            NabledError::NumericalInstability
        ));
    }
}
