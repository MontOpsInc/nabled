//! Ndarray-native ML-oriented numerical domains for the `nabled` workspace.
//!
//! `nabled-ml` composes `nabled-linalg` kernels into higher-level ML/statistical
//! routines over ndarray vectors and matrices.
//!
//! # Included Domains
//!
//! 1. [`iterative`]: iterative linear-system solvers.
//! 2. [`optimization`]: first/second-order optimization routines.
//! 3. [`jacobian`]: numerical Jacobian/gradient/Hessian estimators.
//! 4. [`pca`]: principal component analysis and transforms.
//! 5. [`regression`]: linear regression routines.
//! 6. [`stats`]: covariance/correlation/centering utilities.
//!
//! # Feature Flags
//!
//! 1. `blas`: enables BLAS acceleration via `nabled-linalg/blas`.
//! 2. `openblas-system`: enables provider-backed `LAPACK` paths via system `OpenBLAS`.
//! 3. `openblas-static`: enables provider-backed `LAPACK` paths via statically linked `OpenBLAS`.
//! 4. `netlib-system`: enables provider-backed `LAPACK` paths via system `Netlib` `LAPACK`.
//! 5. `netlib-static`: enables provider-backed `LAPACK` paths via statically linked `Netlib`
//!    `LAPACK`.
//!
//! # Example
//!
//! ```rust
//! use ndarray::{arr1, arr2};
//! use nabled_ml::regression;
//!
//! let x = arr2(&[[1.0_f64, 1.0], [1.0, 2.0], [1.0, 3.0]]);
//! let y = arr1(&[1.0_f64, 2.0, 3.0]);
//! let model = regression::linear_regression(&x, &y)?;
//! assert_eq!(model.coefficients.len(), 2);
//! # Ok::<(), nabled_ml::regression::RegressionError>(())
//! ```

use nabled_core::errors::{IntoNabledError, NabledError, ShapeError};

use crate::iterative::IterativeError;
use crate::jacobian::JacobianError;
use crate::optimization::OptimizationError;
use crate::pca::PCAError;
use crate::regression::RegressionError;
use crate::stats::StatsError;

pub mod iterative;
pub mod jacobian;
pub mod optimization;
pub mod pca;
pub mod regression;
pub mod stats;

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
