use nabled_core::errors::{IntoNabledError, NabledError, ShapeError};
use nabled_ml::{
    IterativeError, JacobianError, OptimizationError, PCAError, RegressionError, StatsError,
};

#[test]
fn ml_error_mappings_are_stable_and_exhaustive() {
    assert_eq!(
        IterativeError::EmptyMatrix.into_nabled_error(),
        NabledError::Shape(ShapeError::EmptyInput)
    );
    assert_eq!(
        IterativeError::DimensionMismatch.into_nabled_error(),
        NabledError::Shape(ShapeError::DimensionMismatch)
    );
    assert_eq!(
        IterativeError::MaxIterationsExceeded.into_nabled_error(),
        NabledError::ConvergenceFailed
    );
    assert_eq!(IterativeError::Breakdown.into_nabled_error(), NabledError::ConvergenceFailed);
    assert_eq!(
        IterativeError::NotPositiveDefinite.into_nabled_error(),
        NabledError::NotPositiveDefinite
    );

    assert_eq!(
        JacobianError::FunctionError("fn".to_string()).into_nabled_error(),
        NabledError::Other("function error: fn".to_string())
    );
    assert_eq!(
        JacobianError::InvalidDimensions("dims".to_string()).into_nabled_error(),
        NabledError::InvalidInput("dims".to_string())
    );
    assert_eq!(
        JacobianError::InvalidStepSize.into_nabled_error(),
        NabledError::InvalidInput("invalid step size".to_string())
    );
    assert_eq!(
        JacobianError::ConvergenceFailed.into_nabled_error(),
        NabledError::ConvergenceFailed
    );
    assert_eq!(
        JacobianError::EmptyInput.into_nabled_error(),
        NabledError::Shape(ShapeError::EmptyInput)
    );
    assert_eq!(
        JacobianError::DimensionMismatch.into_nabled_error(),
        NabledError::Shape(ShapeError::DimensionMismatch)
    );

    assert_eq!(
        OptimizationError::EmptyInput.into_nabled_error(),
        NabledError::Shape(ShapeError::EmptyInput)
    );
    assert_eq!(
        OptimizationError::DimensionMismatch.into_nabled_error(),
        NabledError::Shape(ShapeError::DimensionMismatch)
    );
    assert_eq!(
        OptimizationError::NonFiniteInput.into_nabled_error(),
        NabledError::NumericalInstability
    );
    assert_eq!(
        OptimizationError::InvalidConfig.into_nabled_error(),
        NabledError::InvalidInput("invalid optimizer configuration".to_string())
    );
    assert_eq!(
        OptimizationError::MaxIterationsExceeded.into_nabled_error(),
        NabledError::ConvergenceFailed
    );

    assert_eq!(
        PCAError::EmptyMatrix.into_nabled_error(),
        NabledError::Shape(ShapeError::EmptyInput)
    );
    assert_eq!(
        PCAError::InvalidInput("pca".to_string()).into_nabled_error(),
        NabledError::InvalidInput("pca".to_string())
    );
    assert_eq!(PCAError::DecompositionFailed.into_nabled_error(), NabledError::ConvergenceFailed);

    assert_eq!(
        RegressionError::EmptyInput.into_nabled_error(),
        NabledError::Shape(ShapeError::EmptyInput)
    );
    assert_eq!(
        RegressionError::DimensionMismatch.into_nabled_error(),
        NabledError::Shape(ShapeError::DimensionMismatch)
    );
    assert_eq!(RegressionError::Singular.into_nabled_error(), NabledError::SingularMatrix);
    assert_eq!(
        RegressionError::InvalidInput("reg".to_string()).into_nabled_error(),
        NabledError::InvalidInput("reg".to_string())
    );

    assert_eq!(
        StatsError::EmptyMatrix.into_nabled_error(),
        NabledError::Shape(ShapeError::EmptyInput)
    );
    assert_eq!(
        StatsError::InsufficientSamples.into_nabled_error(),
        NabledError::InvalidInput("at least two observations are required".to_string())
    );
    assert_eq!(
        StatsError::NumericalInstability.into_nabled_error(),
        NabledError::NumericalInstability
    );
}
