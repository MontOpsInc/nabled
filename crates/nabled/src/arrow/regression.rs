//! Arrow adapters for regression workflows.

use arrow_array::types::{Float32Type, Float64Type};
use arrow_array::{FixedSizeListArray, PrimitiveArray};

use super::{ArrowInteropError, fixed_size_list_view, primitive_array_view};

/// Solve `f32` linear regression directly from Arrow dense inputs.
///
/// # Errors
/// Returns an error when inputs contain nulls, are empty, or the regression system is singular.
pub fn linear_regression_f32(
    x: &FixedSizeListArray,
    y: &PrimitiveArray<Float32Type>,
    add_intercept: bool,
) -> Result<crate::ml::regression::NdarrayRegressionResult<f32>, ArrowInteropError> {
    let x_view = fixed_size_list_view::<Float32Type>(x)?;
    let y_view = primitive_array_view(y)?;
    Ok(crate::ml::regression::linear_regression_view(&x_view, &y_view, add_intercept)?)
}

/// Solve `f64` linear regression directly from Arrow dense inputs.
///
/// # Errors
/// Returns an error when inputs contain nulls, are empty, or the regression system is singular.
pub fn linear_regression_f64(
    x: &FixedSizeListArray,
    y: &PrimitiveArray<Float64Type>,
    add_intercept: bool,
) -> Result<crate::ml::regression::NdarrayRegressionResult<f64>, ArrowInteropError> {
    let x_view = fixed_size_list_view::<Float64Type>(x)?;
    let y_view = primitive_array_view(y)?;
    Ok(crate::ml::regression::linear_regression_view(&x_view, &y_view, add_intercept)?)
}
