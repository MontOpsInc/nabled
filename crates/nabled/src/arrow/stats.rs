//! Arrow adapters for matrix-statistics workflows.

use arrow_array::types::{Float32Type, Float64Type};
use arrow_array::{FixedSizeListArray, PrimitiveArray};

use super::{
    ArrowInteropError, fixed_size_list_from_owned, fixed_size_list_view, primitive_array_from_owned,
};

/// Compute `f32` column means directly from Arrow dense input.
///
/// # Errors
/// Returns an error when the matrix contains nulls or shape conversion fails.
pub fn column_means_f32(
    matrix: &FixedSizeListArray,
) -> Result<PrimitiveArray<Float32Type>, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float32Type>(matrix)?;
    Ok(primitive_array_from_owned::<Float32Type>(crate::ml::stats::column_means_view(
        &matrix_view,
    )))
}

/// Compute `f64` column means directly from Arrow dense input.
///
/// # Errors
/// Returns an error when the matrix contains nulls or shape conversion fails.
pub fn column_means_f64(
    matrix: &FixedSizeListArray,
) -> Result<PrimitiveArray<Float64Type>, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float64Type>(matrix)?;
    Ok(primitive_array_from_owned::<Float64Type>(crate::ml::stats::column_means_view(
        &matrix_view,
    )))
}

/// Center `f32` columns directly from Arrow dense input.
///
/// # Errors
/// Returns an error when the matrix contains nulls or shape conversion fails.
pub fn center_columns_f32(
    matrix: &FixedSizeListArray,
) -> Result<FixedSizeListArray, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float32Type>(matrix)?;
    let output = crate::ml::stats::center_columns_view(&matrix_view);
    fixed_size_list_from_owned::<Float32Type>(output)
}

/// Center `f64` columns directly from Arrow dense input.
///
/// # Errors
/// Returns an error when the matrix contains nulls or shape conversion fails.
pub fn center_columns_f64(
    matrix: &FixedSizeListArray,
) -> Result<FixedSizeListArray, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float64Type>(matrix)?;
    let output = crate::ml::stats::center_columns_view(&matrix_view);
    fixed_size_list_from_owned::<Float64Type>(output)
}

/// Compute the `f32` covariance matrix directly from Arrow dense input.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or covariance fails.
pub fn covariance_matrix_f32(
    matrix: &FixedSizeListArray,
) -> Result<FixedSizeListArray, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float32Type>(matrix)?;
    let output = crate::ml::stats::covariance_matrix_view(&matrix_view)?;
    fixed_size_list_from_owned::<Float32Type>(output)
}

/// Compute the `f64` covariance matrix directly from Arrow dense input.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or covariance fails.
pub fn covariance_matrix_f64(
    matrix: &FixedSizeListArray,
) -> Result<FixedSizeListArray, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float64Type>(matrix)?;
    let output = crate::ml::stats::covariance_matrix_view(&matrix_view)?;
    fixed_size_list_from_owned::<Float64Type>(output)
}

/// Compute the `f32` correlation matrix directly from Arrow dense input.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or correlation fails.
pub fn correlation_matrix_f32(
    matrix: &FixedSizeListArray,
) -> Result<FixedSizeListArray, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float32Type>(matrix)?;
    let output = crate::ml::stats::correlation_matrix_view(&matrix_view)?;
    fixed_size_list_from_owned::<Float32Type>(output)
}

/// Compute the `f64` correlation matrix directly from Arrow dense input.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or correlation fails.
pub fn correlation_matrix_f64(
    matrix: &FixedSizeListArray,
) -> Result<FixedSizeListArray, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float64Type>(matrix)?;
    let output = crate::ml::stats::correlation_matrix_view(&matrix_view)?;
    fixed_size_list_from_owned::<Float64Type>(output)
}
