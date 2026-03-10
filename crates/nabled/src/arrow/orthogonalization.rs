//! Arrow adapters for orthogonalization workflows.

use arrow_array::FixedSizeListArray;
use arrow_array::types::{Float32Type, Float64Type};

use super::{ArrowInteropError, fixed_size_list_from_owned, fixed_size_list_view};

/// Compute modified Gram-Schmidt orthogonalization for `f32` Arrow dense input.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or orthogonalization fails.
pub fn gram_schmidt_f32(
    matrix: &FixedSizeListArray,
) -> Result<FixedSizeListArray, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float32Type>(matrix)?;
    let output = crate::linalg::orthogonalization::gram_schmidt_view(&matrix_view)?;
    fixed_size_list_from_owned::<Float32Type>(output)
}

/// Compute modified Gram-Schmidt orthogonalization for `f64` Arrow dense input.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or orthogonalization fails.
pub fn gram_schmidt_f64(
    matrix: &FixedSizeListArray,
) -> Result<FixedSizeListArray, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float64Type>(matrix)?;
    let output = crate::linalg::orthogonalization::gram_schmidt_view(&matrix_view)?;
    fixed_size_list_from_owned::<Float64Type>(output)
}

/// Compute classical Gram-Schmidt orthogonalization for `f32` Arrow dense input.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or orthogonalization fails.
pub fn gram_schmidt_classic_f32(
    matrix: &FixedSizeListArray,
) -> Result<FixedSizeListArray, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float32Type>(matrix)?;
    let output = crate::linalg::orthogonalization::gram_schmidt_classic_view(&matrix_view)?;
    fixed_size_list_from_owned::<Float32Type>(output)
}

/// Compute classical Gram-Schmidt orthogonalization for `f64` Arrow dense input.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or orthogonalization fails.
pub fn gram_schmidt_classic_f64(
    matrix: &FixedSizeListArray,
) -> Result<FixedSizeListArray, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float64Type>(matrix)?;
    let output = crate::linalg::orthogonalization::gram_schmidt_classic_view(&matrix_view)?;
    fixed_size_list_from_owned::<Float64Type>(output)
}
