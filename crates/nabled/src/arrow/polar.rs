//! Arrow adapters for polar decomposition workflows.

use arrow_array::FixedSizeListArray;
use arrow_array::types::{Float32Type, Float64Type};

use super::{ArrowInteropError, complex64_matrix_view, fixed_size_list_view};

/// Compute `f32` polar decomposition directly from an Arrow dense matrix.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or decomposition fails.
pub fn compute_f32(
    matrix: &FixedSizeListArray,
) -> Result<crate::linalg::polar::NdarrayPolarResult<f32>, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float32Type>(matrix)?;
    Ok(crate::linalg::polar::compute_polar_view(&matrix_view)?)
}

/// Compute `f64` polar decomposition directly from an Arrow dense matrix.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or decomposition fails.
pub fn compute_f64(
    matrix: &FixedSizeListArray,
) -> Result<crate::linalg::polar::NdarrayPolarResult<f64>, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float64Type>(matrix)?;
    Ok(crate::linalg::polar::compute_polar_view(&matrix_view)?)
}

/// Compute complex polar decomposition directly from Arrow complex dense input.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or decomposition fails.
pub fn compute_complex(
    matrix: &FixedSizeListArray,
) -> Result<crate::linalg::polar::NdarrayComplexPolarResult, ArrowInteropError> {
    let matrix_view = complex64_matrix_view(matrix)?;
    Ok(crate::linalg::polar::compute_polar_complex_view(&matrix_view)?)
}
