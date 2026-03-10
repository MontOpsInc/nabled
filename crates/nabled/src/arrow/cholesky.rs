//! Arrow adapters for Cholesky decomposition workflows.

use arrow_array::types::{Float32Type, Float64Type};
use arrow_array::{FixedSizeListArray, PrimitiveArray};

use super::{
    ArrowInteropError, fixed_size_list_from_owned, fixed_size_list_view,
    primitive_array_from_owned, primitive_array_view,
};

/// Compute `f32` Cholesky decomposition directly from an Arrow dense matrix.
///
/// The inbound Arrow -> ndarray bridge is zero-copy. The decomposition result remains in
/// `nabled`'s ndarray-native result struct.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or is not SPD.
pub fn decompose_f32(
    matrix: &FixedSizeListArray,
) -> Result<crate::linalg::cholesky::NdarrayCholeskyResult<f32>, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float32Type>(matrix)?;
    Ok(crate::linalg::cholesky::decompose_view(&matrix_view)?)
}

/// Compute `f64` Cholesky decomposition directly from an Arrow dense matrix.
///
/// The inbound Arrow -> ndarray bridge is zero-copy. The decomposition result remains in
/// `nabled`'s ndarray-native result struct.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or is not SPD.
pub fn decompose_f64(
    matrix: &FixedSizeListArray,
) -> Result<crate::linalg::cholesky::NdarrayCholeskyResult<f64>, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float64Type>(matrix)?;
    Ok(crate::linalg::cholesky::decompose_view(&matrix_view)?)
}

/// Solve `Ax=b` directly from `f32` Arrow dense inputs using Cholesky decomposition.
///
/// # Errors
/// Returns an error when inputs contain nulls, are empty, dimensions mismatch, or the matrix is
/// not SPD.
pub fn solve_f32(
    matrix: &FixedSizeListArray,
    rhs: &PrimitiveArray<Float32Type>,
) -> Result<PrimitiveArray<Float32Type>, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float32Type>(matrix)?;
    let rhs_view = primitive_array_view(rhs)?;
    let solution = crate::linalg::cholesky::solve_view(&matrix_view, &rhs_view)?;
    Ok(primitive_array_from_owned::<Float32Type>(solution))
}

/// Solve `Ax=b` directly from `f64` Arrow dense inputs using Cholesky decomposition.
///
/// # Errors
/// Returns an error when inputs contain nulls, are empty, dimensions mismatch, or the matrix is
/// not SPD.
pub fn solve_f64(
    matrix: &FixedSizeListArray,
    rhs: &PrimitiveArray<Float64Type>,
) -> Result<PrimitiveArray<Float64Type>, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float64Type>(matrix)?;
    let rhs_view = primitive_array_view(rhs)?;
    let solution = crate::linalg::cholesky::solve_view(&matrix_view, &rhs_view)?;
    Ok(primitive_array_from_owned::<Float64Type>(solution))
}

/// Compute the `f32` inverse directly from an Arrow SPD dense matrix.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or is not SPD.
pub fn inverse_f32(matrix: &FixedSizeListArray) -> Result<FixedSizeListArray, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float32Type>(matrix)?;
    let output = crate::linalg::cholesky::inverse_view(&matrix_view)?;
    fixed_size_list_from_owned::<Float32Type>(output)
}

/// Compute the `f64` inverse directly from an Arrow SPD dense matrix.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or is not SPD.
pub fn inverse_f64(matrix: &FixedSizeListArray) -> Result<FixedSizeListArray, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float64Type>(matrix)?;
    let output = crate::linalg::cholesky::inverse_view(&matrix_view)?;
    fixed_size_list_from_owned::<Float64Type>(output)
}
