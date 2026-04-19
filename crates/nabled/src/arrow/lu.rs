//! Arrow adapters for LU decomposition workflows.

use arrow_array::types::{Float32Type, Float64Type};
use arrow_array::{FixedSizeListArray, PrimitiveArray};
use arrow_schema::Field;
use num_complex::Complex64;

use super::{
    ArrowInteropError, complex64_matrix_from_owned, complex64_matrix_view,
    complex64_vector_from_owned, complex64_vector_view, fixed_size_list_from_owned,
    fixed_size_list_view, primitive_array_from_owned, primitive_array_view,
};

/// Compute `f32` LU decomposition directly from an Arrow dense matrix.
///
/// The inbound Arrow -> ndarray bridge is zero-copy. The decomposition result remains in
/// `nabled`'s ndarray-native result struct.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or decomposition fails.
pub fn decompose_f32(
    matrix: &FixedSizeListArray,
) -> Result<crate::linalg::lu::NdarrayLUResult<f32>, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float32Type>(matrix)?;
    Ok(crate::linalg::lu::decompose_view(&matrix_view)?)
}

/// Compute `f32` LU decomposition directly from an Arrow dense matrix and preserve factor metadata.
///
/// The returned tuple is `(result, pivots, permutation_sign)`.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or decomposition fails.
pub fn decompose_f32_with_metadata(
    matrix: &FixedSizeListArray,
) -> Result<(crate::linalg::lu::NdarrayLUResult<f32>, Vec<usize>, i8), ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float32Type>(matrix)?;
    Ok(crate::linalg::lu::decompose_view_with_metadata(&matrix_view)?)
}

/// Compute `f64` LU decomposition directly from an Arrow dense matrix.
///
/// The inbound Arrow -> ndarray bridge is zero-copy. The decomposition result remains in
/// `nabled`'s ndarray-native result struct.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or decomposition fails.
pub fn decompose_f64(
    matrix: &FixedSizeListArray,
) -> Result<crate::linalg::lu::NdarrayLUResult<f64>, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float64Type>(matrix)?;
    Ok(crate::linalg::lu::decompose_view(&matrix_view)?)
}

/// Compute `f64` LU decomposition directly from an Arrow dense matrix and preserve factor metadata.
///
/// The returned tuple is `(result, pivots, permutation_sign)`.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or decomposition fails.
pub fn decompose_f64_with_metadata(
    matrix: &FixedSizeListArray,
) -> Result<(crate::linalg::lu::NdarrayLUResult<f64>, Vec<usize>, i8), ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float64Type>(matrix)?;
    Ok(crate::linalg::lu::decompose_view_with_metadata(&matrix_view)?)
}

/// Solve `Ax=b` directly from `f32` Arrow dense inputs.
///
/// The inbound Arrow -> ndarray bridge is zero-copy. The solution is returned as an Arrow
/// primitive array.
///
/// # Errors
/// Returns an error when inputs contain nulls, are empty, dimensions mismatch, or the matrix is
/// singular.
pub fn solve_f32(
    matrix: &FixedSizeListArray,
    rhs: &PrimitiveArray<Float32Type>,
) -> Result<PrimitiveArray<Float32Type>, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float32Type>(matrix)?;
    let rhs_view = primitive_array_view(rhs)?;
    let solution = crate::linalg::lu::solve_view(&matrix_view, &rhs_view)?;
    Ok(primitive_array_from_owned::<Float32Type>(solution))
}

/// Solve `Ax=b` directly from `f64` Arrow dense inputs.
///
/// The inbound Arrow -> ndarray bridge is zero-copy. The solution is returned as an Arrow
/// primitive array.
///
/// # Errors
/// Returns an error when inputs contain nulls, are empty, dimensions mismatch, or the matrix is
/// singular.
pub fn solve_f64(
    matrix: &FixedSizeListArray,
    rhs: &PrimitiveArray<Float64Type>,
) -> Result<PrimitiveArray<Float64Type>, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float64Type>(matrix)?;
    let rhs_view = primitive_array_view(rhs)?;
    let solution = crate::linalg::lu::solve_view(&matrix_view, &rhs_view)?;
    Ok(primitive_array_from_owned::<Float64Type>(solution))
}

/// Compute the `f32` inverse directly from an Arrow dense matrix.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or is singular.
pub fn inverse_f32(matrix: &FixedSizeListArray) -> Result<FixedSizeListArray, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float32Type>(matrix)?;
    let output = crate::linalg::lu::inverse_view(&matrix_view)?;
    fixed_size_list_from_owned::<Float32Type>(output)
}

/// Compute the `f64` inverse directly from an Arrow dense matrix.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or is singular.
pub fn inverse_f64(matrix: &FixedSizeListArray) -> Result<FixedSizeListArray, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float64Type>(matrix)?;
    let output = crate::linalg::lu::inverse_view(&matrix_view)?;
    fixed_size_list_from_owned::<Float64Type>(output)
}

/// Compute the `f32` determinant directly from an Arrow dense matrix.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or is singular.
pub fn determinant_f32(matrix: &FixedSizeListArray) -> Result<f32, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float32Type>(matrix)?;
    Ok(crate::linalg::lu::determinant_view(&matrix_view)?)
}

/// Compute the `f64` determinant directly from an Arrow dense matrix.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or is singular.
pub fn determinant_f64(matrix: &FixedSizeListArray) -> Result<f64, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float64Type>(matrix)?;
    Ok(crate::linalg::lu::determinant_view(&matrix_view)?)
}

/// Compute the `f32` signed log-determinant directly from an Arrow dense matrix.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or is singular.
pub fn log_determinant_f32(
    matrix: &FixedSizeListArray,
) -> Result<crate::linalg::lu::LogDetResult<f32>, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float32Type>(matrix)?;
    Ok(crate::linalg::lu::log_determinant_view(&matrix_view)?)
}

/// Compute the `f64` signed log-determinant directly from an Arrow dense matrix.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or is singular.
pub fn log_determinant_f64(
    matrix: &FixedSizeListArray,
) -> Result<crate::linalg::lu::LogDetResult<f64>, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float64Type>(matrix)?;
    Ok(crate::linalg::lu::log_determinant_view(&matrix_view)?)
}

/// Solve a complex dense linear system directly from Arrow complex inputs.
///
/// # Errors
/// Returns an error when inputs contain nulls, are empty, dimensions mismatch, or the matrix is
/// singular.
pub fn solve_complex(
    matrix: &FixedSizeListArray,
    rhs_field: &Field,
    rhs: &FixedSizeListArray,
) -> Result<(Field, FixedSizeListArray), ArrowInteropError> {
    let matrix_view = complex64_matrix_view(matrix)?;
    let rhs_view = complex64_vector_view(rhs_field, rhs)?;
    let output = crate::linalg::lu::solve_complex_view(&matrix_view, &rhs_view)?;
    complex64_vector_from_owned("lu_solve_complex", output)
}

/// Compute the complex inverse directly from Arrow complex dense input.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or is singular.
pub fn inverse_complex(
    matrix: &FixedSizeListArray,
) -> Result<FixedSizeListArray, ArrowInteropError> {
    let matrix_view = complex64_matrix_view(matrix)?;
    let output = crate::linalg::lu::inverse_complex_view(&matrix_view)?;
    complex64_matrix_from_owned(output)
}

/// Compute the complex determinant directly from Arrow complex dense input.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or is singular.
pub fn determinant_complex(matrix: &FixedSizeListArray) -> Result<Complex64, ArrowInteropError> {
    let matrix_view = complex64_matrix_view(matrix)?;
    Ok(crate::linalg::lu::determinant_complex_view(&matrix_view)?)
}
