//! Arrow adapters for triangular solve workflows.

use arrow_array::types::{Float32Type, Float64Type};
use arrow_array::{FixedSizeListArray, PrimitiveArray};

use super::{
    ArrowInteropError, fixed_size_list_from_owned, fixed_size_list_view,
    primitive_array_from_owned, primitive_array_view,
};

/// Solve lower-triangular `f32` systems directly from Arrow dense inputs.
///
/// # Errors
/// Returns an error when inputs contain nulls, are empty, or dimensions mismatch.
pub fn solve_lower_f32(
    matrix: &FixedSizeListArray,
    rhs: &PrimitiveArray<Float32Type>,
) -> Result<PrimitiveArray<Float32Type>, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float32Type>(matrix)?;
    let rhs_view = primitive_array_view(rhs)?;
    let output = crate::linalg::triangular::solve_lower_view(&matrix_view, &rhs_view)?;
    Ok(primitive_array_from_owned::<Float32Type>(output))
}

/// Solve lower-triangular `f64` systems directly from Arrow dense inputs.
///
/// # Errors
/// Returns an error when inputs contain nulls, are empty, or dimensions mismatch.
pub fn solve_lower_f64(
    matrix: &FixedSizeListArray,
    rhs: &PrimitiveArray<Float64Type>,
) -> Result<PrimitiveArray<Float64Type>, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float64Type>(matrix)?;
    let rhs_view = primitive_array_view(rhs)?;
    let output = crate::linalg::triangular::solve_lower_view(&matrix_view, &rhs_view)?;
    Ok(primitive_array_from_owned::<Float64Type>(output))
}

/// Solve upper-triangular `f32` systems directly from Arrow dense inputs.
///
/// # Errors
/// Returns an error when inputs contain nulls, are empty, or dimensions mismatch.
pub fn solve_upper_f32(
    matrix: &FixedSizeListArray,
    rhs: &PrimitiveArray<Float32Type>,
) -> Result<PrimitiveArray<Float32Type>, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float32Type>(matrix)?;
    let rhs_view = primitive_array_view(rhs)?;
    let output = crate::linalg::triangular::solve_upper_view(&matrix_view, &rhs_view)?;
    Ok(primitive_array_from_owned::<Float32Type>(output))
}

/// Solve upper-triangular `f64` systems directly from Arrow dense inputs.
///
/// # Errors
/// Returns an error when inputs contain nulls, are empty, or dimensions mismatch.
pub fn solve_upper_f64(
    matrix: &FixedSizeListArray,
    rhs: &PrimitiveArray<Float64Type>,
) -> Result<PrimitiveArray<Float64Type>, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float64Type>(matrix)?;
    let rhs_view = primitive_array_view(rhs)?;
    let output = crate::linalg::triangular::solve_upper_view(&matrix_view, &rhs_view)?;
    Ok(primitive_array_from_owned::<Float64Type>(output))
}

/// Solve lower-triangular `f32` systems with matrix RHS directly from Arrow dense inputs.
///
/// # Errors
/// Returns an error when inputs contain nulls, are empty, or dimensions mismatch.
pub fn solve_lower_matrix_f32(
    matrix: &FixedSizeListArray,
    rhs: &FixedSizeListArray,
) -> Result<FixedSizeListArray, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float32Type>(matrix)?;
    let rhs_view = fixed_size_list_view::<Float32Type>(rhs)?;
    let output = crate::linalg::triangular::solve_lower_matrix_view(&matrix_view, &rhs_view)?;
    fixed_size_list_from_owned::<Float32Type>(output)
}

/// Solve lower-triangular `f64` systems with matrix RHS directly from Arrow dense inputs.
///
/// # Errors
/// Returns an error when inputs contain nulls, are empty, or dimensions mismatch.
pub fn solve_lower_matrix_f64(
    matrix: &FixedSizeListArray,
    rhs: &FixedSizeListArray,
) -> Result<FixedSizeListArray, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float64Type>(matrix)?;
    let rhs_view = fixed_size_list_view::<Float64Type>(rhs)?;
    let output = crate::linalg::triangular::solve_lower_matrix_view(&matrix_view, &rhs_view)?;
    fixed_size_list_from_owned::<Float64Type>(output)
}

/// Solve upper-triangular `f32` systems with matrix RHS directly from Arrow dense inputs.
///
/// # Errors
/// Returns an error when inputs contain nulls, are empty, or dimensions mismatch.
pub fn solve_upper_matrix_f32(
    matrix: &FixedSizeListArray,
    rhs: &FixedSizeListArray,
) -> Result<FixedSizeListArray, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float32Type>(matrix)?;
    let rhs_view = fixed_size_list_view::<Float32Type>(rhs)?;
    let output = crate::linalg::triangular::solve_upper_matrix_view(&matrix_view, &rhs_view)?;
    fixed_size_list_from_owned::<Float32Type>(output)
}

/// Solve upper-triangular `f64` systems with matrix RHS directly from Arrow dense inputs.
///
/// # Errors
/// Returns an error when inputs contain nulls, are empty, or dimensions mismatch.
pub fn solve_upper_matrix_f64(
    matrix: &FixedSizeListArray,
    rhs: &FixedSizeListArray,
) -> Result<FixedSizeListArray, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float64Type>(matrix)?;
    let rhs_view = fixed_size_list_view::<Float64Type>(rhs)?;
    let output = crate::linalg::triangular::solve_upper_matrix_view(&matrix_view, &rhs_view)?;
    fixed_size_list_from_owned::<Float64Type>(output)
}
