//! Arrow adapters for dense iterative-solver workflows.

use arrow_array::types::{Float32Type, Float64Type};
use arrow_array::{FixedSizeListArray, PrimitiveArray};

use super::{
    ArrowInteropError, fixed_size_list_view, primitive_array_from_owned, primitive_array_view,
};

/// Solve an SPD `f32` system with conjugate gradient directly from Arrow dense inputs.
///
/// # Errors
/// Returns an error when inputs contain nulls, are empty, or the solver fails.
pub fn conjugate_gradient_f32(
    matrix: &FixedSizeListArray,
    rhs: &PrimitiveArray<Float32Type>,
    config: &crate::ml::iterative::IterativeConfig<f32>,
) -> Result<PrimitiveArray<Float32Type>, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float32Type>(matrix)?;
    let rhs_view = primitive_array_view(rhs)?;
    let output = crate::ml::iterative::conjugate_gradient_view(&matrix_view, &rhs_view, config)?;
    Ok(primitive_array_from_owned::<Float32Type>(output))
}

/// Solve an SPD `f64` system with conjugate gradient directly from Arrow dense inputs.
///
/// # Errors
/// Returns an error when inputs contain nulls, are empty, or the solver fails.
pub fn conjugate_gradient_f64(
    matrix: &FixedSizeListArray,
    rhs: &PrimitiveArray<Float64Type>,
    config: &crate::ml::iterative::IterativeConfig<f64>,
) -> Result<PrimitiveArray<Float64Type>, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float64Type>(matrix)?;
    let rhs_view = primitive_array_view(rhs)?;
    let output = crate::ml::iterative::conjugate_gradient_view(&matrix_view, &rhs_view, config)?;
    Ok(primitive_array_from_owned::<Float64Type>(output))
}

/// Solve a general `f32` system with GMRES directly from Arrow dense inputs.
///
/// # Errors
/// Returns an error when inputs contain nulls, are empty, or the solver fails.
pub fn gmres_f32(
    matrix: &FixedSizeListArray,
    rhs: &PrimitiveArray<Float32Type>,
    config: &crate::ml::iterative::IterativeConfig<f32>,
) -> Result<PrimitiveArray<Float32Type>, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float32Type>(matrix)?;
    let rhs_view = primitive_array_view(rhs)?;
    let output = crate::ml::iterative::gmres_view(&matrix_view, &rhs_view, config)?;
    Ok(primitive_array_from_owned::<Float32Type>(output))
}

/// Solve a general `f64` system with GMRES directly from Arrow dense inputs.
///
/// # Errors
/// Returns an error when inputs contain nulls, are empty, or the solver fails.
pub fn gmres_f64(
    matrix: &FixedSizeListArray,
    rhs: &PrimitiveArray<Float64Type>,
    config: &crate::ml::iterative::IterativeConfig<f64>,
) -> Result<PrimitiveArray<Float64Type>, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float64Type>(matrix)?;
    let rhs_view = primitive_array_view(rhs)?;
    let output = crate::ml::iterative::gmres_view(&matrix_view, &rhs_view, config)?;
    Ok(primitive_array_from_owned::<Float64Type>(output))
}
