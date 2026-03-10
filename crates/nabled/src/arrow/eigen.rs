//! Arrow adapters for eigenvalue decomposition workflows.

use arrow_array::types::{Float32Type, Float64Type};
use arrow_array::{FixedSizeListArray, PrimitiveArray};

use super::{
    ArrowInteropError, fixed_size_list_from_owned, fixed_size_list_view, primitive_array_from_owned,
};

/// Compute symmetric `f32` eigen decomposition directly from an Arrow dense matrix.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or eigendecomposition fails.
pub fn symmetric_f32(
    matrix: &FixedSizeListArray,
) -> Result<crate::linalg::eigen::NdarrayEigenResult<f32>, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float32Type>(matrix)?;
    Ok(crate::linalg::eigen::symmetric_view(&matrix_view)?)
}

/// Compute symmetric `f64` eigen decomposition directly from an Arrow dense matrix.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or eigendecomposition fails.
pub fn symmetric_f64(
    matrix: &FixedSizeListArray,
) -> Result<crate::linalg::eigen::NdarrayEigenResult<f64>, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float64Type>(matrix)?;
    Ok(crate::linalg::eigen::symmetric_view(&matrix_view)?)
}

/// Compute generalized `f32` eigen decomposition directly from Arrow dense matrices.
///
/// # Errors
/// Returns an error when inputs contain nulls, are empty, or eigendecomposition fails.
pub fn generalized_f32(
    matrix_a: &FixedSizeListArray,
    matrix_b: &FixedSizeListArray,
) -> Result<crate::linalg::eigen::NdarrayGeneralizedEigenResult<f32>, ArrowInteropError> {
    let left_view = fixed_size_list_view::<Float32Type>(matrix_a)?;
    let right_view = fixed_size_list_view::<Float32Type>(matrix_b)?;
    Ok(crate::linalg::eigen::generalized_view(&left_view, &right_view)?)
}

/// Compute generalized `f64` eigen decomposition directly from Arrow dense matrices.
///
/// # Errors
/// Returns an error when inputs contain nulls, are empty, or eigendecomposition fails.
pub fn generalized_f64(
    matrix_a: &FixedSizeListArray,
    matrix_b: &FixedSizeListArray,
) -> Result<crate::linalg::eigen::NdarrayGeneralizedEigenResult<f64>, ArrowInteropError> {
    let left_view = fixed_size_list_view::<Float64Type>(matrix_a)?;
    let right_view = fixed_size_list_view::<Float64Type>(matrix_b)?;
    Ok(crate::linalg::eigen::generalized_view(&left_view, &right_view)?)
}

/// Compute non-symmetric `f32` eigen decomposition directly from an Arrow dense matrix.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or eigendecomposition fails.
pub fn nonsymmetric_f32(
    matrix: &FixedSizeListArray,
) -> Result<crate::linalg::eigen::NdarrayNonsymmetricEigenResult<f32>, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float32Type>(matrix)?;
    Ok(crate::linalg::eigen::nonsymmetric_view(&matrix_view)?)
}

/// Compute non-symmetric `f64` eigen decomposition directly from an Arrow dense matrix.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or eigendecomposition fails.
pub fn nonsymmetric_f64(
    matrix: &FixedSizeListArray,
) -> Result<crate::linalg::eigen::NdarrayNonsymmetricEigenResult<f64>, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float64Type>(matrix)?;
    Ok(crate::linalg::eigen::nonsymmetric_view(&matrix_view)?)
}

/// Balance a non-symmetric `f32` matrix directly from Arrow dense storage.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or balancing fails.
pub fn balance_nonsymmetric_f32(
    matrix: &FixedSizeListArray,
    config: &crate::linalg::eigen::NonsymmetricEigenConfig<f32>,
) -> Result<(FixedSizeListArray, PrimitiveArray<Float32Type>), ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float32Type>(matrix)?;
    let (balanced, diagonal) =
        crate::linalg::eigen::balance_nonsymmetric_view(&matrix_view, config)?;
    Ok((
        fixed_size_list_from_owned::<Float32Type>(balanced)?,
        primitive_array_from_owned::<Float32Type>(diagonal),
    ))
}

/// Balance a non-symmetric `f64` matrix directly from Arrow dense storage.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or balancing fails.
pub fn balance_nonsymmetric_f64(
    matrix: &FixedSizeListArray,
    config: &crate::linalg::eigen::NonsymmetricEigenConfig<f64>,
) -> Result<(FixedSizeListArray, PrimitiveArray<Float64Type>), ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float64Type>(matrix)?;
    let (balanced, diagonal) =
        crate::linalg::eigen::balance_nonsymmetric_view(&matrix_view, config)?;
    Ok((
        fixed_size_list_from_owned::<Float64Type>(balanced)?,
        primitive_array_from_owned::<Float64Type>(diagonal),
    ))
}

/// Compute matched left/right non-symmetric `f32` eigenvectors directly from an Arrow matrix.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or eigendecomposition fails.
pub fn nonsymmetric_bi_f32(
    matrix: &FixedSizeListArray,
    config: &crate::linalg::eigen::NonsymmetricEigenConfig<f32>,
) -> Result<crate::linalg::eigen::NdarrayNonsymmetricBiEigenResult<f32>, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float32Type>(matrix)?;
    Ok(crate::linalg::eigen::nonsymmetric_bi_view(&matrix_view, config)?)
}

/// Compute matched left/right non-symmetric `f64` eigenvectors directly from an Arrow matrix.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or eigendecomposition fails.
pub fn nonsymmetric_bi_f64(
    matrix: &FixedSizeListArray,
    config: &crate::linalg::eigen::NonsymmetricEigenConfig<f64>,
) -> Result<crate::linalg::eigen::NdarrayNonsymmetricBiEigenResult<f64>, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float64Type>(matrix)?;
    Ok(crate::linalg::eigen::nonsymmetric_bi_view(&matrix_view, config)?)
}
