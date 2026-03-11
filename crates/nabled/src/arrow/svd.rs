//! Arrow adapters for singular value decomposition workflows.

use arrow_array::FixedSizeListArray;
use arrow_array::types::{Float32Type, Float64Type};

use super::{
    ArrowInteropError, complex64_matrix_from_owned, complex64_matrix_view,
    fixed_size_list_from_owned, fixed_size_list_view,
};

/// Compute `f32` SVD directly from an Arrow dense matrix.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or decomposition fails.
pub fn decompose_f32(
    matrix: &FixedSizeListArray,
) -> Result<crate::linalg::svd::NdarraySVD<f32>, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float32Type>(matrix)?;
    Ok(crate::linalg::svd::decompose_view(&matrix_view)?)
}

/// Compute `f64` SVD directly from an Arrow dense matrix.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or decomposition fails.
pub fn decompose_f64(
    matrix: &FixedSizeListArray,
) -> Result<crate::linalg::svd::NdarraySVD<f64>, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float64Type>(matrix)?;
    Ok(crate::linalg::svd::decompose_view(&matrix_view)?)
}

/// Compute truncated `f32` SVD directly from an Arrow dense matrix.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or decomposition fails.
pub fn decompose_truncated_f32(
    matrix: &FixedSizeListArray,
    k: usize,
) -> Result<crate::linalg::svd::NdarraySVD<f32>, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float32Type>(matrix)?;
    Ok(crate::linalg::svd::decompose_truncated_view(&matrix_view, k)?)
}

/// Compute truncated `f64` SVD directly from an Arrow dense matrix.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or decomposition fails.
pub fn decompose_truncated_f64(
    matrix: &FixedSizeListArray,
    k: usize,
) -> Result<crate::linalg::svd::NdarraySVD<f64>, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float64Type>(matrix)?;
    Ok(crate::linalg::svd::decompose_truncated_view(&matrix_view, k)?)
}

/// Compute `f32` SVD with tolerance directly from an Arrow dense matrix.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or decomposition fails.
pub fn decompose_with_tolerance_f32(
    matrix: &FixedSizeListArray,
    tolerance: f32,
) -> Result<crate::linalg::svd::NdarraySVD<f32>, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float32Type>(matrix)?;
    Ok(crate::linalg::svd::decompose_with_tolerance_view(&matrix_view, tolerance)?)
}

/// Compute `f64` SVD with tolerance directly from an Arrow dense matrix.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or decomposition fails.
pub fn decompose_with_tolerance_f64(
    matrix: &FixedSizeListArray,
    tolerance: f64,
) -> Result<crate::linalg::svd::NdarraySVD<f64>, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float64Type>(matrix)?;
    Ok(crate::linalg::svd::decompose_with_tolerance_view(&matrix_view, tolerance)?)
}

/// Compute the `f32` pseudo-inverse directly from an Arrow dense matrix.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or decomposition fails.
pub fn pseudo_inverse_f32(
    matrix: &FixedSizeListArray,
    config: crate::linalg::svd::PseudoInverseConfig<f32>,
) -> Result<FixedSizeListArray, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float32Type>(matrix)?;
    let output = crate::linalg::svd::pseudo_inverse_view(&matrix_view, &config)?;
    fixed_size_list_from_owned::<Float32Type>(output)
}

/// Compute the `f64` pseudo-inverse directly from an Arrow dense matrix.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or decomposition fails.
pub fn pseudo_inverse_f64(
    matrix: &FixedSizeListArray,
    config: crate::linalg::svd::PseudoInverseConfig<f64>,
) -> Result<FixedSizeListArray, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float64Type>(matrix)?;
    let output = crate::linalg::svd::pseudo_inverse_view(&matrix_view, &config)?;
    fixed_size_list_from_owned::<Float64Type>(output)
}

/// Compute the `f32` null space directly from an Arrow dense matrix.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or decomposition fails.
pub fn null_space_f32(
    matrix: &FixedSizeListArray,
    tolerance: Option<f32>,
) -> Result<FixedSizeListArray, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float32Type>(matrix)?;
    let output = crate::linalg::svd::null_space_view(&matrix_view, tolerance)?;
    fixed_size_list_from_owned::<Float32Type>(output)
}

/// Compute the `f64` null space directly from an Arrow dense matrix.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or decomposition fails.
pub fn null_space_f64(
    matrix: &FixedSizeListArray,
    tolerance: Option<f64>,
) -> Result<FixedSizeListArray, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float64Type>(matrix)?;
    let output = crate::linalg::svd::null_space_view(&matrix_view, tolerance)?;
    fixed_size_list_from_owned::<Float64Type>(output)
}

/// Compute complex SVD directly from an Arrow complex dense matrix.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or decomposition fails.
pub fn decompose_complex(
    matrix: &FixedSizeListArray,
) -> Result<crate::linalg::svd::NdarrayComplexSVD, ArrowInteropError> {
    let matrix_view = complex64_matrix_view(matrix)?;
    Ok(crate::linalg::svd::decompose_complex_view(&matrix_view)?)
}

/// Reconstruct a dense `f32` matrix from an ndarray-native SVD result into Arrow form.
///
/// # Errors
/// Returns an error when the reconstructed matrix cannot be represented as Arrow dense data.
pub fn reconstruct_f32(
    svd: &crate::linalg::svd::NdarraySVD<f32>,
) -> Result<FixedSizeListArray, ArrowInteropError> {
    fixed_size_list_from_owned::<Float32Type>(crate::linalg::svd::reconstruct_matrix(svd))
}

/// Reconstruct a dense `f64` matrix from an ndarray-native SVD result into Arrow form.
///
/// # Errors
/// Returns an error when the reconstructed matrix cannot be represented as Arrow dense data.
pub fn reconstruct_f64(
    svd: &crate::linalg::svd::NdarraySVD<f64>,
) -> Result<FixedSizeListArray, ArrowInteropError> {
    fixed_size_list_from_owned::<Float64Type>(crate::linalg::svd::reconstruct_matrix(svd))
}

/// Reconstruct a dense complex matrix from an ndarray-native SVD result into Arrow form.
///
/// # Errors
/// Returns an error when the reconstructed matrix cannot be represented as Arrow dense data.
pub fn reconstruct_complex(
    svd: &crate::linalg::svd::NdarrayComplexSVD,
) -> Result<FixedSizeListArray, ArrowInteropError> {
    complex64_matrix_from_owned(crate::linalg::svd::reconstruct_matrix_complex(svd))
}

/// Compute the condition number from an ndarray-native real SVD result.
pub fn condition_number<T: nabled_core::scalar::NabledReal>(
    svd: &crate::linalg::svd::NdarraySVD<T>,
) -> T {
    crate::linalg::svd::condition_number(svd)
}

/// Compute the numerical rank from an ndarray-native real SVD result.
pub fn rank<T: nabled_core::scalar::NabledReal>(
    svd: &crate::linalg::svd::NdarraySVD<T>,
    tolerance: Option<T>,
) -> usize {
    crate::linalg::svd::rank(svd, tolerance)
}
