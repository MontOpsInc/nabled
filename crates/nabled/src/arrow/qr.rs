//! Arrow adapters for QR decomposition workflows.

use arrow_array::types::{Float32Type, Float64Type};
use arrow_array::{FixedSizeListArray, PrimitiveArray};

use super::{
    ArrowInteropError, fixed_size_list_from_owned, fixed_size_list_view,
    primitive_array_from_owned, primitive_array_view,
};

/// Compute `f32` QR decomposition directly from an Arrow dense matrix.
///
/// The inbound Arrow -> ndarray bridge is zero-copy. The decomposition result remains in
/// `nabled`'s ndarray-native result struct.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or decomposition fails.
pub fn decompose_f32(
    matrix: &FixedSizeListArray,
    config: &crate::linalg::qr::QRConfig<f32>,
) -> Result<crate::linalg::qr::QRResult<f32>, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float32Type>(matrix)?;
    Ok(crate::linalg::qr::decompose_view(&matrix_view, config)?)
}

/// Compute `f64` QR decomposition directly from an Arrow dense matrix.
///
/// The inbound Arrow -> ndarray bridge is zero-copy. The decomposition result remains in
/// `nabled`'s ndarray-native result struct.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or decomposition fails.
pub fn decompose_f64(
    matrix: &FixedSizeListArray,
    config: &crate::linalg::qr::QRConfig<f64>,
) -> Result<crate::linalg::qr::QRResult<f64>, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float64Type>(matrix)?;
    Ok(crate::linalg::qr::decompose_view(&matrix_view, config)?)
}

/// Solve `f32` least-squares directly from Arrow dense inputs.
///
/// # Errors
/// Returns an error when inputs contain nulls, are empty, or dimensions mismatch.
pub fn solve_least_squares_f32(
    matrix: &FixedSizeListArray,
    rhs: &PrimitiveArray<Float32Type>,
    config: &crate::linalg::qr::QRConfig<f32>,
) -> Result<PrimitiveArray<Float32Type>, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float32Type>(matrix)?;
    let rhs_view = primitive_array_view(rhs)?;
    let output = crate::linalg::qr::solve_least_squares_view(&matrix_view, &rhs_view, config)?;
    Ok(primitive_array_from_owned::<Float32Type>(output))
}

/// Solve `f64` least-squares directly from Arrow dense inputs.
///
/// # Errors
/// Returns an error when inputs contain nulls, are empty, or dimensions mismatch.
pub fn solve_least_squares_f64(
    matrix: &FixedSizeListArray,
    rhs: &PrimitiveArray<Float64Type>,
    config: &crate::linalg::qr::QRConfig<f64>,
) -> Result<PrimitiveArray<Float64Type>, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float64Type>(matrix)?;
    let rhs_view = primitive_array_view(rhs)?;
    let output = crate::linalg::qr::solve_least_squares_view(&matrix_view, &rhs_view, config)?;
    Ok(primitive_array_from_owned::<Float64Type>(output))
}

/// Reconstruct a dense `f32` matrix from an ndarray-native QR result into Arrow form.
///
/// # Errors
/// Returns an error when the reconstructed matrix cannot be represented as Arrow dense data.
pub fn reconstruct_f32(
    qr: &crate::linalg::qr::QRResult<f32>,
) -> Result<FixedSizeListArray, ArrowInteropError> {
    let output = crate::linalg::qr::reconstruct_matrix(qr);
    fixed_size_list_from_owned::<Float32Type>(output)
}

/// Reconstruct a dense `f64` matrix from an ndarray-native QR result into Arrow form.
///
/// # Errors
/// Returns an error when the reconstructed matrix cannot be represented as Arrow dense data.
pub fn reconstruct_f64(
    qr: &crate::linalg::qr::QRResult<f64>,
) -> Result<FixedSizeListArray, ArrowInteropError> {
    let output = crate::linalg::qr::reconstruct_matrix(qr);
    fixed_size_list_from_owned::<Float64Type>(output)
}
