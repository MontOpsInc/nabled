//! Arrow adapters for batched decomposition workflows.

use arrow_array::FixedSizeListArray;
use arrow_array::types::{Float32Type, Float64Type};
use arrow_schema::Field;
use ndarray::Ix3;

use super::{ArrowInteropError, fixed_shape_tensor_viewd};

fn fixed_shape_tensor_view3_f32<'a>(
    field: &'a Field,
    array: &'a FixedSizeListArray,
) -> Result<ndarray::ArrayView3<'a, f32>, ArrowInteropError> {
    let view = fixed_shape_tensor_viewd::<Float32Type>(field, array)?;
    view.into_dimensionality::<Ix3>()
        .map_err(|error: ndarray::ShapeError| ArrowInteropError::InvalidShape(error.to_string()))
}

fn fixed_shape_tensor_view3_f64<'a>(
    field: &'a Field,
    array: &'a FixedSizeListArray,
) -> Result<ndarray::ArrayView3<'a, f64>, ArrowInteropError> {
    let view = fixed_shape_tensor_viewd::<Float64Type>(field, array)?;
    view.into_dimensionality::<Ix3>()
        .map_err(|error: ndarray::ShapeError| ArrowInteropError::InvalidShape(error.to_string()))
}

/// Compute batched `f32` QR decomposition directly from Arrow fixed-shape tensor input.
///
/// # Errors
/// Returns an error when the tensor is invalid, not rank-3, or decomposition fails.
pub fn qr_f32(
    field: &Field,
    array: &FixedSizeListArray,
    config: &crate::linalg::qr::QRConfig<f32>,
) -> Result<Vec<crate::linalg::qr::QRResult<f32>>, ArrowInteropError> {
    let view = fixed_shape_tensor_view3_f32(field, array)?;
    Ok(crate::linalg::batched::qr_view(&view, config)?)
}

/// Compute batched `f64` QR decomposition directly from Arrow fixed-shape tensor input.
///
/// # Errors
/// Returns an error when the tensor is invalid, not rank-3, or decomposition fails.
pub fn qr_f64(
    field: &Field,
    array: &FixedSizeListArray,
    config: &crate::linalg::qr::QRConfig<f64>,
) -> Result<Vec<crate::linalg::qr::QRResult<f64>>, ArrowInteropError> {
    let view = fixed_shape_tensor_view3_f64(field, array)?;
    Ok(crate::linalg::batched::qr_view(&view, config)?)
}

/// Compute batched `f32` SVD directly from Arrow fixed-shape tensor input.
///
/// # Errors
/// Returns an error when the tensor is invalid, not rank-3, or decomposition fails.
pub fn svd_f32(
    field: &Field,
    array: &FixedSizeListArray,
) -> Result<Vec<crate::linalg::svd::NdarraySVD<f32>>, ArrowInteropError> {
    let view = fixed_shape_tensor_view3_f32(field, array)?;
    Ok(crate::linalg::batched::svd_view(&view)?)
}

/// Compute batched `f64` SVD directly from Arrow fixed-shape tensor input.
///
/// # Errors
/// Returns an error when the tensor is invalid, not rank-3, or decomposition fails.
pub fn svd_f64(
    field: &Field,
    array: &FixedSizeListArray,
) -> Result<Vec<crate::linalg::svd::NdarraySVD<f64>>, ArrowInteropError> {
    let view = fixed_shape_tensor_view3_f64(field, array)?;
    Ok(crate::linalg::batched::svd_view(&view)?)
}

/// Compute batched `f32` LU decomposition directly from Arrow fixed-shape tensor input.
///
/// # Errors
/// Returns an error when the tensor is invalid, not rank-3, or decomposition fails.
pub fn lu_f32(
    field: &Field,
    array: &FixedSizeListArray,
) -> Result<Vec<crate::linalg::lu::NdarrayLUResult<f32>>, ArrowInteropError> {
    let view = fixed_shape_tensor_view3_f32(field, array)?;
    Ok(crate::linalg::batched::lu_view(&view)?)
}

/// Compute batched `f64` LU decomposition directly from Arrow fixed-shape tensor input.
///
/// # Errors
/// Returns an error when the tensor is invalid, not rank-3, or decomposition fails.
pub fn lu_f64(
    field: &Field,
    array: &FixedSizeListArray,
) -> Result<Vec<crate::linalg::lu::NdarrayLUResult<f64>>, ArrowInteropError> {
    let view = fixed_shape_tensor_view3_f64(field, array)?;
    Ok(crate::linalg::batched::lu_view(&view)?)
}

/// Compute batched `f32` Cholesky decomposition directly from Arrow fixed-shape tensor input.
///
/// # Errors
/// Returns an error when the tensor is invalid, not rank-3, or decomposition fails.
pub fn cholesky_f32(
    field: &Field,
    array: &FixedSizeListArray,
) -> Result<Vec<crate::linalg::cholesky::NdarrayCholeskyResult<f32>>, ArrowInteropError> {
    let view = fixed_shape_tensor_view3_f32(field, array)?;
    Ok(crate::linalg::batched::cholesky_view(&view)?)
}

/// Compute batched `f64` Cholesky decomposition directly from Arrow fixed-shape tensor input.
///
/// # Errors
/// Returns an error when the tensor is invalid, not rank-3, or decomposition fails.
pub fn cholesky_f64(
    field: &Field,
    array: &FixedSizeListArray,
) -> Result<Vec<crate::linalg::cholesky::NdarrayCholeskyResult<f64>>, ArrowInteropError> {
    let view = fixed_shape_tensor_view3_f64(field, array)?;
    Ok(crate::linalg::batched::cholesky_view(&view)?)
}

/// Compute batched `f32` symmetric eigendecomposition directly from Arrow fixed-shape tensor input.
///
/// # Errors
/// Returns an error when the tensor is invalid, not rank-3, or decomposition fails.
pub fn symmetric_eigen_f32(
    field: &Field,
    array: &FixedSizeListArray,
) -> Result<Vec<crate::linalg::eigen::NdarrayEigenResult<f32>>, ArrowInteropError> {
    let view = fixed_shape_tensor_view3_f32(field, array)?;
    Ok(crate::linalg::batched::symmetric_eigen_view(&view)?)
}

/// Compute batched `f64` symmetric eigendecomposition directly from Arrow fixed-shape tensor input.
///
/// # Errors
/// Returns an error when the tensor is invalid, not rank-3, or decomposition fails.
pub fn symmetric_eigen_f64(
    field: &Field,
    array: &FixedSizeListArray,
) -> Result<Vec<crate::linalg::eigen::NdarrayEigenResult<f64>>, ArrowInteropError> {
    let view = fixed_shape_tensor_view3_f64(field, array)?;
    Ok(crate::linalg::batched::symmetric_eigen_view(&view)?)
}
