//! Arrow adapters for matrix-function workflows.

use arrow_array::FixedSizeListArray;
use arrow_array::types::{Float32Type, Float64Type};

use super::{
    ArrowInteropError, complex64_matrix_from_owned, complex64_matrix_view,
    fixed_size_list_from_owned, fixed_size_list_view,
};

/// Compute the `f32` matrix exponential directly from Arrow dense input.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or the operation fails.
pub fn exp_f32(
    matrix: &FixedSizeListArray,
    max_terms: usize,
    tolerance: f32,
) -> Result<FixedSizeListArray, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float32Type>(matrix)?;
    let output =
        crate::linalg::matrix_functions::matrix_exp_view(&matrix_view, max_terms, tolerance)?;
    fixed_size_list_from_owned::<Float32Type>(output)
}

/// Compute the `f64` matrix exponential directly from Arrow dense input.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or the operation fails.
pub fn exp_f64(
    matrix: &FixedSizeListArray,
    max_terms: usize,
    tolerance: f64,
) -> Result<FixedSizeListArray, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float64Type>(matrix)?;
    let output =
        crate::linalg::matrix_functions::matrix_exp_view(&matrix_view, max_terms, tolerance)?;
    fixed_size_list_from_owned::<Float64Type>(output)
}

/// Compute the `f32` matrix exponential via eigen decomposition directly from Arrow dense input.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or the operation fails.
pub fn exp_eigen_f32(matrix: &FixedSizeListArray) -> Result<FixedSizeListArray, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float32Type>(matrix)?;
    let output = crate::linalg::matrix_functions::matrix_exp_eigen_view(&matrix_view)?;
    fixed_size_list_from_owned::<Float32Type>(output)
}

/// Compute the `f64` matrix exponential via eigen decomposition directly from Arrow dense input.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or the operation fails.
pub fn exp_eigen_f64(matrix: &FixedSizeListArray) -> Result<FixedSizeListArray, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float64Type>(matrix)?;
    let output = crate::linalg::matrix_functions::matrix_exp_eigen_view(&matrix_view)?;
    fixed_size_list_from_owned::<Float64Type>(output)
}

/// Compute the `f32` matrix logarithm via Taylor expansion directly from Arrow dense input.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or the operation fails.
pub fn log_taylor_f32(
    matrix: &FixedSizeListArray,
    max_terms: usize,
    tolerance: f32,
) -> Result<FixedSizeListArray, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float32Type>(matrix)?;
    let output = crate::linalg::matrix_functions::matrix_log_taylor_view(
        &matrix_view,
        max_terms,
        tolerance,
    )?;
    fixed_size_list_from_owned::<Float32Type>(output)
}

/// Compute the `f64` matrix logarithm via Taylor expansion directly from Arrow dense input.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or the operation fails.
pub fn log_taylor_f64(
    matrix: &FixedSizeListArray,
    max_terms: usize,
    tolerance: f64,
) -> Result<FixedSizeListArray, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float64Type>(matrix)?;
    let output = crate::linalg::matrix_functions::matrix_log_taylor_view(
        &matrix_view,
        max_terms,
        tolerance,
    )?;
    fixed_size_list_from_owned::<Float64Type>(output)
}

/// Compute the `f32` matrix logarithm via eigen decomposition directly from Arrow dense input.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or the operation fails.
pub fn log_eigen_f32(matrix: &FixedSizeListArray) -> Result<FixedSizeListArray, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float32Type>(matrix)?;
    let output = crate::linalg::matrix_functions::matrix_log_eigen_view(&matrix_view)?;
    fixed_size_list_from_owned::<Float32Type>(output)
}

/// Compute the `f64` matrix logarithm via eigen decomposition directly from Arrow dense input.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or the operation fails.
pub fn log_eigen_f64(matrix: &FixedSizeListArray) -> Result<FixedSizeListArray, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float64Type>(matrix)?;
    let output = crate::linalg::matrix_functions::matrix_log_eigen_view(&matrix_view)?;
    fixed_size_list_from_owned::<Float64Type>(output)
}

/// Compute the `f32` matrix logarithm via SVD directly from Arrow dense input.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or the operation fails.
pub fn log_svd_f32(matrix: &FixedSizeListArray) -> Result<FixedSizeListArray, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float32Type>(matrix)?;
    let output = crate::linalg::matrix_functions::matrix_log_svd_view(&matrix_view)?;
    fixed_size_list_from_owned::<Float32Type>(output)
}

/// Compute the `f64` matrix logarithm via SVD directly from Arrow dense input.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or the operation fails.
pub fn log_svd_f64(matrix: &FixedSizeListArray) -> Result<FixedSizeListArray, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float64Type>(matrix)?;
    let output = crate::linalg::matrix_functions::matrix_log_svd_view(&matrix_view)?;
    fixed_size_list_from_owned::<Float64Type>(output)
}

/// Compute the `f32` matrix power directly from Arrow dense input.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or the operation fails.
pub fn power_f32(
    matrix: &FixedSizeListArray,
    power: f32,
) -> Result<FixedSizeListArray, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float32Type>(matrix)?;
    let output = crate::linalg::matrix_functions::matrix_power_view(&matrix_view, power)?;
    fixed_size_list_from_owned::<Float32Type>(output)
}

/// Compute the `f64` matrix power directly from Arrow dense input.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or the operation fails.
pub fn power_f64(
    matrix: &FixedSizeListArray,
    power: f64,
) -> Result<FixedSizeListArray, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float64Type>(matrix)?;
    let output = crate::linalg::matrix_functions::matrix_power_view(&matrix_view, power)?;
    fixed_size_list_from_owned::<Float64Type>(output)
}

/// Compute the `f32` matrix sign directly from Arrow dense input.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or the operation fails.
pub fn sign_f32(matrix: &FixedSizeListArray) -> Result<FixedSizeListArray, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float32Type>(matrix)?;
    let output = crate::linalg::matrix_functions::matrix_sign_view(&matrix_view)?;
    fixed_size_list_from_owned::<Float32Type>(output)
}

/// Compute the `f64` matrix sign directly from Arrow dense input.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or the operation fails.
pub fn sign_f64(matrix: &FixedSizeListArray) -> Result<FixedSizeListArray, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float64Type>(matrix)?;
    let output = crate::linalg::matrix_functions::matrix_sign_view(&matrix_view)?;
    fixed_size_list_from_owned::<Float64Type>(output)
}

/// Compute the complex matrix exponential directly from Arrow complex dense input.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or the operation fails.
pub fn exp_complex(
    matrix: &FixedSizeListArray,
    max_terms: usize,
    tolerance: f64,
) -> Result<FixedSizeListArray, ArrowInteropError> {
    let matrix_view = complex64_matrix_view(matrix)?;
    complex64_matrix_from_owned(crate::linalg::matrix_functions::matrix_exp_complex_view(
        &matrix_view,
        max_terms,
        tolerance,
    )?)
}

/// Compute the complex matrix exponential via eigen decomposition directly from Arrow complex
/// input.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or the operation fails.
pub fn exp_eigen_complex(
    matrix: &FixedSizeListArray,
) -> Result<FixedSizeListArray, ArrowInteropError> {
    let matrix_view = complex64_matrix_view(matrix)?;
    complex64_matrix_from_owned(crate::linalg::matrix_functions::matrix_exp_eigen_complex_view(
        &matrix_view,
    )?)
}

/// Compute the complex matrix logarithm via eigen decomposition directly from Arrow complex input.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or the operation fails.
pub fn log_eigen_complex(
    matrix: &FixedSizeListArray,
) -> Result<FixedSizeListArray, ArrowInteropError> {
    let matrix_view = complex64_matrix_view(matrix)?;
    complex64_matrix_from_owned(crate::linalg::matrix_functions::matrix_log_eigen_complex_view(
        &matrix_view,
    )?)
}

/// Compute the complex matrix logarithm via SVD directly from Arrow complex input.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or the operation fails.
pub fn log_svd_complex(
    matrix: &FixedSizeListArray,
) -> Result<FixedSizeListArray, ArrowInteropError> {
    let matrix_view = complex64_matrix_view(matrix)?;
    complex64_matrix_from_owned(crate::linalg::matrix_functions::matrix_log_svd_complex_view(
        &matrix_view,
    )?)
}

/// Compute the complex matrix power directly from Arrow complex input.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or the operation fails.
pub fn power_complex(
    matrix: &FixedSizeListArray,
    power: f64,
) -> Result<FixedSizeListArray, ArrowInteropError> {
    let matrix_view = complex64_matrix_view(matrix)?;
    complex64_matrix_from_owned(crate::linalg::matrix_functions::matrix_power_complex_view(
        &matrix_view,
        power,
    )?)
}

/// Compute the complex matrix sign directly from Arrow complex input.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or the operation fails.
pub fn sign_complex(matrix: &FixedSizeListArray) -> Result<FixedSizeListArray, ArrowInteropError> {
    let matrix_view = complex64_matrix_view(matrix)?;
    complex64_matrix_from_owned(crate::linalg::matrix_functions::matrix_sign_complex_view(
        &matrix_view,
    )?)
}
