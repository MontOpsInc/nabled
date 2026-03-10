//! Arrow adapters for PCA workflows.

use arrow_array::FixedSizeListArray;
use arrow_array::types::{Float32Type, Float64Type};

use super::{ArrowInteropError, fixed_size_list_from_owned, fixed_size_list_view};

/// Compute `f32` PCA directly from Arrow dense input.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or PCA fails.
pub fn compute_f32(
    matrix: &FixedSizeListArray,
    n_components: Option<usize>,
) -> Result<crate::ml::pca::NdarrayPCAResult<f32>, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float32Type>(matrix)?;
    Ok(crate::ml::pca::compute_pca_view(&matrix_view, n_components)?)
}

/// Compute `f64` PCA directly from Arrow dense input.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or PCA fails.
pub fn compute_f64(
    matrix: &FixedSizeListArray,
    n_components: Option<usize>,
) -> Result<crate::ml::pca::NdarrayPCAResult<f64>, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float64Type>(matrix)?;
    Ok(crate::ml::pca::compute_pca_view(&matrix_view, n_components)?)
}

/// Project `f32` Arrow dense data into PCA score space.
///
/// # Errors
/// Returns an error when the matrix contains nulls or shape conversion fails.
pub fn transform_f32(
    matrix: &FixedSizeListArray,
    pca: &crate::ml::pca::NdarrayPCAResult<f32>,
) -> Result<FixedSizeListArray, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float32Type>(matrix)?;
    let output = crate::ml::pca::transform_view(&matrix_view, pca);
    fixed_size_list_from_owned::<Float32Type>(output)
}

/// Project `f64` Arrow dense data into PCA score space.
///
/// # Errors
/// Returns an error when the matrix contains nulls or shape conversion fails.
pub fn transform_f64(
    matrix: &FixedSizeListArray,
    pca: &crate::ml::pca::NdarrayPCAResult<f64>,
) -> Result<FixedSizeListArray, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float64Type>(matrix)?;
    let output = crate::ml::pca::transform_view(&matrix_view, pca);
    fixed_size_list_from_owned::<Float64Type>(output)
}

/// Reconstruct `f32` Arrow dense data from PCA scores.
///
/// # Errors
/// Returns an error when the score matrix contains nulls or shape conversion fails.
pub fn inverse_transform_f32(
    scores: &FixedSizeListArray,
    pca: &crate::ml::pca::NdarrayPCAResult<f32>,
) -> Result<FixedSizeListArray, ArrowInteropError> {
    let score_view = fixed_size_list_view::<Float32Type>(scores)?;
    let output = crate::ml::pca::inverse_transform_view(&score_view, pca);
    fixed_size_list_from_owned::<Float32Type>(output)
}

/// Reconstruct `f64` Arrow dense data from PCA scores.
///
/// # Errors
/// Returns an error when the score matrix contains nulls or shape conversion fails.
pub fn inverse_transform_f64(
    scores: &FixedSizeListArray,
    pca: &crate::ml::pca::NdarrayPCAResult<f64>,
) -> Result<FixedSizeListArray, ArrowInteropError> {
    let score_view = fixed_size_list_view::<Float64Type>(scores)?;
    let output = crate::ml::pca::inverse_transform_view(&score_view, pca);
    fixed_size_list_from_owned::<Float64Type>(output)
}
