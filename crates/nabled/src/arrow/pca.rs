//! Arrow adapters for PCA workflows.

use arrow_array::FixedSizeListArray;
use arrow_array::types::{Float32Type, Float64Type};
use ndarray::{ArrayView1, ArrayView2};
use num_complex::Complex64;

use super::{
    ArrowInteropError, complex64_matrix_from_owned, complex64_matrix_view,
    fixed_size_list_from_owned, fixed_size_list_view,
};

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
    transform_f32_from_components_view(matrix, &pca.components.view(), &pca.mean.view())
}

/// Project `f32` Arrow dense data into PCA score space using borrowed PCA factors.
///
/// # Errors
/// Returns an error when the matrix contains nulls or shape conversion fails.
pub fn transform_f32_from_components_view(
    matrix: &FixedSizeListArray,
    components: &ArrayView2<'_, f32>,
    mean: &ArrayView1<'_, f32>,
) -> Result<FixedSizeListArray, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float32Type>(matrix)?;
    let output = crate::ml::pca::transform_from_components_view(&matrix_view, components, mean)?;
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
    transform_f64_from_components_view(matrix, &pca.components.view(), &pca.mean.view())
}

/// Project `f64` Arrow dense data into PCA score space using borrowed PCA factors.
///
/// # Errors
/// Returns an error when the matrix contains nulls or shape conversion fails.
pub fn transform_f64_from_components_view(
    matrix: &FixedSizeListArray,
    components: &ArrayView2<'_, f64>,
    mean: &ArrayView1<'_, f64>,
) -> Result<FixedSizeListArray, ArrowInteropError> {
    let matrix_view = fixed_size_list_view::<Float64Type>(matrix)?;
    let output = crate::ml::pca::transform_from_components_view(&matrix_view, components, mean)?;
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
    inverse_transform_f32_from_components_view(scores, &pca.components.view(), &pca.mean.view())
}

/// Reconstruct `f32` Arrow dense data from PCA scores using borrowed PCA factors.
///
/// # Errors
/// Returns an error when the score matrix contains nulls or shape conversion fails.
pub fn inverse_transform_f32_from_components_view(
    scores: &FixedSizeListArray,
    components: &ArrayView2<'_, f32>,
    mean: &ArrayView1<'_, f32>,
) -> Result<FixedSizeListArray, ArrowInteropError> {
    let score_view = fixed_size_list_view::<Float32Type>(scores)?;
    let output =
        crate::ml::pca::inverse_transform_from_components_view(&score_view, components, mean)?;
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
    inverse_transform_f64_from_components_view(scores, &pca.components.view(), &pca.mean.view())
}

/// Reconstruct `f64` Arrow dense data from PCA scores using borrowed PCA factors.
///
/// # Errors
/// Returns an error when the score matrix contains nulls or shape conversion fails.
pub fn inverse_transform_f64_from_components_view(
    scores: &FixedSizeListArray,
    components: &ArrayView2<'_, f64>,
    mean: &ArrayView1<'_, f64>,
) -> Result<FixedSizeListArray, ArrowInteropError> {
    let score_view = fixed_size_list_view::<Float64Type>(scores)?;
    let output =
        crate::ml::pca::inverse_transform_from_components_view(&score_view, components, mean)?;
    fixed_size_list_from_owned::<Float64Type>(output)
}

/// Compute complex PCA directly from Arrow complex dense input.
///
/// # Errors
/// Returns an error when the matrix contains nulls, is empty, or PCA fails.
pub fn compute_complex(
    matrix: &FixedSizeListArray,
    n_components: Option<usize>,
) -> Result<crate::ml::pca::NdarrayComplexPCAResult, ArrowInteropError> {
    let matrix_view = complex64_matrix_view(matrix)?;
    Ok(crate::ml::pca::compute_pca_complex_view(&matrix_view, n_components)?)
}

/// Project complex Arrow dense data into PCA score space.
///
/// # Errors
/// Returns an error when the matrix contains nulls or shape conversion fails.
pub fn transform_complex(
    matrix: &FixedSizeListArray,
    pca: &crate::ml::pca::NdarrayComplexPCAResult,
) -> Result<FixedSizeListArray, ArrowInteropError> {
    transform_complex_from_components_view(matrix, &pca.components.view(), &pca.mean.view())
}

/// Project complex Arrow dense data into PCA score space using borrowed PCA factors.
///
/// # Errors
/// Returns an error when the matrix contains nulls or shape conversion fails.
pub fn transform_complex_from_components_view(
    matrix: &FixedSizeListArray,
    components: &ArrayView2<'_, Complex64>,
    mean: &ArrayView1<'_, Complex64>,
) -> Result<FixedSizeListArray, ArrowInteropError> {
    let matrix_view = complex64_matrix_view(matrix)?;
    complex64_matrix_from_owned(crate::ml::pca::transform_complex_from_components_view(
        &matrix_view,
        components,
        mean,
    )?)
}

/// Reconstruct complex Arrow dense data from PCA scores.
///
/// # Errors
/// Returns an error when the score matrix contains nulls or shape conversion fails.
pub fn inverse_transform_complex(
    scores: &FixedSizeListArray,
    pca: &crate::ml::pca::NdarrayComplexPCAResult,
) -> Result<FixedSizeListArray, ArrowInteropError> {
    inverse_transform_complex_from_components_view(scores, &pca.components.view(), &pca.mean.view())
}

/// Reconstruct complex Arrow dense data from PCA scores using borrowed PCA factors.
///
/// # Errors
/// Returns an error when the score matrix contains nulls or shape conversion fails.
pub fn inverse_transform_complex_from_components_view(
    scores: &FixedSizeListArray,
    components: &ArrayView2<'_, Complex64>,
    mean: &ArrayView1<'_, Complex64>,
) -> Result<FixedSizeListArray, ArrowInteropError> {
    let score_view = complex64_matrix_view(scores)?;
    complex64_matrix_from_owned(crate::ml::pca::inverse_transform_complex_from_components_view(
        &score_view,
        components,
        mean,
    )?)
}
