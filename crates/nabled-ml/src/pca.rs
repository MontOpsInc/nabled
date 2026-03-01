//! Principal component analysis over ndarray matrices.

use std::fmt;

use nabled_linalg::svd;
use ndarray::{Array1, Array2, ArrayView2, Axis, s};

/// PCA result for ndarray matrices.
#[derive(Debug, Clone)]
pub struct NdarrayPCAResult {
    /// Principal components as rows (`k x features`).
    pub components:               Array2<f64>,
    /// Explained variance for each retained component.
    pub explained_variance:       Array1<f64>,
    /// Explained variance ratio for each retained component.
    pub explained_variance_ratio: Array1<f64>,
    /// Column means used for centering.
    pub mean:                     Array1<f64>,
    /// Scores (`samples x k`).
    pub scores:                   Array2<f64>,
}

/// Error type for PCA operations.
#[derive(Debug, Clone, PartialEq)]
pub enum PCAError {
    /// Input matrix is empty.
    EmptyMatrix,
    /// Invalid user input.
    InvalidInput(String),
    /// Decomposition failed.
    DecompositionFailed,
}

impl fmt::Display for PCAError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PCAError::EmptyMatrix => write!(f, "Matrix cannot be empty"),
            PCAError::InvalidInput(message) => write!(f, "Invalid input: {message}"),
            PCAError::DecompositionFailed => write!(f, "PCA decomposition failed"),
        }
    }
}

impl std::error::Error for PCAError {}

fn usize_to_f64(value: usize) -> f64 { u32::try_from(value).map_or(f64::from(u32::MAX), f64::from) }

fn center_columns(matrix: &Array2<f64>) -> Result<(Array2<f64>, Array1<f64>), PCAError> {
    if matrix.is_empty() {
        return Err(PCAError::EmptyMatrix);
    }
    let mean = matrix
        .mean_axis(Axis(0))
        .ok_or_else(|| PCAError::InvalidInput("failed to compute column means".to_string()))?;
    let mut centered = matrix.clone();
    for row in 0..matrix.nrows() {
        for col in 0..matrix.ncols() {
            centered[[row, col]] -= mean[col];
        }
    }
    Ok((centered, mean))
}

/// Compute principal component analysis.
///
/// # Errors
/// Returns an error for invalid input or decomposition failure.
pub fn compute_pca(
    matrix: &Array2<f64>,
    n_components: Option<usize>,
) -> Result<NdarrayPCAResult, PCAError> {
    let (centered, mean) = center_columns(matrix)?;
    let svd = svd::decompose(&centered).map_err(|_| PCAError::DecompositionFailed)?;

    let max_components = centered.nrows().min(centered.ncols());
    let keep = n_components.unwrap_or(max_components).min(max_components);
    if keep == 0 {
        return Err(PCAError::InvalidInput("n_components must be greater than 0".to_string()));
    }

    let components = svd.vt.slice(s![..keep, ..]).to_owned();
    let scores = centered.dot(&components.t());

    let denominator = (usize_to_f64(centered.nrows()) - 1.0).max(1.0);
    let mut explained_variance = Array1::<f64>::zeros(keep);
    for i in 0..keep {
        explained_variance[i] = (svd.singular_values[i] * svd.singular_values[i]) / denominator;
    }

    let total_variance = explained_variance.iter().sum::<f64>().max(f64::EPSILON);
    let explained_variance_ratio = explained_variance.map(|value| *value / total_variance);

    Ok(NdarrayPCAResult { components, explained_variance, explained_variance_ratio, mean, scores })
}

/// Compute principal component analysis from a matrix view.
///
/// # Errors
/// Returns an error for invalid input or decomposition failure.
pub fn compute_pca_view(
    matrix: &ArrayView2<'_, f64>,
    n_components: Option<usize>,
) -> Result<NdarrayPCAResult, PCAError> {
    compute_pca(&matrix.to_owned(), n_components)
}

/// Project data to PCA score space.
#[must_use]
pub fn transform(matrix: &Array2<f64>, pca: &NdarrayPCAResult) -> Array2<f64> {
    let mut centered = matrix.clone();
    for row in 0..matrix.nrows() {
        for col in 0..matrix.ncols() {
            centered[[row, col]] -= pca.mean[col];
        }
    }
    centered.dot(&pca.components.t())
}

/// Project data to PCA score space from a matrix view.
#[must_use]
pub fn transform_view(matrix: &ArrayView2<'_, f64>, pca: &NdarrayPCAResult) -> Array2<f64> {
    transform(&matrix.to_owned(), pca)
}

/// Reconstruct from PCA scores.
#[must_use]
pub fn inverse_transform(scores: &Array2<f64>, pca: &NdarrayPCAResult) -> Array2<f64> {
    let mut reconstructed = scores.dot(&pca.components);
    for row in 0..reconstructed.nrows() {
        for col in 0..reconstructed.ncols() {
            reconstructed[[row, col]] += pca.mean[col];
        }
    }
    reconstructed
}

/// Reconstruct from PCA scores provided as a matrix view.
#[must_use]
pub fn inverse_transform_view(scores: &ArrayView2<'_, f64>, pca: &NdarrayPCAResult) -> Array2<f64> {
    inverse_transform(&scores.to_owned(), pca)
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use super::*;

    #[test]
    fn pca_roundtrip_is_consistent() {
        let matrix =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 2.0, 1.0, 3.0, 4.0, 4.0, 3.0]).unwrap();
        let pca = compute_pca(&matrix, Some(2)).unwrap();
        let transformed = transform(&matrix, &pca);
        let reconstructed = inverse_transform(&transformed, &pca);
        for i in 0..matrix.nrows() {
            for j in 0..matrix.ncols() {
                assert!((matrix[[i, j]] - reconstructed[[i, j]]).abs() < 1e-8);
            }
        }
    }

    #[test]
    fn pca_rejects_zero_components() {
        let matrix =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 2.0, 1.0, 3.0, 4.0, 4.0, 3.0]).unwrap();
        let result = compute_pca(&matrix, Some(0));
        assert!(matches!(result, Err(PCAError::InvalidInput(_))));
    }

    #[test]
    fn explained_variance_ratio_sums_to_one() {
        let matrix =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 2.0, 1.0, 3.0, 4.0, 4.0, 3.0]).unwrap();
        let pca = compute_pca(&matrix, Some(2)).unwrap();
        let sum = pca.explained_variance_ratio.iter().sum::<f64>();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn pca_view_variants_match_owned() {
        let matrix =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 2.0, 1.0, 3.0, 4.0, 4.0, 3.0]).unwrap();
        let pca_owned = compute_pca(&matrix, Some(2)).unwrap();
        let pca_view = compute_pca_view(&matrix.view(), Some(2)).unwrap();

        assert_eq!(pca_owned.components.dim(), pca_view.components.dim());
        assert_eq!(pca_owned.scores.dim(), pca_view.scores.dim());

        let transformed_owned = transform(&matrix, &pca_owned);
        let transformed_view = transform_view(&matrix.view(), &pca_owned);
        let reconstructed_owned = inverse_transform(&transformed_owned, &pca_owned);
        let reconstructed_view = inverse_transform_view(&transformed_owned.view(), &pca_owned);

        for i in 0..matrix.nrows() {
            for j in 0..matrix.ncols() {
                assert!((transformed_owned[[i, j]] - transformed_view[[i, j]]).abs() < 1e-12);
                assert!((reconstructed_owned[[i, j]] - reconstructed_view[[i, j]]).abs() < 1e-12);
            }
        }
    }
}
