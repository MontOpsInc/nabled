//! Principal component analysis over ndarray matrices.

use std::fmt;

use nabled_core::scalar::NabledReal;
use nabled_linalg::svd;
use ndarray::{Array1, Array2, ArrayView2, Axis, s};
use num_complex::Complex64;

/// PCA result for ndarray matrices.
#[derive(Debug, Clone)]
pub struct NdarrayPCAResult<T: NabledReal> {
    /// Principal components as rows (`k x features`).
    pub components:               Array2<T>,
    /// Explained variance for each retained component.
    pub explained_variance:       Array1<T>,
    /// Explained variance ratio for each retained component.
    pub explained_variance_ratio: Array1<T>,
    /// Column means used for centering.
    pub mean:                     Array1<T>,
    /// Scores (`samples x k`).
    pub scores:                   Array2<T>,
}

/// PCA result for complex ndarray matrices.
#[derive(Debug, Clone)]
pub struct NdarrayComplexPCAResult {
    /// Principal components as rows (`k x features`).
    pub components:               Array2<Complex64>,
    /// Explained variance for each retained component.
    pub explained_variance:       Array1<f64>,
    /// Explained variance ratio for each retained component.
    pub explained_variance_ratio: Array1<f64>,
    /// Column means used for centering.
    pub mean:                     Array1<Complex64>,
    /// Scores (`samples x k`).
    pub scores:                   Array2<Complex64>,
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

fn usize_to_real<T: NabledReal>(value: usize) -> T {
    let fallback = T::from_u32(u32::MAX).unwrap_or(T::one());
    T::from_usize(value).unwrap_or(fallback)
}

fn center_columns<T: NabledReal>(
    matrix: &ArrayView2<'_, T>,
) -> Result<(Array2<T>, Array1<T>), PCAError> {
    if matrix.is_empty() {
        return Err(PCAError::EmptyMatrix);
    }
    let mean = matrix
        .mean_axis(Axis(0))
        .ok_or_else(|| PCAError::InvalidInput("failed to compute column means".to_string()))?;
    let mut centered = matrix.to_owned();
    for row in 0..matrix.nrows() {
        for col in 0..matrix.ncols() {
            centered[[row, col]] -= mean[col];
        }
    }
    Ok((centered, mean))
}

fn transform_impl<T: NabledReal>(
    matrix: &ArrayView2<'_, T>,
    pca: &NdarrayPCAResult<T>,
) -> Array2<T> {
    let mut centered = Array2::<T>::zeros((matrix.nrows(), matrix.ncols()));
    for row in 0..matrix.nrows() {
        for col in 0..matrix.ncols() {
            centered[[row, col]] = matrix[[row, col]] - pca.mean[col];
        }
    }
    centered.dot(&pca.components.t())
}

fn inverse_transform_impl<T: NabledReal>(
    scores: &ArrayView2<'_, T>,
    pca: &NdarrayPCAResult<T>,
) -> Array2<T> {
    let mut reconstructed = scores.dot(&pca.components);
    for row in 0..reconstructed.nrows() {
        for col in 0..reconstructed.ncols() {
            reconstructed[[row, col]] += pca.mean[col];
        }
    }
    reconstructed
}

fn center_columns_complex(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<(Array2<Complex64>, Array1<Complex64>), PCAError> {
    if matrix.is_empty() {
        return Err(PCAError::EmptyMatrix);
    }
    let mut mean = Array1::<Complex64>::zeros(matrix.ncols());
    for col in 0..matrix.ncols() {
        let mut sum = Complex64::new(0.0, 0.0);
        for row in 0..matrix.nrows() {
            sum += matrix[[row, col]];
        }
        mean[col] = sum / usize_to_real::<f64>(matrix.nrows());
    }

    let mut centered = matrix.to_owned();
    for row in 0..matrix.nrows() {
        for col in 0..matrix.ncols() {
            centered[[row, col]] -= mean[col];
        }
    }
    Ok((centered, mean))
}

fn transform_complex_impl(
    matrix: &ArrayView2<'_, Complex64>,
    pca: &NdarrayComplexPCAResult,
) -> Array2<Complex64> {
    let mut centered = Array2::<Complex64>::zeros((matrix.nrows(), matrix.ncols()));
    for row in 0..matrix.nrows() {
        for col in 0..matrix.ncols() {
            centered[[row, col]] = matrix[[row, col]] - pca.mean[col];
        }
    }

    let projection = pca.components.t().mapv(|value| value.conj());
    centered.dot(&projection)
}

fn inverse_transform_complex_impl(
    scores: &ArrayView2<'_, Complex64>,
    pca: &NdarrayComplexPCAResult,
) -> Array2<Complex64> {
    let mut reconstructed = scores.dot(&pca.components);
    for row in 0..reconstructed.nrows() {
        for col in 0..reconstructed.ncols() {
            reconstructed[[row, col]] += pca.mean[col];
        }
    }
    reconstructed
}

/// Compute principal component analysis.
///
/// # Errors
/// Returns an error for invalid input or decomposition failure.
#[cfg(feature = "lapack-provider")]
pub fn compute_pca<T>(
    matrix: &Array2<T>,
    n_components: Option<usize>,
) -> Result<NdarrayPCAResult<T>, PCAError>
where
    T: NabledReal + ndarray_linalg::Lapack<Real = T>,
{
    compute_pca_impl(&matrix.view(), n_components)
}

/// Compute principal component analysis.
///
/// # Errors
/// Returns an error for invalid input or decomposition failure.
#[cfg(not(feature = "lapack-provider"))]
pub fn compute_pca<T: NabledReal>(
    matrix: &Array2<T>,
    n_components: Option<usize>,
) -> Result<NdarrayPCAResult<T>, PCAError> {
    compute_pca_impl(&matrix.view(), n_components)
}

#[cfg(feature = "lapack-provider")]
fn compute_pca_impl<T>(
    matrix: &ArrayView2<'_, T>,
    n_components: Option<usize>,
) -> Result<NdarrayPCAResult<T>, PCAError>
where
    T: NabledReal + ndarray_linalg::Lapack<Real = T>,
{
    let (centered, mean) = center_columns(matrix)?;
    let svd = svd::decompose(&centered).map_err(|_| PCAError::DecompositionFailed)?;

    let max_components = centered.nrows().min(centered.ncols());
    let keep = n_components.unwrap_or(max_components).min(max_components);
    if keep == 0 {
        return Err(PCAError::InvalidInput("n_components must be greater than 0".to_string()));
    }

    let components = svd.vt.slice(s![..keep, ..]).to_owned();
    let scores = centered.dot(&components.t());

    let one = T::one();
    let denominator = (usize_to_real::<T>(centered.nrows()) - one).max(one);
    let mut explained_variance = Array1::<T>::zeros(keep);
    for i in 0..keep {
        explained_variance[i] = (svd.singular_values[i] * svd.singular_values[i]) / denominator;
    }

    let total_variance = explained_variance
        .iter()
        .copied()
        .fold(T::zero(), |acc, value| acc + value)
        .max(T::epsilon());
    let explained_variance_ratio = explained_variance.map(|value| *value / total_variance);

    Ok(NdarrayPCAResult { components, explained_variance, explained_variance_ratio, mean, scores })
}

#[cfg(not(feature = "lapack-provider"))]
fn compute_pca_impl<T: NabledReal>(
    matrix: &ArrayView2<'_, T>,
    n_components: Option<usize>,
) -> Result<NdarrayPCAResult<T>, PCAError> {
    let (centered, mean) = center_columns(matrix)?;
    let svd = svd::decompose(&centered).map_err(|_| PCAError::DecompositionFailed)?;

    let max_components = centered.nrows().min(centered.ncols());
    let keep = n_components.unwrap_or(max_components).min(max_components);
    if keep == 0 {
        return Err(PCAError::InvalidInput("n_components must be greater than 0".to_string()));
    }

    let components = svd.vt.slice(s![..keep, ..]).to_owned();
    let scores = centered.dot(&components.t());

    let one = T::one();
    let denominator = (usize_to_real::<T>(centered.nrows()) - one).max(one);
    let mut explained_variance = Array1::<T>::zeros(keep);
    for i in 0..keep {
        explained_variance[i] = (svd.singular_values[i] * svd.singular_values[i]) / denominator;
    }

    let total_variance = explained_variance
        .iter()
        .copied()
        .fold(T::zero(), |acc, value| acc + value)
        .max(T::epsilon());
    let explained_variance_ratio = explained_variance.map(|value| *value / total_variance);

    Ok(NdarrayPCAResult { components, explained_variance, explained_variance_ratio, mean, scores })
}

/// Compute principal component analysis from a matrix view.
///
/// # Errors
/// Returns an error for invalid input or decomposition failure.
#[cfg(feature = "lapack-provider")]
pub fn compute_pca_view<T>(
    matrix: &ArrayView2<'_, T>,
    n_components: Option<usize>,
) -> Result<NdarrayPCAResult<T>, PCAError>
where
    T: NabledReal + ndarray_linalg::Lapack<Real = T>,
{
    compute_pca_impl(matrix, n_components)
}

/// Compute principal component analysis from a matrix view.
///
/// # Errors
/// Returns an error for invalid input or decomposition failure.
#[cfg(not(feature = "lapack-provider"))]
pub fn compute_pca_view<T: NabledReal>(
    matrix: &ArrayView2<'_, T>,
    n_components: Option<usize>,
) -> Result<NdarrayPCAResult<T>, PCAError> {
    compute_pca_impl(matrix, n_components)
}

/// Compute principal component analysis for complex matrices.
///
/// # Errors
/// Returns an error for invalid input or decomposition failure.
pub fn compute_pca_complex(
    matrix: &Array2<Complex64>,
    n_components: Option<usize>,
) -> Result<NdarrayComplexPCAResult, PCAError> {
    compute_pca_complex_impl(&matrix.view(), n_components)
}

fn compute_pca_complex_impl(
    matrix: &ArrayView2<'_, Complex64>,
    n_components: Option<usize>,
) -> Result<NdarrayComplexPCAResult, PCAError> {
    let (centered, mean) = center_columns_complex(matrix)?;
    let svd = svd::decompose_complex(&centered).map_err(|_| PCAError::DecompositionFailed)?;

    let max_components = centered.nrows().min(centered.ncols());
    let keep = n_components.unwrap_or(max_components).min(max_components);
    if keep == 0 {
        return Err(PCAError::InvalidInput("n_components must be greater than 0".to_string()));
    }

    let components = svd.vt.slice(s![..keep, ..]).to_owned();
    let projection = components.t().mapv(|value| value.conj());
    let scores = centered.dot(&projection);

    let denominator = (usize_to_real::<f64>(centered.nrows()) - 1.0_f64).max(1.0_f64);
    let mut explained_variance = Array1::<f64>::zeros(keep);
    for i in 0..keep {
        explained_variance[i] = (svd.singular_values[i] * svd.singular_values[i]) / denominator;
    }

    let total_variance = explained_variance.iter().sum::<f64>().max(f64::EPSILON);
    let explained_variance_ratio = explained_variance.map(|value| *value / total_variance);

    Ok(NdarrayComplexPCAResult {
        components,
        explained_variance,
        explained_variance_ratio,
        mean,
        scores,
    })
}

/// Compute principal component analysis for complex matrices from a matrix view.
///
/// # Errors
/// Returns an error for invalid input or decomposition failure.
pub fn compute_pca_complex_view(
    matrix: &ArrayView2<'_, Complex64>,
    n_components: Option<usize>,
) -> Result<NdarrayComplexPCAResult, PCAError> {
    compute_pca_complex_impl(matrix, n_components)
}

/// Project data to PCA score space.
#[must_use]
pub fn transform<T: NabledReal>(matrix: &Array2<T>, pca: &NdarrayPCAResult<T>) -> Array2<T> {
    transform_impl(&matrix.view(), pca)
}

/// Project data to PCA score space from a matrix view.
#[must_use]
pub fn transform_view<T: NabledReal>(
    matrix: &ArrayView2<'_, T>,
    pca: &NdarrayPCAResult<T>,
) -> Array2<T> {
    transform_impl(matrix, pca)
}

/// Reconstruct from PCA scores.
#[must_use]
pub fn inverse_transform<T: NabledReal>(
    scores: &Array2<T>,
    pca: &NdarrayPCAResult<T>,
) -> Array2<T> {
    inverse_transform_impl(&scores.view(), pca)
}

/// Reconstruct from PCA scores provided as a matrix view.
#[must_use]
pub fn inverse_transform_view<T: NabledReal>(
    scores: &ArrayView2<'_, T>,
    pca: &NdarrayPCAResult<T>,
) -> Array2<T> {
    inverse_transform_impl(scores, pca)
}

/// Project complex data to PCA score space.
#[must_use]
pub fn transform_complex(
    matrix: &Array2<Complex64>,
    pca: &NdarrayComplexPCAResult,
) -> Array2<Complex64> {
    transform_complex_impl(&matrix.view(), pca)
}

/// Project complex data to PCA score space from a matrix view.
#[must_use]
pub fn transform_complex_view(
    matrix: &ArrayView2<'_, Complex64>,
    pca: &NdarrayComplexPCAResult,
) -> Array2<Complex64> {
    transform_complex_impl(matrix, pca)
}

/// Reconstruct complex inputs from PCA scores.
#[must_use]
pub fn inverse_transform_complex(
    scores: &Array2<Complex64>,
    pca: &NdarrayComplexPCAResult,
) -> Array2<Complex64> {
    inverse_transform_complex_impl(&scores.view(), pca)
}

/// Reconstruct complex inputs from PCA scores provided as a matrix view.
#[must_use]
pub fn inverse_transform_complex_view(
    scores: &ArrayView2<'_, Complex64>,
    pca: &NdarrayComplexPCAResult,
) -> Array2<Complex64> {
    inverse_transform_complex_impl(scores, pca)
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;
    use num_complex::Complex64;

    use super::*;

    #[test]
    fn pca_roundtrip_is_consistent() {
        let matrix = Array2::<f64>::from_shape_vec((4, 2), vec![
            1.0_f64, 2.0_f64, 2.0_f64, 1.0_f64, 3.0_f64, 4.0_f64, 4.0_f64, 3.0_f64,
        ])
        .unwrap();
        let pca = compute_pca(&matrix, Some(2)).unwrap();
        let transformed = transform(&matrix, &pca);
        let reconstructed = inverse_transform(&transformed, &pca);
        for i in 0..matrix.nrows() {
            for j in 0..matrix.ncols() {
                assert!((matrix[[i, j]] - reconstructed[[i, j]]).abs() < 1e-8_f64);
            }
        }
    }

    #[test]
    fn pca_rejects_zero_components() {
        let matrix = Array2::<f64>::from_shape_vec((4, 2), vec![
            1.0_f64, 2.0_f64, 2.0_f64, 1.0_f64, 3.0_f64, 4.0_f64, 4.0_f64, 3.0_f64,
        ])
        .unwrap();
        let result = compute_pca(&matrix, Some(0));
        assert!(matches!(result, Err(PCAError::InvalidInput(_))));
    }

    #[test]
    fn explained_variance_ratio_sums_to_one() {
        let matrix = Array2::<f64>::from_shape_vec((4, 2), vec![
            1.0_f64, 2.0_f64, 2.0_f64, 1.0_f64, 3.0_f64, 4.0_f64, 4.0_f64, 3.0_f64,
        ])
        .unwrap();
        let pca = compute_pca(&matrix, Some(2)).unwrap();
        let sum = pca.explained_variance_ratio.iter().sum::<f64>();
        assert!((sum - 1.0_f64).abs() < 1e-10_f64);
    }

    #[test]
    fn pca_view_variants_match_owned() {
        let matrix = Array2::<f64>::from_shape_vec((4, 2), vec![
            1.0_f64, 2.0_f64, 2.0_f64, 1.0_f64, 3.0_f64, 4.0_f64, 4.0_f64, 3.0_f64,
        ])
        .unwrap();
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
                assert!((transformed_owned[[i, j]] - transformed_view[[i, j]]).abs() < 1e-12_f64);
                assert!(
                    (reconstructed_owned[[i, j]] - reconstructed_view[[i, j]]).abs() < 1e-12_f64
                );
            }
        }
    }

    #[test]
    fn pca_real_f32_paths_are_consistent() {
        let matrix = Array2::<f32>::from_shape_vec((4, 2), vec![
            1.0_f32, 2.0_f32, 2.0_f32, 1.0_f32, 3.0_f32, 4.0_f32, 4.0_f32, 3.0_f32,
        ])
        .unwrap();
        let pca = compute_pca(&matrix, Some(2)).unwrap();
        let transformed = transform(&matrix, &pca);
        let reconstructed = inverse_transform(&transformed, &pca);

        assert_eq!(pca.components.dim(), (2, 2));
        assert_eq!(pca.explained_variance.len(), 2);
        assert_eq!(pca.explained_variance_ratio.len(), 2);
        for i in 0..matrix.nrows() {
            for j in 0..matrix.ncols() {
                assert!((matrix[[i, j]] - reconstructed[[i, j]]).abs() < 1e-4_f32);
            }
        }
    }

    #[test]
    fn complex_pca_roundtrip_is_consistent() {
        let matrix = Array2::from_shape_vec((4, 2), vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.5),
            Complex64::new(2.0, -1.0),
            Complex64::new(1.0, 0.2),
            Complex64::new(3.0, 1.1),
            Complex64::new(4.0, -0.3),
            Complex64::new(4.0, 0.9),
            Complex64::new(3.0, 0.4),
        ])
        .unwrap();

        let pca = compute_pca_complex(&matrix, Some(2)).unwrap();
        let transformed = transform_complex(&matrix, &pca);
        let reconstructed = inverse_transform_complex(&transformed, &pca);
        for i in 0..matrix.nrows() {
            for j in 0..matrix.ncols() {
                assert!((matrix[[i, j]] - reconstructed[[i, j]]).norm() < 1e-8);
            }
        }
    }

    #[test]
    fn complex_pca_view_variants_match_owned() {
        let matrix = Array2::from_shape_vec((4, 2), vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.5),
            Complex64::new(2.0, -1.0),
            Complex64::new(1.0, 0.2),
            Complex64::new(3.0, 1.1),
            Complex64::new(4.0, -0.3),
            Complex64::new(4.0, 0.9),
            Complex64::new(3.0, 0.4),
        ])
        .unwrap();

        let pca_owned = compute_pca_complex(&matrix, Some(2)).unwrap();
        let pca_view = compute_pca_complex_view(&matrix.view(), Some(2)).unwrap();
        assert_eq!(pca_owned.components.dim(), pca_view.components.dim());
        assert_eq!(pca_owned.scores.dim(), pca_view.scores.dim());

        let transformed_owned = transform_complex(&matrix, &pca_owned);
        let transformed_view = transform_complex_view(&matrix.view(), &pca_owned);
        let reconstructed_owned = inverse_transform_complex(&transformed_owned, &pca_owned);
        let reconstructed_view =
            inverse_transform_complex_view(&transformed_owned.view(), &pca_owned);

        for i in 0..matrix.nrows() {
            for j in 0..matrix.ncols() {
                assert!((transformed_owned[[i, j]] - transformed_view[[i, j]]).norm() < 1e-12);
                assert!((reconstructed_owned[[i, j]] - reconstructed_view[[i, j]]).norm() < 1e-12);
            }
        }
    }
}
