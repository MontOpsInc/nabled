//! # Principal Component Analysis (PCA)
//!
//! PCA via SVD of the centered data matrix.

use std::fmt;

use nalgebra::{DMatrix, DVector, RealField};
use ndarray::{Array1, Array2};
use num_traits::Float;

/// Error types for PCA
#[derive(Debug, Clone, PartialEq)]
pub enum PCAError {
    /// Matrix is empty
    EmptyMatrix,
    /// Insufficient samples
    InsufficientSamples,
    /// `n_components` too large
    InvalidComponents,
    /// SVD or stats error
    Computation(String),
}

impl fmt::Display for PCAError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PCAError::EmptyMatrix => write!(f, "Matrix is empty"),
            PCAError::InsufficientSamples => write!(f, "Insufficient samples"),
            PCAError::InvalidComponents => write!(f, "Invalid number of components"),
            PCAError::Computation(msg) => write!(f, "Computation error: {msg}"),
        }
    }
}

impl std::error::Error for PCAError {}

/// PCA result for nalgebra
#[derive(Debug, Clone)]
pub struct NalgebraPCAResult<T: RealField> {
    /// Principal components (rows = components, cols = features)
    pub components:               DMatrix<T>,
    /// Projected data (scores)
    pub scores:                   DMatrix<T>,
    /// Explained variance (variance of each PC)
    pub explained_variance:       DVector<T>,
    /// Explained variance ratio (fraction of total variance)
    pub explained_variance_ratio: DVector<T>,
    /// Column means of original data
    pub mean:                     DVector<T>,
}

/// PCA result for ndarray
#[derive(Debug, Clone)]
pub struct NdarrayPCAResult<T: Float> {
    /// Principal components (rows = components, cols = features)
    pub components:               Array2<T>,
    /// Projected data (scores)
    pub scores:                   Array2<T>,
    /// Explained variance
    pub explained_variance:       Array1<T>,
    /// Explained variance ratio
    pub explained_variance_ratio: Array1<T>,
    /// Column means
    pub mean:                     Array1<T>,
}

/// Nalgebra PCA
pub mod nalgebra_pca {
    use super::*;

    /// Compute PCA
    /// # Panics
    /// Panics if internal numeric assumptions are violated during setup or
    /// intermediate conversion steps.
    ///
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn compute_pca<T: RealField + Copy + Float + num_traits::NumCast>(
        matrix: &DMatrix<T>,
        n_components: Option<usize>,
    ) -> Result<NalgebraPCAResult<T>, PCAError> {
        crate::backend::pca::compute_nalgebra_pca(matrix, n_components)
    }

    /// Compute PCA with a LAPACK-backed kernel.
    ///
    /// This path is available on Linux when the `lapack-kernels` feature is enabled.
    ///
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    pub fn compute_pca_lapack(
        matrix: &DMatrix<f64>,
        n_components: Option<usize>,
    ) -> Result<NalgebraPCAResult<f64>, PCAError> {
        crate::backend::pca::compute_nalgebra_lapack_pca(matrix, n_components)
    }

    /// Transform new data using fitted PCA
    #[must_use]
    pub fn transform<T: RealField + Copy>(
        matrix: &DMatrix<T>,
        pca_result: &NalgebraPCAResult<T>,
    ) -> DMatrix<T> {
        let mut centered = matrix.clone();
        let (rows, cols) = matrix.shape();
        for j in 0..cols {
            for i in 0..rows {
                centered[(i, j)] -= pca_result.mean[j];
            }
        }
        &centered * &pca_result.components
    }

    /// Inverse transform from scores to original space
    #[must_use]
    pub fn inverse_transform<T: RealField + Copy>(
        scores: &DMatrix<T>,
        pca_result: &NalgebraPCAResult<T>,
    ) -> DMatrix<T> {
        let mut out = scores * pca_result.components.transpose();
        let (rows, cols) = out.shape();
        for j in 0..cols {
            for i in 0..rows {
                out[(i, j)] += pca_result.mean[j];
            }
        }
        out
    }
}

/// Ndarray PCA
pub mod ndarray_pca {
    use super::*;
    use crate::interop::{nalgebra_to_ndarray, ndarray_to_nalgebra};

    /// Compute PCA
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn compute_pca<T: Float + RealField + num_traits::NumCast>(
        matrix: &Array2<T>,
        n_components: Option<usize>,
    ) -> Result<NdarrayPCAResult<T>, PCAError> {
        crate::backend::pca::compute_ndarray_pca(matrix, n_components)
    }

    /// Compute PCA with a LAPACK-backed kernel.
    ///
    /// This path is available on Linux when the `lapack-kernels` feature is enabled.
    ///
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    pub fn compute_pca_lapack(
        matrix: &Array2<f64>,
        n_components: Option<usize>,
    ) -> Result<NdarrayPCAResult<f64>, PCAError> {
        crate::backend::pca::compute_ndarray_lapack_pca(matrix, n_components)
    }

    /// Transform new data
    #[must_use]
    pub fn transform<T: Float + RealField>(
        matrix: &Array2<T>,
        pca_result: &NdarrayPCAResult<T>,
    ) -> Array2<T> {
        let nalg_matrix = ndarray_to_nalgebra(matrix);
        let nalg_result = NalgebraPCAResult {
            components:               ndarray_to_nalgebra(&pca_result.components),
            scores:                   ndarray_to_nalgebra(&pca_result.scores),
            explained_variance:       DVector::from_vec(pca_result.explained_variance.to_vec()),
            explained_variance_ratio: DVector::from_vec(
                pca_result.explained_variance_ratio.to_vec(),
            ),
            mean:                     DVector::from_vec(pca_result.mean.to_vec()),
        };
        let transformed = nalgebra_pca::transform(&nalg_matrix, &nalg_result);
        nalgebra_to_ndarray(&transformed)
    }

    /// Inverse transform
    #[must_use]
    pub fn inverse_transform<T: Float + RealField>(
        scores: &Array2<T>,
        pca_result: &NdarrayPCAResult<T>,
    ) -> Array2<T> {
        let nalg_scores = ndarray_to_nalgebra(scores);
        let nalg_result = NalgebraPCAResult {
            components:               ndarray_to_nalgebra(&pca_result.components),
            scores:                   ndarray_to_nalgebra(&pca_result.scores),
            explained_variance:       DVector::from_vec(pca_result.explained_variance.to_vec()),
            explained_variance_ratio: DVector::from_vec(
                pca_result.explained_variance_ratio.to_vec(),
            ),
            mean:                     DVector::from_vec(pca_result.mean.to_vec()),
        };
        let reconstructed = nalgebra_pca::inverse_transform(&nalg_scores, &nalg_result);
        nalgebra_to_ndarray(&reconstructed)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn test_pca_roundtrip() {
        let m = DMatrix::from_row_slice(4, 3, &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ]);
        let pca = nalgebra_pca::compute_pca(&m, Some(3)).unwrap();
        let reconstructed = nalgebra_pca::inverse_transform(&pca.scores, &pca);
        for i in 0..4 {
            for j in 0..3 {
                assert_relative_eq!(reconstructed[(i, j)], m[(i, j)], epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_explained_variance_ratio_sums_to_one() {
        let m = DMatrix::from_row_slice(5, 3, &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        ]);
        let pca = nalgebra_pca::compute_pca(&m, Some(3)).unwrap();
        let sum: f64 = pca.explained_variance_ratio.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 0.01);
    }

    #[test]
    fn test_ndarray_pca_roundtrip() {
        let m = Array2::from_shape_vec((5, 3), vec![
            1.0, 2.0, 1.5, 2.0, 1.0, 3.5, 3.0, 4.0, 2.0, 4.0, 3.0, 5.0, 5.0, 5.0, 4.5,
        ])
        .unwrap();

        let pca = ndarray_pca::compute_pca(&m, Some(3)).unwrap();
        let transformed = ndarray_pca::transform(&m, &pca);
        let reconstructed = ndarray_pca::inverse_transform(&transformed, &pca);

        for i in 0..m.nrows() {
            for j in 0..m.ncols() {
                assert_relative_eq!(reconstructed[[i, j]], m[[i, j]], epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_ndarray_pca_explained_variance_ratio_sums_to_one() {
        let m = Array2::from_shape_vec((5, 3), vec![
            1.0, 2.0, 1.5, 2.0, 1.0, 3.5, 3.0, 4.0, 2.0, 4.0, 3.0, 5.0, 5.0, 5.0, 4.5,
        ])
        .unwrap();

        let pca = ndarray_pca::compute_pca(&m, Some(3)).unwrap();
        let sum: f64 = pca.explained_variance_ratio.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-8);
    }

    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    #[test]
    fn test_nalgebra_pca_lapack_basic() {
        let m = DMatrix::from_row_slice(4, 3, &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ]);
        let result = nalgebra_pca::compute_pca_lapack(&m, Some(2)).unwrap();
        assert_eq!(result.components.shape(), (3, 2));
        assert_eq!(result.scores.shape(), (4, 2));
    }

    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    #[test]
    fn test_ndarray_pca_lapack_basic() {
        let m = Array2::from_shape_vec((4, 3), vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ])
        .unwrap();
        let result = ndarray_pca::compute_pca_lapack(&m, Some(2)).unwrap();
        assert_eq!(result.components.dim(), (3, 2));
        assert_eq!(result.scores.dim(), (4, 2));
    }
}
