//! # Statistical Functions
//!
//! Covariance and correlation matrices for numerical data.

use std::fmt;

use nalgebra::{DMatrix, DVector, RealField};
use ndarray::{Array1, Array2};
use num_traits::Float;

/// Error types for stats operations
#[derive(Debug, Clone, PartialEq)]
pub enum StatsError {
    /// Matrix is empty
    EmptyMatrix,
    /// Not enough samples (need at least 2 for covariance)
    InsufficientSamples,
    /// Invalid input
    InvalidInput(String),
}

impl fmt::Display for StatsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StatsError::EmptyMatrix => write!(f, "Matrix is empty"),
            StatsError::InsufficientSamples => write!(f, "Need at least 2 samples for covariance"),
            StatsError::InvalidInput(msg) => write!(f, "Invalid input: {msg}"),
        }
    }
}

impl std::error::Error for StatsError {}

/// Nalgebra stats functions
pub mod nalgebra_stats {
    use super::*;

    /// Compute column means
    /// # Panics
    /// Panics if internal numeric assumptions are violated during setup or
    /// intermediate conversion steps.
    #[must_use]
    pub fn column_means<T: RealField + Copy + num_traits::NumCast>(
        matrix: &DMatrix<T>,
    ) -> DVector<T> {
        let (rows, cols) = matrix.shape();
        let n: T = num_traits::NumCast::from(rows).expect("rows must fit in T");
        let mut means = DVector::zeros(cols);
        for j in 0..cols {
            let mut sum = T::zero();
            for i in 0..rows {
                sum += matrix[(i, j)];
            }
            means[j] = sum / n;
        }
        means
    }

    /// Center columns (subtract mean from each column)
    #[must_use]
    pub fn center_columns<T: RealField + Copy + num_traits::NumCast>(
        matrix: &DMatrix<T>,
    ) -> DMatrix<T> {
        let means = column_means(matrix);
        let mut centered = matrix.clone();
        let (rows, cols) = matrix.shape();
        for j in 0..cols {
            for i in 0..rows {
                centered[(i, j)] -= means[j];
            }
        }
        centered
    }

    /// Compute sample covariance matrix (Bessel correction, n-1)
    /// # Panics
    /// Panics if internal numeric assumptions are violated during setup or
    /// intermediate conversion steps.
    ///
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn covariance_matrix<T: RealField + Copy + Float + num_traits::NumCast>(
        matrix: &DMatrix<T>,
    ) -> Result<DMatrix<T>, StatsError> {
        if matrix.is_empty() {
            return Err(StatsError::EmptyMatrix);
        }
        let (rows, _cols) = matrix.shape();
        if rows < 2 {
            return Err(StatsError::InsufficientSamples);
        }

        let centered = center_columns(matrix);
        let n_minus_1: T = num_traits::NumCast::from(rows - 1).expect("n-1 must fit in T");
        let cov = centered.transpose() * &centered / n_minus_1;
        Ok(cov)
    }

    /// Compute correlation matrix
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn correlation_matrix<T: RealField + Copy + Float + num_traits::NumCast>(
        matrix: &DMatrix<T>,
    ) -> Result<DMatrix<T>, StatsError> {
        let cov = covariance_matrix(matrix)?;
        let (rows, cols) = cov.shape();
        let mut corr = cov.clone();
        for i in 0..rows {
            for j in 0..cols {
                let si = nalgebra::ComplexField::sqrt(cov[(i, i)]);
                let sj = nalgebra::ComplexField::sqrt(cov[(j, j)]);
                if si > T::zero() && sj > T::zero() {
                    corr[(i, j)] = cov[(i, j)] / (si * sj);
                } else {
                    corr[(i, j)] = T::nan();
                }
            }
        }
        Ok(corr)
    }
}

/// Ndarray stats functions
pub mod ndarray_stats {
    use super::*;
    use crate::interop::{nalgebra_to_ndarray, ndarray_to_nalgebra};

    /// Compute column means
    #[must_use]
    pub fn column_means<T: Float + RealField>(matrix: &Array2<T>) -> Array1<T> {
        let nalg = ndarray_to_nalgebra(matrix);
        let means = nalgebra_stats::column_means(&nalg);
        Array1::from_vec(means.as_slice().to_vec())
    }

    /// Center columns
    #[must_use]
    pub fn center_columns<T: Float + RealField>(matrix: &Array2<T>) -> Array2<T> {
        let nalg = ndarray_to_nalgebra(matrix);
        let centered = nalgebra_stats::center_columns(&nalg);
        nalgebra_to_ndarray(&centered)
    }

    /// Compute sample covariance matrix (Bessel correction)
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn covariance_matrix<T: Float + RealField>(
        matrix: &Array2<T>,
    ) -> Result<Array2<T>, StatsError> {
        let nalg = ndarray_to_nalgebra(matrix);
        let cov = nalgebra_stats::covariance_matrix(&nalg)?;
        Ok(nalgebra_to_ndarray(&cov))
    }

    /// Compute correlation matrix
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn correlation_matrix<T: Float + RealField>(
        matrix: &Array2<T>,
    ) -> Result<Array2<T>, StatsError> {
        let nalg = ndarray_to_nalgebra(matrix);
        let corr = nalgebra_stats::correlation_matrix(&nalg)?;
        Ok(nalgebra_to_ndarray(&corr))
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn test_covariance_bessel() {
        let m = DMatrix::from_row_slice(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let cov = nalgebra_stats::covariance_matrix(&m).unwrap();
        assert_relative_eq!(cov[(0, 0)], 4.0, epsilon = 1e-10);
        assert_relative_eq!(cov[(1, 1)], 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_correlation_bounds() {
        let m = DMatrix::from_row_slice(10, 2, &[
            1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0,
            9.0, 10.0, 10.0,
        ]);
        let corr = nalgebra_stats::correlation_matrix(&m).unwrap();
        assert_relative_eq!(corr[(0, 1)], 1.0, epsilon = 1e-10);
        assert_relative_eq!(corr[(0, 0)], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_column_means_and_center_columns() {
        let m = DMatrix::from_row_slice(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let means = nalgebra_stats::column_means(&m);
        assert_relative_eq!(means[0], 3.0, epsilon = 1e-12);
        assert_relative_eq!(means[1], 4.0, epsilon = 1e-12);

        let centered = nalgebra_stats::center_columns(&m);
        let centered_means = nalgebra_stats::column_means(&centered);
        assert_relative_eq!(centered_means[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(centered_means[1], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_ndarray_stats_wrappers() {
        let m = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let means = ndarray_stats::column_means(&m);
        assert_relative_eq!(means[0], 3.0, epsilon = 1e-12);
        assert_relative_eq!(means[1], 4.0, epsilon = 1e-12);

        let centered = ndarray_stats::center_columns(&m);
        let centered_means = ndarray_stats::column_means(&centered);
        assert_relative_eq!(centered_means[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(centered_means[1], 0.0, epsilon = 1e-12);

        let cov = ndarray_stats::covariance_matrix(&m).unwrap();
        assert_relative_eq!(cov[[0, 0]], 4.0, epsilon = 1e-12);
        assert_relative_eq!(cov[[1, 1]], 4.0, epsilon = 1e-12);
    }

    #[test]
    fn test_stats_error_display_variants() {
        assert!(format!("{}", StatsError::EmptyMatrix).contains("empty"));
        assert!(format!("{}", StatsError::InsufficientSamples).contains("at least 2"));
        assert!(format!("{}", StatsError::InvalidInput("x".to_string())).contains('x'));
    }

    #[test]
    fn test_stats_error_paths() {
        let empty = DMatrix::<f64>::zeros(0, 0);
        assert!(matches!(nalgebra_stats::covariance_matrix(&empty), Err(StatsError::EmptyMatrix)));

        let one_row = DMatrix::from_row_slice(1, 2, &[1.0, 2.0]);
        assert!(matches!(
            nalgebra_stats::covariance_matrix(&one_row),
            Err(StatsError::InsufficientSamples)
        ));

        let with_constant_column =
            DMatrix::<f64>::from_row_slice(3, 2, &[1.0, 7.0, 2.0, 7.0, 3.0, 7.0]);
        let corr: DMatrix<f64> = nalgebra_stats::correlation_matrix(&with_constant_column).unwrap();
        assert!(corr[(0, 1)].is_nan());
        assert!(corr[(1, 0)].is_nan());

        let with_constant_column_nd =
            Array2::<f64>::from_shape_vec((3, 2), vec![1.0, 7.0, 2.0, 7.0, 3.0, 7.0]).unwrap();
        let corr_nd: Array2<f64> =
            ndarray_stats::correlation_matrix(&with_constant_column_nd).unwrap();
        assert!(corr_nd[[0, 1]].is_nan());
    }
}
