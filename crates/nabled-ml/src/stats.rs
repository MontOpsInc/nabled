//! Statistical utilities over ndarray matrices.

use std::fmt;

use ndarray::{Array1, Array2, ArrayView2, Axis};

/// Error type for matrix statistics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StatsError {
    /// Matrix is empty.
    EmptyMatrix,
    /// Matrix needs at least two rows.
    InsufficientSamples,
    /// Numerical instability detected.
    NumericalInstability,
}

impl fmt::Display for StatsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StatsError::EmptyMatrix => write!(f, "Matrix cannot be empty"),
            StatsError::InsufficientSamples => {
                write!(f, "At least two observations are required")
            }
            StatsError::NumericalInstability => write!(f, "Numerical instability detected"),
        }
    }
}

impl std::error::Error for StatsError {}

fn usize_to_f64(value: usize) -> f64 { u32::try_from(value).map_or(f64::from(u32::MAX), f64::from) }

/// Ndarray statistics functions.
pub mod ndarray_stats {
    use super::*;

    /// Compute column means.
    #[must_use]
    pub fn column_means(matrix: &Array2<f64>) -> Array1<f64> {
        matrix.mean_axis(Axis(0)).unwrap_or_else(|| Array1::zeros(matrix.ncols()))
    }

    /// Compute column means from a matrix view.
    #[must_use]
    pub fn column_means_view(matrix: &ArrayView2<'_, f64>) -> Array1<f64> {
        column_means(&matrix.to_owned())
    }

    /// Center columns by subtracting their means.
    #[must_use]
    pub fn center_columns(matrix: &Array2<f64>) -> Array2<f64> {
        let means = column_means(matrix);
        let mut centered = matrix.clone();
        for row in 0..matrix.nrows() {
            for col in 0..matrix.ncols() {
                centered[[row, col]] -= means[col];
            }
        }
        centered
    }

    /// Center columns by subtracting their means from a matrix view.
    #[must_use]
    pub fn center_columns_view(matrix: &ArrayView2<'_, f64>) -> Array2<f64> {
        center_columns(&matrix.to_owned())
    }

    /// Compute sample covariance matrix.
    ///
    /// # Errors
    /// Returns an error for empty input or fewer than two samples.
    pub fn covariance_matrix(matrix: &Array2<f64>) -> Result<Array2<f64>, StatsError> {
        if matrix.is_empty() {
            return Err(StatsError::EmptyMatrix);
        }
        if matrix.nrows() < 2 {
            return Err(StatsError::InsufficientSamples);
        }

        let centered = center_columns(matrix);
        let covariance = centered.t().dot(&centered) / usize_to_f64(matrix.nrows() - 1);

        if covariance.iter().any(|value| !value.is_finite()) {
            return Err(StatsError::NumericalInstability);
        }

        Ok(covariance)
    }

    /// Compute sample covariance matrix from a matrix view.
    ///
    /// # Errors
    /// Returns an error for empty input or fewer than two samples.
    pub fn covariance_matrix_view(matrix: &ArrayView2<'_, f64>) -> Result<Array2<f64>, StatsError> {
        covariance_matrix(&matrix.to_owned())
    }

    /// Compute correlation matrix.
    ///
    /// # Errors
    /// Returns an error if covariance computation fails.
    pub fn correlation_matrix(matrix: &Array2<f64>) -> Result<Array2<f64>, StatsError> {
        let covariance = covariance_matrix(matrix)?;
        let n = covariance.nrows();
        let mut correlation = Array2::<f64>::zeros((n, n));

        for i in 0..n {
            let sigma_i = covariance[[i, i]].sqrt();
            for j in 0..n {
                let sigma_j = covariance[[j, j]].sqrt();
                let denom = (sigma_i * sigma_j).max(f64::EPSILON);
                correlation[[i, j]] = covariance[[i, j]] / denom;
            }
        }

        Ok(correlation)
    }

    /// Compute correlation matrix from a matrix view.
    ///
    /// # Errors
    /// Returns an error if covariance computation fails.
    pub fn correlation_matrix_view(
        matrix: &ArrayView2<'_, f64>,
    ) -> Result<Array2<f64>, StatsError> {
        correlation_matrix(&matrix.to_owned())
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use super::{StatsError, ndarray_stats};

    #[test]
    fn covariance_and_correlation_are_well_formed() {
        let matrix =
            Array2::from_shape_vec((4, 2), vec![1.0, 3.0, 2.0, 2.0, 3.0, 1.0, 4.0, 0.0]).unwrap();
        let covariance = ndarray_stats::covariance_matrix(&matrix).unwrap();
        let correlation = ndarray_stats::correlation_matrix(&matrix).unwrap();
        assert_eq!(covariance.dim(), (2, 2));
        assert_eq!(correlation.dim(), (2, 2));
    }

    #[test]
    fn stats_rejects_empty_and_insufficient_inputs() {
        let empty = Array2::<f64>::zeros((0, 0));
        assert!(matches!(ndarray_stats::covariance_matrix(&empty), Err(StatsError::EmptyMatrix)));

        let one_row = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
        assert!(matches!(
            ndarray_stats::covariance_matrix(&one_row),
            Err(StatsError::InsufficientSamples)
        ));
    }

    #[test]
    fn center_columns_zeroes_means() {
        let matrix = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0]).unwrap();
        let centered = ndarray_stats::center_columns(&matrix);
        let means = ndarray_stats::column_means(&centered);
        assert!(means.iter().all(|value| value.abs() < 1e-12));
    }

    #[test]
    fn column_means_handles_empty_input() {
        let matrix = Array2::<f64>::zeros((0, 3));
        let means = ndarray_stats::column_means(&matrix);
        assert_eq!(means.len(), 3);
        assert!(means.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn covariance_reports_numerical_instability() {
        let matrix = Array2::from_shape_vec((2, 2), vec![f64::MAX, 0.0, -f64::MAX, 0.0]).unwrap();
        let result = ndarray_stats::covariance_matrix(&matrix);
        assert!(matches!(result, Err(StatsError::NumericalInstability)));
    }

    #[test]
    fn correlation_handles_zero_variance_column() {
        let matrix = Array2::from_shape_vec((3, 2), vec![1.0, 10.0, 1.0, 20.0, 1.0, 30.0]).unwrap();
        let correlation = ndarray_stats::correlation_matrix(&matrix).unwrap();
        assert!(correlation[[0, 0]].is_finite());
        assert!(correlation[[0, 1]].is_finite());
        assert!(correlation[[1, 0]].is_finite());
        assert!(correlation[[1, 1]].is_finite());
    }

    #[test]
    fn view_variants_match_owned() {
        let matrix =
            Array2::from_shape_vec((4, 2), vec![1.0, 3.0, 2.0, 2.0, 3.0, 1.0, 4.0, 0.0]).unwrap();
        let means_owned = ndarray_stats::column_means(&matrix);
        let means_view = ndarray_stats::column_means_view(&matrix.view());
        let centered_owned = ndarray_stats::center_columns(&matrix);
        let centered_view = ndarray_stats::center_columns_view(&matrix.view());
        let covariance_owned = ndarray_stats::covariance_matrix(&matrix).unwrap();
        let covariance_view = ndarray_stats::covariance_matrix_view(&matrix.view()).unwrap();
        let correlation_owned = ndarray_stats::correlation_matrix(&matrix).unwrap();
        let correlation_view = ndarray_stats::correlation_matrix_view(&matrix.view()).unwrap();

        for i in 0..means_owned.len() {
            assert!((means_owned[i] - means_view[i]).abs() < 1e-12);
        }
        for i in 0..matrix.nrows() {
            for j in 0..matrix.ncols() {
                assert!((centered_owned[[i, j]] - centered_view[[i, j]]).abs() < 1e-12);
            }
        }
        for i in 0..2 {
            for j in 0..2 {
                assert!((covariance_owned[[i, j]] - covariance_view[[i, j]]).abs() < 1e-12);
                assert!((correlation_owned[[i, j]] - correlation_view[[i, j]]).abs() < 1e-12);
            }
        }
    }
}
