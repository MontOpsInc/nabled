//! Statistical utilities over ndarray matrices.

use std::fmt;

use ndarray::{Array1, Array2, Axis};

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
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use super::ndarray_stats;

    #[test]
    fn covariance_and_correlation_are_well_formed() {
        let matrix =
            Array2::from_shape_vec((4, 2), vec![1.0, 3.0, 2.0, 2.0, 3.0, 1.0, 4.0, 0.0]).unwrap();
        let covariance = ndarray_stats::covariance_matrix(&matrix).unwrap();
        let correlation = ndarray_stats::correlation_matrix(&matrix).unwrap();
        assert_eq!(covariance.dim(), (2, 2));
        assert_eq!(correlation.dim(), (2, 2));
    }
}
