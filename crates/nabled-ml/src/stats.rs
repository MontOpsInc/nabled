//! Statistical utilities over ndarray matrices.

use std::fmt;

use nabled_core::scalar::NabledReal;
use ndarray::{Array1, Array2, ArrayView2, Axis};
use num_complex::Complex64;

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

fn usize_to_scalar<T: NabledReal>(value: usize) -> T {
    T::from_usize(value).unwrap_or(T::max_value())
}

fn complex_is_finite(value: Complex64) -> bool { value.re.is_finite() && value.im.is_finite() }

fn column_means_impl<T: NabledReal>(matrix: &ArrayView2<'_, T>) -> Array1<T> {
    matrix.mean_axis(Axis(0)).unwrap_or_else(|| Array1::zeros(matrix.ncols()))
}

/// Compute column means.
#[must_use]
pub fn column_means<T: NabledReal>(matrix: &Array2<T>) -> Array1<T> {
    column_means_impl(&matrix.view())
}

/// Compute column means from a matrix view.
#[must_use]
pub fn column_means_view<T: NabledReal>(matrix: &ArrayView2<'_, T>) -> Array1<T> {
    column_means_impl(matrix)
}

fn center_columns_impl<T: NabledReal>(matrix: &ArrayView2<'_, T>) -> Array2<T> {
    let means = column_means_impl(matrix);
    let mut centered = Array2::<T>::zeros((matrix.nrows(), matrix.ncols()));
    for row in 0..matrix.nrows() {
        for col in 0..matrix.ncols() {
            centered[[row, col]] = matrix[[row, col]] - means[col];
        }
    }
    centered
}

/// Center columns by subtracting their means.
#[must_use]
pub fn center_columns<T: NabledReal>(matrix: &Array2<T>) -> Array2<T> {
    center_columns_impl(&matrix.view())
}

/// Center columns by subtracting their means from a matrix view.
#[must_use]
pub fn center_columns_view<T: NabledReal>(matrix: &ArrayView2<'_, T>) -> Array2<T> {
    center_columns_impl(matrix)
}

fn covariance_matrix_impl<T: NabledReal>(
    matrix: &ArrayView2<'_, T>,
) -> Result<Array2<T>, StatsError> {
    if matrix.is_empty() {
        return Err(StatsError::EmptyMatrix);
    }
    if matrix.nrows() < 2 {
        return Err(StatsError::InsufficientSamples);
    }

    let centered = center_columns_impl(matrix);
    let covariance: Array2<T> =
        centered.t().dot(&centered) / usize_to_scalar::<T>(matrix.nrows() - 1);

    if covariance.iter().any(|value| !value.is_finite()) {
        return Err(StatsError::NumericalInstability);
    }

    Ok(covariance)
}

/// Compute sample covariance matrix.
///
/// # Errors
/// Returns an error for empty input or fewer than two samples.
pub fn covariance_matrix<T: NabledReal>(matrix: &Array2<T>) -> Result<Array2<T>, StatsError> {
    covariance_matrix_impl(&matrix.view())
}

/// Compute sample covariance matrix from a matrix view.
///
/// # Errors
/// Returns an error for empty input or fewer than two samples.
pub fn covariance_matrix_view<T: NabledReal>(
    matrix: &ArrayView2<'_, T>,
) -> Result<Array2<T>, StatsError> {
    covariance_matrix_impl(matrix)
}

fn correlation_matrix_impl<T: NabledReal>(
    matrix: &ArrayView2<'_, T>,
) -> Result<Array2<T>, StatsError> {
    let covariance = covariance_matrix_impl(matrix)?;
    let n = covariance.nrows();
    let mut correlation = Array2::<T>::zeros((n, n));

    for i in 0..n {
        let sigma_i = covariance[[i, i]].sqrt();
        for j in 0..n {
            let sigma_j = covariance[[j, j]].sqrt();
            let denom = (sigma_i * sigma_j).max(T::epsilon());
            correlation[[i, j]] = covariance[[i, j]] / denom;
        }
    }

    Ok(correlation)
}

/// Compute correlation matrix.
///
/// # Errors
/// Returns an error if covariance computation fails.
pub fn correlation_matrix<T: NabledReal>(matrix: &Array2<T>) -> Result<Array2<T>, StatsError> {
    correlation_matrix_impl(&matrix.view())
}

/// Compute correlation matrix from a matrix view.
///
/// # Errors
/// Returns an error if covariance computation fails.
pub fn correlation_matrix_view<T: NabledReal>(
    matrix: &ArrayView2<'_, T>,
) -> Result<Array2<T>, StatsError> {
    correlation_matrix_impl(matrix)
}

fn column_means_complex_impl(matrix: &ArrayView2<'_, Complex64>) -> Array1<Complex64> {
    if matrix.nrows() == 0 {
        return Array1::zeros(matrix.ncols());
    }

    let mut means = Array1::<Complex64>::zeros(matrix.ncols());
    for col in 0..matrix.ncols() {
        let mut sum = Complex64::new(0.0, 0.0);
        for row in 0..matrix.nrows() {
            sum += matrix[[row, col]];
        }
        means[col] = sum / usize_to_scalar::<f64>(matrix.nrows());
    }
    means
}

/// Compute complex column means.
#[must_use]
pub fn column_means_complex(matrix: &Array2<Complex64>) -> Array1<Complex64> {
    column_means_complex_impl(&matrix.view())
}

/// Compute complex column means from a matrix view.
#[must_use]
pub fn column_means_complex_view(matrix: &ArrayView2<'_, Complex64>) -> Array1<Complex64> {
    column_means_complex_impl(matrix)
}

fn center_columns_complex_impl(matrix: &ArrayView2<'_, Complex64>) -> Array2<Complex64> {
    let means = column_means_complex_impl(matrix);
    let mut centered = Array2::<Complex64>::zeros((matrix.nrows(), matrix.ncols()));
    for row in 0..matrix.nrows() {
        for col in 0..matrix.ncols() {
            centered[[row, col]] = matrix[[row, col]] - means[col];
        }
    }
    centered
}

/// Center complex columns by subtracting their means.
#[must_use]
pub fn center_columns_complex(matrix: &Array2<Complex64>) -> Array2<Complex64> {
    center_columns_complex_impl(&matrix.view())
}

/// Center complex columns by subtracting their means from a matrix view.
#[must_use]
pub fn center_columns_complex_view(matrix: &ArrayView2<'_, Complex64>) -> Array2<Complex64> {
    center_columns_complex_impl(matrix)
}

fn covariance_matrix_complex_impl(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<Array2<Complex64>, StatsError> {
    if matrix.is_empty() {
        return Err(StatsError::EmptyMatrix);
    }
    if matrix.nrows() < 2 {
        return Err(StatsError::InsufficientSamples);
    }

    let centered = center_columns_complex_impl(matrix);
    let conjugate_transpose = centered.t().mapv(|value| value.conj());
    let covariance: Array2<Complex64> =
        conjugate_transpose.dot(&centered) / usize_to_scalar::<f64>(matrix.nrows() - 1);

    if covariance.iter().any(|value| !complex_is_finite(*value)) {
        return Err(StatsError::NumericalInstability);
    }

    Ok(covariance)
}

/// Compute sample covariance matrix for complex observations.
///
/// # Errors
/// Returns an error for empty input or fewer than two samples.
pub fn covariance_matrix_complex(
    matrix: &Array2<Complex64>,
) -> Result<Array2<Complex64>, StatsError> {
    covariance_matrix_complex_impl(&matrix.view())
}

/// Compute sample covariance matrix for complex observations from a matrix view.
///
/// # Errors
/// Returns an error for empty input or fewer than two samples.
pub fn covariance_matrix_complex_view(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<Array2<Complex64>, StatsError> {
    covariance_matrix_complex_impl(matrix)
}

fn correlation_matrix_complex_impl(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<Array2<Complex64>, StatsError> {
    let covariance = covariance_matrix_complex_impl(matrix)?;
    let n = covariance.nrows();
    let mut correlation = Array2::<Complex64>::zeros((n, n));

    for i in 0..n {
        let sigma_i = covariance[[i, i]].re.max(0.0).sqrt();
        for j in 0..n {
            let sigma_j = covariance[[j, j]].re.max(0.0).sqrt();
            let denom = (sigma_i * sigma_j).max(f64::EPSILON);
            correlation[[i, j]] = covariance[[i, j]] / denom;
        }
    }

    if correlation.iter().any(|value| !complex_is_finite(*value)) {
        return Err(StatsError::NumericalInstability);
    }

    Ok(correlation)
}

/// Compute correlation matrix for complex observations.
///
/// # Errors
/// Returns an error if covariance computation fails.
pub fn correlation_matrix_complex(
    matrix: &Array2<Complex64>,
) -> Result<Array2<Complex64>, StatsError> {
    correlation_matrix_complex_impl(&matrix.view())
}

/// Compute correlation matrix for complex observations from a matrix view.
///
/// # Errors
/// Returns an error if covariance computation fails.
pub fn correlation_matrix_complex_view(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<Array2<Complex64>, StatsError> {
    correlation_matrix_complex_impl(matrix)
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;
    use num_complex::Complex64;

    use super::*;

    #[test]
    fn covariance_and_correlation_are_well_formed() {
        let matrix =
            Array2::from_shape_vec((4, 2), vec![1.0_f64, 3.0, 2.0, 2.0, 3.0, 1.0, 4.0, 0.0])
                .unwrap();
        let covariance = covariance_matrix(&matrix).unwrap();
        let correlation = correlation_matrix(&matrix).unwrap();
        assert_eq!(covariance.dim(), (2, 2));
        assert_eq!(correlation.dim(), (2, 2));
    }

    #[test]
    fn stats_rejects_empty_and_insufficient_inputs() {
        let empty = Array2::<f64>::zeros((0, 0));
        assert!(matches!(covariance_matrix(&empty), Err(StatsError::EmptyMatrix)));

        let one_row = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
        assert!(matches!(covariance_matrix(&one_row), Err(StatsError::InsufficientSamples)));
    }

    #[test]
    fn center_columns_zeroes_means() {
        let matrix =
            Array2::from_shape_vec((3, 2), vec![1.0_f64, 2.0, 2.0, 3.0, 3.0, 4.0]).unwrap();
        let centered = center_columns(&matrix);
        let means = column_means(&centered);
        assert!(means.iter().all(|value| num_traits::Float::abs(*value) < 1e-12));
    }

    #[test]
    fn column_means_handles_empty_input() {
        let matrix = Array2::<f64>::zeros((0, 3));
        let means = column_means(&matrix);
        assert_eq!(means.len(), 3);
        assert!(means.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn covariance_reports_numerical_instability() {
        let matrix = Array2::from_shape_vec((2, 2), vec![f64::MAX, 0.0, -f64::MAX, 0.0]).unwrap();
        let result = covariance_matrix(&matrix);
        assert!(matches!(result, Err(StatsError::NumericalInstability)));
    }

    #[test]
    fn correlation_handles_zero_variance_column() {
        let matrix =
            Array2::from_shape_vec((3, 2), vec![1.0_f64, 10.0, 1.0, 20.0, 1.0, 30.0]).unwrap();
        let correlation = correlation_matrix(&matrix).unwrap();
        assert!(correlation[[0, 0]].is_finite());
        assert!(correlation[[0, 1]].is_finite());
        assert!(correlation[[1, 0]].is_finite());
        assert!(correlation[[1, 1]].is_finite());
    }

    #[test]
    fn view_variants_match_owned() {
        let matrix =
            Array2::from_shape_vec((4, 2), vec![1.0_f64, 3.0, 2.0, 2.0, 3.0, 1.0, 4.0, 0.0])
                .unwrap();
        let means_owned = column_means(&matrix);
        let means_view = column_means_view(&matrix.view());
        let centered_owned = center_columns(&matrix);
        let centered_view = center_columns_view(&matrix.view());
        let covariance_owned = covariance_matrix(&matrix).unwrap();
        let covariance_view = covariance_matrix_view(&matrix.view()).unwrap();
        let correlation_owned = correlation_matrix(&matrix).unwrap();
        let correlation_view = correlation_matrix_view(&matrix.view()).unwrap();

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

    #[test]
    fn complex_covariance_and_correlation_are_well_formed() {
        let matrix = Array2::from_shape_vec((4, 2), vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(3.0, 1.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(2.0, 0.5),
            Complex64::new(3.0, 0.2),
            Complex64::new(1.0, -0.3),
            Complex64::new(4.0, 0.7),
            Complex64::new(0.0, 0.0),
        ])
        .unwrap();

        let covariance = covariance_matrix_complex(&matrix).unwrap();
        let correlation = correlation_matrix_complex(&matrix).unwrap();
        assert_eq!(covariance.dim(), (2, 2));
        assert_eq!(correlation.dim(), (2, 2));
    }

    #[test]
    fn complex_view_variants_match_owned() {
        let matrix = Array2::from_shape_vec((3, 2), vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(2.0, 2.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(3.0, -2.0),
            Complex64::new(4.0, 1.0),
        ])
        .unwrap();

        let means_owned = column_means_complex(&matrix);
        let means_view = column_means_complex_view(&matrix.view());
        let centered_owned = center_columns_complex(&matrix);
        let centered_view = center_columns_complex_view(&matrix.view());
        let covariance_owned = covariance_matrix_complex(&matrix).unwrap();
        let covariance_view = covariance_matrix_complex_view(&matrix.view()).unwrap();
        let correlation_owned = correlation_matrix_complex(&matrix).unwrap();
        let correlation_view = correlation_matrix_complex_view(&matrix.view()).unwrap();

        for i in 0..means_owned.len() {
            assert!((means_owned[i] - means_view[i]).norm() < 1e-12);
        }
        for i in 0..matrix.nrows() {
            for j in 0..matrix.ncols() {
                assert!((centered_owned[[i, j]] - centered_view[[i, j]]).norm() < 1e-12);
            }
        }
        for i in 0..2 {
            for j in 0..2 {
                assert!((covariance_owned[[i, j]] - covariance_view[[i, j]]).norm() < 1e-12);
                assert!((correlation_owned[[i, j]] - correlation_view[[i, j]]).norm() < 1e-12);
            }
        }
    }
}
