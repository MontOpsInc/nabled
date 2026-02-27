//! # Linear Regression
//!
//! Ordinary least squares linear regression via QR decomposition.

use std::fmt;

use nalgebra::{DMatrix, DVector, RealField};
use ndarray::{Array1, Array2};
use num_traits::Float;

/// Error type for regression
#[derive(Debug, Clone, PartialEq)]
pub enum RegressionError {
    /// Empty inputs
    EmptyInput,
    /// Dimension mismatch
    DimensionMismatch(String),
    /// Singular matrix
    SingularMatrix,
    /// QR error
    QRError(String),
}

impl fmt::Display for RegressionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RegressionError::EmptyInput => write!(f, "Empty input"),
            RegressionError::DimensionMismatch(msg) => write!(f, "Dimension mismatch: {msg}"),
            RegressionError::SingularMatrix => write!(f, "Singular matrix"),
            RegressionError::QRError(msg) => write!(f, "QR error: {msg}"),
        }
    }
}

impl std::error::Error for RegressionError {}

/// Regression result for nalgebra
#[derive(Debug, Clone)]
pub struct NalgebraRegressionResult<T: RealField> {
    /// Coefficient estimates
    pub coefficients:  DVector<T>,
    /// Fitted values (X * coefficients)
    pub fitted_values: DVector<T>,
    /// Residuals (`y - fitted_values`)
    pub residuals:     DVector<T>,
    /// R-squared
    pub r_squared:     T,
}

/// Regression result for ndarray
#[derive(Debug, Clone)]
pub struct NdarrayRegressionResult<T: Float> {
    /// Coefficient estimates
    pub coefficients:  Array1<T>,
    /// Fitted values
    pub fitted_values: Array1<T>,
    /// Residuals
    pub residuals:     Array1<T>,
    /// R-squared
    pub r_squared:     T,
}

/// Nalgebra linear regression
pub mod nalgebra_regression {
    use super::*;

    /// Compute linear regression: min ||X*beta - y||^2
    /// # Panics
    /// Panics if internal numeric assumptions are violated during setup or
    /// intermediate conversion steps.
    ///
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn linear_regression<
        T: RealField + Copy + num_traits::float::FloatCore + Float + num_traits::NumCast,
    >(
        x: &DMatrix<T>,
        y: &DVector<T>,
        add_intercept: bool,
    ) -> Result<NalgebraRegressionResult<T>, RegressionError> {
        crate::backend::regression::linear_regression_nalgebra(x, y, add_intercept)
    }

    /// Compute linear regression with a LAPACK-backed kernel.
    ///
    /// This path is available on Linux when the `lapack-kernels` feature is enabled.
    ///
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    pub fn linear_regression_lapack(
        x: &DMatrix<f64>,
        y: &DVector<f64>,
        add_intercept: bool,
    ) -> Result<NalgebraRegressionResult<f64>, RegressionError> {
        crate::backend::regression::linear_regression_nalgebra_lapack(x, y, add_intercept)
    }
}

/// Ndarray linear regression
pub mod ndarray_regression {
    use super::*;

    /// Compute linear regression
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn linear_regression<
        T: Float + RealField + num_traits::float::FloatCore + num_traits::NumCast,
    >(
        x: &Array2<T>,
        y: &Array1<T>,
        add_intercept: bool,
    ) -> Result<NdarrayRegressionResult<T>, RegressionError> {
        crate::backend::regression::linear_regression_ndarray(x, y, add_intercept)
    }

    /// Compute linear regression with a LAPACK-backed kernel.
    ///
    /// This path is available on Linux when the `lapack-kernels` feature is enabled.
    ///
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    pub fn linear_regression_lapack(
        x: &Array2<f64>,
        y: &Array1<f64>,
        add_intercept: bool,
    ) -> Result<NdarrayRegressionResult<f64>, RegressionError> {
        crate::backend::regression::linear_regression_ndarray_lapack(x, y, add_intercept)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn test_linear_regression_known_slope() {
        let x = DMatrix::from_row_slice(4, 1, &[1.0, 2.0, 3.0, 4.0]);
        let y = DVector::from_vec(vec![3.0, 5.0, 7.0, 9.0]);
        let result = nalgebra_regression::linear_regression(&x, &y, true).unwrap();
        assert_relative_eq!(result.coefficients[0], 1.0, epsilon = 0.1);
        assert_relative_eq!(result.coefficients[1], 2.0, epsilon = 0.1);
        assert_relative_eq!(result.r_squared, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_linear_regression_r_squared() {
        let x = DMatrix::from_row_slice(5, 1, &[1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = DVector::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0]);
        let result = nalgebra_regression::linear_regression(&x, &y, true).unwrap();
        assert_relative_eq!(result.r_squared, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_ndarray_linear_regression_known_slope() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = Array1::from_vec(vec![3.0, 5.0, 7.0, 9.0]);
        let result = ndarray_regression::linear_regression(&x, &y, true).unwrap();

        assert_relative_eq!(result.coefficients[0], 1.0, epsilon = 0.1);
        assert_relative_eq!(result.coefficients[1], 2.0, epsilon = 0.1);
        assert_relative_eq!(result.r_squared, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_regression_error_display_variants() {
        assert!(format!("{}", RegressionError::EmptyInput).contains("Empty"));
        assert!(format!("{}", RegressionError::DimensionMismatch("x".to_string())).contains('x'));
        assert!(format!("{}", RegressionError::SingularMatrix).contains("Singular"));
        assert!(format!("{}", RegressionError::QRError("x".to_string())).contains('x'));
    }

    #[test]
    fn test_regression_error_paths() {
        let x_empty = DMatrix::<f64>::zeros(0, 1);
        let y_empty = DVector::<f64>::zeros(0);
        assert!(matches!(
            nalgebra_regression::linear_regression(&x_empty, &y_empty, true),
            Err(RegressionError::EmptyInput)
        ));

        let x = DMatrix::from_row_slice(3, 1, &[1.0, 2.0, 3.0]);
        let y_bad = DVector::from_vec(vec![1.0, 2.0]);
        assert!(matches!(
            nalgebra_regression::linear_regression(&x, &y_bad, true),
            Err(RegressionError::DimensionMismatch(_))
        ));

        let y = DVector::from_vec(vec![2.0, 4.0, 6.0]);
        let no_intercept = nalgebra_regression::linear_regression(&x, &y, false).unwrap();
        assert_eq!(no_intercept.coefficients.len(), 1);
    }

    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    #[test]
    fn test_nalgebra_linear_regression_lapack_known_slope() {
        let x = DMatrix::from_row_slice(4, 1, &[1.0, 2.0, 3.0, 4.0]);
        let y = DVector::from_vec(vec![3.0, 5.0, 7.0, 9.0]);
        let result = nalgebra_regression::linear_regression_lapack(&x, &y, true).unwrap();
        assert_relative_eq!(result.coefficients[0], 1.0, epsilon = 0.1);
        assert_relative_eq!(result.coefficients[1], 2.0, epsilon = 0.1);
    }

    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    #[test]
    fn test_ndarray_linear_regression_lapack_known_slope() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = Array1::from_vec(vec![3.0, 5.0, 7.0, 9.0]);
        let result = ndarray_regression::linear_regression_lapack(&x, &y, true).unwrap();
        assert_relative_eq!(result.coefficients[0], 1.0, epsilon = 0.1);
        assert_relative_eq!(result.coefficients[1], 2.0, epsilon = 0.1);
    }
}
