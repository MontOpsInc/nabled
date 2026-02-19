//! # Linear Regression
//!
//! Ordinary least squares linear regression via QR decomposition.

use nalgebra::{DMatrix, DVector, RealField};
use ndarray::{Array1, Array2};
use num_traits::Float;
use std::fmt;

use crate::qr::nalgebra_qr;
use crate::qr::QRConfig;

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
            RegressionError::DimensionMismatch(msg) => write!(f, "Dimension mismatch: {}", msg),
            RegressionError::SingularMatrix => write!(f, "Singular matrix"),
            RegressionError::QRError(msg) => write!(f, "QR error: {}", msg),
        }
    }
}

impl std::error::Error for RegressionError {}

/// Regression result for nalgebra
#[derive(Debug, Clone)]
pub struct NalgebraRegressionResult<T: RealField> {
    /// Coefficient estimates
    pub coefficients: DVector<T>,
    /// Fitted values (X * coefficients)
    pub fitted_values: DVector<T>,
    /// Residuals (y - fitted_values)
    pub residuals: DVector<T>,
    /// R-squared
    pub r_squared: T,
}

/// Regression result for ndarray
#[derive(Debug, Clone)]
pub struct NdarrayRegressionResult<T: Float> {
    /// Coefficient estimates
    pub coefficients: Array1<T>,
    /// Fitted values
    pub fitted_values: Array1<T>,
    /// Residuals
    pub residuals: Array1<T>,
    /// R-squared
    pub r_squared: T,
}

/// Nalgebra linear regression
pub mod nalgebra_regression {
    use super::*;
    use num_traits::float::FloatCore;

    /// Compute linear regression: min ||X*beta - y||^2
    pub fn linear_regression<T: RealField + Copy + FloatCore + num_traits::Float>(
        x: &DMatrix<T>,
        y: &DVector<T>,
        add_intercept: bool,
    ) -> Result<NalgebraRegressionResult<T>, RegressionError> {
        if x.is_empty() || y.is_empty() {
            return Err(RegressionError::EmptyInput);
        }
        let (n_samples, n_features) = x.shape();
        if n_samples != y.len() {
            return Err(RegressionError::DimensionMismatch(
                "X rows must match y length".to_string(),
            ));
        }

        let x_design = if add_intercept {
            let mut x_new = DMatrix::zeros(n_samples, n_features + 1);
            for i in 0..n_samples {
                x_new[(i, 0)] = T::one();
                for j in 0..n_features {
                    x_new[(i, j + 1)] = x[(i, j)];
                }
            }
            x_new
        } else {
            x.clone()
        };

        let config = QRConfig::default();
        let coefficients =
            nalgebra_qr::solve_least_squares(&x_design, y, &config).map_err(|e| {
                RegressionError::QRError(e.to_string())
            })?;

        let fitted_values = &x_design * &coefficients;
        let residuals = y - &fitted_values;

        let mut y_sum = T::zero();
        for i in 0..n_samples {
            y_sum = y_sum + y[i];
        }
        let y_mean = y_sum / num_traits::NumCast::from(n_samples).unwrap();
        let mut ss_tot = T::zero();
        let mut ss_res = T::zero();
        for i in 0..n_samples {
            let diff = y[i] - y_mean;
            ss_tot = ss_tot + diff * diff;
            ss_res = ss_res + residuals[i] * residuals[i];
        }
        let r_squared = if ss_tot > T::zero() {
            T::one() - ss_res / ss_tot
        } else {
            num_traits::Float::nan()
        };

        Ok(NalgebraRegressionResult {
            coefficients,
            fitted_values,
            residuals,
            r_squared,
        })
    }
}

/// Ndarray linear regression
pub mod ndarray_regression {
    use super::*;
    use crate::utils::ndarray_to_nalgebra;
    use num_traits::float::FloatCore;

    /// Compute linear regression
    pub fn linear_regression<T: Float + RealField + FloatCore + num_traits::NumCast>(
        x: &Array2<T>,
        y: &Array1<T>,
        add_intercept: bool,
    ) -> Result<NdarrayRegressionResult<T>, RegressionError> {
        let nalg_x = ndarray_to_nalgebra(x);
        let nalg_y = DVector::from_vec(y.to_vec());
        let result = super::nalgebra_regression::linear_regression(
            &nalg_x,
            &nalg_y,
            add_intercept,
        )?;
        Ok(NdarrayRegressionResult {
            coefficients: Array1::from_vec(result.coefficients.as_slice().to_vec()),
            fitted_values: Array1::from_vec(result.fitted_values.as_slice().to_vec()),
            residuals: Array1::from_vec(result.residuals.as_slice().to_vec()),
            r_squared: result.r_squared,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

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
}
