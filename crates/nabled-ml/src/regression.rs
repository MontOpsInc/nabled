//! Linear regression over ndarray matrices.

use std::fmt;

use nabled_linalg::lu::{LUError, ndarray_lu};
use ndarray::{Array1, Array2};

/// Regression result for ndarray inputs.
#[derive(Debug, Clone)]
pub struct NdarrayRegressionResult {
    /// Regression coefficients.
    pub coefficients:  Array1<f64>,
    /// Model fitted values.
    pub fitted_values: Array1<f64>,
    /// Residuals (`y - y_hat`).
    pub residuals:     Array1<f64>,
    /// Coefficient of determination.
    pub r_squared:     f64,
}

/// Error type for regression operations.
#[derive(Debug, Clone, PartialEq)]
pub enum RegressionError {
    /// Input arrays are empty.
    EmptyInput,
    /// Input dimensions are incompatible.
    DimensionMismatch,
    /// Regression problem is singular.
    Singular,
    /// Invalid user input.
    InvalidInput(String),
}

impl fmt::Display for RegressionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RegressionError::EmptyInput => write!(f, "Input arrays cannot be empty"),
            RegressionError::DimensionMismatch => write!(f, "Input dimensions are incompatible"),
            RegressionError::Singular => write!(f, "Regression system is singular"),
            RegressionError::InvalidInput(message) => write!(f, "Invalid input: {message}"),
        }
    }
}

impl std::error::Error for RegressionError {}

fn usize_to_f64(value: usize) -> f64 { u32::try_from(value).map_or(f64::from(u32::MAX), f64::from) }

fn map_lu_error(error: LUError) -> RegressionError {
    match error {
        LUError::EmptyMatrix => RegressionError::EmptyInput,
        LUError::NotSquare => RegressionError::InvalidInput("normal matrix was not square".into()),
        LUError::InvalidInput(message) => RegressionError::InvalidInput(message),
        LUError::SingularMatrix | LUError::NumericalInstability => RegressionError::Singular,
    }
}

/// Ndarray regression functions.
pub mod ndarray_regression {
    use super::*;

    /// Solve linear regression with optional intercept.
    ///
    /// # Errors
    /// Returns an error for invalid dimensions or singular design matrix.
    pub fn linear_regression(
        x: &Array2<f64>,
        y: &Array1<f64>,
        add_intercept: bool,
    ) -> Result<NdarrayRegressionResult, RegressionError> {
        if x.is_empty() || y.is_empty() {
            return Err(RegressionError::EmptyInput);
        }
        if x.nrows() != y.len() {
            return Err(RegressionError::DimensionMismatch);
        }

        let design = if add_intercept {
            let mut with_intercept = Array2::<f64>::zeros((x.nrows(), x.ncols() + 1));
            for row in 0..x.nrows() {
                with_intercept[[row, 0]] = 1.0;
                for col in 0..x.ncols() {
                    with_intercept[[row, col + 1]] = x[[row, col]];
                }
            }
            with_intercept
        } else {
            x.clone()
        };

        let xt = design.t();
        let normal_matrix = xt.dot(&design);
        let normal_rhs = xt.dot(y);
        let coefficients = ndarray_lu::solve(&normal_matrix, &normal_rhs).map_err(map_lu_error)?;

        let fitted_values = design.dot(&coefficients);
        let residuals = y - &fitted_values;

        let y_mean = y.iter().sum::<f64>() / usize_to_f64(y.len());
        let ss_total = y
            .iter()
            .map(|value| {
                let centered = *value - y_mean;
                centered * centered
            })
            .sum::<f64>();
        let ss_residual = residuals.iter().map(|value| value * value).sum::<f64>();
        let r_squared = if ss_total <= f64::EPSILON { 1.0 } else { 1.0 - ss_residual / ss_total };

        Ok(NdarrayRegressionResult { coefficients, fitted_values, residuals, r_squared })
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2};

    use super::ndarray_regression;

    #[test]
    fn linear_regression_fits_known_line() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = Array1::from_vec(vec![3.0, 5.0, 7.0, 9.0]);
        let result = ndarray_regression::linear_regression(&x, &y, true).unwrap();
        assert!((result.coefficients[0] - 1.0).abs() < 1e-8);
        assert!((result.coefficients[1] - 2.0).abs() < 1e-8);
        assert!(result.r_squared > 0.999_999);
    }
}
