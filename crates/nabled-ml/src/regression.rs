//! Linear regression over ndarray matrices.

use std::fmt;

use nabled_core::scalar::NabledReal;
use nabled_linalg::lu::{self as lu, LUError};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_complex::Complex64;

/// Regression result for ndarray inputs.
#[derive(Debug, Clone)]
pub struct NdarrayRegressionResult<T = f64> {
    /// Regression coefficients.
    pub coefficients:  Array1<T>,
    /// Model fitted values.
    pub fitted_values: Array1<T>,
    /// Residuals (`y - y_hat`).
    pub residuals:     Array1<T>,
    /// Coefficient of determination.
    pub r_squared:     T,
}

/// Complex regression result for ndarray inputs.
#[derive(Debug, Clone)]
pub struct NdarrayComplexRegressionResult {
    /// Regression coefficients.
    pub coefficients:  Array1<Complex64>,
    /// Model fitted values.
    pub fitted_values: Array1<Complex64>,
    /// Residuals (`y - y_hat`).
    pub residuals:     Array1<Complex64>,
    /// Coefficient of determination (real-valued).
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

fn usize_to_scalar<T: NabledReal>(value: usize) -> T {
    T::from_usize(value).unwrap_or(T::max_value())
}

fn map_lu_error(error: LUError) -> RegressionError {
    match error {
        LUError::EmptyMatrix => RegressionError::EmptyInput,
        LUError::NotSquare => RegressionError::InvalidInput("normal matrix was not square".into()),
        LUError::InvalidInput(message) => RegressionError::InvalidInput(message),
        LUError::SingularMatrix | LUError::NumericalInstability => RegressionError::Singular,
    }
}

#[cfg(any(
    feature = "openblas-system",
    feature = "openblas-static",
    feature = "netlib-system",
    feature = "netlib-static"
))]
fn linear_regression_impl<T>(
    x: &ArrayView2<'_, T>,
    y: &ArrayView1<'_, T>,
    add_intercept: bool,
) -> Result<NdarrayRegressionResult<T>, RegressionError>
where
    T: NabledReal + ndarray_linalg::Lapack,
{
    if x.is_empty() || y.is_empty() {
        return Err(RegressionError::EmptyInput);
    }
    if x.nrows() != y.len() {
        return Err(RegressionError::DimensionMismatch);
    }

    let maybe_design = if add_intercept {
        let mut with_intercept = Array2::<T>::zeros((x.nrows(), x.ncols() + 1));
        for row in 0..x.nrows() {
            with_intercept[[row, 0]] = T::one();
            for col in 0..x.ncols() {
                with_intercept[[row, col + 1]] = x[[row, col]];
            }
        }
        Some(with_intercept)
    } else {
        None
    };
    let design = maybe_design.as_ref().map_or_else(|| x.view(), |owned| owned.view());

    let xt = design.t();
    let normal_matrix = xt.dot(&design);
    let normal_rhs = xt.dot(y);
    let coefficients = lu::solve(&normal_matrix, &normal_rhs).map_err(map_lu_error)?;

    let fitted_values = design.dot(&coefficients);
    let residuals = y - &fitted_values;

    let y_sum = y.iter().copied().fold(T::zero(), |acc, value| acc + value);
    let y_mean = y_sum / usize_to_scalar::<T>(y.len());

    let ss_total = y
        .iter()
        .copied()
        .map(|value| {
            let centered = value - y_mean;
            centered * centered
        })
        .fold(T::zero(), |acc, value| acc + value);

    let ss_residual = residuals
        .iter()
        .copied()
        .map(|value| value * value)
        .fold(T::zero(), |acc, value| acc + value);
    let r_squared =
        if ss_total <= T::epsilon() { T::one() } else { T::one() - ss_residual / ss_total };

    Ok(NdarrayRegressionResult { coefficients, fitted_values, residuals, r_squared })
}

#[cfg(not(any(
    feature = "openblas-system",
    feature = "openblas-static",
    feature = "netlib-system",
    feature = "netlib-static"
)))]
fn linear_regression_impl<T>(
    x: &ArrayView2<'_, T>,
    y: &ArrayView1<'_, T>,
    add_intercept: bool,
) -> Result<NdarrayRegressionResult<T>, RegressionError>
where
    T: NabledReal,
{
    if x.is_empty() || y.is_empty() {
        return Err(RegressionError::EmptyInput);
    }
    if x.nrows() != y.len() {
        return Err(RegressionError::DimensionMismatch);
    }

    let maybe_design = if add_intercept {
        let mut with_intercept = Array2::<T>::zeros((x.nrows(), x.ncols() + 1));
        for row in 0..x.nrows() {
            with_intercept[[row, 0]] = T::one();
            for col in 0..x.ncols() {
                with_intercept[[row, col + 1]] = x[[row, col]];
            }
        }
        Some(with_intercept)
    } else {
        None
    };
    let design = maybe_design.as_ref().map_or_else(|| x.view(), |owned| owned.view());

    let xt = design.t();
    let normal_matrix = xt.dot(&design);
    let normal_rhs = xt.dot(y);
    let coefficients = lu::solve(&normal_matrix, &normal_rhs).map_err(map_lu_error)?;

    let fitted_values = design.dot(&coefficients);
    let residuals = y - &fitted_values;

    let y_sum = y.iter().copied().fold(T::zero(), |acc, value| acc + value);
    let y_mean = y_sum / usize_to_scalar::<T>(y.len());

    let ss_total = y
        .iter()
        .copied()
        .map(|value| {
            let centered = value - y_mean;
            centered * centered
        })
        .fold(T::zero(), |acc, value| acc + value);

    let ss_residual = residuals
        .iter()
        .copied()
        .map(|value| value * value)
        .fold(T::zero(), |acc, value| acc + value);
    let r_squared =
        if ss_total <= T::epsilon() { T::one() } else { T::one() - ss_residual / ss_total };

    Ok(NdarrayRegressionResult { coefficients, fitted_values, residuals, r_squared })
}

/// Solve linear regression with optional intercept.
///
/// # Errors
/// Returns an error for invalid dimensions or singular design matrix.
#[cfg(not(any(
    feature = "openblas-system",
    feature = "openblas-static",
    feature = "netlib-system",
    feature = "netlib-static"
)))]
pub fn linear_regression<T>(
    x: &Array2<T>,
    y: &Array1<T>,
    add_intercept: bool,
) -> Result<NdarrayRegressionResult<T>, RegressionError>
where
    T: NabledReal,
{
    linear_regression_impl(&x.view(), &y.view(), add_intercept)
}

/// Solve linear regression with optional intercept.
///
/// # Errors
/// Returns an error for invalid dimensions or singular design matrix.
#[cfg(any(
    feature = "openblas-system",
    feature = "openblas-static",
    feature = "netlib-system",
    feature = "netlib-static"
))]
pub fn linear_regression<T>(
    x: &Array2<T>,
    y: &Array1<T>,
    add_intercept: bool,
) -> Result<NdarrayRegressionResult<T>, RegressionError>
where
    T: NabledReal + ndarray_linalg::Lapack,
{
    linear_regression_impl(&x.view(), &y.view(), add_intercept)
}

/// Solve linear regression with optional intercept from matrix/vector views.
///
/// # Errors
/// Returns an error for invalid dimensions or singular design matrix.
#[cfg(not(any(
    feature = "openblas-system",
    feature = "openblas-static",
    feature = "netlib-system",
    feature = "netlib-static"
)))]
pub fn linear_regression_view<T>(
    x: &ArrayView2<'_, T>,
    y: &ArrayView1<'_, T>,
    add_intercept: bool,
) -> Result<NdarrayRegressionResult<T>, RegressionError>
where
    T: NabledReal,
{
    linear_regression_impl(x, y, add_intercept)
}

/// Solve linear regression with optional intercept from matrix/vector views.
///
/// # Errors
/// Returns an error for invalid dimensions or singular design matrix.
#[cfg(any(
    feature = "openblas-system",
    feature = "openblas-static",
    feature = "netlib-system",
    feature = "netlib-static"
))]
pub fn linear_regression_view<T>(
    x: &ArrayView2<'_, T>,
    y: &ArrayView1<'_, T>,
    add_intercept: bool,
) -> Result<NdarrayRegressionResult<T>, RegressionError>
where
    T: NabledReal + ndarray_linalg::Lapack,
{
    linear_regression_impl(x, y, add_intercept)
}

fn linear_regression_complex_impl(
    x: &ArrayView2<'_, Complex64>,
    y: &ArrayView1<'_, Complex64>,
    add_intercept: bool,
) -> Result<NdarrayComplexRegressionResult, RegressionError> {
    if x.is_empty() || y.is_empty() {
        return Err(RegressionError::EmptyInput);
    }
    if x.nrows() != y.len() {
        return Err(RegressionError::DimensionMismatch);
    }

    let maybe_design = if add_intercept {
        let mut with_intercept = Array2::<Complex64>::zeros((x.nrows(), x.ncols() + 1));
        for row in 0..x.nrows() {
            with_intercept[[row, 0]] = Complex64::new(1.0, 0.0);
            for col in 0..x.ncols() {
                with_intercept[[row, col + 1]] = x[[row, col]];
            }
        }
        Some(with_intercept)
    } else {
        None
    };
    let design = maybe_design.as_ref().map_or_else(|| x.view(), |owned| owned.view());

    let xh = design.t().mapv(|value| value.conj());
    let normal_matrix = xh.dot(&design);
    let normal_rhs = xh.dot(y);
    let coefficients = lu::solve_complex(&normal_matrix, &normal_rhs).map_err(map_lu_error)?;

    let fitted_values = design.dot(&coefficients);
    let residuals = y - &fitted_values;

    let y_mean = y.iter().copied().sum::<Complex64>() / usize_to_scalar::<f64>(y.len());
    let ss_total = y.iter().map(|value| (*value - y_mean).norm_sqr()).sum::<f64>();
    let ss_residual = residuals.iter().map(Complex64::norm_sqr).sum::<f64>();
    let r_squared = if ss_total <= f64::EPSILON { 1.0 } else { 1.0 - ss_residual / ss_total };

    Ok(NdarrayComplexRegressionResult { coefficients, fitted_values, residuals, r_squared })
}

/// Solve complex linear regression with optional intercept.
///
/// # Errors
/// Returns an error for invalid dimensions or singular design matrix.
pub fn linear_regression_complex(
    x: &Array2<Complex64>,
    y: &Array1<Complex64>,
    add_intercept: bool,
) -> Result<NdarrayComplexRegressionResult, RegressionError> {
    linear_regression_complex_impl(&x.view(), &y.view(), add_intercept)
}

/// Solve complex linear regression with optional intercept from matrix/vector views.
///
/// # Errors
/// Returns an error for invalid dimensions or singular design matrix.
pub fn linear_regression_complex_view(
    x: &ArrayView2<'_, Complex64>,
    y: &ArrayView1<'_, Complex64>,
    add_intercept: bool,
) -> Result<NdarrayComplexRegressionResult, RegressionError> {
    linear_regression_complex_impl(x, y, add_intercept)
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2};
    use num_complex::Complex64;

    use super::*;

    #[test]
    fn linear_regression_fits_known_line() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
        let y = Array1::from_vec(vec![3.0_f64, 5.0, 7.0, 9.0]);
        let result = linear_regression(&x, &y, true).unwrap();
        assert!((result.coefficients[0] - 1.0_f64).abs() < 1e-8);
        assert!((result.coefficients[1] - 2.0_f64).abs() < 1e-8);
        assert!(result.r_squared > 0.999_999);
    }

    #[test]
    fn regression_without_intercept_fits_origin_line() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
        let y = Array1::from_vec(vec![2.0_f64, 4.0, 6.0, 8.0]);
        let result = linear_regression(&x, &y, false).unwrap();
        assert_eq!(result.coefficients.len(), 1);
        assert!((result.coefficients[0] - 2.0_f64).abs() < 1e-8);
    }

    #[test]
    fn regression_rejects_dimension_mismatch() {
        let x = Array2::from_shape_vec((2, 1), vec![1.0, 2.0]).unwrap();
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let result = linear_regression(&x, &y, true);
        assert!(matches!(result, Err(RegressionError::DimensionMismatch)));
    }

    #[test]
    fn regression_rejects_empty_inputs() {
        let x = Array2::<f64>::zeros((0, 0));
        let y = Array1::<f64>::zeros(0);
        let result = linear_regression(&x, &y, true);
        assert!(matches!(result, Err(RegressionError::EmptyInput)));
    }

    #[test]
    fn regression_reports_singular_system() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 1.0, 1.0]).unwrap();
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let result = linear_regression(&x, &y, true);
        assert!(matches!(result, Err(RegressionError::Singular)));
    }

    #[test]
    fn regression_constant_response_has_unit_r_squared() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
        let y = Array1::from_vec(vec![3.0_f64, 3.0, 3.0, 3.0]);
        let result = linear_regression(&x, &y, true).unwrap();
        assert!((result.r_squared - 1.0_f64).abs() < 1e-12);
        assert_eq!(result.fitted_values.len(), y.len());
        assert_eq!(result.residuals.len(), y.len());
    }

    #[test]
    fn regression_view_matches_owned() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
        let y = Array1::from_vec(vec![3.0_f64, 5.0, 7.0, 9.0]);
        let owned = linear_regression(&x, &y, true).unwrap();
        let viewed = linear_regression_view(&x.view(), &y.view(), true).unwrap();

        assert_eq!(owned.coefficients.len(), viewed.coefficients.len());
        for i in 0..owned.coefficients.len() {
            assert!((owned.coefficients[i] - viewed.coefficients[i]).abs() < 1e-12);
        }
        assert!((owned.r_squared - viewed.r_squared).abs() < 1e-12);
    }

    #[test]
    fn complex_regression_fits_known_line() {
        let x = Array2::from_shape_vec((4, 1), vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        ])
        .unwrap();
        let y = Array1::from_vec(vec![
            Complex64::new(3.0, 1.0),
            Complex64::new(5.0, 1.0),
            Complex64::new(7.0, 1.0),
            Complex64::new(9.0, 1.0),
        ]);

        let result = linear_regression_complex(&x, &y, true).unwrap();
        assert!((result.coefficients[0] - Complex64::new(1.0, 1.0)).norm() < 1e-8);
        assert!((result.coefficients[1] - Complex64::new(2.0, 0.0)).norm() < 1e-8);
        assert!(result.r_squared > 0.999_999);
    }

    #[test]
    fn complex_regression_view_matches_owned() {
        let x = Array2::from_shape_vec((4, 1), vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        ])
        .unwrap();
        let y = Array1::from_vec(vec![
            Complex64::new(3.0, 1.0),
            Complex64::new(5.0, 1.0),
            Complex64::new(7.0, 1.0),
            Complex64::new(9.0, 1.0),
        ]);

        let owned = linear_regression_complex(&x, &y, true).unwrap();
        let viewed = linear_regression_complex_view(&x.view(), &y.view(), true).unwrap();

        assert_eq!(owned.coefficients.len(), viewed.coefficients.len());
        for i in 0..owned.coefficients.len() {
            assert!((owned.coefficients[i] - viewed.coefficients[i]).norm() < 1e-12);
        }
        assert!((owned.r_squared - viewed.r_squared).abs() < 1e-12);
    }
}
