//! Internal linear-regression kernel traits and backend adapters.

use std::marker::PhantomData;

use nalgebra::{DMatrix, DVector, RealField};
use ndarray::{Array1, Array2};
use num_traits::Float;
use num_traits::float::FloatCore;

use crate::interop::ndarray_to_nalgebra;
use crate::qr::QRConfig;
use crate::regression::{NalgebraRegressionResult, NdarrayRegressionResult, RegressionError};

/// Internal kernel trait for linear regression operations.
pub(crate) trait RegressionKernel {
    type Matrix;
    type Vector;
    type Output;

    fn linear_regression(
        x: &Self::Matrix,
        y: &Self::Vector,
        add_intercept: bool,
    ) -> Result<Self::Output, RegressionError>;
}

/// Nalgebra-backed linear-regression kernel.
pub(crate) struct NalgebraRegressionKernel<T>(PhantomData<T>);

impl<T> RegressionKernel for NalgebraRegressionKernel<T>
where
    T: RealField + Copy + FloatCore + Float + num_traits::NumCast,
{
    type Matrix = DMatrix<T>;
    type Output = NalgebraRegressionResult<T>;
    type Vector = DVector<T>;

    #[inline]
    fn linear_regression(
        x: &Self::Matrix,
        y: &Self::Vector,
        add_intercept: bool,
    ) -> Result<Self::Output, RegressionError> {
        linear_regression_nalgebra_with_solver(
            x,
            y,
            add_intercept,
            crate::backend::qr::solve_nalgebra_least_squares,
        )
    }
}

/// Ndarray-backed linear-regression kernel.
pub(crate) struct NdarrayRegressionKernel<T>(PhantomData<T>);

impl<T> RegressionKernel for NdarrayRegressionKernel<T>
where
    T: Float + RealField + FloatCore + num_traits::NumCast,
{
    type Matrix = Array2<T>;
    type Output = NdarrayRegressionResult<T>;
    type Vector = Array1<T>;

    #[inline]
    fn linear_regression(
        x: &Self::Matrix,
        y: &Self::Vector,
        add_intercept: bool,
    ) -> Result<Self::Output, RegressionError> {
        let nalg_x = ndarray_to_nalgebra(x);
        let nalg_y = DVector::from_vec(y.to_vec());
        let result = <NalgebraRegressionKernel<T> as RegressionKernel>::linear_regression(
            &nalg_x,
            &nalg_y,
            add_intercept,
        )?;

        Ok(NdarrayRegressionResult {
            coefficients:  Array1::from_vec(result.coefficients.as_slice().to_vec()),
            fitted_values: Array1::from_vec(result.fitted_values.as_slice().to_vec()),
            residuals:     Array1::from_vec(result.residuals.as_slice().to_vec()),
            r_squared:     result.r_squared,
        })
    }
}

/// Nalgebra LAPACK-backed linear-regression kernel.
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
pub(crate) struct NalgebraLapackRegressionKernel;

#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
impl RegressionKernel for NalgebraLapackRegressionKernel {
    type Matrix = DMatrix<f64>;
    type Output = NalgebraRegressionResult<f64>;
    type Vector = DVector<f64>;

    #[inline]
    fn linear_regression(
        x: &Self::Matrix,
        y: &Self::Vector,
        add_intercept: bool,
    ) -> Result<Self::Output, RegressionError> {
        linear_regression_nalgebra_with_solver(
            x,
            y,
            add_intercept,
            crate::backend::qr::solve_nalgebra_lapack_least_squares,
        )
    }
}

/// Ndarray LAPACK-backed linear-regression kernel.
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
pub(crate) struct NdarrayLapackRegressionKernel;

#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
impl RegressionKernel for NdarrayLapackRegressionKernel {
    type Matrix = Array2<f64>;
    type Output = NdarrayRegressionResult<f64>;
    type Vector = Array1<f64>;

    #[inline]
    fn linear_regression(
        x: &Self::Matrix,
        y: &Self::Vector,
        add_intercept: bool,
    ) -> Result<Self::Output, RegressionError> {
        let nalg_x = ndarray_to_nalgebra(x);
        let nalg_y = DVector::from_vec(y.to_vec());
        let result = <NalgebraLapackRegressionKernel as RegressionKernel>::linear_regression(
            &nalg_x,
            &nalg_y,
            add_intercept,
        )?;

        Ok(NdarrayRegressionResult {
            coefficients:  Array1::from_vec(result.coefficients.as_slice().to_vec()),
            fitted_values: Array1::from_vec(result.fitted_values.as_slice().to_vec()),
            residuals:     Array1::from_vec(result.residuals.as_slice().to_vec()),
            r_squared:     result.r_squared,
        })
    }
}

/// Dispatch linear regression to the nalgebra kernel.
#[inline]
pub(crate) fn linear_regression_nalgebra<T>(
    x: &DMatrix<T>,
    y: &DVector<T>,
    add_intercept: bool,
) -> Result<NalgebraRegressionResult<T>, RegressionError>
where
    T: RealField + Copy + FloatCore + Float + num_traits::NumCast,
{
    <NalgebraRegressionKernel<T> as RegressionKernel>::linear_regression(x, y, add_intercept)
}

/// Dispatch linear regression to the ndarray kernel.
#[inline]
pub(crate) fn linear_regression_ndarray<T>(
    x: &Array2<T>,
    y: &Array1<T>,
    add_intercept: bool,
) -> Result<NdarrayRegressionResult<T>, RegressionError>
where
    T: Float + RealField + FloatCore + num_traits::NumCast,
{
    <NdarrayRegressionKernel<T> as RegressionKernel>::linear_regression(x, y, add_intercept)
}

/// Dispatch linear regression to the nalgebra LAPACK kernel.
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
#[inline]
pub(crate) fn linear_regression_nalgebra_lapack(
    x: &DMatrix<f64>,
    y: &DVector<f64>,
    add_intercept: bool,
) -> Result<NalgebraRegressionResult<f64>, RegressionError> {
    <NalgebraLapackRegressionKernel as RegressionKernel>::linear_regression(x, y, add_intercept)
}

/// Dispatch linear regression to the ndarray LAPACK kernel.
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
#[inline]
pub(crate) fn linear_regression_ndarray_lapack(
    x: &Array2<f64>,
    y: &Array1<f64>,
    add_intercept: bool,
) -> Result<NdarrayRegressionResult<f64>, RegressionError> {
    <NdarrayLapackRegressionKernel as RegressionKernel>::linear_regression(x, y, add_intercept)
}

fn linear_regression_nalgebra_with_solver<T, F>(
    x: &DMatrix<T>,
    y: &DVector<T>,
    add_intercept: bool,
    solve_fn: F,
) -> Result<NalgebraRegressionResult<T>, RegressionError>
where
    T: RealField + Copy + FloatCore + Float + num_traits::NumCast,
    F: Fn(&DMatrix<T>, &DVector<T>, &QRConfig<T>) -> Result<DVector<T>, crate::qr::QRError>,
{
    if x.is_empty() || y.is_empty() {
        return Err(RegressionError::EmptyInput);
    }

    let (n_samples, _) = x.shape();
    if n_samples != y.len() {
        return Err(RegressionError::DimensionMismatch("X rows must match y length".to_string()));
    }

    let x_design = if add_intercept { augment_intercept(x) } else { x.clone() };

    let config = QRConfig::default();
    let coefficients =
        solve_fn(&x_design, y, &config).map_err(|e| RegressionError::QRError(e.to_string()))?;

    let fitted_values = &x_design * &coefficients;
    let residuals = y - &fitted_values;

    let y_mean = mean_vector(y)?;
    let mut ss_tot = T::zero();
    let mut ss_res = T::zero();
    for i in 0..n_samples {
        let diff = y[i] - y_mean;
        ss_tot += diff * diff;
        ss_res += residuals[i] * residuals[i];
    }

    let r_squared = if ss_tot > T::zero() { T::one() - ss_res / ss_tot } else { Float::nan() };

    Ok(NalgebraRegressionResult { coefficients, fitted_values, residuals, r_squared })
}

fn augment_intercept<T>(x: &DMatrix<T>) -> DMatrix<T>
where
    T: RealField + Copy,
{
    let (n_samples, n_features) = x.shape();
    let mut x_new = DMatrix::zeros(n_samples, n_features + 1);
    for i in 0..n_samples {
        x_new[(i, 0)] = T::one();
        for j in 0..n_features {
            x_new[(i, j + 1)] = x[(i, j)];
        }
    }
    x_new
}

fn mean_vector<T>(y: &DVector<T>) -> Result<T, RegressionError>
where
    T: RealField + Copy + num_traits::NumCast,
{
    let mut sum = T::zero();
    for &value in y.iter() {
        sum += value;
    }

    let count = num_traits::NumCast::from(y.len()).ok_or_else(|| {
        RegressionError::DimensionMismatch("failed to cast sample count to scalar".to_string())
    })?;
    Ok(sum / count)
}
