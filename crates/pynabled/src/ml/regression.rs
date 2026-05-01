//! Regression bindings for Python.

use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::error::to_py_err;
use crate::utils;

/// Linear regression. Returns `(coefficients, fitted_values, residuals, r_squared)`.
#[pyfunction]
pub fn linear_regression<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    y: &Bound<'py, PyAny>,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>, f64)> {
    match (utils::real_array2(x, "x")?, utils::real_array1(y, "y")?) {
        (utils::RealReadonlyArray2::F32(x_arr), utils::RealReadonlyArray1::F32(y_arr)) => {
            let result = nabled_ml::regression::linear_regression_view(
                &x_arr.as_array(),
                &y_arr.as_array(),
                true,
            )
            .map_err(to_py_err)?;
            Ok((
                utils::pyarray1_from_owned(py, result.coefficients),
                utils::pyarray1_from_owned(py, result.fitted_values),
                utils::pyarray1_from_owned(py, result.residuals),
                f64::from(result.r_squared),
            ))
        }
        (utils::RealReadonlyArray2::F64(x_arr), utils::RealReadonlyArray1::F64(y_arr)) => {
            let result = nabled_ml::regression::linear_regression_view(
                &x_arr.as_array(),
                &y_arr.as_array(),
                true,
            )
            .map_err(to_py_err)?;
            Ok((
                utils::pyarray1_from_owned(py, result.coefficients),
                utils::pyarray1_from_owned(py, result.fitted_values),
                utils::pyarray1_from_owned(py, result.residuals),
                result.r_squared,
            ))
        }
        _ => Err(utils::matching_real_dtype_error(&["x", "y"])),
    }
}

/// Linear regression into caller-provided outputs.
#[pyfunction]
pub fn linear_regression_into(
    x: &Bound<'_, PyAny>,
    y: &Bound<'_, PyAny>,
    coefficients: &Bound<'_, PyAny>,
    fitted_values: &Bound<'_, PyAny>,
    residuals: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    match (utils::real_array2(x, "x")?, utils::real_array1(y, "y")?) {
        (utils::RealReadonlyArray2::F32(x_arr), utils::RealReadonlyArray1::F32(y_arr)) => {
            let mut coefficients_arr =
                utils::output_array1::<f32>(coefficients, "coefficients", "float32")?;
            let mut fitted_values_arr =
                utils::output_array1::<f32>(fitted_values, "fitted_values", "float32")?;
            let mut residuals_arr = utils::output_array1::<f32>(residuals, "residuals", "float32")?;
            nabled_ml::regression::linear_regression_view_into(
                &x_arr.as_array(),
                &y_arr.as_array(),
                true,
                coefficients_arr.as_array_mut(),
                fitted_values_arr.as_array_mut(),
                residuals_arr.as_array_mut(),
            )
            .map(f64::from)
            .map_err(to_py_err)
        }
        (utils::RealReadonlyArray2::F64(x_arr), utils::RealReadonlyArray1::F64(y_arr)) => {
            let mut coefficients_arr =
                utils::output_array1::<f64>(coefficients, "coefficients", "float64")?;
            let mut fitted_values_arr =
                utils::output_array1::<f64>(fitted_values, "fitted_values", "float64")?;
            let mut residuals_arr = utils::output_array1::<f64>(residuals, "residuals", "float64")?;
            nabled_ml::regression::linear_regression_view_into(
                &x_arr.as_array(),
                &y_arr.as_array(),
                true,
                coefficients_arr.as_array_mut(),
                fitted_values_arr.as_array_mut(),
                residuals_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        _ => Err(utils::matching_real_dtype_error(&[
            "x",
            "y",
            "coefficients",
            "fitted_values",
            "residuals",
        ])),
    }
}

/// Linear regression for complex inputs. Returns
/// `(coefficients, fitted_values, residuals, r_squared)`.
#[pyfunction]
pub fn linear_regression_complex<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    y: &Bound<'py, PyAny>,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>, f64)> {
    match (utils::numeric_array2(x, "x")?, utils::numeric_array1(y, "y")?) {
        (utils::NumericReadonlyArray2::C64(x_arr), utils::NumericReadonlyArray1::C64(y_arr)) => {
            let result = nabled_ml::regression::linear_regression_complex_view(
                &x_arr.as_array(),
                &y_arr.as_array(),
                true,
            )
            .map_err(to_py_err)?;
            Ok((
                utils::pyarray1_from_owned(py, result.coefficients),
                utils::pyarray1_from_owned(py, result.fitted_values),
                utils::pyarray1_from_owned(py, result.residuals),
                result.r_squared,
            ))
        }
        _ => Err(utils::matching_complex_dtype_error(&["x", "y"])),
    }
}

/// Linear regression for complex inputs into caller-provided outputs.
#[pyfunction]
pub fn linear_regression_complex_into(
    x: &Bound<'_, PyAny>,
    y: &Bound<'_, PyAny>,
    coefficients: &Bound<'_, PyAny>,
    fitted_values: &Bound<'_, PyAny>,
    residuals: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    match (utils::numeric_array2(x, "x")?, utils::numeric_array1(y, "y")?) {
        (utils::NumericReadonlyArray2::C64(x_arr), utils::NumericReadonlyArray1::C64(y_arr)) => {
            let mut coefficients_arr = utils::output_array1::<num_complex::Complex64>(
                coefficients,
                "coefficients",
                "complex128",
            )?;
            let mut fitted_values_arr = utils::output_array1::<num_complex::Complex64>(
                fitted_values,
                "fitted_values",
                "complex128",
            )?;
            let mut residuals_arr = utils::output_array1::<num_complex::Complex64>(
                residuals,
                "residuals",
                "complex128",
            )?;
            nabled_ml::regression::linear_regression_complex_view_into(
                &x_arr.as_array(),
                &y_arr.as_array(),
                true,
                coefficients_arr.as_array_mut(),
                fitted_values_arr.as_array_mut(),
                residuals_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        _ => Err(utils::matching_complex_dtype_error(&[
            "x",
            "y",
            "coefficients",
            "fitted_values",
            "residuals",
        ])),
    }
}
