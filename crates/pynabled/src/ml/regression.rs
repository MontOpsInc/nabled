//! Regression bindings for Python.

use num_complex::Complex64;
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;

use crate::error::to_py_err;
use crate::utils;

/// Linear regression. Returns (coefficients, r_squared).
#[pyfunction]
pub fn linear_regression<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyArray2<f64>>,
    y: &Bound<'py, PyArray1<f64>>,
) -> PyResult<(Py<PyArray1<f64>>, f64)> {
    utils::require_contiguous(x)?;
    utils::require_contiguous(y)?;
    let x_arr = x.readonly();
    let y_arr = y.readonly();
    let result = nabled_ml::regression::linear_regression_view(
        &x_arr.as_array(),
        &y_arr.as_array(),
        true, // add_intercept
    )
    .map_err(to_py_err)?;
    Ok((PyArray1::from_owned_array(py, result.coefficients).unbind(), result.r_squared))
}

/// Linear regression for complex inputs. Returns (coefficients, r_squared).
#[pyfunction]
pub fn linear_regression_complex<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyArray2<Complex64>>,
    y: &Bound<'py, PyArray1<Complex64>>,
) -> PyResult<(Py<PyArray1<Complex64>>, f64)> {
    utils::require_contiguous(x)?;
    utils::require_contiguous(y)?;
    let x_arr = x.readonly();
    let y_arr = y.readonly();
    let result = nabled_ml::regression::linear_regression_complex_view(
        &x_arr.as_array(),
        &y_arr.as_array(),
        true,
    )
    .map_err(to_py_err)?;
    Ok((PyArray1::from_owned_array(py, result.coefficients).unbind(), result.r_squared))
}
