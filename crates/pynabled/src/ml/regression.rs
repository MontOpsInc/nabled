//! Regression bindings for Python.

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;

use crate::error::to_py_err;

/// Linear regression. Returns (coefficients, r_squared).
#[pyfunction]
pub fn linear_regression<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyArray2<f64>>,
    y: &Bound<'py, PyArray1<f64>>,
) -> PyResult<(Py<PyArray1<f64>>, f64)> {
    let x_arr = x.readonly();
    let y_arr = y.readonly();
    let result = nabled_ml::regression::linear_regression(
        &x_arr.as_array().to_owned(),
        &y_arr.as_array().to_owned(),
        true, // add_intercept
    )
    .map_err(to_py_err)?;
    Ok((PyArray1::from_owned_array(py, result.coefficients).unbind(), result.r_squared))
}
