//! Regression bindings for Python.

use num_complex::Complex64;
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::error::to_py_err;
use crate::utils;

/// Linear regression. Returns (coefficients, r_squared).
#[pyfunction]
pub fn linear_regression<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    y: &Bound<'py, PyAny>,
) -> PyResult<(Py<PyAny>, f64)> {
    match (utils::real_array2(x, "x")?, utils::real_array1(y, "y")?) {
        (utils::RealReadonlyArray2::F32(x_arr), utils::RealReadonlyArray1::F32(y_arr)) => {
            let result = nabled_ml::regression::linear_regression_view(
                &x_arr.as_array(),
                &y_arr.as_array(),
                true,
            )
            .map_err(to_py_err)?;
            Ok((utils::pyarray1_from_owned(py, result.coefficients), result.r_squared.into()))
        }
        (utils::RealReadonlyArray2::F64(x_arr), utils::RealReadonlyArray1::F64(y_arr)) => {
            let result = nabled_ml::regression::linear_regression_view(
                &x_arr.as_array(),
                &y_arr.as_array(),
                true,
            )
            .map_err(to_py_err)?;
            Ok((utils::pyarray1_from_owned(py, result.coefficients), result.r_squared))
        }
        _ => Err(utils::matching_real_dtype_error(&["x", "y"])),
    }
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
