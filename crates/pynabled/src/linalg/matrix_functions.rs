//! Matrix function bindings for Python.

use numpy::{PyArray2, PyArrayMethods};
use pyo3::prelude::*;

use crate::error::to_py_err;

const DEFAULT_MAX_TERMS: usize = 64;

const DEFAULT_TOLERANCE: f64 = 1e-14;

/// Matrix exponential via Taylor series.
#[pyfunction(name = "matrix_exp")]
pub fn matrix_exp<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyArray2<f64>>,
    max_terms: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let arr = matrix.readonly();
    let terms = max_terms.unwrap_or(DEFAULT_MAX_TERMS);
    let tol = tolerance.unwrap_or(DEFAULT_TOLERANCE);
    let result =
        nabled_linalg::matrix_functions::matrix_exp(&arr.as_array().to_owned(), terms, tol)
            .map_err(to_py_err)?;
    Ok(PyArray2::from_owned_array(py, result).unbind())
}

/// Matrix exponential via eigendecomposition.
#[pyfunction(name = "matrix_exp_eigen")]
pub fn matrix_exp_eigen<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyArray2<f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    let arr = matrix.readonly();
    let result = nabled_linalg::matrix_functions::matrix_exp_eigen(&arr.as_array().to_owned())
        .map_err(to_py_err)?;
    Ok(PyArray2::from_owned_array(py, result).unbind())
}

/// Matrix log via Taylor series.
#[pyfunction(name = "matrix_log_taylor")]
pub fn matrix_log_taylor<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyArray2<f64>>,
    max_terms: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let arr = matrix.readonly();
    let terms = max_terms.unwrap_or(DEFAULT_MAX_TERMS);
    let tol = tolerance.unwrap_or(DEFAULT_TOLERANCE);
    let result =
        nabled_linalg::matrix_functions::matrix_log_taylor(&arr.as_array().to_owned(), terms, tol)
            .map_err(to_py_err)?;
    Ok(PyArray2::from_owned_array(py, result).unbind())
}

/// Matrix log via eigendecomposition.
#[pyfunction(name = "matrix_log_eigen")]
pub fn matrix_log_eigen<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyArray2<f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    let arr = matrix.readonly();
    let result = nabled_linalg::matrix_functions::matrix_log_eigen(&arr.as_array().to_owned())
        .map_err(to_py_err)?;
    Ok(PyArray2::from_owned_array(py, result).unbind())
}

/// Matrix log via SVD.
#[pyfunction(name = "matrix_log_svd")]
pub fn matrix_log_svd<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyArray2<f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    let arr = matrix.readonly();
    let result = nabled_linalg::matrix_functions::matrix_log_svd(&arr.as_array().to_owned())
        .map_err(to_py_err)?;
    Ok(PyArray2::from_owned_array(py, result).unbind())
}

/// Matrix power A^p.
#[pyfunction(name = "matrix_power")]
pub fn matrix_power<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyArray2<f64>>,
    power: f64,
) -> PyResult<Py<PyArray2<f64>>> {
    let arr = matrix.readonly();
    let result = nabled_linalg::matrix_functions::matrix_power(&arr.as_array().to_owned(), power)
        .map_err(to_py_err)?;
    Ok(PyArray2::from_owned_array(py, result).unbind())
}

/// Matrix sign function.
#[pyfunction(name = "matrix_sign")]
pub fn matrix_sign<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyArray2<f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    let arr = matrix.readonly();
    let result = nabled_linalg::matrix_functions::matrix_sign(&arr.as_array().to_owned())
        .map_err(to_py_err)?;
    Ok(PyArray2::from_owned_array(py, result).unbind())
}
