//! Matrix function bindings for Python.

use numpy::{PyArray2, PyArrayMethods};
use pyo3::prelude::*;

use crate::error::to_py_err;
use crate::utils;

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
    utils::require_contiguous(matrix)?;
    let arr = matrix.readonly();
    let terms = max_terms.unwrap_or(DEFAULT_MAX_TERMS);
    let tol = tolerance.unwrap_or(DEFAULT_TOLERANCE);
    let result = nabled_linalg::matrix_functions::matrix_exp_view(&arr.as_array(), terms, tol)
        .map_err(to_py_err)?;
    Ok(PyArray2::from_owned_array(py, result).unbind())
}

/// Matrix exponential via eigendecomposition.
#[pyfunction(name = "matrix_exp_eigen")]
pub fn matrix_exp_eigen<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyArray2<f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    utils::require_contiguous(matrix)?;
    let arr = matrix.readonly();
    let result = nabled_linalg::matrix_functions::matrix_exp_eigen_view(&arr.as_array())
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
    utils::require_contiguous(matrix)?;
    let arr = matrix.readonly();
    let terms = max_terms.unwrap_or(DEFAULT_MAX_TERMS);
    let tol = tolerance.unwrap_or(DEFAULT_TOLERANCE);
    let result =
        nabled_linalg::matrix_functions::matrix_log_taylor_view(&arr.as_array(), terms, tol)
            .map_err(to_py_err)?;
    Ok(PyArray2::from_owned_array(py, result).unbind())
}

/// Matrix log via eigendecomposition.
#[pyfunction(name = "matrix_log_eigen")]
pub fn matrix_log_eigen<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyArray2<f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    utils::require_contiguous(matrix)?;
    let arr = matrix.readonly();
    let result = nabled_linalg::matrix_functions::matrix_log_eigen_view(&arr.as_array())
        .map_err(to_py_err)?;
    Ok(PyArray2::from_owned_array(py, result).unbind())
}

/// Matrix log via SVD.
#[pyfunction(name = "matrix_log_svd")]
pub fn matrix_log_svd<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyArray2<f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    utils::require_contiguous(matrix)?;
    let arr = matrix.readonly();
    let result =
        nabled_linalg::matrix_functions::matrix_log_svd_view(&arr.as_array()).map_err(to_py_err)?;
    Ok(PyArray2::from_owned_array(py, result).unbind())
}

/// Matrix power A^p.
#[pyfunction(name = "matrix_power")]
pub fn matrix_power<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyArray2<f64>>,
    power: f64,
) -> PyResult<Py<PyArray2<f64>>> {
    utils::require_contiguous(matrix)?;
    let arr = matrix.readonly();
    let result = nabled_linalg::matrix_functions::matrix_power_view(&arr.as_array(), power)
        .map_err(to_py_err)?;
    Ok(PyArray2::from_owned_array(py, result).unbind())
}

/// Matrix sign function.
#[pyfunction(name = "matrix_sign")]
pub fn matrix_sign<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyArray2<f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    utils::require_contiguous(matrix)?;
    let arr = matrix.readonly();
    let result =
        nabled_linalg::matrix_functions::matrix_sign_view(&arr.as_array()).map_err(to_py_err)?;
    Ok(PyArray2::from_owned_array(py, result).unbind())
}
