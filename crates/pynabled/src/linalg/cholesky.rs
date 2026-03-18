//! Cholesky decomposition bindings for Python.

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;

use crate::error::to_py_err;
use crate::utils;

/// Compute Cholesky decomposition. Returns L where A = L L^T.
#[pyfunction(name = "cholesky_decompose")]
pub fn decompose<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyArray2<f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    utils::require_contiguous(a)?;
    let arr = a.readonly();
    let view = arr.as_array();
    let result = nabled_linalg::cholesky::decompose_view(&view).map_err(to_py_err)?;
    Ok(PyArray2::from_owned_array(py, result.l).unbind())
}

/// Solve Ax = b for symmetric positive definite A.
#[pyfunction(name = "cholesky_solve")]
pub fn solve<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyArray2<f64>>,
    b: &Bound<'py, PyArray1<f64>>,
) -> PyResult<Py<PyArray1<f64>>> {
    utils::require_contiguous(a)?;
    utils::require_contiguous(b)?;
    let a_arr = a.readonly();
    let b_arr = b.readonly();
    let result = nabled_linalg::cholesky::solve_view(&a_arr.as_array(), &b_arr.as_array())
        .map_err(to_py_err)?;
    Ok(PyArray1::from_owned_array(py, result).unbind())
}

/// Compute matrix inverse using Cholesky.
#[pyfunction(name = "cholesky_inverse")]
pub fn inverse<'py>(py: Python<'py>, a: &Bound<'py, PyArray2<f64>>) -> PyResult<Py<PyArray2<f64>>> {
    utils::require_contiguous(a)?;
    let arr = a.readonly();
    let view = arr.as_array();
    let result = nabled_linalg::cholesky::inverse_view(&view).map_err(to_py_err)?;
    Ok(PyArray2::from_owned_array(py, result).unbind())
}
