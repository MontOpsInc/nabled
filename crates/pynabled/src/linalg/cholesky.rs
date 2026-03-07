//! Cholesky decomposition bindings for Python.

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;

use crate::error::to_py_err;

/// Compute Cholesky decomposition. Returns L where A = L L^T.
#[pyfunction(name = "cholesky_decompose")]
pub fn decompose<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyArray2<f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    let arr = a.readonly();
    let view = arr.as_array();
    let result = nabled_linalg::cholesky::decompose(&view.to_owned()).map_err(to_py_err)?;
    Ok(PyArray2::from_owned_array(py, result.l).unbind())
}

/// Solve Ax = b for symmetric positive definite A.
#[pyfunction(name = "cholesky_solve")]
pub fn solve<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyArray2<f64>>,
    b: &Bound<'py, PyArray1<f64>>,
) -> PyResult<Py<PyArray1<f64>>> {
    let a_arr = a.readonly();
    let b_arr = b.readonly();
    let result =
        nabled_linalg::cholesky::solve(&a_arr.as_array().to_owned(), &b_arr.as_array().to_owned())
            .map_err(to_py_err)?;
    Ok(PyArray1::from_owned_array(py, result).unbind())
}

/// Compute matrix inverse using Cholesky.
#[pyfunction(name = "cholesky_inverse")]
pub fn inverse<'py>(py: Python<'py>, a: &Bound<'py, PyArray2<f64>>) -> PyResult<Py<PyArray2<f64>>> {
    let arr = a.readonly();
    let view = arr.as_array();
    let result = nabled_linalg::cholesky::inverse(&view.to_owned()).map_err(to_py_err)?;
    Ok(PyArray2::from_owned_array(py, result).unbind())
}
