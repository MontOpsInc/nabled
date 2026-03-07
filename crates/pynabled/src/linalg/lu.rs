//! LU decomposition bindings for Python.

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;

use crate::error::to_py_err;

/// Compute LU decomposition. Returns (L, U).
#[pyfunction(name = "lu_decompose")]
pub fn decompose<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyArray2<f64>>,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray2<f64>>)> {
    let arr = a.readonly();
    let view = arr.as_array();
    let result = nabled_linalg::lu::decompose(&view.to_owned()).map_err(to_py_err)?;
    Ok((
        PyArray2::from_owned_array(py, result.l).unbind(),
        PyArray2::from_owned_array(py, result.u).unbind(),
    ))
}

/// Solve Ax = b using LU decomposition.
#[pyfunction(name = "lu_solve")]
pub fn solve<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyArray2<f64>>,
    b: &Bound<'py, PyArray1<f64>>,
) -> PyResult<Py<PyArray1<f64>>> {
    let a_arr = a.readonly();
    let b_arr = b.readonly();
    let result =
        nabled_linalg::lu::solve(&a_arr.as_array().to_owned(), &b_arr.as_array().to_owned())
            .map_err(to_py_err)?;
    Ok(PyArray1::from_owned_array(py, result).unbind())
}

/// Compute matrix inverse using LU.
#[pyfunction(name = "lu_inverse")]
pub fn inverse<'py>(py: Python<'py>, a: &Bound<'py, PyArray2<f64>>) -> PyResult<Py<PyArray2<f64>>> {
    let arr = a.readonly();
    let view = arr.as_array();
    let result = nabled_linalg::lu::inverse(&view.to_owned()).map_err(to_py_err)?;
    Ok(PyArray2::from_owned_array(py, result).unbind())
}

/// Compute determinant.
#[pyfunction(name = "lu_determinant")]
pub fn determinant(a: &Bound<'_, PyArray2<f64>>) -> PyResult<f64> {
    let arr = a.readonly();
    nabled_linalg::lu::determinant(&arr.as_array().to_owned()).map_err(to_py_err)
}
