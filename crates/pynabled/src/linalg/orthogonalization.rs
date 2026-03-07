//! Orthogonalization bindings for Python.

use numpy::{PyArray2, PyArrayMethods};
use pyo3::prelude::*;

use crate::error::to_py_err;

/// Gram-Schmidt orthogonalization.
#[pyfunction(name = "gram_schmidt")]
pub fn gram_schmidt<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyArray2<f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    let arr = matrix.readonly();
    let result = nabled_linalg::orthogonalization::gram_schmidt(&arr.as_array().to_owned())
        .map_err(to_py_err)?;
    Ok(PyArray2::from_owned_array(py, result).unbind())
}

/// Classic Gram-Schmidt orthogonalization.
#[pyfunction(name = "gram_schmidt_classic")]
pub fn gram_schmidt_classic<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyArray2<f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    let arr = matrix.readonly();
    let result = nabled_linalg::orthogonalization::gram_schmidt_classic(&arr.as_array().to_owned())
        .map_err(to_py_err)?;
    Ok(PyArray2::from_owned_array(py, result).unbind())
}
