//! Orthogonalization bindings for Python.

use numpy::{PyArray2, PyArrayMethods};
use pyo3::prelude::*;

use crate::error::to_py_err;
use crate::utils;

/// Gram-Schmidt orthogonalization.
#[pyfunction(name = "gram_schmidt")]
pub fn gram_schmidt<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyArray2<f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    utils::require_contiguous(matrix)?;
    let arr = matrix.readonly();
    let result =
        nabled_linalg::orthogonalization::gram_schmidt_view(&arr.as_array()).map_err(to_py_err)?;
    Ok(PyArray2::from_owned_array(py, result).unbind())
}

/// Classic Gram-Schmidt orthogonalization.
#[pyfunction(name = "gram_schmidt_classic")]
pub fn gram_schmidt_classic<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyArray2<f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    utils::require_contiguous(matrix)?;
    let arr = matrix.readonly();
    let result = nabled_linalg::orthogonalization::gram_schmidt_classic_view(&arr.as_array())
        .map_err(to_py_err)?;
    Ok(PyArray2::from_owned_array(py, result).unbind())
}
