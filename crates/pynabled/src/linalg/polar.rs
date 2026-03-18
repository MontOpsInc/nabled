//! Polar decomposition bindings for Python.

use numpy::{PyArray2, PyArrayMethods};
use pyo3::prelude::*;

use crate::error::to_py_err;
use crate::utils;

/// Compute polar decomposition. Returns (U, P).
#[pyfunction(name = "polar_compute")]
pub fn compute_polar<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyArray2<f64>>,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray2<f64>>)> {
    utils::require_contiguous(matrix)?;
    let arr = matrix.readonly();
    let result = nabled_linalg::polar::compute_polar_view(&arr.as_array()).map_err(to_py_err)?;
    Ok((
        PyArray2::from_owned_array(py, result.u).unbind(),
        PyArray2::from_owned_array(py, result.p).unbind(),
    ))
}
