//! Schur decomposition bindings for Python.

use numpy::{PyArray2, PyArrayMethods};
use pyo3::prelude::*;

use crate::error::to_py_err;

/// Compute Schur decomposition. Returns (T, Q).
#[pyfunction(name = "schur_compute")]
pub fn compute_schur<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyArray2<f64>>,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray2<f64>>)> {
    let arr = matrix.readonly();
    let result =
        nabled_linalg::schur::compute_schur(&arr.as_array().to_owned()).map_err(to_py_err)?;
    Ok((
        PyArray2::from_owned_array(py, result.t).unbind(),
        PyArray2::from_owned_array(py, result.q).unbind(),
    ))
}
