//! PCA bindings for Python.

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;

use crate::error::to_py_err;

/// Compute PCA. Returns (components, explained_variance, explained_variance_ratio, mean, scores).
#[pyfunction]
pub fn compute_pca<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyArray2<f64>>,
    n_components: Option<usize>,
) -> PyResult<(
    Py<PyArray2<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray2<f64>>,
)> {
    let x_arr = x.readonly();
    let result = nabled_ml::pca::compute_pca(&x_arr.as_array().to_owned(), n_components)
        .map_err(to_py_err)?;
    Ok((
        PyArray2::from_owned_array(py, result.components).unbind(),
        PyArray1::from_owned_array(py, result.explained_variance).unbind(),
        PyArray1::from_owned_array(py, result.explained_variance_ratio).unbind(),
        PyArray1::from_owned_array(py, result.mean).unbind(),
        PyArray2::from_owned_array(py, result.scores).unbind(),
    ))
}
