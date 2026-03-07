//! Statistics bindings for Python.

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;

use crate::error::to_py_err;

/// Compute column means.
#[pyfunction]
pub fn column_means<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyArray2<f64>>,
) -> PyResult<Py<PyArray1<f64>>> {
    let arr = matrix.readonly();
    let result = nabled_ml::stats::column_means(&arr.as_array().to_owned());
    Ok(PyArray1::from_owned_array(py, result).unbind())
}

/// Center columns (subtract mean).
#[pyfunction]
pub fn center_columns<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyArray2<f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    let arr = matrix.readonly();
    let result = nabled_ml::stats::center_columns(&arr.as_array().to_owned());
    Ok(PyArray2::from_owned_array(py, result).unbind())
}

/// Compute covariance matrix.
#[pyfunction]
pub fn covariance_matrix<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyArray2<f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    let arr = matrix.readonly();
    let result =
        nabled_ml::stats::covariance_matrix(&arr.as_array().to_owned()).map_err(to_py_err)?;
    Ok(PyArray2::from_owned_array(py, result).unbind())
}

/// Compute correlation matrix.
#[pyfunction]
pub fn correlation_matrix<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyArray2<f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    let arr = matrix.readonly();
    let result =
        nabled_ml::stats::correlation_matrix(&arr.as_array().to_owned()).map_err(to_py_err)?;
    Ok(PyArray2::from_owned_array(py, result).unbind())
}
