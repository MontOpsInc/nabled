//! Statistics bindings for Python.

use num_complex::Complex64;
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;

use crate::error::to_py_err;
use crate::utils;

/// Compute column means.
#[pyfunction]
pub fn column_means<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyArray2<f64>>,
) -> PyResult<Py<PyArray1<f64>>> {
    utils::require_contiguous(matrix)?;
    let arr = matrix.readonly();
    let result = nabled_ml::stats::column_means_view(&arr.as_array());
    Ok(PyArray1::from_owned_array(py, result).unbind())
}

/// Center columns (subtract mean).
#[pyfunction]
pub fn center_columns<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyArray2<f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    utils::require_contiguous(matrix)?;
    let arr = matrix.readonly();
    let result = nabled_ml::stats::center_columns_view(&arr.as_array());
    Ok(PyArray2::from_owned_array(py, result).unbind())
}

/// Compute covariance matrix.
#[pyfunction]
pub fn covariance_matrix<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyArray2<f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    utils::require_contiguous(matrix)?;
    let arr = matrix.readonly();
    let result = nabled_ml::stats::covariance_matrix_view(&arr.as_array()).map_err(to_py_err)?;
    Ok(PyArray2::from_owned_array(py, result).unbind())
}

/// Compute correlation matrix.
#[pyfunction]
pub fn correlation_matrix<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyArray2<f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    utils::require_contiguous(matrix)?;
    let arr = matrix.readonly();
    let result = nabled_ml::stats::correlation_matrix_view(&arr.as_array()).map_err(to_py_err)?;
    Ok(PyArray2::from_owned_array(py, result).unbind())
}

/// Compute column means for a complex matrix.
#[pyfunction]
pub fn column_means_complex<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyArray2<Complex64>>,
) -> PyResult<Py<PyArray1<Complex64>>> {
    utils::require_contiguous(matrix)?;
    let arr = matrix.readonly();
    let result = nabled_ml::stats::column_means_complex_view(&arr.as_array());
    Ok(PyArray1::from_owned_array(py, result).unbind())
}

/// Center complex columns (subtract mean).
#[pyfunction]
pub fn center_columns_complex<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyArray2<Complex64>>,
) -> PyResult<Py<PyArray2<Complex64>>> {
    utils::require_contiguous(matrix)?;
    let arr = matrix.readonly();
    let result = nabled_ml::stats::center_columns_complex_view(&arr.as_array());
    Ok(PyArray2::from_owned_array(py, result).unbind())
}

/// Compute covariance matrix for a complex matrix.
#[pyfunction]
pub fn covariance_matrix_complex<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyArray2<Complex64>>,
) -> PyResult<Py<PyArray2<Complex64>>> {
    utils::require_contiguous(matrix)?;
    let arr = matrix.readonly();
    let result =
        nabled_ml::stats::covariance_matrix_complex_view(&arr.as_array()).map_err(to_py_err)?;
    Ok(PyArray2::from_owned_array(py, result).unbind())
}

/// Compute correlation matrix for a complex matrix.
#[pyfunction]
pub fn correlation_matrix_complex<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyArray2<Complex64>>,
) -> PyResult<Py<PyArray2<Complex64>>> {
    utils::require_contiguous(matrix)?;
    let arr = matrix.readonly();
    let result =
        nabled_ml::stats::correlation_matrix_complex_view(&arr.as_array()).map_err(to_py_err)?;
    Ok(PyArray2::from_owned_array(py, result).unbind())
}
