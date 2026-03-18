//! Dense matrix pipeline bindings for Python.

use numpy::{PyArray1, PyArray2, PyArray3, PyArrayMethods};
use pyo3::prelude::*;

use crate::error::to_py_err;
use crate::utils;

/// Compute matrix-vector product y = A x.
#[pyfunction]
pub fn matvec<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyArray2<f64>>,
    vector: &Bound<'py, PyArray1<f64>>,
) -> PyResult<Py<PyArray1<f64>>> {
    utils::require_contiguous(matrix)?;
    utils::require_contiguous(vector)?;
    let m = matrix.readonly();
    let v = vector.readonly();
    let result =
        nabled_linalg::matrix::matvec_view(&m.as_array(), &v.as_array()).map_err(to_py_err)?;
    Ok(PyArray1::from_owned_array(py, result).unbind())
}

/// Compute matrix-matrix product C = A B.
#[pyfunction]
pub fn matmat<'py>(
    py: Python<'py>,
    left: &Bound<'py, PyArray2<f64>>,
    right: &Bound<'py, PyArray2<f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    utils::require_contiguous(left)?;
    utils::require_contiguous(right)?;
    let l = left.readonly();
    let r = right.readonly();
    let result =
        nabled_linalg::matrix::matmat_view(&l.as_array(), &r.as_array()).map_err(to_py_err)?;
    Ok(PyArray2::from_owned_array(py, result).unbind())
}

/// Batched matrix-vector product.
#[pyfunction]
pub fn batched_row_matvec<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyArray2<f64>>,
    vectors: &Bound<'py, PyArray2<f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    utils::require_contiguous(matrix)?;
    utils::require_contiguous(vectors)?;
    let m = matrix.readonly();
    let v = vectors.readonly();
    let result = nabled_linalg::matrix::batched_row_matvec_view(&v.as_array(), &m.as_array())
        .map_err(to_py_err)?;
    Ok(PyArray2::from_owned_array(py, result).unbind())
}

/// Batched matrix-matrix product.
#[pyfunction]
pub fn batched_matmat<'py>(
    py: Python<'py>,
    left: &Bound<'py, PyArray3<f64>>,
    right: &Bound<'py, PyArray3<f64>>,
) -> PyResult<Py<PyArray3<f64>>> {
    utils::require_contiguous(left)?;
    utils::require_contiguous(right)?;
    let l = left.readonly();
    let r = right.readonly();
    let result = nabled_linalg::matrix::batched_matmat_view(&l.as_array(), &r.as_array())
        .map_err(to_py_err)?;
    Ok(PyArray3::from_owned_array(py, result).unbind())
}
