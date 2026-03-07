//! Dense matrix pipeline bindings for Python.

use numpy::{PyArray1, PyArray2, PyArray3, PyArrayMethods};
use pyo3::prelude::*;

use crate::error::to_py_err;

/// Compute matrix-vector product y = A x.
#[pyfunction]
pub fn matvec<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyArray2<f64>>,
    vector: &Bound<'py, PyArray1<f64>>,
) -> PyResult<Py<PyArray1<f64>>> {
    let m = matrix.readonly();
    let v = vector.readonly();
    let result = nabled_linalg::matrix::matvec(&m.as_array().to_owned(), &v.as_array().to_owned())
        .map_err(to_py_err)?;
    Ok(PyArray1::from_owned_array(py, result).unbind())
}

/// Compute matrix-matrix product C = A B.
#[pyfunction]
pub fn matmat<'py>(
    py: Python<'py>,
    left: &Bound<'py, PyArray2<f64>>,
    right: &Bound<'py, PyArray2<f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    let l = left.readonly();
    let r = right.readonly();
    let result = nabled_linalg::matrix::matmat(&l.as_array().to_owned(), &r.as_array().to_owned())
        .map_err(to_py_err)?;
    Ok(PyArray2::from_owned_array(py, result).unbind())
}

/// Batched matrix-vector product.
#[pyfunction]
pub fn batched_row_matvec<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyArray2<f64>>,
    vectors: &Bound<'py, PyArray2<f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    let m = matrix.readonly();
    let v = vectors.readonly();
    let result = nabled_linalg::matrix::batched_row_matvec(
        &m.as_array().to_owned(),
        &v.as_array().to_owned(),
    )
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
    let l = left.readonly();
    let r = right.readonly();
    let result =
        nabled_linalg::matrix::batched_matmat(&l.as_array().to_owned(), &r.as_array().to_owned())
            .map_err(to_py_err)?;
    Ok(PyArray3::from_owned_array(py, result).unbind())
}
