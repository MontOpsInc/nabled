//! Vector primitives bindings for Python.

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;

use crate::error::to_py_err;
use crate::utils;

/// Compute dot product of two vectors.
#[pyfunction]
pub fn dot(a: &Bound<'_, PyArray1<f64>>, b: &Bound<'_, PyArray1<f64>>) -> PyResult<f64> {
    utils::require_contiguous(a)?;
    utils::require_contiguous(b)?;
    let a_arr = a.readonly();
    let b_arr = b.readonly();
    nabled_linalg::vector::dot_view(&a_arr.as_array(), &b_arr.as_array()).map_err(to_py_err)
}

/// Compute L2 norm of a vector.
#[pyfunction]
pub fn l2_norm(v: &Bound<'_, PyArray1<f64>>) -> PyResult<f64> {
    utils::require_contiguous(v)?;
    let arr = v.readonly();
    nabled_linalg::vector::l2_norm_view(&arr.as_array()).map_err(to_py_err)
}

/// Compute cosine similarity between two vectors.
#[pyfunction]
pub fn cosine_similarity(
    a: &Bound<'_, PyArray1<f64>>,
    b: &Bound<'_, PyArray1<f64>>,
) -> PyResult<f64> {
    utils::require_contiguous(a)?;
    utils::require_contiguous(b)?;
    let a_arr = a.readonly();
    let b_arr = b.readonly();
    nabled_linalg::vector::cosine_similarity_view(&a_arr.as_array(), &b_arr.as_array())
        .map_err(to_py_err)
}

/// Compute pairwise L2 distances between rows of two matrices.
#[pyfunction]
pub fn pairwise_l2_distance<'py>(
    py: Python<'py>,
    left: &Bound<'py, PyArray2<f64>>,
    right: &Bound<'py, PyArray2<f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    utils::require_contiguous(left)?;
    utils::require_contiguous(right)?;
    let l = left.readonly();
    let r = right.readonly();
    let result = nabled_linalg::vector::pairwise_l2_distance_view(&l.as_array(), &r.as_array())
        .map_err(to_py_err)?;
    Ok(PyArray2::from_owned_array(py, result).unbind())
}

/// Compute pairwise cosine similarity between rows of two matrices.
#[pyfunction]
pub fn pairwise_cosine_similarity<'py>(
    py: Python<'py>,
    left: &Bound<'py, PyArray2<f64>>,
    right: &Bound<'py, PyArray2<f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    utils::require_contiguous(left)?;
    utils::require_contiguous(right)?;
    let l = left.readonly();
    let r = right.readonly();
    let result =
        nabled_linalg::vector::pairwise_cosine_similarity_view(&l.as_array(), &r.as_array())
            .map_err(to_py_err)?;
    Ok(PyArray2::from_owned_array(py, result).unbind())
}
