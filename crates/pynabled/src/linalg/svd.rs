//! SVD bindings for Python.

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;

use crate::error::to_py_err;

/// Compute the SVD of a matrix. Returns (U, singular_values, Vt).
#[pyfunction(name = "svd_decompose")]
pub fn decompose<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyArray2<f64>>,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray1<f64>>, Py<PyArray2<f64>>)> {
    let arr = a.readonly();
    let view = arr.as_array();
    let result = nabled_linalg::svd::decompose(&view.to_owned()).map_err(to_py_err)?;
    Ok((
        PyArray2::from_owned_array(py, result.u).unbind(),
        PyArray1::from_owned_array(py, result.singular_values).unbind(),
        PyArray2::from_owned_array(py, result.vt).unbind(),
    ))
}

/// Compute truncated SVD with k components.
#[pyfunction(name = "svd_decompose_truncated")]
pub fn decompose_truncated<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyArray2<f64>>,
    k: usize,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray1<f64>>, Py<PyArray2<f64>>)> {
    let arr = a.readonly();
    let view = arr.as_array();
    let result = nabled_linalg::svd::decompose_truncated(&view.to_owned(), k).map_err(to_py_err)?;
    Ok((
        PyArray2::from_owned_array(py, result.u).unbind(),
        PyArray1::from_owned_array(py, result.singular_values).unbind(),
        PyArray2::from_owned_array(py, result.vt).unbind(),
    ))
}

/// Compute pseudo-inverse of a matrix.
#[pyfunction(name = "svd_pseudo_inverse")]
pub fn pseudo_inverse<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyArray2<f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    let arr = a.readonly();
    let view = arr.as_array();
    let result = nabled_linalg::svd::pseudo_inverse(
        &view.to_owned(),
        &nabled_linalg::svd::PseudoInverseConfig::default(),
    )
    .map_err(to_py_err)?;
    Ok(PyArray2::from_owned_array(py, result).unbind())
}

/// Reconstruct matrix from SVD components.
#[pyfunction(name = "svd_reconstruct_matrix")]
pub fn reconstruct_matrix<'py>(
    py: Python<'py>,
    u: &Bound<'py, PyArray2<f64>>,
    singular_values: &Bound<'py, PyArray1<f64>>,
    vt: &Bound<'py, PyArray2<f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    let u_arr = u.readonly().as_array().to_owned();
    let s_arr = singular_values.readonly().as_array().to_owned();
    let vt_arr = vt.readonly().as_array().to_owned();
    let svd = nabled_linalg::svd::NdarraySVD {
        u:               u_arr,
        singular_values: s_arr,
        vt:              vt_arr,
    };
    let result = nabled_linalg::svd::reconstruct_matrix(&svd);
    Ok(PyArray2::from_owned_array(py, result).unbind())
}

/// Compute condition number from SVD.
#[pyfunction(name = "svd_condition_number")]
pub fn condition_number(
    u: &Bound<'_, PyArray2<f64>>,
    singular_values: &Bound<'_, PyArray1<f64>>,
    vt: &Bound<'_, PyArray2<f64>>,
) -> PyResult<f64> {
    let svd = nabled_linalg::svd::NdarraySVD {
        u:               u.readonly().as_array().to_owned(),
        singular_values: singular_values.readonly().as_array().to_owned(),
        vt:              vt.readonly().as_array().to_owned(),
    };
    Ok(nabled_linalg::svd::condition_number(&svd))
}

/// Compute numerical rank from singular values.
#[pyfunction(name = "svd_rank", signature = (singular_values, tolerance=None))]
pub fn rank(singular_values: &Bound<'_, PyArray1<f64>>, tolerance: Option<f64>) -> PyResult<usize> {
    let s = singular_values.readonly().as_array().to_owned();
    let len = s.len();
    let svd = nabled_linalg::svd::NdarraySVD {
        u:               ndarray::Array2::zeros((1, len)),
        singular_values: s,
        vt:              ndarray::Array2::zeros((len, 1)),
    };
    Ok(nabled_linalg::svd::rank(&svd, tolerance))
}

/// Compute a basis for the right null-space of a matrix.
#[pyfunction(name = "svd_null_space")]
pub fn null_space<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyArray2<f64>>,
    tolerance: Option<f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let arr = a.readonly();
    let view = arr.as_array();
    let result = nabled_linalg::svd::null_space(&view.to_owned(), tolerance).map_err(to_py_err)?;
    Ok(PyArray2::from_owned_array(py, result).unbind())
}
