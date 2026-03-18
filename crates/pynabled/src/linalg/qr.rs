//! QR decomposition bindings for Python.

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;

use crate::error::to_py_err;
use crate::utils;

/// Compute QR decomposition. Returns (Q, R, rank).
#[pyfunction(name = "qr_decompose")]
pub fn decompose<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyArray2<f64>>,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray2<f64>>, usize)> {
    utils::require_contiguous(a)?;
    let arr = a.readonly();
    let view = arr.as_array();
    let config = nabled_linalg::qr::QRConfig::<f64>::default();
    let result = nabled_linalg::qr::decompose_view(&view, &config).map_err(to_py_err)?;
    Ok((
        PyArray2::from_owned_array(py, result.q).unbind(),
        PyArray2::from_owned_array(py, result.r).unbind(),
        result.rank,
    ))
}

/// Solve least-squares problem min ||Ax - b||.
#[pyfunction(name = "qr_solve_least_squares")]
pub fn solve_least_squares<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyArray2<f64>>,
    b: &Bound<'py, PyArray1<f64>>,
) -> PyResult<Py<PyArray1<f64>>> {
    utils::require_contiguous(a)?;
    utils::require_contiguous(b)?;
    let a_arr = a.readonly();
    let b_arr = b.readonly();
    let config = nabled_linalg::qr::QRConfig::<f64>::default();
    let result =
        nabled_linalg::qr::solve_least_squares_view(&a_arr.as_array(), &b_arr.as_array(), &config)
            .map_err(to_py_err)?;
    Ok(PyArray1::from_owned_array(py, result).unbind())
}
