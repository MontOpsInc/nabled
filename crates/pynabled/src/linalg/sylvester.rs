//! Sylvester and Lyapunov solver bindings for Python.

use numpy::{PyArray2, PyArrayMethods};
use pyo3::prelude::*;

use crate::error::to_py_err;
use crate::utils;

/// Solve Sylvester equation AX + XB = C.
#[pyfunction(name = "sylvester_solve")]
pub fn solve_sylvester<'py>(
    py: Python<'py>,
    matrix_a: &Bound<'py, PyArray2<f64>>,
    matrix_b: &Bound<'py, PyArray2<f64>>,
    matrix_c: &Bound<'py, PyArray2<f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    utils::require_contiguous(matrix_a)?;
    utils::require_contiguous(matrix_b)?;
    utils::require_contiguous(matrix_c)?;
    let a = matrix_a.readonly();
    let b = matrix_b.readonly();
    let c = matrix_c.readonly();
    let result =
        nabled_linalg::sylvester::solve_sylvester_view(&a.as_array(), &b.as_array(), &c.as_array())
            .map_err(to_py_err)?;
    Ok(PyArray2::from_owned_array(py, result).unbind())
}

/// Solve Lyapunov equation AX + XA^T = Q.
#[pyfunction(name = "lyapunov_solve")]
pub fn solve_lyapunov<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyArray2<f64>>,
    q: &Bound<'py, PyArray2<f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    utils::require_contiguous(a)?;
    utils::require_contiguous(q)?;
    let a_arr = a.readonly();
    let q_arr = q.readonly();
    let result =
        nabled_linalg::sylvester::solve_lyapunov_view(&a_arr.as_array(), &q_arr.as_array())
            .map_err(to_py_err)?;
    Ok(PyArray2::from_owned_array(py, result).unbind())
}
