//! Triangular solve bindings for Python.

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;

use crate::error::to_py_err;
use crate::utils;

/// Solve lower triangular system Lx = b.
#[pyfunction(name = "triangular_solve_lower")]
pub fn solve_lower<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyArray2<f64>>,
    rhs: &Bound<'py, PyArray1<f64>>,
) -> PyResult<Py<PyArray1<f64>>> {
    utils::require_contiguous(matrix)?;
    utils::require_contiguous(rhs)?;
    let m = matrix.readonly();
    let r = rhs.readonly();
    let result = nabled_linalg::triangular::solve_lower_view(&m.as_array(), &r.as_array())
        .map_err(to_py_err)?;
    Ok(PyArray1::from_owned_array(py, result).unbind())
}

/// Solve upper triangular system Ux = b.
#[pyfunction(name = "triangular_solve_upper")]
pub fn solve_upper<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyArray2<f64>>,
    rhs: &Bound<'py, PyArray1<f64>>,
) -> PyResult<Py<PyArray1<f64>>> {
    utils::require_contiguous(matrix)?;
    utils::require_contiguous(rhs)?;
    let m = matrix.readonly();
    let r = rhs.readonly();
    let result = nabled_linalg::triangular::solve_upper_view(&m.as_array(), &r.as_array())
        .map_err(to_py_err)?;
    Ok(PyArray1::from_owned_array(py, result).unbind())
}

/// Solve lower triangular matrix system LX = B.
#[pyfunction(name = "triangular_solve_lower_matrix")]
pub fn solve_lower_matrix<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyArray2<f64>>,
    rhs: &Bound<'py, PyArray2<f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    utils::require_contiguous(matrix)?;
    utils::require_contiguous(rhs)?;
    let m = matrix.readonly();
    let r = rhs.readonly();
    let result = nabled_linalg::triangular::solve_lower_matrix_view(&m.as_array(), &r.as_array())
        .map_err(to_py_err)?;
    Ok(PyArray2::from_owned_array(py, result).unbind())
}

/// Solve upper triangular matrix system UX = B.
#[pyfunction(name = "triangular_solve_upper_matrix")]
pub fn solve_upper_matrix<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyArray2<f64>>,
    rhs: &Bound<'py, PyArray2<f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    utils::require_contiguous(matrix)?;
    utils::require_contiguous(rhs)?;
    let m = matrix.readonly();
    let r = rhs.readonly();
    let result = nabled_linalg::triangular::solve_upper_matrix_view(&m.as_array(), &r.as_array())
        .map_err(to_py_err)?;
    Ok(PyArray2::from_owned_array(py, result).unbind())
}
