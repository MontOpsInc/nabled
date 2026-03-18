//! Iterative solver bindings for Python (dense CG, GMRES).

use nabled_ml::iterative::IterativeConfig;
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;

use crate::error::to_py_err;
use crate::utils;

/// Conjugate gradient solve for symmetric positive definite Ax = b.
#[pyfunction(name = "conjugate_gradient")]
pub fn conjugate_gradient<'py>(
    py: Python<'py>,
    matrix_a: &Bound<'py, PyArray2<f64>>,
    matrix_b: &Bound<'py, PyArray1<f64>>,
    tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<Py<PyArray1<f64>>> {
    utils::require_contiguous(matrix_a)?;
    utils::require_contiguous(matrix_b)?;
    let a = matrix_a.readonly();
    let b = matrix_b.readonly();
    let config = IterativeConfig {
        tolerance:      tolerance.unwrap_or(1e-10),
        max_iterations: max_iterations.unwrap_or(1000),
    };
    let result =
        nabled_ml::iterative::conjugate_gradient_view(&a.as_array(), &b.as_array(), &config)
            .map_err(to_py_err)?;
    Ok(PyArray1::from_owned_array(py, result).unbind())
}

/// GMRES solve for general Ax = b.
#[pyfunction(name = "gmres")]
pub fn gmres<'py>(
    py: Python<'py>,
    matrix_a: &Bound<'py, PyArray2<f64>>,
    matrix_b: &Bound<'py, PyArray1<f64>>,
    tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<Py<PyArray1<f64>>> {
    utils::require_contiguous(matrix_a)?;
    utils::require_contiguous(matrix_b)?;
    let a = matrix_a.readonly();
    let b = matrix_b.readonly();
    let config = IterativeConfig {
        tolerance:      tolerance.unwrap_or(1e-10),
        max_iterations: max_iterations.unwrap_or(1000),
    };
    let result = nabled_ml::iterative::gmres_view(&a.as_array(), &b.as_array(), &config)
        .map_err(to_py_err)?;
    Ok(PyArray1::from_owned_array(py, result).unbind())
}
