//! Eigenvalue decomposition bindings for Python.

use num_complex::Complex64;
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;

use crate::error::to_py_err;
use crate::utils;

/// Symmetric eigenvalue decomposition. Returns (eigenvalues, eigenvectors).
#[pyfunction(name = "eigen_symmetric")]
pub fn symmetric<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyArray2<f64>>,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray2<f64>>)> {
    utils::require_contiguous(matrix)?;
    let arr = matrix.readonly();
    let result = nabled_linalg::eigen::symmetric_view(&arr.as_array()).map_err(to_py_err)?;
    Ok((
        PyArray1::from_owned_array(py, result.eigenvalues).unbind(),
        PyArray2::from_owned_array(py, result.eigenvectors).unbind(),
    ))
}

/// Generalized eigenvalue decomposition. Returns (eigenvalues, eigenvectors).
#[pyfunction(name = "eigen_generalized")]
pub fn generalized<'py>(
    py: Python<'py>,
    matrix_a: &Bound<'py, PyArray2<f64>>,
    matrix_b: &Bound<'py, PyArray2<f64>>,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray2<f64>>)> {
    utils::require_contiguous(matrix_a)?;
    utils::require_contiguous(matrix_b)?;
    let a = matrix_a.readonly();
    let b = matrix_b.readonly();
    let result =
        nabled_linalg::eigen::generalized_view(&a.as_array(), &b.as_array()).map_err(to_py_err)?;
    Ok((
        PyArray1::from_owned_array(py, result.eigenvalues).unbind(),
        PyArray2::from_owned_array(py, result.eigenvectors).unbind(),
    ))
}

/// Non-symmetric eigenvalue decomposition. Returns (eigenvalues, schur_vectors).
#[pyfunction(name = "eigen_nonsymmetric")]
pub fn nonsymmetric<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyArray2<f64>>,
) -> PyResult<(Py<PyArray1<Complex64>>, Py<PyArray2<Complex64>>)> {
    utils::require_contiguous(matrix)?;
    let arr = matrix.readonly();
    let result = nabled_linalg::eigen::nonsymmetric_view(&arr.as_array()).map_err(to_py_err)?;
    Ok((
        PyArray1::from_owned_array(py, result.eigenvalues).unbind(),
        PyArray2::from_owned_array(py, result.schur_vectors).unbind(),
    ))
}
