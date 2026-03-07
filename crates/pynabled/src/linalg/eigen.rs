//! Eigenvalue decomposition bindings for Python.

use num_complex::Complex;
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;

use crate::error::to_py_err;

/// Symmetric eigenvalue decomposition. Returns (eigenvalues, eigenvectors).
#[pyfunction(name = "eigen_symmetric")]
pub fn symmetric<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyArray2<f64>>,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray2<f64>>)> {
    let arr = matrix.readonly();
    let result = nabled_linalg::eigen::symmetric(&arr.as_array().to_owned()).map_err(to_py_err)?;
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
    let a = matrix_a.readonly();
    let b = matrix_b.readonly();
    let result =
        nabled_linalg::eigen::generalized(&a.as_array().to_owned(), &b.as_array().to_owned())
            .map_err(to_py_err)?;
    Ok((
        PyArray1::from_owned_array(py, result.eigenvalues).unbind(),
        PyArray2::from_owned_array(py, result.eigenvectors).unbind(),
    ))
}

/// Non-symmetric eigenvalue decomposition. Returns (eigenvalues_real, eigenvalues_imag,
/// schur_vectors_real, schur_vectors_imag).
#[pyfunction(name = "eigen_nonsymmetric")]
pub fn nonsymmetric<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyArray2<f64>>,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray2<f64>>, Py<PyArray2<f64>>)> {
    let arr = matrix.readonly();
    let result =
        nabled_linalg::eigen::nonsymmetric(&arr.as_array().to_owned()).map_err(to_py_err)?;
    let re: Vec<f64> = result.eigenvalues.iter().map(|c| c.re).collect();
    let im: Vec<f64> = result.eigenvalues.iter().map(|c| c.im).collect();
    let rows = result.schur_vectors.nrows();
    let cols = result.schur_vectors.ncols();
    let vec_re: Vec<f64> = result.schur_vectors.iter().map(|c: &Complex<f64>| c.re).collect();
    let vec_im: Vec<f64> = result.schur_vectors.iter().map(|c: &Complex<f64>| c.im).collect();
    let schur_re = ndarray::Array2::from_shape_vec((rows, cols), vec_re).unwrap();
    let schur_im = ndarray::Array2::from_shape_vec((rows, cols), vec_im).unwrap();
    Ok((
        PyArray1::from_owned_array(py, ndarray::Array1::from_vec(re)).unbind(),
        PyArray1::from_owned_array(py, ndarray::Array1::from_vec(im)).unbind(),
        PyArray2::from_owned_array(py, schur_re).unbind(),
        PyArray2::from_owned_array(py, schur_im).unbind(),
    ))
}
