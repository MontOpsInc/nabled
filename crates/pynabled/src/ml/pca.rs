//! PCA bindings for Python.

use ndarray::{Array1, Array2};
use num_complex::Complex64;
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;

use crate::error::to_py_err;
use crate::utils;

/// Compute PCA. Returns (components, explained_variance, explained_variance_ratio, mean, scores).
#[pyfunction]
pub fn compute_pca<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyArray2<f64>>,
    n_components: Option<usize>,
) -> PyResult<(
    Py<PyArray2<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray2<f64>>,
)> {
    utils::require_contiguous(x)?;
    let x_arr = x.readonly();
    let result =
        nabled_ml::pca::compute_pca_view(&x_arr.as_array(), n_components).map_err(to_py_err)?;
    Ok((
        PyArray2::from_owned_array(py, result.components).unbind(),
        PyArray1::from_owned_array(py, result.explained_variance).unbind(),
        PyArray1::from_owned_array(py, result.explained_variance_ratio).unbind(),
        PyArray1::from_owned_array(py, result.mean).unbind(),
        PyArray2::from_owned_array(py, result.scores).unbind(),
    ))
}

fn real_pca_result(
    components: Array2<f64>,
    mean: Array1<f64>,
) -> nabled_ml::pca::NdarrayPCAResult<f64> {
    nabled_ml::pca::NdarrayPCAResult {
        components,
        explained_variance: Array1::zeros(0),
        explained_variance_ratio: Array1::zeros(0),
        mean,
        scores: Array2::zeros((0, 0)),
    }
}

fn complex_pca_result(
    components: Array2<Complex64>,
    mean: Array1<Complex64>,
) -> nabled_ml::pca::NdarrayComplexPCAResult {
    nabled_ml::pca::NdarrayComplexPCAResult {
        components,
        explained_variance: Array1::zeros(0),
        explained_variance_ratio: Array1::zeros(0),
        mean,
        scores: Array2::zeros((0, 0)),
    }
}

/// Compute PCA for a complex matrix. Returns (components, explained_variance,
/// explained_variance_ratio, mean, scores).
#[pyfunction]
pub fn compute_pca_complex<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyArray2<Complex64>>,
    n_components: Option<usize>,
) -> PyResult<(
    Py<PyArray2<Complex64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<Complex64>>,
    Py<PyArray2<Complex64>>,
)> {
    utils::require_contiguous(x)?;
    let x_arr = x.readonly();
    let result = nabled_ml::pca::compute_pca_complex_view(&x_arr.as_array(), n_components)
        .map_err(to_py_err)?;
    Ok((
        PyArray2::from_owned_array(py, result.components).unbind(),
        PyArray1::from_owned_array(py, result.explained_variance).unbind(),
        PyArray1::from_owned_array(py, result.explained_variance_ratio).unbind(),
        PyArray1::from_owned_array(py, result.mean).unbind(),
        PyArray2::from_owned_array(py, result.scores).unbind(),
    ))
}

/// Project data to PCA score space using previously returned components and mean.
#[pyfunction]
pub fn pca_transform<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyArray2<f64>>,
    components: &Bound<'py, PyArray2<f64>>,
    mean: &Bound<'py, PyArray1<f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    utils::require_contiguous(x)?;
    utils::require_contiguous(components)?;
    utils::require_contiguous(mean)?;
    let result = nabled_ml::pca::transform_view(
        &x.readonly().as_array(),
        &real_pca_result(
            components.readonly().as_array().to_owned(),
            mean.readonly().as_array().to_owned(),
        ),
    );
    Ok(PyArray2::from_owned_array(py, result).unbind())
}

/// Project complex data to PCA score space using previously returned components and mean.
#[pyfunction]
pub fn pca_transform_complex<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyArray2<Complex64>>,
    components: &Bound<'py, PyArray2<Complex64>>,
    mean: &Bound<'py, PyArray1<Complex64>>,
) -> PyResult<Py<PyArray2<Complex64>>> {
    utils::require_contiguous(x)?;
    utils::require_contiguous(components)?;
    utils::require_contiguous(mean)?;
    let result = nabled_ml::pca::transform_complex_view(
        &x.readonly().as_array(),
        &complex_pca_result(
            components.readonly().as_array().to_owned(),
            mean.readonly().as_array().to_owned(),
        ),
    );
    Ok(PyArray2::from_owned_array(py, result).unbind())
}

/// Reconstruct inputs from PCA scores using previously returned components and mean.
#[pyfunction]
pub fn pca_inverse_transform<'py>(
    py: Python<'py>,
    scores: &Bound<'py, PyArray2<f64>>,
    components: &Bound<'py, PyArray2<f64>>,
    mean: &Bound<'py, PyArray1<f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    utils::require_contiguous(scores)?;
    utils::require_contiguous(components)?;
    utils::require_contiguous(mean)?;
    let result = nabled_ml::pca::inverse_transform_view(
        &scores.readonly().as_array(),
        &real_pca_result(
            components.readonly().as_array().to_owned(),
            mean.readonly().as_array().to_owned(),
        ),
    );
    Ok(PyArray2::from_owned_array(py, result).unbind())
}

/// Reconstruct complex inputs from PCA scores using previously returned components and mean.
#[pyfunction]
pub fn pca_inverse_transform_complex<'py>(
    py: Python<'py>,
    scores: &Bound<'py, PyArray2<Complex64>>,
    components: &Bound<'py, PyArray2<Complex64>>,
    mean: &Bound<'py, PyArray1<Complex64>>,
) -> PyResult<Py<PyArray2<Complex64>>> {
    utils::require_contiguous(scores)?;
    utils::require_contiguous(components)?;
    utils::require_contiguous(mean)?;
    let result = nabled_ml::pca::inverse_transform_complex_view(
        &scores.readonly().as_array(),
        &complex_pca_result(
            components.readonly().as_array().to_owned(),
            mean.readonly().as_array().to_owned(),
        ),
    );
    Ok(PyArray2::from_owned_array(py, result).unbind())
}
