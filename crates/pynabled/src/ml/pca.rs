//! PCA bindings for Python.

use ndarray::{Array1, Array2};
use num_complex::Complex64;
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::error::to_py_err;
use crate::utils;

/// Compute PCA. Returns (components, explained_variance, explained_variance_ratio, mean, scores).
#[pyfunction]
pub fn compute_pca<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    n_components: Option<usize>,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    match utils::real_array2(x, "x")? {
        utils::RealReadonlyArray2::F32(x_arr) => {
            let result = nabled_ml::pca::compute_pca_view(&x_arr.as_array(), n_components)
                .map_err(to_py_err)?;
            Ok((
                utils::pyarray2_from_owned(py, result.components),
                utils::pyarray1_from_owned(py, result.explained_variance),
                utils::pyarray1_from_owned(py, result.explained_variance_ratio),
                utils::pyarray1_from_owned(py, result.mean),
                utils::pyarray2_from_owned(py, result.scores),
            ))
        }
        utils::RealReadonlyArray2::F64(x_arr) => {
            let result = nabled_ml::pca::compute_pca_view(&x_arr.as_array(), n_components)
                .map_err(to_py_err)?;
            Ok((
                utils::pyarray2_from_owned(py, result.components),
                utils::pyarray1_from_owned(py, result.explained_variance),
                utils::pyarray1_from_owned(py, result.explained_variance_ratio),
                utils::pyarray1_from_owned(py, result.mean),
                utils::pyarray2_from_owned(py, result.scores),
            ))
        }
    }
}

fn real_pca_result_f32(
    components: Array2<f32>,
    mean: Array1<f32>,
) -> nabled_ml::pca::NdarrayPCAResult<f32> {
    nabled_ml::pca::NdarrayPCAResult {
        components,
        explained_variance: Array1::<f32>::zeros(0),
        explained_variance_ratio: Array1::<f32>::zeros(0),
        mean,
        scores: Array2::<f32>::zeros((0, 0)),
    }
}

fn real_pca_result_f64(
    components: Array2<f64>,
    mean: Array1<f64>,
) -> nabled_ml::pca::NdarrayPCAResult<f64> {
    nabled_ml::pca::NdarrayPCAResult {
        components,
        explained_variance: Array1::<f64>::zeros(0),
        explained_variance_ratio: Array1::<f64>::zeros(0),
        mean,
        scores: Array2::<f64>::zeros((0, 0)),
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
    x: &Bound<'py, PyAny>,
    components: &Bound<'py, PyAny>,
    mean: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match (
        utils::real_array2(x, "x")?,
        utils::real_array2(components, "components")?,
        utils::real_array1(mean, "mean")?,
    ) {
        (
            utils::RealReadonlyArray2::F32(x_arr),
            utils::RealReadonlyArray2::F32(components_arr),
            utils::RealReadonlyArray1::F32(mean_arr),
        ) => {
            let result = nabled_ml::pca::transform_view(
                &x_arr.as_array(),
                &real_pca_result_f32(
                    components_arr.as_array().to_owned(),
                    mean_arr.as_array().to_owned(),
                ),
            );
            Ok(utils::pyarray2_from_owned(py, result))
        }
        (
            utils::RealReadonlyArray2::F64(x_arr),
            utils::RealReadonlyArray2::F64(components_arr),
            utils::RealReadonlyArray1::F64(mean_arr),
        ) => {
            let result = nabled_ml::pca::transform_view(
                &x_arr.as_array(),
                &real_pca_result_f64(
                    components_arr.as_array().to_owned(),
                    mean_arr.as_array().to_owned(),
                ),
            );
            Ok(utils::pyarray2_from_owned(py, result))
        }
        _ => Err(utils::matching_real_dtype_error(&["x", "components", "mean"])),
    }
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
    scores: &Bound<'py, PyAny>,
    components: &Bound<'py, PyAny>,
    mean: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match (
        utils::real_array2(scores, "scores")?,
        utils::real_array2(components, "components")?,
        utils::real_array1(mean, "mean")?,
    ) {
        (
            utils::RealReadonlyArray2::F32(scores_arr),
            utils::RealReadonlyArray2::F32(components_arr),
            utils::RealReadonlyArray1::F32(mean_arr),
        ) => {
            let result = nabled_ml::pca::inverse_transform_view(
                &scores_arr.as_array(),
                &real_pca_result_f32(
                    components_arr.as_array().to_owned(),
                    mean_arr.as_array().to_owned(),
                ),
            );
            Ok(utils::pyarray2_from_owned(py, result))
        }
        (
            utils::RealReadonlyArray2::F64(scores_arr),
            utils::RealReadonlyArray2::F64(components_arr),
            utils::RealReadonlyArray1::F64(mean_arr),
        ) => {
            let result = nabled_ml::pca::inverse_transform_view(
                &scores_arr.as_array(),
                &real_pca_result_f64(
                    components_arr.as_array().to_owned(),
                    mean_arr.as_array().to_owned(),
                ),
            );
            Ok(utils::pyarray2_from_owned(py, result))
        }
        _ => Err(utils::matching_real_dtype_error(&["scores", "components", "mean"])),
    }
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
