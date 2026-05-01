//! PCA bindings for Python.

use num_complex::Complex64;
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

/// Compute PCA into caller-provided outputs.
#[pyfunction]
pub fn compute_pca_into(
    x: &Bound<'_, PyAny>,
    n_components: Option<usize>,
    components: &Bound<'_, PyAny>,
    explained_variance: &Bound<'_, PyAny>,
    explained_variance_ratio: &Bound<'_, PyAny>,
    mean: &Bound<'_, PyAny>,
    scores: &Bound<'_, PyAny>,
) -> PyResult<()> {
    match utils::real_array2(x, "x")? {
        utils::RealReadonlyArray2::F32(x_arr) => {
            let mut components_arr =
                utils::output_array2::<f32>(components, "components", "float32")?;
            let mut explained_variance_arr =
                utils::output_array1::<f32>(explained_variance, "explained_variance", "float32")?;
            let mut explained_variance_ratio_arr = utils::output_array1::<f32>(
                explained_variance_ratio,
                "explained_variance_ratio",
                "float32",
            )?;
            let mut mean_arr = utils::output_array1::<f32>(mean, "mean", "float32")?;
            let mut scores_arr = utils::output_array2::<f32>(scores, "scores", "float32")?;
            nabled_ml::pca::compute_pca_view_into(
                &x_arr.as_array(),
                n_components,
                components_arr.as_array_mut(),
                explained_variance_arr.as_array_mut(),
                explained_variance_ratio_arr.as_array_mut(),
                mean_arr.as_array_mut(),
                scores_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        utils::RealReadonlyArray2::F64(x_arr) => {
            let mut components_arr =
                utils::output_array2::<f64>(components, "components", "float64")?;
            let mut explained_variance_arr =
                utils::output_array1::<f64>(explained_variance, "explained_variance", "float64")?;
            let mut explained_variance_ratio_arr = utils::output_array1::<f64>(
                explained_variance_ratio,
                "explained_variance_ratio",
                "float64",
            )?;
            let mut mean_arr = utils::output_array1::<f64>(mean, "mean", "float64")?;
            let mut scores_arr = utils::output_array2::<f64>(scores, "scores", "float64")?;
            nabled_ml::pca::compute_pca_view_into(
                &x_arr.as_array(),
                n_components,
                components_arr.as_array_mut(),
                explained_variance_arr.as_array_mut(),
                explained_variance_ratio_arr.as_array_mut(),
                mean_arr.as_array_mut(),
                scores_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
    }
}

/// Compute PCA for a complex matrix. Returns (components, explained_variance,
/// explained_variance_ratio, mean, scores).
#[pyfunction]
pub fn compute_pca_complex<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    n_components: Option<usize>,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    match utils::numeric_array2(x, "x")? {
        utils::NumericReadonlyArray2::C64(x_arr) => {
            let result = nabled_ml::pca::compute_pca_complex_view(&x_arr.as_array(), n_components)
                .map_err(to_py_err)?;
            Ok((
                utils::pyarray2_from_owned(py, result.components),
                utils::pyarray1_from_owned(py, result.explained_variance),
                utils::pyarray1_from_owned(py, result.explained_variance_ratio),
                utils::pyarray1_from_owned(py, result.mean),
                utils::pyarray2_from_owned(py, result.scores),
            ))
        }
        _ => Err(utils::matching_complex_dtype_error(&["x"])),
    }
}

/// Compute PCA for a complex matrix into caller-provided outputs.
#[pyfunction]
pub fn compute_pca_complex_into(
    x: &Bound<'_, PyAny>,
    n_components: Option<usize>,
    components: &Bound<'_, PyAny>,
    explained_variance: &Bound<'_, PyAny>,
    explained_variance_ratio: &Bound<'_, PyAny>,
    mean: &Bound<'_, PyAny>,
    scores: &Bound<'_, PyAny>,
) -> PyResult<()> {
    match utils::numeric_array2(x, "x")? {
        utils::NumericReadonlyArray2::C64(x_arr) => {
            let mut components_arr =
                utils::output_array2::<Complex64>(components, "components", "complex128")?;
            let mut explained_variance_arr =
                utils::output_array1::<f64>(explained_variance, "explained_variance", "float64")?;
            let mut explained_variance_ratio_arr = utils::output_array1::<f64>(
                explained_variance_ratio,
                "explained_variance_ratio",
                "float64",
            )?;
            let mut mean_arr = utils::output_array1::<Complex64>(mean, "mean", "complex128")?;
            let mut scores_arr = utils::output_array2::<Complex64>(scores, "scores", "complex128")?;
            nabled_ml::pca::compute_pca_complex_view_into(
                &x_arr.as_array(),
                n_components,
                components_arr.as_array_mut(),
                explained_variance_arr.as_array_mut(),
                explained_variance_ratio_arr.as_array_mut(),
                mean_arr.as_array_mut(),
                scores_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        _ => Err(utils::matching_complex_dtype_error(&[
            "x",
            "components",
            "explained_variance",
            "explained_variance_ratio",
            "mean",
            "scores",
        ])),
    }
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
            let result = nabled_ml::pca::transform_from_components_view(
                &x_arr.as_array(),
                &components_arr.as_array(),
                &mean_arr.as_array(),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        (
            utils::RealReadonlyArray2::F64(x_arr),
            utils::RealReadonlyArray2::F64(components_arr),
            utils::RealReadonlyArray1::F64(mean_arr),
        ) => {
            let result = nabled_ml::pca::transform_from_components_view(
                &x_arr.as_array(),
                &components_arr.as_array(),
                &mean_arr.as_array(),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        _ => Err(utils::matching_real_dtype_error(&["x", "components", "mean"])),
    }
}

/// Project data to PCA score space into `output`.
#[pyfunction]
pub fn pca_transform_into(
    x: &Bound<'_, PyAny>,
    components: &Bound<'_, PyAny>,
    mean: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
) -> PyResult<()> {
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
            let mut output_arr = utils::output_array2::<f32>(output, "output", "float32")?;
            nabled_ml::pca::transform_from_components_view_into(
                &x_arr.as_array(),
                &components_arr.as_array(),
                &mean_arr.as_array(),
                output_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        (
            utils::RealReadonlyArray2::F64(x_arr),
            utils::RealReadonlyArray2::F64(components_arr),
            utils::RealReadonlyArray1::F64(mean_arr),
        ) => {
            let mut output_arr = utils::output_array2::<f64>(output, "output", "float64")?;
            nabled_ml::pca::transform_from_components_view_into(
                &x_arr.as_array(),
                &components_arr.as_array(),
                &mean_arr.as_array(),
                output_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        _ => Err(utils::matching_real_dtype_error(&["x", "components", "mean", "output"])),
    }
}

/// Project complex data to PCA score space using previously returned components and mean.
#[pyfunction]
pub fn pca_transform_complex<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    components: &Bound<'py, PyAny>,
    mean: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match (
        utils::numeric_array2(x, "x")?,
        utils::numeric_array2(components, "components")?,
        utils::numeric_array1(mean, "mean")?,
    ) {
        (
            utils::NumericReadonlyArray2::C64(x_arr),
            utils::NumericReadonlyArray2::C64(components_arr),
            utils::NumericReadonlyArray1::C64(mean_arr),
        ) => {
            let result = nabled_ml::pca::transform_complex_from_components_view(
                &x_arr.as_array(),
                &components_arr.as_array(),
                &mean_arr.as_array(),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        _ => Err(utils::matching_complex_dtype_error(&["x", "components", "mean"])),
    }
}

/// Project complex data to PCA score space into `output`.
#[pyfunction]
pub fn pca_transform_complex_into(
    x: &Bound<'_, PyAny>,
    components: &Bound<'_, PyAny>,
    mean: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
) -> PyResult<()> {
    match (
        utils::numeric_array2(x, "x")?,
        utils::numeric_array2(components, "components")?,
        utils::numeric_array1(mean, "mean")?,
    ) {
        (
            utils::NumericReadonlyArray2::C64(x_arr),
            utils::NumericReadonlyArray2::C64(components_arr),
            utils::NumericReadonlyArray1::C64(mean_arr),
        ) => {
            let mut output_arr = utils::output_array2::<Complex64>(output, "output", "complex128")?;
            nabled_ml::pca::transform_complex_from_components_view_into(
                &x_arr.as_array(),
                &components_arr.as_array(),
                &mean_arr.as_array(),
                output_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        _ => Err(utils::matching_complex_dtype_error(&["x", "components", "mean", "output"])),
    }
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
            let result = nabled_ml::pca::inverse_transform_from_components_view(
                &scores_arr.as_array(),
                &components_arr.as_array(),
                &mean_arr.as_array(),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        (
            utils::RealReadonlyArray2::F64(scores_arr),
            utils::RealReadonlyArray2::F64(components_arr),
            utils::RealReadonlyArray1::F64(mean_arr),
        ) => {
            let result = nabled_ml::pca::inverse_transform_from_components_view(
                &scores_arr.as_array(),
                &components_arr.as_array(),
                &mean_arr.as_array(),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        _ => Err(utils::matching_real_dtype_error(&["scores", "components", "mean"])),
    }
}

/// Reconstruct inputs from PCA scores into `output`.
#[pyfunction]
pub fn pca_inverse_transform_into(
    scores: &Bound<'_, PyAny>,
    components: &Bound<'_, PyAny>,
    mean: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
) -> PyResult<()> {
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
            let mut output_arr = utils::output_array2::<f32>(output, "output", "float32")?;
            nabled_ml::pca::inverse_transform_from_components_view_into(
                &scores_arr.as_array(),
                &components_arr.as_array(),
                &mean_arr.as_array(),
                output_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        (
            utils::RealReadonlyArray2::F64(scores_arr),
            utils::RealReadonlyArray2::F64(components_arr),
            utils::RealReadonlyArray1::F64(mean_arr),
        ) => {
            let mut output_arr = utils::output_array2::<f64>(output, "output", "float64")?;
            nabled_ml::pca::inverse_transform_from_components_view_into(
                &scores_arr.as_array(),
                &components_arr.as_array(),
                &mean_arr.as_array(),
                output_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        _ => Err(utils::matching_real_dtype_error(&["scores", "components", "mean", "output"])),
    }
}

/// Reconstruct complex inputs from PCA scores using previously returned components and mean.
#[pyfunction]
pub fn pca_inverse_transform_complex<'py>(
    py: Python<'py>,
    scores: &Bound<'py, PyAny>,
    components: &Bound<'py, PyAny>,
    mean: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match (
        utils::numeric_array2(scores, "scores")?,
        utils::numeric_array2(components, "components")?,
        utils::numeric_array1(mean, "mean")?,
    ) {
        (
            utils::NumericReadonlyArray2::C64(scores_arr),
            utils::NumericReadonlyArray2::C64(components_arr),
            utils::NumericReadonlyArray1::C64(mean_arr),
        ) => {
            let result = nabled_ml::pca::inverse_transform_complex_from_components_view(
                &scores_arr.as_array(),
                &components_arr.as_array(),
                &mean_arr.as_array(),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        _ => Err(utils::matching_complex_dtype_error(&["scores", "components", "mean"])),
    }
}

/// Reconstruct complex inputs from PCA scores into `output`.
#[pyfunction]
pub fn pca_inverse_transform_complex_into(
    scores: &Bound<'_, PyAny>,
    components: &Bound<'_, PyAny>,
    mean: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
) -> PyResult<()> {
    match (
        utils::numeric_array2(scores, "scores")?,
        utils::numeric_array2(components, "components")?,
        utils::numeric_array1(mean, "mean")?,
    ) {
        (
            utils::NumericReadonlyArray2::C64(scores_arr),
            utils::NumericReadonlyArray2::C64(components_arr),
            utils::NumericReadonlyArray1::C64(mean_arr),
        ) => {
            let mut output_arr = utils::output_array2::<Complex64>(output, "output", "complex128")?;
            nabled_ml::pca::inverse_transform_complex_from_components_view_into(
                &scores_arr.as_array(),
                &components_arr.as_array(),
                &mean_arr.as_array(),
                output_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        _ => Err(utils::matching_complex_dtype_error(&["scores", "components", "mean", "output"])),
    }
}
