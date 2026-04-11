//! SVD bindings for Python.

use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::error::to_py_err;
use crate::utils;

fn complex_svd_component_error() -> PyErr {
    pyo3::exceptions::PyTypeError::new_err(
        "u, vt must both be complex128 and singular_values must be float64 for complex SVD results",
    )
}

/// Compute the SVD of a matrix. Returns (U, singular_values, Vt).
#[pyfunction(name = "svd_decompose")]
pub fn decompose<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    match utils::numeric_array2(a, "a")? {
        utils::NumericReadonlyArray2::F32(arr) => {
            let result = nabled_linalg::svd::decompose_view(&arr.as_array()).map_err(to_py_err)?;
            Ok((
                utils::pyarray2_from_owned(py, result.u),
                utils::pyarray1_from_owned(py, result.singular_values),
                utils::pyarray2_from_owned(py, result.vt),
            ))
        }
        utils::NumericReadonlyArray2::F64(arr) => {
            let result = nabled_linalg::svd::decompose_view(&arr.as_array()).map_err(to_py_err)?;
            Ok((
                utils::pyarray2_from_owned(py, result.u),
                utils::pyarray1_from_owned(py, result.singular_values),
                utils::pyarray2_from_owned(py, result.vt),
            ))
        }
        utils::NumericReadonlyArray2::C64(arr) => {
            let result =
                nabled_linalg::svd::decompose_complex_view(&arr.as_array()).map_err(to_py_err)?;
            Ok((
                utils::pyarray2_from_owned(py, result.u),
                utils::pyarray1_from_owned(py, result.singular_values),
                utils::pyarray2_from_owned(py, result.vt),
            ))
        }
    }
}

/// Compute truncated SVD with k components.
#[pyfunction(name = "svd_decompose_truncated")]
pub fn decompose_truncated<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    k: usize,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    match utils::real_array2(a, "a")? {
        utils::RealReadonlyArray2::F32(arr) => {
            let result = nabled_linalg::svd::decompose_truncated_view(&arr.as_array(), k)
                .map_err(to_py_err)?;
            Ok((
                utils::pyarray2_from_owned(py, result.u),
                utils::pyarray1_from_owned(py, result.singular_values),
                utils::pyarray2_from_owned(py, result.vt),
            ))
        }
        utils::RealReadonlyArray2::F64(arr) => {
            let result = nabled_linalg::svd::decompose_truncated_view(&arr.as_array(), k)
                .map_err(to_py_err)?;
            Ok((
                utils::pyarray2_from_owned(py, result.u),
                utils::pyarray1_from_owned(py, result.singular_values),
                utils::pyarray2_from_owned(py, result.vt),
            ))
        }
    }
}

/// Compute pseudo-inverse of a matrix.
#[pyfunction(name = "svd_pseudo_inverse")]
pub fn pseudo_inverse<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
    match utils::real_array2(a, "a")? {
        utils::RealReadonlyArray2::F32(arr) => {
            let mut output =
                ndarray::Array2::<f32>::zeros((arr.as_array().ncols(), arr.as_array().nrows()));
            nabled_linalg::svd::pseudo_inverse_view_into(
                &arr.as_array(),
                &nabled_linalg::svd::PseudoInverseConfig::default(),
                &mut output,
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, output))
        }
        utils::RealReadonlyArray2::F64(arr) => {
            let mut output =
                ndarray::Array2::<f64>::zeros((arr.as_array().ncols(), arr.as_array().nrows()));
            nabled_linalg::svd::pseudo_inverse_view_into(
                &arr.as_array(),
                &nabled_linalg::svd::PseudoInverseConfig::default(),
                &mut output,
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, output))
        }
    }
}

/// Compute pseudo-inverse of a matrix into `output`.
#[pyfunction(name = "svd_pseudo_inverse_into")]
pub fn pseudo_inverse_into(a: &Bound<'_, PyAny>, output: &Bound<'_, PyAny>) -> PyResult<()> {
    match utils::real_array2(a, "a")? {
        utils::RealReadonlyArray2::F32(arr) => {
            let mut out_arr = utils::output_array2::<f32>(output, "output", "float32")?;
            nabled_linalg::svd::pseudo_inverse_view_into(
                &arr.as_array(),
                &nabled_linalg::svd::PseudoInverseConfig::default(),
                &mut out_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        utils::RealReadonlyArray2::F64(arr) => {
            let mut out_arr = utils::output_array2::<f64>(output, "output", "float64")?;
            nabled_linalg::svd::pseudo_inverse_view_into(
                &arr.as_array(),
                &nabled_linalg::svd::PseudoInverseConfig::default(),
                &mut out_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
    }
}

/// Compute pseudo-inverse from precomputed SVD factors.
#[pyfunction(name = "svd_pseudo_inverse_from_factors")]
pub fn pseudo_inverse_from_factors<'py>(
    py: Python<'py>,
    u: &Bound<'py, PyAny>,
    singular_values: &Bound<'py, PyAny>,
    vt: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match (
        utils::numeric_array2(u, "u")?,
        utils::real_array1(singular_values, "singular_values")?,
        utils::numeric_array2(vt, "vt")?,
    ) {
        (
            utils::NumericReadonlyArray2::F32(u_arr),
            utils::RealReadonlyArray1::F32(s_arr),
            utils::NumericReadonlyArray2::F32(vt_arr),
        ) => {
            let mut output = ndarray::Array2::<f32>::zeros((
                vt_arr.as_array().ncols(),
                u_arr.as_array().nrows(),
            ));
            nabled_linalg::svd::pseudo_inverse_from_svd_view_into(
                &u_arr.as_array(),
                &s_arr.as_array(),
                &vt_arr.as_array(),
                &nabled_linalg::svd::PseudoInverseConfig::default(),
                &mut output,
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, output))
        }
        (
            utils::NumericReadonlyArray2::F64(u_arr),
            utils::RealReadonlyArray1::F64(s_arr),
            utils::NumericReadonlyArray2::F64(vt_arr),
        ) => {
            let mut output = ndarray::Array2::<f64>::zeros((
                vt_arr.as_array().ncols(),
                u_arr.as_array().nrows(),
            ));
            nabled_linalg::svd::pseudo_inverse_from_svd_view_into(
                &u_arr.as_array(),
                &s_arr.as_array(),
                &vt_arr.as_array(),
                &nabled_linalg::svd::PseudoInverseConfig::default(),
                &mut output,
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, output))
        }
        (
            utils::NumericReadonlyArray2::C64(u_arr),
            utils::RealReadonlyArray1::F64(s_arr),
            utils::NumericReadonlyArray2::C64(vt_arr),
        ) => {
            let mut output = ndarray::Array2::<num_complex::Complex64>::zeros((
                vt_arr.as_array().ncols(),
                u_arr.as_array().nrows(),
            ));
            nabled_linalg::svd::pseudo_inverse_complex_from_svd_view_into(
                &u_arr.as_array(),
                &s_arr.as_array(),
                &vt_arr.as_array(),
                None,
                &mut output,
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, output))
        }
        (utils::NumericReadonlyArray2::C64(_), _, utils::NumericReadonlyArray2::C64(_)) => {
            Err(complex_svd_component_error())
        }
        _ => Err(utils::matching_real_dtype_error(&["u", "singular_values", "vt"])),
    }
}

/// Compute pseudo-inverse from precomputed SVD factors into `output`.
#[pyfunction(name = "svd_pseudo_inverse_from_factors_into")]
pub fn pseudo_inverse_from_factors_into(
    u: &Bound<'_, PyAny>,
    singular_values: &Bound<'_, PyAny>,
    vt: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
) -> PyResult<()> {
    match (
        utils::numeric_array2(u, "u")?,
        utils::real_array1(singular_values, "singular_values")?,
        utils::numeric_array2(vt, "vt")?,
    ) {
        (
            utils::NumericReadonlyArray2::F32(u_arr),
            utils::RealReadonlyArray1::F32(s_arr),
            utils::NumericReadonlyArray2::F32(vt_arr),
        ) => {
            let mut out_arr = utils::output_array2::<f32>(output, "output", "float32")?;
            nabled_linalg::svd::pseudo_inverse_from_svd_view_into(
                &u_arr.as_array(),
                &s_arr.as_array(),
                &vt_arr.as_array(),
                &nabled_linalg::svd::PseudoInverseConfig::default(),
                &mut out_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        (
            utils::NumericReadonlyArray2::F64(u_arr),
            utils::RealReadonlyArray1::F64(s_arr),
            utils::NumericReadonlyArray2::F64(vt_arr),
        ) => {
            let mut out_arr = utils::output_array2::<f64>(output, "output", "float64")?;
            nabled_linalg::svd::pseudo_inverse_from_svd_view_into(
                &u_arr.as_array(),
                &s_arr.as_array(),
                &vt_arr.as_array(),
                &nabled_linalg::svd::PseudoInverseConfig::default(),
                &mut out_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        (
            utils::NumericReadonlyArray2::C64(u_arr),
            utils::RealReadonlyArray1::F64(s_arr),
            utils::NumericReadonlyArray2::C64(vt_arr),
        ) => {
            let mut out_arr =
                utils::output_array2::<num_complex::Complex64>(output, "output", "complex128")?;
            nabled_linalg::svd::pseudo_inverse_complex_from_svd_view_into(
                &u_arr.as_array(),
                &s_arr.as_array(),
                &vt_arr.as_array(),
                None,
                &mut out_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        (utils::NumericReadonlyArray2::C64(_), _, utils::NumericReadonlyArray2::C64(_)) => {
            Err(complex_svd_component_error())
        }
        _ => Err(utils::matching_real_dtype_error(&["u", "singular_values", "vt"])),
    }
}

/// Reconstruct matrix from SVD components into `output`.
#[pyfunction(name = "svd_reconstruct_matrix_into")]
pub fn reconstruct_matrix_into(
    u: &Bound<'_, PyAny>,
    singular_values: &Bound<'_, PyAny>,
    vt: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
) -> PyResult<()> {
    match (
        utils::numeric_array2(u, "u")?,
        utils::real_array1(singular_values, "singular_values")?,
        utils::numeric_array2(vt, "vt")?,
    ) {
        (
            utils::NumericReadonlyArray2::F32(u_arr),
            utils::RealReadonlyArray1::F32(s_arr),
            utils::NumericReadonlyArray2::F32(vt_arr),
        ) => {
            let mut out_arr = utils::output_array2::<f32>(output, "output", "float32")?;
            nabled_linalg::svd::reconstruct_matrix_view_into(
                &u_arr.as_array(),
                &s_arr.as_array(),
                &vt_arr.as_array(),
                &mut out_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        (
            utils::NumericReadonlyArray2::F64(u_arr),
            utils::RealReadonlyArray1::F64(s_arr),
            utils::NumericReadonlyArray2::F64(vt_arr),
        ) => {
            let mut out_arr = utils::output_array2::<f64>(output, "output", "float64")?;
            nabled_linalg::svd::reconstruct_matrix_view_into(
                &u_arr.as_array(),
                &s_arr.as_array(),
                &vt_arr.as_array(),
                &mut out_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        (
            utils::NumericReadonlyArray2::C64(u_arr),
            utils::RealReadonlyArray1::F64(s_arr),
            utils::NumericReadonlyArray2::C64(vt_arr),
        ) => {
            let mut out_arr =
                utils::output_array2::<num_complex::Complex64>(output, "output", "complex128")?;
            nabled_linalg::svd::reconstruct_matrix_complex_view_into(
                &u_arr.as_array(),
                &s_arr.as_array(),
                &vt_arr.as_array(),
                &mut out_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        (utils::NumericReadonlyArray2::C64(_), _, utils::NumericReadonlyArray2::C64(_)) => {
            Err(complex_svd_component_error())
        }
        _ => Err(utils::matching_real_dtype_error(&["u", "singular_values", "vt"])),
    }
}

/// Reconstruct matrix from SVD components.
#[pyfunction(name = "svd_reconstruct_matrix")]
pub fn reconstruct_matrix<'py>(
    py: Python<'py>,
    u: &Bound<'py, PyAny>,
    singular_values: &Bound<'py, PyAny>,
    vt: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match (
        utils::numeric_array2(u, "u")?,
        utils::real_array1(singular_values, "singular_values")?,
        utils::numeric_array2(vt, "vt")?,
    ) {
        (
            utils::NumericReadonlyArray2::F32(u_arr),
            utils::RealReadonlyArray1::F32(s_arr),
            utils::NumericReadonlyArray2::F32(vt_arr),
        ) => {
            let mut output = ndarray::Array2::<f32>::zeros((
                u_arr.as_array().nrows(),
                vt_arr.as_array().ncols(),
            ));
            nabled_linalg::svd::reconstruct_matrix_view_into(
                &u_arr.as_array(),
                &s_arr.as_array(),
                &vt_arr.as_array(),
                &mut output,
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, output))
        }
        (
            utils::NumericReadonlyArray2::F64(u_arr),
            utils::RealReadonlyArray1::F64(s_arr),
            utils::NumericReadonlyArray2::F64(vt_arr),
        ) => {
            let mut output = ndarray::Array2::<f64>::zeros((
                u_arr.as_array().nrows(),
                vt_arr.as_array().ncols(),
            ));
            nabled_linalg::svd::reconstruct_matrix_view_into(
                &u_arr.as_array(),
                &s_arr.as_array(),
                &vt_arr.as_array(),
                &mut output,
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, output))
        }
        (
            utils::NumericReadonlyArray2::C64(u_arr),
            utils::RealReadonlyArray1::F64(s_arr),
            utils::NumericReadonlyArray2::C64(vt_arr),
        ) => {
            let mut output = ndarray::Array2::<num_complex::Complex64>::zeros((
                u_arr.as_array().nrows(),
                vt_arr.as_array().ncols(),
            ));
            nabled_linalg::svd::reconstruct_matrix_complex_view_into(
                &u_arr.as_array(),
                &s_arr.as_array(),
                &vt_arr.as_array(),
                &mut output,
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, output))
        }
        (utils::NumericReadonlyArray2::C64(_), _, utils::NumericReadonlyArray2::C64(_)) => {
            Err(complex_svd_component_error())
        }
        _ => Err(utils::matching_real_dtype_error(&["u", "singular_values", "vt"])),
    }
}

/// Compute condition number from SVD.
#[pyfunction(name = "svd_condition_number")]
pub fn condition_number(singular_values: &Bound<'_, PyAny>) -> PyResult<f64> {
    match utils::real_array1(singular_values, "singular_values")? {
        utils::RealReadonlyArray1::F32(s_arr) => {
            Ok(nabled_linalg::svd::condition_number_from_singular_values(&s_arr.as_array()).into())
        }
        utils::RealReadonlyArray1::F64(s_arr) => {
            Ok(nabled_linalg::svd::condition_number_from_singular_values(&s_arr.as_array()))
        }
    }
}

/// Compute numerical rank from singular values.
#[pyfunction(name = "svd_rank", signature = (singular_values, tolerance=None))]
pub fn rank(singular_values: &Bound<'_, PyAny>, tolerance: Option<f64>) -> PyResult<usize> {
    match utils::real_array1(singular_values, "singular_values")? {
        utils::RealReadonlyArray1::F32(s_arr) => {
            let tolerance =
                tolerance.map(|value| utils::f64_to_f32(value, "tolerance")).transpose()?;
            Ok(nabled_linalg::svd::rank_from_singular_values(&s_arr.as_array(), tolerance))
        }
        utils::RealReadonlyArray1::F64(s_arr) => {
            Ok(nabled_linalg::svd::rank_from_singular_values(&s_arr.as_array(), tolerance))
        }
    }
}

/// Compute a basis for the right null-space of a matrix.
#[pyfunction(name = "svd_null_space")]
pub fn null_space<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    tolerance: Option<f64>,
) -> PyResult<Py<PyAny>> {
    match utils::real_array2(a, "a")? {
        utils::RealReadonlyArray2::F32(arr) => {
            let tolerance =
                tolerance.map(|value| utils::f64_to_f32(value, "tolerance")).transpose()?;
            let result = nabled_linalg::svd::null_space_view(&arr.as_array(), tolerance)
                .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        utils::RealReadonlyArray2::F64(arr) => {
            let result = nabled_linalg::svd::null_space_view(&arr.as_array(), tolerance)
                .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
    }
}
