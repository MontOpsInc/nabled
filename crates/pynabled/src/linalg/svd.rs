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
            let result = nabled_linalg::svd::pseudo_inverse_view(
                &arr.as_array(),
                &nabled_linalg::svd::PseudoInverseConfig::default(),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        utils::RealReadonlyArray2::F64(arr) => {
            let result = nabled_linalg::svd::pseudo_inverse_view(
                &arr.as_array(),
                &nabled_linalg::svd::PseudoInverseConfig::default(),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
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
            let svd = nabled_linalg::svd::NdarraySVD {
                u:               u_arr.as_array().to_owned(),
                singular_values: s_arr.as_array().to_owned(),
                vt:              vt_arr.as_array().to_owned(),
            };
            Ok(utils::pyarray2_from_owned(py, nabled_linalg::svd::reconstruct_matrix(&svd)))
        }
        (
            utils::NumericReadonlyArray2::F64(u_arr),
            utils::RealReadonlyArray1::F64(s_arr),
            utils::NumericReadonlyArray2::F64(vt_arr),
        ) => {
            let svd = nabled_linalg::svd::NdarraySVD {
                u:               u_arr.as_array().to_owned(),
                singular_values: s_arr.as_array().to_owned(),
                vt:              vt_arr.as_array().to_owned(),
            };
            Ok(utils::pyarray2_from_owned(py, nabled_linalg::svd::reconstruct_matrix(&svd)))
        }
        (
            utils::NumericReadonlyArray2::C64(u_arr),
            utils::RealReadonlyArray1::F64(s_arr),
            utils::NumericReadonlyArray2::C64(vt_arr),
        ) => {
            let svd = nabled_linalg::svd::NdarrayComplexSVD {
                u:               u_arr.as_array().to_owned(),
                singular_values: s_arr.as_array().to_owned(),
                vt:              vt_arr.as_array().to_owned(),
            };
            Ok(utils::pyarray2_from_owned(py, nabled_linalg::svd::reconstruct_matrix_complex(&svd)))
        }
        (utils::NumericReadonlyArray2::C64(_), _, utils::NumericReadonlyArray2::C64(_)) => {
            Err(complex_svd_component_error())
        }
        _ => Err(utils::matching_real_dtype_error(&["u", "singular_values", "vt"])),
    }
}

/// Compute condition number from SVD.
#[pyfunction(name = "svd_condition_number")]
pub fn condition_number(
    u: &Bound<'_, PyAny>,
    singular_values: &Bound<'_, PyAny>,
    vt: &Bound<'_, PyAny>,
) -> PyResult<f64> {
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
            let svd = nabled_linalg::svd::NdarraySVD {
                u:               u_arr.as_array().to_owned(),
                singular_values: s_arr.as_array().to_owned(),
                vt:              vt_arr.as_array().to_owned(),
            };
            Ok(nabled_linalg::svd::condition_number(&svd).into())
        }
        (
            utils::NumericReadonlyArray2::F64(u_arr),
            utils::RealReadonlyArray1::F64(s_arr),
            utils::NumericReadonlyArray2::F64(vt_arr),
        ) => {
            let svd = nabled_linalg::svd::NdarraySVD {
                u:               u_arr.as_array().to_owned(),
                singular_values: s_arr.as_array().to_owned(),
                vt:              vt_arr.as_array().to_owned(),
            };
            Ok(nabled_linalg::svd::condition_number(&svd))
        }
        (
            utils::NumericReadonlyArray2::C64(_),
            utils::RealReadonlyArray1::F64(s_arr),
            utils::NumericReadonlyArray2::C64(_),
        ) => {
            let singular_values = s_arr.as_array();
            if singular_values.is_empty() {
                return Ok(0.0);
            }
            let max_sv = singular_values.iter().copied().fold(0.0, f64::max);
            let tolerance = 1.0e-14_f64;
            let min_sv = singular_values
                .iter()
                .copied()
                .filter(|value| *value > tolerance)
                .fold(f64::INFINITY, f64::min);
            if min_sv.is_finite() { Ok(max_sv / min_sv) } else { Ok(f64::INFINITY) }
        }
        (utils::NumericReadonlyArray2::C64(_), _, utils::NumericReadonlyArray2::C64(_)) => {
            Err(complex_svd_component_error())
        }
        _ => Err(utils::matching_real_dtype_error(&["u", "singular_values", "vt"])),
    }
}

/// Compute numerical rank from singular values.
#[pyfunction(name = "svd_rank", signature = (singular_values, tolerance=None))]
pub fn rank(singular_values: &Bound<'_, PyAny>, tolerance: Option<f64>) -> PyResult<usize> {
    match utils::real_array1(singular_values, "singular_values")? {
        utils::RealReadonlyArray1::F32(s_arr) => {
            let s = s_arr.as_array().to_owned();
            let len = s.len();
            let svd = nabled_linalg::svd::NdarraySVD {
                u:               ndarray::Array2::zeros((1, len)),
                singular_values: s,
                vt:              ndarray::Array2::zeros((len, 1)),
            };
            let tolerance =
                tolerance.map(|value| utils::f64_to_f32(value, "tolerance")).transpose()?;
            Ok(nabled_linalg::svd::rank(&svd, tolerance))
        }
        utils::RealReadonlyArray1::F64(s_arr) => {
            let s = s_arr.as_array().to_owned();
            let len = s.len();
            let svd = nabled_linalg::svd::NdarraySVD {
                u:               ndarray::Array2::zeros((1, len)),
                singular_values: s,
                vt:              ndarray::Array2::zeros((len, 1)),
            };
            Ok(nabled_linalg::svd::rank(&svd, tolerance))
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
