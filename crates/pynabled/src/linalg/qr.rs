//! QR decomposition bindings for Python.

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::error::to_py_err;
use crate::utils;

fn qr_config_f32(
    rank_tolerance: Option<f64>,
    max_iterations: Option<usize>,
    use_pivoting: bool,
) -> PyResult<nabled_linalg::qr::QRConfig<f32>> {
    let mut config = nabled_linalg::qr::QRConfig::<f32>::default();
    if let Some(rank_tolerance) = rank_tolerance {
        config.rank_tolerance = utils::f64_to_real(rank_tolerance, "rank_tolerance")?;
    }
    if let Some(max_iterations) = max_iterations {
        config.max_iterations = max_iterations;
    }
    config.use_pivoting = use_pivoting;
    Ok(config)
}

fn qr_config_f64(
    rank_tolerance: Option<f64>,
    max_iterations: Option<usize>,
    use_pivoting: bool,
) -> nabled_linalg::qr::QRConfig<f64> {
    let mut config = nabled_linalg::qr::QRConfig::<f64>::default();
    if let Some(rank_tolerance) = rank_tolerance {
        config.rank_tolerance = rank_tolerance;
    }
    if let Some(max_iterations) = max_iterations {
        config.max_iterations = max_iterations;
    }
    config.use_pivoting = use_pivoting;
    config
}

fn qr_result_tuple<T: numpy::Element>(
    py: Python<'_>,
    result: nabled_linalg::qr::QRResult<T>,
) -> (Py<PyAny>, Py<PyAny>, usize) {
    (
        utils::pyarray2_from_owned(py, result.q),
        utils::pyarray2_from_owned(py, result.r),
        result.rank,
    )
}

fn qr_pivoted_result_tuple<T: numpy::Element>(
    py: Python<'_>,
    result: nabled_linalg::qr::QRResult<T>,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>, usize)> {
    let permutation = result
        .p
        .ok_or_else(|| PyTypeError::new_err("internal QR pivoting result missing permutation"))?;
    Ok((
        utils::pyarray2_from_owned(py, result.q),
        utils::pyarray2_from_owned(py, result.r),
        utils::pyarray2_from_owned(py, permutation),
        result.rank,
    ))
}

/// Compute QR decomposition. Returns `(Q, R, rank)`.
#[pyfunction(name = "qr_decompose", signature = (a, rank_tolerance=None, max_iterations=None))]
pub fn decompose<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    rank_tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<(Py<PyAny>, Py<PyAny>, usize)> {
    match utils::numeric_array2(a, "a")? {
        utils::NumericReadonlyArray2::F32(arr) => {
            let config = qr_config_f32(rank_tolerance, max_iterations, false)?;
            let result =
                nabled_linalg::qr::decompose_view(&arr.as_array(), &config).map_err(to_py_err)?;
            Ok(qr_result_tuple(py, result))
        }
        utils::NumericReadonlyArray2::F64(arr) => {
            let config = qr_config_f64(rank_tolerance, max_iterations, false);
            let result =
                nabled_linalg::qr::decompose_view(&arr.as_array(), &config).map_err(to_py_err)?;
            Ok(qr_result_tuple(py, result))
        }
        utils::NumericReadonlyArray2::C64(arr) => {
            let config = qr_config_f64(rank_tolerance, max_iterations, false);
            let result = nabled_linalg::qr::decompose_complex_view(&arr.as_array(), &config)
                .map_err(to_py_err)?;
            Ok(qr_result_tuple(py, result))
        }
    }
}

/// Compute reduced (economy) QR decomposition. Returns `(Q, R, rank)`.
#[pyfunction(
    name = "qr_decompose_reduced",
    signature = (a, rank_tolerance=None, max_iterations=None)
)]
pub fn decompose_reduced<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    rank_tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<(Py<PyAny>, Py<PyAny>, usize)> {
    match utils::real_array2(a, "a")? {
        utils::RealReadonlyArray2::F32(arr) => {
            let config = qr_config_f32(rank_tolerance, max_iterations, false)?;
            let result = nabled_linalg::qr::decompose_reduced_view(&arr.as_array(), &config)
                .map_err(to_py_err)?;
            Ok(qr_result_tuple(py, result))
        }
        utils::RealReadonlyArray2::F64(arr) => {
            let config = qr_config_f64(rank_tolerance, max_iterations, false);
            let result = nabled_linalg::qr::decompose_reduced_view(&arr.as_array(), &config)
                .map_err(to_py_err)?;
            Ok(qr_result_tuple(py, result))
        }
    }
}

/// Compute QR decomposition with column pivoting. Returns `(Q, R, P, rank)`.
#[pyfunction(
    name = "qr_decompose_pivoted",
    signature = (a, rank_tolerance=None, max_iterations=None)
)]
pub fn decompose_pivoted<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    rank_tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>, usize)> {
    match utils::numeric_array2(a, "a")? {
        utils::NumericReadonlyArray2::F32(arr) => {
            let config = qr_config_f32(rank_tolerance, max_iterations, true)?;
            let result = nabled_linalg::qr::decompose_with_pivoting_view(&arr.as_array(), &config)
                .map_err(to_py_err)?;
            qr_pivoted_result_tuple(py, result)
        }
        utils::NumericReadonlyArray2::F64(arr) => {
            let config = qr_config_f64(rank_tolerance, max_iterations, true);
            let result = nabled_linalg::qr::decompose_with_pivoting_view(&arr.as_array(), &config)
                .map_err(to_py_err)?;
            qr_pivoted_result_tuple(py, result)
        }
        utils::NumericReadonlyArray2::C64(arr) => {
            let config = qr_config_f64(rank_tolerance, max_iterations, true);
            let result = nabled_linalg::qr::decompose_complex_view(&arr.as_array(), &config)
                .map_err(to_py_err)?;
            qr_pivoted_result_tuple(py, result)
        }
    }
}

/// Solve least-squares problem `min ||Ax - b||`.
#[pyfunction(
    name = "qr_solve_least_squares",
    signature = (a, b, rank_tolerance=None, max_iterations=None)
)]
pub fn solve_least_squares<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
    rank_tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<Py<PyAny>> {
    match (utils::real_array2(a, "a")?, utils::real_array1(b, "b")?) {
        (utils::RealReadonlyArray2::F32(a_arr), utils::RealReadonlyArray1::F32(b_arr)) => {
            let config = qr_config_f32(rank_tolerance, max_iterations, false)?;
            let result = nabled_linalg::qr::solve_least_squares_view(
                &a_arr.as_array(),
                &b_arr.as_array(),
                &config,
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
        (utils::RealReadonlyArray2::F64(a_arr), utils::RealReadonlyArray1::F64(b_arr)) => {
            let config = qr_config_f64(rank_tolerance, max_iterations, false);
            let result = nabled_linalg::qr::solve_least_squares_view(
                &a_arr.as_array(),
                &b_arr.as_array(),
                &config,
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
        _ => Err(utils::matching_real_dtype_error(&["a", "b"])),
    }
}
