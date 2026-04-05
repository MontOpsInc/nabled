//! QR decomposition bindings for Python.

use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::error::to_py_err;
use crate::utils;

/// Compute QR decomposition. Returns (Q, R, rank).
#[pyfunction(name = "qr_decompose")]
pub fn decompose<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
) -> PyResult<(Py<PyAny>, Py<PyAny>, usize)> {
    match utils::real_array2(a, "a")? {
        utils::RealReadonlyArray2::F32(arr) => {
            let config = nabled_linalg::qr::QRConfig::<f32>::default();
            let result =
                nabled_linalg::qr::decompose_view(&arr.as_array(), &config).map_err(to_py_err)?;
            Ok((
                utils::pyarray2_from_owned(py, result.q),
                utils::pyarray2_from_owned(py, result.r),
                result.rank,
            ))
        }
        utils::RealReadonlyArray2::F64(arr) => {
            let config = nabled_linalg::qr::QRConfig::<f64>::default();
            let result =
                nabled_linalg::qr::decompose_view(&arr.as_array(), &config).map_err(to_py_err)?;
            Ok((
                utils::pyarray2_from_owned(py, result.q),
                utils::pyarray2_from_owned(py, result.r),
                result.rank,
            ))
        }
    }
}

/// Solve least-squares problem min ||Ax - b||.
#[pyfunction(name = "qr_solve_least_squares")]
pub fn solve_least_squares<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match (utils::real_array2(a, "a")?, utils::real_array1(b, "b")?) {
        (utils::RealReadonlyArray2::F32(a_arr), utils::RealReadonlyArray1::F32(b_arr)) => {
            let config = nabled_linalg::qr::QRConfig::<f32>::default();
            let result = nabled_linalg::qr::solve_least_squares_view(
                &a_arr.as_array(),
                &b_arr.as_array(),
                &config,
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
        (utils::RealReadonlyArray2::F64(a_arr), utils::RealReadonlyArray1::F64(b_arr)) => {
            let config = nabled_linalg::qr::QRConfig::<f64>::default();
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
