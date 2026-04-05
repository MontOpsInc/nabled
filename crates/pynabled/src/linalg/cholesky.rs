//! Cholesky decomposition bindings for Python.

use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::error::to_py_err;
use crate::utils;

/// Compute Cholesky decomposition. Returns L where A = L L^T.
#[pyfunction(name = "cholesky_decompose")]
pub fn decompose<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
    match utils::real_array2(a, "a")? {
        utils::RealReadonlyArray2::F32(arr) => {
            let result =
                nabled_linalg::cholesky::decompose_view(&arr.as_array()).map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result.l))
        }
        utils::RealReadonlyArray2::F64(arr) => {
            let result =
                nabled_linalg::cholesky::decompose_view(&arr.as_array()).map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result.l))
        }
    }
}

/// Solve Ax = b for symmetric positive definite A.
#[pyfunction(name = "cholesky_solve")]
pub fn solve<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match (utils::real_array2(a, "a")?, utils::real_array1(b, "b")?) {
        (utils::RealReadonlyArray2::F32(a_arr), utils::RealReadonlyArray1::F32(b_arr)) => {
            let result = nabled_linalg::cholesky::solve_view(&a_arr.as_array(), &b_arr.as_array())
                .map_err(to_py_err)?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
        (utils::RealReadonlyArray2::F64(a_arr), utils::RealReadonlyArray1::F64(b_arr)) => {
            let result = nabled_linalg::cholesky::solve_view(&a_arr.as_array(), &b_arr.as_array())
                .map_err(to_py_err)?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
        _ => Err(utils::matching_real_dtype_error(&["a", "b"])),
    }
}

/// Compute matrix inverse using Cholesky.
#[pyfunction(name = "cholesky_inverse")]
pub fn inverse<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
    match utils::real_array2(a, "a")? {
        utils::RealReadonlyArray2::F32(arr) => {
            let result =
                nabled_linalg::cholesky::inverse_view(&arr.as_array()).map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        utils::RealReadonlyArray2::F64(arr) => {
            let result =
                nabled_linalg::cholesky::inverse_view(&arr.as_array()).map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
    }
}
