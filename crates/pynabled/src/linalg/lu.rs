//! LU decomposition bindings for Python.

use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::error::to_py_err;
use crate::utils;

/// Compute LU decomposition. Returns `(L, U)`.
#[pyfunction(name = "lu_decompose")]
pub fn decompose<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    match utils::real_array2(a, "a")? {
        utils::RealReadonlyArray2::F32(arr) => {
            let result = nabled_linalg::lu::decompose_view(&arr.as_array()).map_err(to_py_err)?;
            Ok((utils::pyarray2_from_owned(py, result.l), utils::pyarray2_from_owned(py, result.u)))
        }
        utils::RealReadonlyArray2::F64(arr) => {
            let result = nabled_linalg::lu::decompose_view(&arr.as_array()).map_err(to_py_err)?;
            Ok((utils::pyarray2_from_owned(py, result.l), utils::pyarray2_from_owned(py, result.u)))
        }
    }
}

/// Solve `Ax = b` using LU decomposition.
#[pyfunction(name = "lu_solve")]
pub fn solve<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match (utils::numeric_array2(a, "a")?, utils::numeric_array1(b, "b")?) {
        (utils::NumericReadonlyArray2::F32(a_arr), utils::NumericReadonlyArray1::F32(b_arr)) => {
            let result = nabled_linalg::lu::solve_view(&a_arr.as_array(), &b_arr.as_array())
                .map_err(to_py_err)?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
        (utils::NumericReadonlyArray2::F64(a_arr), utils::NumericReadonlyArray1::F64(b_arr)) => {
            let result = nabled_linalg::lu::solve_view(&a_arr.as_array(), &b_arr.as_array())
                .map_err(to_py_err)?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
        (utils::NumericReadonlyArray2::C64(a_arr), utils::NumericReadonlyArray1::C64(b_arr)) => {
            let result =
                nabled_linalg::lu::solve_complex_view(&a_arr.as_array(), &b_arr.as_array())
                    .map_err(to_py_err)?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
        _ => Err(utils::matching_numeric_dtype_error(&["a", "b"])),
    }
}

/// Compute matrix inverse using LU.
#[pyfunction(name = "lu_inverse")]
pub fn inverse<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
    match utils::numeric_array2(a, "a")? {
        utils::NumericReadonlyArray2::F32(arr) => {
            let result = nabled_linalg::lu::inverse_view(&arr.as_array()).map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        utils::NumericReadonlyArray2::F64(arr) => {
            let result = nabled_linalg::lu::inverse_view(&arr.as_array()).map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        utils::NumericReadonlyArray2::C64(arr) => {
            let result =
                nabled_linalg::lu::inverse_complex_view(&arr.as_array()).map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
    }
}

/// Compute determinant.
#[pyfunction(name = "lu_determinant")]
pub fn determinant<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
    match utils::numeric_array2(a, "a")? {
        utils::NumericReadonlyArray2::F32(arr) => {
            let result = nabled_linalg::lu::determinant_view(&arr.as_array()).map_err(to_py_err)?;
            Ok(utils::py_float(py, result.into()))
        }
        utils::NumericReadonlyArray2::F64(arr) => {
            let result = nabled_linalg::lu::determinant_view(&arr.as_array()).map_err(to_py_err)?;
            Ok(utils::py_float(py, result))
        }
        utils::NumericReadonlyArray2::C64(arr) => {
            let result =
                nabled_linalg::lu::determinant_complex_view(&arr.as_array()).map_err(to_py_err)?;
            Ok(utils::py_complex(py, result))
        }
    }
}

/// Compute signed log-determinant. Returns `(sign, ln_abs_det)`.
#[pyfunction(name = "lu_log_determinant")]
pub fn log_determinant<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<(i8, Py<PyAny>)> {
    match utils::real_array2(a, "a")? {
        utils::RealReadonlyArray2::F32(arr) => {
            let result =
                nabled_linalg::lu::log_determinant_view(&arr.as_array()).map_err(to_py_err)?;
            Ok((result.sign, utils::py_float(py, result.ln_abs_det.into())))
        }
        utils::RealReadonlyArray2::F64(arr) => {
            let result =
                nabled_linalg::lu::log_determinant_view(&arr.as_array()).map_err(to_py_err)?;
            Ok((result.sign, utils::py_float(py, result.ln_abs_det)))
        }
    }
}
