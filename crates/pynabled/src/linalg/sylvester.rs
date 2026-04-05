//! Sylvester and Lyapunov solver bindings for Python.

use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::error::to_py_err;
use crate::utils;

/// Solve Sylvester equation AX + XB = C.
#[pyfunction(name = "sylvester_solve")]
pub fn solve_sylvester<'py>(
    py: Python<'py>,
    matrix_a: &Bound<'py, PyAny>,
    matrix_b: &Bound<'py, PyAny>,
    matrix_c: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match (
        utils::numeric_array2(matrix_a, "matrix_a")?,
        utils::numeric_array2(matrix_b, "matrix_b")?,
        utils::numeric_array2(matrix_c, "matrix_c")?,
    ) {
        (
            utils::NumericReadonlyArray2::F32(a),
            utils::NumericReadonlyArray2::F32(b),
            utils::NumericReadonlyArray2::F32(c),
        ) => {
            let result = nabled_linalg::sylvester::solve_sylvester_view(
                &a.as_array(),
                &b.as_array(),
                &c.as_array(),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        (
            utils::NumericReadonlyArray2::F64(a),
            utils::NumericReadonlyArray2::F64(b),
            utils::NumericReadonlyArray2::F64(c),
        ) => {
            let result = nabled_linalg::sylvester::solve_sylvester_view(
                &a.as_array(),
                &b.as_array(),
                &c.as_array(),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        (
            utils::NumericReadonlyArray2::C64(a),
            utils::NumericReadonlyArray2::C64(b),
            utils::NumericReadonlyArray2::C64(c),
        ) => {
            let result = nabled_linalg::sylvester::solve_sylvester_complex_view(
                &a.as_array(),
                &b.as_array(),
                &c.as_array(),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        _ => Err(utils::matching_numeric_dtype_error(&["matrix_a", "matrix_b", "matrix_c"])),
    }
}

/// Solve Lyapunov equation AX + XA^T = Q.
#[pyfunction(name = "lyapunov_solve")]
pub fn solve_lyapunov<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    q: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match (utils::numeric_array2(a, "a")?, utils::numeric_array2(q, "q")?) {
        (utils::NumericReadonlyArray2::F32(a_arr), utils::NumericReadonlyArray2::F32(q_arr)) => {
            let result =
                nabled_linalg::sylvester::solve_lyapunov_view(&a_arr.as_array(), &q_arr.as_array())
                    .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        (utils::NumericReadonlyArray2::F64(a_arr), utils::NumericReadonlyArray2::F64(q_arr)) => {
            let result =
                nabled_linalg::sylvester::solve_lyapunov_view(&a_arr.as_array(), &q_arr.as_array())
                    .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        (utils::NumericReadonlyArray2::C64(a_arr), utils::NumericReadonlyArray2::C64(q_arr)) => {
            let result = nabled_linalg::sylvester::solve_lyapunov_complex_view(
                &a_arr.as_array(),
                &q_arr.as_array(),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        _ => Err(utils::matching_numeric_dtype_error(&["a", "q"])),
    }
}
