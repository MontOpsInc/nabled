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
        utils::real_array2(matrix_a, "matrix_a")?,
        utils::real_array2(matrix_b, "matrix_b")?,
        utils::real_array2(matrix_c, "matrix_c")?,
    ) {
        (
            utils::RealReadonlyArray2::F32(a),
            utils::RealReadonlyArray2::F32(b),
            utils::RealReadonlyArray2::F32(c),
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
            utils::RealReadonlyArray2::F64(a),
            utils::RealReadonlyArray2::F64(b),
            utils::RealReadonlyArray2::F64(c),
        ) => {
            let result = nabled_linalg::sylvester::solve_sylvester_view(
                &a.as_array(),
                &b.as_array(),
                &c.as_array(),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        _ => Err(utils::matching_real_dtype_error(&["matrix_a", "matrix_b", "matrix_c"])),
    }
}

/// Solve Lyapunov equation AX + XA^T = Q.
#[pyfunction(name = "lyapunov_solve")]
pub fn solve_lyapunov<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    q: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match (utils::real_array2(a, "a")?, utils::real_array2(q, "q")?) {
        (utils::RealReadonlyArray2::F32(a_arr), utils::RealReadonlyArray2::F32(q_arr)) => {
            let result =
                nabled_linalg::sylvester::solve_lyapunov_view(&a_arr.as_array(), &q_arr.as_array())
                    .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        (utils::RealReadonlyArray2::F64(a_arr), utils::RealReadonlyArray2::F64(q_arr)) => {
            let result =
                nabled_linalg::sylvester::solve_lyapunov_view(&a_arr.as_array(), &q_arr.as_array())
                    .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        _ => Err(utils::matching_real_dtype_error(&["a", "q"])),
    }
}
