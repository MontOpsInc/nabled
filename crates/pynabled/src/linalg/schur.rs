//! Schur decomposition bindings for Python.

use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::error::to_py_err;
use crate::utils;

/// Compute Schur decomposition. Returns (T, Q).
#[pyfunction(name = "schur_compute")]
pub fn compute_schur<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyAny>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    match utils::numeric_array2(matrix, "matrix")? {
        utils::NumericReadonlyArray2::F32(arr) => {
            let result =
                nabled_linalg::schur::compute_schur_view(&arr.as_array()).map_err(to_py_err)?;
            Ok((utils::pyarray2_from_owned(py, result.t), utils::pyarray2_from_owned(py, result.q)))
        }
        utils::NumericReadonlyArray2::F64(arr) => {
            let result =
                nabled_linalg::schur::compute_schur_view(&arr.as_array()).map_err(to_py_err)?;
            Ok((utils::pyarray2_from_owned(py, result.t), utils::pyarray2_from_owned(py, result.q)))
        }
        utils::NumericReadonlyArray2::C64(arr) => {
            let result = nabled_linalg::schur::compute_schur_complex_view(&arr.as_array())
                .map_err(to_py_err)?;
            Ok((utils::pyarray2_from_owned(py, result.t), utils::pyarray2_from_owned(py, result.q)))
        }
    }
}
