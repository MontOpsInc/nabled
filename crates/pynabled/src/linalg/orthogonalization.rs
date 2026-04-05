//! Orthogonalization bindings for Python.

use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::error::to_py_err;
use crate::utils;

/// Gram-Schmidt orthogonalization.
#[pyfunction(name = "gram_schmidt")]
pub fn gram_schmidt<'py>(py: Python<'py>, matrix: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
    match utils::numeric_array2(matrix, "matrix")? {
        utils::NumericReadonlyArray2::F32(arr) => {
            let result = nabled_linalg::orthogonalization::gram_schmidt_view(&arr.as_array())
                .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        utils::NumericReadonlyArray2::F64(arr) => {
            let result = nabled_linalg::orthogonalization::gram_schmidt_view(&arr.as_array())
                .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        utils::NumericReadonlyArray2::C64(arr) => {
            let result =
                nabled_linalg::orthogonalization::gram_schmidt_complex_view(&arr.as_array())
                    .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
    }
}

/// Classic Gram-Schmidt orthogonalization.
#[pyfunction(name = "gram_schmidt_classic")]
pub fn gram_schmidt_classic<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match utils::real_array2(matrix, "matrix")? {
        utils::RealReadonlyArray2::F32(arr) => {
            let result =
                nabled_linalg::orthogonalization::gram_schmidt_classic_view(&arr.as_array())
                    .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        utils::RealReadonlyArray2::F64(arr) => {
            let result =
                nabled_linalg::orthogonalization::gram_schmidt_classic_view(&arr.as_array())
                    .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
    }
}
