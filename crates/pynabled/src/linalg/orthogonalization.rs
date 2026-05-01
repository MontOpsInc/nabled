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

/// Gram-Schmidt orthogonalization into caller-provided output.
#[pyfunction(name = "gram_schmidt_into")]
pub fn gram_schmidt_into(matrix: &Bound<'_, PyAny>, output: &Bound<'_, PyAny>) -> PyResult<()> {
    match utils::numeric_array2(matrix, "matrix")? {
        utils::NumericReadonlyArray2::F32(arr) => {
            let mut output_arr = utils::output_array2::<f32>(output, "output", "float32")?;
            nabled_linalg::orthogonalization::gram_schmidt_view_into(
                &arr.as_array(),
                &mut output_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        utils::NumericReadonlyArray2::F64(arr) => {
            let mut output_arr = utils::output_array2::<f64>(output, "output", "float64")?;
            nabled_linalg::orthogonalization::gram_schmidt_view_into(
                &arr.as_array(),
                &mut output_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        utils::NumericReadonlyArray2::C64(arr) => {
            let mut output_arr =
                utils::output_array2::<num_complex::Complex64>(output, "output", "complex128")?;
            nabled_linalg::orthogonalization::gram_schmidt_complex_view_into(
                &arr.as_array(),
                &mut output_arr.as_array_mut(),
            )
            .map_err(to_py_err)
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

/// Classic Gram-Schmidt orthogonalization into caller-provided output.
#[pyfunction(name = "gram_schmidt_classic_into")]
pub fn gram_schmidt_classic_into(
    matrix: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
) -> PyResult<()> {
    match utils::real_array2(matrix, "matrix")? {
        utils::RealReadonlyArray2::F32(arr) => {
            let mut output_arr = utils::output_array2::<f32>(output, "output", "float32")?;
            nabled_linalg::orthogonalization::gram_schmidt_classic_view_into(
                &arr.as_array(),
                &mut output_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        utils::RealReadonlyArray2::F64(arr) => {
            let mut output_arr = utils::output_array2::<f64>(output, "output", "float64")?;
            nabled_linalg::orthogonalization::gram_schmidt_classic_view_into(
                &arr.as_array(),
                &mut output_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
    }
}
