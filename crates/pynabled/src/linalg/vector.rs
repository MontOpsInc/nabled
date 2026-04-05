//! Vector primitives bindings for Python.

use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::error::to_py_err;
use crate::utils;

/// Compute dot product of two vectors.
#[pyfunction]
pub fn dot<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match (utils::numeric_array1(a, "a")?, utils::numeric_array1(b, "b")?) {
        (utils::NumericReadonlyArray1::F32(a_arr), utils::NumericReadonlyArray1::F32(b_arr)) => {
            let result = nabled_linalg::vector::dot_view(&a_arr.as_array(), &b_arr.as_array())
                .map_err(to_py_err)?;
            Ok(utils::py_float(py, result.into()))
        }
        (utils::NumericReadonlyArray1::F64(a_arr), utils::NumericReadonlyArray1::F64(b_arr)) => {
            let result = nabled_linalg::vector::dot_view(&a_arr.as_array(), &b_arr.as_array())
                .map_err(to_py_err)?;
            Ok(utils::py_float(py, result))
        }
        (utils::NumericReadonlyArray1::C64(a_arr), utils::NumericReadonlyArray1::C64(b_arr)) => {
            let result =
                nabled_linalg::vector::dot_hermitian_view(&a_arr.as_array(), &b_arr.as_array())
                    .map_err(to_py_err)?;
            Ok(utils::py_complex(py, result))
        }
        _ => Err(utils::matching_numeric_dtype_error(&["a", "b"])),
    }
}

/// Compute L2 norm of a vector.
#[pyfunction]
pub fn l2_norm(v: &Bound<'_, PyAny>) -> PyResult<f64> {
    match utils::numeric_array1(v, "v")? {
        utils::NumericReadonlyArray1::F32(arr) => {
            Ok(nabled_linalg::vector::l2_norm_view(&arr.as_array()).map_err(to_py_err)?.into())
        }
        utils::NumericReadonlyArray1::F64(arr) => {
            nabled_linalg::vector::l2_norm_view(&arr.as_array()).map_err(to_py_err)
        }
        utils::NumericReadonlyArray1::C64(arr) => {
            nabled_linalg::vector::l2_norm_complex_view(&arr.as_array()).map_err(to_py_err)
        }
    }
}

/// Compute cosine similarity between two vectors.
#[pyfunction]
pub fn cosine_similarity<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match (utils::numeric_array1(a, "a")?, utils::numeric_array1(b, "b")?) {
        (utils::NumericReadonlyArray1::F32(a_arr), utils::NumericReadonlyArray1::F32(b_arr)) => {
            let result =
                nabled_linalg::vector::cosine_similarity_view(&a_arr.as_array(), &b_arr.as_array())
                    .map_err(to_py_err)?;
            Ok(utils::py_float(py, result.into()))
        }
        (utils::NumericReadonlyArray1::F64(a_arr), utils::NumericReadonlyArray1::F64(b_arr)) => {
            let result =
                nabled_linalg::vector::cosine_similarity_view(&a_arr.as_array(), &b_arr.as_array())
                    .map_err(to_py_err)?;
            Ok(utils::py_float(py, result))
        }
        (utils::NumericReadonlyArray1::C64(a_arr), utils::NumericReadonlyArray1::C64(b_arr)) => {
            let result = nabled_linalg::vector::cosine_similarity_complex_view(
                &a_arr.as_array(),
                &b_arr.as_array(),
            )
            .map_err(to_py_err)?;
            Ok(utils::py_complex(py, result))
        }
        _ => Err(utils::matching_numeric_dtype_error(&["a", "b"])),
    }
}

/// Compute pairwise L2 distances between rows of two matrices.
#[pyfunction]
pub fn pairwise_l2_distance<'py>(
    py: Python<'py>,
    left: &Bound<'py, PyAny>,
    right: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match (utils::real_array2(left, "left")?, utils::real_array2(right, "right")?) {
        (utils::RealReadonlyArray2::F32(left_arr), utils::RealReadonlyArray2::F32(right_arr)) => {
            let result = nabled_linalg::vector::pairwise_l2_distance_view(
                &left_arr.as_array(),
                &right_arr.as_array(),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        (utils::RealReadonlyArray2::F64(left_arr), utils::RealReadonlyArray2::F64(right_arr)) => {
            let result = nabled_linalg::vector::pairwise_l2_distance_view(
                &left_arr.as_array(),
                &right_arr.as_array(),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        _ => Err(utils::matching_real_dtype_error(&["left", "right"])),
    }
}

/// Compute pairwise cosine similarity between rows of two matrices.
#[pyfunction]
pub fn pairwise_cosine_similarity<'py>(
    py: Python<'py>,
    left: &Bound<'py, PyAny>,
    right: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match (utils::real_array2(left, "left")?, utils::real_array2(right, "right")?) {
        (utils::RealReadonlyArray2::F32(left_arr), utils::RealReadonlyArray2::F32(right_arr)) => {
            let result = nabled_linalg::vector::pairwise_cosine_similarity_view(
                &left_arr.as_array(),
                &right_arr.as_array(),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        (utils::RealReadonlyArray2::F64(left_arr), utils::RealReadonlyArray2::F64(right_arr)) => {
            let result = nabled_linalg::vector::pairwise_cosine_similarity_view(
                &left_arr.as_array(),
                &right_arr.as_array(),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        _ => Err(utils::matching_real_dtype_error(&["left", "right"])),
    }
}
