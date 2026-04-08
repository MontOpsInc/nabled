//! Dense matrix pipeline bindings for Python.

use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::error::to_py_err;
use crate::utils;

/// Compute matrix-vector product y = A x.
#[pyfunction]
pub fn matvec<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyAny>,
    vector: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match (utils::numeric_array2(matrix, "matrix")?, utils::numeric_array1(vector, "vector")?) {
        (
            utils::NumericReadonlyArray2::F32(matrix_arr),
            utils::NumericReadonlyArray1::F32(vector_arr),
        ) => {
            let result =
                nabled_linalg::matrix::matvec_view(&matrix_arr.as_array(), &vector_arr.as_array())
                    .map_err(to_py_err)?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
        (
            utils::NumericReadonlyArray2::F64(matrix_arr),
            utils::NumericReadonlyArray1::F64(vector_arr),
        ) => {
            let result =
                nabled_linalg::matrix::matvec_view(&matrix_arr.as_array(), &vector_arr.as_array())
                    .map_err(to_py_err)?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
        (
            utils::NumericReadonlyArray2::C64(matrix_arr),
            utils::NumericReadonlyArray1::C64(vector_arr),
        ) => {
            let result = nabled_linalg::matrix::matvec_complex_view(
                &matrix_arr.as_array(),
                &vector_arr.as_array(),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
        _ => Err(utils::matching_numeric_dtype_error(&["matrix", "vector"])),
    }
}

/// Compute matrix-vector product into a caller-provided output array.
#[pyfunction(name = "matvec_into")]
pub fn matvec_into(
    matrix: &Bound<'_, PyAny>,
    vector: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
) -> PyResult<()> {
    match (utils::numeric_array2(matrix, "matrix")?, utils::numeric_array1(vector, "vector")?) {
        (
            utils::NumericReadonlyArray2::F32(matrix_arr),
            utils::NumericReadonlyArray1::F32(vector_arr),
        ) => {
            let mut output_arr = utils::output_array1::<f32>(output, "output", "float32")?;
            nabled_linalg::matrix::matvec_view_into(
                &matrix_arr.as_array(),
                &vector_arr.as_array(),
                output_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        (
            utils::NumericReadonlyArray2::F64(matrix_arr),
            utils::NumericReadonlyArray1::F64(vector_arr),
        ) => {
            let mut output_arr = utils::output_array1::<f64>(output, "output", "float64")?;
            nabled_linalg::matrix::matvec_view_into(
                &matrix_arr.as_array(),
                &vector_arr.as_array(),
                output_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        (
            utils::NumericReadonlyArray2::C64(matrix_arr),
            utils::NumericReadonlyArray1::C64(vector_arr),
        ) => {
            let mut output_arr =
                utils::output_array1::<num_complex::Complex64>(output, "output", "complex128")?;
            nabled_linalg::matrix::matvec_complex_view_into(
                &matrix_arr.as_array(),
                &vector_arr.as_array(),
                output_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        _ => Err(utils::matching_numeric_dtype_error(&["matrix", "vector", "output"])),
    }
}

/// Compute matrix-matrix product C = A B.
#[pyfunction]
pub fn matmat<'py>(
    py: Python<'py>,
    left: &Bound<'py, PyAny>,
    right: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match (utils::numeric_array2(left, "left")?, utils::numeric_array2(right, "right")?) {
        (
            utils::NumericReadonlyArray2::F32(left_arr),
            utils::NumericReadonlyArray2::F32(right_arr),
        ) => {
            let result =
                nabled_linalg::matrix::matmat_view(&left_arr.as_array(), &right_arr.as_array())
                    .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        (
            utils::NumericReadonlyArray2::F64(left_arr),
            utils::NumericReadonlyArray2::F64(right_arr),
        ) => {
            let result =
                nabled_linalg::matrix::matmat_view(&left_arr.as_array(), &right_arr.as_array())
                    .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        (
            utils::NumericReadonlyArray2::C64(left_arr),
            utils::NumericReadonlyArray2::C64(right_arr),
        ) => {
            let result = nabled_linalg::matrix::matmat_complex_view(
                &left_arr.as_array(),
                &right_arr.as_array(),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        _ => Err(utils::matching_numeric_dtype_error(&["left", "right"])),
    }
}

/// Compute matrix-matrix product into a caller-provided output array.
#[pyfunction(name = "matmat_into")]
pub fn matmat_into(
    left: &Bound<'_, PyAny>,
    right: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
) -> PyResult<()> {
    match (utils::numeric_array2(left, "left")?, utils::numeric_array2(right, "right")?) {
        (
            utils::NumericReadonlyArray2::F32(left_arr),
            utils::NumericReadonlyArray2::F32(right_arr),
        ) => {
            let mut output_arr = utils::output_array2::<f32>(output, "output", "float32")?;
            nabled_linalg::matrix::matmat_view_into(
                &left_arr.as_array(),
                &right_arr.as_array(),
                output_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        (
            utils::NumericReadonlyArray2::F64(left_arr),
            utils::NumericReadonlyArray2::F64(right_arr),
        ) => {
            let mut output_arr = utils::output_array2::<f64>(output, "output", "float64")?;
            nabled_linalg::matrix::matmat_view_into(
                &left_arr.as_array(),
                &right_arr.as_array(),
                output_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        (
            utils::NumericReadonlyArray2::C64(left_arr),
            utils::NumericReadonlyArray2::C64(right_arr),
        ) => {
            let mut output_arr =
                utils::output_array2::<num_complex::Complex64>(output, "output", "complex128")?;
            nabled_linalg::matrix::matmat_complex_view_into(
                &left_arr.as_array(),
                &right_arr.as_array(),
                output_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        _ => Err(utils::matching_numeric_dtype_error(&["left", "right", "output"])),
    }
}

/// Batched matrix-vector product.
#[pyfunction]
pub fn batched_row_matvec<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyAny>,
    vectors: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match (utils::real_array2(matrix, "matrix")?, utils::real_array2(vectors, "vectors")?) {
        (
            utils::RealReadonlyArray2::F32(matrix_arr),
            utils::RealReadonlyArray2::F32(vectors_arr),
        ) => {
            let result = nabled_linalg::matrix::batched_row_matvec_view(
                &vectors_arr.as_array(),
                &matrix_arr.as_array(),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        (
            utils::RealReadonlyArray2::F64(matrix_arr),
            utils::RealReadonlyArray2::F64(vectors_arr),
        ) => {
            let result = nabled_linalg::matrix::batched_row_matvec_view(
                &vectors_arr.as_array(),
                &matrix_arr.as_array(),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        _ => Err(utils::matching_real_dtype_error(&["matrix", "vectors"])),
    }
}

/// Batched matrix-vector product into a caller-provided output array.
#[pyfunction(name = "batched_row_matvec_into")]
pub fn batched_row_matvec_into(
    matrix: &Bound<'_, PyAny>,
    vectors: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
) -> PyResult<()> {
    match (utils::real_array2(matrix, "matrix")?, utils::real_array2(vectors, "vectors")?) {
        (
            utils::RealReadonlyArray2::F32(matrix_arr),
            utils::RealReadonlyArray2::F32(vectors_arr),
        ) => {
            let mut output_arr = utils::output_array2::<f32>(output, "output", "float32")?;
            nabled_linalg::matrix::batched_row_matvec_view_into(
                &vectors_arr.as_array(),
                &matrix_arr.as_array(),
                output_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        (
            utils::RealReadonlyArray2::F64(matrix_arr),
            utils::RealReadonlyArray2::F64(vectors_arr),
        ) => {
            let mut output_arr = utils::output_array2::<f64>(output, "output", "float64")?;
            nabled_linalg::matrix::batched_row_matvec_view_into(
                &vectors_arr.as_array(),
                &matrix_arr.as_array(),
                output_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        _ => Err(utils::matching_real_dtype_error(&["matrix", "vectors", "output"])),
    }
}

/// Batched matrix-matrix product.
#[pyfunction]
pub fn batched_matmat<'py>(
    py: Python<'py>,
    left: &Bound<'py, PyAny>,
    right: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match (utils::real_array3(left, "left")?, utils::real_array3(right, "right")?) {
        (utils::RealReadonlyArray3::F32(left_arr), utils::RealReadonlyArray3::F32(right_arr)) => {
            let result = nabled_linalg::matrix::batched_matmat_view(
                &left_arr.as_array(),
                &right_arr.as_array(),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray3_from_owned(py, result))
        }
        (utils::RealReadonlyArray3::F64(left_arr), utils::RealReadonlyArray3::F64(right_arr)) => {
            let result = nabled_linalg::matrix::batched_matmat_view(
                &left_arr.as_array(),
                &right_arr.as_array(),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray3_from_owned(py, result))
        }
        _ => Err(utils::matching_real_dtype_error(&["left", "right"])),
    }
}

/// Batched matrix-matrix product into a caller-provided output array.
#[pyfunction(name = "batched_matmat_into")]
pub fn batched_matmat_into(
    left: &Bound<'_, PyAny>,
    right: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
) -> PyResult<()> {
    match (utils::real_array3(left, "left")?, utils::real_array3(right, "right")?) {
        (utils::RealReadonlyArray3::F32(left_arr), utils::RealReadonlyArray3::F32(right_arr)) => {
            let mut output_arr = utils::output_array3::<f32>(output, "output", "float32")?;
            nabled_linalg::matrix::batched_matmat_view_into(
                &left_arr.as_array(),
                &right_arr.as_array(),
                output_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        (utils::RealReadonlyArray3::F64(left_arr), utils::RealReadonlyArray3::F64(right_arr)) => {
            let mut output_arr = utils::output_array3::<f64>(output, "output", "float64")?;
            nabled_linalg::matrix::batched_matmat_view_into(
                &left_arr.as_array(),
                &right_arr.as_array(),
                output_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        _ => Err(utils::matching_real_dtype_error(&["left", "right", "output"])),
    }
}

/// Batched matrix-matrix product with a broadcast right matrix.
#[pyfunction]
pub fn batched_matmat_broadcast_right<'py>(
    py: Python<'py>,
    left: &Bound<'py, PyAny>,
    right: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match (utils::real_array3(left, "left")?, utils::real_array2(right, "right")?) {
        (utils::RealReadonlyArray3::F32(left_arr), utils::RealReadonlyArray2::F32(right_arr)) => {
            let result = nabled_linalg::matrix::batched_matmat_broadcast_right_view(
                &left_arr.as_array(),
                &right_arr.as_array(),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray3_from_owned(py, result))
        }
        (utils::RealReadonlyArray3::F64(left_arr), utils::RealReadonlyArray2::F64(right_arr)) => {
            let result = nabled_linalg::matrix::batched_matmat_broadcast_right_view(
                &left_arr.as_array(),
                &right_arr.as_array(),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray3_from_owned(py, result))
        }
        _ => Err(utils::matching_real_dtype_error(&["left", "right"])),
    }
}

/// Batched matrix-matrix product with a broadcast right matrix into a caller-provided output array.
#[pyfunction(name = "batched_matmat_broadcast_right_into")]
pub fn batched_matmat_broadcast_right_into(
    left: &Bound<'_, PyAny>,
    right: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
) -> PyResult<()> {
    match (utils::real_array3(left, "left")?, utils::real_array2(right, "right")?) {
        (utils::RealReadonlyArray3::F32(left_arr), utils::RealReadonlyArray2::F32(right_arr)) => {
            let mut output_arr = utils::output_array3::<f32>(output, "output", "float32")?;
            nabled_linalg::matrix::batched_matmat_broadcast_right_view_into(
                &left_arr.as_array(),
                &right_arr.as_array(),
                output_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        (utils::RealReadonlyArray3::F64(left_arr), utils::RealReadonlyArray2::F64(right_arr)) => {
            let mut output_arr = utils::output_array3::<f64>(output, "output", "float64")?;
            nabled_linalg::matrix::batched_matmat_broadcast_right_view_into(
                &left_arr.as_array(),
                &right_arr.as_array(),
                output_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        _ => Err(utils::matching_real_dtype_error(&["left", "right", "output"])),
    }
}

/// Batched matrix-matrix product with a broadcast left matrix.
#[pyfunction]
pub fn batched_matmat_broadcast_left<'py>(
    py: Python<'py>,
    left: &Bound<'py, PyAny>,
    right: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match (utils::real_array2(left, "left")?, utils::real_array3(right, "right")?) {
        (utils::RealReadonlyArray2::F32(left_arr), utils::RealReadonlyArray3::F32(right_arr)) => {
            let result = nabled_linalg::matrix::batched_matmat_broadcast_left_view(
                &left_arr.as_array(),
                &right_arr.as_array(),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray3_from_owned(py, result))
        }
        (utils::RealReadonlyArray2::F64(left_arr), utils::RealReadonlyArray3::F64(right_arr)) => {
            let result = nabled_linalg::matrix::batched_matmat_broadcast_left_view(
                &left_arr.as_array(),
                &right_arr.as_array(),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray3_from_owned(py, result))
        }
        _ => Err(utils::matching_real_dtype_error(&["left", "right"])),
    }
}

/// Batched matrix-matrix product with a broadcast left matrix into a caller-provided output array.
#[pyfunction(name = "batched_matmat_broadcast_left_into")]
pub fn batched_matmat_broadcast_left_into(
    left: &Bound<'_, PyAny>,
    right: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
) -> PyResult<()> {
    match (utils::real_array2(left, "left")?, utils::real_array3(right, "right")?) {
        (utils::RealReadonlyArray2::F32(left_arr), utils::RealReadonlyArray3::F32(right_arr)) => {
            let mut output_arr = utils::output_array3::<f32>(output, "output", "float32")?;
            nabled_linalg::matrix::batched_matmat_broadcast_left_view_into(
                &left_arr.as_array(),
                &right_arr.as_array(),
                output_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        (utils::RealReadonlyArray2::F64(left_arr), utils::RealReadonlyArray3::F64(right_arr)) => {
            let mut output_arr = utils::output_array3::<f64>(output, "output", "float64")?;
            nabled_linalg::matrix::batched_matmat_broadcast_left_view_into(
                &left_arr.as_array(),
                &right_arr.as_array(),
                output_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        _ => Err(utils::matching_real_dtype_error(&["left", "right", "output"])),
    }
}
