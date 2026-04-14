//! Statistics bindings for Python.

use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::error::to_py_err;
use crate::utils;

/// Compute column means.
#[pyfunction]
pub fn column_means<'py>(py: Python<'py>, matrix: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
    match utils::real_array2(matrix, "matrix")? {
        utils::RealReadonlyArray2::F32(arr) => {
            Ok(utils::pyarray1_from_owned(py, nabled_ml::stats::column_means_view(&arr.as_array())))
        }
        utils::RealReadonlyArray2::F64(arr) => {
            Ok(utils::pyarray1_from_owned(py, nabled_ml::stats::column_means_view(&arr.as_array())))
        }
    }
}

/// Compute column means into caller-provided output.
#[pyfunction]
pub fn column_means_into(matrix: &Bound<'_, PyAny>, output: &Bound<'_, PyAny>) -> PyResult<()> {
    match utils::real_array2(matrix, "matrix")? {
        utils::RealReadonlyArray2::F32(arr) => {
            let mut output_arr = utils::output_array1::<f32>(output, "output", "float32")?;
            nabled_ml::stats::column_means_view_into(
                &arr.as_array(),
                &mut output_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        utils::RealReadonlyArray2::F64(arr) => {
            let mut output_arr = utils::output_array1::<f64>(output, "output", "float64")?;
            nabled_ml::stats::column_means_view_into(
                &arr.as_array(),
                &mut output_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
    }
}

/// Center columns (subtract mean).
#[pyfunction]
pub fn center_columns<'py>(py: Python<'py>, matrix: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
    match utils::real_array2(matrix, "matrix")? {
        utils::RealReadonlyArray2::F32(arr) => Ok(utils::pyarray2_from_owned(
            py,
            nabled_ml::stats::center_columns_view(&arr.as_array()),
        )),
        utils::RealReadonlyArray2::F64(arr) => Ok(utils::pyarray2_from_owned(
            py,
            nabled_ml::stats::center_columns_view(&arr.as_array()),
        )),
    }
}

/// Center columns into caller-provided output.
#[pyfunction]
pub fn center_columns_into(matrix: &Bound<'_, PyAny>, output: &Bound<'_, PyAny>) -> PyResult<()> {
    match utils::real_array2(matrix, "matrix")? {
        utils::RealReadonlyArray2::F32(arr) => {
            let mut output_arr = utils::output_array2::<f32>(output, "output", "float32")?;
            nabled_ml::stats::center_columns_view_into(
                &arr.as_array(),
                &mut output_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        utils::RealReadonlyArray2::F64(arr) => {
            let mut output_arr = utils::output_array2::<f64>(output, "output", "float64")?;
            nabled_ml::stats::center_columns_view_into(
                &arr.as_array(),
                &mut output_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
    }
}

/// Compute covariance matrix.
#[pyfunction]
pub fn covariance_matrix<'py>(py: Python<'py>, matrix: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
    match utils::real_array2(matrix, "matrix")? {
        utils::RealReadonlyArray2::F32(arr) => {
            let result =
                nabled_ml::stats::covariance_matrix_view(&arr.as_array()).map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        utils::RealReadonlyArray2::F64(arr) => {
            let result =
                nabled_ml::stats::covariance_matrix_view(&arr.as_array()).map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
    }
}

/// Compute covariance matrix into caller-provided output.
#[pyfunction]
pub fn covariance_matrix_into(
    matrix: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
) -> PyResult<()> {
    match utils::real_array2(matrix, "matrix")? {
        utils::RealReadonlyArray2::F32(arr) => {
            let mut output_arr = utils::output_array2::<f32>(output, "output", "float32")?;
            nabled_ml::stats::covariance_matrix_view_into(
                &arr.as_array(),
                &mut output_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        utils::RealReadonlyArray2::F64(arr) => {
            let mut output_arr = utils::output_array2::<f64>(output, "output", "float64")?;
            nabled_ml::stats::covariance_matrix_view_into(
                &arr.as_array(),
                &mut output_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
    }
}

/// Compute correlation matrix.
#[pyfunction]
pub fn correlation_matrix<'py>(py: Python<'py>, matrix: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
    match utils::real_array2(matrix, "matrix")? {
        utils::RealReadonlyArray2::F32(arr) => {
            let result =
                nabled_ml::stats::correlation_matrix_view(&arr.as_array()).map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        utils::RealReadonlyArray2::F64(arr) => {
            let result =
                nabled_ml::stats::correlation_matrix_view(&arr.as_array()).map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
    }
}

/// Compute correlation matrix into caller-provided output.
#[pyfunction]
pub fn correlation_matrix_into(
    matrix: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
) -> PyResult<()> {
    match utils::real_array2(matrix, "matrix")? {
        utils::RealReadonlyArray2::F32(arr) => {
            let mut output_arr = utils::output_array2::<f32>(output, "output", "float32")?;
            nabled_ml::stats::correlation_matrix_view_into(
                &arr.as_array(),
                &mut output_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        utils::RealReadonlyArray2::F64(arr) => {
            let mut output_arr = utils::output_array2::<f64>(output, "output", "float64")?;
            nabled_ml::stats::correlation_matrix_view_into(
                &arr.as_array(),
                &mut output_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
    }
}

/// Compute column means for a complex matrix.
#[pyfunction]
pub fn column_means_complex<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match utils::numeric_array2(matrix, "matrix")? {
        utils::NumericReadonlyArray2::C64(arr) => Ok(utils::pyarray1_from_owned(
            py,
            nabled_ml::stats::column_means_complex_view(&arr.as_array()),
        )),
        _ => Err(utils::matching_complex_dtype_error(&["matrix"])),
    }
}

/// Compute complex column means into caller-provided output.
#[pyfunction]
pub fn column_means_complex_into(
    matrix: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
) -> PyResult<()> {
    match utils::numeric_array2(matrix, "matrix")? {
        utils::NumericReadonlyArray2::C64(arr) => {
            let mut output_arr =
                utils::output_array1::<num_complex::Complex64>(output, "output", "complex128")?;
            nabled_ml::stats::column_means_complex_view_into(
                &arr.as_array(),
                &mut output_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        _ => Err(utils::matching_complex_dtype_error(&["matrix", "output"])),
    }
}

/// Center complex columns (subtract mean).
#[pyfunction]
pub fn center_columns_complex<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match utils::numeric_array2(matrix, "matrix")? {
        utils::NumericReadonlyArray2::C64(arr) => Ok(utils::pyarray2_from_owned(
            py,
            nabled_ml::stats::center_columns_complex_view(&arr.as_array()),
        )),
        _ => Err(utils::matching_complex_dtype_error(&["matrix"])),
    }
}

/// Center complex columns into caller-provided output.
#[pyfunction]
pub fn center_columns_complex_into(
    matrix: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
) -> PyResult<()> {
    match utils::numeric_array2(matrix, "matrix")? {
        utils::NumericReadonlyArray2::C64(arr) => {
            let mut output_arr =
                utils::output_array2::<num_complex::Complex64>(output, "output", "complex128")?;
            nabled_ml::stats::center_columns_complex_view_into(
                &arr.as_array(),
                &mut output_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        _ => Err(utils::matching_complex_dtype_error(&["matrix", "output"])),
    }
}

/// Compute covariance matrix for a complex matrix.
#[pyfunction]
pub fn covariance_matrix_complex<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match utils::numeric_array2(matrix, "matrix")? {
        utils::NumericReadonlyArray2::C64(arr) => {
            let result = nabled_ml::stats::covariance_matrix_complex_view(&arr.as_array())
                .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        _ => Err(utils::matching_complex_dtype_error(&["matrix"])),
    }
}

/// Compute covariance matrix for a complex matrix into caller-provided output.
#[pyfunction]
pub fn covariance_matrix_complex_into(
    matrix: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
) -> PyResult<()> {
    match utils::numeric_array2(matrix, "matrix")? {
        utils::NumericReadonlyArray2::C64(arr) => {
            let mut output_arr =
                utils::output_array2::<num_complex::Complex64>(output, "output", "complex128")?;
            nabled_ml::stats::covariance_matrix_complex_view_into(
                &arr.as_array(),
                &mut output_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        _ => Err(utils::matching_complex_dtype_error(&["matrix", "output"])),
    }
}

/// Compute correlation matrix for a complex matrix.
#[pyfunction]
pub fn correlation_matrix_complex<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match utils::numeric_array2(matrix, "matrix")? {
        utils::NumericReadonlyArray2::C64(arr) => {
            let result = nabled_ml::stats::correlation_matrix_complex_view(&arr.as_array())
                .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        _ => Err(utils::matching_complex_dtype_error(&["matrix"])),
    }
}

/// Compute correlation matrix for a complex matrix into caller-provided output.
#[pyfunction]
pub fn correlation_matrix_complex_into(
    matrix: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
) -> PyResult<()> {
    match utils::numeric_array2(matrix, "matrix")? {
        utils::NumericReadonlyArray2::C64(arr) => {
            let mut output_arr =
                utils::output_array2::<num_complex::Complex64>(output, "output", "complex128")?;
            nabled_ml::stats::correlation_matrix_complex_view_into(
                &arr.as_array(),
                &mut output_arr.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        _ => Err(utils::matching_complex_dtype_error(&["matrix", "output"])),
    }
}
