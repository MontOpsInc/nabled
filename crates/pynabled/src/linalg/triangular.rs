//! Triangular solve bindings for Python.

use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::error::to_py_err;
use crate::utils;

/// Solve lower triangular system Lx = b.
#[pyfunction(name = "triangular_solve_lower")]
pub fn solve_lower<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyAny>,
    rhs: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match (utils::numeric_array2(matrix, "matrix")?, utils::numeric_array1(rhs, "rhs")?) {
        (utils::NumericReadonlyArray2::F32(matrix), utils::NumericReadonlyArray1::F32(rhs)) => {
            let result =
                nabled_linalg::triangular::solve_lower_view(&matrix.as_array(), &rhs.as_array())
                    .map_err(to_py_err)?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
        (utils::NumericReadonlyArray2::F64(matrix), utils::NumericReadonlyArray1::F64(rhs)) => {
            let result =
                nabled_linalg::triangular::solve_lower_view(&matrix.as_array(), &rhs.as_array())
                    .map_err(to_py_err)?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
        (utils::NumericReadonlyArray2::C64(matrix), utils::NumericReadonlyArray1::C64(rhs)) => {
            let result = nabled_linalg::triangular::solve_lower_complex_view(
                &matrix.as_array(),
                &rhs.as_array(),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
        _ => Err(utils::matching_numeric_dtype_error(&["matrix", "rhs"])),
    }
}

/// Solve lower triangular system `Lx = b` into a caller-provided output vector.
#[pyfunction(name = "triangular_solve_lower_into")]
pub fn solve_lower_into(
    matrix: &Bound<'_, PyAny>,
    rhs: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
) -> PyResult<()> {
    match (utils::numeric_array2(matrix, "matrix")?, utils::numeric_array1(rhs, "rhs")?) {
        (utils::NumericReadonlyArray2::F32(matrix), utils::NumericReadonlyArray1::F32(rhs)) => {
            let mut output = utils::output_array1::<f32>(output, "output", "float32")?;
            let mut output_view = output.as_array_mut();
            nabled_linalg::triangular::solve_lower_view_into(
                &matrix.as_array(),
                &rhs.as_array(),
                &mut output_view,
            )
            .map_err(to_py_err)
        }
        (utils::NumericReadonlyArray2::F64(matrix), utils::NumericReadonlyArray1::F64(rhs)) => {
            let mut output = utils::output_array1::<f64>(output, "output", "float64")?;
            let mut output_view = output.as_array_mut();
            nabled_linalg::triangular::solve_lower_view_into(
                &matrix.as_array(),
                &rhs.as_array(),
                &mut output_view,
            )
            .map_err(to_py_err)
        }
        (utils::NumericReadonlyArray2::C64(matrix), utils::NumericReadonlyArray1::C64(rhs)) => {
            let mut output =
                utils::output_array1::<num_complex::Complex64>(output, "output", "complex128")?;
            let mut output_view = output.as_array_mut();
            nabled_linalg::triangular::solve_lower_complex_view_into(
                &matrix.as_array(),
                &rhs.as_array(),
                &mut output_view,
            )
            .map_err(to_py_err)
        }
        _ => Err(utils::matching_numeric_dtype_error(&["matrix", "rhs", "output"])),
    }
}

/// Solve upper triangular system Ux = b.
#[pyfunction(name = "triangular_solve_upper")]
pub fn solve_upper<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyAny>,
    rhs: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match (utils::numeric_array2(matrix, "matrix")?, utils::numeric_array1(rhs, "rhs")?) {
        (utils::NumericReadonlyArray2::F32(matrix), utils::NumericReadonlyArray1::F32(rhs)) => {
            let result =
                nabled_linalg::triangular::solve_upper_view(&matrix.as_array(), &rhs.as_array())
                    .map_err(to_py_err)?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
        (utils::NumericReadonlyArray2::F64(matrix), utils::NumericReadonlyArray1::F64(rhs)) => {
            let result =
                nabled_linalg::triangular::solve_upper_view(&matrix.as_array(), &rhs.as_array())
                    .map_err(to_py_err)?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
        (utils::NumericReadonlyArray2::C64(matrix), utils::NumericReadonlyArray1::C64(rhs)) => {
            let result = nabled_linalg::triangular::solve_upper_complex_view(
                &matrix.as_array(),
                &rhs.as_array(),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
        _ => Err(utils::matching_numeric_dtype_error(&["matrix", "rhs"])),
    }
}

/// Solve upper triangular system `Ux = b` into a caller-provided output vector.
#[pyfunction(name = "triangular_solve_upper_into")]
pub fn solve_upper_into(
    matrix: &Bound<'_, PyAny>,
    rhs: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
) -> PyResult<()> {
    match (utils::numeric_array2(matrix, "matrix")?, utils::numeric_array1(rhs, "rhs")?) {
        (utils::NumericReadonlyArray2::F32(matrix), utils::NumericReadonlyArray1::F32(rhs)) => {
            let mut output = utils::output_array1::<f32>(output, "output", "float32")?;
            let mut output_view = output.as_array_mut();
            nabled_linalg::triangular::solve_upper_view_into(
                &matrix.as_array(),
                &rhs.as_array(),
                &mut output_view,
            )
            .map_err(to_py_err)
        }
        (utils::NumericReadonlyArray2::F64(matrix), utils::NumericReadonlyArray1::F64(rhs)) => {
            let mut output = utils::output_array1::<f64>(output, "output", "float64")?;
            let mut output_view = output.as_array_mut();
            nabled_linalg::triangular::solve_upper_view_into(
                &matrix.as_array(),
                &rhs.as_array(),
                &mut output_view,
            )
            .map_err(to_py_err)
        }
        (utils::NumericReadonlyArray2::C64(matrix), utils::NumericReadonlyArray1::C64(rhs)) => {
            let mut output =
                utils::output_array1::<num_complex::Complex64>(output, "output", "complex128")?;
            let mut output_view = output.as_array_mut();
            nabled_linalg::triangular::solve_upper_complex_view_into(
                &matrix.as_array(),
                &rhs.as_array(),
                &mut output_view,
            )
            .map_err(to_py_err)
        }
        _ => Err(utils::matching_numeric_dtype_error(&["matrix", "rhs", "output"])),
    }
}

/// Solve lower triangular matrix system LX = B.
#[pyfunction(name = "triangular_solve_lower_matrix")]
pub fn solve_lower_matrix<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyAny>,
    rhs: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match (utils::real_array2(matrix, "matrix")?, utils::real_array2(rhs, "rhs")?) {
        (utils::RealReadonlyArray2::F32(matrix), utils::RealReadonlyArray2::F32(rhs)) => {
            let result = nabled_linalg::triangular::solve_lower_matrix_view(
                &matrix.as_array(),
                &rhs.as_array(),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        (utils::RealReadonlyArray2::F64(matrix), utils::RealReadonlyArray2::F64(rhs)) => {
            let result = nabled_linalg::triangular::solve_lower_matrix_view(
                &matrix.as_array(),
                &rhs.as_array(),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        _ => Err(utils::matching_real_dtype_error(&["matrix", "rhs"])),
    }
}

/// Solve lower triangular matrix system `LX = B` into a caller-provided output matrix.
#[pyfunction(name = "triangular_solve_lower_matrix_into")]
pub fn solve_lower_matrix_into(
    matrix: &Bound<'_, PyAny>,
    rhs: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
) -> PyResult<()> {
    match (utils::real_array2(matrix, "matrix")?, utils::real_array2(rhs, "rhs")?) {
        (utils::RealReadonlyArray2::F32(matrix), utils::RealReadonlyArray2::F32(rhs)) => {
            let mut output = utils::output_array2::<f32>(output, "output", "float32")?;
            let mut output_view = output.as_array_mut();
            nabled_linalg::triangular::solve_lower_matrix_view_into(
                &matrix.as_array(),
                &rhs.as_array(),
                &mut output_view,
            )
            .map_err(to_py_err)
        }
        (utils::RealReadonlyArray2::F64(matrix), utils::RealReadonlyArray2::F64(rhs)) => {
            let mut output = utils::output_array2::<f64>(output, "output", "float64")?;
            let mut output_view = output.as_array_mut();
            nabled_linalg::triangular::solve_lower_matrix_view_into(
                &matrix.as_array(),
                &rhs.as_array(),
                &mut output_view,
            )
            .map_err(to_py_err)
        }
        _ => Err(utils::matching_real_dtype_error(&["matrix", "rhs", "output"])),
    }
}

/// Solve upper triangular matrix system UX = B.
#[pyfunction(name = "triangular_solve_upper_matrix")]
pub fn solve_upper_matrix<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyAny>,
    rhs: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match (utils::real_array2(matrix, "matrix")?, utils::real_array2(rhs, "rhs")?) {
        (utils::RealReadonlyArray2::F32(matrix), utils::RealReadonlyArray2::F32(rhs)) => {
            let result = nabled_linalg::triangular::solve_upper_matrix_view(
                &matrix.as_array(),
                &rhs.as_array(),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        (utils::RealReadonlyArray2::F64(matrix), utils::RealReadonlyArray2::F64(rhs)) => {
            let result = nabled_linalg::triangular::solve_upper_matrix_view(
                &matrix.as_array(),
                &rhs.as_array(),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        _ => Err(utils::matching_real_dtype_error(&["matrix", "rhs"])),
    }
}

/// Solve upper triangular matrix system `UX = B` into a caller-provided output matrix.
#[pyfunction(name = "triangular_solve_upper_matrix_into")]
pub fn solve_upper_matrix_into(
    matrix: &Bound<'_, PyAny>,
    rhs: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
) -> PyResult<()> {
    match (utils::real_array2(matrix, "matrix")?, utils::real_array2(rhs, "rhs")?) {
        (utils::RealReadonlyArray2::F32(matrix), utils::RealReadonlyArray2::F32(rhs)) => {
            let mut output = utils::output_array2::<f32>(output, "output", "float32")?;
            let mut output_view = output.as_array_mut();
            nabled_linalg::triangular::solve_upper_matrix_view_into(
                &matrix.as_array(),
                &rhs.as_array(),
                &mut output_view,
            )
            .map_err(to_py_err)
        }
        (utils::RealReadonlyArray2::F64(matrix), utils::RealReadonlyArray2::F64(rhs)) => {
            let mut output = utils::output_array2::<f64>(output, "output", "float64")?;
            let mut output_view = output.as_array_mut();
            nabled_linalg::triangular::solve_upper_matrix_view_into(
                &matrix.as_array(),
                &rhs.as_array(),
                &mut output_view,
            )
            .map_err(to_py_err)
        }
        _ => Err(utils::matching_real_dtype_error(&["matrix", "rhs", "output"])),
    }
}
