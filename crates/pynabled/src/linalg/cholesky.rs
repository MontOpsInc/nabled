//! Cholesky decomposition bindings for Python.

use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::error::to_py_err;
use crate::utils;

/// Compute Cholesky decomposition. Returns L where A = L L^T.
#[pyfunction(name = "cholesky_decompose")]
pub fn decompose<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
    match utils::numeric_array2(a, "a")? {
        utils::NumericReadonlyArray2::F32(arr) => {
            let result =
                nabled_linalg::cholesky::decompose_view(&arr.as_array()).map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result.l))
        }
        utils::NumericReadonlyArray2::F64(arr) => {
            let result =
                nabled_linalg::cholesky::decompose_view(&arr.as_array()).map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result.l))
        }
        utils::NumericReadonlyArray2::C64(arr) => {
            let result = nabled_linalg::cholesky::decompose_complex_view(&arr.as_array())
                .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result.l))
        }
    }
}

/// Solve Ax = b for symmetric positive definite A.
#[pyfunction(name = "cholesky_solve")]
pub fn solve<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match (utils::numeric_array2(a, "a")?, utils::numeric_array1(b, "b")?) {
        (utils::NumericReadonlyArray2::F32(a_arr), utils::NumericReadonlyArray1::F32(b_arr)) => {
            let result = nabled_linalg::cholesky::solve_view(&a_arr.as_array(), &b_arr.as_array())
                .map_err(to_py_err)?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
        (utils::NumericReadonlyArray2::F64(a_arr), utils::NumericReadonlyArray1::F64(b_arr)) => {
            let result = nabled_linalg::cholesky::solve_view(&a_arr.as_array(), &b_arr.as_array())
                .map_err(to_py_err)?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
        (utils::NumericReadonlyArray2::C64(a_arr), utils::NumericReadonlyArray1::C64(b_arr)) => {
            let result =
                nabled_linalg::cholesky::solve_complex_view(&a_arr.as_array(), &b_arr.as_array())
                    .map_err(to_py_err)?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
        _ => Err(utils::matching_numeric_dtype_error(&["a", "b"])),
    }
}

/// Solve Ax = b into a caller-provided output vector.
#[pyfunction(name = "cholesky_solve_into")]
pub fn solve_into(
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
) -> PyResult<()> {
    match (utils::numeric_array2(a, "a")?, utils::numeric_array1(b, "b")?) {
        (utils::NumericReadonlyArray2::F32(a_arr), utils::NumericReadonlyArray1::F32(b_arr)) => {
            let mut out = utils::output_array1::<f32>(output, "output", "float32")?;
            nabled_linalg::cholesky::solve_into_view(
                &a_arr.as_array(),
                &b_arr.as_array(),
                &mut out.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        (utils::NumericReadonlyArray2::F64(a_arr), utils::NumericReadonlyArray1::F64(b_arr)) => {
            let mut out = utils::output_array1::<f64>(output, "output", "float64")?;
            nabled_linalg::cholesky::solve_into_view(
                &a_arr.as_array(),
                &b_arr.as_array(),
                &mut out.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        (utils::NumericReadonlyArray2::C64(a_arr), utils::NumericReadonlyArray1::C64(b_arr)) => {
            let mut out =
                utils::output_array1::<num_complex::Complex64>(output, "output", "complex128")?;
            nabled_linalg::cholesky::solve_complex_into_view(
                &a_arr.as_array(),
                &b_arr.as_array(),
                &mut out.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        _ => Err(utils::matching_numeric_dtype_error(&["a", "b", "output"])),
    }
}

/// Solve `LL^T x = b` from a precomputed lower-triangular factor.
#[pyfunction(name = "cholesky_solve_from_factor")]
pub fn solve_from_factor<'py>(
    py: Python<'py>,
    lower_factor: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match (utils::numeric_array2(lower_factor, "lower_factor")?, utils::numeric_array1(b, "b")?) {
        (
            utils::NumericReadonlyArray2::F32(lower_arr),
            utils::NumericReadonlyArray1::F32(b_arr),
        ) => {
            let result = nabled_linalg::cholesky::solve_from_factor_view(
                &lower_arr.as_array(),
                &b_arr.as_array(),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
        (
            utils::NumericReadonlyArray2::F64(lower_arr),
            utils::NumericReadonlyArray1::F64(b_arr),
        ) => {
            let result = nabled_linalg::cholesky::solve_from_factor_view(
                &lower_arr.as_array(),
                &b_arr.as_array(),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
        (
            utils::NumericReadonlyArray2::C64(lower_arr),
            utils::NumericReadonlyArray1::C64(b_arr),
        ) => {
            let result = nabled_linalg::cholesky::solve_complex_from_factor_view(
                &lower_arr.as_array(),
                &b_arr.as_array(),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
        _ => Err(utils::matching_numeric_dtype_error(&["lower_factor", "b"])),
    }
}

/// Solve `LL^T x = b` from a precomputed lower-triangular factor into a caller-provided output.
#[pyfunction(name = "cholesky_solve_from_factor_into")]
pub fn solve_from_factor_into(
    lower_factor: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
) -> PyResult<()> {
    match (utils::numeric_array2(lower_factor, "lower_factor")?, utils::numeric_array1(b, "b")?) {
        (
            utils::NumericReadonlyArray2::F32(lower_arr),
            utils::NumericReadonlyArray1::F32(b_arr),
        ) => {
            let mut out = utils::output_array1::<f32>(output, "output", "float32")?;
            nabled_linalg::cholesky::solve_from_factor_into_view(
                &lower_arr.as_array(),
                &b_arr.as_array(),
                &mut out.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        (
            utils::NumericReadonlyArray2::F64(lower_arr),
            utils::NumericReadonlyArray1::F64(b_arr),
        ) => {
            let mut out = utils::output_array1::<f64>(output, "output", "float64")?;
            nabled_linalg::cholesky::solve_from_factor_into_view(
                &lower_arr.as_array(),
                &b_arr.as_array(),
                &mut out.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        (
            utils::NumericReadonlyArray2::C64(lower_arr),
            utils::NumericReadonlyArray1::C64(b_arr),
        ) => {
            let mut out =
                utils::output_array1::<num_complex::Complex64>(output, "output", "complex128")?;
            nabled_linalg::cholesky::solve_complex_from_factor_into_view(
                &lower_arr.as_array(),
                &b_arr.as_array(),
                &mut out.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        _ => Err(utils::matching_numeric_dtype_error(&["lower_factor", "b", "output"])),
    }
}

/// Compute matrix inverse using Cholesky.
#[pyfunction(name = "cholesky_inverse")]
pub fn inverse<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
    match utils::numeric_array2(a, "a")? {
        utils::NumericReadonlyArray2::F32(arr) => {
            let result =
                nabled_linalg::cholesky::inverse_view(&arr.as_array()).map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        utils::NumericReadonlyArray2::F64(arr) => {
            let result =
                nabled_linalg::cholesky::inverse_view(&arr.as_array()).map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        utils::NumericReadonlyArray2::C64(arr) => {
            let result = nabled_linalg::cholesky::inverse_complex_view(&arr.as_array())
                .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
    }
}

/// Compute matrix inverse using Cholesky into a caller-provided output matrix.
#[pyfunction(name = "cholesky_inverse_into")]
pub fn inverse_into(a: &Bound<'_, PyAny>, output: &Bound<'_, PyAny>) -> PyResult<()> {
    match utils::numeric_array2(a, "a")? {
        utils::NumericReadonlyArray2::F32(arr) => {
            let mut out = utils::output_array2::<f32>(output, "output", "float32")?;
            nabled_linalg::cholesky::inverse_into_view(&arr.as_array(), &mut out.as_array_mut())
                .map_err(to_py_err)
        }
        utils::NumericReadonlyArray2::F64(arr) => {
            let mut out = utils::output_array2::<f64>(output, "output", "float64")?;
            nabled_linalg::cholesky::inverse_into_view(&arr.as_array(), &mut out.as_array_mut())
                .map_err(to_py_err)
        }
        utils::NumericReadonlyArray2::C64(arr) => {
            let mut out =
                utils::output_array2::<num_complex::Complex64>(output, "output", "complex128")?;
            nabled_linalg::cholesky::inverse_complex_into_view(
                &arr.as_array(),
                &mut out.as_array_mut(),
            )
            .map_err(to_py_err)
        }
    }
}

/// Compute inverse from a precomputed lower-triangular Cholesky factor.
#[pyfunction(name = "cholesky_inverse_from_factor")]
pub fn inverse_from_factor<'py>(
    py: Python<'py>,
    lower_factor: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match utils::numeric_array2(lower_factor, "lower_factor")? {
        utils::NumericReadonlyArray2::F32(arr) => {
            let result = nabled_linalg::cholesky::inverse_from_factor_view(&arr.as_array())
                .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        utils::NumericReadonlyArray2::F64(arr) => {
            let result = nabled_linalg::cholesky::inverse_from_factor_view(&arr.as_array())
                .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        utils::NumericReadonlyArray2::C64(arr) => {
            let result = nabled_linalg::cholesky::inverse_complex_from_factor_view(&arr.as_array())
                .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
    }
}

/// Compute inverse from a precomputed lower-triangular Cholesky factor into `output`.
#[pyfunction(name = "cholesky_inverse_from_factor_into")]
pub fn inverse_from_factor_into(
    lower_factor: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
) -> PyResult<()> {
    match utils::numeric_array2(lower_factor, "lower_factor")? {
        utils::NumericReadonlyArray2::F32(arr) => {
            let mut out = utils::output_array2::<f32>(output, "output", "float32")?;
            nabled_linalg::cholesky::inverse_from_factor_into_view(
                &arr.as_array(),
                &mut out.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        utils::NumericReadonlyArray2::F64(arr) => {
            let mut out = utils::output_array2::<f64>(output, "output", "float64")?;
            nabled_linalg::cholesky::inverse_from_factor_into_view(
                &arr.as_array(),
                &mut out.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        utils::NumericReadonlyArray2::C64(arr) => {
            let mut out =
                utils::output_array2::<num_complex::Complex64>(output, "output", "complex128")?;
            nabled_linalg::cholesky::inverse_complex_from_factor_into_view(
                &arr.as_array(),
                &mut out.as_array_mut(),
            )
            .map_err(to_py_err)
        }
    }
}
