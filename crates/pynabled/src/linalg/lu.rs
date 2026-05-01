//! LU decomposition bindings for Python.

use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::error::to_py_err;
use crate::utils;

fn pivots_to_pyarray(py: Python<'_>, pivots: Vec<usize>) -> Py<PyAny> {
    let pivots = utils::usize_array1_to_i64(pivots, "pivots")
        .expect("usize pivot indices should fit in Python int64 arrays");
    utils::pyarray1_from_owned(py, pivots)
}

/// Compute LU decomposition. Returns `(L, U, pivots, permutation_sign)`.
#[pyfunction(name = "lu_decompose")]
pub fn decompose<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>, i8)> {
    match utils::real_array2(a, "a")? {
        utils::RealReadonlyArray2::F32(arr) => {
            let (result, pivots, permutation_sign) =
                nabled_linalg::lu::decompose_view_with_metadata(&arr.as_array())
                    .map_err(to_py_err)?;
            Ok((
                utils::pyarray2_from_owned(py, result.l),
                utils::pyarray2_from_owned(py, result.u),
                pivots_to_pyarray(py, pivots),
                permutation_sign,
            ))
        }
        utils::RealReadonlyArray2::F64(arr) => {
            let (result, pivots, permutation_sign) =
                nabled_linalg::lu::decompose_view_with_metadata(&arr.as_array())
                    .map_err(to_py_err)?;
            Ok((
                utils::pyarray2_from_owned(py, result.l),
                utils::pyarray2_from_owned(py, result.u),
                pivots_to_pyarray(py, pivots),
                permutation_sign,
            ))
        }
    }
}

/// Solve `Ax = b` using LU decomposition.
#[pyfunction(name = "lu_solve")]
pub fn solve<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match (utils::numeric_array2(a, "a")?, utils::numeric_array1(b, "b")?) {
        (utils::NumericReadonlyArray2::F32(a_arr), utils::NumericReadonlyArray1::F32(b_arr)) => {
            let result = nabled_linalg::lu::solve_view(&a_arr.as_array(), &b_arr.as_array())
                .map_err(to_py_err)?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
        (utils::NumericReadonlyArray2::F64(a_arr), utils::NumericReadonlyArray1::F64(b_arr)) => {
            let result = nabled_linalg::lu::solve_view(&a_arr.as_array(), &b_arr.as_array())
                .map_err(to_py_err)?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
        (utils::NumericReadonlyArray2::C64(a_arr), utils::NumericReadonlyArray1::C64(b_arr)) => {
            let result =
                nabled_linalg::lu::solve_complex_view(&a_arr.as_array(), &b_arr.as_array())
                    .map_err(to_py_err)?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
        _ => Err(utils::matching_numeric_dtype_error(&["a", "b"])),
    }
}

/// Solve `Ax = b` into a caller-provided output vector.
#[pyfunction(name = "lu_solve_into")]
pub fn solve_into(
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
) -> PyResult<()> {
    match (utils::numeric_array2(a, "a")?, utils::numeric_array1(b, "b")?) {
        (utils::NumericReadonlyArray2::F32(a_arr), utils::NumericReadonlyArray1::F32(b_arr)) => {
            let mut out = utils::output_array1::<f32>(output, "output", "float32")?;
            let result = nabled_linalg::lu::solve_view(&a_arr.as_array(), &b_arr.as_array())
                .map_err(to_py_err)?;
            out.as_array_mut().assign(&result);
            Ok(())
        }
        (utils::NumericReadonlyArray2::F64(a_arr), utils::NumericReadonlyArray1::F64(b_arr)) => {
            let mut out = utils::output_array1::<f64>(output, "output", "float64")?;
            let result = nabled_linalg::lu::solve_view(&a_arr.as_array(), &b_arr.as_array())
                .map_err(to_py_err)?;
            out.as_array_mut().assign(&result);
            Ok(())
        }
        (utils::NumericReadonlyArray2::C64(a_arr), utils::NumericReadonlyArray1::C64(b_arr)) => {
            let mut out =
                utils::output_array1::<num_complex::Complex64>(output, "output", "complex128")?;
            let result =
                nabled_linalg::lu::solve_complex_view(&a_arr.as_array(), &b_arr.as_array())
                    .map_err(to_py_err)?;
            out.as_array_mut().assign(&result);
            Ok(())
        }
        _ => Err(utils::matching_numeric_dtype_error(&["a", "b", "output"])),
    }
}

/// Solve `Ax = b` from a precomputed LU factorization.
#[pyfunction(name = "lu_solve_from_factor")]
pub fn solve_from_factor<'py>(
    py: Python<'py>,
    lower_factor: &Bound<'py, PyAny>,
    upper_factor: &Bound<'py, PyAny>,
    pivots: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match (
        utils::real_array2(lower_factor, "lower_factor")?,
        utils::real_array2(upper_factor, "upper_factor")?,
        utils::index_array1(pivots, "pivots")?,
        utils::real_array1(b, "b")?,
    ) {
        (
            utils::RealReadonlyArray2::F32(lower_arr),
            utils::RealReadonlyArray2::F32(upper_arr),
            utils::IndexReadonlyArray1::I32(pivot_arr),
            utils::RealReadonlyArray1::F32(b_arr),
        ) => {
            let result = nabled_linalg::lu::solve_from_factor_view(
                &lower_arr.as_array(),
                &upper_arr.as_array(),
                &pivot_arr.as_array(),
                &b_arr.as_array(),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
        (
            utils::RealReadonlyArray2::F32(lower_arr),
            utils::RealReadonlyArray2::F32(upper_arr),
            utils::IndexReadonlyArray1::I64(pivot_arr),
            utils::RealReadonlyArray1::F32(b_arr),
        ) => {
            let result = nabled_linalg::lu::solve_from_factor_view(
                &lower_arr.as_array(),
                &upper_arr.as_array(),
                &pivot_arr.as_array(),
                &b_arr.as_array(),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
        (
            utils::RealReadonlyArray2::F64(lower_arr),
            utils::RealReadonlyArray2::F64(upper_arr),
            utils::IndexReadonlyArray1::I32(pivot_arr),
            utils::RealReadonlyArray1::F64(b_arr),
        ) => {
            let result = nabled_linalg::lu::solve_from_factor_view(
                &lower_arr.as_array(),
                &upper_arr.as_array(),
                &pivot_arr.as_array(),
                &b_arr.as_array(),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
        (
            utils::RealReadonlyArray2::F64(lower_arr),
            utils::RealReadonlyArray2::F64(upper_arr),
            utils::IndexReadonlyArray1::I64(pivot_arr),
            utils::RealReadonlyArray1::F64(b_arr),
        ) => {
            let result = nabled_linalg::lu::solve_from_factor_view(
                &lower_arr.as_array(),
                &upper_arr.as_array(),
                &pivot_arr.as_array(),
                &b_arr.as_array(),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
        _ => Err(utils::matching_real_dtype_error(&["lower_factor", "upper_factor", "b"])),
    }
}

/// Solve `Ax = b` from a precomputed LU factorization into a caller-provided output vector.
#[pyfunction(name = "lu_solve_from_factor_into")]
pub fn solve_from_factor_into(
    lower_factor: &Bound<'_, PyAny>,
    upper_factor: &Bound<'_, PyAny>,
    pivots: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
) -> PyResult<()> {
    match (
        utils::real_array2(lower_factor, "lower_factor")?,
        utils::real_array2(upper_factor, "upper_factor")?,
        utils::index_array1(pivots, "pivots")?,
        utils::real_array1(b, "b")?,
    ) {
        (
            utils::RealReadonlyArray2::F32(lower_arr),
            utils::RealReadonlyArray2::F32(upper_arr),
            utils::IndexReadonlyArray1::I32(pivot_arr),
            utils::RealReadonlyArray1::F32(b_arr),
        ) => {
            let mut out = utils::output_array1::<f32>(output, "output", "float32")?;
            nabled_linalg::lu::solve_from_factor_into_view(
                &lower_arr.as_array(),
                &upper_arr.as_array(),
                &pivot_arr.as_array(),
                &b_arr.as_array(),
                &mut out.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        (
            utils::RealReadonlyArray2::F32(lower_arr),
            utils::RealReadonlyArray2::F32(upper_arr),
            utils::IndexReadonlyArray1::I64(pivot_arr),
            utils::RealReadonlyArray1::F32(b_arr),
        ) => {
            let mut out = utils::output_array1::<f32>(output, "output", "float32")?;
            nabled_linalg::lu::solve_from_factor_into_view(
                &lower_arr.as_array(),
                &upper_arr.as_array(),
                &pivot_arr.as_array(),
                &b_arr.as_array(),
                &mut out.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        (
            utils::RealReadonlyArray2::F64(lower_arr),
            utils::RealReadonlyArray2::F64(upper_arr),
            utils::IndexReadonlyArray1::I32(pivot_arr),
            utils::RealReadonlyArray1::F64(b_arr),
        ) => {
            let mut out = utils::output_array1::<f64>(output, "output", "float64")?;
            nabled_linalg::lu::solve_from_factor_into_view(
                &lower_arr.as_array(),
                &upper_arr.as_array(),
                &pivot_arr.as_array(),
                &b_arr.as_array(),
                &mut out.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        (
            utils::RealReadonlyArray2::F64(lower_arr),
            utils::RealReadonlyArray2::F64(upper_arr),
            utils::IndexReadonlyArray1::I64(pivot_arr),
            utils::RealReadonlyArray1::F64(b_arr),
        ) => {
            let mut out = utils::output_array1::<f64>(output, "output", "float64")?;
            nabled_linalg::lu::solve_from_factor_into_view(
                &lower_arr.as_array(),
                &upper_arr.as_array(),
                &pivot_arr.as_array(),
                &b_arr.as_array(),
                &mut out.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        _ => {
            Err(utils::matching_real_dtype_error(&["lower_factor", "upper_factor", "b", "output"]))
        }
    }
}

/// Solve `Ax = b` using MAGMA mixed-precision iterative refinement.
#[pyfunction(name = "lu_solve_mixed")]
pub fn solve_mixed<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<(Py<PyAny>, usize)> {
    match (utils::numeric_array2(a, "a")?, utils::numeric_array1(b, "b")?) {
        (utils::NumericReadonlyArray2::F64(a_arr), utils::NumericReadonlyArray1::F64(b_arr)) => {
            let result =
                nabled_linalg::lu::solve_mixed_f64_view(&a_arr.as_array(), &b_arr.as_array())
                    .map_err(to_py_err)?;
            Ok((utils::pyarray1_from_owned(py, result.solution), result.refinement_iterations))
        }
        (utils::NumericReadonlyArray2::C64(a_arr), utils::NumericReadonlyArray1::C64(b_arr)) => {
            let result =
                nabled_linalg::lu::solve_mixed_complex_view(&a_arr.as_array(), &b_arr.as_array())
                    .map_err(to_py_err)?;
            Ok((utils::pyarray1_from_owned(py, result.solution), result.refinement_iterations))
        }
        _ => Err(utils::matching_mixed_provider_dtype_error(&["a", "b"])),
    }
}

/// Compute matrix inverse using LU.
#[pyfunction(name = "lu_inverse")]
pub fn inverse<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
    match utils::numeric_array2(a, "a")? {
        utils::NumericReadonlyArray2::F32(arr) => {
            let result = nabled_linalg::lu::inverse_view(&arr.as_array()).map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        utils::NumericReadonlyArray2::F64(arr) => {
            let result = nabled_linalg::lu::inverse_view(&arr.as_array()).map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        utils::NumericReadonlyArray2::C64(arr) => {
            let result =
                nabled_linalg::lu::inverse_complex_view(&arr.as_array()).map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
    }
}

/// Compute matrix inverse using LU into a caller-provided output matrix.
#[pyfunction(name = "lu_inverse_into")]
pub fn inverse_into(a: &Bound<'_, PyAny>, output: &Bound<'_, PyAny>) -> PyResult<()> {
    match utils::numeric_array2(a, "a")? {
        utils::NumericReadonlyArray2::F32(arr) => {
            let mut out = utils::output_array2::<f32>(output, "output", "float32")?;
            let result = nabled_linalg::lu::inverse_view(&arr.as_array()).map_err(to_py_err)?;
            out.as_array_mut().assign(&result);
            Ok(())
        }
        utils::NumericReadonlyArray2::F64(arr) => {
            let mut out = utils::output_array2::<f64>(output, "output", "float64")?;
            let result = nabled_linalg::lu::inverse_view(&arr.as_array()).map_err(to_py_err)?;
            out.as_array_mut().assign(&result);
            Ok(())
        }
        utils::NumericReadonlyArray2::C64(arr) => {
            let mut out =
                utils::output_array2::<num_complex::Complex64>(output, "output", "complex128")?;
            let result =
                nabled_linalg::lu::inverse_complex_view(&arr.as_array()).map_err(to_py_err)?;
            out.as_array_mut().assign(&result);
            Ok(())
        }
    }
}

/// Compute matrix inverse from a precomputed LU factorization.
#[pyfunction(name = "lu_inverse_from_factor")]
pub fn inverse_from_factor<'py>(
    py: Python<'py>,
    lower_factor: &Bound<'py, PyAny>,
    upper_factor: &Bound<'py, PyAny>,
    pivots: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match (
        utils::real_array2(lower_factor, "lower_factor")?,
        utils::real_array2(upper_factor, "upper_factor")?,
        utils::index_array1(pivots, "pivots")?,
    ) {
        (
            utils::RealReadonlyArray2::F32(lower_arr),
            utils::RealReadonlyArray2::F32(upper_arr),
            utils::IndexReadonlyArray1::I32(pivot_arr),
        ) => {
            let result = nabled_linalg::lu::inverse_from_factor_view(
                &lower_arr.as_array(),
                &upper_arr.as_array(),
                &pivot_arr.as_array(),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        (
            utils::RealReadonlyArray2::F32(lower_arr),
            utils::RealReadonlyArray2::F32(upper_arr),
            utils::IndexReadonlyArray1::I64(pivot_arr),
        ) => {
            let result = nabled_linalg::lu::inverse_from_factor_view(
                &lower_arr.as_array(),
                &upper_arr.as_array(),
                &pivot_arr.as_array(),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        (
            utils::RealReadonlyArray2::F64(lower_arr),
            utils::RealReadonlyArray2::F64(upper_arr),
            utils::IndexReadonlyArray1::I32(pivot_arr),
        ) => {
            let result = nabled_linalg::lu::inverse_from_factor_view(
                &lower_arr.as_array(),
                &upper_arr.as_array(),
                &pivot_arr.as_array(),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        (
            utils::RealReadonlyArray2::F64(lower_arr),
            utils::RealReadonlyArray2::F64(upper_arr),
            utils::IndexReadonlyArray1::I64(pivot_arr),
        ) => {
            let result = nabled_linalg::lu::inverse_from_factor_view(
                &lower_arr.as_array(),
                &upper_arr.as_array(),
                &pivot_arr.as_array(),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, result))
        }
        _ => Err(utils::matching_real_dtype_error(&["lower_factor", "upper_factor"])),
    }
}

/// Compute matrix inverse from a precomputed LU factorization into a caller-provided output.
#[pyfunction(name = "lu_inverse_from_factor_into")]
pub fn inverse_from_factor_into(
    lower_factor: &Bound<'_, PyAny>,
    upper_factor: &Bound<'_, PyAny>,
    pivots: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
) -> PyResult<()> {
    match (
        utils::real_array2(lower_factor, "lower_factor")?,
        utils::real_array2(upper_factor, "upper_factor")?,
        utils::index_array1(pivots, "pivots")?,
    ) {
        (
            utils::RealReadonlyArray2::F32(lower_arr),
            utils::RealReadonlyArray2::F32(upper_arr),
            utils::IndexReadonlyArray1::I32(pivot_arr),
        ) => {
            let mut out = utils::output_array2::<f32>(output, "output", "float32")?;
            nabled_linalg::lu::inverse_from_factor_into_view(
                &lower_arr.as_array(),
                &upper_arr.as_array(),
                &pivot_arr.as_array(),
                &mut out.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        (
            utils::RealReadonlyArray2::F32(lower_arr),
            utils::RealReadonlyArray2::F32(upper_arr),
            utils::IndexReadonlyArray1::I64(pivot_arr),
        ) => {
            let mut out = utils::output_array2::<f32>(output, "output", "float32")?;
            nabled_linalg::lu::inverse_from_factor_into_view(
                &lower_arr.as_array(),
                &upper_arr.as_array(),
                &pivot_arr.as_array(),
                &mut out.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        (
            utils::RealReadonlyArray2::F64(lower_arr),
            utils::RealReadonlyArray2::F64(upper_arr),
            utils::IndexReadonlyArray1::I32(pivot_arr),
        ) => {
            let mut out = utils::output_array2::<f64>(output, "output", "float64")?;
            nabled_linalg::lu::inverse_from_factor_into_view(
                &lower_arr.as_array(),
                &upper_arr.as_array(),
                &pivot_arr.as_array(),
                &mut out.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        (
            utils::RealReadonlyArray2::F64(lower_arr),
            utils::RealReadonlyArray2::F64(upper_arr),
            utils::IndexReadonlyArray1::I64(pivot_arr),
        ) => {
            let mut out = utils::output_array2::<f64>(output, "output", "float64")?;
            nabled_linalg::lu::inverse_from_factor_into_view(
                &lower_arr.as_array(),
                &upper_arr.as_array(),
                &pivot_arr.as_array(),
                &mut out.as_array_mut(),
            )
            .map_err(to_py_err)
        }
        _ => Err(utils::matching_real_dtype_error(&["lower_factor", "upper_factor", "output"])),
    }
}

/// Compute determinant.
#[pyfunction(name = "lu_determinant")]
pub fn determinant<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
    match utils::numeric_array2(a, "a")? {
        utils::NumericReadonlyArray2::F32(arr) => {
            let result = nabled_linalg::lu::determinant_view(&arr.as_array()).map_err(to_py_err)?;
            Ok(utils::py_float(py, result.into()))
        }
        utils::NumericReadonlyArray2::F64(arr) => {
            let result = nabled_linalg::lu::determinant_view(&arr.as_array()).map_err(to_py_err)?;
            Ok(utils::py_float(py, result))
        }
        utils::NumericReadonlyArray2::C64(arr) => {
            let result =
                nabled_linalg::lu::determinant_complex_view(&arr.as_array()).map_err(to_py_err)?;
            Ok(utils::py_complex(py, result))
        }
    }
}

/// Compute determinant from a precomputed real LU factorization.
#[pyfunction(name = "lu_determinant_from_factor")]
pub fn determinant_from_factor<'py>(
    py: Python<'py>,
    upper_factor: &Bound<'py, PyAny>,
    permutation_sign: i8,
) -> PyResult<Py<PyAny>> {
    match utils::real_array2(upper_factor, "upper_factor")? {
        utils::RealReadonlyArray2::F32(arr) => {
            let result =
                nabled_linalg::lu::determinant_from_factor_view(&arr.as_array(), permutation_sign)
                    .map_err(to_py_err)?;
            Ok(utils::py_float(py, result.into()))
        }
        utils::RealReadonlyArray2::F64(arr) => {
            let result =
                nabled_linalg::lu::determinant_from_factor_view(&arr.as_array(), permutation_sign)
                    .map_err(to_py_err)?;
            Ok(utils::py_float(py, result))
        }
    }
}

/// Compute signed log-determinant. Returns `(sign, ln_abs_det)`.
#[pyfunction(name = "lu_log_determinant")]
pub fn log_determinant<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<(i8, Py<PyAny>)> {
    match utils::real_array2(a, "a")? {
        utils::RealReadonlyArray2::F32(arr) => {
            let result =
                nabled_linalg::lu::log_determinant_view(&arr.as_array()).map_err(to_py_err)?;
            Ok((result.sign, utils::py_float(py, result.ln_abs_det.into())))
        }
        utils::RealReadonlyArray2::F64(arr) => {
            let result =
                nabled_linalg::lu::log_determinant_view(&arr.as_array()).map_err(to_py_err)?;
            Ok((result.sign, utils::py_float(py, result.ln_abs_det)))
        }
    }
}

/// Compute signed log-determinant from a precomputed real LU factorization.
#[pyfunction(name = "lu_log_determinant_from_factor")]
pub fn log_determinant_from_factor<'py>(
    py: Python<'py>,
    upper_factor: &Bound<'py, PyAny>,
    permutation_sign: i8,
) -> PyResult<(i8, Py<PyAny>)> {
    match utils::real_array2(upper_factor, "upper_factor")? {
        utils::RealReadonlyArray2::F32(arr) => {
            let result = nabled_linalg::lu::log_determinant_from_factor_view(
                &arr.as_array(),
                permutation_sign,
            )
            .map_err(to_py_err)?;
            Ok((result.sign, utils::py_float(py, result.ln_abs_det.into())))
        }
        utils::RealReadonlyArray2::F64(arr) => {
            let result = nabled_linalg::lu::log_determinant_from_factor_view(
                &arr.as_array(),
                permutation_sign,
            )
            .map_err(to_py_err)?;
            Ok((result.sign, utils::py_float(py, result.ln_abs_det)))
        }
    }
}
