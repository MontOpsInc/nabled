//! Polar decomposition bindings for Python.

use numpy::{Element, PyReadwriteArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::error::to_py_err;
use crate::utils;

fn complex_svd_component_error() -> PyErr {
    pyo3::exceptions::PyTypeError::new_err(
        "u, vt must both be complex128 and singular_values must be float64 for complex SVD results",
    )
}

fn validate_output_shape<T: Element>(
    output: &PyReadwriteArray2<'_, T>,
    expected: (usize, usize),
    name: &str,
) -> PyResult<()> {
    if output.shape() != [expected.0, expected.1] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{name} must have shape {expected:?}",
        )));
    }
    Ok(())
}

/// Compute polar decomposition. Returns `(U, P)`.
#[pyfunction(name = "polar_compute")]
pub fn compute_polar<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyAny>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    match utils::numeric_array2(matrix, "matrix")? {
        utils::NumericReadonlyArray2::F32(arr) => {
            let result =
                nabled_linalg::polar::compute_polar_view(&arr.as_array()).map_err(to_py_err)?;
            Ok((utils::pyarray2_from_owned(py, result.u), utils::pyarray2_from_owned(py, result.p)))
        }
        utils::NumericReadonlyArray2::F64(arr) => {
            let result =
                nabled_linalg::polar::compute_polar_view(&arr.as_array()).map_err(to_py_err)?;
            Ok((utils::pyarray2_from_owned(py, result.u), utils::pyarray2_from_owned(py, result.p)))
        }
        utils::NumericReadonlyArray2::C64(arr) => {
            let result = nabled_linalg::polar::compute_polar_complex_view(&arr.as_array())
                .map_err(to_py_err)?;
            Ok((utils::pyarray2_from_owned(py, result.u), utils::pyarray2_from_owned(py, result.p)))
        }
    }
}

/// Compute polar decomposition into caller-provided outputs.
#[pyfunction(name = "polar_compute_into")]
pub fn compute_polar_into(
    matrix: &Bound<'_, PyAny>,
    u_output: &Bound<'_, PyAny>,
    p_output: &Bound<'_, PyAny>,
) -> PyResult<()> {
    match utils::numeric_array2(matrix, "matrix")? {
        utils::NumericReadonlyArray2::F32(arr) => {
            let mut u_out = utils::output_array2::<f32>(u_output, "u_output", "float32")?;
            let mut p_out = utils::output_array2::<f32>(p_output, "p_output", "float32")?;
            let result =
                nabled_linalg::polar::compute_polar_view(&arr.as_array()).map_err(to_py_err)?;
            validate_output_shape(&u_out, result.u.dim(), "u_output")?;
            validate_output_shape(&p_out, result.p.dim(), "p_output")?;
            u_out.as_array_mut().assign(&result.u);
            p_out.as_array_mut().assign(&result.p);
            Ok(())
        }
        utils::NumericReadonlyArray2::F64(arr) => {
            let mut u_out = utils::output_array2::<f64>(u_output, "u_output", "float64")?;
            let mut p_out = utils::output_array2::<f64>(p_output, "p_output", "float64")?;
            let result =
                nabled_linalg::polar::compute_polar_view(&arr.as_array()).map_err(to_py_err)?;
            validate_output_shape(&u_out, result.u.dim(), "u_output")?;
            validate_output_shape(&p_out, result.p.dim(), "p_output")?;
            u_out.as_array_mut().assign(&result.u);
            p_out.as_array_mut().assign(&result.p);
            Ok(())
        }
        utils::NumericReadonlyArray2::C64(arr) => {
            let mut u_out =
                utils::output_array2::<num_complex::Complex64>(u_output, "u_output", "complex128")?;
            let mut p_out =
                utils::output_array2::<num_complex::Complex64>(p_output, "p_output", "complex128")?;
            let result = nabled_linalg::polar::compute_polar_complex_view(&arr.as_array())
                .map_err(to_py_err)?;
            validate_output_shape(&u_out, result.u.dim(), "u_output")?;
            validate_output_shape(&p_out, result.p.dim(), "p_output")?;
            u_out.as_array_mut().assign(&result.u);
            p_out.as_array_mut().assign(&result.p);
            Ok(())
        }
    }
}

/// Compute polar decomposition from precomputed SVD factors. Returns `(U, P)`.
#[pyfunction(name = "polar_compute_from_factors")]
pub fn compute_polar_from_factors<'py>(
    py: Python<'py>,
    u: &Bound<'py, PyAny>,
    singular_values: &Bound<'py, PyAny>,
    vt: &Bound<'py, PyAny>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    match (
        utils::numeric_array2(u, "u")?,
        utils::real_array1(singular_values, "singular_values")?,
        utils::numeric_array2(vt, "vt")?,
    ) {
        (
            utils::NumericReadonlyArray2::F32(u_arr),
            utils::RealReadonlyArray1::F32(s_arr),
            utils::NumericReadonlyArray2::F32(vt_arr),
        ) => {
            let result = nabled_linalg::polar::compute_polar_from_svd_view(
                &u_arr.as_array(),
                &s_arr.as_array(),
                &vt_arr.as_array(),
            )
            .map_err(to_py_err)?;
            Ok((utils::pyarray2_from_owned(py, result.u), utils::pyarray2_from_owned(py, result.p)))
        }
        (
            utils::NumericReadonlyArray2::F64(u_arr),
            utils::RealReadonlyArray1::F64(s_arr),
            utils::NumericReadonlyArray2::F64(vt_arr),
        ) => {
            let result = nabled_linalg::polar::compute_polar_from_svd_view(
                &u_arr.as_array(),
                &s_arr.as_array(),
                &vt_arr.as_array(),
            )
            .map_err(to_py_err)?;
            Ok((utils::pyarray2_from_owned(py, result.u), utils::pyarray2_from_owned(py, result.p)))
        }
        (
            utils::NumericReadonlyArray2::C64(u_arr),
            utils::RealReadonlyArray1::F64(s_arr),
            utils::NumericReadonlyArray2::C64(vt_arr),
        ) => {
            let result = nabled_linalg::polar::compute_polar_complex_from_svd_view(
                &u_arr.as_array(),
                &s_arr.as_array(),
                &vt_arr.as_array(),
            )
            .map_err(to_py_err)?;
            Ok((utils::pyarray2_from_owned(py, result.u), utils::pyarray2_from_owned(py, result.p)))
        }
        (utils::NumericReadonlyArray2::C64(_), _, utils::NumericReadonlyArray2::C64(_)) => {
            Err(complex_svd_component_error())
        }
        _ => Err(utils::matching_real_dtype_error(&["u", "singular_values", "vt"])),
    }
}

/// Compute polar decomposition from precomputed SVD factors into caller-provided outputs.
#[pyfunction(name = "polar_compute_from_factors_into")]
pub fn compute_polar_from_factors_into(
    u: &Bound<'_, PyAny>,
    singular_values: &Bound<'_, PyAny>,
    vt: &Bound<'_, PyAny>,
    u_output: &Bound<'_, PyAny>,
    p_output: &Bound<'_, PyAny>,
) -> PyResult<()> {
    match (
        utils::numeric_array2(u, "u")?,
        utils::real_array1(singular_values, "singular_values")?,
        utils::numeric_array2(vt, "vt")?,
    ) {
        (
            utils::NumericReadonlyArray2::F32(u_arr),
            utils::RealReadonlyArray1::F32(s_arr),
            utils::NumericReadonlyArray2::F32(vt_arr),
        ) => {
            let result = nabled_linalg::polar::compute_polar_from_svd_view(
                &u_arr.as_array(),
                &s_arr.as_array(),
                &vt_arr.as_array(),
            )
            .map_err(to_py_err)?;
            let mut u_out = utils::output_array2::<f32>(u_output, "u_output", "float32")?;
            let mut p_out = utils::output_array2::<f32>(p_output, "p_output", "float32")?;
            validate_output_shape(&u_out, result.u.dim(), "u_output")?;
            validate_output_shape(&p_out, result.p.dim(), "p_output")?;
            u_out.as_array_mut().assign(&result.u);
            p_out.as_array_mut().assign(&result.p);
            Ok(())
        }
        (
            utils::NumericReadonlyArray2::F64(u_arr),
            utils::RealReadonlyArray1::F64(s_arr),
            utils::NumericReadonlyArray2::F64(vt_arr),
        ) => {
            let result = nabled_linalg::polar::compute_polar_from_svd_view(
                &u_arr.as_array(),
                &s_arr.as_array(),
                &vt_arr.as_array(),
            )
            .map_err(to_py_err)?;
            let mut u_out = utils::output_array2::<f64>(u_output, "u_output", "float64")?;
            let mut p_out = utils::output_array2::<f64>(p_output, "p_output", "float64")?;
            validate_output_shape(&u_out, result.u.dim(), "u_output")?;
            validate_output_shape(&p_out, result.p.dim(), "p_output")?;
            u_out.as_array_mut().assign(&result.u);
            p_out.as_array_mut().assign(&result.p);
            Ok(())
        }
        (
            utils::NumericReadonlyArray2::C64(u_arr),
            utils::RealReadonlyArray1::F64(s_arr),
            utils::NumericReadonlyArray2::C64(vt_arr),
        ) => {
            let result = nabled_linalg::polar::compute_polar_complex_from_svd_view(
                &u_arr.as_array(),
                &s_arr.as_array(),
                &vt_arr.as_array(),
            )
            .map_err(to_py_err)?;
            let mut u_out =
                utils::output_array2::<num_complex::Complex64>(u_output, "u_output", "complex128")?;
            let mut p_out =
                utils::output_array2::<num_complex::Complex64>(p_output, "p_output", "complex128")?;
            validate_output_shape(&u_out, result.u.dim(), "u_output")?;
            validate_output_shape(&p_out, result.p.dim(), "p_output")?;
            u_out.as_array_mut().assign(&result.u);
            p_out.as_array_mut().assign(&result.p);
            Ok(())
        }
        (utils::NumericReadonlyArray2::C64(_), _, utils::NumericReadonlyArray2::C64(_)) => {
            Err(complex_svd_component_error())
        }
        _ => Err(utils::matching_real_dtype_error(&["u", "singular_values", "vt"])),
    }
}
