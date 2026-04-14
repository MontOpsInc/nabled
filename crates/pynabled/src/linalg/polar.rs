//! Polar decomposition bindings for Python.

use nabled_core::scalar::NabledReal;
use ndarray::linalg::general_mat_mul;
use ndarray::{Array2, ArrayView1, ArrayView2, ArrayViewMut2};
use num_complex::Complex64;
use numpy::Element;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::error::to_py_err;
use crate::utils;

fn complex_svd_component_error() -> PyErr {
    pyo3::exceptions::PyTypeError::new_err(
        "u, vt must both be complex128 and singular_values must be float64 for complex SVD results",
    )
}

fn compute_real_polar_from_svd_into<T>(
    u: &ArrayView2<'_, T>,
    singular_values: &ArrayView1<'_, T>,
    vt: &ArrayView2<'_, T>,
    u_output: &mut ArrayViewMut2<'_, T>,
    p_output: &mut ArrayViewMut2<'_, T>,
) -> PyResult<()>
where
    T: Element + NabledReal,
{
    if u.is_empty() || singular_values.is_empty() || vt.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("Matrix cannot be empty"));
    }
    if u.nrows() != vt.ncols() {
        return Err(pyo3::exceptions::PyValueError::new_err("Matrix must be square"));
    }
    if u.ncols() != singular_values.len() || vt.nrows() != singular_values.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("Polar decomposition failed"));
    }
    if u.iter().any(|value| !value.is_finite())
        || vt.iter().any(|value| !value.is_finite())
        || singular_values.iter().any(|value| !value.is_finite())
    {
        return Err(pyo3::exceptions::PyValueError::new_err("Numerical instability detected"));
    }

    let expected = (u.nrows(), vt.ncols());
    if u_output.dim() != expected {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "u_output must have shape {expected:?}",
        )));
    }
    if p_output.dim() != expected {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "p_output must have shape {expected:?}",
        )));
    }

    let mut psd_scratch = Array2::<T>::zeros(vt.dim());
    for row in 0..singular_values.len() {
        let scale = singular_values[row];
        for col in 0..vt.ncols() {
            psd_scratch[[row, col]] = vt[[row, col]] * scale;
        }
    }

    general_mat_mul(T::one(), u, vt, T::zero(), u_output);
    general_mat_mul(T::one(), &vt.t(), &psd_scratch.view(), T::zero(), p_output);
    Ok(())
}

fn compute_complex_polar_from_svd_into(
    u: &ArrayView2<'_, Complex64>,
    singular_values: &ArrayView1<'_, f64>,
    vt: &ArrayView2<'_, Complex64>,
    u_output: &mut ArrayViewMut2<'_, Complex64>,
    p_output: &mut ArrayViewMut2<'_, Complex64>,
) -> PyResult<()> {
    if u.is_empty() || singular_values.is_empty() || vt.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("Matrix cannot be empty"));
    }
    if u.nrows() != vt.ncols() {
        return Err(pyo3::exceptions::PyValueError::new_err("Matrix must be square"));
    }
    if u.ncols() != singular_values.len() || vt.nrows() != singular_values.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("Polar decomposition failed"));
    }
    if u.iter().any(|value| !value.re.is_finite() || !value.im.is_finite())
        || vt.iter().any(|value| !value.re.is_finite() || !value.im.is_finite())
        || singular_values.iter().any(|value| !value.is_finite())
    {
        return Err(pyo3::exceptions::PyValueError::new_err("Numerical instability detected"));
    }

    let expected = (u.nrows(), vt.ncols());
    if u_output.dim() != expected {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "u_output must have shape {expected:?}",
        )));
    }
    if p_output.dim() != expected {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "p_output must have shape {expected:?}",
        )));
    }

    general_mat_mul(Complex64::new(1.0, 0.0), u, vt, Complex64::new(0.0, 0.0), u_output);

    for i in 0..p_output.nrows() {
        for j in 0..p_output.ncols() {
            let mut value = Complex64::new(0.0, 0.0);
            for k in 0..singular_values.len() {
                value += vt[[k, i]].conj() * Complex64::new(singular_values[k], 0.0) * vt[[k, j]];
            }
            p_output[[i, j]] = value;
        }
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
            let svd = nabled_linalg::svd::decompose_view(&arr.as_array()).map_err(to_py_err)?;
            compute_real_polar_from_svd_into(
                &svd.u.view(),
                &svd.singular_values.view(),
                &svd.vt.view(),
                &mut u_out.as_array_mut(),
                &mut p_out.as_array_mut(),
            )
        }
        utils::NumericReadonlyArray2::F64(arr) => {
            let mut u_out = utils::output_array2::<f64>(u_output, "u_output", "float64")?;
            let mut p_out = utils::output_array2::<f64>(p_output, "p_output", "float64")?;
            let svd = nabled_linalg::svd::decompose_view(&arr.as_array()).map_err(to_py_err)?;
            compute_real_polar_from_svd_into(
                &svd.u.view(),
                &svd.singular_values.view(),
                &svd.vt.view(),
                &mut u_out.as_array_mut(),
                &mut p_out.as_array_mut(),
            )
        }
        utils::NumericReadonlyArray2::C64(arr) => {
            let mut u_out = utils::output_array2::<Complex64>(u_output, "u_output", "complex128")?;
            let mut p_out = utils::output_array2::<Complex64>(p_output, "p_output", "complex128")?;
            let svd =
                nabled_linalg::svd::decompose_complex_view(&arr.as_array()).map_err(to_py_err)?;
            compute_complex_polar_from_svd_into(
                &svd.u.view(),
                &svd.singular_values.view(),
                &svd.vt.view(),
                &mut u_out.as_array_mut(),
                &mut p_out.as_array_mut(),
            )
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
            let mut u_out = utils::output_array2::<f32>(u_output, "u_output", "float32")?;
            let mut p_out = utils::output_array2::<f32>(p_output, "p_output", "float32")?;
            compute_real_polar_from_svd_into(
                &u_arr.as_array(),
                &s_arr.as_array(),
                &vt_arr.as_array(),
                &mut u_out.as_array_mut(),
                &mut p_out.as_array_mut(),
            )
        }
        (
            utils::NumericReadonlyArray2::F64(u_arr),
            utils::RealReadonlyArray1::F64(s_arr),
            utils::NumericReadonlyArray2::F64(vt_arr),
        ) => {
            let mut u_out = utils::output_array2::<f64>(u_output, "u_output", "float64")?;
            let mut p_out = utils::output_array2::<f64>(p_output, "p_output", "float64")?;
            compute_real_polar_from_svd_into(
                &u_arr.as_array(),
                &s_arr.as_array(),
                &vt_arr.as_array(),
                &mut u_out.as_array_mut(),
                &mut p_out.as_array_mut(),
            )
        }
        (
            utils::NumericReadonlyArray2::C64(u_arr),
            utils::RealReadonlyArray1::F64(s_arr),
            utils::NumericReadonlyArray2::C64(vt_arr),
        ) => {
            let mut u_out = utils::output_array2::<Complex64>(u_output, "u_output", "complex128")?;
            let mut p_out = utils::output_array2::<Complex64>(p_output, "p_output", "complex128")?;
            compute_complex_polar_from_svd_into(
                &u_arr.as_array(),
                &s_arr.as_array(),
                &vt_arr.as_array(),
                &mut u_out.as_array_mut(),
                &mut p_out.as_array_mut(),
            )
        }
        (utils::NumericReadonlyArray2::C64(_), _, utils::NumericReadonlyArray2::C64(_)) => {
            Err(complex_svd_component_error())
        }
        _ => Err(utils::matching_real_dtype_error(&["u", "singular_values", "vt"])),
    }
}
