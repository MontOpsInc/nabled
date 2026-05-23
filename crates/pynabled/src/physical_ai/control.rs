//! Control bindings (LQR, DARE, pole placement, observer, gramian).

use nabled::control::dare::{dare_residual_norm, dare_solve};
use nabled::control::gramian::controllability_gramian;
use nabled::control::lqr::{LqrResult, discrete_lqr};
use nabled::control::observer::luenberger_gain;
use nabled::control::pole::place_poles;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::error::to_py_err;
use crate::utils;

/// Discrete LQR solution.
#[pyclass(name = "LqrResult")]
pub struct PyLqrResult {
    #[pyo3(get)]
    pub gain:    Py<PyAny>,
    #[pyo3(get)]
    pub riccati: Py<PyAny>,
}

fn lqr_result_to_py(py: Python<'_>, result: LqrResult<f64>) -> PyResult<PyLqrResult> {
    Ok(PyLqrResult {
        gain:    utils::pyarray2_from_owned(py, result.gain),
        riccati: utils::pyarray2_from_owned(py, result.riccati),
    })
}

/// Compute discrete-time LQR gain.
#[pyfunction]
pub fn discrete_lqr_py<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
    q: &Bound<'py, PyAny>,
    r: &Bound<'py, PyAny>,
) -> PyResult<PyLqrResult> {
    match (
        utils::real_array2(a, "a")?,
        utils::real_array2(b, "b")?,
        utils::real_array2(q, "q")?,
        utils::real_array2(r, "r")?,
    ) {
        (
            utils::RealReadonlyArray2::F64(a_arr),
            utils::RealReadonlyArray2::F64(b_arr),
            utils::RealReadonlyArray2::F64(q_arr),
            utils::RealReadonlyArray2::F64(r_arr),
        ) => {
            let result = discrete_lqr(
                &a_arr.as_array().to_owned(),
                &b_arr.as_array().to_owned(),
                &q_arr.as_array().to_owned(),
                &r_arr.as_array().to_owned(),
            )
            .map_err(to_py_err)?;
            lqr_result_to_py(py, result)
        }
        _ => Err(utils::matching_real_dtype_error(&["a", "b", "q", "r"])),
    }
}

/// Solve the discrete algebraic Riccati equation.
#[pyfunction]
pub fn dare_solve_py<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
    q: &Bound<'py, PyAny>,
    r: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match (
        utils::real_array2(a, "a")?,
        utils::real_array2(b, "b")?,
        utils::real_array2(q, "q")?,
        utils::real_array2(r, "r")?,
    ) {
        (
            utils::RealReadonlyArray2::F64(a_arr),
            utils::RealReadonlyArray2::F64(b_arr),
            utils::RealReadonlyArray2::F64(q_arr),
            utils::RealReadonlyArray2::F64(r_arr),
        ) => {
            let p = dare_solve(
                &a_arr.as_array().to_owned(),
                &b_arr.as_array().to_owned(),
                &q_arr.as_array().to_owned(),
                &r_arr.as_array().to_owned(),
            )
            .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, p))
        }
        _ => Err(utils::matching_real_dtype_error(&["a", "b", "q", "r"])),
    }
}

/// DARE algebraic residual norm.
#[pyfunction]
pub fn dare_residual_norm_py(
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    q: &Bound<'_, PyAny>,
    r: &Bound<'_, PyAny>,
    p: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    match (
        utils::real_array2(a, "a")?,
        utils::real_array2(b, "b")?,
        utils::real_array2(q, "q")?,
        utils::real_array2(r, "r")?,
        utils::real_array2(p, "p")?,
    ) {
        (
            utils::RealReadonlyArray2::F64(a_arr),
            utils::RealReadonlyArray2::F64(b_arr),
            utils::RealReadonlyArray2::F64(q_arr),
            utils::RealReadonlyArray2::F64(r_arr),
            utils::RealReadonlyArray2::F64(p_arr),
        ) => dare_residual_norm(
            &a_arr.as_array().to_owned(),
            &b_arr.as_array().to_owned(),
            &q_arr.as_array().to_owned(),
            &r_arr.as_array().to_owned(),
            &p_arr.as_array().to_owned(),
        )
        .map_err(to_py_err),
        _ => Err(utils::matching_real_dtype_error(&["a", "b", "q", "r", "p"])),
    }
}

/// Continuous controllability gramian.
#[pyfunction]
pub fn controllability_gramian_py<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    match (utils::real_array2(a, "a")?, utils::real_array2(b, "b")?) {
        (utils::RealReadonlyArray2::F64(a_arr), utils::RealReadonlyArray2::F64(b_arr)) => {
            let w =
                controllability_gramian(&a_arr.as_array().to_owned(), &b_arr.as_array().to_owned())
                    .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, w))
        }
        _ => Err(utils::matching_real_dtype_error(&["a", "b"])),
    }
}

/// Pole placement gain.
#[pyfunction]
pub fn place_poles_py<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
    poles: Vec<f64>,
) -> PyResult<Py<PyAny>> {
    match (utils::real_array2(a, "a")?, utils::real_array2(b, "b")?) {
        (utils::RealReadonlyArray2::F64(a_arr), utils::RealReadonlyArray2::F64(b_arr)) => {
            let k = place_poles(&a_arr.as_array().to_owned(), &b_arr.as_array().to_owned(), &poles)
                .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, k))
        }
        _ => Err(utils::matching_real_dtype_error(&["a", "b"])),
    }
}

/// Luenberger observer gain.
#[pyfunction]
pub fn luenberger_gain_py<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    c: &Bound<'py, PyAny>,
    poles: Vec<f64>,
) -> PyResult<Py<PyAny>> {
    match (utils::real_array2(a, "a")?, utils::real_array2(c, "c")?) {
        (utils::RealReadonlyArray2::F64(a_arr), utils::RealReadonlyArray2::F64(c_arr)) => {
            let l =
                luenberger_gain(&a_arr.as_array().to_owned(), &c_arr.as_array().to_owned(), &poles)
                    .map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, l))
        }
        _ => Err(utils::matching_real_dtype_error(&["a", "c"])),
    }
}

/// Discrete LQR into caller-provided gain and riccati buffers.
#[pyfunction]
pub fn discrete_lqr_into(
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    q: &Bound<'_, PyAny>,
    r: &Bound<'_, PyAny>,
    gain: &Bound<'_, PyAny>,
    riccati: &Bound<'_, PyAny>,
) -> PyResult<()> {
    match (
        utils::real_array2(a, "a")?,
        utils::real_array2(b, "b")?,
        utils::real_array2(q, "q")?,
        utils::real_array2(r, "r")?,
    ) {
        (
            utils::RealReadonlyArray2::F64(a_arr),
            utils::RealReadonlyArray2::F64(b_arr),
            utils::RealReadonlyArray2::F64(q_arr),
            utils::RealReadonlyArray2::F64(r_arr),
        ) => {
            let result = discrete_lqr(
                &a_arr.as_array().to_owned(),
                &b_arr.as_array().to_owned(),
                &q_arr.as_array().to_owned(),
                &r_arr.as_array().to_owned(),
            )
            .map_err(to_py_err)?;
            let mut gain_arr = utils::output_array2::<f64>(gain, "gain", "float64")?;
            let mut riccati_arr = utils::output_array2::<f64>(riccati, "riccati", "float64")?;
            gain_arr.as_array_mut().assign(&result.gain);
            riccati_arr.as_array_mut().assign(&result.riccati);
            Ok(())
        }
        _ => Err(utils::matching_real_dtype_error(&["a", "b", "q", "r", "gain", "riccati"])),
    }
}
