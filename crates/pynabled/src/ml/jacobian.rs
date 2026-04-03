//! Numerical Jacobian/gradient/Hessian bindings for Python callables.

use std::cell::RefCell;

use nabled_ml::jacobian::{JacobianConfig, JacobianError};
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;

use crate::error::to_py_err;
use crate::ml::callbacks::call_vector_function_f64;
use crate::utils;

fn jacobian_config(
    step_size: Option<f64>,
    tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<JacobianConfig<f64>> {
    JacobianConfig::new(
        step_size.unwrap_or(1e-6),
        tolerance.unwrap_or(1e-8),
        max_iterations.unwrap_or(100),
    )
    .map_err(to_py_err)
}

/// Compute a numerical Jacobian via forward differences.
#[pyfunction]
#[pyo3(signature = (function, x, step_size=None, tolerance=None, max_iterations=None))]
pub fn numerical_jacobian<'py>(
    py: Python<'py>,
    function: &Bound<'py, PyAny>,
    x: &Bound<'py, PyArray1<f64>>,
    step_size: Option<f64>,
    tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<Py<PyArray2<f64>>> {
    utils::require_contiguous(x)?;
    let config = jacobian_config(step_size, tolerance, max_iterations)?;
    let x_arr = x.readonly();
    let callback_error = RefCell::<Option<PyErr>>::default();

    let wrapped = |input: &ndarray::Array1<f64>| -> Result<ndarray::Array1<f64>, JacobianError> {
        match call_vector_function_f64(function, input) {
            Ok(value) => Ok(value),
            Err(err) => {
                *callback_error.borrow_mut() = Some(err);
                Err(JacobianError::FunctionError("python callback raised".to_string()))
            }
        }
    };

    let result = nabled_ml::jacobian::numerical_jacobian(&wrapped, &x_arr.as_array(), &config);
    if let Some(err) = callback_error.into_inner() {
        return Err(err);
    }
    Ok(PyArray2::from_owned_array(py, result.map_err(to_py_err)?).unbind())
}

/// Compute a numerical Jacobian via central differences.
#[pyfunction]
#[pyo3(signature = (function, x, step_size=None, tolerance=None, max_iterations=None))]
pub fn numerical_jacobian_central<'py>(
    py: Python<'py>,
    function: &Bound<'py, PyAny>,
    x: &Bound<'py, PyArray1<f64>>,
    step_size: Option<f64>,
    tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<Py<PyArray2<f64>>> {
    utils::require_contiguous(x)?;
    let config = jacobian_config(step_size, tolerance, max_iterations)?;
    let x_arr = x.readonly();
    let callback_error = RefCell::<Option<PyErr>>::default();

    let wrapped = |input: &ndarray::Array1<f64>| -> Result<ndarray::Array1<f64>, JacobianError> {
        match call_vector_function_f64(function, input) {
            Ok(value) => Ok(value),
            Err(err) => {
                *callback_error.borrow_mut() = Some(err);
                Err(JacobianError::FunctionError("python callback raised".to_string()))
            }
        }
    };

    let result =
        nabled_ml::jacobian::numerical_jacobian_central(&wrapped, &x_arr.as_array(), &config);
    if let Some(err) = callback_error.into_inner() {
        return Err(err);
    }
    Ok(PyArray2::from_owned_array(py, result.map_err(to_py_err)?).unbind())
}

/// Compute a numerical gradient for a scalar-valued Python callable.
#[pyfunction]
#[pyo3(signature = (function, x, step_size=None, tolerance=None, max_iterations=None))]
pub fn numerical_gradient<'py>(
    py: Python<'py>,
    function: &Bound<'py, PyAny>,
    x: &Bound<'py, PyArray1<f64>>,
    step_size: Option<f64>,
    tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<Py<PyArray1<f64>>> {
    utils::require_contiguous(x)?;
    let config = jacobian_config(step_size, tolerance, max_iterations)?;
    let x_arr = x.readonly();
    let callback_error = RefCell::<Option<PyErr>>::default();

    let wrapped = |input: &ndarray::Array1<f64>| -> Result<f64, JacobianError> {
        match crate::ml::callbacks::call_scalar_function_f64(function, input) {
            Ok(value) => Ok(value),
            Err(err) => {
                *callback_error.borrow_mut() = Some(err);
                Err(JacobianError::FunctionError("python callback raised".to_string()))
            }
        }
    };

    let result = nabled_ml::jacobian::numerical_gradient(&wrapped, &x_arr.as_array(), &config);
    if let Some(err) = callback_error.into_inner() {
        return Err(err);
    }
    Ok(PyArray1::from_owned_array(py, result.map_err(to_py_err)?).unbind())
}

/// Compute a numerical Hessian for a scalar-valued Python callable.
#[pyfunction]
#[pyo3(signature = (function, x, step_size=None, tolerance=None, max_iterations=None))]
pub fn numerical_hessian<'py>(
    py: Python<'py>,
    function: &Bound<'py, PyAny>,
    x: &Bound<'py, PyArray1<f64>>,
    step_size: Option<f64>,
    tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<Py<PyArray2<f64>>> {
    utils::require_contiguous(x)?;
    let config = jacobian_config(step_size, tolerance, max_iterations)?;
    let x_arr = x.readonly();
    let callback_error = RefCell::<Option<PyErr>>::default();

    let wrapped = |input: &ndarray::Array1<f64>| -> Result<f64, JacobianError> {
        match crate::ml::callbacks::call_scalar_function_f64(function, input) {
            Ok(value) => Ok(value),
            Err(err) => {
                *callback_error.borrow_mut() = Some(err);
                Err(JacobianError::FunctionError("python callback raised".to_string()))
            }
        }
    };

    let result = nabled_ml::jacobian::numerical_hessian(&wrapped, &x_arr.as_array(), &config);
    if let Some(err) = callback_error.into_inner() {
        return Err(err);
    }
    Ok(PyArray2::from_owned_array(py, result.map_err(to_py_err)?).unbind())
}
