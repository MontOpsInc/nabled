//! Numerical Jacobian/gradient/Hessian bindings for Python callables.

use std::cell::RefCell;

use nabled_core::scalar::NabledReal;
use nabled_ml::jacobian::{JacobianConfig, JacobianError};
use pyo3::prelude::*;

use crate::error::to_py_err;
use crate::ml::callbacks::{
    call_scalar_function_f32, call_scalar_function_f64, call_vector_function_f32,
    call_vector_function_f64,
};
use crate::utils::{self, RealReadonlyArray1};

fn jacobian_config<T: NabledReal>(
    step_size: Option<f64>,
    tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<JacobianConfig<T>>
where
    JacobianConfig<T>: Default,
{
    let mut config = JacobianConfig::<T>::default();
    if let Some(step_size) = step_size {
        config.step_size = utils::f64_to_real(step_size, "step_size")?;
    }
    if let Some(tolerance) = tolerance {
        config.tolerance = utils::f64_to_real(tolerance, "tolerance")?;
    }
    if let Some(max_iterations) = max_iterations {
        config.max_iterations = max_iterations;
    }
    config.validate().map_err(to_py_err)?;
    Ok(config)
}

/// Compute a numerical Jacobian via forward differences.
#[pyfunction]
#[pyo3(signature = (function, x, step_size=None, tolerance=None, max_iterations=None))]
pub fn numerical_jacobian<'py>(
    py: Python<'py>,
    function: &Bound<'py, PyAny>,
    x: &Bound<'py, PyAny>,
    step_size: Option<f64>,
    tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<Py<PyAny>> {
    match utils::real_array1(x, "x")? {
        RealReadonlyArray1::F32(x_arr) => {
            let config = jacobian_config::<f32>(step_size, tolerance, max_iterations)?;
            let callback_error = RefCell::<Option<PyErr>>::default();

            let wrapped =
                |input: &ndarray::Array1<f32>| -> Result<ndarray::Array1<f32>, JacobianError> {
                    match call_vector_function_f32(function, input) {
                        Ok(value) => Ok(value),
                        Err(err) => {
                            *callback_error.borrow_mut() = Some(err);
                            Err(JacobianError::FunctionError("python callback raised".to_string()))
                        }
                    }
                };

            let result =
                nabled_ml::jacobian::numerical_jacobian(&wrapped, &x_arr.as_array(), &config);
            if let Some(err) = callback_error.into_inner() {
                return Err(err);
            }
            Ok(utils::pyarray2_from_owned(py, result.map_err(to_py_err)?))
        }
        RealReadonlyArray1::F64(x_arr) => {
            let config = jacobian_config::<f64>(step_size, tolerance, max_iterations)?;
            let callback_error = RefCell::<Option<PyErr>>::default();

            let wrapped =
                |input: &ndarray::Array1<f64>| -> Result<ndarray::Array1<f64>, JacobianError> {
                    match call_vector_function_f64(function, input) {
                        Ok(value) => Ok(value),
                        Err(err) => {
                            *callback_error.borrow_mut() = Some(err);
                            Err(JacobianError::FunctionError("python callback raised".to_string()))
                        }
                    }
                };

            let result =
                nabled_ml::jacobian::numerical_jacobian(&wrapped, &x_arr.as_array(), &config);
            if let Some(err) = callback_error.into_inner() {
                return Err(err);
            }
            Ok(utils::pyarray2_from_owned(py, result.map_err(to_py_err)?))
        }
    }
}

/// Compute a numerical Jacobian via central differences.
#[pyfunction]
#[pyo3(signature = (function, x, step_size=None, tolerance=None, max_iterations=None))]
pub fn numerical_jacobian_central<'py>(
    py: Python<'py>,
    function: &Bound<'py, PyAny>,
    x: &Bound<'py, PyAny>,
    step_size: Option<f64>,
    tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<Py<PyAny>> {
    match utils::real_array1(x, "x")? {
        RealReadonlyArray1::F32(x_arr) => {
            let config = jacobian_config::<f32>(step_size, tolerance, max_iterations)?;
            let callback_error = RefCell::<Option<PyErr>>::default();

            let wrapped =
                |input: &ndarray::Array1<f32>| -> Result<ndarray::Array1<f32>, JacobianError> {
                    match call_vector_function_f32(function, input) {
                        Ok(value) => Ok(value),
                        Err(err) => {
                            *callback_error.borrow_mut() = Some(err);
                            Err(JacobianError::FunctionError("python callback raised".to_string()))
                        }
                    }
                };

            let result = nabled_ml::jacobian::numerical_jacobian_central(
                &wrapped,
                &x_arr.as_array(),
                &config,
            );
            if let Some(err) = callback_error.into_inner() {
                return Err(err);
            }
            Ok(utils::pyarray2_from_owned(py, result.map_err(to_py_err)?))
        }
        RealReadonlyArray1::F64(x_arr) => {
            let config = jacobian_config::<f64>(step_size, tolerance, max_iterations)?;
            let callback_error = RefCell::<Option<PyErr>>::default();

            let wrapped =
                |input: &ndarray::Array1<f64>| -> Result<ndarray::Array1<f64>, JacobianError> {
                    match call_vector_function_f64(function, input) {
                        Ok(value) => Ok(value),
                        Err(err) => {
                            *callback_error.borrow_mut() = Some(err);
                            Err(JacobianError::FunctionError("python callback raised".to_string()))
                        }
                    }
                };

            let result = nabled_ml::jacobian::numerical_jacobian_central(
                &wrapped,
                &x_arr.as_array(),
                &config,
            );
            if let Some(err) = callback_error.into_inner() {
                return Err(err);
            }
            Ok(utils::pyarray2_from_owned(py, result.map_err(to_py_err)?))
        }
    }
}

/// Compute a numerical gradient for a scalar-valued Python callable.
#[pyfunction]
#[pyo3(signature = (function, x, step_size=None, tolerance=None, max_iterations=None))]
pub fn numerical_gradient<'py>(
    py: Python<'py>,
    function: &Bound<'py, PyAny>,
    x: &Bound<'py, PyAny>,
    step_size: Option<f64>,
    tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<Py<PyAny>> {
    match utils::real_array1(x, "x")? {
        RealReadonlyArray1::F32(x_arr) => {
            let config = jacobian_config::<f32>(step_size, tolerance, max_iterations)?;
            let callback_error = RefCell::<Option<PyErr>>::default();

            let wrapped = |input: &ndarray::Array1<f32>| -> Result<f32, JacobianError> {
                match call_scalar_function_f32(function, input) {
                    Ok(value) => Ok(value),
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        Err(JacobianError::FunctionError("python callback raised".to_string()))
                    }
                }
            };

            let result =
                nabled_ml::jacobian::numerical_gradient(&wrapped, &x_arr.as_array(), &config);
            if let Some(err) = callback_error.into_inner() {
                return Err(err);
            }
            Ok(utils::pyarray1_from_owned(py, result.map_err(to_py_err)?))
        }
        RealReadonlyArray1::F64(x_arr) => {
            let config = jacobian_config::<f64>(step_size, tolerance, max_iterations)?;
            let callback_error = RefCell::<Option<PyErr>>::default();

            let wrapped = |input: &ndarray::Array1<f64>| -> Result<f64, JacobianError> {
                match call_scalar_function_f64(function, input) {
                    Ok(value) => Ok(value),
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        Err(JacobianError::FunctionError("python callback raised".to_string()))
                    }
                }
            };

            let result =
                nabled_ml::jacobian::numerical_gradient(&wrapped, &x_arr.as_array(), &config);
            if let Some(err) = callback_error.into_inner() {
                return Err(err);
            }
            Ok(utils::pyarray1_from_owned(py, result.map_err(to_py_err)?))
        }
    }
}

/// Compute a numerical Hessian for a scalar-valued Python callable.
#[pyfunction]
#[pyo3(signature = (function, x, step_size=None, tolerance=None, max_iterations=None))]
pub fn numerical_hessian<'py>(
    py: Python<'py>,
    function: &Bound<'py, PyAny>,
    x: &Bound<'py, PyAny>,
    step_size: Option<f64>,
    tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<Py<PyAny>> {
    match utils::real_array1(x, "x")? {
        RealReadonlyArray1::F32(x_arr) => {
            let config = jacobian_config::<f32>(step_size, tolerance, max_iterations)?;
            let callback_error = RefCell::<Option<PyErr>>::default();

            let wrapped = |input: &ndarray::Array1<f32>| -> Result<f32, JacobianError> {
                match call_scalar_function_f32(function, input) {
                    Ok(value) => Ok(value),
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        Err(JacobianError::FunctionError("python callback raised".to_string()))
                    }
                }
            };

            let result =
                nabled_ml::jacobian::numerical_hessian(&wrapped, &x_arr.as_array(), &config);
            if let Some(err) = callback_error.into_inner() {
                return Err(err);
            }
            Ok(utils::pyarray2_from_owned(py, result.map_err(to_py_err)?))
        }
        RealReadonlyArray1::F64(x_arr) => {
            let config = jacobian_config::<f64>(step_size, tolerance, max_iterations)?;
            let callback_error = RefCell::<Option<PyErr>>::default();

            let wrapped = |input: &ndarray::Array1<f64>| -> Result<f64, JacobianError> {
                match call_scalar_function_f64(function, input) {
                    Ok(value) => Ok(value),
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        Err(JacobianError::FunctionError("python callback raised".to_string()))
                    }
                }
            };

            let result =
                nabled_ml::jacobian::numerical_hessian(&wrapped, &x_arr.as_array(), &config);
            if let Some(err) = callback_error.into_inner() {
                return Err(err);
            }
            Ok(utils::pyarray2_from_owned(py, result.map_err(to_py_err)?))
        }
    }
}
