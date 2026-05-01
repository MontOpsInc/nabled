//! Optimization bindings for Python callables.

use std::any::TypeId;
use std::cell::RefCell;

use nabled_core::scalar::NabledReal;
use nabled_ml::optimization::{
    AdamConfig, BFGSConfig, LineSearchConfig, MomentumConfig, OptimizationError,
    ProjectedGradientConfig, RMSPropConfig, SGDConfig,
};
use num_complex::Complex64;
use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;

use crate::error::to_py_err;
use crate::ml::callbacks::{
    call_scalar_function_complex, call_scalar_function_f32, call_scalar_function_f64,
    call_vector_function_complex, call_vector_function_complex_with_iteration,
    call_vector_function_f32, call_vector_function_f32_with_iteration, call_vector_function_f64,
    call_vector_function_f64_with_iteration,
};
use crate::utils::{self, RealReadonlyArray1};

fn line_search_config<T: NabledReal>(
    initial_step: Option<f64>,
    contraction: Option<f64>,
    sufficient_decrease: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<LineSearchConfig<T>>
where
    LineSearchConfig<T>: Default,
{
    let mut config = LineSearchConfig::<T>::default();
    if let Some(value) = initial_step {
        config.initial_step = utils::f64_to_real(value, "initial_step")?;
    }
    if let Some(value) = contraction {
        config.contraction = utils::f64_to_real(value, "contraction")?;
    }
    if let Some(value) = sufficient_decrease {
        config.sufficient_decrease = utils::f64_to_real(value, "sufficient_decrease")?;
    }
    if let Some(value) = max_iterations {
        config.max_iterations = value;
    }
    Ok(config)
}

fn is_f32<T: 'static>() -> bool { TypeId::of::<T>() == TypeId::of::<f32>() }

fn sgd_config<T: NabledReal>(
    learning_rate: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<SGDConfig<T>>
where
    SGDConfig<T>: Default,
{
    let mut config = SGDConfig::<T>::default();
    if let Some(value) = learning_rate {
        config.learning_rate = utils::f64_to_real(value, "learning_rate")?;
    }
    if let Some(value) = max_iterations {
        config.max_iterations = value;
    }
    if let Some(value) = tolerance {
        config.tolerance = utils::f64_to_real(value, "tolerance")?;
    } else if is_f32::<T>() {
        config.tolerance = utils::f64_to_real(1e-5, "tolerance")?;
    }
    Ok(config)
}

fn adam_config<T: NabledReal>(
    learning_rate: Option<f64>,
    beta1: Option<f64>,
    beta2: Option<f64>,
    epsilon: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<AdamConfig<T>>
where
    AdamConfig<T>: Default,
{
    let mut config = AdamConfig::<T>::default();
    if let Some(value) = learning_rate {
        config.learning_rate = utils::f64_to_real(value, "learning_rate")?;
    }
    if let Some(value) = beta1 {
        config.beta1 = utils::f64_to_real(value, "beta1")?;
    }
    if let Some(value) = beta2 {
        config.beta2 = utils::f64_to_real(value, "beta2")?;
    }
    if let Some(value) = epsilon {
        config.epsilon = utils::f64_to_real(value, "epsilon")?;
    } else if is_f32::<T>() {
        config.epsilon = utils::f64_to_real(1e-6, "epsilon")?;
    }
    if let Some(value) = max_iterations {
        config.max_iterations = value;
    }
    if let Some(value) = tolerance {
        config.tolerance = utils::f64_to_real(value, "tolerance")?;
    } else if is_f32::<T>() {
        config.tolerance = utils::f64_to_real(1e-5, "tolerance")?;
    }
    Ok(config)
}

fn momentum_config<T: NabledReal>(
    learning_rate: Option<f64>,
    momentum: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<MomentumConfig<T>>
where
    MomentumConfig<T>: Default,
{
    let mut config = MomentumConfig::<T>::default();
    if let Some(value) = learning_rate {
        config.learning_rate = utils::f64_to_real(value, "learning_rate")?;
    }
    if let Some(value) = momentum {
        config.momentum = utils::f64_to_real(value, "momentum")?;
    }
    if let Some(value) = max_iterations {
        config.max_iterations = value;
    }
    if let Some(value) = tolerance {
        config.tolerance = utils::f64_to_real(value, "tolerance")?;
    } else if is_f32::<T>() {
        config.tolerance = utils::f64_to_real(1e-5, "tolerance")?;
    }
    Ok(config)
}

fn rmsprop_config<T: NabledReal>(
    learning_rate: Option<f64>,
    rho: Option<f64>,
    epsilon: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<RMSPropConfig<T>>
where
    RMSPropConfig<T>: Default,
{
    let mut config = RMSPropConfig::<T>::default();
    if let Some(value) = learning_rate {
        config.learning_rate = utils::f64_to_real(value, "learning_rate")?;
    }
    if let Some(value) = rho {
        config.rho = utils::f64_to_real(value, "rho")?;
    }
    if let Some(value) = epsilon {
        config.epsilon = utils::f64_to_real(value, "epsilon")?;
    } else if is_f32::<T>() {
        config.epsilon = utils::f64_to_real(1e-6, "epsilon")?;
    }
    if let Some(value) = max_iterations {
        config.max_iterations = value;
    }
    if let Some(value) = tolerance {
        config.tolerance = utils::f64_to_real(value, "tolerance")?;
    } else if is_f32::<T>() {
        config.tolerance = utils::f64_to_real(1e-5, "tolerance")?;
    }
    Ok(config)
}

fn projected_gradient_config<T: NabledReal>(
    learning_rate: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<ProjectedGradientConfig<T>>
where
    ProjectedGradientConfig<T>: Default,
{
    let mut config = ProjectedGradientConfig::<T>::default();
    if let Some(value) = learning_rate {
        config.learning_rate = utils::f64_to_real(value, "learning_rate")?;
    }
    if let Some(value) = max_iterations {
        config.max_iterations = value;
    }
    if let Some(value) = tolerance {
        config.tolerance = utils::f64_to_real(value, "tolerance")?;
    } else if is_f32::<T>() {
        config.tolerance = utils::f64_to_real(1e-5, "tolerance")?;
    }
    Ok(config)
}

fn bfgs_config<T: NabledReal>(
    step_size: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
    curvature_tolerance: Option<f64>,
) -> PyResult<BFGSConfig<T>>
where
    BFGSConfig<T>: Default,
{
    let mut config = BFGSConfig::<T>::default();
    if let Some(value) = step_size {
        config.step_size = utils::f64_to_real(value, "step_size")?;
    }
    if let Some(value) = max_iterations {
        config.max_iterations = value;
    }
    if let Some(value) = tolerance {
        config.tolerance = utils::f64_to_real(value, "tolerance")?;
    } else if is_f32::<T>() {
        config.tolerance = utils::f64_to_real(1e-5, "tolerance")?;
    }
    if let Some(value) = curvature_tolerance {
        config.curvature_tolerance = utils::f64_to_real(value, "curvature_tolerance")?;
    } else if is_f32::<T>() {
        config.curvature_tolerance = utils::f64_to_real(1e-6, "curvature_tolerance")?;
    }
    Ok(config)
}

fn map_callback_error<T>(
    callback_error: &RefCell<Option<PyErr>>,
    result: Result<T, OptimizationError>,
) -> PyResult<T> {
    if let Some(err) = callback_error.borrow_mut().take() {
        return Err(err);
    }
    result.map_err(to_py_err)
}

/// Perform Armijo backtracking line search.
#[pyfunction]
#[pyo3(signature = (point, direction, objective, gradient, initial_step=None, contraction=None, sufficient_decrease=None, max_iterations=None))]
pub fn backtracking_line_search(
    point: &Bound<'_, PyAny>,
    direction: &Bound<'_, PyAny>,
    objective: &Bound<'_, PyAny>,
    gradient: &Bound<'_, PyAny>,
    initial_step: Option<f64>,
    contraction: Option<f64>,
    sufficient_decrease: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<f64> {
    match (utils::real_array1(point, "point")?, utils::real_array1(direction, "direction")?) {
        (RealReadonlyArray1::F32(point_arr), RealReadonlyArray1::F32(direction_arr)) => {
            let config = line_search_config::<f32>(
                initial_step,
                contraction,
                sufficient_decrease,
                max_iterations,
            )?;
            let callback_error = RefCell::<Option<PyErr>>::default();

            let objective_fn = |input: &ndarray::Array1<f32>| -> f32 {
                match call_scalar_function_f32(objective, input) {
                    Ok(value) => value,
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        f32::NAN
                    }
                }
            };
            let gradient_fn = |input: &ndarray::Array1<f32>| -> ndarray::Array1<f32> {
                match call_vector_function_f32(gradient, input) {
                    Ok(value) => value,
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        ndarray::Array1::from_elem(input.len(), f32::NAN)
                    }
                }
            };

            map_callback_error(
                &callback_error,
                nabled_ml::optimization::backtracking_line_search(
                    &point_arr.as_array(),
                    &direction_arr.as_array(),
                    objective_fn,
                    gradient_fn,
                    &config,
                ),
            )
            .map(f64::from)
        }
        (RealReadonlyArray1::F64(point_arr), RealReadonlyArray1::F64(direction_arr)) => {
            let config = line_search_config::<f64>(
                initial_step,
                contraction,
                sufficient_decrease,
                max_iterations,
            )?;
            let callback_error = RefCell::<Option<PyErr>>::default();

            let objective_fn = |input: &ndarray::Array1<f64>| -> f64 {
                match call_scalar_function_f64(objective, input) {
                    Ok(value) => value,
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        f64::NAN
                    }
                }
            };
            let gradient_fn = |input: &ndarray::Array1<f64>| -> ndarray::Array1<f64> {
                match call_vector_function_f64(gradient, input) {
                    Ok(value) => value,
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        ndarray::Array1::from_elem(input.len(), f64::NAN)
                    }
                }
            };

            map_callback_error(
                &callback_error,
                nabled_ml::optimization::backtracking_line_search(
                    &point_arr.as_array(),
                    &direction_arr.as_array(),
                    objective_fn,
                    gradient_fn,
                    &config,
                ),
            )
        }
        _ => Err(utils::matching_real_dtype_error(&["point", "direction"])),
    }
}

/// Perform Armijo backtracking line search for complex vectors.
#[pyfunction]
#[pyo3(signature = (point, direction, objective, gradient, initial_step=None, contraction=None, sufficient_decrease=None, max_iterations=None))]
pub fn backtracking_line_search_complex(
    point: &Bound<'_, PyArray1<Complex64>>,
    direction: &Bound<'_, PyArray1<Complex64>>,
    objective: &Bound<'_, PyAny>,
    gradient: &Bound<'_, PyAny>,
    initial_step: Option<f64>,
    contraction: Option<f64>,
    sufficient_decrease: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<f64> {
    utils::require_contiguous(point)?;
    utils::require_contiguous(direction)?;
    let config =
        line_search_config::<f64>(initial_step, contraction, sufficient_decrease, max_iterations)?;
    let point_arr = point.readonly();
    let direction_arr = direction.readonly();
    let callback_error = RefCell::<Option<PyErr>>::default();

    let objective_fn = |input: &ndarray::Array1<Complex64>| -> f64 {
        match call_scalar_function_complex(objective, input) {
            Ok(value) => value,
            Err(err) => {
                *callback_error.borrow_mut() = Some(err);
                f64::NAN
            }
        }
    };
    let gradient_fn = |input: &ndarray::Array1<Complex64>| -> ndarray::Array1<Complex64> {
        match call_vector_function_complex(gradient, input) {
            Ok(value) => value,
            Err(err) => {
                *callback_error.borrow_mut() = Some(err);
                ndarray::Array1::from_elem(input.len(), Complex64::new(f64::NAN, f64::NAN))
            }
        }
    };

    map_callback_error(
        &callback_error,
        nabled_ml::optimization::backtracking_line_search_complex(
            &point_arr.as_array(),
            &direction_arr.as_array(),
            objective_fn,
            gradient_fn,
            &config,
        ),
    )
}

/// Minimize an objective with fixed-step gradient descent.
#[pyfunction]
#[pyo3(signature = (initial, objective, gradient, learning_rate=None, max_iterations=None, tolerance=None))]
pub fn gradient_descent<'py>(
    py: Python<'py>,
    initial: &Bound<'py, PyAny>,
    objective: &Bound<'py, PyAny>,
    gradient: &Bound<'py, PyAny>,
    learning_rate: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<Py<PyAny>> {
    match utils::real_array1(initial, "initial")? {
        RealReadonlyArray1::F32(initial_arr) => {
            let config = sgd_config::<f32>(learning_rate, max_iterations, tolerance)?;
            let callback_error = RefCell::<Option<PyErr>>::default();

            let objective_fn = |input: &ndarray::Array1<f32>| -> f32 {
                match call_scalar_function_f32(objective, input) {
                    Ok(value) => value,
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        f32::NAN
                    }
                }
            };
            let gradient_fn = |input: &ndarray::Array1<f32>| -> ndarray::Array1<f32> {
                match call_vector_function_f32(gradient, input) {
                    Ok(value) => value,
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        ndarray::Array1::from_elem(input.len(), f32::NAN)
                    }
                }
            };

            let result = map_callback_error(
                &callback_error,
                nabled_ml::optimization::gradient_descent(
                    &initial_arr.as_array(),
                    objective_fn,
                    gradient_fn,
                    &config,
                ),
            )?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
        RealReadonlyArray1::F64(initial_arr) => {
            let config = sgd_config::<f64>(learning_rate, max_iterations, tolerance)?;
            let callback_error = RefCell::<Option<PyErr>>::default();

            let objective_fn = |input: &ndarray::Array1<f64>| -> f64 {
                match call_scalar_function_f64(objective, input) {
                    Ok(value) => value,
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        f64::NAN
                    }
                }
            };
            let gradient_fn = |input: &ndarray::Array1<f64>| -> ndarray::Array1<f64> {
                match call_vector_function_f64(gradient, input) {
                    Ok(value) => value,
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        ndarray::Array1::from_elem(input.len(), f64::NAN)
                    }
                }
            };

            let result = map_callback_error(
                &callback_error,
                nabled_ml::optimization::gradient_descent(
                    &initial_arr.as_array(),
                    objective_fn,
                    gradient_fn,
                    &config,
                ),
            )?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
    }
}

/// Minimize a complex objective with fixed-step gradient descent.
#[pyfunction]
#[pyo3(signature = (initial, objective, gradient, learning_rate=None, max_iterations=None, tolerance=None))]
pub fn gradient_descent_complex<'py>(
    py: Python<'py>,
    initial: &Bound<'py, PyArray1<Complex64>>,
    objective: &Bound<'py, PyAny>,
    gradient: &Bound<'py, PyAny>,
    learning_rate: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<Py<PyArray1<Complex64>>> {
    utils::require_contiguous(initial)?;
    let config = sgd_config::<f64>(learning_rate, max_iterations, tolerance)?;
    let initial_arr = initial.readonly();
    let callback_error = RefCell::<Option<PyErr>>::default();

    let objective_fn = |input: &ndarray::Array1<Complex64>| -> f64 {
        match call_scalar_function_complex(objective, input) {
            Ok(value) => value,
            Err(err) => {
                *callback_error.borrow_mut() = Some(err);
                f64::NAN
            }
        }
    };
    let gradient_fn = |input: &ndarray::Array1<Complex64>| -> ndarray::Array1<Complex64> {
        match call_vector_function_complex(gradient, input) {
            Ok(value) => value,
            Err(err) => {
                *callback_error.borrow_mut() = Some(err);
                ndarray::Array1::from_elem(input.len(), Complex64::new(f64::NAN, f64::NAN))
            }
        }
    };

    let result = map_callback_error(
        &callback_error,
        nabled_ml::optimization::gradient_descent_complex(
            &initial_arr.as_array(),
            objective_fn,
            gradient_fn,
            &config,
        ),
    )?;
    Ok(PyArray1::from_owned_array(py, result).unbind())
}

/// Minimize an objective with Adam.
#[pyfunction]
#[pyo3(signature = (initial, objective, gradient, learning_rate=None, beta1=None, beta2=None, epsilon=None, max_iterations=None, tolerance=None))]
pub fn adam<'py>(
    py: Python<'py>,
    initial: &Bound<'py, PyAny>,
    objective: &Bound<'py, PyAny>,
    gradient: &Bound<'py, PyAny>,
    learning_rate: Option<f64>,
    beta1: Option<f64>,
    beta2: Option<f64>,
    epsilon: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<Py<PyAny>> {
    match utils::real_array1(initial, "initial")? {
        RealReadonlyArray1::F32(initial_arr) => {
            let config = adam_config::<f32>(
                learning_rate,
                beta1,
                beta2,
                epsilon,
                max_iterations,
                tolerance,
            )?;
            let callback_error = RefCell::<Option<PyErr>>::default();

            let objective_fn = |input: &ndarray::Array1<f32>| -> f32 {
                match call_scalar_function_f32(objective, input) {
                    Ok(value) => value,
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        f32::NAN
                    }
                }
            };
            let gradient_fn = |input: &ndarray::Array1<f32>| -> ndarray::Array1<f32> {
                match call_vector_function_f32(gradient, input) {
                    Ok(value) => value,
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        ndarray::Array1::from_elem(input.len(), f32::NAN)
                    }
                }
            };

            let result = map_callback_error(
                &callback_error,
                nabled_ml::optimization::adam(
                    &initial_arr.as_array(),
                    objective_fn,
                    gradient_fn,
                    &config,
                ),
            )?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
        RealReadonlyArray1::F64(initial_arr) => {
            let config = adam_config::<f64>(
                learning_rate,
                beta1,
                beta2,
                epsilon,
                max_iterations,
                tolerance,
            )?;
            let callback_error = RefCell::<Option<PyErr>>::default();

            let objective_fn = |input: &ndarray::Array1<f64>| -> f64 {
                match call_scalar_function_f64(objective, input) {
                    Ok(value) => value,
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        f64::NAN
                    }
                }
            };
            let gradient_fn = |input: &ndarray::Array1<f64>| -> ndarray::Array1<f64> {
                match call_vector_function_f64(gradient, input) {
                    Ok(value) => value,
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        ndarray::Array1::from_elem(input.len(), f64::NAN)
                    }
                }
            };

            let result = map_callback_error(
                &callback_error,
                nabled_ml::optimization::adam(
                    &initial_arr.as_array(),
                    objective_fn,
                    gradient_fn,
                    &config,
                ),
            )?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
    }
}

/// Minimize a complex objective with Adam.
#[pyfunction]
#[pyo3(signature = (initial, objective, gradient, learning_rate=None, beta1=None, beta2=None, epsilon=None, max_iterations=None, tolerance=None))]
pub fn adam_complex<'py>(
    py: Python<'py>,
    initial: &Bound<'py, PyArray1<Complex64>>,
    objective: &Bound<'py, PyAny>,
    gradient: &Bound<'py, PyAny>,
    learning_rate: Option<f64>,
    beta1: Option<f64>,
    beta2: Option<f64>,
    epsilon: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<Py<PyArray1<Complex64>>> {
    utils::require_contiguous(initial)?;
    let config =
        adam_config::<f64>(learning_rate, beta1, beta2, epsilon, max_iterations, tolerance)?;
    let initial_arr = initial.readonly();
    let callback_error = RefCell::<Option<PyErr>>::default();

    let objective_fn = |input: &ndarray::Array1<Complex64>| -> f64 {
        match call_scalar_function_complex(objective, input) {
            Ok(value) => value,
            Err(err) => {
                *callback_error.borrow_mut() = Some(err);
                f64::NAN
            }
        }
    };
    let gradient_fn = |input: &ndarray::Array1<Complex64>| -> ndarray::Array1<Complex64> {
        match call_vector_function_complex(gradient, input) {
            Ok(value) => value,
            Err(err) => {
                *callback_error.borrow_mut() = Some(err);
                ndarray::Array1::from_elem(input.len(), Complex64::new(f64::NAN, f64::NAN))
            }
        }
    };

    let result = map_callback_error(
        &callback_error,
        nabled_ml::optimization::adam_complex(
            &initial_arr.as_array(),
            objective_fn,
            gradient_fn,
            &config,
        ),
    )?;
    Ok(PyArray1::from_owned_array(py, result).unbind())
}

/// Minimize an objective with momentum gradient descent.
#[pyfunction]
#[pyo3(signature = (initial, objective, gradient, learning_rate=None, momentum=None, max_iterations=None, tolerance=None))]
pub fn momentum_descent<'py>(
    py: Python<'py>,
    initial: &Bound<'py, PyAny>,
    objective: &Bound<'py, PyAny>,
    gradient: &Bound<'py, PyAny>,
    learning_rate: Option<f64>,
    momentum: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<Py<PyAny>> {
    match utils::real_array1(initial, "initial")? {
        RealReadonlyArray1::F32(initial_arr) => {
            let config =
                momentum_config::<f32>(learning_rate, momentum, max_iterations, tolerance)?;
            let callback_error = RefCell::<Option<PyErr>>::default();

            let objective_fn = |input: &ndarray::Array1<f32>| -> f32 {
                match call_scalar_function_f32(objective, input) {
                    Ok(value) => value,
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        f32::NAN
                    }
                }
            };
            let gradient_fn = |input: &ndarray::Array1<f32>| -> ndarray::Array1<f32> {
                match call_vector_function_f32(gradient, input) {
                    Ok(value) => value,
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        ndarray::Array1::from_elem(input.len(), f32::NAN)
                    }
                }
            };

            let result = map_callback_error(
                &callback_error,
                nabled_ml::optimization::momentum_descent(
                    &initial_arr.as_array(),
                    objective_fn,
                    gradient_fn,
                    &config,
                ),
            )?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
        RealReadonlyArray1::F64(initial_arr) => {
            let config =
                momentum_config::<f64>(learning_rate, momentum, max_iterations, tolerance)?;
            let callback_error = RefCell::<Option<PyErr>>::default();

            let objective_fn = |input: &ndarray::Array1<f64>| -> f64 {
                match call_scalar_function_f64(objective, input) {
                    Ok(value) => value,
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        f64::NAN
                    }
                }
            };
            let gradient_fn = |input: &ndarray::Array1<f64>| -> ndarray::Array1<f64> {
                match call_vector_function_f64(gradient, input) {
                    Ok(value) => value,
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        ndarray::Array1::from_elem(input.len(), f64::NAN)
                    }
                }
            };

            let result = map_callback_error(
                &callback_error,
                nabled_ml::optimization::momentum_descent(
                    &initial_arr.as_array(),
                    objective_fn,
                    gradient_fn,
                    &config,
                ),
            )?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
    }
}

/// Minimize a complex objective with momentum gradient descent.
#[pyfunction]
#[pyo3(signature = (initial, objective, gradient, learning_rate=None, momentum=None, max_iterations=None, tolerance=None))]
pub fn momentum_descent_complex<'py>(
    py: Python<'py>,
    initial: &Bound<'py, PyArray1<Complex64>>,
    objective: &Bound<'py, PyAny>,
    gradient: &Bound<'py, PyAny>,
    learning_rate: Option<f64>,
    momentum: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<Py<PyArray1<Complex64>>> {
    utils::require_contiguous(initial)?;
    let config = momentum_config::<f64>(learning_rate, momentum, max_iterations, tolerance)?;
    let initial_arr = initial.readonly();
    let callback_error = RefCell::<Option<PyErr>>::default();

    let objective_fn = |input: &ndarray::Array1<Complex64>| -> f64 {
        match call_scalar_function_complex(objective, input) {
            Ok(value) => value,
            Err(err) => {
                *callback_error.borrow_mut() = Some(err);
                f64::NAN
            }
        }
    };
    let gradient_fn = |input: &ndarray::Array1<Complex64>| -> ndarray::Array1<Complex64> {
        match call_vector_function_complex(gradient, input) {
            Ok(value) => value,
            Err(err) => {
                *callback_error.borrow_mut() = Some(err);
                ndarray::Array1::from_elem(input.len(), Complex64::new(f64::NAN, f64::NAN))
            }
        }
    };

    let result = map_callback_error(
        &callback_error,
        nabled_ml::optimization::momentum_descent_complex(
            &initial_arr.as_array(),
            objective_fn,
            gradient_fn,
            &config,
        ),
    )?;
    Ok(PyArray1::from_owned_array(py, result).unbind())
}

/// Minimize an objective with RMSProp.
#[pyfunction]
#[pyo3(signature = (initial, objective, gradient, learning_rate=None, rho=None, epsilon=None, max_iterations=None, tolerance=None))]
pub fn rmsprop<'py>(
    py: Python<'py>,
    initial: &Bound<'py, PyAny>,
    objective: &Bound<'py, PyAny>,
    gradient: &Bound<'py, PyAny>,
    learning_rate: Option<f64>,
    rho: Option<f64>,
    epsilon: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<Py<PyAny>> {
    match utils::real_array1(initial, "initial")? {
        RealReadonlyArray1::F32(initial_arr) => {
            let config =
                rmsprop_config::<f32>(learning_rate, rho, epsilon, max_iterations, tolerance)?;
            let callback_error = RefCell::<Option<PyErr>>::default();

            let objective_fn = |input: &ndarray::Array1<f32>| -> f32 {
                match call_scalar_function_f32(objective, input) {
                    Ok(value) => value,
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        f32::NAN
                    }
                }
            };
            let gradient_fn = |input: &ndarray::Array1<f32>| -> ndarray::Array1<f32> {
                match call_vector_function_f32(gradient, input) {
                    Ok(value) => value,
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        ndarray::Array1::from_elem(input.len(), f32::NAN)
                    }
                }
            };

            let result = map_callback_error(
                &callback_error,
                nabled_ml::optimization::rmsprop(
                    &initial_arr.as_array(),
                    objective_fn,
                    gradient_fn,
                    &config,
                ),
            )?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
        RealReadonlyArray1::F64(initial_arr) => {
            let config =
                rmsprop_config::<f64>(learning_rate, rho, epsilon, max_iterations, tolerance)?;
            let callback_error = RefCell::<Option<PyErr>>::default();

            let objective_fn = |input: &ndarray::Array1<f64>| -> f64 {
                match call_scalar_function_f64(objective, input) {
                    Ok(value) => value,
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        f64::NAN
                    }
                }
            };
            let gradient_fn = |input: &ndarray::Array1<f64>| -> ndarray::Array1<f64> {
                match call_vector_function_f64(gradient, input) {
                    Ok(value) => value,
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        ndarray::Array1::from_elem(input.len(), f64::NAN)
                    }
                }
            };

            let result = map_callback_error(
                &callback_error,
                nabled_ml::optimization::rmsprop(
                    &initial_arr.as_array(),
                    objective_fn,
                    gradient_fn,
                    &config,
                ),
            )?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
    }
}

/// Minimize a complex objective with RMSProp.
#[pyfunction]
#[pyo3(signature = (initial, objective, gradient, learning_rate=None, rho=None, epsilon=None, max_iterations=None, tolerance=None))]
pub fn rmsprop_complex<'py>(
    py: Python<'py>,
    initial: &Bound<'py, PyArray1<Complex64>>,
    objective: &Bound<'py, PyAny>,
    gradient: &Bound<'py, PyAny>,
    learning_rate: Option<f64>,
    rho: Option<f64>,
    epsilon: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<Py<PyArray1<Complex64>>> {
    utils::require_contiguous(initial)?;
    let config = rmsprop_config::<f64>(learning_rate, rho, epsilon, max_iterations, tolerance)?;
    let initial_arr = initial.readonly();
    let callback_error = RefCell::<Option<PyErr>>::default();

    let objective_fn = |input: &ndarray::Array1<Complex64>| -> f64 {
        match call_scalar_function_complex(objective, input) {
            Ok(value) => value,
            Err(err) => {
                *callback_error.borrow_mut() = Some(err);
                f64::NAN
            }
        }
    };
    let gradient_fn = |input: &ndarray::Array1<Complex64>| -> ndarray::Array1<Complex64> {
        match call_vector_function_complex(gradient, input) {
            Ok(value) => value,
            Err(err) => {
                *callback_error.borrow_mut() = Some(err);
                ndarray::Array1::from_elem(input.len(), Complex64::new(f64::NAN, f64::NAN))
            }
        }
    };

    let result = map_callback_error(
        &callback_error,
        nabled_ml::optimization::rmsprop_complex(
            &initial_arr.as_array(),
            objective_fn,
            gradient_fn,
            &config,
        ),
    )?;
    Ok(PyArray1::from_owned_array(py, result).unbind())
}

/// Minimize an objective with projected gradient descent under box constraints.
#[pyfunction]
#[pyo3(signature = (initial, objective, gradient, lower_bounds, upper_bounds, learning_rate=None, max_iterations=None, tolerance=None))]
pub fn projected_gradient_descent_box<'py>(
    py: Python<'py>,
    initial: &Bound<'py, PyAny>,
    objective: &Bound<'py, PyAny>,
    gradient: &Bound<'py, PyAny>,
    lower_bounds: &Bound<'py, PyAny>,
    upper_bounds: &Bound<'py, PyAny>,
    learning_rate: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<Py<PyAny>> {
    match (
        utils::real_array1(initial, "initial")?,
        utils::real_array1(lower_bounds, "lower_bounds")?,
        utils::real_array1(upper_bounds, "upper_bounds")?,
    ) {
        (
            RealReadonlyArray1::F32(initial_arr),
            RealReadonlyArray1::F32(lower_arr),
            RealReadonlyArray1::F32(upper_arr),
        ) => {
            let config =
                projected_gradient_config::<f32>(learning_rate, max_iterations, tolerance)?;
            let callback_error = RefCell::<Option<PyErr>>::default();

            let objective_fn = |input: &ndarray::Array1<f32>| -> f32 {
                match call_scalar_function_f32(objective, input) {
                    Ok(value) => value,
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        f32::NAN
                    }
                }
            };
            let gradient_fn = |input: &ndarray::Array1<f32>| -> ndarray::Array1<f32> {
                match call_vector_function_f32(gradient, input) {
                    Ok(value) => value,
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        ndarray::Array1::from_elem(input.len(), f32::NAN)
                    }
                }
            };

            let result = map_callback_error(
                &callback_error,
                nabled_ml::optimization::projected_gradient_descent_box(
                    &initial_arr.as_array(),
                    objective_fn,
                    gradient_fn,
                    &lower_arr.as_array(),
                    &upper_arr.as_array(),
                    &config,
                ),
            )?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
        (
            RealReadonlyArray1::F64(initial_arr),
            RealReadonlyArray1::F64(lower_arr),
            RealReadonlyArray1::F64(upper_arr),
        ) => {
            let config =
                projected_gradient_config::<f64>(learning_rate, max_iterations, tolerance)?;
            let callback_error = RefCell::<Option<PyErr>>::default();

            let objective_fn = |input: &ndarray::Array1<f64>| -> f64 {
                match call_scalar_function_f64(objective, input) {
                    Ok(value) => value,
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        f64::NAN
                    }
                }
            };
            let gradient_fn = |input: &ndarray::Array1<f64>| -> ndarray::Array1<f64> {
                match call_vector_function_f64(gradient, input) {
                    Ok(value) => value,
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        ndarray::Array1::from_elem(input.len(), f64::NAN)
                    }
                }
            };

            let result = map_callback_error(
                &callback_error,
                nabled_ml::optimization::projected_gradient_descent_box(
                    &initial_arr.as_array(),
                    objective_fn,
                    gradient_fn,
                    &lower_arr.as_array(),
                    &upper_arr.as_array(),
                    &config,
                ),
            )?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
        _ => Err(utils::matching_real_dtype_error(&["initial", "lower_bounds", "upper_bounds"])),
    }
}

/// Minimize a complex objective with projected gradient descent under box constraints.
#[pyfunction]
#[pyo3(signature = (initial, objective, gradient, lower_bounds, upper_bounds, learning_rate=None, max_iterations=None, tolerance=None))]
pub fn projected_gradient_descent_box_complex<'py>(
    py: Python<'py>,
    initial: &Bound<'py, PyArray1<Complex64>>,
    objective: &Bound<'py, PyAny>,
    gradient: &Bound<'py, PyAny>,
    lower_bounds: &Bound<'py, PyArray1<Complex64>>,
    upper_bounds: &Bound<'py, PyArray1<Complex64>>,
    learning_rate: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<Py<PyArray1<Complex64>>> {
    utils::require_contiguous(initial)?;
    utils::require_contiguous(lower_bounds)?;
    utils::require_contiguous(upper_bounds)?;
    let config = projected_gradient_config::<f64>(learning_rate, max_iterations, tolerance)?;
    let initial_arr = initial.readonly();
    let lower_arr = lower_bounds.readonly();
    let upper_arr = upper_bounds.readonly();
    let callback_error = RefCell::<Option<PyErr>>::default();

    let objective_fn = |input: &ndarray::Array1<Complex64>| -> f64 {
        match call_scalar_function_complex(objective, input) {
            Ok(value) => value,
            Err(err) => {
                *callback_error.borrow_mut() = Some(err);
                f64::NAN
            }
        }
    };
    let gradient_fn = |input: &ndarray::Array1<Complex64>| -> ndarray::Array1<Complex64> {
        match call_vector_function_complex(gradient, input) {
            Ok(value) => value,
            Err(err) => {
                *callback_error.borrow_mut() = Some(err);
                ndarray::Array1::from_elem(input.len(), Complex64::new(f64::NAN, f64::NAN))
            }
        }
    };

    let result = map_callback_error(
        &callback_error,
        nabled_ml::optimization::projected_gradient_descent_box_complex(
            &initial_arr.as_array(),
            objective_fn,
            gradient_fn,
            &lower_arr.as_array(),
            &upper_arr.as_array(),
            &config,
        ),
    )?;
    Ok(PyArray1::from_owned_array(py, result).unbind())
}

/// Minimize an objective with stochastic gradient descent.
#[pyfunction]
#[pyo3(signature = (initial, stochastic_gradient, learning_rate=None, max_iterations=None, tolerance=None))]
pub fn stochastic_gradient_descent<'py>(
    py: Python<'py>,
    initial: &Bound<'py, PyAny>,
    stochastic_gradient: &Bound<'py, PyAny>,
    learning_rate: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<Py<PyAny>> {
    match utils::real_array1(initial, "initial")? {
        RealReadonlyArray1::F32(initial_arr) => {
            let config = sgd_config::<f32>(learning_rate, max_iterations, tolerance)?;
            let callback_error = RefCell::<Option<PyErr>>::default();

            let gradient_fn = |input: &ndarray::Array1<f32>,
                               iteration: usize|
             -> ndarray::Array1<f32> {
                match call_vector_function_f32_with_iteration(stochastic_gradient, input, iteration)
                {
                    Ok(value) => value,
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        ndarray::Array1::from_elem(input.len(), f32::NAN)
                    }
                }
            };

            let result = map_callback_error(
                &callback_error,
                nabled_ml::optimization::stochastic_gradient_descent(
                    &initial_arr.as_array(),
                    gradient_fn,
                    &config,
                ),
            )?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
        RealReadonlyArray1::F64(initial_arr) => {
            let config = sgd_config::<f64>(learning_rate, max_iterations, tolerance)?;
            let callback_error = RefCell::<Option<PyErr>>::default();

            let gradient_fn = |input: &ndarray::Array1<f64>,
                               iteration: usize|
             -> ndarray::Array1<f64> {
                match call_vector_function_f64_with_iteration(stochastic_gradient, input, iteration)
                {
                    Ok(value) => value,
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        ndarray::Array1::from_elem(input.len(), f64::NAN)
                    }
                }
            };

            let result = map_callback_error(
                &callback_error,
                nabled_ml::optimization::stochastic_gradient_descent(
                    &initial_arr.as_array(),
                    gradient_fn,
                    &config,
                ),
            )?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
    }
}

/// Minimize a complex objective with stochastic gradient descent.
#[pyfunction]
#[pyo3(signature = (initial, stochastic_gradient, learning_rate=None, max_iterations=None, tolerance=None))]
pub fn stochastic_gradient_descent_complex<'py>(
    py: Python<'py>,
    initial: &Bound<'py, PyArray1<Complex64>>,
    stochastic_gradient: &Bound<'py, PyAny>,
    learning_rate: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<Py<PyArray1<Complex64>>> {
    utils::require_contiguous(initial)?;
    let config = sgd_config::<f64>(learning_rate, max_iterations, tolerance)?;
    let initial_arr = initial.readonly();
    let callback_error = RefCell::<Option<PyErr>>::default();

    let gradient_fn = |input: &ndarray::Array1<Complex64>,
                       iteration: usize|
     -> ndarray::Array1<Complex64> {
        match call_vector_function_complex_with_iteration(stochastic_gradient, input, iteration) {
            Ok(value) => value,
            Err(err) => {
                *callback_error.borrow_mut() = Some(err);
                ndarray::Array1::from_elem(input.len(), Complex64::new(f64::NAN, f64::NAN))
            }
        }
    };

    let result = map_callback_error(
        &callback_error,
        nabled_ml::optimization::stochastic_gradient_descent_complex(
            &initial_arr.as_array(),
            gradient_fn,
            &config,
        ),
    )?;
    Ok(PyArray1::from_owned_array(py, result).unbind())
}

/// Minimize an objective with BFGS.
#[pyfunction]
#[pyo3(signature = (initial, objective, gradient, step_size=None, max_iterations=None, tolerance=None, curvature_tolerance=None))]
pub fn bfgs<'py>(
    py: Python<'py>,
    initial: &Bound<'py, PyAny>,
    objective: &Bound<'py, PyAny>,
    gradient: &Bound<'py, PyAny>,
    step_size: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
    curvature_tolerance: Option<f64>,
) -> PyResult<Py<PyAny>> {
    match utils::real_array1(initial, "initial")? {
        RealReadonlyArray1::F32(initial_arr) => {
            let config =
                bfgs_config::<f32>(step_size, max_iterations, tolerance, curvature_tolerance)?;
            let callback_error = RefCell::<Option<PyErr>>::default();

            let objective_fn = |input: &ndarray::Array1<f32>| -> f32 {
                match call_scalar_function_f32(objective, input) {
                    Ok(value) => value,
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        f32::NAN
                    }
                }
            };
            let gradient_fn = |input: &ndarray::Array1<f32>| -> ndarray::Array1<f32> {
                match call_vector_function_f32(gradient, input) {
                    Ok(value) => value,
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        ndarray::Array1::from_elem(input.len(), f32::NAN)
                    }
                }
            };

            let result = map_callback_error(
                &callback_error,
                nabled_ml::optimization::bfgs(
                    &initial_arr.as_array(),
                    objective_fn,
                    gradient_fn,
                    &config,
                ),
            )?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
        RealReadonlyArray1::F64(initial_arr) => {
            let config =
                bfgs_config::<f64>(step_size, max_iterations, tolerance, curvature_tolerance)?;
            let callback_error = RefCell::<Option<PyErr>>::default();

            let objective_fn = |input: &ndarray::Array1<f64>| -> f64 {
                match call_scalar_function_f64(objective, input) {
                    Ok(value) => value,
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        f64::NAN
                    }
                }
            };
            let gradient_fn = |input: &ndarray::Array1<f64>| -> ndarray::Array1<f64> {
                match call_vector_function_f64(gradient, input) {
                    Ok(value) => value,
                    Err(err) => {
                        *callback_error.borrow_mut() = Some(err);
                        ndarray::Array1::from_elem(input.len(), f64::NAN)
                    }
                }
            };

            let result = map_callback_error(
                &callback_error,
                nabled_ml::optimization::bfgs(
                    &initial_arr.as_array(),
                    objective_fn,
                    gradient_fn,
                    &config,
                ),
            )?;
            Ok(utils::pyarray1_from_owned(py, result))
        }
    }
}

/// Minimize a complex objective with BFGS.
#[pyfunction]
#[pyo3(signature = (initial, objective, gradient, step_size=None, max_iterations=None, tolerance=None, curvature_tolerance=None))]
pub fn bfgs_complex<'py>(
    py: Python<'py>,
    initial: &Bound<'py, PyArray1<Complex64>>,
    objective: &Bound<'py, PyAny>,
    gradient: &Bound<'py, PyAny>,
    step_size: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
    curvature_tolerance: Option<f64>,
) -> PyResult<Py<PyArray1<Complex64>>> {
    utils::require_contiguous(initial)?;
    let config = bfgs_config::<f64>(step_size, max_iterations, tolerance, curvature_tolerance)?;
    let initial_arr = initial.readonly();
    let callback_error = RefCell::<Option<PyErr>>::default();

    let objective_fn = |input: &ndarray::Array1<Complex64>| -> f64 {
        match call_scalar_function_complex(objective, input) {
            Ok(value) => value,
            Err(err) => {
                *callback_error.borrow_mut() = Some(err);
                f64::NAN
            }
        }
    };
    let gradient_fn = |input: &ndarray::Array1<Complex64>| -> ndarray::Array1<Complex64> {
        match call_vector_function_complex(gradient, input) {
            Ok(value) => value,
            Err(err) => {
                *callback_error.borrow_mut() = Some(err);
                ndarray::Array1::from_elem(input.len(), Complex64::new(f64::NAN, f64::NAN))
            }
        }
    };

    let result = map_callback_error(
        &callback_error,
        nabled_ml::optimization::bfgs_complex(
            &initial_arr.as_array(),
            objective_fn,
            gradient_fn,
            &config,
        ),
    )?;
    Ok(PyArray1::from_owned_array(py, result).unbind())
}
