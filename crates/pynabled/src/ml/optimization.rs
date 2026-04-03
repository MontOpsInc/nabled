//! Optimization bindings for Python callables.

use std::cell::RefCell;

use nabled_ml::optimization::{
    AdamConfig, BFGSConfig, LineSearchConfig, MomentumConfig, OptimizationError,
    ProjectedGradientConfig, RMSPropConfig, SGDConfig,
};
use num_complex::Complex64;
use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;

use crate::error::to_py_err;
use crate::ml::callbacks::{
    call_scalar_function_complex, call_scalar_function_f64, call_vector_function_complex,
    call_vector_function_complex_with_iteration, call_vector_function_f64,
    call_vector_function_f64_with_iteration,
};
use crate::utils;

fn line_search_config(
    initial_step: Option<f64>,
    contraction: Option<f64>,
    sufficient_decrease: Option<f64>,
    max_iterations: Option<usize>,
) -> LineSearchConfig<f64> {
    let mut config = LineSearchConfig::<f64>::default();
    if let Some(value) = initial_step {
        config.initial_step = value;
    }
    if let Some(value) = contraction {
        config.contraction = value;
    }
    if let Some(value) = sufficient_decrease {
        config.sufficient_decrease = value;
    }
    if let Some(value) = max_iterations {
        config.max_iterations = value;
    }
    config
}

fn sgd_config(
    learning_rate: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> SGDConfig<f64> {
    let mut config = SGDConfig::<f64>::default();
    if let Some(value) = learning_rate {
        config.learning_rate = value;
    }
    if let Some(value) = max_iterations {
        config.max_iterations = value;
    }
    if let Some(value) = tolerance {
        config.tolerance = value;
    }
    config
}

fn adam_config(
    learning_rate: Option<f64>,
    beta1: Option<f64>,
    beta2: Option<f64>,
    epsilon: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> AdamConfig<f64> {
    let mut config = AdamConfig::<f64>::default();
    if let Some(value) = learning_rate {
        config.learning_rate = value;
    }
    if let Some(value) = beta1 {
        config.beta1 = value;
    }
    if let Some(value) = beta2 {
        config.beta2 = value;
    }
    if let Some(value) = epsilon {
        config.epsilon = value;
    }
    if let Some(value) = max_iterations {
        config.max_iterations = value;
    }
    if let Some(value) = tolerance {
        config.tolerance = value;
    }
    config
}

fn momentum_config(
    learning_rate: Option<f64>,
    momentum: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> MomentumConfig<f64> {
    let mut config = MomentumConfig::<f64>::default();
    if let Some(value) = learning_rate {
        config.learning_rate = value;
    }
    if let Some(value) = momentum {
        config.momentum = value;
    }
    if let Some(value) = max_iterations {
        config.max_iterations = value;
    }
    if let Some(value) = tolerance {
        config.tolerance = value;
    }
    config
}

fn rmsprop_config(
    learning_rate: Option<f64>,
    rho: Option<f64>,
    epsilon: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> RMSPropConfig<f64> {
    let mut config = RMSPropConfig::<f64>::default();
    if let Some(value) = learning_rate {
        config.learning_rate = value;
    }
    if let Some(value) = rho {
        config.rho = value;
    }
    if let Some(value) = epsilon {
        config.epsilon = value;
    }
    if let Some(value) = max_iterations {
        config.max_iterations = value;
    }
    if let Some(value) = tolerance {
        config.tolerance = value;
    }
    config
}

fn projected_gradient_config(
    learning_rate: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> ProjectedGradientConfig<f64> {
    let mut config = ProjectedGradientConfig::<f64>::default();
    if let Some(value) = learning_rate {
        config.learning_rate = value;
    }
    if let Some(value) = max_iterations {
        config.max_iterations = value;
    }
    if let Some(value) = tolerance {
        config.tolerance = value;
    }
    config
}

fn bfgs_config(
    step_size: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
    curvature_tolerance: Option<f64>,
) -> BFGSConfig<f64> {
    let mut config = BFGSConfig::<f64>::default();
    if let Some(value) = step_size {
        config.step_size = value;
    }
    if let Some(value) = max_iterations {
        config.max_iterations = value;
    }
    if let Some(value) = tolerance {
        config.tolerance = value;
    }
    if let Some(value) = curvature_tolerance {
        config.curvature_tolerance = value;
    }
    config
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
    point: &Bound<'_, PyArray1<f64>>,
    direction: &Bound<'_, PyArray1<f64>>,
    objective: &Bound<'_, PyAny>,
    gradient: &Bound<'_, PyAny>,
    initial_step: Option<f64>,
    contraction: Option<f64>,
    sufficient_decrease: Option<f64>,
    max_iterations: Option<usize>,
) -> PyResult<f64> {
    utils::require_contiguous(point)?;
    utils::require_contiguous(direction)?;
    let config = line_search_config(initial_step, contraction, sufficient_decrease, max_iterations);
    let point_arr = point.readonly();
    let direction_arr = direction.readonly();
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
    let config = line_search_config(initial_step, contraction, sufficient_decrease, max_iterations);
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
    initial: &Bound<'py, PyArray1<f64>>,
    objective: &Bound<'py, PyAny>,
    gradient: &Bound<'py, PyAny>,
    learning_rate: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    utils::require_contiguous(initial)?;
    let config = sgd_config(learning_rate, max_iterations, tolerance);
    let initial_arr = initial.readonly();
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
    Ok(PyArray1::from_owned_array(py, result).unbind())
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
    let config = sgd_config(learning_rate, max_iterations, tolerance);
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
    initial: &Bound<'py, PyArray1<f64>>,
    objective: &Bound<'py, PyAny>,
    gradient: &Bound<'py, PyAny>,
    learning_rate: Option<f64>,
    beta1: Option<f64>,
    beta2: Option<f64>,
    epsilon: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    utils::require_contiguous(initial)?;
    let config = adam_config(learning_rate, beta1, beta2, epsilon, max_iterations, tolerance);
    let initial_arr = initial.readonly();
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
        nabled_ml::optimization::adam(&initial_arr.as_array(), objective_fn, gradient_fn, &config),
    )?;
    Ok(PyArray1::from_owned_array(py, result).unbind())
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
    let config = adam_config(learning_rate, beta1, beta2, epsilon, max_iterations, tolerance);
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
    initial: &Bound<'py, PyArray1<f64>>,
    objective: &Bound<'py, PyAny>,
    gradient: &Bound<'py, PyAny>,
    learning_rate: Option<f64>,
    momentum: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    utils::require_contiguous(initial)?;
    let config = momentum_config(learning_rate, momentum, max_iterations, tolerance);
    let initial_arr = initial.readonly();
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
    Ok(PyArray1::from_owned_array(py, result).unbind())
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
    let config = momentum_config(learning_rate, momentum, max_iterations, tolerance);
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
    initial: &Bound<'py, PyArray1<f64>>,
    objective: &Bound<'py, PyAny>,
    gradient: &Bound<'py, PyAny>,
    learning_rate: Option<f64>,
    rho: Option<f64>,
    epsilon: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    utils::require_contiguous(initial)?;
    let config = rmsprop_config(learning_rate, rho, epsilon, max_iterations, tolerance);
    let initial_arr = initial.readonly();
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
    Ok(PyArray1::from_owned_array(py, result).unbind())
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
    let config = rmsprop_config(learning_rate, rho, epsilon, max_iterations, tolerance);
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
    initial: &Bound<'py, PyArray1<f64>>,
    objective: &Bound<'py, PyAny>,
    gradient: &Bound<'py, PyAny>,
    lower_bounds: &Bound<'py, PyArray1<f64>>,
    upper_bounds: &Bound<'py, PyArray1<f64>>,
    learning_rate: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    utils::require_contiguous(initial)?;
    utils::require_contiguous(lower_bounds)?;
    utils::require_contiguous(upper_bounds)?;
    let config = projected_gradient_config(learning_rate, max_iterations, tolerance);
    let initial_arr = initial.readonly();
    let lower_arr = lower_bounds.readonly();
    let upper_arr = upper_bounds.readonly();
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
    Ok(PyArray1::from_owned_array(py, result).unbind())
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
    let config = projected_gradient_config(learning_rate, max_iterations, tolerance);
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
    initial: &Bound<'py, PyArray1<f64>>,
    stochastic_gradient: &Bound<'py, PyAny>,
    learning_rate: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    utils::require_contiguous(initial)?;
    let config = sgd_config(learning_rate, max_iterations, tolerance);
    let initial_arr = initial.readonly();
    let callback_error = RefCell::<Option<PyErr>>::default();

    let gradient_fn = |input: &ndarray::Array1<f64>, iteration: usize| -> ndarray::Array1<f64> {
        match call_vector_function_f64_with_iteration(stochastic_gradient, input, iteration) {
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
    Ok(PyArray1::from_owned_array(py, result).unbind())
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
    let config = sgd_config(learning_rate, max_iterations, tolerance);
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
    initial: &Bound<'py, PyArray1<f64>>,
    objective: &Bound<'py, PyAny>,
    gradient: &Bound<'py, PyAny>,
    step_size: Option<f64>,
    max_iterations: Option<usize>,
    tolerance: Option<f64>,
    curvature_tolerance: Option<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    utils::require_contiguous(initial)?;
    let config = bfgs_config(step_size, max_iterations, tolerance, curvature_tolerance);
    let initial_arr = initial.readonly();
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
        nabled_ml::optimization::bfgs(&initial_arr.as_array(), objective_fn, gradient_fn, &config),
    )?;
    Ok(PyArray1::from_owned_array(py, result).unbind())
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
    let config = bfgs_config(step_size, max_iterations, tolerance, curvature_tolerance);
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
