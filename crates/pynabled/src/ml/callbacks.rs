//! Helpers for calling Python callbacks from ML bindings.

use ndarray::Array1;
use num_complex::Complex64;
use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;

use crate::utils;

pub(crate) fn call_scalar_function_f32(
    function: &Bound<'_, PyAny>,
    x: &Array1<f32>,
) -> PyResult<f32> {
    let py = function.py();
    let arg = PyArray1::from_owned_array(py, x.to_owned());
    function.call1((arg,))?.extract::<f32>()
}

pub(crate) fn call_vector_function_f32(
    function: &Bound<'_, PyAny>,
    x: &Array1<f32>,
) -> PyResult<Array1<f32>> {
    let py = function.py();
    let arg = PyArray1::from_owned_array(py, x.to_owned());
    let result = function.call1((arg,))?;
    let array = result.cast::<PyArray1<f32>>()?;
    utils::require_contiguous(array)?;
    Ok(array.readonly().as_array().to_owned())
}

pub(crate) fn call_vector_function_f32_with_iteration(
    function: &Bound<'_, PyAny>,
    x: &Array1<f32>,
    iteration: usize,
) -> PyResult<Array1<f32>> {
    let py = function.py();
    let arg = PyArray1::from_owned_array(py, x.to_owned());
    let result = function.call1((arg, iteration))?;
    let array = result.cast::<PyArray1<f32>>()?;
    utils::require_contiguous(array)?;
    Ok(array.readonly().as_array().to_owned())
}

pub(crate) fn call_scalar_function_f64(
    function: &Bound<'_, PyAny>,
    x: &Array1<f64>,
) -> PyResult<f64> {
    let py = function.py();
    let arg = PyArray1::from_owned_array(py, x.to_owned());
    function.call1((arg,))?.extract::<f64>()
}

pub(crate) fn call_vector_function_f64(
    function: &Bound<'_, PyAny>,
    x: &Array1<f64>,
) -> PyResult<Array1<f64>> {
    let py = function.py();
    let arg = PyArray1::from_owned_array(py, x.to_owned());
    let result = function.call1((arg,))?;
    let array = result.cast::<PyArray1<f64>>()?;
    utils::require_contiguous(array)?;
    Ok(array.readonly().as_array().to_owned())
}

pub(crate) fn call_vector_function_f64_with_iteration(
    function: &Bound<'_, PyAny>,
    x: &Array1<f64>,
    iteration: usize,
) -> PyResult<Array1<f64>> {
    let py = function.py();
    let arg = PyArray1::from_owned_array(py, x.to_owned());
    let result = function.call1((arg, iteration))?;
    let array = result.cast::<PyArray1<f64>>()?;
    utils::require_contiguous(array)?;
    Ok(array.readonly().as_array().to_owned())
}

pub(crate) fn call_scalar_function_complex(
    function: &Bound<'_, PyAny>,
    x: &Array1<Complex64>,
) -> PyResult<f64> {
    let py = function.py();
    let arg = PyArray1::from_owned_array(py, x.to_owned());
    function.call1((arg,))?.extract::<f64>()
}

pub(crate) fn call_vector_function_complex(
    function: &Bound<'_, PyAny>,
    x: &Array1<Complex64>,
) -> PyResult<Array1<Complex64>> {
    let py = function.py();
    let arg = PyArray1::from_owned_array(py, x.to_owned());
    let result = function.call1((arg,))?;
    let array = result.cast::<PyArray1<Complex64>>()?;
    utils::require_contiguous(array)?;
    Ok(array.readonly().as_array().to_owned())
}

pub(crate) fn call_vector_function_complex_with_iteration(
    function: &Bound<'_, PyAny>,
    x: &Array1<Complex64>,
    iteration: usize,
) -> PyResult<Array1<Complex64>> {
    let py = function.py();
    let arg = PyArray1::from_owned_array(py, x.to_owned());
    let result = function.call1((arg, iteration))?;
    let array = result.cast::<PyArray1<Complex64>>()?;
    utils::require_contiguous(array)?;
    Ok(array.readonly().as_array().to_owned())
}
