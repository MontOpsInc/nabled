//! Helpers for calling Python callbacks from ML bindings.

#[cfg(feature = "arrow")]
use arrow_array::types::{Float32Type, Float64Type};
#[cfg(feature = "arrow")]
use arrow_array::{Array, FixedSizeListArray, PrimitiveArray, make_array};
#[cfg(feature = "arrow")]
use arrow_data::ArrayData;
#[cfg(feature = "arrow")]
use arrow_pyarrow::PyArrowType;
#[cfg(feature = "arrow")]
use arrow_schema::Field;
#[cfg(feature = "arrow")]
use nabled::ndarrow::{self, Complex64Extension};
use ndarray::Array1;
use num_complex::Complex64;
use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
#[cfg(feature = "arrow")]
use pyo3::types::PyType;
#[cfg(feature = "arrow")]
use pyo3::{IntoPyObjectExt, exceptions::PyTypeError};

#[cfg(feature = "arrow")]
use crate::error::to_py_err;
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

#[cfg(feature = "arrow")]
fn pyarrow_array_type(py: Python<'_>) -> PyResult<Bound<'_, PyType>> {
    Ok(py.import("pyarrow")?.getattr("Array")?.cast_into()?)
}

#[cfg(feature = "arrow")]
fn pyarrow_extension_array_type(py: Python<'_>) -> PyResult<Bound<'_, PyType>> {
    Ok(py.import("pyarrow")?.getattr("ExtensionArray")?.cast_into()?)
}

#[cfg(feature = "arrow")]
fn primitive_array_arg_f32<'py>(py: Python<'py>, x: &Array1<f32>) -> PyResult<Bound<'py, PyAny>> {
    PyArrowType(PrimitiveArray::<Float32Type>::from_iter_values(x.iter().copied()).into_data())
        .into_bound_py_any(py)
}

#[cfg(feature = "arrow")]
fn primitive_array_arg_f64<'py>(py: Python<'py>, x: &Array1<f64>) -> PyResult<Bound<'py, PyAny>> {
    PyArrowType(PrimitiveArray::<Float64Type>::from_iter_values(x.iter().copied()).into_data())
        .into_bound_py_any(py)
}

#[cfg(feature = "arrow")]
fn complex_extension_array_arg<'py>(
    py: Python<'py>,
    x: &Array1<Complex64>,
) -> PyResult<Bound<'py, PyAny>> {
    let (field, storage): (Field, FixedSizeListArray) =
        ndarrow::array1_complex64_to_extension("callback", x.to_owned()).map_err(to_py_err)?;
    let field = PyArrowType(field).into_bound_py_any(py)?;
    let storage = PyArrowType(storage.into_data()).into_bound_py_any(py)?;
    pyarrow_extension_array_type(py)?
        .call_method1("from_storage", (field.getattr("type")?, storage))
}

#[cfg(feature = "arrow")]
fn real_callback_result_type_error(dtype: &str) -> PyErr {
    PyTypeError::new_err(format!("callback must return a PyArrow {dtype} array"))
}

#[cfg(feature = "arrow")]
fn primitive_result_array_f32(result: &Bound<'_, PyAny>) -> PyResult<PrimitiveArray<Float32Type>> {
    let array_type = pyarrow_array_type(result.py())?;
    if !result.is_instance(&array_type)? {
        return Err(real_callback_result_type_error("float32"));
    }
    let data = result.extract::<PyArrowType<ArrayData>>()?;
    let array = make_array(data.0);
    array
        .as_any()
        .downcast_ref::<PrimitiveArray<Float32Type>>()
        .cloned()
        .ok_or_else(|| real_callback_result_type_error("float32"))
}

#[cfg(feature = "arrow")]
fn primitive_result_array_f64(result: &Bound<'_, PyAny>) -> PyResult<PrimitiveArray<Float64Type>> {
    let array_type = pyarrow_array_type(result.py())?;
    if !result.is_instance(&array_type)? {
        return Err(real_callback_result_type_error("float64"));
    }
    let data = result.extract::<PyArrowType<ArrayData>>()?;
    let array = make_array(data.0);
    array
        .as_any()
        .downcast_ref::<PrimitiveArray<Float64Type>>()
        .cloned()
        .ok_or_else(|| real_callback_result_type_error("float64"))
}

#[cfg(feature = "arrow")]
fn primitive_array_to_owned_f32(array: PrimitiveArray<Float32Type>) -> PyResult<Array1<f32>> {
    if array.null_count() > 0 {
        return Err(PyTypeError::new_err("callback result must not contain nulls"));
    }
    Ok(Array1::from_iter(
        array
            .iter()
            .map(|value| value.expect("null-free primitive array iteration should not yield None")),
    ))
}

#[cfg(feature = "arrow")]
fn primitive_array_to_owned_f64(array: PrimitiveArray<Float64Type>) -> PyResult<Array1<f64>> {
    if array.null_count() > 0 {
        return Err(PyTypeError::new_err("callback result must not contain nulls"));
    }
    Ok(Array1::from_iter(
        array
            .iter()
            .map(|value| value.expect("null-free primitive array iteration should not yield None")),
    ))
}

#[cfg(feature = "arrow")]
fn complex_callback_field(array: &FixedSizeListArray) -> PyResult<Field> {
    let mut field = Field::new("callback", array.data_type().clone(), false);
    field.try_with_extension_type(Complex64Extension).map_err(to_py_err)?;
    Ok(field)
}

#[cfg(feature = "arrow")]
fn complex_result_array(result: &Bound<'_, PyAny>) -> PyResult<FixedSizeListArray> {
    let data = result.extract::<PyArrowType<ArrayData>>()?;
    let array = make_array(data.0);
    array.as_any().downcast_ref::<FixedSizeListArray>().cloned().ok_or_else(|| {
        PyTypeError::new_err("callback must return a canonical ndarrow.complex64 PyArrow array")
    })
}

#[cfg(feature = "arrow")]
fn complex_array_to_owned(array: FixedSizeListArray) -> PyResult<Array1<Complex64>> {
    let field = complex_callback_field(&array)?;
    Ok(ndarrow::complex64_as_array_view1(&field, &array).map_err(to_py_err)?.to_owned())
}

#[cfg(feature = "arrow")]
pub(crate) fn call_scalar_function_arrow_f32(
    function: &Bound<'_, PyAny>,
    x: &Array1<f32>,
) -> PyResult<f32> {
    let arg = primitive_array_arg_f32(function.py(), x)?;
    function.call1((arg,))?.extract::<f32>()
}

#[cfg(feature = "arrow")]
pub(crate) fn call_vector_function_arrow_f32(
    function: &Bound<'_, PyAny>,
    x: &Array1<f32>,
) -> PyResult<Array1<f32>> {
    let arg = primitive_array_arg_f32(function.py(), x)?;
    primitive_array_to_owned_f32(primitive_result_array_f32(&function.call1((arg,))?)?)
}

#[cfg(feature = "arrow")]
pub(crate) fn call_vector_function_arrow_f32_with_iteration(
    function: &Bound<'_, PyAny>,
    x: &Array1<f32>,
    iteration: usize,
) -> PyResult<Array1<f32>> {
    let arg = primitive_array_arg_f32(function.py(), x)?;
    primitive_array_to_owned_f32(primitive_result_array_f32(&function.call1((arg, iteration))?)?)
}

#[cfg(feature = "arrow")]
pub(crate) fn call_scalar_function_arrow_f64(
    function: &Bound<'_, PyAny>,
    x: &Array1<f64>,
) -> PyResult<f64> {
    let arg = primitive_array_arg_f64(function.py(), x)?;
    function.call1((arg,))?.extract::<f64>()
}

#[cfg(feature = "arrow")]
pub(crate) fn call_vector_function_arrow_f64(
    function: &Bound<'_, PyAny>,
    x: &Array1<f64>,
) -> PyResult<Array1<f64>> {
    let arg = primitive_array_arg_f64(function.py(), x)?;
    primitive_array_to_owned_f64(primitive_result_array_f64(&function.call1((arg,))?)?)
}

#[cfg(feature = "arrow")]
pub(crate) fn call_vector_function_arrow_f64_with_iteration(
    function: &Bound<'_, PyAny>,
    x: &Array1<f64>,
    iteration: usize,
) -> PyResult<Array1<f64>> {
    let arg = primitive_array_arg_f64(function.py(), x)?;
    primitive_array_to_owned_f64(primitive_result_array_f64(&function.call1((arg, iteration))?)?)
}

#[cfg(feature = "arrow")]
pub(crate) fn call_scalar_function_arrow_complex(
    function: &Bound<'_, PyAny>,
    x: &Array1<Complex64>,
) -> PyResult<f64> {
    let arg = complex_extension_array_arg(function.py(), x)?;
    function.call1((arg,))?.extract::<f64>()
}

#[cfg(feature = "arrow")]
pub(crate) fn call_vector_function_arrow_complex(
    function: &Bound<'_, PyAny>,
    x: &Array1<Complex64>,
) -> PyResult<Array1<Complex64>> {
    let arg = complex_extension_array_arg(function.py(), x)?;
    complex_array_to_owned(complex_result_array(&function.call1((arg,))?)?)
}

#[cfg(feature = "arrow")]
pub(crate) fn call_vector_function_arrow_complex_with_iteration(
    function: &Bound<'_, PyAny>,
    x: &Array1<Complex64>,
    iteration: usize,
) -> PyResult<Array1<Complex64>> {
    let arg = complex_extension_array_arg(function.py(), x)?;
    complex_array_to_owned(complex_result_array(&function.call1((arg, iteration))?)?)
}
