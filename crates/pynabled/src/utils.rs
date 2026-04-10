//! Utilities for Python bindings.

use ndarray::{Array1, Array2, Array3, ArrayD};
use num_complex::Complex64;
use num_traits::{FromPrimitive, ToPrimitive};
use numpy::{
    Element, PyArray1, PyArray2, PyArray3, PyArrayDyn, PyArrayMethods, PyReadonlyArray1,
    PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArrayDyn, PyReadwriteArray1, PyReadwriteArray2,
    PyReadwriteArray3, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyOverflowError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyComplex};

/// Validate NumPy layout compatibility for borrowed-array ingress.
///
/// `pynabled` no longer rejects non-C-contiguous dense NumPy arrays at the Python boundary.
/// Dense kernels should borrow strided views when the Rust API admits them, and wrappers that
/// still materialize owned arrays must do so because of API shape rather than a blanket layout
/// restriction at ingress.
pub fn require_contiguous<'py, A: PyUntypedArrayMethods<'py>>(_array: &A) -> PyResult<()> { Ok(()) }

pub enum RealReadonlyArray1<'py> {
    F32(PyReadonlyArray1<'py, f32>),
    F64(PyReadonlyArray1<'py, f64>),
}

pub enum RealReadonlyArray2<'py> {
    F32(PyReadonlyArray2<'py, f32>),
    F64(PyReadonlyArray2<'py, f64>),
}

pub enum RealReadonlyArray3<'py> {
    F32(PyReadonlyArray3<'py, f32>),
    F64(PyReadonlyArray3<'py, f64>),
}

pub enum RealReadonlyArrayDyn<'py> {
    F32(PyReadonlyArrayDyn<'py, f32>),
    F64(PyReadonlyArrayDyn<'py, f64>),
}

pub enum NumericReadonlyArray1<'py> {
    F32(PyReadonlyArray1<'py, f32>),
    F64(PyReadonlyArray1<'py, f64>),
    C64(PyReadonlyArray1<'py, Complex64>),
}

pub enum NumericReadonlyArray2<'py> {
    F32(PyReadonlyArray2<'py, f32>),
    F64(PyReadonlyArray2<'py, f64>),
    C64(PyReadonlyArray2<'py, Complex64>),
}

pub enum IndexReadonlyArray1<'py> {
    I32(PyReadonlyArray1<'py, i32>),
    I64(PyReadonlyArray1<'py, i64>),
}

fn real_array_type_error(name: &str, rank: usize) -> PyErr {
    PyTypeError::new_err(format!(
        "{name} must be a NumPy array with dtype float32 or float64 and rank {rank}"
    ))
}

fn index_array_type_error(name: &str) -> PyErr {
    PyTypeError::new_err(format!(
        "{name} must be a NumPy array with dtype int32 or int64 and rank 1"
    ))
}

fn numeric_array_type_error(name: &str, rank: usize) -> PyErr {
    PyTypeError::new_err(format!(
        "{name} must be a NumPy array with dtype float32, float64, or complex128 and rank {rank}"
    ))
}

pub fn matching_real_dtype_error(names: &[&str]) -> PyErr {
    PyTypeError::new_err(format!(
        "{} must all have matching dtype (all float32 or all float64)",
        names.join(", ")
    ))
}

pub fn matching_numeric_dtype_error(names: &[&str]) -> PyErr {
    PyTypeError::new_err(format!(
        "{} must all have matching dtype (all float32, all float64, or all complex128)",
        names.join(", ")
    ))
}

pub fn matching_mixed_provider_dtype_error(names: &[&str]) -> PyErr {
    PyTypeError::new_err(format!(
        "{} must all have matching dtype (all float64 or all complex128)",
        names.join(", ")
    ))
}

pub fn matching_index_dtype_error(names: &[&str]) -> PyErr {
    PyTypeError::new_err(format!(
        "{} must all have matching dtype (all int32 or all int64)",
        names.join(", ")
    ))
}

pub fn f64_to_f32(value: f64, name: &str) -> PyResult<f32> {
    value
        .to_f32()
        .ok_or_else(|| PyOverflowError::new_err(format!("{name} must be representable as float32")))
}

pub fn f64_to_real<T: FromPrimitive>(value: f64, name: &str) -> PyResult<T> {
    T::from_f64(value).ok_or_else(|| {
        PyOverflowError::new_err(format!(
            "{name} must be representable as {}",
            std::any::type_name::<T>().rsplit("::").next().unwrap_or("requested dtype")
        ))
    })
}

pub fn real_array1<'py>(
    array: &Bound<'py, PyAny>,
    name: &str,
) -> PyResult<RealReadonlyArray1<'py>> {
    if let Ok(array) = array.cast::<PyArray1<f32>>() {
        require_contiguous(array)?;
        return Ok(RealReadonlyArray1::F32(array.readonly()));
    }
    if let Ok(array) = array.cast::<PyArray1<f64>>() {
        require_contiguous(array)?;
        return Ok(RealReadonlyArray1::F64(array.readonly()));
    }
    Err(real_array_type_error(name, 1))
}

pub fn real_array2<'py>(
    array: &Bound<'py, PyAny>,
    name: &str,
) -> PyResult<RealReadonlyArray2<'py>> {
    if let Ok(array) = array.cast::<PyArray2<f32>>() {
        require_contiguous(array)?;
        return Ok(RealReadonlyArray2::F32(array.readonly()));
    }
    if let Ok(array) = array.cast::<PyArray2<f64>>() {
        require_contiguous(array)?;
        return Ok(RealReadonlyArray2::F64(array.readonly()));
    }
    Err(real_array_type_error(name, 2))
}

pub fn real_array3<'py>(
    array: &Bound<'py, PyAny>,
    name: &str,
) -> PyResult<RealReadonlyArray3<'py>> {
    if let Ok(array) = array.cast::<PyArray3<f32>>() {
        require_contiguous(array)?;
        return Ok(RealReadonlyArray3::F32(array.readonly()));
    }
    if let Ok(array) = array.cast::<PyArray3<f64>>() {
        require_contiguous(array)?;
        return Ok(RealReadonlyArray3::F64(array.readonly()));
    }
    Err(real_array_type_error(name, 3))
}

pub fn real_arrayd<'py>(
    array: &Bound<'py, PyAny>,
    name: &str,
) -> PyResult<RealReadonlyArrayDyn<'py>> {
    if let Ok(array) = array.cast::<PyArrayDyn<f32>>() {
        require_contiguous(array)?;
        return Ok(RealReadonlyArrayDyn::F32(array.readonly()));
    }
    if let Ok(array) = array.cast::<PyArrayDyn<f64>>() {
        require_contiguous(array)?;
        return Ok(RealReadonlyArrayDyn::F64(array.readonly()));
    }
    Err(PyTypeError::new_err(format!(
        "{name} must be a NumPy array with dtype float32 or float64"
    )))
}

pub fn index_array1<'py>(
    array: &Bound<'py, PyAny>,
    name: &str,
) -> PyResult<IndexReadonlyArray1<'py>> {
    if let Ok(array) = array.cast::<PyArray1<i32>>() {
        return Ok(IndexReadonlyArray1::I32(array.readonly()));
    }
    if let Ok(array) = array.cast::<PyArray1<i64>>() {
        return Ok(IndexReadonlyArray1::I64(array.readonly()));
    }
    Err(index_array_type_error(name))
}

pub fn numeric_array1<'py>(
    array: &Bound<'py, PyAny>,
    name: &str,
) -> PyResult<NumericReadonlyArray1<'py>> {
    if let Ok(array) = array.cast::<PyArray1<f32>>() {
        require_contiguous(array)?;
        return Ok(NumericReadonlyArray1::F32(array.readonly()));
    }
    if let Ok(array) = array.cast::<PyArray1<f64>>() {
        require_contiguous(array)?;
        return Ok(NumericReadonlyArray1::F64(array.readonly()));
    }
    if let Ok(array) = array.cast::<PyArray1<Complex64>>() {
        require_contiguous(array)?;
        return Ok(NumericReadonlyArray1::C64(array.readonly()));
    }
    Err(numeric_array_type_error(name, 1))
}

pub fn numeric_array2<'py>(
    array: &Bound<'py, PyAny>,
    name: &str,
) -> PyResult<NumericReadonlyArray2<'py>> {
    if let Ok(array) = array.cast::<PyArray2<f32>>() {
        require_contiguous(array)?;
        return Ok(NumericReadonlyArray2::F32(array.readonly()));
    }
    if let Ok(array) = array.cast::<PyArray2<f64>>() {
        require_contiguous(array)?;
        return Ok(NumericReadonlyArray2::F64(array.readonly()));
    }
    if let Ok(array) = array.cast::<PyArray2<Complex64>>() {
        require_contiguous(array)?;
        return Ok(NumericReadonlyArray2::C64(array.readonly()));
    }
    Err(numeric_array_type_error(name, 2))
}

pub fn pyarray1_from_owned<T: Element>(py: Python<'_>, array: Array1<T>) -> Py<PyAny> {
    PyArray1::from_owned_array(py, array).into_any().unbind()
}

pub fn pyarray2_from_owned<T: Element>(py: Python<'_>, array: Array2<T>) -> Py<PyAny> {
    PyArray2::from_owned_array(py, array).into_any().unbind()
}

pub fn pyarray3_from_owned<T: Element>(py: Python<'_>, array: Array3<T>) -> Py<PyAny> {
    PyArray3::from_owned_array(py, array).into_any().unbind()
}

pub fn pyarrayd_from_owned<T: Element>(py: Python<'_>, array: ArrayD<T>) -> Py<PyAny> {
    PyArrayDyn::from_owned_array(py, array).into_any().unbind()
}

pub fn output_array1<'py, T: Element>(
    array: &Bound<'py, PyAny>,
    name: &str,
    dtype_label: &str,
) -> PyResult<PyReadwriteArray1<'py, T>> {
    array
        .cast::<PyArray1<T>>()
        .map_err(|_| {
            PyTypeError::new_err(format!(
                "{name} must be a writable NumPy array with dtype {dtype_label} and rank 1",
            ))
        })?
        .try_readwrite()
        .map_err(Into::into)
}

pub fn output_array2<'py, T: Element>(
    array: &Bound<'py, PyAny>,
    name: &str,
    dtype_label: &str,
) -> PyResult<PyReadwriteArray2<'py, T>> {
    array
        .cast::<PyArray2<T>>()
        .map_err(|_| {
            PyTypeError::new_err(format!(
                "{name} must be a writable NumPy array with dtype {dtype_label} and rank 2",
            ))
        })?
        .try_readwrite()
        .map_err(Into::into)
}

pub fn output_array3<'py, T: Element>(
    array: &Bound<'py, PyAny>,
    name: &str,
    dtype_label: &str,
) -> PyResult<PyReadwriteArray3<'py, T>> {
    array
        .cast::<PyArray3<T>>()
        .map_err(|_| {
            PyTypeError::new_err(format!(
                "{name} must be a writable NumPy array with dtype {dtype_label} and rank 3",
            ))
        })?
        .try_readwrite()
        .map_err(Into::into)
}

pub fn py_float(py: Python<'_>, value: f64) -> Py<PyAny> {
    value.into_pyobject(py).expect("f64 conversion is infallible").into_any().unbind()
}

pub fn py_complex(py: Python<'_>, value: Complex64) -> Py<PyAny> {
    PyComplex::from_doubles(py, value.re, value.im).into_any().unbind()
}
