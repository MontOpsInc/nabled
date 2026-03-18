//! PyArrow bridge for nabled/ndarrow workflows.
//!
//! When built with the `arrow` feature, this module provides zero-copy conversion
//! from PyArrow arrays to nabled's Arrow-facing APIs.

use arrow_array::types::Float64Type;
use arrow_array::{Array, PrimitiveArray, make_array};
use arrow_data::ArrayData;
use arrow_pyarrow::PyArrowType;
use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;

use crate::error::to_py_err;

fn array_data_to_float64(data: ArrayData) -> PyResult<PrimitiveArray<Float64Type>> {
    let arr = make_array(data);
    let prim = arr
        .as_any()
        .downcast_ref::<PrimitiveArray<Float64Type>>()
        .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("expected float64 array"))?;
    Ok(prim.clone())
}

fn array_data_to_fixed_size_list(data: ArrayData) -> PyResult<arrow_array::FixedSizeListArray> {
    let arr = make_array(data);
    let fsl = arr
        .as_any()
        .downcast_ref::<arrow_array::FixedSizeListArray>()
        .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("expected FixedSizeList array"))?;
    Ok(fsl.clone())
}

/// Compute dot product of two PyArrow float64 arrays.
#[pyfunction(name = "arrow_dot")]
pub fn dot(left: PyArrowType<ArrayData>, right: PyArrowType<ArrayData>) -> PyResult<f64> {
    let left_arr = array_data_to_float64(left.0)?;
    let right_arr = array_data_to_float64(right.0)?;
    nabled::arrow::vector::dot(&left_arr, &right_arr).map_err(to_py_err)
}

/// Compute L2 norm of a PyArrow float64 array.
#[pyfunction(name = "arrow_l2_norm")]
pub fn l2_norm(array: PyArrowType<ArrayData>) -> PyResult<f64> {
    let arr = array_data_to_float64(array.0)?;
    nabled::arrow::vector::l2_norm(&arr).map_err(to_py_err)
}

/// Compute SVD of a PyArrow dense matrix (FixedSizeListArray of float64).
/// Returns (U, singular_values, Vt) as NumPy arrays.
#[pyfunction(name = "arrow_svd_decompose")]
pub fn svd_decompose(
    py: Python<'_>,
    matrix: PyArrowType<ArrayData>,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray1<f64>>, Py<PyArray2<f64>>)> {
    let fsl = array_data_to_fixed_size_list(matrix.0)?;
    let result = nabled::arrow::svd::decompose_f64(&fsl).map_err(to_py_err)?;
    Ok((
        PyArray2::from_owned_array(py, result.u).unbind(),
        PyArray1::from_owned_array(py, result.singular_values).unbind(),
        PyArray2::from_owned_array(py, result.vt).unbind(),
    ))
}
