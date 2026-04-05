//! PyArrow bridge for nabled/ndarrow workflows.
//!
//! When built with the `arrow` feature, this module provides zero-copy conversion
//! from PyArrow arrays to nabled's Arrow-facing APIs.

use arrow_array::types::{Float32Type, Float64Type};
use arrow_array::{Array, FixedSizeListArray, PrimitiveArray, make_array};
use arrow_data::ArrayData;
use arrow_pyarrow::PyArrowType;
use pyo3::prelude::*;

use crate::error::to_py_err;
use crate::utils;

enum RealPrimitiveArray {
    F32(PrimitiveArray<Float32Type>),
    F64(PrimitiveArray<Float64Type>),
}

enum RealFixedSizeListArray {
    F32(FixedSizeListArray),
    F64(FixedSizeListArray),
}

fn array_data_to_real_primitive(data: ArrayData) -> PyResult<RealPrimitiveArray> {
    let arr = make_array(data);
    if let Some(prim) = arr.as_any().downcast_ref::<PrimitiveArray<Float32Type>>() {
        return Ok(RealPrimitiveArray::F32(prim.clone()));
    }
    if let Some(prim) = arr.as_any().downcast_ref::<PrimitiveArray<Float64Type>>() {
        return Ok(RealPrimitiveArray::F64(prim.clone()));
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "expected float32 or float64 Arrow primitive array",
    ))
}

fn array_data_to_real_fixed_size_list(data: ArrayData) -> PyResult<RealFixedSizeListArray> {
    let arr = make_array(data);
    let fsl = arr
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("expected FixedSizeList array"))?;
    let values = fsl.values();
    if values.as_any().downcast_ref::<PrimitiveArray<Float32Type>>().is_some() {
        Ok(RealFixedSizeListArray::F32(fsl.clone()))
    } else if values.as_any().downcast_ref::<PrimitiveArray<Float64Type>>().is_some() {
        Ok(RealFixedSizeListArray::F64(fsl.clone()))
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "expected FixedSizeList array with float32 or float64 values",
        ))
    }
}

/// Compute dot product of two real PyArrow arrays.
#[pyfunction(name = "arrow_dot")]
pub fn dot(left: PyArrowType<ArrayData>, right: PyArrowType<ArrayData>) -> PyResult<f64> {
    match (array_data_to_real_primitive(left.0)?, array_data_to_real_primitive(right.0)?) {
        (RealPrimitiveArray::F32(left_arr), RealPrimitiveArray::F32(right_arr)) => {
            Ok(f64::from(nabled::arrow::vector::dot(&left_arr, &right_arr).map_err(to_py_err)?))
        }
        (RealPrimitiveArray::F64(left_arr), RealPrimitiveArray::F64(right_arr)) => {
            nabled::arrow::vector::dot(&left_arr, &right_arr).map_err(to_py_err)
        }
        _ => Err(utils::matching_real_dtype_error(&["left", "right"])),
    }
}

/// Compute L2 norm of a real PyArrow array.
#[pyfunction(name = "arrow_l2_norm")]
pub fn l2_norm(array: PyArrowType<ArrayData>) -> PyResult<f64> {
    match array_data_to_real_primitive(array.0)? {
        RealPrimitiveArray::F32(arr) => {
            Ok(f64::from(nabled::arrow::vector::l2_norm(&arr).map_err(to_py_err)?))
        }
        RealPrimitiveArray::F64(arr) => nabled::arrow::vector::l2_norm(&arr).map_err(to_py_err),
    }
}

/// Compute SVD of a real PyArrow dense matrix.
/// Returns `(U, singular_values, Vt)` as NumPy arrays preserving `float32` or `float64`.
#[pyfunction(name = "arrow_svd_decompose")]
pub fn svd_decompose(
    py: Python<'_>,
    matrix: PyArrowType<ArrayData>,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    match array_data_to_real_fixed_size_list(matrix.0)? {
        RealFixedSizeListArray::F32(fsl) => {
            let result = nabled::arrow::svd::decompose_f32(&fsl).map_err(to_py_err)?;
            Ok((
                utils::pyarray2_from_owned(py, result.u),
                utils::pyarray1_from_owned(py, result.singular_values),
                utils::pyarray2_from_owned(py, result.vt),
            ))
        }
        RealFixedSizeListArray::F64(fsl) => {
            let result = nabled::arrow::svd::decompose_f64(&fsl).map_err(to_py_err)?;
            Ok((
                utils::pyarray2_from_owned(py, result.u),
                utils::pyarray1_from_owned(py, result.singular_values),
                utils::pyarray2_from_owned(py, result.vt),
            ))
        }
    }
}
