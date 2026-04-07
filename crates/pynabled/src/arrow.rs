//! PyArrow bridge for nabled/ndarrow workflows.
//!
//! When built with the `arrow` feature, this module provides zero-copy conversion
//! from PyArrow arrays to nabled's Arrow-facing APIs.

use std::sync::Arc;

use arrow_array::types::{ArrowPrimitiveType, Float32Type, Float64Type};
use arrow_array::{Array, FixedSizeListArray, PrimitiveArray, make_array};
use arrow_data::ArrayData;
use arrow_pyarrow::PyArrowType;
use arrow_schema::Field;
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

fn array_data_to_fixed_size_list(data: ArrayData) -> PyResult<FixedSizeListArray> {
    make_array(data)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .cloned()
        .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("expected FixedSizeList array"))
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
    let fsl = array_data_to_fixed_size_list(data)?;
    let values = fsl.values();
    if values.as_any().downcast_ref::<PrimitiveArray<Float32Type>>().is_some() {
        Ok(RealFixedSizeListArray::F32(fsl))
    } else if values.as_any().downcast_ref::<PrimitiveArray<Float64Type>>().is_some() {
        Ok(RealFixedSizeListArray::F64(fsl))
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "expected FixedSizeList array with float32 or float64 values",
        ))
    }
}

fn primitive_array_into_pyarrow<T>(array: PrimitiveArray<T>) -> PyArrowType<ArrayData>
where
    T: ArrowPrimitiveType,
{
    PyArrowType(array.into_data())
}

fn fixed_size_list_into_pyarrow(array: FixedSizeListArray) -> PyArrowType<ArrayData> {
    PyArrowType(array.into_data())
}

fn extension_array_into_pyarrow(
    field: Field,
    array: FixedSizeListArray,
) -> (PyArrowType<Field>, PyArrowType<ArrayData>) {
    (PyArrowType(field), fixed_size_list_into_pyarrow(array))
}

fn fixed_size_list_with_item_nullability(
    array: &FixedSizeListArray,
    nullable: bool,
) -> FixedSizeListArray {
    FixedSizeListArray::new(
        Arc::new(Field::new("item", array.value_type().clone(), nullable)),
        array.value_length(),
        Arc::clone(array.values()),
        array.nulls().cloned(),
    )
}

fn fixed_size_list_with_non_null_item(array: &FixedSizeListArray) -> FixedSizeListArray {
    fixed_size_list_with_item_nullability(array, false)
}

fn fixed_size_list_with_nullable_item(array: &FixedSizeListArray) -> FixedSizeListArray {
    fixed_size_list_with_item_nullability(array, true)
}

fn field_with_array_storage(field: &Field, array: &FixedSizeListArray) -> Field {
    Field::new(field.name(), array.data_type().clone(), false)
        .with_metadata(field.metadata().clone())
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

/// Compute cosine similarity of two real PyArrow arrays.
#[pyfunction(name = "arrow_cosine_similarity")]
pub fn cosine_similarity(
    left: PyArrowType<ArrayData>,
    right: PyArrowType<ArrayData>,
) -> PyResult<f64> {
    match (array_data_to_real_primitive(left.0)?, array_data_to_real_primitive(right.0)?) {
        (RealPrimitiveArray::F32(left_arr), RealPrimitiveArray::F32(right_arr)) => Ok(f64::from(
            nabled::arrow::vector::cosine_similarity(&left_arr, &right_arr).map_err(to_py_err)?,
        )),
        (RealPrimitiveArray::F64(left_arr), RealPrimitiveArray::F64(right_arr)) => {
            nabled::arrow::vector::cosine_similarity(&left_arr, &right_arr).map_err(to_py_err)
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

/// Compute cosine distance of two real PyArrow arrays.
#[pyfunction(name = "arrow_cosine_distance")]
pub fn cosine_distance(
    left: PyArrowType<ArrayData>,
    right: PyArrowType<ArrayData>,
) -> PyResult<f64> {
    match (array_data_to_real_primitive(left.0)?, array_data_to_real_primitive(right.0)?) {
        (RealPrimitiveArray::F32(left_arr), RealPrimitiveArray::F32(right_arr)) => Ok(f64::from(
            nabled::arrow::vector::cosine_distance(&left_arr, &right_arr).map_err(to_py_err)?,
        )),
        (RealPrimitiveArray::F64(left_arr), RealPrimitiveArray::F64(right_arr)) => {
            nabled::arrow::vector::cosine_distance(&left_arr, &right_arr).map_err(to_py_err)
        }
        _ => Err(utils::matching_real_dtype_error(&["left", "right"])),
    }
}

/// Compute pairwise L2 distances between Arrow row batches.
#[pyfunction(name = "arrow_pairwise_l2_distance")]
pub fn pairwise_l2_distance(
    left: PyArrowType<ArrayData>,
    right: PyArrowType<ArrayData>,
) -> PyResult<PyArrowType<ArrayData>> {
    match (
        array_data_to_real_fixed_size_list(left.0)?,
        array_data_to_real_fixed_size_list(right.0)?,
    ) {
        (RealFixedSizeListArray::F32(left_arr), RealFixedSizeListArray::F32(right_arr)) => {
            Ok(fixed_size_list_into_pyarrow(
                nabled::arrow::vector::pairwise_l2_distance::<Float32Type>(&left_arr, &right_arr)
                    .map_err(to_py_err)?,
            ))
        }
        (RealFixedSizeListArray::F64(left_arr), RealFixedSizeListArray::F64(right_arr)) => {
            Ok(fixed_size_list_into_pyarrow(
                nabled::arrow::vector::pairwise_l2_distance::<Float64Type>(&left_arr, &right_arr)
                    .map_err(to_py_err)?,
            ))
        }
        _ => Err(utils::matching_real_dtype_error(&["left", "right"])),
    }
}

/// Compute pairwise cosine similarities between Arrow row batches.
#[pyfunction(name = "arrow_pairwise_cosine_similarity")]
pub fn pairwise_cosine_similarity(
    left: PyArrowType<ArrayData>,
    right: PyArrowType<ArrayData>,
) -> PyResult<PyArrowType<ArrayData>> {
    match (
        array_data_to_real_fixed_size_list(left.0)?,
        array_data_to_real_fixed_size_list(right.0)?,
    ) {
        (RealFixedSizeListArray::F32(left_arr), RealFixedSizeListArray::F32(right_arr)) => {
            Ok(fixed_size_list_into_pyarrow(
                nabled::arrow::vector::pairwise_cosine_similarity::<Float32Type>(
                    &left_arr, &right_arr,
                )
                .map_err(to_py_err)?,
            ))
        }
        (RealFixedSizeListArray::F64(left_arr), RealFixedSizeListArray::F64(right_arr)) => {
            Ok(fixed_size_list_into_pyarrow(
                nabled::arrow::vector::pairwise_cosine_similarity::<Float64Type>(
                    &left_arr, &right_arr,
                )
                .map_err(to_py_err)?,
            ))
        }
        _ => Err(utils::matching_real_dtype_error(&["left", "right"])),
    }
}

/// Compute pairwise cosine distances between Arrow row batches.
#[pyfunction(name = "arrow_pairwise_cosine_distance")]
pub fn pairwise_cosine_distance(
    left: PyArrowType<ArrayData>,
    right: PyArrowType<ArrayData>,
) -> PyResult<PyArrowType<ArrayData>> {
    match (
        array_data_to_real_fixed_size_list(left.0)?,
        array_data_to_real_fixed_size_list(right.0)?,
    ) {
        (RealFixedSizeListArray::F32(left_arr), RealFixedSizeListArray::F32(right_arr)) => {
            Ok(fixed_size_list_into_pyarrow(
                nabled::arrow::vector::pairwise_cosine_distance::<Float32Type>(
                    &left_arr, &right_arr,
                )
                .map_err(to_py_err)?,
            ))
        }
        (RealFixedSizeListArray::F64(left_arr), RealFixedSizeListArray::F64(right_arr)) => {
            Ok(fixed_size_list_into_pyarrow(
                nabled::arrow::vector::pairwise_cosine_distance::<Float64Type>(
                    &left_arr, &right_arr,
                )
                .map_err(to_py_err)?,
            ))
        }
        _ => Err(utils::matching_real_dtype_error(&["left", "right"])),
    }
}

/// Compute row-wise dot products across Arrow row batches.
#[pyfunction(name = "arrow_batched_dot")]
pub fn batched_dot(
    left: PyArrowType<ArrayData>,
    right: PyArrowType<ArrayData>,
) -> PyResult<PyArrowType<ArrayData>> {
    match (
        array_data_to_real_fixed_size_list(left.0)?,
        array_data_to_real_fixed_size_list(right.0)?,
    ) {
        (RealFixedSizeListArray::F32(left_arr), RealFixedSizeListArray::F32(right_arr)) => {
            Ok(primitive_array_into_pyarrow(
                nabled::arrow::vector::batched_dot::<Float32Type>(&left_arr, &right_arr)
                    .map_err(to_py_err)?,
            ))
        }
        (RealFixedSizeListArray::F64(left_arr), RealFixedSizeListArray::F64(right_arr)) => {
            Ok(primitive_array_into_pyarrow(
                nabled::arrow::vector::batched_dot::<Float64Type>(&left_arr, &right_arr)
                    .map_err(to_py_err)?,
            ))
        }
        _ => Err(utils::matching_real_dtype_error(&["left", "right"])),
    }
}

/// Compute row-wise L2 norms across Arrow row batches.
#[pyfunction(name = "arrow_batched_l2_norm")]
pub fn batched_l2_norm(array: PyArrowType<ArrayData>) -> PyResult<PyArrowType<ArrayData>> {
    match array_data_to_real_fixed_size_list(array.0)? {
        RealFixedSizeListArray::F32(arr) => Ok(primitive_array_into_pyarrow(
            nabled::arrow::vector::batched_l2_norm::<Float32Type>(&arr).map_err(to_py_err)?,
        )),
        RealFixedSizeListArray::F64(arr) => Ok(primitive_array_into_pyarrow(
            nabled::arrow::vector::batched_l2_norm::<Float64Type>(&arr).map_err(to_py_err)?,
        )),
    }
}

/// Compute row-wise cosine similarities across Arrow row batches.
#[pyfunction(name = "arrow_batched_cosine_similarity")]
pub fn batched_cosine_similarity(
    left: PyArrowType<ArrayData>,
    right: PyArrowType<ArrayData>,
) -> PyResult<PyArrowType<ArrayData>> {
    match (
        array_data_to_real_fixed_size_list(left.0)?,
        array_data_to_real_fixed_size_list(right.0)?,
    ) {
        (RealFixedSizeListArray::F32(left_arr), RealFixedSizeListArray::F32(right_arr)) => {
            Ok(primitive_array_into_pyarrow(
                nabled::arrow::vector::batched_cosine_similarity::<Float32Type>(
                    &left_arr, &right_arr,
                )
                .map_err(to_py_err)?,
            ))
        }
        (RealFixedSizeListArray::F64(left_arr), RealFixedSizeListArray::F64(right_arr)) => {
            Ok(primitive_array_into_pyarrow(
                nabled::arrow::vector::batched_cosine_similarity::<Float64Type>(
                    &left_arr, &right_arr,
                )
                .map_err(to_py_err)?,
            ))
        }
        _ => Err(utils::matching_real_dtype_error(&["left", "right"])),
    }
}

/// Compute row-wise cosine distances across Arrow row batches.
#[pyfunction(name = "arrow_batched_cosine_distance")]
pub fn batched_cosine_distance(
    left: PyArrowType<ArrayData>,
    right: PyArrowType<ArrayData>,
) -> PyResult<PyArrowType<ArrayData>> {
    match (
        array_data_to_real_fixed_size_list(left.0)?,
        array_data_to_real_fixed_size_list(right.0)?,
    ) {
        (RealFixedSizeListArray::F32(left_arr), RealFixedSizeListArray::F32(right_arr)) => {
            Ok(primitive_array_into_pyarrow(
                nabled::arrow::vector::batched_cosine_distance::<Float32Type>(
                    &left_arr, &right_arr,
                )
                .map_err(to_py_err)?,
            ))
        }
        (RealFixedSizeListArray::F64(left_arr), RealFixedSizeListArray::F64(right_arr)) => {
            Ok(primitive_array_into_pyarrow(
                nabled::arrow::vector::batched_cosine_distance::<Float64Type>(
                    &left_arr, &right_arr,
                )
                .map_err(to_py_err)?,
            ))
        }
        _ => Err(utils::matching_real_dtype_error(&["left", "right"])),
    }
}

/// Normalize Arrow row batches.
#[pyfunction(name = "arrow_batched_normalize")]
pub fn batched_normalize(array: PyArrowType<ArrayData>) -> PyResult<PyArrowType<ArrayData>> {
    match array_data_to_real_fixed_size_list(array.0)? {
        RealFixedSizeListArray::F32(arr) => Ok(fixed_size_list_into_pyarrow(
            nabled::arrow::vector::batched_normalize::<Float32Type>(&arr).map_err(to_py_err)?,
        )),
        RealFixedSizeListArray::F64(arr) => Ok(fixed_size_list_into_pyarrow(
            nabled::arrow::vector::batched_normalize::<Float64Type>(&arr).map_err(to_py_err)?,
        )),
    }
}

/// Compute a dense matrix-vector product from PyArrow carriers.
#[pyfunction(name = "arrow_matvec")]
pub fn matvec(
    matrix: PyArrowType<ArrayData>,
    vector: PyArrowType<ArrayData>,
) -> PyResult<PyArrowType<ArrayData>> {
    match (array_data_to_real_fixed_size_list(matrix.0)?, array_data_to_real_primitive(vector.0)?) {
        (RealFixedSizeListArray::F32(matrix_arr), RealPrimitiveArray::F32(vector_arr)) => {
            Ok(primitive_array_into_pyarrow(
                nabled::arrow::matrix::matvec::<Float32Type>(&matrix_arr, &vector_arr)
                    .map_err(to_py_err)?,
            ))
        }
        (RealFixedSizeListArray::F64(matrix_arr), RealPrimitiveArray::F64(vector_arr)) => {
            Ok(primitive_array_into_pyarrow(
                nabled::arrow::matrix::matvec::<Float64Type>(&matrix_arr, &vector_arr)
                    .map_err(to_py_err)?,
            ))
        }
        _ => Err(utils::matching_real_dtype_error(&["matrix", "vector"])),
    }
}

/// Compute a dense matrix-matrix product from PyArrow carriers.
#[pyfunction(name = "arrow_matmat")]
pub fn matmat(
    left: PyArrowType<ArrayData>,
    right: PyArrowType<ArrayData>,
) -> PyResult<PyArrowType<ArrayData>> {
    match (
        array_data_to_real_fixed_size_list(left.0)?,
        array_data_to_real_fixed_size_list(right.0)?,
    ) {
        (RealFixedSizeListArray::F32(left_arr), RealFixedSizeListArray::F32(right_arr)) => {
            Ok(fixed_size_list_into_pyarrow(
                nabled::arrow::matrix::matmat::<Float32Type>(&left_arr, &right_arr)
                    .map_err(to_py_err)?,
            ))
        }
        (RealFixedSizeListArray::F64(left_arr), RealFixedSizeListArray::F64(right_arr)) => {
            Ok(fixed_size_list_into_pyarrow(
                nabled::arrow::matrix::matmat::<Float64Type>(&left_arr, &right_arr)
                    .map_err(to_py_err)?,
            ))
        }
        _ => Err(utils::matching_real_dtype_error(&["left", "right"])),
    }
}

/// Apply one dense matrix to a batch of Arrow row vectors.
#[pyfunction(name = "arrow_batched_row_matvec")]
pub fn batched_row_matvec(
    batch_vectors: PyArrowType<ArrayData>,
    matrix: PyArrowType<ArrayData>,
) -> PyResult<PyArrowType<ArrayData>> {
    match (
        array_data_to_real_fixed_size_list(batch_vectors.0)?,
        array_data_to_real_fixed_size_list(matrix.0)?,
    ) {
        (RealFixedSizeListArray::F32(batch_arr), RealFixedSizeListArray::F32(matrix_arr)) => {
            Ok(fixed_size_list_into_pyarrow(
                nabled::arrow::matrix::batched_row_matvec::<Float32Type>(&batch_arr, &matrix_arr)
                    .map_err(to_py_err)?,
            ))
        }
        (RealFixedSizeListArray::F64(batch_arr), RealFixedSizeListArray::F64(matrix_arr)) => {
            Ok(fixed_size_list_into_pyarrow(
                nabled::arrow::matrix::batched_row_matvec::<Float64Type>(&batch_arr, &matrix_arr)
                    .map_err(to_py_err)?,
            ))
        }
        _ => Err(utils::matching_real_dtype_error(&["batch_vectors", "matrix"])),
    }
}

/// Compute batched dense matrix-matrix products from Arrow fixed-shape tensors.
#[pyfunction(name = "arrow_batched_matmat")]
pub fn batched_matmat(
    left_field: PyArrowType<Field>,
    left: PyArrowType<ArrayData>,
    right_field: PyArrowType<Field>,
    right: PyArrowType<ArrayData>,
) -> PyResult<(PyArrowType<Field>, PyArrowType<ArrayData>)> {
    match (
        array_data_to_real_fixed_size_list(left.0)?,
        array_data_to_real_fixed_size_list(right.0)?,
    ) {
        (RealFixedSizeListArray::F32(left_arr), RealFixedSizeListArray::F32(right_arr)) => {
            let left_arr = fixed_size_list_with_non_null_item(&left_arr);
            let right_arr = fixed_size_list_with_non_null_item(&right_arr);
            let left_field = field_with_array_storage(&left_field.0, &left_arr);
            let right_field = field_with_array_storage(&right_field.0, &right_arr);
            let (field, array) = nabled::arrow::matrix::batched_matmat::<Float32Type>(
                &left_field,
                &left_arr,
                &right_field,
                &right_arr,
            )
            .map_err(to_py_err)?;
            let array = fixed_size_list_with_nullable_item(&array);
            let field = field_with_array_storage(&field, &array);
            Ok(extension_array_into_pyarrow(field, array))
        }
        (RealFixedSizeListArray::F64(left_arr), RealFixedSizeListArray::F64(right_arr)) => {
            let left_arr = fixed_size_list_with_non_null_item(&left_arr);
            let right_arr = fixed_size_list_with_non_null_item(&right_arr);
            let left_field = field_with_array_storage(&left_field.0, &left_arr);
            let right_field = field_with_array_storage(&right_field.0, &right_arr);
            let (field, array) = nabled::arrow::matrix::batched_matmat::<Float64Type>(
                &left_field,
                &left_arr,
                &right_field,
                &right_arr,
            )
            .map_err(to_py_err)?;
            let array = fixed_size_list_with_nullable_item(&array);
            let field = field_with_array_storage(&field, &array);
            Ok(extension_array_into_pyarrow(field, array))
        }
        _ => Err(utils::matching_real_dtype_error(&["left", "right"])),
    }
}

/// Compute batched dense matrix-matrix products with a broadcasted right operand.
#[pyfunction(name = "arrow_batched_matmat_broadcast_right")]
pub fn batched_matmat_broadcast_right(
    left_field: PyArrowType<Field>,
    left: PyArrowType<ArrayData>,
    right: PyArrowType<ArrayData>,
) -> PyResult<(PyArrowType<Field>, PyArrowType<ArrayData>)> {
    match (
        array_data_to_real_fixed_size_list(left.0)?,
        array_data_to_real_fixed_size_list(right.0)?,
    ) {
        (RealFixedSizeListArray::F32(left_arr), RealFixedSizeListArray::F32(right_arr)) => {
            let left_arr = fixed_size_list_with_non_null_item(&left_arr);
            let left_field = field_with_array_storage(&left_field.0, &left_arr);
            let (field, array) =
                nabled::arrow::matrix::batched_matmat_broadcast_right::<Float32Type>(
                    &left_field,
                    &left_arr,
                    &right_arr,
                )
                .map_err(to_py_err)?;
            let array = fixed_size_list_with_nullable_item(&array);
            let field = field_with_array_storage(&field, &array);
            Ok(extension_array_into_pyarrow(field, array))
        }
        (RealFixedSizeListArray::F64(left_arr), RealFixedSizeListArray::F64(right_arr)) => {
            let left_arr = fixed_size_list_with_non_null_item(&left_arr);
            let left_field = field_with_array_storage(&left_field.0, &left_arr);
            let (field, array) =
                nabled::arrow::matrix::batched_matmat_broadcast_right::<Float64Type>(
                    &left_field,
                    &left_arr,
                    &right_arr,
                )
                .map_err(to_py_err)?;
            let array = fixed_size_list_with_nullable_item(&array);
            let field = field_with_array_storage(&field, &array);
            Ok(extension_array_into_pyarrow(field, array))
        }
        _ => Err(utils::matching_real_dtype_error(&["left", "right"])),
    }
}

/// Compute batched dense matrix-matrix products with a broadcasted left operand.
#[pyfunction(name = "arrow_batched_matmat_broadcast_left")]
pub fn batched_matmat_broadcast_left(
    left: PyArrowType<ArrayData>,
    right_field: PyArrowType<Field>,
    right: PyArrowType<ArrayData>,
) -> PyResult<(PyArrowType<Field>, PyArrowType<ArrayData>)> {
    match (
        array_data_to_real_fixed_size_list(left.0)?,
        array_data_to_real_fixed_size_list(right.0)?,
    ) {
        (RealFixedSizeListArray::F32(left_arr), RealFixedSizeListArray::F32(right_arr)) => {
            let right_arr = fixed_size_list_with_non_null_item(&right_arr);
            let right_field = field_with_array_storage(&right_field.0, &right_arr);
            let (field, array) =
                nabled::arrow::matrix::batched_matmat_broadcast_left::<Float32Type>(
                    &left_arr,
                    &right_field,
                    &right_arr,
                )
                .map_err(to_py_err)?;
            let array = fixed_size_list_with_nullable_item(&array);
            let field = field_with_array_storage(&field, &array);
            Ok(extension_array_into_pyarrow(field, array))
        }
        (RealFixedSizeListArray::F64(left_arr), RealFixedSizeListArray::F64(right_arr)) => {
            let right_arr = fixed_size_list_with_non_null_item(&right_arr);
            let right_field = field_with_array_storage(&right_field.0, &right_arr);
            let (field, array) =
                nabled::arrow::matrix::batched_matmat_broadcast_left::<Float64Type>(
                    &left_arr,
                    &right_field,
                    &right_arr,
                )
                .map_err(to_py_err)?;
            let array = fixed_size_list_with_nullable_item(&array);
            let field = field_with_array_storage(&field, &array);
            Ok(extension_array_into_pyarrow(field, array))
        }
        _ => Err(utils::matching_real_dtype_error(&["left", "right"])),
    }
}

/// Compute column means from a dense PyArrow matrix.
#[pyfunction(name = "arrow_column_means")]
pub fn column_means(matrix: PyArrowType<ArrayData>) -> PyResult<PyArrowType<ArrayData>> {
    match array_data_to_real_fixed_size_list(matrix.0)? {
        RealFixedSizeListArray::F32(matrix_arr) => Ok(primitive_array_into_pyarrow(
            nabled::arrow::stats::column_means_f32(&matrix_arr).map_err(to_py_err)?,
        )),
        RealFixedSizeListArray::F64(matrix_arr) => Ok(primitive_array_into_pyarrow(
            nabled::arrow::stats::column_means_f64(&matrix_arr).map_err(to_py_err)?,
        )),
    }
}

/// Center columns from a dense PyArrow matrix.
#[pyfunction(name = "arrow_center_columns")]
pub fn center_columns(matrix: PyArrowType<ArrayData>) -> PyResult<PyArrowType<ArrayData>> {
    match array_data_to_real_fixed_size_list(matrix.0)? {
        RealFixedSizeListArray::F32(matrix_arr) => Ok(fixed_size_list_into_pyarrow(
            nabled::arrow::stats::center_columns_f32(&matrix_arr).map_err(to_py_err)?,
        )),
        RealFixedSizeListArray::F64(matrix_arr) => Ok(fixed_size_list_into_pyarrow(
            nabled::arrow::stats::center_columns_f64(&matrix_arr).map_err(to_py_err)?,
        )),
    }
}

/// Compute covariance from a dense PyArrow matrix.
#[pyfunction(name = "arrow_covariance_matrix")]
pub fn covariance_matrix(matrix: PyArrowType<ArrayData>) -> PyResult<PyArrowType<ArrayData>> {
    match array_data_to_real_fixed_size_list(matrix.0)? {
        RealFixedSizeListArray::F32(matrix_arr) => Ok(fixed_size_list_into_pyarrow(
            nabled::arrow::stats::covariance_matrix_f32(&matrix_arr).map_err(to_py_err)?,
        )),
        RealFixedSizeListArray::F64(matrix_arr) => Ok(fixed_size_list_into_pyarrow(
            nabled::arrow::stats::covariance_matrix_f64(&matrix_arr).map_err(to_py_err)?,
        )),
    }
}

/// Compute correlation from a dense PyArrow matrix.
#[pyfunction(name = "arrow_correlation_matrix")]
pub fn correlation_matrix(matrix: PyArrowType<ArrayData>) -> PyResult<PyArrowType<ArrayData>> {
    match array_data_to_real_fixed_size_list(matrix.0)? {
        RealFixedSizeListArray::F32(matrix_arr) => Ok(fixed_size_list_into_pyarrow(
            nabled::arrow::stats::correlation_matrix_f32(&matrix_arr).map_err(to_py_err)?,
        )),
        RealFixedSizeListArray::F64(matrix_arr) => Ok(fixed_size_list_into_pyarrow(
            nabled::arrow::stats::correlation_matrix_f64(&matrix_arr).map_err(to_py_err)?,
        )),
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
