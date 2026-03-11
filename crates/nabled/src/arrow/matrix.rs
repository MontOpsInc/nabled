//! Arrow adapters for dense matrix primitives.

use arrow_array::types::ArrowPrimitiveType;
use arrow_array::{FixedSizeListArray, PrimitiveArray};
use arrow_schema::Field;
use nabled_core::scalar::NabledReal;
use ndarray::Ix3;
use ndarrow::NdarrowElement;

use super::{
    ArrowInteropError, complex64_matrix_from_owned, complex64_matrix_view,
    fixed_shape_tensor_from_owned, fixed_shape_tensor_viewd, fixed_size_list_from_owned,
    fixed_size_list_view, primitive_array_from_owned, primitive_array_view,
};

fn fixed_shape_tensor_view3<'a, T>(
    field: &'a Field,
    array: &'a FixedSizeListArray,
) -> Result<ndarray::ArrayView3<'a, T::Native>, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NdarrowElement,
{
    let view = fixed_shape_tensor_viewd::<T>(field, array)?;
    view.into_dimensionality::<Ix3>()
        .map_err(|error: ndarray::ShapeError| ArrowInteropError::InvalidShape(error.to_string()))
}

/// Compute dense matrix-vector product directly from Arrow arrays.
///
/// `matrix` is interpreted as an `M x N` dense matrix stored as `FixedSizeList<T>(N)`.
///
/// # Errors
/// Returns an error when inputs contain nulls, are empty, or dimensions mismatch.
pub fn matvec<T>(
    matrix: &FixedSizeListArray,
    vector: &PrimitiveArray<T>,
) -> Result<PrimitiveArray<T>, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let matrix_view = fixed_size_list_view::<T>(matrix)?;
    let vector_view = primitive_array_view(vector)?;
    let output = crate::linalg::matrix::matvec_view(&matrix_view, &vector_view)?;
    Ok(primitive_array_from_owned::<T>(output))
}

/// Compute dense matrix-matrix product directly from Arrow arrays.
///
/// Both inputs are interpreted as dense matrices stored as `FixedSizeList<T>(N)`.
///
/// # Errors
/// Returns an error when inputs contain nulls, are empty, or dimensions mismatch.
pub fn matmat<T>(
    left: &FixedSizeListArray,
    right: &FixedSizeListArray,
) -> Result<FixedSizeListArray, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let left_view = fixed_size_list_view::<T>(left)?;
    let right_view = fixed_size_list_view::<T>(right)?;
    let output = crate::linalg::matrix::matmat_view(&left_view, &right_view)?;
    fixed_size_list_from_owned::<T>(output)
}

/// Apply one dense matrix to a batch of row vectors stored in Arrow dense matrices.
///
/// `batch_vectors` is interpreted as `(batch, cols)` and `matrix` as `(rows, cols)`. The result
/// is returned as Arrow dense matrix data with shape `(batch, rows)`.
///
/// # Errors
/// Returns an error when inputs contain nulls, are empty, or dimensions mismatch.
pub fn batched_row_matvec<T>(
    batch_vectors: &FixedSizeListArray,
    matrix: &FixedSizeListArray,
) -> Result<FixedSizeListArray, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let batch_view = fixed_size_list_view::<T>(batch_vectors)?;
    let matrix_view = fixed_size_list_view::<T>(matrix)?;
    let output = crate::linalg::matrix::batched_row_matvec_view(&batch_view, &matrix_view)?;
    fixed_size_list_from_owned::<T>(output)
}

/// Compute complex matrix-vector product from Arrow complex dense storage.
///
/// # Errors
/// Returns an error when inputs contain nulls, are empty, or dimensions mismatch.
pub fn matvec_complex(
    matrix: &FixedSizeListArray,
    vector_field: &Field,
    vector: &FixedSizeListArray,
) -> Result<(Field, FixedSizeListArray), ArrowInteropError> {
    let matrix_view = complex64_matrix_view(matrix)?;
    let vector_view = super::complex64_vector_view(vector_field, vector)?;
    let output = crate::linalg::matrix::matvec_complex_view(&matrix_view, &vector_view)?;
    super::complex64_vector_from_owned("matvec_complex", output)
}

/// Compute complex matrix-matrix product from Arrow complex dense storage.
///
/// # Errors
/// Returns an error when inputs contain nulls, are empty, or dimensions mismatch.
pub fn matmat_complex(
    left: &FixedSizeListArray,
    right: &FixedSizeListArray,
) -> Result<FixedSizeListArray, ArrowInteropError> {
    let left_view = complex64_matrix_view(left)?;
    let right_view = complex64_matrix_view(right)?;
    let output = crate::linalg::matrix::matmat_complex_view(&left_view, &right_view)?;
    complex64_matrix_from_owned(output)
}

/// Compute batched dense matrix-matrix products from Arrow fixed-shape tensors.
///
/// # Errors
/// Returns an error when inputs are invalid fixed-shape tensors or dimensions mismatch.
pub fn batched_matmat<T>(
    left_field: &Field,
    left: &FixedSizeListArray,
    right_field: &Field,
    right: &FixedSizeListArray,
) -> Result<(Field, FixedSizeListArray), ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement + Default,
{
    let left_view = fixed_shape_tensor_view3::<T>(left_field, left)?;
    let right_view = fixed_shape_tensor_view3::<T>(right_field, right)?;
    let output = crate::linalg::matrix::batched_matmat_view(&left_view, &right_view)?;
    fixed_shape_tensor_from_owned::<T>(left_field.name(), output.into_dyn())
}

/// Compute batched dense matrix-matrix products with a broadcasted right operand.
///
/// # Errors
/// Returns an error when inputs are invalid or dimensions mismatch.
pub fn batched_matmat_broadcast_right<T>(
    left_field: &Field,
    left: &FixedSizeListArray,
    right: &FixedSizeListArray,
) -> Result<(Field, FixedSizeListArray), ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement + Default,
{
    let left_view = fixed_shape_tensor_view3::<T>(left_field, left)?;
    let right_view = fixed_size_list_view::<T>(right)?;
    let output =
        crate::linalg::matrix::batched_matmat_broadcast_right_view(&left_view, &right_view)?;
    fixed_shape_tensor_from_owned::<T>(left_field.name(), output.into_dyn())
}

/// Compute batched dense matrix-matrix products with a broadcasted left operand.
///
/// # Errors
/// Returns an error when inputs are invalid or dimensions mismatch.
pub fn batched_matmat_broadcast_left<T>(
    left: &FixedSizeListArray,
    right_field: &Field,
    right: &FixedSizeListArray,
) -> Result<(Field, FixedSizeListArray), ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement + Default,
{
    let left_view = fixed_size_list_view::<T>(left)?;
    let right_view = fixed_shape_tensor_view3::<T>(right_field, right)?;
    let output =
        crate::linalg::matrix::batched_matmat_broadcast_left_view(&left_view, &right_view)?;
    fixed_shape_tensor_from_owned::<T>(right_field.name(), output.into_dyn())
}
