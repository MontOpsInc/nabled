//! Arrow adapters for vector-first primitives.

use arrow_array::types::{ArrowPrimitiveType, Float64Type};
use arrow_array::{FixedSizeListArray, PrimitiveArray};
use arrow_schema::Field;
use nabled_core::scalar::NabledReal;
use ndarray::Array1;
use ndarrow::NdarrowElement;
use num_complex::Complex64;

use super::{
    ArrowInteropError, complex64_matrix_from_owned, complex64_matrix_view,
    complex64_vector_from_owned, complex64_vector_view, fixed_size_list_from_owned,
    fixed_size_list_view, primitive_array_view,
};

/// Compute dot product directly from Arrow primitive arrays.
///
/// The inbound Arrow -> ndarray bridge is zero-copy.
///
/// # Errors
/// Returns an error when either array contains nulls, is empty, or lengths mismatch.
pub fn dot<T>(
    left: &PrimitiveArray<T>,
    right: &PrimitiveArray<T>,
) -> Result<T::Native, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let left_view = primitive_array_view(left)?;
    let right_view = primitive_array_view(right)?;
    Ok(crate::linalg::vector::dot_view(&left_view, &right_view)?)
}

/// Compute cosine similarity directly from Arrow primitive arrays.
///
/// The inbound Arrow -> ndarray bridge is zero-copy.
///
/// # Errors
/// Returns an error when either array contains nulls, is empty, lengths mismatch, or either
/// vector has zero norm.
pub fn cosine_similarity<T>(
    left: &PrimitiveArray<T>,
    right: &PrimitiveArray<T>,
) -> Result<T::Native, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let left_view = primitive_array_view(left)?;
    let right_view = primitive_array_view(right)?;
    Ok(crate::linalg::vector::cosine_similarity_view(&left_view, &right_view)?)
}

/// Compute L2 norm directly from an Arrow primitive array.
///
/// The inbound Arrow -> ndarray bridge is zero-copy.
///
/// # Errors
/// Returns an error when the array contains nulls or is empty.
pub fn l2_norm<T>(vector: &PrimitiveArray<T>) -> Result<T::Native, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let vector_view = primitive_array_view(vector)?;
    Ok(crate::linalg::vector::l2_norm_view(&vector_view)?)
}

/// Compute cosine distance directly from Arrow primitive arrays.
///
/// The inbound Arrow -> ndarray bridge is zero-copy.
///
/// # Errors
/// Returns an error when either array contains nulls, is empty, lengths mismatch, or either
/// vector has zero norm.
pub fn cosine_distance<T>(
    left: &PrimitiveArray<T>,
    right: &PrimitiveArray<T>,
) -> Result<T::Native, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let left_view = primitive_array_view(left)?;
    let right_view = primitive_array_view(right)?;
    Ok(crate::linalg::vector::cosine_distance_view(&left_view, &right_view)?)
}

/// Compute pairwise L2 distances between row vectors stored in Arrow dense matrices.
///
/// # Errors
/// Returns an error when inputs contain nulls, are empty, or dimensions mismatch.
pub fn pairwise_l2_distance<T>(
    left: &FixedSizeListArray,
    right: &FixedSizeListArray,
) -> Result<FixedSizeListArray, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let left_view = fixed_size_list_view::<T>(left)?;
    let right_view = fixed_size_list_view::<T>(right)?;
    let output = crate::linalg::vector::pairwise_l2_distance_view(&left_view, &right_view)?;
    fixed_size_list_from_owned::<T>(output)
}

/// Compute pairwise cosine similarity between row vectors stored in Arrow dense matrices.
///
/// # Errors
/// Returns an error when inputs contain nulls, are empty, dimensions mismatch, or row norms are
/// zero.
pub fn pairwise_cosine_similarity<T>(
    left: &FixedSizeListArray,
    right: &FixedSizeListArray,
) -> Result<FixedSizeListArray, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let left_view = fixed_size_list_view::<T>(left)?;
    let right_view = fixed_size_list_view::<T>(right)?;
    let output = crate::linalg::vector::pairwise_cosine_similarity_view(&left_view, &right_view)?;
    fixed_size_list_from_owned::<T>(output)
}

/// Compute pairwise cosine distance between row vectors stored in Arrow dense matrices.
///
/// # Errors
/// Returns an error when inputs contain nulls, are empty, dimensions mismatch, or row norms are
/// zero.
pub fn pairwise_cosine_distance<T>(
    left: &FixedSizeListArray,
    right: &FixedSizeListArray,
) -> Result<FixedSizeListArray, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let left_view = fixed_size_list_view::<T>(left)?;
    let right_view = fixed_size_list_view::<T>(right)?;
    let output = crate::linalg::vector::pairwise_cosine_distance_view(&left_view, &right_view)?;
    fixed_size_list_from_owned::<T>(output)
}

/// Compute row-wise dot products for Arrow dense matrices of identical shape.
///
/// # Errors
/// Returns an error when inputs contain nulls, are empty, or dimensions mismatch.
pub fn batched_dot<T>(
    left: &FixedSizeListArray,
    right: &FixedSizeListArray,
) -> Result<PrimitiveArray<T>, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let left_view = fixed_size_list_view::<T>(left)?;
    let right_view = fixed_size_list_view::<T>(right)?;
    let output = crate::linalg::vector::batched_dot_view(&left_view, &right_view)?;
    Ok(super::primitive_array_from_owned::<T>(output))
}

/// Compute row-wise L2 norms for Arrow dense matrices interpreted as `rows-of-vectors`.
///
/// # Errors
/// Returns an error when the input contains nulls or is empty.
pub fn batched_l2_norm<T>(rows: &FixedSizeListArray) -> Result<PrimitiveArray<T>, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let rows_view = fixed_size_list_view::<T>(rows)?;
    let output = crate::linalg::vector::batched_l2_norm_view(&rows_view)?;
    Ok(super::primitive_array_from_owned::<T>(output))
}

/// Compute row-wise cosine similarities for paired Arrow dense matrices.
///
/// # Errors
/// Returns an error when inputs contain nulls, are empty, dimensions mismatch, or row norms are
/// zero.
pub fn batched_cosine_similarity<T>(
    left: &FixedSizeListArray,
    right: &FixedSizeListArray,
) -> Result<PrimitiveArray<T>, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let left_view = fixed_size_list_view::<T>(left)?;
    let right_view = fixed_size_list_view::<T>(right)?;
    let output = crate::linalg::vector::batched_cosine_similarity_view(&left_view, &right_view)?;
    Ok(super::primitive_array_from_owned::<T>(output))
}

/// Compute row-wise cosine distances for paired Arrow dense matrices.
///
/// # Errors
/// Returns an error when inputs contain nulls, are empty, dimensions mismatch, or row norms are
/// zero.
pub fn batched_cosine_distance<T>(
    left: &FixedSizeListArray,
    right: &FixedSizeListArray,
) -> Result<PrimitiveArray<T>, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let left_view = fixed_size_list_view::<T>(left)?;
    let right_view = fixed_size_list_view::<T>(right)?;
    let output = crate::linalg::vector::batched_cosine_distance_view(&left_view, &right_view)?;
    Ok(super::primitive_array_from_owned::<T>(output))
}

/// Normalize each row in an Arrow dense matrix interpreted as `rows-of-vectors`.
///
/// # Errors
/// Returns an error when the input contains nulls or is empty.
pub fn batched_normalize<T>(
    rows: &FixedSizeListArray,
) -> Result<FixedSizeListArray, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let rows_view = fixed_size_list_view::<T>(rows)?;
    let output = crate::linalg::vector::batched_normalize_view(&rows_view)?;
    fixed_size_list_from_owned::<T>(output)
}

/// Compute Hermitian dot product directly from Arrow complex extension vectors.
///
/// # Errors
/// Returns an error when either complex vector contains nulls, is empty, or lengths mismatch.
pub fn dot_hermitian(
    left_field: &Field,
    left: &FixedSizeListArray,
    right_field: &Field,
    right: &FixedSizeListArray,
) -> Result<Complex64, ArrowInteropError> {
    let left_view = complex64_vector_view(left_field, left)?;
    let right_view = complex64_vector_view(right_field, right)?;
    Ok(crate::linalg::vector::dot_hermitian_view(&left_view, &right_view)?)
}

/// Compute complex-vector L2 norm directly from an Arrow complex extension vector.
///
/// # Errors
/// Returns an error when the vector contains nulls or is empty.
pub fn l2_norm_complex(
    field: &Field,
    vector: &FixedSizeListArray,
) -> Result<f64, ArrowInteropError> {
    let vector_view = complex64_vector_view(field, vector)?;
    Ok(crate::linalg::vector::l2_norm_complex_view(&vector_view)?)
}

/// Compute complex cosine similarity directly from Arrow complex extension vectors.
///
/// # Errors
/// Returns an error when inputs contain nulls, are empty, lengths mismatch, or norms vanish.
pub fn cosine_similarity_complex(
    left_field: &Field,
    left: &FixedSizeListArray,
    right_field: &Field,
    right: &FixedSizeListArray,
) -> Result<(Field, FixedSizeListArray), ArrowInteropError> {
    let left_view = complex64_vector_view(left_field, left)?;
    let right_view = complex64_vector_view(right_field, right)?;
    let value = crate::linalg::vector::cosine_similarity_complex_view(&left_view, &right_view)?;
    complex64_vector_from_owned("cosine_similarity", Array1::from_vec(vec![value]))
}

/// Compute row-wise Hermitian dot products for paired Arrow complex vector batches.
///
/// # Errors
/// Returns an error when inputs contain nulls, are empty, or dimensions mismatch.
pub fn batched_dot_hermitian(
    left: &FixedSizeListArray,
    right: &FixedSizeListArray,
) -> Result<(Field, FixedSizeListArray), ArrowInteropError> {
    let left_view = complex64_matrix_view(left)?;
    let right_view = complex64_matrix_view(right)?;
    let output = crate::linalg::vector::batched_dot_hermitian_view(&left_view, &right_view)?;
    complex64_vector_from_owned("batched_dot_hermitian", output)
}

/// Compute row-wise complex-vector L2 norms for Arrow complex vector batches.
///
/// # Errors
/// Returns an error when the input contains nulls or is empty.
pub fn batched_l2_norm_complex(
    rows: &FixedSizeListArray,
) -> Result<PrimitiveArray<Float64Type>, ArrowInteropError> {
    let rows_view = complex64_matrix_view(rows)?;
    let output = crate::linalg::vector::batched_l2_norm_complex_view(&rows_view)?;
    Ok(super::primitive_array_from_owned::<Float64Type>(output))
}

/// Compute row-wise complex cosine similarities for paired Arrow complex vector batches.
///
/// # Errors
/// Returns an error when inputs contain nulls, are empty, dimensions mismatch, or row norms are
/// zero.
pub fn batched_cosine_similarity_complex(
    left: &FixedSizeListArray,
    right: &FixedSizeListArray,
) -> Result<(Field, FixedSizeListArray), ArrowInteropError> {
    let left_view = complex64_matrix_view(left)?;
    let right_view = complex64_matrix_view(right)?;
    let output =
        crate::linalg::vector::batched_cosine_similarity_complex_view(&left_view, &right_view)?;
    complex64_vector_from_owned("batched_cosine_similarity", output)
}

/// Normalize each row in an Arrow complex dense matrix interpreted as `rows-of-vectors`.
///
/// # Errors
/// Returns an error when the input contains nulls or is empty.
pub fn batched_normalize_complex(
    rows: &FixedSizeListArray,
) -> Result<FixedSizeListArray, ArrowInteropError> {
    let rows_view = complex64_matrix_view(rows)?;
    let output = crate::linalg::vector::batched_normalize_complex_view(&rows_view)?;
    complex64_matrix_from_owned(output)
}
