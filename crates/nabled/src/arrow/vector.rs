//! Arrow adapters for vector-first primitives.

use arrow_array::types::ArrowPrimitiveType;
use arrow_array::{FixedSizeListArray, PrimitiveArray};
use nabled_core::scalar::NabledReal;
use ndarrow::NdarrowElement;

use super::{
    ArrowInteropError, fixed_size_list_from_owned, fixed_size_list_view, primitive_array_view,
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
