//! Arrow adapters for dense matrix primitives.

use arrow_array::types::ArrowPrimitiveType;
use arrow_array::{FixedSizeListArray, PrimitiveArray};
use nabled_core::scalar::NabledReal;
use ndarrow::NdarrowElement;

use super::{
    ArrowInteropError, fixed_size_list_from_owned, fixed_size_list_view,
    primitive_array_from_owned, primitive_array_view,
};

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
