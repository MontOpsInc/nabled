//! Arrow adapters for fixed-shape tensor workflows.

use arrow_array::FixedSizeListArray;
use arrow_array::types::ArrowPrimitiveType;
use arrow_schema::Field;
use nabled_core::scalar::NabledReal;
use ndarray::Ix3;
use ndarrow::NdarrowElement;

use super::{
    ArrowInteropError, fixed_shape_tensor_from_owned, fixed_shape_tensor_viewd,
    fixed_size_list_from_owned, fixed_size_list_view,
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

/// Sum over the last axis of a canonical `arrow.fixed_shape_tensor` batch.
///
/// `field` and `array` must come from the same fixed-shape tensor column. The returned field keeps
/// the input field name and carries updated fixed-shape tensor metadata.
///
/// # Errors
/// Returns an error when the input field/array do not represent a valid fixed-shape tensor batch,
/// contain nulls, or the tensor reduction fails.
pub fn sum_last_axis<T>(
    field: &Field,
    array: &FixedSizeListArray,
) -> Result<(Field, FixedSizeListArray), ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement + Default,
{
    let tensor_view = fixed_shape_tensor_viewd::<T>(field, array)?;
    let output = crate::linalg::tensor::sum_last_axis_view(&tensor_view)?;
    fixed_shape_tensor_from_owned::<T>(field.name(), output)
}

/// Compute L2 norms over the last axis of a fixed-shape tensor batch.
///
/// # Errors
/// Returns an error when the input field/array do not represent a valid fixed-shape tensor batch,
/// contain nulls, or the tensor reduction fails.
pub fn l2_norm_last_axis<T>(
    field: &Field,
    array: &FixedSizeListArray,
) -> Result<(Field, FixedSizeListArray), ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement + Default,
{
    let tensor_view = fixed_shape_tensor_viewd::<T>(field, array)?;
    let output = crate::linalg::tensor::l2_norm_last_axis_view(&tensor_view)?;
    fixed_shape_tensor_from_owned::<T>(field.name(), output)
}

/// Normalize a fixed-shape tensor batch over the last axis.
///
/// # Errors
/// Returns an error when the input field/array do not represent a valid fixed-shape tensor batch,
/// contain nulls, or the tensor normalization fails.
pub fn normalize_last_axis<T>(
    field: &Field,
    array: &FixedSizeListArray,
) -> Result<(Field, FixedSizeListArray), ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement + Default,
{
    let tensor_view = fixed_shape_tensor_viewd::<T>(field, array)?;
    let output = crate::linalg::tensor::normalize_last_axis_view(&tensor_view)?;
    fixed_shape_tensor_from_owned::<T>(field.name(), output)
}

/// Compute batched dot products over the last axis of two fixed-shape tensor batches.
///
/// # Errors
/// Returns an error when inputs do not represent valid fixed-shape tensors, contain nulls, or
/// dimensions are incompatible.
pub fn batched_dot_last_axis<T>(
    left_field: &Field,
    left: &FixedSizeListArray,
    right_field: &Field,
    right: &FixedSizeListArray,
) -> Result<(Field, FixedSizeListArray), ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement + Default,
{
    let left_view = fixed_shape_tensor_viewd::<T>(left_field, left)?;
    let right_view = fixed_shape_tensor_viewd::<T>(right_field, right)?;
    let output = crate::linalg::tensor::batched_dot_last_axis_view(&left_view, &right_view)?;
    fixed_shape_tensor_from_owned::<T>(left_field.name(), output)
}

/// Permute the axes of a fixed-shape tensor batch.
///
/// # Errors
/// Returns an error when the input does not represent a valid fixed-shape tensor batch, contains
/// nulls, or the permutation is invalid.
pub fn permute_axes<T>(
    field: &Field,
    array: &FixedSizeListArray,
    permutation: &[usize],
) -> Result<(Field, FixedSizeListArray), ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement + Default,
{
    let tensor_view = fixed_shape_tensor_viewd::<T>(field, array)?;
    let output = crate::linalg::tensor::permute_axes_view(&tensor_view, permutation)?;
    fixed_shape_tensor_from_owned::<T>(field.name(), output)
}

/// Contract two fixed-shape tensors along explicit axis sets.
///
/// # Errors
/// Returns an error when either input is invalid, contains nulls, or contraction axes are
/// incompatible.
pub fn contract_axes<T>(
    left_field: &Field,
    left: &FixedSizeListArray,
    right_field: &Field,
    right: &FixedSizeListArray,
    left_axes: &[usize],
    right_axes: &[usize],
) -> Result<(Field, FixedSizeListArray), ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement + Default,
{
    let left_view = fixed_shape_tensor_viewd::<T>(left_field, left)?;
    let right_view = fixed_shape_tensor_viewd::<T>(right_field, right)?;
    let output =
        crate::linalg::tensor::contract_axes_view(&left_view, &right_view, left_axes, right_axes)?;
    fixed_shape_tensor_from_owned::<T>(left_field.name(), output)
}

/// Perform N-D batched matrix multiplication over the last two axes of fixed-shape tensors.
///
/// # Errors
/// Returns an error when either input is invalid, contains nulls, or dimensions are incompatible.
pub fn batched_matmul_last_two<T>(
    left_field: &Field,
    left: &FixedSizeListArray,
    right_field: &Field,
    right: &FixedSizeListArray,
) -> Result<(Field, FixedSizeListArray), ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement + Default,
{
    let left_view = fixed_shape_tensor_viewd::<T>(left_field, left)?;
    let right_view = fixed_shape_tensor_viewd::<T>(right_field, right)?;
    let output = crate::linalg::tensor::batched_matmul_last_two_view(&left_view, &right_view)?;
    fixed_shape_tensor_from_owned::<T>(left_field.name(), output)
}

/// Compute batched cube-matrix vector products from fixed-shape tensor / Arrow dense inputs.
///
/// `cube` is interpreted as rank-3 `(batch, rows, cols)` tensor data and `vectors` as `(batch,
/// cols)` dense matrix data.
///
/// # Errors
/// Returns an error when inputs are invalid, contain nulls, or dimensions are incompatible.
pub fn cube_matvec<T>(
    cube_field: &Field,
    cube: &FixedSizeListArray,
    vectors: &FixedSizeListArray,
) -> Result<FixedSizeListArray, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let cube_view = fixed_shape_tensor_view3::<T>(cube_field, cube)?;
    let vectors_view = fixed_size_list_view::<T>(vectors)?;
    let output = crate::linalg::tensor::cube_matvec_view(&cube_view, &vectors_view)?;
    fixed_size_list_from_owned::<T>(output)
}

/// Compute batched cube matrix-matrix products from two rank-3 fixed-shape tensors.
///
/// # Errors
/// Returns an error when inputs are invalid, contain nulls, or dimensions are incompatible.
pub fn cube_matmat<T>(
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
    let output = crate::linalg::tensor::cube_matmat_view(&left_view, &right_view)?;
    fixed_shape_tensor_from_owned::<T>(left_field.name(), output.into_dyn())
}
