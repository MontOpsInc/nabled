//! Arrow adapters for fixed-shape tensor workflows.

use arrow_array::FixedSizeListArray;
use arrow_array::types::{ArrowPrimitiveType, Float64Type};
use arrow_schema::Field;
use nabled_core::scalar::NabledReal;
use ndarray::Ix3;
use ndarrow::NdarrowElement;
use num_complex::Complex64;

use super::{
    ArrowInteropError, complex64_fixed_shape_tensor_from_owned, complex64_fixed_shape_tensor_viewd,
    complex64_matrix_from_owned, complex64_matrix_view, fixed_shape_tensor_from_owned,
    fixed_shape_tensor_viewd, fixed_size_list_from_owned, fixed_size_list_view,
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

fn complex64_fixed_shape_tensor_view3<'a>(
    field: &'a Field,
    array: &'a FixedSizeListArray,
) -> Result<ndarray::ArrayView3<'a, Complex64>, ArrowInteropError> {
    let view = complex64_fixed_shape_tensor_viewd(field, array)?;
    view.into_dimensionality::<Ix3>()
        .map_err(|error: ndarray::ShapeError| ArrowInteropError::InvalidShape(error.to_string()))
}

/// Arrow-facing rank-3 CP-ALS result paired with convergence diagnostics.
pub type ArrowCpAls3WithReport<T> =
    (crate::linalg::tensor::CpAls3Result<T>, crate::linalg::tensor::CpAlsReport<T>);

/// Arrow-facing N-D CP-ALS result paired with convergence diagnostics.
pub type ArrowCpAlsNdWithReport<T> =
    (crate::linalg::tensor::CpAlsNdResult<T>, crate::linalg::tensor::CpAlsReport<T>);

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

/// Flatten a rank-3 tensor batch into a dense matrix.
///
/// # Errors
/// Returns an error when the input is not rank-3, contains nulls, or flattening fails.
pub fn flatten_cubes<T>(
    field: &Field,
    array: &FixedSizeListArray,
) -> Result<FixedSizeListArray, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let cube_view = fixed_shape_tensor_view3::<T>(field, array)?;
    let output = crate::linalg::tensor::flatten_cubes_view(&cube_view)?;
    fixed_size_list_from_owned::<T>(output)
}

/// Sum over the last axis of a complex fixed-shape tensor batch.
///
/// # Errors
/// Returns an error when the tensor metadata is invalid, contains nulls, or the reduction fails.
pub fn sum_last_axis_complex(
    field: &Field,
    array: &FixedSizeListArray,
) -> Result<(Field, FixedSizeListArray), ArrowInteropError> {
    let tensor_view = complex64_fixed_shape_tensor_viewd(field, array)?;
    let output = crate::linalg::tensor::sum_last_axis_complex_view(&tensor_view)?;
    complex64_fixed_shape_tensor_from_owned(field.name(), output)
}

/// Compute L2 norms over the last axis of a complex fixed-shape tensor batch.
///
/// # Errors
/// Returns an error when the tensor metadata is invalid, contains nulls, or the reduction fails.
pub fn l2_norm_last_axis_complex(
    field: &Field,
    array: &FixedSizeListArray,
) -> Result<(Field, FixedSizeListArray), ArrowInteropError> {
    let tensor_view = complex64_fixed_shape_tensor_viewd(field, array)?;
    let output = crate::linalg::tensor::l2_norm_last_axis_complex_view(&tensor_view)?;
    fixed_shape_tensor_from_owned::<Float64Type>(field.name(), output)
}

/// Normalize a complex fixed-shape tensor batch over the last axis.
///
/// # Errors
/// Returns an error when the tensor metadata is invalid, contains nulls, or normalization fails.
pub fn normalize_last_axis_complex(
    field: &Field,
    array: &FixedSizeListArray,
) -> Result<(Field, FixedSizeListArray), ArrowInteropError> {
    let tensor_view = complex64_fixed_shape_tensor_viewd(field, array)?;
    let output = crate::linalg::tensor::normalize_last_axis_complex_view(&tensor_view)?;
    complex64_fixed_shape_tensor_from_owned(field.name(), output)
}

/// Compute batched complex dot products over the last axis of two fixed-shape tensor batches.
///
/// # Errors
/// Returns an error when either tensor is invalid, contains nulls, or shapes are incompatible.
pub fn batched_dot_last_axis_complex(
    left_field: &Field,
    left: &FixedSizeListArray,
    right_field: &Field,
    right: &FixedSizeListArray,
) -> Result<(Field, FixedSizeListArray), ArrowInteropError> {
    let left_view = complex64_fixed_shape_tensor_viewd(left_field, left)?;
    let right_view = complex64_fixed_shape_tensor_viewd(right_field, right)?;
    let output =
        crate::linalg::tensor::batched_dot_last_axis_complex_view(&left_view, &right_view)?;
    complex64_fixed_shape_tensor_from_owned(left_field.name(), output)
}

/// Permute the axes of a complex fixed-shape tensor batch.
///
/// # Errors
/// Returns an error when the tensor is invalid, contains nulls, or the permutation is invalid.
pub fn permute_axes_complex(
    field: &Field,
    array: &FixedSizeListArray,
    permutation: &[usize],
) -> Result<(Field, FixedSizeListArray), ArrowInteropError> {
    let tensor_view = complex64_fixed_shape_tensor_viewd(field, array)?;
    let output = crate::linalg::tensor::permute_axes_complex_view(&tensor_view, permutation)?;
    complex64_fixed_shape_tensor_from_owned(field.name(), output)
}

/// Contract two complex fixed-shape tensors along explicit axis sets.
///
/// # Errors
/// Returns an error when either tensor is invalid, contains nulls, or contraction axes mismatch.
pub fn contract_axes_complex(
    left_field: &Field,
    left: &FixedSizeListArray,
    right_field: &Field,
    right: &FixedSizeListArray,
    left_axes: &[usize],
    right_axes: &[usize],
) -> Result<(Field, FixedSizeListArray), ArrowInteropError> {
    let left_view = complex64_fixed_shape_tensor_viewd(left_field, left)?;
    let right_view = complex64_fixed_shape_tensor_viewd(right_field, right)?;
    let output = crate::linalg::tensor::contract_axes_complex_view(
        &left_view,
        &right_view,
        left_axes,
        right_axes,
    )?;
    complex64_fixed_shape_tensor_from_owned(left_field.name(), output)
}

/// Perform N-D batched complex matrix multiplication over the last two axes.
///
/// # Errors
/// Returns an error when either tensor is invalid, contains nulls, or dimensions mismatch.
pub fn batched_matmul_last_two_complex(
    left_field: &Field,
    left: &FixedSizeListArray,
    right_field: &Field,
    right: &FixedSizeListArray,
) -> Result<(Field, FixedSizeListArray), ArrowInteropError> {
    let left_view = complex64_fixed_shape_tensor_viewd(left_field, left)?;
    let right_view = complex64_fixed_shape_tensor_viewd(right_field, right)?;
    let output =
        crate::linalg::tensor::batched_matmul_last_two_complex_view(&left_view, &right_view)?;
    complex64_fixed_shape_tensor_from_owned(left_field.name(), output)
}

/// Compute batched complex cube-matrix vector products from Arrow inputs.
///
/// # Errors
/// Returns an error when inputs are invalid, contain nulls, or dimensions mismatch.
pub fn cube_matvec_complex(
    cube_field: &Field,
    cube: &FixedSizeListArray,
    vectors: &FixedSizeListArray,
) -> Result<FixedSizeListArray, ArrowInteropError> {
    let cube_view = complex64_fixed_shape_tensor_view3(cube_field, cube)?;
    let vectors_view = complex64_matrix_view(vectors)?;
    let output = crate::linalg::tensor::cube_matvec_complex_view(&cube_view, &vectors_view)?;
    complex64_matrix_from_owned(output)
}

/// Compute batched complex cube matrix-matrix products from Arrow inputs.
///
/// # Errors
/// Returns an error when inputs are invalid, contain nulls, or dimensions mismatch.
pub fn cube_matmat_complex(
    left_field: &Field,
    left: &FixedSizeListArray,
    right_field: &Field,
    right: &FixedSizeListArray,
) -> Result<(Field, FixedSizeListArray), ArrowInteropError> {
    let left_view = complex64_fixed_shape_tensor_view3(left_field, left)?;
    let right_view = complex64_fixed_shape_tensor_view3(right_field, right)?;
    let output = crate::linalg::tensor::cube_matmat_complex_view(&left_view, &right_view)?;
    complex64_fixed_shape_tensor_from_owned(left_field.name(), output.into_dyn())
}

/// Evaluate two-operand Einstein summation over real fixed-shape tensors.
///
/// # Errors
/// Returns an error when expression syntax is invalid or the tensor shapes mismatch.
pub fn einsum<T>(
    expression: &str,
    left_field: &Field,
    left: &FixedSizeListArray,
    right_field: &Field,
    right: &FixedSizeListArray,
) -> Result<(Field, FixedSizeListArray), ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let left_view = fixed_shape_tensor_viewd::<T>(left_field, left)?;
    let right_view = fixed_shape_tensor_viewd::<T>(right_field, right)?;
    let output = crate::linalg::tensor::einsum_view(expression, &left_view, &right_view)?;
    fixed_shape_tensor_from_owned::<T>(left_field.name(), output)
}

/// Evaluate two-operand Einstein summation over complex fixed-shape tensors.
///
/// # Errors
/// Returns an error when expression syntax is invalid or the tensor shapes mismatch.
pub fn einsum_complex(
    expression: &str,
    left_field: &Field,
    left: &FixedSizeListArray,
    right_field: &Field,
    right: &FixedSizeListArray,
) -> Result<(Field, FixedSizeListArray), ArrowInteropError> {
    let left_view = complex64_fixed_shape_tensor_viewd(left_field, left)?;
    let right_view = complex64_fixed_shape_tensor_viewd(right_field, right)?;
    let output = crate::linalg::tensor::einsum_complex_view(expression, &left_view, &right_view)?;
    complex64_fixed_shape_tensor_from_owned(left_field.name(), output)
}

/// Compute rank-`R` CP decomposition for a rank-3 tensor from Arrow fixed-shape tensor input.
///
/// # Errors
/// Returns an error when the tensor is not rank-3, contains nulls, or ALS fails.
pub fn cp_als3<T>(
    field: &Field,
    array: &FixedSizeListArray,
    rank: usize,
    config: &crate::linalg::tensor::CpAlsConfig<T::Native>,
) -> Result<crate::linalg::tensor::CpAls3Result<T::Native>, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: crate::linalg::tensor::CpAlsScalar + NdarrowElement,
{
    let cube_view = fixed_shape_tensor_view3::<T>(field, array)?;
    Ok(crate::linalg::tensor::cp_als3_view(&cube_view, rank, config)?)
}

/// Compute rank-`R` CP decomposition plus diagnostics for a rank-3 Arrow tensor input.
///
/// # Errors
/// Returns an error when the tensor is not rank-3, contains nulls, or ALS fails.
pub fn cp_als3_with_report<T>(
    field: &Field,
    array: &FixedSizeListArray,
    rank: usize,
    config: &crate::linalg::tensor::CpAlsConfig<T::Native>,
) -> Result<ArrowCpAls3WithReport<T::Native>, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: crate::linalg::tensor::CpAlsScalar + NdarrowElement,
{
    let cube_view = fixed_shape_tensor_view3::<T>(field, array)?;
    Ok(crate::linalg::tensor::cp_als3_view_with_report(&cube_view, rank, config)?)
}

/// Compute reconstruction diagnostics for a rank-3 CP decomposition against an Arrow tensor.
///
/// # Errors
/// Returns an error when the tensor is not rank-3, contains nulls, or dimensions mismatch.
pub fn cp_als3_diagnostics<T>(
    field: &Field,
    array: &FixedSizeListArray,
    result: &crate::linalg::tensor::CpAls3Result<T::Native>,
) -> Result<crate::linalg::tensor::CpErrorMetrics<T::Native>, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let cube_view = fixed_shape_tensor_view3::<T>(field, array)?;
    Ok(crate::linalg::tensor::cp_als3_diagnostics_view(&cube_view, result)?)
}

/// Reconstruct a rank-3 tensor from CP factors into Arrow fixed-shape tensor form.
///
/// # Errors
/// Returns an error when factor dimensions are incompatible.
pub fn cp_als3_reconstruct<T>(
    field_name: &str,
    result: &crate::linalg::tensor::CpAls3Result<T::Native>,
) -> Result<(Field, FixedSizeListArray), ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let output = crate::linalg::tensor::cp_als3_reconstruct(result)?;
    fixed_shape_tensor_from_owned::<T>(field_name, output.into_dyn())
}

/// Compute rank-`R` CP decomposition for an `N`-D tensor from Arrow fixed-shape tensor input.
///
/// # Errors
/// Returns an error when the tensor is invalid, contains nulls, or ALS fails.
pub fn cp_als_nd<T>(
    field: &Field,
    array: &FixedSizeListArray,
    rank: usize,
    config: &crate::linalg::tensor::CpAlsConfig<T::Native>,
) -> Result<crate::linalg::tensor::CpAlsNdResult<T::Native>, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: crate::linalg::tensor::CpAlsScalar + NdarrowElement,
{
    let tensor_view = fixed_shape_tensor_viewd::<T>(field, array)?;
    Ok(crate::linalg::tensor::cp_als_nd_view(&tensor_view, rank, config)?)
}

/// Compute rank-`R` CP decomposition plus diagnostics for an `N`-D Arrow tensor input.
///
/// # Errors
/// Returns an error when the tensor is invalid, contains nulls, or ALS fails.
pub fn cp_als_nd_with_report<T>(
    field: &Field,
    array: &FixedSizeListArray,
    rank: usize,
    config: &crate::linalg::tensor::CpAlsConfig<T::Native>,
) -> Result<ArrowCpAlsNdWithReport<T::Native>, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: crate::linalg::tensor::CpAlsScalar + NdarrowElement,
{
    let tensor_view = fixed_shape_tensor_viewd::<T>(field, array)?;
    Ok(crate::linalg::tensor::cp_als_nd_view_with_report(&tensor_view, rank, config)?)
}

/// Compute reconstruction diagnostics for an `N`-D CP decomposition against an Arrow tensor.
///
/// # Errors
/// Returns an error when the tensor is invalid, contains nulls, or dimensions mismatch.
pub fn cp_als_nd_diagnostics<T>(
    field: &Field,
    array: &FixedSizeListArray,
    result: &crate::linalg::tensor::CpAlsNdResult<T::Native>,
) -> Result<crate::linalg::tensor::CpErrorMetrics<T::Native>, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let tensor_view = fixed_shape_tensor_viewd::<T>(field, array)?;
    Ok(crate::linalg::tensor::cp_als_nd_diagnostics_view(&tensor_view, result)?)
}

/// Reconstruct an `N`-D tensor from CP factors into Arrow fixed-shape tensor form.
///
/// # Errors
/// Returns an error when factor dimensions are incompatible.
pub fn cp_als_nd_reconstruct<T>(
    field_name: &str,
    result: &crate::linalg::tensor::CpAlsNdResult<T::Native>,
) -> Result<(Field, FixedSizeListArray), ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let output = crate::linalg::tensor::cp_als_nd_reconstruct(result)?;
    fixed_shape_tensor_from_owned::<T>(field_name, output)
}

/// Compute rank-truncated `N`-D HOSVD from Arrow fixed-shape tensor input.
///
/// # Errors
/// Returns an error when the tensor is invalid, contains nulls, or factorization fails.
pub fn hosvd_nd<T>(
    field: &Field,
    array: &FixedSizeListArray,
    ranks: &[usize],
) -> Result<crate::linalg::tensor::HosvdNdResult<T::Native>, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: crate::linalg::tensor::HosvdNdScalar + NdarrowElement,
{
    let tensor_view = fixed_shape_tensor_viewd::<T>(field, array)?;
    Ok(crate::linalg::tensor::hosvd_nd_view(&tensor_view, ranks)?)
}

/// Compute rank-truncated `N`-D Tucker decomposition via HOOI from Arrow tensor input.
///
/// # Errors
/// Returns an error when the tensor is invalid, contains nulls, or factorization fails.
pub fn hooi_nd<T>(
    field: &Field,
    array: &FixedSizeListArray,
    ranks: &[usize],
    config: &crate::linalg::tensor::HooiConfig<T::Native>,
) -> Result<crate::linalg::tensor::HosvdNdResult<T::Native>, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: crate::linalg::tensor::HooiNdScalar + NdarrowElement,
{
    let tensor_view = fixed_shape_tensor_viewd::<T>(field, array)?;
    Ok(crate::linalg::tensor::hooi_nd_view(&tensor_view, ranks, config)?)
}

/// Project an Arrow tensor into a Tucker core using ndarray-native factor matrices.
///
/// # Errors
/// Returns an error when tensor/factor dimensions are incompatible.
pub fn tucker_project<T>(
    field: &Field,
    array: &FixedSizeListArray,
    factors: &[ndarray::Array2<T::Native>],
) -> Result<(Field, FixedSizeListArray), ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let tensor_view = fixed_shape_tensor_viewd::<T>(field, array)?;
    let output = crate::linalg::tensor::tucker_project_view(&tensor_view, factors)?;
    fixed_shape_tensor_from_owned::<T>(field.name(), output)
}

/// Expand a Tucker core from Arrow into the original tensor space.
///
/// # Errors
/// Returns an error when core/factor dimensions are incompatible.
pub fn tucker_expand<T>(
    field: &Field,
    array: &FixedSizeListArray,
    factors: &[ndarray::Array2<T::Native>],
) -> Result<(Field, FixedSizeListArray), ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let tensor_view = fixed_shape_tensor_viewd::<T>(field, array)?;
    let output = crate::linalg::tensor::tucker_expand_view(&tensor_view, factors)?;
    fixed_shape_tensor_from_owned::<T>(field.name(), output)
}

/// Reconstruct an `N`-D tensor from an ndarray-native HOSVD/Tucker result.
///
/// # Errors
/// Returns an error when factor dimensions are incompatible.
pub fn hosvd_nd_reconstruct<T>(
    field_name: &str,
    result: &crate::linalg::tensor::HosvdNdResult<T::Native>,
) -> Result<(Field, FixedSizeListArray), ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let output = crate::linalg::tensor::hosvd_nd_reconstruct(result)?;
    fixed_shape_tensor_from_owned::<T>(field_name, output)
}

/// Compute Tensor-Train decomposition from Arrow fixed-shape tensor input.
///
/// # Errors
/// Returns an error when the tensor is invalid, contains nulls, or decomposition fails.
pub fn tt_svd<T>(
    field: &Field,
    array: &FixedSizeListArray,
    config: &crate::linalg::tensor::TtSvdConfig<T::Native>,
) -> Result<crate::linalg::tensor::TensorTrainResult<T::Native>, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: crate::linalg::tensor::TtSvdScalar + NdarrowElement,
{
    let tensor_view = fixed_shape_tensor_viewd::<T>(field, array)?;
    Ok(crate::linalg::tensor::tt_svd_view(&tensor_view, config)?)
}

/// Left-orthogonalize a Tensor-Train result.
///
/// # Errors
/// Returns an error when TT core dimensions are incompatible.
pub fn tt_orthogonalize_left<T: crate::linalg::tensor::TtSvdScalar>(
    result: &crate::linalg::tensor::TensorTrainResult<T>,
) -> Result<crate::linalg::tensor::TensorTrainResult<T>, ArrowInteropError> {
    Ok(crate::linalg::tensor::tt_orthogonalize_left(result)?)
}

/// Right-orthogonalize a Tensor-Train result.
///
/// # Errors
/// Returns an error when TT core dimensions are incompatible.
pub fn tt_orthogonalize_right<T: crate::linalg::tensor::TtSvdScalar>(
    result: &crate::linalg::tensor::TensorTrainResult<T>,
) -> Result<crate::linalg::tensor::TensorTrainResult<T>, ArrowInteropError> {
    Ok(crate::linalg::tensor::tt_orthogonalize_right(result)?)
}

/// Round/compress a Tensor-Train result.
///
/// # Errors
/// Returns an error when TT core dimensions or configuration are invalid.
pub fn tt_round<T: crate::linalg::tensor::TtSvdScalar>(
    result: &crate::linalg::tensor::TensorTrainResult<T>,
    config: &crate::linalg::tensor::TtRoundConfig<T>,
) -> Result<crate::linalg::tensor::TensorTrainResult<T>, ArrowInteropError> {
    Ok(crate::linalg::tensor::tt_round(result, config)?)
}

/// Compute inner product of two Tensor-Train tensors.
///
/// # Errors
/// Returns an error when TT core dimensions are incompatible or shapes mismatch.
pub fn tt_inner<T: NabledReal>(
    left: &crate::linalg::tensor::TensorTrainResult<T>,
    right: &crate::linalg::tensor::TensorTrainResult<T>,
) -> Result<T, ArrowInteropError> {
    Ok(crate::linalg::tensor::tt_inner(left, right)?)
}

/// Compute Frobenius norm of a Tensor-Train tensor.
///
/// # Errors
/// Returns an error when TT core dimensions are incompatible.
pub fn tt_norm<T: NabledReal>(
    result: &crate::linalg::tensor::TensorTrainResult<T>,
) -> Result<T, ArrowInteropError> {
    Ok(crate::linalg::tensor::tt_norm(result)?)
}

/// Add two Tensor-Train tensors.
///
/// # Errors
/// Returns an error when TT core dimensions are incompatible or shapes mismatch.
pub fn tt_add<T: NabledReal>(
    left: &crate::linalg::tensor::TensorTrainResult<T>,
    right: &crate::linalg::tensor::TensorTrainResult<T>,
) -> Result<crate::linalg::tensor::TensorTrainResult<T>, ArrowInteropError> {
    Ok(crate::linalg::tensor::tt_add(left, right)?)
}

/// Compute elementwise Hadamard product of two Tensor-Train tensors.
///
/// # Errors
/// Returns an error when TT core dimensions are incompatible or shapes mismatch.
pub fn tt_hadamard<T: NabledReal>(
    left: &crate::linalg::tensor::TensorTrainResult<T>,
    right: &crate::linalg::tensor::TensorTrainResult<T>,
) -> Result<crate::linalg::tensor::TensorTrainResult<T>, ArrowInteropError> {
    Ok(crate::linalg::tensor::tt_hadamard(left, right)?)
}

/// Compute Hadamard product followed by TT rank-rounding.
///
/// # Errors
/// Returns an error when TT core dimensions/configuration are invalid.
pub fn tt_hadamard_round<T: crate::linalg::tensor::TtSvdScalar>(
    left: &crate::linalg::tensor::TensorTrainResult<T>,
    right: &crate::linalg::tensor::TensorTrainResult<T>,
    config: &crate::linalg::tensor::TtRoundConfig<T>,
) -> Result<crate::linalg::tensor::TensorTrainResult<T>, ArrowInteropError> {
    Ok(crate::linalg::tensor::tt_hadamard_round(left, right, config)?)
}

/// Reconstruct an `N`-D tensor from Tensor-Train cores into Arrow fixed-shape tensor form.
///
/// # Errors
/// Returns an error when TT core dimensions are incompatible.
pub fn tt_svd_reconstruct<T>(
    field_name: &str,
    result: &crate::linalg::tensor::TensorTrainResult<T::Native>,
) -> Result<(Field, FixedSizeListArray), ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let output = crate::linalg::tensor::tt_svd_reconstruct(result)?;
    fixed_shape_tensor_from_owned::<T>(field_name, output)
}
