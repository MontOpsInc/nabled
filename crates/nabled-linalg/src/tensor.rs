//! Tensor and cube primitives over ndarray higher-rank arrays.

use std::fmt;
use std::ops::{AddAssign, Mul};

use nabled_core::scalar::NabledReal;
use ndarray::{
    Array2, Array3, ArrayD, ArrayView2, ArrayView3, ArrayViewD, ArrayViewMut2, ArrayViewMut3, Axis,
    IxDyn, s,
};
use num_complex::Complex64;

use crate::accelerator::backends::AcceleratorError;
use crate::accelerator::dispatch::{
    tensor_batched_matmul_last_two_cpu, tensor_contract_axes_cpu, tensor_sum_last_axis_cpu,
};
use crate::svd;

/// Error type for tensor/cube operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorError {
    /// Input tensor/matrix is empty.
    EmptyInput,
    /// Input dimensions are incompatible.
    DimensionMismatch,
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorError::EmptyInput => write!(f, "input cannot be empty"),
            TensorError::DimensionMismatch => write!(f, "input dimensions are incompatible"),
        }
    }
}

impl std::error::Error for TensorError {}

fn map_accelerator_error_to_tensor(_error: AcceleratorError) -> TensorError {
    TensorError::DimensionMismatch
}

/// HOSVD decomposition result for rank-3 real tensors.
#[derive(Debug, Clone)]
pub struct Hosvd3Result<T: NabledReal = f64> {
    /// Core tensor with shape `(r0, r1, r2)`.
    pub core: Array3<T>,
    /// Mode-0 factor matrix `(i0, r0)`.
    pub u0:   Array2<T>,
    /// Mode-1 factor matrix `(i1, r1)`.
    pub u1:   Array2<T>,
    /// Mode-2 factor matrix `(i2, r2)`.
    pub u2:   Array2<T>,
}

type EinsumOperands = (Vec<char>, Vec<char>, Vec<char>);

fn validate_cube_non_empty<T>(cube: &ArrayView3<'_, T>) -> Result<(), TensorError> {
    if cube.is_empty() {
        return Err(TensorError::EmptyInput);
    }
    Ok(())
}

fn validate_matrix_non_empty<T>(matrix: &ArrayView2<'_, T>) -> Result<(), TensorError> {
    if matrix.is_empty() {
        return Err(TensorError::EmptyInput);
    }
    Ok(())
}

fn validate_cube_non_empty_complex(cube: &ArrayView3<'_, Complex64>) -> Result<(), TensorError> {
    if cube.is_empty() {
        return Err(TensorError::EmptyInput);
    }
    Ok(())
}

fn validate_matrix_non_empty_complex(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<(), TensorError> {
    if matrix.is_empty() {
        return Err(TensorError::EmptyInput);
    }
    Ok(())
}

fn validate_tensor_nd_non_empty<T>(tensor: &ArrayViewD<'_, T>) -> Result<(), TensorError> {
    if tensor.is_empty() {
        return Err(TensorError::EmptyInput);
    }
    if tensor.ndim() == 0 {
        return Err(TensorError::DimensionMismatch);
    }
    Ok(())
}

fn validate_tensor_nd_non_empty_complex(
    tensor: &ArrayViewD<'_, Complex64>,
) -> Result<(), TensorError> {
    if tensor.is_empty() {
        return Err(TensorError::EmptyInput);
    }
    if tensor.ndim() == 0 {
        return Err(TensorError::DimensionMismatch);
    }
    Ok(())
}

fn validate_permutation(ndim: usize, permutation: &[usize]) -> bool {
    if permutation.len() != ndim {
        return false;
    }

    let mut seen = vec![false; ndim];
    for &axis in permutation {
        if axis >= ndim || seen[axis] {
            return false;
        }
        seen[axis] = true;
    }

    true
}

fn validate_axes(ndim: usize, axes: &[usize]) -> bool {
    let mut seen = vec![false; ndim];
    for &axis in axes {
        if axis >= ndim || seen[axis] {
            return false;
        }
        seen[axis] = true;
    }
    true
}

fn uncontracted_axes(ndim: usize, contracted: &[usize]) -> Vec<usize> {
    let mut is_contracted = vec![false; ndim];
    for &axis in contracted {
        is_contracted[axis] = true;
    }

    (0..ndim).filter(|axis| !is_contracted[*axis]).collect()
}

fn shape_product(shape: &[usize]) -> usize { shape.iter().copied().product::<usize>().max(1) }

pub(crate) fn contract_view_into_impl<T>(
    left: &ArrayViewD<'_, T>,
    right: &ArrayViewD<'_, T>,
    left_axes: &[usize],
    right_axes: &[usize],
    output: &mut ArrayD<T>,
) -> Result<(), TensorError>
where
    T: Copy + Default + AddAssign + Mul<Output = T>,
{
    if left_axes.len() != right_axes.len() {
        return Err(TensorError::DimensionMismatch);
    }
    if !validate_axes(left.ndim(), left_axes) || !validate_axes(right.ndim(), right_axes) {
        return Err(TensorError::DimensionMismatch);
    }

    for (&left_axis, &right_axis) in left_axes.iter().zip(right_axes.iter()) {
        if left.shape()[left_axis] != right.shape()[right_axis] {
            return Err(TensorError::DimensionMismatch);
        }
    }

    let left_free_axes = uncontracted_axes(left.ndim(), left_axes);
    let right_free_axes = uncontracted_axes(right.ndim(), right_axes);

    let mut expected_shape =
        left_free_axes.iter().map(|axis| left.shape()[*axis]).collect::<Vec<_>>();
    expected_shape.extend(right_free_axes.iter().map(|axis| right.shape()[*axis]));
    if output.shape() != expected_shape.as_slice() {
        return Err(TensorError::DimensionMismatch);
    }

    let mut left_order = left_free_axes.clone();
    left_order.extend_from_slice(left_axes);
    let mut right_order = right_axes.to_vec();
    right_order.extend_from_slice(&right_free_axes);

    let left_outer =
        shape_product(&left_free_axes.iter().map(|axis| left.shape()[*axis]).collect::<Vec<_>>());
    let right_outer =
        shape_product(&right_free_axes.iter().map(|axis| right.shape()[*axis]).collect::<Vec<_>>());
    let contract_size =
        shape_product(&left_axes.iter().map(|axis| left.shape()[*axis]).collect::<Vec<_>>());

    let left_width = if left_axes.is_empty() { 1 } else { contract_size };
    let right_height = if right_axes.is_empty() { 1 } else { contract_size };

    let left_permuted = left.view().permuted_axes(left_order).to_owned();
    let right_permuted = right.view().permuted_axes(right_order).to_owned();
    let left_standard = left_permuted.as_standard_layout().to_owned();
    let right_standard = right_permuted.as_standard_layout().to_owned();

    let left_2d = left_standard
        .view()
        .into_shape_with_order((left_outer, left_width))
        .map_err(|_| TensorError::DimensionMismatch)?;
    let right_2d = right_standard
        .view()
        .into_shape_with_order((right_height, right_outer))
        .map_err(|_| TensorError::DimensionMismatch)?;

    let mut output_2d = output
        .view_mut()
        .into_shape_with_order((left_outer, right_outer))
        .map_err(|_| TensorError::DimensionMismatch)?;
    output_2d.fill(T::default());

    for i in 0..left_outer {
        for k in 0..left_width {
            let lhs = left_2d[[i, k]];
            for j in 0..right_outer {
                output_2d[[i, j]] += lhs * right_2d[[k, j]];
            }
        }
    }

    Ok(())
}

pub(crate) fn batched_matmul_last_two_view_into_impl<T>(
    left: &ArrayViewD<'_, T>,
    right: &ArrayViewD<'_, T>,
    output: &mut ArrayD<T>,
) -> Result<(), TensorError>
where
    T: Copy + Default + AddAssign + Mul<Output = T>,
{
    if left.ndim() < 2 || right.ndim() < 2 || left.ndim() != right.ndim() {
        return Err(TensorError::DimensionMismatch);
    }

    let batch_ndim = left.ndim() - 2;
    if left.shape()[..batch_ndim] != right.shape()[..batch_ndim] {
        return Err(TensorError::DimensionMismatch);
    }

    let rows = left.shape()[left.ndim() - 2];
    let inner = left.shape()[left.ndim() - 1];
    let inner_rhs = right.shape()[right.ndim() - 2];
    let cols = right.shape()[right.ndim() - 1];
    if inner != inner_rhs {
        return Err(TensorError::DimensionMismatch);
    }

    let mut expected_shape = left.shape()[..batch_ndim].to_vec();
    expected_shape.push(rows);
    expected_shape.push(cols);
    if output.shape() != expected_shape.as_slice() {
        return Err(TensorError::DimensionMismatch);
    }

    let batches = shape_product(&left.shape()[..batch_ndim]);
    let left_standard = left.as_standard_layout().to_owned();
    let right_standard = right.as_standard_layout().to_owned();
    let left_3d = left_standard
        .view()
        .into_shape_with_order((batches, rows, inner))
        .map_err(|_| TensorError::DimensionMismatch)?;
    let right_3d = right_standard
        .view()
        .into_shape_with_order((batches, inner, cols))
        .map_err(|_| TensorError::DimensionMismatch)?;
    let mut output_3d = output
        .view_mut()
        .into_shape_with_order((batches, rows, cols))
        .map_err(|_| TensorError::DimensionMismatch)?;
    output_3d.fill(T::default());

    for batch in 0..batches {
        for row in 0..rows {
            for k in 0..inner {
                let lhs = left_3d[[batch, row, k]];
                for col in 0..cols {
                    output_3d[[batch, row, col]] += lhs * right_3d[[batch, k, col]];
                }
            }
        }
    }

    Ok(())
}

/// Compute batched cube-matrix vector products.
///
/// Inputs are `cube=(batch, rows, cols)` and `vectors=(batch, cols)`.
/// Output is `(batch, rows)`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn cube_matvec<T: NabledReal>(
    cube: &Array3<T>,
    vectors: &Array2<T>,
) -> Result<Array2<T>, TensorError> {
    let mut output = Array2::<T>::zeros((cube.dim().0, cube.dim().1));
    cube_matvec_view_into(&cube.view(), &vectors.view(), output.view_mut())?;
    Ok(output)
}

/// Compute batched cube-matrix vector products from views.
///
/// Inputs are `cube=(batch, rows, cols)` and `vectors=(batch, cols)`.
/// Output is `(batch, rows)`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn cube_matvec_view<T: NabledReal>(
    cube: &ArrayView3<'_, T>,
    vectors: &ArrayView2<'_, T>,
) -> Result<Array2<T>, TensorError> {
    let mut output = Array2::<T>::zeros((cube.dim().0, cube.dim().1));
    cube_matvec_view_into(cube, vectors, output.view_mut())?;
    Ok(output)
}

/// Compute batched cube-matrix vector products into `output`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn cube_matvec_into<T: NabledReal>(
    cube: &Array3<T>,
    vectors: &Array2<T>,
    output: &mut Array2<T>,
) -> Result<(), TensorError> {
    cube_matvec_view_into(&cube.view(), &vectors.view(), output.view_mut())
}

/// Compute batched cube-matrix vector products from views into `output`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn cube_matvec_view_into<T: NabledReal>(
    cube: &ArrayView3<'_, T>,
    vectors: &ArrayView2<'_, T>,
    mut output: ArrayViewMut2<'_, T>,
) -> Result<(), TensorError> {
    validate_cube_non_empty(cube)?;
    validate_matrix_non_empty(vectors)?;
    if vectors.dim() != (cube.dim().0, cube.dim().2) || output.dim() != (cube.dim().0, cube.dim().1)
    {
        return Err(TensorError::DimensionMismatch);
    }

    output.fill(T::zero());
    let (batch, rows, cols) = cube.dim();
    for b in 0..batch {
        for row in 0..rows {
            let mut sum = T::zero();
            for col in 0..cols {
                sum += cube[[b, row, col]] * vectors[[b, col]];
            }
            output[[b, row]] = sum;
        }
    }

    Ok(())
}

/// Compute batched complex cube-matrix vector products.
///
/// Inputs are `cube=(batch, rows, cols)` and `vectors=(batch, cols)`.
/// Output is `(batch, rows)`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn cube_matvec_complex(
    cube: &Array3<Complex64>,
    vectors: &Array2<Complex64>,
) -> Result<Array2<Complex64>, TensorError> {
    let mut output = Array2::<Complex64>::zeros((cube.dim().0, cube.dim().1));
    cube_matvec_complex_view_into(&cube.view(), &vectors.view(), output.view_mut())?;
    Ok(output)
}

/// Compute batched complex cube-matrix vector products from views.
///
/// Inputs are `cube=(batch, rows, cols)` and `vectors=(batch, cols)`.
/// Output is `(batch, rows)`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn cube_matvec_complex_view(
    cube: &ArrayView3<'_, Complex64>,
    vectors: &ArrayView2<'_, Complex64>,
) -> Result<Array2<Complex64>, TensorError> {
    let mut output = Array2::<Complex64>::zeros((cube.dim().0, cube.dim().1));
    cube_matvec_complex_view_into(cube, vectors, output.view_mut())?;
    Ok(output)
}

/// Compute batched complex cube-matrix vector products into `output`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn cube_matvec_complex_into(
    cube: &Array3<Complex64>,
    vectors: &Array2<Complex64>,
    output: &mut Array2<Complex64>,
) -> Result<(), TensorError> {
    cube_matvec_complex_view_into(&cube.view(), &vectors.view(), output.view_mut())
}

/// Compute batched complex cube-matrix vector products from views into `output`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn cube_matvec_complex_view_into(
    cube: &ArrayView3<'_, Complex64>,
    vectors: &ArrayView2<'_, Complex64>,
    mut output: ArrayViewMut2<'_, Complex64>,
) -> Result<(), TensorError> {
    validate_cube_non_empty_complex(cube)?;
    validate_matrix_non_empty_complex(vectors)?;
    if vectors.dim() != (cube.dim().0, cube.dim().2) || output.dim() != (cube.dim().0, cube.dim().1)
    {
        return Err(TensorError::DimensionMismatch);
    }

    output.fill(Complex64::new(0.0, 0.0));
    let (batch, rows, cols) = cube.dim();
    for b in 0..batch {
        for row in 0..rows {
            let mut sum = Complex64::new(0.0, 0.0);
            for col in 0..cols {
                sum += cube[[b, row, col]] * vectors[[b, col]];
            }
            output[[b, row]] = sum;
        }
    }

    Ok(())
}

/// Compute batched cube matrix-matrix products.
///
/// Inputs are `(batch, m, k)` and `(batch, k, n)` and output is `(batch, m, n)`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn cube_matmat<T: NabledReal>(
    left_cubes: &Array3<T>,
    right_cubes: &Array3<T>,
) -> Result<Array3<T>, TensorError> {
    let mut output =
        Array3::<T>::zeros((left_cubes.dim().0, left_cubes.dim().1, right_cubes.dim().2));
    cube_matmat_view_into(&left_cubes.view(), &right_cubes.view(), output.view_mut())?;
    Ok(output)
}

/// Compute batched cube matrix-matrix products from views.
///
/// Inputs are `(batch, m, k)` and `(batch, k, n)` and output is `(batch, m, n)`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn cube_matmat_view<T: NabledReal>(
    left_cubes: &ArrayView3<'_, T>,
    right_cubes: &ArrayView3<'_, T>,
) -> Result<Array3<T>, TensorError> {
    let mut output =
        Array3::<T>::zeros((left_cubes.dim().0, left_cubes.dim().1, right_cubes.dim().2));
    cube_matmat_view_into(left_cubes, right_cubes, output.view_mut())?;
    Ok(output)
}

/// Compute batched cube matrix-matrix products into `output`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn cube_matmat_into<T: NabledReal>(
    left_cubes: &Array3<T>,
    right_cubes: &Array3<T>,
    output: &mut Array3<T>,
) -> Result<(), TensorError> {
    cube_matmat_view_into(&left_cubes.view(), &right_cubes.view(), output.view_mut())
}

/// Compute batched cube matrix-matrix products from views into `output`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn cube_matmat_view_into<T: NabledReal>(
    left_cubes: &ArrayView3<'_, T>,
    right_cubes: &ArrayView3<'_, T>,
    mut output: ArrayViewMut3<'_, T>,
) -> Result<(), TensorError> {
    validate_cube_non_empty(left_cubes)?;
    validate_cube_non_empty(right_cubes)?;
    if left_cubes.dim().0 != right_cubes.dim().0
        || left_cubes.dim().2 != right_cubes.dim().1
        || output.dim() != (left_cubes.dim().0, left_cubes.dim().1, right_cubes.dim().2)
    {
        return Err(TensorError::DimensionMismatch);
    }

    output.fill(T::zero());
    let (batch, rows, inner) = left_cubes.dim();
    let cols = right_cubes.dim().2;
    for b in 0..batch {
        for row in 0..rows {
            for k in 0..inner {
                let lhs = left_cubes[[b, row, k]];
                for col in 0..cols {
                    output[[b, row, col]] += lhs * right_cubes[[b, k, col]];
                }
            }
        }
    }

    Ok(())
}

/// Compute batched complex cube matrix-matrix products.
///
/// Inputs are `(batch, m, k)` and `(batch, k, n)` and output is `(batch, m, n)`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn cube_matmat_complex(
    left_cubes: &Array3<Complex64>,
    right_cubes: &Array3<Complex64>,
) -> Result<Array3<Complex64>, TensorError> {
    let mut output =
        Array3::<Complex64>::zeros((left_cubes.dim().0, left_cubes.dim().1, right_cubes.dim().2));
    cube_matmat_complex_view_into(&left_cubes.view(), &right_cubes.view(), output.view_mut())?;
    Ok(output)
}

/// Compute batched complex cube matrix-matrix products from views.
///
/// Inputs are `(batch, m, k)` and `(batch, k, n)` and output is `(batch, m, n)`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn cube_matmat_complex_view(
    left_cubes: &ArrayView3<'_, Complex64>,
    right_cubes: &ArrayView3<'_, Complex64>,
) -> Result<Array3<Complex64>, TensorError> {
    let mut output =
        Array3::<Complex64>::zeros((left_cubes.dim().0, left_cubes.dim().1, right_cubes.dim().2));
    cube_matmat_complex_view_into(left_cubes, right_cubes, output.view_mut())?;
    Ok(output)
}

/// Compute batched complex cube matrix-matrix products into `output`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn cube_matmat_complex_into(
    left_cubes: &Array3<Complex64>,
    right_cubes: &Array3<Complex64>,
    output: &mut Array3<Complex64>,
) -> Result<(), TensorError> {
    cube_matmat_complex_view_into(&left_cubes.view(), &right_cubes.view(), output.view_mut())
}

/// Compute batched complex cube matrix-matrix products from views into `output`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn cube_matmat_complex_view_into(
    left_cubes: &ArrayView3<'_, Complex64>,
    right_cubes: &ArrayView3<'_, Complex64>,
    mut output: ArrayViewMut3<'_, Complex64>,
) -> Result<(), TensorError> {
    validate_cube_non_empty_complex(left_cubes)?;
    validate_cube_non_empty_complex(right_cubes)?;
    if left_cubes.dim().0 != right_cubes.dim().0
        || left_cubes.dim().2 != right_cubes.dim().1
        || output.dim() != (left_cubes.dim().0, left_cubes.dim().1, right_cubes.dim().2)
    {
        return Err(TensorError::DimensionMismatch);
    }

    output.fill(Complex64::new(0.0, 0.0));
    let (batch, rows, inner) = left_cubes.dim();
    let cols = right_cubes.dim().2;
    for b in 0..batch {
        for row in 0..rows {
            for k in 0..inner {
                let lhs = left_cubes[[b, row, k]];
                for col in 0..cols {
                    output[[b, row, col]] += lhs * right_cubes[[b, k, col]];
                }
            }
        }
    }

    Ok(())
}

/// Flatten each cube slice `(rows, cols)` into one row.
///
/// Input `(batch, rows, cols)` becomes `(batch, rows * cols)`.
///
/// # Errors
/// Returns an error if input is empty.
pub fn flatten_cubes<T: NabledReal>(cube: &Array3<T>) -> Result<Array2<T>, TensorError> {
    let cube_view = cube.view();
    validate_cube_non_empty(&cube_view)?;

    let (batch, rows, cols) = cube.dim();
    let mut output = Array2::<T>::zeros((batch, rows * cols));
    for b in 0..batch {
        for row in 0..rows {
            for col in 0..cols {
                output[[b, row * cols + col]] = cube[[b, row, col]];
            }
        }
    }
    Ok(output)
}

/// Reduce a tensor along its last axis by summation.
///
/// # Errors
/// Returns an error if tensor is empty or has zero dimensions.
pub fn sum_last_axis<T: NabledReal + Default>(
    tensor: &ArrayD<T>,
) -> Result<ArrayD<T>, TensorError> {
    sum_last_axis_view(&tensor.view())
}

/// Reduce a tensor view along its last axis by summation.
///
/// # Errors
/// Returns an error if tensor is empty or has zero dimensions.
pub fn sum_last_axis_view<T: NabledReal + Default>(
    tensor: &ArrayViewD<'_, T>,
) -> Result<ArrayD<T>, TensorError> {
    validate_tensor_nd_non_empty(tensor)?;
    let owned = tensor.to_owned();
    tensor_sum_last_axis_cpu(&owned).map_err(map_accelerator_error_to_tensor)
}

/// Reduce a tensor view along its last axis by summation into `output`.
///
/// # Errors
/// Returns an error if tensor is empty, has zero dimensions, or output shape mismatches.
pub fn sum_last_axis_view_into<T: NabledReal>(
    tensor: &ArrayViewD<'_, T>,
    output: &mut ArrayD<T>,
) -> Result<(), TensorError> {
    let tensor_view = tensor.view();
    validate_tensor_nd_non_empty(&tensor_view)?;
    let axis = Axis(tensor_view.ndim() - 1);
    let reduced = tensor_view.sum_axis(axis).into_dyn();
    if output.shape() != reduced.shape() {
        return Err(TensorError::DimensionMismatch);
    }
    output.assign(&reduced);
    Ok(())
}

/// Compute L2 norm along the last axis of a tensor.
///
/// # Errors
/// Returns an error if tensor is empty or has zero dimensions.
pub fn l2_norm_last_axis<T: NabledReal>(tensor: &ArrayD<T>) -> Result<ArrayD<T>, TensorError> {
    let tensor_view = tensor.view();
    validate_tensor_nd_non_empty(&tensor_view)?;

    let axis = Axis(tensor_view.ndim() - 1);
    let mut output_shape = tensor_view.shape().to_vec();
    let _ = output_shape.pop();
    let mut output = ArrayD::<T>::zeros(IxDyn(&output_shape));
    for (out_value, lane) in output.iter_mut().zip(tensor_view.lanes(axis)) {
        let sum_sq = lane
            .iter()
            .copied()
            .map(|value| value * value)
            .fold(T::zero(), |acc, value| acc + value);
        *out_value = sum_sq.sqrt();
    }
    Ok(output)
}

/// Normalize tensor values along the last axis.
///
/// # Errors
/// Returns an error if tensor is empty or has zero dimensions.
pub fn normalize_last_axis<T: NabledReal>(tensor: &ArrayD<T>) -> Result<ArrayD<T>, TensorError> {
    let tensor_view = tensor.view();
    validate_tensor_nd_non_empty(&tensor_view)?;

    let mut output = tensor.clone();
    let axis = Axis(tensor_view.ndim() - 1);
    for mut lane in output.lanes_mut(axis) {
        let norm = lane
            .iter()
            .copied()
            .map(|value| value * value)
            .fold(T::zero(), |acc, value| acc + value)
            .sqrt();
        let denominator = norm.max(T::epsilon());
        for value in &mut lane {
            *value /= denominator;
        }
    }
    Ok(output)
}

/// Compute batched dot products along the last axis of two tensors.
///
/// The input tensors must have identical shape and `ndim >= 1`.
/// Output shape is the input shape without the last axis.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn batched_dot_last_axis<T: NabledReal>(
    left: &ArrayD<T>,
    right: &ArrayD<T>,
) -> Result<ArrayD<T>, TensorError> {
    let left_view = left.view();
    let right_view = right.view();
    validate_tensor_nd_non_empty(&left_view)?;
    validate_tensor_nd_non_empty(&right_view)?;
    if left_view.shape() != right_view.shape() {
        return Err(TensorError::DimensionMismatch);
    }

    let axis = Axis(left_view.ndim() - 1);
    let mut output_shape = left_view.shape().to_vec();
    let _ = output_shape.pop();
    let mut output = ArrayD::<T>::zeros(IxDyn(&output_shape));
    for ((out_value, left_lane), right_lane) in
        output.iter_mut().zip(left_view.lanes(axis)).zip(right_view.lanes(axis))
    {
        let dot = left_lane
            .iter()
            .zip(right_lane.iter())
            .map(|(lhs, rhs)| *lhs * *rhs)
            .fold(T::zero(), |acc, value| acc + value);
        *out_value = dot;
    }
    Ok(output)
}

/// Permute tensor axes using an explicit axis ordering.
///
/// # Errors
/// Returns an error if the tensor is empty, has zero dimensions, or permutation is invalid.
pub fn permute_axes<T: NabledReal>(
    tensor: &ArrayD<T>,
    permutation: &[usize],
) -> Result<ArrayD<T>, TensorError> {
    let tensor_view = tensor.view();
    validate_tensor_nd_non_empty(&tensor_view)?;
    if !validate_permutation(tensor_view.ndim(), permutation) {
        return Err(TensorError::DimensionMismatch);
    }
    Ok(tensor_view.permuted_axes(permutation.to_vec()).to_owned())
}

/// Contract two tensors along explicit axis sets.
///
/// Output shape is:
/// - uncontracted axes of `left` (in original order), followed by
/// - uncontracted axes of `right` (in original order).
///
/// # Errors
/// Returns an error if inputs are empty, axes are invalid, or dimensions are incompatible.
pub fn contract_axes<T: NabledReal + Default>(
    left: &ArrayD<T>,
    right: &ArrayD<T>,
    left_axes: &[usize],
    right_axes: &[usize],
) -> Result<ArrayD<T>, TensorError> {
    contract_axes_view(&left.view(), &right.view(), left_axes, right_axes)
}

/// Contract two tensor views along explicit axis sets.
///
/// # Errors
/// Returns an error if inputs are empty, axes are invalid, or dimensions are incompatible.
pub fn contract_axes_view<T: NabledReal + Default>(
    left: &ArrayViewD<'_, T>,
    right: &ArrayViewD<'_, T>,
    left_axes: &[usize],
    right_axes: &[usize],
) -> Result<ArrayD<T>, TensorError> {
    let left_view = left.view();
    let right_view = right.view();
    validate_tensor_nd_non_empty(&left_view)?;
    validate_tensor_nd_non_empty(&right_view)?;

    if left_axes.len() != right_axes.len() {
        return Err(TensorError::DimensionMismatch);
    }
    if !validate_axes(left_view.ndim(), left_axes) || !validate_axes(right_view.ndim(), right_axes)
    {
        return Err(TensorError::DimensionMismatch);
    }

    if left_axes.len() == 1 {
        let left_owned = left.to_owned();
        let right_owned = right.to_owned();
        return tensor_contract_axes_cpu(&left_owned, &right_owned, left_axes[0], right_axes[0])
            .map_err(map_accelerator_error_to_tensor);
    }

    let left_free_axes = uncontracted_axes(left_view.ndim(), left_axes);
    let right_free_axes = uncontracted_axes(right_view.ndim(), right_axes);
    let mut output_shape =
        left_free_axes.iter().map(|axis| left_view.shape()[*axis]).collect::<Vec<_>>();
    output_shape.extend(right_free_axes.iter().map(|axis| right_view.shape()[*axis]));
    let mut output = ArrayD::<T>::zeros(IxDyn(&output_shape));
    contract_view_into_impl(&left_view, &right_view, left_axes, right_axes, &mut output)?;
    Ok(output)
}

/// Contract two tensors along explicit axis sets into `output`.
///
/// Output shape must match:
/// - uncontracted axes of `left` (in original order), followed by
/// - uncontracted axes of `right` (in original order).
///
/// # Errors
/// Returns an error if inputs are empty, axes are invalid, or dimensions are incompatible.
pub fn contract_axes_into<T: NabledReal + Default>(
    left: &ArrayD<T>,
    right: &ArrayD<T>,
    left_axes: &[usize],
    right_axes: &[usize],
    output: &mut ArrayD<T>,
) -> Result<(), TensorError> {
    contract_axes_view_into(&left.view(), &right.view(), left_axes, right_axes, output)
}

/// Contract two tensor views along explicit axis sets into `output`.
///
/// # Errors
/// Returns an error if inputs are empty, axes are invalid, or dimensions are incompatible.
pub fn contract_axes_view_into<T: NabledReal + Default>(
    left: &ArrayViewD<'_, T>,
    right: &ArrayViewD<'_, T>,
    left_axes: &[usize],
    right_axes: &[usize],
    output: &mut ArrayD<T>,
) -> Result<(), TensorError> {
    let left_view = left.view();
    let right_view = right.view();
    validate_tensor_nd_non_empty(&left_view)?;
    validate_tensor_nd_non_empty(&right_view)?;
    contract_view_into_impl(&left_view, &right_view, left_axes, right_axes, output)
}

/// Perform N-D batched matrix multiplication over the last two axes.
///
/// Inputs:
/// - `left`: `[..., m, k]`
/// - `right`: `[..., k, n]`
///
/// Output:
/// - `[..., m, n]`
///
/// # Errors
/// Returns an error if inputs are empty, have rank < 2, or dimensions are incompatible.
pub fn batched_matmul_last_two<T: NabledReal + Default>(
    left: &ArrayD<T>,
    right: &ArrayD<T>,
) -> Result<ArrayD<T>, TensorError> {
    batched_matmul_last_two_view(&left.view(), &right.view())
}

/// Perform N-D batched matrix multiplication over the last two axes from views.
///
/// # Errors
/// Returns an error if inputs are empty, have rank < 2, or dimensions are incompatible.
pub fn batched_matmul_last_two_view<T: NabledReal + Default>(
    left: &ArrayViewD<'_, T>,
    right: &ArrayViewD<'_, T>,
) -> Result<ArrayD<T>, TensorError> {
    let left_view = left.view();
    let right_view = right.view();
    validate_tensor_nd_non_empty(&left_view)?;
    validate_tensor_nd_non_empty(&right_view)?;
    if left_view.ndim() < 2 || right_view.ndim() < 2 {
        return Err(TensorError::DimensionMismatch);
    }

    let batch_ndim = left_view.ndim() - 2;
    if left_view.ndim() != right_view.ndim()
        || left_view.shape()[..batch_ndim] != right_view.shape()[..batch_ndim]
        || left_view.shape()[left_view.ndim() - 1] != right_view.shape()[right_view.ndim() - 2]
    {
        return Err(TensorError::DimensionMismatch);
    }

    let left_owned = left.to_owned();
    let right_owned = right.to_owned();
    tensor_batched_matmul_last_two_cpu(&left_owned, &right_owned)
        .map_err(map_accelerator_error_to_tensor)
}

/// Perform N-D batched matrix multiplication over the last two axes into `output`.
///
/// Inputs:
/// - `left`: `[..., m, k]`
/// - `right`: `[..., k, n]`
///
/// Output:
/// - `[..., m, n]`
///
/// # Errors
/// Returns an error if inputs are empty, have rank < 2, or dimensions are incompatible.
pub fn batched_matmul_last_two_into<T: NabledReal + Default>(
    left: &ArrayD<T>,
    right: &ArrayD<T>,
    output: &mut ArrayD<T>,
) -> Result<(), TensorError> {
    batched_matmul_last_two_view_into(&left.view(), &right.view(), output)
}

/// Perform N-D batched matrix multiplication over the last two axes from views into `output`.
///
/// # Errors
/// Returns an error if inputs are empty, have rank < 2, or dimensions are incompatible.
pub fn batched_matmul_last_two_view_into<T: NabledReal + Default>(
    left: &ArrayViewD<'_, T>,
    right: &ArrayViewD<'_, T>,
    output: &mut ArrayD<T>,
) -> Result<(), TensorError> {
    let left_view = left.view();
    let right_view = right.view();
    validate_tensor_nd_non_empty(&left_view)?;
    validate_tensor_nd_non_empty(&right_view)?;
    batched_matmul_last_two_view_into_impl(&left_view, &right_view, output)
}

/// Reduce a complex tensor along its last axis by summation.
///
/// # Errors
/// Returns an error if tensor is empty or has zero dimensions.
pub fn sum_last_axis_complex(tensor: &ArrayD<Complex64>) -> Result<ArrayD<Complex64>, TensorError> {
    sum_last_axis_complex_view(&tensor.view())
}

/// Reduce a complex tensor view along its last axis by summation.
///
/// # Errors
/// Returns an error if tensor is empty or has zero dimensions.
pub fn sum_last_axis_complex_view(
    tensor: &ArrayViewD<'_, Complex64>,
) -> Result<ArrayD<Complex64>, TensorError> {
    let tensor_view = tensor.view();
    validate_tensor_nd_non_empty_complex(&tensor_view)?;
    let owned = tensor.to_owned();
    tensor_sum_last_axis_cpu(&owned).map_err(map_accelerator_error_to_tensor)
}

/// Reduce a complex tensor view along its last axis by summation into `output`.
///
/// # Errors
/// Returns an error if tensor is empty, has zero dimensions, or output shape mismatches.
pub fn sum_last_axis_complex_view_into(
    tensor: &ArrayViewD<'_, Complex64>,
    output: &mut ArrayD<Complex64>,
) -> Result<(), TensorError> {
    let tensor_view = tensor.view();
    validate_tensor_nd_non_empty_complex(&tensor_view)?;
    let axis = Axis(tensor_view.ndim() - 1);
    let reduced = tensor_view.sum_axis(axis).into_dyn();
    if output.shape() != reduced.shape() {
        return Err(TensorError::DimensionMismatch);
    }
    output.assign(&reduced);
    Ok(())
}

/// Compute L2 norm along the last axis of a complex tensor.
///
/// # Errors
/// Returns an error if tensor is empty or has zero dimensions.
pub fn l2_norm_last_axis_complex(tensor: &ArrayD<Complex64>) -> Result<ArrayD<f64>, TensorError> {
    let tensor_view = tensor.view();
    validate_tensor_nd_non_empty_complex(&tensor_view)?;

    let axis = Axis(tensor_view.ndim() - 1);
    let mut output_shape = tensor_view.shape().to_vec();
    let _ = output_shape.pop();
    let mut output = ArrayD::<f64>::zeros(IxDyn(&output_shape));
    for (out_value, lane) in output.iter_mut().zip(tensor_view.lanes(axis)) {
        let sum_sq = lane.iter().map(Complex64::norm_sqr).sum::<f64>();
        *out_value = sum_sq.sqrt();
    }
    Ok(output)
}

/// Normalize complex tensor values along the last axis.
///
/// # Errors
/// Returns an error if tensor is empty or has zero dimensions.
pub fn normalize_last_axis_complex(
    tensor: &ArrayD<Complex64>,
) -> Result<ArrayD<Complex64>, TensorError> {
    let tensor_view = tensor.view();
    validate_tensor_nd_non_empty_complex(&tensor_view)?;

    let mut output = tensor.clone();
    let axis = Axis(tensor_view.ndim() - 1);
    for mut lane in output.lanes_mut(axis) {
        let norm = lane.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt();
        let denominator = norm.max(f64::EPSILON);
        for value in &mut lane {
            *value /= denominator;
        }
    }
    Ok(output)
}

/// Compute batched complex dot products along the last axis of two tensors.
///
/// The input tensors must have identical shape and `ndim >= 1`.
/// Output shape is the input shape without the last axis.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn batched_dot_last_axis_complex(
    left: &ArrayD<Complex64>,
    right: &ArrayD<Complex64>,
) -> Result<ArrayD<Complex64>, TensorError> {
    let left_view = left.view();
    let right_view = right.view();
    validate_tensor_nd_non_empty_complex(&left_view)?;
    validate_tensor_nd_non_empty_complex(&right_view)?;
    if left_view.shape() != right_view.shape() {
        return Err(TensorError::DimensionMismatch);
    }

    let axis = Axis(left_view.ndim() - 1);
    let mut output_shape = left_view.shape().to_vec();
    let _ = output_shape.pop();
    let mut output = ArrayD::<Complex64>::zeros(IxDyn(&output_shape));
    for ((out_value, left_lane), right_lane) in
        output.iter_mut().zip(left_view.lanes(axis)).zip(right_view.lanes(axis))
    {
        let dot = left_lane
            .iter()
            .zip(right_lane.iter())
            .map(|(lhs, rhs)| lhs.conj() * rhs)
            .sum::<Complex64>();
        *out_value = dot;
    }
    Ok(output)
}

/// Permute complex tensor axes using an explicit axis ordering.
///
/// # Errors
/// Returns an error if the tensor is empty, has zero dimensions, or permutation is invalid.
pub fn permute_axes_complex(
    tensor: &ArrayD<Complex64>,
    permutation: &[usize],
) -> Result<ArrayD<Complex64>, TensorError> {
    let tensor_view = tensor.view();
    validate_tensor_nd_non_empty_complex(&tensor_view)?;
    if !validate_permutation(tensor_view.ndim(), permutation) {
        return Err(TensorError::DimensionMismatch);
    }
    Ok(tensor_view.permuted_axes(permutation.to_vec()).to_owned())
}

/// Contract two complex tensors along explicit axis sets.
///
/// Output shape is:
/// - uncontracted axes of `left` (in original order), followed by
/// - uncontracted axes of `right` (in original order).
///
/// # Errors
/// Returns an error if inputs are empty, axes are invalid, or dimensions are incompatible.
pub fn contract_axes_complex(
    left: &ArrayD<Complex64>,
    right: &ArrayD<Complex64>,
    left_axes: &[usize],
    right_axes: &[usize],
) -> Result<ArrayD<Complex64>, TensorError> {
    contract_axes_complex_view(&left.view(), &right.view(), left_axes, right_axes)
}

/// Contract two complex tensor views along explicit axis sets.
///
/// # Errors
/// Returns an error if inputs are empty, axes are invalid, or dimensions are incompatible.
pub fn contract_axes_complex_view(
    left: &ArrayViewD<'_, Complex64>,
    right: &ArrayViewD<'_, Complex64>,
    left_axes: &[usize],
    right_axes: &[usize],
) -> Result<ArrayD<Complex64>, TensorError> {
    let left_view = left.view();
    let right_view = right.view();
    validate_tensor_nd_non_empty_complex(&left_view)?;
    validate_tensor_nd_non_empty_complex(&right_view)?;

    if left_axes.len() != right_axes.len() {
        return Err(TensorError::DimensionMismatch);
    }
    if !validate_axes(left_view.ndim(), left_axes) || !validate_axes(right_view.ndim(), right_axes)
    {
        return Err(TensorError::DimensionMismatch);
    }

    if left_axes.len() == 1 {
        let left_owned = left.to_owned();
        let right_owned = right.to_owned();
        return tensor_contract_axes_cpu(&left_owned, &right_owned, left_axes[0], right_axes[0])
            .map_err(map_accelerator_error_to_tensor);
    }

    let left_free_axes = uncontracted_axes(left_view.ndim(), left_axes);
    let right_free_axes = uncontracted_axes(right_view.ndim(), right_axes);
    let mut output_shape =
        left_free_axes.iter().map(|axis| left_view.shape()[*axis]).collect::<Vec<_>>();
    output_shape.extend(right_free_axes.iter().map(|axis| right_view.shape()[*axis]));
    let mut output = ArrayD::<Complex64>::zeros(IxDyn(&output_shape));
    contract_view_into_impl(&left_view, &right_view, left_axes, right_axes, &mut output)?;
    Ok(output)
}

/// Contract two complex tensors along explicit axis sets into `output`.
///
/// Output shape must match:
/// - uncontracted axes of `left` (in original order), followed by
/// - uncontracted axes of `right` (in original order).
///
/// # Errors
/// Returns an error if inputs are empty, axes are invalid, or dimensions are incompatible.
pub fn contract_axes_complex_into(
    left: &ArrayD<Complex64>,
    right: &ArrayD<Complex64>,
    left_axes: &[usize],
    right_axes: &[usize],
    output: &mut ArrayD<Complex64>,
) -> Result<(), TensorError> {
    contract_axes_complex_view_into(&left.view(), &right.view(), left_axes, right_axes, output)
}

/// Contract two complex tensor views along explicit axis sets into `output`.
///
/// # Errors
/// Returns an error if inputs are empty, axes are invalid, or dimensions are incompatible.
pub fn contract_axes_complex_view_into(
    left: &ArrayViewD<'_, Complex64>,
    right: &ArrayViewD<'_, Complex64>,
    left_axes: &[usize],
    right_axes: &[usize],
    output: &mut ArrayD<Complex64>,
) -> Result<(), TensorError> {
    let left_view = left.view();
    let right_view = right.view();
    validate_tensor_nd_non_empty_complex(&left_view)?;
    validate_tensor_nd_non_empty_complex(&right_view)?;
    contract_view_into_impl(&left_view, &right_view, left_axes, right_axes, output)
}

/// Perform N-D batched complex matrix multiplication over the last two axes.
///
/// Inputs:
/// - `left`: `[..., m, k]`
/// - `right`: `[..., k, n]`
///
/// Output:
/// - `[..., m, n]`
///
/// # Errors
/// Returns an error if inputs are empty, have rank < 2, or dimensions are incompatible.
pub fn batched_matmul_last_two_complex(
    left: &ArrayD<Complex64>,
    right: &ArrayD<Complex64>,
) -> Result<ArrayD<Complex64>, TensorError> {
    batched_matmul_last_two_complex_view(&left.view(), &right.view())
}

/// Perform N-D batched complex matrix multiplication over the last two axes from views.
///
/// # Errors
/// Returns an error if inputs are empty, have rank < 2, or dimensions are incompatible.
pub fn batched_matmul_last_two_complex_view(
    left: &ArrayViewD<'_, Complex64>,
    right: &ArrayViewD<'_, Complex64>,
) -> Result<ArrayD<Complex64>, TensorError> {
    let left_view = left.view();
    let right_view = right.view();
    validate_tensor_nd_non_empty_complex(&left_view)?;
    validate_tensor_nd_non_empty_complex(&right_view)?;
    if left_view.ndim() < 2 || right_view.ndim() < 2 {
        return Err(TensorError::DimensionMismatch);
    }

    let batch_ndim = left_view.ndim() - 2;
    if left_view.ndim() != right_view.ndim()
        || left_view.shape()[..batch_ndim] != right_view.shape()[..batch_ndim]
        || left_view.shape()[left_view.ndim() - 1] != right_view.shape()[right_view.ndim() - 2]
    {
        return Err(TensorError::DimensionMismatch);
    }

    let left_owned = left.to_owned();
    let right_owned = right.to_owned();
    tensor_batched_matmul_last_two_cpu(&left_owned, &right_owned)
        .map_err(map_accelerator_error_to_tensor)
}

/// Perform N-D batched complex matrix multiplication over the last two axes into `output`.
///
/// Inputs:
/// - `left`: `[..., m, k]`
/// - `right`: `[..., k, n]`
///
/// Output:
/// - `[..., m, n]`
///
/// # Errors
/// Returns an error if inputs are empty, have rank < 2, or dimensions are incompatible.
pub fn batched_matmul_last_two_complex_into(
    left: &ArrayD<Complex64>,
    right: &ArrayD<Complex64>,
    output: &mut ArrayD<Complex64>,
) -> Result<(), TensorError> {
    batched_matmul_last_two_complex_view_into(&left.view(), &right.view(), output)
}

/// Perform N-D batched complex matrix multiplication over the last two axes from views into
/// `output`.
///
/// # Errors
/// Returns an error if inputs are empty, have rank < 2, or dimensions are incompatible.
pub fn batched_matmul_last_two_complex_view_into(
    left: &ArrayViewD<'_, Complex64>,
    right: &ArrayViewD<'_, Complex64>,
    output: &mut ArrayD<Complex64>,
) -> Result<(), TensorError> {
    let left_view = left.view();
    let right_view = right.view();
    validate_tensor_nd_non_empty_complex(&left_view)?;
    validate_tensor_nd_non_empty_complex(&right_view)?;
    batched_matmul_last_two_view_into_impl(&left_view, &right_view, output)
}

fn parse_einsum_two_operands(expression: &str) -> Result<EinsumOperands, TensorError> {
    let Some((inputs, output)) = expression.split_once("->") else {
        return Err(TensorError::DimensionMismatch);
    };
    let mut input_parts = inputs.split(',');
    let Some(left_part) = input_parts.next() else {
        return Err(TensorError::DimensionMismatch);
    };
    let Some(right_part) = input_parts.next() else {
        return Err(TensorError::DimensionMismatch);
    };
    if input_parts.next().is_some() {
        return Err(TensorError::DimensionMismatch);
    }

    let left_labels = left_part.chars().collect::<Vec<_>>();
    let right_labels = right_part.chars().collect::<Vec<_>>();
    let output_labels = output.chars().collect::<Vec<_>>();
    if left_labels.is_empty() || right_labels.is_empty() {
        return Err(TensorError::DimensionMismatch);
    }
    Ok((left_labels, right_labels, output_labels))
}

fn validate_einsum_label_set(labels: &[char]) -> bool {
    let mut seen = std::collections::BTreeSet::<char>::new();
    for &label in labels {
        if !label.is_ascii_alphabetic() || !seen.insert(label) {
            return false;
        }
    }
    true
}

fn decode_flat_index(mut index: usize, shape: &[usize], coords: &mut [usize]) {
    if shape.is_empty() {
        return;
    }
    for axis_rev in (0..shape.len()).rev() {
        let extent = shape[axis_rev].max(1);
        coords[axis_rev] = index % extent;
        index /= extent;
    }
}

fn label_index_map(labels: &[char]) -> std::collections::BTreeMap<char, usize> {
    let mut map = std::collections::BTreeMap::<char, usize>::new();
    for (idx, label) in labels.iter().copied().enumerate() {
        let _ = map.insert(label, idx);
    }
    map
}

fn union_labels(left: &[char], right: &[char]) -> Vec<char> {
    let mut labels = std::collections::BTreeSet::<char>::new();
    for label in left.iter().copied().chain(right.iter().copied()) {
        let _ = labels.insert(label);
    }
    labels.into_iter().collect::<Vec<_>>()
}

fn build_einsum_dimensions<T>(
    left: &ArrayViewD<'_, T>,
    right: &ArrayViewD<'_, T>,
    left_labels: &[char],
    right_labels: &[char],
) -> Result<std::collections::BTreeMap<char, usize>, TensorError> {
    let mut dims = std::collections::BTreeMap::<char, usize>::new();
    for (&label, &extent) in left_labels.iter().zip(left.shape().iter()) {
        if let Some(existing) = dims.get(&label).copied() {
            if existing != extent {
                return Err(TensorError::DimensionMismatch);
            }
        } else {
            let _ = dims.insert(label, extent);
        }
    }
    for (&label, &extent) in right_labels.iter().zip(right.shape().iter()) {
        if let Some(existing) = dims.get(&label).copied() {
            if existing != extent {
                return Err(TensorError::DimensionMismatch);
            }
        } else {
            let _ = dims.insert(label, extent);
        }
    }
    Ok(dims)
}

fn build_einsum_dimensions_complex(
    left: &ArrayViewD<'_, Complex64>,
    right: &ArrayViewD<'_, Complex64>,
    left_labels: &[char],
    right_labels: &[char],
) -> Result<std::collections::BTreeMap<char, usize>, TensorError> {
    let mut dims = std::collections::BTreeMap::<char, usize>::new();
    for (&label, &extent) in left_labels.iter().zip(left.shape().iter()) {
        if let Some(existing) = dims.get(&label).copied() {
            if existing != extent {
                return Err(TensorError::DimensionMismatch);
            }
        } else {
            let _ = dims.insert(label, extent);
        }
    }
    for (&label, &extent) in right_labels.iter().zip(right.shape().iter()) {
        if let Some(existing) = dims.get(&label).copied() {
            if existing != extent {
                return Err(TensorError::DimensionMismatch);
            }
        } else {
            let _ = dims.insert(label, extent);
        }
    }
    Ok(dims)
}

fn einsum_binary_impl<T: NabledReal>(
    left: &ArrayViewD<'_, T>,
    right: &ArrayViewD<'_, T>,
    left_labels: &[char],
    right_labels: &[char],
    output_labels: &[char],
) -> Result<ArrayD<T>, TensorError> {
    if left_labels.len() != left.ndim() || right_labels.len() != right.ndim() {
        return Err(TensorError::DimensionMismatch);
    }
    if !validate_einsum_label_set(left_labels)
        || !validate_einsum_label_set(right_labels)
        || !validate_einsum_label_set(output_labels)
    {
        return Err(TensorError::DimensionMismatch);
    }

    let dims = build_einsum_dimensions(left, right, left_labels, right_labels)?;
    for label in output_labels {
        if !dims.contains_key(label) {
            return Err(TensorError::DimensionMismatch);
        }
    }

    let union = union_labels(left_labels, right_labels);
    let sum_labels =
        union.iter().copied().filter(|label| !output_labels.contains(label)).collect::<Vec<_>>();
    let output_shape = output_labels
        .iter()
        .map(|label| dims.get(label).copied().unwrap_or(0))
        .collect::<Vec<_>>();
    let sum_shape =
        sum_labels.iter().map(|label| dims.get(label).copied().unwrap_or(0)).collect::<Vec<_>>();
    let output_size = shape_product(&output_shape);
    let sum_size = shape_product(&sum_shape);

    let mut output = ArrayD::<T>::zeros(IxDyn(&output_shape));
    let mut output_coords = vec![0_usize; output_shape.len()];
    let mut sum_coords = vec![0_usize; sum_shape.len()];
    let label_to_slot = label_index_map(&union);
    let mut label_values = vec![0_usize; union.len()];
    let left_label_pos = label_index_map(left_labels);
    let right_label_pos = label_index_map(right_labels);

    for output_flat in 0..output_size {
        decode_flat_index(output_flat, &output_shape, &mut output_coords);
        for (&label, &coord) in output_labels.iter().zip(output_coords.iter()) {
            let slot = label_to_slot[&label];
            label_values[slot] = coord;
        }

        let mut sum = T::zero();
        for sum_flat in 0..sum_size {
            decode_flat_index(sum_flat, &sum_shape, &mut sum_coords);
            for (&label, &coord) in sum_labels.iter().zip(sum_coords.iter()) {
                let slot = label_to_slot[&label];
                label_values[slot] = coord;
            }

            let mut left_index = vec![0_usize; left_labels.len()];
            for (&label, &position) in &left_label_pos {
                let slot = label_to_slot[&label];
                left_index[position] = label_values[slot];
            }
            let mut right_index = vec![0_usize; right_labels.len()];
            for (&label, &position) in &right_label_pos {
                let slot = label_to_slot[&label];
                right_index[position] = label_values[slot];
            }
            sum += left[IxDyn(&left_index)] * right[IxDyn(&right_index)];
        }

        output[IxDyn(&output_coords)] = sum;
    }

    Ok(output)
}

fn einsum_binary_impl_complex(
    left: &ArrayViewD<'_, Complex64>,
    right: &ArrayViewD<'_, Complex64>,
    left_labels: &[char],
    right_labels: &[char],
    output_labels: &[char],
) -> Result<ArrayD<Complex64>, TensorError> {
    if left_labels.len() != left.ndim() || right_labels.len() != right.ndim() {
        return Err(TensorError::DimensionMismatch);
    }
    if !validate_einsum_label_set(left_labels)
        || !validate_einsum_label_set(right_labels)
        || !validate_einsum_label_set(output_labels)
    {
        return Err(TensorError::DimensionMismatch);
    }

    let dims = build_einsum_dimensions_complex(left, right, left_labels, right_labels)?;
    for label in output_labels {
        if !dims.contains_key(label) {
            return Err(TensorError::DimensionMismatch);
        }
    }

    let union = union_labels(left_labels, right_labels);
    let sum_labels =
        union.iter().copied().filter(|label| !output_labels.contains(label)).collect::<Vec<_>>();
    let output_shape = output_labels
        .iter()
        .map(|label| dims.get(label).copied().unwrap_or(0))
        .collect::<Vec<_>>();
    let sum_shape =
        sum_labels.iter().map(|label| dims.get(label).copied().unwrap_or(0)).collect::<Vec<_>>();
    let output_size = shape_product(&output_shape);
    let sum_size = shape_product(&sum_shape);

    let mut output = ArrayD::<Complex64>::zeros(IxDyn(&output_shape));
    let mut output_coords = vec![0_usize; output_shape.len()];
    let mut sum_coords = vec![0_usize; sum_shape.len()];
    let label_to_slot = label_index_map(&union);
    let mut label_values = vec![0_usize; union.len()];
    let left_label_pos = label_index_map(left_labels);
    let right_label_pos = label_index_map(right_labels);

    for output_flat in 0..output_size {
        decode_flat_index(output_flat, &output_shape, &mut output_coords);
        for (&label, &coord) in output_labels.iter().zip(output_coords.iter()) {
            let slot = label_to_slot[&label];
            label_values[slot] = coord;
        }

        let mut sum = Complex64::new(0.0, 0.0);
        for sum_flat in 0..sum_size {
            decode_flat_index(sum_flat, &sum_shape, &mut sum_coords);
            for (&label, &coord) in sum_labels.iter().zip(sum_coords.iter()) {
                let slot = label_to_slot[&label];
                label_values[slot] = coord;
            }

            let mut left_index = vec![0_usize; left_labels.len()];
            for (&label, &position) in &left_label_pos {
                let slot = label_to_slot[&label];
                left_index[position] = label_values[slot];
            }
            let mut right_index = vec![0_usize; right_labels.len()];
            for (&label, &position) in &right_label_pos {
                let slot = label_to_slot[&label];
                right_index[position] = label_values[slot];
            }
            sum += left[IxDyn(&left_index)] * right[IxDyn(&right_index)];
        }

        output[IxDyn(&output_coords)] = sum;
    }

    Ok(output)
}

/// Evaluate two-operand Einstein summation over real tensors.
///
/// Expression format: `"labels_left,labels_right->labels_out"`, for example
/// `"bij,bjk->bik"` or `"ab,bc->ac"`.
///
/// # Errors
/// Returns an error if expression syntax is invalid or dimensions are incompatible.
pub fn einsum<T: NabledReal>(
    expression: &str,
    left: &ArrayD<T>,
    right: &ArrayD<T>,
) -> Result<ArrayD<T>, TensorError> {
    let left_view = left.view();
    let right_view = right.view();
    validate_tensor_nd_non_empty(&left_view)?;
    validate_tensor_nd_non_empty(&right_view)?;
    let (left_labels, right_labels, output_labels) = parse_einsum_two_operands(expression)?;
    einsum_binary_impl(&left_view, &right_view, &left_labels, &right_labels, &output_labels)
}

/// Evaluate two-operand Einstein summation over complex tensors.
///
/// Expression format: `"labels_left,labels_right->labels_out"`, for example
/// `"bij,bjk->bik"` or `"ab,bc->ac"`.
///
/// # Errors
/// Returns an error if expression syntax is invalid or dimensions are incompatible.
pub fn einsum_complex(
    expression: &str,
    left: &ArrayD<Complex64>,
    right: &ArrayD<Complex64>,
) -> Result<ArrayD<Complex64>, TensorError> {
    let left_view = left.view();
    let right_view = right.view();
    validate_tensor_nd_non_empty_complex(&left_view)?;
    validate_tensor_nd_non_empty_complex(&right_view)?;
    let (left_labels, right_labels, output_labels) = parse_einsum_two_operands(expression)?;
    einsum_binary_impl_complex(&left_view, &right_view, &left_labels, &right_labels, &output_labels)
}

fn mode0_product<T: NabledReal>(
    tensor: &Array3<T>,
    matrix: &Array2<T>,
) -> Result<Array3<T>, TensorError> {
    let (i0, i1, i2) = tensor.dim();
    if matrix.ncols() != i0 {
        return Err(TensorError::DimensionMismatch);
    }
    let mut output = Array3::<T>::zeros((matrix.nrows(), i1, i2));
    for r in 0..matrix.nrows() {
        for i in 0..i0 {
            let weight = matrix[[r, i]];
            for j in 0..i1 {
                for k in 0..i2 {
                    output[[r, j, k]] += weight * tensor[[i, j, k]];
                }
            }
        }
    }
    Ok(output)
}

fn mode1_product<T: NabledReal>(
    tensor: &Array3<T>,
    matrix: &Array2<T>,
) -> Result<Array3<T>, TensorError> {
    let (i0, i1, i2) = tensor.dim();
    if matrix.ncols() != i1 {
        return Err(TensorError::DimensionMismatch);
    }
    let mut output = Array3::<T>::zeros((i0, matrix.nrows(), i2));
    for r in 0..matrix.nrows() {
        for j in 0..i1 {
            let weight = matrix[[r, j]];
            for i in 0..i0 {
                for k in 0..i2 {
                    output[[i, r, k]] += weight * tensor[[i, j, k]];
                }
            }
        }
    }
    Ok(output)
}

fn mode2_product<T: NabledReal>(
    tensor: &Array3<T>,
    matrix: &Array2<T>,
) -> Result<Array3<T>, TensorError> {
    let (i0, i1, i2) = tensor.dim();
    if matrix.ncols() != i2 {
        return Err(TensorError::DimensionMismatch);
    }
    let mut output = Array3::<T>::zeros((i0, i1, matrix.nrows()));
    for r in 0..matrix.nrows() {
        for k in 0..i2 {
            let weight = matrix[[r, k]];
            for i in 0..i0 {
                for j in 0..i1 {
                    output[[i, j, r]] += weight * tensor[[i, j, k]];
                }
            }
        }
    }
    Ok(output)
}

fn unfold_mode0<T: NabledReal>(tensor: &Array3<T>) -> Array2<T> {
    let (i0, i1, i2) = tensor.dim();
    let mut unfolded = Array2::<T>::zeros((i0, i1 * i2));
    for i in 0..i0 {
        for j in 0..i1 {
            for k in 0..i2 {
                unfolded[[i, j * i2 + k]] = tensor[[i, j, k]];
            }
        }
    }
    unfolded
}

fn unfold_mode1<T: NabledReal>(tensor: &Array3<T>) -> Array2<T> {
    let (i0, i1, i2) = tensor.dim();
    let mut unfolded = Array2::<T>::zeros((i1, i0 * i2));
    for j in 0..i1 {
        for i in 0..i0 {
            for k in 0..i2 {
                unfolded[[j, i * i2 + k]] = tensor[[i, j, k]];
            }
        }
    }
    unfolded
}

fn unfold_mode2<T: NabledReal>(tensor: &Array3<T>) -> Array2<T> {
    let (i0, i1, i2) = tensor.dim();
    let mut unfolded = Array2::<T>::zeros((i2, i0 * i1));
    for k in 0..i2 {
        for i in 0..i0 {
            for j in 0..i1 {
                unfolded[[k, i * i1 + j]] = tensor[[i, j, k]];
            }
        }
    }
    unfolded
}

/// Compute rank-truncated HOSVD for a rank-3 real tensor.
///
/// # Errors
/// Returns an error if input is empty, ranks are invalid, or factorization fails.
#[cfg(feature = "lapack-provider")]
pub fn hosvd3<T>(
    cube: &Array3<T>,
    ranks: (usize, usize, usize),
) -> Result<Hosvd3Result<T>, TensorError>
where
    T: NabledReal + ndarray_linalg::Lapack<Real = T> + AddAssign,
{
    hosvd3_impl(cube, ranks)
}

/// Compute rank-truncated HOSVD for a rank-3 real tensor.
///
/// # Errors
/// Returns an error if input is empty, ranks are invalid, or factorization fails.
#[cfg(not(feature = "lapack-provider"))]
pub fn hosvd3<T: NabledReal>(
    cube: &Array3<T>,
    ranks: (usize, usize, usize),
) -> Result<Hosvd3Result<T>, TensorError> {
    hosvd3_impl(cube, ranks)
}

#[cfg(not(feature = "lapack-provider"))]
fn hosvd3_impl<T: NabledReal>(
    cube: &Array3<T>,
    ranks: (usize, usize, usize),
) -> Result<Hosvd3Result<T>, TensorError> {
    let cube_view = cube.view();
    validate_cube_non_empty(&cube_view)?;
    let (i0, i1, i2) = cube.dim();
    if ranks.0 == 0 || ranks.1 == 0 || ranks.2 == 0 || ranks.0 > i0 || ranks.1 > i1 || ranks.2 > i2
    {
        return Err(TensorError::DimensionMismatch);
    }

    let u0_full =
        svd::decompose(&unfold_mode0(cube)).map_err(|_| TensorError::DimensionMismatch)?.u;
    let u1_full =
        svd::decompose(&unfold_mode1(cube)).map_err(|_| TensorError::DimensionMismatch)?.u;
    let u2_full =
        svd::decompose(&unfold_mode2(cube)).map_err(|_| TensorError::DimensionMismatch)?.u;

    let u0 = u0_full.slice(s![.., 0..ranks.0]).to_owned();
    let u1 = u1_full.slice(s![.., 0..ranks.1]).to_owned();
    let u2 = u2_full.slice(s![.., 0..ranks.2]).to_owned();

    let core_mode0 = mode0_product(cube, &u0.t().to_owned())?;
    let core_mode1 = mode1_product(&core_mode0, &u1.t().to_owned())?;
    let core = mode2_product(&core_mode1, &u2.t().to_owned())?;
    Ok(Hosvd3Result { core, u0, u1, u2 })
}

#[cfg(feature = "lapack-provider")]
fn hosvd3_impl<T>(
    cube: &Array3<T>,
    ranks: (usize, usize, usize),
) -> Result<Hosvd3Result<T>, TensorError>
where
    T: NabledReal + ndarray_linalg::Lapack<Real = T> + AddAssign,
{
    let cube_view = cube.view();
    validate_cube_non_empty(&cube_view)?;
    let (i0, i1, i2) = cube.dim();
    if ranks.0 == 0 || ranks.1 == 0 || ranks.2 == 0 || ranks.0 > i0 || ranks.1 > i1 || ranks.2 > i2
    {
        return Err(TensorError::DimensionMismatch);
    }

    let u0_full =
        svd::decompose(&unfold_mode0(cube)).map_err(|_| TensorError::DimensionMismatch)?.u;
    let u1_full =
        svd::decompose(&unfold_mode1(cube)).map_err(|_| TensorError::DimensionMismatch)?.u;
    let u2_full =
        svd::decompose(&unfold_mode2(cube)).map_err(|_| TensorError::DimensionMismatch)?.u;

    let u0 = u0_full.slice(s![.., 0..ranks.0]).to_owned();
    let u1 = u1_full.slice(s![.., 0..ranks.1]).to_owned();
    let u2 = u2_full.slice(s![.., 0..ranks.2]).to_owned();

    let core_mode0 = mode0_product(cube, &u0.t().to_owned())?;
    let core_mode1 = mode1_product(&core_mode0, &u1.t().to_owned())?;
    let core = mode2_product(&core_mode1, &u2.t().to_owned())?;
    Ok(Hosvd3Result { core, u0, u1, u2 })
}

/// Reconstruct a rank-3 tensor from HOSVD factors.
///
/// # Errors
/// Returns an error if factor dimensions are incompatible.
pub fn hosvd3_reconstruct<T: NabledReal>(
    result: &Hosvd3Result<T>,
) -> Result<Array3<T>, TensorError> {
    let mode0 = mode0_product(&result.core, &result.u0)?;
    let mode1 = mode1_product(&mode0, &result.u1)?;
    mode2_product(&mode1, &result.u2)
}

#[cfg(test)]
mod tests {
    use ndarray::{Array2, Array3, ArrayD, IxDyn};
    use num_complex::Complex64;

    use super::*;

    #[test]
    fn cube_matvec_variants_match() {
        let cube = Array3::from_shape_vec((2, 2, 3), vec![
            1.0_f64, 2.0_f64, 3.0_f64, 0.0_f64, 1.0_f64, 1.0_f64, 2.0_f64, -1.0_f64, 0.5_f64,
            3.0_f64, 0.0_f64, 2.0_f64,
        ])
        .unwrap();
        let vectors = Array2::from_shape_vec((2, 3), vec![
            1.0_f64, 0.0_f64, 2.0_f64, 0.5_f64, -1.0_f64, 1.0_f64,
        ])
        .unwrap();

        let allocating = cube_matvec(&cube, &vectors).unwrap();
        let viewed = cube_matvec_view(&cube.view(), &vectors.view()).unwrap();
        let mut into = Array2::<f64>::zeros((2, 2));
        cube_matvec_into(&cube, &vectors, &mut into).unwrap();

        for b in 0..2 {
            for row in 0..2 {
                assert!((allocating[[b, row]] - viewed[[b, row]]).abs() < 1e-12_f64);
                assert!((allocating[[b, row]] - into[[b, row]]).abs() < 1e-12_f64);
            }
        }
    }

    #[test]
    fn cube_matmat_variants_match() {
        let left = Array3::from_shape_vec((2, 2, 3), vec![
            1.0_f64, 2.0_f64, 0.0_f64, 0.0_f64, 1.0_f64, 1.0_f64, 2.0_f64, 0.0_f64, 1.0_f64,
            1.0_f64, 3.0_f64, 2.0_f64,
        ])
        .unwrap();
        let right = Array3::from_shape_vec((2, 3, 2), vec![
            1.0_f64, 0.0_f64, 2.0_f64, 1.0_f64, 1.0_f64, 3.0_f64, 0.0_f64, 2.0_f64, 1.0_f64,
            1.0_f64, 3.0_f64, 0.0_f64,
        ])
        .unwrap();

        let allocating = cube_matmat(&left, &right).unwrap();
        let viewed = cube_matmat_view(&left.view(), &right.view()).unwrap();
        let mut into = Array3::<f64>::zeros((2, 2, 2));
        cube_matmat_into(&left, &right, &mut into).unwrap();

        for b in 0..2 {
            for i in 0..2 {
                for j in 0..2 {
                    assert!((allocating[[b, i, j]] - viewed[[b, i, j]]).abs() < 1e-12_f64);
                    assert!((allocating[[b, i, j]] - into[[b, i, j]]).abs() < 1e-12_f64);
                }
            }
        }
    }

    #[test]
    fn cube_matvec_complex_variants_match() {
        let cube = Array3::from_shape_vec((2, 2, 2), vec![
            Complex64::new(1.0_f64, 0.0_f64),
            Complex64::new(2.0_f64, -1.0_f64),
            Complex64::new(0.0_f64, 1.0_f64),
            Complex64::new(1.0_f64, 0.0_f64),
            Complex64::new(2.0_f64, 1.0_f64),
            Complex64::new(-1.0_f64, 0.0_f64),
            Complex64::new(0.5_f64, 0.5_f64),
            Complex64::new(3.0_f64, -2.0_f64),
        ])
        .unwrap();
        let vectors = Array2::from_shape_vec((2, 2), vec![
            Complex64::new(1.0_f64, 0.0_f64),
            Complex64::new(0.5_f64, -0.5_f64),
            Complex64::new(-1.0_f64, 1.0_f64),
            Complex64::new(2.0_f64, 0.0_f64),
        ])
        .unwrap();

        let allocating = cube_matvec_complex(&cube, &vectors).unwrap();
        let viewed = cube_matvec_complex_view(&cube.view(), &vectors.view()).unwrap();
        let mut into = Array2::<Complex64>::zeros((2, 2));
        cube_matvec_complex_into(&cube, &vectors, &mut into).unwrap();

        for b in 0..2 {
            for row in 0..2 {
                assert!((allocating[[b, row]] - viewed[[b, row]]).norm() < 1e-12_f64);
                assert!((allocating[[b, row]] - into[[b, row]]).norm() < 1e-12_f64);
            }
        }
    }

    #[test]
    fn cube_matmat_complex_variants_match() {
        let left = Array3::from_shape_vec((1, 2, 2), vec![
            Complex64::new(1.0_f64, 0.0_f64),
            Complex64::new(2.0_f64, -1.0_f64),
            Complex64::new(0.0_f64, 1.0_f64),
            Complex64::new(1.0_f64, 0.0_f64),
        ])
        .unwrap();
        let right = Array3::from_shape_vec((1, 2, 2), vec![
            Complex64::new(1.0_f64, 1.0_f64),
            Complex64::new(0.0_f64, 1.0_f64),
            Complex64::new(2.0_f64, 0.0_f64),
            Complex64::new(1.0_f64, -1.0_f64),
        ])
        .unwrap();

        let allocating = cube_matmat_complex(&left, &right).unwrap();
        let viewed = cube_matmat_complex_view(&left.view(), &right.view()).unwrap();
        let mut into = Array3::<Complex64>::zeros((1, 2, 2));
        cube_matmat_complex_into(&left, &right, &mut into).unwrap();

        for i in 0..2 {
            for j in 0..2 {
                assert!((allocating[[0, i, j]] - viewed[[0, i, j]]).norm() < 1e-12_f64);
                assert!((allocating[[0, i, j]] - into[[0, i, j]]).norm() < 1e-12_f64);
            }
        }
    }

    #[test]
    fn flatten_cubes_is_shape_stable() {
        let cube = Array3::from_shape_vec((2, 2, 2), vec![
            1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64, 0.0_f64, 1.0_f64, -1.0_f64, 2.0_f64,
        ])
        .unwrap();
        let flattened = flatten_cubes(&cube).unwrap();
        assert_eq!(flattened.dim(), (2, 4));
        assert!((flattened[[0, 0]] - 1.0_f64).abs() < 1e-12_f64);
        assert!((flattened[[0, 3]] - 4.0_f64).abs() < 1e-12_f64);
        assert!((flattened[[1, 1]] - 1.0_f64).abs() < 1e-12_f64);
        assert!((flattened[[1, 2]] + 1.0_f64).abs() < 1e-12_f64);
    }

    #[test]
    fn tensor_ops_reject_invalid_shapes() {
        let cube = Array3::<f64>::zeros((1, 2, 3));
        let vectors = Array2::<f64>::zeros((2, 3));
        assert!(matches!(cube_matvec(&cube, &vectors), Err(TensorError::DimensionMismatch)));

        let empty = Array3::<f64>::zeros((0, 0, 0));
        assert!(matches!(flatten_cubes(&empty), Err(TensorError::EmptyInput)));
    }

    #[test]
    fn arrayd_last_axis_ops_match_expected() {
        let tensor = ArrayD::from_shape_vec(IxDyn(&[2, 2, 3]), vec![
            1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 3.0_f64, 4.0_f64,
            1.0_f64, 2.0_f64, 2.0_f64,
        ])
        .unwrap();

        let sum = sum_last_axis(&tensor).unwrap();
        assert_eq!(sum.shape(), &[2, 2]);
        assert!((sum[[0, 0]] - 6.0_f64).abs() < 1e-12_f64);
        assert!((sum[[0, 1]] - 4.0_f64).abs() < 1e-12_f64);

        let norms = l2_norm_last_axis(&tensor).unwrap();
        assert_eq!(norms.shape(), &[2, 2]);
        assert!((norms[[0, 0]] - (14.0_f64).sqrt()).abs() < 1e-12_f64);
        assert!((norms[[0, 1]] - 4.0_f64).abs() < 1e-12_f64);

        let normalized = normalize_last_axis(&tensor).unwrap();
        let normalized_norms = l2_norm_last_axis(&normalized).unwrap();
        for value in &normalized_norms {
            assert!((value - 1.0_f64).abs() < 1e-10_f64);
        }
    }

    #[test]
    fn sum_last_axis_view_and_into_match_allocating_path() {
        let tensor = ArrayD::from_shape_vec(IxDyn(&[2, 2, 3]), vec![
            1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 3.0_f64, 4.0_f64,
            1.0_f64, 2.0_f64, 2.0_f64,
        ])
        .unwrap();

        let allocating = sum_last_axis(&tensor).unwrap();
        let viewed = sum_last_axis_view(&tensor.view()).unwrap();
        let mut output = ArrayD::<f64>::zeros(IxDyn(&[2, 2]));
        sum_last_axis_view_into(&tensor.view(), &mut output).unwrap();

        assert_eq!(allocating.shape(), viewed.shape());
        assert_eq!(allocating.shape(), output.shape());
        for ((lhs, rhs), into_value) in allocating.iter().zip(viewed.iter()).zip(output.iter()) {
            assert!((lhs - rhs).abs() < 1e-12_f64);
            assert!((lhs - into_value).abs() < 1e-12_f64);
        }
    }

    #[test]
    fn batched_dot_last_axis_matches_manual() {
        let left = ArrayD::from_shape_vec(IxDyn(&[2, 2, 3]), vec![
            1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 3.0_f64, 4.0_f64,
            1.0_f64, 2.0_f64, 2.0_f64,
        ])
        .unwrap();
        let right = ArrayD::from_shape_vec(IxDyn(&[2, 2, 3]), vec![
            0.5_f64, 1.0_f64, -1.0_f64, 0.0_f64, 2.0_f64, 3.0_f64, 1.0_f64, 1.0_f64, 1.0_f64,
            2.0_f64, 0.0_f64, 1.0_f64,
        ])
        .unwrap();

        let dots = batched_dot_last_axis(&left, &right).unwrap();
        assert_eq!(dots.shape(), &[2, 2]);
        assert!((dots[[0, 0]] - (0.5_f64 + 2.0_f64 - 3.0_f64)).abs() < 1e-12_f64);
        assert!((dots[[0, 1]] - 0.0_f64).abs() < 1e-12_f64);
        assert!((dots[[1, 0]] - 7.0_f64).abs() < 1e-12_f64);
        assert!((dots[[1, 1]] - 4.0_f64).abs() < 1e-12_f64);
    }

    #[test]
    fn arrayd_complex_last_axis_ops_match_expected() {
        let tensor = ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![
            Complex64::new(1.0_f64, 1.0_f64),
            Complex64::new(2.0_f64, 0.0_f64),
            Complex64::new(0.0_f64, 1.0_f64),
            Complex64::new(1.0_f64, -1.0_f64),
        ])
        .unwrap();

        let sum = sum_last_axis_complex(&tensor).unwrap();
        assert_eq!(sum.shape(), &[1, 2]);
        assert!((sum[[0, 0]] - Complex64::new(3.0_f64, 1.0_f64)).norm() < 1e-12_f64);

        let norms = l2_norm_last_axis_complex(&tensor).unwrap();
        assert_eq!(norms.shape(), &[1, 2]);
        assert!((norms[[0, 0]] - (6.0_f64).sqrt()).abs() < 1e-12_f64);

        let normalized = normalize_last_axis_complex(&tensor).unwrap();
        let normalized_norms = l2_norm_last_axis_complex(&normalized).unwrap();
        for value in &normalized_norms {
            assert!((value - 1.0_f64).abs() < 1e-10_f64);
        }
    }

    #[test]
    fn sum_last_axis_complex_view_and_into_match_allocating_path() {
        let tensor = ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![
            Complex64::new(1.0_f64, 1.0_f64),
            Complex64::new(2.0_f64, 0.0_f64),
            Complex64::new(0.0_f64, 1.0_f64),
            Complex64::new(1.0_f64, -1.0_f64),
        ])
        .unwrap();

        let allocating = sum_last_axis_complex(&tensor).unwrap();
        let viewed = sum_last_axis_complex_view(&tensor.view()).unwrap();
        let mut output = ArrayD::<Complex64>::zeros(IxDyn(&[1, 2]));
        sum_last_axis_complex_view_into(&tensor.view(), &mut output).unwrap();

        assert_eq!(allocating.shape(), viewed.shape());
        assert_eq!(allocating.shape(), output.shape());
        for ((lhs, rhs), into_value) in allocating.iter().zip(viewed.iter()).zip(output.iter()) {
            assert!((*lhs - *rhs).norm() < 1e-12_f64);
            assert!((*lhs - *into_value).norm() < 1e-12_f64);
        }
    }

    #[test]
    fn batched_dot_last_axis_complex_matches_manual() {
        let left = ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![
            Complex64::new(1.0_f64, 1.0_f64),
            Complex64::new(2.0_f64, 0.0_f64),
            Complex64::new(0.0_f64, 1.0_f64),
            Complex64::new(1.0_f64, -1.0_f64),
        ])
        .unwrap();
        let right = ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![
            Complex64::new(0.5_f64, 0.0_f64),
            Complex64::new(-1.0_f64, 1.0_f64),
            Complex64::new(2.0_f64, -1.0_f64),
            Complex64::new(0.0_f64, 1.0_f64),
        ])
        .unwrap();

        let dots = batched_dot_last_axis_complex(&left, &right).unwrap();
        assert_eq!(dots.shape(), &[1, 2]);

        let expected_00 =
            left[[0, 0, 0]].conj() * right[[0, 0, 0]] + left[[0, 0, 1]].conj() * right[[0, 0, 1]];
        let expected_01 =
            left[[0, 1, 0]].conj() * right[[0, 1, 0]] + left[[0, 1, 1]].conj() * right[[0, 1, 1]];
        assert!((dots[[0, 0]] - expected_00).norm() < 1e-12_f64);
        assert!((dots[[0, 1]] - expected_01).norm() < 1e-12_f64);
    }

    #[test]
    fn permute_axes_reorders_shape_and_values() {
        let tensor =
            ArrayD::from_shape_vec(IxDyn(&[2, 3, 4]), (0..24).map(f64::from).collect()).unwrap();
        let permuted = permute_axes(&tensor, &[1, 0, 2]).unwrap();
        assert_eq!(permuted.shape(), &[3, 2, 4]);
        assert!((permuted[[2, 1, 3]] - tensor[[1, 2, 3]]).abs() < 1e-12_f64);
    }

    #[test]
    fn contract_axes_matches_matrix_multiply() {
        let left = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![
            1.0_f64, 2.0_f64, 3.0_f64, //
            4.0_f64, 5.0_f64, 6.0_f64,
        ])
        .unwrap();
        let right = ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![
            7.0_f64, 8.0_f64, //
            9.0_f64, 10.0_f64, //
            11.0_f64, 12.0_f64,
        ])
        .unwrap();

        let contracted = contract_axes(&left, &right, &[1], &[0]).unwrap();
        assert_eq!(contracted.shape(), &[2, 2]);
        assert!((contracted[[0, 0]] - 58.0_f64).abs() < 1e-12_f64);
        assert!((contracted[[0, 1]] - 64.0_f64).abs() < 1e-12_f64);
        assert!((contracted[[1, 0]] - 139.0_f64).abs() < 1e-12_f64);
        assert!((contracted[[1, 1]] - 154.0_f64).abs() < 1e-12_f64);
    }

    #[test]
    fn contract_axes_into_matches_allocating_path() {
        let left =
            ArrayD::from_shape_vec(IxDyn(&[2, 2, 3]), (0..12).map(f64::from).collect()).unwrap();
        let right = ArrayD::from_shape_vec(
            IxDyn(&[2, 3, 4]),
            (0..24).map(|value| f64::from(value) * 0.5_f64).collect(),
        )
        .unwrap();

        let allocating = contract_axes(&left, &right, &[2], &[1]).unwrap();
        let mut into = ArrayD::<f64>::zeros(IxDyn(&[2, 2, 2, 4]));
        contract_axes_into(&left, &right, &[2], &[1], &mut into).unwrap();

        assert_eq!(allocating.shape(), into.shape());
        for (lhs, rhs) in allocating.iter().zip(into.iter()) {
            assert!((lhs - rhs).abs() < 1e-12_f64);
        }
    }

    #[test]
    fn contract_axes_view_variants_match_allocating_path() {
        let left =
            ArrayD::from_shape_vec(IxDyn(&[2, 2, 3]), (0..12).map(f64::from).collect()).unwrap();
        let right = ArrayD::from_shape_vec(
            IxDyn(&[2, 3, 4]),
            (0..24).map(|value| f64::from(value) * 0.5_f64).collect(),
        )
        .unwrap();

        let allocating = contract_axes(&left, &right, &[2], &[1]).unwrap();
        let viewed = contract_axes_view(&left.view(), &right.view(), &[2], &[1]).unwrap();
        let mut into = ArrayD::<f64>::zeros(IxDyn(&[2, 2, 2, 4]));
        contract_axes_view_into(&left.view(), &right.view(), &[2], &[1], &mut into).unwrap();

        for ((lhs, rhs), into_value) in allocating.iter().zip(viewed.iter()).zip(into.iter()) {
            assert!((lhs - rhs).abs() < 1e-12_f64);
            assert!((lhs - into_value).abs() < 1e-12_f64);
        }
    }

    #[test]
    fn batched_matmul_last_two_matches_cube_matmat() {
        let left = ArrayD::from_shape_vec(IxDyn(&[2, 2, 3]), vec![
            1.0_f64, 2.0_f64, 0.0_f64, 0.0_f64, 1.0_f64, 1.0_f64, //
            2.0_f64, 0.0_f64, 1.0_f64, 1.0_f64, 3.0_f64, 2.0_f64,
        ])
        .unwrap();
        let right = ArrayD::from_shape_vec(IxDyn(&[2, 3, 2]), vec![
            1.0_f64, 0.0_f64, 2.0_f64, 1.0_f64, 1.0_f64, 3.0_f64, //
            0.0_f64, 2.0_f64, 1.0_f64, 1.0_f64, 3.0_f64, 0.0_f64,
        ])
        .unwrap();

        let nd_output = batched_matmul_last_two(&left, &right).unwrap();
        let cube_output = cube_matmat(
            &left.clone().into_dimensionality().unwrap(),
            &right.clone().into_dimensionality().unwrap(),
        )
        .unwrap()
        .into_dyn();

        for (lhs, rhs) in nd_output.iter().zip(cube_output.iter()) {
            assert!((lhs - rhs).abs() < 1e-12_f64);
        }
    }

    #[test]
    fn batched_matmul_last_two_into_matches_allocating_path() {
        let left =
            ArrayD::from_shape_vec(IxDyn(&[2, 2, 2, 3]), (0..24).map(f64::from).collect()).unwrap();
        let right = ArrayD::from_shape_vec(
            IxDyn(&[2, 2, 3, 2]),
            (0..24).map(|value| f64::from(value) * 0.25_f64).collect(),
        )
        .unwrap();

        let allocating = batched_matmul_last_two(&left, &right).unwrap();
        let mut into = ArrayD::<f64>::zeros(IxDyn(&[2, 2, 2, 2]));
        batched_matmul_last_two_into(&left, &right, &mut into).unwrap();

        for (lhs, rhs) in allocating.iter().zip(into.iter()) {
            assert!((lhs - rhs).abs() < 1e-12_f64);
        }
    }

    #[test]
    fn batched_matmul_last_two_view_variants_match_allocating_path() {
        let left =
            ArrayD::from_shape_vec(IxDyn(&[2, 2, 2, 3]), (0..24).map(f64::from).collect()).unwrap();
        let right = ArrayD::from_shape_vec(
            IxDyn(&[2, 2, 3, 2]),
            (0..24).map(|value| f64::from(value) * 0.25_f64).collect(),
        )
        .unwrap();

        let allocating = batched_matmul_last_two(&left, &right).unwrap();
        let viewed = batched_matmul_last_two_view(&left.view(), &right.view()).unwrap();
        let mut into = ArrayD::<f64>::zeros(IxDyn(&[2, 2, 2, 2]));
        batched_matmul_last_two_view_into(&left.view(), &right.view(), &mut into).unwrap();

        for ((lhs, rhs), into_value) in allocating.iter().zip(viewed.iter()).zip(into.iter()) {
            assert!((lhs - rhs).abs() < 1e-12_f64);
            assert!((lhs - into_value).abs() < 1e-12_f64);
        }
    }

    #[test]
    fn complex_contract_and_batched_matmul_paths_work() {
        let left = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![
            Complex64::new(1.0_f64, 1.0_f64),
            Complex64::new(2.0_f64, 0.0_f64),
            Complex64::new(0.0_f64, -1.0_f64),
            Complex64::new(1.0_f64, 2.0_f64),
        ])
        .unwrap();
        let right = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![
            Complex64::new(0.0_f64, 1.0_f64),
            Complex64::new(1.0_f64, 0.0_f64),
            Complex64::new(2.0_f64, -1.0_f64),
            Complex64::new(-1.0_f64, 1.0_f64),
        ])
        .unwrap();

        let contract = contract_axes_complex(&left, &right, &[1], &[0]).unwrap();
        assert_eq!(contract.shape(), &[2, 2]);

        let left_batch = left.clone().into_shape_with_order(IxDyn(&[1, 2, 2])).unwrap();
        let right_batch = right.clone().into_shape_with_order(IxDyn(&[1, 2, 2])).unwrap();
        let matmul = batched_matmul_last_two_complex(&left_batch, &right_batch).unwrap();
        assert_eq!(matmul.shape(), &[1, 2, 2]);

        let mut into = ArrayD::<Complex64>::zeros(IxDyn(&[1, 2, 2]));
        batched_matmul_last_two_complex_into(&left_batch, &right_batch, &mut into).unwrap();
        for (lhs, rhs) in matmul.iter().zip(into.iter()) {
            assert!((*lhs - *rhs).norm() < 1e-12_f64);
        }
    }

    #[test]
    fn complex_contract_and_batched_matmul_view_variants_match() {
        let left = ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![
            Complex64::new(1.0_f64, 1.0_f64),
            Complex64::new(2.0_f64, 0.0_f64),
            Complex64::new(0.0_f64, -1.0_f64),
            Complex64::new(1.0_f64, 2.0_f64),
        ])
        .unwrap();
        let right = ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![
            Complex64::new(0.0_f64, 1.0_f64),
            Complex64::new(1.0_f64, 0.0_f64),
            Complex64::new(2.0_f64, -1.0_f64),
            Complex64::new(-1.0_f64, 1.0_f64),
        ])
        .unwrap();

        let allocating_contract = contract_axes_complex(&left, &right, &[2], &[1]).unwrap();
        let viewed_contract =
            contract_axes_complex_view(&left.view(), &right.view(), &[2], &[1]).unwrap();
        let mut contract_into = ArrayD::<Complex64>::zeros(IxDyn(&[1, 2, 1, 2]));
        contract_axes_complex_view_into(
            &left.view(),
            &right.view(),
            &[2],
            &[1],
            &mut contract_into,
        )
        .unwrap();
        for ((lhs, rhs), into_value) in
            allocating_contract.iter().zip(viewed_contract.iter()).zip(contract_into.iter())
        {
            assert!((*lhs - *rhs).norm() < 1e-12_f64);
            assert!((*lhs - *into_value).norm() < 1e-12_f64);
        }

        let allocating_matmul = batched_matmul_last_two_complex(&left, &right).unwrap();
        let viewed_matmul =
            batched_matmul_last_two_complex_view(&left.view(), &right.view()).unwrap();
        let mut matmul_into = ArrayD::<Complex64>::zeros(IxDyn(&[1, 2, 2]));
        batched_matmul_last_two_complex_view_into(&left.view(), &right.view(), &mut matmul_into)
            .unwrap();
        for ((lhs, rhs), into_value) in
            allocating_matmul.iter().zip(viewed_matmul.iter()).zip(matmul_into.iter())
        {
            assert!((*lhs - *rhs).norm() < 1e-12_f64);
            assert!((*lhs - *into_value).norm() < 1e-12_f64);
        }
    }

    #[test]
    fn einsum_matches_matrix_multiply_and_batch_path() {
        let left = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![
            1.0_f64, 2.0_f64, 3.0_f64, //
            4.0_f64, 5.0_f64, 6.0_f64,
        ])
        .unwrap();
        let right = ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![
            7.0_f64, 8.0_f64, //
            9.0_f64, 10.0_f64, //
            11.0_f64, 12.0_f64,
        ])
        .unwrap();
        let product = einsum("ab,bc->ac", &left, &right).unwrap();
        assert_eq!(product.shape(), &[2, 2]);
        assert!((product[[0, 0]] - 58.0_f64).abs() < 1e-12_f64);
        assert!((product[[0, 1]] - 64.0_f64).abs() < 1e-12_f64);
        assert!((product[[1, 0]] - 139.0_f64).abs() < 1e-12_f64);
        assert!((product[[1, 1]] - 154.0_f64).abs() < 1e-12_f64);

        let left_batch = ArrayD::from_shape_vec(IxDyn(&[2, 2, 3]), vec![
            1.0_f64, 2.0_f64, 0.0_f64, 0.0_f64, 1.0_f64, 1.0_f64, //
            2.0_f64, 0.0_f64, 1.0_f64, 1.0_f64, 3.0_f64, 2.0_f64,
        ])
        .unwrap();
        let right_batch = ArrayD::from_shape_vec(IxDyn(&[2, 3, 2]), vec![
            1.0_f64, 0.0_f64, 2.0_f64, 1.0_f64, 1.0_f64, 3.0_f64, //
            0.0_f64, 2.0_f64, 1.0_f64, 1.0_f64, 3.0_f64, 0.0_f64,
        ])
        .unwrap();
        let batch_product = einsum("bij,bjk->bik", &left_batch, &right_batch).unwrap();
        let nd_output = batched_matmul_last_two(&left_batch, &right_batch).unwrap();
        for (lhs, rhs) in batch_product.iter().zip(nd_output.iter()) {
            assert!((lhs - rhs).abs() < 1e-12_f64);
        }
    }

    #[test]
    fn complex_einsum_matches_manual() {
        let left = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![
            Complex64::new(1.0_f64, 1.0_f64),
            Complex64::new(2.0_f64, 0.0_f64),
            Complex64::new(0.0_f64, -1.0_f64),
            Complex64::new(1.0_f64, 2.0_f64),
        ])
        .unwrap();
        let right = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![
            Complex64::new(0.0_f64, 1.0_f64),
            Complex64::new(1.0_f64, 0.0_f64),
            Complex64::new(2.0_f64, -1.0_f64),
            Complex64::new(-1.0_f64, 1.0_f64),
        ])
        .unwrap();
        let product = einsum_complex("ab,bc->ac", &left, &right).unwrap();
        let reference = contract_axes_complex(&left, &right, &[1], &[0]).unwrap();
        for (lhs, rhs) in product.iter().zip(reference.iter()) {
            assert!((*lhs - *rhs).norm() < 1e-12_f64);
        }
    }

    #[test]
    fn hosvd3_roundtrip_is_consistent() {
        let cube = Array3::from_shape_vec((3, 3, 2), vec![
            1.0_f64, 0.5_f64, 2.0_f64, -1.0_f64, 0.0_f64, 1.0_f64, 0.0_f64, 2.0_f64, 1.0_f64,
            1.5_f64, 3.0_f64, 0.0_f64, -1.0_f64, 1.0_f64, 2.5_f64, -0.5_f64, 0.5_f64, 2.0_f64,
        ])
        .unwrap();
        let decomposition = hosvd3(&cube, (3, 3, 2)).unwrap();
        let reconstructed = hosvd3_reconstruct(&decomposition).unwrap();
        assert_eq!(reconstructed.dim(), cube.dim());
        for (lhs, rhs) in reconstructed.iter().zip(cube.iter()) {
            assert!((lhs - rhs).abs() < 1e-8_f64);
        }
    }

    #[test]
    fn arrayd_ops_reject_invalid_dimensions() {
        let scalar = ArrayD::from_shape_vec(IxDyn(&[]), vec![1.0_f64]).unwrap();
        assert!(matches!(sum_last_axis(&scalar), Err(TensorError::DimensionMismatch)));

        let left = ArrayD::from_shape_vec(IxDyn(&[2, 2, 3]), vec![1.0_f64; 12]).unwrap();
        let right = ArrayD::from_shape_vec(IxDyn(&[2, 2, 2]), vec![1.0_f64; 8]).unwrap();
        assert!(matches!(
            batched_dot_last_axis(&left, &right),
            Err(TensorError::DimensionMismatch)
        ));

        let bad_permutation = permute_axes(&left, &[0, 0, 1]);
        assert!(matches!(bad_permutation, Err(TensorError::DimensionMismatch)));

        let bad_contract = contract_axes(&left, &right, &[2], &[1]);
        assert!(matches!(bad_contract, Err(TensorError::DimensionMismatch)));

        let bad_einsum = einsum("ab,bc->ad", &left, &right);
        assert!(matches!(bad_einsum, Err(TensorError::DimensionMismatch)));

        let mut bad_output = ArrayD::<f64>::zeros(IxDyn(&[2, 2, 3]));
        let matmul_into = batched_matmul_last_two_into(&left, &left, &mut bad_output);
        assert!(matches!(matmul_into, Err(TensorError::DimensionMismatch)));

        let cube = Array3::<f64>::zeros((2, 2, 2));
        let bad_hosvd = hosvd3(&cube, (3, 1, 1));
        assert!(matches!(bad_hosvd, Err(TensorError::DimensionMismatch)));
    }
}
