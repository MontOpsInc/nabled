//! Tensor and cube primitives over ndarray higher-rank arrays.

use std::fmt;
use std::ops::{AddAssign, Mul};

use nabled_core::scalar::NabledReal;
use ndarray::{
    Array1, Array2, Array3, ArrayBase, ArrayD, ArrayView1, ArrayView2, ArrayView3, ArrayViewD,
    ArrayViewMut2, ArrayViewMut3, Axis, Data, DataMut, Ix2, Ix3, IxDyn, s,
};
use num_complex::Complex64;

#[cfg(feature = "accelerator-wgpu")]
use crate::accelerator::backends::GpuBackend;
use crate::accelerator::backends::{AcceleratorError, CpuBackend};
use crate::accelerator::dispatch::{
    tensor_batched_matmul_last_two_cpu, tensor_batched_matmul_last_two_with_backend,
    tensor_contract_axes_cpu, tensor_contract_axes_with_backend, tensor_sum_last_axis_cpu,
    tensor_sum_last_axis_with_backend,
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

fn tensor_contract_axes_complex_dispatch(
    left: &ArrayD<Complex64>,
    right: &ArrayD<Complex64>,
    left_axis: usize,
    right_axis: usize,
) -> Result<ArrayD<Complex64>, TensorError> {
    #[cfg(feature = "accelerator-wgpu")]
    {
        tensor_contract_axes_with_backend::<GpuBackend, Complex64>(
            left, right, left_axis, right_axis,
        )
        .or_else(|_| {
            tensor_contract_axes_with_backend::<CpuBackend, Complex64>(
                left, right, left_axis, right_axis,
            )
        })
        .map_err(map_accelerator_error_to_tensor)
    }
    #[cfg(not(feature = "accelerator-wgpu"))]
    {
        tensor_contract_axes_with_backend::<CpuBackend, Complex64>(
            left, right, left_axis, right_axis,
        )
        .map_err(map_accelerator_error_to_tensor)
    }
}

fn tensor_batched_matmul_last_two_complex_dispatch(
    left: &ArrayD<Complex64>,
    right: &ArrayD<Complex64>,
) -> Result<ArrayD<Complex64>, TensorError> {
    #[cfg(feature = "accelerator-wgpu")]
    {
        tensor_batched_matmul_last_two_with_backend::<GpuBackend, Complex64>(left, right)
            .or_else(|_| {
                tensor_batched_matmul_last_two_with_backend::<CpuBackend, Complex64>(left, right)
            })
            .map_err(map_accelerator_error_to_tensor)
    }
    #[cfg(not(feature = "accelerator-wgpu"))]
    {
        tensor_batched_matmul_last_two_with_backend::<CpuBackend, Complex64>(left, right)
            .map_err(map_accelerator_error_to_tensor)
    }
}

fn tensor_sum_last_axis_complex_dispatch(
    input: &ArrayD<Complex64>,
) -> Result<ArrayD<Complex64>, TensorError> {
    #[cfg(feature = "accelerator-wgpu")]
    {
        tensor_sum_last_axis_with_backend::<GpuBackend, Complex64>(input)
            .or_else(|_| tensor_sum_last_axis_with_backend::<CpuBackend, Complex64>(input))
            .map_err(map_accelerator_error_to_tensor)
    }
    #[cfg(not(feature = "accelerator-wgpu"))]
    {
        tensor_sum_last_axis_with_backend::<CpuBackend, Complex64>(input)
            .map_err(map_accelerator_error_to_tensor)
    }
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

/// Configuration for rank-`R` CP-ALS decomposition over rank-3 tensors.
#[derive(Debug, Clone)]
pub struct CpAlsConfig<T: NabledReal = f64> {
    /// Maximum ALS sweeps before returning the current iterate.
    pub max_iterations: usize,
    /// Relative factor-change tolerance for convergence.
    pub tolerance:      T,
}

impl Default for CpAlsConfig<f64> {
    fn default() -> Self { Self { max_iterations: 200, tolerance: 1.0e-8 } }
}

impl Default for CpAlsConfig<f32> {
    fn default() -> Self { Self { max_iterations: 200, tolerance: 1.0e-5 } }
}

/// CP decomposition result for rank-3 real tensors.
#[derive(Debug, Clone)]
pub struct CpAls3Result<T: NabledReal = f64> {
    /// Rank weights (`R`).
    pub weights:  Array1<T>,
    /// Mode-0 factor matrix (`I0 x R`).
    pub factor_0: Array2<T>,
    /// Mode-1 factor matrix (`I1 x R`).
    pub factor_1: Array2<T>,
    /// Mode-2 factor matrix (`I2 x R`).
    pub factor_2: Array2<T>,
}

/// CP decomposition result for rank-`N` real tensors.
#[derive(Debug, Clone)]
pub struct CpAlsNdResult<T: NabledReal = f64> {
    /// Rank weights (`R`).
    pub weights: Array1<T>,
    /// Per-mode factor matrices (`I_mode x R`), length `N`.
    pub factors: Vec<Array2<T>>,
    /// Original tensor shape `(I_0, ..., I_{N-1})`.
    pub shape:   Vec<usize>,
}

/// CP decomposition reconstruction quality metrics.
#[derive(Debug, Clone)]
pub struct CpErrorMetrics<T: NabledReal = f64> {
    /// L2 norm of the reference tensor.
    pub signal_norm:    T,
    /// L2 norm of residual (`reference - reconstruction`).
    pub residual_norm:  T,
    /// Relative residual (`residual_norm / signal_norm` when `signal_norm > 0`).
    pub relative_error: T,
    /// Fit score (`1 - relative_error`, clamped to `[0, 1]` when defined).
    pub fit:            T,
}

/// CP-ALS convergence summary for a decomposition run.
#[derive(Debug, Clone)]
pub struct CpConvergenceReport<T: NabledReal = f64> {
    /// Number of ALS sweeps executed.
    pub iterations_run:          usize,
    /// Whether convergence tolerance was reached before iteration limit.
    pub converged:               bool,
    /// Maximum factor-relative-change observed on the final sweep.
    pub final_max_factor_change: T,
}

/// CP-ALS decomposition report bundling convergence and reconstruction metrics.
#[derive(Debug, Clone)]
pub struct CpAlsReport<T: NabledReal = f64> {
    /// ALS convergence summary.
    pub convergence: CpConvergenceReport<T>,
    /// Reconstruction metrics against the input tensor.
    pub metrics:     CpErrorMetrics<T>,
}

/// HOSVD/Tucker decomposition result for rank-`N` real tensors.
#[derive(Debug, Clone)]
pub struct HosvdNdResult<T: NabledReal = f64> {
    /// Core tensor with shape `ranks`.
    pub core:    ArrayD<T>,
    /// Per-mode factor matrices (`I_mode x R_mode`), length `N`.
    pub factors: Vec<Array2<T>>,
}

/// Configuration for `N`-D HOOI Tucker refinement.
#[derive(Debug, Clone)]
pub struct HooiConfig<T: NabledReal = f64> {
    /// Maximum refinement sweeps.
    pub max_iterations: usize,
    /// Relative core-change tolerance used for convergence.
    pub tolerance:      T,
}

impl Default for HooiConfig<f64> {
    fn default() -> Self { Self { max_iterations: 50, tolerance: 1.0e-8 } }
}

impl Default for HooiConfig<f32> {
    fn default() -> Self { Self { max_iterations: 50, tolerance: 1.0e-5 } }
}

/// Configuration for Tensor-Train decomposition via TT-SVD.
#[derive(Debug, Clone)]
pub struct TtSvdConfig<T: NabledReal = f64> {
    /// Optional global maximum TT rank (`None` means unconstrained).
    pub max_rank:  Option<usize>,
    /// Relative singular-value cutoff used per unfolding.
    pub tolerance: T,
}

impl Default for TtSvdConfig<f64> {
    fn default() -> Self { Self { max_rank: None, tolerance: 1.0e-8 } }
}

impl Default for TtSvdConfig<f32> {
    fn default() -> Self { Self { max_rank: None, tolerance: 1.0e-5 } }
}

/// Configuration for Tensor-Train rank truncation/rounding.
#[derive(Debug, Clone)]
pub struct TtRoundConfig<T: NabledReal = f64> {
    /// Optional global maximum TT rank (`None` means unconstrained).
    pub max_rank:  Option<usize>,
    /// Relative singular-value cutoff used when truncating intermediate ranks.
    pub tolerance: T,
}

impl Default for TtRoundConfig<f64> {
    fn default() -> Self { Self { max_rank: None, tolerance: 1.0e-8 } }
}

impl Default for TtRoundConfig<f32> {
    fn default() -> Self { Self { max_rank: None, tolerance: 1.0e-5 } }
}

/// Tensor-Train decomposition result for rank-`N` real tensors.
#[derive(Debug, Clone)]
pub struct TensorTrainResult<T: NabledReal = f64> {
    /// Per-mode TT cores with shape `(r_k, n_k, r_{k+1})`.
    pub cores: Vec<Array3<T>>,
    /// Original tensor shape `(n_0, ..., n_{N-1})`.
    pub shape: Vec<usize>,
}

type EinsumOperands = (Vec<char>, Vec<char>, Vec<char>);

#[cfg(feature = "lapack-provider")]
#[doc(hidden)]
pub trait CpAlsScalar: NabledReal + ndarray_linalg::Lapack<Real = Self> + AddAssign {}

#[cfg(feature = "lapack-provider")]
impl<T> CpAlsScalar for T where T: NabledReal + ndarray_linalg::Lapack<Real = T> + AddAssign {}

#[cfg(not(feature = "lapack-provider"))]
#[doc(hidden)]
pub trait CpAlsScalar: svd::SvdInternalScalar {}

#[cfg(not(feature = "lapack-provider"))]
impl<T> CpAlsScalar for T where T: svd::SvdInternalScalar {}

#[cfg(feature = "lapack-provider")]
#[doc(hidden)]
pub trait HosvdNdScalar: NabledReal + ndarray_linalg::Lapack<Real = Self> + AddAssign {}

#[cfg(feature = "lapack-provider")]
impl<T> HosvdNdScalar for T where T: NabledReal + ndarray_linalg::Lapack<Real = T> + AddAssign {}

#[cfg(not(feature = "lapack-provider"))]
#[doc(hidden)]
pub trait HosvdNdScalar: svd::SvdInternalScalar {}

#[cfg(not(feature = "lapack-provider"))]
impl<T> HosvdNdScalar for T where T: svd::SvdInternalScalar {}

#[cfg(feature = "lapack-provider")]
#[doc(hidden)]
pub trait HooiNdScalar: NabledReal + ndarray_linalg::Lapack<Real = Self> + AddAssign {}

#[cfg(feature = "lapack-provider")]
impl<T> HooiNdScalar for T where T: NabledReal + ndarray_linalg::Lapack<Real = T> + AddAssign {}

#[cfg(not(feature = "lapack-provider"))]
#[doc(hidden)]
pub trait HooiNdScalar: svd::SvdInternalScalar {}

#[cfg(not(feature = "lapack-provider"))]
impl<T> HooiNdScalar for T where T: svd::SvdInternalScalar {}

#[cfg(feature = "lapack-provider")]
#[doc(hidden)]
pub trait TtSvdScalar: NabledReal + ndarray_linalg::Lapack<Real = Self> + AddAssign {}

#[cfg(feature = "lapack-provider")]
impl<T> TtSvdScalar for T where T: NabledReal + ndarray_linalg::Lapack<Real = T> + AddAssign {}

#[cfg(not(feature = "lapack-provider"))]
#[doc(hidden)]
pub trait TtSvdScalar: svd::SvdInternalScalar {}

#[cfg(not(feature = "lapack-provider"))]
impl<T> TtSvdScalar for T where T: svd::SvdInternalScalar {}

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

fn unflatten_prefix_index(mut flat: usize, shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }

    let mut index = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        index[axis] = flat % shape[axis];
        flat /= shape[axis];
    }
    index
}

pub(crate) fn contract_view_into_impl<T, S>(
    left: &ArrayViewD<'_, T>,
    right: &ArrayViewD<'_, T>,
    left_axes: &[usize],
    right_axes: &[usize],
    output: &mut ArrayBase<S, IxDyn>,
) -> Result<(), TensorError>
where
    T: Copy + Default + AddAssign + Mul<Output = T>,
    S: DataMut<Elem = T>,
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
    let left_free_shape = left_free_axes.iter().map(|axis| left.shape()[*axis]).collect::<Vec<_>>();
    let right_free_shape =
        right_free_axes.iter().map(|axis| right.shape()[*axis]).collect::<Vec<_>>();

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

    for i in 0..left_outer {
        let left_index = unflatten_prefix_index(i, &left_free_shape);
        let left_len = left_index.len();
        let mut output_index = left_index;
        output_index.resize(left_len + right_free_shape.len(), 0);

        for j in 0..right_outer {
            let right_index = unflatten_prefix_index(j, &right_free_shape);
            if !right_index.is_empty() {
                output_index[left_len..].copy_from_slice(&right_index);
            }

            let mut accumulated = T::default();
            for k in 0..left_width {
                accumulated += left_2d[[i, k]] * right_2d[[k, j]];
            }
            output[IxDyn(&output_index)] = accumulated;
        }
    }

    Ok(())
}

pub(crate) fn batched_matmul_last_two_view_into_impl<T, S>(
    left: &ArrayViewD<'_, T>,
    right: &ArrayViewD<'_, T>,
    output: &mut ArrayBase<S, IxDyn>,
) -> Result<(), TensorError>
where
    T: Copy + Default + AddAssign + Mul<Output = T>,
    S: DataMut<Elem = T>,
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
    output.fill(T::default());

    for batch in 0..batches {
        let batch_index = unflatten_prefix_index(batch, &left.shape()[..batch_ndim]);
        let mut output_index = batch_index;
        output_index.push(0);
        output_index.push(0);
        for row in 0..rows {
            let row_axis = output_index.len() - 2;
            let col_axis = output_index.len() - 1;
            output_index[row_axis] = row;
            for k in 0..inner {
                let lhs = left_3d[[batch, row, k]];
                for col in 0..cols {
                    output_index[col_axis] = col;
                    output[IxDyn(&output_index)] += lhs * right_3d[[batch, k, col]];
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
    flatten_cubes_view(&cube.view())
}

/// Flatten each cube slice `(rows, cols)` into one row from a view.
///
/// Input `(batch, rows, cols)` becomes `(batch, rows * cols)`.
///
/// # Errors
/// Returns an error if input is empty.
pub fn flatten_cubes_view<T: NabledReal>(
    cube: &ArrayView3<'_, T>,
) -> Result<Array2<T>, TensorError> {
    validate_cube_non_empty(cube)?;

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
pub fn sum_last_axis_view_into<T, S>(
    tensor: &ArrayViewD<'_, T>,
    output: &mut ArrayBase<S, IxDyn>,
) -> Result<(), TensorError>
where
    T: NabledReal,
    S: DataMut<Elem = T>,
{
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
    l2_norm_last_axis_view(&tensor.view())
}

/// Compute L2 norm along the last axis of a tensor view.
///
/// # Errors
/// Returns an error if tensor is empty or has zero dimensions.
pub fn l2_norm_last_axis_view<T: NabledReal>(
    tensor: &ArrayViewD<'_, T>,
) -> Result<ArrayD<T>, TensorError> {
    let tensor_view = tensor.view();
    validate_tensor_nd_non_empty(&tensor_view)?;

    let mut output_shape = tensor_view.shape().to_vec();
    let _ = output_shape.pop();
    let mut output = ArrayD::<T>::zeros(IxDyn(&output_shape));
    l2_norm_last_axis_view_into(tensor, &mut output)?;
    Ok(output)
}

/// Compute L2 norm along the last axis of a tensor view into `output`.
///
/// # Errors
/// Returns an error if tensor is empty, has zero dimensions, or output shape mismatches.
pub fn l2_norm_last_axis_view_into<T, S>(
    tensor: &ArrayViewD<'_, T>,
    output: &mut ArrayBase<S, IxDyn>,
) -> Result<(), TensorError>
where
    T: NabledReal,
    S: DataMut<Elem = T>,
{
    let tensor_view = tensor.view();
    validate_tensor_nd_non_empty(&tensor_view)?;

    let axis = Axis(tensor_view.ndim() - 1);
    let mut output_shape = tensor_view.shape().to_vec();
    let _ = output_shape.pop();
    if output.shape() != output_shape.as_slice() {
        return Err(TensorError::DimensionMismatch);
    }
    for (out_value, lane) in output.iter_mut().zip(tensor_view.lanes(axis)) {
        let sum_sq = lane
            .iter()
            .copied()
            .map(|value| value * value)
            .fold(T::zero(), |acc, value| acc + value);
        *out_value = sum_sq.sqrt();
    }
    Ok(())
}

/// Normalize tensor values along the last axis.
///
/// # Errors
/// Returns an error if tensor is empty or has zero dimensions.
pub fn normalize_last_axis<T: NabledReal>(tensor: &ArrayD<T>) -> Result<ArrayD<T>, TensorError> {
    normalize_last_axis_view(&tensor.view())
}

/// Normalize tensor values along the last axis from a tensor view.
///
/// # Errors
/// Returns an error if tensor is empty or has zero dimensions.
pub fn normalize_last_axis_view<T: NabledReal>(
    tensor: &ArrayViewD<'_, T>,
) -> Result<ArrayD<T>, TensorError> {
    let tensor_view = tensor.view();
    validate_tensor_nd_non_empty(&tensor_view)?;

    let mut output = tensor.to_owned();
    normalize_last_axis_view_into(tensor, &mut output)?;
    Ok(output)
}

/// Normalize tensor values along the last axis into `output`.
///
/// # Errors
/// Returns an error if tensor is empty, has zero dimensions, or output shape mismatches.
pub fn normalize_last_axis_view_into<T, S>(
    tensor: &ArrayViewD<'_, T>,
    output: &mut ArrayBase<S, IxDyn>,
) -> Result<(), TensorError>
where
    T: NabledReal,
    S: DataMut<Elem = T>,
{
    let tensor_view = tensor.view();
    validate_tensor_nd_non_empty(&tensor_view)?;
    if output.shape() != tensor_view.shape() {
        return Err(TensorError::DimensionMismatch);
    }

    output.assign(&tensor_view);
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
    Ok(())
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
    batched_dot_last_axis_view(&left.view(), &right.view())
}

/// Compute batched dot products along the last axis of two tensor views.
///
/// The input tensors must have identical shape and `ndim >= 1`.
/// Output shape is the input shape without the last axis.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn batched_dot_last_axis_view<T: NabledReal>(
    left: &ArrayViewD<'_, T>,
    right: &ArrayViewD<'_, T>,
) -> Result<ArrayD<T>, TensorError> {
    let left_view = left.view();
    let right_view = right.view();
    validate_tensor_nd_non_empty(&left_view)?;
    validate_tensor_nd_non_empty(&right_view)?;
    if left_view.shape() != right_view.shape() {
        return Err(TensorError::DimensionMismatch);
    }

    let mut output_shape = left_view.shape().to_vec();
    let _ = output_shape.pop();
    let mut output = ArrayD::<T>::zeros(IxDyn(&output_shape));
    batched_dot_last_axis_view_into(left, right, &mut output)?;
    Ok(output)
}

/// Compute batched dot products along the last axis of two tensor views into `output`.
///
/// # Errors
/// Returns an error if inputs are empty, dimensions are incompatible, or output shape mismatches.
pub fn batched_dot_last_axis_view_into<T, S>(
    left: &ArrayViewD<'_, T>,
    right: &ArrayViewD<'_, T>,
    output: &mut ArrayBase<S, IxDyn>,
) -> Result<(), TensorError>
where
    T: NabledReal,
    S: DataMut<Elem = T>,
{
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
    if output.shape() != output_shape.as_slice() {
        return Err(TensorError::DimensionMismatch);
    }
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
    Ok(())
}

/// Permute tensor axes using an explicit axis ordering.
///
/// # Errors
/// Returns an error if the tensor is empty, has zero dimensions, or permutation is invalid.
pub fn permute_axes<T: NabledReal>(
    tensor: &ArrayD<T>,
    permutation: &[usize],
) -> Result<ArrayD<T>, TensorError> {
    permute_axes_view(&tensor.view(), permutation)
}

/// Permute tensor axes from a tensor view using an explicit axis ordering.
///
/// # Errors
/// Returns an error if the tensor is empty, has zero dimensions, or permutation is invalid.
pub fn permute_axes_view<T: NabledReal>(
    tensor: &ArrayViewD<'_, T>,
    permutation: &[usize],
) -> Result<ArrayD<T>, TensorError> {
    let tensor_view = tensor.view();
    validate_tensor_nd_non_empty(&tensor_view)?;
    if !validate_permutation(tensor_view.ndim(), permutation) {
        return Err(TensorError::DimensionMismatch);
    }
    let mut output = ArrayD::<T>::zeros(IxDyn(
        &permutation.iter().map(|axis| tensor_view.shape()[*axis]).collect::<Vec<_>>(),
    ));
    permute_axes_view_into(tensor, permutation, &mut output)?;
    Ok(output)
}

/// Permute tensor axes from a tensor view into `output`.
///
/// # Errors
/// Returns an error if the tensor is empty, has zero dimensions, permutation is invalid, or the
/// output shape mismatches.
pub fn permute_axes_view_into<T, S>(
    tensor: &ArrayViewD<'_, T>,
    permutation: &[usize],
    output: &mut ArrayBase<S, IxDyn>,
) -> Result<(), TensorError>
where
    T: NabledReal,
    S: DataMut<Elem = T>,
{
    let tensor_view = tensor.view();
    validate_tensor_nd_non_empty(&tensor_view)?;
    if !validate_permutation(tensor_view.ndim(), permutation) {
        return Err(TensorError::DimensionMismatch);
    }
    let permuted = tensor_view.permuted_axes(permutation.to_vec());
    if output.shape() != permuted.shape() {
        return Err(TensorError::DimensionMismatch);
    }
    output.assign(&permuted);
    Ok(())
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
pub fn contract_axes_view_into<T, S>(
    left: &ArrayViewD<'_, T>,
    right: &ArrayViewD<'_, T>,
    left_axes: &[usize],
    right_axes: &[usize],
    output: &mut ArrayBase<S, IxDyn>,
) -> Result<(), TensorError>
where
    T: NabledReal + Default,
    S: DataMut<Elem = T>,
{
    let left_view = left.view();
    let right_view = right.view();
    validate_tensor_nd_non_empty(&left_view)?;
    validate_tensor_nd_non_empty(&right_view)?;
    let mut output_view = output.view_mut();
    contract_view_into_impl(&left_view, &right_view, left_axes, right_axes, &mut output_view)
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
pub fn batched_matmul_last_two_view_into<T, S>(
    left: &ArrayViewD<'_, T>,
    right: &ArrayViewD<'_, T>,
    output: &mut ArrayBase<S, IxDyn>,
) -> Result<(), TensorError>
where
    T: NabledReal + Default,
    S: DataMut<Elem = T>,
{
    let left_view = left.view();
    let right_view = right.view();
    validate_tensor_nd_non_empty(&left_view)?;
    validate_tensor_nd_non_empty(&right_view)?;
    let mut output_view = output.view_mut();
    batched_matmul_last_two_view_into_impl(&left_view, &right_view, &mut output_view)
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
    tensor_sum_last_axis_complex_dispatch(&owned)
}

/// Reduce a complex tensor view along its last axis by summation into `output`.
///
/// # Errors
/// Returns an error if tensor is empty, has zero dimensions, or output shape mismatches.
pub fn sum_last_axis_complex_view_into<S>(
    tensor: &ArrayViewD<'_, Complex64>,
    output: &mut ArrayBase<S, IxDyn>,
) -> Result<(), TensorError>
where
    S: DataMut<Elem = Complex64>,
{
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
    l2_norm_last_axis_complex_view(&tensor.view())
}

/// Compute L2 norm along the last axis of a complex tensor view.
///
/// # Errors
/// Returns an error if tensor is empty or has zero dimensions.
pub fn l2_norm_last_axis_complex_view(
    tensor: &ArrayViewD<'_, Complex64>,
) -> Result<ArrayD<f64>, TensorError> {
    let tensor_view = tensor.view();
    validate_tensor_nd_non_empty_complex(&tensor_view)?;

    let mut output_shape = tensor_view.shape().to_vec();
    let _ = output_shape.pop();
    let mut output = ArrayD::<f64>::zeros(IxDyn(&output_shape));
    l2_norm_last_axis_complex_view_into(tensor, &mut output)?;
    Ok(output)
}

/// Compute L2 norm along the last axis of a complex tensor view into `output`.
///
/// # Errors
/// Returns an error if tensor is empty, has zero dimensions, or output shape mismatches.
pub fn l2_norm_last_axis_complex_view_into<S>(
    tensor: &ArrayViewD<'_, Complex64>,
    output: &mut ArrayBase<S, IxDyn>,
) -> Result<(), TensorError>
where
    S: DataMut<Elem = f64>,
{
    let tensor_view = tensor.view();
    validate_tensor_nd_non_empty_complex(&tensor_view)?;

    let axis = Axis(tensor_view.ndim() - 1);
    let mut output_shape = tensor_view.shape().to_vec();
    let _ = output_shape.pop();
    if output.shape() != output_shape.as_slice() {
        return Err(TensorError::DimensionMismatch);
    }
    for (out_value, lane) in output.iter_mut().zip(tensor_view.lanes(axis)) {
        let sum_sq = lane.iter().map(Complex64::norm_sqr).sum::<f64>();
        *out_value = sum_sq.sqrt();
    }
    Ok(())
}

/// Normalize complex tensor values along the last axis.
///
/// # Errors
/// Returns an error if tensor is empty or has zero dimensions.
pub fn normalize_last_axis_complex(
    tensor: &ArrayD<Complex64>,
) -> Result<ArrayD<Complex64>, TensorError> {
    normalize_last_axis_complex_view(&tensor.view())
}

/// Normalize complex tensor values along the last axis from a tensor view.
///
/// # Errors
/// Returns an error if tensor is empty or has zero dimensions.
pub fn normalize_last_axis_complex_view(
    tensor: &ArrayViewD<'_, Complex64>,
) -> Result<ArrayD<Complex64>, TensorError> {
    let tensor_view = tensor.view();
    validate_tensor_nd_non_empty_complex(&tensor_view)?;

    let mut output = tensor_view.to_owned();
    normalize_last_axis_complex_view_into(tensor, &mut output)?;
    Ok(output)
}

/// Normalize complex tensor values along the last axis into `output`.
///
/// # Errors
/// Returns an error if tensor is empty, has zero dimensions, or output shape mismatches.
pub fn normalize_last_axis_complex_view_into<S>(
    tensor: &ArrayViewD<'_, Complex64>,
    output: &mut ArrayBase<S, IxDyn>,
) -> Result<(), TensorError>
where
    S: DataMut<Elem = Complex64>,
{
    let tensor_view = tensor.view();
    validate_tensor_nd_non_empty_complex(&tensor_view)?;
    if output.shape() != tensor_view.shape() {
        return Err(TensorError::DimensionMismatch);
    }

    output.assign(&tensor_view);
    let axis = Axis(tensor_view.ndim() - 1);
    for mut lane in output.lanes_mut(axis) {
        let norm = lane.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt();
        let denominator = norm.max(f64::EPSILON);
        for value in &mut lane {
            *value /= denominator;
        }
    }
    Ok(())
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
    batched_dot_last_axis_complex_view(&left.view(), &right.view())
}

/// Compute batched complex dot products along the last axis of two tensor views.
///
/// The input tensors must have identical shape and `ndim >= 1`.
/// Output shape is the input shape without the last axis.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn batched_dot_last_axis_complex_view(
    left: &ArrayViewD<'_, Complex64>,
    right: &ArrayViewD<'_, Complex64>,
) -> Result<ArrayD<Complex64>, TensorError> {
    let left_view = left.view();
    let right_view = right.view();
    validate_tensor_nd_non_empty_complex(&left_view)?;
    validate_tensor_nd_non_empty_complex(&right_view)?;
    if left_view.shape() != right_view.shape() {
        return Err(TensorError::DimensionMismatch);
    }

    let mut output_shape = left_view.shape().to_vec();
    let _ = output_shape.pop();
    let mut output = ArrayD::<Complex64>::zeros(IxDyn(&output_shape));
    batched_dot_last_axis_complex_view_into(left, right, &mut output)?;
    Ok(output)
}

/// Compute batched complex dot products along the last axis of two tensor views into `output`.
///
/// # Errors
/// Returns an error if inputs are empty, dimensions are incompatible, or output shape mismatches.
pub fn batched_dot_last_axis_complex_view_into<S>(
    left: &ArrayViewD<'_, Complex64>,
    right: &ArrayViewD<'_, Complex64>,
    output: &mut ArrayBase<S, IxDyn>,
) -> Result<(), TensorError>
where
    S: DataMut<Elem = Complex64>,
{
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
    if output.shape() != output_shape.as_slice() {
        return Err(TensorError::DimensionMismatch);
    }
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
    Ok(())
}

/// Permute complex tensor axes using an explicit axis ordering.
///
/// # Errors
/// Returns an error if the tensor is empty, has zero dimensions, or permutation is invalid.
pub fn permute_axes_complex(
    tensor: &ArrayD<Complex64>,
    permutation: &[usize],
) -> Result<ArrayD<Complex64>, TensorError> {
    permute_axes_complex_view(&tensor.view(), permutation)
}

/// Permute complex tensor axes from a tensor view using an explicit axis ordering.
///
/// # Errors
/// Returns an error if the tensor is empty, has zero dimensions, or permutation is invalid.
pub fn permute_axes_complex_view(
    tensor: &ArrayViewD<'_, Complex64>,
    permutation: &[usize],
) -> Result<ArrayD<Complex64>, TensorError> {
    let tensor_view = tensor.view();
    validate_tensor_nd_non_empty_complex(&tensor_view)?;
    if !validate_permutation(tensor_view.ndim(), permutation) {
        return Err(TensorError::DimensionMismatch);
    }
    let mut output = ArrayD::<Complex64>::zeros(IxDyn(
        &permutation.iter().map(|axis| tensor_view.shape()[*axis]).collect::<Vec<_>>(),
    ));
    permute_axes_complex_view_into(tensor, permutation, &mut output)?;
    Ok(output)
}

/// Permute complex tensor axes from a tensor view into `output`.
///
/// # Errors
/// Returns an error if the tensor is empty, has zero dimensions, permutation is invalid, or the
/// output shape mismatches.
pub fn permute_axes_complex_view_into<S>(
    tensor: &ArrayViewD<'_, Complex64>,
    permutation: &[usize],
    output: &mut ArrayBase<S, IxDyn>,
) -> Result<(), TensorError>
where
    S: DataMut<Elem = Complex64>,
{
    let tensor_view = tensor.view();
    validate_tensor_nd_non_empty_complex(&tensor_view)?;
    if !validate_permutation(tensor_view.ndim(), permutation) {
        return Err(TensorError::DimensionMismatch);
    }
    let permuted = tensor_view.permuted_axes(permutation.to_vec());
    if output.shape() != permuted.shape() {
        return Err(TensorError::DimensionMismatch);
    }
    output.assign(&permuted);
    Ok(())
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
        return tensor_contract_axes_complex_dispatch(
            &left_owned,
            &right_owned,
            left_axes[0],
            right_axes[0],
        );
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
pub fn contract_axes_complex_view_into<S>(
    left: &ArrayViewD<'_, Complex64>,
    right: &ArrayViewD<'_, Complex64>,
    left_axes: &[usize],
    right_axes: &[usize],
    output: &mut ArrayBase<S, IxDyn>,
) -> Result<(), TensorError>
where
    S: DataMut<Elem = Complex64>,
{
    let left_view = left.view();
    let right_view = right.view();
    validate_tensor_nd_non_empty_complex(&left_view)?;
    validate_tensor_nd_non_empty_complex(&right_view)?;
    let mut output_view = output.view_mut();
    contract_view_into_impl(&left_view, &right_view, left_axes, right_axes, &mut output_view)
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
    tensor_batched_matmul_last_two_complex_dispatch(&left_owned, &right_owned)
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
pub fn batched_matmul_last_two_complex_view_into<S>(
    left: &ArrayViewD<'_, Complex64>,
    right: &ArrayViewD<'_, Complex64>,
    output: &mut ArrayBase<S, IxDyn>,
) -> Result<(), TensorError>
where
    S: DataMut<Elem = Complex64>,
{
    let left_view = left.view();
    let right_view = right.view();
    validate_tensor_nd_non_empty_complex(&left_view)?;
    validate_tensor_nd_non_empty_complex(&right_view)?;
    let mut output_view = output.view_mut();
    batched_matmul_last_two_view_into_impl(&left_view, &right_view, &mut output_view)
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

fn einsum_binary_into_impl<T, S>(
    left: &ArrayViewD<'_, T>,
    right: &ArrayViewD<'_, T>,
    left_labels: &[char],
    right_labels: &[char],
    output_labels: &[char],
    output: &mut ArrayBase<S, IxDyn>,
) -> Result<(), TensorError>
where
    T: NabledReal,
    S: DataMut<Elem = T>,
{
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
    if output.shape() != output_shape.as_slice() {
        return Err(TensorError::DimensionMismatch);
    }

    let output_size = shape_product(&output_shape);
    let sum_size = shape_product(&sum_shape);
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

    Ok(())
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

fn einsum_binary_into_impl_complex<S>(
    left: &ArrayViewD<'_, Complex64>,
    right: &ArrayViewD<'_, Complex64>,
    left_labels: &[char],
    right_labels: &[char],
    output_labels: &[char],
    output: &mut ArrayBase<S, IxDyn>,
) -> Result<(), TensorError>
where
    S: DataMut<Elem = Complex64>,
{
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
    if output.shape() != output_shape.as_slice() {
        return Err(TensorError::DimensionMismatch);
    }

    let output_size = shape_product(&output_shape);
    let sum_size = shape_product(&sum_shape);
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

    Ok(())
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
    einsum_view(expression, &left.view(), &right.view())
}

/// Evaluate two-operand Einstein summation over real tensor views.
///
/// Expression format: `"labels_left,labels_right->labels_out"`, for example
/// `"bij,bjk->bik"` or `"ab,bc->ac"`.
///
/// # Errors
/// Returns an error if expression syntax is invalid or dimensions are incompatible.
pub fn einsum_view<T: NabledReal>(
    expression: &str,
    left: &ArrayViewD<'_, T>,
    right: &ArrayViewD<'_, T>,
) -> Result<ArrayD<T>, TensorError> {
    let left_view = left.view();
    let right_view = right.view();
    validate_tensor_nd_non_empty(&left_view)?;
    validate_tensor_nd_non_empty(&right_view)?;
    let (left_labels, right_labels, output_labels) = parse_einsum_two_operands(expression)?;
    einsum_binary_impl(&left_view, &right_view, &left_labels, &right_labels, &output_labels)
}

/// Evaluate two-operand Einstein summation over real tensors into `output`.
///
/// # Errors
/// Returns an error if expression syntax is invalid, dimensions are incompatible, or output shape
/// mismatches.
pub fn einsum_into<T: NabledReal, S: DataMut<Elem = T>>(
    expression: &str,
    left: &ArrayD<T>,
    right: &ArrayD<T>,
    output: &mut ArrayBase<S, IxDyn>,
) -> Result<(), TensorError> {
    einsum_view_into(expression, &left.view(), &right.view(), output)
}

/// Evaluate two-operand Einstein summation over real tensor views into `output`.
///
/// # Errors
/// Returns an error if expression syntax is invalid, dimensions are incompatible, or output shape
/// mismatches.
pub fn einsum_view_into<T: NabledReal, S: DataMut<Elem = T>>(
    expression: &str,
    left: &ArrayViewD<'_, T>,
    right: &ArrayViewD<'_, T>,
    output: &mut ArrayBase<S, IxDyn>,
) -> Result<(), TensorError> {
    let left_view = left.view();
    let right_view = right.view();
    validate_tensor_nd_non_empty(&left_view)?;
    validate_tensor_nd_non_empty(&right_view)?;
    let (left_labels, right_labels, output_labels) = parse_einsum_two_operands(expression)?;
    einsum_binary_into_impl(
        &left_view,
        &right_view,
        &left_labels,
        &right_labels,
        &output_labels,
        output,
    )
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
    einsum_complex_view(expression, &left.view(), &right.view())
}

/// Evaluate two-operand Einstein summation over complex tensor views.
///
/// Expression format: `"labels_left,labels_right->labels_out"`, for example
/// `"bij,bjk->bik"` or `"ab,bc->ac"`.
///
/// # Errors
/// Returns an error if expression syntax is invalid or dimensions are incompatible.
pub fn einsum_complex_view(
    expression: &str,
    left: &ArrayViewD<'_, Complex64>,
    right: &ArrayViewD<'_, Complex64>,
) -> Result<ArrayD<Complex64>, TensorError> {
    let left_view = left.view();
    let right_view = right.view();
    validate_tensor_nd_non_empty_complex(&left_view)?;
    validate_tensor_nd_non_empty_complex(&right_view)?;
    let (left_labels, right_labels, output_labels) = parse_einsum_two_operands(expression)?;
    einsum_binary_impl_complex(&left_view, &right_view, &left_labels, &right_labels, &output_labels)
}

/// Evaluate two-operand Einstein summation over complex tensors into `output`.
///
/// # Errors
/// Returns an error if expression syntax is invalid, dimensions are incompatible, or output shape
/// mismatches.
pub fn einsum_complex_into<S: DataMut<Elem = Complex64>>(
    expression: &str,
    left: &ArrayD<Complex64>,
    right: &ArrayD<Complex64>,
    output: &mut ArrayBase<S, IxDyn>,
) -> Result<(), TensorError> {
    einsum_complex_view_into(expression, &left.view(), &right.view(), output)
}

/// Evaluate two-operand Einstein summation over complex tensor views into `output`.
///
/// # Errors
/// Returns an error if expression syntax is invalid, dimensions are incompatible, or output shape
/// mismatches.
pub fn einsum_complex_view_into<S: DataMut<Elem = Complex64>>(
    expression: &str,
    left: &ArrayViewD<'_, Complex64>,
    right: &ArrayViewD<'_, Complex64>,
    output: &mut ArrayBase<S, IxDyn>,
) -> Result<(), TensorError> {
    let left_view = left.view();
    let right_view = right.view();
    validate_tensor_nd_non_empty_complex(&left_view)?;
    validate_tensor_nd_non_empty_complex(&right_view)?;
    let (left_labels, right_labels, output_labels) = parse_einsum_two_operands(expression)?;
    einsum_binary_into_impl_complex(
        &left_view,
        &right_view,
        &left_labels,
        &right_labels,
        &output_labels,
        output,
    )
}

fn mode0_product<T, S, M>(
    tensor: &ArrayBase<S, Ix3>,
    matrix: &ArrayBase<M, Ix2>,
) -> Result<Array3<T>, TensorError>
where
    T: NabledReal,
    S: Data<Elem = T>,
    M: Data<Elem = T>,
{
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

fn mode1_product<T, S, M>(
    tensor: &ArrayBase<S, Ix3>,
    matrix: &ArrayBase<M, Ix2>,
) -> Result<Array3<T>, TensorError>
where
    T: NabledReal,
    S: Data<Elem = T>,
    M: Data<Elem = T>,
{
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

fn mode2_product<T, S, M>(
    tensor: &ArrayBase<S, Ix3>,
    matrix: &ArrayBase<M, Ix2>,
) -> Result<Array3<T>, TensorError>
where
    T: NabledReal,
    S: Data<Elem = T>,
    M: Data<Elem = T>,
{
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

fn unfold_mode0<T, S>(tensor: &ArrayBase<S, Ix3>) -> Array2<T>
where
    T: NabledReal,
    S: Data<Elem = T>,
{
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

fn unfold_mode1<T, S>(tensor: &ArrayBase<S, Ix3>) -> Array2<T>
where
    T: NabledReal,
    S: Data<Elem = T>,
{
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

fn unfold_mode2<T, S>(tensor: &ArrayBase<S, Ix3>) -> Array2<T>
where
    T: NabledReal,
    S: Data<Elem = T>,
{
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

fn gram_matrix<T: NabledReal>(factor: &Array2<T>) -> Array2<T> {
    let (rows, rank) = factor.dim();
    let mut gram = Array2::<T>::zeros((rank, rank));
    for i in 0..rank {
        for j in 0..rank {
            let mut sum = T::zero();
            for row in 0..rows {
                sum += factor[[row, i]] * factor[[row, j]];
            }
            gram[[i, j]] = sum;
        }
    }
    gram
}

fn hadamard_product<T: NabledReal>(left: &Array2<T>, right: &Array2<T>) -> Array2<T> {
    let (rows, cols) = left.dim();
    let mut output = Array2::<T>::zeros((rows, cols));
    for i in 0..rows {
        for j in 0..cols {
            output[[i, j]] = left[[i, j]] * right[[i, j]];
        }
    }
    output
}

fn mttkrp_mode0<T: NabledReal>(
    tensor: &Array3<T>,
    factor_1: &Array2<T>,
    factor_2: &Array2<T>,
) -> Array2<T> {
    let (i0, i1, i2) = tensor.dim();
    let rank = factor_1.ncols();
    let mut output = Array2::<T>::zeros((i0, rank));
    for i in 0..i0 {
        for j in 0..i1 {
            for k in 0..i2 {
                let value = tensor[[i, j, k]];
                for r in 0..rank {
                    output[[i, r]] += value * factor_1[[j, r]] * factor_2[[k, r]];
                }
            }
        }
    }
    output
}

fn mttkrp_mode1<T: NabledReal>(
    tensor: &Array3<T>,
    factor_0: &Array2<T>,
    factor_2: &Array2<T>,
) -> Array2<T> {
    let (i0, i1, i2) = tensor.dim();
    let rank = factor_0.ncols();
    let mut output = Array2::<T>::zeros((i1, rank));
    for i in 0..i0 {
        for j in 0..i1 {
            for k in 0..i2 {
                let value = tensor[[i, j, k]];
                for r in 0..rank {
                    output[[j, r]] += value * factor_0[[i, r]] * factor_2[[k, r]];
                }
            }
        }
    }
    output
}

fn mttkrp_mode2<T: NabledReal>(
    tensor: &Array3<T>,
    factor_0: &Array2<T>,
    factor_1: &Array2<T>,
) -> Array2<T> {
    let (i0, i1, i2) = tensor.dim();
    let rank = factor_0.ncols();
    let mut output = Array2::<T>::zeros((i2, rank));
    for i in 0..i0 {
        for j in 0..i1 {
            for k in 0..i2 {
                let value = tensor[[i, j, k]];
                for r in 0..rank {
                    output[[k, r]] += value * factor_0[[i, r]] * factor_1[[j, r]];
                }
            }
        }
    }
    output
}

fn decode_row_major_index(mut index: usize, extents: &[usize], coordinates: &mut [usize]) {
    for axis in (0..extents.len()).rev() {
        let extent = extents[axis];
        coordinates[axis] = index % extent;
        index /= extent;
    }
}

fn validate_cp_nd_factors<T: NabledReal>(factors: &[Array2<T>]) -> Result<usize, TensorError> {
    if factors.is_empty() {
        return Err(TensorError::DimensionMismatch);
    }
    let rank = factors[0].ncols();
    if rank == 0 {
        return Err(TensorError::DimensionMismatch);
    }
    for factor in factors {
        if factor.nrows() == 0 || factor.ncols() != rank {
            return Err(TensorError::DimensionMismatch);
        }
    }
    Ok(rank)
}

fn khatri_rao_except_mode_nd<T: NabledReal>(
    factors: &[Array2<T>],
    mode: usize,
) -> Result<Array2<T>, TensorError> {
    let rank = validate_cp_nd_factors(factors)?;
    if mode >= factors.len() {
        return Err(TensorError::DimensionMismatch);
    }
    let order = mode_axes_order(factors.len(), mode)?;
    let other_axes = &order[1..];
    let other_extents = other_axes.iter().map(|axis| factors[*axis].nrows()).collect::<Vec<_>>();
    let rows = shape_product(&other_extents);
    let mut output = Array2::<T>::zeros((rows, rank));
    let mut coordinates = vec![0_usize; other_extents.len()];

    for row in 0..rows {
        decode_row_major_index(row, &other_extents, &mut coordinates);
        for component in 0..rank {
            let mut value = T::one();
            for (position, axis) in other_axes.iter().enumerate() {
                value *= factors[*axis][[coordinates[position], component]];
            }
            output[[row, component]] = value;
        }
    }
    Ok(output)
}

fn mttkrp_mode_nd<T: NabledReal>(
    tensor: &ArrayViewD<'_, T>,
    factors: &[Array2<T>],
    mode: usize,
) -> Result<Array2<T>, TensorError> {
    let unfolded = unfold_mode_nd(tensor, mode)?;
    let khatri_rao = khatri_rao_except_mode_nd(factors, mode)?;
    matrix_times_unfolded(&unfolded, &khatri_rao)
}

fn cp_rhs_gram_except_mode_nd<T: NabledReal>(
    factors: &[Array2<T>],
    mode: usize,
) -> Result<Array2<T>, TensorError> {
    let rank = validate_cp_nd_factors(factors)?;
    if mode >= factors.len() {
        return Err(TensorError::DimensionMismatch);
    }
    let mut gram = Array2::<T>::from_elem((rank, rank), T::one());
    for (axis, factor) in factors.iter().enumerate() {
        if axis == mode {
            continue;
        }
        let factor_gram = gram_matrix(factor);
        gram = hadamard_product(&gram, &factor_gram);
    }
    Ok(gram)
}

fn invert_small_matrix<T: NabledReal>(
    matrix: &Array2<T>,
    regularization: T,
) -> Result<Array2<T>, TensorError> {
    let (rows, cols) = matrix.dim();
    if rows == 0 || rows != cols {
        return Err(TensorError::DimensionMismatch);
    }

    let n = rows;
    let mut augmented = Array2::<T>::zeros((n, 2 * n));
    for row in 0..n {
        for col in 0..n {
            augmented[[row, col]] = matrix[[row, col]];
        }
        augmented[[row, row]] += regularization;
        augmented[[row, n + row]] = T::one();
    }

    for pivot in 0..n {
        let mut best_row = pivot;
        let mut best_value = augmented[[pivot, pivot]].abs();
        for row in (pivot + 1)..n {
            let candidate = augmented[[row, pivot]].abs();
            if candidate > best_value {
                best_row = row;
                best_value = candidate;
            }
        }
        if best_value <= regularization {
            return Err(TensorError::DimensionMismatch);
        }

        if best_row != pivot {
            for col in 0..(2 * n) {
                let tmp = augmented[[pivot, col]];
                augmented[[pivot, col]] = augmented[[best_row, col]];
                augmented[[best_row, col]] = tmp;
            }
        }

        let pivot_value = augmented[[pivot, pivot]];
        for col in 0..(2 * n) {
            augmented[[pivot, col]] /= pivot_value;
        }

        for row in 0..n {
            if row == pivot {
                continue;
            }
            let factor = augmented[[row, pivot]];
            if factor.abs() <= regularization {
                augmented[[row, pivot]] = T::zero();
                continue;
            }
            for col in 0..(2 * n) {
                let pivot_entry = augmented[[pivot, col]];
                augmented[[row, col]] -= factor * pivot_entry;
            }
            augmented[[row, pivot]] = T::zero();
        }
    }

    let mut inverse = Array2::<T>::zeros((n, n));
    for row in 0..n {
        for col in 0..n {
            inverse[[row, col]] = augmented[[row, n + col]];
        }
    }
    Ok(inverse)
}

fn solve_right_with_gram<T: NabledReal>(
    right_hand: &Array2<T>,
    gram: &Array2<T>,
    regularization: T,
) -> Result<Array2<T>, TensorError> {
    let inverse = invert_small_matrix(gram, regularization)?;
    let (rows, inner) = right_hand.dim();
    if inverse.nrows() != inner || inverse.ncols() != inner {
        return Err(TensorError::DimensionMismatch);
    }

    let mut output = Array2::<T>::zeros((rows, inner));
    for row in 0..rows {
        for col in 0..inner {
            let mut sum = T::zero();
            for k in 0..inner {
                sum += right_hand[[row, k]] * inverse[[k, col]];
            }
            output[[row, col]] = sum;
        }
    }
    Ok(output)
}

fn normalize_factor_column<T: NabledReal>(factor: &mut Array2<T>, column: usize) -> T {
    let mut norm_sq = T::zero();
    for row in 0..factor.nrows() {
        let value = factor[[row, column]];
        norm_sq += value * value;
    }
    let norm = norm_sq.sqrt();
    if norm > T::zero() {
        for row in 0..factor.nrows() {
            factor[[row, column]] /= norm;
        }
    }
    norm
}

fn normalize_cp_columns<T: NabledReal>(
    factor_0: &mut Array2<T>,
    factor_1: &mut Array2<T>,
    factor_2: &mut Array2<T>,
    weights: &mut Array1<T>,
    tolerance: T,
) {
    for component in 0..weights.len() {
        let norm_0 = normalize_factor_column(factor_0, component);
        let norm_1 = normalize_factor_column(factor_1, component);
        let norm_2 = normalize_factor_column(factor_2, component);
        let mut weight = norm_0 * norm_1 * norm_2;
        if weight.abs() <= tolerance {
            weight = T::one();
        }
        weights[component] = weight;
    }
}

fn normalize_cp_nd_columns<T: NabledReal>(
    factors: &mut [Array2<T>],
    weights: &mut Array1<T>,
    tolerance: T,
) -> Result<(), TensorError> {
    let rank = validate_cp_nd_factors(factors)?;
    if rank != weights.len() {
        return Err(TensorError::DimensionMismatch);
    }
    for component in 0..weights.len() {
        let mut weight = T::one();
        for factor in factors.iter_mut() {
            let norm = normalize_factor_column(factor, component);
            weight *= norm;
        }
        if weight.abs() <= tolerance {
            weight = T::one();
        }
        weights[component] = weight;
    }
    Ok(())
}

fn factor_relative_change<T: NabledReal>(
    current: &Array2<T>,
    previous: &Array2<T>,
    tolerance: T,
) -> T {
    let mut delta_sq = T::zero();
    let mut baseline_sq = T::zero();
    for (current_value, previous_value) in current.iter().zip(previous.iter()) {
        let delta = *current_value - *previous_value;
        delta_sq += delta * delta;
        baseline_sq += *previous_value * *previous_value;
    }
    let delta = delta_sq.sqrt();
    let baseline = baseline_sq.sqrt();
    if baseline <= tolerance { delta } else { delta / baseline }
}

fn cp_nd_relative_change<T: NabledReal>(
    current: &[Array2<T>],
    previous: &[Array2<T>],
    tolerance: T,
) -> Result<T, TensorError> {
    if current.len() != previous.len() || current.is_empty() {
        return Err(TensorError::DimensionMismatch);
    }
    let mut max_change = T::zero();
    for (current_factor, previous_factor) in current.iter().zip(previous.iter()) {
        if current_factor.dim() != previous_factor.dim() {
            return Err(TensorError::DimensionMismatch);
        }
        let change = factor_relative_change(current_factor, previous_factor, tolerance);
        if change > max_change {
            max_change = change;
        }
    }
    Ok(max_change)
}

fn cp_error_metrics_from_squares<T: NabledReal>(signal_sq: T, residual_sq: T) -> CpErrorMetrics<T> {
    let signal_norm = signal_sq.sqrt();
    let residual_norm = residual_sq.sqrt();
    let relative_error = if signal_norm <= T::epsilon() {
        if residual_norm <= T::epsilon() { T::zero() } else { T::one() }
    } else {
        residual_norm / signal_norm
    };

    let mut fit = T::one() - relative_error;
    if fit < T::zero() {
        fit = T::zero();
    }
    if fit > T::one() {
        fit = T::one();
    }

    CpErrorMetrics { signal_norm, residual_norm, relative_error, fit }
}

fn cp_als3_impl_with_convergence<T: CpAlsScalar>(
    cube: &ArrayView3<'_, T>,
    rank: usize,
    config: &CpAlsConfig<T>,
) -> Result<(CpAls3Result<T>, CpConvergenceReport<T>), TensorError> {
    validate_cube_non_empty(cube)?;
    let (i0, i1, i2) = cube.dim();
    if rank == 0 || config.max_iterations == 0 {
        return Err(TensorError::DimensionMismatch);
    }
    let max_rank = i0.min(i1).min(i2);
    if rank > max_rank {
        return Err(TensorError::DimensionMismatch);
    }

    let regularization = config.tolerance.max(T::epsilon());
    let cube_owned = cube.to_owned();

    let factor_0_full =
        svd::decompose(&unfold_mode0(&cube_owned)).map_err(|_| TensorError::DimensionMismatch)?.u;
    let factor_1_full =
        svd::decompose(&unfold_mode1(&cube_owned)).map_err(|_| TensorError::DimensionMismatch)?.u;
    let factor_2_full =
        svd::decompose(&unfold_mode2(&cube_owned)).map_err(|_| TensorError::DimensionMismatch)?.u;

    if rank > factor_0_full.ncols() || rank > factor_1_full.ncols() || rank > factor_2_full.ncols()
    {
        return Err(TensorError::DimensionMismatch);
    }

    let mut factor_0 = factor_0_full.slice(s![.., 0..rank]).to_owned();
    let mut factor_1 = factor_1_full.slice(s![.., 0..rank]).to_owned();
    let mut factor_2 = factor_2_full.slice(s![.., 0..rank]).to_owned();
    let mut weights = Array1::<T>::from_elem(rank, T::one());

    normalize_cp_columns(&mut factor_0, &mut factor_1, &mut factor_2, &mut weights, regularization);

    let mut iterations_run = 0_usize;
    let mut converged = false;
    let mut final_max_change = T::zero();
    for _ in 0..config.max_iterations {
        iterations_run += 1;
        let previous_0 = factor_0.clone();
        let previous_1 = factor_1.clone();
        let previous_2 = factor_2.clone();

        let factor_1_gram = gram_matrix(&factor_1);
        let factor_2_gram = gram_matrix(&factor_2);
        let mode0_rhs_gram = hadamard_product(&factor_1_gram, &factor_2_gram);
        let mttkrp_0 = mttkrp_mode0(&cube_owned, &factor_1, &factor_2);
        factor_0 = solve_right_with_gram(&mttkrp_0, &mode0_rhs_gram, regularization)?;

        let factor_0_gram = gram_matrix(&factor_0);
        let mode1_rhs_gram = hadamard_product(&factor_0_gram, &factor_2_gram);
        let mttkrp_1 = mttkrp_mode1(&cube_owned, &factor_0, &factor_2);
        factor_1 = solve_right_with_gram(&mttkrp_1, &mode1_rhs_gram, regularization)?;

        let mode2_rhs_gram = hadamard_product(&factor_0_gram, &factor_1_gram);
        let mttkrp_2 = mttkrp_mode2(&cube_owned, &factor_0, &factor_1);
        factor_2 = solve_right_with_gram(&mttkrp_2, &mode2_rhs_gram, regularization)?;

        normalize_cp_columns(
            &mut factor_0,
            &mut factor_1,
            &mut factor_2,
            &mut weights,
            regularization,
        );

        let mut max_change = factor_relative_change(&factor_0, &previous_0, regularization);
        let change_1 = factor_relative_change(&factor_1, &previous_1, regularization);
        if change_1 > max_change {
            max_change = change_1;
        }
        let change_2 = factor_relative_change(&factor_2, &previous_2, regularization);
        if change_2 > max_change {
            max_change = change_2;
        }
        final_max_change = max_change;

        if max_change <= config.tolerance {
            converged = true;
            break;
        }
    }

    let decomposition = CpAls3Result { weights, factor_0, factor_1, factor_2 };
    let convergence = CpConvergenceReport {
        iterations_run,
        converged,
        final_max_factor_change: final_max_change,
    };
    Ok((decomposition, convergence))
}

fn cp_als3_impl<T: CpAlsScalar>(
    cube: &ArrayView3<'_, T>,
    rank: usize,
    config: &CpAlsConfig<T>,
) -> Result<CpAls3Result<T>, TensorError> {
    cp_als3_impl_with_convergence(cube, rank, config).map(|(decomposition, _)| decomposition)
}

/// Compute rank-`R` CP decomposition for a rank-3 real tensor using ALS.
///
/// # Errors
/// Returns an error if input is empty, rank/config are invalid, or the linearized ALS update
/// systems are singular.
pub fn cp_als3<T: CpAlsScalar>(
    cube: &Array3<T>,
    rank: usize,
    config: &CpAlsConfig<T>,
) -> Result<CpAls3Result<T>, TensorError> {
    cp_als3_impl(&cube.view(), rank, config)
}

/// Compute rank-`R` CP decomposition for a rank-3 real tensor from a view.
///
/// # Errors
/// Returns an error if input is empty, rank/config are invalid, or the linearized ALS update
/// systems are singular.
pub fn cp_als3_view<T: CpAlsScalar>(
    cube: &ArrayView3<'_, T>,
    rank: usize,
    config: &CpAlsConfig<T>,
) -> Result<CpAls3Result<T>, TensorError> {
    cp_als3_impl(cube, rank, config)
}

/// Compute rank-`R` CP decomposition for a rank-3 real tensor using ALS and return diagnostics.
///
/// # Errors
/// Returns an error if input is empty, rank/config are invalid, or ALS update systems are
/// singular.
pub fn cp_als3_with_report<T: CpAlsScalar>(
    cube: &Array3<T>,
    rank: usize,
    config: &CpAlsConfig<T>,
) -> Result<(CpAls3Result<T>, CpAlsReport<T>), TensorError> {
    cp_als3_view_with_report(&cube.view(), rank, config)
}

/// Compute rank-`R` CP decomposition for a rank-3 real tensor view using ALS and return
/// diagnostics.
///
/// # Errors
/// Returns an error if input is empty, rank/config are invalid, or ALS update systems are
/// singular.
pub fn cp_als3_view_with_report<T: CpAlsScalar>(
    cube: &ArrayView3<'_, T>,
    rank: usize,
    config: &CpAlsConfig<T>,
) -> Result<(CpAls3Result<T>, CpAlsReport<T>), TensorError> {
    let (decomposition, convergence) = cp_als3_impl_with_convergence(cube, rank, config)?;
    let metrics = cp_als3_diagnostics_view(cube, &decomposition)?;
    Ok((decomposition, CpAlsReport { convergence, metrics }))
}

/// Compute reconstruction diagnostics for a rank-3 CP decomposition against a reference tensor.
///
/// # Errors
/// Returns an error if dimensions are incompatible.
pub fn cp_als3_diagnostics<T: NabledReal>(
    cube: &Array3<T>,
    result: &CpAls3Result<T>,
) -> Result<CpErrorMetrics<T>, TensorError> {
    cp_als3_diagnostics_view(&cube.view(), result)
}

/// Compute reconstruction diagnostics for a rank-3 CP decomposition against a reference tensor
/// view.
///
/// # Errors
/// Returns an error if dimensions are incompatible.
pub fn cp_als3_diagnostics_from_factors_view<T: NabledReal>(
    cube: &ArrayView3<'_, T>,
    weights: &ArrayView1<'_, T>,
    factor_0: &ArrayView2<'_, T>,
    factor_1: &ArrayView2<'_, T>,
    factor_2: &ArrayView2<'_, T>,
) -> Result<CpErrorMetrics<T>, TensorError> {
    validate_cube_non_empty(cube)?;
    let rank = weights.len();
    if rank == 0
        || factor_0.ncols() != rank
        || factor_1.ncols() != rank
        || factor_2.ncols() != rank
        || cube.dim() != (factor_0.nrows(), factor_1.nrows(), factor_2.nrows())
    {
        return Err(TensorError::DimensionMismatch);
    }

    let mut signal_sq = T::zero();
    let mut residual_sq = T::zero();
    for i in 0..cube.dim().0 {
        for j in 0..cube.dim().1 {
            for k in 0..cube.dim().2 {
                let mut candidate = T::zero();
                for component in 0..rank {
                    candidate += weights[component]
                        * factor_0[[i, component]]
                        * factor_1[[j, component]]
                        * factor_2[[k, component]];
                }
                let reference_value = cube[[i, j, k]];
                signal_sq += reference_value * reference_value;
                let delta = reference_value - candidate;
                residual_sq += delta * delta;
            }
        }
    }
    Ok(cp_error_metrics_from_squares(signal_sq, residual_sq))
}

/// Compute reconstruction diagnostics for a rank-3 CP decomposition against a reference tensor
/// view.
///
/// # Errors
/// Returns an error if dimensions are incompatible.
pub fn cp_als3_diagnostics_view<T: NabledReal>(
    cube: &ArrayView3<'_, T>,
    result: &CpAls3Result<T>,
) -> Result<CpErrorMetrics<T>, TensorError> {
    cp_als3_diagnostics_from_factors_view(
        cube,
        &result.weights.view(),
        &result.factor_0.view(),
        &result.factor_1.view(),
        &result.factor_2.view(),
    )
}

/// Reconstruct a rank-3 tensor from CP factors.
///
/// # Errors
/// Returns an error if factor dimensions are incompatible.
pub fn cp_als3_reconstruct_from_factors_view<T: NabledReal>(
    weights: &ArrayView1<'_, T>,
    factor_0: &ArrayView2<'_, T>,
    factor_1: &ArrayView2<'_, T>,
    factor_2: &ArrayView2<'_, T>,
) -> Result<Array3<T>, TensorError> {
    let mut output = Array3::<T>::zeros((factor_0.nrows(), factor_1.nrows(), factor_2.nrows()));
    cp_als3_reconstruct_from_factors_view_into(weights, factor_0, factor_1, factor_2, &mut output)?;
    Ok(output)
}

/// Reconstruct a rank-3 tensor from CP factors into `output`.
///
/// # Errors
/// Returns an error if factor dimensions are incompatible.
pub fn cp_als3_reconstruct_from_factors_view_into<T: NabledReal, S: DataMut<Elem = T>>(
    weights: &ArrayView1<'_, T>,
    factor_0: &ArrayView2<'_, T>,
    factor_1: &ArrayView2<'_, T>,
    factor_2: &ArrayView2<'_, T>,
    output: &mut ArrayBase<S, Ix3>,
) -> Result<(), TensorError> {
    let rank = weights.len();
    if rank == 0 || factor_0.ncols() != rank || factor_1.ncols() != rank || factor_2.ncols() != rank
    {
        return Err(TensorError::DimensionMismatch);
    }
    if output.dim() != (factor_0.nrows(), factor_1.nrows(), factor_2.nrows()) {
        return Err(TensorError::DimensionMismatch);
    }

    output.fill(T::zero());
    for i in 0..factor_0.nrows() {
        for j in 0..factor_1.nrows() {
            for k in 0..factor_2.nrows() {
                let mut value = T::zero();
                for component in 0..rank {
                    value += weights[component]
                        * factor_0[[i, component]]
                        * factor_1[[j, component]]
                        * factor_2[[k, component]];
                }
                output[[i, j, k]] = value;
            }
        }
    }

    Ok(())
}

/// Reconstruct a rank-3 tensor from CP factors.
///
/// # Errors
/// Returns an error if factor dimensions are incompatible.
pub fn cp_als3_reconstruct<T: NabledReal>(
    result: &CpAls3Result<T>,
) -> Result<Array3<T>, TensorError> {
    cp_als3_reconstruct_from_factors_view(
        &result.weights.view(),
        &result.factor_0.view(),
        &result.factor_1.view(),
        &result.factor_2.view(),
    )
}

/// Reconstruct a rank-3 tensor from CP factors into `output`.
///
/// # Errors
/// Returns an error if factor dimensions are incompatible.
pub fn cp_als3_reconstruct_into<T: NabledReal, S: DataMut<Elem = T>>(
    result: &CpAls3Result<T>,
    output: &mut ArrayBase<S, Ix3>,
) -> Result<(), TensorError> {
    cp_als3_reconstruct_from_factors_view_into(
        &result.weights.view(),
        &result.factor_0.view(),
        &result.factor_1.view(),
        &result.factor_2.view(),
        output,
    )
}

fn validate_cp_als_nd_input<T: NabledReal>(
    tensor: &ArrayViewD<'_, T>,
    rank: usize,
    config: &CpAlsConfig<T>,
) -> Result<(), TensorError> {
    validate_tensor_nd_non_empty(tensor)?;
    if tensor.ndim() < 2 || rank == 0 || config.max_iterations == 0 || !config.tolerance.is_finite()
    {
        return Err(TensorError::DimensionMismatch);
    }
    let max_rank = tensor.shape().iter().copied().min().unwrap_or(0);
    if rank > max_rank {
        return Err(TensorError::DimensionMismatch);
    }
    Ok(())
}

fn cp_als_nd_impl_with_convergence<T: CpAlsScalar>(
    tensor: &ArrayViewD<'_, T>,
    rank: usize,
    config: &CpAlsConfig<T>,
) -> Result<(CpAlsNdResult<T>, CpConvergenceReport<T>), TensorError> {
    validate_cp_als_nd_input(tensor, rank, config)?;
    let regularization = config.tolerance.max(T::epsilon());
    let shape = tensor.shape().to_vec();
    let mut factors = Vec::<Array2<T>>::with_capacity(tensor.ndim());

    for mode in 0..tensor.ndim() {
        let unfolded = unfold_mode_nd(tensor, mode)?;
        let u_full = svd::decompose(&unfolded).map_err(|_| TensorError::DimensionMismatch)?.u;
        if rank > u_full.ncols() {
            return Err(TensorError::DimensionMismatch);
        }
        factors.push(u_full.slice(s![.., 0..rank]).to_owned());
    }

    let mut weights = Array1::<T>::from_elem(rank, T::one());
    normalize_cp_nd_columns(&mut factors, &mut weights, regularization)?;

    let mut iterations_run = 0_usize;
    let mut converged = false;
    let mut final_max_change = T::zero();
    for _ in 0..config.max_iterations {
        iterations_run += 1;
        let previous = factors.clone();
        for mode in 0..tensor.ndim() {
            let rhs_gram = cp_rhs_gram_except_mode_nd(&factors, mode)?;
            let mttkrp = mttkrp_mode_nd(tensor, &factors, mode)?;
            factors[mode] = solve_right_with_gram(&mttkrp, &rhs_gram, regularization)?;
        }

        normalize_cp_nd_columns(&mut factors, &mut weights, regularization)?;
        let max_change = cp_nd_relative_change(&factors, &previous, regularization)?;
        final_max_change = max_change;
        if max_change <= config.tolerance {
            converged = true;
            break;
        }
    }

    let decomposition = CpAlsNdResult { weights, factors, shape };
    let convergence = CpConvergenceReport {
        iterations_run,
        converged,
        final_max_factor_change: final_max_change,
    };
    Ok((decomposition, convergence))
}

fn cp_als_nd_impl<T: CpAlsScalar>(
    tensor: &ArrayViewD<'_, T>,
    rank: usize,
    config: &CpAlsConfig<T>,
) -> Result<CpAlsNdResult<T>, TensorError> {
    cp_als_nd_impl_with_convergence(tensor, rank, config).map(|(decomposition, _)| decomposition)
}

fn validate_cp_als_nd_result<T: NabledReal>(result: &CpAlsNdResult<T>) -> Result<(), TensorError> {
    if result.factors.is_empty() || result.shape.len() != result.factors.len() {
        return Err(TensorError::DimensionMismatch);
    }
    let rank = result.weights.len();
    if rank == 0 {
        return Err(TensorError::DimensionMismatch);
    }
    for (axis, factor) in result.factors.iter().enumerate() {
        if factor.nrows() != result.shape[axis] || factor.ncols() != rank || factor.nrows() == 0 {
            return Err(TensorError::DimensionMismatch);
        }
    }
    Ok(())
}

fn validate_cp_als_nd_factors<T, S>(
    weights: &ArrayView1<'_, T>,
    factors: &[ArrayBase<S, Ix2>],
) -> Result<Vec<usize>, TensorError>
where
    T: NabledReal,
    S: Data<Elem = T>,
{
    if factors.is_empty() {
        return Err(TensorError::DimensionMismatch);
    }
    let rank = weights.len();
    if rank == 0 {
        return Err(TensorError::DimensionMismatch);
    }

    let mut shape = Vec::with_capacity(factors.len());
    for factor in factors {
        if factor.nrows() == 0 || factor.ncols() != rank {
            return Err(TensorError::DimensionMismatch);
        }
        shape.push(factor.nrows());
    }
    Ok(shape)
}

/// Compute rank-`R` CP decomposition for an `N`-D real tensor using ALS.
///
/// # Errors
/// Returns an error if input is empty, rank/config are invalid, or ALS update systems are
/// singular.
pub fn cp_als_nd<T: CpAlsScalar>(
    tensor: &ArrayD<T>,
    rank: usize,
    config: &CpAlsConfig<T>,
) -> Result<CpAlsNdResult<T>, TensorError> {
    cp_als_nd_impl(&tensor.view(), rank, config)
}

/// Compute rank-`R` CP decomposition for an `N`-D real tensor from a view.
///
/// # Errors
/// Returns an error if input is empty, rank/config are invalid, or ALS update systems are
/// singular.
pub fn cp_als_nd_view<T: CpAlsScalar>(
    tensor: &ArrayViewD<'_, T>,
    rank: usize,
    config: &CpAlsConfig<T>,
) -> Result<CpAlsNdResult<T>, TensorError> {
    cp_als_nd_impl(tensor, rank, config)
}

/// Compute rank-`R` CP decomposition for an `N`-D real tensor using ALS and return diagnostics.
///
/// # Errors
/// Returns an error if input is empty, rank/config are invalid, or ALS update systems are
/// singular.
pub fn cp_als_nd_with_report<T: CpAlsScalar>(
    tensor: &ArrayD<T>,
    rank: usize,
    config: &CpAlsConfig<T>,
) -> Result<(CpAlsNdResult<T>, CpAlsReport<T>), TensorError> {
    cp_als_nd_view_with_report(&tensor.view(), rank, config)
}

/// Compute rank-`R` CP decomposition for an `N`-D real tensor view using ALS and return
/// diagnostics.
///
/// # Errors
/// Returns an error if input is empty, rank/config are invalid, or ALS update systems are
/// singular.
pub fn cp_als_nd_view_with_report<T: CpAlsScalar>(
    tensor: &ArrayViewD<'_, T>,
    rank: usize,
    config: &CpAlsConfig<T>,
) -> Result<(CpAlsNdResult<T>, CpAlsReport<T>), TensorError> {
    let (decomposition, convergence) = cp_als_nd_impl_with_convergence(tensor, rank, config)?;
    let metrics = cp_als_nd_diagnostics_view(tensor, &decomposition)?;
    Ok((decomposition, CpAlsReport { convergence, metrics }))
}

/// Compute reconstruction diagnostics for an `N`-D CP decomposition against a reference tensor.
///
/// # Errors
/// Returns an error if dimensions are incompatible.
pub fn cp_als_nd_diagnostics<T: NabledReal>(
    tensor: &ArrayD<T>,
    result: &CpAlsNdResult<T>,
) -> Result<CpErrorMetrics<T>, TensorError> {
    cp_als_nd_diagnostics_view(&tensor.view(), result)
}

/// Compute reconstruction diagnostics for an `N`-D CP decomposition against a reference tensor
/// view.
///
/// # Errors
/// Returns an error if dimensions are incompatible.
pub fn cp_als_nd_diagnostics_view<T: NabledReal>(
    tensor: &ArrayViewD<'_, T>,
    result: &CpAlsNdResult<T>,
) -> Result<CpErrorMetrics<T>, TensorError> {
    validate_cp_als_nd_result(result)?;
    cp_als_nd_diagnostics_from_factors_view(
        tensor,
        &result.weights.view(),
        &result.factors.iter().map(|factor| factor.view()).collect::<Vec<_>>(),
    )
}

/// Compute reconstruction diagnostics for an `N`-D CP decomposition against a reference tensor
/// view using borrowed factor views.
///
/// # Errors
/// Returns an error if dimensions are incompatible.
pub fn cp_als_nd_diagnostics_from_factors_view<T: NabledReal>(
    tensor: &ArrayViewD<'_, T>,
    weights: &ArrayView1<'_, T>,
    factors: &[ArrayView2<'_, T>],
) -> Result<CpErrorMetrics<T>, TensorError> {
    validate_tensor_nd_non_empty(tensor)?;
    let shape = validate_cp_als_nd_factors(weights, factors)?;
    if tensor.shape() != shape.as_slice() {
        return Err(TensorError::DimensionMismatch);
    }

    let total = shape_product(&shape);
    let rank = weights.len();
    let mut coordinates = vec![0_usize; shape.len()];
    let mut signal_sq = T::zero();
    let mut residual_sq = T::zero();

    for linear in 0..total {
        decode_row_major_index(linear, &shape, &mut coordinates);
        let mut candidate = T::zero();
        for component in 0..rank {
            let mut term = weights[component];
            for (axis, coordinate) in coordinates.iter().enumerate() {
                term *= factors[axis][[*coordinate, component]];
            }
            candidate += term;
        }

        let reference_value = tensor[IxDyn(&coordinates)];
        signal_sq += reference_value * reference_value;
        let delta = reference_value - candidate;
        residual_sq += delta * delta;
    }

    Ok(cp_error_metrics_from_squares(signal_sq, residual_sq))
}

/// Reconstruct an `N`-D tensor from CP factors.
///
/// # Errors
/// Returns an error if factor dimensions are incompatible.
pub fn cp_als_nd_reconstruct<T: NabledReal>(
    result: &CpAlsNdResult<T>,
) -> Result<ArrayD<T>, TensorError> {
    validate_cp_als_nd_result(result)?;
    cp_als_nd_reconstruct_from_factors_view(
        &result.weights.view(),
        &result.factors.iter().map(|factor| factor.view()).collect::<Vec<_>>(),
    )
}

/// Reconstruct an `N`-D tensor from CP factors into `output`.
///
/// # Errors
/// Returns an error if factor dimensions are incompatible.
pub fn cp_als_nd_reconstruct_into<T: NabledReal, S: DataMut<Elem = T>>(
    result: &CpAlsNdResult<T>,
    output: &mut ArrayBase<S, IxDyn>,
) -> Result<(), TensorError> {
    validate_cp_als_nd_result(result)?;
    cp_als_nd_reconstruct_from_factors_view_into(
        &result.weights.view(),
        &result.factors.iter().map(|factor| factor.view()).collect::<Vec<_>>(),
        output,
    )
}

/// Reconstruct an `N`-D tensor from borrowed CP factor views.
///
/// # Errors
/// Returns an error if factor dimensions are incompatible.
pub fn cp_als_nd_reconstruct_from_factors_view<T: NabledReal>(
    weights: &ArrayView1<'_, T>,
    factors: &[ArrayView2<'_, T>],
) -> Result<ArrayD<T>, TensorError> {
    let shape = validate_cp_als_nd_factors(weights, factors)?;
    let mut output = ArrayD::<T>::zeros(IxDyn(&shape));
    cp_als_nd_reconstruct_from_factors_view_into(weights, factors, &mut output)?;
    Ok(output)
}

/// Reconstruct an `N`-D tensor from borrowed CP factor views into `output`.
///
/// # Errors
/// Returns an error if factor dimensions are incompatible.
pub fn cp_als_nd_reconstruct_from_factors_view_into<T: NabledReal, S: DataMut<Elem = T>>(
    weights: &ArrayView1<'_, T>,
    factors: &[ArrayView2<'_, T>],
    output: &mut ArrayBase<S, IxDyn>,
) -> Result<(), TensorError> {
    let shape = validate_cp_als_nd_factors(weights, factors)?;
    if output.shape() != shape.as_slice() {
        return Err(TensorError::DimensionMismatch);
    }

    let rank = weights.len();
    let total = shape_product(&shape);
    let mut coordinates = vec![0_usize; shape.len()];
    output.fill(T::zero());

    for linear in 0..total {
        decode_row_major_index(linear, &shape, &mut coordinates);
        let mut value = T::zero();
        for component in 0..rank {
            let mut term = weights[component];
            for (axis, coordinate) in coordinates.iter().enumerate() {
                term *= factors[axis][[*coordinate, component]];
            }
            value += term;
        }
        output[IxDyn(&coordinates)] = value;
    }

    Ok(())
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
    hosvd3_view(&cube.view(), ranks)
}

/// Compute rank-truncated HOSVD for a rank-3 real tensor view.
///
/// # Errors
/// Returns an error if input is empty, ranks are invalid, or factorization fails.
#[cfg(feature = "lapack-provider")]
pub fn hosvd3_view<T>(
    cube: &ArrayView3<'_, T>,
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
pub fn hosvd3<T: svd::SvdInternalScalar>(
    cube: &Array3<T>,
    ranks: (usize, usize, usize),
) -> Result<Hosvd3Result<T>, TensorError> {
    hosvd3_view(&cube.view(), ranks)
}

/// Compute rank-truncated HOSVD for a rank-3 real tensor view.
///
/// # Errors
/// Returns an error if input is empty, ranks are invalid, or factorization fails.
#[cfg(not(feature = "lapack-provider"))]
pub fn hosvd3_view<T: svd::SvdInternalScalar>(
    cube: &ArrayView3<'_, T>,
    ranks: (usize, usize, usize),
) -> Result<Hosvd3Result<T>, TensorError> {
    hosvd3_impl(cube, ranks)
}

#[cfg(not(feature = "lapack-provider"))]
fn hosvd3_impl<T: svd::SvdInternalScalar>(
    cube: &ArrayView3<'_, T>,
    ranks: (usize, usize, usize),
) -> Result<Hosvd3Result<T>, TensorError> {
    validate_cube_non_empty(cube)?;
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

    let core_mode0 = mode0_product(cube, &u0.t())?;
    let core_mode1 = mode1_product(&core_mode0, &u1.t())?;
    let core = mode2_product(&core_mode1, &u2.t())?;
    Ok(Hosvd3Result { core, u0, u1, u2 })
}

#[cfg(feature = "lapack-provider")]
fn hosvd3_impl<T>(
    cube: &ArrayView3<'_, T>,
    ranks: (usize, usize, usize),
) -> Result<Hosvd3Result<T>, TensorError>
where
    T: NabledReal + ndarray_linalg::Lapack<Real = T> + AddAssign,
{
    validate_cube_non_empty(cube)?;
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

    let core_mode0 = mode0_product(cube, &u0.t())?;
    let core_mode1 = mode1_product(&core_mode0, &u1.t())?;
    let core = mode2_product(&core_mode1, &u2.t())?;
    Ok(Hosvd3Result { core, u0, u1, u2 })
}

/// Reconstruct a rank-3 tensor from HOSVD factors.
///
/// # Errors
/// Returns an error if factor dimensions are incompatible.
pub fn hosvd3_reconstruct_from_factors_view<T: NabledReal>(
    core: &ArrayView3<'_, T>,
    u0: &ArrayView2<'_, T>,
    u1: &ArrayView2<'_, T>,
    u2: &ArrayView2<'_, T>,
) -> Result<Array3<T>, TensorError> {
    let mut output = Array3::<T>::zeros((u0.nrows(), u1.nrows(), u2.nrows()));
    hosvd3_reconstruct_from_factors_view_into(core, u0, u1, u2, &mut output)?;
    Ok(output)
}

/// Reconstruct a rank-3 tensor from HOSVD factors into `output`.
///
/// # Errors
/// Returns an error if factor dimensions are incompatible or output shape mismatches.
pub fn hosvd3_reconstruct_from_factors_view_into<T: NabledReal, S: DataMut<Elem = T>>(
    core: &ArrayView3<'_, T>,
    u0: &ArrayView2<'_, T>,
    u1: &ArrayView2<'_, T>,
    u2: &ArrayView2<'_, T>,
    output: &mut ArrayBase<S, Ix3>,
) -> Result<(), TensorError> {
    let (r0, r1, r2) = core.dim();
    if u0.ncols() != r0
        || u1.ncols() != r1
        || u2.ncols() != r2
        || output.dim() != (u0.nrows(), u1.nrows(), u2.nrows())
    {
        return Err(TensorError::DimensionMismatch);
    }

    for i in 0..output.dim().0 {
        for j in 0..output.dim().1 {
            for k in 0..output.dim().2 {
                let mut value = T::zero();
                for a in 0..r0 {
                    let u0_value = u0[[i, a]];
                    for b in 0..r1 {
                        let u1_value = u1[[j, b]];
                        for c in 0..r2 {
                            value += core[[a, b, c]] * u0_value * u1_value * u2[[k, c]];
                        }
                    }
                }
                output[[i, j, k]] = value;
            }
        }
    }
    Ok(())
}

/// Reconstruct a rank-3 tensor from HOSVD factors.
///
/// # Errors
/// Returns an error if factor dimensions are incompatible.
pub fn hosvd3_reconstruct<T: NabledReal>(
    result: &Hosvd3Result<T>,
) -> Result<Array3<T>, TensorError> {
    hosvd3_reconstruct_from_factors_view(
        &result.core.view(),
        &result.u0.view(),
        &result.u1.view(),
        &result.u2.view(),
    )
}

/// Reconstruct a rank-3 tensor from HOSVD factors into `output`.
///
/// # Errors
/// Returns an error if factor dimensions are incompatible or output shape mismatches.
pub fn hosvd3_reconstruct_into<T: NabledReal, S: DataMut<Elem = T>>(
    result: &Hosvd3Result<T>,
    output: &mut ArrayBase<S, Ix3>,
) -> Result<(), TensorError> {
    hosvd3_reconstruct_from_factors_view_into(
        &result.core.view(),
        &result.u0.view(),
        &result.u1.view(),
        &result.u2.view(),
        output,
    )
}

fn mode_axes_order(ndim: usize, mode: usize) -> Result<Vec<usize>, TensorError> {
    if mode >= ndim {
        return Err(TensorError::DimensionMismatch);
    }
    let mut order = Vec::with_capacity(ndim);
    order.push(mode);
    for axis in 0..ndim {
        if axis != mode {
            order.push(axis);
        }
    }
    Ok(order)
}

fn invert_axes_order(order: &[usize]) -> Vec<usize> {
    let mut inverse = vec![0_usize; order.len()];
    for (position, &axis) in order.iter().enumerate() {
        inverse[axis] = position;
    }
    inverse
}

fn matrix_times_unfolded<T, L, R>(
    left: &ArrayBase<L, Ix2>,
    right: &ArrayBase<R, Ix2>,
) -> Result<Array2<T>, TensorError>
where
    T: NabledReal,
    L: Data<Elem = T>,
    R: Data<Elem = T>,
{
    let (rows, inner) = left.dim();
    let (right_inner, cols) = right.dim();
    if inner != right_inner {
        return Err(TensorError::DimensionMismatch);
    }

    let mut output = Array2::<T>::zeros((rows, cols));
    for row in 0..rows {
        for k in 0..inner {
            let lhs = left[[row, k]];
            for col in 0..cols {
                output[[row, col]] += lhs * right[[k, col]];
            }
        }
    }
    Ok(output)
}

fn unfold_mode_nd<T: NabledReal>(
    tensor: &ArrayViewD<'_, T>,
    mode: usize,
) -> Result<Array2<T>, TensorError> {
    let order = mode_axes_order(tensor.ndim(), mode)?;
    let rows = tensor.shape()[mode];
    let cols =
        shape_product(&order[1..].iter().map(|axis| tensor.shape()[*axis]).collect::<Vec<_>>());
    let permuted = tensor.view().permuted_axes(order).to_owned();
    let standard = permuted.as_standard_layout().to_owned();
    standard.into_shape_with_order((rows, cols)).map_err(|_| TensorError::DimensionMismatch)
}

fn mode_n_product_nd<T, M>(
    tensor: &ArrayViewD<'_, T>,
    matrix: &ArrayBase<M, Ix2>,
    mode: usize,
) -> Result<ArrayD<T>, TensorError>
where
    T: NabledReal,
    M: Data<Elem = T>,
{
    let order = mode_axes_order(tensor.ndim(), mode)?;
    let other_shape = order[1..].iter().map(|axis| tensor.shape()[*axis]).collect::<Vec<_>>();
    let unfolded = unfold_mode_nd(tensor, mode)?;
    if matrix.ncols() != unfolded.nrows() {
        return Err(TensorError::DimensionMismatch);
    }

    let projected = matrix_times_unfolded(matrix, &unfolded)?;
    let mut permuted_shape = Vec::with_capacity(tensor.ndim());
    permuted_shape.push(matrix.nrows());
    permuted_shape.extend(other_shape.iter().copied());

    let permuted = projected
        .into_shape_with_order(IxDyn(&permuted_shape))
        .map_err(|_| TensorError::DimensionMismatch)?;
    let inverse = invert_axes_order(&order);
    Ok(permuted.permuted_axes(inverse).to_owned())
}

fn validate_hosvd_nd_ranks(shape: &[usize], ranks: &[usize]) -> Result<(), TensorError> {
    if ranks.len() != shape.len() {
        return Err(TensorError::DimensionMismatch);
    }
    for (&rank, &extent) in ranks.iter().zip(shape.iter()) {
        if rank == 0 || rank > extent {
            return Err(TensorError::DimensionMismatch);
        }
    }
    Ok(())
}

fn frobenius_norm_sq_nd<T: NabledReal>(tensor: &ArrayViewD<'_, T>) -> T {
    tensor.iter().fold(T::zero(), |sum, value| sum + *value * *value)
}

fn core_from_factors_nd<T, S>(
    tensor: &ArrayViewD<'_, T>,
    factors: &[ArrayBase<S, Ix2>],
) -> Result<ArrayD<T>, TensorError>
where
    T: NabledReal,
    S: Data<Elem = T>,
{
    let mut core = tensor.to_owned();
    for (mode, factor) in factors.iter().enumerate() {
        let projection = factor.t();
        core = mode_n_product_nd(&core.view(), &projection, mode)?;
    }
    Ok(core)
}

fn project_except_mode_nd<T, S>(
    tensor: &ArrayViewD<'_, T>,
    factors: &[ArrayBase<S, Ix2>],
    mode: usize,
) -> Result<ArrayD<T>, TensorError>
where
    T: NabledReal,
    S: Data<Elem = T>,
{
    let mut projected = tensor.to_owned();
    for (axis, factor) in factors.iter().enumerate() {
        if axis == mode {
            continue;
        }
        let projection = factor.t();
        projected = mode_n_product_nd(&projected.view(), &projection, axis)?;
    }
    Ok(projected)
}

fn hosvd_nd_impl<T: HosvdNdScalar>(
    tensor: &ArrayViewD<'_, T>,
    ranks: &[usize],
) -> Result<HosvdNdResult<T>, TensorError> {
    validate_tensor_nd_non_empty(tensor)?;
    validate_hosvd_nd_ranks(tensor.shape(), ranks)?;

    let mut factors = Vec::<Array2<T>>::with_capacity(tensor.ndim());
    for (mode, &rank) in ranks.iter().enumerate() {
        let unfolded = unfold_mode_nd(tensor, mode)?;
        let u_full = svd::decompose(&unfolded).map_err(|_| TensorError::DimensionMismatch)?.u;
        if rank > u_full.ncols() {
            return Err(TensorError::DimensionMismatch);
        }
        factors.push(u_full.slice(s![.., 0..rank]).to_owned());
    }

    let core = core_from_factors_nd(tensor, &factors)?;

    Ok(HosvdNdResult { core, factors })
}

fn validate_hooi_config<T: NabledReal>(config: &HooiConfig<T>) -> Result<(), TensorError> {
    if config.max_iterations == 0 || !config.tolerance.is_finite() || config.tolerance <= T::zero()
    {
        return Err(TensorError::DimensionMismatch);
    }
    Ok(())
}

fn hooi_nd_impl<T: HooiNdScalar>(
    tensor: &ArrayViewD<'_, T>,
    ranks: &[usize],
    config: &HooiConfig<T>,
) -> Result<HosvdNdResult<T>, TensorError> {
    validate_tensor_nd_non_empty(tensor)?;
    validate_hosvd_nd_ranks(tensor.shape(), ranks)?;
    validate_hooi_config(config)?;

    let mut estimate = hosvd_nd_impl(tensor, ranks)?;
    let mut previous_core = estimate.core.clone();

    for _ in 0..config.max_iterations {
        for (mode, &rank) in ranks.iter().enumerate() {
            let projected = project_except_mode_nd(tensor, &estimate.factors, mode)?;
            let unfolded = unfold_mode_nd(&projected.view(), mode)?;
            let u_full = svd::decompose(&unfolded).map_err(|_| TensorError::DimensionMismatch)?.u;
            if rank > u_full.ncols() {
                return Err(TensorError::DimensionMismatch);
            }
            estimate.factors[mode] = u_full.slice(s![.., 0..rank]).to_owned();
        }

        estimate.core = core_from_factors_nd(tensor, &estimate.factors)?;

        let mut diff_sq = T::zero();
        for (current, previous) in estimate.core.iter().zip(previous_core.iter()) {
            let delta = *current - *previous;
            diff_sq += delta * delta;
        }
        let base_sq = frobenius_norm_sq_nd(&previous_core.view()).max(T::epsilon());
        if num_traits::Float::sqrt(diff_sq / base_sq) <= config.tolerance {
            break;
        }

        previous_core.clone_from(&estimate.core);
    }

    Ok(estimate)
}

fn validate_hosvd_nd_result<T: NabledReal>(result: &HosvdNdResult<T>) -> Result<(), TensorError> {
    if result.factors.is_empty() || result.core.ndim() != result.factors.len() {
        return Err(TensorError::DimensionMismatch);
    }
    for (mode, factor) in result.factors.iter().enumerate() {
        if factor.ncols() != result.core.shape()[mode] {
            return Err(TensorError::DimensionMismatch);
        }
    }
    Ok(())
}

fn validate_tucker_factors_for_tensor<T, S>(
    tensor: &ArrayViewD<'_, T>,
    factors: &[ArrayBase<S, Ix2>],
) -> Result<(), TensorError>
where
    T: NabledReal,
    S: Data<Elem = T>,
{
    validate_tensor_nd_non_empty(tensor)?;
    if factors.is_empty() || factors.len() != tensor.ndim() {
        return Err(TensorError::DimensionMismatch);
    }
    for (mode, factor) in factors.iter().enumerate() {
        if factor.nrows() != tensor.shape()[mode] || factor.ncols() == 0 || factor.nrows() == 0 {
            return Err(TensorError::DimensionMismatch);
        }
    }
    Ok(())
}

fn validate_tucker_factors_for_core<T, S>(
    core: &ArrayViewD<'_, T>,
    factors: &[ArrayBase<S, Ix2>],
) -> Result<(), TensorError>
where
    T: NabledReal,
    S: Data<Elem = T>,
{
    validate_tensor_nd_non_empty(core)?;
    if factors.is_empty() || factors.len() != core.ndim() {
        return Err(TensorError::DimensionMismatch);
    }
    for (mode, factor) in factors.iter().enumerate() {
        if factor.ncols() != core.shape()[mode] || factor.ncols() == 0 || factor.nrows() == 0 {
            return Err(TensorError::DimensionMismatch);
        }
    }
    Ok(())
}

/// Compute rank-truncated HOSVD for an `N`-D real tensor.
///
/// `ranks` must contain one rank per mode and satisfy `1 <= ranks[mode] <= shape[mode]`.
///
/// # Errors
/// Returns an error if input is empty, ranks are invalid, or factorization fails.
pub fn hosvd_nd<T: HosvdNdScalar>(
    tensor: &ArrayD<T>,
    ranks: &[usize],
) -> Result<HosvdNdResult<T>, TensorError> {
    hosvd_nd_impl(&tensor.view(), ranks)
}

/// Compute rank-truncated HOSVD for an `N`-D real tensor from a view.
///
/// # Errors
/// Returns an error if input is empty, ranks are invalid, or factorization fails.
pub fn hosvd_nd_view<T: HosvdNdScalar>(
    tensor: &ArrayViewD<'_, T>,
    ranks: &[usize],
) -> Result<HosvdNdResult<T>, TensorError> {
    hosvd_nd_impl(tensor, ranks)
}

/// Compute rank-truncated `N`-D Tucker decomposition via HOOI refinement.
///
/// `ranks` must contain one rank per mode and satisfy `1 <= ranks[mode] <= shape[mode]`.
///
/// # Errors
/// Returns an error if input/configuration is invalid or factorization fails.
pub fn hooi_nd<T: HooiNdScalar>(
    tensor: &ArrayD<T>,
    ranks: &[usize],
    config: &HooiConfig<T>,
) -> Result<HosvdNdResult<T>, TensorError> {
    hooi_nd_impl(&tensor.view(), ranks, config)
}

/// Compute rank-truncated `N`-D Tucker decomposition via HOOI refinement from a view.
///
/// # Errors
/// Returns an error if input/configuration is invalid or factorization fails.
pub fn hooi_nd_view<T: HooiNdScalar>(
    tensor: &ArrayViewD<'_, T>,
    ranks: &[usize],
    config: &HooiConfig<T>,
) -> Result<HosvdNdResult<T>, TensorError> {
    hooi_nd_impl(tensor, ranks, config)
}

/// Project an `N`-D tensor into a Tucker core using per-mode factors.
///
/// Applies each factor transpose along the matching mode.
///
/// # Errors
/// Returns an error if tensor/factor dimensions are incompatible.
pub fn tucker_project<T: NabledReal>(
    tensor: &ArrayD<T>,
    factors: &[Array2<T>],
) -> Result<ArrayD<T>, TensorError> {
    tucker_project_view(&tensor.view(), factors)
}

/// Project an `N`-D tensor view into a Tucker core using per-mode factors.
///
/// # Errors
/// Returns an error if tensor/factor dimensions are incompatible.
pub fn tucker_project_view<T: NabledReal>(
    tensor: &ArrayViewD<'_, T>,
    factors: &[Array2<T>],
) -> Result<ArrayD<T>, TensorError> {
    tucker_project_from_factors_view(
        tensor,
        &factors.iter().map(|factor| factor.view()).collect::<Vec<_>>(),
    )
}

/// Project an `N`-D tensor into a Tucker core using per-mode factors into `output`.
///
/// # Errors
/// Returns an error if tensor/factor dimensions are incompatible or output shape mismatches.
pub fn tucker_project_into<T: NabledReal, S: DataMut<Elem = T>>(
    tensor: &ArrayD<T>,
    factors: &[Array2<T>],
    output: &mut ArrayBase<S, IxDyn>,
) -> Result<(), TensorError> {
    tucker_project_view_into(&tensor.view(), factors, output)
}

/// Project an `N`-D tensor view into a Tucker core using per-mode factors into `output`.
///
/// # Errors
/// Returns an error if tensor/factor dimensions are incompatible or output shape mismatches.
pub fn tucker_project_view_into<T: NabledReal, S: DataMut<Elem = T>>(
    tensor: &ArrayViewD<'_, T>,
    factors: &[Array2<T>],
    output: &mut ArrayBase<S, IxDyn>,
) -> Result<(), TensorError> {
    tucker_project_from_factors_view_into(
        tensor,
        &factors.iter().map(|factor| factor.view()).collect::<Vec<_>>(),
        output,
    )
}

/// Project an `N`-D tensor view into a Tucker core using per-mode factor views.
///
/// # Errors
/// Returns an error if tensor/factor dimensions are incompatible.
pub fn tucker_project_from_factors_view<T: NabledReal>(
    tensor: &ArrayViewD<'_, T>,
    factors: &[ArrayView2<'_, T>],
) -> Result<ArrayD<T>, TensorError> {
    validate_tucker_factors_for_tensor(tensor, factors)?;
    core_from_factors_nd(tensor, factors)
}

/// Project an `N`-D tensor view into a Tucker core using per-mode factor views into `output`.
///
/// # Errors
/// Returns an error if tensor/factor dimensions are incompatible or output shape mismatches.
pub fn tucker_project_from_factors_view_into<T: NabledReal, S: DataMut<Elem = T>>(
    tensor: &ArrayViewD<'_, T>,
    factors: &[ArrayView2<'_, T>],
    output: &mut ArrayBase<S, IxDyn>,
) -> Result<(), TensorError> {
    validate_tucker_factors_for_tensor(tensor, factors)?;
    let expected_shape = factors.iter().map(ArrayView2::ncols).collect::<Vec<_>>();
    if output.shape() != expected_shape.as_slice() {
        return Err(TensorError::DimensionMismatch);
    }
    let projected = core_from_factors_nd(tensor, factors)?;
    output.assign(&projected);
    Ok(())
}

/// Expand a Tucker core into the original tensor space using per-mode factors.
///
/// # Errors
/// Returns an error if core/factor dimensions are incompatible.
pub fn tucker_expand<T: NabledReal>(
    core: &ArrayD<T>,
    factors: &[Array2<T>],
) -> Result<ArrayD<T>, TensorError> {
    tucker_expand_view(&core.view(), factors)
}

/// Expand a Tucker core view into the original tensor space using per-mode factors.
///
/// # Errors
/// Returns an error if core/factor dimensions are incompatible.
pub fn tucker_expand_view<T: NabledReal>(
    core: &ArrayViewD<'_, T>,
    factors: &[Array2<T>],
) -> Result<ArrayD<T>, TensorError> {
    tucker_expand_from_factors_view(
        core,
        &factors.iter().map(|factor| factor.view()).collect::<Vec<_>>(),
    )
}

/// Expand a Tucker core into `output` using per-mode factors.
///
/// # Errors
/// Returns an error if core/factor dimensions or output shape are incompatible.
pub fn tucker_expand_into<T: NabledReal, S: DataMut<Elem = T>>(
    core: &ArrayD<T>,
    factors: &[Array2<T>],
    output: &mut ArrayBase<S, IxDyn>,
) -> Result<(), TensorError> {
    tucker_expand_view_into(&core.view(), factors, output)
}

/// Expand a Tucker core view into `output` using per-mode factors.
///
/// # Errors
/// Returns an error if core/factor dimensions or output shape are incompatible.
pub fn tucker_expand_view_into<T: NabledReal, S: DataMut<Elem = T>>(
    core: &ArrayViewD<'_, T>,
    factors: &[Array2<T>],
    output: &mut ArrayBase<S, IxDyn>,
) -> Result<(), TensorError> {
    tucker_expand_from_factors_view_into(
        core,
        &factors.iter().map(|factor| factor.view()).collect::<Vec<_>>(),
        output,
    )
}

/// Expand a Tucker core view into the original tensor space using per-mode factor views.
///
/// # Errors
/// Returns an error if core/factor dimensions are incompatible.
pub fn tucker_expand_from_factors_view<T: NabledReal>(
    core: &ArrayViewD<'_, T>,
    factors: &[ArrayView2<'_, T>],
) -> Result<ArrayD<T>, TensorError> {
    validate_tucker_factors_for_core(core, factors)?;
    let mut output_shape = Vec::<usize>::with_capacity(factors.len());
    output_shape.extend(factors.iter().map(ArrayView2::nrows));
    let mut output = ArrayD::<T>::zeros(IxDyn(&output_shape));
    tucker_expand_from_factors_view_into(core, factors, &mut output)?;
    Ok(output)
}

/// Expand a Tucker core view into `output` using per-mode factor views.
///
/// # Errors
/// Returns an error if core/factor dimensions or output shape are incompatible.
pub fn tucker_expand_from_factors_view_into<T: NabledReal, S: DataMut<Elem = T>>(
    core: &ArrayViewD<'_, T>,
    factors: &[ArrayView2<'_, T>],
    output: &mut ArrayBase<S, IxDyn>,
) -> Result<(), TensorError> {
    validate_tucker_factors_for_core(core, factors)?;
    let expected_shape = factors.iter().map(ArrayView2::nrows).collect::<Vec<_>>();
    if output.shape() != expected_shape.as_slice() {
        return Err(TensorError::DimensionMismatch);
    }

    let mut tensor = core.to_owned();
    for (mode, factor) in factors.iter().enumerate() {
        tensor = mode_n_product_nd(&tensor.view(), factor, mode)?;
    }
    output.assign(&tensor);
    Ok(())
}

/// Reconstruct an `N`-D tensor from HOSVD/Tucker factors.
///
/// # Errors
/// Returns an error if factor dimensions are incompatible.
pub fn hosvd_nd_reconstruct<T: NabledReal>(
    result: &HosvdNdResult<T>,
) -> Result<ArrayD<T>, TensorError> {
    validate_hosvd_nd_result(result)?;
    hosvd_nd_reconstruct_from_factors_view(
        &result.core.view(),
        &result.factors.iter().map(|factor| factor.view()).collect::<Vec<_>>(),
    )
}

/// Reconstruct an `N`-D tensor from HOSVD/Tucker factors into `output`.
///
/// # Errors
/// Returns an error if factor dimensions are incompatible.
pub fn hosvd_nd_reconstruct_into<T: NabledReal, S: DataMut<Elem = T>>(
    result: &HosvdNdResult<T>,
    output: &mut ArrayBase<S, IxDyn>,
) -> Result<(), TensorError> {
    validate_hosvd_nd_result(result)?;
    hosvd_nd_reconstruct_from_factors_view_into(
        &result.core.view(),
        &result.factors.iter().map(|factor| factor.view()).collect::<Vec<_>>(),
        output,
    )
}

/// Reconstruct an `N`-D tensor from HOSVD/Tucker factor views.
///
/// # Errors
/// Returns an error if factor dimensions are incompatible.
pub fn hosvd_nd_reconstruct_from_factors_view<T: NabledReal>(
    core: &ArrayViewD<'_, T>,
    factors: &[ArrayView2<'_, T>],
) -> Result<ArrayD<T>, TensorError> {
    tucker_expand_from_factors_view(core, factors)
}

/// Reconstruct an `N`-D tensor from HOSVD/Tucker factor views into `output`.
///
/// # Errors
/// Returns an error if factor dimensions are incompatible.
pub fn hosvd_nd_reconstruct_from_factors_view_into<T: NabledReal, S: DataMut<Elem = T>>(
    core: &ArrayViewD<'_, T>,
    factors: &[ArrayView2<'_, T>],
    output: &mut ArrayBase<S, IxDyn>,
) -> Result<(), TensorError> {
    tucker_expand_from_factors_view_into(core, factors, output)
}

fn tt_svd_select_rank<T: NabledReal>(
    singular_values: &Array1<T>,
    tolerance: T,
    max_rank: usize,
) -> usize {
    if singular_values.is_empty() {
        return 0;
    }
    let max_sigma = singular_values.iter().copied().fold(T::zero(), |current, value| {
        if value.abs() > current { value.abs() } else { current }
    });
    let cutoff = tolerance.max(T::epsilon()) * max_sigma;
    let kept = singular_values.iter().filter(|value| value.abs() > cutoff).count();
    kept.max(1).min(max_rank).min(singular_values.len())
}

fn tt_svd_impl<T: TtSvdScalar>(
    tensor: &ArrayViewD<'_, T>,
    config: &TtSvdConfig<T>,
) -> Result<TensorTrainResult<T>, TensorError> {
    validate_tensor_nd_non_empty(tensor)?;
    let shape = tensor.shape().to_vec();
    let ndim = shape.len();
    let max_rank = config.max_rank.unwrap_or(usize::MAX);
    if max_rank == 0 {
        return Err(TensorError::DimensionMismatch);
    }

    if ndim == 1 {
        let mut core = Array3::<T>::zeros((1, shape[0], 1));
        for i in 0..shape[0] {
            core[[0, i, 0]] = tensor[IxDyn(&[i])];
        }
        return Ok(TensorTrainResult { cores: vec![core], shape });
    }

    let mut cores = Vec::<Array3<T>>::with_capacity(ndim);
    // TT-SVD repeatedly reshapes the working tensor into matrices. Normalize the initial dynamic
    // tensor into standard layout once so view-based callers are not rejected purely because the
    // source NumPy/ndarray storage is strided or Fortran-ordered.
    let mut current = tensor.as_standard_layout().to_owned();
    let mut left_rank = 1_usize;

    for mode in 0..(ndim - 1) {
        let mode_extent = shape[mode];
        let right_extent = shape_product(&shape[(mode + 1)..]);
        let matrix = current
            .into_shape_with_order((left_rank * mode_extent, right_extent))
            .map_err(|_| TensorError::DimensionMismatch)?;
        let decomposition = svd::decompose(&matrix).map_err(|_| TensorError::DimensionMismatch)?;
        let rank = tt_svd_select_rank(&decomposition.singular_values, config.tolerance, max_rank)
            .min(decomposition.u.ncols());
        if rank == 0 {
            return Err(TensorError::DimensionMismatch);
        }

        let u = decomposition.u.slice(s![.., 0..rank]).to_owned();
        let mut v = decomposition.vt.slice(s![0..rank, ..]).to_owned();
        for row in 0..rank {
            let sigma = decomposition.singular_values[row];
            for col in 0..right_extent {
                v[[row, col]] *= sigma;
            }
        }

        let core = u
            .into_shape_with_order((left_rank, mode_extent, rank))
            .map_err(|_| TensorError::DimensionMismatch)?;
        cores.push(core);

        let mut next_shape = Vec::<usize>::with_capacity(ndim - mode);
        next_shape.push(rank);
        next_shape.extend_from_slice(&shape[(mode + 1)..]);
        current = v
            .into_shape_with_order(IxDyn(&next_shape))
            .map_err(|_| TensorError::DimensionMismatch)?;
        left_rank = rank;
    }

    let last_extent = shape[ndim - 1];
    let last_core = current
        .into_shape_with_order((left_rank, last_extent, 1))
        .map_err(|_| TensorError::DimensionMismatch)?;
    cores.push(last_core);

    Ok(TensorTrainResult { cores, shape })
}

fn validate_tt_result<T: NabledReal>(result: &TensorTrainResult<T>) -> Result<(), TensorError> {
    if result.cores.is_empty() || result.shape.len() != result.cores.len() {
        return Err(TensorError::DimensionMismatch);
    }
    if result.cores[0].dim().0 != 1 || result.cores[result.cores.len() - 1].dim().2 != 1 {
        return Err(TensorError::DimensionMismatch);
    }
    for (mode, core) in result.cores.iter().enumerate() {
        if core.dim().1 != result.shape[mode] || core.is_empty() {
            return Err(TensorError::DimensionMismatch);
        }
        if mode + 1 < result.cores.len() && core.dim().2 != result.cores[mode + 1].dim().0 {
            return Err(TensorError::DimensionMismatch);
        }
    }
    Ok(())
}

fn validate_tt_cores<T, S>(cores: &[ArrayBase<S, Ix3>]) -> Result<Vec<usize>, TensorError>
where
    T: NabledReal,
    S: Data<Elem = T>,
{
    if cores.is_empty() {
        return Err(TensorError::DimensionMismatch);
    }
    if cores[0].dim().0 != 1 || cores[cores.len() - 1].dim().2 != 1 {
        return Err(TensorError::DimensionMismatch);
    }

    let mut shape = Vec::with_capacity(cores.len());
    for (mode, core) in cores.iter().enumerate() {
        if core.is_empty() {
            return Err(TensorError::DimensionMismatch);
        }
        shape.push(core.dim().1);
        if mode + 1 < cores.len() && core.dim().2 != cores[mode + 1].dim().0 {
            return Err(TensorError::DimensionMismatch);
        }
    }
    Ok(shape)
}

fn validate_tt_binary_inputs<T, L, R>(
    left: &[ArrayBase<L, Ix3>],
    right: &[ArrayBase<R, Ix3>],
) -> Result<Vec<usize>, TensorError>
where
    T: NabledReal,
    L: Data<Elem = T>,
    R: Data<Elem = T>,
{
    let left_shape = validate_tt_cores(left)?;
    let right_shape = validate_tt_cores(right)?;
    if left_shape != right_shape || left.len() != right.len() {
        return Err(TensorError::DimensionMismatch);
    }
    Ok(left_shape)
}

fn validate_tt_round_config<T: NabledReal>(config: &TtRoundConfig<T>) -> Result<(), TensorError> {
    if !config.tolerance.is_finite() || config.tolerance < T::zero() {
        return Err(TensorError::DimensionMismatch);
    }
    if let Some(max_rank) = config.max_rank
        && max_rank == 0
    {
        return Err(TensorError::DimensionMismatch);
    }
    Ok(())
}

fn tt_factor_core_with_svd<T, S>(
    core: &ArrayBase<S, Ix3>,
    truncation: Option<(T, usize)>,
) -> Result<(Array3<T>, Array2<T>), TensorError>
where
    T: TtSvdScalar,
    S: Data<Elem = T>,
{
    let (left_rank, mode_extent, right_rank) = core.dim();
    let mut matrix = Array2::<T>::zeros((left_rank * mode_extent, right_rank));
    for left in 0..left_rank {
        for mode in 0..mode_extent {
            let row = left * mode_extent + mode;
            for right in 0..right_rank {
                matrix[[row, right]] = core[[left, mode, right]];
            }
        }
    }
    let decomposition = svd::decompose(&matrix).map_err(|_| TensorError::DimensionMismatch)?;
    let available_rank = decomposition.singular_values.len();
    if available_rank == 0 {
        return Err(TensorError::DimensionMismatch);
    }

    let kept_rank = match truncation {
        Some((tolerance, max_rank)) => {
            tt_svd_select_rank(&decomposition.singular_values, tolerance, max_rank)
                .min(available_rank)
        }
        None => available_rank,
    };
    if kept_rank == 0 {
        return Err(TensorError::DimensionMismatch);
    }

    let q = decomposition
        .u
        .slice(s![.., 0..kept_rank])
        .to_owned()
        .into_shape_with_order((left_rank, mode_extent, kept_rank))
        .map_err(|_| TensorError::DimensionMismatch)?;

    let mut transfer = decomposition.vt.slice(s![0..kept_rank, ..]).to_owned();
    for row in 0..kept_rank {
        let sigma = decomposition.singular_values[row];
        for col in 0..right_rank {
            transfer[[row, col]] *= sigma;
        }
    }

    Ok((q, transfer))
}

fn tt_apply_transfer_to_next<T, S>(
    transfer: &Array2<T>,
    next: &ArrayBase<S, Ix3>,
) -> Result<Array3<T>, TensorError>
where
    T: NabledReal,
    S: Data<Elem = T>,
{
    let (transfer_rows, transfer_cols) = transfer.dim();
    let (next_left, next_mode, next_right) = next.dim();
    if transfer_cols != next_left {
        return Err(TensorError::DimensionMismatch);
    }

    let mut output = Array3::<T>::zeros((transfer_rows, next_mode, next_right));
    for row in 0..transfer_rows {
        for left in 0..transfer_cols {
            let weight = transfer[[row, left]];
            for mode_index in 0..next_mode {
                for right in 0..next_right {
                    output[[row, mode_index, right]] += weight * next[[left, mode_index, right]];
                }
            }
        }
    }
    Ok(output)
}

fn tt_factor_core_right_with_svd<T, S>(
    core: &ArrayBase<S, Ix3>,
) -> Result<(Array2<T>, Array3<T>), TensorError>
where
    T: TtSvdScalar,
    S: Data<Elem = T>,
{
    let (left_rank, mode_extent, right_rank) = core.dim();
    let mut matrix = Array2::<T>::zeros((left_rank, mode_extent * right_rank));
    for left in 0..left_rank {
        for mode in 0..mode_extent {
            for right in 0..right_rank {
                let col = mode * right_rank + right;
                matrix[[left, col]] = core[[left, mode, right]];
            }
        }
    }
    let decomposition = svd::decompose(&matrix).map_err(|_| TensorError::DimensionMismatch)?;
    let available_rank = decomposition.singular_values.len();
    if available_rank == 0 {
        return Err(TensorError::DimensionMismatch);
    }

    let mut left_transfer = decomposition.u.slice(s![.., 0..available_rank]).to_owned();
    for col in 0..available_rank {
        let sigma = decomposition.singular_values[col];
        for row in 0..left_rank {
            left_transfer[[row, col]] *= sigma;
        }
    }

    let right_core = decomposition
        .vt
        .slice(s![0..available_rank, ..])
        .to_owned()
        .into_shape_with_order((available_rank, mode_extent, right_rank))
        .map_err(|_| TensorError::DimensionMismatch)?;

    Ok((left_transfer, right_core))
}

fn tt_apply_transfer_to_previous<T, S>(
    previous: &ArrayBase<S, Ix3>,
    transfer: &Array2<T>,
) -> Result<Array3<T>, TensorError>
where
    T: NabledReal,
    S: Data<Elem = T>,
{
    let (previous_left, previous_mode, previous_right) = previous.dim();
    let (transfer_rows, transfer_cols) = transfer.dim();
    if previous_right != transfer_rows {
        return Err(TensorError::DimensionMismatch);
    }

    let mut output = Array3::<T>::zeros((previous_left, previous_mode, transfer_cols));
    for left in 0..previous_left {
        for mode_index in 0..previous_mode {
            for old_rank in 0..previous_right {
                let value = previous[[left, mode_index, old_rank]];
                for new_rank in 0..transfer_cols {
                    output[[left, mode_index, new_rank]] += value * transfer[[old_rank, new_rank]];
                }
            }
        }
    }
    Ok(output)
}

fn tt_right_orthogonalize_impl<T, S>(
    cores: &[ArrayBase<S, Ix3>],
) -> Result<TensorTrainResult<T>, TensorError>
where
    T: TtSvdScalar,
    S: Data<Elem = T>,
{
    let shape = validate_tt_cores(cores)?;
    if cores.len() <= 1 {
        return Ok(TensorTrainResult {
            cores: cores.iter().map(ArrayBase::to_owned).collect(),
            shape,
        });
    }

    let mut output_rev = Vec::<Array3<T>>::with_capacity(cores.len());
    let (left_transfer, orth_core) = tt_factor_core_right_with_svd(&cores[cores.len() - 1])?;
    output_rev.push(orth_core);
    let mut previous = tt_apply_transfer_to_previous(&cores[cores.len() - 2], &left_transfer)?;

    for mode in (1..cores.len() - 1).rev() {
        let (left_transfer, orth_core) = tt_factor_core_right_with_svd(&previous)?;
        output_rev.push(orth_core);
        previous = tt_apply_transfer_to_previous(&cores[mode - 1], &left_transfer)?;
    }

    output_rev.push(previous);
    output_rev.reverse();
    let transformed = TensorTrainResult { cores: output_rev, shape };
    validate_tt_result(&transformed)?;
    Ok(transformed)
}

fn tt_transform_impl<T, S>(
    cores: &[ArrayBase<S, Ix3>],
    truncation: Option<(T, usize)>,
) -> Result<TensorTrainResult<T>, TensorError>
where
    T: TtSvdScalar,
    S: Data<Elem = T>,
{
    let shape = validate_tt_cores(cores)?;
    if cores.len() <= 1 {
        return Ok(TensorTrainResult {
            cores: cores.iter().map(ArrayBase::to_owned).collect(),
            shape,
        });
    }

    let mut output = Vec::<Array3<T>>::with_capacity(cores.len());
    let (orth_core, transfer) = tt_factor_core_with_svd(&cores[0], truncation)?;
    output.push(orth_core);
    let mut next = tt_apply_transfer_to_next(&transfer, &cores[1])?;

    for mode in 1..(cores.len() - 1) {
        let (orth_core, transfer) = tt_factor_core_with_svd(&next, truncation)?;
        output.push(orth_core);
        if mode + 1 < cores.len() {
            next = tt_apply_transfer_to_next(&transfer, &cores[mode + 1])?;
        }
    }

    output.push(next);
    let transformed = TensorTrainResult { cores: output, shape };
    validate_tt_result(&transformed)?;
    Ok(transformed)
}

/// Compute Tensor-Train decomposition for an `N`-D real tensor via TT-SVD.
///
/// # Errors
/// Returns an error if the tensor is empty/scalar or if decomposition constraints are invalid.
pub fn tt_svd<T: TtSvdScalar>(
    tensor: &ArrayD<T>,
    config: &TtSvdConfig<T>,
) -> Result<TensorTrainResult<T>, TensorError> {
    tt_svd_impl(&tensor.view(), config)
}

/// Compute Tensor-Train decomposition for an `N`-D real tensor view via TT-SVD.
///
/// # Errors
/// Returns an error if the tensor is empty/scalar or if decomposition constraints are invalid.
pub fn tt_svd_view<T: TtSvdScalar>(
    tensor: &ArrayViewD<'_, T>,
    config: &TtSvdConfig<T>,
) -> Result<TensorTrainResult<T>, TensorError> {
    tt_svd_impl(tensor, config)
}

/// Left-orthogonalize TT cores while preserving represented tensor values.
///
/// All cores except the last become left-orthonormal in matricized form
/// (`(r_k * n_k) x r_{k+1}`), with scaling absorbed into trailing cores.
///
/// # Errors
/// Returns an error if TT core dimensions are incompatible.
pub fn tt_orthogonalize_left<T: TtSvdScalar>(
    result: &TensorTrainResult<T>,
) -> Result<TensorTrainResult<T>, TensorError> {
    tt_orthogonalize_left_from_cores_view(
        &result.cores.iter().map(|core| core.view()).collect::<Vec<_>>(),
    )
}

/// Left-orthogonalize borrowed TT cores while preserving represented tensor values.
///
/// # Errors
/// Returns an error if TT core dimensions are incompatible.
pub fn tt_orthogonalize_left_from_cores_view<T: TtSvdScalar>(
    cores: &[ArrayView3<'_, T>],
) -> Result<TensorTrainResult<T>, TensorError> {
    tt_transform_impl(cores, None)
}

/// Right-orthogonalize TT cores while preserving represented tensor values.
///
/// All cores except the first become right-orthonormal in matricized form
/// (`r_k x (n_k * r_{k+1})`), with scaling absorbed into leading cores.
///
/// # Errors
/// Returns an error if TT core dimensions are incompatible.
pub fn tt_orthogonalize_right<T: TtSvdScalar>(
    result: &TensorTrainResult<T>,
) -> Result<TensorTrainResult<T>, TensorError> {
    tt_orthogonalize_right_from_cores_view(
        &result.cores.iter().map(|core| core.view()).collect::<Vec<_>>(),
    )
}

/// Right-orthogonalize borrowed TT cores while preserving represented tensor values.
///
/// # Errors
/// Returns an error if TT core dimensions are incompatible.
pub fn tt_orthogonalize_right_from_cores_view<T: TtSvdScalar>(
    cores: &[ArrayView3<'_, T>],
) -> Result<TensorTrainResult<T>, TensorError> {
    tt_right_orthogonalize_impl(cores)
}

/// Round/compress a TT decomposition with optional rank truncation.
///
/// This performs a right-orthogonalization sweep followed by a left truncation
/// sweep, reducing intermediate TT ranks using singular-value tolerance and
/// optional max rank.
///
/// # Errors
/// Returns an error if TT core dimensions or configuration are invalid.
pub fn tt_round<T: TtSvdScalar>(
    result: &TensorTrainResult<T>,
    config: &TtRoundConfig<T>,
) -> Result<TensorTrainResult<T>, TensorError> {
    tt_round_from_cores_view(
        &result.cores.iter().map(|core| core.view()).collect::<Vec<_>>(),
        config,
    )
}

/// Round/compress borrowed TT cores with optional rank truncation.
///
/// # Errors
/// Returns an error if TT core dimensions or configuration are invalid.
pub fn tt_round_from_cores_view<T: TtSvdScalar>(
    cores: &[ArrayView3<'_, T>],
    config: &TtRoundConfig<T>,
) -> Result<TensorTrainResult<T>, TensorError> {
    validate_tt_round_config(config)?;
    let orthogonalized = tt_right_orthogonalize_impl(cores)?;
    let max_rank = config.max_rank.unwrap_or(usize::MAX);
    tt_transform_impl(
        &orthogonalized.cores.iter().map(|core| core.view()).collect::<Vec<_>>(),
        Some((config.tolerance, max_rank)),
    )
}

/// Compute inner product of two Tensor-Train tensors with matching shape.
///
/// # Errors
/// Returns an error if TT core dimensions are incompatible or shapes mismatch.
pub fn tt_inner<T: NabledReal>(
    left: &TensorTrainResult<T>,
    right: &TensorTrainResult<T>,
) -> Result<T, TensorError> {
    tt_inner_from_cores_view(
        &left.cores.iter().map(|core| core.view()).collect::<Vec<_>>(),
        &right.cores.iter().map(|core| core.view()).collect::<Vec<_>>(),
    )
}

/// Compute inner product of two borrowed Tensor-Train tensors with matching shape.
///
/// # Errors
/// Returns an error if TT core dimensions are incompatible or shapes mismatch.
pub fn tt_inner_from_cores_view<T: NabledReal>(
    left: &[ArrayView3<'_, T>],
    right: &[ArrayView3<'_, T>],
) -> Result<T, TensorError> {
    let _shape = validate_tt_binary_inputs(left, right)?;

    let mut contraction = Array2::<T>::from_elem((1, 1), T::one());
    for mode in 0..left.len() {
        let left_core = &left[mode];
        let right_core = &right[mode];
        let (left_prev_rank, mode_extent, left_next_rank) = left_core.dim();
        let (right_prev_rank, right_mode_extent, right_next_rank) = right_core.dim();

        if mode_extent != right_mode_extent
            || contraction.dim() != (left_prev_rank, right_prev_rank)
        {
            return Err(TensorError::DimensionMismatch);
        }

        let mut next = Array2::<T>::zeros((left_next_rank, right_next_rank));
        for left_prev in 0..left_prev_rank {
            for right_prev in 0..right_prev_rank {
                let coeff = contraction[[left_prev, right_prev]];
                for mode_index in 0..mode_extent {
                    let left_value_row = left_core.slice(s![left_prev, mode_index, ..]);
                    let right_value_row = right_core.slice(s![right_prev, mode_index, ..]);
                    for left_next in 0..left_next_rank {
                        let left_value = left_value_row[left_next];
                        for right_next in 0..right_next_rank {
                            next[[left_next, right_next]] +=
                                coeff * left_value * right_value_row[right_next];
                        }
                    }
                }
            }
        }
        contraction = next;
    }

    if contraction.dim() != (1, 1) {
        return Err(TensorError::DimensionMismatch);
    }
    Ok(contraction[[0, 0]])
}

/// Compute Frobenius norm of a Tensor-Train tensor.
///
/// # Errors
/// Returns an error if TT core dimensions are incompatible.
pub fn tt_norm<T: NabledReal>(result: &TensorTrainResult<T>) -> Result<T, TensorError> {
    tt_norm_from_cores_view(&result.cores.iter().map(|core| core.view()).collect::<Vec<_>>())
}

/// Compute Frobenius norm of borrowed Tensor-Train cores.
///
/// # Errors
/// Returns an error if TT core dimensions are incompatible.
pub fn tt_norm_from_cores_view<T: NabledReal>(
    cores: &[ArrayView3<'_, T>],
) -> Result<T, TensorError> {
    let squared = tt_inner_from_cores_view(cores, cores)?;
    if squared < T::zero() && squared.abs() > T::epsilon() * T::from_f64(64.0).unwrap_or(T::one()) {
        return Err(TensorError::DimensionMismatch);
    }
    Ok(if squared <= T::zero() { T::zero() } else { squared.sqrt() })
}

/// Add two Tensor-Train tensors with identical shape.
///
/// # Errors
/// Returns an error if TT core dimensions are incompatible or shapes mismatch.
pub fn tt_add<T: NabledReal>(
    left: &TensorTrainResult<T>,
    right: &TensorTrainResult<T>,
) -> Result<TensorTrainResult<T>, TensorError> {
    tt_add_from_cores_view(
        &left.cores.iter().map(|core| core.view()).collect::<Vec<_>>(),
        &right.cores.iter().map(|core| core.view()).collect::<Vec<_>>(),
    )
}

/// Add two borrowed Tensor-Train tensors with identical shape.
///
/// # Errors
/// Returns an error if TT core dimensions are incompatible or shapes mismatch.
pub fn tt_add_from_cores_view<T: NabledReal>(
    left: &[ArrayView3<'_, T>],
    right: &[ArrayView3<'_, T>],
) -> Result<TensorTrainResult<T>, TensorError> {
    let shape = validate_tt_binary_inputs(left, right)?;

    if left.len() == 1 {
        let mut core = left[0].to_owned();
        for mode_index in 0..core.dim().1 {
            core[[0, mode_index, 0]] += right[0][[0, mode_index, 0]];
        }
        return Ok(TensorTrainResult { cores: vec![core], shape });
    }

    let mut cores = Vec::<Array3<T>>::with_capacity(left.len());

    {
        let left_core = &left[0];
        let right_core = &right[0];
        let (_, mode_extent, left_rank) = left_core.dim();
        let (_, _, right_rank) = right_core.dim();
        let mut core = Array3::<T>::zeros((1, mode_extent, left_rank + right_rank));
        for mode_index in 0..mode_extent {
            for rank in 0..left_rank {
                core[[0, mode_index, rank]] = left_core[[0, mode_index, rank]];
            }
            for rank in 0..right_rank {
                core[[0, mode_index, left_rank + rank]] = right_core[[0, mode_index, rank]];
            }
        }
        cores.push(core);
    }

    for mode in 1..(left.len() - 1) {
        let left_core = &left[mode];
        let right_core = &right[mode];
        let (left_prev, mode_extent, left_next) = left_core.dim();
        let (right_prev, right_mode_extent, right_next) = right_core.dim();
        if mode_extent != right_mode_extent {
            return Err(TensorError::DimensionMismatch);
        }
        let mut core =
            Array3::<T>::zeros((left_prev + right_prev, mode_extent, left_next + right_next));
        for prev in 0..left_prev {
            for mode_index in 0..mode_extent {
                for next in 0..left_next {
                    core[[prev, mode_index, next]] = left_core[[prev, mode_index, next]];
                }
            }
        }
        for prev in 0..right_prev {
            for mode_index in 0..mode_extent {
                for next in 0..right_next {
                    core[[left_prev + prev, mode_index, left_next + next]] =
                        right_core[[prev, mode_index, next]];
                }
            }
        }
        cores.push(core);
    }

    {
        let mode = left.len() - 1;
        let left_core = &left[mode];
        let right_core = &right[mode];
        let (left_prev, mode_extent, left_last) = left_core.dim();
        let (right_prev, right_mode_extent, right_last) = right_core.dim();
        if mode_extent != right_mode_extent || left_last != 1 || right_last != 1 {
            return Err(TensorError::DimensionMismatch);
        }
        let mut core = Array3::<T>::zeros((left_prev + right_prev, mode_extent, 1));
        for prev in 0..left_prev {
            for mode_index in 0..mode_extent {
                core[[prev, mode_index, 0]] = left_core[[prev, mode_index, 0]];
            }
        }
        for prev in 0..right_prev {
            for mode_index in 0..mode_extent {
                core[[left_prev + prev, mode_index, 0]] = right_core[[prev, mode_index, 0]];
            }
        }
        cores.push(core);
    }

    let result = TensorTrainResult { cores, shape };
    validate_tt_result(&result)?;
    Ok(result)
}

/// Compute elementwise (Hadamard) product of two Tensor-Train tensors.
///
/// # Errors
/// Returns an error if TT core dimensions are incompatible or shapes mismatch.
pub fn tt_hadamard<T: NabledReal>(
    left: &TensorTrainResult<T>,
    right: &TensorTrainResult<T>,
) -> Result<TensorTrainResult<T>, TensorError> {
    tt_hadamard_from_cores_view(
        &left.cores.iter().map(|core| core.view()).collect::<Vec<_>>(),
        &right.cores.iter().map(|core| core.view()).collect::<Vec<_>>(),
    )
}

/// Compute elementwise (Hadamard) product of two borrowed Tensor-Train tensors.
///
/// # Errors
/// Returns an error if TT core dimensions are incompatible or shapes mismatch.
pub fn tt_hadamard_from_cores_view<T: NabledReal>(
    left: &[ArrayView3<'_, T>],
    right: &[ArrayView3<'_, T>],
) -> Result<TensorTrainResult<T>, TensorError> {
    let shape = validate_tt_binary_inputs(left, right)?;

    let mut cores = Vec::<Array3<T>>::with_capacity(left.len());
    for mode in 0..left.len() {
        let left_core = &left[mode];
        let right_core = &right[mode];
        let (left_prev, mode_extent, left_next) = left_core.dim();
        let (right_prev, right_mode_extent, right_next) = right_core.dim();
        if mode_extent != right_mode_extent {
            return Err(TensorError::DimensionMismatch);
        }

        let mut core =
            Array3::<T>::zeros((left_prev * right_prev, mode_extent, left_next * right_next));
        for left_prev_rank in 0..left_prev {
            for right_prev_rank in 0..right_prev {
                let output_prev = left_prev_rank * right_prev + right_prev_rank;
                for mode_index in 0..mode_extent {
                    for left_next_rank in 0..left_next {
                        let left_value = left_core[[left_prev_rank, mode_index, left_next_rank]];
                        for right_next_rank in 0..right_next {
                            let output_next = left_next_rank * right_next + right_next_rank;
                            core[[output_prev, mode_index, output_next]] = left_value
                                * right_core[[right_prev_rank, mode_index, right_next_rank]];
                        }
                    }
                }
            }
        }
        cores.push(core);
    }

    let result = TensorTrainResult { cores, shape };
    validate_tt_result(&result)?;
    Ok(result)
}

/// Compute Hadamard product followed by TT rank-rounding.
///
/// # Errors
/// Returns an error if TT core dimensions/configuration are invalid.
pub fn tt_hadamard_round<T: TtSvdScalar>(
    left: &TensorTrainResult<T>,
    right: &TensorTrainResult<T>,
    config: &TtRoundConfig<T>,
) -> Result<TensorTrainResult<T>, TensorError> {
    tt_hadamard_round_from_cores_view(
        &left.cores.iter().map(|core| core.view()).collect::<Vec<_>>(),
        &right.cores.iter().map(|core| core.view()).collect::<Vec<_>>(),
        config,
    )
}

/// Compute Hadamard product of borrowed TT cores followed by TT rank-rounding.
///
/// # Errors
/// Returns an error if TT core dimensions/configuration are invalid.
pub fn tt_hadamard_round_from_cores_view<T: TtSvdScalar>(
    left: &[ArrayView3<'_, T>],
    right: &[ArrayView3<'_, T>],
    config: &TtRoundConfig<T>,
) -> Result<TensorTrainResult<T>, TensorError> {
    let product = tt_hadamard_from_cores_view(left, right)?;
    tt_round_from_cores_view(
        &product.cores.iter().map(|core| core.view()).collect::<Vec<_>>(),
        config,
    )
}

/// Reconstruct an `N`-D real tensor from Tensor-Train cores.
///
/// # Errors
/// Returns an error if TT core dimensions are incompatible.
pub fn tt_svd_reconstruct<T: NabledReal>(
    result: &TensorTrainResult<T>,
) -> Result<ArrayD<T>, TensorError> {
    validate_tt_result(result)?;
    tt_svd_reconstruct_from_cores_view(
        &result.cores.iter().map(|core| core.view()).collect::<Vec<_>>(),
    )
}

/// Reconstruct an `N`-D real tensor from Tensor-Train cores into `output`.
///
/// # Errors
/// Returns an error if TT core dimensions are incompatible or output shape mismatches.
pub fn tt_svd_reconstruct_into<T: NabledReal, S: DataMut<Elem = T>>(
    result: &TensorTrainResult<T>,
    output: &mut ArrayBase<S, IxDyn>,
) -> Result<(), TensorError> {
    validate_tt_result(result)?;
    tt_svd_reconstruct_from_cores_view_into(
        &result.cores.iter().map(|core| core.view()).collect::<Vec<_>>(),
        output,
    )
}

/// Reconstruct an `N`-D real tensor from borrowed Tensor-Train core views.
///
/// # Errors
/// Returns an error if TT core dimensions are incompatible.
pub fn tt_svd_reconstruct_from_cores_view<T: NabledReal>(
    cores: &[ArrayView3<'_, T>],
) -> Result<ArrayD<T>, TensorError> {
    let shape = validate_tt_cores(cores)?;
    let mut output = ArrayD::<T>::zeros(IxDyn(&shape));
    tt_svd_reconstruct_from_cores_view_into(cores, &mut output)?;
    Ok(output)
}

/// Reconstruct an `N`-D real tensor from borrowed Tensor-Train core views into `output`.
///
/// # Errors
/// Returns an error if TT core dimensions are incompatible or output shape mismatches.
pub fn tt_svd_reconstruct_from_cores_view_into<T: NabledReal, S: DataMut<Elem = T>>(
    cores: &[ArrayView3<'_, T>],
    output: &mut ArrayBase<S, IxDyn>,
) -> Result<(), TensorError> {
    let shape = validate_tt_cores(cores)?;
    if output.shape() != shape.as_slice() {
        return Err(TensorError::DimensionMismatch);
    }

    let (first_left, first_extent, first_right) = cores[0].dim();
    if first_left != 1 {
        return Err(TensorError::DimensionMismatch);
    }
    let mut accumulated = Array2::<T>::zeros((first_extent, first_right));
    for i in 0..first_extent {
        for r in 0..first_right {
            accumulated[[i, r]] = cores[0][[0, i, r]];
        }
    }
    let mut rows = first_extent;

    for core in cores.iter().skip(1) {
        let (left_rank, mode_extent, right_rank) = core.dim();
        if accumulated.ncols() != left_rank {
            return Err(TensorError::DimensionMismatch);
        }

        let mut next = Array2::<T>::zeros((rows * mode_extent, right_rank));
        for row in 0..rows {
            for mode_index in 0..mode_extent {
                let target_row = row * mode_extent + mode_index;
                for left in 0..left_rank {
                    let lhs = accumulated[[row, left]];
                    for right in 0..right_rank {
                        next[[target_row, right]] += lhs * core[[left, mode_index, right]];
                    }
                }
            }
        }
        accumulated = next;
        rows *= mode_extent;
    }

    if accumulated.ncols() != 1 || rows != shape_product(&shape) {
        return Err(TensorError::DimensionMismatch);
    }

    let final_tensor = accumulated
        .into_shape_with_order(IxDyn(&shape))
        .map_err(|_| TensorError::DimensionMismatch)?;
    output.assign(&final_tensor);
    Ok(())
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2, Array3, ArrayD, IxDyn};
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
    fn einsum_into_variants_match_allocating_paths() {
        let left =
            ArrayD::from_shape_vec(IxDyn(&[2, 2, 3]), (0..12).map(f64::from).collect()).unwrap();
        let right = ArrayD::from_shape_vec(
            IxDyn(&[2, 3, 4]),
            (0..24).map(|value| f64::from(value) * 0.5_f64).collect(),
        )
        .unwrap();

        let allocating = einsum("bij,bjk->bik", &left, &right).unwrap();
        let viewed = einsum_view("bij,bjk->bik", &left.view(), &right.view()).unwrap();
        let mut into = ArrayD::<f64>::zeros(IxDyn(&[2, 2, 4]));
        einsum_into("bij,bjk->bik", &left, &right, &mut into).unwrap();
        let mut view_into = ArrayD::<f64>::zeros(IxDyn(&[2, 2, 4]));
        einsum_view_into("bij,bjk->bik", &left.view(), &right.view(), &mut view_into).unwrap();

        for (((lhs, rhs), into_value), view_into_value) in
            allocating.iter().zip(viewed.iter()).zip(into.iter()).zip(view_into.iter())
        {
            assert!((lhs - rhs).abs() < 1e-12_f64);
            assert!((lhs - into_value).abs() < 1e-12_f64);
            assert!((lhs - view_into_value).abs() < 1e-12_f64);
        }

        let left_complex = left.mapv(|value| Complex64::new(value, value + 1.0_f64));
        let right_complex = right.mapv(|value| Complex64::new(value - 0.5_f64, -value));
        let allocating_complex =
            einsum_complex("bij,bjk->bik", &left_complex, &right_complex).unwrap();
        let mut into_complex = ArrayD::<Complex64>::zeros(IxDyn(&[2, 2, 4]));
        einsum_complex_into("bij,bjk->bik", &left_complex, &right_complex, &mut into_complex)
            .unwrap();
        let mut view_into_complex = ArrayD::<Complex64>::zeros(IxDyn(&[2, 2, 4]));
        einsum_complex_view_into(
            "bij,bjk->bik",
            &left_complex.view(),
            &right_complex.view(),
            &mut view_into_complex,
        )
        .unwrap();
        for ((lhs, rhs), view_rhs) in
            allocating_complex.iter().zip(into_complex.iter()).zip(view_into_complex.iter())
        {
            assert!((*lhs - *rhs).norm() < 1e-12_f64);
            assert!((*lhs - *view_rhs).norm() < 1e-12_f64);
        }

        let mut bad_output = ArrayD::<f64>::zeros(IxDyn(&[2, 2, 5]));
        assert!(matches!(
            einsum_view_into("bij,bjk->bik", &left.view(), &right.view(), &mut bad_output),
            Err(TensorError::DimensionMismatch)
        ));
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
    fn hosvd3_reconstruct_into_matches_allocating_path_and_rejects_bad_shape() {
        let cube = Array3::from_shape_vec((3, 3, 2), vec![
            1.0_f32, 0.5_f32, 2.0_f32, -1.0_f32, 0.0_f32, 1.0_f32, 0.0_f32, 2.0_f32, 1.0_f32,
            1.5_f32, 3.0_f32, 0.0_f32, -1.0_f32, 1.0_f32, 2.5_f32, -0.5_f32, 0.5_f32, 2.0_f32,
        ])
        .unwrap();
        let decomposition = hosvd3(&cube, (3, 3, 2)).unwrap();
        let allocating = hosvd3_reconstruct(&decomposition).unwrap();
        let mut into = Array3::<f32>::zeros(cube.dim());
        hosvd3_reconstruct_into(&decomposition, &mut into).unwrap();
        for (lhs, rhs) in allocating.iter().zip(into.iter()) {
            assert!((lhs - rhs).abs() < 1e-5_f32);
        }

        let mut bad_output = Array3::<f32>::zeros((3, 2, 3));
        assert!(matches!(
            hosvd3_reconstruct_into(&decomposition, &mut bad_output),
            Err(TensorError::DimensionMismatch)
        ));
    }

    #[test]
    fn hosvd_nd_reconstructs_synthetic_rank_constrained_tensor_f64() {
        let reference = HosvdNdResult {
            core:    ArrayD::from_shape_vec(IxDyn(&[2, 2, 2, 2]), vec![
                1.0_f64, 0.4_f64, -0.2_f64, 0.7_f64, 0.5_f64, -0.3_f64, 0.6_f64, 0.2_f64, -0.1_f64,
                0.8_f64, 0.9_f64, -0.4_f64, 0.3_f64, 0.1_f64, 0.2_f64, 0.5_f64,
            ])
            .unwrap(),
            factors: vec![
                Array2::from_shape_vec((3, 2), vec![
                    1.0_f64, 0.2_f64, 0.4_f64, 1.1_f64, 0.7_f64, -0.1_f64,
                ])
                .unwrap(),
                Array2::from_shape_vec((2, 2), vec![0.8_f64, 0.3_f64, 0.2_f64, 1.0_f64]).unwrap(),
                Array2::from_shape_vec((4, 2), vec![
                    1.0_f64, 0.0_f64, 0.6_f64, 0.7_f64, 0.2_f64, 1.1_f64, 0.5_f64, -0.3_f64,
                ])
                .unwrap(),
                Array2::from_shape_vec((2, 2), vec![1.0_f64, -0.2_f64, 0.4_f64, 0.9_f64]).unwrap(),
            ],
        };
        let tensor = hosvd_nd_reconstruct(&reference).unwrap();

        let estimated = hosvd_nd(&tensor, &[2, 2, 2, 2]).unwrap();
        let reconstructed = hosvd_nd_reconstruct(&estimated).unwrap();

        let mut diff_sq = 0.0_f64;
        let mut base_sq = 0.0_f64;
        for (lhs, rhs) in reconstructed.iter().zip(tensor.iter()) {
            let delta = lhs - rhs;
            diff_sq += delta * delta;
            base_sq += rhs * rhs;
        }
        let relative_error = diff_sq.sqrt() / base_sq.sqrt();
        assert!(relative_error < 1.0e-6_f64);
    }

    #[test]
    fn hosvd_nd_view_and_into_variants_match_f32() {
        let reference = HosvdNdResult {
            core:    ArrayD::from_shape_vec(IxDyn(&[2, 2, 2]), vec![
                1.0_f32, 0.3_f32, -0.2_f32, 0.5_f32, 0.4_f32, -0.1_f32, 0.8_f32, 0.6_f32,
            ])
            .unwrap(),
            factors: vec![
                Array2::from_shape_vec((3, 2), vec![
                    1.0_f32, 0.2_f32, 0.4_f32, 1.1_f32, 0.7_f32, -0.1_f32,
                ])
                .unwrap(),
                Array2::from_shape_vec((2, 2), vec![0.8_f32, 0.3_f32, 0.2_f32, 1.0_f32]).unwrap(),
                Array2::from_shape_vec((2, 2), vec![1.0_f32, -0.2_f32, 0.4_f32, 0.9_f32]).unwrap(),
            ],
        };
        let tensor = hosvd_nd_reconstruct(&reference).unwrap();

        let owned = hosvd_nd(&tensor, &[2, 2, 2]).unwrap();
        let viewed = hosvd_nd_view(&tensor.view(), &[2, 2, 2]).unwrap();

        let owned_reconstructed = hosvd_nd_reconstruct(&owned).unwrap();
        let viewed_reconstructed = hosvd_nd_reconstruct(&viewed).unwrap();
        let mut viewed_into = ArrayD::<f32>::zeros(tensor.raw_dim());
        hosvd_nd_reconstruct_into(&viewed, &mut viewed_into).unwrap();

        let mut max_delta = 0.0_f32;
        for ((owned_value, viewed_value), into_value) in
            owned_reconstructed.iter().zip(viewed_reconstructed.iter()).zip(viewed_into.iter())
        {
            max_delta = max_delta.max((owned_value - viewed_value).abs());
            max_delta = max_delta.max((owned_value - into_value).abs());
        }
        assert!(max_delta < 1.0e-4_f32);
    }

    #[test]
    fn hosvd_nd_rejects_invalid_ranks_and_output_shapes() {
        let tensor = ArrayD::from_shape_vec(
            IxDyn(&[2, 3, 2]),
            (1..=12).map(|value| f64::from(value) * 0.25_f64).collect::<Vec<_>>(),
        )
        .unwrap();
        assert!(matches!(hosvd_nd(&tensor, &[2, 2]), Err(TensorError::DimensionMismatch)));
        assert!(matches!(hosvd_nd(&tensor, &[2, 0, 2]), Err(TensorError::DimensionMismatch)));
        assert!(matches!(hosvd_nd(&tensor, &[2, 4, 2]), Err(TensorError::DimensionMismatch)));

        let invalid_result = HosvdNdResult {
            core:    ArrayD::<f64>::zeros(IxDyn(&[2, 2])),
            factors: vec![Array2::<f64>::zeros((3, 2)), Array2::<f64>::zeros((4, 1))],
        };
        assert!(matches!(
            hosvd_nd_reconstruct(&invalid_result),
            Err(TensorError::DimensionMismatch)
        ));

        let valid = hosvd_nd(&tensor, &[2, 2, 2]).unwrap();
        let mut bad_output = ArrayD::<f64>::zeros(IxDyn(&[2, 2, 2]));
        assert!(matches!(
            hosvd_nd_reconstruct_into(&valid, &mut bad_output),
            Err(TensorError::DimensionMismatch)
        ));
    }

    #[test]
    fn hooi_nd_reconstructs_synthetic_rank_constrained_tensor_f64() {
        let reference = HosvdNdResult {
            core:    ArrayD::from_shape_vec(IxDyn(&[2, 2, 2, 2]), vec![
                1.0_f64, 0.4_f64, -0.2_f64, 0.7_f64, 0.5_f64, -0.3_f64, 0.6_f64, 0.2_f64, -0.1_f64,
                0.8_f64, 0.9_f64, -0.4_f64, 0.3_f64, 0.1_f64, 0.2_f64, 0.5_f64,
            ])
            .unwrap(),
            factors: vec![
                Array2::from_shape_vec((3, 2), vec![
                    1.0_f64, 0.2_f64, 0.4_f64, 1.1_f64, 0.7_f64, -0.1_f64,
                ])
                .unwrap(),
                Array2::from_shape_vec((2, 2), vec![0.8_f64, 0.3_f64, 0.2_f64, 1.0_f64]).unwrap(),
                Array2::from_shape_vec((4, 2), vec![
                    1.0_f64, 0.0_f64, 0.6_f64, 0.7_f64, 0.2_f64, 1.1_f64, 0.5_f64, -0.3_f64,
                ])
                .unwrap(),
                Array2::from_shape_vec((2, 2), vec![1.0_f64, -0.2_f64, 0.4_f64, 0.9_f64]).unwrap(),
            ],
        };
        let tensor = hosvd_nd_reconstruct(&reference).unwrap();
        let ranks = [2, 2, 2, 2];
        let config = HooiConfig::<f64> { max_iterations: 80, tolerance: 1.0e-10_f64 };

        let baseline = hosvd_nd(&tensor, &ranks).unwrap();
        let estimate = hooi_nd(&tensor, &ranks, &config).unwrap();
        let baseline_reconstructed = hosvd_nd_reconstruct(&baseline).unwrap();
        let reconstructed = hosvd_nd_reconstruct(&estimate).unwrap();

        let relative_error = |candidate: &ArrayD<f64>| {
            let mut diff_sq = 0.0_f64;
            let mut base_sq = 0.0_f64;
            for (lhs, rhs) in candidate.iter().zip(tensor.iter()) {
                let delta = lhs - rhs;
                diff_sq += delta * delta;
                base_sq += rhs * rhs;
            }
            diff_sq.sqrt() / base_sq.sqrt()
        };

        let hooi_error = relative_error(&reconstructed);
        let hosvd_error = relative_error(&baseline_reconstructed);
        assert!(hooi_error < 1.0e-6_f64);
        assert!(hooi_error <= hosvd_error + 1.0e-8_f64);
    }

    #[test]
    fn hooi_nd_view_variants_match_f32() {
        let reference = HosvdNdResult {
            core:    ArrayD::from_shape_vec(IxDyn(&[2, 2, 2]), vec![
                1.0_f32, 0.3_f32, -0.2_f32, 0.5_f32, 0.4_f32, -0.1_f32, 0.8_f32, 0.6_f32,
            ])
            .unwrap(),
            factors: vec![
                Array2::from_shape_vec((3, 2), vec![
                    1.0_f32, 0.2_f32, 0.4_f32, 1.1_f32, 0.7_f32, -0.1_f32,
                ])
                .unwrap(),
                Array2::from_shape_vec((2, 2), vec![0.8_f32, 0.3_f32, 0.2_f32, 1.0_f32]).unwrap(),
                Array2::from_shape_vec((2, 2), vec![1.0_f32, -0.2_f32, 0.4_f32, 0.9_f32]).unwrap(),
            ],
        };
        let tensor = hosvd_nd_reconstruct(&reference).unwrap();
        let ranks = [2, 2, 2];
        let config = HooiConfig::<f32> { max_iterations: 60, tolerance: 1.0e-5_f32 };

        let owned = hooi_nd(&tensor, &ranks, &config).unwrap();
        let viewed = hooi_nd_view(&tensor.view(), &ranks, &config).unwrap();

        let owned_reconstructed = hosvd_nd_reconstruct(&owned).unwrap();
        let viewed_reconstructed = hosvd_nd_reconstruct(&viewed).unwrap();

        let mut max_delta = 0.0_f32;
        for (owned_value, viewed_value) in
            owned_reconstructed.iter().zip(viewed_reconstructed.iter())
        {
            max_delta = max_delta.max((owned_value - viewed_value).abs());
        }
        assert!(max_delta < 1.0e-4_f32);
    }

    #[test]
    fn hooi_nd_rejects_invalid_configs_and_shapes() {
        let tensor = ArrayD::from_shape_vec(
            IxDyn(&[2, 3, 2]),
            (1..=12).map(|value| f64::from(value) * 0.25_f64).collect::<Vec<_>>(),
        )
        .unwrap();
        let ranks = [2, 2, 2];

        let zero_iterations =
            HooiConfig::<f64> { max_iterations: 0, ..HooiConfig::<f64>::default() };
        assert!(matches!(
            hooi_nd(&tensor, &ranks, &zero_iterations),
            Err(TensorError::DimensionMismatch)
        ));

        let zero_tolerance =
            HooiConfig::<f64> { tolerance: 0.0_f64, ..HooiConfig::<f64>::default() };
        assert!(matches!(
            hooi_nd(&tensor, &ranks, &zero_tolerance),
            Err(TensorError::DimensionMismatch)
        ));

        assert!(matches!(
            hooi_nd(&tensor, &[2, 2], &HooiConfig::<f64>::default()),
            Err(TensorError::DimensionMismatch)
        ));
    }

    #[test]
    fn tucker_project_and_expand_match_hosvd_outputs_f64() {
        let reference = HosvdNdResult {
            core:    ArrayD::from_shape_vec(IxDyn(&[2, 2, 2]), vec![
                1.0_f64, 0.2_f64, 0.4_f64, -0.1_f64, 0.7_f64, 0.3_f64, -0.2_f64, 0.5_f64,
            ])
            .unwrap(),
            factors: vec![
                Array2::from_shape_vec((3, 2), vec![
                    1.0_f64, 0.0_f64, 0.0_f64, 1.0_f64, 1.0_f64, 1.0_f64,
                ])
                .unwrap(),
                Array2::from_shape_vec((2, 2), vec![1.0_f64, 0.0_f64, 0.0_f64, 1.0_f64]).unwrap(),
                Array2::from_shape_vec((4, 2), vec![
                    1.0_f64, 0.0_f64, 0.0_f64, 1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64, -1.0_f64,
                ])
                .unwrap(),
            ],
        };

        let tensor = hosvd_nd_reconstruct(&reference).unwrap();
        let hosvd = hosvd_nd(&tensor, &[2, 2, 2]).unwrap();
        let projected = tucker_project(&tensor, &hosvd.factors).unwrap();
        let expanded = tucker_expand(&projected, &hosvd.factors).unwrap();
        let reconstructed = hosvd_nd_reconstruct(&hosvd).unwrap();

        for (lhs, rhs) in projected.iter().zip(hosvd.core.iter()) {
            assert!((lhs - rhs).abs() < 1.0e-8_f64);
        }
        for (lhs, rhs) in expanded.iter().zip(reconstructed.iter()) {
            assert!((lhs - rhs).abs() < 1.0e-8_f64);
        }
    }

    #[test]
    fn tucker_project_expand_view_and_into_variants_match_f32() {
        let tensor = ArrayD::from_shape_vec(IxDyn(&[2, 3, 2]), vec![
            1.0_f32, 0.5_f32, -0.2_f32, 0.8_f32, 0.7_f32, -0.1_f32, 0.3_f32, 1.2_f32, -0.4_f32,
            0.6_f32, 0.9_f32, 0.2_f32,
        ])
        .unwrap();
        let factors = vec![
            Array2::from_shape_vec((2, 2), vec![1.0_f32, 0.0_f32, 0.0_f32, 1.0_f32]).unwrap(),
            Array2::from_shape_vec((3, 2), vec![
                1.0_f32, 0.0_f32, 0.0_f32, 1.0_f32, 1.0_f32, 1.0_f32,
            ])
            .unwrap(),
            Array2::from_shape_vec((2, 2), vec![1.0_f32, 0.0_f32, 0.0_f32, 1.0_f32]).unwrap(),
        ];

        let projected_owned = tucker_project(&tensor, &factors).unwrap();
        let projected_view = tucker_project_view(&tensor.view(), &factors).unwrap();
        let expanded_owned = tucker_expand(&projected_owned, &factors).unwrap();
        let expanded_view = tucker_expand_view(&projected_view.view(), &factors).unwrap();
        let mut expanded_into = ArrayD::<f32>::zeros(IxDyn(&[2, 3, 2]));
        tucker_expand_view_into(&projected_view.view(), &factors, &mut expanded_into).unwrap();

        for ((owned, viewed), into_value) in
            expanded_owned.iter().zip(expanded_view.iter()).zip(expanded_into.iter())
        {
            assert!((owned - viewed).abs() < 1.0e-5_f32);
            assert!((owned - into_value).abs() < 1.0e-5_f32);
        }
    }

    #[test]
    fn tucker_project_into_matches_allocating_path_and_rejects_bad_shape() {
        let tensor = ArrayD::from_shape_vec(IxDyn(&[2, 3, 2]), vec![
            1.0_f32, 0.5_f32, -0.2_f32, 0.8_f32, 0.7_f32, -0.1_f32, 0.3_f32, 1.2_f32, -0.4_f32,
            0.6_f32, 0.9_f32, 0.2_f32,
        ])
        .unwrap();
        let factors = vec![
            Array2::from_shape_vec((2, 2), vec![1.0_f32, 0.0_f32, 0.0_f32, 1.0_f32]).unwrap(),
            Array2::from_shape_vec((3, 2), vec![
                1.0_f32, 0.0_f32, 0.0_f32, 1.0_f32, 1.0_f32, 1.0_f32,
            ])
            .unwrap(),
            Array2::from_shape_vec((2, 2), vec![1.0_f32, 0.0_f32, 0.0_f32, 1.0_f32]).unwrap(),
        ];

        let allocating = tucker_project(&tensor, &factors).unwrap();
        let viewed = tucker_project_view(&tensor.view(), &factors).unwrap();
        let mut into = ArrayD::<f32>::zeros(IxDyn(&[2, 2, 2]));
        tucker_project_into(&tensor, &factors, &mut into).unwrap();
        let mut view_into = ArrayD::<f32>::zeros(IxDyn(&[2, 2, 2]));
        tucker_project_view_into(&tensor.view(), &factors, &mut view_into).unwrap();

        for (((lhs, rhs), into_value), view_into_value) in
            allocating.iter().zip(viewed.iter()).zip(into.iter()).zip(view_into.iter())
        {
            assert!((lhs - rhs).abs() < 1.0e-5_f32);
            assert!((lhs - into_value).abs() < 1.0e-5_f32);
            assert!((lhs - view_into_value).abs() < 1.0e-5_f32);
        }

        let mut bad_output = ArrayD::<f32>::zeros(IxDyn(&[2, 2, 3]));
        assert!(matches!(
            tucker_project_view_into(&tensor.view(), &factors, &mut bad_output),
            Err(TensorError::DimensionMismatch)
        ));
    }

    #[test]
    fn tucker_utilities_reject_invalid_shapes() {
        let tensor = ArrayD::from_shape_vec(IxDyn(&[2, 2, 2]), vec![1.0_f64; 8]).unwrap();
        let factors = vec![
            Array2::from_shape_vec((2, 1), vec![1.0_f64, 0.0_f64]).unwrap(),
            Array2::from_shape_vec((2, 1), vec![1.0_f64, 0.0_f64]).unwrap(),
            Array2::from_shape_vec((2, 1), vec![1.0_f64, 0.0_f64]).unwrap(),
        ];
        let core = tucker_project(&tensor, &factors).unwrap();

        assert!(matches!(
            tucker_project(&tensor, &factors[..2]),
            Err(TensorError::DimensionMismatch)
        ));

        let bad_projection_factors = vec![
            Array2::from_shape_vec((3, 1), vec![1.0_f64, 0.0_f64, 0.0_f64]).unwrap(),
            factors[1].clone(),
            factors[2].clone(),
        ];
        assert!(matches!(
            tucker_project(&tensor, &bad_projection_factors),
            Err(TensorError::DimensionMismatch)
        ));

        let bad_expand_factors = vec![
            Array2::from_shape_vec((2, 2), vec![1.0_f64, 0.0_f64, 0.0_f64, 1.0_f64]).unwrap(),
            Array2::from_shape_vec((2, 2), vec![1.0_f64, 0.0_f64, 0.0_f64, 1.0_f64]).unwrap(),
            Array2::from_shape_vec((2, 2), vec![1.0_f64, 0.0_f64, 0.0_f64, 1.0_f64]).unwrap(),
        ];
        assert!(matches!(
            tucker_expand(&core, &bad_expand_factors),
            Err(TensorError::DimensionMismatch)
        ));

        let mut bad_output = ArrayD::<f64>::zeros(IxDyn(&[2, 2, 3]));
        assert!(matches!(
            tucker_expand_into(&core, &factors, &mut bad_output),
            Err(TensorError::DimensionMismatch)
        ));
    }

    #[test]
    fn tt_svd_reconstructs_synthetic_tensor_f64() {
        let reference = TensorTrainResult {
            cores: vec![
                Array3::from_shape_vec((1, 3, 2), vec![
                    1.0_f64, 0.2_f64, 0.6_f64, -0.1_f64, 0.4_f64, 0.8_f64,
                ])
                .unwrap(),
                Array3::from_shape_vec((2, 2, 2), vec![
                    0.9_f64, -0.1_f64, 0.3_f64, 0.7_f64, -0.2_f64, 0.5_f64, 1.1_f64, 0.4_f64,
                ])
                .unwrap(),
                Array3::from_shape_vec((2, 4, 1), vec![
                    1.0_f64, 0.3_f64, -0.2_f64, 0.5_f64, 0.6_f64, 0.8_f64, 0.1_f64, 1.2_f64,
                ])
                .unwrap(),
            ],
            shape: vec![3, 2, 4],
        };

        let tensor = tt_svd_reconstruct(&reference).unwrap();
        let config = TtSvdConfig::<f64> { max_rank: Some(2), tolerance: 1.0e-10_f64 };
        let estimated = tt_svd(&tensor, &config).unwrap();
        let reconstructed = tt_svd_reconstruct(&estimated).unwrap();

        let mut diff_sq = 0.0_f64;
        let mut base_sq = 0.0_f64;
        for (lhs, rhs) in reconstructed.iter().zip(tensor.iter()) {
            let delta = lhs - rhs;
            diff_sq += delta * delta;
            base_sq += rhs * rhs;
        }
        let relative_error = diff_sq.sqrt() / base_sq.sqrt();
        assert!(relative_error < 1.0e-6_f64);
    }

    #[test]
    fn tt_svd_view_and_into_variants_match_f32() {
        let reference = TensorTrainResult {
            cores: vec![
                Array3::from_shape_vec((1, 2, 2), vec![1.0_f32, 0.4_f32, -0.2_f32, 0.8_f32])
                    .unwrap(),
                Array3::from_shape_vec((2, 3, 1), vec![
                    0.9_f32, 0.2_f32, -0.1_f32, 0.3_f32, 0.5_f32, 1.1_f32,
                ])
                .unwrap(),
            ],
            shape: vec![2, 3],
        };

        let tensor = tt_svd_reconstruct(&reference).unwrap();
        let config = TtSvdConfig::<f32> { max_rank: Some(2), tolerance: 1.0e-5_f32 };
        let owned = tt_svd(&tensor, &config).unwrap();
        let viewed = tt_svd_view(&tensor.view(), &config).unwrap();

        let owned_reconstructed = tt_svd_reconstruct(&owned).unwrap();
        let viewed_reconstructed = tt_svd_reconstruct(&viewed).unwrap();
        let mut viewed_into = ArrayD::<f32>::zeros(tensor.raw_dim());
        tt_svd_reconstruct_into(&viewed, &mut viewed_into).unwrap();

        let mut max_delta = 0.0_f32;
        for ((owned_value, viewed_value), into_value) in
            owned_reconstructed.iter().zip(viewed_reconstructed.iter()).zip(viewed_into.iter())
        {
            max_delta = max_delta.max((owned_value - viewed_value).abs());
            max_delta = max_delta.max((owned_value - into_value).abs());
        }
        assert!(max_delta < 1.0e-4_f32);
    }

    #[test]
    fn tt_svd_rejects_invalid_inputs_and_shapes() {
        let tensor = ArrayD::from_shape_vec(
            IxDyn(&[2, 3, 2]),
            (1..=12).map(|value| f64::from(value) * 0.5_f64).collect::<Vec<_>>(),
        )
        .unwrap();
        let bad_config = TtSvdConfig::<f64> { max_rank: Some(0), tolerance: 1.0e-8_f64 };
        assert!(matches!(tt_svd(&tensor, &bad_config), Err(TensorError::DimensionMismatch)));

        let scalar = ArrayD::from_shape_vec(IxDyn(&[]), vec![1.0_f64]).unwrap();
        assert!(matches!(
            tt_svd(&scalar, &TtSvdConfig::<f64>::default()),
            Err(TensorError::DimensionMismatch)
        ));

        let invalid_result = TensorTrainResult {
            cores: vec![Array3::<f64>::zeros((2, 2, 2)), Array3::<f64>::zeros((3, 3, 1))],
            shape: vec![2, 3],
        };
        assert!(matches!(tt_svd_reconstruct(&invalid_result), Err(TensorError::DimensionMismatch)));

        let valid = tt_svd(&tensor, &TtSvdConfig::<f64>::default()).unwrap();
        let mut bad_output = ArrayD::<f64>::zeros(IxDyn(&[2, 2, 2]));
        assert!(matches!(
            tt_svd_reconstruct_into(&valid, &mut bad_output),
            Err(TensorError::DimensionMismatch)
        ));
    }

    #[test]
    fn tt_orthogonalize_left_preserves_reconstruction_and_columns() {
        let reference = TensorTrainResult {
            cores: vec![
                Array3::from_shape_vec((1, 3, 2), vec![
                    1.0_f64, 0.2_f64, 0.6_f64, -0.1_f64, 0.4_f64, 0.8_f64,
                ])
                .unwrap(),
                Array3::from_shape_vec((2, 2, 3), vec![
                    0.9_f64, -0.1_f64, 0.3_f64, 0.7_f64, -0.2_f64, 0.5_f64, 1.1_f64, 0.4_f64,
                    0.2_f64, 0.3_f64, 0.8_f64, -0.6_f64,
                ])
                .unwrap(),
                Array3::from_shape_vec((3, 4, 1), vec![
                    1.0_f64, 0.3_f64, -0.2_f64, 0.5_f64, 0.6_f64, 0.8_f64, 0.1_f64, 1.2_f64,
                    -0.3_f64, 0.2_f64, 0.7_f64, 0.4_f64,
                ])
                .unwrap(),
            ],
            shape: vec![3, 2, 4],
        };

        let baseline = tt_svd_reconstruct(&reference).unwrap();
        let orth = tt_orthogonalize_left(&reference).unwrap();
        let reconstructed = tt_svd_reconstruct(&orth).unwrap();

        let mut diff_sq = 0.0_f64;
        let mut base_sq = 0.0_f64;
        for (lhs, rhs) in reconstructed.iter().zip(baseline.iter()) {
            let delta = lhs - rhs;
            diff_sq += delta * delta;
            base_sq += rhs * rhs;
        }
        let relative_error = diff_sq.sqrt() / base_sq.sqrt();
        assert!(relative_error < 1.0e-10_f64);

        for core in orth.cores.iter().take(orth.cores.len() - 1) {
            let (left, mode_extent, right) = core.dim();
            let matrix = core.clone().into_shape_with_order((left * mode_extent, right)).unwrap();
            for i in 0..right {
                for j in 0..right {
                    let mut dot = 0.0_f64;
                    for row in 0..(left * mode_extent) {
                        dot += matrix[[row, i]] * matrix[[row, j]];
                    }
                    if i == j {
                        assert!((dot - 1.0_f64).abs() < 1.0e-8_f64);
                    } else {
                        assert!(dot.abs() < 1.0e-8_f64);
                    }
                }
            }
        }
    }

    #[test]
    fn tt_orthogonalize_right_preserves_reconstruction_and_rows() {
        let reference = TensorTrainResult {
            cores: vec![
                Array3::from_shape_vec((1, 3, 2), vec![
                    1.0_f64, 0.2_f64, 0.6_f64, -0.1_f64, 0.4_f64, 0.8_f64,
                ])
                .unwrap(),
                Array3::from_shape_vec((2, 2, 3), vec![
                    0.9_f64, -0.1_f64, 0.3_f64, 0.7_f64, -0.2_f64, 0.5_f64, 1.1_f64, 0.4_f64,
                    0.2_f64, 0.3_f64, 0.8_f64, -0.6_f64,
                ])
                .unwrap(),
                Array3::from_shape_vec((3, 4, 1), vec![
                    1.0_f64, 0.3_f64, -0.2_f64, 0.5_f64, 0.6_f64, 0.8_f64, 0.1_f64, 1.2_f64,
                    -0.3_f64, 0.2_f64, 0.7_f64, 0.4_f64,
                ])
                .unwrap(),
            ],
            shape: vec![3, 2, 4],
        };

        let baseline = tt_svd_reconstruct(&reference).unwrap();
        let orth = tt_orthogonalize_right(&reference).unwrap();
        let reconstructed = tt_svd_reconstruct(&orth).unwrap();

        let mut diff_sq = 0.0_f64;
        let mut base_sq = 0.0_f64;
        for (lhs, rhs) in reconstructed.iter().zip(baseline.iter()) {
            let delta = lhs - rhs;
            diff_sq += delta * delta;
            base_sq += rhs * rhs;
        }
        let relative_error = diff_sq.sqrt() / base_sq.sqrt();
        assert!(relative_error < 1.0e-10_f64);

        for core in orth.cores.iter().skip(1) {
            let (left, mode_extent, right) = core.dim();
            let matrix = core.clone().into_shape_with_order((left, mode_extent * right)).unwrap();
            for i in 0..left {
                for j in 0..left {
                    let mut dot = 0.0_f64;
                    for col in 0..(mode_extent * right) {
                        dot += matrix[[i, col]] * matrix[[j, col]];
                    }
                    if i == j {
                        assert!((dot - 1.0_f64).abs() < 1.0e-8_f64);
                    } else {
                        assert!(dot.abs() < 1.0e-8_f64);
                    }
                }
            }
        }
    }

    #[test]
    fn tt_round_reduces_ranks_with_small_reconstruction_error() {
        let reference = TensorTrainResult {
            cores: vec![
                Array3::from_shape_vec((1, 3, 3), vec![
                    1.0_f64,
                    0.2_f64,
                    1.0e-6_f64,
                    0.6_f64,
                    -0.1_f64,
                    2.0e-6_f64,
                    0.4_f64,
                    0.8_f64,
                    -1.0e-6_f64,
                ])
                .unwrap(),
                Array3::from_shape_vec((3, 2, 3), vec![
                    1.0_f64, 0.0_f64, 0.0_f64, 0.5_f64, 0.0_f64, 0.0_f64, 0.0_f64, 1.0_f64,
                    0.0_f64, 0.0_f64, 0.4_f64, 0.0_f64, 0.0_f64, 0.0_f64, 1.0e-4_f64, 0.0_f64,
                    0.0_f64, 2.0e-4_f64,
                ])
                .unwrap(),
                Array3::from_shape_vec((3, 4, 1), vec![
                    1.0_f64,
                    0.3_f64,
                    -0.2_f64,
                    0.5_f64,
                    0.6_f64,
                    0.8_f64,
                    0.1_f64,
                    1.2_f64,
                    1.0e-6_f64,
                    -2.0e-6_f64,
                    3.0e-6_f64,
                    1.0e-6_f64,
                ])
                .unwrap(),
            ],
            shape: vec![3, 2, 4],
        };

        let baseline = tt_svd_reconstruct(&reference).unwrap();
        let config = TtRoundConfig::<f64> { max_rank: Some(2), tolerance: 1.0e-8_f64 };
        let rounded = tt_round(&reference, &config).unwrap();
        let reconstructed = tt_svd_reconstruct(&rounded).unwrap();

        for core in rounded.cores.iter().take(rounded.cores.len() - 1) {
            assert!(core.dim().2 <= 2);
        }
        for core in rounded.cores.iter().skip(1) {
            assert!(core.dim().0 <= 2);
        }

        let mut diff_sq = 0.0_f64;
        let mut base_sq = 0.0_f64;
        for (lhs, rhs) in reconstructed.iter().zip(baseline.iter()) {
            let delta = lhs - rhs;
            diff_sq += delta * delta;
            base_sq += rhs * rhs;
        }
        let relative_error = diff_sq.sqrt() / base_sq.sqrt();
        assert!(relative_error < 1.0e-4_f64);
    }

    #[test]
    fn tt_round_rejects_invalid_config() {
        let tensor = ArrayD::from_shape_vec(
            IxDyn(&[2, 3, 2]),
            (1..=12).map(|value| f64::from(value) * 0.5_f64).collect::<Vec<_>>(),
        )
        .unwrap();
        let valid = tt_svd(&tensor, &TtSvdConfig::<f64>::default()).unwrap();

        let bad_rank = TtRoundConfig::<f64> { max_rank: Some(0), tolerance: 1.0e-8_f64 };
        assert!(matches!(tt_round(&valid, &bad_rank), Err(TensorError::DimensionMismatch)));

        let bad_tolerance = TtRoundConfig::<f64> { max_rank: Some(2), tolerance: -1.0e-8_f64 };
        assert!(matches!(tt_round(&valid, &bad_tolerance), Err(TensorError::DimensionMismatch)));
    }

    #[test]
    fn tt_inner_and_norm_match_dense_f64() {
        let left_tensor = ArrayD::from_shape_vec(IxDyn(&[2, 3, 2]), vec![
            1.0_f64, -0.5_f64, 0.7_f64, 0.2_f64, -0.1_f64, 0.9_f64, 0.3_f64, 0.8_f64, -0.6_f64,
            1.2_f64, 0.4_f64, -0.3_f64,
        ])
        .unwrap();
        let right_tensor = ArrayD::from_shape_vec(IxDyn(&[2, 3, 2]), vec![
            0.6_f64, 0.2_f64, -0.4_f64, 1.1_f64, 0.5_f64, -0.7_f64, 0.9_f64, 0.3_f64, 0.8_f64,
            -0.2_f64, 0.4_f64, 1.0_f64,
        ])
        .unwrap();
        let config = TtSvdConfig::<f64> { max_rank: Some(4), tolerance: 1.0e-12_f64 };
        let left_tt = tt_svd(&left_tensor, &config).unwrap();
        let right_tt = tt_svd(&right_tensor, &config).unwrap();

        let expected_inner = left_tensor
            .iter()
            .zip(right_tensor.iter())
            .fold(0.0_f64, |acc, (lhs, rhs)| acc + lhs * rhs);
        let expected_norm =
            left_tensor.iter().fold(0.0_f64, |acc, value| acc + value * value).sqrt();

        let observed_inner = tt_inner(&left_tt, &right_tt).unwrap();
        let observed_norm = tt_norm(&left_tt).unwrap();
        assert!((observed_inner - expected_inner).abs() < 1.0e-10_f64);
        assert!((observed_norm - expected_norm).abs() < 1.0e-10_f64);
    }

    #[test]
    fn tt_add_and_hadamard_match_dense_f32() {
        let left_tensor = ArrayD::from_shape_vec(IxDyn(&[2, 2, 2]), vec![
            1.0_f32, 0.5_f32, -0.2_f32, 0.8_f32, 0.7_f32, -0.1_f32, 0.3_f32, 1.2_f32,
        ])
        .unwrap();
        let right_tensor = ArrayD::from_shape_vec(IxDyn(&[2, 2, 2]), vec![
            0.4_f32, -0.3_f32, 1.0_f32, 0.6_f32, -0.5_f32, 0.9_f32, 0.2_f32, -0.7_f32,
        ])
        .unwrap();
        let config = TtSvdConfig::<f32> { max_rank: Some(4), tolerance: 1.0e-6_f32 };
        let left_tt = tt_svd(&left_tensor, &config).unwrap();
        let right_tt = tt_svd(&right_tensor, &config).unwrap();

        let added = tt_add(&left_tt, &right_tt).unwrap();
        let hadamard = tt_hadamard(&left_tt, &right_tt).unwrap();
        let added_dense = tt_svd_reconstruct(&added).unwrap();
        let hadamard_dense = tt_svd_reconstruct(&hadamard).unwrap();

        for (((added_value, hadamard_value), lhs), rhs) in added_dense
            .iter()
            .zip(hadamard_dense.iter())
            .zip(left_tensor.iter())
            .zip(right_tensor.iter())
        {
            assert!((*added_value - (*lhs + *rhs)).abs() < 5.0e-4_f32);
            assert!((*hadamard_value - (*lhs * *rhs)).abs() < 5.0e-4_f32);
        }
    }

    #[test]
    fn tt_hadamard_round_limits_rank_and_preserves_product() {
        let left_tensor = ArrayD::from_shape_vec(IxDyn(&[2, 3, 2]), vec![
            1.0_f64, 0.1_f64, 0.8_f64, -0.2_f64, 0.4_f64, 0.9_f64, 0.3_f64, 0.7_f64, -0.5_f64,
            1.1_f64, 0.6_f64, -0.4_f64,
        ])
        .unwrap();
        let right_tensor = ArrayD::from_shape_vec(IxDyn(&[2, 3, 2]), vec![
            0.7_f64, -0.3_f64, 0.5_f64, 0.2_f64, 1.0_f64, -0.6_f64, 0.4_f64, 0.8_f64, 0.9_f64,
            -0.1_f64, 0.3_f64, 0.6_f64,
        ])
        .unwrap();
        let svd_config = TtSvdConfig::<f64> { max_rank: Some(4), tolerance: 1.0e-12_f64 };
        let left_tt = tt_svd(&left_tensor, &svd_config).unwrap();
        let right_tt = tt_svd(&right_tensor, &svd_config).unwrap();
        let round_config = TtRoundConfig::<f64> { max_rank: Some(2), tolerance: 1.0e-8_f64 };

        let rounded = tt_hadamard_round(&left_tt, &right_tt, &round_config).unwrap();
        let reconstructed = tt_svd_reconstruct(&rounded).unwrap();
        let baseline = left_tensor * right_tensor;

        for core in rounded.cores.iter().take(rounded.cores.len() - 1) {
            assert!(core.dim().2 <= 2);
        }
        for core in rounded.cores.iter().skip(1) {
            assert!(core.dim().0 <= 2);
        }

        let mut diff_sq = 0.0_f64;
        let mut base_sq = 0.0_f64;
        for (observed, expected) in reconstructed.iter().zip(baseline.iter()) {
            let delta = observed - expected;
            diff_sq += delta * delta;
            base_sq += expected * expected;
        }
        let relative_error = diff_sq.sqrt() / base_sq.sqrt();
        assert!(relative_error < 1.0e-2_f64);
    }

    #[test]
    fn tt_binary_ops_reject_shape_mismatch() {
        let left = tt_svd(
            &ArrayD::from_shape_vec(IxDyn(&[2, 2, 2]), vec![1.0_f64; 8]).unwrap(),
            &TtSvdConfig::<f64>::default(),
        )
        .unwrap();
        let right = tt_svd(
            &ArrayD::from_shape_vec(IxDyn(&[2, 2, 3]), vec![1.0_f64; 12]).unwrap(),
            &TtSvdConfig::<f64>::default(),
        )
        .unwrap();

        assert!(matches!(tt_inner(&left, &right), Err(TensorError::DimensionMismatch)));
        assert!(matches!(tt_add(&left, &right), Err(TensorError::DimensionMismatch)));
        assert!(matches!(tt_hadamard(&left, &right), Err(TensorError::DimensionMismatch)));
        assert!(matches!(
            tt_hadamard_round(&left, &right, &TtRoundConfig::<f64>::default()),
            Err(TensorError::DimensionMismatch)
        ));
    }

    #[test]
    fn cp_als3_reconstructs_synthetic_rank2_tensor_f64() {
        let reference = CpAls3Result {
            weights:  Array1::from_vec(vec![1.5_f64, 0.8_f64]),
            factor_0: Array2::from_shape_vec((4, 2), vec![
                1.0_f64, 0.2_f64, 0.7_f64, 1.1_f64, 0.3_f64, 0.9_f64, 1.2_f64, 0.4_f64,
            ])
            .unwrap(),
            factor_1: Array2::from_shape_vec((3, 2), vec![
                0.5_f64, 1.0_f64, 1.3_f64, 0.4_f64, 0.8_f64, 1.2_f64,
            ])
            .unwrap(),
            factor_2: Array2::from_shape_vec((2, 2), vec![1.0_f64, 0.6_f64, 0.9_f64, 1.4_f64])
                .unwrap(),
        };
        let tensor = cp_als3_reconstruct(&reference).unwrap();

        let config = CpAlsConfig { max_iterations: 300, tolerance: 1.0e-10_f64 };
        let estimated = cp_als3(&tensor, 2, &config).unwrap();
        let reconstructed = cp_als3_reconstruct(&estimated).unwrap();

        let mut diff_sq = 0.0_f64;
        let mut base_sq = 0.0_f64;
        for (lhs, rhs) in reconstructed.iter().zip(tensor.iter()) {
            let delta = lhs - rhs;
            diff_sq += delta * delta;
            base_sq += rhs * rhs;
        }
        let relative_error = diff_sq.sqrt() / base_sq.sqrt();
        assert!(relative_error < 1.0e-6_f64);
    }

    #[test]
    fn cp_als3_view_and_into_variants_match_f32() {
        let reference = CpAls3Result {
            weights:  Array1::from_vec(vec![1.0_f32, 0.6_f32]),
            factor_0: Array2::from_shape_vec((3, 2), vec![
                1.0_f32, 0.2_f32, 0.7_f32, 1.1_f32, 0.3_f32, 0.9_f32,
            ])
            .unwrap(),
            factor_1: Array2::from_shape_vec((3, 2), vec![
                0.5_f32, 1.0_f32, 1.3_f32, 0.4_f32, 0.8_f32, 1.2_f32,
            ])
            .unwrap(),
            factor_2: Array2::from_shape_vec((2, 2), vec![1.0_f32, 0.6_f32, 0.9_f32, 1.4_f32])
                .unwrap(),
        };
        let tensor = cp_als3_reconstruct(&reference).unwrap();
        let config = CpAlsConfig { max_iterations: 250, tolerance: 1.0e-5_f32 };

        let owned = cp_als3(&tensor, 2, &config).unwrap();
        let viewed = cp_als3_view(&tensor.view(), 2, &config).unwrap();
        let owned_reconstructed = cp_als3_reconstruct(&owned).unwrap();
        let viewed_reconstructed = cp_als3_reconstruct(&viewed).unwrap();
        let mut viewed_into = Array3::<f32>::zeros(tensor.dim());
        cp_als3_reconstruct_into(&viewed, &mut viewed_into).unwrap();

        let mut max_delta = 0.0_f32;
        for ((owned_value, viewed_value), into_value) in
            owned_reconstructed.iter().zip(viewed_reconstructed.iter()).zip(viewed_into.iter())
        {
            max_delta = max_delta.max((owned_value - viewed_value).abs());
            max_delta = max_delta.max((owned_value - into_value).abs());
        }
        assert!(max_delta < 1.0e-4_f32);
    }

    #[test]
    fn cp_als3_rejects_invalid_rank_and_reconstruction_shapes() {
        let cube = Array3::<f64>::zeros((2, 2, 2));
        let config = CpAlsConfig::<f64>::default();
        assert!(matches!(cp_als3(&cube, 0, &config), Err(TensorError::DimensionMismatch)));

        let invalid_config =
            CpAlsConfig::<f64> { max_iterations: 0, ..CpAlsConfig::<f64>::default() };
        assert!(matches!(cp_als3(&cube, 1, &invalid_config), Err(TensorError::DimensionMismatch)));

        let invalid_factors = CpAls3Result {
            weights:  Array1::from_vec(vec![1.0_f64]),
            factor_0: Array2::<f64>::zeros((2, 2)),
            factor_1: Array2::<f64>::zeros((2, 1)),
            factor_2: Array2::<f64>::zeros((2, 1)),
        };
        assert!(matches!(
            cp_als3_reconstruct(&invalid_factors),
            Err(TensorError::DimensionMismatch)
        ));
    }

    #[test]
    fn cp_als_nd_reconstructs_synthetic_rank2_tensor_f64() {
        let reference = CpAlsNdResult {
            weights: Array1::from_vec(vec![1.5_f64, 0.8_f64]),
            factors: vec![
                Array2::from_shape_vec((3, 2), vec![
                    1.0_f64, 0.0_f64, 0.0_f64, 1.0_f64, 1.0_f64, 1.0_f64,
                ])
                .unwrap(),
                Array2::from_shape_vec((2, 2), vec![1.0_f64, 0.0_f64, 0.0_f64, 1.0_f64]).unwrap(),
                Array2::from_shape_vec((4, 2), vec![
                    1.0_f64, 0.0_f64, 0.0_f64, 1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64, -1.0_f64,
                ])
                .unwrap(),
                Array2::from_shape_vec((2, 2), vec![1.0_f64, 0.0_f64, 0.0_f64, 1.0_f64]).unwrap(),
            ],
            shape:   vec![3, 2, 4, 2],
        };
        let tensor = cp_als_nd_reconstruct(&reference).unwrap();

        let config = CpAlsConfig { max_iterations: 400, tolerance: 1.0e-6_f64 };
        let estimated = cp_als_nd(&tensor, 2, &config).unwrap();
        let reconstructed = cp_als_nd_reconstruct(&estimated).unwrap();

        let mut diff_sq = 0.0_f64;
        let mut base_sq = 0.0_f64;
        for (lhs, rhs) in reconstructed.iter().zip(tensor.iter()) {
            let delta = lhs - rhs;
            diff_sq += delta * delta;
            base_sq += rhs * rhs;
        }
        let relative_error = diff_sq.sqrt() / base_sq.sqrt();
        assert!(relative_error < 1.0e-5_f64);
    }

    #[test]
    fn cp_als_nd_view_and_into_variants_match_f32() {
        let reference = CpAlsNdResult {
            weights: Array1::from_vec(vec![1.0_f32, 0.6_f32]),
            factors: vec![
                Array2::from_shape_vec((2, 2), vec![1.0_f32, 0.0_f32, 0.0_f32, 1.0_f32]).unwrap(),
                Array2::from_shape_vec((3, 2), vec![
                    1.0_f32, 0.0_f32, 0.0_f32, 1.0_f32, 1.0_f32, 1.0_f32,
                ])
                .unwrap(),
                Array2::from_shape_vec((2, 2), vec![1.0_f32, 0.0_f32, 0.0_f32, 1.0_f32]).unwrap(),
            ],
            shape:   vec![2, 3, 2],
        };
        let tensor = cp_als_nd_reconstruct(&reference).unwrap();
        let config = CpAlsConfig { max_iterations: 300, tolerance: 1.0e-6_f32 };

        let owned = cp_als_nd(&tensor, 2, &config).unwrap();
        let viewed = cp_als_nd_view(&tensor.view(), 2, &config).unwrap();
        let owned_reconstructed = cp_als_nd_reconstruct(&owned).unwrap();
        let viewed_reconstructed = cp_als_nd_reconstruct(&viewed).unwrap();
        let mut viewed_into = ArrayD::<f32>::zeros(tensor.raw_dim());
        cp_als_nd_reconstruct_into(&viewed, &mut viewed_into).unwrap();

        let mut max_delta = 0.0_f32;
        for ((owned_value, viewed_value), into_value) in
            owned_reconstructed.iter().zip(viewed_reconstructed.iter()).zip(viewed_into.iter())
        {
            max_delta = max_delta.max((owned_value - viewed_value).abs());
            max_delta = max_delta.max((owned_value - into_value).abs());
        }
        assert!(max_delta < 1.0e-4_f32);
    }

    #[test]
    fn cp_als_nd_rejects_invalid_rank_and_reconstruction_shapes() {
        let tensor = ArrayD::from_shape_vec(
            IxDyn(&[2, 3, 2]),
            (1..=12).map(|value| f64::from(value) * 0.5_f64).collect::<Vec<_>>(),
        )
        .unwrap();
        let config = CpAlsConfig::<f64>::default();
        assert!(matches!(cp_als_nd(&tensor, 0, &config), Err(TensorError::DimensionMismatch)));

        let invalid_config =
            CpAlsConfig::<f64> { max_iterations: 0, ..CpAlsConfig::<f64>::default() };
        assert!(matches!(
            cp_als_nd(&tensor, 1, &invalid_config),
            Err(TensorError::DimensionMismatch)
        ));

        let scalar = ArrayD::from_shape_vec(IxDyn(&[]), vec![1.0_f64]).unwrap();
        assert!(matches!(
            cp_als_nd(&scalar, 1, &CpAlsConfig::<f64>::default()),
            Err(TensorError::DimensionMismatch)
        ));

        let invalid_result = CpAlsNdResult {
            weights: Array1::from_vec(vec![1.0_f64]),
            factors: vec![Array2::<f64>::zeros((2, 2)), Array2::<f64>::zeros((3, 1))],
            shape:   vec![2, 3],
        };
        assert!(matches!(
            cp_als_nd_reconstruct(&invalid_result),
            Err(TensorError::DimensionMismatch)
        ));

        let valid = cp_als_nd(&tensor, 1, &CpAlsConfig::<f64>::default()).unwrap();
        let mut bad_output = ArrayD::<f64>::zeros(IxDyn(&[2, 2, 2]));
        assert!(matches!(
            cp_als_nd_reconstruct_into(&valid, &mut bad_output),
            Err(TensorError::DimensionMismatch)
        ));
    }

    #[test]
    fn cp_als3_report_and_diagnostics_match_f64() {
        let reference = CpAls3Result {
            weights:  Array1::from_vec(vec![1.5_f64, 0.8_f64]),
            factor_0: Array2::from_shape_vec((4, 2), vec![
                1.0_f64, 0.2_f64, 0.7_f64, 1.1_f64, 0.3_f64, 0.9_f64, 1.2_f64, 0.4_f64,
            ])
            .unwrap(),
            factor_1: Array2::from_shape_vec((3, 2), vec![
                0.5_f64, 1.0_f64, 1.3_f64, 0.4_f64, 0.8_f64, 1.2_f64,
            ])
            .unwrap(),
            factor_2: Array2::from_shape_vec((2, 2), vec![1.0_f64, 0.6_f64, 0.9_f64, 1.4_f64])
                .unwrap(),
        };
        let tensor = cp_als3_reconstruct(&reference).unwrap();
        let config = CpAlsConfig { max_iterations: 300, tolerance: 1.0e-10_f64 };

        let (estimated, report) = cp_als3_with_report(&tensor, 2, &config).unwrap();
        let diagnostics = cp_als3_diagnostics(&tensor, &estimated).unwrap();

        assert!(report.convergence.iterations_run >= 1);
        assert!(report.convergence.iterations_run <= config.max_iterations);
        assert!((report.metrics.fit - diagnostics.fit).abs() < 1.0e-12_f64);
        assert!((report.metrics.relative_error - diagnostics.relative_error).abs() < 1.0e-12_f64);
        assert!((report.metrics.residual_norm - diagnostics.residual_norm).abs() < 1.0e-12_f64);
        assert!(report.metrics.relative_error < 1.0e-6_f64);
        assert!(report.metrics.fit > 0.999_999_f64);
    }

    #[test]
    fn cp_als_nd_report_and_diagnostics_match_f32() {
        let reference = CpAlsNdResult {
            weights: Array1::from_vec(vec![1.0_f32, 0.6_f32]),
            factors: vec![
                Array2::from_shape_vec((2, 2), vec![1.0_f32, 0.0_f32, 0.0_f32, 1.0_f32]).unwrap(),
                Array2::from_shape_vec((3, 2), vec![
                    1.0_f32, 0.0_f32, 0.0_f32, 1.0_f32, 1.0_f32, 1.0_f32,
                ])
                .unwrap(),
                Array2::from_shape_vec((2, 2), vec![1.0_f32, 0.0_f32, 0.0_f32, 1.0_f32]).unwrap(),
            ],
            shape:   vec![2, 3, 2],
        };
        let tensor = cp_als_nd_reconstruct(&reference).unwrap();
        let config = CpAlsConfig { max_iterations: 300, tolerance: 1.0e-6_f32 };

        let (estimated, report) = cp_als_nd_with_report(&tensor, 2, &config).unwrap();
        let diagnostics = cp_als_nd_diagnostics(&tensor, &estimated).unwrap();

        assert!(report.convergence.iterations_run >= 1);
        assert!(report.convergence.iterations_run <= config.max_iterations);
        assert!((report.metrics.fit - diagnostics.fit).abs() < 1.0e-5_f32);
        assert!((report.metrics.relative_error - diagnostics.relative_error).abs() < 1.0e-5_f32);
        assert!((report.metrics.residual_norm - diagnostics.residual_norm).abs() < 1.0e-5_f32);
        assert!(report.metrics.relative_error < 1.0e-3_f32);
        assert!(report.metrics.fit > 0.99_f32);
    }

    #[test]
    fn cp_als_diagnostics_reject_mismatched_reference_shapes() {
        let cube = Array3::<f64>::from_shape_vec((2, 2, 2), vec![
            1.0_f64, 0.0_f64, 0.0_f64, 1.0_f64, 1.0_f64, 1.0_f64, 0.0_f64, 1.0_f64,
        ])
        .unwrap();
        let result3 = cp_als3(&cube, 1, &CpAlsConfig::<f64>::default()).unwrap();
        let bad_cube = Array3::<f64>::zeros((2, 2, 3));
        assert!(matches!(
            cp_als3_diagnostics(&bad_cube, &result3),
            Err(TensorError::DimensionMismatch)
        ));

        let tensor = ArrayD::from_shape_vec(IxDyn(&[2, 2, 2]), vec![
            1.0_f64, 0.0_f64, 0.0_f64, 1.0_f64, 1.0_f64, 1.0_f64, 0.0_f64, 1.0_f64,
        ])
        .unwrap();
        let result_nd = cp_als_nd(&tensor, 1, &CpAlsConfig::<f64>::default()).unwrap();
        let bad_tensor = ArrayD::<f64>::zeros(IxDyn(&[2, 2, 3]));
        assert!(matches!(
            cp_als_nd_diagnostics(&bad_tensor, &result_nd),
            Err(TensorError::DimensionMismatch)
        ));
    }

    #[test]
    fn tensor_factor_view_reconstruct_helpers_match_owned_paths() {
        let cube =
            Array3::<f64>::from_shape_vec((3, 4, 2), (0..24).map(f64::from).collect()).unwrap();
        let hosvd3_result = hosvd3(&cube, (3, 4, 2)).unwrap();

        let mut hosvd3_core_storage = Array3::<f64>::zeros((4, 5, 3));
        hosvd3_core_storage.slice_mut(s![1.., 1.., 1..]).assign(&hosvd3_result.core);
        let hosvd3_core_view = hosvd3_core_storage.slice(s![1.., 1.., 1..]);

        let mut hosvd3_u0_storage =
            Array2::<f64>::zeros((hosvd3_result.u0.nrows() + 1, hosvd3_result.u0.ncols() + 1));
        hosvd3_u0_storage.slice_mut(s![1.., 1..]).assign(&hosvd3_result.u0);
        let hosvd3_u0_view = hosvd3_u0_storage.slice(s![1.., 1..]);

        let mut hosvd3_u1_storage =
            Array2::<f64>::zeros((hosvd3_result.u1.nrows() + 1, hosvd3_result.u1.ncols() + 1));
        hosvd3_u1_storage.slice_mut(s![1.., 1..]).assign(&hosvd3_result.u1);
        let hosvd3_u1_view = hosvd3_u1_storage.slice(s![1.., 1..]);

        let mut hosvd3_u2_storage =
            Array2::<f64>::zeros((hosvd3_result.u2.nrows() + 1, hosvd3_result.u2.ncols() + 1));
        hosvd3_u2_storage.slice_mut(s![1.., 1..]).assign(&hosvd3_result.u2);
        let hosvd3_u2_view = hosvd3_u2_storage.slice(s![1.., 1..]);

        let hosvd3_owned = hosvd3_reconstruct(&hosvd3_result).unwrap();
        let hosvd3_viewed = hosvd3_reconstruct_from_factors_view(
            &hosvd3_core_view,
            &hosvd3_u0_view,
            &hosvd3_u1_view,
            &hosvd3_u2_view,
        )
        .unwrap();
        assert_eq!(hosvd3_owned, hosvd3_viewed);

        let tensor = cube.into_dyn();
        let hosvd_nd_result = hosvd_nd(&tensor, &[3, 4, 2]).unwrap();
        let mut hosvd_nd_core_storage = ArrayD::<f64>::zeros(IxDyn(&[4, 5, 3]));
        hosvd_nd_core_storage.slice_mut(s![1.., 1.., 1..]).assign(&hosvd_nd_result.core);
        let hosvd_nd_core_view = hosvd_nd_core_storage.slice(s![1.., 1.., 1..]).into_dyn();

        let mut factor_storages = Vec::new();
        for factor in &hosvd_nd_result.factors {
            let mut storage = Array2::<f64>::zeros((factor.nrows() + 1, factor.ncols() + 1));
            storage.slice_mut(s![1.., 1..]).assign(factor);
            factor_storages.push(storage);
        }
        let factor_views =
            factor_storages.iter().map(|factor| factor.slice(s![1.., 1..])).collect::<Vec<_>>();

        let hosvd_nd_owned = hosvd_nd_reconstruct(&hosvd_nd_result).unwrap();
        let hosvd_nd_viewed =
            hosvd_nd_reconstruct_from_factors_view(&hosvd_nd_core_view, &factor_views).unwrap();
        assert_eq!(hosvd_nd_owned, hosvd_nd_viewed);

        let projected_owned = tucker_project(&tensor, &hosvd_nd_result.factors).unwrap();
        let projected_viewed =
            tucker_project_from_factors_view(&tensor.view(), &factor_views).unwrap();
        assert_eq!(projected_owned, projected_viewed);

        let expanded_viewed =
            tucker_expand_from_factors_view(&hosvd_nd_core_view, &factor_views).unwrap();
        assert_eq!(hosvd_nd_owned, expanded_viewed);
    }

    #[test]
    fn cp_factor_view_helpers_match_owned_paths_and_diagnostics() {
        let cp3_result = CpAls3Result {
            weights:  Array1::from_vec(vec![1.5_f64, 0.8_f64]),
            factor_0: Array2::from_shape_vec((4, 2), vec![
                1.0_f64, 0.2_f64, 0.7_f64, 1.1_f64, 0.3_f64, 0.9_f64, 1.2_f64, 0.4_f64,
            ])
            .unwrap(),
            factor_1: Array2::from_shape_vec((3, 2), vec![
                0.5_f64, 1.0_f64, 1.3_f64, 0.4_f64, 0.8_f64, 1.2_f64,
            ])
            .unwrap(),
            factor_2: Array2::from_shape_vec((2, 2), vec![1.0_f64, 0.6_f64, 0.9_f64, 1.4_f64])
                .unwrap(),
        };
        let cp3_tensor = cp_als3_reconstruct(&cp3_result).unwrap();

        let mut cp3_weight_storage = Array1::<f64>::zeros(4);
        cp3_weight_storage.slice_mut(s![1..3]).assign(&cp3_result.weights);
        let cp3_weight_view = cp3_weight_storage.slice(s![1..3]);

        let mut cp3_factor_0_storage = Array2::<f64>::zeros((5, 3));
        cp3_factor_0_storage.slice_mut(s![1.., 1..]).assign(&cp3_result.factor_0);
        let cp3_factor_0_view = cp3_factor_0_storage.slice(s![1.., 1..]);

        let mut cp3_factor_1_storage = Array2::<f64>::zeros((4, 3));
        cp3_factor_1_storage.slice_mut(s![1.., 1..]).assign(&cp3_result.factor_1);
        let cp3_factor_1_view = cp3_factor_1_storage.slice(s![1.., 1..]);

        let mut cp3_factor_2_storage = Array2::<f64>::zeros((3, 3));
        cp3_factor_2_storage.slice_mut(s![1.., 1..]).assign(&cp3_result.factor_2);
        let cp3_factor_2_view = cp3_factor_2_storage.slice(s![1.., 1..]);

        let cp3_reconstructed = cp_als3_reconstruct_from_factors_view(
            &cp3_weight_view,
            &cp3_factor_0_view,
            &cp3_factor_1_view,
            &cp3_factor_2_view,
        )
        .unwrap();
        assert_eq!(cp3_tensor, cp3_reconstructed);

        let cp3_metrics_owned = cp_als3_diagnostics(&cp3_tensor, &cp3_result).unwrap();
        let cp3_metrics_viewed = cp_als3_diagnostics_from_factors_view(
            &cp3_tensor.view(),
            &cp3_weight_view,
            &cp3_factor_0_view,
            &cp3_factor_1_view,
            &cp3_factor_2_view,
        )
        .unwrap();
        assert!(
            (cp3_metrics_owned.relative_error - cp3_metrics_viewed.relative_error).abs() < 1e-12
        );

        let cp_nd_result = CpAlsNdResult {
            weights: Array1::from_vec(vec![1.5_f64, 0.8_f64]),
            factors: vec![
                Array2::from_shape_vec((3, 2), vec![
                    1.0_f64, 0.0_f64, 0.0_f64, 1.0_f64, 1.0_f64, 1.0_f64,
                ])
                .unwrap(),
                Array2::from_shape_vec((2, 2), vec![1.0_f64, 0.0_f64, 0.0_f64, 1.0_f64]).unwrap(),
                Array2::from_shape_vec((4, 2), vec![
                    1.0_f64, 0.0_f64, 0.0_f64, 1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64, -1.0_f64,
                ])
                .unwrap(),
                Array2::from_shape_vec((2, 2), vec![1.0_f64, 0.0_f64, 0.0_f64, 1.0_f64]).unwrap(),
            ],
            shape:   vec![3, 2, 4, 2],
        };
        let cp_nd_tensor = cp_als_nd_reconstruct(&cp_nd_result).unwrap();
        let mut cp_nd_weight_storage = Array1::<f64>::zeros(4);
        cp_nd_weight_storage.slice_mut(s![1..3]).assign(&cp_nd_result.weights);
        let cp_nd_weight_view = cp_nd_weight_storage.slice(s![1..3]);

        let mut cp_nd_factor_storages = Vec::new();
        for factor in &cp_nd_result.factors {
            let mut storage = Array2::<f64>::zeros((factor.nrows() + 1, factor.ncols() + 1));
            storage.slice_mut(s![1.., 1..]).assign(factor);
            cp_nd_factor_storages.push(storage);
        }
        let cp_nd_factor_views = cp_nd_factor_storages
            .iter()
            .map(|factor| factor.slice(s![1.., 1..]))
            .collect::<Vec<_>>();

        let cp_nd_reconstructed =
            cp_als_nd_reconstruct_from_factors_view(&cp_nd_weight_view, &cp_nd_factor_views)
                .unwrap();
        assert_eq!(cp_nd_tensor, cp_nd_reconstructed);

        let cp_nd_metrics_owned = cp_als_nd_diagnostics(&cp_nd_tensor, &cp_nd_result).unwrap();
        let cp_nd_metrics_viewed = cp_als_nd_diagnostics_from_factors_view(
            &cp_nd_tensor.view(),
            &cp_nd_weight_view,
            &cp_nd_factor_views,
        )
        .unwrap();
        assert!(
            (cp_nd_metrics_owned.relative_error - cp_nd_metrics_viewed.relative_error).abs()
                < 1e-12
        );
    }

    #[test]
    fn tt_reconstruct_from_core_views_matches_owned_path() {
        let tensor = ArrayD::from_shape_vec(
            IxDyn(&[2, 3, 4]),
            (0..24).map(|value| 0.25_f64 * f64::from(value) - 1.0_f64).collect(),
        )
        .unwrap();
        let tt = tt_svd(&tensor, &TtSvdConfig::<f64>::default()).unwrap();
        let owned = tt_svd_reconstruct(&tt).unwrap();

        let mut core_storages = Vec::new();
        for core in &tt.cores {
            let mut storage =
                Array3::<f64>::zeros((core.dim().0 + 1, core.dim().1 + 1, core.dim().2 + 1));
            storage.slice_mut(s![1.., 1.., 1..]).assign(core);
            core_storages.push(storage);
        }
        let core_views =
            core_storages.iter().map(|core| core.slice(s![1.., 1.., 1..])).collect::<Vec<_>>();

        let viewed = tt_svd_reconstruct_from_cores_view(&core_views).unwrap();
        assert_eq!(owned, viewed);
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "this regression keeps the TT borrowed-view parity cases together"
    )]
    fn hosvd3_and_tt_borrowed_view_helpers_match_owned_paths() {
        let cube =
            Array3::<f64>::from_shape_vec((3, 4, 2), (0..24).map(f64::from).collect()).unwrap();
        let hosvd3_owned = hosvd3(&cube, (3, 4, 2)).unwrap();

        let mut cube_storage = Array3::<f64>::zeros((4, 5, 3));
        cube_storage.slice_mut(s![1.., 1.., 1..]).assign(&cube);
        let cube_view = cube_storage.slice(s![1.., 1.., 1..]);

        let hosvd3_viewed = hosvd3_view(&cube_view, (3, 4, 2)).unwrap();
        let assert_close3 = |left: &Array3<f64>, right: &Array3<f64>, tolerance: f64| {
            assert_eq!(left.shape(), right.shape());
            for (left_value, right_value) in left.iter().zip(right.iter()) {
                assert!((left_value - right_value).abs() <= tolerance);
            }
        };
        assert_close3(&hosvd3_reconstruct(&hosvd3_owned).unwrap(), &cube, 1.0e-10);
        assert_close3(&hosvd3_reconstruct(&hosvd3_viewed).unwrap(), &cube, 1.0e-10);

        let left_tensor = ArrayD::from_shape_vec(IxDyn(&[2, 3, 2]), vec![
            1.0_f64, -0.5, 0.7, 0.2, -0.1, 0.9, 0.3, 0.8, -0.6, 1.2, 0.4, -0.3,
        ])
        .unwrap();
        let right_tensor = ArrayD::from_shape_vec(IxDyn(&[2, 3, 2]), vec![
            0.6_f64, 0.2, -0.4, 1.1, 0.5, -0.7, 0.9, 0.3, 0.8, -0.2, 0.4, 1.0,
        ])
        .unwrap();
        let round_config = TtRoundConfig::<f64> { max_rank: Some(2), tolerance: 1.0e-8 };
        let left_tt = tt_svd(&left_tensor, &TtSvdConfig::<f64>::default()).unwrap();
        let right_tt = tt_svd(&right_tensor, &TtSvdConfig::<f64>::default()).unwrap();

        let mut left_core_storages = Vec::new();
        for core in &left_tt.cores {
            let mut storage =
                Array3::<f64>::zeros((core.dim().0 + 1, core.dim().1 + 1, core.dim().2 + 1));
            storage.slice_mut(s![1.., 1.., 1..]).assign(core);
            left_core_storages.push(storage);
        }
        let left_core_views =
            left_core_storages.iter().map(|core| core.slice(s![1.., 1.., 1..])).collect::<Vec<_>>();

        let mut right_core_storages = Vec::new();
        for core in &right_tt.cores {
            let mut storage =
                Array3::<f64>::zeros((core.dim().0 + 1, core.dim().1 + 1, core.dim().2 + 1));
            storage.slice_mut(s![1.., 1.., 1..]).assign(core);
            right_core_storages.push(storage);
        }
        let right_core_views = right_core_storages
            .iter()
            .map(|core| core.slice(s![1.., 1.., 1..]))
            .collect::<Vec<_>>();

        let assert_close = |left: &ArrayD<f64>, right: &ArrayD<f64>, tolerance: f64| {
            assert_eq!(left.shape(), right.shape());
            for (left_value, right_value) in left.iter().zip(right.iter()) {
                assert!((left_value - right_value).abs() <= tolerance);
            }
        };

        let left_orth_owned = tt_orthogonalize_left(&left_tt).unwrap();
        let left_orth_viewed = tt_orthogonalize_left_from_cores_view(&left_core_views).unwrap();
        assert_close(
            &tt_svd_reconstruct(&left_orth_owned).unwrap(),
            &tt_svd_reconstruct(&left_orth_viewed).unwrap(),
            1.0e-10,
        );

        let right_orth_owned = tt_orthogonalize_right(&left_tt).unwrap();
        let right_orth_viewed = tt_orthogonalize_right_from_cores_view(&left_core_views).unwrap();
        assert_close(
            &tt_svd_reconstruct(&right_orth_owned).unwrap(),
            &tt_svd_reconstruct(&right_orth_viewed).unwrap(),
            1.0e-10,
        );

        let round_owned = tt_round(&left_tt, &round_config).unwrap();
        let round_viewed = tt_round_from_cores_view(&left_core_views, &round_config).unwrap();
        assert_close(
            &tt_svd_reconstruct(&round_owned).unwrap(),
            &tt_svd_reconstruct(&round_viewed).unwrap(),
            1.0e-10,
        );

        assert!(
            (tt_inner(&left_tt, &right_tt).unwrap()
                - tt_inner_from_cores_view(&left_core_views, &right_core_views).unwrap())
            .abs()
                <= 1.0e-10
        );
        assert!(
            (tt_norm(&left_tt).unwrap() - tt_norm_from_cores_view(&left_core_views).unwrap()).abs()
                <= 1.0e-10
        );

        let add_owned = tt_add(&left_tt, &right_tt).unwrap();
        let add_viewed = tt_add_from_cores_view(&left_core_views, &right_core_views).unwrap();
        assert_close(
            &tt_svd_reconstruct(&add_owned).unwrap(),
            &tt_svd_reconstruct(&add_viewed).unwrap(),
            1.0e-10,
        );

        let hadamard_owned = tt_hadamard(&left_tt, &right_tt).unwrap();
        let hadamard_viewed =
            tt_hadamard_from_cores_view(&left_core_views, &right_core_views).unwrap();
        assert_close(
            &tt_svd_reconstruct(&hadamard_owned).unwrap(),
            &tt_svd_reconstruct(&hadamard_viewed).unwrap(),
            1.0e-10,
        );

        let hadamard_round_owned = tt_hadamard_round(&left_tt, &right_tt, &round_config).unwrap();
        let hadamard_round_viewed =
            tt_hadamard_round_from_cores_view(&left_core_views, &right_core_views, &round_config)
                .unwrap();
        assert_close(
            &tt_svd_reconstruct(&hadamard_round_owned).unwrap(),
            &tt_svd_reconstruct(&hadamard_round_viewed).unwrap(),
            1.0e-8,
        );
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
