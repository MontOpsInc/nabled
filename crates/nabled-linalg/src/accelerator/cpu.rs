use std::collections::BTreeMap;
use std::ops::{AddAssign, Mul};

use nabled_core::scalar::NabledReal;
use ndarray::linalg::general_mat_mul;
use ndarray::{Array1, Array2, Array3, ArrayD, ArrayView2, IxDyn, s};
use num_traits::Float;
#[cfg(feature = "accelerator-rayon")]
use rayon::prelude::*;

use super::backends::AcceleratorError;
use crate::sparse::CsrMatrix;
use crate::tensor::{batched_matmul_last_two_view_into_impl, contract_view_into_impl};

const SPARSE_TOLERANCE: f64 = 1.0e-12;

fn sparse_tolerance<T: NabledReal>() -> T { T::from_f64(SPARSE_TOLERANCE).unwrap_or(T::epsilon()) }

/// Apply a CPU closure over row chunks.
///
/// This provides a deterministic chunking contract for CPU kernel
/// partitioning without introducing runtime backend switching.
///
/// # Errors
/// Returns an error for invalid chunking policy.
pub fn for_each_row_chunk(
    matrix: &Array2<f64>,
    chunk_rows: usize,
    mut operation: impl FnMut(ArrayView2<'_, f64>),
) -> Result<(), AcceleratorError> {
    if chunk_rows == 0 {
        return Err(AcceleratorError::InvalidChunkSize);
    }

    let mut row = 0_usize;
    while row < matrix.nrows() {
        let end = (row + chunk_rows).min(matrix.nrows());
        operation(matrix.slice(s![row..end, ..]));
        row = end;
    }
    Ok(())
}

/// Compute matrix-vector product with explicit serial CPU kernel.
///
/// # Errors
/// Returns an error if dimensions are incompatible.
pub fn matvec_serial(
    matrix: &Array2<f64>,
    vector: &Array1<f64>,
) -> Result<Array1<f64>, AcceleratorError> {
    if matrix.ncols() != vector.len() {
        return Err(AcceleratorError::DimensionMismatch);
    }
    Ok(matrix.dot(vector))
}

/// Compute matrix-vector product with explicit serial CPU kernel for `f32`.
///
/// # Errors
/// Returns an error if dimensions are incompatible.
pub(crate) fn matvec_serial_f32(
    matrix: &Array2<f32>,
    vector: &Array1<f32>,
) -> Result<Array1<f32>, AcceleratorError> {
    if matrix.ncols() != vector.len() {
        return Err(AcceleratorError::DimensionMismatch);
    }
    Ok(matrix.dot(vector))
}

/// Compute matrix-matrix product with explicit serial CPU kernel.
///
/// # Errors
/// Returns an error if dimensions are incompatible.
pub fn matmat_serial(
    left: &Array2<f64>,
    right: &Array2<f64>,
) -> Result<Array2<f64>, AcceleratorError> {
    if left.ncols() != right.nrows() {
        return Err(AcceleratorError::DimensionMismatch);
    }
    Ok(left.dot(right))
}

/// Compute matrix-matrix product with explicit serial CPU kernel for `f32`.
///
/// # Errors
/// Returns an error if dimensions are incompatible.
pub(crate) fn matmat_serial_f32(
    left: &Array2<f32>,
    right: &Array2<f32>,
) -> Result<Array2<f32>, AcceleratorError> {
    if left.ncols() != right.nrows() {
        return Err(AcceleratorError::DimensionMismatch);
    }
    Ok(left.dot(right))
}

/// Compute batched matrix-matrix products with explicit serial CPU kernel.
///
/// # Errors
/// Returns an error if dimensions are incompatible.
pub fn batched_matmat_serial(
    left_batches: &Array3<f64>,
    right_batches: &Array3<f64>,
) -> Result<Array3<f64>, AcceleratorError> {
    if left_batches.dim().0 != right_batches.dim().0
        || left_batches.dim().2 != right_batches.dim().1
    {
        return Err(AcceleratorError::DimensionMismatch);
    }

    let batch = left_batches.dim().0;
    let rows = left_batches.dim().1;
    let cols = right_batches.dim().2;
    let mut output = Array3::<f64>::zeros((batch, rows, cols));

    for b in 0..batch {
        let left = left_batches.slice(s![b, .., ..]);
        let right = right_batches.slice(s![b, .., ..]);
        let mut out = output.slice_mut(s![b, .., ..]);
        general_mat_mul(1.0_f64, &left, &right, 0.0_f64, &mut out);
    }

    Ok(output)
}

/// Compute batched matrix-matrix products with explicit serial CPU kernel for `f32`.
///
/// # Errors
/// Returns an error if dimensions are incompatible.
pub(crate) fn batched_matmat_serial_f32(
    left_batches: &Array3<f32>,
    right_batches: &Array3<f32>,
) -> Result<Array3<f32>, AcceleratorError> {
    if left_batches.dim().0 != right_batches.dim().0
        || left_batches.dim().2 != right_batches.dim().1
    {
        return Err(AcceleratorError::DimensionMismatch);
    }

    let batch = left_batches.dim().0;
    let rows = left_batches.dim().1;
    let cols = right_batches.dim().2;
    let mut output = Array3::<f32>::zeros((batch, rows, cols));

    for b in 0..batch {
        let left = left_batches.slice(s![b, .., ..]);
        let right = right_batches.slice(s![b, .., ..]);
        let mut out = output.slice_mut(s![b, .., ..]);
        general_mat_mul(1.0_f32, &left, &right, 0.0_f32, &mut out);
    }

    Ok(output)
}

/// Compute sparse matrix-vector product with explicit serial CPU kernel.
///
/// # Errors
/// Returns an error if dimensions are incompatible.
pub fn sparse_matvec_serial<T: NabledReal>(
    matrix: &CsrMatrix<T>,
    vector: &Array1<T>,
) -> Result<Array1<T>, AcceleratorError> {
    if matrix.ncols != vector.len() {
        return Err(AcceleratorError::DimensionMismatch);
    }

    let mut output = Array1::<T>::zeros(matrix.nrows);
    for row in 0..matrix.nrows {
        let start = matrix.indptr[row];
        let end = matrix.indptr[row + 1];
        let mut sum = T::zero();
        for entry in start..end {
            sum += matrix.data[entry] * vector[matrix.indices[entry]];
        }
        output[row] = sum;
    }
    Ok(output)
}

/// Compute dense row-batch by matrix product with explicit serial CPU kernel.
///
/// # Errors
/// Returns an error if dimensions are incompatible.
pub fn batched_row_matvec_serial<T: NabledReal>(
    batch_vectors: &Array2<T>,
    matrix: &Array2<T>,
) -> Result<Array2<T>, AcceleratorError> {
    if batch_vectors.ncols() != matrix.ncols() {
        return Err(AcceleratorError::DimensionMismatch);
    }

    let mut output = Array2::<T>::zeros((batch_vectors.nrows(), matrix.nrows()));
    for batch in 0..batch_vectors.nrows() {
        for row in 0..matrix.nrows() {
            let mut sum = T::zero();
            for col in 0..matrix.ncols() {
                sum += batch_vectors[[batch, col]] * matrix[[row, col]];
            }
            output[[batch, row]] = sum;
        }
    }

    Ok(output)
}

/// Compute sparse-dense matrix multiplication with explicit serial CPU kernel.
///
/// # Errors
/// Returns an error if dimensions are incompatible.
pub fn sparse_matmat_dense_serial<T: NabledReal>(
    matrix: &CsrMatrix<T>,
    dense: &Array2<T>,
) -> Result<Array2<T>, AcceleratorError> {
    if dense.nrows() != matrix.ncols {
        return Err(AcceleratorError::DimensionMismatch);
    }

    let mut output = Array2::<T>::zeros((matrix.nrows, dense.ncols()));
    for row in 0..matrix.nrows {
        for entry in matrix.indptr[row]..matrix.indptr[row + 1] {
            let col = matrix.indices[entry];
            let value = matrix.data[entry];
            for dense_col in 0..dense.ncols() {
                output[[row, dense_col]] += value * dense[[col, dense_col]];
            }
        }
    }

    Ok(output)
}

/// Compute sparse-sparse matrix multiplication with explicit serial CPU kernel.
///
/// # Errors
/// Returns an error if dimensions are incompatible or sparse structure is invalid.
pub fn sparse_matmat_sparse_serial<T: NabledReal>(
    left: &CsrMatrix<T>,
    right: &CsrMatrix<T>,
) -> Result<CsrMatrix<T>, AcceleratorError> {
    if left.ncols != right.nrows {
        return Err(AcceleratorError::DimensionMismatch);
    }

    let mut indptr = Vec::<usize>::with_capacity(left.nrows + 1);
    let mut indices = Vec::<usize>::new();
    let mut data = Vec::<T>::new();
    indptr.push(0);

    for row in 0..left.nrows {
        let mut row_accumulator = BTreeMap::<usize, T>::new();
        for left_entry in left.indptr[row]..left.indptr[row + 1] {
            let inner = left.indices[left_entry];
            let left_value = left.data[left_entry];

            for right_entry in right.indptr[inner]..right.indptr[inner + 1] {
                let col = right.indices[right_entry];
                let right_value = right.data[right_entry];
                let accumulator = row_accumulator.entry(col).or_insert(T::zero());
                *accumulator += left_value * right_value;
            }
        }

        for (col, value) in row_accumulator {
            if value.abs() > sparse_tolerance::<T>() {
                indices.push(col);
                data.push(value);
            }
        }
        indptr.push(indices.len());
    }

    CsrMatrix::new(left.nrows, right.ncols, indptr, indices, data)
        .map_err(|_| AcceleratorError::KernelExecutionFailed)
}

/// Compute dense triangular solve against a vector with explicit serial CPU kernel.
///
/// # Errors
/// Returns an error if dimensions are incompatible or a singular pivot is encountered.
pub fn triangular_solve_vec_serial<T>(
    matrix: &Array2<T>,
    rhs: &Array1<T>,
    lower: bool,
    unit_diagonal: bool,
) -> Result<Array1<T>, AcceleratorError>
where
    T: Float,
{
    if matrix.nrows() != matrix.ncols() || matrix.nrows() != rhs.len() {
        return Err(AcceleratorError::DimensionMismatch);
    }

    let n = matrix.nrows();
    let mut output = Array1::<T>::zeros(n);

    if lower {
        for i in 0..n {
            let mut sum = rhs[i];
            for j in 0..i {
                sum = sum - matrix[[i, j]] * output[j];
            }
            if unit_diagonal {
                output[i] = sum;
            } else {
                let pivot = matrix[[i, i]];
                if pivot == T::zero() {
                    return Err(AcceleratorError::KernelExecutionFailed);
                }
                output[i] = sum / pivot;
            }
        }
    } else {
        for i_rev in 0..n {
            let i = n - 1 - i_rev;
            let mut sum = rhs[i];
            for j in (i + 1)..n {
                sum = sum - matrix[[i, j]] * output[j];
            }
            if unit_diagonal {
                output[i] = sum;
            } else {
                let pivot = matrix[[i, i]];
                if pivot == T::zero() {
                    return Err(AcceleratorError::KernelExecutionFailed);
                }
                output[i] = sum / pivot;
            }
        }
    }

    Ok(output)
}

/// Compute dense triangular solve against a matrix with explicit serial CPU kernel.
///
/// # Errors
/// Returns an error if dimensions are incompatible or a singular pivot is encountered.
pub fn triangular_solve_mat_serial<T>(
    matrix: &Array2<T>,
    rhs: &Array2<T>,
    lower: bool,
    unit_diagonal: bool,
) -> Result<Array2<T>, AcceleratorError>
where
    T: Float,
{
    if matrix.nrows() != matrix.ncols() || matrix.nrows() != rhs.nrows() {
        return Err(AcceleratorError::DimensionMismatch);
    }

    let mut output = Array2::<T>::zeros((rhs.nrows(), rhs.ncols()));
    for col in 0..rhs.ncols() {
        let rhs_col = rhs.column(col).to_owned();
        let solution = triangular_solve_vec_serial(matrix, &rhs_col, lower, unit_diagonal)?;
        output.column_mut(col).assign(&solution);
    }
    Ok(output)
}

/// Compute vector dot product with explicit serial CPU kernel.
///
/// # Errors
/// Returns an error if dimensions are incompatible.
pub fn dot_serial<T: NabledReal>(
    left: &Array1<T>,
    right: &Array1<T>,
) -> Result<T, AcceleratorError> {
    if left.len() != right.len() {
        return Err(AcceleratorError::DimensionMismatch);
    }
    Ok(left.dot(right))
}

/// Compute pairwise L2 distance matrix with explicit serial CPU kernel.
///
/// # Errors
/// Returns an error if dimensions are incompatible.
pub fn pairwise_l2_serial<T: NabledReal>(
    left: &Array2<T>,
    right: &Array2<T>,
) -> Result<Array2<T>, AcceleratorError> {
    if left.ncols() != right.ncols() {
        return Err(AcceleratorError::DimensionMismatch);
    }

    let mut output = Array2::<T>::zeros((left.nrows(), right.nrows()));
    for i in 0..left.nrows() {
        for j in 0..right.nrows() {
            let mut sum = T::zero();
            for k in 0..left.ncols() {
                let delta = left[[i, k]] - right[[j, k]];
                sum += delta * delta;
            }
            output[[i, j]] = sum.sqrt();
        }
    }

    Ok(output)
}

/// Compute pairwise cosine similarity matrix with explicit serial CPU kernel.
///
/// # Errors
/// Returns an error if dimensions are incompatible or if any row has zero norm.
pub fn pairwise_cosine_serial<T: NabledReal>(
    left: &Array2<T>,
    right: &Array2<T>,
) -> Result<Array2<T>, AcceleratorError> {
    if left.ncols() != right.ncols() {
        return Err(AcceleratorError::DimensionMismatch);
    }

    let mut left_norms = Array1::<T>::zeros(left.nrows());
    let mut right_norms = Array1::<T>::zeros(right.nrows());

    for i in 0..left.nrows() {
        let mut sq_sum = T::zero();
        for k in 0..left.ncols() {
            let value = left[[i, k]];
            sq_sum += value * value;
        }
        let norm = sq_sum.sqrt();
        if norm <= T::epsilon() {
            return Err(AcceleratorError::KernelExecutionFailed);
        }
        left_norms[i] = norm;
    }

    for j in 0..right.nrows() {
        let mut sq_sum = T::zero();
        for k in 0..right.ncols() {
            let value = right[[j, k]];
            sq_sum += value * value;
        }
        let norm = sq_sum.sqrt();
        if norm <= T::epsilon() {
            return Err(AcceleratorError::KernelExecutionFailed);
        }
        right_norms[j] = norm;
    }

    let mut output = Array2::<T>::zeros((left.nrows(), right.nrows()));
    for i in 0..left.nrows() {
        for j in 0..right.nrows() {
            let mut dot = T::zero();
            for k in 0..left.ncols() {
                dot += left[[i, k]] * right[[j, k]];
            }
            output[[i, j]] = dot / (left_norms[i] * right_norms[j]);
        }
    }

    Ok(output)
}

fn uncontracted_axes(ndim: usize, contracted: &[usize]) -> Option<Vec<usize>> {
    let mut flags = vec![false; ndim];
    for &axis in contracted {
        if axis >= ndim || flags[axis] {
            return None;
        }
        flags[axis] = true;
    }

    Some((0..ndim).filter(|axis| !flags[*axis]).collect())
}

/// Compute tensor contraction with explicit serial CPU kernel.
///
/// # Errors
/// Returns an error if dimensions are incompatible.
pub fn tensor_contract_axes_serial<T>(
    left: &ArrayD<T>,
    right: &ArrayD<T>,
    left_axes: usize,
    right_axes: usize,
) -> Result<ArrayD<T>, AcceleratorError>
where
    T: Copy + Default + AddAssign + Mul<Output = T>,
{
    let left_axes_slice = [left_axes];
    let right_axes_slice = [right_axes];

    let Some(left_free_axes) = uncontracted_axes(left.ndim(), &left_axes_slice) else {
        return Err(AcceleratorError::DimensionMismatch);
    };
    let Some(right_free_axes) = uncontracted_axes(right.ndim(), &right_axes_slice) else {
        return Err(AcceleratorError::DimensionMismatch);
    };

    let mut output_shape =
        left_free_axes.iter().map(|axis| left.shape()[*axis]).collect::<Vec<_>>();
    output_shape.extend(right_free_axes.iter().map(|axis| right.shape()[*axis]));

    let mut output = ArrayD::<T>::default(IxDyn(&output_shape));
    contract_view_into_impl(
        &left.view(),
        &right.view(),
        &left_axes_slice,
        &right_axes_slice,
        &mut output,
    )
    .map_err(|_| AcceleratorError::DimensionMismatch)?;

    Ok(output)
}

/// Compute N-D batched matrix multiplication over the last two axes with explicit serial CPU
/// kernel.
///
/// # Errors
/// Returns an error if dimensions are incompatible.
pub fn tensor_batched_matmul_last_two_serial<T>(
    left: &ArrayD<T>,
    right: &ArrayD<T>,
) -> Result<ArrayD<T>, AcceleratorError>
where
    T: Copy + Default + AddAssign + Mul<Output = T>,
{
    if left.ndim() < 2 || right.ndim() < 2 {
        return Err(AcceleratorError::DimensionMismatch);
    }

    let batch_ndim = left.ndim() - 2;
    if left.ndim() != right.ndim()
        || left.shape()[..batch_ndim] != right.shape()[..batch_ndim]
        || left.shape()[left.ndim() - 1] != right.shape()[right.ndim() - 2]
    {
        return Err(AcceleratorError::DimensionMismatch);
    }

    let mut output_shape = left.shape()[..batch_ndim].to_vec();
    output_shape.push(left.shape()[left.ndim() - 2]);
    output_shape.push(right.shape()[right.ndim() - 1]);

    let mut output = ArrayD::<T>::default(IxDyn(&output_shape));
    batched_matmul_last_two_view_into_impl(&left.view(), &right.view(), &mut output)
        .map_err(|_| AcceleratorError::DimensionMismatch)?;

    Ok(output)
}

/// Compute tensor reduction along the last axis with explicit serial CPU kernel.
///
/// # Errors
/// Returns an error if dimensions are incompatible.
pub fn tensor_sum_last_axis_serial<T>(input: &ArrayD<T>) -> Result<ArrayD<T>, AcceleratorError>
where
    T: Copy + Default + AddAssign,
{
    if input.ndim() == 0 {
        return Err(AcceleratorError::DimensionMismatch);
    }

    let Some(last) = input.shape().last().copied() else {
        return Err(AcceleratorError::DimensionMismatch);
    };
    if last == 0 {
        return Err(AcceleratorError::DimensionMismatch);
    }

    let mut output_shape = input.shape().to_vec();
    let _ = output_shape.pop();
    let outer = input.len() / last;

    let standard = input.as_standard_layout().to_owned();
    let input_2d = standard
        .view()
        .into_shape_with_order((outer, last))
        .map_err(|_| AcceleratorError::DimensionMismatch)?;

    let mut output_flat = Array1::<T>::default(outer);
    for row in 0..outer {
        let mut sum = T::default();
        for col in 0..last {
            sum += input_2d[[row, col]];
        }
        output_flat[row] = sum;
    }

    output_flat
        .into_shape_with_order(IxDyn(&output_shape))
        .map_err(|_| AcceleratorError::DimensionMismatch)
}

/// Compute matrix-matrix product using feature-gated accelerated CPU kernel.
///
/// When `accelerator-rayon` is enabled, rows are computed in parallel.
/// Otherwise, this returns [`AcceleratorError::FeatureNotEnabled`].
///
/// # Errors
/// Returns an error for incompatible dimensions or if accelerator feature is disabled.
pub fn matmat_accelerated(
    left: &Array2<f64>,
    right: &Array2<f64>,
) -> Result<Array2<f64>, AcceleratorError> {
    if left.ncols() != right.nrows() {
        return Err(AcceleratorError::DimensionMismatch);
    }

    #[cfg(feature = "accelerator-rayon")]
    {
        let cols = right.ncols();
        let rows = left.nrows();
        let inner = left.ncols();
        let row_results = (0..rows)
            .into_par_iter()
            .map(|row| {
                let mut out_row = vec![0.0_f64; cols];
                for k in 0..inner {
                    let lhs = left[[row, k]];
                    for col in 0..cols {
                        out_row[col] += lhs * right[[k, col]];
                    }
                }
                out_row
            })
            .collect::<Vec<_>>();

        let mut output = Array2::<f64>::zeros((rows, cols));
        for (row, row_values) in row_results.into_iter().enumerate() {
            for (col, value) in row_values.into_iter().enumerate() {
                output[[row, col]] = value;
            }
        }
        Ok(output)
    }

    #[cfg(not(feature = "accelerator-rayon"))]
    {
        let _ = left;
        let _ = right;
        Err(AcceleratorError::FeatureNotEnabled)
    }
}
