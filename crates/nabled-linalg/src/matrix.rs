//! Dense matrix pipeline primitives over ndarray arrays.

use std::fmt;

use nabled_core::scalar::NabledReal;
use ndarray::linalg::general_mat_mul;
use ndarray::{
    Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, ArrayViewMut1, ArrayViewMut2,
    ArrayViewMut3, s,
};
use num_complex::Complex64;

use crate::accelerator::backends::AcceleratorError;
use crate::accelerator::dispatch::{
    batched_matmat_cpu, batched_row_matvec_cpu, matmat_cpu, matvec_cpu,
};

/// Error type for dense matrix operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixError {
    /// Input matrix/vector is empty.
    EmptyInput,
    /// Input dimensions are incompatible.
    DimensionMismatch,
}

impl fmt::Display for MatrixError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MatrixError::EmptyInput => write!(f, "input cannot be empty"),
            MatrixError::DimensionMismatch => write!(f, "input dimensions are incompatible"),
        }
    }
}

impl std::error::Error for MatrixError {}

fn map_accelerator_error_to_matrix(_error: AcceleratorError) -> MatrixError {
    MatrixError::DimensionMismatch
}

fn validate_matrix_non_empty<T>(matrix: &ArrayView2<'_, T>) -> Result<(), MatrixError> {
    if matrix.is_empty() {
        return Err(MatrixError::EmptyInput);
    }
    Ok(())
}

fn validate_vector_non_empty<T>(vector: &ArrayView1<'_, T>) -> Result<(), MatrixError> {
    if vector.is_empty() {
        return Err(MatrixError::EmptyInput);
    }
    Ok(())
}

fn validate_tensor_non_empty<T>(tensor: &ArrayView3<'_, T>) -> Result<(), MatrixError> {
    if tensor.is_empty() {
        return Err(MatrixError::EmptyInput);
    }
    Ok(())
}

fn validate_matrix_non_empty_complex(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<(), MatrixError> {
    if matrix.is_empty() {
        return Err(MatrixError::EmptyInput);
    }
    Ok(())
}

fn validate_vector_non_empty_complex(
    vector: &ArrayView1<'_, Complex64>,
) -> Result<(), MatrixError> {
    if vector.is_empty() {
        return Err(MatrixError::EmptyInput);
    }
    Ok(())
}

/// Compute dense matrix-vector product `y = A x`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn matvec<T>(matrix: &Array2<T>, vector: &Array1<T>) -> Result<Array1<T>, MatrixError>
where
    T: NabledReal,
    crate::accelerator::backends::CpuBackend: crate::accelerator::kernels::MatVecKernel<T>,
{
    let matrix_view = matrix.view();
    let vector_view = vector.view();
    validate_matrix_non_empty(&matrix_view)?;
    validate_vector_non_empty(&vector_view)?;
    if vector.len() != matrix.ncols() {
        return Err(MatrixError::DimensionMismatch);
    }
    matvec_cpu(matrix, vector).map_err(map_accelerator_error_to_matrix)
}

/// Compute dense matrix-vector product `y = A x` from views.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn matvec_view<T>(
    matrix: &ArrayView2<'_, T>,
    vector: &ArrayView1<'_, T>,
) -> Result<Array1<T>, MatrixError>
where
    T: NabledReal,
{
    validate_matrix_non_empty(matrix)?;
    validate_vector_non_empty(vector)?;
    if vector.len() != matrix.ncols() {
        return Err(MatrixError::DimensionMismatch);
    }
    Ok(matrix.dot(vector))
}

/// Compute dense matrix-vector product `y = A x` into `output`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn matvec_into<T>(
    matrix: &Array2<T>,
    vector: &Array1<T>,
    output: &mut Array1<T>,
) -> Result<(), MatrixError>
where
    T: NabledReal,
{
    let matrix_view = matrix.view();
    let vector_view = vector.view();
    matvec_view_into(&matrix_view, &vector_view, output.view_mut())
}

/// Compute dense matrix-vector product `y = A x` from views into `output`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn matvec_view_into<T>(
    matrix: &ArrayView2<'_, T>,
    vector: &ArrayView1<'_, T>,
    mut output: ArrayViewMut1<'_, T>,
) -> Result<(), MatrixError>
where
    T: NabledReal,
{
    validate_matrix_non_empty(matrix)?;
    validate_vector_non_empty(vector)?;
    if vector.len() != matrix.ncols() || output.len() != matrix.nrows() {
        return Err(MatrixError::DimensionMismatch);
    }

    for row in 0..matrix.nrows() {
        let mut sum = T::zero();
        for col in 0..matrix.ncols() {
            sum += matrix[[row, col]] * vector[col];
        }
        output[row] = sum;
    }
    Ok(())
}

/// Compute complex dense matrix-vector product `y = A x`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn matvec_complex(
    matrix: &Array2<Complex64>,
    vector: &Array1<Complex64>,
) -> Result<Array1<Complex64>, MatrixError> {
    let matrix_view = matrix.view();
    let vector_view = vector.view();
    validate_matrix_non_empty_complex(&matrix_view)?;
    validate_vector_non_empty_complex(&vector_view)?;
    if vector.len() != matrix.ncols() {
        return Err(MatrixError::DimensionMismatch);
    }
    Ok(matrix.dot(vector))
}

/// Compute complex dense matrix-vector product `y = A x` from views.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn matvec_complex_view(
    matrix: &ArrayView2<'_, Complex64>,
    vector: &ArrayView1<'_, Complex64>,
) -> Result<Array1<Complex64>, MatrixError> {
    validate_matrix_non_empty_complex(matrix)?;
    validate_vector_non_empty_complex(vector)?;
    if vector.len() != matrix.ncols() {
        return Err(MatrixError::DimensionMismatch);
    }
    Ok(matrix.dot(vector))
}

/// Compute complex dense matrix-vector product `y = A x` into `output`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn matvec_complex_into(
    matrix: &Array2<Complex64>,
    vector: &Array1<Complex64>,
    output: &mut Array1<Complex64>,
) -> Result<(), MatrixError> {
    matvec_complex_view_into(&matrix.view(), &vector.view(), output.view_mut())
}

/// Compute complex dense matrix-vector product `y = A x` from views into `output`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn matvec_complex_view_into(
    matrix: &ArrayView2<'_, Complex64>,
    vector: &ArrayView1<'_, Complex64>,
    mut output: ArrayViewMut1<'_, Complex64>,
) -> Result<(), MatrixError> {
    validate_matrix_non_empty_complex(matrix)?;
    validate_vector_non_empty_complex(vector)?;
    if vector.len() != matrix.ncols() || output.len() != matrix.nrows() {
        return Err(MatrixError::DimensionMismatch);
    }

    for row in 0..matrix.nrows() {
        let mut sum = Complex64::new(0.0, 0.0);
        for col in 0..matrix.ncols() {
            sum += matrix[[row, col]] * vector[col];
        }
        output[row] = sum;
    }
    Ok(())
}

/// Compute dense matrix-matrix product `C = A B`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn matmat<T>(left: &Array2<T>, right: &Array2<T>) -> Result<Array2<T>, MatrixError>
where
    T: NabledReal,
    crate::accelerator::backends::CpuBackend: crate::accelerator::kernels::MatMatKernel<T>,
{
    let left_view = left.view();
    let right_view = right.view();
    validate_matrix_non_empty(&left_view)?;
    validate_matrix_non_empty(&right_view)?;
    if left.ncols() != right.nrows() {
        return Err(MatrixError::DimensionMismatch);
    }
    matmat_cpu(left, right).map_err(map_accelerator_error_to_matrix)
}

/// Compute dense matrix-matrix product `C = A B` from views.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn matmat_view<T>(
    left: &ArrayView2<'_, T>,
    right: &ArrayView2<'_, T>,
) -> Result<Array2<T>, MatrixError>
where
    T: NabledReal,
{
    validate_matrix_non_empty(left)?;
    validate_matrix_non_empty(right)?;
    if left.ncols() != right.nrows() {
        return Err(MatrixError::DimensionMismatch);
    }
    Ok(left.dot(right))
}

/// Compute dense matrix-matrix product `C = A B` into `output`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn matmat_into<T>(
    left: &Array2<T>,
    right: &Array2<T>,
    output: &mut Array2<T>,
) -> Result<(), MatrixError>
where
    T: NabledReal,
{
    let left_view = left.view();
    let right_view = right.view();
    matmat_view_into(&left_view, &right_view, output.view_mut())
}

/// Compute dense matrix-matrix product `C = A B` from views into `output`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn matmat_view_into<T>(
    left: &ArrayView2<'_, T>,
    right: &ArrayView2<'_, T>,
    mut output: ArrayViewMut2<'_, T>,
) -> Result<(), MatrixError>
where
    T: NabledReal,
{
    validate_matrix_non_empty(left)?;
    validate_matrix_non_empty(right)?;
    if left.ncols() != right.nrows() || output.dim() != (left.nrows(), right.ncols()) {
        return Err(MatrixError::DimensionMismatch);
    }

    general_mat_mul(T::one(), left, right, T::zero(), &mut output);
    Ok(())
}

/// Compute complex dense matrix-matrix product `C = A B`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn matmat_complex(
    left: &Array2<Complex64>,
    right: &Array2<Complex64>,
) -> Result<Array2<Complex64>, MatrixError> {
    let left_view = left.view();
    let right_view = right.view();
    validate_matrix_non_empty_complex(&left_view)?;
    validate_matrix_non_empty_complex(&right_view)?;
    if left.ncols() != right.nrows() {
        return Err(MatrixError::DimensionMismatch);
    }
    Ok(left.dot(right))
}

/// Compute complex dense matrix-matrix product `C = A B` from views.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn matmat_complex_view(
    left: &ArrayView2<'_, Complex64>,
    right: &ArrayView2<'_, Complex64>,
) -> Result<Array2<Complex64>, MatrixError> {
    validate_matrix_non_empty_complex(left)?;
    validate_matrix_non_empty_complex(right)?;
    if left.ncols() != right.nrows() {
        return Err(MatrixError::DimensionMismatch);
    }
    Ok(left.dot(right))
}

/// Compute complex dense matrix-matrix product `C = A B` into `output`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn matmat_complex_into(
    left: &Array2<Complex64>,
    right: &Array2<Complex64>,
    output: &mut Array2<Complex64>,
) -> Result<(), MatrixError> {
    matmat_complex_view_into(&left.view(), &right.view(), output.view_mut())
}

/// Compute complex dense matrix-matrix product `C = A B` from views into `output`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn matmat_complex_view_into(
    left: &ArrayView2<'_, Complex64>,
    right: &ArrayView2<'_, Complex64>,
    mut output: ArrayViewMut2<'_, Complex64>,
) -> Result<(), MatrixError> {
    validate_matrix_non_empty_complex(left)?;
    validate_matrix_non_empty_complex(right)?;
    if left.ncols() != right.nrows() || output.dim() != (left.nrows(), right.ncols()) {
        return Err(MatrixError::DimensionMismatch);
    }

    general_mat_mul(Complex64::new(1.0, 0.0), left, right, Complex64::new(0.0, 0.0), &mut output);
    Ok(())
}

/// Apply one matrix to a batch of row-vectors.
///
/// Input is `(batch, cols)` and matrix is `(rows, cols)`, output is `(batch, rows)`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn batched_row_matvec<T>(
    batch_vectors: &Array2<T>,
    matrix: &Array2<T>,
) -> Result<Array2<T>, MatrixError>
where
    T: NabledReal,
    crate::accelerator::backends::CpuBackend:
        crate::accelerator::kernels::BatchedRowMatVecKernel<T>,
{
    let batch_view = batch_vectors.view();
    let matrix_view = matrix.view();
    validate_matrix_non_empty(&batch_view)?;
    validate_matrix_non_empty(&matrix_view)?;
    if batch_vectors.ncols() != matrix.ncols() {
        return Err(MatrixError::DimensionMismatch);
    }

    batched_row_matvec_cpu(batch_vectors, matrix).map_err(map_accelerator_error_to_matrix)
}

/// Apply one matrix to a batch of row-vectors from views.
///
/// Input is `(batch, cols)` and matrix is `(rows, cols)`, output is `(batch, rows)`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn batched_row_matvec_view<T>(
    batch_vectors: &ArrayView2<'_, T>,
    matrix: &ArrayView2<'_, T>,
) -> Result<Array2<T>, MatrixError>
where
    T: NabledReal,
{
    let mut output = Array2::<T>::zeros((batch_vectors.nrows(), matrix.nrows()));
    batched_row_matvec_view_into(batch_vectors, matrix, output.view_mut())?;
    Ok(output)
}

/// Apply one matrix to a batch of row-vectors into `output`.
///
/// Input is `(batch, cols)` and matrix is `(rows, cols)`, output must be `(batch, rows)`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn batched_row_matvec_into<T>(
    batch_vectors: &Array2<T>,
    matrix: &Array2<T>,
    output: &mut Array2<T>,
) -> Result<(), MatrixError>
where
    T: NabledReal,
{
    let batch_view = batch_vectors.view();
    let matrix_view = matrix.view();
    batched_row_matvec_view_into(&batch_view, &matrix_view, output.view_mut())
}

/// Apply one matrix to a batch of row-vectors from views into `output`.
///
/// Input is `(batch, cols)` and matrix is `(rows, cols)`, output must be `(batch, rows)`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn batched_row_matvec_view_into<T>(
    batch_vectors: &ArrayView2<'_, T>,
    matrix: &ArrayView2<'_, T>,
    mut output: ArrayViewMut2<'_, T>,
) -> Result<(), MatrixError>
where
    T: NabledReal,
{
    validate_matrix_non_empty(batch_vectors)?;
    validate_matrix_non_empty(matrix)?;
    if batch_vectors.ncols() != matrix.ncols()
        || output.dim() != (batch_vectors.nrows(), matrix.nrows())
    {
        return Err(MatrixError::DimensionMismatch);
    }

    output.fill(T::zero());
    for batch in 0..batch_vectors.nrows() {
        for row in 0..matrix.nrows() {
            let mut sum = T::zero();
            for col in 0..matrix.ncols() {
                sum += batch_vectors[[batch, col]] * matrix[[row, col]];
            }
            output[[batch, row]] = sum;
        }
    }
    Ok(())
}

/// Compute batched dense matrix-matrix products.
///
/// Inputs are `(batch, m, k)` and `(batch, k, n)` and output is `(batch, m, n)`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn batched_matmat<T>(
    left_batches: &Array3<T>,
    right_batches: &Array3<T>,
) -> Result<Array3<T>, MatrixError>
where
    T: NabledReal,
    crate::accelerator::backends::CpuBackend: crate::accelerator::kernels::BatchedMatMatKernel<T>,
{
    let left_view = left_batches.view();
    let right_view = right_batches.view();
    validate_tensor_non_empty(&left_view)?;
    validate_tensor_non_empty(&right_view)?;
    if left_batches.dim().0 != right_batches.dim().0
        || left_batches.dim().2 != right_batches.dim().1
    {
        return Err(MatrixError::DimensionMismatch);
    }
    batched_matmat_cpu(left_batches, right_batches).map_err(map_accelerator_error_to_matrix)
}

/// Compute batched dense matrix-matrix products from views.
///
/// Inputs are `(batch, m, k)` and `(batch, k, n)` and output is `(batch, m, n)`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn batched_matmat_view<T>(
    left_batches: &ArrayView3<'_, T>,
    right_batches: &ArrayView3<'_, T>,
) -> Result<Array3<T>, MatrixError>
where
    T: NabledReal,
{
    let mut output =
        Array3::<T>::zeros((left_batches.dim().0, left_batches.dim().1, right_batches.dim().2));
    batched_matmat_view_into(left_batches, right_batches, output.view_mut())?;
    Ok(output)
}

/// Compute batched dense matrix-matrix products into `output`.
///
/// Inputs are `(batch, m, k)` and `(batch, k, n)` and output is `(batch, m, n)`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn batched_matmat_into<T>(
    left_batches: &Array3<T>,
    right_batches: &Array3<T>,
    output: &mut Array3<T>,
) -> Result<(), MatrixError>
where
    T: NabledReal,
{
    batched_matmat_view_into(&left_batches.view(), &right_batches.view(), output.view_mut())
}

/// Compute batched dense matrix-matrix products from views into `output`.
///
/// Inputs are `(batch, m, k)` and `(batch, k, n)` and output is `(batch, m, n)`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn batched_matmat_view_into<T>(
    left_batches: &ArrayView3<'_, T>,
    right_batches: &ArrayView3<'_, T>,
    mut output: ArrayViewMut3<'_, T>,
) -> Result<(), MatrixError>
where
    T: NabledReal,
{
    validate_tensor_non_empty(left_batches)?;
    validate_tensor_non_empty(right_batches)?;
    if left_batches.dim().0 != right_batches.dim().0
        || left_batches.dim().2 != right_batches.dim().1
        || output.dim() != (left_batches.dim().0, left_batches.dim().1, right_batches.dim().2)
    {
        return Err(MatrixError::DimensionMismatch);
    }

    let batch = left_batches.dim().0;
    for b in 0..batch {
        let left_matrix = left_batches.slice(s![b, .., ..]);
        let right_matrix = right_batches.slice(s![b, .., ..]);
        let mut out_matrix = output.slice_mut(s![b, .., ..]);
        general_mat_mul(T::one(), &left_matrix, &right_matrix, T::zero(), &mut out_matrix);
    }

    Ok(())
}

/// Compute batched dense matrix-matrix products with a broadcast right matrix.
///
/// Inputs are `(batch, m, k)` and `(k, n)` and output is `(batch, m, n)`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn batched_matmat_broadcast_right<T>(
    left_batches: &Array3<T>,
    right: &Array2<T>,
) -> Result<Array3<T>, MatrixError>
where
    T: NabledReal,
{
    let mut output =
        Array3::<T>::zeros((left_batches.dim().0, left_batches.dim().1, right.ncols()));
    batched_matmat_broadcast_right_view_into(
        &left_batches.view(),
        &right.view(),
        output.view_mut(),
    )?;
    Ok(output)
}

/// Compute batched dense matrix-matrix products with a broadcast right matrix from views.
///
/// Inputs are `(batch, m, k)` and `(k, n)` and output is `(batch, m, n)`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn batched_matmat_broadcast_right_view<T>(
    left_batches: &ArrayView3<'_, T>,
    right: &ArrayView2<'_, T>,
) -> Result<Array3<T>, MatrixError>
where
    T: NabledReal,
{
    let mut output =
        Array3::<T>::zeros((left_batches.dim().0, left_batches.dim().1, right.ncols()));
    batched_matmat_broadcast_right_view_into(left_batches, right, output.view_mut())?;
    Ok(output)
}

/// Compute batched dense matrix-matrix products with a broadcast right matrix into `output`.
///
/// Inputs are `(batch, m, k)` and `(k, n)` and output is `(batch, m, n)`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn batched_matmat_broadcast_right_into<T>(
    left_batches: &Array3<T>,
    right: &Array2<T>,
    output: &mut Array3<T>,
) -> Result<(), MatrixError>
where
    T: NabledReal,
{
    batched_matmat_broadcast_right_view_into(&left_batches.view(), &right.view(), output.view_mut())
}

/// Compute batched dense matrix-matrix products with a broadcast right matrix from views into
/// `output`.
///
/// Inputs are `(batch, m, k)` and `(k, n)` and output is `(batch, m, n)`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn batched_matmat_broadcast_right_view_into<T>(
    left_batches: &ArrayView3<'_, T>,
    right: &ArrayView2<'_, T>,
    mut output: ArrayViewMut3<'_, T>,
) -> Result<(), MatrixError>
where
    T: NabledReal,
{
    validate_tensor_non_empty(left_batches)?;
    validate_matrix_non_empty(right)?;
    if left_batches.dim().2 != right.nrows()
        || output.dim() != (left_batches.dim().0, left_batches.dim().1, right.ncols())
    {
        return Err(MatrixError::DimensionMismatch);
    }

    let batch = left_batches.dim().0;
    for b in 0..batch {
        let left_matrix = left_batches.slice(s![b, .., ..]);
        let mut out_matrix = output.slice_mut(s![b, .., ..]);
        general_mat_mul(T::one(), &left_matrix, right, T::zero(), &mut out_matrix);
    }
    Ok(())
}

/// Compute batched dense matrix-matrix products with a broadcast left matrix.
///
/// Inputs are `(m, k)` and `(batch, k, n)` and output is `(batch, m, n)`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn batched_matmat_broadcast_left<T>(
    left: &Array2<T>,
    right_batches: &Array3<T>,
) -> Result<Array3<T>, MatrixError>
where
    T: NabledReal,
{
    let mut output =
        Array3::<T>::zeros((right_batches.dim().0, left.nrows(), right_batches.dim().2));
    batched_matmat_broadcast_left_view_into(
        &left.view(),
        &right_batches.view(),
        output.view_mut(),
    )?;
    Ok(output)
}

/// Compute batched dense matrix-matrix products with a broadcast left matrix from views.
///
/// Inputs are `(m, k)` and `(batch, k, n)` and output is `(batch, m, n)`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn batched_matmat_broadcast_left_view<T>(
    left: &ArrayView2<'_, T>,
    right_batches: &ArrayView3<'_, T>,
) -> Result<Array3<T>, MatrixError>
where
    T: NabledReal,
{
    let mut output =
        Array3::<T>::zeros((right_batches.dim().0, left.nrows(), right_batches.dim().2));
    batched_matmat_broadcast_left_view_into(left, right_batches, output.view_mut())?;
    Ok(output)
}

/// Compute batched dense matrix-matrix products with a broadcast left matrix into `output`.
///
/// Inputs are `(m, k)` and `(batch, k, n)` and output is `(batch, m, n)`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn batched_matmat_broadcast_left_into<T>(
    left: &Array2<T>,
    right_batches: &Array3<T>,
    output: &mut Array3<T>,
) -> Result<(), MatrixError>
where
    T: NabledReal,
{
    batched_matmat_broadcast_left_view_into(&left.view(), &right_batches.view(), output.view_mut())
}

/// Compute batched dense matrix-matrix products with a broadcast left matrix from views into
/// `output`.
///
/// Inputs are `(m, k)` and `(batch, k, n)` and output is `(batch, m, n)`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn batched_matmat_broadcast_left_view_into<T>(
    left: &ArrayView2<'_, T>,
    right_batches: &ArrayView3<'_, T>,
    mut output: ArrayViewMut3<'_, T>,
) -> Result<(), MatrixError>
where
    T: NabledReal,
{
    validate_matrix_non_empty(left)?;
    validate_tensor_non_empty(right_batches)?;
    if left.ncols() != right_batches.dim().1
        || output.dim() != (right_batches.dim().0, left.nrows(), right_batches.dim().2)
    {
        return Err(MatrixError::DimensionMismatch);
    }

    let batch = right_batches.dim().0;
    for b in 0..batch {
        let right_matrix = right_batches.slice(s![b, .., ..]);
        let mut out_matrix = output.slice_mut(s![b, .., ..]);
        general_mat_mul(T::one(), left, &right_matrix, T::zero(), &mut out_matrix);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2, Array3};
    use num_complex::Complex64;

    use super::*;

    #[test]
    fn matvec_variants_match() {
        let matrix = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 0.0, 1.0, 1.0]).unwrap();
        let vector = Array1::from_vec(vec![1.0, 0.0, 2.0]);

        let allocating = matvec(&matrix, &vector).unwrap();
        let viewed = matvec_view(&matrix.view(), &vector.view()).unwrap();
        let mut into = Array1::<f64>::zeros(2);
        matvec_into(&matrix, &vector, &mut into).unwrap();

        for i in 0..2 {
            assert!((allocating[i] - viewed[i]).abs() < 1e-12);
            assert!((allocating[i] - into[i]).abs() < 1e-12);
        }
        assert!((allocating[0] - 7.0).abs() < 1e-12);
        assert!((allocating[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn real_f32_variants_match() {
        let matrix =
            Array2::from_shape_vec((2, 3), vec![1.0_f32, 2.0, 3.0, 0.0, 1.0, 1.0]).unwrap();
        let vector = Array1::from_vec(vec![1.0_f32, 0.0, 2.0]);

        let matvec_alloc = matvec(&matrix, &vector).unwrap();
        let mut matvec_into_out = Array1::<f32>::zeros(2);
        matvec_into(&matrix, &vector, &mut matvec_into_out).unwrap();
        for i in 0..2 {
            assert!((matvec_alloc[i] - matvec_into_out[i]).abs() < 1e-6);
        }

        let left = Array2::from_shape_vec((2, 3), vec![1.0_f32, 2.0, 0.0, 0.0, 1.0, 1.0]).unwrap();
        let right = Array2::from_shape_vec((3, 2), vec![1.0_f32, 0.0, 2.0, 1.0, 1.0, 3.0]).unwrap();
        let matmat_alloc = matmat(&left, &right).unwrap();
        let mut matmat_into_out = Array2::<f32>::zeros((2, 2));
        matmat_into(&left, &right, &mut matmat_into_out).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!((matmat_alloc[[i, j]] - matmat_into_out[[i, j]]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn matvec_complex_variants_match() {
        let matrix = Array2::from_shape_vec((2, 2), vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(0.0, -1.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(1.0, 2.0),
        ])
        .unwrap();
        let vector = Array1::from_vec(vec![Complex64::new(1.0, 0.0), Complex64::new(0.5, -0.5)]);

        let allocating = matvec_complex(&matrix, &vector).unwrap();
        let viewed = matvec_complex_view(&matrix.view(), &vector.view()).unwrap();
        let mut into = Array1::<Complex64>::zeros(2);
        matvec_complex_into(&matrix, &vector, &mut into).unwrap();

        for i in 0..2 {
            assert!((allocating[i] - viewed[i]).norm() < 1e-12);
            assert!((allocating[i] - into[i]).norm() < 1e-12);
        }
    }

    #[test]
    fn matmat_variants_match() {
        let left = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 0.0, 0.0, 1.0, 1.0]).unwrap();
        let right = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 2.0, 1.0, 1.0, 3.0]).unwrap();

        let allocating = matmat(&left, &right).unwrap();
        let viewed = matmat_view(&left.view(), &right.view()).unwrap();
        let mut into = Array2::<f64>::zeros((2, 2));
        matmat_into(&left, &right, &mut into).unwrap();

        for i in 0..2 {
            for j in 0..2 {
                assert!((allocating[[i, j]] - viewed[[i, j]]).abs() < 1e-12);
                assert!((allocating[[i, j]] - into[[i, j]]).abs() < 1e-12);
            }
        }
        assert!((allocating[[0, 0]] - 5.0).abs() < 1e-12);
        assert!((allocating[[0, 1]] - 2.0).abs() < 1e-12);
        assert!((allocating[[1, 0]] - 3.0).abs() < 1e-12);
        assert!((allocating[[1, 1]] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn matmat_complex_variants_match() {
        let left = Array2::from_shape_vec((2, 2), vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(1.0, 0.0),
        ])
        .unwrap();
        let right = Array2::from_shape_vec((2, 2), vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(1.0, -1.0),
        ])
        .unwrap();

        let allocating = matmat_complex(&left, &right).unwrap();
        let viewed = matmat_complex_view(&left.view(), &right.view()).unwrap();
        let mut into = Array2::<Complex64>::zeros((2, 2));
        matmat_complex_into(&left, &right, &mut into).unwrap();

        for i in 0..2 {
            for j in 0..2 {
                assert!((allocating[[i, j]] - viewed[[i, j]]).norm() < 1e-12);
                assert!((allocating[[i, j]] - into[[i, j]]).norm() < 1e-12);
            }
        }
    }

    #[test]
    fn batched_row_matvec_variants_match() {
        let vectors = Array2::from_shape_vec((2, 3), vec![1.0, 0.0, 2.0, 0.5, -1.0, 1.0]).unwrap();
        let matrix = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 0.0, 1.0, 1.0]).unwrap();

        let allocating = batched_row_matvec(&vectors, &matrix).unwrap();
        let viewed = batched_row_matvec_view(&vectors.view(), &matrix.view()).unwrap();
        let mut into = Array2::<f64>::zeros((2, 2));
        batched_row_matvec_into(&vectors, &matrix, &mut into).unwrap();

        for i in 0..2 {
            for j in 0..2 {
                assert!((allocating[[i, j]] - viewed[[i, j]]).abs() < 1e-12);
                assert!((allocating[[i, j]] - into[[i, j]]).abs() < 1e-12);
            }
        }

        assert!((allocating[[0, 0]] - 7.0).abs() < 1e-12);
        assert!((allocating[[0, 1]] - 2.0).abs() < 1e-12);
        assert!((allocating[[1, 0]] - 1.5).abs() < 1e-12);
        assert!((allocating[[1, 1]] - 0.0).abs() < 1e-12);
    }

    #[test]
    fn rejects_dimension_mismatch_and_empty_inputs() {
        let matrix = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let bad_vector = Array1::from_vec(vec![1.0]);
        assert!(matches!(matvec(&matrix, &bad_vector), Err(MatrixError::DimensionMismatch)));

        let left = Array2::from_shape_vec((2, 3), vec![1.0; 6]).unwrap();
        let right = Array2::from_shape_vec((2, 2), vec![1.0; 4]).unwrap();
        assert!(matches!(matmat(&left, &right), Err(MatrixError::DimensionMismatch)));

        let empty_matrix = Array2::<f64>::zeros((0, 0));
        let empty_vector = Array1::<f64>::zeros(0);
        assert!(matches!(matvec(&empty_matrix, &empty_vector), Err(MatrixError::EmptyInput)));

        let left_batches = Array3::<f64>::zeros((1, 2, 3));
        let right_batches = Array3::<f64>::zeros((1, 2, 2));
        assert!(matches!(
            batched_matmat(&left_batches, &right_batches),
            Err(MatrixError::DimensionMismatch)
        ));
    }

    #[test]
    fn batched_matmat_variants_match() {
        let left_batches = Array3::from_shape_vec((2, 2, 3), vec![
            1.0, 2.0, 0.0, 0.0, 1.0, 1.0, 2.0, 0.0, 1.0, 1.0, 3.0, 2.0,
        ])
        .unwrap();
        let right_batches = Array3::from_shape_vec((2, 3, 2), vec![
            1.0, 0.0, 2.0, 1.0, 1.0, 3.0, 0.0, 2.0, 1.0, 1.0, 3.0, 0.0,
        ])
        .unwrap();

        let allocating = batched_matmat(&left_batches, &right_batches).unwrap();
        let viewed = batched_matmat_view(&left_batches.view(), &right_batches.view()).unwrap();
        let mut into = Array3::<f64>::zeros((2, 2, 2));
        batched_matmat_into(&left_batches, &right_batches, &mut into).unwrap();

        for b in 0..2 {
            for i in 0..2 {
                for j in 0..2 {
                    assert!((allocating[[b, i, j]] - viewed[[b, i, j]]).abs() < 1e-12);
                    assert!((allocating[[b, i, j]] - into[[b, i, j]]).abs() < 1e-12);
                }
            }
        }

        assert!((allocating[[0, 0, 0]] - 5.0).abs() < 1e-12);
        assert!((allocating[[0, 0, 1]] - 2.0).abs() < 1e-12);
        assert!((allocating[[0, 1, 0]] - 3.0).abs() < 1e-12);
        assert!((allocating[[0, 1, 1]] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn batched_f32_variants_match() {
        let left_batches = Array3::from_shape_vec((2, 2, 3), vec![
            1.0_f32, 2.0, 0.0, 0.0, 1.0, 1.0, 2.0, 0.0, 1.0, 1.0, 3.0, 2.0,
        ])
        .unwrap();
        let right_batches = Array3::from_shape_vec((2, 3, 2), vec![
            1.0_f32, 0.0, 2.0, 1.0, 1.0, 3.0, 0.0, 2.0, 1.0, 1.0, 3.0, 0.0,
        ])
        .unwrap();

        let alloc = batched_matmat(&left_batches, &right_batches).unwrap();
        let mut into = Array3::<f32>::zeros((2, 2, 2));
        batched_matmat_into(&left_batches, &right_batches, &mut into).unwrap();
        for b in 0..2 {
            for i in 0..2 {
                for j in 0..2 {
                    assert!((alloc[[b, i, j]] - into[[b, i, j]]).abs() < 1e-5);
                }
            }
        }
    }

    #[test]
    fn batched_matmat_broadcast_variants_match() {
        let left_batches = Array3::from_shape_vec((2, 2, 3), vec![
            1.0, 2.0, 0.0, 0.0, 1.0, 1.0, //
            2.0, 0.0, 1.0, 1.0, 3.0, 2.0,
        ])
        .unwrap();
        let right = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 2.0, 1.0, 1.0, 3.0]).unwrap();

        let right_alloc = batched_matmat_broadcast_right(&left_batches, &right).unwrap();
        let right_view =
            batched_matmat_broadcast_right_view(&left_batches.view(), &right.view()).unwrap();
        let mut right_into = Array3::<f64>::zeros((2, 2, 2));
        batched_matmat_broadcast_right_into(&left_batches, &right, &mut right_into).unwrap();

        for b in 0..2 {
            for i in 0..2 {
                for j in 0..2 {
                    assert!((right_alloc[[b, i, j]] - right_view[[b, i, j]]).abs() < 1e-12);
                    assert!((right_alloc[[b, i, j]] - right_into[[b, i, j]]).abs() < 1e-12);
                }
            }
        }

        let left = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 0.0, 0.0, 1.0, 1.0]).unwrap();
        let right_batches = Array3::from_shape_vec((2, 3, 2), vec![
            1.0, 0.0, 2.0, 1.0, 1.0, 3.0, //
            0.0, 2.0, 1.0, 1.0, 3.0, 0.0,
        ])
        .unwrap();

        let left_alloc = batched_matmat_broadcast_left(&left, &right_batches).unwrap();
        let left_view =
            batched_matmat_broadcast_left_view(&left.view(), &right_batches.view()).unwrap();
        let mut left_into = Array3::<f64>::zeros((2, 2, 2));
        batched_matmat_broadcast_left_into(&left, &right_batches, &mut left_into).unwrap();

        for b in 0..2 {
            for i in 0..2 {
                for j in 0..2 {
                    assert!((left_alloc[[b, i, j]] - left_view[[b, i, j]]).abs() < 1e-12);
                    assert!((left_alloc[[b, i, j]] - left_into[[b, i, j]]).abs() < 1e-12);
                }
            }
        }
    }
}
