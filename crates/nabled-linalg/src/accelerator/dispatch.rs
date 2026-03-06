use std::ops::{AddAssign, Mul};

use nabled_core::scalar::NabledReal;
use ndarray::{Array1, Array2, Array3, ArrayD};
use num_traits::Float;

use super::backends::{AcceleratorError, BackendKind, CpuBackend, GpuBackend};
use super::cpu::{
    batched_matmat_serial, batched_matmat_serial_f32, batched_row_matvec_serial, dot_serial,
    matmat_serial, matmat_serial_f32, matvec_serial, matvec_serial_f32, pairwise_cosine_serial,
    pairwise_l2_serial, sparse_matmat_dense_serial, sparse_matmat_sparse_serial,
    sparse_matvec_serial, tensor_batched_matmul_last_two_serial, tensor_contract_axes_serial,
    tensor_sum_last_axis_serial, triangular_solve_mat_serial, triangular_solve_vec_serial,
};
use super::gpu::{
    batched_matmat_gpu_f32, batched_matmat_gpu_f64, batched_row_matvec_gpu_f32,
    batched_row_matvec_gpu_f64, dot_gpu_f32, dot_gpu_f64, matmat_gpu_f32, matmat_gpu_f64,
    matvec_gpu_f32, matvec_gpu_f64, pairwise_cosine_gpu_f32, pairwise_cosine_gpu_f64,
    pairwise_l2_gpu_f32, pairwise_l2_gpu_f64, sparse_matmat_dense_gpu_f32,
    sparse_matmat_dense_gpu_f64, sparse_matmat_sparse_gpu_f32, sparse_matmat_sparse_gpu_f64,
    sparse_matvec_gpu_f32, sparse_matvec_gpu_f64, tensor_batched_matmul_last_two_gpu_complex64,
    tensor_batched_matmul_last_two_gpu_f32, tensor_batched_matmul_last_two_gpu_f64,
    tensor_contract_axes_gpu_complex64, tensor_contract_axes_gpu_f32, tensor_contract_axes_gpu_f64,
    tensor_sum_last_axis_gpu_complex64, tensor_sum_last_axis_gpu_f32, tensor_sum_last_axis_gpu_f64,
    triangular_solve_mat_gpu_f32, triangular_solve_mat_gpu_f64, triangular_solve_vec_gpu_f32,
    triangular_solve_vec_gpu_f64,
};
use super::kernels::{
    BatchedMatMatKernel, BatchedRowMatVecKernel, DotKernel, MatMatKernel, MatVecKernel,
    PairwiseCosineKernel, PairwiseL2Kernel, SparseMatMatDenseKernel, SparseMatMatSparseKernel,
    SparseMatVecKernel, TensorBatchedMatMulKernel, TensorContractKernel,
    TensorLastAxisReductionKernel, TriangularSolveMatKernel, TriangularSolveVecKernel,
};
use super::policy::{
    should_attempt_gpu_batched_matmat, should_attempt_gpu_batched_row_matvec,
    should_attempt_gpu_dot, should_attempt_gpu_matmat, should_attempt_gpu_matvec,
    should_attempt_gpu_pairwise, should_attempt_gpu_sparse_matmat_dense,
    should_attempt_gpu_sparse_matmat_sparse, should_attempt_gpu_sparse_matvec,
    should_attempt_gpu_tensor_batched_matmul, should_attempt_gpu_tensor_contract,
    should_attempt_gpu_tensor_sum, should_attempt_gpu_triangular_solve_mat,
    should_attempt_gpu_triangular_solve_vec,
};
use crate::sparse::CsrMatrix;

#[inline]
fn fallback_or_cpu_gpu<T, F>(
    attempt: Result<T, AcceleratorError>,
    fallback: F,
) -> Result<T, AcceleratorError>
where
    F: FnOnce() -> Result<T, AcceleratorError>,
{
    match attempt {
        Ok(value) => Ok(value),
        Err(
            AcceleratorError::FeatureNotEnabled
            | AcceleratorError::DeviceUnavailable
            | AcceleratorError::DimensionMismatch
            | AcceleratorError::KernelExecutionFailed
            | AcceleratorError::UnsupportedBackend(BackendKind::Gpu),
        ) => fallback(),
        Err(error) => Err(error),
    }
}

impl MatMatKernel<f64> for CpuBackend {
    fn matmat(left: &Array2<f64>, right: &Array2<f64>) -> Result<Array2<f64>, AcceleratorError> {
        matmat_serial(left, right)
    }
}

/// Note: current GPU backend path attempts GPU execution and falls back to CPU serial execution.
impl MatMatKernel<f64> for GpuBackend {
    fn matmat(left: &Array2<f64>, right: &Array2<f64>) -> Result<Array2<f64>, AcceleratorError> {
        if !should_attempt_gpu_matmat(left.nrows(), left.ncols(), right.ncols()) {
            return matmat_serial(left, right);
        }
        fallback_or_cpu_gpu(matmat_gpu_f64(left, right), || matmat_serial(left, right))
    }
}

impl MatMatKernel<f32> for CpuBackend {
    fn matmat(left: &Array2<f32>, right: &Array2<f32>) -> Result<Array2<f32>, AcceleratorError> {
        matmat_serial_f32(left, right)
    }
}

/// Note: current GPU backend path attempts GPU execution and falls back to CPU serial execution.
impl MatMatKernel<f32> for GpuBackend {
    fn matmat(left: &Array2<f32>, right: &Array2<f32>) -> Result<Array2<f32>, AcceleratorError> {
        if !should_attempt_gpu_matmat(left.nrows(), left.ncols(), right.ncols()) {
            return matmat_serial_f32(left, right);
        }
        fallback_or_cpu_gpu(matmat_gpu_f32(left, right), || matmat_serial_f32(left, right))
    }
}

impl MatVecKernel<f64> for CpuBackend {
    fn matvec(matrix: &Array2<f64>, vector: &Array1<f64>) -> Result<Array1<f64>, AcceleratorError> {
        matvec_serial(matrix, vector)
    }
}

/// Note: current GPU backend path attempts GPU execution and falls back to CPU serial execution.
impl MatVecKernel<f64> for GpuBackend {
    fn matvec(matrix: &Array2<f64>, vector: &Array1<f64>) -> Result<Array1<f64>, AcceleratorError> {
        if !should_attempt_gpu_matvec(matrix.nrows(), matrix.ncols()) {
            return matvec_serial(matrix, vector);
        }
        fallback_or_cpu_gpu(matvec_gpu_f64(matrix, vector), || matvec_serial(matrix, vector))
    }
}

impl MatVecKernel<f32> for CpuBackend {
    fn matvec(matrix: &Array2<f32>, vector: &Array1<f32>) -> Result<Array1<f32>, AcceleratorError> {
        matvec_serial_f32(matrix, vector)
    }
}

/// Note: current GPU backend path attempts GPU execution and falls back to CPU serial execution.
impl MatVecKernel<f32> for GpuBackend {
    fn matvec(matrix: &Array2<f32>, vector: &Array1<f32>) -> Result<Array1<f32>, AcceleratorError> {
        if !should_attempt_gpu_matvec(matrix.nrows(), matrix.ncols()) {
            return matvec_serial_f32(matrix, vector);
        }
        fallback_or_cpu_gpu(matvec_gpu_f32(matrix, vector), || matvec_serial_f32(matrix, vector))
    }
}

impl BatchedMatMatKernel<f64> for CpuBackend {
    fn batched_matmat(
        left_batches: &Array3<f64>,
        right_batches: &Array3<f64>,
    ) -> Result<Array3<f64>, AcceleratorError> {
        batched_matmat_serial(left_batches, right_batches)
    }
}

/// Note: current GPU backend path attempts GPU execution and falls back to CPU serial execution.
impl BatchedMatMatKernel<f64> for GpuBackend {
    fn batched_matmat(
        left_batches: &Array3<f64>,
        right_batches: &Array3<f64>,
    ) -> Result<Array3<f64>, AcceleratorError> {
        if !should_attempt_gpu_batched_matmat(
            left_batches.dim().0,
            left_batches.dim().1,
            left_batches.dim().2,
            right_batches.dim().2,
        ) {
            return batched_matmat_serial(left_batches, right_batches);
        }
        fallback_or_cpu_gpu(batched_matmat_gpu_f64(left_batches, right_batches), || {
            batched_matmat_serial(left_batches, right_batches)
        })
    }
}

impl BatchedMatMatKernel<f32> for CpuBackend {
    fn batched_matmat(
        left_batches: &Array3<f32>,
        right_batches: &Array3<f32>,
    ) -> Result<Array3<f32>, AcceleratorError> {
        batched_matmat_serial_f32(left_batches, right_batches)
    }
}

/// Note: current GPU backend path attempts GPU execution and falls back to CPU serial execution.
impl BatchedMatMatKernel<f32> for GpuBackend {
    fn batched_matmat(
        left_batches: &Array3<f32>,
        right_batches: &Array3<f32>,
    ) -> Result<Array3<f32>, AcceleratorError> {
        if !should_attempt_gpu_batched_matmat(
            left_batches.dim().0,
            left_batches.dim().1,
            left_batches.dim().2,
            right_batches.dim().2,
        ) {
            return batched_matmat_serial_f32(left_batches, right_batches);
        }
        fallback_or_cpu_gpu(batched_matmat_gpu_f32(left_batches, right_batches), || {
            batched_matmat_serial_f32(left_batches, right_batches)
        })
    }
}

impl<T> SparseMatVecKernel<T> for CpuBackend
where
    T: NabledReal,
{
    fn sparse_matvec(
        matrix: &CsrMatrix<T>,
        vector: &Array1<T>,
    ) -> Result<Array1<T>, AcceleratorError> {
        sparse_matvec_serial(matrix, vector)
    }
}

/// Note: current GPU backend path attempts GPU execution and falls back to CPU serial execution.
impl SparseMatVecKernel<f32> for GpuBackend {
    fn sparse_matvec(
        matrix: &CsrMatrix<f32>,
        vector: &Array1<f32>,
    ) -> Result<Array1<f32>, AcceleratorError> {
        if !should_attempt_gpu_sparse_matvec(matrix.data.len()) {
            return sparse_matvec_serial(matrix, vector);
        }
        fallback_or_cpu_gpu(sparse_matvec_gpu_f32(matrix, vector), || {
            sparse_matvec_serial(matrix, vector)
        })
    }
}

/// Note: current GPU backend path attempts GPU execution and falls back to CPU serial execution.
impl SparseMatVecKernel<f64> for GpuBackend {
    fn sparse_matvec(
        matrix: &CsrMatrix<f64>,
        vector: &Array1<f64>,
    ) -> Result<Array1<f64>, AcceleratorError> {
        if !should_attempt_gpu_sparse_matvec(matrix.data.len()) {
            return sparse_matvec_serial(matrix, vector);
        }
        fallback_or_cpu_gpu(sparse_matvec_gpu_f64(matrix, vector), || {
            sparse_matvec_serial(matrix, vector)
        })
    }
}

impl BatchedRowMatVecKernel<f64> for CpuBackend {
    fn batched_row_matvec(
        batch_vectors: &Array2<f64>,
        matrix: &Array2<f64>,
    ) -> Result<Array2<f64>, AcceleratorError> {
        batched_row_matvec_serial(batch_vectors, matrix)
    }
}

/// Note: current GPU backend path attempts GPU execution and falls back to CPU serial execution.
impl BatchedRowMatVecKernel<f64> for GpuBackend {
    fn batched_row_matvec(
        batch_vectors: &Array2<f64>,
        matrix: &Array2<f64>,
    ) -> Result<Array2<f64>, AcceleratorError> {
        if !should_attempt_gpu_batched_row_matvec(
            batch_vectors.nrows(),
            batch_vectors.ncols(),
            matrix.ncols(),
        ) {
            return batched_row_matvec_serial(batch_vectors, matrix);
        }
        fallback_or_cpu_gpu(batched_row_matvec_gpu_f64(batch_vectors, matrix), || {
            batched_row_matvec_serial(batch_vectors, matrix)
        })
    }
}

impl BatchedRowMatVecKernel<f32> for CpuBackend {
    fn batched_row_matvec(
        batch_vectors: &Array2<f32>,
        matrix: &Array2<f32>,
    ) -> Result<Array2<f32>, AcceleratorError> {
        batched_row_matvec_serial(batch_vectors, matrix)
    }
}

/// Note: current GPU backend path attempts GPU execution and falls back to CPU serial execution.
impl BatchedRowMatVecKernel<f32> for GpuBackend {
    fn batched_row_matvec(
        batch_vectors: &Array2<f32>,
        matrix: &Array2<f32>,
    ) -> Result<Array2<f32>, AcceleratorError> {
        if !should_attempt_gpu_batched_row_matvec(
            batch_vectors.nrows(),
            batch_vectors.ncols(),
            matrix.ncols(),
        ) {
            return batched_row_matvec_serial(batch_vectors, matrix);
        }
        fallback_or_cpu_gpu(batched_row_matvec_gpu_f32(batch_vectors, matrix), || {
            batched_row_matvec_serial(batch_vectors, matrix)
        })
    }
}

impl<T> SparseMatMatDenseKernel<T> for CpuBackend
where
    T: NabledReal,
{
    fn sparse_matmat_dense(
        matrix: &CsrMatrix<T>,
        dense: &Array2<T>,
    ) -> Result<Array2<T>, AcceleratorError> {
        sparse_matmat_dense_serial(matrix, dense)
    }
}

/// Note: current GPU backend path attempts GPU execution and falls back to CPU serial execution.
impl SparseMatMatDenseKernel<f32> for GpuBackend {
    fn sparse_matmat_dense(
        matrix: &CsrMatrix<f32>,
        dense: &Array2<f32>,
    ) -> Result<Array2<f32>, AcceleratorError> {
        if !should_attempt_gpu_sparse_matmat_dense(matrix.data.len(), dense.ncols()) {
            return sparse_matmat_dense_serial(matrix, dense);
        }
        fallback_or_cpu_gpu(sparse_matmat_dense_gpu_f32(matrix, dense), || {
            sparse_matmat_dense_serial(matrix, dense)
        })
    }
}

/// Note: current GPU backend path attempts GPU execution and falls back to CPU serial execution.
impl SparseMatMatDenseKernel<f64> for GpuBackend {
    fn sparse_matmat_dense(
        matrix: &CsrMatrix<f64>,
        dense: &Array2<f64>,
    ) -> Result<Array2<f64>, AcceleratorError> {
        if !should_attempt_gpu_sparse_matmat_dense(matrix.data.len(), dense.ncols()) {
            return sparse_matmat_dense_serial(matrix, dense);
        }
        fallback_or_cpu_gpu(sparse_matmat_dense_gpu_f64(matrix, dense), || {
            sparse_matmat_dense_serial(matrix, dense)
        })
    }
}

impl<T> SparseMatMatSparseKernel<T> for CpuBackend
where
    T: NabledReal,
{
    fn sparse_matmat_sparse(
        left: &CsrMatrix<T>,
        right: &CsrMatrix<T>,
    ) -> Result<CsrMatrix<T>, AcceleratorError> {
        sparse_matmat_sparse_serial(left, right)
    }
}

/// Note: current GPU backend path attempts GPU execution and falls back to CPU serial execution.
impl SparseMatMatSparseKernel<f32> for GpuBackend {
    fn sparse_matmat_sparse(
        left: &CsrMatrix<f32>,
        right: &CsrMatrix<f32>,
    ) -> Result<CsrMatrix<f32>, AcceleratorError> {
        if !should_attempt_gpu_sparse_matmat_sparse(left.data.len(), right.data.len()) {
            return sparse_matmat_sparse_serial(left, right);
        }
        fallback_or_cpu_gpu(sparse_matmat_sparse_gpu_f32(left, right), || {
            sparse_matmat_sparse_serial(left, right)
        })
    }
}

/// Note: current GPU backend path attempts GPU execution and falls back to CPU serial execution.
impl SparseMatMatSparseKernel<f64> for GpuBackend {
    fn sparse_matmat_sparse(
        left: &CsrMatrix<f64>,
        right: &CsrMatrix<f64>,
    ) -> Result<CsrMatrix<f64>, AcceleratorError> {
        if !should_attempt_gpu_sparse_matmat_sparse(left.data.len(), right.data.len()) {
            return sparse_matmat_sparse_serial(left, right);
        }
        fallback_or_cpu_gpu(sparse_matmat_sparse_gpu_f64(left, right), || {
            sparse_matmat_sparse_serial(left, right)
        })
    }
}

impl<T> TriangularSolveVecKernel<T> for CpuBackend
where
    T: Float,
{
    fn triangular_solve_vec(
        matrix: &Array2<T>,
        rhs: &Array1<T>,
        lower: bool,
        unit_diagonal: bool,
    ) -> Result<Array1<T>, AcceleratorError> {
        triangular_solve_vec_serial(matrix, rhs, lower, unit_diagonal)
    }
}

/// Note: current GPU backend path attempts GPU execution and falls back to CPU serial execution.
impl TriangularSolveVecKernel<f32> for GpuBackend {
    fn triangular_solve_vec(
        matrix: &Array2<f32>,
        rhs: &Array1<f32>,
        lower: bool,
        unit_diagonal: bool,
    ) -> Result<Array1<f32>, AcceleratorError> {
        if !should_attempt_gpu_triangular_solve_vec(matrix.nrows()) {
            return triangular_solve_vec_serial(matrix, rhs, lower, unit_diagonal);
        }
        fallback_or_cpu_gpu(triangular_solve_vec_gpu_f32(matrix, rhs, lower, unit_diagonal), || {
            triangular_solve_vec_serial(matrix, rhs, lower, unit_diagonal)
        })
    }
}

/// Note: current GPU backend path attempts GPU execution and falls back to CPU serial execution.
impl TriangularSolveVecKernel<f64> for GpuBackend {
    fn triangular_solve_vec(
        matrix: &Array2<f64>,
        rhs: &Array1<f64>,
        lower: bool,
        unit_diagonal: bool,
    ) -> Result<Array1<f64>, AcceleratorError> {
        if !should_attempt_gpu_triangular_solve_vec(matrix.nrows()) {
            return triangular_solve_vec_serial(matrix, rhs, lower, unit_diagonal);
        }
        fallback_or_cpu_gpu(triangular_solve_vec_gpu_f64(matrix, rhs, lower, unit_diagonal), || {
            triangular_solve_vec_serial(matrix, rhs, lower, unit_diagonal)
        })
    }
}

impl<T> TriangularSolveMatKernel<T> for CpuBackend
where
    T: Float,
{
    fn triangular_solve_mat(
        matrix: &Array2<T>,
        rhs: &Array2<T>,
        lower: bool,
        unit_diagonal: bool,
    ) -> Result<Array2<T>, AcceleratorError> {
        triangular_solve_mat_serial(matrix, rhs, lower, unit_diagonal)
    }
}

/// Note: current GPU backend path attempts GPU execution and falls back to CPU serial execution.
impl TriangularSolveMatKernel<f32> for GpuBackend {
    fn triangular_solve_mat(
        matrix: &Array2<f32>,
        rhs: &Array2<f32>,
        lower: bool,
        unit_diagonal: bool,
    ) -> Result<Array2<f32>, AcceleratorError> {
        if !should_attempt_gpu_triangular_solve_mat(matrix.nrows(), rhs.ncols()) {
            return triangular_solve_mat_serial(matrix, rhs, lower, unit_diagonal);
        }
        fallback_or_cpu_gpu(triangular_solve_mat_gpu_f32(matrix, rhs, lower, unit_diagonal), || {
            triangular_solve_mat_serial(matrix, rhs, lower, unit_diagonal)
        })
    }
}

/// Note: current GPU backend path attempts GPU execution and falls back to CPU serial execution.
impl TriangularSolveMatKernel<f64> for GpuBackend {
    fn triangular_solve_mat(
        matrix: &Array2<f64>,
        rhs: &Array2<f64>,
        lower: bool,
        unit_diagonal: bool,
    ) -> Result<Array2<f64>, AcceleratorError> {
        if !should_attempt_gpu_triangular_solve_mat(matrix.nrows(), rhs.ncols()) {
            return triangular_solve_mat_serial(matrix, rhs, lower, unit_diagonal);
        }
        fallback_or_cpu_gpu(triangular_solve_mat_gpu_f64(matrix, rhs, lower, unit_diagonal), || {
            triangular_solve_mat_serial(matrix, rhs, lower, unit_diagonal)
        })
    }
}

impl<T> DotKernel<T> for CpuBackend
where
    T: NabledReal,
{
    fn dot(left: &Array1<T>, right: &Array1<T>) -> Result<T, AcceleratorError> {
        dot_serial(left, right)
    }
}

/// Note: current GPU backend path attempts GPU execution and falls back to CPU serial execution.
impl DotKernel<f32> for GpuBackend {
    fn dot(left: &Array1<f32>, right: &Array1<f32>) -> Result<f32, AcceleratorError> {
        if !should_attempt_gpu_dot(left.len()) {
            return dot_serial(left, right);
        }
        fallback_or_cpu_gpu(dot_gpu_f32(left, right), || dot_serial(left, right))
    }
}

/// Note: current GPU backend path attempts GPU execution and falls back to CPU serial execution.
impl DotKernel<f64> for GpuBackend {
    fn dot(left: &Array1<f64>, right: &Array1<f64>) -> Result<f64, AcceleratorError> {
        if !should_attempt_gpu_dot(left.len()) {
            return dot_serial(left, right);
        }
        fallback_or_cpu_gpu(dot_gpu_f64(left, right), || dot_serial(left, right))
    }
}

impl<T> PairwiseL2Kernel<T> for CpuBackend
where
    T: NabledReal,
{
    fn pairwise_l2(left: &Array2<T>, right: &Array2<T>) -> Result<Array2<T>, AcceleratorError> {
        pairwise_l2_serial(left, right)
    }
}

/// Note: current GPU backend path attempts GPU execution and falls back to CPU serial execution.
impl PairwiseL2Kernel<f32> for GpuBackend {
    fn pairwise_l2(
        left: &Array2<f32>,
        right: &Array2<f32>,
    ) -> Result<Array2<f32>, AcceleratorError> {
        if !should_attempt_gpu_pairwise(left.nrows(), right.nrows(), left.ncols()) {
            return pairwise_l2_serial(left, right);
        }
        fallback_or_cpu_gpu(pairwise_l2_gpu_f32(left, right), || pairwise_l2_serial(left, right))
    }
}

/// Note: current GPU backend path attempts GPU execution and falls back to CPU serial execution.
impl PairwiseL2Kernel<f64> for GpuBackend {
    fn pairwise_l2(
        left: &Array2<f64>,
        right: &Array2<f64>,
    ) -> Result<Array2<f64>, AcceleratorError> {
        if !should_attempt_gpu_pairwise(left.nrows(), right.nrows(), left.ncols()) {
            return pairwise_l2_serial(left, right);
        }
        fallback_or_cpu_gpu(pairwise_l2_gpu_f64(left, right), || pairwise_l2_serial(left, right))
    }
}

impl<T> PairwiseCosineKernel<T> for CpuBackend
where
    T: NabledReal,
{
    fn pairwise_cosine(left: &Array2<T>, right: &Array2<T>) -> Result<Array2<T>, AcceleratorError> {
        pairwise_cosine_serial(left, right)
    }
}

/// Note: current GPU backend path attempts GPU execution and falls back to CPU serial execution.
impl PairwiseCosineKernel<f32> for GpuBackend {
    fn pairwise_cosine(
        left: &Array2<f32>,
        right: &Array2<f32>,
    ) -> Result<Array2<f32>, AcceleratorError> {
        if !should_attempt_gpu_pairwise(left.nrows(), right.nrows(), left.ncols()) {
            return pairwise_cosine_serial(left, right);
        }
        fallback_or_cpu_gpu(pairwise_cosine_gpu_f32(left, right), || {
            pairwise_cosine_serial(left, right)
        })
    }
}

/// Note: current GPU backend path attempts GPU execution and falls back to CPU serial execution.
impl PairwiseCosineKernel<f64> for GpuBackend {
    fn pairwise_cosine(
        left: &Array2<f64>,
        right: &Array2<f64>,
    ) -> Result<Array2<f64>, AcceleratorError> {
        if !should_attempt_gpu_pairwise(left.nrows(), right.nrows(), left.ncols()) {
            return pairwise_cosine_serial(left, right);
        }
        fallback_or_cpu_gpu(pairwise_cosine_gpu_f64(left, right), || {
            pairwise_cosine_serial(left, right)
        })
    }
}

impl<T> TensorContractKernel<T> for CpuBackend
where
    T: Copy + Default + AddAssign + Mul<Output = T>,
{
    fn contract_axes(
        left: &ArrayD<T>,
        right: &ArrayD<T>,
        left_axis: usize,
        right_axis: usize,
    ) -> Result<ArrayD<T>, AcceleratorError> {
        tensor_contract_axes_serial(left, right, left_axis, right_axis)
    }
}

/// Note: current GPU backend path attempts GPU execution and falls back to CPU serial execution.
impl TensorContractKernel<f32> for GpuBackend {
    fn contract_axes(
        left: &ArrayD<f32>,
        right: &ArrayD<f32>,
        left_axis: usize,
        right_axis: usize,
    ) -> Result<ArrayD<f32>, AcceleratorError> {
        if !should_attempt_gpu_tensor_contract(left.len(), right.len()) {
            return tensor_contract_axes_serial(left, right, left_axis, right_axis);
        }
        fallback_or_cpu_gpu(
            tensor_contract_axes_gpu_f32(left, right, left_axis, right_axis),
            || tensor_contract_axes_serial(left, right, left_axis, right_axis),
        )
    }
}

/// Note: current GPU backend path attempts GPU execution and falls back to CPU serial execution.
impl TensorContractKernel<f64> for GpuBackend {
    fn contract_axes(
        left: &ArrayD<f64>,
        right: &ArrayD<f64>,
        left_axis: usize,
        right_axis: usize,
    ) -> Result<ArrayD<f64>, AcceleratorError> {
        if !should_attempt_gpu_tensor_contract(left.len(), right.len()) {
            return tensor_contract_axes_serial(left, right, left_axis, right_axis);
        }
        fallback_or_cpu_gpu(
            tensor_contract_axes_gpu_f64(left, right, left_axis, right_axis),
            || tensor_contract_axes_serial(left, right, left_axis, right_axis),
        )
    }
}

/// Note: current GPU backend path attempts GPU execution and falls back to CPU serial execution.
impl TensorContractKernel<num_complex::Complex64> for GpuBackend {
    fn contract_axes(
        left: &ArrayD<num_complex::Complex64>,
        right: &ArrayD<num_complex::Complex64>,
        left_axis: usize,
        right_axis: usize,
    ) -> Result<ArrayD<num_complex::Complex64>, AcceleratorError> {
        if !should_attempt_gpu_tensor_contract(left.len(), right.len()) {
            return tensor_contract_axes_serial(left, right, left_axis, right_axis);
        }
        fallback_or_cpu_gpu(
            tensor_contract_axes_gpu_complex64(left, right, left_axis, right_axis),
            || tensor_contract_axes_serial(left, right, left_axis, right_axis),
        )
    }
}

impl<T> TensorBatchedMatMulKernel<T> for CpuBackend
where
    T: Copy + Default + AddAssign + Mul<Output = T>,
{
    fn batched_matmul_last_two(
        left: &ArrayD<T>,
        right: &ArrayD<T>,
    ) -> Result<ArrayD<T>, AcceleratorError> {
        tensor_batched_matmul_last_two_serial(left, right)
    }
}

/// Note: current GPU backend path attempts GPU execution and falls back to CPU serial execution.
impl TensorBatchedMatMulKernel<f32> for GpuBackend {
    fn batched_matmul_last_two(
        left: &ArrayD<f32>,
        right: &ArrayD<f32>,
    ) -> Result<ArrayD<f32>, AcceleratorError> {
        if left.ndim() >= 2 && right.ndim() >= 2 {
            let left_shape = left.shape();
            let right_shape = right.shape();
            let batch = left_shape[..left.ndim() - 2].iter().copied().product::<usize>().max(1);
            let rows = left_shape[left.ndim() - 2];
            let inner = left_shape[left.ndim() - 1];
            let cols = right_shape[right.ndim() - 1];
            if !should_attempt_gpu_tensor_batched_matmul(batch, rows, inner, cols) {
                return tensor_batched_matmul_last_two_serial(left, right);
            }
        }
        fallback_or_cpu_gpu(tensor_batched_matmul_last_two_gpu_f32(left, right), || {
            tensor_batched_matmul_last_two_serial(left, right)
        })
    }
}

/// Note: current GPU backend path attempts GPU execution and falls back to CPU serial execution.
impl TensorBatchedMatMulKernel<f64> for GpuBackend {
    fn batched_matmul_last_two(
        left: &ArrayD<f64>,
        right: &ArrayD<f64>,
    ) -> Result<ArrayD<f64>, AcceleratorError> {
        if left.ndim() >= 2 && right.ndim() >= 2 {
            let left_shape = left.shape();
            let right_shape = right.shape();
            let batch = left_shape[..left.ndim() - 2].iter().copied().product::<usize>().max(1);
            let rows = left_shape[left.ndim() - 2];
            let inner = left_shape[left.ndim() - 1];
            let cols = right_shape[right.ndim() - 1];
            if !should_attempt_gpu_tensor_batched_matmul(batch, rows, inner, cols) {
                return tensor_batched_matmul_last_two_serial(left, right);
            }
        }
        fallback_or_cpu_gpu(tensor_batched_matmul_last_two_gpu_f64(left, right), || {
            tensor_batched_matmul_last_two_serial(left, right)
        })
    }
}

/// Note: current GPU backend path attempts GPU execution and falls back to CPU serial execution.
impl TensorBatchedMatMulKernel<num_complex::Complex64> for GpuBackend {
    fn batched_matmul_last_two(
        left: &ArrayD<num_complex::Complex64>,
        right: &ArrayD<num_complex::Complex64>,
    ) -> Result<ArrayD<num_complex::Complex64>, AcceleratorError> {
        if left.ndim() >= 2 && right.ndim() >= 2 {
            let left_shape = left.shape();
            let right_shape = right.shape();
            let batch = left_shape[..left.ndim() - 2].iter().copied().product::<usize>().max(1);
            let rows = left_shape[left.ndim() - 2];
            let inner = left_shape[left.ndim() - 1];
            let cols = right_shape[right.ndim() - 1];
            if !should_attempt_gpu_tensor_batched_matmul(batch, rows, inner, cols) {
                return tensor_batched_matmul_last_two_serial(left, right);
            }
        }
        fallback_or_cpu_gpu(tensor_batched_matmul_last_two_gpu_complex64(left, right), || {
            tensor_batched_matmul_last_two_serial(left, right)
        })
    }
}

impl<T> TensorLastAxisReductionKernel<T> for CpuBackend
where
    T: Copy + Default + AddAssign,
{
    fn sum_last_axis(input: &ArrayD<T>) -> Result<ArrayD<T>, AcceleratorError> {
        tensor_sum_last_axis_serial(input)
    }
}

/// Note: current GPU backend path attempts GPU execution and falls back to CPU serial execution.
impl TensorLastAxisReductionKernel<f32> for GpuBackend {
    fn sum_last_axis(input: &ArrayD<f32>) -> Result<ArrayD<f32>, AcceleratorError> {
        if !should_attempt_gpu_tensor_sum(input.len()) {
            return tensor_sum_last_axis_serial(input);
        }
        fallback_or_cpu_gpu(tensor_sum_last_axis_gpu_f32(input), || {
            tensor_sum_last_axis_serial(input)
        })
    }
}

/// Note: current GPU backend path attempts GPU execution and falls back to CPU serial execution.
impl TensorLastAxisReductionKernel<f64> for GpuBackend {
    fn sum_last_axis(input: &ArrayD<f64>) -> Result<ArrayD<f64>, AcceleratorError> {
        if !should_attempt_gpu_tensor_sum(input.len()) {
            return tensor_sum_last_axis_serial(input);
        }
        fallback_or_cpu_gpu(tensor_sum_last_axis_gpu_f64(input), || {
            tensor_sum_last_axis_serial(input)
        })
    }
}

/// Note: current GPU backend path attempts GPU execution and falls back to CPU serial execution.
impl TensorLastAxisReductionKernel<num_complex::Complex64> for GpuBackend {
    fn sum_last_axis(
        input: &ArrayD<num_complex::Complex64>,
    ) -> Result<ArrayD<num_complex::Complex64>, AcceleratorError> {
        if !should_attempt_gpu_tensor_sum(input.len()) {
            return tensor_sum_last_axis_serial(input);
        }
        fallback_or_cpu_gpu(tensor_sum_last_axis_gpu_complex64(input), || {
            tensor_sum_last_axis_serial(input)
        })
    }
}

/// Compute matrix-matrix product using compile-time backend dispatch.
///
/// # Errors
/// Returns an error for unsupported backends, invalid dimensions, or kernel errors.
pub fn matmat_with_backend<B, T>(
    left: &Array2<T>,
    right: &Array2<T>,
) -> Result<Array2<T>, AcceleratorError>
where
    B: MatMatKernel<T>,
{
    B::matmat(left, right)
}

/// Compute matrix-vector product using compile-time backend dispatch.
///
/// # Errors
/// Returns an error for unsupported backends, invalid dimensions, or kernel errors.
pub fn matvec_with_backend<B, T>(
    matrix: &Array2<T>,
    vector: &Array1<T>,
) -> Result<Array1<T>, AcceleratorError>
where
    B: MatVecKernel<T>,
{
    B::matvec(matrix, vector)
}

/// Compute batched matrix-matrix products using compile-time backend dispatch.
///
/// # Errors
/// Returns an error for unsupported backends, invalid dimensions, or kernel errors.
pub fn batched_matmat_with_backend<B, T>(
    left_batches: &Array3<T>,
    right_batches: &Array3<T>,
) -> Result<Array3<T>, AcceleratorError>
where
    B: BatchedMatMatKernel<T>,
{
    B::batched_matmat(left_batches, right_batches)
}

/// Compute sparse matrix-vector product using compile-time backend dispatch.
///
/// # Errors
/// Returns an error for unsupported backends, invalid dimensions, or kernel errors.
pub fn sparse_matvec_with_backend<B, T>(
    matrix: &CsrMatrix<T>,
    vector: &Array1<T>,
) -> Result<Array1<T>, AcceleratorError>
where
    T: NabledReal,
    B: SparseMatVecKernel<T>,
{
    B::sparse_matvec(matrix, vector)
}

/// Compute row-batch by matrix product using compile-time backend dispatch.
///
/// # Errors
/// Returns an error for unsupported backends, invalid dimensions, or kernel errors.
pub fn batched_row_matvec_with_backend<B, T>(
    batch_vectors: &Array2<T>,
    matrix: &Array2<T>,
) -> Result<Array2<T>, AcceleratorError>
where
    B: BatchedRowMatVecKernel<T>,
{
    B::batched_row_matvec(batch_vectors, matrix)
}

/// Compute sparse-dense matrix product using compile-time backend dispatch.
///
/// # Errors
/// Returns an error for unsupported backends, invalid dimensions, or kernel errors.
pub fn sparse_matmat_dense_with_backend<B, T>(
    matrix: &CsrMatrix<T>,
    dense: &Array2<T>,
) -> Result<Array2<T>, AcceleratorError>
where
    T: NabledReal,
    B: SparseMatMatDenseKernel<T>,
{
    B::sparse_matmat_dense(matrix, dense)
}

/// Compute sparse-sparse matrix product using compile-time backend dispatch.
///
/// # Errors
/// Returns an error for unsupported backends, invalid dimensions, or kernel errors.
pub fn sparse_matmat_sparse_with_backend<B, T>(
    left: &CsrMatrix<T>,
    right: &CsrMatrix<T>,
) -> Result<CsrMatrix<T>, AcceleratorError>
where
    T: NabledReal,
    B: SparseMatMatSparseKernel<T>,
{
    B::sparse_matmat_sparse(left, right)
}

/// Compute triangular solve against vector using compile-time backend dispatch.
///
/// # Errors
/// Returns an error for unsupported backends, invalid dimensions, or kernel errors.
pub fn triangular_solve_vec_with_backend<B, T>(
    matrix: &Array2<T>,
    rhs: &Array1<T>,
    lower: bool,
    unit_diagonal: bool,
) -> Result<Array1<T>, AcceleratorError>
where
    B: TriangularSolveVecKernel<T>,
{
    B::triangular_solve_vec(matrix, rhs, lower, unit_diagonal)
}

/// Compute triangular solve against matrix using compile-time backend dispatch.
///
/// # Errors
/// Returns an error for unsupported backends, invalid dimensions, or kernel errors.
pub fn triangular_solve_mat_with_backend<B, T>(
    matrix: &Array2<T>,
    rhs: &Array2<T>,
    lower: bool,
    unit_diagonal: bool,
) -> Result<Array2<T>, AcceleratorError>
where
    B: TriangularSolveMatKernel<T>,
{
    B::triangular_solve_mat(matrix, rhs, lower, unit_diagonal)
}

/// Compute vector dot-product using compile-time backend dispatch.
///
/// # Errors
/// Returns an error for unsupported backends, invalid dimensions, or kernel errors.
pub fn dot_with_backend<B, T>(left: &Array1<T>, right: &Array1<T>) -> Result<T, AcceleratorError>
where
    B: DotKernel<T>,
{
    B::dot(left, right)
}

/// Compute pairwise L2 distances using compile-time backend dispatch.
///
/// # Errors
/// Returns an error for unsupported backends, invalid dimensions, or kernel errors.
pub fn pairwise_l2_with_backend<B, T>(
    left: &Array2<T>,
    right: &Array2<T>,
) -> Result<Array2<T>, AcceleratorError>
where
    B: PairwiseL2Kernel<T>,
{
    B::pairwise_l2(left, right)
}

/// Compute pairwise cosine similarities using compile-time backend dispatch.
///
/// # Errors
/// Returns an error for unsupported backends, invalid dimensions, or kernel errors.
pub fn pairwise_cosine_with_backend<B, T>(
    left: &Array2<T>,
    right: &Array2<T>,
) -> Result<Array2<T>, AcceleratorError>
where
    B: PairwiseCosineKernel<T>,
{
    B::pairwise_cosine(left, right)
}

/// Compute tensor contraction using compile-time backend dispatch.
///
/// # Errors
/// Returns an error for unsupported backends, invalid dimensions, or kernel errors.
pub fn tensor_contract_axes_with_backend<B, T>(
    left: &ArrayD<T>,
    right: &ArrayD<T>,
    left_axis: usize,
    right_axis: usize,
) -> Result<ArrayD<T>, AcceleratorError>
where
    B: TensorContractKernel<T>,
{
    B::contract_axes(left, right, left_axis, right_axis)
}

/// Compute N-D batched matrix multiply over the last two axes using compile-time backend dispatch.
///
/// # Errors
/// Returns an error for unsupported backends, invalid dimensions, or kernel errors.
pub fn tensor_batched_matmul_last_two_with_backend<B, T>(
    left: &ArrayD<T>,
    right: &ArrayD<T>,
) -> Result<ArrayD<T>, AcceleratorError>
where
    B: TensorBatchedMatMulKernel<T>,
{
    B::batched_matmul_last_two(left, right)
}

/// Compute tensor reduction over the last axis using compile-time backend dispatch.
///
/// # Errors
/// Returns an error for unsupported backends, invalid dimensions, or kernel errors.
pub fn tensor_sum_last_axis_with_backend<B, T>(
    input: &ArrayD<T>,
) -> Result<ArrayD<T>, AcceleratorError>
where
    B: TensorLastAxisReductionKernel<T>,
{
    B::sum_last_axis(input)
}

/// Compute matrix-matrix product on the default CPU backend.
///
/// # Errors
/// Returns an error for invalid dimensions or kernel execution failures.
pub fn matmat_cpu<T>(left: &Array2<T>, right: &Array2<T>) -> Result<Array2<T>, AcceleratorError>
where
    CpuBackend: MatMatKernel<T>,
{
    matmat_with_backend::<CpuBackend, T>(left, right)
}

/// Compute matrix-vector product on the default CPU backend.
///
/// # Errors
/// Returns an error for invalid dimensions or kernel execution failures.
pub fn matvec_cpu<T>(matrix: &Array2<T>, vector: &Array1<T>) -> Result<Array1<T>, AcceleratorError>
where
    CpuBackend: MatVecKernel<T>,
{
    matvec_with_backend::<CpuBackend, T>(matrix, vector)
}

/// Compute batched matrix-matrix products on the default CPU backend.
///
/// # Errors
/// Returns an error for invalid dimensions or kernel execution failures.
pub fn batched_matmat_cpu<T>(
    left_batches: &Array3<T>,
    right_batches: &Array3<T>,
) -> Result<Array3<T>, AcceleratorError>
where
    CpuBackend: BatchedMatMatKernel<T>,
{
    batched_matmat_with_backend::<CpuBackend, T>(left_batches, right_batches)
}

/// Compute row-batch by matrix products on the default CPU backend.
///
/// # Errors
/// Returns an error for invalid dimensions or kernel execution failures.
pub fn batched_row_matvec_cpu<T>(
    batch_vectors: &Array2<T>,
    matrix: &Array2<T>,
) -> Result<Array2<T>, AcceleratorError>
where
    CpuBackend: BatchedRowMatVecKernel<T>,
{
    batched_row_matvec_with_backend::<CpuBackend, T>(batch_vectors, matrix)
}

/// Compute sparse matrix-vector product on the default CPU backend.
///
/// # Errors
/// Returns an error for invalid dimensions or kernel execution failures.
pub fn sparse_matvec_cpu<T>(
    matrix: &CsrMatrix<T>,
    vector: &Array1<T>,
) -> Result<Array1<T>, AcceleratorError>
where
    T: NabledReal,
    CpuBackend: SparseMatVecKernel<T>,
{
    sparse_matvec_with_backend::<CpuBackend, T>(matrix, vector)
}

/// Compute sparse-dense matrix product on the default CPU backend.
///
/// # Errors
/// Returns an error for invalid dimensions or kernel execution failures.
pub fn sparse_matmat_dense_cpu<T>(
    matrix: &CsrMatrix<T>,
    dense: &Array2<T>,
) -> Result<Array2<T>, AcceleratorError>
where
    T: NabledReal,
    CpuBackend: SparseMatMatDenseKernel<T>,
{
    sparse_matmat_dense_with_backend::<CpuBackend, T>(matrix, dense)
}

/// Compute sparse-sparse matrix product on the default CPU backend.
///
/// # Errors
/// Returns an error for invalid dimensions or kernel execution failures.
pub fn sparse_matmat_sparse_cpu<T>(
    left: &CsrMatrix<T>,
    right: &CsrMatrix<T>,
) -> Result<CsrMatrix<T>, AcceleratorError>
where
    T: NabledReal,
    CpuBackend: SparseMatMatSparseKernel<T>,
{
    sparse_matmat_sparse_with_backend::<CpuBackend, T>(left, right)
}

/// Solve a triangular system against a vector on the default CPU backend.
///
/// # Errors
/// Returns an error for invalid dimensions, singular matrices, or kernel failures.
pub fn triangular_solve_vec_cpu<T>(
    matrix: &Array2<T>,
    rhs: &Array1<T>,
    lower: bool,
    unit_diagonal: bool,
) -> Result<Array1<T>, AcceleratorError>
where
    CpuBackend: TriangularSolveVecKernel<T>,
{
    triangular_solve_vec_with_backend::<CpuBackend, T>(matrix, rhs, lower, unit_diagonal)
}

/// Solve a triangular system against a matrix on the default CPU backend.
///
/// # Errors
/// Returns an error for invalid dimensions, singular matrices, or kernel failures.
pub fn triangular_solve_mat_cpu<T>(
    matrix: &Array2<T>,
    rhs: &Array2<T>,
    lower: bool,
    unit_diagonal: bool,
) -> Result<Array2<T>, AcceleratorError>
where
    CpuBackend: TriangularSolveMatKernel<T>,
{
    triangular_solve_mat_with_backend::<CpuBackend, T>(matrix, rhs, lower, unit_diagonal)
}

/// Compute dot product on the default CPU backend.
///
/// # Errors
/// Returns an error for invalid dimensions or kernel execution failures.
pub fn dot_cpu<T>(left: &Array1<T>, right: &Array1<T>) -> Result<T, AcceleratorError>
where
    CpuBackend: DotKernel<T>,
{
    dot_with_backend::<CpuBackend, T>(left, right)
}

/// Compute pairwise L2 distances on the default CPU backend.
///
/// # Errors
/// Returns an error for invalid dimensions or kernel execution failures.
pub fn pairwise_l2_cpu<T>(
    left: &Array2<T>,
    right: &Array2<T>,
) -> Result<Array2<T>, AcceleratorError>
where
    CpuBackend: PairwiseL2Kernel<T>,
{
    pairwise_l2_with_backend::<CpuBackend, T>(left, right)
}

/// Compute pairwise cosine similarity on the default CPU backend.
///
/// # Errors
/// Returns an error for invalid dimensions or kernel execution failures.
pub fn pairwise_cosine_cpu<T>(
    left: &Array2<T>,
    right: &Array2<T>,
) -> Result<Array2<T>, AcceleratorError>
where
    CpuBackend: PairwiseCosineKernel<T>,
{
    pairwise_cosine_with_backend::<CpuBackend, T>(left, right)
}

/// Contract tensor axes on the default CPU backend.
///
/// # Errors
/// Returns an error for invalid dimensions or kernel execution failures.
pub fn tensor_contract_axes_cpu<T>(
    left: &ArrayD<T>,
    right: &ArrayD<T>,
    left_axis: usize,
    right_axis: usize,
) -> Result<ArrayD<T>, AcceleratorError>
where
    CpuBackend: TensorContractKernel<T>,
{
    tensor_contract_axes_with_backend::<CpuBackend, T>(left, right, left_axis, right_axis)
}

/// Compute N-D batched matrix multiplication over the last two axes on the default CPU backend.
///
/// # Errors
/// Returns an error for invalid dimensions or kernel execution failures.
pub fn tensor_batched_matmul_last_two_cpu<T>(
    left: &ArrayD<T>,
    right: &ArrayD<T>,
) -> Result<ArrayD<T>, AcceleratorError>
where
    CpuBackend: TensorBatchedMatMulKernel<T>,
{
    tensor_batched_matmul_last_two_with_backend::<CpuBackend, T>(left, right)
}

/// Reduce a tensor over the last axis on the default CPU backend.
///
/// # Errors
/// Returns an error for invalid dimensions or kernel execution failures.
pub fn tensor_sum_last_axis_cpu<T>(input: &ArrayD<T>) -> Result<ArrayD<T>, AcceleratorError>
where
    CpuBackend: TensorLastAxisReductionKernel<T>,
{
    tensor_sum_last_axis_with_backend::<CpuBackend, T>(input)
}
