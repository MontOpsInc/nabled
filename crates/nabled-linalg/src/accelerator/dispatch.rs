use std::ops::{AddAssign, Mul};

use ndarray::{Array1, Array2, Array3, ArrayD};
use num_traits::Float;

use super::backends::{AcceleratorError, BackendKind, CpuBackend, CudaBackend};
use super::cpu::{
    batched_matmat_serial, batched_matmat_serial_f32, batched_row_matvec_serial, dot_serial,
    matmat_serial, matmat_serial_f32, matvec_serial, matvec_serial_f32, pairwise_cosine_serial,
    pairwise_l2_serial, sparse_matmat_dense_serial, sparse_matmat_sparse_serial,
    sparse_matvec_serial, tensor_batched_matmul_last_two_serial, tensor_contract_axes_serial,
    tensor_sum_last_axis_serial, triangular_solve_mat_serial, triangular_solve_vec_serial,
};
use super::gpu::{
    batched_matmat_gpu_f32, matmat_gpu_f32, matvec_gpu_f32, tensor_batched_matmul_last_two_gpu_f32,
};
use super::kernels::{
    BatchedMatMatKernel, BatchedRowMatVecKernel, DotKernel, MatMatKernel, MatVecKernel,
    PairwiseCosineKernel, PairwiseL2Kernel, SparseMatMatDenseKernel, SparseMatMatSparseKernel,
    SparseMatVecKernel, TensorBatchedMatMulKernel, TensorContractKernel,
    TensorLastAxisReductionKernel, TriangularSolveMatKernel, TriangularSolveVecKernel,
};
use crate::sparse::CsrMatrix;

#[inline]
fn fallback_or_cpu_cuda<T, F>(
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
            | AcceleratorError::KernelExecutionFailed
            | AcceleratorError::UnsupportedBackend(BackendKind::Cuda),
        ) => fallback(),
        Err(error) => Err(error),
    }
}

impl MatMatKernel<f64> for CpuBackend {
    fn matmat(left: &Array2<f64>, right: &Array2<f64>) -> Result<Array2<f64>, AcceleratorError> {
        matmat_serial(left, right)
    }
}

impl MatMatKernel<f64> for CudaBackend {
    fn matmat(left: &Array2<f64>, right: &Array2<f64>) -> Result<Array2<f64>, AcceleratorError> {
        matmat_serial(left, right)
    }
}

impl MatMatKernel<f32> for CpuBackend {
    fn matmat(left: &Array2<f32>, right: &Array2<f32>) -> Result<Array2<f32>, AcceleratorError> {
        matmat_serial_f32(left, right)
    }
}

impl MatMatKernel<f32> for CudaBackend {
    fn matmat(left: &Array2<f32>, right: &Array2<f32>) -> Result<Array2<f32>, AcceleratorError> {
        fallback_or_cpu_cuda(matmat_gpu_f32(left, right), || matmat_serial_f32(left, right))
    }
}

impl MatVecKernel<f64> for CpuBackend {
    fn matvec(matrix: &Array2<f64>, vector: &Array1<f64>) -> Result<Array1<f64>, AcceleratorError> {
        matvec_serial(matrix, vector)
    }
}

impl MatVecKernel<f64> for CudaBackend {
    fn matvec(matrix: &Array2<f64>, vector: &Array1<f64>) -> Result<Array1<f64>, AcceleratorError> {
        matvec_serial(matrix, vector)
    }
}

impl MatVecKernel<f32> for CpuBackend {
    fn matvec(matrix: &Array2<f32>, vector: &Array1<f32>) -> Result<Array1<f32>, AcceleratorError> {
        matvec_serial_f32(matrix, vector)
    }
}

impl MatVecKernel<f32> for CudaBackend {
    fn matvec(matrix: &Array2<f32>, vector: &Array1<f32>) -> Result<Array1<f32>, AcceleratorError> {
        fallback_or_cpu_cuda(matvec_gpu_f32(matrix, vector), || matvec_serial_f32(matrix, vector))
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

impl BatchedMatMatKernel<f64> for CudaBackend {
    fn batched_matmat(
        left_batches: &Array3<f64>,
        right_batches: &Array3<f64>,
    ) -> Result<Array3<f64>, AcceleratorError> {
        batched_matmat_serial(left_batches, right_batches)
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

impl BatchedMatMatKernel<f32> for CudaBackend {
    fn batched_matmat(
        left_batches: &Array3<f32>,
        right_batches: &Array3<f32>,
    ) -> Result<Array3<f32>, AcceleratorError> {
        fallback_or_cpu_cuda(batched_matmat_gpu_f32(left_batches, right_batches), || {
            batched_matmat_serial_f32(left_batches, right_batches)
        })
    }
}

impl SparseMatVecKernel for CpuBackend {
    fn sparse_matvec(
        matrix: &CsrMatrix,
        vector: &Array1<f64>,
    ) -> Result<Array1<f64>, AcceleratorError> {
        sparse_matvec_serial(matrix, vector)
    }
}

impl SparseMatVecKernel for CudaBackend {
    fn sparse_matvec(
        matrix: &CsrMatrix,
        vector: &Array1<f64>,
    ) -> Result<Array1<f64>, AcceleratorError> {
        sparse_matvec_serial(matrix, vector)
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

impl BatchedRowMatVecKernel<f64> for CudaBackend {
    fn batched_row_matvec(
        batch_vectors: &Array2<f64>,
        matrix: &Array2<f64>,
    ) -> Result<Array2<f64>, AcceleratorError> {
        batched_row_matvec_serial(batch_vectors, matrix)
    }
}

impl SparseMatMatDenseKernel for CpuBackend {
    fn sparse_matmat_dense(
        matrix: &CsrMatrix,
        dense: &Array2<f64>,
    ) -> Result<Array2<f64>, AcceleratorError> {
        sparse_matmat_dense_serial(matrix, dense)
    }
}

impl SparseMatMatDenseKernel for CudaBackend {
    fn sparse_matmat_dense(
        matrix: &CsrMatrix,
        dense: &Array2<f64>,
    ) -> Result<Array2<f64>, AcceleratorError> {
        sparse_matmat_dense_serial(matrix, dense)
    }
}

impl SparseMatMatSparseKernel for CpuBackend {
    fn sparse_matmat_sparse(
        left: &CsrMatrix,
        right: &CsrMatrix,
    ) -> Result<CsrMatrix, AcceleratorError> {
        sparse_matmat_sparse_serial(left, right)
    }
}

impl SparseMatMatSparseKernel for CudaBackend {
    fn sparse_matmat_sparse(
        left: &CsrMatrix,
        right: &CsrMatrix,
    ) -> Result<CsrMatrix, AcceleratorError> {
        sparse_matmat_sparse_serial(left, right)
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

impl<T> TriangularSolveVecKernel<T> for CudaBackend
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

impl<T> TriangularSolveMatKernel<T> for CudaBackend
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

impl DotKernel<f64> for CpuBackend {
    fn dot(left: &Array1<f64>, right: &Array1<f64>) -> Result<f64, AcceleratorError> {
        dot_serial(left, right)
    }
}

impl DotKernel<f64> for CudaBackend {
    fn dot(left: &Array1<f64>, right: &Array1<f64>) -> Result<f64, AcceleratorError> {
        dot_serial(left, right)
    }
}

impl PairwiseL2Kernel for CpuBackend {
    fn pairwise_l2(
        left: &Array2<f64>,
        right: &Array2<f64>,
    ) -> Result<Array2<f64>, AcceleratorError> {
        pairwise_l2_serial(left, right)
    }
}

impl PairwiseL2Kernel for CudaBackend {
    fn pairwise_l2(
        left: &Array2<f64>,
        right: &Array2<f64>,
    ) -> Result<Array2<f64>, AcceleratorError> {
        pairwise_l2_serial(left, right)
    }
}

impl PairwiseCosineKernel for CpuBackend {
    fn pairwise_cosine(
        left: &Array2<f64>,
        right: &Array2<f64>,
    ) -> Result<Array2<f64>, AcceleratorError> {
        pairwise_cosine_serial(left, right)
    }
}

impl PairwiseCosineKernel for CudaBackend {
    fn pairwise_cosine(
        left: &Array2<f64>,
        right: &Array2<f64>,
    ) -> Result<Array2<f64>, AcceleratorError> {
        pairwise_cosine_serial(left, right)
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

impl<T> TensorContractKernel<T> for CudaBackend
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

impl TensorBatchedMatMulKernel<f32> for CudaBackend {
    fn batched_matmul_last_two(
        left: &ArrayD<f32>,
        right: &ArrayD<f32>,
    ) -> Result<ArrayD<f32>, AcceleratorError> {
        fallback_or_cpu_cuda(tensor_batched_matmul_last_two_gpu_f32(left, right), || {
            tensor_batched_matmul_last_two_serial(left, right)
        })
    }
}

impl TensorBatchedMatMulKernel<f64> for CudaBackend {
    fn batched_matmul_last_two(
        left: &ArrayD<f64>,
        right: &ArrayD<f64>,
    ) -> Result<ArrayD<f64>, AcceleratorError> {
        tensor_batched_matmul_last_two_serial(left, right)
    }
}

impl TensorBatchedMatMulKernel<num_complex::Complex64> for CudaBackend {
    fn batched_matmul_last_two(
        left: &ArrayD<num_complex::Complex64>,
        right: &ArrayD<num_complex::Complex64>,
    ) -> Result<ArrayD<num_complex::Complex64>, AcceleratorError> {
        tensor_batched_matmul_last_two_serial(left, right)
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

impl<T> TensorLastAxisReductionKernel<T> for CudaBackend
where
    T: Copy + Default + AddAssign,
{
    fn sum_last_axis(input: &ArrayD<T>) -> Result<ArrayD<T>, AcceleratorError> {
        tensor_sum_last_axis_serial(input)
    }
}

/// Compute matrix-matrix product using compile-time backend dispatch.
///
/// # Errors
/// Returns an error for unsupported backends, invalid dimensions, or kernel errors.
pub fn matmat_with_backend<B>(
    left: &Array2<f64>,
    right: &Array2<f64>,
) -> Result<Array2<f64>, AcceleratorError>
where
    B: MatMatKernel<f64>,
{
    B::matmat(left, right)
}

/// Compute matrix-matrix product using compile-time backend dispatch for `f32`.
///
/// # Errors
/// Returns an error for unsupported backends, invalid dimensions, or kernel errors.
pub fn matmat_with_backend_f32<B>(
    left: &Array2<f32>,
    right: &Array2<f32>,
) -> Result<Array2<f32>, AcceleratorError>
where
    B: MatMatKernel<f32>,
{
    B::matmat(left, right)
}

/// Compute matrix-vector product using compile-time backend dispatch.
///
/// # Errors
/// Returns an error for unsupported backends, invalid dimensions, or kernel errors.
pub fn matvec_with_backend<B>(
    matrix: &Array2<f64>,
    vector: &Array1<f64>,
) -> Result<Array1<f64>, AcceleratorError>
where
    B: MatVecKernel<f64>,
{
    B::matvec(matrix, vector)
}

/// Compute matrix-vector product using compile-time backend dispatch for `f32`.
///
/// # Errors
/// Returns an error for unsupported backends, invalid dimensions, or kernel errors.
pub fn matvec_with_backend_f32<B>(
    matrix: &Array2<f32>,
    vector: &Array1<f32>,
) -> Result<Array1<f32>, AcceleratorError>
where
    B: MatVecKernel<f32>,
{
    B::matvec(matrix, vector)
}

/// Compute batched matrix-matrix products using compile-time backend dispatch.
///
/// # Errors
/// Returns an error for unsupported backends, invalid dimensions, or kernel errors.
pub fn batched_matmat_with_backend<B>(
    left_batches: &Array3<f64>,
    right_batches: &Array3<f64>,
) -> Result<Array3<f64>, AcceleratorError>
where
    B: BatchedMatMatKernel<f64>,
{
    B::batched_matmat(left_batches, right_batches)
}

/// Compute batched matrix-matrix products using compile-time backend dispatch for `f32`.
///
/// # Errors
/// Returns an error for unsupported backends, invalid dimensions, or kernel errors.
pub fn batched_matmat_with_backend_f32<B>(
    left_batches: &Array3<f32>,
    right_batches: &Array3<f32>,
) -> Result<Array3<f32>, AcceleratorError>
where
    B: BatchedMatMatKernel<f32>,
{
    B::batched_matmat(left_batches, right_batches)
}

/// Compute sparse matrix-vector product using compile-time backend dispatch.
///
/// # Errors
/// Returns an error for unsupported backends, invalid dimensions, or kernel errors.
pub fn sparse_matvec_with_backend<B>(
    matrix: &CsrMatrix,
    vector: &Array1<f64>,
) -> Result<Array1<f64>, AcceleratorError>
where
    B: SparseMatVecKernel,
{
    B::sparse_matvec(matrix, vector)
}

/// Compute row-batch by matrix product using compile-time backend dispatch.
///
/// # Errors
/// Returns an error for unsupported backends, invalid dimensions, or kernel errors.
pub fn batched_row_matvec_with_backend<B>(
    batch_vectors: &Array2<f64>,
    matrix: &Array2<f64>,
) -> Result<Array2<f64>, AcceleratorError>
where
    B: BatchedRowMatVecKernel<f64>,
{
    B::batched_row_matvec(batch_vectors, matrix)
}

/// Compute sparse-dense matrix product using compile-time backend dispatch.
///
/// # Errors
/// Returns an error for unsupported backends, invalid dimensions, or kernel errors.
pub fn sparse_matmat_dense_with_backend<B>(
    matrix: &CsrMatrix,
    dense: &Array2<f64>,
) -> Result<Array2<f64>, AcceleratorError>
where
    B: SparseMatMatDenseKernel,
{
    B::sparse_matmat_dense(matrix, dense)
}

/// Compute sparse-sparse matrix product using compile-time backend dispatch.
///
/// # Errors
/// Returns an error for unsupported backends, invalid dimensions, or kernel errors.
pub fn sparse_matmat_sparse_with_backend<B>(
    left: &CsrMatrix,
    right: &CsrMatrix,
) -> Result<CsrMatrix, AcceleratorError>
where
    B: SparseMatMatSparseKernel,
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
pub fn dot_with_backend<B>(left: &Array1<f64>, right: &Array1<f64>) -> Result<f64, AcceleratorError>
where
    B: DotKernel<f64>,
{
    B::dot(left, right)
}

/// Compute pairwise L2 distances using compile-time backend dispatch.
///
/// # Errors
/// Returns an error for unsupported backends, invalid dimensions, or kernel errors.
pub fn pairwise_l2_with_backend<B>(
    left: &Array2<f64>,
    right: &Array2<f64>,
) -> Result<Array2<f64>, AcceleratorError>
where
    B: PairwiseL2Kernel,
{
    B::pairwise_l2(left, right)
}

/// Compute pairwise cosine similarities using compile-time backend dispatch.
///
/// # Errors
/// Returns an error for unsupported backends, invalid dimensions, or kernel errors.
pub fn pairwise_cosine_with_backend<B>(
    left: &Array2<f64>,
    right: &Array2<f64>,
) -> Result<Array2<f64>, AcceleratorError>
where
    B: PairwiseCosineKernel,
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
pub fn matmat_cpu(
    left: &Array2<f64>,
    right: &Array2<f64>,
) -> Result<Array2<f64>, AcceleratorError> {
    matmat_with_backend::<CpuBackend>(left, right)
}

/// Compute matrix-vector product on the default CPU backend.
///
/// # Errors
/// Returns an error for invalid dimensions or kernel execution failures.
pub fn matvec_cpu(
    matrix: &Array2<f64>,
    vector: &Array1<f64>,
) -> Result<Array1<f64>, AcceleratorError> {
    matvec_with_backend::<CpuBackend>(matrix, vector)
}

/// Compute batched matrix-matrix products on the default CPU backend.
///
/// # Errors
/// Returns an error for invalid dimensions or kernel execution failures.
pub fn batched_matmat_cpu(
    left_batches: &Array3<f64>,
    right_batches: &Array3<f64>,
) -> Result<Array3<f64>, AcceleratorError> {
    batched_matmat_with_backend::<CpuBackend>(left_batches, right_batches)
}

/// Compute row-batch by matrix products on the default CPU backend.
///
/// # Errors
/// Returns an error for invalid dimensions or kernel execution failures.
pub fn batched_row_matvec_cpu(
    batch_vectors: &Array2<f64>,
    matrix: &Array2<f64>,
) -> Result<Array2<f64>, AcceleratorError> {
    batched_row_matvec_with_backend::<CpuBackend>(batch_vectors, matrix)
}

/// Compute sparse matrix-vector product on the default CPU backend.
///
/// # Errors
/// Returns an error for invalid dimensions or kernel execution failures.
pub fn sparse_matvec_cpu(
    matrix: &CsrMatrix,
    vector: &Array1<f64>,
) -> Result<Array1<f64>, AcceleratorError> {
    sparse_matvec_with_backend::<CpuBackend>(matrix, vector)
}

/// Compute sparse-dense matrix product on the default CPU backend.
///
/// # Errors
/// Returns an error for invalid dimensions or kernel execution failures.
pub fn sparse_matmat_dense_cpu(
    matrix: &CsrMatrix,
    dense: &Array2<f64>,
) -> Result<Array2<f64>, AcceleratorError> {
    sparse_matmat_dense_with_backend::<CpuBackend>(matrix, dense)
}

/// Compute sparse-sparse matrix product on the default CPU backend.
///
/// # Errors
/// Returns an error for invalid dimensions or kernel execution failures.
pub fn sparse_matmat_sparse_cpu(
    left: &CsrMatrix,
    right: &CsrMatrix,
) -> Result<CsrMatrix, AcceleratorError> {
    sparse_matmat_sparse_with_backend::<CpuBackend>(left, right)
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
pub fn dot_cpu(left: &Array1<f64>, right: &Array1<f64>) -> Result<f64, AcceleratorError> {
    dot_with_backend::<CpuBackend>(left, right)
}

/// Compute pairwise L2 distances on the default CPU backend.
///
/// # Errors
/// Returns an error for invalid dimensions or kernel execution failures.
pub fn pairwise_l2_cpu(
    left: &Array2<f64>,
    right: &Array2<f64>,
) -> Result<Array2<f64>, AcceleratorError> {
    pairwise_l2_with_backend::<CpuBackend>(left, right)
}

/// Compute pairwise cosine similarity on the default CPU backend.
///
/// # Errors
/// Returns an error for invalid dimensions or kernel execution failures.
pub fn pairwise_cosine_cpu(
    left: &Array2<f64>,
    right: &Array2<f64>,
) -> Result<Array2<f64>, AcceleratorError> {
    pairwise_cosine_with_backend::<CpuBackend>(left, right)
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
