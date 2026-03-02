use std::ops::{AddAssign, Mul};

use ndarray::{Array1, Array2, Array3, ArrayD};
use num_traits::Float;

use super::backends::{AcceleratorError, BackendKind, CpuBackend, CudaBackend, DistributedBackend};
use super::cpu::{
    batched_matmat_serial, batched_matmat_serial_f32, batched_row_matvec_serial, dot_serial,
    matmat_serial, matmat_serial_f32, matvec_serial, matvec_serial_f32, pairwise_cosine_serial,
    pairwise_l2_serial, sparse_matmat_dense_serial, sparse_matmat_sparse_serial,
    sparse_matvec_serial, tensor_batched_matmul_last_two_serial, tensor_contract_axes_serial,
    tensor_sum_last_axis_serial, triangular_solve_mat_serial, triangular_solve_vec_serial,
};
use super::distributed::{DistributedConfig, matmat_distributed};
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

impl MatMatKernel<f64> for CpuBackend {
    fn matmat(left: &Array2<f64>, right: &Array2<f64>) -> Result<Array2<f64>, AcceleratorError> {
        matmat_serial(left, right)
    }
}

impl MatMatKernel<f64> for DistributedBackend {
    fn matmat(left: &Array2<f64>, right: &Array2<f64>) -> Result<Array2<f64>, AcceleratorError> {
        matmat_distributed(left, right, DistributedConfig::default())
    }
}

impl MatMatKernel<f64> for CudaBackend {
    fn matmat(_left: &Array2<f64>, _right: &Array2<f64>) -> Result<Array2<f64>, AcceleratorError> {
        Err(AcceleratorError::UnsupportedBackend(BackendKind::Cuda))
    }
}

impl MatMatKernel<f32> for CpuBackend {
    fn matmat(left: &Array2<f32>, right: &Array2<f32>) -> Result<Array2<f32>, AcceleratorError> {
        matmat_serial_f32(left, right)
    }
}

impl MatMatKernel<f32> for DistributedBackend {
    fn matmat(left: &Array2<f32>, right: &Array2<f32>) -> Result<Array2<f32>, AcceleratorError> {
        matmat_serial_f32(left, right)
    }
}

impl MatMatKernel<f32> for CudaBackend {
    fn matmat(left: &Array2<f32>, right: &Array2<f32>) -> Result<Array2<f32>, AcceleratorError> {
        matmat_gpu_f32(left, right)
    }
}

impl MatVecKernel<f64> for CpuBackend {
    fn matvec(matrix: &Array2<f64>, vector: &Array1<f64>) -> Result<Array1<f64>, AcceleratorError> {
        matvec_serial(matrix, vector)
    }
}

impl MatVecKernel<f64> for DistributedBackend {
    fn matvec(matrix: &Array2<f64>, vector: &Array1<f64>) -> Result<Array1<f64>, AcceleratorError> {
        matvec_serial(matrix, vector)
    }
}

impl MatVecKernel<f64> for CudaBackend {
    fn matvec(
        _matrix: &Array2<f64>,
        _vector: &Array1<f64>,
    ) -> Result<Array1<f64>, AcceleratorError> {
        Err(AcceleratorError::UnsupportedBackend(BackendKind::Cuda))
    }
}

impl MatVecKernel<f32> for CpuBackend {
    fn matvec(matrix: &Array2<f32>, vector: &Array1<f32>) -> Result<Array1<f32>, AcceleratorError> {
        matvec_serial_f32(matrix, vector)
    }
}

impl MatVecKernel<f32> for DistributedBackend {
    fn matvec(matrix: &Array2<f32>, vector: &Array1<f32>) -> Result<Array1<f32>, AcceleratorError> {
        matvec_serial_f32(matrix, vector)
    }
}

impl MatVecKernel<f32> for CudaBackend {
    fn matvec(matrix: &Array2<f32>, vector: &Array1<f32>) -> Result<Array1<f32>, AcceleratorError> {
        matvec_gpu_f32(matrix, vector)
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

impl BatchedMatMatKernel<f64> for DistributedBackend {
    fn batched_matmat(
        left_batches: &Array3<f64>,
        right_batches: &Array3<f64>,
    ) -> Result<Array3<f64>, AcceleratorError> {
        batched_matmat_serial(left_batches, right_batches)
    }
}

impl BatchedMatMatKernel<f64> for CudaBackend {
    fn batched_matmat(
        _left_batches: &Array3<f64>,
        _right_batches: &Array3<f64>,
    ) -> Result<Array3<f64>, AcceleratorError> {
        Err(AcceleratorError::UnsupportedBackend(BackendKind::Cuda))
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

impl BatchedMatMatKernel<f32> for DistributedBackend {
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
        batched_matmat_gpu_f32(left_batches, right_batches)
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

impl SparseMatVecKernel for DistributedBackend {
    fn sparse_matvec(
        matrix: &CsrMatrix,
        vector: &Array1<f64>,
    ) -> Result<Array1<f64>, AcceleratorError> {
        sparse_matvec_serial(matrix, vector)
    }
}

impl SparseMatVecKernel for CudaBackend {
    fn sparse_matvec(
        _matrix: &CsrMatrix,
        _vector: &Array1<f64>,
    ) -> Result<Array1<f64>, AcceleratorError> {
        Err(AcceleratorError::UnsupportedBackend(BackendKind::Cuda))
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

impl BatchedRowMatVecKernel<f64> for DistributedBackend {
    fn batched_row_matvec(
        batch_vectors: &Array2<f64>,
        matrix: &Array2<f64>,
    ) -> Result<Array2<f64>, AcceleratorError> {
        batched_row_matvec_serial(batch_vectors, matrix)
    }
}

impl BatchedRowMatVecKernel<f64> for CudaBackend {
    fn batched_row_matvec(
        _batch_vectors: &Array2<f64>,
        _matrix: &Array2<f64>,
    ) -> Result<Array2<f64>, AcceleratorError> {
        Err(AcceleratorError::UnsupportedBackend(BackendKind::Cuda))
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

impl SparseMatMatDenseKernel for DistributedBackend {
    fn sparse_matmat_dense(
        matrix: &CsrMatrix,
        dense: &Array2<f64>,
    ) -> Result<Array2<f64>, AcceleratorError> {
        sparse_matmat_dense_serial(matrix, dense)
    }
}

impl SparseMatMatDenseKernel for CudaBackend {
    fn sparse_matmat_dense(
        _matrix: &CsrMatrix,
        _dense: &Array2<f64>,
    ) -> Result<Array2<f64>, AcceleratorError> {
        Err(AcceleratorError::UnsupportedBackend(BackendKind::Cuda))
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

impl SparseMatMatSparseKernel for DistributedBackend {
    fn sparse_matmat_sparse(
        left: &CsrMatrix,
        right: &CsrMatrix,
    ) -> Result<CsrMatrix, AcceleratorError> {
        sparse_matmat_sparse_serial(left, right)
    }
}

impl SparseMatMatSparseKernel for CudaBackend {
    fn sparse_matmat_sparse(
        _left: &CsrMatrix,
        _right: &CsrMatrix,
    ) -> Result<CsrMatrix, AcceleratorError> {
        Err(AcceleratorError::UnsupportedBackend(BackendKind::Cuda))
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

impl<T> TriangularSolveVecKernel<T> for DistributedBackend
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
        _matrix: &Array2<T>,
        _rhs: &Array1<T>,
        _lower: bool,
        _unit_diagonal: bool,
    ) -> Result<Array1<T>, AcceleratorError> {
        Err(AcceleratorError::UnsupportedBackend(BackendKind::Cuda))
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

impl<T> TriangularSolveMatKernel<T> for DistributedBackend
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
        _matrix: &Array2<T>,
        _rhs: &Array2<T>,
        _lower: bool,
        _unit_diagonal: bool,
    ) -> Result<Array2<T>, AcceleratorError> {
        Err(AcceleratorError::UnsupportedBackend(BackendKind::Cuda))
    }
}

impl DotKernel<f64> for CpuBackend {
    fn dot(left: &Array1<f64>, right: &Array1<f64>) -> Result<f64, AcceleratorError> {
        dot_serial(left, right)
    }
}

impl DotKernel<f64> for DistributedBackend {
    fn dot(left: &Array1<f64>, right: &Array1<f64>) -> Result<f64, AcceleratorError> {
        dot_serial(left, right)
    }
}

impl DotKernel<f64> for CudaBackend {
    fn dot(_left: &Array1<f64>, _right: &Array1<f64>) -> Result<f64, AcceleratorError> {
        Err(AcceleratorError::UnsupportedBackend(BackendKind::Cuda))
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

impl PairwiseL2Kernel for DistributedBackend {
    fn pairwise_l2(
        left: &Array2<f64>,
        right: &Array2<f64>,
    ) -> Result<Array2<f64>, AcceleratorError> {
        pairwise_l2_serial(left, right)
    }
}

impl PairwiseL2Kernel for CudaBackend {
    fn pairwise_l2(
        _left: &Array2<f64>,
        _right: &Array2<f64>,
    ) -> Result<Array2<f64>, AcceleratorError> {
        Err(AcceleratorError::UnsupportedBackend(BackendKind::Cuda))
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

impl PairwiseCosineKernel for DistributedBackend {
    fn pairwise_cosine(
        left: &Array2<f64>,
        right: &Array2<f64>,
    ) -> Result<Array2<f64>, AcceleratorError> {
        pairwise_cosine_serial(left, right)
    }
}

impl PairwiseCosineKernel for CudaBackend {
    fn pairwise_cosine(
        _left: &Array2<f64>,
        _right: &Array2<f64>,
    ) -> Result<Array2<f64>, AcceleratorError> {
        Err(AcceleratorError::UnsupportedBackend(BackendKind::Cuda))
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

impl<T> TensorContractKernel<T> for DistributedBackend
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
        _left: &ArrayD<T>,
        _right: &ArrayD<T>,
        _left_axis: usize,
        _right_axis: usize,
    ) -> Result<ArrayD<T>, AcceleratorError> {
        Err(AcceleratorError::UnsupportedBackend(BackendKind::Cuda))
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

impl<T> TensorBatchedMatMulKernel<T> for DistributedBackend
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
        tensor_batched_matmul_last_two_gpu_f32(left, right)
    }
}

impl TensorBatchedMatMulKernel<f64> for CudaBackend {
    fn batched_matmul_last_two(
        _left: &ArrayD<f64>,
        _right: &ArrayD<f64>,
    ) -> Result<ArrayD<f64>, AcceleratorError> {
        Err(AcceleratorError::UnsupportedBackend(BackendKind::Cuda))
    }
}

impl TensorBatchedMatMulKernel<num_complex::Complex64> for CudaBackend {
    fn batched_matmul_last_two(
        _left: &ArrayD<num_complex::Complex64>,
        _right: &ArrayD<num_complex::Complex64>,
    ) -> Result<ArrayD<num_complex::Complex64>, AcceleratorError> {
        Err(AcceleratorError::UnsupportedBackend(BackendKind::Cuda))
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

impl<T> TensorLastAxisReductionKernel<T> for DistributedBackend
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
    fn sum_last_axis(_input: &ArrayD<T>) -> Result<ArrayD<T>, AcceleratorError> {
        Err(AcceleratorError::UnsupportedBackend(BackendKind::Cuda))
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
