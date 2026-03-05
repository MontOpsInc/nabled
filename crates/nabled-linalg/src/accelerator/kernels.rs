use nabled_core::scalar::NabledReal;
use ndarray::{Array1, Array2, Array3, ArrayD};

use super::backends::{AcceleratorError, ComputeBackend};
use crate::sparse::CsrMatrix;

/// Kernel contract for matrix-matrix multiplication.
pub trait MatMatKernel<T>: ComputeBackend {
    /// Compute a matrix-matrix product for backend `Self`.
    ///
    /// # Errors
    /// Returns an error when dimensions are incompatible or backend execution fails.
    fn matmat(left: &Array2<T>, right: &Array2<T>) -> Result<Array2<T>, AcceleratorError>;
}

/// Kernel contract for matrix-vector multiplication.
pub trait MatVecKernel<T>: ComputeBackend {
    /// Compute a matrix-vector product for backend `Self`.
    ///
    /// # Errors
    /// Returns an error when dimensions are incompatible or backend execution fails.
    fn matvec(matrix: &Array2<T>, vector: &Array1<T>) -> Result<Array1<T>, AcceleratorError>;
}

/// Kernel contract for batched matrix-matrix multiplication.
pub trait BatchedMatMatKernel<T>: ComputeBackend {
    /// Compute batched matrix-matrix products for backend `Self`.
    ///
    /// # Errors
    /// Returns an error when dimensions are incompatible or backend execution fails.
    fn batched_matmat(
        left_batches: &Array3<T>,
        right_batches: &Array3<T>,
    ) -> Result<Array3<T>, AcceleratorError>;
}

/// Kernel contract for sparse matrix-vector multiplication.
pub trait SparseMatVecKernel<T: NabledReal>: ComputeBackend {
    /// Compute a sparse matrix-vector product for backend `Self`.
    ///
    /// # Errors
    /// Returns an error when dimensions are incompatible or backend execution fails.
    fn sparse_matvec(
        matrix: &CsrMatrix<T>,
        vector: &Array1<T>,
    ) -> Result<Array1<T>, AcceleratorError>;
}

/// Kernel contract for batched row-vector by matrix transforms.
pub trait BatchedRowMatVecKernel<T>: ComputeBackend {
    /// Apply one matrix to a batch of row vectors.
    ///
    /// # Errors
    /// Returns an error when dimensions are incompatible or backend execution fails.
    fn batched_row_matvec(
        batch_vectors: &Array2<T>,
        matrix: &Array2<T>,
    ) -> Result<Array2<T>, AcceleratorError>;
}

/// Kernel contract for sparse-dense matrix multiplication.
pub trait SparseMatMatDenseKernel<T: NabledReal>: ComputeBackend {
    /// Multiply sparse matrix with dense right-hand matrix.
    ///
    /// # Errors
    /// Returns an error when dimensions are incompatible or backend execution fails.
    fn sparse_matmat_dense(
        matrix: &CsrMatrix<T>,
        dense: &Array2<T>,
    ) -> Result<Array2<T>, AcceleratorError>;
}

/// Kernel contract for sparse-sparse matrix multiplication.
pub trait SparseMatMatSparseKernel<T: NabledReal>: ComputeBackend {
    /// Multiply sparse matrix with sparse matrix.
    ///
    /// # Errors
    /// Returns an error when dimensions are incompatible or backend execution fails.
    fn sparse_matmat_sparse(
        left: &CsrMatrix<T>,
        right: &CsrMatrix<T>,
    ) -> Result<CsrMatrix<T>, AcceleratorError>;
}

/// Kernel contract for dense triangular solves with vector right-hand side.
pub trait TriangularSolveVecKernel<T>: ComputeBackend {
    /// Solve triangular system `A x = b`.
    ///
    /// # Errors
    /// Returns an error when dimensions are incompatible, matrix is singular, or backend
    /// execution fails.
    fn triangular_solve_vec(
        matrix: &Array2<T>,
        rhs: &Array1<T>,
        lower: bool,
        unit_diagonal: bool,
    ) -> Result<Array1<T>, AcceleratorError>;
}

/// Kernel contract for dense triangular solves with matrix right-hand side.
pub trait TriangularSolveMatKernel<T>: ComputeBackend {
    /// Solve triangular system `A X = B`.
    ///
    /// # Errors
    /// Returns an error when dimensions are incompatible, matrix is singular, or backend
    /// execution fails.
    fn triangular_solve_mat(
        matrix: &Array2<T>,
        rhs: &Array2<T>,
        lower: bool,
        unit_diagonal: bool,
    ) -> Result<Array2<T>, AcceleratorError>;
}

/// Kernel contract for vector dot products.
pub trait DotKernel<T>: ComputeBackend {
    /// Compute the dot product of two vectors.
    ///
    /// # Errors
    /// Returns an error when dimensions are incompatible or backend execution fails.
    fn dot(left: &Array1<T>, right: &Array1<T>) -> Result<T, AcceleratorError>;
}

/// Kernel contract for pairwise L2 distance matrix computation.
pub trait PairwiseL2Kernel<T>: ComputeBackend {
    /// Compute pairwise L2 distance matrix between row vectors.
    ///
    /// # Errors
    /// Returns an error when dimensions are incompatible or backend execution fails.
    fn pairwise_l2(left: &Array2<T>, right: &Array2<T>) -> Result<Array2<T>, AcceleratorError>;
}

/// Kernel contract for pairwise cosine similarity matrix computation.
pub trait PairwiseCosineKernel<T>: ComputeBackend {
    /// Compute pairwise cosine similarity matrix between row vectors.
    ///
    /// # Errors
    /// Returns an error when dimensions are incompatible or backend execution fails.
    fn pairwise_cosine(left: &Array2<T>, right: &Array2<T>) -> Result<Array2<T>, AcceleratorError>;
}

/// Kernel contract for tensor axis contraction.
pub trait TensorContractKernel<T>: ComputeBackend {
    /// Contract two tensors along explicit axes.
    ///
    /// # Errors
    /// Returns an error when dimensions are incompatible or backend execution fails.
    fn contract_axes(
        left: &ArrayD<T>,
        right: &ArrayD<T>,
        left_axis: usize,
        right_axis: usize,
    ) -> Result<ArrayD<T>, AcceleratorError>;
}

/// Kernel contract for N-D batched matrix multiplication over the last two axes.
pub trait TensorBatchedMatMulKernel<T>: ComputeBackend {
    /// Compute batched matrix multiplication over the last two axes.
    ///
    /// # Errors
    /// Returns an error when dimensions are incompatible or backend execution fails.
    fn batched_matmul_last_two(
        left: &ArrayD<T>,
        right: &ArrayD<T>,
    ) -> Result<ArrayD<T>, AcceleratorError>;
}

/// Kernel contract for tensor reductions over the last axis.
pub trait TensorLastAxisReductionKernel<T>: ComputeBackend {
    /// Sum over the last axis.
    ///
    /// # Errors
    /// Returns an error when dimensions are incompatible or backend execution fails.
    fn sum_last_axis(input: &ArrayD<T>) -> Result<ArrayD<T>, AcceleratorError>;
}
