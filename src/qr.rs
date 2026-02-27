//! # QR Decomposition Module
//!
//! This module provides QR decomposition implementations for both `nalgebra` and `ndarray`.
//! QR decomposition factors a matrix A into A = QR where Q is orthogonal and R is upper triangular.
//!
//! ## Features
//!
//! - **Full QR decomposition** using Householder reflections
//! - **Reduced QR (economy size)** for rectangular matrices
//! - **QR with column pivoting** for numerical stability
//! - **Least squares solving** using QR decomposition
//! - **Support for both nalgebra and ndarray**
//! - **Comprehensive error handling**
//! - **Numerical stability considerations**

use std::fmt;

use nalgebra::{DMatrix, RealField};
use num_traits::Float;
use num_traits::float::FloatCore;

/// Error types for QR decomposition
#[derive(Debug, Clone)]
pub enum QRError {
    /// Matrix is empty
    EmptyMatrix,
    /// Matrix is singular or rank-deficient
    SingularMatrix,
    /// Convergence failed for iterative methods
    ConvergenceFailed,
    /// Invalid input dimensions
    InvalidDimensions(String),
    /// Numerical instability detected
    NumericalInstability,
    /// Invalid input parameters
    InvalidInput(String),
}

impl fmt::Display for QRError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QRError::EmptyMatrix => write!(f, "Matrix is empty"),
            QRError::SingularMatrix => write!(f, "Matrix is singular or rank-deficient"),
            QRError::ConvergenceFailed => write!(f, "Convergence failed"),
            QRError::InvalidDimensions(msg) => write!(f, "Invalid dimensions: {msg}"),
            QRError::NumericalInstability => write!(f, "Numerical instability detected"),
            QRError::InvalidInput(msg) => write!(f, "Invalid input: {msg}"),
        }
    }
}

impl std::error::Error for QRError {}

/// Configuration for QR decomposition
#[derive(Debug, Clone)]
pub struct QRConfig<T> {
    /// Tolerance for rank determination
    pub rank_tolerance: T,
    /// Maximum number of iterations for iterative methods
    pub max_iterations: usize,
    /// Enable column pivoting for numerical stability
    pub use_pivoting:   bool,
}

impl<T: Float> Default for QRConfig<T> {
    fn default() -> Self {
        Self { rank_tolerance: T::from(1e-12).unwrap(), max_iterations: 100, use_pivoting: false }
    }
}

/// Result of QR decomposition
#[derive(Debug, Clone)]
pub struct QRResult<T> {
    /// Orthogonal matrix Q
    pub q:    DMatrix<T>,
    /// Upper triangular matrix R
    pub r:    DMatrix<T>,
    /// Column permutation matrix (if pivoting was used)
    pub p:    Option<DMatrix<T>>,
    /// Matrix rank
    pub rank: usize,
}

/// Nalgebra QR decomposition functions
pub mod nalgebra_qr {
    use nalgebra::{DMatrix, DVector, RealField};

    use super::*;

    /// Compute QR decomposition using nalgebra's built-in implementation
    ///
    /// Note: Currently uses nalgebra's QR decomposition. A custom Householder
    /// reflection implementation is planned for future versions.
    ///
    /// # Arguments
    /// * `matrix` - Input matrix to decompose
    /// * `config` - Configuration for the decomposition
    ///
    /// # Returns
    /// * `Result<QRResult<T>, QRError>` - QR decomposition result
    ///
    /// # Example
    /// ```rust
    /// use nabled::qr::nalgebra_qr;
    /// use nalgebra::DMatrix;
    ///
    /// let matrix = DMatrix::from_row_slice(3, 3, &[
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    ///     7.0, 8.0, 9.0
    /// ]);
    ///
    /// let qr = nalgebra_qr::compute_qr(&matrix, &Default::default())?;
    /// println!("Q: {}", qr.q);
    /// println!("R: {}", qr.r);
    /// # Ok::<(), nabled::qr::QRError>(())
    /// ```
    /// # Panics
    /// Panics if internal numeric assumptions are violated during setup or
    /// intermediate conversion steps.
    ///
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn compute_qr<T>(matrix: &DMatrix<T>, config: &QRConfig<T>) -> Result<QRResult<T>, QRError>
    where
        T: RealField + FloatCore + Float,
    {
        crate::backend::qr::compute_nalgebra_qr(matrix, config)
    }

    /// Compute reduced QR decomposition (economy size)
    ///
    /// For an m×n matrix with m ≥ n, returns Q as m×n and R as n×n
    ///
    /// # Arguments
    /// * `matrix` - Input matrix to decompose
    /// * `config` - Configuration for the decomposition
    ///
    /// # Returns
    /// * `Result<QRResult<T>, QRError>` - Reduced QR decomposition result
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn compute_reduced_qr<T>(
        matrix: &DMatrix<T>,
        config: &QRConfig<T>,
    ) -> Result<QRResult<T>, QRError>
    where
        T: RealField + FloatCore + Float,
    {
        crate::backend::qr::compute_nalgebra_reduced_qr(matrix, config)
    }

    /// Compute QR decomposition with column pivoting using nalgebra's built-in implementation
    ///
    /// Note: Currently uses nalgebra's QR decomposition. A custom Householder
    /// reflection implementation with pivoting is planned for future versions.
    ///
    /// # Arguments
    /// * `matrix` - Input matrix to decompose
    /// * `config` - Configuration for the decomposition
    ///
    /// # Returns
    /// * `Result<QRResult<T>, QRError>` - QR decomposition with pivoting result
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn compute_qr_with_pivoting<T>(
        matrix: &DMatrix<T>,
        config: &QRConfig<T>,
    ) -> Result<QRResult<T>, QRError>
    where
        T: RealField + FloatCore + Float,
    {
        crate::backend::qr::compute_nalgebra_qr_with_pivoting(matrix, config)
    }

    /// Solve least squares problem using QR decomposition
    ///
    /// Solves min ||Ax - b||₂ using QR decomposition
    ///
    /// # Arguments
    /// * `matrix` - Coefficient matrix A
    /// * `rhs` - Right-hand side vector b
    /// * `config` - Configuration for the decomposition
    ///
    /// # Returns
    /// * `Result<DVector<T>, QRError>` - Solution vector x
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn solve_least_squares<T>(
        matrix: &DMatrix<T>,
        rhs: &DVector<T>,
        config: &QRConfig<T>,
    ) -> Result<DVector<T>, QRError>
    where
        T: RealField + FloatCore + Float,
    {
        crate::backend::qr::solve_nalgebra_least_squares(matrix, rhs, config)
    }

    /// Reconstruct original matrix from QR decomposition
    ///
    /// # Arguments
    /// * `qr` - QR decomposition result
    ///
    /// # Returns
    /// * `DMatrix<T>` - Reconstructed matrix
    #[must_use]
    pub fn reconstruct_matrix<T>(qr: &QRResult<T>) -> DMatrix<T>
    where
        T: RealField,
    {
        match &qr.p {
            Some(p) => &qr.q * &qr.r * p.transpose(),
            None => &qr.q * &qr.r,
        }
    }

    /// Compute condition number from QR decomposition
    ///
    /// # Arguments
    /// * `qr` - QR decomposition result
    ///
    /// # Returns
    /// * `T` - Condition number
    #[must_use]
    pub fn condition_number<T>(qr: &QRResult<T>) -> T
    where
        T: RealField + FloatCore + Float,
    {
        let r = &qr.r;
        let (m, n) = r.shape();
        let min_dim = m.min(n);

        let mut max_diag = T::zero();
        let mut min_diag = Float::infinity();

        for i in 0..min_dim {
            let diag = FloatCore::abs(r[(i, i)]);
            if diag > max_diag {
                max_diag = diag;
            }
            if diag < min_diag && diag > T::zero() {
                min_diag = diag;
            }
        }

        if min_diag == Float::infinity() { Float::infinity() } else { max_diag / min_diag }
    }
}

/// Ndarray QR decomposition functions
pub mod ndarray_qr {
    use ndarray::{Array1, Array2};

    use super::*;

    /// Compute QR decomposition using Householder reflections
    ///
    /// # Arguments
    /// * `matrix` - Input matrix to decompose
    /// * `config` - Configuration for the decomposition
    ///
    /// # Returns
    /// * `Result<QRResult<T>, QRError>` - QR decomposition result
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn compute_qr<T>(matrix: &Array2<T>, config: &QRConfig<T>) -> Result<QRResult<T>, QRError>
    where
        T: Float + FloatCore + RealField,
    {
        crate::backend::qr::compute_ndarray_qr(matrix, config)
    }

    /// Compute reduced QR decomposition (economy size)
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn compute_reduced_qr<T>(
        matrix: &Array2<T>,
        config: &QRConfig<T>,
    ) -> Result<QRResult<T>, QRError>
    where
        T: Float + FloatCore + RealField,
    {
        crate::backend::qr::compute_ndarray_reduced_qr(matrix, config)
    }

    /// Compute QR decomposition with column pivoting
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn compute_qr_with_pivoting<T>(
        matrix: &Array2<T>,
        config: &QRConfig<T>,
    ) -> Result<QRResult<T>, QRError>
    where
        T: Float + FloatCore + RealField,
    {
        crate::backend::qr::compute_ndarray_qr_with_pivoting(matrix, config)
    }

    /// Solve least squares problem using QR decomposition
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn solve_least_squares<T>(
        matrix: &Array2<T>,
        rhs: &Array1<T>,
        config: &QRConfig<T>,
    ) -> Result<Array1<T>, QRError>
    where
        T: Float + FloatCore + RealField,
    {
        crate::backend::qr::solve_ndarray_least_squares(matrix, rhs, config)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nalgebra::{DMatrix, DVector};
    use ndarray::Array2;

    use super::*;
    use crate::interop::ndarray_to_nalgebra;

    #[test]
    fn test_nalgebra_qr_basic() {
        let matrix = DMatrix::from_row_slice(3, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]);

        let config = QRConfig::default();
        let qr = nalgebra_qr::compute_qr(&matrix, &config).unwrap();

        // Check that Q is orthogonal (Q^T * Q = I)
        let qt_q = qr.q.transpose() * &qr.q;
        let identity = DMatrix::identity(3, 3);
        assert_relative_eq!(qt_q, identity, epsilon = 1e-10);

        // Check that R is upper triangular
        for i in 1..3 {
            for j in 0..i {
                assert_relative_eq!(qr.r[(i, j)], 0.0, epsilon = 1e-10);
            }
        }

        // Check reconstruction
        let reconstructed = nalgebra_qr::reconstruct_matrix(&qr);
        assert_relative_eq!(reconstructed, matrix, epsilon = 1e-10);
    }

    #[test]
    fn test_nalgebra_qr_rectangular() {
        let matrix = DMatrix::from_row_slice(4, 3, &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ]);

        let config = QRConfig::default();
        let qr = nalgebra_qr::compute_qr(&matrix, &config).unwrap();

        // Check dimensions - nalgebra returns Q as m×min(m,n) and R as min(m,n)×n for m×n input
        assert_eq!(qr.q.nrows(), 4);
        assert_eq!(qr.q.ncols(), 3); // min(4,3) = 3
        assert_eq!(qr.r.nrows(), 3); // min(4,3) = 3
        assert_eq!(qr.r.ncols(), 3);

        // Check reconstruction
        let reconstructed = nalgebra_qr::reconstruct_matrix(&qr);
        assert_relative_eq!(reconstructed, matrix, epsilon = 1e-10);
    }

    #[test]
    fn test_nalgebra_reduced_qr() {
        let matrix = DMatrix::from_row_slice(4, 3, &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ]);

        let config = QRConfig::default();
        let qr = nalgebra_qr::compute_reduced_qr(&matrix, &config).unwrap();

        // Check dimensions for reduced QR
        assert_eq!(qr.q.nrows(), 4);
        assert_eq!(qr.q.ncols(), 3);
        assert_eq!(qr.r.nrows(), 3);
        assert_eq!(qr.r.ncols(), 3);

        // Check that Q^T * Q = I (for reduced Q)
        let qt_q = qr.q.transpose() * &qr.q;
        let identity = DMatrix::identity(3, 3);
        assert_relative_eq!(qt_q, identity, epsilon = 1e-10);
    }

    #[test]
    fn test_nalgebra_least_squares() {
        // Test case: solve Ax = b where A is overdetermined
        let a = DMatrix::from_row_slice(4, 2, &[1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);

        let b = DVector::from_vec(vec![2.0, 3.0, 4.0, 5.0]);

        let config = QRConfig::default();
        let x = nalgebra_qr::solve_least_squares(&a, &b, &config).unwrap();

        // Expected solution for this linear system
        // The system is consistent, so we should get exact solution
        assert_relative_eq!(x[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(x[1], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_ndarray_qr_basic() {
        let matrix =
            Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0])
                .unwrap();

        let config = QRConfig::default();
        let qr = ndarray_qr::compute_qr(&matrix, &config).unwrap();

        // Check that Q is orthogonal
        let qt_q = qr.q.transpose() * &qr.q;
        let identity = DMatrix::identity(3, 3);
        assert_relative_eq!(qt_q, identity, epsilon = 1e-10);

        // Check reconstruction
        let reconstructed = nalgebra_qr::reconstruct_matrix(&qr);
        let expected = ndarray_to_nalgebra(&matrix);
        assert_relative_eq!(reconstructed, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_error_handling() {
        // Test empty matrix
        let empty_matrix = DMatrix::<f64>::zeros(0, 0);
        let config = QRConfig::default();

        let result = nalgebra_qr::compute_qr(&empty_matrix, &config);
        assert!(result.is_err());

        if let Err(QRError::EmptyMatrix) = result {
            // Expected
        } else {
            panic!("Expected EmptyMatrix error");
        }
    }

    #[test]
    fn test_condition_number() {
        // Well-conditioned matrix
        let well_conditioned = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);

        let config = QRConfig::default();
        let qr = nalgebra_qr::compute_qr(&well_conditioned, &config).unwrap();
        let cond = nalgebra_qr::condition_number(&qr);

        assert_relative_eq!(cond, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_edge_cases() {
        // Test empty matrix
        let empty_matrix = DMatrix::<f64>::zeros(0, 0);
        let config = QRConfig::default();
        assert!(matches!(
            nalgebra_qr::compute_qr(&empty_matrix, &config),
            Err(QRError::EmptyMatrix)
        ));

        // Test single element matrix
        let single_element = DMatrix::from_element(1, 1, 5.0);
        let qr = nalgebra_qr::compute_qr(&single_element, &config).unwrap();
        assert_eq!(qr.q.nrows(), 1);
        assert_eq!(qr.q.ncols(), 1);
        assert_eq!(qr.r.nrows(), 1);
        assert_eq!(qr.r.ncols(), 1);
        assert_relative_eq!(qr.q[(0, 0)], 1.0, epsilon = 1e-10);
        assert_relative_eq!(qr.r[(0, 0)], 5.0, epsilon = 1e-10);
        assert_eq!(qr.rank, 1);

        // Test zero matrix
        let zero_matrix = DMatrix::zeros(3, 3);
        let qr = nalgebra_qr::compute_qr(&zero_matrix, &config).unwrap();
        assert_eq!(qr.rank, 0);
        assert_relative_eq!(qr.r, DMatrix::zeros(3, 3), epsilon = 1e-10);

        // Test matrix with NaN
        let mut nan_matrix = DMatrix::from_element(2, 2, 1.0);
        nan_matrix[(0, 0)] = f64::NAN;
        assert!(matches!(
            nalgebra_qr::compute_qr(&nan_matrix, &config),
            Err(QRError::NumericalInstability)
        ));

        // Test matrix with infinity
        let mut inf_matrix = DMatrix::from_element(2, 2, 1.0);
        inf_matrix[(0, 0)] = f64::INFINITY;
        assert!(matches!(
            nalgebra_qr::compute_qr(&inf_matrix, &config),
            Err(QRError::NumericalInstability)
        ));

        // Test invalid rank tolerance
        let invalid_config =
            QRConfig { rank_tolerance: 0.0, max_iterations: 100, use_pivoting: false };
        let matrix = DMatrix::from_element(2, 2, 1.0);
        assert!(matches!(
            nalgebra_qr::compute_qr(&matrix, &invalid_config),
            Err(QRError::InvalidInput(_))
        ));

        // Test very small rank tolerance
        let small_tol_config =
            QRConfig { rank_tolerance: 1e-20, max_iterations: 100, use_pivoting: false };
        assert!(matches!(
            nalgebra_qr::compute_qr(&matrix, &small_tol_config),
            Err(QRError::InvalidInput(_))
        ));
    }

    #[test]
    fn test_least_squares_edge_cases() {
        let config = QRConfig::default();

        // Test empty inputs
        let empty_matrix = DMatrix::<f64>::zeros(0, 0);
        let empty_vector = DVector::<f64>::zeros(0);
        assert!(matches!(
            nalgebra_qr::solve_least_squares(&empty_matrix, &empty_vector, &config),
            Err(QRError::EmptyMatrix)
        ));

        // Test dimension mismatch
        let matrix = DMatrix::from_element(2, 2, 1.0);
        let vector = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        assert!(matches!(
            nalgebra_qr::solve_least_squares(&matrix, &vector, &config),
            Err(QRError::InvalidDimensions(_))
        ));

        // Test underdetermined system
        let matrix = DMatrix::from_element(2, 3, 1.0);
        let vector = DVector::from_vec(vec![1.0, 2.0]);
        assert!(matches!(
            nalgebra_qr::solve_least_squares(&matrix, &vector, &config),
            Err(QRError::InvalidDimensions(_))
        ));

        // Test single equation
        let matrix = DMatrix::from_element(1, 1, 2.0);
        let vector = DVector::from_vec(vec![6.0]);
        let solution = nalgebra_qr::solve_least_squares(&matrix, &vector, &config).unwrap();
        assert_relative_eq!(solution[0], 3.0, epsilon = 1e-10);

        // Test singular matrix
        let singular_matrix = DMatrix::from_row_slice(2, 2, &[
            1.0, 2.0, 2.0, 4.0, // Second row is 2 * first row
        ]);
        let vector = DVector::from_vec(vec![1.0, 2.0]);
        assert!(matches!(
            nalgebra_qr::solve_least_squares(&singular_matrix, &vector, &config),
            Err(QRError::SingularMatrix)
        ));

        // Test with NaN in inputs
        let mut nan_matrix = DMatrix::from_element(2, 2, 1.0);
        nan_matrix[(0, 0)] = f64::NAN;
        let vector = DVector::from_vec(vec![1.0, 2.0]);
        assert!(matches!(
            nalgebra_qr::solve_least_squares(&nan_matrix, &vector, &config),
            Err(QRError::NumericalInstability)
        ));

        // Test with NaN in RHS
        let matrix = DMatrix::from_element(2, 2, 1.0);
        let mut nan_vector = DVector::from_vec(vec![1.0, 2.0]);
        nan_vector[0] = f64::NAN;
        assert!(matches!(
            nalgebra_qr::solve_least_squares(&matrix, &nan_vector, &config),
            Err(QRError::NumericalInstability)
        ));
    }

    #[test]
    fn test_rank_deficient_matrix() {
        // Test rank-deficient matrix (rank 1 instead of 2)
        let rank_deficient = DMatrix::from_row_slice(2, 2, &[
            1.0, 2.0, 2.0, 4.0, // Second row is 2 * first row
        ]);

        let config = QRConfig::default();
        let qr = nalgebra_qr::compute_qr(&rank_deficient, &config).unwrap();

        // Should detect rank deficiency
        assert_eq!(qr.rank, 1);

        // Reconstruction should still work
        let reconstructed = nalgebra_qr::reconstruct_matrix(&qr);
        assert_relative_eq!(reconstructed, rank_deficient, epsilon = 1e-10);
    }

    #[test]
    fn test_nalgebra_qr_with_pivoting() {
        let matrix = DMatrix::from_row_slice(3, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]);

        let config = QRConfig::default();
        let qr = nalgebra_qr::compute_qr_with_pivoting(&matrix, &config).unwrap();

        // Check that Q is orthogonal (Q^T * Q = I)
        let qt_q = qr.q.transpose() * &qr.q;
        let identity = DMatrix::identity(3, 3);
        assert_relative_eq!(qt_q, identity, epsilon = 1e-10);

        // Check that R is upper triangular
        for i in 1..3 {
            for j in 0..i {
                assert_relative_eq!(qr.r[(i, j)], 0.0, epsilon = 1e-10);
            }
        }

        // Check reconstruction
        let reconstructed = nalgebra_qr::reconstruct_matrix(&qr);
        assert_relative_eq!(reconstructed, matrix, epsilon = 1e-10);
    }

    #[test]
    fn test_ndarray_qr_with_pivoting() {
        let matrix =
            Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0])
                .unwrap();

        let config = QRConfig::default();
        let qr = ndarray_qr::compute_qr_with_pivoting(&matrix, &config).unwrap();

        // Check that Q is orthogonal
        let qt_q = qr.q.transpose() * &qr.q;
        let identity = DMatrix::identity(3, 3);
        assert_relative_eq!(qt_q, identity, epsilon = 1e-10);

        // Check reconstruction
        let reconstructed = nalgebra_qr::reconstruct_matrix(&qr);
        let expected = ndarray_to_nalgebra(&matrix);
        assert_relative_eq!(reconstructed, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_ndarray_reduced_qr() {
        let matrix = Array2::from_shape_vec((4, 3), vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ])
        .unwrap();

        let config = QRConfig::default();
        let qr = ndarray_qr::compute_reduced_qr(&matrix, &config).unwrap();

        // Check dimensions for reduced QR
        assert_eq!(qr.q.nrows(), 4);
        assert_eq!(qr.q.ncols(), 3);
        assert_eq!(qr.r.nrows(), 3);
        assert_eq!(qr.r.ncols(), 3);

        // Check that Q^T * Q = I (for reduced Q)
        let qt_q = qr.q.transpose() * &qr.q;
        let identity = DMatrix::identity(3, 3);
        assert_relative_eq!(qt_q, identity, epsilon = 1e-10);
    }

    #[test]
    fn test_ndarray_least_squares() {
        use ndarray::Array1;

        let a =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0]).unwrap();
        let b = Array1::from_vec(vec![2.0, 3.0, 4.0, 5.0]);

        let config = QRConfig::default();
        let x = ndarray_qr::solve_least_squares(&a, &b, &config).unwrap();

        // Expected solution: [1, 1] for this consistent system
        assert_relative_eq!(x[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(x[1], 1.0, epsilon = 1e-10);
    }
}
