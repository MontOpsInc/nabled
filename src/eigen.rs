//! # Eigenvalue Decomposition
//!
//! Symmetric eigenvalue decomposition for real symmetric matrices.

use std::fmt;

use nalgebra::{DMatrix, DVector, RealField};
use ndarray::{Array1, Array2};
use num_traits::Float;

/// Error types for eigenvalue decomposition
#[derive(Debug, Clone, PartialEq)]
pub enum EigenError {
    /// Matrix is empty
    EmptyMatrix,
    /// Matrix is not square
    NotSquare,
    /// Matrix is not symmetric
    NonSymmetric,
    /// Matrix B is not positive definite (for generalized eigen)
    NotPositiveDefinite,
    /// Dimension mismatch between A and B
    DimensionMismatch,
    /// Convergence failed
    ConvergenceFailed,
    /// Numerical instability
    NumericalInstability,
    /// Invalid input
    InvalidInput(String),
}

impl fmt::Display for EigenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EigenError::EmptyMatrix => write!(f, "Matrix is empty"),
            EigenError::NotSquare => write!(f, "Matrix must be square"),
            EigenError::NonSymmetric => write!(f, "Matrix must be symmetric"),
            EigenError::NotPositiveDefinite => {
                write!(f, "Matrix B must be positive definite")
            }
            EigenError::DimensionMismatch => write!(f, "Matrix dimensions must match"),
            EigenError::ConvergenceFailed => write!(f, "Eigenvalue algorithm failed to converge"),
            EigenError::NumericalInstability => write!(f, "Numerical instability detected"),
            EigenError::InvalidInput(msg) => write!(f, "Invalid input: {msg}"),
        }
    }
}

impl std::error::Error for EigenError {}

/// Eigen decomposition result for nalgebra
#[derive(Debug, Clone)]
pub struct NalgebraEigenResult<T: RealField> {
    /// Eigenvalues
    pub eigenvalues:  DVector<T>,
    /// Eigenvectors (columns)
    pub eigenvectors: DMatrix<T>,
}

/// Eigen decomposition result for ndarray
#[derive(Debug, Clone)]
pub struct NdarrayEigenResult<T: Float> {
    /// Eigenvalues
    pub eigenvalues:  Array1<T>,
    /// Eigenvectors (columns)
    pub eigenvectors: Array2<T>,
}

/// Generalized eigenvalue result for nalgebra
#[derive(Debug, Clone)]
pub struct NalgebraGeneralizedEigenResult<T: RealField> {
    /// Generalized eigenvalues
    pub eigenvalues:  DVector<T>,
    /// Generalized eigenvectors (columns)
    pub eigenvectors: DMatrix<T>,
}

/// Generalized eigenvalue result for ndarray
#[derive(Debug, Clone)]
pub struct NdarrayGeneralizedEigenResult<T: Float> {
    /// Generalized eigenvalues
    pub eigenvalues:  Array1<T>,
    /// Generalized eigenvectors (columns)
    pub eigenvectors: Array2<T>,
}

/// Nalgebra symmetric eigenvalue decomposition
pub mod nalgebra_eigen {
    use super::*;

    /// Compute symmetric eigenvalue decomposition.
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn compute_symmetric_eigen<T: RealField + Copy + Float>(
        matrix: &DMatrix<T>,
    ) -> Result<NalgebraEigenResult<T>, EigenError> {
        crate::backend::eigen::compute_nalgebra_symmetric_eigen(matrix)
    }

    /// Compute symmetric eigenvalue decomposition with a LAPACK-backed kernel.
    ///
    /// This path is available on Linux when the `lapack-kernels` feature is enabled.
    ///
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the LAPACK routine cannot produce a stable result.
    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    pub fn compute_symmetric_eigen_lapack(
        matrix: &DMatrix<f64>,
    ) -> Result<NalgebraEigenResult<f64>, EigenError> {
        crate::backend::eigen::compute_nalgebra_lapack_symmetric_eigen(matrix)
    }

    /// Compute generalized eigenvalue decomposition Av = λBv for symmetric A and SPD B.
    ///
    /// This routine reduces to a symmetric standard eigen problem using a Cholesky factor of `B`.
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn compute_generalized_eigen<T: RealField + Copy + Float>(
        matrix_a: &DMatrix<T>,
        matrix_b: &DMatrix<T>,
    ) -> Result<NalgebraGeneralizedEigenResult<T>, EigenError> {
        crate::backend::eigen::compute_nalgebra_generalized_eigen(matrix_a, matrix_b)
    }

    /// Compute generalized eigenvalue decomposition with a LAPACK-backed symmetric eigensolver.
    ///
    /// This path is available on Linux when the `lapack-kernels` feature is enabled.
    ///
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the LAPACK routine cannot produce a stable result.
    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    pub fn compute_generalized_eigen_lapack(
        matrix_a: &DMatrix<f64>,
        matrix_b: &DMatrix<f64>,
    ) -> Result<NalgebraGeneralizedEigenResult<f64>, EigenError> {
        crate::backend::eigen::compute_nalgebra_lapack_generalized_eigen(matrix_a, matrix_b)
    }
}

/// Ndarray symmetric eigenvalue decomposition
pub mod ndarray_eigen {
    use super::*;

    /// Compute symmetric eigenvalue decomposition.
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn compute_symmetric_eigen<T: Float + RealField>(
        matrix: &Array2<T>,
    ) -> Result<NdarrayEigenResult<T>, EigenError> {
        crate::backend::eigen::compute_ndarray_symmetric_eigen(matrix)
    }

    /// Compute symmetric eigenvalue decomposition with a LAPACK-backed kernel.
    ///
    /// This path is available on Linux when the `lapack-kernels` feature is enabled.
    ///
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the LAPACK routine cannot produce a stable result.
    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    pub fn compute_symmetric_eigen_lapack(
        matrix: &Array2<f64>,
    ) -> Result<NdarrayEigenResult<f64>, EigenError> {
        crate::backend::eigen::compute_ndarray_lapack_symmetric_eigen(matrix)
    }

    /// Compute generalized eigenvalue decomposition Av = λBv.
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn compute_generalized_eigen<T: Float + RealField>(
        matrix_a: &Array2<T>,
        matrix_b: &Array2<T>,
    ) -> Result<NdarrayGeneralizedEigenResult<T>, EigenError> {
        crate::backend::eigen::compute_ndarray_generalized_eigen(matrix_a, matrix_b)
    }

    /// Compute generalized eigenvalue decomposition with a LAPACK-backed kernel.
    ///
    /// This path is available on Linux when the `lapack-kernels` feature is enabled.
    ///
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the LAPACK routine cannot produce a stable result.
    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    pub fn compute_generalized_eigen_lapack(
        matrix_a: &Array2<f64>,
        matrix_b: &Array2<f64>,
    ) -> Result<NdarrayGeneralizedEigenResult<f64>, EigenError> {
        crate::backend::eigen::compute_ndarray_lapack_generalized_eigen(matrix_a, matrix_b)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn test_nalgebra_eigen_reconstruct() {
        let a = DMatrix::from_row_slice(2, 2, &[4.0, 1.0, 1.0, 3.0]);
        let result = nalgebra_eigen::compute_symmetric_eigen(&a).unwrap();
        let q = &result.eigenvectors;
        let d = DMatrix::from_diagonal(&result.eigenvalues);
        let reconstructed = q * d * q.transpose();
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(reconstructed[(i, j)], a[(i, j)], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_generalized_eigen() {
        let a = DMatrix::from_row_slice(2, 2, &[4.0, 1.0, 1.0, 3.0]);
        let b = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let result = nalgebra_eigen::compute_generalized_eigen(&a, &b).unwrap();

        // For B=I, generalized eigen reduces to standard eigen.
        for i in 0..2 {
            let av = &a * result.eigenvectors.column(i);
            let bv = &b * result.eigenvectors.column(i);
            let lam = result.eigenvalues[i];
            for j in 0..2 {
                assert_relative_eq!(av[j], lam * bv[j], epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_ndarray_eigen_reconstruct() {
        let a = Array2::from_shape_vec((2, 2), vec![4.0, 1.0, 1.0, 3.0]).unwrap();
        let result = ndarray_eigen::compute_symmetric_eigen(&a).unwrap();
        let n = 2;
        let mut reconstructed: Array2<f64> = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    reconstructed[[i, j]] += result.eigenvectors[[i, k]]
                        * result.eigenvalues[k]
                        * result.eigenvectors[[j, k]];
                }
            }
        }
        for i in 0..n {
            for j in 0..n {
                assert_relative_eq!(reconstructed[[i, j]], a[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_eigen_error_display_variants() {
        assert!(format!("{}", EigenError::EmptyMatrix).contains("empty"));
        assert!(format!("{}", EigenError::NotSquare).contains("square"));
        assert!(format!("{}", EigenError::NonSymmetric).contains("symmetric"));
        assert!(format!("{}", EigenError::NotPositiveDefinite).contains("positive definite"));
        assert!(format!("{}", EigenError::DimensionMismatch).contains("dimensions"));
        assert!(format!("{}", EigenError::ConvergenceFailed).contains("failed"));
        assert!(format!("{}", EigenError::NumericalInstability).contains("instability"));
        assert!(format!("{}", EigenError::InvalidInput("x".to_string())).contains('x'));
    }

    #[test]
    fn test_eigen_error_paths() {
        let empty = DMatrix::<f64>::zeros(0, 0);
        assert!(matches!(
            nalgebra_eigen::compute_symmetric_eigen(&empty),
            Err(EigenError::EmptyMatrix)
        ));

        let non_square = DMatrix::from_row_slice(1, 2, &[1.0, 2.0]);
        assert!(matches!(
            nalgebra_eigen::compute_symmetric_eigen(&non_square),
            Err(EigenError::NotSquare)
        ));

        let non_symmetric = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 0.0, 1.0]);
        assert!(matches!(
            nalgebra_eigen::compute_symmetric_eigen(&non_symmetric),
            Err(EigenError::NonSymmetric)
        ));

        let non_finite = DMatrix::from_row_slice(2, 2, &[1.0, f64::NAN, 0.0, 1.0]);
        assert!(matches!(
            nalgebra_eigen::compute_symmetric_eigen(&non_finite),
            Err(EigenError::NumericalInstability)
        ));

        let a = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 3.0]);
        let b_dim_bad =
            DMatrix::from_row_slice(3, 3, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        assert!(matches!(
            nalgebra_eigen::compute_generalized_eigen(&a, &b_dim_bad),
            Err(EigenError::DimensionMismatch)
        ));

        let b_non_spd = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 0.0]);
        assert!(matches!(
            nalgebra_eigen::compute_generalized_eigen(&a, &b_non_spd),
            Err(EigenError::NotPositiveDefinite)
        ));

        let a_nd = Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 0.0, 3.0]).unwrap();
        let b_nd = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let result = ndarray_eigen::compute_generalized_eigen(&a_nd, &b_nd).unwrap();
        assert_eq!(result.eigenvalues.len(), 2);
        assert_eq!(result.eigenvectors.dim(), (2, 2));
    }

    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    #[test]
    fn test_nalgebra_symmetric_eigen_lapack_basic() {
        let a = DMatrix::from_row_slice(2, 2, &[4.0, 1.0, 1.0, 3.0]);
        let result = nalgebra_eigen::compute_symmetric_eigen_lapack(&a).unwrap();

        assert_eq!(result.eigenvalues.len(), 2);
        assert_eq!(result.eigenvectors.shape(), (2, 2));
    }

    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    #[test]
    fn test_nalgebra_generalized_eigen_lapack_basic() {
        let a = DMatrix::from_row_slice(2, 2, &[4.0, 1.0, 1.0, 3.0]);
        let b = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let result = nalgebra_eigen::compute_generalized_eigen_lapack(&a, &b).unwrap();

        for i in 0..2 {
            let av = &a * result.eigenvectors.column(i);
            let bv = &b * result.eigenvectors.column(i);
            let lam = result.eigenvalues[i];
            for j in 0..2 {
                assert_relative_eq!(av[j], lam * bv[j], epsilon = 1e-8);
            }
        }
    }

    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    #[test]
    fn test_ndarray_symmetric_eigen_lapack_basic() {
        let a = Array2::from_shape_vec((2, 2), vec![4.0, 1.0, 1.0, 3.0]).unwrap();
        let result = ndarray_eigen::compute_symmetric_eigen_lapack(&a).unwrap();

        assert_eq!(result.eigenvalues.len(), 2);
        assert_eq!(result.eigenvectors.dim(), (2, 2));
    }

    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    #[test]
    fn test_ndarray_generalized_eigen_lapack_basic() {
        let a = Array2::from_shape_vec((2, 2), vec![4.0, 1.0, 1.0, 3.0]).unwrap();
        let b = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let result = ndarray_eigen::compute_generalized_eigen_lapack(&a, &b).unwrap();

        assert_eq!(result.eigenvalues.len(), 2);
        assert_eq!(result.eigenvectors.dim(), (2, 2));
    }
}
