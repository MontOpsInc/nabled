//! # Schur Decomposition
//!
//! Schur decomposition: A = Q T Q^H where Q is unitary and T is upper triangular.
//! Used for eigenvalue problems and matrix functions.

use std::fmt;

use nalgebra::{DMatrix, RealField};
use ndarray::Array2;
use num_traits::Float;

/// Error types for Schur decomposition
#[derive(Debug, Clone, PartialEq)]
pub enum SchurError {
    /// Matrix is empty
    EmptyMatrix,
    /// Matrix is not square
    NotSquare,
    /// Algorithm failed to converge
    ConvergenceFailed,
    /// Numerical instability detected
    NumericalInstability,
    /// Invalid input
    InvalidInput(String),
}

impl fmt::Display for SchurError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SchurError::EmptyMatrix => write!(f, "Matrix is empty"),
            SchurError::NotSquare => write!(f, "Matrix must be square"),
            SchurError::ConvergenceFailed => {
                write!(f, "Schur decomposition failed to converge")
            }
            SchurError::NumericalInstability => write!(f, "Numerical instability detected"),
            SchurError::InvalidInput(msg) => write!(f, "Invalid input: {msg}"),
        }
    }
}

impl std::error::Error for SchurError {}

/// Schur decomposition result for nalgebra
#[derive(Debug, Clone)]
pub struct NalgebraSchurResult<T: RealField> {
    /// Unitary matrix Q
    pub q: DMatrix<T>,
    /// Upper triangular matrix T
    pub t: DMatrix<T>,
}

/// Schur decomposition result for ndarray
#[derive(Debug, Clone)]
pub struct NdarraySchurResult<T: Float> {
    /// Unitary matrix Q
    pub q: Array2<T>,
    /// Upper triangular matrix T
    pub t: Array2<T>,
}

/// Nalgebra Schur decomposition
pub mod nalgebra_schur {
    use super::*;

    /// Compute Schur decomposition A = Q T Q^H.
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn compute_schur<T: RealField + Copy + Float>(
        matrix: &DMatrix<T>,
    ) -> Result<NalgebraSchurResult<T>, SchurError> {
        crate::backend::schur::compute_nalgebra_schur(matrix)
    }

    /// Compute Schur decomposition with a LAPACK-backed kernel.
    ///
    /// This path is available on Linux when the `lapack-kernels` feature is enabled.
    ///
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the LAPACK routine cannot produce a stable result.
    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    pub fn compute_schur_lapack(
        matrix: &DMatrix<f64>,
    ) -> Result<NalgebraSchurResult<f64>, SchurError> {
        crate::backend::schur::compute_nalgebra_lapack_schur(matrix)
    }
}

/// Ndarray Schur decomposition
pub mod ndarray_schur {
    use super::*;

    /// Compute Schur decomposition.
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn compute_schur<T: Float + RealField>(
        matrix: &Array2<T>,
    ) -> Result<NdarraySchurResult<T>, SchurError> {
        crate::backend::schur::compute_ndarray_schur(matrix)
    }

    /// Compute Schur decomposition with a LAPACK-backed kernel.
    ///
    /// This path is available on Linux when the `lapack-kernels` feature is enabled.
    ///
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the LAPACK routine cannot produce a stable result.
    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    pub fn compute_schur_lapack(
        matrix: &Array2<f64>,
    ) -> Result<NdarraySchurResult<f64>, SchurError> {
        crate::backend::schur::compute_ndarray_lapack_schur(matrix)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn test_nalgebra_schur_reconstruct() {
        let a = DMatrix::from_row_slice(2, 2, &[4.0, 1.0, 1.0, 3.0]);
        let result = nalgebra_schur::compute_schur(&a).unwrap();
        let reconstructed = &result.q * &result.t * result.q.transpose();
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(reconstructed[(i, j)], a[(i, j)], epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_ndarray_schur_reconstruct() {
        let a = Array2::from_shape_vec((2, 2), vec![4.0, 1.0, 1.0, 3.0]).unwrap();
        let result = ndarray_schur::compute_schur(&a).unwrap();
        let mut reconstructed: Array2<f64> = Array2::zeros((2, 2));
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    for l in 0..2 {
                        reconstructed[[i, j]] +=
                            result.q[[i, k]] * result.t[[k, l]] * result.q[[j, l]];
                    }
                }
            }
        }
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(reconstructed[[i, j]], a[[i, j]], epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_schur_error_display_variants() {
        assert!(format!("{}", SchurError::EmptyMatrix).contains("empty"));
        assert!(format!("{}", SchurError::NotSquare).contains("square"));
        assert!(format!("{}", SchurError::ConvergenceFailed).contains("failed"));
        assert!(format!("{}", SchurError::NumericalInstability).contains("instability"));
        assert!(format!("{}", SchurError::InvalidInput("x".to_string())).contains('x'));
    }

    #[test]
    fn test_schur_error_paths() {
        let empty = DMatrix::<f64>::zeros(0, 0);
        assert!(matches!(nalgebra_schur::compute_schur(&empty), Err(SchurError::EmptyMatrix)));

        let non_square = DMatrix::from_row_slice(1, 2, &[1.0, 2.0]);
        assert!(matches!(nalgebra_schur::compute_schur(&non_square), Err(SchurError::NotSquare)));

        let non_finite = DMatrix::from_row_slice(2, 2, &[1.0, f64::NAN, 0.0, 1.0]);
        assert!(matches!(
            nalgebra_schur::compute_schur(&non_finite),
            Err(SchurError::NumericalInstability)
        ));
    }

    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    #[test]
    fn test_nalgebra_schur_lapack_basic() {
        let matrix = DMatrix::from_row_slice(2, 2, &[4.0, 1.0, 1.0, 3.0]);
        let schur = nalgebra_schur::compute_schur_lapack(&matrix).unwrap();

        assert_eq!(schur.q.shape(), (2, 2));
        assert_eq!(schur.t.shape(), (2, 2));
    }

    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    #[test]
    fn test_ndarray_schur_lapack_basic() {
        let matrix = Array2::from_shape_vec((2, 2), vec![4.0, 1.0, 1.0, 3.0]).unwrap();
        let schur = ndarray_schur::compute_schur_lapack(&matrix).unwrap();

        assert_eq!(schur.q.dim(), (2, 2));
        assert_eq!(schur.t.dim(), (2, 2));
    }
}
