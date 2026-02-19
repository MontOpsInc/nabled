//! # Eigenvalue Decomposition
//!
//! Symmetric eigenvalue decomposition for real symmetric matrices.

use nalgebra::{DMatrix, DVector, RealField};
use ndarray::{Array1, Array2};
use num_traits::Float;
use std::fmt;

/// Error types for eigenvalue decomposition
#[derive(Debug, Clone, PartialEq)]
pub enum EigenError {
    /// Matrix is empty
    EmptyMatrix,
    /// Matrix is not square
    NotSquare,
    /// Matrix is not symmetric
    NonSymmetric,
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
            EigenError::ConvergenceFailed => write!(f, "Eigenvalue algorithm failed to converge"),
            EigenError::NumericalInstability => write!(f, "Numerical instability detected"),
            EigenError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for EigenError {}

/// Eigen decomposition result for nalgebra
#[derive(Debug, Clone)]
pub struct NalgebraEigenResult<T: RealField> {
    /// Eigenvalues
    pub eigenvalues: DVector<T>,
    /// Eigenvectors (columns)
    pub eigenvectors: DMatrix<T>,
}

/// Eigen decomposition result for ndarray
#[derive(Debug, Clone)]
pub struct NdarrayEigenResult<T: Float> {
    /// Eigenvalues
    pub eigenvalues: Array1<T>,
    /// Eigenvectors (columns)
    pub eigenvectors: Array2<T>,
}

/// Check if matrix is symmetric within tolerance
fn is_symmetric<T: RealField + Copy>(matrix: &DMatrix<T>, tol: T) -> bool {
    let (n, m) = matrix.shape();
    if n != m {
        return false;
    }
    for i in 0..n {
        for j in (i + 1)..n {
            if (matrix[(i, j)] - matrix[(j, i)]).abs() > tol {
                return false;
            }
        }
    }
    true
}

/// Nalgebra symmetric eigenvalue decomposition
pub mod nalgebra_eigen {
    use super::*;

    /// Compute symmetric eigenvalue decomposition
    pub fn compute_symmetric_eigen<T: RealField + Copy + num_traits::Float>(
        matrix: &DMatrix<T>,
    ) -> Result<NalgebraEigenResult<T>, EigenError> {
        if matrix.is_empty() {
            return Err(EigenError::EmptyMatrix);
        }
        if !matrix.is_square() {
            return Err(EigenError::NotSquare);
        }
        let tol = T::from(1e-10).unwrap_or_else(T::nan);
        if !is_symmetric(matrix, tol) {
            return Err(EigenError::NonSymmetric);
        }
        if matrix.iter().any(|&x| !num_traits::Float::is_finite(x)) {
            return Err(EigenError::NumericalInstability);
        }

        let eigen = matrix.clone().symmetric_eigen();
        Ok(NalgebraEigenResult {
            eigenvalues: eigen.eigenvalues,
            eigenvectors: eigen.eigenvectors,
        })
    }
}

/// Ndarray symmetric eigenvalue decomposition (via nalgebra)
pub mod ndarray_eigen {
    use super::*;
    use crate::utils::{nalgebra_to_ndarray, ndarray_to_nalgebra};

    /// Compute symmetric eigenvalue decomposition
    pub fn compute_symmetric_eigen<T: Float + RealField>(
        matrix: &Array2<T>,
    ) -> Result<NdarrayEigenResult<T>, EigenError> {
        let nalg = ndarray_to_nalgebra(matrix);
        let result = super::nalgebra_eigen::compute_symmetric_eigen(&nalg)?;
        Ok(NdarrayEigenResult {
            eigenvalues: Array1::from_vec(result.eigenvalues.as_slice().to_vec()),
            eigenvectors: nalgebra_to_ndarray(&result.eigenvectors),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

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
}
