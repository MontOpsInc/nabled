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
    /// # Errors
    /// Returns an error if inputs are invalid, dimensions are incompatible, or the
    /// underlying numerical routine fails to converge or produce a valid result.
    pub fn compute_symmetric_eigen<T: RealField + Copy + Float>(
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
        if matrix.iter().any(|&x| !Float::is_finite(x)) {
            return Err(EigenError::NumericalInstability);
        }

        let eigen = matrix.clone().symmetric_eigen();
        Ok(NalgebraEigenResult {
            eigenvalues:  eigen.eigenvalues,
            eigenvectors: eigen.eigenvectors,
        })
    }

    /// Compute generalized eigenvalue decomposition Av = λBv for symmetric A and SPD B
    /// Reduces to standard eigen via Cholesky on B: C = L^{-1} A L^{-T}, then v = L^{-T} w
    /// # Errors
    /// Returns an error if inputs are invalid, dimensions are incompatible, or the
    /// underlying numerical routine fails to converge or produce a valid result.
    pub fn compute_generalized_eigen<T: RealField + Copy + Float>(
        matrix_a: &DMatrix<T>,
        matrix_b: &DMatrix<T>,
    ) -> Result<NalgebraGeneralizedEigenResult<T>, EigenError> {
        if matrix_a.is_empty() || matrix_b.is_empty() {
            return Err(EigenError::EmptyMatrix);
        }
        let (ar, ac) = matrix_a.shape();
        let (br, bc) = matrix_b.shape();
        if ar != ac || br != bc || ar != br {
            return Err(EigenError::DimensionMismatch);
        }
        let tol = T::from(1e-10).unwrap_or_else(T::nan);
        if !is_symmetric(matrix_a, tol) {
            return Err(EigenError::NonSymmetric);
        }
        if !is_symmetric(matrix_b, tol) {
            return Err(EigenError::NonSymmetric);
        }

        let cholesky = nalgebra::linalg::Cholesky::new(matrix_b.clone())
            .ok_or(EigenError::NotPositiveDefinite)?;
        let l = cholesky.l();

        // C = L^{-1} A L^{-T}
        let linv_a = l.solve_lower_triangular(matrix_a).ok_or(EigenError::NumericalInstability)?;
        let c = l
            .transpose()
            .solve_upper_triangular(&linv_a)
            .ok_or(EigenError::NumericalInstability)?;

        let eigen = c.symmetric_eigen();

        // v = L^{-T} w (eigenvectors of generalized problem)
        let w = eigen.eigenvectors;
        let eigenvectors =
            l.transpose().solve_lower_triangular(&w).ok_or(EigenError::NumericalInstability)?;

        Ok(NalgebraGeneralizedEigenResult { eigenvalues: eigen.eigenvalues, eigenvectors })
    }
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

/// Ndarray symmetric eigenvalue decomposition (via nalgebra)
pub mod ndarray_eigen {
    use super::*;
    use crate::interop::{nalgebra_to_ndarray, ndarray_to_nalgebra};

    /// Compute symmetric eigenvalue decomposition
    /// # Errors
    /// Returns an error if inputs are invalid, dimensions are incompatible, or the
    /// underlying numerical routine fails to converge or produce a valid result.
    pub fn compute_symmetric_eigen<T: Float + RealField>(
        matrix: &Array2<T>,
    ) -> Result<NdarrayEigenResult<T>, EigenError> {
        let nalg = ndarray_to_nalgebra(matrix);
        let result = nalgebra_eigen::compute_symmetric_eigen(&nalg)?;
        Ok(NdarrayEigenResult {
            eigenvalues:  Array1::from_vec(result.eigenvalues.as_slice().to_vec()),
            eigenvectors: nalgebra_to_ndarray(&result.eigenvectors),
        })
    }

    /// Compute generalized eigenvalue decomposition Av = λBv
    /// # Errors
    /// Returns an error if inputs are invalid, dimensions are incompatible, or the
    /// underlying numerical routine fails to converge or produce a valid result.
    pub fn compute_generalized_eigen<T: Float + RealField>(
        a: &Array2<T>,
        b: &Array2<T>,
    ) -> Result<NdarrayGeneralizedEigenResult<T>, EigenError> {
        let nalg_a = ndarray_to_nalgebra(a);
        let nalg_b = ndarray_to_nalgebra(b);
        let result = nalgebra_eigen::compute_generalized_eigen(&nalg_a, &nalg_b)?;
        Ok(NdarrayGeneralizedEigenResult {
            eigenvalues:  Array1::from_vec(result.eigenvalues.as_slice().to_vec()),
            eigenvectors: nalgebra_to_ndarray(&result.eigenvectors),
        })
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
        // For B=I, generalized eigen reduces to standard eigen
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
}
