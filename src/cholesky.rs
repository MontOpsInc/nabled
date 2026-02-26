//! # Cholesky Decomposition
//!
//! Cholesky decomposition for symmetric positive-definite matrices.
//! Used for solving linear systems and computing matrix inverses.

use std::fmt;

use nalgebra::{DMatrix, DVector, RealField};
use ndarray::{Array1, Array2};
use num_traits::Float;

/// Error types for Cholesky decomposition
#[derive(Debug, Clone, PartialEq)]
pub enum CholeskyError {
    /// Matrix is empty
    EmptyMatrix,
    /// Matrix is not square
    NotSquare,
    /// Matrix is not positive-definite
    NotPositiveDefinite,
    /// Numerical instability detected
    NumericalInstability,
    /// Invalid input
    InvalidInput(String),
}

impl fmt::Display for CholeskyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CholeskyError::EmptyMatrix => write!(f, "Matrix is empty"),
            CholeskyError::NotSquare => write!(f, "Matrix must be square"),
            CholeskyError::NotPositiveDefinite => write!(f, "Matrix is not positive-definite"),
            CholeskyError::NumericalInstability => write!(f, "Numerical instability detected"),
            CholeskyError::InvalidInput(msg) => write!(f, "Invalid input: {msg}"),
        }
    }
}

impl std::error::Error for CholeskyError {}

/// Cholesky decomposition result for nalgebra
#[derive(Debug, Clone)]
pub struct NalgebraCholeskyResult<T: RealField> {
    /// Lower triangular factor L
    pub l: DMatrix<T>,
}

/// Cholesky decomposition result for ndarray
#[derive(Debug, Clone)]
pub struct NdarrayCholeskyResult<T: Float> {
    /// Lower triangular factor L
    pub l: Array2<T>,
}

/// Nalgebra Cholesky decomposition
pub mod nalgebra_cholesky {
    use nalgebra::linalg::Cholesky;

    use super::*;

    /// Compute Cholesky decomposition
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn compute_cholesky<T: RealField + Copy + Float>(
        matrix: &DMatrix<T>,
    ) -> Result<NalgebraCholeskyResult<T>, CholeskyError> {
        if matrix.is_empty() {
            return Err(CholeskyError::EmptyMatrix);
        }
        let (rows, cols) = matrix.shape();
        if rows != cols {
            return Err(CholeskyError::NotSquare);
        }
        if matrix.iter().any(|&x| !Float::is_finite(x)) {
            return Err(CholeskyError::NumericalInstability);
        }

        let cholesky = Cholesky::new(matrix.clone()).ok_or(CholeskyError::NotPositiveDefinite)?;
        let l = cholesky.l().clone();

        Ok(NalgebraCholeskyResult { l })
    }

    /// Solve Ax = b for symmetric positive-definite A
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn solve<T: RealField + Copy>(
        matrix: &DMatrix<T>,
        rhs: &DVector<T>,
    ) -> Result<DVector<T>, CholeskyError> {
        if matrix.is_empty() || rhs.is_empty() {
            return Err(CholeskyError::EmptyMatrix);
        }
        let (rows, cols) = matrix.shape();
        if rows != cols {
            return Err(CholeskyError::NotSquare);
        }
        if rows != rhs.len() {
            return Err(CholeskyError::InvalidInput(
                "RHS length must match matrix dimensions".to_string(),
            ));
        }

        let cholesky = Cholesky::new(matrix.clone()).ok_or(CholeskyError::NotPositiveDefinite)?;
        Ok(cholesky.solve(rhs))
    }

    /// Compute matrix inverse
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn inverse<T: RealField + Copy>(matrix: &DMatrix<T>) -> Result<DMatrix<T>, CholeskyError> {
        if matrix.is_empty() {
            return Err(CholeskyError::EmptyMatrix);
        }
        let (rows, cols) = matrix.shape();
        if rows != cols {
            return Err(CholeskyError::NotSquare);
        }

        let cholesky = Cholesky::new(matrix.clone()).ok_or(CholeskyError::NotPositiveDefinite)?;
        Ok(cholesky.inverse())
    }
}

/// Ndarray Cholesky decomposition (via nalgebra)
pub mod ndarray_cholesky {
    use super::*;
    use crate::interop::{nalgebra_to_ndarray, ndarray_to_nalgebra};

    /// Compute Cholesky decomposition
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn compute_cholesky<T: Float + RealField>(
        matrix: &Array2<T>,
    ) -> Result<NdarrayCholeskyResult<T>, CholeskyError> {
        let nalg = ndarray_to_nalgebra(matrix);
        let result = nalgebra_cholesky::compute_cholesky(&nalg)?;
        Ok(NdarrayCholeskyResult { l: nalgebra_to_ndarray(&result.l) })
    }

    /// Solve Ax = b
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn solve<T: Float + RealField>(
        matrix: &Array2<T>,
        rhs: &Array1<T>,
    ) -> Result<Array1<T>, CholeskyError> {
        let nalg_matrix = ndarray_to_nalgebra(matrix);
        let nalg_rhs = DVector::from_vec(rhs.to_vec());
        let solution = nalgebra_cholesky::solve(&nalg_matrix, &nalg_rhs)?;
        Ok(Array1::from_vec(solution.as_slice().to_vec()))
    }

    /// Compute matrix inverse
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn inverse<T: Float + RealField>(matrix: &Array2<T>) -> Result<Array2<T>, CholeskyError> {
        let nalg = ndarray_to_nalgebra(matrix);
        let inv = nalgebra_cholesky::inverse(&nalg)?;
        Ok(nalgebra_to_ndarray(&inv))
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn test_nalgebra_cholesky_solve() {
        // Create symmetric positive-definite matrix: A = L * L^T
        let l = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 1.0, 3.0]);
        let a = &l * l.transpose();
        let b = DVector::from_vec(vec![1.0, 2.0]);
        let x = nalgebra_cholesky::solve(&a, &b).unwrap();
        let ax = &a * &x;
        for i in 0..2 {
            assert_relative_eq!(ax[i], b[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_nalgebra_cholesky_inverse() {
        let l = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 1.0, 3.0]);
        let a = &l * l.transpose();
        let inv = nalgebra_cholesky::inverse(&a).unwrap();
        let identity = &a * &inv;
        assert_relative_eq!(identity[(0, 0)], 1.0, epsilon = 1e-10);
        assert_relative_eq!(identity[(1, 1)], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_ndarray_cholesky_solve() {
        let l = Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 1.0, 3.0]).unwrap();
        let mut a: Array2<f64> = Array2::zeros((2, 2));
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    a[[i, j]] += l[[i, k]] * l[[j, k]];
                }
            }
        }
        let b = Array1::from_vec(vec![1.0, 2.0]);
        let x = ndarray_cholesky::solve(&a, &b).unwrap();
        let mut ax: Array1<f64> = Array1::zeros(2);
        for i in 0..2 {
            for j in 0..2 {
                ax[i] += a[[i, j]] * x[j];
            }
        }
        for i in 0..2 {
            assert_relative_eq!(ax[i], b[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_ndarray_cholesky_inverse() {
        let l = Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 1.0, 3.0]).unwrap();
        let mut a: Array2<f64> = Array2::zeros((2, 2));
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    a[[i, j]] += l[[i, k]] * l[[j, k]];
                }
            }
        }
        let inv = ndarray_cholesky::inverse(&a).unwrap();
        let mut identity: Array2<f64> = Array2::zeros((2, 2));
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    identity[[i, j]] += a[[i, k]] * inv[[k, j]];
                }
            }
        }
        assert_relative_eq!(identity[[0, 0]], 1.0, epsilon = 1e-10);
        assert_relative_eq!(identity[[1, 1]], 1.0, epsilon = 1e-10);
    }
}
