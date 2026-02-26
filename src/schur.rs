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

    /// Compute Schur decomposition A = Q T Q^H
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn compute_schur<T: RealField + Copy + Float>(
        matrix: &DMatrix<T>,
    ) -> Result<NalgebraSchurResult<T>, SchurError> {
        if matrix.is_empty() {
            return Err(SchurError::EmptyMatrix);
        }
        if !matrix.is_square() {
            return Err(SchurError::NotSquare);
        }
        if matrix.iter().any(|&x| !Float::is_finite(x)) {
            return Err(SchurError::NumericalInstability);
        }

        let eps = T::epsilon();
        let max_iter = 200;
        let schur = nalgebra::linalg::Schur::try_new(matrix.clone(), eps, max_iter)
            .ok_or(SchurError::ConvergenceFailed)?;
        let (q, t) = schur.unpack();

        Ok(NalgebraSchurResult { q, t })
    }
}

/// Ndarray Schur decomposition (via nalgebra)
pub mod ndarray_schur {
    use super::*;
    use crate::interop::{nalgebra_to_ndarray, ndarray_to_nalgebra};

    /// Compute Schur decomposition
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn compute_schur<T: Float + RealField>(
        matrix: &Array2<T>,
    ) -> Result<NdarraySchurResult<T>, SchurError> {
        let nalg = ndarray_to_nalgebra(matrix);
        let result = nalgebra_schur::compute_schur(&nalg)?;
        Ok(NdarraySchurResult {
            q: nalgebra_to_ndarray(&result.q),
            t: nalgebra_to_ndarray(&result.t),
        })
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
}
