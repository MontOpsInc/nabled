//! # Orthogonalization
//!
//! Gram-Schmidt process for computing orthonormal bases from a set of vectors.

use std::fmt;

use nalgebra::{DMatrix, RealField};
use ndarray::Array2;
use num_traits::Float;

/// Error types for orthogonalization
#[derive(Debug, Clone, PartialEq)]
pub enum OrthogonalizationError {
    /// Matrix is empty
    EmptyMatrix,
    /// Zero norm vector encountered (linear dependency)
    ZeroNorm,
    /// Invalid input
    InvalidInput(String),
}

impl fmt::Display for OrthogonalizationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OrthogonalizationError::EmptyMatrix => write!(f, "Matrix is empty"),
            OrthogonalizationError::ZeroNorm => {
                write!(f, "Zero norm vector encountered (linear dependency)")
            }
            OrthogonalizationError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for OrthogonalizationError {}

/// Nalgebra Gram-Schmidt orthogonalization
pub mod nalgebra_orthogonalization {
    use super::*;

    /// Modified Gram-Schmidt: orthogonalize columns of matrix, return orthonormal Q
    pub fn gram_schmidt<T: RealField + Copy + num_traits::Float>(
        matrix: &DMatrix<T>,
    ) -> Result<DMatrix<T>, OrthogonalizationError> {
        if matrix.is_empty() {
            return Err(OrthogonalizationError::EmptyMatrix);
        }

        let (rows, cols) = matrix.shape();
        let mut q = matrix.clone();

        for j in 0..cols {
            // Subtract projections onto previous columns
            for i in 0..j {
                let qi = q.column(i);
                let qj = q.column(j);
                let dot = qi.dot(&qj);
                let norm_sq = qi.norm_squared();
                if norm_sq > T::epsilon() {
                    let coef = dot / norm_sq;
                    for k in 0..rows {
                        q[(k, j)] = q[(k, j)] - coef * q[(k, i)];
                    }
                }
            }
            // Normalize
            let norm = q.column(j).norm();
            if norm < T::epsilon() {
                return Err(OrthogonalizationError::ZeroNorm);
            }
            for k in 0..rows {
                q[(k, j)] /= norm;
            }
        }
        Ok(q)
    }

    /// Classic Gram-Schmidt (less numerically stable than modified)
    pub fn gram_schmidt_classic<T: RealField + Copy + num_traits::Float>(
        matrix: &DMatrix<T>,
    ) -> Result<DMatrix<T>, OrthogonalizationError> {
        if matrix.is_empty() {
            return Err(OrthogonalizationError::EmptyMatrix);
        }

        let (rows, cols) = matrix.shape();
        let mut q = DMatrix::zeros(rows, cols);

        for j in 0..cols {
            let mut v = matrix.column(j).clone_owned();
            for i in 0..j {
                let qi = q.column(i);
                let vi = matrix.column(j);
                let dot = qi.dot(&vi);
                v -= qi * dot;
            }
            let norm = v.norm();
            if norm < T::epsilon() {
                return Err(OrthogonalizationError::ZeroNorm);
            }
            for k in 0..rows {
                q[(k, j)] = v[k] / norm;
            }
        }
        Ok(q)
    }
}

/// Ndarray Gram-Schmidt orthogonalization
pub mod ndarray_orthogonalization {
    use super::*;
    use crate::utils::{nalgebra_to_ndarray, ndarray_to_nalgebra};

    /// Modified Gram-Schmidt
    pub fn gram_schmidt<T: Float + RealField>(
        matrix: &Array2<T>,
    ) -> Result<Array2<T>, OrthogonalizationError> {
        let nalg = ndarray_to_nalgebra(matrix);
        let result = super::nalgebra_orthogonalization::gram_schmidt(&nalg)?;
        Ok(nalgebra_to_ndarray(&result))
    }

    /// Classic Gram-Schmidt
    pub fn gram_schmidt_classic<T: Float + RealField>(
        matrix: &Array2<T>,
    ) -> Result<Array2<T>, OrthogonalizationError> {
        let nalg = ndarray_to_nalgebra(matrix);
        let result = super::nalgebra_orthogonalization::gram_schmidt_classic(&nalg)?;
        Ok(nalgebra_to_ndarray(&result))
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn test_nalgebra_gram_schmidt_orthonormal() {
        let a = DMatrix::from_row_slice(3, 3, &[1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0]);
        let q = nalgebra_orthogonalization::gram_schmidt(&a).unwrap();
        // Check columns are orthonormal
        for i in 0..3 {
            assert_relative_eq!(q.column(i).norm(), 1.0, epsilon = 1e-10);
            for j in (i + 1)..3 {
                assert_relative_eq!(q.column(i).dot(&q.column(j)), 0.0, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_ndarray_gram_schmidt() {
        let a = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 1.0, 1.0]).unwrap();
        let q = ndarray_orthogonalization::gram_schmidt(&a).unwrap();
        assert_eq!(q.shape(), &[2, 2]);
    }
}
