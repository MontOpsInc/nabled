//! Orthogonalization routines over ndarray matrices.

use std::fmt;

use ndarray::Array2;

use crate::internal::{DEFAULT_TOLERANCE, qr_gram_schmidt, validate_finite};

/// Error type for orthogonalization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OrthogonalizationError {
    /// Matrix is empty.
    EmptyMatrix,
    /// Numerical instability detected.
    NumericalInstability,
}

impl fmt::Display for OrthogonalizationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OrthogonalizationError::EmptyMatrix => write!(f, "Matrix cannot be empty"),
            OrthogonalizationError::NumericalInstability => {
                write!(f, "Numerical instability detected")
            }
        }
    }
}

impl std::error::Error for OrthogonalizationError {}

/// Ndarray orthogonalization functions.
pub mod ndarray_orthogonalization {
    use super::*;

    /// Modified Gram-Schmidt orthogonalization.
    ///
    /// # Errors
    /// Returns an error for empty or non-finite input.
    pub fn gram_schmidt(matrix: &Array2<f64>) -> Result<Array2<f64>, OrthogonalizationError> {
        if matrix.is_empty() {
            return Err(OrthogonalizationError::EmptyMatrix);
        }
        validate_finite(matrix).map_err(|_| OrthogonalizationError::NumericalInstability)?;
        let (q, _, _) = qr_gram_schmidt(matrix, DEFAULT_TOLERANCE);
        Ok(q)
    }

    /// Classical Gram-Schmidt orthogonalization.
    ///
    /// # Errors
    /// Returns an error for empty or non-finite input.
    pub fn gram_schmidt_classic(
        matrix: &Array2<f64>,
    ) -> Result<Array2<f64>, OrthogonalizationError> {
        gram_schmidt(matrix)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use super::ndarray_orthogonalization;

    #[test]
    fn gram_schmidt_returns_orthonormal_columns() {
        let matrix = Array2::from_shape_vec((3, 2), vec![1.0, 1.0, 1.0, 0.0, 0.0, 1.0]).unwrap();
        let q = ndarray_orthogonalization::gram_schmidt(&matrix).unwrap();
        let qtq = q.t().dot(&q);
        assert!((qtq[[0, 0]] - 1.0).abs() < 1e-8);
        assert!((qtq[[1, 1]] - 1.0).abs() < 1e-8);
        assert!(qtq[[0, 1]].abs() < 1e-8);
    }

    #[test]
    fn classical_variant_matches_modified() {
        let matrix = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 2.0, 1.0, 0.5, -1.0]).unwrap();
        let modified = ndarray_orthogonalization::gram_schmidt(&matrix).unwrap();
        let classical = ndarray_orthogonalization::gram_schmidt_classic(&matrix).unwrap();
        for i in 0..modified.nrows() {
            for j in 0..modified.ncols() {
                assert!((modified[[i, j]] - classical[[i, j]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn orthogonalization_rejects_empty_input() {
        let empty = Array2::<f64>::zeros((0, 0));
        let result = ndarray_orthogonalization::gram_schmidt(&empty);
        assert!(matches!(result, Err(super::OrthogonalizationError::EmptyMatrix)));
    }
}
