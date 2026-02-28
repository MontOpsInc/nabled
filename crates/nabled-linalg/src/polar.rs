//! Polar decomposition over ndarray matrices.

use std::fmt;

use ndarray::Array2;

use crate::internal::{validate_finite, validate_square_non_empty};
use crate::svd::ndarray_svd;

/// Result of polar decomposition `A = U P`.
#[derive(Debug, Clone)]
pub struct NdarrayPolarResult {
    /// Orthogonal/unitary factor.
    pub u: Array2<f64>,
    /// Symmetric positive-semidefinite factor.
    pub p: Array2<f64>,
}

/// Error type for polar decomposition.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PolarError {
    /// Matrix is empty.
    EmptyMatrix,
    /// Matrix is not square.
    NotSquare,
    /// Decomposition failed.
    DecompositionFailed,
    /// Numerical instability.
    NumericalInstability,
}

impl fmt::Display for PolarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PolarError::EmptyMatrix => write!(f, "Matrix cannot be empty"),
            PolarError::NotSquare => write!(f, "Matrix must be square"),
            PolarError::DecompositionFailed => write!(f, "Polar decomposition failed"),
            PolarError::NumericalInstability => write!(f, "Numerical instability detected"),
        }
    }
}

impl std::error::Error for PolarError {}

/// Ndarray polar decomposition functions.
pub mod ndarray_polar {
    use super::*;

    /// Compute polar decomposition using SVD.
    ///
    /// # Errors
    /// Returns an error if matrix is invalid or SVD fails.
    pub fn compute_polar(matrix: &Array2<f64>) -> Result<NdarrayPolarResult, PolarError> {
        validate_square_non_empty(matrix).map_err(|error| match error {
            "empty" => PolarError::EmptyMatrix,
            _ => PolarError::NotSquare,
        })?;
        validate_finite(matrix).map_err(|_| PolarError::NumericalInstability)?;

        let svd = ndarray_svd::decompose(matrix).map_err(|_| PolarError::DecompositionFailed)?;

        let orthogonal_factor = svd.u.dot(&svd.vt);

        let column_count = matrix.ncols();
        let retained_rank = svd.singular_values.len();
        let mut sigma = Array2::<f64>::zeros((retained_rank, retained_rank));
        for i in 0..retained_rank {
            sigma[[i, i]] = svd.singular_values[i];
        }

        let right_vectors = svd.vt.t().to_owned();
        let psd_factor = right_vectors.dot(&sigma).dot(&svd.vt);
        debug_assert_eq!(psd_factor.nrows(), column_count);

        Ok(NdarrayPolarResult { u: orthogonal_factor, p: psd_factor })
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use super::ndarray_polar;

    #[test]
    fn polar_reconstructs_input() {
        let matrix = Array2::from_shape_vec((2, 2), vec![3.0, 1.0, 1.0, 3.0]).unwrap();
        let polar = ndarray_polar::compute_polar(&matrix).unwrap();
        let reconstructed = polar.u.dot(&polar.p);
        for i in 0..2 {
            for j in 0..2 {
                assert!((matrix[[i, j]] - reconstructed[[i, j]]).abs() < 1e-8);
            }
        }
    }

    #[test]
    fn polar_rejects_non_square_input() {
        let matrix = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let result = ndarray_polar::compute_polar(&matrix);
        assert!(matches!(result, Err(super::PolarError::NotSquare)));
    }
}
