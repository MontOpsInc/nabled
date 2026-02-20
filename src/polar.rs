//! # Polar Decomposition
//!
//! Polar decomposition: A = U P where U is orthogonal and P is symmetric positive semi-definite.
//! Computed via SVD: A = U_svd Σ V_svd^T ⇒ U = U_svd V_svd^T, P = V_svd Σ V_svd^T.

use nalgebra::{DMatrix, RealField};
use ndarray::Array2;
use num_traits::Float;
use std::fmt;

use crate::svd::{nalgebra_svd, SVDError};

/// Error types for polar decomposition
#[derive(Debug, Clone, PartialEq)]
pub enum PolarError {
    /// Matrix is empty
    EmptyMatrix,
    /// Matrix is not square
    NotSquare,
    /// SVD failed
    SVDFailed(SVDError),
    /// Numerical instability
    NumericalInstability,
}

impl fmt::Display for PolarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PolarError::EmptyMatrix => write!(f, "Matrix is empty"),
            PolarError::NotSquare => write!(f, "Matrix must be square"),
            PolarError::SVDFailed(e) => write!(f, "SVD failed: {}", e),
            PolarError::NumericalInstability => write!(f, "Numerical instability detected"),
        }
    }
}

impl std::error::Error for PolarError {}

impl From<SVDError> for PolarError {
    fn from(e: SVDError) -> Self {
        PolarError::SVDFailed(e)
    }
}

/// Polar decomposition result for nalgebra
#[derive(Debug, Clone)]
pub struct NalgebraPolarResult<T: RealField> {
    /// Orthogonal matrix U
    pub u: DMatrix<T>,
    /// Symmetric positive semi-definite matrix P
    pub p: DMatrix<T>,
}

/// Polar decomposition result for ndarray
#[derive(Debug, Clone)]
pub struct NdarrayPolarResult<T: Float> {
    /// Orthogonal matrix U
    pub u: Array2<T>,
    /// Symmetric positive semi-definite matrix P
    pub p: Array2<T>,
}

/// Nalgebra polar decomposition
pub mod nalgebra_polar {
    use super::*;

    /// Compute polar decomposition A = U P
    pub fn compute_polar<T: RealField + Copy + num_traits::Float>(
        matrix: &DMatrix<T>,
    ) -> Result<NalgebraPolarResult<T>, PolarError> {
        if matrix.is_empty() {
            return Err(PolarError::EmptyMatrix);
        }
        if !matrix.is_square() {
            return Err(PolarError::NotSquare);
        }
        if matrix.iter().any(|&x| !num_traits::Float::is_finite(x)) {
            return Err(PolarError::NumericalInstability);
        }

        let svd = nalgebra_svd::compute_svd(matrix)?;
        // U = U_svd * V_svd^T, P = V_svd * Σ * V_svd^T
        let u = &svd.u * svd.vt.transpose();
        let sigma = DMatrix::from_diagonal(&svd.singular_values);
        let v = svd.vt.transpose();
        let p = &v * &sigma * &svd.vt;

        Ok(NalgebraPolarResult { u, p })
    }
}

/// Ndarray polar decomposition (via nalgebra)
pub mod ndarray_polar {
    use super::*;
    use crate::utils::{nalgebra_to_ndarray, ndarray_to_nalgebra};

    /// Compute polar decomposition
    pub fn compute_polar<T: Float + RealField>(
        matrix: &Array2<T>,
    ) -> Result<NdarrayPolarResult<T>, PolarError> {
        let nalg = ndarray_to_nalgebra(matrix);
        let result = super::nalgebra_polar::compute_polar(&nalg)?;
        Ok(NdarrayPolarResult {
            u: nalgebra_to_ndarray(&result.u),
            p: nalgebra_to_ndarray(&result.p),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_nalgebra_polar_reconstruct() {
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let result = nalgebra_polar::compute_polar(&a).unwrap();
        let reconstructed = &result.u * &result.p;
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(reconstructed[(i, j)], a[(i, j)], epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_nalgebra_polar_u_orthogonal() {
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let result = nalgebra_polar::compute_polar(&a).unwrap();
        let identity = &result.u * result.u.transpose();
        assert_relative_eq!(identity[(0, 0)], 1.0, epsilon = 1e-10);
        assert_relative_eq!(identity[(1, 1)], 1.0, epsilon = 1e-10);
    }
}
