//! Eigenvalue decompositions over ndarray matrices.

use std::fmt;

use ndarray::{Array1, Array2};

use crate::cholesky::ndarray_cholesky;
use crate::internal::{
    DEFAULT_TOLERANCE, is_symmetric, jacobi_eigen_symmetric, sort_eigenpairs_desc, validate_finite,
    validate_square_non_empty,
};

/// Result of symmetric eigen decomposition.
#[derive(Debug, Clone)]
pub struct NdarrayEigenResult {
    /// Eigenvalues.
    pub eigenvalues:  Array1<f64>,
    /// Eigenvectors by column.
    pub eigenvectors: Array2<f64>,
}

/// Result of generalized eigen decomposition.
#[derive(Debug, Clone)]
pub struct NdarrayGeneralizedEigenResult {
    /// Eigenvalues.
    pub eigenvalues:  Array1<f64>,
    /// Eigenvectors by column.
    pub eigenvectors: Array2<f64>,
}

/// Error type for eigen operations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EigenError {
    /// Matrix is empty.
    EmptyMatrix,
    /// Matrix is not square.
    NotSquare,
    /// Matrix is not symmetric when required.
    NotSymmetric,
    /// Dimensions are incompatible.
    InvalidDimensions,
    /// Matrix is not positive definite.
    NotPositiveDefinite,
    /// Convergence failure.
    ConvergenceFailed,
    /// Numerical instability.
    NumericalInstability,
}

impl fmt::Display for EigenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EigenError::EmptyMatrix => write!(f, "Matrix cannot be empty"),
            EigenError::NotSquare => write!(f, "Matrix must be square"),
            EigenError::NotSymmetric => write!(f, "Matrix must be symmetric"),
            EigenError::InvalidDimensions => write!(f, "Matrix dimensions are incompatible"),
            EigenError::NotPositiveDefinite => write!(f, "Matrix is not positive definite"),
            EigenError::ConvergenceFailed => write!(f, "Eigen solver failed to converge"),
            EigenError::NumericalInstability => write!(f, "Numerical instability detected"),
        }
    }
}

impl std::error::Error for EigenError {}

fn map_validation_error(error: &'static str) -> EigenError {
    match error {
        "empty" => EigenError::EmptyMatrix,
        "not_square" => EigenError::NotSquare,
        "non_finite" => EigenError::NumericalInstability,
        _ => EigenError::ConvergenceFailed,
    }
}

fn validate_symmetric_input(matrix: &Array2<f64>) -> Result<(), EigenError> {
    validate_square_non_empty(matrix).map_err(map_validation_error)?;
    validate_finite(matrix).map_err(map_validation_error)?;
    if !is_symmetric(matrix, DEFAULT_TOLERANCE) {
        return Err(EigenError::NotSymmetric);
    }
    Ok(())
}

fn symmetric_internal(matrix: &Array2<f64>) -> Result<NdarrayEigenResult, EigenError> {
    validate_symmetric_input(matrix)?;
    let (eigenvalues, eigenvectors) = jacobi_eigen_symmetric(matrix, DEFAULT_TOLERANCE, 256)
        .map_err(|_| EigenError::ConvergenceFailed)?;
    let (eigenvalues, eigenvectors) = sort_eigenpairs_desc(&eigenvalues, &eigenvectors);
    Ok(NdarrayEigenResult { eigenvalues, eigenvectors })
}

#[cfg(feature = "openblas-system")]
fn symmetric_provider(matrix: &Array2<f64>) -> Result<NdarrayEigenResult, EigenError> {
    // Provider-specific symmetric eigensolvers can be introduced here without changing API shape.
    symmetric_internal(matrix)
}

fn generalized_internal(
    matrix_a: &Array2<f64>,
    matrix_b: &Array2<f64>,
) -> Result<NdarrayGeneralizedEigenResult, EigenError> {
    validate_symmetric_input(matrix_a)?;
    validate_symmetric_input(matrix_b)?;
    if matrix_a.dim() != matrix_b.dim() {
        return Err(EigenError::InvalidDimensions);
    }

    let b_inverse = ndarray_cholesky::inverse(matrix_b).map_err(|error| match error {
        crate::cholesky::CholeskyError::NotPositiveDefinite => EigenError::NotPositiveDefinite,
        crate::cholesky::CholeskyError::EmptyMatrix => EigenError::EmptyMatrix,
        crate::cholesky::CholeskyError::NotSquare => EigenError::NotSquare,
        _ => EigenError::NumericalInstability,
    })?;

    let c = b_inverse.dot(matrix_a);
    let symmetric_c = (&c + &c.t()) * 0.5;

    let (eigenvalues, eigenvectors) = jacobi_eigen_symmetric(&symmetric_c, DEFAULT_TOLERANCE, 256)
        .map_err(|_| EigenError::ConvergenceFailed)?;
    let (eigenvalues, eigenvectors) = sort_eigenpairs_desc(&eigenvalues, &eigenvectors);

    Ok(NdarrayGeneralizedEigenResult { eigenvalues, eigenvectors })
}

#[cfg(feature = "openblas-system")]
fn generalized_provider(
    matrix_a: &Array2<f64>,
    matrix_b: &Array2<f64>,
) -> Result<NdarrayGeneralizedEigenResult, EigenError> {
    // Provider-specific generalized eigensolvers can be introduced here without changing API shape.
    generalized_internal(matrix_a, matrix_b)
}

/// Ndarray eigen decomposition functions.
pub mod ndarray_eigen {
    use super::*;

    /// Compute symmetric eigen decomposition.
    ///
    /// # Errors
    /// Returns an error for non-symmetric input or convergence failure.
    pub fn symmetric(matrix: &Array2<f64>) -> Result<NdarrayEigenResult, EigenError> {
        #[cfg(feature = "openblas-system")]
        {
            symmetric_provider(matrix)
        }
        #[cfg(not(feature = "openblas-system"))]
        {
            symmetric_internal(matrix)
        }
    }

    /// Compute generalized symmetric eigen decomposition `(A, B)`.
    ///
    /// # Errors
    /// Returns an error when dimensions are incompatible or `B` is not SPD.
    pub fn generalized(
        matrix_a: &Array2<f64>,
        matrix_b: &Array2<f64>,
    ) -> Result<NdarrayGeneralizedEigenResult, EigenError> {
        #[cfg(feature = "openblas-system")]
        {
            generalized_provider(matrix_a, matrix_b)
        }
        #[cfg(not(feature = "openblas-system"))]
        {
            generalized_internal(matrix_a, matrix_b)
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use super::{EigenError, ndarray_eigen};

    #[test]
    fn symmetric_eigen_reconstructs() {
        let matrix = Array2::from_shape_vec((2, 2), vec![4.0, 1.0, 1.0, 3.0]).unwrap();
        let eigen = ndarray_eigen::symmetric(&matrix).unwrap();

        let diagonal = Array2::from_diag(&eigen.eigenvalues);
        let reconstructed = eigen.eigenvectors.dot(&diagonal).dot(&eigen.eigenvectors.t());

        for i in 0..2 {
            for j in 0..2 {
                assert!((reconstructed[[i, j]] - matrix[[i, j]]).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn non_symmetric_matrix_errors() {
        let matrix = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let result = ndarray_eigen::symmetric(&matrix);
        assert!(matches!(result, Err(EigenError::NotSymmetric)));
    }
}
