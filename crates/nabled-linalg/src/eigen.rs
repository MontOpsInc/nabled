//! Eigenvalue decompositions over ndarray matrices.

use std::fmt;

use ndarray::{Array1, Array2, ArrayView2};

use crate::cholesky;
#[cfg(not(feature = "openblas-system"))]
use crate::internal::jacobi_eigen_symmetric;
use crate::internal::{
    DEFAULT_TOLERANCE, is_symmetric, sort_eigenpairs_desc, validate_finite,
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

#[cfg(not(feature = "openblas-system"))]
fn symmetric_internal(matrix: &Array2<f64>) -> Result<NdarrayEigenResult, EigenError> {
    validate_symmetric_input(matrix)?;
    let (eigenvalues, eigenvectors) = jacobi_eigen_symmetric(matrix, DEFAULT_TOLERANCE, 256)
        .map_err(|_| EigenError::ConvergenceFailed)?;
    let (eigenvalues, eigenvectors) = sort_eigenpairs_desc(&eigenvalues, &eigenvectors);
    Ok(NdarrayEigenResult { eigenvalues, eigenvectors })
}

#[cfg(feature = "openblas-system")]
fn symmetric_provider(matrix: &Array2<f64>) -> Result<NdarrayEigenResult, EigenError> {
    use ndarray_linalg::{Eigh as _, UPLO};

    validate_symmetric_input(matrix)?;
    let (eigenvalues, eigenvectors) =
        matrix.eigh(UPLO::Lower).map_err(|_| EigenError::ConvergenceFailed)?;
    let (eigenvalues, eigenvectors) = sort_eigenpairs_desc(&eigenvalues, &eigenvectors);
    Ok(NdarrayEigenResult { eigenvalues, eigenvectors })
}

#[cfg(not(feature = "openblas-system"))]
fn generalized_internal(
    matrix_a: &Array2<f64>,
    matrix_b: &Array2<f64>,
) -> Result<NdarrayGeneralizedEigenResult, EigenError> {
    validate_symmetric_input(matrix_a)?;
    validate_symmetric_input(matrix_b)?;
    if matrix_a.dim() != matrix_b.dim() {
        return Err(EigenError::InvalidDimensions);
    }

    let b_inverse = cholesky::inverse(matrix_b).map_err(|error| match error {
        cholesky::CholeskyError::NotPositiveDefinite => EigenError::NotPositiveDefinite,
        cholesky::CholeskyError::EmptyMatrix => EigenError::EmptyMatrix,
        cholesky::CholeskyError::NotSquare => EigenError::NotSquare,
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
    validate_symmetric_input(matrix_a)?;
    validate_symmetric_input(matrix_b)?;
    if matrix_a.dim() != matrix_b.dim() {
        return Err(EigenError::InvalidDimensions);
    }

    // Reduce generalized SPD problem A x = lambda B x to a standard symmetric
    // problem via B^{-1}A, while reusing provider-backed Cholesky inverse.
    let b_inverse = cholesky::inverse(matrix_b).map_err(|error| match error {
        cholesky::CholeskyError::NotPositiveDefinite => EigenError::NotPositiveDefinite,
        cholesky::CholeskyError::EmptyMatrix => EigenError::EmptyMatrix,
        cholesky::CholeskyError::NotSquare => EigenError::NotSquare,
        _ => EigenError::NumericalInstability,
    })?;
    let c = b_inverse.dot(matrix_a);
    let symmetric_c = (&c + &c.t()) * 0.5;
    let NdarrayEigenResult { eigenvalues, eigenvectors } = symmetric_provider(&symmetric_c)?;
    Ok(NdarrayGeneralizedEigenResult { eigenvalues, eigenvectors })
}

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

/// Compute symmetric eigen decomposition from a matrix view.
///
/// # Errors
/// Returns an error for non-symmetric input or convergence failure.
pub fn symmetric_view(matrix: &ArrayView2<'_, f64>) -> Result<NdarrayEigenResult, EigenError> {
    symmetric(&matrix.to_owned())
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

/// Compute generalized symmetric eigen decomposition `(A, B)` from matrix views.
///
/// # Errors
/// Returns an error when dimensions are incompatible or `B` is not SPD.
pub fn generalized_view(
    matrix_a: &ArrayView2<'_, f64>,
    matrix_b: &ArrayView2<'_, f64>,
) -> Result<NdarrayGeneralizedEigenResult, EigenError> {
    generalized(&matrix_a.to_owned(), &matrix_b.to_owned())
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use super::*;

    #[test]
    fn symmetric_eigen_reconstructs() {
        let matrix = Array2::from_shape_vec((2, 2), vec![4.0, 1.0, 1.0, 3.0]).unwrap();
        let eigen = symmetric(&matrix).unwrap();

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
        let result = symmetric(&matrix);
        assert!(matches!(result, Err(EigenError::NotSymmetric)));
    }

    #[test]
    fn generalized_eigen_solves_spd_pair() {
        let a = Array2::from_shape_vec((2, 2), vec![4.0, 1.0, 1.0, 3.0]).unwrap();
        let b = Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 0.0, 1.0]).unwrap();
        let result = generalized(&a, &b).unwrap();
        assert_eq!(result.eigenvalues.len(), 2);
        assert_eq!(result.eigenvectors.dim(), (2, 2));
    }

    #[test]
    fn generalized_eigen_rejects_dimension_mismatch() {
        let a = Array2::eye(2);
        let b = Array2::eye(3);
        let result = generalized(&a, &b);
        assert!(matches!(result, Err(EigenError::InvalidDimensions)));
    }

    #[test]
    fn generalized_eigen_rejects_non_spd_b() {
        let a = Array2::eye(2);
        let b = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, -1.0]).unwrap();
        let result = generalized(&a, &b);
        assert!(matches!(result, Err(EigenError::NotPositiveDefinite)));
    }

    #[test]
    fn view_variants_match_owned() {
        let matrix = Array2::from_shape_vec((2, 2), vec![5.0, 1.0, 1.0, 4.0]).unwrap();
        let owned = symmetric(&matrix).unwrap();
        let viewed = symmetric_view(&matrix.view()).unwrap();
        assert_eq!(owned.eigenvalues.len(), viewed.eigenvalues.len());
        assert_eq!(owned.eigenvectors.dim(), viewed.eigenvectors.dim());

        let a = Array2::from_shape_vec((2, 2), vec![4.0, 1.0, 1.0, 3.0]).unwrap();
        let b = Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 0.0, 1.0]).unwrap();
        let owned_generalized = generalized(&a, &b).unwrap();
        let viewed_generalized = generalized_view(&a.view(), &b.view()).unwrap();
        assert_eq!(owned_generalized.eigenvalues.len(), viewed_generalized.eigenvalues.len());
        assert_eq!(owned_generalized.eigenvectors.dim(), viewed_generalized.eigenvectors.dim());
    }

    #[test]
    fn symmetric_eigen_rejects_empty_not_square_and_non_finite() {
        let empty = Array2::<f64>::zeros((0, 0));
        assert!(matches!(symmetric(&empty), Err(EigenError::EmptyMatrix)));

        let non_square = Array2::<f64>::zeros((2, 3));
        assert!(matches!(symmetric(&non_square), Err(EigenError::NotSquare)));

        let non_finite =
            Array2::from_shape_vec((2, 2), vec![1.0, f64::NAN, f64::NAN, 2.0]).unwrap();
        assert!(matches!(symmetric(&non_finite), Err(EigenError::NumericalInstability)));
    }

    #[test]
    fn generalized_eigen_rejects_non_symmetric_a() {
        let a = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 0.0, 1.0]).unwrap();
        let b = Array2::eye(2);
        let result = generalized(&a, &b);
        assert!(matches!(result, Err(EigenError::NotSymmetric)));
    }

    #[test]
    fn symmetric_eigenvalues_are_sorted_descending() {
        let matrix = Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 0.0, 5.0]).unwrap();
        let eigen = symmetric(&matrix).unwrap();
        assert!(eigen.eigenvalues[0] >= eigen.eigenvalues[1]);
    }
}
