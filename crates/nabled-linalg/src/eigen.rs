//! Eigenvalue decompositions over ndarray matrices.

use std::fmt;

use ndarray::{Array1, Array2, ArrayView2};
use num_complex::Complex64;

use crate::cholesky;
#[cfg(not(feature = "openblas-system"))]
use crate::internal::jacobi_eigen_symmetric;
use crate::internal::{DenseKernelPolicy, sort_eigenpairs_desc};
#[cfg(not(feature = "openblas-system"))]
use crate::schur;

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

/// Result of non-symmetric eigen decomposition (Schur-based).
#[derive(Debug, Clone)]
pub struct NdarrayNonsymmetricEigenResult {
    /// Eigenvalues.
    pub eigenvalues:   Array1<Complex64>,
    /// Schur vectors by column.
    pub schur_vectors: Array2<Complex64>,
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

fn validate_symmetric_input(matrix: &ArrayView2<'_, f64>) -> Result<(), EigenError> {
    if matrix.is_empty() {
        return Err(EigenError::EmptyMatrix);
    }
    if matrix.nrows() != matrix.ncols() {
        return Err(EigenError::NotSquare);
    }
    if matrix.iter().any(|value| !value.is_finite()) {
        return Err(EigenError::NumericalInstability);
    }
    let tolerance = DenseKernelPolicy::BASE_TOLERANCE;
    let n = matrix.nrows();
    for i in 0..n {
        for j in (i + 1)..n {
            if (matrix[[i, j]] - matrix[[j, i]]).abs() > tolerance {
                return Err(EigenError::NotSymmetric);
            }
        }
    }
    Ok(())
}

fn validate_nonsymmetric_input(matrix: &ArrayView2<'_, f64>) -> Result<(), EigenError> {
    if matrix.is_empty() {
        return Err(EigenError::EmptyMatrix);
    }
    if matrix.nrows() != matrix.ncols() {
        return Err(EigenError::NotSquare);
    }
    if matrix.iter().any(|value| !value.is_finite()) {
        return Err(EigenError::NumericalInstability);
    }
    Ok(())
}

fn validate_complex_square_finite(matrix: &ArrayView2<'_, Complex64>) -> Result<(), EigenError> {
    if matrix.is_empty() {
        return Err(EigenError::EmptyMatrix);
    }
    if matrix.nrows() != matrix.ncols() {
        return Err(EigenError::NotSquare);
    }
    if matrix.iter().any(|value| !value.re.is_finite() || !value.im.is_finite()) {
        return Err(EigenError::NumericalInstability);
    }
    Ok(())
}

#[cfg(not(feature = "openblas-system"))]
fn symmetric_internal(matrix: &ArrayView2<'_, f64>) -> Result<NdarrayEigenResult, EigenError> {
    validate_symmetric_input(matrix)?;
    let (eigenvalues, eigenvectors) = jacobi_eigen_symmetric(
        &matrix.to_owned(),
        DenseKernelPolicy::BASE_TOLERANCE,
        DenseKernelPolicy::JACOBI_MAX_ITERATIONS,
    )
    .map_err(|_| EigenError::ConvergenceFailed)?;
    let (eigenvalues, eigenvectors) = sort_eigenpairs_desc(&eigenvalues, &eigenvectors);
    Ok(NdarrayEigenResult { eigenvalues, eigenvectors })
}

#[cfg(feature = "openblas-system")]
fn symmetric_provider(matrix: &ArrayView2<'_, f64>) -> Result<NdarrayEigenResult, EigenError> {
    use ndarray_linalg::{Eigh as _, UPLO};

    validate_symmetric_input(matrix)?;
    let (eigenvalues, eigenvectors) =
        matrix.eigh(UPLO::Lower).map_err(|_| EigenError::ConvergenceFailed)?;
    let (eigenvalues, eigenvectors) = sort_eigenpairs_desc(&eigenvalues, &eigenvectors);
    Ok(NdarrayEigenResult { eigenvalues, eigenvectors })
}

#[cfg(not(feature = "openblas-system"))]
fn generalized_internal(
    matrix_a: &ArrayView2<'_, f64>,
    matrix_b: &ArrayView2<'_, f64>,
) -> Result<NdarrayGeneralizedEigenResult, EigenError> {
    validate_symmetric_input(matrix_a)?;
    validate_symmetric_input(matrix_b)?;
    if matrix_a.dim() != matrix_b.dim() {
        return Err(EigenError::InvalidDimensions);
    }

    let b_inverse = cholesky::inverse_view(matrix_b).map_err(|error| match error {
        cholesky::CholeskyError::NotPositiveDefinite => EigenError::NotPositiveDefinite,
        cholesky::CholeskyError::EmptyMatrix => EigenError::EmptyMatrix,
        cholesky::CholeskyError::NotSquare => EigenError::NotSquare,
        _ => EigenError::NumericalInstability,
    })?;

    let c = b_inverse.dot(matrix_a);
    let symmetric_c = (&c + &c.t()) * 0.5;

    let (eigenvalues, eigenvectors) = jacobi_eigen_symmetric(
        &symmetric_c,
        DenseKernelPolicy::BASE_TOLERANCE,
        DenseKernelPolicy::JACOBI_MAX_ITERATIONS,
    )
    .map_err(|_| EigenError::ConvergenceFailed)?;
    let (eigenvalues, eigenvectors) = sort_eigenpairs_desc(&eigenvalues, &eigenvectors);

    Ok(NdarrayGeneralizedEigenResult { eigenvalues, eigenvectors })
}

#[cfg(feature = "openblas-system")]
fn generalized_provider(
    matrix_a: &ArrayView2<'_, f64>,
    matrix_b: &ArrayView2<'_, f64>,
) -> Result<NdarrayGeneralizedEigenResult, EigenError> {
    validate_symmetric_input(matrix_a)?;
    validate_symmetric_input(matrix_b)?;
    if matrix_a.dim() != matrix_b.dim() {
        return Err(EigenError::InvalidDimensions);
    }

    // Reduce generalized SPD problem A x = lambda B x to a standard symmetric
    // problem via B^{-1}A, while reusing provider-backed Cholesky inverse.
    let b_inverse = cholesky::inverse_view(matrix_b).map_err(|error| match error {
        cholesky::CholeskyError::NotPositiveDefinite => EigenError::NotPositiveDefinite,
        cholesky::CholeskyError::EmptyMatrix => EigenError::EmptyMatrix,
        cholesky::CholeskyError::NotSquare => EigenError::NotSquare,
        _ => EigenError::NumericalInstability,
    })?;
    let c = b_inverse.dot(matrix_a);
    let symmetric_c = (&c + &c.t()) * 0.5;
    let NdarrayEigenResult { eigenvalues, eigenvectors } = symmetric_provider(&symmetric_c.view())?;
    Ok(NdarrayGeneralizedEigenResult { eigenvalues, eigenvectors })
}

#[cfg(not(feature = "openblas-system"))]
fn nonsymmetric_complex_internal(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<NdarrayNonsymmetricEigenResult, EigenError> {
    validate_complex_square_finite(matrix)?;
    let dimension = matrix.nrows();

    if dimension == 1 {
        let mut eigenvalues = Array1::<Complex64>::zeros(1);
        eigenvalues[0] = matrix[[0, 0]];
        let mut schur_vectors = Array2::<Complex64>::zeros((1, 1));
        schur_vectors[[0, 0]] = Complex64::new(1.0, 0.0);
        return Ok(NdarrayNonsymmetricEigenResult { eigenvalues, schur_vectors });
    }

    // Closed-form quadratic solve stabilizes the common 2x2 path and avoids
    // relying on iterative Schur convergence for tiny inputs.
    if dimension == 2 {
        let m00 = matrix[[0, 0]];
        let m01 = matrix[[0, 1]];
        let m10 = matrix[[1, 0]];
        let m11 = matrix[[1, 1]];
        let trace = m00 + m11;
        let determinant = m00 * m11 - m01 * m10;
        let discriminant = trace * trace - Complex64::new(4.0, 0.0) * determinant;
        let root = discriminant.sqrt();

        let mut eigenvalues = Array1::<Complex64>::zeros(2);
        eigenvalues[0] = (trace + root) / 2.0;
        eigenvalues[1] = (trace - root) / 2.0;

        let mut schur_vectors = Array2::<Complex64>::zeros((2, 2));
        for i in 0..2 {
            let lambda = eigenvalues[i];
            let candidate =
                if m01.norm() >= m10.norm() { [m01, lambda - m00] } else { [lambda - m11, m10] };
            let norm = (candidate[0].norm_sqr() + candidate[1].norm_sqr()).sqrt().max(f64::EPSILON);
            schur_vectors[[0, i]] = candidate[0] / norm;
            schur_vectors[[1, i]] = candidate[1] / norm;
        }
        return Ok(NdarrayNonsymmetricEigenResult { eigenvalues, schur_vectors });
    }

    let decomposition =
        schur::compute_schur_complex_view(matrix).map_err(|_| EigenError::ConvergenceFailed)?;
    let mut eigenvalues = Array1::<Complex64>::zeros(dimension);
    for i in 0..dimension {
        eigenvalues[i] = decomposition.t[[i, i]];
    }
    Ok(NdarrayNonsymmetricEigenResult { eigenvalues, schur_vectors: decomposition.q })
}

#[cfg(feature = "openblas-system")]
fn nonsymmetric_provider(
    matrix: &ArrayView2<'_, f64>,
) -> Result<NdarrayNonsymmetricEigenResult, EigenError> {
    use ndarray_linalg::Eig as _;

    validate_nonsymmetric_input(matrix)?;
    let (eigenvalues, right_eigenvectors) =
        matrix.eig().map_err(|_| EigenError::ConvergenceFailed)?;
    Ok(NdarrayNonsymmetricEigenResult { eigenvalues, schur_vectors: right_eigenvectors })
}

#[cfg(feature = "openblas-system")]
fn nonsymmetric_complex_provider(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<NdarrayNonsymmetricEigenResult, EigenError> {
    use ndarray_linalg::Eig as _;

    validate_complex_square_finite(matrix)?;
    let (eigenvalues, right_eigenvectors) =
        matrix.eig().map_err(|_| EigenError::ConvergenceFailed)?;
    Ok(NdarrayNonsymmetricEigenResult { eigenvalues, schur_vectors: right_eigenvectors })
}

/// Compute symmetric eigen decomposition.
///
/// # Errors
/// Returns an error for non-symmetric input or convergence failure.
pub fn symmetric(matrix: &Array2<f64>) -> Result<NdarrayEigenResult, EigenError> {
    symmetric_impl(&matrix.view())
}

fn symmetric_impl(matrix: &ArrayView2<'_, f64>) -> Result<NdarrayEigenResult, EigenError> {
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
    symmetric_impl(matrix)
}

/// Compute generalized symmetric eigen decomposition `(A, B)`.
///
/// # Errors
/// Returns an error when dimensions are incompatible or `B` is not SPD.
pub fn generalized(
    matrix_a: &Array2<f64>,
    matrix_b: &Array2<f64>,
) -> Result<NdarrayGeneralizedEigenResult, EigenError> {
    generalized_impl(&matrix_a.view(), &matrix_b.view())
}

fn generalized_impl(
    matrix_a: &ArrayView2<'_, f64>,
    matrix_b: &ArrayView2<'_, f64>,
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
    generalized_impl(matrix_a, matrix_b)
}

/// Compute non-symmetric eigen decomposition via complex Schur reduction.
///
/// # Errors
/// Returns an error for non-square, non-finite, or non-convergent inputs.
pub fn nonsymmetric(matrix: &Array2<f64>) -> Result<NdarrayNonsymmetricEigenResult, EigenError> {
    nonsymmetric_impl(&matrix.view())
}

fn nonsymmetric_impl(
    matrix: &ArrayView2<'_, f64>,
) -> Result<NdarrayNonsymmetricEigenResult, EigenError> {
    #[cfg(feature = "openblas-system")]
    {
        nonsymmetric_provider(matrix)
    }
    #[cfg(not(feature = "openblas-system"))]
    {
        validate_nonsymmetric_input(matrix)?;
        let matrix_complex = matrix.mapv(|value| Complex64::new(value, 0.0));
        nonsymmetric_complex_internal(&matrix_complex.view())
    }
}

/// Compute non-symmetric eigen decomposition from a real matrix view.
///
/// # Errors
/// Returns an error for non-square, non-finite, or non-convergent inputs.
pub fn nonsymmetric_view(
    matrix: &ArrayView2<'_, f64>,
) -> Result<NdarrayNonsymmetricEigenResult, EigenError> {
    nonsymmetric_impl(matrix)
}

/// Compute non-symmetric eigen decomposition for complex matrices.
///
/// # Errors
/// Returns an error for non-square, non-finite, or non-convergent inputs.
pub fn nonsymmetric_complex(
    matrix: &Array2<Complex64>,
) -> Result<NdarrayNonsymmetricEigenResult, EigenError> {
    nonsymmetric_complex_impl(&matrix.view())
}

fn nonsymmetric_complex_impl(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<NdarrayNonsymmetricEigenResult, EigenError> {
    #[cfg(feature = "openblas-system")]
    {
        nonsymmetric_complex_provider(matrix)
    }
    #[cfg(not(feature = "openblas-system"))]
    {
        nonsymmetric_complex_internal(matrix)
    }
}

/// Compute non-symmetric eigen decomposition for complex matrix views.
///
/// # Errors
/// Returns an error for non-square, non-finite, or non-convergent inputs.
pub fn nonsymmetric_complex_view(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<NdarrayNonsymmetricEigenResult, EigenError> {
    nonsymmetric_complex_impl(matrix)
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;
    use num_complex::Complex64;

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

    #[test]
    fn nonsymmetric_real_eigenvalues_cover_complex_pair() {
        let rotation = Array2::from_shape_vec((2, 2), vec![0.0, -1.0, 1.0, 0.0]).unwrap();
        let result = nonsymmetric(&rotation).unwrap();
        assert_eq!(result.eigenvalues.len(), 2);
        let mut imag_parts = result.eigenvalues.iter().map(|value| value.im).collect::<Vec<_>>();
        imag_parts.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap());
        assert!(imag_parts[0] < -0.9);
        assert!(imag_parts[1] > 0.9);
    }

    #[test]
    fn nonsymmetric_complex_eigenvalues_match_diagonal() {
        let diagonal = Array2::from_shape_vec((2, 2), vec![
            Complex64::new(2.0, 1.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-3.0, 0.5),
        ])
        .unwrap();
        let result = nonsymmetric_complex(&diagonal).unwrap();
        assert_eq!(result.eigenvalues.len(), 2);
        assert!((result.eigenvalues[0] - Complex64::new(2.0, 1.0)).norm() < 1e-10);
        assert!((result.eigenvalues[1] - Complex64::new(-3.0, 0.5)).norm() < 1e-10);
    }

    #[test]
    fn nonsymmetric_view_variants_match_owned() {
        let matrix = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, -3.0, 4.0]).unwrap();
        let owned = nonsymmetric(&matrix).unwrap();
        let viewed = nonsymmetric_view(&matrix.view()).unwrap();
        assert_eq!(owned.eigenvalues.len(), viewed.eigenvalues.len());
        assert_eq!(owned.schur_vectors.dim(), viewed.schur_vectors.dim());

        let complex_matrix = matrix.mapv(|value| Complex64::new(value, 0.25 * value));
        let complex_owned = nonsymmetric_complex(&complex_matrix).unwrap();
        let complex_viewed = nonsymmetric_complex_view(&complex_matrix.view()).unwrap();
        assert_eq!(complex_owned.eigenvalues.len(), complex_viewed.eigenvalues.len());
        assert_eq!(complex_owned.schur_vectors.dim(), complex_viewed.schur_vectors.dim());
    }

    #[test]
    fn nonsymmetric_triangular_matrix_matches_diagonal_eigenvalues() {
        let upper_triangular =
            Array2::from_shape_vec((3, 3), vec![4.0, 1.0, 2.0, 0.0, -3.0, 5.0, 0.0, 0.0, 2.5])
                .unwrap();
        let result = nonsymmetric(&upper_triangular).unwrap();
        assert_eq!(result.eigenvalues.len(), 3);

        let mut real_parts = result.eigenvalues.iter().map(|value| value.re).collect::<Vec<_>>();
        real_parts.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap());
        assert!((real_parts[0] + 3.0).abs() < 1e-8);
        assert!((real_parts[1] - 2.5).abs() < 1e-8);
        assert!((real_parts[2] - 4.0).abs() < 1e-8);
    }
}
