//! Schur decomposition over ndarray matrices.

use std::fmt;

use nabled_core::scalar::NabledReal;
use ndarray::{Array2, ArrayView2};
use num_complex::Complex64;

use crate::internal::{DenseKernelPolicy, identity};
#[cfg(all(feature = "lapack-provider", feature = "magma-system"))]
use crate::provider::magma;
use crate::qr::{self as qr, QRConfig};

#[cfg(all(feature = "lapack-provider", feature = "magma-system"))]
#[doc(hidden)]
pub trait SchurProviderScalar:
    NabledReal + ndarray_linalg::Lapack<Real = Self> + std::ops::AddAssign + magma::MagmaReal
{
}

#[cfg(all(feature = "lapack-provider", feature = "magma-system"))]
impl<T> SchurProviderScalar for T where
    T: NabledReal + ndarray_linalg::Lapack<Real = T> + std::ops::AddAssign + magma::MagmaReal
{
}

#[cfg(all(feature = "lapack-provider", not(feature = "magma-system")))]
#[doc(hidden)]
pub trait SchurProviderScalar:
    NabledReal + ndarray_linalg::Lapack<Real = Self> + std::ops::AddAssign
{
}

#[cfg(all(feature = "lapack-provider", not(feature = "magma-system")))]
impl<T> SchurProviderScalar for T where
    T: NabledReal + ndarray_linalg::Lapack<Real = T> + std::ops::AddAssign
{
}

/// Error type for Schur decomposition.
#[derive(Debug, Clone, PartialEq)]
pub enum SchurError {
    /// Matrix is empty.
    EmptyMatrix,
    /// Matrix must be square.
    NotSquare,
    /// Iterative algorithm failed to converge.
    ConvergenceFailed,
    /// Numerical instability detected.
    NumericalInstability,
    /// Invalid input.
    InvalidInput(String),
}

impl fmt::Display for SchurError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SchurError::EmptyMatrix => write!(f, "Matrix is empty"),
            SchurError::NotSquare => write!(f, "Matrix must be square"),
            SchurError::ConvergenceFailed => write!(f, "Schur decomposition failed to converge"),
            SchurError::NumericalInstability => write!(f, "Numerical instability detected"),
            SchurError::InvalidInput(message) => write!(f, "Invalid input: {message}"),
        }
    }
}

impl std::error::Error for SchurError {}

/// Schur decomposition result.
#[derive(Debug, Clone)]
pub struct NdarraySchurResult<T: NabledReal = f64> {
    /// Orthogonal matrix Q.
    pub q: Array2<T>,
    /// Upper triangular matrix T.
    pub t: Array2<T>,
}

/// Complex Schur decomposition result.
#[derive(Debug, Clone)]
pub struct NdarrayComplexSchurResult {
    /// Unitary matrix `Q`.
    pub q: Array2<Complex64>,
    /// Upper triangular matrix `T`.
    pub t: Array2<Complex64>,
}

/// Reusable workspace for Schur decomposition `_into` kernels.
#[derive(Debug, Clone)]
pub struct SchurWorkspace<T: NabledReal = f64> {
    q_scratch: Array2<T>,
    t_scratch: Array2<T>,
}

impl<T: NabledReal> SchurWorkspace<T> {
    fn ensure_square(&mut self, n: usize) {
        if self.q_scratch.dim() != (n, n) {
            self.q_scratch = Array2::<T>::zeros((n, n));
        }
        if self.t_scratch.dim() != (n, n) {
            self.t_scratch = Array2::<T>::zeros((n, n));
        }
    }
}

impl<T: NabledReal> Default for SchurWorkspace<T> {
    fn default() -> Self {
        Self { q_scratch: Array2::<T>::zeros((0, 0)), t_scratch: Array2::<T>::zeros((0, 0)) }
    }
}

/// Reusable workspace for complex Schur decomposition `_into` kernels.
#[derive(Debug, Clone, Default)]
pub struct SchurComplexWorkspace {
    q_scratch: Array2<Complex64>,
    t_scratch: Array2<Complex64>,
}

impl SchurComplexWorkspace {
    fn ensure_square(&mut self, n: usize) {
        if self.q_scratch.dim() != (n, n) {
            self.q_scratch = Array2::<Complex64>::zeros((n, n));
        }
        if self.t_scratch.dim() != (n, n) {
            self.t_scratch = Array2::<Complex64>::zeros((n, n));
        }
    }
}

fn off_diagonal_norm<T: NabledReal>(matrix: &Array2<T>) -> T {
    let n = matrix.nrows();
    let mut sum = T::zero();
    for i in 0..n {
        for j in 0..i {
            let value = matrix[[i, j]];
            sum += value * value;
        }
    }
    sum.sqrt()
}

fn off_diagonal_norm_complex(matrix: &Array2<Complex64>) -> f64 {
    let n = matrix.nrows();
    let mut sum = 0.0_f64;
    for i in 0..n {
        for j in 0..i {
            sum += matrix[[i, j]].norm_sqr();
        }
    }
    sum.sqrt()
}

fn validate_complex_square_non_empty(matrix: &ArrayView2<'_, Complex64>) -> Result<(), SchurError> {
    if matrix.is_empty() {
        return Err(SchurError::EmptyMatrix);
    }
    if matrix.nrows() != matrix.ncols() {
        return Err(SchurError::NotSquare);
    }
    if matrix.iter().any(|value| !value.re.is_finite() || !value.im.is_finite()) {
        return Err(SchurError::NumericalInstability);
    }
    Ok(())
}

fn validate_output_shapes_complex(
    matrix: &ArrayView2<'_, Complex64>,
    output_q: &Array2<Complex64>,
    output_t: &Array2<Complex64>,
) -> Result<(), SchurError> {
    let expected = (matrix.nrows(), matrix.ncols());
    if output_q.dim() != expected || output_t.dim() != expected {
        return Err(SchurError::InvalidInput(
            "output_q/output_t shapes must match input matrix shape".to_string(),
        ));
    }
    Ok(())
}

fn identity_complex(n: usize) -> Array2<Complex64> {
    let mut identity = Array2::<Complex64>::zeros((n, n));
    for i in 0..n {
        identity[[i, i]] = Complex64::new(1.0, 0.0);
    }
    identity
}

fn validate_output_shapes<T: NabledReal>(
    matrix: &ArrayView2<'_, T>,
    output_q: &Array2<T>,
    output_t: &Array2<T>,
) -> Result<(), SchurError> {
    let expected = (matrix.nrows(), matrix.ncols());
    if output_q.dim() != expected || output_t.dim() != expected {
        return Err(SchurError::InvalidInput(
            "output_q/output_t shapes must match input matrix shape".to_string(),
        ));
    }
    Ok(())
}

#[cfg(not(feature = "lapack-provider"))]
fn compute_schur_impl<T: qr::QrInternalScalar>(
    matrix: &ArrayView2<'_, T>,
) -> Result<NdarraySchurResult<T>, SchurError> {
    if matrix.is_empty() {
        return Err(SchurError::EmptyMatrix);
    }
    if matrix.nrows() != matrix.ncols() {
        return Err(SchurError::NotSquare);
    }
    if matrix.iter().any(|value| !value.is_finite()) {
        return Err(SchurError::NumericalInstability);
    }

    let n = matrix.nrows();
    let mut q_total = identity(n);
    let mut t = matrix.to_owned();
    let config = QRConfig {
        rank_tolerance: T::from_f64(DenseKernelPolicy::BASE_TOLERANCE).unwrap_or(T::epsilon()),
        max_iterations: DenseKernelPolicy::QR_MAX_ITERATIONS,
        use_pivoting:   false,
    };

    let mut converged = false;
    for _ in 0..DenseKernelPolicy::schur_iterations(config.max_iterations) {
        let qr = qr::decompose(&t, &config).map_err(|_| SchurError::ConvergenceFailed)?;
        t = qr.r.dot(&qr.q);
        q_total = q_total.dot(&qr.q);
        if off_diagonal_norm(&t) < config.rank_tolerance {
            converged = true;
            break;
        }
    }

    if !converged {
        return Err(SchurError::ConvergenceFailed);
    }

    Ok(NdarraySchurResult { q: q_total, t })
}

#[cfg(feature = "lapack-provider")]
fn compute_schur_impl<T>(matrix: &ArrayView2<'_, T>) -> Result<NdarraySchurResult<T>, SchurError>
where
    T: SchurProviderScalar,
{
    if matrix.is_empty() {
        return Err(SchurError::EmptyMatrix);
    }
    if matrix.nrows() != matrix.ncols() {
        return Err(SchurError::NotSquare);
    }
    if matrix.iter().any(|value| !value.is_finite()) {
        return Err(SchurError::NumericalInstability);
    }

    let n = matrix.nrows();
    let mut q_total = identity(n);
    let mut t = matrix.to_owned();
    let config = QRConfig {
        rank_tolerance: T::from_f64(DenseKernelPolicy::BASE_TOLERANCE).unwrap_or(T::epsilon()),
        max_iterations: DenseKernelPolicy::QR_MAX_ITERATIONS,
        use_pivoting:   false,
    };

    let mut converged = false;
    for _ in 0..DenseKernelPolicy::schur_iterations(config.max_iterations) {
        let qr = qr::decompose(&t, &config).map_err(|_| SchurError::ConvergenceFailed)?;
        t = qr.r.dot(&qr.q);
        q_total = q_total.dot(&qr.q);
        if off_diagonal_norm(&t) < config.rank_tolerance {
            converged = true;
            break;
        }
    }

    if !converged {
        return Err(SchurError::ConvergenceFailed);
    }

    Ok(NdarraySchurResult { q: q_total, t })
}

fn compute_schur_complex_impl(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<NdarrayComplexSchurResult, SchurError> {
    validate_complex_square_non_empty(matrix)?;
    let n = matrix.nrows();
    let mut q_total = identity_complex(n);
    let mut t = matrix.to_owned();
    let config = QRConfig::default();

    let mut converged = false;
    for _ in 0..DenseKernelPolicy::schur_iterations(config.max_iterations) {
        let qr = qr::decompose_complex(&t, &config).map_err(|_| SchurError::ConvergenceFailed)?;
        t = qr.r.dot(&qr.q);
        q_total = q_total.dot(&qr.q);
        if off_diagonal_norm_complex(&t) < DenseKernelPolicy::rank_tolerance(config.rank_tolerance)
        {
            converged = true;
            break;
        }
    }

    if !converged {
        return Err(SchurError::ConvergenceFailed);
    }
    Ok(NdarrayComplexSchurResult { q: q_total, t })
}

/// Compute Schur decomposition `A = Q T Q^T`.
///
/// # Errors
/// Returns an error for invalid input or convergence failure.
#[cfg(feature = "lapack-provider")]
pub fn compute_schur<T>(matrix: &Array2<T>) -> Result<NdarraySchurResult<T>, SchurError>
where
    T: SchurProviderScalar,
{
    compute_schur_impl(&matrix.view())
}

/// Compute Schur decomposition `A = Q T Q^T`.
///
/// # Errors
/// Returns an error for invalid input or convergence failure.
#[cfg(not(feature = "lapack-provider"))]
pub fn compute_schur<T: qr::QrInternalScalar>(
    matrix: &Array2<T>,
) -> Result<NdarraySchurResult<T>, SchurError> {
    compute_schur_impl(&matrix.view())
}

/// Compute complex Schur decomposition `A = Q T Q^H`.
///
/// # Errors
/// Returns an error for invalid input or convergence failure.
pub fn compute_schur_complex(
    matrix: &Array2<Complex64>,
) -> Result<NdarrayComplexSchurResult, SchurError> {
    compute_schur_complex_impl(&matrix.view())
}

/// Compute Schur decomposition `A = Q T Q^T` from a matrix view.
///
/// # Errors
/// Returns an error for invalid input or convergence failure.
#[cfg(feature = "lapack-provider")]
pub fn compute_schur_view<T>(
    matrix: &ArrayView2<'_, T>,
) -> Result<NdarraySchurResult<T>, SchurError>
where
    T: SchurProviderScalar,
{
    compute_schur_impl(matrix)
}

/// Compute Schur decomposition `A = Q T Q^T` from a matrix view.
///
/// # Errors
/// Returns an error for invalid input or convergence failure.
#[cfg(not(feature = "lapack-provider"))]
pub fn compute_schur_view<T: qr::QrInternalScalar>(
    matrix: &ArrayView2<'_, T>,
) -> Result<NdarraySchurResult<T>, SchurError> {
    compute_schur_impl(matrix)
}

/// Compute complex Schur decomposition `A = Q T Q^H` from a matrix view.
///
/// # Errors
/// Returns an error for invalid input or convergence failure.
pub fn compute_schur_complex_view(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<NdarrayComplexSchurResult, SchurError> {
    compute_schur_complex_impl(matrix)
}

/// Compute Schur decomposition into caller-provided outputs.
///
/// # Errors
/// Returns an error for invalid inputs, output shapes, or convergence failure.
#[cfg(feature = "lapack-provider")]
pub fn compute_schur_into<T>(
    matrix: &Array2<T>,
    output_q: &mut Array2<T>,
    output_t: &mut Array2<T>,
) -> Result<(), SchurError>
where
    T: SchurProviderScalar,
{
    let mut workspace = SchurWorkspace::default();
    compute_schur_with_workspace_into(matrix, output_q, output_t, &mut workspace)
}

/// Compute Schur decomposition into caller-provided outputs.
///
/// # Errors
/// Returns an error for invalid inputs, output shapes, or convergence failure.
#[cfg(not(feature = "lapack-provider"))]
pub fn compute_schur_into<T: qr::QrInternalScalar>(
    matrix: &Array2<T>,
    output_q: &mut Array2<T>,
    output_t: &mut Array2<T>,
) -> Result<(), SchurError> {
    let mut workspace = SchurWorkspace::default();
    compute_schur_with_workspace_into(matrix, output_q, output_t, &mut workspace)
}

/// Compute complex Schur decomposition into caller-provided outputs.
///
/// # Errors
/// Returns an error for invalid inputs, output shapes, or convergence failure.
pub fn compute_schur_complex_into(
    matrix: &Array2<Complex64>,
    output_q: &mut Array2<Complex64>,
    output_t: &mut Array2<Complex64>,
) -> Result<(), SchurError> {
    let mut workspace = SchurComplexWorkspace::default();
    compute_schur_complex_with_workspace_into(matrix, output_q, output_t, &mut workspace)
}

/// Compute Schur decomposition into caller-provided outputs from a matrix view.
///
/// # Errors
/// Returns an error for invalid inputs, output shapes, or convergence failure.
#[cfg(feature = "lapack-provider")]
pub fn compute_schur_into_view<T>(
    matrix: &ArrayView2<'_, T>,
    output_q: &mut Array2<T>,
    output_t: &mut Array2<T>,
) -> Result<(), SchurError>
where
    T: SchurProviderScalar,
{
    validate_output_shapes(matrix, output_q, output_t)?;
    let result = compute_schur_impl(matrix)?;
    output_q.assign(&result.q);
    output_t.assign(&result.t);
    Ok(())
}

/// Compute Schur decomposition into caller-provided outputs from a matrix view.
///
/// # Errors
/// Returns an error for invalid inputs, output shapes, or convergence failure.
#[cfg(not(feature = "lapack-provider"))]
pub fn compute_schur_into_view<T: qr::QrInternalScalar>(
    matrix: &ArrayView2<'_, T>,
    output_q: &mut Array2<T>,
    output_t: &mut Array2<T>,
) -> Result<(), SchurError> {
    validate_output_shapes(matrix, output_q, output_t)?;
    let result = compute_schur_impl(matrix)?;
    output_q.assign(&result.q);
    output_t.assign(&result.t);
    Ok(())
}

/// Compute complex Schur decomposition into caller-provided outputs from a view.
///
/// # Errors
/// Returns an error for invalid inputs, output shapes, or convergence failure.
pub fn compute_schur_complex_into_view(
    matrix: &ArrayView2<'_, Complex64>,
    output_q: &mut Array2<Complex64>,
    output_t: &mut Array2<Complex64>,
) -> Result<(), SchurError> {
    validate_output_shapes_complex(matrix, output_q, output_t)?;
    let result = compute_schur_complex_impl(matrix)?;
    output_q.assign(&result.q);
    output_t.assign(&result.t);
    Ok(())
}

/// Compute Schur decomposition into caller-provided outputs using reusable `workspace`.
///
/// # Errors
/// Returns an error for invalid inputs, output shapes, or convergence failure.
#[cfg(feature = "lapack-provider")]
pub fn compute_schur_with_workspace_into<T>(
    matrix: &Array2<T>,
    output_q: &mut Array2<T>,
    output_t: &mut Array2<T>,
    workspace: &mut SchurWorkspace<T>,
) -> Result<(), SchurError>
where
    T: SchurProviderScalar,
{
    validate_output_shapes(&matrix.view(), output_q, output_t)?;
    workspace.ensure_square(matrix.nrows());

    let result = compute_schur_impl(&matrix.view())?;
    workspace.q_scratch.assign(&result.q);
    workspace.t_scratch.assign(&result.t);
    output_q.assign(&workspace.q_scratch);
    output_t.assign(&workspace.t_scratch);
    Ok(())
}

/// Compute Schur decomposition into caller-provided outputs using reusable `workspace`.
///
/// # Errors
/// Returns an error for invalid inputs, output shapes, or convergence failure.
#[cfg(not(feature = "lapack-provider"))]
pub fn compute_schur_with_workspace_into<T: qr::QrInternalScalar>(
    matrix: &Array2<T>,
    output_q: &mut Array2<T>,
    output_t: &mut Array2<T>,
    workspace: &mut SchurWorkspace<T>,
) -> Result<(), SchurError> {
    validate_output_shapes(&matrix.view(), output_q, output_t)?;
    workspace.ensure_square(matrix.nrows());

    let result = compute_schur_impl(&matrix.view())?;
    workspace.q_scratch.assign(&result.q);
    workspace.t_scratch.assign(&result.t);
    output_q.assign(&workspace.q_scratch);
    output_t.assign(&workspace.t_scratch);
    Ok(())
}

/// Compute complex Schur decomposition into outputs using reusable `workspace`.
///
/// # Errors
/// Returns an error for invalid inputs, output shapes, or convergence failure.
pub fn compute_schur_complex_with_workspace_into(
    matrix: &Array2<Complex64>,
    output_q: &mut Array2<Complex64>,
    output_t: &mut Array2<Complex64>,
    workspace: &mut SchurComplexWorkspace,
) -> Result<(), SchurError> {
    validate_output_shapes_complex(&matrix.view(), output_q, output_t)?;
    workspace.ensure_square(matrix.nrows());

    let result = compute_schur_complex_impl(&matrix.view())?;
    workspace.q_scratch.assign(&result.q);
    workspace.t_scratch.assign(&result.t);
    output_q.assign(&workspace.q_scratch);
    output_t.assign(&workspace.t_scratch);
    Ok(())
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;
    use num_complex::Complex64;

    use super::*;

    #[test]
    fn schur_reconstructs_matrix() {
        let matrix =
            Array2::from_shape_vec((2, 2), vec![4.0_f64, 1.0_f64, 2.0_f64, 3.0_f64]).unwrap();
        let schur = compute_schur(&matrix).unwrap();
        let reconstructed = schur.q.dot(&schur.t).dot(&schur.q.t());
        for i in 0..2 {
            for j in 0..2 {
                assert!((matrix[[i, j]] - reconstructed[[i, j]]).abs() < 1e-6_f64);
            }
        }
    }

    #[test]
    fn schur_into_matches_allocating_path() {
        let matrix =
            Array2::from_shape_vec((2, 2), vec![5.0_f64, 2.0_f64, 1.0_f64, 4.0_f64]).unwrap();
        let expected = compute_schur(&matrix).unwrap();

        let mut q = Array2::<f64>::zeros((2, 2));
        let mut t = Array2::<f64>::zeros((2, 2));
        let mut workspace = SchurWorkspace::default();
        compute_schur_with_workspace_into(&matrix, &mut q, &mut t, &mut workspace).unwrap();

        for i in 0..2 {
            for j in 0..2 {
                assert!((q[[i, j]] - expected.q[[i, j]]).abs() < 1e-8_f64);
                assert!((t[[i, j]] - expected.t[[i, j]]).abs() < 1e-8_f64);
            }
        }
    }

    #[test]
    fn schur_rejects_invalid_inputs() {
        let empty = Array2::<f64>::zeros((0, 0));
        assert!(matches!(compute_schur(&empty), Err(SchurError::EmptyMatrix)));

        let non_square = Array2::<f64>::zeros((2, 3));
        assert!(matches!(compute_schur(&non_square), Err(SchurError::NotSquare)));

        let non_finite =
            Array2::from_shape_vec((2, 2), vec![1.0_f64, f64::NAN, 0.0_f64, 1.0_f64]).unwrap();
        assert!(matches!(compute_schur(&non_finite), Err(SchurError::NumericalInstability)));
    }

    #[test]
    fn schur_into_rejects_bad_output_shapes() {
        let matrix = Array2::eye(2);
        let mut bad_q = Array2::<f64>::zeros((1, 2));
        let mut bad_t = Array2::<f64>::zeros((2, 2));
        assert!(matches!(
            compute_schur_into(&matrix, &mut bad_q, &mut bad_t),
            Err(SchurError::InvalidInput(_))
        ));
    }

    #[test]
    fn schur_view_variants_match_owned() {
        let matrix =
            Array2::from_shape_vec((2, 2), vec![3.0_f64, 1.0_f64, 0.0_f64, 2.0_f64]).unwrap();
        let owned = compute_schur(&matrix).unwrap();
        let viewed = compute_schur_view(&matrix.view()).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!((owned.q[[i, j]] - viewed.q[[i, j]]).abs() < 1e-12_f64);
                assert!((owned.t[[i, j]] - viewed.t[[i, j]]).abs() < 1e-12_f64);
            }
        }

        let mut q = Array2::<f64>::zeros((2, 2));
        let mut t = Array2::<f64>::zeros((2, 2));
        compute_schur_into_view(&matrix.view(), &mut q, &mut t).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!((owned.q[[i, j]] - q[[i, j]]).abs() < 1e-12_f64);
                assert!((owned.t[[i, j]] - t[[i, j]]).abs() < 1e-12_f64);
            }
        }
    }

    #[test]
    fn complex_schur_reconstructs_and_view_into_match() {
        let matrix = Array2::from_shape_vec((2, 2), vec![
            Complex64::new(3.0_f64, 1.0_f64),
            Complex64::new(1.0_f64, -0.5_f64),
            Complex64::new(0.0_f64, 1.0_f64),
            Complex64::new(2.0_f64, -0.25_f64),
        ])
        .unwrap();

        let owned = compute_schur_complex(&matrix).unwrap();
        let q_h = owned.q.t().mapv(|value| value.conj());
        let reconstructed = owned.q.dot(&owned.t).dot(&q_h);
        for i in 0..matrix.nrows() {
            for j in 0..matrix.ncols() {
                assert!((reconstructed[[i, j]] - matrix[[i, j]]).norm() < 1e-7_f64);
            }
        }

        let viewed = compute_schur_complex_view(&matrix.view()).unwrap();
        for i in 0..matrix.nrows() {
            for j in 0..matrix.ncols() {
                assert!((owned.q[[i, j]] - viewed.q[[i, j]]).norm() < 1e-10_f64);
                assert!((owned.t[[i, j]] - viewed.t[[i, j]]).norm() < 1e-10_f64);
            }
        }

        let mut q = Array2::<Complex64>::zeros((2, 2));
        let mut t = Array2::<Complex64>::zeros((2, 2));
        let mut workspace = SchurComplexWorkspace::default();
        compute_schur_complex_with_workspace_into(&matrix, &mut q, &mut t, &mut workspace).unwrap();
        for i in 0..matrix.nrows() {
            for j in 0..matrix.ncols() {
                assert!((owned.q[[i, j]] - q[[i, j]]).norm() < 1e-10_f64);
                assert!((owned.t[[i, j]] - t[[i, j]]).norm() < 1e-10_f64);
            }
        }
    }
}
