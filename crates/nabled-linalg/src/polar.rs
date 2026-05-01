//! Polar decomposition over ndarray matrices.

use std::fmt;

use nabled_core::scalar::NabledReal;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_complex::Complex64;

#[cfg(not(feature = "lapack-provider"))]
use crate::internal::DenseKernelPolicy;
#[cfg(not(feature = "lapack-provider"))]
use crate::lu;
use crate::svd;

/// Result of polar decomposition `A = U P`.
#[derive(Debug, Clone)]
pub struct NdarrayPolarResult<T: NabledReal = f64> {
    /// Orthogonal/unitary factor.
    pub u: Array2<T>,
    /// Symmetric positive-semidefinite factor.
    pub p: Array2<T>,
}

/// Result of complex polar decomposition `A = U P`.
#[derive(Debug, Clone)]
pub struct NdarrayComplexPolarResult {
    /// Unitary factor.
    pub u: Array2<Complex64>,
    /// Hermitian positive-semidefinite factor.
    pub p: Array2<Complex64>,
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

fn validate_square_non_empty_view<T>(matrix: &ArrayView2<'_, T>) -> Result<(), PolarError> {
    if matrix.is_empty() {
        return Err(PolarError::EmptyMatrix);
    }
    if matrix.nrows() != matrix.ncols() {
        return Err(PolarError::NotSquare);
    }
    Ok(())
}

fn validate_finite_view<T: NabledReal>(matrix: &ArrayView2<'_, T>) -> Result<(), PolarError> {
    if matrix.iter().any(|value| !value.is_finite()) {
        return Err(PolarError::NumericalInstability);
    }
    Ok(())
}

fn validate_svd_factor_dimensions<T>(
    u: &ArrayView2<'_, T>,
    singular_values: &ArrayView1<'_, T>,
    vt: &ArrayView2<'_, T>,
) -> Result<(), PolarError> {
    if u.is_empty() || singular_values.is_empty() || vt.is_empty() {
        return Err(PolarError::EmptyMatrix);
    }
    if u.nrows() != vt.ncols() {
        return Err(PolarError::NotSquare);
    }
    if u.ncols() != singular_values.len() || vt.nrows() != singular_values.len() {
        return Err(PolarError::DecompositionFailed);
    }
    Ok(())
}

fn validate_complex_svd_factor_dimensions(
    u: &ArrayView2<'_, Complex64>,
    singular_values: &ArrayView1<'_, f64>,
    vt: &ArrayView2<'_, Complex64>,
) -> Result<(), PolarError> {
    if u.is_empty() || singular_values.is_empty() || vt.is_empty() {
        return Err(PolarError::EmptyMatrix);
    }
    if u.nrows() != vt.ncols() {
        return Err(PolarError::NotSquare);
    }
    if u.ncols() != singular_values.len() || vt.nrows() != singular_values.len() {
        return Err(PolarError::DecompositionFailed);
    }
    Ok(())
}

fn validate_complex_square_non_empty(matrix: &ArrayView2<'_, Complex64>) -> Result<(), PolarError> {
    if matrix.is_empty() {
        return Err(PolarError::EmptyMatrix);
    }
    if matrix.nrows() != matrix.ncols() {
        return Err(PolarError::NotSquare);
    }
    Ok(())
}

fn validate_complex_finite(matrix: &ArrayView2<'_, Complex64>) -> Result<(), PolarError> {
    if matrix.iter().any(|value| !value.re.is_finite() || !value.im.is_finite()) {
        return Err(PolarError::NumericalInstability);
    }
    Ok(())
}

fn validate_finite_vector<T: NabledReal>(values: &ArrayView1<'_, T>) -> Result<(), PolarError> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(PolarError::NumericalInstability);
    }
    Ok(())
}

fn validate_complex_real_vector(values: &ArrayView1<'_, f64>) -> Result<(), PolarError> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(PolarError::NumericalInstability);
    }
    Ok(())
}

#[cfg(not(feature = "lapack-provider"))]
fn conjugate_transpose(matrix: &Array2<Complex64>) -> Array2<Complex64> {
    matrix.t().mapv(|value| value.conj())
}

#[cfg(not(feature = "lapack-provider"))]
fn frobenius_norm_complex(matrix: &Array2<Complex64>) -> f64 {
    matrix.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt()
}

#[cfg(not(feature = "lapack-provider"))]
fn max_abs_diff_complex(left: &Array2<Complex64>, right: &Array2<Complex64>) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(lhs, rhs)| (*lhs - *rhs).norm())
        .fold(0.0_f64, f64::max)
}

#[cfg(not(feature = "lapack-provider"))]
fn compute_polar_complex_internal(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<NdarrayComplexPolarResult, PolarError> {
    let mut unitary = matrix.to_owned();
    let max_iterations = DenseKernelPolicy::POLAR_MAX_ITERATIONS;
    let tolerance = DenseKernelPolicy::polar_convergence_tolerance();
    let half = Complex64::new(0.5, 0.0);
    let one = Complex64::new(1.0, 0.0);

    let mut converged = false;
    for _ in 0..max_iterations {
        let unitary_h = conjugate_transpose(&unitary);
        let inverse_h =
            lu::inverse_complex(&unitary_h).map_err(|_| PolarError::DecompositionFailed)?;
        let next = (&unitary + &inverse_h).mapv(|value| value * half);

        let delta = max_abs_diff_complex(&next, &unitary);
        let scale = one.re + frobenius_norm_complex(&next);
        unitary = next;
        if delta <= tolerance * scale {
            converged = true;
            break;
        }
    }

    if !converged {
        return Err(PolarError::DecompositionFailed);
    }

    let unitary_h = conjugate_transpose(&unitary);
    let mut psd = unitary_h.dot(matrix);
    let psd_h = conjugate_transpose(&psd);
    psd = (&psd + &psd_h).mapv(|value| value * half);

    Ok(NdarrayComplexPolarResult { u: unitary, p: psd })
}

/// Compute polar decomposition using SVD.
///
/// # Errors
/// Returns an error if matrix is invalid or SVD fails.
#[cfg(feature = "lapack-provider")]
pub fn compute_polar<T>(matrix: &Array2<T>) -> Result<NdarrayPolarResult<T>, PolarError>
where
    T: NabledReal + ndarray_linalg::Lapack<Real = T> + std::ops::AddAssign,
{
    compute_polar_impl(&matrix.view())
}

/// Compute polar decomposition using SVD.
///
/// # Errors
/// Returns an error if matrix is invalid or SVD fails.
#[cfg(not(feature = "lapack-provider"))]
pub fn compute_polar<T: svd::SvdInternalScalar>(
    matrix: &Array2<T>,
) -> Result<NdarrayPolarResult<T>, PolarError> {
    compute_polar_impl(&matrix.view())
}

#[cfg(feature = "lapack-provider")]
fn compute_polar_impl<T>(matrix: &ArrayView2<'_, T>) -> Result<NdarrayPolarResult<T>, PolarError>
where
    T: NabledReal + ndarray_linalg::Lapack<Real = T> + std::ops::AddAssign,
{
    validate_square_non_empty_view(matrix)?;
    validate_finite_view(matrix)?;

    let svd = svd::decompose_view(matrix).map_err(|_| PolarError::DecompositionFailed)?;
    compute_polar_from_svd_view(&svd.u.view(), &svd.singular_values.view(), &svd.vt.view())
}

#[cfg(not(feature = "lapack-provider"))]
fn compute_polar_impl<T: svd::SvdInternalScalar>(
    matrix: &ArrayView2<'_, T>,
) -> Result<NdarrayPolarResult<T>, PolarError> {
    validate_square_non_empty_view(matrix)?;
    validate_finite_view(matrix)?;

    let svd = svd::decompose_view(matrix).map_err(|_| PolarError::DecompositionFailed)?;
    compute_polar_from_svd_view(&svd.u.view(), &svd.singular_values.view(), &svd.vt.view())
}

/// Compute polar decomposition from precomputed SVD factors.
///
/// # Errors
/// Returns an error if factor dimensions are inconsistent or contain non-finite values.
pub fn compute_polar_from_svd<T>(
    u: &Array2<T>,
    singular_values: &Array1<T>,
    vt: &Array2<T>,
) -> Result<NdarrayPolarResult<T>, PolarError>
where
    T: NabledReal,
{
    compute_polar_from_svd_view(&u.view(), &singular_values.view(), &vt.view())
}

/// Compute polar decomposition from borrowed SVD factor views.
///
/// # Errors
/// Returns an error if factor dimensions are inconsistent or contain non-finite values.
pub fn compute_polar_from_svd_view<T>(
    u: &ArrayView2<'_, T>,
    singular_values: &ArrayView1<'_, T>,
    vt: &ArrayView2<'_, T>,
) -> Result<NdarrayPolarResult<T>, PolarError>
where
    T: NabledReal,
{
    validate_svd_factor_dimensions(u, singular_values, vt)?;
    validate_finite_view(u)?;
    validate_finite_vector(singular_values)?;
    validate_finite_view(vt)?;

    let column_count = vt.ncols();
    let retained_rank = singular_values.len();
    let mut sigma = Array2::<T>::zeros((retained_rank, retained_rank));
    for i in 0..retained_rank {
        sigma[[i, i]] = singular_values[i];
    }

    let orthogonal_factor = u.dot(vt);
    let psd_factor = vt.t().dot(&sigma).dot(vt);
    debug_assert_eq!(psd_factor.nrows(), column_count);

    Ok(NdarrayPolarResult { u: orthogonal_factor, p: psd_factor })
}

/// Compute polar decomposition using SVD from a matrix view.
///
/// # Errors
/// Returns an error if matrix is invalid or SVD fails.
#[cfg(feature = "lapack-provider")]
pub fn compute_polar_view<T>(
    matrix: &ArrayView2<'_, T>,
) -> Result<NdarrayPolarResult<T>, PolarError>
where
    T: NabledReal + ndarray_linalg::Lapack<Real = T> + std::ops::AddAssign,
{
    compute_polar_impl(matrix)
}

/// Compute polar decomposition using SVD from a matrix view.
///
/// # Errors
/// Returns an error if matrix is invalid or SVD fails.
#[cfg(not(feature = "lapack-provider"))]
pub fn compute_polar_view<T: svd::SvdInternalScalar>(
    matrix: &ArrayView2<'_, T>,
) -> Result<NdarrayPolarResult<T>, PolarError> {
    compute_polar_impl(matrix)
}

/// Compute complex polar decomposition using complex SVD.
///
/// # Errors
/// Returns an error if matrix is invalid or complex SVD fails.
pub fn compute_polar_complex(
    matrix: &Array2<Complex64>,
) -> Result<NdarrayComplexPolarResult, PolarError> {
    compute_polar_complex_impl(&matrix.view())
}

fn compute_polar_complex_impl(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<NdarrayComplexPolarResult, PolarError> {
    validate_complex_square_non_empty(matrix)?;
    validate_complex_finite(matrix)?;
    #[cfg(feature = "lapack-provider")]
    {
        let svd =
            svd::decompose_complex_view(matrix).map_err(|_| PolarError::DecompositionFailed)?;
        compute_polar_complex_from_svd_view(
            &svd.u.view(),
            &svd.singular_values.view(),
            &svd.vt.view(),
        )
    }
    #[cfg(not(feature = "lapack-provider"))]
    {
        compute_polar_complex_internal(matrix)
    }
}

/// Compute complex polar decomposition from a matrix view.
///
/// # Errors
/// Returns an error if matrix is invalid or complex SVD fails.
pub fn compute_polar_complex_view(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<NdarrayComplexPolarResult, PolarError> {
    compute_polar_complex_impl(matrix)
}

/// Compute complex polar decomposition from precomputed complex SVD factors.
///
/// # Errors
/// Returns an error if factor dimensions are inconsistent or contain non-finite values.
pub fn compute_polar_complex_from_svd(
    u: &Array2<Complex64>,
    singular_values: &Array1<f64>,
    vt: &Array2<Complex64>,
) -> Result<NdarrayComplexPolarResult, PolarError> {
    compute_polar_complex_from_svd_view(&u.view(), &singular_values.view(), &vt.view())
}

/// Compute complex polar decomposition from borrowed complex SVD factor views.
///
/// # Errors
/// Returns an error if factor dimensions are inconsistent or contain non-finite values.
pub fn compute_polar_complex_from_svd_view(
    u: &ArrayView2<'_, Complex64>,
    singular_values: &ArrayView1<'_, f64>,
    vt: &ArrayView2<'_, Complex64>,
) -> Result<NdarrayComplexPolarResult, PolarError> {
    validate_complex_svd_factor_dimensions(u, singular_values, vt)?;
    validate_complex_finite(u)?;
    validate_complex_real_vector(singular_values)?;
    validate_complex_finite(vt)?;

    let unitary_factor = u.dot(vt);
    let retained_rank = singular_values.len();
    let mut sigma = Array2::<Complex64>::zeros((retained_rank, retained_rank));
    for i in 0..retained_rank {
        sigma[[i, i]] = Complex64::new(singular_values[i], 0.0);
    }

    let right_vectors = vt.t().mapv(|value| value.conj());
    let psd_factor = right_vectors.dot(&sigma).dot(vt);
    debug_assert_eq!(psd_factor.nrows(), u.nrows());

    Ok(NdarrayComplexPolarResult { u: unitary_factor, p: psd_factor })
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;
    use num_complex::Complex64;

    use super::*;

    #[test]
    fn polar_reconstructs_input() {
        let matrix =
            Array2::from_shape_vec((2, 2), vec![3.0_f64, 1.0_f64, 1.0_f64, 3.0_f64]).unwrap();
        let polar = compute_polar(&matrix).unwrap();
        let reconstructed = polar.u.dot(&polar.p);
        for i in 0..2 {
            for j in 0..2 {
                assert!((matrix[[i, j]] - reconstructed[[i, j]]).abs() < 1e-8_f64);
            }
        }
    }

    #[test]
    fn polar_rejects_non_square_input() {
        let matrix = Array2::from_shape_vec((2, 3), vec![
            1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64, 5.0_f64, 6.0_f64,
        ])
        .unwrap();
        let result = compute_polar(&matrix);
        assert!(matches!(result, Err(PolarError::NotSquare)));
    }

    #[test]
    fn polar_view_matches_owned() {
        let matrix =
            Array2::from_shape_vec((2, 2), vec![2.0_f64, 1.0_f64, 1.0_f64, 2.0_f64]).unwrap();
        let owned = compute_polar(&matrix).unwrap();
        let viewed = compute_polar_view(&matrix.view()).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!((owned.u[[i, j]] - viewed.u[[i, j]]).abs() < 1e-12_f64);
                assert!((owned.p[[i, j]] - viewed.p[[i, j]]).abs() < 1e-12_f64);
            }
        }
    }

    #[test]
    fn polar_rejects_empty_and_non_finite_input() {
        let empty = Array2::<f64>::zeros((0, 0));
        assert!(matches!(compute_polar(&empty), Err(PolarError::EmptyMatrix)));

        let non_finite =
            Array2::from_shape_vec((2, 2), vec![1.0_f64, f64::NAN, 0.0_f64, 1.0_f64]).unwrap();
        assert!(matches!(compute_polar(&non_finite), Err(PolarError::NumericalInstability)));
    }

    #[test]
    fn polar_from_svd_factors_matches_direct() {
        let matrix =
            Array2::from_shape_vec((2, 2), vec![3.0_f64, 1.0_f64, 1.0_f64, 3.0_f64]).unwrap();
        let direct = compute_polar(&matrix).unwrap();
        let svd = svd::decompose(&matrix).unwrap();
        let from_factors = compute_polar_from_svd(&svd.u, &svd.singular_values, &svd.vt).unwrap();

        for i in 0..2 {
            for j in 0..2 {
                assert!((direct.u[[i, j]] - from_factors.u[[i, j]]).abs() < 1e-12_f64);
                assert!((direct.p[[i, j]] - from_factors.p[[i, j]]).abs() < 1e-12_f64);
            }
        }
    }

    #[test]
    fn complex_polar_reconstructs_input_and_view_matches_owned() {
        let matrix = Array2::from_shape_vec((2, 2), vec![
            Complex64::new(2.0_f64, 0.5_f64),
            Complex64::new(0.5_f64, -0.25_f64),
            Complex64::new(-1.0_f64, 1.0_f64),
            Complex64::new(1.5_f64, -0.75_f64),
        ])
        .unwrap();

        let owned = compute_polar_complex(&matrix).unwrap();
        let viewed = compute_polar_complex_view(&matrix.view()).unwrap();
        let reconstructed = owned.u.dot(&owned.p);

        for i in 0..matrix.nrows() {
            for j in 0..matrix.ncols() {
                assert!((reconstructed[[i, j]] - matrix[[i, j]]).norm() < 1e-8_f64);
                assert!((owned.u[[i, j]] - viewed.u[[i, j]]).norm() < 1e-10_f64);
                assert!((owned.p[[i, j]] - viewed.p[[i, j]]).norm() < 1e-10_f64);
            }
        }
    }

    #[test]
    fn complex_polar_validation_errors() {
        let empty = Array2::<Complex64>::zeros((0, 0));
        assert!(matches!(compute_polar_complex(&empty), Err(PolarError::EmptyMatrix)));

        let non_square =
            Array2::from_shape_vec((1, 2), vec![Complex64::new(1.0_f64, 0.0_f64); 2]).unwrap();
        assert!(matches!(compute_polar_complex(&non_square), Err(PolarError::NotSquare)));

        let non_finite =
            Array2::from_shape_vec((1, 1), vec![Complex64::new(f64::NAN, 0.0_f64)]).unwrap();
        assert!(matches!(
            compute_polar_complex(&non_finite),
            Err(PolarError::NumericalInstability)
        ));
    }

    #[test]
    fn complex_polar_from_svd_factors_matches_direct() {
        let matrix = Array2::from_shape_vec((2, 2), vec![
            Complex64::new(2.0_f64, 0.5_f64),
            Complex64::new(0.5_f64, -0.25_f64),
            Complex64::new(-1.0_f64, 1.0_f64),
            Complex64::new(1.5_f64, -0.75_f64),
        ])
        .unwrap();
        let direct = compute_polar_complex(&matrix).unwrap();
        let svd = svd::decompose_complex(&matrix).unwrap();
        let from_factors =
            compute_polar_complex_from_svd(&svd.u, &svd.singular_values, &svd.vt).unwrap();

        for i in 0..2 {
            for j in 0..2 {
                assert!((direct.u[[i, j]] - from_factors.u[[i, j]]).norm() < 1e-12_f64);
                assert!((direct.p[[i, j]] - from_factors.p[[i, j]]).norm() < 1e-12_f64);
            }
        }
    }

    #[test]
    fn polar_error_messages_are_stable() {
        assert_eq!(PolarError::EmptyMatrix.to_string(), "Matrix cannot be empty");
        assert_eq!(PolarError::NotSquare.to_string(), "Matrix must be square");
        assert_eq!(PolarError::DecompositionFailed.to_string(), "Polar decomposition failed");
        assert_eq!(PolarError::NumericalInstability.to_string(), "Numerical instability detected");
    }

    #[cfg(feature = "magma-system")]
    #[test]
    fn tiny_complex_polar_shape_skips_magma_provider() {
        use crate::provider::magma;

        let n = 8_usize;
        let mut matrix = Array2::<Complex64>::zeros((n, n));
        let mut diagonal = 1.0_f64;
        for i in 0..n {
            matrix[[i, i]] = Complex64::new(diagonal, 0.0);
            diagonal += 0.05;
            if i + 1 < n {
                matrix[[i, i + 1]] = Complex64::new(0.02, -0.01);
                matrix[[i + 1, i]] = Complex64::new(-0.01, 0.02);
            }
        }

        magma::reset_magma_provider_call_count();
        let result = compute_polar_complex(&matrix);
        assert!(result.is_ok(), "complex polar failed for tiny shape");
        assert_eq!(
            magma::magma_provider_call_count(),
            0,
            "expected CPU path for complex polar 8x8"
        );
    }
}
