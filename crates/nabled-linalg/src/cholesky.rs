//! Cholesky decomposition over ndarray matrices.

use std::fmt;

use nabled_core::scalar::NabledReal;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2};
use num_complex::Complex64;

use crate::internal::DenseKernelPolicy;
#[cfg(feature = "magma-system")]
use crate::provider::magma;
#[cfg(feature = "magma-system")]
use crate::provider::policy::MagmaProviderPolicy;

/// Result of Cholesky decomposition.
#[derive(Debug, Clone)]
pub struct NdarrayCholeskyResult<T: NabledReal> {
    /// Lower-triangular factor `L` where `A = L L^T`.
    pub l: Array2<T>,
}

/// Result of complex Cholesky decomposition.
#[derive(Debug, Clone)]
pub struct NdarrayComplexCholeskyResult {
    /// Lower-triangular factor `L` where `A = L L^H`.
    pub l: Array2<Complex64>,
}

/// Error type for Cholesky operations.
#[derive(Debug, Clone, PartialEq)]
pub enum CholeskyError {
    /// Matrix is empty.
    EmptyMatrix,
    /// Matrix is not square.
    NotSquare,
    /// Matrix is not symmetric positive definite.
    NotPositiveDefinite,
    /// Invalid input.
    InvalidInput(String),
    /// Numerical instability detected.
    NumericalInstability,
}

impl fmt::Display for CholeskyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CholeskyError::EmptyMatrix => write!(f, "Matrix cannot be empty"),
            CholeskyError::NotSquare => write!(f, "Matrix must be square"),
            CholeskyError::NotPositiveDefinite => {
                write!(f, "Matrix is not symmetric positive definite")
            }
            CholeskyError::InvalidInput(message) => write!(f, "Invalid input: {message}"),
            CholeskyError::NumericalInstability => write!(f, "Numerical instability detected"),
        }
    }
}

impl std::error::Error for CholeskyError {}

#[cfg(all(feature = "magma-system", feature = "lapack-provider"))]
#[doc(hidden)]
pub trait CholeskyProviderScalar:
    NabledReal + magma::MagmaReal + ndarray_linalg::Lapack<Real = Self>
{
}

#[cfg(all(feature = "magma-system", feature = "lapack-provider"))]
impl<T> CholeskyProviderScalar for T where
    T: NabledReal + magma::MagmaReal + ndarray_linalg::Lapack<Real = T>
{
}

#[cfg(all(feature = "magma-system", not(feature = "lapack-provider")))]
#[doc(hidden)]
pub trait CholeskyProviderScalar: NabledReal + magma::MagmaReal {}

#[cfg(all(feature = "magma-system", not(feature = "lapack-provider")))]
impl<T> CholeskyProviderScalar for T where T: NabledReal + magma::MagmaReal {}

#[cfg(all(feature = "lapack-provider", not(feature = "magma-system")))]
#[doc(hidden)]
pub trait CholeskyProviderScalar: NabledReal + ndarray_linalg::Lapack<Real = Self> {}

#[cfg(all(feature = "lapack-provider", not(feature = "magma-system")))]
impl<T> CholeskyProviderScalar for T where T: NabledReal + ndarray_linalg::Lapack<Real = T> {}

fn validate_square_finite_view<T: NabledReal>(
    matrix: &ArrayView2<'_, T>,
) -> Result<(), CholeskyError> {
    if matrix.is_empty() {
        return Err(CholeskyError::EmptyMatrix);
    }
    if matrix.nrows() != matrix.ncols() {
        return Err(CholeskyError::NotSquare);
    }
    if matrix.iter().any(|value| !value.is_finite()) {
        return Err(CholeskyError::NumericalInstability);
    }
    Ok(())
}

fn validate_complex_square_finite_view(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<(), CholeskyError> {
    if matrix.is_empty() {
        return Err(CholeskyError::EmptyMatrix);
    }
    if matrix.nrows() != matrix.ncols() {
        return Err(CholeskyError::NotSquare);
    }
    if matrix.iter().any(|value| !value.re.is_finite() || !value.im.is_finite()) {
        return Err(CholeskyError::NumericalInstability);
    }
    Ok(())
}

#[cfg(feature = "magma-system")]
fn map_cholesky_provider_error(error: &'static str) -> CholeskyError {
    match error {
        "empty" => CholeskyError::EmptyMatrix,
        "not_square" => CholeskyError::NotSquare,
        "non_finite" => CholeskyError::NumericalInstability,
        "bad_dimensions" => {
            CholeskyError::InvalidInput("RHS length must match matrix dimensions".to_string())
        }
        "not_positive_definite" => CholeskyError::NotPositiveDefinite,
        _ => CholeskyError::InvalidInput(error.to_string()),
    }
}

#[cfg(feature = "magma-system")]
fn is_magma_runtime_failure(error: &CholeskyError) -> bool {
    matches!(
        error,
        CholeskyError::InvalidInput(message)
            if message.contains("provider")
                || message.contains("invalid_dimensions")
                || message.contains("invalid_input")
    )
}

#[cfg(feature = "magma-system")]
fn should_fallback_magma_runtime(error: &CholeskyError) -> bool {
    !MagmaProviderPolicy::fail_fast_mode() && is_magma_runtime_failure(error)
}

#[cfg(not(feature = "lapack-provider"))]
fn decompose_complex_internal(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<Array2<Complex64>, CholeskyError> {
    validate_complex_square_finite_view(matrix)?;

    let n = matrix.nrows();
    let mut l = Array2::<Complex64>::zeros((n, n));

    for i in 0..n {
        for j in 0..=i {
            let mut sum = matrix[[i, j]];
            for k in 0..j {
                sum -= l[[i, k]] * l[[j, k]].conj();
            }

            if i == j {
                if sum.im.abs() > DenseKernelPolicy::BASE_TOLERANCE
                    || sum.re <= DenseKernelPolicy::BASE_TOLERANCE
                {
                    return Err(CholeskyError::NotPositiveDefinite);
                }
                l[[i, j]] = Complex64::new(sum.re.sqrt(), 0.0);
            } else {
                let diagonal = l[[j, j]];
                if diagonal.norm() <= DenseKernelPolicy::BASE_TOLERANCE {
                    return Err(CholeskyError::NotPositiveDefinite);
                }
                l[[i, j]] = sum / diagonal;
            }
        }
    }

    Ok(l)
}

#[cfg(not(feature = "lapack-provider"))]
fn decompose_internal<T: NabledReal>(
    matrix: &ArrayView2<'_, T>,
) -> Result<Array2<T>, CholeskyError> {
    validate_square_finite_view(matrix)?;

    let n = matrix.nrows();
    let mut lower = Array2::<T>::zeros((n, n));
    let tolerance = T::from_f64(DenseKernelPolicy::BASE_TOLERANCE).unwrap_or(T::epsilon());

    for i in 0..n {
        for j in 0..=i {
            let mut sum = matrix[[i, j]];
            for k in 0..j {
                sum -= lower[[i, k]] * lower[[j, k]];
            }

            if i == j {
                if sum <= tolerance {
                    return Err(CholeskyError::NotPositiveDefinite);
                }
                lower[[i, j]] = sum.sqrt();
            } else {
                let diagonal = lower[[j, j]];
                if diagonal.abs() <= tolerance {
                    return Err(CholeskyError::NotPositiveDefinite);
                }
                lower[[i, j]] = sum / diagonal;
            }
        }
    }

    Ok(lower)
}

#[cfg(all(feature = "magma-system", feature = "lapack-provider"))]
fn decompose_lapack_provider<T>(matrix: &ArrayView2<'_, T>) -> Result<Array2<T>, CholeskyError>
where
    T: CholeskyProviderScalar,
{
    use ndarray_linalg::{Cholesky as _, UPLO};

    matrix.cholesky(UPLO::Lower).map_err(|_| CholeskyError::NotPositiveDefinite)
}

#[cfg(all(feature = "magma-system", feature = "lapack-provider"))]
fn solve_lapack_provider<T>(
    matrix: &ArrayView2<'_, T>,
    rhs: &ArrayView1<'_, T>,
) -> Result<Array1<T>, CholeskyError>
where
    T: CholeskyProviderScalar,
{
    use ndarray_linalg::SolveC as _;

    matrix.solvec(rhs).map_err(|_| CholeskyError::NotPositiveDefinite)
}

#[cfg(all(feature = "magma-system", feature = "lapack-provider"))]
fn inverse_lapack_provider<T>(matrix: &ArrayView2<'_, T>) -> Result<Array2<T>, CholeskyError>
where
    T: CholeskyProviderScalar,
{
    use ndarray_linalg::InverseC as _;

    matrix.invc().map_err(|_| CholeskyError::NotPositiveDefinite)
}

#[cfg(all(feature = "magma-system", feature = "lapack-provider"))]
fn decompose_complex_lapack_provider(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<Array2<Complex64>, CholeskyError> {
    use ndarray_linalg::{Cholesky as _, UPLO};

    matrix.cholesky(UPLO::Lower).map_err(|_| CholeskyError::NotPositiveDefinite)
}

#[cfg(all(feature = "magma-system", feature = "lapack-provider"))]
fn solve_complex_lapack_provider(
    matrix: &ArrayView2<'_, Complex64>,
    rhs: &ArrayView1<'_, Complex64>,
) -> Result<Array1<Complex64>, CholeskyError> {
    use ndarray_linalg::SolveC as _;

    matrix.solvec(rhs).map_err(|_| CholeskyError::NotPositiveDefinite)
}

#[cfg(all(feature = "magma-system", feature = "lapack-provider"))]
fn inverse_complex_lapack_provider(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<Array2<Complex64>, CholeskyError> {
    use ndarray_linalg::InverseC as _;

    matrix.invc().map_err(|_| CholeskyError::NotPositiveDefinite)
}

#[cfg(feature = "magma-system")]
fn decompose_provider<T>(matrix: &ArrayView2<'_, T>) -> Result<Array2<T>, CholeskyError>
where
    T: CholeskyProviderScalar,
{
    validate_square_finite_view(matrix)?;
    if !MagmaProviderPolicy::prefer_cholesky_decomposition(matrix.nrows(), matrix.ncols()) {
        #[cfg(feature = "lapack-provider")]
        {
            return decompose_lapack_provider(matrix);
        }
        #[cfg(not(feature = "lapack-provider"))]
        {
            return decompose_internal(matrix);
        }
    }
    match magma::cholesky_decompose(matrix) {
        Ok(lower) => Ok(lower),
        Err(error) => {
            let mapped = map_cholesky_provider_error(error);
            if should_fallback_magma_runtime(&mapped) {
                #[cfg(feature = "lapack-provider")]
                {
                    return decompose_lapack_provider(matrix);
                }
                #[cfg(not(feature = "lapack-provider"))]
                {
                    return decompose_internal(matrix);
                }
            }
            Err(mapped)
        }
    }
}

#[cfg(feature = "magma-system")]
fn solve_provider<T>(
    matrix: &ArrayView2<'_, T>,
    rhs: &ArrayView1<'_, T>,
) -> Result<Array1<T>, CholeskyError>
where
    T: CholeskyProviderScalar,
{
    validate_square_finite_view(matrix)?;
    if !MagmaProviderPolicy::prefer_cholesky_decomposition(matrix.nrows(), matrix.ncols()) {
        #[cfg(feature = "lapack-provider")]
        {
            return solve_lapack_provider(matrix, rhs);
        }
        #[cfg(not(feature = "lapack-provider"))]
        {
            let lower = decompose_internal(matrix)?;
            let mut output = Array1::<T>::zeros(rhs.len());
            solve_from_factor_into_view(&lower.view(), rhs, &mut output.view_mut())?;
            return Ok(output);
        }
    }
    match magma::cholesky_solve(matrix, rhs) {
        Ok(solution) => Ok(solution),
        Err(error) => {
            let mapped = map_cholesky_provider_error(error);
            if should_fallback_magma_runtime(&mapped) {
                #[cfg(feature = "lapack-provider")]
                {
                    return solve_lapack_provider(matrix, rhs);
                }
                #[cfg(not(feature = "lapack-provider"))]
                {
                    let lower = decompose_internal(matrix)?;
                    let mut output = Array1::<T>::zeros(rhs.len());
                    solve_from_factor_into_view(&lower.view(), rhs, &mut output.view_mut())?;
                    return Ok(output);
                }
            }
            Err(mapped)
        }
    }
}

#[cfg(feature = "magma-system")]
fn inverse_provider<T>(matrix: &ArrayView2<'_, T>) -> Result<Array2<T>, CholeskyError>
where
    T: CholeskyProviderScalar,
{
    validate_square_finite_view(matrix)?;
    if !MagmaProviderPolicy::prefer_cholesky_decomposition(matrix.nrows(), matrix.ncols()) {
        #[cfg(feature = "lapack-provider")]
        {
            return inverse_lapack_provider(matrix);
        }
        #[cfg(not(feature = "lapack-provider"))]
        {
            let lower = decompose_internal(matrix)?;
            return inverse_from_factor_view(&lower.view());
        }
    }
    match magma::cholesky_inverse(matrix) {
        Ok(inverse) => Ok(inverse),
        Err(error) => {
            let mapped = map_cholesky_provider_error(error);
            if should_fallback_magma_runtime(&mapped) {
                #[cfg(feature = "lapack-provider")]
                {
                    return inverse_lapack_provider(matrix);
                }
                #[cfg(not(feature = "lapack-provider"))]
                {
                    let lower = decompose_internal(matrix)?;
                    return inverse_from_factor_view(&lower.view());
                }
            }
            Err(mapped)
        }
    }
}

#[cfg(all(feature = "lapack-provider", not(feature = "magma-system")))]
fn decompose_provider<T>(matrix: &ArrayView2<'_, T>) -> Result<Array2<T>, CholeskyError>
where
    T: CholeskyProviderScalar,
{
    use ndarray_linalg::{Cholesky as _, UPLO};

    validate_square_finite_view(matrix)?;

    matrix.cholesky(UPLO::Lower).map_err(|_| CholeskyError::NotPositiveDefinite)
}

#[cfg(all(feature = "lapack-provider", not(feature = "magma-system")))]
fn solve_provider<T>(
    matrix: &ArrayView2<'_, T>,
    rhs: &ArrayView1<'_, T>,
) -> Result<Array1<T>, CholeskyError>
where
    T: CholeskyProviderScalar,
{
    use ndarray_linalg::SolveC as _;

    validate_square_finite_view(matrix)?;
    matrix.solvec(rhs).map_err(|_| CholeskyError::NotPositiveDefinite)
}

#[cfg(all(feature = "lapack-provider", not(feature = "magma-system")))]
fn inverse_provider<T>(matrix: &ArrayView2<'_, T>) -> Result<Array2<T>, CholeskyError>
where
    T: CholeskyProviderScalar,
{
    use ndarray_linalg::InverseC as _;

    validate_square_finite_view(matrix)?;
    matrix.invc().map_err(|_| CholeskyError::NotPositiveDefinite)
}

#[cfg(feature = "magma-system")]
fn decompose_complex_provider(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<Array2<Complex64>, CholeskyError> {
    validate_complex_square_finite_view(matrix)?;
    if !MagmaProviderPolicy::prefer_cholesky_decomposition(matrix.nrows(), matrix.ncols()) {
        #[cfg(feature = "lapack-provider")]
        {
            return decompose_complex_lapack_provider(matrix);
        }
        #[cfg(not(feature = "lapack-provider"))]
        {
            return decompose_complex_internal(matrix);
        }
    }
    match magma::cholesky_decompose_complex(matrix) {
        Ok(lower) => Ok(lower),
        Err(error) => {
            let mapped = map_cholesky_provider_error(error);
            if should_fallback_magma_runtime(&mapped) {
                #[cfg(feature = "lapack-provider")]
                {
                    return decompose_complex_lapack_provider(matrix);
                }
                #[cfg(not(feature = "lapack-provider"))]
                {
                    return decompose_complex_internal(matrix);
                }
            }
            Err(mapped)
        }
    }
}

#[cfg(feature = "magma-system")]
fn solve_complex_provider(
    matrix: &ArrayView2<'_, Complex64>,
    rhs: &ArrayView1<'_, Complex64>,
) -> Result<Array1<Complex64>, CholeskyError> {
    validate_complex_square_finite_view(matrix)?;
    if rhs.len() != matrix.nrows() {
        return Err(CholeskyError::InvalidInput(
            "RHS length must match matrix dimensions".to_string(),
        ));
    }
    if !MagmaProviderPolicy::prefer_cholesky_decomposition(matrix.nrows(), matrix.ncols()) {
        #[cfg(feature = "lapack-provider")]
        {
            return solve_complex_lapack_provider(matrix, rhs);
        }
        #[cfg(not(feature = "lapack-provider"))]
        {
            let lower = decompose_complex_internal(matrix)?;
            return solve_complex_from_factor_view(&lower.view(), rhs);
        }
    }
    match magma::cholesky_solve_complex(matrix, rhs) {
        Ok(solution) => Ok(solution),
        Err(error) => {
            let mapped = map_cholesky_provider_error(error);
            if should_fallback_magma_runtime(&mapped) {
                #[cfg(feature = "lapack-provider")]
                {
                    return solve_complex_lapack_provider(matrix, rhs);
                }
                #[cfg(not(feature = "lapack-provider"))]
                {
                    let lower = decompose_complex_internal(matrix)?;
                    return solve_complex_from_factor_view(&lower.view(), rhs);
                }
            }
            Err(mapped)
        }
    }
}

#[cfg(feature = "magma-system")]
fn inverse_complex_provider(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<Array2<Complex64>, CholeskyError> {
    validate_complex_square_finite_view(matrix)?;
    if !MagmaProviderPolicy::prefer_cholesky_decomposition(matrix.nrows(), matrix.ncols()) {
        #[cfg(feature = "lapack-provider")]
        {
            return inverse_complex_lapack_provider(matrix);
        }
        #[cfg(not(feature = "lapack-provider"))]
        {
            let lower = decompose_complex_internal(matrix)?;
            let n = lower.nrows();
            let mut inverse = Array2::<Complex64>::zeros((n, n));
            for col in 0..n {
                let mut basis = Array1::<Complex64>::zeros(n);
                basis[col] = Complex64::new(1.0, 0.0);
                let solution = solve_complex_from_factor_view(&lower.view(), &basis.view())?;
                for row in 0..n {
                    inverse[[row, col]] = solution[row];
                }
            }
            return Ok(inverse);
        }
    }
    match magma::cholesky_inverse_complex(matrix) {
        Ok(inverse) => Ok(inverse),
        Err(error) => {
            let mapped = map_cholesky_provider_error(error);
            if should_fallback_magma_runtime(&mapped) {
                #[cfg(feature = "lapack-provider")]
                {
                    return inverse_complex_lapack_provider(matrix);
                }
                #[cfg(not(feature = "lapack-provider"))]
                {
                    let lower = decompose_complex_internal(matrix)?;
                    let n = lower.nrows();
                    let mut inverse = Array2::<Complex64>::zeros((n, n));
                    for col in 0..n {
                        let mut basis = Array1::<Complex64>::zeros(n);
                        basis[col] = Complex64::new(1.0, 0.0);
                        let solution =
                            solve_complex_from_factor_view(&lower.view(), &basis.view())?;
                        for row in 0..n {
                            inverse[[row, col]] = solution[row];
                        }
                    }
                    return Ok(inverse);
                }
            }
            Err(mapped)
        }
    }
}

#[cfg(all(feature = "lapack-provider", not(feature = "magma-system")))]
fn decompose_complex_provider(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<Array2<Complex64>, CholeskyError> {
    use ndarray_linalg::{Cholesky as _, UPLO};

    validate_complex_square_finite_view(matrix)?;
    matrix.cholesky(UPLO::Lower).map_err(|_| CholeskyError::NotPositiveDefinite)
}

#[cfg(all(feature = "lapack-provider", not(feature = "magma-system")))]
fn solve_complex_provider(
    matrix: &ArrayView2<'_, Complex64>,
    rhs: &ArrayView1<'_, Complex64>,
) -> Result<Array1<Complex64>, CholeskyError> {
    use ndarray_linalg::SolveC as _;

    validate_complex_square_finite_view(matrix)?;
    if rhs.len() != matrix.nrows() {
        return Err(CholeskyError::InvalidInput(
            "RHS length must match matrix dimensions".to_string(),
        ));
    }
    matrix.solvec(rhs).map_err(|_| CholeskyError::NotPositiveDefinite)
}

#[cfg(all(feature = "lapack-provider", not(feature = "magma-system")))]
fn inverse_complex_provider(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<Array2<Complex64>, CholeskyError> {
    use ndarray_linalg::InverseC as _;

    validate_complex_square_finite_view(matrix)?;
    matrix.invc().map_err(|_| CholeskyError::NotPositiveDefinite)
}

fn solve_complex_from_factor_into_impl(
    lower_factor: &ArrayView2<'_, Complex64>,
    rhs: &ArrayView1<'_, Complex64>,
    output: &mut ArrayViewMut1<'_, Complex64>,
) -> Result<(), CholeskyError> {
    let size = lower_factor.nrows();
    if lower_factor.ncols() != size {
        return Err(CholeskyError::NotSquare);
    }
    if rhs.len() != size || output.len() != size {
        return Err(CholeskyError::InvalidInput(
            "RHS/output length must match matrix dimensions".to_string(),
        ));
    }

    let mut y = Array1::<Complex64>::zeros(size);
    for i in 0..size {
        let mut sum = rhs[i];
        for j in 0..i {
            sum -= lower_factor[[i, j]] * y[j];
        }
        let diagonal = lower_factor[[i, i]];
        if diagonal.norm() <= DenseKernelPolicy::BASE_TOLERANCE {
            return Err(CholeskyError::NotPositiveDefinite);
        }
        y[i] = sum / diagonal;
    }

    let mut x = Array1::<Complex64>::zeros(size);
    for i_rev in 0..size {
        let i = size - 1 - i_rev;
        let mut sum = y[i];
        for j in (i + 1)..size {
            sum -= lower_factor[[j, i]].conj() * x[j];
        }
        let diagonal = lower_factor[[i, i]].conj();
        if diagonal.norm() <= DenseKernelPolicy::BASE_TOLERANCE {
            return Err(CholeskyError::NotPositiveDefinite);
        }
        x[i] = sum / diagonal;
    }

    output.assign(&x.view());
    Ok(())
}

#[cfg(any(feature = "lapack-provider", feature = "magma-system"))]
fn decompose_dispatch<T>(matrix: &ArrayView2<'_, T>) -> Result<Array2<T>, CholeskyError>
where
    T: CholeskyProviderScalar,
{
    decompose_provider(matrix)
}

#[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
fn decompose_dispatch<T: NabledReal>(
    matrix: &ArrayView2<'_, T>,
) -> Result<Array2<T>, CholeskyError> {
    decompose_internal(matrix)
}

fn solve_from_factor_into_impl<T: NabledReal>(
    lower_factor: &ArrayView2<'_, T>,
    rhs: &ArrayView1<'_, T>,
    output: &mut ArrayViewMut1<'_, T>,
) -> Result<(), CholeskyError> {
    let size = lower_factor.nrows();
    if lower_factor.ncols() != size {
        return Err(CholeskyError::NotSquare);
    }
    if rhs.len() != size || output.len() != size {
        return Err(CholeskyError::InvalidInput(
            "RHS/output length must match matrix dimensions".to_string(),
        ));
    }

    let tolerance = T::from_f64(DenseKernelPolicy::BASE_TOLERANCE).unwrap_or(T::epsilon());
    let mut y = Array1::<T>::zeros(size);
    for i in 0..size {
        let mut sum = rhs[i];
        for j in 0..i {
            sum -= lower_factor[[i, j]] * y[j];
        }
        let diagonal = lower_factor[[i, i]];
        if diagonal.abs() <= tolerance {
            return Err(CholeskyError::NotPositiveDefinite);
        }
        y[i] = sum / diagonal;
    }

    for i_rev in 0..size {
        let i = size - 1 - i_rev;
        let mut sum = y[i];
        for j in (i + 1)..size {
            sum -= lower_factor[[j, i]] * output[j];
        }
        let diagonal = lower_factor[[i, i]];
        if diagonal.abs() <= tolerance {
            return Err(CholeskyError::NotPositiveDefinite);
        }
        output[i] = sum / diagonal;
    }

    Ok(())
}

/// Solve `LL^T x = b` from a precomputed lower-triangular Cholesky factor.
///
/// # Errors
/// Returns an error if dimensions are incompatible or the factor is invalid.
pub fn solve_from_factor_view<T: NabledReal>(
    lower_factor: &ArrayView2<'_, T>,
    rhs: &ArrayView1<'_, T>,
) -> Result<Array1<T>, CholeskyError> {
    let mut output = Array1::<T>::zeros(rhs.len());
    solve_from_factor_into_view(lower_factor, rhs, &mut output.view_mut())?;
    Ok(output)
}

/// Solve `LL^T x = b` from a precomputed lower-triangular Cholesky factor into `output`.
///
/// # Errors
/// Returns an error if dimensions are incompatible or the factor is invalid.
pub fn solve_from_factor_into_view<T: NabledReal>(
    lower_factor: &ArrayView2<'_, T>,
    rhs: &ArrayView1<'_, T>,
    output: &mut ArrayViewMut1<'_, T>,
) -> Result<(), CholeskyError> {
    solve_from_factor_into_impl(lower_factor, rhs, output)
}

/// Solve `LL^H x = b` from a precomputed complex Cholesky factor.
///
/// # Errors
/// Returns an error if dimensions are incompatible or the factor is invalid.
pub fn solve_complex_from_factor_view(
    lower_factor: &ArrayView2<'_, Complex64>,
    rhs: &ArrayView1<'_, Complex64>,
) -> Result<Array1<Complex64>, CholeskyError> {
    let mut output = Array1::<Complex64>::zeros(rhs.len());
    solve_complex_from_factor_into_view(lower_factor, rhs, &mut output.view_mut())?;
    Ok(output)
}

/// Solve `LL^H x = b` from a precomputed complex Cholesky factor into `output`.
///
/// # Errors
/// Returns an error if dimensions are incompatible or the factor is invalid.
pub fn solve_complex_from_factor_into_view(
    lower_factor: &ArrayView2<'_, Complex64>,
    rhs: &ArrayView1<'_, Complex64>,
    output: &mut ArrayViewMut1<'_, Complex64>,
) -> Result<(), CholeskyError> {
    solve_complex_from_factor_into_impl(lower_factor, rhs, output)
}

/// Compute inverse from a precomputed lower-triangular Cholesky factor.
///
/// # Errors
/// Returns an error if the factor is invalid.
pub fn inverse_from_factor_view<T: NabledReal>(
    lower_factor: &ArrayView2<'_, T>,
) -> Result<Array2<T>, CholeskyError> {
    let n = lower_factor.nrows();
    let mut output = Array2::<T>::zeros((n, n));
    inverse_from_factor_into_view(lower_factor, &mut output.view_mut())?;
    Ok(output)
}

/// Compute inverse from a precomputed lower-triangular Cholesky factor into `output`.
///
/// # Errors
/// Returns an error if dimensions are incompatible or the factor is invalid.
pub fn inverse_from_factor_into_view<T: NabledReal>(
    lower_factor: &ArrayView2<'_, T>,
    output: &mut ArrayViewMut2<'_, T>,
) -> Result<(), CholeskyError> {
    let n = lower_factor.nrows();
    if lower_factor.ncols() != n {
        return Err(CholeskyError::NotSquare);
    }
    if output.dim() != (n, n) {
        return Err(CholeskyError::InvalidInput(
            "output shape must match lower_factor dimensions".to_string(),
        ));
    }

    for col in 0..n {
        let mut basis = Array1::<T>::zeros(n);
        basis[col] = T::one();
        let mut column = output.column_mut(col);
        solve_from_factor_into_view(lower_factor, &basis.view(), &mut column)?;
    }
    Ok(())
}

/// Compute inverse from a precomputed complex lower-triangular Cholesky factor.
///
/// # Errors
/// Returns an error if the factor is invalid.
pub fn inverse_complex_from_factor_view(
    lower_factor: &ArrayView2<'_, Complex64>,
) -> Result<Array2<Complex64>, CholeskyError> {
    let n = lower_factor.nrows();
    let mut output = Array2::<Complex64>::zeros((n, n));
    inverse_complex_from_factor_into_view(lower_factor, &mut output.view_mut())?;
    Ok(output)
}

/// Compute inverse from a precomputed complex lower-triangular Cholesky factor into `output`.
///
/// # Errors
/// Returns an error if dimensions are incompatible or the factor is invalid.
pub fn inverse_complex_from_factor_into_view(
    lower_factor: &ArrayView2<'_, Complex64>,
    output: &mut ArrayViewMut2<'_, Complex64>,
) -> Result<(), CholeskyError> {
    let n = lower_factor.nrows();
    if lower_factor.ncols() != n {
        return Err(CholeskyError::NotSquare);
    }
    if output.dim() != (n, n) {
        return Err(CholeskyError::InvalidInput(
            "output shape must match lower_factor dimensions".to_string(),
        ));
    }

    for col in 0..n {
        let mut basis = Array1::<Complex64>::zeros(n);
        basis[col] = Complex64::new(1.0, 0.0);
        let mut column = output.column_mut(col);
        solve_complex_from_factor_into_view(lower_factor, &basis.view(), &mut column)?;
    }
    Ok(())
}

/// Compute Cholesky decomposition.
///
/// # Errors
/// Returns an error if matrix is not SPD.
#[cfg(any(feature = "lapack-provider", feature = "magma-system"))]
pub fn decompose<T>(matrix: &Array2<T>) -> Result<NdarrayCholeskyResult<T>, CholeskyError>
where
    T: CholeskyProviderScalar,
{
    let l = decompose_dispatch(&matrix.view())?;
    Ok(NdarrayCholeskyResult { l })
}

/// Compute Cholesky decomposition.
///
/// # Errors
/// Returns an error if matrix is not SPD.
#[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
pub fn decompose<T: NabledReal>(
    matrix: &Array2<T>,
) -> Result<NdarrayCholeskyResult<T>, CholeskyError> {
    let l = decompose_dispatch(&matrix.view())?;
    Ok(NdarrayCholeskyResult { l })
}

/// Compute complex Cholesky decomposition.
///
/// # Errors
/// Returns an error if matrix is not Hermitian positive definite.
pub fn decompose_complex(
    matrix: &Array2<Complex64>,
) -> Result<NdarrayComplexCholeskyResult, CholeskyError> {
    decompose_complex_impl(&matrix.view())
}

fn decompose_complex_impl(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<NdarrayComplexCholeskyResult, CholeskyError> {
    #[cfg(feature = "magma-system")]
    {
        let l = decompose_complex_provider(matrix)?;
        Ok(NdarrayComplexCholeskyResult { l })
    }
    #[cfg(all(feature = "lapack-provider", not(feature = "magma-system")))]
    {
        let l = decompose_complex_provider(matrix)?;
        Ok(NdarrayComplexCholeskyResult { l })
    }
    #[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
    {
        let l = decompose_complex_internal(matrix)?;
        Ok(NdarrayComplexCholeskyResult { l })
    }
}

/// Compute Cholesky decomposition from a matrix view.
///
/// # Errors
/// Returns an error if matrix is not SPD.
#[cfg(any(feature = "lapack-provider", feature = "magma-system"))]
pub fn decompose_view<T>(
    matrix: &ArrayView2<'_, T>,
) -> Result<NdarrayCholeskyResult<T>, CholeskyError>
where
    T: CholeskyProviderScalar,
{
    let l = decompose_dispatch(matrix)?;
    Ok(NdarrayCholeskyResult { l })
}

/// Compute Cholesky decomposition from a matrix view.
///
/// # Errors
/// Returns an error if matrix is not SPD.
#[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
pub fn decompose_view<T: NabledReal>(
    matrix: &ArrayView2<'_, T>,
) -> Result<NdarrayCholeskyResult<T>, CholeskyError> {
    let l = decompose_dispatch(matrix)?;
    Ok(NdarrayCholeskyResult { l })
}

/// Compute complex Cholesky decomposition from a matrix view.
///
/// # Errors
/// Returns an error if matrix is not Hermitian positive definite or provider
/// support is unavailable.
pub fn decompose_complex_view(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<NdarrayComplexCholeskyResult, CholeskyError> {
    decompose_complex_impl(matrix)
}

/// Solve `Ax=b` using Cholesky decomposition.
///
/// # Errors
/// Returns an error for invalid dimensions or non-SPD matrix.
#[cfg(any(feature = "lapack-provider", feature = "magma-system"))]
pub fn solve<T>(matrix: &Array2<T>, rhs: &Array1<T>) -> Result<Array1<T>, CholeskyError>
where
    T: CholeskyProviderScalar,
{
    solve_impl(&matrix.view(), &rhs.view())
}

/// Solve `Ax=b` using Cholesky decomposition.
///
/// # Errors
/// Returns an error for invalid dimensions or non-SPD matrix.
#[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
pub fn solve<T: NabledReal>(
    matrix: &Array2<T>,
    rhs: &Array1<T>,
) -> Result<Array1<T>, CholeskyError> {
    solve_impl(&matrix.view(), &rhs.view())
}

#[cfg(any(feature = "lapack-provider", feature = "magma-system"))]
fn solve_impl<T>(
    matrix: &ArrayView2<'_, T>,
    rhs: &ArrayView1<'_, T>,
) -> Result<Array1<T>, CholeskyError>
where
    T: CholeskyProviderScalar,
{
    solve_provider(matrix, rhs)
}

#[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
fn solve_impl<T: NabledReal>(
    matrix: &ArrayView2<'_, T>,
    rhs: &ArrayView1<'_, T>,
) -> Result<Array1<T>, CholeskyError> {
    let mut output = Array1::<T>::zeros(rhs.len());
    solve_into_impl(matrix, rhs, &mut output)?;
    Ok(output)
}

/// Solve complex-valued `Ax=b` using Cholesky decomposition.
///
/// # Errors
/// Returns an error for invalid dimensions or non-HPD matrix.
pub fn solve_complex(
    matrix: &Array2<Complex64>,
    rhs: &Array1<Complex64>,
) -> Result<Array1<Complex64>, CholeskyError> {
    solve_complex_impl(&matrix.view(), &rhs.view())
}

fn solve_complex_impl(
    matrix: &ArrayView2<'_, Complex64>,
    rhs: &ArrayView1<'_, Complex64>,
) -> Result<Array1<Complex64>, CholeskyError> {
    #[cfg(feature = "magma-system")]
    {
        solve_complex_provider(matrix, rhs)
    }
    #[cfg(all(feature = "lapack-provider", not(feature = "magma-system")))]
    {
        solve_complex_provider(matrix, rhs)
    }
    #[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
    {
        let lower_factor = decompose_complex_internal(matrix)?;
        solve_complex_from_factor_view(&lower_factor.view(), rhs)
    }
}

/// Solve `Ax=b` using Cholesky decomposition from matrix/vector views.
///
/// # Errors
/// Returns an error for invalid dimensions or non-SPD matrix.
#[cfg(any(feature = "lapack-provider", feature = "magma-system"))]
pub fn solve_view<T>(
    matrix: &ArrayView2<'_, T>,
    rhs: &ArrayView1<'_, T>,
) -> Result<Array1<T>, CholeskyError>
where
    T: CholeskyProviderScalar,
{
    solve_impl(matrix, rhs)
}

/// Solve `Ax=b` using Cholesky decomposition from matrix/vector views.
///
/// # Errors
/// Returns an error for invalid dimensions or non-SPD matrix.
#[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
pub fn solve_view<T: NabledReal>(
    matrix: &ArrayView2<'_, T>,
    rhs: &ArrayView1<'_, T>,
) -> Result<Array1<T>, CholeskyError> {
    solve_impl(matrix, rhs)
}

/// Solve complex-valued `Ax=b` from matrix/vector views.
///
/// # Errors
/// Returns an error for invalid dimensions or non-HPD matrix.
pub fn solve_complex_view(
    matrix: &ArrayView2<'_, Complex64>,
    rhs: &ArrayView1<'_, Complex64>,
) -> Result<Array1<Complex64>, CholeskyError> {
    solve_complex_impl(matrix, rhs)
}

/// Solve `Ax=b` into `output` from matrix/vector views.
///
/// # Errors
/// Returns an error for invalid dimensions or non-SPD matrix.
#[cfg(any(feature = "lapack-provider", feature = "magma-system"))]
pub fn solve_into_view<T>(
    matrix: &ArrayView2<'_, T>,
    rhs: &ArrayView1<'_, T>,
    output: &mut ArrayViewMut1<'_, T>,
) -> Result<(), CholeskyError>
where
    T: CholeskyProviderScalar,
{
    if rhs.len() != matrix.nrows() {
        return Err(CholeskyError::InvalidInput(
            "RHS length must match matrix dimensions".to_string(),
        ));
    }
    if output.len() != rhs.len() {
        return Err(CholeskyError::InvalidInput("output length must match rhs length".to_string()));
    }

    let solution = solve_provider(matrix, rhs)?;
    output.assign(&solution.view());
    Ok(())
}

/// Solve `Ax=b` into `output` from matrix/vector views.
///
/// # Errors
/// Returns an error for invalid dimensions or non-SPD matrix.
#[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
pub fn solve_into_view<T: NabledReal>(
    matrix: &ArrayView2<'_, T>,
    rhs: &ArrayView1<'_, T>,
    output: &mut ArrayViewMut1<'_, T>,
) -> Result<(), CholeskyError> {
    if rhs.len() != matrix.nrows() {
        return Err(CholeskyError::InvalidInput(
            "RHS length must match matrix dimensions".to_string(),
        ));
    }
    if output.len() != rhs.len() {
        return Err(CholeskyError::InvalidInput("output length must match rhs length".to_string()));
    }

    let lower_factor = decompose_dispatch(matrix)?;
    solve_from_factor_into_view(&lower_factor.view(), rhs, output)
}

/// Solve complex-valued `Ax=b` into `output` from matrix/vector views.
///
/// # Errors
/// Returns an error for invalid dimensions or non-HPD matrix.
pub fn solve_complex_into_view(
    matrix: &ArrayView2<'_, Complex64>,
    rhs: &ArrayView1<'_, Complex64>,
    output: &mut ArrayViewMut1<'_, Complex64>,
) -> Result<(), CholeskyError> {
    if rhs.len() != matrix.nrows() {
        return Err(CholeskyError::InvalidInput(
            "RHS length must match matrix dimensions".to_string(),
        ));
    }
    if output.len() != rhs.len() {
        return Err(CholeskyError::InvalidInput("output length must match rhs length".to_string()));
    }

    let solution = solve_complex_impl(matrix, rhs)?;
    output.assign(&solution.view());
    Ok(())
}

/// Solve `Ax=b` into `output` using Cholesky decomposition.
///
/// # Errors
/// Returns an error for invalid dimensions or non-SPD matrix.
#[cfg(any(feature = "lapack-provider", feature = "magma-system"))]
pub fn solve_into<T>(
    matrix: &Array2<T>,
    rhs: &Array1<T>,
    output: &mut Array1<T>,
) -> Result<(), CholeskyError>
where
    T: CholeskyProviderScalar,
{
    solve_into_impl(&matrix.view(), &rhs.view(), output)
}

/// Solve `Ax=b` into `output` using Cholesky decomposition.
///
/// # Errors
/// Returns an error for invalid dimensions or non-SPD matrix.
#[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
pub fn solve_into<T: NabledReal>(
    matrix: &Array2<T>,
    rhs: &Array1<T>,
    output: &mut Array1<T>,
) -> Result<(), CholeskyError> {
    solve_into_impl(&matrix.view(), &rhs.view(), output)
}

#[cfg(any(feature = "lapack-provider", feature = "magma-system"))]
fn solve_into_impl<T>(
    matrix: &ArrayView2<'_, T>,
    rhs: &ArrayView1<'_, T>,
    output: &mut Array1<T>,
) -> Result<(), CholeskyError>
where
    T: CholeskyProviderScalar,
{
    if rhs.len() != matrix.nrows() {
        return Err(CholeskyError::InvalidInput(
            "RHS length must match matrix dimensions".to_string(),
        ));
    }
    if output.len() != rhs.len() {
        return Err(CholeskyError::InvalidInput("output length must match rhs length".to_string()));
    }

    let solution = solve_provider(matrix, rhs)?;
    output.assign(&solution);
    Ok(())
}

#[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
fn solve_into_impl<T: NabledReal>(
    matrix: &ArrayView2<'_, T>,
    rhs: &ArrayView1<'_, T>,
    output: &mut Array1<T>,
) -> Result<(), CholeskyError> {
    if rhs.len() != matrix.nrows() {
        return Err(CholeskyError::InvalidInput(
            "RHS length must match matrix dimensions".to_string(),
        ));
    }
    if output.len() != rhs.len() {
        return Err(CholeskyError::InvalidInput("output length must match rhs length".to_string()));
    }

    let lower_factor = decompose_dispatch(matrix)?;
    solve_from_factor_into_view(&lower_factor.view(), rhs, &mut output.view_mut())
}

/// Compute inverse via Cholesky decomposition.
///
/// # Errors
/// Returns an error if matrix is not SPD.
#[cfg(any(feature = "lapack-provider", feature = "magma-system"))]
pub fn inverse<T>(matrix: &Array2<T>) -> Result<Array2<T>, CholeskyError>
where
    T: CholeskyProviderScalar,
{
    inverse_impl(&matrix.view())
}

/// Compute inverse via Cholesky decomposition.
///
/// # Errors
/// Returns an error if matrix is not SPD.
#[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
pub fn inverse<T: NabledReal>(matrix: &Array2<T>) -> Result<Array2<T>, CholeskyError> {
    inverse_impl(&matrix.view())
}

#[cfg(any(feature = "lapack-provider", feature = "magma-system"))]
fn inverse_impl<T>(matrix: &ArrayView2<'_, T>) -> Result<Array2<T>, CholeskyError>
where
    T: CholeskyProviderScalar,
{
    inverse_provider(matrix)
}

#[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
fn inverse_impl<T: NabledReal>(matrix: &ArrayView2<'_, T>) -> Result<Array2<T>, CholeskyError> {
    let lower_factor = decompose_dispatch(matrix)?;
    inverse_from_factor_view(&lower_factor.view())
}

/// Compute complex inverse via Cholesky decomposition.
///
/// # Errors
/// Returns an error if matrix is not HPD.
pub fn inverse_complex(matrix: &Array2<Complex64>) -> Result<Array2<Complex64>, CholeskyError> {
    inverse_complex_impl(&matrix.view())
}

fn inverse_complex_impl(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<Array2<Complex64>, CholeskyError> {
    #[cfg(feature = "magma-system")]
    {
        inverse_complex_provider(matrix)
    }
    #[cfg(all(feature = "lapack-provider", not(feature = "magma-system")))]
    {
        inverse_complex_provider(matrix)
    }
    #[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
    {
        let lower_factor = decompose_complex_internal(matrix)?;
        let size = lower_factor.nrows();
        let mut inverse = Array2::<Complex64>::zeros((size, size));

        for col in 0..size {
            let mut basis = Array1::<Complex64>::zeros(size);
            basis[col] = Complex64::new(1.0, 0.0);
            let solution = solve_complex_from_factor_view(&lower_factor.view(), &basis.view())?;
            for row in 0..size {
                inverse[[row, col]] = solution[row];
            }
        }

        Ok(inverse)
    }
}

/// Compute inverse via Cholesky decomposition from a matrix view.
///
/// # Errors
/// Returns an error if matrix is not SPD.
#[cfg(any(feature = "lapack-provider", feature = "magma-system"))]
pub fn inverse_view<T>(matrix: &ArrayView2<'_, T>) -> Result<Array2<T>, CholeskyError>
where
    T: CholeskyProviderScalar,
{
    inverse_impl(matrix)
}

/// Compute inverse via Cholesky decomposition from a matrix view.
///
/// # Errors
/// Returns an error if matrix is not SPD.
#[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
pub fn inverse_view<T: NabledReal>(matrix: &ArrayView2<'_, T>) -> Result<Array2<T>, CholeskyError> {
    inverse_impl(matrix)
}

/// Compute complex inverse from a matrix view.
///
/// # Errors
/// Returns an error if matrix is not HPD.
pub fn inverse_complex_view(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<Array2<Complex64>, CholeskyError> {
    inverse_complex_impl(matrix)
}

/// Compute inverse via Cholesky decomposition into `output` from a matrix view.
///
/// # Errors
/// Returns an error if dimensions are incompatible or matrix is not SPD.
#[cfg(any(feature = "lapack-provider", feature = "magma-system"))]
pub fn inverse_into_view<T>(
    matrix: &ArrayView2<'_, T>,
    output: &mut ArrayViewMut2<'_, T>,
) -> Result<(), CholeskyError>
where
    T: CholeskyProviderScalar,
{
    if output.dim() != matrix.dim() {
        return Err(CholeskyError::InvalidInput(
            "output shape must match matrix dimensions".to_string(),
        ));
    }

    let inverse = inverse_provider(matrix)?;
    output.assign(&inverse.view());
    Ok(())
}

/// Compute inverse via Cholesky decomposition into `output` from a matrix view.
///
/// # Errors
/// Returns an error if dimensions are incompatible or matrix is not SPD.
#[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
pub fn inverse_into_view<T: NabledReal>(
    matrix: &ArrayView2<'_, T>,
    output: &mut ArrayViewMut2<'_, T>,
) -> Result<(), CholeskyError> {
    let lower_factor = decompose_dispatch(matrix)?;
    inverse_from_factor_into_view(&lower_factor.view(), output)
}

/// Compute complex inverse into `output` from a matrix view.
///
/// # Errors
/// Returns an error if dimensions are incompatible or matrix is not HPD.
pub fn inverse_complex_into_view(
    matrix: &ArrayView2<'_, Complex64>,
    output: &mut ArrayViewMut2<'_, Complex64>,
) -> Result<(), CholeskyError> {
    if output.dim() != matrix.dim() {
        return Err(CholeskyError::InvalidInput(
            "output shape must match matrix dimensions".to_string(),
        ));
    }

    let inverse = inverse_complex_impl(matrix)?;
    output.assign(&inverse.view());
    Ok(())
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2};
    use num_complex::Complex64;

    use super::*;

    #[test]
    fn cholesky_reconstructs_spd_matrix() {
        let matrix =
            Array2::<f64>::from_shape_vec((2, 2), vec![4.0_f64, 2.0_f64, 2.0_f64, 3.0_f64])
                .unwrap();
        let decomposition = decompose(&matrix).unwrap();
        let reconstructed = decomposition.l.dot(&decomposition.l.t());
        for i in 0..2 {
            for j in 0..2 {
                assert!((matrix[[i, j]] - reconstructed[[i, j]]).abs() < 1e-10_f64);
            }
        }
    }

    #[test]
    fn cholesky_view_variants_match_owned() {
        let matrix =
            Array2::<f64>::from_shape_vec((2, 2), vec![9.0_f64, 3.0_f64, 3.0_f64, 5.0_f64])
                .unwrap();
        let rhs = Array1::from_vec(vec![12.0_f64, 8.0_f64]);

        let owned = decompose(&matrix).unwrap();
        let viewed = decompose_view(&matrix.view()).unwrap();
        assert_eq!(owned.l.dim(), viewed.l.dim());

        let solution_owned = solve(&matrix, &rhs).unwrap();
        let solution_viewed = solve_view(&matrix.view(), &rhs.view()).unwrap();
        for i in 0..rhs.len() {
            assert!((solution_owned[i] - solution_viewed[i]).abs() < 1e-12_f64);
        }

        let inverse_owned = inverse(&matrix).unwrap();
        let inverse_viewed = inverse_view(&matrix.view()).unwrap();
        for i in 0..matrix.nrows() {
            for j in 0..matrix.ncols() {
                assert!((inverse_owned[[i, j]] - inverse_viewed[[i, j]]).abs() < 1e-12_f64);
            }
        }
    }

    #[test]
    fn solve_reconstructs_rhs() {
        let matrix =
            Array2::<f64>::from_shape_vec((2, 2), vec![4.0_f64, 2.0_f64, 2.0_f64, 3.0_f64])
                .unwrap();
        let rhs = Array1::from_vec(vec![1.0_f64, 1.0_f64]);
        let x = solve(&matrix, &rhs).unwrap();
        let reconstructed = matrix.dot(&x);
        assert!((reconstructed[0] - rhs[0]).abs() < 1e-10_f64);
        assert!((reconstructed[1] - rhs[1]).abs() < 1e-10_f64);
    }

    #[test]
    fn non_spd_input_errors() {
        let matrix =
            Array2::<f64>::from_shape_vec((2, 2), vec![1.0_f64, 2.0_f64, 2.0_f64, 1.0_f64])
                .unwrap();
        let result = decompose(&matrix);
        assert!(matches!(result, Err(CholeskyError::NotPositiveDefinite)));
    }

    #[test]
    fn inverse_multiplied_by_matrix_is_identity() {
        let matrix =
            Array2::<f64>::from_shape_vec((2, 2), vec![4.0_f64, 2.0_f64, 2.0_f64, 3.0_f64])
                .unwrap();
        let inverse = inverse(&matrix).unwrap();
        let product = matrix.dot(&inverse);
        assert!((product[[0, 0]] - 1.0_f64).abs() < 1e-8_f64);
        assert!((product[[1, 1]] - 1.0_f64).abs() < 1e-8_f64);
        assert!(product[[0, 1]].abs() < 1e-8_f64);
        assert!(product[[1, 0]].abs() < 1e-8_f64);
    }

    #[test]
    fn solve_into_rejects_bad_output_length() {
        let matrix =
            Array2::<f64>::from_shape_vec((2, 2), vec![4.0_f64, 2.0_f64, 2.0_f64, 3.0_f64])
                .unwrap();
        let rhs = Array1::from_vec(vec![1.0_f64, 1.0_f64]);
        let mut output = Array1::from_vec(vec![0.0_f64]);
        let result = solve_into(&matrix, &rhs, &mut output);
        assert!(matches!(result, Err(CholeskyError::InvalidInput(_))));
    }

    #[test]
    fn solve_into_matches_solve() {
        let matrix =
            Array2::<f64>::from_shape_vec((2, 2), vec![5.0_f64, 1.0_f64, 1.0_f64, 2.0_f64])
                .unwrap();
        let rhs = Array1::from_vec(vec![3.0_f64, 4.0_f64]);
        let expected = solve(&matrix, &rhs).unwrap();
        let mut output = Array1::<f64>::zeros(2);
        solve_into(&matrix, &rhs, &mut output).unwrap();
        assert!((output[0] - expected[0]).abs() < 1e-10_f64);
        assert!((output[1] - expected[1]).abs() < 1e-10_f64);
    }

    #[test]
    fn factor_view_helpers_match_matrix_paths() {
        let matrix =
            Array2::<f64>::from_shape_vec((2, 2), vec![5.0_f64, 1.0_f64, 1.0_f64, 2.0_f64])
                .unwrap();
        let rhs = Array1::from_vec(vec![3.0_f64, 4.0_f64]);
        let factor = decompose(&matrix).unwrap();

        let expected_solution = solve(&matrix, &rhs).unwrap();
        let actual_solution = solve_from_factor_view(&factor.l.view(), &rhs.view()).unwrap();
        let mut output = Array1::<f64>::zeros(rhs.len());
        solve_from_factor_into_view(&factor.l.view(), &rhs.view(), &mut output.view_mut()).unwrap();
        let actual_inverse = inverse_from_factor_view(&factor.l.view()).unwrap();
        let mut inverse_out = Array2::<f64>::zeros(matrix.dim());
        inverse_from_factor_into_view(&factor.l.view(), &mut inverse_out.view_mut()).unwrap();

        assert!((expected_solution[0] - actual_solution[0]).abs() < 1e-10_f64);
        assert!((expected_solution[1] - actual_solution[1]).abs() < 1e-10_f64);
        assert!((actual_solution[0] - output[0]).abs() < 1e-10_f64);
        assert!((actual_solution[1] - output[1]).abs() < 1e-10_f64);

        let matrix_inverse = inverse(&matrix).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!((matrix_inverse[[i, j]] - actual_inverse[[i, j]]).abs() < 1e-10_f64);
                assert!((actual_inverse[[i, j]] - inverse_out[[i, j]]).abs() < 1e-10_f64);
            }
        }
    }

    #[test]
    fn solve_rejects_bad_rhs_length() {
        let matrix = Array2::<f64>::eye(2);
        let rhs = Array1::from_vec(vec![1.0_f64, 2.0_f64, 3.0_f64]);
        let mut output = Array1::<f64>::zeros(3);
        let result = solve_into(&matrix, &rhs, &mut output);
        assert!(matches!(result, Err(CholeskyError::InvalidInput(_))));
    }

    #[test]
    fn decompose_rejects_non_finite_input() {
        let matrix =
            Array2::<f64>::from_shape_vec((2, 2), vec![1.0_f64, f64::NAN, 0.0_f64, 1.0_f64])
                .unwrap();
        let result = decompose(&matrix);
        assert!(matches!(result, Err(CholeskyError::NumericalInstability)));
    }

    #[test]
    fn real_f32_paths_match_expected() {
        let matrix =
            Array2::<f32>::from_shape_vec((2, 2), vec![4.0_f32, 2.0_f32, 2.0_f32, 3.0_f32])
                .unwrap();
        let rhs = Array1::from_vec(vec![1.0_f32, 1.0_f32]);

        let decomposition = decompose(&matrix).unwrap();
        let reconstructed = decomposition.l.dot(&decomposition.l.t());
        for i in 0..2 {
            for j in 0..2 {
                assert!((matrix[[i, j]] - reconstructed[[i, j]]).abs() < 1e-4_f32);
            }
        }

        let solution = solve(&matrix, &rhs).unwrap();
        let mut output = Array1::<f32>::zeros(rhs.len());
        solve_into(&matrix, &rhs, &mut output).unwrap();
        for i in 0..rhs.len() {
            assert!((solution[i] - output[i]).abs() < 1e-5_f32);
        }

        let inverse = inverse(&matrix).unwrap();
        let product = matrix.dot(&inverse);
        assert!((product[[0, 0]] - 1.0_f32).abs() < 1e-3_f32);
        assert!((product[[1, 1]] - 1.0_f32).abs() < 1e-3_f32);
    }

    #[test]
    fn complex_cholesky_paths_work() {
        let matrix = Array2::from_shape_vec((2, 2), vec![
            Complex64::new(5.0, 0.0),
            Complex64::new(1.0, -1.0),
            Complex64::new(1.0, 1.0),
            Complex64::new(4.0, 0.0),
        ])
        .unwrap();
        let rhs = Array1::from_vec(vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)]);

        let factor = decompose_complex(&matrix).unwrap();
        assert_eq!(factor.l.dim(), (2, 2));

        let solution = solve_complex(&matrix, &rhs).unwrap();
        let reconstructed = matrix.dot(&solution);
        for i in 0..rhs.len() {
            assert!((reconstructed[i] - rhs[i]).norm() < 1e-8);
        }

        let inverse = inverse_complex(&matrix).unwrap();
        let product = matrix.dot(&inverse);
        assert!((product[[0, 0]] - Complex64::new(1.0, 0.0)).norm() < 1e-8);
        assert!((product[[1, 1]] - Complex64::new(1.0, 0.0)).norm() < 1e-8);
        assert!(product[[0, 1]].norm() < 1e-8);
        assert!(product[[1, 0]].norm() < 1e-8);

        let view_factor = decompose_complex_view(&matrix.view()).unwrap();
        let view_solution = solve_complex_view(&matrix.view(), &rhs.view()).unwrap();
        let view_inverse = inverse_complex_view(&matrix.view()).unwrap();
        assert_eq!(factor.l.dim(), view_factor.l.dim());
        for i in 0..rhs.len() {
            assert!((solution[i] - view_solution[i]).norm() < 1e-10);
        }
        for i in 0..2 {
            for j in 0..2 {
                assert!((inverse[[i, j]] - view_inverse[[i, j]]).norm() < 1e-10);
            }
        }

        let factor_solution =
            solve_complex_from_factor_view(&factor.l.view(), &rhs.view()).unwrap();
        let mut factor_solution_out = Array1::<Complex64>::zeros(rhs.len());
        solve_complex_from_factor_into_view(
            &factor.l.view(),
            &rhs.view(),
            &mut factor_solution_out.view_mut(),
        )
        .unwrap();
        let factor_inverse = inverse_complex_from_factor_view(&factor.l.view()).unwrap();
        let mut factor_inverse_out = Array2::<Complex64>::zeros(matrix.dim());
        inverse_complex_from_factor_into_view(&factor.l.view(), &mut factor_inverse_out.view_mut())
            .unwrap();
        for i in 0..rhs.len() {
            assert!((solution[i] - factor_solution[i]).norm() < 1e-10);
            assert!((factor_solution[i] - factor_solution_out[i]).norm() < 1e-10);
        }
        for i in 0..2 {
            for j in 0..2 {
                assert!((inverse[[i, j]] - factor_inverse[[i, j]]).norm() < 1e-10);
                assert!((factor_inverse[[i, j]] - factor_inverse_out[[i, j]]).norm() < 1e-10);
            }
        }
    }

    #[cfg(feature = "magma-system")]
    #[test]
    fn small_cholesky_decompose_shapes_skip_magma_provider() {
        use crate::provider::magma;

        for size in [16_usize, 32, 64] {
            let mut matrix = Array2::<f64>::zeros((size, size));
            let mut diagonal = 4.0_f64;
            for i in 0..size {
                matrix[[i, i]] = diagonal;
                diagonal += 0.01;
                if i + 1 < size {
                    matrix[[i, i + 1]] = 0.1;
                    matrix[[i + 1, i]] = 0.1;
                }
            }

            magma::reset_magma_provider_call_count();
            let result = decompose(&matrix);
            assert!(result.is_ok(), "cholesky decompose failed for size {size}");
            assert_eq!(
                magma::magma_provider_call_count(),
                0,
                "expected CPU path for cholesky decompose size {size}"
            );
        }
    }
}
