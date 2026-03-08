//! LU decomposition over ndarray matrices.

use std::fmt;

use nabled_core::scalar::NabledReal;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_complex::Complex64;

use crate::internal::{DenseKernelPolicy, lu_decompose};
#[cfg(not(feature = "lapack-provider"))]
use crate::internal::{inverse_from_lu, lu_solve};
#[cfg(feature = "magma-system")]
use crate::provider::magma;
#[cfg(feature = "magma-system")]
use crate::provider::policy::MagmaProviderPolicy;

/// Result of LU decomposition.
#[derive(Debug, Clone)]
pub struct NdarrayLUResult<T = f64> {
    /// Lower-triangular factor.
    pub l: Array2<T>,
    /// Upper-triangular factor.
    pub u: Array2<T>,
}

/// Sign and log-absolute value of determinant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogDetResult<T> {
    /// Determinant sign.
    pub sign:       i8,
    /// Natural logarithm of absolute determinant.
    pub ln_abs_det: T,
}

/// Result metadata for mixed-precision LU solve routines.
#[derive(Debug, Clone, PartialEq)]
pub struct MixedSolveResult<T> {
    /// The solved vector `x` from `Ax=b`.
    pub solution:              Array1<T>,
    /// Iterative-refinement steps performed by the provider.
    pub refinement_iterations: usize,
}

/// Error type for LU operations.
#[derive(Debug, Clone, PartialEq)]
pub enum LUError {
    /// Matrix is empty.
    EmptyMatrix,
    /// Matrix must be square.
    NotSquare,
    /// Matrix is singular.
    SingularMatrix,
    /// Iterative solver failed to converge.
    ConvergenceFailed,
    /// Invalid input.
    InvalidInput(String),
    /// Numerical instability detected.
    NumericalInstability,
}

impl fmt::Display for LUError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LUError::EmptyMatrix => write!(f, "Matrix cannot be empty"),
            LUError::NotSquare => write!(f, "Matrix must be square"),
            LUError::SingularMatrix => write!(f, "Matrix is singular"),
            LUError::ConvergenceFailed => write!(f, "Convergence failed"),
            LUError::InvalidInput(message) => write!(f, "Invalid input: {message}"),
            LUError::NumericalInstability => write!(f, "Numerical instability detected"),
        }
    }
}

impl std::error::Error for LUError {}

fn map_lu_error(error: &'static str) -> LUError {
    match error {
        "empty" => LUError::EmptyMatrix,
        "not_square" => LUError::NotSquare,
        "singular" => LUError::SingularMatrix,
        "convergence_failed" => LUError::ConvergenceFailed,
        "non_finite" => LUError::NumericalInstability,
        "bad_dimensions" => {
            LUError::InvalidInput("RHS length must match matrix dimensions".to_string())
        }
        "provider_init_failed" | "invalid_dimensions" | "invalid_input" => {
            LUError::InvalidInput("provider failure".to_string())
        }
        _ => LUError::InvalidInput(error.to_string()),
    }
}

#[cfg(feature = "magma-system")]
fn is_magma_runtime_failure(error: &LUError) -> bool {
    matches!(
        error,
        LUError::InvalidInput(message)
            if message.contains("provider")
                || message.contains("invalid_dimensions")
                || message.contains("invalid_input")
                || message.contains("provider_alloc_failed")
    )
}

#[cfg(feature = "magma-system")]
fn should_fallback_magma_runtime(error: &LUError) -> bool {
    !MagmaProviderPolicy::fail_fast_mode() && is_magma_runtime_failure(error)
}

#[cfg(all(feature = "magma-system", feature = "lapack-provider"))]
#[doc(hidden)]
pub trait LuProviderScalar:
    NabledReal + magma::MagmaReal + ndarray_linalg::Lapack<Real = Self>
{
}

#[cfg(all(feature = "magma-system", feature = "lapack-provider"))]
impl<T> LuProviderScalar for T where
    T: NabledReal + magma::MagmaReal + ndarray_linalg::Lapack<Real = T>
{
}

#[cfg(all(feature = "magma-system", not(feature = "lapack-provider")))]
#[doc(hidden)]
pub trait LuProviderScalar: NabledReal + magma::MagmaReal {}

#[cfg(all(feature = "magma-system", not(feature = "lapack-provider")))]
impl<T> LuProviderScalar for T where T: NabledReal + magma::MagmaReal {}

#[cfg(all(feature = "lapack-provider", not(feature = "magma-system")))]
#[doc(hidden)]
pub trait LuProviderScalar: NabledReal + ndarray_linalg::Lapack {}

#[cfg(all(feature = "lapack-provider", not(feature = "magma-system")))]
impl<T> LuProviderScalar for T where T: NabledReal + ndarray_linalg::Lapack {}

#[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
#[doc(hidden)]
pub trait LuProviderScalar: NabledReal {}

#[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
impl<T> LuProviderScalar for T where T: NabledReal {}

fn validate_square_finite_view<T: NabledReal>(matrix: &ArrayView2<'_, T>) -> Result<(), LUError> {
    if matrix.is_empty() {
        return Err(LUError::EmptyMatrix);
    }
    if matrix.nrows() != matrix.ncols() {
        return Err(LUError::NotSquare);
    }
    if matrix.iter().any(|value| !value.is_finite()) {
        return Err(LUError::NumericalInstability);
    }
    Ok(())
}

fn validate_complex_square_finite_view(matrix: &ArrayView2<'_, Complex64>) -> Result<(), LUError> {
    if matrix.is_empty() {
        return Err(LUError::EmptyMatrix);
    }
    if matrix.nrows() != matrix.ncols() {
        return Err(LUError::NotSquare);
    }
    if matrix.iter().any(|value| !value.re.is_finite() || !value.im.is_finite()) {
        return Err(LUError::NumericalInstability);
    }
    Ok(())
}

#[cfg(not(feature = "lapack-provider"))]
type ComplexLUFactors = (Array2<Complex64>, Array2<Complex64>, Vec<usize>, i8);

#[cfg(not(feature = "lapack-provider"))]
#[allow(clippy::many_single_char_names)]
fn decompose_complex_internal(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<ComplexLUFactors, LUError> {
    validate_complex_square_finite_view(matrix)?;

    let n = matrix.nrows();
    let mut l = Array2::<Complex64>::zeros((n, n));
    let mut u = matrix.to_owned();
    let mut pivots = (0..n).collect::<Vec<usize>>();
    let mut sign = 1_i8;

    for i in 0..n {
        l[[i, i]] = Complex64::new(1.0, 0.0);
    }

    for k in 0..n {
        let mut pivot_row = k;
        let mut pivot_norm = u[[k, k]].norm();
        for i in (k + 1)..n {
            let norm = u[[i, k]].norm();
            if norm > pivot_norm {
                pivot_norm = norm;
                pivot_row = i;
            }
        }
        if pivot_norm <= DenseKernelPolicy::BASE_TOLERANCE {
            return Err(LUError::SingularMatrix);
        }

        if pivot_row != k {
            for j in 0..n {
                let temp = u[[k, j]];
                u[[k, j]] = u[[pivot_row, j]];
                u[[pivot_row, j]] = temp;
            }
            for j in 0..k {
                let temp = l[[k, j]];
                l[[k, j]] = l[[pivot_row, j]];
                l[[pivot_row, j]] = temp;
            }
            pivots.swap(k, pivot_row);
            sign *= -1;
        }

        let pivot = u[[k, k]];
        for i in (k + 1)..n {
            let factor = u[[i, k]] / pivot;
            l[[i, k]] = factor;
            u[[i, k]] = Complex64::new(0.0, 0.0);
            for j in (k + 1)..n {
                let top_row_value = u[[k, j]];
                u[[i, j]] -= factor * top_row_value;
            }
        }
    }

    Ok((l, u, pivots, sign))
}

#[cfg(not(feature = "lapack-provider"))]
#[allow(clippy::many_single_char_names)]
fn solve_complex_from_factors(
    l: &Array2<Complex64>,
    u: &Array2<Complex64>,
    pivots: &[usize],
    rhs: &ArrayView1<'_, Complex64>,
) -> Result<Array1<Complex64>, LUError> {
    let n = l.nrows();
    if rhs.len() != n {
        return Err(LUError::InvalidInput("RHS length must match matrix dimensions".to_string()));
    }

    let mut b = Array1::<Complex64>::zeros(n);
    for i in 0..n {
        b[i] = rhs[pivots[i]];
    }

    let mut y = Array1::<Complex64>::zeros(n);
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= l[[i, j]] * y[j];
        }
        y[i] = sum;
    }

    let mut x = Array1::<Complex64>::zeros(n);
    for i_rev in 0..n {
        let i = n - 1 - i_rev;
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= u[[i, j]] * x[j];
        }
        let diagonal = u[[i, i]];
        if diagonal.norm() <= DenseKernelPolicy::BASE_TOLERANCE {
            return Err(LUError::SingularMatrix);
        }
        x[i] = sum / diagonal;
    }
    Ok(x)
}

#[cfg(not(feature = "lapack-provider"))]
fn inverse_complex_internal(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<Array2<Complex64>, LUError> {
    let (l, u, pivots, _) = decompose_complex_internal(matrix)?;
    let n = matrix.nrows();
    let mut inverse = Array2::<Complex64>::zeros((n, n));

    for col in 0..n {
        let mut e = Array1::<Complex64>::zeros(n);
        e[col] = Complex64::new(1.0, 0.0);
        let solution = solve_complex_from_factors(&l, &u, &pivots, &e.view())?;
        for row in 0..n {
            inverse[[row, col]] = solution[row];
        }
    }
    Ok(inverse)
}

fn decompose_internal<T: NabledReal>(
    matrix: &ArrayView2<'_, T>,
) -> Result<(NdarrayLUResult<T>, Vec<usize>, i8), LUError> {
    let (l, u, pivots, sign) = lu_decompose(matrix).map_err(map_lu_error)?;
    Ok((NdarrayLUResult { l, u }, pivots, sign))
}

#[cfg(not(feature = "magma-system"))]
fn mixed_lu_requires_magma_error() -> LUError {
    LUError::InvalidInput("mixed LU solve requires feature `magma-system`".to_string())
}

#[cfg(all(feature = "magma-system", feature = "lapack-provider"))]
fn solve_lapack_provider<T>(
    matrix: &ArrayView2<'_, T>,
    rhs: &ArrayView1<'_, T>,
) -> Result<Array1<T>, LUError>
where
    T: LuProviderScalar,
{
    use ndarray_linalg::Solve as _;

    matrix.solve(rhs).map_err(|_| LUError::SingularMatrix)
}

#[cfg(all(feature = "magma-system", feature = "lapack-provider"))]
fn inverse_lapack_provider<T>(matrix: &ArrayView2<'_, T>) -> Result<Array2<T>, LUError>
where
    T: LuProviderScalar,
{
    use ndarray_linalg::Inverse as _;

    matrix.inv().map_err(|_| LUError::SingularMatrix)
}

#[cfg(all(feature = "magma-system", feature = "lapack-provider"))]
fn determinant_lapack_provider<T>(matrix: &ArrayView2<'_, T>) -> Result<T, LUError>
where
    T: LuProviderScalar,
{
    use ndarray_linalg::Determinant as _;

    matrix.det().map_err(|_| LUError::SingularMatrix)
}

#[cfg(all(feature = "magma-system", feature = "lapack-provider"))]
fn solve_complex_lapack_provider(
    matrix: &ArrayView2<'_, Complex64>,
    rhs: &ArrayView1<'_, Complex64>,
) -> Result<Array1<Complex64>, LUError> {
    use ndarray_linalg::Solve as _;

    matrix.solve(rhs).map_err(|_| LUError::SingularMatrix)
}

#[cfg(all(feature = "magma-system", feature = "lapack-provider"))]
fn inverse_complex_lapack_provider(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<Array2<Complex64>, LUError> {
    use ndarray_linalg::Inverse as _;

    matrix.inv().map_err(|_| LUError::SingularMatrix)
}

#[cfg(all(feature = "magma-system", feature = "lapack-provider"))]
fn determinant_complex_lapack_provider(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<Complex64, LUError> {
    use ndarray_linalg::Determinant as _;

    matrix.det().map_err(|_| LUError::SingularMatrix)
}

#[cfg(feature = "magma-system")]
fn solve_provider<T>(
    matrix: &ArrayView2<'_, T>,
    rhs: &ArrayView1<'_, T>,
) -> Result<Array1<T>, LUError>
where
    T: LuProviderScalar,
{
    validate_square_finite_view(matrix)?;
    if !MagmaProviderPolicy::prefer_decomposition(matrix.nrows(), matrix.ncols()) {
        #[cfg(feature = "lapack-provider")]
        {
            return solve_lapack_provider(matrix, rhs);
        }
        #[cfg(not(feature = "lapack-provider"))]
        {
            let (decomposition, pivots, _) = decompose_with_metadata(matrix)?;
            return lu_solve(&decomposition.l, &decomposition.u, &pivots, rhs)
                .map_err(map_lu_error);
        }
    }
    match magma::lu_solve(matrix, rhs) {
        Ok(solution) => Ok(solution),
        Err(error) => {
            let mapped = map_lu_error(error);
            if should_fallback_magma_runtime(&mapped) {
                #[cfg(feature = "lapack-provider")]
                {
                    return solve_lapack_provider(matrix, rhs);
                }
                #[cfg(not(feature = "lapack-provider"))]
                {
                    let (decomposition, pivots, _) = decompose_with_metadata(matrix)?;
                    return lu_solve(&decomposition.l, &decomposition.u, &pivots, rhs)
                        .map_err(map_lu_error);
                }
            }
            Err(mapped)
        }
    }
}

#[cfg(feature = "magma-system")]
fn inverse_provider<T>(matrix: &ArrayView2<'_, T>) -> Result<Array2<T>, LUError>
where
    T: LuProviderScalar,
{
    validate_square_finite_view(matrix)?;
    if !MagmaProviderPolicy::prefer_decomposition(matrix.nrows(), matrix.ncols()) {
        #[cfg(feature = "lapack-provider")]
        {
            return inverse_lapack_provider(matrix);
        }
        #[cfg(not(feature = "lapack-provider"))]
        {
            let (decomposition, pivots, _) = decompose_with_metadata(matrix)?;
            return inverse_from_lu(&decomposition.l, &decomposition.u, &pivots)
                .map_err(map_lu_error);
        }
    }
    match magma::lu_inverse(matrix) {
        Ok(inverse) => Ok(inverse),
        Err(error) => {
            let mapped = map_lu_error(error);
            if should_fallback_magma_runtime(&mapped) {
                #[cfg(feature = "lapack-provider")]
                {
                    return inverse_lapack_provider(matrix);
                }
                #[cfg(not(feature = "lapack-provider"))]
                {
                    let (decomposition, pivots, _) = decompose_with_metadata(matrix)?;
                    return inverse_from_lu(&decomposition.l, &decomposition.u, &pivots)
                        .map_err(map_lu_error);
                }
            }
            Err(mapped)
        }
    }
}

#[cfg(feature = "magma-system")]
fn determinant_provider<T>(matrix: &ArrayView2<'_, T>) -> Result<T, LUError>
where
    T: LuProviderScalar,
{
    validate_square_finite_view(matrix)?;
    if !MagmaProviderPolicy::prefer_decomposition(matrix.nrows(), matrix.ncols()) {
        #[cfg(feature = "lapack-provider")]
        {
            return determinant_lapack_provider(matrix);
        }
        #[cfg(not(feature = "lapack-provider"))]
        {
            let (decomposition, _, sign) = decompose_with_metadata(matrix)?;
            let mut determinant = if sign >= 0 { T::one() } else { -T::one() };
            for i in 0..decomposition.u.nrows() {
                determinant *= decomposition.u[[i, i]];
            }
            if !determinant.is_finite() {
                return Err(LUError::NumericalInstability);
            }
            return Ok(determinant);
        }
    }
    match magma::lu_determinant(matrix) {
        Ok(determinant) => Ok(determinant),
        Err(error) => {
            let mapped = map_lu_error(error);
            if should_fallback_magma_runtime(&mapped) {
                #[cfg(feature = "lapack-provider")]
                {
                    return determinant_lapack_provider(matrix);
                }
                #[cfg(not(feature = "lapack-provider"))]
                {
                    let (decomposition, _, sign) = decompose_with_metadata(matrix)?;
                    let mut determinant = if sign >= 0 { T::one() } else { -T::one() };
                    for i in 0..decomposition.u.nrows() {
                        determinant *= decomposition.u[[i, i]];
                    }
                    if !determinant.is_finite() {
                        return Err(LUError::NumericalInstability);
                    }
                    return Ok(determinant);
                }
            }
            Err(mapped)
        }
    }
}

#[cfg(feature = "magma-system")]
fn solve_mixed_f64_provider(
    matrix: &ArrayView2<'_, f64>,
    rhs: &ArrayView1<'_, f64>,
) -> Result<MixedSolveResult<f64>, LUError> {
    validate_square_finite_view(matrix)?;
    if rhs.len() != matrix.nrows() {
        return Err(LUError::InvalidInput("RHS length must match matrix dimensions".to_string()));
    }
    match magma::lu_solve_mixed_f64(matrix, rhs) {
        Ok((solution, refinement_iterations)) => {
            Ok(MixedSolveResult { solution, refinement_iterations })
        }
        Err(error) => {
            let mapped = map_lu_error(error);
            if should_fallback_magma_runtime(&mapped) {
                #[cfg(feature = "lapack-provider")]
                {
                    let solution = solve_lapack_provider(matrix, rhs)?;
                    return Ok(MixedSolveResult { solution, refinement_iterations: 0 });
                }
                #[cfg(not(feature = "lapack-provider"))]
                {
                    let (decomposition, pivots, _) = decompose_with_metadata(matrix)?;
                    let solution = lu_solve(&decomposition.l, &decomposition.u, &pivots, rhs)
                        .map_err(map_lu_error)?;
                    return Ok(MixedSolveResult { solution, refinement_iterations: 0 });
                }
            }
            Err(mapped)
        }
    }
}

#[cfg(not(feature = "magma-system"))]
fn solve_mixed_f64_provider(
    matrix: &ArrayView2<'_, f64>,
    rhs: &ArrayView1<'_, f64>,
) -> Result<MixedSolveResult<f64>, LUError> {
    validate_square_finite_view(matrix)?;
    if rhs.len() != matrix.nrows() {
        return Err(LUError::InvalidInput("RHS length must match matrix dimensions".to_string()));
    }
    Err(mixed_lu_requires_magma_error())
}

#[cfg(feature = "magma-system")]
fn solve_mixed_complex_provider(
    matrix: &ArrayView2<'_, Complex64>,
    rhs: &ArrayView1<'_, Complex64>,
) -> Result<MixedSolveResult<Complex64>, LUError> {
    validate_complex_square_finite_view(matrix)?;
    if rhs.len() != matrix.nrows() {
        return Err(LUError::InvalidInput("RHS length must match matrix dimensions".to_string()));
    }
    match magma::lu_solve_mixed_complex(matrix, rhs) {
        Ok((solution, refinement_iterations)) => {
            Ok(MixedSolveResult { solution, refinement_iterations })
        }
        Err(error) => {
            let mapped = map_lu_error(error);
            if should_fallback_magma_runtime(&mapped) {
                #[cfg(feature = "lapack-provider")]
                {
                    let solution = solve_complex_lapack_provider(matrix, rhs)?;
                    return Ok(MixedSolveResult { solution, refinement_iterations: 0 });
                }
                #[cfg(not(feature = "lapack-provider"))]
                {
                    let (l, u, pivots, _) = decompose_complex_internal(matrix)?;
                    let solution = solve_complex_from_factors(&l, &u, &pivots, rhs)?;
                    return Ok(MixedSolveResult { solution, refinement_iterations: 0 });
                }
            }
            Err(mapped)
        }
    }
}

#[cfg(not(feature = "magma-system"))]
fn solve_mixed_complex_provider(
    matrix: &ArrayView2<'_, Complex64>,
    rhs: &ArrayView1<'_, Complex64>,
) -> Result<MixedSolveResult<Complex64>, LUError> {
    validate_complex_square_finite_view(matrix)?;
    if rhs.len() != matrix.nrows() {
        return Err(LUError::InvalidInput("RHS length must match matrix dimensions".to_string()));
    }
    Err(mixed_lu_requires_magma_error())
}

#[cfg(all(feature = "lapack-provider", not(feature = "magma-system")))]
fn solve_provider<T>(
    matrix: &ArrayView2<'_, T>,
    rhs: &ArrayView1<'_, T>,
) -> Result<Array1<T>, LUError>
where
    T: LuProviderScalar,
{
    use ndarray_linalg::Solve as _;

    validate_square_finite_view(matrix)?;
    matrix.solve(rhs).map_err(|_| LUError::SingularMatrix)
}

#[cfg(all(feature = "lapack-provider", not(feature = "magma-system")))]
fn inverse_provider<T>(matrix: &ArrayView2<'_, T>) -> Result<Array2<T>, LUError>
where
    T: LuProviderScalar,
{
    use ndarray_linalg::Inverse as _;

    validate_square_finite_view(matrix)?;
    matrix.inv().map_err(|_| LUError::SingularMatrix)
}

#[cfg(all(feature = "lapack-provider", not(feature = "magma-system")))]
fn determinant_provider<T>(matrix: &ArrayView2<'_, T>) -> Result<T, LUError>
where
    T: LuProviderScalar,
{
    use ndarray_linalg::Determinant as _;

    validate_square_finite_view(matrix)?;
    matrix.det().map_err(|_| LUError::SingularMatrix)
}

#[cfg(feature = "magma-system")]
fn solve_complex_provider(
    matrix: &ArrayView2<'_, Complex64>,
    rhs: &ArrayView1<'_, Complex64>,
) -> Result<Array1<Complex64>, LUError> {
    validate_complex_square_finite_view(matrix)?;
    if rhs.len() != matrix.nrows() {
        return Err(LUError::InvalidInput("RHS length must match matrix dimensions".to_string()));
    }
    if !MagmaProviderPolicy::prefer_decomposition(matrix.nrows(), matrix.ncols()) {
        #[cfg(feature = "lapack-provider")]
        {
            return solve_complex_lapack_provider(matrix, rhs);
        }
        #[cfg(not(feature = "lapack-provider"))]
        {
            let (l, u, pivots, _) = decompose_complex_internal(matrix)?;
            return solve_complex_from_factors(&l, &u, &pivots, rhs);
        }
    }
    match magma::lu_solve_complex(matrix, rhs) {
        Ok(solution) => Ok(solution),
        Err(error) => {
            let mapped = map_lu_error(error);
            if should_fallback_magma_runtime(&mapped) {
                #[cfg(feature = "lapack-provider")]
                {
                    return solve_complex_lapack_provider(matrix, rhs);
                }
                #[cfg(not(feature = "lapack-provider"))]
                {
                    let (l, u, pivots, _) = decompose_complex_internal(matrix)?;
                    return solve_complex_from_factors(&l, &u, &pivots, rhs);
                }
            }
            Err(mapped)
        }
    }
}

#[cfg(all(feature = "lapack-provider", not(feature = "magma-system")))]
fn solve_complex_provider(
    matrix: &ArrayView2<'_, Complex64>,
    rhs: &ArrayView1<'_, Complex64>,
) -> Result<Array1<Complex64>, LUError> {
    use ndarray_linalg::Solve as _;

    validate_complex_square_finite_view(matrix)?;
    if rhs.len() != matrix.nrows() {
        return Err(LUError::InvalidInput("RHS length must match matrix dimensions".to_string()));
    }
    matrix.solve(rhs).map_err(|_| LUError::SingularMatrix)
}

#[cfg(feature = "magma-system")]
fn inverse_complex_provider(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<Array2<Complex64>, LUError> {
    validate_complex_square_finite_view(matrix)?;
    if !MagmaProviderPolicy::prefer_decomposition(matrix.nrows(), matrix.ncols()) {
        #[cfg(feature = "lapack-provider")]
        {
            return inverse_complex_lapack_provider(matrix);
        }
        #[cfg(not(feature = "lapack-provider"))]
        return inverse_complex_internal(matrix);
    }
    match magma::lu_inverse_complex(matrix) {
        Ok(inverse) => Ok(inverse),
        Err(error) => {
            let mapped = map_lu_error(error);
            if should_fallback_magma_runtime(&mapped) {
                #[cfg(feature = "lapack-provider")]
                {
                    return inverse_complex_lapack_provider(matrix);
                }
                #[cfg(not(feature = "lapack-provider"))]
                return inverse_complex_internal(matrix);
            }
            Err(mapped)
        }
    }
}

#[cfg(all(feature = "lapack-provider", not(feature = "magma-system")))]
fn inverse_complex_provider(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<Array2<Complex64>, LUError> {
    use ndarray_linalg::Inverse as _;

    validate_complex_square_finite_view(matrix)?;
    matrix.inv().map_err(|_| LUError::SingularMatrix)
}

#[cfg(feature = "magma-system")]
fn determinant_complex_provider(matrix: &ArrayView2<'_, Complex64>) -> Result<Complex64, LUError> {
    validate_complex_square_finite_view(matrix)?;
    if !MagmaProviderPolicy::prefer_decomposition(matrix.nrows(), matrix.ncols()) {
        #[cfg(feature = "lapack-provider")]
        {
            return determinant_complex_lapack_provider(matrix);
        }
        #[cfg(not(feature = "lapack-provider"))]
        {
            let (_l, u, _, sign) = decompose_complex_internal(matrix)?;
            let mut determinant =
                if sign >= 0 { Complex64::new(1.0, 0.0) } else { Complex64::new(-1.0, 0.0) };
            for i in 0..u.nrows() {
                determinant *= u[[i, i]];
            }
            if !determinant.re.is_finite() || !determinant.im.is_finite() {
                return Err(LUError::NumericalInstability);
            }
            return Ok(determinant);
        }
    }
    match magma::lu_determinant_complex(matrix) {
        Ok(determinant) => Ok(determinant),
        Err(error) => {
            let mapped = map_lu_error(error);
            if should_fallback_magma_runtime(&mapped) {
                #[cfg(feature = "lapack-provider")]
                {
                    return determinant_complex_lapack_provider(matrix);
                }
                #[cfg(not(feature = "lapack-provider"))]
                {
                    let (_l, u, _, sign) = decompose_complex_internal(matrix)?;
                    let mut determinant = if sign >= 0 {
                        Complex64::new(1.0, 0.0)
                    } else {
                        Complex64::new(-1.0, 0.0)
                    };
                    for i in 0..u.nrows() {
                        determinant *= u[[i, i]];
                    }
                    if !determinant.re.is_finite() || !determinant.im.is_finite() {
                        return Err(LUError::NumericalInstability);
                    }
                    return Ok(determinant);
                }
            }
            Err(mapped)
        }
    }
}

#[cfg(all(feature = "lapack-provider", not(feature = "magma-system")))]
fn determinant_complex_provider(matrix: &ArrayView2<'_, Complex64>) -> Result<Complex64, LUError> {
    use ndarray_linalg::Determinant as _;

    validate_complex_square_finite_view(matrix)?;
    matrix.det().map_err(|_| LUError::SingularMatrix)
}

fn decompose_with_metadata<T: NabledReal>(
    matrix: &ArrayView2<'_, T>,
) -> Result<(NdarrayLUResult<T>, Vec<usize>, i8), LUError> {
    decompose_internal(matrix)
}

/// Compute LU decomposition with partial pivoting.
///
/// # Errors
/// Returns an error if input is invalid or decomposition fails.
pub fn decompose<T: NabledReal>(matrix: &Array2<T>) -> Result<NdarrayLUResult<T>, LUError> {
    let (result, _, _) = decompose_with_metadata(&matrix.view())?;
    Ok(result)
}

/// Compute LU decomposition with partial pivoting from a matrix view.
///
/// # Errors
/// Returns an error if input is invalid or decomposition fails.
pub fn decompose_view<T: NabledReal>(
    matrix: &ArrayView2<'_, T>,
) -> Result<NdarrayLUResult<T>, LUError> {
    let (result, _, _) = decompose_with_metadata(matrix)?;
    Ok(result)
}

/// Solve `Ax=b` using LU decomposition.
///
/// # Errors
/// Returns an error if dimensions are incompatible or matrix is singular.
#[cfg(any(feature = "lapack-provider", feature = "magma-system"))]
pub fn solve<T: LuProviderScalar>(
    matrix: &Array2<T>,
    rhs: &Array1<T>,
) -> Result<Array1<T>, LUError> {
    solve_impl(&matrix.view(), &rhs.view())
}

/// Solve `Ax=b` using LU decomposition.
///
/// # Errors
/// Returns an error if dimensions are incompatible or matrix is singular.
#[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
pub fn solve<T: NabledReal>(matrix: &Array2<T>, rhs: &Array1<T>) -> Result<Array1<T>, LUError> {
    solve_impl(&matrix.view(), &rhs.view())
}

#[cfg(any(feature = "lapack-provider", feature = "magma-system"))]
fn solve_impl<T>(matrix: &ArrayView2<'_, T>, rhs: &ArrayView1<'_, T>) -> Result<Array1<T>, LUError>
where
    T: LuProviderScalar,
{
    validate_square_finite_view(matrix)?;
    if rhs.len() != matrix.nrows() {
        return Err(LUError::InvalidInput("RHS length must match matrix dimensions".to_string()));
    }

    solve_provider(matrix, rhs)
}

#[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
fn solve_impl<T>(matrix: &ArrayView2<'_, T>, rhs: &ArrayView1<'_, T>) -> Result<Array1<T>, LUError>
where
    T: NabledReal,
{
    validate_square_finite_view(matrix)?;
    if rhs.len() != matrix.nrows() {
        return Err(LUError::InvalidInput("RHS length must match matrix dimensions".to_string()));
    }

    let (decomposition, pivots, _) = decompose_with_metadata(matrix)?;
    lu_solve(&decomposition.l, &decomposition.u, &pivots, rhs).map_err(map_lu_error)
}

/// Solve complex-valued `Ax=b` using LU decomposition.
///
/// # Errors
/// Returns an error if dimensions are incompatible or matrix is singular.
pub fn solve_complex(
    matrix: &Array2<Complex64>,
    rhs: &Array1<Complex64>,
) -> Result<Array1<Complex64>, LUError> {
    solve_complex_impl(&matrix.view(), &rhs.view())
}

fn solve_complex_impl(
    matrix: &ArrayView2<'_, Complex64>,
    rhs: &ArrayView1<'_, Complex64>,
) -> Result<Array1<Complex64>, LUError> {
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
        let (l, u, pivots, _) = decompose_complex_internal(matrix)?;
        solve_complex_from_factors(&l, &u, &pivots, rhs)
    }
}

/// Solve `Ax=b` using LU decomposition from matrix/vector views.
///
/// # Errors
/// Returns an error if dimensions are incompatible or matrix is singular.
#[cfg(any(feature = "lapack-provider", feature = "magma-system"))]
pub fn solve_view<T: LuProviderScalar>(
    matrix: &ArrayView2<'_, T>,
    rhs: &ArrayView1<'_, T>,
) -> Result<Array1<T>, LUError> {
    solve_impl(matrix, rhs)
}

/// Solve `Ax=b` using LU decomposition from matrix/vector views.
///
/// # Errors
/// Returns an error if dimensions are incompatible or matrix is singular.
#[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
pub fn solve_view<T: NabledReal>(
    matrix: &ArrayView2<'_, T>,
    rhs: &ArrayView1<'_, T>,
) -> Result<Array1<T>, LUError> {
    solve_impl(matrix, rhs)
}

/// Solve `Ax=b` using MAGMA mixed-precision iterative refinement (`f64` -> `f32` work buffers).
///
/// This API is available in all builds, but requires feature `magma-system` at runtime.
///
/// # Errors
/// Returns an error if input is invalid, matrix is singular, convergence fails, or `magma-system`
/// is not enabled.
pub fn solve_mixed_f64(
    matrix: &Array2<f64>,
    rhs: &Array1<f64>,
) -> Result<MixedSolveResult<f64>, LUError> {
    solve_mixed_f64_provider(&matrix.view(), &rhs.view())
}

/// Solve `Ax=b` using MAGMA mixed-precision iterative refinement from views (`f64` -> `f32` work
/// buffers).
///
/// This API is available in all builds, but requires feature `magma-system` at runtime.
///
/// # Errors
/// Returns an error if input is invalid, matrix is singular, convergence fails, or `magma-system`
/// is not enabled.
pub fn solve_mixed_f64_view(
    matrix: &ArrayView2<'_, f64>,
    rhs: &ArrayView1<'_, f64>,
) -> Result<MixedSolveResult<f64>, LUError> {
    solve_mixed_f64_provider(matrix, rhs)
}

/// Solve complex-valued `Ax=b` using MAGMA mixed-precision iterative refinement
/// (`Complex64` -> `Complex32` work buffers).
///
/// This API is available in all builds, but requires feature `magma-system` at runtime.
///
/// # Errors
/// Returns an error if input is invalid, matrix is singular, convergence fails, or `magma-system`
/// is not enabled.
pub fn solve_mixed_complex(
    matrix: &Array2<Complex64>,
    rhs: &Array1<Complex64>,
) -> Result<MixedSolveResult<Complex64>, LUError> {
    solve_mixed_complex_provider(&matrix.view(), &rhs.view())
}

/// Solve complex-valued `Ax=b` using MAGMA mixed-precision iterative refinement from views
/// (`Complex64` -> `Complex32` work buffers).
///
/// This API is available in all builds, but requires feature `magma-system` at runtime.
///
/// # Errors
/// Returns an error if input is invalid, matrix is singular, convergence fails, or `magma-system`
/// is not enabled.
pub fn solve_mixed_complex_view(
    matrix: &ArrayView2<'_, Complex64>,
    rhs: &ArrayView1<'_, Complex64>,
) -> Result<MixedSolveResult<Complex64>, LUError> {
    solve_mixed_complex_provider(matrix, rhs)
}

/// Solve complex-valued `Ax=b` from matrix/vector views.
///
/// # Errors
/// Returns an error if dimensions are incompatible or matrix is singular.
pub fn solve_complex_view(
    matrix: &ArrayView2<'_, Complex64>,
    rhs: &ArrayView1<'_, Complex64>,
) -> Result<Array1<Complex64>, LUError> {
    solve_complex_impl(matrix, rhs)
}

/// Compute matrix inverse via LU decomposition.
///
/// # Errors
/// Returns an error if matrix is singular.
#[cfg(any(feature = "lapack-provider", feature = "magma-system"))]
pub fn inverse<T: LuProviderScalar>(matrix: &Array2<T>) -> Result<Array2<T>, LUError> {
    inverse_impl(&matrix.view())
}

/// Compute matrix inverse via LU decomposition.
///
/// # Errors
/// Returns an error if matrix is singular.
#[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
pub fn inverse<T: NabledReal>(matrix: &Array2<T>) -> Result<Array2<T>, LUError> {
    inverse_impl(&matrix.view())
}

#[cfg(any(feature = "lapack-provider", feature = "magma-system"))]
fn inverse_impl<T>(matrix: &ArrayView2<'_, T>) -> Result<Array2<T>, LUError>
where
    T: LuProviderScalar,
{
    inverse_provider(matrix)
}

#[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
fn inverse_impl<T>(matrix: &ArrayView2<'_, T>) -> Result<Array2<T>, LUError>
where
    T: NabledReal,
{
    let (decomposition, pivots, _) = decompose_with_metadata(matrix)?;
    inverse_from_lu(&decomposition.l, &decomposition.u, &pivots).map_err(map_lu_error)
}

/// Compute complex matrix inverse via LU decomposition.
///
/// # Errors
/// Returns an error if matrix is singular.
pub fn inverse_complex(matrix: &Array2<Complex64>) -> Result<Array2<Complex64>, LUError> {
    inverse_complex_impl(&matrix.view())
}

fn inverse_complex_impl(matrix: &ArrayView2<'_, Complex64>) -> Result<Array2<Complex64>, LUError> {
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
        inverse_complex_internal(matrix)
    }
}

/// Compute matrix inverse via LU decomposition from a matrix view.
///
/// # Errors
/// Returns an error if matrix is singular.
#[cfg(any(feature = "lapack-provider", feature = "magma-system"))]
pub fn inverse_view<T: LuProviderScalar>(matrix: &ArrayView2<'_, T>) -> Result<Array2<T>, LUError> {
    inverse_impl(matrix)
}

/// Compute matrix inverse via LU decomposition from a matrix view.
///
/// # Errors
/// Returns an error if matrix is singular.
#[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
pub fn inverse_view<T: NabledReal>(matrix: &ArrayView2<'_, T>) -> Result<Array2<T>, LUError> {
    inverse_impl(matrix)
}

/// Compute complex matrix inverse from a matrix view.
///
/// # Errors
/// Returns an error if matrix is singular.
pub fn inverse_complex_view(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<Array2<Complex64>, LUError> {
    inverse_complex_impl(matrix)
}

/// Compute determinant via LU decomposition.
///
/// # Errors
/// Returns an error if decomposition fails.
#[cfg(any(feature = "lapack-provider", feature = "magma-system"))]
pub fn determinant<T: LuProviderScalar>(matrix: &Array2<T>) -> Result<T, LUError> {
    determinant_impl(&matrix.view())
}

/// Compute determinant via LU decomposition.
///
/// # Errors
/// Returns an error if decomposition fails.
#[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
pub fn determinant<T: NabledReal>(matrix: &Array2<T>) -> Result<T, LUError> {
    determinant_impl(&matrix.view())
}

#[cfg(any(feature = "lapack-provider", feature = "magma-system"))]
fn determinant_impl<T>(matrix: &ArrayView2<'_, T>) -> Result<T, LUError>
where
    T: LuProviderScalar,
{
    determinant_provider(matrix)
}

#[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
fn determinant_impl<T>(matrix: &ArrayView2<'_, T>) -> Result<T, LUError>
where
    T: NabledReal,
{
    let (decomposition, _, sign) = decompose_with_metadata(matrix)?;
    let mut determinant = if sign >= 0 { T::one() } else { -T::one() };
    for i in 0..decomposition.u.nrows() {
        determinant *= decomposition.u[[i, i]];
    }
    if !determinant.is_finite() {
        return Err(LUError::NumericalInstability);
    }
    Ok(determinant)
}

/// Compute complex determinant via LU decomposition.
///
/// # Errors
/// Returns an error if decomposition fails.
pub fn determinant_complex(matrix: &Array2<Complex64>) -> Result<Complex64, LUError> {
    determinant_complex_impl(&matrix.view())
}

fn determinant_complex_impl(matrix: &ArrayView2<'_, Complex64>) -> Result<Complex64, LUError> {
    #[cfg(feature = "magma-system")]
    {
        determinant_complex_provider(matrix)
    }
    #[cfg(all(feature = "lapack-provider", not(feature = "magma-system")))]
    {
        determinant_complex_provider(matrix)
    }
    #[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
    {
        let (_, u, _, sign) = decompose_complex_internal(matrix)?;
        let mut determinant = Complex64::new(f64::from(sign), 0.0);
        for i in 0..u.nrows() {
            determinant *= u[[i, i]];
        }
        if !determinant.re.is_finite() || !determinant.im.is_finite() {
            return Err(LUError::NumericalInstability);
        }
        Ok(determinant)
    }
}

/// Compute determinant via LU decomposition from a matrix view.
///
/// # Errors
/// Returns an error if decomposition fails.
#[cfg(any(feature = "lapack-provider", feature = "magma-system"))]
pub fn determinant_view<T: LuProviderScalar>(matrix: &ArrayView2<'_, T>) -> Result<T, LUError> {
    determinant_impl(matrix)
}

/// Compute determinant via LU decomposition from a matrix view.
///
/// # Errors
/// Returns an error if decomposition fails.
#[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
pub fn determinant_view<T: NabledReal>(matrix: &ArrayView2<'_, T>) -> Result<T, LUError> {
    determinant_impl(matrix)
}

/// Compute complex determinant from a matrix view.
///
/// # Errors
/// Returns an error if decomposition fails.
pub fn determinant_complex_view(matrix: &ArrayView2<'_, Complex64>) -> Result<Complex64, LUError> {
    determinant_complex_impl(matrix)
}

/// Compute signed log-determinant via LU decomposition.
///
/// # Errors
/// Returns an error if matrix is singular.
#[cfg(any(feature = "lapack-provider", feature = "magma-system"))]
pub fn log_determinant<T: LuProviderScalar>(
    matrix: &Array2<T>,
) -> Result<LogDetResult<T>, LUError> {
    let determinant = determinant_impl(&matrix.view())?;
    let tolerance = T::from_f64(DenseKernelPolicy::BASE_TOLERANCE).unwrap_or(T::epsilon());
    let determinant_abs = num_traits::Float::abs(determinant);
    if determinant_abs <= tolerance {
        return Err(LUError::SingularMatrix);
    }
    let sign = if determinant.is_sign_positive() { 1 } else { -1 };
    Ok(LogDetResult { sign, ln_abs_det: num_traits::Float::ln(determinant_abs) })
}

/// Compute signed log-determinant via LU decomposition.
///
/// # Errors
/// Returns an error if matrix is singular.
#[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
pub fn log_determinant<T: NabledReal>(matrix: &Array2<T>) -> Result<LogDetResult<T>, LUError> {
    let determinant = determinant_impl(&matrix.view())?;
    let tolerance = T::from_f64(DenseKernelPolicy::BASE_TOLERANCE).unwrap_or(T::epsilon());
    let determinant_abs = num_traits::Float::abs(determinant);
    if determinant_abs <= tolerance {
        return Err(LUError::SingularMatrix);
    }
    let sign = if determinant.is_sign_positive() { 1 } else { -1 };
    Ok(LogDetResult { sign, ln_abs_det: num_traits::Float::ln(determinant_abs) })
}

/// Compute signed log-determinant via LU decomposition from a matrix view.
///
/// # Errors
/// Returns an error if matrix is singular.
#[cfg(any(feature = "lapack-provider", feature = "magma-system"))]
pub fn log_determinant_view<T: LuProviderScalar>(
    matrix: &ArrayView2<'_, T>,
) -> Result<LogDetResult<T>, LUError> {
    let determinant = determinant_impl(matrix)?;
    let tolerance = T::from_f64(DenseKernelPolicy::BASE_TOLERANCE).unwrap_or(T::epsilon());
    let determinant_abs = num_traits::Float::abs(determinant);
    if determinant_abs <= tolerance {
        return Err(LUError::SingularMatrix);
    }
    let sign = if determinant.is_sign_positive() { 1 } else { -1 };
    Ok(LogDetResult { sign, ln_abs_det: num_traits::Float::ln(determinant_abs) })
}

/// Compute signed log-determinant via LU decomposition from a matrix view.
///
/// # Errors
/// Returns an error if matrix is singular.
#[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
pub fn log_determinant_view<T: NabledReal>(
    matrix: &ArrayView2<'_, T>,
) -> Result<LogDetResult<T>, LUError> {
    let determinant = determinant_impl(matrix)?;
    let tolerance = T::from_f64(DenseKernelPolicy::BASE_TOLERANCE).unwrap_or(T::epsilon());
    let determinant_abs = num_traits::Float::abs(determinant);
    if determinant_abs <= tolerance {
        return Err(LUError::SingularMatrix);
    }
    let sign = if determinant.is_sign_positive() { 1 } else { -1 };
    Ok(LogDetResult { sign, ln_abs_det: num_traits::Float::ln(determinant_abs) })
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2};
    use num_complex::Complex64;

    use super::*;

    #[test]
    fn solve_reconstructs_rhs() {
        let matrix = Array2::from_shape_vec((2, 2), vec![4.0_f64, 3.0, 6.0, 3.0]).unwrap();
        let rhs = Array1::from_vec(vec![10.0_f64, 12.0]);
        let solution = solve(&matrix, &rhs).unwrap();
        let reconstructed = matrix.dot(&solution);
        assert!((reconstructed[0] - rhs[0]).abs() < 1e-8);
        assert!((reconstructed[1] - rhs[1]).abs() < 1e-8);
    }

    #[test]
    fn determinant_matches_expected() {
        let matrix = Array2::from_shape_vec((2, 2), vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
        let determinant = determinant(&matrix).unwrap();
        assert!((determinant + 2.0).abs() < 1e-12);
    }

    #[test]
    fn singular_matrix_is_rejected() {
        let singular = Array2::from_shape_vec((2, 2), vec![1.0_f64, 2.0, 2.0, 4.0]).unwrap();
        let rhs = Array1::from_vec(vec![1.0_f64, 2.0]);
        assert!(matches!(solve(&singular, &rhs), Err(LUError::SingularMatrix)));
    }

    #[test]
    fn inverse_multiplied_by_matrix_is_identity() {
        let matrix = Array2::from_shape_vec((2, 2), vec![4.0_f64, 7.0, 2.0, 6.0]).unwrap();
        let inverse = inverse(&matrix).unwrap();
        let product = matrix.dot(&inverse);
        assert!((product[[0, 0]] - 1.0).abs() < 1e-8);
        assert!((product[[1, 1]] - 1.0).abs() < 1e-8);
        assert!(product[[0, 1]].abs() < 1e-8);
        assert!(product[[1, 0]].abs() < 1e-8);
    }

    #[test]
    fn log_determinant_has_expected_sign_and_value() {
        let matrix = Array2::from_shape_vec((2, 2), vec![2.0_f64, 0.0, 0.0, -3.0]).unwrap();
        let result = log_determinant(&matrix).unwrap();
        assert_eq!(result.sign, -1);
        assert!((result.ln_abs_det - (6.0_f64).ln()).abs() < 1e-10);
    }

    #[test]
    fn solve_rejects_bad_rhs_length() {
        let matrix = Array2::eye(2);
        let rhs = Array1::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let result = solve(&matrix, &rhs);
        assert!(matches!(result, Err(LUError::InvalidInput(_))));
    }

    #[test]
    fn decompose_exposes_factors() {
        let matrix = Array2::from_shape_vec((2, 2), vec![2.0_f64, 1.0, 4.0, 3.0]).unwrap();
        let lu = decompose(&matrix).unwrap();
        assert_eq!(lu.l.dim(), (2, 2));
        assert_eq!(lu.u.dim(), (2, 2));
    }

    #[cfg(not(feature = "magma-system"))]
    #[test]
    fn mixed_solve_requires_magma_feature() {
        let matrix = Array2::from_shape_vec((2, 2), vec![4.0_f64, 1.0, 1.0, 3.0]).unwrap();
        let rhs = Array1::from_vec(vec![1.0_f64, 2.0]);
        let error = solve_mixed_f64(&matrix, &rhs).unwrap_err();
        assert!(
            matches!(error, LUError::InvalidInput(message) if message.contains("magma-system"))
        );
    }

    #[cfg(feature = "magma-system")]
    #[test]
    fn mixed_f64_solve_reconstructs_rhs() {
        let matrix = Array2::from_shape_vec((2, 2), vec![5.0_f64, 1.0, 1.0, 4.0]).unwrap();
        let rhs = Array1::from_vec(vec![2.0_f64, 3.0]);
        let result = solve_mixed_f64(&matrix, &rhs).unwrap();
        let reconstructed = matrix.dot(&result.solution);
        assert!((reconstructed[0] - rhs[0]).abs() < 1e-9);
        assert!((reconstructed[1] - rhs[1]).abs() < 1e-9);
    }

    #[cfg(feature = "magma-system")]
    #[test]
    fn mixed_complex_solve_reconstructs_rhs() {
        let matrix = Array2::from_shape_vec((2, 2), vec![
            Complex64::new(3.0, 0.5),
            Complex64::new(1.0, -0.25),
            Complex64::new(0.5, 0.5),
            Complex64::new(2.5, -0.75),
        ])
        .unwrap();
        let rhs = Array1::from_vec(vec![Complex64::new(1.0, 0.5), Complex64::new(0.0, 1.0)]);
        let result = solve_mixed_complex(&matrix, &rhs).unwrap();
        let reconstructed = matrix.dot(&result.solution);
        for i in 0..rhs.len() {
            assert!((reconstructed[i] - rhs[i]).norm() < 1e-8);
        }
    }

    #[test]
    fn view_variants_match_owned() {
        let matrix = Array2::from_shape_vec((2, 2), vec![4.0_f64, 7.0, 2.0, 6.0]).unwrap();
        let rhs = Array1::from_vec(vec![5.0_f64, 7.0]);

        let owned = decompose(&matrix).unwrap();
        let viewed = decompose_view(&matrix.view()).unwrap();
        assert_eq!(owned.l.dim(), viewed.l.dim());
        assert_eq!(owned.u.dim(), viewed.u.dim());

        let solution_owned = solve(&matrix, &rhs).unwrap();
        let solution_view = solve_view(&matrix.view(), &rhs.view()).unwrap();
        for i in 0..rhs.len() {
            assert!((solution_owned[i] - solution_view[i]).abs() < 1e-12);
        }

        let inverse_owned = inverse(&matrix).unwrap();
        let inverse_view = inverse_view(&matrix.view()).unwrap();
        for i in 0..matrix.nrows() {
            for j in 0..matrix.ncols() {
                assert!((inverse_owned[[i, j]] - inverse_view[[i, j]]).abs() < 1e-12);
            }
        }

        let det_owned = determinant(&matrix).unwrap();
        let det_view = determinant_view(&matrix.view()).unwrap();
        assert!((det_owned - det_view).abs() < 1e-12);

        let logdet_owned = log_determinant(&matrix).unwrap();
        let logdet_view = log_determinant_view(&matrix.view()).unwrap();
        assert_eq!(logdet_owned.sign, logdet_view.sign);
        assert!((logdet_owned.ln_abs_det - logdet_view.ln_abs_det).abs() < 1e-12);
    }

    #[test]
    fn complex_lu_paths_solve_inverse_and_determinant() {
        let matrix = Array2::from_shape_vec((2, 2), vec![
            Complex64::new(2.0, 1.0),
            Complex64::new(1.0, -1.0),
            Complex64::new(0.5, 0.25),
            Complex64::new(3.0, -0.5),
        ])
        .unwrap();
        let rhs = Array1::from_vec(vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)]);

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

        let determinant = determinant_complex(&matrix).unwrap();
        assert!(determinant.norm() > 1e-8);

        let solution_view = solve_complex_view(&matrix.view(), &rhs.view()).unwrap();
        let inverse_viewed = inverse_complex_view(&matrix.view()).unwrap();
        let determinant_viewed = determinant_complex_view(&matrix.view()).unwrap();
        for i in 0..rhs.len() {
            assert!((solution_view[i] - solution[i]).norm() < 1e-10);
        }
        for i in 0..2 {
            for j in 0..2 {
                assert!((inverse_viewed[[i, j]] - inverse[[i, j]]).norm() < 1e-10);
            }
        }
        assert!((determinant_viewed - determinant).norm() < 1e-10);
    }
}
