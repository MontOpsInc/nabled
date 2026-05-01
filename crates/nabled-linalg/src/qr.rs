//! QR decomposition over ndarray matrices.

use std::fmt;

use nabled_core::scalar::NabledReal;
use ndarray::{Array1, Array2, ArrayBase, ArrayView1, ArrayView2, DataMut, Ix1, s};
use num_complex::Complex64;

use crate::internal::DenseKernelPolicy;
#[cfg(not(feature = "lapack-provider"))]
#[cfg(any(feature = "magma-system", not(feature = "lapack-provider")))]
use crate::internal::qr_gram_schmidt;
#[cfg(feature = "magma-system")]
use crate::provider::magma;
#[cfg(feature = "magma-system")]
use crate::provider::policy::MagmaProviderPolicy;
use crate::svd::{self, PseudoInverseConfig};

#[cfg(feature = "magma-system")]
#[doc(hidden)]
pub trait QrInternalScalar: NabledReal + magma::MagmaReal {}

#[cfg(feature = "magma-system")]
impl<T> QrInternalScalar for T where T: NabledReal + magma::MagmaReal {}

#[cfg(not(feature = "magma-system"))]
#[doc(hidden)]
pub trait QrInternalScalar: NabledReal {}

#[cfg(not(feature = "magma-system"))]
impl<T> QrInternalScalar for T where T: NabledReal {}

#[cfg(all(feature = "lapack-provider", feature = "magma-system"))]
#[doc(hidden)]
pub trait QrProviderScalar:
    NabledReal + ndarray_linalg::Lapack<Real = Self> + std::ops::AddAssign + magma::MagmaReal
{
}

#[cfg(all(feature = "lapack-provider", feature = "magma-system"))]
impl<T> QrProviderScalar for T where
    T: NabledReal + ndarray_linalg::Lapack<Real = T> + std::ops::AddAssign + magma::MagmaReal
{
}

#[cfg(all(feature = "lapack-provider", not(feature = "magma-system")))]
#[doc(hidden)]
pub trait QrProviderScalar:
    NabledReal + ndarray_linalg::Lapack<Real = Self> + std::ops::AddAssign
{
}

#[cfg(all(feature = "lapack-provider", not(feature = "magma-system")))]
impl<T> QrProviderScalar for T where
    T: NabledReal + ndarray_linalg::Lapack<Real = T> + std::ops::AddAssign
{
}

/// Error types for QR decomposition.
#[derive(Debug, Clone, PartialEq)]
pub enum QRError {
    /// Matrix is empty.
    EmptyMatrix,
    /// Matrix is singular or rank-deficient.
    SingularMatrix,
    /// Convergence failed.
    ConvergenceFailed,
    /// Invalid dimensions.
    InvalidDimensions(String),
    /// Numerical instability detected.
    NumericalInstability,
    /// Invalid user input.
    InvalidInput(String),
}

impl fmt::Display for QRError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QRError::EmptyMatrix => write!(f, "Matrix is empty"),
            QRError::SingularMatrix => write!(f, "Matrix is singular or rank-deficient"),
            QRError::ConvergenceFailed => write!(f, "Convergence failed"),
            QRError::InvalidDimensions(message) => write!(f, "Invalid dimensions: {message}"),
            QRError::NumericalInstability => write!(f, "Numerical instability detected"),
            QRError::InvalidInput(message) => write!(f, "Invalid input: {message}"),
        }
    }
}

impl std::error::Error for QRError {}

#[cfg(feature = "magma-system")]
fn map_qr_magma_error(error: &'static str) -> QRError {
    match error {
        "empty" => QRError::EmptyMatrix,
        "convergence_failed" => QRError::ConvergenceFailed,
        "non_finite" => QRError::NumericalInstability,
        "bad_dimensions" | "invalid_dimensions" => {
            QRError::InvalidDimensions("RHS length must equal matrix rows".to_string())
        }
        _ => QRError::InvalidInput(error.to_string()),
    }
}

/// Configuration for QR decomposition.
#[derive(Debug, Clone)]
pub struct QRConfig<T = f64> {
    /// Tolerance for rank determination.
    pub rank_tolerance: T,
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Enable column pivoting.
    pub use_pivoting:   bool,
}

impl<T: NabledReal> Default for QRConfig<T> {
    fn default() -> Self {
        Self {
            rank_tolerance: T::from_f64(DenseKernelPolicy::BASE_TOLERANCE).unwrap_or(T::epsilon()),
            max_iterations: DenseKernelPolicy::QR_MAX_ITERATIONS,
            use_pivoting:   false,
        }
    }
}

/// Result of QR decomposition.
#[derive(Debug, Clone)]
pub struct QRResult<T = f64> {
    /// Orthogonal matrix Q.
    pub q:    Array2<T>,
    /// Upper triangular matrix R.
    pub r:    Array2<T>,
    /// Column permutation matrix when pivoting is requested.
    pub p:    Option<Array2<T>>,
    /// Numerical rank.
    pub rank: usize,
}

fn permutation_matrix_from_order<T: NabledReal>(order: &[usize]) -> Array2<T> {
    let n = order.len();
    let mut permutation = Array2::<T>::zeros((n, n));
    for (col, &original_col) in order.iter().enumerate() {
        permutation[[original_col, col]] = T::one();
    }
    permutation
}

fn complex_permutation_matrix_from_order(order: &[usize]) -> Array2<Complex64> {
    let n = order.len();
    let mut permutation = Array2::<Complex64>::zeros((n, n));
    for (col, &original_col) in order.iter().enumerate() {
        permutation[[original_col, col]] = Complex64::new(1.0, 0.0);
    }
    permutation
}

fn validate_qr_input<T: NabledReal>(matrix: &ArrayView2<'_, T>) -> Result<(), QRError> {
    if matrix.is_empty() {
        return Err(QRError::EmptyMatrix);
    }
    if matrix.iter().any(|value| !value.is_finite()) {
        return Err(QRError::NumericalInstability);
    }
    Ok(())
}

fn validate_qr_complex_input(matrix: &ArrayView2<'_, Complex64>) -> Result<(), QRError> {
    if matrix.is_empty() {
        return Err(QRError::EmptyMatrix);
    }
    if matrix.iter().any(|value| !value.re.is_finite() || !value.im.is_finite()) {
        return Err(QRError::NumericalInstability);
    }
    Ok(())
}

fn decompose_pivoted_internal<T: NabledReal>(
    matrix: &ArrayView2<'_, T>,
    config: &QRConfig<T>,
) -> Result<QRResult<T>, QRError> {
    validate_qr_input(matrix)?;

    let (rows, cols) = matrix.dim();
    let tolerance = config
        .rank_tolerance
        .max(T::from_f64(DenseKernelPolicy::BASE_TOLERANCE).unwrap_or(T::epsilon()));
    let mut residual = matrix.to_owned();
    let mut q = Array2::<T>::zeros((rows, cols));
    let mut r = Array2::<T>::zeros((cols, cols));
    let mut order = (0..cols).collect::<Vec<_>>();
    let mut residual_norm_sq = (0..cols)
        .map(|col| {
            residual
                .column(col)
                .iter()
                .map(|value| *value * *value)
                .fold(T::zero(), |sum, value| sum + value)
        })
        .collect::<Vec<_>>();
    let mut rank = 0_usize;

    for k in 0..cols {
        let mut pivot = k;
        let mut pivot_norm = residual_norm_sq[k];
        for (j, &norm_sq) in residual_norm_sq.iter().enumerate().skip(k + 1) {
            if norm_sq > pivot_norm {
                pivot = j;
                pivot_norm = norm_sq;
            }
        }

        if pivot != k {
            let col_k = residual.column(k).to_owned();
            let col_pivot = residual.column(pivot).to_owned();
            residual.column_mut(k).assign(&col_pivot);
            residual.column_mut(pivot).assign(&col_k);
            residual_norm_sq.swap(k, pivot);
            order.swap(k, pivot);

            for i in 0..k {
                let temp = r[[i, k]];
                r[[i, k]] = r[[i, pivot]];
                r[[i, pivot]] = temp;
            }
        }

        let v = residual.column(k).to_owned();

        let norm_sq =
            v.iter().map(|value| *value * *value).fold(T::zero(), |sum, value| sum + value);
        let norm = norm_sq.sqrt();
        r[[k, k]] = norm;
        if norm > tolerance {
            rank += 1;
            for row in 0..rows {
                q[[row, k]] = v[row] / norm;
            }
            for j in (k + 1)..cols {
                let mut projection = T::zero();
                for row in 0..rows {
                    projection += q[[row, k]] * residual[[row, j]];
                }
                r[[k, j]] = projection;
                for row in 0..rows {
                    residual[[row, j]] -= q[[row, k]] * projection;
                }
                residual_norm_sq[j] = residual
                    .column(j)
                    .iter()
                    .map(|value| *value * *value)
                    .fold(T::zero(), |sum, value| sum + value);
            }
        } else {
            for j in (k + 1)..cols {
                r[[k, j]] = T::zero();
            }
        }
        residual_norm_sq[k] = T::zero();
    }

    Ok(QRResult { q, r, p: Some(permutation_matrix_from_order::<T>(&order)), rank })
}

fn decompose_complex_pivoted_internal(
    matrix: &ArrayView2<'_, Complex64>,
    config: &QRConfig<f64>,
) -> Result<QRResult<Complex64>, QRError> {
    validate_qr_complex_input(matrix)?;

    let (rows, cols) = matrix.dim();
    let tolerance = DenseKernelPolicy::rank_tolerance(config.rank_tolerance);
    let mut residual = matrix.to_owned();
    let mut q = Array2::<Complex64>::zeros((rows, cols));
    let mut r = Array2::<Complex64>::zeros((cols, cols));
    let mut order = (0..cols).collect::<Vec<_>>();
    let mut residual_norm_sq = (0..cols)
        .map(|col| residual.column(col).iter().map(Complex64::norm_sqr).sum::<f64>())
        .collect::<Vec<_>>();
    let mut rank = 0_usize;

    for k in 0..cols {
        let mut pivot = k;
        let mut pivot_norm = residual_norm_sq[k];
        for (j, &norm_sq) in residual_norm_sq.iter().enumerate().skip(k + 1) {
            if norm_sq > pivot_norm {
                pivot = j;
                pivot_norm = norm_sq;
            }
        }

        if pivot != k {
            let col_k = residual.column(k).to_owned();
            let col_pivot = residual.column(pivot).to_owned();
            residual.column_mut(k).assign(&col_pivot);
            residual.column_mut(pivot).assign(&col_k);
            residual_norm_sq.swap(k, pivot);
            order.swap(k, pivot);

            for i in 0..k {
                let temp = r[[i, k]];
                r[[i, k]] = r[[i, pivot]];
                r[[i, pivot]] = temp;
            }
        }

        let v = residual.column(k).to_owned();

        let norm_sq = v.iter().map(Complex64::norm_sqr).sum::<f64>();
        let norm = norm_sq.sqrt();
        r[[k, k]] = Complex64::new(norm, 0.0);
        if norm > tolerance {
            rank += 1;
            for row in 0..rows {
                q[[row, k]] = v[row] / norm;
            }
            for j in (k + 1)..cols {
                let mut projection = Complex64::new(0.0, 0.0);
                for row in 0..rows {
                    projection += q[[row, k]].conj() * residual[[row, j]];
                }
                r[[k, j]] = projection;
                for row in 0..rows {
                    residual[[row, j]] -= q[[row, k]] * projection;
                }
                residual_norm_sq[j] =
                    residual.column(j).iter().map(Complex64::norm_sqr).sum::<f64>();
            }
        } else {
            for j in (k + 1)..cols {
                r[[k, j]] = Complex64::new(0.0, 0.0);
            }
        }
        residual_norm_sq[k] = 0.0;
    }

    Ok(QRResult { q, r, p: Some(complex_permutation_matrix_from_order(&order)), rank })
}

#[cfg(all(not(feature = "lapack-provider"), feature = "magma-system"))]
fn decompose_internal<T: NabledReal + magma::MagmaReal>(
    matrix: &ArrayView2<'_, T>,
    config: &QRConfig<T>,
) -> Result<QRResult<T>, QRError> {
    if config.use_pivoting {
        return decompose_pivoted_internal(matrix, config);
    }
    validate_qr_input(matrix)?;

    // MAGMA's QR provider path is used for sufficiently large
    // overdetermined/square real matrices; underdetermined/small shapes
    // remain on the internal path to avoid fixed provider overhead.
    if matrix.nrows() >= matrix.ncols()
        && MagmaProviderPolicy::prefer_decomposition(matrix.nrows(), matrix.ncols())
    {
        let tolerance = config
            .rank_tolerance
            .max(T::from_f64(DenseKernelPolicy::BASE_TOLERANCE).unwrap_or(T::epsilon()));
        match magma::qr_decompose(matrix, tolerance) {
            Ok((q, r, rank)) => return Ok(QRResult { q, r, p: None, rank }),
            Err(error) => {
                if MagmaProviderPolicy::fail_fast_mode() {
                    return Err(map_qr_magma_error(error));
                }
            }
        }
    }

    let (q, r, rank) = qr_gram_schmidt(
        matrix,
        config
            .rank_tolerance
            .max(T::from_f64(DenseKernelPolicy::BASE_TOLERANCE).unwrap_or(T::epsilon())),
    );
    Ok(QRResult { q, r, p: None, rank })
}

#[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
fn decompose_internal<T: NabledReal>(
    matrix: &ArrayView2<'_, T>,
    config: &QRConfig<T>,
) -> Result<QRResult<T>, QRError> {
    if config.use_pivoting {
        return decompose_pivoted_internal(matrix, config);
    }
    validate_qr_input(matrix)?;

    let (q, r, rank) = qr_gram_schmidt(
        matrix,
        config
            .rank_tolerance
            .max(T::from_f64(DenseKernelPolicy::BASE_TOLERANCE).unwrap_or(T::epsilon())),
    );
    Ok(QRResult { q, r, p: None, rank })
}

#[cfg(feature = "lapack-provider")]
fn decompose_provider<T>(
    matrix: &ArrayView2<'_, T>,
    config: &QRConfig<T>,
) -> Result<QRResult<T>, QRError>
where
    T: QrProviderScalar,
{
    use ndarray_linalg::QR as _;

    if config.use_pivoting {
        return decompose_pivoted_internal(matrix, config);
    }

    validate_qr_input(matrix)?;

    #[cfg(feature = "magma-system")]
    if matrix.nrows() >= matrix.ncols()
        && MagmaProviderPolicy::prefer_decomposition(matrix.nrows(), matrix.ncols())
    {
        let tolerance = config
            .rank_tolerance
            .max(T::from_f64(DenseKernelPolicy::BASE_TOLERANCE).unwrap_or(T::epsilon()));
        match magma::qr_decompose(matrix, tolerance) {
            Ok((q, r, rank)) => return Ok(QRResult { q, r, p: None, rank }),
            Err(error) => {
                if MagmaProviderPolicy::fail_fast_mode() {
                    return Err(map_qr_magma_error(error));
                }
            }
        }
    }

    let (q, r) = matrix.qr().map_err(|_| QRError::ConvergenceFailed)?;

    let diagonal = r.nrows().min(r.ncols());
    let tolerance = config
        .rank_tolerance
        .max(T::from_f64(DenseKernelPolicy::BASE_TOLERANCE).unwrap_or(T::epsilon()));
    let rank = (0..diagonal)
        .filter(|&index| num_traits::Float::abs(r[[index, index]]) > tolerance)
        .count();

    Ok(QRResult { q, r, p: None, rank })
}

#[cfg(all(feature = "lapack-provider", feature = "magma-system"))]
#[expect(clippy::many_single_char_names)]
fn solve_least_squares_provider<T>(
    matrix: &ArrayView2<'_, T>,
    rhs: &ArrayView1<'_, T>,
) -> Result<Array1<T>, QRError>
where
    T: QrProviderScalar,
{
    use ndarray_linalg::LeastSquaresSvd as _;

    if matrix.nrows() >= matrix.ncols()
        && MagmaProviderPolicy::prefer_decomposition(matrix.nrows(), matrix.ncols())
    {
        let tolerance =
            T::from_f64(DenseKernelPolicy::BASE_TOLERANCE).unwrap_or_else(|| T::epsilon());
        match magma::qr_decompose(matrix, tolerance) {
            Ok((q, r, rank)) => {
                let n = matrix.ncols();
                if rank < n {
                    return Err(QRError::SingularMatrix);
                }

                let mut y = Array1::<T>::zeros(n);
                for i in 0..n {
                    let mut dot = T::zero();
                    for row in 0..matrix.nrows() {
                        dot += q[[row, i]] * rhs[row];
                    }
                    y[i] = dot;
                }

                let mut solution = Array1::<T>::zeros(n);
                for i_rev in 0..n {
                    let i = n - 1 - i_rev;
                    let mut sum = y[i];
                    for j in (i + 1)..n {
                        sum -= r[[i, j]] * solution[j];
                    }
                    let diagonal = r[[i, i]];
                    if num_traits::Float::abs(diagonal) <= tolerance {
                        return Err(QRError::SingularMatrix);
                    }
                    solution[i] = sum / diagonal;
                }
                return Ok(solution);
            }
            Err(error) => {
                if MagmaProviderPolicy::fail_fast_mode() {
                    return Err(map_qr_magma_error(error));
                }
            }
        }
    }

    let result = matrix.least_squares(rhs).map_err(|_| QRError::ConvergenceFailed)?;
    let rank = usize::try_from(result.rank).map_err(|_| QRError::ConvergenceFailed)?;
    if rank < matrix.ncols() {
        return Err(QRError::SingularMatrix);
    }
    Ok(result.solution)
}

#[cfg(all(feature = "lapack-provider", not(feature = "magma-system")))]
fn solve_least_squares_provider<T>(
    matrix: &ArrayView2<'_, T>,
    rhs: &ArrayView1<'_, T>,
) -> Result<Array1<T>, QRError>
where
    T: QrProviderScalar,
{
    use ndarray_linalg::LeastSquaresSvd as _;

    let result = matrix.least_squares(rhs).map_err(|_| QRError::ConvergenceFailed)?;
    let rank = usize::try_from(result.rank).map_err(|_| QRError::ConvergenceFailed)?;
    if rank < matrix.ncols() {
        return Err(QRError::SingularMatrix);
    }
    Ok(result.solution)
}

#[cfg(not(feature = "lapack-provider"))]
fn decompose_complex_internal(
    matrix: &ArrayView2<'_, Complex64>,
    config: &QRConfig<f64>,
) -> Result<QRResult<Complex64>, QRError> {
    if config.use_pivoting {
        return decompose_complex_pivoted_internal(matrix, config);
    }
    validate_qr_complex_input(matrix)?;

    let rows = matrix.nrows();
    let cols = matrix.ncols();
    let mut q = Array2::<Complex64>::zeros((rows, cols));
    let mut r = Array2::<Complex64>::zeros((cols, cols));
    let mut rank = 0_usize;
    let tolerance = DenseKernelPolicy::rank_tolerance(config.rank_tolerance);
    let mut v = Array1::<Complex64>::zeros(rows);

    for j in 0..cols {
        for row in 0..rows {
            v[row] = matrix[[row, j]];
        }
        for i in 0..j {
            let mut projection = Complex64::new(0.0, 0.0);
            for row in 0..rows {
                projection += q[[row, i]].conj() * v[row];
            }
            r[[i, j]] = projection;
            for row in 0..rows {
                v[row] -= projection * q[[row, i]];
            }
        }

        let norm = v.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt();
        r[[j, j]] = Complex64::new(norm, 0.0);
        if norm > tolerance {
            rank += 1;
            for row in 0..rows {
                q[[row, j]] = v[row] / norm;
            }
        }
    }

    Ok(QRResult { q, r, p: None, rank })
}

#[cfg(feature = "lapack-provider")]
fn decompose_complex_lapack(
    matrix: &ArrayView2<'_, Complex64>,
    config: &QRConfig<f64>,
) -> Result<QRResult<Complex64>, QRError> {
    use ndarray_linalg::QR as _;

    if config.use_pivoting {
        return decompose_complex_pivoted_internal(matrix, config);
    }

    validate_qr_complex_input(matrix)?;
    let (q, r) = matrix.qr().map_err(|_| QRError::ConvergenceFailed)?;
    let diagonal = r.nrows().min(r.ncols());
    let rank = (0..diagonal)
        .filter(|&index| {
            r[[index, index]].norm() > DenseKernelPolicy::rank_tolerance(config.rank_tolerance)
        })
        .count();
    Ok(QRResult { q, r, p: None, rank })
}

#[cfg(all(feature = "magma-system", feature = "lapack-provider"))]
fn decompose_complex_provider(
    matrix: &ArrayView2<'_, Complex64>,
    config: &QRConfig<f64>,
) -> Result<QRResult<Complex64>, QRError> {
    if config.use_pivoting {
        return decompose_complex_pivoted_internal(matrix, config);
    }

    validate_qr_complex_input(matrix)?;
    if matrix.nrows() >= matrix.ncols()
        && MagmaProviderPolicy::prefer_decomposition(matrix.nrows(), matrix.ncols())
    {
        let tolerance = DenseKernelPolicy::rank_tolerance(config.rank_tolerance);
        match magma::qr_decompose_complex(matrix, tolerance) {
            Ok((q, r, rank)) => return Ok(QRResult { q, r, p: None, rank }),
            Err(error) => {
                if MagmaProviderPolicy::fail_fast_mode() {
                    return Err(map_qr_magma_error(error));
                }
                return decompose_complex_lapack(matrix, config);
            }
        }
    }

    decompose_complex_lapack(matrix, config)
}

#[cfg(all(feature = "magma-system", not(feature = "lapack-provider")))]
fn decompose_complex_provider(
    matrix: &ArrayView2<'_, Complex64>,
    config: &QRConfig<f64>,
) -> Result<QRResult<Complex64>, QRError> {
    if config.use_pivoting {
        return decompose_complex_pivoted_internal(matrix, config);
    }

    validate_qr_complex_input(matrix)?;
    if matrix.nrows() >= matrix.ncols()
        && MagmaProviderPolicy::prefer_decomposition(matrix.nrows(), matrix.ncols())
    {
        let tolerance = DenseKernelPolicy::rank_tolerance(config.rank_tolerance);
        match magma::qr_decompose_complex(matrix, tolerance) {
            Ok((q, r, rank)) => return Ok(QRResult { q, r, p: None, rank }),
            Err(error) => {
                if MagmaProviderPolicy::fail_fast_mode() {
                    return Err(map_qr_magma_error(error));
                }
            }
        }
    }

    decompose_complex_internal(matrix, config)
}

#[cfg(all(feature = "lapack-provider", not(feature = "magma-system")))]
fn decompose_complex_provider(
    matrix: &ArrayView2<'_, Complex64>,
    config: &QRConfig<f64>,
) -> Result<QRResult<Complex64>, QRError> {
    decompose_complex_lapack(matrix, config)
}

/// Compute full QR decomposition.
///
/// # Errors
/// Returns an error if the matrix is empty or non-finite.
#[cfg(feature = "lapack-provider")]
pub fn decompose<T>(matrix: &Array2<T>, config: &QRConfig<T>) -> Result<QRResult<T>, QRError>
where
    T: QrProviderScalar,
{
    decompose_provider(&matrix.view(), config)
}

/// Compute full QR decomposition.
///
/// # Errors
/// Returns an error if the matrix is empty or non-finite.
#[cfg(not(feature = "lapack-provider"))]
pub fn decompose<T: QrInternalScalar>(
    matrix: &Array2<T>,
    config: &QRConfig<T>,
) -> Result<QRResult<T>, QRError> {
    decompose_internal(&matrix.view(), config)
}

/// Compute full QR decomposition from a matrix view.
///
/// # Errors
/// Returns an error if decomposition fails.
#[cfg(feature = "lapack-provider")]
pub fn decompose_view<T>(
    matrix: &ArrayView2<'_, T>,
    config: &QRConfig<T>,
) -> Result<QRResult<T>, QRError>
where
    T: QrProviderScalar,
{
    decompose_provider(matrix, config)
}

/// Compute full QR decomposition from a matrix view.
///
/// # Errors
/// Returns an error if decomposition fails.
#[cfg(not(feature = "lapack-provider"))]
pub fn decompose_view<T: QrInternalScalar>(
    matrix: &ArrayView2<'_, T>,
    config: &QRConfig<T>,
) -> Result<QRResult<T>, QRError> {
    decompose_internal(matrix, config)
}

/// Compute full QR decomposition for complex matrices.
///
/// # Errors
/// Returns an error if decomposition fails.
pub fn decompose_complex(
    matrix: &Array2<Complex64>,
    config: &QRConfig<f64>,
) -> Result<QRResult<Complex64>, QRError> {
    decompose_complex_impl(&matrix.view(), config)
}

fn decompose_complex_impl(
    matrix: &ArrayView2<'_, Complex64>,
    config: &QRConfig<f64>,
) -> Result<QRResult<Complex64>, QRError> {
    #[cfg(feature = "magma-system")]
    {
        decompose_complex_provider(matrix, config)
    }
    #[cfg(all(feature = "lapack-provider", not(feature = "magma-system")))]
    {
        decompose_complex_provider(matrix, config)
    }
    #[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
    {
        decompose_complex_internal(matrix, config)
    }
}

/// Compute full complex QR decomposition from a matrix view.
///
/// # Errors
/// Returns an error if decomposition fails.
pub fn decompose_complex_view(
    matrix: &ArrayView2<'_, Complex64>,
    config: &QRConfig<f64>,
) -> Result<QRResult<Complex64>, QRError> {
    decompose_complex_impl(matrix, config)
}

/// Compute reduced (economy) QR decomposition.
///
/// # Errors
/// Returns an error if the matrix is empty or non-finite.
#[cfg(feature = "lapack-provider")]
pub fn decompose_reduced<T>(
    matrix: &Array2<T>,
    config: &QRConfig<T>,
) -> Result<QRResult<T>, QRError>
where
    T: QrProviderScalar,
{
    decompose_reduced_view(&matrix.view(), config)
}

/// Compute reduced (economy) QR decomposition from a view.
///
/// # Errors
/// Returns an error if the matrix is empty or non-finite.
#[cfg(feature = "lapack-provider")]
pub fn decompose_reduced_view<T>(
    matrix: &ArrayView2<'_, T>,
    config: &QRConfig<T>,
) -> Result<QRResult<T>, QRError>
where
    T: QrProviderScalar,
{
    let full = decompose_view(matrix, config)?;
    let keep = matrix.nrows().min(matrix.ncols());
    Ok(QRResult {
        q:    full.q.slice(s![.., ..keep]).to_owned(),
        r:    full.r.slice(s![..keep, ..]).to_owned(),
        p:    full.p,
        rank: full.rank.min(keep),
    })
}

/// Compute reduced (economy) QR decomposition.
///
/// # Errors
/// Returns an error if the matrix is empty or non-finite.
#[cfg(not(feature = "lapack-provider"))]
pub fn decompose_reduced<T: QrInternalScalar>(
    matrix: &Array2<T>,
    config: &QRConfig<T>,
) -> Result<QRResult<T>, QRError> {
    decompose_reduced_view(&matrix.view(), config)
}

/// Compute reduced (economy) QR decomposition from a view.
///
/// # Errors
/// Returns an error if the matrix is empty or non-finite.
#[cfg(not(feature = "lapack-provider"))]
pub fn decompose_reduced_view<T: QrInternalScalar>(
    matrix: &ArrayView2<'_, T>,
    config: &QRConfig<T>,
) -> Result<QRResult<T>, QRError> {
    let full = decompose_view(matrix, config)?;
    let keep = matrix.nrows().min(matrix.ncols());
    Ok(QRResult {
        q:    full.q.slice(s![.., ..keep]).to_owned(),
        r:    full.r.slice(s![..keep, ..]).to_owned(),
        p:    full.p,
        rank: full.rank.min(keep),
    })
}

/// Compute QR decomposition with column pivoting.
///
/// # Errors
/// Returns an error if decomposition fails.
#[cfg(feature = "lapack-provider")]
pub fn decompose_with_pivoting<T>(
    matrix: &Array2<T>,
    config: &QRConfig<T>,
) -> Result<QRResult<T>, QRError>
where
    T: QrProviderScalar,
{
    decompose_with_pivoting_view(&matrix.view(), config)
}

/// Compute QR decomposition with column pivoting from a view.
///
/// # Errors
/// Returns an error if decomposition fails.
#[cfg(feature = "lapack-provider")]
pub fn decompose_with_pivoting_view<T>(
    matrix: &ArrayView2<'_, T>,
    config: &QRConfig<T>,
) -> Result<QRResult<T>, QRError>
where
    T: QrProviderScalar,
{
    let mut adjusted = config.clone();
    adjusted.use_pivoting = true;
    decompose_view(matrix, &adjusted)
}

/// Compute QR decomposition with column pivoting.
///
/// # Errors
/// Returns an error if decomposition fails.
#[cfg(not(feature = "lapack-provider"))]
pub fn decompose_with_pivoting<T: QrInternalScalar>(
    matrix: &Array2<T>,
    config: &QRConfig<T>,
) -> Result<QRResult<T>, QRError> {
    decompose_with_pivoting_view(&matrix.view(), config)
}

/// Compute QR decomposition with column pivoting from a view.
///
/// # Errors
/// Returns an error if decomposition fails.
#[cfg(not(feature = "lapack-provider"))]
pub fn decompose_with_pivoting_view<T: QrInternalScalar>(
    matrix: &ArrayView2<'_, T>,
    config: &QRConfig<T>,
) -> Result<QRResult<T>, QRError> {
    let mut adjusted = config.clone();
    adjusted.use_pivoting = true;
    decompose_view(matrix, &adjusted)
}

/// Solve least squares `argmin ||Ax - b||_2`.
///
/// For underdetermined systems (`m < n`), this returns the minimum-norm solution.
///
/// # Errors
/// Returns an error for invalid dimensions or rank-deficient systems.
#[cfg(feature = "lapack-provider")]
pub fn solve_least_squares<T>(
    matrix: &Array2<T>,
    rhs: &Array1<T>,
    config: &QRConfig<T>,
) -> Result<Array1<T>, QRError>
where
    T: QrProviderScalar,
{
    solve_least_squares_impl(&matrix.view(), &rhs.view(), config)
}

/// Solve least squares `argmin ||Ax - b||_2`.
///
/// For underdetermined systems (`m < n`), this returns the minimum-norm solution.
///
/// # Errors
/// Returns an error for invalid dimensions or rank-deficient systems.
#[cfg(not(feature = "lapack-provider"))]
pub fn solve_least_squares<T: QrInternalScalar>(
    matrix: &Array2<T>,
    rhs: &Array1<T>,
    config: &QRConfig<T>,
) -> Result<Array1<T>, QRError> {
    solve_least_squares_impl(&matrix.view(), &rhs.view(), config)
}

#[cfg(feature = "lapack-provider")]
fn solve_least_squares_impl<T>(
    matrix: &ArrayView2<'_, T>,
    rhs: &ArrayView1<'_, T>,
    config: &QRConfig<T>,
) -> Result<Array1<T>, QRError>
where
    T: QrProviderScalar,
{
    validate_qr_input(matrix)?;
    if rhs.len() != matrix.nrows() {
        return Err(QRError::InvalidDimensions("RHS length must equal matrix rows".to_string()));
    }

    if matrix.nrows() < matrix.ncols() {
        // Some LAPACK/OpenBLAS builds fail to converge on small underdetermined SVDs.
        // Use the internal SVD fallback here so least-squares keeps a stable
        // minimum-norm contract across provider builds.
        let svd =
            svd::decompose_internal_fallback(matrix).map_err(|_| QRError::ConvergenceFailed)?;
        let required_rank = matrix.nrows();
        let computed_rank = svd::rank(
            &svd,
            Some(
                config
                    .rank_tolerance
                    .max(T::from_f64(DenseKernelPolicy::BASE_TOLERANCE).unwrap_or(T::epsilon())),
            ),
        );
        if computed_rank < required_rank {
            return Err(QRError::SingularMatrix);
        }

        let mut pseudo_inverse = Array2::<T>::zeros((matrix.ncols(), matrix.nrows()));
        svd::pseudo_inverse_from_svd_view_into(
            &svd.u.view(),
            &svd.singular_values.view(),
            &svd.vt.view(),
            &PseudoInverseConfig {
                tolerance: Some(
                    config.rank_tolerance.max(
                        T::from_f64(DenseKernelPolicy::BASE_TOLERANCE).unwrap_or(T::epsilon()),
                    ),
                ),
            },
            &mut pseudo_inverse,
        )
        .map_err(|_| QRError::ConvergenceFailed)?;

        return Ok(pseudo_inverse.dot(rhs));
    }

    if !config.use_pivoting {
        return solve_least_squares_provider(matrix, rhs);
    }

    let full = decompose_view(matrix, config)?;
    let keep = matrix.nrows().min(matrix.ncols());
    let qr = QRResult {
        q:    full.q.slice(s![.., ..keep]).to_owned(),
        r:    full.r.slice(s![..keep, ..]).to_owned(),
        p:    full.p,
        rank: full.rank.min(keep),
    };
    let n = matrix.ncols();
    if qr.rank < n {
        return Err(QRError::SingularMatrix);
    }

    let mut y = Array1::<T>::zeros(n);
    for i in 0..n {
        let mut dot = T::zero();
        for row in 0..matrix.nrows() {
            dot += qr.q[[row, i]] * rhs[row];
        }
        y[i] = dot;
    }

    let mut permuted_solution = Array1::<T>::zeros(n);
    for i_rev in 0..n {
        let i = n - 1 - i_rev;
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= qr.r[[i, j]] * permuted_solution[j];
        }
        let diagonal = qr.r[[i, i]];
        if num_traits::Float::abs(diagonal)
            <= config
                .rank_tolerance
                .max(T::from_f64(DenseKernelPolicy::BASE_TOLERANCE).unwrap_or(T::epsilon()))
        {
            return Err(QRError::SingularMatrix);
        }
        permuted_solution[i] = sum / diagonal;
    }

    if let Some(permutation) = qr.p {
        Ok(permutation.dot(&permuted_solution))
    } else {
        Ok(permuted_solution)
    }
}

#[cfg(not(feature = "lapack-provider"))]
fn solve_least_squares_impl<T: QrInternalScalar>(
    matrix: &ArrayView2<'_, T>,
    rhs: &ArrayView1<'_, T>,
    config: &QRConfig<T>,
) -> Result<Array1<T>, QRError> {
    validate_qr_input(matrix)?;
    if rhs.len() != matrix.nrows() {
        return Err(QRError::InvalidDimensions("RHS length must equal matrix rows".to_string()));
    }

    if matrix.nrows() < matrix.ncols() {
        let svd = svd::decompose_view(matrix).map_err(|_| QRError::ConvergenceFailed)?;
        let required_rank = matrix.nrows();
        let computed_rank = svd::rank(
            &svd,
            Some(
                config
                    .rank_tolerance
                    .max(T::from_f64(DenseKernelPolicy::BASE_TOLERANCE).unwrap_or(T::epsilon())),
            ),
        );
        if computed_rank < required_rank {
            return Err(QRError::SingularMatrix);
        }

        let mut pseudo_inverse = Array2::<T>::zeros((matrix.ncols(), matrix.nrows()));
        svd::pseudo_inverse_from_svd_view_into(
            &svd.u.view(),
            &svd.singular_values.view(),
            &svd.vt.view(),
            &PseudoInverseConfig {
                tolerance: Some(
                    config.rank_tolerance.max(
                        T::from_f64(DenseKernelPolicy::BASE_TOLERANCE).unwrap_or(T::epsilon()),
                    ),
                ),
            },
            &mut pseudo_inverse,
        )
        .map_err(|_| QRError::ConvergenceFailed)?;

        return Ok(pseudo_inverse.dot(rhs));
    }

    let full = decompose_view(matrix, config)?;
    let keep = matrix.nrows().min(matrix.ncols());
    let qr = QRResult {
        q:    full.q.slice(s![.., ..keep]).to_owned(),
        r:    full.r.slice(s![..keep, ..]).to_owned(),
        p:    full.p,
        rank: full.rank.min(keep),
    };
    let n = matrix.ncols();
    if qr.rank < n {
        return Err(QRError::SingularMatrix);
    }

    let mut y = Array1::<T>::zeros(n);
    for i in 0..n {
        let mut dot = T::zero();
        for row in 0..matrix.nrows() {
            dot += qr.q[[row, i]] * rhs[row];
        }
        y[i] = dot;
    }

    let mut permuted_solution = Array1::<T>::zeros(n);
    for i_rev in 0..n {
        let i = n - 1 - i_rev;
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= qr.r[[i, j]] * permuted_solution[j];
        }
        let diagonal = qr.r[[i, i]];
        if num_traits::Float::abs(diagonal)
            <= config
                .rank_tolerance
                .max(T::from_f64(DenseKernelPolicy::BASE_TOLERANCE).unwrap_or(T::epsilon()))
        {
            return Err(QRError::SingularMatrix);
        }
        permuted_solution[i] = sum / diagonal;
    }

    if let Some(permutation) = qr.p {
        Ok(permutation.dot(&permuted_solution))
    } else {
        Ok(permuted_solution)
    }
}

/// Solve least squares directly from precomputed QR factors into caller-provided `output`.
///
/// # Errors
/// Returns an error if the QR factors are inconsistent, correspond to an underdetermined reduced
/// QR result, the system is rank-deficient, or `output` has the wrong length.
pub fn solve_least_squares_from_qr_result_view_into<T, S>(
    qr: &QRResult<T>,
    rhs: &ArrayView1<'_, T>,
    config: &QRConfig<T>,
    output: &mut ArrayBase<S, Ix1>,
) -> Result<(), QRError>
where
    T: NabledReal,
    S: DataMut<Elem = T>,
{
    solve_least_squares_from_qr_factors_view_into(
        &qr.q.view(),
        &qr.r.view(),
        qr.p.as_ref().map(|permutation| permutation.view()),
        rhs,
        config,
        output,
    )
}

/// Solve least squares directly from borrowed QR factor views into caller-provided `output`.
///
/// # Errors
/// Returns an error if the QR factors are inconsistent, correspond to an underdetermined reduced
/// QR result, the system is rank-deficient, or `output` has the wrong length.
pub fn solve_least_squares_from_qr_factors_view_into<T, S>(
    q_factors: &ArrayView2<'_, T>,
    r_factors: &ArrayView2<'_, T>,
    permutation: Option<ArrayView2<'_, T>>,
    rhs: &ArrayView1<'_, T>,
    config: &QRConfig<T>,
    output: &mut ArrayBase<S, Ix1>,
) -> Result<(), QRError>
where
    T: NabledReal,
    S: DataMut<Elem = T>,
{
    if q_factors.ncols() != r_factors.nrows() {
        return Err(QRError::InvalidDimensions("q.ncols() must equal r.nrows()".to_string()));
    }
    if rhs.len() != q_factors.nrows() {
        return Err(QRError::InvalidDimensions("RHS length must equal q rows".to_string()));
    }
    if output.len() != r_factors.ncols() {
        return Err(QRError::InvalidDimensions("output length must equal r columns".to_string()));
    }
    if q_factors.ncols() < r_factors.ncols() {
        return Err(QRError::InvalidInput(
            "QR factors do not retain enough columns for underdetermined least-squares; pass the \
             original matrix"
                .to_string(),
        ));
    }

    let column_count = r_factors.ncols();
    let mut projected_rhs = Array1::<T>::zeros(column_count);
    for col in 0..column_count {
        let mut dot = T::zero();
        for row in 0..q_factors.nrows() {
            dot += q_factors[[row, col]] * rhs[row];
        }
        projected_rhs[col] = dot;
    }

    let mut permuted_solution = Array1::<T>::zeros(column_count);
    for reverse_col in 0..column_count {
        let col = column_count - 1 - reverse_col;
        let mut sum = projected_rhs[col];
        for upper_col in (col + 1)..column_count {
            sum -= r_factors[[col, upper_col]] * permuted_solution[upper_col];
        }
        let diagonal = r_factors[[col, col]];
        if num_traits::Float::abs(diagonal)
            <= config
                .rank_tolerance
                .max(T::from_f64(DenseKernelPolicy::BASE_TOLERANCE).unwrap_or(T::epsilon()))
        {
            return Err(QRError::SingularMatrix);
        }
        permuted_solution[col] = sum / diagonal;
    }

    if let Some(permutation) = permutation {
        if permutation.nrows() != column_count || permutation.ncols() != column_count {
            return Err(QRError::InvalidDimensions(
                "permutation shape must match r column dimensions".to_string(),
            ));
        }
        let solution = permutation.dot(&permuted_solution);
        output.assign(&solution);
    } else {
        output.assign(&permuted_solution);
    }
    Ok(())
}

/// Solve least squares directly from precomputed QR factors.
///
/// # Errors
/// Returns an error if the QR factors are inconsistent, correspond to an underdetermined reduced
/// QR result, or the system is rank-deficient.
pub fn solve_least_squares_from_qr_result_view<T: NabledReal>(
    qr: &QRResult<T>,
    rhs: &ArrayView1<'_, T>,
    config: &QRConfig<T>,
) -> Result<Array1<T>, QRError> {
    let mut output = Array1::<T>::zeros(qr.r.ncols());
    solve_least_squares_from_qr_result_view_into(qr, rhs, config, &mut output)?;
    Ok(output)
}

/// Solve least squares directly from borrowed QR factor views.
///
/// # Errors
/// Returns an error if the QR factors are inconsistent, correspond to an underdetermined reduced
/// QR result, or the system is rank-deficient.
pub fn solve_least_squares_from_qr_factors_view<T: NabledReal>(
    q: &ArrayView2<'_, T>,
    r: &ArrayView2<'_, T>,
    permutation: Option<ArrayView2<'_, T>>,
    rhs: &ArrayView1<'_, T>,
    config: &QRConfig<T>,
) -> Result<Array1<T>, QRError> {
    let mut output = Array1::<T>::zeros(r.ncols());
    solve_least_squares_from_qr_factors_view_into(q, r, permutation, rhs, config, &mut output)?;
    Ok(output)
}

/// Solve least squares from matrix/vector views.
///
/// # Errors
/// Returns an error for invalid dimensions or rank-deficient systems.
#[cfg(feature = "lapack-provider")]
pub fn solve_least_squares_view<T>(
    matrix: &ArrayView2<'_, T>,
    rhs: &ArrayView1<'_, T>,
    config: &QRConfig<T>,
) -> Result<Array1<T>, QRError>
where
    T: QrProviderScalar,
{
    solve_least_squares_impl(matrix, rhs, config)
}

/// Solve least squares from matrix/vector views.
///
/// # Errors
/// Returns an error for invalid dimensions or rank-deficient systems.
#[cfg(not(feature = "lapack-provider"))]
pub fn solve_least_squares_view<T: QrInternalScalar>(
    matrix: &ArrayView2<'_, T>,
    rhs: &ArrayView1<'_, T>,
    config: &QRConfig<T>,
) -> Result<Array1<T>, QRError> {
    solve_least_squares_impl(matrix, rhs, config)
}

/// Solve least squares from matrix/vector views into caller-provided `output`.
///
/// # Errors
/// Returns an error for invalid dimensions or rank-deficient systems.
#[cfg(feature = "lapack-provider")]
pub fn solve_least_squares_view_into<T, S>(
    matrix: &ArrayView2<'_, T>,
    rhs: &ArrayView1<'_, T>,
    config: &QRConfig<T>,
    output: &mut ArrayBase<S, Ix1>,
) -> Result<(), QRError>
where
    T: QrProviderScalar,
    S: DataMut<Elem = T>,
{
    if matrix.nrows() < matrix.ncols() {
        let result = solve_least_squares_impl(matrix, rhs, config)?;
        if output.len() != result.len() {
            return Err(QRError::InvalidDimensions(
                "output length must equal matrix columns".to_string(),
            ));
        }
        output.assign(&result);
        return Ok(());
    }

    let full = decompose_view(matrix, config)?;
    let keep = matrix.nrows().min(matrix.ncols());
    let qr = QRResult {
        q:    full.q.slice(s![.., ..keep]).to_owned(),
        r:    full.r.slice(s![..keep, ..]).to_owned(),
        p:    full.p,
        rank: full.rank.min(keep),
    };
    solve_least_squares_from_qr_result_view_into(&qr, rhs, config, output)
}

/// Solve least squares from matrix/vector views into caller-provided `output`.
///
/// # Errors
/// Returns an error for invalid dimensions or rank-deficient systems.
#[cfg(not(feature = "lapack-provider"))]
pub fn solve_least_squares_view_into<T, S>(
    matrix: &ArrayView2<'_, T>,
    rhs: &ArrayView1<'_, T>,
    config: &QRConfig<T>,
    output: &mut ArrayBase<S, Ix1>,
) -> Result<(), QRError>
where
    T: QrInternalScalar,
    S: DataMut<Elem = T>,
{
    if matrix.nrows() < matrix.ncols() {
        let result = solve_least_squares_impl(matrix, rhs, config)?;
        if output.len() != result.len() {
            return Err(QRError::InvalidDimensions(
                "output length must equal matrix columns".to_string(),
            ));
        }
        output.assign(&result);
        return Ok(());
    }

    let full = decompose_view(matrix, config)?;
    let keep = matrix.nrows().min(matrix.ncols());
    let qr = QRResult {
        q:    full.q.slice(s![.., ..keep]).to_owned(),
        r:    full.r.slice(s![..keep, ..]).to_owned(),
        p:    full.p,
        rank: full.rank.min(keep),
    };
    solve_least_squares_from_qr_result_view_into(&qr, rhs, config, output)
}

/// Reconstruct matrix `Q * R`.
#[must_use]
pub fn reconstruct_matrix<T: NabledReal>(qr: &QRResult<T>) -> Array2<T> { qr.q.dot(&qr.r) }

/// Reconstruct complex matrix `Q * R`.
#[must_use]
pub fn reconstruct_matrix_complex(qr: &QRResult<Complex64>) -> Array2<Complex64> { qr.q.dot(&qr.r) }

fn permutation_order<T: NabledReal>(permutation: &Array2<T>) -> Result<Vec<usize>, QRError> {
    if permutation.nrows() != permutation.ncols() {
        return Err(QRError::InvalidDimensions("permutation matrix must be square".to_string()));
    }

    let tolerance = T::from_f64(DenseKernelPolicy::BASE_TOLERANCE).unwrap_or(T::epsilon());
    let mut order = vec![usize::MAX; permutation.ncols()];
    for col in 0..permutation.ncols() {
        for row in 0..permutation.nrows() {
            if permutation[[row, col]].abs() > tolerance {
                order[col] = row;
                break;
            }
        }
        if order[col] == usize::MAX {
            return Err(QRError::InvalidInput(
                "permutation matrix must contain one non-zero entry per column".to_string(),
            ));
        }
    }
    Ok(order)
}

fn complex_permutation_order(permutation: &Array2<Complex64>) -> Result<Vec<usize>, QRError> {
    if permutation.nrows() != permutation.ncols() {
        return Err(QRError::InvalidDimensions("permutation matrix must be square".to_string()));
    }

    let mut order = vec![usize::MAX; permutation.ncols()];
    for col in 0..permutation.ncols() {
        for row in 0..permutation.nrows() {
            if permutation[[row, col]].norm() > DenseKernelPolicy::BASE_TOLERANCE {
                order[col] = row;
                break;
            }
        }
        if order[col] == usize::MAX {
            return Err(QRError::InvalidInput(
                "permutation matrix must contain one non-zero entry per column".to_string(),
            ));
        }
    }
    Ok(order)
}

/// Reconstruct matrix `Q * R` into `output`.
///
/// # Errors
/// Returns an error if output dimensions do not match `Q * R`.
pub fn reconstruct_matrix_into<T: NabledReal>(
    qr: &QRResult<T>,
    output: &mut Array2<T>,
) -> Result<(), QRError> {
    if qr.q.ncols() != qr.r.nrows() {
        return Err(QRError::InvalidDimensions("q.ncols() must equal r.nrows()".to_string()));
    }
    if output.dim() != (qr.q.nrows(), qr.r.ncols()) {
        return Err(QRError::InvalidDimensions(
            "output shape must match q.rows x r.cols".to_string(),
        ));
    }

    output.fill(T::zero());
    for i in 0..qr.q.nrows() {
        for j in 0..qr.r.ncols() {
            let mut sum = T::zero();
            for p in 0..qr.q.ncols() {
                sum += qr.q[[i, p]] * qr.r[[p, j]];
            }
            output[[i, j]] = sum;
        }
    }

    Ok(())
}

/// Reconstruct complex matrix `Q * R` into `output`.
///
/// # Errors
/// Returns an error if output dimensions do not match `Q * R`.
pub fn reconstruct_matrix_complex_into(
    qr: &QRResult<Complex64>,
    output: &mut Array2<Complex64>,
) -> Result<(), QRError> {
    if qr.q.ncols() != qr.r.nrows() {
        return Err(QRError::InvalidDimensions("q.ncols() must equal r.nrows()".to_string()));
    }
    if output.dim() != (qr.q.nrows(), qr.r.ncols()) {
        return Err(QRError::InvalidDimensions(
            "output shape must match q.rows x r.cols".to_string(),
        ));
    }

    output.fill(Complex64::new(0.0, 0.0));
    for i in 0..qr.q.nrows() {
        for j in 0..qr.r.ncols() {
            let mut sum = Complex64::new(0.0, 0.0);
            for p in 0..qr.q.ncols() {
                sum += qr.q[[i, p]] * qr.r[[p, j]];
            }
            output[[i, j]] = sum;
        }
    }

    Ok(())
}

/// Reconstruct the original matrix from a pivoted QR result into `output`.
///
/// # Errors
/// Returns an error if the QR result is missing a permutation or dimensions are incompatible.
pub fn reconstruct_original_matrix_into<T: NabledReal>(
    qr: &QRResult<T>,
    output: &mut Array2<T>,
) -> Result<(), QRError> {
    let permutation = qr.p.as_ref().ok_or_else(|| {
        QRError::InvalidInput("pivoted QR result missing permutation".to_string())
    })?;
    if qr.q.ncols() != qr.r.nrows() {
        return Err(QRError::InvalidDimensions("q.ncols() must equal r.nrows()".to_string()));
    }
    if permutation.nrows() != qr.r.ncols() || permutation.ncols() != qr.r.ncols() {
        return Err(QRError::InvalidDimensions(
            "permutation shape must match r column dimensions".to_string(),
        ));
    }
    if output.dim() != (qr.q.nrows(), qr.r.ncols()) {
        return Err(QRError::InvalidDimensions(
            "output shape must match q.rows x r.cols".to_string(),
        ));
    }

    let order = permutation_order(permutation)?;
    output.fill(T::zero());
    for (pivoted_col, &output_col) in order.iter().enumerate().take(qr.r.ncols()) {
        for row in 0..qr.q.nrows() {
            let mut sum = T::zero();
            for inner in 0..qr.q.ncols() {
                sum += qr.q[[row, inner]] * qr.r[[inner, pivoted_col]];
            }
            output[[row, output_col]] = sum;
        }
    }
    Ok(())
}

/// Reconstruct the original matrix from a complex pivoted QR result into `output`.
///
/// # Errors
/// Returns an error if the QR result is missing a permutation or dimensions are incompatible.
pub fn reconstruct_original_matrix_complex_into(
    qr: &QRResult<Complex64>,
    output: &mut Array2<Complex64>,
) -> Result<(), QRError> {
    let permutation = qr.p.as_ref().ok_or_else(|| {
        QRError::InvalidInput("pivoted QR result missing permutation".to_string())
    })?;
    if qr.q.ncols() != qr.r.nrows() {
        return Err(QRError::InvalidDimensions("q.ncols() must equal r.nrows()".to_string()));
    }
    if permutation.nrows() != qr.r.ncols() || permutation.ncols() != qr.r.ncols() {
        return Err(QRError::InvalidDimensions(
            "permutation shape must match r column dimensions".to_string(),
        ));
    }
    if output.dim() != (qr.q.nrows(), qr.r.ncols()) {
        return Err(QRError::InvalidDimensions(
            "output shape must match q.rows x r.cols".to_string(),
        ));
    }

    let order = complex_permutation_order(permutation)?;
    output.fill(Complex64::new(0.0, 0.0));
    for (pivoted_col, &output_col) in order.iter().enumerate().take(qr.r.ncols()) {
        for row in 0..qr.q.nrows() {
            let mut sum = Complex64::new(0.0, 0.0);
            for inner in 0..qr.q.ncols() {
                sum += qr.q[[row, inner]] * qr.r[[inner, pivoted_col]];
            }
            output[[row, output_col]] = sum;
        }
    }
    Ok(())
}

/// Estimate condition number from the `R` diagonal.
#[must_use]
pub fn condition_number<T: NabledReal>(qr: &QRResult<T>) -> T {
    if qr.r.is_empty() {
        return T::zero();
    }

    let n = qr.r.nrows().min(qr.r.ncols());
    let mut max_diagonal = T::zero();
    let mut min_diagonal = T::infinity();
    let tolerance = T::from_f64(DenseKernelPolicy::BASE_TOLERANCE).unwrap_or(T::epsilon());
    for i in 0..n {
        let value = qr.r[[i, i]].abs();
        max_diagonal = max_diagonal.max(value);
        if value > tolerance {
            min_diagonal = min_diagonal.min(value);
        }
    }

    if min_diagonal.is_finite() { max_diagonal / min_diagonal } else { T::infinity() }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2};
    use num_complex::Complex64;

    use super::*;

    #[test]
    fn qr_reconstructs_input() {
        let matrix = Array2::from_shape_vec((3, 2), vec![
            1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64, 5.0_f64, 6.0_f64,
        ])
        .unwrap();
        let qr = decompose(&matrix, &QRConfig::default()).unwrap();
        let reconstructed = reconstruct_matrix(&qr);
        for i in 0..3 {
            for j in 0..2 {
                assert!((matrix[[i, j]] - reconstructed[[i, j]]).abs() < 1e-8_f64);
            }
        }
    }

    #[test]
    fn least_squares_solves_overdetermined_system() {
        let matrix = Array2::from_shape_vec((4, 2), vec![
            1.0_f64, 1.0_f64, 1.0_f64, 2.0_f64, 1.0_f64, 3.0_f64, 1.0_f64, 4.0_f64,
        ])
        .unwrap();
        let rhs = Array1::from_vec(vec![2.0_f64, 3.0_f64, 4.0_f64, 5.0_f64]);
        let x = solve_least_squares(&matrix, &rhs, &QRConfig::default()).unwrap();
        assert!((x[0] - 1.0_f64).abs() < 1e-8_f64);
        assert!((x[1] - 1.0_f64).abs() < 1e-8_f64);
    }

    #[test]
    fn least_squares_rejects_bad_dimensions() {
        let matrix = Array2::eye(2);
        let rhs = Array1::from_vec(vec![1.0_f64, 2.0_f64, 3.0_f64]);
        let result = solve_least_squares(&matrix, &rhs, &QRConfig::default());
        assert!(matches!(result, Err(QRError::InvalidDimensions(_))));
    }

    #[test]
    fn least_squares_from_qr_result_matches_matrix_path() {
        let matrix = Array2::from_shape_vec((4, 2), vec![
            1.0_f64, 1.0_f64, 1.0_f64, 2.0_f64, 1.0_f64, 3.0_f64, 1.0_f64, 4.0_f64,
        ])
        .unwrap();
        let rhs = Array1::from_vec(vec![2.0_f64, 3.0_f64, 4.0_f64, 5.0_f64]);
        let config = QRConfig::default();

        let direct = solve_least_squares(&matrix, &rhs, &config).unwrap();
        let qr = decompose_reduced(&matrix, &config).unwrap();
        let from_factors =
            solve_least_squares_from_qr_result_view(&qr, &rhs.view(), &config).unwrap();
        let mut out = Array1::<f64>::zeros(2);
        solve_least_squares_from_qr_result_view_into(
            &qr,
            &rhs.view(),
            &config,
            &mut out.view_mut(),
        )
        .unwrap();

        for i in 0..direct.len() {
            assert!((direct[i] - from_factors[i]).abs() < 1.0e-10_f64);
            assert!((direct[i] - out[i]).abs() < 1.0e-10_f64);
        }

        let from_factor_views = solve_least_squares_from_qr_factors_view(
            &qr.q.view(),
            &qr.r.view(),
            qr.p.as_ref().map(|permutation| permutation.view()),
            &rhs.view(),
            &config,
        )
        .unwrap();
        let mut from_factor_views_into = Array1::<f64>::zeros(2);
        solve_least_squares_from_qr_factors_view_into(
            &qr.q.view(),
            &qr.r.view(),
            qr.p.as_ref().map(|permutation| permutation.view()),
            &rhs.view(),
            &config,
            &mut from_factor_views_into.view_mut(),
        )
        .unwrap();

        for i in 0..direct.len() {
            assert!((direct[i] - from_factor_views[i]).abs() < 1.0e-10_f64);
            assert!((direct[i] - from_factor_views_into[i]).abs() < 1.0e-10_f64);
        }
    }

    #[test]
    fn least_squares_from_wide_qr_result_is_rejected() {
        let matrix = Array2::from_shape_vec((2, 3), vec![
            1.0_f64, 0.0_f64, 2.0_f64, 0.0_f64, 1.0_f64, 3.0_f64,
        ])
        .unwrap();
        let rhs = Array1::from_vec(vec![1.0_f64, 2.0_f64]);
        let config = QRConfig::default();
        let qr = decompose_reduced(&matrix, &config).unwrap();

        let result = solve_least_squares_from_qr_result_view(&qr, &rhs.view(), &config);

        assert!(matches!(result, Err(QRError::InvalidInput(_))));
    }

    #[test]
    fn least_squares_from_factor_views_reject_inconsistent_inputs() {
        let q = Array2::<f64>::eye(2);
        let r = Array2::<f64>::eye(2);
        let rhs = Array1::from_vec(vec![1.0_f64, 2.0_f64]);
        let config = QRConfig::default();

        let bad_r_rows = Array2::<f64>::zeros((3, 2));
        let mut output = Array1::<f64>::zeros(2);
        assert!(matches!(
            solve_least_squares_from_qr_factors_view_into(
                &q.view(),
                &bad_r_rows.view(),
                None,
                &rhs.view(),
                &config,
                &mut output.view_mut(),
            ),
            Err(QRError::InvalidDimensions(_))
        ));

        let bad_rhs = Array1::from_vec(vec![1.0_f64]);
        assert!(matches!(
            solve_least_squares_from_qr_factors_view_into(
                &q.view(),
                &r.view(),
                None,
                &bad_rhs.view(),
                &config,
                &mut output.view_mut(),
            ),
            Err(QRError::InvalidDimensions(_))
        ));

        let mut short_output = Array1::<f64>::zeros(1);
        assert!(matches!(
            solve_least_squares_from_qr_factors_view_into(
                &q.view(),
                &r.view(),
                None,
                &rhs.view(),
                &config,
                &mut short_output.view_mut(),
            ),
            Err(QRError::InvalidDimensions(_))
        ));

        let singular_r = Array2::from_shape_vec((2, 2), vec![1.0_f64, 0.0, 0.0, 0.0]).unwrap();
        assert!(matches!(
            solve_least_squares_from_qr_factors_view_into(
                &q.view(),
                &singular_r.view(),
                None,
                &rhs.view(),
                &config,
                &mut output.view_mut(),
            ),
            Err(QRError::SingularMatrix)
        ));

        let bad_permutation = Array2::<f64>::eye(3);
        assert!(matches!(
            solve_least_squares_from_qr_factors_view_into(
                &q.view(),
                &r.view(),
                Some(bad_permutation.view()),
                &rhs.view(),
                &config,
                &mut output.view_mut(),
            ),
            Err(QRError::InvalidDimensions(_))
        ));
    }

    #[test]
    fn least_squares_from_factor_views_applies_permutation() {
        let q = Array2::<f64>::eye(2);
        let r = Array2::<f64>::eye(2);
        let permutation =
            Array2::from_shape_vec((2, 2), vec![0.0_f64, 1.0_f64, 1.0_f64, 0.0_f64]).unwrap();
        let rhs = Array1::from_vec(vec![1.0_f64, 2.0_f64]);
        let mut output = Array1::<f64>::zeros(2);

        solve_least_squares_from_qr_factors_view_into(
            &q.view(),
            &r.view(),
            Some(permutation.view()),
            &rhs.view(),
            &QRConfig::default(),
            &mut output.view_mut(),
        )
        .unwrap();

        assert!((output[0] - 2.0_f64).abs() < 1e-12_f64);
        assert!((output[1] - 1.0_f64).abs() < 1e-12_f64);
    }

    #[test]
    fn least_squares_view_into_handles_tall_and_wide_systems() {
        let tall_matrix = Array2::from_shape_vec((4, 2), vec![
            1.0_f64, 1.0_f64, 1.0_f64, 2.0_f64, 1.0_f64, 3.0_f64, 1.0_f64, 4.0_f64,
        ])
        .unwrap();
        let tall_rhs = Array1::from_vec(vec![2.0_f64, 3.0_f64, 4.0_f64, 5.0_f64]);
        let mut tall_output = Array1::<f64>::zeros(2);
        solve_least_squares_view_into(
            &tall_matrix.view(),
            &tall_rhs.view(),
            &QRConfig::default(),
            &mut tall_output.view_mut(),
        )
        .unwrap();
        assert!((tall_output[0] - 1.0_f64).abs() < 1e-8_f64);
        assert!((tall_output[1] - 1.0_f64).abs() < 1e-8_f64);

        let wide_matrix = Array2::from_shape_vec((2, 3), vec![
            1.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 1.0_f64, 0.0_f64,
        ])
        .unwrap();
        let wide_rhs = Array1::from_vec(vec![1.0_f64, 2.0_f64]);
        let mut wide_output = Array1::<f64>::zeros(3);
        solve_least_squares_view_into(
            &wide_matrix.view(),
            &wide_rhs.view(),
            &QRConfig::default(),
            &mut wide_output.view_mut(),
        )
        .unwrap();
        assert!((wide_output[0] - 1.0_f64).abs() < 1e-10_f64);
        assert!((wide_output[1] - 2.0_f64).abs() < 1e-10_f64);
        assert!(wide_output[2].abs() < 1e-10_f64);

        let mut short_output = Array1::<f64>::zeros(2);
        assert!(matches!(
            solve_least_squares_view_into(
                &wide_matrix.view(),
                &wide_rhs.view(),
                &QRConfig::default(),
                &mut short_output.view_mut(),
            ),
            Err(QRError::InvalidDimensions(_))
        ));
    }

    #[test]
    fn permutation_order_rejects_invalid_permutations() {
        let rectangular = Array2::<f64>::zeros((2, 3));
        assert!(matches!(permutation_order(&rectangular), Err(QRError::InvalidDimensions(_))));

        let empty_column = Array2::<f64>::zeros((2, 2));
        assert!(matches!(permutation_order(&empty_column), Err(QRError::InvalidInput(_))));

        let rectangular_complex = Array2::<Complex64>::zeros((2, 3));
        assert!(matches!(
            complex_permutation_order(&rectangular_complex),
            Err(QRError::InvalidDimensions(_))
        ));

        let empty_complex_column = Array2::<Complex64>::zeros((2, 2));
        assert!(matches!(
            complex_permutation_order(&empty_complex_column),
            Err(QRError::InvalidInput(_))
        ));
    }

    #[test]
    fn complex_qr_reconstructs_input() {
        let matrix = Array2::from_shape_vec((2, 2), vec![
            Complex64::new(1.0_f64, 1.0_f64),
            Complex64::new(2.0_f64, -1.0_f64),
            Complex64::new(0.5_f64, 0.25_f64),
            Complex64::new(-1.0_f64, 2.0_f64),
        ])
        .unwrap();
        let qr = decompose_complex(&matrix, &QRConfig::default()).unwrap();
        let reconstructed = reconstruct_matrix_complex(&qr);
        for i in 0..2 {
            for j in 0..2 {
                assert!((reconstructed[[i, j]] - matrix[[i, j]]).norm() < 1e-8_f64);
            }
        }
    }

    #[test]
    fn complex_pivoted_qr_reconstructs_permuted_input() {
        let matrix = Array2::from_shape_vec((3, 3), vec![
            Complex64::new(1.0_f64, 0.0_f64),
            Complex64::new(10.0_f64, -1.0_f64),
            Complex64::new(0.0_f64, 0.0_f64),
            Complex64::new(0.0_f64, 0.0_f64),
            Complex64::new(11.0_f64, 2.0_f64),
            Complex64::new(1.0_f64, 0.0_f64),
            Complex64::new(0.0_f64, 0.0_f64),
            Complex64::new(12.0_f64, 0.5_f64),
            Complex64::new(0.0_f64, 0.0_f64),
        ])
        .unwrap();
        let config = QRConfig { use_pivoting: true, ..QRConfig::default() };
        let qr = decompose_complex(&matrix, &config).unwrap();
        let permutation = qr.p.as_ref().unwrap();
        let reconstructed = reconstruct_matrix_complex(&qr);
        let permuted_input = matrix.dot(permutation);
        for i in 0..matrix.nrows() {
            for j in 0..matrix.ncols() {
                assert!((reconstructed[[i, j]] - permuted_input[[i, j]]).norm() < 1e-8_f64);
            }
        }
    }

    #[test]
    fn decompose_view_matches_owned() {
        let matrix = Array2::from_shape_vec((3, 2), vec![
            1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64, 5.0_f64, 6.0_f64,
        ])
        .unwrap();
        let from_owned = decompose(&matrix, &QRConfig::default()).unwrap();
        let matrix_view = matrix.view();
        let from_view = decompose_view(&matrix_view, &QRConfig::default()).unwrap();
        assert_eq!(from_owned.rank, from_view.rank);
    }

    #[test]
    fn reduced_and_pivoted_qr_shapes_are_consistent() {
        let matrix = Array2::from_shape_vec((4, 2), vec![
            1.0_f64, 0.0_f64, 2.0_f64, 1.0_f64, 3.0_f64, 1.0_f64, 4.0_f64, 2.0_f64,
        ])
        .unwrap();
        let reduced = decompose_reduced(&matrix, &QRConfig::default()).unwrap();
        assert_eq!(reduced.q.dim(), (4, 2));
        assert_eq!(reduced.r.dim(), (2, 2));

        let pivoted = decompose_with_pivoting(&matrix, &QRConfig::default()).unwrap();
        assert!(pivoted.p.is_some());
    }

    #[test]
    fn pivoted_qr_reconstructs_permuted_input() {
        let matrix = Array2::from_shape_vec((3, 3), vec![
            1.0_f64, 10.0_f64, 0.0_f64, //
            0.0_f64, 11.0_f64, 1.0_f64, //
            0.0_f64, 12.0_f64, 0.0_f64, //
        ])
        .unwrap();
        let qr = decompose_with_pivoting(&matrix, &QRConfig::default()).unwrap();
        let permutation = qr.p.as_ref().unwrap();
        let reconstructed = reconstruct_matrix(&qr);
        let permuted_input = matrix.dot(permutation);
        for i in 0..matrix.nrows() {
            for j in 0..matrix.ncols() {
                assert!((reconstructed[[i, j]] - permuted_input[[i, j]]).abs() < 1e-8_f64);
            }
        }
    }

    #[test]
    fn reconstruct_into_and_condition_number_work() {
        let matrix =
            Array2::from_shape_vec((2, 2), vec![3.0_f64, 1.0_f64, 0.0_f64, 2.0_f64]).unwrap();
        let qr = decompose(&matrix, &QRConfig::default()).unwrap();
        let mut out = Array2::<f64>::zeros((2, 2));
        reconstruct_matrix_into(&qr, &mut out).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!((out[[i, j]] - matrix[[i, j]]).abs() < 1e-8_f64);
            }
        }
        assert!(condition_number(&qr).is_finite());
    }

    #[test]
    fn pivoted_reconstruct_original_into_restores_input() {
        let matrix = Array2::from_shape_vec((3, 3), vec![
            1.0_f64, 10.0_f64, 0.0_f64, //
            0.0_f64, 11.0_f64, 1.0_f64, //
            0.0_f64, 12.0_f64, 0.0_f64, //
        ])
        .unwrap();
        let qr = decompose_with_pivoting(&matrix, &QRConfig::default()).unwrap();
        let mut out = Array2::<f64>::zeros(matrix.dim());

        reconstruct_original_matrix_into(&qr, &mut out).unwrap();

        for i in 0..matrix.nrows() {
            for j in 0..matrix.ncols() {
                assert!((out[[i, j]] - matrix[[i, j]]).abs() < 1e-8_f64);
            }
        }
    }

    #[test]
    fn complex_reconstruct_into_variants_work() {
        let matrix = Array2::from_shape_vec((3, 2), vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(3.0, 0.5),
            Complex64::new(4.0, 0.25),
            Complex64::new(5.0, -0.5),
            Complex64::new(6.0, 1.0),
        ])
        .unwrap();
        let qr = decompose_complex_view(&matrix.view(), &QRConfig::default()).unwrap();
        let mut out = Array2::<Complex64>::zeros(matrix.dim());

        reconstruct_matrix_complex_into(&qr, &mut out).unwrap();

        for i in 0..matrix.nrows() {
            for j in 0..matrix.ncols() {
                assert!((out[[i, j]] - matrix[[i, j]]).norm() < 1e-8_f64);
            }
        }
    }

    #[test]
    fn least_squares_view_matches_owned() {
        let matrix = Array2::from_shape_vec((4, 2), vec![
            1.0_f64, 1.0_f64, 1.0_f64, 2.0_f64, 1.0_f64, 3.0_f64, 1.0_f64, 4.0_f64,
        ])
        .unwrap();
        let rhs = Array1::from_vec(vec![2.0_f64, 3.0_f64, 4.0_f64, 5.0_f64]);
        let owned = solve_least_squares(&matrix, &rhs, &QRConfig::default()).unwrap();
        let matrix_view = matrix.view();
        let rhs_view = rhs.view();
        let viewed =
            solve_least_squares_view(&matrix_view, &rhs_view, &QRConfig::default()).unwrap();
        assert!((owned[0] - viewed[0]).abs() < 1e-8_f64);
        assert!((owned[1] - viewed[1]).abs() < 1e-8_f64);
    }

    #[test]
    fn least_squares_solves_underdetermined_minimum_norm_system() {
        let matrix = Array2::from_shape_vec((2, 3), vec![
            1.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 1.0_f64, 0.0_f64,
        ])
        .unwrap();
        let rhs = Array1::from_vec(vec![1.0_f64, 2.0_f64]);
        let solution = solve_least_squares(&matrix, &rhs, &QRConfig::default()).unwrap();
        assert!((solution[0] - 1.0_f64).abs() < 1e-10_f64);
        assert!((solution[1] - 2.0_f64).abs() < 1e-10_f64);
        assert!(solution[2].abs() < 1e-10_f64);
    }

    #[test]
    fn least_squares_rejects_rank_deficient_underdetermined_input() {
        let matrix = Array2::from_shape_vec((2, 3), vec![
            1.0_f64, 0.0_f64, 0.0_f64, 2.0_f64, 0.0_f64, 0.0_f64,
        ])
        .unwrap();
        let rhs = Array1::from_vec(vec![1.0_f64, 2.0_f64]);
        let result = solve_least_squares(&matrix, &rhs, &QRConfig::default());
        assert!(matches!(result, Err(QRError::SingularMatrix)));
    }

    #[test]
    fn least_squares_with_pivoting_matches_unpivoted_solution() {
        let matrix = Array2::from_shape_vec((4, 2), vec![
            1.0_f64, 3.0_f64, 1.0_f64, 4.0_f64, 1.0_f64, 5.0_f64, 1.0_f64, 6.0_f64,
        ])
        .unwrap();
        let rhs = Array1::from_vec(vec![4.0_f64, 5.0_f64, 6.0_f64, 7.0_f64]);

        let unpivoted = solve_least_squares(&matrix, &rhs, &QRConfig::default()).unwrap();
        let pivoted = solve_least_squares(&matrix, &rhs, &QRConfig {
            use_pivoting: true,
            ..QRConfig::default()
        })
        .unwrap();
        for i in 0..unpivoted.len() {
            assert!((unpivoted[i] - pivoted[i]).abs() < 1e-8_f64);
        }
    }

    #[test]
    fn reconstruct_into_rejects_invalid_shapes() {
        let matrix =
            Array2::from_shape_vec((2, 2), vec![1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64]).unwrap();
        let qr = decompose(&matrix, &QRConfig::default()).unwrap();

        let mut bad_out = Array2::<f64>::zeros((1, 1));
        assert!(matches!(
            reconstruct_matrix_into(&qr, &mut bad_out),
            Err(QRError::InvalidDimensions(_))
        ));

        let bad_qr = QRResult {
            q:    Array2::<f64>::zeros((2, 3)),
            r:    Array2::<f64>::zeros((2, 2)),
            p:    None,
            rank: 0,
        };
        let mut out = Array2::<f64>::zeros((2, 2));
        assert!(matches!(
            reconstruct_matrix_into(&bad_qr, &mut out),
            Err(QRError::InvalidDimensions(_))
        ));
    }

    #[test]
    fn condition_number_of_empty_factor_is_zero() {
        let qr = QRResult {
            q:    Array2::<f64>::zeros((0, 0)),
            r:    Array2::<f64>::zeros((0, 0)),
            p:    None,
            rank: 0,
        };
        assert!(condition_number(&qr).abs() < 1e-12_f64);
    }
}
