//! QR decomposition over ndarray matrices.

use std::fmt;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};
use num_complex::Complex64;

#[cfg(not(feature = "openblas-system"))]
use crate::internal::qr_gram_schmidt;
use crate::internal::{DEFAULT_TOLERANCE, identity, validate_finite};

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

impl Default for QRConfig<f64> {
    fn default() -> Self {
        Self { rank_tolerance: 1e-12, max_iterations: 100, use_pivoting: false }
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

fn complex_identity(n: usize) -> Array2<Complex64> {
    let mut identity = Array2::<Complex64>::zeros((n, n));
    for i in 0..n {
        identity[[i, i]] = Complex64::new(1.0, 0.0);
    }
    identity
}

fn validate_qr_input(matrix: &Array2<f64>) -> Result<(), QRError> {
    if matrix.is_empty() {
        return Err(QRError::EmptyMatrix);
    }
    validate_finite(matrix).map_err(|_| QRError::NumericalInstability)
}

fn validate_qr_complex_input(matrix: &Array2<Complex64>) -> Result<(), QRError> {
    if matrix.is_empty() {
        return Err(QRError::EmptyMatrix);
    }
    if matrix.iter().any(|value| !value.re.is_finite() || !value.im.is_finite()) {
        return Err(QRError::NumericalInstability);
    }
    Ok(())
}

#[cfg(not(feature = "openblas-system"))]
fn decompose_internal(
    matrix: &Array2<f64>,
    config: &QRConfig<f64>,
) -> Result<QRResult<f64>, QRError> {
    validate_qr_input(matrix)?;

    let (q, r, rank) = qr_gram_schmidt(matrix, config.rank_tolerance.max(DEFAULT_TOLERANCE));
    let p = config.use_pivoting.then(|| identity(matrix.ncols()));
    Ok(QRResult { q, r, p, rank })
}

#[cfg(feature = "openblas-system")]
fn decompose_provider(
    matrix: &Array2<f64>,
    config: &QRConfig<f64>,
) -> Result<QRResult<f64>, QRError> {
    use ndarray_linalg::QR as _;

    validate_qr_input(matrix)?;
    let (q, r) = matrix.qr().map_err(|_| QRError::ConvergenceFailed)?;
    let p = config.use_pivoting.then(|| identity(matrix.ncols()));

    let diagonal = r.nrows().min(r.ncols());
    let rank = (0..diagonal)
        .filter(|&index| r[[index, index]].abs() > config.rank_tolerance.max(DEFAULT_TOLERANCE))
        .count();

    Ok(QRResult { q, r, p, rank })
}

#[cfg(feature = "openblas-system")]
fn solve_least_squares_provider(
    matrix: &Array2<f64>,
    rhs: &Array1<f64>,
) -> Result<Array1<f64>, QRError> {
    use ndarray_linalg::LeastSquaresSvd as _;

    let result = matrix.least_squares(rhs).map_err(|_| QRError::ConvergenceFailed)?;
    let rank = usize::try_from(result.rank).map_err(|_| QRError::ConvergenceFailed)?;
    if rank < matrix.ncols() {
        return Err(QRError::SingularMatrix);
    }
    Ok(result.solution)
}

#[cfg(not(feature = "openblas-system"))]
fn decompose_complex_internal(
    matrix: &Array2<Complex64>,
    config: &QRConfig<f64>,
) -> Result<QRResult<Complex64>, QRError> {
    validate_qr_complex_input(matrix)?;

    let rows = matrix.nrows();
    let cols = matrix.ncols();
    let mut q = Array2::<Complex64>::zeros((rows, cols));
    let mut r = Array2::<Complex64>::zeros((cols, cols));
    let mut rank = 0_usize;
    let tolerance = config.rank_tolerance.max(DEFAULT_TOLERANCE);

    for j in 0..cols {
        let mut v = matrix.column(j).to_owned();
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

    let p = config.use_pivoting.then(|| complex_identity(matrix.ncols()));
    Ok(QRResult { q, r, p, rank })
}

#[cfg(feature = "openblas-system")]
fn decompose_complex_provider(
    matrix: &Array2<Complex64>,
    config: &QRConfig<f64>,
) -> Result<QRResult<Complex64>, QRError> {
    use ndarray_linalg::QR as _;

    validate_qr_complex_input(matrix)?;
    let (q, r) = matrix.qr().map_err(|_| QRError::ConvergenceFailed)?;
    let diagonal = r.nrows().min(r.ncols());
    let rank = (0..diagonal)
        .filter(|&index| r[[index, index]].norm() > config.rank_tolerance.max(DEFAULT_TOLERANCE))
        .count();
    let p = config.use_pivoting.then(|| complex_identity(matrix.ncols()));
    Ok(QRResult { q, r, p, rank })
}

/// Compute full QR decomposition.
///
/// # Errors
/// Returns an error if the matrix is empty or non-finite.
pub fn decompose(matrix: &Array2<f64>, config: &QRConfig<f64>) -> Result<QRResult<f64>, QRError> {
    #[cfg(feature = "openblas-system")]
    {
        decompose_provider(matrix, config)
    }
    #[cfg(not(feature = "openblas-system"))]
    {
        decompose_internal(matrix, config)
    }
}

/// Compute full QR decomposition from a matrix view.
///
/// # Performance
/// This convenience wrapper currently materializes an owned matrix via
/// `to_owned()` before dispatching to [`decompose`].
///
/// # Errors
/// Returns an error if decomposition fails.
pub fn decompose_view(
    matrix: &ArrayView2<'_, f64>,
    config: &QRConfig<f64>,
) -> Result<QRResult<f64>, QRError> {
    decompose(&matrix.to_owned(), config)
}

/// Compute full QR decomposition for complex matrices.
///
/// # Errors
/// Returns an error if decomposition fails.
pub fn decompose_complex(
    matrix: &Array2<Complex64>,
    config: &QRConfig<f64>,
) -> Result<QRResult<Complex64>, QRError> {
    #[cfg(feature = "openblas-system")]
    {
        decompose_complex_provider(matrix, config)
    }
    #[cfg(not(feature = "openblas-system"))]
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
    decompose_complex(&matrix.to_owned(), config)
}

/// Compute reduced (economy) QR decomposition.
///
/// # Errors
/// Returns an error if the matrix is empty or non-finite.
pub fn decompose_reduced(
    matrix: &Array2<f64>,
    config: &QRConfig<f64>,
) -> Result<QRResult<f64>, QRError> {
    let full = decompose(matrix, config)?;
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
/// This implementation currently reuses the non-pivoted decomposition while
/// returning an identity permutation matrix.
///
/// # Errors
/// Returns an error if decomposition fails.
pub fn decompose_with_pivoting(
    matrix: &Array2<f64>,
    config: &QRConfig<f64>,
) -> Result<QRResult<f64>, QRError> {
    let mut adjusted = config.clone();
    adjusted.use_pivoting = true;
    decompose(matrix, &adjusted)
}

/// Solve least squares `argmin ||Ax - b||_2`.
///
/// # Errors
/// Returns an error for invalid dimensions or rank-deficient systems.
pub fn solve_least_squares(
    matrix: &Array2<f64>,
    rhs: &Array1<f64>,
    config: &QRConfig<f64>,
) -> Result<Array1<f64>, QRError> {
    validate_qr_input(matrix)?;
    if rhs.len() != matrix.nrows() {
        return Err(QRError::InvalidDimensions("RHS length must equal matrix rows".to_string()));
    }
    if matrix.nrows() < matrix.ncols() {
        return Err(QRError::InvalidDimensions(
            "Underdetermined systems are not supported in this solver".to_string(),
        ));
    }

    #[cfg(feature = "openblas-system")]
    {
        let _ = config;
        solve_least_squares_provider(matrix, rhs)
    }
    #[cfg(not(feature = "openblas-system"))]
    {
        let qr = decompose_reduced(matrix, config)?;
        let n = matrix.ncols();
        if qr.rank < n {
            return Err(QRError::SingularMatrix);
        }

        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut dot = 0.0_f64;
            for row in 0..matrix.nrows() {
                dot += qr.q[[row, i]] * rhs[row];
            }
            y[i] = dot;
        }

        let mut x = Array1::<f64>::zeros(n);
        for i_rev in 0..n {
            let i = n - 1 - i_rev;
            let mut sum = y[i];
            for j in (i + 1)..n {
                sum -= qr.r[[i, j]] * x[j];
            }
            let diagonal = qr.r[[i, i]];
            if diagonal.abs() <= config.rank_tolerance.max(DEFAULT_TOLERANCE) {
                return Err(QRError::SingularMatrix);
            }
            x[i] = sum / diagonal;
        }

        Ok(x)
    }
}

/// Solve least squares from matrix/vector views.
///
/// # Errors
/// Returns an error for invalid dimensions or rank-deficient systems.
pub fn solve_least_squares_view(
    matrix: &ArrayView2<'_, f64>,
    rhs: &ArrayView1<'_, f64>,
    config: &QRConfig<f64>,
) -> Result<Array1<f64>, QRError> {
    solve_least_squares(&matrix.to_owned(), &rhs.to_owned(), config)
}

/// Reconstruct matrix `Q * R`.
#[must_use]
pub fn reconstruct_matrix(qr: &QRResult<f64>) -> Array2<f64> { qr.q.dot(&qr.r) }

/// Reconstruct complex matrix `Q * R`.
#[must_use]
pub fn reconstruct_matrix_complex(qr: &QRResult<Complex64>) -> Array2<Complex64> { qr.q.dot(&qr.r) }

/// Reconstruct matrix `Q * R` into `output`.
///
/// # Errors
/// Returns an error if output dimensions do not match `Q * R`.
pub fn reconstruct_matrix_into(
    qr: &QRResult<f64>,
    output: &mut Array2<f64>,
) -> Result<(), QRError> {
    if qr.q.ncols() != qr.r.nrows() {
        return Err(QRError::InvalidDimensions("q.ncols() must equal r.nrows()".to_string()));
    }
    if output.dim() != (qr.q.nrows(), qr.r.ncols()) {
        return Err(QRError::InvalidDimensions(
            "output shape must match q.rows x r.cols".to_string(),
        ));
    }

    output.fill(0.0);
    for i in 0..qr.q.nrows() {
        for j in 0..qr.r.ncols() {
            let mut sum = 0.0_f64;
            for p in 0..qr.q.ncols() {
                sum += qr.q[[i, p]] * qr.r[[p, j]];
            }
            output[[i, j]] = sum;
        }
    }

    Ok(())
}

/// Estimate condition number from the `R` diagonal.
#[must_use]
pub fn condition_number(qr: &QRResult<f64>) -> f64 {
    if qr.r.is_empty() {
        return 0.0;
    }

    let n = qr.r.nrows().min(qr.r.ncols());
    let mut max_diagonal = 0.0_f64;
    let mut min_diagonal = f64::INFINITY;
    for i in 0..n {
        let value = qr.r[[i, i]].abs();
        max_diagonal = max_diagonal.max(value);
        if value > DEFAULT_TOLERANCE {
            min_diagonal = min_diagonal.min(value);
        }
    }

    if min_diagonal.is_finite() { max_diagonal / min_diagonal } else { f64::INFINITY }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2};
    use num_complex::Complex64;

    use super::*;

    #[test]
    fn qr_reconstructs_input() {
        let matrix = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let qr = decompose(&matrix, &QRConfig::default()).unwrap();
        let reconstructed = reconstruct_matrix(&qr);
        for i in 0..3 {
            for j in 0..2 {
                assert!((matrix[[i, j]] - reconstructed[[i, j]]).abs() < 1e-8);
            }
        }
    }

    #[test]
    fn least_squares_solves_overdetermined_system() {
        let matrix =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0]).unwrap();
        let rhs = Array1::from_vec(vec![2.0, 3.0, 4.0, 5.0]);
        let x = solve_least_squares(&matrix, &rhs, &QRConfig::default()).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-8);
        assert!((x[1] - 1.0).abs() < 1e-8);
    }

    #[test]
    fn least_squares_rejects_bad_dimensions() {
        let matrix = Array2::eye(2);
        let rhs = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let result = solve_least_squares(&matrix, &rhs, &QRConfig::default());
        assert!(matches!(result, Err(QRError::InvalidDimensions(_))));
    }

    #[test]
    fn complex_qr_reconstructs_input() {
        let matrix = Array2::from_shape_vec((2, 2), vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(0.5, 0.25),
            Complex64::new(-1.0, 2.0),
        ])
        .unwrap();
        let qr = decompose_complex(&matrix, &QRConfig::default()).unwrap();
        let reconstructed = reconstruct_matrix_complex(&qr);
        for i in 0..2 {
            for j in 0..2 {
                assert!((reconstructed[[i, j]] - matrix[[i, j]]).norm() < 1e-8);
            }
        }
    }

    #[test]
    fn decompose_view_matches_owned() {
        let matrix = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let from_owned = decompose(&matrix, &QRConfig::default()).unwrap();
        let matrix_view = matrix.view();
        let from_view = decompose_view(&matrix_view, &QRConfig::default()).unwrap();
        assert_eq!(from_owned.rank, from_view.rank);
    }

    #[test]
    fn reduced_and_pivoted_qr_shapes_are_consistent() {
        let matrix =
            Array2::from_shape_vec((4, 2), vec![1.0, 0.0, 2.0, 1.0, 3.0, 1.0, 4.0, 2.0]).unwrap();
        let reduced = decompose_reduced(&matrix, &QRConfig::default()).unwrap();
        assert_eq!(reduced.q.dim(), (4, 2));
        assert_eq!(reduced.r.dim(), (2, 2));

        let pivoted = decompose_with_pivoting(&matrix, &QRConfig::default()).unwrap();
        assert!(pivoted.p.is_some());
    }

    #[test]
    fn reconstruct_into_and_condition_number_work() {
        let matrix = Array2::from_shape_vec((2, 2), vec![3.0, 1.0, 0.0, 2.0]).unwrap();
        let qr = decompose(&matrix, &QRConfig::default()).unwrap();
        let mut out = Array2::<f64>::zeros((2, 2));
        reconstruct_matrix_into(&qr, &mut out).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!((out[[i, j]] - matrix[[i, j]]).abs() < 1e-8);
            }
        }
        assert!(condition_number(&qr).is_finite());
    }

    #[test]
    fn least_squares_view_matches_owned() {
        let matrix =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0]).unwrap();
        let rhs = Array1::from_vec(vec![2.0, 3.0, 4.0, 5.0]);
        let owned = solve_least_squares(&matrix, &rhs, &QRConfig::default()).unwrap();
        let matrix_view = matrix.view();
        let rhs_view = rhs.view();
        let viewed =
            solve_least_squares_view(&matrix_view, &rhs_view, &QRConfig::default()).unwrap();
        assert!((owned[0] - viewed[0]).abs() < 1e-8);
        assert!((owned[1] - viewed[1]).abs() < 1e-8);
    }

    #[test]
    fn least_squares_rejects_underdetermined_input() {
        let matrix = Array2::from_shape_vec((2, 3), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();
        let rhs = Array1::from_vec(vec![1.0, 2.0]);
        let result = solve_least_squares(&matrix, &rhs, &QRConfig::default());
        assert!(matches!(result, Err(QRError::InvalidDimensions(_))));
    }

    #[test]
    fn reconstruct_into_rejects_invalid_shapes() {
        let matrix = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
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
        assert!(condition_number(&qr).abs() < 1e-12);
    }
}
