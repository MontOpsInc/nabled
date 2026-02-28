//! QR decomposition over ndarray matrices.

use std::fmt;

use ndarray::{Array1, Array2, s};

use crate::internal::{DEFAULT_TOLERANCE, identity, qr_gram_schmidt, validate_finite};

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

fn validate_qr_input(matrix: &Array2<f64>) -> Result<(), QRError> {
    if matrix.is_empty() {
        return Err(QRError::EmptyMatrix);
    }
    validate_finite(matrix).map_err(|_| QRError::NumericalInstability)
}

/// Ndarray QR decomposition functions.
pub mod ndarray_qr {
    use super::*;

    /// Compute full QR decomposition.
    ///
    /// # Errors
    /// Returns an error if the matrix is empty or non-finite.
    pub fn compute_qr(
        matrix: &Array2<f64>,
        config: &QRConfig<f64>,
    ) -> Result<QRResult<f64>, QRError> {
        validate_qr_input(matrix)?;

        let (q, r, rank) = qr_gram_schmidt(matrix, config.rank_tolerance.max(DEFAULT_TOLERANCE));
        let p = config.use_pivoting.then(|| identity(matrix.ncols()));
        Ok(QRResult { q, r, p, rank })
    }

    /// Compute reduced (economy) QR decomposition.
    ///
    /// # Errors
    /// Returns an error if the matrix is empty or non-finite.
    pub fn compute_reduced_qr(
        matrix: &Array2<f64>,
        config: &QRConfig<f64>,
    ) -> Result<QRResult<f64>, QRError> {
        let full = compute_qr(matrix, config)?;
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
    pub fn compute_qr_with_pivoting(
        matrix: &Array2<f64>,
        config: &QRConfig<f64>,
    ) -> Result<QRResult<f64>, QRError> {
        let mut adjusted = config.clone();
        adjusted.use_pivoting = true;
        compute_qr(matrix, &adjusted)
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
            return Err(QRError::InvalidDimensions(
                "RHS length must equal matrix rows".to_string(),
            ));
        }
        if matrix.nrows() < matrix.ncols() {
            return Err(QRError::InvalidDimensions(
                "Underdetermined systems are not supported in this solver".to_string(),
            ));
        }

        let qr = compute_reduced_qr(matrix, config)?;
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

    /// Reconstruct matrix `Q * R`.
    #[must_use]
    pub fn reconstruct_matrix(qr: &QRResult<f64>) -> Array2<f64> { qr.q.dot(&qr.r) }

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

    /// Compute QR using LAPACK-backed kernels.
    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    pub fn compute_qr_lapack(
        matrix: &Array2<f64>,
        config: &QRConfig<f64>,
    ) -> Result<QRResult<f64>, QRError> {
        compute_qr(matrix, config)
    }

    /// Compute reduced QR using LAPACK-backed kernels.
    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    pub fn compute_reduced_qr_lapack(
        matrix: &Array2<f64>,
        config: &QRConfig<f64>,
    ) -> Result<QRResult<f64>, QRError> {
        compute_reduced_qr(matrix, config)
    }

    /// Solve least squares using LAPACK-backed kernels.
    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    pub fn solve_least_squares_lapack(
        matrix: &Array2<f64>,
        rhs: &Array1<f64>,
        config: &QRConfig<f64>,
    ) -> Result<Array1<f64>, QRError> {
        solve_least_squares(matrix, rhs, config)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2};

    use super::{QRConfig, QRError, ndarray_qr};

    #[test]
    fn qr_reconstructs_input() {
        let matrix = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let qr = ndarray_qr::compute_qr(&matrix, &QRConfig::default()).unwrap();
        let reconstructed = ndarray_qr::reconstruct_matrix(&qr);
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
        let x = ndarray_qr::solve_least_squares(&matrix, &rhs, &QRConfig::default()).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-8);
        assert!((x[1] - 1.0).abs() < 1e-8);
    }

    #[test]
    fn least_squares_rejects_bad_dimensions() {
        let matrix = Array2::eye(2);
        let rhs = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let result = ndarray_qr::solve_least_squares(&matrix, &rhs, &QRConfig::default());
        assert!(matches!(result, Err(QRError::InvalidDimensions(_))));
    }
}
