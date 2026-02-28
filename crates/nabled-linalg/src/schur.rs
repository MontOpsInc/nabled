//! Schur decomposition over ndarray matrices.

use std::fmt;

use ndarray::Array2;

use crate::internal::{DEFAULT_TOLERANCE, identity, validate_finite, validate_square_non_empty};
use crate::qr::{QRConfig, ndarray_qr};

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
pub struct NdarraySchurResult {
    /// Orthogonal matrix Q.
    pub q: Array2<f64>,
    /// Upper triangular matrix T.
    pub t: Array2<f64>,
}

fn off_diagonal_norm(matrix: &Array2<f64>) -> f64 {
    let n = matrix.nrows();
    let mut sum = 0.0_f64;
    for i in 0..n {
        for j in 0..i {
            let value = matrix[[i, j]];
            sum += value * value;
        }
    }
    sum.sqrt()
}

/// Ndarray Schur functions.
pub mod ndarray_schur {
    use super::*;

    /// Compute Schur decomposition `A = Q T Q^T`.
    ///
    /// # Errors
    /// Returns an error for invalid input or convergence failure.
    pub fn compute_schur(matrix: &Array2<f64>) -> Result<NdarraySchurResult, SchurError> {
        validate_square_non_empty(matrix).map_err(|error| match error {
            "empty" => SchurError::EmptyMatrix,
            "not_square" => SchurError::NotSquare,
            _ => SchurError::InvalidInput(error.to_string()),
        })?;
        validate_finite(matrix).map_err(|_| SchurError::NumericalInstability)?;

        let n = matrix.nrows();
        let mut q_total = identity(n);
        let mut t = matrix.clone();
        let config = QRConfig::default();

        let mut converged = false;
        for _ in 0..config.max_iterations.max(128) {
            let qr =
                ndarray_qr::compute_qr(&t, &config).map_err(|_| SchurError::ConvergenceFailed)?;
            t = qr.r.dot(&qr.q);
            q_total = q_total.dot(&qr.q);
            if off_diagonal_norm(&t) < config.rank_tolerance.max(DEFAULT_TOLERANCE) {
                converged = true;
                break;
            }
        }

        if !converged {
            return Err(SchurError::ConvergenceFailed);
        }

        Ok(NdarraySchurResult { q: q_total, t })
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use super::ndarray_schur;

    #[test]
    fn schur_reconstructs_matrix() {
        let matrix = Array2::from_shape_vec((2, 2), vec![4.0, 1.0, 2.0, 3.0]).unwrap();
        let schur = ndarray_schur::compute_schur(&matrix).unwrap();
        let reconstructed = schur.q.dot(&schur.t).dot(&schur.q.t());
        for i in 0..2 {
            for j in 0..2 {
                assert!((matrix[[i, j]] - reconstructed[[i, j]]).abs() < 1e-6);
            }
        }
    }
}
