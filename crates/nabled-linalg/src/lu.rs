//! LU decomposition over ndarray matrices.

use std::fmt;

use ndarray::{Array1, Array2};

use crate::internal::{
    DEFAULT_TOLERANCE, inverse_from_lu, lu_decompose, lu_solve, validate_finite,
    validate_square_non_empty,
};

/// Result of LU decomposition.
#[derive(Debug, Clone)]
pub struct NdarrayLUResult {
    /// Lower-triangular factor.
    pub l: Array2<f64>,
    /// Upper-triangular factor.
    pub u: Array2<f64>,
}

/// Sign and log-absolute value of determinant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogDetResult<T> {
    /// Determinant sign.
    pub sign:       i8,
    /// Natural logarithm of absolute determinant.
    pub ln_abs_det: T,
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
        "non_finite" => LUError::NumericalInstability,
        _ => LUError::InvalidInput(error.to_string()),
    }
}

/// Ndarray LU functions.
pub mod ndarray_lu {
    use super::*;

    /// Compute LU decomposition with partial pivoting.
    ///
    /// # Errors
    /// Returns an error if input is invalid or decomposition fails.
    pub fn compute_lu(matrix: &Array2<f64>) -> Result<NdarrayLUResult, LUError> {
        let (l, u, _, _) = lu_decompose(matrix).map_err(map_lu_error)?;
        Ok(NdarrayLUResult { l, u })
    }

    /// Solve `Ax=b` using LU decomposition.
    ///
    /// # Errors
    /// Returns an error if dimensions are incompatible or matrix is singular.
    pub fn solve(matrix: &Array2<f64>, rhs: &Array1<f64>) -> Result<Array1<f64>, LUError> {
        validate_square_non_empty(matrix).map_err(map_lu_error)?;
        validate_finite(matrix).map_err(map_lu_error)?;
        if rhs.len() != matrix.nrows() {
            return Err(LUError::InvalidInput(
                "RHS length must match matrix dimensions".to_string(),
            ));
        }

        let (l, u, pivots, _) = lu_decompose(matrix).map_err(map_lu_error)?;
        lu_solve(&l, &u, &pivots, rhs).map_err(map_lu_error)
    }

    /// Compute matrix inverse via LU decomposition.
    ///
    /// # Errors
    /// Returns an error if matrix is singular.
    pub fn inverse(matrix: &Array2<f64>) -> Result<Array2<f64>, LUError> {
        let (l, u, pivots, _) = lu_decompose(matrix).map_err(map_lu_error)?;
        inverse_from_lu(&l, &u, &pivots).map_err(map_lu_error)
    }

    /// Compute determinant via LU decomposition.
    ///
    /// # Errors
    /// Returns an error if decomposition fails.
    pub fn determinant(matrix: &Array2<f64>) -> Result<f64, LUError> {
        let (_, u, _, sign) = lu_decompose(matrix).map_err(map_lu_error)?;
        let mut determinant = f64::from(sign);
        for i in 0..u.nrows() {
            determinant *= u[[i, i]];
        }
        if !determinant.is_finite() {
            return Err(LUError::NumericalInstability);
        }
        Ok(determinant)
    }

    /// Compute signed log-determinant via LU decomposition.
    ///
    /// # Errors
    /// Returns an error if matrix is singular.
    pub fn log_determinant(matrix: &Array2<f64>) -> Result<LogDetResult<f64>, LUError> {
        let determinant = determinant(matrix)?;
        if determinant.abs() <= DEFAULT_TOLERANCE {
            return Err(LUError::SingularMatrix);
        }
        let sign = if determinant.is_sign_positive() { 1 } else { -1 };
        Ok(LogDetResult { sign, ln_abs_det: determinant.abs().ln() })
    }

    /// Compute LU decomposition using LAPACK-backed kernels.
    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    pub fn compute_lu_lapack(matrix: &Array2<f64>) -> Result<NdarrayLUResult, LUError> {
        compute_lu(matrix)
    }

    /// Solve `Ax=b` using LAPACK-backed kernels.
    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    pub fn solve_lapack(matrix: &Array2<f64>, rhs: &Array1<f64>) -> Result<Array1<f64>, LUError> {
        solve(matrix, rhs)
    }

    /// Compute inverse using LAPACK-backed kernels.
    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    pub fn inverse_lapack(matrix: &Array2<f64>) -> Result<Array2<f64>, LUError> { inverse(matrix) }

    /// Compute determinant using LAPACK-backed kernels.
    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    pub fn determinant_lapack(matrix: &Array2<f64>) -> Result<f64, LUError> { determinant(matrix) }

    /// Compute log-determinant using LAPACK-backed kernels.
    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    pub fn log_determinant_lapack(matrix: &Array2<f64>) -> Result<LogDetResult<f64>, LUError> {
        log_determinant(matrix)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2};

    use super::{LUError, ndarray_lu};

    #[test]
    fn solve_reconstructs_rhs() {
        let matrix = Array2::from_shape_vec((2, 2), vec![4.0, 3.0, 6.0, 3.0]).unwrap();
        let rhs = Array1::from_vec(vec![10.0, 12.0]);
        let solution = ndarray_lu::solve(&matrix, &rhs).unwrap();
        let reconstructed = matrix.dot(&solution);
        assert!((reconstructed[0] - rhs[0]).abs() < 1e-8);
        assert!((reconstructed[1] - rhs[1]).abs() < 1e-8);
    }

    #[test]
    fn determinant_matches_expected() {
        let matrix = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let determinant = ndarray_lu::determinant(&matrix).unwrap();
        assert!((determinant + 2.0).abs() < 1e-12);
    }

    #[test]
    fn singular_matrix_is_rejected() {
        let singular = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 2.0, 4.0]).unwrap();
        let rhs = Array1::from_vec(vec![1.0, 2.0]);
        assert!(matches!(ndarray_lu::solve(&singular, &rhs), Err(LUError::SingularMatrix)));
    }
}
