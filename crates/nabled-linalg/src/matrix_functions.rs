//! Matrix functions over ndarray matrices.

use std::fmt;

use ndarray::{Array1, Array2};

use crate::eigen::ndarray_eigen;
use crate::internal::{
    DEFAULT_TOLERANCE, identity, is_symmetric, usize_to_f64, validate_finite,
    validate_square_non_empty,
};
use crate::svd::ndarray_svd;

/// Error type for matrix functions.
#[derive(Debug, Clone, PartialEq)]
pub enum MatrixFunctionError {
    /// Matrix is empty.
    EmptyMatrix,
    /// Matrix must be square.
    NotSquare,
    /// Matrix is not symmetric when required.
    NotSymmetric,
    /// Matrix is not positive definite when required.
    NotPositiveDefinite,
    /// Algorithm failed to converge.
    ConvergenceFailed,
    /// Invalid input.
    InvalidInput(String),
}

impl fmt::Display for MatrixFunctionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MatrixFunctionError::EmptyMatrix => write!(f, "Matrix cannot be empty"),
            MatrixFunctionError::NotSquare => write!(f, "Matrix must be square"),
            MatrixFunctionError::NotSymmetric => write!(f, "Matrix must be symmetric"),
            MatrixFunctionError::NotPositiveDefinite => {
                write!(f, "Matrix must be positive definite")
            }
            MatrixFunctionError::ConvergenceFailed => write!(f, "Algorithm failed to converge"),
            MatrixFunctionError::InvalidInput(message) => write!(f, "Invalid input: {message}"),
        }
    }
}

impl std::error::Error for MatrixFunctionError {}

fn validate_square(matrix: &Array2<f64>) -> Result<(), MatrixFunctionError> {
    validate_square_non_empty(matrix).map_err(|error| match error {
        "empty" => MatrixFunctionError::EmptyMatrix,
        _ => MatrixFunctionError::NotSquare,
    })?;
    validate_finite(matrix)
        .map_err(|_| MatrixFunctionError::InvalidInput("matrix must be finite".into()))?;
    Ok(())
}

fn diagonal_from(values: &Array1<f64>) -> Array2<f64> {
    let n = values.len();
    let mut diagonal = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        diagonal[[i, i]] = values[i];
    }
    diagonal
}

fn taylor_matrix_exp(
    matrix: &Array2<f64>,
    max_terms: usize,
    tolerance: f64,
) -> Result<Array2<f64>, MatrixFunctionError> {
    validate_square(matrix)?;
    let n = matrix.nrows();
    let mut result = identity(n);
    let mut term = identity(n);

    for k in 1..=max_terms.max(1) {
        term = term.dot(matrix) / usize_to_f64(k);
        result = &result + &term;
        let delta = term.iter().map(|value| value.abs()).fold(0.0_f64, f64::max);
        if delta <= tolerance.max(DEFAULT_TOLERANCE) {
            return Ok(result);
        }
    }

    Ok(result)
}

/// Ndarray matrix functions.
pub mod ndarray_matrix_functions {
    use super::*;

    /// Compute matrix exponential via Taylor series.
    ///
    /// # Errors
    /// Returns an error for invalid input.
    pub fn matrix_exp(
        matrix: &Array2<f64>,
        max_terms: usize,
        tolerance: f64,
    ) -> Result<Array2<f64>, MatrixFunctionError> {
        taylor_matrix_exp(matrix, max_terms, tolerance)
    }

    /// Compute matrix exponential via eigen decomposition when symmetric.
    ///
    /// Falls back to Taylor series for non-symmetric matrices.
    ///
    /// # Errors
    /// Returns an error for invalid input.
    pub fn matrix_exp_eigen(matrix: &Array2<f64>) -> Result<Array2<f64>, MatrixFunctionError> {
        validate_square(matrix)?;
        if !is_symmetric(matrix, DEFAULT_TOLERANCE) {
            return matrix_exp(matrix, 128, 1e-12);
        }

        let eigen =
            ndarray_eigen::symmetric(matrix).map_err(|_| MatrixFunctionError::ConvergenceFailed)?;
        let exp_values = eigen.eigenvalues.map(|value| value.exp());
        let diagonal = diagonal_from(&exp_values);
        Ok(eigen.eigenvectors.dot(&diagonal).dot(&eigen.eigenvectors.t()))
    }

    /// Compute matrix logarithm via Taylor expansion around identity.
    ///
    /// # Errors
    /// Returns an error for invalid input.
    pub fn matrix_log_taylor(
        matrix: &Array2<f64>,
        max_terms: usize,
        tolerance: f64,
    ) -> Result<Array2<f64>, MatrixFunctionError> {
        validate_square(matrix)?;
        let n = matrix.nrows();
        let identity = identity(n);
        let x = matrix - &identity;

        let mut result = Array2::<f64>::zeros((n, n));
        let mut term = x.clone();

        for k in 1..=max_terms.max(1) {
            let scale = if k % 2 == 0 { -1.0 } else { 1.0 } / usize_to_f64(k);
            result = &result + &(term.mapv(|value| scale * value));
            term = term.dot(&x);

            let delta = term.iter().map(|value| value.abs()).fold(0.0_f64, f64::max);
            if delta <= tolerance.max(DEFAULT_TOLERANCE) {
                break;
            }
        }

        Ok(result)
    }

    /// Compute matrix logarithm via eigen decomposition (symmetric PSD matrices).
    ///
    /// # Errors
    /// Returns an error if eigenvalues are non-positive.
    pub fn matrix_log_eigen(matrix: &Array2<f64>) -> Result<Array2<f64>, MatrixFunctionError> {
        validate_square(matrix)?;
        if !is_symmetric(matrix, DEFAULT_TOLERANCE) {
            return Err(MatrixFunctionError::NotSymmetric);
        }

        let eigen =
            ndarray_eigen::symmetric(matrix).map_err(|_| MatrixFunctionError::ConvergenceFailed)?;
        if eigen.eigenvalues.iter().any(|value| *value <= DEFAULT_TOLERANCE) {
            return Err(MatrixFunctionError::NotPositiveDefinite);
        }

        let log_values = eigen.eigenvalues.map(|value| value.ln());
        let diagonal = diagonal_from(&log_values);
        Ok(eigen.eigenvectors.dot(&diagonal).dot(&eigen.eigenvectors.t()))
    }

    /// Compute matrix logarithm via SVD.
    ///
    /// # Errors
    /// Returns an error if SVD fails.
    pub fn matrix_log_svd(matrix: &Array2<f64>) -> Result<Array2<f64>, MatrixFunctionError> {
        validate_square(matrix)?;
        let svd =
            ndarray_svd::decompose(matrix).map_err(|_| MatrixFunctionError::ConvergenceFailed)?;
        if svd.singular_values.iter().any(|value| *value <= DEFAULT_TOLERANCE) {
            return Err(MatrixFunctionError::NotPositiveDefinite);
        }

        let log_sigma = diagonal_from(&svd.singular_values.map(|value| value.ln()));
        Ok(svd.u.dot(&log_sigma).dot(&svd.vt))
    }

    /// Compute matrix power via eigen decomposition (symmetric matrices).
    ///
    /// # Errors
    /// Returns an error for non-symmetric inputs.
    pub fn matrix_power(
        matrix: &Array2<f64>,
        power: f64,
    ) -> Result<Array2<f64>, MatrixFunctionError> {
        validate_square(matrix)?;
        if !is_symmetric(matrix, DEFAULT_TOLERANCE) {
            return Err(MatrixFunctionError::NotSymmetric);
        }

        let eigen =
            ndarray_eigen::symmetric(matrix).map_err(|_| MatrixFunctionError::ConvergenceFailed)?;
        let powered_values = eigen.eigenvalues.map(|value| value.powf(power));
        let diagonal = diagonal_from(&powered_values);
        Ok(eigen.eigenvectors.dot(&diagonal).dot(&eigen.eigenvectors.t()))
    }

    /// Compute matrix sign via eigen decomposition (symmetric matrices).
    ///
    /// # Errors
    /// Returns an error for non-symmetric inputs.
    pub fn matrix_sign(matrix: &Array2<f64>) -> Result<Array2<f64>, MatrixFunctionError> {
        validate_square(matrix)?;
        if !is_symmetric(matrix, DEFAULT_TOLERANCE) {
            return Err(MatrixFunctionError::NotSymmetric);
        }

        let eigen =
            ndarray_eigen::symmetric(matrix).map_err(|_| MatrixFunctionError::ConvergenceFailed)?;
        let sign_values = eigen.eigenvalues.map(|value| {
            if *value > DEFAULT_TOLERANCE {
                1.0
            } else if *value < -DEFAULT_TOLERANCE {
                -1.0
            } else {
                0.0
            }
        });
        let diagonal = diagonal_from(&sign_values);
        Ok(eigen.eigenvectors.dot(&diagonal).dot(&eigen.eigenvectors.t()))
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use super::{MatrixFunctionError, ndarray_matrix_functions};

    #[test]
    fn exp_and_log_roundtrip_for_spd() {
        let matrix = Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 1.0, 2.0]).unwrap();
        let log_matrix = ndarray_matrix_functions::matrix_log_eigen(&matrix).unwrap();
        let roundtrip = ndarray_matrix_functions::matrix_exp_eigen(&log_matrix).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!((roundtrip[[i, j]] - matrix[[i, j]]).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn non_symmetric_log_is_rejected() {
        let matrix = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 0.0, 1.0]).unwrap();
        let result = ndarray_matrix_functions::matrix_log_eigen(&matrix);
        assert!(matches!(result, Err(MatrixFunctionError::NotSymmetric)));
    }
}
