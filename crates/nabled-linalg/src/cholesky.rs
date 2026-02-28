//! Cholesky decomposition over ndarray matrices.

use std::fmt;

use ndarray::{Array1, Array2};

use crate::internal::{DEFAULT_TOLERANCE, validate_finite, validate_square_non_empty};

/// Result of Cholesky decomposition.
#[derive(Debug, Clone)]
pub struct NdarrayCholeskyResult {
    /// Lower-triangular factor `L` where `A = L L^T`.
    pub l: Array2<f64>,
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

fn map_validation_error(error: &'static str) -> CholeskyError {
    match error {
        "empty" => CholeskyError::EmptyMatrix,
        "not_square" => CholeskyError::NotSquare,
        "non_finite" => CholeskyError::NumericalInstability,
        _ => CholeskyError::InvalidInput(error.to_string()),
    }
}

fn compute_cholesky_impl(matrix: &Array2<f64>) -> Result<Array2<f64>, CholeskyError> {
    validate_square_non_empty(matrix).map_err(map_validation_error)?;
    validate_finite(matrix).map_err(map_validation_error)?;

    let n = matrix.nrows();
    let mut l = Array2::<f64>::zeros((n, n));

    for i in 0..n {
        for j in 0..=i {
            let mut sum = matrix[[i, j]];
            for k in 0..j {
                sum -= l[[i, k]] * l[[j, k]];
            }

            if i == j {
                if sum <= DEFAULT_TOLERANCE {
                    return Err(CholeskyError::NotPositiveDefinite);
                }
                l[[i, j]] = sum.sqrt();
            } else {
                let diagonal = l[[j, j]];
                if diagonal.abs() <= DEFAULT_TOLERANCE {
                    return Err(CholeskyError::NotPositiveDefinite);
                }
                l[[i, j]] = sum / diagonal;
            }
        }
    }

    Ok(l)
}

/// Ndarray Cholesky functions.
pub mod ndarray_cholesky {
    use super::*;

    /// Compute Cholesky decomposition.
    ///
    /// # Errors
    /// Returns an error if matrix is not SPD.
    pub fn compute_cholesky(matrix: &Array2<f64>) -> Result<NdarrayCholeskyResult, CholeskyError> {
        let l = compute_cholesky_impl(matrix)?;
        Ok(NdarrayCholeskyResult { l })
    }

    /// Solve `Ax=b` using Cholesky decomposition.
    ///
    /// # Errors
    /// Returns an error for invalid dimensions or non-SPD matrix.
    #[allow(clippy::many_single_char_names)]
    pub fn solve(matrix: &Array2<f64>, rhs: &Array1<f64>) -> Result<Array1<f64>, CholeskyError> {
        if rhs.len() != matrix.nrows() {
            return Err(CholeskyError::InvalidInput(
                "RHS length must match matrix dimensions".to_string(),
            ));
        }

        let l = compute_cholesky_impl(matrix)?;
        let n = l.nrows();

        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut sum = rhs[i];
            for j in 0..i {
                sum -= l[[i, j]] * y[j];
            }
            y[i] = sum / l[[i, i]];
        }

        let mut x = Array1::<f64>::zeros(n);
        for i_rev in 0..n {
            let i = n - 1 - i_rev;
            let mut sum = y[i];
            for j in (i + 1)..n {
                sum -= l[[j, i]] * x[j];
            }
            x[i] = sum / l[[i, i]];
        }

        Ok(x)
    }

    /// Compute inverse via Cholesky decomposition.
    ///
    /// # Errors
    /// Returns an error if matrix is not SPD.
    pub fn inverse(matrix: &Array2<f64>) -> Result<Array2<f64>, CholeskyError> {
        let l = compute_cholesky_impl(matrix)?;
        let n = l.nrows();
        let mut inverse = Array2::<f64>::zeros((n, n));

        for col in 0..n {
            let mut e = Array1::<f64>::zeros(n);
            e[col] = 1.0;
            let solution = solve(matrix, &e)?;
            for row in 0..n {
                inverse[[row, col]] = solution[row];
            }
        }

        Ok(inverse)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2};

    use super::{CholeskyError, ndarray_cholesky};

    #[test]
    fn cholesky_reconstructs_spd_matrix() {
        let matrix = Array2::from_shape_vec((2, 2), vec![4.0, 2.0, 2.0, 3.0]).unwrap();
        let decomposition = ndarray_cholesky::compute_cholesky(&matrix).unwrap();
        let reconstructed = decomposition.l.dot(&decomposition.l.t());
        for i in 0..2 {
            for j in 0..2 {
                assert!((matrix[[i, j]] - reconstructed[[i, j]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn solve_reconstructs_rhs() {
        let matrix = Array2::from_shape_vec((2, 2), vec![4.0, 2.0, 2.0, 3.0]).unwrap();
        let rhs = Array1::from_vec(vec![1.0, 1.0]);
        let x = ndarray_cholesky::solve(&matrix, &rhs).unwrap();
        let reconstructed = matrix.dot(&x);
        assert!((reconstructed[0] - rhs[0]).abs() < 1e-10);
        assert!((reconstructed[1] - rhs[1]).abs() < 1e-10);
    }

    #[test]
    fn non_spd_input_errors() {
        let matrix = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 2.0, 1.0]).unwrap();
        let result = ndarray_cholesky::compute_cholesky(&matrix);
        assert!(matches!(result, Err(CholeskyError::NotPositiveDefinite)));
    }
}
