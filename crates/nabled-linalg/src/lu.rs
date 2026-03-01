//! LU decomposition over ndarray matrices.

use std::fmt;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::internal::{
    DEFAULT_TOLERANCE, lu_decompose, validate_finite, validate_square_non_empty,
};
#[cfg(not(feature = "openblas-system"))]
use crate::internal::{inverse_from_lu, lu_solve};

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

fn decompose_internal(matrix: &Array2<f64>) -> Result<(NdarrayLUResult, Vec<usize>, i8), LUError> {
    let (l, u, pivots, sign) = lu_decompose(matrix).map_err(map_lu_error)?;
    Ok((NdarrayLUResult { l, u }, pivots, sign))
}

#[cfg(feature = "openblas-system")]
fn solve_provider(matrix: &Array2<f64>, rhs: &Array1<f64>) -> Result<Array1<f64>, LUError> {
    use ndarray_linalg::Solve as _;

    validate_square_non_empty(matrix).map_err(map_lu_error)?;
    validate_finite(matrix).map_err(map_lu_error)?;
    matrix.solve(rhs).map_err(|_| LUError::SingularMatrix)
}

#[cfg(feature = "openblas-system")]
fn inverse_provider(matrix: &Array2<f64>) -> Result<Array2<f64>, LUError> {
    use ndarray_linalg::Inverse as _;

    validate_square_non_empty(matrix).map_err(map_lu_error)?;
    validate_finite(matrix).map_err(map_lu_error)?;
    matrix.inv().map_err(|_| LUError::SingularMatrix)
}

#[cfg(feature = "openblas-system")]
fn determinant_provider(matrix: &Array2<f64>) -> Result<f64, LUError> {
    use ndarray_linalg::Determinant as _;

    validate_square_non_empty(matrix).map_err(map_lu_error)?;
    validate_finite(matrix).map_err(map_lu_error)?;
    matrix.det().map_err(|_| LUError::SingularMatrix)
}

/// Ndarray LU functions.
pub mod ndarray_lu {
    use super::*;

    fn decompose_with_metadata(
        matrix: &Array2<f64>,
    ) -> Result<(NdarrayLUResult, Vec<usize>, i8), LUError> {
        decompose_internal(matrix)
    }

    /// Compute LU decomposition with partial pivoting.
    ///
    /// # Errors
    /// Returns an error if input is invalid or decomposition fails.
    pub fn decompose(matrix: &Array2<f64>) -> Result<NdarrayLUResult, LUError> {
        let (result, _, _) = decompose_with_metadata(matrix)?;
        Ok(result)
    }

    /// Compute LU decomposition with partial pivoting from a matrix view.
    ///
    /// # Errors
    /// Returns an error if input is invalid or decomposition fails.
    pub fn decompose_view(matrix: &ArrayView2<'_, f64>) -> Result<NdarrayLUResult, LUError> {
        decompose(&matrix.to_owned())
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

        #[cfg(feature = "openblas-system")]
        {
            solve_provider(matrix, rhs)
        }
        #[cfg(not(feature = "openblas-system"))]
        {
            let (decomposition, pivots, _) = decompose_with_metadata(matrix)?;
            lu_solve(&decomposition.l, &decomposition.u, &pivots, rhs).map_err(map_lu_error)
        }
    }

    /// Solve `Ax=b` using LU decomposition from matrix/vector views.
    ///
    /// # Errors
    /// Returns an error if dimensions are incompatible or matrix is singular.
    pub fn solve_view(
        matrix: &ArrayView2<'_, f64>,
        rhs: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, LUError> {
        solve(&matrix.to_owned(), &rhs.to_owned())
    }

    /// Compute matrix inverse via LU decomposition.
    ///
    /// # Errors
    /// Returns an error if matrix is singular.
    pub fn inverse(matrix: &Array2<f64>) -> Result<Array2<f64>, LUError> {
        #[cfg(feature = "openblas-system")]
        {
            inverse_provider(matrix)
        }
        #[cfg(not(feature = "openblas-system"))]
        {
            let (decomposition, pivots, _) = decompose_with_metadata(matrix)?;
            inverse_from_lu(&decomposition.l, &decomposition.u, &pivots).map_err(map_lu_error)
        }
    }

    /// Compute matrix inverse via LU decomposition from a matrix view.
    ///
    /// # Errors
    /// Returns an error if matrix is singular.
    pub fn inverse_view(matrix: &ArrayView2<'_, f64>) -> Result<Array2<f64>, LUError> {
        inverse(&matrix.to_owned())
    }

    /// Compute determinant via LU decomposition.
    ///
    /// # Errors
    /// Returns an error if decomposition fails.
    pub fn determinant(matrix: &Array2<f64>) -> Result<f64, LUError> {
        #[cfg(feature = "openblas-system")]
        {
            determinant_provider(matrix)
        }
        #[cfg(not(feature = "openblas-system"))]
        {
            let (decomposition, _, sign) = decompose_with_metadata(matrix)?;
            let mut determinant = f64::from(sign);
            for i in 0..decomposition.u.nrows() {
                determinant *= decomposition.u[[i, i]];
            }
            if !determinant.is_finite() {
                return Err(LUError::NumericalInstability);
            }
            Ok(determinant)
        }
    }

    /// Compute determinant via LU decomposition from a matrix view.
    ///
    /// # Errors
    /// Returns an error if decomposition fails.
    pub fn determinant_view(matrix: &ArrayView2<'_, f64>) -> Result<f64, LUError> {
        determinant(&matrix.to_owned())
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

    /// Compute signed log-determinant via LU decomposition from a matrix view.
    ///
    /// # Errors
    /// Returns an error if matrix is singular.
    pub fn log_determinant_view(
        matrix: &ArrayView2<'_, f64>,
    ) -> Result<LogDetResult<f64>, LUError> {
        log_determinant(&matrix.to_owned())
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

    #[test]
    fn inverse_multiplied_by_matrix_is_identity() {
        let matrix = Array2::from_shape_vec((2, 2), vec![4.0, 7.0, 2.0, 6.0]).unwrap();
        let inverse = ndarray_lu::inverse(&matrix).unwrap();
        let product = matrix.dot(&inverse);
        assert!((product[[0, 0]] - 1.0).abs() < 1e-8);
        assert!((product[[1, 1]] - 1.0).abs() < 1e-8);
        assert!(product[[0, 1]].abs() < 1e-8);
        assert!(product[[1, 0]].abs() < 1e-8);
    }

    #[test]
    fn log_determinant_has_expected_sign_and_value() {
        let matrix = Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 0.0, -3.0]).unwrap();
        let result = ndarray_lu::log_determinant(&matrix).unwrap();
        assert_eq!(result.sign, -1);
        assert!((result.ln_abs_det - (6.0_f64).ln()).abs() < 1e-10);
    }

    #[test]
    fn solve_rejects_bad_rhs_length() {
        let matrix = Array2::eye(2);
        let rhs = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let result = ndarray_lu::solve(&matrix, &rhs);
        assert!(matches!(result, Err(LUError::InvalidInput(_))));
    }

    #[test]
    fn decompose_exposes_factors() {
        let matrix = Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 4.0, 3.0]).unwrap();
        let lu = ndarray_lu::decompose(&matrix).unwrap();
        assert_eq!(lu.l.dim(), (2, 2));
        assert_eq!(lu.u.dim(), (2, 2));
    }

    #[test]
    fn view_variants_match_owned() {
        let matrix = Array2::from_shape_vec((2, 2), vec![4.0, 7.0, 2.0, 6.0]).unwrap();
        let rhs = Array1::from_vec(vec![5.0, 7.0]);

        let owned = ndarray_lu::decompose(&matrix).unwrap();
        let viewed = ndarray_lu::decompose_view(&matrix.view()).unwrap();
        assert_eq!(owned.l.dim(), viewed.l.dim());
        assert_eq!(owned.u.dim(), viewed.u.dim());

        let solution_owned = ndarray_lu::solve(&matrix, &rhs).unwrap();
        let solution_view = ndarray_lu::solve_view(&matrix.view(), &rhs.view()).unwrap();
        for i in 0..rhs.len() {
            assert!((solution_owned[i] - solution_view[i]).abs() < 1e-12);
        }

        let inverse_owned = ndarray_lu::inverse(&matrix).unwrap();
        let inverse_view = ndarray_lu::inverse_view(&matrix.view()).unwrap();
        for i in 0..matrix.nrows() {
            for j in 0..matrix.ncols() {
                assert!((inverse_owned[[i, j]] - inverse_view[[i, j]]).abs() < 1e-12);
            }
        }

        let det_owned = ndarray_lu::determinant(&matrix).unwrap();
        let det_view = ndarray_lu::determinant_view(&matrix.view()).unwrap();
        assert!((det_owned - det_view).abs() < 1e-12);

        let logdet_owned = ndarray_lu::log_determinant(&matrix).unwrap();
        let logdet_view = ndarray_lu::log_determinant_view(&matrix.view()).unwrap();
        assert_eq!(logdet_owned.sign, logdet_view.sign);
        assert!((logdet_owned.ln_abs_det - logdet_view.ln_abs_det).abs() < 1e-12);
    }
}
