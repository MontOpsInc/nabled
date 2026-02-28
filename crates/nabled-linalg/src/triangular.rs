//! Ndarray-native triangular solve kernels.

use std::fmt;

use nabled_core::errors::ShapeError;
use nabled_core::validation::validate_square_system;
use ndarray::{Array1, Array2};
use num_traits::Float;
use thiserror::Error;

/// Error returned by triangular solve kernels.
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum TriangularSolveError {
    /// Shape or dimensional validation failed.
    #[error(transparent)]
    Shape(#[from] ShapeError),
    /// A zero pivot was encountered.
    #[error("matrix is singular (zero on diagonal)")]
    Singular,
}

/// Compatibility error type for ndarray triangular solve API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriangularError {
    /// Matrix is empty.
    EmptyMatrix,
    /// Matrix is not square.
    NotSquare,
    /// Matrix/vector dimensions do not align.
    DimensionMismatch,
    /// Matrix is singular.
    Singular,
}

impl fmt::Display for TriangularError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TriangularError::EmptyMatrix => write!(f, "Matrix is empty"),
            TriangularError::NotSquare => write!(f, "Matrix must be square"),
            TriangularError::DimensionMismatch => write!(f, "Dimension mismatch"),
            TriangularError::Singular => write!(f, "Matrix is singular (zero on diagonal)"),
        }
    }
}

impl std::error::Error for TriangularError {}

fn map_triangular_error(error: TriangularSolveError) -> TriangularError {
    match error {
        TriangularSolveError::Shape(shape_error) => match shape_error {
            ShapeError::EmptyInput => TriangularError::EmptyMatrix,
            ShapeError::NotSquare => TriangularError::NotSquare,
            ShapeError::DimensionMismatch => TriangularError::DimensionMismatch,
        },
        TriangularSolveError::Singular => TriangularError::Singular,
    }
}

/// Solve `Lx = b` with forward substitution.
///
/// # Errors
///
/// Returns [`TriangularSolveError::Shape`] when `matrix`/`rhs` dimensions are invalid,
/// and [`TriangularSolveError::Singular`] when a zero pivot is encountered.
pub fn solve_lower<T>(
    matrix: &Array2<T>,
    rhs: &Array1<T>,
) -> Result<Array1<T>, TriangularSolveError>
where
    T: Float,
{
    validate_square_system(matrix, rhs)?;

    let n = matrix.nrows();
    let mut solution = rhs.clone();
    for i in 0..n {
        if matrix[[i, i]] == T::zero() {
            return Err(TriangularSolveError::Singular);
        }

        let mut sum = T::zero();
        for j in 0..i {
            sum = sum + matrix[[i, j]] * solution[j];
        }

        solution[i] = (rhs[i] - sum) / matrix[[i, i]];
    }

    Ok(solution)
}

/// Solve `Ux = b` with back substitution.
///
/// # Errors
///
/// Returns [`TriangularSolveError::Shape`] when `matrix`/`rhs` dimensions are invalid,
/// and [`TriangularSolveError::Singular`] when a zero pivot is encountered.
pub fn solve_upper<T>(
    matrix: &Array2<T>,
    rhs: &Array1<T>,
) -> Result<Array1<T>, TriangularSolveError>
where
    T: Float,
{
    validate_square_system(matrix, rhs)?;

    let n = matrix.nrows();
    let mut solution = rhs.clone();
    for i in (0..n).rev() {
        if matrix[[i, i]] == T::zero() {
            return Err(TriangularSolveError::Singular);
        }

        let mut sum = T::zero();
        for j in (i + 1)..n {
            sum = sum + matrix[[i, j]] * solution[j];
        }

        solution[i] = (rhs[i] - sum) / matrix[[i, i]];
    }

    Ok(solution)
}

/// Ndarray triangular solve compatibility API.
pub mod ndarray_triangular {
    use super::*;

    /// Solve `Lx=b` with forward substitution.
    ///
    /// # Errors
    /// Returns an error for invalid dimensions or singular matrices.
    pub fn solve_lower<T>(matrix: &Array2<T>, rhs: &Array1<T>) -> Result<Array1<T>, TriangularError>
    where
        T: Float,
    {
        super::solve_lower(matrix, rhs).map_err(map_triangular_error)
    }

    /// Solve `Ux=b` with back substitution.
    ///
    /// # Errors
    /// Returns an error for invalid dimensions or singular matrices.
    pub fn solve_upper<T>(matrix: &Array2<T>, rhs: &Array1<T>) -> Result<Array1<T>, TriangularError>
    where
        T: Float,
    {
        super::solve_upper(matrix, rhs).map_err(map_triangular_error)
    }
}

#[cfg(test)]
mod tests {
    use nabled_core::errors::ShapeError;
    use ndarray::{arr1, arr2};

    use super::{TriangularSolveError, solve_lower, solve_upper};

    #[test]
    fn lower_solve_reconstructs_rhs() {
        let lower = arr2(&[[2.0_f64, 0.0], [1.0, 3.0]]);
        let rhs = arr1(&[4.0_f64, 8.0]);

        let solution = solve_lower(&lower, &rhs).expect("lower triangular solve should succeed");
        let reconstructed = lower.dot(&solution);

        assert!((reconstructed[0] - rhs[0]).abs() < 1e-10);
        assert!((reconstructed[1] - rhs[1]).abs() < 1e-10);
    }

    #[test]
    fn upper_solve_reconstructs_rhs() {
        let upper = arr2(&[[2.0_f64, 1.0], [0.0, 3.0]]);
        let rhs = arr1(&[4.0_f64, 9.0]);

        let solution = solve_upper(&upper, &rhs).expect("upper triangular solve should succeed");
        let reconstructed = upper.dot(&solution);

        assert!((reconstructed[0] - rhs[0]).abs() < 1e-10);
        assert!((reconstructed[1] - rhs[1]).abs() < 1e-10);
    }

    #[test]
    fn singular_matrix_errors() {
        let lower = arr2(&[[0.0_f64, 0.0], [1.0, 1.0]]);
        let rhs = arr1(&[1.0_f64, 2.0]);

        let result = solve_lower(&lower, &rhs);
        assert!(matches!(result, Err(TriangularSolveError::Singular)));
    }

    #[test]
    fn invalid_shape_errors() {
        let non_square = arr2(&[[1.0_f64, 0.0]]);
        let rhs = arr1(&[1.0_f64]);

        let result = solve_upper(&non_square, &rhs);
        assert!(matches!(result, Err(TriangularSolveError::Shape(ShapeError::NotSquare))));
    }
}
