//! Shared ndarray validation helpers.

use ndarray::{Array1, Array2};

use crate::errors::ShapeError;

/// Validate a square matrix input.
///
/// # Errors
///
/// Returns [`ShapeError::EmptyInput`] when `matrix` has zero elements, and
/// [`ShapeError::NotSquare`] when row and column counts differ.
pub fn validate_square_matrix<T>(matrix: &Array2<T>) -> Result<(), ShapeError> {
    if matrix.is_empty() {
        return Err(ShapeError::EmptyInput);
    }

    if matrix.nrows() != matrix.ncols() {
        return Err(ShapeError::NotSquare);
    }

    Ok(())
}

/// Validate a square matrix and right-hand-side vector pair.
///
/// # Errors
///
/// Returns [`ShapeError::EmptyInput`] when `matrix` or `rhs` is empty,
/// [`ShapeError::NotSquare`] when `matrix` is not square, and
/// [`ShapeError::DimensionMismatch`] when `rhs.len() != matrix.nrows()`.
pub fn validate_square_system<T>(matrix: &Array2<T>, rhs: &Array1<T>) -> Result<(), ShapeError> {
    validate_square_matrix(matrix)?;

    if rhs.is_empty() {
        return Err(ShapeError::EmptyInput);
    }

    if matrix.nrows() != rhs.len() {
        return Err(ShapeError::DimensionMismatch);
    }

    Ok(())
}
