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

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2};

    use super::{validate_square_matrix, validate_square_system};
    use crate::errors::ShapeError;

    #[test]
    fn validate_square_matrix_rejects_non_square() {
        let matrix = Array2::<f64>::zeros((2, 3));
        assert!(matches!(validate_square_matrix(&matrix), Err(ShapeError::NotSquare)));
    }

    #[test]
    fn validate_square_system_rejects_empty_rhs_and_dimension_mismatch() {
        let matrix = Array2::<f64>::eye(2);
        let empty_rhs = Array1::<f64>::zeros(0);
        assert!(matches!(validate_square_system(&matrix, &empty_rhs), Err(ShapeError::EmptyInput)));

        let bad_rhs = Array1::<f64>::zeros(3);
        assert!(matches!(
            validate_square_system(&matrix, &bad_rhs),
            Err(ShapeError::DimensionMismatch)
        ));
    }

    #[test]
    fn validate_square_system_accepts_matching_shapes() {
        let matrix = Array2::<f64>::eye(3);
        let rhs = Array1::<f64>::ones(3);
        assert!(validate_square_system(&matrix, &rhs).is_ok());
    }
}
