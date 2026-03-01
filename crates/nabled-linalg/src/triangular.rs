//! Ndarray-native triangular solve kernels.

use nabled_core::errors::ShapeError;
use nabled_core::validation::validate_square_system;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_traits::Float;
use thiserror::Error;

/// Error returned by triangular solve kernels.
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum TriangularError {
    /// Shape or dimensional validation failed.
    #[error(transparent)]
    Shape(#[from] ShapeError),
    /// A zero pivot was encountered.
    #[error("matrix is singular (zero on diagonal)")]
    Singular,
}

fn solve_lower_into_internal<T>(
    matrix: &Array2<T>,
    rhs: &Array1<T>,
    output: &mut Array1<T>,
) -> Result<(), TriangularError>
where
    T: Float,
{
    validate_square_system(matrix, rhs)?;
    if output.len() != rhs.len() {
        return Err(TriangularError::Shape(ShapeError::DimensionMismatch));
    }

    let n = matrix.nrows();
    output.assign(rhs);
    for i in 0..n {
        if matrix[[i, i]] == T::zero() {
            return Err(TriangularError::Singular);
        }

        let mut sum = T::zero();
        for j in 0..i {
            sum = sum + matrix[[i, j]] * output[j];
        }

        output[i] = (rhs[i] - sum) / matrix[[i, i]];
    }

    Ok(())
}

fn solve_upper_into_internal<T>(
    matrix: &Array2<T>,
    rhs: &Array1<T>,
    output: &mut Array1<T>,
) -> Result<(), TriangularError>
where
    T: Float,
{
    validate_square_system(matrix, rhs)?;
    if output.len() != rhs.len() {
        return Err(TriangularError::Shape(ShapeError::DimensionMismatch));
    }

    let n = matrix.nrows();
    output.assign(rhs);
    for i in (0..n).rev() {
        if matrix[[i, i]] == T::zero() {
            return Err(TriangularError::Singular);
        }

        let mut sum = T::zero();
        for j in (i + 1)..n {
            sum = sum + matrix[[i, j]] * output[j];
        }

        output[i] = (rhs[i] - sum) / matrix[[i, i]];
    }

    Ok(())
}

/// Solve `Lx = b` with forward substitution.
///
/// # Errors
/// Returns an error for shape mismatches or singular pivots.
pub fn solve_lower<T>(matrix: &Array2<T>, rhs: &Array1<T>) -> Result<Array1<T>, TriangularError>
where
    T: Float,
{
    let mut output = Array1::<T>::zeros(rhs.len());
    solve_lower_into(matrix, rhs, &mut output)?;
    Ok(output)
}

/// Solve `Lx = b` with forward substitution from matrix/vector views.
///
/// # Errors
/// Returns an error for shape mismatches or singular pivots.
pub fn solve_lower_view<T>(
    matrix: &ArrayView2<'_, T>,
    rhs: &ArrayView1<'_, T>,
) -> Result<Array1<T>, TriangularError>
where
    T: Float,
{
    solve_lower(&matrix.to_owned(), &rhs.to_owned())
}

/// Solve `Lx = b` with forward substitution into `output`.
///
/// # Errors
/// Returns an error for shape mismatches or singular pivots.
pub fn solve_lower_into<T>(
    matrix: &Array2<T>,
    rhs: &Array1<T>,
    output: &mut Array1<T>,
) -> Result<(), TriangularError>
where
    T: Float,
{
    solve_lower_into_internal(matrix, rhs, output)
}

/// Solve `Ux = b` with back substitution.
///
/// # Errors
/// Returns an error for shape mismatches or singular pivots.
pub fn solve_upper<T>(matrix: &Array2<T>, rhs: &Array1<T>) -> Result<Array1<T>, TriangularError>
where
    T: Float,
{
    let mut output = Array1::<T>::zeros(rhs.len());
    solve_upper_into(matrix, rhs, &mut output)?;
    Ok(output)
}

/// Solve `Ux = b` with back substitution from matrix/vector views.
///
/// # Errors
/// Returns an error for shape mismatches or singular pivots.
pub fn solve_upper_view<T>(
    matrix: &ArrayView2<'_, T>,
    rhs: &ArrayView1<'_, T>,
) -> Result<Array1<T>, TriangularError>
where
    T: Float,
{
    solve_upper(&matrix.to_owned(), &rhs.to_owned())
}

/// Solve `Ux = b` with back substitution into `output`.
///
/// # Errors
/// Returns an error for shape mismatches or singular pivots.
pub fn solve_upper_into<T>(
    matrix: &Array2<T>,
    rhs: &Array1<T>,
    output: &mut Array1<T>,
) -> Result<(), TriangularError>
where
    T: Float,
{
    solve_upper_into_internal(matrix, rhs, output)
}

#[cfg(test)]
mod tests {
    use nabled_core::errors::ShapeError;
    use ndarray::{arr1, arr2};

    use super::*;

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
        assert!(matches!(result, Err(TriangularError::Singular)));
    }

    #[test]
    fn invalid_shape_errors() {
        let non_square = arr2(&[[1.0_f64, 0.0]]);
        let rhs = arr1(&[1.0_f64]);

        let result = solve_upper(&non_square, &rhs);
        assert!(matches!(result, Err(TriangularError::Shape(ShapeError::NotSquare))));
    }

    #[test]
    fn view_variants_match_owned() {
        let lower = arr2(&[[2.0_f64, 0.0], [1.0, 3.0]]);
        let upper = arr2(&[[2.0_f64, 1.0], [0.0, 3.0]]);
        let rhs = arr1(&[4.0_f64, 9.0]);

        let lower_owned = solve_lower(&lower, &rhs).unwrap();
        let lower_view = solve_lower_view(&lower.view(), &rhs.view()).unwrap();
        let upper_owned = solve_upper(&upper, &rhs).unwrap();
        let upper_view = solve_upper_view(&upper.view(), &rhs.view()).unwrap();

        for i in 0..rhs.len() {
            assert!((lower_owned[i] - lower_view[i]).abs() < 1e-12);
            assert!((upper_owned[i] - upper_view[i]).abs() < 1e-12);
        }
    }
}
