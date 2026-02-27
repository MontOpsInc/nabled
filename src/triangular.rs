//! # Triangular Solve
//!
//! Solve linear systems with lower or upper triangular matrices.

use std::fmt;

use nalgebra::{DMatrix, DVector, RealField};
use ndarray::{Array1, Array2};
use num_traits::Float;

/// Error types for triangular solve
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TriangularError {
    /// Matrix is empty
    EmptyMatrix,
    /// Matrix is not square
    NotSquare,
    /// Singular (zero on diagonal)
    Singular,
    /// Dimension mismatch
    DimensionMismatch,
}

impl fmt::Display for TriangularError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TriangularError::EmptyMatrix => write!(f, "Matrix is empty"),
            TriangularError::NotSquare => write!(f, "Matrix must be square"),
            TriangularError::Singular => write!(f, "Matrix is singular (zero on diagonal)"),
            TriangularError::DimensionMismatch => write!(f, "Dimension mismatch"),
        }
    }
}

impl std::error::Error for TriangularError {}

/// Nalgebra triangular solve (forward/back substitution)
pub mod nalgebra_triangular {
    use super::*;

    /// Solve Lx = b where L is lower triangular (forward substitution).
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn solve_lower<T: RealField + Copy>(
        l: &DMatrix<T>,
        b: &DVector<T>,
    ) -> Result<DVector<T>, TriangularError> {
        crate::backend::triangular::solve_nalgebra_lower(l, b)
    }

    /// Solve Ux = b where U is upper triangular (back substitution).
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn solve_upper<T: RealField + Copy>(
        u: &DMatrix<T>,
        b: &DVector<T>,
    ) -> Result<DVector<T>, TriangularError> {
        crate::backend::triangular::solve_nalgebra_upper(u, b)
    }
}

/// Ndarray triangular solve
pub mod ndarray_triangular {
    use super::*;

    /// Solve Lx = b where L is lower triangular.
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn solve_lower<T: Float + RealField>(
        l: &Array2<T>,
        b: &Array1<T>,
    ) -> Result<Array1<T>, TriangularError> {
        crate::backend::triangular::solve_ndarray_lower(l, b)
    }

    /// Solve Ux = b where U is upper triangular.
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn solve_upper<T: Float + RealField>(
        u: &Array2<T>,
        b: &Array1<T>,
    ) -> Result<Array1<T>, TriangularError> {
        crate::backend::triangular::solve_ndarray_upper(u, b)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn test_solve_lower() {
        let l = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 1.0, 3.0]);
        let b = DVector::from_vec(vec![4.0, 8.0]);
        let x = nalgebra_triangular::solve_lower(&l, &b).unwrap();
        let lx = &l * &x;
        for i in 0..2 {
            assert_relative_eq!(lx[i], b[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_solve_upper() {
        let u = DMatrix::from_row_slice(2, 2, &[2.0, 1.0, 0.0, 3.0]);
        let b = DVector::from_vec(vec![4.0, 9.0]);
        let x = nalgebra_triangular::solve_upper(&u, &b).unwrap();
        let ux = &u * &x;
        for i in 0..2 {
            assert_relative_eq!(ux[i], b[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_ndarray_solve_lower() {
        let l = Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 1.0, 3.0]).unwrap();
        let b = Array1::from_vec(vec![4.0, 8.0]);
        let x = ndarray_triangular::solve_lower(&l, &b).unwrap();
        let lx = l.dot(&x);

        for i in 0..2 {
            assert_relative_eq!(lx[i], b[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_ndarray_solve_upper() {
        let u = Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 0.0, 3.0]).unwrap();
        let b = Array1::from_vec(vec![4.0, 9.0]);
        let x = ndarray_triangular::solve_upper(&u, &b).unwrap();
        let ux = u.dot(&x);

        for i in 0..2 {
            assert_relative_eq!(ux[i], b[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_triangular_error_display_variants() {
        assert!(format!("{}", TriangularError::EmptyMatrix).contains("empty"));
        assert!(format!("{}", TriangularError::NotSquare).contains("square"));
        assert!(format!("{}", TriangularError::Singular).contains("singular"));
        assert!(format!("{}", TriangularError::DimensionMismatch).contains("Dimension"));
    }

    #[test]
    fn test_triangular_error_paths() {
        let lower_non_square = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
        let vec1 = DVector::from_vec(vec![1.0]);
        assert!(matches!(
            nalgebra_triangular::solve_lower(&lower_non_square, &vec1),
            Err(TriangularError::NotSquare)
        ));
        let lower_singular = DMatrix::from_row_slice(2, 2, &[0.0, 0.0, 1.0, 1.0]);
        assert!(matches!(
            nalgebra_triangular::solve_lower(&lower_singular, &DVector::from_vec(vec![1.0, 2.0])),
            Err(TriangularError::Singular)
        ));

        let upper_non_square = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
        assert!(matches!(
            nalgebra_triangular::solve_upper(&upper_non_square, &vec1),
            Err(TriangularError::NotSquare)
        ));
        let upper_singular = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 0.0, 0.0]);
        assert!(matches!(
            nalgebra_triangular::solve_upper(&upper_singular, &DVector::from_vec(vec![1.0, 2.0])),
            Err(TriangularError::Singular)
        ));
    }
}
