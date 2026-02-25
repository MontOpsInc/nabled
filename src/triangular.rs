//! # Triangular Solve
//!
//! Solve linear systems with lower or upper triangular matrices.

use std::fmt;

use nalgebra::{DMatrix, DVector, RealField};
use ndarray::{Array1, Array2};
use num_traits::Float;

/// Error types for triangular solve
#[derive(Debug, Clone, PartialEq)]
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

    /// Solve Lx = b where L is lower triangular (forward substitution)
    pub fn solve_lower<T: RealField + Copy>(
        l: &DMatrix<T>,
        b: &DVector<T>,
    ) -> Result<DVector<T>, TriangularError> {
        if l.is_empty() || b.is_empty() {
            return Err(TriangularError::EmptyMatrix);
        }
        if !l.is_square() {
            return Err(TriangularError::NotSquare);
        }
        if l.nrows() != b.len() {
            return Err(TriangularError::DimensionMismatch);
        }
        let n = l.nrows();
        let mut x = b.clone();
        for i in 0..n {
            if l[(i, i)] == T::zero() {
                return Err(TriangularError::Singular);
            }
            let mut sum = T::zero();
            for j in 0..i {
                sum += l[(i, j)] * x[j];
            }
            x[i] = (b[i] - sum) / l[(i, i)];
        }
        Ok(x)
    }

    /// Solve Ux = b where U is upper triangular (back substitution)
    pub fn solve_upper<T: RealField + Copy>(
        u: &DMatrix<T>,
        b: &DVector<T>,
    ) -> Result<DVector<T>, TriangularError> {
        if u.is_empty() || b.is_empty() {
            return Err(TriangularError::EmptyMatrix);
        }
        if !u.is_square() {
            return Err(TriangularError::NotSquare);
        }
        if u.nrows() != b.len() {
            return Err(TriangularError::DimensionMismatch);
        }
        let n = u.nrows();
        let mut x = b.clone();
        for i in (0..n).rev() {
            if u[(i, i)] == T::zero() {
                return Err(TriangularError::Singular);
            }
            let mut sum = T::zero();
            for j in (i + 1)..n {
                sum += u[(i, j)] * x[j];
            }
            x[i] = (b[i] - sum) / u[(i, i)];
        }
        Ok(x)
    }
}

/// Ndarray triangular solve
pub mod ndarray_triangular {
    use super::*;
    use crate::utils::ndarray_to_nalgebra;

    /// Solve Lx = b where L is lower triangular
    pub fn solve_lower<T: Float + RealField>(
        l: &Array2<T>,
        b: &Array1<T>,
    ) -> Result<Array1<T>, TriangularError> {
        let nalg_l = ndarray_to_nalgebra(l);
        let nalg_b = DVector::from_vec(b.to_vec());
        let result = super::nalgebra_triangular::solve_lower(&nalg_l, &nalg_b)?;
        Ok(Array1::from_vec(result.as_slice().to_vec()))
    }

    /// Solve Ux = b where U is upper triangular
    pub fn solve_upper<T: Float + RealField>(
        u: &Array2<T>,
        b: &Array1<T>,
    ) -> Result<Array1<T>, TriangularError> {
        let nalg_u = ndarray_to_nalgebra(u);
        let nalg_b = DVector::from_vec(b.to_vec());
        let result = super::nalgebra_triangular::solve_upper(&nalg_u, &nalg_b)?;
        Ok(Array1::from_vec(result.as_slice().to_vec()))
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
}
