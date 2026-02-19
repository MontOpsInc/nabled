//! # LU Decomposition
//!
//! LU decomposition with partial pivoting for solving linear systems and computing matrix inverses.

use nalgebra::{DMatrix, DVector, RealField};
use ndarray::{Array1, Array2};
use num_traits::Float;
use std::fmt;

/// Error types for LU decomposition
#[derive(Debug, Clone, PartialEq)]
pub enum LUError {
    /// Matrix is empty
    EmptyMatrix,
    /// Matrix is not square
    NotSquare,
    /// Matrix is singular (not invertible)
    SingularMatrix,
    /// Numerical instability detected
    NumericalInstability,
    /// Invalid input
    InvalidInput(String),
}

impl fmt::Display for LUError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LUError::EmptyMatrix => write!(f, "Matrix is empty"),
            LUError::NotSquare => write!(f, "Matrix must be square"),
            LUError::SingularMatrix => write!(f, "Matrix is singular"),
            LUError::NumericalInstability => write!(f, "Numerical instability detected"),
            LUError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for LUError {}

/// LU decomposition result for nalgebra
#[derive(Debug, Clone)]
pub struct NalgebraLUResult<T: RealField> {
    /// Lower triangular matrix L
    pub l: DMatrix<T>,
    /// Upper triangular matrix U
    pub u: DMatrix<T>,
    /// Internal LU decomposition (for solve/inverse)
    #[allow(dead_code)]
    lu: nalgebra::linalg::LU<T, nalgebra::Dyn, nalgebra::Dyn>,
}

/// LU decomposition result for ndarray
#[derive(Debug, Clone)]
pub struct NdarrayLUResult<T: Float> {
    /// Lower triangular matrix L
    pub l: Array2<T>,
    /// Upper triangular matrix U
    pub u: Array2<T>,
}

/// Nalgebra LU decomposition
pub mod nalgebra_lu {
    use super::*;
    use nalgebra::linalg::LU;

    /// Compute LU decomposition with partial pivoting
    pub fn compute_lu<T: RealField + Copy + num_traits::Float>(
        matrix: &DMatrix<T>,
    ) -> Result<NalgebraLUResult<T>, LUError> {
        if matrix.is_empty() {
            return Err(LUError::EmptyMatrix);
        }
        let (rows, cols) = matrix.shape();
        if rows != cols {
            return Err(LUError::NotSquare);
        }
        if matrix.iter().any(|&x| !num_traits::Float::is_finite(x)) {
            return Err(LUError::NumericalInstability);
        }

        let lu = LU::new(matrix.clone());
        let l = lu.l().clone();
        let u = lu.u().clone();

        Ok(NalgebraLUResult { l, u, lu })
    }

    /// Solve Ax = b for square matrix A
    pub fn solve<T: RealField + Copy>(
        matrix: &DMatrix<T>,
        rhs: &DVector<T>,
    ) -> Result<DVector<T>, LUError> {
        if matrix.is_empty() || rhs.is_empty() {
            return Err(LUError::EmptyMatrix);
        }
        let (rows, cols) = matrix.shape();
        if rows != cols {
            return Err(LUError::NotSquare);
        }
        if rows != rhs.len() {
            return Err(LUError::InvalidInput(
                "RHS length must match matrix dimensions".to_string(),
            ));
        }

        let lu = LU::new(matrix.clone());
        lu.solve(rhs).ok_or(LUError::SingularMatrix)
    }

    /// Compute matrix inverse
    pub fn inverse<T: RealField + Copy>(matrix: &DMatrix<T>) -> Result<DMatrix<T>, LUError> {
        if matrix.is_empty() {
            return Err(LUError::EmptyMatrix);
        }
        let (rows, cols) = matrix.shape();
        if rows != cols {
            return Err(LUError::NotSquare);
        }

        let lu = LU::new(matrix.clone());
        lu.try_inverse().ok_or(LUError::SingularMatrix)
    }
}

/// Ndarray LU decomposition (via nalgebra)
pub mod ndarray_lu {
    use super::*;
    use crate::utils::{ndarray_to_nalgebra, nalgebra_to_ndarray};

    /// Compute LU decomposition
    pub fn compute_lu<T: Float + RealField>(
        matrix: &Array2<T>,
    ) -> Result<NdarrayLUResult<T>, LUError> {
        let nalg = ndarray_to_nalgebra(matrix);
        let result = super::nalgebra_lu::compute_lu(&nalg)?;
        Ok(NdarrayLUResult {
            l: nalgebra_to_ndarray(&result.l),
            u: nalgebra_to_ndarray(&result.u),
        })
    }

    /// Solve Ax = b
    pub fn solve<T: Float + RealField>(
        matrix: &Array2<T>,
        rhs: &Array1<T>,
    ) -> Result<Array1<T>, LUError> {
        let nalg_matrix = ndarray_to_nalgebra(matrix);
        let nalg_rhs = DVector::from_vec(rhs.to_vec());
        let solution = super::nalgebra_lu::solve(&nalg_matrix, &nalg_rhs)?;
        Ok(Array1::from_vec(solution.as_slice().to_vec()))
    }

    /// Compute matrix inverse
    pub fn inverse<T: Float + RealField>(matrix: &Array2<T>) -> Result<Array2<T>, LUError> {
        let nalg = ndarray_to_nalgebra(matrix);
        let inv = super::nalgebra_lu::inverse(&nalg)?;
        Ok(nalgebra_to_ndarray(&inv))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_nalgebra_lu_solve() {
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let b = DVector::from_vec(vec![5.0, 6.0]);
        let x = nalgebra_lu::solve(&a, &b).unwrap();
        let ax = &a * &x;
        for i in 0..2 {
            assert_relative_eq!(ax[i], b[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_nalgebra_lu_inverse() {
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let inv = nalgebra_lu::inverse(&a).unwrap();
        let identity = &a * &inv;
        assert_relative_eq!(identity[(0, 0)], 1.0, epsilon = 1e-10);
        assert_relative_eq!(identity[(1, 1)], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_ndarray_lu_solve() {
        let a = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Array1::from_vec(vec![5.0, 6.0]);
        let x = ndarray_lu::solve(&a, &b).unwrap();
        let mut ax: Array1<f64> = Array1::zeros(2);
        for i in 0..2 {
            for j in 0..2 {
                ax[i] += a[[i, j]] * x[j];
            }
        }
        for i in 0..2 {
            assert_relative_eq!(ax[i], b[i], epsilon = 1e-10);
        }
    }
}
