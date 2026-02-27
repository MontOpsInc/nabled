//! # Sylvester and Lyapunov Equations
//!
//! Sylvester equation: AX + XB = C
//! Lyapunov equation: AX + XA^T = Q (special case with B = A^T)
//!
//! Uses Bartels-Stewart algorithm via Schur decomposition.

use std::fmt;

use nalgebra::{DMatrix, RealField};
use ndarray::Array2;
use num_traits::Float;

use crate::schur::{SchurError, nalgebra_schur};

/// Error types for matrix equations
#[derive(Debug, Clone, PartialEq)]
pub enum SylvesterError {
    /// Matrix is empty
    EmptyMatrix,
    /// Dimension mismatch
    DimensionMismatch,
    /// Singular system (eigenvalues of A and -B overlap)
    SingularSystem,
    /// Schur decomposition failed
    SchurFailed(SchurError),
}

impl fmt::Display for SylvesterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SylvesterError::EmptyMatrix => write!(f, "Matrix is empty"),
            SylvesterError::DimensionMismatch => write!(f, "Dimension mismatch"),
            SylvesterError::SingularSystem => {
                write!(f, "Singular system: eigenvalues of A and -B overlap")
            }
            SylvesterError::SchurFailed(e) => write!(f, "Schur decomposition failed: {e}"),
        }
    }
}

impl std::error::Error for SylvesterError {}

impl From<SchurError> for SylvesterError {
    fn from(e: SchurError) -> Self { SylvesterError::SchurFailed(e) }
}

/// Nalgebra Sylvester equation solver
pub mod nalgebra_sylvester {
    use super::*;

    /// Solve Sylvester equation AX + XB = C
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn solve_sylvester<T: RealField + Copy + Float>(
        matrix_a: &DMatrix<T>,
        matrix_b: &DMatrix<T>,
        matrix_c: &DMatrix<T>,
    ) -> Result<DMatrix<T>, SylvesterError> {
        let (m, n) = (matrix_a.nrows(), matrix_b.ncols());
        if matrix_a.is_empty() || matrix_b.is_empty() || matrix_c.is_empty() {
            return Err(SylvesterError::EmptyMatrix);
        }
        if matrix_a.nrows() != matrix_a.ncols() || matrix_b.nrows() != matrix_b.ncols() {
            return Err(SylvesterError::DimensionMismatch);
        }
        if matrix_a.nrows() != matrix_c.nrows() || matrix_b.ncols() != matrix_c.ncols() {
            return Err(SylvesterError::DimensionMismatch);
        }

        let schur_a = nalgebra_schur::compute_schur(matrix_a)?;
        let schur_b = nalgebra_schur::compute_schur(matrix_b)?;

        let ta = schur_a.t;
        let tb = schur_b.t;
        let qa = schur_a.q;
        let qb = schur_b.q;

        let d = qa.transpose() * matrix_c * &qb;

        let mut y = DMatrix::zeros(m, n);
        for j in 0..n {
            let mut rhs = d.column(j).clone_owned();
            for k in 0..j {
                rhs -= y.column(k) * tb[(k, j)];
            }
            let diag = ta.clone()
                + DMatrix::from_diagonal(&nalgebra::DVector::from_element(m, tb[(j, j)]));
            let inv = diag.try_inverse().ok_or(SylvesterError::SingularSystem)?;
            let col = &inv * rhs;
            y.set_column(j, &col);
        }

        Ok(&qa * &y * qb.transpose())
    }

    /// Solve Lyapunov equation AX + XA^T = Q
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn solve_lyapunov<T: RealField + Copy + Float>(
        a: &DMatrix<T>,
        q: &DMatrix<T>,
    ) -> Result<DMatrix<T>, SylvesterError> {
        let at = a.transpose();
        solve_sylvester(a, &at, q)
    }
}

/// Ndarray Sylvester equation solver
pub mod ndarray_sylvester {
    use super::*;
    use crate::interop::{nalgebra_to_ndarray, ndarray_to_nalgebra};

    /// Solve Sylvester equation AX + XB = C
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn solve_sylvester<T: Float + RealField>(
        a: &Array2<T>,
        b: &Array2<T>,
        c: &Array2<T>,
    ) -> Result<Array2<T>, SylvesterError> {
        let nalg_a = ndarray_to_nalgebra(a);
        let nalg_b = ndarray_to_nalgebra(b);
        let nalg_c = ndarray_to_nalgebra(c);
        let result = nalgebra_sylvester::solve_sylvester(&nalg_a, &nalg_b, &nalg_c)?;
        Ok(nalgebra_to_ndarray(&result))
    }

    /// Solve Lyapunov equation AX + XA^T = Q
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn solve_lyapunov<T: Float + RealField>(
        a: &Array2<T>,
        q: &Array2<T>,
    ) -> Result<Array2<T>, SylvesterError> {
        let nalg_a = ndarray_to_nalgebra(a);
        let nalg_q = ndarray_to_nalgebra(q);
        let result = nalgebra_sylvester::solve_lyapunov(&nalg_a, &nalg_q)?;
        Ok(nalgebra_to_ndarray(&result))
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;
    use crate::interop::ndarray_to_nalgebra;

    #[test]
    fn test_sylvester_simple() {
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 2.0]);
        let b = DMatrix::from_row_slice(2, 2, &[3.0, 0.0, 0.0, 4.0]);
        let c = DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 1.0, 1.0]);
        let x = nalgebra_sylvester::solve_sylvester(&a, &b, &c).unwrap();
        let ax_xb = &a * &x + &x * &b;
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(ax_xb[(i, j)], c[(i, j)], epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_lyapunov_simple() {
        let a = DMatrix::from_row_slice(2, 2, &[-1.0, 0.0, 0.0, -2.0]);
        let q = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let x = nalgebra_sylvester::solve_lyapunov(&a, &q).unwrap();
        let ax_xat = &a * &x + &x * &a.transpose();
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(ax_xat[(i, j)], q[(i, j)], epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_ndarray_sylvester_simple() {
        let a = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 2.0]).unwrap();
        let b = Array2::from_shape_vec((2, 2), vec![3.0, 0.0, 0.0, 4.0]).unwrap();
        let c = Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 1.0, 1.0]).unwrap();
        let x = ndarray_sylvester::solve_sylvester(&a, &b, &c).unwrap();

        let nalg_a = ndarray_to_nalgebra(&a);
        let nalg_b = ndarray_to_nalgebra(&b);
        let nalg_c = ndarray_to_nalgebra(&c);
        let nalg_x = ndarray_to_nalgebra(&x);
        let ax_xb = &nalg_a * &nalg_x + &nalg_x * &nalg_b;
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(ax_xb[(i, j)], nalg_c[(i, j)], epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_ndarray_lyapunov_simple() {
        let a = Array2::from_shape_vec((2, 2), vec![-1.0, 0.0, 0.0, -2.0]).unwrap();
        let q = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let x = ndarray_sylvester::solve_lyapunov(&a, &q).unwrap();

        let nalg_a = ndarray_to_nalgebra(&a);
        let nalg_q = ndarray_to_nalgebra(&q);
        let nalg_x = ndarray_to_nalgebra(&x);
        let ax_xat = &nalg_a * &nalg_x + &nalg_x * nalg_a.transpose();
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(ax_xat[(i, j)], nalg_q[(i, j)], epsilon = 1e-8);
            }
        }
    }
}
