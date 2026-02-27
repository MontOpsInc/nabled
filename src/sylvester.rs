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

use crate::schur::SchurError;

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
        crate::backend::sylvester::solve_nalgebra_sylvester(matrix_a, matrix_b, matrix_c)
    }

    /// Solve Lyapunov equation AX + XA^T = Q
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn solve_lyapunov<T: RealField + Copy + Float>(
        a: &DMatrix<T>,
        q: &DMatrix<T>,
    ) -> Result<DMatrix<T>, SylvesterError> {
        crate::backend::sylvester::solve_nalgebra_lyapunov(a, q)
    }

    /// Solve Sylvester equation with a LAPACK-backed Schur kernel.
    ///
    /// This path is available on Linux when the `lapack-kernels` feature is enabled.
    ///
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    pub fn solve_sylvester_lapack(
        matrix_a: &DMatrix<f64>,
        matrix_b: &DMatrix<f64>,
        matrix_c: &DMatrix<f64>,
    ) -> Result<DMatrix<f64>, SylvesterError> {
        crate::backend::sylvester::solve_nalgebra_lapack_sylvester(matrix_a, matrix_b, matrix_c)
    }

    /// Solve Lyapunov equation with a LAPACK-backed Schur kernel.
    ///
    /// This path is available on Linux when the `lapack-kernels` feature is enabled.
    ///
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    pub fn solve_lyapunov_lapack(
        a: &DMatrix<f64>,
        q: &DMatrix<f64>,
    ) -> Result<DMatrix<f64>, SylvesterError> {
        crate::backend::sylvester::solve_nalgebra_lapack_lyapunov(a, q)
    }
}

/// Ndarray Sylvester equation solver
pub mod ndarray_sylvester {
    use super::*;

    /// Solve Sylvester equation AX + XB = C
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn solve_sylvester<T: Float + RealField>(
        a: &Array2<T>,
        b: &Array2<T>,
        c: &Array2<T>,
    ) -> Result<Array2<T>, SylvesterError> {
        crate::backend::sylvester::solve_ndarray_sylvester(a, b, c)
    }

    /// Solve Lyapunov equation AX + XA^T = Q
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn solve_lyapunov<T: Float + RealField>(
        a: &Array2<T>,
        q: &Array2<T>,
    ) -> Result<Array2<T>, SylvesterError> {
        crate::backend::sylvester::solve_ndarray_lyapunov(a, q)
    }

    /// Solve Sylvester equation with a LAPACK-backed Schur kernel.
    ///
    /// This path is available on Linux when the `lapack-kernels` feature is enabled.
    ///
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    pub fn solve_sylvester_lapack(
        a: &Array2<f64>,
        b: &Array2<f64>,
        c: &Array2<f64>,
    ) -> Result<Array2<f64>, SylvesterError> {
        crate::backend::sylvester::solve_ndarray_lapack_sylvester(a, b, c)
    }

    /// Solve Lyapunov equation with a LAPACK-backed Schur kernel.
    ///
    /// This path is available on Linux when the `lapack-kernels` feature is enabled.
    ///
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    pub fn solve_lyapunov_lapack(
        a: &Array2<f64>,
        q: &Array2<f64>,
    ) -> Result<Array2<f64>, SylvesterError> {
        crate::backend::sylvester::solve_ndarray_lapack_lyapunov(a, q)
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

    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    #[test]
    fn test_nalgebra_sylvester_lapack_basic() {
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 2.0]);
        let b = DMatrix::from_row_slice(2, 2, &[3.0, 0.0, 0.0, 4.0]);
        let c = DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 1.0, 1.0]);
        let x = nalgebra_sylvester::solve_sylvester_lapack(&a, &b, &c).unwrap();
        assert_eq!(x.shape(), (2, 2));
    }

    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    #[test]
    fn test_ndarray_sylvester_lapack_basic() {
        let a = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 2.0]).unwrap();
        let b = Array2::from_shape_vec((2, 2), vec![3.0, 0.0, 0.0, 4.0]).unwrap();
        let c = Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 1.0, 1.0]).unwrap();
        let x = ndarray_sylvester::solve_sylvester_lapack(&a, &b, &c).unwrap();
        assert_eq!(x.dim(), (2, 2));
    }
}
