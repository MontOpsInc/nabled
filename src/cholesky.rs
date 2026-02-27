//! # Cholesky Decomposition
//!
//! Cholesky decomposition for symmetric positive-definite matrices.
//! Used for solving linear systems and computing matrix inverses.

use std::fmt;

use nalgebra::{DMatrix, DVector, RealField};
use ndarray::{Array1, Array2};
use num_traits::Float;

/// Error types for Cholesky decomposition
#[derive(Debug, Clone, PartialEq)]
pub enum CholeskyError {
    /// Matrix is empty
    EmptyMatrix,
    /// Matrix is not square
    NotSquare,
    /// Matrix is not positive-definite
    NotPositiveDefinite,
    /// Numerical instability detected
    NumericalInstability,
    /// Invalid input
    InvalidInput(String),
}

impl fmt::Display for CholeskyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CholeskyError::EmptyMatrix => write!(f, "Matrix is empty"),
            CholeskyError::NotSquare => write!(f, "Matrix must be square"),
            CholeskyError::NotPositiveDefinite => write!(f, "Matrix is not positive-definite"),
            CholeskyError::NumericalInstability => write!(f, "Numerical instability detected"),
            CholeskyError::InvalidInput(msg) => write!(f, "Invalid input: {msg}"),
        }
    }
}

impl std::error::Error for CholeskyError {}

/// Cholesky decomposition result for nalgebra
#[derive(Debug, Clone)]
pub struct NalgebraCholeskyResult<T: RealField> {
    /// Lower triangular factor L
    pub l: DMatrix<T>,
}

/// Cholesky decomposition result for ndarray
#[derive(Debug, Clone)]
pub struct NdarrayCholeskyResult<T: Float> {
    /// Lower triangular factor L
    pub l: Array2<T>,
}

/// Nalgebra Cholesky decomposition
pub mod nalgebra_cholesky {
    use super::*;

    /// Compute Cholesky decomposition.
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn compute_cholesky<T: RealField + Copy + Float>(
        matrix: &DMatrix<T>,
    ) -> Result<NalgebraCholeskyResult<T>, CholeskyError> {
        crate::backend::cholesky::compute_nalgebra_cholesky(matrix)
    }

    /// Compute Cholesky decomposition with a LAPACK-backed kernel.
    ///
    /// This path is available on Linux when the `lapack-kernels` feature is enabled.
    ///
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the LAPACK routine cannot produce a stable result.
    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    pub fn compute_cholesky_lapack(
        matrix: &DMatrix<f64>,
    ) -> Result<NalgebraCholeskyResult<f64>, CholeskyError> {
        crate::backend::cholesky::compute_nalgebra_lapack_cholesky(matrix)
    }

    /// Solve Ax = b for symmetric positive-definite A.
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn solve<T: RealField + Copy + Float>(
        matrix: &DMatrix<T>,
        rhs: &DVector<T>,
    ) -> Result<DVector<T>, CholeskyError> {
        crate::backend::cholesky::solve_nalgebra_cholesky(matrix, rhs)
    }

    /// Solve Ax = b with a LAPACK-backed Cholesky kernel.
    ///
    /// This path is available on Linux when the `lapack-kernels` feature is enabled.
    ///
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the LAPACK routine cannot produce a stable result.
    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    pub fn solve_lapack(
        matrix: &DMatrix<f64>,
        rhs: &DVector<f64>,
    ) -> Result<DVector<f64>, CholeskyError> {
        crate::backend::cholesky::solve_nalgebra_lapack_cholesky(matrix, rhs)
    }

    /// Compute matrix inverse.
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn inverse<T: RealField + Copy + Float>(
        matrix: &DMatrix<T>,
    ) -> Result<DMatrix<T>, CholeskyError> {
        crate::backend::cholesky::inverse_nalgebra_cholesky(matrix)
    }

    /// Compute matrix inverse with a LAPACK-backed Cholesky kernel.
    ///
    /// This path is available on Linux when the `lapack-kernels` feature is enabled.
    ///
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the LAPACK routine cannot produce a stable result.
    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    pub fn inverse_lapack(matrix: &DMatrix<f64>) -> Result<DMatrix<f64>, CholeskyError> {
        crate::backend::cholesky::inverse_nalgebra_lapack_cholesky(matrix)
    }
}

/// Ndarray Cholesky decomposition
pub mod ndarray_cholesky {
    use super::*;

    /// Compute Cholesky decomposition.
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn compute_cholesky<T: Float + RealField>(
        matrix: &Array2<T>,
    ) -> Result<NdarrayCholeskyResult<T>, CholeskyError> {
        crate::backend::cholesky::compute_ndarray_cholesky(matrix)
    }

    /// Compute Cholesky decomposition with a LAPACK-backed kernel.
    ///
    /// This path is available on Linux when the `lapack-kernels` feature is enabled.
    ///
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the LAPACK routine cannot produce a stable result.
    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    pub fn compute_cholesky_lapack(
        matrix: &Array2<f64>,
    ) -> Result<NdarrayCholeskyResult<f64>, CholeskyError> {
        crate::backend::cholesky::compute_ndarray_lapack_cholesky(matrix)
    }

    /// Solve Ax = b.
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn solve<T: Float + RealField>(
        matrix: &Array2<T>,
        rhs: &Array1<T>,
    ) -> Result<Array1<T>, CholeskyError> {
        crate::backend::cholesky::solve_ndarray_cholesky(matrix, rhs)
    }

    /// Solve Ax = b with a LAPACK-backed Cholesky kernel.
    ///
    /// This path is available on Linux when the `lapack-kernels` feature is enabled.
    ///
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the LAPACK routine cannot produce a stable result.
    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    pub fn solve_lapack(
        matrix: &Array2<f64>,
        rhs: &Array1<f64>,
    ) -> Result<Array1<f64>, CholeskyError> {
        crate::backend::cholesky::solve_ndarray_lapack_cholesky(matrix, rhs)
    }

    /// Compute matrix inverse.
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn inverse<T: Float + RealField>(matrix: &Array2<T>) -> Result<Array2<T>, CholeskyError> {
        crate::backend::cholesky::inverse_ndarray_cholesky(matrix)
    }

    /// Compute matrix inverse with a LAPACK-backed Cholesky kernel.
    ///
    /// This path is available on Linux when the `lapack-kernels` feature is enabled.
    ///
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the LAPACK routine cannot produce a stable result.
    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    pub fn inverse_lapack(matrix: &Array2<f64>) -> Result<Array2<f64>, CholeskyError> {
        crate::backend::cholesky::inverse_ndarray_lapack_cholesky(matrix)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn test_nalgebra_cholesky_solve() {
        // Create symmetric positive-definite matrix: A = L * L^T
        let l = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 1.0, 3.0]);
        let a = &l * l.transpose();
        let b = DVector::from_vec(vec![1.0, 2.0]);
        let x = nalgebra_cholesky::solve(&a, &b).unwrap();
        let ax = &a * &x;
        for i in 0..2 {
            assert_relative_eq!(ax[i], b[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_nalgebra_cholesky_inverse() {
        let l = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 1.0, 3.0]);
        let a = &l * l.transpose();
        let inv = nalgebra_cholesky::inverse(&a).unwrap();
        let identity = &a * &inv;
        assert_relative_eq!(identity[(0, 0)], 1.0, epsilon = 1e-10);
        assert_relative_eq!(identity[(0, 1)], 0.0, epsilon = 1e-10);
        assert_relative_eq!(identity[(1, 0)], 0.0, epsilon = 1e-10);
        assert_relative_eq!(identity[(1, 1)], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_ndarray_cholesky_solve() {
        let l = Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 1.0, 3.0]).unwrap();
        let mut a: Array2<f64> = Array2::zeros((2, 2));
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    a[[i, j]] += l[[i, k]] * l[[j, k]];
                }
            }
        }
        let b = Array1::from_vec(vec![1.0, 2.0]);
        let x = ndarray_cholesky::solve(&a, &b).unwrap();
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

    #[test]
    fn test_ndarray_cholesky_inverse() {
        let l = Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 1.0, 3.0]).unwrap();
        let mut a: Array2<f64> = Array2::zeros((2, 2));
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    a[[i, j]] += l[[i, k]] * l[[j, k]];
                }
            }
        }
        let inv = ndarray_cholesky::inverse(&a).unwrap();
        let mut identity: Array2<f64> = Array2::zeros((2, 2));
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    identity[[i, j]] += a[[i, k]] * inv[[k, j]];
                }
            }
        }
        assert_relative_eq!(identity[[0, 0]], 1.0, epsilon = 1e-10);
        assert_relative_eq!(identity[[1, 1]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cholesky_error_display_variants() {
        assert!(format!("{}", CholeskyError::EmptyMatrix).contains("empty"));
        assert!(format!("{}", CholeskyError::NotSquare).contains("square"));
        assert!(format!("{}", CholeskyError::NotPositiveDefinite).contains("positive-definite"));
        assert!(format!("{}", CholeskyError::NumericalInstability).contains("instability"));
        assert!(format!("{}", CholeskyError::InvalidInput("x".to_string())).contains('x'));
    }

    #[test]
    fn test_cholesky_error_paths() {
        let empty = DMatrix::<f64>::zeros(0, 0);
        assert!(matches!(
            nalgebra_cholesky::compute_cholesky(&empty),
            Err(CholeskyError::EmptyMatrix)
        ));

        let non_square = DMatrix::from_row_slice(1, 2, &[1.0, 2.0]);
        assert!(matches!(
            nalgebra_cholesky::compute_cholesky(&non_square),
            Err(CholeskyError::NotSquare)
        ));
        let non_square_nd = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
        assert!(matches!(
            ndarray_cholesky::compute_cholesky(&non_square_nd),
            Err(CholeskyError::NotSquare)
        ));

        let non_finite = DMatrix::from_row_slice(2, 2, &[1.0, f64::NAN, 0.0, 1.0]);
        assert!(matches!(
            nalgebra_cholesky::compute_cholesky(&non_finite),
            Err(CholeskyError::NumericalInstability)
        ));

        let non_pd = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 1.0]);
        assert!(matches!(
            nalgebra_cholesky::compute_cholesky(&non_pd),
            Err(CholeskyError::NotPositiveDefinite)
        ));
        assert!(matches!(
            nalgebra_cholesky::inverse(&non_pd),
            Err(CholeskyError::NotPositiveDefinite)
        ));

        let a = DMatrix::identity(2, 2);
        let b_bad = DVector::from_vec(vec![1.0]);
        assert!(matches!(
            nalgebra_cholesky::solve(&a, &b_bad),
            Err(CholeskyError::InvalidInput(_))
        ));
    }

    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    #[test]
    fn test_nalgebra_cholesky_lapack_basic() {
        let l = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 1.0, 3.0]);
        let a = &l * l.transpose();

        let cholesky = nalgebra_cholesky::compute_cholesky_lapack(&a).unwrap();
        assert_eq!(cholesky.l.shape(), (2, 2));
    }

    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    #[test]
    fn test_ndarray_cholesky_lapack_basic() {
        let l = Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 1.0, 3.0]).unwrap();
        let mut a: Array2<f64> = Array2::zeros((2, 2));
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    a[[i, j]] += l[[i, k]] * l[[j, k]];
                }
            }
        }

        let cholesky = ndarray_cholesky::compute_cholesky_lapack(&a).unwrap();
        assert_eq!(cholesky.l.dim(), (2, 2));
    }
}
