//! # LU Decomposition
//!
//! LU decomposition with partial pivoting for solving linear systems and computing matrix inverses.

use std::fmt;

use nalgebra::{DMatrix, DVector, RealField};
use ndarray::{Array1, Array2};
use num_traits::Float;

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
            LUError::InvalidInput(msg) => write!(f, "Invalid input: {msg}"),
        }
    }
}

impl std::error::Error for LUError {}

/// Result of log-determinant computation (handles sign for general matrices)
#[derive(Debug, Clone, PartialEq)]
pub struct LogDetResult<T> {
    /// Sign of determinant: 1 (positive), -1 (negative)
    pub sign:       i8,
    /// Natural log of absolute value of determinant
    pub ln_abs_det: T,
}

/// LU decomposition result for nalgebra
#[derive(Debug, Clone)]
pub struct NalgebraLUResult<T: RealField> {
    /// Lower triangular matrix L
    pub l: DMatrix<T>,
    /// Upper triangular matrix U
    pub u: DMatrix<T>,
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

    /// Compute LU decomposition with partial pivoting
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn compute_lu<T: RealField + Copy + Float>(
        matrix: &DMatrix<T>,
    ) -> Result<NalgebraLUResult<T>, LUError> {
        crate::backend::lu::compute_nalgebra_lu(matrix)
    }

    /// Compute LU decomposition with a LAPACK-backed kernel.
    ///
    /// This path is available on Linux when the `lapack-kernels` feature is enabled.
    ///
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the LAPACK routine cannot produce a stable result.
    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    pub fn compute_lu_lapack(matrix: &DMatrix<f64>) -> Result<NalgebraLUResult<f64>, LUError> {
        crate::backend::lu::compute_nalgebra_lapack_lu(matrix)
    }

    /// Solve Ax = b for square matrix A
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn solve<T: RealField + Copy + Float>(
        matrix: &DMatrix<T>,
        rhs: &DVector<T>,
    ) -> Result<DVector<T>, LUError> {
        crate::backend::lu::solve_nalgebra_lu(matrix, rhs)
    }

    /// Solve Ax = b with a LAPACK-backed LU kernel.
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
    ) -> Result<DVector<f64>, LUError> {
        crate::backend::lu::solve_nalgebra_lapack_lu(matrix, rhs)
    }

    /// Compute matrix inverse
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn inverse<T: RealField + Copy + Float>(
        matrix: &DMatrix<T>,
    ) -> Result<DMatrix<T>, LUError> {
        crate::backend::lu::inverse_nalgebra_lu(matrix)
    }

    /// Compute matrix inverse with a LAPACK-backed LU kernel.
    ///
    /// This path is available on Linux when the `lapack-kernels` feature is enabled.
    ///
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the LAPACK routine cannot produce a stable result.
    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    pub fn inverse_lapack(matrix: &DMatrix<f64>) -> Result<DMatrix<f64>, LUError> {
        crate::backend::lu::inverse_nalgebra_lapack_lu(matrix)
    }

    /// Compute the determinant (det = sign(permutation) * prod(diag(U)))
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn determinant<T: RealField + Copy + Float>(matrix: &DMatrix<T>) -> Result<T, LUError> {
        crate::backend::lu::determinant_nalgebra_lu(matrix)
    }

    /// Compute determinant with a LAPACK-backed LU kernel.
    ///
    /// This path is available on Linux when the `lapack-kernels` feature is enabled.
    ///
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the LAPACK routine cannot produce a stable result.
    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    pub fn determinant_lapack(matrix: &DMatrix<f64>) -> Result<f64, LUError> {
        crate::backend::lu::determinant_nalgebra_lapack_lu(matrix)
    }

    /// Compute log-determinant for general matrices (handles sign)
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn log_determinant<T: RealField + Copy + Float>(
        matrix: &DMatrix<T>,
    ) -> Result<LogDetResult<T>, LUError> {
        crate::backend::lu::log_determinant_nalgebra_lu(matrix)
    }

    /// Compute log-determinant with a LAPACK-backed LU kernel.
    ///
    /// This path is available on Linux when the `lapack-kernels` feature is enabled.
    ///
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the LAPACK routine cannot produce a stable result.
    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    pub fn log_determinant_lapack(matrix: &DMatrix<f64>) -> Result<LogDetResult<f64>, LUError> {
        crate::backend::lu::log_determinant_nalgebra_lapack_lu(matrix)
    }
}

/// Ndarray LU decomposition (via nalgebra)
pub mod ndarray_lu {
    use super::*;

    /// Compute LU decomposition
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn compute_lu<T: Float + RealField>(
        matrix: &Array2<T>,
    ) -> Result<NdarrayLUResult<T>, LUError> {
        crate::backend::lu::compute_ndarray_lu(matrix)
    }

    /// Compute LU decomposition with a LAPACK-backed kernel.
    ///
    /// This path is available on Linux when the `lapack-kernels` feature is enabled.
    ///
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the LAPACK routine cannot produce a stable result.
    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    pub fn compute_lu_lapack(matrix: &Array2<f64>) -> Result<NdarrayLUResult<f64>, LUError> {
        crate::backend::lu::compute_ndarray_lapack_lu(matrix)
    }

    /// Solve Ax = b
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn solve<T: Float + RealField>(
        matrix: &Array2<T>,
        rhs: &Array1<T>,
    ) -> Result<Array1<T>, LUError> {
        crate::backend::lu::solve_ndarray_lu(matrix, rhs)
    }

    /// Solve Ax = b with a LAPACK-backed LU kernel.
    ///
    /// This path is available on Linux when the `lapack-kernels` feature is enabled.
    ///
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the LAPACK routine cannot produce a stable result.
    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    pub fn solve_lapack(matrix: &Array2<f64>, rhs: &Array1<f64>) -> Result<Array1<f64>, LUError> {
        crate::backend::lu::solve_ndarray_lapack_lu(matrix, rhs)
    }

    /// Compute matrix inverse
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn inverse<T: Float + RealField>(matrix: &Array2<T>) -> Result<Array2<T>, LUError> {
        crate::backend::lu::inverse_ndarray_lu(matrix)
    }

    /// Compute matrix inverse with a LAPACK-backed LU kernel.
    ///
    /// This path is available on Linux when the `lapack-kernels` feature is enabled.
    ///
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the LAPACK routine cannot produce a stable result.
    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    pub fn inverse_lapack(matrix: &Array2<f64>) -> Result<Array2<f64>, LUError> {
        crate::backend::lu::inverse_ndarray_lapack_lu(matrix)
    }

    /// Compute the determinant
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn determinant<T: Float + RealField>(matrix: &Array2<T>) -> Result<T, LUError> {
        crate::backend::lu::determinant_ndarray_lu(matrix)
    }

    /// Compute determinant with a LAPACK-backed LU kernel.
    ///
    /// This path is available on Linux when the `lapack-kernels` feature is enabled.
    ///
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the LAPACK routine cannot produce a stable result.
    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    pub fn determinant_lapack(matrix: &Array2<f64>) -> Result<f64, LUError> {
        crate::backend::lu::determinant_ndarray_lapack_lu(matrix)
    }

    /// Compute log-determinant (handles sign for general matrices)
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn log_determinant<T: Float + RealField>(
        matrix: &Array2<T>,
    ) -> Result<LogDetResult<T>, LUError> {
        crate::backend::lu::log_determinant_ndarray_lu(matrix)
    }

    /// Compute log-determinant with a LAPACK-backed LU kernel.
    ///
    /// This path is available on Linux when the `lapack-kernels` feature is enabled.
    ///
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the LAPACK routine cannot produce a stable result.
    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    pub fn log_determinant_lapack(matrix: &Array2<f64>) -> Result<LogDetResult<f64>, LUError> {
        crate::backend::lu::log_determinant_ndarray_lapack_lu(matrix)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

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

    #[test]
    fn test_ndarray_lu_inverse() {
        let a = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let inv = ndarray_lu::inverse(&a).unwrap();
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
        assert_relative_eq!(identity[[0, 1]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(identity[[1, 0]], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_nalgebra_determinant() {
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let det = nalgebra_lu::determinant(&a).unwrap();
        assert_relative_eq!(det, -2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_nalgebra_log_determinant() {
        let a = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 3.0]);
        let log_det = nalgebra_lu::log_determinant(&a).unwrap();
        assert_eq!(log_det.sign, 1);
        assert_relative_eq!(log_det.ln_abs_det, 6.0_f64.ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_ndarray_determinant() {
        let a = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let det = ndarray_lu::determinant(&a).unwrap();
        assert_relative_eq!(det, -2.0, epsilon = 1e-10);
    }

    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    #[test]
    fn test_nalgebra_lu_lapack_basic() {
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let lu = nalgebra_lu::compute_lu_lapack(&a).unwrap();
        let reconstructed = &lu.l * &lu.u;
        assert_eq!(reconstructed.nrows(), 2);
        assert_eq!(reconstructed.ncols(), 2);
    }

    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    #[test]
    fn test_ndarray_lu_lapack_basic() {
        let a = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let lu = ndarray_lu::compute_lu_lapack(&a).unwrap();
        assert_eq!(lu.l.dim(), (2, 2));
        assert_eq!(lu.u.dim(), (2, 2));
    }

    #[test]
    fn test_lu_error_display_variants() {
        assert!(format!("{}", LUError::EmptyMatrix).contains("empty"));
        assert!(format!("{}", LUError::NotSquare).contains("square"));
        assert!(format!("{}", LUError::SingularMatrix).contains("singular"));
        assert!(format!("{}", LUError::NumericalInstability).contains("instability"));
        assert!(format!("{}", LUError::InvalidInput("x".to_string())).contains('x'));
    }

    #[test]
    fn test_lu_error_paths() {
        let empty = DMatrix::<f64>::zeros(0, 0);
        assert!(matches!(nalgebra_lu::compute_lu(&empty), Err(LUError::EmptyMatrix)));

        let non_square = DMatrix::from_row_slice(1, 2, &[1.0, 2.0]);
        assert!(matches!(nalgebra_lu::compute_lu(&non_square), Err(LUError::NotSquare)));
        assert!(matches!(nalgebra_lu::determinant(&non_square), Err(LUError::NotSquare)));

        let non_finite = DMatrix::from_row_slice(2, 2, &[1.0, f64::NAN, 0.0, 1.0]);
        assert!(matches!(nalgebra_lu::compute_lu(&non_finite), Err(LUError::NumericalInstability)));
        assert!(matches!(
            nalgebra_lu::determinant(&non_finite),
            Err(LUError::NumericalInstability)
        ));

        let singular = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 4.0]);
        let rhs = DVector::from_vec(vec![1.0, 2.0]);
        assert!(matches!(nalgebra_lu::solve(&singular, &rhs), Err(LUError::SingularMatrix)));
        assert!(matches!(nalgebra_lu::inverse(&singular), Err(LUError::SingularMatrix)));
        assert!(matches!(nalgebra_lu::log_determinant(&singular), Err(LUError::SingularMatrix)));

        let rhs_bad = DVector::from_vec(vec![1.0]);
        let a = DMatrix::identity(2, 2);
        assert!(matches!(nalgebra_lu::solve(&a, &rhs_bad), Err(LUError::InvalidInput(_))));

        let singular_nd = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 2.0, 4.0]).unwrap();
        assert!(matches!(ndarray_lu::log_determinant(&singular_nd), Err(LUError::SingularMatrix)));
    }
}
