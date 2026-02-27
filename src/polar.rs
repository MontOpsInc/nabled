//! # Polar Decomposition
//!
//! Polar decomposition: `A = U P` where `U` is orthogonal and `P` is symmetric positive
//! semi-definite. Computed via SVD: `A = U_svd Σ V_svd^T => U = U_svd V_svd^T, P = V_svd Σ
//! V_svd^T`.

use std::fmt;

use nalgebra::{DMatrix, RealField};
use ndarray::Array2;
use num_traits::Float;

use crate::svd::SVDError;

/// Error types for polar decomposition
#[derive(Debug, Clone, PartialEq)]
pub enum PolarError {
    /// Matrix is empty
    EmptyMatrix,
    /// Matrix is not square
    NotSquare,
    /// SVD failed
    SVDFailed(SVDError),
    /// Numerical instability
    NumericalInstability,
}

impl fmt::Display for PolarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PolarError::EmptyMatrix => write!(f, "Matrix is empty"),
            PolarError::NotSquare => write!(f, "Matrix must be square"),
            PolarError::SVDFailed(e) => write!(f, "SVD failed: {e}"),
            PolarError::NumericalInstability => write!(f, "Numerical instability detected"),
        }
    }
}

impl std::error::Error for PolarError {}

impl From<SVDError> for PolarError {
    fn from(e: SVDError) -> Self { PolarError::SVDFailed(e) }
}

/// Polar decomposition result for nalgebra
#[derive(Debug, Clone)]
pub struct NalgebraPolarResult<T: RealField> {
    /// Orthogonal matrix U
    pub u: DMatrix<T>,
    /// Symmetric positive semi-definite matrix P
    pub p: DMatrix<T>,
}

/// Polar decomposition result for ndarray
#[derive(Debug, Clone)]
pub struct NdarrayPolarResult<T: Float> {
    /// Orthogonal matrix U
    pub u: Array2<T>,
    /// Symmetric positive semi-definite matrix P
    pub p: Array2<T>,
}

/// Nalgebra polar decomposition
pub mod nalgebra_polar {
    use super::*;

    /// Compute polar decomposition A = U P.
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn compute_polar<T: RealField + Copy + Float>(
        matrix: &DMatrix<T>,
    ) -> Result<NalgebraPolarResult<T>, PolarError> {
        crate::backend::polar::compute_nalgebra_polar(matrix)
    }

    /// Compute polar decomposition with a LAPACK-backed kernel.
    ///
    /// This path is available on Linux when the `lapack-kernels` feature is enabled.
    ///
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the LAPACK routine cannot produce a stable result.
    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    pub fn compute_polar_lapack(
        matrix: &DMatrix<f64>,
    ) -> Result<NalgebraPolarResult<f64>, PolarError> {
        crate::backend::polar::compute_nalgebra_lapack_polar(matrix)
    }
}

/// Ndarray polar decomposition
pub mod ndarray_polar {
    use super::*;

    /// Compute polar decomposition.
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn compute_polar<T: Float + RealField>(
        matrix: &Array2<T>,
    ) -> Result<NdarrayPolarResult<T>, PolarError> {
        crate::backend::polar::compute_ndarray_polar(matrix)
    }

    /// Compute polar decomposition with a LAPACK-backed kernel.
    ///
    /// This path is available on Linux when the `lapack-kernels` feature is enabled.
    ///
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the LAPACK routine cannot produce a stable result.
    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    pub fn compute_polar_lapack(
        matrix: &Array2<f64>,
    ) -> Result<NdarrayPolarResult<f64>, PolarError> {
        crate::backend::polar::compute_ndarray_lapack_polar(matrix)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn test_nalgebra_polar_reconstruct() {
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let result = nalgebra_polar::compute_polar(&a).unwrap();
        let reconstructed = &result.u * &result.p;
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(reconstructed[(i, j)], a[(i, j)], epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_nalgebra_polar_u_orthogonal() {
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let result = nalgebra_polar::compute_polar(&a).unwrap();
        let identity = &result.u * result.u.transpose();
        assert_relative_eq!(identity[(0, 0)], 1.0, epsilon = 1e-10);
        assert_relative_eq!(identity[(0, 1)], 0.0, epsilon = 1e-10);
        assert_relative_eq!(identity[(1, 0)], 0.0, epsilon = 1e-10);
        assert_relative_eq!(identity[(1, 1)], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_ndarray_polar_reconstruct_and_orthogonal() {
        let a = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let result = ndarray_polar::compute_polar(&a).unwrap();

        let reconstructed = result.u.dot(&result.p);
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(reconstructed[[i, j]], a[[i, j]], epsilon = 1e-8);
            }
        }

        let u_t = result.u.t().to_owned();
        let identity = u_t.dot(&result.u);
        assert_relative_eq!(identity[[0, 0]], 1.0, epsilon = 1e-8);
        assert_relative_eq!(identity[[0, 1]], 0.0, epsilon = 1e-8);
        assert_relative_eq!(identity[[1, 0]], 0.0, epsilon = 1e-8);
        assert_relative_eq!(identity[[1, 1]], 1.0, epsilon = 1e-8);
    }

    #[test]
    fn test_polar_error_display_variants() {
        assert!(format!("{}", PolarError::EmptyMatrix).contains("empty"));
        assert!(format!("{}", PolarError::NotSquare).contains("square"));
        assert!(format!("{}", PolarError::SVDFailed(SVDError::EmptyMatrix)).contains("SVD failed"));
        assert!(format!("{}", PolarError::NumericalInstability).contains("instability"));
    }

    #[test]
    fn test_polar_error_paths() {
        assert!(matches!(
            nalgebra_polar::compute_polar(&DMatrix::<f64>::zeros(0, 0)),
            Err(PolarError::EmptyMatrix)
        ));
        assert!(matches!(
            nalgebra_polar::compute_polar(&DMatrix::from_row_slice(1, 2, &[1.0, 2.0])),
            Err(PolarError::NotSquare)
        ));
        assert!(matches!(
            nalgebra_polar::compute_polar(&DMatrix::from_row_slice(2, 2, &[
                1.0,
                f64::NAN,
                0.0,
                1.0
            ])),
            Err(PolarError::NumericalInstability)
        ));

        let non_square_nd = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
        assert!(matches!(ndarray_polar::compute_polar(&non_square_nd), Err(PolarError::NotSquare)));

        let converted: PolarError = SVDError::EmptyMatrix.into();
        assert!(matches!(converted, PolarError::SVDFailed(SVDError::EmptyMatrix)));
    }

    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    #[test]
    fn test_nalgebra_polar_lapack_basic() {
        let matrix = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let polar = nalgebra_polar::compute_polar_lapack(&matrix).unwrap();

        assert_eq!(polar.u.shape(), (2, 2));
        assert_eq!(polar.p.shape(), (2, 2));
    }

    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    #[test]
    fn test_ndarray_polar_lapack_basic() {
        let matrix = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let polar = ndarray_polar::compute_polar_lapack(&matrix).unwrap();

        assert_eq!(polar.u.dim(), (2, 2));
        assert_eq!(polar.p.dim(), (2, 2));
    }
}
