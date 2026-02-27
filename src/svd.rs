//! # Singular Value Decomposition (SVD)
//!
//! This module provides SVD implementations using both nalgebra and ndarray.
//! It includes enhanced algorithms and utilities for SVD computation.

use std::fmt;

use nalgebra::{DMatrix, DVector, RealField};
use ndarray::{Array1, Array2, s};
use num_traits::Float;
use num_traits::float::FloatCore;

/// SVD result structure for nalgebra
#[derive(Debug, Clone)]
pub struct NalgebraSVD<T: RealField> {
    /// Left singular vectors (U matrix)
    pub u:               DMatrix<T>,
    /// Singular values
    pub singular_values: DVector<T>,
    /// Right singular vectors (V^T matrix)
    pub vt:              DMatrix<T>,
}

/// SVD result structure for ndarray
#[derive(Debug, Clone)]
pub struct NdarraySVD<T: Float> {
    /// Left singular vectors (U matrix)
    pub u:               Array2<T>,
    /// Singular values
    pub singular_values: Array1<T>,
    /// Right singular vectors (V^T matrix)
    pub vt:              Array2<T>,
}

/// Error types for SVD computation
#[derive(Debug, Clone, PartialEq)]
pub enum SVDError {
    /// Matrix is not square when required
    NotSquare,
    /// Matrix is empty
    EmptyMatrix,
    /// Convergence failed in iterative algorithm
    ConvergenceFailed,
    /// Invalid input parameters
    InvalidInput(String),
}

impl fmt::Display for SVDError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SVDError::NotSquare => write!(f, "Matrix must be square"),
            SVDError::EmptyMatrix => write!(f, "Matrix cannot be empty"),
            SVDError::ConvergenceFailed => write!(f, "SVD algorithm failed to converge"),
            SVDError::InvalidInput(msg) => write!(f, "Invalid input: {msg}"),
        }
    }
}

impl std::error::Error for SVDError {}

/// Configuration for pseudo-inverse computation
#[derive(Debug, Clone, Default)]
pub struct PseudoInverseConfig<T> {
    /// Tolerance for truncating tiny singular values (default: eps * max(s) * max(m,n))
    pub tolerance: Option<T>,
}

/// Enhanced SVD computation using nalgebra
pub mod nalgebra_svd {
    use super::*;
    // Note: ComplexField and SymmetricEigen imports removed as they are not used

    /// Compute SVD using nalgebra's built-in implementation
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn compute_svd<T: RealField>(matrix: &DMatrix<T>) -> Result<NalgebraSVD<T>, SVDError> {
        crate::backend::svd::compute_nalgebra_svd(matrix)
    }

    /// Compute SVD using a LAPACK-backed nalgebra kernel.
    ///
    /// This path is available on Linux when the `lapack-kernels` feature is enabled.
    ///
    /// # Errors
    /// Returns an error when the matrix is empty or when the LAPACK routine fails.
    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    pub fn compute_svd_lapack(matrix: &DMatrix<f64>) -> Result<NalgebraSVD<f64>, SVDError> {
        crate::backend::svd::compute_nalgebra_lapack_svd(matrix)
    }

    /// Compute SVD with custom tolerance.
    /// Singular values below the tolerance are set to zero (useful for rank determination and
    /// low-rank approximation).
    #[allow(clippy::needless_pass_by_value)]
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn compute_svd_with_tolerance<T: RealField>(
        matrix: &DMatrix<T>,
        tolerance: T,
    ) -> Result<NalgebraSVD<T>, SVDError> {
        if matrix.is_empty() {
            return Err(SVDError::EmptyMatrix);
        }

        let mut svd = compute_svd(matrix)?;
        let sv_values: Vec<T> = svd
            .singular_values
            .iter()
            .map(|sv| if *sv < tolerance { T::zero() } else { sv.clone() })
            .collect();
        svd.singular_values = DVector::from_vec(sv_values);
        Ok(svd)
    }

    /// Compute truncated SVD (keeping only the k largest singular values)
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn compute_truncated_svd<T: RealField>(
        matrix: &DMatrix<T>,
        k: usize,
    ) -> Result<NalgebraSVD<T>, SVDError> {
        let full_svd = compute_svd(matrix)?;

        let _m = full_svd.u.nrows();
        let _n = full_svd.vt.ncols();
        let k = k.min(full_svd.singular_values.len());

        if k == 0 {
            return Err(SVDError::InvalidInput("k must be greater than 0".to_string()));
        }

        // Truncate the matrices
        let u_truncated = full_svd.u.columns(0, k).into_owned();
        let singular_values_truncated = full_svd.singular_values.rows(0, k).into_owned();
        let vt_truncated = full_svd.vt.rows(0, k).into_owned();

        Ok(NalgebraSVD {
            u:               u_truncated,
            singular_values: singular_values_truncated,
            vt:              vt_truncated,
        })
    }

    /// Reconstruct the original matrix from SVD components
    #[must_use]
    pub fn reconstruct_matrix<T: RealField>(svd: &NalgebraSVD<T>) -> DMatrix<T> {
        let sigma = DMatrix::from_diagonal(&svd.singular_values);
        &svd.u * &sigma * &svd.vt
    }

    /// Compute the condition number from singular values
    #[must_use]
    pub fn condition_number<T: RealField + FloatCore>(svd: &NalgebraSVD<T>) -> T {
        let singular_values = &svd.singular_values;
        if singular_values.is_empty() {
            return T::zero();
        }

        let max_sv = singular_values.max();
        let min_sv = singular_values.min();

        if min_sv.is_zero() { T::infinity() } else { max_sv / min_sv }
    }

    /// Compute the rank of the matrix from its SVD
    /// # Panics
    /// Panics if internal numeric assumptions are violated during setup or
    /// intermediate conversion steps.
    pub fn matrix_rank<T: RealField>(svd: &NalgebraSVD<T>, tolerance: Option<T>) -> usize {
        let tol = tolerance.unwrap_or_else(|| {
            let max_sv = svd.singular_values.max();
            T::from_f64(1e-10).unwrap() * max_sv
        });

        svd.singular_values.iter().filter(|&sv| *sv > tol).count()
    }

    /// Compute Moore-Penrose pseudo-inverse pinv(A) = V Σ⁻¹ U^T
    /// # Panics
    /// Panics if internal numeric assumptions are violated during setup or
    /// intermediate conversion steps.
    ///
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn pseudo_inverse<T: RealField + Copy + Float>(
        matrix: &DMatrix<T>,
        config: &PseudoInverseConfig<T>,
    ) -> Result<DMatrix<T>, SVDError> {
        if matrix.is_empty() {
            return Err(SVDError::EmptyMatrix);
        }
        let svd = compute_svd(matrix)?;
        let (m, n) = matrix.shape();
        let max_dim = T::from(m.max(n)).unwrap_or_else(|| T::from_f64(f64::NAN).unwrap());
        let max_sv = svd.singular_values.max();
        let tol = config.tolerance.unwrap_or_else(|| max_sv * max_dim * T::epsilon());

        let r = svd.singular_values.len();
        let ncols_v = svd.vt.ncols();
        let nrows_u = svd.u.nrows();

        let mut result = DMatrix::zeros(ncols_v, nrows_u);
        for i in 0..r {
            let s = svd.singular_values[i];
            if s > tol {
                let inv_s = T::one() / s;
                for j in 0..ncols_v {
                    for k in 0..nrows_u {
                        result[(j, k)] += svd.vt[(i, j)] * inv_s * svd.u[(k, i)];
                    }
                }
            }
        }
        Ok(result)
    }

    /// Compute null space (kernel) - columns of V for singular values below tolerance
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn null_space<T: RealField + Copy + Float>(
        matrix: &DMatrix<T>,
        tolerance: Option<T>,
    ) -> Result<DMatrix<T>, SVDError> {
        if matrix.is_empty() {
            return Err(SVDError::EmptyMatrix);
        }
        let svd = compute_svd(matrix)?;
        let max_sv = svd.singular_values.max();
        let tol = tolerance.unwrap_or_else(|| T::from_f64(1e-10).unwrap_or_else(T::nan) * max_sv);

        let null_indices: Vec<usize> = svd
            .singular_values
            .iter()
            .enumerate()
            .filter(|(_, s)| **s <= tol)
            .map(|(i, _)| i)
            .collect();

        if null_indices.is_empty() {
            return Ok(DMatrix::zeros(svd.vt.ncols(), 0));
        }

        let k = null_indices.len();
        let n = svd.vt.ncols();
        let mut result = DMatrix::zeros(n, k);
        for (col_j, &idx) in null_indices.iter().enumerate() {
            for i in 0..n {
                result[(i, col_j)] = svd.vt[(idx, i)];
            }
        }
        Ok(result)
    }
}

/// Enhanced SVD computation using ndarray
pub mod ndarray_svd {
    use super::*;
    use crate::interop::{nalgebra_to_ndarray, ndarray_to_nalgebra};
    // Note: general_mat_mul import removed as it's not used

    /// Compute SVD using conversion to nalgebra and back
    /// Note: This is a simplified implementation that converts to nalgebra for computation
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn compute_svd<T: Float + RealField>(
        matrix: &Array2<T>,
    ) -> Result<NdarraySVD<T>, SVDError> {
        crate::backend::svd::compute_ndarray_svd(matrix)
    }

    /// Compute SVD using a LAPACK-backed ndarray kernel.
    ///
    /// This path is available on Linux when the `lapack-kernels` feature is enabled.
    ///
    /// # Errors
    /// Returns an error when the matrix is empty or when the LAPACK routine fails.
    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    pub fn compute_svd_lapack(matrix: &Array2<f64>) -> Result<NdarraySVD<f64>, SVDError> {
        crate::backend::svd::compute_ndarray_lapack_svd(matrix)
    }

    /// Compute SVD with custom tolerance.
    /// Singular values below the tolerance are set to zero.
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn compute_svd_with_tolerance<T: Float + RealField>(
        matrix: &Array2<T>,
        tolerance: T,
    ) -> Result<NdarraySVD<T>, SVDError> {
        if matrix.is_empty() {
            return Err(SVDError::EmptyMatrix);
        }

        let nalgebra_matrix = ndarray_to_nalgebra(matrix);
        let nalgebra_svd = nalgebra_svd::compute_svd_with_tolerance(&nalgebra_matrix, tolerance)?;

        let u = nalgebra_to_ndarray(&nalgebra_svd.u);
        let vt = nalgebra_to_ndarray(&nalgebra_svd.vt);
        let singular_values = Array1::from_vec(nalgebra_svd.singular_values.as_slice().to_vec());

        Ok(NdarraySVD { u, singular_values, vt })
    }

    /// Compute truncated SVD (keeping only the k largest singular values)
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn compute_truncated_svd<T: Float + RealField>(
        matrix: &Array2<T>,
        k: usize,
    ) -> Result<NdarraySVD<T>, SVDError> {
        let full_svd = compute_svd(matrix)?;

        let k = k.min(full_svd.singular_values.len());

        if k == 0 {
            return Err(SVDError::InvalidInput("k must be greater than 0".to_string()));
        }

        // Truncate the matrices
        let u_truncated = full_svd.u.slice(s![.., ..k]).to_owned();
        let singular_values_truncated = full_svd.singular_values.slice(s![..k]).to_owned();
        let vt_truncated = full_svd.vt.slice(s![..k, ..]).to_owned();

        Ok(NdarraySVD {
            u:               u_truncated,
            singular_values: singular_values_truncated,
            vt:              vt_truncated,
        })
    }

    /// Reconstruct the original matrix from SVD components
    #[must_use]
    pub fn reconstruct_matrix<T: Float>(svd: &NdarraySVD<T>) -> Array2<T> {
        let mut result = Array2::zeros((svd.u.nrows(), svd.vt.ncols()));

        for i in 0..svd.singular_values.len() {
            let sigma = svd.singular_values[i];
            let u_col = svd.u.column(i);
            let vt_row = svd.vt.row(i);

            // Add sigma * u_col * vt_row to result
            for (j, &u_val) in u_col.iter().enumerate() {
                for (k, &vt_val) in vt_row.iter().enumerate() {
                    result[[j, k]] = result[[j, k]] + sigma * u_val * vt_val;
                }
            }
        }

        result
    }

    /// Compute the condition number from singular values
    #[must_use]
    pub fn condition_number<T: Float>(svd: &NdarraySVD<T>) -> T {
        if svd.singular_values.is_empty() {
            return T::zero();
        }

        let max_sv = svd.singular_values.iter().fold(T::zero(), |a, &b| a.max(b));
        let min_sv = svd.singular_values.iter().fold(T::infinity(), |a, &b| a.min(b));

        if min_sv.is_zero() { T::infinity() } else { max_sv / min_sv }
    }

    /// Compute the rank of the matrix from its SVD
    /// # Panics
    /// Panics if internal numeric assumptions are violated during setup or
    /// intermediate conversion steps.
    pub fn matrix_rank<T: Float>(svd: &NdarraySVD<T>, tolerance: Option<T>) -> usize {
        let tol = tolerance.unwrap_or_else(|| {
            let max_sv = svd.singular_values.iter().fold(T::zero(), |a, &b| a.max(b));
            T::from(1e-10).unwrap() * max_sv
        });

        svd.singular_values.iter().filter(|&sv| *sv > tol).count()
    }

    /// Compute Moore-Penrose pseudo-inverse
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn pseudo_inverse<T: Float + RealField>(
        matrix: &Array2<T>,
        config: &PseudoInverseConfig<T>,
    ) -> Result<Array2<T>, SVDError> {
        let nalg = ndarray_to_nalgebra(matrix);
        let pinv = nalgebra_svd::pseudo_inverse(&nalg, config)?;
        Ok(nalgebra_to_ndarray(&pinv))
    }

    /// Compute null space (kernel)
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn null_space<T: Float + RealField>(
        matrix: &Array2<T>,
        tolerance: Option<T>,
    ) -> Result<Array2<T>, SVDError> {
        let nalg = ndarray_to_nalgebra(matrix);
        let nulls = nalgebra_svd::null_space(&nalg, tolerance)?;
        Ok(nalgebra_to_ndarray(&nulls))
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{DMatrix, DVector};
    use ndarray::{Array1, Array2};

    use super::*;

    #[test]
    fn test_nalgebra_svd_basic() {
        let matrix = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let svd = nalgebra_svd::compute_svd(&matrix).unwrap();

        // Reconstruct and check
        let reconstructed = nalgebra_svd::reconstruct_matrix(&svd);
        for i in 0..matrix.nrows() {
            for j in 0..matrix.ncols() {
                assert!(approx::relative_eq!(
                    matrix[(i, j)],
                    reconstructed[(i, j)],
                    epsilon = 1e-10
                ));
            }
        }
    }

    #[test]
    fn test_ndarray_svd_basic() {
        let matrix = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let svd = ndarray_svd::compute_svd(&matrix).unwrap();

        // Reconstruct and check
        let reconstructed = ndarray_svd::reconstruct_matrix(&svd);
        for i in 0..matrix.nrows() {
            for j in 0..matrix.ncols() {
                assert!(approx::relative_eq!(
                    matrix[[i, j]],
                    reconstructed[[i, j]],
                    epsilon = 1e-10
                ));
            }
        }
    }

    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    #[test]
    fn test_nalgebra_svd_lapack_basic() {
        let matrix = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let svd = nalgebra_svd::compute_svd_lapack(&matrix).unwrap();
        let reconstructed = nalgebra_svd::reconstruct_matrix(&svd);

        for i in 0..matrix.nrows() {
            for j in 0..matrix.ncols() {
                assert!(approx::relative_eq!(
                    matrix[(i, j)],
                    reconstructed[(i, j)],
                    epsilon = 1e-10
                ));
            }
        }
    }

    #[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
    #[test]
    fn test_ndarray_svd_lapack_basic() {
        let matrix = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let svd = ndarray_svd::compute_svd_lapack(&matrix).unwrap();
        let reconstructed = ndarray_svd::reconstruct_matrix(&svd);

        for i in 0..matrix.nrows() {
            for j in 0..matrix.ncols() {
                assert!(approx::relative_eq!(
                    matrix[[i, j]],
                    reconstructed[[i, j]],
                    epsilon = 1e-10
                ));
            }
        }
    }

    #[test]
    fn test_truncated_svd() {
        let matrix = DMatrix::from_row_slice(3, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

        let _full_svd = nalgebra_svd::compute_svd(&matrix).unwrap();
        let truncated_svd = nalgebra_svd::compute_truncated_svd(&matrix, 2).unwrap();

        assert_eq!(truncated_svd.singular_values.len(), 2);
        assert_eq!(truncated_svd.u.ncols(), 2);
        assert_eq!(truncated_svd.vt.nrows(), 2);
    }

    #[test]
    fn test_nalgebra_svd_with_tolerance() {
        let matrix = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let svd = nalgebra_svd::compute_svd_with_tolerance(&matrix, 1e-10_f64).unwrap();

        // Reconstruct and check
        let reconstructed = nalgebra_svd::reconstruct_matrix(&svd);
        for i in 0..matrix.nrows() {
            for j in 0..matrix.ncols() {
                assert!(approx::relative_eq!(
                    matrix[(i, j)],
                    reconstructed[(i, j)],
                    epsilon = 1e-10
                ));
            }
        }
    }

    #[test]
    fn test_nalgebra_svd_with_tolerance_empty_matrix() {
        let empty_matrix = DMatrix::<f64>::zeros(0, 0);
        let result = nalgebra_svd::compute_svd_with_tolerance(&empty_matrix, 1e-10_f64);
        assert!(matches!(result, Err(SVDError::EmptyMatrix)));
    }

    #[test]
    fn test_pseudo_inverse() {
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 2.0]);
        let pinv = nalgebra_svd::pseudo_inverse(&a, &PseudoInverseConfig::default()).unwrap();
        let product = &a * &pinv;
        assert!(approx::relative_eq!(product[(0, 0)], 1.0, epsilon = 1e-10));
        assert!(approx::relative_eq!(product[(1, 1)], 1.0, epsilon = 1e-10));
    }

    #[test]
    fn test_null_space_rank_deficient() {
        // Matrix with second column = 2 * first column, so rank 1, null space dim 1
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 4.0]);
        let nulls = nalgebra_svd::null_space(&a, Some(1e-10)).unwrap();
        assert_eq!(nulls.ncols(), 1);
        // Verify A * null_col ≈ 0
        let col = nulls.column(0);
        let ax = &a * col;
        assert!(ax.norm() < 1e-8);
    }

    #[test]
    fn test_svd_error_display_variants() {
        assert!(format!("{}", SVDError::NotSquare).contains("square"));
        assert!(format!("{}", SVDError::EmptyMatrix).contains("empty"));
        assert!(format!("{}", SVDError::ConvergenceFailed).contains("failed"));
        assert!(format!("{}", SVDError::InvalidInput("x".to_string())).contains('x'));
    }

    #[test]
    fn test_svd_ndarray_wrappers_and_edge_paths() {
        let empty_nalg = NalgebraSVD {
            u:               DMatrix::<f64>::zeros(0, 0),
            singular_values: DVector::<f64>::zeros(0),
            vt:              DMatrix::<f64>::zeros(0, 0),
        };
        assert!(nalgebra_svd::condition_number(&empty_nalg).abs() < f64::EPSILON);

        let empty_nd = NdarraySVD {
            u:               Array2::<f64>::zeros((0, 0)),
            singular_values: Array1::<f64>::zeros(0),
            vt:              Array2::<f64>::zeros((0, 0)),
        };
        assert!(ndarray_svd::condition_number(&empty_nd).abs() < f64::EPSILON);

        let full_rank = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 2.0]);
        let null_space = nalgebra_svd::null_space(&full_rank, Some(1e-12)).unwrap();
        assert_eq!(null_space.ncols(), 0);

        let matrix_nd = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let svd_tol = ndarray_svd::compute_svd_with_tolerance(&matrix_nd, 1e-12).unwrap();
        let reconstructed = ndarray_svd::reconstruct_matrix(&svd_tol);
        assert_eq!(reconstructed.dim(), (2, 2));

        assert!(matches!(
            ndarray_svd::compute_svd_with_tolerance(&Array2::<f64>::zeros((0, 0)), 1e-12),
            Err(SVDError::EmptyMatrix)
        ));
        assert!(matches!(
            ndarray_svd::compute_truncated_svd(&matrix_nd, 0),
            Err(SVDError::InvalidInput(_))
        ));

        let pinv =
            ndarray_svd::pseudo_inverse(&matrix_nd, &PseudoInverseConfig::default()).unwrap();
        assert_eq!(pinv.dim(), (2, 2));

        let rank_def = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 2.0, 4.0]).unwrap();
        let nulls = ndarray_svd::null_space(&rank_def, Some(1e-10)).unwrap();
        assert_eq!(nulls.ncols(), 1);
    }
}
