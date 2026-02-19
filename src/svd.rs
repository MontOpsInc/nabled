//! # Singular Value Decomposition (SVD)
//!
//! This module provides SVD implementations using both nalgebra and ndarray.
//! It includes enhanced algorithms and utilities for SVD computation.

use nalgebra::{DMatrix, DVector, RealField};
use ndarray::{s, Array1, Array2};
use num_traits::{float::FloatCore, Float};
use std::fmt;

/// SVD result structure for nalgebra
#[derive(Debug, Clone)]
pub struct NalgebraSVD<T: RealField> {
    /// Left singular vectors (U matrix)
    pub u: DMatrix<T>,
    /// Singular values
    pub singular_values: DVector<T>,
    /// Right singular vectors (V^T matrix)
    pub vt: DMatrix<T>,
}

/// SVD result structure for ndarray
#[derive(Debug, Clone)]
pub struct NdarraySVD<T: Float> {
    /// Left singular vectors (U matrix)
    pub u: Array2<T>,
    /// Singular values
    pub singular_values: Array1<T>,
    /// Right singular vectors (V^T matrix)
    pub vt: Array2<T>,
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
            SVDError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for SVDError {}

/// Enhanced SVD computation using nalgebra
pub mod nalgebra_svd {
    use super::*;
    // Note: ComplexField and SymmetricEigen imports removed as they are not used

    /// Compute SVD using nalgebra's built-in implementation
    pub fn compute_svd<T: RealField>(matrix: &DMatrix<T>) -> Result<NalgebraSVD<T>, SVDError> {
        if matrix.is_empty() {
            return Err(SVDError::EmptyMatrix);
        }

        // Use nalgebra's SVD implementation
        let svd = matrix.clone().svd(true, true);

        // Check if U and V^T were computed successfully
        if svd.u.is_some() && svd.v_t.is_some() {
            Ok(NalgebraSVD {
                u: svd.u.unwrap(),
                singular_values: svd.singular_values,
                vt: svd.v_t.unwrap(),
            })
        } else {
            Err(SVDError::ConvergenceFailed)
        }
    }

    /// Compute SVD with custom tolerance.
    /// Singular values below the tolerance are set to zero (useful for rank determination and low-rank approximation).
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
            .map(|sv| {
                if *sv < tolerance {
                    T::zero()
                } else {
                    sv.clone()
                }
            })
            .collect();
        svd.singular_values = DVector::from_vec(sv_values);
        Ok(svd)
    }

    /// Compute truncated SVD (keeping only the k largest singular values)
    pub fn compute_truncated_svd<T: RealField>(
        matrix: &DMatrix<T>,
        k: usize,
    ) -> Result<NalgebraSVD<T>, SVDError> {
        let full_svd = compute_svd(matrix)?;

        let _m = full_svd.u.nrows();
        let _n = full_svd.vt.ncols();
        let k = k.min(full_svd.singular_values.len());

        if k == 0 {
            return Err(SVDError::InvalidInput(
                "k must be greater than 0".to_string(),
            ));
        }

        // Truncate the matrices
        let u_truncated = full_svd.u.columns(0, k).into_owned();
        let singular_values_truncated = full_svd.singular_values.rows(0, k).into_owned();
        let vt_truncated = full_svd.vt.rows(0, k).into_owned();

        Ok(NalgebraSVD {
            u: u_truncated,
            singular_values: singular_values_truncated,
            vt: vt_truncated,
        })
    }

    /// Reconstruct the original matrix from SVD components
    pub fn reconstruct_matrix<T: RealField>(svd: &NalgebraSVD<T>) -> DMatrix<T> {
        let sigma = DMatrix::from_diagonal(&svd.singular_values);
        &svd.u * &sigma * &svd.vt
    }

    /// Compute the condition number from singular values
    pub fn condition_number<T: RealField + FloatCore>(svd: &NalgebraSVD<T>) -> T {
        let singular_values = &svd.singular_values;
        if singular_values.is_empty() {
            return T::zero();
        }

        let max_sv = singular_values.max();
        let min_sv = singular_values.min();

        if min_sv.is_zero() {
            T::infinity()
        } else {
            max_sv / min_sv
        }
    }

    /// Compute the rank of the matrix from its SVD
    pub fn matrix_rank<T: RealField>(svd: &NalgebraSVD<T>, tolerance: Option<T>) -> usize {
        let tol = tolerance.unwrap_or_else(|| {
            let max_sv = svd.singular_values.max();
            T::from_f64(1e-10).unwrap() * max_sv
        });

        svd.singular_values.iter().filter(|&sv| *sv > tol).count()
    }
}

/// Enhanced SVD computation using ndarray
pub mod ndarray_svd {
    use super::*;
    // Note: general_mat_mul import removed as it's not used

    /// Compute SVD using conversion to nalgebra and back
    /// Note: This is a simplified implementation that converts to nalgebra for computation
    pub fn compute_svd<T: Float + RealField>(
        matrix: &Array2<T>,
    ) -> Result<NdarraySVD<T>, SVDError> {
        if matrix.is_empty() {
            return Err(SVDError::EmptyMatrix);
        }

        // Convert ndarray to nalgebra
        use crate::utils::ndarray_to_nalgebra;
        let nalgebra_matrix = ndarray_to_nalgebra(matrix);

        // Compute SVD using nalgebra
        let nalgebra_svd = crate::svd::nalgebra_svd::compute_svd(&nalgebra_matrix)?;

        // Convert results back to ndarray
        use crate::utils::nalgebra_to_ndarray;
        let u = nalgebra_to_ndarray(&nalgebra_svd.u);
        let vt = nalgebra_to_ndarray(&nalgebra_svd.vt);
        let singular_values = Array1::from_vec(nalgebra_svd.singular_values.as_slice().to_vec());

        Ok(NdarraySVD {
            u,
            singular_values,
            vt,
        })
    }

    /// Compute SVD with custom tolerance.
    /// Singular values below the tolerance are set to zero.
    pub fn compute_svd_with_tolerance<T: Float + RealField>(
        matrix: &Array2<T>,
        tolerance: T,
    ) -> Result<NdarraySVD<T>, SVDError> {
        if matrix.is_empty() {
            return Err(SVDError::EmptyMatrix);
        }

        use crate::utils::{nalgebra_to_ndarray, ndarray_to_nalgebra};
        let nalgebra_matrix = ndarray_to_nalgebra(matrix);
        let nalgebra_svd =
            crate::svd::nalgebra_svd::compute_svd_with_tolerance(&nalgebra_matrix, tolerance)?;

        let u = nalgebra_to_ndarray(&nalgebra_svd.u);
        let vt = nalgebra_to_ndarray(&nalgebra_svd.vt);
        let singular_values = Array1::from_vec(nalgebra_svd.singular_values.as_slice().to_vec());

        Ok(NdarraySVD {
            u,
            singular_values,
            vt,
        })
    }

    /// Compute truncated SVD (keeping only the k largest singular values)
    pub fn compute_truncated_svd<T: Float + RealField>(
        matrix: &Array2<T>,
        k: usize,
    ) -> Result<NdarraySVD<T>, SVDError> {
        let full_svd = compute_svd(matrix)?;

        let k = k.min(full_svd.singular_values.len());

        if k == 0 {
            return Err(SVDError::InvalidInput(
                "k must be greater than 0".to_string(),
            ));
        }

        // Truncate the matrices
        let u_truncated = full_svd.u.slice(s![.., ..k]).to_owned();
        let singular_values_truncated = full_svd.singular_values.slice(s![..k]).to_owned();
        let vt_truncated = full_svd.vt.slice(s![..k, ..]).to_owned();

        Ok(NdarraySVD {
            u: u_truncated,
            singular_values: singular_values_truncated,
            vt: vt_truncated,
        })
    }

    /// Reconstruct the original matrix from SVD components
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
    pub fn condition_number<T: Float>(svd: &NdarraySVD<T>) -> T {
        if svd.singular_values.is_empty() {
            return T::zero();
        }

        let max_sv = svd.singular_values.iter().fold(T::zero(), |a, &b| a.max(b));
        let min_sv = svd
            .singular_values
            .iter()
            .fold(T::infinity(), |a, &b| a.min(b));

        if min_sv.is_zero() {
            T::infinity()
        } else {
            max_sv / min_sv
        }
    }

    /// Compute the rank of the matrix from its SVD
    pub fn matrix_rank<T: Float>(svd: &NdarraySVD<T>, tolerance: Option<T>) -> usize {
        let tol = tolerance.unwrap_or_else(|| {
            let max_sv = svd.singular_values.iter().fold(T::zero(), |a, &b| a.max(b));
            T::from(1e-10).unwrap() * max_sv
        });

        svd.singular_values.iter().filter(|&sv| *sv > tol).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;
    use ndarray::Array2;

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
}
