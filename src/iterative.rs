//! # Iterative Linear System Solvers
//!
//! Conjugate Gradient (CG) for symmetric positive-definite systems.
//! GMRES for general non-singular systems.

use std::fmt;

use nalgebra::{DMatrix, DVector, RealField};
use ndarray::{Array1, Array2};
use num_traits::Float;

/// Configuration for iterative solvers
#[derive(Debug, Clone)]
pub struct IterativeConfig<T> {
    /// Relative tolerance for residual
    pub tolerance:      T,
    /// Maximum number of iterations
    pub max_iterations: usize,
}

impl IterativeConfig<f64> {
    /// Default configuration for f64
    #[must_use]
    pub const fn default_f64() -> Self { Self { tolerance: 1e-10, max_iterations: 1000 } }
}

impl Default for IterativeConfig<f64> {
    fn default() -> Self { Self::default_f64() }
}

/// Error types for iterative solvers
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IterativeError {
    /// Matrix is empty
    EmptyMatrix,
    /// Dimension mismatch
    DimensionMismatch,
    /// Maximum iterations exceeded
    MaxIterationsExceeded,
    /// Matrix is not positive definite (for CG)
    NotPositiveDefinite,
    /// Breakdown in algorithm
    Breakdown,
}

impl fmt::Display for IterativeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IterativeError::EmptyMatrix => write!(f, "Matrix is empty"),
            IterativeError::DimensionMismatch => write!(f, "Dimension mismatch"),
            IterativeError::MaxIterationsExceeded => write!(f, "Maximum iterations exceeded"),
            IterativeError::NotPositiveDefinite => write!(f, "Matrix is not positive definite"),
            IterativeError::Breakdown => write!(f, "Algorithm breakdown"),
        }
    }
}

impl std::error::Error for IterativeError {}

/// Nalgebra iterative solvers
pub mod nalgebra_iterative {
    use super::*;

    /// Conjugate Gradient for SPD system Ax = b
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn conjugate_gradient<T: RealField + Copy + Float>(
        matrix_a: &DMatrix<T>,
        matrix_b: &DVector<T>,
        config: &IterativeConfig<T>,
    ) -> Result<DVector<T>, IterativeError> {
        if matrix_a.is_empty() || matrix_b.is_empty() {
            return Err(IterativeError::EmptyMatrix);
        }
        if !matrix_a.is_square() || matrix_a.nrows() != matrix_b.len() {
            return Err(IterativeError::DimensionMismatch);
        }

        let n = matrix_b.len();
        let mut x = DVector::zeros(n);
        let mut r = matrix_b.clone();
        let mut p = r.clone();
        let mut rs_old = r.dot(&r);

        for _ in 0..config.max_iterations {
            let ap = matrix_a * &p;
            let pap = p.dot(&ap);
            if pap <= T::zero() {
                return Err(IterativeError::NotPositiveDefinite);
            }
            let alpha = rs_old / pap;
            x += &p * alpha;
            r -= &ap * alpha;
            let rs_new = r.dot(&r);
            if Float::sqrt(rs_new) < config.tolerance {
                return Ok(x);
            }
            let beta = rs_new / rs_old;
            p = &r + &p * beta;
            rs_old = rs_new;
        }
        Err(IterativeError::MaxIterationsExceeded)
    }

    /// GMRES for general system Ax = b (restart version)
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn gmres<T: RealField + Copy + Float + num_traits::float::FloatCore>(
        matrix_a: &DMatrix<T>,
        matrix_b: &DVector<T>,
        config: &IterativeConfig<T>,
    ) -> Result<DVector<T>, IterativeError> {
        if matrix_a.is_empty() || matrix_b.is_empty() {
            return Err(IterativeError::EmptyMatrix);
        }
        if !matrix_a.is_square() || matrix_a.nrows() != matrix_b.len() {
            return Err(IterativeError::DimensionMismatch);
        }

        let b_len = matrix_b.len();
        let restart = b_len.clamp(1, 50);
        let mut b_len_zeroes = DVector::zeros(b_len);

        for _outer in 0..(config.max_iterations / restart).max(1) {
            let r = matrix_b - matrix_a * &b_len_zeroes;
            let beta = r.norm();
            if beta < config.tolerance {
                return Ok(b_len_zeroes);
            }

            let mut v = vec![DVector::zeros(b_len); restart + 1];
            v[0] = r / beta;

            let mut h = DMatrix::zeros(restart + 1, restart);
            let mut krylov_dim = restart;

            for j in 0..restart {
                let mut w = matrix_a * &v[j];
                for i in 0..=j {
                    h[(i, j)] = v[i].dot(&w);
                    w -= &v[i] * h[(i, j)];
                }
                h[(j + 1, j)] = w.norm();
                if h[(j + 1, j)] < <T as Float>::epsilon() * beta {
                    krylov_dim = j;
                    break;
                }
                v[j + 1] = &w / h[(j + 1, j)];
            }

            let h_small = h.view((0, 0), (krylov_dim + 1, krylov_dim)).clone_owned();
            let mut e1 = DVector::zeros(krylov_dim + 1);
            e1[0] = beta;

            let y = crate::qr::nalgebra_qr::solve_least_squares(
                &h_small,
                &e1,
                &crate::qr::QRConfig::default(),
            )
            .map_err(|_| IterativeError::Breakdown)?;

            for i in 0..krylov_dim {
                b_len_zeroes += &v[i] * y[i];
            }
        }
        Err(IterativeError::MaxIterationsExceeded)
    }
}

/// Ndarray iterative solvers
pub mod ndarray_iterative {
    use super::*;
    use crate::interop::ndarray_to_nalgebra;

    /// Conjugate Gradient for SPD system Ax = b
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn conjugate_gradient<T: Float + RealField>(
        a: &Array2<T>,
        b: &Array1<T>,
        config: &IterativeConfig<T>,
    ) -> Result<Array1<T>, IterativeError> {
        let nalg_a = ndarray_to_nalgebra(a);
        let nalg_b = DVector::from_vec(b.to_vec());
        let result = nalgebra_iterative::conjugate_gradient(&nalg_a, &nalg_b, config)?;
        Ok(Array1::from_vec(result.as_slice().to_vec()))
    }

    /// GMRES for general system Ax = b
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn gmres<T: Float + RealField + num_traits::float::FloatCore>(
        a: &Array2<T>,
        b: &Array1<T>,
        config: &IterativeConfig<T>,
    ) -> Result<Array1<T>, IterativeError> {
        let nalg_a = ndarray_to_nalgebra(a);
        let nalg_b = DVector::from_vec(b.to_vec());
        let result = nalgebra_iterative::gmres(&nalg_a, &nalg_b, config)?;
        Ok(Array1::from_vec(result.as_slice().to_vec()))
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn test_conjugate_gradient() {
        let a = DMatrix::from_row_slice(2, 2, &[4.0, 1.0, 1.0, 3.0]);
        let b = DVector::from_vec(vec![1.0, 2.0]);
        let x =
            nalgebra_iterative::conjugate_gradient(&a, &b, &IterativeConfig::default()).unwrap();
        let ax = &a * &x;
        for i in 0..2 {
            assert_relative_eq!(ax[i], b[i], epsilon = 1e-8);
        }
    }

    #[test]
    fn test_gmres() {
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let b = DVector::from_vec(vec![5.0, 11.0]);
        let x = nalgebra_iterative::gmres(&a, &b, &IterativeConfig::default()).unwrap();
        let ax = &a * &x;
        for i in 0..2 {
            assert_relative_eq!(ax[i], b[i], epsilon = 1e-8);
        }
    }

    #[test]
    fn test_ndarray_conjugate_gradient() {
        let a = Array2::from_shape_vec((2, 2), vec![4.0, 1.0, 1.0, 3.0]).unwrap();
        let b = Array1::from_vec(vec![1.0, 2.0]);
        let x = ndarray_iterative::conjugate_gradient(&a, &b, &IterativeConfig::default()).unwrap();
        let ax = a.dot(&x);

        for i in 0..2 {
            assert_relative_eq!(ax[i], b[i], epsilon = 1e-8);
        }
    }

    #[test]
    fn test_ndarray_gmres() {
        let a = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Array1::from_vec(vec![5.0, 11.0]);
        let x = ndarray_iterative::gmres(&a, &b, &IterativeConfig::default()).unwrap();
        let ax = a.dot(&x);

        for i in 0..2 {
            assert_relative_eq!(ax[i], b[i], epsilon = 1e-8);
        }
    }

    #[test]
    fn test_iterative_error_display_variants() {
        assert!(format!("{}", IterativeError::EmptyMatrix).contains("empty"));
        assert!(format!("{}", IterativeError::DimensionMismatch).contains("Dimension"));
        assert!(format!("{}", IterativeError::MaxIterationsExceeded).contains("Maximum"));
        assert!(format!("{}", IterativeError::NotPositiveDefinite).contains("positive definite"));
        assert!(format!("{}", IterativeError::Breakdown).contains("breakdown"));
    }

    #[test]
    fn test_iterative_error_paths() {
        let empty_a = DMatrix::<f64>::zeros(0, 0);
        let empty_b = DVector::<f64>::zeros(0);
        let cfg = IterativeConfig::default();
        assert!(matches!(
            nalgebra_iterative::conjugate_gradient(&empty_a, &empty_b, &cfg),
            Err(IterativeError::EmptyMatrix)
        ));
        assert!(matches!(
            nalgebra_iterative::gmres(&empty_a, &empty_b, &cfg),
            Err(IterativeError::EmptyMatrix)
        ));

        let a = DMatrix::<f64>::identity(2, 2);
        let b_bad = DVector::from_vec(vec![1.0]);
        assert!(matches!(
            nalgebra_iterative::conjugate_gradient(&a, &b_bad, &cfg),
            Err(IterativeError::DimensionMismatch)
        ));
        assert!(matches!(
            nalgebra_iterative::gmres(&a, &b_bad, &cfg),
            Err(IterativeError::DimensionMismatch)
        ));

        let not_pd = DMatrix::from_row_slice(2, 2, &[0.0, 0.0, 0.0, 0.0]);
        let b = DVector::from_vec(vec![1.0, 1.0]);
        assert!(matches!(
            nalgebra_iterative::conjugate_gradient(&not_pd, &b, &cfg),
            Err(IterativeError::NotPositiveDefinite)
        ));

        let cfg_zero_iter = IterativeConfig { tolerance: 1e-12, max_iterations: 0 };
        assert!(matches!(
            nalgebra_iterative::conjugate_gradient(
                &a,
                &DVector::from_vec(vec![1.0, 2.0]),
                &cfg_zero_iter
            ),
            Err(IterativeError::MaxIterationsExceeded)
        ));

        let cfg_one_iter = IterativeConfig { tolerance: 1e-30, max_iterations: 1 };
        let gmres_result = nalgebra_iterative::gmres(
            &DMatrix::from_row_slice(2, 2, &[4.0, 1.0, 1.0, 3.0]),
            &DVector::from_vec(vec![1.0, 2.0]),
            &cfg_one_iter,
        );
        assert!(matches!(
            gmres_result,
            Err(IterativeError::MaxIterationsExceeded | IterativeError::Breakdown)
        ));
    }
}
