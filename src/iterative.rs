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
#[derive(Debug, Clone, PartialEq)]
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
    pub fn conjugate_gradient<T: RealField + Copy + num_traits::Float>(
        a: &DMatrix<T>,
        b: &DVector<T>,
        config: &IterativeConfig<T>,
    ) -> Result<DVector<T>, IterativeError> {
        if a.is_empty() || b.is_empty() {
            return Err(IterativeError::EmptyMatrix);
        }
        if !a.is_square() || a.nrows() != b.len() {
            return Err(IterativeError::DimensionMismatch);
        }

        let n = b.len();
        let mut x = DVector::zeros(n);
        let mut r = b.clone();
        let mut p = r.clone();
        let mut rs_old = r.dot(&r);

        for _ in 0..config.max_iterations {
            let ap = a * &p;
            let pap = p.dot(&ap);
            if pap <= T::zero() {
                return Err(IterativeError::NotPositiveDefinite);
            }
            let alpha = rs_old / pap;
            x += &p * alpha;
            r -= &ap * alpha;
            let rs_new = r.dot(&r);
            if num_traits::Float::sqrt(rs_new) < config.tolerance {
                return Ok(x);
            }
            let beta = rs_new / rs_old;
            p = &r + &p * beta;
            rs_old = rs_new;
        }
        Err(IterativeError::MaxIterationsExceeded)
    }

    /// GMRES for general system Ax = b (restart version)
    pub fn gmres<T: RealField + Copy + num_traits::Float + num_traits::float::FloatCore>(
        a: &DMatrix<T>,
        b: &DVector<T>,
        config: &IterativeConfig<T>,
    ) -> Result<DVector<T>, IterativeError> {
        if a.is_empty() || b.is_empty() {
            return Err(IterativeError::EmptyMatrix);
        }
        if !a.is_square() || a.nrows() != b.len() {
            return Err(IterativeError::DimensionMismatch);
        }

        let n = b.len();
        let restart = n.clamp(1, 50);
        let mut x = DVector::zeros(n);

        for _outer in 0..(config.max_iterations / restart).max(1) {
            let r = b - a * &x;
            let beta = r.norm();
            if beta < config.tolerance {
                return Ok(x);
            }

            let mut v = vec![DVector::zeros(n); restart + 1];
            v[0] = r / beta;

            let mut h = DMatrix::zeros(restart + 1, restart);
            let mut krylov_dim = restart;

            for j in 0..restart {
                let mut w = a * &v[j];
                for i in 0..=j {
                    h[(i, j)] = v[i].dot(&w);
                    w -= &v[i] * h[(i, j)];
                }
                h[(j + 1, j)] = w.norm();
                if h[(j + 1, j)] < <T as num_traits::Float>::epsilon() * beta {
                    krylov_dim = j;
                    break;
                }
                v[j + 1] = &w / h[(j + 1, j)];
            }

            let h_small = h.view((0, 0), (krylov_dim + 1, krylov_dim)).clone_owned();
            let mut e1 = DVector::zeros(krylov_dim + 1);
            e1[0] = beta;

            let y = crate::qr::nalgebra_qr::solve_least_squares(&h_small, &e1, &Default::default())
                .map_err(|_| IterativeError::Breakdown)?;

            for i in 0..krylov_dim {
                x += &v[i] * y[i];
            }
        }
        Err(IterativeError::MaxIterationsExceeded)
    }
}

/// Ndarray iterative solvers
pub mod ndarray_iterative {
    use super::*;
    use crate::utils::ndarray_to_nalgebra;

    /// Conjugate Gradient for SPD system Ax = b
    pub fn conjugate_gradient<T: Float + RealField>(
        a: &Array2<T>,
        b: &Array1<T>,
        config: &IterativeConfig<T>,
    ) -> Result<Array1<T>, IterativeError> {
        let nalg_a = ndarray_to_nalgebra(a);
        let nalg_b = nalgebra::DVector::from_vec(b.to_vec());
        let result = super::nalgebra_iterative::conjugate_gradient(&nalg_a, &nalg_b, config)?;
        Ok(Array1::from_vec(result.as_slice().to_vec()))
    }

    /// GMRES for general system Ax = b
    pub fn gmres<T: Float + RealField + num_traits::float::FloatCore>(
        a: &Array2<T>,
        b: &Array1<T>,
        config: &IterativeConfig<T>,
    ) -> Result<Array1<T>, IterativeError> {
        let nalg_a = ndarray_to_nalgebra(a);
        let nalg_b = nalgebra::DVector::from_vec(b.to_vec());
        let result = super::nalgebra_iterative::gmres(&nalg_a, &nalg_b, config)?;
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
}
