//! Iterative linear system solvers over ndarray matrices.

use std::fmt;

use nabled_linalg::lu::ndarray_lu;
use ndarray::{Array1, Array2};

const DEFAULT_TOLERANCE: f64 = 1.0e-12;

/// Configuration for iterative solvers.
#[derive(Debug, Clone)]
pub struct IterativeConfig<T = f64> {
    /// Relative residual tolerance.
    pub tolerance:      T,
    /// Maximum iterations.
    pub max_iterations: usize,
}

impl IterativeConfig<f64> {
    /// Default configuration for `f64`.
    #[must_use]
    pub const fn default_f64() -> Self { Self { tolerance: 1e-10, max_iterations: 1000 } }
}

impl Default for IterativeConfig<f64> {
    fn default() -> Self { Self::default_f64() }
}

/// Error type for iterative solvers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IterativeError {
    /// Matrix is empty.
    EmptyMatrix,
    /// Dimensions do not align.
    DimensionMismatch,
    /// Maximum iterations reached without convergence.
    MaxIterationsExceeded,
    /// Matrix is not positive definite (CG).
    NotPositiveDefinite,
    /// Algorithm breakdown.
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

fn vector_norm(vector: &Array1<f64>) -> f64 {
    vector.iter().map(|value| value * value).sum::<f64>().sqrt()
}

/// Ndarray iterative solvers.
pub mod ndarray_iterative {
    use super::*;

    /// Conjugate Gradient for SPD systems `Ax=b`.
    ///
    /// # Errors
    /// Returns an error when inputs are invalid or convergence fails.
    pub fn conjugate_gradient(
        matrix_a: &Array2<f64>,
        matrix_b: &Array1<f64>,
        config: &IterativeConfig<f64>,
    ) -> Result<Array1<f64>, IterativeError> {
        if matrix_a.is_empty() || matrix_b.is_empty() {
            return Err(IterativeError::EmptyMatrix);
        }
        if matrix_a.nrows() != matrix_a.ncols() || matrix_a.nrows() != matrix_b.len() {
            return Err(IterativeError::DimensionMismatch);
        }

        let n = matrix_b.len();
        let mut x = Array1::<f64>::zeros(n);
        let mut r = matrix_b.clone();
        let mut p = r.clone();
        let mut rs_old = r.dot(&r);

        if rs_old.sqrt() <= config.tolerance.max(DEFAULT_TOLERANCE) {
            return Ok(x);
        }

        for _ in 0..config.max_iterations {
            let ap = matrix_a.dot(&p);
            let curvature = p.dot(&ap);
            if curvature <= DEFAULT_TOLERANCE {
                return Err(IterativeError::NotPositiveDefinite);
            }

            let alpha = rs_old / curvature;
            x = &x + &(alpha * &p);
            r = &r - &(alpha * &ap);

            let rs_new = r.dot(&r);
            if rs_new.sqrt() <= config.tolerance.max(DEFAULT_TOLERANCE) {
                return Ok(x);
            }

            let beta = rs_new / rs_old;
            p = &r + &(beta * &p);
            rs_old = rs_new;
        }

        Err(IterativeError::MaxIterationsExceeded)
    }

    /// GMRES for general systems `Ax=b`.
    ///
    /// # Errors
    /// Returns an error when inputs are invalid or convergence fails.
    #[allow(clippy::many_single_char_names)]
    pub fn gmres(
        matrix_a: &Array2<f64>,
        matrix_b: &Array1<f64>,
        config: &IterativeConfig<f64>,
    ) -> Result<Array1<f64>, IterativeError> {
        if matrix_a.is_empty() || matrix_b.is_empty() {
            return Err(IterativeError::EmptyMatrix);
        }
        if matrix_a.nrows() != matrix_a.ncols() || matrix_a.nrows() != matrix_b.len() {
            return Err(IterativeError::DimensionMismatch);
        }

        let n = matrix_b.len();
        let m = n.min(config.max_iterations.max(1));
        let mut basis = Array2::<f64>::zeros((n, m + 1));
        let mut hessenberg = Array2::<f64>::zeros((m + 1, m));

        let beta = vector_norm(matrix_b);
        if beta <= config.tolerance.max(DEFAULT_TOLERANCE) {
            return Ok(Array1::<f64>::zeros(n));
        }

        for row in 0..n {
            basis[[row, 0]] = matrix_b[row] / beta;
        }

        let mut effective_m = m;
        for j in 0..m {
            let vj = basis.column(j).to_owned();
            let mut w = matrix_a.dot(&vj);

            for i in 0..=j {
                let vi = basis.column(i);
                let hij = vi.dot(&w);
                hessenberg[[i, j]] = hij;
                for row in 0..n {
                    w[row] -= hij * basis[[row, i]];
                }
            }

            let norm_w = vector_norm(&w);
            hessenberg[[j + 1, j]] = norm_w;
            if norm_w <= config.tolerance.max(DEFAULT_TOLERANCE) {
                effective_m = j + 1;
                break;
            }
            for row in 0..n {
                basis[[row, j + 1]] = w[row] / norm_w;
            }
        }

        let h = hessenberg.slice(ndarray::s![..(effective_m + 1), ..effective_m]).to_owned();
        let ht = h.t().to_owned();
        let normal_matrix = ht.dot(&h);

        let mut rhs_ls = Array1::<f64>::zeros(effective_m + 1);
        rhs_ls[0] = beta;
        let normal_rhs = ht.dot(&rhs_ls);

        let y = ndarray_lu::solve(&normal_matrix, &normal_rhs)
            .map_err(|_| IterativeError::Breakdown)?;
        let v = basis.slice(ndarray::s![.., ..effective_m]).to_owned();
        let x = v.dot(&y);

        let residual = matrix_b - &matrix_a.dot(&x);
        if vector_norm(&residual) <= config.tolerance.max(DEFAULT_TOLERANCE) {
            Ok(x)
        } else {
            Err(IterativeError::MaxIterationsExceeded)
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2};

    use super::{IterativeConfig, ndarray_iterative};

    #[test]
    fn cg_solves_spd_system() {
        let matrix = Array2::from_shape_vec((2, 2), vec![4.0, 1.0, 1.0, 3.0]).unwrap();
        let rhs = Array1::from_vec(vec![1.0, 2.0]);
        let solution =
            ndarray_iterative::conjugate_gradient(&matrix, &rhs, &IterativeConfig::default())
                .unwrap();
        let reconstructed = matrix.dot(&solution);
        assert!((reconstructed[0] - rhs[0]).abs() < 1e-8);
        assert!((reconstructed[1] - rhs[1]).abs() < 1e-8);
    }

    #[test]
    fn gmres_solves_small_system() {
        let matrix = Array2::from_shape_vec((2, 2), vec![3.0, 1.0, 1.0, 2.0]).unwrap();
        let rhs = Array1::from_vec(vec![9.0, 8.0]);
        let solution =
            ndarray_iterative::gmres(&matrix, &rhs, &IterativeConfig::default()).unwrap();
        let reconstructed = matrix.dot(&solution);
        assert!((reconstructed[0] - rhs[0]).abs() < 1e-8);
        assert!((reconstructed[1] - rhs[1]).abs() < 1e-8);
    }

    #[test]
    fn cg_rejects_dimension_mismatch() {
        let matrix = Array2::eye(2);
        let rhs = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let result =
            ndarray_iterative::conjugate_gradient(&matrix, &rhs, &IterativeConfig::default());
        assert!(matches!(result, Err(super::IterativeError::DimensionMismatch)));
    }

    #[test]
    fn gmres_rejects_empty_input() {
        let matrix = Array2::<f64>::zeros((0, 0));
        let rhs = Array1::<f64>::zeros(0);
        let result = ndarray_iterative::gmres(&matrix, &rhs, &IterativeConfig::default());
        assert!(matches!(result, Err(super::IterativeError::EmptyMatrix)));
    }

    #[test]
    fn cg_returns_zero_for_zero_rhs() {
        let matrix = Array2::eye(2);
        let rhs = Array1::from_vec(vec![0.0, 0.0]);
        let solution =
            ndarray_iterative::conjugate_gradient(&matrix, &rhs, &IterativeConfig::default())
                .unwrap();
        assert!(solution.iter().all(|value| value.abs() < 1e-12));
    }
}
