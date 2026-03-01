//! Schur decomposition over ndarray matrices.

use std::fmt;

use ndarray::{Array2, ArrayView2};

use crate::internal::{DEFAULT_TOLERANCE, identity, validate_finite, validate_square_non_empty};
use crate::qr::{QRConfig, ndarray_qr};

/// Error type for Schur decomposition.
#[derive(Debug, Clone, PartialEq)]
pub enum SchurError {
    /// Matrix is empty.
    EmptyMatrix,
    /// Matrix must be square.
    NotSquare,
    /// Iterative algorithm failed to converge.
    ConvergenceFailed,
    /// Numerical instability detected.
    NumericalInstability,
    /// Invalid input.
    InvalidInput(String),
}

impl fmt::Display for SchurError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SchurError::EmptyMatrix => write!(f, "Matrix is empty"),
            SchurError::NotSquare => write!(f, "Matrix must be square"),
            SchurError::ConvergenceFailed => write!(f, "Schur decomposition failed to converge"),
            SchurError::NumericalInstability => write!(f, "Numerical instability detected"),
            SchurError::InvalidInput(message) => write!(f, "Invalid input: {message}"),
        }
    }
}

impl std::error::Error for SchurError {}

/// Schur decomposition result.
#[derive(Debug, Clone)]
pub struct NdarraySchurResult {
    /// Orthogonal matrix Q.
    pub q: Array2<f64>,
    /// Upper triangular matrix T.
    pub t: Array2<f64>,
}

/// Reusable workspace for Schur decomposition `_into` kernels.
#[derive(Debug, Clone, Default)]
pub struct SchurWorkspace {
    q_scratch: Array2<f64>,
    t_scratch: Array2<f64>,
}

impl SchurWorkspace {
    fn ensure_square(&mut self, n: usize) {
        if self.q_scratch.dim() != (n, n) {
            self.q_scratch = Array2::<f64>::zeros((n, n));
        }
        if self.t_scratch.dim() != (n, n) {
            self.t_scratch = Array2::<f64>::zeros((n, n));
        }
    }
}

fn off_diagonal_norm(matrix: &Array2<f64>) -> f64 {
    let n = matrix.nrows();
    let mut sum = 0.0_f64;
    for i in 0..n {
        for j in 0..i {
            let value = matrix[[i, j]];
            sum += value * value;
        }
    }
    sum.sqrt()
}

/// Ndarray Schur functions.
pub mod ndarray_schur {
    use super::*;

    fn validate_output_shapes(
        matrix: &Array2<f64>,
        output_q: &Array2<f64>,
        output_t: &Array2<f64>,
    ) -> Result<(), SchurError> {
        let expected = (matrix.nrows(), matrix.ncols());
        if output_q.dim() != expected || output_t.dim() != expected {
            return Err(SchurError::InvalidInput(
                "output_q/output_t shapes must match input matrix shape".to_string(),
            ));
        }
        Ok(())
    }

    /// Compute Schur decomposition `A = Q T Q^T`.
    ///
    /// # Errors
    /// Returns an error for invalid input or convergence failure.
    pub fn compute_schur(matrix: &Array2<f64>) -> Result<NdarraySchurResult, SchurError> {
        validate_square_non_empty(matrix).map_err(|error| match error {
            "empty" => SchurError::EmptyMatrix,
            "not_square" => SchurError::NotSquare,
            _ => SchurError::InvalidInput(error.to_string()),
        })?;
        validate_finite(matrix).map_err(|_| SchurError::NumericalInstability)?;

        let n = matrix.nrows();
        let mut q_total = identity(n);
        let mut t = matrix.clone();
        let config = QRConfig::default();

        let mut converged = false;
        for _ in 0..config.max_iterations.max(128) {
            let qr =
                ndarray_qr::decompose(&t, &config).map_err(|_| SchurError::ConvergenceFailed)?;
            t = qr.r.dot(&qr.q);
            q_total = q_total.dot(&qr.q);
            if off_diagonal_norm(&t) < config.rank_tolerance.max(DEFAULT_TOLERANCE) {
                converged = true;
                break;
            }
        }

        if !converged {
            return Err(SchurError::ConvergenceFailed);
        }

        Ok(NdarraySchurResult { q: q_total, t })
    }

    /// Compute Schur decomposition `A = Q T Q^T` from a matrix view.
    ///
    /// # Errors
    /// Returns an error for invalid input or convergence failure.
    pub fn compute_schur_view(
        matrix: &ArrayView2<'_, f64>,
    ) -> Result<NdarraySchurResult, SchurError> {
        compute_schur(&matrix.to_owned())
    }

    /// Compute Schur decomposition into caller-provided outputs.
    ///
    /// # Errors
    /// Returns an error for invalid inputs, output shapes, or convergence failure.
    pub fn compute_schur_into(
        matrix: &Array2<f64>,
        output_q: &mut Array2<f64>,
        output_t: &mut Array2<f64>,
    ) -> Result<(), SchurError> {
        let mut workspace = SchurWorkspace::default();
        compute_schur_with_workspace_into(matrix, output_q, output_t, &mut workspace)
    }

    /// Compute Schur decomposition into caller-provided outputs from a matrix view.
    ///
    /// # Errors
    /// Returns an error for invalid inputs, output shapes, or convergence failure.
    pub fn compute_schur_into_view(
        matrix: &ArrayView2<'_, f64>,
        output_q: &mut Array2<f64>,
        output_t: &mut Array2<f64>,
    ) -> Result<(), SchurError> {
        compute_schur_into(&matrix.to_owned(), output_q, output_t)
    }

    /// Compute Schur decomposition into caller-provided outputs using reusable `workspace`.
    ///
    /// # Errors
    /// Returns an error for invalid inputs, output shapes, or convergence failure.
    pub fn compute_schur_with_workspace_into(
        matrix: &Array2<f64>,
        output_q: &mut Array2<f64>,
        output_t: &mut Array2<f64>,
        workspace: &mut SchurWorkspace,
    ) -> Result<(), SchurError> {
        validate_output_shapes(matrix, output_q, output_t)?;
        workspace.ensure_square(matrix.nrows());

        let result = compute_schur(matrix)?;
        workspace.q_scratch.assign(&result.q);
        workspace.t_scratch.assign(&result.t);
        output_q.assign(&workspace.q_scratch);
        output_t.assign(&workspace.t_scratch);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use super::{SchurError, SchurWorkspace, ndarray_schur};

    #[test]
    fn schur_reconstructs_matrix() {
        let matrix = Array2::from_shape_vec((2, 2), vec![4.0, 1.0, 2.0, 3.0]).unwrap();
        let schur = ndarray_schur::compute_schur(&matrix).unwrap();
        let reconstructed = schur.q.dot(&schur.t).dot(&schur.q.t());
        for i in 0..2 {
            for j in 0..2 {
                assert!((matrix[[i, j]] - reconstructed[[i, j]]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn schur_into_matches_allocating_path() {
        let matrix = Array2::from_shape_vec((2, 2), vec![5.0, 2.0, 1.0, 4.0]).unwrap();
        let expected = ndarray_schur::compute_schur(&matrix).unwrap();

        let mut q = Array2::<f64>::zeros((2, 2));
        let mut t = Array2::<f64>::zeros((2, 2));
        let mut workspace = SchurWorkspace::default();
        ndarray_schur::compute_schur_with_workspace_into(&matrix, &mut q, &mut t, &mut workspace)
            .unwrap();

        for i in 0..2 {
            for j in 0..2 {
                assert!((q[[i, j]] - expected.q[[i, j]]).abs() < 1e-8);
                assert!((t[[i, j]] - expected.t[[i, j]]).abs() < 1e-8);
            }
        }
    }

    #[test]
    fn schur_rejects_invalid_inputs() {
        let empty = Array2::<f64>::zeros((0, 0));
        assert!(matches!(ndarray_schur::compute_schur(&empty), Err(SchurError::EmptyMatrix)));

        let non_square = Array2::<f64>::zeros((2, 3));
        assert!(matches!(ndarray_schur::compute_schur(&non_square), Err(SchurError::NotSquare)));

        let non_finite = Array2::from_shape_vec((2, 2), vec![1.0, f64::NAN, 0.0, 1.0]).unwrap();
        assert!(matches!(
            ndarray_schur::compute_schur(&non_finite),
            Err(SchurError::NumericalInstability)
        ));
    }

    #[test]
    fn schur_into_rejects_bad_output_shapes() {
        let matrix = Array2::eye(2);
        let mut bad_q = Array2::<f64>::zeros((1, 2));
        let mut bad_t = Array2::<f64>::zeros((2, 2));
        assert!(matches!(
            ndarray_schur::compute_schur_into(&matrix, &mut bad_q, &mut bad_t),
            Err(SchurError::InvalidInput(_))
        ));
    }

    #[test]
    fn schur_view_variants_match_owned() {
        let matrix = Array2::from_shape_vec((2, 2), vec![3.0, 1.0, 0.0, 2.0]).unwrap();
        let owned = ndarray_schur::compute_schur(&matrix).unwrap();
        let viewed = ndarray_schur::compute_schur_view(&matrix.view()).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!((owned.q[[i, j]] - viewed.q[[i, j]]).abs() < 1e-12);
                assert!((owned.t[[i, j]] - viewed.t[[i, j]]).abs() < 1e-12);
            }
        }

        let mut q = Array2::<f64>::zeros((2, 2));
        let mut t = Array2::<f64>::zeros((2, 2));
        ndarray_schur::compute_schur_into_view(&matrix.view(), &mut q, &mut t).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!((owned.q[[i, j]] - q[[i, j]]).abs() < 1e-12);
                assert!((owned.t[[i, j]] - t[[i, j]]).abs() < 1e-12);
            }
        }
    }
}
