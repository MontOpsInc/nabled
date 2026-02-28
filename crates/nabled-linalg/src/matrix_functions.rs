//! Matrix functions over ndarray matrices.

use std::fmt;

use ndarray::{Array1, Array2};

use crate::eigen::ndarray_eigen;
use crate::internal::{
    DEFAULT_TOLERANCE, identity, is_symmetric, usize_to_f64, validate_finite,
    validate_square_non_empty,
};
use crate::svd::ndarray_svd;

/// Error type for matrix functions.
#[derive(Debug, Clone, PartialEq)]
pub enum MatrixFunctionError {
    /// Matrix is empty.
    EmptyMatrix,
    /// Matrix must be square.
    NotSquare,
    /// Matrix is not symmetric when required.
    NotSymmetric,
    /// Matrix is not positive definite when required.
    NotPositiveDefinite,
    /// Algorithm failed to converge.
    ConvergenceFailed,
    /// Invalid input.
    InvalidInput(String),
}

impl fmt::Display for MatrixFunctionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MatrixFunctionError::EmptyMatrix => write!(f, "Matrix cannot be empty"),
            MatrixFunctionError::NotSquare => write!(f, "Matrix must be square"),
            MatrixFunctionError::NotSymmetric => write!(f, "Matrix must be symmetric"),
            MatrixFunctionError::NotPositiveDefinite => {
                write!(f, "Matrix must be positive definite")
            }
            MatrixFunctionError::ConvergenceFailed => write!(f, "Algorithm failed to converge"),
            MatrixFunctionError::InvalidInput(message) => write!(f, "Invalid input: {message}"),
        }
    }
}

impl std::error::Error for MatrixFunctionError {}

/// Reusable workspace for matrix-function `_into` kernels.
#[derive(Debug, Clone, Default)]
pub struct MatrixFunctionWorkspace {
    scratch: Array2<f64>,
}

impl MatrixFunctionWorkspace {
    fn ensure_square(&mut self, n: usize) {
        if self.scratch.dim() != (n, n) {
            self.scratch = Array2::<f64>::zeros((n, n));
        }
    }
}

fn validate_square(matrix: &Array2<f64>) -> Result<(), MatrixFunctionError> {
    validate_square_non_empty(matrix).map_err(|error| match error {
        "empty" => MatrixFunctionError::EmptyMatrix,
        _ => MatrixFunctionError::NotSquare,
    })?;
    validate_finite(matrix)
        .map_err(|_| MatrixFunctionError::InvalidInput("matrix must be finite".into()))?;
    Ok(())
}

fn diagonal_from(values: &Array1<f64>) -> Array2<f64> {
    let n = values.len();
    let mut diagonal = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        diagonal[[i, i]] = values[i];
    }
    diagonal
}

fn taylor_matrix_exp(
    matrix: &Array2<f64>,
    max_terms: usize,
    tolerance: f64,
) -> Result<Array2<f64>, MatrixFunctionError> {
    validate_square(matrix)?;
    let n = matrix.nrows();
    let mut result = identity(n);
    let mut term = identity(n);

    for k in 1..=max_terms.max(1) {
        term = term.dot(matrix) / usize_to_f64(k);
        result = &result + &term;
        let delta = term.iter().map(|value| value.abs()).fold(0.0_f64, f64::max);
        if delta <= tolerance.max(DEFAULT_TOLERANCE) {
            return Ok(result);
        }
    }

    Ok(result)
}

/// Ndarray matrix functions.
pub mod ndarray_matrix_functions {
    use super::*;

    fn validate_output_shape(
        matrix: &Array2<f64>,
        output: &Array2<f64>,
    ) -> Result<(), MatrixFunctionError> {
        if output.dim() != matrix.dim() {
            return Err(MatrixFunctionError::InvalidInput(
                "output shape must match input matrix shape".to_string(),
            ));
        }
        Ok(())
    }

    /// Compute matrix exponential via Taylor series.
    ///
    /// # Errors
    /// Returns an error for invalid input.
    pub fn matrix_exp(
        matrix: &Array2<f64>,
        max_terms: usize,
        tolerance: f64,
    ) -> Result<Array2<f64>, MatrixFunctionError> {
        taylor_matrix_exp(matrix, max_terms, tolerance)
    }

    /// Compute matrix exponential into `output`.
    ///
    /// # Errors
    /// Returns an error for invalid inputs or output shape mismatch.
    pub fn matrix_exp_into(
        matrix: &Array2<f64>,
        max_terms: usize,
        tolerance: f64,
        output: &mut Array2<f64>,
    ) -> Result<(), MatrixFunctionError> {
        let mut workspace = MatrixFunctionWorkspace::default();
        matrix_exp_with_workspace_into(matrix, max_terms, tolerance, output, &mut workspace)
    }

    /// Compute matrix exponential into `output` using reusable `workspace`.
    ///
    /// # Errors
    /// Returns an error for invalid inputs or output shape mismatch.
    pub fn matrix_exp_with_workspace_into(
        matrix: &Array2<f64>,
        max_terms: usize,
        tolerance: f64,
        output: &mut Array2<f64>,
        workspace: &mut MatrixFunctionWorkspace,
    ) -> Result<(), MatrixFunctionError> {
        validate_output_shape(matrix, output)?;
        workspace.ensure_square(matrix.nrows());
        let result = matrix_exp(matrix, max_terms, tolerance)?;
        workspace.scratch.assign(&result);
        output.assign(&workspace.scratch);
        Ok(())
    }

    /// Compute matrix exponential via eigen decomposition when symmetric.
    ///
    /// Falls back to Taylor series for non-symmetric matrices.
    ///
    /// # Errors
    /// Returns an error for invalid input.
    pub fn matrix_exp_eigen(matrix: &Array2<f64>) -> Result<Array2<f64>, MatrixFunctionError> {
        validate_square(matrix)?;
        if !is_symmetric(matrix, DEFAULT_TOLERANCE) {
            return matrix_exp(matrix, 128, 1e-12);
        }

        let eigen =
            ndarray_eigen::symmetric(matrix).map_err(|_| MatrixFunctionError::ConvergenceFailed)?;
        let exp_values = eigen.eigenvalues.map(|value| value.exp());
        let diagonal = diagonal_from(&exp_values);
        Ok(eigen.eigenvectors.dot(&diagonal).dot(&eigen.eigenvectors.t()))
    }

    /// Compute matrix logarithm via Taylor expansion around identity.
    ///
    /// # Errors
    /// Returns an error for invalid input.
    pub fn matrix_log_taylor(
        matrix: &Array2<f64>,
        max_terms: usize,
        tolerance: f64,
    ) -> Result<Array2<f64>, MatrixFunctionError> {
        validate_square(matrix)?;
        let n = matrix.nrows();
        let identity = identity(n);
        let x = matrix - &identity;

        let mut result = Array2::<f64>::zeros((n, n));
        let mut term = x.clone();

        for k in 1..=max_terms.max(1) {
            let scale = if k % 2 == 0 { -1.0 } else { 1.0 } / usize_to_f64(k);
            result = &result + &(term.mapv(|value| scale * value));
            term = term.dot(&x);

            let delta = term.iter().map(|value| value.abs()).fold(0.0_f64, f64::max);
            if delta <= tolerance.max(DEFAULT_TOLERANCE) {
                break;
            }
        }

        Ok(result)
    }

    /// Compute matrix logarithm via Taylor expansion into `output`.
    ///
    /// # Errors
    /// Returns an error for invalid inputs or output shape mismatch.
    pub fn matrix_log_taylor_into(
        matrix: &Array2<f64>,
        max_terms: usize,
        tolerance: f64,
        output: &mut Array2<f64>,
    ) -> Result<(), MatrixFunctionError> {
        let mut workspace = MatrixFunctionWorkspace::default();
        matrix_log_taylor_with_workspace_into(matrix, max_terms, tolerance, output, &mut workspace)
    }

    /// Compute matrix logarithm via Taylor expansion into `output` with reusable `workspace`.
    ///
    /// # Errors
    /// Returns an error for invalid inputs or output shape mismatch.
    pub fn matrix_log_taylor_with_workspace_into(
        matrix: &Array2<f64>,
        max_terms: usize,
        tolerance: f64,
        output: &mut Array2<f64>,
        workspace: &mut MatrixFunctionWorkspace,
    ) -> Result<(), MatrixFunctionError> {
        validate_output_shape(matrix, output)?;
        workspace.ensure_square(matrix.nrows());
        let result = matrix_log_taylor(matrix, max_terms, tolerance)?;
        workspace.scratch.assign(&result);
        output.assign(&workspace.scratch);
        Ok(())
    }

    /// Compute matrix logarithm via eigen decomposition (symmetric PSD matrices).
    ///
    /// # Errors
    /// Returns an error if eigenvalues are non-positive.
    pub fn matrix_log_eigen(matrix: &Array2<f64>) -> Result<Array2<f64>, MatrixFunctionError> {
        validate_square(matrix)?;
        if !is_symmetric(matrix, DEFAULT_TOLERANCE) {
            return Err(MatrixFunctionError::NotSymmetric);
        }

        let eigen =
            ndarray_eigen::symmetric(matrix).map_err(|_| MatrixFunctionError::ConvergenceFailed)?;
        if eigen.eigenvalues.iter().any(|value| *value <= DEFAULT_TOLERANCE) {
            return Err(MatrixFunctionError::NotPositiveDefinite);
        }

        let log_values = eigen.eigenvalues.map(|value| value.ln());
        let diagonal = diagonal_from(&log_values);
        Ok(eigen.eigenvectors.dot(&diagonal).dot(&eigen.eigenvectors.t()))
    }

    /// Compute matrix logarithm via eigen decomposition into `output`.
    ///
    /// # Errors
    /// Returns an error for invalid inputs or output shape mismatch.
    pub fn matrix_log_eigen_into(
        matrix: &Array2<f64>,
        output: &mut Array2<f64>,
    ) -> Result<(), MatrixFunctionError> {
        let mut workspace = MatrixFunctionWorkspace::default();
        matrix_log_eigen_with_workspace_into(matrix, output, &mut workspace)
    }

    /// Compute matrix logarithm via eigen decomposition into `output` with reusable `workspace`.
    ///
    /// # Errors
    /// Returns an error for invalid inputs or output shape mismatch.
    pub fn matrix_log_eigen_with_workspace_into(
        matrix: &Array2<f64>,
        output: &mut Array2<f64>,
        workspace: &mut MatrixFunctionWorkspace,
    ) -> Result<(), MatrixFunctionError> {
        validate_output_shape(matrix, output)?;
        workspace.ensure_square(matrix.nrows());
        let result = matrix_log_eigen(matrix)?;
        workspace.scratch.assign(&result);
        output.assign(&workspace.scratch);
        Ok(())
    }

    /// Compute matrix logarithm via SVD.
    ///
    /// # Errors
    /// Returns an error if SVD fails.
    pub fn matrix_log_svd(matrix: &Array2<f64>) -> Result<Array2<f64>, MatrixFunctionError> {
        validate_square(matrix)?;
        let svd =
            ndarray_svd::decompose(matrix).map_err(|_| MatrixFunctionError::ConvergenceFailed)?;
        if svd.singular_values.iter().any(|value| *value <= DEFAULT_TOLERANCE) {
            return Err(MatrixFunctionError::NotPositiveDefinite);
        }

        let log_sigma = diagonal_from(&svd.singular_values.map(|value| value.ln()));
        Ok(svd.u.dot(&log_sigma).dot(&svd.vt))
    }

    /// Compute matrix logarithm via SVD into `output`.
    ///
    /// # Errors
    /// Returns an error for invalid inputs or output shape mismatch.
    pub fn matrix_log_svd_into(
        matrix: &Array2<f64>,
        output: &mut Array2<f64>,
    ) -> Result<(), MatrixFunctionError> {
        let mut workspace = MatrixFunctionWorkspace::default();
        matrix_log_svd_with_workspace_into(matrix, output, &mut workspace)
    }

    /// Compute matrix logarithm via SVD into `output` with reusable `workspace`.
    ///
    /// # Errors
    /// Returns an error for invalid inputs or output shape mismatch.
    pub fn matrix_log_svd_with_workspace_into(
        matrix: &Array2<f64>,
        output: &mut Array2<f64>,
        workspace: &mut MatrixFunctionWorkspace,
    ) -> Result<(), MatrixFunctionError> {
        validate_output_shape(matrix, output)?;
        workspace.ensure_square(matrix.nrows());
        let result = matrix_log_svd(matrix)?;
        workspace.scratch.assign(&result);
        output.assign(&workspace.scratch);
        Ok(())
    }

    /// Compute matrix power via eigen decomposition (symmetric matrices).
    ///
    /// # Errors
    /// Returns an error for non-symmetric inputs.
    pub fn matrix_power(
        matrix: &Array2<f64>,
        power: f64,
    ) -> Result<Array2<f64>, MatrixFunctionError> {
        validate_square(matrix)?;
        if !is_symmetric(matrix, DEFAULT_TOLERANCE) {
            return Err(MatrixFunctionError::NotSymmetric);
        }

        let eigen =
            ndarray_eigen::symmetric(matrix).map_err(|_| MatrixFunctionError::ConvergenceFailed)?;
        let powered_values = eigen.eigenvalues.map(|value| value.powf(power));
        let diagonal = diagonal_from(&powered_values);
        Ok(eigen.eigenvectors.dot(&diagonal).dot(&eigen.eigenvectors.t()))
    }

    /// Compute matrix power into `output`.
    ///
    /// # Errors
    /// Returns an error for invalid inputs or output shape mismatch.
    pub fn matrix_power_into(
        matrix: &Array2<f64>,
        power: f64,
        output: &mut Array2<f64>,
    ) -> Result<(), MatrixFunctionError> {
        let mut workspace = MatrixFunctionWorkspace::default();
        matrix_power_with_workspace_into(matrix, power, output, &mut workspace)
    }

    /// Compute matrix power into `output` using reusable `workspace`.
    ///
    /// # Errors
    /// Returns an error for invalid inputs or output shape mismatch.
    pub fn matrix_power_with_workspace_into(
        matrix: &Array2<f64>,
        power: f64,
        output: &mut Array2<f64>,
        workspace: &mut MatrixFunctionWorkspace,
    ) -> Result<(), MatrixFunctionError> {
        validate_output_shape(matrix, output)?;
        workspace.ensure_square(matrix.nrows());
        let result = matrix_power(matrix, power)?;
        workspace.scratch.assign(&result);
        output.assign(&workspace.scratch);
        Ok(())
    }

    /// Compute matrix sign via eigen decomposition (symmetric matrices).
    ///
    /// # Errors
    /// Returns an error for non-symmetric inputs.
    pub fn matrix_sign(matrix: &Array2<f64>) -> Result<Array2<f64>, MatrixFunctionError> {
        validate_square(matrix)?;
        if !is_symmetric(matrix, DEFAULT_TOLERANCE) {
            return Err(MatrixFunctionError::NotSymmetric);
        }

        let eigen =
            ndarray_eigen::symmetric(matrix).map_err(|_| MatrixFunctionError::ConvergenceFailed)?;
        let sign_values = eigen.eigenvalues.map(|value| {
            if *value > DEFAULT_TOLERANCE {
                1.0
            } else if *value < -DEFAULT_TOLERANCE {
                -1.0
            } else {
                0.0
            }
        });
        let diagonal = diagonal_from(&sign_values);
        Ok(eigen.eigenvectors.dot(&diagonal).dot(&eigen.eigenvectors.t()))
    }

    /// Compute matrix sign into `output`.
    ///
    /// # Errors
    /// Returns an error for invalid inputs or output shape mismatch.
    pub fn matrix_sign_into(
        matrix: &Array2<f64>,
        output: &mut Array2<f64>,
    ) -> Result<(), MatrixFunctionError> {
        let mut workspace = MatrixFunctionWorkspace::default();
        matrix_sign_with_workspace_into(matrix, output, &mut workspace)
    }

    /// Compute matrix sign into `output` using reusable `workspace`.
    ///
    /// # Errors
    /// Returns an error for invalid inputs or output shape mismatch.
    pub fn matrix_sign_with_workspace_into(
        matrix: &Array2<f64>,
        output: &mut Array2<f64>,
        workspace: &mut MatrixFunctionWorkspace,
    ) -> Result<(), MatrixFunctionError> {
        validate_output_shape(matrix, output)?;
        workspace.ensure_square(matrix.nrows());
        let result = matrix_sign(matrix)?;
        workspace.scratch.assign(&result);
        output.assign(&workspace.scratch);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use super::{MatrixFunctionError, MatrixFunctionWorkspace, ndarray_matrix_functions};

    #[test]
    fn exp_and_log_roundtrip_for_spd() {
        let matrix = Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 1.0, 2.0]).unwrap();
        let log_matrix = ndarray_matrix_functions::matrix_log_eigen(&matrix).unwrap();
        let roundtrip = ndarray_matrix_functions::matrix_exp_eigen(&log_matrix).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!((roundtrip[[i, j]] - matrix[[i, j]]).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn non_symmetric_log_is_rejected() {
        let matrix = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 0.0, 1.0]).unwrap();
        let result = ndarray_matrix_functions::matrix_log_eigen(&matrix);
        assert!(matches!(result, Err(MatrixFunctionError::NotSymmetric)));
    }

    #[test]
    fn matrix_power_into_matches_allocating_path() {
        let matrix = Array2::from_shape_vec((2, 2), vec![3.0, 1.0, 1.0, 3.0]).unwrap();
        let expected = ndarray_matrix_functions::matrix_power(&matrix, 0.5).unwrap();

        let mut output = Array2::<f64>::zeros((2, 2));
        let mut workspace = MatrixFunctionWorkspace::default();
        ndarray_matrix_functions::matrix_power_with_workspace_into(
            &matrix,
            0.5,
            &mut output,
            &mut workspace,
        )
        .unwrap();

        for i in 0..2 {
            for j in 0..2 {
                assert!((output[[i, j]] - expected[[i, j]]).abs() < 1e-8);
            }
        }
    }

    #[test]
    fn exp_into_and_workspace_match_allocating_path() {
        let matrix = Array2::from_shape_vec((2, 2), vec![0.2, 0.1, 0.0, 0.3]).unwrap();
        let expected = ndarray_matrix_functions::matrix_exp(&matrix, 64, 1e-12).unwrap();

        let mut output = Array2::<f64>::zeros((2, 2));
        ndarray_matrix_functions::matrix_exp_into(&matrix, 64, 1e-12, &mut output).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!((output[[i, j]] - expected[[i, j]]).abs() < 1e-8);
            }
        }
    }

    #[test]
    fn log_taylor_into_matches_allocating_path() {
        let matrix = Array2::from_shape_vec((2, 2), vec![1.1, 0.0, 0.0, 0.9]).unwrap();
        let expected = ndarray_matrix_functions::matrix_log_taylor(&matrix, 128, 1e-12).unwrap();

        let mut output = Array2::<f64>::zeros((2, 2));
        ndarray_matrix_functions::matrix_log_taylor_into(&matrix, 128, 1e-12, &mut output).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!((output[[i, j]] - expected[[i, j]]).abs() < 1e-8);
            }
        }
    }

    #[test]
    fn into_rejects_bad_output_shape() {
        let matrix = Array2::eye(2);
        let mut bad = Array2::<f64>::zeros((1, 1));
        let err = ndarray_matrix_functions::matrix_log_eigen_into(&matrix, &mut bad).unwrap_err();
        assert!(matches!(err, MatrixFunctionError::InvalidInput(_)));
    }

    #[test]
    fn log_svd_rejects_singular_input() {
        let matrix = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        let result = ndarray_matrix_functions::matrix_log_svd(&matrix);
        assert!(matches!(result, Err(MatrixFunctionError::NotPositiveDefinite)));
    }

    #[test]
    fn power_and_sign_reject_non_symmetric_input() {
        let matrix = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 0.0, 1.0]).unwrap();
        assert!(matches!(
            ndarray_matrix_functions::matrix_power(&matrix, 2.0),
            Err(MatrixFunctionError::NotSymmetric)
        ));
        assert!(matches!(
            ndarray_matrix_functions::matrix_sign(&matrix),
            Err(MatrixFunctionError::NotSymmetric)
        ));
    }

    #[test]
    fn sign_into_matches_allocating_path() {
        let matrix = Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 0.0, -3.0]).unwrap();
        let expected = ndarray_matrix_functions::matrix_sign(&matrix).unwrap();

        let mut output = Array2::<f64>::zeros((2, 2));
        ndarray_matrix_functions::matrix_sign_into(&matrix, &mut output).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!((output[[i, j]] - expected[[i, j]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn exp_eigen_falls_back_for_non_symmetric_input() {
        let matrix = Array2::from_shape_vec((2, 2), vec![0.0, 1.0, 0.0, 0.0]).unwrap();
        let eigen_path = ndarray_matrix_functions::matrix_exp_eigen(&matrix).unwrap();
        let taylor_path = ndarray_matrix_functions::matrix_exp(&matrix, 128, 1e-12).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!((eigen_path[[i, j]] - taylor_path[[i, j]]).abs() < 1e-8);
            }
        }
    }

    #[test]
    fn matrix_functions_reject_non_finite_input() {
        let matrix = Array2::from_shape_vec((2, 2), vec![1.0, f64::NAN, 0.0, 1.0]).unwrap();
        let result = ndarray_matrix_functions::matrix_exp(&matrix, 32, 1e-8);
        assert!(matches!(result, Err(MatrixFunctionError::InvalidInput(_))));
    }

    #[test]
    fn into_variants_reject_bad_output_shape() {
        let matrix = Array2::eye(2);
        let mut bad = Array2::<f64>::zeros((1, 1));
        assert!(matches!(
            ndarray_matrix_functions::matrix_exp_into(&matrix, 32, 1e-8, &mut bad),
            Err(MatrixFunctionError::InvalidInput(_))
        ));
        assert!(matches!(
            ndarray_matrix_functions::matrix_log_svd_into(&matrix, &mut bad),
            Err(MatrixFunctionError::InvalidInput(_))
        ));
        assert!(matches!(
            ndarray_matrix_functions::matrix_power_into(&matrix, 0.5, &mut bad),
            Err(MatrixFunctionError::InvalidInput(_))
        ));
        assert!(matches!(
            ndarray_matrix_functions::matrix_sign_into(&matrix, &mut bad),
            Err(MatrixFunctionError::InvalidInput(_))
        ));
    }

    #[test]
    fn matrix_log_eigen_rejects_non_positive_eigenvalues() {
        let matrix = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        let result = ndarray_matrix_functions::matrix_log_eigen(&matrix);
        assert!(matches!(result, Err(MatrixFunctionError::NotPositiveDefinite)));
    }

    #[test]
    fn matrix_sign_handles_negative_positive_and_zero_spectrum() {
        let matrix =
            Array2::from_shape_vec((3, 3), vec![2.0, 0.0, 0.0, 0.0, -4.0, 0.0, 0.0, 0.0, 0.0])
                .unwrap();
        let sign = ndarray_matrix_functions::matrix_sign(&matrix).unwrap();
        assert!((sign[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((sign[[1, 1]] + 1.0).abs() < 1e-10);
        assert!(sign[[2, 2]].abs() < 1e-10);
    }

    #[test]
    fn zero_max_terms_is_clamped_to_single_iteration() {
        let matrix = Array2::eye(2);
        let exp = ndarray_matrix_functions::matrix_exp(&matrix, 0, 1e-12).unwrap();
        assert!(exp[[0, 0]].is_finite());
        assert!(exp[[1, 1]].is_finite());

        let log = ndarray_matrix_functions::matrix_log_taylor(&matrix, 0, 1e-12).unwrap();
        assert!(log[[0, 0]].abs() < 1e-12);
        assert!(log[[1, 1]].abs() < 1e-12);
    }
}
