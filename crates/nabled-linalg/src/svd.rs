//! Singular value decomposition over ndarray matrices.

use std::fmt;

use ndarray::{Array1, Array2, s};

use crate::internal::{
    DEFAULT_TOLERANCE, jacobi_eigen_symmetric, sort_eigenpairs_desc, usize_to_f64,
};

/// SVD result for ndarray matrices.
#[derive(Debug, Clone)]
pub struct NdarraySVD {
    /// Left singular vectors (`m x k`).
    pub u:               Array2<f64>,
    /// Singular values (`k`).
    pub singular_values: Array1<f64>,
    /// Right singular vectors transposed (`k x n`).
    pub vt:              Array2<f64>,
}

/// Error types for SVD computation.
#[derive(Debug, Clone, PartialEq)]
pub enum SVDError {
    /// Matrix is empty.
    EmptyMatrix,
    /// Matrix is not square when required.
    NotSquare,
    /// Iterative algorithm failed to converge.
    ConvergenceFailed,
    /// Invalid user input.
    InvalidInput(String),
}

impl fmt::Display for SVDError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SVDError::EmptyMatrix => write!(f, "Matrix cannot be empty"),
            SVDError::NotSquare => write!(f, "Matrix must be square"),
            SVDError::ConvergenceFailed => write!(f, "SVD algorithm failed to converge"),
            SVDError::InvalidInput(message) => write!(f, "Invalid input: {message}"),
        }
    }
}

impl std::error::Error for SVDError {}

/// Configuration for pseudo-inverse computation.
#[derive(Debug, Clone, Copy, Default)]
pub struct PseudoInverseConfig {
    /// Tolerance for truncating tiny singular values.
    pub tolerance: Option<f64>,
}

#[cfg(not(feature = "openblas-system"))]
fn decompose_internal(matrix: &Array2<f64>) -> Result<NdarraySVD, SVDError> {
    if matrix.is_empty() {
        return Err(SVDError::EmptyMatrix);
    }
    crate::internal::validate_finite(matrix)
        .map_err(|_| SVDError::InvalidInput("matrix must be finite".into()))?;

    let (rows, cols) = matrix.dim();
    let k = rows.min(cols);

    let ata = matrix.t().dot(matrix);
    let (eigenvalues, eigenvectors) = jacobi_eigen_symmetric(&ata, DEFAULT_TOLERANCE, 256)
        .map_err(|_| SVDError::ConvergenceFailed)?;
    let (sorted_values, sorted_vectors) = sort_eigenpairs_desc(&eigenvalues, &eigenvectors);

    let mut singular_values = Array1::<f64>::zeros(k);
    let mut vt = Array2::<f64>::zeros((k, cols));
    for i in 0..k {
        let value = sorted_values[i].max(0.0).sqrt();
        singular_values[i] = value;
        for j in 0..cols {
            vt[[i, j]] = sorted_vectors[[j, i]];
        }
    }

    let mut u = Array2::<f64>::zeros((rows, k));
    for i in 0..k {
        let sigma = singular_values[i];
        if sigma > DEFAULT_TOLERANCE {
            let v_i = sorted_vectors.column(i).to_owned();
            let av = matrix.dot(&v_i);
            for row in 0..rows {
                u[[row, i]] = av[row] / sigma;
            }
        }
    }

    Ok(NdarraySVD { u, singular_values, vt })
}

#[cfg(feature = "openblas-system")]
fn decompose_provider(matrix: &Array2<f64>) -> Result<NdarraySVD, SVDError> {
    use ndarray_linalg::SVD as _;

    if matrix.is_empty() {
        return Err(SVDError::EmptyMatrix);
    }

    let (u_opt, singular_values, vt_opt) =
        matrix.clone().svd(true, true).map_err(|_| SVDError::ConvergenceFailed)?;
    let u = u_opt.ok_or(SVDError::ConvergenceFailed)?;
    let vt = vt_opt.ok_or(SVDError::ConvergenceFailed)?;

    Ok(NdarraySVD { u, singular_values, vt })
}

fn null_space_internal(
    matrix: &Array2<f64>,
    tolerance: Option<f64>,
) -> Result<Array2<f64>, SVDError> {
    if matrix.is_empty() {
        return Err(SVDError::EmptyMatrix);
    }

    let ata = matrix.t().dot(matrix);
    let (eigenvalues, eigenvectors) = jacobi_eigen_symmetric(&ata, DEFAULT_TOLERANCE, 256)
        .map_err(|_| SVDError::ConvergenceFailed)?;
    let (sorted_values, sorted_vectors) = sort_eigenpairs_desc(&eigenvalues, &eigenvectors);

    let max_sv =
        sorted_values.iter().copied().map(|value| value.max(0.0).sqrt()).fold(0.0_f64, f64::max);
    let tol = tolerance
        .unwrap_or(max_sv * usize_to_f64(matrix.ncols()) * f64::EPSILON.max(DEFAULT_TOLERANCE));

    let mut null_indices = Vec::new();
    for (index, value) in sorted_values.iter().copied().enumerate() {
        let singular = value.max(0.0).sqrt();
        if singular <= tol {
            null_indices.push(index);
        }
    }

    if null_indices.is_empty() {
        return Ok(Array2::<f64>::zeros((matrix.ncols(), 0)));
    }

    let mut basis = Array2::<f64>::zeros((matrix.ncols(), null_indices.len()));
    for (col_out, col_in) in null_indices.into_iter().enumerate() {
        for row in 0..matrix.ncols() {
            basis[[row, col_out]] = sorted_vectors[[row, col_in]];
        }
    }

    Ok(basis)
}

/// Ndarray SVD functions.
pub mod ndarray_svd {
    use super::*;

    /// Compute the SVD of `matrix`.
    ///
    /// # Errors
    /// Returns an error if the matrix is empty, non-finite, or decomposition fails.
    pub fn decompose(matrix: &Array2<f64>) -> Result<NdarraySVD, SVDError> {
        #[cfg(feature = "openblas-system")]
        {
            decompose_provider(matrix)
        }
        #[cfg(not(feature = "openblas-system"))]
        {
            decompose_internal(matrix)
        }
    }

    /// Compute SVD and zero out singular values below `tolerance`.
    ///
    /// # Errors
    /// Returns an error if decomposition fails.
    pub fn decompose_with_tolerance(
        matrix: &Array2<f64>,
        tolerance: f64,
    ) -> Result<NdarraySVD, SVDError> {
        let mut svd = decompose(matrix)?;
        for value in &mut svd.singular_values {
            if *value < tolerance {
                *value = 0.0;
            }
        }
        Ok(svd)
    }

    /// Compute truncated SVD by keeping only the `k` largest singular values.
    ///
    /// # Errors
    /// Returns an error if `k == 0` or decomposition fails.
    pub fn decompose_truncated(matrix: &Array2<f64>, k: usize) -> Result<NdarraySVD, SVDError> {
        if k == 0 {
            return Err(SVDError::InvalidInput("k must be greater than 0".to_string()));
        }

        let full_svd = decompose(matrix)?;
        let keep = k.min(full_svd.singular_values.len());

        Ok(NdarraySVD {
            u:               full_svd.u.slice(s![.., ..keep]).to_owned(),
            singular_values: full_svd.singular_values.slice(s![..keep]).to_owned(),
            vt:              full_svd.vt.slice(s![..keep, ..]).to_owned(),
        })
    }

    /// Reconstruct the original matrix from SVD components.
    #[must_use]
    pub fn reconstruct_matrix(svd: &NdarraySVD) -> Array2<f64> {
        let cols = svd.vt.ncols();
        let mut sigma_vt = svd.vt.clone();
        for i in 0..svd.singular_values.len().min(svd.u.ncols()) {
            for j in 0..cols {
                sigma_vt[[i, j]] *= svd.singular_values[i];
            }
        }
        svd.u.dot(&sigma_vt)
    }

    /// Reconstruct the original matrix from SVD components into `output`.
    ///
    /// # Errors
    /// Returns an error if `output` shape is incompatible with SVD factors.
    pub fn reconstruct_matrix_into(
        svd: &NdarraySVD,
        output: &mut Array2<f64>,
    ) -> Result<(), SVDError> {
        let rows = svd.u.nrows();
        let cols = svd.vt.ncols();
        let k = svd.u.ncols();

        if output.dim() != (rows, cols) {
            return Err(SVDError::InvalidInput(
                "output shape must match reconstructed matrix shape".to_string(),
            ));
        }
        if svd.singular_values.len() != svd.vt.nrows() || svd.singular_values.len() != k {
            return Err(SVDError::InvalidInput("inconsistent SVD factor dimensions".to_string()));
        }

        output.fill(0.0);
        for i in 0..rows {
            for j in 0..cols {
                let mut sum = 0.0_f64;
                for p in 0..k {
                    sum += svd.u[[i, p]] * svd.singular_values[p] * svd.vt[[p, j]];
                }
                output[[i, j]] = sum;
            }
        }

        Ok(())
    }

    /// Compute condition number from singular values.
    #[must_use]
    pub fn condition_number(svd: &NdarraySVD) -> f64 {
        if svd.singular_values.is_empty() {
            return 0.0;
        }

        let max_sv = svd.singular_values.iter().copied().fold(0.0_f64, f64::max);
        let min_sv = svd
            .singular_values
            .iter()
            .copied()
            .filter(|value| *value > DEFAULT_TOLERANCE)
            .fold(f64::INFINITY, f64::min);

        if min_sv.is_finite() { max_sv / min_sv } else { f64::INFINITY }
    }

    /// Estimate numerical rank from singular values.
    #[must_use]
    pub fn rank(svd: &NdarraySVD, tolerance: Option<f64>) -> usize {
        let max_sv = svd.singular_values.iter().copied().fold(0.0_f64, f64::max);
        let tol =
            tolerance.unwrap_or(max_sv * usize_to_f64(svd.singular_values.len()) * f64::EPSILON);
        svd.singular_values.iter().filter(|value| **value > tol).count()
    }

    /// Compute Moore-Penrose pseudo-inverse.
    ///
    /// # Errors
    /// Returns an error if input is invalid or decomposition fails.
    pub fn pseudo_inverse(
        matrix: &Array2<f64>,
        config: &PseudoInverseConfig,
    ) -> Result<Array2<f64>, SVDError> {
        if matrix.is_empty() {
            return Err(SVDError::EmptyMatrix);
        }

        let mut output = Array2::<f64>::zeros((matrix.ncols(), matrix.nrows()));
        pseudo_inverse_into(matrix, config, &mut output)?;
        Ok(output)
    }

    /// Compute Moore-Penrose pseudo-inverse into `output`.
    ///
    /// # Errors
    /// Returns an error if dimensions are invalid or decomposition fails.
    pub fn pseudo_inverse_into(
        matrix: &Array2<f64>,
        config: &PseudoInverseConfig,
        output: &mut Array2<f64>,
    ) -> Result<(), SVDError> {
        if matrix.is_empty() {
            return Err(SVDError::EmptyMatrix);
        }
        if output.dim() != (matrix.ncols(), matrix.nrows()) {
            return Err(SVDError::InvalidInput(
                "output shape must be (matrix.ncols(), matrix.nrows())".to_string(),
            ));
        }

        let svd = decompose(matrix)?;
        let (rows, cols) = matrix.dim();
        let max_sv = svd.singular_values.iter().copied().fold(0.0_f64, f64::max);
        let tolerance = config
            .tolerance
            .unwrap_or(max_sv * usize_to_f64(rows.max(cols)) * f64::EPSILON.max(DEFAULT_TOLERANCE));

        output.fill(0.0);
        let k = svd.singular_values.len();
        for i in 0..k {
            let sigma = svd.singular_values[i];
            if sigma <= tolerance {
                continue;
            }
            let inv_sigma = 1.0 / sigma;
            for row in 0..cols {
                for col in 0..rows {
                    output[[row, col]] += svd.vt[[i, row]] * inv_sigma * svd.u[[col, i]];
                }
            }
        }

        Ok(())
    }

    /// Compute a basis for the right null-space of `matrix`.
    ///
    /// # Errors
    /// Returns an error if decomposition fails.
    pub fn null_space(
        matrix: &Array2<f64>,
        tolerance: Option<f64>,
    ) -> Result<Array2<f64>, SVDError> {
        null_space_internal(matrix, tolerance)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use super::{PseudoInverseConfig, SVDError, ndarray_svd};

    #[test]
    fn svd_reconstructs_small_matrix() {
        let matrix = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let svd = ndarray_svd::decompose(&matrix).unwrap();
        let reconstructed = ndarray_svd::reconstruct_matrix(&svd);
        for i in 0..2 {
            for j in 0..2 {
                assert!((matrix[[i, j]] - reconstructed[[i, j]]).abs() < 1e-8);
            }
        }
    }

    #[test]
    fn truncated_svd_requires_positive_rank() {
        let matrix = Array2::eye(2);
        let result = ndarray_svd::decompose_truncated(&matrix, 0);
        assert!(matches!(result, Err(SVDError::InvalidInput(_))));
    }

    #[test]
    fn pseudo_inverse_matches_identity_for_diagonal() {
        let matrix = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 2.0]).unwrap();
        let pinv = ndarray_svd::pseudo_inverse(&matrix, &PseudoInverseConfig::default()).unwrap();
        let product = matrix.dot(&pinv);
        assert!((product[[0, 0]] - 1.0).abs() < 1e-8);
        assert!((product[[1, 1]] - 1.0).abs() < 1e-8);
    }

    #[test]
    fn null_space_detects_rank_deficiency() {
        let matrix = Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 1.0, 1.0]).unwrap();
        let basis = ndarray_svd::null_space(&matrix, Some(1e-10)).unwrap();
        assert_eq!(basis.ncols(), 1);
        let residual = matrix.dot(&basis.column(0).to_owned());
        assert!(residual.iter().all(|value| value.abs() < 1e-6));
    }
}
