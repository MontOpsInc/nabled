//! Vector-first primitives for embedding-style workloads.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_complex::Complex64;
use thiserror::Error;

/// Errors for vector primitives.
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum VectorError {
    /// Input vectors/matrices are empty.
    #[error("input cannot be empty")]
    EmptyInput,
    /// Input dimensions do not match required shapes.
    #[error("input dimensions are incompatible")]
    DimensionMismatch,
    /// Cosine similarity is undefined for zero-norm vectors.
    #[error("cosine similarity is undefined for zero-norm vectors")]
    ZeroNorm,
}

/// Reusable workspace for pairwise cosine similarity kernels.
#[derive(Debug, Clone, Default)]
pub struct PairwiseCosineWorkspace {
    left_norms:  Array1<f64>,
    right_norms: Array1<f64>,
}

impl PairwiseCosineWorkspace {
    /// Ensure workspace vectors are sized for `left` and `right` row counts.
    fn ensure_dims(&mut self, left_rows: usize, right_rows: usize) {
        if self.left_norms.len() != left_rows {
            self.left_norms = Array1::<f64>::zeros(left_rows);
        }
        if self.right_norms.len() != right_rows {
            self.right_norms = Array1::<f64>::zeros(right_rows);
        }
    }
}

fn validate_vector_pair(a: &Array1<f64>, b: &Array1<f64>) -> Result<(), VectorError> {
    if a.is_empty() || b.is_empty() {
        return Err(VectorError::EmptyInput);
    }
    if a.len() != b.len() {
        return Err(VectorError::DimensionMismatch);
    }
    Ok(())
}

fn validate_pairwise_inputs(left: &Array2<f64>, right: &Array2<f64>) -> Result<(), VectorError> {
    if left.is_empty() || right.is_empty() {
        return Err(VectorError::EmptyInput);
    }
    if left.ncols() != right.ncols() {
        return Err(VectorError::DimensionMismatch);
    }
    Ok(())
}

/// Compute dot product of two vectors.
///
/// # Errors
/// Returns an error when vector lengths mismatch or either input is empty.
pub fn dot(a: &Array1<f64>, b: &Array1<f64>) -> Result<f64, VectorError> {
    validate_vector_pair(a, b)?;
    Ok(a.iter().zip(b.iter()).map(|(x, y)| x * y).sum())
}

/// Compute dot product of two vector views.
///
/// # Errors
/// Returns an error when vector lengths mismatch or either input is empty.
pub fn dot_view(a: &ArrayView1<'_, f64>, b: &ArrayView1<'_, f64>) -> Result<f64, VectorError> {
    if a.is_empty() || b.is_empty() {
        return Err(VectorError::EmptyInput);
    }
    if a.len() != b.len() {
        return Err(VectorError::DimensionMismatch);
    }
    Ok(a.iter().zip(b.iter()).map(|(x, y)| x * y).sum())
}

/// Compute Hermitian dot product `a^H b` for complex vectors.
///
/// # Errors
/// Returns an error when vector lengths mismatch or either input is empty.
pub fn dot_hermitian(
    a: &Array1<Complex64>,
    b: &Array1<Complex64>,
) -> Result<Complex64, VectorError> {
    if a.is_empty() || b.is_empty() {
        return Err(VectorError::EmptyInput);
    }
    if a.len() != b.len() {
        return Err(VectorError::DimensionMismatch);
    }
    Ok(a.iter().zip(b.iter()).map(|(x, y)| x.conj() * y).sum())
}

/// Compute L2 norm of a vector.
///
/// # Errors
/// Returns an error if the vector is empty.
pub fn l2_norm(v: &Array1<f64>) -> Result<f64, VectorError> {
    if v.is_empty() {
        return Err(VectorError::EmptyInput);
    }
    Ok(v.iter().map(|value| value * value).sum::<f64>().sqrt())
}

/// Compute L2 norm of a complex vector.
///
/// # Errors
/// Returns an error if the vector is empty.
pub fn l2_norm_complex(v: &Array1<Complex64>) -> Result<f64, VectorError> {
    if v.is_empty() {
        return Err(VectorError::EmptyInput);
    }
    Ok(v.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt())
}

/// Compute cosine similarity of two vectors.
///
/// # Errors
/// Returns an error for invalid dimensions, empty inputs, or zero-norm vectors.
pub fn cosine_similarity(a: &Array1<f64>, b: &Array1<f64>) -> Result<f64, VectorError> {
    let dot_value = dot(a, b)?;
    let norm_a = l2_norm(a)?;
    let norm_b = l2_norm(b)?;
    let denominator = norm_a * norm_b;
    if denominator <= f64::EPSILON {
        return Err(VectorError::ZeroNorm);
    }
    Ok(dot_value / denominator)
}

/// Compute cosine similarity of two vector views.
///
/// # Errors
/// Returns an error for invalid dimensions, empty inputs, or zero-norm vectors.
pub fn cosine_similarity_view(
    a: &ArrayView1<'_, f64>,
    b: &ArrayView1<'_, f64>,
) -> Result<f64, VectorError> {
    let dot_value = dot_view(a, b)?;
    let norm_a = a.iter().map(|value| value * value).sum::<f64>().sqrt();
    let norm_b = b.iter().map(|value| value * value).sum::<f64>().sqrt();
    let denominator = norm_a * norm_b;
    if denominator <= f64::EPSILON {
        return Err(VectorError::ZeroNorm);
    }
    Ok(dot_value / denominator)
}

/// Compute cosine similarity for complex vectors.
///
/// # Errors
/// Returns an error for invalid dimensions, empty inputs, or zero-norm vectors.
pub fn cosine_similarity_complex(
    a: &Array1<Complex64>,
    b: &Array1<Complex64>,
) -> Result<Complex64, VectorError> {
    let dot_value = dot_hermitian(a, b)?;
    let norm_a = l2_norm_complex(a)?;
    let norm_b = l2_norm_complex(b)?;
    let denominator = norm_a * norm_b;
    if denominator <= f64::EPSILON {
        return Err(VectorError::ZeroNorm);
    }
    Ok(dot_value / denominator)
}

/// Compute cosine distance (`1 - cosine_similarity`).
///
/// # Errors
/// Returns an error for invalid dimensions, empty inputs, or zero-norm vectors.
pub fn cosine_distance(a: &Array1<f64>, b: &Array1<f64>) -> Result<f64, VectorError> {
    Ok(1.0 - cosine_similarity(a, b)?)
}

/// Compute pairwise L2 distances between row vectors in `left` and `right`.
///
/// # Errors
/// Returns an error for invalid dimensions or empty inputs.
pub fn pairwise_l2_distance(
    left: &Array2<f64>,
    right: &Array2<f64>,
) -> Result<Array2<f64>, VectorError> {
    let mut output = Array2::<f64>::zeros((left.nrows(), right.nrows()));
    pairwise_l2_distance_into(left, right, &mut output)?;
    Ok(output)
}

/// Compute pairwise L2 distances into `output`.
///
/// # Errors
/// Returns an error for invalid dimensions or empty inputs.
pub fn pairwise_l2_distance_into(
    left: &Array2<f64>,
    right: &Array2<f64>,
    output: &mut Array2<f64>,
) -> Result<(), VectorError> {
    validate_pairwise_inputs(left, right)?;
    if output.dim() != (left.nrows(), right.nrows()) {
        return Err(VectorError::DimensionMismatch);
    }

    for i in 0..left.nrows() {
        for j in 0..right.nrows() {
            let mut sum = 0.0_f64;
            for k in 0..left.ncols() {
                let delta = left[[i, k]] - right[[j, k]];
                sum += delta * delta;
            }
            output[[i, j]] = sum.sqrt();
        }
    }

    Ok(())
}

/// Compute pairwise L2 distances from matrix views into `output`.
///
/// # Errors
/// Returns an error for invalid dimensions or empty inputs.
pub fn pairwise_l2_distance_view_into(
    left: &ArrayView2<'_, f64>,
    right: &ArrayView2<'_, f64>,
    output: &mut Array2<f64>,
) -> Result<(), VectorError> {
    if left.is_empty() || right.is_empty() {
        return Err(VectorError::EmptyInput);
    }
    if left.ncols() != right.ncols() || output.dim() != (left.nrows(), right.nrows()) {
        return Err(VectorError::DimensionMismatch);
    }

    for i in 0..left.nrows() {
        for j in 0..right.nrows() {
            let mut sum = 0.0_f64;
            for k in 0..left.ncols() {
                let delta = left[[i, k]] - right[[j, k]];
                sum += delta * delta;
            }
            output[[i, j]] = sum.sqrt();
        }
    }

    Ok(())
}

/// Compute pairwise cosine similarity between row vectors.
///
/// # Errors
/// Returns an error for invalid dimensions, empty inputs, or zero-norm rows.
pub fn pairwise_cosine_similarity(
    left: &Array2<f64>,
    right: &Array2<f64>,
) -> Result<Array2<f64>, VectorError> {
    let mut output = Array2::<f64>::zeros((left.nrows(), right.nrows()));
    let mut workspace = PairwiseCosineWorkspace::default();
    pairwise_cosine_similarity_with_workspace_into(left, right, &mut output, &mut workspace)?;
    Ok(output)
}

/// Compute pairwise cosine similarity into `output`.
///
/// # Errors
/// Returns an error for invalid dimensions, empty inputs, or zero-norm rows.
pub fn pairwise_cosine_similarity_into(
    left: &Array2<f64>,
    right: &Array2<f64>,
    output: &mut Array2<f64>,
) -> Result<(), VectorError> {
    let mut workspace = PairwiseCosineWorkspace::default();
    pairwise_cosine_similarity_with_workspace_into(left, right, output, &mut workspace)
}

/// Compute pairwise cosine similarity into `output` using reusable `workspace`.
///
/// # Errors
/// Returns an error for invalid dimensions, empty inputs, or zero-norm rows.
pub fn pairwise_cosine_similarity_with_workspace_into(
    left: &Array2<f64>,
    right: &Array2<f64>,
    output: &mut Array2<f64>,
    workspace: &mut PairwiseCosineWorkspace,
) -> Result<(), VectorError> {
    validate_pairwise_inputs(left, right)?;
    if output.dim() != (left.nrows(), right.nrows()) {
        return Err(VectorError::DimensionMismatch);
    }

    workspace.ensure_dims(left.nrows(), right.nrows());

    for i in 0..left.nrows() {
        let mut sq_sum = 0.0_f64;
        for k in 0..left.ncols() {
            let value = left[[i, k]];
            sq_sum += value * value;
        }
        let norm = sq_sum.sqrt();
        if norm <= f64::EPSILON {
            return Err(VectorError::ZeroNorm);
        }
        workspace.left_norms[i] = norm;
    }

    for j in 0..right.nrows() {
        let mut sq_sum = 0.0_f64;
        for k in 0..right.ncols() {
            let value = right[[j, k]];
            sq_sum += value * value;
        }
        let norm = sq_sum.sqrt();
        if norm <= f64::EPSILON {
            return Err(VectorError::ZeroNorm);
        }
        workspace.right_norms[j] = norm;
    }

    for i in 0..left.nrows() {
        for j in 0..right.nrows() {
            let mut dot = 0.0_f64;
            for k in 0..left.ncols() {
                dot += left[[i, k]] * right[[j, k]];
            }
            output[[i, j]] = dot / (workspace.left_norms[i] * workspace.right_norms[j]);
        }
    }

    Ok(())
}

/// Compute pairwise cosine distances between row vectors.
///
/// # Errors
/// Returns an error for invalid dimensions, empty inputs, or zero-norm rows.
pub fn pairwise_cosine_distance(
    left: &Array2<f64>,
    right: &Array2<f64>,
) -> Result<Array2<f64>, VectorError> {
    let similarity = pairwise_cosine_similarity(left, right)?;
    Ok(similarity.mapv(|value| 1.0 - value))
}

/// Compute row-wise dot products for two matrices of equal shape.
///
/// # Errors
/// Returns an error for invalid dimensions or empty inputs.
pub fn batched_dot(left: &Array2<f64>, right: &Array2<f64>) -> Result<Array1<f64>, VectorError> {
    let mut output = Array1::<f64>::zeros(left.nrows());
    batched_dot_into(left, right, &mut output)?;
    Ok(output)
}

/// Compute row-wise dot products into `output`.
///
/// # Errors
/// Returns an error for invalid dimensions or empty inputs.
pub fn batched_dot_into(
    left: &Array2<f64>,
    right: &Array2<f64>,
    output: &mut Array1<f64>,
) -> Result<(), VectorError> {
    if left.is_empty() || right.is_empty() {
        return Err(VectorError::EmptyInput);
    }
    if left.dim() != right.dim() || output.len() != left.nrows() {
        return Err(VectorError::DimensionMismatch);
    }

    for i in 0..left.nrows() {
        let mut sum = 0.0_f64;
        for j in 0..left.ncols() {
            sum += left[[i, j]] * right[[i, j]];
        }
        output[i] = sum;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use ndarray::{arr1, arr2};
    use num_complex::Complex64;

    use super::*;

    #[test]
    fn dot_and_norm_are_correct() {
        let a = arr1(&[1.0_f64, 2.0, 3.0]);
        let b = arr1(&[4.0_f64, 5.0, 6.0]);

        let dot = dot(&a, &b).unwrap();
        let norm = l2_norm(&a).unwrap();

        assert!((dot - 32.0).abs() < 1e-12);
        assert!((norm - 14.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn cosine_similarity_works() {
        let a = arr1(&[1.0_f64, 0.0]);
        let b = arr1(&[0.0_f64, 1.0]);
        let similarity = cosine_similarity(&a, &b).unwrap();
        let distance = cosine_distance(&a, &b).unwrap();

        assert!(similarity.abs() < 1e-12);
        assert!((distance - 1.0).abs() < 1e-12);
    }

    #[test]
    fn pairwise_l2_distance_matches_expected() {
        let left = arr2(&[[0.0_f64, 0.0], [1.0, 1.0]]);
        let right = arr2(&[[1.0_f64, 0.0], [2.0, 2.0]]);
        let distance = pairwise_l2_distance(&left, &right).unwrap();

        assert!((distance[[0, 0]] - 1.0).abs() < 1e-12);
        assert!((distance[[1, 1]] - 2.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn pairwise_cosine_workspace_reuse() {
        let left = arr2(&[[1.0_f64, 0.0], [1.0, 1.0]]);
        let right = arr2(&[[1.0_f64, 0.0], [0.0, 1.0]]);
        let mut output = Array2::<f64>::zeros((left.nrows(), right.nrows()));
        let mut workspace = PairwiseCosineWorkspace::default();

        pairwise_cosine_similarity_with_workspace_into(&left, &right, &mut output, &mut workspace)
            .unwrap();

        assert!((output[[0, 0]] - 1.0).abs() < 1e-12);
        assert!((output[[0, 1]] - 0.0).abs() < 1e-12);
        assert!((output[[1, 0]] - (2.0_f64).sqrt() / 2.0).abs() < 1e-12);
    }

    #[test]
    fn errors_on_bad_inputs() {
        let empty = arr1(&[] as &[f64]);
        let v = arr1(&[1.0_f64]);
        assert!(matches!(dot(&empty, &v), Err(VectorError::EmptyInput)));

        let a = arr2(&[[1.0_f64, 2.0]]);
        let b = arr2(&[[1.0_f64], [2.0]]);
        assert!(matches!(pairwise_l2_distance(&a, &b), Err(VectorError::DimensionMismatch)));
    }

    #[test]
    fn complex_dot_and_cosine_work() {
        let a = arr1(&[Complex64::new(1.0, 1.0), Complex64::new(2.0, -1.0)]);
        let b = arr1(&[Complex64::new(0.5, -0.5), Complex64::new(-1.0, 3.0)]);

        let dot = dot_hermitian(&a, &b).unwrap();
        let cosine = cosine_similarity_complex(&a, &b).unwrap();

        assert!(dot.norm() > 0.0);
        assert!(cosine.norm() <= 1.0 + 1e-12);
    }

    #[test]
    fn view_first_apis_match_owned() {
        let a = arr1(&[1.0_f64, 2.0, 3.0]);
        let b = arr1(&[4.0_f64, 5.0, 6.0]);
        let dot_owned = dot(&a, &b).unwrap();
        let a_view = a.view();
        let b_view = b.view();
        let dot_view = dot_view(&a_view, &b_view).unwrap();
        assert!((dot_owned - dot_view).abs() < 1e-12);

        let left = arr2(&[[0.0_f64, 0.0], [1.0, 1.0]]);
        let right = arr2(&[[1.0_f64, 0.0], [2.0, 2.0]]);
        let mut output = Array2::<f64>::zeros((2, 2));
        let left_view = left.view();
        let right_view = right.view();
        pairwise_l2_distance_view_into(&left_view, &right_view, &mut output).unwrap();
        let expected = pairwise_l2_distance(&left, &right).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!((output[[i, j]] - expected[[i, j]]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn cosine_view_matches_owned_and_batched_dot_into_works() {
        let a = arr1(&[1.0_f64, 2.0, 3.0]);
        let b = arr1(&[4.0_f64, 5.0, 6.0]);
        let owned = cosine_similarity(&a, &b).unwrap();
        let a_view = a.view();
        let b_view = b.view();
        let viewed = cosine_similarity_view(&a_view, &b_view).unwrap();
        assert!((owned - viewed).abs() < 1e-12);

        let left = arr2(&[[1.0_f64, 2.0], [3.0, 4.0]]);
        let right = arr2(&[[5.0_f64, 6.0], [7.0, 8.0]]);
        let mut out = Array1::<f64>::zeros(2);
        batched_dot_into(&left, &right, &mut out).unwrap();
        assert!((out[0] - 17.0).abs() < 1e-12);
        assert!((out[1] - 53.0).abs() < 1e-12);
    }

    #[test]
    fn pairwise_cosine_into_matches_allocating_path() {
        let left = arr2(&[[1.0_f64, 0.0], [0.0, 1.0]]);
        let right = arr2(&[[1.0_f64, 0.0], [1.0, 1.0]]);
        let expected = pairwise_cosine_similarity(&left, &right).unwrap();
        let mut output = Array2::<f64>::zeros((2, 2));
        pairwise_cosine_similarity_into(&left, &right, &mut output).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!((output[[i, j]] - expected[[i, j]]).abs() < 1e-12);
            }
        }
    }
}
