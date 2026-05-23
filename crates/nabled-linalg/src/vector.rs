//! Vector-first primitives for embedding-style workloads.

use nabled_core::scalar::NabledReal;
use ndarray::{Array1, Array2, ArrayBase, ArrayView1, ArrayView2, Axis, Data, DataMut, Ix1, Ix2};
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
#[derive(Debug, Clone)]
pub struct PairwiseCosineWorkspace<T: NabledReal> {
    left_norms: Array1<T>,
    right_norms: Array1<T>,
}

impl<T: NabledReal> Default for PairwiseCosineWorkspace<T> {
    fn default() -> Self {
        Self { left_norms: Array1::<T>::zeros(0), right_norms: Array1::<T>::zeros(0) }
    }
}

impl<T: NabledReal> PairwiseCosineWorkspace<T> {
    /// Ensure workspace vectors are sized for `left` and `right` row counts.
    fn ensure_dims(&mut self, left_rows: usize, right_rows: usize) {
        if self.left_norms.len() != left_rows {
            self.left_norms = Array1::<T>::zeros(left_rows);
        }
        if self.right_norms.len() != right_rows {
            self.right_norms = Array1::<T>::zeros(right_rows);
        }
    }
}

fn validate_vector_pair<T: NabledReal>(a: &Array1<T>, b: &Array1<T>) -> Result<(), VectorError> {
    if a.is_empty() || b.is_empty() {
        return Err(VectorError::EmptyInput);
    }
    if a.len() != b.len() {
        return Err(VectorError::DimensionMismatch);
    }
    Ok(())
}

fn validate_pairwise_inputs<T, S1, S2>(
    left: &ArrayBase<S1, Ix2>,
    right: &ArrayBase<S2, Ix2>,
) -> Result<(), VectorError>
where
    T: NabledReal,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
{
    if left.is_empty() || right.is_empty() {
        return Err(VectorError::EmptyInput);
    }
    if left.ncols() != right.ncols() {
        return Err(VectorError::DimensionMismatch);
    }
    Ok(())
}

fn validate_batched_row_inputs<T: NabledReal>(rows: &ArrayView2<'_, T>) -> Result<(), VectorError> {
    if rows.is_empty() {
        return Err(VectorError::EmptyInput);
    }
    Ok(())
}

fn validate_batched_row_pair_inputs<T: NabledReal>(
    left: &ArrayView2<'_, T>,
    right: &ArrayView2<'_, T>,
) -> Result<(), VectorError> {
    validate_batched_row_inputs(left)?;
    validate_batched_row_inputs(right)?;
    if left.dim() != right.dim() {
        return Err(VectorError::DimensionMismatch);
    }
    Ok(())
}

/// Compute dot product of two vectors.
///
/// # Errors
/// Returns an error when vector lengths mismatch or either input is empty.
pub fn dot<T: NabledReal>(a: &Array1<T>, b: &Array1<T>) -> Result<T, VectorError> {
    validate_vector_pair(a, b)?;
    Ok(a.dot(b))
}

/// Compute dot product of two vector views.
///
/// # Errors
/// Returns an error when vector lengths mismatch or either input is empty.
pub fn dot_view<T: NabledReal>(
    a: &ArrayView1<'_, T>,
    b: &ArrayView1<'_, T>,
) -> Result<T, VectorError> {
    if a.is_empty() || b.is_empty() {
        return Err(VectorError::EmptyInput);
    }
    if a.len() != b.len() {
        return Err(VectorError::DimensionMismatch);
    }
    Ok(a.dot(b))
}

/// Compute Hermitian dot product `a^H b` for complex vectors.
///
/// # Errors
/// Returns an error when vector lengths mismatch or either input is empty.
pub fn dot_hermitian(
    a: &Array1<Complex64>,
    b: &Array1<Complex64>,
) -> Result<Complex64, VectorError> {
    dot_hermitian_view(&a.view(), &b.view())
}

/// Compute Hermitian dot product `a^H b` for complex vector views.
///
/// # Errors
/// Returns an error when vector lengths mismatch or either input is empty.
pub fn dot_hermitian_view(
    a: &ArrayView1<'_, Complex64>,
    b: &ArrayView1<'_, Complex64>,
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
pub fn l2_norm<T: NabledReal>(v: &Array1<T>) -> Result<T, VectorError> {
    l2_norm_view(&v.view())
}

/// Compute L2 norm of a vector view.
///
/// # Errors
/// Returns an error if the vector is empty.
pub fn l2_norm_view<T: NabledReal>(v: &ArrayView1<'_, T>) -> Result<T, VectorError> {
    if v.is_empty() {
        return Err(VectorError::EmptyInput);
    }
    Ok(v.dot(v).sqrt())
}

/// Compute L2 norm of a complex vector.
///
/// # Errors
/// Returns an error if the vector is empty.
pub fn l2_norm_complex(v: &Array1<Complex64>) -> Result<f64, VectorError> {
    l2_norm_complex_view(&v.view())
}

/// Compute L2 norm of a complex vector view.
///
/// # Errors
/// Returns an error if the vector is empty.
pub fn l2_norm_complex_view(v: &ArrayView1<'_, Complex64>) -> Result<f64, VectorError> {
    if v.is_empty() {
        return Err(VectorError::EmptyInput);
    }
    Ok(v.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt())
}

/// Compute cosine similarity of two vectors.
///
/// # Errors
/// Returns an error for invalid dimensions, empty inputs, or zero-norm vectors.
pub fn cosine_similarity<T: NabledReal>(a: &Array1<T>, b: &Array1<T>) -> Result<T, VectorError> {
    validate_vector_pair(a, b)?;
    let dot_value = a.dot(b);
    let denominator = a.dot(a).sqrt() * b.dot(b).sqrt();
    if denominator <= T::epsilon() {
        return Err(VectorError::ZeroNorm);
    }
    Ok(dot_value / denominator)
}

/// Compute cosine similarity of two vector views.
///
/// # Errors
/// Returns an error for invalid dimensions, empty inputs, or zero-norm vectors.
pub fn cosine_similarity_view<T: NabledReal>(
    a: &ArrayView1<'_, T>,
    b: &ArrayView1<'_, T>,
) -> Result<T, VectorError> {
    if a.is_empty() || b.is_empty() {
        return Err(VectorError::EmptyInput);
    }
    if a.len() != b.len() {
        return Err(VectorError::DimensionMismatch);
    }
    let dot_value = a.dot(b);
    let denominator = a.dot(a).sqrt() * b.dot(b).sqrt();
    if denominator <= T::epsilon() {
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
    cosine_similarity_complex_view(&a.view(), &b.view())
}

/// Compute cosine similarity for complex vector views.
///
/// # Errors
/// Returns an error for invalid dimensions, empty inputs, or zero-norm vectors.
pub fn cosine_similarity_complex_view(
    a: &ArrayView1<'_, Complex64>,
    b: &ArrayView1<'_, Complex64>,
) -> Result<Complex64, VectorError> {
    let dot_value = dot_hermitian_view(a, b)?;
    let norm_a = l2_norm_complex_view(a)?;
    let norm_b = l2_norm_complex_view(b)?;
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
pub fn cosine_distance<T: NabledReal>(a: &Array1<T>, b: &Array1<T>) -> Result<T, VectorError> {
    Ok(T::one() - cosine_similarity(a, b)?)
}

/// Compute cosine distance (`1 - cosine_similarity`) from vector views.
///
/// # Errors
/// Returns an error for invalid dimensions, empty inputs, or zero-norm vectors.
pub fn cosine_distance_view<T: NabledReal>(
    a: &ArrayView1<'_, T>,
    b: &ArrayView1<'_, T>,
) -> Result<T, VectorError> {
    Ok(T::one() - cosine_similarity_view(a, b)?)
}

/// Compute pairwise L2 distances between row vectors in `left` and `right`.
///
/// # Errors
/// Returns an error for invalid dimensions or empty inputs.
pub fn pairwise_l2_distance<T: NabledReal>(
    left: &Array2<T>,
    right: &Array2<T>,
) -> Result<Array2<T>, VectorError> {
    validate_pairwise_inputs(left, right)?;
    let mut output = Array2::<T>::zeros((left.nrows(), right.nrows()));
    pairwise_l2_distance_view_into(&left.view(), &right.view(), &mut output)?;
    Ok(output)
}

/// Compute pairwise L2 distances between row vectors from matrix views.
///
/// # Errors
/// Returns an error for invalid dimensions or empty inputs.
pub fn pairwise_l2_distance_view<T: NabledReal>(
    left: &ArrayView2<'_, T>,
    right: &ArrayView2<'_, T>,
) -> Result<Array2<T>, VectorError> {
    if left.is_empty() || right.is_empty() {
        return Err(VectorError::EmptyInput);
    }
    if left.ncols() != right.ncols() {
        return Err(VectorError::DimensionMismatch);
    }

    let mut output = Array2::<T>::zeros((left.nrows(), right.nrows()));
    pairwise_l2_distance_view_into(left, right, &mut output)?;
    Ok(output)
}

/// Compute pairwise L2 distances into `output`.
///
/// # Errors
/// Returns an error for invalid dimensions or empty inputs.
pub fn pairwise_l2_distance_into<T, S1, S2>(
    left: &ArrayBase<S1, Ix2>,
    right: &ArrayBase<S2, Ix2>,
    output: &mut ArrayBase<impl DataMut<Elem = T>, Ix2>,
) -> Result<(), VectorError>
where
    T: NabledReal,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
{
    pairwise_l2_distance_view_into(&left.view(), &right.view(), output)
}

/// Compute pairwise L2 distances from matrix views into `output`.
///
/// # Errors
/// Returns an error for invalid dimensions or empty inputs.
pub fn pairwise_l2_distance_view_into<T: NabledReal>(
    left: &ArrayView2<'_, T>,
    right: &ArrayView2<'_, T>,
    output: &mut ArrayBase<impl DataMut<Elem = T>, Ix2>,
) -> Result<(), VectorError> {
    if left.is_empty() || right.is_empty() {
        return Err(VectorError::EmptyInput);
    }
    if left.ncols() != right.ncols() || output.dim() != (left.nrows(), right.nrows()) {
        return Err(VectorError::DimensionMismatch);
    }

    for i in 0..left.nrows() {
        for j in 0..right.nrows() {
            let mut sum = T::zero();
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
pub fn pairwise_cosine_similarity<T: NabledReal>(
    left: &Array2<T>,
    right: &Array2<T>,
) -> Result<Array2<T>, VectorError> {
    validate_pairwise_inputs(left, right)?;
    let mut output = Array2::<T>::zeros((left.nrows(), right.nrows()));
    pairwise_cosine_similarity_into(left, right, &mut output)?;
    Ok(output)
}

/// Compute pairwise cosine similarity between row vectors from matrix views.
///
/// # Errors
/// Returns an error for invalid dimensions, empty inputs, or zero-norm rows.
pub fn pairwise_cosine_similarity_view<T: NabledReal>(
    left: &ArrayView2<'_, T>,
    right: &ArrayView2<'_, T>,
) -> Result<Array2<T>, VectorError> {
    if left.is_empty() || right.is_empty() {
        return Err(VectorError::EmptyInput);
    }
    if left.ncols() != right.ncols() {
        return Err(VectorError::DimensionMismatch);
    }

    let mut left_norms = Array1::<T>::zeros(left.nrows());
    let mut right_norms = Array1::<T>::zeros(right.nrows());
    let mut output = Array2::<T>::zeros((left.nrows(), right.nrows()));

    for i in 0..left.nrows() {
        let mut sq_sum = T::zero();
        for k in 0..left.ncols() {
            let value = left[[i, k]];
            sq_sum += value * value;
        }
        let norm = sq_sum.sqrt();
        if norm <= T::epsilon() {
            return Err(VectorError::ZeroNorm);
        }
        left_norms[i] = norm;
    }

    for j in 0..right.nrows() {
        let mut sq_sum = T::zero();
        for k in 0..right.ncols() {
            let value = right[[j, k]];
            sq_sum += value * value;
        }
        let norm = sq_sum.sqrt();
        if norm <= T::epsilon() {
            return Err(VectorError::ZeroNorm);
        }
        right_norms[j] = norm;
    }

    for i in 0..left.nrows() {
        for j in 0..right.nrows() {
            let mut dot = T::zero();
            for k in 0..left.ncols() {
                dot += left[[i, k]] * right[[j, k]];
            }
            output[[i, j]] = dot / (left_norms[i] * right_norms[j]);
        }
    }

    Ok(output)
}

/// Compute pairwise cosine similarity into `output`.
///
/// # Errors
/// Returns an error for invalid dimensions, empty inputs, or zero-norm rows.
pub fn pairwise_cosine_similarity_into<T, S1, S2>(
    left: &ArrayBase<S1, Ix2>,
    right: &ArrayBase<S2, Ix2>,
    output: &mut ArrayBase<impl DataMut<Elem = T>, Ix2>,
) -> Result<(), VectorError>
where
    T: NabledReal,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
{
    let mut workspace = PairwiseCosineWorkspace::<T>::default();
    pairwise_cosine_similarity_with_workspace_into(left, right, output, &mut workspace)
}

/// Compute pairwise cosine similarity into `output` using reusable `workspace`.
///
/// # Errors
/// Returns an error for invalid dimensions, empty inputs, or zero-norm rows.
pub fn pairwise_cosine_similarity_with_workspace_into<T, S1, S2>(
    left: &ArrayBase<S1, Ix2>,
    right: &ArrayBase<S2, Ix2>,
    output: &mut ArrayBase<impl DataMut<Elem = T>, Ix2>,
    workspace: &mut PairwiseCosineWorkspace<T>,
) -> Result<(), VectorError>
where
    T: NabledReal,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
{
    validate_pairwise_inputs(left, right)?;
    if output.dim() != (left.nrows(), right.nrows()) {
        return Err(VectorError::DimensionMismatch);
    }

    workspace.ensure_dims(left.nrows(), right.nrows());

    for i in 0..left.nrows() {
        let mut sq_sum = T::zero();
        for k in 0..left.ncols() {
            let value = left[[i, k]];
            sq_sum += value * value;
        }
        let norm = sq_sum.sqrt();
        if norm <= T::epsilon() {
            return Err(VectorError::ZeroNorm);
        }
        workspace.left_norms[i] = norm;
    }

    for j in 0..right.nrows() {
        let mut sq_sum = T::zero();
        for k in 0..right.ncols() {
            let value = right[[j, k]];
            sq_sum += value * value;
        }
        let norm = sq_sum.sqrt();
        if norm <= T::epsilon() {
            return Err(VectorError::ZeroNorm);
        }
        workspace.right_norms[j] = norm;
    }

    for i in 0..left.nrows() {
        for j in 0..right.nrows() {
            let mut dot = T::zero();
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
pub fn pairwise_cosine_distance<T: NabledReal>(
    left: &Array2<T>,
    right: &Array2<T>,
) -> Result<Array2<T>, VectorError> {
    let similarity = pairwise_cosine_similarity(left, right)?;
    Ok(similarity.mapv(|value| T::one() - value))
}

/// Compute pairwise cosine distances between row vectors from matrix views.
///
/// # Errors
/// Returns an error for invalid dimensions, empty inputs, or zero-norm rows.
pub fn pairwise_cosine_distance_view<T: NabledReal>(
    left: &ArrayView2<'_, T>,
    right: &ArrayView2<'_, T>,
) -> Result<Array2<T>, VectorError> {
    let similarity = pairwise_cosine_similarity_view(left, right)?;
    Ok(similarity.mapv(|value| T::one() - value))
}

/// Compute pairwise cosine distances into `output`.
///
/// # Errors
/// Returns an error for invalid dimensions, empty inputs, or zero-norm rows.
pub fn pairwise_cosine_distance_into<T, S1, S2>(
    left: &ArrayBase<S1, Ix2>,
    right: &ArrayBase<S2, Ix2>,
    output: &mut ArrayBase<impl DataMut<Elem = T>, Ix2>,
) -> Result<(), VectorError>
where
    T: NabledReal,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
{
    pairwise_cosine_similarity_into(left, right, output)?;
    output.mapv_inplace(|value| T::one() - value);
    Ok(())
}

/// Compute row-wise dot products for two matrices of equal shape.
///
/// # Errors
/// Returns an error for invalid dimensions or empty inputs.
pub fn batched_dot<T: NabledReal>(
    left: &Array2<T>,
    right: &Array2<T>,
) -> Result<Array1<T>, VectorError> {
    let mut output = Array1::<T>::zeros(left.nrows());
    batched_dot_into(left, right, &mut output)?;
    Ok(output)
}

/// Compute row-wise dot products for two matrix views of equal shape.
///
/// # Errors
/// Returns an error for invalid dimensions or empty inputs.
pub fn batched_dot_view<T: NabledReal>(
    left: &ArrayView2<'_, T>,
    right: &ArrayView2<'_, T>,
) -> Result<Array1<T>, VectorError> {
    if left.is_empty() || right.is_empty() {
        return Err(VectorError::EmptyInput);
    }
    if left.dim() != right.dim() {
        return Err(VectorError::DimensionMismatch);
    }

    let mut output = Array1::<T>::zeros(left.nrows());
    if output.len() != left.nrows() {
        return Err(VectorError::DimensionMismatch);
    }

    for i in 0..left.nrows() {
        let mut sum = T::zero();
        for j in 0..left.ncols() {
            sum += left[[i, j]] * right[[i, j]];
        }
        output[i] = sum;
    }

    Ok(output)
}

/// Compute row-wise dot products into `output`.
///
/// # Errors
/// Returns an error for invalid dimensions or empty inputs.
pub fn batched_dot_into<T, S1, S2>(
    left: &ArrayBase<S1, Ix2>,
    right: &ArrayBase<S2, Ix2>,
    output: &mut ArrayBase<impl DataMut<Elem = T>, Ix1>,
) -> Result<(), VectorError>
where
    T: NabledReal,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
{
    if left.is_empty() || right.is_empty() {
        return Err(VectorError::EmptyInput);
    }
    if left.dim() != right.dim() || output.len() != left.nrows() {
        return Err(VectorError::DimensionMismatch);
    }

    for i in 0..left.nrows() {
        let mut sum = T::zero();
        for j in 0..left.ncols() {
            sum += left[[i, j]] * right[[i, j]];
        }
        output[i] = sum;
    }

    Ok(())
}

/// Compute row-wise L2 norms for a matrix interpreted as a batch of vectors.
///
/// # Errors
/// Returns an error for empty input.
pub fn batched_l2_norm<T: NabledReal>(rows: &Array2<T>) -> Result<Array1<T>, VectorError> {
    batched_l2_norm_view(&rows.view())
}

/// Compute row-wise L2 norms for matrix views interpreted as a batch of vectors.
///
/// # Errors
/// Returns an error for empty input.
pub fn batched_l2_norm_view<T: NabledReal>(
    rows: &ArrayView2<'_, T>,
) -> Result<Array1<T>, VectorError> {
    validate_batched_row_inputs(rows)?;

    let mut output = Array1::<T>::zeros(rows.nrows());
    for (index, row) in rows.axis_iter(Axis(0)).enumerate() {
        let sum_sq = row
            .iter()
            .copied()
            .map(|value| value * value)
            .fold(T::zero(), |acc, value| acc + value);
        output[index] = sum_sq.sqrt();
    }
    Ok(output)
}

/// Compute row-wise L2 norms into `output`.
///
/// # Errors
/// Returns an error for empty input or mismatched output dimensions.
pub fn batched_l2_norm_into<T, S>(
    rows: &ArrayBase<S, Ix2>,
    output: &mut ArrayBase<impl DataMut<Elem = T>, Ix1>,
) -> Result<(), VectorError>
where
    T: NabledReal,
    S: Data<Elem = T>,
{
    if rows.is_empty() {
        return Err(VectorError::EmptyInput);
    }
    if output.len() != rows.nrows() {
        return Err(VectorError::DimensionMismatch);
    }

    for (index, row) in rows.axis_iter(Axis(0)).enumerate() {
        let sum_sq = row
            .iter()
            .copied()
            .map(|value| value * value)
            .fold(T::zero(), |acc, value| acc + value);
        output[index] = sum_sq.sqrt();
    }
    Ok(())
}

/// Compute row-wise cosine similarities for paired batches of vectors.
///
/// # Errors
/// Returns an error for empty input, mismatched dimensions, or zero-norm rows.
pub fn batched_cosine_similarity<T: NabledReal>(
    left: &Array2<T>,
    right: &Array2<T>,
) -> Result<Array1<T>, VectorError> {
    batched_cosine_similarity_view(&left.view(), &right.view())
}

/// Compute row-wise cosine similarities for paired matrix views.
///
/// # Errors
/// Returns an error for empty input, mismatched dimensions, or zero-norm rows.
pub fn batched_cosine_similarity_view<T: NabledReal>(
    left: &ArrayView2<'_, T>,
    right: &ArrayView2<'_, T>,
) -> Result<Array1<T>, VectorError> {
    validate_batched_row_pair_inputs(left, right)?;

    let mut output = Array1::<T>::zeros(left.nrows());
    for (index, (left_row, right_row)) in
        left.axis_iter(Axis(0)).zip(right.axis_iter(Axis(0))).enumerate()
    {
        let mut dot_value = T::zero();
        let mut left_sq_sum = T::zero();
        let mut right_sq_sum = T::zero();

        for (lhs, rhs) in left_row.iter().zip(right_row.iter()) {
            dot_value += *lhs * *rhs;
            left_sq_sum += *lhs * *lhs;
            right_sq_sum += *rhs * *rhs;
        }

        let denominator = left_sq_sum.sqrt() * right_sq_sum.sqrt();
        if denominator <= T::epsilon() {
            return Err(VectorError::ZeroNorm);
        }

        output[index] = dot_value / denominator;
    }
    Ok(output)
}

/// Compute row-wise cosine similarities into `output`.
///
/// # Errors
/// Returns an error for empty input, mismatched dimensions, or zero-norm rows.
pub fn batched_cosine_similarity_into<T, S1, S2>(
    left: &ArrayBase<S1, Ix2>,
    right: &ArrayBase<S2, Ix2>,
    output: &mut ArrayBase<impl DataMut<Elem = T>, Ix1>,
) -> Result<(), VectorError>
where
    T: NabledReal,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
{
    if left.is_empty() || right.is_empty() {
        return Err(VectorError::EmptyInput);
    }
    if left.dim() != right.dim() || output.len() != left.nrows() {
        return Err(VectorError::DimensionMismatch);
    }

    for (index, (left_row, right_row)) in
        left.axis_iter(Axis(0)).zip(right.axis_iter(Axis(0))).enumerate()
    {
        let mut dot_value = T::zero();
        let mut left_sq_sum = T::zero();
        let mut right_sq_sum = T::zero();

        for (lhs, rhs) in left_row.iter().zip(right_row.iter()) {
            dot_value += *lhs * *rhs;
            left_sq_sum += *lhs * *lhs;
            right_sq_sum += *rhs * *rhs;
        }

        let denominator = left_sq_sum.sqrt() * right_sq_sum.sqrt();
        if denominator <= T::epsilon() {
            return Err(VectorError::ZeroNorm);
        }

        output[index] = dot_value / denominator;
    }
    Ok(())
}

/// Compute row-wise cosine distances for paired batches of vectors.
///
/// # Errors
/// Returns an error for empty input, mismatched dimensions, or zero-norm rows.
pub fn batched_cosine_distance<T: NabledReal>(
    left: &Array2<T>,
    right: &Array2<T>,
) -> Result<Array1<T>, VectorError> {
    let similarity = batched_cosine_similarity(left, right)?;
    Ok(similarity.mapv(|value| T::one() - value))
}

/// Compute row-wise cosine distances for paired matrix views.
///
/// # Errors
/// Returns an error for empty input, mismatched dimensions, or zero-norm rows.
pub fn batched_cosine_distance_view<T: NabledReal>(
    left: &ArrayView2<'_, T>,
    right: &ArrayView2<'_, T>,
) -> Result<Array1<T>, VectorError> {
    let similarity = batched_cosine_similarity_view(left, right)?;
    Ok(similarity.mapv(|value| T::one() - value))
}

/// Compute row-wise cosine distances into `output`.
///
/// # Errors
/// Returns an error for empty input, mismatched dimensions, or zero-norm rows.
pub fn batched_cosine_distance_into<T, S1, S2>(
    left: &ArrayBase<S1, Ix2>,
    right: &ArrayBase<S2, Ix2>,
    output: &mut ArrayBase<impl DataMut<Elem = T>, Ix1>,
) -> Result<(), VectorError>
where
    T: NabledReal,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
{
    batched_cosine_similarity_into(left, right, output)?;
    output.mapv_inplace(|value| T::one() - value);
    Ok(())
}

/// Normalize each row in a matrix interpreted as a batch of vectors.
///
/// # Errors
/// Returns an error for empty input.
pub fn batched_normalize<T: NabledReal>(rows: &Array2<T>) -> Result<Array2<T>, VectorError> {
    batched_normalize_view(&rows.view())
}

/// Normalize each row in a matrix view interpreted as a batch of vectors.
///
/// # Errors
/// Returns an error for empty input.
pub fn batched_normalize_view<T: NabledReal>(
    rows: &ArrayView2<'_, T>,
) -> Result<Array2<T>, VectorError> {
    validate_batched_row_inputs(rows)?;

    let mut output = rows.to_owned();
    for mut row in output.axis_iter_mut(Axis(0)) {
        let norm = row
            .iter()
            .copied()
            .map(|value| value * value)
            .fold(T::zero(), |acc, value| acc + value)
            .sqrt();
        let denominator = norm.max(T::epsilon());
        for value in &mut row {
            *value /= denominator;
        }
    }
    Ok(output)
}

/// Normalize each row into `output`.
///
/// # Errors
/// Returns an error for empty input or mismatched output dimensions.
pub fn batched_normalize_into<T, S>(
    rows: &ArrayBase<S, Ix2>,
    output: &mut ArrayBase<impl DataMut<Elem = T>, Ix2>,
) -> Result<(), VectorError>
where
    T: NabledReal,
    S: Data<Elem = T>,
{
    if rows.is_empty() {
        return Err(VectorError::EmptyInput);
    }
    if output.dim() != rows.dim() {
        return Err(VectorError::DimensionMismatch);
    }

    output.assign(rows);
    for mut row in output.axis_iter_mut(Axis(0)) {
        let norm = row
            .iter()
            .copied()
            .map(|value| value * value)
            .fold(T::zero(), |acc, value| acc + value)
            .sqrt();
        let denominator = norm.max(T::epsilon());
        for value in &mut row {
            *value /= denominator;
        }
    }
    Ok(())
}

/// Compute row-wise Hermitian dot products for paired batches of complex vectors.
///
/// # Errors
/// Returns an error for empty input or mismatched dimensions.
pub fn batched_dot_hermitian(
    left: &Array2<Complex64>,
    right: &Array2<Complex64>,
) -> Result<Array1<Complex64>, VectorError> {
    batched_dot_hermitian_view(&left.view(), &right.view())
}

/// Compute row-wise Hermitian dot products for paired complex matrix views.
///
/// # Errors
/// Returns an error for empty input or mismatched dimensions.
pub fn batched_dot_hermitian_view(
    left: &ArrayView2<'_, Complex64>,
    right: &ArrayView2<'_, Complex64>,
) -> Result<Array1<Complex64>, VectorError> {
    if left.is_empty() || right.is_empty() {
        return Err(VectorError::EmptyInput);
    }
    if left.dim() != right.dim() {
        return Err(VectorError::DimensionMismatch);
    }

    let mut output = Array1::<Complex64>::zeros(left.nrows());
    for (index, (left_row, right_row)) in
        left.axis_iter(Axis(0)).zip(right.axis_iter(Axis(0))).enumerate()
    {
        output[index] =
            left_row.iter().zip(right_row.iter()).map(|(lhs, rhs)| lhs.conj() * *rhs).sum();
    }
    Ok(output)
}

/// Compute row-wise Hermitian dot products into `output`.
///
/// # Errors
/// Returns an error for empty input or mismatched dimensions.
pub fn batched_dot_hermitian_into<S1, S2>(
    left: &ArrayBase<S1, Ix2>,
    right: &ArrayBase<S2, Ix2>,
    output: &mut ArrayBase<impl DataMut<Elem = Complex64>, Ix1>,
) -> Result<(), VectorError>
where
    S1: Data<Elem = Complex64>,
    S2: Data<Elem = Complex64>,
{
    if left.is_empty() || right.is_empty() {
        return Err(VectorError::EmptyInput);
    }
    if left.dim() != right.dim() || output.len() != left.nrows() {
        return Err(VectorError::DimensionMismatch);
    }

    for (index, (left_row, right_row)) in
        left.axis_iter(Axis(0)).zip(right.axis_iter(Axis(0))).enumerate()
    {
        output[index] =
            left_row.iter().zip(right_row.iter()).map(|(lhs, rhs)| lhs.conj() * *rhs).sum();
    }
    Ok(())
}

/// Compute row-wise complex L2 norms for a matrix interpreted as a batch of vectors.
///
/// # Errors
/// Returns an error for empty input.
pub fn batched_l2_norm_complex(rows: &Array2<Complex64>) -> Result<Array1<f64>, VectorError> {
    batched_l2_norm_complex_view(&rows.view())
}

/// Compute row-wise complex L2 norms for matrix views interpreted as a batch of vectors.
///
/// # Errors
/// Returns an error for empty input.
pub fn batched_l2_norm_complex_view(
    rows: &ArrayView2<'_, Complex64>,
) -> Result<Array1<f64>, VectorError> {
    if rows.is_empty() {
        return Err(VectorError::EmptyInput);
    }

    let mut output = Array1::<f64>::zeros(rows.nrows());
    for (index, row) in rows.axis_iter(Axis(0)).enumerate() {
        output[index] = row.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt();
    }
    Ok(output)
}

/// Compute row-wise complex L2 norms into `output`.
///
/// # Errors
/// Returns an error for empty input or mismatched output dimensions.
pub fn batched_l2_norm_complex_into<S>(
    rows: &ArrayBase<S, Ix2>,
    output: &mut ArrayBase<impl DataMut<Elem = f64>, Ix1>,
) -> Result<(), VectorError>
where
    S: Data<Elem = Complex64>,
{
    if rows.is_empty() {
        return Err(VectorError::EmptyInput);
    }
    if output.len() != rows.nrows() {
        return Err(VectorError::DimensionMismatch);
    }

    for (index, row) in rows.axis_iter(Axis(0)).enumerate() {
        output[index] = row.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt();
    }
    Ok(())
}

/// Compute row-wise complex cosine similarities for paired batches of vectors.
///
/// # Errors
/// Returns an error for empty input, mismatched dimensions, or zero-norm rows.
pub fn batched_cosine_similarity_complex(
    left: &Array2<Complex64>,
    right: &Array2<Complex64>,
) -> Result<Array1<Complex64>, VectorError> {
    batched_cosine_similarity_complex_view(&left.view(), &right.view())
}

/// Compute row-wise complex cosine similarities for paired complex matrix views.
///
/// # Errors
/// Returns an error for empty input, mismatched dimensions, or zero-norm rows.
pub fn batched_cosine_similarity_complex_view(
    left: &ArrayView2<'_, Complex64>,
    right: &ArrayView2<'_, Complex64>,
) -> Result<Array1<Complex64>, VectorError> {
    if left.is_empty() || right.is_empty() {
        return Err(VectorError::EmptyInput);
    }
    if left.dim() != right.dim() {
        return Err(VectorError::DimensionMismatch);
    }

    let mut output = Array1::<Complex64>::zeros(left.nrows());
    for (index, (left_row, right_row)) in
        left.axis_iter(Axis(0)).zip(right.axis_iter(Axis(0))).enumerate()
    {
        let dot_value: Complex64 =
            left_row.iter().zip(right_row.iter()).map(|(lhs, rhs)| lhs.conj() * *rhs).sum();
        let left_norm = left_row.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt();
        let right_norm = right_row.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt();
        let denominator = left_norm * right_norm;
        if denominator <= f64::EPSILON {
            return Err(VectorError::ZeroNorm);
        }
        output[index] = dot_value / denominator;
    }
    Ok(output)
}

/// Compute row-wise complex cosine similarities into `output`.
///
/// # Errors
/// Returns an error for empty input, mismatched dimensions, or zero-norm rows.
pub fn batched_cosine_similarity_complex_into<S1, S2>(
    left: &ArrayBase<S1, Ix2>,
    right: &ArrayBase<S2, Ix2>,
    output: &mut ArrayBase<impl DataMut<Elem = Complex64>, Ix1>,
) -> Result<(), VectorError>
where
    S1: Data<Elem = Complex64>,
    S2: Data<Elem = Complex64>,
{
    if left.is_empty() || right.is_empty() {
        return Err(VectorError::EmptyInput);
    }
    if left.dim() != right.dim() || output.len() != left.nrows() {
        return Err(VectorError::DimensionMismatch);
    }

    for (index, (left_row, right_row)) in
        left.axis_iter(Axis(0)).zip(right.axis_iter(Axis(0))).enumerate()
    {
        let dot_value: Complex64 =
            left_row.iter().zip(right_row.iter()).map(|(lhs, rhs)| lhs.conj() * *rhs).sum();
        let left_norm = left_row.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt();
        let right_norm = right_row.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt();
        let denominator = left_norm * right_norm;
        if denominator <= f64::EPSILON {
            return Err(VectorError::ZeroNorm);
        }
        output[index] = dot_value / denominator;
    }
    Ok(())
}

/// Normalize each row in a complex matrix interpreted as a batch of vectors.
///
/// # Errors
/// Returns an error for empty input.
pub fn batched_normalize_complex(
    rows: &Array2<Complex64>,
) -> Result<Array2<Complex64>, VectorError> {
    batched_normalize_complex_view(&rows.view())
}

/// Normalize each row in a complex matrix view interpreted as a batch of vectors.
///
/// # Errors
/// Returns an error for empty input.
pub fn batched_normalize_complex_view(
    rows: &ArrayView2<'_, Complex64>,
) -> Result<Array2<Complex64>, VectorError> {
    if rows.is_empty() {
        return Err(VectorError::EmptyInput);
    }

    let mut output = rows.to_owned();
    for mut row in output.axis_iter_mut(Axis(0)) {
        let norm = row.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt();
        let denominator = norm.max(f64::EPSILON);
        for value in &mut row {
            *value /= denominator;
        }
    }
    Ok(output)
}

/// Normalize each complex row into `output`.
///
/// # Errors
/// Returns an error for empty input or mismatched output dimensions.
pub fn batched_normalize_complex_into<S>(
    rows: &ArrayBase<S, Ix2>,
    output: &mut ArrayBase<impl DataMut<Elem = Complex64>, Ix2>,
) -> Result<(), VectorError>
where
    S: Data<Elem = Complex64>,
{
    if rows.is_empty() {
        return Err(VectorError::EmptyInput);
    }
    if output.dim() != rows.dim() {
        return Err(VectorError::DimensionMismatch);
    }

    output.assign(rows);
    for mut row in output.axis_iter_mut(Axis(0)) {
        let norm = row.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt();
        let denominator = norm.max(f64::EPSILON);
        for value in &mut row {
            *value /= denominator;
        }
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
    fn dot_and_norm_support_f32() {
        let a = arr1(&[1.0_f32, 2.0, 3.0]);
        let b = arr1(&[4.0_f32, 5.0, 6.0]);

        let dot = dot(&a, &b).unwrap();
        let norm = l2_norm(&a).unwrap();

        assert!((dot - 32.0).abs() < 1e-5);
        assert!((norm - 14.0_f32.sqrt()).abs() < 1e-5);
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
        let dot_viewed = dot_view(&a_view, &b_view).unwrap();
        assert!((dot_owned - dot_viewed).abs() < 1e-12);

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

    #[test]
    fn batched_rowwise_real_vector_kernels_match_expected() {
        let left = arr2(&[[3.0_f64, 4.0], [1.0, 1.0]]);
        let right = arr2(&[[4.0_f64, 3.0], [1.0, -1.0]]);

        let norms = batched_l2_norm(&left).unwrap();
        let cosine = batched_cosine_similarity(&left, &right).unwrap();
        let distance = batched_cosine_distance(&left, &right).unwrap();
        let normalized = batched_normalize(&left).unwrap();

        assert!((norms[0] - 5.0).abs() < 1e-12);
        assert!((norms[1] - (2.0_f64).sqrt()).abs() < 1e-12);
        assert!((cosine[0] - 24.0_f64 / 25.0_f64).abs() < 1e-12);
        assert!(cosine[1].abs() < 1e-12);
        assert!((distance[0] - (1.0_f64 - 24.0_f64 / 25.0_f64)).abs() < 1e-12);
        assert!((distance[1] - 1.0).abs() < 1e-12);
        assert!((normalized[[0, 0]] - 0.6).abs() < 1e-12);
        assert!((normalized[[0, 1]] - 0.8).abs() < 1e-12);
    }

    #[test]
    fn batched_rowwise_real_vector_into_paths_work() {
        let left = arr2(&[[3.0_f64, 4.0], [1.0, 1.0]]);
        let right = arr2(&[[4.0_f64, 3.0], [1.0, -1.0]]);

        let mut norms = Array1::<f64>::zeros(2);
        let mut cosine = Array1::<f64>::zeros(2);
        let mut distance = Array1::<f64>::zeros(2);
        let mut normalized = Array2::<f64>::zeros((2, 2));

        batched_l2_norm_into(&left, &mut norms).unwrap();
        batched_cosine_similarity_into(&left, &right, &mut cosine).unwrap();
        batched_cosine_distance_into(&left, &right, &mut distance).unwrap();
        batched_normalize_into(&left, &mut normalized).unwrap();

        assert!((norms[0] - 5.0).abs() < 1e-12);
        assert!((cosine[0] - 24.0_f64 / 25.0_f64).abs() < 1e-12);
        assert!((distance[1] - 1.0).abs() < 1e-12);
        assert!((normalized[[0, 0]] - 0.6).abs() < 1e-12);
    }

    #[test]
    fn batched_rowwise_complex_vector_kernels_match_expected() {
        let left = arr2(&[
            [Complex64::new(1.0, 1.0), Complex64::new(0.0, 2.0)],
            [Complex64::new(2.0, 0.0), Complex64::new(0.0, 2.0)],
        ]);
        let right = arr2(&[
            [Complex64::new(1.0, -1.0), Complex64::new(2.0, 0.0)],
            [Complex64::new(0.0, 2.0), Complex64::new(2.0, 0.0)],
        ]);

        let dots = batched_dot_hermitian(&left, &right).unwrap();
        let norms = batched_l2_norm_complex(&left).unwrap();
        let cosine = batched_cosine_similarity_complex(&left, &right).unwrap();
        let normalized = batched_normalize_complex(&left).unwrap();

        assert!((dots[0] - Complex64::new(0.0, -6.0)).norm() < 1e-12);
        assert!((dots[1] - Complex64::new(0.0, 0.0)).norm() < 1e-12);
        assert!((norms[0] - (6.0_f64).sqrt()).abs() < 1e-12);
        assert!(cosine[0].norm() <= 1.0 + 1e-12);
        assert!(
            (normalized.row(0).iter().map(Complex64::norm_sqr).sum::<f64>() - 1.0).abs() < 1e-12
        );
    }

    #[test]
    fn pairwise_cosine_distance_view_and_into_paths_work() {
        let left = arr2(&[[1.0_f64, 0.0], [1.0, 1.0]]);
        let right = arr2(&[[1.0_f64, 0.0], [0.0, 1.0]]);

        let expected = pairwise_cosine_distance(&left, &right).unwrap();
        let viewed = pairwise_cosine_distance_view(&left.view(), &right.view()).unwrap();
        let mut output = Array2::<f64>::zeros((left.nrows(), right.nrows()));

        pairwise_cosine_distance_into(&left, &right, &mut output).unwrap();

        assert_eq!(viewed.shape(), expected.shape());
        for i in 0..expected.nrows() {
            for j in 0..expected.ncols() {
                assert!((viewed[[i, j]] - expected[[i, j]]).abs() < 1e-12);
                assert!((output[[i, j]] - expected[[i, j]]).abs() < 1e-12);
            }
        }

        let zero_row = arr2(&[[0.0_f64, 0.0], [1.0, 0.0]]);
        assert!(matches!(
            pairwise_cosine_distance_view(&zero_row.view(), &right.view()),
            Err(VectorError::ZeroNorm)
        ));

        let mut wrong_shape = Array2::<f64>::zeros((1, 2));
        assert!(matches!(
            pairwise_cosine_distance_into(&left, &right, &mut wrong_shape),
            Err(VectorError::DimensionMismatch)
        ));
    }

    #[test]
    fn batched_complex_into_paths_match_allocating_variants() {
        let left = arr2(&[
            [Complex64::new(1.0, 1.0), Complex64::new(0.0, 2.0)],
            [Complex64::new(2.0, 0.0), Complex64::new(0.0, 2.0)],
        ]);
        let right = arr2(&[
            [Complex64::new(1.0, -1.0), Complex64::new(2.0, 0.0)],
            [Complex64::new(0.0, 2.0), Complex64::new(2.0, 0.0)],
        ]);

        let expected_dots = batched_dot_hermitian(&left, &right).unwrap();
        let expected_norms = batched_l2_norm_complex(&left).unwrap();
        let expected_cosine = batched_cosine_similarity_complex(&left, &right).unwrap();
        let expected_normalized = batched_normalize_complex(&left).unwrap();

        let mut dots = Array1::<Complex64>::zeros(left.nrows());
        let mut norms = Array1::<f64>::zeros(left.nrows());
        let mut cosine = Array1::<Complex64>::zeros(left.nrows());
        let mut normalized = Array2::<Complex64>::zeros(left.dim());

        batched_dot_hermitian_into(&left, &right, &mut dots).unwrap();
        batched_l2_norm_complex_into(&left, &mut norms).unwrap();
        batched_cosine_similarity_complex_into(&left, &right, &mut cosine).unwrap();
        batched_normalize_complex_into(&left, &mut normalized).unwrap();

        for i in 0..left.nrows() {
            assert!((dots[i] - expected_dots[i]).norm() < 1e-12);
            assert!((norms[i] - expected_norms[i]).abs() < 1e-12);
            assert!((cosine[i] - expected_cosine[i]).norm() < 1e-12);
        }
        for i in 0..left.nrows() {
            for j in 0..left.ncols() {
                assert!((normalized[[i, j]] - expected_normalized[[i, j]]).norm() < 1e-12);
            }
        }
    }

    #[test]
    fn batched_vector_error_paths_are_explicit() {
        let empty = Array2::<f64>::zeros((0, 0));
        assert!(matches!(batched_l2_norm_view(&empty.view()), Err(VectorError::EmptyInput)));
        assert!(matches!(
            batched_dot_view(&empty.view(), &empty.view()),
            Err(VectorError::EmptyInput)
        ));

        let left = arr2(&[[1.0_f64, 2.0], [0.0, 0.0]]);
        let right = arr2(&[[2.0_f64, 1.0], [1.0, 1.0]]);
        assert!(matches!(
            batched_cosine_similarity_view(&left.view(), &right.view()),
            Err(VectorError::ZeroNorm)
        ));

        let complex_zero = arr2(&[[Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)]]);
        assert!(matches!(
            batched_cosine_similarity_complex_view(&complex_zero.view(), &complex_zero.view()),
            Err(VectorError::ZeroNorm)
        ));

        let mut wrong_shape = Array2::<f64>::zeros((1, 1));
        assert!(matches!(
            batched_normalize_into(&arr2(&[[1.0_f64, 2.0], [3.0, 4.0]]), &mut wrong_shape),
            Err(VectorError::DimensionMismatch)
        ));
    }
}
