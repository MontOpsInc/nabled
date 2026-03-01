//! Dense matrix pipeline primitives over ndarray arrays.

use std::fmt;

use ndarray::{
    Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, ArrayViewMut1, ArrayViewMut2,
    ArrayViewMut3,
};

/// Error type for dense matrix operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixError {
    /// Input matrix/vector is empty.
    EmptyInput,
    /// Input dimensions are incompatible.
    DimensionMismatch,
}

impl fmt::Display for MatrixError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MatrixError::EmptyInput => write!(f, "input cannot be empty"),
            MatrixError::DimensionMismatch => write!(f, "input dimensions are incompatible"),
        }
    }
}

impl std::error::Error for MatrixError {}

fn validate_matrix_non_empty(matrix: &ArrayView2<'_, f64>) -> Result<(), MatrixError> {
    if matrix.is_empty() {
        return Err(MatrixError::EmptyInput);
    }
    Ok(())
}

fn validate_vector_non_empty(vector: &ArrayView1<'_, f64>) -> Result<(), MatrixError> {
    if vector.is_empty() {
        return Err(MatrixError::EmptyInput);
    }
    Ok(())
}

fn validate_tensor_non_empty(tensor: &ArrayView3<'_, f64>) -> Result<(), MatrixError> {
    if tensor.is_empty() {
        return Err(MatrixError::EmptyInput);
    }
    Ok(())
}

/// Compute dense matrix-vector product `y = A x`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn matvec(matrix: &Array2<f64>, vector: &Array1<f64>) -> Result<Array1<f64>, MatrixError> {
    let mut output = Array1::<f64>::zeros(matrix.nrows());
    let matrix_view = matrix.view();
    let vector_view = vector.view();
    matvec_view_into(&matrix_view, &vector_view, output.view_mut())?;
    Ok(output)
}

/// Compute dense matrix-vector product `y = A x` from views.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn matvec_view(
    matrix: &ArrayView2<'_, f64>,
    vector: &ArrayView1<'_, f64>,
) -> Result<Array1<f64>, MatrixError> {
    let mut output = Array1::<f64>::zeros(matrix.nrows());
    matvec_view_into(matrix, vector, output.view_mut())?;
    Ok(output)
}

/// Compute dense matrix-vector product `y = A x` into `output`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn matvec_into(
    matrix: &Array2<f64>,
    vector: &Array1<f64>,
    output: &mut Array1<f64>,
) -> Result<(), MatrixError> {
    let matrix_view = matrix.view();
    let vector_view = vector.view();
    matvec_view_into(&matrix_view, &vector_view, output.view_mut())
}

/// Compute dense matrix-vector product `y = A x` from views into `output`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn matvec_view_into(
    matrix: &ArrayView2<'_, f64>,
    vector: &ArrayView1<'_, f64>,
    mut output: ArrayViewMut1<'_, f64>,
) -> Result<(), MatrixError> {
    validate_matrix_non_empty(matrix)?;
    validate_vector_non_empty(vector)?;
    if vector.len() != matrix.ncols() || output.len() != matrix.nrows() {
        return Err(MatrixError::DimensionMismatch);
    }

    for row in 0..matrix.nrows() {
        let mut sum = 0.0_f64;
        for col in 0..matrix.ncols() {
            sum += matrix[[row, col]] * vector[col];
        }
        output[row] = sum;
    }
    Ok(())
}

/// Compute dense matrix-matrix product `C = A B`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn matmat(left: &Array2<f64>, right: &Array2<f64>) -> Result<Array2<f64>, MatrixError> {
    let mut output = Array2::<f64>::zeros((left.nrows(), right.ncols()));
    let left_view = left.view();
    let right_view = right.view();
    matmat_view_into(&left_view, &right_view, output.view_mut())?;
    Ok(output)
}

/// Compute dense matrix-matrix product `C = A B` from views.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn matmat_view(
    left: &ArrayView2<'_, f64>,
    right: &ArrayView2<'_, f64>,
) -> Result<Array2<f64>, MatrixError> {
    let mut output = Array2::<f64>::zeros((left.nrows(), right.ncols()));
    matmat_view_into(left, right, output.view_mut())?;
    Ok(output)
}

/// Compute dense matrix-matrix product `C = A B` into `output`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn matmat_into(
    left: &Array2<f64>,
    right: &Array2<f64>,
    output: &mut Array2<f64>,
) -> Result<(), MatrixError> {
    let left_view = left.view();
    let right_view = right.view();
    matmat_view_into(&left_view, &right_view, output.view_mut())
}

/// Compute dense matrix-matrix product `C = A B` from views into `output`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn matmat_view_into(
    left: &ArrayView2<'_, f64>,
    right: &ArrayView2<'_, f64>,
    mut output: ArrayViewMut2<'_, f64>,
) -> Result<(), MatrixError> {
    validate_matrix_non_empty(left)?;
    validate_matrix_non_empty(right)?;
    if left.ncols() != right.nrows() || output.dim() != (left.nrows(), right.ncols()) {
        return Err(MatrixError::DimensionMismatch);
    }

    output.fill(0.0);
    for row in 0..left.nrows() {
        for k in 0..left.ncols() {
            let lhs = left[[row, k]];
            for col in 0..right.ncols() {
                output[[row, col]] += lhs * right[[k, col]];
            }
        }
    }
    Ok(())
}

/// Apply one matrix to a batch of row-vectors.
///
/// Input is `(batch, cols)` and matrix is `(rows, cols)`, output is `(batch, rows)`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn batched_row_matvec(
    batch_vectors: &Array2<f64>,
    matrix: &Array2<f64>,
) -> Result<Array2<f64>, MatrixError> {
    let mut output = Array2::<f64>::zeros((batch_vectors.nrows(), matrix.nrows()));
    let batch_view = batch_vectors.view();
    let matrix_view = matrix.view();
    batched_row_matvec_view_into(&batch_view, &matrix_view, output.view_mut())?;
    Ok(output)
}

/// Apply one matrix to a batch of row-vectors from views.
///
/// Input is `(batch, cols)` and matrix is `(rows, cols)`, output is `(batch, rows)`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn batched_row_matvec_view(
    batch_vectors: &ArrayView2<'_, f64>,
    matrix: &ArrayView2<'_, f64>,
) -> Result<Array2<f64>, MatrixError> {
    let mut output = Array2::<f64>::zeros((batch_vectors.nrows(), matrix.nrows()));
    batched_row_matvec_view_into(batch_vectors, matrix, output.view_mut())?;
    Ok(output)
}

/// Apply one matrix to a batch of row-vectors into `output`.
///
/// Input is `(batch, cols)` and matrix is `(rows, cols)`, output must be `(batch, rows)`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn batched_row_matvec_into(
    batch_vectors: &Array2<f64>,
    matrix: &Array2<f64>,
    output: &mut Array2<f64>,
) -> Result<(), MatrixError> {
    let batch_view = batch_vectors.view();
    let matrix_view = matrix.view();
    batched_row_matvec_view_into(&batch_view, &matrix_view, output.view_mut())
}

/// Apply one matrix to a batch of row-vectors from views into `output`.
///
/// Input is `(batch, cols)` and matrix is `(rows, cols)`, output must be `(batch, rows)`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn batched_row_matvec_view_into(
    batch_vectors: &ArrayView2<'_, f64>,
    matrix: &ArrayView2<'_, f64>,
    mut output: ArrayViewMut2<'_, f64>,
) -> Result<(), MatrixError> {
    validate_matrix_non_empty(batch_vectors)?;
    validate_matrix_non_empty(matrix)?;
    if batch_vectors.ncols() != matrix.ncols()
        || output.dim() != (batch_vectors.nrows(), matrix.nrows())
    {
        return Err(MatrixError::DimensionMismatch);
    }

    output.fill(0.0);
    for batch in 0..batch_vectors.nrows() {
        for row in 0..matrix.nrows() {
            let mut sum = 0.0_f64;
            for col in 0..matrix.ncols() {
                sum += batch_vectors[[batch, col]] * matrix[[row, col]];
            }
            output[[batch, row]] = sum;
        }
    }
    Ok(())
}

/// Compute batched dense matrix-matrix products.
///
/// Inputs are `(batch, m, k)` and `(batch, k, n)` and output is `(batch, m, n)`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn batched_matmat(
    left_batches: &Array3<f64>,
    right_batches: &Array3<f64>,
) -> Result<Array3<f64>, MatrixError> {
    let mut output =
        Array3::<f64>::zeros((left_batches.dim().0, left_batches.dim().1, right_batches.dim().2));
    batched_matmat_view_into(&left_batches.view(), &right_batches.view(), output.view_mut())?;
    Ok(output)
}

/// Compute batched dense matrix-matrix products from views.
///
/// Inputs are `(batch, m, k)` and `(batch, k, n)` and output is `(batch, m, n)`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn batched_matmat_view(
    left_batches: &ArrayView3<'_, f64>,
    right_batches: &ArrayView3<'_, f64>,
) -> Result<Array3<f64>, MatrixError> {
    let mut output =
        Array3::<f64>::zeros((left_batches.dim().0, left_batches.dim().1, right_batches.dim().2));
    batched_matmat_view_into(left_batches, right_batches, output.view_mut())?;
    Ok(output)
}

/// Compute batched dense matrix-matrix products into `output`.
///
/// Inputs are `(batch, m, k)` and `(batch, k, n)` and output is `(batch, m, n)`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn batched_matmat_into(
    left_batches: &Array3<f64>,
    right_batches: &Array3<f64>,
    output: &mut Array3<f64>,
) -> Result<(), MatrixError> {
    batched_matmat_view_into(&left_batches.view(), &right_batches.view(), output.view_mut())
}

/// Compute batched dense matrix-matrix products from views into `output`.
///
/// Inputs are `(batch, m, k)` and `(batch, k, n)` and output is `(batch, m, n)`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn batched_matmat_view_into(
    left_batches: &ArrayView3<'_, f64>,
    right_batches: &ArrayView3<'_, f64>,
    mut output: ArrayViewMut3<'_, f64>,
) -> Result<(), MatrixError> {
    validate_tensor_non_empty(left_batches)?;
    validate_tensor_non_empty(right_batches)?;
    if left_batches.dim().0 != right_batches.dim().0
        || left_batches.dim().2 != right_batches.dim().1
        || output.dim() != (left_batches.dim().0, left_batches.dim().1, right_batches.dim().2)
    {
        return Err(MatrixError::DimensionMismatch);
    }

    output.fill(0.0);
    let (batch, rows, inner) = left_batches.dim();
    let cols = right_batches.dim().2;
    for b in 0..batch {
        for row in 0..rows {
            for k in 0..inner {
                let lhs = left_batches[[b, row, k]];
                for col in 0..cols {
                    output[[b, row, col]] += lhs * right_batches[[b, k, col]];
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2, Array3};

    use super::*;

    #[test]
    fn matvec_variants_match() {
        let matrix = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 0.0, 1.0, 1.0]).unwrap();
        let vector = Array1::from_vec(vec![1.0, 0.0, 2.0]);

        let allocating = matvec(&matrix, &vector).unwrap();
        let viewed = matvec_view(&matrix.view(), &vector.view()).unwrap();
        let mut into = Array1::<f64>::zeros(2);
        matvec_into(&matrix, &vector, &mut into).unwrap();

        for i in 0..2 {
            assert!((allocating[i] - viewed[i]).abs() < 1e-12);
            assert!((allocating[i] - into[i]).abs() < 1e-12);
        }
        assert!((allocating[0] - 7.0).abs() < 1e-12);
        assert!((allocating[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn matmat_variants_match() {
        let left = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 0.0, 0.0, 1.0, 1.0]).unwrap();
        let right = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 2.0, 1.0, 1.0, 3.0]).unwrap();

        let allocating = matmat(&left, &right).unwrap();
        let viewed = matmat_view(&left.view(), &right.view()).unwrap();
        let mut into = Array2::<f64>::zeros((2, 2));
        matmat_into(&left, &right, &mut into).unwrap();

        for i in 0..2 {
            for j in 0..2 {
                assert!((allocating[[i, j]] - viewed[[i, j]]).abs() < 1e-12);
                assert!((allocating[[i, j]] - into[[i, j]]).abs() < 1e-12);
            }
        }
        assert!((allocating[[0, 0]] - 5.0).abs() < 1e-12);
        assert!((allocating[[0, 1]] - 2.0).abs() < 1e-12);
        assert!((allocating[[1, 0]] - 3.0).abs() < 1e-12);
        assert!((allocating[[1, 1]] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn batched_row_matvec_variants_match() {
        let vectors = Array2::from_shape_vec((2, 3), vec![1.0, 0.0, 2.0, 0.5, -1.0, 1.0]).unwrap();
        let matrix = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 0.0, 1.0, 1.0]).unwrap();

        let allocating = batched_row_matvec(&vectors, &matrix).unwrap();
        let viewed = batched_row_matvec_view(&vectors.view(), &matrix.view()).unwrap();
        let mut into = Array2::<f64>::zeros((2, 2));
        batched_row_matvec_into(&vectors, &matrix, &mut into).unwrap();

        for i in 0..2 {
            for j in 0..2 {
                assert!((allocating[[i, j]] - viewed[[i, j]]).abs() < 1e-12);
                assert!((allocating[[i, j]] - into[[i, j]]).abs() < 1e-12);
            }
        }

        assert!((allocating[[0, 0]] - 7.0).abs() < 1e-12);
        assert!((allocating[[0, 1]] - 2.0).abs() < 1e-12);
        assert!((allocating[[1, 0]] - 1.5).abs() < 1e-12);
        assert!((allocating[[1, 1]] - 0.0).abs() < 1e-12);
    }

    #[test]
    fn rejects_dimension_mismatch_and_empty_inputs() {
        let matrix = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let bad_vector = Array1::from_vec(vec![1.0]);
        assert!(matches!(matvec(&matrix, &bad_vector), Err(MatrixError::DimensionMismatch)));

        let left = Array2::from_shape_vec((2, 3), vec![1.0; 6]).unwrap();
        let right = Array2::from_shape_vec((2, 2), vec![1.0; 4]).unwrap();
        assert!(matches!(matmat(&left, &right), Err(MatrixError::DimensionMismatch)));

        let empty_matrix = Array2::<f64>::zeros((0, 0));
        let empty_vector = Array1::<f64>::zeros(0);
        assert!(matches!(matvec(&empty_matrix, &empty_vector), Err(MatrixError::EmptyInput)));

        let left_batches = Array3::<f64>::zeros((1, 2, 3));
        let right_batches = Array3::<f64>::zeros((1, 2, 2));
        assert!(matches!(
            batched_matmat(&left_batches, &right_batches),
            Err(MatrixError::DimensionMismatch)
        ));
    }

    #[test]
    fn batched_matmat_variants_match() {
        let left_batches = Array3::from_shape_vec((2, 2, 3), vec![
            1.0, 2.0, 0.0, 0.0, 1.0, 1.0, 2.0, 0.0, 1.0, 1.0, 3.0, 2.0,
        ])
        .unwrap();
        let right_batches = Array3::from_shape_vec((2, 3, 2), vec![
            1.0, 0.0, 2.0, 1.0, 1.0, 3.0, 0.0, 2.0, 1.0, 1.0, 3.0, 0.0,
        ])
        .unwrap();

        let allocating = batched_matmat(&left_batches, &right_batches).unwrap();
        let viewed = batched_matmat_view(&left_batches.view(), &right_batches.view()).unwrap();
        let mut into = Array3::<f64>::zeros((2, 2, 2));
        batched_matmat_into(&left_batches, &right_batches, &mut into).unwrap();

        for b in 0..2 {
            for i in 0..2 {
                for j in 0..2 {
                    assert!((allocating[[b, i, j]] - viewed[[b, i, j]]).abs() < 1e-12);
                    assert!((allocating[[b, i, j]] - into[[b, i, j]]).abs() < 1e-12);
                }
            }
        }

        assert!((allocating[[0, 0, 0]] - 5.0).abs() < 1e-12);
        assert!((allocating[[0, 0, 1]] - 2.0).abs() < 1e-12);
        assert!((allocating[[0, 1, 0]] - 3.0).abs() < 1e-12);
        assert!((allocating[[0, 1, 1]] - 4.0).abs() < 1e-12);
    }
}
