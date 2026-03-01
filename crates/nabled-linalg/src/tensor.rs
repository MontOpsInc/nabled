//! Tensor and cube primitives over ndarray higher-rank arrays.

use std::fmt;

use ndarray::{Array2, Array3, ArrayView2, ArrayView3, ArrayViewMut2, ArrayViewMut3};

/// Error type for tensor/cube operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorError {
    /// Input tensor/matrix is empty.
    EmptyInput,
    /// Input dimensions are incompatible.
    DimensionMismatch,
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorError::EmptyInput => write!(f, "input cannot be empty"),
            TensorError::DimensionMismatch => write!(f, "input dimensions are incompatible"),
        }
    }
}

impl std::error::Error for TensorError {}

fn validate_cube_non_empty(cube: &ArrayView3<'_, f64>) -> Result<(), TensorError> {
    if cube.is_empty() {
        return Err(TensorError::EmptyInput);
    }
    Ok(())
}

fn validate_matrix_non_empty(matrix: &ArrayView2<'_, f64>) -> Result<(), TensorError> {
    if matrix.is_empty() {
        return Err(TensorError::EmptyInput);
    }
    Ok(())
}

/// Compute batched cube-matrix vector products.
///
/// Inputs are `cube=(batch, rows, cols)` and `vectors=(batch, cols)`.
/// Output is `(batch, rows)`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn cube_matvec(cube: &Array3<f64>, vectors: &Array2<f64>) -> Result<Array2<f64>, TensorError> {
    let mut output = Array2::<f64>::zeros((cube.dim().0, cube.dim().1));
    cube_matvec_view_into(&cube.view(), &vectors.view(), output.view_mut())?;
    Ok(output)
}

/// Compute batched cube-matrix vector products from views.
///
/// Inputs are `cube=(batch, rows, cols)` and `vectors=(batch, cols)`.
/// Output is `(batch, rows)`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn cube_matvec_view(
    cube: &ArrayView3<'_, f64>,
    vectors: &ArrayView2<'_, f64>,
) -> Result<Array2<f64>, TensorError> {
    let mut output = Array2::<f64>::zeros((cube.dim().0, cube.dim().1));
    cube_matvec_view_into(cube, vectors, output.view_mut())?;
    Ok(output)
}

/// Compute batched cube-matrix vector products into `output`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn cube_matvec_into(
    cube: &Array3<f64>,
    vectors: &Array2<f64>,
    output: &mut Array2<f64>,
) -> Result<(), TensorError> {
    cube_matvec_view_into(&cube.view(), &vectors.view(), output.view_mut())
}

/// Compute batched cube-matrix vector products from views into `output`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn cube_matvec_view_into(
    cube: &ArrayView3<'_, f64>,
    vectors: &ArrayView2<'_, f64>,
    mut output: ArrayViewMut2<'_, f64>,
) -> Result<(), TensorError> {
    validate_cube_non_empty(cube)?;
    validate_matrix_non_empty(vectors)?;
    if vectors.dim() != (cube.dim().0, cube.dim().2) || output.dim() != (cube.dim().0, cube.dim().1)
    {
        return Err(TensorError::DimensionMismatch);
    }

    output.fill(0.0);
    let (batch, rows, cols) = cube.dim();
    for b in 0..batch {
        for row in 0..rows {
            let mut sum = 0.0_f64;
            for col in 0..cols {
                sum += cube[[b, row, col]] * vectors[[b, col]];
            }
            output[[b, row]] = sum;
        }
    }

    Ok(())
}

/// Compute batched cube matrix-matrix products.
///
/// Inputs are `(batch, m, k)` and `(batch, k, n)` and output is `(batch, m, n)`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn cube_matmat(
    left_cubes: &Array3<f64>,
    right_cubes: &Array3<f64>,
) -> Result<Array3<f64>, TensorError> {
    let mut output =
        Array3::<f64>::zeros((left_cubes.dim().0, left_cubes.dim().1, right_cubes.dim().2));
    cube_matmat_view_into(&left_cubes.view(), &right_cubes.view(), output.view_mut())?;
    Ok(output)
}

/// Compute batched cube matrix-matrix products from views.
///
/// Inputs are `(batch, m, k)` and `(batch, k, n)` and output is `(batch, m, n)`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn cube_matmat_view(
    left_cubes: &ArrayView3<'_, f64>,
    right_cubes: &ArrayView3<'_, f64>,
) -> Result<Array3<f64>, TensorError> {
    let mut output =
        Array3::<f64>::zeros((left_cubes.dim().0, left_cubes.dim().1, right_cubes.dim().2));
    cube_matmat_view_into(left_cubes, right_cubes, output.view_mut())?;
    Ok(output)
}

/// Compute batched cube matrix-matrix products into `output`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn cube_matmat_into(
    left_cubes: &Array3<f64>,
    right_cubes: &Array3<f64>,
    output: &mut Array3<f64>,
) -> Result<(), TensorError> {
    cube_matmat_view_into(&left_cubes.view(), &right_cubes.view(), output.view_mut())
}

/// Compute batched cube matrix-matrix products from views into `output`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn cube_matmat_view_into(
    left_cubes: &ArrayView3<'_, f64>,
    right_cubes: &ArrayView3<'_, f64>,
    mut output: ArrayViewMut3<'_, f64>,
) -> Result<(), TensorError> {
    validate_cube_non_empty(left_cubes)?;
    validate_cube_non_empty(right_cubes)?;
    if left_cubes.dim().0 != right_cubes.dim().0
        || left_cubes.dim().2 != right_cubes.dim().1
        || output.dim() != (left_cubes.dim().0, left_cubes.dim().1, right_cubes.dim().2)
    {
        return Err(TensorError::DimensionMismatch);
    }

    output.fill(0.0);
    let (batch, rows, inner) = left_cubes.dim();
    let cols = right_cubes.dim().2;
    for b in 0..batch {
        for row in 0..rows {
            for k in 0..inner {
                let lhs = left_cubes[[b, row, k]];
                for col in 0..cols {
                    output[[b, row, col]] += lhs * right_cubes[[b, k, col]];
                }
            }
        }
    }

    Ok(())
}

/// Flatten each cube slice `(rows, cols)` into one row.
///
/// Input `(batch, rows, cols)` becomes `(batch, rows * cols)`.
///
/// # Errors
/// Returns an error if input is empty.
pub fn flatten_cubes(cube: &Array3<f64>) -> Result<Array2<f64>, TensorError> {
    let cube_view = cube.view();
    validate_cube_non_empty(&cube_view)?;

    let (batch, rows, cols) = cube.dim();
    let mut output = Array2::<f64>::zeros((batch, rows * cols));
    for b in 0..batch {
        for row in 0..rows {
            for col in 0..cols {
                output[[b, row * cols + col]] = cube[[b, row, col]];
            }
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use ndarray::{Array2, Array3};

    use super::*;

    #[test]
    fn cube_matvec_variants_match() {
        let cube = Array3::from_shape_vec((2, 2, 3), vec![
            1.0, 2.0, 3.0, 0.0, 1.0, 1.0, 2.0, -1.0, 0.5, 3.0, 0.0, 2.0,
        ])
        .unwrap();
        let vectors = Array2::from_shape_vec((2, 3), vec![1.0, 0.0, 2.0, 0.5, -1.0, 1.0]).unwrap();

        let allocating = cube_matvec(&cube, &vectors).unwrap();
        let viewed = cube_matvec_view(&cube.view(), &vectors.view()).unwrap();
        let mut into = Array2::<f64>::zeros((2, 2));
        cube_matvec_into(&cube, &vectors, &mut into).unwrap();

        for b in 0..2 {
            for row in 0..2 {
                assert!((allocating[[b, row]] - viewed[[b, row]]).abs() < 1e-12);
                assert!((allocating[[b, row]] - into[[b, row]]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn cube_matmat_variants_match() {
        let left = Array3::from_shape_vec((2, 2, 3), vec![
            1.0, 2.0, 0.0, 0.0, 1.0, 1.0, 2.0, 0.0, 1.0, 1.0, 3.0, 2.0,
        ])
        .unwrap();
        let right = Array3::from_shape_vec((2, 3, 2), vec![
            1.0, 0.0, 2.0, 1.0, 1.0, 3.0, 0.0, 2.0, 1.0, 1.0, 3.0, 0.0,
        ])
        .unwrap();

        let allocating = cube_matmat(&left, &right).unwrap();
        let viewed = cube_matmat_view(&left.view(), &right.view()).unwrap();
        let mut into = Array3::<f64>::zeros((2, 2, 2));
        cube_matmat_into(&left, &right, &mut into).unwrap();

        for b in 0..2 {
            for i in 0..2 {
                for j in 0..2 {
                    assert!((allocating[[b, i, j]] - viewed[[b, i, j]]).abs() < 1e-12);
                    assert!((allocating[[b, i, j]] - into[[b, i, j]]).abs() < 1e-12);
                }
            }
        }
    }

    #[test]
    fn flatten_cubes_is_shape_stable() {
        let cube = Array3::from_shape_vec((2, 2, 2), vec![1.0, 2.0, 3.0, 4.0, 0.0, 1.0, -1.0, 2.0])
            .unwrap();
        let flattened = flatten_cubes(&cube).unwrap();
        assert_eq!(flattened.dim(), (2, 4));
        assert!((flattened[[0, 0]] - 1.0).abs() < 1e-12);
        assert!((flattened[[0, 3]] - 4.0).abs() < 1e-12);
        assert!((flattened[[1, 1]] - 1.0).abs() < 1e-12);
        assert!((flattened[[1, 2]] + 1.0).abs() < 1e-12);
    }

    #[test]
    fn tensor_ops_reject_invalid_shapes() {
        let cube = Array3::<f64>::zeros((1, 2, 3));
        let vectors = Array2::<f64>::zeros((2, 3));
        assert!(matches!(cube_matvec(&cube, &vectors), Err(TensorError::DimensionMismatch)));

        let empty = Array3::<f64>::zeros((0, 0, 0));
        assert!(matches!(flatten_cubes(&empty), Err(TensorError::EmptyInput)));
    }
}
