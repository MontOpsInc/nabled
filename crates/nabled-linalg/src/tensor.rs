//! Tensor and cube primitives over ndarray higher-rank arrays.

use std::fmt;

use ndarray::{
    Array2, Array3, ArrayD, ArrayView2, ArrayView3, ArrayViewD, ArrayViewMut2, ArrayViewMut3, Axis,
    IxDyn,
};
use num_complex::Complex64;

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

fn validate_cube_non_empty_complex(cube: &ArrayView3<'_, Complex64>) -> Result<(), TensorError> {
    if cube.is_empty() {
        return Err(TensorError::EmptyInput);
    }
    Ok(())
}

fn validate_matrix_non_empty_complex(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<(), TensorError> {
    if matrix.is_empty() {
        return Err(TensorError::EmptyInput);
    }
    Ok(())
}

fn validate_tensor_nd_non_empty(tensor: &ArrayViewD<'_, f64>) -> Result<(), TensorError> {
    if tensor.is_empty() {
        return Err(TensorError::EmptyInput);
    }
    if tensor.ndim() == 0 {
        return Err(TensorError::DimensionMismatch);
    }
    Ok(())
}

fn validate_tensor_nd_non_empty_complex(
    tensor: &ArrayViewD<'_, Complex64>,
) -> Result<(), TensorError> {
    if tensor.is_empty() {
        return Err(TensorError::EmptyInput);
    }
    if tensor.ndim() == 0 {
        return Err(TensorError::DimensionMismatch);
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

/// Compute batched complex cube-matrix vector products.
///
/// Inputs are `cube=(batch, rows, cols)` and `vectors=(batch, cols)`.
/// Output is `(batch, rows)`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn cube_matvec_complex(
    cube: &Array3<Complex64>,
    vectors: &Array2<Complex64>,
) -> Result<Array2<Complex64>, TensorError> {
    let mut output = Array2::<Complex64>::zeros((cube.dim().0, cube.dim().1));
    cube_matvec_complex_view_into(&cube.view(), &vectors.view(), output.view_mut())?;
    Ok(output)
}

/// Compute batched complex cube-matrix vector products from views.
///
/// Inputs are `cube=(batch, rows, cols)` and `vectors=(batch, cols)`.
/// Output is `(batch, rows)`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn cube_matvec_complex_view(
    cube: &ArrayView3<'_, Complex64>,
    vectors: &ArrayView2<'_, Complex64>,
) -> Result<Array2<Complex64>, TensorError> {
    let mut output = Array2::<Complex64>::zeros((cube.dim().0, cube.dim().1));
    cube_matvec_complex_view_into(cube, vectors, output.view_mut())?;
    Ok(output)
}

/// Compute batched complex cube-matrix vector products into `output`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn cube_matvec_complex_into(
    cube: &Array3<Complex64>,
    vectors: &Array2<Complex64>,
    output: &mut Array2<Complex64>,
) -> Result<(), TensorError> {
    cube_matvec_complex_view_into(&cube.view(), &vectors.view(), output.view_mut())
}

/// Compute batched complex cube-matrix vector products from views into `output`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn cube_matvec_complex_view_into(
    cube: &ArrayView3<'_, Complex64>,
    vectors: &ArrayView2<'_, Complex64>,
    mut output: ArrayViewMut2<'_, Complex64>,
) -> Result<(), TensorError> {
    validate_cube_non_empty_complex(cube)?;
    validate_matrix_non_empty_complex(vectors)?;
    if vectors.dim() != (cube.dim().0, cube.dim().2) || output.dim() != (cube.dim().0, cube.dim().1)
    {
        return Err(TensorError::DimensionMismatch);
    }

    output.fill(Complex64::new(0.0, 0.0));
    let (batch, rows, cols) = cube.dim();
    for b in 0..batch {
        for row in 0..rows {
            let mut sum = Complex64::new(0.0, 0.0);
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

/// Compute batched complex cube matrix-matrix products.
///
/// Inputs are `(batch, m, k)` and `(batch, k, n)` and output is `(batch, m, n)`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn cube_matmat_complex(
    left_cubes: &Array3<Complex64>,
    right_cubes: &Array3<Complex64>,
) -> Result<Array3<Complex64>, TensorError> {
    let mut output =
        Array3::<Complex64>::zeros((left_cubes.dim().0, left_cubes.dim().1, right_cubes.dim().2));
    cube_matmat_complex_view_into(&left_cubes.view(), &right_cubes.view(), output.view_mut())?;
    Ok(output)
}

/// Compute batched complex cube matrix-matrix products from views.
///
/// Inputs are `(batch, m, k)` and `(batch, k, n)` and output is `(batch, m, n)`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn cube_matmat_complex_view(
    left_cubes: &ArrayView3<'_, Complex64>,
    right_cubes: &ArrayView3<'_, Complex64>,
) -> Result<Array3<Complex64>, TensorError> {
    let mut output =
        Array3::<Complex64>::zeros((left_cubes.dim().0, left_cubes.dim().1, right_cubes.dim().2));
    cube_matmat_complex_view_into(left_cubes, right_cubes, output.view_mut())?;
    Ok(output)
}

/// Compute batched complex cube matrix-matrix products into `output`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn cube_matmat_complex_into(
    left_cubes: &Array3<Complex64>,
    right_cubes: &Array3<Complex64>,
    output: &mut Array3<Complex64>,
) -> Result<(), TensorError> {
    cube_matmat_complex_view_into(&left_cubes.view(), &right_cubes.view(), output.view_mut())
}

/// Compute batched complex cube matrix-matrix products from views into `output`.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn cube_matmat_complex_view_into(
    left_cubes: &ArrayView3<'_, Complex64>,
    right_cubes: &ArrayView3<'_, Complex64>,
    mut output: ArrayViewMut3<'_, Complex64>,
) -> Result<(), TensorError> {
    validate_cube_non_empty_complex(left_cubes)?;
    validate_cube_non_empty_complex(right_cubes)?;
    if left_cubes.dim().0 != right_cubes.dim().0
        || left_cubes.dim().2 != right_cubes.dim().1
        || output.dim() != (left_cubes.dim().0, left_cubes.dim().1, right_cubes.dim().2)
    {
        return Err(TensorError::DimensionMismatch);
    }

    output.fill(Complex64::new(0.0, 0.0));
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

/// Reduce a tensor along its last axis by summation.
///
/// # Errors
/// Returns an error if tensor is empty or has zero dimensions.
pub fn sum_last_axis(tensor: &ArrayD<f64>) -> Result<ArrayD<f64>, TensorError> {
    let tensor_view = tensor.view();
    validate_tensor_nd_non_empty(&tensor_view)?;
    let axis = Axis(tensor_view.ndim() - 1);
    let reduced = tensor_view.sum_axis(axis);
    Ok(reduced.into_dyn())
}

/// Compute L2 norm along the last axis of a tensor.
///
/// # Errors
/// Returns an error if tensor is empty or has zero dimensions.
pub fn l2_norm_last_axis(tensor: &ArrayD<f64>) -> Result<ArrayD<f64>, TensorError> {
    let tensor_view = tensor.view();
    validate_tensor_nd_non_empty(&tensor_view)?;

    let axis = Axis(tensor_view.ndim() - 1);
    let mut output_shape = tensor_view.shape().to_vec();
    let _ = output_shape.pop();
    let mut output = ArrayD::<f64>::zeros(IxDyn(&output_shape));
    for (out_value, lane) in output.iter_mut().zip(tensor_view.lanes(axis)) {
        let sum_sq = lane.iter().map(|value| value * value).sum::<f64>();
        *out_value = sum_sq.sqrt();
    }
    Ok(output)
}

/// Normalize tensor values along the last axis.
///
/// # Errors
/// Returns an error if tensor is empty or has zero dimensions.
pub fn normalize_last_axis(tensor: &ArrayD<f64>) -> Result<ArrayD<f64>, TensorError> {
    let tensor_view = tensor.view();
    validate_tensor_nd_non_empty(&tensor_view)?;

    let mut output = tensor.clone();
    let axis = Axis(tensor_view.ndim() - 1);
    for mut lane in output.lanes_mut(axis) {
        let norm = lane.iter().map(|value| value * value).sum::<f64>().sqrt();
        let denominator = norm.max(f64::EPSILON);
        for value in &mut lane {
            *value /= denominator;
        }
    }
    Ok(output)
}

/// Compute batched dot products along the last axis of two tensors.
///
/// The input tensors must have identical shape and `ndim >= 1`.
/// Output shape is the input shape without the last axis.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn batched_dot_last_axis(
    left: &ArrayD<f64>,
    right: &ArrayD<f64>,
) -> Result<ArrayD<f64>, TensorError> {
    let left_view = left.view();
    let right_view = right.view();
    validate_tensor_nd_non_empty(&left_view)?;
    validate_tensor_nd_non_empty(&right_view)?;
    if left_view.shape() != right_view.shape() {
        return Err(TensorError::DimensionMismatch);
    }

    let axis = Axis(left_view.ndim() - 1);
    let mut output_shape = left_view.shape().to_vec();
    let _ = output_shape.pop();
    let mut output = ArrayD::<f64>::zeros(IxDyn(&output_shape));
    for ((out_value, left_lane), right_lane) in
        output.iter_mut().zip(left_view.lanes(axis)).zip(right_view.lanes(axis))
    {
        let dot = left_lane.iter().zip(right_lane.iter()).map(|(lhs, rhs)| lhs * rhs).sum::<f64>();
        *out_value = dot;
    }
    Ok(output)
}

/// Reduce a complex tensor along its last axis by summation.
///
/// # Errors
/// Returns an error if tensor is empty or has zero dimensions.
pub fn sum_last_axis_complex(tensor: &ArrayD<Complex64>) -> Result<ArrayD<Complex64>, TensorError> {
    let tensor_view = tensor.view();
    validate_tensor_nd_non_empty_complex(&tensor_view)?;
    let axis = Axis(tensor_view.ndim() - 1);
    let reduced = tensor_view.sum_axis(axis);
    Ok(reduced.into_dyn())
}

/// Compute L2 norm along the last axis of a complex tensor.
///
/// # Errors
/// Returns an error if tensor is empty or has zero dimensions.
pub fn l2_norm_last_axis_complex(tensor: &ArrayD<Complex64>) -> Result<ArrayD<f64>, TensorError> {
    let tensor_view = tensor.view();
    validate_tensor_nd_non_empty_complex(&tensor_view)?;

    let axis = Axis(tensor_view.ndim() - 1);
    let mut output_shape = tensor_view.shape().to_vec();
    let _ = output_shape.pop();
    let mut output = ArrayD::<f64>::zeros(IxDyn(&output_shape));
    for (out_value, lane) in output.iter_mut().zip(tensor_view.lanes(axis)) {
        let sum_sq = lane.iter().map(Complex64::norm_sqr).sum::<f64>();
        *out_value = sum_sq.sqrt();
    }
    Ok(output)
}

/// Normalize complex tensor values along the last axis.
///
/// # Errors
/// Returns an error if tensor is empty or has zero dimensions.
pub fn normalize_last_axis_complex(
    tensor: &ArrayD<Complex64>,
) -> Result<ArrayD<Complex64>, TensorError> {
    let tensor_view = tensor.view();
    validate_tensor_nd_non_empty_complex(&tensor_view)?;

    let mut output = tensor.clone();
    let axis = Axis(tensor_view.ndim() - 1);
    for mut lane in output.lanes_mut(axis) {
        let norm = lane.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt();
        let denominator = norm.max(f64::EPSILON);
        for value in &mut lane {
            *value /= denominator;
        }
    }
    Ok(output)
}

/// Compute batched complex dot products along the last axis of two tensors.
///
/// The input tensors must have identical shape and `ndim >= 1`.
/// Output shape is the input shape without the last axis.
///
/// # Errors
/// Returns an error if inputs are empty or dimensions are incompatible.
pub fn batched_dot_last_axis_complex(
    left: &ArrayD<Complex64>,
    right: &ArrayD<Complex64>,
) -> Result<ArrayD<Complex64>, TensorError> {
    let left_view = left.view();
    let right_view = right.view();
    validate_tensor_nd_non_empty_complex(&left_view)?;
    validate_tensor_nd_non_empty_complex(&right_view)?;
    if left_view.shape() != right_view.shape() {
        return Err(TensorError::DimensionMismatch);
    }

    let axis = Axis(left_view.ndim() - 1);
    let mut output_shape = left_view.shape().to_vec();
    let _ = output_shape.pop();
    let mut output = ArrayD::<Complex64>::zeros(IxDyn(&output_shape));
    for ((out_value, left_lane), right_lane) in
        output.iter_mut().zip(left_view.lanes(axis)).zip(right_view.lanes(axis))
    {
        let dot = left_lane
            .iter()
            .zip(right_lane.iter())
            .map(|(lhs, rhs)| lhs.conj() * rhs)
            .sum::<Complex64>();
        *out_value = dot;
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use ndarray::{Array2, Array3, ArrayD, IxDyn};
    use num_complex::Complex64;

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
    fn cube_matvec_complex_variants_match() {
        let cube = Array3::from_shape_vec((2, 2, 2), vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 1.0),
            Complex64::new(-1.0, 0.0),
            Complex64::new(0.5, 0.5),
            Complex64::new(3.0, -2.0),
        ])
        .unwrap();
        let vectors = Array2::from_shape_vec((2, 2), vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.5, -0.5),
            Complex64::new(-1.0, 1.0),
            Complex64::new(2.0, 0.0),
        ])
        .unwrap();

        let allocating = cube_matvec_complex(&cube, &vectors).unwrap();
        let viewed = cube_matvec_complex_view(&cube.view(), &vectors.view()).unwrap();
        let mut into = Array2::<Complex64>::zeros((2, 2));
        cube_matvec_complex_into(&cube, &vectors, &mut into).unwrap();

        for b in 0..2 {
            for row in 0..2 {
                assert!((allocating[[b, row]] - viewed[[b, row]]).norm() < 1e-12);
                assert!((allocating[[b, row]] - into[[b, row]]).norm() < 1e-12);
            }
        }
    }

    #[test]
    fn cube_matmat_complex_variants_match() {
        let left = Array3::from_shape_vec((1, 2, 2), vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(1.0, 0.0),
        ])
        .unwrap();
        let right = Array3::from_shape_vec((1, 2, 2), vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(1.0, -1.0),
        ])
        .unwrap();

        let allocating = cube_matmat_complex(&left, &right).unwrap();
        let viewed = cube_matmat_complex_view(&left.view(), &right.view()).unwrap();
        let mut into = Array3::<Complex64>::zeros((1, 2, 2));
        cube_matmat_complex_into(&left, &right, &mut into).unwrap();

        for i in 0..2 {
            for j in 0..2 {
                assert!((allocating[[0, i, j]] - viewed[[0, i, j]]).norm() < 1e-12);
                assert!((allocating[[0, i, j]] - into[[0, i, j]]).norm() < 1e-12);
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

    #[test]
    fn arrayd_last_axis_ops_match_expected() {
        let tensor = ArrayD::from_shape_vec(IxDyn(&[2, 2, 3]), vec![
            1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 3.0, 4.0, 1.0, 2.0, 2.0,
        ])
        .unwrap();

        let sum = sum_last_axis(&tensor).unwrap();
        assert_eq!(sum.shape(), &[2, 2]);
        assert!((sum[[0, 0]] - 6.0).abs() < 1e-12);
        assert!((sum[[0, 1]] - 4.0).abs() < 1e-12);

        let norms = l2_norm_last_axis(&tensor).unwrap();
        assert_eq!(norms.shape(), &[2, 2]);
        assert!((norms[[0, 0]] - (14.0_f64).sqrt()).abs() < 1e-12);
        assert!((norms[[0, 1]] - 4.0).abs() < 1e-12);

        let normalized = normalize_last_axis(&tensor).unwrap();
        let normalized_norms = l2_norm_last_axis(&normalized).unwrap();
        for value in &normalized_norms {
            assert!((value - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn batched_dot_last_axis_matches_manual() {
        let left = ArrayD::from_shape_vec(IxDyn(&[2, 2, 3]), vec![
            1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 3.0, 4.0, 1.0, 2.0, 2.0,
        ])
        .unwrap();
        let right = ArrayD::from_shape_vec(IxDyn(&[2, 2, 3]), vec![
            0.5, 1.0, -1.0, 0.0, 2.0, 3.0, 1.0, 1.0, 1.0, 2.0, 0.0, 1.0,
        ])
        .unwrap();

        let dots = batched_dot_last_axis(&left, &right).unwrap();
        assert_eq!(dots.shape(), &[2, 2]);
        assert!((dots[[0, 0]] - (0.5 + 2.0 - 3.0)).abs() < 1e-12);
        assert!((dots[[0, 1]] - 0.0).abs() < 1e-12);
        assert!((dots[[1, 0]] - 7.0).abs() < 1e-12);
        assert!((dots[[1, 1]] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn arrayd_complex_last_axis_ops_match_expected() {
        let tensor = ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(1.0, -1.0),
        ])
        .unwrap();

        let sum = sum_last_axis_complex(&tensor).unwrap();
        assert_eq!(sum.shape(), &[1, 2]);
        assert!((sum[[0, 0]] - Complex64::new(3.0, 1.0)).norm() < 1e-12);

        let norms = l2_norm_last_axis_complex(&tensor).unwrap();
        assert_eq!(norms.shape(), &[1, 2]);
        assert!((norms[[0, 0]] - (6.0_f64).sqrt()).abs() < 1e-12);

        let normalized = normalize_last_axis_complex(&tensor).unwrap();
        let normalized_norms = l2_norm_last_axis_complex(&normalized).unwrap();
        for value in &normalized_norms {
            assert!((value - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn batched_dot_last_axis_complex_matches_manual() {
        let left = ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(1.0, -1.0),
        ])
        .unwrap();
        let right = ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![
            Complex64::new(0.5, 0.0),
            Complex64::new(-1.0, 1.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(0.0, 1.0),
        ])
        .unwrap();

        let dots = batched_dot_last_axis_complex(&left, &right).unwrap();
        assert_eq!(dots.shape(), &[1, 2]);

        let expected_00 =
            left[[0, 0, 0]].conj() * right[[0, 0, 0]] + left[[0, 0, 1]].conj() * right[[0, 0, 1]];
        let expected_01 =
            left[[0, 1, 0]].conj() * right[[0, 1, 0]] + left[[0, 1, 1]].conj() * right[[0, 1, 1]];
        assert!((dots[[0, 0]] - expected_00).norm() < 1e-12);
        assert!((dots[[0, 1]] - expected_01).norm() < 1e-12);
    }

    #[test]
    fn arrayd_ops_reject_invalid_dimensions() {
        let scalar = ArrayD::from_shape_vec(IxDyn(&[]), vec![1.0]).unwrap();
        assert!(matches!(sum_last_axis(&scalar), Err(TensorError::DimensionMismatch)));

        let left = ArrayD::from_shape_vec(IxDyn(&[2, 2, 3]), vec![1.0; 12]).unwrap();
        let right = ArrayD::from_shape_vec(IxDyn(&[2, 2, 2]), vec![1.0; 8]).unwrap();
        assert!(matches!(
            batched_dot_last_axis(&left, &right),
            Err(TensorError::DimensionMismatch)
        ));
    }
}
