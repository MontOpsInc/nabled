//! Cholesky decomposition over ndarray matrices.

use std::fmt;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_complex::Complex64;

#[cfg(not(feature = "openblas-system"))]
use crate::internal::DenseKernelPolicy;

/// Result of Cholesky decomposition.
#[derive(Debug, Clone)]
pub struct NdarrayCholeskyResult {
    /// Lower-triangular factor `L` where `A = L L^T`.
    pub l: Array2<f64>,
}

/// Result of complex Cholesky decomposition.
#[derive(Debug, Clone)]
pub struct NdarrayComplexCholeskyResult {
    /// Lower-triangular factor `L` where `A = L L^H`.
    pub l: Array2<Complex64>,
}

/// Error type for Cholesky operations.
#[derive(Debug, Clone, PartialEq)]
pub enum CholeskyError {
    /// Matrix is empty.
    EmptyMatrix,
    /// Matrix is not square.
    NotSquare,
    /// Matrix is not symmetric positive definite.
    NotPositiveDefinite,
    /// Invalid input.
    InvalidInput(String),
    /// Numerical instability detected.
    NumericalInstability,
}

impl fmt::Display for CholeskyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CholeskyError::EmptyMatrix => write!(f, "Matrix cannot be empty"),
            CholeskyError::NotSquare => write!(f, "Matrix must be square"),
            CholeskyError::NotPositiveDefinite => {
                write!(f, "Matrix is not symmetric positive definite")
            }
            CholeskyError::InvalidInput(message) => write!(f, "Invalid input: {message}"),
            CholeskyError::NumericalInstability => write!(f, "Numerical instability detected"),
        }
    }
}

impl std::error::Error for CholeskyError {}

fn validate_square_finite_view(matrix: &ArrayView2<'_, f64>) -> Result<(), CholeskyError> {
    if matrix.is_empty() {
        return Err(CholeskyError::EmptyMatrix);
    }
    if matrix.nrows() != matrix.ncols() {
        return Err(CholeskyError::NotSquare);
    }
    if matrix.iter().any(|value| !value.is_finite()) {
        return Err(CholeskyError::NumericalInstability);
    }
    Ok(())
}

fn validate_complex_square_finite_view(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<(), CholeskyError> {
    if matrix.is_empty() {
        return Err(CholeskyError::EmptyMatrix);
    }
    if matrix.nrows() != matrix.ncols() {
        return Err(CholeskyError::NotSquare);
    }
    if matrix.iter().any(|value| !value.re.is_finite() || !value.im.is_finite()) {
        return Err(CholeskyError::NumericalInstability);
    }
    Ok(())
}

#[cfg(not(feature = "openblas-system"))]
#[allow(clippy::many_single_char_names)]
fn decompose_complex_internal(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<Array2<Complex64>, CholeskyError> {
    validate_complex_square_finite_view(matrix)?;

    let n = matrix.nrows();
    let mut l = Array2::<Complex64>::zeros((n, n));

    for i in 0..n {
        for j in 0..=i {
            let mut sum = matrix[[i, j]];
            for k in 0..j {
                sum -= l[[i, k]] * l[[j, k]].conj();
            }

            if i == j {
                if sum.im.abs() > DenseKernelPolicy::BASE_TOLERANCE
                    || sum.re <= DenseKernelPolicy::BASE_TOLERANCE
                {
                    return Err(CholeskyError::NotPositiveDefinite);
                }
                l[[i, j]] = Complex64::new(sum.re.sqrt(), 0.0);
            } else {
                let diagonal = l[[j, j]];
                if diagonal.norm() <= DenseKernelPolicy::BASE_TOLERANCE {
                    return Err(CholeskyError::NotPositiveDefinite);
                }
                l[[i, j]] = sum / diagonal;
            }
        }
    }

    Ok(l)
}

#[cfg(not(feature = "openblas-system"))]
fn decompose_internal(matrix: &ArrayView2<'_, f64>) -> Result<Array2<f64>, CholeskyError> {
    validate_square_finite_view(matrix)?;

    let n = matrix.nrows();
    let mut lower = Array2::<f64>::zeros((n, n));

    if let (Some(input), Some(data)) =
        (matrix.as_slice_memory_order(), lower.as_slice_memory_order_mut())
    {
        for i in 0..n {
            let row_i = i * n;
            for j in 0..=i {
                let row_j = j * n;
                let mut sum = {
                    // SAFETY: `row_i + j < n * n` from loop bounds.
                    unsafe { *input.get_unchecked(row_i + j) }
                };
                let mut dot = 0.0_f64;
                let mut k = 0;
                while k + 3 < j {
                    dot += {
                        // SAFETY: offsets are bounded by loop invariants.
                        unsafe {
                            *data.get_unchecked(row_i + k) * *data.get_unchecked(row_j + k)
                                + *data.get_unchecked(row_i + k + 1)
                                    * *data.get_unchecked(row_j + k + 1)
                                + *data.get_unchecked(row_i + k + 2)
                                    * *data.get_unchecked(row_j + k + 2)
                                + *data.get_unchecked(row_i + k + 3)
                                    * *data.get_unchecked(row_j + k + 3)
                        }
                    };
                    k += 4;
                }
                while k < j {
                    dot += {
                        // SAFETY: row/column offsets are bounded by loop invariants.
                        unsafe { *data.get_unchecked(row_i + k) * *data.get_unchecked(row_j + k) }
                    };
                    k += 1;
                }
                sum -= dot;

                if i == j {
                    if sum <= DenseKernelPolicy::BASE_TOLERANCE {
                        return Err(CholeskyError::NotPositiveDefinite);
                    }
                    // SAFETY: `row_i + j < n * n` from loop bounds.
                    unsafe {
                        *data.get_unchecked_mut(row_i + j) = sum.sqrt();
                    }
                } else {
                    let diagonal = {
                        // SAFETY: `row_j + j < n * n` from loop bounds.
                        unsafe { *data.get_unchecked(row_j + j) }
                    };
                    if diagonal.abs() <= DenseKernelPolicy::BASE_TOLERANCE {
                        return Err(CholeskyError::NotPositiveDefinite);
                    }
                    // SAFETY: `row_i + j < n * n` from loop bounds.
                    unsafe {
                        *data.get_unchecked_mut(row_i + j) = sum / diagonal;
                    }
                }
            }
        }
        Ok(lower)
    } else {
        for i in 0..n {
            for j in 0..=i {
                let mut sum = matrix[[i, j]];
                for k in 0..j {
                    sum -= lower[[i, k]] * lower[[j, k]];
                }

                if i == j {
                    if sum <= DenseKernelPolicy::BASE_TOLERANCE {
                        return Err(CholeskyError::NotPositiveDefinite);
                    }
                    lower[[i, j]] = sum.sqrt();
                } else {
                    let diagonal = lower[[j, j]];
                    if diagonal.abs() <= DenseKernelPolicy::BASE_TOLERANCE {
                        return Err(CholeskyError::NotPositiveDefinite);
                    }
                    lower[[i, j]] = sum / diagonal;
                }
            }
        }
        Ok(lower)
    }
}

#[cfg(feature = "openblas-system")]
fn decompose_provider(matrix: &ArrayView2<'_, f64>) -> Result<Array2<f64>, CholeskyError> {
    use ndarray_linalg::{Cholesky as _, UPLO};

    validate_square_finite_view(matrix)?;

    matrix.cholesky(UPLO::Lower).map_err(|_| CholeskyError::NotPositiveDefinite)
}

#[cfg(feature = "openblas-system")]
fn solve_provider(
    matrix: &ArrayView2<'_, f64>,
    rhs: &ArrayView1<'_, f64>,
) -> Result<Array1<f64>, CholeskyError> {
    use ndarray_linalg::SolveC as _;

    validate_square_finite_view(matrix)?;
    matrix.solvec(rhs).map_err(|_| CholeskyError::NotPositiveDefinite)
}

#[cfg(feature = "openblas-system")]
fn inverse_provider(matrix: &ArrayView2<'_, f64>) -> Result<Array2<f64>, CholeskyError> {
    use ndarray_linalg::InverseC as _;

    validate_square_finite_view(matrix)?;
    matrix.invc().map_err(|_| CholeskyError::NotPositiveDefinite)
}

#[cfg(feature = "openblas-system")]
fn decompose_complex_provider(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<Array2<Complex64>, CholeskyError> {
    use ndarray_linalg::{Cholesky as _, UPLO};

    validate_complex_square_finite_view(matrix)?;
    matrix.cholesky(UPLO::Lower).map_err(|_| CholeskyError::NotPositiveDefinite)
}

#[cfg(feature = "openblas-system")]
fn solve_complex_provider(
    matrix: &ArrayView2<'_, Complex64>,
    rhs: &ArrayView1<'_, Complex64>,
) -> Result<Array1<Complex64>, CholeskyError> {
    use ndarray_linalg::SolveC as _;

    validate_complex_square_finite_view(matrix)?;
    if rhs.len() != matrix.nrows() {
        return Err(CholeskyError::InvalidInput(
            "RHS length must match matrix dimensions".to_string(),
        ));
    }
    matrix.solvec(rhs).map_err(|_| CholeskyError::NotPositiveDefinite)
}

#[cfg(feature = "openblas-system")]
fn inverse_complex_provider(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<Array2<Complex64>, CholeskyError> {
    use ndarray_linalg::InverseC as _;

    validate_complex_square_finite_view(matrix)?;
    matrix.invc().map_err(|_| CholeskyError::NotPositiveDefinite)
}

#[cfg(not(feature = "openblas-system"))]
#[allow(clippy::many_single_char_names)]
fn solve_complex_from_factor(
    lower_factor: &Array2<Complex64>,
    rhs: &ArrayView1<'_, Complex64>,
) -> Result<Array1<Complex64>, CholeskyError> {
    let size = lower_factor.nrows();
    if rhs.len() != size {
        return Err(CholeskyError::InvalidInput(
            "RHS length must match matrix dimensions".to_string(),
        ));
    }

    let mut y = Array1::<Complex64>::zeros(size);
    for i in 0..size {
        let mut sum = rhs[i];
        for j in 0..i {
            sum -= lower_factor[[i, j]] * y[j];
        }
        let diagonal = lower_factor[[i, i]];
        if diagonal.norm() <= DenseKernelPolicy::BASE_TOLERANCE {
            return Err(CholeskyError::NotPositiveDefinite);
        }
        y[i] = sum / diagonal;
    }

    let mut x = Array1::<Complex64>::zeros(size);
    for i_rev in 0..size {
        let i = size - 1 - i_rev;
        let mut sum = y[i];
        for j in (i + 1)..size {
            sum -= lower_factor[[j, i]].conj() * x[j];
        }
        let diagonal = lower_factor[[i, i]].conj();
        if diagonal.norm() <= DenseKernelPolicy::BASE_TOLERANCE {
            return Err(CholeskyError::NotPositiveDefinite);
        }
        x[i] = sum / diagonal;
    }

    Ok(x)
}

fn decompose_dispatch(matrix: &ArrayView2<'_, f64>) -> Result<Array2<f64>, CholeskyError> {
    #[cfg(feature = "openblas-system")]
    {
        decompose_provider(matrix)
    }
    #[cfg(not(feature = "openblas-system"))]
    {
        decompose_internal(matrix)
    }
}

#[cfg(not(feature = "openblas-system"))]
#[allow(clippy::many_single_char_names)]
#[allow(clippy::too_many_lines)]
fn solve_from_factor(
    lower_factor: &Array2<f64>,
    rhs: &ArrayView1<'_, f64>,
    output: &mut Array1<f64>,
) -> Result<(), CholeskyError> {
    let size = lower_factor.nrows();
    if rhs.len() != size || output.len() != size {
        return Err(CholeskyError::InvalidInput(
            "RHS/output length must match matrix dimensions".to_string(),
        ));
    }

    if let (Some(factor), Some(rhs_slice), Some(output_slice)) = (
        lower_factor.as_slice_memory_order(),
        rhs.as_slice_memory_order(),
        output.as_slice_memory_order_mut(),
    ) {
        let mut y = vec![0.0_f64; size];
        for i in 0..size {
            let row_i = i * size;
            let mut sum = {
                // SAFETY: `i < rhs_slice.len() == size`.
                unsafe { *rhs_slice.get_unchecked(i) }
            };
            let mut dot = 0.0_f64;
            let mut j = 0;
            while j + 3 < i {
                dot += {
                    // SAFETY: offsets are bounded by loop bounds.
                    unsafe {
                        *factor.get_unchecked(row_i + j) * *y.get_unchecked(j)
                            + *factor.get_unchecked(row_i + j + 1) * *y.get_unchecked(j + 1)
                            + *factor.get_unchecked(row_i + j + 2) * *y.get_unchecked(j + 2)
                            + *factor.get_unchecked(row_i + j + 3) * *y.get_unchecked(j + 3)
                    }
                };
                j += 4;
            }
            while j < i {
                dot += {
                    // SAFETY: offsets are bounded by loop bounds.
                    unsafe { *factor.get_unchecked(row_i + j) * *y.get_unchecked(j) }
                };
                j += 1;
            }
            sum -= dot;
            let diagonal = {
                // SAFETY: `row_i + i < size * size`.
                unsafe { *factor.get_unchecked(row_i + i) }
            };
            if diagonal.abs() <= DenseKernelPolicy::BASE_TOLERANCE {
                return Err(CholeskyError::NotPositiveDefinite);
            }
            // SAFETY: `i < y.len()`.
            unsafe {
                *y.get_unchecked_mut(i) = sum / diagonal;
            }
        }

        for i_rev in 0..size {
            let i = size - 1 - i_rev;
            let mut sum = {
                // SAFETY: `i < y.len()`.
                unsafe { *y.get_unchecked(i) }
            };
            let mut dot = 0.0_f64;
            let mut j = i + 1;
            while j + 3 < size {
                dot += {
                    // SAFETY: offsets are bounded by loop bounds.
                    unsafe {
                        *factor.get_unchecked(j * size + i) * *output_slice.get_unchecked(j)
                            + *factor.get_unchecked((j + 1) * size + i)
                                * *output_slice.get_unchecked(j + 1)
                            + *factor.get_unchecked((j + 2) * size + i)
                                * *output_slice.get_unchecked(j + 2)
                            + *factor.get_unchecked((j + 3) * size + i)
                                * *output_slice.get_unchecked(j + 3)
                    }
                };
                j += 4;
            }
            while j < size {
                dot += {
                    // SAFETY: offsets are bounded by loop bounds.
                    unsafe { *factor.get_unchecked(j * size + i) * *output_slice.get_unchecked(j) }
                };
                j += 1;
            }
            sum -= dot;
            let diagonal = {
                // SAFETY: `i * size + i < size * size`.
                unsafe { *factor.get_unchecked(i * size + i) }
            };
            if diagonal.abs() <= DenseKernelPolicy::BASE_TOLERANCE {
                return Err(CholeskyError::NotPositiveDefinite);
            }
            // SAFETY: `i < output_slice.len()`.
            unsafe {
                *output_slice.get_unchecked_mut(i) = sum / diagonal;
            }
        }
        Ok(())
    } else {
        let mut y = Array1::<f64>::zeros(size);
        for i in 0..size {
            let mut sum = rhs[i];
            for j in 0..i {
                sum -= lower_factor[[i, j]] * y[j];
            }
            y[i] = sum / lower_factor[[i, i]];
        }

        for i_rev in 0..size {
            let i = size - 1 - i_rev;
            let mut sum = y[i];
            for j in (i + 1)..size {
                sum -= lower_factor[[j, i]] * output[j];
            }
            output[i] = sum / lower_factor[[i, i]];
        }

        Ok(())
    }
}

#[cfg(not(feature = "openblas-system"))]
#[allow(clippy::many_single_char_names)]
fn inverse_from_factor(lower_factor: &Array2<f64>) -> Result<Array2<f64>, CholeskyError> {
    let n = lower_factor.nrows();
    if let Some(factor) = lower_factor.as_slice_memory_order() {
        let mut inv_lower = vec![0.0_f64; n * n];
        for col in 0..n {
            let diag = {
                // SAFETY: `col * n + col < n * n`.
                unsafe { *factor.get_unchecked(col * n + col) }
            };
            if diag.abs() <= DenseKernelPolicy::BASE_TOLERANCE {
                return Err(CholeskyError::NotPositiveDefinite);
            }
            // SAFETY: `col * n + col < n * n`.
            unsafe {
                *inv_lower.get_unchecked_mut(col * n + col) = 1.0 / diag;
            }

            for row in (col + 1)..n {
                let row_offset = row * n;
                let mut sum = 0.0_f64;
                for k in col..row {
                    sum += {
                        // SAFETY: offsets are bounded by loop bounds.
                        unsafe {
                            *factor.get_unchecked(row_offset + k)
                                * *inv_lower.get_unchecked(k * n + col)
                        }
                    };
                }

                let row_diag = {
                    // SAFETY: `row_offset + row < n * n`.
                    unsafe { *factor.get_unchecked(row_offset + row) }
                };
                if row_diag.abs() <= DenseKernelPolicy::BASE_TOLERANCE {
                    return Err(CholeskyError::NotPositiveDefinite);
                }
                // SAFETY: `row_offset + col < n * n`.
                unsafe {
                    *inv_lower.get_unchecked_mut(row_offset + col) = -sum / row_diag;
                }
            }
        }

        let mut inverse = Array2::<f64>::zeros((n, n));
        if let Some(inverse_data) = inverse.as_slice_memory_order_mut() {
            for i in 0..n {
                let row_i = i * n;
                for j in 0..=i {
                    let mut sum = 0.0_f64;
                    for k in usize::max(i, j)..n {
                        sum += {
                            // SAFETY: offsets are bounded by loop bounds.
                            unsafe {
                                *inv_lower.get_unchecked(k * n + i)
                                    * *inv_lower.get_unchecked(k * n + j)
                            }
                        };
                    }
                    // SAFETY: `row_i + j < n * n`.
                    unsafe {
                        *inverse_data.get_unchecked_mut(row_i + j) = sum;
                    }
                    if i != j {
                        // SAFETY: `j * n + i < n * n`.
                        unsafe {
                            *inverse_data.get_unchecked_mut(j * n + i) = sum;
                        }
                    }
                }
            }
            Ok(inverse)
        } else {
            Err(CholeskyError::InvalidInput("inverse storage must be contiguous".to_string()))
        }
    } else {
        Err(CholeskyError::InvalidInput("factor storage must be contiguous".to_string()))
    }
}

/// Compute Cholesky decomposition.
///
/// # Errors
/// Returns an error if matrix is not SPD.
pub fn decompose(matrix: &Array2<f64>) -> Result<NdarrayCholeskyResult, CholeskyError> {
    let l = decompose_dispatch(&matrix.view())?;
    Ok(NdarrayCholeskyResult { l })
}

/// Compute complex Cholesky decomposition.
///
/// # Errors
/// Returns an error if matrix is not Hermitian positive definite.
pub fn decompose_complex(
    matrix: &Array2<Complex64>,
) -> Result<NdarrayComplexCholeskyResult, CholeskyError> {
    decompose_complex_impl(&matrix.view())
}

fn decompose_complex_impl(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<NdarrayComplexCholeskyResult, CholeskyError> {
    #[cfg(feature = "openblas-system")]
    {
        let l = decompose_complex_provider(matrix)?;
        Ok(NdarrayComplexCholeskyResult { l })
    }
    #[cfg(not(feature = "openblas-system"))]
    {
        let l = decompose_complex_internal(matrix)?;
        Ok(NdarrayComplexCholeskyResult { l })
    }
}

/// Compute Cholesky decomposition from a matrix view.
///
/// # Errors
/// Returns an error if matrix is not SPD.
pub fn decompose_view(
    matrix: &ArrayView2<'_, f64>,
) -> Result<NdarrayCholeskyResult, CholeskyError> {
    let l = decompose_dispatch(matrix)?;
    Ok(NdarrayCholeskyResult { l })
}

/// Compute complex Cholesky decomposition from a matrix view.
///
/// # Errors
/// Returns an error if matrix is not Hermitian positive definite or provider
/// support is unavailable.
pub fn decompose_complex_view(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<NdarrayComplexCholeskyResult, CholeskyError> {
    decompose_complex_impl(matrix)
}

/// Solve `Ax=b` using Cholesky decomposition.
///
/// # Errors
/// Returns an error for invalid dimensions or non-SPD matrix.
pub fn solve(matrix: &Array2<f64>, rhs: &Array1<f64>) -> Result<Array1<f64>, CholeskyError> {
    solve_impl(&matrix.view(), &rhs.view())
}

fn solve_impl(
    matrix: &ArrayView2<'_, f64>,
    rhs: &ArrayView1<'_, f64>,
) -> Result<Array1<f64>, CholeskyError> {
    #[cfg(feature = "openblas-system")]
    {
        solve_provider(matrix, rhs)
    }
    #[cfg(not(feature = "openblas-system"))]
    {
        let mut output = Array1::<f64>::zeros(rhs.len());
        solve_into_impl(matrix, rhs, &mut output)?;
        Ok(output)
    }
}

/// Solve complex-valued `Ax=b` using Cholesky decomposition.
///
/// # Errors
/// Returns an error for invalid dimensions or non-HPD matrix.
pub fn solve_complex(
    matrix: &Array2<Complex64>,
    rhs: &Array1<Complex64>,
) -> Result<Array1<Complex64>, CholeskyError> {
    solve_complex_impl(&matrix.view(), &rhs.view())
}

fn solve_complex_impl(
    matrix: &ArrayView2<'_, Complex64>,
    rhs: &ArrayView1<'_, Complex64>,
) -> Result<Array1<Complex64>, CholeskyError> {
    #[cfg(feature = "openblas-system")]
    {
        solve_complex_provider(matrix, rhs)
    }
    #[cfg(not(feature = "openblas-system"))]
    {
        let lower_factor = decompose_complex_internal(matrix)?;
        solve_complex_from_factor(&lower_factor, rhs)
    }
}

/// Solve `Ax=b` using Cholesky decomposition from matrix/vector views.
///
/// # Errors
/// Returns an error for invalid dimensions or non-SPD matrix.
pub fn solve_view(
    matrix: &ArrayView2<'_, f64>,
    rhs: &ArrayView1<'_, f64>,
) -> Result<Array1<f64>, CholeskyError> {
    solve_impl(matrix, rhs)
}

/// Solve complex-valued `Ax=b` from matrix/vector views.
///
/// # Errors
/// Returns an error for invalid dimensions or non-HPD matrix.
pub fn solve_complex_view(
    matrix: &ArrayView2<'_, Complex64>,
    rhs: &ArrayView1<'_, Complex64>,
) -> Result<Array1<Complex64>, CholeskyError> {
    solve_complex_impl(matrix, rhs)
}

/// Solve `Ax=b` into `output` using Cholesky decomposition.
///
/// # Errors
/// Returns an error for invalid dimensions or non-SPD matrix.
#[allow(clippy::many_single_char_names)]
pub fn solve_into(
    matrix: &Array2<f64>,
    rhs: &Array1<f64>,
    output: &mut Array1<f64>,
) -> Result<(), CholeskyError> {
    solve_into_impl(&matrix.view(), &rhs.view(), output)
}

#[allow(clippy::many_single_char_names)]
fn solve_into_impl(
    matrix: &ArrayView2<'_, f64>,
    rhs: &ArrayView1<'_, f64>,
    output: &mut Array1<f64>,
) -> Result<(), CholeskyError> {
    if rhs.len() != matrix.nrows() {
        return Err(CholeskyError::InvalidInput(
            "RHS length must match matrix dimensions".to_string(),
        ));
    }
    if output.len() != rhs.len() {
        return Err(CholeskyError::InvalidInput("output length must match rhs length".to_string()));
    }

    #[cfg(feature = "openblas-system")]
    {
        let solution = solve_provider(matrix, rhs)?;
        output.assign(&solution);
        Ok(())
    }
    #[cfg(not(feature = "openblas-system"))]
    {
        let lower_factor = decompose_dispatch(matrix)?;
        solve_from_factor(&lower_factor, rhs, output)
    }
}

/// Compute inverse via Cholesky decomposition.
///
/// # Errors
/// Returns an error if matrix is not SPD.
pub fn inverse(matrix: &Array2<f64>) -> Result<Array2<f64>, CholeskyError> {
    inverse_impl(&matrix.view())
}

fn inverse_impl(matrix: &ArrayView2<'_, f64>) -> Result<Array2<f64>, CholeskyError> {
    #[cfg(feature = "openblas-system")]
    {
        inverse_provider(matrix)
    }
    #[cfg(not(feature = "openblas-system"))]
    {
        let lower_factor = decompose_dispatch(matrix)?;
        inverse_from_factor(&lower_factor)
    }
}

/// Compute complex inverse via Cholesky decomposition.
///
/// # Errors
/// Returns an error if matrix is not HPD.
pub fn inverse_complex(matrix: &Array2<Complex64>) -> Result<Array2<Complex64>, CholeskyError> {
    inverse_complex_impl(&matrix.view())
}

fn inverse_complex_impl(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<Array2<Complex64>, CholeskyError> {
    #[cfg(feature = "openblas-system")]
    {
        inverse_complex_provider(matrix)
    }
    #[cfg(not(feature = "openblas-system"))]
    {
        let lower_factor = decompose_complex_internal(matrix)?;
        let size = lower_factor.nrows();
        let mut inverse = Array2::<Complex64>::zeros((size, size));

        for col in 0..size {
            let mut basis = Array1::<Complex64>::zeros(size);
            basis[col] = Complex64::new(1.0, 0.0);
            let solution = solve_complex_from_factor(&lower_factor, &basis.view())?;
            for row in 0..size {
                inverse[[row, col]] = solution[row];
            }
        }

        Ok(inverse)
    }
}

/// Compute inverse via Cholesky decomposition from a matrix view.
///
/// # Errors
/// Returns an error if matrix is not SPD.
pub fn inverse_view(matrix: &ArrayView2<'_, f64>) -> Result<Array2<f64>, CholeskyError> {
    inverse_impl(matrix)
}

/// Compute complex inverse from a matrix view.
///
/// # Errors
/// Returns an error if matrix is not HPD.
pub fn inverse_complex_view(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<Array2<Complex64>, CholeskyError> {
    inverse_complex_impl(matrix)
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2};
    use num_complex::Complex64;

    use super::*;

    #[test]
    fn cholesky_reconstructs_spd_matrix() {
        let matrix = Array2::from_shape_vec((2, 2), vec![4.0, 2.0, 2.0, 3.0]).unwrap();
        let decomposition = decompose(&matrix).unwrap();
        let reconstructed = decomposition.l.dot(&decomposition.l.t());
        for i in 0..2 {
            for j in 0..2 {
                assert!((matrix[[i, j]] - reconstructed[[i, j]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn cholesky_view_variants_match_owned() {
        let matrix = Array2::from_shape_vec((2, 2), vec![9.0, 3.0, 3.0, 5.0]).unwrap();
        let rhs = Array1::from_vec(vec![12.0, 8.0]);

        let owned = decompose(&matrix).unwrap();
        let viewed = decompose_view(&matrix.view()).unwrap();
        assert_eq!(owned.l.dim(), viewed.l.dim());

        let solution_owned = solve(&matrix, &rhs).unwrap();
        let solution_viewed = solve_view(&matrix.view(), &rhs.view()).unwrap();
        for i in 0..rhs.len() {
            assert!((solution_owned[i] - solution_viewed[i]).abs() < 1e-12);
        }

        let inverse_owned = inverse(&matrix).unwrap();
        let inverse_viewed = inverse_view(&matrix.view()).unwrap();
        for i in 0..matrix.nrows() {
            for j in 0..matrix.ncols() {
                assert!((inverse_owned[[i, j]] - inverse_viewed[[i, j]]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn solve_reconstructs_rhs() {
        let matrix = Array2::from_shape_vec((2, 2), vec![4.0, 2.0, 2.0, 3.0]).unwrap();
        let rhs = Array1::from_vec(vec![1.0, 1.0]);
        let x = solve(&matrix, &rhs).unwrap();
        let reconstructed = matrix.dot(&x);
        assert!((reconstructed[0] - rhs[0]).abs() < 1e-10);
        assert!((reconstructed[1] - rhs[1]).abs() < 1e-10);
    }

    #[test]
    fn non_spd_input_errors() {
        let matrix = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 2.0, 1.0]).unwrap();
        let result = decompose(&matrix);
        assert!(matches!(result, Err(CholeskyError::NotPositiveDefinite)));
    }

    #[test]
    fn inverse_multiplied_by_matrix_is_identity() {
        let matrix = Array2::from_shape_vec((2, 2), vec![4.0, 2.0, 2.0, 3.0]).unwrap();
        let inverse = inverse(&matrix).unwrap();
        let product = matrix.dot(&inverse);
        assert!((product[[0, 0]] - 1.0).abs() < 1e-8);
        assert!((product[[1, 1]] - 1.0).abs() < 1e-8);
        assert!(product[[0, 1]].abs() < 1e-8);
        assert!(product[[1, 0]].abs() < 1e-8);
    }

    #[test]
    fn solve_into_rejects_bad_output_length() {
        let matrix = Array2::from_shape_vec((2, 2), vec![4.0, 2.0, 2.0, 3.0]).unwrap();
        let rhs = Array1::from_vec(vec![1.0, 1.0]);
        let mut output = Array1::from_vec(vec![0.0]);
        let result = solve_into(&matrix, &rhs, &mut output);
        assert!(matches!(result, Err(CholeskyError::InvalidInput(_))));
    }

    #[test]
    fn solve_into_matches_solve() {
        let matrix = Array2::from_shape_vec((2, 2), vec![5.0, 1.0, 1.0, 2.0]).unwrap();
        let rhs = Array1::from_vec(vec![3.0, 4.0]);
        let expected = solve(&matrix, &rhs).unwrap();
        let mut output = Array1::<f64>::zeros(2);
        solve_into(&matrix, &rhs, &mut output).unwrap();
        assert!((output[0] - expected[0]).abs() < 1e-10);
        assert!((output[1] - expected[1]).abs() < 1e-10);
    }

    #[test]
    fn solve_rejects_bad_rhs_length() {
        let matrix = Array2::eye(2);
        let rhs = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let mut output = Array1::<f64>::zeros(3);
        let result = solve_into(&matrix, &rhs, &mut output);
        assert!(matches!(result, Err(CholeskyError::InvalidInput(_))));
    }

    #[test]
    fn decompose_rejects_non_finite_input() {
        let matrix = Array2::from_shape_vec((2, 2), vec![1.0, f64::NAN, 0.0, 1.0]).unwrap();
        let result = decompose(&matrix);
        assert!(matches!(result, Err(CholeskyError::NumericalInstability)));
    }

    #[test]
    fn complex_cholesky_paths_work() {
        let matrix = Array2::from_shape_vec((2, 2), vec![
            Complex64::new(5.0, 0.0),
            Complex64::new(1.0, -1.0),
            Complex64::new(1.0, 1.0),
            Complex64::new(4.0, 0.0),
        ])
        .unwrap();
        let rhs = Array1::from_vec(vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)]);

        let factor = decompose_complex(&matrix).unwrap();
        assert_eq!(factor.l.dim(), (2, 2));

        let solution = solve_complex(&matrix, &rhs).unwrap();
        let reconstructed = matrix.dot(&solution);
        for i in 0..rhs.len() {
            assert!((reconstructed[i] - rhs[i]).norm() < 1e-8);
        }

        let inverse = inverse_complex(&matrix).unwrap();
        let product = matrix.dot(&inverse);
        assert!((product[[0, 0]] - Complex64::new(1.0, 0.0)).norm() < 1e-8);
        assert!((product[[1, 1]] - Complex64::new(1.0, 0.0)).norm() < 1e-8);
        assert!(product[[0, 1]].norm() < 1e-8);
        assert!(product[[1, 0]].norm() < 1e-8);

        let view_factor = decompose_complex_view(&matrix.view()).unwrap();
        let view_solution = solve_complex_view(&matrix.view(), &rhs.view()).unwrap();
        let view_inverse = inverse_complex_view(&matrix.view()).unwrap();
        assert_eq!(factor.l.dim(), view_factor.l.dim());
        for i in 0..rhs.len() {
            assert!((solution[i] - view_solution[i]).norm() < 1e-10);
        }
        for i in 0..2 {
            for j in 0..2 {
                assert!((inverse[[i, j]] - view_inverse[[i, j]]).norm() < 1e-10);
            }
        }
    }
}
