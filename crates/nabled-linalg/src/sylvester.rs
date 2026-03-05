//! Sylvester and Lyapunov solvers over ndarray matrices.

use std::fmt;

use nabled_core::scalar::NabledReal;
use ndarray::{Array1, Array2, ArrayView2};
use num_complex::Complex64;

use crate::lu;

/// Error type for Sylvester/Lyapunov solvers.
#[derive(Debug, Clone, PartialEq)]
pub enum SylvesterError {
    /// Matrix input is empty.
    EmptyMatrix,
    /// Matrix must be square.
    NotSquare,
    /// Input dimensions are incompatible.
    DimensionMismatch,
    /// Linear system is singular.
    SingularSystem,
    /// Invalid input.
    InvalidInput(String),
}

impl fmt::Display for SylvesterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SylvesterError::EmptyMatrix => write!(f, "Matrix cannot be empty"),
            SylvesterError::NotSquare => write!(f, "Matrix must be square"),
            SylvesterError::DimensionMismatch => write!(f, "Input dimensions are incompatible"),
            SylvesterError::SingularSystem => write!(f, "Sylvester system is singular"),
            SylvesterError::InvalidInput(message) => write!(f, "Invalid input: {message}"),
        }
    }
}

impl std::error::Error for SylvesterError {}

/// Reusable workspace for Sylvester/Lyapunov solves.
#[derive(Debug, Clone)]
pub struct SylvesterWorkspace<T: NabledReal = f64> {
    coefficient: Array2<T>,
    rhs:         Array1<T>,
    solution:    Array1<T>,
}

impl<T: NabledReal> SylvesterWorkspace<T> {
    fn ensure_dims(&mut self, rows: usize, cols: usize) {
        let system_size = rows * cols;
        if self.coefficient.dim() == (system_size, system_size) {
            self.coefficient.fill(T::zero());
        } else {
            self.coefficient = Array2::<T>::zeros((system_size, system_size));
        }
        if self.rhs.len() == system_size {
            self.rhs.fill(T::zero());
        } else {
            self.rhs = Array1::<T>::zeros(system_size);
        }
        if self.solution.len() != system_size {
            self.solution = Array1::<T>::zeros(system_size);
        }
    }
}

impl<T: NabledReal> Default for SylvesterWorkspace<T> {
    fn default() -> Self {
        Self {
            coefficient: Array2::<T>::zeros((0, 0)),
            rhs:         Array1::<T>::zeros(0),
            solution:    Array1::<T>::zeros(0),
        }
    }
}

/// Reusable workspace for complex Sylvester/Lyapunov solves.
#[derive(Debug, Clone, Default)]
pub struct SylvesterComplexWorkspace {
    coefficient: Array2<Complex64>,
    rhs:         Array1<Complex64>,
    solution:    Array1<Complex64>,
}

impl SylvesterComplexWorkspace {
    fn ensure_dims(&mut self, rows: usize, cols: usize) {
        let system_size = rows * cols;
        if self.coefficient.dim() == (system_size, system_size) {
            self.coefficient.fill(Complex64::new(0.0, 0.0));
        } else {
            self.coefficient = Array2::<Complex64>::zeros((system_size, system_size));
        }
        if self.rhs.len() == system_size {
            self.rhs.fill(Complex64::new(0.0, 0.0));
        } else {
            self.rhs = Array1::<Complex64>::zeros(system_size);
        }
        if self.solution.len() != system_size {
            self.solution = Array1::<Complex64>::zeros(system_size);
        }
    }
}

fn validate_sylvester_dims(
    matrix_a: &ArrayView2<'_, impl NabledReal>,
    matrix_b: &ArrayView2<'_, impl NabledReal>,
    matrix_c: &ArrayView2<'_, impl NabledReal>,
) -> Result<(usize, usize), SylvesterError> {
    if matrix_a.is_empty() || matrix_b.is_empty() || matrix_c.is_empty() {
        return Err(SylvesterError::EmptyMatrix);
    }
    if matrix_a.nrows() != matrix_a.ncols() || matrix_b.nrows() != matrix_b.ncols() {
        return Err(SylvesterError::NotSquare);
    }

    let n = matrix_a.nrows();
    let m = matrix_b.nrows();
    if matrix_c.dim() != (n, m) {
        return Err(SylvesterError::DimensionMismatch);
    }
    Ok((n, m))
}

fn validate_sylvester_complex_dims(
    matrix_a: &ArrayView2<'_, Complex64>,
    matrix_b: &ArrayView2<'_, Complex64>,
    matrix_c: &ArrayView2<'_, Complex64>,
) -> Result<(usize, usize), SylvesterError> {
    if matrix_a.is_empty() || matrix_b.is_empty() || matrix_c.is_empty() {
        return Err(SylvesterError::EmptyMatrix);
    }
    if matrix_a.nrows() != matrix_a.ncols() || matrix_b.nrows() != matrix_b.ncols() {
        return Err(SylvesterError::NotSquare);
    }
    if matrix_a.iter().any(|value| !value.re.is_finite() || !value.im.is_finite())
        || matrix_b.iter().any(|value| !value.re.is_finite() || !value.im.is_finite())
        || matrix_c.iter().any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(SylvesterError::InvalidInput(
            "complex matrix inputs must be finite".to_string(),
        ));
    }

    let n = matrix_a.nrows();
    let m = matrix_b.nrows();
    if matrix_c.dim() != (n, m) {
        return Err(SylvesterError::DimensionMismatch);
    }
    Ok((n, m))
}

/// Solve Sylvester equation `A X + X B = C`.
///
/// # Errors
/// Returns an error if dimensions are invalid or system is singular.
#[cfg(feature = "lapack-provider")]
pub fn solve_sylvester<T>(
    matrix_a: &Array2<T>,
    matrix_b: &Array2<T>,
    matrix_c: &Array2<T>,
) -> Result<Array2<T>, SylvesterError>
where
    T: NabledReal + ndarray_linalg::Lapack,
{
    let (n, m) = validate_sylvester_dims(&matrix_a.view(), &matrix_b.view(), &matrix_c.view())?;
    let mut workspace = SylvesterWorkspace::default();
    let mut output = Array2::<T>::zeros((n, m));
    solve_sylvester_with_workspace_into_impl(
        &matrix_a.view(),
        &matrix_b.view(),
        &matrix_c.view(),
        &mut output,
        &mut workspace,
    )?;
    Ok(output)
}

/// Solve Sylvester equation `A X + X B = C`.
///
/// # Errors
/// Returns an error if dimensions are invalid or system is singular.
#[cfg(not(feature = "lapack-provider"))]
pub fn solve_sylvester<T: NabledReal>(
    matrix_a: &Array2<T>,
    matrix_b: &Array2<T>,
    matrix_c: &Array2<T>,
) -> Result<Array2<T>, SylvesterError> {
    let (n, m) = validate_sylvester_dims(&matrix_a.view(), &matrix_b.view(), &matrix_c.view())?;
    let mut workspace = SylvesterWorkspace::default();
    let mut output = Array2::<T>::zeros((n, m));
    solve_sylvester_with_workspace_into_impl(
        &matrix_a.view(),
        &matrix_b.view(),
        &matrix_c.view(),
        &mut output,
        &mut workspace,
    )?;
    Ok(output)
}

/// Solve Sylvester equation `A X + X B = C` from matrix views.
///
/// # Errors
/// Returns an error if dimensions are invalid or system is singular.
#[cfg(feature = "lapack-provider")]
pub fn solve_sylvester_view<T>(
    matrix_a: &ArrayView2<'_, T>,
    matrix_b: &ArrayView2<'_, T>,
    matrix_c: &ArrayView2<'_, T>,
) -> Result<Array2<T>, SylvesterError>
where
    T: NabledReal + ndarray_linalg::Lapack,
{
    let (n, m) = validate_sylvester_dims(matrix_a, matrix_b, matrix_c)?;
    let mut workspace = SylvesterWorkspace::default();
    let mut output = Array2::<T>::zeros((n, m));
    solve_sylvester_with_workspace_into_impl(
        matrix_a,
        matrix_b,
        matrix_c,
        &mut output,
        &mut workspace,
    )?;
    Ok(output)
}

/// Solve Sylvester equation `A X + X B = C` from matrix views.
///
/// # Errors
/// Returns an error if dimensions are invalid or system is singular.
#[cfg(not(feature = "lapack-provider"))]
pub fn solve_sylvester_view<T: NabledReal>(
    matrix_a: &ArrayView2<'_, T>,
    matrix_b: &ArrayView2<'_, T>,
    matrix_c: &ArrayView2<'_, T>,
) -> Result<Array2<T>, SylvesterError> {
    let (n, m) = validate_sylvester_dims(matrix_a, matrix_b, matrix_c)?;
    let mut workspace = SylvesterWorkspace::default();
    let mut output = Array2::<T>::zeros((n, m));
    solve_sylvester_with_workspace_into_impl(
        matrix_a,
        matrix_b,
        matrix_c,
        &mut output,
        &mut workspace,
    )?;
    Ok(output)
}

/// Solve Sylvester equation `A X + X B = C` into `output`.
///
/// # Errors
/// Returns an error if dimensions are invalid or system is singular.
#[cfg(feature = "lapack-provider")]
pub fn solve_sylvester_into<T>(
    matrix_a: &Array2<T>,
    matrix_b: &Array2<T>,
    matrix_c: &Array2<T>,
    output: &mut Array2<T>,
) -> Result<(), SylvesterError>
where
    T: NabledReal + ndarray_linalg::Lapack,
{
    let mut workspace = SylvesterWorkspace::default();
    solve_sylvester_with_workspace_into_impl(
        &matrix_a.view(),
        &matrix_b.view(),
        &matrix_c.view(),
        output,
        &mut workspace,
    )
}

/// Solve Sylvester equation `A X + X B = C` into `output`.
///
/// # Errors
/// Returns an error if dimensions are invalid or system is singular.
#[cfg(not(feature = "lapack-provider"))]
pub fn solve_sylvester_into<T: NabledReal>(
    matrix_a: &Array2<T>,
    matrix_b: &Array2<T>,
    matrix_c: &Array2<T>,
    output: &mut Array2<T>,
) -> Result<(), SylvesterError> {
    let mut workspace = SylvesterWorkspace::default();
    solve_sylvester_with_workspace_into_impl(
        &matrix_a.view(),
        &matrix_b.view(),
        &matrix_c.view(),
        output,
        &mut workspace,
    )
}

/// Solve Sylvester equation `A X + X B = C` into `output` with reusable `workspace`.
///
/// # Errors
/// Returns an error if dimensions are invalid, output shape mismatches, or system is singular.
#[cfg(feature = "lapack-provider")]
pub fn solve_sylvester_with_workspace_into<T>(
    matrix_a: &Array2<T>,
    matrix_b: &Array2<T>,
    matrix_c: &Array2<T>,
    output: &mut Array2<T>,
    workspace: &mut SylvesterWorkspace<T>,
) -> Result<(), SylvesterError>
where
    T: NabledReal + ndarray_linalg::Lapack,
{
    solve_sylvester_with_workspace_into_impl(
        &matrix_a.view(),
        &matrix_b.view(),
        &matrix_c.view(),
        output,
        workspace,
    )
}

/// Solve Sylvester equation `A X + X B = C` into `output` with reusable `workspace`.
///
/// # Errors
/// Returns an error if dimensions are invalid, output shape mismatches, or system is singular.
#[cfg(not(feature = "lapack-provider"))]
pub fn solve_sylvester_with_workspace_into<T: NabledReal>(
    matrix_a: &Array2<T>,
    matrix_b: &Array2<T>,
    matrix_c: &Array2<T>,
    output: &mut Array2<T>,
    workspace: &mut SylvesterWorkspace<T>,
) -> Result<(), SylvesterError> {
    solve_sylvester_with_workspace_into_impl(
        &matrix_a.view(),
        &matrix_b.view(),
        &matrix_c.view(),
        output,
        workspace,
    )
}

#[cfg(feature = "lapack-provider")]
fn solve_sylvester_with_workspace_into_impl<T>(
    matrix_a: &ArrayView2<'_, T>,
    matrix_b: &ArrayView2<'_, T>,
    matrix_c: &ArrayView2<'_, T>,
    output: &mut Array2<T>,
    workspace: &mut SylvesterWorkspace<T>,
) -> Result<(), SylvesterError>
where
    T: NabledReal + ndarray_linalg::Lapack,
{
    let (n, m) = validate_sylvester_dims(matrix_a, matrix_b, matrix_c)?;
    if output.dim() != (n, m) {
        return Err(SylvesterError::DimensionMismatch);
    }

    workspace.ensure_dims(n, m);

    for i in 0..n {
        for j in 0..m {
            let row = i * m + j;
            workspace.rhs[row] = matrix_c[[i, j]];

            for p in 0..n {
                let col = p * m + j;
                workspace.coefficient[[row, col]] += matrix_a[[i, p]];
            }
            for q in 0..m {
                let col = i * m + q;
                workspace.coefficient[[row, col]] += matrix_b[[q, j]];
            }
        }
    }

    workspace.solution = lu::solve(&workspace.coefficient, &workspace.rhs)
        .map_err(|_| SylvesterError::SingularSystem)?;

    for i in 0..n {
        for j in 0..m {
            output[[i, j]] = workspace.solution[i * m + j];
        }
    }
    Ok(())
}

#[cfg(not(feature = "lapack-provider"))]
fn solve_sylvester_with_workspace_into_impl<T: NabledReal>(
    matrix_a: &ArrayView2<'_, T>,
    matrix_b: &ArrayView2<'_, T>,
    matrix_c: &ArrayView2<'_, T>,
    output: &mut Array2<T>,
    workspace: &mut SylvesterWorkspace<T>,
) -> Result<(), SylvesterError> {
    let (n, m) = validate_sylvester_dims(matrix_a, matrix_b, matrix_c)?;
    if output.dim() != (n, m) {
        return Err(SylvesterError::DimensionMismatch);
    }

    workspace.ensure_dims(n, m);

    for i in 0..n {
        for j in 0..m {
            let row = i * m + j;
            workspace.rhs[row] = matrix_c[[i, j]];

            for p in 0..n {
                let col = p * m + j;
                workspace.coefficient[[row, col]] += matrix_a[[i, p]];
            }
            for q in 0..m {
                let col = i * m + q;
                workspace.coefficient[[row, col]] += matrix_b[[q, j]];
            }
        }
    }

    workspace.solution = lu::solve(&workspace.coefficient, &workspace.rhs)
        .map_err(|_| SylvesterError::SingularSystem)?;

    for i in 0..n {
        for j in 0..m {
            output[[i, j]] = workspace.solution[i * m + j];
        }
    }
    Ok(())
}

fn solve_sylvester_complex_with_workspace_impl(
    matrix_a: &ArrayView2<'_, Complex64>,
    matrix_b: &ArrayView2<'_, Complex64>,
    matrix_c: &ArrayView2<'_, Complex64>,
    output: &mut Array2<Complex64>,
    workspace: &mut SylvesterComplexWorkspace,
) -> Result<(), SylvesterError> {
    let (n, m) = validate_sylvester_complex_dims(matrix_a, matrix_b, matrix_c)?;
    if output.dim() != (n, m) {
        return Err(SylvesterError::DimensionMismatch);
    }

    workspace.ensure_dims(n, m);

    for i in 0..n {
        for j in 0..m {
            let row = i * m + j;
            workspace.rhs[row] = matrix_c[[i, j]];

            for p in 0..n {
                let col = p * m + j;
                workspace.coefficient[[row, col]] += matrix_a[[i, p]];
            }
            for q in 0..m {
                let col = i * m + q;
                workspace.coefficient[[row, col]] += matrix_b[[q, j]];
            }
        }
    }

    workspace.solution = lu::solve_complex(&workspace.coefficient, &workspace.rhs)
        .map_err(|_| SylvesterError::SingularSystem)?;
    for i in 0..n {
        for j in 0..m {
            output[[i, j]] = workspace.solution[i * m + j];
        }
    }
    Ok(())
}

/// Solve complex Sylvester equation `A X + X B = C`.
///
/// # Errors
/// Returns an error if dimensions are invalid or the linear system is singular.
pub fn solve_sylvester_complex(
    matrix_a: &Array2<Complex64>,
    matrix_b: &Array2<Complex64>,
    matrix_c: &Array2<Complex64>,
) -> Result<Array2<Complex64>, SylvesterError> {
    let (n, m) =
        validate_sylvester_complex_dims(&matrix_a.view(), &matrix_b.view(), &matrix_c.view())?;
    let mut workspace = SylvesterComplexWorkspace::default();
    let mut output = Array2::<Complex64>::zeros((n, m));
    solve_sylvester_complex_with_workspace_impl(
        &matrix_a.view(),
        &matrix_b.view(),
        &matrix_c.view(),
        &mut output,
        &mut workspace,
    )?;
    Ok(output)
}

/// Solve complex Sylvester equation `A X + X B = C` from matrix views.
///
/// # Errors
/// Returns an error if dimensions are invalid or the linear system is singular.
pub fn solve_sylvester_complex_view(
    matrix_a: &ArrayView2<'_, Complex64>,
    matrix_b: &ArrayView2<'_, Complex64>,
    matrix_c: &ArrayView2<'_, Complex64>,
) -> Result<Array2<Complex64>, SylvesterError> {
    let (n, m) = validate_sylvester_complex_dims(matrix_a, matrix_b, matrix_c)?;
    let mut workspace = SylvesterComplexWorkspace::default();
    let mut output = Array2::<Complex64>::zeros((n, m));
    solve_sylvester_complex_with_workspace_impl(
        matrix_a,
        matrix_b,
        matrix_c,
        &mut output,
        &mut workspace,
    )?;
    Ok(output)
}

/// Solve complex Sylvester equation `A X + X B = C` into `output`.
///
/// # Errors
/// Returns an error if dimensions are invalid, output shape mismatches, the
/// linear system is singular.
pub fn solve_sylvester_complex_into(
    matrix_a: &Array2<Complex64>,
    matrix_b: &Array2<Complex64>,
    matrix_c: &Array2<Complex64>,
    output: &mut Array2<Complex64>,
) -> Result<(), SylvesterError> {
    let mut workspace = SylvesterComplexWorkspace::default();
    solve_sylvester_complex_with_workspace_impl(
        &matrix_a.view(),
        &matrix_b.view(),
        &matrix_c.view(),
        output,
        &mut workspace,
    )
}

/// Solve complex Sylvester equation `A X + X B = C` into `output` with reusable `workspace`.
///
/// # Errors
/// Returns an error if dimensions are invalid, output shape mismatches, the
/// linear system is singular.
pub fn solve_sylvester_complex_with_workspace_into(
    matrix_a: &Array2<Complex64>,
    matrix_b: &Array2<Complex64>,
    matrix_c: &Array2<Complex64>,
    output: &mut Array2<Complex64>,
    workspace: &mut SylvesterComplexWorkspace,
) -> Result<(), SylvesterError> {
    solve_sylvester_complex_with_workspace_impl(
        &matrix_a.view(),
        &matrix_b.view(),
        &matrix_c.view(),
        output,
        workspace,
    )
}

/// Solve continuous Lyapunov equation `A X + X A^T + Q = 0`.
///
/// # Errors
/// Returns an error if dimensions are invalid or system is singular.
#[cfg(feature = "lapack-provider")]
pub fn solve_lyapunov<T>(a: &Array2<T>, q: &Array2<T>) -> Result<Array2<T>, SylvesterError>
where
    T: NabledReal + ndarray_linalg::Lapack,
{
    if q.nrows() != q.ncols() || q.nrows() != a.nrows() {
        return Err(SylvesterError::DimensionMismatch);
    }
    let neg_q = q.mapv(|value| -value);
    solve_sylvester_view(&a.view(), &a.t(), &neg_q.view())
}

/// Solve continuous Lyapunov equation `A X + X A^T + Q = 0`.
///
/// # Errors
/// Returns an error if dimensions are invalid or system is singular.
#[cfg(not(feature = "lapack-provider"))]
pub fn solve_lyapunov<T: NabledReal>(
    a: &Array2<T>,
    q: &Array2<T>,
) -> Result<Array2<T>, SylvesterError> {
    if q.nrows() != q.ncols() || q.nrows() != a.nrows() {
        return Err(SylvesterError::DimensionMismatch);
    }
    let neg_q = q.mapv(|value| -value);
    solve_sylvester_view(&a.view(), &a.t(), &neg_q.view())
}

/// Solve complex continuous Lyapunov equation `A X + X A^H + Q = 0`.
///
/// # Errors
/// Returns an error if dimensions are invalid or the linear system is singular.
pub fn solve_lyapunov_complex(
    a: &Array2<Complex64>,
    q: &Array2<Complex64>,
) -> Result<Array2<Complex64>, SylvesterError> {
    if q.nrows() != q.ncols() || q.nrows() != a.nrows() {
        return Err(SylvesterError::DimensionMismatch);
    }
    let neg_q = q.mapv(|value| -value);
    let conjugate_transpose = a.t().mapv(|value| value.conj());
    solve_sylvester_complex(a, &conjugate_transpose, &neg_q)
}

/// Solve continuous Lyapunov equation `A X + X A^T + Q = 0` from matrix views.
///
/// # Errors
/// Returns an error if dimensions are invalid or system is singular.
#[cfg(feature = "lapack-provider")]
pub fn solve_lyapunov_view<T>(
    a: &ArrayView2<'_, T>,
    q: &ArrayView2<'_, T>,
) -> Result<Array2<T>, SylvesterError>
where
    T: NabledReal + ndarray_linalg::Lapack,
{
    if q.nrows() != q.ncols() || q.nrows() != a.nrows() {
        return Err(SylvesterError::DimensionMismatch);
    }
    let neg_q = q.mapv(|value| -value);
    solve_sylvester_view(a, &a.t(), &neg_q.view())
}

/// Solve continuous Lyapunov equation `A X + X A^T + Q = 0` from matrix views.
///
/// # Errors
/// Returns an error if dimensions are invalid or system is singular.
#[cfg(not(feature = "lapack-provider"))]
pub fn solve_lyapunov_view<T: NabledReal>(
    a: &ArrayView2<'_, T>,
    q: &ArrayView2<'_, T>,
) -> Result<Array2<T>, SylvesterError> {
    if q.nrows() != q.ncols() || q.nrows() != a.nrows() {
        return Err(SylvesterError::DimensionMismatch);
    }
    let neg_q = q.mapv(|value| -value);
    solve_sylvester_view(a, &a.t(), &neg_q.view())
}

/// Solve complex continuous Lyapunov equation `A X + X A^H + Q = 0` from matrix views.
///
/// # Errors
/// Returns an error if dimensions are invalid or the linear system is singular.
pub fn solve_lyapunov_complex_view(
    a: &ArrayView2<'_, Complex64>,
    q: &ArrayView2<'_, Complex64>,
) -> Result<Array2<Complex64>, SylvesterError> {
    if q.nrows() != q.ncols() || q.nrows() != a.nrows() {
        return Err(SylvesterError::DimensionMismatch);
    }
    let neg_q = q.mapv(|value| -value);
    let conjugate_transpose = a.t().mapv(|value| value.conj());
    solve_sylvester_complex_view(a, &conjugate_transpose.view(), &neg_q.view())
}

/// Solve continuous Lyapunov equation into `output`.
///
/// # Errors
/// Returns an error if dimensions are invalid, output shape mismatches, or system is singular.
#[cfg(feature = "lapack-provider")]
pub fn solve_lyapunov_into<T>(
    a: &Array2<T>,
    q: &Array2<T>,
    output: &mut Array2<T>,
) -> Result<(), SylvesterError>
where
    T: NabledReal + ndarray_linalg::Lapack,
{
    if q.nrows() != q.ncols() || q.nrows() != a.nrows() {
        return Err(SylvesterError::DimensionMismatch);
    }
    let neg_q = q.mapv(|value| -value);
    let mut workspace = SylvesterWorkspace::default();
    solve_sylvester_with_workspace_into_impl(
        &a.view(),
        &a.t(),
        &neg_q.view(),
        output,
        &mut workspace,
    )
}

/// Solve continuous Lyapunov equation into `output`.
///
/// # Errors
/// Returns an error if dimensions are invalid, output shape mismatches, or system is singular.
#[cfg(not(feature = "lapack-provider"))]
pub fn solve_lyapunov_into<T: NabledReal>(
    a: &Array2<T>,
    q: &Array2<T>,
    output: &mut Array2<T>,
) -> Result<(), SylvesterError> {
    if q.nrows() != q.ncols() || q.nrows() != a.nrows() {
        return Err(SylvesterError::DimensionMismatch);
    }
    let neg_q = q.mapv(|value| -value);
    let mut workspace = SylvesterWorkspace::default();
    solve_sylvester_with_workspace_into_impl(
        &a.view(),
        &a.t(),
        &neg_q.view(),
        output,
        &mut workspace,
    )
}

/// Solve complex continuous Lyapunov equation into `output`.
///
/// # Errors
/// Returns an error if dimensions are invalid, output shape mismatches, the
/// linear system is singular.
pub fn solve_lyapunov_complex_into(
    a: &Array2<Complex64>,
    q: &Array2<Complex64>,
    output: &mut Array2<Complex64>,
) -> Result<(), SylvesterError> {
    if q.nrows() != q.ncols() || q.nrows() != a.nrows() {
        return Err(SylvesterError::DimensionMismatch);
    }
    let neg_q = q.mapv(|value| -value);
    let conjugate_transpose = a.t().mapv(|value| value.conj());
    solve_sylvester_complex_into(a, &conjugate_transpose, &neg_q, output)
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;
    use num_complex::Complex64;

    use super::*;

    #[test]
    fn solves_diagonal_sylvester() {
        let a = Array2::from_shape_vec((2, 2), vec![1.0_f64, 0.0_f64, 0.0_f64, 2.0_f64]).unwrap();
        let b = Array2::from_shape_vec((2, 2), vec![3.0_f64, 0.0_f64, 0.0_f64, 4.0_f64]).unwrap();
        let c = Array2::from_shape_vec((2, 2), vec![1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64]).unwrap();

        let x = solve_sylvester(&a, &b, &c).unwrap();
        let residual = a.dot(&x) + x.dot(&b) - c;
        assert!(residual.iter().map(|value| value.abs()).fold(0.0_f64, f64::max) < 1e-8_f64);
    }

    #[test]
    fn solves_diagonal_sylvester_into_with_workspace() {
        let a = Array2::from_shape_vec((2, 2), vec![2.0_f64, 0.0_f64, 0.0_f64, 3.0_f64]).unwrap();
        let b = Array2::from_shape_vec((2, 2), vec![4.0_f64, 0.0_f64, 0.0_f64, 5.0_f64]).unwrap();
        let c = Array2::from_shape_vec((2, 2), vec![2.0_f64, 1.0_f64, 6.0_f64, 4.0_f64]).unwrap();

        let mut output = Array2::<f64>::zeros((2, 2));
        let mut workspace = SylvesterWorkspace::default();
        solve_sylvester_with_workspace_into(&a, &b, &c, &mut output, &mut workspace).unwrap();

        let residual = a.dot(&output) + output.dot(&b) - c;
        assert!(residual.iter().map(|value| value.abs()).fold(0.0_f64, f64::max) < 1e-8_f64);
    }

    #[test]
    fn solves_lyapunov_equation() {
        let a = Array2::from_shape_vec((2, 2), vec![-2.0_f64, 0.0_f64, 0.0_f64, -3.0_f64]).unwrap();
        let q = Array2::eye(2);
        let x = solve_lyapunov(&a, &q).unwrap();
        let residual = a.dot(&x) + x.dot(&a.t().to_owned()) + q;
        assert!(residual.iter().map(|value| value.abs()).fold(0.0_f64, f64::max) < 1e-8_f64);
    }

    #[test]
    fn lyapunov_into_rejects_bad_output_shape() {
        let a = Array2::eye(2);
        let q = Array2::eye(2);
        let mut output = Array2::<f64>::zeros((1, 1));
        let result = solve_lyapunov_into(&a, &q, &mut output);
        assert!(matches!(result, Err(SylvesterError::DimensionMismatch)));
    }

    #[test]
    fn view_variants_match_owned() {
        let a = Array2::from_shape_vec((2, 2), vec![2.0_f64, 0.0_f64, 0.0_f64, 3.0_f64]).unwrap();
        let b = Array2::from_shape_vec((2, 2), vec![1.0_f64, 0.0_f64, 0.0_f64, 4.0_f64]).unwrap();
        let c = Array2::from_shape_vec((2, 2), vec![1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64]).unwrap();

        let owned = solve_sylvester(&a, &b, &c).unwrap();
        let viewed = solve_sylvester_view(&a.view(), &b.view(), &c.view()).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!((owned[[i, j]] - viewed[[i, j]]).abs() < 1e-12_f64);
            }
        }

        let q = Array2::eye(2);
        let lyapunov_owned = solve_lyapunov(&a, &q).unwrap();
        let lyapunov_viewed = solve_lyapunov_view(&a.view(), &q.view()).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!((lyapunov_owned[[i, j]] - lyapunov_viewed[[i, j]]).abs() < 1e-12_f64);
            }
        }
    }

    #[test]
    fn complex_sylvester_and_lyapunov_paths_work() {
        let a = Array2::from_shape_vec((2, 2), vec![
            Complex64::new(2.0_f64, 0.5_f64),
            Complex64::new(0.0_f64, 0.0_f64),
            Complex64::new(0.0_f64, 0.0_f64),
            Complex64::new(3.0_f64, -0.25_f64),
        ])
        .unwrap();
        let b = Array2::from_shape_vec((2, 2), vec![
            Complex64::new(1.0_f64, 0.75_f64),
            Complex64::new(0.0_f64, 0.0_f64),
            Complex64::new(0.0_f64, 0.0_f64),
            Complex64::new(4.0_f64, -0.5_f64),
        ])
        .unwrap();
        let c = Array2::from_shape_vec((2, 2), vec![
            Complex64::new(1.0_f64, 0.0_f64),
            Complex64::new(0.5_f64, -0.2_f64),
            Complex64::new(-1.0_f64, 0.4_f64),
            Complex64::new(2.0_f64, 0.1_f64),
        ])
        .unwrap();

        let owned = solve_sylvester_complex(&a, &b, &c).unwrap();
        let viewed = solve_sylvester_complex_view(&a.view(), &b.view(), &c.view()).unwrap();
        let residual = a.dot(&owned) + owned.dot(&b) - c.clone();
        let max_residual = residual.iter().map(|value| value.norm()).fold(0.0_f64, f64::max);
        assert!(max_residual < 1e-8_f64);
        for i in 0..2 {
            for j in 0..2 {
                assert!((owned[[i, j]] - viewed[[i, j]]).norm() < 1e-10_f64);
            }
        }

        let mut into = Array2::<Complex64>::zeros((2, 2));
        let mut workspace = SylvesterComplexWorkspace::default();
        solve_sylvester_complex_with_workspace_into(&a, &b, &c, &mut into, &mut workspace).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!((owned[[i, j]] - into[[i, j]]).norm() < 1e-10_f64);
            }
        }

        let q = Array2::eye(2).mapv(|value| Complex64::new(value, 0.0_f64));
        let lyapunov = solve_lyapunov_complex(&a, &q).unwrap();
        let a_h = a.t().mapv(|value| value.conj());
        let lyapunov_residual = a.dot(&lyapunov) + lyapunov.dot(&a_h) + q.clone();
        let lyapunov_max =
            lyapunov_residual.iter().map(|value| value.norm()).fold(0.0_f64, f64::max);
        assert!(lyapunov_max < 1e-8_f64);

        let lyapunov_viewed = solve_lyapunov_complex_view(&a.view(), &q.view()).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!((lyapunov[[i, j]] - lyapunov_viewed[[i, j]]).norm() < 1e-10_f64);
            }
        }

        let mut lyapunov_into = Array2::<Complex64>::zeros((2, 2));
        solve_lyapunov_complex_into(&a, &q, &mut lyapunov_into).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!((lyapunov[[i, j]] - lyapunov_into[[i, j]]).norm() < 1e-10_f64);
            }
        }
    }
}
