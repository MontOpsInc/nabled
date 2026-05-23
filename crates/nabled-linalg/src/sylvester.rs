//! Sylvester and Lyapunov solvers over ndarray matrices.

use std::fmt;

use nabled_core::scalar::NabledReal;
use ndarray::{Array1, Array2, ArrayBase, ArrayView2, DataMut, Ix2};
use num_complex::Complex64;

#[cfg(all(feature = "magma-system", not(feature = "lapack-provider")))]
use crate::internal::{lu_decompose, lu_solve};
use crate::lu;
#[cfg(feature = "magma-system")]
use crate::provider::policy::MagmaProviderPolicy;

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

/// Result metadata for mixed-precision Sylvester/Lyapunov solves.
#[derive(Debug, Clone, PartialEq)]
pub struct MixedSylvesterResult<T = f64> {
    /// Solution matrix.
    pub solution: Array2<T>,
    /// Iterative-refinement steps performed by the provider.
    pub refinement_iterations: usize,
}

/// Reusable workspace for Sylvester/Lyapunov solves.
#[derive(Debug, Clone)]
pub struct SylvesterWorkspace<T: NabledReal = f64> {
    coefficient: Array2<T>,
    rhs: Array1<T>,
    solution: Array1<T>,
}

fn map_lu_error_to_sylvester(error: lu::LUError) -> SylvesterError {
    match error {
        lu::LUError::EmptyMatrix => SylvesterError::EmptyMatrix,
        lu::LUError::NotSquare => SylvesterError::NotSquare,
        lu::LUError::InvalidInput(message) => SylvesterError::InvalidInput(message),
        lu::LUError::SingularMatrix
        | lu::LUError::ConvergenceFailed
        | lu::LUError::NumericalInstability => SylvesterError::SingularSystem,
    }
}

#[cfg(all(feature = "magma-system", feature = "lapack-provider"))]
fn solve_linear_real_system<T>(
    matrix_rows: usize,
    matrix_cols: usize,
    coefficient: &Array2<T>,
    rhs: &Array1<T>,
) -> Result<Array1<T>, SylvesterError>
where
    T: NabledReal + lu::LuProviderScalar,
{
    use ndarray_linalg::Solve as _;

    if !MagmaProviderPolicy::verify_force_mode()
        && !MagmaProviderPolicy::prefer_decomposition(matrix_rows, matrix_cols)
    {
        return coefficient.solve(rhs).map_err(|_| SylvesterError::SingularSystem);
    }

    lu::solve(coefficient, rhs).map_err(|_| SylvesterError::SingularSystem)
}

#[cfg(all(
    any(feature = "lapack-provider", feature = "magma-system"),
    not(all(feature = "magma-system", feature = "lapack-provider"))
))]
fn solve_linear_real_system<T>(
    matrix_rows: usize,
    matrix_cols: usize,
    coefficient: &Array2<T>,
    rhs: &Array1<T>,
) -> Result<Array1<T>, SylvesterError>
where
    T: NabledReal + lu::LuProviderScalar,
{
    #[cfg(not(all(feature = "magma-system", not(feature = "lapack-provider"))))]
    let _ = (matrix_rows, matrix_cols);

    #[cfg(all(feature = "magma-system", not(feature = "lapack-provider")))]
    {
        // For Sylvester, provider routing should consider the original matrix
        // size `(n, m)` rather than expanded Kronecker size `(n*m, n*m)`.
        if !MagmaProviderPolicy::verify_force_mode()
            && !MagmaProviderPolicy::prefer_decomposition(matrix_rows, matrix_cols)
        {
            let (lower, upper, pivots, _) =
                lu_decompose(&coefficient.view()).map_err(|_| SylvesterError::SingularSystem)?;
            return lu_solve(&lower, &upper, &pivots, &rhs.view())
                .map_err(|_| SylvesterError::SingularSystem);
        }
    }

    lu::solve(coefficient, rhs).map_err(|_| SylvesterError::SingularSystem)
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
            rhs: Array1::<T>::zeros(0),
            solution: Array1::<T>::zeros(0),
        }
    }
}

/// Reusable workspace for complex Sylvester/Lyapunov solves.
#[derive(Debug, Clone, Default)]
pub struct SylvesterComplexWorkspace {
    coefficient: Array2<Complex64>,
    rhs: Array1<Complex64>,
    solution: Array1<Complex64>,
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
#[cfg(any(feature = "lapack-provider", feature = "magma-system"))]
pub fn solve_sylvester<T>(
    matrix_a: &Array2<T>,
    matrix_b: &Array2<T>,
    matrix_c: &Array2<T>,
) -> Result<Array2<T>, SylvesterError>
where
    T: NabledReal + lu::LuProviderScalar,
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
#[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
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
#[cfg(any(feature = "lapack-provider", feature = "magma-system"))]
pub fn solve_sylvester_view<T>(
    matrix_a: &ArrayView2<'_, T>,
    matrix_b: &ArrayView2<'_, T>,
    matrix_c: &ArrayView2<'_, T>,
) -> Result<Array2<T>, SylvesterError>
where
    T: NabledReal + lu::LuProviderScalar,
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
#[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
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
#[cfg(any(feature = "lapack-provider", feature = "magma-system"))]
pub fn solve_sylvester_into<T>(
    matrix_a: &Array2<T>,
    matrix_b: &Array2<T>,
    matrix_c: &Array2<T>,
    output: &mut Array2<T>,
) -> Result<(), SylvesterError>
where
    T: NabledReal + lu::LuProviderScalar,
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

/// Solve Sylvester equation `A X + X B = C` from views into `output`.
///
/// # Errors
/// Returns an error if dimensions are invalid or system is singular.
#[cfg(any(feature = "lapack-provider", feature = "magma-system"))]
pub fn solve_sylvester_view_into<T, S>(
    matrix_a: &ArrayView2<'_, T>,
    matrix_b: &ArrayView2<'_, T>,
    matrix_c: &ArrayView2<'_, T>,
    output: &mut ArrayBase<S, Ix2>,
) -> Result<(), SylvesterError>
where
    T: NabledReal + lu::LuProviderScalar,
    S: DataMut<Elem = T>,
{
    let mut workspace = SylvesterWorkspace::default();
    solve_sylvester_with_workspace_into_impl(matrix_a, matrix_b, matrix_c, output, &mut workspace)
}

/// Solve Sylvester equation `A X + X B = C` into `output`.
///
/// # Errors
/// Returns an error if dimensions are invalid or system is singular.
#[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
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

/// Solve Sylvester equation `A X + X B = C` from views into `output`.
///
/// # Errors
/// Returns an error if dimensions are invalid or system is singular.
#[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
pub fn solve_sylvester_view_into<T: NabledReal, S>(
    matrix_a: &ArrayView2<'_, T>,
    matrix_b: &ArrayView2<'_, T>,
    matrix_c: &ArrayView2<'_, T>,
    output: &mut ArrayBase<S, Ix2>,
) -> Result<(), SylvesterError>
where
    S: DataMut<Elem = T>,
{
    let mut workspace = SylvesterWorkspace::default();
    solve_sylvester_with_workspace_into_impl(matrix_a, matrix_b, matrix_c, output, &mut workspace)
}

/// Solve Sylvester equation `A X + X B = C` into `output` with reusable `workspace`.
///
/// # Errors
/// Returns an error if dimensions are invalid, output shape mismatches, or system is singular.
#[cfg(any(feature = "lapack-provider", feature = "magma-system"))]
pub fn solve_sylvester_with_workspace_into<T>(
    matrix_a: &Array2<T>,
    matrix_b: &Array2<T>,
    matrix_c: &Array2<T>,
    output: &mut Array2<T>,
    workspace: &mut SylvesterWorkspace<T>,
) -> Result<(), SylvesterError>
where
    T: NabledReal + lu::LuProviderScalar,
{
    solve_sylvester_with_workspace_into_impl(
        &matrix_a.view(),
        &matrix_b.view(),
        &matrix_c.view(),
        output,
        workspace,
    )
}

/// Solve Sylvester equation `A X + X B = C` from views into `output` with reusable `workspace`.
///
/// # Errors
/// Returns an error if dimensions are invalid, output shape mismatches, or system is singular.
#[cfg(any(feature = "lapack-provider", feature = "magma-system"))]
pub fn solve_sylvester_view_with_workspace_into<T, S>(
    matrix_a: &ArrayView2<'_, T>,
    matrix_b: &ArrayView2<'_, T>,
    matrix_c: &ArrayView2<'_, T>,
    output: &mut ArrayBase<S, Ix2>,
    workspace: &mut SylvesterWorkspace<T>,
) -> Result<(), SylvesterError>
where
    T: NabledReal + lu::LuProviderScalar,
    S: DataMut<Elem = T>,
{
    solve_sylvester_with_workspace_into_impl(matrix_a, matrix_b, matrix_c, output, workspace)
}

/// Solve Sylvester equation `A X + X B = C` into `output` with reusable `workspace`.
///
/// # Errors
/// Returns an error if dimensions are invalid, output shape mismatches, or system is singular.
#[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
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

/// Solve Sylvester equation `A X + X B = C` from views into `output` with reusable `workspace`.
///
/// # Errors
/// Returns an error if dimensions are invalid, output shape mismatches, or system is singular.
#[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
pub fn solve_sylvester_view_with_workspace_into<T: NabledReal, S>(
    matrix_a: &ArrayView2<'_, T>,
    matrix_b: &ArrayView2<'_, T>,
    matrix_c: &ArrayView2<'_, T>,
    output: &mut ArrayBase<S, Ix2>,
    workspace: &mut SylvesterWorkspace<T>,
) -> Result<(), SylvesterError>
where
    S: DataMut<Elem = T>,
{
    solve_sylvester_with_workspace_into_impl(matrix_a, matrix_b, matrix_c, output, workspace)
}

#[cfg(any(feature = "lapack-provider", feature = "magma-system"))]
fn solve_sylvester_with_workspace_into_impl<T, S>(
    matrix_a: &ArrayView2<'_, T>,
    matrix_b: &ArrayView2<'_, T>,
    matrix_c: &ArrayView2<'_, T>,
    output: &mut ArrayBase<S, Ix2>,
    workspace: &mut SylvesterWorkspace<T>,
) -> Result<(), SylvesterError>
where
    T: NabledReal + lu::LuProviderScalar,
    S: DataMut<Elem = T>,
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

    workspace.solution = solve_linear_real_system(n, m, &workspace.coefficient, &workspace.rhs)?;

    for i in 0..n {
        for j in 0..m {
            output[[i, j]] = workspace.solution[i * m + j];
        }
    }
    Ok(())
}

#[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
fn solve_sylvester_with_workspace_into_impl<T: NabledReal, S>(
    matrix_a: &ArrayView2<'_, T>,
    matrix_b: &ArrayView2<'_, T>,
    matrix_c: &ArrayView2<'_, T>,
    output: &mut ArrayBase<S, Ix2>,
    workspace: &mut SylvesterWorkspace<T>,
) -> Result<(), SylvesterError>
where
    S: DataMut<Elem = T>,
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

fn solve_sylvester_complex_with_workspace_impl<S>(
    matrix_a: &ArrayView2<'_, Complex64>,
    matrix_b: &ArrayView2<'_, Complex64>,
    matrix_c: &ArrayView2<'_, Complex64>,
    output: &mut ArrayBase<S, Ix2>,
    workspace: &mut SylvesterComplexWorkspace,
) -> Result<(), SylvesterError>
where
    S: DataMut<Elem = Complex64>,
{
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

fn solve_sylvester_mixed_f64_view_impl(
    matrix_a: &ArrayView2<'_, f64>,
    matrix_b: &ArrayView2<'_, f64>,
    matrix_c: &ArrayView2<'_, f64>,
) -> Result<MixedSylvesterResult<f64>, SylvesterError> {
    let (n, m) = validate_sylvester_dims(matrix_a, matrix_b, matrix_c)?;
    let system_size = n * m;
    let mut coefficient = Array2::<f64>::zeros((system_size, system_size));
    let mut rhs = Array1::<f64>::zeros(system_size);

    for i in 0..n {
        for j in 0..m {
            let row = i * m + j;
            rhs[row] = matrix_c[[i, j]];

            for p in 0..n {
                let col = p * m + j;
                coefficient[[row, col]] += matrix_a[[i, p]];
            }
            for q in 0..m {
                let col = i * m + q;
                coefficient[[row, col]] += matrix_b[[q, j]];
            }
        }
    }

    let mixed = lu::solve_mixed_f64_view(&coefficient.view(), &rhs.view())
        .map_err(map_lu_error_to_sylvester)?;
    let mut solution = Array2::<f64>::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            solution[[i, j]] = mixed.solution[i * m + j];
        }
    }

    Ok(MixedSylvesterResult { solution, refinement_iterations: mixed.refinement_iterations })
}

fn solve_sylvester_mixed_complex_view_impl(
    matrix_a: &ArrayView2<'_, Complex64>,
    matrix_b: &ArrayView2<'_, Complex64>,
    matrix_c: &ArrayView2<'_, Complex64>,
) -> Result<MixedSylvesterResult<Complex64>, SylvesterError> {
    let (n, m) = validate_sylvester_complex_dims(matrix_a, matrix_b, matrix_c)?;
    let system_size = n * m;
    let mut coefficient = Array2::<Complex64>::zeros((system_size, system_size));
    let mut rhs = Array1::<Complex64>::zeros(system_size);

    for i in 0..n {
        for j in 0..m {
            let row = i * m + j;
            rhs[row] = matrix_c[[i, j]];

            for p in 0..n {
                let col = p * m + j;
                coefficient[[row, col]] += matrix_a[[i, p]];
            }
            for q in 0..m {
                let col = i * m + q;
                coefficient[[row, col]] += matrix_b[[q, j]];
            }
        }
    }

    let mixed = lu::solve_mixed_complex_view(&coefficient.view(), &rhs.view())
        .map_err(map_lu_error_to_sylvester)?;
    let mut solution = Array2::<Complex64>::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            solution[[i, j]] = mixed.solution[i * m + j];
        }
    }

    Ok(MixedSylvesterResult { solution, refinement_iterations: mixed.refinement_iterations })
}

/// Solve Sylvester equation `A X + X B = C` using mixed-precision iterative refinement.
///
/// This API is available in all builds, but requires feature `magma-system` at runtime.
///
/// # Errors
/// Returns an error if dimensions are invalid, the linear system is singular, convergence fails,
/// or `magma-system` is not enabled.
pub fn solve_sylvester_mixed_f64(
    matrix_a: &Array2<f64>,
    matrix_b: &Array2<f64>,
    matrix_c: &Array2<f64>,
) -> Result<MixedSylvesterResult<f64>, SylvesterError> {
    solve_sylvester_mixed_f64_view_impl(&matrix_a.view(), &matrix_b.view(), &matrix_c.view())
}

/// Solve Sylvester equation `A X + X B = C` from views using mixed-precision iterative
/// refinement.
///
/// This API is available in all builds, but requires feature `magma-system` at runtime.
///
/// # Errors
/// Returns an error if dimensions are invalid, the linear system is singular, convergence fails,
/// or `magma-system` is not enabled.
pub fn solve_sylvester_mixed_f64_view(
    matrix_a: &ArrayView2<'_, f64>,
    matrix_b: &ArrayView2<'_, f64>,
    matrix_c: &ArrayView2<'_, f64>,
) -> Result<MixedSylvesterResult<f64>, SylvesterError> {
    solve_sylvester_mixed_f64_view_impl(matrix_a, matrix_b, matrix_c)
}

/// Solve complex Sylvester equation `A X + X B = C` using mixed-precision iterative refinement.
///
/// This API is available in all builds, but requires feature `magma-system` at runtime.
///
/// # Errors
/// Returns an error if dimensions are invalid, the linear system is singular, convergence fails,
/// or `magma-system` is not enabled.
pub fn solve_sylvester_mixed_complex(
    matrix_a: &Array2<Complex64>,
    matrix_b: &Array2<Complex64>,
    matrix_c: &Array2<Complex64>,
) -> Result<MixedSylvesterResult<Complex64>, SylvesterError> {
    solve_sylvester_mixed_complex_view_impl(&matrix_a.view(), &matrix_b.view(), &matrix_c.view())
}

/// Solve complex Sylvester equation `A X + X B = C` from views using mixed-precision iterative
/// refinement.
///
/// This API is available in all builds, but requires feature `magma-system` at runtime.
///
/// # Errors
/// Returns an error if dimensions are invalid, the linear system is singular, convergence fails,
/// or `magma-system` is not enabled.
pub fn solve_sylvester_mixed_complex_view(
    matrix_a: &ArrayView2<'_, Complex64>,
    matrix_b: &ArrayView2<'_, Complex64>,
    matrix_c: &ArrayView2<'_, Complex64>,
) -> Result<MixedSylvesterResult<Complex64>, SylvesterError> {
    solve_sylvester_mixed_complex_view_impl(matrix_a, matrix_b, matrix_c)
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

/// Solve complex Sylvester equation `A X + X B = C` from views into `output`.
///
/// # Errors
/// Returns an error if dimensions are invalid or system is singular.
pub fn solve_sylvester_complex_view_into<S>(
    matrix_a: &ArrayView2<'_, Complex64>,
    matrix_b: &ArrayView2<'_, Complex64>,
    matrix_c: &ArrayView2<'_, Complex64>,
    output: &mut ArrayBase<S, Ix2>,
) -> Result<(), SylvesterError>
where
    S: DataMut<Elem = Complex64>,
{
    let mut workspace = SylvesterComplexWorkspace::default();
    solve_sylvester_complex_with_workspace_impl(
        matrix_a,
        matrix_b,
        matrix_c,
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

/// Solve complex Sylvester equation from views into outputs using reusable `workspace`.
///
/// # Errors
/// Returns an error if dimensions are invalid, output shape mismatches, or system is singular.
pub fn solve_sylvester_complex_view_with_workspace_into<S>(
    matrix_a: &ArrayView2<'_, Complex64>,
    matrix_b: &ArrayView2<'_, Complex64>,
    matrix_c: &ArrayView2<'_, Complex64>,
    output: &mut ArrayBase<S, Ix2>,
    workspace: &mut SylvesterComplexWorkspace,
) -> Result<(), SylvesterError>
where
    S: DataMut<Elem = Complex64>,
{
    solve_sylvester_complex_with_workspace_impl(matrix_a, matrix_b, matrix_c, output, workspace)
}

/// Solve continuous Lyapunov equation `A X + X A^T + Q = 0`.
///
/// # Errors
/// Returns an error if dimensions are invalid or system is singular.
#[cfg(any(feature = "lapack-provider", feature = "magma-system"))]
pub fn solve_lyapunov<T>(a: &Array2<T>, q: &Array2<T>) -> Result<Array2<T>, SylvesterError>
where
    T: NabledReal + lu::LuProviderScalar,
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
#[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
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

/// Solve continuous Lyapunov equation `A X + X A^T + Q = 0` using mixed-precision iterative
/// refinement.
///
/// This API is available in all builds, but requires feature `magma-system` at runtime.
///
/// # Errors
/// Returns an error if dimensions are invalid, the linear system is singular, convergence fails,
/// or `magma-system` is not enabled.
pub fn solve_lyapunov_mixed_f64(
    a: &Array2<f64>,
    q: &Array2<f64>,
) -> Result<MixedSylvesterResult<f64>, SylvesterError> {
    solve_lyapunov_mixed_f64_view(&a.view(), &q.view())
}

/// Solve continuous Lyapunov equation `A X + X A^T + Q = 0` from views using mixed-precision
/// iterative refinement.
///
/// This API is available in all builds, but requires feature `magma-system` at runtime.
///
/// # Errors
/// Returns an error if dimensions are invalid, the linear system is singular, convergence fails,
/// or `magma-system` is not enabled.
pub fn solve_lyapunov_mixed_f64_view(
    a: &ArrayView2<'_, f64>,
    q: &ArrayView2<'_, f64>,
) -> Result<MixedSylvesterResult<f64>, SylvesterError> {
    if q.nrows() != q.ncols() || q.nrows() != a.nrows() {
        return Err(SylvesterError::DimensionMismatch);
    }
    let neg_q = q.mapv(|value| -value);
    solve_sylvester_mixed_f64_view(a, &a.t(), &neg_q.view())
}

/// Solve complex continuous Lyapunov equation `A X + X A^H + Q = 0` using mixed-precision
/// iterative refinement.
///
/// This API is available in all builds, but requires feature `magma-system` at runtime.
///
/// # Errors
/// Returns an error if dimensions are invalid, the linear system is singular, convergence fails,
/// or `magma-system` is not enabled.
pub fn solve_lyapunov_mixed_complex(
    a: &Array2<Complex64>,
    q: &Array2<Complex64>,
) -> Result<MixedSylvesterResult<Complex64>, SylvesterError> {
    solve_lyapunov_mixed_complex_view(&a.view(), &q.view())
}

/// Solve complex continuous Lyapunov equation `A X + X A^H + Q = 0` from views using
/// mixed-precision iterative refinement.
///
/// This API is available in all builds, but requires feature `magma-system` at runtime.
///
/// # Errors
/// Returns an error if dimensions are invalid, the linear system is singular, convergence fails,
/// or `magma-system` is not enabled.
pub fn solve_lyapunov_mixed_complex_view(
    a: &ArrayView2<'_, Complex64>,
    q: &ArrayView2<'_, Complex64>,
) -> Result<MixedSylvesterResult<Complex64>, SylvesterError> {
    if q.nrows() != q.ncols() || q.nrows() != a.nrows() {
        return Err(SylvesterError::DimensionMismatch);
    }
    let neg_q = q.mapv(|value| -value);
    let conjugate_transpose = a.t().mapv(|value| value.conj());
    solve_sylvester_mixed_complex_view(a, &conjugate_transpose.view(), &neg_q.view())
}

/// Solve continuous Lyapunov equation `A X + X A^T + Q = 0` from matrix views.
///
/// # Errors
/// Returns an error if dimensions are invalid or system is singular.
#[cfg(any(feature = "lapack-provider", feature = "magma-system"))]
pub fn solve_lyapunov_view<T>(
    a: &ArrayView2<'_, T>,
    q: &ArrayView2<'_, T>,
) -> Result<Array2<T>, SylvesterError>
where
    T: NabledReal + lu::LuProviderScalar,
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
#[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
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
#[cfg(any(feature = "lapack-provider", feature = "magma-system"))]
pub fn solve_lyapunov_into<T>(
    a: &Array2<T>,
    q: &Array2<T>,
    output: &mut Array2<T>,
) -> Result<(), SylvesterError>
where
    T: NabledReal + lu::LuProviderScalar,
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

/// Solve continuous Lyapunov equation into `output` with reusable `workspace`.
///
/// # Errors
/// Returns an error if dimensions are invalid, output shape mismatches, or system is singular.
#[cfg(any(feature = "lapack-provider", feature = "magma-system"))]
pub fn solve_lyapunov_with_workspace_into<T>(
    a: &Array2<T>,
    q: &Array2<T>,
    output: &mut Array2<T>,
    workspace: &mut SylvesterWorkspace<T>,
) -> Result<(), SylvesterError>
where
    T: NabledReal + lu::LuProviderScalar,
{
    solve_lyapunov_view_with_workspace_into(&a.view(), &q.view(), output, workspace)
}

/// Solve continuous Lyapunov equation from views into `output`.
///
/// # Errors
/// Returns an error if dimensions are invalid, output shape mismatches, or system is singular.
#[cfg(any(feature = "lapack-provider", feature = "magma-system"))]
pub fn solve_lyapunov_view_into<T, S>(
    a: &ArrayView2<'_, T>,
    q: &ArrayView2<'_, T>,
    output: &mut ArrayBase<S, Ix2>,
) -> Result<(), SylvesterError>
where
    T: NabledReal + lu::LuProviderScalar,
    S: DataMut<Elem = T>,
{
    if q.nrows() != q.ncols() || q.nrows() != a.nrows() {
        return Err(SylvesterError::DimensionMismatch);
    }
    let neg_q = q.mapv(|value| -value);
    let mut workspace = SylvesterWorkspace::default();
    solve_sylvester_with_workspace_into_impl(a, &a.t(), &neg_q.view(), output, &mut workspace)
}

/// Solve continuous Lyapunov equation from views into `output` with reusable `workspace`.
///
/// # Errors
/// Returns an error if dimensions are invalid, output shape mismatches, or system is singular.
#[cfg(any(feature = "lapack-provider", feature = "magma-system"))]
pub fn solve_lyapunov_view_with_workspace_into<T, S>(
    a: &ArrayView2<'_, T>,
    q: &ArrayView2<'_, T>,
    output: &mut ArrayBase<S, Ix2>,
    workspace: &mut SylvesterWorkspace<T>,
) -> Result<(), SylvesterError>
where
    T: NabledReal + lu::LuProviderScalar,
    S: DataMut<Elem = T>,
{
    if q.nrows() != q.ncols() || q.nrows() != a.nrows() {
        return Err(SylvesterError::DimensionMismatch);
    }
    let neg_q = q.mapv(|value| -value);
    solve_sylvester_with_workspace_into_impl(a, &a.t(), &neg_q.view(), output, workspace)
}

/// Solve continuous Lyapunov equation into `output`.
///
/// # Errors
/// Returns an error if dimensions are invalid, output shape mismatches, or system is singular.
#[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
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

/// Solve continuous Lyapunov equation into `output` with reusable `workspace`.
///
/// # Errors
/// Returns an error if dimensions are invalid, output shape mismatches, or system is singular.
#[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
pub fn solve_lyapunov_with_workspace_into<T: NabledReal>(
    a: &Array2<T>,
    q: &Array2<T>,
    output: &mut Array2<T>,
    workspace: &mut SylvesterWorkspace<T>,
) -> Result<(), SylvesterError> {
    solve_lyapunov_view_with_workspace_into(&a.view(), &q.view(), output, workspace)
}

/// Solve continuous Lyapunov equation from views into `output`.
///
/// # Errors
/// Returns an error if dimensions are invalid, output shape mismatches, or system is singular.
#[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
pub fn solve_lyapunov_view_into<T: NabledReal, S>(
    a: &ArrayView2<'_, T>,
    q: &ArrayView2<'_, T>,
    output: &mut ArrayBase<S, Ix2>,
) -> Result<(), SylvesterError>
where
    S: DataMut<Elem = T>,
{
    if q.nrows() != q.ncols() || q.nrows() != a.nrows() {
        return Err(SylvesterError::DimensionMismatch);
    }
    let neg_q = q.mapv(|value| -value);
    let mut workspace = SylvesterWorkspace::default();
    solve_sylvester_with_workspace_into_impl(a, &a.t(), &neg_q.view(), output, &mut workspace)
}

/// Solve continuous Lyapunov equation from views into `output` with reusable `workspace`.
///
/// # Errors
/// Returns an error if dimensions are invalid, output shape mismatches, or system is singular.
#[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
pub fn solve_lyapunov_view_with_workspace_into<T: NabledReal, S>(
    a: &ArrayView2<'_, T>,
    q: &ArrayView2<'_, T>,
    output: &mut ArrayBase<S, Ix2>,
    workspace: &mut SylvesterWorkspace<T>,
) -> Result<(), SylvesterError>
where
    S: DataMut<Elem = T>,
{
    if q.nrows() != q.ncols() || q.nrows() != a.nrows() {
        return Err(SylvesterError::DimensionMismatch);
    }
    let neg_q = q.mapv(|value| -value);
    solve_sylvester_with_workspace_into_impl(a, &a.t(), &neg_q.view(), output, workspace)
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

/// Solve complex continuous Lyapunov equation from views into `output`.
///
/// # Errors
/// Returns an error if dimensions are invalid, output shape mismatches, the
/// linear system is singular.
pub fn solve_lyapunov_complex_view_into<S>(
    a: &ArrayView2<'_, Complex64>,
    q: &ArrayView2<'_, Complex64>,
    output: &mut ArrayBase<S, Ix2>,
) -> Result<(), SylvesterError>
where
    S: DataMut<Elem = Complex64>,
{
    if q.nrows() != q.ncols() || q.nrows() != a.nrows() {
        return Err(SylvesterError::DimensionMismatch);
    }
    let neg_q = q.mapv(|value| -value);
    let conjugate_transpose = a.t().mapv(|value| value.conj());
    solve_sylvester_complex_view_into(a, &conjugate_transpose.view(), &neg_q.view(), output)
}

/// Solve complex continuous Lyapunov equation into `output` with reusable `workspace`.
///
/// # Errors
/// Returns an error if dimensions are invalid, output shape mismatches, or system is singular.
pub fn solve_lyapunov_complex_with_workspace_into(
    a: &Array2<Complex64>,
    q: &Array2<Complex64>,
    output: &mut Array2<Complex64>,
    workspace: &mut SylvesterComplexWorkspace,
) -> Result<(), SylvesterError> {
    solve_lyapunov_complex_view_with_workspace_into(&a.view(), &q.view(), output, workspace)
}

/// Solve complex continuous Lyapunov equation from views into `output` with reusable `workspace`.
///
/// # Errors
/// Returns an error if dimensions are invalid, output shape mismatches, or system is singular.
pub fn solve_lyapunov_complex_view_with_workspace_into<S>(
    a: &ArrayView2<'_, Complex64>,
    q: &ArrayView2<'_, Complex64>,
    output: &mut ArrayBase<S, Ix2>,
    workspace: &mut SylvesterComplexWorkspace,
) -> Result<(), SylvesterError>
where
    S: DataMut<Elem = Complex64>,
{
    if q.nrows() != q.ncols() || q.nrows() != a.nrows() {
        return Err(SylvesterError::DimensionMismatch);
    }
    let neg_q = q.mapv(|value| -value);
    let conjugate_transpose = a.t().mapv(|value| value.conj());
    solve_sylvester_complex_with_workspace_impl(
        a,
        &conjugate_transpose.view(),
        &neg_q.view(),
        output,
        workspace,
    )
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
    fn real_view_workspace_variants_match_allocating_paths() {
        let a = Array2::from_shape_vec((2, 2), vec![2.0_f64, 0.0_f64, 0.0_f64, 3.0_f64]).unwrap();
        let b = Array2::from_shape_vec((2, 2), vec![1.0_f64, 0.0_f64, 0.0_f64, 4.0_f64]).unwrap();
        let c = Array2::from_shape_vec((2, 2), vec![3.0_f64, 2.0_f64, 1.0_f64, 5.0_f64]).unwrap();
        let expected = solve_sylvester(&a, &b, &c).unwrap();

        let mut workspace = SylvesterWorkspace::default();
        let mut output = Array2::<f64>::zeros((2, 2));
        {
            let mut out = output.view_mut();
            solve_sylvester_view_with_workspace_into(
                &a.view(),
                &b.view(),
                &c.view(),
                &mut out,
                &mut workspace,
            )
            .unwrap();
        }
        for i in 0..2 {
            for j in 0..2 {
                assert!((expected[[i, j]] - output[[i, j]]).abs() < 1e-10_f64);
            }
        }

        let q = Array2::eye(2);
        let lyapunov_expected = solve_lyapunov(&a, &q).unwrap();
        let mut lyapunov_output = Array2::<f64>::zeros((2, 2));
        {
            let mut out = lyapunov_output.view_mut();
            solve_lyapunov_view_with_workspace_into(&a.view(), &q.view(), &mut out, &mut workspace)
                .unwrap();
        }
        for i in 0..2 {
            for j in 0..2 {
                assert!((lyapunov_expected[[i, j]] - lyapunov_output[[i, j]]).abs() < 1e-10_f64);
            }
        }
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
        let a = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(2.0_f64, 0.5_f64),
                Complex64::new(0.0_f64, 0.0_f64),
                Complex64::new(0.0_f64, 0.0_f64),
                Complex64::new(3.0_f64, -0.25_f64),
            ],
        )
        .unwrap();
        let b = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(1.0_f64, 0.75_f64),
                Complex64::new(0.0_f64, 0.0_f64),
                Complex64::new(0.0_f64, 0.0_f64),
                Complex64::new(4.0_f64, -0.5_f64),
            ],
        )
        .unwrap();
        let c = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(1.0_f64, 0.0_f64),
                Complex64::new(0.5_f64, -0.2_f64),
                Complex64::new(-1.0_f64, 0.4_f64),
                Complex64::new(2.0_f64, 0.1_f64),
            ],
        )
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

    #[test]
    fn real_view_into_variants_match_owned_with_output_views() {
        let a = Array2::from_shape_vec((2, 2), vec![2.0_f64, 0.0_f64, 0.0_f64, 3.0_f64]).unwrap();
        let b = Array2::from_shape_vec((2, 2), vec![1.0_f64, 0.0_f64, 0.0_f64, 4.0_f64]).unwrap();
        let c = Array2::from_shape_vec((2, 2), vec![1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64]).unwrap();
        let expected = solve_sylvester(&a, &b, &c).unwrap();

        let mut output = Array2::<f64>::zeros((2, 2));
        {
            let mut out = output.view_mut();
            solve_sylvester_view_into(&a.view(), &b.view(), &c.view(), &mut out).unwrap();
        }
        for i in 0..2 {
            for j in 0..2 {
                assert!((expected[[i, j]] - output[[i, j]]).abs() < 1e-12_f64);
            }
        }

        let q = Array2::eye(2);
        let lyapunov_expected = solve_lyapunov(&a, &q).unwrap();
        let mut lyapunov_output = Array2::<f64>::zeros((2, 2));
        {
            let mut out = lyapunov_output.view_mut();
            solve_lyapunov_view_into(&a.view(), &q.view(), &mut out).unwrap();
        }
        for i in 0..2 {
            for j in 0..2 {
                assert!((lyapunov_expected[[i, j]] - lyapunov_output[[i, j]]).abs() < 1e-12_f64);
            }
        }
    }

    #[test]
    fn complex_view_into_variants_match_owned_with_output_views() {
        let a = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(2.0_f64, 0.5_f64),
                Complex64::new(0.0_f64, 0.0_f64),
                Complex64::new(0.0_f64, 0.0_f64),
                Complex64::new(3.0_f64, -0.25_f64),
            ],
        )
        .unwrap();
        let b = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(1.0_f64, 0.75_f64),
                Complex64::new(0.0_f64, 0.0_f64),
                Complex64::new(0.0_f64, 0.0_f64),
                Complex64::new(4.0_f64, -0.5_f64),
            ],
        )
        .unwrap();
        let c = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(1.0_f64, 0.0_f64),
                Complex64::new(0.5_f64, -0.2_f64),
                Complex64::new(-1.0_f64, 0.4_f64),
                Complex64::new(2.0_f64, 0.1_f64),
            ],
        )
        .unwrap();
        let expected = solve_sylvester_complex(&a, &b, &c).unwrap();

        let mut output = Array2::<Complex64>::zeros((2, 2));
        {
            let mut out = output.view_mut();
            solve_sylvester_complex_view_into(&a.view(), &b.view(), &c.view(), &mut out).unwrap();
        }
        for i in 0..2 {
            for j in 0..2 {
                assert!((expected[[i, j]] - output[[i, j]]).norm() < 1e-10_f64);
            }
        }

        let q = Array2::eye(2).mapv(|value| Complex64::new(value, 0.0_f64));
        let lyapunov_expected = solve_lyapunov_complex(&a, &q).unwrap();
        let mut lyapunov_output = Array2::<Complex64>::zeros((2, 2));
        {
            let mut out = lyapunov_output.view_mut();
            solve_lyapunov_complex_view_into(&a.view(), &q.view(), &mut out).unwrap();
        }
        for i in 0..2 {
            for j in 0..2 {
                assert!((lyapunov_expected[[i, j]] - lyapunov_output[[i, j]]).norm() < 1e-10_f64);
            }
        }
    }

    #[test]
    fn complex_view_workspace_variants_match_allocating_paths() {
        let a = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(2.0_f64, 0.5_f64),
                Complex64::new(0.0_f64, 0.0_f64),
                Complex64::new(0.0_f64, 0.0_f64),
                Complex64::new(3.0_f64, -0.25_f64),
            ],
        )
        .unwrap();
        let b = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(1.0_f64, 0.75_f64),
                Complex64::new(0.0_f64, 0.0_f64),
                Complex64::new(0.0_f64, 0.0_f64),
                Complex64::new(4.0_f64, -0.5_f64),
            ],
        )
        .unwrap();
        let c = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(1.0_f64, 0.0_f64),
                Complex64::new(0.5_f64, -0.2_f64),
                Complex64::new(-1.0_f64, 0.4_f64),
                Complex64::new(2.0_f64, 0.1_f64),
            ],
        )
        .unwrap();
        let expected = solve_sylvester_complex(&a, &b, &c).unwrap();

        let mut workspace = SylvesterComplexWorkspace::default();
        let mut output = Array2::<Complex64>::zeros((2, 2));
        {
            let mut out = output.view_mut();
            solve_sylvester_complex_view_with_workspace_into(
                &a.view(),
                &b.view(),
                &c.view(),
                &mut out,
                &mut workspace,
            )
            .unwrap();
        }
        for i in 0..2 {
            for j in 0..2 {
                assert!((expected[[i, j]] - output[[i, j]]).norm() < 1e-10_f64);
            }
        }

        let q = Array2::eye(2).mapv(|value| Complex64::new(value, 0.0_f64));
        let lyapunov_expected = solve_lyapunov_complex(&a, &q).unwrap();
        let mut lyapunov_output = Array2::<Complex64>::zeros((2, 2));
        {
            let mut out = lyapunov_output.view_mut();
            solve_lyapunov_complex_view_with_workspace_into(
                &a.view(),
                &q.view(),
                &mut out,
                &mut workspace,
            )
            .unwrap();
        }
        for i in 0..2 {
            for j in 0..2 {
                assert!((lyapunov_expected[[i, j]] - lyapunov_output[[i, j]]).norm() < 1e-10_f64);
            }
        }
    }

    #[test]
    fn view_into_variants_reject_bad_output_shape() {
        let a = Array2::from_shape_vec((2, 2), vec![2.0_f64, 0.0_f64, 0.0_f64, 3.0_f64]).unwrap();
        let b = Array2::from_shape_vec((2, 2), vec![1.0_f64, 0.0_f64, 0.0_f64, 4.0_f64]).unwrap();
        let c = Array2::from_shape_vec((2, 2), vec![1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64]).unwrap();
        let q = Array2::eye(2);
        let mut bad = Array2::<f64>::zeros((1, 1));
        {
            let mut out = bad.view_mut();
            assert!(matches!(
                solve_sylvester_view_into(&a.view(), &b.view(), &c.view(), &mut out),
                Err(SylvesterError::DimensionMismatch)
            ));
        }
        {
            let mut out = bad.view_mut();
            assert!(matches!(
                solve_lyapunov_view_into(&a.view(), &q.view(), &mut out),
                Err(SylvesterError::DimensionMismatch)
            ));
        }

        let a_complex = a.mapv(|value| Complex64::new(value, 0.0_f64));
        let b_complex = b.mapv(|value| Complex64::new(value, 0.0_f64));
        let c_complex = c.mapv(|value| Complex64::new(value, 0.0_f64));
        let q_complex = q.mapv(|value| Complex64::new(value, 0.0_f64));
        let mut complex_bad = Array2::<Complex64>::zeros((1, 1));
        {
            let mut out = complex_bad.view_mut();
            assert!(matches!(
                solve_sylvester_complex_view_into(
                    &a_complex.view(),
                    &b_complex.view(),
                    &c_complex.view(),
                    &mut out
                ),
                Err(SylvesterError::DimensionMismatch)
            ));
        }
        {
            let mut out = complex_bad.view_mut();
            assert!(matches!(
                solve_lyapunov_complex_view_into(&a_complex.view(), &q_complex.view(), &mut out),
                Err(SylvesterError::DimensionMismatch)
            ));
        }
    }

    #[cfg(not(feature = "magma-system"))]
    #[test]
    fn mixed_sylvester_requires_magma_feature() {
        let a = Array2::from_shape_vec((2, 2), vec![1.0_f64, 0.0_f64, 0.0_f64, 2.0_f64]).unwrap();
        let b = Array2::from_shape_vec((2, 2), vec![3.0_f64, 0.0_f64, 0.0_f64, 4.0_f64]).unwrap();
        let c = Array2::from_shape_vec((2, 2), vec![1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64]).unwrap();
        let error = solve_sylvester_mixed_f64(&a, &b, &c).unwrap_err();
        assert!(
            matches!(error, SylvesterError::InvalidInput(message) if message.contains("magma-system"))
        );
    }

    #[cfg(feature = "magma-system")]
    #[test]
    fn mixed_real_sylvester_and_lyapunov_paths_work() {
        let a = Array2::from_shape_vec((2, 2), vec![2.0_f64, 0.0_f64, 0.0_f64, 3.0_f64]).unwrap();
        let b = Array2::from_shape_vec((2, 2), vec![1.0_f64, 0.0_f64, 0.0_f64, 4.0_f64]).unwrap();
        let c = Array2::from_shape_vec((2, 2), vec![1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64]).unwrap();

        let mixed = solve_sylvester_mixed_f64(&a, &b, &c).unwrap();
        let residual = a.dot(&mixed.solution) + mixed.solution.dot(&b) - c.clone();
        let max_residual = residual.iter().map(|value| value.abs()).fold(0.0_f64, f64::max);
        assert!(max_residual < 1e-7_f64);
        let _iters = mixed.refinement_iterations;

        let q = Array2::eye(2);
        let lyapunov_mixed = solve_lyapunov_mixed_f64(&a, &q).unwrap();
        let lyapunov_residual =
            a.dot(&lyapunov_mixed.solution) + lyapunov_mixed.solution.dot(&a.t()) + q;
        let max_lyapunov =
            lyapunov_residual.iter().map(|value| value.abs()).fold(0.0_f64, f64::max);
        assert!(max_lyapunov < 1e-7_f64);
    }

    #[cfg(feature = "magma-system")]
    #[test]
    fn mixed_complex_sylvester_and_lyapunov_paths_work() {
        let a = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(2.0_f64, 0.5_f64),
                Complex64::new(0.0_f64, 0.0_f64),
                Complex64::new(0.0_f64, 0.0_f64),
                Complex64::new(3.0_f64, -0.25_f64),
            ],
        )
        .unwrap();
        let b = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(1.0_f64, 0.75_f64),
                Complex64::new(0.0_f64, 0.0_f64),
                Complex64::new(0.0_f64, 0.0_f64),
                Complex64::new(4.0_f64, -0.5_f64),
            ],
        )
        .unwrap();
        let c = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(1.0_f64, 0.0_f64),
                Complex64::new(0.5_f64, -0.2_f64),
                Complex64::new(-1.0_f64, 0.4_f64),
                Complex64::new(2.0_f64, 0.1_f64),
            ],
        )
        .unwrap();

        let mixed = solve_sylvester_mixed_complex(&a, &b, &c).unwrap();
        let residual = a.dot(&mixed.solution) + mixed.solution.dot(&b) - c.clone();
        let max_residual = residual.iter().map(|value| value.norm()).fold(0.0_f64, f64::max);
        assert!(max_residual < 1e-7_f64);
        let _iters = mixed.refinement_iterations;

        let q = Array2::eye(2).mapv(|value| Complex64::new(value, 0.0_f64));
        let lyapunov_mixed = solve_lyapunov_mixed_complex(&a, &q).unwrap();
        let a_h = a.t().mapv(|value| value.conj());
        let lyapunov_residual =
            a.dot(&lyapunov_mixed.solution) + lyapunov_mixed.solution.dot(&a_h) + q;
        let max_lyapunov =
            lyapunov_residual.iter().map(|value| value.norm()).fold(0.0_f64, f64::max);
        assert!(max_lyapunov < 1e-7_f64);
    }
}
