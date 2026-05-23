//! Matrix functions over ndarray matrices.

use std::fmt;

use nabled_core::scalar::NabledReal;
use ndarray::linalg::general_mat_mul;
use ndarray::{Array1, Array2, ArrayBase, ArrayView1, ArrayView2, Data, DataMut, Ix2};
use num_complex::Complex64;

use crate::internal::{DenseKernelPolicy, identity, is_symmetric};
#[cfg(feature = "magma-system")]
use crate::provider::magma;
#[cfg(not(feature = "lapack-provider"))]
use crate::schur;
use crate::{eigen, svd};

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

/// Real scalar contract for matrix-function real-valued APIs.
#[cfg(all(feature = "lapack-provider", feature = "magma-system"))]
pub trait MatrixFunctionScalar:
    NabledReal + magma::MagmaReal + ndarray_linalg::Lapack<Real = Self> + std::ops::AddAssign
{
}

#[cfg(all(feature = "lapack-provider", feature = "magma-system"))]
impl<T> MatrixFunctionScalar for T where
    T: NabledReal + magma::MagmaReal + ndarray_linalg::Lapack<Real = T> + std::ops::AddAssign
{
}

/// Real scalar contract for matrix-function real-valued APIs.
#[cfg(all(feature = "lapack-provider", not(feature = "magma-system")))]
pub trait MatrixFunctionScalar:
    NabledReal + ndarray_linalg::Lapack<Real = Self> + std::ops::AddAssign
{
}

#[cfg(all(feature = "lapack-provider", not(feature = "magma-system")))]
impl<T> MatrixFunctionScalar for T where
    T: NabledReal + ndarray_linalg::Lapack<Real = T> + std::ops::AddAssign
{
}

/// Real scalar contract for matrix-function real-valued APIs.
#[cfg(all(not(feature = "lapack-provider"), feature = "magma-system"))]
pub trait MatrixFunctionScalar: NabledReal + magma::MagmaReal {}

#[cfg(all(not(feature = "lapack-provider"), feature = "magma-system"))]
impl<T> MatrixFunctionScalar for T where T: NabledReal + magma::MagmaReal {}

/// Real scalar contract for matrix-function real-valued APIs.
#[cfg(not(feature = "lapack-provider"))]
#[cfg(not(feature = "magma-system"))]
pub trait MatrixFunctionScalar: NabledReal {}

#[cfg(not(feature = "lapack-provider"))]
#[cfg(not(feature = "magma-system"))]
impl<T: NabledReal> MatrixFunctionScalar for T {}

/// Reusable workspace for matrix-function `_into` kernels.
#[derive(Debug, Clone)]
pub struct MatrixFunctionWorkspace<T: NabledReal = f64> {
    scratch: Array2<T>,
}

impl<T: NabledReal> MatrixFunctionWorkspace<T> {
    fn ensure_square(&mut self, n: usize) {
        if self.scratch.dim() != (n, n) {
            self.scratch = Array2::<T>::zeros((n, n));
        }
    }
}

impl<T: NabledReal> Default for MatrixFunctionWorkspace<T> {
    fn default() -> Self { Self { scratch: Array2::<T>::zeros((0, 0)) } }
}

/// Reusable workspace for complex matrix-function `_into` kernels.
#[derive(Debug, Clone, Default)]
pub struct MatrixFunctionComplexWorkspace {
    scratch: Array2<Complex64>,
}

impl MatrixFunctionComplexWorkspace {
    fn ensure_square(&mut self, n: usize) {
        if self.scratch.dim() != (n, n) {
            self.scratch = Array2::<Complex64>::zeros((n, n));
        }
    }
}

fn usize_to_real<T: NabledReal>(value: usize) -> T {
    let fallback = T::from_u32(u32::MAX).unwrap_or(T::one());
    T::from_usize(value).unwrap_or(fallback)
}

fn normalized_taylor_tolerance<T: NabledReal>(requested: T) -> T {
    let requested_f64 = requested.to_f64().unwrap_or(DenseKernelPolicy::taylor_tolerance(0.0));
    let normalized = DenseKernelPolicy::taylor_tolerance(requested_f64);
    T::from_f64(normalized).unwrap_or(requested)
}

fn base_tolerance<T: NabledReal>() -> T {
    T::from_f64(DenseKernelPolicy::BASE_TOLERANCE)
        .unwrap_or_else(|| num_traits::Float::sqrt(T::epsilon()).max(T::epsilon()))
}

fn validate_real_svd_factor_dimensions<T>(
    u: &ArrayView2<'_, T>,
    singular_values: &ArrayView1<'_, T>,
    vt: &ArrayView2<'_, T>,
) -> Result<(), MatrixFunctionError>
where
    T: NabledReal,
{
    if u.is_empty() || singular_values.is_empty() || vt.is_empty() {
        return Err(MatrixFunctionError::EmptyMatrix);
    }
    if u.nrows() != vt.ncols() {
        return Err(MatrixFunctionError::NotSquare);
    }
    if u.ncols() != singular_values.len() || vt.nrows() != singular_values.len() {
        return Err(MatrixFunctionError::InvalidInput(
            "SVD factor shapes are inconsistent".to_string(),
        ));
    }
    if u.iter().any(|value| !value.is_finite())
        || vt.iter().any(|value| !value.is_finite())
        || singular_values.iter().any(|value| !value.is_finite())
    {
        return Err(MatrixFunctionError::InvalidInput("SVD factors must be finite".to_string()));
    }
    Ok(())
}

fn validate_complex_svd_factor_dimensions(
    u: &ArrayView2<'_, Complex64>,
    singular_values: &ArrayView1<'_, f64>,
    vt: &ArrayView2<'_, Complex64>,
) -> Result<(), MatrixFunctionError> {
    if u.is_empty() || singular_values.is_empty() || vt.is_empty() {
        return Err(MatrixFunctionError::EmptyMatrix);
    }
    if u.nrows() != vt.ncols() {
        return Err(MatrixFunctionError::NotSquare);
    }
    if u.ncols() != singular_values.len() || vt.nrows() != singular_values.len() {
        return Err(MatrixFunctionError::InvalidInput(
            "SVD factor shapes are inconsistent".to_string(),
        ));
    }
    if u.iter().any(|value| !value.re.is_finite() || !value.im.is_finite())
        || vt.iter().any(|value| !value.re.is_finite() || !value.im.is_finite())
        || singular_values.iter().any(|value| !value.is_finite())
    {
        return Err(MatrixFunctionError::InvalidInput("SVD factors must be finite".to_string()));
    }
    Ok(())
}

fn validate_real_eigen_factor_dimensions<T>(
    eigenvalues: &ArrayView1<'_, T>,
    eigenvectors: &ArrayView2<'_, T>,
) -> Result<(), MatrixFunctionError>
where
    T: NabledReal,
{
    if eigenvalues.is_empty() || eigenvectors.is_empty() {
        return Err(MatrixFunctionError::EmptyMatrix);
    }
    if eigenvectors.nrows() != eigenvectors.ncols() {
        return Err(MatrixFunctionError::NotSquare);
    }
    if eigenvectors.nrows() != eigenvalues.len() {
        return Err(MatrixFunctionError::InvalidInput(
            "eigen factor shapes are inconsistent".to_string(),
        ));
    }
    if eigenvalues.iter().any(|value| !value.is_finite())
        || eigenvectors.iter().any(|value| !value.is_finite())
    {
        return Err(MatrixFunctionError::InvalidInput("eigen factors must be finite".to_string()));
    }
    Ok(())
}

fn validate_square<T: NabledReal>(matrix: &ArrayView2<'_, T>) -> Result<(), MatrixFunctionError> {
    if matrix.is_empty() {
        return Err(MatrixFunctionError::EmptyMatrix);
    }
    if matrix.nrows() != matrix.ncols() {
        return Err(MatrixFunctionError::NotSquare);
    }
    if matrix.iter().any(|value| !value.is_finite()) {
        return Err(MatrixFunctionError::InvalidInput("matrix must be finite".into()));
    }
    Ok(())
}

fn validate_square_complex(matrix: &ArrayView2<'_, Complex64>) -> Result<(), MatrixFunctionError> {
    if matrix.is_empty() {
        return Err(MatrixFunctionError::EmptyMatrix);
    }
    if matrix.nrows() != matrix.ncols() {
        return Err(MatrixFunctionError::NotSquare);
    }
    if matrix.iter().any(|value| !value.re.is_finite() || !value.im.is_finite()) {
        return Err(MatrixFunctionError::InvalidInput("matrix must be finite".to_string()));
    }
    Ok(())
}

fn diagonal_from<T: NabledReal>(values: &Array1<T>) -> Array2<T> {
    let n = values.len();
    let mut diagonal = Array2::<T>::zeros((n, n));
    for i in 0..n {
        diagonal[[i, i]] = values[i];
    }
    diagonal
}

fn diagonal_from_real_complex(values: &Array1<f64>) -> Array2<Complex64> {
    let n = values.len();
    let mut diagonal = Array2::<Complex64>::zeros((n, n));
    for i in 0..n {
        diagonal[[i, i]] = Complex64::new(values[i], 0.0);
    }
    diagonal
}

fn compose_real_symmetric_factors_into<T, S, F>(
    eigenvalues: &ArrayView1<'_, T>,
    eigenvectors: &ArrayView2<'_, T>,
    output: &mut ArrayBase<S, Ix2>,
    scratch: &mut Array2<T>,
    mut transform: F,
) -> Result<(), MatrixFunctionError>
where
    T: MatrixFunctionScalar,
    S: DataMut<Elem = T>,
    F: FnMut(T) -> Result<T, MatrixFunctionError>,
{
    validate_real_eigen_factor_dimensions(eigenvalues, eigenvectors)?;
    validate_output_shape(eigenvectors, output)?;

    let n = eigenvalues.len();
    if scratch.dim() != (n, n) {
        *scratch = Array2::<T>::zeros((n, n));
    }

    for row in 0..n {
        let scale = transform(eigenvalues[row])?;
        for col in 0..n {
            scratch[[row, col]] = eigenvectors[[col, row]] * scale;
        }
    }

    general_mat_mul(T::one(), eigenvectors, &scratch.view(), T::zero(), output);
    Ok(())
}

fn compose_complex_hermitian_factors_into<S, F>(
    eigenvalues: &ArrayView1<'_, f64>,
    eigenvectors: &ArrayView2<'_, Complex64>,
    output: &mut ArrayBase<S, Ix2>,
    scratch: &mut Array2<Complex64>,
    mut transform: F,
) -> Result<(), MatrixFunctionError>
where
    S: DataMut<Elem = Complex64>,
    F: FnMut(f64) -> Result<f64, MatrixFunctionError>,
{
    if eigenvalues.is_empty() || eigenvectors.is_empty() {
        return Err(MatrixFunctionError::EmptyMatrix);
    }
    validate_output_shape_complex(eigenvectors, output)?;

    let n = eigenvalues.len();
    if eigenvectors.nrows() != eigenvectors.ncols() || eigenvectors.nrows() != n {
        return Err(MatrixFunctionError::InvalidInput(
            "eigen factor shapes are inconsistent".to_string(),
        ));
    }
    if eigenvalues.iter().any(|value| !value.is_finite())
        || eigenvectors.iter().any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(MatrixFunctionError::InvalidInput("eigen factors must be finite".to_string()));
    }

    if scratch.dim() != (n, n) {
        *scratch = Array2::<Complex64>::zeros((n, n));
    }

    for row in 0..n {
        let scale = Complex64::new(transform(eigenvalues[row])?, 0.0);
        for col in 0..n {
            scratch[[row, col]] = eigenvectors[[col, row]].conj() * scale;
        }
    }

    general_mat_mul(
        Complex64::new(1.0, 0.0),
        eigenvectors,
        &scratch.view(),
        Complex64::new(0.0, 0.0),
        output,
    );
    Ok(())
}

fn compose_real_svd_factors_into<T, S, F>(
    u: &ArrayView2<'_, T>,
    singular_values: &ArrayView1<'_, T>,
    vt: &ArrayView2<'_, T>,
    output: &mut ArrayBase<S, Ix2>,
    scratch: &mut Array2<T>,
    mut transform: F,
) -> Result<(), MatrixFunctionError>
where
    T: MatrixFunctionScalar,
    S: DataMut<Elem = T>,
    F: FnMut(T) -> Result<T, MatrixFunctionError>,
{
    validate_real_svd_factor_dimensions(u, singular_values, vt)?;
    if output.dim() != (u.nrows(), vt.ncols()) {
        return Err(MatrixFunctionError::InvalidInput(
            "output shape must match the square matrix implied by the SVD factors".to_string(),
        ));
    }

    if scratch.dim() != vt.dim() {
        *scratch = Array2::<T>::zeros(vt.dim());
    }

    for row in 0..singular_values.len() {
        let scale = transform(singular_values[row])?;
        for col in 0..vt.ncols() {
            scratch[[row, col]] = vt[[row, col]] * scale;
        }
    }

    general_mat_mul(T::one(), u, &scratch.view(), T::zero(), output);
    Ok(())
}

fn compose_complex_svd_factors_into<S, F>(
    u: &ArrayView2<'_, Complex64>,
    singular_values: &ArrayView1<'_, f64>,
    vt: &ArrayView2<'_, Complex64>,
    output: &mut ArrayBase<S, Ix2>,
    scratch: &mut Array2<Complex64>,
    mut transform: F,
) -> Result<(), MatrixFunctionError>
where
    S: DataMut<Elem = Complex64>,
    F: FnMut(f64) -> Result<f64, MatrixFunctionError>,
{
    validate_complex_svd_factor_dimensions(u, singular_values, vt)?;
    if output.dim() != (u.nrows(), vt.ncols()) {
        return Err(MatrixFunctionError::InvalidInput(
            "output shape must match the square matrix implied by the SVD factors".to_string(),
        ));
    }

    if scratch.dim() != vt.dim() {
        *scratch = Array2::<Complex64>::zeros(vt.dim());
    }

    for row in 0..singular_values.len() {
        let scale = Complex64::new(transform(singular_values[row])?, 0.0);
        for col in 0..vt.ncols() {
            scratch[[row, col]] = vt[[row, col]] * scale;
        }
    }

    general_mat_mul(Complex64::new(1.0, 0.0), u, &scratch.view(), Complex64::new(0.0, 0.0), output);
    Ok(())
}

fn taylor_matrix_exp<T: NabledReal>(
    matrix: &ArrayView2<'_, T>,
    max_terms: usize,
    tolerance: T,
) -> Result<Array2<T>, MatrixFunctionError> {
    validate_square(matrix)?;
    let n = matrix.nrows();
    let mut result = identity::<T>(n);
    let mut term = identity::<T>(n);

    for k in 1..=max_terms.max(1) {
        term = term.dot(matrix) / usize_to_real::<T>(k);
        result = &result + &term;
        let delta = term
            .iter()
            .copied()
            .map(num_traits::Float::abs)
            .fold(T::zero(), num_traits::Float::max);
        if delta <= normalized_taylor_tolerance(tolerance) {
            return Ok(result);
        }
    }

    Ok(result)
}

fn taylor_matrix_exp_complex(
    matrix: &ArrayView2<'_, Complex64>,
    max_terms: usize,
    tolerance: f64,
) -> Result<Array2<Complex64>, MatrixFunctionError> {
    validate_square_complex(matrix)?;
    let n = matrix.nrows();
    let mut result = Array2::<Complex64>::zeros((n, n));
    let mut term = Array2::<Complex64>::zeros((n, n));
    for i in 0..n {
        result[[i, i]] = Complex64::new(1.0, 0.0);
        term[[i, i]] = Complex64::new(1.0, 0.0);
    }

    for k in 1..=max_terms.max(1) {
        term = term.dot(matrix) / usize_to_real::<f64>(k);
        result = &result + &term;
        let delta = term.iter().map(|value| value.norm()).fold(0.0_f64, f64::max);
        if delta <= DenseKernelPolicy::taylor_tolerance(tolerance) {
            return Ok(result);
        }
    }

    Ok(result)
}

fn validate_output_shape<T: NabledReal, S1: Data<Elem = T>, S2: Data<Elem = T>>(
    matrix: &ArrayBase<S1, Ix2>,
    output: &ArrayBase<S2, Ix2>,
) -> Result<(), MatrixFunctionError> {
    if output.dim() != matrix.dim() {
        return Err(MatrixFunctionError::InvalidInput(
            "output shape must match input matrix shape".to_string(),
        ));
    }
    Ok(())
}

fn validate_output_shape_complex<S1: Data<Elem = Complex64>, S2: Data<Elem = Complex64>>(
    matrix: &ArrayBase<S1, Ix2>,
    output: &ArrayBase<S2, Ix2>,
) -> Result<(), MatrixFunctionError> {
    if output.dim() != matrix.dim() {
        return Err(MatrixFunctionError::InvalidInput(
            "output shape must match input matrix shape".to_string(),
        ));
    }
    Ok(())
}

fn is_hermitian(matrix: &ArrayView2<'_, Complex64>, tolerance: f64) -> bool {
    if matrix.nrows() != matrix.ncols() {
        return false;
    }
    let n = matrix.nrows();
    for i in 0..n {
        for j in 0..n {
            if (matrix[[i, j]] - matrix[[j, i]].conj()).norm() > tolerance {
                return false;
            }
        }
    }
    true
}

#[cfg(feature = "lapack-provider")]
fn hermitian_eigen_provider(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<(Array1<f64>, Array2<Complex64>), MatrixFunctionError> {
    use ndarray_linalg::{Eigh as _, UPLO};

    matrix.to_owned().eigh(UPLO::Lower).map_err(|_| MatrixFunctionError::ConvergenceFailed)
}

#[cfg(not(feature = "lapack-provider"))]
fn hermitian_eigen_internal(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<(Array1<f64>, Array2<Complex64>), MatrixFunctionError> {
    let decomposition = schur::compute_schur_complex_view(matrix)
        .map_err(|_| MatrixFunctionError::ConvergenceFailed)?;
    let n = matrix.nrows();
    let mut eigenvalues = Array1::<f64>::zeros(n);
    let imag_tolerance = DenseKernelPolicy::polar_convergence_tolerance();

    for i in 0..n {
        let lambda = decomposition.t[[i, i]];
        if lambda.im.abs() > imag_tolerance {
            return Err(MatrixFunctionError::ConvergenceFailed);
        }
        eigenvalues[i] = lambda.re;
    }

    Ok((eigenvalues, decomposition.q))
}

fn hermitian_eigen_dispatch(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<(Array1<f64>, Array2<Complex64>), MatrixFunctionError> {
    #[cfg(feature = "lapack-provider")]
    {
        hermitian_eigen_provider(matrix)
    }
    #[cfg(not(feature = "lapack-provider"))]
    {
        hermitian_eigen_internal(matrix)
    }
}

/// Compute matrix exponential via Taylor series.
///
/// # Errors
/// Returns an error for invalid input.
pub fn matrix_exp<T>(
    matrix: &Array2<T>,
    max_terms: usize,
    tolerance: T,
) -> Result<Array2<T>, MatrixFunctionError>
where
    T: MatrixFunctionScalar,
{
    matrix_exp_impl(&matrix.view(), max_terms, tolerance)
}

/// Compute complex matrix exponential via Taylor series.
///
/// # Errors
/// Returns an error for invalid input.
pub fn matrix_exp_complex(
    matrix: &Array2<Complex64>,
    max_terms: usize,
    tolerance: f64,
) -> Result<Array2<Complex64>, MatrixFunctionError> {
    matrix_exp_complex_impl(&matrix.view(), max_terms, tolerance)
}

fn matrix_exp_impl<T: NabledReal>(
    matrix: &ArrayView2<'_, T>,
    max_terms: usize,
    tolerance: T,
) -> Result<Array2<T>, MatrixFunctionError> {
    taylor_matrix_exp(matrix, max_terms, tolerance)
}

fn matrix_exp_complex_impl(
    matrix: &ArrayView2<'_, Complex64>,
    max_terms: usize,
    tolerance: f64,
) -> Result<Array2<Complex64>, MatrixFunctionError> {
    taylor_matrix_exp_complex(matrix, max_terms, tolerance)
}

/// Compute matrix exponential via Taylor series from a matrix view.
///
/// # Errors
/// Returns an error for invalid input.
pub fn matrix_exp_view<T>(
    matrix: &ArrayView2<'_, T>,
    max_terms: usize,
    tolerance: T,
) -> Result<Array2<T>, MatrixFunctionError>
where
    T: MatrixFunctionScalar,
{
    matrix_exp_impl(matrix, max_terms, tolerance)
}

/// Compute complex matrix exponential via Taylor series from a matrix view.
///
/// # Errors
/// Returns an error for invalid input.
pub fn matrix_exp_complex_view(
    matrix: &ArrayView2<'_, Complex64>,
    max_terms: usize,
    tolerance: f64,
) -> Result<Array2<Complex64>, MatrixFunctionError> {
    matrix_exp_complex_impl(matrix, max_terms, tolerance)
}

/// Compute matrix exponential into `output`.
///
/// # Errors
/// Returns an error for invalid inputs or output shape mismatch.
pub fn matrix_exp_into<T>(
    matrix: &Array2<T>,
    max_terms: usize,
    tolerance: T,
    output: &mut Array2<T>,
) -> Result<(), MatrixFunctionError>
where
    T: MatrixFunctionScalar,
{
    let mut workspace = MatrixFunctionWorkspace::<T>::default();
    matrix_exp_with_workspace_into(matrix, max_terms, tolerance, output, &mut workspace)
}

/// Compute matrix exponential from a view into `output`.
///
/// # Errors
/// Returns an error for invalid inputs or output shape mismatch.
pub fn matrix_exp_view_into<T, S>(
    matrix: &ArrayView2<'_, T>,
    max_terms: usize,
    tolerance: T,
    output: &mut ArrayBase<S, Ix2>,
) -> Result<(), MatrixFunctionError>
where
    T: MatrixFunctionScalar,
    S: DataMut<Elem = T>,
{
    let mut workspace = MatrixFunctionWorkspace::<T>::default();
    matrix_exp_view_with_workspace_into(matrix, max_terms, tolerance, output, &mut workspace)
}

/// Compute complex matrix exponential into `output`.
///
/// # Errors
/// Returns an error for invalid inputs or output shape mismatch.
pub fn matrix_exp_complex_into(
    matrix: &Array2<Complex64>,
    max_terms: usize,
    tolerance: f64,
    output: &mut Array2<Complex64>,
) -> Result<(), MatrixFunctionError> {
    let mut workspace = MatrixFunctionComplexWorkspace::default();
    matrix_exp_complex_with_workspace_into(matrix, max_terms, tolerance, output, &mut workspace)
}

/// Compute complex matrix exponential from a view into `output`.
///
/// # Errors
/// Returns an error for invalid inputs or output shape mismatch.
pub fn matrix_exp_complex_view_into<S>(
    matrix: &ArrayView2<'_, Complex64>,
    max_terms: usize,
    tolerance: f64,
    output: &mut ArrayBase<S, Ix2>,
) -> Result<(), MatrixFunctionError>
where
    S: DataMut<Elem = Complex64>,
{
    let mut workspace = MatrixFunctionComplexWorkspace::default();
    matrix_exp_complex_view_with_workspace_into(
        matrix,
        max_terms,
        tolerance,
        output,
        &mut workspace,
    )
}

/// Compute matrix exponential into `output` using reusable `workspace`.
///
/// # Errors
/// Returns an error for invalid inputs or output shape mismatch.
pub fn matrix_exp_with_workspace_into<T>(
    matrix: &Array2<T>,
    max_terms: usize,
    tolerance: T,
    output: &mut Array2<T>,
    workspace: &mut MatrixFunctionWorkspace<T>,
) -> Result<(), MatrixFunctionError>
where
    T: MatrixFunctionScalar,
{
    validate_output_shape(matrix, output)?;
    workspace.ensure_square(matrix.nrows());
    let result = matrix_exp_impl(&matrix.view(), max_terms, tolerance)?;
    workspace.scratch.assign(&result);
    output.assign(&workspace.scratch);
    Ok(())
}

/// Compute matrix exponential from a view into `output` using reusable `workspace`.
///
/// # Errors
/// Returns an error for invalid inputs or output shape mismatch.
pub fn matrix_exp_view_with_workspace_into<T, S>(
    matrix: &ArrayView2<'_, T>,
    max_terms: usize,
    tolerance: T,
    output: &mut ArrayBase<S, Ix2>,
    workspace: &mut MatrixFunctionWorkspace<T>,
) -> Result<(), MatrixFunctionError>
where
    T: MatrixFunctionScalar,
    S: DataMut<Elem = T>,
{
    if output.dim() != matrix.dim() {
        return Err(MatrixFunctionError::InvalidInput(
            "output shape must match input matrix shape".to_string(),
        ));
    }
    workspace.ensure_square(matrix.nrows());
    let result = matrix_exp_impl(matrix, max_terms, tolerance)?;
    workspace.scratch.assign(&result);
    output.assign(&workspace.scratch);
    Ok(())
}

/// Compute complex matrix exponential into `output` using reusable `workspace`.
///
/// # Errors
/// Returns an error for invalid inputs or output shape mismatch.
pub fn matrix_exp_complex_with_workspace_into(
    matrix: &Array2<Complex64>,
    max_terms: usize,
    tolerance: f64,
    output: &mut Array2<Complex64>,
    workspace: &mut MatrixFunctionComplexWorkspace,
) -> Result<(), MatrixFunctionError> {
    validate_output_shape_complex(matrix, output)?;
    workspace.ensure_square(matrix.nrows());
    let result = matrix_exp_complex_impl(&matrix.view(), max_terms, tolerance)?;
    workspace.scratch.assign(&result);
    output.assign(&workspace.scratch);
    Ok(())
}

/// Compute complex matrix exponential from a view into `output` using reusable `workspace`.
///
/// # Errors
/// Returns an error for invalid inputs or output shape mismatch.
pub fn matrix_exp_complex_view_with_workspace_into<S>(
    matrix: &ArrayView2<'_, Complex64>,
    max_terms: usize,
    tolerance: f64,
    output: &mut ArrayBase<S, Ix2>,
    workspace: &mut MatrixFunctionComplexWorkspace,
) -> Result<(), MatrixFunctionError>
where
    S: DataMut<Elem = Complex64>,
{
    if output.dim() != matrix.dim() {
        return Err(MatrixFunctionError::InvalidInput(
            "output shape must match input matrix shape".to_string(),
        ));
    }
    workspace.ensure_square(matrix.nrows());
    let result = matrix_exp_complex_impl(matrix, max_terms, tolerance)?;
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
pub fn matrix_exp_eigen<T>(matrix: &Array2<T>) -> Result<Array2<T>, MatrixFunctionError>
where
    T: MatrixFunctionScalar,
{
    matrix_exp_eigen_impl(&matrix.view())
}

fn matrix_exp_eigen_impl<T>(matrix: &ArrayView2<'_, T>) -> Result<Array2<T>, MatrixFunctionError>
where
    T: MatrixFunctionScalar,
{
    validate_square(matrix)?;
    let tolerance = base_tolerance::<T>();
    if !is_symmetric(matrix, tolerance) {
        return matrix_exp_impl(matrix, DenseKernelPolicy::MATRIX_FUNCTION_SERIES_TERMS, tolerance);
    }

    let eigen =
        eigen::symmetric_view(matrix).map_err(|_| MatrixFunctionError::ConvergenceFailed)?;
    matrix_exp_eigen_from_factors_view(&eigen.eigenvalues.view(), &eigen.eigenvectors.view())
}

/// Compute matrix exponential directly from real symmetric eigendecomposition factors.
///
/// # Errors
/// Returns an error if the eigen factor dimensions are inconsistent.
pub fn matrix_exp_eigen_from_factors_view<T>(
    eigenvalues: &ArrayView1<'_, T>,
    eigenvectors: &ArrayView2<'_, T>,
) -> Result<Array2<T>, MatrixFunctionError>
where
    T: MatrixFunctionScalar,
{
    validate_real_eigen_factor_dimensions(eigenvalues, eigenvectors)?;
    let exp_values = eigenvalues.mapv(num_traits::Float::exp);
    let diagonal = diagonal_from(&exp_values);
    Ok(eigenvectors.dot(&diagonal).dot(&eigenvectors.t()))
}

/// Compute matrix exponential directly from real symmetric eigendecomposition factors into
/// `output`.
///
/// # Errors
/// Returns an error if the eigen factor dimensions are inconsistent or `output` has the wrong
/// shape.
pub fn matrix_exp_eigen_from_factors_view_into<T, S>(
    eigenvalues: &ArrayView1<'_, T>,
    eigenvectors: &ArrayView2<'_, T>,
    output: &mut ArrayBase<S, Ix2>,
) -> Result<(), MatrixFunctionError>
where
    T: MatrixFunctionScalar,
    S: DataMut<Elem = T>,
{
    let mut scratch = Array2::<T>::zeros((0, 0));
    compose_real_symmetric_factors_into(eigenvalues, eigenvectors, output, &mut scratch, |value| {
        Ok(num_traits::Float::exp(value))
    })
}

/// Compute matrix exponential via eigen decomposition from a matrix view.
///
/// Falls back to Taylor series for non-symmetric matrices.
///
/// # Errors
/// Returns an error for invalid input.
pub fn matrix_exp_eigen_view<T>(
    matrix: &ArrayView2<'_, T>,
) -> Result<Array2<T>, MatrixFunctionError>
where
    T: MatrixFunctionScalar,
{
    matrix_exp_eigen_impl(matrix)
}

/// Compute matrix exponential via eigendecomposition into `output`.
///
/// Falls back to Taylor series for non-symmetric matrices.
///
/// # Errors
/// Returns an error for invalid inputs or output shape mismatch.
pub fn matrix_exp_eigen_into<T>(
    matrix: &Array2<T>,
    output: &mut ArrayBase<impl DataMut<Elem = T>, Ix2>,
) -> Result<(), MatrixFunctionError>
where
    T: MatrixFunctionScalar,
{
    let mut workspace = MatrixFunctionWorkspace::<T>::default();
    matrix_exp_eigen_with_workspace_into(matrix, output, &mut workspace)
}

/// Compute matrix exponential via eigendecomposition from a view into `output`.
///
/// Falls back to Taylor series for non-symmetric matrices.
///
/// # Errors
/// Returns an error for invalid inputs or output shape mismatch.
pub fn matrix_exp_eigen_view_into<T, S>(
    matrix: &ArrayBase<S, Ix2>,
    output: &mut ArrayBase<impl DataMut<Elem = T>, Ix2>,
) -> Result<(), MatrixFunctionError>
where
    T: MatrixFunctionScalar,
    S: Data<Elem = T>,
{
    let mut workspace = MatrixFunctionWorkspace::<T>::default();
    matrix_exp_eigen_view_with_workspace_into(&matrix.view(), output, &mut workspace)
}

/// Compute matrix exponential via eigendecomposition into `output` using reusable `workspace`.
///
/// Falls back to Taylor series for non-symmetric matrices.
///
/// # Errors
/// Returns an error for invalid inputs or output shape mismatch.
pub fn matrix_exp_eigen_with_workspace_into<T>(
    matrix: &Array2<T>,
    output: &mut ArrayBase<impl DataMut<Elem = T>, Ix2>,
    workspace: &mut MatrixFunctionWorkspace<T>,
) -> Result<(), MatrixFunctionError>
where
    T: MatrixFunctionScalar,
{
    matrix_exp_eigen_view_with_workspace_into(&matrix.view(), output, workspace)
}

/// Compute matrix exponential via eigendecomposition from a view into `output` using reusable
/// `workspace`.
///
/// Falls back to Taylor series for non-symmetric matrices.
///
/// # Errors
/// Returns an error for invalid inputs or output shape mismatch.
pub fn matrix_exp_eigen_view_with_workspace_into<T, S>(
    matrix: &ArrayBase<S, Ix2>,
    output: &mut ArrayBase<impl DataMut<Elem = T>, Ix2>,
    workspace: &mut MatrixFunctionWorkspace<T>,
) -> Result<(), MatrixFunctionError>
where
    T: MatrixFunctionScalar,
    S: Data<Elem = T>,
{
    validate_output_shape(&matrix.view(), output)?;
    if !is_symmetric(&matrix.view(), base_tolerance::<T>()) {
        return matrix_exp_view_with_workspace_into(
            &matrix.view(),
            DenseKernelPolicy::MATRIX_FUNCTION_SERIES_TERMS,
            base_tolerance::<T>(),
            output,
            workspace,
        );
    }
    let decomposition = eigen::symmetric_view(&matrix.view())
        .map_err(|_| MatrixFunctionError::ConvergenceFailed)?;
    compose_real_symmetric_factors_into(
        &decomposition.eigenvalues.view(),
        &decomposition.eigenvectors.view(),
        output,
        &mut workspace.scratch,
        |value| Ok(num_traits::Float::exp(value)),
    )
}

/// Compute complex matrix exponential via Hermitian eigen decomposition.
///
/// Falls back to Taylor series for non-Hermitian matrices.
///
/// # Errors
/// Returns an error for invalid input.
pub fn matrix_exp_eigen_complex(
    matrix: &Array2<Complex64>,
) -> Result<Array2<Complex64>, MatrixFunctionError> {
    matrix_exp_eigen_complex_impl(&matrix.view())
}

fn matrix_exp_eigen_complex_impl(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<Array2<Complex64>, MatrixFunctionError> {
    validate_square_complex(matrix)?;
    if !is_hermitian(matrix, DenseKernelPolicy::BASE_TOLERANCE) {
        return matrix_exp_complex_impl(
            matrix,
            DenseKernelPolicy::MATRIX_FUNCTION_SERIES_TERMS,
            DenseKernelPolicy::BASE_TOLERANCE,
        );
    }

    let (eigenvalues, eigenvectors) = hermitian_eigen_dispatch(matrix)?;
    let exp_values = eigenvalues.mapv(num_traits::Float::exp);
    let diagonal = diagonal_from_real_complex(&exp_values);
    let qh = eigenvectors.t().mapv(|value| value.conj());
    Ok(eigenvectors.dot(&diagonal).dot(&qh))
}

/// Compute complex matrix exponential via Hermitian eigen decomposition from a matrix view.
///
/// Falls back to Taylor series for non-Hermitian matrices.
///
/// # Errors
/// Returns an error for invalid input.
pub fn matrix_exp_eigen_complex_view(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<Array2<Complex64>, MatrixFunctionError> {
    matrix_exp_eigen_complex_impl(matrix)
}

/// Compute complex matrix exponential via Hermitian eigendecomposition into `output`.
///
/// Falls back to Taylor series for non-Hermitian matrices.
///
/// # Errors
/// Returns an error for invalid inputs or output shape mismatch.
pub fn matrix_exp_eigen_complex_into(
    matrix: &Array2<Complex64>,
    output: &mut ArrayBase<impl DataMut<Elem = Complex64>, Ix2>,
) -> Result<(), MatrixFunctionError> {
    let mut workspace = MatrixFunctionComplexWorkspace::default();
    matrix_exp_eigen_complex_with_workspace_into(matrix, output, &mut workspace)
}

/// Compute complex matrix exponential via Hermitian eigendecomposition from a view into `output`.
///
/// Falls back to Taylor series for non-Hermitian matrices.
///
/// # Errors
/// Returns an error for invalid inputs or output shape mismatch.
pub fn matrix_exp_eigen_complex_view_into<S>(
    matrix: &ArrayBase<S, Ix2>,
    output: &mut ArrayBase<impl DataMut<Elem = Complex64>, Ix2>,
) -> Result<(), MatrixFunctionError>
where
    S: Data<Elem = Complex64>,
{
    let mut workspace = MatrixFunctionComplexWorkspace::default();
    matrix_exp_eigen_complex_view_with_workspace_into(&matrix.view(), output, &mut workspace)
}

/// Compute complex matrix exponential via Hermitian eigendecomposition into `output` using
/// reusable `workspace`.
///
/// Falls back to Taylor series for non-Hermitian matrices.
///
/// # Errors
/// Returns an error for invalid inputs or output shape mismatch.
pub fn matrix_exp_eigen_complex_with_workspace_into(
    matrix: &Array2<Complex64>,
    output: &mut ArrayBase<impl DataMut<Elem = Complex64>, Ix2>,
    workspace: &mut MatrixFunctionComplexWorkspace,
) -> Result<(), MatrixFunctionError> {
    matrix_exp_eigen_complex_view_with_workspace_into(&matrix.view(), output, workspace)
}

/// Compute complex matrix exponential via Hermitian eigendecomposition from a view into `output`
/// using reusable `workspace`.
///
/// Falls back to Taylor series for non-Hermitian matrices.
///
/// # Errors
/// Returns an error for invalid inputs or output shape mismatch.
pub fn matrix_exp_eigen_complex_view_with_workspace_into<S>(
    matrix: &ArrayBase<S, Ix2>,
    output: &mut ArrayBase<impl DataMut<Elem = Complex64>, Ix2>,
    workspace: &mut MatrixFunctionComplexWorkspace,
) -> Result<(), MatrixFunctionError>
where
    S: Data<Elem = Complex64>,
{
    validate_output_shape_complex(&matrix.view(), output)?;
    if !is_hermitian(&matrix.view(), DenseKernelPolicy::BASE_TOLERANCE) {
        return matrix_exp_complex_view_with_workspace_into(
            &matrix.view(),
            DenseKernelPolicy::MATRIX_FUNCTION_SERIES_TERMS,
            DenseKernelPolicy::BASE_TOLERANCE,
            output,
            workspace,
        );
    }
    let (eigenvalues, eigenvectors) = hermitian_eigen_dispatch(&matrix.view())?;
    compose_complex_hermitian_factors_into(
        &eigenvalues.view(),
        &eigenvectors.view(),
        output,
        &mut workspace.scratch,
        |value| Ok(num_traits::Float::exp(value)),
    )
}

/// Compute matrix logarithm via Taylor expansion around identity.
///
/// # Errors
/// Returns an error for invalid input.
pub fn matrix_log_taylor<T>(
    matrix: &Array2<T>,
    max_terms: usize,
    tolerance: T,
) -> Result<Array2<T>, MatrixFunctionError>
where
    T: MatrixFunctionScalar,
{
    matrix_log_taylor_impl(&matrix.view(), max_terms, tolerance)
}

fn matrix_log_taylor_impl<T>(
    matrix: &ArrayView2<'_, T>,
    max_terms: usize,
    tolerance: T,
) -> Result<Array2<T>, MatrixFunctionError>
where
    T: MatrixFunctionScalar,
{
    validate_square(matrix)?;
    let n = matrix.nrows();
    let identity = identity::<T>(n);
    let x = matrix - &identity;

    let mut result = Array2::<T>::zeros((n, n));
    let mut term = x.clone();

    for k in 1..=max_terms.max(1) {
        let scale = if k % 2 == 0 { -T::one() } else { T::one() } / usize_to_real::<T>(k);
        result = &result + &(term.mapv(|value| scale * value));
        term = term.dot(&x);

        let delta = term
            .iter()
            .copied()
            .map(num_traits::Float::abs)
            .fold(T::zero(), num_traits::Float::max);
        if delta <= normalized_taylor_tolerance(tolerance) {
            break;
        }
    }

    Ok(result)
}

/// Compute matrix logarithm via Taylor expansion around identity from a matrix view.
///
/// # Errors
/// Returns an error for invalid input.
pub fn matrix_log_taylor_view<T>(
    matrix: &ArrayView2<'_, T>,
    max_terms: usize,
    tolerance: T,
) -> Result<Array2<T>, MatrixFunctionError>
where
    T: MatrixFunctionScalar,
{
    matrix_log_taylor_impl(matrix, max_terms, tolerance)
}

/// Compute matrix logarithm via Taylor expansion into `output`.
///
/// # Errors
/// Returns an error for invalid inputs or output shape mismatch.
pub fn matrix_log_taylor_into<T>(
    matrix: &Array2<T>,
    max_terms: usize,
    tolerance: T,
    output: &mut Array2<T>,
) -> Result<(), MatrixFunctionError>
where
    T: MatrixFunctionScalar,
{
    let mut workspace = MatrixFunctionWorkspace::<T>::default();
    matrix_log_taylor_with_workspace_into(matrix, max_terms, tolerance, output, &mut workspace)
}

/// Compute matrix logarithm via Taylor expansion from a view into `output`.
///
/// # Errors
/// Returns an error for invalid inputs or output shape mismatch.
pub fn matrix_log_taylor_view_into<T, S>(
    matrix: &ArrayView2<'_, T>,
    max_terms: usize,
    tolerance: T,
    output: &mut ArrayBase<S, Ix2>,
) -> Result<(), MatrixFunctionError>
where
    T: MatrixFunctionScalar,
    S: DataMut<Elem = T>,
{
    let mut workspace = MatrixFunctionWorkspace::<T>::default();
    matrix_log_taylor_view_with_workspace_into(matrix, max_terms, tolerance, output, &mut workspace)
}

/// Compute matrix logarithm via Taylor expansion into `output` with reusable `workspace`.
///
/// # Errors
/// Returns an error for invalid inputs or output shape mismatch.
pub fn matrix_log_taylor_with_workspace_into<T>(
    matrix: &Array2<T>,
    max_terms: usize,
    tolerance: T,
    output: &mut Array2<T>,
    workspace: &mut MatrixFunctionWorkspace<T>,
) -> Result<(), MatrixFunctionError>
where
    T: MatrixFunctionScalar,
{
    validate_output_shape(matrix, output)?;
    workspace.ensure_square(matrix.nrows());
    let result = matrix_log_taylor_impl(&matrix.view(), max_terms, tolerance)?;
    workspace.scratch.assign(&result);
    output.assign(&workspace.scratch);
    Ok(())
}

/// Compute matrix logarithm via Taylor expansion from a view into `output` with reusable
/// `workspace`.
///
/// # Errors
/// Returns an error for invalid inputs or output shape mismatch.
pub fn matrix_log_taylor_view_with_workspace_into<T, S>(
    matrix: &ArrayView2<'_, T>,
    max_terms: usize,
    tolerance: T,
    output: &mut ArrayBase<S, Ix2>,
    workspace: &mut MatrixFunctionWorkspace<T>,
) -> Result<(), MatrixFunctionError>
where
    T: MatrixFunctionScalar,
    S: DataMut<Elem = T>,
{
    if output.dim() != matrix.dim() {
        return Err(MatrixFunctionError::InvalidInput(
            "output shape must match input matrix shape".to_string(),
        ));
    }
    workspace.ensure_square(matrix.nrows());
    let result = matrix_log_taylor_impl(matrix, max_terms, tolerance)?;
    workspace.scratch.assign(&result);
    output.assign(&workspace.scratch);
    Ok(())
}

/// Compute matrix logarithm via eigen decomposition (symmetric PSD matrices).
///
/// # Errors
/// Returns an error if eigenvalues are non-positive.
pub fn matrix_log_eigen<T>(matrix: &Array2<T>) -> Result<Array2<T>, MatrixFunctionError>
where
    T: MatrixFunctionScalar,
{
    matrix_log_eigen_impl(&matrix.view())
}

fn matrix_log_eigen_impl<T>(matrix: &ArrayView2<'_, T>) -> Result<Array2<T>, MatrixFunctionError>
where
    T: MatrixFunctionScalar,
{
    validate_square(matrix)?;
    let tolerance = base_tolerance::<T>();
    if !is_symmetric(matrix, tolerance) {
        return Err(MatrixFunctionError::NotSymmetric);
    }

    let eigen =
        eigen::symmetric_view(matrix).map_err(|_| MatrixFunctionError::ConvergenceFailed)?;
    matrix_log_eigen_from_factors_view(&eigen.eigenvalues.view(), &eigen.eigenvectors.view())
}

/// Compute matrix logarithm directly from real symmetric eigendecomposition factors.
///
/// # Errors
/// Returns an error if the eigen factor dimensions are inconsistent or the spectrum is not
/// positive definite.
pub fn matrix_log_eigen_from_factors_view<T>(
    eigenvalues: &ArrayView1<'_, T>,
    eigenvectors: &ArrayView2<'_, T>,
) -> Result<Array2<T>, MatrixFunctionError>
where
    T: MatrixFunctionScalar,
{
    validate_real_eigen_factor_dimensions(eigenvalues, eigenvectors)?;
    let tolerance = base_tolerance::<T>();
    if eigenvalues.iter().any(|value| *value <= tolerance) {
        return Err(MatrixFunctionError::NotPositiveDefinite);
    }

    let log_values = eigenvalues.mapv(num_traits::Float::ln);
    let diagonal = diagonal_from(&log_values);
    Ok(eigenvectors.dot(&diagonal).dot(&eigenvectors.t()))
}

/// Compute matrix logarithm directly from real symmetric eigendecomposition factors into
/// `output`.
///
/// # Errors
/// Returns an error if the eigen factor dimensions are inconsistent, the spectrum is not positive
/// definite, or `output` has the wrong shape.
pub fn matrix_log_eigen_from_factors_view_into<T, S>(
    eigenvalues: &ArrayView1<'_, T>,
    eigenvectors: &ArrayView2<'_, T>,
    output: &mut ArrayBase<S, Ix2>,
) -> Result<(), MatrixFunctionError>
where
    T: MatrixFunctionScalar,
    S: DataMut<Elem = T>,
{
    let tolerance = base_tolerance::<T>();
    let mut scratch = Array2::<T>::zeros((0, 0));
    compose_real_symmetric_factors_into(eigenvalues, eigenvectors, output, &mut scratch, |value| {
        if value <= tolerance {
            Err(MatrixFunctionError::NotPositiveDefinite)
        } else {
            Ok(num_traits::Float::ln(value))
        }
    })
}

/// Compute matrix logarithm via eigen decomposition from a matrix view.
///
/// # Errors
/// Returns an error if eigenvalues are non-positive.
pub fn matrix_log_eigen_view<T>(
    matrix: &ArrayView2<'_, T>,
) -> Result<Array2<T>, MatrixFunctionError>
where
    T: MatrixFunctionScalar,
{
    matrix_log_eigen_impl(matrix)
}

/// Compute matrix logarithm via eigen decomposition into `output`.
///
/// # Errors
/// Returns an error for invalid inputs or output shape mismatch.
pub fn matrix_log_eigen_into<T>(
    matrix: &Array2<T>,
    output: &mut Array2<T>,
) -> Result<(), MatrixFunctionError>
where
    T: MatrixFunctionScalar,
{
    let mut workspace = MatrixFunctionWorkspace::<T>::default();
    matrix_log_eigen_with_workspace_into(matrix, output, &mut workspace)
}

/// Compute matrix logarithm via eigen decomposition from a view into `output`.
///
/// # Errors
/// Returns an error for invalid inputs or output shape mismatch.
pub fn matrix_log_eigen_view_into<T, S>(
    matrix: &ArrayView2<'_, T>,
    output: &mut ArrayBase<S, Ix2>,
) -> Result<(), MatrixFunctionError>
where
    T: MatrixFunctionScalar,
    S: DataMut<Elem = T>,
{
    let mut workspace = MatrixFunctionWorkspace::<T>::default();
    matrix_log_eigen_view_with_workspace_into(matrix, output, &mut workspace)
}

/// Compute matrix logarithm via eigen decomposition into `output` with reusable `workspace`.
///
/// # Errors
/// Returns an error for invalid inputs or output shape mismatch.
pub fn matrix_log_eigen_with_workspace_into<T>(
    matrix: &Array2<T>,
    output: &mut Array2<T>,
    workspace: &mut MatrixFunctionWorkspace<T>,
) -> Result<(), MatrixFunctionError>
where
    T: MatrixFunctionScalar,
{
    matrix_log_eigen_view_with_workspace_into(&matrix.view(), output, workspace)
}

/// Compute matrix logarithm via eigen decomposition from a view into `output` with reusable
/// `workspace`.
///
/// # Errors
/// Returns an error for invalid inputs or output shape mismatch.
pub fn matrix_log_eigen_view_with_workspace_into<T, S>(
    matrix: &ArrayView2<'_, T>,
    output: &mut ArrayBase<S, Ix2>,
    workspace: &mut MatrixFunctionWorkspace<T>,
) -> Result<(), MatrixFunctionError>
where
    T: MatrixFunctionScalar,
    S: DataMut<Elem = T>,
{
    if output.dim() != matrix.dim() {
        return Err(MatrixFunctionError::InvalidInput(
            "output shape must match input matrix shape".to_string(),
        ));
    }
    let tolerance = base_tolerance::<T>();
    if !is_symmetric(matrix, tolerance) {
        return Err(MatrixFunctionError::NotSymmetric);
    }
    let decomposition =
        eigen::symmetric_view(matrix).map_err(|_| MatrixFunctionError::ConvergenceFailed)?;
    compose_real_symmetric_factors_into(
        &decomposition.eigenvalues.view(),
        &decomposition.eigenvectors.view(),
        output,
        &mut workspace.scratch,
        |value| {
            if value <= tolerance {
                Err(MatrixFunctionError::NotPositiveDefinite)
            } else {
                Ok(num_traits::Float::ln(value))
            }
        },
    )
}

/// Compute complex matrix logarithm via Hermitian eigen decomposition.
///
/// # Errors
/// Returns an error for non-Hermitian or non-positive-definite inputs.
pub fn matrix_log_eigen_complex(
    matrix: &Array2<Complex64>,
) -> Result<Array2<Complex64>, MatrixFunctionError> {
    matrix_log_eigen_complex_impl(&matrix.view())
}

fn matrix_log_eigen_complex_impl(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<Array2<Complex64>, MatrixFunctionError> {
    validate_square_complex(matrix)?;
    if !is_hermitian(matrix, DenseKernelPolicy::BASE_TOLERANCE) {
        return Err(MatrixFunctionError::NotSymmetric);
    }

    let (eigenvalues, eigenvectors) = hermitian_eigen_dispatch(matrix)?;
    if eigenvalues.iter().any(|value| *value <= DenseKernelPolicy::BASE_TOLERANCE) {
        return Err(MatrixFunctionError::NotPositiveDefinite);
    }
    let log_values = eigenvalues.mapv(num_traits::Float::ln);
    let diagonal = diagonal_from_real_complex(&log_values);
    let qh = eigenvectors.t().mapv(|value| value.conj());
    Ok(eigenvectors.dot(&diagonal).dot(&qh))
}

/// Compute complex matrix logarithm via Hermitian eigen decomposition from a matrix view.
///
/// # Errors
/// Returns an error for non-Hermitian or non-positive-definite inputs.
pub fn matrix_log_eigen_complex_view(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<Array2<Complex64>, MatrixFunctionError> {
    matrix_log_eigen_complex_impl(matrix)
}

/// Compute complex matrix logarithm via Hermitian eigen decomposition into `output`.
///
/// # Errors
/// Returns an error for invalid inputs, output shape mismatch, non-Hermitian
/// or non-positive-definite inputs.
pub fn matrix_log_eigen_complex_into(
    matrix: &Array2<Complex64>,
    output: &mut Array2<Complex64>,
) -> Result<(), MatrixFunctionError> {
    let mut workspace = MatrixFunctionComplexWorkspace::default();
    matrix_log_eigen_complex_with_workspace_into(matrix, output, &mut workspace)
}

/// Compute complex matrix logarithm via Hermitian eigen decomposition from a view into `output`.
///
/// # Errors
/// Returns an error for invalid inputs, output shape mismatch, non-Hermitian
/// or non-positive-definite inputs.
pub fn matrix_log_eigen_complex_view_into<S>(
    matrix: &ArrayView2<'_, Complex64>,
    output: &mut ArrayBase<S, Ix2>,
) -> Result<(), MatrixFunctionError>
where
    S: DataMut<Elem = Complex64>,
{
    let mut workspace = MatrixFunctionComplexWorkspace::default();
    matrix_log_eigen_complex_view_with_workspace_into(matrix, output, &mut workspace)
}

/// Compute complex matrix logarithm via Hermitian eigen decomposition into `output`.
///
/// # Errors
/// Returns an error for invalid inputs, output shape mismatch, non-Hermitian
/// or non-positive-definite inputs.
pub fn matrix_log_eigen_complex_with_workspace_into(
    matrix: &Array2<Complex64>,
    output: &mut Array2<Complex64>,
    workspace: &mut MatrixFunctionComplexWorkspace,
) -> Result<(), MatrixFunctionError> {
    matrix_log_eigen_complex_view_with_workspace_into(&matrix.view(), output, workspace)
}

/// Compute complex matrix logarithm via Hermitian eigen decomposition from a view into `output`
/// using reusable `workspace`.
///
/// # Errors
/// Returns an error for invalid inputs, output shape mismatch, non-Hermitian
/// or non-positive-definite inputs.
pub fn matrix_log_eigen_complex_view_with_workspace_into<S>(
    matrix: &ArrayView2<'_, Complex64>,
    output: &mut ArrayBase<S, Ix2>,
    workspace: &mut MatrixFunctionComplexWorkspace,
) -> Result<(), MatrixFunctionError>
where
    S: DataMut<Elem = Complex64>,
{
    if output.dim() != matrix.dim() {
        return Err(MatrixFunctionError::InvalidInput(
            "output shape must match input matrix shape".to_string(),
        ));
    }
    if !is_hermitian(matrix, DenseKernelPolicy::BASE_TOLERANCE) {
        return Err(MatrixFunctionError::NotSymmetric);
    }
    let (eigenvalues, eigenvectors) = hermitian_eigen_dispatch(matrix)?;
    compose_complex_hermitian_factors_into(
        &eigenvalues.view(),
        &eigenvectors.view(),
        output,
        &mut workspace.scratch,
        |value| {
            if value <= DenseKernelPolicy::BASE_TOLERANCE {
                Err(MatrixFunctionError::NotPositiveDefinite)
            } else {
                Ok(num_traits::Float::ln(value))
            }
        },
    )
}

/// Compute matrix logarithm via SVD.
///
/// # Errors
/// Returns an error if SVD fails.
pub fn matrix_log_svd<T>(matrix: &Array2<T>) -> Result<Array2<T>, MatrixFunctionError>
where
    T: MatrixFunctionScalar,
{
    matrix_log_svd_impl(&matrix.view())
}

fn matrix_log_svd_impl<T>(matrix: &ArrayView2<'_, T>) -> Result<Array2<T>, MatrixFunctionError>
where
    T: MatrixFunctionScalar,
{
    validate_square(matrix)?;
    let svd = svd::decompose_view(matrix).map_err(|_| MatrixFunctionError::ConvergenceFailed)?;
    matrix_log_svd_from_svd_view(&svd.u.view(), &svd.singular_values.view(), &svd.vt.view())
}

/// Compute complex matrix logarithm via SVD.
///
/// # Errors
/// Returns an error if decomposition fails.
pub fn matrix_log_svd_complex(
    matrix: &Array2<Complex64>,
) -> Result<Array2<Complex64>, MatrixFunctionError> {
    matrix_log_svd_complex_impl(&matrix.view())
}

fn matrix_log_svd_complex_impl(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<Array2<Complex64>, MatrixFunctionError> {
    validate_square_complex(matrix)?;
    let svd = svd::decompose_complex_view(matrix).map_err(|error| match error {
        svd::SVDError::EmptyMatrix => MatrixFunctionError::EmptyMatrix,
        svd::SVDError::NotSquare => MatrixFunctionError::NotSquare,
        svd::SVDError::ConvergenceFailed => MatrixFunctionError::ConvergenceFailed,
        svd::SVDError::InvalidInput(message) => MatrixFunctionError::InvalidInput(message),
    })?;
    matrix_log_svd_complex_from_svd_view(&svd.u.view(), &svd.singular_values.view(), &svd.vt.view())
}

/// Compute matrix logarithm via precomputed SVD factors.
///
/// # Errors
/// Returns an error if the factors are inconsistent or not positive definite.
pub fn matrix_log_svd_from_svd<T>(
    u: &Array2<T>,
    singular_values: &Array1<T>,
    vt: &Array2<T>,
) -> Result<Array2<T>, MatrixFunctionError>
where
    T: MatrixFunctionScalar,
{
    matrix_log_svd_from_svd_view(&u.view(), &singular_values.view(), &vt.view())
}

/// Compute matrix logarithm via borrowed SVD factor views.
///
/// # Errors
/// Returns an error if the factors are inconsistent or not positive definite.
pub fn matrix_log_svd_from_svd_view<T>(
    u: &ArrayView2<'_, T>,
    singular_values: &ArrayView1<'_, T>,
    vt: &ArrayView2<'_, T>,
) -> Result<Array2<T>, MatrixFunctionError>
where
    T: MatrixFunctionScalar,
{
    validate_real_svd_factor_dimensions(u, singular_values, vt)?;
    let tolerance = base_tolerance::<T>();
    if singular_values.iter().any(|value| *value <= tolerance) {
        return Err(MatrixFunctionError::NotPositiveDefinite);
    }

    let log_sigma = diagonal_from(&singular_values.mapv(num_traits::Float::ln));
    Ok(u.dot(&log_sigma).dot(vt))
}

/// Compute complex matrix logarithm via precomputed complex SVD factors.
///
/// # Errors
/// Returns an error if the factors are inconsistent or not positive definite.
pub fn matrix_log_svd_complex_from_svd(
    u: &Array2<Complex64>,
    singular_values: &Array1<f64>,
    vt: &Array2<Complex64>,
) -> Result<Array2<Complex64>, MatrixFunctionError> {
    matrix_log_svd_complex_from_svd_view(&u.view(), &singular_values.view(), &vt.view())
}

/// Compute complex matrix logarithm via borrowed complex SVD factor views.
///
/// # Errors
/// Returns an error if the factors are inconsistent or not positive definite.
pub fn matrix_log_svd_complex_from_svd_view(
    u: &ArrayView2<'_, Complex64>,
    singular_values: &ArrayView1<'_, f64>,
    vt: &ArrayView2<'_, Complex64>,
) -> Result<Array2<Complex64>, MatrixFunctionError> {
    validate_complex_svd_factor_dimensions(u, singular_values, vt)?;
    if singular_values.iter().any(|value| *value <= DenseKernelPolicy::BASE_TOLERANCE) {
        return Err(MatrixFunctionError::NotPositiveDefinite);
    }

    let log_sigma = diagonal_from_real_complex(&singular_values.mapv(num_traits::Float::ln));
    Ok(u.dot(&log_sigma).dot(vt))
}

/// Compute matrix logarithm via SVD from a matrix view.
///
/// # Errors
/// Returns an error if SVD fails.
pub fn matrix_log_svd_view<T>(matrix: &ArrayView2<'_, T>) -> Result<Array2<T>, MatrixFunctionError>
where
    T: MatrixFunctionScalar,
{
    matrix_log_svd_impl(matrix)
}

/// Compute complex matrix logarithm via SVD from a matrix view.
///
/// # Errors
/// Returns an error if decomposition fails.
pub fn matrix_log_svd_complex_view(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<Array2<Complex64>, MatrixFunctionError> {
    matrix_log_svd_complex_impl(matrix)
}

/// Compute matrix logarithm via SVD into `output`.
///
/// # Errors
/// Returns an error for invalid inputs or output shape mismatch.
pub fn matrix_log_svd_into<T>(
    matrix: &Array2<T>,
    output: &mut Array2<T>,
) -> Result<(), MatrixFunctionError>
where
    T: MatrixFunctionScalar,
{
    let mut workspace = MatrixFunctionWorkspace::<T>::default();
    matrix_log_svd_with_workspace_into(matrix, output, &mut workspace)
}

/// Compute matrix logarithm via precomputed SVD factors into `output`.
///
/// # Errors
/// Returns an error if the factors are inconsistent or `output` has the wrong shape.
pub fn matrix_log_svd_from_svd_into<T, S>(
    u: &ArrayView2<'_, T>,
    singular_values: &ArrayView1<'_, T>,
    vt: &ArrayView2<'_, T>,
    output: &mut ArrayBase<S, Ix2>,
) -> Result<(), MatrixFunctionError>
where
    T: MatrixFunctionScalar,
    S: DataMut<Elem = T>,
{
    let tolerance = base_tolerance::<T>();
    let mut scratch = Array2::<T>::zeros((0, 0));
    compose_real_svd_factors_into(u, singular_values, vt, output, &mut scratch, |value| {
        if value <= tolerance {
            Err(MatrixFunctionError::NotPositiveDefinite)
        } else {
            Ok(num_traits::Float::ln(value))
        }
    })
}

/// Compute matrix logarithm via SVD from a view into `output`.
///
/// # Errors
/// Returns an error for invalid inputs or output shape mismatch.
pub fn matrix_log_svd_view_into<T, S>(
    matrix: &ArrayView2<'_, T>,
    output: &mut ArrayBase<S, Ix2>,
) -> Result<(), MatrixFunctionError>
where
    T: MatrixFunctionScalar,
    S: DataMut<Elem = T>,
{
    let mut workspace = MatrixFunctionWorkspace::<T>::default();
    matrix_log_svd_view_with_workspace_into(matrix, output, &mut workspace)
}

/// Compute complex matrix logarithm via SVD into `output`.
///
/// # Errors
/// Returns an error for invalid inputs, output shape mismatch, decomposition
/// failure.
pub fn matrix_log_svd_complex_into(
    matrix: &Array2<Complex64>,
    output: &mut Array2<Complex64>,
) -> Result<(), MatrixFunctionError> {
    let mut workspace = MatrixFunctionComplexWorkspace::default();
    matrix_log_svd_complex_with_workspace_into(matrix, output, &mut workspace)
}

/// Compute complex matrix logarithm via precomputed complex SVD factors into `output`.
///
/// # Errors
/// Returns an error if the factors are inconsistent or `output` has the wrong shape.
pub fn matrix_log_svd_complex_from_svd_into<S>(
    u: &ArrayView2<'_, Complex64>,
    singular_values: &ArrayView1<'_, f64>,
    vt: &ArrayView2<'_, Complex64>,
    output: &mut ArrayBase<S, Ix2>,
) -> Result<(), MatrixFunctionError>
where
    S: DataMut<Elem = Complex64>,
{
    let mut scratch = Array2::<Complex64>::zeros((0, 0));
    compose_complex_svd_factors_into(u, singular_values, vt, output, &mut scratch, |value| {
        if value <= DenseKernelPolicy::BASE_TOLERANCE {
            Err(MatrixFunctionError::NotPositiveDefinite)
        } else {
            Ok(value.ln())
        }
    })
}

/// Compute complex matrix logarithm via SVD from a view into `output`.
///
/// # Errors
/// Returns an error for invalid inputs, output shape mismatch, decomposition
/// failure.
pub fn matrix_log_svd_complex_view_into<S>(
    matrix: &ArrayView2<'_, Complex64>,
    output: &mut ArrayBase<S, Ix2>,
) -> Result<(), MatrixFunctionError>
where
    S: DataMut<Elem = Complex64>,
{
    let mut workspace = MatrixFunctionComplexWorkspace::default();
    matrix_log_svd_complex_view_with_workspace_into(matrix, output, &mut workspace)
}

/// Compute matrix logarithm via SVD into `output` with reusable `workspace`.
///
/// # Errors
/// Returns an error for invalid inputs or output shape mismatch.
pub fn matrix_log_svd_with_workspace_into<T>(
    matrix: &Array2<T>,
    output: &mut Array2<T>,
    workspace: &mut MatrixFunctionWorkspace<T>,
) -> Result<(), MatrixFunctionError>
where
    T: MatrixFunctionScalar,
{
    matrix_log_svd_view_with_workspace_into(&matrix.view(), output, workspace)
}

/// Compute matrix logarithm via SVD from a view into `output` with reusable `workspace`.
///
/// # Errors
/// Returns an error for invalid inputs or output shape mismatch.
pub fn matrix_log_svd_view_with_workspace_into<T, S>(
    matrix: &ArrayView2<'_, T>,
    output: &mut ArrayBase<S, Ix2>,
    workspace: &mut MatrixFunctionWorkspace<T>,
) -> Result<(), MatrixFunctionError>
where
    T: MatrixFunctionScalar,
    S: DataMut<Elem = T>,
{
    if output.dim() != matrix.dim() {
        return Err(MatrixFunctionError::InvalidInput(
            "output shape must match input matrix shape".to_string(),
        ));
    }
    let svd = svd::decompose_view(matrix).map_err(|_| MatrixFunctionError::ConvergenceFailed)?;
    let tolerance = base_tolerance::<T>();
    compose_real_svd_factors_into(
        &svd.u.view(),
        &svd.singular_values.view(),
        &svd.vt.view(),
        output,
        &mut workspace.scratch,
        |value| {
            if value <= tolerance {
                Err(MatrixFunctionError::NotPositiveDefinite)
            } else {
                Ok(num_traits::Float::ln(value))
            }
        },
    )
}

/// Compute complex matrix logarithm via SVD into `output` with reusable `workspace`.
///
/// # Errors
/// Returns an error for invalid inputs, output shape mismatch, decomposition
/// failure.
pub fn matrix_log_svd_complex_with_workspace_into(
    matrix: &Array2<Complex64>,
    output: &mut Array2<Complex64>,
    workspace: &mut MatrixFunctionComplexWorkspace,
) -> Result<(), MatrixFunctionError> {
    matrix_log_svd_complex_view_with_workspace_into(&matrix.view(), output, workspace)
}

/// Compute complex matrix logarithm via SVD from a view into `output` with reusable `workspace`.
///
/// # Errors
/// Returns an error for invalid inputs, output shape mismatch, decomposition
/// failure.
pub fn matrix_log_svd_complex_view_with_workspace_into<S>(
    matrix: &ArrayView2<'_, Complex64>,
    output: &mut ArrayBase<S, Ix2>,
    workspace: &mut MatrixFunctionComplexWorkspace,
) -> Result<(), MatrixFunctionError>
where
    S: DataMut<Elem = Complex64>,
{
    if output.dim() != matrix.dim() {
        return Err(MatrixFunctionError::InvalidInput(
            "output shape must match input matrix shape".to_string(),
        ));
    }
    let svd = svd::decompose_complex_view(matrix).map_err(|error| match error {
        svd::SVDError::EmptyMatrix => MatrixFunctionError::EmptyMatrix,
        svd::SVDError::NotSquare => MatrixFunctionError::NotSquare,
        svd::SVDError::ConvergenceFailed => MatrixFunctionError::ConvergenceFailed,
        svd::SVDError::InvalidInput(message) => MatrixFunctionError::InvalidInput(message),
    })?;
    compose_complex_svd_factors_into(
        &svd.u.view(),
        &svd.singular_values.view(),
        &svd.vt.view(),
        output,
        &mut workspace.scratch,
        |value| {
            if value <= DenseKernelPolicy::BASE_TOLERANCE {
                Err(MatrixFunctionError::NotPositiveDefinite)
            } else {
                Ok(value.ln())
            }
        },
    )
}

/// Compute matrix power via eigen decomposition (symmetric matrices).
///
/// # Errors
/// Returns an error for non-symmetric inputs.
pub fn matrix_power<T>(matrix: &Array2<T>, power: T) -> Result<Array2<T>, MatrixFunctionError>
where
    T: MatrixFunctionScalar,
{
    matrix_power_impl(&matrix.view(), power)
}

fn matrix_power_impl<T>(
    matrix: &ArrayView2<'_, T>,
    power: T,
) -> Result<Array2<T>, MatrixFunctionError>
where
    T: MatrixFunctionScalar,
{
    validate_square(matrix)?;
    let tolerance = base_tolerance::<T>();
    if !is_symmetric(matrix, tolerance) {
        return Err(MatrixFunctionError::NotSymmetric);
    }

    let eigen =
        eigen::symmetric_view(matrix).map_err(|_| MatrixFunctionError::ConvergenceFailed)?;
    matrix_power_from_factors_view(&eigen.eigenvalues.view(), &eigen.eigenvectors.view(), power)
}

/// Compute matrix power directly from real symmetric eigendecomposition factors.
///
/// # Errors
/// Returns an error if the eigen factor dimensions are inconsistent.
pub fn matrix_power_from_factors_view<T>(
    eigenvalues: &ArrayView1<'_, T>,
    eigenvectors: &ArrayView2<'_, T>,
    power: T,
) -> Result<Array2<T>, MatrixFunctionError>
where
    T: MatrixFunctionScalar,
{
    validate_real_eigen_factor_dimensions(eigenvalues, eigenvectors)?;
    let powered_values = eigenvalues.mapv(|value| num_traits::Float::powf(value, power));
    let diagonal = diagonal_from(&powered_values);
    Ok(eigenvectors.dot(&diagonal).dot(&eigenvectors.t()))
}

/// Compute matrix power directly from real symmetric eigendecomposition factors into `output`.
///
/// # Errors
/// Returns an error if the eigen factor dimensions are inconsistent or `output` has the wrong
/// shape.
pub fn matrix_power_from_factors_view_into<T, S>(
    eigenvalues: &ArrayView1<'_, T>,
    eigenvectors: &ArrayView2<'_, T>,
    power: T,
    output: &mut ArrayBase<S, Ix2>,
) -> Result<(), MatrixFunctionError>
where
    T: MatrixFunctionScalar,
    S: DataMut<Elem = T>,
{
    let mut scratch = Array2::<T>::zeros((0, 0));
    compose_real_symmetric_factors_into(eigenvalues, eigenvectors, output, &mut scratch, |value| {
        Ok(num_traits::Float::powf(value, power))
    })
}

/// Compute matrix power via eigen decomposition from a matrix view.
///
/// # Errors
/// Returns an error for non-symmetric inputs.
pub fn matrix_power_view<T>(
    matrix: &ArrayView2<'_, T>,
    power: T,
) -> Result<Array2<T>, MatrixFunctionError>
where
    T: MatrixFunctionScalar,
{
    matrix_power_impl(matrix, power)
}

/// Compute matrix power into `output`.
///
/// # Errors
/// Returns an error for invalid inputs or output shape mismatch.
pub fn matrix_power_into<T>(
    matrix: &Array2<T>,
    power: T,
    output: &mut Array2<T>,
) -> Result<(), MatrixFunctionError>
where
    T: MatrixFunctionScalar,
{
    let mut workspace = MatrixFunctionWorkspace::<T>::default();
    matrix_power_with_workspace_into(matrix, power, output, &mut workspace)
}

/// Compute matrix power from a view into `output`.
///
/// # Errors
/// Returns an error for invalid inputs or output shape mismatch.
pub fn matrix_power_view_into<T, S>(
    matrix: &ArrayView2<'_, T>,
    power: T,
    output: &mut ArrayBase<S, Ix2>,
) -> Result<(), MatrixFunctionError>
where
    T: MatrixFunctionScalar,
    S: DataMut<Elem = T>,
{
    let mut workspace = MatrixFunctionWorkspace::<T>::default();
    matrix_power_view_with_workspace_into(matrix, power, output, &mut workspace)
}

/// Compute matrix power into `output` using reusable `workspace`.
///
/// # Errors
/// Returns an error for invalid inputs or output shape mismatch.
pub fn matrix_power_with_workspace_into<T>(
    matrix: &Array2<T>,
    power: T,
    output: &mut Array2<T>,
    workspace: &mut MatrixFunctionWorkspace<T>,
) -> Result<(), MatrixFunctionError>
where
    T: MatrixFunctionScalar,
{
    matrix_power_view_with_workspace_into(&matrix.view(), power, output, workspace)
}

/// Compute matrix power from a view into `output` using reusable `workspace`.
///
/// # Errors
/// Returns an error for invalid inputs or output shape mismatch.
pub fn matrix_power_view_with_workspace_into<T, S>(
    matrix: &ArrayView2<'_, T>,
    power: T,
    output: &mut ArrayBase<S, Ix2>,
    workspace: &mut MatrixFunctionWorkspace<T>,
) -> Result<(), MatrixFunctionError>
where
    T: MatrixFunctionScalar,
    S: DataMut<Elem = T>,
{
    if output.dim() != matrix.dim() {
        return Err(MatrixFunctionError::InvalidInput(
            "output shape must match input matrix shape".to_string(),
        ));
    }
    let tolerance = base_tolerance::<T>();
    if !is_symmetric(matrix, tolerance) {
        return Err(MatrixFunctionError::NotSymmetric);
    }
    let decomposition =
        eigen::symmetric_view(matrix).map_err(|_| MatrixFunctionError::ConvergenceFailed)?;
    compose_real_symmetric_factors_into(
        &decomposition.eigenvalues.view(),
        &decomposition.eigenvectors.view(),
        output,
        &mut workspace.scratch,
        |value| Ok(num_traits::Float::powf(value, power)),
    )
}

/// Compute complex matrix power via Hermitian eigen decomposition.
///
/// # Errors
/// Returns an error for non-Hermitian inputs.
pub fn matrix_power_complex(
    matrix: &Array2<Complex64>,
    power: f64,
) -> Result<Array2<Complex64>, MatrixFunctionError> {
    matrix_power_complex_impl(&matrix.view(), power)
}

fn matrix_power_complex_impl(
    matrix: &ArrayView2<'_, Complex64>,
    power: f64,
) -> Result<Array2<Complex64>, MatrixFunctionError> {
    validate_square_complex(matrix)?;
    if !is_hermitian(matrix, DenseKernelPolicy::BASE_TOLERANCE) {
        return Err(MatrixFunctionError::NotSymmetric);
    }

    let (eigenvalues, eigenvectors) = hermitian_eigen_dispatch(matrix)?;
    let powered_values = eigenvalues.mapv(|value| num_traits::Float::powf(value, power));
    let diagonal = diagonal_from_real_complex(&powered_values);
    let qh = eigenvectors.t().mapv(|value| value.conj());
    Ok(eigenvectors.dot(&diagonal).dot(&qh))
}

/// Compute complex matrix power via Hermitian eigen decomposition from a matrix view.
///
/// # Errors
/// Returns an error for non-Hermitian inputs.
pub fn matrix_power_complex_view(
    matrix: &ArrayView2<'_, Complex64>,
    power: f64,
) -> Result<Array2<Complex64>, MatrixFunctionError> {
    matrix_power_complex_impl(matrix, power)
}

/// Compute complex matrix power into `output`.
///
/// # Errors
/// Returns an error for invalid inputs, output shape mismatch, non-Hermitian
/// inputs.
pub fn matrix_power_complex_into(
    matrix: &Array2<Complex64>,
    power: f64,
    output: &mut Array2<Complex64>,
) -> Result<(), MatrixFunctionError> {
    let mut workspace = MatrixFunctionComplexWorkspace::default();
    matrix_power_complex_with_workspace_into(matrix, power, output, &mut workspace)
}

/// Compute complex matrix power from a view into `output`.
///
/// # Errors
/// Returns an error for invalid inputs, output shape mismatch, non-Hermitian
/// inputs.
pub fn matrix_power_complex_view_into<S>(
    matrix: &ArrayView2<'_, Complex64>,
    power: f64,
    output: &mut ArrayBase<S, Ix2>,
) -> Result<(), MatrixFunctionError>
where
    S: DataMut<Elem = Complex64>,
{
    let mut workspace = MatrixFunctionComplexWorkspace::default();
    matrix_power_complex_view_with_workspace_into(matrix, power, output, &mut workspace)
}

/// Compute complex matrix power into `output` using reusable `workspace`.
///
/// # Errors
/// Returns an error for invalid inputs, output shape mismatch, non-Hermitian
/// inputs.
pub fn matrix_power_complex_with_workspace_into(
    matrix: &Array2<Complex64>,
    power: f64,
    output: &mut Array2<Complex64>,
    workspace: &mut MatrixFunctionComplexWorkspace,
) -> Result<(), MatrixFunctionError> {
    matrix_power_complex_view_with_workspace_into(&matrix.view(), power, output, workspace)
}

/// Compute complex matrix power from a view into `output` using reusable `workspace`.
///
/// # Errors
/// Returns an error for invalid inputs, output shape mismatch, non-Hermitian
/// inputs.
pub fn matrix_power_complex_view_with_workspace_into<S>(
    matrix: &ArrayView2<'_, Complex64>,
    power: f64,
    output: &mut ArrayBase<S, Ix2>,
    workspace: &mut MatrixFunctionComplexWorkspace,
) -> Result<(), MatrixFunctionError>
where
    S: DataMut<Elem = Complex64>,
{
    if output.dim() != matrix.dim() {
        return Err(MatrixFunctionError::InvalidInput(
            "output shape must match input matrix shape".to_string(),
        ));
    }
    if !is_hermitian(matrix, DenseKernelPolicy::BASE_TOLERANCE) {
        return Err(MatrixFunctionError::NotSymmetric);
    }
    let (eigenvalues, eigenvectors) = hermitian_eigen_dispatch(matrix)?;
    compose_complex_hermitian_factors_into(
        &eigenvalues.view(),
        &eigenvectors.view(),
        output,
        &mut workspace.scratch,
        |value| Ok(num_traits::Float::powf(value, power)),
    )
}

/// Compute matrix sign via eigen decomposition (symmetric matrices).
///
/// # Errors
/// Returns an error for non-symmetric inputs.
pub fn matrix_sign<T>(matrix: &Array2<T>) -> Result<Array2<T>, MatrixFunctionError>
where
    T: MatrixFunctionScalar,
{
    matrix_sign_impl(&matrix.view())
}

fn matrix_sign_impl<T>(matrix: &ArrayView2<'_, T>) -> Result<Array2<T>, MatrixFunctionError>
where
    T: MatrixFunctionScalar,
{
    validate_square(matrix)?;
    let tolerance = base_tolerance::<T>();
    if !is_symmetric(matrix, tolerance) {
        return Err(MatrixFunctionError::NotSymmetric);
    }

    let eigen =
        eigen::symmetric_view(matrix).map_err(|_| MatrixFunctionError::ConvergenceFailed)?;
    matrix_sign_from_factors_view(&eigen.eigenvalues.view(), &eigen.eigenvectors.view())
}

/// Compute matrix sign directly from real symmetric eigendecomposition factors.
///
/// # Errors
/// Returns an error if the eigen factor dimensions are inconsistent.
pub fn matrix_sign_from_factors_view<T>(
    eigenvalues: &ArrayView1<'_, T>,
    eigenvectors: &ArrayView2<'_, T>,
) -> Result<Array2<T>, MatrixFunctionError>
where
    T: MatrixFunctionScalar,
{
    validate_real_eigen_factor_dimensions(eigenvalues, eigenvectors)?;
    let tolerance = base_tolerance::<T>();
    let sign_values = eigenvalues.map(|value| {
        if *value > tolerance {
            T::one()
        } else if *value < -tolerance {
            -T::one()
        } else {
            T::zero()
        }
    });
    let diagonal = diagonal_from(&sign_values);
    Ok(eigenvectors.dot(&diagonal).dot(&eigenvectors.t()))
}

/// Compute matrix sign directly from real symmetric eigendecomposition factors into `output`.
///
/// # Errors
/// Returns an error if the eigen factor dimensions are inconsistent or `output` has the wrong
/// shape.
pub fn matrix_sign_from_factors_view_into<T, S>(
    eigenvalues: &ArrayView1<'_, T>,
    eigenvectors: &ArrayView2<'_, T>,
    output: &mut ArrayBase<S, Ix2>,
) -> Result<(), MatrixFunctionError>
where
    T: MatrixFunctionScalar,
    S: DataMut<Elem = T>,
{
    let tolerance = base_tolerance::<T>();
    let mut scratch = Array2::<T>::zeros((0, 0));
    compose_real_symmetric_factors_into(eigenvalues, eigenvectors, output, &mut scratch, |value| {
        Ok(if value > tolerance {
            T::one()
        } else if value < -tolerance {
            -T::one()
        } else {
            T::zero()
        })
    })
}

/// Compute matrix sign via eigen decomposition from a matrix view.
///
/// # Errors
/// Returns an error for non-symmetric inputs.
pub fn matrix_sign_view<T>(matrix: &ArrayView2<'_, T>) -> Result<Array2<T>, MatrixFunctionError>
where
    T: MatrixFunctionScalar,
{
    matrix_sign_impl(matrix)
}

/// Compute matrix sign into `output`.
///
/// # Errors
/// Returns an error for invalid inputs or output shape mismatch.
pub fn matrix_sign_into<T>(
    matrix: &Array2<T>,
    output: &mut Array2<T>,
) -> Result<(), MatrixFunctionError>
where
    T: MatrixFunctionScalar,
{
    let mut workspace = MatrixFunctionWorkspace::<T>::default();
    matrix_sign_with_workspace_into(matrix, output, &mut workspace)
}

/// Compute matrix sign from a view into `output`.
///
/// # Errors
/// Returns an error for invalid inputs or output shape mismatch.
pub fn matrix_sign_view_into<T, S>(
    matrix: &ArrayView2<'_, T>,
    output: &mut ArrayBase<S, Ix2>,
) -> Result<(), MatrixFunctionError>
where
    T: MatrixFunctionScalar,
    S: DataMut<Elem = T>,
{
    let mut workspace = MatrixFunctionWorkspace::<T>::default();
    matrix_sign_view_with_workspace_into(matrix, output, &mut workspace)
}

/// Compute matrix sign into `output` using reusable `workspace`.
///
/// # Errors
/// Returns an error for invalid inputs or output shape mismatch.
pub fn matrix_sign_with_workspace_into<T>(
    matrix: &Array2<T>,
    output: &mut Array2<T>,
    workspace: &mut MatrixFunctionWorkspace<T>,
) -> Result<(), MatrixFunctionError>
where
    T: MatrixFunctionScalar,
{
    matrix_sign_view_with_workspace_into(&matrix.view(), output, workspace)
}

/// Compute matrix sign from a view into `output` using reusable `workspace`.
///
/// # Errors
/// Returns an error for invalid inputs or output shape mismatch.
pub fn matrix_sign_view_with_workspace_into<T, S>(
    matrix: &ArrayView2<'_, T>,
    output: &mut ArrayBase<S, Ix2>,
    workspace: &mut MatrixFunctionWorkspace<T>,
) -> Result<(), MatrixFunctionError>
where
    T: MatrixFunctionScalar,
    S: DataMut<Elem = T>,
{
    if output.dim() != matrix.dim() {
        return Err(MatrixFunctionError::InvalidInput(
            "output shape must match input matrix shape".to_string(),
        ));
    }
    let tolerance = base_tolerance::<T>();
    if !is_symmetric(matrix, tolerance) {
        return Err(MatrixFunctionError::NotSymmetric);
    }
    let decomposition =
        eigen::symmetric_view(matrix).map_err(|_| MatrixFunctionError::ConvergenceFailed)?;
    compose_real_symmetric_factors_into(
        &decomposition.eigenvalues.view(),
        &decomposition.eigenvectors.view(),
        output,
        &mut workspace.scratch,
        |value| {
            Ok(if value > tolerance {
                T::one()
            } else if value < -tolerance {
                -T::one()
            } else {
                T::zero()
            })
        },
    )
}

/// Compute complex matrix sign via Hermitian eigen decomposition.
///
/// # Errors
/// Returns an error for non-Hermitian inputs.
pub fn matrix_sign_complex(
    matrix: &Array2<Complex64>,
) -> Result<Array2<Complex64>, MatrixFunctionError> {
    matrix_sign_complex_impl(&matrix.view())
}

fn matrix_sign_complex_impl(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<Array2<Complex64>, MatrixFunctionError> {
    validate_square_complex(matrix)?;
    if !is_hermitian(matrix, DenseKernelPolicy::BASE_TOLERANCE) {
        return Err(MatrixFunctionError::NotSymmetric);
    }

    let (eigenvalues, eigenvectors) = hermitian_eigen_dispatch(matrix)?;
    let sign_values = eigenvalues.map(|value| {
        if *value > DenseKernelPolicy::BASE_TOLERANCE {
            1.0
        } else if *value < -DenseKernelPolicy::BASE_TOLERANCE {
            -1.0
        } else {
            0.0
        }
    });
    let diagonal = diagonal_from_real_complex(&sign_values);
    let qh = eigenvectors.t().mapv(|value| value.conj());
    Ok(eigenvectors.dot(&diagonal).dot(&qh))
}

/// Compute complex matrix sign via Hermitian eigen decomposition from a matrix view.
///
/// # Errors
/// Returns an error for non-Hermitian inputs.
pub fn matrix_sign_complex_view(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<Array2<Complex64>, MatrixFunctionError> {
    matrix_sign_complex_impl(matrix)
}

/// Compute complex matrix sign into `output`.
///
/// # Errors
/// Returns an error for invalid inputs, output shape mismatch, non-Hermitian
/// inputs.
pub fn matrix_sign_complex_into(
    matrix: &Array2<Complex64>,
    output: &mut Array2<Complex64>,
) -> Result<(), MatrixFunctionError> {
    let mut workspace = MatrixFunctionComplexWorkspace::default();
    matrix_sign_complex_with_workspace_into(matrix, output, &mut workspace)
}

/// Compute complex matrix sign from a view into `output`.
///
/// # Errors
/// Returns an error for invalid inputs, output shape mismatch, non-Hermitian
/// inputs.
pub fn matrix_sign_complex_view_into<S>(
    matrix: &ArrayView2<'_, Complex64>,
    output: &mut ArrayBase<S, Ix2>,
) -> Result<(), MatrixFunctionError>
where
    S: DataMut<Elem = Complex64>,
{
    let mut workspace = MatrixFunctionComplexWorkspace::default();
    matrix_sign_complex_view_with_workspace_into(matrix, output, &mut workspace)
}

/// Compute complex matrix sign into `output` using reusable `workspace`.
///
/// # Errors
/// Returns an error for invalid inputs, output shape mismatch, non-Hermitian
/// inputs.
pub fn matrix_sign_complex_with_workspace_into(
    matrix: &Array2<Complex64>,
    output: &mut Array2<Complex64>,
    workspace: &mut MatrixFunctionComplexWorkspace,
) -> Result<(), MatrixFunctionError> {
    matrix_sign_complex_view_with_workspace_into(&matrix.view(), output, workspace)
}

/// Compute complex matrix sign from a view into `output` using reusable `workspace`.
///
/// # Errors
/// Returns an error for invalid inputs, output shape mismatch, non-Hermitian
/// inputs.
pub fn matrix_sign_complex_view_with_workspace_into<S>(
    matrix: &ArrayView2<'_, Complex64>,
    output: &mut ArrayBase<S, Ix2>,
    workspace: &mut MatrixFunctionComplexWorkspace,
) -> Result<(), MatrixFunctionError>
where
    S: DataMut<Elem = Complex64>,
{
    if output.dim() != matrix.dim() {
        return Err(MatrixFunctionError::InvalidInput(
            "output shape must match input matrix shape".to_string(),
        ));
    }
    if !is_hermitian(matrix, DenseKernelPolicy::BASE_TOLERANCE) {
        return Err(MatrixFunctionError::NotSymmetric);
    }
    let (eigenvalues, eigenvectors) = hermitian_eigen_dispatch(matrix)?;
    compose_complex_hermitian_factors_into(
        &eigenvalues.view(),
        &eigenvectors.view(),
        output,
        &mut workspace.scratch,
        |value| {
            Ok(if value > DenseKernelPolicy::BASE_TOLERANCE {
                1.0
            } else if value < -DenseKernelPolicy::BASE_TOLERANCE {
                -1.0
            } else {
                0.0
            })
        },
    )
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;
    use num_complex::Complex64;

    use super::*;

    #[test]
    fn exp_and_log_roundtrip_for_spd() {
        let matrix =
            Array2::from_shape_vec((2, 2), vec![2.0_f64, 1.0_f64, 1.0_f64, 2.0_f64]).unwrap();
        let log_matrix = matrix_log_eigen(&matrix).unwrap();
        let roundtrip = matrix_exp_eigen(&log_matrix).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!((roundtrip[[i, j]] - matrix[[i, j]]).abs() < 1e-4_f64);
            }
        }
    }

    #[test]
    fn non_symmetric_log_is_rejected() {
        let matrix =
            Array2::from_shape_vec((2, 2), vec![1.0_f64, 2.0_f64, 0.0_f64, 1.0_f64]).unwrap();
        let result = matrix_log_eigen(&matrix);
        assert!(matches!(result, Err(MatrixFunctionError::NotSymmetric)));
    }

    #[test]
    fn matrix_power_into_matches_allocating_path() {
        let matrix =
            Array2::from_shape_vec((2, 2), vec![3.0_f64, 1.0_f64, 1.0_f64, 3.0_f64]).unwrap();
        let expected = matrix_power(&matrix, 0.5_f64).unwrap();

        let mut output = Array2::<f64>::zeros((2, 2));
        let mut workspace = MatrixFunctionWorkspace::default();
        matrix_power_with_workspace_into(&matrix, 0.5_f64, &mut output, &mut workspace).unwrap();

        for i in 0..2 {
            for j in 0..2 {
                assert!((output[[i, j]] - expected[[i, j]]).abs() < 1e-8_f64);
            }
        }
    }

    #[test]
    fn exp_into_and_workspace_match_allocating_path() {
        let matrix =
            Array2::from_shape_vec((2, 2), vec![0.2_f64, 0.1_f64, 0.0_f64, 0.3_f64]).unwrap();
        let expected = matrix_exp(&matrix, 64, 1e-12_f64).unwrap();

        let mut output = Array2::<f64>::zeros((2, 2));
        matrix_exp_into(&matrix, 64, 1e-12_f64, &mut output).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!((output[[i, j]] - expected[[i, j]]).abs() < 1e-8_f64);
            }
        }
    }

    #[test]
    fn log_taylor_into_matches_allocating_path() {
        let matrix =
            Array2::from_shape_vec((2, 2), vec![1.1_f64, 0.0_f64, 0.0_f64, 0.9_f64]).unwrap();
        let expected = matrix_log_taylor(&matrix, 128, 1e-12_f64).unwrap();

        let mut output = Array2::<f64>::zeros((2, 2));
        matrix_log_taylor_into(&matrix, 128, 1e-12_f64, &mut output).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!((output[[i, j]] - expected[[i, j]]).abs() < 1e-8_f64);
            }
        }
    }

    #[test]
    fn into_rejects_bad_output_shape() {
        let matrix = Array2::eye(2);
        let mut bad = Array2::<f64>::zeros((1, 1));
        let err = matrix_log_eigen_into(&matrix, &mut bad).unwrap_err();
        assert!(matches!(err, MatrixFunctionError::InvalidInput(_)));
    }

    #[test]
    fn log_svd_rejects_singular_input() {
        let matrix =
            Array2::from_shape_vec((2, 2), vec![1.0_f64, 0.0_f64, 0.0_f64, 0.0_f64]).unwrap();
        let result = matrix_log_svd(&matrix);
        assert!(matches!(result, Err(MatrixFunctionError::NotPositiveDefinite)));
    }

    #[test]
    fn power_and_sign_reject_non_symmetric_input() {
        let matrix =
            Array2::from_shape_vec((2, 2), vec![1.0_f64, 2.0_f64, 0.0_f64, 1.0_f64]).unwrap();
        assert!(matches!(matrix_power(&matrix, 2.0_f64), Err(MatrixFunctionError::NotSymmetric)));
        assert!(matches!(matrix_sign(&matrix), Err(MatrixFunctionError::NotSymmetric)));
    }

    #[test]
    fn sign_into_matches_allocating_path() {
        let matrix =
            Array2::from_shape_vec((2, 2), vec![2.0_f64, 0.0_f64, 0.0_f64, -3.0_f64]).unwrap();
        let expected = matrix_sign(&matrix).unwrap();

        let mut output = Array2::<f64>::zeros((2, 2));
        matrix_sign_into(&matrix, &mut output).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!((output[[i, j]] - expected[[i, j]]).abs() < 1e-10_f64);
            }
        }
    }

    #[test]
    fn exp_eigen_falls_back_for_non_symmetric_input() {
        let matrix =
            Array2::from_shape_vec((2, 2), vec![0.0_f64, 1.0_f64, 0.0_f64, 0.0_f64]).unwrap();
        let eigen_path = matrix_exp_eigen(&matrix).unwrap();
        let taylor_path = matrix_exp(&matrix, 128, 1e-12_f64).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!((eigen_path[[i, j]] - taylor_path[[i, j]]).abs() < 1e-8_f64);
            }
        }
    }

    #[test]
    fn matrix_functions_reject_non_finite_input() {
        let matrix =
            Array2::from_shape_vec((2, 2), vec![1.0_f64, f64::NAN, 0.0_f64, 1.0_f64]).unwrap();
        let result = matrix_exp(&matrix, 32, 1e-8_f64);
        assert!(matches!(result, Err(MatrixFunctionError::InvalidInput(_))));
    }

    #[test]
    fn into_variants_reject_bad_output_shape() {
        let matrix = Array2::eye(2);
        let mut bad = Array2::<f64>::zeros((1, 1));
        assert!(matches!(
            matrix_exp_into(&matrix, 32, 1e-8_f64, &mut bad),
            Err(MatrixFunctionError::InvalidInput(_))
        ));
        assert!(matches!(
            matrix_log_svd_into(&matrix, &mut bad),
            Err(MatrixFunctionError::InvalidInput(_))
        ));
        assert!(matches!(
            matrix_power_into(&matrix, 0.5_f64, &mut bad),
            Err(MatrixFunctionError::InvalidInput(_))
        ));
        assert!(matches!(
            matrix_sign_into(&matrix, &mut bad),
            Err(MatrixFunctionError::InvalidInput(_))
        ));
    }

    #[test]
    fn matrix_log_eigen_rejects_non_positive_eigenvalues() {
        let matrix =
            Array2::from_shape_vec((2, 2), vec![1.0_f64, 0.0_f64, 0.0_f64, 0.0_f64]).unwrap();
        let result = matrix_log_eigen(&matrix);
        assert!(matches!(result, Err(MatrixFunctionError::NotPositiveDefinite)));
    }

    #[test]
    fn matrix_sign_handles_negative_positive_and_zero_spectrum() {
        let matrix = Array2::from_shape_vec((3, 3), vec![
            2.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, -4.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64,
        ])
        .unwrap();
        let sign = matrix_sign(&matrix).unwrap();
        assert!((sign[[0, 0]] - 1.0_f64).abs() < 1e-10_f64);
        assert!((sign[[1, 1]] + 1.0_f64).abs() < 1e-10_f64);
        assert!(sign[[2, 2]].abs() < 1e-10_f64);
    }

    #[test]
    fn zero_max_terms_is_clamped_to_single_iteration() {
        let matrix = Array2::eye(2);
        let exp = matrix_exp(&matrix, 0, 1e-12_f64).unwrap();
        assert!(exp[[0, 0]].is_finite());
        assert!(exp[[1, 1]].is_finite());

        let log = matrix_log_taylor(&matrix, 0, 1e-12_f64).unwrap();
        assert!(log[[0, 0]].abs() < 1e-12_f64);
        assert!(log[[1, 1]].abs() < 1e-12_f64);
    }

    #[test]
    fn view_variants_match_owned() {
        let symmetric =
            Array2::from_shape_vec((2, 2), vec![2.0_f64, 0.2_f64, 0.2_f64, 3.0_f64]).unwrap();
        let non_symmetric =
            Array2::from_shape_vec((2, 2), vec![0.0_f64, 1.0_f64, 0.0_f64, 0.0_f64]).unwrap();

        let exp_owned = matrix_exp(&symmetric, 64, 1e-12_f64).unwrap();
        let exp_view = matrix_exp_view(&symmetric.view(), 64, 1e-12_f64).unwrap();
        let exp_eigen_owned = matrix_exp_eigen(&non_symmetric).unwrap();
        let exp_eigen_view = matrix_exp_eigen_view(&non_symmetric.view()).unwrap();

        let log_taylor_owned = matrix_log_taylor(&symmetric, 64, 1e-12_f64).unwrap();
        let log_taylor_view = matrix_log_taylor_view(&symmetric.view(), 64, 1e-12_f64).unwrap();
        let log_eigen_owned = matrix_log_eigen(&symmetric).unwrap();
        let log_eigen_view = matrix_log_eigen_view(&symmetric.view()).unwrap();
        let log_svd_owned = matrix_log_svd(&symmetric).unwrap();
        let log_svd_view = matrix_log_svd_view(&symmetric.view()).unwrap();

        let power_owned = matrix_power(&symmetric, 0.5_f64).unwrap();
        let power_view = matrix_power_view(&symmetric.view(), 0.5_f64).unwrap();
        let sign_owned = matrix_sign(&symmetric).unwrap();
        let sign_view = matrix_sign_view(&symmetric.view()).unwrap();

        for i in 0..2 {
            for j in 0..2 {
                assert!((exp_owned[[i, j]] - exp_view[[i, j]]).abs() < 1e-12_f64);
                assert!((exp_eigen_owned[[i, j]] - exp_eigen_view[[i, j]]).abs() < 1e-12_f64);
                assert!((log_taylor_owned[[i, j]] - log_taylor_view[[i, j]]).abs() < 1e-12_f64);
                assert!((log_eigen_owned[[i, j]] - log_eigen_view[[i, j]]).abs() < 1e-12_f64);
                assert!((log_svd_owned[[i, j]] - log_svd_view[[i, j]]).abs() < 1e-12_f64);
                assert!((power_owned[[i, j]] - power_view[[i, j]]).abs() < 1e-12_f64);
                assert!((sign_owned[[i, j]] - sign_view[[i, j]]).abs() < 1e-12_f64);
            }
        }
    }

    #[test]
    fn real_f32_paths_match_expected() {
        let symmetric =
            Array2::from_shape_vec((2, 2), vec![2.0_f32, 0.2_f32, 0.2_f32, 3.0_f32]).unwrap();
        let near_identity =
            Array2::from_shape_vec((2, 2), vec![1.1_f32, 0.0_f32, 0.0_f32, 0.9_f32]).unwrap();
        let non_symmetric =
            Array2::from_shape_vec((2, 2), vec![0.0_f32, 1.0_f32, 0.0_f32, 0.0_f32]).unwrap();

        let exp_owned = matrix_exp(&symmetric, 64, 1e-6_f32).unwrap();
        let exp_view = matrix_exp_view(&symmetric.view(), 64, 1e-6_f32).unwrap();
        let exp_eigen = matrix_exp_eigen(&non_symmetric).unwrap();
        let exp_taylor = matrix_exp(&non_symmetric, 128, 1e-6_f32).unwrap();

        let log_taylor = matrix_log_taylor(&near_identity, 128, 1e-6_f32).unwrap();
        let log_eigen = matrix_log_eigen(&symmetric).unwrap();
        let log_svd = matrix_log_svd(&symmetric).unwrap();

        let power_owned = matrix_power(&symmetric, 0.5_f32).unwrap();
        let power_view = matrix_power_view(&symmetric.view(), 0.5_f32).unwrap();
        let sign_owned = matrix_sign(&symmetric).unwrap();
        let mut sign_into = Array2::<f32>::zeros((2, 2));
        matrix_sign_into(&symmetric, &mut sign_into).unwrap();

        for i in 0..2 {
            for j in 0..2 {
                assert!((exp_owned[[i, j]] - exp_view[[i, j]]).abs() < 1e-5_f32);
                assert!((exp_eigen[[i, j]] - exp_taylor[[i, j]]).abs() < 1e-4_f32);
                assert!((power_owned[[i, j]] - power_view[[i, j]]).abs() < 1e-5_f32);
                assert!((sign_owned[[i, j]] - sign_into[[i, j]]).abs() < 1e-5_f32);
                assert!(log_taylor[[i, j]].is_finite());
                assert!(log_eigen[[i, j]].is_finite());
                assert!(log_svd[[i, j]].is_finite());
            }
        }
    }

    #[test]
    fn complex_exp_variants_match_and_into_paths_work() {
        let matrix = Array2::from_shape_vec((2, 2), vec![
            Complex64::new(0.0_f64, 0.0_f64),
            Complex64::new(0.0_f64, 1.0_f64),
            Complex64::new(0.0_f64, 0.0_f64),
            Complex64::new(0.0_f64, 0.0_f64),
        ])
        .unwrap();

        let owned = matrix_exp_complex(&matrix, 32, 1e-12_f64).unwrap();
        let viewed = matrix_exp_complex_view(&matrix.view(), 32, 1e-12_f64).unwrap();

        let mut output = Array2::<Complex64>::zeros((2, 2));
        matrix_exp_complex_into(&matrix, 32, 1e-12_f64, &mut output).unwrap();

        let mut workspace = MatrixFunctionComplexWorkspace::default();
        let mut workspace_output = Array2::<Complex64>::zeros((2, 2));
        matrix_exp_complex_with_workspace_into(
            &matrix,
            32,
            1e-12_f64,
            &mut workspace_output,
            &mut workspace,
        )
        .unwrap();

        let expected = Array2::from_shape_vec((2, 2), vec![
            Complex64::new(1.0_f64, 0.0_f64),
            Complex64::new(0.0_f64, 1.0_f64),
            Complex64::new(0.0_f64, 0.0_f64),
            Complex64::new(1.0_f64, 0.0_f64),
        ])
        .unwrap();

        for i in 0..2 {
            for j in 0..2 {
                assert!((owned[[i, j]] - viewed[[i, j]]).norm() < 1e-12_f64);
                assert!((owned[[i, j]] - output[[i, j]]).norm() < 1e-12_f64);
                assert!((owned[[i, j]] - workspace_output[[i, j]]).norm() < 1e-12_f64);
                assert!((owned[[i, j]] - expected[[i, j]]).norm() < 1e-10_f64);
            }
        }
    }

    #[test]
    fn complex_into_rejects_bad_output_shape() {
        let matrix = Array2::from_shape_vec((2, 2), vec![
            Complex64::new(1.0_f64, 0.0_f64),
            Complex64::new(0.0_f64, 0.0_f64),
            Complex64::new(0.0_f64, 0.0_f64),
            Complex64::new(1.0_f64, 0.0_f64),
        ])
        .unwrap();
        let mut bad = Array2::<Complex64>::zeros((1, 1));
        assert!(matches!(
            matrix_exp_complex_into(&matrix, 16, 1e-10_f64, &mut bad),
            Err(MatrixFunctionError::InvalidInput(_))
        ));
        assert!(matches!(
            matrix_log_svd_complex_into(&matrix, &mut bad),
            Err(MatrixFunctionError::InvalidInput(_))
        ));
        assert!(matches!(
            matrix_log_eigen_complex_into(&matrix, &mut bad),
            Err(MatrixFunctionError::InvalidInput(_))
        ));
        assert!(matches!(
            matrix_power_complex_into(&matrix, 0.5_f64, &mut bad),
            Err(MatrixFunctionError::InvalidInput(_))
        ));
        assert!(matches!(
            matrix_sign_complex_into(&matrix, &mut bad),
            Err(MatrixFunctionError::InvalidInput(_))
        ));
    }

    #[test]
    fn complex_log_svd_paths_work() {
        let matrix = Array2::from_shape_vec((2, 2), vec![
            Complex64::new(2.0_f64, 0.0_f64),
            Complex64::new(0.0_f64, 0.0_f64),
            Complex64::new(0.0_f64, 0.0_f64),
            Complex64::new(3.0_f64, 0.0_f64),
        ])
        .unwrap();

        let owned = matrix_log_svd_complex(&matrix).unwrap();
        let viewed = matrix_log_svd_complex_view(&matrix.view()).unwrap();
        let mut output = Array2::<Complex64>::zeros((2, 2));
        matrix_log_svd_complex_into(&matrix, &mut output).unwrap();
        let mut workspace = MatrixFunctionComplexWorkspace::default();
        let mut workspace_output = Array2::<Complex64>::zeros((2, 2));
        matrix_log_svd_complex_with_workspace_into(&matrix, &mut workspace_output, &mut workspace)
            .unwrap();

        assert!((owned[[0, 0]].re - 2.0_f64.ln()).abs() < 1e-10_f64);
        assert!(owned[[0, 0]].im.abs() < 1e-10_f64);
        assert!((owned[[1, 1]].re - 3.0_f64.ln()).abs() < 1e-10_f64);
        assert!(owned[[1, 1]].im.abs() < 1e-10_f64);
        assert!(owned[[0, 1]].norm() < 1e-10_f64);
        assert!(owned[[1, 0]].norm() < 1e-10_f64);

        for i in 0..2 {
            for j in 0..2 {
                assert!((owned[[i, j]] - viewed[[i, j]]).norm() < 1e-12_f64);
                assert!((owned[[i, j]] - output[[i, j]]).norm() < 1e-12_f64);
                assert!((owned[[i, j]] - workspace_output[[i, j]]).norm() < 1e-12_f64);
            }
        }
    }

    #[test]
    fn complex_eigen_power_sign_paths_work() {
        let matrix = Array2::from_shape_vec((2, 2), vec![
            Complex64::new(4.0_f64, 0.0_f64),
            Complex64::new(0.0_f64, 0.0_f64),
            Complex64::new(0.0_f64, 0.0_f64),
            Complex64::new(9.0_f64, 0.0_f64),
        ])
        .unwrap();
        let signed_matrix = Array2::from_shape_vec((2, 2), vec![
            Complex64::new(-4.0_f64, 0.0_f64),
            Complex64::new(0.0_f64, 0.0_f64),
            Complex64::new(0.0_f64, 0.0_f64),
            Complex64::new(9.0_f64, 0.0_f64),
        ])
        .unwrap();

        let exp_eigen_owned = matrix_exp_eigen_complex(&matrix).unwrap();
        let exp_eigen_view = matrix_exp_eigen_complex_view(&matrix.view()).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!((exp_eigen_owned[[i, j]] - exp_eigen_view[[i, j]]).norm() < 1e-12_f64);
            }
        }

        let log_owned = matrix_log_eigen_complex(&matrix).unwrap();
        let log_view = matrix_log_eigen_complex_view(&matrix.view()).unwrap();
        let mut log_into = Array2::<Complex64>::zeros((2, 2));
        matrix_log_eigen_complex_into(&matrix, &mut log_into).unwrap();
        let mut complex_workspace = MatrixFunctionComplexWorkspace::default();
        let mut log_ws = Array2::<Complex64>::zeros((2, 2));
        matrix_log_eigen_complex_with_workspace_into(&matrix, &mut log_ws, &mut complex_workspace)
            .unwrap();

        assert!((log_owned[[0, 0]].re - 4.0_f64.ln()).abs() < 1e-10_f64);
        assert!((log_owned[[1, 1]].re - 9.0_f64.ln()).abs() < 1e-10_f64);

        for i in 0..2 {
            for j in 0..2 {
                assert!((log_owned[[i, j]] - log_view[[i, j]]).norm() < 1e-12_f64);
                assert!((log_owned[[i, j]] - log_into[[i, j]]).norm() < 1e-12_f64);
                assert!((log_owned[[i, j]] - log_ws[[i, j]]).norm() < 1e-12_f64);
            }
        }

        let power_owned = matrix_power_complex(&matrix, 0.5_f64).unwrap();
        let power_view = matrix_power_complex_view(&matrix.view(), 0.5_f64).unwrap();
        let mut power_into = Array2::<Complex64>::zeros((2, 2));
        matrix_power_complex_into(&matrix, 0.5_f64, &mut power_into).unwrap();
        let mut power_ws = Array2::<Complex64>::zeros((2, 2));
        matrix_power_complex_with_workspace_into(
            &matrix,
            0.5_f64,
            &mut power_ws,
            &mut complex_workspace,
        )
        .unwrap();
        assert!((power_owned[[0, 0]].re - 2.0_f64).abs() < 1e-10_f64);
        assert!((power_owned[[1, 1]].re - 3.0_f64).abs() < 1e-10_f64);
        for i in 0..2 {
            for j in 0..2 {
                assert!((power_owned[[i, j]] - power_view[[i, j]]).norm() < 1e-12_f64);
                assert!((power_owned[[i, j]] - power_into[[i, j]]).norm() < 1e-12_f64);
                assert!((power_owned[[i, j]] - power_ws[[i, j]]).norm() < 1e-12_f64);
            }
        }

        let sign_owned = matrix_sign_complex(&signed_matrix).unwrap();
        let sign_view = matrix_sign_complex_view(&signed_matrix.view()).unwrap();
        let mut sign_into = Array2::<Complex64>::zeros((2, 2));
        matrix_sign_complex_into(&signed_matrix, &mut sign_into).unwrap();
        let mut sign_ws = Array2::<Complex64>::zeros((2, 2));
        matrix_sign_complex_with_workspace_into(
            &signed_matrix,
            &mut sign_ws,
            &mut complex_workspace,
        )
        .unwrap();
        assert!((sign_owned[[0, 0]] - Complex64::new(-1.0_f64, 0.0_f64)).norm() < 1e-10_f64);
        assert!((sign_owned[[1, 1]] - Complex64::new(1.0_f64, 0.0_f64)).norm() < 1e-10_f64);
        for i in 0..2 {
            for j in 0..2 {
                assert!((sign_owned[[i, j]] - sign_view[[i, j]]).norm() < 1e-12_f64);
                assert!((sign_owned[[i, j]] - sign_into[[i, j]]).norm() < 1e-12_f64);
                assert!((sign_owned[[i, j]] - sign_ws[[i, j]]).norm() < 1e-12_f64);
            }
        }
    }

    #[test]
    #[expect(clippy::too_many_lines)]
    fn real_view_into_and_workspace_view_into_variants_match_allocating_paths() {
        let matrix =
            Array2::from_shape_vec((2, 2), vec![2.0_f64, 0.2_f64, 0.2_f64, 3.0_f64]).unwrap();

        let exp_expected = matrix_exp(&matrix, 64, 1e-12_f64).unwrap();
        let mut exp_into = Array2::<f64>::zeros((2, 2));
        {
            let mut out = exp_into.view_mut();
            matrix_exp_view_into(&matrix.view(), 64, 1e-12_f64, &mut out).unwrap();
        }
        let mut exp_ws = Array2::<f64>::zeros((2, 2));
        let mut exp_workspace = MatrixFunctionWorkspace::default();
        {
            let mut out = exp_ws.view_mut();
            matrix_exp_view_with_workspace_into(
                &matrix.view(),
                64,
                1e-12_f64,
                &mut out,
                &mut exp_workspace,
            )
            .unwrap();
        }

        let exp_eigen_expected = matrix_exp_eigen(&matrix).unwrap();
        let mut exp_eigen_into = Array2::<f64>::zeros((2, 2));
        {
            let mut out = exp_eigen_into.view_mut();
            matrix_exp_eigen_view_into(&matrix.view(), &mut out).unwrap();
        }
        let mut exp_eigen_ws = Array2::<f64>::zeros((2, 2));
        let mut exp_eigen_workspace = MatrixFunctionWorkspace::default();
        {
            let mut out = exp_eigen_ws.view_mut();
            matrix_exp_eigen_view_with_workspace_into(
                &matrix.view(),
                &mut out,
                &mut exp_eigen_workspace,
            )
            .unwrap();
        }

        let log_taylor_expected = matrix_log_taylor(&matrix, 64, 1e-12_f64).unwrap();
        let mut log_taylor_into = Array2::<f64>::zeros((2, 2));
        {
            let mut out = log_taylor_into.view_mut();
            matrix_log_taylor_view_into(&matrix.view(), 64, 1e-12_f64, &mut out).unwrap();
        }
        let mut log_taylor_ws = Array2::<f64>::zeros((2, 2));
        let mut log_taylor_workspace = MatrixFunctionWorkspace::default();
        {
            let mut out = log_taylor_ws.view_mut();
            matrix_log_taylor_view_with_workspace_into(
                &matrix.view(),
                64,
                1e-12_f64,
                &mut out,
                &mut log_taylor_workspace,
            )
            .unwrap();
        }

        let log_eigen_expected = matrix_log_eigen(&matrix).unwrap();
        let mut log_eigen_into = Array2::<f64>::zeros((2, 2));
        {
            let mut out = log_eigen_into.view_mut();
            matrix_log_eigen_view_into(&matrix.view(), &mut out).unwrap();
        }
        let mut log_eigen_ws = Array2::<f64>::zeros((2, 2));
        let mut log_eigen_workspace = MatrixFunctionWorkspace::default();
        {
            let mut out = log_eigen_ws.view_mut();
            matrix_log_eigen_view_with_workspace_into(
                &matrix.view(),
                &mut out,
                &mut log_eigen_workspace,
            )
            .unwrap();
        }

        let log_svd_expected = matrix_log_svd(&matrix).unwrap();
        let mut log_svd_into = Array2::<f64>::zeros((2, 2));
        {
            let mut out = log_svd_into.view_mut();
            matrix_log_svd_view_into(&matrix.view(), &mut out).unwrap();
        }
        let mut log_svd_ws = Array2::<f64>::zeros((2, 2));
        let mut log_svd_workspace = MatrixFunctionWorkspace::default();
        {
            let mut out = log_svd_ws.view_mut();
            matrix_log_svd_view_with_workspace_into(
                &matrix.view(),
                &mut out,
                &mut log_svd_workspace,
            )
            .unwrap();
        }

        let power_expected = matrix_power(&matrix, 0.5_f64).unwrap();
        let mut power_into = Array2::<f64>::zeros((2, 2));
        {
            let mut out = power_into.view_mut();
            matrix_power_view_into(&matrix.view(), 0.5_f64, &mut out).unwrap();
        }
        let mut power_ws = Array2::<f64>::zeros((2, 2));
        let mut power_workspace = MatrixFunctionWorkspace::default();
        {
            let mut out = power_ws.view_mut();
            matrix_power_view_with_workspace_into(
                &matrix.view(),
                0.5_f64,
                &mut out,
                &mut power_workspace,
            )
            .unwrap();
        }

        let sign_expected = matrix_sign(&matrix).unwrap();
        let mut sign_into = Array2::<f64>::zeros((2, 2));
        {
            let mut out = sign_into.view_mut();
            matrix_sign_view_into(&matrix.view(), &mut out).unwrap();
        }
        let mut sign_ws = Array2::<f64>::zeros((2, 2));
        let mut sign_workspace = MatrixFunctionWorkspace::default();
        {
            let mut out = sign_ws.view_mut();
            matrix_sign_view_with_workspace_into(&matrix.view(), &mut out, &mut sign_workspace)
                .unwrap();
        }

        for i in 0..2 {
            for j in 0..2 {
                assert!((exp_expected[[i, j]] - exp_into[[i, j]]).abs() < 1e-12_f64);
                assert!((exp_expected[[i, j]] - exp_ws[[i, j]]).abs() < 1e-12_f64);
                assert!((exp_eigen_expected[[i, j]] - exp_eigen_into[[i, j]]).abs() < 1e-12_f64);
                assert!((exp_eigen_expected[[i, j]] - exp_eigen_ws[[i, j]]).abs() < 1e-12_f64);
                assert!((log_taylor_expected[[i, j]] - log_taylor_into[[i, j]]).abs() < 1e-12_f64);
                assert!((log_taylor_expected[[i, j]] - log_taylor_ws[[i, j]]).abs() < 1e-12_f64);
                assert!((log_eigen_expected[[i, j]] - log_eigen_into[[i, j]]).abs() < 1e-12_f64);
                assert!((log_eigen_expected[[i, j]] - log_eigen_ws[[i, j]]).abs() < 1e-12_f64);
                assert!((log_svd_expected[[i, j]] - log_svd_into[[i, j]]).abs() < 1e-12_f64);
                assert!((log_svd_expected[[i, j]] - log_svd_ws[[i, j]]).abs() < 1e-12_f64);
                assert!((power_expected[[i, j]] - power_into[[i, j]]).abs() < 1e-12_f64);
                assert!((power_expected[[i, j]] - power_ws[[i, j]]).abs() < 1e-12_f64);
                assert!((sign_expected[[i, j]] - sign_into[[i, j]]).abs() < 1e-12_f64);
                assert!((sign_expected[[i, j]] - sign_ws[[i, j]]).abs() < 1e-12_f64);
            }
        }
    }

    #[test]
    #[expect(clippy::too_many_lines)]
    fn complex_view_into_and_workspace_view_into_variants_match_allocating_paths() {
        let matrix = Array2::from_shape_vec((2, 2), vec![
            Complex64::new(4.0_f64, 0.0_f64),
            Complex64::new(0.0_f64, 0.0_f64),
            Complex64::new(0.0_f64, 0.0_f64),
            Complex64::new(9.0_f64, 0.0_f64),
        ])
        .unwrap();
        let signed_matrix = Array2::from_shape_vec((2, 2), vec![
            Complex64::new(-4.0_f64, 0.0_f64),
            Complex64::new(0.0_f64, 0.0_f64),
            Complex64::new(0.0_f64, 0.0_f64),
            Complex64::new(9.0_f64, 0.0_f64),
        ])
        .unwrap();

        let exp_expected = matrix_exp_complex(&matrix, 32, 1e-12_f64).unwrap();
        let mut exp_into = Array2::<Complex64>::zeros((2, 2));
        {
            let mut out = exp_into.view_mut();
            matrix_exp_complex_view_into(&matrix.view(), 32, 1e-12_f64, &mut out).unwrap();
        }
        let mut exp_ws = Array2::<Complex64>::zeros((2, 2));
        let mut complex_workspace = MatrixFunctionComplexWorkspace::default();
        {
            let mut out = exp_ws.view_mut();
            matrix_exp_complex_view_with_workspace_into(
                &matrix.view(),
                32,
                1e-12_f64,
                &mut out,
                &mut complex_workspace,
            )
            .unwrap();
        }

        let exp_eigen_expected = matrix_exp_eigen_complex(&matrix).unwrap();
        let mut exp_eigen_into = Array2::<Complex64>::zeros((2, 2));
        {
            let mut out = exp_eigen_into.view_mut();
            matrix_exp_eigen_complex_view_into(&matrix.view(), &mut out).unwrap();
        }
        let mut exp_eigen_ws = Array2::<Complex64>::zeros((2, 2));
        {
            let mut out = exp_eigen_ws.view_mut();
            matrix_exp_eigen_complex_view_with_workspace_into(
                &matrix.view(),
                &mut out,
                &mut complex_workspace,
            )
            .unwrap();
        }

        let log_eigen_expected = matrix_log_eigen_complex(&matrix).unwrap();
        let mut log_eigen_into = Array2::<Complex64>::zeros((2, 2));
        {
            let mut out = log_eigen_into.view_mut();
            matrix_log_eigen_complex_view_into(&matrix.view(), &mut out).unwrap();
        }
        let mut log_eigen_ws = Array2::<Complex64>::zeros((2, 2));
        {
            let mut out = log_eigen_ws.view_mut();
            matrix_log_eigen_complex_view_with_workspace_into(
                &matrix.view(),
                &mut out,
                &mut complex_workspace,
            )
            .unwrap();
        }

        let log_svd_expected = matrix_log_svd_complex(&matrix).unwrap();
        let mut log_svd_into = Array2::<Complex64>::zeros((2, 2));
        {
            let mut out = log_svd_into.view_mut();
            matrix_log_svd_complex_view_into(&matrix.view(), &mut out).unwrap();
        }
        let mut log_svd_ws = Array2::<Complex64>::zeros((2, 2));
        {
            let mut out = log_svd_ws.view_mut();
            matrix_log_svd_complex_view_with_workspace_into(
                &matrix.view(),
                &mut out,
                &mut complex_workspace,
            )
            .unwrap();
        }

        let power_expected = matrix_power_complex(&matrix, 0.5_f64).unwrap();
        let mut power_into = Array2::<Complex64>::zeros((2, 2));
        {
            let mut out = power_into.view_mut();
            matrix_power_complex_view_into(&matrix.view(), 0.5_f64, &mut out).unwrap();
        }
        let mut power_ws = Array2::<Complex64>::zeros((2, 2));
        {
            let mut out = power_ws.view_mut();
            matrix_power_complex_view_with_workspace_into(
                &matrix.view(),
                0.5_f64,
                &mut out,
                &mut complex_workspace,
            )
            .unwrap();
        }

        let sign_expected = matrix_sign_complex(&signed_matrix).unwrap();
        let mut sign_into = Array2::<Complex64>::zeros((2, 2));
        {
            let mut out = sign_into.view_mut();
            matrix_sign_complex_view_into(&signed_matrix.view(), &mut out).unwrap();
        }
        let mut sign_ws = Array2::<Complex64>::zeros((2, 2));
        {
            let mut out = sign_ws.view_mut();
            matrix_sign_complex_view_with_workspace_into(
                &signed_matrix.view(),
                &mut out,
                &mut complex_workspace,
            )
            .unwrap();
        }

        for i in 0..2 {
            for j in 0..2 {
                assert!((exp_expected[[i, j]] - exp_into[[i, j]]).norm() < 1e-12_f64);
                assert!((exp_expected[[i, j]] - exp_ws[[i, j]]).norm() < 1e-12_f64);
                assert!((exp_eigen_expected[[i, j]] - exp_eigen_into[[i, j]]).norm() < 1e-12_f64);
                assert!((exp_eigen_expected[[i, j]] - exp_eigen_ws[[i, j]]).norm() < 1e-12_f64);
                assert!((log_eigen_expected[[i, j]] - log_eigen_into[[i, j]]).norm() < 1e-12_f64);
                assert!((log_eigen_expected[[i, j]] - log_eigen_ws[[i, j]]).norm() < 1e-12_f64);
                assert!((log_svd_expected[[i, j]] - log_svd_into[[i, j]]).norm() < 1e-12_f64);
                assert!((log_svd_expected[[i, j]] - log_svd_ws[[i, j]]).norm() < 1e-12_f64);
                assert!((power_expected[[i, j]] - power_into[[i, j]]).norm() < 1e-12_f64);
                assert!((power_expected[[i, j]] - power_ws[[i, j]]).norm() < 1e-12_f64);
                assert!((sign_expected[[i, j]] - sign_into[[i, j]]).norm() < 1e-12_f64);
                assert!((sign_expected[[i, j]] - sign_ws[[i, j]]).norm() < 1e-12_f64);
            }
        }
    }

    #[test]
    fn view_into_variants_reject_bad_output_shape() {
        let matrix =
            Array2::from_shape_vec((2, 2), vec![2.0_f64, 0.2_f64, 0.2_f64, 3.0_f64]).unwrap();
        let mut bad = Array2::<f64>::zeros((1, 1));
        let mut bad_view = bad.view_mut();
        assert!(matches!(
            matrix_exp_view_into(&matrix.view(), 32, 1e-8_f64, &mut bad_view),
            Err(MatrixFunctionError::InvalidInput(_))
        ));
        assert!(matches!(
            matrix_exp_eigen_view_into(&matrix.view(), &mut bad_view),
            Err(MatrixFunctionError::InvalidInput(_))
        ));
        assert!(matches!(
            matrix_log_taylor_view_into(&matrix.view(), 32, 1e-8_f64, &mut bad_view),
            Err(MatrixFunctionError::InvalidInput(_))
        ));
        assert!(matches!(
            matrix_log_eigen_view_into(&matrix.view(), &mut bad_view),
            Err(MatrixFunctionError::InvalidInput(_))
        ));
        assert!(matches!(
            matrix_log_svd_view_into(&matrix.view(), &mut bad_view),
            Err(MatrixFunctionError::InvalidInput(_))
        ));
        assert!(matches!(
            matrix_power_view_into(&matrix.view(), 0.5_f64, &mut bad_view),
            Err(MatrixFunctionError::InvalidInput(_))
        ));
        assert!(matches!(
            matrix_sign_view_into(&matrix.view(), &mut bad_view),
            Err(MatrixFunctionError::InvalidInput(_))
        ));

        let complex_matrix = Array2::from_shape_vec((2, 2), vec![
            Complex64::new(4.0_f64, 0.0_f64),
            Complex64::new(0.0_f64, 0.0_f64),
            Complex64::new(0.0_f64, 0.0_f64),
            Complex64::new(9.0_f64, 0.0_f64),
        ])
        .unwrap();
        let mut complex_bad = Array2::<Complex64>::zeros((1, 1));
        let mut complex_bad_view = complex_bad.view_mut();
        assert!(matches!(
            matrix_exp_complex_view_into(
                &complex_matrix.view(),
                32,
                1e-8_f64,
                &mut complex_bad_view
            ),
            Err(MatrixFunctionError::InvalidInput(_))
        ));
        assert!(matches!(
            matrix_exp_eigen_complex_view_into(&complex_matrix.view(), &mut complex_bad_view),
            Err(MatrixFunctionError::InvalidInput(_))
        ));
        assert!(matches!(
            matrix_log_eigen_complex_view_into(&complex_matrix.view(), &mut complex_bad_view),
            Err(MatrixFunctionError::InvalidInput(_))
        ));
        assert!(matches!(
            matrix_log_svd_complex_view_into(&complex_matrix.view(), &mut complex_bad_view),
            Err(MatrixFunctionError::InvalidInput(_))
        ));
        assert!(matches!(
            matrix_power_complex_view_into(&complex_matrix.view(), 0.5_f64, &mut complex_bad_view),
            Err(MatrixFunctionError::InvalidInput(_))
        ));
        assert!(matches!(
            matrix_sign_complex_view_into(&complex_matrix.view(), &mut complex_bad_view),
            Err(MatrixFunctionError::InvalidInput(_))
        ));
    }

    #[test]
    fn matrix_log_svd_from_factors_matches_direct() {
        let matrix =
            Array2::from_shape_vec((2, 2), vec![2.0_f64, 0.2_f64, 0.2_f64, 3.0_f64]).unwrap();
        let direct = matrix_log_svd(&matrix).unwrap();
        let svd = svd::decompose(&matrix).unwrap();
        let from_factors = matrix_log_svd_from_svd(&svd.u, &svd.singular_values, &svd.vt).unwrap();
        let mut into = Array2::<f64>::zeros((2, 2));
        matrix_log_svd_from_svd_into(
            &svd.u.view(),
            &svd.singular_values.view(),
            &svd.vt.view(),
            &mut into.view_mut(),
        )
        .unwrap();

        for i in 0..2 {
            for j in 0..2 {
                assert!((direct[[i, j]] - from_factors[[i, j]]).abs() < 1e-12_f64);
                assert!((direct[[i, j]] - into[[i, j]]).abs() < 1e-12_f64);
            }
        }
    }

    #[test]
    fn matrix_log_svd_complex_from_factors_matches_direct() {
        let matrix = Array2::from_shape_vec((2, 2), vec![
            Complex64::new(4.0_f64, 0.0_f64),
            Complex64::new(0.5_f64, 0.25_f64),
            Complex64::new(0.5_f64, -0.25_f64),
            Complex64::new(3.0_f64, 0.0_f64),
        ])
        .unwrap();
        let direct = matrix_log_svd_complex(&matrix).unwrap();
        let svd = svd::decompose_complex(&matrix).unwrap();
        let from_factors =
            matrix_log_svd_complex_from_svd(&svd.u, &svd.singular_values, &svd.vt).unwrap();
        let mut into = Array2::<Complex64>::zeros((2, 2));
        matrix_log_svd_complex_from_svd_into(
            &svd.u.view(),
            &svd.singular_values.view(),
            &svd.vt.view(),
            &mut into.view_mut(),
        )
        .unwrap();

        for i in 0..2 {
            for j in 0..2 {
                assert!((direct[[i, j]] - from_factors[[i, j]]).norm() < 1e-12_f64);
                assert!((direct[[i, j]] - into[[i, j]]).norm() < 1e-12_f64);
            }
        }
    }

    #[test]
    fn real_eigen_factor_paths_match_direct_matrix_paths() {
        let spd = Array2::from_shape_vec((2, 2), vec![4.0_f64, 1.0_f64, 1.0_f64, 3.0_f64]).unwrap();
        let signed =
            Array2::from_shape_vec((2, 2), vec![2.0_f64, 0.0_f64, 0.0_f64, -3.0_f64]).unwrap();
        let eigen = eigen::symmetric(&spd).unwrap();
        let signed_eigen = eigen::symmetric(&signed).unwrap();

        let exp_direct = matrix_exp_eigen(&spd).unwrap();
        let exp_from_factors = matrix_exp_eigen_from_factors_view(
            &eigen.eigenvalues.view(),
            &eigen.eigenvectors.view(),
        )
        .unwrap();
        let mut exp_out = Array2::<f64>::zeros((2, 2));
        matrix_exp_eigen_from_factors_view_into(
            &eigen.eigenvalues.view(),
            &eigen.eigenvectors.view(),
            &mut exp_out.view_mut(),
        )
        .unwrap();

        let log_direct = matrix_log_eigen(&spd).unwrap();
        let log_from_factors = matrix_log_eigen_from_factors_view(
            &eigen.eigenvalues.view(),
            &eigen.eigenvectors.view(),
        )
        .unwrap();
        let mut log_out = Array2::<f64>::zeros((2, 2));
        matrix_log_eigen_from_factors_view_into(
            &eigen.eigenvalues.view(),
            &eigen.eigenvectors.view(),
            &mut log_out.view_mut(),
        )
        .unwrap();

        let power_direct = matrix_power(&spd, 2.0_f64).unwrap();
        let power_from_factors = matrix_power_from_factors_view(
            &eigen.eigenvalues.view(),
            &eigen.eigenvectors.view(),
            2.0_f64,
        )
        .unwrap();
        let mut power_out = Array2::<f64>::zeros((2, 2));
        matrix_power_from_factors_view_into(
            &eigen.eigenvalues.view(),
            &eigen.eigenvectors.view(),
            2.0_f64,
            &mut power_out.view_mut(),
        )
        .unwrap();

        let sign_direct = matrix_sign(&signed).unwrap();
        let sign_from_factors = matrix_sign_from_factors_view(
            &signed_eigen.eigenvalues.view(),
            &signed_eigen.eigenvectors.view(),
        )
        .unwrap();
        let mut sign_out = Array2::<f64>::zeros((2, 2));
        matrix_sign_from_factors_view_into(
            &signed_eigen.eigenvalues.view(),
            &signed_eigen.eigenvectors.view(),
            &mut sign_out.view_mut(),
        )
        .unwrap();

        for i in 0..2 {
            for j in 0..2 {
                assert!((exp_direct[[i, j]] - exp_from_factors[[i, j]]).abs() < 1e-12_f64);
                assert!((exp_direct[[i, j]] - exp_out[[i, j]]).abs() < 1e-12_f64);
                assert!((log_direct[[i, j]] - log_from_factors[[i, j]]).abs() < 1e-12_f64);
                assert!((log_direct[[i, j]] - log_out[[i, j]]).abs() < 1e-12_f64);
                assert!((power_direct[[i, j]] - power_from_factors[[i, j]]).abs() < 1e-12_f64);
                assert!((power_direct[[i, j]] - power_out[[i, j]]).abs() < 1e-12_f64);
                assert!((sign_direct[[i, j]] - sign_from_factors[[i, j]]).abs() < 1e-12_f64);
                assert!((sign_direct[[i, j]] - sign_out[[i, j]]).abs() < 1e-12_f64);
            }
        }
    }
}
