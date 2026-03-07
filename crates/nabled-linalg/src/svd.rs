//! Singular value decomposition over ndarray matrices.

#[cfg(any(feature = "magma-system", not(feature = "lapack-provider")))]
use std::cmp::Ordering;
use std::fmt;

use nabled_core::scalar::NabledReal;
use ndarray::{Array1, Array2, ArrayView2, s};
use num_complex::Complex64;

use crate::internal::{DenseKernelPolicy, jacobi_eigen_symmetric, sort_eigenpairs_desc};
#[cfg(feature = "magma-system")]
use crate::provider::magma;

#[cfg(feature = "magma-system")]
#[doc(hidden)]
pub trait SvdInternalScalar: NabledReal + magma::MagmaReal {}

#[cfg(feature = "magma-system")]
impl<T> SvdInternalScalar for T where T: NabledReal + magma::MagmaReal {}

#[cfg(not(feature = "magma-system"))]
#[doc(hidden)]
pub trait SvdInternalScalar: NabledReal {}

#[cfg(not(feature = "magma-system"))]
impl<T> SvdInternalScalar for T where T: NabledReal {}
#[cfg(any(feature = "magma-system", not(feature = "lapack-provider")))]
use crate::schur;

/// SVD result for ndarray matrices.
#[derive(Debug, Clone)]
pub struct NdarraySVD<T: NabledReal> {
    /// Left singular vectors (`m x k`).
    pub u:               Array2<T>,
    /// Singular values (`k`).
    pub singular_values: Array1<T>,
    /// Right singular vectors transposed (`k x n`).
    pub vt:              Array2<T>,
}

/// Complex SVD result for ndarray matrices.
#[derive(Debug, Clone)]
pub struct NdarrayComplexSVD {
    /// Left singular vectors (`m x k`).
    pub u:               Array2<Complex64>,
    /// Singular values (`k`).
    pub singular_values: Array1<f64>,
    /// Right singular vectors transposed (`k x n`).
    pub vt:              Array2<Complex64>,
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

#[cfg(feature = "magma-system")]
fn map_svd_magma_error(error: &'static str) -> SVDError {
    match error {
        "empty" => SVDError::EmptyMatrix,
        "convergence_failed" => SVDError::ConvergenceFailed,
        "non_finite" => SVDError::InvalidInput("matrix must be finite".to_string()),
        _ => SVDError::InvalidInput(error.to_string()),
    }
}

/// Configuration for pseudo-inverse computation.
#[derive(Debug, Clone, Copy, Default)]
pub struct PseudoInverseConfig<T: NabledReal> {
    /// Tolerance for truncating tiny singular values.
    pub tolerance: Option<T>,
}

#[cfg(not(feature = "lapack-provider"))]
fn decompose_internal_fallback<T: NabledReal>(
    matrix: &ArrayView2<'_, T>,
) -> Result<NdarraySVD<T>, SVDError> {
    if matrix.is_empty() {
        return Err(SVDError::EmptyMatrix);
    }
    if matrix.iter().any(|value| !value.is_finite()) {
        return Err(SVDError::InvalidInput("matrix must be finite".into()));
    }
    let (rows, cols) = matrix.dim();
    let k = rows.min(cols);

    let ata = matrix.t().dot(matrix);
    let (eigenvalues, eigenvectors) = jacobi_eigen_symmetric(
        &ata,
        T::from_f64(DenseKernelPolicy::BASE_TOLERANCE).unwrap_or(T::epsilon()),
        DenseKernelPolicy::JACOBI_MAX_ITERATIONS,
    )
    .map_err(|_| SVDError::ConvergenceFailed)?;
    let (sorted_values, sorted_vectors) = sort_eigenpairs_desc(&eigenvalues, &eigenvectors);

    let mut singular_values = Array1::<T>::zeros(k);
    let mut vt = Array2::<T>::zeros((k, cols));
    for i in 0..k {
        let value = sorted_values[i].max(T::zero()).sqrt();
        singular_values[i] = value;
        for j in 0..cols {
            vt[[i, j]] = sorted_vectors[[j, i]];
        }
    }

    let mut u = Array2::<T>::zeros((rows, k));
    let tolerance = T::from_f64(DenseKernelPolicy::BASE_TOLERANCE).unwrap_or(T::epsilon());
    for i in 0..k {
        let sigma = singular_values[i];
        if sigma > tolerance {
            let av = matrix.dot(&sorted_vectors.column(i));
            for row in 0..rows {
                u[[row, i]] = av[row] / sigma;
            }
        }
    }

    Ok(NdarraySVD { u, singular_values, vt })
}

#[cfg(all(not(feature = "lapack-provider"), feature = "magma-system"))]
fn decompose_internal<T: NabledReal + magma::MagmaReal>(
    matrix: &ArrayView2<'_, T>,
) -> Result<NdarraySVD<T>, SVDError> {
    if DenseKernelPolicy::prefer_magma_decomposition(matrix.nrows(), matrix.ncols()) {
        match magma::svd_decompose(matrix) {
            Ok((u, singular_values, vt)) => {
                return Ok(NdarraySVD { u, singular_values, vt });
            }
            Err(error) => {
                if DenseKernelPolicy::magma_strict_mode() {
                    return Err(map_svd_magma_error(error));
                }
            }
        }
    }

    decompose_internal_fallback(matrix)
}

#[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
fn decompose_internal<T: NabledReal>(
    matrix: &ArrayView2<'_, T>,
) -> Result<NdarraySVD<T>, SVDError> {
    decompose_internal_fallback(matrix)
}

#[cfg(feature = "lapack-provider")]
fn decompose_provider<T>(matrix: &ArrayView2<'_, T>) -> Result<NdarraySVD<T>, SVDError>
where
    T: NabledReal + ndarray_linalg::Lapack<Real = T> + std::ops::AddAssign,
{
    use ndarray_linalg::SVD as _;

    if matrix.is_empty() {
        return Err(SVDError::EmptyMatrix);
    }

    let (u_opt, singular_values, vt_opt) =
        matrix.to_owned().svd(true, true).map_err(|_| SVDError::ConvergenceFailed)?;
    let u = u_opt.ok_or(SVDError::ConvergenceFailed)?;
    let vt = vt_opt.ok_or(SVDError::ConvergenceFailed)?;

    Ok(NdarraySVD { u, singular_values, vt })
}

#[cfg(any(feature = "magma-system", not(feature = "lapack-provider")))]
fn validate_complex_finite(matrix: &ArrayView2<'_, Complex64>) -> Result<(), SVDError> {
    if matrix.iter().any(|value| !value.re.is_finite() || !value.im.is_finite()) {
        return Err(SVDError::InvalidInput("matrix must be finite".to_string()));
    }
    Ok(())
}

#[cfg(any(feature = "magma-system", not(feature = "lapack-provider")))]
#[allow(clippy::many_single_char_names)]
fn decompose_complex_internal(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<NdarrayComplexSVD, SVDError> {
    validate_complex_finite(matrix)?;

    let (rows, cols) = matrix.dim();
    let keep = rows.min(cols);

    let conjugate_transpose = matrix.t().mapv(|value| value.conj());
    let gram = conjugate_transpose.dot(matrix);
    let schur = schur::compute_schur_complex(&gram).map_err(|_| SVDError::ConvergenceFailed)?;

    let mut singular_pairs = (0..cols)
        .map(|index| {
            let eigenvalue = schur.t[[index, index]].re.max(0.0);
            (eigenvalue.sqrt(), index)
        })
        .collect::<Vec<_>>();
    singular_pairs.sort_by(|(lhs, _), (rhs, _)| rhs.partial_cmp(lhs).unwrap_or(Ordering::Equal));

    let mut singular_values = Array1::<f64>::zeros(keep);
    let mut vt = Array2::<Complex64>::zeros((keep, cols));
    let mut u = Array2::<Complex64>::zeros((rows, keep));

    for out in 0..keep {
        let (sigma, in_col) = singular_pairs[out];
        singular_values[out] = sigma;

        for j in 0..cols {
            vt[[out, j]] = schur.q[[j, in_col]].conj();
        }

        if sigma > DenseKernelPolicy::BASE_TOLERANCE {
            let av = matrix.dot(&schur.q.column(in_col));
            let scale = 1.0_f64 / sigma;
            for i in 0..rows {
                u[[i, out]] = av[i] * scale;
            }
        }
    }

    Ok(NdarrayComplexSVD { u, singular_values, vt })
}

#[cfg(feature = "magma-system")]
fn decompose_complex_provider(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<NdarrayComplexSVD, SVDError> {
    if matrix.is_empty() {
        return Err(SVDError::EmptyMatrix);
    }
    if !DenseKernelPolicy::prefer_magma_decomposition(matrix.nrows(), matrix.ncols()) {
        return decompose_complex_internal(matrix);
    }
    match magma::svd_decompose_complex(matrix) {
        Ok((u, singular_values, vt)) => Ok(NdarrayComplexSVD { u, singular_values, vt }),
        Err(error) => {
            if DenseKernelPolicy::magma_strict_mode() {
                return Err(map_svd_magma_error(error));
            }
            decompose_complex_internal(matrix)
        }
    }
}

fn svd_relative_tolerance<T: NabledReal>(max_sv: T, dimension: usize) -> T {
    let dimension_fallback = T::from_u32(u32::MAX).unwrap_or(T::one());
    let dimension_as_t = T::from_usize(dimension).unwrap_or(dimension_fallback);
    let base_tolerance = T::from_f64(DenseKernelPolicy::BASE_TOLERANCE).unwrap_or(T::epsilon());
    max_sv * dimension_as_t * T::epsilon().max(base_tolerance)
}

fn rank_estimation_tolerance<T: NabledReal>(max_sv: T, dimension: usize) -> T {
    let dimension_fallback = T::from_u32(u32::MAX).unwrap_or(T::one());
    let dimension_as_t = T::from_usize(dimension).unwrap_or(dimension_fallback);
    max_sv * dimension_as_t * T::epsilon()
}

fn null_space_internal<T: NabledReal>(
    matrix: &Array2<T>,
    tolerance: Option<T>,
) -> Result<Array2<T>, SVDError> {
    if matrix.is_empty() {
        return Err(SVDError::EmptyMatrix);
    }

    let ata = matrix.t().dot(matrix);
    let (eigenvalues, eigenvectors) = jacobi_eigen_symmetric(
        &ata,
        T::from_f64(DenseKernelPolicy::BASE_TOLERANCE).unwrap_or(T::epsilon()),
        DenseKernelPolicy::JACOBI_MAX_ITERATIONS,
    )
    .map_err(|_| SVDError::ConvergenceFailed)?;
    let (sorted_values, sorted_vectors) = sort_eigenpairs_desc(&eigenvalues, &eigenvectors);

    let max_sv = sorted_values
        .iter()
        .copied()
        .map(|value| value.max(T::zero()).sqrt())
        .fold(T::zero(), T::max);
    let tol = tolerance.unwrap_or(svd_relative_tolerance(max_sv, matrix.ncols()));

    let mut null_indices = Vec::new();
    for (index, value) in sorted_values.iter().copied().enumerate() {
        let singular = value.max(T::zero()).sqrt();
        if singular <= tol {
            null_indices.push(index);
        }
    }

    if null_indices.is_empty() {
        return Ok(Array2::<T>::zeros((matrix.ncols(), 0)));
    }

    let mut basis = Array2::<T>::zeros((matrix.ncols(), null_indices.len()));
    for (col_out, col_in) in null_indices.into_iter().enumerate() {
        for row in 0..matrix.ncols() {
            basis[[row, col_out]] = sorted_vectors[[row, col_in]];
        }
    }

    Ok(basis)
}

/// Compute the SVD of `matrix`.
///
/// # Errors
/// Returns an error if the matrix is empty, non-finite, or decomposition fails.
#[cfg(feature = "lapack-provider")]
pub fn decompose<T>(matrix: &Array2<T>) -> Result<NdarraySVD<T>, SVDError>
where
    T: NabledReal + ndarray_linalg::Lapack<Real = T> + std::ops::AddAssign,
{
    decompose_impl(&matrix.view())
}

/// Compute the SVD of `matrix`.
///
/// # Errors
/// Returns an error if the matrix is empty, non-finite, or decomposition fails.
#[cfg(not(feature = "lapack-provider"))]
pub fn decompose<T: SvdInternalScalar>(matrix: &Array2<T>) -> Result<NdarraySVD<T>, SVDError> {
    decompose_impl(&matrix.view())
}

#[cfg(feature = "lapack-provider")]
fn decompose_impl<T>(matrix: &ArrayView2<'_, T>) -> Result<NdarraySVD<T>, SVDError>
where
    T: NabledReal + ndarray_linalg::Lapack<Real = T>,
{
    decompose_provider(matrix)
}

#[cfg(not(feature = "lapack-provider"))]
fn decompose_impl<T: SvdInternalScalar>(
    matrix: &ArrayView2<'_, T>,
) -> Result<NdarraySVD<T>, SVDError> {
    decompose_internal(matrix)
}

/// Compute the SVD of `matrix` from a matrix view.
///
/// # Errors
/// Returns an error if decomposition fails.
#[cfg(feature = "lapack-provider")]
pub fn decompose_view<T>(matrix: &ArrayView2<'_, T>) -> Result<NdarraySVD<T>, SVDError>
where
    T: NabledReal + ndarray_linalg::Lapack<Real = T>,
{
    decompose_impl(matrix)
}

/// Compute the SVD of `matrix` from a matrix view.
///
/// # Errors
/// Returns an error if decomposition fails.
#[cfg(not(feature = "lapack-provider"))]
pub fn decompose_view<T: SvdInternalScalar>(
    matrix: &ArrayView2<'_, T>,
) -> Result<NdarraySVD<T>, SVDError> {
    decompose_impl(matrix)
}

/// Compute the SVD of a complex matrix.
///
/// # Errors
/// Returns an error if decomposition fails.
pub fn decompose_complex(matrix: &Array2<Complex64>) -> Result<NdarrayComplexSVD, SVDError> {
    decompose_complex_impl(&matrix.view())
}

fn decompose_complex_impl(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<NdarrayComplexSVD, SVDError> {
    if matrix.is_empty() {
        return Err(SVDError::EmptyMatrix);
    }

    #[cfg(feature = "magma-system")]
    {
        decompose_complex_provider(matrix)
    }
    #[cfg(all(feature = "lapack-provider", not(feature = "magma-system")))]
    {
        use ndarray_linalg::SVD as _;
        let (u_opt, singular_values, vt_opt) =
            matrix.to_owned().svd(true, true).map_err(|_| SVDError::ConvergenceFailed)?;
        let u = u_opt.ok_or(SVDError::ConvergenceFailed)?;
        let vt = vt_opt.ok_or(SVDError::ConvergenceFailed)?;
        Ok(NdarrayComplexSVD { u, singular_values, vt })
    }
    #[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
    {
        decompose_complex_internal(matrix)
    }
}

/// Compute complex SVD from a matrix view.
///
/// # Errors
/// Returns an error if decomposition fails.
pub fn decompose_complex_view(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<NdarrayComplexSVD, SVDError> {
    decompose_complex_impl(matrix)
}

/// Compute SVD and zero out singular values below `tolerance`.
///
/// # Errors
/// Returns an error if decomposition fails.
#[cfg(feature = "lapack-provider")]
pub fn decompose_with_tolerance<T>(
    matrix: &Array2<T>,
    tolerance: T,
) -> Result<NdarraySVD<T>, SVDError>
where
    T: NabledReal + ndarray_linalg::Lapack<Real = T>,
{
    let mut svd = decompose(matrix)?;
    for value in &mut svd.singular_values {
        if *value < tolerance {
            *value = T::zero();
        }
    }
    Ok(svd)
}

/// Compute SVD and zero out singular values below `tolerance`.
///
/// # Errors
/// Returns an error if decomposition fails.
#[cfg(not(feature = "lapack-provider"))]
pub fn decompose_with_tolerance<T: SvdInternalScalar>(
    matrix: &Array2<T>,
    tolerance: T,
) -> Result<NdarraySVD<T>, SVDError> {
    let mut svd = decompose(matrix)?;
    for value in &mut svd.singular_values {
        if *value < tolerance {
            *value = T::zero();
        }
    }
    Ok(svd)
}

/// Compute truncated SVD by keeping only the `k` largest singular values.
///
/// # Errors
/// Returns an error if `k == 0` or decomposition fails.
#[cfg(feature = "lapack-provider")]
pub fn decompose_truncated<T>(matrix: &Array2<T>, k: usize) -> Result<NdarraySVD<T>, SVDError>
where
    T: NabledReal + ndarray_linalg::Lapack<Real = T>,
{
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

/// Compute truncated SVD by keeping only the `k` largest singular values.
///
/// # Errors
/// Returns an error if `k == 0` or decomposition fails.
#[cfg(not(feature = "lapack-provider"))]
pub fn decompose_truncated<T: SvdInternalScalar>(
    matrix: &Array2<T>,
    k: usize,
) -> Result<NdarraySVD<T>, SVDError> {
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
pub fn reconstruct_matrix<T: NabledReal>(svd: &NdarraySVD<T>) -> Array2<T> {
    let cols = svd.vt.ncols();
    let mut sigma_vt = svd.vt.clone();
    for i in 0..svd.singular_values.len().min(svd.u.ncols()) {
        for j in 0..cols {
            sigma_vt[[i, j]] *= svd.singular_values[i];
        }
    }
    svd.u.dot(&sigma_vt)
}

/// Reconstruct a complex matrix from SVD components.
#[must_use]
pub fn reconstruct_matrix_complex(svd: &NdarrayComplexSVD) -> Array2<Complex64> {
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
pub fn reconstruct_matrix_into<T: NabledReal>(
    svd: &NdarraySVD<T>,
    output: &mut Array2<T>,
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

    output.fill(T::zero());
    for i in 0..rows {
        for j in 0..cols {
            let mut sum = T::zero();
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
pub fn condition_number<T: NabledReal>(svd: &NdarraySVD<T>) -> T {
    if svd.singular_values.is_empty() {
        return T::zero();
    }

    let max_sv = svd.singular_values.iter().copied().fold(T::zero(), T::max);
    let tolerance = T::from_f64(DenseKernelPolicy::BASE_TOLERANCE).unwrap_or(T::epsilon());
    let min_sv = svd
        .singular_values
        .iter()
        .copied()
        .filter(|value| *value > tolerance)
        .fold(T::infinity(), T::min);

    if min_sv.is_finite() { max_sv / min_sv } else { T::infinity() }
}

/// Estimate numerical rank from singular values.
#[must_use]
pub fn rank<T: NabledReal>(svd: &NdarraySVD<T>, tolerance: Option<T>) -> usize {
    let max_sv = svd.singular_values.iter().copied().fold(T::zero(), T::max);
    let tol = tolerance.unwrap_or(rank_estimation_tolerance(max_sv, svd.singular_values.len()));
    svd.singular_values.iter().filter(|value| **value > tol).count()
}

/// Compute Moore-Penrose pseudo-inverse.
///
/// # Errors
/// Returns an error if input is invalid or decomposition fails.
#[cfg(feature = "lapack-provider")]
pub fn pseudo_inverse<T>(
    matrix: &Array2<T>,
    config: &PseudoInverseConfig<T>,
) -> Result<Array2<T>, SVDError>
where
    T: NabledReal + ndarray_linalg::Lapack<Real = T>,
{
    if matrix.is_empty() {
        return Err(SVDError::EmptyMatrix);
    }

    let mut output = Array2::<T>::zeros((matrix.ncols(), matrix.nrows()));
    pseudo_inverse_into(matrix, config, &mut output)?;
    Ok(output)
}

/// Compute Moore-Penrose pseudo-inverse.
///
/// # Errors
/// Returns an error if input is invalid or decomposition fails.
#[cfg(not(feature = "lapack-provider"))]
pub fn pseudo_inverse<T: SvdInternalScalar>(
    matrix: &Array2<T>,
    config: &PseudoInverseConfig<T>,
) -> Result<Array2<T>, SVDError> {
    if matrix.is_empty() {
        return Err(SVDError::EmptyMatrix);
    }

    let mut output = Array2::<T>::zeros((matrix.ncols(), matrix.nrows()));
    pseudo_inverse_into(matrix, config, &mut output)?;
    Ok(output)
}

/// Compute Moore-Penrose pseudo-inverse into `output`.
///
/// # Errors
/// Returns an error if dimensions are invalid or decomposition fails.
#[cfg(feature = "lapack-provider")]
pub fn pseudo_inverse_into<T>(
    matrix: &Array2<T>,
    config: &PseudoInverseConfig<T>,
    output: &mut Array2<T>,
) -> Result<(), SVDError>
where
    T: NabledReal + ndarray_linalg::Lapack<Real = T>,
{
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
    let max_sv = svd.singular_values.iter().copied().fold(T::zero(), T::max);
    let tolerance = config.tolerance.unwrap_or(svd_relative_tolerance(max_sv, rows.max(cols)));

    output.fill(T::zero());
    let k = svd.singular_values.len();
    for i in 0..k {
        let sigma = svd.singular_values[i];
        if sigma <= tolerance {
            continue;
        }
        let inv_sigma = T::one() / sigma;
        for row in 0..cols {
            for col in 0..rows {
                output[[row, col]] += svd.vt[[i, row]] * inv_sigma * svd.u[[col, i]];
            }
        }
    }

    Ok(())
}

/// Compute Moore-Penrose pseudo-inverse into `output`.
///
/// # Errors
/// Returns an error if dimensions are invalid or decomposition fails.
#[cfg(not(feature = "lapack-provider"))]
pub fn pseudo_inverse_into<T: SvdInternalScalar>(
    matrix: &Array2<T>,
    config: &PseudoInverseConfig<T>,
    output: &mut Array2<T>,
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
    let max_sv = svd.singular_values.iter().copied().fold(T::zero(), T::max);
    let tolerance = config.tolerance.unwrap_or(svd_relative_tolerance(max_sv, rows.max(cols)));

    output.fill(T::zero());
    let k = svd.singular_values.len();
    for i in 0..k {
        let sigma = svd.singular_values[i];
        if sigma <= tolerance {
            continue;
        }
        let inv_sigma = T::one() / sigma;
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
pub fn null_space<T: NabledReal>(
    matrix: &Array2<T>,
    tolerance: Option<T>,
) -> Result<Array2<T>, SVDError> {
    null_space_internal(matrix, tolerance)
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;
    use num_complex::Complex64;

    use super::*;

    #[test]
    fn svd_reconstructs_small_matrix() {
        let matrix =
            Array2::<f64>::from_shape_vec((2, 2), vec![1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64])
                .unwrap();
        let svd = decompose(&matrix).unwrap();
        let reconstructed = reconstruct_matrix(&svd);
        for i in 0..2 {
            for j in 0..2 {
                assert!((matrix[[i, j]] - reconstructed[[i, j]]).abs() < 1e-8_f64);
            }
        }
    }

    #[test]
    fn truncated_svd_requires_positive_rank() {
        let matrix = Array2::<f64>::eye(2);
        let result = decompose_truncated(&matrix, 0);
        assert!(matches!(result, Err(SVDError::InvalidInput(_))));
    }

    #[test]
    fn pseudo_inverse_matches_identity_for_diagonal() {
        let matrix =
            Array2::<f64>::from_shape_vec((2, 2), vec![1.0_f64, 0.0_f64, 0.0_f64, 2.0_f64])
                .unwrap();
        let pinv = pseudo_inverse(&matrix, &PseudoInverseConfig::<f64>::default()).unwrap();
        let product = matrix.dot(&pinv);
        assert!((product[[0, 0]] - 1.0_f64).abs() < 1e-8_f64);
        assert!((product[[1, 1]] - 1.0_f64).abs() < 1e-8_f64);
    }

    #[test]
    fn null_space_detects_rank_deficiency() {
        let matrix =
            Array2::<f64>::from_shape_vec((2, 2), vec![1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64])
                .unwrap();
        let basis = null_space(&matrix, Some(1e-10_f64)).unwrap();
        assert_eq!(basis.ncols(), 1);
        let residual = matrix.dot(&basis.column(0).to_owned());
        assert!(residual.iter().all(|value| value.abs() < 1e-6_f64));
    }

    #[test]
    fn decompose_view_matches_owned() {
        let matrix =
            Array2::<f64>::from_shape_vec((2, 2), vec![3.0_f64, 1.0_f64, 1.0_f64, 3.0_f64])
                .unwrap();
        let from_owned = decompose(&matrix).unwrap();
        let matrix_view = matrix.view();
        let from_view = decompose_view(&matrix_view).unwrap();
        assert_eq!(from_owned.singular_values.len(), from_view.singular_values.len());
    }

    #[test]
    fn complex_svd_reconstructs_input() {
        let matrix = Array2::from_shape_vec((2, 2), vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(0.5, 0.25),
            Complex64::new(-1.0, 2.0),
        ])
        .unwrap();
        let svd = decompose_complex(&matrix).unwrap();
        let reconstructed = reconstruct_matrix_complex(&svd);
        for i in 0..2 {
            for j in 0..2 {
                assert!((reconstructed[[i, j]] - matrix[[i, j]]).norm() < 1e-8);
            }
        }
    }

    #[test]
    fn tolerance_rank_and_condition_number_paths() {
        let matrix =
            Array2::<f64>::from_shape_vec((2, 2), vec![3.0_f64, 0.0_f64, 0.0_f64, 1.0_f64])
                .unwrap();
        let svd = decompose_with_tolerance(&matrix, 2.0_f64).unwrap();
        assert!(svd.singular_values[1].abs() < 1e-12_f64);
        assert!(condition_number(&svd).is_finite());
        assert_eq!(rank(&svd, Some(1e-8_f64)), 1);
    }

    #[test]
    fn reconstruct_into_and_pseudo_inverse_into_paths() {
        let matrix =
            Array2::<f64>::from_shape_vec((2, 2), vec![2.0_f64, 0.0_f64, 0.0_f64, 4.0_f64])
                .unwrap();
        let svd = decompose(&matrix).unwrap();
        let mut reconstructed = Array2::<f64>::zeros((2, 2));
        reconstruct_matrix_into(&svd, &mut reconstructed).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!((reconstructed[[i, j]] - matrix[[i, j]]).abs() < 1e-8_f64);
            }
        }

        let mut pinv = Array2::<f64>::zeros((2, 2));
        pseudo_inverse_into(&matrix, &PseudoInverseConfig::<f64>::default(), &mut pinv).unwrap();
        let identity = matrix.dot(&pinv);
        assert!((identity[[0, 0]] - 1.0_f64).abs() < 1e-8_f64);
        assert!((identity[[1, 1]] - 1.0_f64).abs() < 1e-8_f64);
    }

    #[test]
    fn reconstruct_into_rejects_bad_output_shape() {
        let matrix = Array2::<f64>::eye(2);
        let svd = decompose(&matrix).unwrap();
        let mut bad = Array2::<f64>::zeros((1, 1));
        let result = reconstruct_matrix_into(&svd, &mut bad);
        assert!(matches!(result, Err(SVDError::InvalidInput(_))));
    }

    #[test]
    fn decompose_rejects_empty_input() {
        let empty = Array2::<f64>::zeros((0, 0));
        assert!(matches!(decompose(&empty), Err(SVDError::EmptyMatrix)));
    }

    #[cfg(not(feature = "lapack-provider"))]
    #[test]
    fn internal_decompose_rejects_non_finite_input() {
        let non_finite = Array2::from_shape_vec((2, 2), vec![1.0, f64::NAN, 0.0, 1.0]).unwrap();
        assert!(matches!(decompose(&non_finite), Err(SVDError::InvalidInput(_))));
    }

    #[test]
    fn pseudo_inverse_into_rejects_empty_and_bad_output_shape() {
        let empty = Array2::<f64>::zeros((0, 0));
        let mut output = Array2::<f64>::zeros((0, 0));
        assert!(matches!(
            pseudo_inverse_into(&empty, &PseudoInverseConfig::<f64>::default(), &mut output),
            Err(SVDError::EmptyMatrix)
        ));

        let matrix = Array2::<f64>::eye(2);
        let mut bad = Array2::<f64>::zeros((1, 2));
        assert!(matches!(
            pseudo_inverse_into(&matrix, &PseudoInverseConfig::<f64>::default(), &mut bad),
            Err(SVDError::InvalidInput(_))
        ));
    }

    #[test]
    fn null_space_of_full_rank_matrix_is_empty() {
        let matrix = Array2::<f64>::eye(3);
        let basis = null_space(&matrix, None).unwrap();
        assert_eq!(basis.ncols(), 0);
        assert_eq!(basis.nrows(), 3);
    }

    #[test]
    fn real_f32_paths_match_expected() {
        let matrix =
            Array2::from_shape_vec((2, 2), vec![3.0_f32, 1.0_f32, 1.0_f32, 3.0_f32]).unwrap();
        let svd = decompose(&matrix).unwrap();
        let reconstructed = reconstruct_matrix(&svd);
        for i in 0..2 {
            for j in 0..2 {
                assert!((reconstructed[[i, j]] - matrix[[i, j]]).abs() < 1e-4_f32);
            }
        }

        let truncated = decompose_truncated(&matrix, 1).unwrap();
        assert_eq!(truncated.singular_values.len(), 1);

        let tol = decompose_with_tolerance(&matrix, 2.1_f32).unwrap();
        assert!(tol.singular_values[1].abs() < 1e-4_f32);
        assert_eq!(rank(&tol, Some(1.0e-3_f32)), 1);
        assert!(condition_number(&tol).is_finite());

        let pinv = pseudo_inverse(&matrix, &PseudoInverseConfig::<f32>::default())
            .expect("pseudo inverse");
        let product = matrix.dot(&pinv);
        assert!((product[[0, 0]] - 1.0_f32).abs() < 1e-3_f32);
        assert!((product[[1, 1]] - 1.0_f32).abs() < 1e-3_f32);

        let rank_deficient =
            Array2::from_shape_vec((2, 2), vec![1.0_f32, 1.0_f32, 1.0_f32, 1.0_f32]).unwrap();
        let basis = null_space(&rank_deficient, Some(1.0e-5_f32)).unwrap();
        assert_eq!(basis.ncols(), 1);
    }
}
