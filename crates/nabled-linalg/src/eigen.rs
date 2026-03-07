//! Eigenvalue decompositions over ndarray matrices.

use std::fmt;

use nabled_core::scalar::NabledReal;
use ndarray::{Array1, Array2, ArrayView2};
use num_complex::{Complex, Complex64};

#[cfg(not(feature = "lapack-provider"))]
use crate::internal::jacobi_eigen_symmetric;
use crate::internal::{DenseKernelPolicy, sort_eigenpairs_desc};
#[cfg(feature = "magma-system")]
use crate::provider::magma;
#[cfg(not(feature = "lapack-provider"))]
use crate::qr;
use crate::{cholesky, schur};

#[cfg(feature = "magma-system")]
#[doc(hidden)]
pub trait EigenInternalScalar: NabledReal + magma::MagmaReal {}

#[cfg(feature = "magma-system")]
impl<T> EigenInternalScalar for T where T: NabledReal + magma::MagmaReal {}

#[cfg(not(feature = "magma-system"))]
#[doc(hidden)]
pub trait EigenInternalScalar: NabledReal {}

#[cfg(not(feature = "magma-system"))]
impl<T> EigenInternalScalar for T where T: NabledReal {}

/// Result of symmetric eigen decomposition.
#[derive(Debug, Clone)]
pub struct NdarrayEigenResult<T: NabledReal = f64> {
    /// Eigenvalues.
    pub eigenvalues:  Array1<T>,
    /// Eigenvectors by column.
    pub eigenvectors: Array2<T>,
}

/// Result of generalized eigen decomposition.
#[derive(Debug, Clone)]
pub struct NdarrayGeneralizedEigenResult<T: NabledReal = f64> {
    /// Eigenvalues.
    pub eigenvalues:  Array1<T>,
    /// Eigenvectors by column.
    pub eigenvectors: Array2<T>,
}

/// Result of non-symmetric eigen decomposition (Schur-based).
#[derive(Debug, Clone)]
pub struct NdarrayNonsymmetricEigenResult<T: NabledReal = f64> {
    /// Eigenvalues.
    pub eigenvalues:   Array1<Complex<T>>,
    /// Schur vectors by column.
    pub schur_vectors: Array2<Complex<T>>,
}

/// Result of non-symmetric eigen decomposition with matched left/right eigenvectors.
#[derive(Debug, Clone)]
pub struct NdarrayNonsymmetricBiEigenResult<T: NabledReal = f64> {
    /// Eigenvalues.
    pub eigenvalues:        Array1<Complex<T>>,
    /// Right eigenvectors (by column) for the original unbalanced matrix.
    pub right_eigenvectors: Array2<Complex<T>>,
    /// Left eigenvectors (by column) for the original unbalanced matrix.
    pub left_eigenvectors:  Array2<Complex<T>>,
    /// Balancing diagonal scales used before decomposition.
    pub balancing_diagonal: Array1<T>,
    /// Matrix used for eigendecomposition after optional balancing.
    pub balanced_matrix:    Array2<T>,
}

/// Configuration for non-symmetric balancing and bi-eigen decomposition.
#[derive(Debug, Clone, Copy)]
pub struct NonsymmetricEigenConfig<T: NabledReal = f64> {
    /// Whether to apply Osborne balancing before eigendecomposition.
    pub balance:                bool,
    /// Maximum balancing sweeps.
    pub balance_max_iterations: usize,
    /// Relative improvement threshold for applying a balance update.
    pub balance_tolerance:      T,
}

impl<T: NabledReal> Default for NonsymmetricEigenConfig<T> {
    fn default() -> Self {
        Self {
            balance:                true,
            balance_max_iterations: 32,
            balance_tolerance:      T::from_f64(0.05).unwrap_or(T::epsilon()),
        }
    }
}

/// Error type for eigen operations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EigenError {
    /// Matrix is empty.
    EmptyMatrix,
    /// Matrix is not square.
    NotSquare,
    /// Matrix is not symmetric when required.
    NotSymmetric,
    /// Dimensions are incompatible.
    InvalidDimensions,
    /// Matrix is not positive definite.
    NotPositiveDefinite,
    /// Convergence failure.
    ConvergenceFailed,
    /// Numerical instability.
    NumericalInstability,
}

impl fmt::Display for EigenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EigenError::EmptyMatrix => write!(f, "Matrix cannot be empty"),
            EigenError::NotSquare => write!(f, "Matrix must be square"),
            EigenError::NotSymmetric => write!(f, "Matrix must be symmetric"),
            EigenError::InvalidDimensions => write!(f, "Matrix dimensions are incompatible"),
            EigenError::NotPositiveDefinite => write!(f, "Matrix is not positive definite"),
            EigenError::ConvergenceFailed => write!(f, "Eigen solver failed to converge"),
            EigenError::NumericalInstability => write!(f, "Numerical instability detected"),
        }
    }
}

impl std::error::Error for EigenError {}

fn validate_symmetric_input<T: NabledReal>(matrix: &ArrayView2<'_, T>) -> Result<(), EigenError> {
    if matrix.is_empty() {
        return Err(EigenError::EmptyMatrix);
    }
    if matrix.nrows() != matrix.ncols() {
        return Err(EigenError::NotSquare);
    }
    if matrix.iter().any(|value| !value.is_finite()) {
        return Err(EigenError::NumericalInstability);
    }
    let tolerance = T::from_f64(DenseKernelPolicy::BASE_TOLERANCE).unwrap_or(T::epsilon());
    let n = matrix.nrows();
    for i in 0..n {
        for j in (i + 1)..n {
            if (matrix[[i, j]] - matrix[[j, i]]).abs() > tolerance {
                return Err(EigenError::NotSymmetric);
            }
        }
    }
    Ok(())
}

fn validate_nonsymmetric_input<T: NabledReal>(
    matrix: &ArrayView2<'_, T>,
) -> Result<(), EigenError> {
    if matrix.is_empty() {
        return Err(EigenError::EmptyMatrix);
    }
    if matrix.nrows() != matrix.ncols() {
        return Err(EigenError::NotSquare);
    }
    if matrix.iter().any(|value| !value.is_finite()) {
        return Err(EigenError::NumericalInstability);
    }
    Ok(())
}

fn validate_complex_square_finite(matrix: &ArrayView2<'_, Complex64>) -> Result<(), EigenError> {
    if matrix.is_empty() {
        return Err(EigenError::EmptyMatrix);
    }
    if matrix.nrows() != matrix.ncols() {
        return Err(EigenError::NotSquare);
    }
    if matrix.iter().any(|value| !value.re.is_finite() || !value.im.is_finite()) {
        return Err(EigenError::NumericalInstability);
    }
    Ok(())
}

fn nonsymmetric_small_matrix<T: NabledReal>(
    matrix: &ArrayView2<'_, T>,
) -> Option<NdarrayNonsymmetricEigenResult<T>> {
    let dimension = matrix.nrows();
    if dimension == 1 {
        let mut eigenvalues = Array1::<Complex<T>>::zeros(1);
        eigenvalues[0] = Complex::new(matrix[[0, 0]], T::zero());
        let mut schur_vectors = Array2::<Complex<T>>::zeros((1, 1));
        schur_vectors[[0, 0]] = Complex::new(T::one(), T::zero());
        return Some(NdarrayNonsymmetricEigenResult { eigenvalues, schur_vectors });
    }
    if dimension != 2 {
        return None;
    }

    let m00 = matrix[[0, 0]];
    let m01 = matrix[[0, 1]];
    let m10 = matrix[[1, 0]];
    let m11 = matrix[[1, 1]];

    let trace = m00 + m11;
    let determinant = m00 * m11 - m01 * m10;
    let two = T::one() + T::one();
    let four = two + two;
    let discriminant = Complex::new(trace * trace - four * determinant, T::zero()).sqrt();
    let trace_complex = Complex::new(trace, T::zero());

    let mut eigenvalues = Array1::<Complex<T>>::zeros(2);
    eigenvalues[0] = (trace_complex + discriminant) / two;
    eigenvalues[1] = (trace_complex - discriminant) / two;

    let mut schur_vectors = Array2::<Complex<T>>::zeros((2, 2));
    for col in 0..2 {
        let lambda = eigenvalues[col];
        let mut candidate = if m01.abs() >= m10.abs() {
            [Complex::new(m01, T::zero()), lambda - Complex::new(m00, T::zero())]
        } else {
            [lambda - Complex::new(m11, T::zero()), Complex::new(m10, T::zero())]
        };
        let mut norm = (candidate[0].norm_sqr() + candidate[1].norm_sqr()).sqrt();
        if norm <= T::epsilon() {
            candidate = if col == 0 {
                [Complex::new(T::one(), T::zero()), Complex::new(T::zero(), T::zero())]
            } else {
                [Complex::new(T::zero(), T::zero()), Complex::new(T::one(), T::zero())]
            };
            norm = T::one();
        }
        schur_vectors[[0, col]] = candidate[0] / norm;
        schur_vectors[[1, col]] = candidate[1] / norm;
    }

    Some(NdarrayNonsymmetricEigenResult { eigenvalues, schur_vectors })
}

fn balance_nonsymmetric_matrix<T: NabledReal>(
    matrix: &ArrayView2<'_, T>,
    config: &NonsymmetricEigenConfig<T>,
) -> (Array2<T>, Array1<T>) {
    if !config.balance || matrix.nrows() <= 1 {
        return (matrix.to_owned(), Array1::<T>::from_elem(matrix.nrows(), T::one()));
    }

    let mut balanced = matrix.to_owned();
    let mut diagonal = Array1::<T>::from_elem(matrix.nrows(), T::one());
    let radix = T::from_f64(2.0).unwrap_or(T::one() + T::one());
    let threshold = config
        .balance_tolerance
        .clamp(T::from_f64(1.0e-6).unwrap_or(T::epsilon()), T::from_f64(0.5).unwrap_or(T::one()));
    let max_iterations = config.balance_max_iterations.max(1);

    for _ in 0..max_iterations {
        let mut changed = false;
        for i in 0..balanced.nrows() {
            let mut row_norm = T::zero();
            let mut col_norm = T::zero();
            for j in 0..balanced.ncols() {
                if i != j {
                    row_norm += balanced[[i, j]].abs();
                    col_norm += balanced[[j, i]].abs();
                }
            }

            if row_norm <= T::epsilon() || col_norm <= T::epsilon() {
                continue;
            }

            let mut factor = T::one();
            let mut col = col_norm;
            let row = row_norm;
            while col < row / radix {
                factor *= radix;
                col = col * radix * radix;
            }
            while col > row * radix {
                factor /= radix;
                col /= radix * radix;
            }

            let one = T::one();
            if (col + row) < (one - threshold) * (col_norm + row_norm) {
                changed = true;
                diagonal[i] *= factor;
                for j in 0..balanced.ncols() {
                    balanced[[i, j]] *= factor;
                    balanced[[j, i]] /= factor;
                }
            }
        }
        if !changed {
            break;
        }
    }

    (balanced, diagonal)
}

fn normalize_complex_columns<T: NabledReal>(vectors: &mut Array2<Complex<T>>) {
    for col in 0..vectors.ncols() {
        let mut norm_sq = T::zero();
        for row in 0..vectors.nrows() {
            norm_sq += vectors[[row, col]].norm_sqr();
        }
        let norm = norm_sq.sqrt().max(T::epsilon());
        for row in 0..vectors.nrows() {
            vectors[[row, col]] = vectors[[row, col]] / norm;
        }
    }
}

fn match_left_eigenvectors<T: NabledReal>(
    target_eigenvalues: &Array1<Complex<T>>,
    source_eigenvalues: &Array1<Complex<T>>,
    source_vectors: &Array2<Complex<T>>,
) -> Array2<Complex<T>> {
    let mut matched = Array2::<Complex<T>>::zeros((source_vectors.nrows(), source_vectors.ncols()));
    let mut used = vec![false; source_eigenvalues.len()];

    for target_col in 0..target_eigenvalues.len() {
        let target = target_eigenvalues[target_col].conj();
        let mut best_index = None;
        let mut best_distance = T::max_value();

        for source_col in 0..source_eigenvalues.len() {
            if used[source_col] {
                continue;
            }
            let distance = (source_eigenvalues[source_col] - target).norm();
            if distance < best_distance {
                best_distance = distance;
                best_index = Some(source_col);
            }
        }

        if let Some(source_col) = best_index {
            used[source_col] = true;
            for row in 0..source_vectors.nrows() {
                matched[[row, target_col]] = source_vectors[[row, source_col]];
            }
        }
    }

    matched
}

#[cfg(feature = "magma-system")]
fn symmetric_internal<T: NabledReal + magma::MagmaReal>(
    matrix: &ArrayView2<'_, T>,
) -> Result<NdarrayEigenResult<T>, EigenError> {
    validate_symmetric_input(matrix)?;
    if let Ok((eigenvalues, eigenvectors)) = magma::symmetric_eigen(matrix) {
        let (eigenvalues, eigenvectors) = sort_eigenpairs_desc(&eigenvalues, &eigenvectors);
        return Ok(NdarrayEigenResult { eigenvalues, eigenvectors });
    }

    let (eigenvalues, eigenvectors) = jacobi_eigen_symmetric(
        &matrix.to_owned(),
        T::from_f64(DenseKernelPolicy::BASE_TOLERANCE).unwrap_or(T::epsilon()),
        DenseKernelPolicy::JACOBI_MAX_ITERATIONS,
    )
    .map_err(|_| EigenError::ConvergenceFailed)?;
    let (eigenvalues, eigenvectors) = sort_eigenpairs_desc(&eigenvalues, &eigenvectors);
    Ok(NdarrayEigenResult { eigenvalues, eigenvectors })
}

#[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
fn symmetric_internal<T: NabledReal>(
    matrix: &ArrayView2<'_, T>,
) -> Result<NdarrayEigenResult<T>, EigenError> {
    validate_symmetric_input(matrix)?;

    let (eigenvalues, eigenvectors) = jacobi_eigen_symmetric(
        &matrix.to_owned(),
        T::from_f64(DenseKernelPolicy::BASE_TOLERANCE).unwrap_or(T::epsilon()),
        DenseKernelPolicy::JACOBI_MAX_ITERATIONS,
    )
    .map_err(|_| EigenError::ConvergenceFailed)?;
    let (eigenvalues, eigenvectors) = sort_eigenpairs_desc(&eigenvalues, &eigenvectors);
    Ok(NdarrayEigenResult { eigenvalues, eigenvectors })
}

#[cfg(feature = "lapack-provider")]
fn symmetric_provider<T>(matrix: &ArrayView2<'_, T>) -> Result<NdarrayEigenResult<T>, EigenError>
where
    T: NabledReal + ndarray_linalg::Lapack<Real = T> + std::ops::AddAssign,
{
    use ndarray_linalg::{Eigh as _, UPLO};

    validate_symmetric_input(matrix)?;
    let (eigenvalues, eigenvectors) =
        matrix.eigh(UPLO::Lower).map_err(|_| EigenError::ConvergenceFailed)?;
    let (eigenvalues, eigenvectors) = sort_eigenpairs_desc(&eigenvalues, &eigenvectors);
    Ok(NdarrayEigenResult { eigenvalues, eigenvectors })
}

#[cfg(feature = "magma-system")]
fn generalized_internal<T: cholesky::CholeskyProviderScalar>(
    matrix_a: &ArrayView2<'_, T>,
    matrix_b: &ArrayView2<'_, T>,
) -> Result<NdarrayGeneralizedEigenResult<T>, EigenError> {
    validate_symmetric_input(matrix_a)?;
    validate_symmetric_input(matrix_b)?;
    if matrix_a.dim() != matrix_b.dim() {
        return Err(EigenError::InvalidDimensions);
    }

    let b_inverse = cholesky::inverse_view(matrix_b).map_err(|error| match error {
        cholesky::CholeskyError::NotPositiveDefinite => EigenError::NotPositiveDefinite,
        cholesky::CholeskyError::EmptyMatrix => EigenError::EmptyMatrix,
        cholesky::CholeskyError::NotSquare => EigenError::NotSquare,
        _ => EigenError::NumericalInstability,
    })?;

    let c = b_inverse.dot(matrix_a);
    let symmetric_c = (&c + &c.t()) * T::from_f64(0.5).unwrap_or(T::one() / (T::one() + T::one()));

    let NdarrayEigenResult { eigenvalues, eigenvectors } = symmetric_internal(&symmetric_c.view())?;

    Ok(NdarrayGeneralizedEigenResult { eigenvalues, eigenvectors })
}

#[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
fn generalized_internal<T: NabledReal>(
    matrix_a: &ArrayView2<'_, T>,
    matrix_b: &ArrayView2<'_, T>,
) -> Result<NdarrayGeneralizedEigenResult<T>, EigenError> {
    validate_symmetric_input(matrix_a)?;
    validate_symmetric_input(matrix_b)?;
    if matrix_a.dim() != matrix_b.dim() {
        return Err(EigenError::InvalidDimensions);
    }

    let b_inverse = cholesky::inverse_view(matrix_b).map_err(|error| match error {
        cholesky::CholeskyError::NotPositiveDefinite => EigenError::NotPositiveDefinite,
        cholesky::CholeskyError::EmptyMatrix => EigenError::EmptyMatrix,
        cholesky::CholeskyError::NotSquare => EigenError::NotSquare,
        _ => EigenError::NumericalInstability,
    })?;

    let c = b_inverse.dot(matrix_a);
    let symmetric_c = (&c + &c.t()) * T::from_f64(0.5).unwrap_or(T::one() / (T::one() + T::one()));

    let (eigenvalues, eigenvectors) = jacobi_eigen_symmetric(
        &symmetric_c,
        T::from_f64(DenseKernelPolicy::BASE_TOLERANCE).unwrap_or(T::epsilon()),
        DenseKernelPolicy::JACOBI_MAX_ITERATIONS,
    )
    .map_err(|_| EigenError::ConvergenceFailed)?;
    let (eigenvalues, eigenvectors) = sort_eigenpairs_desc(&eigenvalues, &eigenvectors);

    Ok(NdarrayGeneralizedEigenResult { eigenvalues, eigenvectors })
}

#[cfg(all(feature = "lapack-provider", not(feature = "magma-system")))]
fn generalized_provider<T>(
    matrix_a: &ArrayView2<'_, T>,
    matrix_b: &ArrayView2<'_, T>,
) -> Result<NdarrayGeneralizedEigenResult<T>, EigenError>
where
    T: NabledReal + ndarray_linalg::Lapack<Real = T> + std::ops::AddAssign,
{
    validate_symmetric_input(matrix_a)?;
    validate_symmetric_input(matrix_b)?;
    if matrix_a.dim() != matrix_b.dim() {
        return Err(EigenError::InvalidDimensions);
    }

    // Reduce generalized SPD problem A x = lambda B x to a standard symmetric
    // problem via B^{-1}A, while reusing provider-backed Cholesky inverse.
    let b_inverse = cholesky::inverse_view(matrix_b).map_err(|error| match error {
        cholesky::CholeskyError::NotPositiveDefinite => EigenError::NotPositiveDefinite,
        cholesky::CholeskyError::EmptyMatrix => EigenError::EmptyMatrix,
        cholesky::CholeskyError::NotSquare => EigenError::NotSquare,
        _ => EigenError::NumericalInstability,
    })?;
    let c = b_inverse.dot(matrix_a);
    let symmetric_c = (&c + &c.t()) * T::from_f64(0.5).unwrap_or(T::one() / (T::one() + T::one()));
    let NdarrayEigenResult { eigenvalues, eigenvectors } = symmetric_provider(&symmetric_c.view())?;
    Ok(NdarrayGeneralizedEigenResult { eigenvalues, eigenvectors })
}

#[cfg(not(feature = "lapack-provider"))]
fn nonsymmetric_complex_internal(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<NdarrayNonsymmetricEigenResult, EigenError> {
    validate_complex_square_finite(matrix)?;
    let dimension = matrix.nrows();

    if dimension == 1 {
        let mut eigenvalues = Array1::<Complex64>::zeros(1);
        eigenvalues[0] = matrix[[0, 0]];
        let mut schur_vectors = Array2::<Complex64>::zeros((1, 1));
        schur_vectors[[0, 0]] = Complex64::new(1.0, 0.0);
        return Ok(NdarrayNonsymmetricEigenResult { eigenvalues, schur_vectors });
    }

    // Closed-form quadratic solve stabilizes the common 2x2 path and avoids
    // relying on iterative Schur convergence for tiny inputs.
    if dimension == 2 {
        let m00 = matrix[[0, 0]];
        let m01 = matrix[[0, 1]];
        let m10 = matrix[[1, 0]];
        let m11 = matrix[[1, 1]];
        let trace = m00 + m11;
        let determinant = m00 * m11 - m01 * m10;
        let discriminant = trace * trace - Complex64::new(4.0, 0.0) * determinant;
        let root = discriminant.sqrt();

        let mut eigenvalues = Array1::<Complex64>::zeros(2);
        eigenvalues[0] = (trace + root) / 2.0;
        eigenvalues[1] = (trace - root) / 2.0;

        let mut schur_vectors = Array2::<Complex64>::zeros((2, 2));
        for i in 0..2 {
            let lambda = eigenvalues[i];
            let candidate =
                if m01.norm() >= m10.norm() { [m01, lambda - m00] } else { [lambda - m11, m10] };
            let norm = (candidate[0].norm_sqr() + candidate[1].norm_sqr()).sqrt().max(f64::EPSILON);
            schur_vectors[[0, i]] = candidate[0] / norm;
            schur_vectors[[1, i]] = candidate[1] / norm;
        }
        return Ok(NdarrayNonsymmetricEigenResult { eigenvalues, schur_vectors });
    }

    let decomposition =
        schur::compute_schur_complex_view(matrix).map_err(|_| EigenError::ConvergenceFailed)?;
    let mut eigenvalues = Array1::<Complex64>::zeros(dimension);
    for i in 0..dimension {
        eigenvalues[i] = decomposition.t[[i, i]];
    }
    Ok(NdarrayNonsymmetricEigenResult { eigenvalues, schur_vectors: decomposition.q })
}

#[cfg(not(feature = "lapack-provider"))]
fn nonsymmetric_internal<T: qr::QrInternalScalar>(
    matrix: &ArrayView2<'_, T>,
) -> Result<NdarrayNonsymmetricEigenResult<T>, EigenError> {
    validate_nonsymmetric_input(matrix)?;
    if let Some(result) = nonsymmetric_small_matrix(matrix) {
        return Ok(result);
    }
    let schur = schur::compute_schur_view(matrix).map_err(|_| EigenError::ConvergenceFailed)?;
    let dimension = matrix.nrows();
    let tolerance = T::from_f64(DenseKernelPolicy::BASE_TOLERANCE).unwrap_or(T::epsilon());

    let mut eigenvalues = Array1::<Complex<T>>::zeros(dimension);
    let mut index = 0_usize;
    while index < dimension {
        if index + 1 < dimension && num_traits::Float::abs(schur.t[[index + 1, index]]) > tolerance
        {
            let block_00 = schur.t[[index, index]];
            let block_01 = schur.t[[index, index + 1]];
            let block_10 = schur.t[[index + 1, index]];
            let block_11 = schur.t[[index + 1, index + 1]];
            let trace = block_00 + block_11;
            let determinant = block_00 * block_11 - block_01 * block_10;
            let four = T::from_f64(4.0).unwrap_or(T::one() + T::one() + T::one() + T::one());
            let discriminant = Complex::new(trace * trace - four * determinant, T::zero()).sqrt();
            let two = T::from_f64(2.0).unwrap_or(T::one() + T::one());
            let trace_complex = Complex::new(trace, T::zero());
            eigenvalues[index] = (trace_complex + discriminant) / two;
            eigenvalues[index + 1] = (trace_complex - discriminant) / two;
            index += 2;
        } else {
            eigenvalues[index] = Complex::new(schur.t[[index, index]], T::zero());
            index += 1;
        }
    }

    let schur_vectors = schur.q.mapv(|value| Complex::new(value, T::zero()));
    Ok(NdarrayNonsymmetricEigenResult { eigenvalues, schur_vectors })
}

#[cfg(feature = "lapack-provider")]
fn nonsymmetric_internal<T>(
    matrix: &ArrayView2<'_, T>,
) -> Result<NdarrayNonsymmetricEigenResult<T>, EigenError>
where
    T: schur::SchurProviderScalar,
{
    validate_nonsymmetric_input(matrix)?;
    if let Some(result) = nonsymmetric_small_matrix(matrix) {
        return Ok(result);
    }
    let schur = schur::compute_schur_view(matrix).map_err(|_| EigenError::ConvergenceFailed)?;
    let dimension = matrix.nrows();
    let tolerance = T::from_f64(DenseKernelPolicy::BASE_TOLERANCE).unwrap_or(T::epsilon());

    let mut eigenvalues = Array1::<Complex<T>>::zeros(dimension);
    let mut index = 0_usize;
    while index < dimension {
        if index + 1 < dimension && num_traits::Float::abs(schur.t[[index + 1, index]]) > tolerance
        {
            let block_00 = schur.t[[index, index]];
            let block_01 = schur.t[[index, index + 1]];
            let block_10 = schur.t[[index + 1, index]];
            let block_11 = schur.t[[index + 1, index + 1]];
            let trace = block_00 + block_11;
            let determinant = block_00 * block_11 - block_01 * block_10;
            let four = T::from_f64(4.0).unwrap_or(T::one() + T::one() + T::one() + T::one());
            let discriminant = Complex::new(trace * trace - four * determinant, T::zero()).sqrt();
            let two = T::from_f64(2.0).unwrap_or(T::one() + T::one());
            let trace_complex = Complex::new(trace, T::zero());
            eigenvalues[index] = (trace_complex + discriminant) / two;
            eigenvalues[index + 1] = (trace_complex - discriminant) / two;
            index += 2;
        } else {
            eigenvalues[index] = Complex::new(schur.t[[index, index]], T::zero());
            index += 1;
        }
    }

    let schur_vectors = schur.q.mapv(|value| Complex::new(value, T::zero()));
    Ok(NdarrayNonsymmetricEigenResult { eigenvalues, schur_vectors })
}

#[cfg(feature = "magma-system")]
fn nonsymmetric_complex_provider(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<NdarrayNonsymmetricEigenResult, EigenError> {
    validate_complex_square_finite(matrix)?;
    match magma::nonsymmetric_eigen_complex(matrix) {
        Ok((eigenvalues, right_eigenvectors)) => {
            Ok(NdarrayNonsymmetricEigenResult { eigenvalues, schur_vectors: right_eigenvectors })
        }
        Err(error) => {
            #[cfg(not(feature = "lapack-provider"))]
            {
                let _ = error;
                nonsymmetric_complex_internal(matrix)
            }
            #[cfg(feature = "lapack-provider")]
            {
                match error {
                    "empty" => Err(EigenError::EmptyMatrix),
                    "not_square" => Err(EigenError::NotSquare),
                    "convergence_failed" => Err(EigenError::ConvergenceFailed),
                    _ => Err(EigenError::NumericalInstability),
                }
            }
        }
    }
}

#[cfg(all(feature = "lapack-provider", not(feature = "magma-system")))]
fn nonsymmetric_complex_provider(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<NdarrayNonsymmetricEigenResult, EigenError> {
    use ndarray_linalg::Eig as _;

    validate_complex_square_finite(matrix)?;
    let (eigenvalues, right_eigenvectors) =
        matrix.eig().map_err(|_| EigenError::ConvergenceFailed)?;
    Ok(NdarrayNonsymmetricEigenResult { eigenvalues, schur_vectors: right_eigenvectors })
}

/// Compute symmetric eigen decomposition.
///
/// # Errors
/// Returns an error for non-symmetric input or convergence failure.
#[cfg(feature = "lapack-provider")]
pub fn symmetric<T>(matrix: &Array2<T>) -> Result<NdarrayEigenResult<T>, EigenError>
where
    T: NabledReal + ndarray_linalg::Lapack<Real = T> + std::ops::AddAssign,
{
    symmetric_provider(&matrix.view())
}

/// Compute symmetric eigen decomposition.
///
/// # Errors
/// Returns an error for non-symmetric input or convergence failure.
#[cfg(not(feature = "lapack-provider"))]
pub fn symmetric<T: EigenInternalScalar>(
    matrix: &Array2<T>,
) -> Result<NdarrayEigenResult<T>, EigenError> {
    symmetric_internal(&matrix.view())
}

/// Compute symmetric eigen decomposition from a matrix view.
///
/// # Errors
/// Returns an error for non-symmetric input or convergence failure.
#[cfg(feature = "lapack-provider")]
pub fn symmetric_view<T>(matrix: &ArrayView2<'_, T>) -> Result<NdarrayEigenResult<T>, EigenError>
where
    T: NabledReal + ndarray_linalg::Lapack<Real = T> + std::ops::AddAssign,
{
    symmetric_provider(matrix)
}

/// Compute symmetric eigen decomposition from a matrix view.
///
/// # Errors
/// Returns an error for non-symmetric input or convergence failure.
#[cfg(not(feature = "lapack-provider"))]
pub fn symmetric_view<T: EigenInternalScalar>(
    matrix: &ArrayView2<'_, T>,
) -> Result<NdarrayEigenResult<T>, EigenError> {
    symmetric_internal(matrix)
}

/// Compute generalized symmetric eigen decomposition `(A, B)`.
///
/// # Errors
/// Returns an error when dimensions are incompatible or `B` is not SPD.
#[cfg(feature = "magma-system")]
pub fn generalized<T>(
    matrix_a: &Array2<T>,
    matrix_b: &Array2<T>,
) -> Result<NdarrayGeneralizedEigenResult<T>, EigenError>
where
    T: cholesky::CholeskyProviderScalar,
{
    generalized_internal(&matrix_a.view(), &matrix_b.view())
}

/// Compute generalized symmetric eigen decomposition `(A, B)`.
///
/// # Errors
/// Returns an error when dimensions are incompatible or `B` is not SPD.
#[cfg(all(feature = "lapack-provider", not(feature = "magma-system")))]
pub fn generalized<T>(
    matrix_a: &Array2<T>,
    matrix_b: &Array2<T>,
) -> Result<NdarrayGeneralizedEigenResult<T>, EigenError>
where
    T: NabledReal + ndarray_linalg::Lapack<Real = T> + std::ops::AddAssign,
{
    generalized_provider(&matrix_a.view(), &matrix_b.view())
}

/// Compute generalized symmetric eigen decomposition `(A, B)`.
///
/// # Errors
/// Returns an error when dimensions are incompatible or `B` is not SPD.
#[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
pub fn generalized<T: NabledReal>(
    matrix_a: &Array2<T>,
    matrix_b: &Array2<T>,
) -> Result<NdarrayGeneralizedEigenResult<T>, EigenError> {
    generalized_internal(&matrix_a.view(), &matrix_b.view())
}

/// Compute generalized symmetric eigen decomposition `(A, B)` from matrix views.
///
/// # Errors
/// Returns an error when dimensions are incompatible or `B` is not SPD.
#[cfg(feature = "magma-system")]
pub fn generalized_view<T>(
    matrix_a: &ArrayView2<'_, T>,
    matrix_b: &ArrayView2<'_, T>,
) -> Result<NdarrayGeneralizedEigenResult<T>, EigenError>
where
    T: cholesky::CholeskyProviderScalar,
{
    generalized_internal(matrix_a, matrix_b)
}

/// Compute generalized symmetric eigen decomposition `(A, B)` from matrix views.
///
/// # Errors
/// Returns an error when dimensions are incompatible or `B` is not SPD.
#[cfg(all(feature = "lapack-provider", not(feature = "magma-system")))]
pub fn generalized_view<T>(
    matrix_a: &ArrayView2<'_, T>,
    matrix_b: &ArrayView2<'_, T>,
) -> Result<NdarrayGeneralizedEigenResult<T>, EigenError>
where
    T: NabledReal + ndarray_linalg::Lapack<Real = T> + std::ops::AddAssign,
{
    generalized_provider(matrix_a, matrix_b)
}

/// Compute generalized symmetric eigen decomposition `(A, B)` from matrix views.
///
/// # Errors
/// Returns an error when dimensions are incompatible or `B` is not SPD.
#[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
pub fn generalized_view<T: NabledReal>(
    matrix_a: &ArrayView2<'_, T>,
    matrix_b: &ArrayView2<'_, T>,
) -> Result<NdarrayGeneralizedEigenResult<T>, EigenError> {
    generalized_internal(matrix_a, matrix_b)
}

/// Compute non-symmetric eigen decomposition via real Schur reduction.
///
/// # Errors
/// Returns an error for non-square, non-finite, or non-convergent inputs.
#[cfg(feature = "lapack-provider")]
pub fn nonsymmetric<T>(matrix: &Array2<T>) -> Result<NdarrayNonsymmetricEigenResult<T>, EigenError>
where
    T: schur::SchurProviderScalar,
{
    nonsymmetric_internal(&matrix.view())
}

/// Compute non-symmetric eigen decomposition via real Schur reduction.
///
/// # Errors
/// Returns an error for non-square, non-finite, or non-convergent inputs.
#[cfg(not(feature = "lapack-provider"))]
pub fn nonsymmetric<T: qr::QrInternalScalar>(
    matrix: &Array2<T>,
) -> Result<NdarrayNonsymmetricEigenResult<T>, EigenError> {
    nonsymmetric_internal(&matrix.view())
}

/// Compute non-symmetric eigen decomposition from a real matrix view.
///
/// # Errors
/// Returns an error for non-square, non-finite, or non-convergent inputs.
#[cfg(feature = "lapack-provider")]
pub fn nonsymmetric_view<T>(
    matrix: &ArrayView2<'_, T>,
) -> Result<NdarrayNonsymmetricEigenResult<T>, EigenError>
where
    T: schur::SchurProviderScalar,
{
    nonsymmetric_internal(matrix)
}

/// Compute non-symmetric eigen decomposition from a real matrix view.
///
/// # Errors
/// Returns an error for non-square, non-finite, or non-convergent inputs.
#[cfg(not(feature = "lapack-provider"))]
pub fn nonsymmetric_view<T: qr::QrInternalScalar>(
    matrix: &ArrayView2<'_, T>,
) -> Result<NdarrayNonsymmetricEigenResult<T>, EigenError> {
    nonsymmetric_internal(matrix)
}

/// Compute non-symmetric eigen decomposition for complex matrices.
///
/// # Errors
/// Returns an error for non-square, non-finite, or non-convergent inputs.
pub fn nonsymmetric_complex(
    matrix: &Array2<Complex64>,
) -> Result<NdarrayNonsymmetricEigenResult<f64>, EigenError> {
    nonsymmetric_complex_impl(&matrix.view())
}

fn nonsymmetric_complex_impl(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<NdarrayNonsymmetricEigenResult<f64>, EigenError> {
    #[cfg(feature = "magma-system")]
    {
        nonsymmetric_complex_provider(matrix)
    }
    #[cfg(all(feature = "lapack-provider", not(feature = "magma-system")))]
    {
        nonsymmetric_complex_provider(matrix)
    }
    #[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
    {
        nonsymmetric_complex_internal(matrix)
    }
}

/// Compute non-symmetric eigen decomposition for complex matrix views.
///
/// # Errors
/// Returns an error for non-square, non-finite, or non-convergent inputs.
pub fn nonsymmetric_complex_view(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<NdarrayNonsymmetricEigenResult<f64>, EigenError> {
    nonsymmetric_complex_impl(matrix)
}

/// Balance a real non-symmetric matrix using diagonal similarity scaling.
///
/// # Errors
/// Returns an error for non-square or non-finite input.
pub fn balance_nonsymmetric<T: NabledReal>(
    matrix: &Array2<T>,
    config: &NonsymmetricEigenConfig<T>,
) -> Result<(Array2<T>, Array1<T>), EigenError> {
    balance_nonsymmetric_view(&matrix.view(), config)
}

/// Balance a real non-symmetric matrix view using diagonal similarity scaling.
///
/// # Errors
/// Returns an error for non-square or non-finite input.
pub fn balance_nonsymmetric_view<T: NabledReal>(
    matrix: &ArrayView2<'_, T>,
    config: &NonsymmetricEigenConfig<T>,
) -> Result<(Array2<T>, Array1<T>), EigenError> {
    validate_nonsymmetric_input(matrix)?;
    Ok(balance_nonsymmetric_matrix(matrix, config))
}

/// Compute non-symmetric eigen decomposition with matched left/right eigenvectors.
///
/// When balancing is enabled in `config`, decomposition runs on the balanced matrix and
/// vectors are mapped back to the original matrix coordinates.
///
/// # Errors
/// Returns an error for non-square, non-finite, or non-convergent inputs.
#[cfg(feature = "lapack-provider")]
pub fn nonsymmetric_bi<T>(
    matrix: &Array2<T>,
    config: &NonsymmetricEigenConfig<T>,
) -> Result<NdarrayNonsymmetricBiEigenResult<T>, EigenError>
where
    T: schur::SchurProviderScalar,
{
    nonsymmetric_bi_view(&matrix.view(), config)
}

/// Compute non-symmetric eigen decomposition with matched left/right eigenvectors.
///
/// When balancing is enabled in `config`, decomposition runs on the balanced matrix and
/// vectors are mapped back to the original matrix coordinates.
///
/// # Errors
/// Returns an error for non-square, non-finite, or non-convergent inputs.
#[cfg(not(feature = "lapack-provider"))]
pub fn nonsymmetric_bi<T: qr::QrInternalScalar>(
    matrix: &Array2<T>,
    config: &NonsymmetricEigenConfig<T>,
) -> Result<NdarrayNonsymmetricBiEigenResult<T>, EigenError> {
    nonsymmetric_bi_view(&matrix.view(), config)
}

/// Compute non-symmetric eigen decomposition with matched left/right eigenvectors from a view.
///
/// # Errors
/// Returns an error for non-square, non-finite, or non-convergent inputs.
#[cfg(feature = "lapack-provider")]
pub fn nonsymmetric_bi_view<T>(
    matrix: &ArrayView2<'_, T>,
    config: &NonsymmetricEigenConfig<T>,
) -> Result<NdarrayNonsymmetricBiEigenResult<T>, EigenError>
where
    T: schur::SchurProviderScalar,
{
    nonsymmetric_bi_view_impl(matrix, config)
}

/// Compute non-symmetric eigen decomposition with matched left/right eigenvectors from a view.
///
/// # Errors
/// Returns an error for non-square, non-finite, or non-convergent inputs.
#[cfg(not(feature = "lapack-provider"))]
pub fn nonsymmetric_bi_view<T: qr::QrInternalScalar>(
    matrix: &ArrayView2<'_, T>,
    config: &NonsymmetricEigenConfig<T>,
) -> Result<NdarrayNonsymmetricBiEigenResult<T>, EigenError> {
    nonsymmetric_bi_view_impl(matrix, config)
}

#[cfg(not(feature = "lapack-provider"))]
fn nonsymmetric_bi_view_impl<T: qr::QrInternalScalar>(
    matrix: &ArrayView2<'_, T>,
    config: &NonsymmetricEigenConfig<T>,
) -> Result<NdarrayNonsymmetricBiEigenResult<T>, EigenError> {
    validate_nonsymmetric_input(matrix)?;
    let (balanced_matrix, balancing_diagonal) = balance_nonsymmetric_matrix(matrix, config);
    let right = nonsymmetric_view(&balanced_matrix.view())?;

    let balanced_transpose = balanced_matrix.t();
    let left_seed = nonsymmetric_view(&balanced_transpose)?;
    let mut left_eigenvectors = match_left_eigenvectors(
        &right.eigenvalues,
        &left_seed.eigenvalues,
        &left_seed.schur_vectors,
    );
    let mut right_eigenvectors = right.schur_vectors;

    for row in 0..balanced_matrix.nrows() {
        let scale = balancing_diagonal[row].max(T::epsilon());
        for col in 0..right_eigenvectors.ncols() {
            right_eigenvectors[[row, col]] = right_eigenvectors[[row, col]] / scale;
            left_eigenvectors[[row, col]] = left_eigenvectors[[row, col]] * scale;
        }
    }
    normalize_complex_columns(&mut right_eigenvectors);
    normalize_complex_columns(&mut left_eigenvectors);

    Ok(NdarrayNonsymmetricBiEigenResult {
        eigenvalues: right.eigenvalues,
        right_eigenvectors,
        left_eigenvectors,
        balancing_diagonal,
        balanced_matrix,
    })
}

#[cfg(feature = "lapack-provider")]
#[allow(clippy::assign_op_pattern)]
fn nonsymmetric_bi_view_impl<T>(
    matrix: &ArrayView2<'_, T>,
    config: &NonsymmetricEigenConfig<T>,
) -> Result<NdarrayNonsymmetricBiEigenResult<T>, EigenError>
where
    T: schur::SchurProviderScalar,
{
    validate_nonsymmetric_input(matrix)?;
    let (balanced_matrix, balancing_diagonal) = balance_nonsymmetric_matrix(matrix, config);
    let right = nonsymmetric_view(&balanced_matrix.view())?;

    let balanced_transpose = balanced_matrix.t();
    let left_seed = nonsymmetric_view(&balanced_transpose)?;
    let mut left_eigenvectors = match_left_eigenvectors(
        &right.eigenvalues,
        &left_seed.eigenvalues,
        &left_seed.schur_vectors,
    );
    let mut right_eigenvectors = right.schur_vectors;

    for row in 0..balanced_matrix.nrows() {
        let scale = balancing_diagonal[row].max(T::epsilon());
        for col in 0..right_eigenvectors.ncols() {
            right_eigenvectors[[row, col]] = right_eigenvectors[[row, col]] / scale;
            left_eigenvectors[[row, col]] = left_eigenvectors[[row, col]] * scale;
        }
    }
    normalize_complex_columns(&mut right_eigenvectors);
    normalize_complex_columns(&mut left_eigenvectors);

    Ok(NdarrayNonsymmetricBiEigenResult {
        eigenvalues: right.eigenvalues,
        right_eigenvectors,
        left_eigenvectors,
        balancing_diagonal,
        balanced_matrix,
    })
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;
    use num_complex::Complex64;

    use super::*;

    #[test]
    fn symmetric_eigen_reconstructs() {
        let matrix =
            Array2::from_shape_vec((2, 2), vec![4.0_f64, 1.0_f64, 1.0_f64, 3.0_f64]).unwrap();
        let eigen = symmetric(&matrix).unwrap();

        let diagonal = Array2::from_diag(&eigen.eigenvalues);
        let reconstructed = eigen.eigenvectors.dot(&diagonal).dot(&eigen.eigenvectors.t());

        for i in 0..2 {
            for j in 0..2 {
                assert!((reconstructed[[i, j]] - matrix[[i, j]]).abs() < 1e-5_f64);
            }
        }
    }

    #[test]
    fn non_symmetric_matrix_errors() {
        let matrix =
            Array2::from_shape_vec((2, 2), vec![1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64]).unwrap();
        let result = symmetric(&matrix);
        assert!(matches!(result, Err(EigenError::NotSymmetric)));
    }

    #[test]
    fn generalized_eigen_solves_spd_pair() {
        let a = Array2::from_shape_vec((2, 2), vec![4.0_f64, 1.0_f64, 1.0_f64, 3.0_f64]).unwrap();
        let b = Array2::from_shape_vec((2, 2), vec![2.0_f64, 0.0_f64, 0.0_f64, 1.0_f64]).unwrap();
        let result = generalized(&a, &b).unwrap();
        assert_eq!(result.eigenvalues.len(), 2);
        assert_eq!(result.eigenvectors.dim(), (2, 2));
    }

    #[test]
    fn generalized_eigen_rejects_dimension_mismatch() {
        let a = Array2::<f64>::eye(2);
        let b = Array2::<f64>::eye(3);
        let result = generalized(&a, &b);
        assert!(matches!(result, Err(EigenError::InvalidDimensions)));
    }

    #[test]
    fn generalized_eigen_rejects_non_spd_b() {
        let a = Array2::eye(2);
        let b = Array2::from_shape_vec((2, 2), vec![1.0_f64, 0.0_f64, 0.0_f64, -1.0_f64]).unwrap();
        let result = generalized(&a, &b);
        assert!(matches!(result, Err(EigenError::NotPositiveDefinite)));
    }

    #[test]
    fn view_variants_match_owned() {
        let matrix =
            Array2::from_shape_vec((2, 2), vec![5.0_f64, 1.0_f64, 1.0_f64, 4.0_f64]).unwrap();
        let owned = symmetric(&matrix).unwrap();
        let viewed = symmetric_view(&matrix.view()).unwrap();
        assert_eq!(owned.eigenvalues.len(), viewed.eigenvalues.len());
        assert_eq!(owned.eigenvectors.dim(), viewed.eigenvectors.dim());

        let a = Array2::from_shape_vec((2, 2), vec![4.0_f64, 1.0_f64, 1.0_f64, 3.0_f64]).unwrap();
        let b = Array2::from_shape_vec((2, 2), vec![2.0_f64, 0.0_f64, 0.0_f64, 1.0_f64]).unwrap();
        let owned_generalized = generalized(&a, &b).unwrap();
        let viewed_generalized = generalized_view(&a.view(), &b.view()).unwrap();
        assert_eq!(owned_generalized.eigenvalues.len(), viewed_generalized.eigenvalues.len());
        assert_eq!(owned_generalized.eigenvectors.dim(), viewed_generalized.eigenvectors.dim());
    }

    #[test]
    fn symmetric_eigen_rejects_empty_not_square_and_non_finite() {
        let empty = Array2::<f64>::zeros((0, 0));
        assert!(matches!(symmetric(&empty), Err(EigenError::EmptyMatrix)));

        let non_square = Array2::<f64>::zeros((2, 3));
        assert!(matches!(symmetric(&non_square), Err(EigenError::NotSquare)));

        let non_finite =
            Array2::from_shape_vec((2, 2), vec![1.0_f64, f64::NAN, f64::NAN, 2.0_f64]).unwrap();
        assert!(matches!(symmetric(&non_finite), Err(EigenError::NumericalInstability)));
    }

    #[test]
    fn generalized_eigen_rejects_non_symmetric_a() {
        let a = Array2::from_shape_vec((2, 2), vec![1.0_f64, 2.0_f64, 0.0_f64, 1.0_f64]).unwrap();
        let b = Array2::eye(2);
        let result = generalized(&a, &b);
        assert!(matches!(result, Err(EigenError::NotSymmetric)));
    }

    #[test]
    fn symmetric_eigenvalues_are_sorted_descending() {
        let matrix =
            Array2::from_shape_vec((2, 2), vec![2.0_f64, 0.0_f64, 0.0_f64, 5.0_f64]).unwrap();
        let eigen = symmetric(&matrix).unwrap();
        assert!(eigen.eigenvalues[0] >= eigen.eigenvalues[1]);
    }

    #[test]
    fn nonsymmetric_real_eigenvalues_cover_complex_pair() {
        let rotation =
            Array2::from_shape_vec((2, 2), vec![0.0_f64, -1.0_f64, 1.0_f64, 0.0_f64]).unwrap();
        let result = nonsymmetric(&rotation).unwrap();
        assert_eq!(result.eigenvalues.len(), 2);
        let mut imag_parts = result.eigenvalues.iter().map(|value| value.im).collect::<Vec<_>>();
        imag_parts.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap());
        assert!(imag_parts[0] < -0.9_f64);
        assert!(imag_parts[1] > 0.9_f64);
    }

    #[test]
    fn nonsymmetric_complex_eigenvalues_match_diagonal() {
        let diagonal = Array2::from_shape_vec((2, 2), vec![
            Complex64::new(2.0_f64, 1.0_f64),
            Complex64::new(0.0_f64, 0.0_f64),
            Complex64::new(0.0_f64, 0.0_f64),
            Complex64::new(-3.0_f64, 0.5_f64),
        ])
        .unwrap();
        let result = nonsymmetric_complex(&diagonal).unwrap();
        assert_eq!(result.eigenvalues.len(), 2);
        assert!((result.eigenvalues[0] - Complex64::new(2.0_f64, 1.0_f64)).norm() < 1e-10_f64);
        assert!((result.eigenvalues[1] - Complex64::new(-3.0_f64, 0.5_f64)).norm() < 1e-10_f64);
    }

    #[test]
    fn nonsymmetric_view_variants_match_owned() {
        let matrix =
            Array2::from_shape_vec((2, 2), vec![1.0_f64, 2.0_f64, -3.0_f64, 4.0_f64]).unwrap();
        let owned = nonsymmetric(&matrix).unwrap();
        let viewed = nonsymmetric_view(&matrix.view()).unwrap();
        assert_eq!(owned.eigenvalues.len(), viewed.eigenvalues.len());
        assert_eq!(owned.schur_vectors.dim(), viewed.schur_vectors.dim());

        let complex_matrix = matrix.mapv(|value| Complex64::new(value, 0.25_f64 * value));
        let complex_owned = nonsymmetric_complex(&complex_matrix).unwrap();
        let complex_viewed = nonsymmetric_complex_view(&complex_matrix.view()).unwrap();
        assert_eq!(complex_owned.eigenvalues.len(), complex_viewed.eigenvalues.len());
        assert_eq!(complex_owned.schur_vectors.dim(), complex_viewed.schur_vectors.dim());
    }

    #[test]
    fn nonsymmetric_triangular_matrix_matches_diagonal_eigenvalues() {
        let upper_triangular = Array2::from_shape_vec((3, 3), vec![
            4.0_f64, 1.0_f64, 2.0_f64, 0.0_f64, -3.0_f64, 5.0_f64, 0.0_f64, 0.0_f64, 2.5_f64,
        ])
        .unwrap();
        let result = nonsymmetric(&upper_triangular).unwrap();
        assert_eq!(result.eigenvalues.len(), 3);

        let mut real_parts = result.eigenvalues.iter().map(|value| value.re).collect::<Vec<_>>();
        real_parts.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap());
        assert!((real_parts[0] + 3.0_f64).abs() < 1e-8_f64);
        assert!((real_parts[1] - 2.5_f64).abs() < 1e-8_f64);
        assert!((real_parts[2] - 4.0_f64).abs() < 1e-8_f64);
    }

    #[test]
    fn balancing_changes_scale_but_preserves_shape() {
        let matrix = Array2::from_shape_vec((3, 3), vec![
            1.0_f64, 200.0_f64, 0.0_f64, //
            0.001_f64, 2.0_f64, 30.0_f64, //
            0.0_f64, 0.02_f64, 3.0_f64,
        ])
        .unwrap();
        let config = NonsymmetricEigenConfig::default();
        let (balanced, diagonal) = balance_nonsymmetric(&matrix, &config).unwrap();
        assert_eq!(balanced.dim(), matrix.dim());
        assert_eq!(diagonal.len(), matrix.nrows());
        assert!(diagonal.iter().all(|value| value.is_finite() && *value > 0.0_f64));
    }

    #[test]
    fn nonsymmetric_bi_produces_left_and_right_vectors() {
        let matrix =
            Array2::from_shape_vec((2, 2), vec![2.0_f64, 4.0_f64, -1.0_f64, 3.0_f64]).unwrap();
        let result = nonsymmetric_bi(&matrix, &NonsymmetricEigenConfig::default()).unwrap();
        assert_eq!(result.eigenvalues.len(), 2);
        assert_eq!(result.right_eigenvectors.dim(), (2, 2));
        assert_eq!(result.left_eigenvectors.dim(), (2, 2));
        assert_eq!(result.balancing_diagonal.len(), 2);

        let reference = nonsymmetric(&result.balanced_matrix).unwrap();
        let mut result_values = result.eigenvalues.iter().copied().collect::<Vec<_>>();
        let mut reference_values = reference.eigenvalues.iter().copied().collect::<Vec<_>>();
        result_values.sort_by(|lhs, rhs| {
            lhs.re
                .partial_cmp(&rhs.re)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(lhs.im.partial_cmp(&rhs.im).unwrap_or(std::cmp::Ordering::Equal))
        });
        reference_values.sort_by(|lhs, rhs| {
            lhs.re
                .partial_cmp(&rhs.re)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(lhs.im.partial_cmp(&rhs.im).unwrap_or(std::cmp::Ordering::Equal))
        });
        for col in 0..2 {
            assert!((result_values[col] - reference_values[col]).norm() < 1e-8_f64);
            let right_norm = result
                .right_eigenvectors
                .column(col)
                .iter()
                .map(Complex64::norm_sqr)
                .sum::<f64>()
                .sqrt();
            let left_norm = result
                .left_eigenvectors
                .column(col)
                .iter()
                .map(Complex64::norm_sqr)
                .sum::<f64>()
                .sqrt();
            assert!((right_norm - 1.0_f64).abs() < 1e-8_f64);
            assert!((left_norm - 1.0_f64).abs() < 1e-8_f64);
        }
    }

    #[test]
    fn real_f32_paths_match_expected() {
        let symmetric_matrix =
            Array2::from_shape_vec((2, 2), vec![4.0_f32, 1.0_f32, 1.0_f32, 3.0_f32]).unwrap();
        let symmetric_result = symmetric(&symmetric_matrix).unwrap();
        assert_eq!(symmetric_result.eigenvalues.len(), 2);
        assert_eq!(symmetric_result.eigenvectors.dim(), (2, 2));

        let b = Array2::from_shape_vec((2, 2), vec![2.0_f32, 0.0_f32, 0.0_f32, 1.0_f32]).unwrap();
        let generalized_result = generalized(&symmetric_matrix, &b).unwrap();
        assert_eq!(generalized_result.eigenvalues.len(), 2);
        assert_eq!(generalized_result.eigenvectors.dim(), (2, 2));

        let rotation =
            Array2::from_shape_vec((2, 2), vec![0.0_f32, -1.0_f32, 1.0_f32, 0.0_f32]).unwrap();
        let nonsymmetric_result = nonsymmetric(&rotation).unwrap();
        assert_eq!(nonsymmetric_result.eigenvalues.len(), 2);
        let imag_sum =
            nonsymmetric_result.eigenvalues.iter().map(|value| value.im.abs()).sum::<f32>();
        assert!(imag_sum > 1.0_f32);

        let config = NonsymmetricEigenConfig::<f32>::default();
        let (balanced, balancing_diagonal) = balance_nonsymmetric(&rotation, &config).unwrap();
        assert_eq!(balanced.dim(), rotation.dim());
        assert!(balancing_diagonal.iter().all(|value| value.is_finite() && *value > 0.0_f32));

        let bi_matrix =
            Array2::from_shape_vec((2, 2), vec![2.0_f32, 4.0_f32, -1.0_f32, 3.0_f32]).unwrap();
        let bi_result = nonsymmetric_bi(&bi_matrix, &config).unwrap();
        assert_eq!(bi_result.eigenvalues.len(), 2);
        assert_eq!(bi_result.right_eigenvectors.dim(), (2, 2));
        assert_eq!(bi_result.left_eigenvectors.dim(), (2, 2));
    }
}
