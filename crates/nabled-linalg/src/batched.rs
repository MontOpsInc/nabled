//! Batched decomposition helpers over stacks of matrices.

#[cfg(any(not(feature = "magma-system"), feature = "lapack-provider"))]
use nabled_core::scalar::NabledReal;
use ndarray::{Array3, ArrayView3, Axis};

#[cfg(feature = "magma-system")]
use crate::provider::magma;
#[cfg(feature = "magma-system")]
use crate::provider::policy::MagmaProviderPolicy;
use crate::{cholesky, eigen, lu, qr, svd};

#[cfg(feature = "magma-system")]
fn map_magma_batched_qr_error(error: &'static str) -> qr::QRError {
    match error {
        "empty" => qr::QRError::EmptyMatrix,
        "convergence_failed" => qr::QRError::ConvergenceFailed,
        "non_finite" => qr::QRError::NumericalInstability,
        "bad_dimensions" | "invalid_dimensions" => {
            qr::QRError::InvalidDimensions("RHS length must equal matrix rows".to_string())
        }
        _ => qr::QRError::InvalidInput(error.to_string()),
    }
}

#[cfg(feature = "magma-system")]
fn map_magma_batched_lu_error(error: &'static str) -> lu::LUError {
    match error {
        "empty" => lu::LUError::EmptyMatrix,
        "not_square" => lu::LUError::NotSquare,
        "singular" => lu::LUError::SingularMatrix,
        "convergence_failed" => lu::LUError::ConvergenceFailed,
        "non_finite" => lu::LUError::NumericalInstability,
        "bad_dimensions" | "invalid_dimensions" => {
            lu::LUError::InvalidInput("RHS length must match matrix dimensions".to_string())
        }
        _ => lu::LUError::InvalidInput(error.to_string()),
    }
}

#[cfg(feature = "magma-system")]
fn map_magma_batched_cholesky_error(error: &'static str) -> cholesky::CholeskyError {
    match error {
        "empty" => cholesky::CholeskyError::EmptyMatrix,
        "not_square" => cholesky::CholeskyError::NotSquare,
        "not_positive_definite" => cholesky::CholeskyError::NotPositiveDefinite,
        "non_finite" => cholesky::CholeskyError::NumericalInstability,
        "bad_dimensions" | "invalid_dimensions" => cholesky::CholeskyError::InvalidInput(
            "RHS length must match matrix dimensions".to_string(),
        ),
        _ => cholesky::CholeskyError::InvalidInput(error.to_string()),
    }
}

#[cfg(feature = "magma-system")]
fn map_magma_batched_svd_error(error: &'static str) -> svd::SVDError {
    match error {
        "empty" => svd::SVDError::EmptyMatrix,
        "convergence_failed" => svd::SVDError::ConvergenceFailed,
        "non_finite" => svd::SVDError::InvalidInput("matrix must be finite".to_string()),
        _ => svd::SVDError::InvalidInput(error.to_string()),
    }
}

#[cfg(feature = "magma-system")]
fn map_magma_batched_eigen_error(error: &'static str) -> eigen::EigenError {
    match error {
        "empty" => eigen::EigenError::EmptyMatrix,
        "not_square" => eigen::EigenError::NotSquare,
        "not_symmetric" => eigen::EigenError::NotSymmetric,
        "invalid_dimensions" | "bad_dimensions" => eigen::EigenError::InvalidDimensions,
        "not_positive_definite" => eigen::EigenError::NotPositiveDefinite,
        "convergence_failed" => eigen::EigenError::ConvergenceFailed,
        _ => eigen::EigenError::NumericalInstability,
    }
}

/// Compute QR decomposition for each matrix in a batch.
///
/// Input shape is `(batch, rows, cols)`.
///
/// # Errors
/// Returns an error if the batch is empty or any per-matrix decomposition fails.
#[cfg(all(feature = "magma-system", feature = "lapack-provider"))]
pub fn qr<T>(
    matrices: &Array3<T>,
    config: &qr::QRConfig<T>,
) -> Result<Vec<qr::QRResult<T>>, qr::QRError>
where
    T: magma::MagmaRealBatched + ndarray_linalg::Lapack<Real = T> + std::ops::AddAssign,
{
    qr_view(&matrices.view(), config)
}

/// Compute QR decomposition for each matrix in a batch.
///
/// Input shape is `(batch, rows, cols)`.
///
/// # Errors
/// Returns an error if the batch is empty or any per-matrix decomposition fails.
#[cfg(all(feature = "magma-system", not(feature = "lapack-provider")))]
pub fn qr<T>(
    matrices: &Array3<T>,
    config: &qr::QRConfig<T>,
) -> Result<Vec<qr::QRResult<T>>, qr::QRError>
where
    T: magma::MagmaRealBatched,
{
    qr_view(&matrices.view(), config)
}

/// Compute QR decomposition for each matrix in a batch.
///
/// Input shape is `(batch, rows, cols)`.
///
/// # Errors
/// Returns an error if the batch is empty or any per-matrix decomposition fails.
#[cfg(all(feature = "lapack-provider", not(feature = "magma-system")))]
pub fn qr<T>(
    matrices: &Array3<T>,
    config: &qr::QRConfig<T>,
) -> Result<Vec<qr::QRResult<T>>, qr::QRError>
where
    T: NabledReal + ndarray_linalg::Lapack<Real = T> + std::ops::AddAssign,
{
    qr_view(&matrices.view(), config)
}

/// Compute QR decomposition for each matrix in a batch.
///
/// Input shape is `(batch, rows, cols)`.
///
/// # Errors
/// Returns an error if the batch is empty or any per-matrix decomposition fails.
#[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
pub fn qr<T: qr::QrInternalScalar>(
    matrices: &Array3<T>,
    config: &qr::QRConfig<T>,
) -> Result<Vec<qr::QRResult<T>>, qr::QRError> {
    qr_view(&matrices.view(), config)
}

/// Compute QR decomposition for each matrix in a batch view.
///
/// Input shape is `(batch, rows, cols)`.
///
/// # Errors
/// Returns an error if the batch is empty or any per-matrix decomposition fails.
#[cfg(all(feature = "magma-system", feature = "lapack-provider"))]
pub fn qr_view<T>(
    matrices: &ArrayView3<'_, T>,
    config: &qr::QRConfig<T>,
) -> Result<Vec<qr::QRResult<T>>, qr::QRError>
where
    T: magma::MagmaRealBatched + ndarray_linalg::Lapack<Real = T> + std::ops::AddAssign,
{
    if matrices.is_empty() || matrices.dim().0 == 0 {
        return Err(qr::QRError::EmptyMatrix);
    }
    let (batch_count, rows, cols) = matrices.dim();

    // Pivoted and underdetermined QR stay on per-slice paths.
    if config.use_pivoting
        || rows < cols
        || !MagmaProviderPolicy::prefer_batched_decomposition(batch_count, rows, cols)
    {
        let mut fallback = Vec::with_capacity(matrices.dim().0);
        for matrix in matrices.axis_iter(Axis(0)) {
            fallback.push(qr::decompose_view(&matrix, config)?);
        }
        return Ok(fallback);
    }

    match magma::qr_decompose_batched(matrices, config.rank_tolerance) {
        Ok(decompositions) => {
            let mut output = Vec::with_capacity(decompositions.len());
            for (q, r, rank) in decompositions {
                output.push(qr::QRResult { q, r, p: None, rank });
            }
            Ok(output)
        }
        Err(error) => {
            if MagmaProviderPolicy::fail_fast_mode() {
                return Err(map_magma_batched_qr_error(error));
            }
            // Runtime MAGMA init/provider failures fall back to per-slice decomposition.
            let mut fallback = Vec::with_capacity(batch_count);
            for matrix in matrices.axis_iter(Axis(0)) {
                fallback.push(qr::decompose_view(&matrix, config)?);
            }
            Ok(fallback)
        }
    }
}

/// Compute QR decomposition for each matrix in a batch view.
///
/// Input shape is `(batch, rows, cols)`.
///
/// # Errors
/// Returns an error if the batch is empty or any per-matrix decomposition fails.
#[cfg(all(feature = "magma-system", not(feature = "lapack-provider")))]
pub fn qr_view<T>(
    matrices: &ArrayView3<'_, T>,
    config: &qr::QRConfig<T>,
) -> Result<Vec<qr::QRResult<T>>, qr::QRError>
where
    T: magma::MagmaRealBatched,
{
    if matrices.is_empty() || matrices.dim().0 == 0 {
        return Err(qr::QRError::EmptyMatrix);
    }
    let (batch_count, rows, cols) = matrices.dim();

    // Pivoted and underdetermined QR stay on per-slice paths.
    if config.use_pivoting
        || rows < cols
        || !MagmaProviderPolicy::prefer_batched_decomposition(batch_count, rows, cols)
    {
        let mut fallback = Vec::with_capacity(matrices.dim().0);
        for matrix in matrices.axis_iter(Axis(0)) {
            fallback.push(qr::decompose_view(&matrix, config)?);
        }
        return Ok(fallback);
    }

    match magma::qr_decompose_batched(matrices, config.rank_tolerance) {
        Ok(decompositions) => {
            let mut output = Vec::with_capacity(decompositions.len());
            for (q, r, rank) in decompositions {
                output.push(qr::QRResult { q, r, p: None, rank });
            }
            Ok(output)
        }
        Err(error) => {
            if MagmaProviderPolicy::fail_fast_mode() {
                return Err(map_magma_batched_qr_error(error));
            }
            // Runtime MAGMA init/provider failures fall back to per-slice decomposition.
            let mut fallback = Vec::with_capacity(batch_count);
            for matrix in matrices.axis_iter(Axis(0)) {
                fallback.push(qr::decompose_view(&matrix, config)?);
            }
            Ok(fallback)
        }
    }
}

/// Compute QR decomposition for each matrix in a batch view.
///
/// Input shape is `(batch, rows, cols)`.
///
/// # Errors
/// Returns an error if the batch is empty or any per-matrix decomposition fails.
#[cfg(all(feature = "lapack-provider", not(feature = "magma-system")))]
pub fn qr_view<T>(
    matrices: &ArrayView3<'_, T>,
    config: &qr::QRConfig<T>,
) -> Result<Vec<qr::QRResult<T>>, qr::QRError>
where
    T: NabledReal + ndarray_linalg::Lapack<Real = T> + std::ops::AddAssign,
{
    if matrices.is_empty() || matrices.dim().0 == 0 {
        return Err(qr::QRError::EmptyMatrix);
    }
    let mut output = Vec::with_capacity(matrices.dim().0);
    for matrix in matrices.axis_iter(Axis(0)) {
        output.push(qr::decompose_view(&matrix, config)?);
    }
    Ok(output)
}

/// Compute QR decomposition for each matrix in a batch view.
///
/// Input shape is `(batch, rows, cols)`.
///
/// # Errors
/// Returns an error if the batch is empty or any per-matrix decomposition fails.
#[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
pub fn qr_view<T: qr::QrInternalScalar>(
    matrices: &ArrayView3<'_, T>,
    config: &qr::QRConfig<T>,
) -> Result<Vec<qr::QRResult<T>>, qr::QRError> {
    if matrices.is_empty() || matrices.dim().0 == 0 {
        return Err(qr::QRError::EmptyMatrix);
    }
    let mut output = Vec::with_capacity(matrices.dim().0);
    for matrix in matrices.axis_iter(Axis(0)) {
        output.push(qr::decompose_view(&matrix, config)?);
    }
    Ok(output)
}

/// Compute SVD decomposition for each matrix in a batch.
///
/// Input shape is `(batch, rows, cols)`.
///
/// # Errors
/// Returns an error if the batch is empty or any per-matrix decomposition fails.
#[cfg(all(feature = "magma-system", feature = "lapack-provider"))]
pub fn svd<T>(matrices: &Array3<T>) -> Result<Vec<svd::NdarraySVD<T>>, svd::SVDError>
where
    T: NabledReal + magma::MagmaReal + ndarray_linalg::Lapack<Real = T> + std::ops::AddAssign,
{
    svd_view(&matrices.view())
}

/// Compute SVD decomposition for each matrix in a batch.
///
/// Input shape is `(batch, rows, cols)`.
///
/// # Errors
/// Returns an error if the batch is empty or any per-matrix decomposition fails.
#[cfg(all(feature = "magma-system", not(feature = "lapack-provider")))]
pub fn svd<T: svd::SvdInternalScalar>(
    matrices: &Array3<T>,
) -> Result<Vec<svd::NdarraySVD<T>>, svd::SVDError> {
    svd_view(&matrices.view())
}

/// Compute SVD decomposition for each matrix in a batch.
///
/// Input shape is `(batch, rows, cols)`.
///
/// # Errors
/// Returns an error if the batch is empty or any per-matrix decomposition fails.
#[cfg(all(feature = "lapack-provider", not(feature = "magma-system")))]
pub fn svd<T>(matrices: &Array3<T>) -> Result<Vec<svd::NdarraySVD<T>>, svd::SVDError>
where
    T: NabledReal + ndarray_linalg::Lapack<Real = T> + std::ops::AddAssign,
{
    svd_view(&matrices.view())
}

/// Compute SVD decomposition for each matrix in a batch.
///
/// Input shape is `(batch, rows, cols)`.
///
/// # Errors
/// Returns an error if the batch is empty or any per-matrix decomposition fails.
#[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
pub fn svd<T: svd::SvdInternalScalar>(
    matrices: &Array3<T>,
) -> Result<Vec<svd::NdarraySVD<T>>, svd::SVDError> {
    svd_view(&matrices.view())
}

/// Compute SVD decomposition for each matrix in a batch view.
///
/// Input shape is `(batch, rows, cols)`.
///
/// # Errors
/// Returns an error if the batch is empty or any per-matrix decomposition fails.
#[cfg(all(feature = "magma-system", feature = "lapack-provider"))]
pub fn svd_view<T>(matrices: &ArrayView3<'_, T>) -> Result<Vec<svd::NdarraySVD<T>>, svd::SVDError>
where
    T: NabledReal + magma::MagmaReal + ndarray_linalg::Lapack<Real = T> + std::ops::AddAssign,
{
    if matrices.is_empty() || matrices.dim().0 == 0 {
        return Err(svd::SVDError::EmptyMatrix);
    }
    let (batch_count, rows, cols) = matrices.dim();

    if MagmaProviderPolicy::prefer_batched_decomposition(batch_count, rows, cols) {
        let mut output = Vec::with_capacity(batch_count);
        let mut provider_error = None;
        for matrix in matrices.axis_iter(Axis(0)) {
            match magma::svd_decompose(&matrix) {
                Ok((u, singular_values, vt)) => {
                    output.push(svd::NdarraySVD { u, singular_values, vt });
                }
                Err(error) => {
                    provider_error = Some(error);
                    break;
                }
            }
        }
        if let Some(error) = provider_error {
            if MagmaProviderPolicy::fail_fast_mode() {
                return Err(map_magma_batched_svd_error(error));
            }
        } else {
            return Ok(output);
        }
    }

    let mut output = Vec::with_capacity(batch_count);
    for matrix in matrices.axis_iter(Axis(0)) {
        output.push(svd::decompose_view(&matrix)?);
    }
    Ok(output)
}

/// Compute SVD decomposition for each matrix in a batch view.
///
/// Input shape is `(batch, rows, cols)`.
///
/// # Errors
/// Returns an error if the batch is empty or any per-matrix decomposition fails.
#[cfg(all(feature = "magma-system", not(feature = "lapack-provider")))]
pub fn svd_view<T: svd::SvdInternalScalar>(
    matrices: &ArrayView3<'_, T>,
) -> Result<Vec<svd::NdarraySVD<T>>, svd::SVDError> {
    if matrices.is_empty() || matrices.dim().0 == 0 {
        return Err(svd::SVDError::EmptyMatrix);
    }
    let (batch_count, rows, cols) = matrices.dim();

    if MagmaProviderPolicy::prefer_batched_decomposition(batch_count, rows, cols) {
        let mut output = Vec::with_capacity(batch_count);
        let mut provider_error = None;
        for matrix in matrices.axis_iter(Axis(0)) {
            match magma::svd_decompose(&matrix) {
                Ok((u, singular_values, vt)) => {
                    output.push(svd::NdarraySVD { u, singular_values, vt });
                }
                Err(error) => {
                    provider_error = Some(error);
                    break;
                }
            }
        }
        if let Some(error) = provider_error {
            if MagmaProviderPolicy::fail_fast_mode() {
                return Err(map_magma_batched_svd_error(error));
            }
        } else {
            return Ok(output);
        }
    }

    let mut output = Vec::with_capacity(batch_count);
    for matrix in matrices.axis_iter(Axis(0)) {
        output.push(svd::decompose_view(&matrix)?);
    }
    Ok(output)
}

/// Compute SVD decomposition for each matrix in a batch view.
///
/// Input shape is `(batch, rows, cols)`.
///
/// # Errors
/// Returns an error if the batch is empty or any per-matrix decomposition fails.
#[cfg(all(feature = "lapack-provider", not(feature = "magma-system")))]
pub fn svd_view<T>(matrices: &ArrayView3<'_, T>) -> Result<Vec<svd::NdarraySVD<T>>, svd::SVDError>
where
    T: NabledReal + ndarray_linalg::Lapack<Real = T> + std::ops::AddAssign,
{
    if matrices.is_empty() || matrices.dim().0 == 0 {
        return Err(svd::SVDError::EmptyMatrix);
    }
    let mut output = Vec::with_capacity(matrices.dim().0);
    for matrix in matrices.axis_iter(Axis(0)) {
        output.push(svd::decompose_view(&matrix)?);
    }
    Ok(output)
}

/// Compute SVD decomposition for each matrix in a batch view.
///
/// Input shape is `(batch, rows, cols)`.
///
/// # Errors
/// Returns an error if the batch is empty or any per-matrix decomposition fails.
#[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
pub fn svd_view<T: svd::SvdInternalScalar>(
    matrices: &ArrayView3<'_, T>,
) -> Result<Vec<svd::NdarraySVD<T>>, svd::SVDError> {
    if matrices.is_empty() || matrices.dim().0 == 0 {
        return Err(svd::SVDError::EmptyMatrix);
    }
    let mut output = Vec::with_capacity(matrices.dim().0);
    for matrix in matrices.axis_iter(Axis(0)) {
        output.push(svd::decompose_view(&matrix)?);
    }
    Ok(output)
}

/// Compute LU decomposition for each matrix in a batch.
///
/// Input shape is `(batch, rows, cols)`.
///
/// # Errors
/// Returns an error if the batch is empty or any per-matrix decomposition fails.
#[cfg(feature = "magma-system")]
pub fn lu<T>(matrices: &Array3<T>) -> Result<Vec<lu::NdarrayLUResult<T>>, lu::LUError>
where
    T: lu::LuProviderScalar + magma::MagmaRealBatched,
{
    Ok(lu_view_with_metadata(&matrices.view())?
        .into_iter()
        .map(|(result, _, _)| result)
        .collect())
}

/// Compute LU decomposition for each matrix in a batch.
///
/// Input shape is `(batch, rows, cols)`.
///
/// # Errors
/// Returns an error if the batch is empty or any per-matrix decomposition fails.
#[cfg(all(feature = "lapack-provider", not(feature = "magma-system")))]
pub fn lu<T>(matrices: &Array3<T>) -> Result<Vec<lu::NdarrayLUResult<T>>, lu::LUError>
where
    T: NabledReal + ndarray_linalg::Lapack,
{
    Ok(lu_view_with_metadata(&matrices.view())?
        .into_iter()
        .map(|(result, _, _)| result)
        .collect())
}

/// Compute LU decomposition for each matrix in a batch.
///
/// Input shape is `(batch, rows, cols)`.
///
/// # Errors
/// Returns an error if the batch is empty or any per-matrix decomposition fails.
#[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
pub fn lu<T: NabledReal>(matrices: &Array3<T>) -> Result<Vec<lu::NdarrayLUResult<T>>, lu::LUError> {
    Ok(lu_view_with_metadata(&matrices.view())?
        .into_iter()
        .map(|(result, _, _)| result)
        .collect())
}

/// Compute LU decomposition with metadata for each matrix in a batch.
///
/// Input shape is `(batch, rows, cols)`.
///
/// # Errors
/// Returns an error if the batch is empty or any per-matrix decomposition fails.
#[cfg(feature = "magma-system")]
#[expect(clippy::type_complexity)]
pub fn lu_with_metadata<T>(
    matrices: &Array3<T>,
) -> Result<Vec<(lu::NdarrayLUResult<T>, Vec<usize>, i8)>, lu::LUError>
where
    T: lu::LuProviderScalar + magma::MagmaRealBatched,
{
    lu_view_with_metadata(&matrices.view())
}

/// Compute LU decomposition with metadata for each matrix in a batch.
///
/// Input shape is `(batch, rows, cols)`.
///
/// # Errors
/// Returns an error if the batch is empty or any per-matrix decomposition fails.
#[cfg(all(feature = "lapack-provider", not(feature = "magma-system")))]
#[expect(clippy::type_complexity)]
pub fn lu_with_metadata<T>(
    matrices: &Array3<T>,
) -> Result<Vec<(lu::NdarrayLUResult<T>, Vec<usize>, i8)>, lu::LUError>
where
    T: NabledReal + ndarray_linalg::Lapack,
{
    lu_view_with_metadata(&matrices.view())
}

/// Compute LU decomposition with metadata for each matrix in a batch.
///
/// Input shape is `(batch, rows, cols)`.
///
/// # Errors
/// Returns an error if the batch is empty or any per-matrix decomposition fails.
#[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
#[expect(clippy::type_complexity)]
pub fn lu_with_metadata<T: NabledReal>(
    matrices: &Array3<T>,
) -> Result<Vec<(lu::NdarrayLUResult<T>, Vec<usize>, i8)>, lu::LUError> {
    lu_view_with_metadata(&matrices.view())
}

/// Compute LU decomposition for each matrix in a batch view.
///
/// Input shape is `(batch, rows, cols)`.
///
/// # Errors
/// Returns an error if the batch is empty or any per-matrix decomposition fails.
#[cfg(feature = "magma-system")]
pub fn lu_view<T>(matrices: &ArrayView3<'_, T>) -> Result<Vec<lu::NdarrayLUResult<T>>, lu::LUError>
where
    T: lu::LuProviderScalar + magma::MagmaRealBatched,
{
    Ok(lu_view_with_metadata(matrices)?.into_iter().map(|(result, _, _)| result).collect())
}

/// Compute LU decomposition with metadata for each matrix in a batch view.
///
/// Input shape is `(batch, rows, cols)`.
///
/// # Errors
/// Returns an error if the batch is empty or any per-matrix decomposition fails.
#[cfg(feature = "magma-system")]
#[expect(clippy::type_complexity)]
pub fn lu_view_with_metadata<T>(
    matrices: &ArrayView3<'_, T>,
) -> Result<Vec<(lu::NdarrayLUResult<T>, Vec<usize>, i8)>, lu::LUError>
where
    T: lu::LuProviderScalar + magma::MagmaRealBatched,
{
    if matrices.is_empty() || matrices.dim().0 == 0 {
        return Err(lu::LUError::EmptyMatrix);
    }
    let (batch_count, rows, cols) = matrices.dim();
    if !MagmaProviderPolicy::prefer_batched_decomposition(batch_count, rows, cols) {
        let mut output = Vec::with_capacity(batch_count);
        for matrix in matrices.axis_iter(Axis(0)) {
            output.push(lu::decompose_view_with_metadata(&matrix)?);
        }
        return Ok(output);
    }

    match magma::lu_decompose_batched(matrices) {
        Ok(factors) => {
            let mut output = Vec::with_capacity(factors.len());
            for (l, u, pivots, sign) in factors {
                output.push((lu::NdarrayLUResult { l, u }, pivots, sign));
            }
            Ok(output)
        }
        Err(error) => {
            if MagmaProviderPolicy::fail_fast_mode() {
                return Err(map_magma_batched_lu_error(error));
            }
            // Runtime MAGMA init/provider failures fall back to per-slice decomposition.
            let mut fallback = Vec::with_capacity(batch_count);
            for matrix in matrices.axis_iter(Axis(0)) {
                fallback.push(lu::decompose_view_with_metadata(&matrix)?);
            }
            Ok(fallback)
        }
    }
}

/// Compute LU decomposition for each matrix in a batch view.
///
/// Input shape is `(batch, rows, cols)`.
///
/// # Errors
/// Returns an error if the batch is empty or any per-matrix decomposition fails.
#[cfg(all(feature = "lapack-provider", not(feature = "magma-system")))]
pub fn lu_view<T>(matrices: &ArrayView3<'_, T>) -> Result<Vec<lu::NdarrayLUResult<T>>, lu::LUError>
where
    T: NabledReal + ndarray_linalg::Lapack,
{
    Ok(lu_view_with_metadata(matrices)?.into_iter().map(|(result, _, _)| result).collect())
}

/// Compute LU decomposition with metadata for each matrix in a batch view.
///
/// Input shape is `(batch, rows, cols)`.
///
/// # Errors
/// Returns an error if the batch is empty or any per-matrix decomposition fails.
#[cfg(all(feature = "lapack-provider", not(feature = "magma-system")))]
#[expect(clippy::type_complexity)]
pub fn lu_view_with_metadata<T>(
    matrices: &ArrayView3<'_, T>,
) -> Result<Vec<(lu::NdarrayLUResult<T>, Vec<usize>, i8)>, lu::LUError>
where
    T: NabledReal + ndarray_linalg::Lapack,
{
    if matrices.is_empty() || matrices.dim().0 == 0 {
        return Err(lu::LUError::EmptyMatrix);
    }
    let mut output = Vec::with_capacity(matrices.dim().0);
    for matrix in matrices.axis_iter(Axis(0)) {
        output.push(lu::decompose_view_with_metadata(&matrix)?);
    }
    Ok(output)
}

/// Compute LU decomposition for each matrix in a batch view.
///
/// Input shape is `(batch, rows, cols)`.
///
/// # Errors
/// Returns an error if the batch is empty or any per-matrix decomposition fails.
#[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
pub fn lu_view<T: NabledReal>(
    matrices: &ArrayView3<'_, T>,
) -> Result<Vec<lu::NdarrayLUResult<T>>, lu::LUError> {
    Ok(lu_view_with_metadata(matrices)?.into_iter().map(|(result, _, _)| result).collect())
}

/// Compute LU decomposition with metadata for each matrix in a batch view.
///
/// Input shape is `(batch, rows, cols)`.
///
/// # Errors
/// Returns an error if the batch is empty or any per-matrix decomposition fails.
#[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
#[expect(clippy::type_complexity)]
pub fn lu_view_with_metadata<T: NabledReal>(
    matrices: &ArrayView3<'_, T>,
) -> Result<Vec<(lu::NdarrayLUResult<T>, Vec<usize>, i8)>, lu::LUError> {
    if matrices.is_empty() || matrices.dim().0 == 0 {
        return Err(lu::LUError::EmptyMatrix);
    }
    let mut output = Vec::with_capacity(matrices.dim().0);
    for matrix in matrices.axis_iter(Axis(0)) {
        output.push(lu::decompose_view_with_metadata(&matrix)?);
    }
    Ok(output)
}

/// Compute Cholesky decomposition for each matrix in a batch.
///
/// Input shape is `(batch, rows, cols)`.
///
/// # Errors
/// Returns an error if the batch is empty or any per-matrix decomposition fails.
#[cfg(feature = "magma-system")]
pub fn cholesky<T>(
    matrices: &Array3<T>,
) -> Result<Vec<cholesky::NdarrayCholeskyResult<T>>, cholesky::CholeskyError>
where
    T: cholesky::CholeskyProviderScalar + magma::MagmaRealBatched,
{
    cholesky_view(&matrices.view())
}

/// Compute Cholesky decomposition for each matrix in a batch.
///
/// Input shape is `(batch, rows, cols)`.
///
/// # Errors
/// Returns an error if the batch is empty or any per-matrix decomposition fails.
#[cfg(all(feature = "lapack-provider", not(feature = "magma-system")))]
pub fn cholesky<T>(
    matrices: &Array3<T>,
) -> Result<Vec<cholesky::NdarrayCholeskyResult<T>>, cholesky::CholeskyError>
where
    T: cholesky::CholeskyProviderScalar,
{
    cholesky_view(&matrices.view())
}

/// Compute Cholesky decomposition for each matrix in a batch.
///
/// Input shape is `(batch, rows, cols)`.
///
/// # Errors
/// Returns an error if the batch is empty or any per-matrix decomposition fails.
#[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
pub fn cholesky<T: NabledReal>(
    matrices: &Array3<T>,
) -> Result<Vec<cholesky::NdarrayCholeskyResult<T>>, cholesky::CholeskyError> {
    cholesky_view(&matrices.view())
}

/// Compute Cholesky decomposition for each matrix in a batch view.
///
/// Input shape is `(batch, rows, cols)`.
///
/// # Errors
/// Returns an error if the batch is empty or any per-matrix decomposition fails.
#[cfg(feature = "magma-system")]
pub fn cholesky_view<T>(
    matrices: &ArrayView3<'_, T>,
) -> Result<Vec<cholesky::NdarrayCholeskyResult<T>>, cholesky::CholeskyError>
where
    T: cholesky::CholeskyProviderScalar + magma::MagmaRealBatched,
{
    if matrices.is_empty() || matrices.dim().0 == 0 {
        return Err(cholesky::CholeskyError::EmptyMatrix);
    }
    let (batch_count, rows, cols) = matrices.dim();
    if !MagmaProviderPolicy::prefer_batched_decomposition(batch_count, rows, cols) {
        let mut output = Vec::with_capacity(batch_count);
        for matrix in matrices.axis_iter(Axis(0)) {
            output.push(cholesky::decompose_view(&matrix)?);
        }
        return Ok(output);
    }

    match magma::cholesky_decompose_batched(matrices) {
        Ok(factors) => {
            let mut output = Vec::with_capacity(factors.len());
            for l in factors {
                output.push(cholesky::NdarrayCholeskyResult { l });
            }
            Ok(output)
        }
        Err(error) => {
            if MagmaProviderPolicy::fail_fast_mode() {
                return Err(map_magma_batched_cholesky_error(error));
            }
            // Runtime MAGMA init/provider failures fall back to per-slice decomposition.
            let mut fallback = Vec::with_capacity(batch_count);
            for matrix in matrices.axis_iter(Axis(0)) {
                fallback.push(cholesky::decompose_view(&matrix)?);
            }
            Ok(fallback)
        }
    }
}

/// Compute Cholesky decomposition for each matrix in a batch view.
///
/// Input shape is `(batch, rows, cols)`.
///
/// # Errors
/// Returns an error if the batch is empty or any per-matrix decomposition fails.
#[cfg(all(feature = "lapack-provider", not(feature = "magma-system")))]
pub fn cholesky_view<T>(
    matrices: &ArrayView3<'_, T>,
) -> Result<Vec<cholesky::NdarrayCholeskyResult<T>>, cholesky::CholeskyError>
where
    T: cholesky::CholeskyProviderScalar,
{
    if matrices.is_empty() || matrices.dim().0 == 0 {
        return Err(cholesky::CholeskyError::EmptyMatrix);
    }
    let mut output = Vec::with_capacity(matrices.dim().0);
    for matrix in matrices.axis_iter(Axis(0)) {
        output.push(cholesky::decompose_view(&matrix)?);
    }
    Ok(output)
}

/// Compute Cholesky decomposition for each matrix in a batch view.
///
/// Input shape is `(batch, rows, cols)`.
///
/// # Errors
/// Returns an error if the batch is empty or any per-matrix decomposition fails.
#[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
pub fn cholesky_view<T: NabledReal>(
    matrices: &ArrayView3<'_, T>,
) -> Result<Vec<cholesky::NdarrayCholeskyResult<T>>, cholesky::CholeskyError> {
    if matrices.is_empty() || matrices.dim().0 == 0 {
        return Err(cholesky::CholeskyError::EmptyMatrix);
    }
    let mut output = Vec::with_capacity(matrices.dim().0);
    for matrix in matrices.axis_iter(Axis(0)) {
        output.push(cholesky::decompose_view(&matrix)?);
    }
    Ok(output)
}

/// Compute symmetric eigen decomposition for each matrix in a batch.
///
/// Input shape is `(batch, n, n)`.
///
/// # Errors
/// Returns an error if the batch is empty or any per-matrix decomposition fails.
#[cfg(all(feature = "magma-system", feature = "lapack-provider"))]
pub fn symmetric_eigen<T>(
    matrices: &Array3<T>,
) -> Result<Vec<eigen::NdarrayEigenResult<T>>, eigen::EigenError>
where
    T: NabledReal + magma::MagmaReal + ndarray_linalg::Lapack<Real = T> + std::ops::AddAssign,
{
    symmetric_eigen_view(&matrices.view())
}

/// Compute symmetric eigen decomposition for each matrix in a batch.
///
/// Input shape is `(batch, n, n)`.
///
/// # Errors
/// Returns an error if the batch is empty or any per-matrix decomposition fails.
#[cfg(all(feature = "magma-system", not(feature = "lapack-provider")))]
pub fn symmetric_eigen<T: eigen::EigenInternalScalar>(
    matrices: &Array3<T>,
) -> Result<Vec<eigen::NdarrayEigenResult<T>>, eigen::EigenError> {
    symmetric_eigen_view(&matrices.view())
}

/// Compute symmetric eigen decomposition for each matrix in a batch.
///
/// Input shape is `(batch, n, n)`.
///
/// # Errors
/// Returns an error if the batch is empty or any per-matrix decomposition fails.
#[cfg(all(feature = "lapack-provider", not(feature = "magma-system")))]
pub fn symmetric_eigen<T>(
    matrices: &Array3<T>,
) -> Result<Vec<eigen::NdarrayEigenResult<T>>, eigen::EigenError>
where
    T: NabledReal + ndarray_linalg::Lapack<Real = T> + std::ops::AddAssign,
{
    symmetric_eigen_view(&matrices.view())
}

/// Compute symmetric eigen decomposition for each matrix in a batch.
///
/// Input shape is `(batch, n, n)`.
///
/// # Errors
/// Returns an error if the batch is empty or any per-matrix decomposition fails.
#[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
pub fn symmetric_eigen<T: eigen::EigenInternalScalar>(
    matrices: &Array3<T>,
) -> Result<Vec<eigen::NdarrayEigenResult<T>>, eigen::EigenError> {
    symmetric_eigen_view(&matrices.view())
}

/// Compute symmetric eigen decomposition for each matrix in a batch view.
///
/// Input shape is `(batch, n, n)`.
///
/// # Errors
/// Returns an error if the batch is empty or any per-matrix decomposition fails.
#[cfg(all(feature = "magma-system", feature = "lapack-provider"))]
pub fn symmetric_eigen_view<T>(
    matrices: &ArrayView3<'_, T>,
) -> Result<Vec<eigen::NdarrayEigenResult<T>>, eigen::EigenError>
where
    T: NabledReal + magma::MagmaReal + ndarray_linalg::Lapack<Real = T> + std::ops::AddAssign,
{
    if matrices.is_empty() || matrices.dim().0 == 0 {
        return Err(eigen::EigenError::EmptyMatrix);
    }
    let (batch_count, rows, cols) = matrices.dim();

    if rows == cols && MagmaProviderPolicy::prefer_batched_decomposition(batch_count, rows, cols) {
        let mut output = Vec::with_capacity(batch_count);
        let mut provider_error = None;
        for matrix in matrices.axis_iter(Axis(0)) {
            match magma::symmetric_eigen(&matrix) {
                Ok((eigenvalues, eigenvectors)) => {
                    output.push(eigen::NdarrayEigenResult { eigenvalues, eigenvectors });
                }
                Err(error) => {
                    provider_error = Some(error);
                    break;
                }
            }
        }
        if let Some(error) = provider_error {
            if MagmaProviderPolicy::fail_fast_mode() {
                return Err(map_magma_batched_eigen_error(error));
            }
        } else {
            return Ok(output);
        }
    }

    let mut output = Vec::with_capacity(batch_count);
    for matrix in matrices.axis_iter(Axis(0)) {
        output.push(eigen::symmetric_view(&matrix)?);
    }
    Ok(output)
}

/// Compute symmetric eigen decomposition for each matrix in a batch view.
///
/// Input shape is `(batch, n, n)`.
///
/// # Errors
/// Returns an error if the batch is empty or any per-matrix decomposition fails.
#[cfg(all(feature = "magma-system", not(feature = "lapack-provider")))]
pub fn symmetric_eigen_view<T: eigen::EigenInternalScalar>(
    matrices: &ArrayView3<'_, T>,
) -> Result<Vec<eigen::NdarrayEigenResult<T>>, eigen::EigenError> {
    if matrices.is_empty() || matrices.dim().0 == 0 {
        return Err(eigen::EigenError::EmptyMatrix);
    }
    let (batch_count, rows, cols) = matrices.dim();

    if rows == cols && MagmaProviderPolicy::prefer_batched_decomposition(batch_count, rows, cols) {
        let mut output = Vec::with_capacity(batch_count);
        let mut provider_error = None;
        for matrix in matrices.axis_iter(Axis(0)) {
            match magma::symmetric_eigen(&matrix) {
                Ok((eigenvalues, eigenvectors)) => {
                    output.push(eigen::NdarrayEigenResult { eigenvalues, eigenvectors });
                }
                Err(error) => {
                    provider_error = Some(error);
                    break;
                }
            }
        }
        if let Some(error) = provider_error {
            if MagmaProviderPolicy::fail_fast_mode() {
                return Err(map_magma_batched_eigen_error(error));
            }
        } else {
            return Ok(output);
        }
    }

    let mut output = Vec::with_capacity(batch_count);
    for matrix in matrices.axis_iter(Axis(0)) {
        output.push(eigen::symmetric_view(&matrix)?);
    }
    Ok(output)
}

/// Compute symmetric eigen decomposition for each matrix in a batch view.
///
/// Input shape is `(batch, n, n)`.
///
/// # Errors
/// Returns an error if the batch is empty or any per-matrix decomposition fails.
#[cfg(all(feature = "lapack-provider", not(feature = "magma-system")))]
pub fn symmetric_eigen_view<T>(
    matrices: &ArrayView3<'_, T>,
) -> Result<Vec<eigen::NdarrayEigenResult<T>>, eigen::EigenError>
where
    T: NabledReal + ndarray_linalg::Lapack<Real = T> + std::ops::AddAssign,
{
    if matrices.is_empty() || matrices.dim().0 == 0 {
        return Err(eigen::EigenError::EmptyMatrix);
    }
    let mut output = Vec::with_capacity(matrices.dim().0);
    for matrix in matrices.axis_iter(Axis(0)) {
        output.push(eigen::symmetric_view(&matrix)?);
    }
    Ok(output)
}

/// Compute symmetric eigen decomposition for each matrix in a batch view.
///
/// Input shape is `(batch, n, n)`.
///
/// # Errors
/// Returns an error if the batch is empty or any per-matrix decomposition fails.
#[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
pub fn symmetric_eigen_view<T: eigen::EigenInternalScalar>(
    matrices: &ArrayView3<'_, T>,
) -> Result<Vec<eigen::NdarrayEigenResult<T>>, eigen::EigenError> {
    if matrices.is_empty() || matrices.dim().0 == 0 {
        return Err(eigen::EigenError::EmptyMatrix);
    }
    let mut output = Vec::with_capacity(matrices.dim().0);
    for matrix in matrices.axis_iter(Axis(0)) {
        output.push(eigen::symmetric_view(&matrix)?);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use ndarray::{Array2, Array3};

    use super::*;

    #[test]
    fn batched_qr_reconstructs_inputs() {
        let matrices = Array3::from_shape_vec((2, 2, 2), vec![
            1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64, //
            2.0_f64, 0.0_f64, 1.0_f64, 2.0_f64,
        ])
        .expect("valid shape");
        let results = qr(&matrices, &qr::QRConfig::default()).expect("batched qr");
        assert_eq!(results.len(), 2);

        for (batch_idx, result) in results.iter().enumerate() {
            let original = matrices.index_axis(Axis(0), batch_idx).to_owned();
            let reconstructed = qr::reconstruct_matrix(result);
            for row in 0..2 {
                for col in 0..2 {
                    assert!((original[[row, col]] - reconstructed[[row, col]]).abs() < 1e-8_f64);
                }
            }
        }
    }

    #[test]
    fn batched_svd_matches_single_path() {
        let matrices = Array3::from_shape_vec((2, 2, 2), vec![
            3.0_f64, 0.0_f64, 0.0_f64, 2.0_f64, //
            1.0_f64, 2.0_f64, 2.0_f64, 4.0_f64,
        ])
        .expect("valid shape");
        let batch = svd(&matrices).expect("batched svd");
        assert_eq!(batch.len(), 2);

        for (batch_idx, decomposition) in batch.iter().enumerate() {
            let matrix = matrices.index_axis(Axis(0), batch_idx).to_owned();
            let direct = svd::decompose(&matrix).expect("direct svd");
            assert_eq!(decomposition.singular_values.len(), direct.singular_values.len());
            for i in 0..decomposition.singular_values.len() {
                assert!(
                    (decomposition.singular_values[i] - direct.singular_values[i]).abs() < 1e-8_f64
                );
            }
        }
    }

    #[test]
    fn batched_lu_cholesky_and_eigen_work() {
        let lu_matrices = Array3::from_shape_vec((2, 2, 2), vec![
            4.0_f64, 3.0_f64, 6.0_f64, 3.0_f64, //
            2.0_f64, 1.0_f64, 5.0_f64, 3.0_f64,
        ])
        .expect("valid shape");
        let lu_results = lu(&lu_matrices).expect("batched lu");
        let lu_results_with_metadata = lu_with_metadata(&lu_matrices).expect("batched lu metadata");
        assert_eq!(lu_results.len(), 2);
        assert_eq!(lu_results_with_metadata.len(), 2);
        for (batch_idx, (result, pivots, permutation_sign)) in
            lu_results_with_metadata.iter().enumerate()
        {
            assert_eq!(result.l, lu_results[batch_idx].l);
            assert_eq!(result.u, lu_results[batch_idx].u);
            assert_eq!(pivots.len(), 2);
            assert!(matches!(*permutation_sign, -1 | 1));
        }

        let spd = Array3::from_shape_vec((2, 2, 2), vec![
            4.0_f64, 1.0_f64, 1.0_f64, 3.0_f64, //
            9.0_f64, 3.0_f64, 3.0_f64, 5.0_f64,
        ])
        .expect("valid shape");
        let chol_results = cholesky(&spd).expect("batched cholesky");
        let eig_results = symmetric_eigen(&spd).expect("batched eigen");
        assert_eq!(chol_results.len(), 2);
        assert_eq!(eig_results.len(), 2);

        let first = Array2::from_shape_vec((2, 2), vec![4.0_f64, 1.0_f64, 1.0_f64, 3.0_f64])
            .expect("valid shape");
        let reconstructed = chol_results[0].l.dot(&chol_results[0].l.t());
        for row in 0..2 {
            for col in 0..2 {
                assert!((first[[row, col]] - reconstructed[[row, col]]).abs() < 1e-8_f64);
            }
        }
        assert!(eig_results[0].eigenvalues[0] >= eig_results[0].eigenvalues[1]);
    }

    #[test]
    fn batched_decompositions_reject_empty_batch() {
        let empty = Array3::<f64>::zeros((0, 2, 2));
        assert!(matches!(qr(&empty, &qr::QRConfig::default()), Err(qr::QRError::EmptyMatrix)));
        assert!(matches!(svd(&empty), Err(svd::SVDError::EmptyMatrix)));
        assert!(matches!(lu(&empty), Err(lu::LUError::EmptyMatrix)));
        assert!(matches!(cholesky(&empty), Err(cholesky::CholeskyError::EmptyMatrix)));
        assert!(matches!(symmetric_eigen(&empty), Err(eigen::EigenError::EmptyMatrix)));
    }
}
