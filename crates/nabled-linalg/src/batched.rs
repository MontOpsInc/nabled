//! Batched decomposition helpers over stacks of matrices.

use nabled_core::scalar::NabledReal;
use ndarray::{Array3, ArrayView3, Axis};

use crate::{cholesky, eigen, lu, qr, svd};

/// Compute QR decomposition for each matrix in a batch.
///
/// Input shape is `(batch, rows, cols)`.
///
/// # Errors
/// Returns an error if the batch is empty or any per-matrix decomposition fails.
#[cfg(feature = "lapack-provider")]
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
#[cfg(not(feature = "lapack-provider"))]
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
#[cfg(feature = "lapack-provider")]
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
#[cfg(not(feature = "lapack-provider"))]
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
#[cfg(feature = "lapack-provider")]
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
#[cfg(not(feature = "lapack-provider"))]
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
#[cfg(feature = "lapack-provider")]
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
#[cfg(not(feature = "lapack-provider"))]
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
#[cfg(feature = "lapack-provider")]
pub fn lu<T>(matrices: &Array3<T>) -> Result<Vec<lu::NdarrayLUResult<T>>, lu::LUError>
where
    T: NabledReal + ndarray_linalg::Lapack,
{
    lu_view(&matrices.view())
}

/// Compute LU decomposition for each matrix in a batch.
///
/// Input shape is `(batch, rows, cols)`.
///
/// # Errors
/// Returns an error if the batch is empty or any per-matrix decomposition fails.
#[cfg(not(feature = "lapack-provider"))]
pub fn lu<T: NabledReal>(matrices: &Array3<T>) -> Result<Vec<lu::NdarrayLUResult<T>>, lu::LUError> {
    lu_view(&matrices.view())
}

/// Compute LU decomposition for each matrix in a batch view.
///
/// Input shape is `(batch, rows, cols)`.
///
/// # Errors
/// Returns an error if the batch is empty or any per-matrix decomposition fails.
#[cfg(feature = "lapack-provider")]
pub fn lu_view<T>(matrices: &ArrayView3<'_, T>) -> Result<Vec<lu::NdarrayLUResult<T>>, lu::LUError>
where
    T: NabledReal + ndarray_linalg::Lapack,
{
    if matrices.is_empty() || matrices.dim().0 == 0 {
        return Err(lu::LUError::EmptyMatrix);
    }
    let mut output = Vec::with_capacity(matrices.dim().0);
    for matrix in matrices.axis_iter(Axis(0)) {
        output.push(lu::decompose_view(&matrix)?);
    }
    Ok(output)
}

/// Compute LU decomposition for each matrix in a batch view.
///
/// Input shape is `(batch, rows, cols)`.
///
/// # Errors
/// Returns an error if the batch is empty or any per-matrix decomposition fails.
#[cfg(not(feature = "lapack-provider"))]
pub fn lu_view<T: NabledReal>(
    matrices: &ArrayView3<'_, T>,
) -> Result<Vec<lu::NdarrayLUResult<T>>, lu::LUError> {
    if matrices.is_empty() || matrices.dim().0 == 0 {
        return Err(lu::LUError::EmptyMatrix);
    }
    let mut output = Vec::with_capacity(matrices.dim().0);
    for matrix in matrices.axis_iter(Axis(0)) {
        output.push(lu::decompose_view(&matrix)?);
    }
    Ok(output)
}

/// Compute Cholesky decomposition for each matrix in a batch.
///
/// Input shape is `(batch, rows, cols)`.
///
/// # Errors
/// Returns an error if the batch is empty or any per-matrix decomposition fails.
#[cfg(any(feature = "lapack-provider", feature = "magma-system"))]
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
#[cfg(any(feature = "lapack-provider", feature = "magma-system"))]
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
#[cfg(feature = "lapack-provider")]
pub fn symmetric_eigen<T>(
    matrices: &Array3<T>,
) -> Result<Vec<eigen::NdarrayEigenResult<T>>, eigen::EigenError>
where
    T: NabledReal + ndarray_linalg::Lapack<Real = T>,
{
    symmetric_eigen_view(&matrices.view())
}

/// Compute symmetric eigen decomposition for each matrix in a batch.
///
/// Input shape is `(batch, n, n)`.
///
/// # Errors
/// Returns an error if the batch is empty or any per-matrix decomposition fails.
#[cfg(not(feature = "lapack-provider"))]
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
#[cfg(feature = "lapack-provider")]
pub fn symmetric_eigen_view<T>(
    matrices: &ArrayView3<'_, T>,
) -> Result<Vec<eigen::NdarrayEigenResult<T>>, eigen::EigenError>
where
    T: NabledReal + ndarray_linalg::Lapack<Real = T>,
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
#[cfg(not(feature = "lapack-provider"))]
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
        assert_eq!(lu_results.len(), 2);

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
