//! PCA-based dimensionality reduction for embedding matrices.
//!
//! Fit a PCA basis on a representative embedding matrix, then project query and corpus vectors into
//! the lower-dimensional score space for cheaper storage and scoring. This wraps
//! `nabled_ml::pca::compute_pca` + `nabled_ml::pca::transform`; the fitted basis is returned as a
//! [`PcaModel`] so it can be reused across many `compress` calls. Reducing dimensionality is lossy
//! and changes scores slightly, so refit when the corpus distribution shifts.

use nabled_core::scalar::NabledReal;
use nabled_ml::pca;
use ndarray::{Array2, ArrayBase, ArrayView2, Data, Ix2};

use crate::error::EmbeddingError;

/// A fitted PCA basis usable to compress embeddings into score space.
pub type PcaModel<T> = pca::NdarrayPCAResult<T>;

/// Fit a PCA basis that projects embeddings down to `dims` components.
///
/// # Errors
/// Returns [`EmbeddingError::EmptyInput`] for empty input,
/// [`EmbeddingError::InvalidInput`] when `dims` is zero, and
/// [`EmbeddingError::DecompositionFailed`] when the underlying SVD does not converge.
#[cfg(feature = "lapack-provider")]
pub fn fit_pca<T>(embeddings: &Array2<T>, dims: usize) -> Result<PcaModel<T>, EmbeddingError>
where
    T: NabledReal + ndarray_linalg::Lapack<Real = T>,
{
    Ok(pca::compute_pca(embeddings, Some(dims))?)
}

/// Fit a PCA basis that projects embeddings down to `dims` components.
///
/// # Errors
/// Returns [`EmbeddingError::EmptyInput`] for empty input,
/// [`EmbeddingError::InvalidInput`] when `dims` is zero, and
/// [`EmbeddingError::DecompositionFailed`] when the underlying SVD does not converge.
#[cfg(not(feature = "lapack-provider"))]
pub fn fit_pca<T: nabled_linalg::svd::SvdInternalScalar>(
    embeddings: &Array2<T>,
    dims: usize,
) -> Result<PcaModel<T>, EmbeddingError> {
    Ok(pca::compute_pca(embeddings, Some(dims))?)
}

/// Project `embeddings` into the fitted PCA score space.
#[must_use]
pub fn compress<T: NabledReal>(embeddings: &Array2<T>, model: &PcaModel<T>) -> Array2<T> {
    pca::transform(embeddings, model)
}

/// Project an embedding matrix view into the fitted PCA score space.
#[must_use]
pub fn compress_view<T: NabledReal>(
    embeddings: &ArrayView2<'_, T>,
    model: &PcaModel<T>,
) -> Array2<T> {
    pca::transform_view(embeddings, model)
}

/// Project `embeddings` into the fitted PCA score space, writing into `output`.
///
/// `output` must be shaped `(embeddings.nrows(), model.components.nrows())`.
///
/// # Errors
/// Returns [`EmbeddingError::InvalidInput`] when the feature width, mean length, or `output`
/// shape do not line up with `model`.
pub fn compress_into<T, S>(
    embeddings: &ArrayBase<S, Ix2>,
    model: &PcaModel<T>,
    output: &mut Array2<T>,
) -> Result<(), EmbeddingError>
where
    T: NabledReal,
    S: Data<Elem = T>,
{
    pca::transform_from_components_view_into(
        &embeddings.view(),
        &model.components.view(),
        &model.mean.view(),
        output.view_mut(),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use ndarray::{Array2, arr2};

    use super::*;

    fn sample_f64() -> Array2<f64> {
        arr2(&[
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 4.0],
            [3.0, 4.0, 1.0],
            [4.0, 3.0, 2.0],
        ])
    }

    #[test]
    fn fit_and_compress_reduce_dimensionality() {
        let embeddings = sample_f64();
        let model = fit_pca(&embeddings, 2).unwrap();
        let compressed = compress(&embeddings, &model);
        assert_eq!(compressed.dim(), (4, 2));
    }

    #[test]
    fn compress_view_matches_owned() {
        let embeddings = sample_f64();
        let model = fit_pca(&embeddings, 2).unwrap();
        let owned = compress(&embeddings, &model);
        let viewed = compress_view(&embeddings.view(), &model);
        assert_eq!(owned, viewed);
    }

    #[test]
    fn compress_into_matches_allocating() {
        let embeddings = sample_f64();
        let model = fit_pca(&embeddings, 2).unwrap();
        let expected = compress(&embeddings, &model);
        let mut output = Array2::<f64>::zeros((embeddings.nrows(), 2));
        compress_into(&embeddings, &model, &mut output).unwrap();
        assert_eq!(output, expected);
    }

    #[test]
    fn compress_into_rejects_wrong_output_shape() {
        let embeddings = sample_f64();
        let model = fit_pca(&embeddings, 2).unwrap();
        let mut output = Array2::<f64>::zeros((embeddings.nrows(), 3));
        assert!(matches!(
            compress_into(&embeddings, &model, &mut output),
            Err(EmbeddingError::DimensionMismatch | EmbeddingError::InvalidInput(_))
        ));
    }

    #[test]
    fn fit_pca_rejects_zero_dims() {
        let embeddings = sample_f64();
        assert!(matches!(fit_pca(&embeddings, 0), Err(EmbeddingError::InvalidInput(_))));
    }

    #[test]
    fn fit_and_compress_support_f32() {
        let embeddings = arr2(&[[1.0_f32, 2.0, 3.0], [2.0, 1.0, 4.0], [3.0, 4.0, 1.0]]);
        let model = fit_pca(&embeddings, 2).unwrap();
        let compressed = compress(&embeddings, &model);
        assert_eq!(compressed.dim(), (3, 2));
    }
}
