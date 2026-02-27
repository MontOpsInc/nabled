//! Internal PCA kernel traits and backend adapters.

use std::marker::PhantomData;

use nalgebra::{DMatrix, DVector, RealField};
use ndarray::{Array1, Array2};
use num_traits::Float;

use crate::interop::{nalgebra_to_ndarray, ndarray_to_nalgebra};
use crate::pca::{NalgebraPCAResult, NdarrayPCAResult, PCAError};
use crate::stats::nalgebra_stats;

/// Internal kernel trait for PCA operations.
pub(crate) trait PcaKernel {
    type Matrix;
    type Output;

    fn compute_pca(
        matrix: &Self::Matrix,
        n_components: Option<usize>,
    ) -> Result<Self::Output, PCAError>;
}

/// Nalgebra-backed PCA kernel.
pub(crate) struct NalgebraPcaKernel<T>(PhantomData<T>);

impl<T> PcaKernel for NalgebraPcaKernel<T>
where
    T: RealField + Copy + Float + num_traits::NumCast,
{
    type Matrix = DMatrix<T>;
    type Output = NalgebraPCAResult<T>;

    #[inline]
    fn compute_pca(
        matrix: &Self::Matrix,
        n_components: Option<usize>,
    ) -> Result<Self::Output, PCAError> {
        compute_nalgebra_pca_with_svd(
            matrix,
            n_components,
            crate::backend::svd::compute_nalgebra_svd,
        )
    }
}

/// Ndarray-backed PCA kernel.
pub(crate) struct NdarrayPcaKernel<T>(PhantomData<T>);

impl<T> PcaKernel for NdarrayPcaKernel<T>
where
    T: Float + RealField + num_traits::NumCast,
{
    type Matrix = Array2<T>;
    type Output = NdarrayPCAResult<T>;

    #[inline]
    fn compute_pca(
        matrix: &Self::Matrix,
        n_components: Option<usize>,
    ) -> Result<Self::Output, PCAError> {
        let nalg = ndarray_to_nalgebra(matrix);
        let result = <NalgebraPcaKernel<T> as PcaKernel>::compute_pca(&nalg, n_components)?;

        Ok(NdarrayPCAResult {
            components:               nalgebra_to_ndarray(&result.components),
            scores:                   nalgebra_to_ndarray(&result.scores),
            explained_variance:       Array1::from_vec(
                result.explained_variance.as_slice().to_vec(),
            ),
            explained_variance_ratio: Array1::from_vec(
                result.explained_variance_ratio.as_slice().to_vec(),
            ),
            mean:                     Array1::from_vec(result.mean.as_slice().to_vec()),
        })
    }
}

/// Nalgebra LAPACK-backed PCA kernel.
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
pub(crate) struct NalgebraLapackPcaKernel;

#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
impl PcaKernel for NalgebraLapackPcaKernel {
    type Matrix = DMatrix<f64>;
    type Output = NalgebraPCAResult<f64>;

    #[inline]
    fn compute_pca(
        matrix: &Self::Matrix,
        n_components: Option<usize>,
    ) -> Result<Self::Output, PCAError> {
        compute_nalgebra_pca_with_svd(
            matrix,
            n_components,
            crate::backend::svd::compute_nalgebra_lapack_svd,
        )
    }
}

/// Ndarray LAPACK-backed PCA kernel.
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
pub(crate) struct NdarrayLapackPcaKernel;

#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
impl PcaKernel for NdarrayLapackPcaKernel {
    type Matrix = Array2<f64>;
    type Output = NdarrayPCAResult<f64>;

    #[inline]
    fn compute_pca(
        matrix: &Self::Matrix,
        n_components: Option<usize>,
    ) -> Result<Self::Output, PCAError> {
        let nalg = ndarray_to_nalgebra(matrix);
        let result = <NalgebraLapackPcaKernel as PcaKernel>::compute_pca(&nalg, n_components)?;

        Ok(NdarrayPCAResult {
            components:               nalgebra_to_ndarray(&result.components),
            scores:                   nalgebra_to_ndarray(&result.scores),
            explained_variance:       Array1::from_vec(
                result.explained_variance.as_slice().to_vec(),
            ),
            explained_variance_ratio: Array1::from_vec(
                result.explained_variance_ratio.as_slice().to_vec(),
            ),
            mean:                     Array1::from_vec(result.mean.as_slice().to_vec()),
        })
    }
}

/// Dispatch PCA computation to the nalgebra kernel.
#[inline]
pub(crate) fn compute_nalgebra_pca<T>(
    matrix: &DMatrix<T>,
    n_components: Option<usize>,
) -> Result<NalgebraPCAResult<T>, PCAError>
where
    T: RealField + Copy + Float + num_traits::NumCast,
{
    <NalgebraPcaKernel<T> as PcaKernel>::compute_pca(matrix, n_components)
}

/// Dispatch PCA computation to the ndarray kernel.
#[inline]
pub(crate) fn compute_ndarray_pca<T>(
    matrix: &Array2<T>,
    n_components: Option<usize>,
) -> Result<NdarrayPCAResult<T>, PCAError>
where
    T: Float + RealField + num_traits::NumCast,
{
    <NdarrayPcaKernel<T> as PcaKernel>::compute_pca(matrix, n_components)
}

/// Dispatch PCA computation to the nalgebra LAPACK kernel.
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
#[inline]
pub(crate) fn compute_nalgebra_lapack_pca(
    matrix: &DMatrix<f64>,
    n_components: Option<usize>,
) -> Result<NalgebraPCAResult<f64>, PCAError> {
    <NalgebraLapackPcaKernel as PcaKernel>::compute_pca(matrix, n_components)
}

/// Dispatch PCA computation to the ndarray LAPACK kernel.
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
#[inline]
pub(crate) fn compute_ndarray_lapack_pca(
    matrix: &Array2<f64>,
    n_components: Option<usize>,
) -> Result<NdarrayPCAResult<f64>, PCAError> {
    <NdarrayLapackPcaKernel as PcaKernel>::compute_pca(matrix, n_components)
}

fn compute_nalgebra_pca_with_svd<T, F>(
    matrix: &DMatrix<T>,
    n_components: Option<usize>,
    svd_fn: F,
) -> Result<NalgebraPCAResult<T>, PCAError>
where
    T: RealField + Copy + Float + num_traits::NumCast,
    F: Fn(&DMatrix<T>) -> Result<crate::svd::NalgebraSVD<T>, crate::svd::SVDError>,
{
    if matrix.is_empty() {
        return Err(PCAError::EmptyMatrix);
    }

    let (n_samples, n_features) = matrix.shape();
    if n_samples < 2 {
        return Err(PCAError::InsufficientSamples);
    }

    let mean = nalgebra_stats::column_means(matrix);
    let centered = nalgebra_stats::center_columns(matrix);

    let k = n_components.unwrap_or(n_features.min(n_samples));
    if k == 0 || k > n_features || k > n_samples {
        return Err(PCAError::InvalidComponents);
    }

    let svd = svd_fn(&centered).map_err(|e| PCAError::Computation(e.to_string()))?;

    let n_sv = svd.singular_values.len();
    let k_actual = k.min(n_sv);
    let sample_scale = cast_usize::<T>(n_samples - 1, "sample count")?;

    let components = svd.vt.rows(0, k_actual).transpose().clone();
    let u = svd.u.columns(0, k_actual);
    let s = svd.singular_values.rows(0, k_actual);
    let scores = u * DMatrix::from_diagonal(&s);

    let mut total_var = T::zero();
    for i in 0..n_sv {
        total_var += svd.singular_values[i] * svd.singular_values[i];
    }
    total_var /= sample_scale;

    let mut explained_variance = DVector::zeros(k_actual);
    for i in 0..k_actual {
        explained_variance[i] = (svd.singular_values[i] * svd.singular_values[i]) / sample_scale;
    }

    let mut explained_variance_ratio = DVector::zeros(k_actual);
    if total_var > T::zero() {
        for i in 0..k_actual {
            explained_variance_ratio[i] = explained_variance[i] / total_var;
        }
    }

    Ok(
        NalgebraPCAResult {
            components,
            scores,
            explained_variance,
            explained_variance_ratio,
            mean,
        },
    )
}

fn cast_usize<T: num_traits::NumCast>(value: usize, field: &str) -> Result<T, PCAError> {
    num_traits::NumCast::from(value)
        .ok_or_else(|| PCAError::Computation(format!("failed to cast {field} to scalar")))
}
