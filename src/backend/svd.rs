//! Internal SVD kernel traits and backend adapters.

use std::marker::PhantomData;

use nalgebra::{DMatrix, RealField};
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
use nalgebra_lapack::SVD as NalgebraLapackSvd;
use ndarray::{Array1, Array2};
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
use ndarray_linalg::SVD as NdarrayLinalgSvd;
use num_traits::Float;

use crate::interop::{nalgebra_to_ndarray, ndarray_to_nalgebra};
use crate::svd::{NalgebraSVD, NdarraySVD, SVDError};

/// Internal kernel trait for SVD operations.
pub(crate) trait SvdKernel {
    type Matrix;
    type Output;

    fn compute_svd(matrix: &Self::Matrix) -> Result<Self::Output, SVDError>;
}

/// Nalgebra-backed SVD kernel.
pub(crate) struct NalgebraSvdKernel<T>(PhantomData<T>);

impl<T: RealField> SvdKernel for NalgebraSvdKernel<T> {
    type Matrix = DMatrix<T>;
    type Output = NalgebraSVD<T>;

    #[inline]
    fn compute_svd(matrix: &Self::Matrix) -> Result<Self::Output, SVDError> {
        if matrix.is_empty() {
            return Err(SVDError::EmptyMatrix);
        }

        let svd = matrix.clone().svd(true, true);
        match (svd.u, svd.v_t) {
            (Some(u), Some(vt)) => Ok(NalgebraSVD { u, singular_values: svd.singular_values, vt }),
            _ => Err(SVDError::ConvergenceFailed),
        }
    }
}

/// Ndarray-backed SVD kernel.
pub(crate) struct NdarraySvdKernel<T>(PhantomData<T>);

impl<T: Float + RealField> SvdKernel for NdarraySvdKernel<T> {
    type Matrix = Array2<T>;
    type Output = NdarraySVD<T>;

    #[inline]
    fn compute_svd(matrix: &Self::Matrix) -> Result<Self::Output, SVDError> {
        if matrix.is_empty() {
            return Err(SVDError::EmptyMatrix);
        }

        let nalgebra_matrix = ndarray_to_nalgebra(matrix);
        let nalgebra_svd = <NalgebraSvdKernel<T> as SvdKernel>::compute_svd(&nalgebra_matrix)?;

        let u = nalgebra_to_ndarray(&nalgebra_svd.u);
        let vt = nalgebra_to_ndarray(&nalgebra_svd.vt);
        let singular_values = Array1::from_vec(nalgebra_svd.singular_values.as_slice().to_vec());

        Ok(NdarraySVD { u, singular_values, vt })
    }
}

/// Nalgebra LAPACK-backed SVD kernel.
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
pub(crate) struct NalgebraLapackSvdKernel;

#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
impl SvdKernel for NalgebraLapackSvdKernel {
    type Matrix = DMatrix<f64>;
    type Output = NalgebraSVD<f64>;

    #[inline]
    fn compute_svd(matrix: &Self::Matrix) -> Result<Self::Output, SVDError> {
        if matrix.is_empty() {
            return Err(SVDError::EmptyMatrix);
        }

        let svd = NalgebraLapackSvd::new(matrix.clone()).ok_or(SVDError::ConvergenceFailed)?;

        Ok(NalgebraSVD {
            u:               svd.u,
            singular_values: svd.singular_values,
            vt:              svd.vt,
        })
    }
}

/// Ndarray LAPACK-backed SVD kernel.
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
pub(crate) struct NdarrayLapackSvdKernel;

#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
impl SvdKernel for NdarrayLapackSvdKernel {
    type Matrix = Array2<f64>;
    type Output = NdarraySVD<f64>;

    #[inline]
    fn compute_svd(matrix: &Self::Matrix) -> Result<Self::Output, SVDError> {
        if matrix.is_empty() {
            return Err(SVDError::EmptyMatrix);
        }

        let (u_opt, singular_values, vt_opt) =
            matrix.view().svd(true, true).map_err(|_| SVDError::ConvergenceFailed)?;
        let u = u_opt.ok_or(SVDError::ConvergenceFailed)?;
        let vt = vt_opt.ok_or(SVDError::ConvergenceFailed)?;

        Ok(NdarraySVD { u, singular_values, vt })
    }
}

/// Dispatch SVD computation to the nalgebra kernel.
#[inline]
pub(crate) fn compute_nalgebra_svd<T: RealField>(
    matrix: &DMatrix<T>,
) -> Result<NalgebraSVD<T>, SVDError> {
    <NalgebraSvdKernel<T> as SvdKernel>::compute_svd(matrix)
}

/// Dispatch SVD computation to the ndarray kernel.
#[inline]
pub(crate) fn compute_ndarray_svd<T: Float + RealField>(
    matrix: &Array2<T>,
) -> Result<NdarraySVD<T>, SVDError> {
    <NdarraySvdKernel<T> as SvdKernel>::compute_svd(matrix)
}

/// Dispatch SVD computation to the nalgebra LAPACK kernel.
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
#[inline]
pub(crate) fn compute_nalgebra_lapack_svd(
    matrix: &DMatrix<f64>,
) -> Result<NalgebraSVD<f64>, SVDError> {
    <NalgebraLapackSvdKernel as SvdKernel>::compute_svd(matrix)
}

/// Dispatch SVD computation to the ndarray LAPACK kernel.
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
#[inline]
pub(crate) fn compute_ndarray_lapack_svd(
    matrix: &Array2<f64>,
) -> Result<NdarraySVD<f64>, SVDError> {
    <NdarrayLapackSvdKernel as SvdKernel>::compute_svd(matrix)
}
