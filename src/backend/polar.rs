//! Internal Polar kernel traits and backend adapters.

use std::marker::PhantomData;

use nalgebra::{DMatrix, RealField};
use ndarray::Array2;
use num_traits::Float;

use crate::interop::{nalgebra_to_ndarray, ndarray_to_nalgebra};
use crate::polar::{NalgebraPolarResult, NdarrayPolarResult, PolarError};

/// Internal kernel trait for polar decomposition operations.
pub(crate) trait PolarKernel {
    type Matrix;
    type Output;

    fn compute_polar(matrix: &Self::Matrix) -> Result<Self::Output, PolarError>;
}

/// Nalgebra-backed polar kernel.
pub(crate) struct NalgebraPolarKernel<T>(PhantomData<T>);

impl<T> PolarKernel for NalgebraPolarKernel<T>
where
    T: RealField + Copy + Float,
{
    type Matrix = DMatrix<T>;
    type Output = NalgebraPolarResult<T>;

    #[inline]
    fn compute_polar(matrix: &Self::Matrix) -> Result<Self::Output, PolarError> {
        validate_nalgebra_matrix(matrix)?;

        let svd = crate::backend::svd::compute_nalgebra_svd(matrix)?;
        let u = &svd.u * svd.vt.transpose();
        let sigma = DMatrix::from_diagonal(&svd.singular_values);
        let v = svd.vt.transpose();
        let p = &v * &sigma * &svd.vt;

        if u.iter().any(|&x| !Float::is_finite(x)) || p.iter().any(|&x| !Float::is_finite(x)) {
            return Err(PolarError::NumericalInstability);
        }

        Ok(NalgebraPolarResult { u, p })
    }
}

/// Ndarray-backed polar kernel.
pub(crate) struct NdarrayPolarKernel<T>(PhantomData<T>);

impl<T> PolarKernel for NdarrayPolarKernel<T>
where
    T: Float + RealField,
{
    type Matrix = Array2<T>;
    type Output = NdarrayPolarResult<T>;

    #[inline]
    fn compute_polar(matrix: &Self::Matrix) -> Result<Self::Output, PolarError> {
        let nalg = ndarray_to_nalgebra(matrix);
        let result = <NalgebraPolarKernel<T> as PolarKernel>::compute_polar(&nalg)?;

        Ok(NdarrayPolarResult {
            u: nalgebra_to_ndarray(&result.u),
            p: nalgebra_to_ndarray(&result.p),
        })
    }
}

/// Nalgebra LAPACK-backed polar kernel.
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
pub(crate) struct NalgebraLapackPolarKernel;

#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
impl PolarKernel for NalgebraLapackPolarKernel {
    type Matrix = DMatrix<f64>;
    type Output = NalgebraPolarResult<f64>;

    #[inline]
    fn compute_polar(matrix: &Self::Matrix) -> Result<Self::Output, PolarError> {
        validate_nalgebra_matrix(matrix)?;

        let svd = crate::backend::svd::compute_nalgebra_lapack_svd(matrix)?;
        let u = &svd.u * svd.vt.transpose();
        let sigma = DMatrix::from_diagonal(&svd.singular_values);
        let v = svd.vt.transpose();
        let p = &v * &sigma * &svd.vt;

        if u.iter().any(|x| !x.is_finite()) || p.iter().any(|x| !x.is_finite()) {
            return Err(PolarError::NumericalInstability);
        }

        Ok(NalgebraPolarResult { u, p })
    }
}

/// Ndarray LAPACK-backed polar kernel.
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
pub(crate) struct NdarrayLapackPolarKernel;

#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
impl PolarKernel for NdarrayLapackPolarKernel {
    type Matrix = Array2<f64>;
    type Output = NdarrayPolarResult<f64>;

    #[inline]
    fn compute_polar(matrix: &Self::Matrix) -> Result<Self::Output, PolarError> {
        let nalg = ndarray_to_nalgebra(matrix);
        let result = <NalgebraLapackPolarKernel as PolarKernel>::compute_polar(&nalg)?;

        Ok(NdarrayPolarResult {
            u: nalgebra_to_ndarray(&result.u),
            p: nalgebra_to_ndarray(&result.p),
        })
    }
}

/// Dispatch polar decomposition to the nalgebra kernel.
#[inline]
pub(crate) fn compute_nalgebra_polar<T>(
    matrix: &DMatrix<T>,
) -> Result<NalgebraPolarResult<T>, PolarError>
where
    T: RealField + Copy + Float,
{
    <NalgebraPolarKernel<T> as PolarKernel>::compute_polar(matrix)
}

/// Dispatch polar decomposition to the ndarray kernel.
#[inline]
pub(crate) fn compute_ndarray_polar<T>(
    matrix: &Array2<T>,
) -> Result<NdarrayPolarResult<T>, PolarError>
where
    T: Float + RealField,
{
    <NdarrayPolarKernel<T> as PolarKernel>::compute_polar(matrix)
}

/// Dispatch polar decomposition to the nalgebra LAPACK kernel.
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
#[inline]
pub(crate) fn compute_nalgebra_lapack_polar(
    matrix: &DMatrix<f64>,
) -> Result<NalgebraPolarResult<f64>, PolarError> {
    <NalgebraLapackPolarKernel as PolarKernel>::compute_polar(matrix)
}

/// Dispatch polar decomposition to the ndarray LAPACK kernel.
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
#[inline]
pub(crate) fn compute_ndarray_lapack_polar(
    matrix: &Array2<f64>,
) -> Result<NdarrayPolarResult<f64>, PolarError> {
    <NdarrayLapackPolarKernel as PolarKernel>::compute_polar(matrix)
}

fn validate_nalgebra_matrix<T>(matrix: &DMatrix<T>) -> Result<(), PolarError>
where
    T: RealField + Copy + Float,
{
    if matrix.is_empty() {
        return Err(PolarError::EmptyMatrix);
    }
    if !matrix.is_square() {
        return Err(PolarError::NotSquare);
    }
    if matrix.iter().any(|&x| !Float::is_finite(x)) {
        return Err(PolarError::NumericalInstability);
    }

    Ok(())
}
