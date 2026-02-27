//! Internal Schur kernel traits and backend adapters.

use std::marker::PhantomData;

use nalgebra::{DMatrix, RealField};
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
use nalgebra_lapack::Schur as NalgebraLapackSchur;
use ndarray::Array2;
use num_traits::Float;

use crate::interop::{nalgebra_to_ndarray, ndarray_to_nalgebra};
use crate::schur::{NalgebraSchurResult, NdarraySchurResult, SchurError};

/// Internal kernel trait for Schur operations.
pub(crate) trait SchurKernel {
    type Matrix;
    type Output;

    fn compute_schur(matrix: &Self::Matrix) -> Result<Self::Output, SchurError>;
}

/// Nalgebra-backed Schur kernel.
pub(crate) struct NalgebraSchurKernel<T>(PhantomData<T>);

impl<T> SchurKernel for NalgebraSchurKernel<T>
where
    T: RealField + Copy + Float,
{
    type Matrix = DMatrix<T>;
    type Output = NalgebraSchurResult<T>;

    #[inline]
    fn compute_schur(matrix: &Self::Matrix) -> Result<Self::Output, SchurError> {
        validate_nalgebra_matrix(matrix)?;

        let eps = T::epsilon();
        let max_iter = 200;
        let schur = nalgebra::linalg::Schur::try_new(matrix.clone(), eps, max_iter)
            .ok_or(SchurError::ConvergenceFailed)?;
        let (q, t) = schur.unpack();

        if q.iter().any(|&x| !Float::is_finite(x)) || t.iter().any(|&x| !Float::is_finite(x)) {
            return Err(SchurError::NumericalInstability);
        }

        Ok(NalgebraSchurResult { q, t })
    }
}

/// Ndarray-backed Schur kernel.
pub(crate) struct NdarraySchurKernel<T>(PhantomData<T>);

impl<T> SchurKernel for NdarraySchurKernel<T>
where
    T: Float + RealField,
{
    type Matrix = Array2<T>;
    type Output = NdarraySchurResult<T>;

    #[inline]
    fn compute_schur(matrix: &Self::Matrix) -> Result<Self::Output, SchurError> {
        let nalg = ndarray_to_nalgebra(matrix);
        let result = <NalgebraSchurKernel<T> as SchurKernel>::compute_schur(&nalg)?;

        Ok(NdarraySchurResult {
            q: nalgebra_to_ndarray(&result.q),
            t: nalgebra_to_ndarray(&result.t),
        })
    }
}

/// Nalgebra LAPACK-backed Schur kernel.
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
pub(crate) struct NalgebraLapackSchurKernel;

#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
impl SchurKernel for NalgebraLapackSchurKernel {
    type Matrix = DMatrix<f64>;
    type Output = NalgebraSchurResult<f64>;

    #[inline]
    fn compute_schur(matrix: &Self::Matrix) -> Result<Self::Output, SchurError> {
        validate_nalgebra_matrix(matrix)?;

        let schur =
            NalgebraLapackSchur::try_new(matrix.clone()).ok_or(SchurError::ConvergenceFailed)?;
        let (q, t) = schur.unpack();

        if q.iter().any(|x| !x.is_finite()) || t.iter().any(|x| !x.is_finite()) {
            return Err(SchurError::NumericalInstability);
        }

        Ok(NalgebraSchurResult { q, t })
    }
}

/// Ndarray LAPACK-backed Schur kernel.
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
pub(crate) struct NdarrayLapackSchurKernel;

#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
impl SchurKernel for NdarrayLapackSchurKernel {
    type Matrix = Array2<f64>;
    type Output = NdarraySchurResult<f64>;

    #[inline]
    fn compute_schur(matrix: &Self::Matrix) -> Result<Self::Output, SchurError> {
        let nalg = ndarray_to_nalgebra(matrix);
        let result = <NalgebraLapackSchurKernel as SchurKernel>::compute_schur(&nalg)?;

        Ok(NdarraySchurResult {
            q: nalgebra_to_ndarray(&result.q),
            t: nalgebra_to_ndarray(&result.t),
        })
    }
}

/// Dispatch Schur decomposition to the nalgebra kernel.
#[inline]
pub(crate) fn compute_nalgebra_schur<T>(
    matrix: &DMatrix<T>,
) -> Result<NalgebraSchurResult<T>, SchurError>
where
    T: RealField + Copy + Float,
{
    <NalgebraSchurKernel<T> as SchurKernel>::compute_schur(matrix)
}

/// Dispatch Schur decomposition to the ndarray kernel.
#[inline]
pub(crate) fn compute_ndarray_schur<T>(
    matrix: &Array2<T>,
) -> Result<NdarraySchurResult<T>, SchurError>
where
    T: Float + RealField,
{
    <NdarraySchurKernel<T> as SchurKernel>::compute_schur(matrix)
}

/// Dispatch Schur decomposition to the nalgebra LAPACK kernel.
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
#[inline]
pub(crate) fn compute_nalgebra_lapack_schur(
    matrix: &DMatrix<f64>,
) -> Result<NalgebraSchurResult<f64>, SchurError> {
    <NalgebraLapackSchurKernel as SchurKernel>::compute_schur(matrix)
}

/// Dispatch Schur decomposition to the ndarray LAPACK kernel.
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
#[inline]
pub(crate) fn compute_ndarray_lapack_schur(
    matrix: &Array2<f64>,
) -> Result<NdarraySchurResult<f64>, SchurError> {
    <NdarrayLapackSchurKernel as SchurKernel>::compute_schur(matrix)
}

fn validate_nalgebra_matrix<T>(matrix: &DMatrix<T>) -> Result<(), SchurError>
where
    T: RealField + Copy + Float,
{
    if matrix.is_empty() {
        return Err(SchurError::EmptyMatrix);
    }
    if !matrix.is_square() {
        return Err(SchurError::NotSquare);
    }
    if matrix.iter().any(|&x| !Float::is_finite(x)) {
        return Err(SchurError::NumericalInstability);
    }

    Ok(())
}
