//! Internal Cholesky kernel traits and backend adapters.

use std::marker::PhantomData;

use nalgebra::linalg::Cholesky;
use nalgebra::{DMatrix, DVector, RealField};
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
use nalgebra_lapack::Cholesky as NalgebraLapackCholesky;
use ndarray::{Array1, Array2};
use num_traits::Float;

use crate::cholesky::{CholeskyError, NalgebraCholeskyResult, NdarrayCholeskyResult};
use crate::interop::{nalgebra_to_ndarray, ndarray_to_nalgebra};

/// Internal kernel trait for Cholesky operations.
pub(crate) trait CholeskyKernel {
    type Decomposition;
    type Matrix;
    type Vector;

    fn compute_cholesky(matrix: &Self::Matrix) -> Result<Self::Decomposition, CholeskyError>;
    fn solve(matrix: &Self::Matrix, rhs: &Self::Vector) -> Result<Self::Vector, CholeskyError>;
    fn inverse(matrix: &Self::Matrix) -> Result<Self::Matrix, CholeskyError>;
}

/// Nalgebra-backed Cholesky kernel.
pub(crate) struct NalgebraCholeskyKernel<T>(PhantomData<T>);

impl<T> CholeskyKernel for NalgebraCholeskyKernel<T>
where
    T: RealField + Copy + Float,
{
    type Decomposition = NalgebraCholeskyResult<T>;
    type Matrix = DMatrix<T>;
    type Vector = DVector<T>;

    #[inline]
    fn compute_cholesky(matrix: &Self::Matrix) -> Result<Self::Decomposition, CholeskyError> {
        validate_nalgebra_matrix(matrix)?;

        let cholesky = Cholesky::new(matrix.clone()).ok_or(CholeskyError::NotPositiveDefinite)?;
        let l = cholesky.l().clone();
        Ok(NalgebraCholeskyResult { l })
    }

    #[inline]
    fn solve(matrix: &Self::Matrix, rhs: &Self::Vector) -> Result<Self::Vector, CholeskyError> {
        validate_nalgebra_matrix(matrix)?;
        if rhs.is_empty() {
            return Err(CholeskyError::EmptyMatrix);
        }
        if matrix.nrows() != rhs.len() {
            return Err(CholeskyError::InvalidInput(
                "RHS length must match matrix dimensions".to_string(),
            ));
        }
        if rhs.iter().any(|&x| !Float::is_finite(x)) {
            return Err(CholeskyError::NumericalInstability);
        }

        let cholesky = Cholesky::new(matrix.clone()).ok_or(CholeskyError::NotPositiveDefinite)?;
        let solution = cholesky.solve(rhs);
        if solution.iter().any(|&x| !Float::is_finite(x)) {
            return Err(CholeskyError::NumericalInstability);
        }

        Ok(solution)
    }

    #[inline]
    fn inverse(matrix: &Self::Matrix) -> Result<Self::Matrix, CholeskyError> {
        validate_nalgebra_matrix(matrix)?;

        let cholesky = Cholesky::new(matrix.clone()).ok_or(CholeskyError::NotPositiveDefinite)?;
        let inverse = cholesky.inverse();
        if inverse.iter().any(|&x| !Float::is_finite(x)) {
            return Err(CholeskyError::NumericalInstability);
        }

        Ok(inverse)
    }
}

/// Ndarray-backed Cholesky kernel.
pub(crate) struct NdarrayCholeskyKernel<T>(PhantomData<T>);

impl<T> CholeskyKernel for NdarrayCholeskyKernel<T>
where
    T: Float + RealField,
{
    type Decomposition = NdarrayCholeskyResult<T>;
    type Matrix = Array2<T>;
    type Vector = Array1<T>;

    #[inline]
    fn compute_cholesky(matrix: &Self::Matrix) -> Result<Self::Decomposition, CholeskyError> {
        let nalg = ndarray_to_nalgebra(matrix);
        let result = <NalgebraCholeskyKernel<T> as CholeskyKernel>::compute_cholesky(&nalg)?;
        Ok(NdarrayCholeskyResult { l: nalgebra_to_ndarray(&result.l) })
    }

    #[inline]
    fn solve(matrix: &Self::Matrix, rhs: &Self::Vector) -> Result<Self::Vector, CholeskyError> {
        let nalg_matrix = ndarray_to_nalgebra(matrix);
        let nalg_rhs = DVector::from_vec(rhs.to_vec());
        let solution =
            <NalgebraCholeskyKernel<T> as CholeskyKernel>::solve(&nalg_matrix, &nalg_rhs)?;
        Ok(Array1::from_vec(solution.as_slice().to_vec()))
    }

    #[inline]
    fn inverse(matrix: &Self::Matrix) -> Result<Self::Matrix, CholeskyError> {
        let nalg = ndarray_to_nalgebra(matrix);
        let inv = <NalgebraCholeskyKernel<T> as CholeskyKernel>::inverse(&nalg)?;
        Ok(nalgebra_to_ndarray(&inv))
    }
}

/// Nalgebra LAPACK-backed Cholesky kernel.
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
pub(crate) struct NalgebraLapackCholeskyKernel;

#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
impl CholeskyKernel for NalgebraLapackCholeskyKernel {
    type Decomposition = NalgebraCholeskyResult<f64>;
    type Matrix = DMatrix<f64>;
    type Vector = DVector<f64>;

    #[inline]
    fn compute_cholesky(matrix: &Self::Matrix) -> Result<Self::Decomposition, CholeskyError> {
        validate_nalgebra_matrix(matrix)?;

        let cholesky = NalgebraLapackCholesky::new(matrix.clone())
            .ok_or(CholeskyError::NotPositiveDefinite)?;
        Ok(NalgebraCholeskyResult { l: cholesky.l() })
    }

    #[inline]
    fn solve(matrix: &Self::Matrix, rhs: &Self::Vector) -> Result<Self::Vector, CholeskyError> {
        validate_nalgebra_matrix(matrix)?;
        if rhs.is_empty() {
            return Err(CholeskyError::EmptyMatrix);
        }
        if matrix.nrows() != rhs.len() {
            return Err(CholeskyError::InvalidInput(
                "RHS length must match matrix dimensions".to_string(),
            ));
        }
        if rhs.iter().any(|x| !x.is_finite()) {
            return Err(CholeskyError::NumericalInstability);
        }

        let cholesky = NalgebraLapackCholesky::new(matrix.clone())
            .ok_or(CholeskyError::NotPositiveDefinite)?;
        let solution = cholesky.solve(rhs).ok_or(CholeskyError::NumericalInstability)?;
        if solution.iter().any(|x| !x.is_finite()) {
            return Err(CholeskyError::NumericalInstability);
        }

        Ok(solution)
    }

    #[inline]
    fn inverse(matrix: &Self::Matrix) -> Result<Self::Matrix, CholeskyError> {
        validate_nalgebra_matrix(matrix)?;

        let cholesky = NalgebraLapackCholesky::new(matrix.clone())
            .ok_or(CholeskyError::NotPositiveDefinite)?;
        let inverse = cholesky.inverse().ok_or(CholeskyError::NumericalInstability)?;
        if inverse.iter().any(|x| !x.is_finite()) {
            return Err(CholeskyError::NumericalInstability);
        }

        Ok(inverse)
    }
}

/// Ndarray LAPACK-backed Cholesky kernel.
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
pub(crate) struct NdarrayLapackCholeskyKernel;

#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
impl CholeskyKernel for NdarrayLapackCholeskyKernel {
    type Decomposition = NdarrayCholeskyResult<f64>;
    type Matrix = Array2<f64>;
    type Vector = Array1<f64>;

    #[inline]
    fn compute_cholesky(matrix: &Self::Matrix) -> Result<Self::Decomposition, CholeskyError> {
        let nalg = ndarray_to_nalgebra(matrix);
        let result = <NalgebraLapackCholeskyKernel as CholeskyKernel>::compute_cholesky(&nalg)?;
        Ok(NdarrayCholeskyResult { l: nalgebra_to_ndarray(&result.l) })
    }

    #[inline]
    fn solve(matrix: &Self::Matrix, rhs: &Self::Vector) -> Result<Self::Vector, CholeskyError> {
        let nalg_matrix = ndarray_to_nalgebra(matrix);
        let nalg_rhs = DVector::from_vec(rhs.to_vec());
        let solution =
            <NalgebraLapackCholeskyKernel as CholeskyKernel>::solve(&nalg_matrix, &nalg_rhs)?;
        Ok(Array1::from_vec(solution.as_slice().to_vec()))
    }

    #[inline]
    fn inverse(matrix: &Self::Matrix) -> Result<Self::Matrix, CholeskyError> {
        let nalg = ndarray_to_nalgebra(matrix);
        let inv = <NalgebraLapackCholeskyKernel as CholeskyKernel>::inverse(&nalg)?;
        Ok(nalgebra_to_ndarray(&inv))
    }
}

/// Dispatch Cholesky decomposition to the nalgebra kernel.
#[inline]
pub(crate) fn compute_nalgebra_cholesky<T>(
    matrix: &DMatrix<T>,
) -> Result<NalgebraCholeskyResult<T>, CholeskyError>
where
    T: RealField + Copy + Float,
{
    <NalgebraCholeskyKernel<T> as CholeskyKernel>::compute_cholesky(matrix)
}

/// Dispatch linear solve to the nalgebra kernel.
#[inline]
pub(crate) fn solve_nalgebra_cholesky<T>(
    matrix: &DMatrix<T>,
    rhs: &DVector<T>,
) -> Result<DVector<T>, CholeskyError>
where
    T: RealField + Copy + Float,
{
    <NalgebraCholeskyKernel<T> as CholeskyKernel>::solve(matrix, rhs)
}

/// Dispatch matrix inversion to the nalgebra kernel.
#[inline]
pub(crate) fn inverse_nalgebra_cholesky<T>(matrix: &DMatrix<T>) -> Result<DMatrix<T>, CholeskyError>
where
    T: RealField + Copy + Float,
{
    <NalgebraCholeskyKernel<T> as CholeskyKernel>::inverse(matrix)
}

/// Dispatch Cholesky decomposition to the ndarray kernel.
#[inline]
pub(crate) fn compute_ndarray_cholesky<T>(
    matrix: &Array2<T>,
) -> Result<NdarrayCholeskyResult<T>, CholeskyError>
where
    T: Float + RealField,
{
    <NdarrayCholeskyKernel<T> as CholeskyKernel>::compute_cholesky(matrix)
}

/// Dispatch linear solve to the ndarray kernel.
#[inline]
pub(crate) fn solve_ndarray_cholesky<T>(
    matrix: &Array2<T>,
    rhs: &Array1<T>,
) -> Result<Array1<T>, CholeskyError>
where
    T: Float + RealField,
{
    <NdarrayCholeskyKernel<T> as CholeskyKernel>::solve(matrix, rhs)
}

/// Dispatch matrix inversion to the ndarray kernel.
#[inline]
pub(crate) fn inverse_ndarray_cholesky<T>(matrix: &Array2<T>) -> Result<Array2<T>, CholeskyError>
where
    T: Float + RealField,
{
    <NdarrayCholeskyKernel<T> as CholeskyKernel>::inverse(matrix)
}

/// Dispatch Cholesky decomposition to the nalgebra LAPACK kernel.
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
#[inline]
pub(crate) fn compute_nalgebra_lapack_cholesky(
    matrix: &DMatrix<f64>,
) -> Result<NalgebraCholeskyResult<f64>, CholeskyError> {
    <NalgebraLapackCholeskyKernel as CholeskyKernel>::compute_cholesky(matrix)
}

/// Dispatch linear solve to the nalgebra LAPACK kernel.
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
#[inline]
pub(crate) fn solve_nalgebra_lapack_cholesky(
    matrix: &DMatrix<f64>,
    rhs: &DVector<f64>,
) -> Result<DVector<f64>, CholeskyError> {
    <NalgebraLapackCholeskyKernel as CholeskyKernel>::solve(matrix, rhs)
}

/// Dispatch matrix inversion to the nalgebra LAPACK kernel.
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
#[inline]
pub(crate) fn inverse_nalgebra_lapack_cholesky(
    matrix: &DMatrix<f64>,
) -> Result<DMatrix<f64>, CholeskyError> {
    <NalgebraLapackCholeskyKernel as CholeskyKernel>::inverse(matrix)
}

/// Dispatch Cholesky decomposition to the ndarray LAPACK kernel.
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
#[inline]
pub(crate) fn compute_ndarray_lapack_cholesky(
    matrix: &Array2<f64>,
) -> Result<NdarrayCholeskyResult<f64>, CholeskyError> {
    <NdarrayLapackCholeskyKernel as CholeskyKernel>::compute_cholesky(matrix)
}

/// Dispatch linear solve to the ndarray LAPACK kernel.
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
#[inline]
pub(crate) fn solve_ndarray_lapack_cholesky(
    matrix: &Array2<f64>,
    rhs: &Array1<f64>,
) -> Result<Array1<f64>, CholeskyError> {
    <NdarrayLapackCholeskyKernel as CholeskyKernel>::solve(matrix, rhs)
}

/// Dispatch matrix inversion to the ndarray LAPACK kernel.
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
#[inline]
pub(crate) fn inverse_ndarray_lapack_cholesky(
    matrix: &Array2<f64>,
) -> Result<Array2<f64>, CholeskyError> {
    <NdarrayLapackCholeskyKernel as CholeskyKernel>::inverse(matrix)
}

fn validate_nalgebra_matrix<T>(matrix: &DMatrix<T>) -> Result<(), CholeskyError>
where
    T: RealField + Copy + Float,
{
    if matrix.is_empty() {
        return Err(CholeskyError::EmptyMatrix);
    }

    let (rows, cols) = matrix.shape();
    if rows != cols {
        return Err(CholeskyError::NotSquare);
    }

    if matrix.iter().any(|&x| !Float::is_finite(x)) {
        return Err(CholeskyError::NumericalInstability);
    }

    Ok(())
}
