//! Internal triangular solve kernel traits and backend adapters.

use std::marker::PhantomData;

use nalgebra::{DMatrix, DVector, RealField};
use ndarray::{Array1, Array2};
use num_traits::Float;

use crate::interop::ndarray_to_nalgebra;
use crate::triangular::TriangularError;

/// Internal kernel trait for triangular solve operations.
pub(crate) trait TriangularSolveKernel {
    type Matrix;
    type Vector;

    fn solve_lower(
        matrix: &Self::Matrix,
        rhs: &Self::Vector,
    ) -> Result<Self::Vector, TriangularError>;
    fn solve_upper(
        matrix: &Self::Matrix,
        rhs: &Self::Vector,
    ) -> Result<Self::Vector, TriangularError>;
}

/// Nalgebra-backed triangular solve kernel.
pub(crate) struct NalgebraTriangularSolveKernel<T>(PhantomData<T>);

impl<T> TriangularSolveKernel for NalgebraTriangularSolveKernel<T>
where
    T: RealField + Copy,
{
    type Matrix = DMatrix<T>;
    type Vector = DVector<T>;

    #[inline]
    fn solve_lower(
        matrix: &Self::Matrix,
        rhs: &Self::Vector,
    ) -> Result<Self::Vector, TriangularError> {
        validate_inputs(matrix, rhs)?;

        let n = matrix.nrows();
        let mut solution = rhs.clone();
        for i in 0..n {
            if matrix[(i, i)] == T::zero() {
                return Err(TriangularError::Singular);
            }

            let mut sum = T::zero();
            for j in 0..i {
                sum += matrix[(i, j)] * solution[j];
            }
            solution[i] = (rhs[i] - sum) / matrix[(i, i)];
        }

        Ok(solution)
    }

    #[inline]
    fn solve_upper(
        matrix: &Self::Matrix,
        rhs: &Self::Vector,
    ) -> Result<Self::Vector, TriangularError> {
        validate_inputs(matrix, rhs)?;

        let n = matrix.nrows();
        let mut solution = rhs.clone();
        for i in (0..n).rev() {
            if matrix[(i, i)] == T::zero() {
                return Err(TriangularError::Singular);
            }

            let mut sum = T::zero();
            for j in (i + 1)..n {
                sum += matrix[(i, j)] * solution[j];
            }
            solution[i] = (rhs[i] - sum) / matrix[(i, i)];
        }

        Ok(solution)
    }
}

/// Ndarray-backed triangular solve kernel.
pub(crate) struct NdarrayTriangularSolveKernel<T>(PhantomData<T>);

impl<T> TriangularSolveKernel for NdarrayTriangularSolveKernel<T>
where
    T: Float + RealField,
{
    type Matrix = Array2<T>;
    type Vector = Array1<T>;

    #[inline]
    fn solve_lower(
        matrix: &Self::Matrix,
        rhs: &Self::Vector,
    ) -> Result<Self::Vector, TriangularError> {
        let nalg_matrix = ndarray_to_nalgebra(matrix);
        let nalg_rhs = DVector::from_vec(rhs.to_vec());
        let solution = <NalgebraTriangularSolveKernel<T> as TriangularSolveKernel>::solve_lower(
            &nalg_matrix,
            &nalg_rhs,
        )?;

        Ok(Array1::from_vec(solution.as_slice().to_vec()))
    }

    #[inline]
    fn solve_upper(
        matrix: &Self::Matrix,
        rhs: &Self::Vector,
    ) -> Result<Self::Vector, TriangularError> {
        let nalg_matrix = ndarray_to_nalgebra(matrix);
        let nalg_rhs = DVector::from_vec(rhs.to_vec());
        let solution = <NalgebraTriangularSolveKernel<T> as TriangularSolveKernel>::solve_upper(
            &nalg_matrix,
            &nalg_rhs,
        )?;

        Ok(Array1::from_vec(solution.as_slice().to_vec()))
    }
}

/// Dispatch lower-triangular solve to the nalgebra kernel.
#[inline]
pub(crate) fn solve_nalgebra_lower<T>(
    matrix: &DMatrix<T>,
    rhs: &DVector<T>,
) -> Result<DVector<T>, TriangularError>
where
    T: RealField + Copy,
{
    <NalgebraTriangularSolveKernel<T> as TriangularSolveKernel>::solve_lower(matrix, rhs)
}

/// Dispatch upper-triangular solve to the nalgebra kernel.
#[inline]
pub(crate) fn solve_nalgebra_upper<T>(
    matrix: &DMatrix<T>,
    rhs: &DVector<T>,
) -> Result<DVector<T>, TriangularError>
where
    T: RealField + Copy,
{
    <NalgebraTriangularSolveKernel<T> as TriangularSolveKernel>::solve_upper(matrix, rhs)
}

/// Dispatch lower-triangular solve to the ndarray kernel.
#[inline]
pub(crate) fn solve_ndarray_lower<T>(
    matrix: &Array2<T>,
    rhs: &Array1<T>,
) -> Result<Array1<T>, TriangularError>
where
    T: Float + RealField,
{
    <NdarrayTriangularSolveKernel<T> as TriangularSolveKernel>::solve_lower(matrix, rhs)
}

/// Dispatch upper-triangular solve to the ndarray kernel.
#[inline]
pub(crate) fn solve_ndarray_upper<T>(
    matrix: &Array2<T>,
    rhs: &Array1<T>,
) -> Result<Array1<T>, TriangularError>
where
    T: Float + RealField,
{
    <NdarrayTriangularSolveKernel<T> as TriangularSolveKernel>::solve_upper(matrix, rhs)
}

fn validate_inputs<T>(matrix: &DMatrix<T>, rhs: &DVector<T>) -> Result<(), TriangularError>
where
    T: RealField + Copy,
{
    if matrix.is_empty() || rhs.is_empty() {
        return Err(TriangularError::EmptyMatrix);
    }
    if !matrix.is_square() {
        return Err(TriangularError::NotSquare);
    }
    if matrix.nrows() != rhs.len() {
        return Err(TriangularError::DimensionMismatch);
    }

    Ok(())
}
