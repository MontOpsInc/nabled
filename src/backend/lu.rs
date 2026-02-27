//! Internal LU kernel traits and backend adapters.

use std::marker::PhantomData;

use nalgebra::{DMatrix, DVector, RealField};
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
use nalgebra_lapack::LU as NalgebraLapackLu;
use ndarray::{Array1, Array2};
use num_traits::Float;

use crate::interop::{nalgebra_to_ndarray, ndarray_to_nalgebra};
use crate::lu::{LUError, LogDetResult, NalgebraLUResult, NdarrayLUResult};

/// Internal kernel trait for LU operations.
pub(crate) trait LuKernel {
    type Matrix;
    type Scalar;
    type SolveOutput;
    type Decomposition;

    fn compute_lu(matrix: &Self::Matrix) -> Result<Self::Decomposition, LUError>;
    fn solve(matrix: &Self::Matrix, rhs: &Self::SolveOutput) -> Result<Self::SolveOutput, LUError>;
    fn inverse(matrix: &Self::Matrix) -> Result<Self::Matrix, LUError>;
    fn determinant(matrix: &Self::Matrix) -> Result<Self::Scalar, LUError>;
    fn log_determinant(matrix: &Self::Matrix) -> Result<LogDetResult<Self::Scalar>, LUError>;
}

/// Nalgebra-backed LU kernel.
pub(crate) struct NalgebraLuKernel<T>(PhantomData<T>);

impl<T> LuKernel for NalgebraLuKernel<T>
where
    T: RealField + Copy + Float,
{
    type Decomposition = NalgebraLUResult<T>;
    type Matrix = DMatrix<T>;
    type Scalar = T;
    type SolveOutput = DVector<T>;

    #[inline]
    fn compute_lu(matrix: &Self::Matrix) -> Result<Self::Decomposition, LUError> {
        if matrix.is_empty() {
            return Err(LUError::EmptyMatrix);
        }
        let (rows, cols) = matrix.shape();
        if rows != cols {
            return Err(LUError::NotSquare);
        }
        if matrix.iter().any(|&x| !Float::is_finite(x)) {
            return Err(LUError::NumericalInstability);
        }

        let lu = matrix.clone().lu();
        let l = lu.l();
        let u = lu.u();
        Ok(NalgebraLUResult { l, u })
    }

    #[inline]
    fn solve(matrix: &Self::Matrix, rhs: &Self::SolveOutput) -> Result<Self::SolveOutput, LUError> {
        if matrix.is_empty() || rhs.is_empty() {
            return Err(LUError::EmptyMatrix);
        }
        let (rows, cols) = matrix.shape();
        if rows != cols {
            return Err(LUError::NotSquare);
        }
        if rows != rhs.len() {
            return Err(LUError::InvalidInput(
                "RHS length must match matrix dimensions".to_string(),
            ));
        }
        if matrix.iter().any(|&x| !Float::is_finite(x)) || rhs.iter().any(|&x| !Float::is_finite(x))
        {
            return Err(LUError::NumericalInstability);
        }

        let lu = matrix.clone().lu();
        lu.solve(rhs).ok_or(LUError::SingularMatrix)
    }

    #[inline]
    fn inverse(matrix: &Self::Matrix) -> Result<Self::Matrix, LUError> {
        if matrix.is_empty() {
            return Err(LUError::EmptyMatrix);
        }
        let (rows, cols) = matrix.shape();
        if rows != cols {
            return Err(LUError::NotSquare);
        }
        if matrix.iter().any(|&x| !Float::is_finite(x)) {
            return Err(LUError::NumericalInstability);
        }

        let lu = matrix.clone().lu();
        lu.try_inverse().ok_or(LUError::SingularMatrix)
    }

    #[inline]
    fn determinant(matrix: &Self::Matrix) -> Result<Self::Scalar, LUError> {
        if matrix.is_empty() {
            return Err(LUError::EmptyMatrix);
        }
        let (rows, cols) = matrix.shape();
        if rows != cols {
            return Err(LUError::NotSquare);
        }
        if matrix.iter().any(|&x| !Float::is_finite(x)) {
            return Err(LUError::NumericalInstability);
        }

        let det = matrix.clone().lu().determinant();
        if !Float::is_finite(det) {
            return Err(LUError::NumericalInstability);
        }
        Ok(det)
    }

    #[inline]
    fn log_determinant(matrix: &Self::Matrix) -> Result<LogDetResult<Self::Scalar>, LUError> {
        let det = Self::determinant(matrix)?;
        if det == T::zero() {
            return Err(LUError::SingularMatrix);
        }

        let sign = if det > T::zero() { 1_i8 } else { -1_i8 };
        let ln_abs_det = Float::ln(Float::abs(det));
        Ok(LogDetResult { sign, ln_abs_det })
    }
}

/// Ndarray-backed LU kernel.
pub(crate) struct NdarrayLuKernel<T>(PhantomData<T>);

impl<T> LuKernel for NdarrayLuKernel<T>
where
    T: Float + RealField,
{
    type Decomposition = NdarrayLUResult<T>;
    type Matrix = Array2<T>;
    type Scalar = T;
    type SolveOutput = Array1<T>;

    #[inline]
    fn compute_lu(matrix: &Self::Matrix) -> Result<Self::Decomposition, LUError> {
        let nalg = ndarray_to_nalgebra(matrix);
        let result = <NalgebraLuKernel<T> as LuKernel>::compute_lu(&nalg)?;
        Ok(NdarrayLUResult { l: nalgebra_to_ndarray(&result.l), u: nalgebra_to_ndarray(&result.u) })
    }

    #[inline]
    fn solve(matrix: &Self::Matrix, rhs: &Self::SolveOutput) -> Result<Self::SolveOutput, LUError> {
        let nalg_matrix = ndarray_to_nalgebra(matrix);
        let nalg_rhs = DVector::from_vec(rhs.to_vec());
        let solution = <NalgebraLuKernel<T> as LuKernel>::solve(&nalg_matrix, &nalg_rhs)?;
        Ok(Array1::from_vec(solution.as_slice().to_vec()))
    }

    #[inline]
    fn inverse(matrix: &Self::Matrix) -> Result<Self::Matrix, LUError> {
        let nalg = ndarray_to_nalgebra(matrix);
        let inv = <NalgebraLuKernel<T> as LuKernel>::inverse(&nalg)?;
        Ok(nalgebra_to_ndarray(&inv))
    }

    #[inline]
    fn determinant(matrix: &Self::Matrix) -> Result<Self::Scalar, LUError> {
        let nalg = ndarray_to_nalgebra(matrix);
        <NalgebraLuKernel<T> as LuKernel>::determinant(&nalg)
    }

    #[inline]
    fn log_determinant(matrix: &Self::Matrix) -> Result<LogDetResult<Self::Scalar>, LUError> {
        let nalg = ndarray_to_nalgebra(matrix);
        <NalgebraLuKernel<T> as LuKernel>::log_determinant(&nalg)
    }
}

/// Nalgebra LAPACK-backed LU kernel.
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
pub(crate) struct NalgebraLapackLuKernel;

#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
impl LuKernel for NalgebraLapackLuKernel {
    type Decomposition = NalgebraLUResult<f64>;
    type Matrix = DMatrix<f64>;
    type Scalar = f64;
    type SolveOutput = DVector<f64>;

    #[inline]
    fn compute_lu(matrix: &Self::Matrix) -> Result<Self::Decomposition, LUError> {
        if matrix.is_empty() {
            return Err(LUError::EmptyMatrix);
        }
        let (rows, cols) = matrix.shape();
        if rows != cols {
            return Err(LUError::NotSquare);
        }
        if matrix.iter().any(|x| !x.is_finite()) {
            return Err(LUError::NumericalInstability);
        }

        let lu = NalgebraLapackLu::new(matrix.clone());
        let l = lu.l();
        let u = lu.u();
        Ok(NalgebraLUResult { l, u })
    }

    #[inline]
    fn solve(matrix: &Self::Matrix, rhs: &Self::SolveOutput) -> Result<Self::SolveOutput, LUError> {
        if matrix.is_empty() || rhs.is_empty() {
            return Err(LUError::EmptyMatrix);
        }
        let (rows, cols) = matrix.shape();
        if rows != cols {
            return Err(LUError::NotSquare);
        }
        if rows != rhs.len() {
            return Err(LUError::InvalidInput(
                "RHS length must match matrix dimensions".to_string(),
            ));
        }
        if matrix.iter().any(|x| !x.is_finite()) || rhs.iter().any(|x| !x.is_finite()) {
            return Err(LUError::NumericalInstability);
        }

        let lu = NalgebraLapackLu::new(matrix.clone());
        lu.solve(rhs).ok_or(LUError::SingularMatrix)
    }

    #[inline]
    fn inverse(matrix: &Self::Matrix) -> Result<Self::Matrix, LUError> {
        if matrix.is_empty() {
            return Err(LUError::EmptyMatrix);
        }
        let (rows, cols) = matrix.shape();
        if rows != cols {
            return Err(LUError::NotSquare);
        }
        if matrix.iter().any(|x| !x.is_finite()) {
            return Err(LUError::NumericalInstability);
        }

        let lu = NalgebraLapackLu::new(matrix.clone());
        lu.inverse().ok_or(LUError::SingularMatrix)
    }

    #[inline]
    fn determinant(matrix: &Self::Matrix) -> Result<Self::Scalar, LUError> {
        if matrix.is_empty() {
            return Err(LUError::EmptyMatrix);
        }
        let (rows, cols) = matrix.shape();
        if rows != cols {
            return Err(LUError::NotSquare);
        }
        if matrix.iter().any(|x| !x.is_finite()) {
            return Err(LUError::NumericalInstability);
        }

        let lu = NalgebraLapackLu::new(matrix.clone());
        let u = lu.u();

        let mut det = 1.0_f64;
        for i in 0..rows {
            det *= u[(i, i)];
        }

        for (i, pivot) in lu.permutation_indices().iter().enumerate() {
            if *pivot as usize != i + 1 {
                det = -det;
            }
        }

        if !det.is_finite() {
            return Err(LUError::NumericalInstability);
        }
        Ok(det)
    }

    #[inline]
    fn log_determinant(matrix: &Self::Matrix) -> Result<LogDetResult<Self::Scalar>, LUError> {
        let det = Self::determinant(matrix)?;
        if det == 0.0 {
            return Err(LUError::SingularMatrix);
        }
        let sign = if det > 0.0 { 1_i8 } else { -1_i8 };
        Ok(LogDetResult { sign, ln_abs_det: det.abs().ln() })
    }
}

/// Ndarray LAPACK-backed LU kernel.
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
pub(crate) struct NdarrayLapackLuKernel;

#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
impl LuKernel for NdarrayLapackLuKernel {
    type Decomposition = NdarrayLUResult<f64>;
    type Matrix = Array2<f64>;
    type Scalar = f64;
    type SolveOutput = Array1<f64>;

    #[inline]
    fn compute_lu(matrix: &Self::Matrix) -> Result<Self::Decomposition, LUError> {
        let nalg = ndarray_to_nalgebra(matrix);
        let result = <NalgebraLapackLuKernel as LuKernel>::compute_lu(&nalg)?;
        Ok(NdarrayLUResult { l: nalgebra_to_ndarray(&result.l), u: nalgebra_to_ndarray(&result.u) })
    }

    #[inline]
    fn solve(matrix: &Self::Matrix, rhs: &Self::SolveOutput) -> Result<Self::SolveOutput, LUError> {
        let nalg_matrix = ndarray_to_nalgebra(matrix);
        let nalg_rhs = DVector::from_vec(rhs.to_vec());
        let solution = <NalgebraLapackLuKernel as LuKernel>::solve(&nalg_matrix, &nalg_rhs)?;
        Ok(Array1::from_vec(solution.as_slice().to_vec()))
    }

    #[inline]
    fn inverse(matrix: &Self::Matrix) -> Result<Self::Matrix, LUError> {
        let nalg = ndarray_to_nalgebra(matrix);
        let inv = <NalgebraLapackLuKernel as LuKernel>::inverse(&nalg)?;
        Ok(nalgebra_to_ndarray(&inv))
    }

    #[inline]
    fn determinant(matrix: &Self::Matrix) -> Result<Self::Scalar, LUError> {
        let nalg = ndarray_to_nalgebra(matrix);
        <NalgebraLapackLuKernel as LuKernel>::determinant(&nalg)
    }

    #[inline]
    fn log_determinant(matrix: &Self::Matrix) -> Result<LogDetResult<Self::Scalar>, LUError> {
        let nalg = ndarray_to_nalgebra(matrix);
        <NalgebraLapackLuKernel as LuKernel>::log_determinant(&nalg)
    }
}

/// Dispatch LU decomposition to the nalgebra kernel.
#[inline]
pub(crate) fn compute_nalgebra_lu<T>(matrix: &DMatrix<T>) -> Result<NalgebraLUResult<T>, LUError>
where
    T: RealField + Copy + Float,
{
    <NalgebraLuKernel<T> as LuKernel>::compute_lu(matrix)
}

/// Dispatch linear solve to the nalgebra kernel.
#[inline]
pub(crate) fn solve_nalgebra_lu<T>(
    matrix: &DMatrix<T>,
    rhs: &DVector<T>,
) -> Result<DVector<T>, LUError>
where
    T: RealField + Copy + Float,
{
    <NalgebraLuKernel<T> as LuKernel>::solve(matrix, rhs)
}

/// Dispatch matrix inversion to the nalgebra kernel.
#[inline]
pub(crate) fn inverse_nalgebra_lu<T>(matrix: &DMatrix<T>) -> Result<DMatrix<T>, LUError>
where
    T: RealField + Copy + Float,
{
    <NalgebraLuKernel<T> as LuKernel>::inverse(matrix)
}

/// Dispatch determinant computation to the nalgebra kernel.
#[inline]
pub(crate) fn determinant_nalgebra_lu<T>(matrix: &DMatrix<T>) -> Result<T, LUError>
where
    T: RealField + Copy + Float,
{
    <NalgebraLuKernel<T> as LuKernel>::determinant(matrix)
}

/// Dispatch log-determinant computation to the nalgebra kernel.
#[inline]
pub(crate) fn log_determinant_nalgebra_lu<T>(
    matrix: &DMatrix<T>,
) -> Result<LogDetResult<T>, LUError>
where
    T: RealField + Copy + Float,
{
    <NalgebraLuKernel<T> as LuKernel>::log_determinant(matrix)
}

/// Dispatch LU decomposition to the ndarray kernel.
#[inline]
pub(crate) fn compute_ndarray_lu<T>(matrix: &Array2<T>) -> Result<NdarrayLUResult<T>, LUError>
where
    T: Float + RealField,
{
    <NdarrayLuKernel<T> as LuKernel>::compute_lu(matrix)
}

/// Dispatch linear solve to the ndarray kernel.
#[inline]
pub(crate) fn solve_ndarray_lu<T>(matrix: &Array2<T>, rhs: &Array1<T>) -> Result<Array1<T>, LUError>
where
    T: Float + RealField,
{
    <NdarrayLuKernel<T> as LuKernel>::solve(matrix, rhs)
}

/// Dispatch matrix inversion to the ndarray kernel.
#[inline]
pub(crate) fn inverse_ndarray_lu<T>(matrix: &Array2<T>) -> Result<Array2<T>, LUError>
where
    T: Float + RealField,
{
    <NdarrayLuKernel<T> as LuKernel>::inverse(matrix)
}

/// Dispatch determinant computation to the ndarray kernel.
#[inline]
pub(crate) fn determinant_ndarray_lu<T>(matrix: &Array2<T>) -> Result<T, LUError>
where
    T: Float + RealField,
{
    <NdarrayLuKernel<T> as LuKernel>::determinant(matrix)
}

/// Dispatch log-determinant computation to the ndarray kernel.
#[inline]
pub(crate) fn log_determinant_ndarray_lu<T>(matrix: &Array2<T>) -> Result<LogDetResult<T>, LUError>
where
    T: Float + RealField,
{
    <NdarrayLuKernel<T> as LuKernel>::log_determinant(matrix)
}

/// Dispatch LU decomposition to the nalgebra LAPACK kernel.
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
#[inline]
pub(crate) fn compute_nalgebra_lapack_lu(
    matrix: &DMatrix<f64>,
) -> Result<NalgebraLUResult<f64>, LUError> {
    <NalgebraLapackLuKernel as LuKernel>::compute_lu(matrix)
}

/// Dispatch linear solve to the nalgebra LAPACK kernel.
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
#[inline]
pub(crate) fn solve_nalgebra_lapack_lu(
    matrix: &DMatrix<f64>,
    rhs: &DVector<f64>,
) -> Result<DVector<f64>, LUError> {
    <NalgebraLapackLuKernel as LuKernel>::solve(matrix, rhs)
}

/// Dispatch matrix inversion to the nalgebra LAPACK kernel.
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
#[inline]
pub(crate) fn inverse_nalgebra_lapack_lu(matrix: &DMatrix<f64>) -> Result<DMatrix<f64>, LUError> {
    <NalgebraLapackLuKernel as LuKernel>::inverse(matrix)
}

/// Dispatch determinant computation to the nalgebra LAPACK kernel.
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
#[inline]
pub(crate) fn determinant_nalgebra_lapack_lu(matrix: &DMatrix<f64>) -> Result<f64, LUError> {
    <NalgebraLapackLuKernel as LuKernel>::determinant(matrix)
}

/// Dispatch log-determinant computation to the nalgebra LAPACK kernel.
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
#[inline]
pub(crate) fn log_determinant_nalgebra_lapack_lu(
    matrix: &DMatrix<f64>,
) -> Result<LogDetResult<f64>, LUError> {
    <NalgebraLapackLuKernel as LuKernel>::log_determinant(matrix)
}

/// Dispatch LU decomposition to the ndarray LAPACK kernel.
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
#[inline]
pub(crate) fn compute_ndarray_lapack_lu(
    matrix: &Array2<f64>,
) -> Result<NdarrayLUResult<f64>, LUError> {
    <NdarrayLapackLuKernel as LuKernel>::compute_lu(matrix)
}

/// Dispatch linear solve to the ndarray LAPACK kernel.
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
#[inline]
pub(crate) fn solve_ndarray_lapack_lu(
    matrix: &Array2<f64>,
    rhs: &Array1<f64>,
) -> Result<Array1<f64>, LUError> {
    <NdarrayLapackLuKernel as LuKernel>::solve(matrix, rhs)
}

/// Dispatch matrix inversion to the ndarray LAPACK kernel.
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
#[inline]
pub(crate) fn inverse_ndarray_lapack_lu(matrix: &Array2<f64>) -> Result<Array2<f64>, LUError> {
    <NdarrayLapackLuKernel as LuKernel>::inverse(matrix)
}

/// Dispatch determinant computation to the ndarray LAPACK kernel.
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
#[inline]
pub(crate) fn determinant_ndarray_lapack_lu(matrix: &Array2<f64>) -> Result<f64, LUError> {
    <NdarrayLapackLuKernel as LuKernel>::determinant(matrix)
}

/// Dispatch log-determinant computation to the ndarray LAPACK kernel.
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
#[inline]
pub(crate) fn log_determinant_ndarray_lapack_lu(
    matrix: &Array2<f64>,
) -> Result<LogDetResult<f64>, LUError> {
    <NdarrayLapackLuKernel as LuKernel>::log_determinant(matrix)
}
