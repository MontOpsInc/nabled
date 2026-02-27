//! Internal matrix-function kernel traits and backend adapters.

use std::marker::PhantomData;

use nalgebra::{DMatrix, RealField};
use ndarray::Array2;
use num_traits::Float;
use num_traits::float::FloatCore;

use crate::interop::{nalgebra_to_ndarray, ndarray_to_nalgebra};
use crate::matrix_functions::MatrixFunctionError;

/// Internal kernel trait for matrix-function operations.
pub(crate) trait MatrixFunctionsKernel {
    type Matrix;
    type Scalar;

    fn matrix_exp(
        matrix: &Self::Matrix,
        max_iterations: usize,
        tolerance: Self::Scalar,
    ) -> Result<Self::Matrix, MatrixFunctionError>;

    fn matrix_exp_eigen(matrix: &Self::Matrix) -> Result<Self::Matrix, MatrixFunctionError>;

    fn matrix_log_taylor(
        matrix: &Self::Matrix,
        max_iterations: usize,
        tolerance: Self::Scalar,
    ) -> Result<Self::Matrix, MatrixFunctionError>;

    fn matrix_log_eigen(matrix: &Self::Matrix) -> Result<Self::Matrix, MatrixFunctionError>;

    fn matrix_log_svd(matrix: &Self::Matrix) -> Result<Self::Matrix, MatrixFunctionError>;

    fn matrix_power(
        matrix: &Self::Matrix,
        power: Self::Scalar,
    ) -> Result<Self::Matrix, MatrixFunctionError>;

    fn matrix_sign(matrix: &Self::Matrix) -> Result<Self::Matrix, MatrixFunctionError>;
}

/// Nalgebra-backed matrix-functions kernel.
pub(crate) struct NalgebraMatrixFunctionsKernel<T>(PhantomData<T>);

impl<T> MatrixFunctionsKernel for NalgebraMatrixFunctionsKernel<T>
where
    T: RealField + FloatCore,
{
    type Matrix = DMatrix<T>;
    type Scalar = T;

    #[inline]
    fn matrix_exp(
        matrix: &Self::Matrix,
        max_iterations: usize,
        tolerance: Self::Scalar,
    ) -> Result<Self::Matrix, MatrixFunctionError> {
        matrix_exp_impl(matrix, max_iterations, tolerance)
    }

    #[inline]
    fn matrix_exp_eigen(matrix: &Self::Matrix) -> Result<Self::Matrix, MatrixFunctionError> {
        matrix_exp_eigen_impl(matrix)
    }

    #[inline]
    fn matrix_log_taylor(
        matrix: &Self::Matrix,
        max_iterations: usize,
        tolerance: Self::Scalar,
    ) -> Result<Self::Matrix, MatrixFunctionError> {
        matrix_log_taylor_impl(matrix, max_iterations, tolerance)
    }

    #[inline]
    fn matrix_log_eigen(matrix: &Self::Matrix) -> Result<Self::Matrix, MatrixFunctionError> {
        matrix_log_eigen_impl(matrix)
    }

    #[inline]
    fn matrix_log_svd(matrix: &Self::Matrix) -> Result<Self::Matrix, MatrixFunctionError> {
        matrix_log_svd_impl(matrix)
    }

    #[inline]
    fn matrix_power(
        matrix: &Self::Matrix,
        power: Self::Scalar,
    ) -> Result<Self::Matrix, MatrixFunctionError> {
        matrix_power_impl(matrix, power)
    }

    #[inline]
    fn matrix_sign(matrix: &Self::Matrix) -> Result<Self::Matrix, MatrixFunctionError> {
        matrix_sign_impl(matrix)
    }
}

/// Ndarray-backed matrix-functions kernel.
pub(crate) struct NdarrayMatrixFunctionsKernel<T>(PhantomData<T>);

impl<T> MatrixFunctionsKernel for NdarrayMatrixFunctionsKernel<T>
where
    T: Float + RealField + FloatCore,
{
    type Matrix = Array2<T>;
    type Scalar = T;

    #[inline]
    fn matrix_exp(
        matrix: &Self::Matrix,
        max_iterations: usize,
        tolerance: Self::Scalar,
    ) -> Result<Self::Matrix, MatrixFunctionError> {
        let nalgebra_matrix = ndarray_to_nalgebra(matrix);
        let result = <NalgebraMatrixFunctionsKernel<T> as MatrixFunctionsKernel>::matrix_exp(
            &nalgebra_matrix,
            max_iterations,
            tolerance,
        )?;
        Ok(nalgebra_to_ndarray(&result))
    }

    #[inline]
    fn matrix_exp_eigen(matrix: &Self::Matrix) -> Result<Self::Matrix, MatrixFunctionError> {
        let nalgebra_matrix = ndarray_to_nalgebra(matrix);
        let result = <NalgebraMatrixFunctionsKernel<T> as MatrixFunctionsKernel>::matrix_exp_eigen(
            &nalgebra_matrix,
        )?;
        Ok(nalgebra_to_ndarray(&result))
    }

    #[inline]
    fn matrix_log_taylor(
        matrix: &Self::Matrix,
        max_iterations: usize,
        tolerance: Self::Scalar,
    ) -> Result<Self::Matrix, MatrixFunctionError> {
        let nalgebra_matrix = ndarray_to_nalgebra(matrix);
        let result =
            <NalgebraMatrixFunctionsKernel<T> as MatrixFunctionsKernel>::matrix_log_taylor(
                &nalgebra_matrix,
                max_iterations,
                tolerance,
            )?;
        Ok(nalgebra_to_ndarray(&result))
    }

    #[inline]
    fn matrix_log_eigen(matrix: &Self::Matrix) -> Result<Self::Matrix, MatrixFunctionError> {
        let nalgebra_matrix = ndarray_to_nalgebra(matrix);
        let result = <NalgebraMatrixFunctionsKernel<T> as MatrixFunctionsKernel>::matrix_log_eigen(
            &nalgebra_matrix,
        )?;
        Ok(nalgebra_to_ndarray(&result))
    }

    #[inline]
    fn matrix_log_svd(matrix: &Self::Matrix) -> Result<Self::Matrix, MatrixFunctionError> {
        let nalgebra_matrix = ndarray_to_nalgebra(matrix);
        let result = <NalgebraMatrixFunctionsKernel<T> as MatrixFunctionsKernel>::matrix_log_svd(
            &nalgebra_matrix,
        )?;
        Ok(nalgebra_to_ndarray(&result))
    }

    #[inline]
    fn matrix_power(
        matrix: &Self::Matrix,
        power: Self::Scalar,
    ) -> Result<Self::Matrix, MatrixFunctionError> {
        let nalgebra_matrix = ndarray_to_nalgebra(matrix);
        let result = <NalgebraMatrixFunctionsKernel<T> as MatrixFunctionsKernel>::matrix_power(
            &nalgebra_matrix,
            power,
        )?;
        Ok(nalgebra_to_ndarray(&result))
    }

    #[inline]
    fn matrix_sign(matrix: &Self::Matrix) -> Result<Self::Matrix, MatrixFunctionError> {
        let nalgebra_matrix = ndarray_to_nalgebra(matrix);
        let result = <NalgebraMatrixFunctionsKernel<T> as MatrixFunctionsKernel>::matrix_sign(
            &nalgebra_matrix,
        )?;
        Ok(nalgebra_to_ndarray(&result))
    }
}

/// Dispatch matrix exponential to the nalgebra kernel.
#[inline]
pub(crate) fn matrix_exp_nalgebra<T>(
    matrix: &DMatrix<T>,
    max_iterations: usize,
    tolerance: T,
) -> Result<DMatrix<T>, MatrixFunctionError>
where
    T: RealField + FloatCore,
{
    <NalgebraMatrixFunctionsKernel<T> as MatrixFunctionsKernel>::matrix_exp(
        matrix,
        max_iterations,
        tolerance,
    )
}

/// Dispatch matrix exponential to the ndarray kernel.
#[inline]
pub(crate) fn matrix_exp_ndarray<T>(
    matrix: &Array2<T>,
    max_iterations: usize,
    tolerance: T,
) -> Result<Array2<T>, MatrixFunctionError>
where
    T: Float + RealField + FloatCore,
{
    <NdarrayMatrixFunctionsKernel<T> as MatrixFunctionsKernel>::matrix_exp(
        matrix,
        max_iterations,
        tolerance,
    )
}

/// Dispatch eigen-based matrix exponential to the nalgebra kernel.
#[inline]
pub(crate) fn matrix_exp_eigen_nalgebra<T>(
    matrix: &DMatrix<T>,
) -> Result<DMatrix<T>, MatrixFunctionError>
where
    T: RealField + FloatCore,
{
    <NalgebraMatrixFunctionsKernel<T> as MatrixFunctionsKernel>::matrix_exp_eigen(matrix)
}

/// Dispatch eigen-based matrix exponential to the ndarray kernel.
#[inline]
pub(crate) fn matrix_exp_eigen_ndarray<T>(
    matrix: &Array2<T>,
) -> Result<Array2<T>, MatrixFunctionError>
where
    T: Float + RealField + FloatCore,
{
    <NdarrayMatrixFunctionsKernel<T> as MatrixFunctionsKernel>::matrix_exp_eigen(matrix)
}

/// Dispatch Taylor-series matrix logarithm to the nalgebra kernel.
#[inline]
pub(crate) fn matrix_log_taylor_nalgebra<T>(
    matrix: &DMatrix<T>,
    max_iterations: usize,
    tolerance: T,
) -> Result<DMatrix<T>, MatrixFunctionError>
where
    T: RealField + FloatCore,
{
    <NalgebraMatrixFunctionsKernel<T> as MatrixFunctionsKernel>::matrix_log_taylor(
        matrix,
        max_iterations,
        tolerance,
    )
}

/// Dispatch Taylor-series matrix logarithm to the ndarray kernel.
#[inline]
pub(crate) fn matrix_log_taylor_ndarray<T>(
    matrix: &Array2<T>,
    max_iterations: usize,
    tolerance: T,
) -> Result<Array2<T>, MatrixFunctionError>
where
    T: Float + RealField + FloatCore,
{
    <NdarrayMatrixFunctionsKernel<T> as MatrixFunctionsKernel>::matrix_log_taylor(
        matrix,
        max_iterations,
        tolerance,
    )
}

/// Dispatch eigen-based matrix logarithm to the nalgebra kernel.
#[inline]
pub(crate) fn matrix_log_eigen_nalgebra<T>(
    matrix: &DMatrix<T>,
) -> Result<DMatrix<T>, MatrixFunctionError>
where
    T: RealField + FloatCore,
{
    <NalgebraMatrixFunctionsKernel<T> as MatrixFunctionsKernel>::matrix_log_eigen(matrix)
}

/// Dispatch eigen-based matrix logarithm to the ndarray kernel.
#[inline]
pub(crate) fn matrix_log_eigen_ndarray<T>(
    matrix: &Array2<T>,
) -> Result<Array2<T>, MatrixFunctionError>
where
    T: Float + RealField + FloatCore,
{
    <NdarrayMatrixFunctionsKernel<T> as MatrixFunctionsKernel>::matrix_log_eigen(matrix)
}

/// Dispatch SVD-based matrix logarithm to the nalgebra kernel.
#[inline]
pub(crate) fn matrix_log_svd_nalgebra<T>(
    matrix: &DMatrix<T>,
) -> Result<DMatrix<T>, MatrixFunctionError>
where
    T: RealField + FloatCore,
{
    <NalgebraMatrixFunctionsKernel<T> as MatrixFunctionsKernel>::matrix_log_svd(matrix)
}

/// Dispatch SVD-based matrix logarithm to the ndarray kernel.
#[inline]
pub(crate) fn matrix_log_svd_ndarray<T>(
    matrix: &Array2<T>,
) -> Result<Array2<T>, MatrixFunctionError>
where
    T: Float + RealField + FloatCore,
{
    <NdarrayMatrixFunctionsKernel<T> as MatrixFunctionsKernel>::matrix_log_svd(matrix)
}

/// Dispatch matrix power to the nalgebra kernel.
#[inline]
pub(crate) fn matrix_power_nalgebra<T>(
    matrix: &DMatrix<T>,
    power: T,
) -> Result<DMatrix<T>, MatrixFunctionError>
where
    T: RealField + FloatCore,
{
    <NalgebraMatrixFunctionsKernel<T> as MatrixFunctionsKernel>::matrix_power(matrix, power)
}

/// Dispatch matrix power to the ndarray kernel.
#[inline]
pub(crate) fn matrix_power_ndarray<T>(
    matrix: &Array2<T>,
    power: T,
) -> Result<Array2<T>, MatrixFunctionError>
where
    T: Float + RealField + FloatCore,
{
    <NdarrayMatrixFunctionsKernel<T> as MatrixFunctionsKernel>::matrix_power(matrix, power)
}

/// Dispatch matrix sign to the nalgebra kernel.
#[inline]
pub(crate) fn matrix_sign_nalgebra<T>(
    matrix: &DMatrix<T>,
) -> Result<DMatrix<T>, MatrixFunctionError>
where
    T: RealField + FloatCore,
{
    <NalgebraMatrixFunctionsKernel<T> as MatrixFunctionsKernel>::matrix_sign(matrix)
}

/// Dispatch matrix sign to the ndarray kernel.
#[inline]
pub(crate) fn matrix_sign_ndarray<T>(matrix: &Array2<T>) -> Result<Array2<T>, MatrixFunctionError>
where
    T: Float + RealField + FloatCore,
{
    <NdarrayMatrixFunctionsKernel<T> as MatrixFunctionsKernel>::matrix_sign(matrix)
}

#[inline]
fn is_symmetric<T: RealField + FloatCore>(matrix: &DMatrix<T>, tol: T) -> bool {
    if !matrix.is_square() {
        return false;
    }
    let n = matrix.nrows();
    for i in 0..n {
        for j in (i + 1)..n {
            if FloatCore::abs(matrix[(i, j)] - matrix[(j, i)]) > tol {
                return false;
            }
        }
    }
    true
}

fn validate_square_non_empty<T>(matrix: &DMatrix<T>) -> Result<(), MatrixFunctionError>
where
    T: RealField,
{
    if matrix.is_empty() {
        return Err(MatrixFunctionError::EmptyMatrix);
    }
    if !matrix.is_square() {
        return Err(MatrixFunctionError::NotSquare);
    }
    Ok(())
}

fn matrix_exp_impl<T: RealField + FloatCore>(
    matrix: &DMatrix<T>,
    max_iterations: usize,
    tolerance: T,
) -> Result<DMatrix<T>, MatrixFunctionError> {
    validate_square_non_empty(matrix)?;

    let n = matrix.nrows();
    let identity = DMatrix::<T>::identity(n, n);
    let mut result = identity.clone();
    let mut term = identity;

    for k in 1..=max_iterations {
        term = &term * matrix / T::from(k).unwrap();
        let new_result = &result + &term;

        if (&new_result - &result).norm() < tolerance {
            return Ok(new_result);
        }

        result = new_result;
    }

    Err(MatrixFunctionError::ConvergenceFailed)
}

fn matrix_exp_eigen_impl<T: RealField + FloatCore>(
    matrix: &DMatrix<T>,
) -> Result<DMatrix<T>, MatrixFunctionError> {
    validate_square_non_empty(matrix)?;

    let tol = T::from_f64(1e-10).unwrap_or_else(T::epsilon);
    if is_symmetric(matrix, tol) {
        let eigen = matrix.clone().symmetric_eigen();
        let exp_eigenvalues = eigen.eigenvalues.map(nalgebra::ComplexField::exp);
        let exp_diagonal = DMatrix::from_diagonal(&exp_eigenvalues);
        Ok(&eigen.eigenvectors * &exp_diagonal * eigen.eigenvectors.transpose())
    } else {
        matrix_exp_impl(matrix, 100, tol)
    }
}

fn matrix_log_taylor_impl<T: RealField + FloatCore>(
    matrix: &DMatrix<T>,
    max_iterations: usize,
    tolerance: T,
) -> Result<DMatrix<T>, MatrixFunctionError> {
    validate_square_non_empty(matrix)?;

    let n = matrix.nrows();
    let identity = DMatrix::<T>::identity(n, n);
    let diff = matrix - &identity;
    if diff.norm() >= T::one() {
        return Err(MatrixFunctionError::InvalidInput(
            "Matrix must be close to identity (||A - I|| < 1) for Taylor series".to_string(),
        ));
    }

    let mut result = DMatrix::<T>::zeros(n, n);
    let mut term = diff.clone();
    for k in 1..=max_iterations {
        let sign = if k % 2 == 1 { T::one() } else { -T::one() };
        let coeff = sign / T::from(k).unwrap();

        let new_term = &term * coeff;
        let new_result = &result + &new_term;
        if (&new_result - &result).norm() < tolerance {
            return Ok(new_result);
        }

        result = new_result;
        term = &term * &diff;
    }

    Err(MatrixFunctionError::ConvergenceFailed)
}

fn matrix_log_eigen_impl<T: RealField + FloatCore>(
    matrix: &DMatrix<T>,
) -> Result<DMatrix<T>, MatrixFunctionError> {
    validate_square_non_empty(matrix)?;

    let tol = T::from_f64(1e-10).unwrap_or_else(T::epsilon);
    if !is_symmetric(matrix, tol) {
        return Err(MatrixFunctionError::InvalidInput(
            "matrix_log_eigen requires a symmetric matrix; use matrix_log_svd for general matrices"
                .to_string(),
        ));
    }

    let eigen = matrix.clone().symmetric_eigen();
    for &lambda in eigen.eigenvalues.iter() {
        if lambda <= T::zero() {
            return Err(MatrixFunctionError::NegativeEigenvalues);
        }
    }

    let log_eigenvalues = eigen.eigenvalues.map(nalgebra::ComplexField::ln);
    let log_diagonal = DMatrix::from_diagonal(&log_eigenvalues);
    Ok(&eigen.eigenvectors * &log_diagonal * eigen.eigenvectors.transpose())
}

fn matrix_log_svd_impl<T: RealField + FloatCore>(
    matrix: &DMatrix<T>,
) -> Result<DMatrix<T>, MatrixFunctionError> {
    validate_square_non_empty(matrix)?;

    let svd = crate::backend::svd::compute_nalgebra_svd(matrix)
        .map_err(|_| MatrixFunctionError::ConvergenceFailed)?;

    for &sigma in svd.singular_values.iter() {
        if sigma <= T::zero() {
            return Err(MatrixFunctionError::SingularMatrix);
        }
    }

    let log_singular_values = svd.singular_values.map(nalgebra::ComplexField::ln);
    let log_diagonal = DMatrix::from_diagonal(&log_singular_values);
    Ok(&svd.u * &log_diagonal * &svd.vt)
}

fn matrix_power_impl<T: RealField + FloatCore>(
    matrix: &DMatrix<T>,
    power: T,
) -> Result<DMatrix<T>, MatrixFunctionError> {
    validate_square_non_empty(matrix)?;

    let tol = T::from_f64(1e-10).unwrap_or_else(T::epsilon);
    if !is_symmetric(matrix, tol) {
        return Err(MatrixFunctionError::InvalidInput(
            "matrix_power currently requires a symmetric matrix".to_string(),
        ));
    }

    let eigen = matrix.clone().symmetric_eigen();
    if FloatCore::fract(power) != T::zero() {
        for &lambda in eigen.eigenvalues.iter() {
            if lambda <= T::zero() {
                return Err(MatrixFunctionError::NegativeEigenvalues);
            }
        }
    }

    let powered_eigenvalues = eigen.eigenvalues.map(|lambda| lambda.powf(power));
    let powered_diagonal = DMatrix::from_diagonal(&powered_eigenvalues);
    Ok(&eigen.eigenvectors * &powered_diagonal * eigen.eigenvectors.transpose())
}

fn matrix_sign_impl<T: RealField + FloatCore>(
    matrix: &DMatrix<T>,
) -> Result<DMatrix<T>, MatrixFunctionError> {
    validate_square_non_empty(matrix)?;

    let tol = T::from_f64(1e-10).unwrap_or_else(T::epsilon);
    if !is_symmetric(matrix, tol) {
        return Err(MatrixFunctionError::InvalidInput(
            "matrix_sign requires a symmetric matrix".to_string(),
        ));
    }

    let eigen = matrix.clone().symmetric_eigen();
    let sign_eigenvalues = eigen
        .eigenvalues
        .iter()
        .map(|&lambda| {
            if lambda > T::zero() {
                Ok(T::one())
            } else if lambda < T::zero() {
                Ok(-T::one())
            } else {
                Err(MatrixFunctionError::ZeroEigenvalue)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;

    let sign_diagonal = DMatrix::from_diagonal(&nalgebra::DVector::from_vec(sign_eigenvalues));
    Ok(&eigen.eigenvectors * &sign_diagonal * eigen.eigenvectors.transpose())
}
