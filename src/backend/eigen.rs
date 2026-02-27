//! Internal Eigen kernel traits and backend adapters.

use std::marker::PhantomData;

use nalgebra::{DMatrix, RealField};
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
use nalgebra_lapack::SymmetricEigen as NalgebraLapackSymmetricEigen;
use ndarray::{Array1, Array2};
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
use ndarray_linalg::{Eigh, UPLO};
use num_traits::Float;

use crate::eigen::{
    EigenError, NalgebraEigenResult, NalgebraGeneralizedEigenResult, NdarrayEigenResult,
    NdarrayGeneralizedEigenResult,
};
use crate::interop::{nalgebra_to_ndarray, ndarray_to_nalgebra};

/// Internal kernel trait for eigen decomposition operations.
pub(crate) trait EigenKernel {
    type GeneralizedOutput;
    type Matrix;
    type SymmetricOutput;

    fn compute_symmetric_eigen(matrix: &Self::Matrix) -> Result<Self::SymmetricOutput, EigenError>;

    fn compute_generalized_eigen(
        matrix_a: &Self::Matrix,
        matrix_b: &Self::Matrix,
    ) -> Result<Self::GeneralizedOutput, EigenError>;
}

/// Nalgebra-backed eigen kernel.
pub(crate) struct NalgebraEigenKernel<T>(PhantomData<T>);

impl<T> EigenKernel for NalgebraEigenKernel<T>
where
    T: RealField + Copy + Float,
{
    type GeneralizedOutput = NalgebraGeneralizedEigenResult<T>;
    type Matrix = DMatrix<T>;
    type SymmetricOutput = NalgebraEigenResult<T>;

    #[inline]
    fn compute_symmetric_eigen(matrix: &Self::Matrix) -> Result<Self::SymmetricOutput, EigenError> {
        validate_nalgebra_symmetric_input(matrix)?;

        let eigen = matrix.clone().symmetric_eigen();
        if eigen.eigenvalues.iter().any(|&x| !Float::is_finite(x))
            || eigen.eigenvectors.iter().any(|&x| !Float::is_finite(x))
        {
            return Err(EigenError::NumericalInstability);
        }

        Ok(NalgebraEigenResult {
            eigenvalues:  eigen.eigenvalues,
            eigenvectors: eigen.eigenvectors,
        })
    }

    #[inline]
    fn compute_generalized_eigen(
        matrix_a: &Self::Matrix,
        matrix_b: &Self::Matrix,
    ) -> Result<Self::GeneralizedOutput, EigenError> {
        validate_nalgebra_generalized_input(matrix_a, matrix_b)?;

        let cholesky = nalgebra::linalg::Cholesky::new(matrix_b.clone())
            .ok_or(EigenError::NotPositiveDefinite)?;
        let l = cholesky.l();

        let linv_a = l.solve_lower_triangular(matrix_a).ok_or(EigenError::NumericalInstability)?;
        let c = l
            .transpose()
            .solve_upper_triangular(&linv_a)
            .ok_or(EigenError::NumericalInstability)?;

        let eigen = c.symmetric_eigen();
        let w = eigen.eigenvectors;
        let eigenvectors =
            l.transpose().solve_lower_triangular(&w).ok_or(EigenError::NumericalInstability)?;

        if eigen.eigenvalues.iter().any(|&x| !Float::is_finite(x))
            || eigenvectors.iter().any(|&x| !Float::is_finite(x))
        {
            return Err(EigenError::NumericalInstability);
        }

        Ok(NalgebraGeneralizedEigenResult { eigenvalues: eigen.eigenvalues, eigenvectors })
    }
}

/// Ndarray-backed eigen kernel.
pub(crate) struct NdarrayEigenKernel<T>(PhantomData<T>);

impl<T> EigenKernel for NdarrayEigenKernel<T>
where
    T: Float + RealField,
{
    type GeneralizedOutput = NdarrayGeneralizedEigenResult<T>;
    type Matrix = Array2<T>;
    type SymmetricOutput = NdarrayEigenResult<T>;

    #[inline]
    fn compute_symmetric_eigen(matrix: &Self::Matrix) -> Result<Self::SymmetricOutput, EigenError> {
        let nalg = ndarray_to_nalgebra(matrix);
        let result = <NalgebraEigenKernel<T> as EigenKernel>::compute_symmetric_eigen(&nalg)?;

        Ok(NdarrayEigenResult {
            eigenvalues:  Array1::from_vec(result.eigenvalues.as_slice().to_vec()),
            eigenvectors: nalgebra_to_ndarray(&result.eigenvectors),
        })
    }

    #[inline]
    fn compute_generalized_eigen(
        matrix_a: &Self::Matrix,
        matrix_b: &Self::Matrix,
    ) -> Result<Self::GeneralizedOutput, EigenError> {
        let nalg_a = ndarray_to_nalgebra(matrix_a);
        let nalg_b = ndarray_to_nalgebra(matrix_b);
        let result =
            <NalgebraEigenKernel<T> as EigenKernel>::compute_generalized_eigen(&nalg_a, &nalg_b)?;

        Ok(NdarrayGeneralizedEigenResult {
            eigenvalues:  Array1::from_vec(result.eigenvalues.as_slice().to_vec()),
            eigenvectors: nalgebra_to_ndarray(&result.eigenvectors),
        })
    }
}

/// Nalgebra LAPACK-backed eigen kernel.
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
pub(crate) struct NalgebraLapackEigenKernel;

#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
impl EigenKernel for NalgebraLapackEigenKernel {
    type GeneralizedOutput = NalgebraGeneralizedEigenResult<f64>;
    type Matrix = DMatrix<f64>;
    type SymmetricOutput = NalgebraEigenResult<f64>;

    #[inline]
    fn compute_symmetric_eigen(matrix: &Self::Matrix) -> Result<Self::SymmetricOutput, EigenError> {
        validate_nalgebra_symmetric_input(matrix)?;

        let eigen = NalgebraLapackSymmetricEigen::try_new(matrix.clone())
            .ok_or(EigenError::ConvergenceFailed)?;
        if eigen.eigenvalues.iter().any(|x| !x.is_finite())
            || eigen.eigenvectors.iter().any(|x| !x.is_finite())
        {
            return Err(EigenError::NumericalInstability);
        }

        Ok(NalgebraEigenResult {
            eigenvalues:  eigen.eigenvalues,
            eigenvectors: eigen.eigenvectors,
        })
    }

    #[inline]
    fn compute_generalized_eigen(
        matrix_a: &Self::Matrix,
        matrix_b: &Self::Matrix,
    ) -> Result<Self::GeneralizedOutput, EigenError> {
        validate_nalgebra_generalized_input(matrix_a, matrix_b)?;

        let cholesky = nalgebra::linalg::Cholesky::new(matrix_b.clone())
            .ok_or(EigenError::NotPositiveDefinite)?;
        let l = cholesky.l();

        let linv_a = l.solve_lower_triangular(matrix_a).ok_or(EigenError::NumericalInstability)?;
        let c = l
            .transpose()
            .solve_upper_triangular(&linv_a)
            .ok_or(EigenError::NumericalInstability)?;

        let eigen =
            NalgebraLapackSymmetricEigen::try_new(c).ok_or(EigenError::ConvergenceFailed)?;
        let eigenvectors = l
            .transpose()
            .solve_lower_triangular(&eigen.eigenvectors)
            .ok_or(EigenError::NumericalInstability)?;

        if eigen.eigenvalues.iter().any(|x| !x.is_finite())
            || eigenvectors.iter().any(|x| !x.is_finite())
        {
            return Err(EigenError::NumericalInstability);
        }

        Ok(NalgebraGeneralizedEigenResult { eigenvalues: eigen.eigenvalues, eigenvectors })
    }
}

/// Ndarray LAPACK-backed eigen kernel.
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
pub(crate) struct NdarrayLapackEigenKernel;

#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
impl EigenKernel for NdarrayLapackEigenKernel {
    type GeneralizedOutput = NdarrayGeneralizedEigenResult<f64>;
    type Matrix = Array2<f64>;
    type SymmetricOutput = NdarrayEigenResult<f64>;

    #[inline]
    fn compute_symmetric_eigen(matrix: &Self::Matrix) -> Result<Self::SymmetricOutput, EigenError> {
        validate_ndarray_symmetric_input(matrix)?;

        let (eigenvalues, eigenvectors) =
            matrix.view().eigh(UPLO::Lower).map_err(|_| EigenError::ConvergenceFailed)?;

        if eigenvalues.iter().any(|x| !x.is_finite()) || eigenvectors.iter().any(|x| !x.is_finite())
        {
            return Err(EigenError::NumericalInstability);
        }

        Ok(NdarrayEigenResult { eigenvalues, eigenvectors })
    }

    #[inline]
    fn compute_generalized_eigen(
        matrix_a: &Self::Matrix,
        matrix_b: &Self::Matrix,
    ) -> Result<Self::GeneralizedOutput, EigenError> {
        let nalg_a = ndarray_to_nalgebra(matrix_a);
        let nalg_b = ndarray_to_nalgebra(matrix_b);
        let result = <NalgebraLapackEigenKernel as EigenKernel>::compute_generalized_eigen(
            &nalg_a, &nalg_b,
        )?;

        Ok(NdarrayGeneralizedEigenResult {
            eigenvalues:  Array1::from_vec(result.eigenvalues.as_slice().to_vec()),
            eigenvectors: nalgebra_to_ndarray(&result.eigenvectors),
        })
    }
}

/// Dispatch symmetric eigen decomposition to the nalgebra kernel.
#[inline]
pub(crate) fn compute_nalgebra_symmetric_eigen<T>(
    matrix: &DMatrix<T>,
) -> Result<NalgebraEigenResult<T>, EigenError>
where
    T: RealField + Copy + Float,
{
    <NalgebraEigenKernel<T> as EigenKernel>::compute_symmetric_eigen(matrix)
}

/// Dispatch generalized eigen decomposition to the nalgebra kernel.
#[inline]
pub(crate) fn compute_nalgebra_generalized_eigen<T>(
    matrix_a: &DMatrix<T>,
    matrix_b: &DMatrix<T>,
) -> Result<NalgebraGeneralizedEigenResult<T>, EigenError>
where
    T: RealField + Copy + Float,
{
    <NalgebraEigenKernel<T> as EigenKernel>::compute_generalized_eigen(matrix_a, matrix_b)
}

/// Dispatch symmetric eigen decomposition to the ndarray kernel.
#[inline]
pub(crate) fn compute_ndarray_symmetric_eigen<T>(
    matrix: &Array2<T>,
) -> Result<NdarrayEigenResult<T>, EigenError>
where
    T: Float + RealField,
{
    <NdarrayEigenKernel<T> as EigenKernel>::compute_symmetric_eigen(matrix)
}

/// Dispatch generalized eigen decomposition to the ndarray kernel.
#[inline]
pub(crate) fn compute_ndarray_generalized_eigen<T>(
    matrix_a: &Array2<T>,
    matrix_b: &Array2<T>,
) -> Result<NdarrayGeneralizedEigenResult<T>, EigenError>
where
    T: Float + RealField,
{
    <NdarrayEigenKernel<T> as EigenKernel>::compute_generalized_eigen(matrix_a, matrix_b)
}

/// Dispatch symmetric eigen decomposition to the nalgebra LAPACK kernel.
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
#[inline]
pub(crate) fn compute_nalgebra_lapack_symmetric_eigen(
    matrix: &DMatrix<f64>,
) -> Result<NalgebraEigenResult<f64>, EigenError> {
    <NalgebraLapackEigenKernel as EigenKernel>::compute_symmetric_eigen(matrix)
}

/// Dispatch generalized eigen decomposition to the nalgebra LAPACK kernel.
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
#[inline]
pub(crate) fn compute_nalgebra_lapack_generalized_eigen(
    matrix_a: &DMatrix<f64>,
    matrix_b: &DMatrix<f64>,
) -> Result<NalgebraGeneralizedEigenResult<f64>, EigenError> {
    <NalgebraLapackEigenKernel as EigenKernel>::compute_generalized_eigen(matrix_a, matrix_b)
}

/// Dispatch symmetric eigen decomposition to the ndarray LAPACK kernel.
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
#[inline]
pub(crate) fn compute_ndarray_lapack_symmetric_eigen(
    matrix: &Array2<f64>,
) -> Result<NdarrayEigenResult<f64>, EigenError> {
    <NdarrayLapackEigenKernel as EigenKernel>::compute_symmetric_eigen(matrix)
}

/// Dispatch generalized eigen decomposition to the ndarray LAPACK kernel.
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
#[inline]
pub(crate) fn compute_ndarray_lapack_generalized_eigen(
    matrix_a: &Array2<f64>,
    matrix_b: &Array2<f64>,
) -> Result<NdarrayGeneralizedEigenResult<f64>, EigenError> {
    <NdarrayLapackEigenKernel as EigenKernel>::compute_generalized_eigen(matrix_a, matrix_b)
}

fn validate_nalgebra_symmetric_input<T>(matrix: &DMatrix<T>) -> Result<(), EigenError>
where
    T: RealField + Copy + Float,
{
    if matrix.is_empty() {
        return Err(EigenError::EmptyMatrix);
    }
    if !matrix.is_square() {
        return Err(EigenError::NotSquare);
    }

    let tolerance = T::from(1e-10).unwrap_or_else(T::nan);
    if !is_nalgebra_symmetric(matrix, tolerance) {
        return Err(EigenError::NonSymmetric);
    }

    if matrix.iter().any(|&x| !Float::is_finite(x)) {
        return Err(EigenError::NumericalInstability);
    }

    Ok(())
}

fn validate_nalgebra_generalized_input<T>(
    matrix_a: &DMatrix<T>,
    matrix_b: &DMatrix<T>,
) -> Result<(), EigenError>
where
    T: RealField + Copy + Float,
{
    if matrix_a.is_empty() || matrix_b.is_empty() {
        return Err(EigenError::EmptyMatrix);
    }

    let (ar, ac) = matrix_a.shape();
    let (br, bc) = matrix_b.shape();
    if ar != ac || br != bc || ar != br {
        return Err(EigenError::DimensionMismatch);
    }

    let tolerance = T::from(1e-10).unwrap_or_else(T::nan);
    if !is_nalgebra_symmetric(matrix_a, tolerance) || !is_nalgebra_symmetric(matrix_b, tolerance) {
        return Err(EigenError::NonSymmetric);
    }

    if matrix_a.iter().any(|&x| !Float::is_finite(x))
        || matrix_b.iter().any(|&x| !Float::is_finite(x))
    {
        return Err(EigenError::NumericalInstability);
    }

    Ok(())
}

#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
fn validate_ndarray_symmetric_input(matrix: &Array2<f64>) -> Result<(), EigenError> {
    if matrix.is_empty() {
        return Err(EigenError::EmptyMatrix);
    }

    let (rows, cols) = matrix.dim();
    if rows != cols {
        return Err(EigenError::NotSquare);
    }

    if !is_ndarray_symmetric(matrix, 1e-10_f64) {
        return Err(EigenError::NonSymmetric);
    }

    if matrix.iter().any(|x| !x.is_finite()) {
        return Err(EigenError::NumericalInstability);
    }

    Ok(())
}

fn is_nalgebra_symmetric<T>(matrix: &DMatrix<T>, tolerance: T) -> bool
where
    T: RealField + Copy,
{
    let (rows, cols) = matrix.shape();
    if rows != cols {
        return false;
    }

    for i in 0..rows {
        for j in (i + 1)..rows {
            if (matrix[(i, j)] - matrix[(j, i)]).abs() > tolerance {
                return false;
            }
        }
    }

    true
}

#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
fn is_ndarray_symmetric(matrix: &Array2<f64>, tolerance: f64) -> bool {
    let (rows, cols) = matrix.dim();
    if rows != cols {
        return false;
    }

    for i in 0..rows {
        for j in (i + 1)..rows {
            if (matrix[[i, j]] - matrix[[j, i]]).abs() > tolerance {
                return false;
            }
        }
    }

    true
}
