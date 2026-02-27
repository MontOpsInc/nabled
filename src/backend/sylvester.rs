//! Internal Sylvester/Lyapunov kernel traits and backend adapters.

use std::marker::PhantomData;

use nalgebra::{DMatrix, DVector, RealField};
use ndarray::Array2;
use num_traits::Float;

use crate::interop::{nalgebra_to_ndarray, ndarray_to_nalgebra};
use crate::sylvester::SylvesterError;

/// Internal kernel trait for Sylvester/Lyapunov operations.
pub(crate) trait SylvesterKernel {
    type Matrix;

    fn solve_sylvester(
        matrix_a: &Self::Matrix,
        matrix_b: &Self::Matrix,
        matrix_c: &Self::Matrix,
    ) -> Result<Self::Matrix, SylvesterError>;

    fn solve_lyapunov(a: &Self::Matrix, q: &Self::Matrix) -> Result<Self::Matrix, SylvesterError>;
}

/// Nalgebra-backed Sylvester/Lyapunov kernel.
pub(crate) struct NalgebraSylvesterKernel<T>(PhantomData<T>);

impl<T> SylvesterKernel for NalgebraSylvesterKernel<T>
where
    T: RealField + Copy + Float,
{
    type Matrix = DMatrix<T>;

    #[inline]
    fn solve_sylvester(
        matrix_a: &Self::Matrix,
        matrix_b: &Self::Matrix,
        matrix_c: &Self::Matrix,
    ) -> Result<Self::Matrix, SylvesterError> {
        solve_nalgebra_sylvester_with_schur(
            matrix_a,
            matrix_b,
            matrix_c,
            crate::backend::schur::compute_nalgebra_schur,
        )
    }

    #[inline]
    fn solve_lyapunov(a: &Self::Matrix, q: &Self::Matrix) -> Result<Self::Matrix, SylvesterError> {
        let at = a.transpose();
        Self::solve_sylvester(a, &at, q)
    }
}

/// Ndarray-backed Sylvester/Lyapunov kernel.
pub(crate) struct NdarraySylvesterKernel<T>(PhantomData<T>);

impl<T> SylvesterKernel for NdarraySylvesterKernel<T>
where
    T: Float + RealField,
{
    type Matrix = Array2<T>;

    #[inline]
    fn solve_sylvester(
        matrix_a: &Self::Matrix,
        matrix_b: &Self::Matrix,
        matrix_c: &Self::Matrix,
    ) -> Result<Self::Matrix, SylvesterError> {
        let nalg_a = ndarray_to_nalgebra(matrix_a);
        let nalg_b = ndarray_to_nalgebra(matrix_b);
        let nalg_c = ndarray_to_nalgebra(matrix_c);
        let result = <NalgebraSylvesterKernel<T> as SylvesterKernel>::solve_sylvester(
            &nalg_a, &nalg_b, &nalg_c,
        )?;
        Ok(nalgebra_to_ndarray(&result))
    }

    #[inline]
    fn solve_lyapunov(a: &Self::Matrix, q: &Self::Matrix) -> Result<Self::Matrix, SylvesterError> {
        let nalg_a = ndarray_to_nalgebra(a);
        let nalg_q = ndarray_to_nalgebra(q);
        let result =
            <NalgebraSylvesterKernel<T> as SylvesterKernel>::solve_lyapunov(&nalg_a, &nalg_q)?;
        Ok(nalgebra_to_ndarray(&result))
    }
}

/// Nalgebra LAPACK-backed Sylvester/Lyapunov kernel.
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
pub(crate) struct NalgebraLapackSylvesterKernel;

#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
impl SylvesterKernel for NalgebraLapackSylvesterKernel {
    type Matrix = DMatrix<f64>;

    #[inline]
    fn solve_sylvester(
        matrix_a: &Self::Matrix,
        matrix_b: &Self::Matrix,
        matrix_c: &Self::Matrix,
    ) -> Result<Self::Matrix, SylvesterError> {
        solve_nalgebra_sylvester_with_schur(
            matrix_a,
            matrix_b,
            matrix_c,
            crate::backend::schur::compute_nalgebra_lapack_schur,
        )
    }

    #[inline]
    fn solve_lyapunov(a: &Self::Matrix, q: &Self::Matrix) -> Result<Self::Matrix, SylvesterError> {
        let at = a.transpose();
        Self::solve_sylvester(a, &at, q)
    }
}

/// Ndarray LAPACK-backed Sylvester/Lyapunov kernel.
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
pub(crate) struct NdarrayLapackSylvesterKernel;

#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
impl SylvesterKernel for NdarrayLapackSylvesterKernel {
    type Matrix = Array2<f64>;

    #[inline]
    fn solve_sylvester(
        matrix_a: &Self::Matrix,
        matrix_b: &Self::Matrix,
        matrix_c: &Self::Matrix,
    ) -> Result<Self::Matrix, SylvesterError> {
        let nalg_a = ndarray_to_nalgebra(matrix_a);
        let nalg_b = ndarray_to_nalgebra(matrix_b);
        let nalg_c = ndarray_to_nalgebra(matrix_c);
        let result = <NalgebraLapackSylvesterKernel as SylvesterKernel>::solve_sylvester(
            &nalg_a, &nalg_b, &nalg_c,
        )?;
        Ok(nalgebra_to_ndarray(&result))
    }

    #[inline]
    fn solve_lyapunov(a: &Self::Matrix, q: &Self::Matrix) -> Result<Self::Matrix, SylvesterError> {
        let nalg_a = ndarray_to_nalgebra(a);
        let nalg_q = ndarray_to_nalgebra(q);
        let result =
            <NalgebraLapackSylvesterKernel as SylvesterKernel>::solve_lyapunov(&nalg_a, &nalg_q)?;
        Ok(nalgebra_to_ndarray(&result))
    }
}

/// Dispatch Sylvester solve to the nalgebra kernel.
#[inline]
pub(crate) fn solve_nalgebra_sylvester<T>(
    matrix_a: &DMatrix<T>,
    matrix_b: &DMatrix<T>,
    matrix_c: &DMatrix<T>,
) -> Result<DMatrix<T>, SylvesterError>
where
    T: RealField + Copy + Float,
{
    <NalgebraSylvesterKernel<T> as SylvesterKernel>::solve_sylvester(matrix_a, matrix_b, matrix_c)
}

/// Dispatch Lyapunov solve to the nalgebra kernel.
#[inline]
pub(crate) fn solve_nalgebra_lyapunov<T>(
    a: &DMatrix<T>,
    q: &DMatrix<T>,
) -> Result<DMatrix<T>, SylvesterError>
where
    T: RealField + Copy + Float,
{
    <NalgebraSylvesterKernel<T> as SylvesterKernel>::solve_lyapunov(a, q)
}

/// Dispatch Sylvester solve to the ndarray kernel.
#[inline]
pub(crate) fn solve_ndarray_sylvester<T>(
    matrix_a: &Array2<T>,
    matrix_b: &Array2<T>,
    matrix_c: &Array2<T>,
) -> Result<Array2<T>, SylvesterError>
where
    T: Float + RealField,
{
    <NdarraySylvesterKernel<T> as SylvesterKernel>::solve_sylvester(matrix_a, matrix_b, matrix_c)
}

/// Dispatch Lyapunov solve to the ndarray kernel.
#[inline]
pub(crate) fn solve_ndarray_lyapunov<T>(
    a: &Array2<T>,
    q: &Array2<T>,
) -> Result<Array2<T>, SylvesterError>
where
    T: Float + RealField,
{
    <NdarraySylvesterKernel<T> as SylvesterKernel>::solve_lyapunov(a, q)
}

/// Dispatch Sylvester solve to the nalgebra LAPACK kernel.
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
#[inline]
pub(crate) fn solve_nalgebra_lapack_sylvester(
    matrix_a: &DMatrix<f64>,
    matrix_b: &DMatrix<f64>,
    matrix_c: &DMatrix<f64>,
) -> Result<DMatrix<f64>, SylvesterError> {
    <NalgebraLapackSylvesterKernel as SylvesterKernel>::solve_sylvester(
        matrix_a, matrix_b, matrix_c,
    )
}

/// Dispatch Lyapunov solve to the nalgebra LAPACK kernel.
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
#[inline]
pub(crate) fn solve_nalgebra_lapack_lyapunov(
    a: &DMatrix<f64>,
    q: &DMatrix<f64>,
) -> Result<DMatrix<f64>, SylvesterError> {
    <NalgebraLapackSylvesterKernel as SylvesterKernel>::solve_lyapunov(a, q)
}

/// Dispatch Sylvester solve to the ndarray LAPACK kernel.
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
#[inline]
pub(crate) fn solve_ndarray_lapack_sylvester(
    matrix_a: &Array2<f64>,
    matrix_b: &Array2<f64>,
    matrix_c: &Array2<f64>,
) -> Result<Array2<f64>, SylvesterError> {
    <NdarrayLapackSylvesterKernel as SylvesterKernel>::solve_sylvester(matrix_a, matrix_b, matrix_c)
}

/// Dispatch Lyapunov solve to the ndarray LAPACK kernel.
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
#[inline]
pub(crate) fn solve_ndarray_lapack_lyapunov(
    a: &Array2<f64>,
    q: &Array2<f64>,
) -> Result<Array2<f64>, SylvesterError> {
    <NdarrayLapackSylvesterKernel as SylvesterKernel>::solve_lyapunov(a, q)
}

fn solve_nalgebra_sylvester_with_schur<T, F>(
    matrix_a: &DMatrix<T>,
    matrix_b: &DMatrix<T>,
    matrix_c: &DMatrix<T>,
    schur_fn: F,
) -> Result<DMatrix<T>, SylvesterError>
where
    T: RealField + Copy + Float,
    F: Fn(&DMatrix<T>) -> Result<crate::schur::NalgebraSchurResult<T>, crate::schur::SchurError>,
{
    let (m, n) = (matrix_a.nrows(), matrix_b.ncols());
    if matrix_a.is_empty() || matrix_b.is_empty() || matrix_c.is_empty() {
        return Err(SylvesterError::EmptyMatrix);
    }
    if matrix_a.nrows() != matrix_a.ncols() || matrix_b.nrows() != matrix_b.ncols() {
        return Err(SylvesterError::DimensionMismatch);
    }
    if matrix_a.nrows() != matrix_c.nrows() || matrix_b.ncols() != matrix_c.ncols() {
        return Err(SylvesterError::DimensionMismatch);
    }

    let schur_a = schur_fn(matrix_a)?;
    let schur_b = schur_fn(matrix_b)?;

    let ta = schur_a.t;
    let tb = schur_b.t;
    let qa = schur_a.q;
    let qb = schur_b.q;

    let d = qa.transpose() * matrix_c * &qb;

    let mut y = DMatrix::zeros(m, n);
    for j in 0..n {
        let mut rhs = d.column(j).clone_owned();
        for k in 0..j {
            rhs -= y.column(k) * tb[(k, j)];
        }

        let shift = DVector::from_element(m, tb[(j, j)]);
        let diag = &ta + DMatrix::from_diagonal(&shift);
        let inv = diag.try_inverse().ok_or(SylvesterError::SingularSystem)?;
        let col = &inv * rhs;
        y.set_column(j, &col);
    }

    Ok(&qa * &y * qb.transpose())
}
