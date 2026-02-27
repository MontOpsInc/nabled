//! Internal QR kernel traits and backend adapters.

use std::marker::PhantomData;

use nalgebra::linalg::ColPivQR;
use nalgebra::{DMatrix, DVector, RealField};
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
use nalgebra_lapack::QR as NalgebraLapackQr;
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
use nalgebra_lapack::qr::QrDecomposition;
use ndarray::{Array1, Array2};
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
use ndarray_linalg::QR as NdarrayLinalgQr;
use num_traits::Float;
use num_traits::float::FloatCore;

use crate::interop::ndarray_to_nalgebra;
use crate::qr::{QRConfig, QRError, QRResult};

/// Internal kernel trait for QR operations.
pub(crate) trait QrKernel {
    type Scalar;
    type Matrix;
    type Vector;
    type Decomposition;

    fn compute_qr(
        matrix: &Self::Matrix,
        config: &QRConfig<Self::Scalar>,
    ) -> Result<Self::Decomposition, QRError>;

    fn compute_reduced_qr(
        matrix: &Self::Matrix,
        config: &QRConfig<Self::Scalar>,
    ) -> Result<Self::Decomposition, QRError>;

    fn compute_qr_with_pivoting(
        matrix: &Self::Matrix,
        config: &QRConfig<Self::Scalar>,
    ) -> Result<Self::Decomposition, QRError>;

    fn solve_least_squares(
        matrix: &Self::Matrix,
        rhs: &Self::Vector,
        config: &QRConfig<Self::Scalar>,
    ) -> Result<Self::Vector, QRError>;
}

/// Nalgebra-backed QR kernel.
pub(crate) struct NalgebraQrKernel<T>(PhantomData<T>);

impl<T> QrKernel for NalgebraQrKernel<T>
where
    T: RealField + FloatCore + Float,
{
    type Decomposition = QRResult<T>;
    type Matrix = DMatrix<T>;
    type Scalar = T;
    type Vector = DVector<T>;

    #[inline]
    fn compute_qr(
        matrix: &Self::Matrix,
        config: &QRConfig<Self::Scalar>,
    ) -> Result<Self::Decomposition, QRError> {
        if matrix.is_empty() {
            return Err(QRError::EmptyMatrix);
        }

        if matrix.nrows() == 1 && matrix.ncols() == 1 {
            let val = matrix[(0, 0)];
            if Float::is_finite(val) {
                return Ok(QRResult {
                    q:    DMatrix::from_element(1, 1, T::one()),
                    r:    DMatrix::from_element(1, 1, val),
                    p:    None,
                    rank: usize::from(Float::abs(val) >= config.rank_tolerance),
                });
            }
            return Err(QRError::NumericalInstability);
        }

        if matrix.iter().all(|&x| Float::abs(x) < config.rank_tolerance) {
            let (m, n) = matrix.shape();
            let min_dim = m.min(n);
            return Ok(QRResult {
                q:    DMatrix::identity(m, min_dim),
                r:    DMatrix::zeros(min_dim, n),
                p:    None,
                rank: 0,
            });
        }

        if matrix.iter().any(|&x| !Float::is_finite(x)) {
            return Err(QRError::NumericalInstability);
        }

        if config.rank_tolerance <= T::zero() {
            return Err(QRError::InvalidInput("Rank tolerance must be positive".to_string()));
        }

        if config.rank_tolerance < T::from(1e-15).unwrap() {
            return Err(QRError::InvalidInput(
                "Rank tolerance too small, may cause numerical issues".to_string(),
            ));
        }

        let qr = matrix.clone().qr();
        let q = qr.q();
        let r = qr.r();

        if q.iter().any(|&x| !Float::is_finite(x)) || r.iter().any(|&x| !Float::is_finite(x)) {
            return Err(QRError::NumericalInstability);
        }

        let rank = determine_rank(&r, config.rank_tolerance);
        Ok(QRResult { q: q.clone(), r: r.clone(), p: None, rank })
    }

    #[inline]
    fn compute_reduced_qr(
        matrix: &Self::Matrix,
        config: &QRConfig<Self::Scalar>,
    ) -> Result<Self::Decomposition, QRError> {
        let full_qr = Self::compute_qr(matrix, config)?;
        let (m, n) = matrix.shape();
        let min_dim = m.min(n);

        let q_reduced = full_qr.q.columns(0, min_dim);
        let r_reduced = full_qr.r.rows(0, min_dim);

        Ok(QRResult {
            q:    q_reduced.into(),
            r:    r_reduced.into(),
            p:    full_qr.p,
            rank: full_qr.rank,
        })
    }

    #[inline]
    fn compute_qr_with_pivoting(
        matrix: &Self::Matrix,
        config: &QRConfig<Self::Scalar>,
    ) -> Result<Self::Decomposition, QRError> {
        if matrix.is_empty() {
            return Err(QRError::EmptyMatrix);
        }

        let col_piv_qr = ColPivQR::new(matrix.clone());
        let q = col_piv_qr.q().clone();
        let r = col_piv_qr.r().clone();
        let p_seq = col_piv_qr.p();

        let n_cols = matrix.ncols();
        let mut p_matrix = DMatrix::identity(n_cols, n_cols);
        p_seq.permute_columns(&mut p_matrix);

        let rank = determine_rank(&r, config.rank_tolerance);
        Ok(QRResult { q, r, p: Some(p_matrix), rank })
    }

    #[inline]
    fn solve_least_squares(
        matrix: &Self::Matrix,
        rhs: &Self::Vector,
        config: &QRConfig<Self::Scalar>,
    ) -> Result<Self::Vector, QRError> {
        if matrix.is_empty() || rhs.is_empty() {
            return Err(QRError::EmptyMatrix);
        }

        if matrix.iter().any(|&x| !Float::is_finite(x)) || rhs.iter().any(|&x| !Float::is_finite(x))
        {
            return Err(QRError::NumericalInstability);
        }

        let (m, n) = matrix.shape();
        if m != rhs.len() {
            return Err(QRError::InvalidDimensions(format!(
                "Matrix rows ({}) must match RHS length ({})",
                m,
                rhs.len()
            )));
        }

        if n > m {
            return Err(QRError::InvalidDimensions(
                "Underdetermined system: more unknowns than equations".to_string(),
            ));
        }

        if m == 1 {
            if n == 1 {
                let a_val = matrix[(0, 0)];
                let b_val = rhs[0];
                if Float::abs(a_val) < config.rank_tolerance {
                    return Err(QRError::SingularMatrix);
                }
                return Ok(DVector::from_vec(vec![b_val / a_val]));
            }
            return Err(QRError::InvalidDimensions(
                "Single equation with multiple unknowns".to_string(),
            ));
        }

        let qr = Self::compute_qr(matrix, config)?;
        if qr.rank < n {
            return Err(QRError::SingularMatrix);
        }

        let qt_b = qr.q.transpose() * rhs;
        if qt_b.iter().any(|&x| !Float::is_finite(x)) {
            return Err(QRError::NumericalInstability);
        }

        let mut x = DVector::zeros(n);
        for i in (0..n).rev() {
            if i >= qt_b.len() {
                continue;
            }

            let mut sum = qt_b[i];
            for j in (i + 1)..n {
                sum -= qr.r[(i, j)] * x[j];
            }

            if FloatCore::abs(qr.r[(i, i)]) < config.rank_tolerance {
                return Err(QRError::SingularMatrix);
            }

            x[i] = sum / qr.r[(i, i)];
            if !Float::is_finite(x[i]) {
                return Err(QRError::NumericalInstability);
            }
        }

        Ok(x)
    }
}

/// Ndarray-backed QR kernel.
pub(crate) struct NdarrayQrKernel<T>(PhantomData<T>);

impl<T> QrKernel for NdarrayQrKernel<T>
where
    T: RealField + FloatCore + Float,
{
    type Decomposition = QRResult<T>;
    type Matrix = Array2<T>;
    type Scalar = T;
    type Vector = Array1<T>;

    #[inline]
    fn compute_qr(
        matrix: &Self::Matrix,
        config: &QRConfig<Self::Scalar>,
    ) -> Result<Self::Decomposition, QRError> {
        let nalgebra_matrix = ndarray_to_nalgebra(matrix);
        <NalgebraQrKernel<T> as QrKernel>::compute_qr(&nalgebra_matrix, config)
    }

    #[inline]
    fn compute_reduced_qr(
        matrix: &Self::Matrix,
        config: &QRConfig<Self::Scalar>,
    ) -> Result<Self::Decomposition, QRError> {
        let nalgebra_matrix = ndarray_to_nalgebra(matrix);
        <NalgebraQrKernel<T> as QrKernel>::compute_reduced_qr(&nalgebra_matrix, config)
    }

    #[inline]
    fn compute_qr_with_pivoting(
        matrix: &Self::Matrix,
        config: &QRConfig<Self::Scalar>,
    ) -> Result<Self::Decomposition, QRError> {
        let nalgebra_matrix = ndarray_to_nalgebra(matrix);
        <NalgebraQrKernel<T> as QrKernel>::compute_qr_with_pivoting(&nalgebra_matrix, config)
    }

    #[inline]
    fn solve_least_squares(
        matrix: &Self::Matrix,
        rhs: &Self::Vector,
        config: &QRConfig<Self::Scalar>,
    ) -> Result<Self::Vector, QRError> {
        let nalgebra_matrix = ndarray_to_nalgebra(matrix);
        let nalgebra_rhs = ndarray_to_nalgebra_vector(rhs);
        let solution = <NalgebraQrKernel<T> as QrKernel>::solve_least_squares(
            &nalgebra_matrix,
            &nalgebra_rhs,
            config,
        )?;
        Ok(nalgebra_to_ndarray_vector(&solution))
    }
}

/// Nalgebra LAPACK-backed QR kernel.
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
pub(crate) struct NalgebraLapackQrKernel;

#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
impl QrKernel for NalgebraLapackQrKernel {
    type Decomposition = QRResult<f64>;
    type Matrix = DMatrix<f64>;
    type Scalar = f64;
    type Vector = DVector<f64>;

    #[inline]
    fn compute_qr(
        matrix: &Self::Matrix,
        config: &QRConfig<Self::Scalar>,
    ) -> Result<Self::Decomposition, QRError> {
        if matrix.is_empty() {
            return Err(QRError::EmptyMatrix);
        }

        if matrix.iter().any(|x| !x.is_finite()) {
            return Err(QRError::NumericalInstability);
        }

        if config.rank_tolerance <= 0.0 {
            return Err(QRError::InvalidInput("Rank tolerance must be positive".to_string()));
        }
        if config.rank_tolerance < 1e-15 {
            return Err(QRError::InvalidInput(
                "Rank tolerance too small, may cause numerical issues".to_string(),
            ));
        }

        let qr = NalgebraLapackQr::new(matrix.clone()).map_err(|_| QRError::ConvergenceFailed)?;
        let q = qr.q();
        let r = qr.r();

        if q.iter().any(|x: &f64| !x.is_finite()) || r.iter().any(|x: &f64| !x.is_finite()) {
            return Err(QRError::NumericalInstability);
        }

        let rank = determine_rank(&r, config.rank_tolerance);
        Ok(QRResult { q, r, p: None, rank })
    }

    #[inline]
    fn compute_reduced_qr(
        matrix: &Self::Matrix,
        config: &QRConfig<Self::Scalar>,
    ) -> Result<Self::Decomposition, QRError> {
        let full_qr = Self::compute_qr(matrix, config)?;
        let (m, n) = matrix.shape();
        let min_dim = m.min(n);

        let q_reduced = full_qr.q.columns(0, min_dim);
        let r_reduced = full_qr.r.rows(0, min_dim);

        Ok(QRResult {
            q:    q_reduced.into(),
            r:    r_reduced.into(),
            p:    None,
            rank: full_qr.rank,
        })
    }

    #[inline]
    fn compute_qr_with_pivoting(
        matrix: &Self::Matrix,
        config: &QRConfig<Self::Scalar>,
    ) -> Result<Self::Decomposition, QRError> {
        <NalgebraQrKernel<f64> as QrKernel>::compute_qr_with_pivoting(matrix, config)
    }

    #[inline]
    fn solve_least_squares(
        matrix: &Self::Matrix,
        rhs: &Self::Vector,
        config: &QRConfig<Self::Scalar>,
    ) -> Result<Self::Vector, QRError> {
        if matrix.is_empty() || rhs.is_empty() {
            return Err(QRError::EmptyMatrix);
        }

        if matrix.iter().any(|x| !x.is_finite()) || rhs.iter().any(|x| !x.is_finite()) {
            return Err(QRError::NumericalInstability);
        }

        let (m, n) = matrix.shape();
        if m != rhs.len() {
            return Err(QRError::InvalidDimensions(format!(
                "Matrix rows ({}) must match RHS length ({})",
                m,
                rhs.len()
            )));
        }

        if n > m {
            return Err(QRError::InvalidDimensions(
                "Underdetermined system: more unknowns than equations".to_string(),
            ));
        }

        if m == 1 {
            let a_val = matrix[(0, 0)];
            if a_val.abs() < config.rank_tolerance {
                return Err(QRError::SingularMatrix);
            }
            return Ok(DVector::from_vec(vec![rhs[0] / a_val]));
        }

        let qr = Self::compute_qr(matrix, config)?;
        if qr.rank < n {
            return Err(QRError::SingularMatrix);
        }

        let qt_b = qr.q.transpose() * rhs;
        if qt_b.iter().any(|x| !x.is_finite()) {
            return Err(QRError::NumericalInstability);
        }

        let mut x = DVector::zeros(n);
        for i in (0..n).rev() {
            let mut sum = qt_b[i];
            for j in (i + 1)..n {
                sum -= qr.r[(i, j)] * x[j];
            }

            if qr.r[(i, i)].abs() < config.rank_tolerance {
                return Err(QRError::SingularMatrix);
            }

            x[i] = sum / qr.r[(i, i)];
            if !x[i].is_finite() {
                return Err(QRError::NumericalInstability);
            }
        }

        Ok(x)
    }
}

/// Ndarray LAPACK-backed QR kernel.
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
pub(crate) struct NdarrayLapackQrKernel;

#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
impl QrKernel for NdarrayLapackQrKernel {
    type Decomposition = QRResult<f64>;
    type Matrix = Array2<f64>;
    type Scalar = f64;
    type Vector = Array1<f64>;

    #[inline]
    fn compute_qr(
        matrix: &Self::Matrix,
        config: &QRConfig<Self::Scalar>,
    ) -> Result<Self::Decomposition, QRError> {
        if matrix.is_empty() {
            return Err(QRError::EmptyMatrix);
        }

        if matrix.iter().any(|x| !x.is_finite()) {
            return Err(QRError::NumericalInstability);
        }

        if config.rank_tolerance <= 0.0 {
            return Err(QRError::InvalidInput("Rank tolerance must be positive".to_string()));
        }
        if config.rank_tolerance < 1e-15 {
            return Err(QRError::InvalidInput(
                "Rank tolerance too small, may cause numerical issues".to_string(),
            ));
        }

        let (q_nd, r_nd) = matrix.view().qr().map_err(|_| QRError::ConvergenceFailed)?;
        if q_nd.iter().any(|x| !x.is_finite()) || r_nd.iter().any(|x| !x.is_finite()) {
            return Err(QRError::NumericalInstability);
        }

        let q = ndarray_to_nalgebra(&q_nd);
        let r = ndarray_to_nalgebra(&r_nd);
        let rank = determine_rank(&r, config.rank_tolerance);

        Ok(QRResult { q, r, p: None, rank })
    }

    #[inline]
    fn compute_reduced_qr(
        matrix: &Self::Matrix,
        config: &QRConfig<Self::Scalar>,
    ) -> Result<Self::Decomposition, QRError> {
        let full_qr = Self::compute_qr(matrix, config)?;
        let (m, n) = matrix.dim();
        let min_dim = m.min(n);

        let q_reduced = full_qr.q.columns(0, min_dim);
        let r_reduced = full_qr.r.rows(0, min_dim);

        Ok(QRResult {
            q:    q_reduced.into(),
            r:    r_reduced.into(),
            p:    None,
            rank: full_qr.rank,
        })
    }

    #[inline]
    fn compute_qr_with_pivoting(
        matrix: &Self::Matrix,
        config: &QRConfig<Self::Scalar>,
    ) -> Result<Self::Decomposition, QRError> {
        <NdarrayQrKernel<f64> as QrKernel>::compute_qr_with_pivoting(matrix, config)
    }

    #[inline]
    fn solve_least_squares(
        matrix: &Self::Matrix,
        rhs: &Self::Vector,
        config: &QRConfig<Self::Scalar>,
    ) -> Result<Self::Vector, QRError> {
        let nalgebra_matrix = ndarray_to_nalgebra(matrix);
        let nalgebra_rhs = ndarray_to_nalgebra_vector(rhs);
        let solution = <NalgebraLapackQrKernel as QrKernel>::solve_least_squares(
            &nalgebra_matrix,
            &nalgebra_rhs,
            config,
        )?;
        Ok(nalgebra_to_ndarray_vector(&solution))
    }
}

/// Dispatch QR decomposition to the nalgebra kernel.
#[inline]
pub(crate) fn compute_nalgebra_qr<T>(
    matrix: &DMatrix<T>,
    config: &QRConfig<T>,
) -> Result<QRResult<T>, QRError>
where
    T: RealField + FloatCore + Float,
{
    <NalgebraQrKernel<T> as QrKernel>::compute_qr(matrix, config)
}

/// Dispatch reduced QR decomposition to the nalgebra kernel.
#[inline]
pub(crate) fn compute_nalgebra_reduced_qr<T>(
    matrix: &DMatrix<T>,
    config: &QRConfig<T>,
) -> Result<QRResult<T>, QRError>
where
    T: RealField + FloatCore + Float,
{
    <NalgebraQrKernel<T> as QrKernel>::compute_reduced_qr(matrix, config)
}

/// Dispatch pivoted QR decomposition to the nalgebra kernel.
#[inline]
pub(crate) fn compute_nalgebra_qr_with_pivoting<T>(
    matrix: &DMatrix<T>,
    config: &QRConfig<T>,
) -> Result<QRResult<T>, QRError>
where
    T: RealField + FloatCore + Float,
{
    <NalgebraQrKernel<T> as QrKernel>::compute_qr_with_pivoting(matrix, config)
}

/// Dispatch least-squares solving to the nalgebra kernel.
#[inline]
pub(crate) fn solve_nalgebra_least_squares<T>(
    matrix: &DMatrix<T>,
    rhs: &DVector<T>,
    config: &QRConfig<T>,
) -> Result<DVector<T>, QRError>
where
    T: RealField + FloatCore + Float,
{
    <NalgebraQrKernel<T> as QrKernel>::solve_least_squares(matrix, rhs, config)
}

/// Dispatch QR decomposition to the ndarray kernel.
#[inline]
pub(crate) fn compute_ndarray_qr<T>(
    matrix: &Array2<T>,
    config: &QRConfig<T>,
) -> Result<QRResult<T>, QRError>
where
    T: Float + FloatCore + RealField,
{
    <NdarrayQrKernel<T> as QrKernel>::compute_qr(matrix, config)
}

/// Dispatch QR decomposition to the nalgebra LAPACK kernel.
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
#[inline]
pub(crate) fn compute_nalgebra_lapack_qr(
    matrix: &DMatrix<f64>,
    config: &QRConfig<f64>,
) -> Result<QRResult<f64>, QRError> {
    <NalgebraLapackQrKernel as QrKernel>::compute_qr(matrix, config)
}

/// Dispatch reduced QR decomposition to the nalgebra LAPACK kernel.
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
#[inline]
pub(crate) fn compute_nalgebra_lapack_reduced_qr(
    matrix: &DMatrix<f64>,
    config: &QRConfig<f64>,
) -> Result<QRResult<f64>, QRError> {
    <NalgebraLapackQrKernel as QrKernel>::compute_reduced_qr(matrix, config)
}

/// Dispatch least-squares solving to the nalgebra LAPACK kernel.
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
#[inline]
pub(crate) fn solve_nalgebra_lapack_least_squares(
    matrix: &DMatrix<f64>,
    rhs: &DVector<f64>,
    config: &QRConfig<f64>,
) -> Result<DVector<f64>, QRError> {
    <NalgebraLapackQrKernel as QrKernel>::solve_least_squares(matrix, rhs, config)
}

/// Dispatch reduced QR decomposition to the ndarray kernel.
#[inline]
pub(crate) fn compute_ndarray_reduced_qr<T>(
    matrix: &Array2<T>,
    config: &QRConfig<T>,
) -> Result<QRResult<T>, QRError>
where
    T: Float + FloatCore + RealField,
{
    <NdarrayQrKernel<T> as QrKernel>::compute_reduced_qr(matrix, config)
}

/// Dispatch QR decomposition to the ndarray LAPACK kernel.
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
#[inline]
pub(crate) fn compute_ndarray_lapack_qr(
    matrix: &Array2<f64>,
    config: &QRConfig<f64>,
) -> Result<QRResult<f64>, QRError> {
    <NdarrayLapackQrKernel as QrKernel>::compute_qr(matrix, config)
}

/// Dispatch reduced QR decomposition to the ndarray LAPACK kernel.
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
#[inline]
pub(crate) fn compute_ndarray_lapack_reduced_qr(
    matrix: &Array2<f64>,
    config: &QRConfig<f64>,
) -> Result<QRResult<f64>, QRError> {
    <NdarrayLapackQrKernel as QrKernel>::compute_reduced_qr(matrix, config)
}

/// Dispatch least-squares solving to the ndarray LAPACK kernel.
#[cfg(all(feature = "lapack-kernels", target_os = "linux"))]
#[inline]
pub(crate) fn solve_ndarray_lapack_least_squares(
    matrix: &Array2<f64>,
    rhs: &Array1<f64>,
    config: &QRConfig<f64>,
) -> Result<Array1<f64>, QRError> {
    <NdarrayLapackQrKernel as QrKernel>::solve_least_squares(matrix, rhs, config)
}

/// Dispatch pivoted QR decomposition to the ndarray kernel.
#[inline]
pub(crate) fn compute_ndarray_qr_with_pivoting<T>(
    matrix: &Array2<T>,
    config: &QRConfig<T>,
) -> Result<QRResult<T>, QRError>
where
    T: Float + FloatCore + RealField,
{
    <NdarrayQrKernel<T> as QrKernel>::compute_qr_with_pivoting(matrix, config)
}

/// Dispatch least-squares solving to the ndarray kernel.
#[inline]
pub(crate) fn solve_ndarray_least_squares<T>(
    matrix: &Array2<T>,
    rhs: &Array1<T>,
    config: &QRConfig<T>,
) -> Result<Array1<T>, QRError>
where
    T: Float + FloatCore + RealField,
{
    <NdarrayQrKernel<T> as QrKernel>::solve_least_squares(matrix, rhs, config)
}

fn determine_rank<T>(r: &DMatrix<T>, tolerance: T) -> usize
where
    T: RealField + FloatCore,
{
    let (m, n) = r.shape();
    let min_dim = m.min(n);

    let mut rank = 0;
    for i in 0..min_dim {
        if FloatCore::abs(r[(i, i)]) > tolerance {
            rank += 1;
        }
    }

    rank
}

fn ndarray_to_nalgebra_vector<T>(array: &Array1<T>) -> DVector<T>
where
    T: RealField,
{
    DVector::from_vec(array.to_vec())
}

fn nalgebra_to_ndarray_vector<T>(vector: &DVector<T>) -> Array1<T>
where
    T: Float + FloatCore,
{
    Array1::from_vec(vector.as_slice().to_vec())
}
