//! Iterative linear system solvers over ndarray matrices.

use std::fmt;

use nabled_core::scalar::NabledReal;
use nabled_linalg::lu;
use ndarray::{Array1, Array2, ArrayBase, ArrayView1, ArrayView2, Data, Ix1};
use num_complex::Complex64;

const DEFAULT_TOLERANCE: f64 = 1.0e-12;

/// Configuration for iterative solvers.
#[derive(Debug, Clone)]
pub struct IterativeConfig<T = f64> {
    /// Relative residual tolerance.
    pub tolerance:      T,
    /// Maximum iterations.
    pub max_iterations: usize,
}

impl IterativeConfig<f64> {
    /// Default configuration for `f64`.
    #[must_use]
    pub const fn default_f64() -> Self { Self { tolerance: 1e-10, max_iterations: 1000 } }
}

impl Default for IterativeConfig<f64> {
    fn default() -> Self { Self::default_f64() }
}

impl IterativeConfig<f32> {
    /// Default configuration for `f32`.
    #[must_use]
    pub const fn default_f32() -> Self { Self { tolerance: 1e-6, max_iterations: 1000 } }
}

impl Default for IterativeConfig<f32> {
    fn default() -> Self { Self::default_f32() }
}

/// Error type for iterative solvers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IterativeError {
    /// Matrix is empty.
    EmptyMatrix,
    /// Dimensions do not align.
    DimensionMismatch,
    /// Maximum iterations reached without convergence.
    MaxIterationsExceeded,
    /// Matrix is not positive definite (CG).
    NotPositiveDefinite,
    /// Algorithm breakdown.
    Breakdown,
}

impl fmt::Display for IterativeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IterativeError::EmptyMatrix => write!(f, "Matrix is empty"),
            IterativeError::DimensionMismatch => write!(f, "Dimension mismatch"),
            IterativeError::MaxIterationsExceeded => write!(f, "Maximum iterations exceeded"),
            IterativeError::NotPositiveDefinite => write!(f, "Matrix is not positive definite"),
            IterativeError::Breakdown => write!(f, "Algorithm breakdown"),
        }
    }
}

impl std::error::Error for IterativeError {}

fn default_tolerance<T: NabledReal>() -> T {
    T::from_f64(DEFAULT_TOLERANCE).unwrap_or_else(T::epsilon)
}

fn vector_norm<T, S>(vector: &ArrayBase<S, Ix1>) -> T
where
    T: NabledReal,
    S: Data<Elem = T>,
{
    vector
        .iter()
        .map(|value| *value * *value)
        .fold(T::zero(), |acc, value| acc + value)
        .sqrt()
}

fn vector_norm_complex(vector: &Array1<Complex64>) -> f64 {
    vector.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt()
}

#[cfg(all(feature = "lapack-provider", feature = "magma-system"))]
trait IterativeLinearScalar:
    NabledReal + std::ops::SubAssign + ndarray_linalg::Lapack + lu::LuProviderScalar
{
}

#[cfg(all(feature = "lapack-provider", feature = "magma-system"))]
impl<T> IterativeLinearScalar for T where
    T: NabledReal + std::ops::SubAssign + ndarray_linalg::Lapack + lu::LuProviderScalar
{
}

#[cfg(all(feature = "lapack-provider", not(feature = "magma-system")))]
trait IterativeLinearScalar: NabledReal + std::ops::SubAssign + ndarray_linalg::Lapack {}

#[cfg(all(feature = "lapack-provider", not(feature = "magma-system")))]
impl<T> IterativeLinearScalar for T where
    T: NabledReal + std::ops::SubAssign + ndarray_linalg::Lapack
{
}

#[cfg(all(not(feature = "lapack-provider"), feature = "magma-system"))]
trait IterativeLinearScalar: NabledReal + std::ops::SubAssign + lu::LuProviderScalar {}

#[cfg(all(not(feature = "lapack-provider"), feature = "magma-system"))]
impl<T> IterativeLinearScalar for T where T: NabledReal + std::ops::SubAssign + lu::LuProviderScalar {}

#[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
trait IterativeLinearScalar: NabledReal + std::ops::SubAssign {}

#[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
impl<T> IterativeLinearScalar for T where T: NabledReal + std::ops::SubAssign {}

/// Conjugate Gradient for SPD systems `Ax=b`.
///
/// # Errors
/// Returns an error when inputs are invalid or convergence fails.
pub fn conjugate_gradient<T>(
    matrix_a: &Array2<T>,
    matrix_b: &Array1<T>,
    config: &IterativeConfig<T>,
) -> Result<Array1<T>, IterativeError>
where
    T: NabledReal,
{
    conjugate_gradient_view(&matrix_a.view(), &matrix_b.view(), config)
}

/// Conjugate Gradient for SPD systems `Ax=b` from matrix/vector views.
///
/// # Errors
/// Returns an error when inputs are invalid or convergence fails.
pub fn conjugate_gradient_view<T>(
    matrix_a: &ArrayView2<'_, T>,
    matrix_b: &ArrayView1<'_, T>,
    config: &IterativeConfig<T>,
) -> Result<Array1<T>, IterativeError>
where
    T: NabledReal,
{
    if matrix_a.is_empty() || matrix_b.is_empty() {
        return Err(IterativeError::EmptyMatrix);
    }
    if matrix_a.nrows() != matrix_a.ncols() || matrix_a.nrows() != matrix_b.len() {
        return Err(IterativeError::DimensionMismatch);
    }

    let n = matrix_b.len();
    let mut x = Array1::<T>::zeros(n);
    let mut r = matrix_b.to_owned();
    let mut p = r.clone();
    let mut rs_old = r.dot(&r);

    let tolerance = config.tolerance.max(default_tolerance::<T>());
    if rs_old.sqrt() <= tolerance {
        return Ok(x);
    }

    for _ in 0..config.max_iterations {
        let ap = matrix_a.dot(&p);
        let curvature = p.dot(&ap);
        if curvature <= tolerance {
            return Err(IterativeError::NotPositiveDefinite);
        }

        let alpha = rs_old / curvature;
        x = &x + &p.mapv(|value| alpha * value);
        r = &r - &ap.mapv(|value| alpha * value);

        let rs_new = r.dot(&r);
        if rs_new.sqrt() <= tolerance {
            return Ok(x);
        }

        let beta = rs_new / rs_old;
        p = &r + &p.mapv(|value| beta * value);
        rs_old = rs_new;
    }

    Err(IterativeError::MaxIterationsExceeded)
}

/// Conjugate Gradient for Hermitian positive-definite systems `Ax=b`.
///
/// # Errors
/// Returns an error when inputs are invalid or convergence fails.
pub fn conjugate_gradient_complex(
    matrix_a: &Array2<Complex64>,
    matrix_b: &Array1<Complex64>,
    config: &IterativeConfig<f64>,
) -> Result<Array1<Complex64>, IterativeError> {
    if matrix_a.is_empty() || matrix_b.is_empty() {
        return Err(IterativeError::EmptyMatrix);
    }
    if matrix_a.nrows() != matrix_a.ncols() || matrix_a.nrows() != matrix_b.len() {
        return Err(IterativeError::DimensionMismatch);
    }

    let n = matrix_b.len();
    let mut x = Array1::<Complex64>::zeros(n);
    let mut r = matrix_b.clone();
    let mut p = r.clone();
    let mut rs_old = r.iter().zip(r.iter()).map(|(lhs, rhs)| lhs.conj() * rhs).sum::<Complex64>();
    let tolerance = config.tolerance.max(DEFAULT_TOLERANCE);

    if rs_old.re.max(0.0).sqrt() <= tolerance {
        return Ok(x);
    }

    for _ in 0..config.max_iterations {
        let ap = matrix_a.dot(&p);
        let curvature =
            p.iter().zip(ap.iter()).map(|(lhs, rhs)| lhs.conj() * rhs).sum::<Complex64>();
        if curvature.re <= tolerance || curvature.norm() <= tolerance {
            return Err(IterativeError::NotPositiveDefinite);
        }

        let alpha = rs_old / curvature;
        x = &x + &(alpha * &p);
        r = &r - &(alpha * &ap);

        let rs_new = r.iter().zip(r.iter()).map(|(lhs, rhs)| lhs.conj() * rhs).sum::<Complex64>();
        if rs_new.re.max(0.0).sqrt() <= tolerance {
            return Ok(x);
        }

        if rs_old.norm() <= tolerance {
            return Err(IterativeError::Breakdown);
        }
        let beta = rs_new / rs_old;
        p = &r + &(beta * &p);
        rs_old = rs_new;
    }

    Err(IterativeError::MaxIterationsExceeded)
}

fn solve_linear<T>(
    matrix: &ArrayView2<'_, T>,
    rhs: &ArrayView1<'_, T>,
) -> Result<Array1<T>, IterativeError>
where
    T: IterativeLinearScalar,
{
    lu::solve_view(matrix, rhs).map_err(|_| IterativeError::Breakdown)
}

/// GMRES for general systems `Ax=b`.
///
/// # Errors
/// Returns an error when inputs are invalid or convergence fails.
#[allow(clippy::many_single_char_names)]
#[cfg(all(feature = "lapack-provider", feature = "magma-system"))]
pub fn gmres<T>(
    matrix_a: &Array2<T>,
    matrix_b: &Array1<T>,
    config: &IterativeConfig<T>,
) -> Result<Array1<T>, IterativeError>
where
    T: NabledReal + std::ops::SubAssign + ndarray_linalg::Lapack + lu::LuProviderScalar,
{
    gmres_view(&matrix_a.view(), &matrix_b.view(), config)
}

/// GMRES for general systems `Ax=b`.
///
/// # Errors
/// Returns an error when inputs are invalid or convergence fails.
#[allow(clippy::many_single_char_names)]
#[cfg(all(feature = "lapack-provider", not(feature = "magma-system")))]
pub fn gmres<T>(
    matrix_a: &Array2<T>,
    matrix_b: &Array1<T>,
    config: &IterativeConfig<T>,
) -> Result<Array1<T>, IterativeError>
where
    T: NabledReal + std::ops::SubAssign + ndarray_linalg::Lapack,
{
    gmres_view(&matrix_a.view(), &matrix_b.view(), config)
}

/// GMRES for general systems `Ax=b`.
///
/// # Errors
/// Returns an error when inputs are invalid or convergence fails.
#[allow(clippy::many_single_char_names)]
#[cfg(all(not(feature = "lapack-provider"), feature = "magma-system"))]
pub fn gmres<T>(
    matrix_a: &Array2<T>,
    matrix_b: &Array1<T>,
    config: &IterativeConfig<T>,
) -> Result<Array1<T>, IterativeError>
where
    T: NabledReal + std::ops::SubAssign + lu::LuProviderScalar,
{
    gmres_view(&matrix_a.view(), &matrix_b.view(), config)
}

/// GMRES for general systems `Ax=b`.
///
/// # Errors
/// Returns an error when inputs are invalid or convergence fails.
#[allow(clippy::many_single_char_names)]
#[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
pub fn gmres<T>(
    matrix_a: &Array2<T>,
    matrix_b: &Array1<T>,
    config: &IterativeConfig<T>,
) -> Result<Array1<T>, IterativeError>
where
    T: NabledReal + std::ops::SubAssign,
{
    gmres_view(&matrix_a.view(), &matrix_b.view(), config)
}

#[allow(clippy::many_single_char_names)]
fn gmres_impl<T>(
    matrix_a: &ArrayView2<'_, T>,
    matrix_b: &ArrayView1<'_, T>,
    config: &IterativeConfig<T>,
) -> Result<Array1<T>, IterativeError>
where
    T: IterativeLinearScalar,
{
    if matrix_a.is_empty() || matrix_b.is_empty() {
        return Err(IterativeError::EmptyMatrix);
    }
    if matrix_a.nrows() != matrix_a.ncols() || matrix_a.nrows() != matrix_b.len() {
        return Err(IterativeError::DimensionMismatch);
    }

    let n = matrix_b.len();
    let m = n.min(config.max_iterations.max(1));
    let mut basis = Array2::<T>::zeros((n, m + 1));
    let mut hessenberg = Array2::<T>::zeros((m + 1, m));

    let beta = vector_norm(matrix_b);
    let tolerance = config.tolerance.max(default_tolerance::<T>());
    if beta <= tolerance {
        return Ok(Array1::<T>::zeros(n));
    }

    for row in 0..n {
        basis[[row, 0]] = matrix_b[row] / beta;
    }

    let mut effective_m = m;
    for j in 0..m {
        let mut w = matrix_a.dot(&basis.column(j));

        for i in 0..=j {
            let vi = basis.column(i);
            let hij = vi.dot(&w);
            hessenberg[[i, j]] = hij;
            for row in 0..n {
                w[row] -= hij * basis[[row, i]];
            }
        }

        let norm_w = vector_norm(&w);
        hessenberg[[j + 1, j]] = norm_w;
        if norm_w <= tolerance {
            effective_m = j + 1;
            break;
        }
        for row in 0..n {
            basis[[row, j + 1]] = w[row] / norm_w;
        }
    }

    let h = hessenberg.slice(ndarray::s![..(effective_m + 1), ..effective_m]);
    let ht = h.t();
    let normal_matrix = ht.dot(&h);

    let mut rhs_ls = Array1::<T>::zeros(effective_m + 1);
    rhs_ls[0] = beta;
    let normal_rhs = ht.dot(&rhs_ls);

    let y = solve_linear(&normal_matrix.view(), &normal_rhs.view())?;
    let x = basis.slice(ndarray::s![.., ..effective_m]).dot(&y);

    let residual = matrix_b - &matrix_a.dot(&x);
    if vector_norm(&residual) <= tolerance {
        Ok(x)
    } else {
        Err(IterativeError::MaxIterationsExceeded)
    }
}

/// GMRES for general systems `Ax=b` from matrix/vector views.
///
/// # Errors
/// Returns an error when inputs are invalid or convergence fails.
#[allow(clippy::many_single_char_names)]
#[cfg(all(feature = "lapack-provider", feature = "magma-system"))]
pub fn gmres_view<T>(
    matrix_a: &ArrayView2<'_, T>,
    matrix_b: &ArrayView1<'_, T>,
    config: &IterativeConfig<T>,
) -> Result<Array1<T>, IterativeError>
where
    T: NabledReal + std::ops::SubAssign + ndarray_linalg::Lapack + lu::LuProviderScalar,
{
    gmres_impl(matrix_a, matrix_b, config)
}

/// GMRES for general systems `Ax=b` from matrix/vector views.
///
/// # Errors
/// Returns an error when inputs are invalid or convergence fails.
#[allow(clippy::many_single_char_names)]
#[cfg(all(feature = "lapack-provider", not(feature = "magma-system")))]
pub fn gmres_view<T>(
    matrix_a: &ArrayView2<'_, T>,
    matrix_b: &ArrayView1<'_, T>,
    config: &IterativeConfig<T>,
) -> Result<Array1<T>, IterativeError>
where
    T: NabledReal + std::ops::SubAssign + ndarray_linalg::Lapack,
{
    gmres_impl(matrix_a, matrix_b, config)
}

/// GMRES for general systems `Ax=b` from matrix/vector views.
///
/// # Errors
/// Returns an error when inputs are invalid or convergence fails.
#[allow(clippy::many_single_char_names)]
#[cfg(all(not(feature = "lapack-provider"), feature = "magma-system"))]
pub fn gmres_view<T>(
    matrix_a: &ArrayView2<'_, T>,
    matrix_b: &ArrayView1<'_, T>,
    config: &IterativeConfig<T>,
) -> Result<Array1<T>, IterativeError>
where
    T: NabledReal + std::ops::SubAssign + lu::LuProviderScalar,
{
    gmres_impl(matrix_a, matrix_b, config)
}

/// GMRES for general systems `Ax=b` from matrix/vector views.
///
/// # Errors
/// Returns an error when inputs are invalid or convergence fails.
#[allow(clippy::many_single_char_names)]
#[cfg(not(any(feature = "lapack-provider", feature = "magma-system")))]
pub fn gmres_view<T>(
    matrix_a: &ArrayView2<'_, T>,
    matrix_b: &ArrayView1<'_, T>,
    config: &IterativeConfig<T>,
) -> Result<Array1<T>, IterativeError>
where
    T: NabledReal + std::ops::SubAssign,
{
    gmres_impl(matrix_a, matrix_b, config)
}

/// GMRES for complex general systems `Ax=b`.
///
/// # Errors
/// Returns an error when inputs are invalid or convergence fails.
#[allow(clippy::many_single_char_names)]
pub fn gmres_complex(
    matrix_a: &Array2<Complex64>,
    matrix_b: &Array1<Complex64>,
    config: &IterativeConfig<f64>,
) -> Result<Array1<Complex64>, IterativeError> {
    if matrix_a.is_empty() || matrix_b.is_empty() {
        return Err(IterativeError::EmptyMatrix);
    }
    if matrix_a.nrows() != matrix_a.ncols() || matrix_a.nrows() != matrix_b.len() {
        return Err(IterativeError::DimensionMismatch);
    }

    let n = matrix_b.len();
    let m = n.min(config.max_iterations.max(1));
    let mut basis = Array2::<Complex64>::zeros((n, m + 1));
    let mut hessenberg = Array2::<Complex64>::zeros((m + 1, m));
    let tolerance = config.tolerance.max(DEFAULT_TOLERANCE);

    let beta = vector_norm_complex(matrix_b);
    if beta <= tolerance {
        return Ok(Array1::<Complex64>::zeros(n));
    }

    for row in 0..n {
        basis[[row, 0]] = matrix_b[row] / beta;
    }

    let mut effective_m = m;
    for j in 0..m {
        let mut w = matrix_a.dot(&basis.column(j));

        for i in 0..=j {
            let vi = basis.column(i);
            let hij = vi.iter().zip(w.iter()).map(|(lhs, rhs)| lhs.conj() * rhs).sum::<Complex64>();
            hessenberg[[i, j]] = hij;
            for row in 0..n {
                w[row] -= hij * basis[[row, i]];
            }
        }

        let norm_w = vector_norm_complex(&w);
        hessenberg[[j + 1, j]] = Complex64::new(norm_w, 0.0);
        if norm_w <= tolerance {
            effective_m = j + 1;
            break;
        }
        for row in 0..n {
            basis[[row, j + 1]] = w[row] / norm_w;
        }
    }

    let h = hessenberg.slice(ndarray::s![..(effective_m + 1), ..effective_m]);
    let h_conj_t = h.mapv(|value| value.conj()).reversed_axes();
    let normal_matrix = h_conj_t.dot(&h);

    let mut rhs_ls = Array1::<Complex64>::zeros(effective_m + 1);
    rhs_ls[0] = Complex64::new(beta, 0.0);
    let normal_rhs = h_conj_t.dot(&rhs_ls);

    let y =
        lu::solve_complex(&normal_matrix, &normal_rhs).map_err(|_| IterativeError::Breakdown)?;
    let x = basis.slice(ndarray::s![.., ..effective_m]).dot(&y);

    let residual = matrix_b - &matrix_a.dot(&x);
    if vector_norm_complex(&residual) <= tolerance {
        Ok(x)
    } else {
        Err(IterativeError::MaxIterationsExceeded)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2};
    use num_complex::Complex64;

    use super::*;

    #[test]
    fn cg_solves_spd_system() {
        let matrix = Array2::from_shape_vec((2, 2), vec![4.0_f64, 1.0, 1.0, 3.0]).unwrap();
        let rhs = Array1::from_vec(vec![1.0_f64, 2.0]);
        let solution =
            conjugate_gradient(&matrix, &rhs, &IterativeConfig::<f64>::default()).unwrap();
        let reconstructed = matrix.dot(&solution);
        assert!((reconstructed[0] - rhs[0]).abs() < 1e-8);
        assert!((reconstructed[1] - rhs[1]).abs() < 1e-8);
    }

    #[test]
    fn gmres_solves_small_system() {
        let matrix = Array2::from_shape_vec((2, 2), vec![3.0_f64, 1.0, 1.0, 2.0]).unwrap();
        let rhs = Array1::from_vec(vec![9.0_f64, 8.0]);
        let solution = gmres(&matrix, &rhs, &IterativeConfig::<f64>::default()).unwrap();
        let reconstructed = matrix.dot(&solution);
        assert!((reconstructed[0] - rhs[0]).abs() < 1e-8);
        assert!((reconstructed[1] - rhs[1]).abs() < 1e-8);
    }

    #[test]
    fn real_f32_solvers_work() {
        let matrix = Array2::from_shape_vec((2, 2), vec![4.0_f32, 1.0, 1.0, 3.0]).unwrap();
        let rhs = Array1::from_vec(vec![1.0_f32, 2.0]);
        let config = IterativeConfig::<f32>::default();

        let cg = conjugate_gradient(&matrix, &rhs, &config).unwrap();
        let gm = gmres(&matrix, &rhs, &config).unwrap();

        let cg_reconstructed = matrix.dot(&cg);
        let gm_reconstructed = matrix.dot(&gm);
        for i in 0..rhs.len() {
            assert!((cg_reconstructed[i] - rhs[i]).abs() < 1e-4);
            assert!((gm_reconstructed[i] - rhs[i]).abs() < 1e-4);
        }
    }

    #[test]
    fn cg_rejects_dimension_mismatch() {
        let matrix = Array2::<f64>::eye(2);
        let rhs = Array1::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let result = conjugate_gradient(&matrix, &rhs, &IterativeConfig::<f64>::default());
        assert!(matches!(result, Err(IterativeError::DimensionMismatch)));
    }

    #[test]
    fn gmres_rejects_empty_input() {
        let matrix = Array2::<f64>::zeros((0, 0));
        let rhs = Array1::<f64>::zeros(0);
        let result = gmres(&matrix, &rhs, &IterativeConfig::<f64>::default());
        assert!(matches!(result, Err(IterativeError::EmptyMatrix)));
    }

    #[test]
    fn cg_returns_zero_for_zero_rhs() {
        let matrix = Array2::<f64>::eye(2);
        let rhs = Array1::from_vec(vec![0.0_f64, 0.0]);
        let solution =
            conjugate_gradient(&matrix, &rhs, &IterativeConfig::<f64>::default()).unwrap();
        assert!(solution.iter().all(|value| value.abs() < 1e-12_f64));
    }

    #[test]
    fn cg_complex_solves_hermitian_spd_system() {
        let matrix = Array2::from_shape_vec((2, 2), vec![
            Complex64::new(4.0, 0.0),
            Complex64::new(1.0, 1.0),
            Complex64::new(1.0, -1.0),
            Complex64::new(3.0, 0.0),
        ])
        .unwrap();
        let rhs = Array1::from_vec(vec![Complex64::new(1.0, 0.5), Complex64::new(2.0, -1.0)]);
        let solution =
            conjugate_gradient_complex(&matrix, &rhs, &IterativeConfig::default()).unwrap();
        let reconstructed = matrix.dot(&solution);
        for i in 0..rhs.len() {
            assert!((reconstructed[i] - rhs[i]).norm() < 1e-7);
        }
    }

    #[test]
    fn gmres_complex_solves_small_system() {
        let matrix = Array2::from_shape_vec((2, 2), vec![
            Complex64::new(3.0, 1.0),
            Complex64::new(1.0, -0.5),
            Complex64::new(0.5, 1.0),
            Complex64::new(2.0, -1.0),
        ])
        .unwrap();
        let rhs = Array1::from_vec(vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, -1.0)]);
        let solution = gmres_complex(&matrix, &rhs, &IterativeConfig::default()).unwrap();
        let reconstructed = matrix.dot(&solution);
        for i in 0..rhs.len() {
            assert!((reconstructed[i] - rhs[i]).norm() < 1e-7);
        }
    }

    #[test]
    fn cg_complex_rejects_dimension_mismatch() {
        let matrix = Array2::from_shape_vec((2, 2), vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ])
        .unwrap();
        let rhs = Array1::from_vec(vec![Complex64::new(1.0, 0.0)]);
        let result = conjugate_gradient_complex(&matrix, &rhs, &IterativeConfig::default());
        assert!(matches!(result, Err(IterativeError::DimensionMismatch)));
    }

    #[test]
    fn gmres_complex_rejects_empty_input() {
        let matrix = Array2::<Complex64>::zeros((0, 0));
        let rhs = Array1::<Complex64>::zeros(0);
        let result = gmres_complex(&matrix, &rhs, &IterativeConfig::default());
        assert!(matches!(result, Err(IterativeError::EmptyMatrix)));
    }
}
