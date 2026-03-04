//! LU decomposition over ndarray matrices.

use std::fmt;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_complex::Complex64;

use crate::internal::{DenseKernelPolicy, lu_decompose};
#[cfg(not(feature = "lapack-provider"))]
use crate::internal::{inverse_from_lu, lu_solve};

/// Result of LU decomposition.
#[derive(Debug, Clone)]
pub struct NdarrayLUResult {
    /// Lower-triangular factor.
    pub l: Array2<f64>,
    /// Upper-triangular factor.
    pub u: Array2<f64>,
}

/// Sign and log-absolute value of determinant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogDetResult<T> {
    /// Determinant sign.
    pub sign:       i8,
    /// Natural logarithm of absolute determinant.
    pub ln_abs_det: T,
}

/// Error type for LU operations.
#[derive(Debug, Clone, PartialEq)]
pub enum LUError {
    /// Matrix is empty.
    EmptyMatrix,
    /// Matrix must be square.
    NotSquare,
    /// Matrix is singular.
    SingularMatrix,
    /// Invalid input.
    InvalidInput(String),
    /// Numerical instability detected.
    NumericalInstability,
}

impl fmt::Display for LUError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LUError::EmptyMatrix => write!(f, "Matrix cannot be empty"),
            LUError::NotSquare => write!(f, "Matrix must be square"),
            LUError::SingularMatrix => write!(f, "Matrix is singular"),
            LUError::InvalidInput(message) => write!(f, "Invalid input: {message}"),
            LUError::NumericalInstability => write!(f, "Numerical instability detected"),
        }
    }
}

impl std::error::Error for LUError {}

fn map_lu_error(error: &'static str) -> LUError {
    match error {
        "empty" => LUError::EmptyMatrix,
        "not_square" => LUError::NotSquare,
        "singular" => LUError::SingularMatrix,
        "non_finite" => LUError::NumericalInstability,
        _ => LUError::InvalidInput(error.to_string()),
    }
}

fn validate_square_finite_view(matrix: &ArrayView2<'_, f64>) -> Result<(), LUError> {
    if matrix.is_empty() {
        return Err(LUError::EmptyMatrix);
    }
    if matrix.nrows() != matrix.ncols() {
        return Err(LUError::NotSquare);
    }
    if matrix.iter().any(|value| !value.is_finite()) {
        return Err(LUError::NumericalInstability);
    }
    Ok(())
}

fn validate_complex_square_finite_view(matrix: &ArrayView2<'_, Complex64>) -> Result<(), LUError> {
    if matrix.is_empty() {
        return Err(LUError::EmptyMatrix);
    }
    if matrix.nrows() != matrix.ncols() {
        return Err(LUError::NotSquare);
    }
    if matrix.iter().any(|value| !value.re.is_finite() || !value.im.is_finite()) {
        return Err(LUError::NumericalInstability);
    }
    Ok(())
}

#[cfg(not(feature = "lapack-provider"))]
type ComplexLUFactors = (Array2<Complex64>, Array2<Complex64>, Vec<usize>, i8);

#[cfg(not(feature = "lapack-provider"))]
#[allow(clippy::many_single_char_names)]
fn decompose_complex_internal(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<ComplexLUFactors, LUError> {
    validate_complex_square_finite_view(matrix)?;

    let n = matrix.nrows();
    let mut l = Array2::<Complex64>::zeros((n, n));
    let mut u = matrix.to_owned();
    let mut pivots = (0..n).collect::<Vec<usize>>();
    let mut sign = 1_i8;

    for i in 0..n {
        l[[i, i]] = Complex64::new(1.0, 0.0);
    }

    for k in 0..n {
        let mut pivot_row = k;
        let mut pivot_norm = u[[k, k]].norm();
        for i in (k + 1)..n {
            let norm = u[[i, k]].norm();
            if norm > pivot_norm {
                pivot_norm = norm;
                pivot_row = i;
            }
        }
        if pivot_norm <= DenseKernelPolicy::BASE_TOLERANCE {
            return Err(LUError::SingularMatrix);
        }

        if pivot_row != k {
            for j in 0..n {
                let temp = u[[k, j]];
                u[[k, j]] = u[[pivot_row, j]];
                u[[pivot_row, j]] = temp;
            }
            for j in 0..k {
                let temp = l[[k, j]];
                l[[k, j]] = l[[pivot_row, j]];
                l[[pivot_row, j]] = temp;
            }
            pivots.swap(k, pivot_row);
            sign *= -1;
        }

        let pivot = u[[k, k]];
        for i in (k + 1)..n {
            let factor = u[[i, k]] / pivot;
            l[[i, k]] = factor;
            u[[i, k]] = Complex64::new(0.0, 0.0);
            for j in (k + 1)..n {
                let top_row_value = u[[k, j]];
                u[[i, j]] -= factor * top_row_value;
            }
        }
    }

    Ok((l, u, pivots, sign))
}

#[cfg(not(feature = "lapack-provider"))]
#[allow(clippy::many_single_char_names)]
fn solve_complex_from_factors(
    l: &Array2<Complex64>,
    u: &Array2<Complex64>,
    pivots: &[usize],
    rhs: &ArrayView1<'_, Complex64>,
) -> Result<Array1<Complex64>, LUError> {
    let n = l.nrows();
    if rhs.len() != n {
        return Err(LUError::InvalidInput("RHS length must match matrix dimensions".to_string()));
    }

    let mut b = Array1::<Complex64>::zeros(n);
    for i in 0..n {
        b[i] = rhs[pivots[i]];
    }

    let mut y = Array1::<Complex64>::zeros(n);
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= l[[i, j]] * y[j];
        }
        y[i] = sum;
    }

    let mut x = Array1::<Complex64>::zeros(n);
    for i_rev in 0..n {
        let i = n - 1 - i_rev;
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= u[[i, j]] * x[j];
        }
        let diagonal = u[[i, i]];
        if diagonal.norm() <= DenseKernelPolicy::BASE_TOLERANCE {
            return Err(LUError::SingularMatrix);
        }
        x[i] = sum / diagonal;
    }
    Ok(x)
}

#[cfg(not(feature = "lapack-provider"))]
fn inverse_complex_internal(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<Array2<Complex64>, LUError> {
    let (l, u, pivots, _) = decompose_complex_internal(matrix)?;
    let n = matrix.nrows();
    let mut inverse = Array2::<Complex64>::zeros((n, n));

    for col in 0..n {
        let mut e = Array1::<Complex64>::zeros(n);
        e[col] = Complex64::new(1.0, 0.0);
        let solution = solve_complex_from_factors(&l, &u, &pivots, &e.view())?;
        for row in 0..n {
            inverse[[row, col]] = solution[row];
        }
    }
    Ok(inverse)
}

fn decompose_internal(
    matrix: &ArrayView2<'_, f64>,
) -> Result<(NdarrayLUResult, Vec<usize>, i8), LUError> {
    let (l, u, pivots, sign) = lu_decompose(matrix).map_err(map_lu_error)?;
    Ok((NdarrayLUResult { l, u }, pivots, sign))
}

#[cfg(feature = "lapack-provider")]
fn solve_provider(
    matrix: &ArrayView2<'_, f64>,
    rhs: &ArrayView1<'_, f64>,
) -> Result<Array1<f64>, LUError> {
    use ndarray_linalg::Solve as _;

    validate_square_finite_view(matrix)?;
    matrix.solve(rhs).map_err(|_| LUError::SingularMatrix)
}

#[cfg(feature = "lapack-provider")]
fn inverse_provider(matrix: &ArrayView2<'_, f64>) -> Result<Array2<f64>, LUError> {
    use ndarray_linalg::Inverse as _;

    validate_square_finite_view(matrix)?;
    matrix.inv().map_err(|_| LUError::SingularMatrix)
}

#[cfg(feature = "lapack-provider")]
fn determinant_provider(matrix: &ArrayView2<'_, f64>) -> Result<f64, LUError> {
    use ndarray_linalg::Determinant as _;

    validate_square_finite_view(matrix)?;
    matrix.det().map_err(|_| LUError::SingularMatrix)
}

#[cfg(feature = "lapack-provider")]
fn solve_complex_provider(
    matrix: &ArrayView2<'_, Complex64>,
    rhs: &ArrayView1<'_, Complex64>,
) -> Result<Array1<Complex64>, LUError> {
    use ndarray_linalg::Solve as _;

    validate_complex_square_finite_view(matrix)?;
    if rhs.len() != matrix.nrows() {
        return Err(LUError::InvalidInput("RHS length must match matrix dimensions".to_string()));
    }
    matrix.solve(rhs).map_err(|_| LUError::SingularMatrix)
}

#[cfg(feature = "lapack-provider")]
fn inverse_complex_provider(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<Array2<Complex64>, LUError> {
    use ndarray_linalg::Inverse as _;

    validate_complex_square_finite_view(matrix)?;
    matrix.inv().map_err(|_| LUError::SingularMatrix)
}

#[cfg(feature = "lapack-provider")]
fn determinant_complex_provider(matrix: &ArrayView2<'_, Complex64>) -> Result<Complex64, LUError> {
    use ndarray_linalg::Determinant as _;

    validate_complex_square_finite_view(matrix)?;
    matrix.det().map_err(|_| LUError::SingularMatrix)
}

fn decompose_with_metadata(
    matrix: &ArrayView2<'_, f64>,
) -> Result<(NdarrayLUResult, Vec<usize>, i8), LUError> {
    decompose_internal(matrix)
}

/// Compute LU decomposition with partial pivoting.
///
/// # Errors
/// Returns an error if input is invalid or decomposition fails.
pub fn decompose(matrix: &Array2<f64>) -> Result<NdarrayLUResult, LUError> {
    let (result, _, _) = decompose_with_metadata(&matrix.view())?;
    Ok(result)
}

/// Compute LU decomposition with partial pivoting from a matrix view.
///
/// # Errors
/// Returns an error if input is invalid or decomposition fails.
pub fn decompose_view(matrix: &ArrayView2<'_, f64>) -> Result<NdarrayLUResult, LUError> {
    let (result, _, _) = decompose_with_metadata(matrix)?;
    Ok(result)
}

/// Solve `Ax=b` using LU decomposition.
///
/// # Errors
/// Returns an error if dimensions are incompatible or matrix is singular.
pub fn solve(matrix: &Array2<f64>, rhs: &Array1<f64>) -> Result<Array1<f64>, LUError> {
    solve_impl(&matrix.view(), &rhs.view())
}

fn solve_impl(
    matrix: &ArrayView2<'_, f64>,
    rhs: &ArrayView1<'_, f64>,
) -> Result<Array1<f64>, LUError> {
    validate_square_finite_view(matrix)?;
    if rhs.len() != matrix.nrows() {
        return Err(LUError::InvalidInput("RHS length must match matrix dimensions".to_string()));
    }

    #[cfg(feature = "lapack-provider")]
    {
        solve_provider(matrix, rhs)
    }
    #[cfg(not(feature = "lapack-provider"))]
    {
        let (decomposition, pivots, _) = decompose_with_metadata(matrix)?;
        lu_solve(&decomposition.l, &decomposition.u, &pivots, rhs).map_err(map_lu_error)
    }
}

/// Solve complex-valued `Ax=b` using LU decomposition.
///
/// # Errors
/// Returns an error if dimensions are incompatible or matrix is singular.
pub fn solve_complex(
    matrix: &Array2<Complex64>,
    rhs: &Array1<Complex64>,
) -> Result<Array1<Complex64>, LUError> {
    solve_complex_impl(&matrix.view(), &rhs.view())
}

fn solve_complex_impl(
    matrix: &ArrayView2<'_, Complex64>,
    rhs: &ArrayView1<'_, Complex64>,
) -> Result<Array1<Complex64>, LUError> {
    #[cfg(feature = "lapack-provider")]
    {
        solve_complex_provider(matrix, rhs)
    }
    #[cfg(not(feature = "lapack-provider"))]
    {
        let (l, u, pivots, _) = decompose_complex_internal(matrix)?;
        solve_complex_from_factors(&l, &u, &pivots, rhs)
    }
}

/// Solve `Ax=b` using LU decomposition from matrix/vector views.
///
/// # Errors
/// Returns an error if dimensions are incompatible or matrix is singular.
pub fn solve_view(
    matrix: &ArrayView2<'_, f64>,
    rhs: &ArrayView1<'_, f64>,
) -> Result<Array1<f64>, LUError> {
    solve_impl(matrix, rhs)
}

/// Solve complex-valued `Ax=b` from matrix/vector views.
///
/// # Errors
/// Returns an error if dimensions are incompatible or matrix is singular.
pub fn solve_complex_view(
    matrix: &ArrayView2<'_, Complex64>,
    rhs: &ArrayView1<'_, Complex64>,
) -> Result<Array1<Complex64>, LUError> {
    solve_complex_impl(matrix, rhs)
}

/// Compute matrix inverse via LU decomposition.
///
/// # Errors
/// Returns an error if matrix is singular.
pub fn inverse(matrix: &Array2<f64>) -> Result<Array2<f64>, LUError> {
    inverse_impl(&matrix.view())
}

fn inverse_impl(matrix: &ArrayView2<'_, f64>) -> Result<Array2<f64>, LUError> {
    #[cfg(feature = "lapack-provider")]
    {
        inverse_provider(matrix)
    }
    #[cfg(not(feature = "lapack-provider"))]
    {
        let (decomposition, pivots, _) = decompose_with_metadata(matrix)?;
        inverse_from_lu(&decomposition.l, &decomposition.u, &pivots).map_err(map_lu_error)
    }
}

/// Compute complex matrix inverse via LU decomposition.
///
/// # Errors
/// Returns an error if matrix is singular.
pub fn inverse_complex(matrix: &Array2<Complex64>) -> Result<Array2<Complex64>, LUError> {
    inverse_complex_impl(&matrix.view())
}

fn inverse_complex_impl(matrix: &ArrayView2<'_, Complex64>) -> Result<Array2<Complex64>, LUError> {
    #[cfg(feature = "lapack-provider")]
    {
        inverse_complex_provider(matrix)
    }
    #[cfg(not(feature = "lapack-provider"))]
    {
        inverse_complex_internal(matrix)
    }
}

/// Compute matrix inverse via LU decomposition from a matrix view.
///
/// # Errors
/// Returns an error if matrix is singular.
pub fn inverse_view(matrix: &ArrayView2<'_, f64>) -> Result<Array2<f64>, LUError> {
    inverse_impl(matrix)
}

/// Compute complex matrix inverse from a matrix view.
///
/// # Errors
/// Returns an error if matrix is singular.
pub fn inverse_complex_view(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<Array2<Complex64>, LUError> {
    inverse_complex_impl(matrix)
}

/// Compute determinant via LU decomposition.
///
/// # Errors
/// Returns an error if decomposition fails.
pub fn determinant(matrix: &Array2<f64>) -> Result<f64, LUError> {
    determinant_impl(&matrix.view())
}

fn determinant_impl(matrix: &ArrayView2<'_, f64>) -> Result<f64, LUError> {
    #[cfg(feature = "lapack-provider")]
    {
        determinant_provider(matrix)
    }
    #[cfg(not(feature = "lapack-provider"))]
    {
        let (decomposition, _, sign) = decompose_with_metadata(matrix)?;
        let mut determinant = f64::from(sign);
        for i in 0..decomposition.u.nrows() {
            determinant *= decomposition.u[[i, i]];
        }
        if !determinant.is_finite() {
            return Err(LUError::NumericalInstability);
        }
        Ok(determinant)
    }
}

/// Compute complex determinant via LU decomposition.
///
/// # Errors
/// Returns an error if decomposition fails.
pub fn determinant_complex(matrix: &Array2<Complex64>) -> Result<Complex64, LUError> {
    determinant_complex_impl(&matrix.view())
}

fn determinant_complex_impl(matrix: &ArrayView2<'_, Complex64>) -> Result<Complex64, LUError> {
    #[cfg(feature = "lapack-provider")]
    {
        determinant_complex_provider(matrix)
    }
    #[cfg(not(feature = "lapack-provider"))]
    {
        let (_, u, _, sign) = decompose_complex_internal(matrix)?;
        let mut determinant = Complex64::new(f64::from(sign), 0.0);
        for i in 0..u.nrows() {
            determinant *= u[[i, i]];
        }
        if !determinant.re.is_finite() || !determinant.im.is_finite() {
            return Err(LUError::NumericalInstability);
        }
        Ok(determinant)
    }
}

/// Compute determinant via LU decomposition from a matrix view.
///
/// # Errors
/// Returns an error if decomposition fails.
pub fn determinant_view(matrix: &ArrayView2<'_, f64>) -> Result<f64, LUError> {
    determinant_impl(matrix)
}

/// Compute complex determinant from a matrix view.
///
/// # Errors
/// Returns an error if decomposition fails.
pub fn determinant_complex_view(matrix: &ArrayView2<'_, Complex64>) -> Result<Complex64, LUError> {
    determinant_complex_impl(matrix)
}

/// Compute signed log-determinant via LU decomposition.
///
/// # Errors
/// Returns an error if matrix is singular.
pub fn log_determinant(matrix: &Array2<f64>) -> Result<LogDetResult<f64>, LUError> {
    let determinant = determinant_impl(&matrix.view())?;
    if determinant.abs() <= DenseKernelPolicy::BASE_TOLERANCE {
        return Err(LUError::SingularMatrix);
    }
    let sign = if determinant.is_sign_positive() { 1 } else { -1 };
    Ok(LogDetResult { sign, ln_abs_det: determinant.abs().ln() })
}

/// Compute signed log-determinant via LU decomposition from a matrix view.
///
/// # Errors
/// Returns an error if matrix is singular.
pub fn log_determinant_view(matrix: &ArrayView2<'_, f64>) -> Result<LogDetResult<f64>, LUError> {
    let determinant = determinant_impl(matrix)?;
    if determinant.abs() <= DenseKernelPolicy::BASE_TOLERANCE {
        return Err(LUError::SingularMatrix);
    }
    let sign = if determinant.is_sign_positive() { 1 } else { -1 };
    Ok(LogDetResult { sign, ln_abs_det: determinant.abs().ln() })
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2};
    use num_complex::Complex64;

    use super::*;

    #[test]
    fn solve_reconstructs_rhs() {
        let matrix = Array2::from_shape_vec((2, 2), vec![4.0, 3.0, 6.0, 3.0]).unwrap();
        let rhs = Array1::from_vec(vec![10.0, 12.0]);
        let solution = solve(&matrix, &rhs).unwrap();
        let reconstructed = matrix.dot(&solution);
        assert!((reconstructed[0] - rhs[0]).abs() < 1e-8);
        assert!((reconstructed[1] - rhs[1]).abs() < 1e-8);
    }

    #[test]
    fn determinant_matches_expected() {
        let matrix = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let determinant = determinant(&matrix).unwrap();
        assert!((determinant + 2.0).abs() < 1e-12);
    }

    #[test]
    fn singular_matrix_is_rejected() {
        let singular = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 2.0, 4.0]).unwrap();
        let rhs = Array1::from_vec(vec![1.0, 2.0]);
        assert!(matches!(solve(&singular, &rhs), Err(LUError::SingularMatrix)));
    }

    #[test]
    fn inverse_multiplied_by_matrix_is_identity() {
        let matrix = Array2::from_shape_vec((2, 2), vec![4.0, 7.0, 2.0, 6.0]).unwrap();
        let inverse = inverse(&matrix).unwrap();
        let product = matrix.dot(&inverse);
        assert!((product[[0, 0]] - 1.0).abs() < 1e-8);
        assert!((product[[1, 1]] - 1.0).abs() < 1e-8);
        assert!(product[[0, 1]].abs() < 1e-8);
        assert!(product[[1, 0]].abs() < 1e-8);
    }

    #[test]
    fn log_determinant_has_expected_sign_and_value() {
        let matrix = Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 0.0, -3.0]).unwrap();
        let result = log_determinant(&matrix).unwrap();
        assert_eq!(result.sign, -1);
        assert!((result.ln_abs_det - (6.0_f64).ln()).abs() < 1e-10);
    }

    #[test]
    fn solve_rejects_bad_rhs_length() {
        let matrix = Array2::eye(2);
        let rhs = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let result = solve(&matrix, &rhs);
        assert!(matches!(result, Err(LUError::InvalidInput(_))));
    }

    #[test]
    fn decompose_exposes_factors() {
        let matrix = Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 4.0, 3.0]).unwrap();
        let lu = decompose(&matrix).unwrap();
        assert_eq!(lu.l.dim(), (2, 2));
        assert_eq!(lu.u.dim(), (2, 2));
    }

    #[test]
    fn view_variants_match_owned() {
        let matrix = Array2::from_shape_vec((2, 2), vec![4.0, 7.0, 2.0, 6.0]).unwrap();
        let rhs = Array1::from_vec(vec![5.0, 7.0]);

        let owned = decompose(&matrix).unwrap();
        let viewed = decompose_view(&matrix.view()).unwrap();
        assert_eq!(owned.l.dim(), viewed.l.dim());
        assert_eq!(owned.u.dim(), viewed.u.dim());

        let solution_owned = solve(&matrix, &rhs).unwrap();
        let solution_view = solve_view(&matrix.view(), &rhs.view()).unwrap();
        for i in 0..rhs.len() {
            assert!((solution_owned[i] - solution_view[i]).abs() < 1e-12);
        }

        let inverse_owned = inverse(&matrix).unwrap();
        let inverse_view = inverse_view(&matrix.view()).unwrap();
        for i in 0..matrix.nrows() {
            for j in 0..matrix.ncols() {
                assert!((inverse_owned[[i, j]] - inverse_view[[i, j]]).abs() < 1e-12);
            }
        }

        let det_owned = determinant(&matrix).unwrap();
        let det_view = determinant_view(&matrix.view()).unwrap();
        assert!((det_owned - det_view).abs() < 1e-12);

        let logdet_owned = log_determinant(&matrix).unwrap();
        let logdet_view = log_determinant_view(&matrix.view()).unwrap();
        assert_eq!(logdet_owned.sign, logdet_view.sign);
        assert!((logdet_owned.ln_abs_det - logdet_view.ln_abs_det).abs() < 1e-12);
    }

    #[test]
    fn complex_lu_paths_solve_inverse_and_determinant() {
        let matrix = Array2::from_shape_vec((2, 2), vec![
            Complex64::new(2.0, 1.0),
            Complex64::new(1.0, -1.0),
            Complex64::new(0.5, 0.25),
            Complex64::new(3.0, -0.5),
        ])
        .unwrap();
        let rhs = Array1::from_vec(vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)]);

        let solution = solve_complex(&matrix, &rhs).unwrap();
        let reconstructed = matrix.dot(&solution);
        for i in 0..rhs.len() {
            assert!((reconstructed[i] - rhs[i]).norm() < 1e-8);
        }

        let inverse = inverse_complex(&matrix).unwrap();
        let product = matrix.dot(&inverse);
        assert!((product[[0, 0]] - Complex64::new(1.0, 0.0)).norm() < 1e-8);
        assert!((product[[1, 1]] - Complex64::new(1.0, 0.0)).norm() < 1e-8);
        assert!(product[[0, 1]].norm() < 1e-8);
        assert!(product[[1, 0]].norm() < 1e-8);

        let determinant = determinant_complex(&matrix).unwrap();
        assert!(determinant.norm() > 1e-8);

        let solution_view = solve_complex_view(&matrix.view(), &rhs.view()).unwrap();
        let inverse_viewed = inverse_complex_view(&matrix.view()).unwrap();
        let determinant_viewed = determinant_complex_view(&matrix.view()).unwrap();
        for i in 0..rhs.len() {
            assert!((solution_view[i] - solution[i]).norm() < 1e-10);
        }
        for i in 0..2 {
            for j in 0..2 {
                assert!((inverse_viewed[[i, j]] - inverse[[i, j]]).norm() < 1e-10);
            }
        }
        assert!((determinant_viewed - determinant).norm() < 1e-10);
    }
}
