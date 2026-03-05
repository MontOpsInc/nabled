//! Orthogonalization routines over ndarray matrices.

use std::fmt;

use nabled_core::scalar::NabledReal;
use ndarray::{Array1, Array2, ArrayView2};
use num_complex::Complex64;

use crate::internal::{DEFAULT_TOLERANCE, qr_gram_schmidt};

/// Error type for orthogonalization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OrthogonalizationError {
    /// Matrix is empty.
    EmptyMatrix,
    /// Numerical instability detected.
    NumericalInstability,
}

impl fmt::Display for OrthogonalizationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OrthogonalizationError::EmptyMatrix => write!(f, "Matrix cannot be empty"),
            OrthogonalizationError::NumericalInstability => {
                write!(f, "Numerical instability detected")
            }
        }
    }
}

impl std::error::Error for OrthogonalizationError {}

/// Modified Gram-Schmidt orthogonalization.
///
/// # Errors
/// Returns an error for empty or non-finite input.
pub fn gram_schmidt<T: NabledReal>(
    matrix: &Array2<T>,
) -> Result<Array2<T>, OrthogonalizationError> {
    gram_schmidt_impl(&matrix.view())
}

fn gram_schmidt_impl<T: NabledReal>(
    matrix: &ArrayView2<'_, T>,
) -> Result<Array2<T>, OrthogonalizationError> {
    if matrix.is_empty() {
        return Err(OrthogonalizationError::EmptyMatrix);
    }
    if matrix.iter().any(|value| !value.is_finite()) {
        return Err(OrthogonalizationError::NumericalInstability);
    }
    let tolerance = T::from_f64(DEFAULT_TOLERANCE).unwrap_or(T::epsilon());
    let (q, _, _) = qr_gram_schmidt(matrix, tolerance);
    Ok(q)
}

fn gram_schmidt_complex_impl(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<Array2<Complex64>, OrthogonalizationError> {
    if matrix.is_empty() {
        return Err(OrthogonalizationError::EmptyMatrix);
    }
    if matrix.iter().any(|value| !value.re.is_finite() || !value.im.is_finite()) {
        return Err(OrthogonalizationError::NumericalInstability);
    }

    let rows = matrix.nrows();
    let cols = matrix.ncols();
    let mut q = Array2::<Complex64>::zeros((rows, cols));
    let mut v = Array1::<Complex64>::zeros(rows);

    for j in 0..cols {
        for row in 0..rows {
            v[row] = matrix[[row, j]];
        }

        for i in 0..j {
            let mut projection = Complex64::new(0.0, 0.0);
            for row in 0..rows {
                projection += q[[row, i]].conj() * v[row];
            }
            for row in 0..rows {
                v[row] -= projection * q[[row, i]];
            }
        }

        let norm = v.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt();
        if norm > DEFAULT_TOLERANCE {
            for row in 0..rows {
                q[[row, j]] = v[row] / norm;
            }
        }
    }

    Ok(q)
}

/// Modified Gram-Schmidt orthogonalization from a matrix view.
///
/// # Errors
/// Returns an error for empty or non-finite input.
pub fn gram_schmidt_view<T: NabledReal>(
    matrix: &ArrayView2<'_, T>,
) -> Result<Array2<T>, OrthogonalizationError> {
    gram_schmidt_impl(matrix)
}

/// Modified Gram-Schmidt orthogonalization for complex matrices.
///
/// # Errors
/// Returns an error for empty or non-finite input.
pub fn gram_schmidt_complex(
    matrix: &Array2<Complex64>,
) -> Result<Array2<Complex64>, OrthogonalizationError> {
    gram_schmidt_complex_impl(&matrix.view())
}

/// Modified Gram-Schmidt orthogonalization for complex matrix views.
///
/// # Errors
/// Returns an error for empty or non-finite input.
pub fn gram_schmidt_complex_view(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<Array2<Complex64>, OrthogonalizationError> {
    gram_schmidt_complex_impl(matrix)
}

/// Classical Gram-Schmidt orthogonalization.
///
/// # Errors
/// Returns an error for empty or non-finite input.
pub fn gram_schmidt_classic<T: NabledReal>(
    matrix: &Array2<T>,
) -> Result<Array2<T>, OrthogonalizationError> {
    gram_schmidt_impl(&matrix.view())
}

/// Classical Gram-Schmidt orthogonalization from a matrix view.
///
/// # Errors
/// Returns an error for empty or non-finite input.
pub fn gram_schmidt_classic_view<T: NabledReal>(
    matrix: &ArrayView2<'_, T>,
) -> Result<Array2<T>, OrthogonalizationError> {
    gram_schmidt_impl(matrix)
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;
    use num_complex::Complex64;

    use super::*;

    #[test]
    fn gram_schmidt_returns_orthonormal_columns() {
        let matrix = Array2::from_shape_vec((3, 2), vec![
            1.0_f64, 1.0_f64, 1.0_f64, 0.0_f64, 0.0_f64, 1.0_f64,
        ])
        .unwrap();
        let q = gram_schmidt(&matrix).unwrap();
        let qtq = q.t().dot(&q);
        assert!((qtq[[0, 0]] - 1.0_f64).abs() < 1e-8_f64);
        assert!((qtq[[1, 1]] - 1.0_f64).abs() < 1e-8_f64);
        assert!(qtq[[0, 1]].abs() < 1e-8_f64);
    }

    #[test]
    fn classical_variant_matches_modified() {
        let matrix = Array2::from_shape_vec((3, 2), vec![
            1.0_f64, 2.0_f64, 2.0_f64, 1.0_f64, 0.5_f64, -1.0_f64,
        ])
        .unwrap();
        let modified = gram_schmidt(&matrix).unwrap();
        let classical = gram_schmidt_classic(&matrix).unwrap();
        for i in 0..modified.nrows() {
            for j in 0..modified.ncols() {
                assert!((modified[[i, j]] - classical[[i, j]]).abs() < 1e-10_f64);
            }
        }
    }

    #[test]
    fn orthogonalization_rejects_empty_input() {
        let empty = Array2::<f64>::zeros((0, 0));
        let result = gram_schmidt(&empty);
        assert!(matches!(result, Err(OrthogonalizationError::EmptyMatrix)));
    }

    #[test]
    fn view_variants_match_owned() {
        let matrix = Array2::from_shape_vec((3, 2), vec![
            1.0_f64, 2.0_f64, 3.0_f64, 1.0_f64, 0.0_f64, 1.0_f64,
        ])
        .unwrap();
        let modified_owned = gram_schmidt(&matrix).unwrap();
        let modified_view = gram_schmidt_view(&matrix.view()).unwrap();
        let classic_owned = gram_schmidt_classic(&matrix).unwrap();
        let classic_view = gram_schmidt_classic_view(&matrix.view()).unwrap();

        for i in 0..matrix.nrows() {
            for j in 0..matrix.ncols() {
                assert!((modified_owned[[i, j]] - modified_view[[i, j]]).abs() < 1e-12_f64);
                assert!((classic_owned[[i, j]] - classic_view[[i, j]]).abs() < 1e-12_f64);
            }
        }
    }

    #[test]
    fn gram_schmidt_complex_returns_orthonormal_columns() {
        let matrix = Array2::from_shape_vec((3, 2), vec![
            Complex64::new(1.0_f64, 0.0_f64),
            Complex64::new(1.0_f64, 1.0_f64),
            Complex64::new(1.0_f64, -1.0_f64),
            Complex64::new(0.0_f64, 1.0_f64),
            Complex64::new(0.0_f64, 0.0_f64),
            Complex64::new(1.0_f64, 0.0_f64),
        ])
        .unwrap();
        let q = gram_schmidt_complex(&matrix).unwrap();
        let q_h_q = q.t().mapv(|value| value.conj()).dot(&q);
        assert!((q_h_q[[0, 0]] - Complex64::new(1.0_f64, 0.0_f64)).norm() < 1e-8_f64);
        assert!((q_h_q[[1, 1]] - Complex64::new(1.0_f64, 0.0_f64)).norm() < 1e-8_f64);
        assert!(q_h_q[[0, 1]].norm() < 1e-8_f64);
        assert!(q_h_q[[1, 0]].norm() < 1e-8_f64);
    }

    #[test]
    fn gram_schmidt_complex_view_matches_owned() {
        let matrix = Array2::from_shape_vec((3, 2), vec![
            Complex64::new(1.0_f64, 0.0_f64),
            Complex64::new(2.0_f64, 1.0_f64),
            Complex64::new(3.0_f64, -1.0_f64),
            Complex64::new(1.0_f64, 0.0_f64),
            Complex64::new(0.0_f64, 1.0_f64),
            Complex64::new(1.0_f64, 0.0_f64),
        ])
        .unwrap();
        let owned = gram_schmidt_complex(&matrix).unwrap();
        let viewed = gram_schmidt_complex_view(&matrix.view()).unwrap();
        for i in 0..matrix.nrows() {
            for j in 0..matrix.ncols() {
                assert!((owned[[i, j]] - viewed[[i, j]]).norm() < 1e-12_f64);
            }
        }
    }
}
