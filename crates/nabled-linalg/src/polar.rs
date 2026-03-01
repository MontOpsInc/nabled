//! Polar decomposition over ndarray matrices.

use std::fmt;

use ndarray::{Array2, ArrayView2};
use num_complex::Complex64;

use crate::internal::{validate_finite, validate_square_non_empty};
use crate::svd;

/// Result of polar decomposition `A = U P`.
#[derive(Debug, Clone)]
pub struct NdarrayPolarResult {
    /// Orthogonal/unitary factor.
    pub u: Array2<f64>,
    /// Symmetric positive-semidefinite factor.
    pub p: Array2<f64>,
}

/// Result of complex polar decomposition `A = U P`.
#[derive(Debug, Clone)]
pub struct NdarrayComplexPolarResult {
    /// Unitary factor.
    pub u: Array2<Complex64>,
    /// Hermitian positive-semidefinite factor.
    pub p: Array2<Complex64>,
}

/// Error type for polar decomposition.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PolarError {
    /// Matrix is empty.
    EmptyMatrix,
    /// Matrix is not square.
    NotSquare,
    /// Decomposition failed.
    DecompositionFailed,
    /// Numerical instability.
    NumericalInstability,
}

impl fmt::Display for PolarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PolarError::EmptyMatrix => write!(f, "Matrix cannot be empty"),
            PolarError::NotSquare => write!(f, "Matrix must be square"),
            PolarError::DecompositionFailed => write!(f, "Polar decomposition failed"),
            PolarError::NumericalInstability => write!(f, "Numerical instability detected"),
        }
    }
}

impl std::error::Error for PolarError {}

fn validate_complex_square_non_empty(matrix: &Array2<Complex64>) -> Result<(), PolarError> {
    if matrix.is_empty() {
        return Err(PolarError::EmptyMatrix);
    }
    if matrix.nrows() != matrix.ncols() {
        return Err(PolarError::NotSquare);
    }
    Ok(())
}

fn validate_complex_finite(matrix: &Array2<Complex64>) -> Result<(), PolarError> {
    if matrix.iter().any(|value| !value.re.is_finite() || !value.im.is_finite()) {
        return Err(PolarError::NumericalInstability);
    }
    Ok(())
}

/// Compute polar decomposition using SVD.
///
/// # Errors
/// Returns an error if matrix is invalid or SVD fails.
pub fn compute_polar(matrix: &Array2<f64>) -> Result<NdarrayPolarResult, PolarError> {
    validate_square_non_empty(matrix).map_err(|error| match error {
        "empty" => PolarError::EmptyMatrix,
        _ => PolarError::NotSquare,
    })?;
    validate_finite(matrix).map_err(|_| PolarError::NumericalInstability)?;

    let svd = svd::decompose(matrix).map_err(|_| PolarError::DecompositionFailed)?;

    let orthogonal_factor = svd.u.dot(&svd.vt);

    let column_count = matrix.ncols();
    let retained_rank = svd.singular_values.len();
    let mut sigma = Array2::<f64>::zeros((retained_rank, retained_rank));
    for i in 0..retained_rank {
        sigma[[i, i]] = svd.singular_values[i];
    }

    let right_vectors = svd.vt.t().to_owned();
    let psd_factor = right_vectors.dot(&sigma).dot(&svd.vt);
    debug_assert_eq!(psd_factor.nrows(), column_count);

    Ok(NdarrayPolarResult { u: orthogonal_factor, p: psd_factor })
}

/// Compute polar decomposition using SVD from a matrix view.
///
/// # Errors
/// Returns an error if matrix is invalid or SVD fails.
pub fn compute_polar_view(matrix: &ArrayView2<'_, f64>) -> Result<NdarrayPolarResult, PolarError> {
    compute_polar(&matrix.to_owned())
}

/// Compute complex polar decomposition using complex SVD.
///
/// # Errors
/// Returns an error if matrix is invalid or complex SVD fails.
pub fn compute_polar_complex(
    matrix: &Array2<Complex64>,
) -> Result<NdarrayComplexPolarResult, PolarError> {
    validate_complex_square_non_empty(matrix)?;
    validate_complex_finite(matrix)?;
    #[cfg(feature = "openblas-system")]
    {
        let svd = svd::decompose_complex(matrix).map_err(|_| PolarError::DecompositionFailed)?;
        let unitary_factor = svd.u.dot(&svd.vt);

        let retained_rank = svd.singular_values.len();
        let mut sigma = Array2::<Complex64>::zeros((retained_rank, retained_rank));
        for i in 0..retained_rank {
            sigma[[i, i]] = Complex64::new(svd.singular_values[i], 0.0);
        }

        let right_vectors = svd.vt.t().mapv(|value| value.conj());
        let psd_factor = right_vectors.dot(&sigma).dot(&svd.vt);
        debug_assert_eq!(psd_factor.nrows(), matrix.nrows());

        Ok(NdarrayComplexPolarResult { u: unitary_factor, p: psd_factor })
    }
    #[cfg(not(feature = "openblas-system"))]
    {
        let _ = matrix;
        Err(PolarError::DecompositionFailed)
    }
}

/// Compute complex polar decomposition from a matrix view.
///
/// # Errors
/// Returns an error if matrix is invalid or complex SVD fails.
pub fn compute_polar_complex_view(
    matrix: &ArrayView2<'_, Complex64>,
) -> Result<NdarrayComplexPolarResult, PolarError> {
    compute_polar_complex(&matrix.to_owned())
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;
    use num_complex::Complex64;

    use super::*;

    #[test]
    fn polar_reconstructs_input() {
        let matrix = Array2::from_shape_vec((2, 2), vec![3.0, 1.0, 1.0, 3.0]).unwrap();
        let polar = compute_polar(&matrix).unwrap();
        let reconstructed = polar.u.dot(&polar.p);
        for i in 0..2 {
            for j in 0..2 {
                assert!((matrix[[i, j]] - reconstructed[[i, j]]).abs() < 1e-8);
            }
        }
    }

    #[test]
    fn polar_rejects_non_square_input() {
        let matrix = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let result = compute_polar(&matrix);
        assert!(matches!(result, Err(PolarError::NotSquare)));
    }

    #[test]
    fn polar_view_matches_owned() {
        let matrix = Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 1.0, 2.0]).unwrap();
        let owned = compute_polar(&matrix).unwrap();
        let viewed = compute_polar_view(&matrix.view()).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!((owned.u[[i, j]] - viewed.u[[i, j]]).abs() < 1e-12);
                assert!((owned.p[[i, j]] - viewed.p[[i, j]]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn polar_rejects_empty_and_non_finite_input() {
        let empty = Array2::<f64>::zeros((0, 0));
        assert!(matches!(compute_polar(&empty), Err(PolarError::EmptyMatrix)));

        let non_finite = Array2::from_shape_vec((2, 2), vec![1.0, f64::NAN, 0.0, 1.0]).unwrap();
        assert!(matches!(compute_polar(&non_finite), Err(PolarError::NumericalInstability)));
    }

    #[cfg(feature = "openblas-system")]
    #[test]
    fn complex_polar_reconstructs_input_and_view_matches_owned() {
        let matrix = Array2::from_shape_vec((2, 2), vec![
            Complex64::new(2.0, 0.5),
            Complex64::new(0.5, -0.25),
            Complex64::new(-1.0, 1.0),
            Complex64::new(1.5, -0.75),
        ])
        .unwrap();

        let owned = compute_polar_complex(&matrix).unwrap();
        let viewed = compute_polar_complex_view(&matrix.view()).unwrap();
        let reconstructed = owned.u.dot(&owned.p);

        for i in 0..matrix.nrows() {
            for j in 0..matrix.ncols() {
                assert!((reconstructed[[i, j]] - matrix[[i, j]]).norm() < 1e-8);
                assert!((owned.u[[i, j]] - viewed.u[[i, j]]).norm() < 1e-10);
                assert!((owned.p[[i, j]] - viewed.p[[i, j]]).norm() < 1e-10);
            }
        }
    }

    #[cfg(not(feature = "openblas-system"))]
    #[test]
    fn complex_polar_without_provider_errors() {
        let matrix = Array2::from_shape_vec((1, 1), vec![Complex64::new(1.0, 0.0)]).unwrap();
        let result_owned = compute_polar_complex(&matrix);
        let result_view = compute_polar_complex_view(&matrix.view());
        assert!(matches!(result_owned, Err(PolarError::DecompositionFailed)));
        assert!(matches!(result_view, Err(PolarError::DecompositionFailed)));
    }

    #[test]
    fn complex_polar_validation_errors() {
        let empty = Array2::<Complex64>::zeros((0, 0));
        assert!(matches!(compute_polar_complex(&empty), Err(PolarError::EmptyMatrix)));

        let non_square = Array2::from_shape_vec((1, 2), vec![Complex64::new(1.0, 0.0); 2]).unwrap();
        assert!(matches!(compute_polar_complex(&non_square), Err(PolarError::NotSquare)));

        let non_finite =
            Array2::from_shape_vec((1, 1), vec![Complex64::new(f64::NAN, 0.0)]).unwrap();
        assert!(matches!(
            compute_polar_complex(&non_finite),
            Err(PolarError::NumericalInstability)
        ));
    }

    #[test]
    fn polar_error_messages_are_stable() {
        assert_eq!(PolarError::EmptyMatrix.to_string(), "Matrix cannot be empty");
        assert_eq!(PolarError::NotSquare.to_string(), "Matrix must be square");
        assert_eq!(PolarError::DecompositionFailed.to_string(), "Polar decomposition failed");
        assert_eq!(PolarError::NumericalInstability.to_string(), "Numerical instability detected");
    }
}
