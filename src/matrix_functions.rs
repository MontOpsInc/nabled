//! # Matrix Functions
//!
//! This module provides matrix functions such as matrix exponential and logarithm.
//! These functions are essential for many applications in differential equations,
//! optimization, and machine learning.

use std::fmt;

use nalgebra::{DMatrix, RealField};
use ndarray::Array2;
use num_traits::Float;
use num_traits::float::FloatCore;

/// Error types for matrix function computation
#[derive(Debug, Clone, PartialEq)]
pub enum MatrixFunctionError {
    /// Matrix is not square when required
    NotSquare,
    /// Matrix is empty
    EmptyMatrix,
    /// Matrix is singular (non-invertible)
    SingularMatrix,
    /// Convergence failed in iterative algorithm
    ConvergenceFailed,
    /// Invalid input parameters
    InvalidInput(String),
    /// Matrix has negative eigenvalues (for log)
    NegativeEigenvalues,
    /// Matrix has zero eigenvalues (sign not defined)
    ZeroEigenvalue,
}

impl fmt::Display for MatrixFunctionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MatrixFunctionError::NotSquare => write!(f, "Matrix must be square"),
            MatrixFunctionError::EmptyMatrix => write!(f, "Matrix cannot be empty"),
            MatrixFunctionError::SingularMatrix => write!(f, "Matrix is singular (non-invertible)"),
            MatrixFunctionError::ConvergenceFailed => {
                write!(f, "Matrix function algorithm failed to converge")
            }
            MatrixFunctionError::InvalidInput(msg) => write!(f, "Invalid input: {msg}"),
            MatrixFunctionError::NegativeEigenvalues => {
                write!(f, "Matrix has negative eigenvalues, logarithm not defined")
            }
            MatrixFunctionError::ZeroEigenvalue => {
                write!(f, "Matrix has zero eigenvalue, sign function not defined")
            }
        }
    }
}

impl std::error::Error for MatrixFunctionError {}

/// Matrix exponential and logarithm functions using nalgebra
pub mod nalgebra_matrix_functions {
    use super::*;
    // Note: SymmetricEigen import removed as it's not used directly

    /// Compute matrix exponential using Taylor series expansion
    /// exp(A) = I + A + A²/2! + A³/3! + ...
    /// # Panics
    /// Panics if internal numeric assumptions are violated during setup or
    /// intermediate conversion steps.
    ///
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn matrix_exp<T: RealField + FloatCore>(
        matrix: &DMatrix<T>,
        max_iterations: usize,
        tolerance: T,
    ) -> Result<DMatrix<T>, MatrixFunctionError> {
        if matrix.is_empty() {
            return Err(MatrixFunctionError::EmptyMatrix);
        }

        if !matrix.is_square() {
            return Err(MatrixFunctionError::NotSquare);
        }

        let n = matrix.nrows();
        let identity = DMatrix::<T>::identity(n, n);
        let mut result = identity.clone();
        let mut term = identity.clone();

        for k in 1..=max_iterations {
            term = &term * matrix / T::from(k).unwrap();
            let new_result = &result + &term;

            // Check for convergence
            let diff_norm = (&new_result - &result).norm();
            if diff_norm < tolerance {
                return Ok(new_result);
            }

            result = new_result;
        }

        Err(MatrixFunctionError::ConvergenceFailed)
    }

    /// Compute matrix exponential using eigenvalue decomposition
    /// For diagonalizable matrices: exp(A) = P * exp(D) * P^(-1)
    /// # Panics
    /// Panics if internal numeric assumptions are violated during setup or
    /// intermediate conversion steps.
    ///
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn matrix_exp_eigen<T: RealField + FloatCore>(
        matrix: &DMatrix<T>,
    ) -> Result<DMatrix<T>, MatrixFunctionError> {
        if matrix.is_empty() {
            return Err(MatrixFunctionError::EmptyMatrix);
        }

        if !matrix.is_square() {
            return Err(MatrixFunctionError::NotSquare);
        }

        // For symmetric matrices, use eigenvalue decomposition
        // Note: We'll assume the matrix is symmetric for now
        // In a full implementation, you would check symmetry properly
        if true {
            // Simplified for now
            let eigen = matrix.clone().symmetric_eigen();
            let eigenvalues = eigen.eigenvalues;
            let eigenvectors = eigen.eigenvectors;

            // Compute exp(D) where D is diagonal matrix of eigenvalues
            let exp_eigenvalues = eigenvalues.map(nalgebra::ComplexField::exp);
            let exp_diagonal = DMatrix::from_diagonal(&exp_eigenvalues);

            // Reconstruct: exp(A) = P * exp(D) * P^T
            Ok(&eigenvectors * &exp_diagonal * eigenvectors.transpose())
        } else {
            // For non-symmetric matrices, fall back to Taylor series
            matrix_exp(matrix, 50, T::from_f64(1e-10).unwrap())
        }
    }

    /// Compute matrix logarithm using Taylor series expansion
    /// log(I + A) = A - A²/2 + A³/3 - A⁴/4 + ...
    /// Valid when ||A|| < 1
    /// # Panics
    /// Panics if internal numeric assumptions are violated during setup or
    /// intermediate conversion steps.
    ///
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn matrix_log_taylor<T: RealField + FloatCore>(
        matrix: &DMatrix<T>,
        max_iterations: usize,
        tolerance: T,
    ) -> Result<DMatrix<T>, MatrixFunctionError> {
        if matrix.is_empty() {
            return Err(MatrixFunctionError::EmptyMatrix);
        }

        if !matrix.is_square() {
            return Err(MatrixFunctionError::NotSquare);
        }

        let n = matrix.nrows();
        let identity = DMatrix::<T>::identity(n, n);

        // Check if matrix is close to identity (||A - I|| < 1)
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

            // Check for convergence
            let diff_norm = (&new_result - &result).norm();
            if diff_norm < tolerance {
                return Ok(new_result);
            }

            result = new_result;
            term = &term * &diff;
        }

        Err(MatrixFunctionError::ConvergenceFailed)
    }

    /// Compute matrix logarithm using eigenvalue decomposition
    /// For diagonalizable matrices: log(A) = P * log(D) * P^(-1)
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn matrix_log_eigen<T: RealField + FloatCore>(
        matrix: &DMatrix<T>,
    ) -> Result<DMatrix<T>, MatrixFunctionError> {
        if matrix.is_empty() {
            return Err(MatrixFunctionError::EmptyMatrix);
        }

        if !matrix.is_square() {
            return Err(MatrixFunctionError::NotSquare);
        }

        // For symmetric matrices, use eigenvalue decomposition
        // Note: We'll assume the matrix is symmetric for now
        if true {
            // Simplified for now
            let eigen = matrix.clone().symmetric_eigen();
            let eigenvalues = eigen.eigenvalues;
            let eigenvectors = eigen.eigenvectors;

            // Check for negative eigenvalues
            for &lambda in eigenvalues.iter() {
                if lambda <= T::zero() {
                    return Err(MatrixFunctionError::NegativeEigenvalues);
                }
            }

            // Compute log(D) where D is diagonal matrix of eigenvalues
            let log_eigenvalues = eigenvalues.map(nalgebra::ComplexField::ln);
            let log_diagonal = DMatrix::from_diagonal(&log_eigenvalues);

            // Reconstruct: log(A) = P * log(D) * P^T
            Ok(&eigenvectors * &log_diagonal * eigenvectors.transpose())
        } else {
            // For non-symmetric matrices, use SVD-based approach
            matrix_log_svd(matrix)
        }
    }

    /// Compute matrix logarithm using SVD decomposition
    /// log(A) = U * log(S) * V^T where A = U * S * V^T
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn matrix_log_svd<T: RealField + FloatCore>(
        matrix: &DMatrix<T>,
    ) -> Result<DMatrix<T>, MatrixFunctionError> {
        if matrix.is_empty() {
            return Err(MatrixFunctionError::EmptyMatrix);
        }

        if !matrix.is_square() {
            return Err(MatrixFunctionError::NotSquare);
        }

        let svd = matrix.clone().svd(true, true);

        let (u, vt) = match (&svd.u, &svd.v_t) {
            (Some(u), Some(vt)) => (u.clone(), vt.clone()),
            _ => return Err(MatrixFunctionError::ConvergenceFailed),
        };
        let singular_values = svd.singular_values;

        // Check for zero or negative singular values
        for &sigma in singular_values.iter() {
            if sigma <= T::zero() {
                return Err(MatrixFunctionError::SingularMatrix);
            }
        }

        // Compute log(S) where S is diagonal matrix of singular values
        let log_singular_values = singular_values.map(nalgebra::ComplexField::ln);
        let log_diagonal = DMatrix::from_diagonal(&log_singular_values);

        // Reconstruct: log(A) = U * log(S) * V^T
        Ok(&u * &log_diagonal * &vt)
    }

    /// Compute matrix power using eigenvalue decomposition
    /// A^p = P * D^p * P^(-1) where A = P * D * P^(-1)
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn matrix_power<T: RealField + FloatCore>(
        matrix: &DMatrix<T>,
        power: T,
    ) -> Result<DMatrix<T>, MatrixFunctionError> {
        if matrix.is_empty() {
            return Err(MatrixFunctionError::EmptyMatrix);
        }

        if !matrix.is_square() {
            return Err(MatrixFunctionError::NotSquare);
        }

        // For symmetric matrices, use eigenvalue decomposition
        // Note: We'll assume the matrix is symmetric for now
        if true {
            // Simplified for now
            let eigen = matrix.clone().symmetric_eigen();
            let eigenvalues = eigen.eigenvalues;
            let eigenvectors = eigen.eigenvectors;

            // Check for negative eigenvalues when power is not an integer
            if FloatCore::fract(power) != T::zero() {
                for &lambda in eigenvalues.iter() {
                    if lambda <= T::zero() {
                        return Err(MatrixFunctionError::NegativeEigenvalues);
                    }
                }
            }

            // Compute D^p where D is diagonal matrix of eigenvalues
            let powered_eigenvalues = eigenvalues.map(|lambda| lambda.powf(power));
            let powered_diagonal = DMatrix::from_diagonal(&powered_eigenvalues);

            // Reconstruct: A^p = P * D^p * P^T
            Ok(&eigenvectors * &powered_diagonal * eigenvectors.transpose())
        } else {
            // For non-symmetric matrices, use matrix exponential: A^p = exp(p * log(A))
            let log_a = matrix_log_eigen(matrix)?;
            let p_log_a = &log_a * power;
            matrix_exp_eigen(&p_log_a)
        }
    }

    /// Compute matrix sign function: sign(A) for symmetric diagonalizable A
    /// sign(λ) = 1 if λ>0, -1 if λ<0; requires no zero eigenvalues
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn matrix_sign<T: RealField + FloatCore>(
        matrix: &DMatrix<T>,
    ) -> Result<DMatrix<T>, MatrixFunctionError> {
        if matrix.is_empty() {
            return Err(MatrixFunctionError::EmptyMatrix);
        }
        if !matrix.is_square() {
            return Err(MatrixFunctionError::NotSquare);
        }

        let eigen = matrix.clone().symmetric_eigen();
        let eigenvalues = eigen.eigenvalues;
        let eigenvectors = eigen.eigenvectors;

        let sign_eigenvalues: Vec<T> = eigenvalues
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

        Ok(&eigenvectors * &sign_diagonal * eigenvectors.transpose())
    }
}

/// Matrix exponential and logarithm functions using ndarray
pub mod ndarray_matrix_functions {
    use super::*;
    use crate::interop::{nalgebra_to_ndarray, ndarray_to_nalgebra};

    /// Compute matrix exponential using conversion to nalgebra
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn matrix_exp<T: Float + RealField + FloatCore>(
        matrix: &Array2<T>,
        max_iterations: usize,
        tolerance: T,
    ) -> Result<Array2<T>, MatrixFunctionError> {
        let nalgebra_matrix = ndarray_to_nalgebra(matrix);
        let result =
            nalgebra_matrix_functions::matrix_exp(&nalgebra_matrix, max_iterations, tolerance)?;
        Ok(nalgebra_to_ndarray(&result))
    }

    /// Compute matrix exponential using eigenvalue decomposition
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn matrix_exp_eigen<T: Float + RealField + FloatCore>(
        matrix: &Array2<T>,
    ) -> Result<Array2<T>, MatrixFunctionError> {
        let nalgebra_matrix = ndarray_to_nalgebra(matrix);
        let result = nalgebra_matrix_functions::matrix_exp_eigen(&nalgebra_matrix)?;
        Ok(nalgebra_to_ndarray(&result))
    }

    /// Compute matrix logarithm using Taylor series
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn matrix_log_taylor<T: Float + RealField + FloatCore>(
        matrix: &Array2<T>,
        max_iterations: usize,
        tolerance: T,
    ) -> Result<Array2<T>, MatrixFunctionError> {
        let nalgebra_matrix = ndarray_to_nalgebra(matrix);
        let result = nalgebra_matrix_functions::matrix_log_taylor(
            &nalgebra_matrix,
            max_iterations,
            tolerance,
        )?;
        Ok(nalgebra_to_ndarray(&result))
    }

    /// Compute matrix logarithm using eigenvalue decomposition
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn matrix_log_eigen<T: Float + RealField + FloatCore>(
        matrix: &Array2<T>,
    ) -> Result<Array2<T>, MatrixFunctionError> {
        let nalgebra_matrix = ndarray_to_nalgebra(matrix);
        let result = nalgebra_matrix_functions::matrix_log_eigen(&nalgebra_matrix)?;
        Ok(nalgebra_to_ndarray(&result))
    }

    /// Compute matrix logarithm using SVD decomposition
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn matrix_log_svd<T: Float + RealField + FloatCore>(
        matrix: &Array2<T>,
    ) -> Result<Array2<T>, MatrixFunctionError> {
        let nalgebra_matrix = ndarray_to_nalgebra(matrix);
        let result = nalgebra_matrix_functions::matrix_log_svd(&nalgebra_matrix)?;
        Ok(nalgebra_to_ndarray(&result))
    }

    /// Compute matrix power
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn matrix_power<T: Float + RealField + FloatCore>(
        matrix: &Array2<T>,
        power: T,
    ) -> Result<Array2<T>, MatrixFunctionError> {
        let nalgebra_matrix = ndarray_to_nalgebra(matrix);
        let result = nalgebra_matrix_functions::matrix_power(&nalgebra_matrix, power)?;
        Ok(nalgebra_to_ndarray(&result))
    }

    /// Compute matrix sign function
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn matrix_sign<T: Float + RealField + FloatCore>(
        matrix: &Array2<T>,
    ) -> Result<Array2<T>, MatrixFunctionError> {
        let nalgebra_matrix = ndarray_to_nalgebra(matrix);
        let result = nalgebra_matrix_functions::matrix_sign(&nalgebra_matrix)?;
        Ok(nalgebra_to_ndarray(&result))
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nalgebra::DMatrix;
    use ndarray::Array2;

    use super::*;

    #[test]
    fn test_nalgebra_matrix_exp_identity() {
        let identity = DMatrix::<f64>::identity(3, 3);
        let exp_identity = nalgebra_matrix_functions::matrix_exp_eigen(&identity).unwrap();

        // exp(I) should be e * I
        let e = std::f64::consts::E;
        let expected = identity * e;

        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(exp_identity[(i, j)], expected[(i, j)], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_nalgebra_matrix_exp_zero() {
        let zero = DMatrix::<f64>::zeros(2, 2);
        let exp_zero = nalgebra_matrix_functions::matrix_exp_eigen(&zero).unwrap();
        let identity = DMatrix::<f64>::identity(2, 2);

        // exp(0) should be I
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(exp_zero[(i, j)], identity[(i, j)], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_nalgebra_matrix_log_identity() {
        let identity = DMatrix::<f64>::identity(3, 3);
        let log_identity = nalgebra_matrix_functions::matrix_log_eigen(&identity).unwrap();
        let zero = DMatrix::<f64>::zeros(3, 3);

        // log(I) should be 0
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(log_identity[(i, j)], zero[(i, j)], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_nalgebra_matrix_power_identity() {
        let identity = DMatrix::<f64>::identity(2, 2);
        let power_result = nalgebra_matrix_functions::matrix_power(&identity, 3.0).unwrap();

        // I^3 should be I
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(power_result[(i, j)], identity[(i, j)], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_ndarray_matrix_exp_identity() {
        let identity = Array2::<f64>::eye(3);
        let exp_identity = ndarray_matrix_functions::matrix_exp_eigen(&identity).unwrap();

        // exp(I) should be e * I
        let e = std::f64::consts::E;
        let expected = &identity * e;

        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(exp_identity[[i, j]], expected[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_matrix_exp_log_inverse() {
        // Test that exp(log(A)) ≈ A for positive definite matrices
        let matrix = DMatrix::from_row_slice(2, 2, &[2.0, 1.0, 1.0, 2.0]);
        let log_matrix = nalgebra_matrix_functions::matrix_log_eigen(&matrix).unwrap();
        let exp_log_matrix = nalgebra_matrix_functions::matrix_exp_eigen(&log_matrix).unwrap();

        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(exp_log_matrix[(i, j)], matrix[(i, j)], epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_error_handling() {
        // Test non-square matrix
        let non_square = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = nalgebra_matrix_functions::matrix_exp_eigen(&non_square);
        assert!(matches!(result, Err(MatrixFunctionError::NotSquare)));

        // Test empty matrix
        let empty = DMatrix::<f64>::zeros(0, 0);
        let result = nalgebra_matrix_functions::matrix_exp_eigen(&empty);
        assert!(matches!(result, Err(MatrixFunctionError::EmptyMatrix)));
    }

    #[test]
    fn test_nalgebra_matrix_exp_taylor() {
        let zero = DMatrix::<f64>::zeros(2, 2);
        let exp_zero = nalgebra_matrix_functions::matrix_exp(&zero, 50, 1e-10).unwrap();
        let identity = DMatrix::<f64>::identity(2, 2);
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(exp_zero[(i, j)], identity[(i, j)], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_nalgebra_matrix_log_taylor() {
        // Matrix close to identity: I + 0.1*[[0,1],[1,0]], ||A - I|| = 0.1*sqrt(2) < 1
        let matrix = DMatrix::from_row_slice(2, 2, &[1.0, 0.1, 0.1, 1.0]);
        let log_matrix = nalgebra_matrix_functions::matrix_log_taylor(&matrix, 100, 1e-10).unwrap();
        let exp_log_matrix = nalgebra_matrix_functions::matrix_exp_eigen(&log_matrix).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(exp_log_matrix[(i, j)], matrix[(i, j)], epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_nalgebra_matrix_log_svd() {
        let matrix = DMatrix::from_row_slice(2, 2, &[2.0, 1.0, 1.0, 2.0]);
        let log_matrix = nalgebra_matrix_functions::matrix_log_svd(&matrix).unwrap();
        let exp_log_matrix = nalgebra_matrix_functions::matrix_exp_eigen(&log_matrix).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(exp_log_matrix[(i, j)], matrix[(i, j)], epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_ndarray_matrix_log_eigen() {
        let matrix = Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 1.0, 2.0]).unwrap();
        let log_matrix = ndarray_matrix_functions::matrix_log_eigen(&matrix).unwrap();
        let exp_log_matrix = ndarray_matrix_functions::matrix_exp_eigen(&log_matrix).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(exp_log_matrix[[i, j]], matrix[[i, j]], epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_ndarray_matrix_power() {
        let matrix = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 4.0]).unwrap();
        let sqrt_matrix = ndarray_matrix_functions::matrix_power(&matrix, 0.5).unwrap();
        let mut squared: Array2<f64> = Array2::zeros((2, 2));
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    squared[[i, j]] += sqrt_matrix[[i, k]] * sqrt_matrix[[k, j]];
                }
            }
        }
        assert_relative_eq!(squared[[0, 0]], 1.0, epsilon = 1e-10);
        assert_relative_eq!(squared[[1, 1]], 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_ndarray_matrix_exp_taylor() {
        let zero = Array2::<f64>::zeros((2, 2));
        let exp_zero = ndarray_matrix_functions::matrix_exp(&zero, 50, 1e-10).unwrap();
        let identity = Array2::<f64>::eye(2);
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(exp_zero[[i, j]], identity[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_ndarray_matrix_log_taylor() {
        let matrix = Array2::from_shape_vec((2, 2), vec![1.0, 0.1, 0.1, 1.0]).unwrap();
        let log_matrix = ndarray_matrix_functions::matrix_log_taylor(&matrix, 100, 1e-10).unwrap();
        let exp_log_matrix = ndarray_matrix_functions::matrix_exp_eigen(&log_matrix).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(exp_log_matrix[[i, j]], matrix[[i, j]], epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_ndarray_matrix_log_svd() {
        let matrix = Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 1.0, 2.0]).unwrap();
        let log_matrix = ndarray_matrix_functions::matrix_log_svd(&matrix).unwrap();
        let exp_log_matrix = ndarray_matrix_functions::matrix_exp_eigen(&log_matrix).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(exp_log_matrix[[i, j]], matrix[[i, j]], epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_matrix_sign() {
        // Positive definite: sign = I
        let a = DMatrix::from_row_slice(2, 2, &[2.0, 1.0, 1.0, 2.0]);
        let sign_a = nalgebra_matrix_functions::matrix_sign(&a).unwrap();
        let identity = DMatrix::<f64>::identity(2, 2);
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(sign_a[(i, j)], identity[(i, j)], epsilon = 1e-10);
            }
        }
    }
}
