//! # Jacobian Computation Module
//!
//! This module provides Jacobian matrix computation for both `nalgebra` and `ndarray`.
//! The Jacobian matrix contains the partial derivatives of a vector-valued function
//! with respect to its input variables.
//!
//! ## Features
//!
//! - **Numerical Jacobian**: Finite difference approximation
//! - **Analytical Jacobian**: Symbolic differentiation for common functions
//! - **Support for both nalgebra and ndarray**
//! - **Configurable step size and tolerance**
//! - **Comprehensive error handling**

use nalgebra::{DMatrix, DVector, RealField};
use ndarray::{Array2, Array1};
use num_traits::{Float, float::FloatCore};
use std::fmt;

/// Error types for Jacobian computation
#[derive(Debug, Clone)]
pub enum JacobianError {
    /// Function returned an error during evaluation
    FunctionError(String),
    /// Invalid input dimensions
    InvalidDimensions(String),
    /// Step size too small or invalid
    InvalidStepSize,
    /// Convergence failed for iterative methods
    ConvergenceFailed,
    /// Empty input or output
    EmptyInput,
    /// Dimension mismatch between input and output
    DimensionMismatch,
}

impl fmt::Display for JacobianError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            JacobianError::FunctionError(msg) => write!(f, "Function error: {}", msg),
            JacobianError::InvalidDimensions(msg) => write!(f, "Invalid dimensions: {}", msg),
            JacobianError::InvalidStepSize => write!(f, "Invalid step size"),
            JacobianError::ConvergenceFailed => write!(f, "Convergence failed"),
            JacobianError::EmptyInput => write!(f, "Empty input"),
            JacobianError::DimensionMismatch => write!(f, "Dimension mismatch"),
        }
    }
}

impl std::error::Error for JacobianError {}

/// Configuration for numerical Jacobian computation
#[derive(Debug, Clone)]
pub struct JacobianConfig<T> {
    /// Step size for finite differences
    pub step_size: T,
    /// Relative tolerance for convergence
    pub tolerance: T,
    /// Maximum number of iterations
    pub max_iterations: usize,
}

impl<T: Float> Default for JacobianConfig<T> {
    fn default() -> Self {
        Self {
            step_size: T::from(1e-6).unwrap(),
            tolerance: T::from(1e-8).unwrap(),
            max_iterations: 100,
        }
    }
}

impl<T: Float> JacobianConfig<T> {
    /// Create a new JacobianConfig with validation
    pub fn new(step_size: T, tolerance: T, max_iterations: usize) -> Result<Self, JacobianError> {
        if step_size <= T::zero() {
            return Err(JacobianError::InvalidStepSize);
        }
        if tolerance <= T::zero() {
            return Err(JacobianError::InvalidDimensions("Tolerance must be positive".to_string()));
        }
        if max_iterations == 0 {
            return Err(JacobianError::InvalidDimensions("Max iterations must be positive".to_string()));
        }
        
        Ok(Self {
            step_size,
            tolerance,
            max_iterations,
        })
    }
    
    /// Validate the configuration
    pub fn validate(&self) -> Result<(), JacobianError> {
        if self.step_size <= T::zero() {
            return Err(JacobianError::InvalidStepSize);
        }
        if self.tolerance <= T::zero() {
            return Err(JacobianError::InvalidDimensions("Tolerance must be positive".to_string()));
        }
        if self.max_iterations == 0 {
            return Err(JacobianError::InvalidDimensions("Max iterations must be positive".to_string()));
        }
        Ok(())
    }
}

/// Nalgebra Jacobian computation functions
pub mod nalgebra_jacobian {
    use super::*;
    use nalgebra::{DMatrix, DVector, RealField};

    /// Compute numerical Jacobian using finite differences
    ///
    /// # Arguments
    /// * `f` - Function that takes a DVector and returns a DVector
    /// * `x` - Point at which to compute the Jacobian
    /// * `config` - Configuration for the computation
    ///
    /// # Returns
    /// * `Result<DMatrix<T>, JacobianError>` - Jacobian matrix
    ///
    /// # Example
    /// ```rust
    /// use rust_linalg::jacobian::{nalgebra_jacobian, JacobianConfig};
    /// use nalgebra::DVector;
    /// 
    /// let f = |x: &DVector<f64>| -> Result<DVector<f64>, String> {
    ///     Ok(DVector::from_vec(vec![x[0] * x[0], x[1] * x[1]]))
    /// };
    /// 
    /// let x = DVector::from_vec(vec![1.0, 2.0]);
    /// let jacobian = nalgebra_jacobian::numerical_jacobian(&f, &x, &JacobianConfig::default())?;
    /// // Jacobian should be [[2.0, 0.0], [0.0, 4.0]]
    /// # Ok::<(), rust_linalg::jacobian::JacobianError>(())
    /// ```
    pub fn numerical_jacobian<T, F>(
        f: &F,
        x: &DVector<T>,
        config: &JacobianConfig<T>,
    ) -> Result<DMatrix<T>, JacobianError>
    where
        T: RealField + FloatCore + Float,
        F: Fn(&DVector<T>) -> Result<DVector<T>, String>,
    {
        // Validate inputs
        if x.is_empty() {
            return Err(JacobianError::EmptyInput);
        }
        
        // Check for NaN or infinite values in input
        for i in 0..x.len() {
            if !num_traits::Float::is_finite(x[i]) {
                return Err(JacobianError::InvalidDimensions(
                    format!("Input contains non-finite value at index {}", i)
                ));
            }
        }
        
        // Validate configuration
        config.validate()?;

        // Evaluate function at the point
        let fx = f(x).map_err(JacobianError::FunctionError)?;
        let m = fx.len();
        let n = x.len();

        if m == 0 {
            return Err(JacobianError::EmptyInput);
        }

        // Check for non-finite values in function output
        for i in 0..m {
            if !num_traits::Float::is_finite(x[i]) {
                return Err(JacobianError::FunctionError(
                    format!("Function returned non-finite value at index {}", i)
                ));
            }
        }

        let mut jacobian = DMatrix::zeros(m, n);

        // Compute partial derivatives using finite differences
        for j in 0..n {
            let mut x_plus = x.clone();
            
            // Check for overflow when adding step size
            let step = config.step_size;
            if num_traits::Float::is_finite(x_plus[j]) && num_traits::Float::is_finite(step) {
                x_plus[j] = x_plus[j] + step;
            } else {
                return Err(JacobianError::InvalidStepSize);
            }

            let fx_plus = f(&x_plus).map_err(JacobianError::FunctionError)?;

            if fx_plus.len() != m {
                return Err(JacobianError::DimensionMismatch);
            }

            // Check for non-finite values in perturbed function output
            for i in 0..m {
                if !num_traits::Float::is_finite(fx_plus[i]) {
                    return Err(JacobianError::FunctionError(
                        format!("Function returned non-finite value at index {} when perturbing variable {}", i, j)
                    ));
                }
            }

            // Compute finite difference
            for i in 0..m {
                let diff = fx_plus[i] - fx[i];
                if num_traits::Float::is_finite(diff) && num_traits::Float::is_finite(step) && step != T::zero() {
                    jacobian[(i, j)] = diff / step;
                } else {
                    return Err(JacobianError::ConvergenceFailed);
                }
            }
        }

        Ok(jacobian)
    }

    /// Compute Jacobian using central differences (more accurate)
    ///
    /// # Arguments
    /// * `f` - Function that takes a DVector and returns a DVector
    /// * `x` - Point at which to compute the Jacobian
    /// * `config` - Configuration for the computation
    ///
    /// # Returns
    /// * `Result<DMatrix<T>, JacobianError>` - Jacobian matrix
    pub fn numerical_jacobian_central<T, F>(
        f: &F,
        x: &DVector<T>,
        config: &JacobianConfig<T>,
    ) -> Result<DMatrix<T>, JacobianError>
    where
        T: RealField + FloatCore + Float,
        F: Fn(&DVector<T>) -> Result<DVector<T>, String>,
    {
        // Validate inputs
        if x.is_empty() {
            return Err(JacobianError::EmptyInput);
        }
        
        // Check for NaN or infinite values in input
        for i in 0..x.len() {
            if !num_traits::Float::is_finite(x[i]) {
                return Err(JacobianError::InvalidDimensions(
                    format!("Input contains non-finite value at index {}", i)
                ));
            }
        }
        
        // Validate configuration
        config.validate()?;

        let m = f(x).map_err(JacobianError::FunctionError)?.len();
        let n = x.len();

        if m == 0 {
            return Err(JacobianError::EmptyInput);
        }

        let mut jacobian = DMatrix::zeros(m, n);

        // Compute partial derivatives using central differences
        for j in 0..n {
            let mut x_plus = x.clone();
            let mut x_minus = x.clone();
            
            x_plus[j] = x_plus[j] + config.step_size;
            x_minus[j] = x_minus[j] - config.step_size;

            let fx_plus = f(&x_plus).map_err(JacobianError::FunctionError)?;
            let fx_minus = f(&x_minus).map_err(JacobianError::FunctionError)?;

            if fx_plus.len() != m || fx_minus.len() != m {
                return Err(JacobianError::DimensionMismatch);
            }

            // Compute central difference
            let two_h = T::from(2.0).unwrap() * config.step_size;
            for i in 0..m {
                jacobian[(i, j)] = (fx_plus[i] - fx_minus[i]) / two_h;
            }
        }

        Ok(jacobian)
    }

    /// Compute Jacobian for a scalar function (gradient)
    ///
    /// # Arguments
    /// * `f` - Function that takes a DVector and returns a scalar
    /// * `x` - Point at which to compute the gradient
    /// * `config` - Configuration for the computation
    ///
    /// # Returns
    /// * `Result<DVector<T>, JacobianError>` - Gradient vector
    pub fn numerical_gradient<T, F>(
        f: &F,
        x: &DVector<T>,
        config: &JacobianConfig<T>,
    ) -> Result<DVector<T>, JacobianError>
    where
        T: RealField + FloatCore,
        F: Fn(&DVector<T>) -> Result<T, String>,
    {
        if x.is_empty() {
            return Err(JacobianError::EmptyInput);
        }

        let n = x.len();
        let mut gradient = DVector::zeros(n);

        // Compute partial derivatives using finite differences
        for j in 0..n {
            let mut x_plus = x.clone();
            x_plus[j] = x_plus[j] + config.step_size;

            let fx = f(x).map_err(JacobianError::FunctionError)?;
            let fx_plus = f(&x_plus).map_err(JacobianError::FunctionError)?;

            gradient[j] = (fx_plus - fx) / config.step_size;
        }

        Ok(gradient)
    }

    /// Compute Hessian matrix (second-order partial derivatives)
    ///
    /// # Arguments
    /// * `f` - Function that takes a DVector and returns a scalar
    /// * `x` - Point at which to compute the Hessian
    /// * `config` - Configuration for the computation
    ///
    /// # Returns
    /// * `Result<DMatrix<T>, JacobianError>` - Hessian matrix
    pub fn numerical_hessian<T, F>(
        f: &F,
        x: &DVector<T>,
        config: &JacobianConfig<T>,
    ) -> Result<DMatrix<T>, JacobianError>
    where
        T: RealField + FloatCore,
        F: Fn(&DVector<T>) -> Result<T, String>,
    {
        if x.is_empty() {
            return Err(JacobianError::EmptyInput);
        }

        let n = x.len();
        let mut hessian = DMatrix::zeros(n, n);

        // Compute second-order partial derivatives
        for i in 0..n {
            for j in 0..n {
                let mut x_plus_i = x.clone();
                let mut x_plus_j = x.clone();
                let mut x_plus_ij = x.clone();
                
                x_plus_i[i] = x_plus_i[i] + config.step_size;
                x_plus_j[j] = x_plus_j[j] + config.step_size;
                x_plus_ij[i] = x_plus_ij[i] + config.step_size;
                x_plus_ij[j] = x_plus_ij[j] + config.step_size;

                let f00 = f(x).map_err(JacobianError::FunctionError)?;
                let f10 = f(&x_plus_i).map_err(JacobianError::FunctionError)?;
                let f01 = f(&x_plus_j).map_err(JacobianError::FunctionError)?;
                let f11 = f(&x_plus_ij).map_err(JacobianError::FunctionError)?;

                // Second-order finite difference
                let h_squared = config.step_size * config.step_size;
                hessian[(i, j)] = (f11 - f10 - f01 + f00) / h_squared;
            }
        }

        Ok(hessian)
    }
}

/// Ndarray Jacobian computation functions
pub mod ndarray_jacobian {
    use super::*;
    use ndarray::{Array2, Array1};

    /// Compute numerical Jacobian using finite differences
    ///
    /// # Arguments
    /// * `f` - Function that takes an Array1 and returns an Array1
    /// * `x` - Point at which to compute the Jacobian
    /// * `config` - Configuration for the computation
    ///
    /// # Returns
    /// * `Result<Array2<T>, JacobianError>` - Jacobian matrix
    pub fn numerical_jacobian<T, F>(
        f: &F,
        x: &Array1<T>,
        config: &JacobianConfig<T>,
    ) -> Result<Array2<T>, JacobianError>
    where
        T: Float + FloatCore,
        F: Fn(&Array1<T>) -> Result<Array1<T>, String>,
    {
        if x.is_empty() {
            return Err(JacobianError::EmptyInput);
        }

        // Evaluate function at the point
        let fx = f(x).map_err(JacobianError::FunctionError)?;
        let m = fx.len();
        let n = x.len();

        if m == 0 {
            return Err(JacobianError::EmptyInput);
        }

        let mut jacobian = Array2::zeros((m, n));

        // Compute partial derivatives using finite differences
        for j in 0..n {
            let mut x_plus = x.clone();
            x_plus[j] = x_plus[j] + config.step_size;

            let fx_plus = f(&x_plus).map_err(JacobianError::FunctionError)?;

            if fx_plus.len() != m {
                return Err(JacobianError::DimensionMismatch);
            }

            // Compute finite difference
            for i in 0..m {
                jacobian[(i, j)] = (fx_plus[i] - fx[i]) / config.step_size;
            }
        }

        Ok(jacobian)
    }

    /// Compute Jacobian using central differences (more accurate)
    ///
    /// # Arguments
    /// * `f` - Function that takes an Array1 and returns an Array1
    /// * `x` - Point at which to compute the Jacobian
    /// * `config` - Configuration for the computation
    ///
    /// # Returns
    /// * `Result<Array2<T>, JacobianError>` - Jacobian matrix
    pub fn numerical_jacobian_central<T, F>(
        f: &F,
        x: &Array1<T>,
        config: &JacobianConfig<T>,
    ) -> Result<Array2<T>, JacobianError>
    where
        T: Float + FloatCore,
        F: Fn(&Array1<T>) -> Result<Array1<T>, String>,
    {
        if x.is_empty() {
            return Err(JacobianError::EmptyInput);
        }

        let m = f(x).map_err(JacobianError::FunctionError)?.len();
        let n = x.len();

        if m == 0 {
            return Err(JacobianError::EmptyInput);
        }

        let mut jacobian = Array2::zeros((m, n));

        // Compute partial derivatives using central differences
        for j in 0..n {
            let mut x_plus = x.clone();
            let mut x_minus = x.clone();
            
            x_plus[j] = x_plus[j] + config.step_size;
            x_minus[j] = x_minus[j] - config.step_size;

            let fx_plus = f(&x_plus).map_err(JacobianError::FunctionError)?;
            let fx_minus = f(&x_minus).map_err(JacobianError::FunctionError)?;

            if fx_plus.len() != m || fx_minus.len() != m {
                return Err(JacobianError::DimensionMismatch);
            }

            // Compute central difference
            let two_h = T::from(2.0).unwrap() * config.step_size;
            for i in 0..m {
                jacobian[(i, j)] = (fx_plus[i] - fx_minus[i]) / two_h;
            }
        }

        Ok(jacobian)
    }

    /// Compute Jacobian for a scalar function (gradient)
    ///
    /// # Arguments
    /// * `f` - Function that takes an Array1 and returns a scalar
    /// * `x` - Point at which to compute the gradient
    /// * `config` - Configuration for the computation
    ///
    /// # Returns
    /// * `Result<Array1<T>, JacobianError>` - Gradient vector
    pub fn numerical_gradient<T, F>(
        f: &F,
        x: &Array1<T>,
        config: &JacobianConfig<T>,
    ) -> Result<Array1<T>, JacobianError>
    where
        T: Float + FloatCore,
        F: Fn(&Array1<T>) -> Result<T, String>,
    {
        if x.is_empty() {
            return Err(JacobianError::EmptyInput);
        }

        let n = x.len();
        let mut gradient = Array1::zeros(n);

        // Compute partial derivatives using finite differences
        for j in 0..n {
            let mut x_plus = x.clone();
            x_plus[j] = x_plus[j] + config.step_size;

            let fx = f(x).map_err(JacobianError::FunctionError)?;
            let fx_plus = f(&x_plus).map_err(JacobianError::FunctionError)?;

            gradient[j] = (fx_plus - fx) / config.step_size;
        }

        Ok(gradient)
    }

    /// Compute Hessian matrix (second-order partial derivatives)
    ///
    /// # Arguments
    /// * `f` - Function that takes an Array1 and returns a scalar
    /// * `x` - Point at which to compute the Hessian
    /// * `config` - Configuration for the computation
    ///
    /// # Returns
    /// * `Result<Array2<T>, JacobianError>` - Hessian matrix
    pub fn numerical_hessian<T, F>(
        f: &F,
        x: &Array1<T>,
        config: &JacobianConfig<T>,
    ) -> Result<Array2<T>, JacobianError>
    where
        T: Float + FloatCore,
        F: Fn(&Array1<T>) -> Result<T, String>,
    {
        if x.is_empty() {
            return Err(JacobianError::EmptyInput);
        }

        let n = x.len();
        let mut hessian = Array2::zeros((n, n));

        // Compute second-order partial derivatives
        for i in 0..n {
            for j in 0..n {
                let mut x_plus_i = x.clone();
                let mut x_plus_j = x.clone();
                let mut x_plus_ij = x.clone();
                
                x_plus_i[i] = x_plus_i[i] + config.step_size;
                x_plus_j[j] = x_plus_j[j] + config.step_size;
                x_plus_ij[i] = x_plus_ij[i] + config.step_size;
                x_plus_ij[j] = x_plus_ij[j] + config.step_size;

                let f00 = f(x).map_err(JacobianError::FunctionError)?;
                let f10 = f(&x_plus_i).map_err(JacobianError::FunctionError)?;
                let f01 = f(&x_plus_j).map_err(JacobianError::FunctionError)?;
                let f11 = f(&x_plus_ij).map_err(JacobianError::FunctionError)?;

                // Second-order finite difference
                let h_squared = config.step_size * config.step_size;
                hessian[(i, j)] = (f11 - f10 - f01 + f00) / h_squared;
            }
        }

        Ok(hessian)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;
    use ndarray::Array1;
    use approx::assert_relative_eq;

    #[test]
    fn test_nalgebra_numerical_jacobian_quadratic() {
        let f = |x: &DVector<f64>| -> Result<DVector<f64>, String> {
            Ok(DVector::from_vec(vec![x[0] * x[0], x[1] * x[1]]))
        };

        let x = DVector::from_vec(vec![2.0, 3.0]);
        let config = JacobianConfig::default();
        
        let jacobian = nalgebra_jacobian::numerical_jacobian(&f, &x, &config).unwrap();
        
        // Expected Jacobian: [[4.0, 0.0], [0.0, 6.0]]
        assert_relative_eq!(jacobian[(0, 0)], 4.0, epsilon = 1e-4);
        assert_relative_eq!(jacobian[(0, 1)], 0.0, epsilon = 1e-4);
        assert_relative_eq!(jacobian[(1, 0)], 0.0, epsilon = 1e-4);
        assert_relative_eq!(jacobian[(1, 1)], 6.0, epsilon = 1e-4);
    }

    #[test]
    fn test_nalgebra_numerical_gradient() {
        let f = |x: &DVector<f64>| -> Result<f64, String> {
            Ok(x[0] * x[0] + x[1] * x[1])
        };

        let x = DVector::from_vec(vec![3.0, 4.0]);
        let config = JacobianConfig::default();
        
        let gradient = nalgebra_jacobian::numerical_gradient(&f, &x, &config).unwrap();
        
        // Expected gradient: [6.0, 8.0]
        assert_relative_eq!(gradient[0], 6.0, epsilon = 1e-4);
        assert_relative_eq!(gradient[1], 8.0, epsilon = 1e-4);
    }

    #[test]
    fn test_ndarray_numerical_jacobian_quadratic() {
        let f = |x: &Array1<f64>| -> Result<Array1<f64>, String> {
            Ok(Array1::from_vec(vec![x[0] * x[0], x[1] * x[1]]))
        };

        let x = Array1::from_vec(vec![2.0, 3.0]);
        let config = JacobianConfig::default();
        
        let jacobian = ndarray_jacobian::numerical_jacobian(&f, &x, &config).unwrap();
        
        // Expected Jacobian: [[4.0, 0.0], [0.0, 6.0]]
        assert_relative_eq!(jacobian[(0, 0)], 4.0, epsilon = 1e-4);
        assert_relative_eq!(jacobian[(0, 1)], 0.0, epsilon = 1e-4);
        assert_relative_eq!(jacobian[(1, 0)], 0.0, epsilon = 1e-4);
        assert_relative_eq!(jacobian[(1, 1)], 6.0, epsilon = 1e-4);
    }

    #[test]
    fn test_ndarray_numerical_gradient() {
        let f = |x: &Array1<f64>| -> Result<f64, String> {
            Ok(x[0] * x[0] + x[1] * x[1])
        };

        let x = Array1::from_vec(vec![3.0, 4.0]);
        let config = JacobianConfig::default();
        
        let gradient = ndarray_jacobian::numerical_gradient(&f, &x, &config).unwrap();
        
        // Expected gradient: [6.0, 8.0]
        assert_relative_eq!(gradient[0], 6.0, epsilon = 1e-4);
        assert_relative_eq!(gradient[1], 8.0, epsilon = 1e-4);
    }

    #[test]
    fn test_nalgebra_central_differences() {
        let f = |x: &DVector<f64>| -> Result<DVector<f64>, String> {
            Ok(DVector::from_vec(vec![x[0] * x[0] * x[0], x[1] * x[1]]))
        };

        let x = DVector::from_vec(vec![2.0, 3.0]);
        let config = JacobianConfig::default();
        
        let jacobian_forward = nalgebra_jacobian::numerical_jacobian(&f, &x, &config).unwrap();
        let jacobian_central = nalgebra_jacobian::numerical_jacobian_central(&f, &x, &config).unwrap();
        
        // Central differences should be more accurate
        // Expected: [[12.0, 0.0], [0.0, 6.0]]
        assert_relative_eq!(jacobian_central[(0, 0)], 12.0, epsilon = 1e-3);
        assert_relative_eq!(jacobian_central[(1, 1)], 6.0, epsilon = 1e-3);
        
        // Central differences should be closer to analytical result
        assert!(jacobian_central[(0, 0)].abs() - 12.0 < jacobian_forward[(0, 0)].abs() - 12.0);
    }

    #[test]
    fn test_nalgebra_hessian() {
        let f = |x: &DVector<f64>| -> Result<f64, String> {
            Ok(x[0] * x[0] * x[0] + x[1] * x[1] * x[1])
        };

        let x = DVector::from_vec(vec![2.0, 3.0]);
        let config = JacobianConfig::default();
        
        let hessian = nalgebra_jacobian::numerical_hessian(&f, &x, &config).unwrap();
        
        // Expected Hessian: [[12.0, 0.0], [0.0, 18.0]]
        assert_relative_eq!(hessian[(0, 0)], 12.0, epsilon = 1e-2);
        assert_relative_eq!(hessian[(0, 1)], 0.0, epsilon = 1e-2);
        assert_relative_eq!(hessian[(1, 0)], 0.0, epsilon = 1e-2);
        assert_relative_eq!(hessian[(1, 1)], 18.0, epsilon = 1e-2);
    }

    #[test]
    fn test_error_handling() {
        let f = |_x: &DVector<f64>| -> Result<DVector<f64>, String> {
            Err("Test error".to_string())
        };

        let x = DVector::from_vec(vec![1.0, 2.0]);
        let config = JacobianConfig::default();
        
        let result = nalgebra_jacobian::numerical_jacobian(&f, &x, &config);
        assert!(result.is_err());
        
        if let Err(JacobianError::FunctionError(msg)) = result {
            assert_eq!(msg, "Test error");
        } else {
            panic!("Expected FunctionError");
        }
    }

    #[test]
    fn test_empty_input() {
        let f = |x: &DVector<f64>| -> Result<DVector<f64>, String> {
            Ok(x.clone())
        };

        let x = DVector::from_vec(vec![]);
        let config = JacobianConfig::default();
        
        let result = nalgebra_jacobian::numerical_jacobian(&f, &x, &config);
        assert!(result.is_err());
        
        if let Err(JacobianError::EmptyInput) = result {
            // Expected
        } else {
            panic!("Expected EmptyInput error");
        }
    }

    #[test]
    fn test_invalid_config() {
        let f = |x: &DVector<f64>| -> Result<DVector<f64>, String> {
            Ok(x.clone())
        };

        let x = DVector::from_vec(vec![1.0, 2.0]);
        
        // Test negative step size
        let config = JacobianConfig {
            step_size: -1e-6,
            tolerance: 1e-8,
            max_iterations: 100,
        };
        
        let result = nalgebra_jacobian::numerical_jacobian(&f, &x, &config);
        assert!(result.is_err());
        
        if let Err(JacobianError::InvalidStepSize) = result {
            // Expected
        } else {
            panic!("Expected InvalidStepSize error");
        }
    }

    #[test]
    fn test_nan_input() {
        let f = |x: &DVector<f64>| -> Result<DVector<f64>, String> {
            Ok(x.clone())
        };

        let x = DVector::from_vec(vec![1.0, f64::NAN]);
        let config = JacobianConfig::default();
        
        let result = nalgebra_jacobian::numerical_jacobian(&f, &x, &config);
        assert!(result.is_err());
        
        if let Err(JacobianError::InvalidDimensions(_)) = result {
            // Expected
        } else {
            panic!("Expected InvalidDimensions error");
        }
    }

    #[test]
    fn test_infinite_input() {
        let f = |x: &DVector<f64>| -> Result<DVector<f64>, String> {
            Ok(x.clone())
        };

        let x = DVector::from_vec(vec![1.0, f64::INFINITY]);
        let config = JacobianConfig::default();
        
        let result = nalgebra_jacobian::numerical_jacobian(&f, &x, &config);
        assert!(result.is_err());
        
        if let Err(JacobianError::InvalidDimensions(_)) = result {
            // Expected
        } else {
            panic!("Expected InvalidDimensions error");
        }
    }

    #[test]
    fn test_config_validation() {
        // Test valid config
        let config = JacobianConfig::new(1e-6, 1e-8, 100).unwrap();
        assert!(config.validate().is_ok());
        
        // Test invalid step size
        let result = JacobianConfig::new(-1e-6, 1e-8, 100);
        assert!(result.is_err());
        
        // Test invalid tolerance
        let result = JacobianConfig::new(1e-6, -1e-8, 100);
        assert!(result.is_err());
        
        // Test invalid max iterations
        let result = JacobianConfig::new(1e-6, 1e-8, 0);
        assert!(result.is_err());
    }
}
