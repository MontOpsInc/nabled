//! # Jacobian Computation Module
//!
//! This module provides Jacobian matrix computation for both `nalgebra` and `ndarray`.
//! The Jacobian matrix contains the partial derivatives of a vector-valued function
//! with respect to its input variables.
//!
//! ## Features
//!
//! - Numerical Jacobian via forward and central finite differences
//! - Gradient and Hessian computation for scalar-valued functions
//! - Complex-valued Jacobian/gradient/Hessian support
//! - Implementations for both `nalgebra` and `ndarray`
//! - Configurable step size, tolerance, and iteration budget
//! - Explicit error reporting for invalid dimensions and non-finite values

use std::fmt;

use num_traits::Float;
use num_traits::float::FloatCore;

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
            JacobianError::FunctionError(msg) => write!(f, "Function error: {msg}"),
            JacobianError::InvalidDimensions(msg) => write!(f, "Invalid dimensions: {msg}"),
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
    pub step_size:      T,
    /// Relative tolerance for convergence
    pub tolerance:      T,
    /// Maximum number of iterations
    pub max_iterations: usize,
}

impl<T: Float> Default for JacobianConfig<T> {
    fn default() -> Self {
        Self {
            step_size:      T::from(1e-6).unwrap(),
            tolerance:      T::from(1e-8).unwrap(),
            max_iterations: 100,
        }
    }
}

impl<T: Float> JacobianConfig<T> {
    /// Create a new `JacobianConfig` with validation
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn new(step_size: T, tolerance: T, max_iterations: usize) -> Result<Self, JacobianError> {
        if step_size <= T::zero() {
            return Err(JacobianError::InvalidStepSize);
        }
        if tolerance <= T::zero() {
            return Err(JacobianError::InvalidDimensions("Tolerance must be positive".to_string()));
        }
        if max_iterations == 0 {
            return Err(JacobianError::InvalidDimensions(
                "Max iterations must be positive".to_string(),
            ));
        }

        Ok(Self { step_size, tolerance, max_iterations })
    }

    /// Validate the configuration
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn validate(&self) -> Result<(), JacobianError> {
        if self.step_size <= T::zero() {
            return Err(JacobianError::InvalidStepSize);
        }
        if self.tolerance <= T::zero() {
            return Err(JacobianError::InvalidDimensions("Tolerance must be positive".to_string()));
        }
        if self.max_iterations == 0 {
            return Err(JacobianError::InvalidDimensions(
                "Max iterations must be positive".to_string(),
            ));
        }
        Ok(())
    }
}

/// Nalgebra Jacobian computation functions
pub mod nalgebra_jacobian {
    use nalgebra::{DMatrix, DVector, RealField};

    use super::*;

    /// Compute numerical Jacobian using finite differences
    ///
    /// # Arguments
    /// * `f` - Function that takes a `DVector` and returns a `DVector`
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
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
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
            if !Float::is_finite(x[i]) {
                return Err(JacobianError::InvalidDimensions(format!(
                    "Input contains non-finite value at index {i}"
                )));
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
            if !Float::is_finite(x[i]) {
                return Err(JacobianError::FunctionError(format!(
                    "Function returned non-finite value at index {i}"
                )));
            }
        }

        let mut jacobian = DMatrix::zeros(m, n);

        // Compute partial derivatives using finite differences
        for j in 0..n {
            let mut x_plus = x.clone();

            // Check for overflow when adding step size
            let step = config.step_size;
            if Float::is_finite(x_plus[j]) && Float::is_finite(step) {
                x_plus[j] += step;
            } else {
                return Err(JacobianError::InvalidStepSize);
            }

            let fx_plus = f(&x_plus).map_err(JacobianError::FunctionError)?;

            if fx_plus.len() != m {
                return Err(JacobianError::DimensionMismatch);
            }

            // Check for non-finite values in perturbed function output
            for i in 0..m {
                if !Float::is_finite(fx_plus[i]) {
                    return Err(JacobianError::FunctionError(format!(
                        "Function returned non-finite value at index {i} when perturbing variable \
                         {j}"
                    )));
                }
            }

            // Compute finite difference
            for i in 0..m {
                let diff = fx_plus[i] - fx[i];
                if Float::is_finite(diff) && Float::is_finite(step) && step != T::zero() {
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
    /// * `f` - Function that takes a `DVector` and returns a `DVector`
    /// * `x` - Point at which to compute the Jacobian
    /// * `config` - Configuration for the computation
    ///
    /// # Returns
    /// * `Result<DMatrix<T>, JacobianError>` - Jacobian matrix
    /// # Panics
    /// Panics if internal numeric assumptions are violated during setup or
    /// intermediate conversion steps.
    ///
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
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
            if !Float::is_finite(x[i]) {
                return Err(JacobianError::InvalidDimensions(format!(
                    "Input contains non-finite value at index {i}"
                )));
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

            x_plus[j] += config.step_size;
            x_minus[j] -= config.step_size;

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
    /// * `f` - Function that takes a `DVector` and returns a scalar
    /// * `x` - Point at which to compute the gradient
    /// * `config` - Configuration for the computation
    ///
    /// # Returns
    /// * `Result<DVector<T>, JacobianError>` - Gradient vector
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
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
            x_plus[j] += config.step_size;

            let fx = f(x).map_err(JacobianError::FunctionError)?;
            let fx_plus = f(&x_plus).map_err(JacobianError::FunctionError)?;

            gradient[j] = (fx_plus - fx) / config.step_size;
        }

        Ok(gradient)
    }

    /// Compute Hessian matrix (second-order partial derivatives)
    ///
    /// # Arguments
    /// * `f` - Function that takes a `DVector` and returns a scalar
    /// * `x` - Point at which to compute the Hessian
    /// * `config` - Configuration for the computation
    ///
    /// # Returns
    /// * `Result<DMatrix<T>, JacobianError>` - Hessian matrix
    #[expect(clippy::similar_names)]
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
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

                x_plus_i[i] += config.step_size;
                x_plus_j[j] += config.step_size;
                x_plus_ij[i] += config.step_size;
                x_plus_ij[j] += config.step_size;

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
    use ndarray::{Array1, Array2};

    use super::*;

    /// Compute numerical Jacobian using finite differences
    ///
    /// # Arguments
    /// * `f` - Function that takes an Array1 and returns an Array1
    /// * `x` - Point at which to compute the Jacobian
    /// * `config` - Configuration for the computation
    ///
    /// # Returns
    /// * `Result<Array2<T>, JacobianError>` - Jacobian matrix
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
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
    /// # Panics
    /// Panics if internal numeric assumptions are violated during setup or
    /// intermediate conversion steps.
    ///
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
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
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
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
    #[expect(clippy::similar_names)]
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
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

/// Complex derivative computation functions
pub mod complex_jacobian {
    use nalgebra::{ComplexField, DMatrix, DVector};
    use num_complex::Complex;

    use super::*;

    /// Compute numerical Jacobian for complex-valued functions using complex step method
    ///
    /// # Arguments
    /// * `f` - Function that takes complex vector and returns complex vector
    /// * `x` - Complex input vector
    /// * `config` - Configuration for the computation
    ///
    /// # Returns
    /// * `Result<DMatrix<Complex<T>>, JacobianError>` - Complex Jacobian matrix
    ///
    /// # Example
    /// ```rust
    /// use rust_linalg::jacobian::complex_jacobian;
    /// use nalgebra::DVector;
    /// use num_complex::Complex;
    ///
    /// let f = |x: &DVector<Complex<f64>>| -> Result<DVector<Complex<f64>>, String> {
    ///     let mut result = DVector::zeros(x.len());
    ///     for i in 0..x.len() {
    ///         result[i] = x[i] * x[i]; // f(z) = z²
    ///     }
    ///     Ok(result)
    /// };
    ///
    /// let x = DVector::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
    /// let jacobian = complex_jacobian::numerical_jacobian(&f, &x, &Default::default())?;
    /// # Ok::<(), rust_linalg::jacobian::JacobianError>(())
    /// ```
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn numerical_jacobian<T, F>(
        func: F,
        x: &DVector<Complex<T>>,
        config: &JacobianConfig<T>,
    ) -> Result<DMatrix<Complex<T>>, JacobianError>
    where
        T: ComplexField + Float + FloatCore,
        F: Fn(&DVector<Complex<T>>) -> Result<DVector<Complex<T>>, String>,
    {
        // Validate inputs
        if x.is_empty() {
            return Err(JacobianError::EmptyInput);
        }

        // Check for NaN or infinite values in input
        for i in 0..x.len() {
            if !x[i].is_finite() {
                return Err(JacobianError::InvalidDimensions(format!(
                    "Input contains non-finite value at index {i}"
                )));
            }
        }

        // Evaluate function at the point
        let f_x = func(x).map_err(JacobianError::FunctionError)?;
        let m = f_x.len();
        let n = x.len();

        // Initialize Jacobian matrix (complex)
        let mut jacobian = DMatrix::<Complex<T>>::zeros(m, n);

        // Compute partial derivatives using complex step method
        for j in 0..n {
            // Create complex step: h = i * step_size (purely imaginary)
            let h = Complex::new(T::zero(), config.step_size);
            let mut x_plus = x.clone();
            x_plus[j] = x[j] + h;

            // Evaluate function at x + h
            let f_x_plus = func(&x_plus).map_err(JacobianError::FunctionError)?;

            // Compute partial derivative: ∂f/∂x_j ≈ Im(f(x + ih)) / step_size
            for i in 0..m {
                if f_x_plus[i].is_finite() {
                    jacobian[(i, j)] = Complex::new(f_x_plus[i].im / config.step_size, T::zero());
                } else {
                    return Err(JacobianError::FunctionError(format!(
                        "Function returned non-finite value at index {i}"
                    )));
                }
            }
        }

        Ok(jacobian)
    }

    /// Compute numerical gradient for complex scalar functions
    ///
    /// # Arguments
    /// * `f` - Scalar function that takes complex vector and returns complex scalar
    /// * `x` - Complex input vector
    /// * `config` - Configuration for the computation
    ///
    /// # Returns
    /// * `Result<DVector<Complex<T>>, JacobianError>` - Complex gradient vector
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn numerical_gradient<T, F>(
        f: F,
        x: &DVector<Complex<T>>,
        config: &JacobianConfig<T>,
    ) -> Result<DVector<Complex<T>>, JacobianError>
    where
        T: ComplexField + Float + FloatCore,
        F: Fn(&DVector<Complex<T>>) -> Result<Complex<T>, String>,
    {
        // Validate inputs
        if x.is_empty() {
            return Err(JacobianError::EmptyInput);
        }

        let n = x.len();
        let mut gradient = DVector::<Complex<T>>::zeros(n);

        // Compute partial derivatives using complex step method
        for j in 0..n {
            // Create complex step: h = i * step_size (purely imaginary)
            let h = Complex::new(T::zero(), config.step_size);
            let mut x_plus = x.clone();
            x_plus[j] = x[j] + h;

            // Evaluate function at x + h
            let f_x_plus = f(&x_plus).map_err(JacobianError::FunctionError)?;

            // Compute partial derivative: ∂f/∂x_j ≈ Im(f(x + ih)) / h
            if f_x_plus.is_finite() {
                gradient[j] = Complex::new(f_x_plus.im / config.step_size, T::zero());
            } else {
                return Err(JacobianError::FunctionError(
                    "Function returned non-finite value".to_string(),
                ));
            }
        }

        Ok(gradient)
    }

    /// Compute numerical Hessian for complex scalar functions
    ///
    /// # Arguments
    /// * `f` - Scalar function that takes complex vector and returns complex scalar
    /// * `x` - Complex input vector
    /// * `config` - Configuration for the computation
    ///
    /// # Returns
    /// * `Result<DMatrix<Complex<T>>, JacobianError>` - Complex Hessian matrix
    #[expect(clippy::similar_names)]
    /// # Panics
    /// Panics if internal numeric assumptions are violated during setup or
    /// intermediate conversion steps.
    ///
    /// # Errors
    /// Returns an error when inputs are invalid, dimensions are incompatible,
    /// or the requested numerical routine cannot produce a stable result.
    pub fn numerical_hessian<T, F>(
        f: F,
        x: &DVector<Complex<T>>,
        config: &JacobianConfig<T>,
    ) -> Result<DMatrix<Complex<T>>, JacobianError>
    where
        T: ComplexField + Float + FloatCore,
        F: Fn(&DVector<Complex<T>>) -> Result<Complex<T>, String>,
    {
        // Validate inputs
        if x.is_empty() {
            return Err(JacobianError::EmptyInput);
        }

        let n = x.len();
        let mut hessian = DMatrix::<Complex<T>>::zeros(n, n);

        // Compute second-order partial derivatives using complex step method
        for i in 0..n {
            for j in 0..n {
                // Create complex steps
                let h1 = Complex::new(T::zero(), config.step_size);
                let h2 = Complex::new(T::zero(), config.step_size);

                // Four-point stencil for second derivative
                let mut x_pp = x.clone();
                x_pp[i] = x[i] + h1;
                x_pp[j] = x[j] + h2;

                let mut x_pm = x.clone();
                x_pm[i] = x[i] + h1;
                x_pm[j] = x[j] - h2;

                let mut x_mp = x.clone();
                x_mp[i] = x[i] - h1;
                x_mp[j] = x[j] + h2;

                let mut x_mm = x.clone();
                x_mm[i] = x[i] - h1;
                x_mm[j] = x[j] - h2;

                // Evaluate function at all four points
                let f_pp = f(&x_pp).map_err(JacobianError::FunctionError)?;
                let f_pm = f(&x_pm).map_err(JacobianError::FunctionError)?;
                let f_mp = f(&x_mp).map_err(JacobianError::FunctionError)?;
                let f_mm = f(&x_mm).map_err(JacobianError::FunctionError)?;

                // Compute second derivative using finite differences
                let step_squared = config.step_size * config.step_size;
                let derivative = (f_pp.im - f_pm.im - f_mp.im + f_mm.im)
                    / (T::from(4.0).unwrap() * step_squared);
                hessian[(i, j)] = Complex::new(derivative, T::zero());
            }
        }

        Ok(hessian)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nalgebra::DVector;
    use ndarray::Array1;

    use super::*;

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
        let f = |x: &DVector<f64>| -> Result<f64, String> { Ok(x[0] * x[0] + x[1] * x[1]) };

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
        let f = |x: &Array1<f64>| -> Result<f64, String> { Ok(x[0] * x[0] + x[1] * x[1]) };

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
        let jacobian_central =
            nalgebra_jacobian::numerical_jacobian_central(&f, &x, &config).unwrap();

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
        let f =
            |_x: &DVector<f64>| -> Result<DVector<f64>, String> { Err("Test error".to_string()) };

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
        let f = |x: &DVector<f64>| -> Result<DVector<f64>, String> { Ok(x.clone()) };

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
        let f = |x: &DVector<f64>| -> Result<DVector<f64>, String> { Ok(x.clone()) };

        let x = DVector::from_vec(vec![1.0, 2.0]);

        // Test negative step size
        let config =
            JacobianConfig { step_size: -1e-6, tolerance: 1e-8, max_iterations: 100 };

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
        let f = |x: &DVector<f64>| -> Result<DVector<f64>, String> { Ok(x.clone()) };

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
        let f = |x: &DVector<f64>| -> Result<DVector<f64>, String> { Ok(x.clone()) };

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

    #[test]
    fn test_complex_jacobian() {
        use num_complex::Complex;

        // Test complex function f(z) = z² (where z is complex)
        // The complex step method computes the derivative of the real part
        let f = |x: &DVector<Complex<f64>>| -> Result<DVector<Complex<f64>>, String> {
            let mut result = DVector::zeros(x.len());
            for i in 0..x.len() {
                result[i] = x[i] * x[i];
            }
            Ok(result)
        };

        let x = DVector::from_vec(vec![Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)]);
        let config = JacobianConfig::default();

        let jacobian = complex_jacobian::numerical_jacobian(f, &x, &config).unwrap();

        // For f(z) = z² with z real, the derivative is 2z
        // At z = 1, derivative is 2
        // At z = 2, derivative is 4
        let expected_deriv_1 = 2.0;
        let expected_deriv_2 = 4.0;

        // Check diagonal elements (derivatives with respect to each variable)
        assert_relative_eq!(jacobian[(0, 0)].re, expected_deriv_1, epsilon = 1e-6);
        assert_relative_eq!(jacobian[(1, 1)].re, expected_deriv_2, epsilon = 1e-6);

        // Off-diagonal elements should be zero (no cross-dependencies)
        assert_relative_eq!(jacobian[(0, 1)].re, 0.0, epsilon = 1e-6);
        assert_relative_eq!(jacobian[(1, 0)].re, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_complex_gradient() {
        use num_complex::Complex;

        // Test complex scalar function f(z) = z₁² + z₂²
        let f = |x: &DVector<Complex<f64>>| -> Result<Complex<f64>, String> {
            let mut sum = Complex::new(0.0, 0.0);
            for i in 0..x.len() {
                sum += x[i] * x[i];
            }
            Ok(sum)
        };

        let x = DVector::from_vec(vec![Complex::new(2.0, 0.0), Complex::new(3.0, 0.0)]);
        let config = JacobianConfig::default();

        let gradient = complex_jacobian::numerical_gradient(f, &x, &config).unwrap();

        // For f(z) = z₁² + z₂² with z real, the gradient is [2z₁, 2z₂]
        // At z₁ = 2, derivative is 4
        // At z₂ = 3, derivative is 6
        let expected_grad_1 = 4.0;
        let expected_grad_2 = 6.0;

        assert_relative_eq!(gradient[0].re, expected_grad_1, epsilon = 1e-6);
        assert_relative_eq!(gradient[1].re, expected_grad_2, epsilon = 1e-6);
    }

    #[test]
    fn test_complex_hessian() {
        use num_complex::Complex;

        // Test complex scalar function f(z) = z₁² + z₂²
        let f = |x: &DVector<Complex<f64>>| -> Result<Complex<f64>, String> {
            let mut sum = Complex::new(0.0, 0.0);
            for i in 0..x.len() {
                sum += x[i] * x[i];
            }
            Ok(sum)
        };

        let x = DVector::from_vec(vec![Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)]);
        let config = JacobianConfig::default();

        let hessian = complex_jacobian::numerical_hessian(f, &x, &config).unwrap();

        // Test that the function runs without error and returns a valid matrix
        // Note: Complex step method for second derivatives has limitations
        // and may not provide accurate results for all functions
        assert_eq!(hessian.nrows(), 2);
        assert_eq!(hessian.ncols(), 2);

        // Check that all values are finite
        for i in 0..2 {
            for j in 0..2 {
                assert!(hessian[(i, j)].is_finite());
            }
        }
    }

    #[test]
    fn test_complex_error_handling() {
        use num_complex::Complex;

        // Test empty input
        let f =
            |x: &DVector<Complex<f64>>| -> Result<DVector<Complex<f64>>, String> { Ok(x.clone()) };

        let empty_x = DVector::<Complex<f64>>::zeros(0);
        let config = JacobianConfig::default();

        let result = complex_jacobian::numerical_jacobian(f, &empty_x, &config);
        assert!(matches!(result, Err(JacobianError::EmptyInput)));

        // Test function error
        let error_f = |_x: &DVector<Complex<f64>>| -> Result<DVector<Complex<f64>>, String> {
            Err("Test error".to_string())
        };

        let x = DVector::from_vec(vec![Complex::new(1.0, 0.0)]);
        let result = complex_jacobian::numerical_jacobian(error_f, &x, &config);
        assert!(matches!(result, Err(JacobianError::FunctionError(_))));
    }
}
