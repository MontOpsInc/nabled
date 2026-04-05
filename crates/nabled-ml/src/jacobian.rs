//! Numerical Jacobian/gradient/Hessian computation over ndarray vectors.

use std::fmt;

use nabled_core::scalar::NabledReal;
use ndarray::{Array1, Array2, ArrayBase, Data, Ix1};

/// Error type for Jacobian-related computations.
#[derive(Debug, Clone, PartialEq)]
pub enum JacobianError {
    /// Function evaluation returned an error.
    FunctionError(String),
    /// Invalid dimensions.
    InvalidDimensions(String),
    /// Step size is invalid.
    InvalidStepSize,
    /// Iterative convergence failed.
    ConvergenceFailed,
    /// Input is empty.
    EmptyInput,
    /// Input/output dimensions mismatch.
    DimensionMismatch,
}

impl fmt::Display for JacobianError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            JacobianError::FunctionError(message) => write!(f, "Function error: {message}"),
            JacobianError::InvalidDimensions(message) => write!(f, "Invalid dimensions: {message}"),
            JacobianError::InvalidStepSize => write!(f, "Invalid step size"),
            JacobianError::ConvergenceFailed => write!(f, "Convergence failed"),
            JacobianError::EmptyInput => write!(f, "Empty input"),
            JacobianError::DimensionMismatch => write!(f, "Dimension mismatch"),
        }
    }
}

impl std::error::Error for JacobianError {}

/// Configuration for finite-difference numerical derivatives.
#[derive(Debug, Clone)]
pub struct JacobianConfig<T = f64> {
    /// Finite-difference step size.
    pub step_size:      T,
    /// Relative tolerance.
    pub tolerance:      T,
    /// Maximum iterations.
    pub max_iterations: usize,
}

impl Default for JacobianConfig<f64> {
    fn default() -> Self { Self { step_size: 1e-6, tolerance: 1e-8, max_iterations: 100 } }
}

impl Default for JacobianConfig<f32> {
    fn default() -> Self { Self { step_size: 1e-4, tolerance: 1e-5, max_iterations: 100 } }
}

impl<T: NabledReal> JacobianConfig<T> {
    /// Create a validated config.
    ///
    /// # Errors
    /// Returns an error when any parameter is invalid.
    pub fn new(step_size: T, tolerance: T, max_iterations: usize) -> Result<Self, JacobianError> {
        let config = Self { step_size, tolerance, max_iterations };
        config.validate()?;
        Ok(config)
    }

    /// Validate the configuration.
    ///
    /// # Errors
    /// Returns an error when parameters are invalid.
    pub fn validate(&self) -> Result<(), JacobianError> {
        if !self.step_size.is_finite() || self.step_size <= T::zero() {
            return Err(JacobianError::InvalidStepSize);
        }
        if !self.tolerance.is_finite() || self.tolerance <= T::zero() {
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

/// Compute numerical Jacobian via forward differences.
///
/// # Errors
/// Returns an error for invalid input/config or function failures.
pub fn numerical_jacobian<T, F, S>(
    function: &F,
    x: &ArrayBase<S, Ix1>,
    config: &JacobianConfig<T>,
) -> Result<Array2<T>, JacobianError>
where
    F: Fn(&Array1<T>) -> Result<Array1<T>, JacobianError>,
    T: NabledReal,
    S: Data<Elem = T>,
{
    config.validate()?;
    if x.is_empty() {
        return Err(JacobianError::EmptyInput);
    }

    let fx = function(&x.to_owned())?;
    if fx.is_empty() {
        return Err(JacobianError::EmptyInput);
    }

    let mut jacobian = Array2::<T>::zeros((fx.len(), x.len()));
    for j in 0..x.len() {
        let mut perturbed = x.to_owned();
        perturbed[j] += config.step_size;
        let f_perturbed = function(&perturbed)?;
        if f_perturbed.len() != fx.len() {
            return Err(JacobianError::DimensionMismatch);
        }
        for i in 0..fx.len() {
            jacobian[[i, j]] = (f_perturbed[i] - fx[i]) / config.step_size;
        }
    }

    Ok(jacobian)
}

/// Compute numerical Jacobian via central differences.
///
/// # Errors
/// Returns an error for invalid input/config or function failures.
pub fn numerical_jacobian_central<T, F, S>(
    function: &F,
    x: &ArrayBase<S, Ix1>,
    config: &JacobianConfig<T>,
) -> Result<Array2<T>, JacobianError>
where
    F: Fn(&Array1<T>) -> Result<Array1<T>, JacobianError>,
    T: NabledReal,
    S: Data<Elem = T>,
{
    config.validate()?;
    if x.is_empty() {
        return Err(JacobianError::EmptyInput);
    }

    let fx = function(&x.to_owned())?;
    if fx.is_empty() {
        return Err(JacobianError::EmptyInput);
    }

    let mut jacobian = Array2::<T>::zeros((fx.len(), x.len()));
    let step = config.step_size;

    for j in 0..x.len() {
        let mut x_plus = x.to_owned();
        let mut x_minus = x.to_owned();
        x_plus[j] += step;
        x_minus[j] -= step;
        let f_plus = function(&x_plus)?;
        let f_minus = function(&x_minus)?;
        if f_plus.len() != fx.len() || f_minus.len() != fx.len() {
            return Err(JacobianError::DimensionMismatch);
        }
        for i in 0..fx.len() {
            jacobian[[i, j]] =
                (f_plus[i] - f_minus[i]) / (T::from_f64(2.0).unwrap_or(T::one() + T::one()) * step);
        }
    }

    Ok(jacobian)
}

/// Compute numerical gradient of a scalar function.
///
/// # Errors
/// Returns an error for invalid input/config or function failures.
pub fn numerical_gradient<T, F, S>(
    function: &F,
    x: &ArrayBase<S, Ix1>,
    config: &JacobianConfig<T>,
) -> Result<Array1<T>, JacobianError>
where
    F: Fn(&Array1<T>) -> Result<T, JacobianError>,
    T: NabledReal,
    S: Data<Elem = T>,
{
    config.validate()?;
    if x.is_empty() {
        return Err(JacobianError::EmptyInput);
    }

    let fx = function(&x.to_owned())?;
    let mut gradient = Array1::<T>::zeros(x.len());
    for j in 0..x.len() {
        let mut perturbed = x.to_owned();
        perturbed[j] += config.step_size;
        let f_perturbed = function(&perturbed)?;
        gradient[j] = (f_perturbed - fx) / config.step_size;
    }

    Ok(gradient)
}

/// Compute numerical Hessian of a scalar function.
///
/// # Errors
/// Returns an error for invalid input/config or function failures.
#[expect(clippy::similar_names)]
pub fn numerical_hessian<T, F, S>(
    function: &F,
    x: &ArrayBase<S, Ix1>,
    config: &JacobianConfig<T>,
) -> Result<Array2<T>, JacobianError>
where
    F: Fn(&Array1<T>) -> Result<T, JacobianError>,
    T: NabledReal,
    S: Data<Elem = T>,
{
    config.validate()?;
    if x.is_empty() {
        return Err(JacobianError::EmptyInput);
    }

    let n = x.len();
    let mut hessian = Array2::<T>::zeros((n, n));
    let step = config.step_size;

    for i in 0..n {
        for j in 0..n {
            let mut x_pp = x.to_owned();
            let mut x_pm = x.to_owned();
            let mut x_mp = x.to_owned();
            let mut x_mm = x.to_owned();

            x_pp[i] += step;
            x_pp[j] += step;

            x_pm[i] += step;
            x_pm[j] -= step;

            x_mp[i] -= step;
            x_mp[j] += step;

            x_mm[i] -= step;
            x_mm[j] -= step;

            let f_pp = function(&x_pp)?;
            let f_pm = function(&x_pm)?;
            let f_mp = function(&x_mp)?;
            let f_mm = function(&x_mm)?;

            hessian[[i, j]] = (f_pp - f_pm - f_mp + f_mm)
                / (T::from_f64(4.0).unwrap_or(T::one() + T::one() + T::one() + T::one())
                    * step
                    * step);
        }
    }

    Ok(hessian)
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, array};

    use super::*;

    #[test]
    fn jacobian_of_quadratic_map() {
        let f = |x: &Array1<f64>| -> Result<Array1<f64>, JacobianError> {
            Ok(array![x[0] * x[0], x[1] * x[1]])
        };
        let x = array![2.0, 3.0];
        let jacobian = numerical_jacobian(&f, &x, &JacobianConfig::default()).unwrap();
        assert!((jacobian[[0, 0]] - 4.0).abs() < 1e-4);
        assert!((jacobian[[1, 1]] - 6.0).abs() < 1e-4);
    }

    #[test]
    fn gradient_of_sphere() {
        let f = |x: &Array1<f64>| -> Result<f64, JacobianError> {
            Ok(x.iter().map(|value| value * value).sum())
        };
        let x = array![1.0, 2.0, 3.0];
        let gradient = numerical_gradient(&f, &x, &JacobianConfig::default()).unwrap();
        assert!((gradient[0] - 2.0).abs() < 1e-4);
        assert!((gradient[1] - 4.0).abs() < 1e-4);
        assert!((gradient[2] - 6.0).abs() < 1e-4);
    }

    #[test]
    fn central_jacobian_matches_forward_for_linear_map() {
        let f = |x: &Array1<f64>| -> Result<Array1<f64>, JacobianError> {
            Ok(array![2.0 * x[0] + x[1], -x[0] + 3.0 * x[1]])
        };
        let x = array![0.5, -1.0];
        let config = JacobianConfig::default();
        let forward = numerical_jacobian(&f, &x, &config).unwrap();
        let central = numerical_jacobian_central(&f, &x, &config).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!((forward[[i, j]] - central[[i, j]]).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn hessian_of_quadratic_is_constant() {
        let f = |x: &Array1<f64>| -> Result<f64, JacobianError> {
            Ok(2.0 * x[0] * x[0] + 3.0 * x[0] * x[1] + 4.0 * x[1] * x[1])
        };
        let x = array![0.7, -1.2];
        let hessian = numerical_hessian(&f, &x, &JacobianConfig::default()).unwrap();
        assert!((hessian[[0, 0]] - 4.0).abs() < 1e-3);
        assert!((hessian[[0, 1]] - 3.0).abs() < 1e-3);
        assert!((hessian[[1, 0]] - 3.0).abs() < 1e-3);
        assert!((hessian[[1, 1]] - 8.0).abs() < 1e-3);
    }

    #[test]
    fn config_validation_rejects_invalid_values() {
        assert!(JacobianConfig::new(0.0, 1e-8, 10).is_err());
        assert!(JacobianConfig::new(1e-6, -1.0, 10).is_err());
        assert!(JacobianConfig::new(1e-6, 1e-8, 0).is_err());
    }

    #[test]
    fn central_jacobian_detects_dimension_mismatch() {
        let f = |x: &Array1<f64>| -> Result<Array1<f64>, JacobianError> {
            if x[0] > 1.0 { Ok(array![1.0, 2.0, 3.0]) } else { Ok(array![1.0, 2.0]) }
        };
        let x = array![1.0, 2.0];
        let result = numerical_jacobian_central(&f, &x, &JacobianConfig::default());
        assert!(matches!(result, Err(JacobianError::DimensionMismatch)));
    }
}
