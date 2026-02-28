//! First-order optimization primitives over ndarray vectors.

use std::fmt;

use ndarray::Array1;

const DEFAULT_TOLERANCE: f64 = 1.0e-12;

/// Error type for optimization routines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationError {
    /// Input vectors are empty.
    EmptyInput,
    /// Input dimensions are incompatible.
    DimensionMismatch,
    /// Non-finite values were observed.
    NonFiniteInput,
    /// Invalid optimizer configuration.
    InvalidConfig,
    /// Optimizer exceeded iteration budget.
    MaxIterationsExceeded,
}

impl fmt::Display for OptimizationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OptimizationError::EmptyInput => write!(f, "input cannot be empty"),
            OptimizationError::DimensionMismatch => write!(f, "input dimensions are incompatible"),
            OptimizationError::NonFiniteInput => write!(f, "input contains non-finite values"),
            OptimizationError::InvalidConfig => write!(f, "invalid optimizer configuration"),
            OptimizationError::MaxIterationsExceeded => write!(f, "maximum iterations exceeded"),
        }
    }
}

impl std::error::Error for OptimizationError {}

/// Configuration for backtracking line search.
#[derive(Debug, Clone, Copy)]
pub struct LineSearchConfig {
    /// Initial step size.
    pub initial_step:        f64,
    /// Contraction factor in `(0, 1)`.
    pub contraction:         f64,
    /// Armijo sufficient decrease coefficient in `(0, 1)`.
    pub sufficient_decrease: f64,
    /// Maximum backtracking iterations.
    pub max_iterations:      usize,
}

impl Default for LineSearchConfig {
    fn default() -> Self {
        Self {
            initial_step:        1.0,
            contraction:         0.5,
            sufficient_decrease: 1e-4,
            max_iterations:      64,
        }
    }
}

/// Configuration for SGD.
#[derive(Debug, Clone, Copy)]
pub struct SGDConfig {
    /// Fixed learning rate.
    pub learning_rate:  f64,
    /// Maximum optimization iterations.
    pub max_iterations: usize,
    /// Gradient norm tolerance for convergence.
    pub tolerance:      f64,
}

impl Default for SGDConfig {
    fn default() -> Self { Self { learning_rate: 1e-2, max_iterations: 10_000, tolerance: 1e-8 } }
}

/// Configuration for Adam optimizer.
#[derive(Debug, Clone, Copy)]
pub struct AdamConfig {
    /// Base learning rate.
    pub learning_rate:  f64,
    /// Exponential decay for first moment.
    pub beta1:          f64,
    /// Exponential decay for second moment.
    pub beta2:          f64,
    /// Numerical epsilon.
    pub epsilon:        f64,
    /// Maximum optimization iterations.
    pub max_iterations: usize,
    /// Gradient norm tolerance for convergence.
    pub tolerance:      f64,
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            learning_rate:  1e-2,
            beta1:          0.9,
            beta2:          0.999,
            epsilon:        1e-8,
            max_iterations: 10_000,
            tolerance:      1e-8,
        }
    }
}

fn l2_norm(vector: &Array1<f64>) -> f64 {
    vector.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn validate_vector(vector: &Array1<f64>) -> Result<(), OptimizationError> {
    if vector.is_empty() {
        return Err(OptimizationError::EmptyInput);
    }
    if vector.iter().any(|value| !value.is_finite()) {
        return Err(OptimizationError::NonFiniteInput);
    }
    Ok(())
}

fn validate_line_search_config(config: &LineSearchConfig) -> Result<(), OptimizationError> {
    if config.initial_step <= 0.0
        || !(0.0..1.0).contains(&config.contraction)
        || !(0.0..1.0).contains(&config.sufficient_decrease)
        || config.max_iterations == 0
    {
        return Err(OptimizationError::InvalidConfig);
    }
    Ok(())
}

fn validate_sgd_config(config: &SGDConfig) -> Result<(), OptimizationError> {
    if config.learning_rate <= 0.0 || config.max_iterations == 0 || config.tolerance < 0.0 {
        return Err(OptimizationError::InvalidConfig);
    }
    Ok(())
}

fn validate_adam_config(config: &AdamConfig) -> Result<(), OptimizationError> {
    if config.learning_rate <= 0.0
        || !(0.0..1.0).contains(&config.beta1)
        || !(0.0..1.0).contains(&config.beta2)
        || config.epsilon <= 0.0
        || config.max_iterations == 0
        || config.tolerance < 0.0
    {
        return Err(OptimizationError::InvalidConfig);
    }
    Ok(())
}

/// Ndarray optimization routines.
pub mod ndarray_optimization {
    use super::*;

    /// Perform Armijo backtracking line search.
    ///
    /// # Errors
    /// Returns an error for invalid inputs/configuration or non-finite objective evaluations.
    pub fn backtracking_line_search<F, G>(
        point: &Array1<f64>,
        direction: &Array1<f64>,
        objective: F,
        gradient: G,
        config: &LineSearchConfig,
    ) -> Result<f64, OptimizationError>
    where
        F: Fn(&Array1<f64>) -> f64,
        G: Fn(&Array1<f64>) -> Array1<f64>,
    {
        validate_vector(point)?;
        validate_vector(direction)?;
        if point.len() != direction.len() {
            return Err(OptimizationError::DimensionMismatch);
        }
        validate_line_search_config(config)?;

        let grad = gradient(point);
        if grad.len() != point.len() || grad.iter().any(|value| !value.is_finite()) {
            return Err(OptimizationError::NonFiniteInput);
        }

        let fx = objective(point);
        if !fx.is_finite() {
            return Err(OptimizationError::NonFiniteInput);
        }
        let directional_derivative = grad.dot(direction);

        let mut alpha = config.initial_step;
        for _ in 0..config.max_iterations {
            let candidate = point + &(alpha * direction);
            let candidate_value = objective(&candidate);
            if !candidate_value.is_finite() {
                return Err(OptimizationError::NonFiniteInput);
            }
            if candidate_value <= fx + config.sufficient_decrease * alpha * directional_derivative {
                return Ok(alpha);
            }
            alpha *= config.contraction;
        }
        Err(OptimizationError::MaxIterationsExceeded)
    }

    /// Minimize objective with fixed-step gradient descent.
    ///
    /// # Errors
    /// Returns an error for invalid inputs/configuration or non-finite gradients.
    pub fn gradient_descent<F, G>(
        initial: &Array1<f64>,
        objective: F,
        gradient: G,
        config: &SGDConfig,
    ) -> Result<Array1<f64>, OptimizationError>
    where
        F: Fn(&Array1<f64>) -> f64,
        G: Fn(&Array1<f64>) -> Array1<f64>,
    {
        validate_vector(initial)?;
        validate_sgd_config(config)?;

        let mut x = initial.clone();
        let _ = objective(&x);
        let tolerance = config.tolerance.max(DEFAULT_TOLERANCE);

        for _ in 0..config.max_iterations {
            let grad = gradient(&x);
            if grad.len() != x.len() || grad.iter().any(|value| !value.is_finite()) {
                return Err(OptimizationError::NonFiniteInput);
            }
            if l2_norm(&grad) <= tolerance {
                return Ok(x);
            }
            x = &x - &(config.learning_rate * &grad);
        }

        Err(OptimizationError::MaxIterationsExceeded)
    }

    /// Minimize objective with Adam.
    ///
    /// # Errors
    /// Returns an error for invalid inputs/configuration or non-finite gradients.
    pub fn adam<F, G>(
        initial: &Array1<f64>,
        objective: F,
        gradient: G,
        config: &AdamConfig,
    ) -> Result<Array1<f64>, OptimizationError>
    where
        F: Fn(&Array1<f64>) -> f64,
        G: Fn(&Array1<f64>) -> Array1<f64>,
    {
        validate_vector(initial)?;
        validate_adam_config(config)?;

        let mut x = initial.clone();
        let mut m = Array1::<f64>::zeros(x.len());
        let mut v = Array1::<f64>::zeros(x.len());
        let mut beta1_power = 1.0_f64;
        let mut beta2_power = 1.0_f64;
        let tolerance = config.tolerance.max(DEFAULT_TOLERANCE);

        let _ = objective(&x);
        for _ in 0..config.max_iterations {
            let grad = gradient(&x);
            if grad.len() != x.len() || grad.iter().any(|value| !value.is_finite()) {
                return Err(OptimizationError::NonFiniteInput);
            }
            if l2_norm(&grad) <= tolerance {
                return Ok(x);
            }

            beta1_power *= config.beta1;
            beta2_power *= config.beta2;

            for i in 0..x.len() {
                m[i] = config.beta1 * m[i] + (1.0 - config.beta1) * grad[i];
                v[i] = config.beta2 * v[i] + (1.0 - config.beta2) * grad[i] * grad[i];

                let m_hat = m[i] / (1.0 - beta1_power);
                let v_hat = v[i] / (1.0 - beta2_power);
                x[i] -= config.learning_rate * m_hat / (v_hat.sqrt() + config.epsilon);
            }
        }

        Err(OptimizationError::MaxIterationsExceeded)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::arr1;

    use super::{AdamConfig, LineSearchConfig, SGDConfig, ndarray_optimization};

    fn objective(x: &ndarray::Array1<f64>) -> f64 {
        let delta = x[0] - 3.0;
        delta * delta
    }

    fn gradient(x: &ndarray::Array1<f64>) -> ndarray::Array1<f64> { arr1(&[2.0 * (x[0] - 3.0)]) }

    #[test]
    fn backtracking_line_search_finds_descent_step() {
        let x = arr1(&[0.0_f64]);
        let direction = arr1(&[1.0_f64]);
        let alpha = ndarray_optimization::backtracking_line_search(
            &x,
            &direction,
            objective,
            gradient,
            &LineSearchConfig::default(),
        )
        .unwrap();
        assert!(alpha > 0.0);
    }

    #[test]
    fn gradient_descent_converges_on_quadratic() {
        let x0 = arr1(&[0.0_f64]);
        let solution =
            ndarray_optimization::gradient_descent(&x0, objective, gradient, &SGDConfig::default())
                .unwrap();
        assert!((solution[0] - 3.0).abs() < 1e-4);
    }

    #[test]
    fn adam_converges_on_quadratic() {
        let x0 = arr1(&[-5.0_f64]);
        let solution =
            ndarray_optimization::adam(&x0, objective, gradient, &AdamConfig::default()).unwrap();
        assert!((solution[0] - 3.0).abs() < 1e-3);
    }
}
