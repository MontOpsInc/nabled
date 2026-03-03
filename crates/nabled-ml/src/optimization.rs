//! Optimization primitives over ndarray vectors.

use std::fmt;

use ndarray::{Array1, Array2};
use num_complex::Complex64;

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

/// Configuration for momentum gradient descent.
#[derive(Debug, Clone, Copy)]
pub struct MomentumConfig {
    /// Base learning rate.
    pub learning_rate:  f64,
    /// Momentum coefficient in `[0, 1)`.
    pub momentum:       f64,
    /// Maximum optimization iterations.
    pub max_iterations: usize,
    /// Gradient norm tolerance for convergence.
    pub tolerance:      f64,
}

impl Default for MomentumConfig {
    fn default() -> Self {
        Self {
            learning_rate:  1e-2,
            momentum:       0.9,
            max_iterations: 10_000,
            tolerance:      1e-8,
        }
    }
}

/// Configuration for `RMSProp` optimizer.
#[derive(Debug, Clone, Copy)]
pub struct RMSPropConfig {
    /// Base learning rate.
    pub learning_rate:  f64,
    /// Exponential decay factor for squared gradients in `[0, 1)`.
    pub rho:            f64,
    /// Numerical epsilon.
    pub epsilon:        f64,
    /// Maximum optimization iterations.
    pub max_iterations: usize,
    /// Gradient norm tolerance for convergence.
    pub tolerance:      f64,
}

impl Default for RMSPropConfig {
    fn default() -> Self {
        Self {
            learning_rate:  1e-2,
            rho:            0.9,
            epsilon:        1e-8,
            max_iterations: 10_000,
            tolerance:      1e-8,
        }
    }
}

/// Configuration for projected gradient descent with box constraints.
#[derive(Debug, Clone, Copy)]
pub struct ProjectedGradientConfig {
    /// Base learning rate.
    pub learning_rate:  f64,
    /// Maximum optimization iterations.
    pub max_iterations: usize,
    /// Gradient norm tolerance for convergence.
    pub tolerance:      f64,
}

impl Default for ProjectedGradientConfig {
    fn default() -> Self { Self { learning_rate: 1e-2, max_iterations: 10_000, tolerance: 1e-8 } }
}

/// Configuration for `BFGS` quasi-Newton optimization.
#[derive(Debug, Clone, Copy)]
pub struct BFGSConfig {
    /// Initial step size multiplier.
    pub step_size:           f64,
    /// Maximum optimization iterations.
    pub max_iterations:      usize,
    /// Gradient norm tolerance for convergence.
    pub tolerance:           f64,
    /// Minimum curvature `s^T y` required for Hessian updates.
    pub curvature_tolerance: f64,
}

impl Default for BFGSConfig {
    fn default() -> Self {
        Self {
            step_size:           1.0,
            max_iterations:      2_000,
            tolerance:           1e-8,
            curvature_tolerance: 1e-12,
        }
    }
}

fn l2_norm(vector: &Array1<f64>) -> f64 {
    vector.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn l2_norm_complex(vector: &Array1<Complex64>) -> f64 {
    vector.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt()
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

fn validate_vector_complex(vector: &Array1<Complex64>) -> Result<(), OptimizationError> {
    if vector.is_empty() {
        return Err(OptimizationError::EmptyInput);
    }
    if vector.iter().any(|value| !value.re.is_finite() || !value.im.is_finite()) {
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

fn validate_momentum_config(config: &MomentumConfig) -> Result<(), OptimizationError> {
    if config.learning_rate <= 0.0
        || !(0.0..1.0).contains(&config.momentum)
        || config.max_iterations == 0
        || config.tolerance < 0.0
    {
        return Err(OptimizationError::InvalidConfig);
    }
    Ok(())
}

fn validate_rmsprop_config(config: &RMSPropConfig) -> Result<(), OptimizationError> {
    if config.learning_rate <= 0.0
        || !(0.0..1.0).contains(&config.rho)
        || config.epsilon <= 0.0
        || config.max_iterations == 0
        || config.tolerance < 0.0
    {
        return Err(OptimizationError::InvalidConfig);
    }
    Ok(())
}

fn validate_projected_gradient_config(
    config: &ProjectedGradientConfig,
) -> Result<(), OptimizationError> {
    if config.learning_rate <= 0.0 || config.max_iterations == 0 || config.tolerance < 0.0 {
        return Err(OptimizationError::InvalidConfig);
    }
    Ok(())
}

fn validate_bfgs_config(config: &BFGSConfig) -> Result<(), OptimizationError> {
    if config.step_size <= 0.0
        || config.max_iterations == 0
        || config.tolerance < 0.0
        || config.curvature_tolerance <= 0.0
    {
        return Err(OptimizationError::InvalidConfig);
    }
    Ok(())
}

fn validate_bounds(
    initial: &Array1<f64>,
    lower_bounds: &Array1<f64>,
    upper_bounds: &Array1<f64>,
) -> Result<(), OptimizationError> {
    if initial.len() != lower_bounds.len() || initial.len() != upper_bounds.len() {
        return Err(OptimizationError::DimensionMismatch);
    }
    for i in 0..initial.len() {
        if !lower_bounds[i].is_finite()
            || !upper_bounds[i].is_finite()
            || lower_bounds[i] > upper_bounds[i]
        {
            return Err(OptimizationError::InvalidConfig);
        }
    }
    Ok(())
}

fn validate_bounds_complex(
    initial: &Array1<Complex64>,
    lower_bounds: &Array1<Complex64>,
    upper_bounds: &Array1<Complex64>,
) -> Result<(), OptimizationError> {
    if initial.len() != lower_bounds.len() || initial.len() != upper_bounds.len() {
        return Err(OptimizationError::DimensionMismatch);
    }
    for i in 0..initial.len() {
        let lower = lower_bounds[i];
        let upper = upper_bounds[i];
        if !lower.re.is_finite()
            || !lower.im.is_finite()
            || !upper.re.is_finite()
            || !upper.im.is_finite()
            || lower.re > upper.re
            || lower.im > upper.im
        {
            return Err(OptimizationError::InvalidConfig);
        }
    }
    Ok(())
}

fn project_to_bounds(
    point: &mut Array1<f64>,
    lower_bounds: &Array1<f64>,
    upper_bounds: &Array1<f64>,
) {
    for i in 0..point.len() {
        point[i] = point[i].clamp(lower_bounds[i], upper_bounds[i]);
    }
}

fn project_to_bounds_complex(
    point: &mut Array1<Complex64>,
    lower_bounds: &Array1<Complex64>,
    upper_bounds: &Array1<Complex64>,
) {
    for i in 0..point.len() {
        let real = point[i].re.clamp(lower_bounds[i].re, upper_bounds[i].re);
        let imag = point[i].im.clamp(lower_bounds[i].im, upper_bounds[i].im);
        point[i] = Complex64::new(real, imag);
    }
}

fn outer_product(left: &Array1<f64>, right: &Array1<f64>) -> Array2<f64> {
    let mut output = Array2::<f64>::zeros((left.len(), right.len()));
    for row in 0..left.len() {
        for col in 0..right.len() {
            output[[row, col]] = left[row] * right[col];
        }
    }
    output
}

fn outer_product_complex(left: &Array1<Complex64>, right: &Array1<Complex64>) -> Array2<Complex64> {
    let mut output = Array2::<Complex64>::zeros((left.len(), right.len()));
    for row in 0..left.len() {
        for col in 0..right.len() {
            output[[row, col]] = left[row] * right[col].conj();
        }
    }
    output
}

fn hermitian_transpose(matrix: &Array2<Complex64>) -> Array2<Complex64> {
    matrix.t().mapv(|value| value.conj())
}

fn hermitian_dot(left: &Array1<Complex64>, right: &Array1<Complex64>) -> Complex64 {
    left.iter().zip(right.iter()).map(|(lhs, rhs)| lhs.conj() * rhs).sum()
}

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
        let candidate = point + &direction.mapv(|value| value * alpha);
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
        x = &x - &grad.mapv(|value| value * config.learning_rate);
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

/// Minimize objective with momentum gradient descent.
///
/// # Errors
/// Returns an error for invalid inputs/configuration or non-finite gradients.
pub fn momentum_descent<F, G>(
    initial: &Array1<f64>,
    objective: F,
    gradient: G,
    config: &MomentumConfig,
) -> Result<Array1<f64>, OptimizationError>
where
    F: Fn(&Array1<f64>) -> f64,
    G: Fn(&Array1<f64>) -> Array1<f64>,
{
    validate_vector(initial)?;
    validate_momentum_config(config)?;

    let mut x = initial.clone();
    let mut velocity = Array1::<f64>::zeros(x.len());
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

        for i in 0..x.len() {
            velocity[i] = config.momentum * velocity[i] + grad[i];
            x[i] -= config.learning_rate * velocity[i];
        }
    }

    Err(OptimizationError::MaxIterationsExceeded)
}

/// Minimize objective with `RMSProp`.
///
/// # Errors
/// Returns an error for invalid inputs/configuration or non-finite gradients.
pub fn rmsprop<F, G>(
    initial: &Array1<f64>,
    objective: F,
    gradient: G,
    config: &RMSPropConfig,
) -> Result<Array1<f64>, OptimizationError>
where
    F: Fn(&Array1<f64>) -> f64,
    G: Fn(&Array1<f64>) -> Array1<f64>,
{
    validate_vector(initial)?;
    validate_rmsprop_config(config)?;

    let mut x = initial.clone();
    let mut avg_sq = Array1::<f64>::zeros(x.len());
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

        for i in 0..x.len() {
            avg_sq[i] = config.rho * avg_sq[i] + (1.0 - config.rho) * grad[i] * grad[i];
            x[i] -= config.learning_rate * grad[i] / (avg_sq[i].sqrt() + config.epsilon);
        }
    }

    Err(OptimizationError::MaxIterationsExceeded)
}

/// Minimize objective with projected gradient descent under box constraints.
///
/// # Errors
/// Returns an error for invalid inputs/configuration, invalid bounds, or non-finite gradients.
pub fn projected_gradient_descent_box<F, G>(
    initial: &Array1<f64>,
    objective: F,
    gradient: G,
    lower_bounds: &Array1<f64>,
    upper_bounds: &Array1<f64>,
    config: &ProjectedGradientConfig,
) -> Result<Array1<f64>, OptimizationError>
where
    F: Fn(&Array1<f64>) -> f64,
    G: Fn(&Array1<f64>) -> Array1<f64>,
{
    validate_vector(initial)?;
    validate_vector(lower_bounds)?;
    validate_vector(upper_bounds)?;
    validate_projected_gradient_config(config)?;
    validate_bounds(initial, lower_bounds, upper_bounds)?;

    let mut x = initial.clone();
    project_to_bounds(&mut x, lower_bounds, upper_bounds);
    let _ = objective(&x);
    let tolerance = config.tolerance.max(DEFAULT_TOLERANCE);

    for _ in 0..config.max_iterations {
        let grad = gradient(&x);
        if grad.len() != x.len() || grad.iter().any(|value| !value.is_finite()) {
            return Err(OptimizationError::NonFiniteInput);
        }
        let previous = x.clone();
        x = &x - &grad.mapv(|value| value * config.learning_rate);
        project_to_bounds(&mut x, lower_bounds, upper_bounds);
        let step_norm = l2_norm(&(&x - &previous));
        if step_norm <= tolerance || l2_norm(&grad) <= tolerance {
            return Ok(x);
        }
    }

    Err(OptimizationError::MaxIterationsExceeded)
}

/// Minimize objective with stochastic gradient descent.
///
/// `stochastic_gradient` receives `(current_point, iteration)` and returns a gradient sample.
///
/// # Errors
/// Returns an error for invalid inputs/configuration or non-finite gradients.
pub fn stochastic_gradient_descent<G>(
    initial: &Array1<f64>,
    stochastic_gradient: G,
    config: &SGDConfig,
) -> Result<Array1<f64>, OptimizationError>
where
    G: Fn(&Array1<f64>, usize) -> Array1<f64>,
{
    validate_vector(initial)?;
    validate_sgd_config(config)?;

    let mut x = initial.clone();
    let tolerance = config.tolerance.max(DEFAULT_TOLERANCE);
    for iteration in 0..config.max_iterations {
        let grad = stochastic_gradient(&x, iteration);
        if grad.len() != x.len() || grad.iter().any(|value| !value.is_finite()) {
            return Err(OptimizationError::NonFiniteInput);
        }
        if l2_norm(&grad) <= tolerance {
            return Ok(x);
        }
        x = &x - &grad.mapv(|value| value * config.learning_rate);
    }
    Err(OptimizationError::MaxIterationsExceeded)
}

/// Minimize objective with `BFGS` quasi-Newton updates.
///
/// # Errors
/// Returns an error for invalid inputs/configuration or non-finite gradients.
pub fn bfgs<F, G>(
    initial: &Array1<f64>,
    objective: F,
    gradient: G,
    config: &BFGSConfig,
) -> Result<Array1<f64>, OptimizationError>
where
    F: Fn(&Array1<f64>) -> f64,
    G: Fn(&Array1<f64>) -> Array1<f64>,
{
    validate_vector(initial)?;
    validate_bfgs_config(config)?;

    let dimension = initial.len();
    let mut x = initial.clone();
    let mut h_inv = Array2::<f64>::eye(dimension);
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

        let direction = -h_inv.dot(&grad);
        let step = config.step_size;
        let x_next = &x + &direction.mapv(|value| value * step);
        let grad_next = gradient(&x_next);
        if grad_next.len() != x.len() || grad_next.iter().any(|value| !value.is_finite()) {
            return Err(OptimizationError::NonFiniteInput);
        }

        let s = &x_next - &x;
        let y = &grad_next - &grad;
        let ys = y.dot(&s);
        if ys.abs() > config.curvature_tolerance {
            let rho = 1.0 / ys;
            let identity = Array2::<f64>::eye(dimension);
            let sy = outer_product(&s, &y);
            let ys_outer = outer_product(&y, &s);
            let ss = outer_product(&s, &s);

            let left = &identity - &(rho * sy);
            let right = &identity - &(rho * ys_outer);
            h_inv = left.dot(&h_inv).dot(&right) + rho * ss;
        }

        x = x_next;
    }

    Err(OptimizationError::MaxIterationsExceeded)
}

/// Perform Armijo backtracking line search for complex vectors.
///
/// # Errors
/// Returns an error for invalid inputs/configuration or non-finite objective evaluations.
pub fn backtracking_line_search_complex<F, G>(
    point: &Array1<Complex64>,
    direction: &Array1<Complex64>,
    objective: F,
    gradient: G,
    config: &LineSearchConfig,
) -> Result<f64, OptimizationError>
where
    F: Fn(&Array1<Complex64>) -> f64,
    G: Fn(&Array1<Complex64>) -> Array1<Complex64>,
{
    validate_vector_complex(point)?;
    validate_vector_complex(direction)?;
    if point.len() != direction.len() {
        return Err(OptimizationError::DimensionMismatch);
    }
    validate_line_search_config(config)?;

    let grad = gradient(point);
    if grad.len() != point.len()
        || grad.iter().any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(OptimizationError::NonFiniteInput);
    }

    let fx = objective(point);
    if !fx.is_finite() {
        return Err(OptimizationError::NonFiniteInput);
    }
    let directional_derivative = hermitian_dot(&grad, direction).re;

    let mut alpha = config.initial_step;
    for _ in 0..config.max_iterations {
        let candidate = point + &direction.mapv(|value| value * alpha);
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

/// Minimize objective with fixed-step complex gradient descent.
///
/// # Errors
/// Returns an error for invalid inputs/configuration or non-finite gradients.
pub fn gradient_descent_complex<F, G>(
    initial: &Array1<Complex64>,
    objective: F,
    gradient: G,
    config: &SGDConfig,
) -> Result<Array1<Complex64>, OptimizationError>
where
    F: Fn(&Array1<Complex64>) -> f64,
    G: Fn(&Array1<Complex64>) -> Array1<Complex64>,
{
    validate_vector_complex(initial)?;
    validate_sgd_config(config)?;

    let mut x = initial.clone();
    let _ = objective(&x);
    let tolerance = config.tolerance.max(DEFAULT_TOLERANCE);

    for _ in 0..config.max_iterations {
        let grad = gradient(&x);
        if grad.len() != x.len()
            || grad.iter().any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(OptimizationError::NonFiniteInput);
        }
        if l2_norm_complex(&grad) <= tolerance {
            return Ok(x);
        }
        x = &x - &grad.mapv(|value| value * config.learning_rate);
    }

    Err(OptimizationError::MaxIterationsExceeded)
}

/// Minimize objective with complex Adam updates.
///
/// # Errors
/// Returns an error for invalid inputs/configuration or non-finite gradients.
pub fn adam_complex<F, G>(
    initial: &Array1<Complex64>,
    objective: F,
    gradient: G,
    config: &AdamConfig,
) -> Result<Array1<Complex64>, OptimizationError>
where
    F: Fn(&Array1<Complex64>) -> f64,
    G: Fn(&Array1<Complex64>) -> Array1<Complex64>,
{
    validate_vector_complex(initial)?;
    validate_adam_config(config)?;

    let mut x = initial.clone();
    let mut m = Array1::<Complex64>::zeros(x.len());
    let mut v = Array1::<f64>::zeros(x.len());
    let mut beta1_power = 1.0_f64;
    let mut beta2_power = 1.0_f64;
    let tolerance = config.tolerance.max(DEFAULT_TOLERANCE);

    let _ = objective(&x);
    for _ in 0..config.max_iterations {
        let grad = gradient(&x);
        if grad.len() != x.len()
            || grad.iter().any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(OptimizationError::NonFiniteInput);
        }
        if l2_norm_complex(&grad) <= tolerance {
            return Ok(x);
        }

        beta1_power *= config.beta1;
        beta2_power *= config.beta2;

        for i in 0..x.len() {
            m[i] = config.beta1 * m[i] + (1.0 - config.beta1) * grad[i];
            v[i] = config.beta2 * v[i] + (1.0 - config.beta2) * grad[i].norm_sqr();

            let m_hat = m[i] / (1.0 - beta1_power);
            let v_hat = v[i] / (1.0 - beta2_power);
            x[i] -= config.learning_rate * m_hat / (v_hat.sqrt() + config.epsilon);
        }
    }

    Err(OptimizationError::MaxIterationsExceeded)
}

/// Minimize objective with complex momentum gradient descent.
///
/// # Errors
/// Returns an error for invalid inputs/configuration or non-finite gradients.
pub fn momentum_descent_complex<F, G>(
    initial: &Array1<Complex64>,
    objective: F,
    gradient: G,
    config: &MomentumConfig,
) -> Result<Array1<Complex64>, OptimizationError>
where
    F: Fn(&Array1<Complex64>) -> f64,
    G: Fn(&Array1<Complex64>) -> Array1<Complex64>,
{
    validate_vector_complex(initial)?;
    validate_momentum_config(config)?;

    let mut x = initial.clone();
    let mut velocity = Array1::<Complex64>::zeros(x.len());
    let tolerance = config.tolerance.max(DEFAULT_TOLERANCE);

    let _ = objective(&x);
    for _ in 0..config.max_iterations {
        let grad = gradient(&x);
        if grad.len() != x.len()
            || grad.iter().any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(OptimizationError::NonFiniteInput);
        }
        if l2_norm_complex(&grad) <= tolerance {
            return Ok(x);
        }

        for i in 0..x.len() {
            velocity[i] = config.momentum * velocity[i] + grad[i];
            x[i] -= config.learning_rate * velocity[i];
        }
    }

    Err(OptimizationError::MaxIterationsExceeded)
}

/// Minimize objective with complex `RMSProp`.
///
/// # Errors
/// Returns an error for invalid inputs/configuration or non-finite gradients.
pub fn rmsprop_complex<F, G>(
    initial: &Array1<Complex64>,
    objective: F,
    gradient: G,
    config: &RMSPropConfig,
) -> Result<Array1<Complex64>, OptimizationError>
where
    F: Fn(&Array1<Complex64>) -> f64,
    G: Fn(&Array1<Complex64>) -> Array1<Complex64>,
{
    validate_vector_complex(initial)?;
    validate_rmsprop_config(config)?;

    let mut x = initial.clone();
    let mut avg_sq = Array1::<f64>::zeros(x.len());
    let tolerance = config.tolerance.max(DEFAULT_TOLERANCE);

    let _ = objective(&x);
    for _ in 0..config.max_iterations {
        let grad = gradient(&x);
        if grad.len() != x.len()
            || grad.iter().any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(OptimizationError::NonFiniteInput);
        }
        if l2_norm_complex(&grad) <= tolerance {
            return Ok(x);
        }

        for i in 0..x.len() {
            avg_sq[i] = config.rho * avg_sq[i] + (1.0 - config.rho) * grad[i].norm_sqr();
            x[i] -= config.learning_rate * grad[i] / (avg_sq[i].sqrt() + config.epsilon);
        }
    }

    Err(OptimizationError::MaxIterationsExceeded)
}

/// Minimize objective with complex projected gradient descent under box constraints.
///
/// Bounds clamp real and imaginary components independently.
///
/// # Errors
/// Returns an error for invalid inputs/configuration, invalid bounds, or non-finite gradients.
pub fn projected_gradient_descent_box_complex<F, G>(
    initial: &Array1<Complex64>,
    objective: F,
    gradient: G,
    lower_bounds: &Array1<Complex64>,
    upper_bounds: &Array1<Complex64>,
    config: &ProjectedGradientConfig,
) -> Result<Array1<Complex64>, OptimizationError>
where
    F: Fn(&Array1<Complex64>) -> f64,
    G: Fn(&Array1<Complex64>) -> Array1<Complex64>,
{
    validate_vector_complex(initial)?;
    validate_vector_complex(lower_bounds)?;
    validate_vector_complex(upper_bounds)?;
    validate_projected_gradient_config(config)?;
    validate_bounds_complex(initial, lower_bounds, upper_bounds)?;

    let mut x = initial.clone();
    project_to_bounds_complex(&mut x, lower_bounds, upper_bounds);
    let _ = objective(&x);
    let tolerance = config.tolerance.max(DEFAULT_TOLERANCE);

    for _ in 0..config.max_iterations {
        let grad = gradient(&x);
        if grad.len() != x.len()
            || grad.iter().any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(OptimizationError::NonFiniteInput);
        }
        let previous = x.clone();
        x = &x - &grad.mapv(|value| value * config.learning_rate);
        project_to_bounds_complex(&mut x, lower_bounds, upper_bounds);
        let step_norm = l2_norm_complex(&(&x - &previous));
        if step_norm <= tolerance || l2_norm_complex(&grad) <= tolerance {
            return Ok(x);
        }
    }

    Err(OptimizationError::MaxIterationsExceeded)
}

/// Minimize objective with complex stochastic gradient descent.
///
/// `stochastic_gradient` receives `(current_point, iteration)` and returns a gradient sample.
///
/// # Errors
/// Returns an error for invalid inputs/configuration or non-finite gradients.
pub fn stochastic_gradient_descent_complex<G>(
    initial: &Array1<Complex64>,
    stochastic_gradient: G,
    config: &SGDConfig,
) -> Result<Array1<Complex64>, OptimizationError>
where
    G: Fn(&Array1<Complex64>, usize) -> Array1<Complex64>,
{
    validate_vector_complex(initial)?;
    validate_sgd_config(config)?;

    let mut x = initial.clone();
    let tolerance = config.tolerance.max(DEFAULT_TOLERANCE);
    for iteration in 0..config.max_iterations {
        let grad = stochastic_gradient(&x, iteration);
        if grad.len() != x.len()
            || grad.iter().any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(OptimizationError::NonFiniteInput);
        }
        if l2_norm_complex(&grad) <= tolerance {
            return Ok(x);
        }
        x = &x - &grad.mapv(|value| value * config.learning_rate);
    }
    Err(OptimizationError::MaxIterationsExceeded)
}

/// Minimize objective with complex `BFGS` quasi-Newton updates.
///
/// # Errors
/// Returns an error for invalid inputs/configuration or non-finite gradients.
pub fn bfgs_complex<F, G>(
    initial: &Array1<Complex64>,
    objective: F,
    gradient: G,
    config: &BFGSConfig,
) -> Result<Array1<Complex64>, OptimizationError>
where
    F: Fn(&Array1<Complex64>) -> f64,
    G: Fn(&Array1<Complex64>) -> Array1<Complex64>,
{
    validate_vector_complex(initial)?;
    validate_bfgs_config(config)?;

    let dimension = initial.len();
    let mut x = initial.clone();
    let mut h_inv = Array2::<Complex64>::eye(dimension);
    let tolerance = config.tolerance.max(DEFAULT_TOLERANCE);

    let _ = objective(&x);
    for _ in 0..config.max_iterations {
        let grad = gradient(&x);
        if grad.len() != x.len()
            || grad.iter().any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(OptimizationError::NonFiniteInput);
        }
        if l2_norm_complex(&grad) <= tolerance {
            return Ok(x);
        }

        let direction = -h_inv.dot(&grad);
        let step = config.step_size;
        let x_next = &x + &direction.mapv(|value| value * step);
        let grad_next = gradient(&x_next);
        if grad_next.len() != x.len()
            || grad_next.iter().any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(OptimizationError::NonFiniteInput);
        }

        let s = &x_next - &x;
        let y = &grad_next - &grad;
        let curvature = hermitian_dot(&s, &y).re;
        if curvature > config.curvature_tolerance {
            let rho = 1.0 / curvature;
            let identity = Array2::<Complex64>::eye(dimension);
            let sy = outer_product_complex(&s, &y);
            let ys_outer = outer_product_complex(&y, &s);
            let ss = outer_product_complex(&s, &s);

            let left = &identity - &sy.mapv(|value| value * rho);
            let right = &identity - &ys_outer.mapv(|value| value * rho);
            h_inv =
                left.dot(&h_inv).dot(&hermitian_transpose(&right)) + ss.mapv(|value| value * rho);
        }

        x = x_next;
    }

    Err(OptimizationError::MaxIterationsExceeded)
}

#[cfg(test)]
mod tests {
    use ndarray::arr1;
    use num_complex::Complex64;

    use super::*;

    fn objective(x: &Array1<f64>) -> f64 {
        let delta = x[0] - 3.0;
        delta * delta
    }

    fn gradient(x: &Array1<f64>) -> Array1<f64> { arr1(&[2.0 * (x[0] - 3.0)]) }

    fn objective_complex(x: &Array1<Complex64>) -> f64 {
        let delta = x[0] - Complex64::new(3.0, 2.0);
        delta.norm_sqr()
    }

    fn gradient_complex(x: &Array1<Complex64>) -> Array1<Complex64> {
        arr1(&[Complex64::new(2.0, 0.0) * (x[0] - Complex64::new(3.0, 2.0))])
    }

    #[test]
    fn backtracking_line_search_finds_descent_step() {
        let x = arr1(&[0.0_f64]);
        let direction = arr1(&[1.0_f64]);
        let alpha = backtracking_line_search(
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
        let solution = gradient_descent(&x0, objective, gradient, &SGDConfig::default()).unwrap();
        assert!((solution[0] - 3.0).abs() < 1e-4);
    }

    #[test]
    fn adam_converges_on_quadratic() {
        let x0 = arr1(&[-5.0_f64]);
        let solution = adam(&x0, objective, gradient, &AdamConfig::default()).unwrap();
        assert!((solution[0] - 3.0).abs() < 1e-3);
    }

    #[test]
    fn line_search_rejects_invalid_config_and_dimension_mismatch() {
        let x = arr1(&[0.0_f64]);
        let direction = arr1(&[1.0_f64, 2.0_f64]);
        let result = backtracking_line_search(
            &x,
            &direction,
            objective,
            gradient,
            &LineSearchConfig::default(),
        );
        assert!(matches!(result, Err(OptimizationError::DimensionMismatch)));

        let bad_config = LineSearchConfig { contraction: 1.0, ..LineSearchConfig::default() };
        let result =
            backtracking_line_search(&x, &arr1(&[1.0_f64]), objective, gradient, &bad_config);
        assert!(matches!(result, Err(OptimizationError::InvalidConfig)));
    }

    #[test]
    fn gradient_descent_and_adam_cover_error_paths() {
        let x0 = arr1(&[0.0_f64]);

        let bad_gradient = |_x: &Array1<f64>| arr1(&[f64::NAN]);
        let gd_non_finite = gradient_descent(&x0, objective, bad_gradient, &SGDConfig::default());
        assert!(matches!(gd_non_finite, Err(OptimizationError::NonFiniteInput)));

        let gd_stall = gradient_descent(&x0, objective, gradient, &SGDConfig {
            learning_rate:  1e-12,
            max_iterations: 1,
            tolerance:      0.0,
        });
        assert!(matches!(gd_stall, Err(OptimizationError::MaxIterationsExceeded)));

        let bad_adam = AdamConfig { beta1: 1.0, ..AdamConfig::default() };
        let adam_invalid = adam(&x0, objective, gradient, &bad_adam);
        assert!(matches!(adam_invalid, Err(OptimizationError::InvalidConfig)));
    }

    #[test]
    fn momentum_and_rmsprop_converge_on_quadratic() {
        let x0 = arr1(&[-4.0_f64]);

        let momentum_solution =
            momentum_descent(&x0, objective, gradient, &MomentumConfig::default()).unwrap();
        assert!((momentum_solution[0] - 3.0).abs() < 1e-3);

        let rmsprop_solution =
            rmsprop(&x0, objective, gradient, &RMSPropConfig::default()).unwrap();
        assert!((rmsprop_solution[0] - 3.0).abs() < 1e-3);
    }

    #[test]
    fn momentum_and_rmsprop_reject_invalid_config() {
        let x0 = arr1(&[0.0_f64]);

        let bad_momentum = MomentumConfig { momentum: 1.0, ..MomentumConfig::default() };
        let momentum_invalid = momentum_descent(&x0, objective, gradient, &bad_momentum);
        assert!(matches!(momentum_invalid, Err(OptimizationError::InvalidConfig)));

        let bad_rmsprop = RMSPropConfig { rho: 1.0, ..RMSPropConfig::default() };
        let rmsprop_invalid = rmsprop(&x0, objective, gradient, &bad_rmsprop);
        assert!(matches!(rmsprop_invalid, Err(OptimizationError::InvalidConfig)));
    }

    #[test]
    fn projected_gradient_descent_respects_bounds() {
        let x0 = arr1(&[-10.0_f64]);
        let lower = arr1(&[0.0_f64]);
        let upper = arr1(&[2.5_f64]);
        let solution = projected_gradient_descent_box(
            &x0,
            objective,
            gradient,
            &lower,
            &upper,
            &ProjectedGradientConfig::default(),
        )
        .unwrap();
        assert!((solution[0] - 2.5).abs() < 1e-8);
    }

    #[test]
    fn stochastic_gradient_descent_converges_on_quadratic() {
        let x0 = arr1(&[-3.0_f64]);
        let solution = stochastic_gradient_descent(
            &x0,
            |x, _iteration| arr1(&[2.0 * (x[0] - 3.0)]),
            &SGDConfig { learning_rate: 5e-2, max_iterations: 2_000, tolerance: 1e-6 },
        )
        .unwrap();
        assert!((solution[0] - 3.0).abs() < 1e-3);
    }

    #[test]
    fn bfgs_converges_on_quadratic() {
        let x0 = arr1(&[-8.0_f64]);
        let solution = bfgs(&x0, objective, gradient, &BFGSConfig::default()).unwrap();
        assert!((solution[0] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn advanced_optimizers_reject_invalid_inputs() {
        let x0 = arr1(&[0.0_f64]);
        let lower = arr1(&[2.0_f64]);
        let upper = arr1(&[1.0_f64]);
        let projected = projected_gradient_descent_box(
            &x0,
            objective,
            gradient,
            &lower,
            &upper,
            &ProjectedGradientConfig::default(),
        );
        assert!(matches!(projected, Err(OptimizationError::InvalidConfig)));

        let sgd_non_finite =
            stochastic_gradient_descent(&x0, |_x, _| arr1(&[f64::NAN]), &SGDConfig::default());
        assert!(matches!(sgd_non_finite, Err(OptimizationError::NonFiniteInput)));

        let bfgs_invalid =
            bfgs(&x0, objective, gradient, &BFGSConfig { step_size: 0.0, ..BFGSConfig::default() });
        assert!(matches!(bfgs_invalid, Err(OptimizationError::InvalidConfig)));
    }

    #[test]
    fn complex_line_search_and_optimizers_converge() {
        let x = arr1(&[Complex64::new(0.0, 0.0)]);
        let direction = arr1(&[Complex64::new(1.0, 1.0)]);
        let alpha = backtracking_line_search_complex(
            &x,
            &direction,
            objective_complex,
            gradient_complex,
            &LineSearchConfig::default(),
        )
        .unwrap();
        assert!(alpha > 0.0);

        let gd = gradient_descent_complex(
            &x,
            objective_complex,
            gradient_complex,
            &SGDConfig::default(),
        )
        .unwrap();
        assert!((gd[0] - Complex64::new(3.0, 2.0)).norm() < 1e-3);

        let adam_solution =
            adam_complex(&x, objective_complex, gradient_complex, &AdamConfig::default()).unwrap();
        assert!((adam_solution[0] - Complex64::new(3.0, 2.0)).norm() < 1e-3);

        let momentum_solution = momentum_descent_complex(
            &x,
            objective_complex,
            gradient_complex,
            &MomentumConfig::default(),
        )
        .unwrap();
        assert!((momentum_solution[0] - Complex64::new(3.0, 2.0)).norm() < 1e-3);

        let rmsprop_solution =
            rmsprop_complex(&x, objective_complex, gradient_complex, &RMSPropConfig::default())
                .unwrap();
        assert!((rmsprop_solution[0] - Complex64::new(3.0, 2.0)).norm() < 1e-3);

        let bfgs_solution =
            bfgs_complex(&x, objective_complex, gradient_complex, &BFGSConfig::default()).unwrap();
        assert!((bfgs_solution[0] - Complex64::new(3.0, 2.0)).norm() < 1e-6);
    }

    #[test]
    fn complex_projected_and_stochastic_paths_work() {
        let x = arr1(&[Complex64::new(-5.0, -5.0)]);
        let lower = arr1(&[Complex64::new(0.0, 0.0)]);
        let upper = arr1(&[Complex64::new(2.5, 2.5)]);
        let projected = projected_gradient_descent_box_complex(
            &x,
            objective_complex,
            gradient_complex,
            &lower,
            &upper,
            &ProjectedGradientConfig::default(),
        )
        .unwrap();
        assert!((projected[0] - Complex64::new(2.5, 2.0)).norm() < 1e-3);

        let stochastic = stochastic_gradient_descent_complex(
            &arr1(&[Complex64::new(-3.0, -1.0)]),
            |point, _iteration| {
                arr1(&[Complex64::new(2.0, 0.0) * (point[0] - Complex64::new(3.0, 2.0))])
            },
            &SGDConfig { learning_rate: 5e-2, max_iterations: 2_000, tolerance: 1e-6 },
        )
        .unwrap();
        assert!((stochastic[0] - Complex64::new(3.0, 2.0)).norm() < 1e-3);
    }
}
