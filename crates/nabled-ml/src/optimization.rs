//! Optimization primitives over ndarray vectors.

use std::fmt;

use nabled_core::scalar::NabledReal;
use ndarray::{Array1, Array2, ArrayBase, Data, Ix1};
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
pub struct LineSearchConfig<T: NabledReal = f64> {
    /// Initial step size.
    pub initial_step:        T,
    /// Contraction factor in `(0, 1)`.
    pub contraction:         T,
    /// Armijo sufficient decrease coefficient in `(0, 1)`.
    pub sufficient_decrease: T,
    /// Maximum backtracking iterations.
    pub max_iterations:      usize,
}

impl<T: NabledReal> Default for LineSearchConfig<T> {
    fn default() -> Self {
        Self {
            initial_step:        T::one(),
            contraction:         T::from_f64(0.5).unwrap_or(T::epsilon()),
            sufficient_decrease: T::from_f64(1e-4).unwrap_or(T::epsilon()),
            max_iterations:      64,
        }
    }
}

/// Configuration for SGD.
#[derive(Debug, Clone, Copy)]
pub struct SGDConfig<T: NabledReal = f64> {
    /// Fixed learning rate.
    pub learning_rate:  T,
    /// Maximum optimization iterations.
    pub max_iterations: usize,
    /// Gradient norm tolerance for convergence.
    pub tolerance:      T,
}

impl<T: NabledReal> Default for SGDConfig<T> {
    fn default() -> Self {
        Self {
            learning_rate:  T::from_f64(1e-2).unwrap_or(T::epsilon()),
            max_iterations: 10_000,
            tolerance:      T::from_f64(1e-8).unwrap_or(T::epsilon()),
        }
    }
}

/// Configuration for Adam optimizer.
#[derive(Debug, Clone, Copy)]
pub struct AdamConfig<T: NabledReal = f64> {
    /// Base learning rate.
    pub learning_rate:  T,
    /// Exponential decay for first moment.
    pub beta1:          T,
    /// Exponential decay for second moment.
    pub beta2:          T,
    /// Numerical epsilon.
    pub epsilon:        T,
    /// Maximum optimization iterations.
    pub max_iterations: usize,
    /// Gradient norm tolerance for convergence.
    pub tolerance:      T,
}

impl<T: NabledReal> Default for AdamConfig<T> {
    fn default() -> Self {
        Self {
            learning_rate:  T::from_f64(1e-2).unwrap_or(T::epsilon()),
            beta1:          T::from_f64(0.9).unwrap_or(T::epsilon()),
            beta2:          T::from_f64(0.999).unwrap_or(T::epsilon()),
            epsilon:        T::from_f64(1e-8).unwrap_or(T::epsilon()),
            max_iterations: 10_000,
            tolerance:      T::from_f64(1e-8).unwrap_or(T::epsilon()),
        }
    }
}

/// Configuration for momentum gradient descent.
#[derive(Debug, Clone, Copy)]
pub struct MomentumConfig<T: NabledReal = f64> {
    /// Base learning rate.
    pub learning_rate:  T,
    /// Momentum coefficient in `[0, 1)`.
    pub momentum:       T,
    /// Maximum optimization iterations.
    pub max_iterations: usize,
    /// Gradient norm tolerance for convergence.
    pub tolerance:      T,
}

impl<T: NabledReal> Default for MomentumConfig<T> {
    fn default() -> Self {
        Self {
            learning_rate:  T::from_f64(1e-2).unwrap_or(T::epsilon()),
            momentum:       T::from_f64(0.9).unwrap_or(T::epsilon()),
            max_iterations: 10_000,
            tolerance:      T::from_f64(1e-8).unwrap_or(T::epsilon()),
        }
    }
}

/// Configuration for `RMSProp` optimizer.
#[derive(Debug, Clone, Copy)]
pub struct RMSPropConfig<T: NabledReal = f64> {
    /// Base learning rate.
    pub learning_rate:  T,
    /// Exponential decay factor for squared gradients in `[0, 1)`.
    pub rho:            T,
    /// Numerical epsilon.
    pub epsilon:        T,
    /// Maximum optimization iterations.
    pub max_iterations: usize,
    /// Gradient norm tolerance for convergence.
    pub tolerance:      T,
}

impl<T: NabledReal> Default for RMSPropConfig<T> {
    fn default() -> Self {
        Self {
            learning_rate:  T::from_f64(1e-2).unwrap_or(T::epsilon()),
            rho:            T::from_f64(0.9).unwrap_or(T::epsilon()),
            epsilon:        T::from_f64(1e-8).unwrap_or(T::epsilon()),
            max_iterations: 10_000,
            tolerance:      T::from_f64(1e-8).unwrap_or(T::epsilon()),
        }
    }
}

/// Configuration for projected gradient descent with box constraints.
#[derive(Debug, Clone, Copy)]
pub struct ProjectedGradientConfig<T: NabledReal = f64> {
    /// Base learning rate.
    pub learning_rate:  T,
    /// Maximum optimization iterations.
    pub max_iterations: usize,
    /// Gradient norm tolerance for convergence.
    pub tolerance:      T,
}

impl<T: NabledReal> Default for ProjectedGradientConfig<T> {
    fn default() -> Self {
        Self {
            learning_rate:  T::from_f64(1e-2).unwrap_or(T::epsilon()),
            max_iterations: 10_000,
            tolerance:      T::from_f64(1e-8).unwrap_or(T::epsilon()),
        }
    }
}

/// Configuration for `BFGS` quasi-Newton optimization.
#[derive(Debug, Clone, Copy)]
pub struct BFGSConfig<T: NabledReal = f64> {
    /// Initial step size multiplier.
    pub step_size:           T,
    /// Maximum optimization iterations.
    pub max_iterations:      usize,
    /// Gradient norm tolerance for convergence.
    pub tolerance:           T,
    /// Minimum curvature `s^T y` required for Hessian updates.
    pub curvature_tolerance: T,
}

impl<T: NabledReal> Default for BFGSConfig<T> {
    fn default() -> Self {
        Self {
            step_size:           T::one(),
            max_iterations:      2_000,
            tolerance:           T::from_f64(1e-8).unwrap_or(T::epsilon()),
            curvature_tolerance: T::from_f64(1e-12).unwrap_or(T::epsilon()),
        }
    }
}

fn l2_norm<T: NabledReal, S: Data<Elem = T>>(vector: &ArrayBase<S, Ix1>) -> T {
    vector
        .iter()
        .map(|value| *value * *value)
        .fold(T::zero(), |acc, value| acc + value)
        .sqrt()
}

fn l2_norm_complex<S>(vector: &ArrayBase<S, Ix1>) -> f64
where
    S: Data<Elem = Complex64>,
{
    vector.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt()
}

fn validate_vector<T: NabledReal, S: Data<Elem = T>>(
    vector: &ArrayBase<S, Ix1>,
) -> Result<(), OptimizationError> {
    if vector.is_empty() {
        return Err(OptimizationError::EmptyInput);
    }
    if vector.iter().any(|value| !value.is_finite()) {
        return Err(OptimizationError::NonFiniteInput);
    }
    Ok(())
}

fn validate_vector_complex<S>(vector: &ArrayBase<S, Ix1>) -> Result<(), OptimizationError>
where
    S: Data<Elem = Complex64>,
{
    if vector.is_empty() {
        return Err(OptimizationError::EmptyInput);
    }
    if vector.iter().any(|value| !value.re.is_finite() || !value.im.is_finite()) {
        return Err(OptimizationError::NonFiniteInput);
    }
    Ok(())
}

fn validate_line_search_config<T: NabledReal>(
    config: &LineSearchConfig<T>,
) -> Result<(), OptimizationError> {
    if config.initial_step <= T::zero()
        || config.contraction <= T::zero()
        || config.contraction >= T::one()
        || config.sufficient_decrease <= T::zero()
        || config.sufficient_decrease >= T::one()
        || config.max_iterations == 0
    {
        return Err(OptimizationError::InvalidConfig);
    }
    Ok(())
}

fn validate_sgd_config<T: NabledReal>(config: &SGDConfig<T>) -> Result<(), OptimizationError> {
    if config.learning_rate <= T::zero()
        || config.max_iterations == 0
        || config.tolerance < T::zero()
    {
        return Err(OptimizationError::InvalidConfig);
    }
    Ok(())
}

fn validate_adam_config<T: NabledReal>(config: &AdamConfig<T>) -> Result<(), OptimizationError> {
    if config.learning_rate <= T::zero()
        || config.beta1 <= T::zero()
        || config.beta1 >= T::one()
        || config.beta2 <= T::zero()
        || config.beta2 >= T::one()
        || config.epsilon <= T::zero()
        || config.max_iterations == 0
        || config.tolerance < T::zero()
    {
        return Err(OptimizationError::InvalidConfig);
    }
    Ok(())
}

fn validate_momentum_config<T: NabledReal>(
    config: &MomentumConfig<T>,
) -> Result<(), OptimizationError> {
    if config.learning_rate <= T::zero()
        || config.momentum < T::zero()
        || config.momentum >= T::one()
        || config.max_iterations == 0
        || config.tolerance < T::zero()
    {
        return Err(OptimizationError::InvalidConfig);
    }
    Ok(())
}

fn validate_rmsprop_config<T: NabledReal>(
    config: &RMSPropConfig<T>,
) -> Result<(), OptimizationError> {
    if config.learning_rate <= T::zero()
        || config.rho <= T::zero()
        || config.rho >= T::one()
        || config.epsilon <= T::zero()
        || config.max_iterations == 0
        || config.tolerance < T::zero()
    {
        return Err(OptimizationError::InvalidConfig);
    }
    Ok(())
}

fn validate_projected_gradient_config<T: NabledReal>(
    config: &ProjectedGradientConfig<T>,
) -> Result<(), OptimizationError> {
    if config.learning_rate <= T::zero()
        || config.max_iterations == 0
        || config.tolerance < T::zero()
    {
        return Err(OptimizationError::InvalidConfig);
    }
    Ok(())
}

fn validate_bfgs_config<T: NabledReal>(config: &BFGSConfig<T>) -> Result<(), OptimizationError> {
    if config.step_size <= T::zero()
        || config.max_iterations == 0
        || config.tolerance < T::zero()
        || config.curvature_tolerance <= T::zero()
    {
        return Err(OptimizationError::InvalidConfig);
    }
    Ok(())
}

fn validate_bounds<T: NabledReal, IS: Data<Elem = T>, LS: Data<Elem = T>, US: Data<Elem = T>>(
    initial: &ArrayBase<IS, Ix1>,
    lower_bounds: &ArrayBase<LS, Ix1>,
    upper_bounds: &ArrayBase<US, Ix1>,
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

fn validate_bounds_complex<IS, LS, US>(
    initial: &ArrayBase<IS, Ix1>,
    lower_bounds: &ArrayBase<LS, Ix1>,
    upper_bounds: &ArrayBase<US, Ix1>,
) -> Result<(), OptimizationError>
where
    IS: Data<Elem = Complex64>,
    LS: Data<Elem = Complex64>,
    US: Data<Elem = Complex64>,
{
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

fn project_to_bounds<T: NabledReal, LS: Data<Elem = T>, US: Data<Elem = T>>(
    point: &mut Array1<T>,
    lower_bounds: &ArrayBase<LS, Ix1>,
    upper_bounds: &ArrayBase<US, Ix1>,
) {
    for i in 0..point.len() {
        point[i] = if point[i] < lower_bounds[i] {
            lower_bounds[i]
        } else if point[i] > upper_bounds[i] {
            upper_bounds[i]
        } else {
            point[i]
        };
    }
}

fn project_to_bounds_complex(
    point: &mut Array1<Complex64>,
    lower_bounds: &ArrayBase<impl Data<Elem = Complex64>, Ix1>,
    upper_bounds: &ArrayBase<impl Data<Elem = Complex64>, Ix1>,
) {
    for i in 0..point.len() {
        let real = point[i].re.clamp(lower_bounds[i].re, upper_bounds[i].re);
        let imag = point[i].im.clamp(lower_bounds[i].im, upper_bounds[i].im);
        point[i] = Complex64::new(real, imag);
    }
}

fn outer_product<T: NabledReal>(left: &Array1<T>, right: &Array1<T>) -> Array2<T> {
    let mut output = Array2::<T>::zeros((left.len(), right.len()));
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

fn hermitian_dot<LS, RS>(left: &ArrayBase<LS, Ix1>, right: &ArrayBase<RS, Ix1>) -> Complex64
where
    LS: Data<Elem = Complex64>,
    RS: Data<Elem = Complex64>,
{
    left.iter().zip(right.iter()).map(|(lhs, rhs)| lhs.conj() * rhs).sum()
}

/// Perform Armijo backtracking line search.
///
/// # Errors
/// Returns an error for invalid inputs/configuration or non-finite objective evaluations.
pub fn backtracking_line_search<T, F, G, PS, DS>(
    point: &ArrayBase<PS, Ix1>,
    direction: &ArrayBase<DS, Ix1>,
    objective: F,
    gradient: G,
    config: &LineSearchConfig<T>,
) -> Result<T, OptimizationError>
where
    T: NabledReal,
    F: Fn(&Array1<T>) -> T,
    G: Fn(&Array1<T>) -> Array1<T>,
    PS: Data<Elem = T>,
    DS: Data<Elem = T>,
{
    validate_vector(point)?;
    validate_vector(direction)?;
    if point.len() != direction.len() {
        return Err(OptimizationError::DimensionMismatch);
    }
    validate_line_search_config(config)?;

    let point = point.to_owned();
    let direction = direction.to_owned();
    let grad = gradient(&point);
    if grad.len() != point.len() || grad.iter().any(|value| !value.is_finite()) {
        return Err(OptimizationError::NonFiniteInput);
    }

    let fx = objective(&point);
    if !fx.is_finite() {
        return Err(OptimizationError::NonFiniteInput);
    }
    let directional_derivative = grad.dot(&direction);

    let mut alpha = config.initial_step;
    let contraction = config.contraction;
    let sufficient_decrease = config.sufficient_decrease;
    for _ in 0..config.max_iterations {
        let candidate = &point + &direction.mapv(|value| value * alpha);
        let candidate_value = objective(&candidate);
        if !candidate_value.is_finite() {
            return Err(OptimizationError::NonFiniteInput);
        }
        if candidate_value <= fx + sufficient_decrease * alpha * directional_derivative {
            return Ok(alpha);
        }
        alpha *= contraction;
    }
    Err(OptimizationError::MaxIterationsExceeded)
}

/// Minimize objective with fixed-step gradient descent.
///
/// # Errors
/// Returns an error for invalid inputs/configuration or non-finite gradients.
pub fn gradient_descent<T, F, G, S>(
    initial: &ArrayBase<S, Ix1>,
    objective: F,
    gradient: G,
    config: &SGDConfig<T>,
) -> Result<Array1<T>, OptimizationError>
where
    T: NabledReal,
    F: Fn(&Array1<T>) -> T,
    G: Fn(&Array1<T>) -> Array1<T>,
    S: Data<Elem = T>,
{
    validate_vector(initial)?;
    validate_sgd_config(config)?;

    let mut x = initial.to_owned();
    let _ = objective(&x);
    let tolerance = config.tolerance.max(T::from_f64(DEFAULT_TOLERANCE).unwrap_or_else(T::epsilon));
    let learning_rate = config.learning_rate;

    for _ in 0..config.max_iterations {
        let grad = gradient(&x);
        if grad.len() != x.len() || grad.iter().any(|value| !value.is_finite()) {
            return Err(OptimizationError::NonFiniteInput);
        }
        if l2_norm(&grad) <= tolerance {
            return Ok(x);
        }
        x = &x - &grad.mapv(|value| value * learning_rate);
    }

    Err(OptimizationError::MaxIterationsExceeded)
}

/// Minimize objective with Adam.
///
/// # Errors
/// Returns an error for invalid inputs/configuration or non-finite gradients.
pub fn adam<T, F, G, S>(
    initial: &ArrayBase<S, Ix1>,
    objective: F,
    gradient: G,
    config: &AdamConfig<T>,
) -> Result<Array1<T>, OptimizationError>
where
    T: NabledReal,
    F: Fn(&Array1<T>) -> T,
    G: Fn(&Array1<T>) -> Array1<T>,
    S: Data<Elem = T>,
{
    validate_vector(initial)?;
    validate_adam_config(config)?;

    let mut x = initial.to_owned();
    let mut m = Array1::<T>::zeros(x.len());
    let mut v = Array1::<T>::zeros(x.len());
    let mut beta1_power = T::one();
    let mut beta2_power = T::one();
    let tolerance = config.tolerance.max(T::from_f64(DEFAULT_TOLERANCE).unwrap_or_else(T::epsilon));
    let beta1 = config.beta1;
    let beta2 = config.beta2;
    let learning_rate = config.learning_rate;
    let epsilon = config.epsilon;

    let _ = objective(&x);
    for _ in 0..config.max_iterations {
        let grad = gradient(&x);
        if grad.len() != x.len() || grad.iter().any(|value| !value.is_finite()) {
            return Err(OptimizationError::NonFiniteInput);
        }
        if l2_norm(&grad) <= tolerance {
            return Ok(x);
        }

        beta1_power *= beta1;
        beta2_power *= beta2;

        for i in 0..x.len() {
            m[i] = beta1 * m[i] + (T::one() - beta1) * grad[i];
            v[i] = beta2 * v[i] + (T::one() - beta2) * grad[i] * grad[i];

            let m_hat = m[i] / (T::one() - beta1_power);
            let v_hat = v[i] / (T::one() - beta2_power);
            x[i] -= learning_rate * m_hat / (v_hat.sqrt() + epsilon);
        }
    }

    Err(OptimizationError::MaxIterationsExceeded)
}

/// Minimize objective with momentum gradient descent.
///
/// # Errors
/// Returns an error for invalid inputs/configuration or non-finite gradients.
pub fn momentum_descent<T, F, G, S>(
    initial: &ArrayBase<S, Ix1>,
    objective: F,
    gradient: G,
    config: &MomentumConfig<T>,
) -> Result<Array1<T>, OptimizationError>
where
    T: NabledReal,
    F: Fn(&Array1<T>) -> T,
    G: Fn(&Array1<T>) -> Array1<T>,
    S: Data<Elem = T>,
{
    validate_vector(initial)?;
    validate_momentum_config(config)?;

    let mut x = initial.to_owned();
    let mut velocity = Array1::<T>::zeros(x.len());
    let tolerance = config.tolerance.max(T::from_f64(DEFAULT_TOLERANCE).unwrap_or_else(T::epsilon));
    let learning_rate = config.learning_rate;
    let momentum = config.momentum;

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
            velocity[i] = momentum * velocity[i] + grad[i];
            x[i] -= learning_rate * velocity[i];
        }
    }

    Err(OptimizationError::MaxIterationsExceeded)
}

/// Minimize objective with `RMSProp`.
///
/// # Errors
/// Returns an error for invalid inputs/configuration or non-finite gradients.
pub fn rmsprop<T, F, G, S>(
    initial: &ArrayBase<S, Ix1>,
    objective: F,
    gradient: G,
    config: &RMSPropConfig<T>,
) -> Result<Array1<T>, OptimizationError>
where
    T: NabledReal,
    F: Fn(&Array1<T>) -> T,
    G: Fn(&Array1<T>) -> Array1<T>,
    S: Data<Elem = T>,
{
    validate_vector(initial)?;
    validate_rmsprop_config(config)?;

    let mut x = initial.to_owned();
    let mut avg_sq = Array1::<T>::zeros(x.len());
    let tolerance = config.tolerance.max(T::from_f64(DEFAULT_TOLERANCE).unwrap_or_else(T::epsilon));
    let learning_rate = config.learning_rate;
    let rho = config.rho;
    let epsilon = config.epsilon;

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
            avg_sq[i] = rho * avg_sq[i] + (T::one() - rho) * grad[i] * grad[i];
            x[i] -= learning_rate * grad[i] / (avg_sq[i].sqrt() + epsilon);
        }
    }

    Err(OptimizationError::MaxIterationsExceeded)
}

/// Minimize objective with projected gradient descent under box constraints.
///
/// # Errors
/// Returns an error for invalid inputs/configuration, invalid bounds, or non-finite gradients.
pub fn projected_gradient_descent_box<T, F, G, IS, LS, US>(
    initial: &ArrayBase<IS, Ix1>,
    objective: F,
    gradient: G,
    lower_bounds: &ArrayBase<LS, Ix1>,
    upper_bounds: &ArrayBase<US, Ix1>,
    config: &ProjectedGradientConfig<T>,
) -> Result<Array1<T>, OptimizationError>
where
    T: NabledReal,
    F: Fn(&Array1<T>) -> T,
    G: Fn(&Array1<T>) -> Array1<T>,
    IS: Data<Elem = T>,
    LS: Data<Elem = T>,
    US: Data<Elem = T>,
{
    validate_vector(initial)?;
    validate_vector(lower_bounds)?;
    validate_vector(upper_bounds)?;
    validate_projected_gradient_config(config)?;
    validate_bounds(initial, lower_bounds, upper_bounds)?;

    let mut x = initial.to_owned();
    project_to_bounds(&mut x, lower_bounds, upper_bounds);
    let _ = objective(&x);
    let tolerance = config.tolerance.max(T::from_f64(DEFAULT_TOLERANCE).unwrap_or_else(T::epsilon));
    let learning_rate = config.learning_rate;

    for _ in 0..config.max_iterations {
        let grad = gradient(&x);
        if grad.len() != x.len() || grad.iter().any(|value| !value.is_finite()) {
            return Err(OptimizationError::NonFiniteInput);
        }
        let previous = x.clone();
        x = &x - &grad.mapv(|value| value * learning_rate);
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
pub fn stochastic_gradient_descent<T, G, S>(
    initial: &ArrayBase<S, Ix1>,
    stochastic_gradient: G,
    config: &SGDConfig<T>,
) -> Result<Array1<T>, OptimizationError>
where
    T: NabledReal,
    G: Fn(&Array1<T>, usize) -> Array1<T>,
    S: Data<Elem = T>,
{
    validate_vector(initial)?;
    validate_sgd_config(config)?;

    let mut x = initial.to_owned();
    let tolerance = config.tolerance.max(T::from_f64(DEFAULT_TOLERANCE).unwrap_or_else(T::epsilon));
    let learning_rate = config.learning_rate;
    for iteration in 0..config.max_iterations {
        let grad = stochastic_gradient(&x, iteration);
        if grad.len() != x.len() || grad.iter().any(|value| !value.is_finite()) {
            return Err(OptimizationError::NonFiniteInput);
        }
        if l2_norm(&grad) <= tolerance {
            return Ok(x);
        }
        x = &x - &grad.mapv(|value| value * learning_rate);
    }
    Err(OptimizationError::MaxIterationsExceeded)
}

/// Minimize objective with `BFGS` quasi-Newton updates.
///
/// # Errors
/// Returns an error for invalid inputs/configuration or non-finite gradients.
pub fn bfgs<T, F, G, S>(
    initial: &ArrayBase<S, Ix1>,
    objective: F,
    gradient: G,
    config: &BFGSConfig<T>,
) -> Result<Array1<T>, OptimizationError>
where
    T: NabledReal,
    F: Fn(&Array1<T>) -> T,
    G: Fn(&Array1<T>) -> Array1<T>,
    S: Data<Elem = T>,
{
    validate_vector(initial)?;
    validate_bfgs_config(config)?;

    let dimension = initial.len();
    let mut x = initial.to_owned();
    let mut h_inv = Array2::<T>::eye(dimension);
    let tolerance = config.tolerance.max(T::from_f64(DEFAULT_TOLERANCE).unwrap_or_else(T::epsilon));
    let step = config.step_size;
    let curvature_tolerance = config.curvature_tolerance;

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
        let x_next = &x + &direction.mapv(|value| value * step);
        let grad_next = gradient(&x_next);
        if grad_next.len() != x.len() || grad_next.iter().any(|value| !value.is_finite()) {
            return Err(OptimizationError::NonFiniteInput);
        }

        let s = &x_next - &x;
        let y = &grad_next - &grad;
        let ys = y.dot(&s);
        if ys.abs() > curvature_tolerance {
            let rho = T::one() / ys;
            let identity = Array2::<T>::eye(dimension);
            let sy = outer_product(&s, &y);
            let ys_outer = outer_product(&y, &s);
            let ss = outer_product(&s, &s);

            let scaled_cross_left = sy.mapv(|value| rho * value);
            let scaled_cross_right = ys_outer.mapv(|value| rho * value);
            let scaled_rank_one = ss.mapv(|value| rho * value);
            let left = &identity - &scaled_cross_left;
            let right = &identity - &scaled_cross_right;
            h_inv = left.dot(&h_inv).dot(&right) + scaled_rank_one;
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
    point: &ArrayBase<impl Data<Elem = Complex64>, Ix1>,
    direction: &ArrayBase<impl Data<Elem = Complex64>, Ix1>,
    objective: F,
    gradient: G,
    config: &LineSearchConfig<f64>,
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

    let point = point.to_owned();
    let direction = direction.to_owned();
    let grad = gradient(&point);
    if grad.len() != point.len()
        || grad.iter().any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(OptimizationError::NonFiniteInput);
    }

    let fx = objective(&point);
    if !fx.is_finite() {
        return Err(OptimizationError::NonFiniteInput);
    }
    let directional_derivative = hermitian_dot(&grad, &direction).re;

    let mut alpha = config.initial_step;
    for _ in 0..config.max_iterations {
        let candidate = &point + &direction.mapv(|value| value * alpha);
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
    initial: &ArrayBase<impl Data<Elem = Complex64>, Ix1>,
    objective: F,
    gradient: G,
    config: &SGDConfig<f64>,
) -> Result<Array1<Complex64>, OptimizationError>
where
    F: Fn(&Array1<Complex64>) -> f64,
    G: Fn(&Array1<Complex64>) -> Array1<Complex64>,
{
    validate_vector_complex(initial)?;
    validate_sgd_config(config)?;

    let mut x = initial.to_owned();
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
    initial: &ArrayBase<impl Data<Elem = Complex64>, Ix1>,
    objective: F,
    gradient: G,
    config: &AdamConfig<f64>,
) -> Result<Array1<Complex64>, OptimizationError>
where
    F: Fn(&Array1<Complex64>) -> f64,
    G: Fn(&Array1<Complex64>) -> Array1<Complex64>,
{
    validate_vector_complex(initial)?;
    validate_adam_config(config)?;

    let mut x = initial.to_owned();
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
    initial: &ArrayBase<impl Data<Elem = Complex64>, Ix1>,
    objective: F,
    gradient: G,
    config: &MomentumConfig<f64>,
) -> Result<Array1<Complex64>, OptimizationError>
where
    F: Fn(&Array1<Complex64>) -> f64,
    G: Fn(&Array1<Complex64>) -> Array1<Complex64>,
{
    validate_vector_complex(initial)?;
    validate_momentum_config(config)?;

    let mut x = initial.to_owned();
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
    initial: &ArrayBase<impl Data<Elem = Complex64>, Ix1>,
    objective: F,
    gradient: G,
    config: &RMSPropConfig<f64>,
) -> Result<Array1<Complex64>, OptimizationError>
where
    F: Fn(&Array1<Complex64>) -> f64,
    G: Fn(&Array1<Complex64>) -> Array1<Complex64>,
{
    validate_vector_complex(initial)?;
    validate_rmsprop_config(config)?;

    let mut x = initial.to_owned();
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
    initial: &ArrayBase<impl Data<Elem = Complex64>, Ix1>,
    objective: F,
    gradient: G,
    lower_bounds: &ArrayBase<impl Data<Elem = Complex64>, Ix1>,
    upper_bounds: &ArrayBase<impl Data<Elem = Complex64>, Ix1>,
    config: &ProjectedGradientConfig<f64>,
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

    let mut x = initial.to_owned();
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
    initial: &ArrayBase<impl Data<Elem = Complex64>, Ix1>,
    stochastic_gradient: G,
    config: &SGDConfig<f64>,
) -> Result<Array1<Complex64>, OptimizationError>
where
    G: Fn(&Array1<Complex64>, usize) -> Array1<Complex64>,
{
    validate_vector_complex(initial)?;
    validate_sgd_config(config)?;

    let mut x = initial.to_owned();
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
    initial: &ArrayBase<impl Data<Elem = Complex64>, Ix1>,
    objective: F,
    gradient: G,
    config: &BFGSConfig<f64>,
) -> Result<Array1<Complex64>, OptimizationError>
where
    F: Fn(&Array1<Complex64>) -> f64,
    G: Fn(&Array1<Complex64>) -> Array1<Complex64>,
{
    validate_vector_complex(initial)?;
    validate_bfgs_config(config)?;

    let dimension = initial.len();
    let mut x = initial.to_owned();
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
        let delta = x[0] - 3.0_f64;
        delta * delta
    }

    fn gradient(x: &Array1<f64>) -> Array1<f64> { arr1(&[2.0_f64 * (x[0] - 3.0_f64)]) }

    fn objective_complex(x: &Array1<Complex64>) -> f64 {
        let delta = x[0] - Complex64::new(3.0_f64, 2.0_f64);
        delta.norm_sqr()
    }

    fn gradient_complex(x: &Array1<Complex64>) -> Array1<Complex64> {
        arr1(&[Complex64::new(2.0_f64, 0.0_f64) * (x[0] - Complex64::new(3.0_f64, 2.0_f64))])
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
        assert!(alpha > 0.0_f64);
    }

    #[test]
    fn gradient_descent_converges_on_quadratic() {
        let x0 = arr1(&[0.0_f64]);
        let solution = gradient_descent(&x0, objective, gradient, &SGDConfig::default()).unwrap();
        assert!((solution[0] - 3.0_f64).abs() < 1e-4_f64);
    }

    #[test]
    fn adam_converges_on_quadratic() {
        let x0 = arr1(&[-5.0_f64]);
        let solution = adam(&x0, objective, gradient, &AdamConfig::default()).unwrap();
        assert!((solution[0] - 3.0_f64).abs() < 1e-3_f64);
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

        let bad_config = LineSearchConfig { contraction: 1.0_f64, ..LineSearchConfig::default() };
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
            learning_rate:  1e-12_f64,
            max_iterations: 1,
            tolerance:      0.0_f64,
        });
        assert!(matches!(gd_stall, Err(OptimizationError::MaxIterationsExceeded)));

        let bad_adam = AdamConfig { beta1: 1.0_f64, ..AdamConfig::default() };
        let adam_invalid = adam(&x0, objective, gradient, &bad_adam);
        assert!(matches!(adam_invalid, Err(OptimizationError::InvalidConfig)));
    }

    #[test]
    fn momentum_and_rmsprop_converge_on_quadratic() {
        let x0 = arr1(&[-4.0_f64]);

        let momentum_solution =
            momentum_descent(&x0, objective, gradient, &MomentumConfig::default()).unwrap();
        assert!((momentum_solution[0] - 3.0_f64).abs() < 1e-3_f64);

        let rmsprop_solution =
            rmsprop(&x0, objective, gradient, &RMSPropConfig::default()).unwrap();
        assert!((rmsprop_solution[0] - 3.0_f64).abs() < 1e-3_f64);
    }

    #[test]
    fn momentum_and_rmsprop_reject_invalid_config() {
        let x0 = arr1(&[0.0_f64]);

        let bad_momentum = MomentumConfig { momentum: 1.0_f64, ..MomentumConfig::default() };
        let momentum_invalid = momentum_descent(&x0, objective, gradient, &bad_momentum);
        assert!(matches!(momentum_invalid, Err(OptimizationError::InvalidConfig)));

        let bad_rmsprop = RMSPropConfig { rho: 1.0_f64, ..RMSPropConfig::default() };
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
        assert!((solution[0] - 2.5_f64).abs() < 1e-8_f64);
    }

    #[test]
    fn stochastic_gradient_descent_converges_on_quadratic() {
        let x0 = arr1(&[-3.0_f64]);
        let solution = stochastic_gradient_descent(
            &x0,
            |x, _iteration| arr1(&[2.0_f64 * (x[0] - 3.0_f64)]),
            &SGDConfig {
                learning_rate:  5e-2_f64,
                max_iterations: 2_000,
                tolerance:      1e-6_f64,
            },
        )
        .unwrap();
        assert!((solution[0] - 3.0_f64).abs() < 1e-3_f64);
    }

    #[test]
    fn bfgs_converges_on_quadratic() {
        let x0 = arr1(&[-8.0_f64]);
        let solution = bfgs(&x0, objective, gradient, &BFGSConfig::default()).unwrap();
        assert!((solution[0] - 3.0_f64).abs() < 1e-6_f64);
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

        let bfgs_invalid = bfgs(&x0, objective, gradient, &BFGSConfig {
            step_size: 0.0_f64,
            ..BFGSConfig::default()
        });
        assert!(matches!(bfgs_invalid, Err(OptimizationError::InvalidConfig)));
    }

    #[test]
    fn complex_line_search_and_optimizers_converge() {
        let x = arr1(&[Complex64::new(0.0_f64, 0.0_f64)]);
        let direction = arr1(&[Complex64::new(1.0_f64, 1.0_f64)]);
        let alpha = backtracking_line_search_complex(
            &x,
            &direction,
            objective_complex,
            gradient_complex,
            &LineSearchConfig::default(),
        )
        .unwrap();
        assert!(alpha > 0.0_f64);

        let gd = gradient_descent_complex(
            &x,
            objective_complex,
            gradient_complex,
            &SGDConfig::default(),
        )
        .unwrap();
        assert!((gd[0] - Complex64::new(3.0_f64, 2.0_f64)).norm() < 1e-3_f64);

        let adam_solution =
            adam_complex(&x, objective_complex, gradient_complex, &AdamConfig::default()).unwrap();
        assert!((adam_solution[0] - Complex64::new(3.0_f64, 2.0_f64)).norm() < 1e-3_f64);

        let momentum_solution = momentum_descent_complex(
            &x,
            objective_complex,
            gradient_complex,
            &MomentumConfig::default(),
        )
        .unwrap();
        assert!((momentum_solution[0] - Complex64::new(3.0_f64, 2.0_f64)).norm() < 1e-3_f64);

        let rmsprop_solution =
            rmsprop_complex(&x, objective_complex, gradient_complex, &RMSPropConfig::default())
                .unwrap();
        assert!((rmsprop_solution[0] - Complex64::new(3.0_f64, 2.0_f64)).norm() < 1e-3_f64);

        let bfgs_solution =
            bfgs_complex(&x, objective_complex, gradient_complex, &BFGSConfig::default()).unwrap();
        assert!((bfgs_solution[0] - Complex64::new(3.0_f64, 2.0_f64)).norm() < 1e-6_f64);
    }

    #[test]
    fn complex_projected_and_stochastic_paths_work() {
        let x = arr1(&[Complex64::new(-5.0_f64, -5.0_f64)]);
        let lower = arr1(&[Complex64::new(0.0_f64, 0.0_f64)]);
        let upper = arr1(&[Complex64::new(2.5_f64, 2.5_f64)]);
        let projected = projected_gradient_descent_box_complex(
            &x,
            objective_complex,
            gradient_complex,
            &lower,
            &upper,
            &ProjectedGradientConfig::default(),
        )
        .unwrap();
        assert!((projected[0] - Complex64::new(2.5_f64, 2.0_f64)).norm() < 1e-3_f64);

        let stochastic = stochastic_gradient_descent_complex(
            &arr1(&[Complex64::new(-3.0_f64, -1.0_f64)]),
            |point, _iteration| {
                let grad_sample = Complex64::new(2.0_f64, 0.0_f64)
                    * (point[0] - Complex64::new(3.0_f64, 2.0_f64));
                arr1(&[grad_sample])
            },
            &SGDConfig {
                learning_rate:  5e-2_f64,
                max_iterations: 2_000,
                tolerance:      1e-6_f64,
            },
        )
        .unwrap();
        assert!((stochastic[0] - Complex64::new(3.0_f64, 2.0_f64)).norm() < 1e-3_f64);
    }
}
