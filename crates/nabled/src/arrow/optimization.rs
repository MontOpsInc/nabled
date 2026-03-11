//! Arrow adapters for optimization workflows.

use arrow_array::types::ArrowPrimitiveType;
use arrow_array::{FixedSizeListArray, PrimitiveArray};
use arrow_schema::Field;
use nabled_core::scalar::NabledReal;
use ndarrow::NdarrowElement;

use super::{
    ArrowInteropError, complex64_vector_from_owned, complex64_vector_view,
    primitive_array_from_owned, primitive_array_view,
};

/// Perform Armijo backtracking line search from Arrow dense vectors.
///
/// # Errors
/// Returns an error when inputs or configuration are invalid.
pub fn backtracking_line_search<T, F, G>(
    point: &PrimitiveArray<T>,
    direction: &PrimitiveArray<T>,
    objective: F,
    gradient: G,
    config: &crate::ml::optimization::LineSearchConfig<T::Native>,
) -> Result<T::Native, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
    F: Fn(&ndarray::Array1<T::Native>) -> T::Native,
    G: Fn(&ndarray::Array1<T::Native>) -> ndarray::Array1<T::Native>,
{
    let point_view = primitive_array_view(point)?;
    let direction_view = primitive_array_view(direction)?;
    Ok(crate::ml::optimization::backtracking_line_search(
        &point_view,
        &direction_view,
        objective,
        gradient,
        config,
    )?)
}

macro_rules! optimize_wrappers {
    ($(($name:ident, $call:path, $config:path)),* $(,)?) => {
        $(
            /// Run an optimization workflow from an Arrow dense initial vector.
            ///
            /// # Errors
            /// Returns an error when inputs or configuration are invalid.
            pub fn $name<T, F, G>(
                initial: &PrimitiveArray<T>,
                objective: F,
                gradient: G,
                config: &$config,
            ) -> Result<PrimitiveArray<T>, ArrowInteropError>
            where
                T: ArrowPrimitiveType,
                T::Native: NabledReal + NdarrowElement,
                F: Fn(&ndarray::Array1<T::Native>) -> T::Native,
                G: Fn(&ndarray::Array1<T::Native>) -> ndarray::Array1<T::Native>,
            {
                let initial_view = primitive_array_view(initial)?;
                let output = $call(&initial_view, objective, gradient, config)?;
                Ok(primitive_array_from_owned::<T>(output))
            }
        )*
    };
}

optimize_wrappers!(
    (
        gradient_descent,
        crate::ml::optimization::gradient_descent,
        crate::ml::optimization::SGDConfig<T::Native>
    ),
    (adam, crate::ml::optimization::adam, crate::ml::optimization::AdamConfig<T::Native>),
    (
        momentum_descent,
        crate::ml::optimization::momentum_descent,
        crate::ml::optimization::MomentumConfig<T::Native>
    ),
    (
        rmsprop,
        crate::ml::optimization::rmsprop,
        crate::ml::optimization::RMSPropConfig<T::Native>
    ),
    (bfgs, crate::ml::optimization::bfgs, crate::ml::optimization::BFGSConfig<T::Native>),
);

/// Run projected gradient descent with box constraints from Arrow dense vectors.
///
/// # Errors
/// Returns an error when inputs or configuration are invalid.
pub fn projected_gradient_descent_box<T, F, G>(
    initial: &PrimitiveArray<T>,
    objective: F,
    gradient: G,
    lower_bounds: &PrimitiveArray<T>,
    upper_bounds: &PrimitiveArray<T>,
    config: &crate::ml::optimization::ProjectedGradientConfig<T::Native>,
) -> Result<PrimitiveArray<T>, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
    F: Fn(&ndarray::Array1<T::Native>) -> T::Native,
    G: Fn(&ndarray::Array1<T::Native>) -> ndarray::Array1<T::Native>,
{
    let initial_view = primitive_array_view(initial)?;
    let lower_view = primitive_array_view(lower_bounds)?;
    let upper_view = primitive_array_view(upper_bounds)?;
    let output = crate::ml::optimization::projected_gradient_descent_box(
        &initial_view,
        objective,
        gradient,
        &lower_view,
        &upper_view,
        config,
    )?;
    Ok(primitive_array_from_owned::<T>(output))
}

/// Run stochastic gradient descent from an Arrow dense initial vector.
///
/// # Errors
/// Returns an error when inputs or configuration are invalid.
pub fn stochastic_gradient_descent<T, G>(
    initial: &PrimitiveArray<T>,
    stochastic_gradient: G,
    config: &crate::ml::optimization::SGDConfig<T::Native>,
) -> Result<PrimitiveArray<T>, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
    G: Fn(&ndarray::Array1<T::Native>, usize) -> ndarray::Array1<T::Native>,
{
    let initial_view = primitive_array_view(initial)?;
    let output = crate::ml::optimization::stochastic_gradient_descent(
        &initial_view,
        stochastic_gradient,
        config,
    )?;
    Ok(primitive_array_from_owned::<T>(output))
}

/// Perform Armijo backtracking line search from Arrow complex dense vectors.
///
/// # Errors
/// Returns an error when inputs or configuration are invalid.
pub fn backtracking_line_search_complex<F, G>(
    point_field: &Field,
    point: &FixedSizeListArray,
    direction_field: &Field,
    direction: &FixedSizeListArray,
    objective: F,
    gradient: G,
    config: &crate::ml::optimization::LineSearchConfig<f64>,
) -> Result<f64, ArrowInteropError>
where
    F: Fn(&ndarray::Array1<num_complex::Complex64>) -> f64,
    G: Fn(&ndarray::Array1<num_complex::Complex64>) -> ndarray::Array1<num_complex::Complex64>,
{
    let point_view = complex64_vector_view(point_field, point)?;
    let direction_view = complex64_vector_view(direction_field, direction)?;
    Ok(crate::ml::optimization::backtracking_line_search_complex(
        &point_view,
        &direction_view,
        objective,
        gradient,
        config,
    )?)
}

macro_rules! optimize_complex_wrappers {
    ($(($name:ident, $call:path, $config:path)),* $(,)?) => {
        $(
            /// Run a complex optimization workflow from an Arrow complex initial vector.
            ///
            /// # Errors
            /// Returns an error when inputs or configuration are invalid.
            pub fn $name<F, G>(
                initial_field: &Field,
                initial: &FixedSizeListArray,
                objective: F,
                gradient: G,
                config: &$config,
            ) -> Result<(Field, FixedSizeListArray), ArrowInteropError>
            where
                F: Fn(&ndarray::Array1<num_complex::Complex64>) -> f64,
                G: Fn(
                    &ndarray::Array1<num_complex::Complex64>,
                ) -> ndarray::Array1<num_complex::Complex64>,
            {
                let initial_view = complex64_vector_view(initial_field, initial)?;
                let output = $call(&initial_view, objective, gradient, config)?;
                complex64_vector_from_owned(stringify!($name), output)
            }
        )*
    };
}

optimize_complex_wrappers!(
    (
        gradient_descent_complex,
        crate::ml::optimization::gradient_descent_complex,
        crate::ml::optimization::SGDConfig<f64>
    ),
    (
        adam_complex,
        crate::ml::optimization::adam_complex,
        crate::ml::optimization::AdamConfig<f64>
    ),
    (
        momentum_descent_complex,
        crate::ml::optimization::momentum_descent_complex,
        crate::ml::optimization::MomentumConfig<f64>
    ),
    (
        rmsprop_complex,
        crate::ml::optimization::rmsprop_complex,
        crate::ml::optimization::RMSPropConfig<f64>
    ),
    (
        bfgs_complex,
        crate::ml::optimization::bfgs_complex,
        crate::ml::optimization::BFGSConfig<f64>
    ),
);

/// Run projected gradient descent with box constraints from Arrow complex dense vectors.
///
/// # Errors
/// Returns an error when inputs or configuration are invalid.
pub fn projected_gradient_descent_box_complex<F, G>(
    field: &Field,
    initial: &FixedSizeListArray,
    objective: F,
    gradient: G,
    lower_bounds: &FixedSizeListArray,
    upper_bounds: &FixedSizeListArray,
    config: &crate::ml::optimization::ProjectedGradientConfig<f64>,
) -> Result<(Field, FixedSizeListArray), ArrowInteropError>
where
    F: Fn(&ndarray::Array1<num_complex::Complex64>) -> f64,
    G: Fn(&ndarray::Array1<num_complex::Complex64>) -> ndarray::Array1<num_complex::Complex64>,
{
    let initial_view = complex64_vector_view(field, initial)?;
    let lower_view = complex64_vector_view(field, lower_bounds)?;
    let upper_view = complex64_vector_view(field, upper_bounds)?;
    let output = crate::ml::optimization::projected_gradient_descent_box_complex(
        &initial_view,
        objective,
        gradient,
        &lower_view,
        &upper_view,
        config,
    )?;
    complex64_vector_from_owned("projected_gradient_descent_box_complex", output)
}

/// Run stochastic gradient descent from an Arrow complex initial vector.
///
/// # Errors
/// Returns an error when inputs or configuration are invalid.
pub fn stochastic_gradient_descent_complex<G>(
    initial_field: &Field,
    initial: &FixedSizeListArray,
    stochastic_gradient: G,
    config: &crate::ml::optimization::SGDConfig<f64>,
) -> Result<(Field, FixedSizeListArray), ArrowInteropError>
where
    G: Fn(
        &ndarray::Array1<num_complex::Complex64>,
        usize,
    ) -> ndarray::Array1<num_complex::Complex64>,
{
    let initial_view = complex64_vector_view(initial_field, initial)?;
    let output = crate::ml::optimization::stochastic_gradient_descent_complex(
        &initial_view,
        stochastic_gradient,
        config,
    )?;
    complex64_vector_from_owned("stochastic_gradient_descent_complex", output)
}
