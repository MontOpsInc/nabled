//! Arrow adapters for numerical Jacobian/gradient/Hessian workflows.

use arrow_array::types::ArrowPrimitiveType;
use arrow_array::{FixedSizeListArray, PrimitiveArray};
use nabled_core::scalar::NabledReal;
use ndarrow::NdarrowElement;

use super::{
    ArrowInteropError, fixed_size_list_from_owned, primitive_array_from_owned, primitive_array_view,
};

/// Compute numerical Jacobian via forward differences from an Arrow dense vector input.
///
/// # Errors
/// Returns an error when inputs are invalid or the callback returns an error.
pub fn numerical_jacobian<T, F>(
    function: &F,
    x: &PrimitiveArray<T>,
    config: &crate::ml::jacobian::JacobianConfig<T::Native>,
) -> Result<FixedSizeListArray, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
    F: Fn(
        &ndarray::Array1<T::Native>,
    ) -> Result<ndarray::Array1<T::Native>, crate::ml::jacobian::JacobianError>,
{
    let x_view = primitive_array_view(x)?;
    let output = crate::ml::jacobian::numerical_jacobian(function, &x_view, config)?;
    fixed_size_list_from_owned::<T>(output)
}

/// Compute numerical Jacobian via central differences from an Arrow dense vector input.
///
/// # Errors
/// Returns an error when inputs are invalid or the callback returns an error.
pub fn numerical_jacobian_central<T, F>(
    function: &F,
    x: &PrimitiveArray<T>,
    config: &crate::ml::jacobian::JacobianConfig<T::Native>,
) -> Result<FixedSizeListArray, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
    F: Fn(
        &ndarray::Array1<T::Native>,
    ) -> Result<ndarray::Array1<T::Native>, crate::ml::jacobian::JacobianError>,
{
    let x_view = primitive_array_view(x)?;
    let output = crate::ml::jacobian::numerical_jacobian_central(function, &x_view, config)?;
    fixed_size_list_from_owned::<T>(output)
}

/// Compute numerical gradient of a scalar function from an Arrow dense vector input.
///
/// # Errors
/// Returns an error when inputs are invalid or the callback returns an error.
pub fn numerical_gradient<T, F>(
    function: &F,
    x: &PrimitiveArray<T>,
    config: &crate::ml::jacobian::JacobianConfig<T::Native>,
) -> Result<PrimitiveArray<T>, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
    F: Fn(&ndarray::Array1<T::Native>) -> Result<T::Native, crate::ml::jacobian::JacobianError>,
{
    let x_view = primitive_array_view(x)?;
    let output = crate::ml::jacobian::numerical_gradient(function, &x_view, config)?;
    Ok(primitive_array_from_owned::<T>(output))
}

/// Compute numerical Hessian of a scalar function from an Arrow dense vector input.
///
/// # Errors
/// Returns an error when inputs are invalid or the callback returns an error.
pub fn numerical_hessian<T, F>(
    function: &F,
    x: &PrimitiveArray<T>,
    config: &crate::ml::jacobian::JacobianConfig<T::Native>,
) -> Result<FixedSizeListArray, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
    F: Fn(&ndarray::Array1<T::Native>) -> Result<T::Native, crate::ml::jacobian::JacobianError>,
{
    let x_view = primitive_array_view(x)?;
    let output = crate::ml::jacobian::numerical_hessian(function, &x_view, config)?;
    fixed_size_list_from_owned::<T>(output)
}
