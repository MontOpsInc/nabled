//! Arrow-native Jacobian and derivative operations

use super::conversions::{
    dvector_to_float64_array, float64_array_to_dvector, ndarray_to_record_batch,
};
use super::error::ArrowLinalgError;
use crate::jacobian::nalgebra_jacobian;
use crate::jacobian::JacobianConfig;
use arrow::array::Float64Array;
use arrow::record_batch::RecordBatch;
use nalgebra::DVector;

/// Compute numerical Jacobian from Arrow vector input
pub fn numerical_jacobian<F>(
    f: F,
    x: &Float64Array,
    config: &JacobianConfig<f64>,
) -> Result<RecordBatch, ArrowLinalgError>
where
    F: Fn(&DVector<f64>) -> Result<DVector<f64>, String>,
{
    let x_vector = float64_array_to_dvector(x)?;
    let jacobian = nalgebra_jacobian::numerical_jacobian(&f, &x_vector, config)?;
    ndarray_to_record_batch(&crate::utils::nalgebra_to_ndarray(&jacobian))
        .map_err(ArrowLinalgError::from)
}

/// Compute numerical Jacobian using central differences
pub fn numerical_jacobian_central<F>(
    f: F,
    x: &Float64Array,
    config: &JacobianConfig<f64>,
) -> Result<RecordBatch, ArrowLinalgError>
where
    F: Fn(&DVector<f64>) -> Result<DVector<f64>, String>,
{
    let x_vector = float64_array_to_dvector(x)?;
    let jacobian = nalgebra_jacobian::numerical_jacobian_central(&f, &x_vector, config)?;
    ndarray_to_record_batch(&crate::utils::nalgebra_to_ndarray(&jacobian))
        .map_err(ArrowLinalgError::from)
}

/// Compute numerical gradient for scalar function
pub fn numerical_gradient<F>(
    f: F,
    x: &Float64Array,
    config: &JacobianConfig<f64>,
) -> Result<Float64Array, ArrowLinalgError>
where
    F: Fn(&DVector<f64>) -> Result<f64, String>,
{
    let x_vector = float64_array_to_dvector(x)?;
    let gradient = nalgebra_jacobian::numerical_gradient(&f, &x_vector, config)?;
    Ok(dvector_to_float64_array(&gradient))
}

/// Compute numerical Hessian for scalar function
pub fn numerical_hessian<F>(
    f: F,
    x: &Float64Array,
    config: &JacobianConfig<f64>,
) -> Result<RecordBatch, ArrowLinalgError>
where
    F: Fn(&DVector<f64>) -> Result<f64, String>,
{
    let x_vector = float64_array_to_dvector(x)?;
    let hessian = nalgebra_jacobian::numerical_hessian(&f, &x_vector, config)?;
    ndarray_to_record_batch(&crate::utils::nalgebra_to_ndarray(&hessian))
        .map_err(ArrowLinalgError::from)
}
