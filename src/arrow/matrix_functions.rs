//! Arrow-native matrix function operations

use super::conversions::{ndarray_to_record_batch, record_batch_to_ndarray};
use super::error::ArrowLinalgError;
use crate::matrix_functions::ndarray_matrix_functions;
use arrow::record_batch::RecordBatch;

/// Matrix exponential using eigenvalue decomposition
pub fn matrix_exp_eigen(batch: &RecordBatch) -> Result<RecordBatch, ArrowLinalgError> {
    let array = record_batch_to_ndarray(batch)?;
    let result = ndarray_matrix_functions::matrix_exp_eigen(&array)?;
    ndarray_to_record_batch(&result).map_err(ArrowLinalgError::from)
}

/// Matrix exponential using Taylor series
pub fn matrix_exp(
    batch: &RecordBatch,
    max_iterations: usize,
    tolerance: f64,
) -> Result<RecordBatch, ArrowLinalgError> {
    let array = record_batch_to_ndarray(batch)?;
    let result = ndarray_matrix_functions::matrix_exp(&array, max_iterations, tolerance)?;
    ndarray_to_record_batch(&result).map_err(ArrowLinalgError::from)
}

/// Matrix logarithm using eigenvalue decomposition
pub fn matrix_log_eigen(batch: &RecordBatch) -> Result<RecordBatch, ArrowLinalgError> {
    let array = record_batch_to_ndarray(batch)?;
    let result = ndarray_matrix_functions::matrix_log_eigen(&array)?;
    ndarray_to_record_batch(&result).map_err(ArrowLinalgError::from)
}

/// Matrix logarithm using Taylor series
pub fn matrix_log_taylor(
    batch: &RecordBatch,
    max_iterations: usize,
    tolerance: f64,
) -> Result<RecordBatch, ArrowLinalgError> {
    let array = record_batch_to_ndarray(batch)?;
    let result = ndarray_matrix_functions::matrix_log_taylor(&array, max_iterations, tolerance)?;
    ndarray_to_record_batch(&result).map_err(ArrowLinalgError::from)
}

/// Matrix logarithm using SVD
pub fn matrix_log_svd(batch: &RecordBatch) -> Result<RecordBatch, ArrowLinalgError> {
    let array = record_batch_to_ndarray(batch)?;
    let result = ndarray_matrix_functions::matrix_log_svd(&array)?;
    ndarray_to_record_batch(&result).map_err(ArrowLinalgError::from)
}

/// Matrix power (e.g., square root with power=0.5)
pub fn matrix_power(batch: &RecordBatch, power: f64) -> Result<RecordBatch, ArrowLinalgError> {
    let array = record_batch_to_ndarray(batch)?;
    let result = ndarray_matrix_functions::matrix_power(&array, power)?;
    ndarray_to_record_batch(&result).map_err(ArrowLinalgError::from)
}
