//! Arrow-native SVD (Singular Value Decomposition) operations

use super::conversions::{ndarray_to_record_batch, record_batch_to_ndarray};
use super::error::{ArrowConversionError, ArrowLinalgError};
use crate::svd::ndarray_svd;
use arrow::array::Float64Array;
use arrow::record_batch::RecordBatch;
use ndarray::Array2;
use std::sync::Arc;

/// Arrow representation of SVD result
#[derive(Debug, Clone)]
pub struct ArrowSVDResult {
    /// Left singular vectors (U matrix) as column arrays
    pub u: Vec<Arc<Float64Array>>,
    /// Singular values
    pub singular_values: Arc<Float64Array>,
    /// Right singular vectors (V^T matrix) as column arrays
    pub vt: Vec<Arc<Float64Array>>,
}

/// Compute SVD from RecordBatch
pub fn compute_svd(batch: &RecordBatch) -> Result<ArrowSVDResult, ArrowLinalgError> {
    let array = record_batch_to_ndarray(batch)?;
    let svd = ndarray_svd::compute_svd(&array)?;
    Ok(svd_to_arrow(svd))
}

/// Compute truncated SVD
pub fn compute_truncated_svd(batch: &RecordBatch, k: usize) -> Result<ArrowSVDResult, ArrowLinalgError> {
    let array = record_batch_to_ndarray(batch)?;
    let svd = ndarray_svd::compute_truncated_svd(&array, k)?;
    Ok(svd_to_arrow(svd))
}

/// Reconstruct matrix from Arrow SVD result
pub fn reconstruct_matrix(svd: &ArrowSVDResult) -> Result<RecordBatch, ArrowLinalgError> {
    let u = arrow_columns_to_ndarray(&svd.u)?;
    let vt = arrow_columns_to_ndarray(&svd.vt)?;
    let singular_values: Vec<f64> = svd
        .singular_values
        .iter()
        .map(|v| v.unwrap_or(0.0))
        .collect();
    let sv = ndarray::Array1::from_vec(singular_values);

    let ndarray_svd_result = crate::NdarraySVD {
        u,
        singular_values: sv,
        vt,
    };

    let reconstructed = ndarray_svd::reconstruct_matrix(&ndarray_svd_result);
    ndarray_to_record_batch(&reconstructed).map_err(ArrowLinalgError::from)
}

/// Compute condition number from SVD result
pub fn condition_number(svd: &ArrowSVDResult) -> f64 {
    let values: Vec<f64> = svd
        .singular_values
        .iter()
        .filter_map(|v| v)
        .collect();
    if values.is_empty() {
        return 0.0;
    }
    let max_sv = values.iter().fold(0.0f64, |a, &b| a.max(b));
    let min_sv = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    if min_sv == 0.0 {
        f64::INFINITY
    } else {
        max_sv / min_sv
    }
}

/// Compute matrix rank from SVD result
pub fn matrix_rank(svd: &ArrowSVDResult, tolerance: Option<f64>) -> usize {
    let values: Vec<f64> = svd
        .singular_values
        .iter()
        .filter_map(|v| v)
        .collect();
    let tol = tolerance.unwrap_or_else(|| {
        let max_sv = values.iter().fold(0.0f64, |a, &b| a.max(b));
        1e-10 * max_sv
    });
    values.iter().filter(|&&sv| sv > tol).count()
}

fn svd_to_arrow(svd: crate::NdarraySVD<f64>) -> ArrowSVDResult {
    let u = ndarray_to_arrow_columns(&svd.u);
    let vt = ndarray_to_arrow_columns(&svd.vt);
    let singular_values = Arc::new(Float64Array::from_iter(
        svd.singular_values.iter().map(|&v| Some(v)),
    ));

    ArrowSVDResult {
        u,
        singular_values,
        vt,
    }
}

fn ndarray_to_arrow_columns(array: &ndarray::Array2<f64>) -> Vec<Arc<Float64Array>> {
    let (rows, cols) = array.dim();
    let mut columns = Vec::with_capacity(cols);
    for col in 0..cols {
        let values: Vec<Option<f64>> = (0..rows).map(|row| Some(array[[row, col]])).collect();
        columns.push(Arc::new(Float64Array::from_iter(values)));
    }
    columns
}

fn arrow_columns_to_ndarray(
    columns: &[Arc<Float64Array>],
) -> Result<ndarray::Array2<f64>, ArrowConversionError> {
    if columns.is_empty() {
        return Err(ArrowConversionError::EmptyArray);
    }
    let rows = columns[0].len();
    let cols = columns.len();
    let mut data = Vec::with_capacity(rows * cols);
    for col in columns {
        if col.len() != rows {
            return Err(ArrowConversionError::DimensionMismatch(format!(
                "Column length {} != expected {}",
                col.len(),
                rows
            )));
        }
        for v in col.iter() {
            data.push(v.ok_or(ArrowConversionError::NullValues)?);
        }
    }
    let mut reshaped = Vec::with_capacity(rows * cols);
    for row in 0..rows {
        for col in 0..cols {
            reshaped.push(data[col * rows + row]);
        }
    }
    Array2::from_shape_vec((rows, cols), reshaped)
        .map_err(|e| ArrowConversionError::InvalidShape(format!("{:?}", e)))
}
