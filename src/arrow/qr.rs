//! Arrow-native QR decomposition operations

use super::conversions::{
    dvector_to_float64_array, float64_array_to_dvector, record_batch_to_nalgebra,
};
use super::error::ArrowLinalgError;
use crate::qr::nalgebra_qr;
use crate::qr::QRConfig;
use arrow::array::Float64Array;
use arrow::record_batch::RecordBatch;
use ndarray::Array2;
use std::sync::Arc;

/// Arrow representation of QR result
#[derive(Debug, Clone)]
pub struct ArrowQRResult {
    /// Orthogonal matrix Q as column arrays
    pub q: Vec<Arc<Float64Array>>,
    /// Upper triangular matrix R as column arrays
    pub r: Vec<Arc<Float64Array>>,
    /// Column permutation (if pivoting used)
    pub p: Option<Vec<Arc<Float64Array>>>,
    /// Matrix rank
    pub rank: usize,
}

/// Compute QR decomposition from RecordBatch
pub fn compute_qr(
    batch: &RecordBatch,
    config: &QRConfig<f64>,
) -> Result<ArrowQRResult, ArrowLinalgError> {
    let matrix = record_batch_to_nalgebra(batch)?;
    let qr = nalgebra_qr::compute_qr(&matrix, config)?;
    Ok(qr_to_arrow(qr))
}

/// Compute reduced QR decomposition
pub fn compute_reduced_qr(
    batch: &RecordBatch,
    config: &QRConfig<f64>,
) -> Result<ArrowQRResult, ArrowLinalgError> {
    let matrix = record_batch_to_nalgebra(batch)?;
    let qr = nalgebra_qr::compute_reduced_qr(&matrix, config)?;
    Ok(qr_to_arrow(qr))
}

/// Compute QR with column pivoting
pub fn compute_qr_with_pivoting(
    batch: &RecordBatch,
    config: &QRConfig<f64>,
) -> Result<ArrowQRResult, ArrowLinalgError> {
    let matrix = record_batch_to_nalgebra(batch)?;
    let qr = nalgebra_qr::compute_qr_with_pivoting(&matrix, config)?;
    Ok(qr_to_arrow(qr))
}

/// Solve least squares: min ||Ax - b||
pub fn solve_least_squares(
    a_batch: &RecordBatch,
    b: &Float64Array,
    config: &QRConfig<f64>,
) -> Result<Float64Array, ArrowLinalgError> {
    let a = record_batch_to_nalgebra(a_batch)?;
    let b_vector = float64_array_to_dvector(b)?;
    let x = nalgebra_qr::solve_least_squares(&a, &b_vector, config)?;
    Ok(dvector_to_float64_array(&x))
}

/// Reconstruct matrix from QR result
pub fn reconstruct_matrix(qr: &ArrowQRResult) -> Result<RecordBatch, ArrowLinalgError> {
    let q = arrow_columns_to_ndarray(&qr.q)?;
    let r = arrow_columns_to_ndarray(&qr.r)?;
    let (m, nq) = q.dim();
    let (nr, nc) = r.dim();
    let n = nq.min(nr);
    let mut result = Array2::zeros((m, nc));
    for i in 0..m {
        for j in 0..nc {
            for k in 0..n {
                result[[i, j]] += q[[i, k]] * r[[k, j]];
            }
        }
    }
    super::conversions::ndarray_to_record_batch(&result).map_err(ArrowLinalgError::from)
}

/// Compute condition number from QR result
pub fn condition_number(qr: &ArrowQRResult) -> f64 {
    let qr_result = arrow_qr_to_nalgebra(qr);
    nalgebra_qr::condition_number(&qr_result)
}

fn qr_to_arrow(qr: crate::QRResult<f64>) -> ArrowQRResult {
    let q = matrix_to_arrow_columns(&qr.q);
    let r = matrix_to_arrow_columns(&qr.r);
    let p = qr.p.as_ref().map(matrix_to_arrow_columns);

    ArrowQRResult {
        q,
        r,
        p,
        rank: qr.rank,
    }
}

fn matrix_to_arrow_columns(matrix: &nalgebra::DMatrix<f64>) -> Vec<Arc<Float64Array>> {
    let (rows, cols) = matrix.shape();
    let mut columns = Vec::with_capacity(cols);
    for col in 0..cols {
        let values: Vec<Option<f64>> = (0..rows).map(|row| Some(matrix[(row, col)])).collect();
        columns.push(Arc::new(Float64Array::from_iter(values)));
    }
    columns
}

fn arrow_columns_to_ndarray(
    columns: &[Arc<Float64Array>],
) -> Result<Array2<f64>, super::error::ArrowConversionError> {
    if columns.is_empty() {
        return Err(super::error::ArrowConversionError::EmptyArray);
    }
    let rows = columns[0].len();
    let cols = columns.len();
    let mut data = Vec::with_capacity(rows * cols);
    for col in columns {
        if col.len() != rows {
            return Err(super::error::ArrowConversionError::DimensionMismatch(
                format!("Column length {} != expected {}", col.len(), rows),
            ));
        }
        for v in col.iter() {
            data.push(v.ok_or(super::error::ArrowConversionError::NullValues)?);
        }
    }
    let mut reshaped = Vec::with_capacity(rows * cols);
    for row in 0..rows {
        for col in 0..cols {
            reshaped.push(data[col * rows + row]);
        }
    }
    Array2::from_shape_vec((rows, cols), reshaped)
        .map_err(|e| super::error::ArrowConversionError::InvalidShape(format!("{:?}", e)))
}

fn arrow_qr_to_nalgebra(qr: &ArrowQRResult) -> crate::QRResult<f64> {
    let q = arrow_columns_to_ndarray(&qr.q).expect("valid q");
    let r = arrow_columns_to_ndarray(&qr.r).expect("valid r");
    let q_nalg = crate::utils::ndarray_to_nalgebra(&q);
    let r_nalg = crate::utils::ndarray_to_nalgebra(&r);
    let p = qr.p.as_ref().map(|p_cols| {
        let p_arr = arrow_columns_to_ndarray(p_cols).expect("valid p");
        crate::utils::ndarray_to_nalgebra(&p_arr)
    });

    crate::QRResult {
        q: q_nalg,
        r: r_nalg,
        p,
        rank: qr.rank,
    }
}
