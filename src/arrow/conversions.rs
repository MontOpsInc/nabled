//! Conversion utilities between Arrow and nalgebra/ndarray types

use super::error::ArrowConversionError;
use arrow::array::{Array, AsArray, Float64Array, PrimitiveArray};
use arrow::datatypes::Float64Type;
use arrow::record_batch::RecordBatch;
use arrow_schema::{DataType, Field, Schema};
use nalgebra::{DMatrix, DVector};
use ndarray::{Array1, Array2};
use std::sync::Arc;

/// Convert a RecordBatch to nalgebra DMatrix (columns = matrix columns, rows = matrix rows)
pub fn record_batch_to_nalgebra(batch: &RecordBatch) -> Result<DMatrix<f64>, ArrowConversionError> {
    let num_rows = batch.num_rows();
    let num_cols = batch.num_columns();

    if num_rows == 0 || num_cols == 0 {
        return Err(ArrowConversionError::EmptyArray);
    }

    let mut matrix = DMatrix::zeros(num_rows, num_cols);

    for (col_idx, column) in batch.columns().iter().enumerate() {
        let float_array = column.as_primitive_opt::<Float64Type>().ok_or_else(|| {
            ArrowConversionError::UnsupportedType("All columns must be Float64 type".to_string())
        })?;

        if float_array.len() != num_rows {
            return Err(ArrowConversionError::DimensionMismatch(format!(
                "Column {} has {} rows, expected {}",
                col_idx,
                float_array.len(),
                num_rows
            )));
        }

        for (row_idx, value) in float_array.iter().enumerate() {
            let v: f64 = value.ok_or(ArrowConversionError::NullValues)?;
            matrix[(row_idx, col_idx)] = v;
        }
    }

    Ok(matrix)
}

/// Convert a RecordBatch to ndarray Array2 (row-major)
pub fn record_batch_to_ndarray(batch: &RecordBatch) -> Result<Array2<f64>, ArrowConversionError> {
    let num_rows = batch.num_rows();
    let num_cols = batch.num_columns();

    if num_rows == 0 || num_cols == 0 {
        return Err(ArrowConversionError::EmptyArray);
    }

    let mut data = Vec::with_capacity(num_rows * num_cols);

    for column in batch.columns() {
        let float_array = column.as_primitive_opt::<Float64Type>().ok_or_else(|| {
            ArrowConversionError::UnsupportedType("All columns must be Float64 type".to_string())
        })?;

        if float_array.len() != num_rows {
            return Err(ArrowConversionError::DimensionMismatch(
                "Column length mismatch".to_string(),
            ));
        }

        for value in float_array.iter() {
            let v: f64 = value.ok_or(ArrowConversionError::NullValues)?;
            data.push(v);
        }
    }

    // Reshape from column-major (Arrow) to row-major (ndarray)
    let mut reshaped = Vec::with_capacity(num_rows * num_cols);
    for row in 0..num_rows {
        for col in 0..num_cols {
            reshaped.push(data[col * num_rows + row]);
        }
    }

    Array2::from_shape_vec((num_rows, num_cols), reshaped)
        .map_err(|e| ArrowConversionError::InvalidShape(format!("{:?}", e)))
}

/// Convert nalgebra DMatrix to RecordBatch
pub fn nalgebra_to_record_batch(
    matrix: &DMatrix<f64>,
) -> Result<RecordBatch, ArrowConversionError> {
    let (rows, cols) = matrix.shape();

    if rows == 0 || cols == 0 {
        return Err(ArrowConversionError::EmptyArray);
    }

    let mut columns: Vec<Arc<dyn Array>> = Vec::with_capacity(cols);
    let mut fields = Vec::with_capacity(cols);

    for col in 0..cols {
        let values: Vec<Option<f64>> = (0..rows).map(|row| Some(matrix[(row, col)])).collect();
        let arr = Arc::new(Float64Array::from_iter(values)) as Arc<dyn Array>;
        columns.push(arr);
        fields.push(Field::new(format!("col_{}", col), DataType::Float64, false));
    }

    let schema = Arc::new(Schema::new(fields));
    RecordBatch::try_new(schema, columns)
        .map_err(|e| ArrowConversionError::InvalidShape(e.to_string()))
}

/// Convert ndarray Array2 to RecordBatch
pub fn ndarray_to_record_batch(array: &Array2<f64>) -> Result<RecordBatch, ArrowConversionError> {
    let (rows, cols) = array.dim();

    if rows == 0 || cols == 0 {
        return Err(ArrowConversionError::EmptyArray);
    }

    let mut columns: Vec<Arc<dyn Array>> = Vec::with_capacity(cols);
    let mut fields = Vec::with_capacity(cols);

    for col in 0..cols {
        let values: Vec<Option<f64>> = (0..rows).map(|row| Some(array[[row, col]])).collect();
        let arr = Arc::new(Float64Array::from_iter(values)) as Arc<dyn Array>;
        columns.push(arr);
        fields.push(Field::new(format!("col_{}", col), DataType::Float64, false));
    }

    let schema = Arc::new(Schema::new(fields));
    RecordBatch::try_new(schema, columns)
        .map_err(|e| ArrowConversionError::InvalidShape(e.to_string()))
}

/// Convert Float64Array to nalgebra DVector
pub fn float64_array_to_dvector(
    array: &PrimitiveArray<Float64Type>,
) -> Result<DVector<f64>, ArrowConversionError> {
    if array.is_empty() {
        return Err(ArrowConversionError::EmptyArray);
    }

    let mut values = Vec::with_capacity(array.len());
    for value in array.iter() {
        let v: f64 = value.ok_or(ArrowConversionError::NullValues)?;
        values.push(v);
    }

    Ok(DVector::from_vec(values))
}

/// Convert nalgebra DVector to Float64Array
pub fn dvector_to_float64_array(vector: &DVector<f64>) -> Float64Array {
    let values: Vec<Option<f64>> = vector.iter().map(|&v| Some(v)).collect();
    Float64Array::from_iter(values)
}

/// Convert Float64Array to ndarray Array1
pub fn float64_array_to_array1(
    array: &PrimitiveArray<Float64Type>,
) -> Result<Array1<f64>, ArrowConversionError> {
    if array.is_empty() {
        return Err(ArrowConversionError::EmptyArray);
    }

    let mut values = Vec::with_capacity(array.len());
    for value in array.iter() {
        let v: f64 = value.ok_or(ArrowConversionError::NullValues)?;
        values.push(v);
    }

    Ok(Array1::from_vec(values))
}

/// Convert ndarray Array1 to Float64Array
pub fn array1_to_float64_array(array: &Array1<f64>) -> Float64Array {
    let values: Vec<Option<f64>> = array.iter().map(|&v| Some(v)).collect();
    Float64Array::from_iter(values)
}

/// Compute Frobenius norm of a RecordBatch (as matrix)
pub fn frobenius_norm_arrow(batch: &RecordBatch) -> Result<f64, ArrowConversionError> {
    let array = record_batch_to_ndarray(batch)?;
    Ok(crate::utils::frobenius_norm(&array))
}

/// Compute spectral norm (largest singular value) of a RecordBatch
pub fn spectral_norm_arrow(batch: &RecordBatch) -> Result<f64, ArrowConversionError> {
    let array = record_batch_to_ndarray(batch)?;
    Ok(crate::utils::spectral_norm(&array))
}

/// Check if two RecordBatches are approximately equal as matrices
pub fn matrix_approx_eq_arrow(
    a: &RecordBatch,
    b: &RecordBatch,
    epsilon: f64,
) -> Result<bool, ArrowConversionError> {
    let arr_a = record_batch_to_ndarray(a)?;
    let arr_b = record_batch_to_ndarray(b)?;
    Ok(crate::utils::matrix_approx_eq(&arr_a, &arr_b, epsilon))
}
