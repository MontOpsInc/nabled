//! Row-wise L2 normalization for embedding matrices.
//!
//! Normalizing rows to unit length is the canonical pre-step for cosine retrieval: once both
//! query and corpus are normalized, a plain dot product equals cosine similarity. These functions
//! delegate to `nabled_linalg::vector::batched_normalize*` and add nothing but the embedding-domain
//! error type. Zero-norm rows are left at the origin (divided by a small epsilon) rather than
//! erroring, matching the underlying kernel.

use nabled_core::scalar::NabledReal;
use nabled_linalg::vector;
use ndarray::{Array2, ArrayBase, ArrayView2, Data, DataMut, Ix2};

use crate::error::EmbeddingError;

/// Normalize each row of `rows` to unit L2 length.
///
/// # Errors
/// Returns [`EmbeddingError::EmptyInput`] when `rows` is empty.
pub fn normalize_rows<T: NabledReal>(rows: &Array2<T>) -> Result<Array2<T>, EmbeddingError> {
    Ok(vector::batched_normalize(rows)?)
}

/// Normalize each row of a matrix view to unit L2 length.
///
/// # Errors
/// Returns [`EmbeddingError::EmptyInput`] when `rows` is empty.
pub fn normalize_rows_view<T: NabledReal>(
    rows: &ArrayView2<'_, T>,
) -> Result<Array2<T>, EmbeddingError> {
    Ok(vector::batched_normalize_view(rows)?)
}

/// Normalize each row into a caller-provided `output` buffer.
///
/// # Errors
/// Returns [`EmbeddingError::EmptyInput`] when `rows` is empty or
/// [`EmbeddingError::DimensionMismatch`] when `output` is not the same shape as `rows`.
pub fn normalize_rows_into<T, S>(
    rows: &ArrayBase<S, Ix2>,
    output: &mut ArrayBase<impl DataMut<Elem = T>, Ix2>,
) -> Result<(), EmbeddingError>
where
    T: NabledReal,
    S: Data<Elem = T>,
{
    vector::batched_normalize_into(rows, output)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use ndarray::{Array2, arr2};

    use super::*;

    fn unit_norm<T: NabledReal>(row: &ndarray::ArrayView1<'_, T>) -> T {
        row.iter().copied().fold(T::zero(), |acc, v| acc + v * v).sqrt()
    }

    #[test]
    fn normalize_rows_yields_unit_rows_f64() {
        let rows = arr2(&[[3.0_f64, 4.0], [1.0, 0.0]]);
        let normalized = normalize_rows(&rows).unwrap();
        assert!((unit_norm(&normalized.row(0)) - 1.0).abs() < 1e-12);
        assert!((normalized[[0, 0]] - 0.6).abs() < 1e-12);
        assert!((normalized[[0, 1]] - 0.8).abs() < 1e-12);
    }

    #[test]
    fn normalize_rows_yields_unit_rows_f32() {
        let rows = arr2(&[[3.0_f32, 4.0]]);
        let normalized = normalize_rows(&rows).unwrap();
        assert!((unit_norm(&normalized.row(0)) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn normalize_rows_view_matches_owned() {
        let rows = arr2(&[[3.0_f64, 4.0], [0.0, 2.0]]);
        let owned = normalize_rows(&rows).unwrap();
        let viewed = normalize_rows_view(&rows.view()).unwrap();
        assert_eq!(owned, viewed);
    }

    #[test]
    fn normalize_rows_into_matches_allocating() {
        let rows = arr2(&[[3.0_f64, 4.0], [0.0, 2.0]]);
        let expected = normalize_rows(&rows).unwrap();
        let mut output = Array2::<f64>::zeros(rows.dim());
        normalize_rows_into(&rows, &mut output).unwrap();
        assert_eq!(output, expected);
    }

    #[test]
    fn normalize_rows_into_rejects_shape_mismatch() {
        let rows = arr2(&[[3.0_f64, 4.0]]);
        let mut output = Array2::<f64>::zeros((2, 2));
        assert_eq!(normalize_rows_into(&rows, &mut output), Err(EmbeddingError::DimensionMismatch));
    }

    #[test]
    fn normalize_rows_rejects_empty() {
        let rows = Array2::<f64>::zeros((0, 0));
        assert_eq!(normalize_rows(&rows), Err(EmbeddingError::EmptyInput));
    }

    #[test]
    fn normalize_rows_keeps_zero_row_finite() {
        let rows = arr2(&[[0.0_f64, 0.0]]);
        let normalized = normalize_rows(&rows).unwrap();
        assert!(normalized.iter().all(|v| v.is_finite()));
    }
}
