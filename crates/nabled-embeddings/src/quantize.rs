//! int8 row quantization for embedding matrices (encode/decode + dequantized scoring).
//!
//! This is an honest, minimal quantization layer: it shrinks an `f32` embedding matrix to one
//! `i8` per element plus one `f32` scale per row, and scores quantized matrices by **dequantizing
//! back to `f32` and reusing the existing kernels**. There is no native int8 kernel and no new
//! `nabled-linalg` surface; the win is ~4x smaller storage/transfer, traded against a small,
//! bounded precision loss (each value is rounded to one of 255 levels scaled per row).
//!
//! # Scheme
//!
//! Per-row **symmetric** quantization. For row `r` with max absolute value `amax`:
//! `scale[r] = amax / 127`, and each element is `round(x / scale[r])` clamped to `[-127, 127]`.
//! A fully-zero row gets `scale = 0` and all-zero codes, and dequantizes back to zeros. The signed
//! range is kept symmetric (`-127..=127`, leaving `-128` unused) so that `dequantize(quantize(x))`
//! has no sign bias.

use ndarray::{Array1, Array2, ArrayView2};

use crate::error::EmbeddingError;
use crate::similarity::{Metric, query_corpus_scores};

/// The largest magnitude code used by the symmetric int8 scheme.
const MAX_CODE: f32 = 127.0;

/// A row-quantized `f32` matrix: one `i8` code per element plus one `f32` scale per row.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedMatrix {
    data:   Array2<i8>,
    scales: Array1<f32>,
}

impl QuantizedMatrix {
    /// The `(n_rows, n_cols)` int8 code matrix.
    #[must_use]
    pub fn data(&self) -> &Array2<i8> { &self.data }

    /// The per-row `f32` scales (`len == n_rows`).
    #[must_use]
    pub fn scales(&self) -> &Array1<f32> { &self.scales }

    /// Number of rows.
    #[must_use]
    pub fn nrows(&self) -> usize { self.data.nrows() }

    /// Number of columns.
    #[must_use]
    pub fn ncols(&self) -> usize { self.data.ncols() }

    /// Reconstruct (decode) the approximate `f32` matrix as `code * scale[row]`.
    #[must_use]
    pub fn dequantize(&self) -> Array2<f32> { dequantize(self) }

    /// Construct a [`QuantizedMatrix`] from existing codes and scales (for example, decoded from a
    /// wire format).
    ///
    /// # Errors
    /// Returns [`EmbeddingError::EmptyInput`] when `data` is empty and
    /// [`EmbeddingError::DimensionMismatch`] when `scales.len() != data.nrows()`.
    pub fn from_parts(data: Array2<i8>, scales: Array1<f32>) -> Result<Self, EmbeddingError> {
        if data.is_empty() {
            return Err(EmbeddingError::EmptyInput);
        }
        if scales.len() != data.nrows() {
            return Err(EmbeddingError::DimensionMismatch);
        }
        Ok(Self { data, scales })
    }

    /// Score this quantized matrix as queries against a quantized `corpus` under `metric`.
    ///
    /// Implemented as dequantize-then-existing-kernel: both operands are decoded to `f32` and
    /// scored with [`query_corpus_scores`]. Results approximate the full-precision path within the
    /// quantization tolerance.
    ///
    /// # Errors
    /// Returns the same errors as [`query_corpus_scores`] (empty input, dimension mismatch, or
    /// cosine zero-norm rows).
    pub fn query_corpus_scores_quantized(
        &self,
        corpus: &QuantizedMatrix,
        metric: Metric,
    ) -> Result<Array2<f32>, EmbeddingError> {
        let queries = self.dequantize();
        let corpus = corpus.dequantize();
        query_corpus_scores(&queries, &corpus, metric)
    }
}

/// Quantize each row of `rows` to int8 with a per-row symmetric scale.
///
/// # Errors
/// Returns [`EmbeddingError::EmptyInput`] when `rows` is empty.
// Codes are clamped to `[-127, 127]` before the `as i8` cast, so truncation/sign-loss cannot occur
// for the values produced here.
#[allow(clippy::cast_possible_truncation)]
pub fn quantize_rows(rows: &ArrayView2<'_, f32>) -> Result<QuantizedMatrix, EmbeddingError> {
    if rows.is_empty() {
        return Err(EmbeddingError::EmptyInput);
    }
    let (n_rows, n_cols) = rows.dim();
    let mut data = Array2::<i8>::zeros((n_rows, n_cols));
    let mut scales = Array1::<f32>::zeros(n_rows);

    for (r, row) in rows.outer_iter().enumerate() {
        let amax = row.iter().fold(0.0_f32, |acc, &v| acc.max(v.abs()));
        if amax == 0.0 {
            // Zero row: scale stays 0, codes stay 0.
            continue;
        }
        let scale = amax / MAX_CODE;
        scales[r] = scale;
        for (c, &value) in row.iter().enumerate() {
            let code = (value / scale).round().clamp(-MAX_CODE, MAX_CODE);
            data[[r, c]] = code as i8;
        }
    }

    Ok(QuantizedMatrix { data, scales })
}

/// Decode a [`QuantizedMatrix`] back to its approximate `f32` values (`code * scale[row]`).
#[must_use]
pub fn dequantize(matrix: &QuantizedMatrix) -> Array2<f32> {
    let mut out = Array2::<f32>::zeros(matrix.data.dim());
    out.outer_iter_mut().zip(matrix.data.outer_iter()).zip(matrix.scales.iter()).for_each(
        |((mut out_row, code_row), &scale)| {
            out_row.iter_mut().zip(code_row.iter()).for_each(|(value, &code)| {
                *value = f32::from(code) * scale;
            });
        },
    );
    out
}

#[cfg(test)]
mod tests {
    use ndarray::arr2;

    use super::*;

    #[test]
    fn round_trip_is_within_scale_tolerance() {
        let rows = arr2(&[[1.0_f32, -2.0, 0.5], [0.1, 0.2, -0.3]]);
        let q = quantize_rows(&rows.view()).unwrap();
        let restored = q.dequantize();
        for (r, row) in rows.outer_iter().enumerate() {
            // Per-row reconstruction error is bounded by half a quantization step (scale/2).
            let half_step = q.scales()[r] / 2.0 + 1e-6;
            for (c, &orig) in row.iter().enumerate() {
                assert!(
                    (restored[[r, c]] - orig).abs() <= half_step,
                    "row {r} col {c}: {orig} vs {}",
                    restored[[r, c]]
                );
            }
        }
    }

    #[test]
    fn extreme_value_maps_to_max_code() {
        let rows = arr2(&[[3.0_f32, -3.0, 1.5]]);
        let q = quantize_rows(&rows.view()).unwrap();
        // The two extremes map to +/-127.
        assert_eq!(q.data()[[0, 0]], 127);
        assert_eq!(q.data()[[0, 1]], -127);
        assert!((q.scales()[0] - 3.0 / 127.0).abs() < 1e-9);
    }

    #[test]
    fn zero_row_is_handled() {
        let rows = arr2(&[[0.0_f32, 0.0], [1.0, 0.0]]);
        let q = quantize_rows(&rows.view()).unwrap();
        assert!(q.scales()[0].abs() < f32::EPSILON);
        assert_eq!(q.data()[[0, 0]], 0);
        assert_eq!(q.data()[[0, 1]], 0);
        let restored = q.dequantize();
        assert!(restored[[0, 0]].abs() < f32::EPSILON);
        assert!(restored[[0, 1]].abs() < f32::EPSILON);
    }

    #[test]
    fn dimensions_are_exposed() {
        let rows = arr2(&[[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let q = quantize_rows(&rows.view()).unwrap();
        assert_eq!(q.nrows(), 2);
        assert_eq!(q.ncols(), 3);
        assert_eq!(q.scales().len(), 2);
    }

    #[test]
    fn quantize_rejects_empty() {
        let rows = Array2::<f32>::zeros((0, 4));
        assert_eq!(quantize_rows(&rows.view()).err(), Some(EmbeddingError::EmptyInput));
    }

    #[test]
    fn from_parts_validates_shape() {
        let data = Array2::<i8>::zeros((2, 3));
        let scales = Array1::<f32>::zeros(3);
        assert_eq!(
            QuantizedMatrix::from_parts(data, scales).err(),
            Some(EmbeddingError::DimensionMismatch)
        );
    }

    #[test]
    fn from_parts_rejects_empty() {
        let data = Array2::<i8>::zeros((0, 0));
        let scales = Array1::<f32>::zeros(0);
        assert_eq!(
            QuantizedMatrix::from_parts(data, scales).err(),
            Some(EmbeddingError::EmptyInput)
        );
    }

    #[test]
    fn quantized_scoring_is_close_to_f32_path() {
        let queries = arr2(&[[0.8_f32, 0.1, 0.5], [0.2, 0.9, 0.4]]);
        let corpus = arr2(&[[0.7_f32, 0.2, 0.3], [0.1, 0.8, 0.6], [0.5, 0.5, 0.5]]);
        let qq = quantize_rows(&queries.view()).unwrap();
        let qc = quantize_rows(&corpus.view()).unwrap();
        for metric in [Metric::Cosine, Metric::Dot, Metric::L2] {
            let exact = query_corpus_scores(&queries, &corpus, metric).unwrap();
            let approx = qq.query_corpus_scores_quantized(&qc, metric).unwrap();
            for (lhs, rhs) in exact.iter().zip(approx.iter()) {
                assert!((lhs - rhs).abs() < 5e-2, "{metric:?}: exact {lhs} vs quantized {rhs}");
            }
        }
    }
}
