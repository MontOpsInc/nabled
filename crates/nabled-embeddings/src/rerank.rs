//! Exact single-query rerank: score candidates, then select the best `k`.
//!
//! This is the headline retrieval entrypoint and the exact step that follows an approximate
//! nearest-neighbor (ANN) recall stage: given a small candidate set (for example the top-N returned
//! by a vector store), it recomputes exact [`Metric`] scores and returns the best `k` in
//! best-first order. It composes [`query_corpus_scores_view`](crate::similarity) with
//! [`top_k`](crate::topk::top_k), passing the metric's ranking polarity so distances and
//! similarities are both handled correctly.

use nabled_core::scalar::NabledReal;
use ndarray::{ArrayView1, ArrayView2, Axis};

use crate::error::EmbeddingError;
use crate::similarity::{Metric, query_corpus_scores_view};
use crate::topk::{Neighbor, top_k};

/// Rerank `candidates` against a single `query`, returning the best `k` neighbors.
///
/// `k` is clamped to the number of candidates. The result is best-first under `metric` (highest
/// similarity for cosine/dot, smallest distance for L2).
///
/// # Errors
/// Returns [`EmbeddingError::EmptyInput`] for empty inputs,
/// [`EmbeddingError::DimensionMismatch`] when `query` length differs from the candidate width, and
/// [`EmbeddingError::ZeroNorm`] for cosine scoring against zero-norm rows.
pub fn rerank<T: NabledReal>(
    query: &ArrayView1<'_, T>,
    candidates: &ArrayView2<'_, T>,
    k: usize,
    metric: Metric,
) -> Result<Vec<Neighbor<T>>, EmbeddingError> {
    let query_rows = query.view().insert_axis(Axis(0));
    let scores = query_corpus_scores_view(&query_rows, candidates, metric)?;
    let row = scores.row(0);
    Ok(top_k(row, k, metric.higher_is_better()))
}

#[cfg(test)]
mod tests {
    use ndarray::{arr1, arr2};

    use super::*;

    #[test]
    fn rerank_cosine_returns_best_first() {
        let candidates = arr2(&[[1.0_f64, 0.0], [0.0, 1.0], [0.9, 0.1]]);
        let query = arr1(&[1.0_f64, 0.0]);
        let result = rerank(&query.view(), &candidates.view(), 2, Metric::Cosine).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].index, 0);
        assert_eq!(result[1].index, 2);
    }

    #[test]
    fn rerank_default_metric_is_cosine() {
        let candidates = arr2(&[[1.0_f64, 0.0], [0.0, 1.0]]);
        let query = arr1(&[1.0_f64, 0.0]);
        let result = rerank(&query.view(), &candidates.view(), 1, Metric::default()).unwrap();
        assert_eq!(result[0].index, 0);
    }

    #[test]
    fn rerank_l2_selects_nearest() {
        let candidates = arr2(&[[10.0_f64, 10.0], [0.0, 1.0], [5.0, 5.0]]);
        let query = arr1(&[0.0_f64, 0.0]);
        let result = rerank(&query.view(), &candidates.view(), 1, Metric::L2).unwrap();
        assert_eq!(result[0].index, 1);
    }

    #[test]
    fn rerank_dot_favors_larger_norm() {
        let candidates = arr2(&[[1.0_f64, 0.0], [2.0, 0.0]]);
        let query = arr1(&[1.0_f64, 0.0]);
        let result = rerank(&query.view(), &candidates.view(), 1, Metric::Dot).unwrap();
        assert_eq!(result[0].index, 1);
        assert!((result[0].score - 2.0).abs() < 1e-12);
    }

    #[test]
    fn rerank_f32_is_consistent() {
        let candidates = arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]);
        let query = arr1(&[1.0_f32, 0.0]);
        let result = rerank(&query.view(), &candidates.view(), 2, Metric::Cosine).unwrap();
        assert_eq!(result[0].index, 0);
    }

    #[test]
    fn rerank_clamps_k() {
        let candidates = arr2(&[[1.0_f64, 0.0], [0.0, 1.0]]);
        let query = arr1(&[1.0_f64, 0.0]);
        let result = rerank(&query.view(), &candidates.view(), 99, Metric::Cosine).unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn rerank_reports_dimension_mismatch() {
        let candidates = arr2(&[[1.0_f64, 0.0, 0.0]]);
        let query = arr1(&[1.0_f64, 0.0]);
        assert_eq!(
            rerank(&query.view(), &candidates.view(), 1, Metric::Cosine),
            Err(EmbeddingError::DimensionMismatch)
        );
    }
}
