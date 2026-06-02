//! Exact brute-force k-nearest-neighbors over a full corpus.
//!
//! Intended for small corpora, evaluation, and golden tests where an exact answer is the
//! reference an approximate index is graded against. It scores every query against every corpus
//! row with [`query_corpus_scores_view`](crate::similarity) and selects the best `k` per query
//! with [`top_k`](crate::topk::top_k). This is `O(n_queries * n_corpus * dim)`; for production
//! recall over large corpora use an ANN index and then [`rerank`](crate::rerank::rerank) the
//! candidates.

use nabled_core::scalar::NabledReal;
use ndarray::ArrayView2;

use crate::error::EmbeddingError;
use crate::similarity::{Metric, query_corpus_scores_view};
use crate::topk::{Neighbor, top_k};

/// Compute the best `k` corpus neighbors for every query row under `metric`.
///
/// Returns one best-first neighbor list per query row (outer length `queries.nrows()`). `k` is
/// clamped to the corpus size.
///
/// # Errors
/// Returns [`EmbeddingError::EmptyInput`] for empty inputs,
/// [`EmbeddingError::DimensionMismatch`] when feature dimensions differ, and
/// [`EmbeddingError::ZeroNorm`] for cosine scoring against zero-norm rows.
pub fn brute_force_knn<T: NabledReal>(
    queries: &ArrayView2<'_, T>,
    corpus: &ArrayView2<'_, T>,
    k: usize,
    metric: Metric,
) -> Result<Vec<Vec<Neighbor<T>>>, EmbeddingError> {
    let scores = query_corpus_scores_view(queries, corpus, metric)?;
    let higher_is_better = metric.higher_is_better();
    Ok(scores.outer_iter().map(|row| top_k(row, k, higher_is_better)).collect())
}

#[cfg(test)]
mod tests {
    use ndarray::arr2;

    use super::*;

    #[test]
    fn knn_cosine_returns_per_query_lists() {
        let queries = arr2(&[[1.0_f64, 0.0], [0.0, 1.0]]);
        let corpus = arr2(&[[1.0_f64, 0.0], [0.0, 1.0], [0.9, 0.1]]);
        let result = brute_force_knn(&queries.view(), &corpus.view(), 2, Metric::Cosine).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0][0].index, 0);
        assert_eq!(result[1][0].index, 1);
    }

    #[test]
    fn knn_l2_finds_nearest() {
        let queries = arr2(&[[0.0_f64, 0.0]]);
        let corpus = arr2(&[[5.0_f64, 5.0], [0.0, 1.0]]);
        let result = brute_force_knn(&queries.view(), &corpus.view(), 1, Metric::L2).unwrap();
        assert_eq!(result[0][0].index, 1);
    }

    #[test]
    fn knn_f32_is_consistent() {
        let queries = arr2(&[[1.0_f32, 0.0]]);
        let corpus = arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]);
        let result = brute_force_knn(&queries.view(), &corpus.view(), 1, Metric::Dot).unwrap();
        assert_eq!(result[0][0].index, 0);
    }

    #[test]
    fn knn_clamps_k_to_corpus() {
        let queries = arr2(&[[1.0_f64, 0.0]]);
        let corpus = arr2(&[[1.0_f64, 0.0], [0.0, 1.0]]);
        let result = brute_force_knn(&queries.view(), &corpus.view(), 50, Metric::Cosine).unwrap();
        assert_eq!(result[0].len(), 2);
    }

    #[test]
    fn knn_reports_dimension_mismatch() {
        let queries = arr2(&[[1.0_f64, 0.0]]);
        let corpus = arr2(&[[1.0_f64, 0.0, 0.0]]);
        assert_eq!(
            brute_force_knn(&queries.view(), &corpus.view(), 1, Metric::Cosine),
            Err(EmbeddingError::DimensionMismatch)
        );
    }
}
