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
use crate::knn::brute_force_knn;
use crate::similarity::{Metric, query_corpus_scores_view};
use crate::topk::{Neighbor, NeighborWithId, attach_ids, top_k};

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

/// Rerank `candidates` against a single `query`, returning the best `k` neighbors carrying ids.
///
/// This is the id-carrying variant of [`rerank`]: `ids` supplies one stable identifier per
/// candidate row (so `ids.len()` must equal `candidates.nrows()`), and each returned
/// [`NeighborWithId`] reports both the local candidate index and its mapped id. It is the genuine
/// new value over [`rerank`] for pipelines that recall global ids upstream and need them threaded
/// back through the exact rerank without a separate index-to-id join.
///
/// # Errors
/// Returns [`EmbeddingError::DimensionMismatch`] when `ids.len()` differs from the candidate count,
/// plus every error condition of [`rerank`].
pub fn rerank_with_ids<T: NabledReal, Id: Copy>(
    query: &ArrayView1<'_, T>,
    candidates: &ArrayView2<'_, T>,
    ids: &[Id],
    k: usize,
    metric: Metric,
) -> Result<Vec<NeighborWithId<T, Id>>, EmbeddingError> {
    let n_candidates = candidates.nrows();
    if ids.len() != n_candidates {
        return Err(EmbeddingError::DimensionMismatch);
    }
    let neighbors = rerank(query, candidates, k, metric)?;
    attach_ids(neighbors, ids, n_candidates)
}

/// Rerank a shared `corpus` for every `query` row, returning per-query id-carrying neighbors.
///
/// The batch path composes the existing [`brute_force_knn`] scoring path (many queries versus one
/// shared corpus) and then maps each local corpus index to its caller-supplied id from `ids`
/// (`ids.len()` must equal `corpus.nrows()`). It deliberately does not introduce a bare
/// `batch_rerank`: that would duplicate [`brute_force_knn`]; the new capability here is id mapping.
///
/// # Errors
/// Returns [`EmbeddingError::DimensionMismatch`] when `ids.len()` differs from the corpus row
/// count, plus every error condition of [`brute_force_knn`].
pub fn batch_rerank_with_ids<T: NabledReal, Id: Copy>(
    queries: &ArrayView2<'_, T>,
    corpus: &ArrayView2<'_, T>,
    ids: &[Id],
    k: usize,
    metric: Metric,
) -> Result<Vec<Vec<NeighborWithId<T, Id>>>, EmbeddingError> {
    let n_corpus = corpus.nrows();
    if ids.len() != n_corpus {
        return Err(EmbeddingError::DimensionMismatch);
    }
    let lists = brute_force_knn(queries, corpus, k, metric)?;
    lists.into_iter().map(|list| attach_ids(list, ids, n_corpus)).collect()
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

    #[test]
    fn rerank_with_ids_maps_ids_in_ranked_order() {
        let candidates = arr2(&[[1.0_f64, 0.0], [0.0, 1.0], [0.9, 0.1]]);
        let query = arr1(&[1.0_f64, 0.0]);
        let ids = [100_i64, 200, 300];
        let result =
            rerank_with_ids(&query.view(), &candidates.view(), &ids, 2, Metric::Cosine).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].index, 0);
        assert_eq!(result[0].id, 100);
        assert_eq!(result[1].index, 2);
        assert_eq!(result[1].id, 300);
    }

    #[test]
    fn rerank_with_ids_f32_matches_rerank_order() {
        let candidates = arr2(&[[1.0_f32, 0.0], [0.0, 1.0], [0.9, 0.1]]);
        let query = arr1(&[1.0_f32, 0.0]);
        let ids = [7_u32, 8, 9];
        let plain = rerank(&query.view(), &candidates.view(), 3, Metric::Cosine).unwrap();
        let with_ids =
            rerank_with_ids(&query.view(), &candidates.view(), &ids, 3, Metric::Cosine).unwrap();
        for (a, b) in plain.iter().zip(with_ids.iter()) {
            assert_eq!(a.index, b.index);
            assert!((a.score - b.score).abs() < 1e-6);
            assert_eq!(b.id, ids[b.index]);
        }
    }

    #[test]
    fn rerank_with_ids_rejects_length_mismatch() {
        let candidates = arr2(&[[1.0_f64, 0.0], [0.0, 1.0]]);
        let query = arr1(&[1.0_f64, 0.0]);
        let ids = [1_i64];
        assert_eq!(
            rerank_with_ids(&query.view(), &candidates.view(), &ids, 1, Metric::Cosine),
            Err(EmbeddingError::DimensionMismatch)
        );
    }

    #[test]
    fn batch_rerank_with_ids_maps_per_query() {
        let queries = arr2(&[[1.0_f64, 0.0], [0.0, 1.0]]);
        let corpus = arr2(&[[1.0_f64, 0.0], [0.0, 1.0], [0.9, 0.1]]);
        let ids = [11_i64, 22, 33];
        let result =
            batch_rerank_with_ids(&queries.view(), &corpus.view(), &ids, 2, Metric::Cosine)
                .unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0][0].index, 0);
        assert_eq!(result[0][0].id, 11);
        assert_eq!(result[1][0].index, 1);
        assert_eq!(result[1][0].id, 22);
    }

    #[test]
    fn batch_rerank_with_ids_matches_brute_force_knn() {
        let queries = arr2(&[[1.0_f64, 0.0], [0.2, 0.9]]);
        let corpus = arr2(&[[1.0_f64, 0.0], [0.0, 1.0], [0.9, 0.1], [0.3, 0.7]]);
        let ids = [5_i64, 6, 7, 8];
        let knn = brute_force_knn(&queries.view(), &corpus.view(), 3, Metric::Cosine).unwrap();
        let with_ids =
            batch_rerank_with_ids(&queries.view(), &corpus.view(), &ids, 3, Metric::Cosine)
                .unwrap();
        for (klist, ilist) in knn.iter().zip(with_ids.iter()) {
            for (kn, inb) in klist.iter().zip(ilist.iter()) {
                assert_eq!(kn.index, inb.index);
                assert_eq!(inb.id, ids[inb.index]);
            }
        }
    }

    #[test]
    fn batch_rerank_with_ids_rejects_length_mismatch() {
        let queries = arr2(&[[1.0_f64, 0.0]]);
        let corpus = arr2(&[[1.0_f64, 0.0], [0.0, 1.0]]);
        let ids = [1_i64, 2, 3];
        assert_eq!(
            batch_rerank_with_ids(&queries.view(), &corpus.view(), &ids, 1, Metric::Cosine),
            Err(EmbeddingError::DimensionMismatch)
        );
    }
}
