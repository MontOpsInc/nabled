//! Reusable corpus workspace for repeated queries against a static corpus.
//!
//! Retrieval serving evaluates many queries against the *same* corpus. The stateless
//! [`query_corpus_scores`](crate::similarity) path recomputes the corpus's contribution (its L2
//! norms for cosine) on every call. [`CorpusWorkspace`] precomputes that work **once** at
//! [`build`](CorpusWorkspace::build) time and reuses it across calls, mirroring the
//! `PairwiseCosineWorkspace::ensure_dims` precedent in `nabled-linalg`.
//!
//! # Chosen approach (cosine)
//!
//! `nabled-linalg` has no "corpus norms already provided" cosine entrypoint, so the workspace owns
//! the corpus together with its precomputed row norms and scores cosine as
//! `dot(query, corpus) / (||query|| * ||corpus||)`, supplying the cached corpus norms directly.
//! This reproduces the stateless cosine kernel bit-for-bit (including its zero-norm rejection)
//! while skipping the per-call corpus-norm recompute. `Dot` and `L2` own the corpus copy and
//! delegate to the same kernels as the stateless path (there is no norm to cache, but the owned
//! corpus still removes a borrow obligation from the caller across repeated queries).

use nabled_core::scalar::NabledReal;
use nabled_linalg::{matrix, vector};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

use crate::error::EmbeddingError;
use crate::similarity::Metric;
use crate::topk::{Neighbor, top_k};

/// Precomputed, metric-specific corpus state.
#[derive(Debug, Clone)]
enum Prepared<T> {
    /// Cosine caches the corpus row L2 norms so they are not recomputed per query.
    Cosine { corpus_norms: Array1<T> },
    /// Dot and L2 have no reusable corpus-side scalar to cache.
    Plain,
}

/// A corpus prepared once for repeated scoring under a fixed [`Metric`].
///
/// Build it with [`CorpusWorkspace::build`], then score many queries with
/// [`query_corpus_scores`](CorpusWorkspace::query_corpus_scores) /
/// [`query_corpus_scores_into`](CorpusWorkspace::query_corpus_scores_into) or rerank a single query
/// with [`rerank_with`](CorpusWorkspace::rerank_with). The corpus and metric are fixed for the life
/// of the workspace.
#[derive(Debug, Clone)]
pub struct CorpusWorkspace<T> {
    metric: Metric,
    corpus: Array2<T>,
    prepared: Prepared<T>,
}

impl<T: NabledReal> CorpusWorkspace<T> {
    /// Precompute reusable corpus state for `metric`.
    ///
    /// For [`Metric::Cosine`] this computes and caches the corpus row norms (rejecting zero-norm
    /// rows up front, exactly as the stateless cosine path would).
    ///
    /// # Errors
    /// Returns [`EmbeddingError::EmptyInput`] when `corpus` is empty and
    /// [`EmbeddingError::ZeroNorm`] when a cosine corpus contains a zero-norm row.
    pub fn build(corpus: &ArrayView2<'_, T>, metric: Metric) -> Result<Self, EmbeddingError> {
        if corpus.is_empty() {
            return Err(EmbeddingError::EmptyInput);
        }
        let prepared = match metric {
            Metric::Cosine => {
                let corpus_norms = vector::batched_l2_norm_view(corpus)?;
                if corpus_norms.iter().any(|&n| n <= T::epsilon()) {
                    return Err(EmbeddingError::ZeroNorm);
                }
                Prepared::Cosine { corpus_norms }
            }
            Metric::Dot | Metric::L2 => Prepared::Plain,
        };
        Ok(Self { metric, corpus: corpus.to_owned(), prepared })
    }

    /// The metric this workspace was built for.
    #[must_use]
    pub const fn metric(&self) -> Metric { self.metric }

    /// Number of corpus rows.
    #[must_use]
    pub fn len(&self) -> usize { self.corpus.nrows() }

    /// Whether the corpus has no rows. Always `false` for a successfully built workspace.
    #[must_use]
    pub fn is_empty(&self) -> bool { self.corpus.nrows() == 0 }

    /// Feature dimension of the corpus rows.
    #[must_use]
    pub fn dim(&self) -> usize { self.corpus.ncols() }

    /// Score every `queries` row against the cached corpus, reusing precomputed corpus state.
    ///
    /// Returns a `(queries.nrows(), corpus_len)` matrix identical to the stateless
    /// [`query_corpus_scores`](crate::similarity::query_corpus_scores) result.
    ///
    /// # Errors
    /// Returns [`EmbeddingError::EmptyInput`] for empty `queries`,
    /// [`EmbeddingError::DimensionMismatch`] when the feature dimension differs, and
    /// [`EmbeddingError::ZeroNorm`] for cosine queries with zero-norm rows.
    pub fn query_corpus_scores(
        &self,
        queries: &ArrayView2<'_, T>,
    ) -> Result<Array2<T>, EmbeddingError> {
        let mut out = Array2::<T>::zeros((queries.nrows(), self.corpus.nrows()));
        self.query_corpus_scores_into(queries, &mut out)?;
        Ok(out)
    }

    /// Score every `queries` row against the cached corpus into a caller-provided `out` buffer.
    ///
    /// `out` must be shaped `(queries.nrows(), corpus_len)`.
    ///
    /// # Errors
    /// See [`query_corpus_scores`](CorpusWorkspace::query_corpus_scores); also returns
    /// [`EmbeddingError::DimensionMismatch`] when `out` is mis-sized.
    pub fn query_corpus_scores_into(
        &self,
        queries: &ArrayView2<'_, T>,
        out: &mut Array2<T>,
    ) -> Result<(), EmbeddingError> {
        if queries.is_empty() {
            return Err(EmbeddingError::EmptyInput);
        }
        if queries.ncols() != self.corpus.ncols() {
            return Err(EmbeddingError::DimensionMismatch);
        }
        if out.dim() != (queries.nrows(), self.corpus.nrows()) {
            return Err(EmbeddingError::DimensionMismatch);
        }

        match &self.prepared {
            Prepared::Cosine { corpus_norms } => {
                let query_norms = vector::batched_l2_norm_view(queries)?;
                if query_norms.iter().any(|&n| n <= T::epsilon()) {
                    return Err(EmbeddingError::ZeroNorm);
                }
                matrix::matmat_view_into(queries, &self.corpus.t(), out.view_mut())?;
                out.outer_iter_mut().zip(query_norms.iter()).for_each(|(mut row, &qn)| {
                    row.iter_mut().zip(corpus_norms.iter()).for_each(|(value, &cn)| {
                        *value /= qn * cn;
                    });
                });
            }
            Prepared::Plain if self.metric == Metric::Dot => {
                matrix::matmat_view_into(queries, &self.corpus.t(), out.view_mut())?;
            }
            Prepared::Plain => {
                vector::pairwise_l2_distance_view_into(queries, &self.corpus.view(), out)?;
            }
        }
        Ok(())
    }

    /// Rerank the cached corpus against a single `query`, returning the best `k` neighbors.
    ///
    /// Equivalent to [`rerank`](crate::rerank::rerank) against this workspace's corpus and metric,
    /// but reusing the precomputed corpus state.
    ///
    /// # Errors
    /// See [`query_corpus_scores`](CorpusWorkspace::query_corpus_scores).
    pub fn rerank_with(
        &self,
        query: &ArrayView1<'_, T>,
        k: usize,
    ) -> Result<Vec<Neighbor<T>>, EmbeddingError> {
        let query_rows = query.view().insert_axis(Axis(0));
        let scores = self.query_corpus_scores(&query_rows)?;
        Ok(top_k(scores.row(0), k, self.metric.higher_is_better()))
    }

    /// Exact brute-force kNN over the cached corpus for every `queries` row.
    ///
    /// Per-query best-first neighbor lists, reusing the precomputed corpus state.
    ///
    /// # Errors
    /// See [`query_corpus_scores`](CorpusWorkspace::query_corpus_scores).
    pub fn knn_with(
        &self,
        queries: &ArrayView2<'_, T>,
        k: usize,
    ) -> Result<Vec<Vec<Neighbor<T>>>, EmbeddingError> {
        let scores = self.query_corpus_scores(queries)?;
        let higher_is_better = self.metric.higher_is_better();
        Ok(scores.outer_iter().map(|row| top_k(row, k, higher_is_better)).collect())
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{arr1, arr2};

    use super::*;
    use crate::knn::brute_force_knn;
    use crate::rerank::rerank;
    use crate::similarity::query_corpus_scores_view;

    fn corpus_f64() -> Array2<f64> {
        arr2(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.2, 0.3, 0.9]])
    }

    fn queries_f64() -> Array2<f64> {
        arr2(&[[1.0, 0.0, 0.0], [0.1, 0.2, 0.9]])
    }

    #[test]
    fn workspace_scores_match_stateless_for_all_metrics() {
        let corpus = corpus_f64();
        let queries = queries_f64();
        for metric in [Metric::Cosine, Metric::Dot, Metric::L2] {
            let ws = CorpusWorkspace::build(&corpus.view(), metric).unwrap();
            let cached = ws.query_corpus_scores(&queries.view()).unwrap();
            let stateless =
                query_corpus_scores_view(&queries.view(), &corpus.view(), metric).unwrap();
            for (lhs, rhs) in cached.iter().zip(stateless.iter()) {
                assert!((lhs - rhs).abs() < 1e-12, "{metric:?}: {lhs} vs {rhs}");
            }
        }
    }

    #[test]
    fn workspace_scores_match_stateless_f32() {
        let corpus = arr2(&[[1.0_f32, 0.0], [0.0, 1.0], [0.7, 0.7]]);
        let queries = arr2(&[[1.0_f32, 0.0], [0.3, 0.4]]);
        for metric in [Metric::Cosine, Metric::Dot, Metric::L2] {
            let ws = CorpusWorkspace::build(&corpus.view(), metric).unwrap();
            let cached = ws.query_corpus_scores(&queries.view()).unwrap();
            let stateless =
                query_corpus_scores_view(&queries.view(), &corpus.view(), metric).unwrap();
            for (lhs, rhs) in cached.iter().zip(stateless.iter()) {
                assert!((lhs - rhs).abs() < 1e-5, "{metric:?}: {lhs} vs {rhs}");
            }
        }
    }

    #[test]
    fn workspace_reuse_across_queries_is_stable() {
        let corpus = corpus_f64();
        let ws = CorpusWorkspace::build(&corpus.view(), Metric::Cosine).unwrap();
        let q1 = arr1(&[1.0_f64, 0.0, 0.0]);
        let q2 = arr1(&[0.0_f64, 1.0, 0.0]);
        // Repeated single-query reuse must equal the stateless rerank for each query.
        let r1 = ws.rerank_with(&q1.view(), 2).unwrap();
        let r2 = ws.rerank_with(&q2.view(), 2).unwrap();
        let e1 = rerank(&q1.view(), &corpus.view(), 2, Metric::Cosine).unwrap();
        let e2 = rerank(&q2.view(), &corpus.view(), 2, Metric::Cosine).unwrap();
        assert_eq!(r1[0].index, e1[0].index);
        assert_eq!(r2[0].index, e2[0].index);
        // Re-run the first query again to confirm the workspace is not mutated by use.
        let r1_again = ws.rerank_with(&q1.view(), 2).unwrap();
        assert_eq!(r1[0].index, r1_again[0].index);
    }

    #[test]
    fn workspace_knn_matches_brute_force() {
        let corpus = corpus_f64();
        let queries = queries_f64();
        let ws = CorpusWorkspace::build(&corpus.view(), Metric::L2).unwrap();
        let cached = ws.knn_with(&queries.view(), 3).unwrap();
        let expected = brute_force_knn(&queries.view(), &corpus.view(), 3, Metric::L2).unwrap();
        for (clist, elist) in cached.iter().zip(expected.iter()) {
            for (cn, en) in clist.iter().zip(elist.iter()) {
                assert_eq!(cn.index, en.index);
                assert!((cn.score - en.score).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn workspace_metadata_is_exposed() {
        let corpus = corpus_f64();
        let ws = CorpusWorkspace::build(&corpus.view(), Metric::Dot).unwrap();
        assert_eq!(ws.metric(), Metric::Dot);
        assert_eq!(ws.len(), 4);
        assert_eq!(ws.dim(), 3);
        assert!(!ws.is_empty());
    }

    #[test]
    fn build_rejects_empty_corpus() {
        let corpus = Array2::<f64>::zeros((0, 3));
        assert_eq!(
            CorpusWorkspace::build(&corpus.view(), Metric::Cosine).err(),
            Some(EmbeddingError::EmptyInput)
        );
    }

    #[test]
    fn build_rejects_cosine_zero_norm_row() {
        let corpus = arr2(&[[1.0_f64, 0.0], [0.0, 0.0]]);
        assert_eq!(
            CorpusWorkspace::build(&corpus.view(), Metric::Cosine).err(),
            Some(EmbeddingError::ZeroNorm)
        );
    }

    #[test]
    fn query_rejects_dimension_mismatch() {
        let corpus = corpus_f64();
        let ws = CorpusWorkspace::build(&corpus.view(), Metric::Dot).unwrap();
        let queries = arr2(&[[1.0_f64, 0.0]]);
        assert_eq!(
            ws.query_corpus_scores(&queries.view()).err(),
            Some(EmbeddingError::DimensionMismatch)
        );
    }

    #[test]
    fn query_rejects_cosine_zero_norm_query() {
        let corpus = corpus_f64();
        let ws = CorpusWorkspace::build(&corpus.view(), Metric::Cosine).unwrap();
        let queries = arr2(&[[0.0_f64, 0.0, 0.0]]);
        assert_eq!(
            ws.query_corpus_scores(&queries.view()).err(),
            Some(EmbeddingError::ZeroNorm)
        );
    }

    #[test]
    fn query_into_rejects_wrong_output_shape() {
        let corpus = corpus_f64();
        let ws = CorpusWorkspace::build(&corpus.view(), Metric::Dot).unwrap();
        let queries = queries_f64();
        let mut out = Array2::<f64>::zeros((1, 1));
        assert_eq!(
            ws.query_corpus_scores_into(&queries.view(), &mut out).err(),
            Some(EmbeddingError::DimensionMismatch)
        );
    }
}
