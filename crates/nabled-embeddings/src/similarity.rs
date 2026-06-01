//! Query-versus-corpus scoring across the three common retrieval metrics.
//!
//! [`query_corpus_scores`] returns a `(n_queries, n_corpus)` score matrix where entry `(i, j)` is
//! the [`Metric`] value between query `i` and corpus row `j`. Each metric delegates to an existing
//! `nabled-linalg` kernel rather than reimplementing it:
//!
//! - [`Metric::Cosine`] -> `vector::pairwise_cosine_similarity` (higher is better).
//! - [`Metric::L2`] -> `vector::pairwise_l2_distance` (lower is better; it is a distance).
//! - [`Metric::Dot`] -> `matrix::matmat(queries, corpus^T)` (higher is better).
//!
//! # Choosing a metric
//!
//! The math cannot pick the metric for you; choose it to match how the model was trained.
//! L2 ranking and normalized-cosine ranking are identical, so they are interchangeable when inputs
//! are unit-length. Dot product is unbounded and rewards larger-norm rows, which is exactly the
//! intended maximum-inner-product (MIPS) behavior for models trained that way; on normalized inputs
//! it collapses to cosine and doubles as the fast cosine path.

use nabled_core::scalar::NabledReal;
use nabled_linalg::{matrix, vector};
use ndarray::{Array2, ArrayBase, ArrayView2, Data, DataMut, Ix2};

use crate::error::EmbeddingError;

/// Similarity/distance metric used to score queries against a corpus.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Metric {
    /// Cosine similarity in `[-1, 1]`; higher is more similar. The default.
    #[default]
    Cosine,
    /// Raw dot product (maximum inner product); higher is more similar, unbounded, norm-sensitive.
    Dot,
    /// Euclidean (L2) distance; lower is more similar. This is a distance, not a similarity.
    L2,
}

impl Metric {
    /// Ranking polarity: `true` when larger scores indicate a better match (cosine, dot), `false`
    /// when smaller scores are better (L2 distance).
    #[must_use]
    pub const fn higher_is_better(self) -> bool {
        match self {
            Metric::Cosine | Metric::Dot => true,
            Metric::L2 => false,
        }
    }
}

/// Score every query row against every corpus row under `metric`.
///
/// Returns a `(queries.nrows(), corpus.nrows())` matrix.
///
/// # Errors
/// Returns [`EmbeddingError::EmptyInput`] for empty inputs,
/// [`EmbeddingError::DimensionMismatch`] when feature dimensions differ, and
/// [`EmbeddingError::ZeroNorm`] for cosine scoring against zero-norm rows.
pub fn query_corpus_scores<T: NabledReal>(
    queries: &Array2<T>,
    corpus: &Array2<T>,
    metric: Metric,
) -> Result<Array2<T>, EmbeddingError> {
    query_corpus_scores_view(&queries.view(), &corpus.view(), metric)
}

/// Score every query row against every corpus row under `metric` from matrix views.
///
/// # Errors
/// See [`query_corpus_scores`].
pub fn query_corpus_scores_view<T: NabledReal>(
    queries: &ArrayView2<'_, T>,
    corpus: &ArrayView2<'_, T>,
    metric: Metric,
) -> Result<Array2<T>, EmbeddingError> {
    match metric {
        Metric::Cosine => Ok(vector::pairwise_cosine_similarity_view(queries, corpus)?),
        Metric::L2 => Ok(vector::pairwise_l2_distance_view(queries, corpus)?),
        Metric::Dot => Ok(matrix::matmat_view(queries, &corpus.t())?),
    }
}

/// Score queries against a corpus under `metric` into a caller-provided `output` buffer.
///
/// `output` must be shaped `(queries.nrows(), corpus.nrows())`.
///
/// # Errors
/// See [`query_corpus_scores`]; also returns [`EmbeddingError::DimensionMismatch`] when `output`
/// has the wrong shape.
pub fn query_corpus_scores_into<T, S1, S2>(
    queries: &ArrayBase<S1, Ix2>,
    corpus: &ArrayBase<S2, Ix2>,
    metric: Metric,
    output: &mut ArrayBase<impl DataMut<Elem = T>, Ix2>,
) -> Result<(), EmbeddingError>
where
    T: NabledReal,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
{
    match metric {
        Metric::Cosine => vector::pairwise_cosine_similarity_into(queries, corpus, output)?,
        Metric::L2 => vector::pairwise_l2_distance_into(queries, corpus, output)?,
        Metric::Dot => {
            if output.dim() != (queries.nrows(), corpus.nrows()) {
                return Err(EmbeddingError::DimensionMismatch);
            }
            matrix::matmat_view_into(&queries.view(), &corpus.t(), output.view_mut())?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use ndarray::{Array2, arr2};

    use super::*;
    use crate::normalize::normalize_rows;

    fn corpus_f64() -> Array2<f64> {
        arr2(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    }

    #[test]
    fn metric_default_is_cosine() {
        assert_eq!(Metric::default(), Metric::Cosine);
    }

    #[test]
    fn metric_polarity_is_correct() {
        assert!(Metric::Cosine.higher_is_better());
        assert!(Metric::Dot.higher_is_better());
        assert!(!Metric::L2.higher_is_better());
    }

    #[test]
    fn cosine_scores_match_kernel() {
        let queries = arr2(&[[1.0_f64, 0.0, 0.0]]);
        let corpus = corpus_f64();
        let scores = query_corpus_scores(&queries, &corpus, Metric::Cosine).unwrap();
        assert!((scores[[0, 0]] - 1.0).abs() < 1e-12);
        assert!(scores[[0, 1]].abs() < 1e-12);
        assert!((scores[[0, 2]] - 1.0 / 2.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn dot_scores_equal_matmul() {
        let queries = arr2(&[[1.0_f64, 2.0, 3.0]]);
        let corpus = arr2(&[[1.0_f64, 0.0, 0.0], [0.0, 0.0, 2.0]]);
        let scores = query_corpus_scores(&queries, &corpus, Metric::Dot).unwrap();
        assert!((scores[[0, 0]] - 1.0).abs() < 1e-12);
        assert!((scores[[0, 1]] - 6.0).abs() < 1e-12);
    }

    #[test]
    fn l2_scores_match_kernel() {
        let queries = arr2(&[[0.0_f64, 0.0]]);
        let corpus = arr2(&[[3.0_f64, 4.0], [0.0, 0.0]]);
        let scores = query_corpus_scores(&queries, &corpus, Metric::L2).unwrap();
        assert!((scores[[0, 0]] - 5.0).abs() < 1e-12);
        assert!(scores[[0, 1]].abs() < 1e-12);
    }

    #[test]
    fn dot_on_normalized_inputs_equals_cosine() {
        let queries = arr2(&[[0.5_f64, 0.8, 0.3], [0.1, 0.2, 0.9]]);
        let corpus = arr2(&[[0.7_f64, 0.1, 0.2], [0.2, 0.6, 0.1], [0.9, 0.3, 0.4]]);
        let cosine = query_corpus_scores(&queries, &corpus, Metric::Cosine).unwrap();
        let norm_queries = normalize_rows(&queries).unwrap();
        let norm_corpus = normalize_rows(&corpus).unwrap();
        let dot = query_corpus_scores(&norm_queries, &norm_corpus, Metric::Dot).unwrap();
        for (lhs, rhs) in cosine.iter().zip(dot.iter()) {
            assert!((lhs - rhs).abs() < 1e-10, "cosine {lhs} vs normalized dot {rhs}");
        }
    }

    #[test]
    fn l2_and_normalized_cosine_rank_identically() {
        let queries = arr2(&[[0.5_f64, 0.8, 0.3]]);
        let corpus = arr2(&[[0.7_f64, 0.1, 0.2], [0.2, 0.6, 0.1], [0.9, 0.3, 0.4]]);
        let norm_queries = normalize_rows(&queries).unwrap();
        let norm_corpus = normalize_rows(&corpus).unwrap();

        let cosine = query_corpus_scores(&norm_queries, &norm_corpus, Metric::Cosine).unwrap();
        let l2 = query_corpus_scores(&norm_queries, &norm_corpus, Metric::L2).unwrap();

        // Best cosine (max) should be the smallest L2 distance.
        let best_cos = (0..3).max_by(|&a, &b| cosine[[0, a]].total_cmp(&cosine[[0, b]])).unwrap();
        let best_l2 = (0..3).min_by(|&a, &b| l2[[0, a]].total_cmp(&l2[[0, b]])).unwrap();
        assert_eq!(best_cos, best_l2);
    }

    #[test]
    fn scores_into_matches_allocating_for_all_metrics() {
        let queries = arr2(&[[0.5_f64, 0.8, 0.3], [0.1, 0.2, 0.9]]);
        let corpus = corpus_f64();
        for metric in [Metric::Cosine, Metric::Dot, Metric::L2] {
            let expected = query_corpus_scores(&queries, &corpus, metric).unwrap();
            let mut output = Array2::<f64>::zeros((queries.nrows(), corpus.nrows()));
            query_corpus_scores_into(&queries, &corpus, metric, &mut output).unwrap();
            for (lhs, rhs) in expected.iter().zip(output.iter()) {
                assert!((lhs - rhs).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn scores_view_matches_owned() {
        let queries = arr2(&[[0.5_f64, 0.8, 0.3]]);
        let corpus = corpus_f64();
        let owned = query_corpus_scores(&queries, &corpus, Metric::Dot).unwrap();
        let viewed =
            query_corpus_scores_view(&queries.view(), &corpus.view(), Metric::Dot).unwrap();
        assert_eq!(owned, viewed);
    }

    #[test]
    fn f32_cosine_scores_are_consistent() {
        let queries = arr2(&[[1.0_f32, 0.0, 0.0]]);
        let corpus = arr2(&[[1.0_f32, 0.0, 0.0], [0.0, 1.0, 0.0]]);
        let scores = query_corpus_scores(&queries, &corpus, Metric::Cosine).unwrap();
        assert!((scores[[0, 0]] - 1.0).abs() < 1e-5);
        assert!(scores[[0, 1]].abs() < 1e-5);
    }

    #[test]
    fn dimension_mismatch_is_reported() {
        let queries = arr2(&[[1.0_f64, 0.0]]);
        let corpus = arr2(&[[1.0_f64, 0.0, 0.0]]);
        for metric in [Metric::Cosine, Metric::Dot, Metric::L2] {
            assert_eq!(
                query_corpus_scores(&queries, &corpus, metric),
                Err(EmbeddingError::DimensionMismatch)
            );
        }
    }

    #[test]
    fn empty_input_is_reported() {
        let queries = Array2::<f64>::zeros((0, 3));
        let corpus = corpus_f64();
        for metric in [Metric::Cosine, Metric::Dot, Metric::L2] {
            assert_eq!(
                query_corpus_scores(&queries, &corpus, metric),
                Err(EmbeddingError::EmptyInput)
            );
        }
    }

    #[test]
    fn cosine_zero_norm_is_reported() {
        let queries = arr2(&[[0.0_f64, 0.0, 0.0]]);
        let corpus = corpus_f64();
        assert_eq!(
            query_corpus_scores(&queries, &corpus, Metric::Cosine),
            Err(EmbeddingError::ZeroNorm)
        );
    }

    #[test]
    fn dot_into_rejects_wrong_output_shape() {
        let queries = arr2(&[[1.0_f64, 0.0, 0.0]]);
        let corpus = corpus_f64();
        let mut output = Array2::<f64>::zeros((1, 1));
        assert_eq!(
            query_corpus_scores_into(&queries, &corpus, Metric::Dot, &mut output),
            Err(EmbeddingError::DimensionMismatch)
        );
    }
}
