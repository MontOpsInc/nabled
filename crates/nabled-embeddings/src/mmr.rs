//! Maximal Marginal Relevance (MMR) re-ranking for result diversification.
//!
//! Plain rerank returns the `k` most *relevant* candidates, which are often near-duplicates. MMR
//! trades a little relevance for diversity by greedily selecting, at each step, the candidate that
//! maximizes
//!
//! ```text
//! lambda * relevance(query, c) - (1 - lambda) * max_{s in selected} similarity(c, s)
//! ```
//!
//! `lambda == 1` ignores the diversity term and reproduces plain [`rerank`](crate::rerank::rerank)
//! order; `lambda == 0` selects purely for novelty relative to already-picked items.
//!
//! Both the query-candidate relevance and the candidate-candidate similarity are computed with the
//! same [`Metric`] via [`query_corpus_scores_view`]. Because L2 is a distance (lower is better), all
//! scores are internally converted to a "higher is better" space (negated for L2) before the greedy
//! selection, so the formula above is applied consistently for every metric. Returned
//! [`Neighbor::score`] values are the original metric scores (a similarity for cosine/dot, a
//! distance for L2), so they match what [`rerank`](crate::rerank::rerank) would report.

use std::cmp::Ordering;

use nabled_core::scalar::NabledReal;
use ndarray::{ArrayView1, ArrayView2, Axis};

use crate::error::EmbeddingError;
use crate::similarity::{Metric, query_corpus_scores_view};
use crate::topk::Neighbor;

/// Greedily select the best `k` diversified candidates for `query` under Maximal Marginal Relevance.
///
/// `lambda` balances relevance (`1.0`) against diversity (`0.0`) and must lie in `[0, 1]`. `k` is
/// clamped to the number of candidates. The result is in MMR selection order (best-first).
///
/// # Errors
/// Returns [`EmbeddingError::InvalidInput`] when `lambda` is outside `[0, 1]`,
/// [`EmbeddingError::EmptyInput`] for empty candidates,
/// [`EmbeddingError::DimensionMismatch`] when `query` length differs from the candidate width, and
/// [`EmbeddingError::ZeroNorm`] for cosine scoring against zero-norm rows.
pub fn mmr<T: NabledReal>(
    query: &ArrayView1<'_, T>,
    candidates: &ArrayView2<'_, T>,
    k: usize,
    lambda: T,
    metric: Metric,
) -> Result<Vec<Neighbor<T>>, EmbeddingError> {
    if lambda < T::zero() || lambda > T::one() {
        return Err(EmbeddingError::InvalidInput(
            "lambda must be in the range [0, 1]".to_string(),
        ));
    }

    let query_rows = query.view().insert_axis(Axis(0));
    let relevance = query_corpus_scores_view(&query_rows, candidates, metric)?;
    let pairwise = query_corpus_scores_view(candidates, candidates, metric)?;

    let n = candidates.nrows();
    let keep = k.min(n);
    if keep == 0 {
        return Ok(Vec::new());
    }

    // Convert to a "higher is better" goodness space so the greedy rule is metric-agnostic.
    let sign = if metric.higher_is_better() { T::one() } else { -T::one() };
    let rel_good: Vec<T> = (0..n).map(|i| sign * relevance[[0, i]]).collect();

    let one_minus_lambda = T::one() - lambda;
    let mut selected: Vec<usize> = Vec::with_capacity(keep);
    let mut chosen = vec![false; n];
    // Running max similarity (in goodness space) of each candidate to the selected set.
    let mut max_sim_to_selected = vec![T::neg_infinity(); n];

    for _ in 0..keep {
        let mut best: Option<(usize, T)> = None;
        for c in 0..n {
            if chosen[c] {
                continue;
            }
            let diversity_penalty = if selected.is_empty() {
                T::zero()
            } else {
                one_minus_lambda * max_sim_to_selected[c]
            };
            let mmr_score = lambda * rel_good[c] - diversity_penalty;
            let better = match best {
                None => true,
                Some((_, best_score)) => {
                    matches!(mmr_score.partial_cmp(&best_score), Some(Ordering::Greater))
                }
            };
            if better {
                best = Some((c, mmr_score));
            }
        }

        let pick = best.map_or(0, |(idx, _)| idx);
        chosen[pick] = true;
        selected.push(pick);

        // Update each remaining candidate's max similarity to the newly selected item.
        for c in 0..n {
            if chosen[c] {
                continue;
            }
            let sim = sign * pairwise[[c, pick]];
            if sim > max_sim_to_selected[c] {
                max_sim_to_selected[c] = sim;
            }
        }
    }

    Ok(selected
        .into_iter()
        .map(|index| Neighbor { index, score: relevance[[0, index]] })
        .collect())
}

#[cfg(test)]
mod tests {
    use ndarray::{arr1, arr2};

    use super::*;
    use crate::rerank::rerank;

    /// Two tight clusters plus the query aligned with the first cluster.
    fn clustered_corpus() -> ndarray::Array2<f64> {
        arr2(&[
            [1.0, 0.0],
            [0.99, 0.01],
            [0.98, 0.02],
            [0.0, 1.0],
            [0.01, 0.99],
        ])
    }

    #[test]
    fn lambda_one_reproduces_plain_rerank_order() {
        let corpus = clustered_corpus();
        let query = arr1(&[1.0_f64, 0.0]);
        let mmr_result = mmr(&query.view(), &corpus.view(), 5, 1.0, Metric::Cosine).unwrap();
        let plain = rerank(&query.view(), &corpus.view(), 5, Metric::Cosine).unwrap();
        let mmr_idx: Vec<usize> = mmr_result.iter().map(|n| n.index).collect();
        let plain_idx: Vec<usize> = plain.iter().map(|n| n.index).collect();
        assert_eq!(mmr_idx, plain_idx);
    }

    #[test]
    fn low_lambda_diversifies_into_second_cluster() {
        let corpus = clustered_corpus();
        let query = arr1(&[1.0_f64, 0.0]);
        // Plain top-2 are both from the first cluster (indices 0 and 1).
        let plain = rerank(&query.view(), &corpus.view(), 2, Metric::Cosine).unwrap();
        assert!(plain[0].index < 3 && plain[1].index < 3);
        // Strongly diversity-weighted MMR should pull in a member of the other cluster.
        let diversified = mmr(&query.view(), &corpus.view(), 2, 0.2, Metric::Cosine).unwrap();
        assert_eq!(diversified[0].index, 0);
        assert!(diversified[1].index >= 3, "expected second pick from the other cluster");
    }

    #[test]
    fn lambda_one_matches_rerank_f32() {
        let corpus = arr2(&[[1.0_f32, 0.0], [0.0, 1.0], [0.9, 0.1]]);
        let query = arr1(&[1.0_f32, 0.0]);
        let mmr_result = mmr(&query.view(), &corpus.view(), 3, 1.0, Metric::Cosine).unwrap();
        let plain = rerank(&query.view(), &corpus.view(), 3, Metric::Cosine).unwrap();
        for (a, b) in mmr_result.iter().zip(plain.iter()) {
            assert_eq!(a.index, b.index);
            assert!((a.score - b.score).abs() < 1e-6);
        }
    }

    #[test]
    fn mmr_l2_lambda_one_matches_rerank() {
        let corpus = arr2(&[[0.0_f64, 0.0], [5.0, 5.0], [0.1, 0.0], [4.9, 5.0]]);
        let query = arr1(&[0.0_f64, 0.0]);
        let mmr_result = mmr(&query.view(), &corpus.view(), 4, 1.0, Metric::L2).unwrap();
        let plain = rerank(&query.view(), &corpus.view(), 4, Metric::L2).unwrap();
        let mmr_idx: Vec<usize> = mmr_result.iter().map(|n| n.index).collect();
        let plain_idx: Vec<usize> = plain.iter().map(|n| n.index).collect();
        assert_eq!(mmr_idx, plain_idx);
    }

    #[test]
    fn mmr_clamps_k() {
        let corpus = arr2(&[[1.0_f64, 0.0], [0.0, 1.0]]);
        let query = arr1(&[1.0_f64, 0.0]);
        let result = mmr(&query.view(), &corpus.view(), 99, 0.5, Metric::Cosine).unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn mmr_zero_k_is_empty() {
        let corpus = arr2(&[[1.0_f64, 0.0], [0.0, 1.0]]);
        let query = arr1(&[1.0_f64, 0.0]);
        let result = mmr(&query.view(), &corpus.view(), 0, 0.5, Metric::Cosine).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn mmr_rejects_lambda_out_of_range() {
        let corpus = arr2(&[[1.0_f64, 0.0], [0.0, 1.0]]);
        let query = arr1(&[1.0_f64, 0.0]);
        assert!(matches!(
            mmr(&query.view(), &corpus.view(), 2, 1.5, Metric::Cosine),
            Err(EmbeddingError::InvalidInput(_))
        ));
        assert!(matches!(
            mmr(&query.view(), &corpus.view(), 2, -0.1, Metric::Cosine),
            Err(EmbeddingError::InvalidInput(_))
        ));
    }

    #[test]
    fn mmr_reports_dimension_mismatch() {
        let corpus = arr2(&[[1.0_f64, 0.0, 0.0]]);
        let query = arr1(&[1.0_f64, 0.0]);
        assert_eq!(
            mmr(&query.view(), &corpus.view(), 1, 0.5, Metric::Cosine),
            Err(EmbeddingError::DimensionMismatch)
        );
    }
}
