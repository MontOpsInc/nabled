//! Offline retrieval-quality metrics over ranked id lists.
//!
//! These are pure ranking metrics: they take a best-first list of *retrieved* ids and a set of
//! *relevant* (ground-truth) ids and score the ranking. No embeddings or scores are needed, so the
//! functions are generic over any hashable id type (`Id: Eq + Hash + Copy`) and work equally on
//! corpus row indices or external string/integer ids. Relevance is treated as **binary** (an id is
//! either relevant or not).
//!
//! All metrics clamp internally and never panic on empty inputs: an empty relevant set yields `0.0`
//! for every metric (there is nothing to find), and `k` is clamped to the retrieved length.

use std::collections::HashSet;
use std::hash::Hash;

use crate::error::EmbeddingError;

/// Recall@k: the fraction of relevant ids that appear in the first `k` retrieved ids.
///
/// Defined as `|{retrieved[..k]} ∩ relevant| / |relevant|`. Returns `0.0` when `relevant` is empty.
/// `k` is clamped to `retrieved.len()`.
// Counts (hit/relevant cardinalities) are small ranking quantities; the f64 cast is exact for the
// list sizes these metrics operate on.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn recall_at_k<Id: Eq + Hash + Copy>(retrieved: &[Id], relevant: &[Id], k: usize) -> f64 {
    if relevant.is_empty() {
        return 0.0;
    }
    let relevant_set: HashSet<Id> = relevant.iter().copied().collect();
    let keep = k.min(retrieved.len());
    let mut hits = 0_usize;
    let mut seen: HashSet<Id> = HashSet::new();
    for &id in &retrieved[..keep] {
        if relevant_set.contains(&id) && seen.insert(id) {
            hits += 1;
        }
    }
    hits as f64 / relevant_set.len() as f64
}

/// Reciprocal rank: `1 / rank` of the first relevant id in `retrieved` (rank is 1-based).
///
/// Returns `0.0` when no retrieved id is relevant (or either input is empty).
// A 1-based rank is small; the f64 cast is exact at these magnitudes.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn reciprocal_rank<Id: Eq + Hash + Copy>(retrieved: &[Id], relevant: &[Id]) -> f64 {
    if relevant.is_empty() {
        return 0.0;
    }
    let relevant_set: HashSet<Id> = relevant.iter().copied().collect();
    retrieved
        .iter()
        .position(|id| relevant_set.contains(id))
        .map_or(0.0, |pos| 1.0 / (pos as f64 + 1.0))
}

/// Mean reciprocal rank across a batch of queries.
///
/// `retrieved` and `relevant` must be parallel: one retrieved list and one relevant set per query.
///
/// # Errors
/// Returns [`EmbeddingError::DimensionMismatch`] when the two outer lengths differ.
// The query count is small; the f64 cast for averaging is exact at these magnitudes.
#[allow(clippy::cast_precision_loss)]
pub fn mean_reciprocal_rank<Id: Eq + Hash + Copy>(
    retrieved: &[Vec<Id>],
    relevant: &[Vec<Id>],
) -> Result<f64, EmbeddingError> {
    if retrieved.len() != relevant.len() {
        return Err(EmbeddingError::DimensionMismatch);
    }
    if retrieved.is_empty() {
        return Ok(0.0);
    }
    let sum: f64 = retrieved.iter().zip(relevant.iter()).map(|(r, g)| reciprocal_rank(r, g)).sum();
    Ok(sum / retrieved.len() as f64)
}

/// Normalized discounted cumulative gain at `k` with binary relevance.
///
/// `DCG@k = Σ_{i<k} rel_i / log2(i + 2)` where `rel_i` is 1 if `retrieved[i]` is relevant. The
/// ideal `IDCG@k` places `min(k, |relevant|)` relevant items first. Returns `DCG/IDCG`, or `0.0`
/// when `IDCG` is zero (no relevant items, or `k == 0`). `k` is clamped to `retrieved.len()`.
// Rank positions are small; the f64 cast inside the log discount is exact at these magnitudes.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn ndcg_at_k<Id: Eq + Hash + Copy>(retrieved: &[Id], relevant: &[Id], k: usize) -> f64 {
    if relevant.is_empty() {
        return 0.0;
    }
    let relevant_set: HashSet<Id> = relevant.iter().copied().collect();
    let keep = k.min(retrieved.len());

    let dcg: f64 = retrieved[..keep]
        .iter()
        .enumerate()
        .filter(|(_, id)| relevant_set.contains(id))
        .map(|(i, _)| 1.0 / ((i as f64) + 2.0).log2())
        .sum();

    let ideal_hits = keep.min(relevant_set.len());
    let idcg: f64 = (0..ideal_hits).map(|i| 1.0 / ((i as f64) + 2.0).log2()).sum();

    if idcg == 0.0 { 0.0 } else { dcg / idcg }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recall_counts_fraction_of_relevant_found() {
        let retrieved = [1_i64, 2, 3, 4];
        let relevant = [2_i64, 4, 9];
        // 2 of 3 relevant found in top-4.
        assert!((recall_at_k(&retrieved, &relevant, 4) - 2.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn recall_respects_k_clamp_window() {
        let retrieved = [5_i64, 2, 3, 4];
        let relevant = [3_i64, 4];
        // top-2 = {5,2}: none relevant.
        assert!((recall_at_k(&retrieved, &relevant, 2) - 0.0).abs() < 1e-12);
        // top-4 finds both.
        assert!((recall_at_k(&retrieved, &relevant, 4) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn recall_k_greater_than_len_is_clamped() {
        let retrieved = [1_i64, 2];
        let relevant = [2_i64];
        assert!((recall_at_k(&retrieved, &relevant, 100) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn recall_empty_relevant_is_zero() {
        let retrieved = [1_i64, 2];
        let relevant: [i64; 0] = [];
        assert!(recall_at_k(&retrieved, &relevant, 2).abs() < 1e-12);
    }

    #[test]
    fn recall_ignores_duplicate_retrieved_relevant() {
        let retrieved = [2_i64, 2, 2];
        let relevant = [2_i64, 5];
        // Only one unique relevant hit out of two relevant ids.
        assert!((recall_at_k(&retrieved, &relevant, 3) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn reciprocal_rank_uses_first_relevant_position() {
        let retrieved = [9_i64, 8, 2, 4];
        let relevant = [2_i64, 4];
        // First relevant (2) is at rank 3.
        assert!((reciprocal_rank(&retrieved, &relevant) - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn reciprocal_rank_zero_when_none_relevant() {
        let retrieved = [9_i64, 8];
        let relevant = [1_i64];
        assert!(reciprocal_rank(&retrieved, &relevant).abs() < 1e-12);
    }

    #[test]
    fn mean_reciprocal_rank_averages_queries() {
        let retrieved = vec![vec![2_i64, 1], vec![5_i64, 6, 3]];
        let relevant = vec![vec![2_i64], vec![3_i64]];
        // RR = 1/1 and 1/3 -> mean = (1 + 1/3)/2 = 2/3.
        let mrr = mean_reciprocal_rank(&retrieved, &relevant).unwrap();
        assert!((mrr - 2.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn mean_reciprocal_rank_rejects_length_mismatch() {
        let retrieved = vec![vec![1_i64]];
        let relevant = vec![vec![1_i64], vec![2_i64]];
        assert_eq!(
            mean_reciprocal_rank(&retrieved, &relevant),
            Err(EmbeddingError::DimensionMismatch)
        );
    }

    #[test]
    fn mean_reciprocal_rank_empty_is_zero() {
        let retrieved: Vec<Vec<i64>> = Vec::new();
        let relevant: Vec<Vec<i64>> = Vec::new();
        assert!(mean_reciprocal_rank(&retrieved, &relevant).unwrap().abs() < 1e-12);
    }

    #[test]
    fn ndcg_perfect_ranking_is_one() {
        let retrieved = [1_i64, 2, 3];
        let relevant = [1_i64, 2];
        assert!((ndcg_at_k(&retrieved, &relevant, 3) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn ndcg_hand_computed_partial_ranking() {
        // Relevant item at position 1 (0-based) only.
        let retrieved = [9_i64, 2, 7];
        let relevant = [2_i64];
        // DCG = 1/log2(3); IDCG = 1/log2(2) = 1.
        let expected = 1.0 / 3.0_f64.log2();
        assert!((ndcg_at_k(&retrieved, &relevant, 3) - expected).abs() < 1e-12);
    }

    #[test]
    fn ndcg_empty_relevant_is_zero() {
        let retrieved = [1_i64, 2];
        let relevant: [i64; 0] = [];
        assert!(ndcg_at_k(&retrieved, &relevant, 2).abs() < 1e-12);
    }

    #[test]
    fn ndcg_zero_k_is_zero() {
        let retrieved = [1_i64, 2];
        let relevant = [1_i64];
        assert!(ndcg_at_k(&retrieved, &relevant, 0).abs() < 1e-12);
    }

    #[test]
    fn metrics_work_with_usize_ids() {
        let retrieved = [0_usize, 3, 1];
        let relevant = [1_usize, 3];
        assert!((recall_at_k(&retrieved, &relevant, 3) - 1.0).abs() < 1e-12);
        assert!((reciprocal_rank(&retrieved, &relevant) - 0.5).abs() < 1e-12);
    }
}
