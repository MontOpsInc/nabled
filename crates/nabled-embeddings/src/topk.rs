//! Direction-aware partial top-k selection over a score vector.
//!
//! This is the one genuinely new primitive in the crate: it generalizes the ad-hoc
//! `argsort + truncate` pattern that retrieval pipelines repeat. Selection is performed with
//! [`select_nth_unstable_by`](slice::select_nth_unstable_by) on an index buffer instead of a full
//! sort, so it is `O(n)` in the average case plus an `O(k log k)` final ordering of the retained
//! slice.

use std::cmp::Ordering;

use nabled_core::scalar::NabledReal;
use ndarray::ArrayView1;

use crate::error::EmbeddingError;

/// A scored corpus position returned by selection and retrieval routines.
///
/// `score` carries the metric value that produced the ranking (a similarity for cosine/dot, a
/// distance for L2); interpret it together with the metric's polarity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Neighbor<T> {
    /// Row index into the corpus the score was computed against.
    pub index: usize,
    /// Metric value for this corpus row.
    pub score: T,
}

impl<T> Neighbor<T> {
    /// Construct a neighbor from an index and score.
    pub const fn new(index: usize, score: T) -> Self { Self { index, score } }
}

/// A scored neighbor that additionally carries a caller-supplied stable identifier.
///
/// This is the id-carrying analogue of [`Neighbor`]: `index` is still the local row position the
/// score was computed against, while `id` is the application's stable identifier for that row (for
/// example a global corpus row id returned by an upstream ANN stage). It is produced by attaching
/// an `&[Id]` to a `Vec<Neighbor<T>>` via [`attach_ids`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NeighborWithId<T, Id> {
    /// Local row index into the candidate set the score was computed against.
    pub index: usize,
    /// Metric value for this candidate row.
    pub score: T,
    /// Caller-supplied stable identifier for this candidate row.
    pub id:    Id,
}

impl<T, Id> NeighborWithId<T, Id> {
    /// Construct an id-carrying neighbor from an index, score, and id.
    pub const fn new(index: usize, score: T, id: Id) -> Self { Self { index, score, id } }
}

/// Attach stable identifiers to ranked [`Neighbor`]s by indexing `ids` with each neighbor index.
///
/// `ids` must hold exactly one identifier per candidate row that was ranked (its length must equal
/// `n_candidates`). Each neighbor's `index` is used to look up its id, so the mapping is correct
/// regardless of the ranked order.
///
/// # Errors
/// Returns [`EmbeddingError::DimensionMismatch`] when `ids.len() != n_candidates`.
pub fn attach_ids<T, Id: Copy>(
    neighbors: Vec<Neighbor<T>>,
    ids: &[Id],
    n_candidates: usize,
) -> Result<Vec<NeighborWithId<T, Id>>, EmbeddingError> {
    if ids.len() != n_candidates {
        return Err(EmbeddingError::DimensionMismatch);
    }
    Ok(neighbors
        .into_iter()
        .map(|n| NeighborWithId { index: n.index, score: n.score, id: ids[n.index] })
        .collect())
}

/// Select the best `k` entries of `scores`, returning them in best-first order.
///
/// `higher_is_better` selects the ranking polarity: pass `true` for similarities (cosine, dot) and
/// `false` for distances (L2). Ties are broken by ascending index so the result is deterministic.
/// `k` is clamped to the number of scores, so `k > n` yields all entries and `k == 0` yields an
/// empty vector. Non-comparable values (`NaN`) are treated as equal and tie-broken by index.
// `scores` is taken by value: `ArrayView1` is a cheap, copy-like handle and the by-value
// signature keeps the call site ergonomic (`top_k(row, k, ...)`).
#[allow(clippy::needless_pass_by_value)]
#[must_use]
pub fn top_k<T: NabledReal>(
    scores: ArrayView1<'_, T>,
    k: usize,
    higher_is_better: bool,
) -> Vec<Neighbor<T>> {
    let n = scores.len();
    let keep = k.min(n);
    if keep == 0 {
        return Vec::new();
    }

    // Total order with deterministic tie-breaking: best entries compare `Less`.
    let order = |a: usize, b: usize| -> Ordering {
        let base = scores[a].partial_cmp(&scores[b]).unwrap_or(Ordering::Equal);
        let directed = if higher_is_better { base.reverse() } else { base };
        directed.then_with(|| a.cmp(&b))
    };

    let mut indices: Vec<usize> = (0..n).collect();
    if keep < n {
        _ = indices.select_nth_unstable_by(keep - 1, |&a, &b| order(a, b));
        indices.truncate(keep);
    }
    indices.sort_unstable_by(|&a, &b| order(a, b));

    indices.into_iter().map(|index| Neighbor { index, score: scores[index] }).collect()
}

#[cfg(test)]
mod tests {
    use ndarray::arr1;

    use super::*;

    #[test]
    fn top_k_higher_is_better_orders_descending() {
        let scores = arr1(&[0.1_f64, 0.9, 0.4, 0.7]);
        let result = top_k(scores.view(), 2, true);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].index, 1);
        assert_eq!(result[1].index, 3);
        assert!((result[0].score - 0.9).abs() < 1e-12);
    }

    #[test]
    fn top_k_lower_is_better_orders_ascending() {
        let scores = arr1(&[0.1_f64, 0.9, 0.4, 0.7]);
        let result = top_k(scores.view(), 2, false);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].index, 0);
        assert_eq!(result[1].index, 2);
    }

    #[test]
    fn top_k_f32_matches_expected() {
        let scores = arr1(&[3.0_f32, 1.0, 2.0]);
        let result = top_k(scores.view(), 3, true);
        let indices: Vec<usize> = result.iter().map(|n| n.index).collect();
        assert_eq!(indices, vec![0, 2, 1]);
    }

    #[test]
    fn top_k_clamps_k_greater_than_n() {
        let scores = arr1(&[1.0_f64, 2.0]);
        let result = top_k(scores.view(), 10, true);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].index, 1);
        assert_eq!(result[1].index, 0);
    }

    #[test]
    fn top_k_zero_k_is_empty() {
        let scores = arr1(&[1.0_f64, 2.0]);
        assert!(top_k(scores.view(), 0, true).is_empty());
    }

    #[test]
    fn top_k_empty_scores_is_empty() {
        let scores = ndarray::Array1::<f64>::zeros(0);
        assert!(top_k(scores.view(), 3, true).is_empty());
    }

    #[test]
    fn top_k_breaks_ties_by_ascending_index() {
        let scores = arr1(&[0.5_f64, 0.5, 0.5, 0.5]);
        let result = top_k(scores.view(), 2, true);
        assert_eq!(result[0].index, 0);
        assert_eq!(result[1].index, 1);
    }

    #[test]
    fn top_k_handles_nan_without_panicking() {
        let scores = arr1(&[f64::NAN, 0.5, 1.0]);
        let result = top_k(scores.view(), 3, true);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn neighbor_new_sets_fields() {
        let neighbor = Neighbor::new(3, 0.5_f64);
        assert_eq!(neighbor.index, 3);
        assert!((neighbor.score - 0.5).abs() < 1e-12);
    }

    #[test]
    fn neighbor_with_id_new_sets_fields() {
        let neighbor = NeighborWithId::new(2, 0.25_f64, 99_i64);
        assert_eq!(neighbor.index, 2);
        assert!((neighbor.score - 0.25).abs() < 1e-12);
        assert_eq!(neighbor.id, 99);
    }

    #[test]
    fn attach_ids_maps_index_to_id() {
        let scores = arr1(&[0.1_f64, 0.9, 0.4]);
        let ranked = top_k(scores.view(), 3, true);
        let ids = [10_i64, 20, 30];
        let with_ids = attach_ids(ranked, &ids, 3).unwrap();
        // Best is index 1 (score 0.9) -> id 20.
        assert_eq!(with_ids[0].index, 1);
        assert_eq!(with_ids[0].id, 20);
        assert_eq!(with_ids[1].index, 2);
        assert_eq!(with_ids[1].id, 30);
        assert_eq!(with_ids[2].index, 0);
        assert_eq!(with_ids[2].id, 10);
    }

    #[test]
    fn attach_ids_f32_preserves_scores() {
        let scores = arr1(&[3.0_f32, 1.0, 2.0]);
        let ranked = top_k(scores.view(), 2, true);
        let ids = ['a', 'b', 'c'];
        let with_ids = attach_ids(ranked, &ids, 3).unwrap();
        assert_eq!(with_ids[0].id, 'a');
        assert!((with_ids[0].score - 3.0).abs() < 1e-6);
    }

    #[test]
    fn attach_ids_rejects_length_mismatch() {
        let scores = arr1(&[0.1_f64, 0.9, 0.4]);
        let ranked = top_k(scores.view(), 3, true);
        let ids = [10_i64, 20];
        assert_eq!(attach_ids(ranked, &ids, 3), Err(EmbeddingError::DimensionMismatch));
    }

    #[test]
    fn attach_ids_empty_is_ok() {
        let ranked: Vec<Neighbor<f64>> = Vec::new();
        let ids: [i64; 0] = [];
        assert!(attach_ids(ranked, &ids, 0).unwrap().is_empty());
    }
}
