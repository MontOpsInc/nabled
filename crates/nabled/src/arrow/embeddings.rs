//! Arrow adapters for embedding retrieval primitives.
//!
//! These wrappers accept Arrow `FixedSizeListArray` embedding columns (rows-of-vectors) and
//! single-vector `PrimitiveArray` queries, bridge them to ndarray views zero-copy via the shared
//! [`fixed_size_list_view`](super::fixed_size_list_view) /
//! [`primitive_array_view`](super::primitive_array_view) helpers, delegate to the ndarray-native
//! `nabled-embeddings` kernels, and return Arrow-native results:
//!
//! - score matrices and normalized rows as [`FixedSizeListArray`] (one inner list per query/row),
//! - single-query rerank neighbors as a [`StructArray`] with an `int64` `index` field and a
//!   metric-typed `score` field,
//! - per-query brute-force kNN neighbors as a [`ListArray`] whose values are the same neighbor
//!   [`StructArray`] layout, one list element per query row.

use std::sync::Arc;

use arrow_array::types::ArrowPrimitiveType;
use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Int64Array, ListArray, PrimitiveArray, StructArray,
};
use arrow_buffer::{OffsetBuffer, ScalarBuffer};
use arrow_schema::{DataType, Field, Fields};
use nabled_core::scalar::NabledReal;
use ndarrow::NdarrowElement;

use super::{
    ArrowInteropError, fixed_size_list_from_owned, fixed_size_list_view, primitive_array_view,
};
use crate::embeddings::{Metric, Neighbor};

fn neighbor_index_to_i64(index: usize) -> Result<i64, ArrowInteropError> {
    i64::try_from(index).map_err(|_| {
        ArrowInteropError::InvalidShape(format!("neighbor index {index} exceeds Arrow int64 range"))
    })
}

fn neighbor_struct_fields<T>() -> Fields
where
    T: ArrowPrimitiveType,
{
    Fields::from(vec![
        Field::new("index", DataType::Int64, false),
        Field::new("score", T::DATA_TYPE, false),
    ])
}

fn build_neighbor_struct<T>(indices: Vec<i64>, scores: Vec<T::Native>) -> StructArray
where
    T: ArrowPrimitiveType,
    T::Native: NdarrowElement,
{
    let index_array = Arc::new(Int64Array::from(indices)) as ArrayRef;
    let score_array =
        Arc::new(PrimitiveArray::<T>::new(ScalarBuffer::from(scores), None)) as ArrayRef;
    StructArray::new(neighbor_struct_fields::<T>(), vec![index_array, score_array], None)
}

fn neighbors_to_struct_array<T>(
    neighbors: &[Neighbor<T::Native>],
) -> Result<StructArray, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NdarrowElement,
{
    let mut indices = Vec::with_capacity(neighbors.len());
    let mut scores = Vec::with_capacity(neighbors.len());
    for neighbor in neighbors {
        indices.push(neighbor_index_to_i64(neighbor.index)?);
        scores.push(neighbor.score);
    }
    Ok(build_neighbor_struct::<T>(indices, scores))
}

/// Score every query row against every corpus row under `metric` for Arrow dense matrices.
///
/// Both inputs are interpreted as `rows-of-vectors`; the result is the `(n_queries, n_corpus)`
/// score matrix encoded as a [`FixedSizeListArray`] with one inner list per query row. The inbound
/// Arrow -> ndarray bridge is zero-copy.
///
/// # Errors
/// Returns an error when inputs contain nulls, are empty, feature dimensions differ, or cosine
/// scoring encounters a zero-norm row.
pub fn arrow_query_corpus_scores<T>(
    queries: &FixedSizeListArray,
    corpus: &FixedSizeListArray,
    metric: Metric,
) -> Result<FixedSizeListArray, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let queries_view = fixed_size_list_view::<T>(queries)?;
    let corpus_view = fixed_size_list_view::<T>(corpus)?;
    let scores = crate::embeddings::query_corpus_scores_view(&queries_view, &corpus_view, metric)?;
    fixed_size_list_from_owned::<T>(scores)
}

/// Rerank `candidates` against a single `query` vector, returning the best `k` neighbors.
///
/// `query` is a single embedding vector stored as a [`PrimitiveArray`]; `candidates` is a
/// `rows-of-vectors` [`FixedSizeListArray`]. The result is a [`StructArray`] with an `int64`
/// `index` field (row into `candidates`) and a metric-typed `score` field, in best-first order.
/// `k` is clamped to the number of candidates. The inbound Arrow -> ndarray bridge is zero-copy.
///
/// # Errors
/// Returns an error when inputs contain nulls, are empty, the query length differs from the
/// candidate width, or cosine scoring encounters a zero-norm row.
pub fn arrow_rerank<T>(
    query: &PrimitiveArray<T>,
    candidates: &FixedSizeListArray,
    k: usize,
    metric: Metric,
) -> Result<StructArray, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let query_view = primitive_array_view(query)?;
    let candidates_view = fixed_size_list_view::<T>(candidates)?;
    let neighbors = crate::embeddings::rerank(&query_view, &candidates_view, k, metric)?;
    neighbors_to_struct_array::<T>(&neighbors)
}

/// Normalize each row of an Arrow dense matrix to unit L2 length.
///
/// `rows` is interpreted as `rows-of-vectors`; the result is the normalized matrix encoded as a
/// [`FixedSizeListArray`] with the same shape. The inbound Arrow -> ndarray bridge is zero-copy.
///
/// # Errors
/// Returns an error when the input contains nulls or is empty.
pub fn arrow_normalize_rows<T>(
    rows: &FixedSizeListArray,
) -> Result<FixedSizeListArray, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let rows_view = fixed_size_list_view::<T>(rows)?;
    let normalized = crate::embeddings::normalize_rows_view(&rows_view)?;
    fixed_size_list_from_owned::<T>(normalized)
}

/// Compute the best `k` corpus neighbors for every query row under `metric` for Arrow matrices.
///
/// Both inputs are interpreted as `rows-of-vectors`. The result is a [`ListArray`] with one list
/// element per query row; each element is a [`StructArray`] (`int64` `index`, metric-typed
/// `score`) listing that query's best-first neighbors. `k` is clamped to the corpus size. The
/// inbound Arrow -> ndarray bridge is zero-copy.
///
/// # Errors
/// Returns an error when inputs contain nulls, are empty, feature dimensions differ, or cosine
/// scoring encounters a zero-norm row.
pub fn arrow_brute_force_knn<T>(
    queries: &FixedSizeListArray,
    corpus: &FixedSizeListArray,
    k: usize,
    metric: Metric,
) -> Result<ListArray, ArrowInteropError>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let queries_view = fixed_size_list_view::<T>(queries)?;
    let corpus_view = fixed_size_list_view::<T>(corpus)?;
    let per_query = crate::embeddings::brute_force_knn(&queries_view, &corpus_view, k, metric)?;

    let total: usize = per_query.iter().map(Vec::len).sum();
    let mut indices = Vec::with_capacity(total);
    let mut scores = Vec::with_capacity(total);
    let mut lengths = Vec::with_capacity(per_query.len());
    for neighbors in &per_query {
        lengths.push(neighbors.len());
        for neighbor in neighbors {
            indices.push(neighbor_index_to_i64(neighbor.index)?);
            scores.push(neighbor.score);
        }
    }

    let values = build_neighbor_struct::<T>(indices, scores);
    let offsets = OffsetBuffer::<i32>::from_lengths(lengths);
    let field = Arc::new(Field::new("item", values.data_type().clone(), false));
    Ok(ListArray::new(field, offsets, Arc::new(values), None))
}
