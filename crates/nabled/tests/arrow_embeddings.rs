#![cfg(all(feature = "arrow", feature = "embeddings"))]

use approx::assert_abs_diff_eq;
use arrow_array::types::{Float32Type, Float64Type};
use arrow_array::{
    Array, FixedSizeListArray, Float32Array, Float64Array, Int64Array, StructArray,
};
use nabled::arrow::embeddings;
use nabled::embeddings::{Metric, brute_force_knn, normalize_rows, query_corpus_scores, rerank};
use nabled::ndarrow::{IntoArrow, fixed_size_list_as_array2};
use ndarray::{Array1, array};

fn struct_index(structs: &StructArray) -> Vec<i64> {
    structs
        .column_by_name("index")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .values()
        .to_vec()
}

fn struct_scores_f64(structs: &StructArray) -> Vec<f64> {
    structs
        .column_by_name("score")
        .unwrap()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap()
        .values()
        .to_vec()
}

#[test]
fn arrow_query_corpus_scores_matches_ndarray_path() {
    let queries_nd = array![[1.0_f64, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let corpus_nd = array![[1.0_f64, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]];

    let queries = queries_nd.clone().into_arrow().unwrap();
    let corpus = corpus_nd.clone().into_arrow().unwrap();

    for metric in [Metric::Cosine, Metric::Dot, Metric::L2] {
        let expected = query_corpus_scores(&queries_nd, &corpus_nd, metric).unwrap();
        let arrow_scores =
            embeddings::arrow_query_corpus_scores::<Float64Type>(&queries, &corpus, metric)
                .unwrap();
        let view = fixed_size_list_as_array2::<Float64Type>(&arrow_scores).unwrap();
        assert_eq!(view.dim(), expected.dim());
        for (lhs, rhs) in view.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(lhs, rhs, epsilon = 1.0e-12);
        }
    }
}

#[test]
fn arrow_query_corpus_scores_f32_parity() {
    let queries_nd = array![[1.0_f32, 0.0, 0.0]];
    let corpus_nd = array![[1.0_f32, 0.0, 0.0], [0.0, 1.0, 0.0]];

    let queries = queries_nd.clone().into_arrow().unwrap();
    let corpus = corpus_nd.clone().into_arrow().unwrap();

    let expected = query_corpus_scores(&queries_nd, &corpus_nd, Metric::Cosine).unwrap();
    let arrow_scores =
        embeddings::arrow_query_corpus_scores::<Float32Type>(&queries, &corpus, Metric::Cosine)
            .unwrap();
    let view = fixed_size_list_as_array2::<Float32Type>(&arrow_scores).unwrap();
    for (lhs, rhs) in view.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(lhs, rhs, epsilon = 1.0e-6);
    }
}

#[test]
fn arrow_normalize_rows_matches_ndarray_path() {
    let rows_nd = array![[3.0_f64, 4.0], [0.0, 2.0], [1.0, 1.0]];
    let rows = rows_nd.clone().into_arrow().unwrap();

    let expected = normalize_rows(&rows_nd).unwrap();
    let arrow_normalized = embeddings::arrow_normalize_rows::<Float64Type>(&rows).unwrap();
    let view = fixed_size_list_as_array2::<Float64Type>(&arrow_normalized).unwrap();
    assert_eq!(view.dim(), expected.dim());
    for (lhs, rhs) in view.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(lhs, rhs, epsilon = 1.0e-12);
    }
}

#[test]
fn arrow_normalize_rows_f32_parity() {
    let rows_nd = array![[3.0_f32, 4.0]];
    let rows = rows_nd.clone().into_arrow().unwrap();

    let expected = normalize_rows(&rows_nd).unwrap();
    let arrow_normalized = embeddings::arrow_normalize_rows::<Float32Type>(&rows).unwrap();
    let view = fixed_size_list_as_array2::<Float32Type>(&arrow_normalized).unwrap();
    for (lhs, rhs) in view.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(lhs, rhs, epsilon = 1.0e-6);
    }
}

#[test]
fn arrow_rerank_matches_ndarray_path() {
    let candidates_nd = array![[1.0_f64, 0.0], [0.0, 1.0], [0.9, 0.1]];
    let candidates = candidates_nd.clone().into_arrow().unwrap();
    let query_nd = Array1::from(vec![1.0_f64, 0.0]);
    let query = Float64Array::from(vec![1.0_f64, 0.0]);

    let expected = rerank(&query_nd.view(), &candidates_nd.view(), 2, Metric::Cosine).unwrap();
    let arrow_neighbors =
        embeddings::arrow_rerank::<Float64Type>(&query, &candidates, 2, Metric::Cosine).unwrap();

    assert_eq!(arrow_neighbors.len(), expected.len());
    let indices = struct_index(&arrow_neighbors);
    let scores = struct_scores_f64(&arrow_neighbors);
    for (i, neighbor) in expected.iter().enumerate() {
        assert_eq!(indices[i], i64::try_from(neighbor.index).unwrap());
        assert_abs_diff_eq!(scores[i], neighbor.score, epsilon = 1.0e-12);
    }
}

#[test]
fn arrow_rerank_f32_parity() {
    let candidates_nd = array![[1.0_f32, 0.0], [0.0, 1.0]];
    let candidates = candidates_nd.clone().into_arrow().unwrap();
    let query_nd = Array1::from(vec![1.0_f32, 0.0]);
    let query = Float32Array::from(vec![1.0_f32, 0.0]);

    let expected = rerank(&query_nd.view(), &candidates_nd.view(), 2, Metric::Cosine).unwrap();
    let arrow_neighbors =
        embeddings::arrow_rerank::<Float32Type>(&query, &candidates, 2, Metric::Cosine).unwrap();

    let indices = struct_index(&arrow_neighbors);
    assert_eq!(indices[0], i64::try_from(expected[0].index).unwrap());
}

#[test]
fn arrow_brute_force_knn_matches_ndarray_path() {
    let queries_nd = array![[1.0_f64, 0.0], [0.0, 1.0]];
    let corpus_nd = array![[1.0_f64, 0.0], [0.0, 1.0], [0.9, 0.1]];
    let queries = queries_nd.clone().into_arrow().unwrap();
    let corpus = corpus_nd.clone().into_arrow().unwrap();

    let expected = brute_force_knn(&queries_nd.view(), &corpus_nd.view(), 2, Metric::Cosine).unwrap();
    let arrow_lists =
        embeddings::arrow_brute_force_knn::<Float64Type>(&queries, &corpus, 2, Metric::Cosine)
            .unwrap();

    assert_eq!(arrow_lists.len(), expected.len());
    for (q, expected_neighbors) in expected.iter().enumerate() {
        let row = arrow_lists.value(q);
        let structs = row.as_any().downcast_ref::<StructArray>().unwrap();
        let indices = struct_index(structs);
        let scores = struct_scores_f64(structs);
        assert_eq!(indices.len(), expected_neighbors.len());
        for (i, neighbor) in expected_neighbors.iter().enumerate() {
            assert_eq!(indices[i], i64::try_from(neighbor.index).unwrap());
            assert_abs_diff_eq!(scores[i], neighbor.score, epsilon = 1.0e-12);
        }
    }
}

#[test]
fn arrow_embedding_inputs_bridge_zero_copy() {
    let rows_nd = array![[3.0_f64, 4.0], [1.0, 2.0]];
    let rows: FixedSizeListArray = rows_nd.into_arrow().unwrap();

    let view = fixed_size_list_as_array2::<Float64Type>(&rows).unwrap();
    let values = rows.values().as_any().downcast_ref::<Float64Array>().unwrap();

    // A zero-copy bridge shares the same backing buffer pointer as the Arrow values array.
    assert_eq!(view.as_ptr(), values.values().as_ptr());
}
