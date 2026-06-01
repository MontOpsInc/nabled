//! Embedding retrieval compute bindings for Python.
//!
//! Thin PyO3 wrappers over `nabled::embeddings`: normalize, query-vs-corpus scoring, exact rerank,
//! brute-force kNN, and PCA compression. Inputs are borrowed NumPy `float32`/`float64` arrays;
//! metrics are selected with the strings `"cosine"`, `"dot"`, or `"l2"`.

use nabled::embeddings::{self, Metric, Neighbor};
use ndarray::{Array1, Array2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::error::to_py_err;
use crate::utils;

fn parse_metric(metric: &str) -> PyResult<Metric> {
    match metric.to_ascii_lowercase().as_str() {
        "cosine" => Ok(Metric::Cosine),
        "dot" => Ok(Metric::Dot),
        "l2" | "euclidean" => Ok(Metric::L2),
        other => Err(PyValueError::new_err(format!(
            "unknown metric '{other}'; expected 'cosine', 'dot', or 'l2'"
        ))),
    }
}

fn neighbors_to_index_score<T: numpy::Element + Copy>(
    py: Python<'_>,
    neighbors: &[Neighbor<T>],
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let indices = utils::usize_array1_to_i64(
        neighbors.iter().map(|n| n.index).collect(),
        "neighbor index",
    )?;
    let scores: Array1<T> = neighbors.iter().map(|n| n.score).collect();
    Ok((utils::pyarray1_from_owned(py, indices), utils::pyarray1_from_owned(py, scores)))
}

fn neighbor_lists_to_matrices<T: numpy::Element + Copy + num_traits::Zero>(
    py: Python<'_>,
    lists: &[Vec<Neighbor<T>>],
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let rows = lists.len();
    let cols = lists.first().map_or(0, Vec::len);
    let mut indices = Array2::<i64>::zeros((rows, cols));
    let mut scores = Array2::<T>::zeros((rows, cols));
    for (r, list) in lists.iter().enumerate() {
        for (c, neighbor) in list.iter().enumerate() {
            indices[[r, c]] = i64::try_from(neighbor.index).map_err(|_| {
                PyValueError::new_err("neighbor index must be representable as int64")
            })?;
            scores[[r, c]] = neighbor.score;
        }
    }
    Ok((utils::pyarray2_from_owned(py, indices), utils::pyarray2_from_owned(py, scores)))
}

/// Normalize each row to unit L2 length. With `out`, writes into it and returns it.
#[pyfunction]
#[pyo3(signature = (rows, out=None))]
pub fn embeddings_normalize_rows<'py>(
    py: Python<'py>,
    rows: &Bound<'py, PyAny>,
    out: Option<&Bound<'py, PyAny>>,
) -> PyResult<Py<PyAny>> {
    match utils::real_array2(rows, "rows")? {
        utils::RealReadonlyArray2::F32(arr) => {
            if let Some(out) = out {
                let mut out_arr = utils::output_array2::<f32>(out, "out", "float32")?;
                embeddings::normalize_rows_into(&arr.as_array(), &mut out_arr.as_array_mut())
                    .map_err(to_py_err)?;
                Ok(out.clone().unbind())
            } else {
                let result =
                    embeddings::normalize_rows_view(&arr.as_array()).map_err(to_py_err)?;
                Ok(utils::pyarray2_from_owned(py, result))
            }
        }
        utils::RealReadonlyArray2::F64(arr) => {
            if let Some(out) = out {
                let mut out_arr = utils::output_array2::<f64>(out, "out", "float64")?;
                embeddings::normalize_rows_into(&arr.as_array(), &mut out_arr.as_array_mut())
                    .map_err(to_py_err)?;
                Ok(out.clone().unbind())
            } else {
                let result =
                    embeddings::normalize_rows_view(&arr.as_array()).map_err(to_py_err)?;
                Ok(utils::pyarray2_from_owned(py, result))
            }
        }
    }
}

/// Score every query row against every corpus row under `metric`. With `out`, writes into it.
#[pyfunction]
#[pyo3(signature = (queries, corpus, metric="cosine", out=None))]
pub fn embeddings_query_corpus_scores<'py>(
    py: Python<'py>,
    queries: &Bound<'py, PyAny>,
    corpus: &Bound<'py, PyAny>,
    metric: &str,
    out: Option<&Bound<'py, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let metric = parse_metric(metric)?;
    match (utils::real_array2(queries, "queries")?, utils::real_array2(corpus, "corpus")?) {
        (utils::RealReadonlyArray2::F32(q), utils::RealReadonlyArray2::F32(c)) => {
            if let Some(out) = out {
                let mut out_arr = utils::output_array2::<f32>(out, "out", "float32")?;
                embeddings::query_corpus_scores_into(
                    &q.as_array(),
                    &c.as_array(),
                    metric,
                    &mut out_arr.as_array_mut(),
                )
                .map_err(to_py_err)?;
                Ok(out.clone().unbind())
            } else {
                let result =
                    embeddings::query_corpus_scores_view(&q.as_array(), &c.as_array(), metric)
                        .map_err(to_py_err)?;
                Ok(utils::pyarray2_from_owned(py, result))
            }
        }
        (utils::RealReadonlyArray2::F64(q), utils::RealReadonlyArray2::F64(c)) => {
            if let Some(out) = out {
                let mut out_arr = utils::output_array2::<f64>(out, "out", "float64")?;
                embeddings::query_corpus_scores_into(
                    &q.as_array(),
                    &c.as_array(),
                    metric,
                    &mut out_arr.as_array_mut(),
                )
                .map_err(to_py_err)?;
                Ok(out.clone().unbind())
            } else {
                let result =
                    embeddings::query_corpus_scores_view(&q.as_array(), &c.as_array(), metric)
                        .map_err(to_py_err)?;
                Ok(utils::pyarray2_from_owned(py, result))
            }
        }
        _ => Err(utils::matching_real_dtype_error(&["queries", "corpus"])),
    }
}

/// Rerank `candidates` against a single `query`. Returns `(indices, scores)`.
#[pyfunction]
#[pyo3(signature = (query, candidates, k, metric="cosine"))]
pub fn embeddings_rerank<'py>(
    py: Python<'py>,
    query: &Bound<'py, PyAny>,
    candidates: &Bound<'py, PyAny>,
    k: usize,
    metric: &str,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let metric = parse_metric(metric)?;
    match (utils::real_array1(query, "query")?, utils::real_array2(candidates, "candidates")?) {
        (utils::RealReadonlyArray1::F32(q), utils::RealReadonlyArray2::F32(c)) => {
            let neighbors = embeddings::rerank(&q.as_array(), &c.as_array(), k, metric)
                .map_err(to_py_err)?;
            neighbors_to_index_score(py, &neighbors)
        }
        (utils::RealReadonlyArray1::F64(q), utils::RealReadonlyArray2::F64(c)) => {
            let neighbors = embeddings::rerank(&q.as_array(), &c.as_array(), k, metric)
                .map_err(to_py_err)?;
            neighbors_to_index_score(py, &neighbors)
        }
        _ => Err(utils::matching_real_dtype_error(&["query", "candidates"])),
    }
}

/// Exact brute-force kNN for every query row. Returns `(indices, scores)` of shape `(n_queries, k)`.
#[pyfunction]
#[pyo3(signature = (queries, corpus, k, metric="cosine"))]
pub fn embeddings_brute_force_knn<'py>(
    py: Python<'py>,
    queries: &Bound<'py, PyAny>,
    corpus: &Bound<'py, PyAny>,
    k: usize,
    metric: &str,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let metric = parse_metric(metric)?;
    match (utils::real_array2(queries, "queries")?, utils::real_array2(corpus, "corpus")?) {
        (utils::RealReadonlyArray2::F32(q), utils::RealReadonlyArray2::F32(c)) => {
            let lists = embeddings::brute_force_knn(&q.as_array(), &c.as_array(), k, metric)
                .map_err(to_py_err)?;
            neighbor_lists_to_matrices(py, &lists)
        }
        (utils::RealReadonlyArray2::F64(q), utils::RealReadonlyArray2::F64(c)) => {
            let lists = embeddings::brute_force_knn(&q.as_array(), &c.as_array(), k, metric)
                .map_err(to_py_err)?;
            neighbor_lists_to_matrices(py, &lists)
        }
        _ => Err(utils::matching_real_dtype_error(&["queries", "corpus"])),
    }
}

/// Fit a PCA basis on `embeddings` and return the compressed `(n_rows, dims)` score matrix.
#[pyfunction]
#[pyo3(signature = (embeddings_matrix, dims))]
pub fn embeddings_compress_pca<'py>(
    py: Python<'py>,
    embeddings_matrix: &Bound<'py, PyAny>,
    dims: usize,
) -> PyResult<Py<PyAny>> {
    match utils::real_array2(embeddings_matrix, "embeddings")? {
        utils::RealReadonlyArray2::F32(arr) => {
            let owned = arr.as_array().to_owned();
            let model = embeddings::fit_pca(&owned, dims).map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, embeddings::compress(&owned, &model)))
        }
        utils::RealReadonlyArray2::F64(arr) => {
            let owned = arr.as_array().to_owned();
            let model = embeddings::fit_pca(&owned, dims).map_err(to_py_err)?;
            Ok(utils::pyarray2_from_owned(py, embeddings::compress(&owned, &model)))
        }
    }
}
