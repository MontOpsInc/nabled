//! Embedding retrieval compute bindings for Python.
//!
//! Thin PyO3 wrappers over `nabled::embeddings`: normalize, query-vs-corpus scoring, exact rerank,
//! brute-force kNN, and PCA compression. Inputs are borrowed NumPy `float32`/`float64` arrays;
//! metrics are selected with the strings `"cosine"`, `"dot"`, or `"l2"`.

use nabled::embeddings::{self, CorpusWorkspace, Metric, Neighbor, NeighborWithId, QuantizedMatrix};
use ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
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

const fn metric_to_str(metric: Metric) -> &'static str {
    match metric {
        Metric::Cosine => "cosine",
        Metric::Dot => "dot",
        Metric::L2 => "l2",
    }
}

/// Read a rank-1 integer id array (int32 or int64) into an owned `Vec<i64>`.
fn read_ids_i64(ids: &Bound<'_, PyAny>) -> PyResult<Vec<i64>> {
    match utils::index_array1(ids, "ids")? {
        utils::IndexReadonlyArray1::I32(arr) => {
            Ok(arr.as_array().iter().map(|&v| i64::from(v)).collect())
        }
        utils::IndexReadonlyArray1::I64(arr) => Ok(arr.as_array().to_vec()),
    }
}

fn neighbors_with_id_to_arrays<T: numpy::Element + Copy>(
    py: Python<'_>,
    neighbors: &[NeighborWithId<T, i64>],
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    let indices = utils::usize_array1_to_i64(
        neighbors.iter().map(|n| n.index).collect(),
        "neighbor index",
    )?;
    let ids: Array1<i64> = neighbors.iter().map(|n| n.id).collect();
    let scores: Array1<T> = neighbors.iter().map(|n| n.score).collect();
    Ok((
        utils::pyarray1_from_owned(py, indices),
        utils::pyarray1_from_owned(py, ids),
        utils::pyarray1_from_owned(py, scores),
    ))
}

fn neighbor_with_id_lists_to_matrices<T: numpy::Element + Copy + num_traits::Zero>(
    py: Python<'_>,
    lists: &[Vec<NeighborWithId<T, i64>>],
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    let rows = lists.len();
    let cols = lists.first().map_or(0, Vec::len);
    let mut indices = Array2::<i64>::zeros((rows, cols));
    let mut ids = Array2::<i64>::zeros((rows, cols));
    let mut scores = Array2::<T>::zeros((rows, cols));
    for (r, list) in lists.iter().enumerate() {
        for (c, neighbor) in list.iter().enumerate() {
            indices[[r, c]] = i64::try_from(neighbor.index).map_err(|_| {
                PyValueError::new_err("neighbor index must be representable as int64")
            })?;
            ids[[r, c]] = neighbor.id;
            scores[[r, c]] = neighbor.score;
        }
    }
    Ok((
        utils::pyarray2_from_owned(py, indices),
        utils::pyarray2_from_owned(py, ids),
        utils::pyarray2_from_owned(py, scores),
    ))
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

/// Rerank `candidates` against a single `query`, mapping local positions to `ids`.
///
/// Returns `(indices, ids, scores)`: the best-first local candidate positions, their mapped int64
/// ids, and the metric scores. `ids` must have one entry per candidate row.
#[pyfunction]
#[pyo3(signature = (query, candidates, ids, k, metric="cosine"))]
pub fn embeddings_rerank_with_ids<'py>(
    py: Python<'py>,
    query: &Bound<'py, PyAny>,
    candidates: &Bound<'py, PyAny>,
    ids: &Bound<'py, PyAny>,
    k: usize,
    metric: &str,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    let metric = parse_metric(metric)?;
    let id_vec = read_ids_i64(ids)?;
    match (utils::real_array1(query, "query")?, utils::real_array2(candidates, "candidates")?) {
        (utils::RealReadonlyArray1::F32(q), utils::RealReadonlyArray2::F32(c)) => {
            let neighbors =
                embeddings::rerank_with_ids(&q.as_array(), &c.as_array(), &id_vec, k, metric)
                    .map_err(to_py_err)?;
            neighbors_with_id_to_arrays(py, &neighbors)
        }
        (utils::RealReadonlyArray1::F64(q), utils::RealReadonlyArray2::F64(c)) => {
            let neighbors =
                embeddings::rerank_with_ids(&q.as_array(), &c.as_array(), &id_vec, k, metric)
                    .map_err(to_py_err)?;
            neighbors_with_id_to_arrays(py, &neighbors)
        }
        _ => Err(utils::matching_real_dtype_error(&["query", "candidates"])),
    }
}

/// Rerank a shared `corpus` for every query row, mapping local positions to `ids`.
///
/// Returns `(indices, ids, scores)` as `(n_queries, k)` matrices. `ids` must have one entry per
/// corpus row.
#[pyfunction]
#[pyo3(signature = (queries, corpus, ids, k, metric="cosine"))]
pub fn embeddings_batch_rerank_with_ids<'py>(
    py: Python<'py>,
    queries: &Bound<'py, PyAny>,
    corpus: &Bound<'py, PyAny>,
    ids: &Bound<'py, PyAny>,
    k: usize,
    metric: &str,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    let metric = parse_metric(metric)?;
    let id_vec = read_ids_i64(ids)?;
    match (utils::real_array2(queries, "queries")?, utils::real_array2(corpus, "corpus")?) {
        (utils::RealReadonlyArray2::F32(q), utils::RealReadonlyArray2::F32(c)) => {
            let lists =
                embeddings::batch_rerank_with_ids(&q.as_array(), &c.as_array(), &id_vec, k, metric)
                    .map_err(to_py_err)?;
            neighbor_with_id_lists_to_matrices(py, &lists)
        }
        (utils::RealReadonlyArray2::F64(q), utils::RealReadonlyArray2::F64(c)) => {
            let lists =
                embeddings::batch_rerank_with_ids(&q.as_array(), &c.as_array(), &id_vec, k, metric)
                    .map_err(to_py_err)?;
            neighbor_with_id_lists_to_matrices(py, &lists)
        }
        _ => Err(utils::matching_real_dtype_error(&["queries", "corpus"])),
    }
}

/// Maximal Marginal Relevance rerank. Returns `(indices, scores)` in selection order.
#[pyfunction]
#[pyo3(signature = (query, candidates, k, lambda, metric="cosine"))]
pub fn embeddings_mmr<'py>(
    py: Python<'py>,
    query: &Bound<'py, PyAny>,
    candidates: &Bound<'py, PyAny>,
    k: usize,
    lambda: f64,
    metric: &str,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let metric = parse_metric(metric)?;
    match (utils::real_array1(query, "query")?, utils::real_array2(candidates, "candidates")?) {
        (utils::RealReadonlyArray1::F32(q), utils::RealReadonlyArray2::F32(c)) => {
            let lambda = utils::f64_to_f32(lambda, "lambda")?;
            let neighbors = embeddings::mmr(&q.as_array(), &c.as_array(), k, lambda, metric)
                .map_err(to_py_err)?;
            neighbors_to_index_score(py, &neighbors)
        }
        (utils::RealReadonlyArray1::F64(q), utils::RealReadonlyArray2::F64(c)) => {
            let neighbors = embeddings::mmr(&q.as_array(), &c.as_array(), k, lambda, metric)
                .map_err(to_py_err)?;
            neighbors_to_index_score(py, &neighbors)
        }
        _ => Err(utils::matching_real_dtype_error(&["query", "candidates"])),
    }
}

/// Recall@k of `retrieved` ids against the `relevant` ground-truth set.
#[pyfunction]
#[pyo3(signature = (retrieved, relevant, k))]
pub fn embeddings_recall_at_k(retrieved: Vec<i64>, relevant: Vec<i64>, k: usize) -> f64 {
    embeddings::recall_at_k(&retrieved, &relevant, k)
}

/// Reciprocal rank of the first relevant id in `retrieved`.
#[pyfunction]
#[pyo3(signature = (retrieved, relevant))]
pub fn embeddings_reciprocal_rank(retrieved: Vec<i64>, relevant: Vec<i64>) -> f64 {
    embeddings::reciprocal_rank(&retrieved, &relevant)
}

/// Mean reciprocal rank across a batch of `(retrieved, relevant)` query pairs.
#[pyfunction]
#[pyo3(signature = (retrieved, relevant))]
pub fn embeddings_mrr(retrieved: Vec<Vec<i64>>, relevant: Vec<Vec<i64>>) -> PyResult<f64> {
    embeddings::mean_reciprocal_rank(&retrieved, &relevant).map_err(to_py_err)
}

/// Normalized discounted cumulative gain at `k` with binary relevance.
#[pyfunction]
#[pyo3(signature = (retrieved, relevant, k))]
pub fn embeddings_ndcg_at_k(retrieved: Vec<i64>, relevant: Vec<i64>, k: usize) -> f64 {
    embeddings::ndcg_at_k(&retrieved, &relevant, k)
}

/// Quantize each row of a float32 matrix to int8. Returns `(codes, scales)`.
#[pyfunction]
#[pyo3(signature = (rows,))]
pub fn embeddings_quantize_rows<'py>(
    py: Python<'py>,
    rows: &Bound<'py, PyAny>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    match utils::real_array2(rows, "rows")? {
        utils::RealReadonlyArray2::F32(arr) => {
            let quantized = embeddings::quantize_rows(&arr.as_array()).map_err(to_py_err)?;
            Ok((
                utils::pyarray2_from_owned(py, quantized.data().clone()),
                utils::pyarray1_from_owned(py, quantized.scales().clone()),
            ))
        }
        utils::RealReadonlyArray2::F64(_) => {
            Err(PyTypeError::new_err("rows must be a float32 NumPy array for int8 quantization"))
        }
    }
}

/// Read an int8 rank-2 code matrix into an owned array.
fn read_i8_matrix(arr: &Bound<'_, PyAny>, name: &str) -> PyResult<Array2<i8>> {
    let typed = arr.cast::<PyArray2<i8>>().map_err(|_| {
        PyTypeError::new_err(format!("{name} must be a NumPy array with dtype int8 and rank 2"))
    })?;
    Ok(typed.readonly().as_array().to_owned())
}

/// Read a float32 rank-1 scales array into an owned array.
fn read_f32_vector(arr: &Bound<'_, PyAny>, name: &str) -> PyResult<Array1<f32>> {
    let typed = arr.cast::<PyArray1<f32>>().map_err(|_| {
        PyTypeError::new_err(format!("{name} must be a NumPy array with dtype float32 and rank 1"))
    })?;
    Ok(typed.readonly().as_array().to_owned())
}

/// Decode an int8 `(codes, scales)` pair back to a float32 matrix.
#[pyfunction]
#[pyo3(signature = (codes, scales))]
pub fn embeddings_dequantize<'py>(
    py: Python<'py>,
    codes: &Bound<'py, PyAny>,
    scales: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    let matrix =
        QuantizedMatrix::from_parts(read_i8_matrix(codes, "codes")?, read_f32_vector(scales, "scales")?)
            .map_err(to_py_err)?;
    Ok(utils::pyarray2_from_owned(py, matrix.dequantize()))
}

/// Score quantized queries against a quantized corpus by dequantizing then reusing the f32 kernel.
#[pyfunction]
#[pyo3(signature = (query_codes, query_scales, corpus_codes, corpus_scales, metric="cosine"))]
pub fn embeddings_query_corpus_scores_quantized<'py>(
    py: Python<'py>,
    query_codes: &Bound<'py, PyAny>,
    query_scales: &Bound<'py, PyAny>,
    corpus_codes: &Bound<'py, PyAny>,
    corpus_scales: &Bound<'py, PyAny>,
    metric: &str,
) -> PyResult<Py<PyAny>> {
    let metric = parse_metric(metric)?;
    let queries = QuantizedMatrix::from_parts(
        read_i8_matrix(query_codes, "query_codes")?,
        read_f32_vector(query_scales, "query_scales")?,
    )
    .map_err(to_py_err)?;
    let corpus = QuantizedMatrix::from_parts(
        read_i8_matrix(corpus_codes, "corpus_codes")?,
        read_f32_vector(corpus_scales, "corpus_scales")?,
    )
    .map_err(to_py_err)?;
    let scores = queries.query_corpus_scores_quantized(&corpus, metric).map_err(to_py_err)?;
    Ok(utils::pyarray2_from_owned(py, scores))
}

/// dtype-dispatched inner of [`PyCorpusWorkspace`].
enum PyCorpusWorkspaceInner {
    F32(CorpusWorkspace<f32>),
    F64(CorpusWorkspace<f64>),
}

/// A corpus prepared once for repeated scoring/rerank against many queries.
#[pyclass(module = "pynabled._pynabled", name = "_CorpusWorkspace")]
pub(crate) struct PyCorpusWorkspace {
    inner: PyCorpusWorkspaceInner,
}

#[pymethods]
impl PyCorpusWorkspace {
    #[new]
    #[pyo3(signature = (corpus, metric="cosine"))]
    fn new(corpus: &Bound<'_, PyAny>, metric: &str) -> PyResult<Self> {
        let metric = parse_metric(metric)?;
        match utils::real_array2(corpus, "corpus")? {
            utils::RealReadonlyArray2::F32(arr) => {
                let workspace =
                    CorpusWorkspace::build(&arr.as_array(), metric).map_err(to_py_err)?;
                Ok(Self { inner: PyCorpusWorkspaceInner::F32(workspace) })
            }
            utils::RealReadonlyArray2::F64(arr) => {
                let workspace =
                    CorpusWorkspace::build(&arr.as_array(), metric).map_err(to_py_err)?;
                Ok(Self { inner: PyCorpusWorkspaceInner::F64(workspace) })
            }
        }
    }

    #[getter]
    fn metric(&self) -> &'static str {
        match &self.inner {
            PyCorpusWorkspaceInner::F32(ws) => metric_to_str(ws.metric()),
            PyCorpusWorkspaceInner::F64(ws) => metric_to_str(ws.metric()),
        }
    }

    #[getter]
    fn len(&self) -> usize {
        match &self.inner {
            PyCorpusWorkspaceInner::F32(ws) => ws.len(),
            PyCorpusWorkspaceInner::F64(ws) => ws.len(),
        }
    }

    #[getter]
    fn dim(&self) -> usize {
        match &self.inner {
            PyCorpusWorkspaceInner::F32(ws) => ws.dim(),
            PyCorpusWorkspaceInner::F64(ws) => ws.dim(),
        }
    }

    /// Score every query row against the cached corpus.
    fn query_corpus_scores<'py>(
        &self,
        py: Python<'py>,
        queries: &Bound<'py, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        match (&self.inner, utils::real_array2(queries, "queries")?) {
            (PyCorpusWorkspaceInner::F32(ws), utils::RealReadonlyArray2::F32(q)) => {
                let scores = ws.query_corpus_scores(&q.as_array()).map_err(to_py_err)?;
                Ok(utils::pyarray2_from_owned(py, scores))
            }
            (PyCorpusWorkspaceInner::F64(ws), utils::RealReadonlyArray2::F64(q)) => {
                let scores = ws.query_corpus_scores(&q.as_array()).map_err(to_py_err)?;
                Ok(utils::pyarray2_from_owned(py, scores))
            }
            _ => Err(workspace_dtype_error()),
        }
    }

    /// Rerank the cached corpus against a single query. Returns `(indices, scores)`.
    fn rerank_with<'py>(
        &self,
        py: Python<'py>,
        query: &Bound<'py, PyAny>,
        k: usize,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        match (&self.inner, utils::real_array1(query, "query")?) {
            (PyCorpusWorkspaceInner::F32(ws), utils::RealReadonlyArray1::F32(q)) => {
                let neighbors = ws.rerank_with(&q.as_array(), k).map_err(to_py_err)?;
                neighbors_to_index_score(py, &neighbors)
            }
            (PyCorpusWorkspaceInner::F64(ws), utils::RealReadonlyArray1::F64(q)) => {
                let neighbors = ws.rerank_with(&q.as_array(), k).map_err(to_py_err)?;
                neighbors_to_index_score(py, &neighbors)
            }
            _ => Err(workspace_dtype_error()),
        }
    }

    /// Exact brute-force kNN over the cached corpus. Returns `(indices, scores)` of `(n_queries, k)`.
    fn knn_with<'py>(
        &self,
        py: Python<'py>,
        queries: &Bound<'py, PyAny>,
        k: usize,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        match (&self.inner, utils::real_array2(queries, "queries")?) {
            (PyCorpusWorkspaceInner::F32(ws), utils::RealReadonlyArray2::F32(q)) => {
                let lists = ws.knn_with(&q.as_array(), k).map_err(to_py_err)?;
                neighbor_lists_to_matrices(py, &lists)
            }
            (PyCorpusWorkspaceInner::F64(ws), utils::RealReadonlyArray2::F64(q)) => {
                let lists = ws.knn_with(&q.as_array(), k).map_err(to_py_err)?;
                neighbor_lists_to_matrices(py, &lists)
            }
            _ => Err(workspace_dtype_error()),
        }
    }
}

fn workspace_dtype_error() -> PyErr {
    PyTypeError::new_err("query dtype must match the workspace corpus dtype (float32 or float64)")
}
