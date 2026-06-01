"""Parity tests for ``pynabled.embeddings`` against NumPy references.

These tests deliberately avoid optional, example-only dependencies (``lance``,
``sentence-transformers``); the bindings are validated purely against NumPy.
"""

from __future__ import annotations

import numpy as np
import pytest

import pynabled
from pynabled import embeddings

DTYPES = [np.float32, np.float64]


def _atol(dtype) -> float:
    return 1e-5 if dtype == np.float32 else 1e-10


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _normalize_ref(rows: np.ndarray) -> np.ndarray:
    norms = np.sqrt((rows * rows).sum(axis=1, keepdims=True))
    eps = np.finfo(rows.dtype).eps
    return rows / np.maximum(norms, eps)


def _scores_ref(queries: np.ndarray, corpus: np.ndarray, metric: str) -> np.ndarray:
    if metric == "dot":
        return queries @ corpus.T
    if metric == "l2":
        diff = queries[:, None, :] - corpus[None, :, :]
        return np.sqrt((diff * diff).sum(axis=2))
    qn = _normalize_ref(queries)
    cn = _normalize_ref(corpus)
    return qn @ cn.T


def test_embeddings_module_is_exported():
    assert pynabled.embeddings is embeddings
    assert "embeddings" in pynabled.__all__


@pytest.mark.parametrize("dtype", DTYPES)
def test_normalize_rows_matches_numpy(dtype):
    rows = _rng(1).standard_normal((5, 7)).astype(dtype)
    result = embeddings.normalize_rows(rows)
    np.testing.assert_allclose(result, _normalize_ref(rows), atol=_atol(dtype))
    norms = np.sqrt((result * result).sum(axis=1))
    np.testing.assert_allclose(norms, np.ones(5), atol=_atol(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
def test_normalize_rows_out_writes_in_place(dtype):
    rows = _rng(2).standard_normal((4, 3)).astype(dtype)
    out = np.empty_like(rows)
    returned = embeddings.normalize_rows(rows, out=out)
    assert returned is out
    np.testing.assert_allclose(out, _normalize_ref(rows), atol=_atol(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("metric", ["cosine", "dot", "l2"])
def test_query_corpus_scores_matches_numpy(dtype, metric):
    queries = _rng(3).standard_normal((4, 6)).astype(dtype)
    corpus = _rng(4).standard_normal((9, 6)).astype(dtype)
    scores = embeddings.query_corpus_scores(queries, corpus, metric)
    assert scores.shape == (4, 9)
    np.testing.assert_allclose(scores, _scores_ref(queries, corpus, metric), atol=_atol(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
def test_query_corpus_scores_out_matches(dtype):
    queries = _rng(5).standard_normal((3, 5)).astype(dtype)
    corpus = _rng(6).standard_normal((8, 5)).astype(dtype)
    out = np.empty((3, 8), dtype=dtype)
    returned = embeddings.query_corpus_scores(queries, corpus, "dot", out=out)
    assert returned is out
    np.testing.assert_allclose(out, queries @ corpus.T, atol=_atol(dtype))


def test_query_corpus_scores_rejects_unknown_metric():
    queries = np.eye(2, dtype=np.float64)
    corpus = np.eye(2, dtype=np.float64)
    with pytest.raises(ValueError, match="unknown metric"):
        embeddings.query_corpus_scores(queries, corpus, "manhattan")


def test_dot_on_normalized_equals_cosine():
    queries = _rng(7).standard_normal((3, 5))
    corpus = _rng(8).standard_normal((6, 5))
    cosine = embeddings.query_corpus_scores(queries, corpus, "cosine")
    dot = embeddings.query_corpus_scores(
        embeddings.normalize_rows(queries),
        embeddings.normalize_rows(corpus),
        "dot",
    )
    np.testing.assert_allclose(cosine, dot, atol=1e-10)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("metric", ["cosine", "dot", "l2"])
def test_rerank_matches_numpy_topk(dtype, metric):
    corpus = _rng(9).standard_normal((12, 4)).astype(dtype)
    query = _rng(10).standard_normal(4).astype(dtype)
    k = 5
    result = embeddings.rerank(query, corpus, k, metric)
    assert result.indices.shape == (k,)
    assert result.scores.shape == (k,)

    row = _scores_ref(query[None, :], corpus, metric)[0]
    order = np.argsort(row) if metric == "l2" else np.argsort(-row)
    expected = order[:k]
    np.testing.assert_array_equal(result.indices, expected)
    np.testing.assert_allclose(result.scores, row[expected], atol=_atol(dtype))


def test_rerank_clamps_k():
    corpus = _rng(11).standard_normal((3, 4))
    query = _rng(12).standard_normal(4)
    result = embeddings.rerank(query, corpus, 99, "cosine")
    assert result.indices.shape == (3,)


@pytest.mark.parametrize("dtype", DTYPES)
def test_brute_force_knn_matches_numpy(dtype):
    queries = _rng(13).standard_normal((3, 4)).astype(dtype)
    corpus = _rng(14).standard_normal((10, 4)).astype(dtype)
    k = 4
    result = embeddings.brute_force_knn(queries, corpus, k, "l2")
    assert result.indices.shape == (3, k)
    assert result.scores.shape == (3, k)

    scores = _scores_ref(queries, corpus, "l2")
    for q in range(queries.shape[0]):
        expected = np.argsort(scores[q])[:k]
        np.testing.assert_array_equal(result.indices[q], expected)
        np.testing.assert_allclose(result.scores[q], scores[q][expected], atol=_atol(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
def test_compress_pca_matches_numpy_projection(dtype):
    data = _rng(15).standard_normal((20, 6)).astype(dtype)
    dims = 3
    compressed = embeddings.compress_pca(data, dims)
    assert compressed.shape == (20, dims)

    # Sign-robust reference: PCA scores' Gram matrix is invariant to per-component sign flips.
    centered = data - data.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered.astype(np.float64), full_matrices=False)
    ref = centered.astype(np.float64) @ vt[:dims].T
    np.testing.assert_allclose(
        compressed.astype(np.float64) @ compressed.T.astype(np.float64),
        ref @ ref.T,
        rtol=1e-4 if dtype == np.float32 else 1e-7,
        atol=1e-4 if dtype == np.float32 else 1e-9,
    )


def test_compress_pca_rejects_zero_dims():
    data = _rng(16).standard_normal((5, 4))
    with pytest.raises(ValueError):
        embeddings.compress_pca(data, 0)


def test_lancedb_style_rerank_pipeline():
    """Mimics the LanceDB plug-in framing without any DB dependency.

    A separate (here, synthetic) producer supplies the top-N candidate batch; the same exact
    rerank entrypoint then orders them. The rerank path is producer-agnostic.
    """
    rng = _rng(17)
    corpus = rng.standard_normal((50, 8)).astype(np.float32)
    query = rng.standard_normal(8).astype(np.float32)

    # Stand-in for an ANN recall stage returning candidate row ids.
    ann_candidate_ids = np.array([3, 7, 11, 19, 25, 33, 41, 48], dtype=np.int64)
    candidate_batch = corpus[ann_candidate_ids]

    reranked = embeddings.rerank(query, candidate_batch, k=3, metric="cosine")
    # Map local candidate positions back to global corpus ids.
    global_ids = ann_candidate_ids[reranked.indices]

    # Reference: exact cosine over the same candidate batch.
    row = _scores_ref(query[None, :], candidate_batch, "cosine")[0]
    expected_local = np.argsort(-row)[:3]
    np.testing.assert_array_equal(reranked.indices, expected_local)
    np.testing.assert_array_equal(global_ids, ann_candidate_ids[expected_local])
