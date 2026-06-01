"""Parity tests for the Arrow embedding wrappers in ``pynabled.arrow``.

Each Arrow wrapper delegates to the same ``nabled-embeddings`` kernels as the NumPy-facing
``pynabled.embeddings`` surface, so the two paths must agree exactly. These tests require a
``pynabled`` build compiled with both the ``arrow`` and ``embeddings`` features plus ``pyarrow``.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import pyarrow as pa
except ImportError:
    pa = None

try:
    from pynabled import embeddings
    from pynabled.arrow import (
        arrow_embeddings_brute_force_knn,
        arrow_embeddings_normalize_rows,
        arrow_embeddings_query_corpus_scores,
        arrow_embeddings_rerank,
    )
except ImportError:
    embeddings = None
    arrow_embeddings_query_corpus_scores = None

pytestmark = [
    pytest.mark.skipif(pa is None, reason="pyarrow not installed"),
    pytest.mark.skipif(
        arrow_embeddings_query_corpus_scores is None,
        reason="pynabled built without arrow+embeddings features",
    ),
]

DTYPES = [np.float32, np.float64]
METRICS = ["cosine", "dot", "l2"]


def _atol(dtype) -> float:
    return 1e-5 if dtype == np.float32 else 1e-10


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _matrix_array(values, dtype):
    np_values = np.asarray(values, dtype=dtype)
    arrow_type = pa.float32() if np_values.dtype == np.float32 else pa.float64()
    return pa.array(np_values.tolist(), type=pa.list_(arrow_type, np_values.shape[1]))


def _vector_array(values, dtype):
    np_values = np.asarray(values, dtype=dtype)
    arrow_type = pa.float32() if np_values.dtype == np.float32 else pa.float64()
    return pa.array(np_values.tolist(), type=arrow_type)


def _matrix_numpy(array, dtype):
    return np.array(array.to_pylist(), dtype=dtype)


def _struct_to_arrays(array):
    indices = np.asarray(array.field("index").to_pylist(), dtype=np.int64)
    scores = np.asarray(array.field("score").to_pylist())
    return indices, scores


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("metric", METRICS)
def test_arrow_query_corpus_scores_matches_embeddings(dtype, metric):
    queries = _rng(1).standard_normal((4, 6)).astype(dtype)
    corpus = _rng(2).standard_normal((7, 6)).astype(dtype)

    arrow_scores = arrow_embeddings_query_corpus_scores(
        _matrix_array(queries, dtype),
        _matrix_array(corpus, dtype),
        metric,
    )
    expected = embeddings.query_corpus_scores(queries, corpus, metric)

    assert arrow_scores.type.list_size == corpus.shape[0]
    np.testing.assert_allclose(_matrix_numpy(arrow_scores, dtype), expected, atol=_atol(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("metric", METRICS)
def test_arrow_rerank_matches_embeddings(dtype, metric):
    query = _rng(3).standard_normal(6).astype(dtype)
    candidates = _rng(4).standard_normal((9, 6)).astype(dtype)
    k = 4

    arrow_result = arrow_embeddings_rerank(
        _vector_array(query, dtype),
        _matrix_array(candidates, dtype),
        k,
        metric,
    )
    expected = embeddings.rerank(query, candidates, k, metric)

    indices, scores = _struct_to_arrays(arrow_result)
    np.testing.assert_array_equal(indices, np.asarray(expected.indices, dtype=np.int64))
    np.testing.assert_allclose(scores, np.asarray(expected.scores), atol=_atol(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
def test_arrow_normalize_rows_matches_embeddings(dtype):
    rows = _rng(5).standard_normal((5, 8)).astype(dtype)

    arrow_result = arrow_embeddings_normalize_rows(_matrix_array(rows, dtype))
    expected = embeddings.normalize_rows(rows)

    np.testing.assert_allclose(_matrix_numpy(arrow_result, dtype), expected, atol=_atol(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("metric", METRICS)
def test_arrow_brute_force_knn_matches_embeddings(dtype, metric):
    queries = _rng(6).standard_normal((3, 5)).astype(dtype)
    corpus = _rng(7).standard_normal((11, 5)).astype(dtype)
    k = 4

    arrow_result = arrow_embeddings_brute_force_knn(
        _matrix_array(queries, dtype),
        _matrix_array(corpus, dtype),
        k,
        metric,
    )
    expected = embeddings.brute_force_knn(queries, corpus, k, metric)

    rows = arrow_result.to_pylist()
    assert len(rows) == queries.shape[0]
    arrow_indices = np.array([[n["index"] for n in row] for row in rows], dtype=np.int64)
    arrow_scores = np.array([[n["score"] for n in row] for row in rows])

    np.testing.assert_array_equal(arrow_indices, np.asarray(expected.indices, dtype=np.int64))
    np.testing.assert_allclose(arrow_scores, np.asarray(expected.scores), atol=_atol(dtype))


def test_arrow_embeddings_functions_are_exported():
    import pynabled.arrow as arrow_module

    for name in (
        "arrow_embeddings_query_corpus_scores",
        "arrow_embeddings_rerank",
        "arrow_embeddings_normalize_rows",
        "arrow_embeddings_brute_force_knn",
    ):
        assert name in arrow_module.__all__
        assert hasattr(arrow_module, name)


@pytest.mark.parametrize("metric", METRICS)
def test_arrow_query_corpus_scores_k_clamped_knn(metric):
    queries = _rng(8).standard_normal((2, 4)).astype(np.float64)
    corpus = _rng(9).standard_normal((3, 4)).astype(np.float64)

    arrow_result = arrow_embeddings_brute_force_knn(
        _matrix_array(queries, np.float64),
        _matrix_array(corpus, np.float64),
        99,
        metric,
    )
    rows = arrow_result.to_pylist()
    assert all(len(row) == corpus.shape[0] for row in rows)
