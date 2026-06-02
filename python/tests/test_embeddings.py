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


# --------------------------------------------------------------------------------------------------
# Item 2: rerank with ids + batch
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
def test_rerank_with_ids_maps_ids(dtype):
    corpus = _rng(20).standard_normal((10, 5)).astype(dtype)
    query = _rng(21).standard_normal(5).astype(dtype)
    ids = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109], dtype=np.int64)
    result = embeddings.rerank_with_ids(query, corpus, ids, 4, "cosine")
    assert result.indices.shape == (4,)
    assert result.ids.shape == (4,)
    assert result.scores.shape == (4,)
    # ids must equal ids[indices].
    np.testing.assert_array_equal(result.ids, ids[result.indices])
    # Order matches plain rerank.
    plain = embeddings.rerank(query, corpus, 4, "cosine")
    np.testing.assert_array_equal(result.indices, plain.indices)


def test_rerank_with_ids_rejects_length_mismatch():
    corpus = _rng(22).standard_normal((4, 3))
    query = _rng(23).standard_normal(3)
    ids = np.array([1, 2], dtype=np.int64)
    with pytest.raises(ValueError):
        embeddings.rerank_with_ids(query, corpus, ids, 2, "cosine")


@pytest.mark.parametrize("dtype", DTYPES)
def test_batch_rerank_with_ids_matches_brute_force(dtype):
    queries = _rng(24).standard_normal((3, 4)).astype(dtype)
    corpus = _rng(25).standard_normal((8, 4)).astype(dtype)
    ids = (np.arange(8) + 50).astype(np.int64)
    result = embeddings.batch_rerank_with_ids(queries, corpus, ids, 3, "l2")
    knn = embeddings.brute_force_knn(queries, corpus, 3, "l2")
    np.testing.assert_array_equal(result.indices, knn.indices)
    np.testing.assert_array_equal(result.ids, ids[knn.indices])


# --------------------------------------------------------------------------------------------------
# Item 3: corpus workspace reuse
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("metric", ["cosine", "dot", "l2"])
def test_corpus_workspace_scores_match_stateless(dtype, metric):
    corpus = _rng(26).standard_normal((9, 6)).astype(dtype)
    queries = _rng(27).standard_normal((4, 6)).astype(dtype)
    ws = embeddings.CorpusWorkspace(corpus, metric)
    assert ws.metric == metric
    assert len(ws) == 9
    assert ws.dim == 6
    cached = ws.query_corpus_scores(queries)
    stateless = embeddings.query_corpus_scores(queries, corpus, metric)
    np.testing.assert_allclose(cached, stateless, atol=_atol(dtype))


def test_corpus_workspace_reuse_matches_rerank():
    corpus = _rng(28).standard_normal((12, 5)).astype(np.float64)
    ws = embeddings.CorpusWorkspace(corpus, "cosine")
    for seed in (29, 30, 31):
        query = _rng(seed).standard_normal(5)
        cached = ws.rerank_with(query, 3)
        plain = embeddings.rerank(query, corpus, 3, "cosine")
        np.testing.assert_array_equal(cached.indices, plain.indices)
        np.testing.assert_allclose(cached.scores, plain.scores, atol=1e-10)


def test_corpus_workspace_knn_matches_brute_force():
    corpus = _rng(32).standard_normal((10, 4)).astype(np.float32)
    queries = _rng(33).standard_normal((3, 4)).astype(np.float32)
    ws = embeddings.CorpusWorkspace(corpus, "l2")
    cached = ws.knn_with(queries, 4)
    knn = embeddings.brute_force_knn(queries, corpus, 4, "l2")
    np.testing.assert_array_equal(cached.indices, knn.indices)
    np.testing.assert_allclose(cached.scores, knn.scores, atol=_atol(np.float32))


# --------------------------------------------------------------------------------------------------
# Item 4: eval metrics
# --------------------------------------------------------------------------------------------------


def test_recall_at_k_hand_computed():
    retrieved = [1, 2, 3, 4]
    relevant = [2, 4, 9]
    assert embeddings.recall_at_k(retrieved, relevant, 4) == pytest.approx(2.0 / 3.0)
    assert embeddings.recall_at_k(retrieved, relevant, 1) == pytest.approx(0.0)


def test_reciprocal_rank_hand_computed():
    assert embeddings.reciprocal_rank([9, 8, 2, 4], [2, 4]) == pytest.approx(1.0 / 3.0)
    assert embeddings.reciprocal_rank([9, 8], [1]) == pytest.approx(0.0)


def test_mean_reciprocal_rank_averages():
    retrieved = [[2, 1], [5, 6, 3]]
    relevant = [[2], [3]]
    assert embeddings.mean_reciprocal_rank(retrieved, relevant) == pytest.approx(2.0 / 3.0)


def test_mean_reciprocal_rank_rejects_length_mismatch():
    with pytest.raises(ValueError):
        embeddings.mean_reciprocal_rank([[1]], [[1], [2]])


def test_ndcg_at_k_hand_computed():
    # Perfect ranking.
    assert embeddings.ndcg_at_k([1, 2, 3], [1, 2], 3) == pytest.approx(1.0)
    # Single relevant at position 1 (0-based): DCG = 1/log2(3), IDCG = 1.
    expected = 1.0 / np.log2(3)
    assert embeddings.ndcg_at_k([9, 2, 7], [2], 3) == pytest.approx(expected)
    # Empty relevant set.
    assert embeddings.ndcg_at_k([1, 2], [], 2) == pytest.approx(0.0)


def test_metrics_accept_numpy_int_arrays():
    retrieved = np.array([1, 2, 3], dtype=np.int64)
    relevant = np.array([3], dtype=np.int32)
    assert embeddings.recall_at_k(retrieved, relevant, 3) == pytest.approx(1.0)


# --------------------------------------------------------------------------------------------------
# Item 5: MMR
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
def test_mmr_lambda_one_matches_rerank(dtype):
    corpus = _rng(34).standard_normal((10, 5)).astype(dtype)
    query = _rng(35).standard_normal(5).astype(dtype)
    mmr_result = embeddings.mmr(query, corpus, 5, 1.0, "cosine")
    plain = embeddings.rerank(query, corpus, 5, "cosine")
    np.testing.assert_array_equal(mmr_result.indices, plain.indices)


def test_mmr_low_lambda_diversifies():
    # Two tight clusters; query aligned with the first.
    corpus = np.array(
        [[1.0, 0.0], [0.99, 0.01], [0.98, 0.02], [0.0, 1.0], [0.01, 0.99]],
        dtype=np.float64,
    )
    query = np.array([1.0, 0.0], dtype=np.float64)
    plain = embeddings.rerank(query, corpus, 2, "cosine")
    assert plain.indices[0] < 3 and plain.indices[1] < 3
    diversified = embeddings.mmr(query, corpus, 2, 0.2, "cosine")
    assert diversified.indices[0] == 0
    assert diversified.indices[1] >= 3


def test_mmr_rejects_lambda_out_of_range():
    corpus = _rng(36).standard_normal((4, 3))
    query = _rng(37).standard_normal(3)
    with pytest.raises(ValueError):
        embeddings.mmr(query, corpus, 2, 1.5, "cosine")


# --------------------------------------------------------------------------------------------------
# Item 6: int8 quantization
# --------------------------------------------------------------------------------------------------


def test_quantize_dequantize_round_trip():
    rows = _rng(38).standard_normal((6, 8)).astype(np.float32)
    quantized = embeddings.quantize_rows(rows)
    assert quantized.codes.dtype == np.int8
    assert quantized.scales.dtype == np.float32
    assert quantized.codes.shape == (6, 8)
    assert quantized.scales.shape == (6,)
    restored = embeddings.dequantize(quantized)
    # Per-row error is bounded by half a quantization step.
    half_step = quantized.scales[:, None] / 2.0 + 1e-6
    assert np.all(np.abs(restored - rows) <= half_step)


def test_quantize_requires_float32():
    rows = _rng(39).standard_normal((3, 4)).astype(np.float64)
    with pytest.raises(TypeError):
        embeddings.quantize_rows(rows)


@pytest.mark.parametrize("metric", ["cosine", "dot", "l2"])
def test_quantized_scoring_close_to_f32(metric):
    queries = np.abs(_rng(40).standard_normal((3, 6))).astype(np.float32) + 0.1
    corpus = np.abs(_rng(41).standard_normal((7, 6))).astype(np.float32) + 0.1
    qq = embeddings.quantize_rows(queries)
    qc = embeddings.quantize_rows(corpus)
    approx = embeddings.query_corpus_scores_quantized(qq, qc, metric)
    exact = embeddings.query_corpus_scores(queries, corpus, metric)
    assert approx.shape == exact.shape
    np.testing.assert_allclose(approx, exact, atol=5e-2, rtol=5e-2)
