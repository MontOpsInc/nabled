"""Lightweight, ndarray-native, Arrow-zero-copy compute and rerank layer for embedding vectors.

Bring vectors from any model, compute exactly, deploy anywhere. ``pynabled.embeddings`` is the
exact rerank/compute step that sits next to a vector store (e.g. LanceDB), not a vector database.
It does **not** do approximate nearest-neighbor (ANN) search, storage, or model inference; bring
dense float vectors from any encoder and compute exact scores here.

Two rules the math cannot enforce:

1. Query and corpus must come from the *same model* and share the same dimension.
2. Pick the metric to match how the model was trained: ``"cosine"`` (default), ``"dot"`` for
   maximum-inner-product (MIPS) models, or ``"l2"`` where applicable. Dot product on
   un-normalized vectors favors larger-norm rows by design.

Headline usage (the same entrypoint reranks any Arrow batch producer's top-N candidates,
including LanceDB):

>>> import numpy as np
>>> import pynabled
>>> corpus = np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]], dtype=np.float32)
>>> query = np.array([1.0, 0.0], dtype=np.float32)
>>> result = pynabled.embeddings.rerank(query, corpus, k=2, metric="cosine")
>>> int(result.indices[0])
0
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

import pynabled._pynabled as _raw

VALID_METRICS = ("cosine", "dot", "l2")


class Neighbors(NamedTuple):
    """Scored neighbor positions, best-first.

    ``indices`` and ``scores`` are NumPy arrays sharing the same shape: 1-D for
    :func:`rerank` and 2-D ``(n_queries, k)`` for :func:`brute_force_knn`.
    """

    indices: np.ndarray
    scores: np.ndarray


class IdNeighbors(NamedTuple):
    """Scored neighbors carrying caller-supplied stable ids, best-first.

    ``indices`` are local candidate/corpus positions, ``ids`` are the mapped int64 identifiers, and
    ``scores`` are the metric values. Arrays are 1-D for :func:`rerank_with_ids` and 2-D
    ``(n_queries, k)`` for :func:`batch_rerank_with_ids`.
    """

    indices: np.ndarray
    ids: np.ndarray
    scores: np.ndarray


class QuantizedMatrix(NamedTuple):
    """An int8 row-quantized matrix: per-element ``codes`` plus a per-row ``scales`` vector."""

    codes: np.ndarray
    scales: np.ndarray


def _as_id_list(ids) -> list[int]:
    """Coerce an array-like of ids to a flat Python list of ints for the pure-index metrics."""
    return np.asarray(ids).astype(np.int64).ravel().tolist()


def normalize_rows(rows, *, out=None):
    """Normalize each row to unit L2 length. With ``out``, writes into it and returns it."""
    return _raw.embeddings_normalize_rows(rows, out)


def query_corpus_scores(queries, corpus, metric="cosine", *, out=None):
    """Score every query row against every corpus row under ``metric``.

    Returns a ``(n_queries, n_corpus)`` matrix. ``metric`` is one of ``"cosine"`` (higher is
    better), ``"dot"`` (higher is better), or ``"l2"`` (lower is better; a distance).
    """
    return _raw.embeddings_query_corpus_scores(queries, corpus, metric, out)


def rerank(query, candidates, k, metric="cosine") -> Neighbors:
    """Rerank ``candidates`` against a single ``query``; return the best ``k`` neighbors.

    ``k`` is clamped to the candidate count. The result is best-first under ``metric``.
    """
    indices, scores = _raw.embeddings_rerank(query, candidates, k, metric)
    return Neighbors(indices=indices, scores=scores)


def brute_force_knn(queries, corpus, k, metric="cosine") -> Neighbors:
    """Exact brute-force kNN for every query row.

    Returns :class:`Neighbors` whose ``indices``/``scores`` are ``(n_queries, k)`` arrays, with
    ``k`` clamped to the corpus size. Intended for small corpora, evaluation, and golden tests;
    use an ANN index plus :func:`rerank` for production recall over large corpora.
    """
    indices, scores = _raw.embeddings_brute_force_knn(queries, corpus, k, metric)
    return Neighbors(indices=indices, scores=scores)


def compress_pca(embeddings, dims):
    """Fit a PCA basis on ``embeddings`` and return the compressed ``(n_rows, dims)`` matrix."""
    return _raw.embeddings_compress_pca(embeddings, dims)


def rerank_with_ids(query, candidates, ids, k, metric="cosine") -> IdNeighbors:
    """Rerank ``candidates`` against ``query``, mapping local positions to ``ids``.

    ``ids`` must hold one identifier per candidate row. Returns :class:`IdNeighbors` with 1-D
    ``indices`` (local positions), ``ids`` (mapped int64 ids), and ``scores``.
    """
    indices, mapped_ids, scores = _raw.embeddings_rerank_with_ids(
        query, candidates, np.asarray(ids, dtype=np.int64), k, metric
    )
    return IdNeighbors(indices=indices, ids=mapped_ids, scores=scores)


def batch_rerank_with_ids(queries, corpus, ids, k, metric="cosine") -> IdNeighbors:
    """Rerank a shared ``corpus`` for every query row, mapping local positions to ``ids``.

    ``ids`` must hold one identifier per corpus row. Returns :class:`IdNeighbors` whose
    ``indices``/``ids``/``scores`` are ``(n_queries, k)`` arrays. The batch path composes the same
    scoring as :func:`brute_force_knn`; the new value is the threaded-through ids.
    """
    indices, mapped_ids, scores = _raw.embeddings_batch_rerank_with_ids(
        queries, corpus, np.asarray(ids, dtype=np.int64), k, metric
    )
    return IdNeighbors(indices=indices, ids=mapped_ids, scores=scores)


def mmr(query, candidates, k, lambda_, metric="cosine") -> Neighbors:
    """Maximal Marginal Relevance rerank balancing relevance and diversity.

    ``lambda_`` is in ``[0, 1]``: ``1.0`` reproduces plain :func:`rerank` order, lower values
    diversify by penalizing similarity to already-selected results. Returns :class:`Neighbors` in
    MMR selection order.
    """
    indices, scores = _raw.embeddings_mmr(query, candidates, k, float(lambda_), metric)
    return Neighbors(indices=indices, scores=scores)


def recall_at_k(retrieved, relevant, k) -> float:
    """Fraction of ``relevant`` ids appearing in the first ``k`` ``retrieved`` ids."""
    return _raw.embeddings_recall_at_k(_as_id_list(retrieved), _as_id_list(relevant), k)


def reciprocal_rank(retrieved, relevant) -> float:
    """Reciprocal of the 1-based rank of the first relevant id in ``retrieved`` (``0.0`` if none)."""
    return _raw.embeddings_reciprocal_rank(_as_id_list(retrieved), _as_id_list(relevant))


def mean_reciprocal_rank(retrieved, relevant) -> float:
    """Mean reciprocal rank across a batch of ``(retrieved, relevant)`` query pairs.

    ``retrieved`` and ``relevant`` are sequences (one per query) of id array-likes and must have
    equal outer length.
    """
    return _raw.embeddings_mrr(
        [_as_id_list(r) for r in retrieved],
        [_as_id_list(g) for g in relevant],
    )


def ndcg_at_k(retrieved, relevant, k) -> float:
    """Normalized discounted cumulative gain at ``k`` with binary relevance."""
    return _raw.embeddings_ndcg_at_k(_as_id_list(retrieved), _as_id_list(relevant), k)


def quantize_rows(rows) -> QuantizedMatrix:
    """Quantize each row of a ``float32`` matrix to int8 with a per-row symmetric scale."""
    codes, scales = _raw.embeddings_quantize_rows(rows)
    return QuantizedMatrix(codes=codes, scales=scales)


def dequantize(quantized: QuantizedMatrix):
    """Decode a :class:`QuantizedMatrix` back to its approximate ``float32`` values."""
    return _raw.embeddings_dequantize(quantized.codes, quantized.scales)


def query_corpus_scores_quantized(query: QuantizedMatrix, corpus: QuantizedMatrix, metric="cosine"):
    """Score quantized ``query`` rows against a quantized ``corpus`` (dequantize then f32 kernel).

    Results approximate :func:`query_corpus_scores` within the int8 quantization tolerance.
    """
    return _raw.embeddings_query_corpus_scores_quantized(
        query.codes, query.scales, corpus.codes, corpus.scales, metric
    )


class CorpusWorkspace:
    """A corpus prepared once for repeated scoring/rerank against many queries.

    Building precomputes the corpus's metric-specific state (its row norms for cosine) so repeated
    queries against a static corpus skip that recompute. The corpus and metric are fixed for the
    life of the workspace.
    """

    def __init__(self, corpus, metric="cosine"):
        self._ws = _raw._CorpusWorkspace(corpus, metric)

    @property
    def metric(self) -> str:
        """The metric this workspace was built for."""
        return self._ws.metric

    @property
    def dim(self) -> int:
        """Feature dimension of the corpus rows."""
        return self._ws.dim

    def __len__(self) -> int:
        return self._ws.len

    def query_corpus_scores(self, queries):
        """Score every ``queries`` row against the cached corpus; returns a ``(n_queries, n_corpus)`` matrix."""
        return self._ws.query_corpus_scores(queries)

    def rerank_with(self, query, k) -> Neighbors:
        """Rerank the cached corpus against a single ``query``; returns the best ``k`` :class:`Neighbors`."""
        indices, scores = self._ws.rerank_with(query, k)
        return Neighbors(indices=indices, scores=scores)

    def knn_with(self, queries, k) -> Neighbors:
        """Exact brute-force kNN over the cached corpus; returns ``(n_queries, k)`` :class:`Neighbors`."""
        indices, scores = self._ws.knn_with(queries, k)
        return Neighbors(indices=indices, scores=scores)


__all__ = [
    "Neighbors",
    "IdNeighbors",
    "QuantizedMatrix",
    "CorpusWorkspace",
    "VALID_METRICS",
    "normalize_rows",
    "query_corpus_scores",
    "rerank",
    "brute_force_knn",
    "compress_pca",
    "rerank_with_ids",
    "batch_rerank_with_ids",
    "mmr",
    "recall_at_k",
    "reciprocal_rank",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "quantize_rows",
    "dequantize",
    "query_corpus_scores_quantized",
]
