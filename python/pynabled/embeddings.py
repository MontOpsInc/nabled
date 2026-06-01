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


__all__ = [
    "Neighbors",
    "VALID_METRICS",
    "normalize_rows",
    "query_corpus_scores",
    "rerank",
    "brute_force_knn",
    "compress_pca",
]
