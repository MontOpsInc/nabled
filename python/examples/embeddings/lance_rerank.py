#!/usr/bin/env python3
"""LanceDB ANN -> nabled exact rerank over a pure Arrow interchange.

This example shows the locked plug-in stance: LanceDB (or any Arrow batch
producer) supplies the top-N candidate batch, then the *same* nabled Arrow
interchange entrypoint does the exact rerank. The rerank path would work
identically on candidates coming from Parquet, an in-memory fixture, or any
other source -- LanceDB is an optional producer, never a load-bearing
dependency of nabled.

Flow:
    external encoder -> Lance dataset -> ANN search (top-N) ->
        FixedSizeList<float32> candidate column -> pynabled.embeddings exact rerank

Example-only requirements (NOT part of the nabled crate graph, pyproject test
deps, or the python-quality / CI gate). Install separately to run this script:

    pip install lance sentence-transformers

``sentence-transformers`` is optional: if it is not installed this example
falls back to a small deterministic hashing encoder so the LanceDB -> rerank
framing still runs end-to-end. ``lance`` is required.

Run from repo root (after `maturin develop`):
    python python/examples/embeddings/lance_rerank.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa

import pynabled
from pynabled import embeddings

CORPUS = [
    "The cat sat quietly on the warm windowsill.",
    "A kitten napped in a sunny spot by the glass.",
    "Quarterly revenue grew on strong cloud demand.",
    "The company reported higher cloud earnings this quarter.",
    "Hikers reached the summit just before sunrise.",
    "The mountain trail was steep and icy near the top.",
    "Fresh basil and tomato make a simple pasta sauce.",
    "He simmered garlic and tomatoes for the pasta.",
]

QUERY = "How were the cloud earnings this quarter?"


def encode(texts: list[str]) -> np.ndarray:
    """Encode texts to dense float32 vectors using any external model.

    Prefers sentence-transformers; falls back to a deterministic hashing
    encoder so the example runs without the optional model dependency.
    """
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=False)
        return np.ascontiguousarray(vecs, dtype=np.float32)
    except Exception:  # noqa: BLE001 - optional dep / offline fallback
        print("(sentence-transformers unavailable; using deterministic hashing encoder)")
        dim = 64
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in text.lower().split():
                h = hash(token) % dim
                out[row, h] += 1.0
        return out


def main() -> int:
    try:
        import lance
    except ImportError:
        print(
            "This example requires the optional `lance` package:\n"
            "    pip install lance sentence-transformers",
            file=sys.stderr,
        )
        return 1

    corpus_vecs = encode(CORPUS)
    query_vec = encode([QUERY])[0]
    dim = corpus_vecs.shape[1]

    # --- Producer side: write vectors to a Lance dataset (the "store"). ---
    table = pa.table(
        {
            "id": pa.array(range(len(CORPUS)), type=pa.int32()),
            "text": pa.array(CORPUS, type=pa.string()),
            "vector": pa.array(
                list(corpus_vecs),
                type=pa.list_(pa.float32(), dim),
            ),
        }
    )
    with tempfile.TemporaryDirectory() as tmp:
        ds_path = str(Path(tmp) / "corpus.lance")
        dataset = lance.write_dataset(table, ds_path)

        # --- ANN side: LanceDB returns the top-N candidate batch. ---
        top_n = 5
        candidates = dataset.to_table(
            nearest={"column": "vector", "q": query_vec, "k": top_n}
        )

        # --- Interchange: read the candidate FixedSizeList<float32> column. ---
        cand_vecs = np.stack(
            [np.asarray(v, dtype=np.float32) for v in candidates["vector"].to_pylist()]
        )
        cand_ids = candidates["id"].to_numpy()
        cand_texts = candidates["text"].to_pylist()

        # --- nabled: the SAME entrypoint does the exact rerank. ---
        reranked = embeddings.rerank(query_vec, cand_vecs, k=top_n, metric="cosine")

    print("LanceDB ANN -> nabled exact rerank")
    print("-" * 52)
    print(f"corpus={len(CORPUS)} vectors, dim={dim}, pynabled features={pynabled.build_features()}")
    print(f"\nquery: {QUERY!r}\n")
    print(f"ANN candidate order (LanceDB top-{top_n}):")
    for rank, cid in enumerate(cand_ids):
        print(f"  {rank}: id={cid}  {cand_texts[rank]!r}")

    print("\nExact cosine rerank (nabled):")
    for rank, pos in enumerate(reranked.indices):
        cid = cand_ids[pos]
        print(f"  {rank}: id={cid}  score={reranked.scores[rank]:.4f}  {cand_texts[pos]!r}")

    best_id = cand_ids[reranked.indices[0]]
    print(f"\nbest match: id={best_id}  {cand_texts[reranked.indices[0]]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
