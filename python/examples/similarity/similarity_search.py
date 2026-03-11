#!/usr/bin/env python3
"""Similarity search example: find nearest neighbors using cosine similarity.

Uses pynabled.pairwise_cosine_similarity to compute similarities between query
vectors and a corpus. Demonstrates k-NN search for digit embeddings.

Requires: numpy, pynabled, scikit-learn, matplotlib

Run from repo root:
    python python/examples/similarity/similarity_search.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

import pynabled


def main():
    # Load digits: each image is 8x8, flatten to 64-dim embedding
    digits = load_digits()
    X = digits.data.astype(np.float64)
    y = digits.target
    images = digits.images

    # Normalize rows for cosine similarity (L2 unit vectors)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    X_norm = X / norms

    # Query: pick a few digits to search for
    query_indices = [0, 100, 500]
    queries = X_norm[query_indices]
    corpus = X_norm

    # Pairwise cosine similarity: (n_queries, n_corpus)
    sim = pynabled.pairwise_cosine_similarity(queries, corpus)

    # For each query, find top-k neighbors (excluding self)
    k = 5
    top_k_indices = []
    for i, qidx in enumerate(query_indices):
        # Mask out self (similarity = 1)
        sim_row = sim[i].copy()
        sim_row[qidx] = -np.inf
        neighbors = np.argsort(sim_row)[::-1][:k]
        top_k_indices.append(neighbors)

    # Print summary
    print("Similarity Search (pynabled) - Digits dataset")
    print("-" * 50)
    print(f"Corpus: {X.shape[0]} vectors, dim={X.shape[1]}")
    for i, qidx in enumerate(query_indices):
        print(f"\nQuery: digit {y[qidx]} (index {qidx})")
        print(f"  Top-{k} neighbors: {[y[j] for j in top_k_indices[i]]}")

    # Figure: query digits and their nearest neighbors
    fig, axes = plt.subplots(len(query_indices), k + 2, figsize=(12, 4 * len(query_indices)))

    for row, (qidx, neighbors) in enumerate(zip(query_indices, top_k_indices)):
        # Query
        axes[row, 0].imshow(images[qidx], cmap="gray", aspect="equal")
        axes[row, 0].set_title(f"Query: {y[qidx]}")
        axes[row, 0].axis("off")

        # Separator
        axes[row, 1].axis("off")
        axes[row, 1].text(0.5, 0.5, "→", fontsize=24, ha="center", va="center")

        # Neighbors
        for col, nidx in enumerate(neighbors):
            ax = axes[row, 2 + col]
            ax.imshow(images[nidx], cmap="gray", aspect="equal")
            ax.set_title(f"{y[nidx]} ({sim[row, nidx]:.3f})")
            ax.axis("off")

    plt.tight_layout()
    out_path = Path(__file__).parent / "similarity_search.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
