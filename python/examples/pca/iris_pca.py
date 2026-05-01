#!/usr/bin/env python3
"""Iris PCA example: load Iris, run PCA via pynabled, visualize with matplotlib.

Requires: numpy, pynabled, scikit-learn, matplotlib

Run from repo root:
    python python/examples/pca/iris_pca.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

import pynabled


def main():
    # Load Iris dataset
    data = load_iris()
    X = data["data"].astype(np.float64)
    y = data["target"]
    target_names = data["target_names"]

    # PCA via pynabled
    components, ev, evr, mean, scores = pynabled.compute_pca(X, n_components=2)

    # Print summary
    print("Iris PCA (pynabled)")
    print("-" * 40)
    print(f"Data shape: {X.shape}")
    print(f"Explained variance ratio: {evr}")
    print(f"Cumulative: {evr.sum():.4f}")

    # Figure: 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Scatter: PC1 vs PC2 colored by species
    colors = ["#e41a1c", "#377eb8", "#4daf4a"]
    for i, name in enumerate(target_names):
        mask = y == i
        ax1.scatter(
            scores[mask, 0],
            scores[mask, 1],
            c=colors[i],
            label=name,
            alpha=0.8,
            edgecolors="black",
            linewidth=0.5,
        )
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.set_title("Iris: PC1 vs PC2")
    ax1.legend()
    ax1.set_aspect("equal")

    # Bar: explained variance ratio
    ax2.bar(range(1, 3), evr, color=["#377eb8", "#4daf4a"], edgecolor="black")
    ax2.set_xlabel("Principal component")
    ax2.set_ylabel("Explained variance ratio")
    ax2.set_title("Explained variance")
    ax2.set_xticks([1, 2])

    plt.tight_layout()
    out_path = Path(__file__).parent / "iris_pca.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
