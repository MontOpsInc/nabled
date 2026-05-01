#!/usr/bin/env python3
"""Full PCA analysis: scree plot, cumulative variance, reconstruction error, 2D projection.

Uses synthetic data from make_classification (10 features, 3 classes). Synthetic data
with unit-scale features is numerically stable for the internal SVD. Real datasets
like Wine can fail when features have vastly different scales.

Demonstrates:
- Scree plot (explained variance per component)
- Cumulative explained variance
- Reconstruction error vs number of components
- 2D projection colored by class

Requires: numpy, pynabled, scikit-learn, matplotlib

Run from repo root:
    python python/examples/pca/pca_full_analysis.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

import pynabled


def main():
    # Synthetic data: 10 features, 3 classes, well-scaled (unit variance)
    np.random.seed(42)
    X, y = make_classification(
        n_samples=300,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
    )
    X = X.astype(np.float64)
    target_names = [f"Class {i}" for i in range(3)]
    n_features = X.shape[1]

    # Full PCA to get all explained variance ratios
    components_full, ev_full, evr_full, mean_full, _ = pynabled.compute_pca(
        X, n_components=None
    )

    # Reconstruction error vs n_components
    k_values = list(range(1, n_features + 1))
    mse_values = []
    for k in k_values:
        comp, _, _, pca_mean, scores = pynabled.compute_pca(X, n_components=k)
        X_recon = pca_mean + scores @ comp
        mse = np.mean((X - X_recon) ** 2)
        mse_values.append(mse)

    # 2D projection for visualization
    components_2d, ev_2d, evr_2d, mean_2d, scores_2d = pynabled.compute_pca(
        X, n_components=2
    )

    # Print summary
    print("PCA Full Analysis (pynabled) - Synthetic dataset")
    print("-" * 50)
    print(f"Data shape: {X.shape}")
    print(f"Explained variance ratio (all): {evr_full}")
    print(f"Cumulative (2 components): {evr_2d.sum():.4f}")
    print(f"Cumulative (all): {evr_full.sum():.4f}")
    print(f"Reconstruction MSE @ k=2: {mse_values[1]:.6f}")
    print(f"Reconstruction MSE @ k=10: {mse_values[-1]:.6f}")

    # Figure: 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    # 1. Scree plot
    ax1 = axes[0, 0]
    n_comp = len(evr_full)
    ax1.bar(range(1, n_comp + 1), evr_full, color="#377eb8", edgecolor="black")
    ax1.set_xlabel("Principal component")
    ax1.set_ylabel("Explained variance ratio")
    ax1.set_title("Scree plot")
    ax1.set_xticks(range(1, n_comp + 1))

    # 2. Cumulative explained variance
    ax2 = axes[0, 1]
    cumsum = np.cumsum(evr_full)
    ax2.plot(range(1, n_comp + 1), cumsum, "o-", color="#e41a1c", linewidth=2)
    ax2.axhline(y=0.95, color="gray", linestyle="--", alpha=0.7, label="95%")
    ax2.set_xlabel("Number of components")
    ax2.set_ylabel("Cumulative explained variance")
    ax2.set_title("Cumulative variance explained")
    ax2.legend()
    ax2.set_ylim(0, 1.05)

    # 3. Reconstruction error vs k
    ax3 = axes[1, 0]
    ax3.semilogy(k_values, mse_values, "o-", color="#4daf4a", linewidth=2)
    ax3.set_xlabel("Number of components (k)")
    ax3.set_ylabel("Reconstruction MSE")
    ax3.set_title("Reconstruction error vs k")

    # 4. 2D projection colored by class
    ax4 = axes[1, 1]
    colors = ["#e41a1c", "#377eb8", "#4daf4a"]
    for i, name in enumerate(target_names):
        mask = y == i
        ax4.scatter(
            scores_2d[mask, 0],
            scores_2d[mask, 1],
            c=colors[i],
            label=name,
            alpha=0.8,
            edgecolors="black",
            linewidth=0.5,
        )
    ax4.set_xlabel("PC1")
    ax4.set_ylabel("PC2")
    ax4.set_title("Synthetic: PC1 vs PC2")
    ax4.legend()
    ax4.set_aspect("equal")

    plt.tight_layout()
    out_path = Path(__file__).parent / "pca_full_analysis.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
