#!/usr/bin/env python3
"""SVD compression example: low-rank approximation and reconstruction error.

Uses truncated SVD to approximate a matrix with fewer components. Demonstrates
the trade-off between compression (rank k) and reconstruction quality.

Requires: numpy, pynabled, scikit-learn, matplotlib

Run from repo root:
    python python/examples/svd/svd_compression.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

import pynabled


def main():
    # Use a 16x16 "image" from tiling 4 digit samples (8x8 each). Smaller matrices
    # converge reliably with the internal Jacobi SVD; 64x64 can fail.
    np.random.seed(42)
    digits = load_digits()
    # Tile 2x2 digits into a 16x16 matrix
    A = np.block([
        [digits.images[0].astype(np.float64), digits.images[1].astype(np.float64)],
        [digits.images[2].astype(np.float64), digits.images[3].astype(np.float64)],
    ])

    m, n = A.shape
    max_rank = min(m, n)

    # Reconstruction error vs rank k
    k_values = [1, 2, 4, 8, max_rank]
    k_values = [k for k in k_values if k <= max_rank]
    errors = []
    for k in k_values:
        u, s, vt = pynabled.svd_decompose_truncated(A, k)
        recon = pynabled.svd_reconstruct_matrix(u, s, vt)
        err = np.linalg.norm(A - recon, "fro")
        errors.append(err)

    # Print summary
    print("SVD Compression (pynabled)")
    print("-" * 50)
    print(f"Matrix shape: {A.shape}")
    print(f"Full storage: {m * n} values")
    for k, err in zip(k_values, errors):
        storage = m * k + k + k * n
        print(f"  k={k:2d}: Frobenius error = {err:.4f}, storage = {storage} ({100*storage/(m*n):.0f}%)")

    # Figure: reconstruction error + example reconstructions
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.2], hspace=0.3)

    # 1. Reconstruction error vs k
    ax1 = fig.add_subplot(gs[0, :])
    ax1.semilogy(k_values, errors, "o-", color="#377eb8", linewidth=2)
    ax1.set_xlabel("Rank k")
    ax1.set_ylabel("Frobenius reconstruction error")
    ax1.set_title("Reconstruction error vs rank")
    ax1.set_xticks(k_values)

    # 2. Original and reconstructions at k=2, k=8
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.imshow(A, cmap="gray", aspect="equal")
    ax2.set_title("Original")
    ax2.axis("off")

    for i, (k, label) in enumerate(zip([2, 8], ["k=2", "k=8"])):
        ax = fig.add_subplot(gs[1, 1 + i])
        u, s, vt = pynabled.svd_decompose_truncated(A, k)
        recon = pynabled.svd_reconstruct_matrix(u, s, vt)
        ax.imshow(recon, cmap="gray", aspect="equal")
        ax.set_title(f"{label} (err={np.linalg.norm(A - recon, 'fro'):.2f})")
        ax.axis("off")

    plt.tight_layout()
    out_path = Path(__file__).parent / "svd_compression.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
