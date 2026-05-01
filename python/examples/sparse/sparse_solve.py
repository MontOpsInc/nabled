#!/usr/bin/env python3
"""Sparse linear solver example: solve Ax = b with PCG.

Builds a sparse symmetric positive definite matrix (diagonal) and solves
Ax = b using pynabled.sparse_pcg_solve. CSR format is compatible with
scipy.sparse.csr_matrix.

Requires: numpy, pynabled, scipy, matplotlib

Run from repo root:
    python python/examples/sparse/sparse_solve.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

import pynabled


def _sparse_spd_diagonal(n):
    """Build sparse SPD diagonal matrix. PCG preconditioner can fail on some
    non-diagonal SPD matrices; diagonal is guaranteed to work."""
    indptr = np.arange(n + 1, dtype=np.int64)
    indices = np.arange(n, dtype=np.int64)
    data = np.arange(1, n + 1, dtype=np.float64)
    return csr_matrix((data, indices, indptr), shape=(n, n))


def main():
    # Build sparse SPD matrix: 100x100 diagonal (1, 2, ..., 100)
    n = 100
    A_sparse = _sparse_spd_diagonal(n)
    A = A_sparse.toarray()
    N = A.shape[0]

    # RHS: random
    np.random.seed(42)
    b = np.random.randn(N).astype(np.float64)

    # Convert to CSR format for pynabled
    indptr = A_sparse.indptr.astype(np.int64)
    indices = A_sparse.indices.astype(np.int64)
    data = A_sparse.data.astype(np.float64)
    nrows, ncols = A_sparse.shape

    # Solve with PCG
    x = pynabled.sparse_pcg_solve(nrows, ncols, indptr, indices, data, b, None, None)

    # Verify: A @ x ≈ b
    residual = np.linalg.norm(A @ x - b)
    print("Sparse PCG Solve (pynabled)")
    print("-" * 50)
    print(f"Matrix: {N}x{N}, nnz={A_sparse.nnz}")
    print(f"Residual ||Ax - b|| = {residual:.2e}")

    # Also verify with sparse_matvec
    Ax = pynabled.sparse_matvec(nrows, ncols, indptr, indices, data, x)
    print(f"sparse_matvec check: ||Ax - b|| = {np.linalg.norm(Ax - b):.2e}")

    # Plot: solution and RHS
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    ax1.plot(x, "o-")
    ax1.set_xlabel("Index")
    ax1.set_ylabel("x")
    ax1.set_title("Solution x")

    ax2.plot(b, "s-", color="C1")
    ax2.set_xlabel("Index")
    ax2.set_ylabel("b")
    ax2.set_title("RHS b")

    plt.tight_layout()
    out_path = Path(__file__).parent / "sparse_solve.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
