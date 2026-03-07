"""Tests for sparse CSR bindings."""

import numpy as np
import pytest

import pynabled


def _make_csr_diagonal(n):
    """Create CSR for n x n diagonal matrix with 1,2,...,n on diagonal."""
    indptr = np.arange(n + 1, dtype=np.int64)
    indices = np.arange(n, dtype=np.int64)
    data = np.arange(1, n + 1, dtype=np.float64)
    return n, n, indptr, indices, data


def test_sparse_matvec():
    nrows, ncols, indptr, indices, data = _make_csr_diagonal(3)
    v = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    out = pynabled.sparse_matvec(nrows, ncols, indptr, indices, data, v)
    expected = np.array([1.0, 4.0, 9.0], dtype=np.float64)
    np.testing.assert_allclose(out, expected, rtol=1e-14)


def test_sparse_jacobi_solve():
    # Diagonal system: diag(1,2,3) x = [1,2,3] -> x = [1,1,1]
    nrows, ncols, indptr, indices, data = _make_csr_diagonal(3)
    rhs = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    x = pynabled.sparse_jacobi_solve(
        nrows, ncols, indptr, indices, data, rhs, None, None
    )
    np.testing.assert_allclose(x, [1.0, 1.0, 1.0], rtol=1e-10)


def test_sparse_pcg_solve():
    # SPD diagonal system
    nrows, ncols, indptr, indices, data = _make_csr_diagonal(3)
    rhs = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    x = pynabled.sparse_pcg_solve(
        nrows, ncols, indptr, indices, data, rhs, None, None
    )
    np.testing.assert_allclose(x, [1.0, 1.0, 1.0], rtol=1e-10)
