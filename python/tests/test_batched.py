"""Tests for batched operation bindings."""

import numpy as np
import pytest

import pynabled


def _make_spd(n):
    np.random.seed(42)
    x = np.random.randn(n, n).astype(np.float64)
    return x.T @ x + 0.5 * np.eye(n)


def test_batched_row_matvec():
    # API: batched_row_matvec(matrix, vectors). matrix (m,n), vectors (B,n) -> (B,m).
    # Binding passes (matrix, vectors) to nabled as (batch_vectors, matrix).
    # So pass vectors (B,n) first, matrix (m,n) second to get (B,m).
    vectors = np.random.randn(2, 4).astype(np.float64)
    matrix = np.random.randn(3, 4).astype(np.float64)
    out = pynabled.batched_row_matvec(vectors, matrix)
    assert out.shape == (2, 3)
    for i in range(2):
        np.testing.assert_allclose(out[i], matrix @ vectors[i], rtol=1e-10)


def test_batched_matmat():
    left = np.random.randn(2, 3, 4).astype(np.float64)
    right = np.random.randn(2, 4, 5).astype(np.float64)
    out = pynabled.batched_matmat(left, right)
    assert out.shape == (2, 3, 5)
    for i in range(2):
        np.testing.assert_allclose(out[i], left[i] @ right[i], rtol=1e-10)


def test_batched_qr():
    matrices = np.random.randn(2, 3, 3).astype(np.float64)
    results = pynabled.batched_qr(matrices)
    assert len(results) == 2
    for i, (q, r) in enumerate(results):
        np.testing.assert_allclose(q @ r, matrices[i], rtol=1e-10)
        np.testing.assert_allclose(q.T @ q, np.eye(3), rtol=1e-10, atol=1e-14)


def test_batched_svd():
    matrices = np.random.randn(2, 3, 3).astype(np.float64)
    results = pynabled.batched_svd(matrices)
    assert len(results) == 2
    for i, (u, s, vt) in enumerate(results):
        recon = u @ np.diag(s) @ vt
        np.testing.assert_allclose(recon, matrices[i], rtol=1e-10)


def test_batched_lu():
    matrices = np.random.randn(2, 3, 3).astype(np.float64)
    results = pynabled.batched_lu(matrices)
    assert len(results) == 2
    for i, (l, u) in enumerate(results):
        assert l.shape == (3, 3)
        assert u.shape == (3, 3)


def test_batched_cholesky():
    a0 = _make_spd(3)
    a1 = _make_spd(3)
    matrices = np.stack([a0, a1], axis=0)
    results = pynabled.batched_cholesky(matrices)
    assert len(results) == 2
    for i, l in enumerate(results):
        np.testing.assert_allclose(l @ l.T, matrices[i], rtol=1e-10)


def test_batched_symmetric_eigen():
    a0 = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float64)
    a1 = np.array([[3.0, 0.5], [0.5, 2.0]], dtype=np.float64)
    matrices = np.stack([a0, a1], axis=0)
    results = pynabled.batched_symmetric_eigen(matrices)
    assert len(results) == 2
    for i, (vals, vecs) in enumerate(results):
        np.testing.assert_allclose(
            matrices[i] @ vecs, vecs @ np.diag(vals), rtol=1e-10
        )
