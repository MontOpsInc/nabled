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
    vectors = np.random.randn(2, 4).astype(np.float64)
    matrix = np.random.randn(3, 4).astype(np.float64)
    out = pynabled.batched_row_matvec(matrix, vectors)
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
    matrices = np.array(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 7.0], [1.0, 0.0, 2.0]],
            [[2.0, -1.0, 0.5], [3.0, 4.0, 1.0], [5.0, 2.0, 6.0]],
        ],
        dtype=np.float64,
    )
    results = pynabled.batched_qr(matrices)
    assert len(results) == 2
    for i, result in enumerate(results):
        np.testing.assert_allclose(result.q @ result.r, matrices[i], rtol=1e-10)
        np.testing.assert_allclose(result.q.T @ result.q, np.eye(3), rtol=1e-10, atol=1e-12)
        assert result.rank == 3


def test_batched_svd():
    matrices = np.random.randn(2, 3, 3).astype(np.float64)
    results = pynabled.batched_svd(matrices)
    assert len(results) == 2
    for i, result in enumerate(results):
        recon = result.u @ np.diag(result.singular_values) @ result.vt
        np.testing.assert_allclose(recon, matrices[i], rtol=1e-10)


def test_batched_lu():
    matrices = np.random.randn(2, 3, 3).astype(np.float64)
    results = pynabled.batched_lu(matrices)
    assert len(results) == 2
    rhs = np.array([1.0, -2.0, 0.5], dtype=np.float64)
    for i, result in enumerate(results):
        assert result.l.shape == (3, 3)
        assert result.u.shape == (3, 3)
        assert result.pivots is not None
        assert result.permutation_sign in (-1, 1)
        assert result.pivots.dtype == np.int64
        assert result.pivots.shape == (3,)
        np.testing.assert_allclose(
            pynabled.lu_solve(result, rhs),
            np.linalg.solve(matrices[i], rhs),
            rtol=1e-10,
        )


def test_batched_cholesky():
    a0 = _make_spd(3)
    a1 = _make_spd(3)
    matrices = np.stack([a0, a1], axis=0)
    results = pynabled.batched_cholesky(matrices)
    assert len(results) == 2
    for i, result in enumerate(results):
        np.testing.assert_allclose(result.l @ result.l.T, matrices[i], rtol=1e-10)


def test_batched_symmetric_eigen():
    a0 = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float64)
    a1 = np.array([[3.0, 0.5], [0.5, 2.0]], dtype=np.float64)
    matrices = np.stack([a0, a1], axis=0)
    results = pynabled.batched_symmetric_eigen(matrices)
    assert len(results) == 2
    for i, result in enumerate(results):
        np.testing.assert_allclose(
            matrices[i] @ result.eigenvectors,
            result.eigenvectors @ np.diag(result.eigenvalues),
            rtol=1e-10,
        )


def test_batched_decompositions_accept_float32():
    qr_input = np.array(
        [
            [[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]],
            [[2.0, 1.0], [5.0, 3.0], [13.0, 8.0]],
        ],
        dtype=np.float32,
    )
    square = np.array(
        [
            [[4.0, 1.0], [2.0, 3.0]],
            [[5.0, 2.0], [1.0, 4.0]],
        ],
        dtype=np.float32,
    )
    spd = np.array(
        [
            [[3.0, 1.0], [1.0, 2.0]],
            [[4.0, 0.5], [0.5, 3.0]],
        ],
        dtype=np.float32,
    )

    qr_results = pynabled.batched_qr(qr_input)
    svd_results = pynabled.batched_svd(square)
    lu_results = pynabled.batched_lu(square)
    cholesky_results = pynabled.batched_cholesky(spd)
    eigen_results = pynabled.batched_symmetric_eigen(spd)

    for i, result in enumerate(qr_results):
        assert result.q.dtype == np.float32
        assert result.r.dtype == np.float32
        np.testing.assert_allclose(result.q @ result.r, qr_input[i], rtol=1e-4, atol=1e-5)

    for i, result in enumerate(svd_results):
        assert result.u.dtype == np.float32
        assert result.singular_values.dtype == np.float32
        assert result.vt.dtype == np.float32
        recon = result.u @ np.diag(result.singular_values) @ result.vt
        np.testing.assert_allclose(recon, square[i], rtol=1e-4, atol=1e-5)

    for result in lu_results:
        assert result.l.dtype == np.float32
        assert result.u.dtype == np.float32
        assert result.pivots is not None
        assert result.permutation_sign in (-1, 1)
        assert result.pivots.dtype == np.int64

    for i, result in enumerate(cholesky_results):
        assert result.l.dtype == np.float32
        np.testing.assert_allclose(result.l @ result.l.T, spd[i], rtol=1e-4, atol=1e-5)

    for i, result in enumerate(eigen_results):
        assert result.eigenvalues.dtype == np.float32
        assert result.eigenvectors.dtype == np.float32
        np.testing.assert_allclose(
            spd[i] @ result.eigenvectors,
            result.eigenvectors @ np.diag(result.eigenvalues),
            rtol=1e-4,
            atol=1e-5,
        )
