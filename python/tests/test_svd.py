"""Tests for SVD bindings."""

import numpy as np
import pytest

import pynabled


def test_svd_decompose():
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64, order="C")
    u, s, vt = pynabled.svd_decompose(a)
    assert u.shape == (2, 2)
    assert s.shape == (2,)
    assert vt.shape == (2, 2)
    recon = u @ np.diag(s) @ vt
    np.testing.assert_allclose(recon, a, rtol=1e-10)


def test_svd_decompose_truncated():
    a = np.random.randn(5, 3).astype(np.float64)
    u, s, vt = pynabled.svd_decompose_truncated(a, 2)
    assert u.shape == (5, 2)
    assert s.shape == (2,)
    assert vt.shape == (2, 3)


def test_svd_pseudo_inverse():
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    pinv = pynabled.svd_pseudo_inverse(a)
    np.testing.assert_allclose(a @ pinv @ a, a, rtol=1e-10)


def test_svd_rank():
    a = np.eye(3, dtype=np.float64)
    _, s, _ = pynabled.svd_decompose(a)
    r = pynabled.svd_rank(s)
    assert r == 3


def test_svd_reconstruct_matrix():
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    u, s, vt = pynabled.svd_decompose(a)
    recon = pynabled.svd_reconstruct_matrix(u, s, vt)
    np.testing.assert_allclose(recon, a, rtol=1e-10)


def test_svd_condition_number():
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    u, s, vt = pynabled.svd_decompose(a)
    kappa = pynabled.svd_condition_number(u, s, vt)
    assert np.isfinite(kappa)
    assert kappa > 0


def test_svd_null_space():
    # Rank-deficient: rows are multiples
    a = np.array([[1.0, 2.0], [2.0, 4.0]], dtype=np.float64)
    null = pynabled.svd_null_space(a, None)
    assert null.ndim == 2
    # Each column should be in null space: A @ null_col ≈ 0
    for j in range(null.shape[1]):
        np.testing.assert_allclose(a @ null[:, j], np.zeros(2), atol=1e-10)
