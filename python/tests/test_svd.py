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
