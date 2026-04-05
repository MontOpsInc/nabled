"""Tests for Cholesky decomposition bindings."""

import numpy as np
import pynabled
import pytest


def _make_spd(n):
    """Create a symmetric positive definite matrix."""
    np.random.seed(42)
    x = np.random.randn(n, n).astype(np.float64)
    return x.T @ x + 0.1 * np.eye(n)


def test_cholesky_decompose():
    a = _make_spd(3)
    result = pynabled.cholesky_decompose(a)
    assert result.l.shape == (3, 3)
    np.testing.assert_allclose(result.l @ result.l.T, a, rtol=1e-10)
    assert np.allclose(np.tril(result.l), result.l)


def test_cholesky_solve():
    a = _make_spd(3)
    b = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    x = pynabled.cholesky_solve(a, b)
    np.testing.assert_allclose(a @ x, b, rtol=1e-10)


def test_cholesky_inverse():
    a = _make_spd(3)
    inv_a = pynabled.cholesky_inverse(a)
    np.testing.assert_allclose(a @ inv_a, np.eye(3), rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(inv_a @ a, np.eye(3), rtol=1e-10, atol=1e-14)
