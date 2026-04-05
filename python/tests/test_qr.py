"""Tests for QR decomposition bindings."""

import numpy as np
import pytest

import pynabled


def test_qr_decompose():
    a = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64, order="C")
    result = pynabled.qr_decompose(a)
    assert result.q.shape == (3, 2)
    assert result.r.shape == (2, 2)
    assert result.rank == 2
    np.testing.assert_allclose(result.q @ result.r, a, rtol=1e-10)
    np.testing.assert_allclose(result.q.T @ result.q, np.eye(2), rtol=1e-10, atol=1e-14)


def test_qr_solve_least_squares():
    np.random.seed(42)
    a = np.random.randn(5, 3).astype(np.float64)
    x_true = np.array([1.0, -2.0, 0.5], dtype=np.float64)
    b = a @ x_true
    x = pynabled.qr_solve_least_squares(a, b)
    np.testing.assert_allclose(a @ x, b, rtol=1e-10)
    np.testing.assert_allclose(x, x_true, rtol=1e-10)
