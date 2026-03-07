"""Tests for iterative solver bindings (CG, GMRES)."""

import numpy as np
import pytest

import pynabled


def _make_spd(n):
    np.random.seed(42)
    x = np.random.randn(n, n).astype(np.float64)
    return x.T @ x + 0.5 * np.eye(n)


def test_conjugate_gradient():
    a = _make_spd(5)
    b = np.random.randn(5).astype(np.float64)
    x = pynabled.conjugate_gradient(a, b, None, None)
    np.testing.assert_allclose(a @ x, b, rtol=1e-10)


def test_gmres():
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    b = np.array([5.0, 6.0], dtype=np.float64)
    x = pynabled.gmres(a, b, None, None)
    np.testing.assert_allclose(a @ x, b, rtol=1e-10)
