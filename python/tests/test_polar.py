"""Tests for polar decomposition bindings."""

import numpy as np
import pytest

import pynabled


def test_polar_compute():
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    result = pynabled.polar_compute(a)
    assert result.u.shape == (2, 2)
    assert result.p.shape == (2, 2)
    np.testing.assert_allclose(result.u @ result.p, a, rtol=1e-10)
    np.testing.assert_allclose(result.u.T @ result.u, np.eye(2), rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(result.p, result.p.T, rtol=1e-14)


def test_polar_accepts_float32():
    a = np.array([[1.0, 2.0], [3.0, 5.0]], dtype=np.float32)
    result = pynabled.polar_compute(a)
    assert result.u.dtype == np.float32
    assert result.p.dtype == np.float32
    np.testing.assert_allclose(result.u @ result.p, a, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(
        result.u.T @ result.u, np.eye(2, dtype=np.float32), rtol=1e-4, atol=1e-5
    )
