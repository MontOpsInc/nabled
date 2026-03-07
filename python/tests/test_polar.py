"""Tests for polar decomposition bindings."""

import numpy as np
import pytest

import pynabled


def test_polar_compute():
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    u, p = pynabled.polar_compute(a)
    assert u.shape == (2, 2)
    assert p.shape == (2, 2)
    np.testing.assert_allclose(u @ p, a, rtol=1e-10)
    np.testing.assert_allclose(u.T @ u, np.eye(2), rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(p, p.T, rtol=1e-14)
