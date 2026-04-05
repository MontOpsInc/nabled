"""Tests for LU decomposition bindings."""

import numpy as np
import pytest

import pynabled


def test_lu_decompose():
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    result = pynabled.lu_decompose(a)
    assert result.l.shape == (2, 2)
    assert result.u.shape == (2, 2)
    # LU with pivoting: P @ L @ U = A. lu_solve uses factors internally.


def test_lu_solve():
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    b = np.array([5.0, 6.0], dtype=np.float64)
    x = pynabled.lu_solve(a, b)
    np.testing.assert_allclose(a @ x, b, rtol=1e-10)


def test_lu_inverse():
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    inv_a = pynabled.lu_inverse(a)
    np.testing.assert_allclose(a @ inv_a, np.eye(2), rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(inv_a @ a, np.eye(2), rtol=1e-10, atol=1e-14)


def test_lu_determinant():
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    det = pynabled.lu_determinant(a)
    np.testing.assert_allclose(det, np.linalg.det(a), rtol=1e-10)
