"""Tests for matrix bindings."""

import numpy as np
import pytest

import pynabled


def test_matvec():
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64, order="C")
    x = np.array([1.0, 1.0], dtype=np.float64)
    y = pynabled.matvec(a, x)
    np.testing.assert_allclose(y, [3.0, 7.0])


def test_matmat():
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    b = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    c = pynabled.matmat(a, b)
    np.testing.assert_allclose(c, a)


def test_dot():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    b = np.array([4.0, 5.0, 6.0], dtype=np.float64)
    d = pynabled.dot(a, b)
    assert d == 32.0


def test_eigen_symmetric():
    a = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float64)
    vals, vecs = pynabled.eigen_symmetric(a)
    assert vals.shape == (2,)
    assert vecs.shape == (2, 2)
    np.testing.assert_allclose(a @ vecs, vecs @ np.diag(vals), rtol=1e-10)


def test_schur():
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    t, q = pynabled.schur_compute(a)
    np.testing.assert_allclose(q @ t @ q.T, a, rtol=1e-10)


def test_gram_schmidt():
    a = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    q = pynabled.gram_schmidt(a)
    assert q.shape == a.shape
    # Columns should be orthonormal
    qtq = q.T @ q
    np.testing.assert_allclose(qtq, np.eye(2), rtol=1e-10, atol=1e-14)
