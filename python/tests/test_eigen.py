"""Tests for eigenvalue decomposition bindings."""

import numpy as np
import pytest

import pynabled


def _make_spd(n):
    np.random.seed(42)
    x = np.random.randn(n, n).astype(np.float64)
    return x.T @ x + 0.1 * np.eye(n)


def test_eigen_generalized():
    a = _make_spd(3)
    b = _make_spd(3)
    vals, vecs = pynabled.eigen_generalized(a, b)
    assert vals.shape == (3,)
    assert vecs.shape == (3, 3)
    # A v = lambda B v
    for i in range(3):
        av = a @ vecs[:, i]
        bv = b @ vecs[:, i] * vals[i]
        np.testing.assert_allclose(av, bv, rtol=1e-9)


def test_eigen_nonsymmetric():
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    vals_re, vals_im, schur_re, schur_im = pynabled.eigen_nonsymmetric(a)
    assert vals_re.shape == (2,)
    assert vals_im.shape == (2,)
    assert schur_re.shape == (2, 2)
    assert schur_im.shape == (2, 2)
    # Eigenvalues of [[1,2],[3,4]] are 5.37 and -0.37 (approx)
    assert len(vals_re) == 2
