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
    result = pynabled.eigen_generalized(a, b)
    assert result.eigenvalues.shape == (3,)
    assert result.eigenvectors.shape == (3, 3)
    # A v = lambda B v
    for i in range(3):
        av = a @ result.eigenvectors[:, i]
        bv = b @ result.eigenvectors[:, i] * result.eigenvalues[i]
        np.testing.assert_allclose(av, bv, rtol=1e-9)


def test_eigen_nonsymmetric():
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    result = pynabled.eigen_nonsymmetric(a)
    assert result.eigenvalues.shape == (2,)
    assert result.schur_vectors.shape == (2, 2)
    assert np.iscomplexobj(result.eigenvalues)
    assert np.iscomplexobj(result.schur_vectors)
    # Eigenvalues of [[1,2],[3,4]] are 5.37 and -0.37 (approx)
    assert len(result.eigenvalues) == 2
