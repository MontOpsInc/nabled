"""Tests for eigenvalue decomposition bindings."""

import numpy as np
import pytest

import pynabled


def _make_spd(n, dtype=np.float64):
    rng = np.random.default_rng(42)
    x = rng.standard_normal((n, n)).astype(dtype)
    return x.T @ x + np.array(0.1, dtype=dtype) * np.eye(n, dtype=dtype)


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


def test_eigen_accepts_float32():
    a = _make_spd(3, dtype=np.float32)
    b = _make_spd(3, dtype=np.float32)
    nonsymmetric = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    generalized = pynabled.eigen_generalized(a, b)
    nonsymmetric_result = pynabled.eigen_nonsymmetric(nonsymmetric)

    assert generalized.eigenvalues.dtype == np.float32
    assert generalized.eigenvectors.dtype == np.float32
    assert nonsymmetric_result.eigenvalues.dtype == np.complex64
    assert nonsymmetric_result.schur_vectors.dtype == np.complex64
    for i in range(3):
        av = a @ generalized.eigenvectors[:, i]
        bv = b @ generalized.eigenvectors[:, i] * generalized.eigenvalues[i]
        np.testing.assert_allclose(av, bv, rtol=1e-4, atol=1e-5)


def test_eigen_nonsymmetric_accepts_complex128():
    matrix = np.array(
        [[2.0 + 1.0j, 0.0 + 0.0j], [0.0 + 0.0j, -3.0 + 0.5j]],
        dtype=np.complex128,
    )

    result = pynabled.eigen_nonsymmetric(matrix)

    assert result.eigenvalues.dtype == np.complex128
    assert result.schur_vectors.dtype == np.complex128
    np.testing.assert_allclose(
        np.sort_complex(result.eigenvalues),
        np.sort_complex(np.array([2.0 + 1.0j, -3.0 + 0.5j], dtype=np.complex128)),
        rtol=1e-10,
        atol=1e-12,
    )
