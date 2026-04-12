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


def test_polar_accepts_complex128():
    a = np.array([[1.0 + 1.0j, 2.0 - 0.5j], [3.0 + 0.25j, 4.0 - 1.0j]], dtype=np.complex128)
    result = pynabled.polar_compute(a)

    assert result.u.dtype == np.complex128
    assert result.p.dtype == np.complex128
    np.testing.assert_allclose(result.u @ result.p, a, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        result.u.conj().T @ result.u,
        np.eye(2, dtype=np.complex128),
        rtol=1e-10,
        atol=1e-12,
    )


def test_polar_reuses_svd_result_and_output_buffers():
    a = np.array([[1.0, 2.0], [3.0, 5.0]], dtype=np.float64)
    svd = pynabled.svd_decompose(a)
    out = pynabled.PolarResult(
        u=np.empty((2, 2), dtype=np.float64, order="F"),
        p=np.empty((2, 2), dtype=np.float64, order="F"),
    )

    returned = pynabled.polar_compute(svd, out=out)

    assert returned is out
    np.testing.assert_allclose(out.u @ out.p, a, rtol=1e-10, atol=1e-12)


def test_polar_reuses_complex_svd_result():
    a = np.array([[1.0 + 1.0j, 2.0 - 0.5j], [3.0 + 0.25j, 4.0 - 1.0j]], dtype=np.complex128)
    svd = pynabled.svd_decompose(a)
    out = pynabled.PolarResult(
        u=np.empty((2, 2), dtype=np.complex128, order="F"),
        p=np.empty((2, 2), dtype=np.complex128, order="F"),
    )

    returned = pynabled.polar_compute(svd, out=out)

    assert returned is out
    np.testing.assert_allclose(out.u @ out.p, a, rtol=1e-10, atol=1e-12)
