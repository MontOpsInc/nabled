"""Tests for Cholesky decomposition bindings."""

import numpy as np
import pynabled
import pytest


def _make_spd(n, dtype=np.float64):
    """Create a symmetric positive definite matrix."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal((n, n)).astype(dtype)
    return x.T @ x + np.array(0.1, dtype=dtype) * np.eye(n, dtype=dtype)


def _make_hpd(n):
    rng = np.random.default_rng(42)
    x = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    x = x.astype(np.complex128)
    return x.conj().T @ x + np.array(0.1, dtype=np.complex128) * np.eye(n, dtype=np.complex128)


def test_cholesky_decompose():
    a = _make_spd(3)
    result = pynabled.cholesky_decompose(a)
    assert result.l.shape == (3, 3)
    np.testing.assert_allclose(result.l @ result.l.T, a, rtol=1e-10)
    assert np.allclose(np.tril(result.l), result.l)


def test_cholesky_solve():
    a = _make_spd(3)
    b = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    x = pynabled.cholesky_solve(a, b)
    np.testing.assert_allclose(a @ x, b, rtol=1e-10)


def test_cholesky_solve_supports_out_and_factor_reuse():
    a = _make_spd(3, dtype=np.float32)
    b = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    factor = pynabled.cholesky_decompose(a)
    direct_out = np.empty(3, dtype=np.float32)
    out = np.empty(3, dtype=np.float32)

    returned_direct = pynabled.cholesky_solve(a, b, out=direct_out)
    returned = pynabled.cholesky_solve(factor, b, out=out)

    assert returned_direct is direct_out
    assert returned is out
    np.testing.assert_allclose(a @ direct_out, b, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(a @ out, b, rtol=1e-4, atol=1e-5)


def test_cholesky_inverse():
    a = _make_spd(3)
    inv_a = pynabled.cholesky_inverse(a)
    np.testing.assert_allclose(a @ inv_a, np.eye(3), rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(inv_a @ a, np.eye(3), rtol=1e-10, atol=1e-14)


def test_cholesky_inverse_supports_out_and_factor_reuse():
    a = _make_hpd(3)
    factor = pynabled.cholesky_decompose(a)
    direct_out = np.empty((3, 3), dtype=np.complex128, order="F")
    out = np.empty((3, 3), dtype=np.complex128, order="F")

    returned_direct = pynabled.cholesky_inverse(a, out=direct_out)
    returned = pynabled.cholesky_inverse(factor, out=out)

    assert returned_direct is direct_out
    assert returned is out
    np.testing.assert_allclose(
        a @ direct_out,
        np.eye(3, dtype=np.complex128),
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(a @ out, np.eye(3, dtype=np.complex128), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(out @ a, np.eye(3, dtype=np.complex128), rtol=1e-10, atol=1e-12)


def test_cholesky_accepts_float32():
    a = _make_spd(3, dtype=np.float32)
    b = np.array([1.0, -2.0, 3.0], dtype=np.float32)

    result = pynabled.cholesky_decompose(a)
    x = pynabled.cholesky_solve(a, b)
    inv_a = pynabled.cholesky_inverse(a)

    assert result.l.dtype == np.float32
    assert x.dtype == np.float32
    assert inv_a.dtype == np.float32
    np.testing.assert_allclose(result.l @ result.l.T, a, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(a @ x, b, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(a @ inv_a, np.eye(3, dtype=np.float32), rtol=1e-4, atol=1e-5)


def test_cholesky_accepts_complex128():
    a = _make_hpd(3)
    b = np.array([1.0 + 0.5j, -2.0 + 1.0j, 3.0 - 0.25j], dtype=np.complex128)

    result = pynabled.cholesky_decompose(a)
    x = pynabled.cholesky_solve(a, b)
    inv_a = pynabled.cholesky_inverse(a)

    assert result.l.dtype == np.complex128
    assert x.dtype == np.complex128
    assert inv_a.dtype == np.complex128
    np.testing.assert_allclose(result.l @ result.l.conj().T, a, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(a @ x, b, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(a @ inv_a, np.eye(3, dtype=np.complex128), rtol=1e-10, atol=1e-12)
