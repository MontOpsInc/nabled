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


def test_lu_log_determinant():
    a = np.array([[4.0, 1.0], [2.0, 3.0]], dtype=np.float64)

    result = pynabled.lu_log_determinant(a)
    sign, ln_abs_det = np.linalg.slogdet(a)

    assert result.sign == int(sign)
    np.testing.assert_allclose(result.ln_abs_det, ln_abs_det, rtol=1e-10, atol=1e-12)


def test_lu_accepts_float32():
    a = np.array([[4.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    b = np.array([1.0, 2.0], dtype=np.float32)

    result = pynabled.lu_decompose(a)
    x = pynabled.lu_solve(a, b)
    inv_a = pynabled.lu_inverse(a)
    det = pynabled.lu_determinant(a)
    log_det = pynabled.lu_log_determinant(a)

    assert result.l.dtype == np.float32
    assert result.u.dtype == np.float32
    assert x.dtype == np.float32
    assert inv_a.dtype == np.float32
    assert log_det.sign == 1
    np.testing.assert_allclose(a @ x, b, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(a @ inv_a, np.eye(2, dtype=np.float32), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(det, np.linalg.det(a).astype(np.float64), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(log_det.ln_abs_det, np.linalg.slogdet(a)[1], rtol=1e-5, atol=1e-6)


def test_lu_solve_inverse_and_determinant_accept_complex128():
    a = np.array([[2.0 + 1.0j, 1.0 - 0.5j], [0.0 + 0.25j, 3.0 - 1.0j]], dtype=np.complex128)
    b = np.array([1.0 + 2.0j, -0.5 + 1.0j], dtype=np.complex128)

    x = pynabled.lu_solve(a, b)
    inv_a = pynabled.lu_inverse(a)
    det = pynabled.lu_determinant(a)

    assert x.dtype == np.complex128
    assert inv_a.dtype == np.complex128
    assert isinstance(det, complex)
    np.testing.assert_allclose(a @ x, b, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(a @ inv_a, np.eye(2, dtype=np.complex128), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(det, np.linalg.det(a), rtol=1e-10, atol=1e-12)
