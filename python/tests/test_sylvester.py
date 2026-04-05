"""Tests for Sylvester and Lyapunov solver bindings."""

import numpy as np
import pytest

import pynabled


def test_sylvester_solve():
    a = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float64)
    b = np.array([[3.0, 0.0], [0.0, 4.0]], dtype=np.float64)
    c = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    x = pynabled.sylvester_solve(a, b, c)
    np.testing.assert_allclose(a @ x + x @ b, c, rtol=1e-10)


def test_lyapunov_solve():
    # Lyapunov: nabled solves AX + XA^T = -Q (control theory convention).
    a = np.array([[-1.0, 0.0], [0.0, -2.0]], dtype=np.float64)
    q = np.eye(2, dtype=np.float64)
    x = pynabled.lyapunov_solve(a, q)
    residual = a @ x + x @ a.T
    np.testing.assert_allclose(residual, -q, rtol=1e-10)


def test_sylvester_and_lyapunov_accept_float32():
    a = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    b = np.array([[3.0, 0.0], [0.0, 4.0]], dtype=np.float32)
    c = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    stable = np.array([[-1.0, 0.0], [0.0, -2.0]], dtype=np.float32)
    q = np.eye(2, dtype=np.float32)

    x = pynabled.sylvester_solve(a, b, c)
    lyapunov = pynabled.lyapunov_solve(stable, q)

    assert x.dtype == np.float32
    assert lyapunov.dtype == np.float32
    np.testing.assert_allclose(a @ x + x @ b, c, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(
        stable @ lyapunov + lyapunov @ stable.T,
        -q,
        rtol=1e-4,
        atol=1e-5,
    )


def test_sylvester_and_lyapunov_accept_complex128():
    a = np.array([[1.0 + 1.0j, 0.0], [0.0, 2.0 - 0.5j]], dtype=np.complex128)
    b = np.array([[3.0 - 0.25j, 0.0], [0.0, 4.0 + 0.5j]], dtype=np.complex128)
    c = np.array([[1.0 + 0.5j, 1.0 - 1.0j], [1.0 + 1.0j, 1.0 - 0.25j]], dtype=np.complex128)
    stable = np.array([[-1.0 + 0.5j, 0.0], [0.0, -2.0 - 0.25j]], dtype=np.complex128)
    q = np.eye(2, dtype=np.complex128)

    x = pynabled.sylvester_solve(a, b, c)
    lyapunov = pynabled.lyapunov_solve(stable, q)

    assert x.dtype == np.complex128
    assert lyapunov.dtype == np.complex128
    np.testing.assert_allclose(a @ x + x @ b, c, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        stable @ lyapunov + lyapunov @ stable.conj().T,
        -q,
        rtol=1e-10,
        atol=1e-12,
    )
