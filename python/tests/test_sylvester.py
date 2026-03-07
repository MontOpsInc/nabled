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
