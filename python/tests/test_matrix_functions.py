"""Tests for matrix function bindings."""

import numpy as np
import pytest

import pynabled


def _make_spd(n):
    np.random.seed(42)
    x = np.random.randn(n, n).astype(np.float64)
    return x.T @ x + 0.5 * np.eye(n)


def test_matrix_exp():
    a = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float64)  # rotation generator
    exp_a = pynabled.matrix_exp(a, None, None)
    assert exp_a.shape == (2, 2)
    # exp of skew-symmetric is orthogonal
    np.testing.assert_allclose(exp_a.T @ exp_a, np.eye(2), rtol=1e-10, atol=1e-14)


def test_matrix_exp_eigen():
    a = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float64)
    exp_a = pynabled.matrix_exp_eigen(a)
    expected = np.array([[np.e, 0.0], [0.0, np.e**2]], dtype=np.float64)
    np.testing.assert_allclose(exp_a, expected, rtol=1e-10)


def test_matrix_log_taylor():
    # Use identity-like matrix for Taylor log convergence
    a = np.eye(2) + 0.1 * np.ones((2, 2))
    log_a = pynabled.matrix_log_taylor(a, None, None)
    assert np.all(np.isfinite(log_a))
    exp_log = pynabled.matrix_exp(log_a, None, None)
    np.testing.assert_allclose(exp_log, a, rtol=1e-8)


def test_matrix_log_eigen():
    a = _make_spd(2)
    log_a = pynabled.matrix_log_eigen(a)
    exp_log = pynabled.matrix_exp_eigen(log_a)
    np.testing.assert_allclose(exp_log, a, rtol=1e-9)


def test_matrix_log_svd():
    a = _make_spd(2)
    log_a = pynabled.matrix_log_svd(a)
    exp_log = pynabled.matrix_exp_eigen(log_a)
    np.testing.assert_allclose(exp_log, a, rtol=1e-9)


def test_matrix_power():
    a = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float64)  # symmetric
    a2 = pynabled.matrix_power(a, 2.0)
    np.testing.assert_allclose(a2, a @ a, rtol=1e-10)


def test_matrix_sign():
    a = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float64)
    s = pynabled.matrix_sign(a)
    # sign of diag(1,-1) is diag(1,-1)
    np.testing.assert_allclose(s @ s, np.eye(2), rtol=1e-10)
    np.testing.assert_allclose(s, a, rtol=1e-10)
