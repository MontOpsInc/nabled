"""Tests for matrix function bindings."""

import numpy as np
import pytest

import pynabled


def _make_spd(n, dtype=np.float64):
    rng = np.random.default_rng(42)
    x = rng.standard_normal((n, n)).astype(dtype)
    return x.T @ x + np.array(0.5, dtype=dtype) * np.eye(n, dtype=dtype)


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


def test_matrix_functions_accept_float32():
    rotation = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    diagonal = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    identity_like = np.eye(2, dtype=np.float32) + np.array(0.1, dtype=np.float32) * np.ones(
        (2, 2), dtype=np.float32
    )
    spd = _make_spd(2, dtype=np.float32)
    symmetric = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float32)
    sign_input = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float32)

    exp_taylor = pynabled.matrix_exp(rotation, None, None)
    exp_eigen = pynabled.matrix_exp_eigen(diagonal)
    log_taylor = pynabled.matrix_log_taylor(identity_like, None, None)
    log_eigen = pynabled.matrix_log_eigen(spd)
    log_svd = pynabled.matrix_log_svd(spd)
    power = pynabled.matrix_power(symmetric, 2.0)
    sign = pynabled.matrix_sign(sign_input)

    for result in (exp_taylor, exp_eigen, log_taylor, log_eigen, log_svd, power, sign):
        assert result.dtype == np.float32

    np.testing.assert_allclose(
        exp_taylor.T @ exp_taylor, np.eye(2, dtype=np.float32), rtol=1e-4, atol=1e-5
    )
    np.testing.assert_allclose(
        exp_eigen, np.array([[np.e, 0.0], [0.0, np.e**2]], dtype=np.float32), rtol=2e-4, atol=2e-5
    )
    np.testing.assert_allclose(
        pynabled.matrix_exp(log_taylor, None, None),
        identity_like,
        rtol=2e-4,
        atol=2e-5,
    )
    np.testing.assert_allclose(pynabled.matrix_exp_eigen(log_eigen), spd, rtol=2e-4, atol=2e-5)
    np.testing.assert_allclose(pynabled.matrix_exp_eigen(log_svd), spd, rtol=3e-4, atol=3e-5)
    np.testing.assert_allclose(power, symmetric @ symmetric, rtol=2e-4, atol=2e-5)
    np.testing.assert_allclose(sign, sign_input, rtol=1e-4, atol=1e-5)
