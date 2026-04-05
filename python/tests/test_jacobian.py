"""Tests for Jacobian / derivative bindings."""

import numpy as np
import pynabled


def test_numerical_jacobian():
    def func(x):
        return np.array([x[0] ** 2, x[1] ** 2], dtype=np.float64)

    x = np.array([2.0, 3.0], dtype=np.float64)
    jac = pynabled.numerical_jacobian(func, x)
    np.testing.assert_allclose(jac, np.array([[4.0, 0.0], [0.0, 6.0]]), rtol=1e-4)


def test_numerical_jacobian_central():
    def func(x):
        return np.array([np.sin(x[0]), x[0] * x[1]], dtype=np.float64)

    x = np.array([0.5, 2.0], dtype=np.float64)
    jac = pynabled.numerical_jacobian_central(func, x)
    expected = np.array([[np.cos(0.5), 0.0], [2.0, 0.5]], dtype=np.float64)
    np.testing.assert_allclose(jac, expected, rtol=1e-4, atol=1e-6)


def test_numerical_gradient():
    def func(x):
        return float((x[0] - 2.0) ** 2 + 3.0 * (x[1] + 1.0) ** 2)

    x = np.array([2.5, -0.5], dtype=np.float64)
    grad = pynabled.numerical_gradient(func, x)
    expected = np.array([1.0, 3.0], dtype=np.float64)
    np.testing.assert_allclose(grad, expected, rtol=1e-4, atol=1e-6)


def test_numerical_hessian():
    def func(x):
        return float(x[0] ** 2 + 3.0 * x[0] * x[1] + 2.0 * x[1] ** 2)

    x = np.array([1.0, -2.0], dtype=np.float64)
    hess = pynabled.numerical_hessian(func, x)
    expected = np.array([[2.0, 3.0], [3.0, 4.0]], dtype=np.float64)
    np.testing.assert_allclose(hess, expected, rtol=5e-4, atol=1e-3)


def test_real_jacobian_bindings_accept_float32():
    def vector_func(x):
        return np.array([x[0] ** 2, x[0] * x[1]], dtype=np.float32)

    def scalar_func(x):
        return np.float32((x[0] - 1.5) ** 2 + np.float32(2.0) * (x[1] + 0.25) ** 2)

    x = np.array([2.0, -0.5], dtype=np.float32)

    jac = pynabled.numerical_jacobian(vector_func, x)
    jac_central = pynabled.numerical_jacobian_central(vector_func, x)
    grad = pynabled.numerical_gradient(scalar_func, x)
    hess = pynabled.numerical_hessian(scalar_func, x)

    assert jac.dtype == np.float32
    assert jac_central.dtype == np.float32
    assert grad.dtype == np.float32
    assert hess.dtype == np.float32
    np.testing.assert_allclose(
        jac,
        np.array([[4.0, 0.0], [-0.5, 2.0]], dtype=np.float32),
        rtol=2e-3,
        atol=5e-3,
    )
    np.testing.assert_allclose(
        jac_central,
        np.array([[4.0, 0.0], [-0.5, 2.0]], dtype=np.float32),
        rtol=2e-3,
        atol=5e-3,
    )
    np.testing.assert_allclose(
        grad,
        np.array([1.0, -1.0], dtype=np.float32),
        rtol=3e-3,
        atol=3e-3,
    )
    np.testing.assert_allclose(
        hess,
        np.array([[2.0, 0.0], [0.0, 4.0]], dtype=np.float32),
        rtol=2e-2,
        atol=2e-2,
    )
