"""Tests for optimization bindings."""

import numpy as np

import pynabled


TARGET = np.array([3.0], dtype=np.float64)
COMPLEX_TARGET = np.array([3.0 + 2.0j], dtype=np.complex128)


def objective(x):
    delta = x - TARGET
    return float(np.dot(delta, delta))


def gradient(x):
    return 2.0 * (x - TARGET)


def complex_objective(x):
    delta = x - COMPLEX_TARGET
    return float(np.vdot(delta, delta).real)


def complex_gradient(x):
    return 2.0 * (x - COMPLEX_TARGET)


def test_backtracking_line_search():
    point = np.array([0.0], dtype=np.float64)
    direction = np.array([1.0], dtype=np.float64)
    step = pynabled.backtracking_line_search(point, direction, objective, gradient)
    assert step > 0
    assert objective(point + step * direction) < objective(point)


def test_gradient_descent():
    initial = np.array([0.0], dtype=np.float64)
    optimum = pynabled.gradient_descent(initial, objective, gradient, learning_rate=0.1)
    np.testing.assert_allclose(optimum, TARGET, atol=1e-4)


def test_adam():
    initial = np.array([0.0], dtype=np.float64)
    optimum = pynabled.adam(initial, objective, gradient, learning_rate=0.1)
    np.testing.assert_allclose(optimum, TARGET, atol=1e-3)


def test_projected_gradient_descent_box():
    initial = np.array([0.0], dtype=np.float64)
    lower = np.array([0.0], dtype=np.float64)
    upper = np.array([2.5], dtype=np.float64)
    optimum = pynabled.projected_gradient_descent_box(
        initial, objective, gradient, lower, upper, learning_rate=0.1
    )
    np.testing.assert_allclose(optimum, upper, atol=1e-4)


def test_stochastic_gradient_descent():
    initial = np.array([0.0], dtype=np.float64)

    def stochastic_grad(x, _iteration):
        return gradient(x)

    optimum = pynabled.stochastic_gradient_descent(
        initial, stochastic_grad, learning_rate=0.1
    )
    np.testing.assert_allclose(optimum, TARGET, atol=1e-4)


def test_bfgs():
    initial = np.array([0.0], dtype=np.float64)
    optimum = pynabled.bfgs(initial, objective, gradient, step_size=0.5)
    np.testing.assert_allclose(optimum, TARGET, atol=1e-4)


def test_complex_gradient_descent():
    initial = np.array([0.0 + 0.0j], dtype=np.complex128)
    optimum = pynabled.gradient_descent_complex(
        initial, complex_objective, complex_gradient, learning_rate=0.1
    )
    np.testing.assert_allclose(optimum, COMPLEX_TARGET, atol=1e-4)
