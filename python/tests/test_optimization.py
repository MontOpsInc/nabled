"""Tests for optimization bindings."""

import numpy as np
import pytest
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
    step = pynabled.backtracking_line_search(
        point,
        direction,
        objective,
        gradient,
        config=pynabled.LineSearchConfig(initial_step=1.0, max_iterations=32),
    )
    assert step > 0
    assert objective(point + step * direction) < objective(point)


def test_gradient_descent():
    initial = np.array([0.0], dtype=np.float64)
    optimum = pynabled.gradient_descent(
        initial,
        objective,
        gradient,
        config=pynabled.GradientDescentConfig(learning_rate=0.1, max_iterations=200),
    )
    np.testing.assert_allclose(optimum, TARGET, atol=1e-4)


def test_adam():
    initial = np.array([0.0], dtype=np.float64)
    optimum = pynabled.adam(
        initial,
        objective,
        gradient,
        config=pynabled.AdamConfig(learning_rate=0.1),
    )
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

    optimum = pynabled.stochastic_gradient_descent(initial, stochastic_grad, learning_rate=0.1)
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


def test_real_optimization_bindings_accept_float32():
    target = np.array([3.0], dtype=np.float32)

    def objective32(x):
        delta = x - target
        return np.dot(delta, delta).astype(np.float32)

    def gradient32(x):
        return np.float32(2.0) * (x - target)

    initial = np.array([0.0], dtype=np.float32)
    direction = np.array([1.0], dtype=np.float32)
    lower = np.array([0.0], dtype=np.float32)
    upper = np.array([2.5], dtype=np.float32)

    line_search = pynabled.backtracking_line_search(initial, direction, objective32, gradient32)
    optimum_gd = pynabled.gradient_descent(initial, objective32, gradient32, learning_rate=0.1)
    optimum_adam = pynabled.adam(initial, objective32, gradient32, learning_rate=0.1)
    optimum_momentum = pynabled.momentum_descent(
        initial, objective32, gradient32, learning_rate=0.05, momentum=0.8
    )
    optimum_rmsprop = pynabled.rmsprop(initial, objective32, gradient32, learning_rate=0.05)
    optimum_projected = pynabled.projected_gradient_descent_box(
        initial, objective32, gradient32, lower, upper, learning_rate=0.1
    )

    def stochastic_gradient32(x, _iteration):
        return gradient32(x)

    optimum_sgd = pynabled.stochastic_gradient_descent(
        initial, stochastic_gradient32, learning_rate=0.1
    )
    optimum_bfgs = pynabled.bfgs(initial, objective32, gradient32, step_size=0.5)

    assert line_search > 0
    for optimum in (
        optimum_gd,
        optimum_adam,
        optimum_momentum,
        optimum_rmsprop,
        optimum_sgd,
        optimum_bfgs,
    ):
        assert optimum.dtype == np.float32
        np.testing.assert_allclose(optimum, target, atol=2e-3, rtol=2e-3)

    assert optimum_projected.dtype == np.float32
    np.testing.assert_allclose(optimum_projected, upper, atol=2e-3, rtol=2e-3)


def test_optimization_rejects_config_and_explicit_kwargs():
    initial = np.array([0.0], dtype=np.float64)

    with pytest.raises(TypeError, match="config="):
        pynabled.gradient_descent(
            initial,
            objective,
            gradient,
            learning_rate=0.1,
            config=pynabled.GradientDescentConfig(max_iterations=128),
        )
