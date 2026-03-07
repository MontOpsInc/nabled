"""Tests for triangular solve bindings."""

import numpy as np
import pytest

import pynabled


def test_triangular_solve_lower():
    l = np.array([[1.0, 0.0, 0.0], [2.0, 3.0, 0.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    b = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    x = pynabled.triangular_solve_lower(l, b)
    np.testing.assert_allclose(l @ x, b, rtol=1e-10)


def test_triangular_solve_upper():
    u = np.array([[1.0, 2.0, 3.0], [0.0, 4.0, 5.0], [0.0, 0.0, 6.0]], dtype=np.float64)
    b = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    x = pynabled.triangular_solve_upper(u, b)
    np.testing.assert_allclose(u @ x, b, rtol=1e-10)


def test_triangular_solve_lower_matrix():
    l = np.array([[1.0, 0.0], [2.0, 3.0]], dtype=np.float64)
    b = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    x = pynabled.triangular_solve_lower_matrix(l, b)
    np.testing.assert_allclose(l @ x, b, rtol=1e-10, atol=1e-14)


def test_triangular_solve_upper_matrix():
    u = np.array([[1.0, 2.0], [0.0, 3.0]], dtype=np.float64)
    b = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    x = pynabled.triangular_solve_upper_matrix(u, b)
    np.testing.assert_allclose(u @ x, b, rtol=1e-10)
