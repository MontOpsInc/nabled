"""Tests for triangular solve bindings."""

import numpy as np
import pynabled
import pytest


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


def test_triangular_accepts_float32():
    lower = np.array([[1.0, 0.0], [2.0, 3.0]], dtype=np.float32)
    upper = np.array([[1.0, 2.0], [0.0, 3.0]], dtype=np.float32)
    vector = np.array([1.0, 2.0], dtype=np.float32)
    matrix_rhs = np.eye(2, dtype=np.float32)

    lower_x = pynabled.triangular_solve_lower(lower, vector)
    upper_x = pynabled.triangular_solve_upper(upper, vector)
    lower_matrix = pynabled.triangular_solve_lower_matrix(lower, matrix_rhs)
    upper_matrix = pynabled.triangular_solve_upper_matrix(upper, matrix_rhs)

    assert lower_x.dtype == np.float32
    assert upper_x.dtype == np.float32
    assert lower_matrix.dtype == np.float32
    assert upper_matrix.dtype == np.float32
    np.testing.assert_allclose(lower @ lower_x, vector, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(upper @ upper_x, vector, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(lower @ lower_matrix, matrix_rhs, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(upper @ upper_matrix, matrix_rhs, rtol=1e-4, atol=1e-5)


def test_triangular_accepts_complex128_vector_rhs():
    lower = np.array(
        [[2.0 + 0.0j, 0.0 + 0.0j], [1.0 - 1.0j, 3.0 + 0.5j]],
        dtype=np.complex128,
    )
    upper = np.array(
        [[2.5 + 0.25j, -1.0 + 0.5j], [0.0 + 0.0j, 1.5 - 0.25j]],
        dtype=np.complex128,
    )
    rhs = np.array([2.0 + 1.0j, 4.0 - 2.0j], dtype=np.complex128)

    lower_x = pynabled.triangular_solve_lower(lower, rhs)
    upper_x = pynabled.triangular_solve_upper(upper, rhs)

    assert lower_x.dtype == np.complex128
    assert upper_x.dtype == np.complex128
    np.testing.assert_allclose(lower @ lower_x, rhs, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(upper @ upper_x, rhs, rtol=1e-10, atol=1e-12)
