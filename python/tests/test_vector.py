"""Tests for vector primitive bindings."""

import numpy as np
import pynabled
import pytest


def test_l2_norm():
    v = np.array([3.0, 4.0], dtype=np.float64)
    n = pynabled.l2_norm(v)
    assert abs(n - 5.0) < 1e-14


def test_cosine_similarity():
    a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    b = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    sim = pynabled.cosine_similarity(a, b)
    assert abs(sim - 1.0) < 1e-14

    a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    b = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
    sim = pynabled.cosine_similarity(a, b)
    assert abs(sim - (-1.0)) < 1e-14


def test_pairwise_l2_distance():
    left = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    right = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    d = pynabled.pairwise_l2_distance(left, right)
    assert d.shape == (2, 2)
    assert np.all(d >= 0)


def test_pairwise_cosine_similarity():
    left = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    right = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    sim = pynabled.pairwise_cosine_similarity(left, right)
    assert sim.shape == (2, 2)
    np.testing.assert_allclose(sim, np.eye(2), rtol=1e-14)


def test_vector_primitives_accept_float32():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    left = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    right = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)

    assert abs(pynabled.dot(a, b) - 32.0) < 1e-5
    assert abs(pynabled.l2_norm(a) - np.linalg.norm(a)) < 1e-5
    assert abs(pynabled.cosine_similarity(a, b) - 0.97463185) < 1e-5

    distances = pynabled.pairwise_l2_distance(left, right)
    similarities = pynabled.pairwise_cosine_similarity(left, left)
    assert distances.dtype == np.float32
    assert similarities.dtype == np.float32
    np.testing.assert_allclose(distances, np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32))
    np.testing.assert_allclose(similarities, np.eye(2, dtype=np.float32), rtol=1e-5, atol=1e-6)
