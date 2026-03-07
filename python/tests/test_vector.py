"""Tests for vector primitive bindings."""

import numpy as np
import pytest

import pynabled


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
