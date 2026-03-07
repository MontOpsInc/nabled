"""Tests for ML bindings (regression, PCA, stats)."""

import numpy as np
import pytest

import pynabled


def test_linear_regression():
    np.random.seed(42)
    n = 100
    x = np.random.randn(n, 1).astype(np.float64)
    y = 2.0 * x.flatten() + 1.0 + 0.1 * np.random.randn(n).astype(np.float64)
    # linear_regression adds intercept automatically
    coef, r_squared = pynabled.linear_regression(x, y)
    assert coef.shape == (2,)  # [intercept, slope]
    np.testing.assert_allclose(coef[1], 2.0, rtol=0.5)
    assert 0 <= r_squared <= 1


def test_compute_pca():
    np.random.seed(42)
    x = np.random.randn(50, 4).astype(np.float64)
    components, ev, evr, mean, scores = pynabled.compute_pca(x, n_components=2)
    assert components.shape == (2, 4)
    assert ev.shape == (2,)
    assert evr.shape == (2,)
    assert mean.shape == (4,)
    assert scores.shape == (50, 2)
    assert evr.sum() <= 1.0 + 1e-10


def test_column_means():
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    means = pynabled.column_means(x)
    assert means.shape == (2,)
    np.testing.assert_allclose(means, [3.0, 4.0], rtol=1e-14)


def test_center_columns():
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    centered = pynabled.center_columns(x)
    means = pynabled.column_means(centered)
    np.testing.assert_allclose(means, [0.0, 0.0], atol=1e-14)


def test_covariance_matrix():
    np.random.seed(42)
    x = np.random.randn(100, 3).astype(np.float64)
    cov = pynabled.covariance_matrix(x)
    assert cov.shape == (3, 3)
    np.testing.assert_allclose(cov, cov.T, rtol=1e-14)


def test_correlation_matrix():
    np.random.seed(42)
    x = np.random.randn(100, 3).astype(np.float64)
    corr = pynabled.correlation_matrix(x)
    assert corr.shape == (3, 3)
    np.testing.assert_allclose(corr, corr.T, rtol=1e-14)
    np.testing.assert_allclose(np.diag(corr), np.ones(3), atol=1e-14)
    assert np.all(np.abs(corr) <= 1.0 + 1e-10)
