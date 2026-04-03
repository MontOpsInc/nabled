"""Tests for ML bindings (regression, PCA, stats)."""

import numpy as np
import pynabled
import pytest


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


def test_pca_transform_and_inverse_transform():
    np.random.seed(42)
    x = np.random.randn(30, 3).astype(np.float64)
    components, _, _, mean, scores = pynabled.compute_pca(x, n_components=3)
    transformed = pynabled.pca_transform(x, components, mean)
    reconstructed = pynabled.pca_inverse_transform(scores, components, mean)
    np.testing.assert_allclose(transformed, scores, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(reconstructed, x, rtol=1e-9, atol=1e-9)


def test_compute_pca_complex():
    x = np.array(
        [[1 + 1j, 2 - 1j], [2 + 0j, 1 + 2j], [3 - 1j, 4 + 1j], [4 + 2j, 3 - 2j]],
        dtype=np.complex128,
    )
    components, ev, evr, mean, scores = pynabled.compute_pca_complex(x, n_components=2)
    transformed = pynabled.pca_transform_complex(x, components, mean)
    reconstructed = pynabled.pca_inverse_transform_complex(scores, components, mean)
    assert components.shape == (2, 2)
    assert ev.shape == (2,)
    assert evr.shape == (2,)
    assert mean.shape == (2,)
    np.testing.assert_allclose(transformed, scores, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(reconstructed, x, rtol=1e-8, atol=1e-8)


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


def test_linear_regression_complex():
    x = np.array([[1 + 0j], [2 + 0j], [3 + 0j], [4 + 0j]], dtype=np.complex128)
    y = (1.0 + 1.0j) + (2.0 - 0.5j) * x.flatten()
    coef, r_squared = pynabled.linear_regression_complex(x, y)
    assert coef.shape == (2,)
    np.testing.assert_allclose(coef[0], 1.0 + 1.0j, atol=1e-8)
    np.testing.assert_allclose(coef[1], 2.0 - 0.5j, atol=1e-8)
    np.testing.assert_allclose(r_squared, 1.0, atol=1e-10)


def test_complex_stats():
    x = np.array(
        [[1 + 1j, 2 - 1j], [3 + 0j, 4 + 1j], [5 - 2j, 6 + 0.5j]],
        dtype=np.complex128,
    )
    means = pynabled.column_means_complex(x)
    centered = pynabled.center_columns_complex(x)
    cov = pynabled.covariance_matrix_complex(x)
    corr = pynabled.correlation_matrix_complex(x)
    assert means.shape == (2,)
    assert centered.shape == x.shape
    assert cov.shape == (2, 2)
    assert corr.shape == (2, 2)
    np.testing.assert_allclose(pynabled.column_means_complex(centered), [0j, 0j], atol=1e-12)


def test_pca_and_stats_accept_fortran_order_inputs():
    x = np.asfortranarray(np.random.randn(24, 3).astype(np.float64))
    assert x.flags["F_CONTIGUOUS"]
    assert not x.flags["C_CONTIGUOUS"]
    components, _, _, mean, scores = pynabled.compute_pca(x, n_components=2)
    transformed = pynabled.pca_transform(x, components, mean)
    np.testing.assert_allclose(transformed, scores, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(pynabled.column_means(x), x.mean(axis=0), rtol=1e-12, atol=1e-12)
