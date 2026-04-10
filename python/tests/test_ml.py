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
    result = pynabled.linear_regression(x, y)
    assert result.coefficients.shape == (2,)  # [intercept, slope]
    assert result.fitted_values.shape == y.shape
    assert result.residuals.shape == y.shape
    np.testing.assert_allclose(result.coefficients[1], 2.0, rtol=0.5)
    assert 0 <= result.r_squared <= 1


def test_compute_pca():
    np.random.seed(42)
    x = np.random.randn(50, 4).astype(np.float64)
    result = pynabled.compute_pca(x, n_components=2)
    assert result.components.shape == (2, 4)
    assert result.explained_variance.shape == (2,)
    assert result.explained_variance_ratio.shape == (2,)
    assert result.mean.shape == (4,)
    assert result.scores.shape == (50, 2)
    assert result.explained_variance_ratio.sum() <= 1.0 + 1e-10


def test_linear_regression_and_pca_reuse_result_buffers():
    np.random.seed(42)
    x = np.random.randn(32, 3).astype(np.float64)
    weights = np.array([1.5, -0.25, 0.75], dtype=np.float64)
    y = x @ weights + 0.5

    regression_out = pynabled.RegressionResult(
        coefficients=np.empty(4, dtype=np.float64),
        fitted_values=np.empty(x.shape[0], dtype=np.float64),
        residuals=np.empty(x.shape[0], dtype=np.float64),
        r_squared=float("nan"),
    )
    pca_out = pynabled.PcaResult(
        components=np.empty((2, x.shape[1]), dtype=np.float64, order="F"),
        explained_variance=np.empty(2, dtype=np.float64),
        explained_variance_ratio=np.empty(2, dtype=np.float64),
        mean=np.empty(x.shape[1], dtype=np.float64),
        scores=np.empty((x.shape[0], 2), dtype=np.float64, order="F"),
    )

    regression = pynabled.linear_regression(x, y, out=regression_out)
    pca = pynabled.compute_pca(x, n_components=2, out=pca_out)

    assert regression is regression_out
    assert pca is pca_out
    assert pca.components.flags["F_CONTIGUOUS"]
    assert pca.scores.flags["F_CONTIGUOUS"]

    expected_regression = pynabled.linear_regression(x, y)
    expected_pca = pynabled.compute_pca(x, n_components=2)
    np.testing.assert_allclose(regression.coefficients, expected_regression.coefficients)
    np.testing.assert_allclose(regression.fitted_values, expected_regression.fitted_values)
    np.testing.assert_allclose(regression.residuals, expected_regression.residuals)
    np.testing.assert_allclose(regression.r_squared, expected_regression.r_squared)
    np.testing.assert_allclose(pca.components, expected_pca.components)
    np.testing.assert_allclose(pca.explained_variance, expected_pca.explained_variance)
    np.testing.assert_allclose(pca.explained_variance_ratio, expected_pca.explained_variance_ratio)
    np.testing.assert_allclose(pca.mean, expected_pca.mean)
    np.testing.assert_allclose(pca.scores, expected_pca.scores)


def test_pca_transform_and_inverse_transform():
    np.random.seed(42)
    x = np.random.randn(30, 3).astype(np.float64)
    result = pynabled.compute_pca(x, n_components=3)
    transformed = pynabled.pca_transform(x, result)
    reconstructed = pynabled.pca_inverse_transform(result.scores, result)
    np.testing.assert_allclose(transformed, result.scores, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(reconstructed, x, rtol=1e-9, atol=1e-9)


def test_compute_pca_complex():
    x = np.array(
        [[1 + 1j, 2 - 1j], [2 + 0j, 1 + 2j], [3 - 1j, 4 + 1j], [4 + 2j, 3 - 2j]],
        dtype=np.complex128,
    )
    result = pynabled.compute_pca_complex(x, n_components=2)
    transformed = pynabled.pca_transform_complex(x, result)
    reconstructed = pynabled.pca_inverse_transform_complex(result.scores, result)
    assert result.components.shape == (2, 2)
    assert result.explained_variance.shape == (2,)
    assert result.explained_variance_ratio.shape == (2,)
    assert result.mean.shape == (2,)
    np.testing.assert_allclose(transformed, result.scores, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(reconstructed, x, rtol=1e-8, atol=1e-8)


def test_pca_transform_and_inverse_transform_reuse_output_buffers():
    np.random.seed(42)
    x = np.random.randn(30, 3).astype(np.float64)
    result = pynabled.compute_pca(x, n_components=3)

    transformed_out = np.empty_like(result.scores, order="F")
    reconstructed_out = np.empty_like(x, order="F")

    transformed = pynabled.pca_transform(x, result, out=transformed_out)
    reconstructed = pynabled.pca_inverse_transform(result.scores, result, out=reconstructed_out)

    assert transformed is transformed_out
    assert reconstructed is reconstructed_out
    assert transformed.flags["F_CONTIGUOUS"]
    assert reconstructed.flags["F_CONTIGUOUS"]
    np.testing.assert_allclose(transformed, result.scores, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(reconstructed, x, rtol=1e-9, atol=1e-9)


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
    result = pynabled.linear_regression_complex(x, y)
    assert result.coefficients.shape == (2,)
    np.testing.assert_allclose(result.coefficients[0], 1.0 + 1.0j, atol=1e-8)
    np.testing.assert_allclose(result.coefficients[1], 2.0 - 0.5j, atol=1e-8)
    np.testing.assert_allclose(result.r_squared, 1.0, atol=1e-10)


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


def test_complex_ml_bindings_accept_strided_inputs_and_pca_out():
    x_base = np.array(
        [[1 + 1j, 2 - 1j], [2 + 0j, 1 + 2j], [3 - 1j, 4 + 1j], [4 + 2j, 3 - 2j]],
        dtype=np.complex128,
    )
    x = np.asfortranarray(x_base)
    assert x.flags["F_CONTIGUOUS"]
    assert not x.flags["C_CONTIGUOUS"]

    y_storage = np.empty(x.shape[0] * 2, dtype=np.complex128)
    y_storage[::2] = (1.0 + 1.0j) + (2.0 - 0.5j) * x[:, 0]
    y = y_storage[::2]
    assert not y.flags["C_CONTIGUOUS"]

    regression_out = pynabled.RegressionResult(
        coefficients=np.empty(2, dtype=np.complex128),
        fitted_values=np.empty(x.shape[0], dtype=np.complex128),
        residuals=np.empty(x.shape[0], dtype=np.complex128),
        r_squared=float("nan"),
    )
    regression = pynabled.linear_regression_complex(x[:, :1], y, out=regression_out)
    means = pynabled.column_means_complex(x)
    centered = pynabled.center_columns_complex(x)
    covariance = pynabled.covariance_matrix_complex(x)
    correlation = pynabled.correlation_matrix_complex(x)
    pca_out = pynabled.PcaResult(
        components=np.empty((2, x.shape[1]), dtype=np.complex128, order="F"),
        explained_variance=np.empty(2, dtype=np.float64),
        explained_variance_ratio=np.empty(2, dtype=np.float64),
        mean=np.empty(x.shape[1], dtype=np.complex128),
        scores=np.empty((x.shape[0], 2), dtype=np.complex128, order="F"),
    )
    pca = pynabled.compute_pca_complex(x, n_components=2, out=pca_out)

    transformed_out = np.empty_like(pca.scores, order="F")
    reconstructed_out = np.empty_like(x, order="F")
    transformed = pynabled.pca_transform_complex(x, pca, out=transformed_out)
    reconstructed = pynabled.pca_inverse_transform_complex(pca.scores, pca, out=reconstructed_out)

    assert regression is regression_out
    assert pca is pca_out
    assert transformed is transformed_out
    assert reconstructed is reconstructed_out
    assert pca.components.flags["F_CONTIGUOUS"]
    assert pca.scores.flags["F_CONTIGUOUS"]
    assert transformed.flags["F_CONTIGUOUS"]
    assert reconstructed.flags["F_CONTIGUOUS"]
    np.testing.assert_allclose(regression.coefficients[0], 1.0 + 1.0j, atol=1e-8)
    np.testing.assert_allclose(regression.coefficients[1], 2.0 - 0.5j, atol=1e-8)
    np.testing.assert_allclose(means, x.mean(axis=0), atol=1e-12)
    np.testing.assert_allclose(pynabled.column_means_complex(centered), [0j, 0j], atol=1e-12)
    np.testing.assert_allclose(covariance, covariance.T.conj(), atol=1e-12)
    np.testing.assert_allclose(correlation, correlation.T.conj(), atol=1e-12)
    np.testing.assert_allclose(transformed, pca.scores, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(reconstructed, x, rtol=1e-8, atol=1e-8)


def test_pca_and_stats_accept_fortran_order_inputs():
    x = np.asfortranarray(np.random.randn(24, 3).astype(np.float64))
    assert x.flags["F_CONTIGUOUS"]
    assert not x.flags["C_CONTIGUOUS"]
    result = pynabled.compute_pca(x, n_components=2)
    transformed = pynabled.pca_transform(x, result)
    np.testing.assert_allclose(transformed, result.scores, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(pynabled.column_means(x), x.mean(axis=0), rtol=1e-12, atol=1e-12)


def test_real_ml_bindings_accept_float32():
    np.random.seed(7)
    x = np.random.randn(64, 3).astype(np.float32)
    weights = np.array([0.5, -1.25, 2.0], dtype=np.float32)
    y = x @ weights + np.float32(0.75)

    regression = pynabled.linear_regression(x, y)
    pca = pynabled.compute_pca(x, n_components=3)
    transformed = pynabled.pca_transform(x, pca)
    reconstructed = pynabled.pca_inverse_transform(pca.scores, pca)
    means = pynabled.column_means(x)
    centered = pynabled.center_columns(x)
    covariance = pynabled.covariance_matrix(x)
    correlation = pynabled.correlation_matrix(x)

    assert regression.coefficients.dtype == np.float32
    assert regression.fitted_values.dtype == np.float32
    assert regression.residuals.dtype == np.float32
    assert pca.components.dtype == np.float32
    assert pca.explained_variance.dtype == np.float32
    assert pca.explained_variance_ratio.dtype == np.float32
    assert pca.mean.dtype == np.float32
    assert pca.scores.dtype == np.float32
    assert transformed.dtype == np.float32
    assert reconstructed.dtype == np.float32
    assert means.dtype == np.float32
    assert centered.dtype == np.float32
    assert covariance.dtype == np.float32
    assert correlation.dtype == np.float32
    np.testing.assert_allclose(regression.coefficients[0], 0.75, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(regression.coefficients[1:], weights, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(transformed, pca.scores, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(reconstructed, x, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(means, x.mean(axis=0), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(
        pynabled.column_means(centered), np.zeros(3, dtype=np.float32), atol=1e-5
    )
    np.testing.assert_allclose(covariance, covariance.T, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(correlation, correlation.T, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.diag(correlation), np.ones(3, dtype=np.float32), atol=1e-5)
    assert 0.99 <= regression.r_squared <= 1.0
