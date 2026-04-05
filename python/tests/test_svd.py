"""Tests for SVD bindings."""

import numpy as np
import pynabled


def test_svd_decompose():
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64, order="C")
    result = pynabled.svd_decompose(a)
    assert result.u.shape == (2, 2)
    assert result.singular_values.shape == (2,)
    assert result.vt.shape == (2, 2)
    recon = result.u @ np.diag(result.singular_values) @ result.vt
    np.testing.assert_allclose(recon, a, rtol=1e-10)


def test_svd_decompose_truncated():
    a = np.random.randn(5, 3).astype(np.float64)
    result = pynabled.svd_decompose_truncated(a, 2)
    assert result.u.shape == (5, 2)
    assert result.singular_values.shape == (2,)
    assert result.vt.shape == (2, 3)


def test_svd_pseudo_inverse():
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    pinv = pynabled.svd_pseudo_inverse(a)
    np.testing.assert_allclose(a @ pinv @ a, a, rtol=1e-10)


def test_svd_rank():
    a = np.eye(3, dtype=np.float64)
    result = pynabled.svd_decompose(a)
    r = pynabled.svd_rank(result)
    assert r == 3


def test_svd_reconstruct_matrix():
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    result = pynabled.svd_decompose(a)
    recon = pynabled.svd_reconstruct_matrix(result)
    np.testing.assert_allclose(recon, a, rtol=1e-10)


def test_svd_condition_number():
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    result = pynabled.svd_decompose(a)
    kappa = pynabled.svd_condition_number(result)
    assert np.isfinite(kappa)
    assert kappa > 0


def test_svd_null_space():
    # Rank-deficient: rows are multiples
    a = np.array([[1.0, 2.0], [2.0, 4.0]], dtype=np.float64)
    null = pynabled.svd_null_space(a, None)
    assert null.ndim == 2
    # Each column should be in null space: A @ null_col ≈ 0
    for j in range(null.shape[1]):
        np.testing.assert_allclose(a @ null[:, j], np.zeros(2), atol=1e-10)


def test_svd_accepts_non_contiguous_inputs():
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    a_non_contig = a.T
    assert not a_non_contig.flags["C_CONTIGUOUS"]
    result = pynabled.svd_decompose(a_non_contig)
    recon = result.u @ np.diag(result.singular_values) @ result.vt
    np.testing.assert_allclose(recon, a_non_contig, rtol=1e-10)


def test_svd_accepts_float32():
    a = np.array([[1.0, 2.0], [3.0, 5.0]], dtype=np.float32)
    rank_deficient = np.array([[1.0, 2.0], [2.0, 4.0]], dtype=np.float32)

    result = pynabled.svd_decompose(a)
    truncated = pynabled.svd_decompose_truncated(a, 1)
    pinv = pynabled.svd_pseudo_inverse(a)
    recon = pynabled.svd_reconstruct_matrix(result)
    kappa = pynabled.svd_condition_number(result)
    rank = pynabled.svd_rank(result)
    null = pynabled.svd_null_space(rank_deficient)

    assert result.u.dtype == np.float32
    assert result.singular_values.dtype == np.float32
    assert result.vt.dtype == np.float32
    assert truncated.u.dtype == np.float32
    assert truncated.singular_values.dtype == np.float32
    assert truncated.vt.dtype == np.float32
    assert pinv.dtype == np.float32
    assert recon.dtype == np.float32
    assert null.dtype == np.float32
    assert np.isfinite(kappa)
    assert rank == 2
    np.testing.assert_allclose(recon, a, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(a @ pinv @ a, a, rtol=1e-4, atol=1e-5)
    for j in range(null.shape[1]):
        np.testing.assert_allclose(
            rank_deficient @ null[:, j], np.zeros(2, dtype=np.float32), atol=1e-5
        )
