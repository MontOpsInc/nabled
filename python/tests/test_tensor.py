"""Tests for tensor operation bindings."""

import numpy as np
import pytest

import pynabled


def test_tensor_cube_matvec():
    cube = np.random.randn(2, 3, 4).astype(np.float64)
    vectors = np.random.randn(2, 4).astype(np.float64)
    out = pynabled.tensor_cube_matvec(cube, vectors)
    assert out.shape == (2, 3)
    for i in range(2):
        np.testing.assert_allclose(out[i], cube[i] @ vectors[i], rtol=1e-10)


def test_tensor_cube_matmat():
    left = np.random.randn(2, 3, 4).astype(np.float64)
    right = np.random.randn(2, 4, 5).astype(np.float64)
    out = pynabled.tensor_cube_matmat(left, right)
    assert out.shape == (2, 3, 5)
    for i in range(2):
        np.testing.assert_allclose(out[i], left[i] @ right[i], rtol=1e-10)


def test_tensor_sum_last_axis():
    t = np.random.randn(2, 3, 4).astype(np.float64)
    out = pynabled.tensor_sum_last_axis(t)
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out, t.sum(axis=-1), rtol=1e-14)


def test_tensor_l2_norm_last_axis():
    t = np.random.randn(2, 3, 4).astype(np.float64)
    out = pynabled.tensor_l2_norm_last_axis(t)
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out, np.linalg.norm(t, axis=-1), rtol=1e-14)


def test_tensor_normalize_last_axis():
    t = np.random.randn(2, 3, 4).astype(np.float64)
    out = pynabled.tensor_normalize_last_axis(t)
    assert out.shape == t.shape
    norms = np.linalg.norm(out, axis=-1)
    np.testing.assert_allclose(norms, np.ones((2, 3)), rtol=1e-14)


def test_tensor_batched_dot_last_axis():
    left = np.random.randn(2, 3, 4).astype(np.float64)
    right = np.random.randn(2, 3, 4).astype(np.float64)
    out = pynabled.tensor_batched_dot_last_axis(left, right)
    assert out.shape == (2, 3)
    expected = (left * right).sum(axis=-1)
    np.testing.assert_allclose(out, expected, rtol=1e-14)


def test_tensor_permute_axes():
    t = np.random.randn(2, 3, 4).astype(np.float64)
    out = pynabled.tensor_permute_axes(t, [2, 1, 0])
    assert out.shape == (4, 3, 2)
    np.testing.assert_allclose(out, np.transpose(t, (2, 1, 0)), rtol=1e-14)


def test_tensor_contract_axes():
    left = np.random.randn(2, 3, 4).astype(np.float64)
    right = np.random.randn(4, 5).astype(np.float64)
    out = pynabled.tensor_contract_axes(left, right, [2], [0])
    assert out.shape == (2, 3, 5)
    expected = np.einsum("ijk,kl->ijl", left, right)
    np.testing.assert_allclose(out, expected, rtol=1e-10)


def test_tensor_batched_matmul_last_two():
    left = np.random.randn(2, 3, 4).astype(np.float64)
    right = np.random.randn(2, 4, 5).astype(np.float64)
    out = pynabled.tensor_batched_matmul_last_two(left, right)
    assert out.shape == (2, 3, 5)
    for i in range(2):
        np.testing.assert_allclose(out[i], left[i] @ right[i], rtol=1e-10)


def test_tensor_hosvd3():
    cube = np.random.randn(3, 4, 5).astype(np.float64)
    core, u0, u1, u2 = pynabled.tensor_hosvd3(cube, 2, 2, 2)
    assert core.shape == (2, 2, 2)
    assert u0.shape == (3, 2)
    assert u1.shape == (4, 2)
    assert u2.shape == (5, 2)


def test_tensor_hosvd3_reconstruct():
    cube = np.random.randn(3, 4, 5).astype(np.float64)
    core, u0, u1, u2 = pynabled.tensor_hosvd3(cube, 3, 4, 5)
    recon = pynabled.tensor_hosvd3_reconstruct(core, u0, u1, u2)
    np.testing.assert_allclose(recon, cube, rtol=1e-10)
