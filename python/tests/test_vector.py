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


def test_vector_primitives_accept_complex128():
    a = np.array([1.0 + 1.0j, 2.0 - 1.0j], dtype=np.complex128)
    b = np.array([0.5 - 0.5j, -1.0 + 3.0j], dtype=np.complex128)

    dot = pynabled.dot(a, b)
    norm = pynabled.l2_norm(a)
    cosine = pynabled.cosine_similarity(a, b)

    np.testing.assert_allclose(dot, np.vdot(a, b), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(norm, np.linalg.norm(a), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        cosine,
        np.vdot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)),
        rtol=1e-12,
        atol=1e-12,
    )


def test_vector_distance_and_batched_primitives_accept_float32():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    left = np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    right = np.array([[0.0, 1.0], [1.0, 1.0]], dtype=np.float32)

    cosine_distance = pynabled.cosine_distance(a, b)
    pairwise_distance = pynabled.pairwise_cosine_distance(left, right)
    dot = pynabled.batched_dot(left, right)
    norms = pynabled.batched_l2_norm(left)
    cosine = pynabled.batched_cosine_similarity(left, right)
    batched_distance = pynabled.batched_cosine_distance(left, right)
    normalized = pynabled.batched_normalize(left)

    expected_cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    expected_pairwise_similarity = (left @ right.T) / (
        np.linalg.norm(left, axis=1)[:, None] * np.linalg.norm(right, axis=1)[None, :]
    )

    assert dot.dtype == np.float32
    assert norms.dtype == np.float32
    assert cosine.dtype == np.float32
    assert batched_distance.dtype == np.float32
    assert normalized.dtype == np.float32
    np.testing.assert_allclose(cosine_distance, 1.0 - expected_cosine, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        pairwise_distance,
        1.0 - expected_pairwise_similarity,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(dot, np.sum(left * right, axis=1), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(norms, np.linalg.norm(left, axis=1), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(cosine, np.diag(expected_pairwise_similarity), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        batched_distance,
        1.0 - np.diag(expected_pairwise_similarity),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), np.ones(2), rtol=1e-5, atol=1e-6)


def test_batched_vector_primitives_accept_complex128():
    left = np.array(
        [[1.0 + 1.0j, 0.0 + 2.0j], [2.0 + 0.0j, 0.0 + 2.0j]],
        dtype=np.complex128,
    )
    right = np.array(
        [[1.0 - 1.0j, 2.0 + 0.0j], [0.0 + 2.0j, 2.0 + 0.0j]],
        dtype=np.complex128,
    )

    dot = pynabled.batched_dot(left, right)
    norms = pynabled.batched_l2_norm(left)
    cosine = pynabled.batched_cosine_similarity(left, right)
    normalized = pynabled.batched_normalize(left)

    expected_dot = np.sum(np.conj(left) * right, axis=1)
    expected_left_norms = np.linalg.norm(left, axis=1)
    expected_right_norms = np.linalg.norm(right, axis=1)
    expected_cosine = expected_dot / (expected_left_norms * expected_right_norms)

    assert dot.dtype == np.complex128
    assert norms.dtype == np.float64
    assert cosine.dtype == np.complex128
    assert normalized.dtype == np.complex128
    np.testing.assert_allclose(dot, expected_dot, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(norms, expected_left_norms, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(cosine, expected_cosine, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        np.linalg.norm(normalized, axis=1), np.ones(2), rtol=1e-12, atol=1e-12
    )


def test_vector_out_reuses_pairwise_and_batched_output_buffers():
    left = np.asfortranarray(np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32))
    right = np.asfortranarray(np.array([[0.0, 1.0], [1.0, 1.0]], dtype=np.float32))

    pairwise_out = np.empty((2, 2), dtype=np.float32, order="F")
    returned_pairwise = pynabled.pairwise_l2_distance(left, right, out=pairwise_out)
    assert returned_pairwise is pairwise_out
    np.testing.assert_allclose(pairwise_out, np.linalg.norm(left[:, None, :] - right[None, :, :], axis=2))

    cosine_out = np.empty((2, 2), dtype=np.float32, order="F")
    returned_cosine = pynabled.pairwise_cosine_similarity(left, right, out=cosine_out)
    assert returned_cosine is cosine_out
    expected_cosine = (left @ right.T) / (
        np.linalg.norm(left, axis=1)[:, None] * np.linalg.norm(right, axis=1)[None, :]
    )
    np.testing.assert_allclose(cosine_out, expected_cosine, rtol=1e-6, atol=1e-6)

    dot_out = np.empty(2, dtype=np.float32)
    returned_dot = pynabled.batched_dot(left, right, out=dot_out)
    assert returned_dot is dot_out
    np.testing.assert_allclose(dot_out, np.sum(left * right, axis=1), rtol=1e-6, atol=1e-6)

    normalize_out = np.empty_like(left, order="F")
    returned_normalized = pynabled.batched_normalize(left, out=normalize_out)
    assert returned_normalized is normalize_out
    np.testing.assert_allclose(
        np.linalg.norm(normalize_out, axis=1),
        np.ones(left.shape[0], dtype=np.float32),
        rtol=1e-5,
        atol=1e-6,
    )


def test_vector_out_reuses_complex_buffers_and_rejects_aliasing():
    left = np.array(
        [[1.0 + 1.0j, 0.0 + 2.0j], [2.0 + 0.0j, 0.0 + 2.0j]],
        dtype=np.complex128,
    )
    right = np.array(
        [[1.0 - 1.0j, 2.0 + 0.0j], [0.0 + 2.0j, 2.0 + 0.0j]],
        dtype=np.complex128,
    )

    dot_out = np.empty(2, dtype=np.complex128)
    returned_dot = pynabled.batched_dot(left, right, out=dot_out)
    assert returned_dot is dot_out
    np.testing.assert_allclose(dot_out, np.sum(np.conj(left) * right, axis=1), rtol=1e-12, atol=1e-12)

    norm_out = np.empty(2, dtype=np.float64)
    returned_norm = pynabled.batched_l2_norm(left, out=norm_out)
    assert returned_norm is norm_out
    np.testing.assert_allclose(norm_out, np.linalg.norm(left, axis=1), rtol=1e-12, atol=1e-12)

    aliased = np.ones((2, 2), dtype=np.float64)
    with pytest.raises(TypeError, match="already borrowed"):
        pynabled.batched_normalize(aliased, out=aliased)
