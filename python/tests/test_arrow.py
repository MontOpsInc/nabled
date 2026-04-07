"""Tests for the PyArrow bridge (requires pynabled built with --features arrow)."""

import numpy as np
import pytest

try:
    import pyarrow as pa
except ImportError:
    pa = None

try:
    from pynabled.arrow import (
        arrow_batched_cosine_distance,
        arrow_batched_cosine_similarity,
        arrow_batched_dot,
        arrow_batched_l2_norm,
        arrow_batched_matmat,
        arrow_batched_matmat_broadcast_left,
        arrow_batched_matmat_broadcast_right,
        arrow_batched_normalize,
        arrow_batched_row_matvec,
        arrow_center_columns,
        arrow_column_means,
        arrow_correlation_matrix,
        arrow_cosine_distance,
        arrow_cosine_similarity,
        arrow_covariance_matrix,
        arrow_dot,
        arrow_l2_norm,
        arrow_matmat,
        arrow_matvec,
        arrow_pairwise_cosine_distance,
        arrow_pairwise_cosine_similarity,
        arrow_pairwise_l2_distance,
        arrow_svd_decompose,
    )
except ImportError:
    arrow_dot = None


pytestmark = [
    pytest.mark.skipif(pa is None, reason="pyarrow not installed"),
    pytest.mark.skipif(arrow_dot is None, reason="pynabled built without arrow feature"),
]


def _matrix_array(values, dtype):
    np_values = np.asarray(values, dtype=dtype)
    arrow_type = pa.float32() if np_values.dtype == np.float32 else pa.float64()
    return pa.array(np_values.tolist(), type=pa.list_(arrow_type, np_values.shape[1]))


def test_arrow_vector_scalar_kernels():
    a = pa.array([1.0, 2.0, 3.0], type=pa.float64())
    b = pa.array([4.0, 5.0, 6.0], type=pa.float64())

    assert abs(arrow_dot(a, b) - 32.0) < 1e-10
    assert abs(arrow_l2_norm(pa.array([3.0, 4.0], type=pa.float64())) - 5.0) < 1e-10
    assert abs(arrow_cosine_similarity(a, b) - 0.974631846) < 1e-8
    assert abs(arrow_cosine_distance(a, b) - (1.0 - 0.974631846)) < 1e-8

    a32 = pa.array([1.0, 2.0, 3.0], type=pa.float32())
    b32 = pa.array([4.0, 5.0, 6.0], type=pa.float32())
    assert abs(arrow_dot(a32, b32) - 32.0) < 1e-5
    assert abs(arrow_cosine_similarity(a32, b32) - 0.974631846) < 1e-5
    assert abs(arrow_cosine_distance(a32, b32) - (1.0 - 0.974631846)) < 1e-5


def test_arrow_vector_pairwise_and_batched_kernels():
    left = _matrix_array([[1.0, 0.0], [1.0, 1.0]], np.float64)
    right = _matrix_array([[0.0, 1.0], [1.0, -1.0]], np.float64)

    pairwise_l2 = arrow_pairwise_l2_distance(left, right)
    np.testing.assert_allclose(
        np.array(pairwise_l2.to_pylist(), dtype=np.float64),
        np.array([[np.sqrt(2.0), 1.0], [1.0, 2.0]], dtype=np.float64),
        rtol=1e-10,
    )

    pairwise_cos = arrow_pairwise_cosine_similarity(left, right)
    np.testing.assert_allclose(
        np.array(pairwise_cos.to_pylist(), dtype=np.float64),
        np.array([[0.0, 1 / np.sqrt(2.0)], [1 / np.sqrt(2.0), 0.0]], dtype=np.float64),
        rtol=1e-10,
    )

    pairwise_cos_dist = arrow_pairwise_cosine_distance(left, right)
    np.testing.assert_allclose(
        np.array(pairwise_cos_dist.to_pylist(), dtype=np.float64),
        1.0 - np.array([[0.0, 1 / np.sqrt(2.0)], [1 / np.sqrt(2.0), 0.0]], dtype=np.float64),
        rtol=1e-10,
    )

    batched_dot = arrow_batched_dot(left, right)
    assert batched_dot.type == pa.float64()
    np.testing.assert_allclose(np.array(batched_dot.to_pylist(), dtype=np.float64), [0.0, 0.0])

    batched_norm = arrow_batched_l2_norm(left)
    assert batched_norm.type == pa.float64()
    np.testing.assert_allclose(
        np.array(batched_norm.to_pylist(), dtype=np.float64),
        [1.0, np.sqrt(2.0)],
        rtol=1e-10,
    )

    batched_cos = arrow_batched_cosine_similarity(left, right)
    np.testing.assert_allclose(
        np.array(batched_cos.to_pylist(), dtype=np.float64),
        [0.0, 0.0],
        rtol=1e-10,
    )

    batched_cos_dist = arrow_batched_cosine_distance(left, right)
    np.testing.assert_allclose(
        np.array(batched_cos_dist.to_pylist(), dtype=np.float64),
        [1.0, 1.0],
        rtol=1e-10,
    )

    normalized = arrow_batched_normalize(left)
    assert normalized.type.value_type == pa.float64()
    np.testing.assert_allclose(
        np.array(normalized.to_pylist(), dtype=np.float64),
        np.array([[1.0, 0.0], [1.0, 1.0]]) / np.array([[1.0], [np.sqrt(2.0)]]),
        rtol=1e-10,
    )


def test_arrow_matrix_kernels():
    matrix = _matrix_array([[1.0, 2.0], [3.0, 4.0]], np.float64)
    vector = pa.array([5.0, 6.0], type=pa.float64())
    matvec = arrow_matvec(matrix, vector)
    np.testing.assert_allclose(np.array(matvec.to_pylist(), dtype=np.float64), [17.0, 39.0])

    left = _matrix_array([[1.0, 2.0], [3.0, 4.0]], np.float32)
    right = _matrix_array([[5.0, 6.0], [7.0, 8.0]], np.float32)
    matmat = arrow_matmat(left, right)
    assert matmat.type.value_type == pa.float32()
    np.testing.assert_allclose(
        np.array(matmat.to_pylist(), dtype=np.float32),
        np.array([[19.0, 22.0], [43.0, 50.0]], dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )

    batch_vectors = _matrix_array([[1.0, 0.0], [0.0, 1.0]], np.float64)
    batched = arrow_batched_row_matvec(batch_vectors, matrix)
    np.testing.assert_allclose(
        np.array(batched.to_pylist(), dtype=np.float64),
        np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float64),
        rtol=1e-10,
    )


def test_arrow_batched_matrix_kernels_preserve_fixed_shape_tensor_contract():
    left = pa.FixedShapeTensorArray.from_numpy_ndarray(
        np.array(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[2.0, 0.0], [0.0, 2.0]],
            ],
            dtype=np.float64,
        )
    )
    right = pa.FixedShapeTensorArray.from_numpy_ndarray(
        np.array(
            [
                [[5.0, 6.0], [7.0, 8.0]],
                [[1.0, 1.0], [1.0, 1.0]],
            ],
            dtype=np.float64,
        )
    )

    result = arrow_batched_matmat(left, right)
    assert isinstance(result, pa.ExtensionArray)
    np.testing.assert_allclose(
        result.to_numpy_ndarray(),
        np.array(
            [
                [[19.0, 22.0], [43.0, 50.0]],
                [[2.0, 2.0], [2.0, 2.0]],
            ],
            dtype=np.float64,
        ),
        rtol=1e-10,
    )

    broadcast_right = arrow_batched_matmat_broadcast_right(
        left,
        _matrix_array([[1.0, 0.0], [0.0, 1.0]], np.float64),
    )
    np.testing.assert_allclose(
        broadcast_right.to_numpy_ndarray(), left.to_numpy_ndarray(), rtol=1e-10
    )

    broadcast_left = arrow_batched_matmat_broadcast_left(
        _matrix_array([[1.0, 0.0], [0.0, 1.0]], np.float64),
        right,
    )
    np.testing.assert_allclose(
        broadcast_left.to_numpy_ndarray(), right.to_numpy_ndarray(), rtol=1e-10
    )


def test_arrow_stats_kernels():
    matrix = _matrix_array([[1.0, 2.0], [3.0, 4.0], [5.0, 8.0]], np.float32)

    means = arrow_column_means(matrix)
    assert means.type == pa.float32()
    np.testing.assert_allclose(
        np.array(means.to_pylist(), dtype=np.float32),
        np.array([3.0, 14.0 / 3.0], dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )

    centered = arrow_center_columns(matrix)
    assert centered.type.value_type == pa.float32()
    np.testing.assert_allclose(
        np.array(centered.to_pylist(), dtype=np.float32).mean(axis=0),
        np.zeros(2, dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )

    covariance = arrow_covariance_matrix(matrix)
    np.testing.assert_allclose(
        np.array(covariance.to_pylist(), dtype=np.float32),
        np.cov(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 8.0]], dtype=np.float32), rowvar=False),
        rtol=1e-5,
        atol=1e-5,
    )

    correlation = arrow_correlation_matrix(matrix)
    np.testing.assert_allclose(
        np.array(correlation.to_pylist(), dtype=np.float32),
        np.corrcoef(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 8.0]], dtype=np.float32), rowvar=False),
        rtol=1e-5,
        atol=1e-5,
    )


def test_arrow_svd_decompose():
    data = _matrix_array([[1.0, 2.0], [3.0, 4.0]], np.float64)
    result = arrow_svd_decompose(data)
    u = np.asarray(result.u)
    s = np.asarray(result.singular_values)
    vt = np.asarray(result.vt)
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    recon = u @ np.diag(s) @ vt
    np.testing.assert_allclose(recon, a, rtol=1e-10)

    data32 = _matrix_array([[1.0, 2.0], [3.0, 4.0]], np.float32)
    result32 = arrow_svd_decompose(data32)
    u32 = np.asarray(result32.u)
    s32 = np.asarray(result32.singular_values)
    vt32 = np.asarray(result32.vt)
    assert u32.dtype == np.float32
    assert s32.dtype == np.float32
    assert vt32.dtype == np.float32
    recon32 = u32 @ np.diag(s32) @ vt32
    np.testing.assert_allclose(recon32, a.astype(np.float32), rtol=5e-5, atol=5e-5)
