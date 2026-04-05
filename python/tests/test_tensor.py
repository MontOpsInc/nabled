"""Tests for tensor operation bindings."""

import numpy as np
import pynabled
import pytest


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
    result = pynabled.tensor_hosvd3(cube, 2, 2, 2)
    assert result.core.shape == (2, 2, 2)
    assert result.u0.shape == (3, 2)
    assert result.u1.shape == (4, 2)
    assert result.u2.shape == (5, 2)


def test_tensor_hosvd3_reconstruct():
    cube = np.random.randn(3, 4, 5).astype(np.float64)
    result = pynabled.tensor_hosvd3(cube, 3, 4, 5)
    recon = pynabled.tensor_hosvd3_reconstruct(result)
    np.testing.assert_allclose(recon, cube, rtol=1e-10)


def _relative_error(lhs, rhs):
    lhs_norm = np.linalg.norm(lhs - rhs)
    rhs_norm = np.linalg.norm(rhs)
    return lhs_norm / max(rhs_norm, 1e-12)


def _cp_als3_reference():
    weights = np.array([1.5, 0.8], dtype=np.float64)
    factor_0 = np.array([[1.0, 0.2], [0.7, 1.1], [0.3, 0.9], [1.2, 0.4]], dtype=np.float64)
    factor_1 = np.array([[0.5, 1.0], [1.3, 0.4], [0.8, 1.2]], dtype=np.float64)
    factor_2 = np.array([[1.0, 0.6], [0.9, 1.4]], dtype=np.float64)
    return weights, factor_0, factor_1, factor_2


def _cp_als_nd_reference():
    weights = np.array([1.5, 0.8], dtype=np.float64)
    factors = [
        np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64),
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]], dtype=np.float64),
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
    ]
    return weights, factors


def _tt_reference():
    return [
        np.array(
            [[[1.0, 0.2], [0.6, -0.1], [0.4, 0.8]]],
            dtype=np.float64,
        ),
        np.array(
            [
                [[0.9, -0.1], [0.3, 0.7]],
                [[-0.2, 0.5], [1.1, 0.4]],
            ],
            dtype=np.float64,
        ),
        np.array(
            [
                [[1.0], [0.3], [-0.2], [0.5]],
                [[0.6], [0.8], [0.1], [1.2]],
            ],
            dtype=np.float64,
        ),
    ]


def test_tensor_complex_kernels_and_einsum():
    cube = (np.random.randn(2, 3, 4) + 1j * np.random.randn(2, 3, 4)).astype(np.complex128)
    vectors = (np.random.randn(2, 4) + 1j * np.random.randn(2, 4)).astype(np.complex128)
    matvec = pynabled.tensor_cube_matvec_complex(cube, vectors)
    for i in range(2):
        np.testing.assert_allclose(matvec[i], cube[i] @ vectors[i], rtol=1e-10, atol=1e-10)

    cube_r = (np.random.randn(2, 3, 4) + 1j * np.random.randn(2, 3, 4)).astype(np.complex128)
    cube_c = (np.random.randn(2, 4, 5) + 1j * np.random.randn(2, 4, 5)).astype(np.complex128)
    matmat = pynabled.tensor_cube_matmat_complex(cube_r, cube_c)
    for i in range(2):
        np.testing.assert_allclose(matmat[i], cube_r[i] @ cube_c[i], rtol=1e-10, atol=1e-10)

    tensor = (np.random.randn(2, 3, 4) + 1j * np.random.randn(2, 3, 4)).astype(np.complex128)
    np.testing.assert_allclose(
        pynabled.tensor_sum_last_axis_complex(tensor),
        tensor.sum(axis=-1),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        pynabled.tensor_l2_norm_last_axis_complex(tensor),
        np.linalg.norm(tensor, axis=-1),
        rtol=1e-10,
        atol=1e-10,
    )
    normalized = pynabled.tensor_normalize_last_axis_complex(tensor)
    np.testing.assert_allclose(
        np.linalg.norm(normalized, axis=-1),
        np.ones(tensor.shape[:-1]),
        rtol=1e-10,
        atol=1e-10,
    )

    left = (np.random.randn(2, 3, 4) + 1j * np.random.randn(2, 3, 4)).astype(np.complex128)
    right = (np.random.randn(2, 3, 4) + 1j * np.random.randn(2, 3, 4)).astype(np.complex128)
    np.testing.assert_allclose(
        pynabled.tensor_batched_dot_last_axis_complex(left, right),
        np.sum(np.conjugate(left) * right, axis=-1),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        pynabled.tensor_permute_axes_complex(tensor, [2, 1, 0]),
        np.transpose(tensor, (2, 1, 0)),
        rtol=1e-10,
        atol=1e-10,
    )

    contract_left = (np.random.randn(2, 3, 4) + 1j * np.random.randn(2, 3, 4)).astype(np.complex128)
    contract_right = (np.random.randn(4, 5) + 1j * np.random.randn(4, 5)).astype(np.complex128)
    np.testing.assert_allclose(
        pynabled.tensor_contract_axes_complex(contract_left, contract_right, [2], [0]),
        np.einsum("ijk,kl->ijl", contract_left, contract_right),
        rtol=1e-10,
        atol=1e-10,
    )

    batched_left = (np.random.randn(2, 3, 4) + 1j * np.random.randn(2, 3, 4)).astype(np.complex128)
    batched_right = (np.random.randn(2, 4, 5) + 1j * np.random.randn(2, 4, 5)).astype(np.complex128)
    np.testing.assert_allclose(
        pynabled.tensor_batched_matmul_last_two_complex(batched_left, batched_right),
        np.matmul(batched_left, batched_right),
        rtol=1e-10,
        atol=1e-10,
    )

    np.testing.assert_allclose(
        pynabled.tensor_einsum(
            "bij,bjk->bik",
            np.asfortranarray(batched_left.real),
            np.asfortranarray(batched_right.real),
        ),
        np.einsum("bij,bjk->bik", batched_left.real, batched_right.real),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        pynabled.tensor_einsum_complex("bij,bjk->bik", batched_left, batched_right),
        np.einsum("bij,bjk->bik", batched_left, batched_right),
        rtol=1e-10,
        atol=1e-10,
    )


def test_tensor_decompositions_accept_fortran_order_inputs():
    cube = np.asfortranarray(np.random.randn(3, 4, 5).astype(np.float64))
    assert cube.flags["F_CONTIGUOUS"]
    assert not cube.flags["C_CONTIGUOUS"]

    hosvd = pynabled.tensor_hosvd_nd(cube, [3, 4, 5])
    reconstructed = pynabled.tensor_hosvd_nd_reconstruct(hosvd)
    np.testing.assert_allclose(reconstructed, cube, rtol=1e-10, atol=1e-10)

    cp = pynabled.tensor_cp_als3(cube, 2, 100, 1e-8)
    reconstructed_cp = pynabled.tensor_cp_als3_reconstruct(cp)
    assert reconstructed_cp.shape == cube.shape

    tt = pynabled.tensor_tt_svd(cube, None, 1e-10)
    reconstructed_tt = pynabled.tensor_tt_svd_reconstruct(tt)
    np.testing.assert_allclose(reconstructed_tt, cube, rtol=1e-10, atol=1e-10)


def test_tensor_hosvd_nd_and_tucker_helpers():
    core = np.array(
        [
            [
                [[1.0, 0.4], [-0.2, 0.7]],
                [[0.5, -0.3], [0.6, 0.2]],
            ],
            [
                [[-0.1, 0.8], [0.9, -0.4]],
                [[0.3, 0.1], [0.2, 0.5]],
            ],
        ],
        dtype=np.float64,
    )
    factors = [
        np.array([[1.0, 0.2], [0.4, 1.1], [0.7, -0.1]], dtype=np.float64),
        np.array([[0.8, 0.3], [0.2, 1.0]], dtype=np.float64),
        np.array([[1.0, 0.0], [0.6, 0.7], [0.2, 1.1], [0.5, -0.3]], dtype=np.float64),
        np.array([[1.0, -0.2], [0.4, 0.9]], dtype=np.float64),
    ]
    reference = pynabled.HosvdNdResult(core=core, factors=factors)
    tensor = pynabled.tensor_tucker_expand(reference)

    estimated = pynabled.tensor_hosvd_nd(tensor, [2, 2, 2, 2])
    projected = pynabled.tensor_tucker_project(tensor, estimated)
    reconstructed = pynabled.tensor_hosvd_nd_reconstruct(estimated)
    expanded = pynabled.tensor_tucker_expand(
        pynabled.HosvdNdResult(core=projected, factors=estimated.factors)
    )

    np.testing.assert_allclose(projected, estimated.core, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(reconstructed, tensor, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(expanded, tensor, rtol=1e-6, atol=1e-6)


def test_tensor_hooi_nd_reconstructs_full_rank_tensor():
    tensor = np.random.randn(2, 3, 2, 2).astype(np.float64)
    result = pynabled.tensor_hooi_nd(tensor, list(tensor.shape), 20, 1e-10)
    reconstructed = pynabled.tensor_hosvd_nd_reconstruct(result)
    np.testing.assert_allclose(reconstructed, tensor, rtol=1e-8, atol=1e-8)


def test_tensor_cp_als3_with_report():
    weights, factor_0, factor_1, factor_2 = _cp_als3_reference()
    reference = pynabled.CpAls3Result(
        weights=weights,
        factor_0=factor_0,
        factor_1=factor_1,
        factor_2=factor_2,
    )
    tensor = pynabled.tensor_cp_als3_reconstruct(reference)

    estimate, report = pynabled.tensor_cp_als3_with_report(tensor, 2, 300, 1e-10)
    reconstructed = pynabled.tensor_cp_als3_reconstruct(estimate)
    metrics = pynabled.tensor_cp_als3_diagnostics(tensor, estimate)

    assert _relative_error(reconstructed, tensor) < 1e-6
    assert 1 <= report.convergence.iterations_run <= 300
    np.testing.assert_allclose(
        [
            report.metrics.signal_norm,
            report.metrics.residual_norm,
            report.metrics.relative_error,
            report.metrics.fit,
        ],
        [
            metrics.signal_norm,
            metrics.residual_norm,
            metrics.relative_error,
            metrics.fit,
        ],
        rtol=1e-10,
        atol=1e-10,
    )
    assert report.metrics.relative_error < 1e-6
    assert report.metrics.fit > 0.999999


def test_tensor_cp_als_nd_with_report():
    weights, factors = _cp_als_nd_reference()
    reference = pynabled.CpAlsNdResult(weights=weights, factors=factors, shape=(3, 2, 4, 2))
    tensor = pynabled.tensor_cp_als_nd_reconstruct(reference)

    estimate, report = pynabled.tensor_cp_als_nd_with_report(tensor, 2, 400, 1e-6)
    reconstructed = pynabled.tensor_cp_als_nd_reconstruct(estimate)
    metrics = pynabled.tensor_cp_als_nd_diagnostics(tensor, estimate)

    assert _relative_error(reconstructed, tensor) < 1e-5
    assert 1 <= report.convergence.iterations_run <= 400
    np.testing.assert_allclose(
        [
            report.metrics.signal_norm,
            report.metrics.residual_norm,
            report.metrics.relative_error,
            report.metrics.fit,
        ],
        [
            metrics.signal_norm,
            metrics.residual_norm,
            metrics.relative_error,
            metrics.fit,
        ],
        rtol=1e-8,
        atol=1e-8,
    )
    assert report.metrics.relative_error < 1e-5
    assert report.metrics.fit > 0.99999


def test_tensor_tt_family():
    reference = pynabled.TensorTrainResult(cores=_tt_reference())
    tensor = pynabled.tensor_tt_svd_reconstruct(reference)

    estimated = pynabled.tensor_tt_svd(tensor, 2, 1e-10)
    reconstructed = pynabled.tensor_tt_svd_reconstruct(estimated)
    assert _relative_error(reconstructed, tensor) < 1e-6

    left_orth = pynabled.tensor_tt_orthogonalize_left(reference)
    right_orth = pynabled.tensor_tt_orthogonalize_right(reference)
    rounded = pynabled.tensor_tt_round(reference, 2, 1e-8)
    np.testing.assert_allclose(
        pynabled.tensor_tt_svd_reconstruct(left_orth),
        tensor,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        pynabled.tensor_tt_svd_reconstruct(right_orth),
        tensor,
        rtol=1e-10,
        atol=1e-10,
    )
    assert all(core.shape[2] <= 2 for core in rounded.cores[:-1])
    assert all(core.shape[0] <= 2 for core in rounded.cores[1:])

    left_tensor = np.array(
        [
            [[1.0, -0.5], [0.7, 0.2], [-0.1, 0.9]],
            [[0.3, 0.8], [-0.6, 1.2], [0.4, -0.3]],
        ],
        dtype=np.float64,
    )
    right_tensor = np.array(
        [
            [[0.6, 0.2], [-0.4, 1.1], [0.5, -0.7]],
            [[0.9, 0.3], [0.8, -0.2], [0.4, 1.0]],
        ],
        dtype=np.float64,
    )
    left_tt = pynabled.tensor_tt_svd(left_tensor, 4, 1e-12)
    right_tt = pynabled.tensor_tt_svd(right_tensor, 4, 1e-12)

    observed_inner = pynabled.tensor_tt_inner(left_tt, right_tt)
    observed_norm = pynabled.tensor_tt_norm(left_tt)
    expected_inner = np.sum(left_tensor * right_tensor)
    expected_norm = np.linalg.norm(left_tensor)
    assert abs(observed_inner - expected_inner) < 1e-10
    assert abs(observed_norm - expected_norm) < 1e-10

    added = pynabled.tensor_tt_add(left_tt, right_tt)
    hadamard = pynabled.tensor_tt_hadamard(left_tt, right_tt)
    rounded_hadamard = pynabled.tensor_tt_hadamard_round(left_tt, right_tt, 2, 1e-8)
    np.testing.assert_allclose(
        pynabled.tensor_tt_svd_reconstruct(added),
        left_tensor + right_tensor,
        rtol=5e-4,
        atol=5e-4,
    )
    np.testing.assert_allclose(
        pynabled.tensor_tt_svd_reconstruct(hadamard),
        left_tensor * right_tensor,
        rtol=5e-4,
        atol=5e-4,
    )
    np.testing.assert_allclose(
        pynabled.tensor_tt_svd_reconstruct(rounded_hadamard),
        left_tensor * right_tensor,
        rtol=5e-3,
        atol=5e-3,
    )
