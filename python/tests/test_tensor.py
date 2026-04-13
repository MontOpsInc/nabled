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


def test_tensor_out_reuses_real_output_buffers():
    tensor = np.asfortranarray(np.random.randn(2, 3, 4).astype(np.float32))
    sum_out = np.empty((2, 3), dtype=np.float32, order="F")
    returned_sum = pynabled.tensor_sum_last_axis(tensor, out=sum_out)
    assert returned_sum is sum_out
    np.testing.assert_allclose(sum_out, tensor.sum(axis=-1), rtol=5e-5, atol=5e-5)

    norm_out = np.empty((2, 3), dtype=np.float32, order="F")
    returned_norm = pynabled.tensor_l2_norm_last_axis(tensor, out=norm_out)
    assert returned_norm is norm_out
    np.testing.assert_allclose(norm_out, np.linalg.norm(tensor, axis=-1), rtol=5e-5, atol=5e-5)

    normalized_out = np.empty_like(tensor, order="F")
    returned_normalized = pynabled.tensor_normalize_last_axis(tensor, out=normalized_out)
    assert returned_normalized is normalized_out
    np.testing.assert_allclose(
        np.linalg.norm(normalized_out, axis=-1),
        np.ones(tensor.shape[:-1], dtype=np.float32),
        rtol=5e-5,
        atol=5e-5,
    )


def test_tensor_out_reuses_binary_output_buffers():
    cube = np.random.randn(2, 3, 4).astype(np.float64)
    vectors = np.random.randn(2, 4).astype(np.float64)
    matvec_out = np.empty((2, 3), dtype=np.float64, order="F")
    returned_matvec = pynabled.tensor_cube_matvec(cube, vectors, out=matvec_out)
    assert returned_matvec is matvec_out
    np.testing.assert_allclose(matvec_out, np.einsum("bij,bj->bi", cube, vectors), rtol=1e-10)

    right = np.random.randn(2, 4, 5).astype(np.float64)
    matmul_out = np.empty((2, 3, 5), dtype=np.float64, order="F")
    returned_matmul = pynabled.tensor_batched_matmul_last_two(cube, right, out=matmul_out)
    assert returned_matmul is matmul_out
    np.testing.assert_allclose(matmul_out, np.matmul(cube, right), rtol=1e-10)

    contract_left = np.random.randn(2, 3, 4).astype(np.float64)
    contract_right = np.random.randn(4, 5).astype(np.float64)
    contract_out = np.empty((2, 3, 5), dtype=np.float64, order="F")
    returned_contract = pynabled.tensor_contract_axes(
        contract_left, contract_right, [2], [0], out=contract_out
    )
    assert returned_contract is contract_out
    np.testing.assert_allclose(
        contract_out,
        np.einsum("ijk,kl->ijl", contract_left, contract_right),
        rtol=1e-10,
        atol=1e-10,
    )


def test_tensor_out_reuses_complex_output_buffers_and_rejects_aliasing():
    tensor = (np.random.randn(2, 3, 4) + 1j * np.random.randn(2, 3, 4)).astype(np.complex128)
    complex_sum_out = np.empty((2, 3), dtype=np.complex128, order="F")
    returned_sum = pynabled.tensor_sum_last_axis_complex(tensor, out=complex_sum_out)
    assert returned_sum is complex_sum_out
    np.testing.assert_allclose(complex_sum_out, tensor.sum(axis=-1), rtol=1e-10, atol=1e-10)

    complex_norm_out = np.empty((2, 3), dtype=np.float64, order="F")
    returned_norm = pynabled.tensor_l2_norm_last_axis_complex(tensor, out=complex_norm_out)
    assert returned_norm is complex_norm_out
    np.testing.assert_allclose(
        complex_norm_out, np.linalg.norm(tensor, axis=-1), rtol=1e-10, atol=1e-10
    )

    aliased = np.ones((2, 3, 4), dtype=np.float64)
    with pytest.raises(TypeError, match="already borrowed"):
        pynabled.tensor_normalize_last_axis(aliased, out=aliased)


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


def _borrowed_vector_view(values):
    storage = np.empty(values.shape[0] + 2, dtype=values.dtype)
    storage[1:-1] = values
    view = storage[1:-1]
    assert not view.flags["OWNDATA"]
    return view


def _borrowed_matrix_view(values):
    storage = np.empty((values.shape[0] + 1, values.shape[1] + 2), dtype=values.dtype, order="F")
    storage[1:, 1:-1] = values
    view = storage[1:, 1:-1]
    assert not view.flags["OWNDATA"]
    return view


def _borrowed_tensor_view(values):
    storage = np.empty(
        tuple(extent + 1 for extent in values.shape),
        dtype=values.dtype,
        order="F",
    )
    storage[(slice(1, None),) * values.ndim] = values
    view = storage[(slice(1, None),) * values.ndim]
    assert not view.flags["OWNDATA"]
    return view


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


def test_tensor_reconstruction_helpers_reuse_output_buffers():
    cube = np.arange(60, dtype=np.float64).reshape(3, 4, 5)
    hosvd3 = pynabled.tensor_hosvd3(cube, 3, 4, 5)
    hosvd_nd = pynabled.tensor_hosvd_nd(cube, [3, 4, 5])

    hosvd3_out = np.empty(cube.shape, dtype=np.float64, order="F")
    returned_hosvd3 = pynabled.tensor_hosvd3_reconstruct(hosvd3, out=hosvd3_out)
    assert returned_hosvd3 is hosvd3_out
    np.testing.assert_allclose(hosvd3_out, cube, rtol=1e-10, atol=1e-10)

    hosvd_nd_out = np.empty(cube.shape, dtype=np.float64, order="F")
    returned_hosvd_nd = pynabled.tensor_hosvd_nd_reconstruct(hosvd_nd, out=hosvd_nd_out)
    assert returned_hosvd_nd is hosvd_nd_out
    np.testing.assert_allclose(hosvd_nd_out, cube, rtol=1e-10, atol=1e-10)

    projected_out = np.empty(hosvd_nd.core.shape, dtype=np.float64, order="F")
    returned_projected = pynabled.tensor_tucker_project(cube, hosvd_nd, out=projected_out)
    assert returned_projected is projected_out
    np.testing.assert_allclose(projected_out, hosvd_nd.core, rtol=1e-10, atol=1e-10)

    tucker_out = np.empty(cube.shape, dtype=np.float64, order="F")
    returned_tucker = pynabled.tensor_tucker_expand(hosvd_nd, out=tucker_out)
    assert returned_tucker is tucker_out
    np.testing.assert_allclose(tucker_out, cube, rtol=1e-10, atol=1e-10)

    cp3_weights, cp3_factor_0, cp3_factor_1, cp3_factor_2 = _cp_als3_reference()
    cp3 = pynabled.CpAls3Result(
        weights=cp3_weights,
        factor_0=cp3_factor_0,
        factor_1=cp3_factor_1,
        factor_2=cp3_factor_2,
    )
    expected_cp3 = pynabled.tensor_cp_als3_reconstruct(cp3)
    cp3_out = np.empty(expected_cp3.shape, dtype=np.float64, order="F")
    returned_cp3 = pynabled.tensor_cp_als3_reconstruct(cp3, out=cp3_out)
    assert returned_cp3 is cp3_out
    np.testing.assert_allclose(cp3_out, expected_cp3, rtol=1e-10, atol=1e-10)

    cp_nd_weights, cp_nd_factors = _cp_als_nd_reference()
    cp_nd = pynabled.CpAlsNdResult(
        weights=cp_nd_weights,
        factors=cp_nd_factors,
        shape=tuple(factor.shape[0] for factor in cp_nd_factors),
    )
    expected_cp_nd = pynabled.tensor_cp_als_nd_reconstruct(cp_nd)
    cp_nd_out = np.empty(expected_cp_nd.shape, dtype=np.float64, order="F")
    returned_cp_nd = pynabled.tensor_cp_als_nd_reconstruct(cp_nd, out=cp_nd_out)
    assert returned_cp_nd is cp_nd_out
    np.testing.assert_allclose(cp_nd_out, expected_cp_nd, rtol=1e-10, atol=1e-10)

    tt = pynabled.TensorTrainResult(cores=_tt_reference())
    expected_tt = pynabled.tensor_tt_svd_reconstruct(tt)
    tt_out = np.empty(expected_tt.shape, dtype=np.float64, order="F")
    returned_tt = pynabled.tensor_tt_svd_reconstruct(tt, out=tt_out)
    assert returned_tt is tt_out
    np.testing.assert_allclose(tt_out, expected_tt, rtol=1e-10, atol=1e-10)


def test_tensor_result_helpers_accept_borrowed_factor_views():
    cube = np.arange(60, dtype=np.float64).reshape(3, 4, 5)
    hosvd3 = pynabled.tensor_hosvd3(cube, 3, 4, 5)
    borrowed_hosvd3 = pynabled.Hosvd3Result(
        core=_borrowed_tensor_view(hosvd3.core),
        u0=_borrowed_matrix_view(hosvd3.u0),
        u1=_borrowed_matrix_view(hosvd3.u1),
        u2=_borrowed_matrix_view(hosvd3.u2),
    )
    hosvd3_out = np.empty(cube.shape, dtype=np.float64, order="F")
    returned_hosvd3 = pynabled.tensor_hosvd3_reconstruct(borrowed_hosvd3, out=hosvd3_out)
    assert returned_hosvd3 is hosvd3_out
    np.testing.assert_allclose(hosvd3_out, cube, rtol=1e-10, atol=1e-10)

    hosvd_nd = pynabled.tensor_hosvd_nd(cube, [3, 4, 5])
    borrowed_hosvd_nd = pynabled.HosvdNdResult(
        core=_borrowed_tensor_view(hosvd_nd.core),
        factors=[_borrowed_matrix_view(factor) for factor in hosvd_nd.factors],
    )
    hosvd_nd_out = np.empty(cube.shape, dtype=np.float64, order="F")
    returned_hosvd_nd = pynabled.tensor_hosvd_nd_reconstruct(borrowed_hosvd_nd, out=hosvd_nd_out)
    assert returned_hosvd_nd is hosvd_nd_out
    np.testing.assert_allclose(hosvd_nd_out, cube, rtol=1e-10, atol=1e-10)

    projected_out = np.empty(hosvd_nd.core.shape, dtype=np.float64, order="F")
    returned_projected = pynabled.tensor_tucker_project(cube, borrowed_hosvd_nd, out=projected_out)
    assert returned_projected is projected_out
    np.testing.assert_allclose(projected_out, hosvd_nd.core, rtol=1e-10, atol=1e-10)

    expanded_out = np.empty(cube.shape, dtype=np.float64, order="F")
    returned_expanded = pynabled.tensor_tucker_expand(borrowed_hosvd_nd, out=expanded_out)
    assert returned_expanded is expanded_out
    np.testing.assert_allclose(expanded_out, cube, rtol=1e-10, atol=1e-10)

    cp3_weights, cp3_factor_0, cp3_factor_1, cp3_factor_2 = _cp_als3_reference()
    cp3 = pynabled.CpAls3Result(
        weights=_borrowed_vector_view(cp3_weights),
        factor_0=_borrowed_matrix_view(cp3_factor_0),
        factor_1=_borrowed_matrix_view(cp3_factor_1),
        factor_2=_borrowed_matrix_view(cp3_factor_2),
    )
    cp3_tensor = pynabled.tensor_cp_als3_reconstruct(cp3)
    cp3_metrics = pynabled.tensor_cp_als3_diagnostics(cp3_tensor, cp3)
    assert cp3_metrics.relative_error < 1e-12
    cp3_out = np.empty(cp3_tensor.shape, dtype=np.float64, order="F")
    returned_cp3 = pynabled.tensor_cp_als3_reconstruct(cp3, out=cp3_out)
    assert returned_cp3 is cp3_out
    np.testing.assert_allclose(cp3_out, cp3_tensor, rtol=1e-10, atol=1e-10)

    cp_nd_weights, cp_nd_factors = _cp_als_nd_reference()
    cp_nd = pynabled.CpAlsNdResult(
        weights=_borrowed_vector_view(cp_nd_weights),
        factors=[_borrowed_matrix_view(factor) for factor in cp_nd_factors],
        shape=tuple(factor.shape[0] for factor in cp_nd_factors),
    )
    cp_nd_tensor = pynabled.tensor_cp_als_nd_reconstruct(cp_nd)
    cp_nd_metrics = pynabled.tensor_cp_als_nd_diagnostics(cp_nd_tensor, cp_nd)
    assert cp_nd_metrics.relative_error < 1e-12
    cp_nd_out = np.empty(cp_nd_tensor.shape, dtype=np.float64, order="F")
    returned_cp_nd = pynabled.tensor_cp_als_nd_reconstruct(cp_nd, out=cp_nd_out)
    assert returned_cp_nd is cp_nd_out
    np.testing.assert_allclose(cp_nd_out, cp_nd_tensor, rtol=1e-10, atol=1e-10)

    tt = pynabled.TensorTrainResult(cores=[_borrowed_tensor_view(core) for core in _tt_reference()])
    tt_expected = pynabled.tensor_tt_svd_reconstruct(tt)
    tt_out = np.empty(tt_expected.shape, dtype=np.float64, order="F")
    returned_tt = pynabled.tensor_tt_svd_reconstruct(tt, out=tt_out)
    assert returned_tt is tt_out
    np.testing.assert_allclose(tt_out, tt_expected, rtol=1e-10, atol=1e-10)


def test_tensor_reconstruction_helpers_reject_wrong_output_dtype():
    cube = np.arange(60, dtype=np.float64).reshape(3, 4, 5)
    hosvd3 = pynabled.tensor_hosvd3(cube, 3, 4, 5)
    hosvd_nd = pynabled.tensor_hosvd_nd(cube, [3, 4, 5])
    bad_hosvd3_out = np.empty(cube.shape, dtype=np.float32)
    with pytest.raises(TypeError, match="output must be a writable NumPy array with dtype float64 and rank 3"):
        pynabled.tensor_hosvd3_reconstruct(hosvd3, out=bad_hosvd3_out)

    bad_hosvd_out = np.empty(cube.shape, dtype=np.float32)
    with pytest.raises(TypeError, match="output must be a writable NumPy array with dtype float64"):
        pynabled.tensor_hosvd_nd_reconstruct(hosvd_nd, out=bad_hosvd_out)

    bad_projected_out = np.empty(hosvd_nd.core.shape, dtype=np.float32)
    with pytest.raises(TypeError, match="output must be a writable NumPy array with dtype float64"):
        pynabled.tensor_tucker_project(cube, hosvd_nd, out=bad_projected_out)

    cp3_weights, cp3_factor_0, cp3_factor_1, cp3_factor_2 = _cp_als3_reference()
    cp3 = pynabled.CpAls3Result(
        weights=cp3_weights,
        factor_0=cp3_factor_0,
        factor_1=cp3_factor_1,
        factor_2=cp3_factor_2,
    )
    bad_cp3_out = np.empty((4, 3, 2), dtype=np.float32)
    with pytest.raises(TypeError, match="output must be a writable NumPy array with dtype float64 and rank 3"):
        pynabled.tensor_cp_als3_reconstruct(cp3, out=bad_cp3_out)


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


def test_tensor_einsum_reuses_output_buffers():
    left = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    right = np.arange(40, dtype=np.float64).reshape(2, 4, 5)
    expected = np.einsum("bij,bjk->bik", left, right)

    out = np.empty(expected.shape, dtype=np.float64, order="F")
    returned = pynabled.tensor_einsum("bij,bjk->bik", left, right, out=out)
    assert returned is out
    np.testing.assert_allclose(out, expected, rtol=1e-10, atol=1e-10)

    left_complex = (left + 1j * (left + 1)).astype(np.complex128)
    right_complex = (right - 1j * (right + 2)).astype(np.complex128)
    expected_complex = np.einsum("bij,bjk->bik", left_complex, right_complex)

    out_complex = np.empty(expected_complex.shape, dtype=np.complex128, order="F")
    returned_complex = pynabled.tensor_einsum_complex(
        "bij,bjk->bik",
        left_complex,
        right_complex,
        out=out_complex,
    )
    assert returned_complex is out_complex
    np.testing.assert_allclose(out_complex, expected_complex, rtol=1e-10, atol=1e-10)


def test_tensor_einsum_rejects_wrong_output_dtype():
    left = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    right = np.arange(40, dtype=np.float64).reshape(2, 4, 5)
    bad_out = np.empty((2, 3, 5), dtype=np.float32)
    with pytest.raises(TypeError, match="output must be a writable NumPy array with dtype float64"):
        pynabled.tensor_einsum("bij,bjk->bik", left, right, out=bad_out)

    left_complex = (left + 1j * left).astype(np.complex128)
    right_complex = (right - 1j * right).astype(np.complex128)
    bad_complex_out = np.empty((2, 3, 5), dtype=np.float64)
    with pytest.raises(TypeError, match="output must be a writable NumPy array with dtype complex128"):
        pynabled.tensor_einsum_complex("bij,bjk->bik", left_complex, right_complex, out=bad_complex_out)


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


def test_tensor_float32_kernel_surface_preserves_dtype():
    cube = np.random.randn(2, 3, 4).astype(np.float32)
    vectors = np.random.randn(2, 4).astype(np.float32)
    cube_matvec = pynabled.tensor_cube_matvec(cube, vectors)
    assert cube_matvec.dtype == np.float32
    np.testing.assert_allclose(
        cube_matvec, np.einsum("bij,bj->bi", cube, vectors), rtol=5e-5, atol=5e-5
    )

    right_cube = np.random.randn(2, 4, 5).astype(np.float32)
    cube_matmat = pynabled.tensor_cube_matmat(cube, right_cube)
    assert cube_matmat.dtype == np.float32
    np.testing.assert_allclose(cube_matmat, np.matmul(cube, right_cube), rtol=5e-5, atol=5e-5)

    tensor = np.random.randn(2, 3, 4).astype(np.float32)
    summed = pynabled.tensor_sum_last_axis(tensor)
    assert summed.dtype == np.float32
    np.testing.assert_allclose(summed, tensor.sum(axis=-1), rtol=5e-5, atol=5e-5)

    norms = pynabled.tensor_l2_norm_last_axis(tensor)
    assert norms.dtype == np.float32
    np.testing.assert_allclose(norms, np.linalg.norm(tensor, axis=-1), rtol=5e-5, atol=5e-5)

    normalized = pynabled.tensor_normalize_last_axis(tensor)
    assert normalized.dtype == np.float32
    np.testing.assert_allclose(
        np.linalg.norm(normalized, axis=-1),
        np.ones(tensor.shape[:-1], dtype=np.float32),
        rtol=5e-5,
        atol=5e-5,
    )

    other = np.random.randn(2, 3, 4).astype(np.float32)
    batched_dot = pynabled.tensor_batched_dot_last_axis(tensor, other)
    assert batched_dot.dtype == np.float32
    np.testing.assert_allclose(batched_dot, (tensor * other).sum(axis=-1), rtol=5e-5, atol=5e-5)

    permuted = pynabled.tensor_permute_axes(tensor, [2, 1, 0])
    assert permuted.dtype == np.float32
    np.testing.assert_allclose(permuted, np.transpose(tensor, (2, 1, 0)), rtol=5e-5, atol=5e-5)

    contract_rhs = np.random.randn(4, 5).astype(np.float32)
    contracted = pynabled.tensor_contract_axes(tensor, contract_rhs, [2], [0])
    assert contracted.dtype == np.float32
    np.testing.assert_allclose(
        contracted,
        np.einsum("ijk,kl->ijl", tensor, contract_rhs),
        rtol=5e-5,
        atol=5e-5,
    )

    batched_mm = pynabled.tensor_batched_matmul_last_two(cube, right_cube)
    assert batched_mm.dtype == np.float32
    np.testing.assert_allclose(batched_mm, np.matmul(cube, right_cube), rtol=5e-5, atol=5e-5)

    einsum = pynabled.tensor_einsum("bij,bjk->bik", cube, right_cube)
    assert einsum.dtype == np.float32
    np.testing.assert_allclose(
        einsum, np.einsum("bij,bjk->bik", cube, right_cube), rtol=5e-5, atol=5e-5
    )


def test_tensor_float32_decomposition_surface_preserves_dtype():
    cube = np.random.randn(3, 4, 2).astype(np.float32)
    hosvd3 = pynabled.tensor_hosvd3(cube, 3, 4, 2)
    assert hosvd3.core.dtype == np.float32
    assert hosvd3.u0.dtype == np.float32
    assert hosvd3.u1.dtype == np.float32
    assert hosvd3.u2.dtype == np.float32
    reconstructed_cube = pynabled.tensor_hosvd3_reconstruct(hosvd3)
    assert reconstructed_cube.dtype == np.float32
    np.testing.assert_allclose(reconstructed_cube, cube, rtol=5e-5, atol=5e-5)

    tensor = np.random.randn(2, 3, 2, 2).astype(np.float32)
    hosvd = pynabled.tensor_hosvd_nd(tensor, list(tensor.shape))
    assert hosvd.core.dtype == np.float32
    assert all(factor.dtype == np.float32 for factor in hosvd.factors)
    projected = pynabled.tensor_tucker_project(tensor, hosvd)
    assert projected.dtype == np.float32
    expanded = pynabled.tensor_tucker_expand(
        pynabled.HosvdNdResult(core=projected, factors=hosvd.factors)
    )
    assert expanded.dtype == np.float32
    np.testing.assert_allclose(expanded, tensor, rtol=5e-4, atol=5e-4)

    hooi = pynabled.tensor_hooi_nd(tensor, list(tensor.shape), 20, 1e-5)
    assert hooi.core.dtype == np.float32
    assert all(factor.dtype == np.float32 for factor in hooi.factors)
    reconstructed = pynabled.tensor_hosvd_nd_reconstruct(hooi)
    assert reconstructed.dtype == np.float32
    np.testing.assert_allclose(reconstructed, tensor, rtol=5e-4, atol=5e-4)


def test_tensor_float32_cp_and_tt_surface_preserves_dtype():
    weights, factor_0, factor_1, factor_2 = _cp_als3_reference()
    cp3_reference = pynabled.CpAls3Result(
        weights=weights.astype(np.float32),
        factor_0=factor_0.astype(np.float32),
        factor_1=factor_1.astype(np.float32),
        factor_2=factor_2.astype(np.float32),
    )
    cp3_tensor = pynabled.tensor_cp_als3_reconstruct(cp3_reference)
    estimate3, report3 = pynabled.tensor_cp_als3_with_report(cp3_tensor, 2, 300, 1e-5)
    assert estimate3.weights.dtype == np.float32
    assert estimate3.factor_0.dtype == np.float32
    assert estimate3.factor_1.dtype == np.float32
    assert estimate3.factor_2.dtype == np.float32
    assert isinstance(report3.metrics.relative_error, float)
    reconstructed3 = pynabled.tensor_cp_als3_reconstruct(estimate3)
    assert reconstructed3.dtype == np.float32
    assert _relative_error(reconstructed3, cp3_tensor) < 5e-4

    weights_nd, factors_nd = _cp_als_nd_reference()
    cp_nd_reference = pynabled.CpAlsNdResult(
        weights=weights_nd.astype(np.float32),
        factors=[factor.astype(np.float32) for factor in factors_nd],
        shape=(3, 2, 4, 2),
    )
    cp_nd_tensor = pynabled.tensor_cp_als_nd_reconstruct(cp_nd_reference)
    estimate_nd, report_nd = pynabled.tensor_cp_als_nd_with_report(cp_nd_tensor, 2, 400, 1e-5)
    assert estimate_nd.weights.dtype == np.float32
    assert all(factor.dtype == np.float32 for factor in estimate_nd.factors)
    assert isinstance(report_nd.metrics.fit, float)
    reconstructed_nd = pynabled.tensor_cp_als_nd_reconstruct(estimate_nd)
    assert reconstructed_nd.dtype == np.float32
    assert _relative_error(reconstructed_nd, cp_nd_tensor) < 5e-4

    tt_reference = pynabled.TensorTrainResult(
        cores=[core.astype(np.float32) for core in _tt_reference()]
    )
    tt_tensor = pynabled.tensor_tt_svd_reconstruct(tt_reference)
    assert tt_tensor.dtype == np.float32

    tt_estimated = pynabled.tensor_tt_svd(tt_tensor, 3, 1e-5)
    assert all(core.dtype == np.float32 for core in tt_estimated.cores)
    reconstructed_tt = pynabled.tensor_tt_svd_reconstruct(tt_estimated)
    assert reconstructed_tt.dtype == np.float32
    np.testing.assert_allclose(reconstructed_tt, tt_tensor, rtol=5e-4, atol=5e-4)

    left_orth = pynabled.tensor_tt_orthogonalize_left(tt_reference)
    right_orth = pynabled.tensor_tt_orthogonalize_right(tt_reference)
    rounded = pynabled.tensor_tt_round(tt_reference, 2, 1e-5)
    assert all(core.dtype == np.float32 for core in left_orth.cores)
    assert all(core.dtype == np.float32 for core in right_orth.cores)
    assert all(core.dtype == np.float32 for core in rounded.cores)

    left_tensor = np.random.randn(2, 3, 2).astype(np.float32)
    right_tensor = np.random.randn(2, 3, 2).astype(np.float32)
    left_tt = pynabled.tensor_tt_svd(left_tensor, 4, 1e-5)
    right_tt = pynabled.tensor_tt_svd(right_tensor, 4, 1e-5)

    observed_inner = pynabled.tensor_tt_inner(left_tt, right_tt)
    observed_norm = pynabled.tensor_tt_norm(left_tt)
    assert abs(observed_inner - float(np.sum(left_tensor * right_tensor))) < 5e-4
    assert abs(observed_norm - float(np.linalg.norm(left_tensor))) < 5e-4

    added = pynabled.tensor_tt_add(left_tt, right_tt)
    hadamard = pynabled.tensor_tt_hadamard(left_tt, right_tt)
    rounded_hadamard = pynabled.tensor_tt_hadamard_round(left_tt, right_tt, 2, 1e-5)
    assert all(core.dtype == np.float32 for core in added.cores)
    assert all(core.dtype == np.float32 for core in hadamard.cores)
    assert all(core.dtype == np.float32 for core in rounded_hadamard.cores)
