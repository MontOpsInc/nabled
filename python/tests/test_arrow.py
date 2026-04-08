"""Tests for the PyArrow bridge (requires pynabled built with --features arrow)."""

import numpy as np
import pytest

try:
    import pyarrow as pa
except ImportError:
    pa = None

try:
    from pynabled import (
        AdamConfig,
        BFGSConfig,
        CsrMatrix,
        GradientDescentConfig,
        ILUKConfig,
        ILUTConfig,
        IterativeConfig,
        JacobianConfig,
        LineSearchConfig,
        MomentumConfig,
        ProjectedGradientConfig,
        RMSPropConfig,
    )
    from pynabled.arrow import (
        arrow_adam,
        arrow_batched_cholesky,
        arrow_batched_cosine_distance,
        arrow_batched_cosine_similarity,
        arrow_batched_dot,
        arrow_batched_l2_norm,
        arrow_batched_lu,
        arrow_batched_matmat,
        arrow_batched_matmat_broadcast_left,
        arrow_batched_matmat_broadcast_right,
        arrow_batched_normalize,
        arrow_batched_qr,
        arrow_batched_row_matvec,
        arrow_batched_svd,
        arrow_batched_symmetric_eigen,
        arrow_bfgs,
        arrow_backtracking_line_search,
        arrow_center_columns,
        arrow_cholesky_decompose,
        arrow_cholesky_inverse,
        arrow_cholesky_solve,
        arrow_column_means,
        arrow_conjugate_gradient,
        arrow_compute_pca,
        arrow_correlation_matrix,
        arrow_cosine_distance,
        arrow_cosine_similarity,
        arrow_covariance_matrix,
        arrow_csr_matrix_array,
        arrow_csr_matrix_batch_array,
        arrow_csr_matrix_batch_rows,
        arrow_csr_matrix_from_array,
        arrow_dot,
        arrow_eigen_generalized,
        arrow_eigen_nonsymmetric,
        arrow_eigen_nonsymmetric_bi,
        arrow_eigen_symmetric,
        arrow_fixed_shape_tensor_array,
        arrow_fixed_shape_tensor_numpy,
        arrow_gram_schmidt,
        arrow_gmres,
        arrow_gradient_descent,
        arrow_l2_norm,
        arrow_linear_regression,
        arrow_lu_decompose,
        arrow_lu_determinant,
        arrow_lu_inverse,
        arrow_lu_log_determinant,
        arrow_lu_solve,
        arrow_matmat,
        arrow_matvec,
        arrow_matrix_exp,
        arrow_matrix_exp_eigen,
        arrow_matrix_log_eigen,
        arrow_matrix_log_svd,
        arrow_matrix_log_taylor,
        arrow_matrix_power,
        arrow_matrix_sign,
        arrow_momentum_descent,
        arrow_numerical_gradient,
        arrow_numerical_hessian,
        arrow_numerical_jacobian,
        arrow_numerical_jacobian_central,
        arrow_pairwise_cosine_distance,
        arrow_pairwise_cosine_similarity,
        arrow_pairwise_l2_distance,
        arrow_pca_inverse_transform,
        arrow_pca_transform,
        arrow_polar_compute,
        arrow_projected_gradient_descent_box,
        arrow_qr_decompose,
        arrow_qr_decompose_pivoted,
        arrow_qr_decompose_reduced,
        arrow_qr_solve_least_squares,
        arrow_rmsprop,
        arrow_schur_compute,
        arrow_solve_lower,
        arrow_solve_upper,
        arrow_sparse_apply_ic0_preconditioner,
        arrow_sparse_apply_ildl0_preconditioner,
        arrow_sparse_apply_ilu0_preconditioner,
        arrow_sparse_apply_iluk_preconditioner,
        arrow_sparse_apply_ilut_preconditioner,
        arrow_sparse_apply_jacobi_preconditioner,
        arrow_sparse_batch_matmat_dense,
        arrow_sparse_batch_matmat_sparse,
        arrow_sparse_batch_matvec,
        arrow_sparse_batch_transpose,
        arrow_sparse_batched_matvec,
        arrow_sparse_conjugate_gradient_solve,
        arrow_sparse_csr_to_csc,
        arrow_sparse_gauss_seidel_solve,
        arrow_sparse_ic0_factor,
        arrow_sparse_ildl0_factor,
        arrow_sparse_ilu0_factor,
        arrow_sparse_iluk_factor,
        arrow_sparse_ilut_factor,
        arrow_sparse_jacobi_preconditioner,
        arrow_sparse_jacobi_solve,
        arrow_sparse_lu_factor,
        arrow_sparse_lu_solve,
        arrow_sparse_lu_solve_multiple_with_factorization,
        arrow_sparse_lu_solve_with_factorization,
        arrow_sparse_matmat_dense,
        arrow_sparse_matmat_sparse,
        arrow_sparse_matvec,
        arrow_sparse_pcg_solve,
        arrow_sparse_transpose,
        arrow_stochastic_gradient_descent,
        arrow_svd_decompose,
        arrow_svd_decompose_truncated,
        arrow_svd_decompose_with_tolerance,
        arrow_svd_null_space,
        arrow_svd_pseudo_inverse,
        arrow_tensor_batched_dot_last_axis,
        arrow_tensor_batched_matmul_last_two,
        arrow_tensor_contract_axes,
        arrow_tensor_cp_als3,
        arrow_tensor_cp_als3_diagnostics,
        arrow_tensor_cp_als3_reconstruct,
        arrow_tensor_cp_als3_with_report,
        arrow_tensor_cp_als_nd,
        arrow_tensor_cp_als_nd_diagnostics,
        arrow_tensor_cp_als_nd_reconstruct,
        arrow_tensor_cp_als_nd_with_report,
        arrow_tensor_cube_matmat,
        arrow_tensor_cube_matvec,
        arrow_tensor_einsum,
        arrow_tensor_flatten_cubes,
        arrow_tensor_hooi_nd,
        arrow_tensor_hosvd_nd,
        arrow_tensor_hosvd_nd_reconstruct,
        arrow_tensor_l2_norm_last_axis,
        arrow_tensor_normalize_last_axis,
        arrow_tensor_permute_axes,
        arrow_tensor_sum_last_axis,
        arrow_tensor_tt_add,
        arrow_tensor_tt_hadamard,
        arrow_tensor_tt_hadamard_round,
        arrow_tensor_tt_inner,
        arrow_tensor_tt_norm,
        arrow_tensor_tt_orthogonalize_left,
        arrow_tensor_tt_orthogonalize_right,
        arrow_tensor_tt_round,
        arrow_tensor_tt_svd,
        arrow_tensor_tt_svd_reconstruct,
        arrow_tensor_tucker_expand,
        arrow_tensor_tucker_project,
        arrow_variable_shape_tensor_array,
        arrow_variable_shape_tensor_rows,
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


def _matrix_numpy(array, dtype):
    return np.array(array.to_pylist(), dtype=dtype)


class NdarrowComplex64Type(pa.ExtensionType):
    def __init__(self):
        super().__init__(
            pa.list_(pa.field("item", pa.float64(), nullable=False), 2),
            "ndarrow.complex64",
        )

    def __arrow_ext_serialize__(self):
        return b""

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        return cls()

    def __reduce__(self):
        return NdarrowComplex64Type, ()


if pa is not None:
    try:
        pa.unregister_extension_type("ndarrow.complex64")
    except Exception:
        pass
    pa.register_extension_type(NdarrowComplex64Type())


def _complex_vector_array(values):
    np_values = np.asarray(values, dtype=np.complex128)
    storage = pa.array(
        [[float(value.real), float(value.imag)] for value in np_values],
        type=pa.list_(pa.field("item", pa.float64(), nullable=False), 2),
    )
    return pa.ExtensionArray.from_storage(NdarrowComplex64Type(), storage)


def _complex_vector_field(name):
    return pa.field(
        name,
        pa.list_(pa.field("item", pa.float64(), nullable=False), 2),
        nullable=False,
        metadata={"ARROW:extension:name": "ndarrow.complex64"},
    )


def _complex_vector_numpy(array):
    storage = array.storage if isinstance(array, pa.ExtensionArray) else array
    return np.array([complex(real, imag) for real, imag in storage.to_pylist()], dtype=np.complex128)


def _complex_matrix_array(values):
    np_values = np.asarray(values, dtype=np.complex128)
    return pa.FixedSizeListArray.from_arrays(
        _complex_vector_array(np_values.reshape(-1)).storage,
        type=pa.list_(_complex_vector_field("item"), np_values.shape[1]),
    )


def _complex_matrix_numpy(array):
    flat = _complex_vector_numpy(array.values)
    return flat.reshape(len(array), array.type.list_size)


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


def test_arrow_svd_qr_and_matrix_function_wrappers():
    data = _matrix_array([[1.0, 2.0], [3.0, 4.0]], np.float64)

    truncated = arrow_svd_decompose_truncated(data, 1)
    assert truncated.singular_values.shape == (1,)

    tolerant = arrow_svd_decompose_with_tolerance(data, 1e-12)
    recon = np.asarray(tolerant.u) @ np.diag(np.asarray(tolerant.singular_values)) @ np.asarray(
        tolerant.vt
    )
    np.testing.assert_allclose(recon, np.array([[1.0, 2.0], [3.0, 4.0]]), rtol=1e-10)

    pinv = arrow_svd_pseudo_inverse(data)
    np.testing.assert_allclose(
        _matrix_numpy(pinv, np.float64),
        np.linalg.pinv(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)),
        rtol=1e-10,
    )

    rank_deficient = _matrix_array([[1.0, 1.0], [2.0, 2.0]], np.float64)
    null_space = arrow_svd_null_space(rank_deficient)
    basis = _matrix_numpy(null_space, np.float64)
    np.testing.assert_allclose(
        np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float64) @ basis,
        np.zeros((2, basis.shape[1]), dtype=np.float64),
        atol=1e-10,
    )

    qr = arrow_qr_decompose(data)
    np.testing.assert_allclose(np.asarray(qr.q) @ np.asarray(qr.r), np.array([[1.0, 2.0], [3.0, 4.0]]))

    qr_reduced = arrow_qr_decompose_reduced(data)
    assert qr_reduced.rank == 2

    qr_pivoted = arrow_qr_decompose_pivoted(data)
    assert qr_pivoted.p is not None
    assert qr_pivoted.rank == 2

    ls_matrix = _matrix_array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], np.float64)
    ls_rhs = pa.array([1.0, 2.0, 3.0], type=pa.float64())
    ls_solution = arrow_qr_solve_least_squares(ls_matrix, ls_rhs)
    np.testing.assert_allclose(np.array(ls_solution.to_pylist(), dtype=np.float64), [1.0, 2.0])

    identity = _matrix_array([[1.0, 0.0], [0.0, 1.0]], np.float64)
    np.testing.assert_allclose(
        _matrix_numpy(arrow_matrix_exp(identity), np.float64),
        np.eye(2, dtype=np.float64) * np.e,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        _matrix_numpy(arrow_matrix_exp_eigen(identity), np.float64),
        np.eye(2, dtype=np.float64) * np.e,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        _matrix_numpy(arrow_matrix_log_taylor(identity), np.float64),
        np.zeros((2, 2), dtype=np.float64),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        _matrix_numpy(arrow_matrix_log_eigen(identity), np.float64),
        np.zeros((2, 2), dtype=np.float64),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        _matrix_numpy(arrow_matrix_log_svd(identity), np.float64),
        np.zeros((2, 2), dtype=np.float64),
        atol=1e-10,
    )
    diagonal = _matrix_array([[2.0, 0.0], [0.0, -3.0]], np.float64)
    np.testing.assert_allclose(
        _matrix_numpy(arrow_matrix_power(diagonal, 2.0), np.float64),
        np.array([[4.0, 0.0], [0.0, 9.0]], dtype=np.float64),
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        _matrix_numpy(arrow_matrix_sign(diagonal), np.float64),
        np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float64),
        rtol=1e-10,
    )


def test_arrow_real_decomposition_wrappers():
    spd = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    spd_arrow = _matrix_array(spd, np.float64)
    rhs = pa.array([1.0, 2.0], type=pa.float64())

    lu = arrow_lu_decompose(spd_arrow)
    np.testing.assert_allclose(np.asarray(lu.l) @ np.asarray(lu.u), spd, rtol=1e-10)
    np.testing.assert_allclose(
        np.array(arrow_lu_solve(spd_arrow, rhs).to_pylist(), dtype=np.float64),
        np.linalg.solve(spd, np.array([1.0, 2.0], dtype=np.float64)),
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        _matrix_numpy(arrow_lu_inverse(spd_arrow), np.float64),
        np.linalg.inv(spd),
        rtol=1e-10,
    )
    assert arrow_lu_determinant(spd_arrow) == pytest.approx(np.linalg.det(spd))
    logdet = arrow_lu_log_determinant(spd_arrow)
    assert logdet.sign == 1
    assert logdet.ln_abs_det == pytest.approx(np.log(np.linalg.det(spd)))

    cholesky = arrow_cholesky_decompose(spd_arrow)
    np.testing.assert_allclose(np.asarray(cholesky.l) @ np.asarray(cholesky.l).T, spd, rtol=1e-10)
    np.testing.assert_allclose(
        np.array(arrow_cholesky_solve(spd_arrow, rhs).to_pylist(), dtype=np.float64),
        np.linalg.solve(spd, np.array([1.0, 2.0], dtype=np.float64)),
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        _matrix_numpy(arrow_cholesky_inverse(spd_arrow), np.float64),
        np.linalg.inv(spd),
        rtol=1e-10,
    )

    symmetric = arrow_eigen_symmetric(spd_arrow)
    np.testing.assert_allclose(
        np.asarray(symmetric.eigenvectors)
        @ np.diag(np.asarray(symmetric.eigenvalues))
        @ np.asarray(symmetric.eigenvectors).T,
        spd,
        rtol=1e-10,
    )

    generalized = arrow_eigen_generalized(spd_arrow, _matrix_array(np.eye(2), np.float64))
    np.testing.assert_allclose(
        np.sort(np.asarray(generalized.eigenvalues)),
        np.sort(np.linalg.eigvalsh(spd)),
        rtol=1e-10,
    )

    nonsymmetric_matrix = np.array([[0.0, 1.0], [-2.0, -3.0]], dtype=np.float64)
    nonsymmetric = arrow_eigen_nonsymmetric(_matrix_array(nonsymmetric_matrix, np.float64))
    np.testing.assert_allclose(
        np.sort(np.asarray(nonsymmetric.eigenvalues)),
        np.sort(np.linalg.eigvals(nonsymmetric_matrix)),
        rtol=1e-10,
    )

    bi = arrow_eigen_nonsymmetric_bi(_matrix_array(nonsymmetric_matrix, np.float64))
    assert bi.right_eigenvectors.shape == (2, 2)
    assert bi.left_eigenvectors.shape == (2, 2)
    assert bi.balancing_diagonal.shape == (2,)

    schur = arrow_schur_compute(_matrix_array(nonsymmetric_matrix, np.float64))
    np.testing.assert_allclose(
        np.asarray(schur.q) @ np.asarray(schur.t) @ np.asarray(schur.q).T,
        nonsymmetric_matrix,
        rtol=1e-10,
        atol=1e-12,
    )

    polar = arrow_polar_compute(spd_arrow)
    np.testing.assert_allclose(np.asarray(polar.u) @ np.asarray(polar.p), spd, rtol=1e-10)


def test_arrow_pca_and_regression_wrappers():
    x = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 1.0]], dtype=np.float32)
    x_arrow = _matrix_array(x, np.float32)
    pca = arrow_compute_pca(x_arrow, n_components=2)
    assert pca.components.dtype == np.float32
    transformed = arrow_pca_transform(x_arrow, pca)
    reconstructed = arrow_pca_inverse_transform(transformed, pca)
    np.testing.assert_allclose(_matrix_numpy(reconstructed, np.float32), x, rtol=1e-5, atol=1e-5)

    y = pa.array([1.0, 3.0, 6.0], type=pa.float32())
    regression = arrow_linear_regression(x_arrow, y)
    np.testing.assert_allclose(
        regression.fitted_values,
        np.array([1.0, 3.0, 6.0], dtype=np.float32),
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(regression.residuals, np.zeros(3, dtype=np.float32), atol=1e-5)
    assert regression.r_squared == pytest.approx(1.0, abs=1e-6)


def test_arrow_complex_vector_and_matrix_dispatch():
    left = np.array([1.0 + 1.0j, 2.0 - 1.0j], dtype=np.complex128)
    right = np.array([0.5 - 0.5j, -1.0 + 3.0j], dtype=np.complex128)
    left_arrow = _complex_vector_array(left)
    right_arrow = _complex_vector_array(right)

    np.testing.assert_allclose(arrow_dot(left_arrow, right_arrow), np.vdot(left, right), rtol=1e-12)
    np.testing.assert_allclose(arrow_l2_norm(left_arrow), np.linalg.norm(left), rtol=1e-12)
    np.testing.assert_allclose(
        _complex_vector_numpy(arrow_cosine_similarity(left_arrow, right_arrow))[0],
        np.vdot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)),
        rtol=1e-12,
        atol=1e-12,
    )

    batch_left = np.array(
        [[1.0 + 1.0j, 0.0 + 2.0j], [2.0 + 0.0j, 0.0 + 2.0j]],
        dtype=np.complex128,
    )
    batch_right = np.array(
        [[1.0 - 1.0j, 2.0 + 0.0j], [0.0 + 2.0j, 2.0 + 0.0j]],
        dtype=np.complex128,
    )
    batch_left_arrow = _complex_matrix_array(batch_left)
    batch_right_arrow = _complex_matrix_array(batch_right)

    np.testing.assert_allclose(
        _complex_vector_numpy(arrow_batched_dot(batch_left_arrow, batch_right_arrow)),
        np.sum(np.conj(batch_left) * batch_right, axis=1),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.array(arrow_batched_l2_norm(batch_left_arrow).to_pylist(), dtype=np.float64),
        np.linalg.norm(batch_left, axis=1),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        _complex_vector_numpy(arrow_batched_cosine_similarity(batch_left_arrow, batch_right_arrow)),
        np.sum(np.conj(batch_left) * batch_right, axis=1)
        / (np.linalg.norm(batch_left, axis=1) * np.linalg.norm(batch_right, axis=1)),
        rtol=1e-12,
        atol=1e-12,
    )
    normalized = _complex_matrix_numpy(arrow_batched_normalize(batch_left_arrow))
    np.testing.assert_allclose(
        np.linalg.norm(normalized, axis=1),
        np.ones(batch_left.shape[0], dtype=np.float64),
        rtol=1e-12,
        atol=1e-12,
    )

    matrix = np.array(
        [[1.0 + 1.0j, 0.0 - 1.0j], [2.0 + 0.0j, 1.0 + 2.0j]],
        dtype=np.complex128,
    )
    vector = np.array([1.0 + 0.0j, 0.5 - 0.5j], dtype=np.complex128)
    other = np.array(
        [[1.0 + 1.0j, 0.0 + 1.0j], [2.0 + 0.0j, 1.0 - 1.0j]],
        dtype=np.complex128,
    )
    np.testing.assert_allclose(
        _complex_vector_numpy(arrow_matvec(_complex_matrix_array(matrix), _complex_vector_array(vector))),
        matrix @ vector,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        _complex_matrix_numpy(arrow_matmat(_complex_matrix_array(matrix), _complex_matrix_array(other))),
        matrix @ other,
        rtol=1e-12,
        atol=1e-12,
    )

    with pytest.raises(TypeError, match="does not currently admit"):
        arrow_cosine_distance(left_arrow, right_arrow)


def test_arrow_complex_stats_orthogonalization_and_triangular_dispatch():
    matrix = np.array(
        [
            [1.0 + 1.0j, 2.0 - 0.5j],
            [3.0 + 0.0j, 4.0 + 1.0j],
            [5.0 - 1.0j, 8.0 + 0.25j],
        ],
        dtype=np.complex128,
    )
    matrix_arrow = _complex_matrix_array(matrix)

    np.testing.assert_allclose(
        _complex_vector_numpy(arrow_column_means(matrix_arrow)),
        matrix.mean(axis=0),
        rtol=1e-12,
        atol=1e-12,
    )

    centered = _complex_matrix_numpy(arrow_center_columns(matrix_arrow))
    np.testing.assert_allclose(centered.mean(axis=0), np.zeros(2, dtype=np.complex128), atol=1e-12)
    np.testing.assert_allclose(
        _complex_matrix_numpy(arrow_covariance_matrix(matrix_arrow)),
        np.cov(matrix, rowvar=False).T,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        _complex_matrix_numpy(arrow_correlation_matrix(matrix_arrow)),
        np.corrcoef(matrix, rowvar=False).T,
        rtol=1e-12,
        atol=1e-12,
    )

    orthogonalized = _complex_matrix_numpy(arrow_gram_schmidt(matrix_arrow))
    np.testing.assert_allclose(
        orthogonalized.conj().T @ orthogonalized,
        np.eye(2, dtype=np.complex128),
        rtol=1e-12,
        atol=1e-12,
    )

    lower = np.array([[2.0 + 0.0j, 0.0], [1.0 - 1.0j, 3.0 + 0.0j]], dtype=np.complex128)
    upper = np.array([[2.0 + 0.5j, 1.0 - 0.5j], [0.0, 3.0 - 1.0j]], dtype=np.complex128)
    rhs = np.array([2.0 + 1.0j, 7.0 - 2.0j], dtype=np.complex128)
    lower_solution = _complex_vector_numpy(
        arrow_solve_lower(_complex_matrix_array(lower), _complex_vector_array(rhs))
    )
    upper_solution = _complex_vector_numpy(
        arrow_solve_upper(_complex_matrix_array(upper), _complex_vector_array(rhs))
    )
    np.testing.assert_allclose(lower @ lower_solution, rhs, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(upper @ upper_solution, rhs, rtol=1e-12, atol=1e-12)


def test_arrow_complex_decomposition_matrix_functions_and_ml_dispatch():
    matrix = np.array(
        [[2.0 + 1.0j, 1.0 - 0.5j], [0.0 + 0.25j, 3.0 - 1.0j]],
        dtype=np.complex128,
    )
    matrix_arrow = _complex_matrix_array(matrix)
    rhs = np.array([1.0 + 2.0j, -0.5 + 1.0j], dtype=np.complex128)
    rhs_arrow = _complex_vector_array(rhs)

    svd = arrow_svd_decompose(matrix_arrow)
    np.testing.assert_allclose(
        np.asarray(svd.u) @ np.diag(np.asarray(svd.singular_values)) @ np.asarray(svd.vt),
        matrix,
        rtol=1e-12,
        atol=1e-12,
    )

    qr = arrow_qr_decompose(matrix_arrow)
    np.testing.assert_allclose(np.asarray(qr.q) @ np.asarray(qr.r), matrix, rtol=1e-12, atol=1e-12)

    lu_solution = _complex_vector_numpy(arrow_lu_solve(matrix_arrow, rhs_arrow))
    np.testing.assert_allclose(lu_solution, np.linalg.solve(matrix, rhs), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        _complex_matrix_numpy(arrow_lu_inverse(matrix_arrow)),
        np.linalg.inv(matrix),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(arrow_lu_determinant(matrix_arrow), np.linalg.det(matrix), rtol=1e-12)

    hermitian = matrix.conj().T @ matrix + np.eye(2, dtype=np.complex128)
    hermitian_arrow = _complex_matrix_array(hermitian)
    cholesky = arrow_cholesky_decompose(hermitian_arrow)
    np.testing.assert_allclose(
        np.asarray(cholesky.l) @ np.asarray(cholesky.l).conj().T,
        hermitian,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        _complex_vector_numpy(arrow_cholesky_solve(hermitian_arrow, rhs_arrow)),
        np.linalg.solve(hermitian, rhs),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        _complex_matrix_numpy(arrow_cholesky_inverse(hermitian_arrow)),
        np.linalg.inv(hermitian),
        rtol=1e-12,
        atol=1e-12,
    )

    nonsymmetric = arrow_eigen_nonsymmetric(matrix_arrow)
    np.testing.assert_allclose(
        np.sort_complex(np.asarray(nonsymmetric.eigenvalues)),
        np.sort_complex(np.linalg.eigvals(matrix)),
        rtol=1e-12,
        atol=1e-12,
    )

    schur = arrow_schur_compute(matrix_arrow)
    np.testing.assert_allclose(
        np.asarray(schur.q) @ np.asarray(schur.t) @ np.asarray(schur.q).conj().T,
        matrix,
        rtol=1e-12,
        atol=1e-12,
    )

    polar = arrow_polar_compute(matrix_arrow)
    np.testing.assert_allclose(np.asarray(polar.u) @ np.asarray(polar.p), matrix, rtol=1e-12, atol=1e-12)

    diagonal = np.diag(np.array([1.0 + 1.0j, 2.0 - 0.25j], dtype=np.complex128))
    diagonal_arrow = _complex_matrix_array(diagonal)
    np.testing.assert_allclose(
        _complex_matrix_numpy(arrow_matrix_exp(diagonal_arrow)),
        np.diag(np.exp(np.diag(diagonal))),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        _complex_matrix_numpy(arrow_matrix_exp_eigen(diagonal_arrow)),
        np.diag(np.exp(np.diag(diagonal))),
        rtol=1e-12,
        atol=1e-12,
    )

    positive = np.diag(np.array([2.0 + 0.0j, 3.0 + 0.0j], dtype=np.complex128))
    positive_arrow = _complex_matrix_array(positive)
    np.testing.assert_allclose(
        _complex_matrix_numpy(arrow_matrix_log_eigen(positive_arrow)),
        np.diag(np.log(np.diag(positive))),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        _complex_matrix_numpy(arrow_matrix_log_svd(positive_arrow)),
        np.diag(np.log(np.diag(positive))),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        _complex_matrix_numpy(arrow_matrix_power(positive_arrow, 2.0)),
        np.diag(np.diag(positive) ** 2),
        rtol=1e-12,
        atol=1e-12,
    )
    signed = np.diag(np.array([1.0 + 0.0j, -3.0 + 0.0j], dtype=np.complex128))
    np.testing.assert_allclose(
        _complex_matrix_numpy(arrow_matrix_sign(_complex_matrix_array(signed))),
        np.diag(np.array([1.0 + 0.0j, -1.0 + 0.0j], dtype=np.complex128)),
        rtol=1e-12,
        atol=1e-12,
    )

    x = np.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 1.0j, 1.0 + 0.0j], [2.0 - 1.0j, 1.0 + 1.0j]], dtype=np.complex128)
    x_arrow = _complex_matrix_array(x)
    pca = arrow_compute_pca(x_arrow, n_components=2)
    transformed = arrow_pca_transform(x_arrow, pca)
    reconstructed = arrow_pca_inverse_transform(transformed, pca)
    np.testing.assert_allclose(_complex_matrix_numpy(reconstructed), x, rtol=1e-10, atol=1e-10)

    x_reg = np.array([[1.0 + 0.0j], [2.0 - 1.0j], [3.0 + 1.0j]], dtype=np.complex128)
    y_reg = (2.0 - 0.5j) * x_reg[:, 0]
    regression = arrow_linear_regression(
        _complex_matrix_array(x_reg),
        _complex_vector_array(y_reg),
        add_intercept=False,
    )
    np.testing.assert_allclose(regression.fitted_values, y_reg, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(regression.residuals, np.zeros_like(y_reg), atol=1e-12)
    np.testing.assert_allclose(regression.coefficients, np.array([2.0 - 0.5j], dtype=np.complex128))
    assert regression.r_squared == pytest.approx(1.0, abs=1e-12)

    with pytest.raises(TypeError, match="does not currently admit"):
        arrow_qr_decompose_reduced(matrix_arrow)


def test_arrow_batched_result_wrappers_return_typed_objects():
    matrices = pa.FixedShapeTensorArray.from_numpy_ndarray(
        np.array(
            [
                [[4.0, 1.0], [1.0, 3.0]],
                [[3.0, 0.0], [0.0, 2.0]],
            ],
            dtype=np.float64,
        )
    )

    qr_results = arrow_batched_qr(matrices)
    assert qr_results[0].rank == 2
    np.testing.assert_allclose(
        np.asarray(qr_results[0].q) @ np.asarray(qr_results[0].r),
        matrices.to_numpy_ndarray()[0],
        rtol=1e-10,
    )

    svd_results = arrow_batched_svd(matrices)
    np.testing.assert_allclose(
        np.asarray(svd_results[0].u)
        @ np.diag(np.asarray(svd_results[0].singular_values))
        @ np.asarray(svd_results[0].vt),
        matrices.to_numpy_ndarray()[0],
        rtol=1e-10,
    )

    lu_results = arrow_batched_lu(matrices)
    np.testing.assert_allclose(
        np.asarray(lu_results[0].l) @ np.asarray(lu_results[0].u),
        matrices.to_numpy_ndarray()[0],
        rtol=1e-10,
    )

    cholesky_results = arrow_batched_cholesky(matrices)
    np.testing.assert_allclose(
        np.asarray(cholesky_results[0].l) @ np.asarray(cholesky_results[0].l).T,
        matrices.to_numpy_ndarray()[0],
        rtol=1e-10,
    )

    eigen_results = arrow_batched_symmetric_eigen(matrices)
    np.testing.assert_allclose(
        np.asarray(eigen_results[0].eigenvectors)
        @ np.diag(np.asarray(eigen_results[0].eigenvalues))
        @ np.asarray(eigen_results[0].eigenvectors).T,
        matrices.to_numpy_ndarray()[0],
        rtol=1e-10,
    )


def test_arrow_iterative_wrappers_real_and_complex():
    matrix = _matrix_array([[4.0, 1.0], [1.0, 3.0]], np.float64)
    rhs = pa.array([1.0, 2.0], type=pa.float64())
    expected = np.linalg.solve(np.array(matrix.to_pylist(), dtype=np.float64), np.array(rhs.to_pylist(), dtype=np.float64))

    cg = arrow_conjugate_gradient(matrix, rhs, config=IterativeConfig(tolerance=1e-12, max_iterations=128))
    gmres = arrow_gmres(matrix, rhs, tolerance=1e-12, max_iterations=128)
    np.testing.assert_allclose(np.array(cg.to_pylist(), dtype=np.float64), expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(np.array(gmres.to_pylist(), dtype=np.float64), expected, rtol=1e-10, atol=1e-10)

    complex_matrix = np.array([[5.0 + 0.0j, 1.0 - 1.0j], [1.0 + 1.0j, 4.0 + 0.0j]], dtype=np.complex128)
    complex_rhs = np.array([1.0 + 1.0j, 2.0 - 1.0j], dtype=np.complex128)
    complex_expected = np.linalg.solve(complex_matrix, complex_rhs)

    cg_complex = arrow_conjugate_gradient(_complex_matrix_array(complex_matrix), _complex_vector_array(complex_rhs))
    gmres_complex = arrow_gmres(
        _complex_matrix_array(complex_matrix),
        _complex_vector_array(complex_rhs),
        config=IterativeConfig(tolerance=1e-12, max_iterations=128),
    )
    np.testing.assert_allclose(_complex_vector_numpy(cg_complex), complex_expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(_complex_vector_numpy(gmres_complex), complex_expected, rtol=1e-10, atol=1e-10)


def test_arrow_jacobian_wrappers():
    x = pa.array([1.5, -0.5], type=pa.float64())

    def vector_fn(vector):
        values = np.array(vector.to_pylist(), dtype=np.float64)
        output = np.array([values[0] ** 2 + values[1], values[0] - values[1] ** 2], dtype=np.float64)
        return pa.array(output.tolist(), type=pa.float64())

    def scalar_fn(vector):
        values = np.array(vector.to_pylist(), dtype=np.float64)
        return float(values[0] ** 2 + 3.0 * values[1] ** 2)

    expected_jacobian = np.array([[3.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    expected_gradient = np.array([3.0, -3.0], dtype=np.float64)
    expected_hessian = np.array([[2.0, 0.0], [0.0, 6.0]], dtype=np.float64)

    jacobian = arrow_numerical_jacobian(vector_fn, x, config=JacobianConfig(step_size=1e-6, tolerance=1e-12, max_iterations=64))
    jacobian_central = arrow_numerical_jacobian_central(vector_fn, x, step_size=1e-6, tolerance=1e-12, max_iterations=64)
    gradient = arrow_numerical_gradient(scalar_fn, x, config=JacobianConfig(step_size=1e-6, tolerance=1e-12, max_iterations=64))
    hessian = arrow_numerical_hessian(scalar_fn, x, step_size=1e-4, tolerance=1e-10, max_iterations=64)

    np.testing.assert_allclose(_matrix_numpy(jacobian, np.float64), expected_jacobian, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(_matrix_numpy(jacobian_central, np.float64), expected_jacobian, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.array(gradient.to_pylist(), dtype=np.float64), expected_gradient, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(_matrix_numpy(hessian, np.float64), expected_hessian, rtol=1e-3, atol=1e-4)


def test_arrow_optimization_wrappers_real_and_complex():
    target_real = np.array([3.0], dtype=np.float64)

    def objective_real(vector):
        values = np.array(vector.to_pylist(), dtype=np.float64)
        return float(np.sum((values - target_real) ** 2))

    def gradient_real(vector):
        values = np.array(vector.to_pylist(), dtype=np.float64)
        return pa.array((2.0 * (values - target_real)).tolist(), type=pa.float64())

    def stochastic_gradient_real(vector, _iteration):
        return gradient_real(vector)

    point = pa.array([0.0], type=pa.float64())
    direction = pa.array([6.0], type=pa.float64())
    line_step = arrow_backtracking_line_search(
        point,
        direction,
        objective_real,
        gradient_real,
        config=LineSearchConfig(initial_step=1.0, contraction=0.5, sufficient_decrease=1e-4, max_iterations=32),
    )
    assert line_step > 0.0

    gd = arrow_gradient_descent(
        pa.array([-5.0], type=pa.float64()),
        objective_real,
        gradient_real,
        config=GradientDescentConfig(learning_rate=0.1, max_iterations=512, tolerance=1e-10),
    )
    adam = arrow_adam(
        pa.array([-5.0], type=pa.float64()),
        objective_real,
        gradient_real,
        config=AdamConfig(learning_rate=0.1, max_iterations=512, tolerance=1e-10),
    )
    momentum = arrow_momentum_descent(
        pa.array([-5.0], type=pa.float64()),
        objective_real,
        gradient_real,
        config=MomentumConfig(learning_rate=0.05, momentum=0.9, max_iterations=512, tolerance=1e-10),
    )
    rmsprop = arrow_rmsprop(
        pa.array([-5.0], type=pa.float64()),
        objective_real,
        gradient_real,
        config=RMSPropConfig(learning_rate=0.1, max_iterations=512, tolerance=1e-10),
    )
    bfgs = arrow_bfgs(
        pa.array([-5.0], type=pa.float64()),
        objective_real,
        gradient_real,
        config=BFGSConfig(step_size=0.5, max_iterations=256, tolerance=1e-12, curvature_tolerance=1e-9),
    )
    projected = arrow_projected_gradient_descent_box(
        pa.array([-5.0], type=pa.float64()),
        objective_real,
        gradient_real,
        pa.array([0.0], type=pa.float64()),
        pa.array([2.5], type=pa.float64()),
        config=ProjectedGradientConfig(learning_rate=0.1, max_iterations=512, tolerance=1e-10),
    )
    stochastic = arrow_stochastic_gradient_descent(
        pa.array([-3.0], type=pa.float64()),
        stochastic_gradient_real,
        config=GradientDescentConfig(learning_rate=0.05, max_iterations=2_000, tolerance=1e-8),
    )

    for result in (gd, adam, momentum, rmsprop, bfgs, stochastic):
        np.testing.assert_allclose(np.array(result.to_pylist(), dtype=np.float64), target_real, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(np.array(projected.to_pylist(), dtype=np.float64), np.array([2.5]), rtol=1e-3, atol=1e-3)

    target_complex = np.array([3.0 + 2.0j], dtype=np.complex128)

    def objective_complex(vector):
        values = _complex_vector_numpy(vector)
        delta = values - target_complex
        return float(np.vdot(delta, delta).real)

    def gradient_complex(vector):
        values = _complex_vector_numpy(vector)
        return _complex_vector_array(2.0 * (values - target_complex))

    def stochastic_gradient_complex(vector, _iteration):
        return gradient_complex(vector)

    complex_point = _complex_vector_array([0.0 + 0.0j])
    complex_direction = _complex_vector_array([6.0 + 4.0j])
    complex_line_step = arrow_backtracking_line_search(
        complex_point,
        complex_direction,
        objective_complex,
        gradient_complex,
        initial_step=1.0,
        contraction=0.5,
        sufficient_decrease=1e-4,
        max_iterations=32,
    )
    assert complex_line_step > 0.0

    gd_complex = arrow_gradient_descent(
        _complex_vector_array([-5.0 - 5.0j]),
        objective_complex,
        gradient_complex,
        config=GradientDescentConfig(learning_rate=0.1, max_iterations=512, tolerance=1e-10),
    )
    adam_complex = arrow_adam(
        _complex_vector_array([-5.0 - 5.0j]),
        objective_complex,
        gradient_complex,
        config=AdamConfig(learning_rate=0.1),
    )
    momentum_complex = arrow_momentum_descent(
        _complex_vector_array([-5.0 - 5.0j]),
        objective_complex,
        gradient_complex,
        config=MomentumConfig(learning_rate=0.05, momentum=0.9, max_iterations=512, tolerance=1e-10),
    )
    rmsprop_complex = arrow_rmsprop(
        _complex_vector_array([-5.0 - 5.0j]),
        objective_complex,
        gradient_complex,
        config=RMSPropConfig(learning_rate=0.1, max_iterations=512, tolerance=1e-10),
    )
    bfgs_complex = arrow_bfgs(
        _complex_vector_array([-5.0 - 5.0j]),
        objective_complex,
        gradient_complex,
        config=BFGSConfig(step_size=0.5, max_iterations=256, tolerance=1e-12, curvature_tolerance=1e-9),
    )
    projected_complex = arrow_projected_gradient_descent_box(
        _complex_vector_array([-5.0 - 5.0j]),
        objective_complex,
        gradient_complex,
        _complex_vector_array([0.0 + 0.0j]),
        _complex_vector_array([2.5 + 2.5j]),
        config=ProjectedGradientConfig(learning_rate=0.1, max_iterations=512, tolerance=1e-10),
    )
    stochastic_complex = arrow_stochastic_gradient_descent(
        _complex_vector_array([-3.0 - 1.0j]),
        stochastic_gradient_complex,
        config=GradientDescentConfig(learning_rate=0.05, max_iterations=2_000, tolerance=1e-8),
    )

    for result in (gd_complex, adam_complex, momentum_complex, rmsprop_complex, bfgs_complex, stochastic_complex):
        np.testing.assert_allclose(_complex_vector_numpy(result), target_complex, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(_complex_vector_numpy(projected_complex), np.array([2.5 + 2.0j]), rtol=1e-3, atol=1e-3)


def test_arrow_sparse_extension_helpers_and_object_kernels():
    matrix = CsrMatrix(
        (3, 4),
        np.array([0, 2, 3, 5], dtype=np.int32),
        np.array([0, 2, 1, 0, 3], dtype=np.int32),
        np.array([1.0, 5.0, 2.0, 3.0, 4.0], dtype=np.float64),
    )
    matrix_arrow = arrow_csr_matrix_array(matrix)
    roundtrip = arrow_csr_matrix_from_array(matrix_arrow)
    np.testing.assert_array_equal(roundtrip.indptr, matrix.indptr)
    np.testing.assert_array_equal(roundtrip.indices, matrix.indices)
    np.testing.assert_allclose(roundtrip.data, matrix.data)

    vector = pa.array([1.0, 2.0, 3.0, 4.0], type=pa.float64())
    matvec = arrow_sparse_matvec(matrix_arrow, vector)
    np.testing.assert_allclose(np.array(matvec.to_pylist(), dtype=np.float64), [16.0, 4.0, 19.0])

    dense = _matrix_array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]], np.float64)
    dense_out = arrow_sparse_matmat_dense(matrix_arrow, dense)
    np.testing.assert_allclose(
        _matrix_numpy(dense_out, np.float64),
        np.array([[6.0, 5.0], [0.0, 2.0], [11.0, 4.0]], dtype=np.float64),
    )

    dense_batch = _matrix_array([[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]], np.float64)
    batched = arrow_sparse_batched_matvec(matrix_arrow, dense_batch)
    np.testing.assert_allclose(
        _matrix_numpy(batched, np.float64),
        np.array([[1.0, 0.0, 3.0], [6.0, 2.0, 7.0]], dtype=np.float64),
    )

    transposed = arrow_sparse_transpose(matrix_arrow)
    assert transposed.shape == (4, 3)
    csc = arrow_sparse_csr_to_csc(matrix_arrow)
    assert csc.shape == (3, 4)

    product = arrow_sparse_matmat_sparse(matrix_arrow, arrow_csr_matrix_array(transposed))
    dense_product = product.matmat_dense(np.eye(product.ncols, dtype=np.float64))
    np.testing.assert_allclose(
        dense_product,
        np.array([[26.0, 0.0, 3.0], [0.0, 4.0, 0.0], [3.0, 0.0, 25.0]], dtype=np.float64),
    )


def test_arrow_sparse_factorization_and_reuse_wrappers():
    matrix = CsrMatrix(
        (2, 2),
        np.array([0, 2, 4], dtype=np.int32),
        np.array([0, 1, 0, 1], dtype=np.int32),
        np.array([4.0, 1.0, 1.0, 3.0], dtype=np.float64),
    )
    matrix_arrow = arrow_csr_matrix_array(matrix)
    rhs = pa.array([1.0, 2.0], type=pa.float64())

    direct = arrow_sparse_lu_solve(matrix_arrow, rhs)
    np.testing.assert_allclose(
        np.array(direct.to_pylist(), dtype=np.float64),
        np.array([1.0 / 11.0, 7.0 / 11.0], dtype=np.float64),
        rtol=1e-10,
    )

    jacobi = arrow_sparse_jacobi_solve(matrix_arrow, rhs, tolerance=1e-12, max_iterations=100)
    gauss = arrow_sparse_gauss_seidel_solve(
        matrix_arrow, rhs, tolerance=1e-12, max_iterations=100
    )
    cg = arrow_sparse_conjugate_gradient_solve(
        matrix_arrow, rhs, tolerance=1e-12, max_iterations=20
    )
    pcg = arrow_sparse_pcg_solve(matrix_arrow, rhs, tolerance=1e-12, max_iterations=20)
    for result in (jacobi, gauss, cg, pcg):
        np.testing.assert_allclose(
            np.array(result.to_pylist(), dtype=np.float64),
            np.array([1.0 / 11.0, 7.0 / 11.0], dtype=np.float64),
            rtol=1e-5,
            atol=1e-5,
        )

    jacobi_pre = arrow_sparse_jacobi_preconditioner(matrix_arrow)
    jacobi_applied = arrow_sparse_apply_jacobi_preconditioner(jacobi_pre, rhs)
    np.testing.assert_allclose(
        np.array(jacobi_applied.to_pylist(), dtype=np.float64),
        np.array([0.25, 2.0 / 3.0], dtype=np.float64),
        rtol=1e-10,
    )

    ilu0 = arrow_sparse_ilu0_factor(matrix_arrow)
    ilu0_applied = arrow_sparse_apply_ilu0_preconditioner(ilu0, rhs)
    assert ilu0.matrix.shape == (2, 2)
    np.testing.assert_allclose(
        np.array(ilu0_applied.to_pylist(), dtype=np.float64),
        np.array([1.0 / 11.0, 7.0 / 11.0], dtype=np.float64),
        rtol=1e-10,
    )

    ilut = arrow_sparse_ilut_factor(matrix_arrow, config=ILUTConfig.balanced())
    iluk = arrow_sparse_iluk_factor(matrix_arrow, config=ILUKConfig.balanced())
    ic0 = arrow_sparse_ic0_factor(matrix_arrow)
    ildl0 = arrow_sparse_ildl0_factor(matrix_arrow)
    for result in (
        arrow_sparse_apply_ilut_preconditioner(ilut, rhs),
        arrow_sparse_apply_iluk_preconditioner(iluk, rhs),
        arrow_sparse_apply_ic0_preconditioner(ic0, rhs),
        arrow_sparse_apply_ildl0_preconditioner(ildl0, rhs),
    ):
        assert isinstance(result, pa.Array)

    lu = arrow_sparse_lu_factor(matrix_arrow)
    solved = arrow_sparse_lu_solve_with_factorization(matrix_arrow, rhs, lu)
    np.testing.assert_allclose(
        np.array(solved.to_pylist(), dtype=np.float64),
        np.array([1.0 / 11.0, 7.0 / 11.0], dtype=np.float64),
        rtol=1e-10,
    )
    rhs_multi = _matrix_array([[1.0, 0.0], [2.0, 1.0]], np.float64)
    solved_multi = arrow_sparse_lu_solve_multiple_with_factorization(matrix_arrow, rhs_multi, lu)
    np.testing.assert_allclose(
        _matrix_numpy(solved_multi, np.float64),
        np.array([[1.0 / 11.0, -1.0 / 11.0], [7.0 / 11.0, 4.0 / 11.0]], dtype=np.float64),
        rtol=1e-10,
    )


def test_arrow_sparse_batch_wrappers():
    batch = arrow_csr_matrix_batch_array(
        [
            CsrMatrix(
                (2, 2),
                np.array([0, 1, 2], dtype=np.int32),
                np.array([0, 1], dtype=np.int32),
                np.array([2.0, 3.0], dtype=np.float64),
            ),
            CsrMatrix(
                (1, 3),
                np.array([0, 2], dtype=np.int32),
                np.array([0, 2], dtype=np.int32),
                np.array([1.0, 4.0], dtype=np.float64),
            ),
        ]
    )
    vectors = arrow_variable_shape_tensor_array(
        [
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([1.0, 0.0, 1.0], dtype=np.float64),
        ],
        uniform_shape=[None],
    )
    matvec = arrow_sparse_batch_matvec(batch, vectors)
    matvec_rows = arrow_variable_shape_tensor_rows(matvec)
    np.testing.assert_allclose(matvec_rows[0], np.array([2.0, 6.0], dtype=np.float64))
    np.testing.assert_allclose(matvec_rows[1], np.array([5.0], dtype=np.float64))

    dense_rhs = arrow_variable_shape_tensor_array(
        [
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
            np.array([[1.0], [0.0], [1.0]], dtype=np.float64),
        ],
        uniform_shape=[None, None],
    )
    dense_out = arrow_sparse_batch_matmat_dense(batch, dense_rhs)
    dense_rows = arrow_variable_shape_tensor_rows(dense_out)
    np.testing.assert_allclose(
        dense_rows[0],
        np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float64),
    )
    np.testing.assert_allclose(dense_rows[1], np.array([[5.0]], dtype=np.float64))

    transposed = arrow_sparse_batch_transpose(batch)
    transpose_rows = arrow_csr_matrix_batch_rows(transposed)
    assert transpose_rows[0].shape == (2, 2)
    assert transpose_rows[1].shape == (3, 1)

    product = arrow_sparse_batch_matmat_sparse(batch, transposed)
    product_rows = arrow_csr_matrix_batch_rows(product)
    first_product_dense = product_rows[0].matmat_dense(np.eye(2, dtype=np.float64))
    np.testing.assert_allclose(
        first_product_dense,
        np.array([[4.0, 0.0], [0.0, 9.0]], dtype=np.float64),
    )


def test_arrow_tensor_fixed_shape_real_surface():
    tensor = arrow_fixed_shape_tensor_array(
        np.array(
            [
                [[3.0, 4.0], [0.0, 5.0]],
                [[8.0, 15.0], [7.0, 24.0]],
            ],
            dtype=np.float64,
        )
    )
    np.testing.assert_allclose(
        arrow_fixed_shape_tensor_numpy(tensor),
        np.array(
            [
                [[3.0, 4.0], [0.0, 5.0]],
                [[8.0, 15.0], [7.0, 24.0]],
            ],
            dtype=np.float64,
        ),
    )

    summed = arrow_tensor_sum_last_axis(tensor)
    np.testing.assert_allclose(
        arrow_fixed_shape_tensor_numpy(summed),
        np.array([[7.0, 5.0], [23.0, 31.0]], dtype=np.float64),
    )

    norms = arrow_tensor_l2_norm_last_axis(tensor)
    np.testing.assert_allclose(
        arrow_fixed_shape_tensor_numpy(norms),
        np.array([[5.0, 5.0], [17.0, 25.0]], dtype=np.float64),
    )

    normalized = arrow_tensor_normalize_last_axis(tensor)
    np.testing.assert_allclose(
        np.linalg.norm(arrow_fixed_shape_tensor_numpy(normalized), axis=-1),
        np.ones((2, 2), dtype=np.float64),
        rtol=1e-10,
        atol=1e-10,
    )

    other = arrow_fixed_shape_tensor_array(
        np.array(
            [
                [[1.0, 1.0], [2.0, 0.0]],
                [[1.0, 0.0], [0.0, 1.0]],
            ],
            dtype=np.float64,
        )
    )
    dot = arrow_tensor_batched_dot_last_axis(tensor, other)
    np.testing.assert_allclose(
        arrow_fixed_shape_tensor_numpy(dot),
        np.array([[7.0, 0.0], [8.0, 24.0]], dtype=np.float64),
    )

    permuted = arrow_tensor_permute_axes(tensor, [1, 0, 2])
    assert arrow_fixed_shape_tensor_numpy(permuted).shape == (2, 2, 2)

    contract_left = arrow_fixed_shape_tensor_array(
        np.array(
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                [[2.0, 1.0], [0.0, 1.0], [1.0, 0.0]],
            ],
            dtype=np.float64,
        )
    )
    contract_right = arrow_fixed_shape_tensor_array(
        np.array(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[2.0, 1.0], [1.0, 2.0]],
            ],
            dtype=np.float64,
        )
    )
    contracted = arrow_tensor_contract_axes(contract_left, contract_right, [2], [1])
    assert arrow_fixed_shape_tensor_numpy(contracted).shape == (2, 3, 2, 2)

    left_batch = arrow_fixed_shape_tensor_array(
        np.array(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[2.0, 0.0], [1.0, 2.0]],
            ],
            dtype=np.float64,
        )
    )
    right_batch = arrow_fixed_shape_tensor_array(
        np.array(
            [
                [[5.0, 6.0], [7.0, 8.0]],
                [[1.0, 0.0], [0.0, 1.0]],
            ],
            dtype=np.float64,
        )
    )
    batched_mm = arrow_tensor_batched_matmul_last_two(left_batch, right_batch)
    np.testing.assert_allclose(
        arrow_fixed_shape_tensor_numpy(batched_mm),
        np.array(
            [
                [[19.0, 22.0], [43.0, 50.0]],
                [[2.0, 0.0], [1.0, 2.0]],
            ],
            dtype=np.float64,
        ),
    )

    cube_matvec = arrow_tensor_cube_matvec(
        left_batch,
        _matrix_array([[1.0, 0.0], [1.0, 1.0]], np.float64),
    )
    np.testing.assert_allclose(
        _matrix_numpy(cube_matvec, np.float64),
        np.array([[1.0, 3.0], [2.0, 3.0]], dtype=np.float64),
    )

    cube_matmat = arrow_tensor_cube_matmat(left_batch, right_batch)
    np.testing.assert_allclose(
        arrow_fixed_shape_tensor_numpy(cube_matmat),
        arrow_fixed_shape_tensor_numpy(batched_mm),
        rtol=1e-10,
        atol=1e-10,
    )

    flat = arrow_tensor_flatten_cubes(left_batch)
    np.testing.assert_allclose(
        _matrix_numpy(flat, np.float64),
        np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 0.0, 1.0, 2.0]], dtype=np.float64),
    )

    einsum = arrow_tensor_einsum("bij,bjk->bik", left_batch, right_batch)
    np.testing.assert_allclose(
        arrow_fixed_shape_tensor_numpy(einsum),
        arrow_fixed_shape_tensor_numpy(batched_mm),
        rtol=1e-10,
        atol=1e-10,
    )


def test_arrow_tensor_advanced_real_result_families():
    tensor = arrow_fixed_shape_tensor_array(
        np.array(
            [
                [[1.0, 3.0], [2.0, 6.0]],
                [[2.0, 6.0], [4.0, 12.0]],
            ],
            dtype=np.float64,
        )
    )

    cp3 = arrow_tensor_cp_als3(tensor, 1, max_iterations=100, tolerance=1e-8)
    cp3_metrics = arrow_tensor_cp_als3_diagnostics(tensor, cp3)
    assert cp3_metrics.fit > 0.99
    cp3_recon = arrow_tensor_cp_als3_reconstruct(cp3, field_name="cp3")
    np.testing.assert_allclose(
        arrow_fixed_shape_tensor_numpy(cp3_recon),
        arrow_fixed_shape_tensor_numpy(tensor),
        rtol=1e-5,
        atol=1e-5,
    )

    cp3_with_report, cp3_report = arrow_tensor_cp_als3_with_report(
        tensor,
        1,
        max_iterations=100,
        tolerance=1e-8,
    )
    assert isinstance(cp3_with_report, type(cp3))
    assert cp3_report.convergence.iterations_run > 0

    cp_nd = arrow_tensor_cp_als_nd(tensor, 1, max_iterations=100, tolerance=1e-8)
    cp_nd_metrics = arrow_tensor_cp_als_nd_diagnostics(tensor, cp_nd)
    assert cp_nd_metrics.fit > 0.99
    cp_nd_recon = arrow_tensor_cp_als_nd_reconstruct(cp_nd, field_name="cp_nd")
    np.testing.assert_allclose(
        arrow_fixed_shape_tensor_numpy(cp_nd_recon),
        arrow_fixed_shape_tensor_numpy(tensor),
        rtol=1e-5,
        atol=1e-5,
    )
    cp_nd_with_report, cp_nd_report = arrow_tensor_cp_als_nd_with_report(
        tensor,
        1,
        max_iterations=100,
        tolerance=1e-8,
    )
    assert cp_nd_report.convergence.iterations_run > 0

    hosvd = arrow_tensor_hosvd_nd(tensor, [1, 1, 1])
    hosvd_recon = arrow_tensor_hosvd_nd_reconstruct(hosvd, field_name="hosvd")
    np.testing.assert_allclose(
        arrow_fixed_shape_tensor_numpy(hosvd_recon),
        arrow_fixed_shape_tensor_numpy(tensor),
        rtol=1e-5,
        atol=1e-5,
    )

    hooi = arrow_tensor_hooi_nd(tensor, [1, 1, 1], max_iterations=10, tolerance=1e-8)
    assert hooi.core.ndim == 3

    tucker_core = arrow_tensor_tucker_project(tensor, hosvd)
    tucker_reexpanded = arrow_tensor_tucker_expand(hosvd, field_name="tucker")
    assert arrow_fixed_shape_tensor_numpy(tucker_core).ndim == 3
    np.testing.assert_allclose(
        arrow_fixed_shape_tensor_numpy(tucker_reexpanded),
        arrow_fixed_shape_tensor_numpy(tensor),
        rtol=1e-5,
        atol=1e-5,
    )

    tt = arrow_tensor_tt_svd(tensor)
    tt_left = arrow_tensor_tt_orthogonalize_left(tt)
    tt_right = arrow_tensor_tt_orthogonalize_right(tt)
    tt_norm = arrow_tensor_tt_norm(tt)
    tt_inner = arrow_tensor_tt_inner(tt, tt)
    assert np.isfinite(tt_norm)
    assert abs(np.sqrt(tt_inner) - tt_norm) < 1e-5
    assert len(tt_left.cores) == len(tt.cores)
    assert len(tt_right.cores) == len(tt.cores)

    tt_rounded = arrow_tensor_tt_round(tt)
    tt_added = arrow_tensor_tt_add(tt, tt)
    tt_hadamard = arrow_tensor_tt_hadamard(tt, tt)
    tt_hadamard_rounded = arrow_tensor_tt_hadamard_round(tt, tt)
    assert len(tt_added.cores) == len(tt.cores)
    assert len(tt_hadamard.cores) == len(tt.cores)
    assert len(tt_hadamard_rounded.cores) == len(tt.cores)

    tt_recon = arrow_tensor_tt_svd_reconstruct(tt_rounded, field_name="tt")
    np.testing.assert_allclose(
        arrow_fixed_shape_tensor_numpy(tt_recon),
        arrow_fixed_shape_tensor_numpy(tensor),
        rtol=1e-5,
        atol=1e-5,
    )


def test_arrow_tensor_complex_and_variable_shape_carriers():
    complex_fixed = arrow_fixed_shape_tensor_array(
        np.array(
            [
                [[1.0 + 1.0j, 0.0 + 2.0j], [2.0 + 0.0j, 0.0 + 1.0j]],
                [[3.0 + 4.0j, 0.0 + 1.0j], [1.0 + 0.0j, 1.0 + 0.0j]],
            ],
            dtype=np.complex128,
        )
    )
    np.testing.assert_allclose(
        arrow_fixed_shape_tensor_numpy(complex_fixed),
        np.array(
            [
                [[1.0 + 1.0j, 0.0 + 2.0j], [2.0 + 0.0j, 0.0 + 1.0j]],
                [[3.0 + 4.0j, 0.0 + 1.0j], [1.0 + 0.0j, 1.0 + 0.0j]],
            ],
            dtype=np.complex128,
        ),
    )

    complex_sum = arrow_tensor_sum_last_axis(complex_fixed)
    assert arrow_fixed_shape_tensor_numpy(complex_sum).shape == (2, 2)

    complex_norm = arrow_tensor_l2_norm_last_axis(complex_fixed)
    np.testing.assert_allclose(
        arrow_fixed_shape_tensor_numpy(complex_norm),
        np.linalg.norm(arrow_fixed_shape_tensor_numpy(complex_fixed), axis=-1),
        rtol=1e-10,
        atol=1e-10,
    )

    complex_normalized = arrow_tensor_normalize_last_axis(complex_fixed)
    np.testing.assert_allclose(
        np.linalg.norm(arrow_fixed_shape_tensor_numpy(complex_normalized), axis=-1),
        np.ones((2, 2), dtype=np.float64),
        rtol=1e-10,
        atol=1e-10,
    )

    complex_rhs = arrow_fixed_shape_tensor_array(
        np.array(
            [
                [[1.0 - 1.0j, 2.0 + 0.0j], [0.0 + 1.0j, 1.0 + 0.0j]],
                [[1.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 0.0 + 1.0j]],
            ],
            dtype=np.complex128,
        )
    )
    complex_dot = arrow_tensor_batched_dot_last_axis(complex_fixed, complex_rhs)
    assert arrow_fixed_shape_tensor_numpy(complex_dot).shape == (2, 2)

    complex_mm = arrow_tensor_batched_matmul_last_two(complex_fixed, complex_rhs)
    assert arrow_fixed_shape_tensor_numpy(complex_mm).shape == (2, 2, 2)

    complex_cube_vec = arrow_tensor_cube_matvec(
        complex_fixed,
        _complex_matrix_array([[1.0 + 0.0j, 0.0 + 0.0j], [1.0 + 0.0j, 1.0 + 0.0j]]),
    )
    assert _complex_matrix_numpy(complex_cube_vec).shape == (2, 2)

    complex_einsum = arrow_tensor_einsum("bij,bjk->bik", complex_fixed, complex_rhs)
    assert arrow_fixed_shape_tensor_numpy(complex_einsum).shape == (2, 2, 2)

    ragged = arrow_variable_shape_tensor_array(
        [
            np.array([[3.0, 4.0], [0.0, 5.0]], dtype=np.float64),
            np.array([[8.0, 15.0, 17.0]], dtype=np.float64),
        ]
    )
    ragged_rows = arrow_variable_shape_tensor_rows(ragged)
    assert ragged_rows[0].shape == (2, 2)
    ragged_sum = arrow_tensor_sum_last_axis(ragged)
    ragged_norm = arrow_tensor_l2_norm_last_axis(ragged)
    ragged_normalized = arrow_tensor_normalize_last_axis(ragged)
    np.testing.assert_allclose(
        arrow_variable_shape_tensor_rows(ragged_sum)[0],
        np.array([7.0, 5.0], dtype=np.float64),
    )
    np.testing.assert_allclose(
        arrow_variable_shape_tensor_rows(ragged_norm)[0],
        np.array([5.0, 5.0], dtype=np.float64),
    )
    np.testing.assert_allclose(
        np.linalg.norm(arrow_variable_shape_tensor_rows(ragged_normalized)[0], axis=-1),
        np.ones(2, dtype=np.float64),
        rtol=1e-10,
        atol=1e-10,
    )

    complex_ragged = arrow_variable_shape_tensor_array(
        [
            np.array([[1.0 + 1.0j, 0.0 + 2.0j], [2.0 + 0.0j, 0.0 + 1.0j]], dtype=np.complex128),
            np.array([[3.0 + 4.0j, 0.0 + 1.0j]], dtype=np.complex128),
        ]
    )
    complex_ragged_rhs = arrow_variable_shape_tensor_array(
        [
            np.array([[1.0 - 1.0j, 2.0 + 0.0j], [0.0 + 1.0j, 1.0 + 0.0j]], dtype=np.complex128),
            np.array([[1.0 + 0.0j, 1.0 + 0.0j]], dtype=np.complex128),
        ]
    )
    complex_ragged_rows = arrow_variable_shape_tensor_rows(complex_ragged)
    assert np.iscomplexobj(complex_ragged_rows[0])
    complex_ragged_sum = arrow_tensor_sum_last_axis(complex_ragged)
    complex_ragged_norm = arrow_tensor_l2_norm_last_axis(complex_ragged)
    complex_ragged_normalized = arrow_tensor_normalize_last_axis(complex_ragged)
    complex_ragged_dot = arrow_tensor_batched_dot_last_axis(complex_ragged, complex_ragged_rhs)
    assert np.iscomplexobj(arrow_variable_shape_tensor_rows(complex_ragged_sum)[0])
    np.testing.assert_allclose(
        arrow_variable_shape_tensor_rows(complex_ragged_norm)[0],
        np.linalg.norm(complex_ragged_rows[0], axis=-1),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.linalg.norm(arrow_variable_shape_tensor_rows(complex_ragged_normalized)[0], axis=-1),
        np.ones(2, dtype=np.float64),
        rtol=1e-10,
        atol=1e-10,
    )
    assert np.iscomplexobj(arrow_variable_shape_tensor_rows(complex_ragged_dot)[0])
