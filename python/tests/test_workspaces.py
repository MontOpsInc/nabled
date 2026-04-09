"""Tests for reusable Python workspace objects."""

import numpy as np
import pynabled
import pytest


def test_pairwise_cosine_workspace_reuses_outputs_and_resizes():
    workspace = pynabled.PairwiseCosineWorkspace(np.float32)
    left = np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    right = np.array([[0.0, 1.0], [1.0, 1.0]], dtype=np.float32)

    similarity_out = np.empty((2, 2), dtype=np.float32, order="F")
    returned_similarity = pynabled.pairwise_cosine_similarity(
        left,
        right,
        out=similarity_out,
        workspace=workspace,
    )
    assert returned_similarity is similarity_out
    np.testing.assert_allclose(
        similarity_out,
        pynabled.pairwise_cosine_similarity(left, right),
        rtol=1e-6,
        atol=1e-6,
    )

    resized_right = np.array([[1.0, 0.0]], dtype=np.float32)
    distance_out = np.empty((2, 1), dtype=np.float32)
    returned_distance = workspace.distance(left, resized_right, out=distance_out)
    assert returned_distance is distance_out
    np.testing.assert_allclose(
        distance_out,
        pynabled.pairwise_cosine_distance(left, resized_right),
        rtol=1e-6,
        atol=1e-6,
    )


def test_pairwise_cosine_workspace_rejects_dtype_mismatch():
    workspace = pynabled.PairwiseCosineWorkspace(np.float32)
    left = np.array([[1.0, 0.0]], dtype=np.float64)
    right = np.array([[1.0, 0.0]], dtype=np.float64)

    with pytest.raises(TypeError, match="matching dtype"):
        pynabled.pairwise_cosine_similarity(left, right, workspace=workspace)


def test_matrix_function_workspace_reuses_real_and_complex_paths():
    real_workspace = pynabled.MatrixFunctionWorkspace(np.float32)
    real_matrix = np.array([[1.05, 0.02], [0.0, 0.97]], dtype=np.float32)

    exp_out = np.empty_like(real_matrix, order="F")
    returned_exp = pynabled.matrix_exp(
        real_matrix,
        max_terms=32,
        tolerance=1e-5,
        out=exp_out,
        workspace=real_workspace,
    )
    assert returned_exp is exp_out
    np.testing.assert_allclose(
        exp_out,
        pynabled.matrix_exp(real_matrix, max_terms=32, tolerance=1e-5),
        rtol=1e-5,
        atol=1e-5,
    )

    log_out = np.empty_like(real_matrix)
    returned_log = real_workspace.log_taylor(real_matrix, max_terms=32, tolerance=1e-5, out=log_out)
    assert returned_log is log_out
    np.testing.assert_allclose(
        log_out,
        pynabled.matrix_log_taylor(real_matrix, max_terms=32, tolerance=1e-5),
        rtol=1e-5,
        atol=1e-5,
    )

    complex_workspace = pynabled.MatrixFunctionWorkspace(np.complex128)
    complex_matrix = np.array(
        [[1.0 + 0.1j, 0.2 - 0.1j], [0.0 + 0.0j, 1.1 + 0.3j]],
        dtype=np.complex128,
    )
    complex_exp_out = np.empty_like(complex_matrix, order="F")
    returned_complex_exp = complex_workspace.exp(
        complex_matrix,
        max_terms=32,
        tolerance=1e-12,
        out=complex_exp_out,
    )
    assert returned_complex_exp is complex_exp_out
    np.testing.assert_allclose(
        complex_exp_out,
        pynabled.matrix_exp(complex_matrix, max_terms=32, tolerance=1e-12),
        rtol=1e-12,
        atol=1e-12,
    )

    with pytest.raises(TypeError, match="matrix_log_taylor workspace must use dtype float32 or float64"):
        complex_workspace.log_taylor(real_matrix.astype(np.float64))


def test_sylvester_workspace_reuses_real_and_complex_paths():
    real_workspace = pynabled.SylvesterWorkspace(np.float64)
    a = np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float64)
    b = np.array([[1.0, 0.0], [0.0, 4.0]], dtype=np.float64)
    c = np.array([[3.0, 2.0], [1.0, 5.0]], dtype=np.float64)

    solve_out = np.empty((2, 2), dtype=np.float64, order="F")
    returned_solve = pynabled.sylvester_solve(a, b, c, out=solve_out, workspace=real_workspace)
    assert returned_solve is solve_out
    np.testing.assert_allclose(solve_out, pynabled.sylvester_solve(a, b, c), rtol=1e-12, atol=1e-12)

    q = np.array([[2.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    lyapunov_out = np.empty((2, 2), dtype=np.float64)
    returned_lyapunov = real_workspace.lyapunov(a, q, out=lyapunov_out)
    assert returned_lyapunov is lyapunov_out
    np.testing.assert_allclose(
        lyapunov_out,
        pynabled.lyapunov_solve(a, q),
        rtol=1e-12,
        atol=1e-12,
    )

    complex_workspace = pynabled.SylvesterWorkspace(np.complex128)
    a_complex = np.array([[1.0 + 1.0j, 0.0], [0.0, 2.0 + 0.5j]], dtype=np.complex128)
    b_complex = np.array([[0.5 - 0.25j, 0.0], [0.0, 1.5 + 0.75j]], dtype=np.complex128)
    c_complex = np.array([[1.0 + 0.5j, 0.2 - 0.1j], [0.0 + 0.3j, 2.0 - 0.25j]], dtype=np.complex128)

    complex_out = np.empty((2, 2), dtype=np.complex128)
    returned_complex = complex_workspace.solve(a_complex, b_complex, c_complex, out=complex_out)
    assert returned_complex is complex_out
    np.testing.assert_allclose(
        complex_out,
        pynabled.sylvester_solve(a_complex, b_complex, c_complex),
        rtol=1e-12,
        atol=1e-12,
    )


def test_schur_workspace_reuses_real_and_complex_paths():
    real_workspace = pynabled.SchurWorkspace(np.float32)
    real_matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    real_out = pynabled.SchurResult(
        q=np.empty((2, 2), dtype=np.float32, order="F"),
        t=np.empty((2, 2), dtype=np.float32, order="F"),
    )

    returned_real = pynabled.schur_compute(real_matrix, out=real_out, workspace=real_workspace)

    assert returned_real is real_out
    np.testing.assert_allclose(real_out.q @ real_out.t @ real_out.q.T, real_matrix, rtol=1e-4, atol=1e-5)

    complex_workspace = pynabled.SchurWorkspace(np.complex128)
    complex_matrix = np.array(
        [[3.0 + 1.0j, 1.0 - 0.5j], [0.0 + 1.0j, 2.0 - 0.25j]],
        dtype=np.complex128,
    )
    complex_out = pynabled.SchurResult(
        q=np.empty((2, 2), dtype=np.complex128),
        t=np.empty((2, 2), dtype=np.complex128),
    )

    returned_complex = complex_workspace.compute(complex_matrix, out=complex_out)

    assert returned_complex is complex_out
    np.testing.assert_allclose(
        complex_out.q @ complex_out.t @ complex_out.q.conj().T,
        complex_matrix,
        rtol=1e-10,
        atol=1e-12,
    )


def test_workspace_kwargs_validate_expected_types():
    wrong_workspace = pynabled.SylvesterWorkspace(np.float64)
    matrix = np.eye(2, dtype=np.float64)

    with pytest.raises(TypeError, match="workspace must be MatrixFunctionWorkspace or None"):
        pynabled.matrix_sign(matrix, workspace=wrong_workspace)

    with pytest.raises(TypeError, match="workspace must be SchurWorkspace or None"):
        pynabled.schur_compute(matrix, workspace=wrong_workspace)
