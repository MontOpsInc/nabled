"""Tests for QR decomposition bindings."""

import numpy as np
import pynabled
import pytest


def test_qr_decompose():
    a = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64, order="C")
    result = pynabled.qr_decompose(a)
    assert result.q.shape == (3, 2)
    assert result.r.shape == (2, 2)
    assert result.rank == 2
    np.testing.assert_allclose(result.q @ result.r, a, rtol=1e-10)
    np.testing.assert_allclose(result.q.T @ result.q, np.eye(2), rtol=1e-10, atol=1e-14)


def test_qr_solve_least_squares():
    np.random.seed(42)
    a = np.random.randn(5, 3).astype(np.float64)
    x_true = np.array([1.0, -2.0, 0.5], dtype=np.float64)
    b = a @ x_true
    x = pynabled.qr_solve_least_squares(a, b, rank_tolerance=1.0e-12, max_iterations=128)
    np.testing.assert_allclose(a @ x, b, rtol=1e-10)
    np.testing.assert_allclose(x, x_true, rtol=1e-10)


def test_qr_solve_least_squares_supports_out_and_factor_reuse():
    np.random.seed(7)
    a = np.random.randn(6, 3).astype(np.float64)
    x_true = np.array([0.25, -1.5, 2.0], dtype=np.float64)
    b = a @ x_true

    direct_out = np.empty(3, dtype=np.float64, order="F")
    direct = pynabled.qr_solve_least_squares(a, b, out=direct_out)
    result = pynabled.qr_decompose_reduced(a)
    factor_out = np.empty(3, dtype=np.float64, order="F")
    factor = pynabled.qr_solve_least_squares(result, b, out=factor_out, rank_tolerance=1.0e-12)

    assert direct is direct_out
    assert factor is factor_out
    np.testing.assert_allclose(direct_out, x_true, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(factor_out, x_true, rtol=1e-10, atol=1e-12)


def test_qr_solve_least_squares_reuses_borrowed_fortran_factor_arrays():
    a = np.array(
        [
            [1.0, 100.0, 2.0],
            [0.0, 1.0, 4.0],
            [0.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    x_true = np.array([0.5, -1.25, 2.0], dtype=np.float64)
    rhs = a @ x_true
    result = pynabled.qr_decompose_pivoted(a, rank_tolerance=1.0e-12)
    factor = pynabled.QrResult(
        q=np.asfortranarray(result.q),
        r=np.asfortranarray(result.r),
        rank=result.rank,
        p=np.asfortranarray(result.p),
    )
    out = np.empty_like(x_true, order="F")

    returned = pynabled.qr_solve_least_squares(factor, rhs, out=out, rank_tolerance=1.0e-12)

    assert factor.q.flags.f_contiguous
    assert factor.r.flags.f_contiguous
    assert factor.p is not None and factor.p.flags.f_contiguous
    assert returned is out
    np.testing.assert_allclose(out, x_true, rtol=1e-10, atol=1e-12)


def test_qr_solve_least_squares_qr_result_rejects_underdetermined_reuse():
    a = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 3.0]], dtype=np.float64)
    b = np.array([1.0, 2.0], dtype=np.float64)
    result = pynabled.qr_decompose_reduced(a)

    with pytest.raises(ValueError, match="underdetermined"):
        pynabled.qr_solve_least_squares(result, b)


def test_qr_solve_least_squares_qr_result_rejects_max_iterations_override():
    a = np.array([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]], dtype=np.float64)
    b = np.array([0.5, -1.0, 3.0], dtype=np.float64)
    result = pynabled.qr_decompose_reduced(a)

    with pytest.raises(TypeError, match="max_iterations="):
        pynabled.qr_solve_least_squares(result, b, max_iterations=32)


def test_qr_accepts_float32():
    a = np.array([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]], dtype=np.float32)
    x_true = np.array([0.5, -1.25], dtype=np.float32)
    b = a @ x_true

    result = pynabled.qr_decompose(a)
    x = pynabled.qr_solve_least_squares(a, b)

    assert result.q.dtype == np.float32
    assert result.r.dtype == np.float32
    assert x.dtype == np.float32
    np.testing.assert_allclose(result.q @ result.r, a, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(a @ x, b, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(x, x_true, rtol=5e-4, atol=2e-5)


def test_qr_decompose_accepts_complex128():
    a = np.array(
        [[1.0 + 1.0j, 2.0 - 1.0j], [3.0 + 0.5j, 4.0 + 0.25j], [5.0 - 0.5j, 6.0 + 1.0j]],
        dtype=np.complex128,
    )

    result = pynabled.qr_decompose(a)

    assert result.q.dtype == np.complex128
    assert result.r.dtype == np.complex128
    assert result.rank == 2
    np.testing.assert_allclose(result.q @ result.r, a, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        result.q.conj().T @ result.q,
        np.eye(2, dtype=np.complex128),
        rtol=1e-10,
        atol=1e-12,
    )


def test_qr_decompose_reduced_and_condition_number():
    a = np.array([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]], dtype=np.float32)

    result = pynabled.qr_decompose_reduced(a, rank_tolerance=1.0e-5, max_iterations=64)
    reconstructed = pynabled.qr_reconstruct_matrix(result)
    condition = pynabled.qr_condition_number(result)

    assert result.q.shape == (3, 2)
    assert result.r.shape == (2, 2)
    assert result.p is None
    np.testing.assert_allclose(reconstructed, a, rtol=1e-4, atol=1e-5)

    diagonal = np.abs(np.diag(result.r))
    tolerance = max(1.0e-12, np.finfo(diagonal.dtype).eps)
    expected = float(diagonal.max() / diagonal[diagonal > tolerance].min())
    assert np.isclose(condition, expected)


def test_qr_reconstruct_matrix_supports_out():
    a = np.array([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]], dtype=np.float32)
    result = pynabled.qr_decompose_reduced(a)
    out = np.empty_like(a, order="F")

    returned = pynabled.qr_reconstruct_matrix(result, out=out)

    assert returned is out
    np.testing.assert_allclose(out, a, rtol=1e-4, atol=1e-5)


def test_qr_decompose_pivoted_reconstructs_original_matrix():
    a = np.array(
        [[1.0, 100.0, 2.0], [0.0, 1.0, 4.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    result = pynabled.qr_decompose_pivoted(a, rank_tolerance=1.0e-12)

    assert result.p is not None
    np.testing.assert_allclose(pynabled.qr_reconstruct_matrix(result), a, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(result.q @ result.r, a @ result.p, rtol=1e-10, atol=1e-12)


def test_qr_decompose_pivoted_accepts_complex128():
    a = np.array(
        [[1.0 + 1.0j, 0.0, 2.0 - 1.0j], [0.0, 3.0 - 0.5j, 4.0], [5.0, 0.0, 6.0 + 0.25j]],
        dtype=np.complex128,
    )

    result = pynabled.qr_decompose_pivoted(a)

    assert result.p is not None
    assert result.p.dtype == np.complex128
    np.testing.assert_allclose(pynabled.qr_reconstruct_matrix(result), a, rtol=1e-10, atol=1e-12)


def test_qr_reconstruct_matrix_pivoted_supports_out():
    a = np.array(
        [[1.0 + 1.0j, 0.0, 2.0 - 1.0j], [0.0, 3.0 - 0.5j, 4.0], [5.0, 0.0, 6.0 + 0.25j]],
        dtype=np.complex128,
    )

    result = pynabled.qr_decompose_pivoted(a)
    out = np.empty_like(a, order="F")

    returned = pynabled.qr_reconstruct_matrix(result, out=out)

    assert returned is out
    np.testing.assert_allclose(out, a, rtol=1e-10, atol=1e-12)
