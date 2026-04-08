"""Tests for matrix bindings."""

import numpy as np
import pynabled
import pytest


def test_matvec():
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64, order="C")
    x = np.array([1.0, 1.0], dtype=np.float64)
    y = pynabled.matvec(a, x)
    np.testing.assert_allclose(y, [3.0, 7.0])


def test_matmat():
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    b = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    c = pynabled.matmat(a, b)
    np.testing.assert_allclose(c, a)


def test_dot():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    b = np.array([4.0, 5.0, 6.0], dtype=np.float64)
    d = pynabled.dot(a, b)
    assert d == 32.0


def test_eigen_symmetric():
    a = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float64)
    result = pynabled.eigen_symmetric(a)
    assert result.eigenvalues.shape == (2,)
    assert result.eigenvectors.shape == (2, 2)
    np.testing.assert_allclose(
        a @ result.eigenvectors,
        result.eigenvectors @ np.diag(result.eigenvalues),
        rtol=1e-10,
    )


def test_schur():
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    result = pynabled.schur_compute(a)
    np.testing.assert_allclose(result.q @ result.t @ result.q.T, a, rtol=1e-10)


def test_gram_schmidt():
    a = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    q = pynabled.gram_schmidt(a)
    assert q.shape == a.shape
    # Columns should be orthonormal
    qtq = q.T @ q
    np.testing.assert_allclose(qtq, np.eye(2), rtol=1e-10, atol=1e-14)


def test_gram_schmidt_classic():
    a = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    q = pynabled.gram_schmidt_classic(a)
    assert q.shape == a.shape
    qtq = q.T @ q
    np.testing.assert_allclose(qtq, np.eye(2), rtol=1e-10, atol=1e-14)


def test_dense_kernels_accept_non_contiguous_inputs():
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64).T
    vector = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)[::2]
    assert not matrix.flags["C_CONTIGUOUS"]
    assert not vector.flags["C_CONTIGUOUS"]
    np.testing.assert_allclose(pynabled.matvec(matrix, vector), matrix @ vector, rtol=1e-10)
    np.testing.assert_allclose(pynabled.matmat(matrix, matrix.T), matrix @ matrix.T, rtol=1e-10)


def test_dense_kernels_accept_float32():
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    vector = np.array([1.0, 1.0], dtype=np.float32)
    left = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    right = np.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[2.0, 0.0], [0.0, 2.0]],
        ],
        dtype=np.float32,
    )

    matvec = pynabled.matvec(matrix, vector)
    matmat = pynabled.matmat(matrix, matrix)
    batched_row = pynabled.batched_row_matvec(
        matrix, np.vstack([vector, 2.0 * vector]).astype(np.float32)
    )
    batched = pynabled.batched_matmat(left, right)

    assert matvec.dtype == np.float32
    assert matmat.dtype == np.float32
    assert batched_row.dtype == np.float32
    assert batched.dtype == np.float32
    np.testing.assert_allclose(matvec, matrix @ vector, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(matmat, matrix @ matrix, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(
        batched_row,
        np.vstack([vector @ matrix.T, (2.0 * vector) @ matrix.T]).astype(np.float32),
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(batched, left @ right, rtol=1e-5, atol=1e-6)


def test_dense_kernels_accept_complex128():
    matrix = np.array([[1.0 + 1.0j, 0.0 - 1.0j], [2.0 + 0.0j, 1.0 + 2.0j]], dtype=np.complex128)
    vector = np.array([1.0 + 0.0j, 0.5 - 0.5j], dtype=np.complex128)
    left = np.array([[1.0 + 0.0j, 2.0 - 1.0j], [0.0 + 1.0j, 1.0 + 0.0j]], dtype=np.complex128)
    right = np.array([[1.0 + 1.0j, 0.0 + 1.0j], [2.0 + 0.0j, 1.0 - 1.0j]], dtype=np.complex128)

    matvec = pynabled.matvec(matrix, vector)
    matmat = pynabled.matmat(left, right)

    assert matvec.dtype == np.complex128
    assert matmat.dtype == np.complex128
    np.testing.assert_allclose(matvec, matrix @ vector, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(matmat, left @ right, rtol=1e-12, atol=1e-12)


def test_batched_matrix_broadcast_kernels_accept_float32():
    left_batches = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    right = np.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]],
        dtype=np.float32,
    )
    left = np.array(
        [[1.0, 2.0, 0.0], [0.0, 1.0, 1.0]],
        dtype=np.float32,
    )
    right_batches = np.array(
        [
            [[1.0, 0.0], [0.0, 1.0], [2.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0], [1.0, -1.0]],
        ],
        dtype=np.float32,
    )

    broadcast_right = pynabled.batched_matmat_broadcast_right(left_batches, right)
    broadcast_left = pynabled.batched_matmat_broadcast_left(left, right_batches)

    assert broadcast_right.dtype == np.float32
    assert broadcast_left.dtype == np.float32
    np.testing.assert_allclose(broadcast_right, left_batches @ right, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(broadcast_left, left @ right_batches, rtol=1e-5, atol=1e-6)


def test_dense_kernels_reject_mixed_real_dtypes():
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    vector = np.array([1.0, 1.0], dtype=np.float64)
    with pytest.raises(TypeError, match="matching dtype"):
        pynabled.matvec(matrix, vector)


def test_eigen_symmetric_and_schur_accept_float32():
    symmetric = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float32)
    general = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    eigen = pynabled.eigen_symmetric(symmetric)
    schur = pynabled.schur_compute(general)

    assert eigen.eigenvalues.dtype == np.float32
    assert eigen.eigenvectors.dtype == np.float32
    assert schur.q.dtype == np.float32
    assert schur.t.dtype == np.float32
    np.testing.assert_allclose(
        symmetric @ eigen.eigenvectors,
        eigen.eigenvectors @ np.diag(eigen.eigenvalues),
        rtol=1e-4,
        atol=1e-5,
    )
    np.testing.assert_allclose(schur.q @ schur.t @ schur.q.T, general, rtol=1e-4, atol=1e-5)


def test_schur_accepts_complex128():
    matrix = np.array([[3.0 + 1.0j, 1.0 - 0.5j], [0.0 + 1.0j, 2.0 - 0.25j]], dtype=np.complex128)
    result = pynabled.schur_compute(matrix)

    assert result.q.dtype == np.complex128
    assert result.t.dtype == np.complex128
    np.testing.assert_allclose(
        result.q @ result.t @ result.q.conj().T,
        matrix,
        rtol=1e-10,
        atol=1e-12,
    )


def test_gram_schmidt_accepts_float32_and_complex128():
    real = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    complex_matrix = np.array(
        [[1.0 + 0.0j, 1.0 - 1.0j], [1.0 + 1.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.5j]],
        dtype=np.complex128,
    )

    q_real = pynabled.gram_schmidt(real)
    q_classic = pynabled.gram_schmidt_classic(real)
    q_complex = pynabled.gram_schmidt(complex_matrix)

    assert q_real.dtype == np.float32
    assert q_classic.dtype == np.float32
    assert q_complex.dtype == np.complex128
    np.testing.assert_allclose(q_real.T @ q_real, np.eye(2, dtype=np.float32), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(
        q_classic.T @ q_classic,
        np.eye(2, dtype=np.float32),
        rtol=1e-4,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        q_complex.conj().T @ q_complex,
        np.eye(2, dtype=np.complex128),
        rtol=1e-10,
        atol=1e-12,
    )


def test_matrix_out_reuses_dense_output_buffers():
    matrix = np.asfortranarray(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    vector = np.array([1.0, 1.0], dtype=np.float32)
    left_batches = np.asfortranarray(np.arange(12, dtype=np.float32).reshape(2, 2, 3))
    right = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]], dtype=np.float32)

    matvec_out = np.empty(2, dtype=np.float32)
    returned_matvec = pynabled.matvec(matrix, vector, out=matvec_out)
    assert returned_matvec is matvec_out
    np.testing.assert_allclose(matvec_out, matrix @ vector, rtol=1e-5, atol=1e-6)

    matmat_out = np.empty((2, 2), dtype=np.float32, order="F")
    returned_matmat = pynabled.matmat(matrix, matrix, out=matmat_out)
    assert returned_matmat is matmat_out
    np.testing.assert_allclose(matmat_out, matrix @ matrix, rtol=1e-5, atol=1e-6)

    broadcast_out = np.empty((2, 2, 2), dtype=np.float32, order="F")
    returned_broadcast = pynabled.batched_matmat_broadcast_right(
        left_batches,
        right,
        out=broadcast_out,
    )
    assert returned_broadcast is broadcast_out
    np.testing.assert_allclose(broadcast_out, left_batches @ right, rtol=1e-5, atol=1e-6)


def test_matrix_out_reuses_complex_buffers_and_rejects_aliasing():
    matrix = np.array([[1.0 + 1.0j, 0.0 - 1.0j], [2.0 + 0.0j, 1.0 + 2.0j]], dtype=np.complex128)
    vector = np.array([1.0 + 0.0j, 0.5 - 0.5j], dtype=np.complex128)
    right = np.array([[1.0 + 1.0j, 0.0 + 1.0j], [2.0 + 0.0j, 1.0 - 1.0j]], dtype=np.complex128)

    matvec_out = np.empty(2, dtype=np.complex128)
    returned_matvec = pynabled.matvec(matrix, vector, out=matvec_out)
    assert returned_matvec is matvec_out
    np.testing.assert_allclose(matvec_out, matrix @ vector, rtol=1e-12, atol=1e-12)

    matmat_out = np.empty((2, 2), dtype=np.complex128, order="F")
    returned_matmat = pynabled.matmat(matrix, right, out=matmat_out)
    assert returned_matmat is matmat_out
    np.testing.assert_allclose(matmat_out, matrix @ right, rtol=1e-12, atol=1e-12)

    aliased = np.ones((2, 2), dtype=np.float64)
    with pytest.raises(TypeError, match="already borrowed"):
        pynabled.matmat(aliased, aliased, out=aliased)
