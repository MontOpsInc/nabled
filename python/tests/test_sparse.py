"""Tests for sparse CSR bindings."""

import numpy as np
import pynabled
import pytest


def _make_csr_diagonal(n, dtype=np.float64, index_dtype=np.int64):
    """Create CSR for n x n diagonal matrix with 1,2,...,n on diagonal."""
    indptr = np.arange(n + 1, dtype=index_dtype)
    indices = np.arange(n, dtype=index_dtype)
    data = np.arange(1, n + 1, dtype=dtype)
    return n, n, indptr, indices, data


class _FakeSciPyCsr:
    format = "csr"

    def __init__(self, shape, indptr, indices, data):
        self.shape = shape
        self.indptr = indptr
        self.indices = indices
        self.data = data


class _FakeSciPyCsc:
    format = "csc"

    def __init__(self, csr):
        self._csr = csr

    def tocsr(self, copy=False):
        if not copy:
            return self._csr
        return _FakeSciPyCsr(
            self._csr.shape,
            self._csr.indptr.copy(),
            self._csr.indices.copy(),
            self._csr.data.copy(),
        )


def test_sparse_carrier_and_matvec():
    matrix = pynabled.CsrMatrix.from_components(*_make_csr_diagonal(3))
    v = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    expected = np.array([1.0, 4.0, 9.0], dtype=np.float64)
    np.testing.assert_allclose(pynabled.sparse_matvec(matrix, v), expected, rtol=1e-14)
    np.testing.assert_allclose(matrix.matvec(v), expected, rtol=1e-14)
    np.testing.assert_allclose(matrix @ v, expected, rtol=1e-14)
    assert matrix.shape == (3, 3)
    assert matrix.nnz == 3
    assert matrix.index_dtype == np.dtype(np.int64)


def test_sparse_matmat_dense_and_transpose():
    matrix = pynabled.CsrMatrix.from_components(*_make_csr_diagonal(3))
    dense = np.arange(6, dtype=np.float64).reshape(3, 2)
    expected = np.diag([1.0, 2.0, 3.0]) @ dense
    np.testing.assert_allclose(pynabled.sparse_matmat_dense(matrix, dense), expected, rtol=1e-14)
    np.testing.assert_allclose(matrix @ dense, expected, rtol=1e-14)

    transposed = pynabled.sparse_transpose(matrix)
    assert isinstance(transposed, pynabled.CsrMatrix)
    np.testing.assert_array_equal(transposed.indptr, matrix.indptr)
    np.testing.assert_array_equal(transposed.indices, matrix.indices)
    np.testing.assert_array_equal(transposed.data, matrix.data)
    np.testing.assert_array_equal(matrix.T.indptr, matrix.indptr)


def test_sparse_jacobi_solve():
    matrix = pynabled.CsrMatrix.from_components(*_make_csr_diagonal(3))
    rhs = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    x = pynabled.sparse_jacobi_solve(matrix, rhs)
    np.testing.assert_allclose(x, [1.0, 1.0, 1.0], rtol=1e-10)
    np.testing.assert_allclose(matrix.jacobi_solve(rhs), x, rtol=1e-10)


def test_sparse_pcg_solve():
    matrix = pynabled.CsrMatrix.from_components(*_make_csr_diagonal(3))
    rhs = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    x = pynabled.sparse_pcg_solve(matrix, rhs)
    np.testing.assert_allclose(x, [1.0, 1.0, 1.0], rtol=1e-10)
    np.testing.assert_allclose(matrix.pcg_solve(rhs), x, rtol=1e-10)


def test_sparse_accepts_scipy_compatible_objects():
    nrows, ncols, indptr, indices, data = _make_csr_diagonal(3)
    csr = _FakeSciPyCsr((nrows, ncols), indptr, indices, data)
    csc = _FakeSciPyCsc(csr)
    v = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    expected = np.array([1.0, 4.0, 9.0], dtype=np.float64)
    np.testing.assert_allclose(pynabled.sparse_matvec(csc, v), expected, rtol=1e-14)
    matrix = pynabled.CsrMatrix.from_scipy(csc)
    assert isinstance(matrix, pynabled.CsrMatrix)
    np.testing.assert_array_equal(matrix.indptr, indptr)
    np.testing.assert_array_equal(matrix.indices, indices)
    np.testing.assert_array_equal(matrix.data, data)


def test_sparse_preserves_int32_index_dtype():
    matrix = pynabled.CsrMatrix.from_components(*_make_csr_diagonal(3, index_dtype=np.int32))
    vector = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    transposed = pynabled.sparse_transpose(matrix)

    assert matrix.index_dtype == np.dtype(np.int32)
    assert transposed.index_dtype == np.dtype(np.int32)
    np.testing.assert_allclose(matrix.matvec(vector), np.array([1.0, 4.0, 9.0], dtype=np.float64))


def test_sparse_carrier_accepts_float32():
    matrix = pynabled.CsrMatrix.from_components(*_make_csr_diagonal(3, dtype=np.float32))
    vector = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    dense = np.arange(6, dtype=np.float32).reshape(3, 2)
    rhs = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    matvec = pynabled.sparse_matvec(matrix, vector)
    matmat = pynabled.sparse_matmat_dense(matrix, dense)
    transpose = pynabled.sparse_transpose(matrix)
    jacobi = pynabled.sparse_jacobi_solve(matrix, rhs)
    pcg = pynabled.sparse_pcg_solve(matrix, rhs)

    assert matrix.data.dtype == np.float32
    assert matvec.dtype == np.float32
    assert matmat.dtype == np.float32
    assert transpose.data.dtype == np.float32
    assert jacobi.dtype == np.float32
    assert pcg.dtype == np.float32
    np.testing.assert_allclose(matvec, np.array([1.0, 4.0, 9.0], dtype=np.float32), rtol=1e-5)
    np.testing.assert_allclose(
        matmat,
        np.diag([1.0, 2.0, 3.0]).astype(np.float32) @ dense,
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(jacobi, np.ones(3, dtype=np.float32), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(pcg, np.ones(3, dtype=np.float32), rtol=1e-4, atol=1e-5)


def test_sparse_rejects_mixed_real_dtypes():
    matrix = pynabled.CsrMatrix.from_components(*_make_csr_diagonal(3, dtype=np.float32))
    with pytest.raises(TypeError, match="match sparse matrix data"):
        pynabled.sparse_matvec(matrix, np.array([1.0, 2.0, 3.0], dtype=np.float64))


def test_sparse_rejects_mixed_index_dtypes_without_explicit_normalization():
    nrows, ncols, indptr, indices, data = _make_csr_diagonal(3)
    with pytest.raises(TypeError, match="indptr and indices must share dtype"):
        pynabled.CsrMatrix.from_components(
            nrows,
            ncols,
            indptr.astype(np.int32),
            indices.astype(np.int64),
            data,
        )


def test_sparse_explicit_index_dtype_normalizes_components():
    nrows, ncols, indptr, indices, data = _make_csr_diagonal(3, index_dtype=np.int64)
    matrix = pynabled.CsrMatrix.from_components(
        (nrows, ncols),
        indptr,
        indices,
        data,
        index_dtype=np.int32,
    )
    assert matrix.index_dtype == np.dtype(np.int32)
    np.testing.assert_array_equal(matrix.indptr, indptr.astype(np.int32))
    np.testing.assert_array_equal(matrix.indices, indices.astype(np.int32))


def test_sparse_explicit_dtype_and_index_dtype_helpers():
    matrix = pynabled.CsrMatrix.from_components(*_make_csr_diagonal(3))

    matrix_f32 = matrix.astype(np.float32)
    matrix_i32 = matrix.with_index_dtype(np.int32)

    assert matrix_f32.dtype == np.dtype(np.float32)
    assert matrix_f32.index_dtype == np.dtype(np.int64)
    assert matrix_i32.dtype == np.dtype(np.float64)
    assert matrix_i32.index_dtype == np.dtype(np.int32)


def test_sparse_carrier_rejects_non_contiguous_csr_buffers_without_copy():
    indptr = np.array([0, 99, 1, 99, 2, 99, 3, 99], dtype=np.int64)[::2]
    indices = np.array([0, 99, 1, 99, 2, 99], dtype=np.int64)[::2]
    data = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    with pytest.raises(ValueError, match="C-contiguous"):
        pynabled.CsrMatrix((3, 3), indptr, indices, data)

    matrix = pynabled.CsrMatrix((3, 3), indptr, indices, data, copy=True)
    assert matrix.index_dtype == np.dtype(np.int64)
