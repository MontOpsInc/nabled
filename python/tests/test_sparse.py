"""Tests for sparse CSR bindings."""

import numpy as np
import pynabled


def _make_csr_diagonal(n):
    """Create CSR for n x n diagonal matrix with 1,2,...,n on diagonal."""
    indptr = np.arange(n + 1, dtype=np.int64)
    indices = np.arange(n, dtype=np.int64)
    data = np.arange(1, n + 1, dtype=np.float64)
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
