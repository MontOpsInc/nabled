"""Tests for sparse CSR bindings."""

import re
import sys
import types

import numpy as np
import pynabled
import pytest


def _make_csr_diagonal(n, dtype=np.float64, index_dtype=np.int64):
    """Create CSR for n x n diagonal matrix with 1,2,...,n on diagonal."""
    indptr = np.arange(n + 1, dtype=index_dtype)
    indices = np.arange(n, dtype=index_dtype)
    data = np.arange(1, n + 1, dtype=dtype)
    return n, n, indptr, indices, data


def _csr_from_dense(dense, *, index_dtype=np.int64):
    dense = np.asarray(dense)
    nrows, ncols = dense.shape
    indptr = [0]
    indices = []
    data = []
    for row in range(nrows):
        for col in range(ncols):
            value = dense[row, col]
            if value != 0:
                indices.append(col)
                data.append(value)
        indptr.append(len(indices))
    return (
        nrows,
        ncols,
        np.asarray(indptr, dtype=index_dtype),
        np.asarray(indices, dtype=index_dtype),
        np.asarray(data, dtype=dense.dtype),
    )


def _dense_from_csr(matrix):
    dense = np.zeros(matrix.shape, dtype=matrix.dtype)
    for row in range(matrix.nrows):
        start = int(matrix.indptr[row])
        end = int(matrix.indptr[row + 1])
        dense[row, matrix.indices[start:end]] = matrix.data[start:end]
    return dense


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


class _FakeSciPyCanonicalCsc:
    format = "csc"

    def __init__(self, shape, indptr, indices, data):
        self.shape = shape
        self.indptr = indptr
        self.indices = indices
        self.data = data


class _FakeSciPyCoo:
    format = "coo"

    def __init__(self, shape, row, col, data):
        self.shape = shape
        self.row = row
        self.col = col
        self.data = data


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


def test_sparse_direct_iterative_solver_surface():
    spd_dense = np.array(
        [
            [4.0, 1.0, 0.0],
            [1.0, 3.0, 1.0],
            [0.0, 1.0, 2.0],
        ],
        dtype=np.float64,
    )
    spd = pynabled.CsrMatrix.from_components(*_csr_from_dense(spd_dense))
    rhs = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    expected_spd = np.linalg.solve(spd_dense, rhs)

    np.testing.assert_allclose(
        pynabled.sparse_gauss_seidel_solve(spd, rhs, tolerance=1e-10, max_iterations=5000),
        expected_spd,
        rtol=1e-8,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        spd.gauss_seidel_solve(rhs, tolerance=1e-10, max_iterations=5000),
        expected_spd,
        rtol=1e-8,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        pynabled.sparse_conjugate_gradient_solve(spd, rhs, tolerance=1e-10, max_iterations=5000),
        expected_spd,
        rtol=1e-8,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        spd.conjugate_gradient_solve(rhs, tolerance=1e-10, max_iterations=5000),
        expected_spd,
        rtol=1e-8,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        pynabled.sparse_pcg_ic0_solve(spd, rhs, tolerance=1e-10, max_iterations=5000),
        expected_spd,
        rtol=1e-8,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        spd.pcg_ic0_solve(rhs, tolerance=1e-10, max_iterations=5000),
        expected_spd,
        rtol=1e-8,
        atol=1e-8,
    )

    nonsymmetric_dense = np.array(
        [
            [4.0, 1.0, 0.0],
            [2.0, 3.0, 1.0],
            [0.0, 1.0, 2.0],
        ],
        dtype=np.float64,
    )
    nonsymmetric = pynabled.CsrMatrix.from_components(*_csr_from_dense(nonsymmetric_dense))
    expected_nonsymmetric = np.linalg.solve(nonsymmetric_dense, rhs)

    np.testing.assert_allclose(
        pynabled.sparse_bicgstab_solve(nonsymmetric, rhs, tolerance=1e-10, max_iterations=5000),
        expected_nonsymmetric,
        rtol=1e-8,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        nonsymmetric.bicgstab_solve(rhs, tolerance=1e-10, max_iterations=5000),
        expected_nonsymmetric,
        rtol=1e-8,
        atol=1e-8,
    )


def test_sparse_reusable_factorizations_and_direct_lu():
    dense = np.array(
        [
            [4.0, 1.0, 0.0],
            [2.0, 3.0, 1.0],
            [0.0, 1.0, 2.0],
        ],
        dtype=np.float64,
    )
    matrix = pynabled.CsrMatrix.from_components(*_csr_from_dense(dense, index_dtype=np.int32))
    rhs = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    rhs_multi = np.column_stack([rhs, rhs * 2.0])
    expected = np.linalg.solve(dense, rhs)
    expected_multi = np.linalg.solve(dense, rhs_multi)

    jacobi = pynabled.sparse_jacobi_preconditioner(matrix)
    np.testing.assert_allclose(jacobi.inverse_diagonal, np.array([0.25, 1 / 3, 0.5]))
    np.testing.assert_allclose(jacobi.apply(rhs), rhs / np.diag(dense), rtol=1e-12)

    ilu0 = matrix.ilu0_factor()
    ilut = matrix.ilut_factor(config=pynabled.ILUTConfig(drop_tolerance=0.0, max_fill=8))
    iluk = matrix.iluk_factor(config=pynabled.ILUKConfig(level_of_fill=1))

    for factorization in (ilu0, ilut, iluk):
        assert factorization.l.index_dtype == np.dtype(np.int32)
        assert factorization.u.index_dtype == np.dtype(np.int32)
        np.testing.assert_allclose(
            _dense_from_csr(factorization.l) @ _dense_from_csr(factorization.u),
            dense,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(factorization.apply(rhs), expected, rtol=1e-10, atol=1e-10)

    sparse_lu = matrix.lu_factor()
    assert sparse_lu.l.index_dtype == np.dtype(np.int32)
    assert sparse_lu.u.index_dtype == np.dtype(np.int32)
    assert sparse_lu.permutation.dtype == np.dtype(np.int32)
    np.testing.assert_allclose(sparse_lu.solve(rhs), expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        sparse_lu.solve_multiple(rhs_multi),
        expected_multi,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        pynabled.sparse_lu_solve(matrix, rhs),
        expected,
        rtol=1e-12,
        atol=1e-15,
    )


def test_sparse_symmetric_reusable_factorizations_preserve_dtype():
    dense = np.array(
        [
            [4.0, 1.0, 0.0],
            [1.0, 3.0, 1.0],
            [0.0, 1.0, 2.0],
        ],
        dtype=np.float32,
    )
    matrix = pynabled.CsrMatrix.from_components(*_csr_from_dense(dense))
    rhs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    expected = np.linalg.solve(dense.astype(np.float64), rhs.astype(np.float64)).astype(np.float32)

    ic0 = pynabled.sparse_ic0_factor(matrix)
    ildl0 = matrix.ildl0_factor()

    assert ic0.l.dtype == np.dtype(np.float32)
    assert ic0.l_transpose.dtype == np.dtype(np.float32)
    assert ildl0.l.dtype == np.dtype(np.float32)
    assert ildl0.l_transpose.dtype == np.dtype(np.float32)
    assert ildl0.d.dtype == np.dtype(np.float32)

    np.testing.assert_allclose(ic0.apply(rhs), expected, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(ildl0.apply(rhs), expected, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(
        ic0.pcg_solve(rhs, tolerance=1e-5, max_iterations=5000),
        expected,
        rtol=1e-4,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        matrix.pcg_ic0_solve(rhs, tolerance=1e-5, max_iterations=5000),
        expected,
        rtol=1e-4,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        pynabled.sparse_pcg_ic0_solve(matrix, rhs, tolerance=1e-5, max_iterations=5000),
        expected,
        rtol=1e-4,
        atol=1e-4,
    )


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


def test_sparse_rejects_invalid_explicit_dtypes():
    nrows, ncols, indptr, indices, data = _make_csr_diagonal(3)
    with pytest.raises(TypeError, match="dtype must be float32 or float64"):
        pynabled.CsrMatrix((nrows, ncols), indptr, indices, data, dtype=np.int32)
    with pytest.raises(TypeError, match="index_dtype must be int32 or int64"):
        pynabled.CsrMatrix((nrows, ncols), indptr, indices, data, index_dtype=np.uint32)


def test_sparse_rejects_bad_array_ranks_and_non_real_data():
    _, _, indptr, indices, data = _make_csr_diagonal(3)
    with pytest.raises(ValueError, match="indptr must be a 1D array"):
        pynabled.CsrMatrix((3, 3), indptr.reshape(2, 2), indices, data)
    with pytest.raises(ValueError, match="data must be a 1D array"):
        pynabled.CsrMatrix((3, 3), indptr, indices, data.reshape(3, 1))
    with pytest.raises(TypeError, match="data must have dtype float32 or float64"):
        pynabled.CsrMatrix((3, 3), indptr, indices, np.array([1, 2, 3], dtype=np.int32))


def test_sparse_index_dtype_inference_preserves_available_numpy_dtype():
    _, _, indptr_i32, indices_i32, data = _make_csr_diagonal(3, index_dtype=np.int32)
    matrix_from_indptr = pynabled.CsrMatrix((3, 3), indptr_i32, indices_i32.tolist(), data)
    matrix_from_indices = pynabled.CsrMatrix((3, 3), indptr_i32.tolist(), indices_i32, data)
    matrix_from_lists = pynabled.CsrMatrix((3, 3), indptr_i32.tolist(), indices_i32.tolist(), data)

    assert matrix_from_indptr.index_dtype == np.dtype(np.int32)
    assert matrix_from_indices.index_dtype == np.dtype(np.int32)
    assert matrix_from_lists.index_dtype == np.dtype(np.int32)


def test_sparse_rejects_invalid_numpy_index_dtypes_without_explicit_normalization():
    _, _, indptr, indices, data = _make_csr_diagonal(3)
    with pytest.raises(TypeError, match="indptr must have dtype int32 or int64"):
        pynabled.CsrMatrix((3, 3), indptr.astype(np.uint32), indices, data)
    with pytest.raises(TypeError, match="indices must have dtype int32 or int64"):
        pynabled.CsrMatrix((3, 3), indptr, indices.astype(np.uint32), data)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"shape": (0, 3)}, "shape dimensions must be positive"),
        ({"indptr": np.array([0, 1, 2], dtype=np.int64)}, "indptr length must equal nrows + 1"),
        (
            {"indices": np.array([0, 1], dtype=np.int64), "data": np.array([1.0, 2.0, 3.0])},
            "indices and data must have matching lengths",
        ),
        ({"indptr": np.array([1, 2, 3, 3], dtype=np.int64)}, "indptr must start at 0"),
        (
            {"indptr": np.array([0, 1, 2, 4], dtype=np.int64)},
            "indptr terminal offset must equal nnz",
        ),
        (
            {"indptr": np.array([0, 2, 1, 3], dtype=np.int64)},
            "indptr must be non-decreasing",
        ),
        (
            {"indices": np.array([0, 1, 3], dtype=np.int64)},
            "indices must lie within matrix column bounds",
        ),
    ],
)
def test_sparse_structural_validation_errors(kwargs, message):
    base = {
        "shape": (3, 3),
        "indptr": np.array([0, 1, 2, 3], dtype=np.int64),
        "indices": np.array([0, 1, 2], dtype=np.int64),
        "data": np.array([1.0, 2.0, 3.0], dtype=np.float64),
    }
    base.update(kwargs)
    with pytest.raises(ValueError, match=re.escape(message)):
        pynabled.CsrMatrix(base["shape"], base["indptr"], base["indices"], base["data"])


def test_sparse_from_components_and_from_scipy_error_paths():
    matrix = pynabled.CsrMatrix.from_components(*_make_csr_diagonal(3))

    with pytest.raises(TypeError, match="from_components expects"):
        pynabled.CsrMatrix.from_components(1, 2, 3)

    assert pynabled.CsrMatrix.from_scipy(matrix) is matrix
    assert pynabled.CsrMatrix.from_scipy(matrix, dtype=np.float32).dtype == np.dtype(np.float32)

    with pytest.raises(
        TypeError,
        match="expected pynabled.CsrMatrix or a scipy.sparse-compatible object with tocsr",
    ):
        pynabled.CsrMatrix.from_scipy(object())

    incomplete = types.SimpleNamespace(format="csr", shape=(1, 1), indptr=np.array([0, 0]))
    with pytest.raises(
        TypeError, match="expected pynabled.CsrMatrix or a scipy.sparse-compatible CSR object"
    ):
        pynabled.CsrMatrix.from_scipy(incomplete)


def test_sparse_carrier_helpers_and_dunders():
    matrix = pynabled.CsrMatrix.from_components(*_make_csr_diagonal(3))

    copied = matrix.copy()
    components = matrix.to_components()

    assert copied is not matrix
    assert matrix.astype(np.float64) is matrix
    assert matrix.with_index_dtype(np.int64) is matrix
    assert components[0] == matrix.nrows
    assert components[1] == matrix.ncols
    assert matrix.__matmul__(np.zeros((1, 1, 1))) is NotImplemented
    assert "index_dtype=int64" in repr(matrix)


def test_sparse_non_csr_carriers_and_sparse_product():
    dense = np.array(
        [
            [4.0, 1.0, 0.0],
            [2.0, 3.0, 1.0],
            [0.0, 1.0, 2.0],
        ],
        dtype=np.float64,
    )
    csr = pynabled.CsrMatrix.from_components(*_csr_from_dense(dense, index_dtype=np.int32))
    csc = csr.to_csc()
    coo = pynabled.CooMatrix(
        dense.shape,
        np.array([0, 0, 1, 1, 1, 2, 2], dtype=np.int32),
        np.array([0, 1, 0, 1, 2, 1, 2], dtype=np.int32),
        np.array([4.0, 1.0, 2.0, 3.0, 1.0, 1.0, 2.0], dtype=np.float64),
    )
    vector = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    assert isinstance(csc, pynabled.CscMatrix)
    assert csc.index_dtype == np.dtype(np.int32)
    np.testing.assert_allclose(csc.matvec(vector), dense @ vector, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        pynabled.sparse_matvec_csc(csc, vector), dense @ vector, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(_dense_from_csr(csc.to_csr()), dense, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(_dense_from_csr(coo.to_csr()), dense, rtol=1e-12, atol=1e-12)

    product = pynabled.sparse_matmat_sparse(csr, csc)
    assert isinstance(product, pynabled.CsrMatrix)
    np.testing.assert_allclose(
        _dense_from_csr(product),
        dense @ dense,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        _dense_from_csr(csr @ coo),
        dense @ dense,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(coo @ vector, dense @ vector, rtol=1e-12, atol=1e-12)

    canonical_csc = pynabled.CscMatrix.from_scipy(
        _FakeSciPyCanonicalCsc(csc.shape, csc.indptr, csc.indices, csc.data)
    )
    canonical_coo = pynabled.CooMatrix.from_scipy(
        _FakeSciPyCoo(coo.shape, coo.row_indices, coo.col_indices, coo.data)
    )
    assert isinstance(canonical_csc, pynabled.CscMatrix)
    assert isinstance(canonical_coo, pynabled.CooMatrix)
    np.testing.assert_allclose(_dense_from_csr(canonical_csc.to_csr()), dense, rtol=1e-12)
    np.testing.assert_allclose(_dense_from_csr(canonical_coo.to_csr()), dense, rtol=1e-12)


def test_sparse_iterative_reuse_methods():
    nonsymmetric_dense = np.array(
        [
            [4.0, 1.0, 0.0],
            [2.0, 3.0, 1.0],
            [0.0, 1.0, 2.0],
        ],
        dtype=np.float64,
    )
    nonsymmetric = pynabled.CsrMatrix.from_components(*_csr_from_dense(nonsymmetric_dense))
    rhs = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    rhs_multi = np.column_stack([rhs, rhs * 2.0])
    expected = np.linalg.solve(nonsymmetric_dense, rhs)
    expected_multi = np.linalg.solve(nonsymmetric_dense, rhs_multi)

    for factorization in (
        nonsymmetric.ilu0_factor(),
        nonsymmetric.ilut_factor(config=pynabled.ILUTConfig(drop_tolerance=0.0, max_fill=8)),
        nonsymmetric.iluk_factor(config=pynabled.ILUKConfig(level_of_fill=1)),
    ):
        np.testing.assert_allclose(
            factorization.gmres_solve(rhs, tolerance=1e-10, max_iterations=16),
            expected,
            rtol=1e-8,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            factorization.gmres_solve_multiple(rhs_multi, tolerance=1e-10, max_iterations=16),
            expected_multi,
            rtol=1e-8,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            factorization.bicgstab_solve(rhs, tolerance=1e-10, max_iterations=512),
            expected,
            rtol=1e-8,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            factorization.bicgstab_solve_multiple(rhs_multi, tolerance=1e-10, max_iterations=512),
            expected_multi,
            rtol=1e-8,
            atol=1e-8,
        )

    symmetric_dense = np.array(
        [
            [4.0, 1.0, 0.0],
            [1.0, 3.0, 1.0],
            [0.0, 1.0, 2.0],
        ],
        dtype=np.float64,
    )
    symmetric = pynabled.CsrMatrix.from_components(*_csr_from_dense(symmetric_dense))
    expected_symmetric = np.linalg.solve(symmetric_dense, rhs)
    expected_symmetric_multi = np.linalg.solve(symmetric_dense, rhs_multi)
    ildl0 = symmetric.ildl0_factor()
    np.testing.assert_allclose(
        ildl0.gmres_solve(rhs, tolerance=1e-10, max_iterations=32),
        expected_symmetric,
        rtol=1e-8,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        ildl0.gmres_solve_multiple(rhs_multi, tolerance=1e-10, max_iterations=32),
        expected_symmetric_multi,
        rtol=1e-8,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        ildl0.bicgstab_solve(rhs, tolerance=1e-10, max_iterations=512),
        expected_symmetric,
        rtol=1e-8,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        ildl0.bicgstab_solve_multiple(rhs_multi, tolerance=1e-10, max_iterations=512),
        expected_symmetric_multi,
        rtol=1e-8,
        atol=1e-8,
    )


def test_sparse_config_profiles_and_factorization_helpers():
    assert pynabled.ILUTConfig.conservative() == pynabled.ILUTConfig(
        drop_tolerance=1e-6, max_fill=8
    )
    assert pynabled.ILUTConfig.balanced() == pynabled.ILUTConfig(drop_tolerance=1e-8, max_fill=16)
    assert pynabled.ILUTConfig.aggressive() == pynabled.ILUTConfig(
        drop_tolerance=1e-10, max_fill=32
    )
    assert pynabled.ILUTConfig.for_dimension(16) == pynabled.ILUTConfig(
        drop_tolerance=1e-8, max_fill=8
    )
    assert pynabled.ILUKConfig.conservative() == pynabled.ILUKConfig(level_of_fill=0)
    assert pynabled.ILUKConfig.balanced() == pynabled.ILUKConfig(level_of_fill=1)
    assert pynabled.ILUKConfig.aggressive() == pynabled.ILUKConfig(level_of_fill=2)

    dense = np.array(
        [
            [4.0, 1.0, 0.0],
            [1.0, 3.0, 1.0],
            [0.0, 1.0, 2.0],
        ],
        dtype=np.float64,
    )
    matrix = pynabled.CsrMatrix.from_components(*_csr_from_dense(dense, index_dtype=np.int32))

    with pytest.raises(TypeError, match="either config"):
        matrix.ilut_factor(config=pynabled.ILUTConfig.balanced(), drop_tolerance=1e-8)
    with pytest.raises(TypeError, match="either config"):
        matrix.iluk_factor(config=pynabled.ILUKConfig.balanced(), level_of_fill=1)

    ilu0 = matrix.ilu0_factor()
    ilut = matrix.ilut_factor(config=pynabled.ILUTConfig.balanced())
    iluk = matrix.iluk_factor(config=pynabled.ILUKConfig.balanced())
    ic0 = matrix.ic0_factor()
    ildl0 = matrix.ildl0_factor()
    lu = matrix.lu_factor()

    for factor in (ilu0, ilut, iluk):
        assert factor.l.nnz > 0
        assert factor.u.nnz > 0
        assert "dtype=float64" in repr(factor)
    assert iluk.level_of_fill == 1
    assert ic0.l.nnz > 0
    assert ic0.l_transpose.nnz > 0
    assert "dtype=float64" in repr(ic0)
    assert ildl0.l.nnz > 0
    assert ildl0.l_transpose.nnz > 0
    assert ildl0.d.shape == (3,)
    assert "dtype=float64" in repr(ildl0)
    assert lu.l.nnz > 0
    assert lu.u.nnz > 0
    assert lu.permutation.shape == (3,)
    assert "dtype=float64" in repr(lu)


def test_csc_carrier_helpers_and_error_paths(monkeypatch):
    dense = np.array(
        [
            [4.0, 0.0, 1.0],
            [2.0, 3.0, 0.0],
            [0.0, 1.0, 2.0],
        ],
        dtype=np.float64,
    )
    csr = pynabled.CsrMatrix.from_components(*_csr_from_dense(dense))
    csc = csr.to_csc()
    copied = csc.copy()
    assert copied is not csc
    assert csc.astype(np.float64) is csc
    assert csc.with_index_dtype(np.int64) is csc
    assert csc.astype(np.float32).dtype == np.dtype(np.float32)
    assert csc.with_index_dtype(np.int32).index_dtype == np.dtype(np.int32)
    assert csc.to_components()[0:2] == (3, 3)
    np.testing.assert_allclose(csc.matmat_dense(np.eye(3)), dense, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(_dense_from_csr(csc.T), dense.T, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        _dense_from_csr(csc @ csr),
        dense @ dense,
        rtol=1e-12,
        atol=1e-12,
    )
    assert "index_dtype=int64" in repr(csc)

    with pytest.raises(ImportError, match="scipy is required"):
        csc.to_scipy()
    scipy_sparse = types.SimpleNamespace(
        csc_matrix=lambda payload, shape, copy: (payload, shape, copy)
    )
    monkeypatch.setitem(sys.modules, "scipy", types.SimpleNamespace(sparse=scipy_sparse))
    payload, shape, copy = csc.to_scipy()
    assert shape == csc.shape
    assert copy is False
    np.testing.assert_array_equal(payload[0], csc.data)

    with pytest.raises(TypeError, match="from_components expects"):
        pynabled.CscMatrix.from_components(1, 2, 3)
    with pytest.raises(ValueError, match="indptr length must equal ncols \\+ 1"):
        pynabled.CscMatrix((3, 3), np.array([0, 1, 2]), np.array([0, 1]), np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="indices must lie within matrix row bounds"):
        pynabled.CscMatrix(
            (3, 3),
            np.array([0, 1, 2, 3], dtype=np.int64),
            np.array([0, 1, 3], dtype=np.int64),
            np.array([1.0, 2.0, 3.0]),
        )
    with pytest.raises(TypeError, match="expected pynabled.CscMatrix"):
        pynabled.CscMatrix.from_scipy(object())
    incomplete = types.SimpleNamespace(format="csc", shape=(1, 1), indptr=np.array([0, 0]))
    with pytest.raises(TypeError, match="expected pynabled.CscMatrix"):
        pynabled.CscMatrix.from_scipy(incomplete)


def test_coo_carrier_helpers_and_error_paths(monkeypatch):
    coo = pynabled.CooMatrix(
        (3, 3),
        np.array([0, 0, 1, 2], dtype=np.int64),
        np.array([0, 2, 1, 2], dtype=np.int64),
        np.array([4.0, 1.0, 3.0, 2.0], dtype=np.float64),
    )
    copied = coo.copy()
    assert copied is not coo
    assert coo.astype(np.float64) is coo
    assert coo.with_index_dtype(np.int64) is coo
    assert coo.astype(np.float32).dtype == np.dtype(np.float32)
    assert coo.with_index_dtype(np.int32).index_dtype == np.dtype(np.int32)
    assert coo.to_components()[0:2] == (3, 3)
    assert "index_dtype=int64" in repr(coo)

    with pytest.raises(ImportError, match="scipy is required"):
        coo.to_scipy()
    scipy_sparse = types.SimpleNamespace(
        coo_matrix=lambda payload, shape, copy: (payload, shape, copy)
    )
    monkeypatch.setitem(sys.modules, "scipy", types.SimpleNamespace(sparse=scipy_sparse))
    payload, shape, copy = coo.to_scipy()
    assert shape == coo.shape
    assert copy is False
    np.testing.assert_array_equal(payload[0], coo.data)

    with pytest.raises(TypeError, match="from_components expects"):
        pynabled.CooMatrix.from_components(1, 2, 3)
    with pytest.raises(ValueError, match="row_indices and data must have matching lengths"):
        pynabled.CooMatrix(
            (3, 3),
            np.array([0, 1], dtype=np.int64),
            np.array([0, 1, 2], dtype=np.int64),
            np.array([1.0, 2.0, 3.0]),
        )
    with pytest.raises(TypeError, match="row_indices and col_indices must share dtype"):
        pynabled.CooMatrix(
            (3, 3),
            np.array([0, 1, 2], dtype=np.int32),
            np.array([0, 1, 2], dtype=np.int64),
            np.array([1.0, 2.0, 3.0]),
        )
    with pytest.raises(ValueError, match="row_indices must lie within matrix row bounds"):
        pynabled.CooMatrix(
            (3, 3),
            np.array([0, 1, 3], dtype=np.int64),
            np.array([0, 1, 2], dtype=np.int64),
            np.array([1.0, 2.0, 3.0]),
        )
    with pytest.raises(TypeError, match="expected pynabled.CooMatrix"):
        pynabled.CooMatrix.from_scipy(object())
    incomplete = types.SimpleNamespace(format="coo", shape=(1, 1), row=np.array([0]))
    with pytest.raises(TypeError, match="expected pynabled.CooMatrix"):
        pynabled.CooMatrix.from_scipy(incomplete)


def test_sparse_conversion_helpers_and_non_csr_coercion():
    dense = np.array(
        [
            [4.0, 1.0, 0.0],
            [2.0, 3.0, 1.0],
            [0.0, 1.0, 2.0],
        ],
        dtype=np.float32,
    )
    csr = pynabled.CsrMatrix.from_components(*_csr_from_dense(dense, index_dtype=np.int32))
    csc = pynabled.sparse_csr_to_csc(csr)
    coo = pynabled.CooMatrix(
        dense.shape,
        np.array([0, 0, 1, 1, 1, 2, 2], dtype=np.int32),
        np.array([0, 1, 0, 1, 2, 1, 2], dtype=np.int32),
        np.array([4.0, 1.0, 2.0, 3.0, 1.0, 1.0, 2.0], dtype=np.float32),
    )

    np.testing.assert_allclose(_dense_from_csr(pynabled.sparse_csc_to_csr(csc)), dense, rtol=1e-5)
    np.testing.assert_allclose(_dense_from_csr(pynabled.sparse_coo_to_csr(coo)), dense, rtol=1e-5)
    np.testing.assert_allclose(
        pynabled.sparse_matmat_dense(csc, np.eye(3, dtype=np.float32)),
        dense,
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(_dense_from_csr(pynabled.sparse_transpose(coo)), dense.T, rtol=1e-5)
    np.testing.assert_allclose(
        _dense_from_csr(pynabled.sparse_matmat_sparse(csc, coo)),
        dense @ dense,
        rtol=1e-5,
        atol=1e-5,
    )


def test_sparse_to_scipy_import_error_and_success(monkeypatch):
    matrix = pynabled.CsrMatrix.from_components(*_make_csr_diagonal(3))

    with pytest.raises(ImportError, match="scipy is required"):
        matrix.to_scipy()

    scipy_sparse = types.SimpleNamespace(
        csr_matrix=lambda payload, shape, copy: (payload, shape, copy)
    )
    scipy_module = types.SimpleNamespace(sparse=scipy_sparse)
    monkeypatch.setitem(sys.modules, "scipy", scipy_module)

    payload, shape, copy = matrix.to_scipy()
    assert shape == matrix.shape
    assert copy is False
    np.testing.assert_array_equal(payload[0], matrix.data)
    np.testing.assert_array_equal(payload[1], matrix.indices)
    np.testing.assert_array_equal(payload[2], matrix.indptr)
